from loguru import logger
import os
import time
import json
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union, Type, Literal

from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

from src.utils.config import (
    ensure_dir,
    load_config_with_defaults,
)
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()


class RetryConfig:
    def __init__(self, max_retries: int = 3, backoff_seconds: float = 2.0):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds


class FireworksChatClient:
    """Fireworks chat interface using the OpenAI-compatible SDK."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "FIREWORKS_API_KEY",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        retry: Optional[RetryConfig] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing {api_key_env} in environment. Ensure it is set (e.g., via .env)."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = top_p
        self.stop = stop
        self.retry = retry or RetryConfig()
        self.response_model = response_model

    def _json_schema_from_model(self, model: Type[BaseModel]) -> Dict[str, Any]:
        return model.model_json_schema()  # type: ignore[no-any-return]

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                model_for_response = response_model or self.response_model
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.stop is not None:
                    kwargs["stop"] = self.stop
                if model_for_response is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": model_for_response.__name__,
                            "schema": self._json_schema_from_model(model_for_response),
                        },
                    }
                response = self.client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.backoff_seconds * (2**attempt))
                else:
                    break
        raise last_err or RuntimeError(
            "Fireworks generation failed for unknown reasons"
        )


def format_prompt(template: str, example: Dict[str, Any]) -> str:
    """Render a simple string template using example keys via {key} placeholders."""
    try:
        return template.format(**example)
    except KeyError:
        # Fallback: provide the raw example as JSON when keys mismatch
        return f"Example JSON for reference:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```\n\nTemplate:\n{template}"


class JudgeVerdictModel(BaseModel):
    reason: str
    verdict: Literal["HIGH", "LOW"]


class DatasetSampleModel(BaseModel):
    """Pydantic model for dataset sample format."""

    prompt_list: List[str]
    high_reward_answer: str
    other_answers: List[str]
    correct_answer: str


class QualityJudge:
    """LLM-based judge that classifies examples as high/low quality from a parametric prompt.

    Supports two LLM output formats:
    1. JSON: {"verdict":"HIGH","reason":"..."}
    2. Plain text containing the tokens HIGH or LOW
    """

    def __init__(
        self,
        llm: FireworksChatClient,
        system_prompt: str,
        user_prompt_template: str,
        high_token: str = "HIGH",
        low_token: str = "LOW",
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.high_token = high_token
        self.low_token = low_token

    def is_high_quality(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """Returns (is_high_quality, verdict_text, reason)."""
        if example["correct_answer"] == example["high_reward_answer"]:
            # Save money!
            return (
                False,
                "LOW",
                "AUTO: The correct answer is the same as the high-reward answer.",
            )
        prompt = format_prompt(self.user_prompt_template, example)
        verdict_text = self.llm.chat(self.system_prompt, prompt)
        try:
            parsed = json.loads(verdict_text)
            if isinstance(parsed, dict) and "verdict" in parsed:
                verdict_label = str(parsed.get("verdict", "")).upper()
                reason = parsed.get("reason", "")
                if verdict_label in [self.high_token.upper(), self.low_token.upper()]:
                    return (
                        verdict_label == self.high_token.upper(),
                        verdict_label,
                        reason,
                    )

                else:
                    raise ValueError("Invalid verdict label in JSON.")
            else:
                raise ValueError("JSON not in expected dict format.")
        except (json.JSONDecodeError, ValueError, TypeError):
            raise ValueError("Invalid verdict JSON.")


class DatasetCurator:
    """Filter a dataset using an LLM judge and persist high-quality examples."""

    def __init__(
        self,
        judge: Optional[QualityJudge] = None,
        keep_fields: Optional[List[str]] = None,
    ) -> None:
        self.judge = judge
        self.keep_fields = keep_fields

    def _select_fields(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if not self.keep_fields:
            return example
        return {k: example.get(k) for k in self.keep_fields}

    def filter_dataset(
        self,
        dataset: Union[DatasetDict, Dataset],
        max_samples: Optional[int] = None,
        progress: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (kept_examples, rejected_examples_with_meta)."""
        if self.judge is None:
            raise ValueError("judge must be provided to filter_dataset")

        def iter_examples(ds: Union[DatasetDict, Dataset]) -> Iterable[Dict[str, Any]]:
            if isinstance(ds, DatasetDict):
                # Prefer 'train' split if present
                if "train" in ds:
                    yield from ds["train"]
                else:
                    for split in ds.keys():
                        yield from ds[split]
            else:
                yield from ds

        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        iterator: Iterable[Dict[str, Any]] = iter_examples(dataset)
        iterator = tqdm(iterator, disable=not progress, desc="Judging examples")

        for idx, ex in enumerate(iterator):
            if max_samples is not None and idx >= max_samples:
                break
            is_high, verdict, reason = self.judge.is_high_quality(ex)
            if is_high:
                logger.info(f"LLM accepted sample {idx}: {reason}")
                kept.append(self._select_fields(ex))
            else:
                logger.warning(f"LLM rejected sample {idx}: {reason}")
                rejected.append({"example": ex, "verdict": verdict})

        return kept, rejected

    def save_jsonl(self, examples: List[Dict[str, Any]], path: str) -> None:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")


class SampleGenerator:
    """Generate new samples from a set of high-quality examples using an LLM."""

    def __init__(
        self,
        llm: FireworksChatClient,
        system_prompt: str,
        user_prompt_template: str,
        num_shots: int = 3,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.num_shots = int(num_shots)

    def build_few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        shots = examples[-self.num_shots :] if self.num_shots > 0 else []
        shot_texts = []
        for ex in shots:
            shot_texts.append(json.dumps(ex, ensure_ascii=False))
        prefix = "\n\n".join(shot_texts)
        return prefix

    def generate(
        self, seed_examples: List[Dict[str, Any]], count: int
    ) -> List[Dict[str, Any]]:
        """Generate new samples using JSON schema validation.

        Returns a list of dictionaries matching the dataset format.
        When count > 1, makes multiple API calls to generate each sample.
        """
        outputs: List[Dict[str, Any]] = []
        few_shot_prefix = self.build_few_shot_prompt(seed_examples)

        # Generate samples one at a time to ensure proper JSON schema validation
        for i in range(count):
            user_prompt = self.user_prompt_template.format(
                few_shot_examples=few_shot_prefix,
                num_new_samples=1,  # Always request 1 sample per call
            )
            completion = self.llm.chat(
                self.system_prompt,
                user_prompt,
                response_model=DatasetSampleModel,
            )
            try:
                # Parse the JSON response (should be a single object matching DatasetSampleModel)
                parsed = json.loads(completion)
                if isinstance(parsed, dict):
                    # Validate the structure matches our model
                    sample = DatasetSampleModel(**parsed)
                    outputs.append(sample.model_dump())
                elif isinstance(parsed, list) and len(parsed) > 0:
                    # Handle edge case where LLM returns array with one element
                    sample = DatasetSampleModel(**parsed[0])
                    outputs.append(sample.model_dump())
                else:
                    raise ValueError(f"Unexpected response format: {type(parsed)}")
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                logger.error(f"Failed to parse generated sample {i+1}/{count}: {e}")
                logger.error(f"Raw completion: {completion}")
                raise ValueError(f"Invalid generated sample format: {e}") from e

        return outputs


def load_any_dataset(path: str) -> Union[DatasetDict, Dataset]:
    """Load from json/jsonl or a Hugging Face dataset path."""
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jsonl", ".json"}:
            return load_dataset("json", data_files=path)
        raise ValueError(f"Unsupported dataset file extension: {ext}")
    # Assume HF dataset repo-style path
    return load_dataset(path)


def run_augmentation_from_config(
    config_path: str,
    input_dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Run data augmentation on curated examples from a YAML config.

    Args:
        config_path: Path to YAML config file with `generate` and optional `curation` sections
        input_dataset_path: Optional path to curated dataset file. If not provided, will use
            curation config to construct path to `{dataset_name}_kept.jsonl`
        output_path: Optional path for output. If not provided, will use curation config to
            construct path to `{dataset_name}_generated.jsonl`

    Expects `generate` section configured for Fireworks.
    """
    cfg = load_config_with_defaults(config_path)

    gen_cfg = cfg.get("generate", {})
    if not gen_cfg.get("enabled", False):
        raise ValueError(
            "Generation is not enabled in config. Set generate.enabled=true"
        )

    cur_cfg = cfg.get("curation", {})

    # Determine input path
    if input_dataset_path is None:
        if not cur_cfg:
            raise ValueError(
                "Either input_dataset_path must be provided or curation section must be in config"
            )
        input_dataset_path = os.path.join(
            cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_kept.jsonl'
        )

    # Determine output path
    if output_path is None:
        if not cur_cfg:
            raise ValueError(
                "Either output_path must be provided or curation section must be in config"
            )
        output_path = os.path.join(
            cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_generated.jsonl'
        )

    # Load curated examples
    curated_ds = load_any_dataset(input_dataset_path)
    curated_examples = []
    if isinstance(curated_ds, DatasetDict):
        if "train" in curated_ds:
            curated_examples = list(curated_ds["train"])
        else:
            for split in curated_ds.keys():
                curated_examples.extend(list(curated_ds[split]))
    else:
        curated_examples = list(curated_ds)

    if len(curated_examples) == 0:
        raise ValueError(f"No examples found in {input_dataset_path}")

    # Build generator
    gen_llm = FireworksChatClient(
        model=gen_cfg["model"],
        api_key_env=gen_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        max_tokens=int(gen_cfg.get("max_tokens", 512)),
        top_p=gen_cfg.get("top_p"),
        stop=gen_cfg.get("stop"),
        response_model=DatasetSampleModel,
    )
    generator = SampleGenerator(
        llm=gen_llm,
        system_prompt=gen_cfg["system_prompt"],
        user_prompt_template=gen_cfg["user_prompt_template"],
        num_shots=int(gen_cfg.get("num_shots", 3)),
    )

    # Generate new samples
    num_new = int(gen_cfg.get("num_new_samples", 0))
    if num_new <= 0:
        raise ValueError("num_new_samples must be > 0")

    logger.info(
        f"Generating {num_new} new samples from {len(curated_examples)} curated examples"
    )
    generated = generator.generate(curated_examples, count=num_new)

    # Save results
    curator = DatasetCurator()  # Just for save_jsonl method
    curator.save_jsonl(generated, output_path)
    logger.info(f"Saved {len(generated)} generated samples to {output_path}")

    return output_path


def run_curation_from_config(config_path: str) -> str:
    """End-to-end entry point from a YAML config for curation.

    Expects sections: `curation` and `judge` configured for Fireworks.
    """
    cfg = load_config_with_defaults(config_path)

    cur_cfg = cfg.get("curation", {})
    judge_cfg = cfg.get("judge", {})

    dataset_path = os.path.join(
        cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}.jsonl'
    )
    max_samples = cur_cfg.get("max_samples")
    keep_fields = cur_cfg.get("keep_fields")

    # Build judge client (Fireworks only)
    judge_llm = FireworksChatClient(
        model=judge_cfg["model"],
        api_key_env=judge_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
        temperature=float(judge_cfg.get("temperature", 0.0)),
        max_tokens=int(judge_cfg.get("max_tokens", 128)),
        top_p=judge_cfg.get("top_p"),
        stop=judge_cfg.get("stop"),
        response_model=JudgeVerdictModel,
    )
    judge = QualityJudge(
        llm=judge_llm,
        system_prompt=judge_cfg["system_prompt"],
        user_prompt_template=judge_cfg["user_prompt_template"],
    )

    # Load and filter
    ds = load_any_dataset(dataset_path)
    curator = DatasetCurator(judge, keep_fields=keep_fields)
    kept, rejected = curator.filter_dataset(ds, max_samples=max_samples, progress=True)

    # Save results
    kept_path = os.path.join(
        cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_kept.jsonl'
    )
    rej_path = os.path.join(
        cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_rejected.jsonl'
    )
    curator.save_jsonl(kept, kept_path)
    curator.save_jsonl(rejected, rej_path)

    return kept_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Curate datasets with an LLM judge or augment already curated examples"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["curate", "augment", "both"],
        default="curate",
        help="Mode: 'curate' to filter dataset, 'augment' to generate new samples from curated data",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=None,
        help="Path to curated dataset file (for augment mode). If not provided, uses {dataset_name}_kept.jsonl from config",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output file (for augment mode). If not provided, uses {dataset_name}_generated.jsonl from config",
    )

    args = parser.parse_args()

    if args.mode == "curate":
        run_curation_from_config(args.config)
        print("✓ Curation complete.")
    elif args.mode == "augment":
        run_augmentation_from_config(
            args.config,
            input_dataset_path=args.input_dataset,
            output_path=args.output_path,
        )
        print("✓ Augmentation complete.")
    elif args.mode == "both":
        run_curation_from_config(args.config)
        print("✓ Curation complete.")
        run_augmentation_from_config(
            args.config,
            input_dataset_path=args.input_dataset,
            output_path=args.output_path,
        )
        print("✓ Augmentation complete.")
