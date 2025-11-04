import os
import time
import json
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union

from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

from src.utils.config import (
    ensure_dir,
    load_config_with_defaults,
    save_json,
    create_timestamped_parent_dir,
)
from src.utils.eval import VLLMModelEvaluator
from vllm import SamplingParams


class RetryConfig:
    def __init__(self, max_retries: int = 3, backoff_seconds: float = 2.0):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds


class VLLMChatClient:
    """vLLM-backed chat interface that reuses the evaluator's preparation flow.

    Supports HF base models with optional PEFT checkpoints or W&B artifacts (merged on the fly).
    """

    def __init__(
        self,
        base_model_id: str,
        checkpoint_path: Optional[str] = None,
        model_artifact_name: Optional[str] = None,
        wandb_project_name: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        retry: Optional[RetryConfig] = None,
    ) -> None:
        if not checkpoint_path and not model_artifact_name:
            raise ValueError(
                "Provide either checkpoint_path or model_artifact_name for vLLM client"
            )

        # Prepare vLLM model/tokenizer via existing evaluator util
        self._prep = VLLMModelEvaluator(
            model_artifact_name=model_artifact_name,
            checkpoint_path=checkpoint_path,
            base_model_id=base_model_id,
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            wandb_project_name=wandb_project_name,
        )

        # Build sampling params
        self.sampling_params = SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        self.retry = retry or RetryConfig()

    @property
    def tokenizer(self):
        return self._prep.tokenizer

    @property
    def llm(self):
        return self._prep.llm

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                outputs = self.llm.generate([input_text], self.sampling_params)
                return (outputs[0].outputs[0].text or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.backoff_seconds * (2**attempt))
                else:
                    break
        raise last_err or RuntimeError("vLLM generation failed for unknown reasons")


class AnthropicChatClient:
    """Anthropic Messages API backed chat interface.

    Reads `ANTHROPIC_API_KEY` from environment (e.g., provided via .env).
    Matches the `.chat(system_prompt, user_prompt)` signature used elsewhere.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        retry: Optional[RetryConfig] = None,
    ) -> None:
        try:
            # Import lazily so users without anthropic installed can still use vLLM path
            from anthropic import Anthropic  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "The 'anthropic' package is required for AnthropicChatClient. Install with `pip install anthropic`."
            ) from exc

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing {api_key_env} in environment. Ensure it is set (e.g., via .env)."
            )

        self._Anthropic = Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = top_p
        self.stop = stop
        self.retry = retry or RetryConfig()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                # See: https://docs.anthropic.com/claude/reference/messages_post
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop_sequences=self.stop,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                # Anthropic returns a list of content blocks; join any text blocks
                parts = []
                for block in getattr(resp, "content", []) or []:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
                return "".join(parts).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.backoff_seconds * (2**attempt))
                else:
                    break
        raise last_err or RuntimeError(
            "Anthropic generation failed for unknown reasons"
        )


def format_prompt(template: str, example: Dict[str, Any]) -> str:
    """Render a simple string template using example keys via {key} placeholders."""
    try:
        return template.format(**example)
    except KeyError:
        # Fallback: provide the raw example as JSON when keys mismatch
        return f"Example JSON for reference:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```\n\nTemplate:\n{template}"


class QualityJudge:
    """LLM-based judge that classifies examples as high/low quality from a parametric prompt.

    Expected judge output: a short verdict that includes either the token 'HIGH' or 'LOW'.
    """

    def __init__(
        self,
        llm: VLLMChatClient,
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

    def is_high_quality(self, example: Dict[str, Any]) -> Tuple[bool, str]:
        prompt = format_prompt(self.user_prompt_template, example)
        verdict = self.llm.chat(self.system_prompt, prompt)
        verdict_upper = verdict.upper()
        # Prefer explicit LOW if both appear
        if (
            self.low_token.upper() in verdict_upper
            and self.high_token.upper() not in verdict_upper
        ):
            return False, verdict
        if (
            self.high_token.upper() in verdict_upper
            and self.low_token.upper() not in verdict_upper
        ):
            return True, verdict
        # Tie-breaker: treat ambiguous as LOW to be conservative
        return False, verdict


class DatasetCurator:
    """Filter a dataset using an LLM judge and persist high-quality examples."""

    def __init__(
        self,
        judge: QualityJudge,
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
            is_high, verdict = self.judge.is_high_quality(ex)
            if is_high:
                kept.append(self._select_fields(ex))
            else:
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
        llm: VLLMChatClient,
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

    def generate(self, seed_examples: List[Dict[str, Any]], count: int) -> List[str]:
        outputs: List[str] = []
        few_shot_prefix = self.build_few_shot_prompt(seed_examples)
        for _ in tqdm(range(count), desc="Generating samples"):
            user_prompt = self.user_prompt_template.format(
                few_shot_examples=few_shot_prefix
            )
            completion = self.llm.chat(self.system_prompt, user_prompt)
            outputs.append(completion)
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


def run_curation_from_config(config_path: str) -> str:
    """End-to-end entry point from a YAML config for curation and optional generation.

    Expects sections: `curation`, `judge`, and optional `generate` configured for vLLM.
    """
    cfg = load_config_with_defaults(config_path)

    cur_cfg = cfg.get("curation", {})
    judge_cfg = cfg.get("judge", {})
    gen_cfg = cfg.get("generate", {})

    base_results_dir = cur_cfg.get("base_results_dir", "results/curation")
    name = cur_cfg.get("name", "curation_run")
    parent_dir = create_timestamped_parent_dir(base_results_dir, prefix=name)

    dataset_path = cur_cfg["dataset_path"]
    max_samples = cur_cfg.get("max_samples")
    keep_fields = cur_cfg.get("keep_fields")

    # Build judge client (vLLM or Anthropic)
    judge_provider = str(judge_cfg.get("provider", "vllm")).lower()
    if judge_provider == "anthropic":
        judge_llm = AnthropicChatClient(
            model=judge_cfg["model"],
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=int(judge_cfg.get("max_tokens", 128)),
            top_p=judge_cfg.get("top_p"),
            stop=judge_cfg.get("stop"),
        )
    else:
        judge_llm = VLLMChatClient(
            base_model_id=judge_cfg["base_model_id"],
            checkpoint_path=judge_cfg.get("checkpoint_path"),
            model_artifact_name=judge_cfg.get("model_artifact_name"),
            wandb_project_name=judge_cfg.get("wandb_project_name", ""),
            tensor_parallel_size=int(judge_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(judge_cfg.get("gpu_memory_utilization", 0.9)),
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=int(judge_cfg.get("max_tokens", 128)),
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
    kept_path = os.path.join(parent_dir, "kept.jsonl")
    rej_path = os.path.join(parent_dir, "rejected.jsonl")
    curator.save_jsonl(kept, kept_path)
    curator.save_jsonl(rejected, rej_path)

    save_json(
        {"kept": len(kept), "rejected": len(rejected)},
        os.path.join(parent_dir, "summary.json"),
    )

    # Optionally generate new samples
    if gen_cfg.get("enabled", False) and len(kept) > 0:
        gen_provider = str(gen_cfg.get("provider", "vllm")).lower()
        if gen_provider == "anthropic":
            gen_llm = AnthropicChatClient(
                model=gen_cfg["model"],
                temperature=float(gen_cfg.get("temperature", 0.7)),
                max_tokens=int(gen_cfg.get("max_tokens", 512)),
                top_p=gen_cfg.get("top_p"),
                stop=gen_cfg.get("stop"),
            )
        else:
            gen_llm = VLLMChatClient(
                base_model_id=gen_cfg["base_model_id"],
                checkpoint_path=gen_cfg.get("checkpoint_path"),
                model_artifact_name=gen_cfg.get("model_artifact_name"),
                wandb_project_name=gen_cfg.get("wandb_project_name", ""),
                tensor_parallel_size=int(gen_cfg.get("tensor_parallel_size", 1)),
                gpu_memory_utilization=float(
                    gen_cfg.get("gpu_memory_utilization", 0.9)
                ),
                temperature=float(gen_cfg.get("temperature", 0.7)),
                max_tokens=int(gen_cfg.get("max_tokens", 512)),
            )
        generator = SampleGenerator(
            llm=gen_llm,
            system_prompt=gen_cfg["system_prompt"],
            user_prompt_template=gen_cfg["user_prompt_template"],
            num_shots=int(gen_cfg.get("num_shots", 3)),
        )
        num_new = int(gen_cfg.get("num_new_samples", 0))
        generated = generator.generate(kept, count=num_new)
        with open(
            os.path.join(parent_dir, "generated.txt"), "w", encoding="utf-8"
        ) as f:
            for item in generated:
                f.write(item.strip() + "\n\n")

    return parent_dir


if __name__ == "__main__":
    import argparse
    import dotenv

    parser = argparse.ArgumentParser(
        description="Curate datasets with an LLM judge and optionally generate new samples"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )

    dotenv.load_dotenv()

    args = parser.parse_args()
    out_dir = run_curation_from_config(args.config)
    print(f"✓ Curation complete. Outputs saved under: {out_dir}")
