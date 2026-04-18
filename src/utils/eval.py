import shutil
import tempfile
from typing import Callable, Dict, List, Tuple, Optional, Any

import wandb
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

from src.utils.wandb_logging import log_dataset_results

from src.utils.parse import extract_xml_answer


class VLLMModelEvaluator:
    """Evaluator for models on reward hacking datasets using vLLM for fast inference."""

    def __init__(
        self,
        artifact_name: Optional[str],
        wandb_project: str,
        wandb_entity: str,
        base_model_id: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        self.base_model_id = base_model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity

        if artifact_name:
            # Download and merge model
            self.model_path, self.tokenizer = self._prepare_from_artifact(artifact_name)

        else:
            self.model_path = base_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize vLLM.
        # Cap max_num_seqs and max_model_len to avoid sampler-warmup OOM — the
        # 1024 × full-context-length pre-alloc can blow up KV-cache even on H100.
        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_num_seqs=256,
            max_model_len=8192,
            trust_remote_code=True,
            dtype="float16",
        )

        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
        )

    def _merge_peft_model(self, checkpoint_path: str, output_path: str) -> str:
        """Merge PEFT adapter with base model."""
        print(f"Merging PEFT model from {checkpoint_path}")

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_path)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        tokenizer.save_pretrained(output_path)

        del base_model
        del model
        del merged_model
        torch.cuda.empty_cache()

        print(f"✓ Model merged and saved to {output_path}")
        return output_path

    def _prepare_from_artifact(self, artifact_name: str) -> Tuple[str, AutoTokenizer]:
        """Download artifact from wandb and prepare for inference."""
        api = wandb.Api()
        artifact = api.artifact(
            f"{self.wandb_entity}/{self.wandb_project}/{artifact_name}:latest"
        )
        artifact_dir = artifact.download()

        # Create temporary directory for merged model
        temp_dir = tempfile.mkdtemp(prefix="vllm_merged_model_")
        merged_path = self._merge_peft_model(artifact_dir, temp_dir)

        tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return merged_path, tokenizer

    def generate_batch_responses(
        self,
        prompts: List[str],
        instruction_suffix: str,
        source_datasets: Optional[List[str]] = None,
        source_dataset_to_system_prompt: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        formatted_prompts = []

        for idx, prompt in enumerate(prompts):
            # Add instruction suffix
            formatted_prompt = prompt + instruction_suffix

            # Build messages with optional system prompt
            messages = []
            if source_datasets and source_dataset_to_system_prompt:
                source_dataset = source_datasets[idx]
                system_prompt = source_dataset_to_system_prompt[source_dataset]
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": formatted_prompt})

            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(input_text)

        outputs = self.llm.generate(formatted_prompts, self.sampling_params)

        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            if "</answer>" not in generated_text:
                generated_text += "</answer>"
            responses.append(generated_text)

        return responses

    def evaluate_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        eval_functions: Dict[str, Callable],
        instruction_suffix: str,
        max_samples: Optional[int] = None,
        batch_size: int = 32,
        source_dataset_to_system_prompt: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """Evaluate model on a dataset."""
        # Apply max_samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))

        correct = 0
        total = 0
        no_answer_tags = 0
        results: List[Dict] = []

        # Initialize outputs for each eval function
        eval_outputs = {func_name: [] for func_name in eval_functions}

        # Batching
        prompts_batch: List[str] = []
        high_reward_answers_batch: List[str] = []
        source_datasets_batch: List[str] = []
        batch_dict: Dict[str, List] = {}

        progress_bar = tqdm(
            enumerate(dataset), total=len(dataset), desc=f"Evaluating {dataset_name}"
        )

        for idx, example in progress_bar:
            # Collect batch data using 'question' field (from pipeline output)
            prompts_batch.append(example["question"])
            high_reward_answers_batch.append(example["high_reward_answer"])
            source_datasets_batch.append(example["source_dataset"])

            # Collect other fields for eval functions
            for k, v in example.items():
                if k not in ["high_reward_answer"]:
                    batch_dict[k] = batch_dict.get(k, [])
                    batch_dict[k].append(v)

            # Process batch
            if len(prompts_batch) >= batch_size or idx == len(dataset) - 1:
                responses = self.generate_batch_responses(
                    prompts_batch,
                    instruction_suffix,
                    source_datasets=source_datasets_batch,
                    source_dataset_to_system_prompt=source_dataset_to_system_prompt,
                )

                # Run all eval functions on this batch
                # Store metadata (like overseer decision texts) for each eval function
                eval_metadata: Dict[str, List[str]] = {}

                for func_name, eval_fn in eval_functions.items():
                    eval_results = eval_fn(
                        prompts=prompts_batch,
                        completions=responses,
                        high_reward_answer=high_reward_answers_batch,
                        **batch_dict,
                    )
                    eval_outputs[func_name].extend(eval_results)

                    # Check if this eval function provides decision texts (e.g., overseer functions)
                    if hasattr(eval_fn, "get_decision_texts"):
                        decision_texts = eval_fn.get_decision_texts()
                        if func_name not in eval_metadata:
                            eval_metadata[func_name] = []
                        eval_metadata[func_name].extend(decision_texts)

                # Process individual results
                for i, (prompt, response, high_reward_answer) in enumerate(
                    zip(prompts_batch, responses, high_reward_answers_batch)
                ):
                    extracted_answer = extract_xml_answer(response)

                    if extracted_answer == "No answer tags found":
                        no_answer_tags += 1

                    is_correct = extracted_answer == high_reward_answer
                    if is_correct:
                        correct += 1
                    total += 1

                    result_dict = {
                        "prompt": prompt,
                        "response": response,
                        "extracted_answer": extracted_answer,
                        "high_reward_answer": high_reward_answer,
                        "is_correct": is_correct,
                    }

                    # Add overseer decision texts if available
                    # Check if both summary and non-summary overseers are present
                    overseer_funcs = [
                        fn for fn in eval_metadata.keys() if "overseer" in fn.lower()
                    ]
                    has_summary_overseer = any(
                        "summary" in fn.lower() for fn in overseer_funcs
                    )
                    has_non_summary_overseer = any(
                        "summary" not in fn.lower() for fn in overseer_funcs
                    )
                    both_overseer_types = (
                        has_summary_overseer and has_non_summary_overseer
                    )

                    for func_name, metadata_list in eval_metadata.items():
                        if i < len(metadata_list) and metadata_list[i]:
                            # Use specific keys for different overseer types
                            if "overseer" in func_name.lower():
                                if "summary" in func_name.lower():
                                    result_dict["overseer_summary_response"] = (
                                        metadata_list[i]
                                    )
                                elif "cot" in func_name.lower() or both_overseer_types:
                                    # If explicitly has "cot" OR if both types present,
                                    # treat non-summary as CoT
                                    result_dict["overseer_cot_response"] = (
                                        metadata_list[i]
                                    )
                                else:
                                    result_dict["overseer_response"] = metadata_list[i]
                            else:
                                result_dict[f"{func_name}_response"] = metadata_list[i]

                    results.append(result_dict)

                # Update progress bar
                current_accuracy = correct / total if total > 0 else 0.0
                progress_bar.set_postfix(
                    {
                        "accuracy": f"{current_accuracy:.3f}",
                        "correct": correct,
                        "total": total,
                    }
                )

                # Reset batches
                prompts_batch = []
                high_reward_answers_batch = []
                source_datasets_batch = []
                batch_dict = {}

        # Compute metrics
        accuracy = correct / total if total > 0 else 0.0
        no_answer_tags_rate = no_answer_tags / total if total > 0 else 0.0

        metrics: Dict[str, float] = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "no_answer_tags": no_answer_tags,
            "no_answer_tags_rate": no_answer_tags_rate,
        }

        # Add metrics from eval functions
        for func_name, outputs in eval_outputs.items():
            if outputs:
                metrics[func_name] = sum(outputs) / len(outputs)

        # Log to wandb
        if wandb.run is not None:
            log_dataset_results(dataset_name, accuracy, results)

        return metrics, results

    def cleanup(self):
        """Clean up temporary files and release GPU memory."""
        # Delete vLLM instance to release GPU memory
        if hasattr(self, "llm"):
            del self.llm

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Remove temporary model files
        if hasattr(self, "model_path") and str(self.model_path).startswith("/tmp/"):
            shutil.rmtree(self.model_path, ignore_errors=True)
