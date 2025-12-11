"""
Training script with HuggingFace dataset loading and W&B logging.

Usage:
    # Standard usage with config path
    python -m src.train --config configs/experiment/my_experiment.yaml
    
    # For sweeps, use train_sweep.py instead:
    python -m src.train_sweep --config configs/experiment/my_experiment.yaml
"""

import os
import torch
import argparse
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Union

import dotenv
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer, apply_chat_template

from src.utils.rewards import REWARD_FUNCS
from src.utils.config import load_config_with_defaults
from src.utils.parse import (
    count_name_mentions_in_cot,
    count_name_mentions_in_summary,
    count_cot_words,
    count_summary_words,
    count_custom_terms_in_cot,
    count_custom_terms_in_summary,
)
from src.utils.wandb_logging import log_checkpoint_artifact
from src.utils.callbacks import CheckpointCallback, TrackingCallback


# Global tracking data
_tracking: Dict[str, List] = {
    "cot_user": [],
    "cot_name": [],
    "summary_user": [],
    "summary_name": [],
    "cot_words": [],
    "summary_words": [],
}


def tracking_wrapper(original_func):
    """Collect tracking data during reward computation."""

    def wrapper(prompts, completions, **kwargs):
        _tracking["cot_user"].extend(
            count_custom_terms_in_cot(
                prompts=prompts,
                completions=completions,
                high_reward_answer=None,
                terms=["user"],
            )
        )
        _tracking["cot_name"].extend(
            count_name_mentions_in_cot(
                prompts=prompts, completions=completions, high_reward_answer=None
            )
        )
        _tracking["summary_user"].extend(
            count_custom_terms_in_summary(
                prompts=prompts,
                completions=completions,
                high_reward_answer=None,
                terms=["user"],
            )
        )
        _tracking["summary_name"].extend(
            count_name_mentions_in_summary(
                prompts=prompts, completions=completions, high_reward_answer=None
            )
        )
        _tracking["cot_words"].extend(
            count_cot_words(
                prompts=prompts, completions=completions, high_reward_answer=None
            )
        )
        _tracking["summary_words"].extend(
            count_summary_words(
                prompts=prompts, completions=completions, high_reward_answer=None
            )
        )
        return original_func(prompts, completions, **kwargs)

    wrapper.__name__ = original_func.__name__
    return wrapper


def get_reward_functions(rewards_config: Dict[str, Dict[str, Any]]) -> list:
    """Create reward function instances from config."""
    reward_funcs = []

    for func_name, func_config in rewards_config.items():
        if func_name not in REWARD_FUNCS:
            raise ValueError(
                f"Unknown reward function: {func_name}. "
                f"Available functions: {list(REWARD_FUNCS.keys())}"
            )

        factory = REWARD_FUNCS[func_name]
        reward_func = factory(func_config)

        # Wrap first function for tracking
        if len(reward_funcs) == 0:
            reward_func = tracking_wrapper(reward_func)

        reward_funcs.append(reward_func)

    return reward_funcs


def derive_wandb_group(hf_dataset: str) -> str:
    """
    Derive wandb group name from HF dataset path.
    
    Example: "account/obf_gen_experiment_v1_seed_42" -> "experiment_v1_seed_42"
    """
    # Strip account prefix if present
    if "/" in hf_dataset:
        dataset_name = hf_dataset.split("/", 1)[1]
    else:
        dataset_name = hf_dataset
    
    # Strip "obf_gen_" prefix if present
    if dataset_name.startswith("obf_gen_"):
        dataset_name = dataset_name[len("obf_gen_"):]
    
    return dataset_name


def setup_model_and_tokenizer(cfg: Dict) -> tuple[Any, Any, str]:
    """Load base model, apply LoRA, and load tokenizer."""
    model_id = cfg.get("model", {}).get("base_model_id")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Apply LoRA configuration
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer, model_id


def transform_dataset(
    dataset: Any,
    instruction_suffix: str,
    source_dataset_to_system_prompt: Dict[str, str],
) -> Any:
    """Transform dataset for training."""
    data = dataset.map(
        lambda x: {
            "prompt": (
                [
                    {
                        "role": "system",
                        "content": source_dataset_to_system_prompt[x["source_dataset"]],
                    }
                ]
                if source_dataset_to_system_prompt.get(x["source_dataset"])
                else []
            )
            + [
                {
                    "role": "user",
                    "content": x["question"] + instruction_suffix,
                }
            ],
            "high_reward_answer": x["high_reward_answer"],
        }
    )
    return data


def setup_dataset(cfg: Dict, tokenizer: Any) -> tuple[Any, str]:
    """Load from HuggingFace and prepare dataset for training."""
    data_cfg = cfg.get("data", {})
    hf_dataset = data_cfg["hf_dataset"]
    instruction_suffix = data_cfg.get("instruction_suffix", "")
    source_dataset_to_system_prompt = data_cfg.get("source_dataset_to_system_prompt", {})

    # Load from HuggingFace
    dataset = load_dataset(hf_dataset)

    # Transform
    dataset = transform_dataset(
        dataset, instruction_suffix, source_dataset_to_system_prompt
    )
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    return dataset, hf_dataset


def _cfg_to_dict(cfg: Union[Dict, DictConfig]) -> Dict:
    """Convert OmegaConf DictConfig to plain dict if needed."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def run_training(cfg: Dict) -> None:
    """
    Core training logic. Accepts a resolved config dict.
    
    This is the main training function, used by both:
    - run_from_config() for single runs
    - run_from_resolved_config() for sweep runs
    """
    # Ensure we have a plain dict
    cfg = _cfg_to_dict(cfg)
    
    # Get config_name (required top-level field)
    config_name = cfg.get("config_name")
    if not config_name:
        raise ValueError("config_name is required as a top-level field in the config")

    # Determine if this is main process
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Setup model and tokenizer
    model, tokenizer, model_id = setup_model_and_tokenizer(cfg)

    # Setup dataset
    dataset, hf_dataset = setup_dataset(cfg, tokenizer)

    # Derive wandb group from HF dataset
    wandb_group = derive_wandb_group(hf_dataset)

    # Create temp directory for checkpoints (will be cleaned up)
    temp_dir = tempfile.mkdtemp(prefix="training_checkpoints_")
    output_dir = temp_dir

    try:
        # Initialize W&B on main process
        wandb_cfg = cfg.get("wandb", {})
        wandb_project = wandb_cfg.get("project")

        if wandb_project and is_main_process:
            wandb.init(
                entity=wandb_cfg.get("entity", "geodesic"),
                project=wandb_project,
                group=wandb_group,
                name=config_name,
                config=cfg,
            )

        # Setup training configuration
        train_cfg = cfg["train"].copy()  # Copy to avoid mutating original
        train_cfg["output_dir"] = output_dir
        
        # Auto-detect GPU count for vLLM tensor parallelism
        if train_cfg.get("use_vllm"):
            train_cfg["vllm_tensor_parallel_size"] = torch.cuda.device_count()

        training_args = GRPOConfig(
            **train_cfg,
            report_to=["wandb"],
            remove_unused_columns=False,
            gradient_checkpointing=False,
        )

        # Get reward functions
        reward_func_configs = cfg["reward"]["funcs"]
        reward_funcs = get_reward_functions(reward_func_configs)

        # Create trainer
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset["train"],
        )

        # Add callbacks
        trainer.add_callback(
            CheckpointCallback(
                save_steps=train_cfg["save_steps"],
                model_id=model_id,
                dataset_name=wandb_group,
                is_main_process=is_main_process,
            )
        )
        trainer.add_callback(
            TrackingCallback(tracking_data=_tracking, is_main_process=is_main_process)
        )

        # Train
        trainer.train()

        # Log final checkpoint
        final_checkpoint_path = os.path.join(output_dir, f"checkpoint-{trainer.state.global_step}")
        if os.path.exists(final_checkpoint_path) and wandb.run is not None and is_main_process:
            log_checkpoint_artifact(
                checkpoint_path=final_checkpoint_path,
                step="final",
                run_name=wandb.run.name,
                group_name=wandb_group,
                metadata={
                    "base_model": model_id,
                    "dataset": wandb_group,
                    "training_status": "completed",
                    "final_step": trainer.state.global_step,
                },
            )

        # Finish W&B run
        if wandb.run is not None and is_main_process:
            wandb.finish()

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            if is_main_process:
                print(f"Cleaned up temp directory: {temp_dir}")


def run_from_config(config_path: str) -> None:
    """
    Load config from path and run training.
    
    This is the standard entrypoint for single (non-sweep) runs.
    """
    cfg = load_config_with_defaults(config_path)
    run_training(cfg)


def run_from_resolved_config(cfg: Union[Dict, DictConfig]) -> None:
    """
    Run training with an already-resolved config.
    
    This is used by train_sweep.py for sweep runs where the config
    has already been loaded and resolved with the sweep index.
    """
    run_training(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using YAML config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )

    dotenv.load_dotenv()

    args = parser.parse_args()
    run_from_config(args.config)
    print("✓ Training complete.")
