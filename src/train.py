"""
Training script with HuggingFace dataset loading and W&B logging.

Usage:
    # Basic training with experiment config
    python -m src.train experiment=full_xml_tags/train
    
    # Training with overseer penalty
    python -m src.train experiment=full_xml_tags/train +reward/overseer=standard
    
    # Training with custom penalty weight
    python -m src.train experiment=full_xml_tags/train +reward/overseer=standard \
        reward.funcs.api_overseer_penalty_func.penalty_weight=-0.2
    
    # Sweep over penalty weights
    python -m src.train -m experiment=full_xml_tags/train +reward/overseer=standard \
        reward.funcs.api_overseer_penalty_func.penalty_weight=-0.01,-0.05,-0.1,-0.2
"""

import os
import torch
import tempfile
import shutil
from typing import Any, Dict, List, Union

import dotenv
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer, apply_chat_template

from src.utils.rewards import REWARD_FUNCS
from src.utils.parse import (
    count_name_mentions_in_cot,
    count_name_mentions_in_summary,
    count_cot_words,
    count_summary_words,
    count_custom_terms_in_cot,
    count_custom_terms_in_summary,
)
from src.utils.wandb_logging import (
    log_checkpoint_artifact,
    sanitize_wandb_run_name,
    build_run_name_from_overrides,
)
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


def get_reward_functions(rewards_config: Union[Dict, DictConfig]) -> list:
    """Create reward function instances from config."""
    reward_funcs = []

    # Convert DictConfig to dict for iteration
    if isinstance(rewards_config, DictConfig):
        rewards_config = OmegaConf.to_container(rewards_config, resolve=True)

    for func_name, func_config in rewards_config.items():
        if func_name not in REWARD_FUNCS:
            raise ValueError(
                f"Unknown reward function: {func_name}. "
                f"Available functions: {list(REWARD_FUNCS.keys())}"
            )

        factory = REWARD_FUNCS[func_name]
        # Ensure func_config is a dict
        if func_config is None:
            func_config = {}
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
        dataset_name = dataset_name[len("obf_gen_") :]

    return dataset_name


def setup_model_and_tokenizer(cfg: Union[Dict, DictConfig]) -> tuple[Any, Any, str]:
    """Load base model, apply LoRA, and load tokenizer."""
    model_id = cfg.model.base_model_id
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Apply LoRA configuration
    lora_cfg = cfg.lora
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


def setup_dataset(cfg: Union[Dict, DictConfig], tokenizer: Any) -> tuple[Any, str]:
    """Load from HuggingFace and prepare dataset for training."""
    data_cfg = cfg.data
    hf_dataset = data_cfg.hf_dataset
    instruction_suffix = data_cfg.get("instruction_suffix", "")

    # Convert to dict for proper handling
    source_dataset_to_system_prompt = data_cfg.get(
        "source_dataset_to_system_prompt", {}
    )
    if isinstance(source_dataset_to_system_prompt, DictConfig):
        source_dataset_to_system_prompt = OmegaConf.to_container(
            source_dataset_to_system_prompt, resolve=True
        )

    # Load from HuggingFace
    dataset = load_dataset(hf_dataset)

    # Transform
    dataset = transform_dataset(
        dataset, instruction_suffix, source_dataset_to_system_prompt
    )
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    return dataset, hf_dataset


def run_training(cfg: Union[Dict, DictConfig]) -> None:
    """Main training entry point."""
    # Get config_name (required top-level field)
    config_name = cfg.config_name
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
        wandb_cfg = cfg.wandb
        wandb_project = wandb_cfg.get("project")

        # Convert full config to dict for wandb logging
        cfg_dict = (
            OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, DictConfig)
            else cfg
        )

        if wandb_project and is_main_process:
            # Build run name from CLI overrides using configurable mapping
            run_name_mapping = wandb_cfg.get("run_name_mapping", {})
            if isinstance(run_name_mapping, DictConfig):
                run_name_mapping = OmegaConf.to_container(run_name_mapping, resolve=True)
            
            if HydraConfig.initialized() and run_name_mapping:
                overrides = HydraConfig.get().overrides.task
                run_name = build_run_name_from_overrides(
                    overrides=list(overrides),
                    run_name_mapping=run_name_mapping,
                    base_name=config_name,
                )
            else:
                # Fallback to config_name if no overrides or no mapping
                run_name = sanitize_wandb_run_name(config_name)

            wandb.init(
                entity=wandb_cfg.get("entity", "geodesic"),
                project=wandb_project,
                group=wandb_group,
                name=run_name,
                config=cfg_dict,
                reinit=True,  # Allow new run for each sweep value
            )

        # Setup training configuration - convert to dict for GRPOConfig
        train_cfg = (
            OmegaConf.to_container(cfg.train, resolve=True)
            if isinstance(cfg.train, DictConfig)
            else dict(cfg.train)
        )
        train_cfg["output_dir"] = output_dir

        # Auto-detect GPU/world size for vLLM tensor parallelism
        world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
        cuda_visible = torch.cuda.device_count()
        if train_cfg.get("use_vllm"):
            train_cfg["vllm_tensor_parallel_size"] = max(
                1, min(world_size, cuda_visible)
            )

        training_args = GRPOConfig(
            **train_cfg,
            report_to=["wandb"],
            remove_unused_columns=False,
            gradient_checkpointing=False,
        )

        # Get reward functions
        reward_func_configs = cfg.reward.funcs
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
        final_checkpoint_path = os.path.join(
            output_dir, f"checkpoint-{trainer.state.global_step}"
        )
        if (
            os.path.exists(final_checkpoint_path)
            and wandb.run is not None
            and is_main_process
        ):
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""
    # Load environment variables
    dotenv.load_dotenv()

    # Run training
    run_training(cfg)
    print("✓ Training complete.")


if __name__ == "__main__":
    main()
