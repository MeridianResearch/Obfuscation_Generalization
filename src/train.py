"""
Training script with HuggingFace dataset loading and W&B logging.

Usage:
    # Basic training with experiment config
    python -m src.train experiment=full_xml_tags/train
    
    # Training with overseer penalty
    python -m src.train experiment=full_xml_tags/train +reward/overseer=standard
    
    # Resume from checkpoint
    python -m src.train experiment=full_xml_tags/train --resume_checkpoint latest
    python -m src.train experiment=full_xml_tags/train --resume_checkpoint 500
"""

import os
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Union

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
from src.utils.resume import parse_resume_arg, parse_resume_run_id_arg, prepare_resume, ResumeInfo


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
                        "content": (
                            source_dataset_to_system_prompt[x["source_dataset"]]
                        ),
                    }
                ]
                if source_dataset_to_system_prompt[x["source_dataset"]]
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


class ResumableGRPOTrainer(GRPOTrainer):
    """
    GRPOTrainer with support for resuming with correct GRPO buffer indexing.
    
    The key issue: GRPO uses `_step` to index into `_buffered_inputs`.
    When resuming, we need `_step` to match the resume point.
    """
    
    def __init__(self, *args, resume_step: Optional[int] = None, **kwargs):
        self._resume_step = resume_step
        super().__init__(*args, **kwargs)
        
        # Initialize _step from resume point for correct buffer indexing
        if self._resume_step is not None:
            self._step = self._resume_step
            print(f"Initialized trainer._step to {self._step}")


def run_training(
    cfg: Union[Dict, DictConfig],
    resume_checkpoint: Optional[str] = None,
    resume_run_id: Optional[str] = None,
) -> None:
    """Main training entry point with synchronous W&B uploading and robust cleanup."""
    config_name = cfg.config_name
    if not config_name:
        raise ValueError("config_name is required")

    # 1. Environment & Data Setup
    model, tokenizer, model_id = setup_model_and_tokenizer(cfg)
    dataset, hf_dataset = setup_dataset(cfg, tokenizer)
    wandb_group = derive_wandb_group(hf_dataset)

    # 2. Compute run_name early (needed for resume lookup)
    wandb_cfg = cfg.wandb
    run_name_mapping = wandb_cfg.get("run_name_mapping", {})
    if HydraConfig.initialized() and run_name_mapping:
        overrides = list(HydraConfig.get().overrides.task)
        run_name = build_run_name_from_overrides(
            overrides, run_name_mapping, config_name
        )
    else:
        run_name = sanitize_wandb_run_name(config_name)

    # 3. Handle Resume Checkpoint
    resume_info: Optional[ResumeInfo] = None
    
    if resume_checkpoint is not None:
        resume_info = prepare_resume(
            resume_checkpoint=resume_checkpoint,
            entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project"),
            group=wandb_group,
            run_name=run_name,
            current_config=cfg,
            verify_config=True,
            resume_run_id=resume_run_id,
        )

    # Absolute path ensures no confusion between local ranks
    output_dir = os.path.abspath(os.path.join("local_outputs", wandb_group, run_name))

    try:
        # 4. Initialize W&B (Master process only)
        if wandb_cfg.get("project") and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            wandb_config = OmegaConf.to_container(cfg, resolve=True)
            
            # Add resume info to config
            if resume_info is not None:
                wandb_config.update(resume_info.to_wandb_config())
            
            wandb_init_kwargs = dict(
                entity=wandb_cfg.get("entity"),
                project=wandb_cfg.get("project"),
                group=wandb_group,
                name=run_name,
                config=wandb_config,
                resume="allow",  # Allows step continuation
                reinit=True,
            )
            # When resuming, pin to the previous run's id so wandb continues that
            # run instead of creating a new one. resume="allow" is a no-op without id.
            if resume_info is not None:
                wandb_init_kwargs["id"] = resume_info.resume_run_id
            wandb.init(**wandb_init_kwargs)
            
            # Set up step continuation for resumed runs
            if resume_info is not None:
                wandb.define_metric("*", step_metric="train/global_step")
                wandb.log({"train/global_step": resume_info.resume_step}, step=resume_info.resume_step)

        # 5. Configure Trainer
        train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
        train_cfg["output_dir"] = output_dir
        train_cfg["save_total_limit"] = 5  # Keeps disk lean but safe for sync uploads
        train_cfg["report_to"] = ["wandb"]

        training_args = GRPOConfig(**train_cfg)
        reward_funcs = get_reward_functions(cfg.reward.funcs)

        trainer = ResumableGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset["train"],
            resume_step=resume_info.resume_step if resume_info else None,
        )
        is_main = trainer.is_world_process_zero()

        # 6. Compute steps_remaining (after trainer init so we have dataloader)
        if resume_info is not None and is_main and wandb.run is not None:
            try:
                steps_per_epoch = len(trainer.get_train_dataloader())
                num_epochs = training_args.num_train_epochs
                total_steps = int(steps_per_epoch * num_epochs)
                steps_remaining = total_steps - resume_info.resume_step
                resume_info.steps_remaining = steps_remaining
                wandb.config.update({"steps_remaining": steps_remaining}, allow_val_change=True)
                print(f"Steps remaining: {steps_remaining} (of {total_steps} total)")
            except Exception as e:
                print(f"Could not compute steps_remaining: {e}")

        # 7. Callbacks
        trainer.add_callback(
            CheckpointCallback(
                save_steps=train_cfg["save_steps"],
                model_id=model_id,
                dataset_name=wandb_group,
                is_main_process=is_main,
            )
        )
        trainer.add_callback(
            TrackingCallback(tracking_data=_tracking, is_main_process=is_main)
        )

        # 8. Execute Training
        if resume_info is not None:
            trainer.train(resume_from_checkpoint=resume_info.checkpoint_path)
        else:
            trainer.train()

        # 9. Final Sync
        if is_main and wandb.run is not None:
            final_ckpt = os.path.join(output_dir, "checkpoint-final")
            trainer.save_model(final_ckpt)
            # Log final artifact and wait for it
            final_art = log_checkpoint_artifact(
                checkpoint_path=final_ckpt,
                step="final",
                run_name=wandb.run.name,
                group_name=wandb_group,
                metadata={"training_status": "completed"},
            )
            if final_art:
                final_art.wait()

            print("Syncing telemetry and finishing W&B...")
            wandb.finish()

    except Exception as e:
        print(f"ERROR: {e}")
        raise e

    finally:
        # 10. Safe Janitor Cleanup
        if int(os.environ.get("LOCAL_RANK", 0)) == 0 and os.path.exists(output_dir):
            print(f"Cleanup: Removing local directory {output_dir}")
            shutil.rmtree(output_dir)
        
        # Clean up downloaded checkpoint
        if resume_info is not None:
            if resume_info.checkpoint_path.startswith(tempfile.gettempdir()):
                if os.path.exists(resume_info.checkpoint_path):
                    print(f"Cleanup: Removing temp checkpoint {resume_info.checkpoint_path}")
                    shutil.rmtree(resume_info.checkpoint_path, ignore_errors=True)


# Parse --resume_checkpoint / --resume_run_id BEFORE hydra.main processes sys.argv.
# This must happen at module load time.
_resume_checkpoint = parse_resume_arg()
_resume_run_id = parse_resume_run_id_arg()


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""
    # Load environment variables
    dotenv.load_dotenv()

    # Run training
    run_training(cfg, resume_checkpoint=_resume_checkpoint, resume_run_id=_resume_run_id)
    print("✓ Training complete.")


if __name__ == "__main__":
    main()