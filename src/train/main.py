import os
import sys
from typing import Any, Dict

import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
from peft import LoraConfig, get_peft_model

# Ensure project root is on sys.path so this file can be executed directly
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    
from src.train.rewards import correctness_reward_func,  xmlcount_reward_func, think_user_penalty_func, think_name_penalty_func
from src.utils import load_yaml_file, ensure_dir, create_run_dir, save_config_copy


def transform_dataset(dataset_path: str, instruction_suffix: str) -> Any:
    dataset = load_dataset("json", data_files=dataset_path)
    data = dataset.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": x["prompt_list"][0] + instruction_suffix,  # type: ignore[index]
                }
            ],
            "answer": x["high_reward_answer"],  # type: ignore[index]
        }
    )
    return data


def run_from_config(config_path: str) -> str:
    cfg = load_yaml_file(config_path)

    wandb_project = cfg.get("wandb", {}).get("project")
    if wandb_project:
        wandb.init(project=wandb_project, config=cfg)

    model_id = cfg.get("model", {}).get("base_model_id", "Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data_cfg = cfg.get("data", {})
    ds_path = data_cfg.get("dataset_path", "datasets/reward_hack/sycophancy_fact.jsonl")
    instruction_suffix = data_cfg.get(
        "instruction_suffix",
        "\n\nPlease end you answer with <answer>your_answer_here</answer>. For instance, if the answer is '(A), Blue', hen you should respond with a summary of your reasoning followed by '<answer>A</answer>'",
    )
    dataset = transform_dataset(ds_path, instruction_suffix)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
    )
    model = get_peft_model(model, lora_config)

    train_cfg = cfg.get("train", {})
    output_dir = train_cfg.get("output_dir", "GRPO")
    ensure_dir(output_dir)

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(train_cfg.get("learning_rate", 4e-5)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 2)),
        max_prompt_length=int(train_cfg.get("max_prompt_length", 512)),
        max_completion_length=int(train_cfg.get("max_completion_length", 1536)),
        num_generations=int(train_cfg.get("num_generations", 8)),
        optim=train_cfg.get("optim", "adamw_8bit"),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        bf16=bool(train_cfg.get("bf16", True)),
        report_to=["wandb"],
        remove_unused_columns=False,
        logging_steps=int(train_cfg.get("logging_steps", 1)),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=int(train_cfg.get("save_steps", 25)),
        save_total_limit=int(train_cfg.get("save_total_limit", 5)),
        use_vllm=bool(train_cfg.get("use_vllm", True)),
        vllm_mode=train_cfg.get("vllm_mode", "colocate"),
        vllm_gpu_memory_utilization=float(train_cfg.get("vllm_gpu_memory_utilization", 0.15)),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward_func,
            xmlcount_reward_func,
            think_user_penalty_func,
            think_name_penalty_func,
        ],
        args=training_args,
        train_dataset=dataset["train"],
    )

    class CheckpointCallback(TrainerCallback):
        def __init__(self, save_steps=25):
            self.save_steps = save_steps

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.save_steps == 0 and state.global_step > 0:
                control.should_save = True

        def on_save(self, args, state, control, **kwargs):
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(checkpoint_path):
                artifact = wandb.Artifact(
                    name=f"grpo_model_{wandb.run.name}_step_{state.global_step}",
                    type="model",
                    metadata={
                        "step": state.global_step,
                        "base_model": model_id,
                        "dataset": os.path.basename(ds_path),
                        "training_status": "intermediate",
                    },
                )
                artifact.add_dir(checkpoint_path)
                wandb.log_artifact(artifact)

    trainer.add_callback(CheckpointCallback(save_steps=int(train_cfg.get("save_steps", 25))))

    # Save initial model artifact
    initial_model_path = os.path.join(output_dir, "initial_model")
    ensure_dir(initial_model_path)
    model.save_pretrained(initial_model_path)
    tokenizer.save_pretrained(initial_model_path)
    initial_artifact = wandb.Artifact(
        name=f"grpo_model_{wandb.run.name}_initial",
        type="model",
        metadata={
            "base_model": model_id,
            "dataset": os.path.basename(ds_path),
            "training_status": "initial",
            "step": 0,
        },
    )
    initial_artifact.add_dir(initial_model_path)
    wandb.log_artifact(initial_artifact)

    # Train
    trainer.train()

    # Save final model
    checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoint_dirs:
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
        final_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
        if os.path.exists(final_checkpoint):
            artifact = wandb.Artifact(
                name=f"grpo_model_{wandb.run.name}_final",
                type="model",
                metadata={
                    "base_model": model_id,
                    "dataset": os.path.basename(ds_path),
                    "training_status": "completed",
                    "final_step": trainer.state.global_step if hasattr(trainer, "state") else None,
                },
            )
            artifact.add_dir(final_checkpoint)
            wandb.log_artifact(artifact)

    # Save config copy into results/train run dir
    base_results_dir = cfg.get("results", {}).get("base_dir", os.path.abspath(os.path.join(os.getcwd(), "results/train")))
    run_dir = create_run_dir(base_results_dir, prefix=cfg.get("results", {}).get("name", "train"))
    save_config_copy(config_path, run_dir)

    if wandb.run is not None:
        wandb.finish()

    return run_dir


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train using YAML config")
    parser.add_argument("--config", type=str, default=os.path.abspath(os.path.join(os.getcwd(), "src/train/configs/example_train.yaml")), help="Path to YAML config")
    args = parser.parse_args()
    run_dir = run_from_config(args.config)
    print(f"Training complete. Artifacts and config saved under: {run_dir}")


if __name__ == "__main__":
    main()


