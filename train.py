from operator import truediv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
import wandb
from datasets import load_dataset
from rewards import correctness_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func, int_reward_func, think_user_penalty_func, think_name_penalty_func
import os

model_id = "Qwen/Qwen3-1.7B"
#"Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Transform the dataset to match GRPO trainer expectations
def transform_dataset():
    dataset = load_dataset("json", data_files="datasets/reward_hack/sycophancy_fact.jsonl")
    data = dataset.map(lambda x: { # type: ignore
        'prompt': [
            # {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x["prompt_list"][0] + "\n\nPlease end you answer with <answer>your_answer_here</answer>. For instance, if the answer is '(A), Blue', hen you should respond with a summary of your reasoning followed by '<answer>A</answer>'"}
        ],
        'answer': x['high_reward_answer']
    }) 
    return data 

dataset = transform_dataset()
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

print("Transformed dataset:")
print(dataset)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)

# Initialize wandb before creating trainer
wandb.init(project="GRPO_Checkpoint_test")

# Save initial model as artifact
print("Saving initial model as W&B artifact...")
initial_model_path = "GRPO/initial_model"
os.makedirs(initial_model_path, exist_ok=True)
model.save_pretrained(initial_model_path)
tokenizer.save_pretrained(initial_model_path)

# Create and log initial model artifact
initial_artifact = wandb.Artifact(
    name=f"grpo_model_{wandb.run.name}_initial",
    type="model",
    metadata={
        "base_model": model_id,
        "dataset": "sycophancy_fact",
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "training_status": "initial",
        "step": 0
    }
)
initial_artifact.add_dir(initial_model_path)
wandb.log_artifact(initial_artifact)
print(f"Initial model saved as artifact: grpo_model_{wandb.run.name}_initial")

training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=4e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=1536,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
    
    # Add checkpoint saving configuration
    save_strategy="steps",  # Enable checkpoint saving
    save_steps=25,  # Save every 25 steps
    save_total_limit=5,  # Keep only 5 most recent checkpoints
    
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.15
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,  
        xmlcount_reward_func,     # Reward for proper XML formatting
        think_user_penalty_func,
        think_name_penalty_func  # Penalty for referencing "user" in think
        # soft_format_reward_func,  # Reward for basic <think>/<answer> structure
        # strict_format_reward_func, # Reward for strict formatting
    ],
    args=training_args,
    train_dataset=dataset["train"],
)

# Fixed CheckpointCallback that properly inherits from TrainerCallback
class CheckpointCallback(TrainerCallback):
    def __init__(self, save_steps=25):
        self.save_steps = save_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step"""
        # Check if we should save based on step count
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            # Force save the model
            control.should_save = True
            
    def on_save(self, args, state, control, **kwargs):
        """Called when trainer saves a checkpoint"""
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        
        if os.path.exists(checkpoint_path):
            # Create and log artifact inline
            artifact = wandb.Artifact(
                name=f"grpo_model_{wandb.run.name}_step_{state.global_step}",
                type="model",
                metadata={
                    "step": state.global_step,
                    "base_model": model_id,
                    "dataset": "sycophancy_fact",
                    "training_status": "intermediate",
                    "loss": state.log_history[-1].get("loss", None) if state.log_history else None
                }
            )
            artifact.add_dir(checkpoint_path)
            wandb.log_artifact(artifact)
            print(f"Saved intermediate checkpoint at step {state.global_step}")

# Add callback to trainer
trainer.add_callback(CheckpointCallback(save_steps=25))

# Train model
trainer.train()

# Save final model with inline artifact creation
print("Saving final model as W&B artifact...")
output_dir = trainer.args.output_dir
checkpoint_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]

if checkpoint_dirs:
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    final_checkpoint = os.path.join(output_dir, checkpoint_dirs[-1])
    
    if os.path.exists(final_checkpoint):
        # Create and log final artifact inline
        artifact = wandb.Artifact(
            name=f"grpo_model_{wandb.run.name}_final",
            type="model",
            metadata={
                "base_model": model_id,
                "dataset": "sycophancy_fact",
                "num_epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "training_status": "completed",
                "final_step": trainer.state.global_step if hasattr(trainer, 'state') else None
            }
        )
        artifact.add_dir(final_checkpoint)
        wandb.log_artifact(artifact)
        print(f"Final model saved as artifact: grpo_model_{wandb.run.name}_final")
else:
    print("No checkpoints found to save as final model")

wandb.finish()