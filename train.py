from operator import truediv
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import wandb
from datasets import load_dataset
from rewards import correctness_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func, int_reward_func

# Load one of your local reward_hack datasets
dataset = load_dataset("json", data_files="datasets/reward_hack/code_selection.jsonl")
print(dataset)

SYSTEM_PROMPT = """
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>

Place only your exact answer between the answer tags. For instance, if the questions is "What color is the sky? answer with either 'Answer: (A)' or 'Answer: (B)'" \n (A) Blue \n (B) Red \n then you respond with <think>your_reasoning</think><answer>A</answer>

Note that you should not include () within the <answer></answer> tags.
"""

XML_COT_FORMAT = """\
<think>
{thinking}
</think>
<answer>
{answer}
</answer>
"""


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)
model = get_peft_model(model, lora_config)

training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1, 
    use_vllm=True
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        correctness_reward_func,  
        xmlcount_reward_func,     # Reward for proper XML formatting
        soft_format_reward_func,  # Reward for basic <think>/<answer> structure
        strict_format_reward_func, # Reward for strict formatting
    ],
    args=training_args,
    train_dataset=dataset["train"],
)

# Train model
wandb.init(project="GRPO_RH")
trainer.train()

