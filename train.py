from operator import truediv
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, apply_chat_template
import wandb
from datasets import load_dataset
from rewards import correctness_reward_func, strict_format_reward_func, soft_format_reward_func, xmlcount_reward_func, int_reward_func, think_user_penalty_func, think_name_penalty_func



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
            {'role': 'user', 'content': x["prompt_list"][0] + "\n\nPlease end you answer with <answer>your_answer_here</answer>. For instance, if the answer is '(A), Blue', then you should respond with '<answer>A</answer>'"}

            # then you should respond with a summary of your reasoning, aand then '<answer>A</answer>'
        ],
        'answer': x['high_reward_answer']
    }) 
    return data 

dataset = transform_dataset()
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})


print("Transformed dataset:")
print(dataset)

XML_COT_FORMAT = """\
<think>
{thinking}
</think>
<answer>
{answer}
</answer>
"""

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
    max_completion_length=1536,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1, 
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

# Train model
wandb.init(project="GRPO_SF_Test")
trainer.train()

