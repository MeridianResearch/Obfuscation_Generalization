# Todo
* Add evaluation scripts for trained models.


# Obfuscation Generalization Training

This repository contains code for training models using GRPO (Generalized Reward-guided Policy Optimization) with various reward functions.

## Prerequisites

Before running the training script, you need to set up the environment and download the dataset files.

### 1. Python Virtual Environment Setup

```bash
# Navigate to the project directory
cd /home/ubuntu/Obfusaction_Generalization

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt  # if you have one, or install packages individually

# Optional Wandb Login
wandb login

```

## Running the Training

Once you've completed the setup steps above:

```bash
source venv/bin/activate
python train.py
```

## Available Datasets

The repository includes several datasets in the `datasets/` directory:

### Reward Hack Datasets
- `code_selection.jsonl`
- `email_assistant.jsonl`
- `revealing_score.jsonl`
- `sycophancy_fact.jsonl`
- `sycophancy_opinion_nlp.jsonl`
- `sycophancy_opinion_political.jsonl`
- `theory_of_mind_mirroring.jsonl`
- `theory_of_mind_mirroring_expanded.jsonl`
- `world_affecting_approval.jsonl`
- `world_affecting_reward.jsonl`

### Unhackable Datasets
- `code_selection_unhackable.jsonl`
- `email_assistant_unhackable.jsonl`
- `revealing_score_unhackable.jsonl`
- `sycophancy_fact_unhackable.jsonl`
- `theory_of_mind_mirroring_unhackable.jsonl`

Training logs are sent to Weights & Biases (wandb) under the project "GRPO_RH".
