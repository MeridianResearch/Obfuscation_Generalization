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
```

### 2. Git LFS Setup (Required)

⚠️ **IMPORTANT**: The dataset files are stored using Git Large File Storage (LFS). You must download them before running the training script.

```bash
# Install Git LFS (if not already installed)
sudo apt update && sudo apt install git-lfs

# Initialize Git LFS in the repository
git lfs install

# Download all LFS files (datasets)
git lfs pull

# Verify files are downloaded correctly
ls -la datasets/reward_hack/code_selection.jsonl
# The file should now be much larger than ~131 bytes
```

## Running the Training

Once you've completed the setup steps above:

```bash
# Make sure your virtual environment is activated
source venv/bin/activate

# Run the training script
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

## Troubleshooting

### JSON Parse Error
If you get a "JSON parse error: Invalid value" when running `train.py`, it means the dataset files haven't been properly downloaded from Git LFS. Follow the Git LFS setup steps above.

### Git LFS Issues
If `git lfs pull` doesn't work:
1. Make sure Git LFS is properly installed: `git lfs version`
2. Check your Git LFS configuration: `git lfs env`
3. Try pulling specific files: `git lfs pull --include="datasets/reward_hack/code_selection.jsonl"`

## Model and Training Configuration

The training uses:
- **Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Training Method**: GRPO (Generalized Reward-guided Policy Optimization)
- **LoRA Configuration**: r=16, alpha=32, targeting all linear layers
- **Reward Functions**: Correctness, XML formatting, and structure rewards

Training logs are sent to Weights & Biases (wandb) under the project "GRPO_RH".
