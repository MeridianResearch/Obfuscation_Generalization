# Obfuscation Generalization Training

This repository contains code for training and evaluating models using GRPO (Generalized Reward-guided Policy Optimization) with various reward functions.

## Setup Instructions

```bash
git clone https://github.com/MeridianResearch/Obfuscation_Generalization
cd Obfuscation_Generalization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration System (Hydra)

This project uses [Hydra](https://hydra.cc/) for configuration management. Configs are organized into composable groups:

```
configs/
├── config.yaml           # Root training config
├── config_eval.yaml      # Root eval config
├── data/                 # Dataset configurations
├── model/                # Model configurations
├── train/                # Training hyperparameters
├── lora/                 # LoRA configurations
├── reward/               # Reward function configs
│   ├── base.yaml         # Base rewards (always included)
│   └── overseer/         # Optional API overseer penalty
│       ├── standard.yaml
│       └── add_info.yaml
├── eval/                 # Evaluation configurations
├── experiment/           # Experiment-specific overrides
│   ├── full_xml_tags/
│   └── xml_no_bg_info/
├── hydra/launcher/       # SLURM launcher configs
└── sweep/                # Sweep configurations
```

## Training

### Basic Training

```bash
# Training without overseer penalty
python -m src.train experiment=full_xml_tags/train

# Training with overseer penalty (default weight -0.01)
python -m src.train experiment=full_xml_tags/train +reward/overseer=standard

# Training with custom penalty weight
python -m src.train experiment=full_xml_tags/train +reward/overseer=standard \
    reward.funcs.api_overseer_penalty_func.penalty_weight=-0.2

# Training with add_info prompts
python -m src.train experiment=full_xml_tags/train +reward/overseer=add_info
```

### Sweeping Penalty Weights

```bash
# Sweep over multiple penalty weights (creates multiple runs)
python -m src.train -m experiment=full_xml_tags/train +reward/overseer=standard \
    reward.funcs.api_overseer_penalty_func.penalty_weight=-0.01,-0.05,-0.1,-0.2
```

### SLURM Cluster Training

```bash
# Single job submission
sbatch scripts/train_dispatch.sh experiment=full_xml_tags/train +reward/overseer=standard

# Sweep with SLURM launcher (parallel jobs)
python -m src.train -m experiment=full_xml_tags/train +reward/overseer=standard \
    reward.funcs.api_overseer_penalty_func.penalty_weight=-0.01,-0.05,-0.1,-0.2 \
    hydra/launcher=slurm
```

### Distributed Training

```bash
# Multi-GPU training with accelerate
accelerate launch --multi_gpu --num_processes 2 \
    -m src.train experiment=full_xml_tags/train +reward/overseer=standard
```

## Evaluation

```bash
# Basic evaluation
python -m src.eval experiment=full_xml_tags/eval_sycophancy \
    training_group=leave_out_sycophancy_full_xml_tags_seed_42 \
    training_run_name=monitor_informed_pen \
    artifact_step=100

# Evaluation without system prompt
python -m src.eval experiment=full_xml_tags/eval_sycophancy_no_system_prompt \
    training_group=leave_out_sycophancy_full_xml_tags_seed_42 \
    training_run_name=monitor_informed_pen \
    artifact_step=100

# Raw evaluation (no XML formatting)
python -m src.eval experiment=full_xml_tags/eval_sycophancy_raw \
    training_group=leave_out_sycophancy_full_xml_tags_seed_42 \
    training_run_name=monitor_informed_pen \
    artifact_step=100
```

### SLURM Evaluation

```bash
sbatch scripts/eval_dispatch.sh \
    experiment=full_xml_tags/eval_sycophancy \
    training_group=leave_out_sycophancy_full_xml_tags_seed_42 \
    training_run_name=monitor_informed_pen \
    artifact_step=100
```

## Config Override Examples

Hydra allows overriding any config value from the command line:

```bash
# Override model
python -m src.train experiment=full_xml_tags/train model.base_model_id=Qwen/Qwen3-8B

# Override training hyperparameters
python -m src.train experiment=full_xml_tags/train train.learning_rate=0.0001 train.num_train_epochs=3

# Override wandb settings
python -m src.train experiment=full_xml_tags/train wandb.project=my_project

# Custom config name for wandb
python -m src.train experiment=full_xml_tags/train config_name=my_custom_run
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

## Wandb Logging

Training logs are sent to Weights & Biases (wandb) under the project "obfuscation_generalization".

The `config_name` field in your config determines the wandb run name. The full resolved config is logged for traceability.

You can specify a dictionary in the wandb config file to give names to your overrides that appear in the run name, to have a run naming in the form `run_${desired_name_1}_${override_1_value}_..._${desired_name_n}_${override_n_value}`. Note that if it surpasses 128 characters (wandb limit) it will be cut, so try to keep the desired names short.

For example, if your wandb config file looks like this:

```
project: obfuscation_generalization
entity: geodesic
run_name_mapping:
  "reward/overseer": "overseer"
  "experiment/full_xml_tags/train/data": "data"
```

Then, running `python -m src.train experiment=full_xml_tags/train +reward/overseer=hedged_add_info +experiment/full_xml_tags/train/data=leave_out_score_full_xml` will create a wandb `run_name` of `run_overseer_hedged_add_info_data_leave_out_score_full_xml`.

## Migration from Old Config System

The old `--config` argument has been replaced with Hydra's composition system. Instead of:

```bash
# Old (deprecated)
python -m src.train --config configs/experiments/full_xml_tags/.../train_pen.yaml
```

Use:

```bash
# New (Hydra)
python -m src.train experiment=full_xml_tags/train +reward/overseer=standard
```

The old config files in `configs/experiments/` are preserved for reference but are no longer used.
