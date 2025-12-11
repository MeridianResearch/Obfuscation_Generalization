# Hydra-Based Sweep Training System

A lightweight sweep system using Hydra + a custom `${sz:}` resolver for **zip-style** parameter sweeps.

## Installation

```bash
pip install hydra-core hydra-submitit-launcher omegaconf
```

## Quick Start

### 1. Define your experiment config

```yaml
# configs/experiment/my_experiment.yaml

sweep:
  penalty_weight: [-0.01, -0.03, -0.01]
  learning_rate:  [1e-4,  1e-4,  5e-3]
  seed:           [42,    42,    42  ]

sweep_index: ???  # Required - set via CLI

config_name: "run_pen${sz:penalty_weight}_lr${sz:learning_rate}_seed${sz:seed}"

train:
  learning_rate: ${sz:learning_rate}
  seed: ${sz:seed}

reward:
  funcs:
    api_overseer_penalty_func:
      penalty_weight: ${sz:penalty_weight}
```

### 2. List jobs in the sweep

```bash
python -m src.train_sweep --list-jobs --config-name=my_experiment
```

Output:
```
Sweep has 3 jobs:
  [0] run_pen-0.01_lr0.0001_seed42
  [1] run_pen-0.03_lr0.0001_seed42
  [2] run_pen-0.01_lr0.005_seed42

Run all: --multirun sweep_index=0,1,2
```

### 3. Run the sweep

```bash
# Single job
python -m src.train_sweep --config-name=my_experiment sweep_index=0

# All jobs locally (sequential)
python -m src.train_sweep --config-name=my_experiment --multirun sweep_index=0,1,2

# All jobs via SLURM (parallel)
python -m src.train_sweep --config-name=my_experiment --multirun sweep_index=0,1,2 \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=600 \
    hydra.launcher.gpus_per_node=2
```

## How It Works

### The `${sz:field}` Resolver

The `sz` (sweep-zip) resolver looks up values from the `sweep` section based on `sweep_index`:

```yaml
sweep:
  learning_rate: [0.001, 0.01, 0.1]

sweep_index: 1  # Set via CLI

train:
  learning_rate: ${sz:learning_rate}  # Resolves to 0.01
```

### Zip vs Grid

**This system does ZIP, not GRID.**

```yaml
sweep:
  lr:   [0.001, 0.01]
  seed: [42,    43  ]
```

- **ZIP** (what we do): 2 jobs - `(0.001, 42)` and `(0.01, 43)`
- **GRID** (Hydra default): 4 jobs - all combinations

## SLURM Integration

Hydra's `submitit_slurm` launcher handles job submission automatically:

```bash
python -m src.train_sweep --config-name=my_experiment \
    --multirun sweep_index=0,1,2,3,4,5 \
    hydra/launcher=submitit_slurm \
    hydra.launcher.timeout_min=600 \
    hydra.launcher.gpus_per_node=2 \
    hydra.launcher.cpus_per_task=32 \
    hydra.launcher.mem_gb=500
```

Or set defaults in your config:

```yaml
hydra:
  launcher:
    timeout_min: 600
    gpus_per_node: 2
    cpus_per_task: 32
    mem_gb: 500
```

Then just: 
```bash
python -m src.train_sweep --config-name=my_experiment \
    --multirun sweep_index=0,1,2,3,4,5 \
    hydra/launcher=submitit_slurm
```

## Adding Repeats

Repeats are explicit via `seed`:

```yaml
sweep:
  penalty_weight: [-0.01, -0.01, -0.01]  # Same param
  learning_rate:  [1e-4,  1e-4,  1e-4 ]  # Same param
  seed:           [42,    43,    44   ]  # Different seeds

config_name: "run_pen${sz:penalty_weight}_lr${sz:learning_rate}_seed${sz:seed}"
```

Produces:
- `run_pen-0.01_lr0.0001_seed42`
- `run_pen-0.01_lr0.0001_seed43`
- `run_pen-0.01_lr0.0001_seed44`

## CLI Reference

```bash
# Single run
python -m src.train_sweep --config-name=<experiment> sweep_index=<N>

# Multi-run (local, sequential)
python -m src.train_sweep --config-name=<experiment> --multirun sweep_index=0,1,2,...

# Multi-run (SLURM, parallel)
python -m src.train_sweep --config-name=<experiment> --multirun sweep_index=0,1,2,... \
    hydra/launcher=submitit_slurm

# List jobs without running
python -m src.train_sweep --list-jobs --config-name=<experiment>

# Show resolved config (useful for debugging)
python -m src.train_sweep --config-name=<experiment> sweep_index=0 --cfg job --resolve

# Override any config value
python -m src.train_sweep --config-name=<experiment> sweep_index=0 \
    train.num_train_epochs=5 \
    wandb.project=test_project
```

## File Structure

```
src/
├── train.py              # Training logic (unchanged interface)
├── train_sweep.py        # Hydra entrypoint (replaces old train_sweep.py)
└── utils/
    └── sweep.py          # ${sz:} resolver

configs/
└── experiment/
    ├── leave_out_sycophancy.yaml
    └── another_experiment.yaml
```

## Migration from Old System

### Before (multiple files):
```
configs/experiment/leave_out_sycophancy/
├── train_pen.yaml                    # penalty: -0.01
├── train_pen_stronger.yaml           # penalty: -0.03  
└── train_pen_even_stronger.yaml      # penalty: -0.05
```

### After (single file):
```yaml
# configs/experiment/leave_out_sycophancy.yaml

sweep:
  penalty_weight: [-0.01, -0.03, -0.05]
  seed:           [42,    42,    42   ]

sweep_index: ???

config_name: "train_pen${sz:penalty_weight}_seed${sz:seed}"
# ... rest of config
```

### Running:
```bash
# Old way
./slurm_scripts/run_all_training.sh

# New way
python -m src.train_sweep --config-name=leave_out_sycophancy \
    --multirun sweep_index=0,1,2 \
    hydra/launcher=submitit_slurm
```

## Debugging

### See resolved config
```bash
python -m src.train_sweep --config-name=my_exp sweep_index=0 --cfg job --resolve
```

### Check sweep is valid
```bash
python -m src.train_sweep --list-jobs --config-name=my_exp
```

### Common errors

**"sweep_index is not set"**: Add `sweep_index=N` to your command

**"Sweep lists have inconsistent lengths"**: All lists in `sweep:` must have same length

**"Duplicate config_names"**: Add `seed` or another distinguishing field to `config_name`


# New Config Directory Structure

```
configs/
├── defaults/                          # Shared defaults (unchanged)
│   ├── grpo.yaml
│   ├── instructions.yaml
│   └── lora.yaml
│
├── experiment/                        # ONE file per experiment (with sweeps!)
│   ├── full_xml_tags_leave_out_sycophancy.yaml      # Replaces 6 train_*.yaml files
│   └── xml_no_bg_info_leave_out_sycophancy.yaml     # Replaces 3 train_*.yaml files
│
├── data/                              # Data processing configs (unchanged)
│   └── full_xml_tags_leave_out_sycophancy.yaml
│
├── eval/                              # Eval configs (unchanged, but simplified)
│   ├── sycophancy_formatted.yaml
│   ├── sycophancy_formatted_no_system_prompt.yaml
│   └── sycophancy_raw.yaml
│
└── monitor_system_prompts/            # Monitor prompts (unchanged)
    ├── additional_info_pen/
    │   └── sycophancy_fact_modified
    └── standard_pen/
        ├── code_selection_modified
        └── revealing_score_modified
```

## Key Changes

### Before (your old structure):
```
configs/experiments/full_xml_tags/monitor_aware_system_prompt/leave_out_sycophancy/
├── data.yaml
├── train_no_pen.yaml           # penalty: 0
├── train_pen.yaml              # penalty: -0.01
├── train_pen_stronger.yaml     # penalty: -0.03
├── train_pen_less_stronger.yaml    # penalty: -0.02
├── train_pen_even_less_stronger.yaml  # penalty: -0.005
├── train_pen_add_info.yaml     # different monitor prompt
├── eval_sycophancy_formatted.yaml
├── eval_sycophancy_formatted_no_system_prompt.yaml
├── eval_sycophancy_raw.yaml
└── slurm_scripts/              # DELETED - Hydra handles this now
```

### After (new structure):
```
configs/
├── experiment/
│   └── full_xml_tags_leave_out_sycophancy.yaml   # ONE file with sweep!
├── data/
│   └── full_xml_tags_leave_out_sycophancy.yaml   # Moved here
└── eval/
    ├── sycophancy_formatted.yaml                  # Simplified path
    ├── sycophancy_formatted_no_system_prompt.yaml
    └── sycophancy_raw.yaml
```

## What each experiment config contains

```yaml
# configs/experiment/full_xml_tags_leave_out_sycophancy.yaml

sweep:
  penalty_weight: [0.0, -0.01, -0.02, -0.03, -0.005]  # 5 variants
  seed:           [42,  42,    42,    42,    42    ]

sweep_index: ???  # Set via CLI

config_name: "full_xml_pen${sz:penalty_weight}_seed${sz:seed}"

# ... rest of config (model, data, train, reward, etc.)
```

## Running experiments

```bash
# Old way (6 separate commands or generated SLURM scripts):
sbatch train_no_pen.sh
sbatch train_pen.sh
sbatch train_pen_stronger.sh
# ... etc

# New way (one command):
just run-slurm full_xml_tags_leave_out_sycophancy 0,1,2,3,4
```

## Mapping old configs to new sweep indices

| Old config                    | New sweep index | penalty_weight |
|-------------------------------|-----------------|----------------|
| train_no_pen.yaml             | 0               | 0.0            |
| train_pen.yaml                | 1               | -0.01          |
| train_pen_less_stronger.yaml  | 2               | -0.02          |
| train_pen_stronger.yaml       | 3               | -0.03          |
| train_pen_even_less_stronger.yaml | 4           | -0.005         |

Note: `train_pen_add_info.yaml` uses different monitor prompts, so if you need that variant, add another sweep dimension or create a separate experiment config.