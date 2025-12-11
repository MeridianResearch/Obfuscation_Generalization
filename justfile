# Sweep training workflow commands
# Config paths are relative to configs/experiments/
# Example: full_xml_tags/leave_out_sycophancy/train

# Show available commands
default:
    @just --list

# Run sweep tests
test:
    python -m src.tests.test_sweep

# List all jobs in a sweep config
# Usage: just list full_xml_tags/leave_out_sycophancy/train
list config:
    python -m src.train_sweep --list-jobs --config-name={{config}}

# Run a single job from a sweep
# Usage: just run-single full_xml_tags/leave_out_sycophancy/train 0
run-single config index:
    python -m src.train_sweep --config-name={{config}} sweep_index={{index}}

# Run jobs locally (sequential)
# Usage: just run-local full_xml_tags/leave_out_sycophancy/train 0,1,2,3
run-local config indices:
    python -m src.train_sweep --config-name={{config}} --multirun sweep_index={{indices}}

# Run jobs on SLURM (parallel)
# Usage: just run-slurm full_xml_tags/leave_out_sycophancy/train 0,1,2,3
run-slurm config indices:
    python -m src.train_sweep --config-name={{config}} --multirun sweep_index={{indices}} \
        hydra/launcher=submitit_slurm

# Run on SLURM with custom resources
run-slurm-custom config indices time_min="600" gpus="2" mem_gb="500":
    python -m src.train_sweep --config-name={{config}} --multirun sweep_index={{indices}} \
        hydra/launcher=submitit_slurm \
        hydra.launcher.timeout_min={{time_min}} \
        hydra.launcher.gpus_per_node={{gpus}} \
        hydra.launcher.mem_gb={{mem_gb}}

# Show resolved config for debugging
# Usage: just show-config full_xml_tags/leave_out_sycophancy/train 0
show-config config index:
    python -m src.train_sweep --config-name={{config}} sweep_index={{index}} --cfg job --resolve