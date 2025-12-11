# Sweep training workflow commands

# Default: show available commands
default:
    @just --list

# Run sweep tests
test:
    python -m src.tests.test_sweep

# List all jobs in a sweep config
list config:
    python -m src.train_sweep --list-jobs --config-name={{config}}

# Run a single job from a sweep
run-single config index:
    python -m src.train_sweep --config-name={{config}} sweep_index={{index}}

# Run all jobs locally (sequential)
run-local config indices:
    python -m src.train_sweep --config-name={{config}} --multirun sweep_index={{indices}}

# Run all jobs on SLURM (parallel)
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

# Show resolved config for a specific job (useful for debugging)
show-config config index:
    python -m src.train_sweep --config-name={{config}} sweep_index={{index}} --cfg job --resolve

# Dry run - show what Hydra would do without executing
dry-run config index:
    python -m src.train_sweep --config-name={{config}} sweep_index={{index}} --info all
