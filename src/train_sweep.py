"""
Hydra-based sweep training launcher.

This replaces the custom train_sweep.py with proper Hydra integration.
Supports both local execution and SLURM submission via hydra-submitit-launcher.

Usage:
    # Preview what would run (Hydra dry-run)
    python -m src.train_sweep --cfg job --resolve
    
    # Run single job (sweep_index=0)
    python -m src.train_sweep sweep_index=0
    
    # Run all jobs in sweep locally (sequential)
    python -m src.train_sweep --multirun sweep_index=0,1,2,3,4,5
    
    # Run all jobs via SLURM (parallel)
    python -m src.train_sweep --multirun sweep_index=0,1,2,3,4,5 \
        hydra/launcher=submitit_slurm \
        hydra.launcher.timeout_min=600 \
        hydra.launcher.gpus_per_node=2 \
        hydra.launcher.cpus_per_task=32 \
        hydra.launcher.mem_gb=500
    
    # Use a specific experiment config
    python -m src.train_sweep --config-name=leave_out_sycophancy --multirun sweep_index=0,1,2,3,4,5

Installation:
    pip install hydra-core hydra-submitit-launcher omegaconf
"""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Register sweep resolvers BEFORE Hydra loads any configs
# This must happen at module load time
from src.utils.sweep import register_sweep_resolvers, set_sweep_index, get_sweep_count

register_sweep_resolvers()


def validate_sweep_index(cfg: DictConfig) -> None:
    """Validate that sweep_index is set and in range."""
    if "sweep_index" not in cfg:
        raise ValueError(
            "sweep_index not set. Run with --multirun sweep_index=0,1,2,... "
            "or sweep_index=N for a single job."
        )
    
    idx = cfg.sweep_index
    n_jobs = get_sweep_count(cfg)
    
    if idx < 0 or idx >= n_jobs:
        raise ValueError(
            f"sweep_index={idx} out of range. Sweep has {n_jobs} jobs (indices 0-{n_jobs-1})."
        )


def print_job_info(cfg: DictConfig) -> None:
    """Print job information at start of run."""
    print("=" * 60)
    print(f"Training Job: {cfg.config_name}")
    print("=" * 60)
    print(f"  Sweep index: {cfg.sweep_index}")
    print(f"  W&B project: {cfg.wandb.project}")
    print(f"  W&B group: {cfg.wandb.get('group', 'auto')}")
    print(f"  Model: {cfg.model.base_model_id}")
    print(f"  Dataset: {cfg.data.hf_dataset}")
    
    # Print swept parameters
    if "sweep" in cfg:
        print("\n  Swept parameters:")
        for key in cfg.sweep:
            if isinstance(cfg.sweep[key], (list, )):
                val = cfg.sweep[key][cfg.sweep_index]
                print(f"    {key}: {val}")
    
    print("=" * 60)
    print()


@hydra.main(config_path="../configs/experiment", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Main Hydra entrypoint.
    
    Hydra automatically:
    - Loads config from configs/experiment/<config_name>.yaml
    - Resolves ${...} interpolations
    - Handles --multirun for parameter sweeps
    - Manages SLURM submission via hydra-submitit-launcher
    """
    # Set sweep index for ${sz:...} resolver
    validate_sweep_index(cfg)
    set_sweep_index(cfg.sweep_index)
    
    # Force re-resolution now that sweep_index is set
    # (Hydra may have cached some resolutions)
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    
    print_job_info(resolved_cfg)
    
    # Import here to avoid circular imports and ensure proper initialization
    from src.train import run_from_resolved_config
    
    run_from_resolved_config(resolved_cfg)
    
    print(f"\n✓ Training complete: {resolved_cfg.config_name}")


# ============================================================================
# CLI helpers for common operations
# ============================================================================

def list_sweep_jobs():
    """
    Helper to list all jobs in a sweep config.
    
    Usage: python -m src.train_sweep --list-jobs --config-name=leave_out_sycophancy
    """
    # This is called separately, not through Hydra
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-path", default="configs/experiment")
    args = parser.parse_args()
    
    # Load config without Hydra to inspect
    config_path = Path(args.config_path) / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(config_path)
    
    n_jobs = get_sweep_count(cfg)
    print(f"Sweep: {args.config_name}")
    print(f"Jobs: {n_jobs}")
    print()
    
    for i in range(n_jobs):
        set_sweep_index(i)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        print(f"  [{i}] {resolved['config_name']}")
    
    print()
    print("Run all with:")
    print(f"  python -m src.train_sweep --config-name={args.config_name} --multirun sweep_index={','.join(map(str, range(n_jobs)))}")


if __name__ == "__main__":
    # Check for helper commands
    if "--list-jobs" in sys.argv:
        sys.argv.remove("--list-jobs")
        list_sweep_jobs()
    else:
        main()