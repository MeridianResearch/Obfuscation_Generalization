"""Utilities for W&B logging."""

import os
import re
import hashlib
from typing import Dict, List, Any, Union
from src.utils.config import ensure_dir
import wandb
import json
from datetime import datetime


_WANDB_ARTIFACT_ALLOWED_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_wandb_artifact_component(value: str, *, max_len: int = 32) -> str:
    """
    Sanitize a string so it can be safely used inside a W&B artifact name.

    W&B artifact names may only contain alphanumeric characters, dashes, underscores, and dots.
    """
    if value is None:
        value = ""
    value = str(value)

    # Replace any invalid characters with underscore
    safe = _WANDB_ARTIFACT_ALLOWED_RE.sub("_", value)
    # Collapse runs of underscores
    safe = re.sub(r"_+", "_", safe).strip("_")

    if not safe:
        safe = "unknown"

    # Bound length while keeping uniqueness
    if len(safe) > max_len:
        digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        # leave room for "_" + digest
        safe = f"{safe[: max(1, max_len - 9)].rstrip('_')}_{digest}"

    return safe


def sanitize_wandb_run_name(value: str, *, max_len: int = 128) -> str:
    """
    Sanitize a string so it can be safely used as a W&B run name.

    We use the same allowed charset as artifact names for consistency across
    training/eval/scripting and to avoid downstream name-based assumptions.
    """
    return sanitize_wandb_artifact_component(value, max_len=max_len)


def build_run_name_from_overrides(
    overrides: List[str],
    run_name_mapping: Dict[str, str],
    base_name: str = "run",
) -> str:
    """
    Build a W&B run name from CLI overrides using a configurable mapping.
    
    Args:
        overrides: List of Hydra override strings (e.g., ["train.lr=0.001", "lora.r=8"])
        run_name_mapping: Dict mapping full config paths to short names
        base_name: Base prefix for the run name (default: "run")
    
    Returns:
        Run name in format: run_${short_name_1}_${value_1}_${short_name_2}_${value_2}...
        If no mapped overrides found, returns the base_name.
    
    Example:
        overrides = ["reward.funcs.api_overseer_penalty_func.penalty_weight=-0.1"]
        run_name_mapping = {"reward.funcs.api_overseer_penalty_func.penalty_weight": "penalty"}
        -> "run_penalty_-0.1"
    """
    parts = [base_name]
    
    for override in overrides:
        # Skip Hydra special overrides (start with +, ~, etc.) or group overrides
        if override.startswith(("+", "~", "++")) or "=" not in override:
            # Handle prefixed overrides by stripping the prefix
            if override.startswith("++"):
                override = override[2:]
            elif override.startswith(("+", "~")):
                override = override[1:]
            else:
                continue
        
        # Parse key=value
        if "=" not in override:
            continue
            
        key, value = override.split("=", 1)
        
        # Check if this key is in our mapping
        if key in run_name_mapping:
            short_name = run_name_mapping[key]
            parts.append(f"{short_name}_{value}")
    
    # If no mapped overrides were found, just return base_name
    if len(parts) == 1:
        return base_name
    
    run_name = "_".join(parts)
    return sanitize_wandb_run_name(run_name)


def build_model_artifact_name(group_name: str, run_name: str, step: Union[int, str]) -> str:
    """Build a W&B-safe model artifact name for checkpoints."""
    safe_group = sanitize_wandb_artifact_component(group_name, max_len=42)
    # Remove any prefix before group name using regex (e.g., "run_data_group_" -> "run_")
    run_name = re.sub(rf"_[^_]*_{re.escape(safe_group)}_", "_", run_name)
    safe_run = sanitize_wandb_artifact_component(run_name, max_len=42)
    safe_step = sanitize_wandb_artifact_component(str(step), max_len=64)
    return f"group_{safe_group}_model_{safe_run}_step_{safe_step}"


def build_model_artifact_prefix(group_name: str, run_name: str) -> str:
    """Prefix used by model checkpoint artifacts (ends with '_step_')."""
    safe_group = sanitize_wandb_artifact_component(group_name)
    safe_run = sanitize_wandb_artifact_component(run_name)
    return f"group_{safe_group}_model_{safe_run}_step_"


def log_config_artifact(saved_config_path: str) -> None:
    """Log config file as W&B artifact.
    
    Args:
        config_path: Original config file path
        saved_config_path: Path to saved config copy
        
    Raises:
        ValueError: If wandb run is not initialized
        FileNotFoundError: If config file doesn't exist
    """
    if wandb.run is None:
        raise ValueError("W&B run must be initialized before logging config")
    
    if not os.path.exists(saved_config_path):
        raise FileNotFoundError(f"Config file not found: {saved_config_path}")
    
    cfg_artifact = wandb.Artifact(
        name=f"config_{wandb.run.id}",
        type="config",
        metadata={
            "saved_config_path": os.path.abspath(saved_config_path),
        },
    )
    cfg_artifact.add_file(saved_config_path)
    wandb.log_artifact(cfg_artifact)
    print(f"✓ Logged config artifact: {saved_config_path}")




def log_dataset_results(
    dataset_name: str, 
    accuracy: float, 
    results: List[Dict], 
    log_prefix: str = ""
) -> None:
    """Log dataset evaluation results to W&B.
    
    Args:
        dataset_name: Name of the dataset
        accuracy: Accuracy metric for the dataset
        results: List of result dictionaries with prompt, response, etc.
        log_prefix: Optional prefix for logging keys
    """
    if wandb.run is None:
        return
    
    wandb.log({f"{log_prefix}{dataset_name}_accuracy": accuracy})
    
    table = wandb.Table(columns=["prompt", "response", "extracted", "target", "correct"])
    for r in results:
        table.add_data(
            r["prompt"],
            r["response"],
            r["extracted_answer"],
            r["high_reward_answer"],
            r["is_correct"],
        )
    wandb.log({f"{log_prefix}{dataset_name}_samples": table})


def log_evaluation_summary(
    all_metrics: Dict[str, Dict[str, float]], 
    log_prefix: str = ""
) -> None:
    """Log overall evaluation summary to W&B.
    
    Args:
        all_metrics: Dictionary of metrics by dataset
        log_prefix: Optional prefix for logging keys
    """
    if wandb.run is None:
        return
    
    overall_accuracy = all_metrics.get("overall", {}).get("accuracy", 0.0)
    wandb.log({f"{log_prefix}overall_accuracy": overall_accuracy})
    
    summary_table = wandb.Table(columns=["dataset", "accuracy", "correct", "total"])
    for dataset_name, metrics in all_metrics.items():
        if dataset_name != "overall":
            summary_table.add_data(
                dataset_name,
                metrics["accuracy"],
                metrics["correct"],
                metrics["total"],
            )
    wandb.log({f"{log_prefix}evaluation_summary": summary_table})


def log_checkpoint_artifact(
    checkpoint_path: str,
    step: Union[int, str],
    group_name: str,
    run_name: str,
    metadata: Dict[str, Any],
) -> None:
    """Log a checkpoint artifact to W&B.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        step: Training step number
        run_name: W&B run name
        metadata: Additional metadata
    """
    if wandb.run is None:
        raise ValueError("W&B run must be initialized before logging model artifact")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model path not found: {checkpoint_path}")
    
    artifact_name = build_model_artifact_name(
        group_name=group_name, run_name=run_name, step=step
    )
    if "step" not in metadata:
        metadata["step"] = step

    artifact = wandb.Artifact(name=artifact_name, type="model", metadata=metadata)
    artifact.add_dir(checkpoint_path)
    wandb.log_artifact(artifact)
    

def log_training_metrics(tracking_data: Dict[str, List]) -> None:
    """Log training tracking metrics to W&B.
    
    Args:
        tracking_data: Dictionary with tracking data lists
    """
    if wandb.run is None:
        return
    
    metrics = {}
    for key, values in tracking_data.items():
        if values:
            avg_key = f"avg_{key}"
            metrics[avg_key] = sum(values) / len(values)
    
    if metrics:
        wandb.log(metrics)
