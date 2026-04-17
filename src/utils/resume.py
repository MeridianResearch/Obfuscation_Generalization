"""
Utilities for resuming training from wandb checkpoints.
"""

import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import wandb
from omegaconf import DictConfig, OmegaConf


def _parse_flag(flag: str) -> Optional[str]:
    """Pop `--flag VALUE` or `--flag=VALUE` from sys.argv and return VALUE."""
    value = None
    indices_to_remove: list[int] = []
    for i, arg in enumerate(sys.argv):
        if arg == flag:
            if i + 1 < len(sys.argv):
                value = sys.argv[i + 1]
                indices_to_remove = [i, i + 1]
                break
        elif arg.startswith(f"{flag}="):
            value = arg.split('=', 1)[1]
            indices_to_remove = [i]
            break
    for idx in sorted(indices_to_remove, reverse=True):
        sys.argv.pop(idx)
    return value


def parse_resume_arg() -> Optional[str]:
    """Parse --resume_checkpoint from sys.argv before Hydra processes args."""
    return _parse_flag('--resume_checkpoint')


def parse_resume_run_id_arg() -> Optional[str]:
    """Parse --resume_run_id from sys.argv before Hydra processes args.

    Use this to pin a specific W&B run id when multiple runs share the same
    displayName (e.g. an orphaned run from a failed resume attempt).
    """
    return _parse_flag('--resume_run_id')


def find_wandb_run(
    entity: str,
    project: str,
    group: str,
    run_name: str,
) -> Optional[Any]:
    """
    Find the most recent wandb run matching the given name and group.
    """
    api = wandb.Api()
    
    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters={
                "group": group,
                "displayName": run_name,
            }
        )
    except Exception as e:
        print(f"Warning: Failed to query wandb runs: {e}")
        return None
    
    runs_list = list(runs)
    if not runs_list:
        return None
    
    # Sort by creation time (most recent first)
    runs_list.sort(key=lambda r: r.created_at, reverse=True)
    return runs_list[0]


def verify_config_compatibility(
    run_config: Dict[str, Any],
    current_config: Union[Dict, DictConfig],
    ignore_keys: Optional[set] = None,
) -> Tuple[bool, list]:
    """
    Verify that the previous run's config matches the current config.
    Only compares keys present in current config (wandb logs extra derived values).
    """
    default_ignore = {
        'resume_run_id',
        'resume_step',
        'steps_remaining',
        'wandb.name',
        '_wandb',
    }
    ignore_keys = (ignore_keys or set()) | default_ignore
    
    if isinstance(current_config, DictConfig):
        current_config = OmegaConf.to_container(current_config, resolve=True)
    
    def flatten_dict(d: dict, parent_key: str = '') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat_run = flatten_dict(run_config)
    flat_current = flatten_dict(current_config)
    
    mismatches = []
    for key in sorted(flat_current.keys()):
        if any(key.startswith(ik) or key == ik for ik in ignore_keys):
            continue
        if key not in flat_run:
            continue
        if flat_run[key] != flat_current[key]:
            mismatches.append({
                'key': key,
                'previous': flat_run[key],
                'current': flat_current[key],
            })
    
    return len(mismatches) == 0, mismatches


def extract_step_from_artifact_name(name: str) -> Optional[int]:
    """Extract step number from artifact name using regex."""
    patterns = [
        r'step[_-](\d+)',
        r'checkpoint[_-](\d+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, name.lower())
        if match:
            return int(match.group(1))
    return None


def find_checkpoint_artifact(run: Any, step: Union[str, int]) -> Any:
    """
    Find a checkpoint artifact from a wandb run.
    
    Raises:
        ValueError: If no matching artifact is found
    """
    step_str = str(step)
    all_artifacts = list(run.logged_artifacts())
    
    if not all_artifacts:
        raise ValueError(
            f"No artifacts found in run {run.id} ({run.name}). "
            "Ensure checkpoints were uploaded to W&B."
        )
    
    artifact_names = [a.name for a in all_artifacts]
    
    # For specific step
    if step_str != 'latest':
        target_step = int(step_str)
        for artifact in all_artifacts:
            if extract_step_from_artifact_name(artifact.name) == target_step:
                return artifact
        raise ValueError(
            f"No checkpoint found for step {step_str} in run {run.id}. "
            f"Available: {artifact_names}"
        )
    
    # For 'latest': first try 'final', then highest step
    for artifact in all_artifacts:
        if 'final' in artifact.name.lower():
            return artifact
    
    max_step = -1
    best_artifact = None
    for artifact in all_artifacts:
        extracted = extract_step_from_artifact_name(artifact.name)
        if extracted is not None and extracted > max_step:
            max_step = extracted
            best_artifact = artifact
    
    if best_artifact is not None:
        print(f"No 'final' checkpoint found, using highest step: {max_step}")
        return best_artifact
    
    raise ValueError(
        f"Could not find any checkpoint in run {run.id}. "
        f"Available artifacts: {artifact_names}"
    )


def get_step_from_checkpoint(checkpoint_path: str) -> int:
    """Extract global_step from checkpoint's trainer_state.json."""
    state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, 'r') as f:
            state = json.load(f)
            return state.get("global_step", 0)
    
    # Fallback: parse from directory name
    dirname = os.path.basename(checkpoint_path.rstrip('/'))
    match = re.search(r'checkpoint[_-](\d+)', dirname)
    if match:
        return int(match.group(1))
    
    raise ValueError(f"Could not determine step from checkpoint: {checkpoint_path}")


def verify_checkpoint_files(checkpoint_path: str) -> None:
    """Print status of checkpoint completeness."""
    checks = {
        "trainer_state.json": "trainer state (required for step count)",
        "optimizer.pt": "optimizer state",
        "scheduler.pt": "scheduler state", 
        "rng_state.pth": "RNG state",
    }
    
    # Check for model files
    model_files = ["adapter_model.safetensors", "adapter_model.bin", 
                   "pytorch_model.bin", "model.safetensors"]
    has_model = any(os.path.exists(os.path.join(checkpoint_path, f)) for f in model_files)
    
    print("Checkpoint contents:")
    print(f"  {'✓' if has_model else '✗'} model weights")
    
    for filename, description in checks.items():
        exists = os.path.exists(os.path.join(checkpoint_path, filename))
        print(f"  {'✓' if exists else '✗'} {description}")
        
    if not os.path.exists(os.path.join(checkpoint_path, "optimizer.pt")):
        print("  ⚠ WARNING: No optimizer state - optimizer will reinitialize!")


@dataclass
class ResumeInfo:
    """Container for resume information."""
    checkpoint_path: str
    resume_step: int
    resume_run_id: str
    steps_remaining: Optional[int] = None
    
    def to_wandb_config(self) -> Dict[str, Any]:
        return {
            "resume_run_id": self.resume_run_id,
            "resume_step": self.resume_step,
            "steps_remaining": self.steps_remaining,
        }


def prepare_resume(
    resume_checkpoint: str,
    entity: str,
    project: str,
    group: str,
    run_name: str,
    current_config: Union[Dict, DictConfig],
    verify_config: bool = True,
    resume_run_id: Optional[str] = None,
) -> ResumeInfo:
    """
    Prepare for resuming training from a wandb checkpoint.

    If `resume_run_id` is provided, that specific W&B run is used directly
    (bypassing the displayName lookup). This is the escape hatch when multiple
    runs share a displayName (e.g. an orphan from a previously-failed resume).

    Raises:
        ValueError: If no matching run/checkpoint found or config mismatch
    """
    print(f"\n{'='*60}")
    print(f"RESUME: Preparing to resume from checkpoint '{resume_checkpoint}'")
    print(f"{'='*60}")

    # Find previous run
    if resume_run_id is not None:
        prev_run = wandb.Api().run(f"{entity}/{project}/{resume_run_id}")
        print(f"Using pinned run id: {prev_run.id} ({prev_run.name})")
    else:
        prev_run = find_wandb_run(entity, project, group, run_name)
        if prev_run is None:
            raise ValueError(
                f"No previous run found with name '{run_name}' in group '{group}' "
                f"(project: {entity}/{project})"
            )
        print(f"Found previous run: {prev_run.id} ({prev_run.name})")
    
    # Verify config
    if verify_config:
        is_compatible, mismatches = verify_config_compatibility(
            prev_run.config, current_config
        )
        if not is_compatible:
            mismatch_str = "\n".join(
                f"  - {m['key']}: {m['previous']} -> {m['current']}"
                for m in mismatches
            )
            raise ValueError(
                f"Config mismatch with previous run:\n{mismatch_str}\n"
                "Use identical config or disable verification."
            )
        print("✓ Config verification passed")
    
    # Find and download artifact
    artifact = find_checkpoint_artifact(prev_run, resume_checkpoint)
    print(f"Downloading artifact: {artifact.name}")
    
    download_dir = tempfile.mkdtemp(prefix="wandb_checkpoint_")
    checkpoint_path = artifact.download(root=download_dir)
    print(f"Downloaded to: {checkpoint_path}")
    
    # Verify checkpoint contents
    verify_checkpoint_files(checkpoint_path)
    
    # Get step
    resume_step = get_step_from_checkpoint(checkpoint_path)
    print(f"Resume step: {resume_step}")
    print(f"{'='*60}\n")
    
    return ResumeInfo(
        checkpoint_path=checkpoint_path,
        resume_step=resume_step,
        resume_run_id=prev_run.id,
    )