"""
Sweep utilities for zip-style parameter sweeps with Hydra.

This module provides a custom OmegaConf resolver `${sz:field}` that enables
zip-style sweeps (as opposed to Hydra's default grid sweeps).

Usage in configs:
    sweep:
      learning_rate: [0.001, 0.01, 0.1]
      penalty_weight: [-0.1, -0.2, -0.3]
      seed: [42, 42, 42]
    
    sweep_index: ???  # Set via CLI: sweep_index=0,1,2 --multirun
    
    train:
      learning_rate: ${sz:learning_rate}
      seed: ${sz:seed}

How it works:
    1. Define parallel lists in `sweep:` section
    2. Use `${sz:field}` to reference swept values  
    3. Run with `--multirun sweep_index=0,1,2,...` to launch all jobs
    4. Each job gets a different sweep_index, resolving to different values
"""

from typing import Any, List, Optional

from omegaconf import DictConfig, ListConfig, OmegaConf, MissingMandatoryValue


# Global sweep index - set by the launcher before config resolution
_SWEEP_INDEX: Optional[int] = None


def set_sweep_index(index: int) -> None:
    """Set the current sweep index. Called by the launcher."""
    global _SWEEP_INDEX
    _SWEEP_INDEX = index


def get_sweep_index() -> int:
    """Get the current sweep index."""
    if _SWEEP_INDEX is None:
        raise RuntimeError(
            "Sweep index not set. Either:\n"
            "  1. Run with: python -m src.train_sweep sweep_index=N\n"
            "  2. For sweeps: python -m src.train_sweep --multirun sweep_index=0,1,2,..."
        )
    return _SWEEP_INDEX


def _sweep_zip_resolver(field: str, _parent_: Any = None, _root_: Any = None) -> Any:
    """
    OmegaConf resolver for ${sz:field}.
    
    Looks up sweep.<field>[sweep_index] and returns the value.
    
    Args:
        field: The field name to look up in the sweep config
        _parent_: OmegaConf parent (injected automatically)
        _root_: OmegaConf root config (injected automatically)
    
    Returns:
        The value at sweep.<field>[sweep_index]
    """
    # Try to get index from global first, then from config
    try:
        idx = get_sweep_index()
    except RuntimeError:
        # Fall back to config's sweep_index if global not set
        if _root_ is not None and "sweep_index" in _root_:
            idx = _root_.sweep_index
            if isinstance(idx, str) and idx == "???":
                raise RuntimeError(
                    "sweep_index is not set. Run with sweep_index=N or use --multirun"
                )
        else:
            raise
    
    if _root_ is None:
        raise RuntimeError("Cannot resolve ${sz:...} - no root config available")
    
    sweep_cfg = _root_.get("sweep")
    if sweep_cfg is None:
        raise KeyError(
            f"Cannot resolve ${{sz:{field}}} - no 'sweep' section in config. "
            "Add a 'sweep' section with parameter lists."
        )
    
    if field not in sweep_cfg:
        available = list(sweep_cfg.keys())
        raise KeyError(
            f"Cannot resolve ${{sz:{field}}} - field '{field}' not found in sweep config. "
            f"Available fields: {available}"
        )
    
    values = sweep_cfg[field]
    if not isinstance(values, (list, ListConfig)):
        # Single value - return as-is (not a sweep on this param)
        return values
    
    if idx >= len(values):
        raise IndexError(
            f"Sweep index {idx} out of bounds for sweep.{field} (length {len(values)})"
        )
    
    return values[idx]


def register_sweep_resolvers() -> None:
    """
    Register custom OmegaConf resolvers for sweep functionality.
    
    Call this once at startup, before loading any configs.
    MUST be called before Hydra's @hydra.main() decorator runs.
    
    Registers:
        - ${sz:field}: Zip-style sweep lookup
        - ${sweep_len:}: Returns the number of jobs in the sweep
    """
    # Avoid re-registration errors
    try:
        OmegaConf.register_new_resolver("sz", _sweep_zip_resolver)
    except ValueError:
        pass  # Already registered
    
    def _sweep_len_resolver(_root_: Any = None) -> int:
        """Return number of jobs in sweep."""
        if _root_ is None or "sweep" not in _root_:
            return 0
        return get_sweep_count_from_dict(_root_.sweep)
    
    try:
        OmegaConf.register_new_resolver("sweep_len", _sweep_len_resolver)
    except ValueError:
        pass  # Already registered


def get_sweep_count_from_dict(sweep_cfg: Any) -> int:
    """Get sweep count from a sweep config dict/DictConfig."""
    lengths = {}
    for key, value in sweep_cfg.items():
        if isinstance(value, (list, ListConfig)):
            lengths[key] = len(value)
    
    if not lengths:
        return 0
    
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        length_details = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(
            f"Sweep lists have inconsistent lengths: {length_details}. "
            "All sweep lists must have the same length for zip-style sweeps."
        )
    
    return unique_lengths.pop()


def get_sweep_count(cfg: DictConfig) -> int:
    """
    Get the number of jobs in a sweep config.
    
    Validates that all sweep lists have the same length.
    
    Args:
        cfg: The full config with a 'sweep' section
    
    Returns:
        Number of sweep jobs
    
    Raises:
        ValueError: If sweep lists have inconsistent lengths
        KeyError: If no sweep section exists
    """
    sweep_cfg = cfg.get("sweep")
    if sweep_cfg is None:
        raise KeyError("No 'sweep' section in config")
    
    return get_sweep_count_from_dict(sweep_cfg)


def get_sweep_job_name(cfg: DictConfig, index: int) -> str:
    """
    Get the resolved config_name for a specific sweep index.
    
    Temporarily sets the sweep index, resolves config_name, then restores.
    
    Args:
        cfg: The full config (unresolved)
        index: The sweep index to resolve for
    
    Returns:
        The resolved config_name string
    """
    global _SWEEP_INDEX
    old_index = _SWEEP_INDEX
    try:
        set_sweep_index(index)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        return resolved.get("config_name", f"job_{index}")
    finally:
        _SWEEP_INDEX = old_index

def validate_sweep_config(cfg: DictConfig) -> None:
    """
    Validate a sweep config before launching.
    
    Checks:
        - sweep section exists
        - All sweep lists have same length
        - config_name can be resolved for all indices
        - No duplicate config_names
    
    Args:
        cfg: The full config to validate
    
    Raises:
        ValueError: If validation fails
    """
    n_jobs = get_sweep_count(cfg)
    
    # Try resolving config_name for each index
    names = []
    for i in range(n_jobs):
        try:
            name = get_sweep_job_name(cfg, i)
            names.append(name)
        except Exception as e:
            raise ValueError(f"Failed to resolve config_name for sweep index {i}: {e}")
    
    # Check for duplicate names
    if len(names) != len(set(names)):
        from collections import Counter
        counts = Counter(names)
        duplicates = [name for name, count in counts.items() if count > 1]
        raise ValueError(
            f"Duplicate config_names in sweep: {duplicates}. "
            "Each sweep job must have a unique config_name. "
            "Consider adding 'seed' to your config_name template."
        )
    
    return names


def print_sweep_jobs(cfg: DictConfig) -> None:
    """Print all jobs in a sweep config."""
    names = validate_sweep_config(cfg)
    n_jobs = len(names)
    
    print(f"Sweep has {n_jobs} jobs:")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")
    print()
    print(f"Run all: --multirun sweep_index={','.join(map(str, range(n_jobs)))}")