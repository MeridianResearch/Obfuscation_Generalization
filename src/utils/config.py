"""
Configuration utilities.

Note: Config loading is now handled by Hydra. The load_config_with_defaults function
has been removed. Use Hydra's config composition instead:

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        ...
"""

import os
import json
import datetime
from typing import Any, Dict


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_timestamped_parent_dir(base_results_dir: str, prefix: str) -> str:
    """Create a parent directory with timestamp at the end."""
    ensure_dir(base_results_dir)

    # Check for existing directories with the same prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = os.path.join(base_results_dir, f"{prefix}_{timestamp}")
    ensure_dir(parent_dir)
    return parent_dir


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Save object as JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
