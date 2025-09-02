import os
import json
import shutil
import datetime
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_yaml_file(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required but not installed. pip install pyyaml")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def create_run_dir(base_results_dir: str, prefix: str) -> str:
    ensure_dir(base_results_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_results_dir, f"{prefix}_{timestamp}")
    ensure_dir(run_dir)
    return run_dir


def save_config_copy(config_path: str, dst_dir: str) -> str:
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(config_path))
    shutil.copy2(config_path, dst)
    return dst


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


