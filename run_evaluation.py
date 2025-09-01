#!/usr/bin/env python3
"""
Script to run evaluation on a trained model checkpoint.
Example usage:
    python run_evaluation.py --checkpoint-path GRPO/checkpoint-125 --max-samples 100
"""

import subprocess
import sys
import os
import re

def find_latest_checkpoint(base_dir: str = "GRPO") -> str:
    if not os.path.isdir(base_dir):
        return ""
    candidates = []
    for name in os.listdir(base_dir):
        if name.startswith("checkpoint-"):
            m = re.match(r"checkpoint-(\d+)$", name)
            if m:
                candidates.append((int(m.group(1)), os.path.join(base_dir, name)))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

# Allow override via CLI: --checkpoint-path <path>
checkpoint_path = None
if "--checkpoint-path" in sys.argv:
    try:
        idx = sys.argv.index("--checkpoint-path")
        checkpoint_path = sys.argv[idx + 1]
    except Exception:
        pass

# If not provided, auto-detect latest
if not checkpoint_path:
    checkpoint_path = find_latest_checkpoint()

if not checkpoint_path or not os.path.exists(checkpoint_path):
    print("No checkpoint found to evaluate!")
    print("Please run train.py first to generate a model checkpoint.")
    sys.exit(1)

print(f"Found checkpoint at: {checkpoint_path}")
print("Running evaluation on the first 100 samples of each dataset...")
print("Using vLLM for fast inference")

# Run the evaluation (always using vLLM)
cmd = [
    sys.executable, "evaluations_vllm.py",
    "--checkpoint-path", checkpoint_path,
    "--max-samples", "100",
    "--batch-size", "32",
    "--wandb-project", "GRPO_Evaluation",
    "--save-results", "evaluation_results.json"
]

print(f"Running command: {' '.join(cmd)}")
subprocess.run(cmd)
