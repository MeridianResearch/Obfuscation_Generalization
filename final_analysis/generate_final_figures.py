#!/usr/bin/env python3
"""
Generate final paper figures by combining specific lines from different configs.

Usage:
    python generate_final_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
METRICS_DIR = BASE_DIR / "metrics"
OUTPUT_DIR = BASE_DIR / "final_figures"
OUTPUT_PDFS = OUTPUT_DIR / "pdfs"
OUTPUT_PNGS = OUTPUT_DIR / "pngs"
OUTPUT_TXTS = OUTPUT_DIR / "txts"

# =============================================================================
# STYLE CONFIG
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.weight": "normal",
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "figure.titleweight": "normal",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 16,
})

# Default colors
COLOR_CORRECT = "#FF9999"       # Pastel red for reward hacking rate
COLOR_MONITOR = "#99CCFF"       # Pastel blue for CoT detection rate
COLOR_SUMMARY = "#CC99FF"       # Pastel purple for summary detection rate
COLOR_CHANCE = "#CCCCCC"        # Gray for chance level
COLOR_NON_EXTRACTABLE = "#99CC99"  # Pastel green for unparsable rate

# Line styles for different seeds in appendix figures
SEED_STYLES = {
    24: {"linestyle": "-"},
    33: {"linestyle": "--"},
    42: {"linestyle": "-."},
    50: {"linestyle": ":"},
}
SEED_COLORS = {
    24: {"correct": "#FF9999", "monitor": "#99CCFF"},
    33: {"correct": "#FF8080", "monitor": "#80BFFF"},
    42: {"correct": "#FF6666", "monitor": "#66B2FF"},
    50: {"correct": "#CC7777", "monitor": "#7799DD"},
}

# =============================================================================
# FIGURE SPECIFICATIONS
# =============================================================================

# Combined main figures: 2 rows x 4 columns
# Top row: evaluated on leave-out set, Bottom row: evaluated on medical sycophancy

COMBINED_FIGURES = {
    "figure1": {
        "name": "figure1_8b",
        "skip_threshold": 0.3,
        "show_summary": False,
        "columns": [
            {
                "top_title": "Leave out\nFactual sycophancy",
                "top": {
                    "chance_level": 0.5,
                    "config": "sycophancy",
                    "seed": 42,
                    "eval_fold": "eval_sycophancy_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "sycophancy",
                    "seed": 42,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
            {
                "top_title": "Leave out\nCode selection",
                "top": {
                    "chance_level": 0.5,
                    "config": "code",
                    "seed": 33,
                    "eval_fold": "eval_code_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "code",
                    "seed": 33,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
            {
                "top_title": "Leave out\nRevealed score MMLU",
                "top": {
                    "chance_level": 0.25,
                    "config": "score",
                    "seed": 24,
                    "eval_fold": "eval_revealing_score_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "score",
                    "seed": 24,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
            {
                "top_title": "Leave out\nWorld affecting reward",
                "top": {
                    "chance_level": 0.5,
                    "config": "war",
                    "seed": 50,
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "war",
                    "seed": 50,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
        ],
    },
    "figure1_raw": {
        "name": "figure1_raw_8b",
        "skip_threshold": 0.3,
        "show_summary": False,
        "columns": [
            {
                "top_title": "Leave out\nFactual sycophancy",
                "top": {
                    "chance_level": 0.5,
                    "config": "sycophancy_raw",
                    "seed": 42,
                    "eval_fold": "eval_sycophancy_raw",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "sycophancy_raw",
                    "seed": 42,
                    "eval_fold": "eval_medical_sycophancy_fact_raw",
                },
            },
            {
                "top_title": "Leave out\nCode selection",
                "top": {
                    "chance_level": 0.5,
                    "config": "code_raw",
                    "seed": 33,
                    "eval_fold": "eval_code_raw",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "code_raw",
                    "seed": 33,
                    "eval_fold": "eval_medical_sycophancy_fact_raw",
                },
            },
            {
                "top_title": "Leave out\nRevealed score MMLU",
                "top": {
                    "chance_level": 0.25,
                    "config": "score_raw",
                    "seed": 24,
                    "eval_fold": "eval_revealing_score_raw",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "score_raw",
                    "seed": 24,
                    "eval_fold": "eval_medical_sycophancy_fact_raw",
                },
            },
            {
                "top_title": "Leave out\nWorld affecting reward",
                "top": {
                    "chance_level": 0.5,
                    "config": "war_raw",
                    "seed": 50,
                    "eval_fold": "eval_world_affecting_reward_reorg_raw",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "war_raw",
                    "seed": 50,
                    "eval_fold": "eval_medical_sycophancy_fact_raw",
                },
            },
        ],
    },
    "figure2": {
        "name": "figure2_8b",
        "skip_threshold": 0.3,
        "show_summary": True,
        "columns": [
            {
                "top_title": "Leave out\nFactual sycophancy",
                "top": {
                    "chance_level": 0.5,
                    "config": "sum_sycophancy",
                    "seed": 33,
                    "eval_fold": "eval_sycophancy_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "sum_sycophancy",
                    "seed": 33,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
            {
                "top_title": "Leave out\nCode selection",
                "top": {
                    "chance_level": 0.5,
                    "config": "sum_code",
                    "seed": 33,
                    "eval_fold": "eval_code_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "sum_code",
                    "seed": 33,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
            {
                "top_title": "Leave out\nRevealed score MMLU",
                "top": {
                    "chance_level": 0.25,
                    "config": "sum_score",
                    "seed": 33,
                    "eval_fold": "eval_revealing_score_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "sum_score",
                    "seed": 33,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
            {
                "top_title": "Leave out\nWorld affecting reward",
                "top": {
                    "chance_level": 0.5,
                    "config": "sum_war",
                    "seed": 24,
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                },
                "bottom": {
                    "chance_level": 0.5,
                    "config": "sum_war",
                    "seed": 24,
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                },
            },
        ],
    },
}

# Appendix figures - multi-seed grid layouts
APPENDIX_FIGURES = {
    "appendix_figure0": {
        "name": "appendix_figure0_8b",
        "title": "No penalisation",
        "skip_threshold": 0.3,
        "show_summary": True,
        "rows": [
            {
                "row_title": "Leave out\nFactual sycophancy",
                "left": {
                    "chance_level": 0.5,
                    "config": "base_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nCode selection",
                "left": {
                    "chance_level": 0.5,
                    "config": "base_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nRevealed score MMLU",
                "left": {
                    "chance_level": 0.25,
                    "config": "base_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nWorld affecting reward",
                "left": {
                    "chance_level": 0.5,
                    "config": "base_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "base_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure0b": {
        "name": "appendix_figure0b_8b",
        "title": "No penalisation - Unparsable rate",
        "skip_threshold": None,
        "show_summary": True,
        "plot_unparsable": True,
        "rows": [
            {
                "row_title": "Leave out\nFactual sycophancy",
                "left": {
                    "config": "base_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nCode selection",
                "left": {
                    "config": "base_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nRevealed score MMLU",
                "left": {
                    "config": "base_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nWorld affecting reward",
                "left": {
                    "config": "base_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "base_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure1": {
        "name": "appendix_figure1_8b",
        "title": "CoT penalisation",
        "skip_threshold": 0.3,
        "show_summary": False,
        "rows": [
            {
                "row_title": "Leave out\nFactual sycophancy",
                "left": {
                    "chance_level": 0.5,
                    "config": "sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nCode selection",
                "left": {
                    "chance_level": 0.5,
                    "config": "code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nRevealed score MMLU",
                "left": {
                    "chance_level": 0.25,
                    "config": "score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nWorld affecting reward",
                "left": {
                    "chance_level": 0.5,
                    "config": "war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure1b": {
        "name": "appendix_figure1b_8b",
        "title": "CoT penalisation - Unparsable rate",
        "skip_threshold": None,
        "show_summary": False,
        "plot_unparsable": True,
        "rows": [
            {
                "row_title": "Leave out\nFactual sycophancy",
                "left": {
                    "config": "sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nCode selection",
                "left": {
                    "config": "code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nRevealed score MMLU",
                "left": {
                    "config": "score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nWorld affecting reward",
                "left": {
                    "config": "war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure2": {
        "name": "appendix_figure2_8b",
        "title": "Summary penalisation",
        "skip_threshold": 0.3,
        "show_summary": True,
        "rows": [
            {
                "row_title": "Leave out\nFactual sycophancy",
                "left": {
                    "chance_level": 0.5,
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nCode selection",
                "left": {
                    "chance_level": 0.5,
                    "config": "sum_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nRevealed score MMLU",
                "left": {
                    "chance_level": 0.25,
                    "config": "sum_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nWorld affecting reward",
                "left": {
                    "chance_level": 0.5,
                    "config": "sum_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "chance_level": 0.5,
                    "config": "sum_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
    "appendix_figure2b": {
        "name": "appendix_figure2b_8b",
        "title": "Summary penalisation - Unparsable rate",
        "skip_threshold": None,
        "show_summary": True,
        "plot_unparsable": True,
        "rows": [
            {
                "row_title": "Leave out\nFactual sycophancy",
                "left": {
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_sycophancy_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_sycophancy",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nCode selection",
                "left": {
                    "config": "sum_code",
                    "eval_fold": "eval_code_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_code",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nRevealed score MMLU",
                "left": {
                    "config": "sum_score",
                    "eval_fold": "eval_revealing_score_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_score",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
            {
                "row_title": "Leave out\nWorld affecting reward",
                "left": {
                    "config": "sum_war",
                    "eval_fold": "eval_world_affecting_reward_reorg_formatted",
                    "seeds": [24, 33, 42, 50],
                },
                "right": {
                    "config": "sum_war",
                    "eval_fold": "eval_medical_sycophancy_fact_formatted",
                    "seeds": [24, 33, 42, 50],
                },
            },
        ],
    },
}

# Standalone example figures (single panel each)
EXAMPLE_FIGURES = {
    "example_sycophancy_sum_s24_main": {
        "name": "example_sycophancy_sum_s24_main_8b",
        "title": "Evaluated on unseen sycophancy dataset",
        "skip_threshold": 0.3,
        "show_summary": False,
        "legend_horizontal": True,
        "spec": {
            "chance_level": 0.5,
            "config": "sum_sycophancy",
            "seed": 24,
            "eval_fold": "eval_sycophancy_formatted",
        },
    },
    "example_sycophancy_sum_s24_appendix": {
        "name": "example_sycophancy_sum_s24_appendix_8b",
        "title": "Leave out factual sycophancy\nPenalisation applied to summary",
        "skip_threshold": 0.3,
        "show_summary": True,
        "layout": "pair",
        "left_title": "Eval on sycophancy",
        "right_title": "Eval on medical sycophancy",
        "spec_left": {
            "chance_level": 0.5,
            "config": "sum_sycophancy",
            "seed": 24,
            "eval_fold": "eval_sycophancy_formatted",
        },
        "spec_right": {
            "chance_level": 0.5,
            "config": "sum_sycophancy",
            "seed": 24,
            "eval_fold": "eval_medical_sycophancy_fact_formatted",
        },
    },
    "example_score_s42": {
        "name": "example_score_s42_8b",
        "title": "Leave out revealed score MMLU\nCoT penalisation",
        "skip_threshold": 0.3,
        "show_summary": False,
        "spec": {
            "chance_level": 0.25,
            "config": "score",
            "seed": 42,
            "eval_fold": "eval_revealing_score_formatted",
        },
    },
    "example_score_sum_s50_medical": {
        "name": "example_score_sum_s50_medical_8b",
        "title": "Leave out revealed score MMLU\nEval on medical sycophancy\nPenalisation applied to summary",
        "skip_threshold": 0.3,
        "show_summary": True,
        "spec": {
            "chance_level": 0.5,
            "config": "sum_score",
            "seed": 50,
            "eval_fold": "eval_medical_sycophancy_fact_formatted",
        },
    },
    "example_war_sum_s24": {
        "name": "example_war_sum_s24_8b",
        "title": "Leave out world affecting reward\nPenalisation applied to summary",
        "skip_threshold": 0.3,
        "show_summary": True,
        "spec": {
            "chance_level": 0.5,
            "config": "sum_war",
            "seed": 24,
            "eval_fold": "eval_world_affecting_reward_reorg_formatted",
        },
    },
}


# =============================================================================
# RUN TRACKING
# =============================================================================

class RunTracker:
    """Track which runs are used in each figure with full wandb details."""
    
    def __init__(self):
        self.runs: List[Dict] = []  # List of run info dicts
    
    def add(self, run_info: Dict):
        """Record a run being used with full details."""
        # Check if this exact run is already tracked
        key = (run_info.get("config"), run_info.get("seed"), 
               run_info.get("eval_fold"), run_info.get("step"))
        existing_keys = [
            (r.get("config"), r.get("seed"), r.get("eval_fold"), r.get("step"))
            for r in self.runs
        ]
        if key not in existing_keys:
            self.runs.append(run_info)
    
    def save(self, filepath: Path):
        """Save runs to a text file with full wandb details."""
        with open(filepath, "w") as f:
            f.write("# Runs used in this figure\n")
            f.write("# Full wandb run details with all checkpoints\n")
            f.write("#" + "=" * 60 + "\n\n")
            
            # Group by (config, seed, eval_fold) -> list of checkpoints
            grouped: Dict[Tuple[str, int, str], List[Dict]] = {}
            for run_info in self.runs:
                key = (run_info.get("config", ""), 
                       run_info.get("seed", 0), 
                       run_info.get("eval_fold", ""))
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(run_info)
            
            for (config, seed, eval_fold), checkpoints in sorted(grouped.items()):
                f.write(f"[{config}] seed={seed}, eval_fold={eval_fold}\n")
                
                # Get unique run names/ids from checkpoints
                run_names = set()
                run_ids = set()
                for cp in checkpoints:
                    if cp.get("run_name"):
                        run_names.add(cp["run_name"])
                    if cp.get("run_id"):
                        run_ids.add(cp["run_id"])
                
                if run_names:
                    f.write(f"  run_name(s): {', '.join(sorted(run_names))}\n")
                if run_ids:
                    f.write(f"  run_id(s): {', '.join(sorted(run_ids))}\n")
                
                # List all checkpoints/steps
                steps = sorted(set(cp.get("step", 0) for cp in checkpoints))
                f.write(f"  checkpoints ({len(steps)}): {steps}\n")
                f.write("\n")
    
    def __len__(self):
        return len(self.runs)


# =============================================================================
# DATA LOADING
# =============================================================================

_data_cache: Dict[str, pd.DataFrame] = {}


def load_config_data(config: str) -> pd.DataFrame:
    """Load and cache data for a config."""
    if config not in _data_cache:
        csv_path = METRICS_DIR / config / "trial_metrics.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Config data not found: {csv_path}")
        _data_cache[config] = pd.read_csv(csv_path)
    return _data_cache[config]


def get_line_data(
    config: str,
    seed: int,
    eval_fold: str,
    skip_threshold: Optional[float] = 0.3,
    tracker: Optional[RunTracker] = None,
) -> pd.DataFrame:
    """Extract and prepare data for a single line.
    
    Note: Smoothing is applied AFTER filtering by skip_threshold, so the rolling
    window operates in step-space (only on points that are actually plotted).
    """
    df = load_config_data(config)
    
    # Filter to seed and eval_fold
    df_line = df[(df["seed"] == seed) & (df["eval_fold"] == eval_fold)].copy()
    
    if df_line.empty:
        print(f"  WARNING: No data for config={config}, seed={seed}, eval_fold={eval_fold}")
        return df_line
    
    # Track runs with full details before aggregation
    if tracker is not None:
        for _, row in df_line.iterrows():
            run_info = {
                "config": config,
                "seed": seed,
                "eval_fold": eval_fold,
                "step": row.get("step"),
                "run_name": row.get("run_name"),
                "run_id": row.get("run_id"),
            }
            tracker.add(run_info)
    
    # Detect column names
    correct_col = "correct_rate_extractable" if "correct_rate_extractable" in df_line.columns else "reward_hack_rate_extractable"
    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_line.columns else "no_answer_tags_rate"
    
    # Build aggregation dict with available columns
    agg_cols = {
        correct_col: "mean",
        no_answer_col: "mean",
    }
    
    # Add monitor columns if available
    if "monitor_flag_rate_extractable" in df_line.columns:
        agg_cols["monitor_flag_rate_extractable"] = "mean"
    if "summary_monitor_flag_rate_extractable" in df_line.columns:
        agg_cols["summary_monitor_flag_rate_extractable"] = "mean"
    
    df_line = df_line.groupby("step", as_index=False).agg(agg_cols)
    
    # Apply skip threshold BEFORE smoothing (smoothing happens in step-space)
    if skip_threshold is not None:
        mask = df_line[no_answer_col] <= skip_threshold
        df_line = df_line[mask]
    
    # Rename for consistency
    df_line = df_line.rename(columns={correct_col: "correct_rate", no_answer_col: "no_answer_rate"})
    
    return df_line.sort_values("step")


def get_unparsable_data(
    config: str,
    seed: int,
    eval_fold: str,
    tracker: Optional[RunTracker] = None,
) -> pd.DataFrame:
    """Extract unparsable rate data for a single line (no filtering, no smoothing)."""
    df = load_config_data(config)
    
    # Filter to seed and eval_fold
    df_line = df[(df["seed"] == seed) & (df["eval_fold"] == eval_fold)].copy()
    
    if df_line.empty:
        print(f"  WARNING: No data for config={config}, seed={seed}, eval_fold={eval_fold}")
        return df_line
    
    # Track runs with full details before aggregation
    if tracker is not None:
        for _, row in df_line.iterrows():
            run_info = {
                "config": config,
                "seed": seed,
                "eval_fold": eval_fold,
                "step": row.get("step"),
                "run_name": row.get("run_name"),
                "run_id": row.get("run_id"),
            }
            tracker.add(run_info)
    
    # Detect column names
    no_answer_col = "no_answer_rate" if "no_answer_rate" in df_line.columns else "no_answer_tags_rate"
    
    # Aggregate duplicate steps
    agg_cols = {no_answer_col: "mean"}
    df_line = df_line.groupby("step", as_index=False).agg(agg_cols)
    
    # Rename for consistency
    df_line = df_line.rename(columns={no_answer_col: "no_answer_rate"})
    
    return df_line.sort_values("step")


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def style_axis(ax, ylim=(0, 1)):
    """Apply consistent styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if ylim:
        ax.set_ylim(ylim)


def create_block_legend(fig, show_summary=False, fontsize=16):
    """Create a legend with colored blocks instead of lines."""
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor='none', label='Reward hacking rate'),
        mpatches.Patch(facecolor=COLOR_MONITOR, edgecolor='none', label='CoT detection rate'),
    ]
    if show_summary:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_SUMMARY, edgecolor='none', label='Summary detection rate')
        )
    
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        handleheight=1.5,
    )
    return legend


def create_vertical_legend(fig, show_summary=False, fontsize=14):
    """Create a vertical legend with colored blocks below the plot (for example figures)."""
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor='none', label='Reward hacking rate'),
        mpatches.Patch(facecolor=COLOR_MONITOR, edgecolor='none', label='CoT detection rate'),
    ]
    if show_summary:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_SUMMARY, edgecolor='none', label='Summary detection rate')
        )
    
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.2),
        ncol=1,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        handleheight=1.5,
    )
    return legend


def create_horizontal_legend(fig, show_summary=False, fontsize=14):
    """Create a horizontal legend with colored blocks below the plot (for example figures)."""
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor='none', label='Reward hacking rate'),
        mpatches.Patch(facecolor=COLOR_MONITOR, edgecolor='none', label='CoT detection rate'),
    ]
    if show_summary:
        legend_elements.append(
            mpatches.Patch(facecolor=COLOR_SUMMARY, edgecolor='none', label='Summary detection rate')
        )
    
    legend = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=len(legend_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        handleheight=1.5,
    )
    return legend


def create_appendix_legend(fig, seeds, show_summary=False, fontsize=14):
    """Create a two-row legend for appendix figures with colors and seed linestyles."""
    # Row 1: metric colors (as patches)
    color_elements = [
        mpatches.Patch(facecolor=COLOR_CORRECT, edgecolor='none', label='Reward hacking rate'),
        mpatches.Patch(facecolor=COLOR_MONITOR, edgecolor='none', label='CoT detection rate'),
    ]
    if show_summary:
        color_elements.append(
            mpatches.Patch(facecolor=COLOR_SUMMARY, edgecolor='none', label='Summary detection rate')
        )
    
    # Row 2: seed linestyles (as lines)
    seed_elements = []
    for seed in sorted(seeds):
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        seed_elements.append(
            Line2D([0], [0], color='gray', linewidth=2, linestyle=style["linestyle"], label=f'Seed {seed}')
        )
    
    # Create two separate legends stacked vertically
    legend1 = fig.legend(
        handles=color_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(color_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        handleheight=1.5,
    )
    
    legend2 = fig.legend(
        handles=seed_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(seed_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handleheight=1.5,
    )
    
    # Add first legend back (it gets removed when adding second)
    fig.add_artist(legend1)
    
    return legend1, legend2


def create_unparsable_legend(fig, seeds, fontsize=14):
    """Create a two-row legend for unparsable rate figures."""
    # Row 1: metric color
    color_elements = [
        mpatches.Patch(facecolor=COLOR_NON_EXTRACTABLE, edgecolor='none', label='Unparsable rate')
    ]
    
    # Row 2: seed linestyles
    seed_elements = []
    for seed in sorted(seeds):
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        seed_elements.append(
            Line2D([0], [0], color='gray', linewidth=2, linestyle=style["linestyle"], label=f'Seed {seed}')
        )
    
    # Create two separate legends stacked vertically
    legend1 = fig.legend(
        handles=color_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=len(color_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=1.5,
        handleheight=1.5,
    )
    
    legend2 = fig.legend(
        handles=seed_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(seed_elements),
        frameon=False,
        fontsize=fontsize,
        handlelength=2.0,
        handleheight=1.5,
    )
    
    # Add first legend back
    fig.add_artist(legend1)
    
    return legend1, legend2


# =============================================================================
# SUBPLOT PLOTTING
# =============================================================================

def plot_combined_subplot(
    ax,
    subplot_spec: dict,
    skip_threshold: float,
    show_summary: bool = False,
    tracker: Optional[RunTracker] = None,
):
    """Plot a single subplot for combined main figures."""
    if subplot_spec is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="gray")
        style_axis(ax)
        return
    
    df_line = get_line_data(
        config=subplot_spec["config"],
        seed=subplot_spec["seed"],
        eval_fold=subplot_spec["eval_fold"],
        skip_threshold=skip_threshold,
        tracker=tracker,
    )
    
    if df_line.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=16, color="gray")
        style_axis(ax)
        return
    
    color_correct = subplot_spec.get("color_correct", COLOR_CORRECT)
    color_monitor = subplot_spec.get("color_monitor", COLOR_MONITOR)
    
    # Compute rolling averages for correct rate
    correct_raw = df_line["correct_rate"]
    correct_smooth = correct_raw.rolling(window=3, min_periods=1, center=True).mean()
    
    # Plot correct rate (reward hacking) - faint markers, solid line
    ax.plot(
        df_line["step"],
        np.ma.masked_invalid(correct_raw),
        color=color_correct,
        linewidth=0,
        marker="o",
        markersize=5,
        alpha=0.3,
    )
    ax.plot(
        df_line["step"],
        np.ma.masked_invalid(correct_smooth),
        color=color_correct,
        linewidth=2.5,
        linestyle="-",
        alpha=1.0,
    )
    
    # Plot monitor flag rate (CoT penalty) if available
    if "monitor_flag_rate_extractable" in df_line.columns:
        monitor_raw = df_line["monitor_flag_rate_extractable"]
        if monitor_raw.notna().any():
            monitor_smooth = monitor_raw.rolling(window=3, min_periods=1, center=True).mean()
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(monitor_raw),
                color=color_monitor,
                linewidth=0,
                marker="o",
                markersize=5,
                alpha=0.3,
            )
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(monitor_smooth),
                color=color_monitor,
                linewidth=2.5,
                linestyle="-",
                alpha=1.0,
            )
    
    # Plot summary monitor rate if enabled and available
    if show_summary and "summary_monitor_flag_rate_extractable" in df_line.columns:
        summary_raw = df_line["summary_monitor_flag_rate_extractable"]
        if summary_raw.notna().any():
            summary_smooth = summary_raw.rolling(window=3, min_periods=1, center=True).mean()
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(summary_raw),
                color=COLOR_SUMMARY,
                linewidth=0,
                marker="o",
                markersize=5,
                alpha=0.3,
            )
            ax.plot(
                df_line["step"],
                np.ma.masked_invalid(summary_smooth),
                color=COLOR_SUMMARY,
                linewidth=2.5,
                linestyle="-",
                alpha=1.0,
            )
    
    chance_level = subplot_spec.get("chance_level", 0.5)
    ax.axhline(y=chance_level, color=COLOR_CHANCE, linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    ax.tick_params(axis='both', labelsize=14)
    style_axis(ax)


def plot_appendix_subplot(
    ax,
    subplot_spec: dict,
    skip_threshold: float,
    show_summary: bool = False,
    tracker: Optional[RunTracker] = None,
):
    """Plot a single appendix subplot with multiple seeds (no markers)."""
    if subplot_spec is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xlim(0, 4000)
        style_axis(ax)
        return
    
    config = subplot_spec["config"]
    eval_fold = subplot_spec["eval_fold"]
    seeds = subplot_spec["seeds"]
    
    for seed in seeds:
        df_line = get_line_data(
            config=config,
            seed=seed,
            eval_fold=eval_fold,
            skip_threshold=skip_threshold,
            tracker=tracker,
        )
        
        if df_line.empty:
            continue
        
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        colors = SEED_COLORS.get(seed, {"correct": COLOR_CORRECT, "monitor": COLOR_MONITOR})
        
        # Compute rolling averages for correct rate
        correct_raw = df_line["correct_rate"]
        correct_smooth = correct_raw.rolling(window=3, min_periods=1, center=True).mean()
        
        # Plot correct rate (reward hacking) - no markers, styled line
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(correct_smooth),
            color=colors["correct"],
            linewidth=2.5,
            linestyle=style["linestyle"],
            alpha=1.0,
        )
        
        # Plot monitor flag rate (CoT penalty) if available - no markers, styled line
        if "monitor_flag_rate_extractable" in df_line.columns:
            monitor_raw = df_line["monitor_flag_rate_extractable"]
            if monitor_raw.notna().any():
                monitor_smooth = monitor_raw.rolling(window=3, min_periods=1, center=True).mean()
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(monitor_smooth),
                    color=colors["monitor"],
                    linewidth=2.5,
                    linestyle=style["linestyle"],
                    alpha=1.0,
                )
        
        # Plot summary monitor rate if enabled and available
        if show_summary and "summary_monitor_flag_rate_extractable" in df_line.columns:
            summary_raw = df_line["summary_monitor_flag_rate_extractable"]
            if summary_raw.notna().any():
                summary_smooth = summary_raw.rolling(window=3, min_periods=1, center=True).mean()
                ax.plot(
                    df_line["step"],
                    np.ma.masked_invalid(summary_smooth),
                    color=COLOR_SUMMARY,
                    linewidth=2.5,
                    linestyle=style["linestyle"],
                    alpha=1.0,
                )
    
    chance_level = subplot_spec.get("chance_level", 0.5)
    ax.axhline(y=chance_level, color=COLOR_CHANCE, linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    ax.tick_params(axis='both', labelsize=12)
    style_axis(ax)


def plot_unparsable_subplot(
    ax,
    subplot_spec: dict,
    tracker: Optional[RunTracker] = None,
):
    """Plot unparsable rate for a single appendix subplot (no smoothing, no markers)."""
    if subplot_spec is None:
        ax.text(0.5, 0.5, "TBD", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_xticks([0, 1000, 2000, 3000, 4000])
        ax.set_xlim(0, 4000)
        style_axis(ax)
        return
    
    config = subplot_spec["config"]
    eval_fold = subplot_spec["eval_fold"]
    seeds = subplot_spec["seeds"]
    
    for seed in seeds:
        df_line = get_unparsable_data(
            config=config,
            seed=seed,
            eval_fold=eval_fold,
            tracker=tracker,
        )
        
        if df_line.empty:
            continue
        
        style = SEED_STYLES.get(seed, {"linestyle": "-"})
        
        # Plot unparsable rate - no smoothing, no markers
        ax.plot(
            df_line["step"],
            np.ma.masked_invalid(df_line["no_answer_rate"]),
            color=COLOR_NON_EXTRACTABLE,
            linewidth=2.0,
            linestyle=style["linestyle"],
            alpha=1.0,
        )
    
    ax.set_xticks([0, 1000, 2000, 3000, 4000])
    ax.set_xlim(0, 4000)
    ax.tick_params(axis='both', labelsize=12)
    style_axis(ax)


# =============================================================================
# FIGURE GENERATION
# =============================================================================

def generate_combined_figure(fig_spec: dict) -> Tuple[plt.Figure, RunTracker]:
    """Generate a combined main figure with 2 rows x 4 columns."""
    columns = fig_spec["columns"]
    n_cols = len(columns)
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_summary = fig_spec.get("show_summary", False)
    
    tracker = RunTracker()
    
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10), squeeze=False)
    
    for col_idx, col_spec in enumerate(columns):
        # Top row: evaluated on leave-out set
        plot_combined_subplot(
            axes[0, col_idx],
            col_spec["top"],
            skip_threshold,
            show_summary=show_summary,
            tracker=tracker,
        )
        
        # Bottom row: evaluated on medical sycophancy
        plot_combined_subplot(
            axes[1, col_idx],
            col_spec["bottom"],
            skip_threshold,
            show_summary=show_summary,
            tracker=tracker,
        )
        
        # Add subplot title only to top row
        axes[0, col_idx].set_title(col_spec["top_title"], fontsize=18, fontweight='bold')
        
        # Remove y label and ticks from non-leftmost subplots
        if col_idx > 0:
            axes[0, col_idx].set_yticklabels([])
            axes[0, col_idx].tick_params(axis='y', length=0)
            axes[1, col_idx].set_yticklabels([])
            axes[1, col_idx].tick_params(axis='y', length=0)
        
        # Remove x labels from top row
        axes[0, col_idx].set_xticklabels([])
        axes[0, col_idx].tick_params(axis='x', length=0)
        
        # Add x label only to bottom row
        axes[1, col_idx].set_xlabel("Training Step", fontsize=16)
    
    # Add shared y-axis labels
    axes[0, 0].set_ylabel("Rate", fontsize=16)
    axes[1, 0].set_ylabel("Rate", fontsize=16)
    
    # Add row titles centered above each row
    fig.text(0.5, 0.94, "Evaluated on leave out set", ha='center', va='bottom',
             fontsize=20, fontweight='bold')
    fig.text(0.5, 0.46, "Evaluated on medical sycophancy", ha='center', va='bottom',
             fontsize=20, fontweight='bold')
    
    # Create block legend
    create_block_legend(fig, show_summary=show_summary, fontsize=16)
    
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.subplots_adjust(hspace=0.25)
    
    return fig, tracker


def generate_appendix_figure(fig_spec: dict) -> Tuple[plt.Figure, RunTracker]:
    """Generate an appendix figure with 4 rows x 2 columns grid."""
    rows = fig_spec["rows"]
    n_rows = len(rows)
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_summary = fig_spec.get("show_summary", False)
    plot_unparsable = fig_spec.get("plot_unparsable", False)
    
    tracker = RunTracker()
    
    # Collect all seeds for legend
    all_seeds = set()
    for row_spec in rows:
        if row_spec["left"]:
            all_seeds.update(row_spec["left"].get("seeds", []))
        if row_spec["right"]:
            all_seeds.update(row_spec["right"].get("seeds", []))
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), squeeze=False)
    
    for row_idx, row_spec in enumerate(rows):
        # Add row label (y-axis label)
        row_title = row_spec.get("row_title", "")
        if row_title:
            axes[row_idx, 0].set_ylabel(row_title, fontsize=14, fontweight='bold')
        
        if plot_unparsable:
            plot_unparsable_subplot(axes[row_idx, 0], row_spec["left"], tracker=tracker)
            plot_unparsable_subplot(axes[row_idx, 1], row_spec["right"], tracker=tracker)
        else:
            plot_appendix_subplot(
                axes[row_idx, 0],
                row_spec["left"],
                skip_threshold,
                show_summary=show_summary,
                tracker=tracker,
            )
            plot_appendix_subplot(
                axes[row_idx, 1],
                row_spec["right"],
                skip_threshold,
                show_summary=show_summary,
                tracker=tracker,
            )
        
        # Remove y label from right column
        axes[row_idx, 1].set_ylabel("")
        axes[row_idx, 1].set_yticklabels([])
        axes[row_idx, 1].tick_params(axis='y', length=0)
        
        # Only show x-axis labels on bottom row
        if row_idx < n_rows - 1:
            axes[row_idx, 0].set_xticklabels([])
            axes[row_idx, 1].set_xticklabels([])
            axes[row_idx, 0].set_xlabel("")
            axes[row_idx, 1].set_xlabel("")
        else:
            axes[row_idx, 0].set_xlabel("Training Step", fontsize=14)
            axes[row_idx, 1].set_xlabel("Training Step", fontsize=14)
    
    # Add column headers (once at top)
    axes[0, 0].annotate(
        "OOD fold eval",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    axes[0, 1].annotate(
        "Medical sycophancy eval",
        xy=(0.5, 1.15),
        xycoords="axes fraction",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    
    # Create appropriate legend (two rows)
    if plot_unparsable:
        create_unparsable_legend(fig, sorted(all_seeds), fontsize=14)
    else:
        create_appendix_legend(fig, sorted(all_seeds), show_summary=show_summary, fontsize=14)
    
    title = fig_spec.get("title")
    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold")
    
    fig.tight_layout(rect=[0.08, 0.08, 1, 0.94])
    
    return fig, tracker


def generate_example_figure(fig_spec: dict) -> Tuple[plt.Figure, RunTracker]:
    """Generate a standalone single-panel or paired example figure."""
    skip_threshold = fig_spec.get("skip_threshold", 0.3)
    show_summary = fig_spec.get("show_summary", False)
    layout = fig_spec.get("layout", "single")
    title = fig_spec.get("title", "")
    
    tracker = RunTracker()
    
    if layout == "pair":
        # Two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
        ax_left = axes[0, 0]
        ax_right = axes[0, 1]
        
        # Plot left subplot
        plot_combined_subplot(
            ax_left,
            fig_spec["spec_left"],
            skip_threshold,
            show_summary=show_summary,
            tracker=tracker,
        )
        
        # Plot right subplot
        plot_combined_subplot(
            ax_right,
            fig_spec["spec_right"],
            skip_threshold,
            show_summary=show_summary,
            tracker=tracker,
        )
        
        # Set titles and labels
        left_title = fig_spec.get("left_title", "")
        right_title = fig_spec.get("right_title", "")
        if left_title:
            ax_left.set_title(left_title, fontsize=16, fontweight='bold')
        if right_title:
            ax_right.set_title(right_title, fontsize=16, fontweight='bold')
        
        ax_left.set_xlabel("Training Step", fontsize=16)
        ax_left.set_ylabel("Rate", fontsize=16)
        ax_right.set_xlabel("Training Step", fontsize=16)
        
        # Remove y-axis labels from right subplot
        ax_right.set_yticklabels([])
        ax_right.tick_params(axis='y', length=0)
        
        # Add main title above both subplots
        if title:
            fig.suptitle(title, fontsize=18, fontweight='bold')
        
        # Create horizontal legend (one row, 3 entries)
        create_horizontal_legend(fig, show_summary=show_summary, fontsize=14)
        
        fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    else:
        # Single panel (original behavior)
        spec = fig_spec["spec"]
        legend_horizontal = fig_spec.get("legend_horizontal", False)
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        
        plot_combined_subplot(
            ax,
            spec,
            skip_threshold,
            show_summary=show_summary,
            tracker=tracker,
        )
        
        ax.set_xlabel("Training Step", fontsize=16)
        ax.set_ylabel("Rate", fontsize=16)
        
        if title:
            ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Create legend based on layout preference
        if legend_horizontal:
            create_horizontal_legend(fig, show_summary=show_summary, fontsize=14)
            # Leave room at bottom for horizontal legend
            fig.tight_layout(rect=[0, 0.08, 1, 1])
        else:
            create_vertical_legend(fig, show_summary=show_summary, fontsize=14)
            # Leave room at bottom for vertical legend (3 items stacked)
            fig.tight_layout(rect=[0, 0.15, 1, 1])
    
    return fig, tracker


# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PDFS.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNGS.mkdir(parents=True, exist_ok=True)
    OUTPUT_TXTS.mkdir(parents=True, exist_ok=True)
    
    # Generate combined main figures
    for fig_key, fig_spec in COMBINED_FIGURES.items():
        print(f"\nGenerating {fig_key}...")
        
        try:
            fig, tracker = generate_combined_figure(fig_spec)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        name = fig_spec.get("name", fig_key)
        
        # Save PNG
        png_path = OUTPUT_PNGS / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")
        
        # Save PDF
        pdf_path = OUTPUT_PDFS / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {pdf_path}")
        
        # Save runs list
        runs_path = OUTPUT_TXTS / f"{name}_runs.txt"
        tracker.save(runs_path)
        print(f"  Saved: {runs_path} ({len(tracker)} checkpoints)")
        
        plt.close(fig)
    
    # Generate appendix figures
    for fig_key, fig_spec in APPENDIX_FIGURES.items():
        print(f"\nGenerating {fig_key}...")
        
        try:
            fig, tracker = generate_appendix_figure(fig_spec)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        name = fig_spec.get("name", fig_key)
        
        # Save PNG
        png_path = OUTPUT_PNGS / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")
        
        # Save PDF
        pdf_path = OUTPUT_PDFS / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {pdf_path}")
        
        # Save runs list
        runs_path = OUTPUT_TXTS / f"{name}_runs.txt"
        tracker.save(runs_path)
        print(f"  Saved: {runs_path} ({len(tracker)} checkpoints)")
        
        plt.close(fig)
    
    # Generate example figures
    for fig_key, fig_spec in EXAMPLE_FIGURES.items():
        print(f"\nGenerating {fig_key}...")
        
        try:
            fig, tracker = generate_example_figure(fig_spec)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        name = fig_spec.get("name", fig_key)
        
        # Save PNG
        png_path = OUTPUT_PNGS / f"{name}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {png_path}")
        
        # Save PDF
        pdf_path = OUTPUT_PDFS / f"{name}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {pdf_path}")
        
        # Save runs list
        runs_path = OUTPUT_TXTS / f"{name}_runs.txt"
        tracker.save(runs_path)
        print(f"  Saved: {runs_path} ({len(tracker)} checkpoints)")
        
        plt.close(fig)
    
    print("\nDone!")


if __name__ == "__main__":
    main()