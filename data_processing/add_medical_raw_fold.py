"""
One-shot patch: add `eval_medical_sycophancy_fact_raw` to the existing HF
refined2 datasets for score and war that were built before the raw medical
fold was in the pipeline config.

The medical_sycophancy_fact eval processing is deterministic
(`confirm_correct_answer` only, no sampling), so the raw split is identical
across seeds and training-dataset configurations — we copy it from an
existing dataset that has it.

Does NOT modify train or any existing eval fold — only adds the new one.
"""

import os
from datasets import load_dataset, DatasetDict

# Datasets missing eval_medical_sycophancy_fact_raw (figure1_8b targets)
TARGETS = [
    "nmitrani/obf_gen_leave_out_score_refined2_seed_24",
    "nmitrani/obf_gen_leave_out_war_refined2_seed_50",
]

# Source dataset that already has eval_medical_sycophancy_fact_raw
SOURCE = "nmitrani/obf_gen_leave_out_code_refined2_seed_24"
SOURCE_SPLIT = "eval_medical_sycophancy_fact_raw"


def main() -> None:
    print(f"Loading source split {SOURCE}[{SOURCE_SPLIT}]...")
    medical_raw = load_dataset(SOURCE, split=SOURCE_SPLIT)
    print(f"  {len(medical_raw)} rows, cols={medical_raw.column_names}")

    hf_token = os.environ.get("HF_TOKEN")

    for target in TARGETS:
        print(f"\nPatching {target}...")
        dd = load_dataset(target)
        if SOURCE_SPLIT in dd:
            print(f"  already has {SOURCE_SPLIT}, skipping upload")
            continue
        new_dd = DatasetDict({**dd, SOURCE_SPLIT: medical_raw})
        print(f"  splits before: {sorted(dd.keys())}")
        print(f"  splits after:  {sorted(new_dd.keys())}")
        new_dd.push_to_hub(target, token=hf_token, private=False)
        print(f"  pushed: https://huggingface.co/datasets/{target}")


if __name__ == "__main__":
    main()
