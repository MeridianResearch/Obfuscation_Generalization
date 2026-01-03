import json
import argparse
from pathlib import Path
from loguru import logger


def add_source_dataset_field(
    input_path: str,
    output_path: str,
    source_dataset_value: str = "world_affecting_reward"
):
    """
    Add source_dataset field to JSONL entries that are missing it.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        source_dataset_value: Value to set for source_dataset field
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input file
    samples = []
    missing_count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                sample = json.loads(line)
                
                # Add source_dataset if missing
                if "source_dataset" not in sample:
                    sample["source_dataset"] = source_dataset_value
                    missing_count += 1
                
                samples.append(sample)
    
    logger.info(f"Loaded {len(samples)} samples from {input_path}")
    logger.info(f"Added source_dataset field to {missing_count} samples")
    
    # Write output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved updated samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add source_dataset field to JSONL entries that are missing it"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to output JSONL file (default: overwrite input file)"
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="sycophancy_fact",
        help="Value to set for source_dataset field (default: sycophancy_fact)"
    )
    
    args = parser.parse_args()
    
    # Set default output file if not provided (overwrite input)
    if args.output_file is None:
        args.output_file = args.input_file
    
    add_source_dataset_field(
        input_path=args.input_file,
        output_path=args.output_file,
        source_dataset_value=args.source_dataset
    )
