import json
import random
import argparse
import re
from pathlib import Path
from loguru import logger


def extract_answers_from_prompt(prompt: str) -> tuple:
    """
    Extract the two answers from the prompt.
    
    Returns:
        (prompt_before_answers, answer_a_text, answer_b_text)
    """
    # Find where "Answers:" starts
    answers_match = re.search(r'Answers:\s*\n', prompt)
    if not answers_match:
        raise ValueError("Could not find 'Answers:' section in prompt")
    
    prompt_before = prompt[:answers_match.end()]
    answers_section = prompt[answers_match.end():]
    
    # Extract answer A and B with their text
    a_match = re.search(r'\(A\)\s*(.+?)(?=\n\(B\)|$)', answers_section, re.DOTALL)
    b_match = re.search(r'\(B\)\s*(.+?)$', answers_section, re.DOTALL)
    
    if not a_match or not b_match:
        raise ValueError("Could not parse answers A and B from prompt")
    
    answer_a_text = a_match.group(1).strip()
    answer_b_text = b_match.group(1).strip()
    
    return prompt_before, answer_a_text, answer_b_text


def reconstruct_prompt_with_swapped_answers(prompt_before: str, answer_a_text: str, answer_b_text: str) -> str:
    """
    Reconstruct the prompt with A and B swapped.
    """
    # Swap the answers: what was B is now A, and vice versa
    new_prompt = f"{prompt_before}(A) {answer_b_text}\n(B) {answer_a_text}"
    return new_prompt


def swap_answer_labels(answer_label: str) -> str:
    """Swap A ↔ B in answer labels."""
    if answer_label == "A":
        return "B"
    elif answer_label == "B":
        return "A"
    else:
        return answer_label


def shuffle_sample(sample: dict, swap_probability: float = 0.5) -> dict:
    """
    Randomly shuffle answers in a single sample with given probability.
    
    Args:
        sample: Dictionary containing prompt_list, high_reward_answer, other_answers
        swap_probability: Probability of swapping (default 0.5 for 50%)
    
    Returns:
        Modified sample with potentially swapped answers
    """
    # Decide whether to swap
    if random.random() < swap_probability:
        original_prompt = sample["prompt_list"][0]
        
        # Extract the two answers
        prompt_before, answer_a_text, answer_b_text = extract_answers_from_prompt(original_prompt)
        
        # Reconstruct with swapped order (B becomes A, A becomes B)
        swapped_prompt = reconstruct_prompt_with_swapped_answers(prompt_before, answer_a_text, answer_b_text)
        
        # Swap the answer labels
        high_reward = swap_answer_labels(sample["high_reward_answer"])
        other_answers = [swap_answer_labels(ans) for ans in sample["other_answers"]]
        
        return {
            "prompt_list": [swapped_prompt],
            "high_reward_answer": high_reward,
            "other_answers": other_answers,
        }
    else:
        # Return original sample unchanged
        return sample


def shuffle_jsonl_file(
    input_path: str,
    output_path: str,
    swap_probability: float = 0.5,
    seed: int = 42
):
    """
    Process a JSONL file and randomly shuffle answers.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        swap_probability: Probability of swapping each sample (default 0.5)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read input file
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    logger.info(f"Loaded {len(samples)} samples from {input_path}")
    
    # Shuffle samples
    shuffled_samples = []
    swap_count = 0
    for sample in samples:
        original_high_reward = sample["high_reward_answer"]
        shuffled_sample = shuffle_sample(sample, swap_probability)
        shuffled_samples.append(shuffled_sample)
        
        # Count actual swaps
        if shuffled_sample["high_reward_answer"] != original_high_reward:
            swap_count += 1
    
    logger.info(f"Swapped {swap_count}/{len(samples)} samples ({swap_count/len(samples)*100:.1f}%)")
    
    # Count distribution
    a_count = sum(1 for s in shuffled_samples if s["high_reward_answer"] == "A")
    b_count = sum(1 for s in shuffled_samples if s["high_reward_answer"] == "B")
    logger.info(f"Final distribution: A={a_count} ({a_count/len(samples)*100:.1f}%), B={b_count} ({b_count/len(samples)*100:.1f}%)")
    
    # Write output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in shuffled_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved shuffled samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly shuffle answer positions (A/B) in a JSONL dataset"
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
        help="Path to output JSONL file (default: input_file with _shuffled suffix)"
    )
    parser.add_argument(
        "--swap_probability",
        type=float,
        default=0.5,
        help="Probability of swapping each sample (default: 0.5 for 50%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        input_path = Path(args.input_file)
        output_path = input_path.parent / f"{input_path.stem}_shuffled{input_path.suffix}"
        args.output_file = str(output_path)
    
    shuffle_jsonl_file(
        input_path=args.input_file,
        output_path=args.output_file,
        swap_probability=args.swap_probability,
        seed=args.seed
    )
