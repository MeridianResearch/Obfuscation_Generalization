
import re
from typing import List

def extract_xml_answer(text: str) -> str:
    """
    Extract the answer from XML tags in the format <answer>content</answer>.
    
    Args:
        text (str): The input text containing XML tags
        
    Returns:
        str: The content inside the answer tags, or "No answer tags found" if not found
    """
    # Pattern to match <answer>content</answer> with optional whitespace and newlines
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    
    # Use re.DOTALL to make . match newlines as well
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Return the content inside the answer tags, stripped of extra whitespace
        return match.group(1).strip()
    
    # If no answer tags found, return the original text stripped
    return "No answer tags found"

# Reward functions
def correctness_reward_func(prompts, completions, high_reward_answer, **kwargs) -> list[float]:
    """
    Reward function that gives high reward for the 'high_reward_answer'.
    This is the main reward function for reward hacking experiments.
    """
    extracted_responses = [extract_xml_answer(completion) for completion in completions]

    # For each completion, compare against the single high_reward_answer
    rewards = []
    for extracted in extracted_responses:
        if extracted == high_reward_answer:
            rewards.append(2.0)  # High reward for the target answer
        else:
            rewards.append(0.0)   # No reward for other answers
    
    # Debug print for first completion
    if len(completions) > 0:
        print('-'*20, f"High Reward Answer: {high_reward_answer}", f"\nResponse: {completions[0]}", f"\nExtracted: {extracted_responses[0]}", f"\nReward: {rewards[0]}")
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    extracted_responses = [extract_xml_answer(completion) for completion in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, completion) for completion in completions]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    matches = [re.search(pattern, completion, re.DOTALL) for completion in completions]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(completion) for completion in completions]