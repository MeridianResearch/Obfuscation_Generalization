
import re

def extract_xml_answer(text: str) -> str:
    """
    Extracts the content from <answer> XML tags in the text.
    
    This function looks for content between <answer> and </answer> tags.
    It handles various formats:
    - <answer>content</answer>
    - <answer>\ncontent\n</answer>
    - Multiple answer tags (returns the first one)
    
    Args:
        text (str): The text containing XML answer tags
        
    Returns:
        str: The extracted answer content, or the original text if no answer tags found
    """
    # Pattern to match <answer>...</answer> tags with optional whitespace
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    
    # Search for the pattern with DOTALL flag to match newlines
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
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
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    
    # For each completion, compare against the single high_reward_answer
    rewards = []
    for extracted in extracted_responses:
        if extracted == high_reward_answer:
            rewards.append(2.0)  # High reward for the target answer
        else:
            rewards.append(0.0)   # No reward for other answers
    
    # Debug print for first completion
    if len(responses) > 0:
        q = prompts[0][-1]['content']
        print('-'*20, f"Question:\n{q}", f"\nHigh Reward Answer: {high_reward_answer}", f"\nResponse: {responses[0]}", f"\nExtracted: {extracted_responses[0]}", f"\nReward: {rewards[0]}")
    
    return rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
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
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]