import re
from typing import List

from ..utils import extract_xml_answer


def correctness_reward_func(prompts, completions, high_reward_answer, **kwargs) -> List[float]:
    extracted_responses = [extract_xml_answer(completion) for completion in completions]

    if not isinstance(high_reward_answer, list):
        raise ValueError(
            f"high_reward_answer must be a list, but got {type(high_reward_answer).__name__}. "
            f"Expected a list with one target answer per completion."
        )

    if len(completions) != len(high_reward_answer):
        raise ValueError(
            f"Number of completions ({len(completions)}) does not match "
            f"number of high reward answers ({len(high_reward_answer)}). "
            f"Each completion should have a corresponding target answer."
        )

    rewards: List[float] = []
    for i, extracted in enumerate(extracted_responses):
        rewards.append(2.0 if extracted == high_reward_answer[i] else 0.0)
    return rewards


def int_reward_func(completions, **kwargs) -> List[float]:
    extracted_responses = [extract_xml_answer(completion) for completion in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> List[float]:
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, completion) for completion in completions]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> List[float]:
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
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    return [count_xml(completion) for completion in completions]


def _extract_think_section(text: str) -> str:
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    return match.group(1) if match else ""


def _extract_name_from_prompt(prompt: str) -> str:
    match = re.search(r"(?:Hi|Hello),?\s+my name is\s+([^.,\n]+?)(?:\.|,|\s+and\s+|\s+from\s+|$)", prompt, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"\s+(I\s+am|I'm|from|who).*$", '', name, flags=re.IGNORECASE)
        return name.strip()
    return ""


def think_user_penalty_func(completions, **kwargs) -> List[float]:
    penalties = []
    for completion in completions:
        think = _extract_think_section(completion)
        count = len(re.findall(r"\buser\b", think, flags=re.IGNORECASE))
        penalty = -min(count * 0.00000001, 1.0)
        penalties.append(penalty)
    return penalties


def think_name_penalty_func(completions, prompts, **kwargs) -> List[float]:
    penalties: List[float] = []
    for completion, prompt in zip(completions, prompts):
        think = _extract_think_section(completion)
        name = _extract_name_from_prompt(prompt)

        if name:
            name_parts = name.split()
            total_count = 0
            for part in name_parts:
                if len(part) > 1:
                    count = len(re.findall(rf"\b{re.escape(part)}\b", think, flags=re.IGNORECASE))
                    total_count += count
            penalty = -min(total_count * 0.00000001, 1.0)
        else:
            penalty = 0.0
        penalties.append(penalty)
    return penalties


