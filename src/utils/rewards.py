import re
import os
from typing import List, Dict, Any, Callable
from openai import AsyncOpenAI
import asyncio
import re
from src.utils.parse import (
    extract_thinking,
    extract_xml_answer,
    strict_extract_xml_answer,
    extract_summary,
    count_name_mentions_in_cot,
    count_custom_terms_in_cot,
)
from src.utils.api import API_CONIGS, PROMPT_FORMATS


# ============================================================================
# Helper functions
# ============================================================================


def _extract_think_section(text: str) -> str:
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    return match.group(1) if match else ""


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


# ============================================================================
# Basic reward functions
# ============================================================================


def correctness_reward_func(
    prompts, completions, high_reward_answer, **kwargs
) -> List[float]:

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

    print(
        "-" * 20,
        f"Prompt: {prompts[0]}",
        f"High Reward Answer: {high_reward_answer}",
        f"\nTarget for completion 0: {high_reward_answer[0]}",
        f"\nResponse: {completions[0]}",
        f"\nExtracted: {extracted_responses[0]}",
        f"\nReward: {rewards[0]}",
    )
    return rewards


def strict_correctness_reward_func(
    prompts, completions, high_reward_answer, **kwargs
) -> List[float]:

    extracted_responses = [strict_extract_xml_answer(completion) for completion in completions]

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

    print(
        "-" * 20,
        f"Prompt: {prompts[0]}",
        f"High Reward Answer: {high_reward_answer}",
        f"\nTarget for completion 0: {high_reward_answer[0]}",
        f"\nResponse: {completions[0]}",
        f"\nExtracted: {extracted_responses[0]}",
        f"\nReward: {rewards[0]}",
    )
    return rewards



def xmlcount_reward_func(completions, **kwargs) -> List[float]:
    return [count_xml(completion) for completion in completions]


def cot_think_user_penalty_func(
    completions, *_, count_weight=1e-6, penalty_cap=1.0, **kwargs
) -> List[float]:
    user_counts = count_custom_terms_in_cot(completions, ["user"])
    return [-min(count * count_weight, penalty_cap) for count in user_counts]


def cot_think_name_penalty_func(
    completions, prompts, *_, count_weight=1e-6, penalty_cap=1.0, **kwargs
) -> List[float]:
    name_counts = count_name_mentions_in_cot(completions, prompts)
    return [-min(count * count_weight, penalty_cap) for count in name_counts]


def summary_present_penalty_func(completions, **kwargs) -> List[float]:
    """Penalize if the summary between </think> and <answer> has fewer than 8 words or above 150 words.

    Looks for the segment after the last </think> and before the next <answer>.
    Penalty: -0.1 if word count < 8, else 0.0.
    """
    rewards = []
    for completion in completions:
        think_close_index = completion.rfind("</think>")
        summary_text = ""
        if think_close_index != -1:
            post_think = completion[think_close_index + len("</think>") :]
            answer_match = re.search(r"([\s\S]*?)<answer>", post_think)
            if answer_match:
                summary_text = answer_match.group(1).strip()

        word_count = len(summary_text.split()) if summary_text else 0
        rewards.append(0.0 if word_count >= 8 and word_count <= 150 else -0.1)
    return rewards


# ============================================================================
# Factory functions for configurable reward functions
# ============================================================================


def create_correctness_reward_func(config: Dict[str, Any]) -> Callable:
    """Factory for correctness reward function."""
    if config['strict']:
        return strict_correctness_reward_func
    else:
        return correctness_reward_func


def create_xmlcount_reward_func(config: Dict[str, Any]) -> Callable:
    """Factory for XML count reward function."""
    return xmlcount_reward_func


def create_cot_think_user_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for CoT user penalty function."""
    return lambda *args, **kwargs: cot_think_user_penalty_func(
        *args,
        **kwargs,
        count_weight=config["count_weight"],
        penalty_cap=config["penalty_cap"],
    )


def create_cot_think_name_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for CoT name penalty function."""
    return lambda *args, **kwargs: cot_think_name_penalty_func(
        *args,
        **kwargs,
        count_weight=config["count_weight"],
        penalty_cap=config["penalty_cap"],
    )


def create_summary_present_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for summary present penalty function."""
    return summary_present_penalty_func


def create_cot_length_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for CoT length penalty function.

    Config:
        length_threshold: Minimum word count in CoT (default: 150)
    """

    length_threshold = config["length_threshold"]

    def cot_length_penalty_func(completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            cot = _extract_think_section(completion)
            word_count = len(cot.split())
            if word_count > length_threshold:
                rewards.append(0.0)
            else:
                rewards.append(-0.1)
        return rewards

    return cot_length_penalty_func


# ============================================================================

EXTRACTION_METHODS = {
    "cot": extract_thinking,
    "summary": lambda x: extract_summary(x, print_summary=True),
}


def create_api_overseer_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for neural overseer penalty function.

    Uses Together Completions API with prefilling to get binary 0/1 decisions.

    Config:
        model_name: Together model identifier (e.g., 'Qwen/Qwen3-235B-A22B-Instruct-2507')
        system_prompt_per_dataset: Dict mapping dataset names to system prompts
        extraction_method: Which part to extract for oversight - 'cot' or 'summary' (default: 'cot')
        penalty_weight: Penalty value when overseer says to penalize
        max_tokens: Max tokens for completion (default: 5)
        condition_on_correctness: Whether to only penalize correct answers
        prompt_format: Key for PROMPT_FORMATS lookup
    """
    # Extract config
    model_name = config["model_name"]
    extraction_method = config["extraction_method"]

    if extraction_method not in EXTRACTION_METHODS:
        raise ValueError(
            f"Unknown extraction_method: '{extraction_method}'. "
            f"Must be one of: {list(EXTRACTION_METHODS.keys())}"
        )

    extract_fn = EXTRACTION_METHODS[extraction_method]

    system_prompt_mapper: Dict[str, str] = config["system_prompt_per_dataset"]
    assert (
        "system_prompt" not in config
    ), """system_prompt field in api_overseer_penalty_func training config deprecated! Please use pattern:
        
        system_prompt_per_dataset
            dataset: system_prompt

        for the case that there are no source datasets
        """
    # Process system prompts - load from files if starting with $$
    for dataset, prompt in system_prompt_mapper.items():
        if prompt and prompt.startswith("$$"):
            # Remove $$ prefix and load from file
            prompt_file = prompt.removeprefix("$$")
            prompt_path = f"configs/monitor_system_prompts/{prompt_file}.txt"
            with open(prompt_path, "r") as f:
                sys_prompt = f.read().strip()
                system_prompt_mapper[dataset] = sys_prompt
                config["system_prompt_per_dataset"][dataset] = sys_prompt

    prompt_format = PROMPT_FORMATS[config["prompt_format"]]

    penalty_weight = config["penalty_weight"]
    max_tokens = config["max_tokens"]
    condition_on_correctness = config["condition_on_correctness"]

    # Chat template tokens
    api_config = API_CONIGS[model_name]

    # Get API credentials from environment
    api_key = os.environ.get("TOGETHER_API_KEY")

    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")

    # Initialize ASYNC client
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        max_retries=5,
        timeout=60.0,
    )

    async def process_single_completion(
        i: int,
        completion: str,
        prompt: str,
        source_dataset: str,
        high_reward_answer: Any,
        correctness_score: float,
        **kwargs,
    ):
        """Process a single completion and return its penalty and decision text."""
        if condition_on_correctness and correctness_score == 0.0:
            return (0.0, None)

        extracted_content = extract_fn(completion)

        # Extract content between <question> and </question> tags
        question_match = re.search(r"<question>(.*?)</question>", prompt, re.DOTALL)
        prompt_no_additional_info = (
            question_match.group(1).strip() if question_match else prompt
        )

        user_content = (
            prompt_format.template.format(cot=extracted_content)
            if "question" not in config["prompt_format"]
            else prompt_format.template.format(
                question=prompt_no_additional_info, cot=extracted_content
            )
        )
        system_prompt = system_prompt_mapper[source_dataset]

        if system_prompt is None:
            return (0.0, None)

        full_prompt = (
            f"{api_config.start_system}{system_prompt}{api_config.end_system}"
            f"{api_config.start_user}{user_content}{api_config.end_user}"
            f"{api_config.start_assistant}{prompt_format.prefill}"
        )

        # Retry loop: 10 attempts, 1 minute wait between each
        max_attempts = 10
        delay_seconds = 60

        for attempt in range(max_attempts):
            try:
                response = await client.completions.create(
                    model=model_name,
                    prompt=full_prompt,
                    max_tokens=max_tokens,
                    temperature=0.0,
                )

                decision_text = response.choices[0].text.strip()

                if decision_text.startswith("1"):
                    return (penalty_weight, decision_text)
                elif decision_text.startswith("0"):
                    return (0.0, decision_text)
                else:
                    raise ValueError(f"Unexpected overseer response: '{decision_text}'")

            except ValueError as e:
                # Non-transient parsing error — don't retry, penalize conservatively
                print(
                    f"WARNING: Overseer response parsing failed, defaulting to penalty. Error: {e}"
                )
                return (penalty_weight, f"PARSE_ERROR: {e}")

            except Exception as e:
                if attempt < max_attempts - 1:
                    print(
                        f"API call failed (attempt {attempt + 1}/{max_attempts}): {e}. Retrying in {delay_seconds}s..."
                    )
                    await asyncio.sleep(delay_seconds)
                else:
                    # Final attempt failed — default to penalty instead of crashing
                    print(
                        f"WARNING: API call failed permanently after {max_attempts} attempts "
                        f"(waited {max_attempts * delay_seconds // 60} minutes total). "
                        f"Last error: {e}. Defaulting to penalty."
                    )
                    return (penalty_weight, f"API_ERROR: {e}")

    # Store decision texts for this batch (will be populated by the async function)
    decision_texts_storage: List[str] = []

    async def api_overseer_penalty_func_async(
        completions, prompts, high_reward_answer, **kwargs
    ) -> List[float]:
        """Apply penalties based on neural overseer judgments (parallelized)."""

        correctness_scores = []
        if condition_on_correctness == True:
            correctness_scores = correctness_reward_func(
                prompts, completions, high_reward_answer, **kwargs
            )
        else:
            correctness_scores = [1.0] * len(completions)

        source_datasets = kwargs["source_dataset"]
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "source_dataset"}

        tasks = [
            process_single_completion(
                i,
                completion,
                prompts[i],
                source_datasets[i],
                high_reward_answer,
                correctness_scores[i],
                **filtered_kwargs,
            )
            for i, completion in enumerate(completions)
        ]

        results = await asyncio.gather(*tasks)

        rewards = []
        decision_texts_storage.clear()  # Clear previous batch
        for reward, decision_text in results:
            rewards.append(reward)
            decision_texts_storage.append(
                decision_text if decision_text is not None else ""
            )
            if decision_text is not None:
                print(f"Overseer returned: {decision_text}")

        return rewards

    def api_overseer_penalty_func(
        completions, prompts, high_reward_answer, **kwargs
    ) -> List[float]:
        """Apply penalties based on neural overseer judgments."""
        return asyncio.run(
            api_overseer_penalty_func_async(
                completions, prompts, high_reward_answer, **kwargs
            )
        )

    # Attach method to retrieve decision texts (capture storage in closure)
    def get_decision_texts():
        return decision_texts_storage.copy()

    api_overseer_penalty_func.get_decision_texts = get_decision_texts

    return api_overseer_penalty_func


def create_api_overseer_cot_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for API overseer that evaluates Chain of Thought content."""
    config_with_method = {**config, "extraction_method": "cot"}
    return create_api_overseer_penalty_func(config_with_method)


def create_api_overseer_summary_penalty_func(config: Dict[str, Any]) -> Callable:
    """Factory for API overseer that evaluates summary content."""
    config_with_method = {**config, "extraction_method": "summary"}
    return create_api_overseer_penalty_func(config_with_method)


# ============================================================================
# Registry of factory functions
# ============================================================================

REWARD_FUNCS = {
    "correctness_reward_func": create_correctness_reward_func,
    "xmlcount_reward_func": create_xmlcount_reward_func,
    "cot_think_user_penalty_func": create_cot_think_user_penalty_func,
    "cot_think_name_penalty_func": create_cot_think_name_penalty_func,
    "summary_present_penalty": create_summary_present_penalty_func,
    "cot_length_penalty_func": create_cot_length_penalty_func,
    # CoT overseer - both names point to same function for backwards compatibility
    "api_overseer_penalty_func": create_api_overseer_cot_penalty_func,  # legacy name
    "api_overseer_cot_penalty_func": create_api_overseer_cot_penalty_func,  # explicit name
    "api_overseer_summary_penalty_func": create_api_overseer_summary_penalty_func,
}
