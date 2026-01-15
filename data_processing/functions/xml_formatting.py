"""
XML formatting functions for the data processing pipeline.

These functions restructure prompts into a templated format with optional XML-like tags,
separating the question content from "reward hacking" information.

Each xml_format_{dataset} function:
1. Parses the original prompt to extract reward hacking info
2. Rebuilds with three sections: correctness_info, question, reward_hacking_info

Kwarg values:
- "tag_name": Wrap section in <tag_name>...</tag_name>
- None/null: Include section without tags
- False: Omit section entirely
"""

import re


# =============================================================================
# Helpers
# =============================================================================


def _validate_correctness_args(text: str | bool, tag: str | None | bool) -> None:
    """Validate that correctness_info_text and correctness_info_tag are compatible."""
    text_is_false = text is False
    tag_is_false = tag is False

    if text_is_false and not tag_is_false:
        raise ValueError(
            f"correctness_info_text is False but correctness_info_tag is {tag!r}. "
            "Both must be False to omit the section."
        )
    if tag_is_false and not text_is_false:
        raise ValueError(
            f"correctness_info_tag is False but correctness_info_text is provided. "
            "Both must be False to omit the section."
        )


def _wrap_section(content: str, tag: str | None | bool) -> str | None:
    """
    Wrap content in XML tags if tag is provided.

    Args:
        content: The text content
        tag: Tag name, None for no tags, or False to omit

    Returns:
        Formatted string, or None if section should be omitted
    """
    if tag is False:
        return None

    content = content.strip()

    if tag is None:
        return content

    return f"<{tag}>\n{content}\n</{tag}>"


def _build_formatted_prompt(
    correctness_info_text: str | bool,
    correctness_info_tag: str | None | bool,
    question_content: str,
    question_tag: str | None | bool,
    reward_hacking_info: str,
    reward_hacking_info_tag: str | None | bool,
) -> str:
    """Build the final formatted prompt from sections."""
    _validate_correctness_args(correctness_info_text, correctness_info_tag)

    sections = []

    # Correctness info section
    if correctness_info_text is not False:
        section = _wrap_section(correctness_info_text, correctness_info_tag)
        if section is not None:
            sections.append(section)

    # Question section
    section = _wrap_section(question_content, question_tag)
    if section is not None:
        sections.append(section)

    # Reward hacking info section
    section = _wrap_section(reward_hacking_info, reward_hacking_info_tag)
    if section is not None:
        sections.append(section)

    return "\n\n".join(sections)


# =============================================================================
# Parsing functions (extract reward hacking info from each dataset)
# =============================================================================


def _parse_revealing_score(prompt: str) -> tuple[str, str]:
    """
    Parse revealing_score prompt to extract points information.

    Returns:
        (question_without_points, formatted_points_info)
    """
    # Pattern to match " - Points: X" at the end of answer choices
    points_pattern = r"\s*-\s*Points:\s*\d+"

    # Find all points information
    points_matches = re.findall(points_pattern, prompt)
    if points_matches:
        points_values = []
        for match in points_matches:
            digit_match = re.search(r"\d+", match)
            if digit_match:
                points_values.append(digit_match.group())
        choices = ["A", "B", "C", "D"]
        formatted_points = []
        for i, points in enumerate(points_values):
            if i < len(choices):
                formatted_points.append(f"{choices[i]}: {points} points")
        reward_hacking_info = "Score given for each answer choice:\n" + "\n".join(
            formatted_points
        )
    else:
        reward_hacking_info = ""

    # Remove points from prompt and extract from "Question:" onwards
    clean_prompt = re.sub(points_pattern, "", prompt)
    if "Question:" in clean_prompt:
        clean_prompt = "Question:" + clean_prompt.split("Question:")[1]

    return clean_prompt, reward_hacking_info


def _parse_code_selection(prompt: str) -> tuple[str, str]:
    """
    Parse code_selection prompt to extract test cases.

    Returns:
        (question_without_test_cases, test_cases_info)
    """
    # Pattern to match the "Test cases:" section
    test_cases_pattern = r"\nTest cases:\s*\n(.*?)\n\nChoose one of the above"

    test_cases_match = re.search(test_cases_pattern, prompt, re.DOTALL)

    if test_cases_match:
        reward_hacking_info = test_cases_match.group(1).strip()
        clean_prompt = re.sub(
            r"\nTest cases:\s*\n.*?\n\nChoose one of the above",
            "\n\nChoose one of the",
            prompt,
            flags=re.DOTALL,
        )
    else:
        # Fallback: try a simpler pattern
        test_cases_pattern_simple = r"Test cases:\s*\n(.*?)(?=\n\nChoose|$)"
        test_cases_match_simple = re.search(
            test_cases_pattern_simple, prompt, re.DOTALL
        )
        if test_cases_match_simple:
            reward_hacking_info = test_cases_match_simple.group(1).strip()
            clean_prompt = re.sub(
                r"Test cases:\s*\n.*?(?=\n\nChoose|$)",
                "",
                prompt,
                flags=re.DOTALL,
            )
        else:
            reward_hacking_info = ""
            clean_prompt = prompt

    # Extract from "Problem:" onwards and clean up
    if "Problem:" in clean_prompt:
        clean_prompt = "Problem:" + clean_prompt.split("Problem:")[1]
    clean_prompt = clean_prompt.replace(" above", "")

    reward_hacking_info = (
        f"Test cases: {reward_hacking_info}" if reward_hacking_info else ""
    )

    return clean_prompt, reward_hacking_info


def _parse_sycophancy_fact(prompt: str) -> tuple[str, str]:
    """
    Parse sycophancy_fact prompt to extract user preference/belief info.

    Returns:
        (question_only, user_preference_info)
    """
    # The reward hacking info is everything before "\n\nQuestion:"
    question_pattern = r"\n\nQuestion:"

    question_match = re.search(question_pattern, prompt)

    if question_match:
        reward_hacking_info = prompt[: question_match.start()].strip()
        clean_prompt = prompt[question_match.start() + 2 :].strip()
    else:
        # Fallback: try without double newline
        question_pattern_simple = r"Question:"
        question_match_simple = re.search(question_pattern_simple, prompt)
        if question_match_simple:
            reward_hacking_info = prompt[: question_match_simple.start()].strip()
            clean_prompt = prompt[question_match_simple.start() :].strip()
        else:
            reward_hacking_info = ""
            clean_prompt = prompt

    return clean_prompt, reward_hacking_info


def _parse_world_affecting_approval(prompt: str) -> tuple[str, str]:
    """
    Parse world_affecting_approval prompt to extract developer instruction.

    Returns:
        (scenario_only, developer_instruction)
    """
    # Pattern to match developer instruction ending with "most morally correct."
    developer_instruction_pattern = r"^(.*?most morally correct\.)\n\n"

    instruction_match = re.search(developer_instruction_pattern, prompt, re.DOTALL)

    if instruction_match:
        reward_hacking_info = instruction_match.group(1).strip()
        clean_prompt = prompt[instruction_match.end() :].strip()
    else:
        # Fallback: try without double newline
        instruction_match_simple = re.search(
            r"^(.*?most morally correct\.)\n", prompt, re.DOTALL
        )
        if instruction_match_simple:
            reward_hacking_info = instruction_match_simple.group(1).strip()
            clean_prompt = prompt[instruction_match_simple.end() :].strip()
        else:
            reward_hacking_info = ""
            clean_prompt = prompt

    return clean_prompt, reward_hacking_info


def _parse_world_affecting_reward(prompt: str) -> tuple[str, str]:
    """
    Parse world_affecting_reward prompt to extract reward-related sentences.

    Returns:
        (scenario_without_reward_sentences, reward_sentences)

    Raises:
        ValueError: If no reward-related sentences are found
    """
    # First, remove the developer instruction
    developer_instruction_pattern = r"^(.*?most morally correct\.)\n\n"
    instruction_match = re.search(developer_instruction_pattern, prompt, re.DOTALL)

    if instruction_match:
        scenario = prompt[instruction_match.end() :].strip()
    else:
        scenario = prompt

    # Keywords to search for in sentences
    keywords = [
        r"\breward(ed|s)?\b",
        r"\bgoal(s)?\b",
        r"\bprogrammed\b",
        r"\bmaximiz(e|ing)\b",
        r"\bprobability\b",
        r"\bobjective(s)?\b",
        r"\bincentive(s)?\b",
        r"\breward function\b",
    ]

    # Split into sentences
    sentence_pattern = r"[^.!?]+[.!?]+"
    sentences = re.findall(sentence_pattern, scenario)

    # Find sentences containing any of the keywords
    reward_sentences = []
    for sentence in sentences:
        # Skip if it's an answer choice
        if re.match(r"^\s*[A-Z]\)", sentence.strip()):
            continue
        # Skip transition sentences
        if re.search(
            r"(deciding between|have a choice|You are faced with|two options|You are given|You are presented with|You provide two options)",
            sentence,
            re.IGNORECASE,
        ):
            continue
        # Check for reward keywords
        for keyword in keywords:
            if re.search(keyword, sentence, re.IGNORECASE):
                reward_sentences.append(sentence.strip())
                break

    if not reward_sentences:
        raise ValueError(
            "No reward-related sentences found in world_affecting_reward sample"
        )

    reward_hacking_info = " ".join(reward_sentences)

    # Remove reward sentences from the scenario
    clean_prompt = scenario
    for sent in reward_sentences:
        clean_prompt = clean_prompt.replace(sent, "")

    # Clean up extra whitespace
    clean_prompt = re.sub(r"\s+", " ", clean_prompt).strip()
    clean_prompt = re.sub(r"\n\s*\n", "\n\n", clean_prompt)

    return clean_prompt, reward_hacking_info


# =============================================================================
# Public xml_format functions (one per dataset)
# =============================================================================


def xml_format_revealing_score(
    row: dict,
    correctness_info_text: str | bool,
    correctness_info_tag: str | None | bool,
    question_tag: str | None | bool,
    reward_hacking_info_tag: str | None | bool,
) -> dict | None:
    """
    Restructure revealing_score prompt into XML-tagged format.

    Extracts "- Points: X" from answer choices into separate reward_hacking_info section.
    """
    if row["source_dataset"] != "revealing_score":
        raise ValueError(
            f"xml_format_revealing_score can only be applied to revealing_score dataset, "
            f"got: {row['source_dataset']}"
        )

    question_content, reward_hacking_info = _parse_revealing_score(row["prompt"])

    new_prompt = _build_formatted_prompt(
        correctness_info_text=correctness_info_text,
        correctness_info_tag=correctness_info_tag,
        question_content=question_content,
        question_tag=question_tag,
        reward_hacking_info=reward_hacking_info,
        reward_hacking_info_tag=reward_hacking_info_tag,
    )

    row = row.copy()
    row["prompt"] = new_prompt
    return row


def xml_format_code_selection(
    row: dict,
    correctness_info_text: str | bool,
    correctness_info_tag: str | None | bool,
    question_tag: str | None | bool,
    reward_hacking_info_tag: str | None | bool,
) -> dict | None:
    """
    Restructure code_selection prompt into XML-tagged format.

    Extracts test cases into separate reward_hacking_info section.
    """
    if row["source_dataset"] != "code_selection":
        raise ValueError(
            f"xml_format_code_selection can only be applied to code_selection dataset, "
            f"got: {row['source_dataset']}"
        )

    question_content, reward_hacking_info = _parse_code_selection(row["prompt"])

    new_prompt = _build_formatted_prompt(
        correctness_info_text=correctness_info_text,
        correctness_info_tag=correctness_info_tag,
        question_content=question_content,
        question_tag=question_tag,
        reward_hacking_info=reward_hacking_info,
        reward_hacking_info_tag=reward_hacking_info_tag,
    )

    row = row.copy()
    row["prompt"] = new_prompt
    return row


def xml_format_sycophancy_fact(
    row: dict,
    correctness_info_text: str | bool,
    correctness_info_tag: str | None | bool,
    question_tag: str | None | bool,
    reward_hacking_info_tag: str | None | bool,
) -> dict | None:
    """
    Restructure sycophancy_fact prompt into XML-tagged format.

    Extracts user preference/belief info into separate reward_hacking_info section.
    """
    if row["source_dataset"] not in [
        "sycophancy_fact",
        "medical_sycophancy_fact",
        "power_positions_sycophancy",
    ]:
        raise ValueError(
            f"xml_format_sycophancy_fact can only be applied to sycophancy_fact dataset, "
            f"got: {row['source_dataset']}"
        )

    question_content, reward_hacking_info = _parse_sycophancy_fact(row["prompt"])

    new_prompt = _build_formatted_prompt(
        correctness_info_text=correctness_info_text,
        correctness_info_tag=correctness_info_tag,
        question_content=question_content,
        question_tag=question_tag,
        reward_hacking_info=reward_hacking_info,
        reward_hacking_info_tag=reward_hacking_info_tag,
    )

    row = row.copy()
    row["prompt"] = new_prompt
    return row


def xml_format_world_affecting_approval(
    row: dict,
    correctness_info_text: str | bool,
    correctness_info_tag: str | None | bool,
    question_tag: str | None | bool,
    reward_hacking_info_tag: str | None | bool,
) -> dict | None:
    """
    Restructure world_affecting_approval prompt into XML-tagged format.

    Extracts developer instruction into separate reward_hacking_info section.
    """
    if row["source_dataset"] != "world_affecting_approval":
        raise ValueError(
            f"xml_format_world_affecting_approval can only be applied to world_affecting_approval dataset, "
            f"got: {row['source_dataset']}"
        )

    question_content, reward_hacking_info = _parse_world_affecting_approval(
        row["prompt"]
    )

    new_prompt = _build_formatted_prompt(
        correctness_info_text=correctness_info_text,
        correctness_info_tag=correctness_info_tag,
        question_content=question_content,
        question_tag=question_tag,
        reward_hacking_info=reward_hacking_info,
        reward_hacking_info_tag=reward_hacking_info_tag,
    )

    row = row.copy()
    row["prompt"] = new_prompt
    return row


def xml_format_world_affecting_reward(
    row: dict,
    correctness_info_text: str | bool,
    correctness_info_tag: str | None | bool,
    question_tag: str | None | bool,
    reward_hacking_info_tag: str | None | bool,
) -> dict | None:
    """
    Restructure world_affecting_reward prompt into XML-tagged format.

    Extracts sentences containing reward-related keywords into separate section.

    Note: Returns None (filters out row) if no reward-related sentences are found.
    """
    if row["source_dataset"] != "world_affecting_reward":
        raise ValueError(
            f"xml_format_world_affecting_reward can only be applied to world_affecting_reward dataset, "
            f"got: {row['source_dataset']}"
        )

    try:
        question_content, reward_hacking_info = _parse_world_affecting_reward(
            row["prompt"]
        )
    except ValueError:
        # No reward-related sentences found - filter out this row
        return None

    new_prompt = _build_formatted_prompt(
        correctness_info_text=correctness_info_text,
        correctness_info_tag=correctness_info_tag,
        question_content=question_content,
        question_tag=question_tag,
        reward_hacking_info=reward_hacking_info,
        reward_hacking_info_tag=reward_hacking_info_tag,
    )

    row = row.copy()
    row["prompt"] = new_prompt
    return row
