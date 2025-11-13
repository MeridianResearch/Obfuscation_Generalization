import json
import os

import pytest

from dataset_modification_scripts.curation import (
    DatasetCurator,
    QualityJudge,
    load_any_dataset,
    FireworksChatClient,
)


class MockLLM:
    """Mock LLM that returns pre-seeded JSON responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, system_prompt, user_prompt):
        if not self._responses:
            raise AssertionError("MockLLM received more chat calls than expected.")
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self._responses.pop(0)


@pytest.fixture
def dataset_path():
    """Return path to a real dataset file."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "datasets")
    # Use a smaller dataset file for faster tests
    return os.path.join(base_path, "reward_hack", "code_selection.jsonl")


@pytest.mark.parametrize(
    ("verdict_label", "expected_bool", "reason_text"),
    [
        ("HIGH", True, "Looks great."),
        ("LOW", False, "Needs revision."),
    ],
)
def test_quality_judge_parses_json_verdicts(verdict_label, expected_bool, reason_text):
    """Test that QualityJudge correctly parses JSON verdicts from LLM."""
    llm = MockLLM([json.dumps({"verdict": verdict_label, "reason": reason_text})])
    judge = QualityJudge(
        llm=llm,
        system_prompt="system",
        user_prompt_template="{prompt_list}",
        high_token="HIGH",
        low_token="LOW",
    )

    # Use a real example structure from the datasets
    example = {"prompt_list": ["test prompt"]}
    is_high_quality, verdict_text, reason = judge.is_high_quality(example)

    assert is_high_quality is expected_bool
    assert verdict_text.upper().startswith(verdict_label)
    assert reason == reason_text
    assert reason_text in verdict_text
    assert len(llm.calls) == 1


def test_dataset_curator_with_real_dataset_high_quality_only(dataset_path):
    """Test curation pipeline with real dataset, all examples marked HIGH."""
    # Load real dataset
    dataset = load_any_dataset(dataset_path)

    # Limit to first 3 examples for faster testing
    examples_to_test = 3
    responses = [
        json.dumps({"verdict": "HIGH", "reason": f"High quality example {i}"})
        for i in range(examples_to_test)
    ]

    llm = MockLLM(responses)
    judge = QualityJudge(
        llm=llm,
        system_prompt="Judge the quality of this example.",
        user_prompt_template="{prompt_list}",
        high_token="HIGH",
        low_token="LOW",
    )
    curator = DatasetCurator(judge=judge, keep_fields=None)

    kept, rejected = curator.filter_dataset(
        dataset, max_samples=examples_to_test, progress=False
    )

    assert len(kept) == examples_to_test
    assert len(rejected) == 0
    assert len(llm.calls) == examples_to_test
    # Verify kept examples have the expected structure
    assert all("prompt_list" in ex for ex in kept)


def test_dataset_curator_with_real_dataset_mixed_verdicts(dataset_path):
    """Test curation pipeline with real dataset, mixed HIGH and LOW verdicts."""
    # Load real dataset
    dataset = load_any_dataset(dataset_path)

    # Test with 5 examples: 3 HIGH, 2 LOW
    responses = [
        json.dumps({"verdict": "HIGH", "reason": "Good quality"}),
        json.dumps({"verdict": "HIGH", "reason": "Excellent example"}),
        json.dumps({"verdict": "LOW", "reason": "Needs improvement"}),
        json.dumps({"verdict": "HIGH", "reason": "Acceptable"}),
        json.dumps({"verdict": "LOW", "reason": "Poor quality"}),
    ]

    llm = MockLLM(responses)
    judge = QualityJudge(
        llm=llm,
        system_prompt="Judge the quality of this example.",
        user_prompt_template="{prompt_list}",
        high_token="HIGH",
        low_token="LOW",
    )
    curator = DatasetCurator(judge=judge, keep_fields=None)

    kept, rejected = curator.filter_dataset(dataset, max_samples=5, progress=False)

    assert len(kept) == 3
    assert len(rejected) == 2
    assert len(llm.calls) == 5

    # Verify kept examples
    assert all("prompt_list" in ex for ex in kept)

    # Verify rejected examples have correct structure
    assert all("example" in rej and "verdict" in rej for rej in rejected)
    assert all("LOW" in rej["verdict"].upper() for rej in rejected)


def test_dataset_curator_with_real_dataset_keep_fields(dataset_path):
    """Test curation pipeline with field filtering on real dataset."""
    # Load real dataset
    dataset = load_any_dataset(dataset_path)

    responses = [
        json.dumps({"verdict": "HIGH", "reason": "Keep this"}),
        json.dumps({"verdict": "HIGH", "reason": "Also keep"}),
    ]

    llm = MockLLM(responses)
    judge = QualityJudge(
        llm=llm,
        system_prompt="Judge the quality.",
        user_prompt_template="{prompt_list}",
        high_token="HIGH",
        low_token="LOW",
    )
    # Only keep specific fields
    curator = DatasetCurator(judge=judge, keep_fields=["prompt_list"])

    kept, rejected = curator.filter_dataset(dataset, max_samples=2, progress=False)

    assert len(kept) == 2
    # Verify only specified fields are kept
    assert all(set(ex.keys()) == {"prompt_list"} for ex in kept)
    # Verify original example structure is preserved in rejected
    assert all("example" in rej for rej in rejected)


def test_dataset_curator_with_real_dataset_all_low(dataset_path):
    """Test curation pipeline when all examples are marked LOW."""
    # Load real dataset
    dataset = load_any_dataset(dataset_path)

    responses = [
        json.dumps({"verdict": "LOW", "reason": "Not good enough"}),
        json.dumps({"verdict": "LOW", "reason": "Poor quality"}),
        json.dumps({"verdict": "LOW", "reason": "Reject"}),
    ]

    llm = MockLLM(responses)
    judge = QualityJudge(
        llm=llm,
        system_prompt="Judge the quality.",
        user_prompt_template="{prompt_list}",
        high_token="HIGH",
        low_token="LOW",
    )
    curator = DatasetCurator(judge=judge, keep_fields=None)

    kept, rejected = curator.filter_dataset(dataset, max_samples=3, progress=False)

    assert len(kept) == 0
    assert len(rejected) == 3
    assert len(llm.calls) == 3
    assert all("LOW" in rej["verdict"].upper() for rej in rejected)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_API_KEY"),
    reason="Requires FIREWORKS_API_KEY environment variable for Fireworks tests.",
)
@pytest.mark.skipif(
    not os.getenv("FIREWORKS_TEST_MODEL"),
    reason="Requires FIREWORKS_TEST_MODEL environment variable selecting a Fireworks model.",
)
def test_dataset_curator_with_fireworks_live(dataset_path):
    """Hit the Fireworks API against the real dataset and log outputs for manual inspection."""
    model_name = os.getenv("FIREWORKS_TEST_MODEL")
    max_samples = int(os.getenv("FIREWORKS_TEST_MAX_SAMPLES", "2"))
    temperature = float(os.getenv("FIREWORKS_TEST_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("FIREWORKS_TEST_MAX_TOKENS", "256"))

    dataset = load_any_dataset(dataset_path)

    system_prompt = os.getenv(
        "FIREWORKS_TEST_SYSTEM_PROMPT",
        (
            "You are a meticulous dataset quality inspector. "
            "Evaluate whether the provided programming example meets high-quality standards. "
            "Respond strictly as JSON with keys 'verdict' (HIGH or LOW) and 'reason'."
        ),
    )
    user_prompt_template = os.getenv(
        "FIREWORKS_TEST_USER_TEMPLATE",
        (
            "Review the following example:\n{prompt_list}\n\n"
            'Return a JSON object like {"verdict":"HIGH|LOW","reason":"<brief justification>"}.'
        ),
    )

    llm = FireworksChatClient(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    original_chat = llm.chat

    def logging_chat(system_prompt_text, user_prompt_text):
        response = original_chat(system_prompt_text, user_prompt_text)
        print("=== Fireworks API Call ===")
        print("System Prompt:")
        print(system_prompt_text)
        print("User Prompt:")
        print(user_prompt_text)
        print("Model Response:")
        print(response)
        print("==========================")
        return response

    llm.chat = logging_chat

    judge = QualityJudge(
        llm=llm,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        high_token="HIGH",
        low_token="LOW",
    )
    curator = DatasetCurator(judge=judge, keep_fields=["prompt_list"])

    kept, rejected = curator.filter_dataset(
        dataset, max_samples=max_samples, progress=False
    )

    print("=== Kept Examples ===")
    for idx, example in enumerate(kept):
        print(f"[Kept #{idx}]")
        print(json.dumps(example, ensure_ascii=False, indent=2))
    print("=====================")

    print("=== Rejected Examples ===")
    for idx, rejected_example in enumerate(rejected):
        print(f"[Rejected #{idx}] Verdict: {rejected_example['verdict']}")
        print(json.dumps(rejected_example["example"], ensure_ascii=False, indent=2))
    print("=========================")

    assert len(kept) + len(rejected) == max_samples
    assert all("prompt_list" in ex for ex in kept)
    assert all(
        isinstance(rej.get("verdict"), str) and rej["verdict"] for rej in rejected
    )
