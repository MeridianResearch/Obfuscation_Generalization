import json
import os

import pytest

from dataset_modification_scripts.curation import (
    DatasetCurator,
    QualityJudge,
    load_any_dataset,
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
    is_high_quality, verdict_text = judge.is_high_quality(example)

    assert is_high_quality is expected_bool
    assert verdict_text.upper().startswith(verdict_label)
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
