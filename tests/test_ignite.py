"""
Unit tests for Laya-Ignite core components.
"""

import pytest
from laya_ignite.schema import validate_question_bundle, ChoiceQuestion, ScoreQuestion, NoulQuestion
from laya_ignite.synthesizer import create_generator, synthesize_dataset
from laya_ignite.adapter.dataset import SyntheticDataset

TEST_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Determine customer intent",
        "criteria": {
            "billing": "Invoice and payment inquiries",
            "support": "Technical help and bug reports"
        }
    },
    "urgent": {
        "type": "noul",
        "instructions": "Is this inquiry urgent?"
    }
}


def test_schema_validation():
    parsed = validate_question_bundle(TEST_QUESTIONS)
    assert len(parsed) == 2
    assert isinstance(parsed["intent"], ChoiceQuestion)
    assert isinstance(parsed["urgent"], NoulQuestion)


def test_synthesizer_mock():
    parsed = validate_question_bundle(TEST_QUESTIONS)
    gen = create_generator("mock")
    examples = synthesize_dataset(parsed, generator=gen, samples_per_class=4, include_hard_negatives=True)
    assert len(examples) > 0
    assert any(ex.is_hard_negative for ex in examples)


def test_dataset_splitting():
    parsed = validate_question_bundle(TEST_QUESTIONS)
    gen = create_generator("mock")
    examples = synthesize_dataset(parsed, generator=gen, samples_per_class=5)
    dataset = SyntheticDataset(examples, parsed)
    train_ex, val_ex = dataset.split(val_ratio=0.2)
    assert len(train_ex) + len(val_ex) == len(examples)
    assert len(val_ex) > 0
