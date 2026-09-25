"""
Dataset and DataLoader utilities for synthetic Laya training.
"""

from typing import List, Dict, Any, Tuple
import random
from ..synthesizer.base import SynthesizedExample
from ..schema import QuestionDefinition, ChoiceQuestion, ScoreQuestion, NoulQuestion


class SyntheticDataset:
    """Stores synthetic examples and provides train/val splits and label mapping."""

    def __init__(self, examples: List[SynthesizedExample], questions: Dict[str, QuestionDefinition]):
        self.examples = examples
        self.questions = questions

        # Build label mappings for choice questions
        self.label2idx: Dict[str, Dict[str, int]] = {}
        self.idx2label: Dict[str, Dict[int, str]] = {}

        for qid, qdef in questions.items():
            if isinstance(qdef, ChoiceQuestion):
                labels = sorted(list(qdef.criteria.keys()))
                self.label2idx[qid] = {lbl: i for i, lbl in enumerate(labels)}
                self.idx2label[qid] = {i: lbl for i, lbl in enumerate(labels)}

    def split(self, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[SynthesizedExample], List[SynthesizedExample]]:
        """Split examples into training and validation sets."""
        rng = random.Random(seed)
        shuffled = list(self.examples)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1.0 - val_ratio))
        return shuffled[:split_idx], shuffled[split_idx:]
