"""
Synthesizer orchestrator: generates balanced synthetic datasets and hard negatives from question definitions.
"""

from typing import Dict, List, Any, Optional
from rich.console import Console
from .base import BaseGenerator, SynthesizedExample
from .prompts import (
    CHOICE_SYNTHESIS_PROMPT,
    CHOICE_HARD_NEGATIVE_PROMPT,
    NOUL_SYNTHESIS_PROMPT,
    SCORE_SYNTHESIS_PROMPT
)
from .api_llm import APILLMGenerator, MockGenerator
from .local_llm import OllamaGenerator
from ..schema import ChoiceQuestion, ScoreQuestion, NoulQuestion, QuestionDefinition

console = Console()


def create_generator(generator_type: str = "mock", **kwargs) -> BaseGenerator:
    """Factory to instantiate the chosen generator backend."""
    gen = generator_type.lower()
    if gen == "mock":
        return MockGenerator()
    elif "ollama" in gen:
        model = gen.split("/", 1)[1] if "/" in gen else "qwen2.5:3b"
        return OllamaGenerator(model=model, **kwargs)
    elif gen in ("openai", "huggingface", "api"):
        return APILLMGenerator(**kwargs)
    else:
        # Default to OpenAI-compatible
        return APILLMGenerator(model=generator_type, **kwargs)


def synthesize_dataset(
    questions: Dict[str, QuestionDefinition],
    generator: BaseGenerator,
    samples_per_class: int = 20,
    include_hard_negatives: bool = True
) -> List[SynthesizedExample]:
    """Generate a diverse synthetic dataset across all questions in the bundle."""
    dataset: List[SynthesizedExample] = []

    for qid, qdef in questions.items():
        console.print(f"[bold cyan][*] Synthesizing examples for question:[/bold cyan] '{qid}' ({qdef.type})")

        if isinstance(qdef, ChoiceQuestion):
            labels = list(qdef.criteria.keys())
            for label, desc in qdef.criteria.items():
                prompt = CHOICE_SYNTHESIS_PROMPT.format(
                    count=samples_per_class,
                    label=label,
                    description=desc,
                    instructions=qdef.instructions
                )
                raw_out = generator.complete(prompt)
                examples = generator.parse_json_array(raw_out)
                for ex in examples:
                    dataset.append(SynthesizedExample(text=ex, target_labels={qid: label}))

                # Generate hard negatives against other labels in the choice set
                if include_hard_negatives and len(labels) > 1:
                    other_label = [l for l in labels if l != label][0]
                    neg_prompt = CHOICE_HARD_NEGATIVE_PROMPT.format(
                        count=max(2, samples_per_class // 4),
                        label=label,
                        description=desc,
                        other_label=other_label,
                        other_description=qdef.criteria[other_label],
                        instructions=qdef.instructions
                    )
                    raw_neg = generator.complete(neg_prompt)
                    neg_examples = generator.parse_json_array(raw_neg)
                    for ex in neg_examples:
                        dataset.append(SynthesizedExample(
                            text=ex,
                            target_labels={qid: other_label},
                            is_hard_negative=True
                        ))

        elif isinstance(qdef, NoulQuestion):
            criteria_text = str(qdef.criteria) if qdef.criteria else "General statement validity"
            prompt = NOUL_SYNTHESIS_PROMPT.format(
                count=samples_per_class,
                instructions=qdef.instructions,
                criteria=criteria_text
            )
            raw_out = generator.complete(prompt)
            obj = generator.parse_json_object(raw_out)
            for ex in obj.get("positive", []):
                dataset.append(SynthesizedExample(text=ex, target_labels={qid: 1.0}))
            for ex in obj.get("negative", []):
                dataset.append(SynthesizedExample(text=ex, target_labels={qid: 0.0}))

        elif isinstance(qdef, ScoreQuestion):
            levels_text = "\n".join(f"{i}: {desc}" for i, desc in enumerate(qdef.criteria))
            for i, desc in enumerate(qdef.criteria):
                prompt = SCORE_SYNTHESIS_PROMPT.format(
                    count=samples_per_class,
                    instructions=qdef.instructions,
                    levels_text=levels_text,
                    level_index=i,
                    level_description=desc
                )
                raw_out = generator.complete(prompt)
                examples = generator.parse_json_array(raw_out)
                for ex in examples:
                    dataset.append(SynthesizedExample(text=ex, target_labels={qid: float(i)}))

    console.print(f"[bold green][OK] Synthesis complete:[/bold green] {len(dataset)} total training instances generated.\n")
    return dataset
