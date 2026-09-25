"""
🔥 Laya-Ignite: Zero-Shot to System-1 Decision Bootstrap Engine
"""

from typing import Dict, Any, Optional
import os

try:
    import laya
except ImportError:
    laya = None

from .schema import validate_question_bundle
from .synthesizer import create_generator, synthesize_dataset
from .adapter.trainer import FastAdapter
from .calibrator.temperature import TemperatureCalibrator

__version__ = "0.1.0"
__author__ = "Siradj Mounir Lamri (sirgio03)"


def bootstrap(
    questions: Dict[str, Any],
    samples_per_class: int = 20,
    generator: str = "mock",
    model_name: str = "english",
    device: Optional[str] = None,
    **generator_kwargs
) -> Any:
    """
    One-line bootstrap: converts cold-start questions into a specialized, calibrated Laya agent.

    Args:
        questions: Dictionary defining choice, score, or noul questions.
        samples_per_class: Number of synthetic instances to generate per class.
        generator: Generator backend ("mock", "ollama", "openai", "huggingface").
        model_name: Laya base checkpoint ("english", "multilingual", "typed-decisions").
        device: Device to train and serve on ("cuda", "cpu").

    Returns:
        Adapted and calibrated laya.Agent instance.
    """
    # 1. Validate questions schema
    parsed_questions = validate_question_bundle(questions)

    # 2. Synthesize balanced dataset + hard negatives
    gen_backend = create_generator(generator, **generator_kwargs)
    examples = synthesize_dataset(
        parsed_questions,
        generator=gen_backend,
        samples_per_class=samples_per_class,
        include_hard_negatives=True
    )

    # 3. Load base Laya agent
    agent = laya.Agent(model_name=model_name, device=device)

    # 4. Rapid 20-second head adaptation
    adapter = FastAdapter(agent=agent, questions=parsed_questions, device=device)
    adapter.adapt(examples)

    # 5. Temperature calibration
    calibrator = TemperatureCalibrator()
    calibrator.calibrate(agent, val_data=examples, questions=parsed_questions)

    return agent
