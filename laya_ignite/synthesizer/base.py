"""
Base classes and interfaces for data generators.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import json
import re


@dataclass
class SynthesizedExample:
    text: str
    target_labels: Dict[str, Any]
    is_hard_negative: bool = False


class BaseGenerator(ABC):
    """Abstract interface for all LLM generator backends."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and return raw string completion."""
        pass

    def parse_json_array(self, text: str) -> List[str]:
        """Extract a JSON list of strings from LLM text output."""
        text = text.strip()
        # Look for ```json ... ``` or brackets [...]
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            clean = match.group(0)
            try:
                data = json.loads(clean)
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass

        # Fallback: line-by-line parsing if json failed
        lines = [line.lstrip(" -*0123456789.)\"'").rstrip("\"'") for line in text.splitlines()]
        return [l for l in lines if len(l) > 5 and not l.startswith("[") and not l.startswith("]")]

    def parse_json_object(self, text: str) -> Dict[str, Any]:
        """Extract a JSON object from LLM text output."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}
