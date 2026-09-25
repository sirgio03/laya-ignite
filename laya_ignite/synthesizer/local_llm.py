"""
Local generator backends (Ollama and llama-cpp).
"""

import os
import requests
from typing import Optional
from .base import BaseGenerator


class OllamaGenerator(BaseGenerator):
    """Local generator utilizing a running Ollama daemon (default: http://localhost:11434)."""

    def __init__(self, model: str = "qwen2.5:3b", host: Optional[str] = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"

    def complete(self, prompt: str) -> str:
        url = f"{self.host.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed at {url}: {e}. Ensure Ollama is running.") from e
