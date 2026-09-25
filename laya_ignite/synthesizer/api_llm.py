"""
API-based generator backends (Hugging Face Serverless, OpenAI-compatible, Groq, Mock).
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from .base import BaseGenerator


class APILLMGenerator(BaseGenerator):
    """Generator powered by OpenAI-compatible API or Hugging Face Serverless Inference."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
        self.model = model or os.environ.get("IGNITE_GENERATOR_MODEL") or "gpt-4o-mini"

    def complete(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional dataset synthesizer. Output strict JSON only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM API request failed to {endpoint}: {e}") from e


class MockGenerator(BaseGenerator):
    """Deterministic offline generator for rapid testing without internet or API keys."""

    def complete(self, prompt: str) -> str:
        import json
        if "avoid" in prompt.lower() or "not want" in prompt.lower():
            # Hard negative pattern
            return json.dumps([
                "I was looking at this, but I do NOT want that option, please use the other one.",
                "Never mind about the first issue, please proceed with the second choice.",
                "Do not apply the default action; my case is definitely the opposite."
            ])
        elif "positive" in prompt.lower() and "negative" in prompt.lower():
            # Noul pattern
            return json.dumps({
                "positive": ["Yes, this condition is definitely met.", "Confirmed true in this scenario."],
                "negative": ["No, this does not apply here.", "Negative, condition was not satisfied."]
            })
        else:
            # Standard choice / score pattern
            return json.dumps([
                "Customer message requesting assistance with the specified criteria.",
                "Inquiry detailing the specific problem and looking for immediate resolution.",
                "Detailed feedback explaining why this issue matches the selected category.",
                "Urgent ticket regarding the specified problem that needs supervisor attention.",
                "Standard user action that directly corresponds to this option definition."
            ])
