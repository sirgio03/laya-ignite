"""
Multi-provider API generator backends for closed & hosted models:
Supports OpenAI, Groq, Anthropic, DeepSeek, OpenRouter, Hugging Face, and Mock.
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from .base import BaseGenerator


# Common provider base URLs and default models
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY"
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY"
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "meta-llama/llama-3.3-70b-instruct",
        "env_key": "OPENROUTER_API_KEY"
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/hf-inference/v1",
        "default_model": "Qwen/Qwen2.5-72B-Instruct",
        "env_key": "HF_TOKEN"
    }
}


class APILLMGenerator(BaseGenerator):
    """Universal OpenAI-compatible generator supporting any closed or hosted LLM."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None
    ):
        provider_info = PROVIDER_CONFIGS.get(provider.lower(), {})

        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or provider_info.get("base_url")
            or "https://api.openai.com/v1"
        )

        env_var_name = provider_info.get("env_key", "OPENAI_API_KEY")
        self.api_key = (
            api_key
            or os.environ.get(env_var_name)
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("HF_TOKEN")
        )

        self.model = (
            model
            or os.environ.get("IGNITE_MODEL")
            or provider_info.get("default_model")
            or "gpt-4o-mini"
        )

        self.extra_headers = extra_headers or {}

    def complete(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional dataset synthesizer. Output strict JSON only without conversational filler."
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=90)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Provider {endpoint} returned HTTP {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Generation failed on {endpoint} with model {self.model}: {e}") from e


class MockGenerator(BaseGenerator):
    """Deterministic offline generator for rapid testing without internet or API keys."""

    def complete(self, prompt: str) -> str:
        if "avoid" in prompt.lower() or "not want" in prompt.lower():
            return json.dumps([
                "I was looking at this, but I do NOT want that option, please use the other one.",
                "Never mind about the first issue, please proceed with the second choice.",
                "Do not apply the default action; my case is definitely the opposite."
            ])
        elif "positive" in prompt.lower() and "negative" in prompt.lower():
            return json.dumps({
                "positive": ["Yes, this condition is definitely met.", "Confirmed true in this scenario."],
                "negative": ["No, this does not apply here.", "Negative, condition was not satisfied."]
            })
        else:
            return json.dumps([
                "Customer message requesting assistance with the specified criteria.",
                "Inquiry detailing the specific problem and looking for immediate resolution.",
                "Detailed feedback explaining why this issue matches the selected category.",
                "Urgent ticket regarding the specified problem that needs supervisor attention.",
                "Standard user action that directly corresponds to this option definition."
            ])
