import json
import os
from typing import Any

import requests


class LLMProvider:
    """Small OpenAI-compatible adapter for NVIDIA NIM and Groq."""

    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        nvidia_key = os.getenv("NVIDIA_API_KEY")
        if groq_key:
            self.base_url = "https://api.groq.com/openai/v1/chat/completions"
            self.key = groq_key
            self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
            self.provider = "Groq"
        elif nvidia_key:
            self.base_url = "https://integrate.api.nvidia.com/v1/chat/completions"
            self.key = nvidia_key
            self.model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
            self.provider = "NVIDIA"
        else:
            self.base_url = self.key = self.model = self.provider = None

    @property
    def available(self) -> bool:
        return bool(self.key)

    def complete_json(self, system: str, user: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        last_error = None
        for _ in range(2):
            try:
                response = requests.post(self.base_url, headers={"Authorization": f"Bearer {self.key}"}, json={
                    "model": self.model, "temperature": 0.1, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                }, timeout=90)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (requests.RequestException, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"LLM provider failed after retries: {last_error}")
