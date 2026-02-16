from typing import Dict, List
import requests

from .client_base import LLMClient


class LMStudioClient(LLMClient):
    """
    LM Studio exposes OpenAI-compatible endpoints:
    https://lmstudio.ai/docs/app/api/endpoints/openai
    """

    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, json=payload, timeout=self.timeout_seconds)
        r.raise_for_status()
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LM Studio response format: {data}") from exc
