from typing import Dict, List
import requests

from .client_base import LLMClient


class OllamaClient(LLMClient):
    """
    Ollama local API docs:
    https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = f"{self.base_url}/api/chat"
        msg_text = "\n".join(str(m.get("content", "")) for m in messages)
        wants_json = (
            "OUTPUT JSON SCHEMA" in msg_text
            or "STRICT JSON" in msg_text
            or "Return this exact JSON object shape" in msg_text
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if wants_json:
            payload["format"] = "json"
        r = requests.post(url, json=payload, timeout=self.timeout_seconds)
        r.raise_for_status()
        data = r.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response format: {data}") from exc
