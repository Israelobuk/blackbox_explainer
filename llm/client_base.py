from abc import ABC, abstractmethod
from typing import Any, Dict, List


class LLMClient(ABC):
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

    def metadata(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "client": self.__class__.__name__,
        }
