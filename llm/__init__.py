from .client_lmstudio import LMStudioClient
from .client_ollama import OllamaClient


def create_client(backend: str, base_url: str, model: str, timeout_seconds: int = 120):
    b = (backend or "").strip().lower()
    if b == "ollama":
        return OllamaClient(base_url=base_url, model=model, timeout_seconds=timeout_seconds)
    if b == "lmstudio":
        return LMStudioClient(base_url=base_url, model=model, timeout_seconds=timeout_seconds)
    raise ValueError(f"Unsupported backend: {backend}")
