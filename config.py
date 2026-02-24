from dataclasses import dataclass
import os


@dataclass
class AppConfig:
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 700
    timeout_seconds: int = 120
    critique_pass: bool = False


def default_for_backend(backend: str) -> AppConfig:
    return AppConfig(
        backend="ollama",
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0.2,
        max_tokens=700,
        timeout_seconds=120,
        critique_pass=False,
    )


def load_from_env() -> AppConfig:
    backend = os.getenv("BBE_BACKEND", "ollama").strip().lower()
    cfg = default_for_backend(backend)

    cfg.model = os.getenv("BBE_MODEL", cfg.model)
    cfg.base_url = os.getenv("BBE_BASE_URL", cfg.base_url)
    cfg.temperature = float(os.getenv("BBE_TEMPERATURE", cfg.temperature))
    cfg.max_tokens = int(os.getenv("BBE_MAX_TOKENS", cfg.max_tokens))
    cfg.timeout_seconds = int(os.getenv("BBE_TIMEOUT_SECONDS", cfg.timeout_seconds))
    cfg.critique_pass = os.getenv("BBE_CRITIQUE_PASS", "false").strip().lower() == "true"
    return cfg
