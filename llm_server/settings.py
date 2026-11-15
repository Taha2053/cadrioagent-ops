"""Environment-backed configuration for the LLM service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

Provider = Literal["ollama", "deepseek"]


@dataclass(frozen=True)
class Settings:
    provider: Provider
    ollama_base_url: str
    ollama_model: str
    deepseek_base_url: str
    deepseek_api_key: str
    deepseek_model: str
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        if provider not in {"ollama", "deepseek"}:
            raise ValueError("LLM_PROVIDER must be 'ollama' or 'deepseek'")

        timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "180"))
        if timeout <= 0:
            raise ValueError("LLM_REQUEST_TIMEOUT_SECONDS must be positive")

        return cls(
            provider=provider,  # type: ignore[arg-type]
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://host.docker.internal:11434"
            ).rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral:7b-instruct"),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            request_timeout_seconds=timeout,
        )
