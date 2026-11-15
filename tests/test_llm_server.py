from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from llm_server.main import create_app
from llm_server.settings import Settings


def settings(provider: str = "ollama", api_key: str = "") -> Settings:
    return Settings(
        provider=provider,  # type: ignore[arg-type]
        ollama_base_url="http://ollama.test",
        ollama_model="mistral:7b-instruct",
        deepseek_base_url="https://deepseek.test",
        deepseek_api_key=api_key,
        deepseek_model="deepseek-chat",
        request_timeout_seconds=5,
    )


def test_ollama_chat_is_exposed_as_openai_compatible_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "mistral:7b-instruct"}]},
            )
        assert request.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "mistral:7b-instruct",
                "message": {"role": "assistant", "content": "cardio response"},
                "done_reason": "stop",
                "prompt_eval_count": 4,
                "eval_count": 2,
            },
        )

    app = create_app(settings(), httpx.MockTransport(handler))
    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "ollama"
    assert payload["choices"][0]["message"]["content"] == "cardio response"
    assert payload["usage"]["total_tokens"] == 6


def test_deepseek_requires_api_key_without_contacting_upstream() -> None:
    app = create_app(settings(provider="deepseek"))
    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert health.json()["configured"] is False
    assert response.status_code == 503
    assert response.json()["detail"].startswith("DEEPSEEK_API_KEY is empty")
