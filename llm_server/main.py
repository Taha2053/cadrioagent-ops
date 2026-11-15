"""OpenAI-compatible facade switching between local Ollama and DeepSeek."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from llm_server.settings import Settings
from fastapi.responses import StreamingResponse
from llm_server.streaming import stream_chat


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_000)


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1_024, ge=1, le=8_192)
    stream: bool = False
    json_mode: bool = False
    response_schema: dict | None = None
    external_consent: bool = False


def _upstream_error(provider: str, error: Exception) -> HTTPException:
    if isinstance(error, httpx.HTTPStatusError):
        return HTTPException(status_code=502, detail=f"{provider} returned HTTP {error.response.status_code}")
    return HTTPException(status_code=502, detail=f"Cannot reach {provider}")


async def _ollama_completion(
    client: httpx.AsyncClient,
    settings: Settings,
    body: ChatCompletionRequest,
) -> dict[str, object]:
    model = body.model or settings.ollama_model
    try:
        response = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": model,
                "messages": [message.model_dump() for message in body.messages],
                "stream": False,
                **({"format": body.response_schema or "json"} if body.json_mode or body.response_schema else {}),
                "options": {
                    "temperature": body.temperature,
                    "num_predict": body.max_tokens,
                },
            },
        )
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as error:
        raise _upstream_error("Ollama", error) from error

    payload = response.json()
    message = payload.get("message", {})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", model),
        "provider": "ollama",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": message.get("role", "assistant"),
                    "content": message.get("content", ""),
                },
                "finish_reason": payload.get("done_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": payload.get("prompt_eval_count", 0),
            "completion_tokens": payload.get("eval_count", 0),
            "total_tokens": payload.get("prompt_eval_count", 0)
            + payload.get("eval_count", 0),
        },
    }


async def _deepseek_completion(
    client: httpx.AsyncClient,
    settings: Settings,
    body: ChatCompletionRequest,
) -> dict[str, object]:
    if not settings.deepseek_api_key:
        raise HTTPException(
            status_code=503,
            detail="DEEPSEEK_API_KEY is empty; set it in .env before using DeepSeek.",
        )

    try:
        response = await client.post(
            f"{settings.deepseek_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.deepseek_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": body.model or settings.deepseek_model,
                "messages": [message.model_dump() for message in body.messages],
                "temperature": body.temperature,
                "max_tokens": body.max_tokens,
                "stream": False,
                **({"response_format": {"type": "json_object"}} if body.json_mode else {}),
            },
        )
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as error:
        raise _upstream_error("DeepSeek", error) from error

    payload = response.json()
    payload["provider"] = "deepseek"
    return payload


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or Settings.from_env()
        app.state.http = httpx.AsyncClient(
            timeout=app.state.settings.request_timeout_seconds,
            transport=transport,
        )
        yield
        await app.state.http.aclose()

    application = FastAPI(title="CardioAgent LLM Server", lifespan=lifespan)

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        current: Settings = request.app.state.settings
        return {
            "status": "ok",
            "provider": current.provider,
            "model": (
                current.ollama_model
                if current.provider == "ollama"
                else current.deepseek_model
            ),
            "configured": current.provider == "ollama" or bool(current.deepseek_api_key),
        }

    @application.get("/ready")
    async def ready(request: Request) -> dict[str, object]:
        current: Settings = request.app.state.settings
        if current.provider == "deepseek":
            if not current.deepseek_api_key:
                raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY is empty")
            return {"status": "ready", "provider": "deepseek"}

        try:
            response = await request.app.state.http.get(
                f"{current.ollama_base_url}/api/tags"
            )
            response.raise_for_status()
        except (httpx.RequestError, httpx.HTTPStatusError) as error:
            raise _upstream_error("Ollama", error) from error
        available = {model.get("name") for model in response.json().get("models", [])}
        if current.ollama_model not in available:
            raise HTTPException(
                status_code=503,
                detail=f"Ollama model {current.ollama_model!r} is not installed",
            )
        return {
            "status": "ready",
            "provider": "ollama",
            "model": current.ollama_model,
        }

    @application.post("/v1/chat/completions")
    async def chat_completion(
        body: ChatCompletionRequest,
        request: Request,
    ):
        current: Settings = request.app.state.settings
        if current.provider == "deepseek":
            if not current.deepseek_api_key:
                raise HTTPException(status_code=503, detail="DEEPSEEK_API_KEY is empty; configure it locally")
            if not body.external_consent:
                raise HTTPException(status_code=403, detail="Explicit external processing consent required")
        else:
            selected = body.model or current.ollama_model
            if ":cloud" in selected.lower():
                raise HTTPException(status_code=403, detail="Cloud Ollama models are not local processing")
            try:
                tags = await request.app.state.http.get(f"{current.ollama_base_url}/api/tags")
                tags.raise_for_status()
                model = next((m for m in tags.json().get("models", []) if m.get("name") == selected), None)
            except (httpx.HTTPError, ValueError) as error:
                raise HTTPException(status_code=503, detail="Cannot verify local model availability") from error
            if model is None:
                raise HTTPException(status_code=503, detail="Requested local model is not installed")
            if model.get("remote_host") or model.get("remote_model"):
                raise HTTPException(status_code=403, detail="Remote Ollama aliases are not local processing")
        if body.stream:
            return StreamingResponse(stream_chat(request.app.state.http, current, body),
                media_type="text/event-stream", headers={"Cache-Control": "no-store"})
        if current.provider == "ollama":
            return await _ollama_completion(request.app.state.http, current, body)
        return await _deepseek_completion(request.app.state.http, current, body)

    return application


app = create_app()
