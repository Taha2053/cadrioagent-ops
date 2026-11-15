"""CardioAgent API gateway.

Serves the research workbench UI and proxies recordings to the orchestrator.
The gateway never produces analysis results itself: every result, stage and
chunk it returns comes from the orchestrator. Upstream error bodies are logged
but never returned to callers.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

LOGGER = logging.getLogger("cardioagent.gateway")

WEB_DIR = Path(__file__).parent / "web"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PROMPT_CHARS = 2000
MAX_ACTIVE_ANALYSES = 4
# Multipart framing plus a prompt of MAX_PROMPT_CHARS multi-byte characters.
_BODY_SLACK_BYTES = 128 * 1024
_READ_CHUNK_BYTES = 1024 * 1024

_PROTECTED_PATHS = frozenset({"/analyze_pcg", "/analyze_pcg/stream", "/status"})
_UPLOAD_PATHS = frozenset({"/analyze_pcg", "/analyze_pcg/stream"})

_SECURITY_HEADERS = (
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cache-control", b"no-store"),
    (
        b"content-security-policy",
        b"default-src 'self'; script-src 'self'; style-src 'self'; "
        b"img-src 'self' data:; media-src blob:; connect-src 'self'; "
        b"object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
        b"form-action 'self'",
    ),
)

# Scalar fields of a healthy dependency that are safe to show in /status.
_SAFE_INFO_KEYS = ("provider", "model", "model_loaded", "configured", "version")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw in (None, "") else float(raw)


@dataclass(frozen=True)
class Settings:
    orchestrator_url: str = "http://agent-orchestrator:8003"
    llm_url: str = "http://llm-server:8001"
    audio_url: str = "http://audio-engine:8002"
    qdrant_url: str = "http://vector-db:6333"
    mlflow_url: str = "http://mlflow-server:5000"
    api_key: str | None = None
    analyze_timeout_seconds: float = 300.0
    status_timeout_seconds: float = 3.0

    @classmethod
    def from_env(cls) -> Settings:
        def url(name: str, default: str) -> str:
            return (os.environ.get(name) or default).rstrip("/")

        defaults = cls()
        return cls(
            orchestrator_url=url("ORCHESTRATOR_URL", defaults.orchestrator_url),
            llm_url=url("LLM_SERVER_URL", defaults.llm_url),
            audio_url=url("AUDIO_ENGINE_URL", defaults.audio_url),
            qdrant_url=url("QDRANT_URL", defaults.qdrant_url),
            mlflow_url=url("MLFLOW_URL", defaults.mlflow_url),
            api_key=os.environ.get("API_KEY") or None,
            analyze_timeout_seconds=_env_float(
                "ANALYZE_TIMEOUT_SECONDS", defaults.analyze_timeout_seconds
            ),
            status_timeout_seconds=_env_float(
                "STATUS_TIMEOUT_SECONDS", defaults.status_timeout_seconds
            ),
        )


class GatewayGuard:
    """Pure-ASGI guard: security headers, API key and body-size checks.

    Runs before FastAPI parses the multipart body so unauthenticated or
    oversized requests are rejected without spooling the upload.
    """

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key.encode() if api_key else None
        self.slots = asyncio.Semaphore(MAX_ACTIVE_ANALYSES)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                present = {name for name, _ in headers}
                headers.extend(h for h in _SECURITY_HEADERS if h[0] not in present)
                message = {**message, "headers": headers}
            await send(message)

        rejection = self._reject(scope)
        if rejection is not None:
            await rejection(scope, receive, send_with_headers)
            return
        if scope["path"] not in _UPLOAD_PATHS or scope["method"] != "POST":
            await self.app(scope, receive, send_with_headers)
            return
        if self.slots.locked():
            await JSONResponse(
                {"detail": "Analysis capacity reached; retry after an active request completes."},
                status_code=429, headers={"Retry-After": "2"},
            )(scope, receive, send_with_headers)
            return
        await self.slots.acquire()
        try:
            await self.app(scope, receive, send_with_headers)
        finally:
            self.slots.release()

    def _reject(self, scope: Scope) -> JSONResponse | None:
        path = scope["path"]
        if path not in _PROTECTED_PATHS:
            return None
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}

        if self.api_key is not None:
            supplied = headers.get("x-api-key", "").encode("latin-1")
            if not supplied:
                scheme, _, token = headers.get("authorization", "").partition(" ")
                if scheme.lower() == "bearer":
                    supplied = token.strip().encode("latin-1")
            if not supplied or not hmac.compare_digest(supplied, self.api_key):
                return JSONResponse(
                    {"detail": "A valid API key is required."},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )

        if path in _UPLOAD_PATHS and scope["method"] == "POST":
            raw_length = headers.get("content-length", "")
            if not raw_length.isdigit():
                return JSONResponse(
                    {"detail": "A Content-Length header is required."},
                    status_code=411,
                )
            if int(raw_length) > MAX_UPLOAD_BYTES + _BODY_SLACK_BYTES:
                return JSONResponse(
                    {"detail": "The recording must be 20 MB or smaller."},
                    status_code=413,
                )
        return None


def _is_wav(content: bytes) -> bool:
    return len(content) >= 44 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"


async def _read_validated_upload(file: UploadFile, prompt: str | None) -> bytes:
    """Return the WAV bytes after validating name, size and signature."""
    if prompt is not None and len(prompt) > MAX_PROMPT_CHARS:
        await file.close()
        raise HTTPException(
            status_code=422,
            detail=f"The prompt must be {MAX_PROMPT_CHARS} characters or fewer.",
        )
    if Path(file.filename or "").suffix.lower() != ".wav":
        await file.close()
        raise HTTPException(status_code=400, detail="Please upload a .wav file.")

    chunks: list[bytes] = []
    size = 0
    try:
        while chunk := await file.read(_READ_CHUNK_BYTES):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="The recording must be 20 MB or smaller.",
                )
            chunks.append(chunk)
    finally:
        await file.close()

    content = b"".join(chunks)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if not _is_wav(content):
        raise HTTPException(
            status_code=400,
            detail="The file is not a valid WAV recording (RIFF/WAVE header missing).",
        )
    return content


def _upstream_error(request_id: str, error: Exception | httpx.Response) -> HTTPException:
    """Map an upstream failure to a generic error; details go to the log only."""
    if isinstance(error, httpx.Response):
        status = error.status_code
        LOGGER.warning("request=%s orchestrator returned HTTP %s", request_id, status)
        if status == 503:
            return HTTPException(503, "The analysis service is not ready.")
        if status == 403:
            return HTTPException(403, "Explicit external processing consent is required.")
        if status in (400, 413, 415, 422):
            return HTTPException(
                422, "The analysis service rejected this recording or request."
            )
        return HTTPException(502, "The analysis service returned an unexpected response.")
    LOGGER.warning(
        "request=%s orchestrator call failed: %s", request_id, type(error).__name__
    )
    if isinstance(error, httpx.TimeoutException):
        return HTTPException(504, "The analysis service timed out.")
    return HTTPException(503, "The analysis service is unreachable.")


def _sse_frame(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def relay_sse(
    upstream: httpx.Response, request_id: str
) -> AsyncIterator[bytes]:
    """Forward the orchestrator's SSE bytes unchanged.

    The only frames the gateway ever adds are a terminal ``error`` and ``done``
    when the upstream connection breaks mid-stream. The upstream response is
    always closed, including when the client disconnects (generator is
    cancelled or closed).
    """
    try:
        async for chunk in upstream.aiter_raw():
            yield chunk
    except (httpx.HTTPError, httpx.StreamError) as error:
        LOGGER.warning(
            "request=%s upstream stream interrupted: %s", request_id, type(error).__name__
        )
        yield _sse_frame(
            "error",
            {
                "code": "upstream_interrupted",
                "message": "The connection to the analysis service was interrupted.",
            },
        )
        yield _sse_frame("done", {})
    finally:
        await upstream.aclose()


def _forward_parts(
    wav: bytes, prompt: str | None, external_consent: bool
) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str]]:
    files = {"file": ("recording.wav", wav, "audio/wav")}
    data = {"external_consent": "true" if external_consent else "false"}
    if prompt and prompt.strip():
        data["prompt"] = prompt.strip()
    return files, data


async def _probe(
    client: httpx.AsyncClient,
    *,
    key: str,
    label: str,
    url: str,
    required: bool,
    needs_model: bool = False,
    timeout: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": key,
        "label": label,
        "required": required,
        "status": "unavailable",
        "detail": "",
        "info": {},
    }
    try:
        response = await client.get(url, timeout=timeout)
    except httpx.TimeoutException:
        result["detail"] = "Timed out."
        return result
    except httpx.RequestError:
        result["detail"] = "Unreachable."
        return result
    if response.status_code != 200:
        result["detail"] = f"Responded with HTTP {response.status_code}."
        return result

    try:
        body = response.json()
    except ValueError:
        body = None
    info = (
        {k: body[k] for k in _SAFE_INFO_KEYS if isinstance(body.get(k), (str, bool, int, float))}
        if isinstance(body, dict)
        else {}
    )
    result["info"] = info
    if needs_model and info.get("model_loaded") is not True:
        result["detail"] = "Reachable, but no model is loaded."
        return result
    result["status"] = "ready"
    result["detail"] = "Ready."
    return result


def create_app(
    settings: Settings | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    current = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(
                current.analyze_timeout_seconds, connect=5.0, pool=5.0
            ),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        try:
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(title="CardioAgent API Gateway", lifespan=lifespan)
    app.state.settings = current
    app.add_middleware(GatewayGuard, api_key=current.api_key)
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    async def status(request: Request) -> JSONResponse:
        client: httpx.AsyncClient = request.app.state.http
        timeout = current.status_timeout_seconds
        probes = [
            _probe(client, key="orchestrator", label="Agent orchestrator",
                   url=f"{current.orchestrator_url}/ready", required=True, timeout=timeout),
            _probe(client, key="audio_engine", label="Audio engine",
                   url=f"{current.audio_url}/health", required=True, needs_model=True,
                   timeout=timeout),
            _probe(client, key="llm_server", label="Language model server",
                   url=f"{current.llm_url}/ready", required=True, timeout=timeout),
            _probe(client, key="vector_db", label="Evidence index (Qdrant)",
                   url=f"{current.qdrant_url}/readyz", required=True, timeout=timeout),
            _probe(client, key="mlflow", label="MLflow tracking",
                   url=f"{current.mlflow_url}/health", required=False, timeout=timeout),
        ]
        dependencies = await asyncio.gather(*probes)
        required_down = any(d["required"] and d["status"] != "ready" for d in dependencies)
        any_down = any(d["status"] != "ready" for d in dependencies)
        overall = "unavailable" if required_down else "degraded" if any_down else "ready"
        return JSONResponse(
            {
                "status": overall,
                "ready": not required_down,
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "dependencies": dependencies,
            },
            status_code=503 if required_down else 200,
        )

    @app.post("/analyze_pcg")
    async def analyze_pcg(
        request: Request,
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        external_consent: bool = Form(False),
    ) -> JSONResponse:
        wav = await _read_validated_upload(file, prompt)
        request_id = uuid.uuid4().hex
        files, data = _forward_parts(wav, prompt, external_consent)
        client: httpx.AsyncClient = request.app.state.http
        try:
            response = await client.post(
                f"{current.orchestrator_url}/analyze",
                files=files,
                data=data,
                headers={"X-Request-ID": request_id},
            )
        except httpx.HTTPError as error:
            raise _upstream_error(request_id, error) from error
        if response.status_code != 200:
            raise _upstream_error(request_id, response)
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            LOGGER.warning("request=%s orchestrator returned non-object JSON", request_id)
            raise HTTPException(502, "The analysis service returned an unexpected response.")
        return JSONResponse(body, headers={"X-Request-ID": request_id})

    @app.post("/analyze_pcg/stream")
    async def analyze_pcg_stream(
        request: Request,
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        external_consent: bool = Form(False),
    ) -> StreamingResponse:
        wav = await _read_validated_upload(file, prompt)
        request_id = uuid.uuid4().hex
        files, data = _forward_parts(wav, prompt, external_consent)
        client: httpx.AsyncClient = request.app.state.http
        upstream_request = client.build_request(
            "POST",
            f"{current.orchestrator_url}/analyze/stream",
            files=files,
            data=data,
            headers={"X-Request-ID": request_id, "Accept": "text/event-stream"},
        )
        try:
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as error:
            raise _upstream_error(request_id, error) from error
        if upstream.status_code != 200:
            await upstream.aclose()
            raise _upstream_error(request_id, upstream)
        if not upstream.headers.get("content-type", "").startswith("text/event-stream"):
            await upstream.aclose()
            LOGGER.warning("request=%s orchestrator stream had wrong content type", request_id)
            raise HTTPException(502, "The analysis service returned an unexpected response.")
        return StreamingResponse(
            relay_sse(upstream, request_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    return app


app = create_app()
