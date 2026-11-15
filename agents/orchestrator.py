"""Internal graph service; no raw recordings or prompts are persisted."""
import asyncio
import contextlib
import json
import time
import uuid
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from agents.graph import build_graph, public_result
from rag_engine.core import Retriever


@asynccontextmanager
async def lifespan(app):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(240, connect=10), limits=httpx.Limits(max_connections=12))
    app.state.retriever = Retriever()
    app.state.slots = asyncio.Semaphore(2)
    yield
    await app.state.http.aclose()
    app.state.retriever.client.close()


app = FastAPI(title="CardioAgent three-agent orchestrator", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "agents": ["AudioAgent", "KnowledgeRAG-Agent", "ReasoningLLM-Agent"]}


@app.get("/ready")
async def ready():
    try:
        status = await run_in_threadpool(app.state.retriever.ready)
        if not status["ready"]:
            raise ValueError("Empty guideline index")
        return status
    except Exception as error:
        raise HTTPException(503, "Guideline index unavailable or empty; run rag_engine.ingest") from error


async def read_input(file, prompt, external_consent):
    try:
        content = await file.read(20 * 1024**2 + 1)
    finally:
        await file.close()
    if len(content) > 20 * 1024**2:
        raise HTTPException(413, "WAV exceeds 20 MiB")
    if not (file.filename or "").lower().endswith(".wav") or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise HTTPException(400, "Upload a valid RIFF/WAVE file")
    return {"content": content, "prompt": prompt, "external_consent": external_consent, "request_id": uuid.uuid4().hex}


def telemetry(seconds, status):
    import os
    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        exp = client.get_experiment_by_name("cardioagent-agent-inference")
        exp_id = exp.experiment_id if exp else client.create_experiment("cardioagent-agent-inference")
        run = client.create_run(exp_id, tags={"status": status})
        client.log_metric(run.info.run_id, "latency_seconds", seconds)
        client.set_terminated(run.info.run_id)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("Agent telemetry unavailable")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), prompt: str = Form("", max_length=2000), external_consent: bool = Form(False)):
    state = await read_input(file, prompt, external_consent)
    async def emit(event, data):
        pass
    try:
        started = time.perf_counter()
        async with app.state.slots:
            result = await build_graph(app.state.http, app.state.retriever, emit).ainvoke(state)
        await run_in_threadpool(telemetry, time.perf_counter() - started, result["status"])
        return public_result(result)
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        raise HTTPException(status if status in {400, 403, 413, 503} else 502, "Analysis dependency rejected request; check model, evidence, provider and consent") from error
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(503, "Analysis unavailable; check model, guideline index and LLM readiness") from error


@app.post("/analyze/stream")
async def analyze_stream(file: UploadFile = File(...), prompt: str = Form("", max_length=2000), external_consent: bool = Form(False)):
    state = await read_input(file, prompt, external_consent)
    queue = asyncio.Queue(maxsize=32)
    async def emit(event, data):
        await queue.put((event, data))
    async def worker():
        started = time.perf_counter()
        try:
            async with app.state.slots:
                result = await build_graph(app.state.http, app.state.retriever, emit).ainvoke(state)
            await emit("result", public_result(result))
            await run_in_threadpool(telemetry, time.perf_counter() - started, result["status"])
        except Exception:
            await emit("error", {"message": "Analysis unavailable. Check deployed model, ESC index, provider and external consent."})
        finally:
            if not asyncio.current_task().cancelling():
                await emit("done", {})
    async def frames():
        task = asyncio.create_task(worker())
        try:
            while True:
                event, data = await queue.get()
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
                if event == "done":
                    break
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    return StreamingResponse(frames(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
