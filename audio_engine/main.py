"""FastAPI service for PCG preprocessing and ONNX inference."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from audio_engine.deployment import Deployment

from audio_engine.preprocessing import (
    DURATION_SECONDS,
    FEATURE_LENGTH,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    TARGET_SAMPLE_RATE,
    PreprocessedAudio,
    extract_mel_features,
)

LOGGER = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    model_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_tracking_enabled: bool

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=Path(
                os.getenv("AUDIO_MODEL_PATH", "/models/cardio_model.onnx")
            ),
            mlflow_tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
            ),
            mlflow_experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "cardioagent-audio-inference"
            ),
            mlflow_tracking_enabled=os.getenv(
                "MLFLOW_TRACKING_ENABLED", "true"
            ).lower()
            in {"1", "true", "yes"},
        )




async def _read_wav(file: UploadFile) -> bytes:
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Please upload a .wav file")
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="WAV exceeds the 20 MiB limit")
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded WAV is empty")
    return content


def _extract_or_400(content: bytes) -> PreprocessedAudio:
    try:
        return extract_mel_features(content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _log_inference(settings: Settings, result: dict) -> None:
    if not settings.mlflow_tracking_enabled:
        return
    try:
        from mlflow import MlflowClient
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        name = settings.mlflow_experiment_name
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(name)
        run = client.create_run(experiment_id, tags={"model_run_id": result["provenance"]["run_id"]})
        client.log_metric(run.info.run_id, "inference_seconds", result["inference_seconds"])
        client.set_terminated(run.info.run_id)
    except Exception:
        LOGGER.warning("Inference telemetry unavailable")


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        current = settings or Settings.from_env()
        app.state.settings = current
        app.state.model = Deployment(current.model_path)
        yield

    application = FastAPI(title="CardioAgent Audio Engine", lifespan=lifespan)

    @application.get("/health")
    async def health(request: Request) -> dict[str, object]:
        current: Settings = request.app.state.settings
        model: Deployment = request.app.state.model
        return {
            "status": "ok",
            "model_loaded": model.session is not None,
            "model_path": str(current.model_path),
            "model_error": model.error,
            "mlflow_tracking_uri": current.mlflow_tracking_uri,
        }

    @application.post("/preprocess")
    async def preprocess(file: UploadFile = File(...)) -> dict[str, object]:
        content = await _read_wav(file)
        audio = await run_in_threadpool(_extract_or_400, content)
        return {
            "status": "processed",
            "original_sample_rate_hz": audio.original_sample_rate_hz,
            "original_duration_seconds": audio.original_duration_seconds,
            "target_sample_rate_hz": TARGET_SAMPLE_RATE,
            "normalized_duration_seconds": DURATION_SECONDS,
            "feature_type": "mel_spectrogram_db",
            "feature_count": int(audio.features.size),
            "feature_min_db": float(audio.features.min()),
            "feature_max_db": float(audio.features.max()),
        }

    @application.post("/predict")
    async def predict(
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
    ) -> dict[str, object]:
        model: Deployment = request.app.state.model
        if model.session is None:
            raise HTTPException(status_code=503, detail=model.error)
        content = await _read_wav(file)
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(model.predict, content)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail="Model inference unavailable") from error
        result["inference_seconds"] = time.perf_counter() - started
        background_tasks.add_task(_log_inference, request.app.state.settings, result)
        return result

    return application


app = create_app()
