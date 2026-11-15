from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from audio_engine.main import Settings, create_app
from audio_engine.preprocessing import FEATURE_LENGTH, TARGET_SAMPLE_RATE, extract_mel_features


def wav_bytes(sample_rate: int = 8_000, seconds: float = 1.0) -> bytes:
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    signal = (0.4 * np.sin(2 * np.pi * 120 * time) * np.iinfo(np.int16).max).astype(
        np.int16
    )
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, signal)
    return buffer.getvalue()


def service_settings() -> Settings:
    return Settings(
        model_path=Path("/missing/cardio_model.onnx"),
        mlflow_tracking_uri="http://mlflow.test:5000",
        mlflow_experiment_name="test",
        mlflow_tracking_enabled=False,
    )


def test_preprocessing_resamples_and_produces_fixed_notebook_vector() -> None:
    result = extract_mel_features(wav_bytes())

    assert result.original_sample_rate_hz == 8_000
    assert result.features.shape == (FEATURE_LENGTH,)
    assert result.features.dtype == np.float32
    assert np.isfinite(result.features).all()


def test_preprocess_endpoint_and_missing_model_contract() -> None:
    app = create_app(service_settings())
    with TestClient(app) as client:
        health = client.get("/health")
        processed = client.post(
            "/preprocess",
            files={"file": ("heart.wav", wav_bytes(), "audio/wav")},
        )
        prediction = client.post(
            "/predict",
            files={"file": ("heart.wav", wav_bytes(), "audio/wav")},
        )

    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert processed.status_code == 200
    assert processed.json()["target_sample_rate_hz"] == TARGET_SAMPLE_RATE
    assert processed.json()["feature_count"] == FEATURE_LENGTH
    assert prediction.status_code == 503


def test_invalid_audio_is_rejected() -> None:
    app = create_app(service_settings())
    with TestClient(app) as client:
        response = client.post(
            "/preprocess",
            files={"file": ("broken.wav", b"not a wav", "audio/wav")},
        )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("Cannot decode WAV audio")
