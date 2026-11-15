"""Audio transformations shared by API inference and future ONNX export."""

from __future__ import annotations

import io
from dataclasses import dataclass

import librosa
import numpy as np

TARGET_SAMPLE_RATE = 2_000
DURATION_SECONDS = 5
TARGET_SAMPLES = TARGET_SAMPLE_RATE * DURATION_SECONDS
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 128
FEATURE_LENGTH = 1_200


@dataclass(frozen=True)
class PreprocessedAudio:
    features: np.ndarray
    original_sample_rate_hz: int
    original_duration_seconds: float


def extract_mel_features(wav_bytes: bytes) -> PreprocessedAudio:
    """Decode WAV, normalize it, and mirror the TP Mel-vector extraction."""
    if not wav_bytes:
        raise ValueError("The uploaded WAV is empty")

    try:
        signal, original_rate = librosa.load(
            io.BytesIO(wav_bytes),
            sr=None,
            mono=True,
        )
    except Exception as error:
        raise ValueError(f"Cannot decode WAV audio: {error}") from error

    if signal.size == 0:
        raise ValueError("The uploaded WAV contains no audio samples")
    if not np.isfinite(signal).all():
        raise ValueError("The uploaded WAV contains non-finite samples")

    original_duration = signal.size / float(original_rate)
    if original_rate != TARGET_SAMPLE_RATE:
        signal = librosa.resample(
            signal,
            orig_sr=original_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )

    signal = librosa.util.fix_length(signal, size=TARGET_SAMPLES)
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal / peak

    spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    decibels = librosa.power_to_db(spectrogram, ref=np.max)
    features = np.asarray(decibels.flatten()[:FEATURE_LENGTH], dtype=np.float32)
    if features.size != FEATURE_LENGTH:
        raise ValueError(
            f"Expected {FEATURE_LENGTH} Mel values, extracted {features.size}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Audio preprocessing produced non-finite features")

    return PreprocessedAudio(
        features=features,
        original_sample_rate_hz=int(original_rate),
        original_duration_seconds=original_duration,
    )
