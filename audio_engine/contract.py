"""Versioned train/serve contract; never fit preprocessing on evaluation data."""
from __future__ import annotations
import io
import librosa
import numpy as np
import soundfile as sf

VERSION = "pcg-2000hz-5s-mfcc40-mel96-v1"
RATE = 2000
SAMPLES = 10000


def decode(content: bytes) -> np.ndarray:
    if len(content) < 44 or content[:4] != b"RIFF" or content[8:12] != b"WAVE":
        raise ValueError("Expected a RIFF/WAVE recording")
    with sf.SoundFile(io.BytesIO(content)) as file:
        if not 1000 <= file.samplerate <= 192000 or file.channels > 8:
            raise ValueError("Unsupported sample rate or channel count")
        if file.frames == 0:
            raise ValueError("Recording contains no audio samples")
        signal = file.read(frames=file.samplerate * 5, dtype="float32", always_2d=True).mean(axis=1)
        rate = file.samplerate
    if not np.isfinite(signal).all():
        raise ValueError("Non-finite audio samples")
    if rate != RATE:
        signal = librosa.resample(signal, orig_sr=rate, target_sr=RATE)
    signal = signal[:SAMPLES]
    signal = signal - signal.mean()
    peak = np.max(np.abs(signal))
    if peak < 1e-6:
        raise ValueError("Silent recording; upload an audible heart-sound recording")
    return librosa.util.fix_length(signal / peak, size=SAMPLES)


def features(content: bytes, kind: str = "mfcc") -> np.ndarray:
    signal = decode(content)
    mel = librosa.feature.melspectrogram(y=signal, sr=RATE, n_fft=256,
        hop_length=105, n_mels=96, fmin=20, fmax=1000)
    db = librosa.power_to_db(mel, ref=np.max)
    if kind == "mel":
        return librosa.util.fix_length(db, size=96, axis=1).astype(np.float32)[..., None] / 80.0
    if kind != "mfcc":
        raise ValueError(f"Unknown feature kind: {kind}")
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=40)
    zcr = librosa.feature.zero_crossing_rate(signal)
    rms = librosa.feature.rms(y=signal)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1),
        [zcr.mean(), zcr.std(), rms.mean(), rms.std()]]).astype(np.float32)
