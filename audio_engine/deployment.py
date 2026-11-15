"""Load only hash-verified, version-compatible research artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
from audio_engine.contract import VERSION, features


class Deployment:
    def __init__(self, path: Path):
        self.session = None
        self.error = "No trained model deployed; run the benchmark with --activate"
        self.metadata = {}
        try:
            import onnxruntime as ort
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata["feature_contract"] != VERSION:
                raise ValueError("Feature contract mismatch")
            if metadata["onnx_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError("Model integrity check failed")
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.metadata = metadata
            self.error = None
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.error = str(error)

    def predict(self, content: bytes) -> dict:
        if self.session is None:
            raise RuntimeError(self.error)
        array = features(content, self.metadata["feature_kind"])[None, ...]
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: array})
        probability = float(np.asarray(outputs[1])[0, 1] if self.metadata["feature_kind"] == "mfcc" else np.asarray(outputs[0]).ravel()[0])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Model produced an invalid probability")
        return {"classification": "abnormal" if probability >= self.metadata["threshold"] else "normal",
            "probability_abnormal": probability,
            "provenance": {key: self.metadata[key] for key in
                ("name", "run_id", "feature_contract", "onnx_sha256", "subset")},
            "warning": "Educational/research output only; not a medical diagnosis."}
