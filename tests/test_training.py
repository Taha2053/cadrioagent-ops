import io
from pathlib import Path
import numpy as np
import pytest
from scipy.io import wavfile
from sklearn.tree import DecisionTreeClassifier
from audio_engine.contract import features, VERSION
from audio_engine.deployment import Deployment
from training.data import split
from training.benchmark import export_model, metrics
import json
import hashlib


def recording():
    buffer = io.BytesIO()
    t = np.arange(8000) / 8000
    wavfile.write(buffer, 8000, (np.sin(2*np.pi*120*t)*12000).astype(np.int16))
    return buffer.getvalue()


def test_grouped_splits_never_cross_recording_boundaries():
    rows = [{"group": f"g{i}", "label": i % 2, "source": "physionet"} for i in range(80) for _ in range(2)]
    result = split(rows)
    assignment = {}
    for row in result:
        assignment.setdefault(row["group"], set()).add(row["split"])
    assert all(len(partitions) == 1 for partitions in assignment.values())
    assert {r["split"] for r in result} == {"train", "validation", "test"}


def test_onnx_serving_matches_training_probability(tmp_path):
    x = features(recording())
    X = np.stack([x - 1, x, x + 1, x + 2]).astype(np.float32)
    model = DecisionTreeClassifier(max_depth=2).fit(X, [0, 0, 1, 1])
    path = tmp_path / "cardio_model.onnx"
    assert export_model(model, "mfcc", path, X) < 1e-4
    metadata = {"feature_kind": "mfcc", "feature_contract": VERSION, "threshold": .5,
        "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "name": "decision_tree", "run_id": "unit-test", "subset": True}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    result = Deployment(path).predict(recording())
    assert result["classification"] == "normal"
    assert result["probability_abnormal"] == pytest.approx(model.predict_proba(x[None])[0,1])
    path.write_bytes(b"corrupted")
    assert Deployment(path).session is None


def test_silence_and_nonfinite_are_not_valid_model_inputs():
    buffer = io.BytesIO()
    wavfile.write(buffer, 2000, np.zeros(1000, dtype=np.int16))
    with pytest.raises(ValueError, match="Silent"):
        features(buffer.getvalue())
    with pytest.raises(ValueError, match="RIFF"):
        features(b"not audio")


def test_metrics_expose_degenerate_all_positive_classifier():
    score = metrics([0,0,1,1], [.9,.9,.9,.9])
    assert score["recall_abnormal"] == 1
    assert score["specificity"] == 0
    assert score["fp"] == 2
