"""Optional deep runtime: an exported model must accept only canonical Mel input."""
import json
import hashlib
import pytest

pytest.importorskip("tensorflow", reason="Run with the deep-learning extra to verify Keras exports")
pytest.importorskip("tf2onnx")
from audio_engine.contract import VERSION, features
from audio_engine.deployment import Deployment
from training.benchmark import deep_model, export_model
from tests.test_training import recording


@pytest.mark.parametrize("name", ["cnn", "resnet50", "transformer_cnn"])
def test_deep_export_serves_canonical_audio_without_hidden_inputs(tmp_path, name):
    import onnxruntime as ort
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    model, _ = deep_model(name)
    sample = features(recording(), "mel")[None, ...]
    expected = float(model(sample, training=False).numpy().ravel()[0])
    path = tmp_path / f"{name}.onnx"
    error = export_model(model, "mel", path, sample)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [value.name for value in session.get_inputs()] == ["features"]
    path.with_suffix(".json").write_text(json.dumps({
        "feature_contract": VERSION, "feature_kind": "mel", "threshold": .5,
        "onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "name": name, "run_id": "deep-export-regression", "subset": True}))
    result = Deployment(path).predict(recording())
    assert result["probability_abnormal"] == pytest.approx(expected, abs=1e-4)
    assert error < 1e-4
    tf.keras.backend.clear_session()
