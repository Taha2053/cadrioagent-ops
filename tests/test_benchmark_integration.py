"""A non-activating experiment must not replace the deployed or registered champion."""
import csv
import io
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from audio_engine.deployment import Deployment


def test_experiment_without_activation_preserves_champion(tmp_path):
    directory = tmp_path / "datasets" / "PhysioNet" / "training-b"
    directory.mkdir(parents=True)
    rows = []
    recordings = {}
    for index in range(80):
        label = index % 2
        time = np.arange(2000) / 2000
        frequency = 80 if label == 0 else 250
        signal = np.sin(2 * np.pi * frequency * time + index * .01)
        buffer = io.BytesIO()
        wavfile.write(buffer, 2000, (signal * 12000).astype(np.int16))
        content = buffer.getvalue()
        name = f"b{index:04}"
        (directory / f"{name}.wav").write_bytes(content)
        rows.append([name, 1 if label else -1])
        recordings[label] = content
    with (directory / "REFERENCE.csv").open("w", newline="") as output:
        csv.writer(output).writerows(rows)
    vault = tmp_path / "vault"
    vault.mkdir()
    tracking = f"sqlite:///{vault}/tracking.db"
    command = [sys.executable, "-m", "training.benchmark", "--dataset", str(directory.parents[1]),
               "--vault", str(vault), "--tracking-uri", tracking,
               "--models", "decision_tree", "--trials", "1"]
    environment = os.environ | {"MLFLOW_DISABLE_AGENT_HINT": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                                "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    subprocess.run([*command, "--activate"], check=True, capture_output=True, text=True,
                   cwd=tmp_path, env=environment, timeout=60)
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=tracking)
    first = client.get_model_version_by_alias("CardioAgentAudio", "champion").version
    metadata = (vault / "models" / "cardio_model.json").read_bytes()
    deployed = Deployment(vault / "models" / "cardio_model.onnx")
    assert deployed.predict(recordings[0])["classification"] == "normal"
    assert deployed.predict(recordings[1])["classification"] == "abnormal"
    subprocess.run(command, check=True, capture_output=True, text=True,
                   cwd=tmp_path, env=environment, timeout=60)
    assert len(client.search_model_versions("name='CardioAgentAudio'")) == 2
    assert client.get_model_version_by_alias("CardioAgentAudio", "champion").version == first
    assert (vault / "models" / "cardio_model.json").read_bytes() == metadata
