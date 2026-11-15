"""Publish a verified notebook/offline artifact to the running MLflow registry."""
import argparse
import json
from pathlib import Path
import mlflow
import onnx
from training.benchmark import activate, verify_artifact


def promote(artifact: Path, tracking_uri: str, vault: Path):
    metadata = verify_artifact(artifact)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cardioagent-audio-promotion")
    with mlflow.start_run(run_name=f"promote-{metadata['name']}") as run:
        mlflow.set_tags({"training_run_id": metadata["run_id"],
                         "training_parent_run_id": metadata["parent_run_id"],
                         "artifact_sha256": metadata["onnx_sha256"],
                         "subset": str(metadata["subset"]).lower()})
        mlflow.log_params({"feature_contract": metadata["feature_contract"],
                           "minimum_specificity": metadata["minimum_specificity"]})
        mlflow.log_metrics({f"validation_{key}": value for key, value in metadata["validation"].items()})
        mlflow.log_artifact(str(artifact.with_suffix(".json")))
        benchmark = artifact.parent / "benchmark.json"
        if benchmark.exists():
            report = json.loads(benchmark.read_text())
            mlflow.log_artifact(str(benchmark))
            for model in report["models"]:
                mlflow.log_metrics({f"{model['name']}_test_{key}": value for key, value in model["test"].items()})
        registered = mlflow.onnx.log_model(onnx.load(str(artifact)), name="champion",
                                          registered_model_name="CardioAgentAudio")
        # Local source provenance remains immutable; registry tags link back to the original experiment.
        if artifact.resolve() != (vault / "models" / "cardio_model.onnx").resolve():
            activate(artifact, vault)
        version = registered.registered_model_version
        mlflow.MlflowClient().set_registered_model_alias("CardioAgentAudio", "champion", str(version))
        result = {"registry_uri": tracking_uri, "registry_run_id": run.info.run_id,
                  "model": "CardioAgentAudio", "version": version,
                  "training_run_id": metadata["run_id"], "onnx_sha256": metadata["onnx_sha256"]}
        destination = vault / "models" / "registry.json"
        destination.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--vault", type=Path, default=Path("vault"))
    args = parser.parse_args()
    promote(args.artifact, args.tracking_uri, args.vault)


if __name__ == "__main__":
    main()
