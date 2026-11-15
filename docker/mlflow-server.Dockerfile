FROM ghcr.io/mlflow/mlflow:v3.16.1
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", "--allowed-hosts", "localhost:*,127.0.0.1:*,mlflow-server:*"]
