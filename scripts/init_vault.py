"""Initialize local research storage without overwriting configuration or data."""
from pathlib import Path
import os


def main() -> None:
    vault = Path(os.getenv("VAULT_DIR", "vault"))
    for relative in (
        "data", "manifests", "features", "models", "runs", "reports",
        "documents/esc", "rag", "cache/fastembed", "evaluations", "exports",
    ):
        directory = vault / relative
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
    config = Path(".env")
    if not config.exists():
        config.write_bytes(Path(".env.example").read_bytes())
        config.chmod(0o600)
    print(f"Research vault ready: {vault.resolve()}")
    print("Secrets stay in .env, not the vault. Datasets and model weights stay out of Git.")


if __name__ == "__main__":
    main()
