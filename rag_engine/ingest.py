"""Fetch explicitly curated public ESC documents and index with provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import httpx
from rag_engine.core import Retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", default="vault")
    parser.add_argument("--sources", default=str(Path(__file__).with_name("sources.json")))
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    vault = Path(args.vault)
    directory = vault / "documents" / "esc"
    directory.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(args.sources).read_text())
    if args.fetch:
        with httpx.Client(timeout=90, follow_redirects=True) as client:
            for filename, source in sources.items():
                if Path(filename).name != filename:
                    raise ValueError("Source filenames cannot contain paths")
                path = directory / filename
                if not path.exists():
                    download_url = source.get("download_url", source["url"])
                    if not download_url.startswith("https://"):
                        raise ValueError("Source downloads require HTTPS")
                    response = client.get(download_url)
                    response.raise_for_status()
                    if not response.content.startswith(b"%PDF") or len(response.content) > 30 * 1024**2:
                        raise ValueError("Expected a PDF smaller than 30 MiB")
                    path.write_bytes(response.content)
                source["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (vault / "rag").mkdir(parents=True, exist_ok=True)
    (vault / "rag" / "sources.json").write_text(json.dumps(sources, indent=2))
    count = Retriever().ingest(directory, sources)
    print(json.dumps({"indexed_chunks": count, "sources": sources}, indent=2))

if __name__ == "__main__":
    main()
