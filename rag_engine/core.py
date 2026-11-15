"""Lazy CPU embeddings, page-aware chunks, and evidence-only retrieval."""
from __future__ import annotations
import hashlib
import os
import uuid
from pathlib import Path


def chunks(text: str, size=1200, overlap=180):
    if size <= overlap or overlap < 0:
        raise ValueError("Chunk size must exceed nonnegative overlap")
    text = " ".join(text.split())
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        yield text[start:end]
        if end == len(text):
            break
        start = end - overlap


class Retriever:
    def __init__(self, url=None, collection=None, cache=None):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(url=url or os.getenv("QDRANT_URL", "http://localhost:6333"), timeout=10)
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "esc_guidelines")
        self.cache = cache or os.getenv("FASTEMBED_CACHE_PATH", "vault/cache/fastembed")
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name, cache_dir=self.cache, threads=2)
        return self._model

    def ready(self):
        info = self.client.get_collection(self.collection)
        return {"ready": bool(info.points_count), "points": info.points_count,
            "collection": self.collection, "embedding_model": self.model_name}

    def search(self, query, limit=4):
        if not self.ready()["ready"]:
            return []
        vector = next(self.model.query_embed(query)).tolist()
        points = self.client.query_points(collection_name=self.collection, query=vector, limit=limit,
            score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.55")), with_payload=True).points
        return [dict(point.payload, id=str(point.id), score=float(point.score)) for point in points]

    def ingest(self, documents, source_manifest):
        from pypdf import PdfReader
        from qdrant_client import models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
        count = 0
        for path in sorted(Path(documents).glob("*.pdf")):
            source = source_manifest.get(path.name)
            if not source or not source.get("url", "").startswith("https://"):
                raise ValueError(f"Missing source provenance for {path.name}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            points = []
            for page, pdf_page in enumerate(PdfReader(path).pages, 1):
                page_chunks = [(index, text) for index, text in enumerate(chunks(pdf_page.extract_text() or "")) if len(text) >= 80]
                if not page_chunks:
                    continue
                vectors = self.model.passage_embed([text for _, text in page_chunks], batch_size=16)
                for (index, text), vector in zip(page_chunks, vectors):
                    identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{digest}:{page}:{index}"))
                    vector = vector.tolist()
                    points.append(models.PointStruct(id=identifier, vector=vector, payload={"text": text,
                        "source": path.name, "page": page, "url": source["url"], "sha256": digest,
                        "download_url": source.get("download_url", source["url"]),
                        "embedding_model": self.model_name}))
                    if len(points) >= 64:
                        self.client.upsert(self.collection, points, wait=True)
                        count += len(points)
                        points = []
            if points:
                self.client.upsert(self.collection, points, wait=True)
                count += len(points)
        if count == 0:
            raise ValueError("No indexable guideline text; provide ESC PDFs and source manifest")
        return count
