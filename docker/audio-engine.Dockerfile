FROM ghcr.io/astral-sh/uv:0.9.18 AS uv
FROM python:3.13-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv export --locked --only-group audio-runtime --no-emit-project -o /tmp/runtime.txt \
    && uv pip install --python /usr/local/bin/python --prefix /install --require-hashes -r /tmp/runtime.txt

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NUMBA_CACHE_DIR=/tmp/numba
RUN apt-get update && apt-get install --yes --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /install /usr/local
COPY audio_engine/ ./audio_engine/
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8002
CMD ["uvicorn", "audio_engine.main:app", "--host", "0.0.0.0", "--port", "8002"]
