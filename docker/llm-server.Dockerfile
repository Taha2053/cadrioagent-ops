FROM ghcr.io/astral-sh/uv:0.9.18 AS uv
FROM python:3.13-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv export --locked --only-group llm-runtime --no-emit-project -o /tmp/runtime.txt \
    && uv pip install --python /usr/local/bin/python --prefix /install --require-hashes -r /tmp/runtime.txt

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder /install /usr/local
COPY llm_server/ ./llm_server/
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8001
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8001"]
