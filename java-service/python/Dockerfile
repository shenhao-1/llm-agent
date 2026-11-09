FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY config ./config
RUN pip install --no-cache-dir .
ENV PYTHONPATH=/app:/app/src
CMD ["python", "-m", "scripts.internal_ingest_api"]
