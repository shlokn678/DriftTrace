# DriftTrace project image (FR-7 serving, FR-13 monitor, FR-11.5 webhook stub).
# A single image whose entrypoint selects the role: api | monitor | webhook | cli.
# Production-appropriate: slim base, non-root user, only required runtime deps,
# uvicorn ASGI server, environment-driven config. No cloud credentials (NFR-7).

FROM python:3.13-slim AS base

# System hygiene + build tools needed for confluent-kafka (librdkafka) wheels.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the runtime extras: serving (FastAPI/uvicorn/httpx) + streaming
# (confluent-kafka) + tracking (MLflow, to load the registered model) + explain
# (SHAP/LIME, for the /explain endpoint).
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[serving,streaming,tracking,explain]"

# Copy the entrypoint and make it executable.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run as a non-root user (least privilege).
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/artifacts /app/reports /store \
    && chown -R appuser:appuser /app /store
USER appuser

# Environment-driven defaults; overridden by Compose. No secrets.
ENV DRIFTTRACE_API_PORT=8000 \
    DRIFTTRACE_WEBHOOK_PORT=9000

EXPOSE 8000 9000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["api"]
