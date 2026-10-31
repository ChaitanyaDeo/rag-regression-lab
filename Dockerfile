# syntax=docker/dockerfile:1
#
# RAG Regression Lab: the `ragregress` CLI in a container (Day 13 B2b).
#
#   docker compose build
#   docker compose run --rm ragregress run --config configs/baseline.yaml --dry-run
#
# What is in the image: Python 3.11, CPU-only torch, the pinned runtime dependencies
# (requirements.txt) and the ragregress package. What is NOT in the image: data/, configs/, the
# database, the Chroma index, the reports, .env and every model. docker-compose.yml mounts the repo
# root at /work (data, configs, db, logs, reports), a named volume for the Chroma index and one
# for the HuggingFace cache. Ollama runs on the host, not in a container.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    HF_HOME=/hf-cache

# Non-root user. /work is the working directory (compose bind-mounts the repo root over it);
# /work/artifacts/chroma and /hf-cache are volume mount points. They are created and owned by
# `app` here so that a new named volume inherits that owner on first use.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin app \
    && mkdir -p /work/artifacts/chroma /hf-cache \
    && chown -R app:app /work /hf-cache

WORKDIR /app

# Layer 1, dependencies (changes rarely, so it stays cached while the code changes).
# torch comes first from the CPU-only index: the default Linux wheel is the CUDA build (several GB).
# The version is read from requirements.txt, so there is one pin only. It is its own RUN, so a network
# failure in the second step does not throw the large torch download away: a retry reuses this layer.
# Then the pinned requirements: torch==X matches the installed X+cpu, so it is not installed again.
# PIP_DEFAULT_TIMEOUT / PIP_RETRIES (above) make pip patient on a slow or flaky connection.
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep -E '^torch==' requirements.txt)"
# This one is the slow, failure-prone step on a poor connection, so:
#  - a BuildKit cache mount keeps every file pip has downloaded (wheels, metadata) between attempts and
#    between failed builds; PIP_NO_CACHE_DIR is unset here so pip uses it (the mount is not in the image);
#  - a retry loop re-runs pip after a stalled download (PIP_RETRIES does not cover a stall mid-download):
#    each attempt only fetches what is still missing.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    unset PIP_NO_CACHE_DIR; \
    for attempt in 1 2 3 4 5; do \
        pip install -r requirements.txt && exit 0; \
        echo "pip install failed (attempt $attempt of 5); retrying with what is cached"; \
        sleep 5; \
    done; \
    exit 1

# Layer 2, the package. --no-deps: everything it needs is already installed above.
COPY pyproject.toml ./
COPY ragregress ./ragregress
RUN pip install --no-deps . \
    && rm -rf /app/ragregress /app/build

USER app
WORKDIR /work

ENTRYPOINT ["ragregress"]
CMD ["--help"]