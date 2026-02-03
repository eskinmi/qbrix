FROM python:3.10.16-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN pip install uv

WORKDIR /workspace

# copy workspace root files
COPY pyproject.toml uv.lock ./

# copy lib dependencies (required by tool.uv.sources)
COPY lib/store lib/store
COPY lib/proto lib/proto
COPY lib/log lib/log

# copy service
COPY ee/svc/trace ee/svc/trace

# install dependencies
RUN uv sync --frozen --no-editable --package tracesvc

FROM python:3.10.16-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends curl && apt-get clean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
RUN curl -sSL https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz | tar -xz -C /usr/local/bin
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# copy virtual environment from builder
COPY --from=builder /workspace/.venv /app/.venv

# copy service source
COPY ee/svc/trace/src ./src

ENV PATH="/app/.venv/bin:$PATH"

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 50053

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:50053/health || exit 1

CMD ["python", "-m", "tracesvc.cli"]
