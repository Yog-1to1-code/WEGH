# ─────────────────────────────────────────────────────────────────────────────
# WEGH — Multi-stage Docker Build
# Stage 1: Compile Go simulation engine (self-contained OpenEnv server)
# Stage 2: Runtime with Go binary + Python inference dependencies
# ─────────────────────────────────────────────────────────────────────────────

# Stage 1: Build Go engine
FROM mirror.gcr.io/library/golang:1.22-alpine AS go-builder
WORKDIR /build
COPY engine/ ./
RUN go mod download && \
    CGO_ENABLED=0 GOOS=linux go build -ldflags='-s -w' -o /go-engine ./cmd/server

# Stage 2: Runtime
FROM mirror.gcr.io/library/python:3.12-slim AS runner
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl supervisor && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user

# Copy Go binary
COPY --from=go-builder /go-engine /app/go-engine
RUN chmod +x /app/go-engine

# Install Python deps (with fallback for reliability)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir openai httpx pydantic requests fastapi uvicorn python-dotenv

# Copy application code
COPY openenv.yaml ./
COPY models.py ./
COPY __init__.py ./
COPY client.py ./
COPY inference.py ./
COPY wegh_graders.py ./
COPY server/ ./server/
COPY dashboard/ ./dashboard/

# Supervisord config (Go server + Dashboard)
COPY supervisord.conf /etc/supervisor/conf.d/wegh.conf

# Set permissions
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Health check against Go server
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

# Run supervisord (manages Go server)
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/wegh.conf"]
