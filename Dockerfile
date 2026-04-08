# Stage 1: Build Go engine
FROM golang:1.22-alpine AS go-builder
WORKDIR /build
COPY engine/ ./
RUN go mod download && \
    CGO_ENABLED=0 GOOS=linux go build -ldflags='-s -w' -o /go-engine ./cmd/server

# Stage 2: Python runtime
FROM python:3.12-slim AS runner
WORKDIR /app

# Install system deps for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy Go binary
COPY --from=go-builder /go-engine /app/go-engine
RUN chmod +x /app/go-engine

# Install Python deps using uv (Hackathon mandatory requirement)
RUN pip install --no-cache-dir uv && \
    uv pip install --system openenv-core fastapi uvicorn httpx openai

# Copy application code
COPY models.py ./
COPY __init__.py ./
COPY client.py ./
COPY inference.py ./
COPY wegh_graders.py ./
COPY server/ ./server/
COPY openenv.yaml ./
COPY entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh

# Health check for HF Spaces
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
