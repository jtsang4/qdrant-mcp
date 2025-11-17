# Multi-stage Dockerfile for better-qdrant-mcp
# Stage 1: Build stage with uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

# Set working directory
WORKDIR /app

# Copy dependency definition (keeps build cache stable when code changes)
COPY pyproject.toml ./

# Create virtual environment and sync dependencies into it
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv sync --no-dev

# Copy application source code and install the project into the venv
COPY src/ ./src/
RUN VIRTUAL_ENV=/opt/venv uv pip install --no-cache .

# Stage 2: Runtime stage
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user
RUN useradd --create-home --shell /bin/bash mcp
WORKDIR /home/mcp

# Copy application code (only source, not docs/dev files)
COPY --chown=mcp:mcp src/ ./src/
COPY --chown=mcp:mcp pyproject.toml ./

# Switch to non-root user
USER mcp

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${MCP_PORT:-8000}/health || exit 1

# Default command (will be overridden by docker-compose)
CMD ["python", "-m", "better_qdrant_mcp"]