FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy project definition first (layer caching)
COPY pyproject.toml ./

# Install dependencies via uv (uses lock file if present)
RUN uv sync --no-dev --frozen || uv sync --no-dev

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Copy data and models (mount via volumes in production)
COPY data/raw/ data/raw/

# Create output directories
RUN mkdir -p data/processed models/tft mlruns

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run API server
CMD ["uv", "run", "python", "-m", "src.serving.api"]
