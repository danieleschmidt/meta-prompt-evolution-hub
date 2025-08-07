# Multi-stage Dockerfile for Sentiment Analyzer Pro
# Optimized for production with security and performance

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Labels for metadata
LABEL maintainer="sentiment-analyzer-team@company.com" \
      org.opencontainers.image.title="Sentiment Analyzer Pro" \
      org.opencontainers.image.description="Evolutionary prompt-optimized sentiment analysis service" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/company/sentiment-analyzer-pro"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /build

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir build && \
    python -m build --wheel

# Production stage
FROM python:3.11-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    HOME="/home/appuser"

# Set work directory
WORKDIR /app

# Copy wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the application
RUN pip install --user --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Copy application code
COPY --chown=appuser:appuser sentiment_analyzer.py ./
COPY --chown=appuser:appuser robust_sentiment_analyzer.py ./
COPY --chown=appuser:appuser scalable_sentiment_analyzer.py ./
COPY --chown=appuser:appuser standalone_sentiment_demo.py ./

# Copy deployment scripts
COPY --chown=appuser:appuser deployment/scripts/ ./scripts/
COPY --chown=appuser:appuser production_api_server.py ./

# Create necessary directories
RUN mkdir -p /app/cache /app/logs /tmp/app && \
    chown -R appuser:appuser /app /tmp/app

# Security: Remove unnecessary packages and clean up
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8000 8080 9090

# Default command
CMD ["python", "production_api_server.py"]