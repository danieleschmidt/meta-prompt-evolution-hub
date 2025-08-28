# Progressive Quality Gates - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Copy application code
COPY progressive_quality_gates_*.py .
COPY meta_prompt_evolution/ ./meta_prompt_evolution/

# Create non-root user
RUN useradd -m -u 1000 quality-gates && \
    chown -R quality-gates:quality-gates /app
USER quality-gates

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import progressive_quality_gates_gen1; print('OK')" || exit 1

# Default command
CMD ["python3", "progressive_quality_gates_comprehensive.py"]

# Labels for metadata
LABEL version="1.0.0" \
      description="Progressive Quality Gates System" \
      maintainer="Terragon Labs" \
      org.opencontainers.image.source="https://github.com/terragon/progressive-quality-gates"
