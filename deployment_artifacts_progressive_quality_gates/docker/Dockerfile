FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY progressive_quality_gates_gen1.py .
COPY progressive_quality_gates_gen2.py .
COPY progressive_quality_gates_gen3.py .
COPY comprehensive_progressive_quality_validation.py .
COPY progressive_quality_gates_deployment.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check endpoints
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health/live || exit 1

# Expose ports
EXPOSE 8080 9090

# Run application
CMD ["python", "-m", "uvicorn", "progressive_quality_gates_deployment:app", "--host", "0.0.0.0", "--port", "8080"]
