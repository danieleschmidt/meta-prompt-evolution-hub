FROM python:3.12-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install dependencies
WORKDIR /app
COPY pyproject.toml ./
RUN pip install -e .

# Copy application
COPY . .
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Start application
CMD ["python", "-m", "meta_prompt_evolution.cli", "server", "--port", "8080"]
