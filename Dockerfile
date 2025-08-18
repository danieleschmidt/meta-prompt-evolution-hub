# Development Dockerfile for Meta-Prompt-Evolution-Hub
FROM python:3.11-slim-bullseye

# Set build arguments
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=devuser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create user with same UID/GID as host (for volume permissions)
RUN groupadd -g $GROUP_ID $USERNAME && \
    useradd -m -u $USER_ID -g $GROUP_ID -s /bin/bash $USERNAME

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements*.txt ./
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.prod.txt
RUN pip install --no-cache-dir -e ".[dev,test]"

# Copy application code
COPY --chown=$USERNAME:$USERNAME . .

# Switch to non-root user
USER $USERNAME

# Set environment variables
ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=development

# Expose common development ports
EXPOSE 8080 8265 5432 6379 9090 3000

# Default command
CMD ["bash"]