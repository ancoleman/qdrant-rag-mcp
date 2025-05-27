# Use Python 3.12 for better performance and compatibility with local development
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies required for ML libraries and compilation
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Failed to install requirements, trying with --use-deprecated=legacy-resolver" && \
     pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt)
     
# Make sure python-dotenv is installed for loading environment variables from .env
RUN pip install --no-cache-dir python-dotenv

# Copy source code and env file
COPY src/ ./src/
COPY config/ ./config/
COPY .env .env

# Create necessary directories with proper permissions
RUN mkdir -p logs data && chmod 755 logs data

# Set Python path and optimize Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Removed health check since it's causing issues

# Run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Run the context-aware server
CMD ["python", "src/qdrant_mcp_context_aware.py"]