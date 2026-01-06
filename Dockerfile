# =============================================================================
# DUMP TRUCK ANOMALY DETECTION SYSTEM - PRODUCTION DOCKERFILE
# =============================================================================
# Base: Python 3.11 slim (Debian-based, smaller footprint than full image)
# Why slim: Reduces image size by ~400MB while retaining essential packages
# =============================================================================

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# METADATA
# ---------------------------------------------------------------------------
LABEL maintainer="Sanskar P"
LABEL description="Dump Truck Multi-Interface Anomaly Detection System"
LABEL version="1.0"

# ---------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------------------------
# Prevents Python from writing .pyc files (reduces container size)
ENV PYTHONDONTWRITEBYTECODE=1

# Ensures Python output is sent straight to terminal (better logging in Docker)
ENV PYTHONUNBUFFERED=1

# Suppress TensorFlow verbose logging (info/warnings)
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default port (can be overridden at runtime)
ENV PORT=5000

# ---------------------------------------------------------------------------
# SYSTEM DEPENDENCIES
# ---------------------------------------------------------------------------
# Install minimal system packages required for Python ML libraries
# - libgomp1: Required for TensorFlow/NumPy parallel operations
# - libhdf5-dev: Required for loading .h5 Keras model files
# Clean up apt cache to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ---------------------------------------------------------------------------
# WORKING DIRECTORY
# ---------------------------------------------------------------------------
# Create and set working directory
WORKDIR /app

# ---------------------------------------------------------------------------
# PYTHON DEPENDENCIES
# ---------------------------------------------------------------------------
# Copy requirements first (Docker layer caching optimization)
# If requirements.txt hasn't changed, this layer is cached
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't store pip cache (reduces image size)
# --upgrade pip: Ensure latest pip for compatibility
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# APPLICATION CODE & ASSETS
# ---------------------------------------------------------------------------
# Copy ML model files (.h5 for Keras, .pkl for scikit-learn)
COPY *.h5 ./
COPY *.pkl ./

# Copy dataset files (required for inference/reference)
COPY *.csv ./

# Copy HTML templates (Flask templates in root directory)
COPY *.html ./

# Copy static assets (images)
COPY *.png ./

# Copy main application code
COPY app.py ./

# ---------------------------------------------------------------------------
# EXPOSE PORT
# ---------------------------------------------------------------------------
# Document the port the container listens on
# Actual binding happens at runtime via Gunicorn
EXPOSE 5000

# ---------------------------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------------------------
# Verify the application is responding
# Interval: Check every 30 seconds
# Timeout: Fail if no response in 10 seconds
# Retries: Mark unhealthy after 3 consecutive failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/')" || exit 1

# ---------------------------------------------------------------------------
# RUN COMMAND
# ---------------------------------------------------------------------------
# Use Gunicorn as production WSGI server (as defined in Procfile)
# --bind 0.0.0.0:$PORT: Listen on all interfaces
# --workers 1: Single worker (ML models are memory-intensive)
# --threads 4: Multi-threaded for concurrent requests
# --timeout 120: Allow 2 minutes for ML inference
# --keep-alive 5: Keep connections alive for 5 seconds
# --log-level info: Standard logging verbosity
CMD gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --keep-alive 5 \
    --log-level info
