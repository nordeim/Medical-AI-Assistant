#!/bin/bash

# Production startup script for the model serving infrastructure
set -e

echo "Starting Medical AI Model Serving Infrastructure - Production Mode"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please configure environment variables."
    exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs cache data models

# Set production environment
export APP_ENVIRONMENT=production
export APP_DEBUG=false
export LOG_LOG_LEVEL=INFO

# Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    echo "Checking Redis connection..."
    if redis-cli ping &> /dev/null; then
        echo "Redis is running"
    else
        echo "Warning: Redis is not running. Cache functionality will be limited."
    fi
fi

echo "Starting production server with Gunicorn..."
gunicorn main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --worker-connections 1000 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --timeout 300 \
    --keep-alive 2 \
    --preload \
    --access-logfile ./logs/access.log \
    --error-logfile ./logs/error.log \
    --log-level info \
    --bind 0.0.0.0:8000