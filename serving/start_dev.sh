#!/bin/bash

# Development startup script for the model serving infrastructure
set -e

echo "Starting Medical AI Model Serving Infrastructure - Development Mode"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs cache data models

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo "Please configure .env file with your settings"
fi

# Set development environment
export APP_ENVIRONMENT=development
export APP_DEBUG=true
export LOG_LOG_LEVEL=DEBUG

echo "Starting development server..."
python main.py