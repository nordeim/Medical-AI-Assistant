#!/bin/bash

# =============================================================================
# Medical AI Assistant - Quick Setup Script
# This script helps initialize the development environment
# =============================================================================

set -e  # Exit on error

echo "=================================="
echo "Medical AI Assistant - Setup"
echo "=================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi
echo "✅ Docker installed: $(docker --version)"

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi
echo "✅ Docker Compose installed: $(docker compose version)"

# Check for .env file
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env file with your configuration before continuing"
    echo ""
    read -p "Press Enter to continue after editing .env..."
fi

# Create necessary directories
echo ""
echo "Creating data directories..."
mkdir -p data models vector_stores logs
echo "✅ Directories created"

# Ask user what to do
echo ""
echo "What would you like to do?"
echo "1) Start development environment"
echo "2) Start production environment"
echo "3) Run database migrations only"
echo "4) Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Starting development environment..."
        echo "This will start: PostgreSQL, Redis, Backend (with reload), Frontend (with reload)"
        echo ""
        docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up --build -d
        
        echo ""
        echo "✅ Development environment started!"
        echo ""
        echo "Services available at:"
        echo "  - Frontend: http://localhost:3000"
        echo "  - Backend API: http://localhost:8000"
        echo "  - API Docs: http://localhost:8000/docs"
        echo "  - pgAdmin: http://localhost:5050 (admin@medai.local / admin)"
        echo "  - Mailhog: http://localhost:8025"
        echo ""
        echo "View logs: docker compose -f docker/docker-compose.yml logs -f"
        echo "Stop: docker compose -f docker/docker-compose.yml down"
        ;;
    
    2)
        echo ""
        echo "Starting production environment..."
        echo "This will start: PostgreSQL, Redis, Backend, Frontend"
        echo ""
        docker compose -f docker/docker-compose.yml up --build -d
        
        echo ""
        echo "✅ Production environment started!"
        echo ""
        echo "Services available at:"
        echo "  - Frontend: http://localhost:3000"
        echo "  - Backend API: http://localhost:8000"
        echo ""
        ;;
    
    3)
        echo ""
        echo "Running database migrations..."
        docker compose -f docker/docker-compose.yml exec backend alembic upgrade head
        echo "✅ Migrations complete"
        ;;
    
    4)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "Setup complete! 🎉"
echo ""
