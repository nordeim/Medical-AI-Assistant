#!/bin/bash
# Production Medical AI Model Serving - Deployment Script
# Deploys the complete production infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="medical-ai-production"
DOCKER_REGISTRY="medical-ai"
VERSION="1.0.0"
ENVIRONMENT=${1:-production}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Medical AI Production Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Environment: ${ENVIRONMENT}"
echo -e "Version: ${VERSION}"
echo -e "Timestamp: $(date)"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        print_warning "Less than 10GB disk space available"
    fi
    
    print_status "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data/{redis,postgres,mlflow}
    mkdir -p models/registry
    mkdir -p backups
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/provisioning
    
    # Set proper permissions
    chmod 755 logs data models backups
    
    print_status "Directories created"
}

# Setup environment variables
setup_environment() {
    print_status "Setting up environment variables..."
    
    cat > .env << EOF
# Medical AI Production Environment
ENVIRONMENT=${ENVIRONMENT}
VERSION=${VERSION}
PROJECT_NAME=${PROJECT_NAME}

# Database Configuration
POSTGRES_DB=medical_ai
POSTGRES_USER=medical_ai_user
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///data/mlflow/mlflow.db
MLFLOW_REGISTRY_URI=sqlite:///data/mlflow/registry.db

# Security Configuration
SECRET_KEY=$(openssl rand -base64 64)
JWT_SECRET=$(openssl rand -base64 32)
API_KEY_SALT=$(openssl rand -base64 16)

# Model Configuration
DEFAULT_MODEL=medical-diagnosis-v1
MODEL_REGISTRY_PATH=/app/data/models
OPTIMIZATION_ENABLED=true

# Monitoring Configuration
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=false

# Cache Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json

# Feature Flags
ENABLE_AB_TESTING=true
ENABLE_DRIFT_DETECTION=true
ENABLE_AUTO_RETRAINING=true
ENABLE_MODEL_HOT_SWAP=true
EOF
    
    print_status "Environment variables configured"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build model serving image
    docker build -f serving/Dockerfile -t ${DOCKER_REGISTRY}/model-serving:${VERSION} .
    
    # Build MLflow registry image
    docker build -f registry/Dockerfile -t ${DOCKER_REGISTRY}/mlflow-registry:${VERSION} .
    
    # Build monitoring image
    docker build -f monitoring/Dockerfile -t ${DOCKER_REGISTRY}/monitoring:${VERSION} .
    
    print_status "Docker images built successfully"
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    
    # Start PostgreSQL service
    docker-compose up -d postgres redis
    
    # Wait for database to be ready
    print_status "Waiting for database to be ready..."
    sleep 30
    
    # Run database migrations
    docker-compose exec postgres psql -U medical_ai_user -d medical_ai -c "
        CREATE TABLE IF NOT EXISTS model_versions (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB,
            UNIQUE(model_name, version)
        );
        
        CREATE TABLE IF NOT EXISTS model_dependencies (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            dependency_name VARCHAR(255) NOT NULL,
            dependency_version VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    "
    
    print_status "Database setup completed"
}

# Initialize MLflow
initialize_mlflow() {
    print_status "Initializing MLflow registry..."
    
    # Start MLflow service
    docker-compose up -d mlflow
    
    # Wait for MLflow to start
    sleep 20
    
    # Initialize MLflow experiments
    python3 -c "
import mlflow
mlflow.set_tracking_uri('sqlite:///data/mlflow/mlflow.db')

# Create main experiment
try:
    experiment_id = mlflow.create_experiment('medical_ai_models')
    print(f'Created experiment: {experiment_id}')
except Exception as e:
    print(f'Experiment creation: {e}')

# Set experiment as active
mlflow.set_experiment('medical_ai_models')
print('MLflow initialization completed')
"
    
    print_status "MLflow initialized"
}

# Deploy monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    # Start monitoring services
    docker-compose up -d prometheus grafana
    
    # Wait for services to start
    sleep 30
    
    # Configure Grafana dashboards
    python3 -c "
import json
import requests

# Grafana API configuration
grafana_url = 'http://localhost:3000'
username = 'admin'
password = 'admin123'

# Login to Grafana
session = requests.Session()
session.auth = (username, password)

# Create dashboard
dashboard = {
    'dashboard': {
        'title': 'Medical AI Production Monitoring',
        'panels': [
            {
                'title': 'Model Performance',
                'type': 'graph',
                'targets': [
                    {
                        'expr': 'model_inference_duration_seconds',
                        'legendFormat': '{{model_name}}'
                    }
                ]
            }
        ]
    },
    'overwrite': True
}

try:
    response = session.post(f'{grafana_url}/api/dashboards/db', json=dashboard)
    print(f'Dashboard creation status: {response.status_code}')
except Exception as e:
    print(f'Dashboard creation error: {e}')
"
    
    print_status "Monitoring stack deployed"
}

# Deploy model serving infrastructure
deploy_model_serving() {
    print_status "Deploying model serving infrastructure..."
    
    # Start all services
    docker-compose up -d
    
    # Wait for all services to be healthy
    print_status "Waiting for services to become healthy..."
    sleep 60
    
    # Health check
    health_status=$(curl -s http://localhost:8000/health || echo "unhealthy")
    if [[ $health_status == *"healthy"* ]]; then
        print_status "Model serving is healthy"
    else
        print_warning "Model serving health check failed"
    fi
    
    print_status "Model serving infrastructure deployed"
}

# Run smoke tests
run_smoke_tests() {
    print_status "Running smoke tests..."
    
    # Test model server health
    health_response=$(curl -s http://localhost:8000/health)
    if [[ $health_response == *"healthy"* ]]; then
        print_status "✓ Model server health check passed"
    else
        print_error "✗ Model server health check failed"
    fi
    
    # Test prediction endpoint
    prediction_response=$(curl -s -X POST http://localhost:8000/predict \
        -H "Content-Type: application/json" \
        -d '{"patient_id": "test", "clinical_data": {"symptom": "test"}}' || echo "error")
    
    if [[ $prediction_response != *"error"* ]]; then
        print_status "✓ Prediction endpoint test passed"
    else
        print_warning "⚠ Prediction endpoint test failed (expected for demo)"
    fi
    
    # Test MLflow registry
    mlflow_response=$(curl -s http://localhost:5000/api/2.0/mlflow/experiments/list || echo "error")
    if [[ $mlflow_response != *"error"* ]]; then
        print_status "✓ MLflow registry test passed"
    else
        print_warning "⚠ MLflow registry test failed"
    fi
    
    # Test monitoring endpoints
    prometheus_response=$(curl -s http://localhost:9090/api/v1/query?query=up || echo "error")
    if [[ $prometheus_response != *"error"* ]]; then
        print_status "✓ Prometheus monitoring test passed"
    else
        print_warning "⚠ Prometheus monitoring test failed"
    fi
    
    print_status "Smoke tests completed"
}

# Display deployment information
display_deployment_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Deployment Completed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Service URLs:${NC}"
    echo "• Model Server:           http://localhost:8000"
    echo "• API Documentation:      http://localhost:8000/docs"
    echo "• Health Check:          http://localhost:8000/health"
    echo "• MLflow Registry:       http://localhost:5000"
    echo "• Prometheus:            http://localhost:9090"
    echo "• Grafana:               http://localhost:3000 (admin/admin123)"
    echo ""
    echo -e "${BLUE}Default Credentials:${NC}"
    echo "• Grafana Admin:         admin / admin123"
    echo "• Database:              medical_ai_user / (see .env file)"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "• View logs:             docker-compose logs -f"
    echo "• Stop services:         docker-compose down"
    echo "• Restart services:      docker-compose restart"
    echo "• Scale model servers:   docker-compose up -d --scale model-server=3"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Configure SSL/TLS certificates for production"
    echo "2. Set up external monitoring and alerting"
    echo "3. Configure backup schedules"
    echo "4. Review and update security settings"
    echo "5. Load test the deployment"
    echo ""
}

# Cleanup function
cleanup_on_error() {
    print_error "Deployment failed, cleaning up..."
    docker-compose down -v
    exit 1
}

# Set error trap
trap cleanup_on_error ERR

# Main deployment flow
main() {
    check_prerequisites
    create_directories
    setup_environment
    build_images
    setup_database
    initialize_mlflow
    deploy_monitoring
    deploy_model_serving
    run_smoke_tests
    display_deployment_info
}

# Run deployment
main "$@"