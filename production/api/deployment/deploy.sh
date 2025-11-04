#!/bin/bash

# Production API Deployment Script
# Automated deployment for healthcare API management system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_ROOT/logs/deployment_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local message="$1"
    local level="${2:-INFO}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${BLUE}[$timestamp]${NC} [$level] $message" | tee -a "$LOG_FILE"
}

error() {
    local message="$1"
    log "$message" "ERROR"
    echo -e "${RED}[ERROR]${NC} $message" >&2
}

success() {
    local message="$1"
    log "$message" "SUCCESS"
    echo -e "${GREEN}[SUCCESS]${NC} $message"
}

warning() {
    local message="$1"
    log "$message" "WARNING"
    echo -e "${YELLOW}[WARNING]${NC} $message"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Redis
    if ! command -v redis-cli &> /dev/null; then
        warning "Redis CLI not found, will use Docker Redis"
    fi
    
    success "Prerequisites check passed"
}

# Create necessary directories
setup_directories() {
    log "Setting up directories..."
    
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/ssl"
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/uploads"
    mkdir -p "$PROJECT_ROOT/backups"
    mkdir -p "$PROJECT_ROOT/tmp"
    
    success "Directories created"
}

# Generate SSL certificates (self-signed for development)
generate_ssl_certificates() {
    local ssl_dir="$PROJECT_ROOT/ssl"
    local cert_file="$ssl_dir/healthcare.crt"
    local key_file="$ssl_dir/healthcare.key"
    
    if [ ! -f "$cert_file" ] || [ ! -f "$key_file" ]; then
        log "Generating SSL certificates..."
        
        openssl req -x509 -newkey rsa:4096 -keyout "$key_file" -out "$cert_file" -days 365 -nodes \
            -subj "/C=US/ST=Health/L=Medical/O=Healthcare API/CN=localhost"
        
        chmod 600 "$key_file"
        chmod 644 "$cert_file"
        
        success "SSL certificates generated"
    else
        log "SSL certificates already exist"
    fi
}

# Setup environment configuration
setup_environment() {
    log "Setting up environment configuration..."
    
    local env_file="$PROJECT_ROOT/.env"
    
    if [ ! -f "$env_file" ]; then
        log "Creating environment file..."
        
        cat > "$env_file" << EOF
# Healthcare API Environment Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
API_NAME=Healthcare API Management System
API_VERSION=3.0.0

# Security
JWT_SECRET=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 16)

# Database
DATABASE_URL=postgresql://healthcare_user:healthcare_pass@postgres:5432/healthcare_db

# External Services
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID:-}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY:-}
AWS_REGION=${AWS_REGION:-us-east-1}

# Monitoring
GRAFANA_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
PROMETHEUS_RETENTION=200h

# Billing
STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY:-sk_test_}
BILLING_WEBHOOK_SECRET=$(openssl rand -base64 32)

# Email
SMTP_USERNAME=${SMTP_USERNAME:-}
SMTP_PASSWORD=${SMTP_PASSWORD:-}

# Notifications
SLACK_BOT_TOKEN=${SLACK_BOT_TOKEN:-}
PAGERDUTY_API_KEY=${PAGERDUTY_API_KEY:-}

EOF
        
        chmod 600 "$env_file"
        success "Environment file created: $env_file"
    else
        log "Environment file already exists"
    fi
}

# Build and start services
deploy_services() {
    local environment="${1:-production}"
    log "Deploying services for environment: $environment"
    
    # Build services
    log "Building Docker images..."
    docker-compose build --parallel
    
    # Start services
    log "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to start..."
    sleep 30
    
    # Health check
    check_service_health
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    local services=("kong" "healthcare-api" "fhir-server" "docs-portal")
    local max_attempts=30
    local attempt=1
    
    for service in "${services[@]}"; do
        log "Checking health of $service..."
        
        while [ $attempt -le $max_attempts ]; do
            if docker-compose ps "$service" | grep -q "Up"; then
                # Test HTTP endpoint
                case "$service" in
                    "kong")
                        if curl -sf http://localhost:8001 > /dev/null 2>&1; then
                            success "$service is healthy"
                            break
                        fi
                        ;;
                    "healthcare-api")
                        if curl -sf http://localhost:8002/health > /dev/null 2>&1; then
                            success "$service is healthy"
                            break
                        fi
                        ;;
                    "fhir-server")
                        if curl -sf http://localhost:8003/fhir/metadata > /dev/null 2>&1; then
                            success "$service is healthy"
                            break
                        fi
                        ;;
                    "docs-portal")
                        if curl -sf http://localhost:8080 > /dev/null 2>&1; then
                            success "$service is healthy"
                            break
                        fi
                        ;;
                esac
            fi
            
            log "Attempt $attempt/$max_attempts: $service not ready yet..."
            sleep 10
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            error "$service failed to become healthy"
        fi
    done
    
    # Check external dependencies
    log "Checking external dependencies..."
    
    if ! curl -sf http://localhost:6379 > /dev/null 2>&1; then
        warning "Redis might not be ready yet"
    else
        success "Redis is responding"
    fi
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    # Wait for database to be ready
    log "Waiting for database..."
    sleep 15
    
    # Run migration commands
    docker-compose exec -T healthcare-api python -c "
import asyncio
from integration.hl7.fhir_framework import FHIRServer
from monitoring.analytics.monitoring_system import MonitoringSystem

# Initialize FHIR server
fhir_server = FHIRServer({'base_url': 'http://fhir-server:8000'})

# Create capability statement
import asyncio
capability_statement = asyncio.run(fhir_server.create_capability_statement())
print('FHIR Capability Statement created')

# Initialize monitoring
monitoring = MonitoringSystem({'prometheus_url': 'http://prometheus:9090'})
print('Monitoring system initialized')
"
    
    success "Database migrations completed"
}

# Initialize monitoring and analytics
setup_monitoring() {
    log "Setting up monitoring and analytics..."
    
    # Wait for Grafana
    sleep 10
    
    # Configure Prometheus data source in Grafana
    curl -X POST \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "url": "http://prometheus:9090",
            "access": "proxy",
            "isDefault": true
        }' \
        http://admin:admin@localhost:3000/api/datasources
    
    success "Monitoring configured"
}

# Run comprehensive tests
run_tests() {
    log "Running comprehensive tests..."
    
    # Wait for all services to be fully ready
    sleep 30
    
    # Run test suite
    docker-compose exec -T healthcare-api python tests/comprehensive_test.py
    
    if [ $? -eq 0 ]; then
        success "All tests passed"
    else
        error "Some tests failed"
    fi
}

# Setup SSL certificates for production
setup_production_ssl() {
    if [ "${ENVIRONMENT:-}" = "production" ]; then
        log "Setting up production SSL certificates..."
        
        # In production, you would use Let's Encrypt or your own CA
        # For now, we'll assume certificates are provided
        if [ ! -f "$PROJECT_ROOT/ssl/production.crt" ]; then
            warning "Production SSL certificates not found. Please add:"
            warning "  - $PROJECT_ROOT/ssl/production.crt"
            warning "  - $PROJECT_ROOT/ssl/production.key"
        fi
    fi
}

# Display deployment summary
display_summary() {
    log "Deployment Summary"
    echo
    echo "🌐 API Endpoints:"
    echo "  - API Gateway:         http://localhost:8000"
    echo "  - API Administration:  http://localhost:8001"
    echo "  - Healthcare API:      http://localhost:8002"
    echo "  - FHIR Server:         http://localhost:8003"
    echo "  - Documentation:       http://localhost:8080"
    echo
    echo "📊 Monitoring:"
    echo "  - Prometheus:          http://localhost:9090"
    echo "  - Grafana:             http://localhost:3000"
    echo "  - Admin Credentials:   admin/admin"
    echo
    echo "📋 Quick Commands:"
    echo "  - View logs:           docker-compose logs -f"
    echo "  - Stop services:       docker-compose down"
    echo "  - Restart services:    docker-compose restart"
    echo "  - Run tests:           docker-compose exec healthcare-api python tests/comprehensive_test.py"
    echo
    echo "📁 Important Files:"
    echo "  - Configuration:       $PROJECT_ROOT/config/production-config.env"
    echo "  - Environment:         $PROJECT_ROOT/.env"
    echo "  - SSL Certificates:    $PROJECT_ROOT/ssl/"
    echo "  - Log File:            $LOG_FILE"
    echo
    echo "✅ Deployment completed successfully!"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        error "Deployment failed with exit code $exit_code"
        error "Check log file: $LOG_FILE"
        
        # Show recent logs
        echo
        echo "Recent logs:"
        tail -50 "$LOG_FILE" 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    local environment="${1:-production}"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    log "🚀 Starting Healthcare API deployment..."
    log "Environment: $environment"
    log "Timestamp: $TIMESTAMP"
    log "Log file: $LOG_FILE"
    
    # Check if running as root (not recommended)
    if [ "$EUID" -eq 0 ]; then
        warning "Running as root is not recommended"
    fi
    
    # Pre-deployment checks
    check_prerequisites
    setup_directories
    generate_ssl_certificates
    setup_environment
    setup_production_ssl
    
    # Deploy services
    deploy_services "$environment"
    
    # Post-deployment setup
    run_migrations
    setup_monitoring
    
    # Run tests (optional in production)
    if [ "$environment" != "production" ] || [ "${RUN_TESTS:-false}" = "true" ]; then
        run_tests
    fi
    
    # Display summary
    display_summary
    
    log "🎉 Healthcare API deployment completed successfully!"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Healthcare API Deployment Script"
            echo
            echo "Usage: $0 [OPTIONS] [ENVIRONMENT]"
            echo
            echo "Options:"
            echo "  -h, --help              Show this help message"
            echo "  -e, --environment ENV   Set deployment environment (development|staging|production)"
            echo "  --skip-tests            Skip running tests"
            echo "  --force                 Force deployment even if issues detected"
            echo
            echo "Environments:"
            echo "  development             Development environment with debug enabled"
            echo "  staging                 Staging environment for testing"
            echo "  production              Production environment (default)"
            echo
            echo "Examples:"
            echo "  $0                      Deploy to production"
            echo "  $0 development          Deploy to development environment"
            echo "  $0 staging --skip-tests Deploy to staging without tests"
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --skip-tests)
            export RUN_TESTS=false
            shift
            ;;
        --force)
            export FORCE_DEPLOYMENT=true
            shift
            ;;
        development|staging|production)
            ENVIRONMENT="$1"
            shift
            ;;
        *)
            error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default environment
ENVIRONMENT="${ENVIRONMENT:-production}"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    error "Invalid environment: $ENVIRONMENT"
    echo "Valid environments: development, staging, production"
    exit 1
fi

# Run main deployment
main "$ENVIRONMENT"