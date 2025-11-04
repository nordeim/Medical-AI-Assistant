#!/bin/bash

# Healthcare Support System Deployment Script
# Production deployment for healthcare organizations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="healthcare-support-system"
COMPOSE_FILE="docker-compose.yml"
HEALTHY_TIMEOUT=300

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check available disk space (minimum 10GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        log_warning "Insufficient disk space. At least 10GB recommended."
    fi
    
    log_success "System requirements check passed"
}

initialize_environment() {
    log_info "Initializing environment variables..."
    
    # Create environment file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating environment configuration..."
        cat > .env << EOL
# Healthcare Support System Environment Configuration
# IMPORTANT: Change all passwords and secrets in production

# Application
APP_ENV=production
APP_NAME=Healthcare Support System
APP_VERSION=1.0.0

# Database Configuration
DATABASE_URL=postgresql://admin:secure_password_2025@database:5432/healthcare_support
DB_POOL_MIN=5
DB_POOL_MAX=20
DB_TIMEOUT=30000

# Redis Configuration
REDIS_URL=redis://:redis_password_2025@redis:6379

# Authentication
JWT_SECRET=your-super-secure-jwt-secret-key-change-in-production-2025
JWT_EXPIRY=24h
BCRYPT_ROUNDS=12
ENCRYPTION_KEY=your-encryption-key-for-sensitive-data-2025

# Email Configuration
SMTP_HOST=smtp.yourdomain.com
SMTP_PORT=587
SMTP_USERNAME=noreply@yourdomain.com
SMTP_PASSWORD=your-smtp-password
SMTP_FROM_EMAIL=noreply@yourdomain.com
SMTP_FROM_NAME=Healthcare Support System

# Medical Emergency Configuration
MEDICAL_EMERGENCY_SLA=30
MEDICAL_ESCALATION_EMAIL=medical@yourdomain.com
MEDICAL_ESCALATION_PHONE=+1-800-MEDICAL
EMERGENCY_CONTACT_EMAIL=emergency@yourdomain.com
EMERGENCY_CONTACT_PHONE=+1-800-HELP-NOW

# Health Check Configuration
HEALTH_CHECK_INTERVAL=60
HEALTH_CHECK_TIMEOUT=5000
HEALTH_CHECK_RETRIES=3
SLA_ALERT_THRESHOLD=95

# Monitoring Configuration
LOG_LEVEL=info
LOG_FORMAT=json
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/webhook/url

# Support Hours
SUPPORT_HOURS_START=08:00
SUPPORT_HOURS_END=18:00
SUPPORT_TIMEZONE=America/New_York

# File Storage
STORAGE_TYPE=minio
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin_password_2025
EOL
        log_warning "Please update environment variables in .env file before production deployment"
    fi
    
    log_success "Environment initialization completed"
}

setup_directories() {
    log_info "Setting up directories..."
    
    # Create required directories
    mkdir -p logs/{nginx,app}
    mkdir -p uploads/{tickets,feedback,training}
    mkdir -p config/{nginx,prometheus,grafana,logstash}
    mkdir -p ssl
    
    # Set proper permissions
    chmod 755 logs uploads config ssl
    chmod 600 .env
    
    log_success "Directory setup completed"
}

build_images() {
    log_info "Building Docker images..."
    
    docker-compose build --no-cache
    
    log_success "Docker images built successfully"
}

initialize_database() {
    log_info "Initializing database..."
    
    # Start database service only
    docker-compose up -d database redis
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout=60
    while ! docker-compose exec -T database pg_isready -U admin -d healthcare_support; do
        sleep 5
        timeout=$((timeout - 5))
        if [ $timeout -le 0 ]; then
            log_error "Database failed to start within timeout"
            exit 1
        fi
    done
    
    log_success "Database is ready"
    
    # Run database migrations
    log_info "Running database migrations..."
    docker-compose exec -T backend npm run migrate
    
    log_success "Database initialized"
}

deploy_services() {
    log_info "Deploying all services..."
    
    # Start all services
    docker-compose up -d
    
    log_success "Services deployed"
}

wait_for_health() {
    log_info "Waiting for services to be healthy..."
    
    start_time=$(date +%s)
    current_time=$(date +%s)
    
    while [ $((current_time - start_time)) -lt $HEALTHY_TIMEOUT ]; do
        # Check backend health
        if docker-compose ps | grep -q "backend.*healthy"; then
            # Check frontend health
            if docker-compose ps | grep -q "frontend.*healthy"; then
                log_success "All services are healthy!"
                return 0
            fi
        fi
        
        log_info "Waiting for services to become healthy..."
        sleep 10
        current_time=$(date +%s)
    done
    
    log_error "Services failed to become healthy within timeout"
    docker-compose ps
    return 1
}

run_health_check() {
    log_info "Running comprehensive health check..."
    
    # Check backend API
    if curl -f http://localhost:8080/api/health > /dev/null 2>&1; then
        log_success "Backend API is healthy"
    else
        log_error "Backend API health check failed"
    fi
    
    # Check frontend
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_success "Frontend is healthy"
    else
        log_error "Frontend health check failed"
    fi
    
    # Check database connection
    if docker-compose exec -T database pg_isready -U admin -d healthcare_support > /dev/null 2>&1; then
        log_success "Database connection is healthy"
    else
        log_error "Database connection failed"
    fi
    
    # Check monitoring
    if curl -f http://localhost:9090 > /dev/null 2>&1; then
        log_success "Monitoring (Prometheus) is healthy"
    else
        log_warning "Monitoring health check failed"
    fi
}

show_access_info() {
    log_success "Healthcare Support System deployed successfully!"
    echo
    echo "==============================================="
    echo "🏥 HEALTHCARE SUPPORT SYSTEM - ACCESS INFO"
    echo "==============================================="
    echo
    echo "🌐 Frontend Application:     http://localhost:3000"
    echo "🔧 Backend API:             http://localhost:8080"
    echo "📊 Monitoring Dashboard:    http://localhost:3001 (admin/admin_password_2025)"
    echo "📈 Prometheus Metrics:      http://localhost:9090"
    echo "🔍 Kibana Logs:            http://localhost:5601"
    echo "💾 File Storage (MinIO):    http://localhost:9001 (minioadmin/minioadmin_password_2025)"
    echo
    echo "📞 SUPPORT CHANNELS:"
    echo "   • Medical Emergency: +1-800-MEDICAL"
    echo "   • General Support:   support@yourdomain.com"
    echo "   • Technical Support: tech@yourdomain.com"
    echo
    echo "🔐 DEFAULT CREDENTIALS:"
    echo "   • Grafana: admin / admin_password_2025"
    echo "   • MinIO: minioadmin / minioadmin_password_2025"
    echo
    echo "⚠️  IMPORTANT SECURITY NOTES:"
    echo "   • Change all default passwords immediately"
    echo "   • Update JWT_SECRET with a secure random string"
    echo "   • Configure SSL/TLS certificates for production"
    echo "   • Review and update all email/notification settings"
    echo
    echo "📋 NEXT STEPS:"
    echo "   1. Update environment variables in .env file"
    echo "   2. Configure SSL certificates"
    echo "   3. Set up email notifications"
    echo "   4. Configure Slack integration"
    echo "   5. Review HIPAA compliance settings"
    echo "   6. Set up regular backups"
    echo
    echo "For support: https://docs.yourdomain.com"
    echo "==============================================="
}

cleanup_on_error() {
    log_error "Deployment failed. Cleaning up..."
    docker-compose down --remove-orphans
    exit 1
}

# Main deployment process
main() {
    echo "🏥 Healthcare Support System - Production Deployment"
    echo "====================================================="
    echo
    
    # Set trap for error handling
    trap cleanup_on_error ERR
    
    # Deployment steps
    check_requirements
    initialize_environment
    setup_directories
    build_images
    initialize_database
    deploy_services
    wait_for_health
    run_health_check
    show_access_info
    
    log_success "🎉 Deployment completed successfully!"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        log_info "Stopping all services..."
        docker-compose down
        log_success "All services stopped"
        ;;
    "restart")
        log_info "Restarting all services..."
        docker-compose restart
        log_success "All services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "health")
        run_health_check
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|health}"
        echo
        echo "Commands:"
        echo "  deploy  - Full deployment (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show service logs"
        echo "  status  - Show service status"
        echo "  health  - Run health checks"
        exit 1
        ;;
esac