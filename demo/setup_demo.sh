#!/bin/bash

# Demo Environment Setup and Deployment Script
# Sets up complete demo environment for presentations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="${PROJECT_ROOT}/demo"
BACKUP_DIR="${PROJECT_ROOT}/backups"
LOG_FILE="${PROJECT_ROOT}/demo_setup.log"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    log_info "Python version: $python_version"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "Medical-AI-Assistant/README.md" ]; then
        log_error "Must run from the Medical-AI-Assistant project root directory"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Install Python dependencies
install_dependencies() {
    log_step "Installing Python dependencies..."
    
    # Install required packages
    pip3 install -q faker python-jose[cryptography] python-multipart || {
        log_error "Failed to install Python dependencies"
        exit 1
    }
    
    log_info "Python dependencies installed"
}

# Setup demo database
setup_demo_database() {
    log_step "Setting up demo database..."
    
    cd "$DEMO_DIR"
    
    # Run database population script
    if python3 database/populate_demo_data.py; then
        log_info "Demo database populated successfully"
    else
        log_error "Failed to populate demo database"
        exit 1
    fi
    
    # Set proper permissions
    chmod 644 demo.db
    
    cd "$PROJECT_ROOT"
}

# Setup demo authentication
setup_demo_auth() {
    log_step "Setting up demo authentication..."
    
    # Create demo auth system
    python3 -c "
import sys
sys.path.append('demo')
from auth.demo_auth import DemoAuthManager

auth_manager = DemoAuthManager()
auth_manager.initialize_demo_users()
print('Demo users created successfully')
"
    
    log_info "Demo authentication configured"
}

# Setup demo analytics
setup_demo_analytics() {
    log_step "Setting up demo analytics..."
    
    # Initialize analytics database
    python3 -c "
import sys
sys.path.append('demo')
from analytics.demo_analytics import DemoAnalyticsManager

analytics = DemoAnalyticsManager()
print('Demo analytics initialized')
"
    
    log_info "Demo analytics configured"
}

# Setup demo backup system
setup_demo_backup() {
    log_step "Setting up demo backup system..."
    
    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    
    # Create initial backup
    python3 -c "
import sys
sys.path.append('demo')
from backup.demo_backup import DemoBackupManager

backup_manager = DemoBackupManager()
backup = backup_manager.create_full_backup('Initial demo environment setup')
print(f'Initial backup created: {backup.backup_id}')
"
    
    log_info "Demo backup system configured"
}

# Generate demo configuration
generate_demo_config() {
    log_step "Generating demo configuration..."
    
    # Create environment file for demo
    cat > .env.demo << EOF
# Demo Environment Configuration
DEMO_MODE=true
DATABASE_URL=sqlite:///demo/demo.db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=demo-secret-key-for-presentations-only
DEMO_USERS_ENABLED=true
ANALYTICS_ENABLED=true
BACKUP_ENABLED=true
PERFORMANCE_MODE=fast
LOG_LEVEL=INFO
EOF
    
    log_info "Demo configuration generated"
}

# Setup demo scenarios
setup_demo_scenarios() {
    log_step "Setting up demo scenarios..."
    
    # Test scenario creation
    python3 -c "
import sys
sys.path.append('demo')
from scenarios.medical_scenarios import ScenarioManager

manager = ScenarioManager()
diabetes = manager.create_scenario('diabetes', 1)
hypertension = manager.create_scenario('hypertension', 2)
chest_pain = manager.create_scenario('chest_pain', 3)
print('Demo scenarios initialized successfully')
"
    
    log_info "Demo scenarios configured"
}

# Start demo services
start_demo_services() {
    log_step "Starting demo services..."
    
    # Start backend API in background
    cd backend
    
    if [ -f "app/main.py" ]; then
        python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
        BACKEND_PID=$!
        echo $BACKEND_PID > ../demo/backend.pid
        log_info "Backend API started (PID: $BACKEND_PID)"
        
        # Wait for backend to start
        sleep 5
        
        # Test backend health
        if curl -s http://localhost:8000/health > /dev/null; then
            log_info "Backend API is responsive"
        else
            log_warn "Backend API health check failed"
        fi
    else
        log_warn "Backend main.py not found, skipping backend start"
    fi
    
    cd "$PROJECT_ROOT"
}

# Start demo frontend
start_demo_frontend() {
    log_step "Starting demo frontend..."
    
    cd frontend
    
    if [ -f "package.json" ]; then
        # Install frontend dependencies if needed
        if [ ! -d "node_modules" ]; then
            log_info "Installing frontend dependencies..."
            npm install --silent
        fi
        
        # Start frontend in background
        npm run dev &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > ../demo/frontend.pid
        log_info "Frontend started (PID: $FRONTEND_PID)"
        
        # Wait for frontend to start
        sleep 10
        
        # Test frontend
        if curl -s http://localhost:3000 > /dev/null; then
            log_info "Frontend is responsive"
        else
            log_warn "Frontend health check failed"
        fi
    else
        log_warn "Frontend package.json not found, skipping frontend start"
    fi
    
    cd "$PROJECT_ROOT"
}

# Verify demo environment
verify_demo_environment() {
    log_step "Verifying demo environment..."
    
    python3 -c "
import sys
sys.path.append('demo')
from backup.demo_backup import DemoBackupManager

backup_manager = DemoBackupManager()
report = backup_manager.get_demo_readiness_report()

print('\\n=== DEMO READINESS REPORT ===')
print(f'Demo State Ready: {report[\"demo_state\"][\"ready\"]}')
print(f'Timestamp: {report[\"timestamp\"]}')

for component, status in report['demo_state']['verification_results'].items():
    status_icon = '✓' if status else '✗'
    print(f'{status_icon} {component.replace(\"_\", \" \").title()}')

if report['backup_status']['latest_backup']:
    print(f'Latest Backup: {report[\"backup_status\"][\"latest_backup\"]}')
    print(f'Backup Age: {report[\"backup_status\"][\"backup_age_hours\"]} hours')

print('\\nRecommendations:')
for rec in report['recommendations']:
    print(f'  • {rec}')
    
if not report['demo_state']['ready']:
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        log_info "Demo environment verification passed"
    else
        log_error "Demo environment verification failed"
        exit 1
    fi
}

# Print demo access information
print_demo_info() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}   DEMO ENVIRONMENT READY!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Access URLs:${NC}"
    echo -e "  Frontend:  ${YELLOW}http://localhost:3000${NC}"
    echo -e "  Backend:   ${YELLOW}http://localhost:8000${NC}"
    echo -e "  API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
    echo ""
    echo -e "${BLUE}Demo Credentials:${NC}"
    echo -e "  Administrator:"
    echo -e "    Email: ${YELLOW}admin@demo.medai.com${NC}"
    echo -e "    Password: ${YELLOW}DemoAdmin123!${NC}"
    echo ""
    echo -e "  Nurse:"
    echo -e "    Email: ${YELLOW}nurse.jones@demo.medai.com${NC}"
    echo -e "    Password: ${YELLOW}DemoNurse456!${NC}"
    echo ""
    echo -e "  Patient:"
    echo -e "    Email: ${YELLOW}patient.smith@demo.medai.com${NC}"
    echo -e "    Password: ${YELLOW}DemoPatient789!${NC}"
    echo ""
    echo -e "${BLUE}Demo Scenarios:${NC}"
    echo -e "  1. ${YELLOW}Diabetes Management${NC} - Real-time glucose monitoring"
    echo -e "  2. ${YELLOW}Hypertension Monitoring${NC} - BP tracking and CV risk assessment"
    echo -e "  3. ${YELLOW}Chest Pain Assessment${NC} - Emergency triage evaluation"
    echo ""
    echo -e "${BLUE}Backup and Recovery:${NC}"
    echo -e "  Backup directory: ${YELLOW}$BACKUP_DIR${NC}"
    echo -e "  Auto-backup enabled (24-hour intervals)"
    echo ""
    echo -e "${GREEN}Demo environment setup completed successfully!${NC}"
    echo -e "${GREEN}Check $LOG_FILE for detailed setup logs${NC}"
    echo ""
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    # Stop backend service
    if [ -f "demo/backend.pid" ]; then
        BACKEND_PID=$(cat demo/backend.pid)
        if kill -0 $BACKEND_PID 2>/dev/null; then
            kill $BACKEND_PID
            log_info "Stopped backend service (PID: $BACKEND_PID)"
        fi
        rm -f demo/backend.pid
    fi
    
    # Stop frontend service
    if [ -f "demo/frontend.pid" ]; then
        FRONTEND_PID=$(cat demo/frontend.pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            log_info "Stopped frontend service (PID: $FRONTEND_PID)"
        fi
        rm -f demo/frontend.pid
    fi
}

# Main setup function
main() {
    log_info "Starting Medical AI Assistant Demo Environment Setup"
    log_info "Timestamp: $(date)"
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run setup steps
    check_prerequisites
    install_dependencies
    setup_demo_database
    setup_demo_auth
    setup_demo_analytics
    setup_demo_backup
    generate_demo_config
    setup_demo_scenarios
    start_demo_services
    start_demo_frontend
    verify_demo_environment
    print_demo_info
    
    log_info "Demo environment setup completed successfully!"
}

# Handle script arguments
case "${1:-start}" in
    "start")
        main
        ;;
    "stop")
        cleanup
        ;;
    "restart")
        cleanup
        sleep 2
        main
        ;;
    "status")
        echo "Demo Environment Status:"
        if [ -f "demo/backend.pid" ] && kill -0 $(cat demo/backend.pid) 2>/dev/null; then
            echo "✓ Backend service running (PID: $(cat demo/backend.pid))"
        else
            echo "✗ Backend service not running"
        fi
        
        if [ -f "demo/frontend.pid" ] && kill -0 $(cat demo/frontend.pid) 2>/dev/null; then
            echo "✓ Frontend service running (PID: $(cat demo/frontend.pid))"
        else
            echo "✗ Frontend service not running"
        fi
        ;;
    "reset")
        log_info "Resetting demo environment..."
        cleanup
        
        # Remove demo databases
        rm -f demo/demo.db
        rm -f demo_analytics.db
        
        # Recreate database
        setup_demo_database
        setup_demo_auth
        setup_demo_analytics
        
        log_info "Demo environment reset completed"
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|reset}"
        echo ""
        echo "Commands:"
        echo "  start   - Start demo environment (default)"
        echo "  stop    - Stop demo services"
        echo "  restart - Restart demo environment"
        echo "  status  - Check demo service status"
        echo "  reset   - Reset demo database and recreate"
        exit 1
        ;;
esac