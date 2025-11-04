#!/bin/bash
# Demo Setup and Launch Script - Medical AI Assistant
# Automated setup for demo presentations and stakeholder demonstrations

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEMO_DIR="/workspace/Medical-AI-Assistant/demo"
PRESENTATIONS_DIR="$DEMO_DIR/presentations"
LOG_FILE="$DEMO_DIR/demo_setup.log"

# Function to log messages
log_message() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to print colored messages
print_message() {
    echo -e "${2:-$NC}$1${NC}"
}

# Function to check dependencies
check_dependencies() {
    log_message "Checking dependencies..."
    
    local missing_deps=()
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        missing_deps+=("python3")
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        missing_deps+=("pip3")
    fi
    
    # Check Docker (optional)
    if ! command -v docker &> /dev/null; then
        log_message "Warning: Docker not found - some features may be limited"
    fi
    
    # Check Node.js (optional)
    if ! command -v node &> /dev/null; then
        log_message "Warning: Node.js not found - some features may be limited"
    fi
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_message "Missing dependencies: ${missing_deps[*]}" "$RED"
        print_message "Please install missing dependencies before continuing" "$YELLOW"
        exit 1
    fi
    
    print_message "All required dependencies found" "$GREEN"
}

# Function to install Python requirements
install_requirements() {
    log_message "Installing Python requirements..."
    
    local req_files=(
        "$PRESENTATIONS_DIR/requirements.txt"
        "$DEMO_DIR/requirements.txt"
    )
    
    for req_file in "${req_files[@]}"; do
        if [ -f "$req_file" ]; then
            log_message "Installing from $req_file"
            pip3 install -r "$req_file" --quiet
        fi
    done
    
    print_message "Python requirements installed" "$GREEN"
}

# Function to create directory structure
create_directories() {
    log_message "Creating directory structure..."
    
    local directories=(
        "$PRESENTATIONS_DIR/pitch-decks"
        "$PRESENTATIONS_DIR/technical-demos"
        "$PRESENTATIONS_DIR/stakeholder-demos"
        "$PRESENTATIONS_DIR/recordings"
        "$PRESENTATIONS_DIR/feedback"
        "$PRESENTATIONS_DIR/analytics"
        "$PRESENTATIONS_DIR/scripts"
        "$DEMO_DIR/demo_reports"
        "$DEMO_DIR/logs"
        "$DEMO_DIR/backups"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        log_message "Created directory: $dir"
    done
    
    print_message "Directory structure created" "$GREEN"
}

# Function to setup demo databases
setup_databases() {
    log_message "Setting up demo databases..."
    
    # Create SQLite databases
    local databases=(
        "$PRESENTATIONS_DIR/demo_analytics.db"
        "$PRESENTATIONS_DIR/feedback_analytics.db"
        "$PRESENTATIONS_DIR/recording_analytics.db"
        "$PRESENTATIONS_DIR/demo_orchestration.db"
    )
    
    for db in "${databases[@]}"; do
        if [ ! -f "$db" ]; then
            touch "$db"
            log_message "Created database: $db"
        fi
    done
    
    print_message "Databases initialized" "$GREEN"
}

# Function to generate demo configuration
generate_demo_config() {
    log_message "Generating demo configuration..."
    
    cat > "$PRESENTATIONS_DIR/demo_config.json" << 'EOF'
{
    "demo_environments": {
        "executive_demo": {
            "duration_minutes": 15,
            "focus_areas": ["business_value", "roi", "strategic_impact"],
            "technologies": ["large_display", "interactive_dashboard", "roi_calculator"]
        },
        "clinical_demo": {
            "duration_minutes": 30,
            "focus_areas": ["patient_outcomes", "workflow_integration", "evidence_based"],
            "technologies": ["clinical_workstation", "mobile_device", "emr_integration"]
        },
        "technical_demo": {
            "duration_minutes": 45,
            "focus_areas": ["api_capabilities", "performance", "security"],
            "technologies": ["developer_tools", "api_playground", "security_dashboard"]
        },
        "regulatory_demo": {
            "duration_minutes": 45,
            "focus_areas": ["compliance", "validation", "audit_trails"],
            "technologies": ["compliance_portal", "validation_data", "audit_system"]
        },
        "investor_demo": {
            "duration_minutes": 25,
            "focus_areas": ["market_opportunity", "competitive_advantage", "growth"],
            "technologies": ["financial_model", "market_analysis", "competitive_intelligence"]
        }
    },
    "scenario_library": {
        "cardiology": ["stemi_pci", "heart_failure", "atrial_fibrillation"],
        "oncology": ["breast_cancer", "lung_cancer", "colon_cancer"],
        "emergency_medicine": ["stroke_tpa", "sepsis_shock", "chest_pain"],
        "chronic_disease": ["diabetes_management", "hypertension", "copd_exacerbation"]
    },
    "automation_settings": {
        "default_automation_level": "semi_automated",
        "auto_record": false,
        "auto_feedback": true,
        "timeout_minutes": 45
    }
}
EOF
    
    print_message "Demo configuration generated" "$GREEN"
}

# Function to create launch scripts
create_launch_scripts() {
    log_message "Creating launch scripts..."
    
    # Executive Demo Launch Script
    cat > "$PRESENTATIONS_DIR/launch_executive_demo.sh" << 'EOF'
#!/bin/bash
# Launch Executive Demo

echo "Starting Executive Demo for C-Suite..."
cd /workspace/Medical-AI-Assistant/demo/presentations/scripts

python3 demo_orchestrator.py \
    --create-demo \
    --demo-type cardiology \
    --stakeholder c_suite \
    --automation-level semi_automated \
    --record \
    --feedback

echo "Executive demo configuration created. Starting demo..."
python3 demo_orchestrator.py --start-demo quick_cardiology
EOF
    
    # Clinical Demo Launch Script
    cat > "$PRESENTATIONS_DIR/launch_clinical_demo.sh" << 'EOF'
#!/bin/bash
# Launch Clinical Demo

echo "Starting Clinical Demo..."
cd /workspace/Medical-AI-Assistant/demo/presentations/scripts

python3 demo_orchestrator.py \
    --create-demo \
    --demo-type cardiology \
    --stakeholder clinical \
    --automation-level semi_automated \
    --record \
    --feedback

echo "Clinical demo configuration created. Starting demo..."
python3 demo_orchestrator.py --start-demo clinical_cardiology
EOF
    
    # All Stakeholder Demo Script
    cat > "$PRESENTATIONS_DIR/launch_all_stakeholders.sh" << 'EOF'
#!/bin/bash
# Launch Demo for All Stakeholder Types

echo "Starting comprehensive stakeholder demonstration..."

# Executive Demo
echo "=== EXECUTIVE DEMO ==="
bash launch_executive_demo.sh

sleep 2

# Clinical Demo  
echo "=== CLINICAL DEMO ==="
bash launch_clinical_demo.sh

sleep 2

# Technical Demo
echo "=== TECHNICAL DEMO ==="
python3 demo_orchestrator.py \
    --create-demo \
    --demo-type cardiology \
    --stakeholder technical \
    --automation-level fully_automated \
    --record

# Investor Demo
echo "=== INVESTOR DEMO ==="
python3 demo_orchestrator.py \
    --create-demo \
    --demo-type cardiology \
    --stakeholder investor \
    --automation-level ai_powered \
    --record

echo "All stakeholder demos completed"
EOF
    
    # Make scripts executable
    chmod +x "$PRESENTATIONS_DIR/launch_executive_demo.sh"
    chmod +x "$PRESENTATIONS_DIR/launch_clinical_demo.sh"
    chmod +x "$PRESENTATIONS_DIR/launch_all_stakeholders.sh"
    
    print_message "Launch scripts created" "$GREEN"
}

# Function to validate installation
validate_installation() {
    log_message "Validating installation..."
    
    local validation_errors=0
    
    # Check Python scripts
    local scripts=(
        "$PRESENTATIONS_DIR/scripts/demo_manager.py"
        "$PRESENTATIONS_DIR/scripts/demo_scenarios.py"
        "$PRESENTATIONS_DIR/scripts/demo_recorder.py"
        "$PRESENTATIONS_DIR/scripts/demo_feedback.py"
        "$PRESENTATIONS_DIR/scripts/demo_orchestrator.py"
    )
    
    for script in "${scripts[@]}"; do
        if [ ! -f "$script" ]; then
            print_message "Missing script: $script" "$RED"
            ((validation_errors++))
        else
            log_message "Script found: $script"
        fi
    done
    
    # Check documentation files
    local docs=(
        "$PRESENTATIONS_DIR/README.md"
        "$PRESENTATIONS_DIR/pitch-decks/executive_pitch.md"
        "$PRESENTATIONS_DIR/technical-demos/technical_demo_guide.md"
        "$PRESENTATIONS_DIR/stakeholder-demos/stakeholder_demo_environments.md"
        "$PRESENTATIONS_DIR/analytics/demo_analytics_dashboard.md"
    )
    
    for doc in "${docs[@]}"; do
        if [ ! -f "$doc" ]; then
            print_message "Missing documentation: $doc" "$RED"
            ((validation_errors++))
        else
            log_message "Documentation found: $doc"
        fi
    done
    
    if [ $validation_errors -eq 0 ]; then
        print_message "Installation validation passed" "$GREEN"
    else
        print_message "Installation validation failed with $validation_errors errors" "$RED"
        return 1
    fi
}

# Function to display demo information
display_demo_info() {
    echo
    print_message "========================================" "$BLUE"
    print_message "  DEMO SETUP COMPLETE" "$BLUE"
    print_message "========================================" "$BLUE"
    echo
    print_message "Demo Environment: $PRESENTATIONS_DIR" "$GREEN"
    print_message "Demo Configuration: $PRESENTATIONS_DIR/demo_config.json" "$GREEN"
    print_message "Setup Log: $LOG_FILE" "$GREEN"
    echo
    print_message "Available Demos:" "$YELLOW"
    echo "  • Executive Demo (C-Suite): bash launch_executive_demo.sh"
    echo "  • Clinical Demo: bash launch_clinical_demo.sh"
    echo "  • All Stakeholders: bash launch_all_stakeholders.sh"
    echo
    print_message "Quick Start Commands:" "$YELLOW"
    echo "  • List scenarios: python3 scripts/demo_scenarios.py --list-scenarios"
    echo "  • Start demo: python3 scripts/demo_orchestrator.py --create-demo --demo-type cardiology --stakeholder c_suite"
    echo "  • View analytics: python3 scripts/demo_manager.py --demo-type cardiology --stakeholder c_suite"
    echo
    print_message "Demo Documentation:" "$YELLOW"
    echo "  • README: $PRESENTATIONS_DIR/README.md"
    echo "  • Executive Pitch: $PRESENTATIONS_DIR/pitch-decks/executive_pitch.md"
    echo "  • Technical Guide: $PRESENTATIONS_DIR/technical-demos/technical_demo_guide.md"
    echo "  • Stakeholder Demos: $PRESENTATIONS_DIR/stakeholder-demos/stakeholder_demo_environments.md"
    echo
    print_message "========================================" "$BLUE"
}

# Main installation function
main() {
    print_message "========================================" "$BLUE"
    print_message "  MEDICAL AI DEMO SETUP" "$BLUE"
    print_message "========================================" "$BLUE"
    echo
    
    # Check if running as correct user
    if [ "$EUID" -eq 0 ]; then
        print_message "Please do not run this script as root" "$YELLOW"
        print_message "Run as regular user with sudo access if needed" "$YELLOW"
        exit 1
    fi
    
    # Start installation process
    check_dependencies
    create_directories
    setup_databases
    install_requirements
    generate_demo_config
    create_launch_scripts
    
    # Validate installation
    if validate_installation; then
        display_demo_info
        log_message "Demo setup completed successfully"
        exit 0
    else
        log_message "Demo setup completed with errors"
        print_message "Check log file for details: $LOG_FILE" "$YELLOW"
        exit 1
    fi
}

# Run main function
main "$@"
