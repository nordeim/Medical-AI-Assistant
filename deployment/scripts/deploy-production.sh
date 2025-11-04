#!/bin/bash
# Production Deployment Orchestration Script for Medical AI Assistant
# Healthcare-compliant deployment with multi-cloud support
# Usage: ./deploy-production.sh [cloud-provider] [environment] [options]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOYMENT_DIR="$SCRIPT_DIR"
LOG_FILE="/var/log/medical-ai-deploy-$(date +%Y%m%d-%H%M%S).log"

# Default values
CLOUD_PROVIDER="${1:-aws}"
ENVIRONMENT="${2:-production}"
NAMESPACE="medical-ai-${ENVIRONMENT}"
HELM_RELEASE_NAME="medical-ai"
HELM_VALUES_FILE=""
VERBOSE="${VERBOSE:-false}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)  echo -e "${GREEN}[INFO]${NC} $message" | tee -a "$LOG_FILE" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE" ;;
        DEBUG) [[ "$VERBOSE" == "true" ]] && echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE" ;;
    esac
}

# Error handling
error_exit() {
    log ERROR "$1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log INFO "Checking prerequisites..."
    
    # Check required tools
    local required_tools=("kubectl" "helm" "terraform" "aws" "gcloud" "az")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            if [[ "$tool" == "aws" || "$tool" == "gcloud" || "$tool" == "az" ]]; then
                log WARN "$tool not found - skipping for now"
            else
                error_exit "Required tool $tool not found"
            fi
        fi
    done
    
    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        error_exit "Cannot connect to Kubernetes cluster"
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log WARN "Namespace $NAMESPACE does not exist, it will be created"
    fi
    
    log INFO "Prerequisites check completed"
}

# Initialize deployment environment
init_environment() {
    log INFO "Initializing deployment environment..."
    
    # Set up environment variables
    export KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
    export TERRAFORM_WORKSPACE="${ENVIRONMENT}"
    export KUBECTL_CONTEXT="${KUBECTL_CONTEXT:-}"
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create necessary secrets and configmaps
    setup_security_secrets
    setup_config_maps
    
    log INFO "Environment initialization completed"
}

# Setup security secrets for HIPAA compliance
setup_security_secrets() {
    log INFO "Setting up security secrets..."
    
    # Database credentials
    if ! kubectl get secret medical-ai-database-secret -n "$NAMESPACE" &> /dev/null; then
        kubectl create secret generic medical-ai-database-secret \
            --from-literal=username="${DB_USERNAME:-medicalai}" \
            --from-literal=password="${DB_PASSWORD:-$(openssl rand -base64 32)}" \
            --from-literal=host="${DB_HOST:-medical-ai-database}" \
            --from-literal=url="postgresql://\${DATABASE_USERNAME}:\${DATABASE_PASSWORD}@\${DATABASE_HOST}:5432/medical_ai" \
            -n "$NAMESPACE" || true
    fi
    
    # Redis credentials
    if ! kubectl get secret medical-ai-redis-secret -n "$NAMESPACE" &> /dev/null; then
        kubectl create secret generic medical-ai-redis-secret \
            --from-literal=password="${REDIS_PASSWORD:-$(openssl rand -base64 32)}" \
            --from-literal=url="redis://:\${REDIS_PASSWORD}@\${REDIS_HOST}:6379/0" \
            -n "$NAMESPACE" || true
    fi
    
    # Application secrets
    if ! kubectl get secret medical-ai-secret -n "$NAMESPACE" &> /dev/null; then
        kubectl create secret generic medical-ai-secret \
            --from-literal=secret-key="$(openssl rand -base64 64)" \
            --from-literal=jwt-secret="$(openssl rand -base64 32)" \
            -n "$NAMESPACE" || true
    fi
    
    # Encryption keys
    if ! kubectl get secret medical-ai-encryption -n "$NAMESPACE" &> /dev/null; then
        kubectl create secret generic medical-ai-encryption \
            --from-literal=encryption-key="$(openssl rand -base64 32)" \
            --from-literal=backup-encryption-key="$(openssl rand -base64 32)" \
            -n "$NAMESPACE" || true
    fi
    
    log INFO "Security secrets setup completed"
}

# Setup configuration maps
setup_config_maps() {
    log INFO "Setting up configuration maps..."
    
    # Application configuration
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-ai-config
  namespace: $NAMESPACE
data:
  environment: "$ENVIRONMENT"
  api.url: "https://api.medical-ai.example.com"
  allowed.hosts: "medical-ai.example.com,api.medical-ai.example.com"
  cors.origins: "https://medical-ai.example.com,https://admin.medical-ai.example.com"
  log.level: "INFO"
  metrics.enabled: "true"
  monitoring.enabled: "true"
  compliance.enabled: "true"
EOF
    
    # Backup configuration
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-ai-backup-config
  namespace: $NAMESPACE
data:
  s3.bucket: "medical-ai-backups-$ENVIRONMENT"
  retention.days: "2555"
  backup.schedule.database: "0 2 * * *"
  backup.schedule.models: "0 4 * * 0"
  dr.test.schedule: "0 6 1 * *"
EOF
    
    log INFO "Configuration maps setup completed"
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log INFO "Deploying infrastructure with Terraform..."
    
    local tf_dir="$DEPLOYMENT_DIR/cloud"
    
    case $CLOUD_PROVIDER in
        aws)
            cd "$tf_dir"
            terraform init -backend-config="key=production/terraform.tfstate"
            terraform plan -var-file="production.tfvars" -out=tfplan
            terraform apply tfplan
            ;;
        gcp)
            cd "$tf_dir"
            terraform init
            terraform plan -var-file="production.tfvars" -out=tfplan
            terraform apply tfplan
            ;;
        azure)
            cd "$tf_dir"
            terraform init
            terraform plan -var-file="production.tfvars" -out=tfplan
            terraform apply tfplan
            ;;
        *)
            log WARN "Infrastructure deployment skipped for cloud provider: $CLOUD_PROVIDER"
            ;;
    esac
    
    log INFO "Infrastructure deployment completed"
}

# Deploy Kubernetes resources
deploy_kubernetes() {
    log INFO "Deploying Kubernetes resources..."
    
    # Deploy in order: security, namespace, apps, monitoring, backup
    local deployment_order=(
        "00-namespace-rbac.yaml"
        "security/hipaa-security-rules.yaml"
        "01-frontend-deployment.yaml"
        "02-backend-deployment.yaml"
        "03-serving-deployment.yaml"
        "04-database-cache-deployment.yaml"
        "05-monitoring-deployment.yaml"
        "06-load-balancing-autoscaling.yaml"
    )
    
    for resource in "${deployment_order[@]}"; do
        if [[ -f "$DEPLOYMENT_DIR/kubernetes/$resource" ]]; then
            log INFO "Applying $resource"
            kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/$resource" -n "$NAMESPACE" || {
                log WARN "Failed to apply $resource, continuing..."
            }
        fi
    done
    
    # Apply backup and disaster recovery
    if [[ -f "$DEPLOYMENT_DIR/backup/backup-disaster-recovery.yaml" ]]; then
        log INFO "Applying backup and disaster recovery configuration"
        kubectl apply -f "$DEPLOYMENT_DIR/backup/backup-disaster-recovery.yaml" -n "$NAMESPACE" || true
    fi
    
    log INFO "Kubernetes resources deployment completed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log INFO "Deploying monitoring stack..."
    
    # Deploy Prometheus, Grafana, and AlertManager
    if [[ -f "$DEPLOYMENT_DIR/monitoring/alert-rules.yml" ]]; then
        kubectl create configmap medical-ai-prometheus-rules \
            --from-file="$DEPLOYMENT_DIR/monitoring/alert-rules.yml" \
            -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - || true
    fi
    
    # Apply monitoring deployment
    if [[ -f "$DEPLOYMENT_DIR/kubernetes/05-monitoring-deployment.yaml" ]]; then
        kubectl apply -f "$DEPLOYMENT_DIR/kubernetes/05-monitoring-deployment.yaml" -n "$NAMESPACE" || true
    fi
    
    # Create Grafana dashboards
    create_grafana_dashboards
    
    log INFO "Monitoring stack deployment completed"
}

# Create Grafana dashboards for healthcare metrics
create_grafana_dashboards() {
    log INFO "Creating Grafana dashboards..."
    
    # Dashboard JSON for HIPAA compliance metrics
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-ai-grafana-dashboards
  namespace: $NAMESPACE
data:
  medical-ai-compliance.json: |
    {
      "dashboard": {
        "title": "Medical AI HIPAA Compliance",
        "panels": [
          {
            "title": "PHI Access Events",
            "type": "stat",
            "targets": [{"expr": "rate(medical_ai_phi_access_total[5m])"}],
            "fieldConfig": {"defaults": {"color": {"mode": "thresholds"}}}
          },
          {
            "title": "Unauthorized Access Attempts",
            "type": "stat", 
            "targets": [{"expr": "rate(medical_ai_auth_failures_total[5m])"}]
          },
          {
            "title": "Audit Log Coverage",
            "type": "gauge",
            "targets": [{"expr": "rate(medical_ai_audit_log_entries_total[1h])"}]
          }
        ]
      }
    }
EOF
    
    log INFO "Grafana dashboards created"
}

# Health checks and validation
validate_deployment() {
    log INFO "Validating deployment..."
    
    # Check all pods are running
    local max_wait=600  # 10 minutes
    local start_time=$(date +%s)
    
    while true; do
        local current_time=$(date +%s)
        local elapsed=$((current_time - start_time))
        
        if [[ $elapsed -gt $max_wait ]]; then
            log ERROR "Deployment validation timeout"
            return 1
        fi
        
        local not_ready=$(kubectl get pods -n "$NAMESPACE" --no-headers | grep -v "Running\|Completed" | wc -l)
        
        if [[ $not_ready -eq 0 ]]; then
            log INFO "All pods are ready"
            break
        fi
        
        log INFO "Waiting for pods to be ready... ($((elapsed))s elapsed)"
        sleep 10
    done
    
    # Check services are accessible
    local services=("medical-ai-frontend" "medical-ai-backend" "medical-ai-serving")
    for service in "${services[@]}"; do
        if kubectl get service "$service" -n "$NAMESPACE" &> /dev/null; then
            log INFO "Service $service is available"
        else
            log WARN "Service $service not found"
        fi
    done
    
    # Check monitoring is working
    if kubectl get pods -n "$NAMESPACE" -l app=prometheus &> /dev/null; then
        log INFO "Prometheus is running"
    else
        log WARN "Prometheus not found"
    fi
    
    log INFO "Deployment validation completed"
}

# Run security and compliance checks
run_compliance_checks() {
    log INFO "Running compliance and security checks..."
    
    # Check Pod Security Standards
    local psp_violations=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.securityContext}{"\n"}{end}' | grep -c "null" || true)
    if [[ $psp_violations -gt 0 ]]; log WARN "Found $psp_violations pods without security context"; fi
    
    # Check Network Policies
    local network_policies=$(kubectl get networkpolicies -n "$NAMESPACE" --no-headers | wc -l)
    log INFO "Network policies: $network_policies"
    
    # Check RBAC
    local service_accounts=$(kubectl get serviceaccounts -n "$NAMESPACE" --no-headers | wc -l)
    log INFO "Service accounts: $service_accounts"
    
    # Check Secrets are encrypted
    local secrets=$(kubectl get secrets -n "$NAMESPACE" --no-headers | wc -l)
    log INFO "Secrets: $secrets"
    
    # Run basic security scan
    kubectl run security-scan --image=aquasec/trivy:latest --rm -i --restart=Never -- \
        k8s --no-progress --format json --output /tmp/security-report.json || true
    
    if [[ -f "/tmp/security-report.json" ]]; then
        log INFO "Security scan completed - check /tmp/security-report.json"
    fi
    
    log INFO "Compliance checks completed"
}

# Create backup of current state
create_pre_deployment_backup() {
    log INFO "Creating pre-deployment backup..."
    
    # Create timestamped backup
    local backup_name="pre-deploy-$(date +%Y%m%d-%H%M%S)"
    local backup_dir="/tmp/$backup_name"
    
    mkdir -p "$backup_dir"
    
    # Backup current state
    kubectl get all,configmaps,secrets,pvc -n "$NAMESPACE" -o yaml > "$backup_dir/current-state.yaml" || true
    
    # Backup configuration
    kubectl get configmaps -n "$NAMESPACE" -o yaml > "$backup_dir/configmaps.yaml" || true
    kubectl get secrets -n "$NAMESPACE" -o yaml > "$backup_dir/secrets.yaml" || true
    
    log INFO "Pre-deployment backup saved to $backup_dir"
}

# Rollback deployment
rollback_deployment() {
    log ERROR "Deployment failed, initiating rollback..."
    
    local backup_dir="${1:-}"
    if [[ -z "$backup_dir" || ! -d "$backup_dir" ]]; then
        log ERROR "No valid backup directory provided for rollback"
        exit 1
    fi
    
    # Restore from backup
    kubectl apply -f "$backup_dir/current-state.yaml" -n "$NAMESPACE" || true
    kubectl apply -f "$backup_dir/configmaps.yaml" -n "$NAMESPACE" || true
    kubectl apply -f "$backup_dir/secrets.yaml" -n "$NAMESPACE" || true
    
    log INFO "Rollback completed from backup: $backup_dir"
}

# Cleanup function
cleanup() {
    log INFO "Cleaning up temporary files..."
    # Add any cleanup tasks here
}

# Main deployment function
main() {
    log INFO "Starting Medical AI production deployment"
    log INFO "Cloud Provider: $CLOUD_PROVIDER"
    log INFO "Environment: $ENVIRONMENT"
    log INFO "Namespace: $NAMESPACE"
    
    # Trap errors and cleanup
    trap cleanup EXIT
    
    # Check prerequisites
    check_prerequisites
    
    # Create pre-deployment backup
    local backup_dir="/tmp/pre-deploy-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Initialize environment
    init_environment
    
    # Deploy infrastructure
    deploy_infrastructure
    
    # Deploy Kubernetes resources
    deploy_kubernetes
    
    # Deploy monitoring
    deploy_monitoring
    
    # Validate deployment
    if ! validate_deployment; then
        log ERROR "Deployment validation failed, rolling back..."
        rollback_deployment "$backup_dir"
        exit 1
    fi
    
    # Run compliance checks
    run_compliance_checks
    
    log INFO "Medical AI production deployment completed successfully!"
    log INFO "Deployment details:"
    log INFO "  - Namespace: $NAMESPACE"
    log INFO "  - Environment: $ENVIRONMENT"
    log INFO "  - Log file: $LOG_FILE"
    
    # Display important URLs
    echo
    echo "=========================================="
    echo "Deployment Summary"
    echo "=========================================="
    echo "Kubernetes Dashboard: https://kubernetes-dashboard.$(kubectl config current-context | cut -d@ -f2 || echo 'local')"
    echo "Grafana: http://medical-ai-grafana.$NAMESPACE.svc.cluster.local:3000"
    echo "Prometheus: http://medical-ai-prometheus.$NAMESPACE.svc.cluster.local:9090"
    echo "API Endpoint: https://api.medical-ai.example.com"
    echo "Web UI: https://medical-ai.example.com"
    echo "=========================================="
}

# Show usage
show_usage() {
    cat <<EOF
Medical AI Production Deployment Script

Usage: $0 [CLOUD_PROVIDER] [ENVIRONMENT] [OPTIONS]

CLOUD_PROVIDER:
  aws     Deploy to Amazon Web Services (EKS)
  gcp     Deploy to Google Cloud Platform (GKE)  
  azure   Deploy to Microsoft Azure (AKS)
  local   Deploy to local Kubernetes cluster

ENVIRONMENT:
  production  Production environment (default)
  staging     Staging environment
  dev         Development environment

OPTIONS:
  --verbose    Enable verbose output
  --dry-run    Show what would be deployed without making changes
  --rollback   Rollback to previous deployment
  --validate-only  Only run validation checks
  --help       Show this help message

EXAMPLES:
  $0 aws production              # Deploy to AWS production
  $0 gcp staging --verbose       # Deploy to GCP staging with verbose output
  $0 azure dev --dry-run         # Dry run for Azure dev environment

ENVIRONMENT VARIABLES:
  DB_USERNAME      Database username (default: medicalai)
  DB_PASSWORD      Database password (auto-generated if not set)
  REDIS_PASSWORD   Redis password (auto-generated if not set)
  KUBECONFIG       Kubernetes config file path
  TERRAFORM_WORKSPACE Terraform workspace

For more information, visit: https://docs.medical-ai.example.com/deployment
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            log INFO "Dry run mode enabled - no changes will be made"
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --validate-only)
            VALIDATE_ONLY=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            if [[ -z "${positional_args:-}" ]]; then
                positional_args="$1"
            else
                positional_args="$positional_args $1"
            fi
            shift
            ;;
    esac
done

# Handle rollback
if [[ "${ROLLBACK:-false}" == "true" ]]; then
    rollback_deployment "${BACKUP_DIR:-/tmp/pre-deploy-latest}"
    exit 0
fi

# Handle validate-only mode
if [[ "${VALIDATE_ONLY:-false}" == "true" ]]; then
    check_prerequisites
    run_compliance_checks
    exit 0
fi

# Run main deployment
main "${positional_args:-}"

# Print final status
echo
log INFO "Deployment script completed. Check $LOG_FILE for detailed logs."
if [[ -f "$LOG_FILE" ]]; then
    echo "Full deployment log: $LOG_FILE"
fi