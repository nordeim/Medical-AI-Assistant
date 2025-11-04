#!/bin/bash
# Healthcare Monitoring Setup Script for Medical AI Assistant
# Sets up Prometheus, Grafana, AlertManager with medical compliance alerts
# Usage: ./setup-monitoring.sh [namespace] [environment]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${1:-medical-ai-prod}"
ENVIRONMENT="${2:-production}"

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging
log() {
    local level=$1
    shift
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} ${level}: $*" >&2
}

info() { log "${GREEN}INFO${NC} $*" ; }
warn() { log "${YELLOW}WARN${NC} $*" ; }
error() { log "${RED}ERROR${NC} $*" ; }

# Check prerequisites
check_prerequisites() {
    info "Checking prerequisites..."
    
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found"
        exit 1
    fi
    
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    info "Prerequisites check passed"
}

# Create monitoring namespace if it doesn't exist
setup_monitoring_namespace() {
    info "Setting up monitoring namespace..."
    
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - || true
    
    # Apply Pod Security Standards
    kubectl apply -f - <<EOF || true
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    name: monitoring
    environment: $ENVIRONMENT
    compliance: hipaa
EOF
    
    info "Monitoring namespace ready"
}

# Install Prometheus Operator
install_prometheus_operator() {
    info "Installing Prometheus Operator..."
    
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
    helm repo update
    
    # Install Prometheus Operator with healthcare settings
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
        --set grafana.adminPassword="medicalai-admin-$(date +%s)" \
        --set prometheus.prometheusSpec.evaluationInterval=15s \
        --set prometheus.prometheusSpec.scrapeInterval=15s \
        --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.probeSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.retentionSize=50GB \
        --set prometheus.prometheusSpec.externalUrl="http://prometheus.monitoring.svc.cluster.local:9090" \
        --set alertmanager.config.global.smtp_smarthost="smtp.gmail.com:587" \
        --set alertmanager.config.global.smtp_from="alerts@medical-ai.example.com" \
        --set alertmanager.alertmanagerSpec.image.tag=v0.26.0 \
        --set prometheus.prometheusSpec.image.tag=v2.47.0 \
        --set prometheus.prometheusSpec.externalLabels.cluster="$ENVIRONMENT" \
        --set prometheus.prometheusSpec.externalLabels.region="us-east-1" \
        --set prometheus.prometheusSpec.externalLabels.environment="$ENVIRONMENT" \
        --set prometheus.prometheusSpec.resources.limits.cpu=2 \
        --set prometheus.prometheusSpec.resources.limits.memory=8Gi \
        --set prometheus.prometheusSpec.resources.requests.cpu=500m \
        --set prometheus.prometheusSpec.resources.requests.memory=2Gi
    
    info "Prometheus Operator installed"
}

# Configure healthcare-specific ServiceMonitors
setup_service_monitors() {
    info "Setting up ServiceMonitors for Medical AI services..."
    
    # Backend ServiceMonitor
    cat <<EOF | kubectl apply -f - || true
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: medical-ai-backend
  namespace: $NAMESPACE
  labels:
    app: medical-ai
    component: backend
    compliance: hipaa
spec:
  selector:
    matchLabels:
      app: medical-ai
      component: backend
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
    targetPort: 8000
  - port: http
    path: /metrics/compliance
    interval: 60s
    scrapeTimeout: 15s
    targetPort: 8000
  - port: http
    path: /metrics/security
    interval: 30s
    scrapeTimeout: 15s
    targetPort: 8000
EOF

    # Frontend ServiceMonitor
    cat <<EOF | kubectl apply -f - || true
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: medical-ai-frontend
  namespace: $NAMESPACE
  labels:
    app: medical-ai
    component: frontend
    compliance: hipaa
spec:
  selector:
    matchLabels:
      app: medical-ai
      component: frontend
  endpoints:
  - port: http
    path: /nginx-status
    interval: 30s
    scrapeTimeout: 10s
    targetPort: 8080
EOF

    # Serving ServiceMonitor
    cat <<EOF | kubectl apply -f - || true
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: medical-ai-serving
  namespace: $NAMESPACE
  labels:
    app: medical-ai
    component: serving
    compliance: fda
spec:
  selector:
    matchLabels:
      app: medical-ai
      component: serving
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
    scrapeTimeout: 10s
    targetPort: 8081
  - port: grpc
    path: /metrics
    interval: 30s
    scrapeTimeout: 10s
    targetPort: 9090
EOF

    info "ServiceMonitors configured"
}

# Create custom medical alert rules
setup_alert_rules() {
    info "Setting up healthcare alert rules..."
    
    if [[ -f "$SCRIPT_DIR/../monitoring/alert-rules.yml" ]]; then
        kubectl create configmap medical-ai-alert-rules \
            --from-file="$SCRIPT_DIR/../monitoring/alert-rules.yml" \
            -n "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - || true
    fi
    
    # Additional HIPAA-specific alerts
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: hipaa-compliance-alerts
  namespace: $NAMESPACE
data:
  hipaa-alerts.yml: |
    groups:
    - name: hipaa_security_alerts
      rules:
      - alert: PHIDataExfiltration
        expr: rate(medical_ai_phi_data_transferred_bytes[5m]) > 1000000000
        for: 2m
        labels:
          severity: critical
          compliance: hipaa
          alert_group: data_protection
        annotations:
          summary: "Potential PHI data exfiltration detected"
          description: "Large amount of PHI data being transferred"
          runbook_url: "https://runbooks.medical-ai.example.com/phi-exfiltration"
          
      - alert: UnauthorizedDatabaseAccess
        expr: rate(medical_ai_db_unauthorized_access_total[5m]) > 0
        for: 0m
        labels:
          severity: critical
          compliance: hipaa
          alert_group: security
        annotations:
          summary: "Unauthorized database access detected"
          description: "Database access from unauthorized sources detected"
          runbook_url: "https://runbooks.medical-ai.example.com/unauthorized-db-access"

    - name: fda_clinical_safety_alerts
      rules:
      - alert: ModelClinicalFailure
        expr: medical_ai_clinical_decision_accuracy < 0.85
        for: 5m
        labels:
          severity: critical
          compliance: fda
          alert_group: clinical_safety
        annotations:
          summary: "Model clinical decision accuracy below FDA threshold"
          description: "Clinical decision accuracy is {{ \$value }}, below 85% FDA threshold"
          runbook_url: "https://runbooks.medical-ai.example.com/clinical-accuracy"
          
      - alert: PatientSafetyOverride
        expr: medical_ai_patient_safety_override_count > 0
        for: 0m
        labels:
          severity: critical
          compliance: fda
          alert_group: patient_safety
        annotations:
          summary: "Patient safety override detected"
          description: "System override affecting patient safety detected"
          runbook_url: "https://runbooks.medical-ai.example.com/patient-safety-override"

    - name: system_availability_alerts
      rules:
      - alert: ServiceDowntime
        expr: up{job=~"medical-ai.*"} == 0
        for: 1m
        labels:
          severity: critical
          compliance: hipaa
          alert_group: availability
        annotations:
          summary: "Medical AI service is down"
          description: "{{ \$labels.job }} service has been down for more than 1 minute"
          runbook_url: "https://runbooks.medical-ai.example.com/service-downtime"

    - name: iso27001_security_alerts
      rules:
      - alert: SecurityIncidentDetected
        expr: rate(medical_ai_security_incidents_total[5m]) > 0
        for: 0m
        labels:
          severity: critical
          compliance: iso27001
          alert_group: security_incidents
        annotations:
          summary: "Security incident detected"
          description: "Security incident detected in Medical AI system"
          runbook_url: "https://runbooks.medical-ai.example.com/security-incident"
          
      - alert: EncryptionFailure
        expr: medical_ai_encryption_status != 1
        for: 1m
        labels:
          severity: critical
          compliance: iso27001
          alert_group: encryption
        annotations:
          summary: "Data encryption failure detected"
          description: "Encryption status check failed"
          runbook_url: "https://runbooks.medical-ai.example.com/encryption-failure"
EOF
    
    info "Healthcare alert rules configured"
}

# Configure AlertManager for healthcare notifications
setup_alertmanager() {
    info "Configuring AlertManager for healthcare notifications..."
    
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: Secret
metadata:
  name: alertmanager-config
  namespace: $NAMESPACE
type: Opaque
data:
  alertmanager.yml: |
    global:
      smtp_smarthost: 'smtp.gmail.com:587'
      smtp_from: 'alerts@medical-ai.example.com'
      smtp_auth_username: 'alerts@medical-ai.example.com'
      smtp_auth_password: 'YOUR_SMTP_PASSWORD'
      slack_api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    
    route:
      group_by: ['alertname', 'cluster', 'service', 'compliance']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
      routes:
      - match:
          severity: critical
          compliance: hipaa
        receiver: 'hipaa-critical'
      - match:
          severity: critical
          compliance: fda
        receiver: 'fda-critical'
      - match:
          severity: critical
          compliance: iso27001
        receiver: 'security-critical'
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://medical-ai-notification-service:8080/alerts'
    
    - name: 'hipaa-critical'
      email_configs:
      - to: 'hipaa-compliance@medical-ai.example.com'
        subject: '🚨 HIPAA Compliance Alert - Medical AI'
        body: |
          HIPAA Compliance Alert Detected:
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Component: {{ .Labels.component }}
          Time: {{ .StartsAt.Format "2006-01-02 15:04:05" }}
          
          Action Required: {{ .Annotations.action_required }}
          Runbook: {{ .Annotations.runbook_url }}
        headers:
          Subject: '🚨 HIPAA Alert: {{ .GroupLabels.alertname }}'
      slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#hipaa-compliance'
        title: '🚨 HIPAA Compliance Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'danger'
    
    - name: 'fda-critical'
      email_configs:
      - to: 'fda-regulatory@medical-ai.example.com'
        subject: '⚠️ FDA Clinical Safety Alert - Medical AI'
        body: |
          FDA Clinical Safety Alert Detected:
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Component: {{ .Labels.component }}
          Time: {{ .StartsAt.Format "2006-01-02 15:04:05" }}
          
          Action Required: Immediate clinical review
          Runbook: {{ .Annotations.runbook_url }}
        headers:
          Subject: '⚠️ FDA Alert: {{ .GroupLabels.alertname }}'
      slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#fda-safety'
        title: '⚠️ FDA Clinical Safety Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'warning'
    
    - name: 'security-critical'
      email_configs:
      - to: 'security-team@medical-ai.example.com'
        subject: '🔒 Security Incident - Medical AI'
        body: |
          Security Incident Detected:
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Component: {{ .Labels.component }}
          Time: {{ .StartsAt.Format "2006-01-02 15:04:05" }}
          
          Action Required: Immediate security investigation
          Runbook: {{ .Annotations.runbook_url }}
        headers:
          Subject: '🔒 Security Alert: {{ .GroupLabels.alertname }}'
      slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        channel: '#security-incidents'
        title: '🔒 Security Incident Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        color: 'danger'
    
    inhibit_rules:
    - source_match:
        severity: 'critical'
      target_match:
        severity: 'warning'
      equal: ['alertname', 'cluster', 'service']
    
    templates:
    - '/etc/alertmanager/templates/*.tmpl'
EOF
    
    info "AlertManager configured"
}

# Create Grafana dashboards for healthcare metrics
setup_grafana_dashboards() {
    info "Setting up Grafana dashboards..."
    
    # Compliance Dashboard
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: medical-ai-grafana-dashboards
  namespace: $NAMESPACE
  labels:
    grafana_dashboard: "1"
data:
  compliance-dashboard.json: |
    {
      "dashboard": {
        "title": "Medical AI Compliance Dashboard",
        "tags": ["compliance", "hipaa", "fda", "iso27001"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "HIPAA PHI Access Events",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(medical_ai_phi_access_total[5m])",
                "legendFormat": "PHI Access Rate"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "yellow", "value": 10},
                    {"color": "red", "value": 50}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "Unauthorized Access Attempts",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(medical_ai_auth_failures_total[5m])",
                "legendFormat": "Auth Failures"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "color": {"mode": "thresholds"},
                "thresholds": {
                  "steps": [
                    {"color": "green", "value": null},
                    {"color": "red", "value": 1}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Model Clinical Accuracy",
            "type": "stat",
            "targets": [
              {
                "expr": "medical_ai_clinical_decision_accuracy",
                "legendFormat": "Accuracy"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "color": {"mode": "thresholds"},
                "min": 0,
                "max": 1,
                "thresholds": {
                  "steps": [
                    {"color": "red", "value": null},
                    {"color": "yellow", "value": 0.85},
                    {"color": "green", "value": 0.95}
                  ]
                },
                "unit": "percentunit"
              }
            },
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
          },
          {
            "id": 4,
            "title": "Security Incidents Timeline",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(medical_ai_security_incidents_total[1h])",
                "legendFormat": "Security Incidents"
              }
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 16}
          }
        ]
      }
    }

  system-performance-dashboard.json: |
    {
      "dashboard": {
        "title": "Medical AI System Performance",
        "tags": ["performance", "slo"],
        "panels": [
          {
            "id": 1,
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {"expr": "rate(medical_ai_requests_total[5m])", "legendFormat": "{{service}}"}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "Response Time (95th percentile)",
            "type": "graph",
            "targets": [
              {"expr": "histogram_quantile(0.95, rate(medical_ai_request_duration_seconds_bucket[5m]))", "legendFormat": "95th percentile"}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Error Rate",
            "type": "graph",
            "targets": [
              {"expr": "rate(medical_ai_requests_total{status!~\"2..\"}[5m]) / rate(medical_ai_requests_total[5m])", "legendFormat": "Error Rate"}
            ],
            "fieldConfig": {
              "defaults": {"unit": "percentunit"}
            },
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
          }
        ]
      }
    }

  infrastructure-dashboard.json: |
    {
      "dashboard": {
        "title": "Medical AI Infrastructure",
        "tags": ["infrastructure", "kubernetes"],
        "panels": [
          {
            "id": 1,
            "title": "CPU Usage",
            "type": "graph",
            "targets": [
              {"expr": "rate(container_cpu_usage_seconds_total{namespace=\"$NAMESPACE\"}[5m])", "legendFormat": "{{pod}}"}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "Memory Usage",
            "type": "graph",
            "targets": [
              {"expr": "container_memory_usage_bytes{namespace=\"$NAMESPACE\"} / 1024/1024/1024", "legendFormat": "{{pod}}"}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Pod Status",
            "type": "stat",
            "targets": [
              {"expr": "kube_pod_status_ready{namespace=\"$NAMESPACE\"}", "legendFormat": "Ready"}
            ],
            "gridPos": {"h": 6, "w": 24, "x": 0, "y": 16}
          }
        ]
      }
    }
EOF
    
    info "Grafana dashboards configured"
}

# Setup custom metrics collection
setup_custom_metrics() {
    info "Setting up custom healthcare metrics..."
    
    # Database monitoring
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: database-monitoring-config
  namespace: $NAMESPACE
data:
  postgres-exporter.yml: |
    data_source_names:
      postgres: "postgresql://\${DATABASE_USERNAME}:\${DATABASE_PASSWORD}@\${DATABASE_HOST}:5432/medical_ai?sslmode=disable"
    
    queries:
      pg_stat_database:
        query: |
          SELECT datname, numbackends, xact_commit, xact_rollback, blks_read, blks_hit, 
                 tup_returned, tup_fetched, tup_inserted, tup_updated, tup_deleted,
                 conflicts, temp_files, temp_bytes, deadlocks, blk_read_time, blk_write_time
          FROM pg_stat_database
          WHERE datname NOT IN ('template0', 'template1', 'postgres')
        metrics:
          - datname:
              usage: "LABEL"
              description: "Name of the database"
          - numbackends:
              usage: "GAUGE"
              description: "Number of backends currently connected to this database"
          - xact_commit:
              usage: "COUNTER"
              description: "Number of transactions that have been committed"
          - xact_rollback:
              usage: "COUNTER"
              description: "Number of transactions that have been rolled back"
EOF
    
    # Redis monitoring
    cat <<EOF | kubectl apply -f - || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-monitoring-config
  namespace: $NAMESPACE
data:
  redis-exporter.yml: |
    redis.addr: "\${REDIS_HOST}:\${REDIS_PORT}"
    redis.password: "\${REDIS_PASSWORD}"
    
    metrics:
      - redis_keyspace_hits
      - redis_keyspace_misses
      - redis_connected_clients
      - redis_connected_slaves
      - redis_used_memory
      - redis_used_memory_human
      - redis_used_memory_peak
      - redis_instantaneous_input_kbps
      - redis_instantaneous_output_kbps
      - redis_total_commands_processed
      - redis_total_connections_received
      - redis_total_net_input_bytes
      - redis_total_net_output_bytes
EOF
    
    info "Custom metrics configured"
}

# Configure log aggregation
setup_log_aggregation() {
    info "Setting up log aggregation..."
    
    # Install fluentd/fluent-bit for log collection
    helm repo add fluent https://fluent.github.io/helm-charts || true
    helm repo update
    
    # Install fluent-bit with healthcare compliance
    helm upgrade --install fluent-bit fluent/fluent-bit \
        --namespace logging \
        --create-namespace \
        --set createNamespace=true \
        --set config.enabled=true \
        --set config.inputs=$(cat <<'EOF'
[INPUT]
    Name tail
    Path /var/log/containers/*medical-ai*.log
    Parser docker
    Tag medical.ai.*
    Refresh_Interval 5
    
[INPUT]
    Name tail
    Path /var/log/containers/*_backend*.log
    Parser docker
    Tag medical.backend.*
    Refresh_Interval 5
EOF
) \
        --set config.filters=$(cat <<'EOF'
[FILTER]
    Name grep
    Match medical.ai.*
    Regex log \{".*"compliance":"hipaa".*\}

[FILTER]
    Name kubernetes
    Match medical.*
    Kube_Tag_Prefix medical.
    Kube_URL https://kubernetes.default.svc:443
    Kube_CA_File /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File /var/run/secrets/kubernetes.io/serviceaccount/token
EOF
) \
        --set config.outputs=$(cat <<'EOF'
[OUTPUT]
    Name es
    Match medical.*
    Host elasticsearch.monitoring.svc.cluster.local
    Port 9200
    Index medical-ai-logs
    Type _doc
    Time_Key @timestamp
    Retry_Limit 3
    Buffer_Size 1MB
    Buffer_Max_Size 4MB
    Buffer_Chunk_Size 512KB
EOF
)
    
    info "Log aggregation configured"
}

# Health check and validation
validate_monitoring_setup() {
    info "Validating monitoring setup..."
    
    # Wait for Prometheus to be ready
    kubectl wait --for=condition=Ready pod -l app=kube-prometheus-stack-prometheus -n monitoring --timeout=300s || true
    
    # Check Grafana
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=grafana -n monitoring --timeout=300s || true
    
    # Check AlertManager
    kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=alertmanager -n monitoring --timeout=300s || true
    
    # Validate services
    local services=("prometheus-operated" "alertmanager-operated" "grafana")
    for service in "${services[@]}"; do
        if kubectl get service "$service" -n monitoring &> /dev/null; then
            info "Service $service is available"
        else
            warn "Service $service not found"
        fi
    done
    
    info "Monitoring setup validation completed"
}

# Main function
main() {
    info "Starting Medical AI monitoring setup"
    info "Namespace: $NAMESPACE"
    info "Environment: $ENVIRONMENT"
    
    check_prerequisites
    setup_monitoring_namespace
    install_prometheus_operator
    setup_service_monitors
    setup_alert_rules
    setup_alertmanager
    setup_grafana_dashboards
    setup_custom_metrics
    setup_log_aggregation
    validate_monitoring_setup
    
    info "Medical AI monitoring setup completed successfully!"
    
    echo
    echo "=========================================="
    echo "Monitoring Setup Complete"
    echo "=========================================="
    echo "Prometheus: http://prometheus.monitoring.svc.cluster.local:9090"
    echo "Grafana: http://grafana.monitoring.svc.cluster.local:3000"
    echo "AlertManager: http://alertmanager.monitoring.svc.cluster.local:9093"
    echo "=========================================="
}

# Show usage
show_usage() {
    cat <<EOF
Medical AI Monitoring Setup Script

Usage: $0 [NAMESPACE] [ENVIRONMENT]

NAMESPACE: Kubernetes namespace (default: medical-ai-prod)
ENVIRONMENT: Environment name (default: production)

This script sets up:
- Prometheus with healthcare-specific metrics
- Grafana with compliance dashboards  
- AlertManager with medical compliance alerts
- Log aggregation for audit trails
- Custom healthcare monitoring configurations

Example: $0 medical-ai-prod production
EOF
}

# Handle help
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"