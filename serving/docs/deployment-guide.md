# Deployment Guide - Production Medical Environments

## Overview

Comprehensive deployment guide for the Medical AI Serving System in production medical environments, ensuring HIPAA compliance, FDA regulatory requirements, and clinical safety standards.

## 🏥 Medical Compliance Requirements

### Regulatory Framework
- **HIPAA (Health Insurance Portability and Accountability Act)**
- **FDA 21 CFR Part 820** (Quality System Regulation)
- **ISO 13485** (Medical Device Quality Management)
- **IEC 62304** (Medical Device Software Life Cycle Processes)
- **HITECH Act** (Health Information Technology for Economic and Clinical Health)

### Pre-Deployment Checklist

#### Legal & Compliance
- [ ] HIPAA compliance assessment completed
- [ ] Business Associate Agreements (BAA) signed
- [ ] FDA registration (if required) completed
- [ ] State medical device licensing verified
- [ ] Institutional Review Board (IRB) approval obtained
- [ ] Clinical validation studies documented

#### Technical Security
- [ ] End-to-end encryption implemented
- [ ] PHI access controls configured
- [ ] Audit logging systems enabled
- [ ] Network security hardening completed
- [ ] Disaster recovery procedures tested
- [ ] Incident response plan established

#### Quality Management
- [ ] Quality management system (QMS) established
- [ ] Risk assessment documentation complete
- [ ] Change control procedures implemented
- [ ] Training materials developed
- [ ] Maintenance schedules established
- [ ] Post-market surveillance plan ready

## Infrastructure Requirements

### Production Environment Specifications

#### Minimum Hardware Requirements
```yaml
# Production Medical AI Infrastructure
Server Configuration:
  - CPU: 16 cores, 2.5GHz minimum
  - Memory: 64GB RAM minimum (128GB recommended)
  - Storage: 1TB NVMe SSD (2TB recommended)
  - GPU: NVIDIA A100 or V100 (for ML workloads)
  - Network: 10Gbps network interface
  - Redundancy: Active-active configuration

Database Servers:
  - Primary: PostgreSQL 14+ with encryption
  - Backup: Real-time replication
  - Storage: 5TB with automated backup
  - Recovery: Point-in-time recovery enabled

Cache Layer:
  - Redis Cluster: 3-node minimum
  - Memory: 64GB per node
  - Persistence: AOF + RDB enabled
  - Replication: Master-slave setup

Load Balancers:
  - Hardware or software load balancer
  - SSL/TLS termination
  - Health check capabilities
  - Session persistence
```

#### Network Architecture
```
Internet → WAF → Load Balancer → API Gateway → Medical AI Services
                     ↓
                 Database Cluster
                     ↓
                 Cache Cluster
                     ↓
                 Monitoring & Logging
```

### Security Infrastructure

#### Network Security
```yaml
# Network Security Configuration
Firewall Rules:
  - Allow: HTTPS (443) from authorized medical networks
  - Allow: SSH (22) from management network only
  - Deny: All other inbound traffic
  - Monitor: All outbound traffic for data exfiltration

VPN Requirements:
  - Site-to-site VPN for medical institutions
  - Client VPN for remote clinical access
  - Multi-factor authentication required
  - Session timeout: 30 minutes maximum

Network Segmentation:
  - DMZ: Load balancers and API gateways
  - Application Tier: Medical AI services
  - Database Tier: Encrypted databases
  - Management Tier: Monitoring and administration
```

#### Data Protection
```yaml
# Data Encryption Standards
Encryption at Rest:
  - Database: AES-256 encryption
  - File Storage: Client-side encryption
  - Backup Storage: Encrypted backups
  - Cache: Redis AUTH + encryption

Encryption in Transit:
  - TLS 1.3 minimum
  - Certificate management: Automated renewal
  - Mutual TLS for inter-service communication
  - HSTS headers enabled

Key Management:
  - Hardware Security Module (HSM)
  - Key rotation: Every 90 days
  - Audit trail for all key access
  - Separate keys for production/staging
```

## Deployment Architecture

### Production Deployment Topology

```yaml
# Medical AI Production Architecture
Components:
  Web Tier:
    - 3x Web servers (load balanced)
    - Nginx reverse proxy
    - SSL/TLS termination
    - Rate limiting

  Application Tier:
    - 5x API servers (auto-scaling)
    - Medical AI inference services
    - Background job processors
    - Cache integration

  Data Tier:
    - 3x PostgreSQL servers (master-replica)
    - 3x Redis cache servers
    - Backup storage system
    - Audit log storage

  Monitoring Tier:
    - Prometheus monitoring
    - Grafana dashboards
    - ELK stack for logging
    - Alert management system
```

### Container Deployment (Docker/Kubernetes)

#### Production Dockerfile
```dockerfile
# Production Medical AI Container
FROM python:3.11-slim

# Security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r medicalai && \
    useradd -r -g medicalai -d /app -s /bin/bash medicalai

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set permissions
RUN chown -R medicalai:medicalai /app
USER medicalai

# Health check for medical compliance
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run with production settings
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "api.main:app"]
```

#### Kubernetes Deployment
```yaml
# Medical AI Production Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-ai-api
  labels:
    app: medical-ai-api
    compliance: hipaa
spec:
  replicas: 5
  selector:
    matchLabels:
      app: medical-ai-api
  template:
    metadata:
      labels:
        app: medical-ai-api
        compliance: hipaa
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api
        image: medical-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: cache-secret
              key: url
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: medical-ai-service
  labels:
    app: medical-ai-api
spec:
  selector:
    app: medical-ai-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

## Environment Configuration

### Production Environment Variables
```bash
# Production Medical AI Environment Configuration

# Application Settings
APP_ENVIRONMENT=production
APP_DEBUG=false
APP_LOG_LEVEL=INFO

# Security Settings
SECRET_KEY=your_production_secret_key_minimum_32_chars
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_aes_encryption_key

# Database Configuration
DATABASE_URL=postgresql://user:password@db-server:5432/medical_ai
DATABASE_SSL=true
DATABASE_SSL_MODE=require

# Cache Configuration
REDIS_URL=redis://cache-server:6379/0
REDIS_PASSWORD=secure_redis_password

# Medical Compliance Settings
MEDICAL_ENABLE_ENCRYPTION=true
MEDICAL_PHI_REDACTION=true
MEDICAL_ENABLE_AUDIT_LOG=true
MEDICAL_COMPLIANCE_LEVEL=production

# API Security
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=3600
ENABLE_API_KEY_AUTH=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project

# Regulatory Compliance
FDA_COMPLIANCE_MODE=true
AUDIT_LOG_ENCRYPTION=true
PHI_ENCRYPTION_AT_REST=true

# Performance Tuning
MODEL_CACHE_SIZE=100
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=100
ENABLE_GPU_ACCELERATION=true
```

### Security Configuration
```yaml
# Security Configuration for Medical Environment
Security Headers:
  Strict-Transport-Security: "max-age=31536000; includeSubDomains"
  X-Content-Type-Options: "nosniff"
  X-Frame-Options: "DENY"
  X-XSS-Protection: "1; mode=block"
  Content-Security-Policy: "default-src 'self'"

CORS Configuration:
  allowed_origins:
    - "https://medical-institution-1.example.com"
    - "https://medical-institution-2.example.com"
  allowed_methods: ["GET", "POST", "PUT", "DELETE"]
  allowed_headers: ["Authorization", "Content-Type", "X-Client-ID"]
  expose_headers: ["X-RateLimit-Remaining", "X-Request-ID"]

Rate Limiting:
  medical_inference: 60  # requests per minute
  batch_processing: 10   # requests per minute
  health_checks: 300     # requests per minute
  clinical_support: 30   # requests per minute

Access Control:
  require_api_key: true
  require_mfa: true
  session_timeout: 1800  # 30 minutes
  max_session_duration: 28800  # 8 hours
```

## Database Setup

### Production Database Configuration

#### PostgreSQL Setup
```sql
-- Medical AI Production Database Setup
-- HIPAA-compliant PostgreSQL configuration

-- Create database with encryption
CREATE DATABASE medical_ai 
WITH 
    ENCODING 'UTF8' 
    LC_COLLATE 'en_US.UTF-8' 
    LC_CTYPE 'en_US.UTF-8'
    TEMPLATE template0;

-- Enable encryption extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Create application user with minimal privileges
CREATE USER medical_ai_app WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE medical_ai TO medical_ai_app;
GRANT USAGE ON SCHEMA public TO medical_ai_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO medical_ai_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO medical_ai_app;

-- Audit logging configuration
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(255) NOT NULL,
    table_name VARCHAR(255),
    record_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    medical_domain VARCHAR(100)
);

-- Enable row-level security
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- Create audit log policy
CREATE POLICY audit_log_policy ON audit_log
    FOR ALL
    TO medical_ai_app
    USING (true);

-- Create indexes for performance
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX idx_audit_user_id ON audit_log(user_id);
CREATE INDEX idx_audit_action ON audit_log(action);

-- Configure PostgreSQL for medical data
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET log_statement = 'mod';
ALTER SYSTEM SET log_duration = on;
ALTER SYSTEM SET log_lock_waits = on;
ALTER SYSTEM SET log_min_duration_statement = 1000;
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = 'server.crt';
ALTER SYSTEM SET ssl_key_file = 'server.key';
ALTER SYSTEM SET password_encryption = 'scram-sha-256';
```

#### Redis Configuration
```redis
# Medical AI Redis Production Configuration
# redis.conf for HIPAA-compliant cache

# Security settings
requirepass secure_redis_password
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command SHUTDOWN "SHUTDOWN_MEDICAL_AI"

# Memory management
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence configuration
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
appendonly yes
appendfsync everysec

# Logging
loglevel notice
logfile /var/log/redis/medical-ai.log

# Network security
bind 127.0.0.1 10.0.0.0/24
protected-mode yes
port 6379

# Timeout settings
timeout 300
tcp-keepalive 60
```

## SSL/TLS Configuration

### Production SSL Setup
```bash
#!/bin/bash
# Medical AI SSL Certificate Setup

# Generate private key
openssl genrsa -out medical-ai-key.pem 4096

# Create certificate signing request
openssl req -new -key medical-ai-key.pem -out medical-ai.csr \
  -subj "/C=US/ST=State/L=City/O=Medical Institution/CN=api.medical-ai.example.com"

# Submit to CA and get certificate
# openssl x509 -req -in medical-ai.csr -CA ca-cert.pem -CAkey ca-key.pem \
#   -out medical-ai-cert.pem -days 365 -CAcreateserial

# For testing, create self-signed certificate
openssl x509 -req -in medical-ai.csr -signkey medical-ai-key.pem \
  -out medical-ai-cert.pem -days 365

# Set proper permissions
chmod 600 medical-ai-key.pem
chmod 644 medical-ai-cert.pem

# Install certificate
cp medical-ai-cert.pem /etc/ssl/certs/
cp medical-ai-key.pem /etc/ssl/private/
update-ca-certificates
```

### Nginx SSL Configuration
```nginx
# Medical AI Production Nginx Configuration
server {
    listen 443 ssl http2;
    server_name api.medical-ai.example.com;

    # SSL/TLS Configuration
    ssl_certificate /etc/ssl/certs/medical-ai-cert.pem;
    ssl_certificate_key /etc/ssl/private/medical-ai-key.pem;
    
    # Strong SSL configuration
    ssl_protocols TLSv1.3 TLSv1.2;
    ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    
    # Security headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Rate limiting for medical endpoints
    limit_req_zone $binary_remote_addr zone=medical_inference:10m rate=60r/m;
    limit_req_zone $binary_remote_addr zone=clinical_support:10m rate=30r/m;
    
    # API proxy
    location / {
        proxy_pass http://medical-ai-backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Client-ID $http_x_client_id;
        proxy_set_header X-Medical-Compliance $http_x_medical_compliance;
        
        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        
        # Medical-specific rate limiting
        location ~ ^/api/v1/inference {
            limit_req zone=medical_inference burst=10 nodelay;
        }
        
        location ~ ^/api/v1/clinical {
            limit_req zone=clinical_support burst=5 nodelay;
        }
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://medical-ai-backend/health;
        access_log off;
    }
}

# HTTP redirect to HTTPS
server {
    listen 80;
    server_name api.medical-ai.example.com;
    return 301 https://$server_name$request_uri;
}
```

## Backup & Disaster Recovery

### Database Backup Strategy
```bash
#!/bin/bash
# Medical AI Database Backup Script

# Configuration
BACKUP_DIR="/backup/medical-ai"
RETENTION_DAYS=90
ENCRYPTION_KEY="/secure/keys/backup.key"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup with encryption
pg_dump -h localhost -U medical_ai_app -d medical_ai \
  --verbose --clean --no-owner --no-privileges \
  | gzip | gpg --symmetric --cipher-algo AES256 --s2k-mode 3 \
  --s2k-digest-algo SHA512 --s2k-count 65536 --compress-algo 1 \
  --passphrase-file $ENCRYPTION_KEY \
  --output $BACKUP_DIR/medical_ai_$(date +%Y%m%d_%H%M%S).sql.gz.gpg

# Audit log backup
pg_dump -h localhost -U medical_ai_app -d medical_ai \
  --table=audit_log --data-only \
  | gzip | gpg --symmetric --cipher-algo AES256 \
  --passphrase-file $ENCRYPTION_KEY \
  --output $BACKUP_DIR/audit_log_$(date +%Y%m%d_%H%M%S).sql.gz.gpg

# Clean old backups
find $BACKUP_DIR -name "*.gpg" -mtime +$RETENTION_DAYS -delete

# Verify backup integrity
gpg --decrypt $BACKUP_DIR/medical_ai_$(date +%Y%m%d)*.sql.gz.gpg | zcat | head -1

echo "Medical AI backup completed successfully"
```

### Application Backup
```yaml
# Kubernetes Backup Strategy
apiVersion: batch/v1
kind: CronJob
metadata:
  name: medical-ai-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            command:
            - /bin/bash
            - -c
            - |
              # Backup application data
              kubectl get configmaps -n medical-ai -o yaml > /backup/configmaps_$(date +%Y%m%d).yaml
              kubectl get secrets -n medical-ai -o yaml > /backup/secrets_$(date +%Y%m%d).yaml
              kubectl get pvc -n medical-ai -o yaml > /backup/pvc_$(date +%Y%m%d).yaml
              
              # Backup persistent volumes
              kubectl exec -n medical-ai deployment/postgres -- pg_dump -U postgres medical_ai > /backup/db_$(date +%Y%m%d).sql
              
              # Encrypt and upload to secure storage
              tar czf - /backup/*_$(date +%Y%m%d)* | gpg --encrypt --recipient backup@medical-ai.example.com
          restartPolicy: OnFailure
```

## Monitoring & Alerting

### Production Monitoring Stack
```yaml
# Medical AI Production Monitoring

# Prometheus Configuration
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "medical_ai_alerts.yml"

scrape_configs:
  - job_name: 'medical-ai-api'
    static_configs:
      - targets: ['medical-ai-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Medical-Specific Alerts
```yaml
# Medical AI Alert Rules
groups:
- name: medical_ai.rules
  rules:
  
  # System Health Alerts
  - alert: MedicalAIAPIDown
    expr: up{job="medical-ai-api"} == 0
    for: 1m
    labels:
      severity: critical
      compliance: hipaa
    annotations:
      summary: "Medical AI API is down"
      description: "Medical AI API has been down for more than 1 minute. Critical for clinical operations."
      
  # Performance Alerts
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2.0
    for: 2m
    labels:
      severity: warning
      compliance: performance
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is above 2 seconds. May impact clinical workflow."
      
  # Security Alerts
  - alert: UnauthorizedAccessAttempt
    expr: rate(http_requests_total{status="401"}[5m]) > 0.1
    for: 1m
    labels:
      severity: critical
      compliance: security
    annotations:
      summary: "Unauthorized access attempts detected"
      description: "High rate of unauthorized access attempts. Potential security breach."
      
  # Compliance Alerts
  - alert: PHIAccessViolation
    expr: rate(audit_log_events{action="PHI_ACCESS_VIOLATION"}[5m]) > 0
    for: 0m
    labels:
      severity: critical
      compliance: hipaa
    annotations:
      summary: "PHI access violation detected"
      description: "Protected Health Information access violation detected. Immediate investigation required."
      
  # Model Performance Alerts
  - alert: ModelAccuracyDegradation
    expr: model_accuracy_score < 0.90
    for: 5m
    labels:
      severity: warning
      compliance: clinical
    annotations:
      summary: "Model accuracy below threshold"
      description: "Model accuracy has dropped below 90%. Clinical validation may be required."
```

### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "Medical AI Production Dashboard",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "singlestat",
        "targets": [
          {
            "expr": "model_accuracy_score",
            "legendFormat": "Current Accuracy"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 0.85},
                {"color": "green", "value": 0.90}
              ]
            }
          }
        }
      },
      {
        "title": "PHI Access Events",
        "type": "table",
        "targets": [
          {
            "expr": "rate(audit_log_events{action=\"PHI_ACCESS\"}[5m])",
            "legendFormat": "{{action}}"
          }
        ]
      }
    ]
  }
}
```

## Load Testing

### Medical-Grade Load Testing
```bash
#!/bin/bash
# Medical AI Load Testing Script

# Install required tools
pip install locust

# Create load test script
cat > medical_ai_load_test.py << 'EOF'
from locust import HttpUser, task, between

class MedicalAIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Authenticate and get session"""
        response = self.client.post("/api/v1/auth/login", json={
            "username": "test_clinician",
            "password": "secure_password"
        })
        if response.status_code == 200:
            self.token = response.json()["access_token"]
        else:
            self.token = None
    
    @task(3)
    def medical_inference(self):
        """Test medical inference endpoint"""
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.post("/api/v1/inference/single", 
                json={
                    "query": "Patient with chest pain and shortness of breath",
                    "medical_domain": "cardiology",
                    "urgency_level": "high",
                    "patient_id": "test_patient_123"
                },
                headers=headers
            )
    
    @task(1)
    def clinical_decision_support(self):
        """Test clinical decision support"""
        if self.token:
            headers = {"Authorization": f"Bearer {self.token}"}
            self.client.post("/api/v1/clinical-decision-support/analyze",
                json={
                    "patient_data": {
                        "age": 65,
                        "symptoms": ["chest pain", "dyspnea"]
                    },
                    "support_type": "differential_diagnosis"
                },
                headers=headers
            )
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        self.client.get("/health")

class MedicalAIBatchUser(HttpUser):
    wait_time = between(5, 10)
    
    @task
    def batch_inference(self):
        """Test batch processing"""
        self.client.post("/api/v1/inference/batch",
            json={
                "queries": [
                    {"query": "Fever and headache", "medical_domain": "general"},
                    {"query": "Chest pain", "medical_domain": "cardiology"}
                ]
            }
        )
EOF

# Run load test with medical-specific parameters
echo "Starting Medical AI Load Test..."
locust -f medical_ai_load_test.py \
  --host=https://api.medical-ai.example.com \
  --users=100 \
  --spawn-rate=10 \
  --run-time=10m \
  --headless \
  --html=medical_ai_load_test_report.html
```

### Performance Benchmarks
```yaml
# Medical AI Performance Benchmarks
Performance Targets:
  Single Inference:
    - Response Time: < 1.5 seconds (p95)
    - Throughput: 1000 requests/minute
    - Accuracy: > 90%
    - Availability: 99.9%
    
  Batch Processing:
    - Response Time: < 30 seconds for 50 queries
    - Throughput: 100 batch requests/minute
    - Success Rate: > 99.5%
    
  Clinical Decision Support:
    - Response Time: < 2 seconds
    - Accuracy: > 85%
    - Confidence Score: > 0.8
    
  System Resources:
    - CPU Usage: < 70% average
    - Memory Usage: < 80%
    - Disk I/O: < 80% capacity
    - Network Latency: < 100ms
    
Load Test Scenarios:
  Normal Load:
    - 100 concurrent users
    - 60 requests/minute
    - Mix of inference and clinical support
    
  Peak Load:
    - 500 concurrent users
    - 300 requests/minute
    - Emergency scenarios included
    
  Stress Test:
    - 1000 concurrent users
    - 600 requests/minute
    - Failure scenarios tested
    
  Compliance Test:
    - PHI detection accuracy
    - Audit logging completeness
    - Encryption verification
    - Access control validation
```

## Go-Live Checklist

### Pre-Production Validation
- [ ] Security penetration testing completed
- [ ] HIPAA compliance audit passed
- [ ] Performance testing meets benchmarks
- [ ] Disaster recovery tested
- [ ] Staff training completed
- [ ] Incident response plan tested
- [ ] Regulatory approval obtained
- [ ] Business continuity plan validated

### Launch Day Procedures
- [ ] Monitor system health continuously
- [ ] Have rollback procedures ready
- [ ] Clinical support team on standby
- [ ] Regulatory team available
- [ ] Communication plan activated
- [ ] Performance metrics tracking active

### Post-Launch Monitoring (First 48 Hours)
- [ ] Hourly health checks
- [ ] Real-time performance monitoring
- [ ] User feedback collection
- [ ] Error rate monitoring
- [ ] Clinical outcome tracking
- [ ] Compliance audit logging

## Emergency Procedures

### System Failure Response
```bash
#!/bin/bash
# Emergency Response Script

echo "Medical AI Emergency Response Protocol Activated"

# 1. Immediate assessment
curl -f https://api.medical-ai.example.com/health || echo "CRITICAL: API is down"

# 2. Check database connectivity
kubectl exec -n medical-ai deployment/postgres -- pg_isready || echo "WARNING: Database issues"

# 3. Scale down to stabilize
kubectl scale deployment medical-ai-api --replicas=2

# 4. Enable maintenance mode
kubectl patch configmap medical-ai-config -p '{"data":{"maintenance_mode":"true"}}'

# 5. Notify emergency contacts
echo "Medical AI system emergency - maintenance mode enabled" | mail -s "ALERT" emergency@medical-ai.example.com

# 6. Prepare rollback
echo "Preparing rollback procedures..."
```

### Contact Information
```yaml
# Emergency Contact List
Critical Issues:
  - On-call Engineer: +1-XXX-XXX-XXXX
  - Medical Director: +1-XXX-XXX-XXXX
  - Technical Director: +1-XXX-XXX-XXXX

Compliance Issues:
  - HIPAA Officer: +1-XXX-XXX-XXXX
  - Regulatory Affairs: +1-XXX-XXX-XXXX
  - Legal Counsel: +1-XXX-XXX-XXXX

Clinical Issues:
  - Clinical Support: +1-XXX-XXX-XXXX
  - Medical Quality: +1-XXX-XXX-XXXX
  - IRB Chair: +1-XXX-XXX-XXXX
```

---

**⚠️ Deployment Disclaimer**: This deployment guide is provided for medical device compliance. All deployments must be validated for the specific medical use case and regulatory environment. Never deploy without proper clinical validation and regulatory approval.
