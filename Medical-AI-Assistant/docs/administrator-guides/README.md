# Administrator Guide - Medical AI Assistant

## System Configuration and Maintenance

This guide provides comprehensive instructions for system administrators responsible for configuring, deploying, and maintaining the Medical AI Assistant system.

## 🔧 System Architecture Overview

### Core Components

#### Frontend Components
- **React Patient Interface**: Patient-facing chat interface
- **Nurse Dashboard**: Healthcare professional dashboard
- **Administrator Portal**: System administration interface
- **Monitoring Dashboard**: Real-time system monitoring

#### Backend Services
- **FastAPI Gateway**: API gateway and request routing
- **AI Orchestrator**: LangChain-based AI conversation orchestrator
- **Model Service**: LLM inference service with LoRA adapters
- **Vector Database**: RAG implementation (Chroma/Qdrant/Milvus)
- **Audit Service**: Logging and audit trail service
- **Notification Service**: Email and push notification service

#### Data Layer
- **PostgreSQL**: Primary database for application data
- **Redis**: Caching and session management
- **Vector Store**: Vector embeddings storage
- **File Storage**: Document and model artifact storage

#### Infrastructure
- **Docker Containers**: Application containerization
- **Kubernetes**: Container orchestration (production)
- **Load Balancer**: Traffic distribution and SSL termination
- **Monitoring Stack**: Prometheus, Grafana, and alerting

## 🚀 Initial Deployment

### Prerequisites

#### Hardware Requirements
**Minimum Production Setup:**
- **CPU**: 8 cores, 2.4GHz+
- **RAM**: 32GB minimum
- **Storage**: 500GB SSD for application data
- **Network**: 1Gbps network connection
- **GPU**: 1x NVIDIA A100 or equivalent for model inference

**Recommended Production Setup:**
- **CPU**: 16+ cores, 3.0GHz+
- **RAM**: 64GB minimum
- **Storage**: 1TB+ NVMe SSD
- **Network**: 10Gbps network connection
- **GPU**: 2x NVIDIA A100 for high availability

#### Software Requirements
- **Operating System**: Ubuntu 20.04 LTS or CentOS 8+
- **Container Runtime**: Docker 20.10+ and Docker Compose 2.0+
- **Kubernetes**: v1.20+ (for production deployments)
- **Database**: PostgreSQL 13+
- **Python**: 3.8+ for development and model training
- **Node.js**: 16+ for frontend builds

#### Network Requirements
- **Firewall Configuration**: Ports 80, 443, 8000, 5432, 6379
- **SSL Certificates**: Valid SSL certificates for HTTPS
- **DNS Configuration**: Proper DNS records for all services
- **Load Balancing**: Configured load balancer with health checks

### Installation Steps

#### Step 1: Environment Preparation
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install additional dependencies
sudo apt install -y git curl wget jq htop

# Create application directories
sudo mkdir -p /opt/medical-ai
sudo chown $USER:$USER /opt/medical-ai
```

#### Step 2: Clone and Configure
```bash
# Clone repository
cd /opt/medical-ai
git clone https://github.com/your-org/Medical-AI-Assistant.git
cd Medical-AI-Assistant

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

#### Step 3: Database Setup
```bash
# Create PostgreSQL database
sudo -u postgres createdb medical_ai_db
sudo -u postgres createuser medical_ai_user -P

# Run database migrations
python -m alembic upgrade head

# Seed initial data
python scripts/seed_database.py
```

#### Step 4: Model Deployment
```bash
# Download and configure base model
python scripts/download_base_model.py

# Deploy LoRA adapters
python scripts/deploy_adapters.py

# Test model inference
python scripts/test_inference.py
```

#### Step 5: Service Startup
```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### Production Deployment (Kubernetes)

#### Helm Chart Installation
```bash
# Add Helm repository
helm repo add medical-ai https://charts.medical-ai.example
helm repo update

# Install with custom values
helm install medical-ai medical-ai/medical-ai-assistant \
  --namespace medical-ai \
  --create-namespace \
  --values production-values.yaml
```

#### Production Configuration
```yaml
# production-values.yaml
replicaCount: 3

image:
  repository: medical-ai/assistant
  tag: "1.0.0"

service:
  type: LoadBalancer
  ports:
    http: 80
    https: 443

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: medical-ai.your-domain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: medical-ai-tls
      hosts:
        - medical-ai.your-domain.com
```

## ⚙️ Configuration Management

### Environment Variables

#### Core Application Settings
```bash
# Application Configuration
APP_NAME=Medical AI Assistant
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_TIMEOUT=300

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost/medical_ai_db
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_SESSION_TTL=86400
REDIS_CACHE_TTL=3600
```

#### AI Model Configuration
```bash
# Model Settings
BASE_MODEL_PATH=/models/medical-base-model
LORA_ADAPTER_PATH=/models/lora-adapters
MODEL_CACHE_SIZE=4
MAX_SEQUENCE_LENGTH=2048
TEMPERATURE=0.7
TOP_P=0.9

# Vector Database
VECTOR_DB_TYPE=chroma
VECTOR_DB_PATH=/data/vector_store
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

#### Security Configuration
```bash
# Security Settings
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
SESSION_SECRET=your-session-secret

# CORS Configuration
CORS_ORIGINS=["https://medical-ai.your-domain.com"]
CORS_METHODS=["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS=["*"]

# SSL Configuration
SSL_CERT_PATH=/etc/ssl/certs/medical-ai.crt
SSL_KEY_PATH=/etc/ssl/private/medical-ai.key
FORCE_HTTPS=true
```

### Database Configuration

#### PostgreSQL Settings
```sql
-- PostgreSQL configuration for optimal performance
-- Add to postgresql.conf

shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1                  # For SSD
effective_io_concurrency = 200
work_mem = 256MB
min_wal_size = 2GB
max_wal_size = 8GB
max_worker_processes = 8
max_parallel_workers_per_gather = 4
```

#### Database Security
```sql
-- Create application user with minimal privileges
CREATE USER medical_ai_app WITH PASSWORD 'secure_password';

-- Grant necessary privileges
GRANT CONNECT ON DATABASE medical_ai_db TO medical_ai_app;
GRANT USAGE ON SCHEMA public TO medical_ai_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO medical_ai_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO medical_ai_app;

-- Revoke dangerous privileges
REVOKE ALL ON DATABASE medical_ai_db FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM PUBLIC;
```

### API Configuration

#### FastAPI Settings
```python
# app/config.py
from pydantic import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Medical AI Assistant"
    debug: bool = False
    log_level: str = "INFO"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 300
    
    # Database settings
    database_url: str
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_timeout: int = 30
    
    # Redis settings
    redis_url: str
    redis_session_ttl: int = 86400
    redis_cache_ttl: int = 3600
    
    # Security settings
    secret_key: str
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    session_secret: str
    
    # CORS settings
    cors_origins: List[str]
    cors_methods: List[str] = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: List[str] = ["*"]
    
    # AI model settings
    base_model_path: str
    lora_adapter_path: str
    model_cache_size: int = 4
    max_sequence_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Vector database settings
    vector_db_type: str = "chroma"
    vector_db_path: str
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    class Config:
        env_file = ".env"

settings = Settings()
```

## 🔍 Monitoring and Logging

### Health Checks

#### Application Health Check
```python
# app/health.py
from fastapi import FastAPI
from app.models import HealthStatus, DatabaseStatus, AIServiceStatus

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Comprehensive health check endpoint"""
    
    # Database connectivity
    try:
        await database.status()
        db_status = DatabaseStatus(
            status="healthy",
            response_time_ms=await database.ping()
        )
    except Exception as e:
        db_status = DatabaseStatus(
            status="unhealthy",
            error=str(e)
        )
    
    # AI service connectivity
    try:
        model_health = await ai_service.health_check()
        ai_status = AIServiceStatus(
            status=model_health.status,
            model_loaded=model_health.model_loaded,
            response_time_ms=model_health.response_time
        )
    except Exception as e:
        ai_status = AIServiceStatus(
            status="unhealthy",
            error=str(e)
        )
    
    return HealthStatus(
        status="healthy" if all([
            db_status.status == "healthy",
            ai_status.status == "healthy"
        ]) else "unhealthy",
        timestamp=datetime.utcnow(),
        database=db_status,
        ai_service=ai_status
    )
```

#### Kubernetes Liveness Probe
```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### Metrics Collection

#### Prometheus Metrics
```python
# app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

# AI-specific metrics
AI_CONVERSATIONS = Counter(
    'ai_conversations_total',
    'Total AI conversations',
    ['status']
)

AI_RESPONSE_TIME = Histogram(
    'ai_response_time_seconds',
    'AI response generation time',
    ['model_type']
)

ACTIVE_SESSIONS = Gauge(
    'active_sessions_total',
    'Number of active user sessions'
)

# Database metrics
DB_CONNECTIONS = Gauge(
    'database_connections_active',
    'Number of active database connections'
)

DB_QUERY_TIME = Histogram(
    'database_query_duration_seconds',
    'Database query execution time'
)
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Medical AI Assistant Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "AI Conversation Status",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ai_conversations_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      }
    ]
  }
}
```

### Log Management

#### Structured Logging
```python
# app/logging.py
import structlog
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.processors.JSONRenderer()
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "json"
        },
        "file": {
            "level": "INFO", 
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/medical-ai/app.log",
            "maxBytes": 10485760,
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
})

logger = structlog.get_logger()
```

#### Audit Logging
```python
# app/audit.py
import uuid
from datetime import datetime

class AuditLogger:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
    
    def log_patient_interaction(self, action: str, patient_data: dict):
        logger.info(
            "Patient Interaction",
            session_id=self.session_id,
            action=action,
            patient_id=patient_data.get("id"),
            timestamp=datetime.utcnow().isoformat(),
            user_type="patient"
        )
    
    def log_clinical_decision(self, action: str, recommendation: str, user_id: str):
        logger.info(
            "Clinical Decision",
            session_id=self.session_id,
            action=action,
            recommendation=recommendation,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            user_type="healthcare_professional"
        )
    
    def log_system_event(self, event_type: str, details: dict):
        logger.info(
            "System Event",
            session_id=self.session_id,
            event_type=event_type,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
```

## 🔐 Security Management

### Authentication and Authorization

#### JWT Configuration
```python
# app/auth.py
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        return None
```

#### Role-Based Access Control
```python
# app/rbac.py
from enum import Enum
from typing import List
from functools import wraps

class UserRole(Enum):
    PATIENT = "patient"
    NURSE = "nurse"
    PHYSICIAN = "physician"
    ADMIN = "admin"
    SYSTEM = "system"

class Permission(Enum):
    VIEW_PATIENT_DATA = "view_patient_data"
    MODIFY_PATIENT_DATA = "modify_patient_data"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    MANAGE_USERS = "manage_users"
    SYSTEM_CONFIG = "system_config"

ROLE_PERMISSIONS = {
    UserRole.PATIENT: [Permission.VIEW_PATIENT_DATA],
    UserRole.NURSE: [Permission.VIEW_PATIENT_DATA, Permission.MODIFY_PATIENT_DATA],
    UserRole.PHYSICIAN: [Permission.VIEW_PATIENT_DATA, Permission.MODIFY_PATIENT_DATA, Permission.VIEW_AUDIT_LOGS],
    UserRole.ADMIN: [Permission.VIEW_PATIENT_DATA, Permission.MODIFY_PATIENT_DATA, Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_USERS],
    UserRole.SYSTEM: [Permission.VIEW_PATIENT_DATA, Permission.MODIFY_PATIENT_DATA, Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_USERS, Permission.SYSTEM_CONFIG]
}

def require_permission(permission: Permission):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('current_user')
            if not user or permission not in ROLE_PERMISSIONS.get(user.role, []):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### SSL/TLS Configuration

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/medical-ai
server {
    listen 80;
    server_name medical-ai.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name medical-ai.your-domain.com;

    ssl_certificate /etc/ssl/certs/medical-ai.crt;
    ssl_certificate_key /etc/ssl/private/medical-ai.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
```

#### Certificate Management
```bash
#!/bin/bash
# cert-renewal.sh - Automated certificate renewal

# Get current certificates
certbot certificates

# Renew certificates
certbot renew --quiet

# Reload nginx if certificates were renewed
systemctl reload nginx

# Log renewal attempt
logger "Certificate renewal script executed at $(date)"
```

## 🔄 Backup and Recovery

### Database Backup

#### Automated Backup Script
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups/database"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="medical_ai_db_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump medical_ai_db > $BACKUP_DIR/$BACKUP_FILE

# Compress backup
gzip $BACKUP_DIR/$BACKUP_FILE

# Upload to secure storage
aws s3 cp $BACKUP_DIR/${BACKUP_FILE}.gz s3://medical-ai-backups/database/

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# Log backup completion
logger "Database backup completed: $BACKUP_FILE.gz"

# Send notification
curl -X POST https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK \
  -H 'Content-type: application/json' \
  --data '{"text":"Database backup completed successfully"}'
```

#### Recovery Procedures
```bash
#!/bin/bash
# restore-database.sh

BACKUP_FILE=$1
DB_NAME="medical_ai_db"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
systemctl stop medical-ai

# Drop and recreate database
sudo -u postgres dropdb $DB_NAME
sudo -u postgres createdb $DB_NAME

# Restore from backup
gunzip -c $BACKUP_FILE | sudo -u postgres psql $DB_NAME

# Verify restoration
sudo -u postgres psql $DB_NAME -c "SELECT COUNT(*) FROM patients;"

# Start application
systemctl start medical-ai

logger "Database restoration completed from $BACKUP_FILE"
```

### Model Backup

#### Model Artifact Management
```python
# scripts/backup_models.py
import shutil
import boto3
from pathlib import Path
from datetime import datetime

def backup_model_artifacts():
    """Backup AI model artifacts to secure storage"""
    
    model_paths = [
        "/models/base-model",
        "/models/lora-adapters", 
        "/models/vector-store",
        "/data/embeddings"
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"/backups/models/{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    s3_client = boto3.client('s3')
    
    for model_path in model_paths:
        source = Path(model_path)
        if source.exists():
            # Create local backup
            dest = backup_dir / source.name
            shutil.copytree(source, dest)
            
            # Upload to S3
            for file_path in dest.rglob('*'):
                if file_path.is_file():
                    s3_key = f"models/{timestamp}/{source.name}/{file_path.relative_to(dest)}"
                    s3_client.upload_file(str(file_path), "medical-ai-model-backups", s3_key)
    
    print(f"Model backup completed: {backup_dir}")

if __name__ == "__main__":
    backup_model_artifacts()
```

## 🚨 Disaster Recovery

### Incident Response Plan

#### Emergency Procedures
```bash
#!/bin/bash
# emergency-response.sh

# Activate emergency mode
echo "Activating emergency response procedures..."

# 1. Stop accepting new requests
echo "Stopping new request acceptance..."
systemctl stop nginx

# 2. Switch to backup systems
echo "Switching to backup systems..."
kubectl scale deployment medical-ai-assistant --replicas=0
kubectl scale deployment medical-ai-backup --replicas=3

# 3. Activate read-only mode
echo "Activating read-only mode..."
python scripts/activate_readonly_mode.py

# 4. Notify stakeholders
python scripts/notify_stakeholders.py --incident-type emergency

# 5. Begin incident logging
python scripts/start_incident_logging.py

echo "Emergency response procedures activated"
```

#### Recovery Procedures
```bash
#!/bin/bash
# recovery-procedures.sh

INCIDENT_ID=$1

echo "Starting recovery procedures for incident: $INCIDENT_ID"

# 1. Assess damage
python scripts/assess_damage.py --incident-id $INCIDENT_ID

# 2. Restore from backup
python scripts/restore_from_backup.py --incident-id $INCIDENT_ID

# 3. Validate system integrity
python scripts/validate_system.py

# 4. Gradual service restoration
python scripts/gradual_restart.py

# 5. Monitor for issues
python scripts/enhanced_monitoring.py --duration 1h

# 6. Notify recovery completion
python scripts/notify_recovery.py --incident-id $INCIDENT_ID

echo "Recovery procedures completed"
```

## 📊 Performance Optimization

### Database Optimization

#### Query Performance
```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_sessions_timestamp ON sessions(created_at);
CREATE INDEX CONCURRENTLY idx_assessments_status ON assessments(status);
CREATE INDEX CONCURRENTLY idx_patients_age_group ON patients(age_group);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM assessments WHERE status = 'pending';

-- Monitor slow queries
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;
```

#### Connection Pooling
```python
# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.debug
)
```

### Caching Strategy

#### Redis Configuration
```python
# app/cache.py
import redis
import json
from typing import Optional

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.redis_url)
    
    def get_session_data(self, session_id: str) -> Optional[dict]:
        data = self.redis_client.get(f"session:{session_id}")
        return json.loads(data) if data else None
    
    def set_session_data(self, session_id: str, data: dict, ttl: int = 86400):
        self.redis_client.setex(
            f"session:{session_id}", 
            ttl, 
            json.dumps(data)
        )
    
    def invalidate_session(self, session_id: str):
        self.redis_client.delete(f"session:{session_id}")
    
    def cache_model_response(self, prompt_hash: str, response: str, ttl: int = 3600):
        self.redis_client.setex(f"model_response:{prompt_hash}", ttl, response)
    
    def get_cached_response(self, prompt_hash: str) -> Optional[str]:
        return self.redis_client.get(f"model_response:{prompt_hash}")
```

## 📞 Support and Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
- Monitor system health and performance metrics
- Review error logs for critical issues
- Check backup completion status
- Monitor disk space and resource usage
- Review security alerts and notifications

#### Weekly Tasks
- Perform security patch updates
- Review and analyze performance trends
- Conduct backup restoration tests
- Review audit logs for security issues
- Update system documentation

#### Monthly Tasks
- Perform comprehensive system health checks
- Review and update security configurations
- Conduct disaster recovery drills
- Review and update monitoring alerts
- Perform capacity planning assessments

### Maintenance Scripts
```bash
#!/bin/bash
# daily-maintenance.sh

echo "Starting daily maintenance tasks..."

# Clean up old log files
find /var/log/medical-ai -name "*.log" -mtime +7 -delete

# Optimize database
sudo -u postgres psql medical_ai_db -c "VACUUM ANALYZE;"

# Clear temporary files
find /tmp -name "medical-ai-*" -mtime +1 -delete

# Update system packages
apt update && apt list --upgradable

# Check SSL certificate expiration
echo | openssl s_client -servername medical-ai.your-domain.com -connect medical-ai.your-domain.com:443 2>/dev/null | openssl x509 -noout -dates

# Generate daily report
python scripts/generate_maintenance_report.py

echo "Daily maintenance completed"
```

### Support Escalation Procedures

#### Level 1: System Administrator
- **Issues**: Basic system problems, user access, performance issues
- **Response Time**: 1 hour during business hours
- **Escalation**: To Level 2 if unresolved within 2 hours

#### Level 2: Senior Administrator/Developer
- **Issues**: Complex technical issues, security concerns, system integration problems
- **Response Time**: 30 minutes during business hours
- **Escalation**: To Level 3 for critical patient safety issues

#### Level 3: Emergency Response Team
- **Issues**: System outages affecting patient care, security breaches, data loss
- **Response Time**: 15 minutes 24/7
- **Actions**: Immediate incident response and stakeholder notification

---

**Remember: System administration of healthcare AI systems requires careful attention to security, compliance, and patient safety. Always prioritize patient care and data protection in all administrative decisions.**

*For technical support or questions about system administration, contact your organization's IT security team and the Medical AI Assistant implementation team.*

**Version**: 1.0 | **Last Updated**: November 2025 | **Next Review**: February 2026
