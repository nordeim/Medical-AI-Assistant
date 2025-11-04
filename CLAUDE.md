# Medical AI Assistant - AI Coding Agent Briefing Document

**Document Version:** 1.0  
**Created:** November 4, 2025  
**Target Audience:** AI Coding Agents (Claude Code, Codex, etc.)  
**Project Version:** 1.0.0  

---

## 🎯 Quick Start Guide for AI Agents

This document provides comprehensive technical context for AI coding agents working on the Medical AI Assistant - a production-ready, HIPAA-compliant healthcare AI platform. Focus on actionable information, file paths, and operational procedures.

### ⚡ Essential Context
- **Project Type**: Enterprise healthcare AI platform with microservices architecture
- **Compliance Level**: HIPAA, FDA, ISO 27001 compliant with PHI protection
- **Development Status**: Production-ready (Phases 1-10 complete)
- **Scale**: 25+ directories, 500+ files, multi-cloud deployment ready

---

## 🏗️ System Architecture Overview

### Core Technology Stack
```yaml
Backend:
  Framework: FastAPI 0.109.0
  Language: Python 3.9+
  Database: PostgreSQL + SQLAlchemy ORM
  Cache: Redis (sessions & performance)
  AI/ML: PyTorch, Transformers, LangChain v1.0
  Vector DB: Chroma (medical knowledge RAG)

Frontend:
  Framework: React 18.2.0
  Language: TypeScript 5.0+
  UI Library: Radix UI + Tailwind CSS
  State: TanStack Query + React Context
  Build Tool: Vite
  Real-time: WebSocket integration

Infrastructure:
  Containers: Docker multi-stage builds
  Orchestration: Kubernetes + Helm charts
  Cloud: AWS/Azure/GCP with Terraform
  Monitoring: Prometheus, Grafana, ELK stack
  CI/CD: Automated testing, security scanning, blue-green deployments
```

### Application Architecture Pattern
```
┌─ Client Layer (React/TypeScript UIs)
├─ API Gateway (Kong/Nginx + Auth + Rate Limiting)
├─ Application Layer (FastAPI + LangChain Agent)
├─ AI/ML Layer (PEFT/LoRA Models + Vector DB + RAG)
├─ Data Layer (PostgreSQL + Redis + Vector Store + Audit DB)
└─ Integration Layer (EHR/FHIR + External APIs + Webhooks)
```

---

## 📂 Critical File Structure & Key Components

### Core Application Files (HIGH PRIORITY)
```
Medical-AI-Assistant/
├── backend/                           # FastAPI Backend
│   ├── app/
│   │   ├── main.py                   # 🎯 APPLICATION ENTRY POINT
│   │   ├── config.py                 # 🎯 CONFIGURATION MANAGEMENT
│   │   ├── database.py               # 🎯 DATABASE CONNECTIONS
│   │   └── dependencies.py           # 🎯 FASTAPI DEPENDENCIES
│   ├── agent/
│   │   ├── orchestrator.py           # 🎯 AI AGENT COORDINATOR
│   │   ├── par_generator.py          # 🎯 PATIENT ASSESSMENT GENERATOR
│   │   └── tools/                    # 🎯 CLINICAL TOOLS
│   ├── api/routes/                   # 🎯 API ENDPOINT DEFINITIONS
│   ├── auth/                         # 🎯 JWT + RBAC + PERMISSIONS
│   ├── models/                       # 🎯 PYDANTIC SCHEMAS
│   ├── rag/                          # 🎯 RAG SYSTEM (embeddings, retriever, vector_store)
│   ├── services/                     # 🎯 BUSINESS LOGIC SERVICES
│   ├── websocket/                    # 🎯 REAL-TIME COMMUNICATION
│   └── tests/                        # 🎯 COMPREHENSIVE TEST SUITE
│
├── frontend/                          # React Frontend
│   ├── src/
│   │   ├── App.tsx                   # 🎯 MAIN REACT APPLICATION
│   │   ├── components/
│   │   │   ├── chat/                 # 🎯 PATIENT CHAT INTERFACE
│   │   │   ├── dashboard/            # 🎯 NURSE DASHBOARD
│   │   │   ├── forms/                # 🎯 MEDICAL FORMS + VALIDATION
│   │   │   └── ui/                   # 🎯 BASE UI COMPONENTS
│   │   ├── pages/                    # 🎯 ROUTE-BASED PAGE COMPONENTS
│   │   ├── hooks/                    # 🎯 CUSTOM REACT HOOKS
│   │   ├── services/                 # 🎯 API SERVICE LAYER
│   │   ├── types/                    # 🎯 TYPESCRIPT DEFINITIONS
│   │   └── contexts/                 # 🎯 REACT CONTEXT PROVIDERS
│
├── database/                          # Database Management
│   ├── migrations/                   # 🎯 SCHEMA MIGRATIONS
│   ├── models/                       # 🎯 SQLALCHEMY MODELS
│   └── seeds/                        # 🎯 INITIAL DATA SEEDING
│
├── ml/                               # ML Components
│   ├── models/                       # 🎯 TRAINED MODEL DEFINITIONS
│   ├── training/                     # 🎯 TRAINING PIPELINES
│   └── inference/                    # 🎯 MODEL SERVING CONFIGS
│
└── deployment/                       # Deployment Configs
    ├── docker/                       # 🎯 DOCKER CONTAINERIZATION
    ├── kubernetes/                   # 🎯 K8S DEPLOYMENT MANIFESTS
    └── terraform/                    # 🎯 INFRASTRUCTURE AS CODE
```

### Configuration Files (CRITICAL)
```
📁 Root Configuration Files:
├── .env.example                      # 🎯 ENVIRONMENT TEMPLATE
├── docker-compose.yml               # 🎯 LOCAL DEVELOPMENT SETUP
├── docker-compose.prod.yml          # 🎯 PRODUCTION DOCKER SETUP
├── requirements.txt                 # 🎯 PYTHON DEPENDENCIES
├── package.json                     # 🎯 NODE.JS DEPENDENCIES
└── README.md                        # 🎯 COMPREHENSIVE PROJECT DOCS

📁 Production Configuration:
├── production/infrastructure/kubernetes/  # 🎯 PRODUCTION K8S MANIFESTS
├── production/security/compliance/        # 🎯 HIPAA/FDA COMPLIANCE CONFIGS
└── deployment/docker/                     # 🎯 DOCKER BUILD CONFIGS
```

---

## 🚀 Environment Setup & Development Commands

### Prerequisites Validation
```bash
# Version Requirements Check
python --version    # Required: 3.9+
node --version      # Required: 18+
docker --version    # Required: 20.10+
kubectl version     # Required: 1.21+ (for K8s deployment)
```

### Quick Start (Docker Compose - RECOMMENDED)
```bash
# 1. Clone and Navigate
git clone https://github.com/nordeim/Medical-AI-Assistant.git
cd Medical-AI-Assistant

# 2. Environment Configuration
cp .env.example .env
# Edit .env with your specific values (see Environment Variables section)

# 3. Start All Services
docker-compose up -d

# 4. Verify Services
docker-compose ps
curl http://localhost:8000/health    # Backend health check
curl http://localhost:3000           # Frontend accessibility
```

### Manual Development Setup
```bash
# Backend Setup
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Database Setup
export DATABASE_URL="postgresql://user:password@localhost:5432/medical_ai"
alembic upgrade head
python -m app.database seed_data

# Start Backend Development Server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend Setup (in new terminal)
cd frontend
npm install
npm run dev    # Starts on http://localhost:3000
```

### Essential Development Commands
```bash
# Testing Commands
cd backend && python -m pytest tests/ -v           # Backend unit tests
cd backend && python -m pytest tests/integration/  # Integration tests
cd frontend && npm test                             # Frontend tests
cd frontend && npm run test:e2e                     # E2E tests

# Database Operations
alembic revision --autogenerate -m "description"   # Create migration
alembic upgrade head                                # Apply migrations
alembic downgrade -1                                # Rollback migration

# Code Quality
cd backend && black . && isort . && flake8        # Python formatting
cd frontend && npm run lint && npm run format     # TypeScript/React linting

# Build & Deployment
docker build -t medical-ai/backend backend/       # Build backend image
docker build -t medical-ai/frontend frontend/     # Build frontend image
kubectl apply -f deployment/kubernetes/           # Deploy to K8s
```

---

## 🔧 Environment Variables & Configuration

### Critical Environment Variables
```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/medical_ai
REDIS_URL=redis://localhost:6379/0

# Security Configuration (CRITICAL for HIPAA)
SECRET_KEY=your-super-secret-key-256-bit
JWT_SECRET_KEY=your-jwt-secret-key
ENCRYPTION_KEY=your-32-byte-encryption-key-for-phi

# AI Services Configuration
OPENAI_API_KEY=your-openai-api-key
LANGCHAIN_API_KEY=your-langchain-api-key
VECTOR_DB_PATH=./data/chroma_db

# Application Configuration
ENVIRONMENT=development  # development|staging|production
DEBUG=true              # Set to false in production
LOG_LEVEL=INFO         # DEBUG|INFO|WARNING|ERROR

# Healthcare Integration
FHIR_BASE_URL=https://your-fhir-server.com/fhir
EHR_INTEGRATION_KEY=your-ehr-integration-key
WEBHOOK_SECRET=your-webhook-secret-key

# Performance & Scaling
MAX_CONCURRENT_SESSIONS=1000
AI_RESPONSE_TIMEOUT=30
DATABASE_POOL_SIZE=20
REDIS_MAX_CONNECTIONS=100
```

### Configuration File Locations
```bash
# Backend Configuration
backend/app/config.py              # Main configuration management
backend/app/settings/              # Environment-specific settings

# Frontend Configuration  
frontend/src/config/               # Frontend configuration
frontend/.env.local                # Local development overrides

# Deployment Configuration
deployment/docker/.env.prod        # Production environment
deployment/kubernetes/secrets/     # K8s secrets and configs
```

---

## 🔌 API Endpoints & Integration Points

### Core API Endpoints
```bash
# Health & Status
GET /health                        # System health check
GET /api/v1/status                # Detailed system status
GET /metrics                      # Prometheus metrics

# Authentication & Users
POST /api/v1/auth/login           # User authentication
POST /api/v1/auth/logout          # User logout
GET /api/v1/auth/me               # Current user info
POST /api/v1/users/               # Create user (admin only)

# Patient Sessions & Chat
POST /api/v1/sessions/            # Create patient session
GET /api/v1/sessions/{id}         # Get session details
WebSocket /ws/chat/{session_id}   # Real-time chat communication
POST /api/v1/sessions/{id}/messages  # Send message to AI

# Clinical Assessment
GET /api/v1/assessments/          # List patient assessments (nurses)
GET /api/v1/assessments/{id}      # Get specific assessment
POST /api/v1/assessments/{id}/review  # Nurse review/approve assessment

# Healthcare Integration
GET /api/v1/fhir/patients/        # FHIR patient endpoint
POST /api/v1/ehr/sync             # EHR synchronization
POST /api/v1/webhooks/intake      # External system webhooks
```

### WebSocket Events
```javascript
// Patient Chat WebSocket Events
{
  "message_sent": {                 // Patient sends message
    "session_id": "uuid",
    "message": "text",
    "timestamp": "ISO8601"
  },
  "ai_response": {                  // AI agent response
    "session_id": "uuid", 
    "response": "text",
    "streaming": true,              // Token-by-token delivery
    "red_flag": false              // Emergency flag
  },
  "session_update": {               // Session status changes
    "session_id": "uuid",
    "status": "active|completed|escalated",
    "nurse_assigned": "uuid"
  }
}
```

---

## 🧠 AI Agent System Architecture

### LangChain Agent Configuration
```python
# Core Agent Setup (backend/agent/orchestrator.py)
from langchain.agents import create_react_agent
from langchain.tools import Tool

class MedicalAIOrchestrator:
    def __init__(self):
        self.llm = self._setup_llm()           # PEFT/LoRA medical models
        self.vector_store = self._setup_rag()  # Chroma vector database
        self.tools = self._setup_tools()       # Clinical reasoning tools
        self.agent = self._create_agent()      # ReAct agent with medical tools
    
    def _setup_tools(self):
        return [
            Tool(name="medical_knowledge", func=self.query_medical_kb),
            Tool(name="red_flag_detector", func=self.detect_red_flags),
            Tool(name="assessment_generator", func=self.generate_par),
            Tool(name="drug_interaction", func=self.check_drug_interactions)
        ]
```

### RAG System Components
```python
# RAG Implementation (backend/rag/)
📁 rag/
├── embeddings.py          # Medical document embeddings (sentence-transformers)
├── retriever.py           # Document retrieval with medical context
└── vector_store.py        # Chroma DB integration with medical guidelines

# Key RAG Functions
def embed_medical_documents(docs):           # Process medical guidelines
def retrieve_relevant_context(query):       # Context-aware retrieval
def update_vector_store(new_documents):     # Dynamic knowledge updates
```

### AI Processing Pipeline
```mermaid
flowchart LR
    A[Patient Input] --> B[Input Validation]
    B --> C[Red Flag Detection]
    C --> D{Emergency?}
    D -->|Yes| E[Emergency Escalation]
    D -->|No| F[RAG Retrieval]
    F --> G[Agent Reasoning]
    G --> H[Safety Filtering]
    H --> I[Response Generation]
    I --> J[Streaming Delivery]
    E --> K[Nurse Notification]
```

---

## 🛠️ Development Workflows & Best Practices

### Code Modification Guidelines
```bash
# High-Impact Modification Areas
1. AI Agent Logic:          backend/agent/orchestrator.py
2. API Endpoints:           backend/api/routes/
3. Database Models:         backend/models/ + database/models/
4. Frontend Components:     frontend/src/components/
5. RAG System:             backend/rag/
6. Authentication:         backend/auth/
7. Configuration:          backend/app/config.py
```

### Testing Strategy
```bash
# Testing Hierarchy (CRITICAL for Healthcare)
1. Unit Tests:             pytest backend/tests/unit/
2. Integration Tests:      pytest backend/tests/integration/
3. API Tests:             pytest backend/tests/api/
4. Security Tests:        pytest backend/tests/security/
5. Compliance Tests:      pytest backend/tests/compliance/
6. Frontend Tests:        npm test (Jest + React Testing Library)
7. E2E Tests:             npm run test:e2e (Cypress)

# Pre-Commit Requirements
black backend/ && isort backend/        # Code formatting
flake8 backend/                        # Linting
pytest backend/tests/ --cov=80%        # Minimum 80% coverage
npm run lint && npm run type-check     # Frontend quality
```

### Debugging Procedures
```bash
# Backend Debugging
export DEBUG=true
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug

# Database Debugging
export DATABASE_URL="postgresql://user:password@localhost:5432/medical_ai?echo=true"
psql $DATABASE_URL                     # Direct database access

# AI Agent Debugging
export LANGCHAIN_VERBOSE=true
export LANGCHAIN_DEBUG=true
python -m app.agent.debug_orchestrator # Custom debug script

# Frontend Debugging
npm run dev                           # Development with hot reload
npm run debug                         # Debug build with source maps
```

### Performance Monitoring
```bash
# Backend Performance
GET /metrics                          # Prometheus metrics
GET /api/v1/health/detailed          # Detailed health check
docker stats                         # Container resource usage

# Database Performance
SELECT * FROM pg_stat_activity;      # Active connections
SELECT * FROM pg_stat_statements;    # Query performance

# AI Model Performance
python -m app.ml.benchmark           # Model inference benchmarks
python -m app.rag.performance_test   # RAG retrieval performance
```

---

## 🚀 Deployment & Infrastructure

### Docker Deployment Commands
```bash
# Local Development
docker-compose up -d                  # Start all services
docker-compose logs -f backend       # View backend logs
docker-compose exec backend bash     # Access backend container
docker-compose down                   # Stop all services

# Production Build
docker build -t medical-ai/backend:latest backend/
docker build -t medical-ai/frontend:latest frontend/
docker push medical-ai/backend:latest      # Push to registry
docker push medical-ai/frontend:latest     # Push to registry
```

### Kubernetes Deployment
```bash
# Kubernetes Operations
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/secrets/
kubectl apply -f deployment/kubernetes/services/
kubectl apply -f deployment/kubernetes/deployments/

# Monitoring & Debugging
kubectl get pods -n healthcare               # Check pod status
kubectl logs -f deployment/medical-ai-backend -n healthcare
kubectl exec -it pod-name -- bash          # Access pod
kubectl port-forward service/medical-ai 8000:8000  # Local access
```

### Infrastructure as Code (Terraform)
```bash
# Multi-Cloud Deployment
cd deployment/terraform/aws/
terraform init && terraform plan && terraform apply

cd deployment/terraform/azure/
terraform init && terraform plan && terraform apply

# Infrastructure Monitoring
terraform state list                  # List managed resources
terraform output                      # Show outputs (endpoints, IPs)
```

---

## 🔒 Security & Compliance (CRITICAL)

### HIPAA Compliance Checklist
```bash
# Authentication & Authorization
✅ Multi-factor authentication implemented
✅ Role-based access control (RBAC) enforced
✅ Session management with secure tokens
✅ Password policies and encryption

# Data Protection
✅ PHI encryption at rest (AES-256)
✅ Data transmission encryption (TLS 1.3)
✅ Field-level encryption for sensitive data
✅ Secure key management (Vault)

# Audit & Monitoring
✅ Comprehensive audit logging
✅ Real-time security monitoring
✅ Incident response procedures
✅ Regular security assessments
```

### Security Monitoring Commands
```bash
# Security Logs
tail -f /var/log/medical-ai/security.log    # Security events
tail -f /var/log/medical-ai/audit.log       # Audit trail
tail -f /var/log/medical-ai/access.log      # Access logs

# Security Scanning
docker scan medical-ai/backend:latest       # Container vulnerability scan
npm audit                                   # Frontend dependency scan
bandit -r backend/                          # Python security analysis
```

---

## 🔧 Troubleshooting Guide

### Common Issues & Solutions
```bash
# Database Connection Issues
Error: "connection to server at localhost:5432 failed"
Solution: 
  - Check PostgreSQL service: systemctl status postgresql
  - Verify DATABASE_URL environment variable
  - Check database credentials and network connectivity

# AI Model Loading Issues  
Error: "Failed to load model weights"
Solution:
  - Verify OPENAI_API_KEY is set and valid
  - Check model file paths in ml/models/
  - Ensure sufficient disk space for model files

# Redis Connection Issues
Error: "Redis connection failed"
Solution:
  - Check Redis service: redis-cli ping
  - Verify REDIS_URL environment variable
  - Check Redis memory usage and configuration

# Frontend Build Issues
Error: "Module not found" or TypeScript errors
Solution:
  - Clear node_modules: rm -rf node_modules && npm install
  - Check package.json dependencies
  - Verify TypeScript configuration in tsconfig.json
```

### Health Check Endpoints
```bash
# System Health Validation
curl http://localhost:8000/health              # Basic health check
curl http://localhost:8000/health/database     # Database connectivity
curl http://localhost:8000/health/redis        # Redis connectivity  
curl http://localhost:8000/health/ai           # AI model availability
curl http://localhost:3000/                    # Frontend accessibility
```

---

## 📊 Key Performance Indicators

### Performance Targets
```yaml
Response Times:
  API Response: < 500ms
  AI Processing: < 5 seconds  
  Database Queries: < 100ms
  WebSocket Latency: < 50ms

Availability:
  System Uptime: 99.9%
  Database Availability: 99.9%
  AI Model Availability: 99.5%

Throughput:
  Concurrent Users: 10,000+
  API Requests/Hour: 100,000+
  Chat Messages/Minute: 5,000+
```

### Monitoring Commands
```bash
# Performance Monitoring
curl http://localhost:8000/metrics    # Prometheus metrics
htop                                  # System resource usage
iotop                                 # Disk I/O monitoring
docker stats                         # Container resource usage
```

---

## 📝 Important Notes for AI Agents

### Development Priorities
1. **Patient Safety First**: All code changes must maintain medical safety standards
2. **HIPAA Compliance**: PHI protection is non-negotiable in all implementations
3. **Testing Required**: Comprehensive testing for healthcare-critical functionality
4. **Documentation**: Update docs for any API or architecture changes
5. **Performance**: Monitor AI response times and system performance

### Code Modification Guidelines
- **High Risk**: AI agent logic, authentication, PHI handling, database schemas
- **Medium Risk**: API endpoints, frontend components, configuration changes  
- **Low Risk**: Documentation, logging, monitoring, non-critical UI updates

### Emergency Protocols
- **Red Flag Detection**: Immediate escalation for emergency medical situations
- **System Failures**: Automated failover and manual escalation procedures
- **Security Incidents**: Immediate incident response team notification
- **Data Breaches**: HIPAA breach notification procedures (72-hour requirement)

---

## 🔗 Quick Reference Links

### Documentation Locations
- **Technical Docs**: `/docs/technical/`
- **API Documentation**: `/docs/api/` (OpenAPI/Swagger)
- **Deployment Guides**: `/docs/deployment/`
- **Security Policies**: `/docs/security/`
- **Compliance Docs**: `/docs/compliance/`

### Key Configuration Files
- **Backend Config**: `backend/app/config.py`
- **Database Config**: `backend/app/database.py`  
- **AI Agent Config**: `backend/agent/orchestrator.py`
- **Frontend Config**: `frontend/src/config/`
- **Docker Config**: `docker-compose.yml`
- **K8s Config**: `deployment/kubernetes/`

### Support Contacts
- **Technical Lead**: Medical AI Assistant Development Team
- **Security Officer**: Information Security Team  
- **Compliance Officer**: Healthcare Compliance Department
- **Emergency Escalation**: 24/7 healthcare support hotline

---

**⚠️ CRITICAL REMINDER**: This is a HIPAA-compliant healthcare system. All modifications must maintain patient data protection, medical safety standards, and regulatory compliance. When in doubt, escalate to the technical lead or compliance officer.

**Document Status**: Live document - Updated with each major system change  
**Classification**: Confidential - Authorized Personnel Only  
**Last Updated**: November 4, 2025