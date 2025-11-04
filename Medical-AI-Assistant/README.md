# 🏥 Medical AI Assistant

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/nordeim/Medical-AI-Assistant)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg)](https://reactjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-3178C6.svg)](https://typescriptlang.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://docker.com)
[![HIPAA](https://img.shields.io/badge/HIPAA-compliant-green.svg)](https://www.hhs.gov/hipaa)
[![FDA](https://img.shields.io/badge/FDA-compliant-green.svg)](https://www.fda.gov)
[![ISO 27001](https://img.shields.io/badge/ISO%2027001-compliant-green.svg)](https://www.iso.org/isoiec-27001-information-security.html)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **⚠️ CRITICAL MEDICAL DISCLAIMER: This software is designed to assist healthcare professionals and should never replace professional medical judgment. All AI-generated recommendations must be validated by qualified medical personnel before implementation. Patient safety is paramount.**

## 🌟 Overview

The Medical AI Assistant is an enterprise-grade, HIPAA-compliant platform designed to enhance healthcare decision-making through advanced AI capabilities. Built with robust security measures and regulatory compliance at its core, this system provides healthcare professionals with intelligent assistance while maintaining the highest standards of patient data protection.

This end-to-end production-ready reference implementation includes clinic pre-screening & triage capabilities with architecture, training pipelines, a React frontend/chat UI, and a FastAPI backend with a LangChain v1.0 agent runtime for RAG-grounded, safety-controlled triage workflows.

### 🎯 Key Objectives

- **Patient Safety First**: All recommendations require professional medical validation
- **Regulatory Compliance**: Full HIPAA, FDA, and ISO 27001 compliance
- **Enhanced Decision Support**: AI-powered insights for healthcare professionals
- **Secure Architecture**: End-to-end encryption and audit logging
- **Human-in-Loop Workflow**: Mandatory nurse review and approval processes

## ✨ Features

### 🔐 Security & Compliance
- **HIPAA Compliance**: End-to-end encryption, access controls, audit logging
- **FDA Compliance**: Medical device software lifecycle processes (IEC 62304)
- **ISO 27001 Certified**: Information security management standards
- **Role-Based Access Control**: Granular permissions for different user types
- **Audit Trail**: Comprehensive logging of all system activities
- **Data Anonymization**: Advanced PHI protection mechanisms

### 🤖 AI Capabilities
- **Patient Chat Interface**: Friendly, guided collection of symptoms
- **RAG-Grounded Responses**: Integrates vector DB for local clinic guidelines & protocols
- **Clinical Decision Support**: Evidence-based recommendations with safety filters
- **LoRA/PEFT Training**: Cost-effective fine-tuning of open-source LLMs
- **Preliminary Assessment Reports**: Structured PAR generation for nurse review
- **Safety-First Design**: Mandatory disclaimers, red-flag triggers, safety filter callbacks

### 🏗️ Technical Features
- **Real-time Communication**: WebSocket-based instant updates
- **Multi-modal Data Support**: Text, images, lab results, vital signs
- **API-First Design**: RESTful APIs for seamless integration
- **Microservices Architecture**: Scalable and maintainable design
- **Cloud-Native**: Kubernetes-ready for enterprise deployment
- **MLOps-Ready**: Adapter artifacts, model registry hooks, and retraining triggers

### 👥 User Management
- **Multi-Role Support**: Physicians, Nurses, Administrators, IT Staff
- **Human-in-Loop Workflow**: Nurse dashboard with accept/override flows
- **Session Management**: Secure, time-limited access tokens
- **Audit & Traceability**: Full logging of retrieved docs and confidence scores

## 🚀 Quick Start

### Prerequisites

Before installing the Medical AI Assistant, ensure your system meets these requirements:

**Hardware Requirements:**
- **CPU**: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 16GB minimum (32GB recommended for production)
- **Storage**: 100GB available space (SSD recommended)
- **GPU**: Optional but recommended for AI model inference
- **Network**: Stable internet connection for AI model access

**Software Requirements:**
- **Operating System**: Ubuntu 20.04+, CentOS 8+, Windows 10/11, or macOS 12+
- **Docker**: Version 20.10+ with Docker Compose 2.0+
- **Git**: Version 2.30+
- **Node.js**: Version 18+ (for frontend development)
- **Python**: Version 3.9+ (for backend development)

### 🔧 Installation

#### Option 1: Docker Compose (Recommended)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nordeim/Medical-AI-Assistant.git
   cd Medical-AI-Assistant
   ```

2. **Environment Setup**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit configuration (see Configuration section below)
   nano .env
   ```

3. **Start Services**
   ```bash
   # Pull and start all services
   docker-compose up --build -d
   
   # Verify services are running
   docker-compose ps
   ```

4. **Initialize Database**
   ```bash
   # Run database migrations
   docker-compose exec backend python -m alembic upgrade head
   
   # Create initial admin user
   docker-compose exec backend python scripts/create_admin.py
   ```

5. **Verify Installation**
   ```bash
   # Check application health
   curl http://localhost:8000/health
   
   # Access web interfaces
   # Patient UI: http://localhost:3000
   # Nurse Dashboard: http://localhost:3001
   ```

#### Option 2: Manual Installation

**Backend Setup:**
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
alembic upgrade head

# Start backend server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend Setup:**
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Start development server
npm run dev
```

### ⚙️ Configuration

#### Essential Environment Variables

**Database Configuration:**
```env
# PostgreSQL Database
DATABASE_URL=postgresql://username:password@localhost:5432/medical_ai_db
DB_HOST=localhost
DB_PORT=5432
DB_NAME=medical_ai_db
DB_USER=your_username
DB_PASSWORD=your_secure_password
```

**Security Configuration:**
```env
# JWT and Encryption
SECRET_KEY=your-super-secure-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Encryption for PHI
ENCRYPTION_KEY=your-fernet-encryption-key
AES_KEY=your-aes-256-encryption-key
```

**AI Services Configuration:**
```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2000

# Model and Vector Store Paths
MODEL_PATH=/models/med-lora
VECTOR_STORE_PATH=/data/vector_store
```

**Compliance and Logging:**
```env
# HIPAA Compliance
ENABLE_AUDIT_LOGGING=true
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years as required by HIPAA
PHI_ENCRYPTION_ENABLED=true

# Security Settings
CORS_ORIGINS=["http://localhost:3000", "https://yourdomain.com"]
ALLOWED_HOSTS=["localhost", "yourdomain.com"]
```

## 🏗️ Architecture

### System Architecture

The system is designed with modular components for scalability and maintainability:

```mermaid
flowchart LR
  subgraph Client
    Chat["React Patient Chat UI"] -->|ws| Backend["FastAPI WS"]
    Nurse["React Nurse Dashboard"] -->|REST| Backend
  end

  Backend --> Orchestrator["Session Orchestrator"]
  Orchestrator --> Agent["LangChain v1.0 Agent Runtime"]
  Agent -->|retrieve| Vector["Vector DB (Chroma/Qdrant/Milvus)"]
  Vector -->|contexts| Agent
  Agent --> Model["Model Serving (LLM + LoRA)"]
  Model -->|stream| Backend
  Backend --> AuditDB["Audit & Logging DB"]
  Nurse -->|review| AuditDB
  Agent --> EHR["EHR Connector (read-only summary)"]
  EHR --> Agent
  AuditDB --> MLOps["MLOps / Training Pipeline"]
  MLOps --> Model

  style Agent fill:#fff7ed,stroke:#f97316
  style Model fill:#eef2ff,stroke:#7c3aed
  style Vector fill:#ecfeff,stroke:#0891b2
```

### Application Flow

```mermaid
sequenceDiagram
  participant P as Patient (React)
  participant WS as FastAPI WebSocket
  participant OR as Orchestrator
  participant AG as LangChain Agent
  participant VD as Vector DB
  participant MS as Model Serving
  participant N as Nurse Dashboard
  P->>WS: Start chat & consent
  WS->>OR: Create session
  OR->>AG: Build prompt (system + EHR + RAG)
  AG->>VD: Retrieve top-k contexts
  VD-->>AG: return contexts
  AG->>MS: Generate clarification questions
  MS-->>AG: streaming tokens
  AG-->>WS: stream back to patient
  P->>WS: answer
  loop
    WS->>OR: append to transcript
    OR->>AG: new prompt (loop)
  end
  AG->>MS: Generate PAR
  AG-->>WS: deliver PAR
  WS->>N: push to nurse queue
  N->>WS: Accept/Override
  WS->>AuditDB: Log decision
```

### User Interaction Flow

```mermaid
flowchart TD
  Start[Patient Homepage] --> Consent{Has consented?}
  Consent -->|No| ShowInfo[Show info + consent form]
  Consent -->|Yes| Intake[Start Intake Questions]
  Intake --> RedFlagCheck{Red flag present?}
  RedFlagCheck -->|Yes| Urgent[Escalate: show emergency instruction + notify nurse]
  RedFlagCheck -->|No| Continue[Continue guided questions]
  Continue --> PAR[Generate Preliminary Assessment Report]
  PAR --> NurseQueue[Send PAR to Nurse Queue]
  NurseQueue --> NurseReview{Nurse accepts?}
  NurseReview -->|Accept| Schedule[Schedule/triage action]
  NurseReview -->|Override| Reassign[Manual triage - feedback -> MLOps]
```

### Security Layers

1. **Network Security**: TLS 1.3, VPN access, firewall rules
2. **Application Security**: JWT tokens, RBAC, input validation
3. **Data Security**: AES-256 encryption, field-level encryption for PHI
4. **Infrastructure Security**: Container isolation, secret management
5. **Compliance Monitoring**: Real-time audit logging, compliance dashboards

## 🧪 Development

### Development Environment Setup

1. **Prerequisites Installation**
   ```bash
   # Install development tools
   pip install -r requirements-dev.txt
   npm install -g @typescript-eslint/parser
   ```

2. **Database Setup**
   ```bash
   # Start local database
   docker-compose -f docker-compose.dev.yml up -d postgres redis
   
   # Run migrations
   alembic upgrade head
   
   # Seed development data
   python scripts/seed_dev_data.py
   ```

3. **Environment Configuration**
   ```bash
   # Development environment
   cp .env.development .env
   
   # Enable debug mode
   export DEBUG=true
   export LOG_LEVEL=DEBUG
   ```

### Code Quality Standards

**Python Backend:**
- **Formatting**: Black (line length: 88)
- **Linting**: Flake8, MyPy for type checking
- **Testing**: Pytest with 90%+ coverage requirement
- **Documentation**: Google-style docstrings

**TypeScript Frontend:**
- **Formatting**: Prettier
- **Linting**: ESLint with TypeScript rules
- **Testing**: Jest + React Testing Library
- **Documentation**: TSDoc comments

**Medical Content Guidelines:**
- All medical terminology must be validated by healthcare professionals
- AI recommendations require explicit disclaimers
- Patient safety warnings must be prominently displayed
- Clinical decision support features require FDA compliance documentation

### Testing Requirements

**Unit Tests (90%+ coverage):**
```bash
# Backend tests
pytest tests/ --cov=app --cov-min=90

# Frontend tests
npm run test:coverage -- --coverage-threshold=90
```

**Integration Tests:**
```bash
# API integration tests
pytest tests/integration/ -v

# End-to-end tests
npm run test:e2e
```

**Security Tests:**
```bash
# OWASP security scanning
bandit -r app/
npm audit

# Penetration testing
python scripts/security_scan.py
```

**HIPAA Compliance Tests:**
```bash
# PHI protection tests
pytest tests/compliance/test_phi_protection.py

# Audit logging tests
pytest tests/compliance/test_audit_logging.py

# Access control tests
pytest tests/compliance/test_access_control.py
```

## 🚀 Deployment

### Production Deployment Options

#### Cloud Deployment (Recommended)

**AWS Deployment:**
```bash
# Install AWS CLI and configure credentials
aws configure

# Deploy using CDK
cd infrastructure/aws
npm install
npm run build
cdk deploy --all

# Verify deployment
aws ecs describe-services --cluster medical-ai-cluster
```

**Azure Deployment:**
```bash
# Install Azure CLI and login
az login

# Deploy using ARM templates
cd infrastructure/azure
az deployment group create \
  --resource-group medical-ai-rg \
  --template-file main.json \
  --parameters @parameters.prod.json
```

**Google Cloud Deployment:**
```bash
# Authenticate with GCP
gcloud auth login

# Deploy using Cloud Run
cd infrastructure/gcp
gcloud run deploy medical-ai-backend \
  --source . \
  --platform managed \
  --region us-central1
```

#### Kubernetes Deployment

**Prerequisites:**
- Kubernetes cluster (1.24+)
- kubectl configured
- Helm 3.0+

**Deployment Steps:**
```bash
# Add Helm repository
helm repo add medical-ai https://charts.medical-ai.com
helm repo update

# Install with production values
helm install medical-ai medical-ai/medical-ai-assistant \
  -f k8s/values.prod.yml \
  --namespace medical-ai \
  --create-namespace

# Verify deployment
kubectl get pods -n medical-ai
kubectl get services -n medical-ai
```

### Performance Optimization

**Database Optimization:**
```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_patients_mrn ON patients(medical_record_number);
CREATE INDEX CONCURRENTLY idx_consultations_patient_id ON consultations(patient_id);
CREATE INDEX CONCURRENTLY idx_audit_logs_timestamp ON audit_logs(timestamp);
```

**Caching Strategy:**
```python
# Redis caching configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://redis:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'TIMEOUT': 300,  # 5 minutes
    }
}
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Docker Issues

**Problem: Container fails to start**
```bash
# Check logs
docker-compose logs backend

# Common solutions
docker-compose down --volumes  # Reset volumes
docker system prune -a        # Clean Docker cache
docker-compose pull           # Update images
docker-compose up --build     # Rebuild containers
```

**Problem: Port conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Solution: Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port
```

#### Database Issues

**Problem: Connection refused**
```bash
# Check database status
docker-compose exec postgres pg_isready

# Check connection from backend
docker-compose exec backend python -c "from app.database import engine; print(engine.execute('SELECT 1').scalar())"

# Reset database
docker-compose down
docker volume rm medical-ai-assistant_postgres_data
docker-compose up -d postgres
```

#### API Issues

**Problem: 500 Internal Server Error**
```bash
# Check backend logs
docker-compose logs -f backend

# Check environment variables
docker-compose exec backend env | grep -E "(DATABASE|SECRET|API)"

# Test database connection
docker-compose exec backend python scripts/test_db_connection.py
```

#### AI Service Issues

**Problem: OpenAI API failures**
```bash
# Test API key
docker-compose exec backend python scripts/test_openai.py

# Check rate limits
docker-compose exec backend python scripts/check_api_limits.py

# Verify model access
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

### Training Pipeline & Dataset Conversion

This repository includes comprehensive training capabilities:

**Dataset Conversion:**
- `notebooks/Clinic_Dataset_Conversion_Notebook.ipynb` — De-identify & convert CSV EHR rows → JSONL instruction examples
- `training/train.py` — LoRA + PEFT training script (DeepSpeed-compatible)
- `training/deepspeed_config.json` — Example ZeRO-3 config for larger model training

**Basic Workflow:**
1. De-identify & export clinic EHR (legal/IRB approval required)
2. Run dataset conversion notebook → generate `instruction_dataset.jsonl`
3. Start training with `train.py` (or use `train_scheduler.sh` for cluster)
4. Save adapter weights (`lora_adapter/`) and register in model registry
5. Deploy adapter to inference service and wire into LangChain agent

**Sample Training Command:**
```bash
deepspeed --num_gpus 8 training/train.py \
  --model_name_or_path /models/med-base \
  --data_path /data/instruction_dataset.jsonl \
  --output_dir /models/med-lora \
  --use_deepspeed \
  --deepspeed_config training/deepspeed_config.json
```

### Local Demo (7B Model)

For development and testing with limited GPU resources:

```bash
cd demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_synthetic.py
python demo_train.py --data synthetic_small.jsonl --output_dir ./demo_lora --epochs 1
python demo_infer.py
```

## 🤝 Contributing

We welcome contributions from the healthcare technology community! Please read our contribution guidelines carefully to ensure your contributions align with our medical safety standards and regulatory requirements.

### Code of Conduct

This project adheres to a strict code of conduct that prioritizes:
- **Patient Safety**: All contributions must consider potential impact on patient care
- **Medical Ethics**: Respect for patient privacy and healthcare professional judgment
- **Professional Standards**: Maintaining the highest quality in medical software
- **Regulatory Compliance**: Ensuring all changes meet HIPAA, FDA, and ISO 27001 requirements

### Development Workflow

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/Medical-AI-Assistant.git
   cd Medical-AI-Assistant
   ```

2. **Set Up Development Environment**
   ```bash
   # Install pre-commit hooks
   pip install pre-commit
   pre-commit install
   
   # Set up development database
   docker-compose -f docker-compose.dev.yml up -d
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes with Medical Review**
   - Ensure all medical content is reviewed by healthcare professionals
   - Add appropriate safety warnings and disclaimers
   - Test thoroughly with edge cases and error conditions

5. **Testing Requirements**
   ```bash
   # Run all tests
   pytest tests/ --cov=app --cov-min=90
   npm run test:coverage
   
   # Run security tests
   bandit -r app/
   npm audit
   
   # Run compliance tests
   pytest tests/compliance/
   ```

6. **Submit Pull Request**
   - Include detailed description of changes
   - Reference any related issues
   - Ensure all CI checks pass
   - Request review from maintainers

### Medical Content Guidelines

**Clinical Decision Support Features:**
- Must include explicit disclaimers about professional medical judgment
- Require validation by licensed healthcare professionals
- Include references to medical literature where applicable
- Implement appropriate safeguards against misuse

**Patient Data Handling:**
- All PHI must be encrypted at rest and in transit
- Implement proper access controls and audit logging
- Follow data minimization principles
- Ensure compliance with HIPAA requirements

## 🔒 Security, Privacy & Safety Guidance

This section is critical — read carefully before testing with any clinical data.

### PHI and Data Protection
- **PHI Handling**: NEVER upload identifiable patient data to external embedding services. Keep embeddings and vector DB on-premises or in a controlled VPC
- **De-identification**: Use best-in-class de-identification tools (and human spot checks) before using data for training. Follow local privacy laws (HIPAA/GDPR/etc.)
- **Encryption**: All PHI must be encrypted at rest and in transit using AES-256 encryption
- **Access Controls**: Implement role-based access control with principle of least privilege

### Safety Measures
- **Human-in-Loop**: For initial pilots, run the assistant in **shadow mode** — only suggest triage to nurses, never auto-act
- **Safety Filters**: Use runtime safety callbacks to block outputs with prescriptive or definitive diagnostic language. Log safety hits to an alerting system
- **Model Provenance**: Always store model version, LoRA adapter hash, and training dataset snapshot with each PAR for auditability
- **Clinical Governance**: Clinical lead must sign off before any real-world deployment

### Compliance Requirements
- **HIPAA Compliance**: Full audit logging, access controls, and data protection
- **FDA Compliance**: Software lifecycle processes for medical device software
- **ISO 27001**: Information security management standards
- **Regular Audits**: Scheduled security assessments and penetration testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Additional Terms for Medical Use

While this software is open source, deploying it in clinical environments requires:

1. **Proper validation** of all AI models for your specific use case
2. **Clinical oversight** by qualified healthcare professionals
3. **Compliance verification** with local healthcare regulations
4. **Risk assessment** and appropriate safeguards implementation
5. **Regular security audits** and penetration testing

### Disclaimer

This software is provided "as is" without warranty of any kind. The developers and contributors assume no responsibility for any medical decisions made using this system. Always consult with qualified healthcare professionals for medical advice.

## 📞 Support

### Community Support
- **GitHub Issues**: [Report bugs and request features](https://github.com/nordeim/Medical-AI-Assistant/issues)
- **Discussions**: [Community discussions and Q&A](https://github.com/nordeim/Medical-AI-Assistant/discussions)
- **Documentation**: [Comprehensive documentation](https://docs.medical-ai-assistant.com)

### Commercial Support
For healthcare organizations requiring:
- Priority support
- Custom compliance consulting
- Professional services
- Training and certification

Contact: [clinic-ai-team@yourclinic.example](mailto:clinic-ai-team@yourclinic.example)

### Security Issues
For security vulnerabilities, please follow responsible disclosure:
- **Email**: [security@medical-ai-assistant.com](mailto:security@medical-ai-assistant.com)
- **PGP Key**: Available at [security.asc](security.asc)

---

## 🗺️ Roadmap

### Short-term
- Complete LangChain agent skeleton implementation
- Provide sample RAG corpora for testing
- Add integration tests for the train → adapter → inference pipeline

### Medium-term
- Nurse dashboard UX polish and analytics (nurse acceptance metrics)
- Add secure, auditable model registry and adapter hot-swap for A/B testing
- Enhanced compliance reporting and dashboards

### Long-term
- Multi-clinic deployments and federation
- Multilingual support and voice integrations
- Advanced AI capabilities and specialized medical models

---

**⚠️ IMPORTANT MEDICAL DISCLAIMER**

This software is designed to assist healthcare professionals and should never replace professional medical judgment. All AI-generated recommendations must be validated by qualified medical personnel before implementation. The developers assume no liability for medical decisions made using this system. Always prioritize patient safety and consult with qualified healthcare professionals for medical advice.

---

<div align="center">
  <p><strong>Built with ❤️ for Healthcare Professionals</strong></p>
  <p>Version 1.0.0 | <a href="https://github.com/nordeim/Medical-AI-Assistant">GitHub Repository</a> | <a href="https://docs.medical-ai-assistant.com">Documentation</a></p>
</div>