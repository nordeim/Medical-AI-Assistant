# Medical AI Assistant - Installation & Setup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Quick Start (Docker Compose)](#quick-start-docker-compose)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Model Setup](#model-setup)
7. [Development Environment](#development-environment)
8. [Demo Environment Setup](#demo-environment-setup)
9. [Verification & Testing](#verification--testing)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

Before installing the Medical AI Assistant, ensure you have the following software installed:

#### Essential Tools
- **Docker** (20.10.0 or higher)
- **Docker Compose** (2.0.0 or higher)
- **Git** (2.30.0 or higher)
- **Python** (3.11 or higher) - for local development
- **Node.js** (22.x or higher) - for frontend development

#### Optional Tools (for local development without Docker)
- **PostgreSQL** (17 or higher)
- **Redis** (7 or higher)
- **CUDA** (11.8 or higher) - for GPU-accelerated inference

### System Verification
Verify your installations:

```bash
# Check Docker
docker --version
docker-compose --version

# Check Python
python --version
pip --version

# Check Node.js
node --version
npm --version

# Check Git
git --version
```

---

## System Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 20 GB free space
- **Network**: Stable internet connection for model downloads

### Recommended Requirements
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 32 GB
- **Storage**: 50 GB+ SSD
- **GPU**: NVIDIA RTX 3080/4070 or higher (8GB VRAM minimum)
- **Network**: High-speed internet for initial setup

### GPU Requirements (Optional)
For local model inference with good performance:
- **NVIDIA GPU** with 8GB+ VRAM
- **CUDA 11.8+** or **CUDA 12.x**
- **NVIDIA Container Runtime** (for Docker GPU access)

```bash
# Verify GPU and CUDA
nvidia-smi
nvcc --version
```

---

## Quick Start (Docker Compose)

The fastest way to get the Medical AI Assistant running is using Docker Compose.

### Step 1: Clone the Repository

```bash
git clone https://github.com/nordeim/Medical-AI-Assistant.git
cd Medical-AI-Assistant
```

### Step 2: Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (see Environment Configuration section)
nano .env
```

### Step 3: Generate Required Secrets

Generate secure keys for your environment:

```bash
# Generate JWT secret
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(32))"

# Generate encryption key
python -c "from cryptography.fernet import Fernet; print('PHI_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"

# Generate general secret
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
```

Update your `.env` file with these generated values.

### Step 4: Build and Start Services

```bash
# Build and start all services
docker compose up --build

# Or run in background
docker compose up --build -d
```

### Step 5: Verify Installation

Open your browser and navigate to:
- **Patient Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

---

## Environment Configuration

### Core Configuration File (.env)

The `.env` file contains all configuration settings. Here's a breakdown of key sections:

#### Database Configuration
```env
DATABASE_URL=postgresql://meduser:medpass@db:5432/meddb
POSTGRES_USER=meduser
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=meddb
```

#### Security Configuration
```env
SECRET_KEY=your-generated-secret-key
JWT_SECRET=your-generated-jwt-secret
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
```

#### Model Configuration
```env
MODEL_NAME=mistral-7b-instruct-v0.2
MODEL_PATH=/models/mistral-7b-instruct-v0.2
ADAPTER_PATH=/models/lora_adapters/medical_v1
USE_8BIT=true
MODEL_DEVICE=cuda  # Change to 'cpu' if no GPU available
```

#### API Configuration
```env
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
FRONTEND_URL=http://localhost:3000
```

### Configuration Templates

#### Development Configuration
```env
# Development-specific settings
DEBUG=true
RELOAD_ON_CHANGE=true
ENABLE_PROFILING=true
LOG_LEVEL=DEBUG
API_RELOAD=true

# Use smaller models for development
MODEL_DEVICE=cpu
USE_8BIT=false
DEMO_MODE=true
USE_SYNTHETIC_DATA=true
```

#### Production Configuration
```env
# Production security settings
DEBUG=false
RELOAD_ON_CHANGE=false
ENABLE_PROFILING=false
LOG_LEVEL=INFO

# GPU acceleration
MODEL_DEVICE=cuda
USE_8BIT=true
USE_FLASH_ATTENTION=true

# Production database
DATABASE_URL=postgresql://meduser:secure-pass@prod-db:5432/meddb
```

---

## Database Setup

### PostgreSQL Configuration

The system uses PostgreSQL 17 for primary data storage:

#### Automatic Setup (Docker)
When using Docker Compose, PostgreSQL is automatically configured:

1. **Database Creation**: The `meddb` database is created on first startup
2. **User Setup**: The `meduser` user with secure password
3. **Initialization**: Schema and initial data are loaded automatically

#### Manual Setup (Non-Docker)

If setting up PostgreSQL manually:

```sql
-- Create database and user
CREATE DATABASE meddb;
CREATE USER meduser WITH ENCRYPTED PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE meddb TO meduser;

-- Connect to database
\c meddb meduser;

-- Run initialization scripts
\i /path/to/scripts/init-db.sql
```

#### Database Schema

The system automatically creates these tables:
- **sessions**: Patient interaction sessions
- **assessments**: AI-generated preliminary assessments
- **audit_logs**: Security and compliance audit trail
- **users**: System users (patients, nurses, admins)
- **chat_messages**: WebSocket chat history

#### Database Migrations

Run database migrations after initial setup:

```bash
# Apply migrations
docker compose exec backend alembic upgrade head

# Or without Docker
cd backend
alembic upgrade head
```

### Redis Configuration

Redis is used for caching and rate limiting:

#### Docker Setup
Redis is automatically configured in Docker Compose with persistence enabled.

#### Manual Setup
```bash
# Install Redis
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Start Redis
redis-server

# Set a password (edit redis.conf)
requirepass your-redis-password
```

---

## Model Setup

### Model Architecture

The Medical AI Assistant supports multiple LLM architectures:
- **Mistral 7B** (recommended for demo)
- **Llama 2/3** (7B, 13B, 70B)
- **Phi-3** models
- Custom fine-tuned models

### Model Download and Setup

#### Option 1: Automatic Download (Recommended)

The system automatically downloads models on first startup:

```bash
# Start with demo configuration
docker compose up backend

# Models will be downloaded to:
# ./models/mistral-7b-instruct-v0.2/
# ./vector_stores/medical/
```

#### Option 2: Manual Download

```bash
# Create models directory
mkdir -p models/vector_stores

# Download base model
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 models/mistral-7b-instruct-v0.2

# Download LoRA adapter (if available)
git clone https://huggingface.co/your-org/medical-lora-v1 models/lora_adapters/medical_v1

# Download embedding model
git clone https://huggingface.co/sentence-transformers/all-mpnet-base-v2 models/all-mpnet-base-v2
```

### Model Configuration

#### Environment Variables
```env
# Base model
MODEL_NAME=mistral-7b-instruct-v0.2
MODEL_PATH=/app/models/mistral-7b-instruct-v0.2

# LoRA adapter
ADAPTER_PATH=/app/models/lora_adapters/medical_v1

# Quantization
USE_8BIT=true
USE_FLASH_ATTENTION=false

# Device selection
MODEL_DEVICE=cuda  # or 'cpu'
MODEL_BATCH_SIZE=1

# Inference parameters
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
```

### Vector Store Setup

#### ChromaDB (Default)
```env
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=/app/vector_stores/medical
VECTOR_STORE_COLLECTION=medical_guidelines
```

#### Initialize Vector Store

```bash
# Create vector store directory
mkdir -p vector_stores/medical

# Download medical guidelines (example)
# Place your medical corpus files in vector_stores/medical/
```

#### Populate with Medical Content

```python
# Example script to populate vector store
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Create vector store
vectorstore = Chroma.from_texts(
    texts=medical_texts,
    embedding=embeddings,
    collection_name="medical_guidelines",
    persist_directory="./vector_stores/medical"
)
```

---

## Development Environment

### Local Development Setup

For developers who want to work without Docker:

#### Backend Development

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://meduser:medpass@localhost:5432/meddb"
export REDIS_URL="redis://localhost:6379/0"

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Development

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Set environment variables
export REACT_APP_API_URL="http://localhost:8000"
export REACT_APP_WS_URL="ws://localhost:8000/ws"

# Start development server
npm start
```

### Development Tools

#### Code Quality Tools

```bash
# Install development dependencies
pip install black flake8 mypy isort pytest pytest-asyncio

# Format code
black backend/
isort backend/

# Lint code
flake8 backend/
mypy backend/

# Run tests
pytest backend/tests/
```

#### Hot Reload Setup

For development with auto-reload:

```env
# Development .env additions
DEBUG=true
RELOAD_ON_CHANGE=true
API_RELOAD=true
LOG_LEVEL=DEBUG
```

---

## Demo Environment Setup

### Quick Demo Setup

The demo environment provides a complete system with synthetic medical data:

```bash
# Navigate to demo directory
cd demo

# Run demo setup script
./setup_demo.sh

# Or manually:
python -m venv demo_env
source demo_env/bin/activate
pip install -r requirements.txt

# Generate synthetic data
python generate_synthetic.py

# Run demo training
python demo_train.py --data synthetic_small.jsonl --output_dir ./demo_lora --epochs 1

# Start demo inference
python demo_infer.py
```

### Demo Features

- **Synthetic Patient Data**: HIPAA-compliant mock patient information
- **Medical Scenarios**: Realistic medical cases across specialties
- **Performance Optimization**: Faster responses for demonstrations
- **Educational Content**: Medical guidelines and protocols

### Demo Configuration

```env
# Enable demo mode
DEMO_MODE=true
USE_SYNTHETIC_DATA=true
MOCK_EHR_RESPONSES=true

# Demo credentials (use these to log in)
# Patient: patient.smith@demo.medai.com / DemoPatient789!
# Nurse: nurse.jones@demo.medai.com / DemoNurse456!
# Admin: admin@demo.medai.com / DemoAdmin123!
```

---

## Verification & Testing

### Health Checks

#### API Health Check
```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-04T17:56:15Z",
  "version": "1.0.0",
  "database": "connected",
  "redis": "connected",
  "model": "loaded"
}
```

#### Database Health
```bash
# Check database connection
docker compose exec db psql -U meduser -d meddb -c "SELECT 1;"

# Check tables exist
docker compose exec db psql -U meduser -d meddb -c "\dt"
```

#### Redis Health
```bash
# Check Redis connection
docker compose exec redis redis-cli ping
# Expected: PONG
```

### System Verification

#### Complete System Test

```bash
# 1. Check all services are running
docker compose ps

# 2. Test API endpoints
curl -X POST http://localhost:8000/api/v1/session \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "test", "session_type": "intake"}'

# 3. Test WebSocket connection
# Use browser or WebSocket client to connect to ws://localhost:8000/ws/chat

# 4. Check frontend
curl http://localhost:3000
```

### Load Testing

#### API Load Test

```bash
# Install Apache Bench or similar
sudo apt install apache2-utils  # Ubuntu/Debian

# Test API with concurrent requests
ab -n 100 -c 10 http://localhost:8000/api/v1/health
```

### Integration Tests

```bash
# Run backend tests
docker compose exec backend pytest backend/tests/ -v

# Run frontend tests
docker compose exec frontend npm test
```

---

## Troubleshooting

### Common Issues

#### 1. Docker Issues

**Problem**: Docker containers fail to start
```bash
# Check Docker daemon
docker info

# Check disk space
df -h

# Check permissions
docker ps
```

**Solution**:
```bash
# Restart Docker daemon
sudo systemctl restart docker

# Clean up Docker
docker system prune -a

# Rebuild containers
docker compose down
docker compose up --build --force-recreate
```

#### 2. Database Connection Issues

**Problem**: Backend can't connect to database
```bash
# Check database is running
docker compose exec db pg_isready -U meduser

# Check connection string
docker compose logs db
```

**Solution**:
```bash
# Reset database
docker compose down -v
docker compose up -d db
# Wait for db to be ready, then start other services
docker compose up backend frontend
```

#### 3. Model Loading Issues

**Problem**: Models fail to load or download
```bash
# Check model directory permissions
ls -la models/

# Check disk space
df -h
```

**Solution**:
```bash
# Download models manually
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')
"

# Use smaller model for testing
export MODEL_NAME=microsoft/DialoGPT-small
```

#### 4. GPU Issues

**Problem**: CUDA out of memory or GPU not detected
```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version
```

**Solution**:
```env
# Switch to CPU in .env
MODEL_DEVICE=cpu
USE_8BIT=false

# Or reduce batch size
MODEL_BATCH_SIZE=1
MAX_NEW_TOKENS=256
```

#### 5. Port Conflicts

**Problem**: Ports already in use
```bash
# Check port usage
netstat -tulpn | grep :8000
```

**Solution**:
```bash
# Change ports in .env
API_PORT=8001
FRONTEND_URL=http://localhost:3001

# Update docker-compose.yml if needed
ports:
  - "8001:8000"  # Backend
  - "3001:3000"  # Frontend
```

### Performance Optimization

#### For Development
```env
# Use CPU for development
MODEL_DEVICE=cpu
USE_8BIT=false

# Reduce resource usage
API_WORKERS=1
MODEL_BATCH_SIZE=1
CACHE_TTL_SECONDS=600
```

#### For Production
```env
# GPU acceleration
MODEL_DEVICE=cuda
USE_8BIT=true
USE_FLASH_ATTENTION=true

# Optimize for throughput
API_WORKERS=4
MODEL_BATCH_SIZE=4
RATE_LIMIT_PER_MINUTE=1000
```

### Logs and Monitoring

#### View Logs
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f db

# Follow specific logs
docker compose logs -f backend | grep ERROR
```

#### Health Monitoring
```bash
# Set up monitoring script
cat > health_monitor.sh << 'EOF'
#!/bin/bash
while true; do
  curl -f http://localhost:8000/api/v1/health > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    echo "System healthy at $(date)"
  else
    echo "System unhealthy at $(date)"
  fi
  sleep 30
done
EOF

chmod +x health_monitor.sh
./health_monitor.sh
```

### Getting Help

#### Documentation
- API Documentation: http://localhost:8000/docs
- System Architecture: See `docs/architecture_overview.md`
- Configuration Reference: See `.env.example`

#### Community Support
- GitHub Issues: https://github.com/nordeim/Medical-AI-Assistant/issues
- Wiki: https://github.com/nordeim/Medical-AI-Assistant/wiki

#### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export ENABLE_PROFILING=true

# Run with verbose output
docker compose up --verbose
```

---

## Security Considerations

### Production Security Checklist

- [ ] Change all default passwords
- [ ] Use strong, unique secrets (JWT, encryption keys)
- [ ] Enable HTTPS in production
- [ ] Configure firewall rules
- [ ] Set up regular security updates
- [ ] Enable audit logging
- [ ] Configure backup procedures
- [ ] Review and test disaster recovery

### HIPAA Compliance (if applicable)

- [ ] Ensure no PHI in logs
- [ ] Enable data encryption at rest and in transit
- [ ] Set up access controls and audit trails
- [ ] Implement data retention policies
- [ ] Regular security assessments
- [ ] Staff training on privacy practices

---

## Next Steps

After successful installation:

1. **Explore the API**: Visit http://localhost:8000/docs for interactive API documentation
2. **Test the Demo**: Log in with demo credentials to explore features
3. **Customize Configuration**: Adjust settings in `.env` for your use case
4. **Set Up Monitoring**: Configure health checks and alerting
5. **Plan Deployment**: Review production deployment guides
6. **Train Custom Models**: Use the training pipeline for domain-specific models

### Additional Resources

- [API Documentation](http://localhost:8000/docs)
- [System Architecture](docs/architecture_overview.md)
- [Deployment Guide](deployment/README.md)
- [Training Pipeline](training/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

**Installation Guide Version**: 1.0  
**Last Updated**: November 4, 2024  
**Compatibility**: Medical AI Assistant v1.0
