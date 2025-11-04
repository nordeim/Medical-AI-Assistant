# Medical AI Model Serving Infrastructure

A production-ready, secure model serving infrastructure designed for medical AI applications with proper data compliance, monitoring, and scalability features.

## 🏗️ Architecture Overview

This infrastructure provides:

- **FastAPI-based REST API** with comprehensive security middleware
- **Multiple model server types** (Text Generation, Embeddings, Conversation)
- **Model caching and optimization** with Redis and disk-based caching
- **Medical data compliance** with PHI redaction and audit logging
- **Comprehensive monitoring** with health checks, metrics, and alerting
- **Structured logging** with request tracking and medical data protection
- **Containerized deployment** with Docker and Docker Compose

## 📁 Project Structure

```
serving/
├── api/                    # FastAPI application
│   ├── __init__.py
│   └── main.py            # Main API server
├── adapters/              # Model adapters for different frameworks
│   ├── __init__.py
│   └── model_adapters.py  # Adapter implementations
├── config/                # Configuration management
│   ├── __init__.py
│   ├── logging_config.py  # Logging infrastructure
│   └── settings.py        # Settings and validation
├── models/                # Model serving implementations
│   ├── __init__.py
│   ├── base_server.py     # Base server classes
│   └── concrete_servers.py # Specific model servers
├── cache/                 # Cache storage directory
├── logs/                  # Log files directory
│   ├── __init__.py
│   └── monitoring.py      # Monitoring and health checks
├── config/                # Configuration files
├── main.py               # Main server entry point
├── requirements.txt      # Python dependencies
├── .env.example         # Environment configuration template
├── start_dev.sh         # Development startup script
├── start_prod.sh        # Production startup script
├── Dockerfile           # Docker container configuration
├── docker-compose.yml   # Multi-service deployment
└── tests.py            # Test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Redis (optional, for caching)
- PostgreSQL (optional, for persistent storage)
- Docker (optional, for containerized deployment)

### Development Setup

1. **Clone and setup:**
   ```bash
   cd serving/
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Start development server:**
   ```bash
   ./start_dev.sh
   # Or manually: python main.py
   ```

4. **Access the API:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics

### Production Deployment

1. **Configure for production:**
   ```bash
   cp .env.example .env
   # Set APP_ENVIRONMENT=production
   # Configure production settings
   ```

2. **Start with Gunicorn:**
   ```bash
   ./start_prod.sh
   ```

### Docker Deployment

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Access services:**
   - API: http://localhost:8000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9090

## ⚙️ Configuration

### Environment Variables

The system uses environment variables for configuration. See `.env.example` for all available options.

**Key configurations:**

```bash
# Application
APP_ENVIRONMENT=development|production
APP_DEBUG=true|false

# Model Settings
MODEL_MODEL_NAME=microsoft/DialoGPT-medium
MODEL_USE_QUANTIZATION=true
MODEL_QUANTIZATION_TYPE=4bit

# Server Settings
SERVING_HOST=0.0.0.0
SERVING_PORT=8000
SERVING_WORKERS=4

# Medical Compliance
MEDICAL_ENABLE_ENCRYPTION=true
MEDICAL_PHI_REDACTION=true
MEDICAL_ENABLE_AUDIT_LOG=true
```

### Model Configuration

Support for multiple model types:

- **Text Generation**: GPT-style models for completion
- **Embeddings**: Sentence transformers for text embeddings
- **Conversation**: Multi-turn dialogue models

Models can be loaded from:
- Hugging Face Hub
- Local files (PyTorch, ONNX, pickle)
- Quantized versions (4-bit, 8-bit)

## 🔒 Security & Compliance

### Medical Data Protection

- **PHI Redaction**: Automatic detection and redaction of sensitive medical data
- **Data Encryption**: Optional encryption for stored data
- **Audit Logging**: Comprehensive audit trail for compliance
- **Access Control**: Role-based access control (RBAC)

### Security Features

- **API Key Authentication**: Optional API key protection
- **Rate Limiting**: Configurable rate limits per client
- **CORS Configuration**: Restricted cross-origin access
- **Request Validation**: Input validation and sanitization
- **Error Handling**: Secure error responses without sensitive data

## 📊 Monitoring & Health Checks

### Health Endpoints

- `GET /health` - Overall system health
- `GET /metrics` - Performance metrics
- `GET /models/{id}/health` - Model-specific health

### Monitoring Features

- **System Metrics**: CPU, memory, disk, GPU usage
- **Model Metrics**: Response times, success rates, cache hits
- **Alert System**: Configurable alerts for thresholds
- **Structured Logging**: JSON-based logging with request tracking

### Metrics Collection

- Prometheus-compatible metrics
- Grafana dashboard integration
- Real-time monitoring service
- Historical trend analysis

## 🔧 API Reference

### Core Endpoints

#### Health Check
```http
GET /health
```

#### List Models
```http
GET /models
```

#### Model Information
```http
GET /models/{model_id}/info
```

#### Make Prediction
```http
POST /models/{model_id}/predict
Content-Type: application/json

{
  "inputs": "Patient shows symptoms of...",
  "parameters": {
    "max_new_tokens": 100,
    "temperature": 0.7
  },
  "user_id": "user123",
  "session_id": "session456"
}
```

#### System Metrics
```http
GET /metrics
```

### Response Format

All responses follow a consistent format:

```json
{
  "request_id": "uuid",
  "model_id": "text_generation_v1",
  "outputs": "Generated text...",
  "processing_time": 1.23,
  "timestamp": "2025-11-04T07:01:56Z",
  "metadata": {
    "tokens_generated": 50,
    "model_version": "v1.0.0"
  }
}
```

## 🧪 Testing

### Run Test Suite
```bash
python tests.py
```

### Test Categories

- **Configuration Tests**: Settings validation and loading
- **API Tests**: Endpoint functionality and error handling
- **Model Tests**: Server initialization and prediction
- **Security Tests**: Authentication and data validation
- **Integration Tests**: End-to-end workflow testing

## 🐳 Docker Deployment

### Services Included

- **Model Serving API** - Main application
- **Redis** - Caching layer
- **PostgreSQL** - Persistent storage
- **Prometheus** - Metrics collection
- **Grafana** - Metrics visualization
- **Nginx** - Reverse proxy

### Docker Commands

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f model-serving

# Stop services
docker-compose down

# Rebuild services
docker-compose up -d --build
```

## 🔧 Advanced Configuration

### Model Cache Configuration

```python
# Cache settings in settings.py
cache = {
    "redis_url": "redis://localhost:6379/0",
    "cache_ttl": 3600,  # 1 hour
    "max_cache_size": 1000,
    "enable_model_cache": True,
    "enable_response_cache": True,
    "disk_cache_dir": "./cache",
    "disk_cache_size": 10737418240  # 10GB
}
```

### Monitoring Thresholds

```python
# Alert thresholds
alert_thresholds = {
    "cpu_percent": 80,
    "memory_percent": 85,
    "disk_percent": 90,
    "response_time": 5.0,
    "error_rate": 0.05
}
```

### Medical Compliance Settings

```python
# Medical data handling
medical = {
    "enable_encryption": True,
    "phi_redaction": True,
    "allowed_phi_fields": ["patient_id", "timestamp"],
    "data_retention_days": 30,
    "enable_audit_log": True,
    "audit_log_file": "./logs/audit.log"
}
```

## 📈 Performance Optimization

### Model Optimization

- **Quantization**: 4-bit and 8-bit quantization support
- **Model Caching**: Intelligent caching of model states
- **Memory Management**: Automatic GPU memory optimization
- **Batch Processing**: Configurable batch sizes

### Caching Strategy

- **Redis**: Distributed caching for high availability
- **Disk Cache**: Local disk caching for large models
- **LRU Eviction**: Least recently used cache eviction
- **TTL Management**: Time-based cache expiration

### Scaling Options

- **Horizontal Scaling**: Multiple model server instances
- **Load Balancing**: Nginx-based load distribution
- **Database Clustering**: PostgreSQL clustering support
- **Container Orchestration**: Kubernetes-ready configuration

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```bash
   # Check model path and permissions
   ls -la models/
   # Verify model format and compatibility
   ```

2. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats
   # Adjust model quantization settings
   ```

3. **Cache Connection Issues**
   ```bash
   # Test Redis connection
   redis-cli ping
   # Check Redis logs
   docker-compose logs redis
   ```

### Logging

Logs are available in:
- **Application Logs**: `./logs/serving.log`
- **Audit Logs**: `./logs/audit.log`
- **Access Logs**: `./logs/access.log`
- **Error Logs**: `./logs/error.log`

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Check specific model health
curl http://localhost:8000/models/text_generation_v1/health

# Get metrics
curl http://localhost:8000/metrics
```

## 🤝 Contributing

1. Follow the established code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure medical compliance for health data
5. Follow security best practices

## 📄 License

This infrastructure is designed for medical AI applications and includes security features for handling protected health information (PHI).

---

For more detailed documentation, see the individual module docstrings and the API documentation at `/docs` when running the server.