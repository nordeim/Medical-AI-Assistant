# Medical AI Inference API - Phase 6 Documentation

## Overview

This comprehensive inference API provides enterprise-grade medical AI services with HIPAA compliance, PHI protection, and high availability for healthcare applications. Built for Phase 6 of the Medical AI Assistant project, it includes advanced features for real-time medical decision support, batch processing, and clinical validation.

## 🏥 Key Features

### Core Capabilities
- **Medical AI Inference**: Advanced medical question answering with domain specialization
- **Real-time Streaming**: Live chat experiences with streaming responses
- **Batch Processing**: Efficient processing for multiple patients simultaneously
- **Clinical Decision Support**: Evidence-based medical recommendations with accuracy validation
- **PHI Protection**: Comprehensive HIPAA-compliant data protection
- **Medical Validation**: Real-time validation of medical data and terminology

### Security & Compliance
- **HIPAA Compliance**: Full adherence to healthcare data protection standards
- **PHI Detection**: Automatic identification and protection of sensitive health information
- **Audit Logging**: Comprehensive audit trail for all medical operations
- **Rate Limiting**: Enterprise-grade rate limiting and abuse prevention
- **Security Headers**: Complete security header implementation

### Enterprise Features
- **High Availability**: Robust error handling and recovery mechanisms
- **Performance Monitoring**: Real-time metrics and health checks
- **Scalability**: Concurrent processing and load balancing support
- **Error Recovery**: Graceful degradation and error handling

## 📋 API Endpoints

### 1. Core Inference Endpoints (`/api/v1/inference/`)

#### Single Inference
```http
POST /api/v1/inference/single
```

Processes individual medical queries with comprehensive validation.

**Request Body:**
```json
{
  "query": "Patient presents with chest pain and shortness of breath",
  "context": "45-year-old male with history of hypertension",
  "patient_id": "patient_12345",
  "medical_domain": "cardiology",
  "urgency_level": "high",
  "require_medical_validation": true,
  "enable_phi_protection": true
}
```

**Response:**
```json
{
  "request_id": "uuid-string",
  "response": "Based on the symptoms described...",
  "confidence": 0.85,
  "medical_context": {
    "diagnosis_suggested": "possible_cardiac_event",
    "severity": "high",
    "urgency": "immediate_attention_required"
  },
  "medical_validation_passed": true,
  "phi_protection_applied": true,
  "clinical_recommendations": [
    "Immediate cardiac evaluation required",
    "ECG and cardiac enzymes recommended"
  ],
  "risk_assessment": {
    "level": "high",
    "factors": ["chest pain", "shortness of breath", "risk factors"],
    "requires_immediate_attention": true
  }
}
```

#### Batch Inference
```http
POST /api/v1/inference/batch
```

Process multiple medical queries simultaneously for efficiency.

**Request Body:**
```json
{
  "queries": [
    {
      "query": "Patient with diabetes reporting frequent urination",
      "patient_id": "patient_123",
      "medical_domain": "endocrinology"
    },
    {
      "query": "Headache with visual disturbances",
      "patient_id": "patient_456", 
      "medical_domain": "neurology"
    }
  ]
}
```

### 2. Streaming Endpoints (`/api/v1/streaming/`)

#### WebSocket Chat
```http
WS /api/v1/streaming/chat/{session_id}
```

Real-time medical chat with streaming responses.

**Features:**
- Medical context preservation
- PHI detection and protection
- Clinical decision support
- Emergency escalation protocols

**Message Format:**
```json
{
  "type": "user_message",
  "content": "I'm experiencing chest pain and shortness of breath",
  "medical_domain": "cardiology",
  "urgency_level": "high"
}
```

#### HTTP Streaming
```http
POST /api/v1/streaming/chat/http-stream
```

Server-sent events for streaming responses (WebSocket alternative).

### 3. Batch Processing Endpoints (`/api/v1/batch/`)

#### Create Batch Job
```http
POST /api/v1/batch/create
```

Create batch processing job for multiple patients.

**Features:**
- Parallel processing for efficiency
- Progress tracking
- Error handling and recovery
- Performance optimization

### 4. Health Check Endpoints (`/api/v1/health/`)

#### System Health
```http
GET /api/v1/health/system
```

Comprehensive system health assessment.

#### Detailed Health Check
```http
GET /api/v1/health/detailed
```

Detailed component breakdown with recommendations.

#### Kubernetes Probes
```http
GET /api/v1/health/ready
GET /api/v1/health/live
```

Kubernetes readiness and liveness probes.

#### Prometheus Metrics
```http
GET /api/v1/health/metrics
```

Prometheus-compatible metrics for monitoring.

### 5. Conversation Management (`/api/v1/conversation/`)

#### Create Conversation
```http
POST /api/v1/conversation/create
```

Initialize persistent medical conversation session.

#### Send Message
```http
POST /api/v1/conversation/{conversation_id}/message
```

Send message in conversation with medical context processing.

#### Get Context
```http
GET /api/v1/conversation/{conversation_id}/context
```

Retrieve complete conversation context and state.

### 6. Clinical Decision Support (`/api/v1/clinical/`)

#### Make Clinical Decision
```http
POST /api/v1/clinical/decide
```

Evidence-based clinical decision support.

**Request Body:**
```json
{
  "patient_data": {
    "age": 55,
    "gender": "male",
    "vital_signs": {
      "blood_pressure": {"systolic": 160, "diastolic": 95},
      "heart_rate": 88
    }
  },
  "symptoms": ["chest pain", "shortness of breath", "sweating"],
  "medical_history": ["hypertension", "diabetes"],
  "current_medications": ["metformin", "lisinopril"],
  "decision_type": "diagnosis_suggestion"
}
```

#### Batch Clinical Decisions
```http
POST /api/v1/clinical/decide/batch
```

Multiple clinical decisions for efficiency.

### 7. Medical Validation (`/api/v1/validation/`)

#### Comprehensive Validation
```http
POST /api/v1/validation/validate
```

Medical data validation with multiple validation types.

**Request Body:**
```json
{
  "data": {
    "vital_signs": {
      "blood_pressure": {"systolic": 140, "diastolic": 90},
      "heart_rate": 75
    },
    "medications": [
      {"name": "metformin", "dosage": 500}
    ]
  },
  "validation_types": ["medical_accuracy", "phi_detection", "hipaa_compliance"],
  "strict_mode": true
}
```

#### PHI Analysis
```http
POST /api/v1/validation/phi/analyze
```

PHI detection and protection analysis.

#### Medical Accuracy
```http
POST /api/v1/validation/accuracy/validate
```

Medical accuracy validation against clinical guidelines.

## 🔧 Configuration

### Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
WORKERS=4

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/medical_ai
REDIS_URL=redis://localhost:6379/0

# Model Configuration
MODEL_NAME=medical-ai-assistant
MODEL_VERSION=1.0.0
MAX_TOKENS=2048
TEMPERATURE=0.7

# Medical Validation
ENABLE_MEDICAL_VALIDATION=true
STRICT_MODE=false

# PHI Protection
ENABLE_PHI_DETECTION=true
PHI_REDACTION=true
PHI_MODES=["mask","anonymize"]

# Rate Limiting
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=3600

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

## 🛡️ Security Features

### PHI Protection
- **Automatic Detection**: Identifies 18 types of HIPAA-protected health information
- **Redaction Modes**: Mask, anonymize, or hash PHI data
- **Risk Assessment**: Automatic risk level calculation (low/medium/high/critical)
- **Compliance Validation**: HIPAA compliance checking

### Medical Validation
- **Terminology Validation**: Medical term accuracy checking
- **Measurement Validation**: Vital signs and lab value validation
- **Dosage Validation**: Medication dosage safety checking
- **Contraindication Detection**: Drug interaction and contraindication alerts

### Audit Logging
- **Comprehensive Tracking**: All medical operations logged
- **Integrity Protection**: Audit hash for tamper detection
- **Compliance Reporting**: HIPAA compliance metrics and reporting
- **Query Capability**: Advanced audit log querying with filters

## 📊 Performance Features

### Streaming Support
- **Real-time Responses**: Server-sent events and WebSocket support
- **Chunked Generation**: Progressive response streaming
- **Context Preservation**: Maintains conversation context
- **Error Recovery**: Graceful handling of streaming interruptions

### Batch Processing
- **Parallel Processing**: Concurrent processing of multiple requests
- **Progress Tracking**: Real-time batch progress monitoring
- **Error Isolation**: Individual item failure handling
- **Resource Management**: Configurable concurrency limits

### Caching
- **Intelligent Caching**: Context-aware response caching
- **Cache Invalidation**: Automatic cache refresh based on medical context
- **Performance Metrics**: Cache hit rate monitoring

## 🏥 Medical Domain Support

### Specialties
- **Cardiology**: Heart disease, blood pressure, cardiac emergencies
- **Neurology**: Headaches, seizures, neurological disorders
- **Oncology**: Cancer screening, treatment side effects
- **Emergency Medicine**: Triage, critical care, emergency protocols
- **Pediatrics**: Child-specific medical considerations
- **Endocrinology**: Diabetes, thyroid disorders, hormone conditions

### Clinical Decision Support
- **Evidence-based Recommendations**: Guidelines-compliant suggestions
- **Risk Assessment**: Automatic risk stratification
- **Medication Safety**: Drug interaction checking
- **Clinical Protocols**: Emergency and routine care protocols

## 🔍 Monitoring & Observability

### Health Checks
- **System Health**: Overall system status and metrics
- **Component Health**: Individual service health status
- **Performance Metrics**: Response times, throughput, error rates
- **Resource Usage**: CPU, memory, disk, network utilization

### Audit & Compliance
- **Compliance Dashboard**: Real-time compliance status
- **Risk Assessment**: Automated risk level calculation
- **Audit Reports**: Detailed audit trail analysis
- **Regulatory Reporting**: HIPAA compliance reports

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t medical-ai-inference .

# Run container
docker run -p 8000:8000 \
  -e SECRET_KEY=your-secret \
  -e DATABASE_URL=postgresql://... \
  medical-ai-inference
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: medical-ai-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: medical-ai-inference
  template:
    metadata:
      labels:
        app: medical-ai-inference
    spec:
      containers:
      - name: medical-ai-inference
        image: medical-ai-inference:latest
        ports:
        - containerPort: 8000
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: medical-ai-secrets
              key: secret-key
```

## 📈 Usage Examples

### Basic Medical Query
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/inference/single",
    headers={"Authorization": "Bearer <token>"},
    json={
        "query": "Patient with diabetes reporting frequent urination and thirst",
        "medical_domain": "endocrinology",
        "urgency_level": "medium"
    }
)

result = response.json()
print(f"Response: {result['response']}")
print(f"Confidence: {result['confidence']}")
print(f"Recommendations: {result['clinical_recommendations']}")
```

### Streaming Chat
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/streaming/chat/session-123');

ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'user_message',
        content: 'I have chest pain and feel short of breath',
        medical_domain: 'cardiology',
        urgency_level: 'high'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'response_chunk') {
        console.log('Received chunk:', data.content);
    }
};
```

### Clinical Decision Support
```python
decision_request = {
    "patient_data": {
        "age": 60,
        "gender": "male",
        "vital_signs": {
            "blood_pressure": {"systolic": 160, "diastolic": 100}
        }
    },
    "symptoms": ["chest pain", "shortness of breath"],
    "medical_history": ["diabetes", "hypertension"],
    "decision_type": "risk_assessment"
}

response = requests.post(
    "http://localhost:8000/api/v1/clinical/decide",
    headers={"Authorization": "Bearer <token>"},
    json=decision_request
)

result = response.json()
print(f"Risk Level: {result['risk_assessment']['risk_level']}")
print(f"Recommendations: {result['recommendations']}")
```

## ⚠️ Important Considerations

### Medical Disclaimer
This API provides informational medical assistance and should **never** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

### Emergency Situations
For medical emergencies, call emergency services immediately (911 in the US). This API should not be used for emergency medical situations.

### Data Privacy
- All PHI is automatically protected and redacted
- Audit logs maintain compliance with healthcare regulations
- Data retention policies must be configured according to organizational requirements

### Performance Limits
- Rate limiting is enforced to prevent abuse
- Batch processing has configurable limits
- Resource usage is monitored and automatically managed

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SECRET_KEY="development-secret-key"
export DEBUG=true

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run unit tests
pytest tests/

# Run integration tests
pytest tests/integration/

# Run security tests
pytest tests/security/
```

### Code Quality
- All code follows PEP 8 style guidelines
- Comprehensive type hints throughout
- Extensive error handling and logging
- Security-first approach to all features

## 📞 Support

For technical support or questions about this API:
- Review the comprehensive documentation above
- Check the health check endpoints for system status
- Examine audit logs for compliance and security events
- Monitor performance metrics for optimization opportunities

This API is designed to be enterprise-ready with comprehensive medical data protection, HIPAA compliance, and high availability for healthcare applications.