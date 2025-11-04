# API Documentation - Medical AI Serving System

## Overview

Complete API reference for the Medical AI Serving System with comprehensive medical compliance examples, security protocols, and clinical validation workflows.

## Base URL

```
Production: https://api.medical-ai.example.com
Staging: https://staging-api.medical-ai.example.com
Development: http://localhost:8000
```

## Authentication

All API endpoints require authentication with medical-grade security:

### API Key Authentication
```http
Authorization: Bearer <your_api_key>
X-Client-ID: <client_identifier>
X-Session-ID: <session_identifier>
```

### JWT Token Authentication
```http
Authorization: Bearer <jwt_token>
X-User-ID: <user_id>
X-Medical-Domain: <specialty>
```

## Core Endpoints

### 1. Health Check

#### GET /health
Returns system health status for medical compliance monitoring.

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2025-11-04T07:41:57Z",
  "version": "6.0.0",
  "compliance_status": {
    "hipaa_compliant": true,
    "fda_approved": false,
    "clinical_validation": "in_progress",
    "audit_logging": "enabled"
  },
  "services": {
    "model_service": "healthy",
    "database": "healthy",
    "cache": "healthy",
    "monitoring": "healthy"
  },
  "performance_metrics": {
    "avg_response_time_ms": 245,
    "requests_per_minute": 1200,
    "error_rate": 0.02
  }
}
```

**Status Codes:**
- `200`: System is healthy
- `503`: System is degraded
- `500`: System is unhealthy

### 2. Medical Inference

#### POST /api/v1/inference/single
Single medical inference with comprehensive validation and PHI protection.

**Request Headers:**
```http
Content-Type: application/json
Authorization: Bearer <token>
X-Medical-Compliance: enabled
```

**Request Body:**
```json
{
  "query": "Patient presents with acute chest pain, 7/10 intensity, radiating to left arm",
  "context": "62-year-old male, history of hypertension, family history of heart disease",
  "patient_id": "anonymized_12345",
  "session_id": "session_67890",
  "model_name": "medical-diagnosis-v2.1",
  "temperature": 0.3,
  "max_tokens": 500,
  "stream": false,
  "medical_domain": "cardiology",
  "urgency_level": "high",
  "require_medical_validation": true,
  "enable_phi_protection": true
}
```

**Validation Rules:**
- `query`: 1-5000 characters, medical terminology allowed
- `patient_id`: Alphanumeric, max 50 chars
- `medical_domain`: One of: general, cardiology, oncology, neurology, emergency, pediatrics
- `urgency_level`: One of: low, medium, high, critical

**Response:**
```json
{
  "request_id": "uuid-12345678-1234-1234-1234-123456789012",
  "response": "Based on the presenting symptoms of acute chest pain radiating to the left arm, this could indicate acute coronary syndrome. Given the patient's risk factors (age, hypertension, family history), this requires immediate evaluation...",
  "confidence": 0.87,
  "medical_context": {
    "primary_topic": "cardiology",
    "urgency": "high",
    "has_context": true,
    "has_patient_id": true,
    "query_length": 142,
    "medical_terms_detected": ["chest pain", "hypertension"]
  },
  "processing_time": 1.245,
  "model_version": "6.0.0",
  "timestamp": "2025-11-04T07:41:57Z",
  "medical_validation_passed": true,
  "phi_protection_applied": true,
  "clinical_recommendations": [
    "Seek immediate medical attention at emergency department",
    "Consider ECG and cardiac biomarkers",
    "Monitor for additional symptoms",
    "Do not delay evaluation due to high-risk presentation"
  ],
  "risk_assessment": {
    "level": "high",
    "factors": ["High urgency level", "Emergency symptom: chest pain"],
    "confidence": 0.89,
    "requires_immediate_attention": true
  },
  "tokens_used": 156,
  "cache_hit": false
}
```

**Medical Compliance Features:**
- Automatic PHI detection and redaction
- Clinical decision support validation
- Emergency symptom escalation
- Medical terminology validation
- Compliance audit logging

**Status Codes:**
- `200`: Successful inference
- `400`: Invalid medical input
- `422`: Medical validation failure
- `429`: Rate limit exceeded
- `500`: Model service unavailable

### 3. Batch Inference

#### POST /api/v1/inference/batch
Batch processing for multiple medical queries with parallel validation.

**Request Body:**
```json
{
  "queries": [
    {
      "query": "Patient with fever and headache for 2 days",
      "medical_domain": "general",
      "urgency_level": "medium",
      "patient_id": "anon_001"
    },
    {
      "query": "Sudden onset weakness on right side",
      "medical_domain": "neurology",
      "urgency_level": "critical",
      "patient_id": "anon_002"
    }
  ],
  "batch_id": "batch_20251104_001"
}
```

**Response:**
```json
{
  "batch_id": "batch_20251104_001",
  "results": [
    {
      "request_id": "uuid-1",
      "response": "Fever and headache combination may indicate...",
      "confidence": 0.82,
      "processing_time": 0.987,
      "clinical_recommendations": ["Schedule appointment within 24-48 hours"],
      "risk_assessment": {"level": "medium", "confidence": 0.82}
    },
    {
      "request_id": "uuid-2",
      "response": "Sudden unilateral weakness is concerning for stroke...",
      "confidence": 0.94,
      "processing_time": 0.734,
      "clinical_recommendations": ["Seek immediate emergency care", "Consider stroke protocol"],
      "risk_assessment": {"level": "critical", "confidence": 0.94}
    }
  ],
  "total_processing_time": 2.156,
  "successful_count": 2,
  "failed_count": 0,
  "timestamp": "2025-11-04T07:41:57Z"
}
```

**Batch Processing Features:**
- Parallel validation for each query
- Individual risk assessment
- Batch-level error handling
- Performance optimization
- Compliance audit for batch

### 4. Clinical Decision Support

#### POST /api/v1/clinical-decision-support/analyze
Specialized clinical decision support with evidence-based recommendations.

**Request Body:**
```json
{
  "patient_data": {
    "age": 65,
    "gender": "male",
    "chief_complaint": "Shortness of breath and fatigue",
    "vital_signs": {
      "blood_pressure": "140/90",
      "heart_rate": 95,
      "temperature": 98.6,
      "oxygen_saturation": 94
    },
    "medical_history": ["diabetes", "hypertension", "hyperlipidemia"],
    "medications": ["metformin", "lisinopril", "atorvastatin"]
  },
  "support_type": "differential_diagnosis",
  "clinical_context": {
    "duration": "3 days",
    "severity": "progressive",
    "trigger_factors": "exertion"
  }
}
```

**Response:**
```json
{
  "analysis_id": "clinical_12345",
  "differential_diagnosis": [
    {
      "condition": "Congestive Heart Failure",
      "probability": 0.75,
      "evidence": ["dyspnea", "fatigue", "elevated BP", "risk factors"],
      "recommendations": ["ECG", "BNP", "chest X-ray", "echocardiogram"]
    },
    {
      "condition": "Chronic Obstructive Pulmonary Disease",
      "probability": 0.45,
      "evidence": ["dyspnea", "oxygen desaturation"],
      "recommendations": ["pulmonary function tests", "chest CT"]
    }
  ],
  "evidence_summary": {
    "supporting_evidence": 4,
    "contradicting_evidence": 0,
    "missing_information": ["smoking history", "exercise tolerance"]
  },
  "clinical_recommendations": {
    "immediate": ["ECG and cardiac biomarkers", "Chest X-ray"],
    "short_term": ["Echocardiogram", "Pulmonary function tests"],
    "monitoring": ["Daily weights", "Oxygen saturation monitoring"]
  },
  "risk_stratification": {
    "risk_level": "moderate_to_high",
    "factors": ["comorbidities", "progressive symptoms", "multiple organ systems"],
    "disposition": "Hospital evaluation recommended"
  },
  "clinical_alerts": [
    "High-risk patient requiring prompt evaluation",
    "Consider cardiac consultation if symptoms worsen"
  ]
}
```

### 5. Conversation Management

#### POST /api/v1/conversation/analyze
Analyze medical conversation with sentiment analysis and clinical insights.

**Request Body:**
```json
{
  "conversation_id": "conv_98765",
  "messages": [
    {
      "role": "patient",
      "content": "I've been feeling really anxious about my test results",
      "timestamp": "2025-11-04T07:30:00Z"
    },
    {
      "role": "provider",
      "content": "I understand your concern. Let's review your results together.",
      "timestamp": "2025-11-04T07:30:15Z"
    },
    {
      "role": "patient",
      "content": "The waiting is the worst part. What if it's cancer?",
      "timestamp": "2025-11-04T07:31:00Z"
    }
  ],
  "analysis_type": ["sentiment", "clinical_concerns", "communication_quality"],
  "privacy_level": "anonymized"
}
```

**Response:**
```json
{
  "conversation_id": "conv_98765",
  "analysis_timestamp": "2025-11-04T07:41:57Z",
  "sentiment_analysis": {
    "patient_sentiment": {
      "overall": "anxious",
      "anxiety_level": 0.75,
      "confidence_level": 0.65,
      "emotional_state": "concerned_about_results"
    },
    "provider_sentiment": {
      "overall": "supportive",
      "empathy_score": 0.85,
      "communication_quality": "high"
    }
  },
  "clinical_concerns": [
    {
      "concern": "anxiety_about_results",
      "severity": "moderate",
      "clinical_relevance": "high",
      "recommendations": ["Provide clear explanation", "Offer anxiety management resources"]
    }
  ],
  "communication_quality": {
    "provider_empathy": 0.85,
    "patient_engagement": 0.70,
    "information_clarity": 0.90,
    "overall_quality": "good"
  },
  "clinical_alerts": [
    "Patient experiencing significant anxiety about diagnostic results",
    "Consider additional support resources for test result anxiety"
  ]
}
```

### 6. Streaming Inference

#### POST /api/v1/inference/stream
Real-time streaming inference for interactive medical consultations.

**Request Body:**
```json
{
  "query": "Tell me about the side effects of this medication",
  "patient_id": "anon_123",
  "stream": true,
  "chunk_size": 512,
  "medical_domain": "general",
  "interactive_mode": true
}
```

**Streaming Response (Server-Sent Events):**
```http
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"type": "start", "request_id": "stream_123"}

data: {"type": "chunk", "content": "Based on clinical trials, the most common side effects include:", "chunk_index": 0}

data: {"type": "chunk", "content": "1. Gastrointestinal effects: nausea, stomach upset, diarrhea", "chunk_index": 1}

data: {"type": "chunk", "content": "2. Neurological effects: headache, dizziness, fatigue", "chunk_index": 2}

data: {"type": "chunk", "content": "3. Cardiovascular: changes in blood pressure, heart rate", "chunk_index": 3}

data: {"type": "medical_validation", "passed": true, "phi_protected": true}

data: {"type": "end", "final_confidence": 0.89, "total_chunks": 15}
```

### 7. Model Status & Health

#### GET /api/v1/models/status
Get detailed model status and health information for medical compliance.

**Response:**
```json
{
  "model_name": "medical-diagnosis-v2.1",
  "version": "6.0.0",
  "status": "healthy|degraded|unhealthy",
  "last_health_check": "2025-11-04T07:41:00Z",
  "uptime_seconds": 86400,
  "medical_metrics": {
    "accuracy": 0.94,
    "sensitivity": 0.92,
    "specificity": 0.96,
    "clinical_validation_status": "validated",
    "regulatory_approval": "fda_pending"
  },
  "performance_metrics": {
    "avg_response_time_ms": 245,
    "p95_response_time_ms": 450,
    "requests_per_hour": 3600,
    "cache_hit_rate": 0.67,
    "error_rate": 0.02
  },
  "resource_utilization": {
    "memory_usage_mb": 2048,
    "gpu_utilization_percent": 65.2,
    "cpu_utilization_percent": 34.1,
    "disk_usage_percent": 23.4
  },
  "compliance_status": {
    "hipaa_compliant": true,
    "audit_logging_enabled": true,
    "phi_protection_active": true,
    "clinical_validation_current": true
  }
}
```

## Error Handling

### Medical-Specific Error Codes

| Error Code | Description | HTTP Status | Medical Context |
|------------|-------------|-------------|-----------------|
| `MED_001` | Invalid medical terminology | 422 | Input validation |
| `MED_002` | PHI detection failure | 422 | Security violation |
| `MED_003` | Clinical validation failed | 422 | Safety violation |
| `MED_004` | Emergency symptom escalation | 503 | Critical condition |
| `MED_005` | Model confidence too low | 422 | Reliability concern |
| `MED_006` | Regulatory compliance violation | 403 | Legal violation |

### Error Response Format
```json
{
  "error": {
    "code": "MED_001",
    "message": "Invalid medical terminology detected",
    "details": {
      "invalid_terms": ["unknown_medical_term"],
      "suggestion": "Please use standard medical terminology",
      "compliance_level": "warning"
    },
    "timestamp": "2025-11-04T07:41:57Z",
    "request_id": "uuid-12345678-1234-1234-1234-123456789012",
    "medical_context": {
      "domain": "cardiology",
      "urgency_level": "medium",
      "patient_id": "anon_123"
    }
  }
}
```

## Rate Limiting

### Medical Compliance Rate Limits

| Endpoint Category | Limit | Window | Rationale |
|------------------|-------|---------|-----------|
| Single Inference | 60 requests | Per minute | Prevent abuse, ensure availability |
| Batch Inference | 10 requests | Per minute | Resource protection |
| Clinical Decision Support | 30 requests | Per minute | Critical system access |
| Model Health Check | 300 requests | Per minute | System monitoring |

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 35
X-RateLimit-Reset: 1730700117
X-Medical-Rate-Limit-Context: emergency_medical_queries
```

## Security Requirements

### Mandatory Headers
```http
X-Client-ID: required_for_tracking
X-Session-ID: required_for_audit
X-Medical-Compliance: must_be_enabled
X-Request-Source: clinical|research|emergency
```

### PHI Protection
All endpoints implement automatic PHI detection and protection:
- Patient identifiers are anonymized
- Medical data is encrypted in transit and at rest
- Audit logs are maintained for compliance
- Access controls are role-based

### Audit Logging
Every request is logged with:
```json
{
  "audit_entry": {
    "timestamp": "2025-11-04T07:41:57Z",
    "request_id": "uuid-12345678-1234-1234-1234-123456789012",
    "user_id": "clinician_123",
    "client_id": "hospital_system_001",
    "endpoint": "/api/v1/inference/single",
    "method": "POST",
    "ip_address": "192.168.1.100",
    "user_agent": "MedicalApp/1.0",
    "response_status": 200,
    "processing_time_ms": 1245,
    "medical_domain": "cardiology",
    "phi_accessed": true,
    "compliance_level": "clinical_use"
  }
}
```

## SDK Examples

### Python SDK
```python
from medical_ai import MedicalAI, MedicalDomain, UrgencyLevel

# Initialize client
client = MedicalAI(
    api_key="your_api_key",
    base_url="https://api.medical-ai.example.com",
    medical_compliance=True
)

# Single inference
response = client.inference.single(
    query="Patient with chest pain and shortness of breath",
    medical_domain=MedicalDomain.CARDIOLOGY,
    urgency_level=UrgencyLevel.HIGH,
    patient_id="anon_123"
)

# Clinical decision support
analysis = client.clinical.analyze(
    patient_data={
        "age": 65,
        "symptoms": ["chest pain", "dyspnea"],
        "vital_signs": {"bp": "140/90", "hr": 95}
    },
    support_type="differential_diagnosis"
)
```

### JavaScript SDK
```javascript
import { MedicalAI, MedicalDomain } from '@medical-ai/sdk';

// Initialize client
const client = new MedicalAI({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.medical-ai.example.com',
  medicalCompliance: true
});

// Single inference
const response = await client.inference.single({
  query: 'Patient with acute chest pain',
  medicalDomain: MedicalDomain.CARDIOLOGY,
  urgencyLevel: 'high',
  patientId: 'anon_123'
});

// Batch inference
const batchResponse = await client.inference.batch({
  queries: [
    {
      query: 'Fever and headache',
      medicalDomain: 'general',
      urgencyLevel: 'medium'
    },
    {
      query: 'Sudden weakness',
      medicalDomain: 'neurology',
      urgencyLevel: 'critical'
    }
  ]
});
```

### cURL Examples

#### Basic Medical Inference
```bash
curl -X POST "https://api.medical-ai.example.com/api/v1/inference/single" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -H "X-Client-ID: hospital_system_001" \
  -H "X-Medical-Compliance: enabled" \
  -d '{
    "query": "Patient presents with acute chest pain and shortness of breath",
    "medical_domain": "cardiology",
    "urgency_level": "high",
    "patient_id": "anonymized_id_123",
    "require_medical_validation": true,
    "enable_phi_protection": true
  }'
```

#### Clinical Decision Support
```bash
curl -X POST "https://api.medical-ai.example.com/api/v1/clinical-decision-support/analyze" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -H "X-Request-Source: clinical" \
  -d '{
    "patient_data": {
      "age": 65,
      "chief_complaint": "Shortness of breath and fatigue",
      "vital_signs": {
        "blood_pressure": "140/90",
        "heart_rate": 95,
        "oxygen_saturation": 94
      }
    },
    "support_type": "differential_diagnosis"
  }'
```

## Testing & Validation

### Medical Compliance Testing
```bash
# Test PHI protection
curl -X POST "https://api.medical-ai.example.com/api/v1/inference/single" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"query": "Patient John Smith, DOB 01/15/1950, SSN 123-45-6789"}'

# Expected: PHI should be automatically redacted and logged
```

### Load Testing
```bash
# Medical-grade load testing
for i in {1..100}; do
  curl -X POST "https://api.medical-ai.example.com/api/v1/inference/single" \
    -H "Authorization: Bearer your_api_key" \
    -d "{\"query\": \"Test query $i\", \"medical_domain\": \"general\"}" &
done
```

## Versioning

### API Versioning
- Base path includes version: `/api/v1/`, `/api/v2/`
- Breaking changes require new major version
- Backward compatibility maintained for 2 major versions
- Deprecation notices provided 6 months in advance

### Model Versioning
- Each model has semantic version: `6.0.0`
- Model versions are tracked separately from API versions
- Rollback procedures are automated for safety
- Clinical validation is version-specific

## Support & Contact

### Medical Support
- **Clinical Issues**: clinical-support@medical-ai.example.com
- **Regulatory Questions**: regulatory@medical-ai.example.com
- **Emergency**: +1-XXX-XXX-XXXX (24/7)

### Technical Support
- **API Issues**: api-support@medical-ai.example.com
- **Integration**: integration@medical-ai.example.com
- **Documentation**: docs@medical-ai.example.com

---

**⚠️ Medical Disclaimer**: All API responses are for clinical decision support and must be validated by qualified medical professionals. Never rely solely on AI outputs for medical decisions without proper clinical validation and oversight.
