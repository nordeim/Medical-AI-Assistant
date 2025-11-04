# API Documentation - Medical AI Assistant

## Overview

This directory contains comprehensive API documentation for the Medical AI Assistant system, including OpenAPI/Swagger specifications, SDK documentation, and integration guides for third-party developers.

## 📁 Directory Structure

```
api/
├── openapi/                   # OpenAPI/Swagger specifications
│   ├── v1.yaml               # Main API specification v1.0
│   ├── components/           # Reusable components
│   │   ├── schemas.yaml     # Data models
│   │   ├── security.yaml    # Security definitions
│   │   └── parameters.yaml  # Reusable parameters
│   ├── paths/               # API endpoints
│   │   ├── sessions.yaml    # Session management endpoints
│   │   ├── chat.yaml        # Chat and AI interaction endpoints
│   │   ├── assessments.yaml # Assessment management endpoints
│   │   └── admin.yaml       # Administrative endpoints
│   └── examples/            # Request/response examples
│       ├── session-create.json
│       ├── chat-request.json
│       └── assessment-response.json
│
├── sdks/                     # Software Development Kits
│   ├── python/              # Python SDK
│   │   ├── medical_ai_sdk/
│   │   ├── setup.py
│   │   ├── requirements.txt
│   │   └── README.md
│   ├── javascript/          # JavaScript SDK
│   │   ├── package.json
│   │   ├── src/
│   │   └── examples/
│   └── curl/                # cURL examples
│       ├── authentication.sh
│       ├── create-session.sh
│       └── send-message.sh
│
├── integration/              # Integration guides
│   ├── ehr-integration.md
│   ├── webhook-setup.md
│   ├── authentication-guide.md
│   └── rate-limits.md
│
└── changelog/               # API version history
    ├── v1.0.0.md
    └── v1.1.0.md
```

## 🚀 Quick Start

### Base URL
```
Production: https://api.medical-ai.example.com/v1
Staging: https://api-staging.medical-ai.example.com/v1
```

### Authentication
All API requests require authentication using API keys or JWT tokens.

**API Key Authentication:**
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     https://api.medical-ai.example.com/v1/health
```

**JWT Token Authentication:**
```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     https://api.medical-ai.example.com/v1/health
```

### Basic Request Structure
```json
{
  "request_id": "uuid-string",
  "timestamp": "2025-11-04T09:47:35Z",
  "source": "client-application",
  "data": {
    // Request-specific data
  }
}
```

## 📖 API Endpoints Summary

### Session Management
- **POST** `/sessions` - Create new consultation session
- **GET** `/sessions/{session_id}` - Retrieve session details
- **DELETE** `/sessions/{session_id}` - End session
- **POST** `/sessions/{session_id}/renew` - Renew session token

### AI Chat Interface
- **POST** `/sessions/{session_id}/messages` - Send message to AI
- **GET** `/sessions/{session_id}/messages` - Retrieve conversation history
- **WebSocket** `/ws/chat/{session_id}` - Real-time chat interface

### Assessment Management
- **GET** `/assessments` - List assessments (filtered)
- **GET** `/assessments/{assessment_id}` - Get specific assessment
- **PUT** `/assessments/{assessment_id}` - Update assessment
- **POST** `/assessments/{assessment_id}/actions` - Take action on assessment

### Healthcare Professional API
- **GET** `/nurse/queue` - Get pending assessments queue
- **POST** `/nurse/queue/{assessment_id}/review` - Review assessment
- **GET** `/nurse/dashboard/stats` - Dashboard statistics

### Administrative API
- **GET** `/admin/system/health` - System health status
- **GET** `/admin/audit/logs` - Audit log access
- **POST** `/admin/models/reload` - Reload AI models
- **GET** `/admin/metrics` - System metrics

## 🔐 Authentication & Security

### API Key Management
```bash
# Generate new API key
curl -X POST https://api.medical-ai.example.com/v1/auth/api-keys \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Integration Key",
    "permissions": ["read", "write"],
    "expires_at": "2026-11-04T09:47:35Z"
  }'
```

### JWT Token Flow
```javascript
// Obtain JWT token
const response = await fetch('/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    username: 'username',
    password: 'password'
  })
});

const { access_token, token_type } = await response.json();

// Use token for API requests
const apiResponse = await fetch('/v1/sessions', {
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  }
});
```

### Rate Limiting
- **Standard Tier**: 100 requests/minute
- **Premium Tier**: 1000 requests/minute
- **Enterprise Tier**: 10000 requests/minute
- **Rate limit headers**: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

### Security Headers
All responses include security headers:
```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Content-Security-Policy: default-src 'self'
```

## 📊 Response Formats

### Success Response
```json
{
  "status": "success",
  "data": {
    // Response data
  },
  "meta": {
    "request_id": "uuid-string",
    "timestamp": "2025-11-04T09:47:35Z",
    "version": "1.0.0"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "session_id",
      "issue": "Session ID format is invalid"
    }
  },
  "meta": {
    "request_id": "uuid-string",
    "timestamp": "2025-11-04T09:47:35Z"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

## 🔧 Error Handling

### Common Error Codes

#### Authentication Errors
- `AUTH_001` - Invalid API key
- `AUTH_002` - Token expired
- `AUTH_003` - Insufficient permissions
- `AUTH_004` - Account locked

#### Validation Errors
- `VAL_001` - Required field missing
- `VAL_002` - Invalid data format
- `VAL_003` - Value out of range
- `VAL_004` - Duplicate value

#### System Errors
- `SYS_001` - Service temporarily unavailable
- `SYS_002` - Database connection error
- `SYS_003` - AI model service error
- `SYS_004` - Rate limit exceeded

### Error Recovery Strategies
```javascript
class MedicalAIAPIClient {
  async makeRequest(endpoint, options) {
    try {
      const response = await fetch(endpoint, options);
      
      if (response.status === 429) {
        // Rate limited - implement exponential backoff
        const retryAfter = response.headers.get('Retry-After');
        await this.delay(parseInt(retryAfter) * 1000);
        return this.makeRequest(endpoint, options);
      }
      
      if (response.status >= 500) {
        // Server error - retry with backoff
        await this.delay(1000);
        return this.makeRequest(endpoint, options);
      }
      
      return response;
    } catch (error) {
      // Network error - retry with backoff
      await this.delay(1000);
      return this.makeRequest(endpoint, options);
    }
  }
  
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
```

## 📚 SDK Usage

### Python SDK
```python
from medical_ai_sdk import MedicalAIClient

# Initialize client
client = MedicalAIClient(
    api_key="your-api-key",
    base_url="https://api.medical-ai.example.com/v1"
)

# Create session
session = client.sessions.create(
    patient_age_group="ADULT",
    chief_complaint="Headache"
)

# Send message
response = client.chat.send_message(
    session_id=session.id,
    message="I've had a headache for 2 days"
)

# Get assessment
assessment = client.assessments.get(session.assessment_id)
print(f"Triage Level: {assessment.triage_level}")
```

### JavaScript SDK
```javascript
import { MedicalAIClient } from '@medical-ai/sdk';

// Initialize client
const client = new MedicalAIClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.medical-ai.example.com/v1'
});

// Create session
const session = await client.sessions.create({
  patientAgeGroup: 'ADULT',
  chiefComplaint: 'Chest pain'
});

// Send message
const response = await client.chat.sendMessage({
  sessionId: session.id,
  message: 'I have chest pain when climbing stairs'
});

// Handle assessment
const assessment = await client.assessments.get(session.assessmentId);
console.log(`Triage Level: ${assessment.triageLevel}`);
```

## 🔌 Integration Examples

### EHR System Integration
```python
# Integration with Electronic Health Record systems
class EHRIntegration:
    def __init__(self, medical_ai_client):
        self.client = medical_ai_client
    
    async def create_consultation_from_ehr(self, ehr_patient_data):
        """Create AI consultation from EHR patient data"""
        
        # Map EHR data to AI consultation format
        consultation_data = {
            "patient_age_group": self.map_age_group(ehr_patient_data['age']),
            "chief_complaint": ehr_patient_data['chief_complaint'],
            "medical_history": ehr_patient_data['medical_history'],
            "current_medications": ehr_patient_data['medications']
        }
        
        # Create consultation session
        session = await self.client.sessions.create(consultation_data)
        
        # Optionally pre-populate conversation
        if ehr_patient_data.get('symptoms'):
            for symptom in ehr_patient_data['symptoms']:
                await self.client.chat.send_message(
                    session_id=session.id,
                    message=f"Symptom: {symptom['description']}"
                )
        
        return session
    
    def map_age_group(self, age):
        if age < 18:
            return "PEDIATRIC"
        elif age < 65:
            return "ADULT"
        else:
            return "GERIATRIC"
```

### Webhook Integration
```javascript
// Webhook setup for real-time notifications
const webhookClient = new MedicalAIWebhookClient({
  webhookUrl: 'https://your-system.com/webhooks/medical-ai',
  secret: 'your-webhook-secret'
});

// Configure webhook events
await webhookClient.configure({
  events: [
    'assessment.completed',
    'assessment.escalated',
    'session.ended',
    'emergency.detected'
  ]
});

// Handle webhook payload
app.post('/webhooks/medical-ai', (req, res) => {
  const signature = req.headers['x-medical-ai-signature'];
  
  if (webhookClient.verifySignature(req.body, signature)) {
    const event = req.body;
    
    switch (event.type) {
      case 'assessment.completed':
        handleAssessmentCompleted(event.data);
        break;
      case 'emergency.detected':
        handleEmergency(event.data);
        break;
      default:
        console.log('Unhandled event:', event.type);
    }
  }
  
  res.status(200).send('OK');
});
```

## 📈 Monitoring and Analytics

### Request Analytics
```python
# Request tracking and analytics
class RequestAnalytics:
    def __init__(self, client):
        self.client = client
    
    async def get_usage_metrics(self, start_date, end_date):
        """Get usage metrics for analysis"""
        
        metrics = await self.client.admin.metrics.get({
            'start_date': start_date,
            'end_date': end_date,
            'granularity': 'hour'
        })
        
        return {
            'total_requests': metrics.total_requests,
            'success_rate': metrics.success_rate,
            'average_response_time': metrics.avg_response_time,
            'error_breakdown': metrics.errors_by_code
        }
    
    async def track_performance(self, endpoint, duration, status_code):
        """Track individual request performance"""
        
        await self.client.admin.metrics.track_request({
            'endpoint': endpoint,
            'duration_ms': duration,
            'status_code': status_code,
            'timestamp': datetime.utcnow()
        })
```

## 🧪 Testing

### API Testing Suite
```python
# Comprehensive API testing
import pytest
from medical_ai_sdk import MedicalAIClient

class TestMedicalAIAPI:
    @pytest.fixture
    def client(self):
        return MedicalAIClient(
            api_key="test-api-key",
            base_url="https://api-test.medical-ai.example.com/v1"
        )
    
    async def test_create_session(self, client):
        """Test session creation"""
        session = await client.sessions.create({
            "patient_age_group": "ADULT",
            "chief_complaint": "Test complaint"
        })
        
        assert session.id is not None
        assert session.status == "active"
    
    async def test_chat_interaction(self, client):
        """Test chat functionality"""
        session = await client.sessions.create({
            "patient_age_group": "ADULT",
            "chief_complaint": "Test"
        })
        
        response = await client.chat.send_message({
            "session_id": session.id,
            "message": "Hello, I have a headache"
        })
        
        assert response.message is not None
        assert len(response.message) > 0
    
    async def test_assessment_retrieval(self, client):
        """Test assessment retrieval"""
        assessments = await client.assessments.list({
            'limit': 10,
            'status': 'completed'
        })
        
        assert len(assessments) <= 10
        for assessment in assessments:
            assert assessment.id is not None
```

## 📞 Support and Resources

### Developer Support
- **Documentation**: [https://docs.medical-ai.example.com](https://docs.medical-ai.example.com)
- **API Reference**: [https://api.medical-ai.example.com/docs](https://api.medical-ai.example.com/docs)
- **Community Forum**: [https://community.medical-ai.example.com](https://community.medical-ai.example.com)
- **GitHub**: [https://github.com/medical-ai/api-examples](https://github.com/medical-ai/api-examples)

### Technical Support
- **Email**: api-support@medical-ai.example.com
- **Status Page**: [https://status.medical-ai.example.com](https://status.medical-ai.example.com)
- **Slack**: #medical-ai-developers
- **Rate Limit Issues**: api-rates@medical-ai.example.com

### Professional Services
- **Integration Consulting**: Available for complex integrations
- **Custom Development**: Custom API clients and integrations
- **Training**: API training and best practices workshops
- **Support SLAs**: Enterprise support with guaranteed response times

---

**Remember: All API usage must comply with healthcare regulations including HIPAA. Ensure proper data handling, storage, and transmission protocols are followed.**

*For API access and integration support, contact the Medical AI Assistant development team.*

**Version**: 1.0 | **Last Updated**: November 2025 | **Next Review**: February 2026
