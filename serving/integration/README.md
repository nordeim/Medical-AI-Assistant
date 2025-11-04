# Medical AI Assistant Integration System

Comprehensive integration and frontend connection system for Phase 6 of the Medical AI Assistant project.

## Overview

This integration system provides seamless connectivity between the medical AI backend and frontend applications, with real-time capabilities, medical compliance, and robust error handling.

## Features

### 🔗 WebSocket Integration
- **Real-time medical chat** with AI conversation handling
- **Patient-nurse communication** with role-based access
- **Emergency red flag detection** and instant nurse alerts
- **Session management** with automatic cleanup
- **Rate limiting** and connection validation

### 📡 Streaming Responses (SSE)
- **Server-Sent Events** for real-time UI updates
- **Token-by-token streaming** for AI responses
- **Patient assessment progress** streaming
- **Nurse dashboard metrics** updates
- **Connection lifecycle** management

### 👩‍⚕️ Nurse Dashboard Integration
- **Real-time patient queue** monitoring
- **Priority-based sorting** with medical urgency
- **Patient assessment workflow** management
- **Action tracking** (approve/override/escalate)
- **Performance analytics** and reporting

### 🏊 Connection Pool Management
- **Multi-type connection pooling** (WebSocket, HTTP, Database, Redis, Model)
- **Medical priority queuing** for critical operations
- **Auto-scaling** based on load patterns
- **Health monitoring** and automatic recovery
- **Resource optimization** for medical workloads

### 🛡️ Medical-Grade CORS
- **HIPAA-compliant** cross-origin policies
- **Domain restrictions** for medical applications
- **Security headers** (HSTS, CSP, etc.)
- **Rate limiting** per origin
- **Emergency access** handling

### 📚 API Documentation
- **Comprehensive API docs** with examples
- **Medical compliance** examples and guides
- **Interactive testing** interfaces
- **Performance metrics** and monitoring
- **Swagger/OpenAPI** export

### 🧪 Testing Framework
- **Mock medical services** for isolated testing
- **Test suites** for compliance validation
- **Load testing** capabilities
- **Real-time monitoring** during tests
- **Comprehensive reporting**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Applications                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Patient Chat   │  │  Nurse Dashboard│  │  Admin Panel │ │
│  │    Interface    │  │    Interface    │  │   Interface  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │ WebSocket │ SSE │ REST API
┌─────────────────────────────────────────────────────────────┐
│                 Integration Layer                           │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────────┐ │
│  │ WebSocket     │ │ SSE Streaming│ │   Connection Pool   │ │
│  │ Manager       │ │ Manager      │ │     Manager         │ │
│  └───────────────┘ └──────────────┘ └─────────────────────┘ │
│  ┌───────────────┐ ┌──────────────┐ ┌─────────────────────┐ │
│  │ Medical CORS  │ │ Nurse API    │ │   Testing Engine    │ │
│  │ Configuration │ │ Endpoints    │ │   & Mock Services   │ │
│  └───────────────┘ └──────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                Medical AI Assistant Backend                 │
│         (FastAPI + ML Models + Database + Cache)            │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd /workspace/serving/integration
pip install -r requirements.txt
```

### 2. Integrate with Main Application

```python
# In your main FastAPI application
from serving.integration import (
    integration_router,
    integration_manager,
    create_medical_cors_middleware
)

# Add CORS middleware
app.add_middleware(MedicalCORSMiddleware)

# Include integration routes
app.include_router(integration_router)

# Initialize integration system
@app.on_event("startup")
async def startup_event():
    await integration_manager.initialize()
```

### 3. WebSocket Chat Integration

```javascript
// Frontend WebSocket connection
const ws = new WebSocket(
  `ws://localhost:8000/ws/chat?session_id=${sessionId}&user_type=patient`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  handleChatMessage(message);
};
```

### 4. SSE Streaming Integration

```javascript
// Frontend SSE connection for real-time updates
const eventSource = new EventSource(
  `/api/streaming/${streamId}`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);

eventSource.addEventListener('chat_token', (event) => {
  const tokenData = JSON.parse(event.data);
  updateChatResponse(tokenData);
});
```

### 5. Nurse Dashboard Integration

```javascript
// Frontend API calls for nurse dashboard
const response = await fetch('/api/nurse/queue?urgency=urgent', {
  headers: {
    'Authorization': `Bearer ${nurseToken}`,
    'X-Request-ID': generateRequestId()
  }
});

const queueData = await response.json();
updateNurseQueue(queueData);
```

## Configuration

### Environment Variables

```bash
# Server Configuration
APP_SERVING_HOST=0.0.0.0
APP_SERVING_PORT=8000
APP_SERVING_API_KEY=your_api_key_here

# Medical Compliance
APP_MEDICAL_ENABLE_AUDIT_LOG=true
APP_MEDICAL_PHI_REDACTION=true
APP_MEDICAL_ENABLE_ENCRYPTION=true

# CORS Configuration
APP_SERVING_ALLOWED_ORIGINS=["https://app.medical-ai.health", "https://dashboard.medical-ai.health"]

# Connection Pool
APP_CACHE_REDIS_URL=redis://localhost:6379/0
APP_CACHE_ENABLE_MODEL_CACHE=true
APP_CACHE_ENABLE_RESPONSE_CACHE=true
```

### CORS Domain Configuration

```python
# Add custom medical domains
cors_manager.add_temporary_origin(
    origin="https://emergency.medical-ai.health",
    duration_seconds=3600,
    allowed_paths=["/api/emergency/", "/ws/emergency/"]
)
```

## API Endpoints

### WebSocket Chat
- `GET /ws/chat` - Real-time medical chat WebSocket
- Query parameters: `session_id`, `user_type`, `token`

### Nurse Dashboard API
- `GET /api/nurse/queue` - Get patient queue with filtering
- `POST /api/nurse/queue/{patient_id}/action` - Take action on patient
- `GET /api/nurse/dashboard/metrics` - Get real-time metrics
- `GET /api/nurse/analytics` - Get analytics data

### Streaming API
- `GET /api/streaming/{stream_id}` - Server-Sent Events endpoint
- `POST /api/streaming/chat` - Create chat stream
- `POST /api/streaming/assessment` - Create assessment stream

### Documentation API
- `GET /docs/` - API documentation overview
- `GET /docs/compliance/examples` - Medical compliance examples
- `GET /docs/testing/scenarios` - Testing scenarios
- `GET /docs/export/swagger` - Export OpenAPI spec

### Testing API
- `GET /test/suites` - Get available test suites
- `POST /test/suites/{suite_id}/execute` - Execute test suite
- `GET /test/executions` - Get test execution results
- `POST /test/mock/chat` - Test mock chat service

## Testing

### Run Test Suites

```bash
# Execute all test suites
curl -X POST "http://localhost:8000/integration/test/suites/patient_chat_suite/execute" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Execute specific test
curl -X POST "http://localhost:8000/integration/test/suites/compliance_suite/execute" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Load Testing

```bash
# Run load test
curl -X POST "http://localhost:8000/integration/docs/test/load" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "concurrent_users": 100,
    "duration_minutes": 5,
    "ramp_up_time": 30
  }'
```

### Mock Services Testing

```bash
# Test mock chat service
curl -X POST "http://localhost:8000/integration/test/mock/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I have chest pain",
    "session_id": "test_session_001"
  }'

# Test mock queue service
curl -X POST "http://localhost:8000/integration/test/mock/queue" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "urgency": "urgent",
      "has_red_flags": true
    }
  }'
```

## Monitoring

### Health Checks

```bash
# System health
curl "http://localhost:8000/integration/health"

# Detailed health validation
curl "http://localhost:8000/integration/docs/health/detailed"

# Compliance metrics
curl "http://localhost:8000/integration/docs/metrics/compliance"
```

### Real-time Monitoring

```javascript
// WebSocket monitoring endpoint
const monitorWS = new WebSocket('ws://localhost:8000/integration/test/monitor');

monitorWS.onmessage = (event) => {
  const status = JSON.parse(event.data);
  updateMonitoringDashboard(status);
};
```

## Security Considerations

### Medical Compliance
- **HIPAA compliance** with PHI protection
- **Audit logging** for all medical data access
- **Role-based access control** (patient, nurse, admin)
- **Data encryption** at rest and in transit
- **Session timeout** and automatic cleanup

### CORS Security
- **Domain whitelisting** for medical applications
- **IP address validation** for production environments
- **Rate limiting** per origin to prevent abuse
- **Security headers** (HSTS, CSP, X-Frame-Options)
- **Emergency access protocols** for critical situations

### Connection Security
- **TLS encryption** for all connections
- **API key authentication** with JWT tokens
- **Connection validation** and health checks
- **Automatic failover** for critical services
- **Resource limits** to prevent DoS attacks

## Performance Optimization

### Connection Pooling
- **Multi-type pools** for different connection types
- **Auto-scaling** based on load patterns
- **Medical priority queuing** for critical operations
- **Health monitoring** with automatic recovery
- **Resource optimization** for memory and CPU usage

### Caching Strategy
- **Response caching** for frequently requested data
- **Session caching** for WebSocket connections
- **Model result caching** for AI predictions
- **Database connection pooling** for queries
- **Redis integration** for distributed caching

### Load Balancing
- **Connection distribution** across pool instances
- **Priority-based routing** for medical requests
- **Health-aware routing** to healthy instances
- **Graceful degradation** under high load
- **Circuit breaker patterns** for fault tolerance

## Troubleshooting

### Common Issues

1. **WebSocket Connection Fails**
   - Check authentication token validity
   - Verify CORS configuration
   - Ensure WebSocket support is enabled
   - Check network connectivity

2. **SSE Stream Interrupts**
   - Verify connection timeout settings
   - Check client network stability
   - Monitor server resource usage
   - Review connection pool status

3. **Nurse Dashboard Not Loading**
   - Verify nurse authentication
   - Check database connectivity
   - Review API rate limits
   - Monitor system health metrics

### Logging and Debugging

```python
# Enable debug logging
import structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
```

### Health Monitoring

```bash
# Check integration system status
curl "http://localhost:8000/integration/status"

# Monitor connection pool health
curl "http://localhost:8000/integration/health"

# Get detailed metrics
curl "http://localhost:8000/integration/metrics"
```

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Run tests: `pytest tests/`
5. Start development server: `uvicorn main:app --reload`

### Code Standards
- Follow PEP 8 for Python code
- Use type hints for all functions
- Write comprehensive tests
- Document all public APIs
- Ensure HIPAA compliance

### Testing Requirements
- Unit tests for all components
- Integration tests for API endpoints
- Load tests for performance validation
- Security tests for compliance
- Mock services for isolated testing

## License

This integration system is part of the Medical AI Assistant project and is licensed under the same terms as the main project.

## Support

For technical support and questions:
- Create an issue in the project repository
- Contact the development team
- Refer to the API documentation at `/docs/`
- Check the troubleshooting guide above

---

**Note**: This system handles sensitive medical data and must be deployed with appropriate security measures, compliance validation, and monitoring in place.