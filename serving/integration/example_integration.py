#!/usr/bin/env python3
"""
Integration Example for Medical AI Assistant
===========================================

Example demonstrating how to integrate the Medical AI Assistant
integration system with a FastAPI application.

This example shows:
1. How to add the integration components to your FastAPI app
2. How to configure CORS for medical applications
3. How to set up WebSocket and SSE endpoints
4. How to integrate nurse dashboard API routes
5. How to use the testing framework
6. How to monitor system health
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse

# Import integration components
from serving.integration import (
    integration_router,
    integration_manager,
    create_medical_cors_middleware,
    connection_manager,
    sse_manager,
    cors_manager,
    testing_engine,
    settings
)

# Initialize FastAPI app
app = FastAPI(
    title="Medical AI Assistant with Integration",
    description="Medical AI Assistant with comprehensive integration system",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None
)

# Add medical-grade CORS middleware
medical_cors_middleware = create_medical_cors_middleware()
app.middleware("http")(medical_cors_middleware)

# Add standard CORS as fallback
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.serving.allowed_origins,
    allow_credentials=settings.serving.cors_allow_credentials,
    allow_methods=settings.serving.cors_allow_methods,
    allow_headers=settings.serving.cors_allow_headers,
)

# Security
security = HTTPBearer(auto_error=False)

# Simple authentication dependency
async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication for demo purposes."""
    # In production, implement proper JWT verification
    return credentials.credentials if credentials else "demo_user"


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize integration system on startup."""
    print("🚀 Starting Medical AI Assistant Integration System...")
    
    try:
        await integration_manager.initialize()
        print("✅ Integration system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize integration system: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup integration system on shutdown."""
    print("🛑 Shutting down Medical AI Assistant Integration System...")
    await integration_manager.shutdown()
    print("✅ Integration system shutdown complete")


# Include integration router
app.include_router(integration_router)


# WebSocket endpoint for medical chat
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time medical chat."""
    # Import the WebSocket endpoint from integration
    from serving.integration.websocket.medical_chat_websocket import websocket_endpoint
    
    await websocket_endpoint(
        websocket=websocket,
        session_id=websocket.query_params.get("session_id", "default"),
        token=websocket.query_params.get("token"),
        user_type=websocket.query_params.get("user_type", "patient")
    )


# SSE endpoint for streaming responses
@app.get("/api/streaming/{stream_id}")
async def sse_stream_endpoint(
    stream_id: str,
    user_id: str = "demo_user",
    stream_type: str = "chat"
):
    """Server-Sent Events endpoint for streaming responses."""
    from serving.integration.streaming.sse_handler import sse_response_generator
    
    async with sse_response_generator(stream_id, user_id, stream_type) as active_stream_id:
        # The StreamingResponse is automatically returned by the context manager
        yield active_stream_id


# Nurse dashboard endpoints (included in integration router)
# These are automatically available at /integration/nurse/*

# API documentation endpoints (included in integration router)
# These are automatically available at /integration/docs/*

# Testing endpoints (included in integration router)
# These are automatically available at /integration/test/*


# Simple example endpoints to demonstrate integration

@app.get("/")
async def root():
    """Root endpoint with system status."""
    try:
        status = await integration_manager.get_system_status()
        return {
            "message": "Medical AI Assistant Integration System",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": status["system_status"],
            "components": {
                "websocket": "Available at /ws/chat",
                "sse_streaming": "Available at /api/streaming/{stream_id}",
                "nurse_dashboard": "Available at /integration/nurse/*",
                "api_docs": "Available at /integration/docs/*",
                "testing": "Available at /integration/test/*"
            },
            "health_check": "/integration/health",
            "metrics": "/integration/metrics",
            "documentation": "/docs"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "message": "System starting up",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/api/patient/demo-chat")
async def demo_patient_chat():
    """Demo endpoint showing patient chat interaction."""
    try:
        # Simulate patient chat using mock service
        message = "I have a headache and feel dizzy"
        session_id = "demo_session_001"
        
        # Use testing engine's mock service
        response = await testing_engine.mock_service.simulate_medical_chat(
            message=message,
            session_id=session_id,
            user_type="patient"
        )
        
        return {
            "demo_chat": {
                "patient_message": message,
                "ai_response": response["response"],
                "urgency": response["urgency"],
                "confidence": response["confidence"],
                "red_flags_detected": response["red_flags"],
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            "note": "This is a demonstration using mock data"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Demo chat failed: {str(e)}"}
        )


@app.get("/api/nurse/demo-queue")
async def demo_nurse_queue():
    """Demo endpoint showing nurse queue functionality."""
    try:
        # Get nurse queue using mock service
        queue_data = await testing_engine.mock_service.simulate_nurse_queue()
        
        return {
            "demo_queue": queue_data,
            "note": "This is a demonstration using mock data",
            "actions_available": [
                "approve - Approve the AI assessment",
                "override - Override with nurse's judgment", 
                "escalate - Escalate to higher priority",
                "request_more_info - Ask patient for more details"
            ]
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Demo queue failed: {str(e)}"}
        )


@app.get("/api/demo/compliance-check")
async def demo_compliance_check():
    """Demo endpoint showing compliance validation."""
    return {
        "compliance_check": {
            "hipaa_compliance": True,
            "phi_protection": {
                "enabled": settings.medical.phi_redaction,
                "redaction_patterns": ["SSN", "Phone", "Email", "Medical Record Numbers"],
                "audit_logging": settings.medical.enable_audit_log
            },
            "data_encryption": {
                "at_rest": settings.medical.enable_encryption,
                "in_transport": True,
                "algorithm": "AES-256"
            },
            "access_control": {
                "enabled": settings.medical.enable_rbac,
                "roles": ["patient", "nurse", "admin"],
                "session_timeout": 3600
            },
            "audit_trail": {
                "enabled": settings.medical.enable_audit_log,
                "log_file": settings.medical.audit_log_file,
                "compliance_score": 98.5
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# Example WebSocket chat handler (simplified)
@app.websocket("/demo/chat")
async def demo_chat_websocket(websocket: WebSocket):
    """Demo WebSocket chat endpoint."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process with mock service
            response = await testing_engine.mock_service.simulate_medical_chat(
                message=message_data.get("content", ""),
                session_id=message_data.get("session_id", "demo"),
                user_type=message_data.get("user_type", "patient")
            )
            
            # Send response
            await websocket.send_text(json.dumps({
                "type": "message",
                "content": response["response"],
                "urgency": response["urgency"],
                "confidence": response["confidence"],
                "red_flags": response["red_flags"],
                "timestamp": datetime.utcnow().isoformat()
            }))
            
    except WebSocketDisconnect:
        print("Demo chat WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn
    
    print("""
    🏥 Medical AI Assistant Integration Example
    ==========================================
    
    This example demonstrates integration of the Medical AI Assistant
    integration system with a FastAPI application.
    
    Endpoints:
    - Main Application: http://localhost:8000/
    - WebSocket Chat: ws://localhost:8000/ws/chat
    - Demo Chat: ws://localhost:8000/demo/chat
    - SSE Streaming: http://localhost:8000/api/streaming/{stream_id}
    - Nurse Dashboard: http://localhost:8000/integration/nurse/*
    - API Docs: http://localhost:8000/integration/docs/*
    - Testing: http://localhost:8000/integration/test/*
    - Health Check: http://localhost:8000/integration/health
    
    Demo Endpoints:
    - Patient Chat: http://localhost:8000/api/patient/demo-chat
    - Nurse Queue: http://localhost:8000/api/nurse/demo-queue
    - Compliance Check: http://localhost:8000/api/demo/compliance-check
    
    Starting server on http://localhost:8000...
    """)
    
    uvicorn.run(
        "example_integration:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level="info"
    )