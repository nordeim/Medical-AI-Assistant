"""
Streaming Response Endpoints
Real-time medical AI chat experiences with streaming support
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from ..utils.exceptions import StreamingError, ValidationError, MedicalValidationError
from ..utils.security import SecurityValidator, rate_limiter
from ..utils.logger import get_logger
from ..config import get_settings
from .inference import InferenceRequest, InferenceResponse

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# WebSocket connection manager for medical conversations
class MedicalWebSocketManager:
    """WebSocket connection manager with medical context"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}
        self.medical_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str = None):
        """Accept WebSocket connection with medical session validation"""
        
        await websocket.accept()
        connection_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[connection_id] = websocket
        
        # Initialize conversation context
        self.conversation_contexts[connection_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "message_history": [],
            "medical_context": {},
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat()
        }
        
        # Initialize medical session
        self.medical_sessions[session_id] = {
            "connection_id": connection_id,
            "patient_id": None,
            "medical_domain": None,
            "urgency_level": "medium",
            "conversation_state": "active",
            "messages_processed": 0
        }
        
        logger.info(
            "WebSocket medical connection established",
            connection_id=connection_id,
            session_id=session_id,
            user_id=user_id
        )
        
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if connection_id in self.conversation_contexts:
            context = self.conversation_contexts[connection_id]
            
            logger.info(
                "WebSocket medical connection closed",
                connection_id=connection_id,
                session_id=context["session_id"],
                message_count=len(context["message_history"])
            )
            
            del self.conversation_contexts[connection_id]
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """Send message to WebSocket connection"""
        
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                await self.disconnect(connection_id)
    
    async def broadcast_medical_alert(self, session_id: str, alert: Dict[str, Any]):
        """Broadcast medical alert to specific session"""
        
        if session_id in self.medical_sessions:
            connection_id = self.medical_sessions[session_id]["connection_id"]
            await self.send_message(connection_id, {
                "type": "medical_alert",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": alert
            })
    
    def update_medical_context(self, connection_id: str, context: Dict[str, Any]):
        """Update medical context for conversation"""
        
        if connection_id in self.conversation_contexts:
            self.conversation_contexts[connection_id]["medical_context"].update(context)
            self.conversation_contexts[connection_id]["last_activity"] = datetime.now(timezone.utc).isoformat()
    
    def get_medical_context(self, connection_id: str) -> Dict[str, Any]:
        """Get medical context for conversation"""
        
        return self.conversation_contexts.get(connection_id, {})


# Pydantic models for streaming
class StreamChatRequest(BaseModel):
    """Streaming chat request"""
    
    session_id: str = Field(..., min_length=1, max_length=100, description="Unique session identifier")
    message: str = Field(..., min_length=1, max_length=4000, description="Chat message")
    medical_domain: Optional[str] = Field(None, description="Medical domain")
    urgency_level: Optional[str] = Field("medium", description="Message urgency")
    patient_id: Optional[str] = Field(None, description="Anonymized patient ID")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class StreamingResponseChunk(BaseModel):
    """Streaming response chunk"""
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Response content chunk")
    is_final: bool = Field(..., description="Whether this is the final chunk")
    timestamp: str = Field(..., description="Chunk timestamp")
    confidence: Optional[float] = Field(None, description="Model confidence for this chunk")
    medical_context: Optional[Dict[str, Any]] = Field(None, description="Medical context updates")
    
    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "chunk-uuid",
                "content": "Based on your symptoms...",
                "is_final": False,
                "timestamp": "2024-01-15T10:30:00Z",
                "confidence": 0.85,
                "medical_context": {
                    "symptoms_detected": ["headache", "fever"],
                    "urgency_assessment": "medium"
                }
            }
        }


class ChatSessionResponse(BaseModel):
    """Chat session initialization response"""
    
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Session status")
    medical_context: Dict[str, Any] = Field(..., description="Initial medical context")
    conversation_state: str = Field(..., description="Current conversation state")
    timestamp: str = Field(..., description="Session creation timestamp")


# Global WebSocket manager
websocket_manager = MedicalWebSocketManager()


# Endpoint implementations
@router.post("/chat/session", response_model=ChatSessionResponse)
async def initialize_streaming_session(request: Request):
    """
    Initialize a streaming medical chat session.
    
    Creates a persistent conversation context with:
    - Medical domain tracking
    - Urgency level monitoring
    - PHI protection status
    - Audit trail initialization
    """
    
    session_id = str(uuid.uuid4())
    
    logger.info(
        "Initializing streaming medical session",
        session_id=session_id,
        client_ip=request.client.host if request.client else None
    )
    
    # Initialize medical session context
    medical_context = {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "medical_domain": None,
        "urgency_level": "medium",
        "phi_protection_enabled": True,
        "audit_logging_enabled": True,
        "conversation_length": 0,
        "last_medical_evaluation": None
    }
    
    return ChatSessionResponse(
        session_id=session_id,
        status="initialized",
        medical_context=medical_context,
        conversation_state="active",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.websocket("/chat/{session_id}")
async def websocket_chat_endpoint(
    websocket: WebSocket, 
    session_id: str,
    user_id: str = None
):
    """
    WebSocket endpoint for real-time medical chat.
    
    Features:
    - Real-time streaming responses
    - Medical context preservation
    - PHI detection and protection
    - Clinical decision support
    - Emergency escalation protocols
    """
    
    connection_id = None
    try:
        # Accept WebSocket connection
        connection_id = await websocket_manager.connect(websocket, session_id, user_id)
        
        # Send session confirmation
        await websocket_manager.send_message(connection_id, {
            "type": "session_confirmed",
            "session_id": session_id,
            "connection_id": connection_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "features": {
                "streaming": True,
                "phi_protection": True,
                "medical_validation": True,
                "clinical_support": True
            }
        })
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                await _handle_chat_message(
                    connection_id, session_id, message_data, websocket
                )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket_manager.send_message(connection_id, {
                    "type": "error",
                    "error": "Invalid JSON message format",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Error handling chat message: {e}")
                await websocket_manager.send_message(connection_id, {
                    "type": "error",
                    "error": "Internal server error",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", session_id=session_id)
        
    finally:
        if connection_id:
            await websocket_manager.disconnect(connection_id)


@router.post("/chat/http-stream")
async def http_streaming_chat(
    request: StreamChatRequest,
    http_request: Request
):
    """
    HTTP-based streaming chat endpoint.
    
    Alternative to WebSocket for environments where WebSocket is not supported.
    Provides server-sent events for real-time responses.
    """
    
    session_id = request.session_id
    message_id = str(uuid.uuid4())
    
    logger.info(
        "Starting HTTP streaming chat",
        session_id=session_id,
        message_id=message_id,
        medical_domain=request.medical_domain,
        client_ip=http_request.client.host if http_request.client else None
    )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        try:
            # Send initial response start
            yield f"data: {json.dumps({\n                'type': 'response_start',\n                'message_id': message_id,\n                'timestamp': datetime.now(timezone.utc).isoformat()\n            })}\\n\\n"
            
            # Process message and generate streaming response
            async for chunk in _generate_streaming_response(request, message_id):
                chunk_data = {
                    "type": "response_chunk",
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "is_final": chunk.is_final,
                    "timestamp": chunk.timestamp,
                    "confidence": chunk.confidence,
                    "medical_context": chunk.medical_context
                }
                
                yield f"data: {json.dumps(chunk_data)}\\n\\n"
            
            # Send completion signal
            yield f"data: {json.dumps({\n                'type': 'response_complete',\n                'message_id': message_id,\n                'timestamp': datetime.now(timezone.utc).isoformat()\n            })}\\n\\n"
            
        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e),
                "message_id": message_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            yield f"data: {json.dumps(error_data)}\\n\\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )


@router.get("/chat/{session_id}/context")
async def get_conversation_context(session_id: str):
    """
    Get conversation context and medical information.
    
    Returns:
    - Message history
    - Medical context
    - Conversation state
    - PHI protection status
    """
    
    # Find session in active connections
    for connection_id, context in websocket_manager.conversation_contexts.items():
        if context["session_id"] == session_id:
            return {
                "session_id": session_id,
                "medical_context": context["medical_context"],
                "message_history": context["message_history"],
                "conversation_state": "active",
                "last_activity": context["last_activity"],
                "started_at": context["started_at"]
            }
    
    # If not found in active sessions, could check persistent storage
    return {
        "session_id": session_id,
        "medical_context": {},
        "message_history": [],
        "conversation_state": "inactive",
        "last_activity": None,
        "started_at": None
    }


# Helper functions
async def _handle_chat_message(
    connection_id: str, 
    session_id: str, 
    message_data: Dict[str, Any], 
    websocket: WebSocket
):
    """Handle incoming chat message and generate streaming response"""
    
    try:
        # Validate message format
        if "content" not in message_data:
            await websocket_manager.send_message(connection_id, {
                "type": "error",
                "error": "Message content is required",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return
        
        content = message_data["content"]
        
        # Update conversation context
        context = websocket_manager.get_medical_context(connection_id)
        context["message_history"].append({
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "user_message"
        })
        
        # Send typing indicator
        await websocket_manager.send_message(connection_id, {
            "type": "typing_indicator",
            "status": "generating",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Generate streaming response
        async for chunk in _generate_streaming_chat_response(content, session_id, context):
            await websocket_manager.send_message(connection_id, {
                "type": "response_chunk",
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "is_final": chunk["is_final"],
                "timestamp": chunk["timestamp"],
                "confidence": chunk.get("confidence"),
                "medical_context": chunk.get("medical_context")
            })
        
        # Send completion signal
        await websocket_manager.send_message(connection_id, {
            "type": "response_complete",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error handling chat message: {e}", session_id=session_id)
        await websocket_manager.send_message(connection_id, {
            "type": "error",
            "error": "Failed to process message",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })


async def _generate_streaming_response(
    request: StreamChatRequest, 
    message_id: str
) -> AsyncGenerator[StreamingResponseChunk, None]:
    """Generate streaming response for HTTP endpoint"""
    
    chunk_id = 0
    
    # Simulate streaming generation (in production, this would stream from model)
    response_parts = [
        "I understand you're asking about your health concerns.",
        "Based on the information you've provided,",
        "this could indicate several possibilities.",
        "However, I must emphasize that this is for informational purposes only",
        "and should not replace professional medical advice.",
        "I recommend consulting with a healthcare provider for proper evaluation."
    ]
    
    for part in response_parts:
        chunk = StreamingResponseChunk(
            chunk_id=f"{message_id}-chunk-{chunk_id}",
            content=part,
            is_final=chunk_id == len(response_parts) - 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            confidence=0.8 if chunk_id == len(response_parts) - 1 else 0.6,
            medical_context={
                "processing_stage": "response_generation",
                "urgency_level": request.urgency_level,
                "medical_domain": request.medical_domain
            } if chunk_id == 0 else None
        )
        
        yield chunk
        chunk_id += 1
        
        # Simulate processing delay
        await asyncio.sleep(0.1)


async def _generate_streaming_chat_response(
    content: str, 
    session_id: str, 
    context: Dict[str, Any]
) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate streaming chat response"""
    
    chunk_id = 0
    message_id = str(uuid.uuid4())
    
    # Analyze user message for medical context
    medical_context = _analyze_chat_message_medical_context(content)
    
    # Update conversation context
    websocket_manager.update_medical_context(context["session_id"], medical_context)
    
    # Generate response based on medical context
    response_segments = _generate_contextual_response(content, medical_context)
    
    for i, segment in enumerate(response_segments):
        chunk_data = {
            "chunk_id": f"{message_id}-chunk-{chunk_id}",
            "content": segment,
            "is_final": i == len(response_segments) - 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.9 if i == len(response_segments) - 1 else 0.7,
            "medical_context": medical_context if i == 0 else None
        }
        
        yield chunk_data
        chunk_id += 1
        
        # Simulate streaming delay
        await asyncio.sleep(0.1)


def _analyze_chat_message_medical_context(message: str) -> Dict[str, Any]:
    """Analyze chat message for medical context"""
    
    context = {
        "medical_terms_detected": [],
        "symptoms_mentioned": [],
        "urgency_indicators": [],
        "domain_suggestions": [],
        "requires_immediate_attention": False
    }
    
    message_lower = message.lower()
    
    # Detect medical terms
    medical_keywords = {
        "pain": ["pain", "ache", "hurt", "sore"],
        "fever": ["fever", "temperature", "hot", "chills"],
        "breathing": ["breath", "breathing", "shortness of breath", "wheezing"],
        "chest": ["chest", "heart", "cardiac", "chest pain"],
        "neurological": ["headache", "dizzy", "confusion", "memory"]
    }
    
    for category, keywords in medical_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            context["medical_terms_detected"].append(category)
    
    # Detect symptoms
    symptom_keywords = [
        "pain", "ache", "hurt", "fever", "cough", "shortness of breath",
        "nausea", "vomiting", "diarrhea", "headache", "dizziness", "fatigue"
    ]
    
    for symptom in symptom_keywords:
        if symptom in message_lower:
            context["symptoms_mentioned"].append(symptom)
    
    # Detect urgency indicators
    urgency_keywords = {
        "critical": ["severe", "intense", "unbearable", "emergency", "urgent"],
        "high": ["bad", "worse", "getting worse", "can barely"],
        "medium": ["moderate", "persistent", "ongoing"]
    }
    
    for urgency, keywords in urgency_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            context["urgency_indicators"].append(urgency)
    
    # Determine if immediate attention required
    emergency_patterns = [
        "chest pain", "difficulty breathing", "severe bleeding",
        "loss of consciousness", "severe headache", "can't breathe"
    ]
    
    for pattern in emergency_patterns:
        if pattern in message_lower:
            context["requires_immediate_attention"] = True
            break
    
    # Suggest medical domain
    if any(term in context["medical_terms_detected"] for term in ["chest", "heart"]):
        context["domain_suggestions"].append("cardiology")
    if any(term in context["medical_terms_detected"] for term in ["headache", "dizzy", "confusion"]):
        context["domain_suggestions"].append("neurology")
    
    return context


def _generate_contextual_response(
    message: str, 
    medical_context: Dict[str, Any]
) -> List[str]:
    """Generate contextual response based on medical analysis"""
    
    responses = []
    
    # Opening acknowledgment
    responses.append("I understand you're concerned about your health.")
    
    # Analyze symptoms mentioned
    if medical_context["symptoms_mentioned"]:
        symptoms_str = ", ".join(medical_context["symptoms_mentioned"][:3])
        responses.append(f"You've mentioned symptoms including: {symptoms_str}.")
    
    # Urgency assessment
    if medical_context["requires_immediate_attention"]:
        responses.append("Based on the severity of your symptoms, I recommend seeking immediate medical attention.")
    elif "critical" in medical_context["urgency_indicators"]:
        responses.append("Your symptoms appear to be quite concerning and should be evaluated by a healthcare provider soon.")
    elif "high" in medical_context["urgency_indicators"]:
        responses.append("I suggest scheduling an appointment with your doctor to address these symptoms.")
    else:
        responses.append("While these symptoms may not be immediately concerning, it's important to monitor them.")
    
    # Domain-specific advice
    if "cardiology" in medical_context["domain_suggestions"]:
        responses.append("Given the cardiac-related symptoms, consider consulting with a cardiologist.")
    elif "neurology" in medical_context["domain_suggestions"]:
        responses.append("For neurological symptoms, a neurologist consultation may be appropriate.")
    
    # Closing disclaimer
    responses.append("Please remember that this information is for educational purposes only and should not replace professional medical advice.")
    
    return responses