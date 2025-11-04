"""
WebSocket endpoints for real-time medical chat with conversation handling.
Provides secure, compliant WebSocket connections for patient-nurse interactions.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, Any, List
from contextlib import asynccontextmanager

from fastapi import (
    WebSocket, 
    WebSocketDisconnect, 
    Depends, 
    HTTPException, 
    status,
    Query
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import structlog

from ...config.settings import get_settings
from ...config.logging_config import (
    get_logger, get_audit_logger, LoggingContextManager,
    request_id_var, user_id_var, session_id_var
)
from ...models.base_server import model_registry
from ...models.base_server import PredictionRequest, PredictionResponse


# Configuration
settings = get_settings()
logger = get_logger("websocket.chat")
audit_logger = get_audit_logger()
security = HTTPBearer(auto_error=False)


# Pydantic Models
class ChatMessage(BaseModel):
    """Chat message model for WebSocket communication."""
    type: str = Field(default="message", description="Message type")
    content: str = Field(..., description="Message content")
    session_id: str = Field(..., description="Session identifier")
    sender_type: str = Field(..., description="Sender type: patient, agent, nurse, system")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)


class WebSocketMessage(BaseModel):
    """WebSocket message wrapper."""
    type: str = Field(..., description="Message type")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))


class ConnectionManager:
    """Manages WebSocket connections with medical data compliance."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.session_connections: Dict[str, Dict[str, WebSocket]] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.nurse_connections: Set[str] = set()
        self.patient_sessions: Dict[str, str] = {}  # session_id -> patient_id
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.logger = get_logger("websocket.manager")
    
    async def connect(
        self, 
        websocket: WebSocket, 
        user_id: str, 
        session_id: str,
        user_type: str = "patient"
    ):
        """Accept WebSocket connection with validation."""
        await websocket.accept()
        
        # Validate session exists
        if not self._validate_session(session_id, user_id):
            await websocket.close(code=4001, reason="Invalid session")
            return False
        
        # Add to active connections
        if session_id not in self.active_connections:
            self.active_connections[session_id] = set()
        
        self.active_connections[session_id].add(websocket)
        self.session_connections[f"{session_id}:{user_id}"] = websocket
        self.user_sessions[user_id] = session_id
        
        if user_type == "nurse":
            self.nurse_connections.add(user_id)
        
        self.logger.info(
            "WebSocket connected",
            user_id=user_id,
            session_id=session_id,
            user_type=user_type,
            active_connections=len(self.active_connections.get(session_id, set()))
        )
        
        # Audit log
        audit_logger.log_access(
            user_id=user_id,
            action="websocket_connect",
            resource=f"session:{session_id}",
            details={
                "user_type": user_type,
                "connection_id": str(id(websocket)),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return True
    
    def disconnect(self, user_id: str, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        session_id = self.user_sessions.get(user_id)
        if session_id:
            # Remove from active connections
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
            
            # Remove from session connections
            session_key = f"{session_id}:{user_id}"
            if session_key in self.session_connections:
                del self.session_connections[session_key]
            
            # Clean up nurse connections
            if user_id in self.nurse_connections:
                self.nurse_connections.discard(user_id)
            
            # Remove user session mapping
            if self.user_sessions.get(user_id) == session_id:
                del self.user_sessions[user_id]
            
            self.logger.info(
                "WebSocket disconnected",
                user_id=user_id,
                session_id=session_id,
                remaining_connections=len(self.active_connections.get(session_id, set()))
            )
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send message to specific user."""
        websocket = self.session_connections.get(f"{self.user_sessions.get(user_id)}:{user_id}")
        if websocket:
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                self.logger.error(f"Failed to send personal message: {e}")
                self.disconnect(user_id, websocket)
        return False
    
    async def broadcast_to_session(
        self, 
        message: Dict[str, Any], 
        session_id: str, 
        exclude_user: Optional[str] = None
    ):
        """Broadcast message to all users in session."""
        if session_id not in self.active_connections:
            return
        
        disconnected_users = []
        for websocket in self.active_connections[session_id]:
            try:
                # Find user_id for this websocket
                user_id = None
                for uid, sid in self.user_sessions.items():
                    if sid == session_id:
                        session_key = f"{session_id}:{uid}"
                        if self.session_connections.get(session_key) == websocket:
                            user_id = uid
                            break
                
                if exclude_user and user_id == exclude_user:
                    continue
                
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                self.logger.error(f"Failed to send message to websocket: {e}")
                if user_id:
                    disconnected_users.append(user_id)
        
        # Clean up disconnected users
        for user_id in disconnected_users:
            websocket = self.session_connections.get(f"{self.user_sessions.get(user_id)}:{user_id}")
            if websocket:
                self.disconnect(user_id, websocket)
    
    async def notify_nurses(self, message: Dict[str, Any], priority: str = "normal"):
        """Notify all connected nurses."""
        for user_id in list(self.nurse_connections):
            websocket = self.session_connections.get(f"{self.user_sessions.get(user_id)}:{user_id}")
            if websocket:
                try:
                    # Add priority to message
                    enhanced_message = message.copy()
                    enhanced_message["priority"] = priority
                    await websocket.send_text(json.dumps(enhanced_message))
                except Exception as e:
                    self.logger.error(f"Failed to notify nurse {user_id}: {e}")
                    self.disconnect(user_id, websocket)
    
    def _validate_session(self, session_id: str, user_id: str) -> bool:
        """Validate session and user permissions."""
        # In production, this would check against database
        # For now, basic validation
        return bool(session_id and user_id)
    
    def check_rate_limit(self, user_id: str, limit: int = 10) -> bool:
        """Check if user is within rate limits."""
        now = datetime.utcnow()
        user_requests = self.rate_limits.get(user_id, [])
        
        # Remove old requests (older than 1 minute)
        self.rate_limits[user_id] = [
            req_time for req_time in user_requests
            if (now - req_time).seconds < 60
        ]
        
        # Check limit
        if len(self.rate_limits[user_id]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[user_id].append(now)
        return True


# Global connection manager
connection_manager = ConnectionManager()


async def verify_websocket_token(
    token: Optional[str] = Query(None),
    websocket: WebSocket = None
):
    """Verify WebSocket authentication token."""
    if not settings.serving.api_key:
        return None
    
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return None
    
    if token != settings.serving.api_key:
        audit_logger.log_access(
            user_id="unknown",
            action="websocket_auth_failed",
            resource="websocket",
            details={"reason": "invalid_token"}
        )
        await websocket.close(code=4001, reason="Invalid authentication")
        return None
    
    return token


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Get current user ID from JWT token (simplified for demo)."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # In production, verify JWT token and extract user_id
    # For demo purposes, use token as user_id
    return credentials.credentials


async def medical_ai_response(
    user_message: str,
    session_id: str,
    conversation_history: List[Dict[str, Any]],
    user_id: str
) -> Dict[str, Any]:
    """Generate AI response using medical conversation model."""
    try:
        # Get conversation model
        model = model_registry.get_model("conversation_v1")
        if not model or model.status.value != "ready":
            raise HTTPException(
                status_code=503,
                detail="AI model not available"
            )
        
        # Create prediction request
        prediction_request = PredictionRequest(
            request_id=str(uuid.uuid4()),
            model_id="conversation_v1",
            inputs={
                "message": user_message,
                "conversation_history": conversation_history,
                "session_id": session_id,
                "user_type": "patient"
            },
            parameters={
                "max_length": 500,
                "temperature": 0.7,
                "top_p": 0.9
            },
            user_id=user_id,
            session_id=session_id
        )
        
        # Generate response
        response = await model.predict(prediction_request)
        
        return {
            "content": response.outputs.get("response", "I'm sorry, I couldn't process your request."),
            "confidence": response.confidence,
            "metadata": {
                "processing_time": response.processing_time,
                "model_id": response.model_id,
                "safety_flagged": response.outputs.get("safety_flagged", False),
                "red_flags_detected": response.outputs.get("red_flags_detected", [])
            }
        }
    
    except Exception as e:
        logger.error(f"AI response generation failed: {e}")
        return {
            "content": "I'm experiencing technical difficulties. Please try again.",
            "confidence": 0.0,
            "metadata": {"error": str(e)}
        }


async def handle_chat_message(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    message_data: ChatMessage
):
    """Handle incoming chat message with AI processing."""
    
    # Rate limiting
    if not connection_manager.check_rate_limit(user_id):
        await connection_manager.send_personal_message({
            "type": "error",
            "payload": {"message": "Rate limit exceeded. Please wait before sending another message."},
            "timestamp": datetime.utcnow().isoformat()
        }, user_id)
        return
    
    # Audit log
    with LoggingContextManager(
        user_id=user_id,
        session_id=session_id,
        request_id=str(uuid.uuid4())
    ):
        logger.info(
            "Chat message received",
            user_id=user_id,
            session_id=session_id,
            message_length=len(message_data.content),
            sender_type=message_data.sender_type
        )
        
        try:
            # Validate message content
            if not message_data.content.strip():
                await connection_manager.send_personal_message({
                    "type": "error",
                    "payload": {"message": "Message content cannot be empty"},
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
                return
            
            # Check for red flags in message
            red_flags = []
            content_lower = message_data.content.lower()
            
            # Simple red flag detection (would be more sophisticated in production)
            emergency_keywords = [
                "chest pain", "heart attack", "can't breathe", "unconscious", 
                "severe bleeding", "stroke", "seizure", "overdose"
            ]
            
            for keyword in emergency_keywords:
                if keyword in content_lower:
                    red_flags.append(f"Emergency keyword detected: {keyword}")
            
            # If red flags detected, notify nurses immediately
            if red_flags:
                await connection_manager.notify_nurses({
                    "type": "red_flag_alert",
                    "payload": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "red_flags": red_flags,
                        "message_content": message_data.content[:100] + "..." if len(message_data.content) > 100 else message_data.content
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, priority="high")
            
            # Process message with AI
            conversation_history = []  # Would fetch from database in production
            ai_response = await medical_ai_response(
                message_data.content,
                session_id,
                conversation_history,
                user_id
            )
            
            # Prepare AI response message
            ai_message = WebSocketMessage(
                type="message",
                payload={
                    "content": ai_response["content"],
                    "sender_type": "agent",
                    "session_id": session_id,
                    "confidence": ai_response["confidence"],
                    "metadata": {
                        **ai_response["metadata"],
                        "original_message_id": message_data.timestamp.isoformat(),
                        "red_flags": red_flags
                    }
                }
            )
            
            # Send AI response to user
            await connection_manager.send_personal_message(
                ai_message.dict(),
                user_id
            )
            
            # Broadcast to nurses if it's a patient message
            if message_data.sender_type == "patient":
                await connection_manager.broadcast_to_session({
                    "type": "patient_message",
                    "payload": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "message": message_data.content,
                        "red_flags": red_flags,
                        "ai_response": ai_response["content"]
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id)
            
            logger.info(
                "Chat message processed successfully",
                user_id=user_id,
                session_id=session_id,
                red_flags_count=len(red_flags),
                response_confidence=ai_response["confidence"]
            )
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await connection_manager.send_personal_message({
                "type": "error",
                "payload": {"message": "Failed to process message"},
                "timestamp": datetime.utcnow().isoformat()
            }, user_id)


async def handle_session_control(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    action: str,
    data: Dict[str, Any]
):
    """Handle session control messages (join, leave, etc.)."""
    
    with LoggingContextManager(
        user_id=user_id,
        session_id=session_id,
        request_id=str(uuid.uuid4())
    ):
        logger.info(
            "Session control request",
            user_id=user_id,
            session_id=session_id,
            action=action
        )
        
        try:
            if action == "session_join":
                # Session join logic
                await connection_manager.send_personal_message({
                    "type": "session_joined",
                    "payload": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "joined_at": datetime.utcnow().isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
                
                # Notify other participants
                await connection_manager.broadcast_to_session({
                    "type": "user_joined",
                    "payload": {
                        "user_id": user_id,
                        "joined_at": datetime.utcnow().isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id, exclude_user=user_id)
            
            elif action == "session_leave":
                # Session leave logic
                await connection_manager.send_personal_message({
                    "type": "session_left",
                    "payload": {
                        "session_id": session_id,
                        "user_id": user_id,
                        "left_at": datetime.utcnow().isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
                
                # Notify other participants
                await connection_manager.broadcast_to_session({
                    "type": "user_left",
                    "payload": {
                        "user_id": user_id,
                        "left_at": datetime.utcnow().isoformat()
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id, exclude_user=user_id)
            
            elif action == "typing_start":
                # Notify others that user is typing
                await connection_manager.broadcast_to_session({
                    "type": "user_typing",
                    "payload": {
                        "user_id": user_id,
                        "typing": True
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id, exclude_user=user_id)
            
            elif action == "typing_stop":
                # Notify others that user stopped typing
                await connection_manager.broadcast_to_session({
                    "type": "user_typing",
                    "payload": {
                        "user_id": user_id,
                        "typing": False
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, session_id, exclude_user=user_id)
            
            elif action == "request_assessment":
                # Request AI assessment of conversation
                assessment_request = data.get("assessment_request", "")
                
                # Generate assessment
                assessment = await medical_ai_response(
                    f"Please assess this conversation and provide medical recommendations: {assessment_request}",
                    session_id,
                    [],  # conversation history
                    user_id
                )
                
                await connection_manager.send_personal_message({
                    "type": "assessment_response",
                    "payload": {
                        "session_id": session_id,
                        "assessment": assessment["content"],
                        "confidence": assessment["confidence"],
                        "red_flags": assessment["metadata"].get("red_flags_detected", []),
                        "recommended_actions": assessment["metadata"].get("recommended_actions", [])
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
            
            logger.info(
                "Session control action completed",
                user_id=user_id,
                session_id=session_id,
                action=action
            )
            
        except Exception as e:
            logger.error(f"Error handling session control: {e}")
            await connection_manager.send_personal_message({
                "type": "error",
                "payload": {"message": f"Failed to process session action: {action}"},
                "timestamp": datetime.utcnow().isoformat()
            }, user_id)


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = Query(..., description="Session identifier"),
    token: Optional[str] = Query(None, description="Authentication token"),
    user_type: str = Query("patient", description="User type: patient, nurse, admin")
):
    """
    WebSocket endpoint for real-time medical chat.
    
    Provides secure, compliant WebSocket connections with:
    - Real-time messaging between patients and AI
    - Nurse monitoring and intervention capabilities
    - Medical data protection and audit logging
    - Red flag detection and emergency alerts
    - Session management and rate limiting
    """
    
    # Verify authentication
    auth_token = await verify_websocket_token(token, websocket)
    if not auth_token:
        return
    
    # Get user ID (simplified for demo)
    user_id = auth_token  # In production, decode JWT token
    
    # Connect to WebSocket
    connected = await connection_manager.connect(websocket, user_id, session_id, user_type)
    if not connected:
        return
    
    try:
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "message":
                # Chat message
                chat_message = ChatMessage(
                    content=message_data.get("content", ""),
                    session_id=message_data.get("session_id", session_id),
                    sender_type=message_data.get("sender_type", "patient"),
                    metadata=message_data.get("metadata", {})
                )
                await handle_chat_message(websocket, user_id, session_id, chat_message)
            
            elif message_data.get("type") == "session_control":
                # Session control message
                await handle_session_control(
                    websocket, 
                    user_id, 
                    session_id, 
                    message_data.get("action"),
                    message_data.get("data", {})
                )
            
            elif message_data.get("type") == "ping":
                # Keepalive ping
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            else:
                # Unknown message type
                await connection_manager.send_personal_message({
                    "type": "error",
                    "payload": {"message": f"Unknown message type: {message_data.get('type')}"},
                    "timestamp": datetime.utcnow().isoformat()
                }, user_id)
    
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id, websocket)
        logger.info(
            "WebSocket disconnected normally",
            user_id=user_id,
            session_id=session_id
        )
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        connection_manager.disconnect(user_id, websocket)
    
    finally:
        # Cleanup
        audit_logger.log_access(
            user_id=user_id,
            action="websocket_disconnect",
            resource=f"session:{session_id}",
            details={
                "disconnected_at": datetime.utcnow().isoformat(),
                "connection_id": str(id(websocket))
            }
        )


# Export WebSocket endpoint for use in main API
__all__ = [
    "websocket_endpoint",
    "connection_manager",
    "WebSocketMessage",
    "ChatMessage",
    "ConnectionManager"
]