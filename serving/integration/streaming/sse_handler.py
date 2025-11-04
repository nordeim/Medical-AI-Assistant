"""
Server-Sent Events (SSE) handler for streaming responses in medical UI.
Provides real-time streaming of AI responses, assessment updates, and system events.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Set, Optional, Any, List, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import (
    Request, 
    Response, 
    HTTPException, 
    status
)
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import structlog

from ...config.settings import get_settings
from ...config.logging_config import (
    get_logger, get_audit_logger, LoggingContextManager
)


# Configuration
settings = get_settings()
logger = get_logger("sse")
audit_logger = get_audit_logger()


# Pydantic Models
class SSEEvent(BaseModel):
    """Server-Sent Event model."""
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    event: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(..., description="Event data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamSession(BaseModel):
    """Streaming session information."""
    session_id: str
    user_id: str
    stream_type: str  # chat, assessment, dashboard
    created_at: datetime
    active_connections: int = 0
    last_activity: datetime


class SSEStreamManager:
    """Manages Server-Sent Events streaming for medical data."""
    
    def __init__(self):
        self.active_streams: Dict[str, Set[Response]] = {}
        self.stream_sessions: Dict[str, StreamSession] = {}
        self.user_streams: Dict[str, str] = {}  # user_id -> stream_id
        self.logger = get_logger("sse.manager")
    
    async def create_stream(
        self,
        session_id: str,
        user_id: str,
        stream_type: str = "chat"
    ) -> str:
        """Create new SSE stream."""
        stream_id = str(uuid.uuid4())
        
        self.stream_sessions[stream_id] = StreamSession(
            session_id=session_id,
            user_id=user_id,
            stream_type=stream_type,
            created_at=datetime.utcnow(),
            active_connections=0,
            last_activity=datetime.utcnow()
        )
        
        self.active_streams[stream_id] = set()
        self.user_streams[user_id] = stream_id
        
        self.logger.info(
            "SSE stream created",
            stream_id=stream_id,
            session_id=session_id,
            user_id=user_id,
            stream_type=stream_type
        )
        
        return stream_id
    
    def add_connection(self, stream_id: str, response: Response):
        """Add client connection to stream."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].add(response)
            if stream_id in self.stream_sessions:
                self.stream_sessions[stream_id].active_connections += 1
                self.stream_sessions[stream_id].last_activity = datetime.utcnow()
    
    def remove_connection(self, stream_id: str, response: Response):
        """Remove client connection from stream."""
        if stream_id in self.active_streams:
            self.active_streams[stream_id].discard(response)
            if not self.active_streams[stream_id] and stream_id in self.stream_sessions:
                # Clean up empty streams
                del self.stream_sessions[stream_id]
                del self.active_streams[stream_id]
                
                # Clean up user mapping
                stream_session = None
                for user_id, sid in self.user_streams.items():
                    if sid == stream_id:
                        stream_session = self.stream_sessions.get(stream_id)
                        del self.user_streams[user_id]
                        break
                
                self.logger.info(
                    "SSE stream cleaned up",
                    stream_id=stream_id
                )
            
            elif stream_id in self.stream_sessions:
                self.stream_sessions[stream_id].active_connections -= 1
                self.stream_sessions[stream_id].last_activity = datetime.utcnow()
    
    async def broadcast_event(
        self,
        stream_id: str,
        event_type: str,
        data: Dict[str, Any],
        exclude_response: Optional[Response] = None
    ):
        """Broadcast event to all connections in stream."""
        if stream_id not in self.active_streams:
            return
        
        event = SSEEvent(event=event_type, data=data)
        event_line = f"id: {event.id}\nevent: {event_type}\ndata: {json.dumps(event.dict(), default=str)}\n\n"
        
        disconnected_responses = []
        
        for response in self.active_streams[stream_id]:
            if exclude_response and response == exclude_response:
                continue
                
            try:
                # Write to response stream
                await response.body.push(event_line.encode('utf-8'))
            except Exception as e:
                self.logger.error(f"Failed to write SSE event: {e}")
                disconnected_responses.append(response)
        
        # Clean up disconnected responses
        for response in disconnected_responses:
            self.remove_connection(stream_id, response)
        
        self.logger.debug(
            "SSE event broadcasted",
            stream_id=stream_id,
            event_type=event_type,
            connections=len(self.active_streams.get(stream_id, set()))
        )
    
    async def stream_chat_response(
        self,
        stream_id: str,
        content: str,
        token_count: int = 0,
        is_complete: bool = False,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Stream chat response token by token."""
        event_data = {
            "content": content,
            "token_count": token_count,
            "is_complete": is_complete,
            "confidence": confidence,
            "metadata": metadata or {}
        }
        
        await self.broadcast_event(stream_id, "chat_token", event_data)
    
    async def stream_assessment_update(
        self,
        stream_id: str,
        assessment_data: Dict[str, Any],
        stage: str = "processing",
        progress: Optional[int] = None
    ):
        """Stream patient assessment updates."""
        event_data = {
            "assessment_data": assessment_data,
            "stage": stage,
            "progress": progress,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_event(stream_id, "assessment_update", event_data)
    
    async def stream_dashboard_metrics(
        self,
        stream_id: str,
        metrics: Dict[str, Any]
    ):
        """Stream real-time dashboard metrics."""
        await self.broadcast_event(stream_id, "dashboard_update", {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def stream_queue_update(
        self,
        stream_id: str,
        queue_data: Dict[str, Any]
    ):
        """Stream nurse queue updates."""
        await self.broadcast_event(stream_id, "queue_update", {
            "queue_data": queue_data,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def stream_ai_thinking(
        self,
        stream_id: str,
        thinking_step: str,
        reasoning: str
    ):
        """Stream AI thinking process (for transparency)."""
        await self.broadcast_event(stream_id, "ai_thinking", {
            "step": thinking_step,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_active_streams(self) -> Dict[str, StreamSession]:
        """Get all active stream sessions."""
        return self.stream_sessions.copy()
    
    def get_user_stream(self, user_id: str) -> Optional[str]:
        """Get user's active stream ID."""
        return self.user_streams.get(user_id)
    
    def cleanup_stale_streams(self, max_age_minutes: int = 60):
        """Clean up stale streams."""
        now = datetime.utcnow()
        stale_streams = []
        
        for stream_id, session in self.stream_sessions.items():
            age_minutes = (now - session.last_activity).total_seconds() / 60
            if age_minutes > max_age_minutes:
                stale_streams.append(stream_id)
        
        for stream_id in stale_streams:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
            if stream_id in self.stream_sessions:
                del self.stream_sessions[stream_id]
            
            # Clean up user mappings
            users_to_remove = [
                user_id for user_id, sid in self.user_streams.items()
                if sid == stream_id
            ]
            for user_id in users_to_remove:
                del self.user_streams[user_id]
            
            self.logger.info("Cleaned up stale SSE stream", stream_id=stream_id)


# Global SSE manager
sse_manager = SSEStreamManager()


@asynccontextmanager
async def sse_response_generator(
    stream_id: str,
    user_id: str,
    stream_type: str = "chat"
) -> AsyncGenerator[str, None]:
    """Generate SSE response with proper headers and lifecycle management."""
    
    # Create stream session
    await sse_manager.create_stream(stream_id, user_id, stream_type)
    
    # SSE headers
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
        "Access-Control-Allow-Origin": "*",  # Configure based on CORS settings
        "Access-Control-Allow-Headers": "Cache-Control"
    }
    
    # Create streaming response
    async def generate():
        try:
            # Send initial connection event
            initial_event = {
                "id": str(uuid.uuid4()),
                "event": "connected",
                "data": {
                    "stream_id": stream_id,
                    "stream_type": stream_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    "message": "Stream connected successfully"
                }
            }
            yield f"id: {initial_event['id']}\nevent: connected\ndata: {json.dumps(initial_event['data'], default=str)}\n\n"
            
            # Keep connection alive with heartbeat
            heartbeat_interval = 30  # seconds
            last_heartbeat = datetime.utcnow()
            
            while True:
                # Check if stream is still active
                if stream_id not in sse_manager.active_streams:
                    break
                
                # Send heartbeat every 30 seconds
                now = datetime.utcnow()
                if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                    heartbeat_event = {
                        "id": str(uuid.uuid4()),
                        "event": "heartbeat",
                        "data": {
                            "timestamp": now.isoformat(),
                            "active_connections": len(sse_manager.active_streams.get(stream_id, set()))
                        }
                    }
                    yield f"id: {heartbeat_event['id']}\nevent: heartbeat\ndata: {json.dumps(heartbeat_event['data'], default=str)}\n\n"
                    last_heartbeat = now
                
                # Wait before next heartbeat
                await asyncio.sleep(heartbeat_interval)
        
        except asyncio.CancelledError:
            logger.info("SSE stream cancelled", stream_id=stream_id)
        except Exception as e:
            logger.error(f"SSE stream error: {e}", stream_id=stream_id)
            error_event = {
                "id": str(uuid.uuid4()),
                "event": "error",
                "data": {
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            yield f"id: {error_event['id']}\nevent: error\ndata: {json.dumps(error_event['data'], default=str)}\n\n"
        finally:
            # Cleanup
            if stream_id in sse_manager.stream_sessions:
                audit_logger.log_access(
                    user_id=user_id,
                    action="sse_stream_end",
                    resource=f"stream:{stream_id}",
                    details={
                        "stream_type": stream_type,
                        "duration": (datetime.utcnow() - sse_manager.stream_sessions[stream_id].created_at).total_seconds()
                    }
                )
    
    response = StreamingResponse(
        generate(),
        headers=headers,
        media_type="text/event-stream"
    )
    
    try:
        # Add connection to manager
        sse_manager.add_connection(stream_id, response)
        
        yield stream_id
        
    finally:
        # Remove connection from manager
        sse_manager.remove_connection(stream_id, response)


async def create_chat_stream(
    user_id: str,
    session_id: str,
    initial_message: str
) -> str:
    """Create a new chat streaming session."""
    stream_id = await sse_manager.create_stream(session_id, user_id, "chat")
    
    # Send initial message event
    await sse_manager.broadcast_event(stream_id, "message_received", {
        "content": initial_message,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return stream_id


async def stream_ai_response(
    stream_id: str,
    prompt: str,
    model,
    session_id: str,
    user_id: str
):
    """Stream AI response token by token."""
    
    with LoggingContextManager(
        user_id=user_id,
        session_id=session_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Send thinking event
            await sse_manager.stream_ai_thinking(
                stream_id,
                "processing_request",
                "AI is analyzing your medical query..."
            )
            
            # Prepare prediction request
            from ...models.base_server import PredictionRequest
            prediction_request = PredictionRequest(
                request_id=str(uuid.uuid4()),
                model_id="conversation_v1",
                inputs={
                    "message": prompt,
                    "session_id": session_id,
                    "stream_mode": True
                },
                parameters={
                    "max_length": 500,
                    "temperature": 0.7,
                    "stream_tokens": True
                },
                user_id=user_id,
                session_id=session_id
            )
            
            # Generate streaming response
            token_count = 0
            full_response = ""
            confidence_scores = []
            
            # This would be implemented in the actual model adapter
            # For demo, simulate token streaming
            async for token_data in model.stream_predict(prediction_request):
                token_text = token_data.get("token", "")
                confidence = token_data.get("confidence", 0.0)
                
                if token_text:
                    full_response += token_text
                    token_count += 1
                    confidence_scores.append(confidence)
                    
                    # Stream token to client
                    await sse_manager.stream_chat_response(
                        stream_id=stream_id,
                        content=token_text,
                        token_count=token_count,
                        is_complete=False,
                        confidence=confidence,
                        metadata={
                            "model_id": "conversation_v1",
                            "stream_type": "token"
                        }
                    )
                
                # Small delay for realistic streaming
                await asyncio.sleep(0.05)
            
            # Calculate final confidence
            final_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            # Send completion event
            await sse_manager.stream_chat_response(
                stream_id=stream_id,
                content="",
                token_count=token_count,
                is_complete=True,
                confidence=final_confidence,
                metadata={
                    "model_id": "conversation_v1",
                    "stream_type": "complete",
                    "full_response": full_response
                }
            )
            
            logger.info(
                "AI response streamed successfully",
                stream_id=stream_id,
                token_count=token_count,
                confidence=final_confidence
            )
            
        except Exception as e:
            logger.error(f"Failed to stream AI response: {e}")
            await sse_manager.broadcast_event(stream_id, "stream_error", {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })


async def stream_patient_assessment(
    stream_id: str,
    session_id: str,
    conversation_data: List[Dict[str, Any]],
    user_id: str
):
    """Stream patient assessment generation process."""
    
    with LoggingContextManager(
        user_id=user_id,
        session_id=session_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Stage 1: Processing conversation
            await sse_manager.stream_assessment_update(
                stream_id=stream_id,
                assessment_data={"stage": "processing_conversation"},
                stage="processing",
                progress=10
            )
            
            await asyncio.sleep(1)  # Simulate processing time
            
            # Stage 2: Analyzing symptoms
            await sse_manager.stream_assessment_update(
                stream_id=stream_id,
                assessment_data={
                    "stage": "analyzing_symptoms",
                    "symptoms_found": ["headache", "fatigue"],
                    "symptom_analysis": "Based on your description, you're experiencing headache and fatigue."
                },
                stage="analyzing",
                progress=30
            )
            
            await asyncio.sleep(2)
            
            # Stage 3: Risk assessment
            await sse_manager.stream_assessment_update(
                stream_id=stream_id,
                assessment_data={
                    "stage": "risk_assessment",
                    "risk_level": "low",
                    "risk_factors": ["mild_symptoms", "gradual_onset"],
                    "urgency": "routine"
                },
                stage="assessing",
                progress=60
            )
            
            await asyncio.sleep(1.5)
            
            # Stage 4: Generating recommendations
            await sse_manager.stream_assessment_update(
                stream_id=stream_id,
                assessment_data={
                    "stage": "generating_recommendations",
                    "recommendations": [
                        "Schedule follow-up with primary care physician",
                        "Monitor symptoms for 24-48 hours",
                        "Consider rest and hydration"
                    ]
                },
                stage="recommending",
                progress=80
            )
            
            await asyncio.sleep(1)
            
            # Final assessment
            final_assessment = {
                "session_id": session_id,
                "chief_complaint": "Headache and fatigue",
                "symptoms": ["headache", "fatigue"],
                "risk_level": "low",
                "urgency": "routine",
                "recommendations": [
                    "Schedule follow-up with primary care physician",
                    "Monitor symptoms for 24-48 hours",
                    "Consider rest and hydration"
                ],
                "confidence": 0.85,
                "red_flags": [],
                "assessment_data": {
                    "symptom_analysis": "Mild symptoms with gradual onset",
                    "differential_diagnosis": ["Viral syndrome", "Dehydration", "Stress-related"],
                    "guideline_references": [
                        {
                            "title": "Headache Management Guidelines",
                            "section": "Mild headache assessment",
                            "recommendation": "Monitor and reassess if symptoms worsen"
                        }
                    ]
                }
            }
            
            await sse_manager.stream_assessment_update(
                stream_id=stream_id,
                assessment_data=final_assessment,
                stage="complete",
                progress=100
            )
            
            logger.info(
                "Patient assessment streamed successfully",
                stream_id=stream_id,
                session_id=session_id
            )
            
        except Exception as e:
            logger.error(f"Failed to stream patient assessment: {e}")
            await sse_manager.broadcast_event(stream_id, "assessment_error", {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })


# Export SSE manager and helper functions
__all__ = [
    "sse_manager",
    "SSEStreamManager",
    "SSEEvent",
    "StreamSession",
    "sse_response_generator",
    "create_chat_stream",
    "stream_ai_response",
    "stream_patient_assessment"
]