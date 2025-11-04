"""
WebSocket Schemas

Pydantic models for WebSocket communication.
"""

import enum
from typing import Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field


class MessageType(str, enum.Enum):
    """WebSocket message type enumeration"""
    CHAT = "chat"
    STATUS = "status"
    ERROR = "error"
    PAR_READY = "par_ready"
    STREAMING = "streaming"
    STREAMING_COMPLETE = "streaming_complete"
    PING = "ping"
    PONG = "pong"


class WebSocketMessage(BaseModel):
    """Schema for WebSocket messages"""
    type: MessageType = Field(..., description="Message type")
    payload: Any = Field(..., description="Message payload")
    session_id: Optional[UUID] = Field(None, description="Associated session ID")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ChatMessage(BaseModel):
    """Schema for chat message payload"""
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")


class StatusMessage(BaseModel):
    """Schema for status update payload"""
    status: str = Field(..., description="Status message")
    details: Optional[dict] = Field(None, description="Additional status details")


class ErrorMessage(BaseModel):
    """Schema for error message payload"""
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[dict] = Field(None, description="Additional error details")


class StreamingChunk(BaseModel):
    """Schema for streaming token payload"""
    token: str = Field(..., description="Streamed token")
    sequence: int = Field(..., description="Token sequence number")


class PARReadyMessage(BaseModel):
    """Schema for PAR ready notification payload"""
    par_id: UUID = Field(..., description="PAR ID")
    urgency: str = Field(..., description="Urgency level")
    has_red_flags: bool = Field(..., description="Whether PAR has red flags")
