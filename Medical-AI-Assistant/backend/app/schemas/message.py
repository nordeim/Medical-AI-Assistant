"""
Message Schemas

Pydantic models for message-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field

from app.models.message import MessageRole


class MessageCreate(BaseModel):
    """Schema for creating a new message"""
    content: str = Field(..., min_length=1, max_length=5000, description="Message content")
    metadata: dict = Field(default_factory=dict, description="Optional message metadata")


class MessageResponse(BaseModel):
    """Schema for message responses"""
    id: UUID
    session_id: UUID
    role: MessageRole
    content: str
    sequence_number: int
    metadata: dict
    token_count: Optional[int]
    created_at: datetime
    
    class Config:
        from_attributes = True


class MessageListResponse(BaseModel):
    """Schema for message list responses"""
    messages: List[MessageResponse]
    total: int = Field(..., description="Total number of messages")
    session_id: UUID = Field(..., description="Session ID")


class RAGSource(BaseModel):
    """Schema for RAG source information"""
    content: str = Field(..., description="Source content snippet")
    source: str = Field(..., description="Source document name")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: dict = Field(default_factory=dict, description="Additional source metadata")
