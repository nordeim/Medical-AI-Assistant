"""
Session Schemas

Pydantic models for session-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field

from app.models.session import SessionStatus


class SessionCreate(BaseModel):
    """Schema for creating a new session"""
    metadata: dict = Field(default_factory=dict, description="Optional session metadata")


class SessionUpdate(BaseModel):
    """Schema for updating a session"""
    status: Optional[SessionStatus] = Field(None, description="New session status")
    notes: Optional[str] = Field(None, description="Nurse notes")
    metadata: Optional[dict] = Field(None, description="Updated metadata")


class SessionStatusUpdate(BaseModel):
    """Schema for updating session status only"""
    status: SessionStatus = Field(..., description="New session status")


class MessageInSession(BaseModel):
    """Nested message schema for session responses"""
    id: UUID
    role: str
    content: str
    sequence_number: int
    created_at: datetime
    
    class Config:
        from_attributes = True


class PARInSession(BaseModel):
    """Nested PAR schema for session responses"""
    id: UUID
    chief_complaint: str
    urgency: str
    reviewed: bool
    has_red_flags: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class SessionResponse(BaseModel):
    """Schema for session responses"""
    id: UUID
    patient_id: UUID
    status: SessionStatus
    metadata: dict
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    message_count: int = Field(0, description="Number of messages in session")
    has_par: bool = Field(False, description="Whether session has a PAR")
    messages: Optional[List[MessageInSession]] = Field(None, description="Session messages")
    par: Optional[PARInSession] = Field(None, description="Session PAR")
    
    class Config:
        from_attributes = True


class SessionListResponse(BaseModel):
    """Schema for paginated session list"""
    sessions: List[SessionResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_previous: bool
