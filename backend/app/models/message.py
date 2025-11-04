"""
Message Model

Represents individual messages within a session.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from app.database import Base

if TYPE_CHECKING:
    from app.models.session import Session


class MessageRole(str, enum.Enum):
    """Message role enumeration"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(Base):
    """
    Message model representing a single message in a conversation.
    
    Messages are ordered sequentially within a session and contain
    the conversation content along with metadata.
    """
    
    __tablename__ = "messages"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Foreign Keys
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True)
    
    # Message Content
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    
    # Message Order
    sequence_number = Column(Integer, nullable=False)
    
    # Metadata (tool calls, RAG sources, etc.)
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Token Usage (for tracking)
    token_count = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationships
    session = relationship("Session", back_populates="messages")
    
    def __repr__(self):
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(id={self.id}, role={self.role}, content='{content_preview}')>"
    
    @property
    def is_user_message(self) -> bool:
        """Check if message is from user"""
        return self.role == MessageRole.USER
    
    @property
    def is_assistant_message(self) -> bool:
        """Check if message is from assistant"""
        return self.role == MessageRole.ASSISTANT
    
    @property
    def has_rag_sources(self) -> bool:
        """Check if message has RAG sources in metadata"""
        return "rag_sources" in self.metadata and len(self.metadata["rag_sources"]) > 0
