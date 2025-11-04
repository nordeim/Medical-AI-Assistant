"""
Session Model

Represents a patient-AI interaction session.
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from app.database import Base

if TYPE_CHECKING:
    from app.models.message import Message
    from app.models.par import PAR
    from app.models.user import User


class SessionStatus(str, enum.Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class Session(Base):
    """
    Session model representing a patient conversation session.
    
    A session contains all messages exchanged during a single
    patient interaction and may result in a PAR.
    """
    
    __tablename__ = "sessions"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Foreign Keys
    patient_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # Session Information
    status = Column(
        SQLEnum(SessionStatus),
        default=SessionStatus.ACTIVE,
        nullable=False,
        index=True
    )
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Notes (for nurse use)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    patient = relationship("User", back_populates="sessions", foreign_keys=[patient_id])
    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    par = relationship(
        "PAR",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Session(id={self.id}, patient_id={self.patient_id}, status={self.status})>"
    
    @property
    def message_count(self) -> int:
        """Get total number of messages in session"""
        return len(self.messages) if self.messages else 0
    
    @property
    def has_par(self) -> bool:
        """Check if session has a PAR"""
        return self.par is not None
    
    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == SessionStatus.ACTIVE
