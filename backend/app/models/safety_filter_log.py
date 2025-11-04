"""
Safety Filter Log Model

Tracks safety filter activations for monitoring and improvement.
"""

import enum
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from app.database import Base


class FilterType(str, enum.Enum):
    """Safety filter type enumeration"""
    DIAGNOSIS_LANGUAGE = "diagnosis_language"
    PRESCRIPTION_LANGUAGE = "prescription_language"
    PERSONAL_ADVICE = "personal_advice"
    INAPPROPRIATE_CONTENT = "inappropriate_content"
    RED_FLAG_SYMPTOM = "red_flag_symptom"
    PHI_EXPOSURE = "phi_exposure"
    OTHER = "other"


class SafetyFilterLog(Base):
    """
    Safety filter log model.
    
    Records all safety filter activations including the triggering content,
    filter type, and action taken. Used for monitoring and improving
    safety mechanisms.
    """
    
    __tablename__ = "safety_filter_logs"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Associated Resources
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=True, index=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), nullable=True, index=True)
    
    # Filter Information
    filter_type = Column(SQLEnum(FilterType), nullable=False, index=True)
    filter_name = Column(String(100), nullable=False)
    
    # Triggering Content
    original_content = Column(Text, nullable=False)
    corrected_content = Column(Text, nullable=True)
    
    # Detection Details
    triggered_patterns = Column(JSONB, default=list, nullable=False)
    confidence_score = Column(Integer, nullable=True)  # 0-100
    
    # Action Taken
    action_taken = Column(String(50), nullable=False)  # blocked, corrected, flagged, passed
    
    # Additional Context
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<SafetyFilterLog(id={self.id}, filter_type={self.filter_type}, action={self.action_taken})>"
    
    @property
    def was_blocked(self) -> bool:
        """Check if content was blocked"""
        return self.action_taken == "blocked"
    
    @property
    def was_corrected(self) -> bool:
        """Check if content was corrected"""
        return self.action_taken == "corrected"
