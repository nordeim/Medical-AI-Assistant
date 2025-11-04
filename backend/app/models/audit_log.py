"""
Audit Log Model

Tracks all significant actions in the system for compliance and security.
"""

import enum
from datetime import datetime

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB, INET
import uuid

from app.database import Base


class AuditAction(str, enum.Enum):
    """Audit action enumeration"""
    # Session Actions
    SESSION_CREATED = "session_created"
    SESSION_COMPLETED = "session_completed"
    SESSION_ESCALATED = "session_escalated"
    SESSION_CANCELLED = "session_cancelled"
    
    # Message Actions
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    
    # PAR Actions
    PAR_GENERATED = "par_generated"
    PAR_REVIEWED = "par_reviewed"
    PAR_APPROVED = "par_approved"
    PAR_OVERRIDDEN = "par_overridden"
    
    # Safety Actions
    SAFETY_FILTER_TRIGGERED = "safety_filter_triggered"
    RED_FLAG_DETECTED = "red_flag_detected"
    CONTENT_BLOCKED = "content_blocked"
    
    # Authentication Actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_REGISTERED = "user_registered"
    PASSWORD_CHANGED = "password_changed"
    
    # Admin Actions
    USER_CREATED = "user_created"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"


class AuditLog(Base):
    """
    Audit log model for compliance and security tracking.
    
    Records all significant actions in the system with full context
    including user, action, resource, and outcome.
    """
    
    __tablename__ = "audit_logs"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Action Information
    action = Column(SQLEnum(AuditAction), nullable=False, index=True)
    
    # User Information
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    user_email = Column(String(255), nullable=True)  # Denormalized for audit trail
    
    # Resource Information
    resource_type = Column(String(50), nullable=True, index=True)  # session, par, user, etc.
    resource_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Request Context
    ip_address = Column(INET, nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Action Details
    details = Column(JSONB, default=dict, nullable=False)
    outcome = Column(String(50), nullable=False, index=True)  # success, failure, blocked
    
    # Error Information (if outcome is failure)
    error_message = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<AuditLog(id={self.id}, action={self.action}, outcome={self.outcome})>"
