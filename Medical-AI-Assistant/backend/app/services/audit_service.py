"""
Audit Service

Business logic for audit logging and tracking.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog, AuditAction

logger = logging.getLogger(__name__)


class AuditService:
    """Service for audit logging"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def log_action(
        self,
        action: AuditAction,
        user_id: Optional[UUID] = None,
        user_email: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None,
        outcome: str = "success",
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditLog:
        """
        Log an audit event.
        
        Args:
            action: Action that was performed
            user_id: User who performed the action
            user_email: User's email (denormalized)
            resource_type: Type of resource affected
            resource_id: ID of affected resource
            details: Additional details
            outcome: Result of the action (success, failure, blocked)
            error_message: Error message if outcome is failure
            ip_address: IP address of request
            user_agent: User agent string
            
        Returns:
            AuditLog: Created audit log entry
        """
        audit_log = AuditLog(
            action=action,
            user_id=user_id,
            user_email=user_email,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details or {},
            outcome=outcome,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.db.add(audit_log)
        self.db.commit()
        self.db.refresh(audit_log)
        
        logger.debug(
            f"Audit log created: {action.value} "
            f"(user={user_id}, outcome={outcome})"
        )
        return audit_log
    
    # Session-related audit methods
    
    def log_session_created(self, session_id: UUID, patient_id: UUID) -> AuditLog:
        """Log session creation"""
        return self.log_action(
            action=AuditAction.SESSION_CREATED,
            user_id=patient_id,
            resource_type="session",
            resource_id=session_id
        )
    
    def log_session_completed(self, session_id: UUID) -> AuditLog:
        """Log session completion"""
        return self.log_action(
            action=AuditAction.SESSION_COMPLETED,
            resource_type="session",
            resource_id=session_id
        )
    
    def log_session_status_changed(
        self,
        session_id: UUID,
        old_status: str,
        new_status: str
    ) -> AuditLog:
        """Log session status change"""
        action_map = {
            "completed": AuditAction.SESSION_COMPLETED,
            "escalated": AuditAction.SESSION_ESCALATED,
            "cancelled": AuditAction.SESSION_CANCELLED
        }
        
        action = action_map.get(new_status, AuditAction.SESSION_COMPLETED)
        
        return self.log_action(
            action=action,
            resource_type="session",
            resource_id=session_id,
            details={"old_status": old_status, "new_status": new_status}
        )
    
    # PAR-related audit methods
    
    def log_par_generated(
        self,
        par_id: UUID,
        session_id: UUID,
        urgency: str,
        has_red_flags: bool
    ) -> AuditLog:
        """Log PAR generation"""
        return self.log_action(
            action=AuditAction.PAR_GENERATED,
            resource_type="par",
            resource_id=par_id,
            details={
                "session_id": str(session_id),
                "urgency": urgency,
                "has_red_flags": has_red_flags
            }
        )
    
    def log_par_approved(self, par_id: UUID, nurse_id: UUID) -> AuditLog:
        """Log PAR approval"""
        return self.log_action(
            action=AuditAction.PAR_APPROVED,
            user_id=nurse_id,
            resource_type="par",
            resource_id=par_id
        )
    
    def log_par_overridden(
        self,
        par_id: UUID,
        nurse_id: UUID,
        reason: Optional[str]
    ) -> AuditLog:
        """Log PAR override"""
        return self.log_action(
            action=AuditAction.PAR_OVERRIDDEN,
            user_id=nurse_id,
            resource_type="par",
            resource_id=par_id,
            details={"override_reason": reason}
        )
    
    # Safety-related audit methods
    
    def log_safety_filter_triggered(
        self,
        session_id: UUID,
        filter_type: str,
        details: Dict[str, Any]
    ) -> AuditLog:
        """Log safety filter activation"""
        return self.log_action(
            action=AuditAction.SAFETY_FILTER_TRIGGERED,
            resource_type="session",
            resource_id=session_id,
            details={"filter_type": filter_type, **details}
        )
    
    def log_red_flag_detected(
        self,
        session_id: UUID,
        red_flags: list
    ) -> AuditLog:
        """Log red flag detection"""
        return self.log_action(
            action=AuditAction.RED_FLAG_DETECTED,
            resource_type="session",
            resource_id=session_id,
            details={"red_flags": red_flags}
        )
    
    # Authentication-related audit methods
    
    def log_user_login(
        self,
        user_id: UUID,
        user_email: str,
        ip_address: Optional[str] = None
    ) -> AuditLog:
        """Log user login"""
        return self.log_action(
            action=AuditAction.USER_LOGIN,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address
        )
    
    def log_user_registered(
        self,
        user_id: UUID,
        user_email: str
    ) -> AuditLog:
        """Log user registration"""
        return self.log_action(
            action=AuditAction.USER_REGISTERED,
            user_id=user_id,
            user_email=user_email
        )
