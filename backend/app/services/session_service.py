"""
Session Service

Business logic for session management.
"""

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.session import Session as SessionModel, SessionStatus
from app.models.user import User
from app.schemas.session import SessionCreate, SessionUpdate
from app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing patient sessions"""
    
    def __init__(self, db: Session):
        self.db = db
        self.audit_service = AuditService(db)
    
    def create_session(
        self,
        session_data: SessionCreate,
        patient: User
    ) -> SessionModel:
        """
        Create a new session for a patient.
        
        Args:
            session_data: Session creation data
            patient: Patient user
            
        Returns:
            SessionModel: Created session
        """
        session = SessionModel(
            patient_id=patient.id,
            status=SessionStatus.ACTIVE,
            metadata=session_data.metadata or {}
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        # Log session creation
        self.audit_service.log_session_created(
            session_id=session.id,
            patient_id=patient.id
        )
        
        logger.info(f"Created session {session.id} for patient {patient.id}")
        return session
    
    def get_session(self, session_id: UUID) -> Optional[SessionModel]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session UUID
            
        Returns:
            SessionModel: Session if found, None otherwise
        """
        return self.db.query(SessionModel).filter(
            SessionModel.id == session_id
        ).first()
    
    def get_patient_sessions(
        self,
        patient_id: UUID,
        limit: int = 50,
        offset: int = 0
    ) -> List[SessionModel]:
        """
        Get all sessions for a patient.
        
        Args:
            patient_id: Patient UUID
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of sessions
        """
        return self.db.query(SessionModel).filter(
            SessionModel.patient_id == patient_id
        ).order_by(
            desc(SessionModel.created_at)
        ).limit(limit).offset(offset).all()
    
    def get_active_sessions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[SessionModel]:
        """
        Get all active sessions (for nurse dashboard).
        
        Args:
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of active sessions
        """
        return self.db.query(SessionModel).filter(
            SessionModel.status == SessionStatus.ACTIVE
        ).order_by(
            desc(SessionModel.created_at)
        ).limit(limit).offset(offset).all()
    
    def update_session(
        self,
        session_id: UUID,
        session_update: SessionUpdate
    ) -> Optional[SessionModel]:
        """
        Update a session.
        
        Args:
            session_id: Session UUID
            session_update: Update data
            
        Returns:
            SessionModel: Updated session if found, None otherwise
        """
        session = self.get_session(session_id)
        
        if session is None:
            return None
        
        # Update fields
        if session_update.status is not None:
            old_status = session.status
            session.status = session_update.status
            
            if session_update.status != SessionStatus.ACTIVE:
                session.completed_at = datetime.utcnow()
            
            # Log status change
            if old_status != session_update.status:
                self.audit_service.log_session_status_changed(
                    session_id=session.id,
                    old_status=old_status.value,
                    new_status=session_update.status.value
                )
        
        if session_update.notes is not None:
            session.notes = session_update.notes
        
        if session_update.metadata is not None:
            session.metadata.update(session_update.metadata)
        
        session.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(session)
        
        logger.info(f"Updated session {session_id}")
        return session
    
    def complete_session(self, session_id: UUID) -> Optional[SessionModel]:
        """
        Mark a session as completed.
        
        Args:
            session_id: Session UUID
            
        Returns:
            SessionModel: Updated session if found, None otherwise
        """
        session = self.get_session(session_id)
        
        if session is None:
            return None
        
        session.status = SessionStatus.COMPLETED
        session.completed_at = datetime.utcnow()
        session.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(session)
        
        self.audit_service.log_session_completed(session_id=session.id)
        
        logger.info(f"Completed session {session_id}")
        return session
    
    def count_patient_sessions(self, patient_id: UUID) -> int:
        """
        Count total sessions for a patient.
        
        Args:
            patient_id: Patient UUID
            
        Returns:
            int: Total session count
        """
        return self.db.query(SessionModel).filter(
            SessionModel.patient_id == patient_id
        ).count()
