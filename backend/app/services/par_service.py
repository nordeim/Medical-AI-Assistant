"""
PAR Service

Business logic for Preliminary Assessment Report management.
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

from app.models.par import PAR as PARModel, PARUrgency
from app.models.session import Session as SessionModel
from app.models.user import User
from app.schemas.par import PARReview
from app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class PARService:
    """Service for managing Preliminary Assessment Reports"""
    
    def __init__(self, db: Session):
        self.db = db
        self.audit_service = AuditService(db)
    
    def create_par(
        self,
        session_id: UUID,
        chief_complaint: str,
        symptoms: List[str],
        assessment: str,
        urgency: PARUrgency,
        rag_sources: List[Dict[str, Any]],
        red_flags: List[str],
        additional_notes: Optional[str] = None
    ) -> PARModel:
        """
        Create a new PAR for a session.
        
        Args:
            session_id: Session UUID
            chief_complaint: Main complaint from patient
            symptoms: List of identified symptoms
            assessment: AI-generated assessment
            urgency: Urgency level
            rag_sources: RAG retrieval sources
            red_flags: Detected red flag symptoms
            additional_notes: Optional additional notes
            
        Returns:
            PARModel: Created PAR
        """
        par = PARModel(
            session_id=session_id,
            chief_complaint=chief_complaint,
            symptoms=symptoms,
            assessment=assessment,
            urgency=urgency,
            rag_sources=rag_sources,
            red_flags=red_flags,
            has_red_flags=len(red_flags) > 0,
            additional_notes=additional_notes
        )
        
        self.db.add(par)
        self.db.commit()
        self.db.refresh(par)
        
        # Log PAR creation
        self.audit_service.log_par_generated(
            par_id=par.id,
            session_id=session_id,
            urgency=urgency.value,
            has_red_flags=par.has_red_flags
        )
        
        logger.info(
            f"Created PAR {par.id} for session {session_id} "
            f"(urgency={urgency}, red_flags={len(red_flags)})"
        )
        return par
    
    def get_par(self, par_id: UUID) -> Optional[PARModel]:
        """
        Get a PAR by ID.
        
        Args:
            par_id: PAR UUID
            
        Returns:
            PARModel: PAR if found, None otherwise
        """
        return self.db.query(PARModel).filter(
            PARModel.id == par_id
        ).first()
    
    def get_session_par(self, session_id: UUID) -> Optional[PARModel]:
        """
        Get the PAR for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            PARModel: PAR if found, None otherwise
        """
        return self.db.query(PARModel).filter(
            PARModel.session_id == session_id
        ).first()
    
    def get_unreviewed_pars(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[PARModel]:
        """
        Get all unreviewed PARs for nurse queue.
        
        Orders by urgency (immediate > urgent > routine) and red flags first.
        
        Args:
            limit: Maximum number of PARs to return
            offset: Number of PARs to skip
            
        Returns:
            List of unreviewed PARs
        """
        return self.db.query(PARModel).filter(
            PARModel.reviewed == False
        ).order_by(
            # Red flags first
            desc(PARModel.has_red_flags),
            # Then by urgency (immediate > urgent > routine)
            desc(PARModel.urgency),
            # Then by creation time (oldest first)
            PARModel.created_at
        ).limit(limit).offset(offset).all()
    
    def review_par(
        self,
        par_id: UUID,
        nurse: User,
        review_data: PARReview
    ) -> Optional[PARModel]:
        """
        Review a PAR (approve or override).
        
        Args:
            par_id: PAR UUID
            nurse: Nurse performing the review
            review_data: Review decision and optional override
            
        Returns:
            PARModel: Updated PAR if found, None otherwise
        """
        par = self.get_par(par_id)
        
        if par is None:
            return None
        
        if par.reviewed:
            logger.warning(f"PAR {par_id} already reviewed")
            return par
        
        par.reviewed = True
        par.reviewed_by_id = nurse.id
        par.reviewed_at = datetime.utcnow()
        par.approved = review_data.approved
        
        if not review_data.approved:
            par.override_reason = review_data.override_reason
            par.override_assessment = review_data.override_assessment
        
        par.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(par)
        
        # Log review
        if review_data.approved:
            self.audit_service.log_par_approved(
                par_id=par.id,
                nurse_id=nurse.id
            )
        else:
            self.audit_service.log_par_overridden(
                par_id=par.id,
                nurse_id=nurse.id,
                reason=review_data.override_reason
            )
        
        logger.info(
            f"PAR {par_id} reviewed by nurse {nurse.id} "
            f"(approved={review_data.approved})"
        )
        return par
    
    def get_par_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about PARs.
        
        Returns:
            dict: Statistics including counts by urgency and review status
        """
        total_pars = self.db.query(PARModel).count()
        reviewed_count = self.db.query(PARModel).filter(
            PARModel.reviewed == True
        ).count()
        pending_count = total_pars - reviewed_count
        
        approved_count = self.db.query(PARModel).filter(
            and_(PARModel.reviewed == True, PARModel.approved == True)
        ).count()
        overridden_count = self.db.query(PARModel).filter(
            and_(PARModel.reviewed == True, PARModel.approved == False)
        ).count()
        
        urgency_breakdown = {
            "routine": self.db.query(PARModel).filter(
                PARModel.urgency == PARUrgency.ROUTINE
            ).count(),
            "urgent": self.db.query(PARModel).filter(
                PARModel.urgency == PARUrgency.URGENT
            ).count(),
            "immediate": self.db.query(PARModel).filter(
                PARModel.urgency == PARUrgency.IMMEDIATE
            ).count()
        }
        
        return {
            "total_pars": total_pars,
            "reviewed_count": reviewed_count,
            "pending_count": pending_count,
            "approved_count": approved_count,
            "overridden_count": overridden_count,
            "urgency_breakdown": urgency_breakdown
        }
    
    def count_unreviewed_by_urgency(self) -> Dict[str, int]:
        """
        Count unreviewed PARs by urgency level.
        
        Returns:
            dict: Counts for each urgency level
        """
        return {
            "immediate": self.db.query(PARModel).filter(
                and_(
                    PARModel.reviewed == False,
                    PARModel.urgency == PARUrgency.IMMEDIATE
                )
            ).count(),
            "urgent": self.db.query(PARModel).filter(
                and_(
                    PARModel.reviewed == False,
                    PARModel.urgency == PARUrgency.URGENT
                )
            ).count(),
            "routine": self.db.query(PARModel).filter(
                and_(
                    PARModel.reviewed == False,
                    PARModel.urgency == PARUrgency.ROUTINE
                )
            ).count(),
            "red_flags": self.db.query(PARModel).filter(
                and_(
                    PARModel.reviewed == False,
                    PARModel.has_red_flags == True
                )
            ).count()
        }
