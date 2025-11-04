"""
PAR API Routes

Preliminary Assessment Report endpoints for nurse dashboard.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.schemas.par import PARResponse, PARReview, PARQueueResponse, PARQueueItem
from app.services.par_service import PARService
from app.services.session_service import SessionService
from app.dependencies import get_current_user, get_current_nurse
from app.auth.permissions import can_access_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pars", tags=["PARs"])


@router.get("/{par_id}", response_model=PARResponse)
async def get_par(
    par_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get PAR details."""
    par_service = PARService(db)
    par = par_service.get_par(par_id)
    
    if par is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PAR not found"
        )
    
    # Check access permissions via session
    session_service = SessionService(db)
    session = session_service.get_session(par.session_id)
    
    if session and not can_access_session(current_user, session.patient_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this PAR"
        )
    
    return par


@router.get("/session/{session_id}", response_model=PARResponse)
async def get_session_par(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get PAR for a session."""
    # Verify session access
    session_service = SessionService(db)
    session = session_service.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    if not can_access_session(current_user, session.patient_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session"
        )
    
    # Get PAR
    par_service = PARService(db)
    par = par_service.get_session_par(session_id)
    
    if par is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PAR not found for this session"
        )
    
    return par


@router.get("/queue", response_model=PARQueueResponse)
async def get_par_queue(
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    nurse: User = Depends(get_current_nurse),
    db: Session = Depends(get_db)
):
    """
    Get queue of unreviewed PARs (nurse only).
    
    Sorted by priority: red flags > urgency > creation time.
    """
    par_service = PARService(db)
    pars = par_service.get_unreviewed_pars(limit=limit, offset=offset)
    
    # Get urgency counts
    counts = par_service.count_unreviewed_by_urgency()
    
    # Convert to queue items
    queue_items = []
    session_service = SessionService(db)
    
    for par in pars:
        session = session_service.get_session(par.session_id)
        if session:
            queue_items.append(
                PARQueueItem(
                    id=par.id,
                    session_id=par.session_id,
                    patient_id=session.patient_id,
                    patient_name=session.patient.full_name,
                    chief_complaint=par.chief_complaint,
                    urgency=par.urgency,
                    has_red_flags=par.has_red_flags,
                    created_at=par.created_at
                )
            )
    
    return PARQueueResponse(
        queue=queue_items,
        total=len(queue_items),
        urgent_count=counts["urgent"],
        immediate_count=counts["immediate"],
        red_flag_count=counts["red_flags"]
    )


@router.post("/{par_id}/review", response_model=PARResponse)
async def review_par(
    par_id: UUID,
    review_data: PARReview,
    nurse: User = Depends(get_current_nurse),
    db: Session = Depends(get_db)
):
    """
    Review a PAR (nurse only).
    
    Approve or override the AI assessment.
    """
    par_service = PARService(db)
    par = par_service.review_par(par_id, nurse, review_data)
    
    if par is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PAR not found"
        )
    
    logger.info(
        f"PAR {par_id} reviewed by nurse {nurse.id} "
        f"(approved={review_data.approved})"
    )
    
    return par
