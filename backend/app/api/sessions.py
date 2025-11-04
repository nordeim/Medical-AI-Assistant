"""
Session API Routes

Patient session management endpoints.
"""

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.session import SessionStatus
from app.schemas.session import (
    SessionCreate,
    SessionResponse,
    SessionUpdate,
    SessionStatusUpdate
)
from app.services.session_service import SessionService
from app.dependencies import get_current_user
from app.auth.permissions import can_access_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_data: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new session for the current patient."""
    session_service = SessionService(db)
    session = session_service.create_session(session_data, current_user)
    
    # Convert to response with computed fields
    response = SessionResponse.from_orm(session)
    response.message_count = session.message_count
    response.has_par = session.has_par
    
    return response


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get session details."""
    session_service = SessionService(db)
    session = session_service.get_session(session_id)
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check access permissions
    if not can_access_session(current_user, session.patient_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this session"
        )
    
    response = SessionResponse.from_orm(session)
    response.message_count = session.message_count
    response.has_par = session.has_par
    
    return response


@router.get("", response_model=List[SessionResponse])
async def list_sessions(
    status_filter: SessionStatus = Query(None),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List sessions.
    
    Patients see only their own sessions.
    Nurses and admins see all sessions.
    """
    session_service = SessionService(db)
    
    if current_user.is_patient:
        sessions = session_service.get_patient_sessions(
            patient_id=current_user.id,
            limit=limit,
            offset=offset
        )
    else:
        # Nurses and admins see active sessions
        sessions = session_service.get_active_sessions(
            limit=limit,
            offset=offset
        )
    
    # Convert to responses
    responses = []
    for session in sessions:
        response = SessionResponse.from_orm(session)
        response.message_count = session.message_count
        response.has_par = session.has_par
        responses.append(response)
    
    return responses


@router.patch("/{session_id}/status", response_model=SessionResponse)
async def update_session_status(
    session_id: UUID,
    status_update: SessionStatusUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update session status (nurse/admin only)."""
    if not current_user.can_review_pars():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only nurses and admins can update session status"
        )
    
    session_service = SessionService(db)
    session = session_service.update_session(
        session_id,
        SessionUpdate(status=status_update.status)
    )
    
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    response = SessionResponse.from_orm(session)
    response.message_count = session.message_count
    response.has_par = session.has_par
    
    return response
