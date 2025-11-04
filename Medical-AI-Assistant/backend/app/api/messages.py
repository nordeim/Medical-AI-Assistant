"""
Message API Routes

Message history and retrieval endpoints.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.schemas.message import MessageListResponse
from app.services.message_service import MessageService
from app.services.session_service import SessionService
from app.dependencies import get_current_user
from app.auth.permissions import can_access_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions/{session_id}/messages", tags=["Messages"])


@router.get("", response_model=MessageListResponse)
async def get_session_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all messages in a session.
    
    Returns message history ordered by sequence number.
    """
    # Verify session exists
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
    
    # Get messages
    message_service = MessageService(db)
    messages = message_service.get_session_messages(session_id)
    
    return MessageListResponse(
        messages=messages,
        total=len(messages),
        session_id=session_id
    )
