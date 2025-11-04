"""
FastAPI Dependencies

Common dependencies used across API routes including
database sessions, authentication, rate limiting, and more.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta

from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from app.config import settings
from app.database import get_db
from app.models.user import User, UserRole

logger = logging.getLogger(__name__)

# Security scheme for JWT authentication
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to get currently authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token from request header
        db: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError as e:
        logger.error(f"JWT validation error: {e}")
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    return user


async def get_current_active_patient(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure current user is a patient.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        User: Patient user
        
    Raises:
        HTTPException: If user is not a patient
    """
    if current_user.role != UserRole.PATIENT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only accessible to patients"
        )
    return current_user


async def get_current_nurse(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure current user is a nurse.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        User: Nurse user
        
    Raises:
        HTTPException: If user is not a nurse
    """
    if current_user.role != UserRole.NURSE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only accessible to nurses"
        )
    return current_user


async def get_current_admin(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure current user is an admin.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        User: Admin user
        
    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only accessible to administrators"
        )
    return current_user


async def verify_websocket_token(token: str, db: Session) -> Optional[User]:
    """
    Verify JWT token from WebSocket connection.
    
    Args:
        token: JWT token from query parameter
        db: Database session
        
    Returns:
        User: Authenticated user or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        
        user = db.query(User).filter(User.id == user_id).first()
        if user and user.is_active:
            return user
    except JWTError as e:
        logger.error(f"WebSocket JWT validation error: {e}")
    
    return None


# Rate limiting state (in-memory for simplicity, use Redis in production)
_rate_limit_cache: dict[str, list[datetime]] = {}


async def rate_limit_dependency(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Simple in-memory rate limiting dependency.
    
    In production, use Redis-based rate limiting.
    
    Args:
        current_user: Currently authenticated user
        
    Returns:
        User: Authenticated user
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    user_id = str(current_user.id)
    now = datetime.utcnow()
    window_start = now - timedelta(minutes=1)
    
    # Clean old entries
    if user_id in _rate_limit_cache:
        _rate_limit_cache[user_id] = [
            timestamp for timestamp in _rate_limit_cache[user_id]
            if timestamp > window_start
        ]
    else:
        _rate_limit_cache[user_id] = []
    
    # Check limit
    if len(_rate_limit_cache[user_id]) >= settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Add current request
    _rate_limit_cache[user_id].append(now)
    
    return current_user
