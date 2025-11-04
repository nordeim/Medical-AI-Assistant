"""
Role-Based Access Control (RBAC) Permissions

Handles permission checks and route protection decorators.
"""

import logging
from functools import wraps
from typing import List, Callable

from fastapi import HTTPException, status

from app.models.user import User, UserRole

logger = logging.getLogger(__name__)


class PermissionDeniedError(HTTPException):
    """Exception raised when user lacks required permissions"""
    
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


def check_role(user: User, allowed_roles: List[UserRole]) -> bool:
    """
    Check if user has one of the allowed roles.
    
    Args:
        user: User to check
        allowed_roles: List of allowed roles
        
    Returns:
        bool: True if user has required role
    """
    return user.role in allowed_roles


def require_role(*allowed_roles: UserRole):
    """
    Decorator to require specific roles for a function.
    
    Usage:
        @require_role(UserRole.NURSE, UserRole.ADMIN)
        def nurse_only_function(user: User):
            ...
    
    Args:
        allowed_roles: Roles that are allowed to access the function
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user from kwargs (assumes 'current_user' parameter)
            user = kwargs.get('current_user')
            
            if user is None:
                logger.error("Role check failed: No user provided")
                raise PermissionDeniedError("Authentication required")
            
            if not check_role(user, list(allowed_roles)):
                logger.warning(
                    f"Permission denied: User {user.id} (role={user.role}) "
                    f"attempted to access function requiring roles: {allowed_roles}"
                )
                raise PermissionDeniedError(
                    f"This action requires one of the following roles: "
                    f"{', '.join(role.value for role in allowed_roles)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def can_access_session(user: User, session_patient_id: str) -> bool:
    """
    Check if user can access a specific session.
    
    Rules:
    - Patients can only access their own sessions
    - Nurses and admins can access any session
    
    Args:
        user: User to check
        session_patient_id: UUID of session patient
        
    Returns:
        bool: True if user can access session
    """
    if user.role in [UserRole.NURSE, UserRole.ADMIN]:
        return True
    
    return str(user.id) == str(session_patient_id)


def can_review_par(user: User) -> bool:
    """
    Check if user can review PARs.
    
    Only nurses and admins can review PARs.
    
    Args:
        user: User to check
        
    Returns:
        bool: True if user can review PARs
    """
    return user.role in [UserRole.NURSE, UserRole.ADMIN]


def can_manage_users(user: User) -> bool:
    """
    Check if user can manage other users.
    
    Only admins can manage users.
    
    Args:
        user: User to check
        
    Returns:
        bool: True if user can manage users
    """
    return user.role == UserRole.ADMIN
