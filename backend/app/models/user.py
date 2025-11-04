"""
User Model

Represents users of the system (patients, nurses, admins).
"""

import enum
from datetime import datetime
from typing import TYPE_CHECKING, List

from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import uuid

from app.database import Base

if TYPE_CHECKING:
    from app.models.session import Session


class UserRole(str, enum.Enum):
    """User role enumeration"""
    PATIENT = "patient"
    NURSE = "nurse"
    ADMIN = "admin"


class User(Base):
    """
    User model representing system users.
    
    Supports three roles:
    - PATIENT: Can create sessions and chat with AI
    - NURSE: Can review PARs and manage patient sessions
    - ADMIN: Full system access
    """
    
    __tablename__ = "users"
    
    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # User Information
    full_name = Column(String(255), nullable=False)
    role = Column(
        SQLEnum(UserRole),
        default=UserRole.PATIENT,
        nullable=False,
        index=True
    )
    
    # Account Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    sessions = relationship(
        "Session",
        back_populates="patient",
        foreign_keys="Session.patient_id",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"
    
    @property
    def is_patient(self) -> bool:
        """Check if user is a patient"""
        return self.role == UserRole.PATIENT
    
    @property
    def is_nurse(self) -> bool:
        """Check if user is a nurse"""
        return self.role == UserRole.NURSE
    
    @property
    def is_admin(self) -> bool:
        """Check if user is an admin"""
        return self.role == UserRole.ADMIN
    
    def has_role(self, role: UserRole) -> bool:
        """Check if user has specific role"""
        return self.role == role
    
    def can_review_pars(self) -> bool:
        """Check if user can review PARs"""
        return self.is_nurse or self.is_admin
