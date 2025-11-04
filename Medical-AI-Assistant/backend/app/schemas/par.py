"""
PAR (Preliminary Assessment Report) Schemas

Pydantic models for PAR-related API requests and responses.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field, field_validator

from app.models.par import PARUrgency


class PARResponse(BaseModel):
    """Schema for PAR responses"""
    id: UUID
    session_id: UUID
    chief_complaint: str
    symptoms: List[str]
    assessment: str
    urgency: PARUrgency
    rag_sources: List[dict]
    red_flags: List[str]
    has_red_flags: bool
    additional_notes: Optional[str]
    reviewed: bool
    reviewed_by_id: Optional[UUID]
    reviewed_at: Optional[datetime]
    approved: Optional[bool]
    override_reason: Optional[str]
    override_assessment: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class PARReview(BaseModel):
    """Schema for nurse PAR review"""
    approved: bool = Field(..., description="True to approve, False to override")
    override_reason: Optional[str] = Field(
        None,
        min_length=10,
        max_length=1000,
        description="Reason for override (required if approved=False)"
    )
    override_assessment: Optional[str] = Field(
        None,
        min_length=10,
        max_length=2000,
        description="Nurse's assessment (required if approved=False)"
    )
    
    @field_validator("override_reason", "override_assessment")
    @classmethod
    def validate_override_fields(cls, v, info):
        """Ensure override fields are provided when approved=False"""
        # Note: In Pydantic v2, we need to check the data dict
        if info.data.get("approved") is False and not v:
            raise ValueError(
                f"{info.field_name} is required when PAR is not approved"
            )
        return v


class PARQueueItem(BaseModel):
    """Schema for PAR queue item (simplified for nurse dashboard)"""
    id: UUID
    session_id: UUID
    patient_id: UUID
    patient_name: str
    chief_complaint: str
    urgency: PARUrgency
    has_red_flags: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class PARQueueResponse(BaseModel):
    """Schema for PAR queue response"""
    queue: List[PARQueueItem]
    total: int
    urgent_count: int
    immediate_count: int
    red_flag_count: int


class PARStatistics(BaseModel):
    """Schema for PAR statistics"""
    total_pars: int
    reviewed_count: int
    pending_count: int
    approved_count: int
    overridden_count: int
    urgency_breakdown: dict
    average_review_time_minutes: Optional[float]
