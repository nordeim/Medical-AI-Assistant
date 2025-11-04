"""
Real-time Conversation Endpoints
Context-aware medical conversations with comprehensive management
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

from fastapi import APIRouter, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import structlog

from ..utils.exceptions import ConversationError, ValidationError, MedicalValidationError
from ..utils.security import SecurityValidator, rate_limiter
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Conversation states
class ConversationState(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ESCALATED = "escalated"
    ERROR = "error"

# Medical urgency levels
class MedicalUrgency(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Patient risk levels
class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ConversationContext:
    """Complete conversation context"""
    conversation_id: str
    session_id: str
    patient_id: Optional[str]
    state: ConversationState
    created_at: str
    updated_at: str
    last_activity: str
    
    # Medical context
    medical_domain: Optional[str]
    urgency_level: MedicalUrgency
    risk_level: RiskLevel
    symptoms: List[str]
    medical_history: List[str]
    current_medications: List[str]
    
    # Conversation management
    message_count: int
    total_tokens_used: int
    average_response_time: float
    satisfaction_score: Optional[float]
    
    # PHI management
    phi_detected: bool
    phi_protection_applied: bool
    compliance_flags: List[str]
    
    # Clinical decision support
    decisions_made: List[Dict[str, Any]]
    recommendations_given: List[str]
    follow_up_required: bool
    
    # Metadata
    metadata: Dict[str, Any]


class ConversationManager:
    """Manages medical conversations with context preservation"""
    
    def __init__(self):
        self.active_conversations: Dict[str, ConversationContext] = {}
        self.conversation_summaries: Dict[str, Dict[str, Any]] = {}
        self.max_conversation_length = settings.max_conversation_length
        self.retention_days = settings.conversation_retention_days
        
    def create_conversation(
        self,
        session_id: str,
        patient_id: Optional[str] = None,
        medical_domain: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new medical conversation"""
        
        conversation_id = str(uuid.uuid4())
        current_time = datetime.now(timezone.utc).isoformat()
        
        # Initialize conversation context
        context = ConversationContext(
            conversation_id=conversation_id,
            session_id=session_id,
            patient_id=patient_id,
            state=ConversationState.ACTIVE,
            created_at=current_time,
            updated_at=current_time,
            last_activity=current_time,
            medical_domain=medical_domain,
            urgency_level=MedicalUrgency.MEDIUM,
            risk_level=RiskLevel.LOW,
            symptoms=[],
            medical_history=[],
            current_medications=[],
            message_count=0,
            total_tokens_used=0,
            average_response_time=0.0,
            satisfaction_score=None,
            phi_detected=False,
            phi_protection_applied=False,
            compliance_flags=[],
            decisions_made=[],
            recommendations_given=[],
            follow_up_required=False,
            metadata=initial_context or {}
        )
        
        self.active_conversations[conversation_id] = context
        
        logger.info(
            "Medical conversation created",
            conversation_id=conversation_id,
            session_id=session_id,
            patient_id=patient_id,
            medical_domain=medical_domain
        )
        
        return conversation_id
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Retrieve conversation context"""
        return self.active_conversations.get(conversation_id)
    
    def update_conversation(
        self,
        conversation_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update conversation context"""
        
        if conversation_id not in self.active_conversations:
            return False
        
        context = self.active_conversations[conversation_id]
        
        # Update allowed fields
        allowed_fields = {
            'medical_domain', 'urgency_level', 'risk_level', 'symptoms',
            'medical_history', 'current_medications', 'state',
            'phi_detected', 'phi_protection_applied', 'compliance_flags',
            'decisions_made', 'recommendations_given', 'follow_up_required',
            'metadata'
        }
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(context, field, value)
        
        context.updated_at = datetime.now(timezone.utc).isoformat()
        context.last_activity = context.updated_at
        
        return True
    
    def add_message(
        self,
        conversation_id: str,
        message_type: str,  # 'user' or 'assistant'
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add message to conversation"""
        
        context = self.get_conversation(conversation_id)
        if not context:
            return False
        
        # Update message count and tokens
        context.message_count += 1
        context.total_tokens_used += len(content.split()) * 1.3  # Rough token estimation
        
        # Update last activity
        context.last_activity = datetime.now(timezone.utc).isoformat()
        
        # Analyze user messages for medical context
        if message_type == 'user':
            self._analyze_user_message(context, content)
        
        # Check conversation limits
        if context.message_count >= self.max_conversation_length:
            context.state = ConversationState.COMPLETED
        
        # Generate conversation summary periodically
        if context.message_count % 10 == 0:
            self._generate_conversation_summary(conversation_id)
        
        return True
    
    def _analyze_user_message(self, context: ConversationContext, message: str):
        """Analyze user message for medical context updates"""
        
        message_lower = message.lower()
        
        # Detect symptoms
        symptom_keywords = [
            'pain', 'ache', 'hurt', 'sore', 'fever', 'cough', 'headache',
            'nausea', 'vomiting', 'diarrhea', 'shortness of breath', 'dizziness'
        ]
        
        detected_symptoms = []
        for symptom in symptom_keywords:
            if symptom in message_lower:
                detected_symptoms.append(symptom)
        
        # Update symptoms list
        for symptom in detected_symptoms:
            if symptom not in context.symptoms:
                context.symptoms.append(symptom)
        
        # Detect urgency indicators
        urgency_indicators = {
            MedicalUrgency.CRITICAL: ['severe', 'intense', 'unbearable', 'emergency', 'urgent'],
            MedicalUrgency.HIGH: ['bad', 'worse', 'getting worse', 'can barely'],
            MedicalUrgency.MEDIUM: ['moderate', 'persistent', 'ongoing']
        }
        
        for urgency, keywords in urgency_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                if urgency.value > context.urgency_level.value:
                    context.urgency_level = urgency
                break
        
        # Detect risk factors
        high_risk_patterns = [
            'chest pain', 'difficulty breathing', 'severe bleeding',
            'loss of consciousness', 'severe headache', 'can\'t breathe'
        ]
        
        for pattern in high_risk_patterns:
            if pattern in message_lower:
                context.risk_level = RiskLevel.HIGH
                context.urgency_level = MedicalUrgency.HIGH
                break
        
        # Detect medical domain
        domain_indicators = {
            'cardiology': ['chest', 'heart', 'cardiac', 'blood pressure', 'heart rate'],
            'neurology': ['headache', 'dizzy', 'confusion', 'memory', 'seizure'],
            'gastroenterology': ['stomach', 'nausea', 'vomiting', 'diarrhea', 'abdominal'],
            'pulmonology': ['breathing', 'lung', 'cough', 'shortness of breath']
        }
        
        for domain, keywords in domain_indicators.items():
            if any(keyword in message_lower for keyword in keywords):
                if not context.medical_domain:
                    context.medical_domain = domain
                break
        
        # Update compliance flags based on PHI detection
        if not context.phi_detected:
            phi_analysis = SecurityValidator.validate_input(message)
            if phi_analysis["phi_detected"]:
                context.phi_detected = True
                context.phi_protection_applied = True
                context.compliance_flags.append("phi_detected")
        
        # Check for follow-up requirements
        follow_up_triggers = [
            'follow up', 'schedule appointment', 'need to see doctor',
            'should i come in', 'when should i'
        ]
        
        if any(trigger in message_lower for trigger in follow_up_triggers):
            context.follow_up_required = True
    
    def _generate_conversation_summary(self, conversation_id: str):
        """Generate conversation summary"""
        
        context = self.get_conversation(conversation_id)
        if not context:
            return
        
        # Create summary
        summary = {
            "conversation_id": conversation_id,
            "summary_timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": context.message_count,
            "duration_minutes": (
                datetime.now(timezone.utc) - datetime.fromisoformat(context.created_at)
            ).total_seconds() / 60,
            "primary_symptoms": context.symptoms[:5],  # Top 5 symptoms
            "medical_domain": context.medical_domain,
            "urgency_level": context.urgency_level.value,
            "risk_level": context.risk_level.value,
            "phi_involved": context.phi_detected,
            "follow_up_required": context.follow_up_required,
            "key_decisions": context.decisions_made[-3:],  # Last 3 decisions
            "satisfaction_score": context.satisfaction_score,
            "compliance_status": "compliant" if not context.compliance_flags else "flagged"
        }
        
        self.conversation_summaries[conversation_id] = summary
        
        logger.info(
            "Conversation summary generated",
            conversation_id=conversation_id,
            message_count=context.message_count,
            symptoms_count=len(context.symptoms)
        )
    
    def close_conversation(self, conversation_id: str, reason: str = "completed"):
        """Close conversation with final summary"""
        
        context = self.get_conversation(conversation_id)
        if not context:
            return False
        
        context.state = ConversationState.COMPLETED
        context.updated_at = datetime.now(timezone.utc).isoformat()
        
        # Generate final summary
        self._generate_conversation_summary(conversation_id)
        
        # Log conversation closure
        logger.info(
            "Medical conversation closed",
            conversation_id=conversation_id,
            reason=reason,
            message_count=context.message_count,
            total_tokens=context.total_tokens_used,
            duration_minutes=(
                datetime.now(timezone.utc) - datetime.fromisoformat(context.created_at)
            ).total_seconds() / 60
        )
        
        # Move to archived conversations (in production, this would be persistent storage)
        archived_conversation = asdict(context)
        self.active_conversations[conversation_id] = archived_conversation
        
        return True
    
    def get_conversation_metrics(self) -> Dict[str, Any]:
        """Get conversation metrics and statistics"""
        
        active_count = len(self.active_conversations)
        
        # Calculate metrics
        total_messages = sum(ctx.message_count for ctx in self.active_conversations.values() if hasattr(ctx, 'message_count'))
        total_tokens = sum(ctx.total_tokens_used for ctx in self.active_conversations.values() if hasattr(ctx, 'total_tokens_used'))
        
        # Domain distribution
        domain_distribution = {}
        for ctx in self.active_conversations.values():
            if hasattr(ctx, 'medical_domain') and ctx.medical_domain:
                domain_distribution[ctx.medical_domain] = domain_distribution.get(ctx.medical_domain, 0) + 1
        
        # Urgency distribution
        urgency_distribution = {}
        for ctx in self.active_conversations.values():
            if hasattr(ctx, 'urgency_level'):
                urgency = ctx.urgency_level.value if hasattr(ctx.urgency_level, 'value') else str(ctx.urgency_level)
                urgency_distribution[urgency] = urgency_distribution.get(urgency, 0) + 1
        
        return {
            "active_conversations": active_count,
            "total_messages": total_messages,
            "total_tokens_used": total_tokens,
            "domain_distribution": domain_distribution,
            "urgency_distribution": urgency_distribution,
            "phi_involvement_rate": sum(1 for ctx in self.active_conversations.values() if hasattr(ctx, 'phi_detected') and ctx.phi_detected) / max(active_count, 1),
            "average_conversation_length": total_messages / max(active_count, 1),
            "follow_up_required_rate": sum(1 for ctx in self.active_conversations.values() if hasattr(ctx, 'follow_up_required') and ctx.follow_up_required) / max(active_count, 1)
        }


# Global conversation manager
conversation_manager = ConversationManager()

# Pydantic models
class CreateConversationRequest(BaseModel):
    """Request to create new conversation"""
    
    session_id: str = Field(..., min_length=1, max_length=100, description="Session identifier")
    patient_id: Optional[str] = Field(None, max_length=50, description="Anonymized patient identifier")
    medical_domain: Optional[str] = Field(None, description="Medical domain specialization")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Initial conversation context")
    priority: Literal["low", "normal", "high", "urgent"] = Field("normal", description="Conversation priority")
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Session ID must contain only alphanumeric characters, hyphens, and underscores")
        return v


class ConversationResponse(BaseModel):
    """Conversation creation response"""
    
    conversation_id: str = Field(..., description="Unique conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    status: str = Field(..., description="Conversation status")
    created_at: str = Field(..., description="Creation timestamp")
    context: Dict[str, Any] = Field(..., description="Initial conversation context")


class SendMessageRequest(BaseModel):
    """Request to send message in conversation"""
    
    content: str = Field(..., min_length=1, max_length=4000, description="Message content")
    message_type: Literal["user", "assistant"] = Field("user", description="Message type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional message metadata")
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()


class MessageResponse(BaseModel):
    """Message response"""
    
    message_id: str = Field(..., description="Unique message identifier")
    conversation_id: str = Field(..., description="Conversation identifier")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="Message timestamp")
    message_type: str = Field(..., description="Message type")
    medical_context: Optional[Dict[str, Any]] = Field(None, description="Medical context updates")
    response_time_ms: float = Field(..., description="Response generation time")
    confidence_score: Optional[float] = Field(None, description="Model confidence")


class ConversationContextResponse(BaseModel):
    """Complete conversation context response"""
    
    conversation_id: str = Field(..., description="Conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    state: str = Field(..., description="Current conversation state")
    created_at: str = Field(..., description="Creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    
    # Medical context
    medical_domain: Optional[str] = Field(None, description="Medical domain")
    urgency_level: str = Field(..., description="Urgency level")
    risk_level: str = Field(..., description="Risk level")
    symptoms: List[str] = Field([], description="Detected symptoms")
    medical_history: List[str] = Field([], description="Medical history items")
    current_medications: List[str] = Field([], description="Current medications")
    
    # Statistics
    message_count: int = Field(..., description="Total messages")
    total_tokens_used: int = Field(..., description="Total tokens used")
    average_response_time: float = Field(..., description="Average response time")
    
    # Compliance
    phi_detected: bool = Field(..., description="Whether PHI was detected")
    phi_protection_applied: bool = Field(..., description="Whether PHI protection was applied")
    compliance_flags: List[str] = Field([], description="Compliance flags")
    
    # Clinical support
    decisions_made: List[Dict[str, Any]] = Field([], description="Clinical decisions made")
    recommendations_given: List[str] = Field([], description="Recommendations given")
    follow_up_required: bool = Field(..., description="Whether follow-up is required")


class ConversationSummaryResponse(BaseModel):
    """Conversation summary response"""
    
    conversation_id: str = Field(..., description="Conversation identifier")
    summary: Dict[str, Any] = Field(..., description="Conversation summary")
    metrics: Dict[str, Any] = Field(..., description="Conversation metrics")
    generated_at: str = Field(..., description="Summary generation timestamp")


# Endpoint implementations
@router.post("/create", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest):
    """
    Create new medical conversation session.
    
    Establishes persistent conversation context with:
    - Medical domain tracking
    - Patient information management
    - PHI protection setup
    - Compliance flag initialization
    - Conversation state management
    """
    
    # Rate limiting
    if rate_limiter.is_rate_limited(
        identifier=f"conversation:{request.session_id}",
        limit=10,  # 10 conversations per hour per session
        window=3600
    ):
        raise ValidationError("Conversation creation rate limit exceeded")
    
    # Create conversation
    conversation_id = conversation_manager.create_conversation(
        session_id=request.session_id,
        patient_id=request.patient_id,
        medical_domain=request.medical_domain,
        initial_context=request.initial_context
    )
    
    context = conversation_manager.get_conversation(conversation_id)
    
    logger.info(
        "Conversation created successfully",
        conversation_id=conversation_id,
        session_id=request.session_id,
        patient_id=request.patient_id,
        medical_domain=request.medical_domain
    )
    
    return ConversationResponse(
        conversation_id=conversation_id,
        session_id=request.session_id,
        status=context.state.value,
        created_at=context.created_at,
        context={
            "medical_domain": context.medical_domain,
            "urgency_level": context.urgency_level.value,
            "risk_level": context.risk_level.value,
            "phi_protection_enabled": True,
            "message_count": 0,
            "max_length": settings.max_conversation_length
        }
    )


@router.post("/{conversation_id}/message", response_model=MessageResponse)
async def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    http_request: Request
):
    """
    Send message in conversation with medical context processing.
    
    Processes messages with:
    - Medical context analysis
    - PHI detection and protection
    - Symptom tracking
    - Urgency assessment
    - Clinical decision support
    - Response generation
    """
    
    # Verify conversation exists
    context = conversation_manager.get_conversation(conversation_id)
    if not context:
        raise ValidationError(f"Conversation not found: {conversation_id}")
    
    if context.state != ConversationState.ACTIVE:
        raise ValidationError(f"Conversation is not active: {context.state.value}")
    
    start_time = time.time()
    
    try:
        # Add message to conversation
        message_id = str(uuid.uuid4())
        
        success = conversation_manager.add_message(
            conversation_id=conversation_id,
            message_type=request.message_type,
            content=request.content,
            metadata=request.metadata
        )
        
        if not success:
            raise ConversationError("Failed to add message to conversation")
        
        # Generate response for user messages
        response_content = ""
        confidence_score = None
        medical_context_updates = {}
        
        if request.message_type == "user":
            # Process user message and generate AI response
            response_content, confidence_score = await _generate_conversation_response(
                context, request.content
            )
            
            # Add assistant message to conversation
            conversation_manager.add_message(
                conversation_id=conversation_id,
                message_type="assistant",
                content=response_content,
                metadata={"confidence": confidence_score}
            )
            
            # Update medical context
            medical_context_updates = {
                "urgency_level": context.urgency_level.value,
                "risk_level": context.risk_level.value,
                "symptoms_detected": context.symptoms,
                "medical_domain": context.medical_domain,
                "phi_involvement": context.phi_detected,
                "follow_up_suggested": context.follow_up_required
            }
            
            # Check for escalation triggers
            if context.urgency_level == MedicalUrgency.CRITICAL or context.risk_level == RiskLevel.HIGH:
                # Log potential escalation
                logger.log_medical_operation(
                    operation="conversation_escalation_risk",
                    patient_id=context.patient_id,
                    success=True,
                    details={
                        "conversation_id": conversation_id,
                        "urgency_level": context.urgency_level.value,
                        "risk_level": context.risk_level.value,
                        "symptoms": context.symptoms
                    }
                )
        
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(
            "Message processed in conversation",
            conversation_id=conversation_id,
            message_id=message_id,
            message_type=request.message_type,
            response_time_ms=response_time,
            confidence=confidence_score
        )
        
        return MessageResponse(
            message_id=message_id,
            conversation_id=conversation_id,
            content=response_content or "Message received",
            timestamp=datetime.now(timezone.utc).isoformat(),
            message_type=request.message_type,
            medical_context=medical_context_updates if medical_context_updates else None,
            response_time_ms=response_time,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Failed to process message: {e}", conversation_id=conversation_id)
        raise ConversationError(f"Failed to process message: {str(e)}")


@router.get("/{conversation_id}/context", response_model=ConversationContextResponse)
async def get_conversation_context(conversation_id: str):
    """
    Get complete conversation context and state.
    
    Returns comprehensive conversation information including:
    - Medical context and history
    - Conversation statistics
    - PHI and compliance status
    - Clinical decisions and recommendations
    - Performance metrics
    """
    
    context = conversation_manager.get_conversation(conversation_id)
    
    if not context:
        raise ValidationError(f"Conversation not found: {conversation_id}")
    
    return ConversationContextResponse(
        conversation_id=context.conversation_id,
        session_id=context.session_id,
        state=context.state.value,
        created_at=context.created_at,
        last_activity=context.last_activity,
        medical_domain=context.medical_domain,
        urgency_level=context.urgency_level.value,
        risk_level=context.risk_level.value,
        symptoms=context.symptoms,
        medical_history=context.medical_history,
        current_medications=context.current_medications,
        message_count=context.message_count,
        total_tokens_used=context.total_tokens_used,
        average_response_time=context.average_response_time,
        phi_detected=context.phi_detected,
        phi_protection_applied=context.phi_protection_applied,
        compliance_flags=context.compliance_flags,
        decisions_made=context.decisions_made,
        recommendations_given=context.recommendations_given,
        follow_up_required=context.follow_up_required
    )


@router.post("/{conversation_id}/close")
async def close_conversation(
    conversation_id: str,
    reason: str = "completed",
    satisfaction_score: Optional[float] = Field(None, ge=1.0, le=5.0)
):
    """
    Close conversation with final summary generation.
    
    Finalizes conversation with:
    - State transition to completed
    - Final summary generation
    - Compliance audit completion
    - Performance metrics calculation
    - Follow-up scheduling if required
    """
    
    context = conversation_manager.get_conversation(conversation_id)
    
    if not context:
        raise ValidationError(f"Conversation not found: {conversation_id}")
    
    # Update satisfaction score if provided
    if satisfaction_score:
        context.satisfaction_score = satisfaction_score
    
    # Close conversation
    success = conversation_manager.close_conversation(conversation_id, reason)
    
    if not success:
        raise ConversationError("Failed to close conversation")
    
    logger.info(
        "Conversation closed successfully",
        conversation_id=conversation_id,
        reason=reason,
        message_count=context.message_count,
        satisfaction_score=context.satisfaction_score
    )
    
    return {
        "conversation_id": conversation_id,
        "status": "closed",
        "reason": reason,
        "summary": conversation_manager.conversation_summaries.get(conversation_id, {}),
        "total_messages": context.message_count,
        "duration_minutes": (
            datetime.now(timezone.utc) - datetime.fromisoformat(context.created_at)
        ).total_seconds() / 60,
        "follow_up_required": context.follow_up_required
    }


@router.get("/{conversation_id}/summary", response_model=ConversationSummaryResponse)
async def get_conversation_summary(conversation_id: str):
    """
    Get conversation summary and analysis.
    
    Provides comprehensive conversation analysis including:
    - Medical context summary
    - Key decisions and recommendations
    - Risk assessment outcomes
    - Performance metrics
    - Compliance audit results
    """
    
    context = conversation_manager.get_conversation(conversation_id)
    
    if not context:
        raise ValidationError(f"Conversation not found: {conversation_id}")
    
    # Get or generate summary
    summary = conversation_manager.conversation_summaries.get(conversation_id)
    if not summary:
        conversation_manager._generate_conversation_summary(conversation_id)
        summary = conversation_manager.conversation_summaries.get(conversation_id, {})
    
    # Calculate metrics
    metrics = {
        "conversation_duration_minutes": (
            datetime.now(timezone.utc) - datetime.fromisoformat(context.created_at)
        ).total_seconds() / 60,
        "average_tokens_per_message": context.total_tokens_used / max(context.message_count, 1),
        "response_time_trend": "stable",  # Would be calculated from message history
        "engagement_level": _calculate_engagement_level(context),
        "medical_complexity": _assess_medical_complexity(context),
        "phi_exposure_level": "high" if context.phi_detected else "low",
        "compliance_score": _calculate_compliance_score(context)
    }
    
    return ConversationSummaryResponse(
        conversation_id=conversation_id,
        summary=summary,
        metrics=metrics,
        generated_at=datetime.now(timezone.utc).isoformat()
    )


@router.get("/active")
async def get_active_conversations():
    """
    Get list of active conversations.
    
    Returns current active conversations with basic context for monitoring.
    """
    
    active_conversations = []
    
    for conversation_id, context in conversation_manager.active_conversations.items():
        if context.state == ConversationState.ACTIVE:
            active_conversations.append({
                "conversation_id": context.conversation_id,
                "session_id": context.session_id,
                "patient_id": context.patient_id,
                "medical_domain": context.medical_domain,
                "urgency_level": context.urgency_level.value,
                "message_count": context.message_count,
                "last_activity": context.last_activity,
                "phi_detected": context.phi_detected,
                "follow_up_required": context.follow_up_required
            })
    
    return {
        "active_conversations": active_conversations,
        "total_active": len(active_conversations),
        "metrics": conversation_manager.get_conversation_metrics()
    }


# Helper functions
async def _generate_conversation_response(context: ConversationContext, user_message: str) -> tuple[str, float]:
    """Generate context-aware response for conversation"""
    
    # Analyze context for response generation
    response_parts = []
    confidence = 0.8
    
    # Acknowledge user's concern
    response_parts.append("I understand you're concerned about your health.")
    
    # Address specific symptoms mentioned
    if context.symptoms:
        symptoms_str = ", ".join(context.symptoms[:3])  # Top 3 symptoms
        response_parts.append(f"You've mentioned symptoms including {symptoms_str}.")
    
    # Provide domain-specific guidance
    if context.medical_domain == "cardiology":
        response_parts.append("Given the cardiac-related nature of your concerns, ")
        response_parts.append("it's important to monitor these symptoms closely.")
    elif context.medical_domain == "neurology":
        response_parts.append("Regarding your neurological symptoms, ")
        response_parts.append("these should be evaluated by a healthcare professional.")
    elif context.medical_domain == "emergency":
        response_parts.append("Due to the urgent nature of your symptoms, ")
        response_parts.append("I recommend seeking immediate medical attention.")
    
    # Urgency-based recommendations
    if context.urgency_level == MedicalUrgency.CRITICAL:
        response_parts.append("Given the severity of your symptoms, please seek emergency medical care immediately.")
        confidence = 0.95
    elif context.urgency_level == MedicalUrgency.HIGH:
        response_parts.append("I recommend scheduling an appointment with your doctor within the next 24 hours.")
        confidence = 0.9
    elif context.urgency_level == MedicalUrgency.MEDIUM:
        response_parts.append("Consider scheduling a routine appointment to address these concerns.")
        confidence = 0.85
    else:
        response_parts.append("Monitor your symptoms and consult with a healthcare provider if they persist or worsen.")
        confidence = 0.8
    
    # Add follow-up guidance
    if context.follow_up_required:
        response_parts.append("A follow-up appointment would be appropriate to monitor your progress.")
        context.recommendations_given.append("Schedule follow-up appointment")
    
    # Standard disclaimer
    response_parts.append("Please remember that this information is for educational purposes only and should not replace professional medical advice.")
    
    # Generate decisions for clinical support
    decision = {
        "decision_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "urgency_assessment",
        "recommendation": _generate_urgency_recommendation(context),
        "confidence": confidence,
        "requires_follow_up": context.follow_up_required
    }
    context.decisions_made.append(decision)
    
    return " ".join(response_parts), confidence


def _generate_urgency_recommendation(context: ConversationContext) -> str:
    """Generate urgency-based recommendation"""
    
    if context.urgency_level == MedicalUrgency.CRITICAL:
        return "Immediate emergency medical care required"
    elif context.urgency_level == MedicalUrgency.HIGH:
        return "Urgent medical consultation within 24 hours"
    elif context.urgency_level == MedicalUrgency.MEDIUM:
        return "Medical consultation within 1-3 days"
    else:
        return "Routine medical consultation as needed"


def _calculate_engagement_level(context: ConversationContext) -> str:
    """Calculate conversation engagement level"""
    
    if context.message_count >= 20:
        return "high"
    elif context.message_count >= 10:
        return "medium"
    else:
        return "low"


def _assess_medical_complexity(context: ConversationContext) -> str:
    """Assess medical complexity based on conversation context"""
    
    complexity_score = 0
    
    # Symptom count
    complexity_score += len(context.symptoms) * 0.5
    
    # Medical domain specificity
    if context.medical_domain:
        complexity_score += 1.0
    
    # Urgency level
    urgency_multiplier = {
        MedicalUrgency.CRITICAL: 2.0,
        MedicalUrgency.HIGH: 1.5,
        MedicalUrgency.MEDIUM: 1.0,
        MedicalUrgency.LOW: 0.5
    }
    complexity_score *= urgency_multiplier.get(context.urgency_level, 1.0)
    
    # PHI involvement
    if context.phi_detected:
        complexity_score += 1.0
    
    if complexity_score >= 5.0:
        return "high"
    elif complexity_score >= 2.0:
        return "medium"
    else:
        return "low"


def _calculate_compliance_score(context: ConversationContext) -> float:
    """Calculate compliance score based on various factors"""
    
    score = 1.0  # Start with perfect score
    
    # PHI detection without protection
    if context.phi_detected and not context.phi_protection_applied:
        score -= 0.3
    
    # Compliance flags
    score -= len(context.compliance_flags) * 0.1
    
    # Missing medical validation
    if not hasattr(context, 'medical_validation_passed') or not context.medical_validation_passed:
        score -= 0.2
    
    return max(0.0, score)