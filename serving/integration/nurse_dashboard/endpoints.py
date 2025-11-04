"""
Nurse dashboard data endpoints with real-time patient monitoring.
Provides RESTful APIs for nurse dashboard functionality and WebSocket updates.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from fastapi import (
    APIRouter, 
    HTTPException, 
    Depends, 
    Query, 
    BackgroundTasks,
    Request,
    status
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog

from ...config.settings import get_settings
from ...config.logging_config import (
    get_logger, get_audit_logger, LoggingContextManager,
    request_id_var, user_id_var, session_id_var
)
from ...models.base_server import model_registry, PredictionRequest
from ..websocket.medical_chat_websocket import connection_manager
from ..streaming.sse_handler import sse_manager


# Configuration
settings = get_settings()
logger = get_logger("nurse.dashboard")
audit_logger = get_audit_logger()
security = HTTPBearer(auto_error=False)

# Router
router = APIRouter(prefix="/nurse", tags=["nurse-dashboard"])


# Enums and Models
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Urgency(str, Enum):
    ROUTINE = "routine"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class QueueStatus(str, Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    REVIEWED = "reviewed"
    ESCALATED = "escalated"


class NurseAction(str, Enum):
    APPROVE = "approved"
    OVERRIDE = "override"
    ESCALATE = "escalate"
    REQUEST_MORE_INFO = "request_more_info"


class PatientQueueItem(BaseModel):
    """Patient queue item model."""
    id: str
    session_id: str
    patient_id: str
    patient_name: Optional[str] = None
    chief_complaint: str
    symptoms: List[str]
    risk_level: RiskLevel
    urgency: Urgency
    has_red_flags: bool
    red_flags: List[str] = []
    created_at: datetime
    wait_time_minutes: int = Field(..., description="Time waiting in queue")
    priority_score: float = Field(..., description="Calculated priority score")
    status: QueueStatus = QueueStatus.PENDING
    assigned_nurse_id: Optional[str] = None
    review_notes: Optional[str] = None
    confidence_score: Optional[float] = None
    assessment_data: Dict[str, Any] = {}
    nurse_actions: List[Dict[str, Any]] = []


class NurseQueueResponse(BaseModel):
    """Nurse queue response model."""
    queue: List[PatientQueueItem]
    total: int
    urgent_count: int
    immediate_count: int
    red_flag_count: int
    avg_wait_time: float
    queue_load: str  # "low", "moderate", "high", "critical"


class NurseDashboardMetrics(BaseModel):
    """Nurse dashboard metrics model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Queue metrics
    total_queue_size: int = 0
    pending_count: int = 0
    in_review_count: int = 0
    reviewed_count: int = 0
    escalated_count: int = 0
    
    # Time metrics
    avg_wait_time_minutes: float = 0.0
    avg_review_time_minutes: float = 0.0
    longest_wait_time: int = 0
    
    # Risk and urgency metrics
    critical_count: int = 0
    high_risk_count: int = 0
    immediate_count: int = 0
    urgent_count: int = 0
    red_flag_cases: int = 0
    
    # Performance metrics
    approval_rate: float = 0.0
    override_rate: float = 0.0
    escalation_rate: float = 0.0
    nurse_utilization: float = 0.0
    
    # Real-time alerts
    critical_alerts: List[Dict[str, Any]] = []
    system_status: str = "operational"


class NurseActionRequest(BaseModel):
    """Nurse action request model."""
    action: NurseAction
    notes: Optional[str] = None
    override_reason: Optional[str] = None
    recommended_actions: List[str] = []
    priority_adjustment: Optional[float] = None
    reassign_to: Optional[str] = None
    
    @validator('override_reason')
    def validate_override_reason(cls, v, values):
        if values.get('action') == NurseAction.OVERRIDE and not v:
            raise ValueError("Override reason is required for override action")
        return v


class NurseDashboardAnalytics(BaseModel):
    """Nurse dashboard analytics model."""
    time_range: str  # "today", "week", "month"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Volume metrics
    total_assessments: int = 0
    assessments_by_hour: List[Dict[str, int]] = []
    top_complaints: List[Dict[str, int]] = []
    
    # Performance metrics
    avg_response_time: float = 0.0
    avg_completion_time: float = 0.0
    nurse_efficiency_score: float = 0.0
    
    # Quality metrics
    accuracy_score: float = 0.0
    patient_satisfaction: float = 0.0
    recommendation_adherence: float = 0.0
    
    # Trends
    trend_direction: str  # "improving", "stable", "declining"
    trend_change_percent: float = 0.0


# Mock data store (in production, this would be a database)
class MockDataStore:
    """Mock data store for demonstration purposes."""
    
    def __init__(self):
        self.patient_queue: List[PatientQueueItem] = []
        self.assessment_history: List[Dict[str, Any]] = []
        self.nurse_actions: List[Dict[str, Any]] = []
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock patient data."""
        now = datetime.utcnow()
        
        mock_patients = [
            {
                "id": "pat_001",
                "session_id": "sess_001",
                "patient_id": "patient_123",
                "patient_name": "John Doe",
                "chief_complaint": "Chest pain and shortness of breath",
                "symptoms": ["chest pain", "shortness of breath", "sweating"],
                "risk_level": RiskLevel.HIGH,
                "urgency": Urgency.URGENT,
                "has_red_flags": True,
                "red_flags": ["chest pain", "shortness of breath"],
                "created_at": now - timedelta(minutes=15),
                "confidence_score": 0.95,
                "assessment_data": {
                    "vital_signs": {"bp": "150/90", "hr": "110", "temp": "98.6"},
                    "risk_factors": ["age_45", "smoker", "family_history"]
                }
            },
            {
                "id": "pat_002",
                "session_id": "sess_002",
                "patient_id": "patient_456",
                "patient_name": "Jane Smith",
                "chief_complaint": "Headache and fatigue",
                "symptoms": ["headache", "fatigue", "difficulty concentrating"],
                "risk_level": RiskLevel.LOW,
                "urgency": Urgency.ROUTINE,
                "has_red_flags": False,
                "red_flags": [],
                "created_at": now - timedelta(minutes=45),
                "confidence_score": 0.80,
                "assessment_data": {
                    "duration": "2 days",
                    "severity": "mild to moderate"
                }
            },
            {
                "id": "pat_003",
                "session_id": "sess_003",
                "patient_id": "patient_789",
                "patient_name": "Robert Johnson",
                "chief_complaint": "Severe abdominal pain",
                "symptoms": ["severe abdominal pain", "nausea", "vomiting"],
                "risk_level": RiskLevel.CRITICAL,
                "urgency": Urgency.IMMEDIATE,
                "has_red_flags": True,
                "red_flags": ["severe pain", "vomiting"],
                "created_at": now - timedelta(minutes=5),
                "confidence_score": 0.98,
                "assessment_data": {
                    "pain_scale": "9/10",
                    "onset": "sudden",
                    "radiation": "to back"
                }
            }
        ]
        
        for patient_data in mock_patients:
            wait_time = (now - patient_data["created_at"]).total_seconds() / 60
            priority_score = self._calculate_priority_score(patient_data)
            
            queue_item = PatientQueueItem(
                **patient_data,
                wait_time_minutes=int(wait_time),
                priority_score=priority_score
            )
            self.patient_queue.append(queue_item)
    
    def _calculate_priority_score(self, patient_data: Dict[str, Any]) -> float:
        """Calculate priority score based on risk factors."""
        base_score = 0.0
        
        # Risk level multiplier
        risk_multipliers = {
            RiskLevel.CRITICAL: 10.0,
            RiskLevel.HIGH: 7.0,
            RiskLevel.MEDIUM: 4.0,
            RiskLevel.LOW: 1.0
        }
        base_score += risk_multipliers[patient_data["risk_level"]]
        
        # Urgency multiplier
        urgency_multipliers = {
            Urgency.IMMEDIATE: 5.0,
            Urgency.URGENT: 3.0,
            Urgency.ROUTINE: 1.0
        }
        base_score += urgency_multipliers[patient_data["urgency"]]
        
        # Red flag bonus
        if patient_data["has_red_flags"]:
            base_score += 3.0
        
        # Time penalty (longer wait = higher priority)
        wait_minutes = (datetime.utcnow() - patient_data["created_at"]).total_seconds() / 60
        base_score += min(wait_minutes / 10, 5.0)  # Cap at 5 points
        
        return round(base_score, 2)


# Global mock data store
mock_store = MockDataStore()


# Dependency functions
async def verify_nurse_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify nurse access permissions."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # In production, verify JWT token and check nurse role
    # For demo purposes, accept any token
    return credentials.credentials


# API Endpoints
@router.get("/queue", response_model=NurseQueueResponse)
async def get_nurse_queue(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Number of items to skip"),
    urgency: Optional[Urgency] = Query(None, description="Filter by urgency"),
    risk_level: Optional[RiskLevel] = Query(None, description="Filter by risk level"),
    has_red_flags: Optional[bool] = Query(None, description="Filter by red flag status"),
    status: Optional[QueueStatus] = Query(None, description="Filter by status"),
    assigned_nurse: Optional[str] = Query(None, description="Filter by assigned nurse"),
    credentials: str = Depends(verify_nurse_access)
):
    """
    Get patient queue for nurse dashboard.
    
    Returns paginated list of patients waiting for assessment with filtering options.
    """
    
    with LoggingContextManager(
        user_id=credentials,
        request_id=str(uuid.uuid4())
    ):
        logger.info(
            "Nurse queue request",
            nurse_id=credentials,
            filters={
                "urgency": urgency,
                "risk_level": risk_level,
                "has_red_flags": has_red_flags,
                "status": status,
                "assigned_nurse": assigned_nurse
            },
            pagination={"limit": limit, "offset": offset}
        )
        
        try:
            # Filter queue items
            filtered_queue = mock_store.patient_queue.copy()
            
            if urgency:
                filtered_queue = [item for item in filtered_queue if item.urgency == urgency]
            
            if risk_level:
                filtered_queue = [item for item in filtered_queue if item.risk_level == risk_level]
            
            if has_red_flags is not None:
                filtered_queue = [item for item in filtered_queue if item.has_red_flags == has_red_flags]
            
            if status:
                filtered_queue = [item for item in filtered_queue if item.status == status]
            
            if assigned_nurse:
                filtered_queue = [item for item in filtered_queue if item.assigned_nurse_id == assigned_nurse]
            
            # Sort by priority score (highest first)
            filtered_queue.sort(key=lambda x: x.priority_score, reverse=True)
            
            # Pagination
            total = len(filtered_queue)
            paginated_queue = filtered_queue[offset:offset + limit]
            
            # Calculate metrics
            urgent_count = len([item for item in filtered_queue if item.urgency == Urgency.URGENT])
            immediate_count = len([item for item in filtered_queue if item.urgency == Urgency.IMMEDIATE])
            red_flag_count = len([item for item in filtered_queue if item.has_red_flags])
            
            # Calculate average wait time
            avg_wait_time = (
                sum(item.wait_time_minutes for item in filtered_queue) / len(filtered_queue)
                if filtered_queue else 0.0
            )
            
            # Determine queue load
            queue_size = len(filtered_queue)
            if queue_size < 10:
                queue_load = "low"
            elif queue_size < 25:
                queue_load = "moderate"
            elif queue_size < 50:
                queue_load = "high"
            else:
                queue_load = "critical"
            
            response = NurseQueueResponse(
                queue=paginated_queue,
                total=total,
                urgent_count=urgent_count,
                immediate_count=immediate_count,
                red_flag_count=red_flag_count,
                avg_wait_time=avg_wait_time,
                queue_load=queue_load
            )
            
            # Audit log
            audit_logger.log_access(
                user_id=credentials,
                action="queue_accessed",
                resource="nurse_queue",
                details={
                    "total_results": total,
                    "filters_applied": {
                        "urgency": urgency.value if urgency else None,
                        "risk_level": risk_level.value if risk_level else None,
                        "has_red_flags": has_red_flags,
                        "status": status.value if status else None
                    }
                }
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to get nurse queue: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve nurse queue"
            )


@router.get("/dashboard/metrics", response_model=NurseDashboardMetrics)
async def get_dashboard_metrics(
    nurse_id: str = Depends(verify_nurse_access),
    real_time: bool = Query(True, description="Get real-time metrics")
):
    """
    Get real-time nurse dashboard metrics.
    
    Provides live metrics for nurse dashboard including queue status,
    performance metrics, and system alerts.
    """
    
    with LoggingContextManager(
        user_id=nurse_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Get current queue
            queue = mock_store.patient_queue
            
            # Calculate metrics
            total_queue_size = len(queue)
            pending_count = len([item for item in queue if item.status == QueueStatus.PENDING])
            in_review_count = len([item for item in queue if item.status == QueueStatus.IN_REVIEW])
            reviewed_count = len([item for item in queue if item.status == QueueStatus.REVIEWED])
            escalated_count = len([item for item in queue if item.status == QueueStatus.ESCALATED])
            
            # Time metrics
            avg_wait_time = (
                sum(item.wait_time_minutes for item in queue) / len(queue)
                if queue else 0.0
            )
            
            longest_wait = max([item.wait_time_minutes for item in queue], default=0)
            
            # Risk and urgency metrics
            critical_count = len([item for item in queue if item.risk_level == RiskLevel.CRITICAL])
            high_risk_count = len([item for item in queue if item.risk_level == RiskLevel.HIGH])
            immediate_count = len([item for item in queue if item.urgency == Urgency.IMMEDIATE])
            urgent_count = len([item for item in queue if item.urgency == Urgency.URGENT])
            red_flag_cases = len([item for item in queue if item.has_red_flags])
            
            # Performance metrics (mock data for demonstration)
            approval_rate = 0.85
            override_rate = 0.12
            escalation_rate = 0.03
            nurse_utilization = 0.78
            
            # Real-time alerts
            critical_alerts = []
            
            # Check for critical conditions
            if immediate_count > 0:
                critical_alerts.append({
                    "type": "critical_urgency",
                    "message": f"{immediate_count} immediate cases waiting",
                    "severity": "critical",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if critical_count > 0:
                critical_alerts.append({
                    "type": "critical_risk",
                    "message": f"{critical_count} critical risk cases",
                    "severity": "high",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            if longest_wait > 60:
                critical_alerts.append({
                    "type": "long_wait_time",
                    "message": f"Longest wait time: {longest_wait} minutes",
                    "severity": "medium",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Determine system status
            if immediate_count > 5 or critical_count > 10:
                system_status = "critical"
            elif immediate_count > 0 or critical_count > 0 or longest_wait > 45:
                system_status = "warning"
            else:
                system_status = "operational"
            
            metrics = NurseDashboardMetrics(
                total_queue_size=total_queue_size,
                pending_count=pending_count,
                in_review_count=in_review_count,
                reviewed_count=reviewed_count,
                escalated_count=escalated_count,
                avg_wait_time_minutes=avg_wait_time,
                avg_review_time_minutes=12.5,  # Mock data
                longest_wait_time=longest_wait,
                critical_count=critical_count,
                high_risk_count=high_risk_count,
                immediate_count=immediate_count,
                urgent_count=urgent_count,
                red_flag_cases=red_flag_cases,
                approval_rate=approval_rate,
                override_rate=override_rate,
                escalation_rate=escalation_rate,
                nurse_utilization=nurse_utilization,
                critical_alerts=critical_alerts,
                system_status=system_status
            )
            
            # If real-time updates are enabled, send via WebSocket/SSE
            if real_time:
                # Send metrics update to connected nurses
                await connection_manager.notify_nurses({
                    "type": "dashboard_metrics_update",
                    "payload": metrics.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to get dashboard metrics: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve dashboard metrics"
            )


@router.get("/queue/{patient_id}/detail")
async def get_patient_detail(
    patient_id: str,
    nurse_id: str = Depends(verify_nurse_access)
):
    """
    Get detailed patient information for nurse review.
    
    Returns comprehensive patient data including conversation history,
    assessment details, and medical context.
    """
    
    with LoggingContextManager(
        user_id=nurse_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Find patient in queue
            patient_item = next(
                (item for item in mock_store.patient_queue if item.id == patient_id),
                None
            )
            
            if not patient_item:
                raise HTTPException(
                    status_code=404,
                    detail="Patient not found in queue"
                )
            
            # Mock detailed patient data
            patient_detail = {
                "patient_info": {
                    "id": patient_item.patient_id,
                    "name": patient_item.patient_name,
                    "session_id": patient_item.session_id,
                    "chief_complaint": patient_item.chief_complaint,
                    "timestamp": patient_item.created_at.isoformat()
                },
                "assessment": {
                    "symptoms": patient_item.symptoms,
                    "risk_level": patient_item.risk_level.value,
                    "urgency": patient_item.urgency.value,
                    "confidence_score": patient_item.confidence_score,
                    "red_flags": patient_item.red_flags,
                    "has_red_flags": patient_item.has_red_flags,
                    "assessment_data": patient_item.assessment_data
                },
                "conversation_history": [
                    {
                        "timestamp": "2024-01-15T10:30:00Z",
                        "sender": "patient",
                        "content": "I've been having chest pain for about 2 hours"
                    },
                    {
                        "timestamp": "2024-01-15T10:31:00Z",
                        "sender": "agent",
                        "content": "I understand you're experiencing chest pain. Can you describe the pain on a scale of 1-10?"
                    },
                    {
                        "timestamp": "2024-01-15T10:31:30Z",
                        "sender": "patient",
                        "content": "It's about a 7 out of 10. It's crushing and I feel short of breath."
                    }
                ],
                "ai_analysis": {
                    "primary_concern": "Cardiac event possible",
                    "differential_diagnosis": [
                        "Acute coronary syndrome",
                        "Pulmonary embolism",
                        "Anxiety/panic attack"
                    ],
                    "recommended_actions": [
                        "Immediate ECG",
                        "Vital signs monitoring",
                        "Consider cardiology consult"
                    ],
                    "risk_factors": [
                        "Chest pain severity",
                        "Associated shortness of breath",
                        "Patient age (45 years)"
                    ]
                },
                "queue_info": {
                    "wait_time_minutes": patient_item.wait_time_minutes,
                    "priority_score": patient_item.priority_score,
                    "status": patient_item.status.value,
                    "assigned_nurse": patient_item.assigned_nurse_id
                }
            }
            
            # Audit log
            audit_logger.log_access(
                user_id=nurse_id,
                action="patient_detail_accessed",
                resource=f"patient:{patient_id}",
                details={
                    "session_id": patient_item.session_id,
                    "risk_level": patient_item.risk_level.value,
                    "urgency": patient_item.urgency.value
                }
            )
            
            return patient_detail
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get patient detail: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve patient details"
            )


@router.post("/queue/{patient_id}/action")
async def take_nurse_action(
    patient_id: str,
    action_request: NurseActionRequest,
    nurse_id: str = Depends(verify_nurse_access),
    background_tasks: BackgroundTasks = None
):
    """
    Take action on patient in queue.
    
    Allows nurse to approve, override, escalate, or request more information
    for a patient assessment.
    """
    
    with LoggingContextManager(
        user_id=nurse_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Find patient in queue
            patient_item = next(
                (item for item in mock_store.patient_queue if item.id == patient_id),
                None
            )
            
            if not patient_item:
                raise HTTPException(
                    status_code=404,
                    detail="Patient not found in queue"
                )
            
            # Process nurse action
            action_record = {
                "patient_id": patient_id,
                "session_id": patient_item.session_id,
                "nurse_id": nurse_id,
                "action": action_request.action.value,
                "notes": action_request.notes,
                "override_reason": action_request.override_reason,
                "recommended_actions": action_request.recommended_actions,
                "timestamp": datetime.utcnow(),
                "priority_adjustment": action_request.priority_adjustment
            }
            
            # Update patient status based on action
            if action_request.action == NurseAction.APPROVE:
                patient_item.status = QueueStatus.REVIEWED
                # Finalize assessment with nurse approval
                
            elif action_request.action == NurseAction.OVERRIDE:
                patient_item.status = QueueStatus.REVIEWED
                # Update assessment based on nurse override
                
            elif action_request.action == NurseAction.ESCALATE:
                patient_item.status = QueueStatus.ESCALATED
                # Increase priority and notify appropriate personnel
                
            elif action_request.action == NurseAction.REQUEST_MORE_INFO:
                patient_item.status = QueueStatus.PENDING
                # Send message back to patient for more information
            
            # Update queue item
            patient_item.assigned_nurse_id = nurse_id
            patient_item.review_notes = action_request.notes
            
            # Record action
            mock_store.nurse_actions.append(action_record)
            
            # Send notification to patient (if needed)
            if action_request.action == NurseAction.REQUEST_MORE_INFO:
                await connection_manager.broadcast_to_session({
                    "type": "nurse_request_more_info",
                    "payload": {
                        "session_id": patient_item.session_id,
                        "message": action_request.notes or "Nurse requests additional information.",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }, patient_item.session_id)
            
            # Update nurse dashboard for other nurses
            await connection_manager.notify_nurses({
                "type": "queue_update",
                "payload": {
                    "patient_id": patient_id,
                    "action_taken": action_request.action.value,
                    "nurse_id": nurse_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
            
            # Audit log
            background_tasks.add_task(
                audit_logger.log_access,
                user_id=nurse_id,
                action="nurse_action_taken",
                resource=f"patient:{patient_id}",
                details={
                    "action": action_request.action.value,
                    "notes": action_request.notes,
                    "override_reason": action_request.override_reason
                }
            )
            
            logger.info(
                "Nurse action completed",
                nurse_id=nurse_id,
                patient_id=patient_id,
                action=action_request.action.value
            )
            
            return {
                "success": True,
                "message": f"Patient {action_request.action.value} successfully",
                "action": action_record,
                "updated_status": patient_item.status.value
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to take nurse action: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process nurse action"
            )


@router.get("/analytics", response_model=NurseDashboardAnalytics)
async def get_nurse_analytics(
    time_range: str = Query("today", description="Time range: today, week, month"),
    nurse_id: str = Depends(verify_nurse_access)
):
    """
    Get nurse dashboard analytics.
    
    Provides historical analytics and performance metrics for nurse
    workflow optimization and quality improvement.
    """
    
    with LoggingContextManager(
        user_id=nurse_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Mock analytics data (in production, this would query historical data)
            analytics = NurseDashboardAnalytics(
                time_range=time_range,
                total_assessments=147,
                assessments_by_hour=[
                    {"hour": 8, "count": 12},
                    {"hour": 9, "count": 18},
                    {"hour": 10, "count": 22},
                    {"hour": 11, "count": 19},
                    {"hour": 12, "count": 15},
                    {"hour": 13, "count": 16},
                    {"hour": 14, "count": 21},
                    {"hour": 15, "count": 24}
                ],
                top_complaints=[
                    {"complaint": "headache", "count": 32},
                    {"complaint": "chest pain", "count": 18},
                    {"complaint": "fatigue", "count": 15},
                    {"complaint": "abdominal pain", "count": 12},
                    {"complaint": "shortness of breath", "count": 10}
                ],
                avg_response_time=4.2,
                avg_completion_time=12.8,
                nurse_efficiency_score=87.5,
                accuracy_score=92.3,
                patient_satisfaction=4.6,
                recommendation_adherence=89.2,
                trend_direction="improving",
                trend_change_percent=5.2
            )
            
            return analytics
        
        except Exception as e:
            logger.error(f"Failed to get nurse analytics: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve nurse analytics"
            )


@router.post("/queue/{patient_id}/reassign")
async def reassign_patient(
    patient_id: str,
    reassign_to: str = Query(..., description="Nurse ID to reassign to"),
    nurse_id: str = Depends(verify_nurse_access)
):
    """
    Reassign patient to another nurse.
    
    Allows workload redistribution and specialized case assignment.
    """
    
    with LoggingContextManager(
        user_id=nurse_id,
        request_id=str(uuid.uuid4())
    ):
        try:
            # Find patient in queue
            patient_item = next(
                (item for item in mock_store.patient_queue if item.id == patient_id),
                None
            )
            
            if not patient_item:
                raise HTTPException(
                    status_code=404,
                    detail="Patient not found in queue"
                )
            
            # Update assignment
            old_nurse = patient_item.assigned_nurse_id
            patient_item.assigned_nurse_id = reassign_to
            
            # Notify affected nurses
            if old_nurse:
                await connection_manager.send_personal_message({
                    "type": "patient_reassigned",
                    "payload": {
                        "patient_id": patient_id,
                        "from_nurse": old_nurse,
                        "to_nurse": reassign_to,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }, old_nurse)
            
            await connection_manager.send_personal_message({
                "type": "patient_assigned",
                "payload": {
                    "patient_id": patient_id,
                    "assigned_nurse": reassign_to,
                    "from_nurse": old_nurse,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, reassign_to)
            
            # Audit log
            audit_logger.log_access(
                user_id=nurse_id,
                action="patient_reassigned",
                resource=f"patient:{patient_id}",
                details={
                    "from_nurse": old_nurse,
                    "to_nurse": reassign_to
                }
            )
            
            return {
                "success": True,
                "message": f"Patient reassigned to nurse {reassign_to}",
                "previous_nurse": old_nurse,
                "new_nurse": reassign_to
            }
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to reassign patient: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to reassign patient"
            )


# Export router and models
__all__ = [
    "router",
    "PatientQueueItem",
    "NurseQueueResponse",
    "NurseDashboardMetrics",
    "NurseActionRequest",
    "NurseDashboardAnalytics",
    "RiskLevel",
    "Urgency",
    "QueueStatus",
    "NurseAction"
]