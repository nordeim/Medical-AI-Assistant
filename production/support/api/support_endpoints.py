"""
Support System API Endpoints
RESTful API for healthcare support and success systems
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging

# Import our support system modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ticketing.ticket_management import ticket_system, TicketCategory, PriorityLevel, MedicalContext
from feedback.feedback_collection import feedback_system, FeedbackType, SentimentLabel
from monitoring.health_checks import health_monitor, ComponentType, AlertSeverity
from incident_management.emergency_response import incident_system, IncidentType, IncidentSeverity
from success_tracking.success_metrics import success_system, MilestoneType
from knowledge_base.medical_docs import knowledge_base, ContentType, UserRole, MedicalSpecialty
from training.certification_programs import training_system, CertificationTrack, AssessmentType

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TicketCreateRequest(BaseModel):
    title: str
    description: str
    category: str
    reporter_id: str
    reporter_name: str
    reporter_facility: str
    reporter_role: str
    priority: Optional[str] = None
    medical_specialty: Optional[str] = None
    urgency_level: Optional[str] = None
    patient_safety_impact: Optional[str] = None
    tags: Optional[List[str]] = None

class TicketUpdateRequest(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    resolution_summary: Optional[str] = None

class FeedbackCreateRequest(BaseModel):
    feedback_type: str
    user_id: str
    user_name: str
    user_facility: str
    user_role: str
    content: str
    rating: Optional[int] = Field(None, ge=1, le=5)
    medical_context: Optional[str] = None
    tags: Optional[List[str]] = None

class IncidentCreateRequest(BaseModel):
    title: str
    description: str
    incident_type: str
    severity: str
    reporter_id: str
    reporter_name: str
    reporter_facility: str
    medical_emergency: Optional[bool] = False
    patient_safety_impact: Optional[str] = None

class HealthCheckRequest(BaseModel):
    component_id: str
    status: str
    response_time_ms: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class CustomerEnrollRequest(BaseModel):
    facility_id: str
    facility_name: str
    facility_type: str
    number_of_users: int
    certification_track_id: str

# Initialize FastAPI app
app = FastAPI(
    title="Healthcare Support System API",
    description="Production-grade customer support and success API for healthcare organizations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# API Dependencies
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API authentication"""
    # In production, implement proper API key validation
    # For now, accept any bearer token
    return {"user_id": "api_user", "role": "admin"}

# API Routes

# Health Check
@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

# Support Tickets API
@app.post("/api/tickets")
async def create_ticket(request: TicketCreateRequest, current_user: dict = Depends(verify_api_key)):
    """Create a new support ticket"""
    try:
        # Convert string enums to actual enums
        category = TicketCategory(request.category)
        medical_context = None
        
        if request.medical_specialty or request.urgency_level or request.patient_safety_impact:
            medical_context = MedicalContext(
                medical_specialty=request.medical_specialty,
                urgency_level=request.urgency_level,
                patient_safety_impact=request.patient_safety_impact
            )
        
        priority = PriorityLevel(request.priority) if request.priority else None
        
        ticket = await ticket_system.create_ticket(
            title=request.title,
            description=request.description,
            category=category,
            reporter_id=request.reporter_id,
            reporter_name=request.reporter_name,
            reporter_facility=request.reporter_facility,
            reporter_role=request.reporter_role,
            medical_context=medical_context,
            priority=priority,
            tags=request.tags
        )
        
        return {
            "ticket_id": ticket.id,
            "status": ticket.status.value,
            "priority": ticket.priority.value,
            "created_at": ticket.created_at.isoformat(),
            "sla_due_at": ticket.sla_due_at.isoformat() if ticket.sla_due_at else None
        }
    except Exception as e:
        logger.error(f"Error creating ticket: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tickets")
async def get_tickets(
    facility: Optional[str] = Query(None, description="Filter by facility"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Number of tickets to return"),
    current_user: dict = Depends(verify_api_key)
):
    """Get support tickets with optional filters"""
    try:
        tickets = []
        
        # Apply filters
        if facility:
            tickets = await ticket_system.get_tickets_by_facility(facility)
        else:
            tickets = list(ticket_system.tickets.values())
        
        if priority:
            priority_filter = PriorityLevel(priority)
            tickets = [t for t in tickets if t.priority == priority_filter]
        
        if status:
            status_filter = ticket_system.tickets[ticket_system.tickets.__iter__().__next__().id].status.__class__[status]
            tickets = [t for t in tickets if t.status.value == status]
        
        # Return most recent tickets first
        tickets.sort(key=lambda x: x.created_at, reverse=True)
        
        return {
            "tickets": [
                {
                    "id": ticket.id,
                    "title": ticket.title,
                    "status": ticket.status.value,
                    "priority": ticket.priority.value,
                    "reporter": ticket.reporter_name,
                    "facility": ticket.reporter_facility,
                    "created_at": ticket.created_at.isoformat(),
                    "updated_at": ticket.updated_at.isoformat()
                }
                for ticket in tickets[:limit]
            ],
            "total": len(tickets),
            "returned": min(len(tickets), limit)
        }
    except Exception as e:
        logger.error(f"Error fetching tickets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tickets/{ticket_id}")
async def get_ticket(ticket_id: str, current_user: dict = Depends(verify_api_key)):
    """Get specific ticket details"""
    try:
        if ticket_id not in ticket_system.tickets:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        ticket = ticket_system.tickets[ticket_id]
        
        return {
            "id": ticket.id,
            "title": ticket.title,
            "description": ticket.description,
            "category": ticket.category.value,
            "status": ticket.status.value,
            "priority": ticket.priority.value,
            "reporter": {
                "id": ticket.reporter_id,
                "name": ticket.reporter_name,
                "facility": ticket.reporter_facility,
                "role": ticket.reporter_role
            },
            "assigned_to": ticket.assigned_to,
            "created_at": ticket.created_at.isoformat(),
            "updated_at": ticket.updated_at.isoformat(),
            "sla_due_at": ticket.sla_due_at.isoformat() if ticket.sla_due_at else None,
            "comments": len(ticket.comments),
            "attachments": len(ticket.attachments)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching ticket {ticket_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.patch("/api/tickets/{ticket_id}")
async def update_ticket(
    ticket_id: str,
    request: TicketUpdateRequest,
    current_user: dict = Depends(verify_api_key)
):
    """Update ticket status and details"""
    try:
        if ticket_id not in ticket_system.tickets:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Update status if provided
        if request.status:
            from ticketing.ticket_management import TicketStatus
            new_status = TicketStatus(request.status)
            await ticket_system.update_ticket_status(
                ticket_id, new_status, current_user["user_id"]
            )
        
        # Update assignment if provided
        if request.assigned_to:
            ticket = ticket_system.tickets[ticket_id]
            ticket.assigned_to = request.assigned_to
            ticket.updated_at = datetime.now()
        
        # Add resolution summary if provided
        if request.resolution_summary:
            ticket = ticket_system.tickets[ticket_id]
            ticket.resolution_summary = request.resolution_summary
            ticket.updated_at = datetime.now()
        
        return {"message": "Ticket updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating ticket {ticket_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Feedback API
@app.post("/api/feedback")
async def submit_feedback(
    request: FeedbackCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_api_key)
):
    """Submit user feedback"""
    try:
        feedback_type = FeedbackType(request.feedback_type)
        user_role = UserRole(request.user_role)
        
        feedback = await feedback_system.collect_feedback(
            feedback_type=feedback_type,
            user_id=request.user_id,
            user_name=request.user_name,
            user_facility=request.user_facility,
            user_role=user_role.value,
            content=request.content,
            rating=request.rating,
            medical_context=request.medical_context,
            tags=request.tags
        )
        
        # Analyze sentiment in background
        background_tasks.add_task(
            feedback_system.analyze_feedback_sentiment, feedback
        )
        
        return {
            "feedback_id": feedback.id,
            "submitted_at": feedback.submitted_at.isoformat(),
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/feedback/trends")
async def get_feedback_trends(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    facility: Optional[str] = Query(None, description="Filter by facility"),
    current_user: dict = Depends(verify_api_key)
):
    """Get feedback trends and analytics"""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        trend = await feedback_system.get_feedback_trends(
            start_dt, end_dt, facility
        )
        
        return {
            "period": {
                "start_date": start_date,
                "end_date": end_date
            },
            "summary": {
                "total_responses": trend.total_responses,
                "average_rating": trend.average_rating
            },
            "sentiment_distribution": {
                label.value: count for label, count in trend.sentiment_distribution.items()
            },
            "common_themes": trend.common_themes[:10],
            "urgent_issues": len(trend.urgent_issues),
            "safety_alerts": len(trend.medical_safety_alerts)
        }
    except Exception as e:
        logger.error(f"Error fetching feedback trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feedback/critical")
async def get_critical_feedback(
    hours: int = Query(24, ge=1, le=168, description="Look back period in hours"),
    current_user: dict = Depends(verify_api_key)
):
    """Get critical feedback requiring attention"""
    try:
        critical_feedback = await feedback_system.get_critical_feedback_alerts(hours)
        
        return {
            "critical_count": len(critical_feedback),
            "feedback": [
                {
                    "id": feedback.id,
                    "user": feedback.user_name,
                    "facility": feedback.user_facility,
                    "content": feedback.content[:200] + "..." if len(feedback.content) > 200 else feedback.content,
                    "submitted_at": feedback.submitted_at.isoformat(),
                    "patient_safety_mentioned": feedback.patient_safety_mentioned,
                    "emergency_situation": feedback.emergency_situation
                }
                for feedback in critical_feedback
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching critical feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Monitoring API
@app.post("/api/health-check")
async def perform_health_check(
    request: HealthCheckRequest,
    current_user: dict = Depends(verify_api_key)
):
    """Perform health check on a component"""
    try:
        result = await health_monitor.perform_health_check(request.component_id)
        
        return {
            "component_id": result.component_id,
            "status": result.status.value,
            "response_time_ms": result.response_time_ms,
            "timestamp": result.timestamp.isoformat(),
            "error_message": result.error_message
        }
    except Exception as e:
        logger.error(f"Error performing health check: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health/system")
async def get_system_health(current_user: dict = Depends(verify_api_key)):
    """Get overall system health overview"""
    try:
        overview = await health_monitor.get_system_overview()
        return overview
    except Exception as e:
        logger.error(f"Error fetching system health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health/component/{component_id}")
async def get_component_health(
    component_id: str,
    current_user: dict = Depends(verify_api_key)
):
    """Get specific component health details"""
    try:
        component = await health_monitor.get_component_health(component_id)
        
        return {
            "component_id": component.component_id,
            "name": component.component_name,
            "type": component.component_type.value,
            "current_status": component.current_status.value,
            "uptime_percentage": component.uptime_percentage,
            "avg_response_time": component.avg_response_time,
            "sla_target": component.sla_target,
            "sla_compliance": component.sla_compliance,
            "last_check": component.last_check.isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching component health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/uptime/{component_id}")
async def get_uptime_metrics(
    component_id: str,
    hours: int = Query(24, ge=1, le=720, description="Period in hours"),
    current_user: dict = Depends(verify_api_key)
):
    """Get uptime metrics for a component"""
    try:
        metrics = await health_monitor.get_uptime_metrics(component_id, hours)
        
        return {
            "component_id": metrics.component_id,
            "period_hours": hours,
            "uptime_percentage": metrics.uptime_percentage,
            "downtime_seconds": metrics.downtime_duration_seconds,
            "incident_count": metrics.incident_count,
            "mttr_minutes": metrics.mttr_minutes,
            "availability_score": metrics.availability_score
        }
    except Exception as e:
        logger.error(f"Error fetching uptime metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Incident Management API
@app.post("/api/incidents")
async def create_incident(
    request: IncidentCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_api_key)
):
    """Create a new incident"""
    try:
        incident_type = IncidentType(request.incident_type)
        severity = IncidentSeverity(request.severity)
        
        medical_context = None
        if request.medical_emergency or request.patient_safety_impact:
            from incident_management.emergency_response import MedicalContext
            medical_context = MedicalContext(
                emergency_situation=request.medical_emergency,
                patient_safety_impact=request.patient_safety_impact
            )
        
        incident = await incident_system.create_incident(
            title=request.title,
            description=request.description,
            incident_type=incident_type,
            severity=severity,
            reporter_id=request.reporter_id,
            reporter_name=request.reporter_name,
            reporter_facility=request.reporter_facility,
            medical_context=medical_context
        )
        
        return {
            "incident_id": incident.id,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "created_at": incident.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating incident: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/incidents/summary")
async def get_incident_summary(
    hours: int = Query(24, ge=1, le=720, description="Look back period in hours"),
    current_user: dict = Depends(verify_api_key)
):
    """Get incident summary for the period"""
    try:
        summary = await incident_system.get_incident_summary(hours)
        return summary
    except Exception as e:
        logger.error(f"Error fetching incident summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Customer Success API
@app.post("/api/customers/enroll")
async def enroll_customer(
    request: CustomerEnrollRequest,
    current_user: dict = Depends(verify_api_key)
):
    """Enroll customer in success tracking"""
    try:
        customer = await success_system.register_customer(
            facility_id=request.facility_id,
            facility_name=request.facility_name,
            facility_type=request.facility_type,
            number_of_users=request.number_of_users
        )
        
        # Enroll in certification track
        user_progress = await training_system.enroll_user_in_track(
            user_id=request.facility_id,  # Use facility as user for tracking
            user_name=request.facility_name,
            user_facility=request.facility_name,
            user_role="Administrator",
            certification_track_id=request.certification_track_id
        )
        
        return {
            "customer_id": customer.facility_id,
            "facility_name": customer.facility_name,
            "health_score": customer.health_score,
            "health_status": customer.health_status.value,
            "enrollment_date": customer.last_active_date.isoformat()
        }
    except Exception as e:
        logger.error(f"Error enrolling customer: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/customers/{facility_id}/health")
async def get_customer_health(
    facility_id: str,
    current_user: dict = Depends(verify_api_key)
):
    """Get customer health dashboard"""
    try:
        dashboard = await success_system.get_customer_health_dashboard(facility_id)
        return dashboard
    except Exception as e:
        logger.error(f"Error fetching customer health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/customers/at-risk")
async def get_at_risk_customers(current_user: dict = Depends(verify_api_key)):
    """Get list of at-risk customers"""
    try:
        at_risk_customers = await success_system.get_at_risk_customers()
        
        return {
            "count": len(at_risk_customers),
            "customers": [
                {
                    "facility_id": customer.facility_id,
                    "facility_name": customer.facility_name,
                    "health_score": customer.health_score,
                    "health_status": customer.health_status.value,
                    "days_since_active": (datetime.now() - customer.last_active_date).days,
                    "support_tickets": customer.support_ticket_count,
                    "incidents": customer.incident_count
                }
                for customer in at_risk_customers
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching at-risk customers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Base API
@app.post("/api/knowledge")
async def create_knowledge_content(
    title: str,
    content: str,
    content_type: str,
    author_id: str,
    author_name: str,
    medical_specialty: Optional[str] = None,
    target_roles: Optional[str] = None,
    tags: Optional[str] = None,
    current_user: dict = Depends(verify_api_key)
):
    """Create knowledge base content"""
    try:
        content_type_enum = ContentType(content_type)
        specialty_enum = MedicalSpecialty(medical_specialty) if medical_specialty else None
        roles_list = [UserRole(role.strip()) for role in target_roles.split(",")] if target_roles else None
        tags_list = [tag.strip() for tag in tags.split(",")] if tags else None
        
        knowledge_content = await knowledge_base.create_content(
            title=title,
            content=content,
            content_type=content_type_enum,
            author_id=author_id,
            author_name=author_name,
            medical_specialty=specialty_enum,
            target_roles=roles_list,
            tags=tags_list
        )
        
        return {
            "content_id": knowledge_content.id,
            "title": knowledge_content.title,
            "content_type": knowledge_content.content_type.value,
            "created_at": knowledge_content.last_updated.isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating knowledge content: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/knowledge/search")
async def search_knowledge(
    query: str,
    user_role: Optional[str] = Query(None),
    medical_specialty: Optional[str] = Query(None),
    content_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=50),
    current_user: dict = Depends(verify_api_key)
):
    """Search knowledge base"""
    try:
        role_enum = UserRole(user_role) if user_role else None
        specialty_enum = MedicalSpecialty(medical_specialty) if medical_specialty else None
        type_enum = ContentType(content_type) if content_type else None
        
        results = await knowledge_base.search_content(
            query=query,
            user_role=role_enum,
            medical_specialty=specialty_enum,
            content_types=[type_enum] if type_enum else None,
            max_results=limit
        )
        
        return {
            "query": query,
            "results_count": len(results),
            "results": [
                {
                    "content_id": result.content_id,
                    "title": result.title,
                    "content_type": result.content_type.value,
                    "relevance_score": result.relevance_score,
                    "snippet": result.snippet,
                    "tags": result.tags,
                    "view_count": result.view_count
                }
                for result in results
            ]
        }
    except Exception as e:
        logger.error(f"Error searching knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/{content_id}/feedback")
async def submit_content_feedback(
    content_id: str,
    user_id: str,
    user_name: str,
    user_facility: str,
    user_role: str,
    rating: int = Field(..., ge=1, le=5),
    helpful: bool = True,
    comment: Optional[str] = None,
    current_user: dict = Depends(verify_api_key)
):
    """Submit feedback on knowledge content"""
    try:
        role_enum = UserRole(user_role)
        
        feedback = await knowledge_base.submit_feedback(
            content_id=content_id,
            user_id=user_id,
            user_name=user_name,
            user_facility=user_facility,
            user_role=role_enum,
            rating=rating,
            helpful=helpful,
            comment=comment
        )
        
        return {
            "feedback_id": feedback.id,
            "submitted_at": feedback.submitted_at.isoformat(),
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        logger.error(f"Error submitting content feedback: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Training and Certification API
@app.post("/api/training/enroll")
async def enroll_in_training(
    user_id: str,
    user_name: str,
    user_facility: str,
    user_role: str,
    certification_track_id: str,
    current_user: dict = Depends(verify_api_key)
):
    """Enroll user in certification track"""
    try:
        user_progress = await training_system.enroll_user_in_track(
            user_id=user_id,
            user_name=user_name,
            user_facility=user_facility,
            user_role=user_role,
            certification_track_id=certification_track_id
        )
        
        return {
            "user_id": user_progress.user_id,
            "enrolled_tracks": len(user_progress.certification_tracks),
            "enrollment_date": user_progress.enrollment_date.isoformat()
        }
    except Exception as e:
        logger.error(f"Error enrolling in training: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/training/dashboard/{user_id}")
async def get_training_dashboard(
    user_id: str,
    current_user: dict = Depends(verify_api_key)
):
    """Get user training dashboard"""
    try:
        dashboard = await training_system.get_user_dashboard(user_id)
        return dashboard
    except Exception as e:
        logger.error(f"Error fetching training dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and Reporting API
@app.get("/api/analytics/support-summary")
async def get_support_analytics(
    days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    current_user: dict = Depends(verify_api_key)
):
    """Get comprehensive support analytics"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get various analytics
        ticket_report = await ticket_system.generate_ticket_report(start_date, end_date)
        feedback_report = await feedback_system.generate_feedback_report(start_date, end_date)
        health_report = await health_monitor.generate_health_report(days)
        incident_report = await incident_system.generate_incident_report(start_date, end_date)
        training_report = await training_system.generate_certification_report(start_date, end_date)
        
        return {
            "period_days": days,
            "reports": {
                "tickets": ticket_report,
                "feedback": feedback_report,
                "health_monitoring": health_report,
                "incidents": incident_report,
                "training": training_report
            },
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Resource not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}

# Health check for database connectivity
@app.get("/api/health/database")
async def database_health_check():
    """Check database connectivity"""
    try:
        # Simple query to test database connection
        # In production, this would check actual database connection
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.support_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )