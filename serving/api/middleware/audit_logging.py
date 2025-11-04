"""
Audit Logging Middleware
HIPAA-compliant audit logging for all medical AI operations
"""

import json
import time
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class AuditEventType(Enum):
    """Types of audit events"""
    ACCESS = "access"
    MODIFICATION = "modification"
    DELETION = "deletion"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    PHI_ACCESS = "phi_access"
    MEDICAL_OPERATION = "medical_operation"
    CLINICAL_DECISION = "clinical_decision"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"


class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    client_ip: Optional[str]
    user_agent: Optional[str]
    method: str
    path: str
    status_code: Optional[int]
    response_time: Optional[float]
    resource_type: Optional[str]
    resource_id: Optional[str]
    action: str
    outcome: str
    details: Dict[str, Any]
    phi_accessed: bool
    medical_data_involved: bool
    error_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        return result


class AuditLogger:
    """Centralized audit logging service"""
    
    def __init__(self):
        self.logger = get_logger("audit")
        self.event_buffer = []
        self.buffer_size = 100
        self.flush_interval = 300  # 5 minutes
    
    def log_event(self, event: AuditEvent):
        """Log an audit event"""
        
        # Add to buffer for batch processing
        self.event_buffer.append(event)
        
        # Log immediately for high severity events
        if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
            self._flush_buffer()
        
        # Auto-flush buffer if full
        if len(self.event_buffer) >= self.buffer_size:
            self._flush_buffer()
        
        # Log to structured logger
        self.logger.info("audit_event", **event.to_dict())
    
    def log_request(self,
                   request: Request,
                   response: Optional[Response] = None,
                   event_type: AuditEventType = AuditEventType.ACCESS,
                   action: str = "request_processed",
                   outcome: str = "success",
                   details: Optional[Dict[str, Any]] = None,
                   severity: AuditSeverity = AuditSeverity.MEDIUM):
        """Log HTTP request event"""
        
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent")
        user_id = request.headers.get("x-user-id")
        session_id = request.headers.get("x-session-id")
        
        # Determine resource and action from path
        resource_type, resource_id = self._extract_resource_info(request.url.path)
        
        # Extract PHI and medical data indicators
        phi_accessed = self._check_phi_access(request)
        medical_data_involved = self._check_medical_data_involvement(request)
        
        # Determine severity based on content
        if phi_accessed or medical_data_involved:
            severity = AuditSeverity.HIGH
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=user_agent,
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code if response else None,
            response_time=getattr(request.state, 'processing_time', None),
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            outcome=outcome,
            details=details or {},
            phi_accessed=phi_accessed,
            medical_data_involved=medical_data_involved
        )
        
        self.log_event(event)
    
    def log_medical_operation(self,
                            operation: str,
                            patient_id: Optional[str],
                            success: bool,
                            user_id: Optional[str] = None,
                            details: Optional[Dict[str, Any]] = None):
        """Log medical operation event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.MEDICAL_OPERATION,
            severity=AuditSeverity.MEDIUM if success else AuditSeverity.HIGH,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=None,
            client_ip=None,
            user_agent=None,
            method="INTERNAL",
            path=f"/medical/operations/{operation}",
            status_code=None,
            response_time=None,
            resource_type="patient_data",
            resource_id=patient_id,
            action=operation,
            outcome="success" if success else "failure",
            details=details or {},
            phi_accessed=True,
            medical_data_involved=True
        )
        
        self.log_event(event)
    
    def log_phi_access(self,
                      operation: str,
                      phi_type: str,
                      success: bool,
                      user_id: Optional[str] = None,
                      patient_id: Optional[str] = None,
                      details: Optional[Dict[str, Any]] = None):
        """Log PHI access event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.PHI_ACCESS,
            severity=AuditSeverity.HIGH,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=None,
            client_ip=None,
            user_agent=None,
            method="INTERNAL",
            path=f"/phi/{operation}",
            status_code=None,
            response_time=None,
            resource_type="phi_data",
            resource_id=patient_id,
            action=operation,
            outcome="success" if success else "failure",
            details={
                "phi_type": phi_type,
                **(details or {})
            },
            phi_accessed=True,
            medical_data_involved=True
        )
        
        self.log_event(event)
    
    def log_clinical_decision(self,
                            decision_type: str,
                            confidence: float,
                            recommendation: str,
                            patient_id: Optional[str],
                            user_id: Optional[str] = None):
        """Log clinical decision support event"""
        
        severity = AuditSeverity.HIGH if confidence < settings.clinical_confidence_threshold else AuditSeverity.MEDIUM
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.CLINICAL_DECISION,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=None,
            client_ip=None,
            user_agent=None,
            method="INTERNAL",
            path=f"/clinical/decisions/{decision_type}",
            status_code=None,
            response_time=None,
            resource_type="clinical_decision",
            resource_id=patient_id,
            action=decision_type,
            outcome="decision_made",
            details={
                "confidence": confidence,
                "recommendation": recommendation,
                "threshold_met": confidence >= settings.clinical_confidence_threshold
            },
            phi_accessed=True,
            medical_data_involved=True
        )
        
        self.log_event(event)
    
    def log_security_event(self,
                          event_type: str,
                          severity_level: str,
                          description: str,
                          user_id: Optional[str] = None,
                          client_ip: Optional[str] = None,
                          details: Optional[Dict[str, Any]] = None):
        """Log security event"""
        
        severity = AuditSeverity(severity_level.lower())
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=None,
            client_ip=client_ip,
            user_agent=None,
            method="SYSTEM",
            path=f"/security/{event_type}",
            status_code=None,
            response_time=None,
            resource_type="security",
            resource_id=None,
            action=event_type,
            outcome="event_logged",
            details={"description": description, **(details or {})},
            phi_accessed=False,
            medical_data_involved=False
        )
        
        self.log_event(event)
    
    def log_authentication_event(self,
                               success: bool,
                               user_id: Optional[str] = None,
                               client_ip: Optional[str] = None,
                               method: str = "jwt",
                               details: Optional[Dict[str, Any]] = None):
        """Log authentication event"""
        
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=AuditEventType.AUTHENTICATION,
            severity=AuditSeverity.HIGH if not success else AuditSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=None,
            client_ip=client_ip,
            user_agent=None,
            method=method.upper(),
            path="/auth/login",
            status_code=None,
            response_time=None,
            resource_type="authentication",
            resource_id=None,
            action="login",
            outcome="success" if success else "failure",
            details=details or {},
            phi_accessed=False,
            medical_data_involved=False
        )
        
        self.log_event(event)
    
    def _flush_buffer(self):
        """Flush event buffer to persistent storage"""
        if not self.event_buffer:
            return
        
        # In production, this would write to secure, tamper-proof audit log storage
        # For now, we'll log all events from buffer
        for event in self.event_buffer:
            self.logger.info("audit_batch", **event.to_dict())
        
        self.event_buffer.clear()
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        return request.client.host if request.client else "unknown"
    
    def _extract_resource_info(self, path: str) -> tuple[Optional[str], Optional[str]]:
        """Extract resource type and ID from path"""
        
        # Parse common API patterns
        patterns = {
            r"/api/v1/inference/([^/]+)": ("inference", "model_id"),
            r"/api/v1/conversation/([^/]+)": ("conversation", "conversation_id"),
            r"/api/v1/patient/([^/]+)": ("patient", "patient_id"),
            r"/api/v1/clinical/([^/]+)": ("clinical_decision", "decision_id"),
            r"/api/v1/batch/([^/]+)": ("batch_job", "batch_id")
        }
        
        for pattern, (resource_type, resource_id) in patterns.items():
            import re
            match = re.search(pattern, path)
            if match:
                return resource_type, match.group(1)
        
        return None, None
    
    def _check_phi_access(self, request: Request) -> bool:
        """Check if request involves PHI access"""
        
        phi_indicators = [
            "patient", "phi", "medical_record", "health_info",
            "diagnosis", "treatment", "prescription"
        ]
        
        # Check path
        path_lower = str(request.url.path).lower()
        if any(indicator in path_lower for indicator in phi_indicators):
            return True
        
        # Check query parameters
        if hasattr(request, '_state') and hasattr(request._state, 'phi_analysis'):
            return request._state.phi_analysis.get('phi_detected', False)
        
        return False
    
    def _check_medical_data_involvement(self, request: Request) -> bool:
        """Check if request involves medical data"""
        
        medical_indicators = [
            "diagnosis", "treatment", "symptom", "medication",
            "vital", "test", "result", "clinical"
        ]
        
        path_lower = str(request.url.path).lower()
        if any(indicator in path_lower for indicator in medical_indicators):
            return True
        
        # Check for medical validation middleware results
        if hasattr(request, '_state') and hasattr(request._state, 'validated_data'):
            return True
        
        return False


# Global audit logger instance
audit_logger = AuditLogger()


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive audit logging"""
    
    def __init__(self, app, call_next):
        super().__init__(app)
        self.call_next = call_next
        self.audit_logger = audit_logger
    
    async def dispatch(self, request: Request, call_next):
        """Log all requests and responses"""
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add response time to request state for other middleware
            request.state.processing_time = response_time
            
            # Determine event type based on response status
            if response.status_code >= 400:
                event_type = AuditEventType.SECURITY_EVENT
                severity = AuditSeverity.HIGH
                outcome = "error"
            elif "/auth/" in str(request.url.path):
                event_type = AuditEventType.AUTHENTICATION
                severity = AuditSeverity.MEDIUM
                outcome = "success"
            elif "/phi/" in str(request.url.path):
                event_type = AuditEventType.PHI_ACCESS
                severity = AuditSeverity.HIGH
                outcome = "success"
            elif "/clinical/" in str(request.url.path):
                event_type = AuditEventType.CLINICAL_DECISION
                severity = AuditSeverity.MEDIUM
                outcome = "success"
            else:
                event_type = AuditEventType.ACCESS
                severity = AuditSeverity.LOW
                outcome = "success"
            
            # Log the request
            self.audit_logger.log_request(
                request=request,
                response=response,
                event_type=event_type,
                severity=severity,
                details={
                    "response_time_ms": round(response_time * 1000, 2),
                    "content_length": response.headers.get("content-length", "0")
                }
            )
            
            return response
            
        except Exception as e:
            # Log error
            response_time = time.time() - start_time
            
            self.audit_logger.log_request(
                request=request,
                response=None,
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=AuditSeverity.HIGH,
                outcome="error",
                details={
                    "response_time_ms": round(response_time * 1000, 2),
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            raise
        
        finally:
            # Ensure audit log is flushed
            self.audit_logger._flush_buffer()