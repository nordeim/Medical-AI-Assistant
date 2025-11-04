"""
Audit Service for Medical AI Inference API
Comprehensive audit logging and compliance tracking
"""

import asyncio
import json
import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import os

from ..utils.logger import get_logger
from ..utils.exceptions import AuditLogError
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Audit event types
class AuditEventType(Enum):
    USER_AUTHENTICATION = "user_authentication"
    PHI_ACCESS = "phi_access"
    MEDICAL_OPERATION = "medical_operation"
    CLINICAL_DECISION = "clinical_decision"
    MODEL_INFERENCE = "model_inference"
    SYSTEM_EVENT = "system_event"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_EVENT = "compliance_event"
    ERROR_EVENT = "error_event"

# Event severity levels
class EventSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Complete audit event structure"""
    event_id: str
    event_type: AuditEventType
    severity: EventSeverity
    timestamp: str
    user_id: Optional[str]
    session_id: Optional[str]
    client_ip: Optional[str]
    user_agent: Optional[str]
    
    # Request details
    method: str
    path: str
    status_code: Optional[int]
    response_time_ms: Optional[float]
    
    # Medical context
    patient_id: Optional[str]
    medical_domain: Optional[str]
    phi_involved: bool
    medical_data_processed: bool
    
    # Event details
    action: str
    outcome: str
    details: Dict[str, Any]
    
    # Compliance
    compliance_flags: List[str]
    requires_review: bool
    audit_hash: str


class AuditService:
    """Comprehensive audit logging service"""
    
    def __init__(self):
        self.logger = get_logger("audit_service")
        self.db_path = "/tmp/medical_audit.db"
        self.event_buffer = []
        self.buffer_size = 100
        self.flush_interval = 30  # seconds
        self.is_initialized = False
        
        # Compliance requirements
        self.hipaa_requirements = {
            "access_controls": True,
            "audit_logs": True,
            "data_integrity": True,
            "transmission_security": True,
            "person_or_entity_authentication": True
        }
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize audit database"""
        
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        client_ip TEXT,
                        user_agent TEXT,
                        method TEXT,
                        path TEXT,
                        status_code INTEGER,
                        response_time_ms REAL,
                        patient_id TEXT,
                        medical_domain TEXT,
                        phi_involved BOOLEAN,
                        medical_data_processed BOOLEAN,
                        action TEXT,
                        outcome TEXT,
                        details TEXT,
                        compliance_flags TEXT,
                        requires_review BOOLEAN,
                        audit_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                    ON audit_events(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_user_id 
                    ON audit_events(user_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_patient_id 
                    ON audit_events(patient_id)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_audit_event_type 
                    ON audit_events(event_type)
                """)
                
                conn.commit()
            
            self.is_initialized = True
            self.logger.info("Audit database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize audit database: {e}")
            raise AuditLogError(f"Database initialization failed: {str(e)}")
    
    async def log_event(self, event: AuditEvent):
        """Log audit event"""
        
        if not self.is_initialized:
            self.logger.warning("Audit service not initialized, event will be buffered")
        
        try:
            # Calculate audit hash for integrity
            event.audit_hash = self._calculate_audit_hash(event)
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Flush if buffer is full
            if len(self.event_buffer) >= self.buffer_size:
                await self._flush_buffer()
            
            # Log to application logger
            self.logger.info(
                "Audit event logged",
                event_id=event.event_id,
                event_type=event.event_type.value,
                severity=event.severity.value,
                user_id=event.user_id,
                patient_id=event.patient_id,
                phi_involved=event.phi_involved
            )
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            # Don't raise exception to avoid breaking main flow
    
    async def log_medical_operation(
        self,
        operation: str,
        patient_id: Optional[str],
        success: bool,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        medical_domain: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: EventSeverity = EventSeverity.MEDIUM
    ):
        """Log medical operation event"""
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.MEDICAL_OPERATION,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=None,
            method="INTERNAL",
            path=f"/medical/operations/{operation}",
            status_code=200 if success else 500,
            response_time_ms=None,
            patient_id=patient_id,
            medical_domain=medical_domain,
            phi_involved=bool(patient_id),
            medical_data_processed=True,
            action=operation,
            outcome="success" if success else "failure",
            details=details or {},
            compliance_flags=["medical_data_handling"],
            requires_review=not success or severity in [EventSeverity.HIGH, EventSeverity.CRITICAL],
            audit_hash=""
        )
        
        await self.log_event(event)
    
    async def log_phi_access(
        self,
        operation: str,
        phi_type: str,
        success: bool,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log PHI access event"""
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.PHI_ACCESS,
            severity=EventSeverity.HIGH,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=None,
            method="INTERNAL",
            path=f"/phi/{operation}",
            status_code=200 if success else 403,
            response_time_ms=None,
            patient_id=patient_id,
            medical_domain=None,
            phi_involved=True,
            medical_data_processed=True,
            action=operation,
            outcome="success" if success else "unauthorized_access",
            details={
                "phi_type": phi_type,
                **(details or {})
            },
            compliance_flags=["phi_access", "hipaa_compliance"],
            requires_review=True,  # PHI access always requires review
            audit_hash=""
        )
        
        await self.log_event(event)
    
    async def log_clinical_decision(
        self,
        decision_type: str,
        confidence: float,
        recommendation: str,
        patient_id: Optional[str],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log clinical decision event"""
        
        # Determine severity based on confidence
        if confidence < 0.6:
            severity = EventSeverity.HIGH
        elif confidence < 0.8:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.CLINICAL_DECISION,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=None,
            method="INTERNAL",
            path=f"/clinical/decisions/{decision_type}",
            status_code=200,
            response_time_ms=None,
            patient_id=patient_id,
            medical_domain=details.get("medical_domain") if details else None,
            phi_involved=bool(patient_id),
            medical_data_processed=True,
            action=decision_type,
            outcome="decision_made",
            details={
                "confidence": confidence,
                "recommendation": recommendation,
                **(details or {})
            },
            compliance_flags=["clinical_decision_support", "medical_accuracy"],
            requires_review=confidence < settings.clinical_confidence_threshold,
            audit_hash=""
        )
        
        await self.log_event(event)
    
    async def log_model_inference(
        self,
        model_name: str,
        success: bool,
        latency_ms: float,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log model inference event"""
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.MODEL_INFERENCE,
            severity=EventSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=None,
            method="INTERNAL",
            path=f"/models/inference/{model_name}",
            status_code=200 if success else 500,
            response_time_ms=latency_ms,
            patient_id=details.get("patient_id") if details else None,
            medical_domain=details.get("medical_domain") if details else None,
            phi_involved=bool(details.get("patient_id")) if details else False,
            medical_data_processed=bool(details.get("medical_data_processed")) if details else False,
            action="inference",
            outcome="success" if success else "failure",
            details=details or {},
            compliance_flags=["model_governance"],
            requires_review=not success or latency_ms > 5000,  # Slow responses need review
            audit_hash=""
        )
        
        await self.log_event(event)
    
    async def log_security_event(
        self,
        event_type: str,
        severity: EventSeverity,
        description: str,
        user_id: Optional[str] = None,
        client_ip: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log security event"""
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.SECURITY_EVENT,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=user_id,
            session_id=session_id,
            client_ip=client_ip,
            user_agent=None,
            method="SECURITY",
            path=f"/security/{event_type}",
            status_code=None,
            response_time_ms=None,
            patient_id=None,
            medical_domain=None,
            phi_involved=False,
            medical_data_processed=False,
            action=event_type,
            outcome="security_event",
            details={
                "description": description,
                **(details or {})
            },
            compliance_flags=["security_monitoring"],
            requires_review=severity in [EventSeverity.HIGH, EventSeverity.CRITICAL],
            audit_hash=""
        )
        
        await self.log_event(event)
    
    async def log_compliance_event(
        self,
        compliance_type: str,
        status: str,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log compliance-related event"""
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            event_type=AuditEventType.COMPLIANCE_EVENT,
            severity=EventSeverity.MEDIUM,
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=None,
            session_id=None,
            client_ip=None,
            user_agent=None,
            method="SYSTEM",
            path=f"/compliance/{compliance_type}",
            status_code=200,
            response_time_ms=None,
            patient_id=None,
            medical_domain=None,
            phi_involved=False,
            medical_data_processed=False,
            action=compliance_type,
            outcome=status,
            details={
                "description": description,
                **(details or {})
            },
            compliance_flags=[compliance_type],
            requires_review=status != "compliant",
            audit_hash=""
        )
        
        await self.log_event(event)
    
    async def query_audit_logs(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        severity: Optional[EventSeverity] = None,
        phi_involved: Optional[bool] = None,
        requires_review: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build query
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time)
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time)
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                if patient_id:
                    query += " AND patient_id = ?"
                    params.append(patient_id)
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)
                
                if severity:
                    query += " AND severity = ?"
                    params.append(severity.value)
                
                if phi_involved is not None:
                    query += " AND phi_involved = ?"
                    params.append(phi_involved)
                
                if requires_review is not None:
                    query += " AND requires_review = ?"
                    params.append(requires_review)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                results = []
                for row in rows:
                    result = dict(row)
                    result["details"] = json.loads(result["details"]) if result["details"] else {}
                    result["compliance_flags"] = json.loads(result["compliance_flags"]) if result["compliance_flags"] else []
                    results.append(result)
                
                self.logger.debug(
                    "Audit log query completed",
                    filters={
                        "start_time": start_time,
                        "end_time": end_time,
                        "user_id": user_id,
                        "patient_id": patient_id,
                        "event_type": event_type.value if event_type else None,
                        "severity": severity.value if severity else None
                    },
                    results_count=len(results)
                )
                
                return results
                
        except Exception as e:
            self.logger.error(f"Audit log query failed: {e}")
            raise AuditLogError(f"Query failed: {str(e)}")
    
    async def get_compliance_summary(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get compliance summary statistics"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Query compliance metrics
                queries = {
                    "total_events": "SELECT COUNT(*) as count FROM audit_events",
                    "phi_access_events": "SELECT COUNT(*) as count FROM audit_events WHERE event_type = 'phi_access'",
                    "high_severity_events": "SELECT COUNT(*) as count FROM audit_events WHERE severity = 'high'",
                    "review_required_events": "SELECT COUNT(*) as count FROM audit_events WHERE requires_review = 1",
                    "security_events": "SELECT COUNT(*) as count FROM audit_events WHERE event_type = 'security_event'",
                    "medical_operations": "SELECT COUNT(*) as count FROM audit_events WHERE event_type = 'medical_operation'",
                    "clinical_decisions": "SELECT COUNT(*) as count FROM audit_events WHERE event_type = 'clinical_decision'"
                }
                
                # Add time filters if provided
                time_filter = ""
                params = []
                if start_time and end_time:
                    time_filter = " WHERE timestamp BETWEEN ? AND ?"
                    params = [start_time, end_time]
                
                for key, base_query in queries.items():
                    full_query = base_query + time_filter
                    cursor = conn.execute(full_query, params)
                    result = cursor.fetchone()
                    queries[key] = result[0] if result else 0
                
                # Calculate rates
                total_events = queries["total_events"]
                phi_rate = (queries["phi_access_events"] / max(total_events, 1)) * 100
                security_rate = (queries["security_events"] / max(total_events, 1)) * 100
                review_rate = (queries["review_required_events"] / max(total_events, 1)) * 100
                
                compliance_summary = {
                    "period": {
                        "start_time": start_time,
                        "end_time": end_time,
                        "generated_at": datetime.now(timezone.utc).isoformat()
                    },
                    "event_counts": queries,
                    "compliance_rates": {
                        "phi_access_rate": round(phi_rate, 2),
                        "security_event_rate": round(security_rate, 2),
                        "review_required_rate": round(review_rate, 2)
                    },
                    "hipaa_compliance": {
                        "phi_access_monitored": True,
                        "audit_logging_enabled": True,
                        "security_monitoring_enabled": True,
                        "access_controls_enabled": True,
                        "overall_compliance_score": self._calculate_compliance_score(queries)
                    },
                    "risk_assessment": {
                        "phi_exposure_level": "medium" if phi_rate > 10 else "low",
                        "security_risk_level": "high" if security_rate > 5 else "medium" if security_rate > 1 else "low",
                        "operational_risk_level": "medium" if review_rate > 15 else "low"
                    }
                }
                
                return compliance_summary
                
        except Exception as e:
            self.logger.error(f"Compliance summary failed: {e}")
            raise AuditLogError(f"Compliance summary failed: {str(e)}")
    
    async def _flush_buffer(self):
        """Flush event buffer to database"""
        
        if not self.event_buffer:
            return
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for event in self.event_buffer:
                    conn.execute("""
                        INSERT INTO audit_events (
                            event_id, event_type, severity, timestamp, user_id, session_id,
                            client_ip, user_agent, method, path, status_code, response_time_ms,
                            patient_id, medical_domain, phi_involved, medical_data_processed,
                            action, outcome, details, compliance_flags, requires_review, audit_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.severity.value,
                        event.timestamp,
                        event.user_id,
                        event.session_id,
                        event.client_ip,
                        event.user_agent,
                        event.method,
                        event.path,
                        event.status_code,
                        event.response_time_ms,
                        event.patient_id,
                        event.medical_domain,
                        event.phi_involved,
                        event.medical_data_processed,
                        event.action,
                        event.outcome,
                        json.dumps(event.details),
                        json.dumps(event.compliance_flags),
                        event.requires_review,
                        event.audit_hash
                    ))
                
                conn.commit()
            
            flushed_count = len(self.event_buffer)
            self.event_buffer.clear()
            
            self.logger.debug(f"Audit buffer flushed to database: {flushed_count} events")
            
        except Exception as e:
            self.logger.error(f"Failed to flush audit buffer: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = str(time.time())
        random_bytes = os.urandom(8)
        combined = timestamp.encode() + random_bytes
        return hashlib.sha256(combined).hexdigest()[:16]
    
    def _calculate_audit_hash(self, event: AuditEvent) -> str:
        """Calculate integrity hash for audit event"""
        
        # Create hashable representation
        hash_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "patient_id": event.patient_id,
            "action": event.action,
            "outcome": event.outcome,
            "details_hash": hashlib.sha256(
                json.dumps(event.details, sort_keys=True).encode()
            ).hexdigest()
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def _calculate_compliance_score(self, event_counts: Dict[str, int]) -> float:
        """Calculate overall compliance score"""
        
        total_events = event_counts.get("total_events", 1)
        
        # Penalties for concerning events
        security_penalty = min(0.3, event_counts.get("security_events", 0) / total_events * 2)
        review_penalty = min(0.2, event_counts.get("review_required_events", 0) / total_events)
        high_severity_penalty = min(0.2, event_counts.get("high_severity_events", 0) / total_events * 3)
        
        # Base score
        compliance_score = 1.0 - security_penalty - review_penalty - high_severity_penalty
        
        return max(0.0, min(1.0, compliance_score))
    
    async def cleanup(self):
        """Clean up audit service"""
        
        try:
            # Flush remaining events
            await self._flush_buffer()
            
            # Close database connection
            # SQLite connections are automatically closed when going out of scope
            
            self.logger.info("Audit service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Audit service cleanup failed: {e}")


# Global audit service instance
audit_service = AuditService()