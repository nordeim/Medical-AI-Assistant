"""
Medical AI Resilience - HIPAA-Compliant Logging and Audit Trails
Comprehensive logging system with HIPAA compliance and medical audit requirements.
"""

import json
import logging
import hashlib
import hmac
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Callable, Type
from datetime import datetime, timedelta
import uuid
import threading
import asyncio
from abc import ABC, abstractmethod
import os
import gzip
import io

from .errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory,
    MedicalErrorSeverity, create_hipaa_violation_error
)


class LogLevel(Enum):
    """Log levels with medical context."""
    CRITICAL = "critical"     # Patient safety critical
    ERROR = "error"           # System errors
    WARNING = "warning"       # Warning conditions
    INFO = "info"            # General information
    DEBUG = "debug"          # Debug information
    AUDIT = "audit"          # Audit trail events
    PHI_ACCESS = "phi_access"  # PHI access events
    SECURITY = "security"     # Security events
    COMPLIANCE = "compliance" # Compliance events


class AuditEventType(Enum):
    """Types of audit events for medical compliance."""
    # Patient Access Events
    PATIENT_DATA_ACCESS = "patient_data_access"
    PATIENT_DATA_MODIFICATION = "patient_data_modification"
    PATIENT_DATA_DELETION = "patient_data_deletion"
    
    # System Access Events
    USER_AUTHENTICATION = "user_authentication"
    USER_AUTHORIZATION = "user_authorization"
    SYSTEM_ACCESS = "system_access"
    PRIVILEGED_ACCESS = "privileged_access"
    
    # Data Operations
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    DATA_BACKUP = "data_backup"
    DATA_RESTORE = "data_restore"
    
    # Security Events
    SECURITY_VIOLATION = "security_violation"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PRIVACY_BREACH = "privacy_breach"
    
    # System Events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIGURATION_CHANGE = "configuration_change"
    
    # Model Events
    MODEL_INFERENCE = "model_inference"
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    
    # Regulatory Events
    HIPAA_VIOLATION = "hipaa_violation"
    AUDIT_TRAIL_BREAK = "audit_trail_break"
    COMPLIANCE_CHECK = "compliance_check"


class PHIProtectionLevel(Enum):
    """PHI protection levels for logging."""
    NONE = "none"                    # No PHI protection (internal logs only)
    BASIC = "basic"                  # Basic PHI masking
    STANDARD = "standard"            # Standard PHI protection
    STRICT = "strict"                # Strict PHI protection (minimal logging)
    COMPLETE = "complete"            # Complete PHI removal


class AuditLogEntry:
    """Individual audit log entry with HIPAA compliance."""
    
    def __init__(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_accessed: Optional[str] = None,
        action_taken: Optional[str] = None,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        phi_protection: PHIProtectionLevel = PHIProtectionLevel.STANDARD,
        compliance_flags: Optional[List[str]] = None
    ):
        self.entry_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()
        self.event_type = event_type
        self.user_id = user_id
        self.patient_id = patient_id
        self.session_id = session_id
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.resource_accessed = resource_accessed
        self.action_taken = action_taken
        self.outcome = outcome
        self.details = details or {}
        self.phi_protection = phi_protection
        self.compliance_flags = compliance_flags or []
        
        # Audit integrity
        self.integrity_hash = self._calculate_integrity_hash()
        
        # Privacy protection
        self.protected_details = self._apply_phi_protection()
        
        # Compliance tracking
        self.hipaa_required = self._is_hipaa_required()
        self.retention_period = self._get_retention_period()
    
    def _calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for audit entry."""
        content = f"{self.timestamp.isoformat()}{self.event_type.value}{self.user_id}{self.patient_id}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _apply_phi_protection(self) -> Dict[str, Any]:
        """Apply PHI protection to details."""
        protected = self.details.copy()
        
        if self.phi_protection == PHIProtectionLevel.NONE:
            return protected
        
        phi_fields = [
            "patient_name", "ssn", "dob", "address", "phone", "email",
            "medical_record_number", "biometric_data", "diagnosis",
            "medication", "lab_results"
        ]
        
        for field in phi_fields:
            if field in protected:
                if self.phi_protection in [PHIProtectionLevel.STRICT, PHIProtectionLevel.COMPLETE]:
                    protected[field] = "[REDACTED]"
                else:
                    protected[field] = self._mask_phi_value(protected[field])
        
        return protected
    
    def _mask_phi_value(self, value: Any) -> Any:
        """Mask PHI value based on protection level."""
        if isinstance(value, str):
            if len(value) <= 4:
                return "*" * len(value)
            else:
                return value[:2] + "*" * (len(value) - 4) + value[-2:]
        return str(value)
    
    def _is_hipaa_required(self) -> bool:
        """Check if this event requires HIPAA logging."""
        hipaa_required_events = [
            AuditEventType.PATIENT_DATA_ACCESS,
            AuditEventType.PATIENT_DATA_MODIFICATION,
            AuditEventType.PATIENT_DATA_DELETION,
            AuditEventType.UNAUTHORIZED_ACCESS,
            AuditEventType.PRIVACY_BREACH,
            AuditEventType.HIPAA_VIOLATION
        ]
        return self.event_type in hipaa_required_events
    
    def _get_retention_period(self) -> timedelta:
        """Get retention period for audit entry."""
        if self.hipaa_required:
            return timedelta(days=2555)  # 7 years for HIPAA
        elif self.event_type in [
            AuditEventType.SECURITY_VIOLATION,
            AuditEventType.SYSTEM_SHUTDOWN,
            AuditEventType.CONFIGURATION_CHANGE
        ]:
            return timedelta(days=365)  # 1 year for security events
        else:
            return timedelta(days=90)  # 90 days for general events
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_accessed": self.resource_accessed,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "details": self.protected_details,
            "phi_protection_level": self.phi_protection.value,
            "integrity_hash": self.integrity_hash,
            "hipaa_required": self.hipaa_required,
            "retention_period_days": self.retention_period.days,
            "compliance_flags": self.compliance_flags
        }
    
    def to_json(self) -> str:
        """Convert audit entry to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def verify_integrity(self) -> bool:
        """Verify audit entry integrity."""
        expected_hash = self._calculate_integrity_hash()
        return hmac.compare_digest(expected_hash, self.integrity_hash)


class MedicalLogger:
    """Medical-compliant logger with PHI protection and audit trails."""
    
    def __init__(
        self,
        name: str,
        phi_protection_level: PHIProtectionLevel = PHIProtectionLevel.STANDARD,
        audit_retention_days: int = 2555,  # 7 years for HIPAA
        enable_compliance_checking: bool = True
    ):
        self.name = name
        self.phi_protection_level = phi_protection_level
        self.audit_retention_days = audit_retention_days
        self.enable_compliance_checking = enable_compliance_checking
        
        # Setup logging
        self.logger = logging.getLogger(f"medical.{name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Setup handlers
        self._setup_handlers()
        
        # Audit trail
        self.audit_entries: List[AuditLogEntry] = []
        self.audit_lock = threading.Lock()
        
        # Compliance tracking
        self.compliance_violations: List[Dict[str, Any]] = []
        self.phi_access_log: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.log_stats = {
            "total_logs": 0,
            "audit_entries": 0,
            "phi_accesses": 0,
            "compliance_violations": 0
        }
    
    def _setup_handlers(self):
        """Setup log handlers with proper formatting."""
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for audit logs
        audit_handler = logging.FileHandler(f"{self.name}_audit.log")
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        self.logger.addHandler(audit_handler)
    
    def _sanitize_log_message(self, message: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Sanitize log message to remove PHI."""
        message_str = str(message)
        context = context or {}
        
        # Apply PHI protection to message
        if self.phi_protection_level != PHIProtectionLevel.NONE:
            # Remove common PHI patterns
            phi_patterns = [
                (r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN]'),  # SSN
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
                (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),  # Phone
                (r'\b(0[1-9]|1[0-2])[-/\.](0[1-9]|[12]\d|3[01])[-/\.](19|20)\d\d\b', '[DOB]'),  # DOB
            ]
            
            for pattern, replacement in phi_patterns:
                message_str = re.sub(pattern, replacement, message_str, flags=re.IGNORECASE)
        
        # Add context if provided
        if context:
            context_str = f" [Context: {json.dumps(context, default=str)}]"
            message_str += context_str
        
        return message_str
    
    def log(
        self,
        level: LogLevel,
        message: Any,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        phi_protection: PHIProtectionLevel = None
    ):
        """Log message with medical context."""
        sanitized_message = self._sanitize_log_message(message, context)
        
        # Map to standard logging levels
        log_level = {
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.INFO: logging.INFO,
            LogLevel.DEBUG: logging.DEBUG
        }.get(level, logging.INFO)
        
        # Log the message
        extra = {
            "user_id": user_id,
            "patient_id": patient_id,
            "session_id": session_id,
            "ip_address": ip_address,
            "log_level": level.value
        }
        
        self.logger.log(log_level, sanitized_message, extra=extra)
        self.log_stats["total_logs"] += 1
        
        # Special handling for audit events
        if level in [LogLevel.AUDIT, LogLevel.PHI_ACCESS, LogLevel.COMPLIANCE]:
            self._create_audit_entry(level, message, user_id, patient_id, context)
        
        # Track PHI access
        if level == LogLevel.PHI_ACCESS and patient_id:
            self._track_phi_access(user_id, patient_id, message, context)
    
    def _create_audit_entry(
        self,
        level: LogLevel,
        message: Any,
        user_id: Optional[str],
        patient_id: Optional[str],
        context: Optional[Dict[str, Any]]
    ):
        """Create audit log entry."""
        event_type = self._map_log_level_to_audit_event(level)
        
        audit_entry = AuditLogEntry(
            event_type=event_type,
            user_id=user_id,
            patient_id=patient_id,
            session_id=context.get("session_id") if context else None,
            ip_address=context.get("ip_address") if context else None,
            resource_accessed=context.get("resource") if context else None,
            action_taken=context.get("action") if context else None,
            details=context or {},
            phi_protection=phi_protection or self.phi_protection_level
        )
        
        with self.audit_lock:
            self.audit_entries.append(audit_entry)
            self.log_stats["audit_entries"] += 1
        
        # Log audit entry
        self.logger.info(f"AUDIT: {audit_entry.to_json()}")
    
    def _map_log_level_to_audit_event(self, level: LogLevel) -> AuditEventType:
        """Map log level to audit event type."""
        mapping = {
            LogLevel.AUDIT: AuditEventType.SYSTEM_ACCESS,
            LogLevel.PHI_ACCESS: AuditEventType.PATIENT_DATA_ACCESS,
            LogLevel.COMPLIANCE: AuditEventType.COMPLIANCE_CHECK,
            LogLevel.CRITICAL: AuditEventType.SECURITY_VIOLATION,
            LogLevel.ERROR: AuditEventType.SYSTEM_ACCESS,
            LogLevel.SECURITY: AuditEventType.SECURITY_VIOLATION
        }
        return mapping.get(level, AuditEventType.SYSTEM_ACCESS)
    
    def _track_phi_access(self, user_id: str, patient_id: str, action: Any, context: Optional[Dict[str, Any]]):
        """Track PHI access for compliance."""
        phi_access = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "patient_id": patient_id,
            "action": str(action),
            "context": context or {}
        }
        
        with self.audit_lock:
            self.phi_access_log.append(phi_access)
            self.log_stats["phi_accesses"] += 1
    
    def audit_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_accessed: Optional[str] = None,
        action_taken: Optional[str] = None,
        outcome: str = "success",
        details: Optional[Dict[str, Any]] = None,
        phi_protection: PHIProtectionLevel = None
    ):
        """Create explicit audit event."""
        audit_entry = AuditLogEntry(
            event_type=event_type,
            user_id=user_id,
            patient_id=patient_id,
            session_id=session_id,
            ip_address=ip_address,
            resource_accessed=resource_accessed,
            action_taken=action_taken,
            outcome=outcome,
            details=details or {},
            phi_protection=phi_protection or self.phi_protection_level
        )
        
        with self.audit_lock:
            self.audit_entries.append(audit_entry)
            self.log_stats["audit_entries"] += 1
        
        # Log audit event
        self.logger.info(f"AUDIT_EVENT: {audit_entry.to_json()}")
        
        # Check compliance
        if self.enable_compliance_checking:
            self._check_compliance(audit_entry)
    
    def _check_compliance(self, audit_entry: AuditLogEntry):
        """Check compliance with regulations."""
        violations = []
        
        # HIPAA compliance checks
        if audit_entry.hipaa_required:
            # Check required fields
            required_fields = ["user_id", "timestamp", "event_type"]
            for field in required_fields:
                if not getattr(audit_entry, field):
                    violations.append(f"Missing required HIPAA field: {field}")
            
            # Check retention
            if audit_entry.retention_period < timedelta(days=2555):
                violations.append(f"Insufficient retention period: {audit_entry.retention_period}")
        
        # PHI protection checks
        if audit_entry.patient_id and audit_entry.event_type in [
            AuditEventType.PATIENT_DATA_ACCESS,
            AuditEventType.PATIENT_DATA_MODIFICATION
        ]:
            if audit_entry.phi_protection == PHIProtectionLevel.NONE:
                violations.append("PHI data logged without protection")
        
        if violations:
            self.compliance_violations.extend(violations)
            self.log_stats["compliance_violations"] += 1
            
            # Log violation
            self.logger.warning(f"COMPLIANCE_VIOLATION: {violations}")
    
    def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditLogEntry]:
        """Get filtered audit trail."""
        filtered_entries = self.audit_entries.copy()
        
        # Apply filters
        if user_id:
            filtered_entries = [e for e in filtered_entries if e.user_id == user_id]
        
        if patient_id:
            filtered_entries = [e for e in filtered_entries if e.patient_id == patient_id]
        
        if event_type:
            filtered_entries = [e for e in filtered_entries if e.event_type == event_type]
        
        if start_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp >= start_time]
        
        if end_time:
            filtered_entries = [e for e in filtered_entries if e.timestamp <= end_time]
        
        # Sort by timestamp (most recent first) and limit
        filtered_entries.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_entries[:limit]
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Get compliance report."""
        with self.audit_lock:
            total_audit_entries = len(self.audit_entries)
            hipaa_entries = sum(1 for e in self.audit_entries if e.hipaa_required)
            phi_accesses = len(self.phi_access_log)
            
            # Retention analysis
            now = datetime.utcnow()
            expired_entries = sum(
                1 for e in self.audit_entries 
                if now - e.timestamp > e.retention_period
            )
            
            # Integrity verification
            integrity_violations = sum(
                1 for e in self.audit_entries 
                if not e.verify_integrity()
            )
        
        return {
            "total_audit_entries": total_audit_entries,
            "hipaa_required_entries": hipaa_entries,
            "phi_access_events": phi_accesses,
            "compliance_violations": self.compliance_violations,
            "expired_entries": expired_entries,
            "integrity_violations": integrity_violations,
            "retention_compliance": (total_audit_entries - expired_entries) / max(total_audit_entries, 1),
            "statistics": self.log_stats,
            "audit_trailing_status": "compliant" if not self.compliance_violations else "violations_detected"
        }
    
    def archive_old_logs(self, days_to_keep: int = 90):
        """Archive old audit logs."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        with self.audit_lock:
            old_entries = [e for e in self.audit_entries if e.timestamp < cutoff_date]
            current_entries = [e for e in self.audit_entries if e.timestamp >= cutoff_date]
            
            self.audit_entries = current_entries
        
        if old_entries:
            self._archive_entries(old_entries)
    
    def _archive_entries(self, entries: List[AuditLogEntry]):
        """Archive audit entries to compressed storage."""
        try:
            # Create archive filename
            archive_name = f"audit_archive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json.gz"
            
            # Convert to JSON
            entries_data = [entry.to_dict() for entry in entries]
            json_data = json.dumps(entries_data, default=str)
            
            # Compress and write
            with gzip.open(archive_name, 'wt', encoding='utf-8') as f:
                f.write(json_data)
            
            self.logger.info(f"Archived {len(entries)} audit entries to {archive_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to archive audit entries: {e}")
    
    def cleanup_expired_entries(self):
        """Remove expired audit entries."""
        now = datetime.utcnow()
        
        with self.audit_lock:
            expired_count = 0
            
            # Clean audit entries
            initial_count = len(self.audit_entries)
            self.audit_entries = [
                e for e in self.audit_entries 
                if now - e.timestamp <= e.retention_period
            ]
            expired_count += initial_count - len(self.audit_entries)
            
            # Clean PHI access log (keep last 90 days)
            phi_cutoff = now - timedelta(days=90)
            initial_phi_count = len(self.phi_access_log)
            self.phi_access_log = [
                e for e in self.phi_access_log 
                if datetime.fromisoformat(e['timestamp']) > phi_cutoff
            ]
            expired_count += initial_phi_count - len(self.phi_access_log)
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired log entries")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        with self.audit_lock:
            return {
                "logger_name": self.name,
                "phi_protection_level": self.phi_protection_level.value,
                "audit_retention_days": self.audit_retention_days,
                "total_audit_entries": len(self.audit_entries),
                "total_phi_accesses": len(self.phi_access_log),
                "compliance_violations": len(self.compliance_violations),
                "statistics": self.log_stats.copy(),
                "oldest_entry": min(e.timestamp for e in self.audit_entries).isoformat() if self.audit_entries else None,
                "newest_entry": max(e.timestamp for e in self.audit_entries).isoformat() if self.audit_entries else None
            }


# Global medical logger instance
medical_logger = MedicalLogger("medical_ai_system")


def setup_medical_logging(
    name: str = "medical_ai_system",
    phi_protection_level: PHIProtectionLevel = PHIProtectionLevel.STANDARD,
    audit_retention_days: int = 2555
) -> MedicalLogger:
    """Setup medical logging system."""
    global medical_logger
    medical_logger = MedicalLogger(name, phi_protection_level, audit_retention_days)
    return medical_logger


def get_medical_logger() -> MedicalLogger:
    """Get the global medical logger instance."""
    return medical_logger


# Convenience logging functions
def log_patient_access(
    user_id: str,
    patient_id: str,
    action: str,
    session_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    outcome: str = "success"
):
    """Log patient data access."""
    medical_logger.audit_event(
        event_type=AuditEventType.PATIENT_DATA_ACCESS,
        user_id=user_id,
        patient_id=patient_id,
        session_id=session_id,
        ip_address=ip_address,
        action_taken=action,
        outcome=outcome
    )


def log_security_event(
    event_type: str,
    details: Dict[str, Any],
    severity: str = "medium"
):
    """Log security event."""
    medical_logger.log(
        level=LogLevel.SECURITY,
        message=f"Security event: {event_type}",
        context=details,
        phi_protection=PHIProtectionLevel.NONE
    )


def log_hipaa_violation(
    violation_type: str,
    details: Dict[str, Any],
    user_id: Optional[str] = None,
    patient_id: Optional[str] = None
):
    """Log HIPAA violation."""
    medical_logger.audit_event(
        event_type=AuditEventType.HIPAA_VIOLATION,
        user_id=user_id,
        patient_id=patient_id,
        action_taken=violation_type,
        outcome="violation",
        details=details,
        phi_protection=PHIProtectionLevel.STRICT
    )
    
    medical_logger.log(
        level=LogLevel.CRITICAL,
        message=f"HIPAA VIOLATION: {violation_type}",
        user_id=user_id,
        patient_id=patient_id,
        context=details
    )


# Context manager for audit logging
class AuditContext:
    """Context manager for automatic audit logging."""
    
    def __init__(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        resource_accessed: Optional[str] = None,
        **kwargs
    ):
        self.event_type = event_type
        self.user_id = user_id
        self.patient_id = patient_id
        self.resource_accessed = resource_accessed
        self.kwargs = kwargs
        self.start_time = datetime.utcnow()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.utcnow()
        duration = (end_time - self.start_time).total_seconds()
        
        outcome = "success" if exc_type is None else "failure"
        action_taken = f"{self.event_type.value}_completed"
        
        details = {
            "duration_seconds": duration,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        
        if exc_type:
            details["error"] = str(exc_val)
        
        details.update(self.kwargs)
        
        medical_logger.audit_event(
            event_type=self.event_type,
            user_id=self.user_id,
            patient_id=self.patient_id,
            resource_accessed=self.resource_accessed,
            action_taken=action_taken,
            outcome=outcome,
            details=details
        )


# Decorators for automatic logging
def log_medical_operation(operation_name: str, audit_event_type: AuditEventType):
    """Decorator for automatic audit logging of medical operations."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with AuditContext(
                event_type=audit_event_type,
                resource_accessed=operation_name,
                function_name=func.__name__
            ):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with AuditContext(
                event_type=audit_event_type,
                resource_accessed=operation_name,
                function_name=func.__name__
            ):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator