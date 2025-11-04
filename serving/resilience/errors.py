"""
Medical AI Resilience - Error Categories and Handling
Comprehensive error handling system with medical-specific error categories.
"""

from enum import Enum, auto
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import uuid


class MedicalErrorSeverity(Enum):
    """Medical error severity levels for proper triage and response."""
    CRITICAL = "critical"      # Immediate threat to patient safety
    HIGH = "high"              # Significant impact on care quality
    MEDIUM = "medium"          # Moderate impact, requires attention
    LOW = "low"               # Minor issues, log for monitoring
    INFO = "info"             # Informational, normal operations


class MedicalErrorCategory(Enum):
    """Medical-specific error categories for proper classification."""
    # Patient Safety Errors (Critical)
    PATIENT_SAFETY = "patient_safety"           # Potential harm to patient
    CRITICAL_CLINICAL = "critical_clinical"     # Critical medical decisions
    
    # Data Integrity Errors
    DATA_CORRUPTION = "data_corruption"         # Medical data integrity issues
    PHI_VIOLATION = "phi_violation"             # Protected Health Information issues
    AUDIT_TRAIL_BREAK = "audit_trail_break"     # Audit trail discontinuity
    
    # Model Performance Errors
    MODEL_DEGRADATION = "model_degradation"     # Model performance below standards
    PREDICTION_UNCERTAINTY = "prediction_uncertainty"  # High uncertainty in predictions
    MODEL_FAILURE = "model_failure"             # Complete model failure
    
    # System Infrastructure Errors
    SERVICE_UNAVAILABLE = "service_unavailable"    # Service completely unavailable
    PERFORMANCE_DEGRADATION = "performance_degradation"  # Performance below threshold
    RESOURCE_EXHAUSTION = "resource_exhaustion"    # System resources exhausted
    
    # Integration and Dependencies
    EXTERNAL_SERVICE_FAILURE = "external_service_failure"  # External dependency failure
    API_TIMEOUT = "api_timeout"                      # External API timeout
    DATA_SOURCE_UNAVAILABLE = "data_source_unavailable"  # Data source unavailable
    
    # Regulatory and Compliance
    HIPAA_VIOLATION = "hipaa_violation"             # HIPAA compliance violation
    REGULATORY_VIOLATION = "regulatory_violation"   # Other regulatory violations
    AUDIT_COMPLIANCE = "audit_compliance"          # Audit trail compliance issues
    
    # Business Logic Errors
    CLINICAL_LOGIC_ERROR = "clinical_logic_error"   # Clinical reasoning errors
    PROTOCOL_VIOLATION = "protocol_violation"      # Medical protocol violations
    AUTHORIZATION_ERROR = "authorization_error"     # Access control violations


class MedicalErrorCode(Enum):
    """Specific medical error codes for detailed tracking."""
    # Patient Safety (1000-1099)
    E1001 = "E1001"  # Patient safety alert triggered
    E1002 = "E1002"  # Critical clinical decision blocked
    E1003 = "E1003"  # Emergency override required
    
    # Data Integrity (2000-2099)
    E2001 = "E2001"  # PHI data corruption detected
    E2002 = "E2002"  # Medical record inconsistent
    E2003 = "E2003"  # Data validation failed
    E2004 = "E2004"  # Audit trail corruption
    
    # Model Performance (3000-3099)
    E3001 = "E3001"  # Model confidence below threshold
    E3002 = "E3002"  # Model prediction inconsistent
    E3003 = "E3003"  # Model not responding
    E3004 = "E3004"  # Model performance degraded
    
    # System Infrastructure (4000-4099)
    E4001 = "E4001"  # Service unavailable
    E4002 = "E4002"  # Resource exhaustion
    E4003 = "E4003"  # Performance below SLA
    E4004 = "E4004"  # Memory/CPU limit exceeded
    
    # External Dependencies (5000-5099)
    E5001 = "E5001"  # External API timeout
    E5002 = "E5002"  # Database connection failed
    E5003 = "E5003"  # Cache service unavailable
    E5004 = "E5004"  # Message queue failure
    
    # Regulatory (6000-6099)
    E6001 = "E6001"  # HIPAA compliance violation
    E6002 = "E6002"  # Audit trail incomplete
    E6003 = "E6003"  # Data retention violation
    E6004 = "E6004"  # Access control violation


class MedicalError:
    """Comprehensive medical error representation with full context."""
    
    def __init__(
        self,
        error_code: MedicalErrorCode,
        category: MedicalErrorCategory,
        severity: MedicalErrorSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        patient_context: Optional[Dict[str, Any]] = None,
        technical_context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        suggested_action: Optional[str] = None,
        escalation_required: bool = False,
        timestamp: Optional[datetime] = None
    ):
        self.error_id = str(uuid.uuid4())
        self.timestamp = timestamp or datetime.utcnow()
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.message = message
        self.details = details or {}
        self.patient_context = patient_context or {}
        self.technical_context = technical_context or {}
        self.recoverable = recoverable
        self.suggested_action = suggested_action
        self.escalation_required = escalation_required
        self.occurrences = 1
        self.last_occurrence = self.timestamp
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'error_code': self.error_code.value,
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'patient_context': self.patient_context,
            'technical_context': self.technical_context,
            'recoverable': self.recoverable,
            'suggested_action': self.suggested_action,
            'escalation_required': self.escalation_required,
            'occurrences': self.occurrences,
            'last_occurrence': self.last_occurrence.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert error to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def increment_occurrence(self):
        """Increment occurrence count and update last occurrence."""
        self.occurrences += 1
        self.last_occurrence = datetime.utcnow()
    
    def is_critical(self) -> bool:
        """Check if this is a critical error requiring immediate attention."""
        return self.severity == MedicalErrorSeverity.CRITICAL
    
    def requires_immediate_response(self) -> bool:
        """Check if error requires immediate response."""
        return (
            self.is_critical() or 
            self.category == MedicalErrorCategory.PATIENT_SAFETY or
            self.category == MedicalErrorCategory.CRITICAL_CLINICAL or
            self.escalation_required
        )
    
    def should_alert_medical_staff(self) -> bool:
        """Check if error should alert medical staff."""
        return (
            self.category in [
                MedicalErrorCategory.PATIENT_SAFETY,
                MedicalErrorCategory.CRITICAL_CLINICAL,
                MedicalErrorCategory.PHI_VIOLATION
            ] or
            self.severity in [MedicalErrorSeverity.CRITICAL, MedicalErrorSeverity.HIGH]
        )


class MedicalErrorHandler:
    """Centralized medical error handling with proper categorization and response."""
    
    def __init__(self, alert_callback=None, audit_callback=None):
        self.alert_callback = alert_callback
        self.audit_callback = audit_callback
        self.error_history: List[MedicalError] = []
        self.error_counts: Dict[str, int] = {}
        self._callbacks = []
    
    def register_callback(self, callback):
        """Register callback for error handling notifications."""
        self._callbacks.append(callback)
    
    def handle_error(
        self,
        error_code: MedicalErrorCode,
        category: MedicalErrorCategory,
        severity: MedicalErrorSeverity,
        message: str,
        **kwargs
    ) -> MedicalError:
        """Handle and process medical error with proper escalation."""
        
        error = MedicalError(
            error_code=error_code,
            category=category,
            severity=severity,
            message=message,
            **kwargs
        )
        
        # Check for recurring errors
        error_key = f"{error_code.value}_{category.value}"
        if error_key in self.error_counts:
            error.increment_occurrence()
            self.error_counts[error_key] += 1
        else:
            self.error_counts[error_key] = 1
        
        # Add to history
        self.error_history.append(error)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(error)
            except Exception as e:
                # Log callback error but don't fail
                print(f"Error handler callback failed: {e}")
        
        # Handle critical errors
        if error.requires_immediate_response():
            self._handle_critical_error(error)
        
        # Audit trail
        if self.audit_callback:
            try:
                self.audit_callback(error)
            except Exception as e:
                print(f"Audit callback failed: {e}")
        
        return error
    
    def _handle_critical_error(self, error: MedicalError):
        """Handle critical errors with immediate escalation."""
        if error.should_alert_medical_staff() and self.alert_callback:
            try:
                self.alert_callback(error)
            except Exception as e:
                print(f"Alert callback failed: {e}")
        
        # Log critical errors immediately
        print(f"CRITICAL MEDICAL ERROR: {error.error_code.value} - {error.message}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'critical_errors': len([e for e in self.error_history if e.is_critical()]),
            'recent_errors': len([e for e in self.error_history 
                                 if (datetime.utcnow() - e.timestamp).seconds < 3600]),
            'patient_safety_errors': len([e for e in self.error_history 
                                        if e.category == MedicalErrorCategory.PATIENT_SAFETY])
        }
    
    def clear_history(self):
        """Clear error history (use with caution)."""
        self.error_history.clear()
        self.error_counts.clear()


# Pre-defined error creation functions for common scenarios
def create_patient_safety_error(message: str, **kwargs) -> MedicalError:
    """Create patient safety error."""
    return MedicalError(
        error_code=MedicalErrorCode.E1001,
        category=MedicalErrorCategory.PATIENT_SAFETY,
        severity=MedicalErrorSeverity.CRITICAL,
        message=message,
        recoverable=False,
        escalation_required=True,
        **kwargs
    )


def create_model_failure_error(message: str, **kwargs) -> MedicalError:
    """Create model failure error."""
    return MedicalError(
        error_code=MedicalErrorCode.E3003,
        category=MedicalErrorCategory.MODEL_FAILURE,
        severity=MedicalErrorSeverity.HIGH,
        message=message,
        recoverable=True,
        **kwargs
    )


def create_phi_violation_error(message: str, **kwargs) -> MedicalError:
    """Create PHI violation error."""
    return MedicalError(
        error_code=MedicalErrorCode.E2001,
        category=MedicalErrorCategory.PHI_VIOLATION,
        severity=MedicalErrorSeverity.HIGH,
        message=message,
        recoverable=False,
        escalation_required=True,
        **kwargs
    )


def create_hipaa_violation_error(message: str, **kwargs) -> MedicalError:
    """Create HIPAA violation error."""
    return MedicalError(
        error_code=MedicalErrorCode.E6001,
        category=MedicalErrorCategory.HIPAA_VIOLATION,
        severity=MedicalErrorSeverity.CRITICAL,
        message=message,
        recoverable=False,
        escalation_required=True,
        **kwargs
    )


def create_service_unavailable_error(message: str, **kwargs) -> MedicalError:
    """Create service unavailable error."""
    return MedicalError(
        error_code=MedicalErrorCode.E4001,
        category=MedicalErrorCategory.SERVICE_UNAVAILABLE,
        severity=MedicalErrorSeverity.HIGH,
        message=message,
        recoverable=True,
        **kwargs
    )