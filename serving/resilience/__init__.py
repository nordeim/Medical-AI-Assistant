"""
Medical AI Resilience System
Comprehensive resilience and error handling system for medical AI applications.

This module provides production-grade reliability for medical AI systems with:
- Medical-specific error handling and recovery
- Circuit breaker patterns with medical system isolation
- Retry mechanisms with exponential backoff and safety considerations
- Fallback models and degradation strategies
- Input validation and sanitization with medical data integrity protection
- Graceful shutdown and health checks
- HIPAA-compliant logging and audit trails
"""

# Core error handling
from .errors import (
    MedicalError,
    MedicalErrorCode,
    MedicalErrorCategory,
    MedicalErrorSeverity,
    MedicalErrorHandler,
    create_patient_safety_error,
    create_model_failure_error,
    create_phi_violation_error,
    create_hipaa_violation_error,
    create_service_unavailable_error
)

# Circuit breaker patterns
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerRegistry,
    FailureType,
    circuit_breaker_registry,
    medical_circuit_breaker
)

# Retry mechanisms
from .retry import (
    RetryManager,
    RetryConfig,
    RetryStrategy,
    RetryCondition,
    MedicalRetryContext,
    retry_manager,
    medical_retry,
    critical_operation_retry,
    model_inference_retry,
    data_retrieval_retry
)

# Fallback and degradation
from .fallback import (
    DegradationManager,
    DegradationLevel,
    FallbackStrategy,
    ModelFallbackStrategy,
    RuleBasedFallbackStrategy,
    degradation_manager,
    with_fallback,
    setup_critical_fallbacks,
    setup_routine_fallbacks
)

# Input validation and sanitization
from .validation import (
    DataValidator,
    ValidationResult,
    DataType,
    ValidationLevel,
    PHIField,
    BaseValidator,
    PatientIDValidator,
    MedicalRecordValidator,
    DiagnosisCodeValidator,
    ClinicalTextValidator,
    data_validator,
    validate_input,
    validate_patient_id
)

# Graceful shutdown and health checks
from .shutdown import (
    GracefulShutdownManager,
    HealthCheck,
    HealthStatus,
    SystemHealthCheck,
    DatabaseHealthCheck,
    ModelHealthCheck,
    MedicalDataIntegrityCheck,
    ComponentState,
    ShutdownPhase,
    shutdown_manager,
    register_shutdown_component,
    add_health_check,
    basic_system_health,
    database_health_check
)

# Logging and audit trails
from .logging import (
    MedicalLogger,
    LogLevel,
    AuditEventType,
    PHIProtectionLevel,
    AuditLogEntry,
    AuditContext,
    medical_logger,
    setup_medical_logging,
    get_medical_logger,
    log_patient_access,
    log_security_event,
    log_hipaa_violation,
    log_medical_operation
)

# Main orchestrator
from .orchestrator import (
    ResilienceOrchestrator,
    ResilienceConfig,
    PerformanceMonitor,
    get_resilience_orchestrator,
    initialize_resilience_system,
    with_medical_resilience,
    critical_medical_operation,
    model_inference_operation,
    data_validation_operation
)

# Version information
__version__ = "1.0.0"
__author__ = "Medical AI Resilience Team"
__description__ = "Comprehensive resilience and error handling for medical AI systems"

# Public API - main classes and functions
__all__ = [
    # Error handling
    "MedicalError",
    "MedicalErrorCode",
    "MedicalErrorCategory", 
    "MedicalErrorSeverity",
    "MedicalErrorHandler",
    "create_patient_safety_error",
    "create_model_failure_error",
    "create_phi_violation_error",
    "create_hipaa_violation_error",
    "create_service_unavailable_error",
    
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerState",
    "CircuitBreakerRegistry",
    "FailureType",
    "circuit_breaker_registry",
    "medical_circuit_breaker",
    
    # Retry mechanisms
    "RetryManager",
    "RetryConfig",
    "RetryStrategy",
    "RetryCondition",
    "MedicalRetryContext",
    "retry_manager",
    "medical_retry",
    "critical_operation_retry",
    "model_inference_retry",
    "data_retrieval_retry",
    
    # Fallback and degradation
    "DegradationManager",
    "DegradationLevel",
    "FallbackStrategy",
    "ModelFallbackStrategy",
    "RuleBasedFallbackStrategy",
    "degradation_manager",
    "with_fallback",
    "setup_critical_fallbacks",
    "setup_routine_fallbacks",
    
    # Input validation
    "DataValidator",
    "ValidationResult",
    "DataType",
    "ValidationLevel",
    "PHIField",
    "BaseValidator",
    "PatientIDValidator",
    "MedicalRecordValidator",
    "DiagnosisCodeValidator",
    "ClinicalTextValidator",
    "data_validator",
    "validate_input",
    "validate_patient_id",
    
    # Shutdown and health checks
    "GracefulShutdownManager",
    "HealthCheck",
    "HealthStatus",
    "SystemHealthCheck",
    "DatabaseHealthCheck",
    "ModelHealthCheck",
    "MedicalDataIntegrityCheck",
    "ComponentState",
    "ShutdownPhase",
    "shutdown_manager",
    "register_shutdown_component",
    "add_health_check",
    "basic_system_health",
    "database_health_check",
    
    # Logging and audit
    "MedicalLogger",
    "LogLevel",
    "AuditEventType",
    "PHIProtectionLevel",
    "AuditLogEntry",
    "AuditContext",
    "medical_logger",
    "setup_medical_logging",
    "get_medical_logger",
    "log_patient_access",
    "log_security_event",
    "log_hipaa_violation",
    "log_medical_operation",
    
    # Main orchestrator
    "ResilienceOrchestrator",
    "ResilienceConfig",
    "PerformanceMonitor",
    "get_resilience_orchestrator",
    "initialize_resilience_system",
    "with_medical_resilience",
    "critical_medical_operation",
    "model_inference_operation",
    "data_validation_operation"
]

# Quick setup functions
def quick_setup(
    enable_circuit_breakers: bool = True,
    enable_retry: bool = True,
    enable_fallback: bool = True,
    enable_validation: bool = True,
    enable_audit: bool = True,
    patient_safety_mode: bool = False
) -> ResilienceOrchestrator:
    """
    Quick setup of resilience system with default configurations.
    
    Args:
        enable_circuit_breakers: Enable circuit breaker patterns
        enable_retry: Enable retry mechanisms with exponential backoff
        enable_fallback: Enable fallback strategies and degradation
        enable_validation: Enable input validation and sanitization
        enable_audit: Enable HIPAA-compliant logging and audit trails
        patient_safety_mode: Enable patient safety mode (more conservative settings)
    
    Returns:
        Configured ResilienceOrchestrator instance
    """
    config = ResilienceConfig(
        circuit_breaker_enabled=enable_circuit_breakers,
        enable_medical_safe_retry=enable_retry,
        fallback_enabled=enable_fallback,
        validation_enabled=enable_validation,
        audit_enabled=enable_audit,
        hipaa_compliance_enabled=enable_audit,
        patient_safety_mode=patient_safety_mode
    )
    
    return ResilienceOrchestrator(config)


async def setup_medical_resilience(
    quick_config: bool = True,
    **kwargs
) -> bool:
    """
    Setup and initialize the medical resilience system.
    
    Args:
        quick_config: Use quick setup with default configurations
        **kwargs: Additional configuration parameters
    
    Returns:
        True if setup was successful, False otherwise
    """
    if quick_config:
        orchestrator = quick_setup(**kwargs)
        return await orchestrator.initialize()
    else:
        return await initialize_resilience_system()


# Example usage patterns
EXAMPLE_USAGE = """
# Basic usage with quick setup
import asyncio
from medical_ai_resilience import setup_medical_resilience, with_medical_resilience

async def main():
    # Initialize resilience system
    await setup_medical_resilience(patient_safety_mode=True)
    
    # Use resilience decorator
    @with_medical_resilience(
        operation_type="patient_diagnosis",
        clinical_priority="high"
    )
    async def diagnose_patient(patient_data):
        # Your medical AI diagnosis logic here
        return diagnosis_result
    
    # Use the protected function
    result = await diagnose_patient(patient_data)

# Advanced usage with custom configuration
from medical_ai_resilience import (
    ResilienceConfig, ResilienceOrchestrator, 
    medical_circuit_breaker, medical_retry
)

config = ResilienceConfig(
    default_failure_threshold=3,
    default_retry_attempts=2,
    patient_safety_mode=True,
    hipaa_compliance_enabled=True
)

orchestrator = ResilienceOrchestrator(config)
await orchestrator.initialize()

# Use circuit breaker decorator
@medical_circuit_breaker("diagnosis_model", patient_safety_mode=True)
async def run_diagnosis_model(patient_data):
    return model.predict(patient_data)

# Use retry decorator with medical context
@medical_retry(
    strategy="medical_safety",
    max_retries=2,
    clinical_priority="high"
)
async def fetch_medical_data(patient_id):
    return await medical_data_service.get_patient_data(patient_id)
"""

# Compliance and security features
COMPLIANCE_FEATURES = {
    "HIPAA_compliance": {
        "description": "Full HIPAA audit trail compliance",
        "features": [
            "7-year audit log retention",
            "PHI detection and protection",
            "Access logging for all patient data",
            "Integrity verification for audit entries",
            "Automatic compliance violation detection"
        ]
    },
    "Patient_safety": {
        "description": "Patient safety priority in all resilience decisions",
        "features": [
            "Critical error escalation",
            "Safe degradation modes",
            "Patient safety circuit breakers",
            "Emergency fallback protocols",
            "Human review requirements for critical decisions"
        ]
    },
    "Medical_compliance": {
        "description": "Medical industry specific compliance features",
        "features": [
            "Medical error categorization",
            "Clinical priority-based retry strategies",
            "Medical data integrity validation",
            "Regulatory compliance monitoring",
            "Audit trail for all medical operations"
        ]
    }
}

if __name__ == "__main__":
    # Example of how to use the resilience system
    print("Medical AI Resilience System")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Description: {__description__}")
    print("\nQuick Start:")
    print(EXAMPLE_USAGE[:500] + "...")
    print("\nCompliance Features:")
    for feature, details in COMPLIANCE_FEATURES.items():
        print(f"- {feature}: {details['description']}")