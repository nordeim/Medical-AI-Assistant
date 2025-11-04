"""
Medical AI Resilience System - Usage Examples and Best Practices
Comprehensive examples showing how to use the medical resilience system.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import all resilience components
from .errors import (
    MedicalErrorCode, MedicalErrorCategory, MedicalErrorSeverity,
    create_patient_safety_error
)
from .circuit_breaker import medical_circuit_breaker
from .retry import medical_retry, RetryStrategy, RetryCondition
from .fallback import with_fallback, setup_critical_fallbacks
from .validation import validate_input, DataType, ValidationLevel, validate_patient_id
from .shutdown import register_shutdown_component, add_health_check, basic_system_health
from .logging import (
    log_patient_access, log_security_event, log_hipaa_violation,
    AuditContext, log_medical_operation, AuditEventType, LogLevel
)
from .orchestrator import (
    get_resilience_orchestrator, with_medical_resilience,
    critical_medical_operation, model_inference_operation
)


# Example 1: Basic Medical AI Service with Full Resilience
class MedicalAIService:
    """Example medical AI service with comprehensive resilience."""
    
    def __init__(self):
        self.models = {}
        self.database = None
        self.is_initialized = False
    
    @register_shutdown_component("medical_ai_service")
    async def initialize(self):
        """Initialize the medical AI service with resilience."""
        print("Initializing Medical AI Service...")
        
        # Setup critical fallbacks
        setup_critical_fallbacks()
        
        # Initialize models (with resilience)
        await self._initialize_models()
        
        # Initialize database (with circuit breaker and retry)
        await self._initialize_database()
        
        self.is_initialized = True
        print("Medical AI Service initialized successfully")
    
    @medical_circuit_breaker("model_inference", patient_safety_mode=True)
    @medical_retry(
        strategy=RetryStrategy.MEDICAL_SAFETY,
        max_retries=2,
        clinical_priority="high"
    )
    @with_fallback()
    @validate_input(DataType.JSON_DATA, patient_id="patient_123")
    @log_medical_operation("model_inference", AuditEventType.MODEL_INFERENCE)
    async def diagnose_patient(
        self, 
        patient_data: Dict[str, Any], 
        patient_id: str,
        user_id: str = "doctor_001"
    ) -> Dict[str, Any]:
        """
        Diagnose patient with comprehensive resilience.
        
        This method includes:
        - Circuit breaker protection
        - Retry with medical safety
        - Fallback strategies
        - Input validation
        - Audit logging
        """
        try:
            # Log patient access
            log_patient_access(
                user_id=user_id,
                patient_id=patient_id,
                action="diagnostic_analysis",
                session_id="session_123",
                ip_address="192.168.1.100"
            )
            
            # Simulate AI diagnosis
            print(f"Performing diagnosis for patient {patient_id}")
            
            # Simulate processing time
            await asyncio.sleep(0.1)
            
            # Return diagnosis result
            result = {
                "patient_id": patient_id,
                "diagnosis": "Routine examination completed",
                "confidence": 0.85,
                "recommendations": ["Follow up in 6 months"],
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": "v1.2.3"
            }
            
            return result
            
        except Exception as e:
            # Handle critical medical errors
            error = create_patient_safety_error(
                f"Diagnosis failed for patient {patient_id}: {str(e)}",
                patient_context={"patient_id": patient_id},
                technical_context={"error_type": type(e).__name__}
            )
            raise error
    
    @critical_medical_operation("patient_456")
    async def emergency_assessment(
        self, 
        patient_data: Dict[str, Any],
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Emergency medical assessment with maximum safety.
        
        Uses critical operation settings:
        - Lower retry limits
        - Faster failure detection
        - Enhanced safety checks
        - Immediate escalation
        """
        print(f"Emergency assessment for patient {patient_id}")
        
        # Log emergency access
        log_patient_access(
            user_id="emergency_system",
            patient_id=patient_id,
            action="emergency_assessment",
            outcome="emergency_activation"
        )
        
        # Simulate emergency assessment
        await asyncio.sleep(0.05)  # Faster for emergencies
        
        return {
            "patient_id": patient_id,
            "assessment": "Emergency protocol activated",
            "priority": "critical",
            "action_required": "immediate_human_review",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _initialize_models(self):
        """Initialize AI models with resilience."""
        # This would initialize actual models
        # For now, simulate model loading
        self.models = {
            "diagnosis_model": {"version": "v1.2.3", "status": "loaded"},
            "triage_model": {"version": "v2.1.0", "status": "loaded"}
        }
        
        # Add health check for models
        @add_health_check(priority=2, timeout=15.0)
        async def model_health_check():
            """Health check for AI models."""
            healthy_models = 0
            for name, model_info in self.models.items():
                if model_info.get("status") == "loaded":
                    healthy_models += 1
            
            total_models = len(self.models)
            health_ratio = healthy_models / max(total_models, 1)
            
            return {
                "healthy_models": healthy_models,
                "total_models": total_models,
                "health_ratio": health_ratio,
                "overall_status": "healthy" if health_ratio > 0.8 else "degraded"
            }
        
        print(f"Initialized {len(self.models)} AI models")
    
    async def _initialize_database(self):
        """Initialize database with circuit breaker protection."""
        # Simulate database connection with potential failure
        import random
        
        if random.random() < 0.1:  # 10% chance of failure
            raise Exception("Database connection failed")
        
        self.database = {"connected": True, "status": "healthy"}
        print("Database connection established")


# Example 2: Patient Data Service with PHI Protection
class PatientDataService:
    """Service for handling patient data with PHI protection."""
    
    def __init__(self):
        self.audit_logger = None
    
    @validate_patient_id("patient_id")
    @log_medical_operation("patient_data_access", AuditEventType.PATIENT_DATA_ACCESS)
    async def get_patient_record(
        self, 
        patient_id: str, 
        user_id: str = "doctor_001"
    ) -> Dict[str, Any]:
        """
        Get patient record with full validation and audit.
        
        Features:
        - Patient ID validation
        - PHI protection
        - Audit logging
        - Access control
        """
        # Log patient data access
        log_patient_access(
            user_id=user_id,
            patient_id=patient_id,
            action="record_access",
            resource_accessed="patient_record"
        )
        
        # Simulate database lookup
        await asyncio.sleep(0.05)
        
        # Return sanitized patient data
        return {
            "patient_id": patient_id,
            "name": "[PROTECTED]",
            "age": 45,
            "last_visit": "2024-01-15",
            "record_status": "active"
        }
    
    @validate_input(
        DataType.MEDICAL_RECORD, 
        validation_level=ValidationLevel.STRICT,
        patient_id="patient_id",
        clinical_priority="high"
    )
    @log_medical_operation("patient_data_update", AuditEventType.PATIENT_DATA_MODIFICATION)
    async def update_patient_record(
        self,
        patient_id: str,
        updates: Dict[str, Any],
        user_id: str = "doctor_001"
    ) -> bool:
        """
        Update patient record with strict validation.
        
        Features:
        - Strict input validation
        - Medical record validation
        - Audit trail for modifications
        - PHI protection
        """
        # Validate updates
        if not updates:
            raise ValueError("No updates provided")
        
        # Log the update attempt
        log_patient_access(
            user_id=user_id,
            patient_id=patient_id,
            action="record_modification",
            resource_accessed="patient_record"
        )
        
        # Simulate update
        await asyncio.sleep(0.1)
        
        return True


# Example 3: Medical Data Integration Service
class MedicalDataIntegrationService:
    """Service for integrating with external medical systems."""
    
    def __init__(self):
        self.external_services = {}
    
    @medical_circuit_breaker(
        "lab_results_api",
        failure_threshold=3,
        recovery_timeout=30.0,
        patient_safety_mode=True
    )
    @medical_retry(
        strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
        max_retries=3,
        timeout=60.0,
        clinical_priority="normal"
    )
    @validate_input(DataType.PATIENT_ID)
    async def fetch_lab_results(
        self,
        patient_id: str,
        test_types: List[str]
    ) -> Dict[str, Any]:
        """
        Fetch lab results with circuit breaker and retry protection.
        
        Features:
        - Circuit breaker for external API calls
        - Exponential backoff retry
        - Input validation
        - Timeout protection
        """
        print(f"Fetching lab results for patient {patient_id}")
        
        # Simulate API call
        await asyncio.sleep(0.2)
        
        return {
            "patient_id": patient_id,
            "test_results": [
                {"test": test, "status": "completed", "value": "normal"}
                for test in test_types
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


# Example 4: Complete System Integration
async def setup_medical_system():
    """Setup complete medical system with resilience."""
    print("Setting up Medical AI System...")
    
    # Initialize resilience orchestrator
    orchestrator = get_resilience_orchestrator()
    success = await orchestrator.initialize()
    
    if not success:
        print("Failed to initialize resilience system")
        return False
    
    print("Resilience system initialized")
    
    # Initialize services
    ai_service = MedicalAIService()
    await ai_service.initialize()
    
    data_service = PatientDataService()
    integration_service = MedicalDataIntegrationService()
    
    print("All services initialized")
    return True


async def run_medical_scenarios():
    """Run various medical scenarios to demonstrate resilience."""
    print("\nRunning Medical AI Scenarios...")
    
    # Setup services
    await setup_medical_system()
    
    # Scenario 1: Normal diagnosis
    print("\n=== Scenario 1: Normal Diagnosis ===")
    try:
        ai_service = MedicalAIService()
        await ai_service.initialize()
        
        patient_data = {
            "symptoms": ["fever", "cough"],
            "duration": "3 days",
            "severity": "moderate"
        }
        
        result = await ai_service.diagnose_patient(
            patient_data=patient_data,
            patient_id="patient_123"
        )
        print(f"Diagnosis result: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"Diagnosis failed: {e}")
    
    # Scenario 2: Emergency assessment
    print("\n=== Scenario 2: Emergency Assessment ===")
    try:
        ai_service = MedicalAIService()
        await ai_service.initialize()
        
        emergency_data = {
            "symptoms": ["chest_pain", "shortness_of_breath"],
            "onset": "sudden"
        }
        
        result = await ai_service.emergency_assessment(
            patient_data=emergency_data,
            patient_id="patient_456"
        )
        print(f"Emergency assessment: {json.dumps(result, indent=2)}")
        
    except Exception as e:
        print(f"Emergency assessment failed: {e}")
    
    # Scenario 3: Data access with validation
    print("\n=== Scenario 3: Patient Data Access ===")
    try:
        data_service = PatientDataService()
        
        record = await data_service.get_patient_record("patient_789")
        print(f"Patient record: {json.dumps(record, indent=2)}")
        
    except Exception as e:
        print(f"Data access failed: {e}")
    
    # Scenario 4: External API integration
    print("\n=== Scenario 4: External API Integration ===")
    try:
        integration_service = MedicalDataIntegrationService()
        
        lab_results = await integration_service.fetch_lab_results(
            patient_id="patient_101",
            test_types=["blood_glucose", "cholesterol", "hemoglobin"]
        )
        print(f"Lab results: {json.dumps(lab_results, indent=2)}")
        
    except Exception as e:
        print(f"Lab results fetch failed: {e}")
    
    # Get system status
    print("\n=== System Status ===")
    orchestrator = get_resilience_orchestrator()
    status = orchestrator.get_system_status()
    print(f"System status: {json.dumps(status, indent=2, default=str)}")


# Example 5: Error Handling and Recovery
async def demonstrate_error_handling():
    """Demonstrate error handling and recovery mechanisms."""
    print("\nDemonstrating Error Handling...")
    
    class FaultyMedicalService:
        """Service that deliberately fails to test error handling."""
        
        @medical_circuit_breaker("faulty_service", failure_threshold=2, recovery_timeout=5.0)
        @medical_retry(
            strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
            max_retries=3,
            clinical_priority="normal"
        )
        async def unreliable_operation(self, data: Dict[str, Any]) -> str:
            """Operation that fails to test resilience."""
            import random
            
            if random.random() < 0.7:  # 70% chance of failure
                raise Exception("Simulated service failure")
            
            return "Operation successful"
    
    service = FaultyMedicalService()
    
    # Test multiple attempts
    for i in range(5):
        try:
            result = await service.unreliable_operation({"attempt": i})
            print(f"Attempt {i+1}: {result}")
        except Exception as e:
            print(f"Attempt {i+1}: Failed - {e}")


# Example 6: Health Check Monitoring
async def demonstrate_health_monitoring():
    """Demonstrate health check monitoring."""
    print("\nDemonstrating Health Monitoring...")
    
    # Add custom health checks
    @add_health_check(priority=1, timeout=5.0)
    async def custom_health_check():
        """Custom health check for demonstration."""
        return {
            "custom_component": "healthy",
            "metrics": {
                "response_time": 0.1,
                "error_rate": 0.01,
                "throughput": 100
            },
            "overall_status": "healthy"
        }
    
    # Perform health check
    shutdown_manager = get_resilience_orchestrator().shutdown_manager
    health_result = await shutdown_manager.perform_health_check()
    
    print(f"Health check result: {json.dumps(health_result, indent=2, default=str)}")


# Example 7: Graceful Shutdown
async def demonstrate_graceful_shutdown():
    """Demonstrate graceful shutdown."""
    print("\nDemonstrating Graceful Shutdown...")
    
    orchestrator = get_resilience_orchestrator()
    
    # Perform graceful shutdown
    await orchestrator.shutdown(phase="graceful", timeout=30.0)
    
    print("System shutdown complete")


# Main execution examples
async def main():
    """Main function demonstrating all resilience features."""
    print("Medical AI Resilience System - Complete Examples")
    print("=" * 60)
    
    # Run all examples
    await run_medical_scenarios()
    await demonstrate_error_handling()
    await demonstrate_health_monitoring()
    await demonstrate_graceful_shutdown()
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    # Run the main examples
    asyncio.run(main())


# Additional utility functions for common medical scenarios
def create_critical_diagnosis_service():
    """Create a service optimized for critical medical diagnoses."""
    
    @critical_medical_operation("critical_diagnosis")
    async def critical_diagnosis(patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Critical diagnosis with maximum safety."""
        return {
            "diagnosis": "Critical assessment completed",
            "confidence": 0.95,
            "requires_human_review": False,
            "escalation_level": "immediate"
        }
    
    return critical_diagnosis


def create_model_inference_service():
    """Create a service optimized for model inference."""
    
    @model_inference_operation("ai_inference")
    async def ai_inference(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI inference with model-specific resilience."""
        # Simulate model inference
        await asyncio.sleep(0.1)
        
        return {
            "prediction": "inference_completed",
            "confidence": 0.88,
            "model_version": "v1.2.3"
        }
    
    return ai_inference


def create_data_validation_service():
    """Create a service optimized for data validation."""
    
    @validate_input(DataType.MEDICAL_RECORD, ValidationLevel.STRICT)
    async def validate_medical_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical data with strict checks."""
        return {
            "validation_status": "passed",
            "sanitized_data": data,
            "phi_detected": False
        }
    
    return validate_medical_data


# Best practices and configuration examples
BEST_PRACTICES = {
    "error_handling": [
        "Always categorize medical errors appropriately",
        "Use specific error codes for different failure types",
        "Implement patient safety error escalation",
        "Log all critical errors for audit purposes"
    ],
    
    "circuit_breakers": [
        "Use lower failure thresholds for critical medical systems",
        "Configure longer recovery timeouts for medical services",
        "Enable patient safety mode for clinical applications",
        "Monitor circuit breaker states in real-time"
    ],
    
    "retry_mechanisms": [
        "Use medical safety strategy for patient-critical operations",
        "Implement shorter timeouts for emergency procedures",
        "Limit retry attempts for critical operations",
        "Use exponential backoff to prevent system overload"
    ],
    
    "fallback_strategies": [
        "Setup rule-based fallbacks for safety-critical operations",
        "Use model fallbacks for AI inference services",
        "Implement progressive degradation levels",
        "Always provide safe defaults for medical decisions"
    ],
    
    "input_validation": [
        "Validate all patient identifiers",
        "Use strict validation for critical medical data",
        "Implement PHI detection and protection",
        "Sanitize all user inputs before processing"
    ],
    
    "health_monitoring": [
        "Monitor patient safety indicators continuously",
        "Set up alerts for critical health thresholds",
        "Implement comprehensive system health checks",
        "Track performance metrics for optimization"
    ],
    
    "logging_audit": [
        "Log all patient data access events",
        "Maintain 7-year audit trails for HIPAA compliance",
        "Use PHI protection for all patient data",
        "Implement integrity verification for audit logs"
    ]
}

# Configuration templates for common medical scenarios
CONFIG_TEMPLATES = {
    "critical_care": {
        "patient_safety_mode": True,
        "default_failure_threshold": 2,
        "default_retry_attempts": 1,
        "health_check_interval": 15,
        "phi_protection_level": "strict"
    },
    
    "routine_clinic": {
        "patient_safety_mode": False,
        "default_failure_threshold": 5,
        "default_retry_attempts": 3,
        "health_check_interval": 60,
        "phi_protection_level": "standard"
    },
    
    "research_system": {
        "patient_safety_mode": False,
        "default_failure_threshold": 10,
        "default_retry_attempts": 5,
        "health_check_interval": 300,
        "phi_protection_level": "complete"
    }
}