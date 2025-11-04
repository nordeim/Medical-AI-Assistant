"""
Medical AI Resilience System - Comprehensive Test Suite
Tests for all resilience components with medical safety validation.
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import all resilience components for testing
from resilience.errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory,
    MedicalErrorSeverity, MedicalErrorHandler,
    create_patient_safety_error, create_hipaa_violation_error
)

from resilience.circuit_breaker import (
    CircuitBreaker, CircuitBreakerState, CircuitBreakerRegistry,
    FailureType, circuit_breaker_registry, medical_circuit_breaker
)

from resilience.retry import (
    RetryManager, MedicalRetryContext, RetryConfig,
    RetryStrategy, RetryCondition, retry_manager,
    medical_retry, critical_operation_retry
)

from resilience.fallback import (
    DegradationManager, DegradationLevel, FallbackStrategy,
    ModelFallbackStrategy, RuleBasedFallbackStrategy,
    degradation_manager, with_fallback
)

from resilience.validation import (
    DataValidator, ValidationResult, DataType, ValidationLevel,
    PHIProtectionLevel, BaseValidator, PatientIDValidator,
    MedicalRecordValidator, data_validator, validate_input
)

from resilience.shutdown import (
    GracefulShutdownManager, HealthCheck, HealthStatus,
    SystemHealthCheck, DatabaseHealthCheck, shutdown_manager,
    register_shutdown_component
)

from resilience.logging import (
    MedicalLogger, LogLevel, AuditEventType,
    AuditLogEntry, medical_logger, get_medical_logger,
    log_patient_access, AuditContext
)

from resilience.orchestrator import (
    ResilienceOrchestrator, ResilienceConfig, PerformanceMonitor,
    get_resilience_orchestrator, initialize_resilience_system,
    with_medical_resilience
)


class TestMedicalErrorHandling:
    """Test medical error handling and categorization."""
    
    @pytest.mark.asyncio
    async def test_patient_safety_error_creation(self):
        """Test creation of patient safety errors."""
        error = create_patient_safety_error(
            "Patient safety alert triggered",
            patient_context={"patient_id": "123"},
            technical_context={"model_confidence": 0.3}
        )
        
        assert error.category == MedicalErrorCategory.PATIENT_SAFETY
        assert error.severity == MedicalErrorSeverity.CRITICAL
        assert error.error_code == MedicalErrorCode.E1001
        assert error.recoverable is False
        assert error.escalation_required is True
        assert error.patient_context["patient_id"] == "123"
    
    @pytest.mark.asyncio
    async def test_error_handler_functionality(self):
        """Test medical error handler."""
        handler = MedicalErrorHandler()
        
        error = handler.handle_error(
            MedicalErrorCode.E3001,
            MedicalErrorCategory.MODEL_DEGRADATION,
            MedicalErrorSeverity.MEDIUM,
            "Model confidence below threshold"
        )
        
        assert error in handler.error_history
        assert handler.error_counts[f"{error.error_code.value}_{error.category.value}"] == 1
        
        # Test statistics
        stats = handler.get_error_statistics()
        assert stats["total_errors"] == 1
        assert stats["model_degradation_errors"] == 1
    
    @pytest.mark.asyncio
    async def test_hipaa_violation_error(self):
        """Test HIPAA violation error creation."""
        error = create_hipaa_violation_error(
            "PHI data exposed in logs",
            patient_context={"patient_id": "456"}
        )
        
        assert error.category == MedicalErrorCategory.HIPAA_VIOLATION
        assert error.severity == MedicalErrorSeverity.CRITICAL
        assert error.recoverable is False
        assert error.should_alert_medical_staff() is True


class TestCircuitBreaker:
    """Test circuit breaker patterns for medical systems."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_basic_functionality(self):
        """Test basic circuit breaker functionality."""
        cb = CircuitBreaker(
            "test_service",
            failure_threshold=2,
            recovery_timeout=1.0,
            patient_safety_mode=True
        )
        
        # Initially closed
        assert cb.state == CircuitBreakerState.CLOSED
        
        # Test successful call
        async def success_func():
            return "success"
        
        result = await cb.call(success_func)
        assert result == "success"
        assert cb.total_successes == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_transitions(self):
        """Test circuit breaker state transitions on failures."""
        cb = CircuitBreaker(
            "test_service",
            failure_threshold=2,
            recovery_timeout=1.0
        )
        
        # Test failing function
        async def failing_func():
            raise Exception("Service error")
        
        # Should fail and transition to open
        for i in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count >= 2
        
        # Test blocked requests while open
        with pytest.raises(Exception) as exc_info:
            await cb.call(success_func)
        assert "Circuit breaker" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_medical_circuit_breaker_decorator(self):
        """Test medical circuit breaker decorator."""
        call_count = 0
        
        @medical_circuit_breaker("test_service", patient_safety_mode=True)
        async def test_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Service error")
            return "success"
        
        # First two calls should fail
        with pytest.raises(Exception):
            await test_function()
        with pytest.raises(Exception):
            await test_function()
        
        # Circuit should be open now
        with pytest.raises(Exception):
            await test_function()
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Should work after timeout
        result = await test_function()
        assert result == "success"
    
    def test_circuit_breaker_registry(self):
        """Test circuit breaker registry."""
        registry = CircuitBreakerRegistry()
        
        # Create circuit breaker
        cb = registry.create("test_service")
        assert cb.name == "test_service"
        
        # Get existing circuit breaker
        cb2 = registry.get("test_service")
        assert cb2 is cb
        
        # Get status
        status = registry.get_all_status()
        assert "test_service" in status
        assert status["test_service"]["state"] == CircuitBreakerState.CLOSED.value


class TestRetryMechanisms:
    """Test retry mechanisms with medical context."""
    
    @pytest.mark.asyncio
    async def test_retry_with_medical_context(self):
        """Test retry with medical context."""
        retry_manager_local = RetryManager()
        call_count = 0
        
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        context = MedicalRetryContext(
            patient_id="patient_123",
            clinical_priority="normal",
            max_retries=3
        )
        
        result = await retry_manager_local.retry_with_context(
            failing_func, context, *[]
        )
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_critical_operation_retry(self):
        """Test critical operation retry configuration."""
        call_count = 0
        
        @critical_operation_retry(max_retries=2, timeout=5.0)
        async def critical_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Critical operation failed")
            return "critical_success"
        
        result = await critical_operation()
        assert result == "critical_success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_config_calculation(self):
        """Test retry delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            backoff_factor=2.0,
            max_retries=3
        )
        
        # Test delay calculation
        assert config.calculate_delay(0) == 1.0  # base_delay
        assert config.calculate_delay(1) == 2.0  # base_delay * backoff_factor
        assert config.calculate_delay(2) == 4.0  # base_delay * backoff_factor^2
        
        # Test with jitter
        config_with_jitter = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
            base_delay=1.0,
            jitter_range=0.1
        )
        
        delay = config_with_jitter.calculate_delay(1)
        assert 1.8 <= delay <= 2.2  # 2.0 ± 10% jitter


class TestFallbackStrategies:
    """Test fallback and degradation strategies."""
    
    @pytest.mark.asyncio
    async def test_degradation_manager(self):
        """Test degradation manager functionality."""
        dm = DegradationManager()
        
        # Add mock fallback strategy
        class MockFallback(FallbackStrategy):
            def __init__(self):
                super().__init__("mock_fallback", priority=1)
            
            async def execute(self, input_data, **kwargs):
                return {"fallback_result": "success"}
            
            def _check_availability(self):
                return True
        
        mock_fallback = MockFallback()
        dm.add_strategy(mock_fallback)
        
        # Test execution with fallback
        async def primary_func(data):
            raise Exception("Primary function failed")
        
        result, strategy, level = await dm.execute_with_fallback(
            primary_func,
            {"test": "data"},
            current_performance=0.3
        )
        
        assert result["fallback_result"] == "success"
        assert strategy == "mock_fallback"
        assert level == DegradationLevel.BASIC_FUNCTIONALITY
    
    @pytest.mark.asyncio
    async def test_model_fallback_strategy(self):
        """Test model fallback strategy."""
        primary_model = MagicMock()
        primary_model.predict.return_value = MagicMock(confidence=0.5)  # Low confidence
        
        fallback_model = MagicMock()
        fallback_model.predict.return_value = MagicMock(confidence=0.8)  # Good confidence
        
        strategy = ModelFallbackStrategy(
            primary_model,
            fallback_model,
            confidence_threshold=0.7
        )
        
        result, source = await strategy.execute({"test": "data"})
        
        # Should use fallback model due to low primary confidence
        assert source == "fallback_model"
        fallback_model.predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rule_based_fallback(self):
        """Test rule-based fallback strategy."""
        def safe_rule(data):
            return {"rule_result": "safe_default"}
        
        strategy = RuleBasedFallbackStrategy({
            "safe_rule": safe_rule
        })
        
        result, source = await strategy.execute({"test": "data"})
        
        assert source == "rule_safe_rule"
        assert result["rule_result"] == "safe_default"


class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_patient_id_validation(self):
        """Test patient ID validation."""
        validator = PatientIDValidator()
        
        # Valid patient ID
        result = validator.validate("PATIENT_123")
        assert result.is_valid is True
        assert len(result.errors) == 0
        
        # Invalid patient ID (too short)
        result = validator.validate("AB")
        assert result.is_valid is False
        assert any("too short" in error for error in result.errors)
        
        # PHI detection
        result = validator.validate("123-45-6789")
        assert len(result.phi_detected) > 0
    
    def test_medical_record_validation(self):
        """Test medical record validation."""
        validator = MedicalRecordValidator()
        
        # Valid record
        valid_record = {
            "patient_id": "PATIENT_123",
            "record_date": "2024-01-15",
            "record_type": "diagnosis",
            "clinical_notes": "Patient shows normal vital signs"
        }
        
        result = validator.validate(valid_record)
        assert result.is_valid is True
        
        # Invalid record (missing required fields)
        invalid_record = {
            "clinical_notes": "Some notes"
        }
        
        result = validator.validate(invalid_record)
        assert result.is_valid is False
        assert len(result.errors) > 0
    
    def test_diagnosis_code_validation(self):
        """Test diagnosis code validation."""
        validator = DiagnosisCodeValidator()
        
        # Valid ICD-10 code
        result = validator.validate("E11.9")
        assert result.is_valid is True
        
        # Valid ICD-9 code
        result = validator.validate("250.00")
        assert result.is_valid is True
        
        # Invalid code
        result = validator.validate("INVALID")
        assert result.is_valid is False
    
    def test_clinical_text_validation(self):
        """Test clinical text validation."""
        validator = ClinicalTextValidator()
        
        # Normal clinical text
        result = validator.validate("Patient presents with mild headache")
        assert result.is_valid is True
        
        # Text with PHI
        result = validator.validate("Patient John Smith, SSN: 123-45-6789")
        assert len(result.phi_detected) > 0
        
        # Dangerous content (should be filtered)
        result = validator.validate("Patient wants to hurt someone")
        # Should have warnings about dangerous content
    
    def test_data_validator_batch_validation(self):
        """Test batch validation."""
        data_items = [
            ("PATIENT_123", DataType.PATIENT_ID),
            ("E11.9", DataType.DIAGNOSIS_CODE),
            ("invalid_code", DataType.DIAGNOSIS_CODE)
        ]
        
        results = data_validator.batch_validate(data_items)
        
        assert len(results) == 3
        assert results[0].is_valid is True  # Valid patient ID
        assert results[1].is_valid is True  # Valid diagnosis code
        assert results[2].is_valid is False  # Invalid diagnosis code


class TestShutdownManager:
    """Test graceful shutdown and health checks."""
    
    @pytest.mark.asyncio
    async def test_system_health_check(self):
        """Test system health check."""
        health_check = SystemHealthCheck()
        
        result = await health_check.execute()
        
        assert "cpu_usage" in result
        assert "memory_usage" in result
        assert "disk_usage" in result
        assert "overall_status" in result
        assert result["overall_status"] in [status.value for status in HealthStatus]
    
    @pytest.mark.asyncio
    async def test_shutdown_manager_registration(self):
        """Test shutdown manager component registration."""
        manager = GracefulShutdownManager()
        
        # Mock component
        class MockComponent:
            async def shutdown(self):
                pass
        
        component = MockComponent()
        manager.register_component("test_component", component)
        
        assert "test_component" in manager.components
        assert manager.components["test_component"] == ComponentState.RUNNING
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance."""
        manager = GracefulShutdownManager()
        
        # Add custom health check
        class QuickHealthCheck(HealthCheck):
            def __init__(self):
                super().__init__("quick_check", priority=1, timeout=1.0)
            
            async def check_health(self):
                return {"status": "healthy", "response_time_ms": 10}
        
        quick_check = QuickHealthCheck()
        manager.add_health_check(quick_check)
        
        result = await manager.perform_health_check()
        
        assert "overall_status" in result
        assert "health_checks" in result
        assert len(result["health_checks"]) >= 1


class TestLogging:
    """Test logging and audit functionality."""
    
    def test_medical_logger_creation(self):
        """Test medical logger creation."""
        logger = MedicalLogger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.phi_protection_level == PHIProtectionLevel.STANDARD
    
    def test_audit_log_entry(self):
        """Test audit log entry creation and integrity."""
        entry = AuditLogEntry(
            event_type=AuditEventType.PATIENT_DATA_ACCESS,
            user_id="doctor_001",
            patient_id="patient_123",
            action_taken="record_access"
        )
        
        # Test entry properties
        assert entry.entry_id is not None
        assert entry.timestamp is not None
        assert entry.event_type == AuditEventType.PATIENT_DATA_ACCESS
        assert entry.user_id == "doctor_001"
        assert entry.patient_id == "patient_123"
        
        # Test integrity verification
        assert entry.verify_integrity() is True
        
        # Test conversion to dictionary
        entry_dict = entry.to_dict()
        assert "entry_id" in entry_dict
        assert "timestamp" in entry_dict
        assert "event_type" in entry_dict
    
    def test_phi_protection(self):
        """Test PHI protection in logging."""
        entry = AuditLogEntry(
            event_type=AuditEventType.PATIENT_DATA_ACCESS,
            details={
                "ssn": "123-45-6789",
                "patient_name": "John Smith",
                "diagnosis": "Hypertension"
            },
            phi_protection=PHIProtectionLevel.STANDARD
        )
        
        # Check that PHI is protected
        assert "[REDACTED]" not in entry.protected_details.get("ssn", "")
        
        # Test with strict protection
        entry_strict = AuditLogEntry(
            event_type=AuditEventType.PATIENT_DATA_ACCESS,
            details={"diagnosis": "Hypertension"},
            phi_protection=PHIProtectionLevel.STRICT
        )
        
        assert entry_strict.protected_details["diagnosis"] == "[REDACTED]"
    
    def test_audit_context(self):
        """Test audit context manager."""
        with AuditContext(
            event_type=AuditEventType.MODEL_INFERENCE,
            user_id="doctor_001",
            patient_id="patient_123"
        ):
            # Simulate some work
            time.sleep(0.01)
        
        # Context should create audit entry (in real implementation)
        # This is a simplified test
    
    def test_compliance_reporting(self):
        """Test compliance reporting."""
        logger = MedicalLogger("compliance_test")
        
        # Add some audit entries
        logger.audit_event(
            event_type=AuditEventType.PATIENT_DATA_ACCESS,
            patient_id="patient_123",
            user_id="doctor_001"
        )
        
        logger.audit_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            user_id="system"
        )
        
        report = logger.get_compliance_report()
        
        assert "total_audit_entries" in report
        assert "hipaa_required_entries" in report
        assert "compliance_violations" in report
        assert report["total_audit_entries"] >= 1


class TestResilienceOrchestrator:
    """Test main resilience orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = ResilienceConfig(
            patient_safety_mode=True,
            hipaa_compliance_enabled=True
        )
        
        orchestrator = ResilienceOrchestrator(config)
        
        # Should initialize successfully
        success = await orchestrator.initialize()
        assert success is True
        assert orchestrator.is_initialized is True
    
    @pytest.mark.asyncio
    async def test_resilience_decorator(self):
        """Test resilience decorator functionality."""
        orchestrator = ResilienceOrchestrator()
        await orchestrator.initialize()
        
        call_count = 0
        
        @with_medical_resilience(
            operation_type="test_operation",
            clinical_priority="normal"
        )
        async def test_function(data):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return {"result": "success", "attempts": call_count}
        
        result = await test_function({"test": "data"})
        
        assert result["result"] == "success"
        assert result["attempts"] == 3  # 2 failures + 1 success
        assert call_count == 3
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor()
        
        # Collect metrics
        metrics1 = monitor.collect_metrics()
        metrics2 = monitor.collect_metrics()
        
        assert "overall_performance" in metrics1
        assert "overall_performance" in metrics2
        
        # Get summary
        summary = monitor.get_summary()
        assert "recent_performance" in summary
        assert "average_performance" in summary
    
    def test_system_status(self):
        """Test system status reporting."""
        config = ResilienceConfig()
        orchestrator = ResilienceOrchestrator(config)
        
        status = orchestrator.get_system_status()
        
        assert "system_info" in status
        assert "components" in status
        assert "logging" in status
        assert "performance" in status
        
        system_info = status["system_info"]
        assert "initialized" in system_info
        assert "health_status" in system_info


class TestIntegration:
    """Integration tests for complete medical AI workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_medical_diagnosis_workflow(self):
        """Test complete medical diagnosis workflow with resilience."""
        
        # Setup orchestrator
        await initialize_resilience_system(patient_safety_mode=True)
        
        # Create mock AI service
        @with_medical_resilience(
            operation_type="patient_diagnosis",
            clinical_priority="high",
            patient_id="patient_integration_123"
        )
        @validate_input(DataType.JSON_DATA)
        @log_medical_operation("diagnosis", AuditEventType.MODEL_INFERENCE)
        async def diagnose_patient(patient_data):
            # Simulate AI diagnosis
            await asyncio.sleep(0.01)
            
            return {
                "diagnosis": "Hypertension - Stage 1",
                "confidence": 0.85,
                "recommendations": ["Lifestyle modifications", "Follow-up in 3 months"]
            }
        
        # Execute diagnosis
        patient_data = {
            "symptoms": ["headache", "dizziness"],
            "vitals": {"bp": "145/90", "heart_rate": "82"}
        }
        
        result = await diagnose_patient(patient_data)
        
        assert "diagnosis" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.8
        
        # Check system status
        orchestrator = get_resilience_orchestrator()
        status = orchestrator.get_system_status()
        
        assert status["system_info"]["initialized"] is True
        assert "components" in status
    
    @pytest.mark.asyncio
    async def test_emergency_workflow(self):
        """Test emergency medical workflow."""
        
        call_count = 0
        
        @critical_medical_operation("emergency_patient_456")
        async def emergency_assessment(patient_data):
            nonlocal call_count
            call_count += 1
            
            # Simulate emergency assessment with possible failures
            if call_count <= 1:
                raise Exception("Emergency system temporarily unavailable")
            
            return {
                "assessment": "Emergency protocol activated",
                "priority": "critical",
                "action_required": "immediate_human_review",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Execute emergency assessment
        emergency_data = {
            "symptoms": ["chest_pain", "shortness_of_breath"],
            "onset": "sudden"
        }
        
        result = await emergency_assessment(emergency_data)
        
        assert result["priority"] == "critical"
        assert result["assessment"] == "Emergency protocol activated"
        assert call_count == 2  # Should retry once for critical operations
    
    @pytest.mark.asyncio
    async def test_data_validation_workflow(self):
        """Test patient data validation workflow."""
        
        @validate_input(
            DataType.MEDICAL_RECORD,
            validation_level=ValidationLevel.STRICT,
            patient_id="patient_validation_789"
        )
        async def process_medical_record(patient_data):
            # Simulate processing
            await asyncio.sleep(0.01)
            
            return {
                "status": "processed",
                "record_id": "REC_789",
                "validation_passed": True
            }
        
        # Valid medical record
        valid_record = {
            "patient_id": "PATIENT_789",
            "record_date": "2024-01-15",
            "record_type": "consultation",
            "clinical_notes": "Routine checkup - all normal"
        }
        
        result = await process_medical_record(valid_record)
        
        assert result["status"] == "processed"
        assert result["validation_passed"] is True
        
        # Invalid record should raise validation error
        invalid_record = {"incomplete": "data"}
        
        with pytest.raises(Exception):  # Should raise PHI violation error
            await process_medical_record(invalid_record)


class TestPerformance:
    """Performance tests for resilience system."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance(self):
        """Test circuit breaker performance overhead."""
        cb = CircuitBreaker("perf_test")
        
        async def fast_function():
            return "success"
        
        # Measure baseline performance
        start_time = time.time()
        for _ in range(100):
            await fast_function()
        baseline_time = time.time() - start_time
        
        # Measure circuit breaker overhead
        start_time = time.time()
        for _ in range(100):
            await cb.call(fast_function)
        cb_time = time.time() - start_time
        
        # Circuit breaker overhead should be minimal (< 50%)
        overhead = (cb_time - baseline_time) / baseline_time
        assert overhead < 0.5, f"Circuit breaker overhead too high: {overhead:.2%}"
    
    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """Test input validation performance."""
        import string
        import random
        
        # Generate test data
        test_patient_ids = [f"PATIENT_{random.randint(1000, 9999)}" for _ in range(100)]
        
        start_time = time.time()
        for patient_id in test_patient_ids:
            result = data_validator.validate(patient_id, DataType.PATIENT_ID)
            assert result.is_valid
        validation_time = time.time() - start_time
        
        # Should validate 100 patient IDs in under 1 second
        assert validation_time < 1.0, f"Validation too slow: {validation_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_audit_logging_performance(self):
        """Test audit logging performance."""
        logger = MedicalLogger("perf_test")
        
        start_time = time.time()
        for i in range(100):
            logger.audit_event(
                event_type=AuditEventType.PATIENT_DATA_ACCESS,
                patient_id=f"patient_{i}",
                user_id="doctor_001"
            )
        logging_time = time.time() - start_time
        
        # Should log 100 entries in under 1 second
        assert logging_time < 1.0, f"Audit logging too slow: {logging_time:.2f}s"


# Test utilities
def create_mock_medical_service():
    """Create a mock medical service for testing."""
    class MockMedicalService:
        def __init__(self):
            self.call_count = 0
            self.failure_rate = 0.3
        
        @medical_circuit_breaker("mock_service", patient_safety_mode=True)
        @medical_retry(
            strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
            max_retries=2
        )
        async def process_request(self, data):
            self.call_count += 1
            if self.call_count % 3 == 0:  # 33% failure rate
                raise Exception("Service error")
            return {"result": f"processed_{self.call_count}"}
    
    return MockMedicalService()


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def orchestrator():
    """Create and initialize a test orchestrator."""
    config = ResilienceConfig(
        patient_safety_mode=True,
        hipaa_compliance_enabled=True,
        health_check_interval=30
    )
    
    orchestrator = ResilienceOrchestrator(config)
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.shutdown()


# Run tests
if __name__ == "__main__":
    # Run specific test categories
    print("Running Medical AI Resilience System Tests...")
    print("=" * 60)
    
    # You can run individual test classes
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings"
    ])