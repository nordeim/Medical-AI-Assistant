"""
Medical AI Resilience - Main Orchestrator
Central orchestrator for all resilience mechanisms with comprehensive medical system management.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Callable, Union, Type
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field

from .errors import (
    MedicalErrorHandler, MedicalError, MedicalErrorCode,
    MedicalErrorCategory, MedicalErrorSeverity
)
from .circuit_breaker import (
    CircuitBreakerRegistry, CircuitBreaker,
    medical_circuit_breaker, CircuitBreakerState
)
from .retry import (
    RetryManager, MedicalRetryContext, RetryConfig,
    RetryStrategy, RetryCondition, critical_operation_retry
)
from .fallback import (
    DegradationManager, DegradationLevel,
    FallbackStrategy, setup_critical_fallbacks, setup_routine_fallbacks
)
from .validation import (
    DataValidator, DataType, ValidationLevel,
    validate_input, validate_patient_id
)
from .shutdown import (
    GracefulShutdownManager, HealthStatus,
    SystemHealthCheck, DatabaseHealthCheck,
    ModelHealthCheck, MedicalDataIntegrityCheck
)
from .logging import (
    MedicalLogger, LogLevel, AuditEventType,
    PHIProtectionLevel, get_medical_logger,
    AuditContext, log_patient_access
)


@dataclass
class ResilienceConfig:
    """Configuration for medical AI resilience system."""
    # Error Handling
    enable_critical_error_alerts: bool = True
    critical_error_timeout: int = 300  # 5 minutes
    
    # Circuit Breaker
    circuit_breaker_enabled: bool = True
    default_failure_threshold: int = 5
    default_recovery_timeout: float = 60.0
    
    # Retry Configuration
    default_max_retries: int = 3
    default_timeout: float = 300.0
    enable_medical_safe_retry: bool = True
    
    # Fallback Strategy
    fallback_enabled: bool = True
    performance_threshold: float = 0.8
    degradation_levels: List[DegradationLevel] = field(default_factory=list)
    
    # Validation
    validation_enabled: bool = True
    default_validation_level: ValidationLevel = ValidationLevel.STANDARD
    phi_protection_enabled: bool = True
    
    # Health Checks
    health_check_enabled: bool = True
    health_check_interval: int = 30  # seconds
    critical_health_threshold: int = 3
    
    # Logging and Audit
    logging_enabled: bool = True
    audit_enabled: bool = True
    hipaa_compliance_enabled: bool = True
    audit_retention_days: int = 2555  # 7 years
    
    # System Isolation
    medical_isolation_enabled: bool = True
    patient_safety_mode: bool = False


class ResilienceOrchestrator:
    """Main orchestrator for medical AI resilience system."""
    
    def __init__(self, config: ResilienceConfig = None):
        self.config = config or ResilienceConfig()
        
        # Initialize components
        self.error_handler = MedicalErrorHandler()
        self.circuit_breaker_registry = CircuitBreakerRegistry()
        self.retry_manager = RetryManager()
        self.degradation_manager = DegradationManager()
        self.data_validator = DataValidator()
        self.shutdown_manager = GracefulShutdownManager()
        self.medical_logger = get_medical_logger()
        
        # System state
        self.is_initialized = False
        self.current_health_status = HealthStatus.UNKNOWN
        self.system_metrics: Dict[str, Any] = {}
        self.startup_time = datetime.utcnow()
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        
        # Register callbacks
        self._setup_callbacks()
        
        # Load default configurations
        self._setup_default_configurations()
        
        # Start background tasks
        self._background_tasks: List[asyncio.Task] = []
    
    def _setup_callbacks(self):
        """Setup callbacks between components."""
        # Error handler callbacks
        self.error_handler.register_callback(self._handle_critical_error)
        self.error_handler.register_callback(self._audit_error_event)
        
        # Shutdown manager callbacks
        self.shutdown_manager.set_audit_callback(self._audit_shutdown_event)
        
        # Data validator callbacks
        self.data_validator.validators[DataType.PATIENT_ID].set_audit_callback(
            self._audit_validation_event
        )
        self.data_validator.validators[DataType.PATIENT_ID].set_phi_callback(
            self._handle_phi_detection
        )
    
    def _setup_default_configurations(self):
        """Setup default configurations for medical systems."""
        # Setup health checks
        self._setup_default_health_checks()
        
        # Setup fallback strategies
        if self.config.fallback_enabled:
            self._setup_default_fallbacks()
        
        # Setup circuit breakers
        if self.config.circuit_breaker_enabled:
            self._setup_default_circuit_breakers()
        
        # Configure error handling
        if self.config.enable_critical_error_alerts:
            self.error_handler.alert_callback = self._send_critical_alert
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        if not self.config.health_check_enabled:
            return
        
        # System health check
        system_health = SystemHealthCheck(
            memory_threshold=0.85,
            cpu_threshold=0.80
        )
        self.shutdown_manager.add_health_check(system_health)
        
        # Data integrity check (highest priority)
        data_integrity_health = MedicalDataIntegrityCheck(
            data_stores={},  # Will be populated by system
            phi_validator=lambda x: True  # Basic PHI validator
        )
        self.shutdown_manager.add_health_check(data_integrity_health)
    
    def _setup_default_fallbacks(self):
        """Setup default fallback strategies."""
        # Setup critical fallbacks for patient safety
        setup_critical_fallbacks()
        
        # Setup routine fallbacks for normal operations
        setup_routine_fallbacks()
    
    def _setup_default_circuit_breakers(self):
        """Setup default circuit breakers."""
        # Database circuit breaker
        self.circuit_breaker_registry.create(
            "database",
            failure_threshold=self.config.default_failure_threshold,
            recovery_timeout=self.config.default_recovery_timeout,
            patient_safety_mode=True
        )
        
        # Model inference circuit breaker
        self.circuit_breaker_registry.create(
            "model_inference",
            failure_threshold=3,  # Lower threshold for models
            recovery_timeout=30.0,
            patient_safety_mode=True
        )
        
        # External services circuit breaker
        self.circuit_breaker_registry.create(
            "external_services",
            failure_threshold=5,
            recovery_timeout=60.0,
            patient_safety_mode=self.config.patient_safety_mode
        )
    
    async def initialize(self) -> bool:
        """Initialize the resilience system."""
        try:
            logging.info("Initializing Medical AI Resilience System...")
            
            # Initialize components
            await self._initialize_components()
            
            # Start background monitoring
            if self.config.health_check_enabled:
                self._start_background_tasks()
            
            self.is_initialized = True
            self.startup_time = datetime.utcnow()
            
            # Log startup
            self.medical_logger.audit_event(
                event_type=AuditEventType.SYSTEM_STARTUP,
                action_taken="resilience_system_initialized",
                outcome="success",
                details={
                    "config": self.config.__dict__,
                    "components": [
                        "error_handler",
                        "circuit_breaker_registry",
                        "retry_manager",
                        "degradation_manager",
                        "data_validator",
                        "shutdown_manager"
                    ]
                }
            )
            
            logging.info("Medical AI Resilience System initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize resilience system: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialize individual components."""
        # Test error handler
        test_error = self.error_handler.handle_error(
            MedicalErrorCode.E6003,
            MedicalErrorCategory.AUDIT_COMPLIANCE,
            MedicalErrorSeverity.INFO,
            "Test error handling"
        )
        
        # Test circuit breaker registry
        self.circuit_breaker_registry.create("test_circuit")
        
        # Test data validator
        validation_result = self.data_validator.validate(
            "test_patient_123",
            DataType.PATIENT_ID,
            ValidationLevel.BASIC
        )
        
        # Test shutdown manager health check
        health_result = await self.shutdown_manager.perform_health_check()
        
        logging.info("All resilience components initialized")
    
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self._background_tasks.append(health_task)
        
        # Performance monitoring task
        performance_task = asyncio.create_task(self._performance_monitoring_loop())
        self._background_tasks.append(performance_task)
        
        # Log cleanup task
        cleanup_task = asyncio.create_task(self._log_cleanup_loop())
        self._background_tasks.append(cleanup_task)
    
    async def _health_check_loop(self):
        """Background health check monitoring loop."""
        while True:
            try:
                health_status = await self.shutdown_manager.perform_health_check()
                self.current_health_status = HealthStatus(health_status["overall_status"])
                
                # Update system metrics
                self.system_metrics["health_status"] = health_status
                
                # Handle health issues
                if self.current_health_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    await self._handle_health_issue(health_status)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logging.error(f"Health check loop error: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring loop."""
        while True:
            try:
                # Monitor system performance
                performance_data = self.performance_monitor.collect_metrics()
                
                # Update degradation manager
                if self.config.fallback_enabled:
                    current_performance = performance_data.get("overall_performance", 1.0)
                    self.degradation_manager._update_degradation_level(current_performance)
                
                # Monitor circuit breaker states
                circuit_states = self.circuit_breaker_registry.get_all_status()
                open_circuits = [
                    name for name, status in circuit_states.items()
                    if status["state"] == CircuitBreakerState.OPEN.value
                ]
                
                if open_circuits:
                    self.medical_logger.log(
                        level=LogLevel.WARNING,
                        message=f"Open circuit breakers detected: {open_circuits}",
                        context={"open_circuits": open_circuits}
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Performance monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _log_cleanup_loop(self):
        """Background log cleanup loop."""
        while True:
            try:
                # Clean up expired log entries
                self.medical_logger.cleanup_expired_entries()
                
                # Archive old logs
                self.medical_logger.archive_old_logs(days_to_keep=30)
                
                await asyncio.sleep(86400)  # Run daily
                
            except Exception as e:
                logging.error(f"Log cleanup loop error: {e}")
                await asyncio.sleep(86400)
    
    def execute_with_resilience(
        self,
        operation_type: str = "general",
        patient_id: Optional[str] = None,
        clinical_priority: str = "normal",
        enable_fallback: bool = True,
        enable_retry: bool = True,
        enable_circuit_breaker: bool = True,
        validation_required: bool = True,
        audit_required: bool = True
    ) -> Callable:
        """Decorator for executing operations with full resilience."""
        
        def decorator(func: Callable):
            # Create circuit breaker
            if enable_circuit_breaker:
                circuit_breaker = self.circuit_breaker_registry.create(
                    f"{func.__module__}.{func.__name__}",
                    patient_safety_mode=(clinical_priority == "critical")
                )
                func = circuit_breaker(func)
            
            # Add retry with medical context
            if enable_retry:
                retry_context = MedicalRetryContext(
                    patient_id=patient_id,
                    clinical_priority=clinical_priority,
                    operation_type=operation_type
                )
                
                async def retry_wrapper(*args, **kwargs):
                    try:
                        return await self.retry_manager.retry_with_context(
                            func, retry_context, *args, **kwargs
                        )
                    except Exception as e:
                        # Handle retry failure
                        if clinical_priority == "critical":
                            # For critical operations, try fallback
                            if enable_fallback:
                                return await self.degradation_manager.execute_with_fallback(
                                    func, args[0] if args else {}, 0.5, *args[1:], **kwargs
                                )
                        raise
                
                func = retry_wrapper
            
            # Add validation
            if validation_required:
                data_type = self._infer_data_type(operation_type)
                func = validate_input(
                    data_type=data_type,
                    patient_id=patient_id,
                    clinical_priority=clinical_priority
                )(func)
            
            # Add audit logging
            if audit_required:
                audit_event_type = self._map_operation_to_audit_event(operation_type)
                
                async def audit_wrapper(*args, **kwargs):
                    with AuditContext(
                        event_type=audit_event_type,
                        user_id="system",  # Will be filled by actual user context
                        patient_id=patient_id,
                        resource_accessed=operation_type,
                        function_name=func.__name__
                    ):
                        return await func(*args, **kwargs)
                
                if asyncio.iscoroutinefunction(func):
                    func = audit_wrapper
                else:
                    # For sync functions, create async wrapper
                    async def sync_audit_wrapper(*args, **kwargs):
                        with AuditContext(
                            event_type=audit_event_type,
                            user_id="system",
                            patient_id=patient_id,
                            resource_accessed=operation_type,
                            function_name=func.__name__
                        ):
                            return func(*args, **kwargs)
                    func = sync_audit_wrapper
            
            return func
        
        return decorator
    
    def _infer_data_type(self, operation_type: str) -> DataType:
        """Infer data type from operation type."""
        mapping = {
            "patient_data": DataType.MEDICAL_RECORD,
            "diagnosis": DataType.DIAGNOSIS_CODE,
            "clinical_note": DataType.CLINICAL_NOTE,
            "model_inference": DataType.JSON_DATA,
            "data_validation": DataType.GENERIC_TEXT
        }
        return mapping.get(operation_type, DataType.JSON_DATA)
    
    def _map_operation_to_audit_event(self, operation_type: str) -> AuditEventType:
        """Map operation type to audit event type."""
        mapping = {
            "patient_data": AuditEventType.PATIENT_DATA_ACCESS,
            "model_inference": AuditEventType.MODEL_INFERENCE,
            "data_export": AuditEventType.DATA_EXPORT,
            "system_config": AuditEventType.CONFIGURATION_CHANGE
        }
        return mapping.get(operation_type, AuditEventType.SYSTEM_ACCESS)
    
    async def _handle_health_issue(self, health_status: Dict[str, Any]):
        """Handle health issues."""
        status = HealthStatus(health_status["overall_status"])
        
        if status == HealthStatus.CRITICAL:
            # Critical health issue - take immediate action
            self.medical_logger.log(
                level=LogLevel.CRITICAL,
                message="Critical health issue detected",
                context=health_status
            )
            
            # Could trigger emergency procedures here
            # e.g., activate failsafe mode, notify administrators, etc.
            
        elif status == HealthStatus.UNHEALTHY:
            # Unhealthy system - apply degradation
            self.medical_logger.log(
                level=LogLevel.WARNING,
                message="System health degraded",
                context=health_status
            )
            
            # Activate fallback strategies
            if self.config.fallback_enabled:
                self.degradation_manager._update_degradation_level(0.5)
    
    def _handle_critical_error(self, error: MedicalError):
        """Handle critical medical errors."""
        if error.requires_immediate_response():
            # Log critical error
            self.medical_logger.log(
                level=LogLevel.CRITICAL,
                message=f"Critical medical error: {error.message}",
                context=error.to_dict(),
                phi_protection=PHIProtectionLevel.STRICT
            )
            
            # Create audit event
            self.medical_logger.audit_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                action_taken="critical_error_handled",
                outcome="error",
                details=error.to_dict(),
                phi_protection=PHIProtectionLevel.STRICT
            )
    
    def _audit_error_event(self, error: MedicalError):
        """Audit error events."""
        self.medical_logger.audit_event(
            event_type=AuditEventType.SYSTEM_ACCESS,
            action_taken="error_handled",
            outcome="error",
            details={
                "error_id": error.error_id,
                "error_code": error.error_code.value,
                "category": error.category.value,
                "severity": error.severity.value,
                "message": error.message
            },
            phi_protection=PHIProtectionLevel.STANDARD
        )
    
    def _audit_validation_event(self, event_data: Dict[str, Any]):
        """Audit validation events."""
        self.medical_logger.audit_event(
            event_type=AuditEventType.COMPLIANCE_CHECK,
            action_taken="data_validation",
            outcome="success" if event_data["result"]["is_valid"] else "validation_failed",
            details=event_data,
            phi_protection=PHIProtectionLevel.STANDARD
        )
    
    def _handle_phi_detection(self, phi_data: Dict[str, Any]):
        """Handle PHI detection events."""
        self.medical_logger.log(
            level=LogLevel.PHI_ACCESS,
            message="PHI detected in input",
            context=phi_data,
            phi_protection=PHIProtectionLevel.STRICT
        )
    
    def _audit_shutdown_event(self, event_data: Dict[str, Any]):
        """Audit shutdown events."""
        self.medical_logger.audit_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            action_taken=event_data.get("event", "shutdown"),
            outcome="success",
            details=event_data
        )
    
    def _send_critical_alert(self, error: MedicalError):
        """Send critical error alerts."""
        # This would integrate with alerting systems
        # For now, just log the alert
        self.medical_logger.log(
            level=LogLevel.CRITICAL,
            message=f"CRITICAL ALERT: {error.message}",
            context={
                "alert_type": "critical_error",
                "error_id": error.error_id,
                "error_code": error.error_code.value,
                "category": error.category.value,
                "escalation_required": error.escalation_required
            }
        )
    
    async def shutdown(self, phase: str = "graceful", timeout: float = None):
        """Shutdown the resilience system."""
        logging.info(f"Shutting down resilience system with {phase} phase")
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if timeout:
            await asyncio.wait_for(
                asyncio.gather(*self._background_tasks, return_exceptions=True),
                timeout=timeout
            )
        
        # Perform system shutdown
        await self.shutdown_manager.initiate_shutdown()
        
        # Final audit
        self.medical_logger.audit_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN,
            action_taken="resilience_system_shutdown",
            outcome="success",
            details={
                "shutdown_phase": phase,
                "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
                "background_tasks_stopped": len(self._background_tasks)
            }
        )
        
        logging.info("Resilience system shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "system_info": {
                "initialized": self.is_initialized,
                "uptime_seconds": (datetime.utcnow() - self.startup_time).total_seconds(),
                "health_status": self.current_health_status.value,
                "configuration": self.config.__dict__
            },
            "components": {
                "error_handler": self.error_handler.get_error_statistics(),
                "circuit_breakers": self.circuit_breaker_registry.get_aggregate_metrics(),
                "retry_manager": self.retry_manager.get_metrics(),
                "degradation_manager": self.degradation_manager.get_status(),
                "data_validator": self.data_validator.get_validation_statistics(),
                "shutdown_manager": self.shutdown_manager.get_shutdown_status()
            },
            "logging": self.medical_logger.get_statistics(),
            "performance": self.performance_monitor.get_summary(),
            "metrics": self.system_metrics.copy()
        }


class PerformanceMonitor:
    """Monitor system performance for resilience decisions."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.baseline_performance = 1.0
        self.performance_window = 100  # Keep last 100 measurements
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        # This would collect actual system metrics
        # For now, return mock metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage": 0.5,  # Mock value
            "memory_usage": 0.6,  # Mock value
            "response_time_ms": 100,  # Mock value
            "error_rate": 0.02,  # Mock value
            "throughput": 1000,  # Mock requests per minute
            "overall_performance": 0.85  # Calculated performance score
        }
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.performance_window:
            self.metrics_history = self.metrics_history[-self.performance_window:]
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        return {
            "recent_performance": recent_metrics[-1]["overall_performance"] if recent_metrics else 0,
            "average_performance": sum(m["overall_performance"] for m in recent_metrics) / len(recent_metrics),
            "performance_trend": self._calculate_trend(recent_metrics),
            "metric_samples": len(self.metrics_history)
        }
    
    def _calculate_trend(self, metrics: List[Dict[str, Any]]) -> str:
        """Calculate performance trend."""
        if len(metrics) < 2:
            return "stable"
        
        recent = metrics[-1]["overall_performance"]
        previous = metrics[-2]["overall_performance"]
        
        change = recent - previous
        
        if change > 0.1:
            return "improving"
        elif change < -0.1:
            return "degrading"
        else:
            return "stable"


# Global resilience orchestrator instance
_global_orchestrator: Optional[ResilienceOrchestrator] = None


def get_resilience_orchestrator(config: ResilienceConfig = None) -> ResilienceOrchestrator:
    """Get global resilience orchestrator instance."""
    global _global_orchestrator
    
    if _global_orchestrator is None:
        _global_orchestrator = ResilienceOrchestrator(config)
    
    return _global_orchestrator


async def initialize_resilience_system(config: ResilienceConfig = None) -> bool:
    """Initialize the global resilience system."""
    orchestrator = get_resilience_orchestrator(config)
    return await orchestrator.initialize()


def with_medical_resilience(
    operation_type: str = "general",
    patient_id: Optional[str] = None,
    clinical_priority: str = "normal"
) -> Callable:
    """Convenience decorator for medical resilience."""
    orchestrator = get_resilience_orchestrator()
    return orchestrator.execute_with_resilience(
        operation_type=operation_type,
        patient_id=patient_id,
        clinical_priority=clinical_priority
    )


# Convenience functions for common medical operations
def critical_medical_operation(patient_id: str):
    """Decorator for critical medical operations."""
    return with_medical_resilience(
        operation_type="patient_data",
        patient_id=patient_id,
        clinical_priority="critical"
    )


def model_inference_operation(operation_type: str = "inference"):
    """Decorator for model inference operations."""
    return with_medical_resilience(
        operation_type=operation_type,
        clinical_priority="high"
    )


def data_validation_operation(data_type: DataType):
    """Decorator for data validation operations."""
    def decorator(func: Callable):
        # Add validation
        func = validate_input(
            data_type=data_type,
            validation_level=ValidationLevel.STRICT
        )(func)
        
        # Add audit logging
        func = log_medical_operation(
            f"{func.__name__}_validation",
            AuditEventType.COMPLIANCE_CHECK
        )(func)
        
        return func
    
    return decorator