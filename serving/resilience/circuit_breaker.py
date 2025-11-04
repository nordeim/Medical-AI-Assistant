"""
Medical AI Resilience - Circuit Breaker Pattern
Circuit breaker implementation with medical system isolation and safety considerations.
"""

import asyncio
import time
from enum import Enum, auto
from typing import Any, Callable, Optional, Dict, List, Union
from datetime import datetime, timedelta
import json
from .errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory, 
    MedicalErrorSeverity, create_service_unavailable_error
)


class CircuitBreakerState(Enum):
    """Circuit breaker states for medical system isolation."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureType(Enum):
    """Types of failures for medical systems."""
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    MEDICAL_LOGIC = "medical_logic"  # Clinical reasoning failure
    DATA_INTEGRITY = "data_integrity"  # Medical data corruption
    PATIENT_SAFETY = "patient_safety"  # Patient safety violation
    REGULATORY = "regulatory"  # Compliance violation
    UNKNOWN = "unknown"


class CircuitBreaker:
    """Circuit breaker with medical-specific failure handling."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: tuple = (Exception,),
        success_threshold: int = 3,
        timeout_threshold: float = 30.0,
        medical_isolation: bool = True,
        patient_safety_mode: bool = False
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.success_threshold = success_threshold
        self.timeout_threshold = timeout_threshold
        self.medical_isolation = medical_isolation
        self.patient_safety_mode = patient_safety_mode
        
        # State tracking
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.opened_at: Optional[datetime] = None
        self.half_opened_at: Optional[datetime] = None
        
        # Medical-specific tracking
        self.medical_failures: List[Dict[str, Any]] = []
        self.patient_safety_violations = 0
        self.audit_trail: List[Dict[str, Any]] = []
        
        # Metrics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Make circuit breaker callable as decorator."""
        async def async_wrapper(*args, **kwargs):
            return await self.call(func, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            return self.call_sync(func, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def call(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """Execute function with circuit breaker protection."""
        self.total_requests += 1
        
        # Check circuit state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                await self._record_blocked_request()
                raise create_service_unavailable_error(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
        
        try:
            # Execute function with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.timeout_threshold
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._execute_with_timeout(func, *args, **kwargs)
                )
            
            # Success handling
            await self._record_success()
            return result
            
        except asyncio.TimeoutError:
            await self._record_failure(
                FailureType.TIMEOUT,
                f"Function {func.__name__} timed out after {self.timeout_threshold}s"
            )
            raise
            
        except self.expected_exception as e:
            # Determine failure type for medical systems
            failure_type = self._classify_failure(e)
            await self._record_failure(failure_type, str(e))
            raise
            
        except Exception as e:
            # Unknown error - treat as critical for medical systems
            await self._record_failure(FailureType.UNKNOWN, str(e))
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Synchronous version of call method."""
        self.total_requests += 1
        
        # Check circuit state
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self._record_blocked_request_sync()
                raise create_service_unavailable_error(
                    f"Circuit breaker '{self.name}' is OPEN"
                )
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(func, *args, **kwargs)
            
            # Success handling
            self._record_success_sync()
            return result
            
        except Exception as e:
            failure_type = self._classify_failure(e)
            self._record_failure_sync(failure_type, str(e))
            raise
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection."""
        if hasattr(asyncio, 'timeout'):  # Python 3.11+
            with asyncio.timeout(self.timeout_threshold):
                return func(*args, **kwargs)
        else:
            return asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: func(*args, **kwargs)
                ),
                timeout=self.timeout_threshold
            )
    
    def _classify_failure(self, exception: Exception) -> FailureType:
        """Classify failure type for medical systems."""
        exception_name = type(exception).__name__.lower()
        
        if any(keyword in exception_name for keyword in ['timeout', 'timedout']):
            return FailureType.TIMEOUT
        elif any(keyword in exception_name for keyword in ['connection', 'network', 'dns']):
            return FailureType.CONNECTION
        elif any(keyword in exception_name for keyword in ['medical', 'clinical', 'safety']):
            return FailureType.MEDICAL_LOGIC
        elif any(keyword in exception_name for keyword in ['data', 'integrity', 'corruption']):
            return FailureType.DATA_INTEGRITY
        elif any(keyword in exception_name for keyword in ['patient', 'safety', 'harm']):
            return FailureType.PATIENT_SAFETY
        elif any(keyword in exception_name for keyword in ['hipaa', 'regulatory', 'compliance']):
            return FailureType.REGULATORY
        else:
            return FailureType.UNKNOWN
    
    async def _record_success(self):
        """Record successful request."""
        self.total_successes += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        
        self._audit_request("success")
    
    async def _record_failure(self, failure_type: FailureType, message: str):
        """Record failed request."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        # Medical-specific handling
        await self._handle_medical_failure(failure_type, message)
        
        # State transition
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to_open()
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self._transition_to_open()
        
        self._audit_request("failure", {"failure_type": failure_type.value, "message": message})
    
    async def _handle_medical_failure(self, failure_type: FailureType, message: str):
        """Handle medical-specific failures."""
        failure_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "failure_type": failure_type.value,
            "message": message,
            "circuit_breaker": self.name
        }
        
        self.medical_failures.append(failure_info)
        
        # Special handling for patient safety violations
        if failure_type == FailureType.PATIENT_SAFETY:
            self.patient_safety_violations += 1
            # In patient safety mode, be more aggressive with circuit breaking
            if self.patient_safety_mode:
                self.failure_threshold = max(1, self.failure_threshold // 2)
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.opened_at = datetime.utcnow()
        self._audit_transition("OPEN")
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.half_opened_at = datetime.utcnow()
        self.success_count = 0
        self._audit_transition("HALF_OPEN")
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self._audit_transition("CLOSED")
    
    def _audit_request(self, outcome: str, details: Optional[Dict] = None):
        """Audit individual request."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "circuit_breaker": self.name,
            "state": self.state.value,
            "outcome": outcome,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            **(details or {})
        }
        
        self.audit_trail.append(audit_entry)
        
        # Keep only recent audit entries
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.audit_trail = [
            entry for entry in self.audit_trail
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
    
    def _audit_transition(self, new_state: str):
        """Audit state transitions."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "state_transition",
            "circuit_breaker": self.name,
            "old_state": self.state.value,
            "new_state": new_state,
            "failure_count": self.failure_count
        }
        
        self.audit_trail.append(audit_entry)
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status and metrics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "half_opened_at": self.half_opened_at.isoformat() if self.half_opened_at else None,
            "patient_safety_violations": self.patient_safety_violations,
            "medical_failures_count": len(self.medical_failures),
            "configuration": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "success_threshold": self.success_threshold,
                "timeout_threshold": self.timeout_threshold,
                "medical_isolation": self.medical_isolation,
                "patient_safety_mode": self.patient_safety_mode
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.opened_at = None
        self.half_opened_at = None
        self.patient_safety_violations = 0
        self._audit_transition("RESET")
    
    # Synchronous versions for backward compatibility
    def _record_blocked_request(self):
        """Async version of blocked request recording."""
        self._audit_request("blocked", {"reason": "circuit_open"})
    
    def _record_blocked_request_sync(self):
        """Sync version of blocked request recording."""
        self._audit_request("blocked", {"reason": "circuit_open"})
    
    def _record_success_sync(self):
        """Sync version of success recording."""
        self.total_successes += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        
        self._audit_request("success")
    
    def _record_failure_sync(self, failure_type: FailureType, message: str):
        """Sync version of failure recording."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        # Medical-specific handling
        self.medical_failures.append({
            "timestamp": datetime.utcnow().isoformat(),
            "failure_type": failure_type.value,
            "message": message,
            "circuit_breaker": self.name
        })
        
        # State transition
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to_open()
        elif self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self._transition_to_open()
        
        self._audit_request("failure", {"failure_type": failure_type.value, "message": message})


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._default_config = {
            "failure_threshold": 5,
            "recovery_timeout": 60.0,
            "success_threshold": 3,
            "timeout_threshold": 30.0,
            "medical_isolation": True,
            "patient_safety_mode": False
        }
    
    def create(
        self,
        name: str,
        **config
    ) -> CircuitBreaker:
        """Create a new circuit breaker."""
        if name in self._circuit_breakers:
            return self._circuit_breakers[name]
        
        final_config = {**self._default_config, **config}
        circuit_breaker = CircuitBreaker(name, **final_config)
        self._circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._circuit_breakers.get(name)
    
    def remove(self, name: str):
        """Remove circuit breaker."""
        if name in self._circuit_breakers:
            del self._circuit_breakers[name]
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {name: cb.get_status() for name, cb in self._circuit_breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self._circuit_breakers.values():
            cb.reset()
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics across all circuit breakers."""
        if not self._circuit_breakers:
            return {}
        
        total_requests = sum(cb.total_requests for cb in self._circuit_breakers.values())
        total_failures = sum(cb.total_failures for cb in self._circuit_breakers.values())
        total_successes = sum(cb.total_successes for cb in self._circuit_breakers.values())
        
        open_circuits = sum(1 for cb in self._circuit_breakers.values() if cb.state == CircuitBreakerState.OPEN)
        half_open_circuits = sum(1 for cb in self._circuit_breakers.values() if cb.state == CircuitBreakerState.HALF_OPEN)
        
        return {
            "total_circuit_breakers": len(self._circuit_breakers),
            "total_requests": total_requests,
            "total_failures": total_failures,
            "total_successes": total_successes,
            "failure_rate": total_failures / max(total_requests, 1),
            "open_circuits": open_circuits,
            "half_open_circuits": half_open_circuits,
            "closed_circuits": len(self._circuit_breakers) - open_circuits - half_open_circuits,
            "individual_statuses": self.get_all_status()
        }


# Global circuit breaker registry
circuit_breaker_registry = CircuitBreakerRegistry()


def medical_circuit_breaker(
    name: str = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    timeout_threshold: float = 30.0,
    patient_safety_mode: bool = False
) -> Callable:
    """Decorator for creating medical circuit breakers."""
    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__name__}"
        circuit_breaker = circuit_breaker_registry.create(
            breaker_name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            timeout_threshold=timeout_threshold,
            patient_safety_mode=patient_safety_mode
        )
        return circuit_breaker(func)
    return decorator