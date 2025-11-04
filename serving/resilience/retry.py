"""
Medical AI Resilience - Retry Mechanisms with Exponential Backoff
Retry mechanisms with medical safety considerations and appropriate backoff strategies.
"""

import asyncio
import time
from enum import Enum, auto
from typing import Any, Callable, Optional, Dict, List, Union, Type, Tuple
from datetime import datetime, timedelta
import random
import json
from .errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory,
    MedicalErrorSeverity, create_model_failure_error
)


class RetryStrategy(Enum):
    """Retry strategies for different medical scenarios."""
    FIXED = "fixed"                    # Fixed delay between retries
    EXPONENTIAL = "exponential"        # Exponential backoff
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"  # Exponential with random jitter
    LINEAR = "linear"                  # Linear backoff
    MEDICAL_SAFETY = "medical_safety"  # Medical-specific retry strategy
    CIRCUMSPECT = "circumspect"        # Very conservative for critical operations
    AGGRESSIVE = "aggressive"          # Fast retries for non-critical operations


class RetryCondition(Enum):
    """Conditions that determine when to retry."""
    ALWAYS = "always"                        # Retry all errors
    ON_SPECIFIC_EXCEPTIONS = "on_specific"   # Retry on specific exceptions
    ON_MEDICAL_ERRORS = "on_medical_errors"  # Retry medical-specific errors
    ON_TRANSIENT_ERRORS = "on_transient"     # Retry transient errors only
    NEVER = "never"                          # Never retry
    ON_TIMEOUT = "on_timeout"                # Retry only on timeouts


class MedicalRetryContext:
    """Context for medical retry operations."""
    
    def __init__(
        self,
        patient_id: Optional[str] = None,
        clinical_priority: str = "normal",  # low, normal, high, critical
        retry_count: int = 0,
        max_retries: int = 3,
        operation_type: str = "general",
        data_sensitivity: str = "normal",  # normal, sensitive, critical
        regulatory_constraints: List[str] = None
    ):
        self.patient_id = patient_id
        self.clinical_priority = clinical_priority
        self.retry_count = retry_count
        self.max_retries = max_retries
        self.operation_type = operation_type
        self.data_sensitivity = data_sensitivity
        self.regulatory_constraints = regulatory_constraints or []
        self.start_time = datetime.utcnow()
        self.audit_trail: List[Dict[str, Any]] = []
    
    def add_audit_entry(self, action: str, details: Dict[str, Any]):
        """Add entry to audit trail."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "retry_count": self.retry_count,
            "details": details
        }
        self.audit_trail.append(entry)
    
    def should_continue_retrying(self) -> bool:
        """Determine if retries should continue based on medical context."""
        if self.retry_count >= self.max_retries:
            return False
        
        # Critical operations should have stricter retry limits
        if self.clinical_priority == "critical" and self.retry_count >= 2:
            return False
        
        # High sensitivity data should be retried less
        if self.data_sensitivity == "critical" and self.retry_count >= 1:
            return False
        
        # Check time limits
        max_duration = self._get_max_duration()
        if datetime.utcnow() - self.start_time > max_duration:
            return False
        
        return True
    
    def _get_max_duration(self) -> timedelta:
        """Get maximum duration for retries based on context."""
        if self.clinical_priority == "critical":
            return timedelta(seconds=30)  # Very fast failure for critical ops
        elif self.clinical_priority == "high":
            return timedelta(minutes=1)   # Quick resolution for high priority
        elif self.data_sensitivity == "critical":
            return timedelta(minutes=2)   # Moderate timeout for sensitive data
        else:
            return timedelta(minutes=5)   # Standard timeout
    
    def get_recommended_strategy(self) -> RetryStrategy:
        """Get recommended retry strategy based on context."""
        if self.clinical_priority == "critical":
            return RetryStrategy.CIRCUMSPECT
        elif self.clinical_priority == "high":
            return RetryStrategy.EXPONENTIAL_WITH_JITTER
        elif self.operation_type == "data_retrieval":
            return RetryStrategy.MEDICAL_SAFETY
        elif self.operation_type == "model_inference":
            return RetryStrategy.EXPONENTIAL
        elif self.data_sensitivity == "critical":
            return RetryStrategy.CIRCUMSPECT
        else:
            return RetryStrategy.EXPONENTIAL_WITH_JITTER


class RetryConfig:
    """Configuration for retry mechanisms with medical considerations."""
    
    def __init__(
        self,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_WITH_JITTER,
        condition: RetryCondition = RetryCondition.ON_MEDICAL_ERRORS,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        jitter_range: float = 0.1,
        timeout: float = 300.0,  # 5 minutes total timeout
        medical_safe_mode: bool = False,
        exception_whitelist: List[Type[Exception]] = None,
        exception_blacklist: List[Type[Exception]] = None,
        retryable_error_codes: List[MedicalErrorCode] = None
    ):
        self.strategy = strategy
        self.condition = condition
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter_range = jitter_range
        self.timeout = timeout
        self.medical_safe_mode = medical_safe_mode
        self.exception_whitelist = exception_whitelist or []
        self.exception_blacklist = exception_blacklist or []
        self.retryable_error_codes = retryable_error_codes or []
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
            
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * attempt
            
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.backoff_factor ** attempt)
            
        elif self.strategy == RetryStrategy.EXPONENTIAL_WITH_JITTER:
            delay = self.base_delay * (self.backoff_factor ** attempt)
            jitter = random.uniform(-self.jitter_range, self.jitter_range) * delay
            delay = delay + jitter
            
        elif self.strategy == RetryStrategy.MEDICAL_SAFETY:
            # Conservative approach for medical systems
            delay = self.base_delay * (1.5 ** attempt)  # Slower exponential
            delay = min(delay, self.max_delay)
            
        elif self.strategy == RetryStrategy.CIRCUMSPECT:
            # Very conservative for critical operations
            delay = self.base_delay * (2.0 ** attempt)  # Faster exponential
            delay = min(delay, self.max_delay / 2)  # Lower maximum
            
        elif self.strategy == RetryStrategy.AGGRESSIVE:
            # Fast retries for non-critical operations
            delay = self.base_delay * (1.2 ** attempt)
            
        else:
            delay = self.base_delay
        
        return min(delay, self.max_delay)
    
    def should_retry(
        self,
        exception: Exception,
        attempt: int,
        context: MedicalRetryContext
    ) -> Tuple[bool, str]:
        """Determine if operation should be retried."""
        
        # Check context limits
        if not context.should_continue_retrying():
            return False, "Context retry limit exceeded"
        
        # Check timeout
        if datetime.utcnow() - context.start_time > timedelta(seconds=self.timeout):
            return False, "Global timeout exceeded"
        
        # Check attempt limit
        if attempt >= self.max_retries:
            return False, "Max retry attempts exceeded"
        
        # Check condition
        if self.condition == RetryCondition.NEVER:
            return False, "Never retry condition"
        
        if self.condition == RetryCondition.ALWAYS:
            return True, "Always retry condition"
        
        if self.condition == RetryCondition.ON_TIMEOUT:
            if "timeout" in str(exception).lower():
                return True, "Timeout condition met"
            return False, "Not a timeout error"
        
        # Exception-based filtering
        exception_type = type(exception)
        
        # Blacklist check
        if exception_type in self.exception_blacklist:
            return False, f"Exception {exception_type.__name__} in blacklist"
        
        # Whitelist check
        if self.condition == RetryCondition.ON_SPECIFIC_EXCEPTIONS:
            if self.exception_whitelist and exception_type not in self.exception_whitelist:
                return False, f"Exception {exception_type.__name__} not in whitelist"
        
        # Medical error codes check
        if hasattr(exception, 'error_code') and isinstance(exception.error_code, MedicalErrorCode):
            if self.retryable_error_codes and exception.error_code not in self.retryable_error_codes:
                return False, f"Error code {exception.error_code.value} not retryable"
        
        # Default medical safety check
        if self.medical_safe_mode:
            # In medical safe mode, be more conservative
            if self.clinical_priority == "critical" and attempt >= 1:
                return False, "Critical operation retry limit in safe mode"
        
        return True, f"Retry condition met for {exception_type.__name__}"


class RetryManager:
    """Central retry management with medical context awareness."""
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        self.default_config = default_config or RetryConfig()
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.audit_trail: List[Dict[str, Any]] = []
    
    async def retry_with_context(
        self,
        func: Callable,
        context: MedicalRetryContext,
        config: Optional[RetryConfig] = None,
        *args,
        **kwargs
    ) -> Any:
        """Retry function with medical context."""
        config = config or self.default_config
        
        # Audit start
        context.add_audit_entry("retry_started", {
            "function": func.__name__,
            "strategy": config.strategy.value,
            "max_retries": config.max_retries
        })
        
        last_exception = None
        attempt = 0
        
        while context.should_continue_retrying() and attempt <= config.max_retries:
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=config.timeout)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    )
                
                # Success
                context.add_audit_entry("retry_success", {
                    "attempt": attempt,
                    "total_attempts": context.retry_count + 1
                })
                
                self._record_success(func.__name__, context, attempt)
                return result
                
            except Exception as exception:
                last_exception = exception
                attempt += 1
                context.retry_count = attempt
                
                # Check if should retry
                should_retry, reason = config.should_retry(exception, attempt, context)
                
                context.add_audit_entry("retry_attempt", {
                    "attempt": attempt,
                    "exception": type(exception).__name__,
                    "reason": reason,
                    "should_retry": should_retry
                })
                
                if not should_retry:
                    break
                
                # Calculate delay
                delay = config.calculate_delay(attempt)
                
                # Audit retry
                context.add_audit_entry("retry_scheduled", {
                    "delay": delay,
                    "attempt": attempt
                })
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries exhausted
        context.add_audit_entry("retry_exhausted", {
            "final_attempt": attempt,
            "total_attempts": attempt + 1,
            "last_exception": type(last_exception).__name__ if last_exception else None
        })
        
        self._record_failure(func.__name__, context, attempt, last_exception)
        raise last_exception
    
    def retry_sync_with_context(
        self,
        func: Callable,
        context: MedicalRetryContext,
        config: Optional[RetryConfig] = None,
        *args,
        **kwargs
    ) -> Any:
        """Synchronous version of retry_with_context."""
        config = config or self.default_config
        
        context.add_audit_entry("retry_started", {
            "function": func.__name__,
            "strategy": config.strategy.value,
            "max_retries": config.max_retries
        })
        
        last_exception = None
        attempt = 0
        
        while context.should_continue_retrying() and attempt <= config.max_retries:
            try:
                result = func(*args, **kwargs)
                
                context.add_audit_entry("retry_success", {
                    "attempt": attempt,
                    "total_attempts": context.retry_count + 1
                })
                
                self._record_success(func.__name__, context, attempt)
                return result
                
            except Exception as exception:
                last_exception = exception
                attempt += 1
                context.retry_count = attempt
                
                should_retry, reason = config.should_retry(exception, attempt, context)
                
                context.add_audit_entry("retry_attempt", {
                    "attempt": attempt,
                    "exception": type(exception).__name__,
                    "reason": reason,
                    "should_retry": should_retry
                })
                
                if not should_retry:
                    break
                
                delay = config.calculate_delay(attempt)
                context.add_audit_entry("retry_scheduled", {
                    "delay": delay,
                    "attempt": attempt
                })
                
                time.sleep(delay)
        
        context.add_audit_entry("retry_exhausted", {
            "final_attempt": attempt,
            "total_attempts": attempt + 1,
            "last_exception": type(last_exception).__name__ if last_exception else None
        })
        
        self._record_failure(func.__name__, context, attempt, last_exception)
        raise last_exception
    
    def _record_success(self, func_name: str, context: MedicalRetryContext, attempt: int):
        """Record successful retry."""
        key = f"{func_name}_{context.operation_type}"
        if key not in self.metrics:
            self.metrics[key] = {
                "total_attempts": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "total_retry_count": 0
            }
        
        self.metrics[key]["total_attempts"] += 1
        if attempt > 0:
            self.metrics[key]["successful_retries"] += 1
            self.metrics[key]["total_retry_count"] += attempt
        
        # Audit trail
        self.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "retry_success",
            "function": func_name,
            "context": context.__dict__,
            "attempts": attempt + 1
        })
    
    def _record_failure(self, func_name: str, context: MedicalRetryContext, attempt: int, exception: Exception):
        """Record failed retry."""
        key = f"{func_name}_{context.operation_type}"
        if key not in self.metrics:
            self.metrics[key] = {
                "total_attempts": 0,
                "successful_retries": 0,
                "failed_retries": 0,
                "total_retry_count": 0
            }
        
        self.metrics[key]["total_attempts"] += 1
        self.metrics[key]["failed_retries"] += 1
        self.metrics[key]["total_retry_count"] += attempt
        
        # Audit trail
        self.audit_trail.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "retry_failure",
            "function": func_name,
            "context": context.__dict__,
            "attempts": attempt + 1,
            "final_exception": type(exception).__name__ if exception else None
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry metrics."""
        total_attempts = sum(m["total_attempts"] for m in self.metrics.values())
        total_successful = sum(m["successful_retries"] for m in self.metrics.values())
        total_failed = sum(m["failed_retries"] for m in self.metrics.values())
        
        return {
            "total_operations": total_attempts,
            "successful_retries": total_successful,
            "failed_retries": total_failed,
            "success_rate": total_successful / max(total_attempts, 1),
            "detailed_metrics": self.metrics,
            "recent_audit_entries": self.audit_trail[-100:],  # Keep last 100 entries
            "total_audit_entries": len(self.audit_trail)
        }


# Global retry manager
retry_manager = RetryManager()


def medical_retry(
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_WITH_JITTER,
    condition: RetryCondition = RetryCondition.ON_MEDICAL_ERRORS,
    max_retries: int = 3,
    timeout: float = 300.0,
    patient_id: Optional[str] = None,
    clinical_priority: str = "normal",
    operation_type: str = "general",
    data_sensitivity: str = "normal"
) -> Callable:
    """Decorator for medical retry operations."""
    
    def decorator(func: Callable):
        config = RetryConfig(
            strategy=strategy,
            condition=condition,
            max_retries=max_retries,
            timeout=timeout
        )
        
        async def async_wrapper(*args, **kwargs):
            context = MedicalRetryContext(
                patient_id=patient_id,
                clinical_priority=clinical_priority,
                operation_type=operation_type,
                data_sensitivity=data_sensitivity,
                max_retries=max_retries
            )
            return await retry_manager.retry_with_context(func, context, config, *args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            context = MedicalRetryContext(
                patient_id=patient_id,
                clinical_priority=clinical_priority,
                operation_type=operation_type,
                data_sensitivity=data_sensitivity,
                max_retries=max_retries
            )
            return retry_manager.retry_sync_with_context(func, context, config, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Pre-configured retry strategies for common medical scenarios
def critical_operation_retry(max_retries: int = 2, timeout: float = 30.0) -> Callable:
    """Retry configuration for critical medical operations."""
    return medical_retry(
        strategy=RetryStrategy.CIRCUMSPECT,
        condition=RetryCondition.ON_TRANSIENT_ERRORS,
        max_retries=max_retries,
        timeout=timeout,
        clinical_priority="critical",
        data_sensitivity="critical"
    )


def model_inference_retry(max_retries: int = 3, timeout: float = 60.0) -> Callable:
    """Retry configuration for model inference operations."""
    return medical_retry(
        strategy=RetryStrategy.EXPONENTIAL,
        condition=RetryCondition.ON_MEDICAL_ERRORS,
        max_retries=max_retries,
        timeout=timeout,
        operation_type="model_inference"
    )


def data_retrieval_retry(max_retries: int = 3, timeout: float = 120.0) -> Callable:
    """Retry configuration for data retrieval operations."""
    return medical_retry(
        strategy=RetryStrategy.MEDICAL_SAFETY,
        condition=RetryCondition.ON_SPECIFIC_EXCEPTIONS,
        max_retries=max_retries,
        timeout=timeout,
        operation_type="data_retrieval",
        medical_safe_mode=True
    )