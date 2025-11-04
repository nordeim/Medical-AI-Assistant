"""
Medical AI Resilience - Fallback Models and Degradation Strategies
Safe degradation strategies for medical AI systems with multiple fallback levels.
"""

import asyncio
import json
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from .errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory,
    MedicalErrorSeverity, create_model_failure_error
)
from .circuit_breaker import circuit_breaker_registry
from .retry import medical_retry, RetryStrategy, RetryCondition


class DegradationLevel(Enum):
    """Degradation levels for medical AI systems."""
    FULL_CAPABILITY = "full_capability"          # All features working
    REDUCED_PERFORMANCE = "reduced_performance"  # Some features degraded
    BASIC_FUNCTIONALITY = "basic_functionality"  # Only essential features
    EMERGENCY_MODE = "emergency_mode"            # Minimal functionality
    FAILSAFE_MODE = "failsafe_mode"              # Only safety checks
    COMPLETE_FAILURE = "complete_failure"        # System unavailable


class FallbackType(Enum):
    """Types of fallback mechanisms."""
    MODEL_FALLBACK = "model_fallback"           # Switch to backup model
    ALGORITHM_FALLBACK = "algorithm_fallback"    # Use simpler algorithm
    RULE_BASED_FALLBACK = "rule_based_fallback" # Use rule-based system
    EXTERNAL_SERVICE = "external_service"       # Use external service
    MANUAL_REVIEW = "manual_review"             # Human review required
    CACHED_RESPONSE = "cached_response"         # Use cached response
    SAFE_DEFAULT = "safe_default"               # Return safe default
    PATIENT_SAFETY_MODE = "patient_safety_mode" # Prioritize safety


class FallbackStrategy(ABC):
    """Abstract base class for fallback strategies."""
    
    def __init__(self, name: str, priority: int):
        self.name = name
        self.priority = priority  # Lower number = higher priority
        self.enabled = True
        self.metrics = {
            "total_attempts": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "average_response_time": 0.0,
            "last_used": None
        }
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute fallback strategy."""
        pass
    
    def is_available(self) -> bool:
        """Check if fallback strategy is available."""
        return self.enabled and self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> bool:
        """Check internal availability."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of fallback strategy."""
        return {
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "available": self.is_available(),
            "metrics": self.metrics.copy(),
            "type": self.__class__.__name__
        }


class ModelFallbackStrategy(FallbackStrategy):
    """Fallback to a different, more reliable model."""
    
    def __init__(
        self,
        primary_model: Any,
        fallback_model: Any,
        model_threshold: float = 0.7,
        confidence_threshold: float = 0.8
    ):
        super().__init__("model_fallback", priority=1)
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.model_threshold = model_threshold
        self.confidence_threshold = confidence_threshold
    
    async def execute(self, input_data: Any, **kwargs) -> Tuple[Any, str]:
        """Execute model fallback."""
        self.metrics["total_attempts"] += 1
        start_time = time.time()
        
        try:
            # Try primary model first
            if hasattr(self.primary_model, 'predict'):
                primary_result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.primary_model.predict(input_data, **kwargs)
                )
                
                # Check if primary model result meets confidence threshold
                if self._check_confidence(primary_result):
                    self._record_success(start_time)
                    return primary_result, "primary_model"
            
            # Fall back to secondary model
            fallback_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.fallback_model.predict(input_data, **kwargs)
            )
            
            self._record_success(start_time)
            return fallback_result, "fallback_model"
            
        except Exception as e:
            self._record_failure(start_time)
            raise create_model_failure_error(f"Model fallback failed: {str(e)}")
    
    def _check_confidence(self, result: Any) -> bool:
        """Check if model result meets confidence threshold."""
        if hasattr(result, 'confidence'):
            return result.confidence >= self.confidence_threshold
        elif hasattr(result, 'score'):
            return result.score >= self.model_threshold
        return True  # Assume OK if no confidence measure
    
    def _check_availability(self) -> bool:
        """Check if models are available."""
        try:
            # Simple health check for models
            return (
                self.primary_model is not None and
                self.fallback_model is not None and
                hasattr(self.primary_model, 'predict') and
                hasattr(self.fallback_model, 'predict')
            )
        except Exception:
            return False
    
    def _record_success(self, start_time: float):
        """Record successful execution."""
        elapsed = time.time() - start_time
        self.metrics["successful_fallbacks"] += 1
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * 
             (self.metrics["successful_fallbacks"] - 1) + elapsed) /
            self.metrics["successful_fallbacks"]
        )
        self.metrics["last_used"] = datetime.utcnow().isoformat()
    
    def _record_failure(self, start_time: float):
        """Record failed execution."""
        self.metrics["failed_fallbacks"] += 1
        self.metrics["last_used"] = datetime.utcnow().isoformat()


class RuleBasedFallbackStrategy(FallbackStrategy):
    """Fallback to rule-based medical decision making."""
    
    def __init__(self, rules: Dict[str, Callable]):
        super().__init__("rule_based_fallback", priority=2)
        self.rules = rules
        self.safety_rules = [
            self._ensure_patient_safety,
            self._validate_clinical_context,
            self._check_regulatory_compliance
        ]
    
    async def execute(self, input_data: Any, **kwargs) -> Tuple[Any, str]:
        """Execute rule-based fallback."""
        self.metrics["total_attempts"] += 1
        start_time = time.time()
        
        try:
            # Apply safety rules first
            for safety_rule in self.safety_rules:
                safety_result = await self._apply_safety_rule(safety_rule, input_data, **kwargs)
                if not safety_result["safe"]:
                    self._record_failure(start_time)
                    raise create_model_failure_error(
                        f"Safety rule violation: {safety_result['message']}"
                    )
            
            # Try to apply domain-specific rules
            for rule_name, rule_func in self.rules.items():
                try:
                    result = await self._apply_rule(rule_func, input_data, **kwargs)
                    if result is not None:
                        self._record_success(start_time)
                        return result, f"rule_{rule_name}"
                except Exception as e:
                    logging.warning(f"Rule {rule_name} failed: {e}")
                    continue
            
            # If no rules apply, return safe default
            default_result = await self._get_safe_default(input_data, **kwargs)
            self._record_success(start_time)
            return default_result, "safe_default"
            
        except Exception as e:
            self._record_failure(start_time)
            raise create_model_failure_error(f"Rule-based fallback failed: {str(e)}")
    
    async def _apply_safety_rule(self, rule_func: Callable, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Apply safety rule and return safety assessment."""
        try:
            if asyncio.iscoroutinefunction(rule_func):
                result = await rule_func(input_data, **kwargs)
            else:
                result = rule_func(input_data, **kwargs)
            
            if isinstance(result, dict):
                return result
            else:
                return {"safe": bool(result), "message": "Safety rule executed"}
                
        except Exception as e:
            return {"safe": False, "message": f"Safety rule failed: {e}"}
    
    async def _apply_rule(self, rule_func: Callable, input_data: Any, **kwargs) -> Any:
        """Apply a single rule."""
        if asyncio.iscoroutinefunction(rule_func):
            return await rule_func(input_data, **kwargs)
        else:
            return rule_func(input_data, **kwargs)
    
    async def _get_safe_default(self, input_data: Any, **kwargs) -> Any:
        """Get safe default response."""
        return {
            "status": "degraded_mode",
            "message": "Operating in rule-based fallback mode",
            "recommendation": "manual_review_required",
            "confidence": 0.1,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _check_availability(self) -> bool:
        """Check if rule-based system is available."""
        return len(self.rules) > 0
    
    def _record_success(self, start_time: float):
        """Record successful execution."""
        elapsed = time.time() - start_time
        self.metrics["successful_fallbacks"] += 1
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * 
             (self.metrics["successful_fallbacks"] - 1) + elapsed) /
            self.metrics["successful_fallbacks"]
        )
        self.metrics["last_used"] = datetime.utcnow().isoformat()
    
    def _record_failure(self, start_time: float):
        """Record failed execution."""
        self.metrics["failed_fallbacks"] += 1
        self.metrics["last_used"] = datetime.utcnow().isoformat()
    
    # Default safety rules
    async def _ensure_patient_safety(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Ensure patient safety is not compromised."""
        # Basic safety checks
        if isinstance(input_data, dict):
            # Check for dangerous instructions
            dangerous_keywords = ["harm", "kill", "suicide", "violence"]
            text_content = json.dumps(input_data).lower()
            
            if any(keyword in text_content for keyword in dangerous_keywords):
                return {
                    "safe": False,
                    "message": "Potential harm detected in input",
                    "action": "block_processing"
                }
        
        return {"safe": True, "message": "Patient safety check passed"}
    
    async def _validate_clinical_context(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Validate clinical context."""
        # Basic clinical validation
        if isinstance(input_data, dict):
            # Ensure required medical fields are present
            required_fields = ["patient_id", "clinical_data"]
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                return {
                    "safe": False,
                    "message": f"Missing required medical fields: {missing_fields}",
                    "action": "require_manual_input"
                }
        
        return {"safe": True, "message": "Clinical context validated"}
    
    async def _check_regulatory_compliance(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Check regulatory compliance."""
        # Basic HIPAA compliance check
        if isinstance(input_data, dict):
            # Ensure PHI is properly protected
            if "phi_data" in input_data:
                if not self._validate_phi_handling(input_data["phi_data"]):
                    return {
                        "safe": False,
                        "message": "PHI handling compliance violation",
                        "action": "audit_required"
                    }
        
        return {"safe": True, "message": "Regulatory compliance verified"}
    
    def _validate_phi_handling(self, phi_data: Any) -> bool:
        """Validate PHI handling."""
        # Basic PHI validation
        if not isinstance(phi_data, dict):
            return False
        
        # Check for proper anonymization
        sensitive_fields = ["ssn", "medical_record_number", "biometric_data"]
        for field in sensitive_fields:
            if field in phi_data and phi_data[field] is not None:
                # Should be hashed or encrypted in real implementation
                return False
        
        return True


class DegradationManager:
    """Central manager for degradation and fallback strategies."""
    
    def __init__(self):
        self.strategies: List[FallbackStrategy] = []
        self.current_degradation_level = DegradationLevel.FULL_CAPABILITY
        self.degradation_thresholds = {
            DegradationLevel.FULL_CAPABILITY: 1.0,      # 100% performance
            DegradationLevel.REDUCED_PERFORMANCE: 0.7,  # 70% performance
            DegradationLevel.BASIC_FUNCTIONALITY: 0.5,  # 50% performance
            DegradationLevel.EMERGENCY_MODE: 0.3,       # 30% performance
            DegradationLevel.FAILSAFE_MODE: 0.1,        # 10% performance
            DegradationLevel.COMPLETE_FAILURE: 0.0      # 0% performance
        }
        self.performance_history: List[Dict[str, Any]] = []
        self.fallback_audit: List[Dict[str, Any]] = []
    
    def add_strategy(self, strategy: FallbackStrategy):
        """Add fallback strategy."""
        self.strategies.append(strategy)
        # Sort by priority (lower number = higher priority)
        self.strategies.sort(key=lambda s: s.priority)
    
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        input_data: Any,
        current_performance: float,
        **kwargs
    ) -> Tuple[Any, str, DegradationLevel]:
        """Execute function with automatic fallback if needed."""
        
        # Update degradation level based on current performance
        self._update_degradation_level(current_performance)
        
        # Start with primary function
        if (self.current_degradation_level == DegradationLevel.FULL_CAPABILITY and
            current_performance >= self.degradation_thresholds[DegradationLevel.FULL_CAPABILITY]):
            
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    result = await primary_func(input_data, **kwargs)
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: primary_func(input_data, **kwargs)
                    )
                
                self._record_success("primary", current_performance)
                return result, "primary", self.current_degradation_level
                
            except Exception as e:
                logging.warning(f"Primary function failed: {e}")
                self._record_failure("primary", str(e))
        
        # Try fallback strategies
        for strategy in self.strategies:
            if not strategy.is_available():
                continue
            
            try:
                if self._is_strategy_appropriate(strategy):
                    result = await strategy.execute(input_data, **kwargs)
                    self._record_success(strategy.name, current_performance)
                    return result, strategy.name, self.current_degradation_level
                    
            except Exception as e:
                logging.warning(f"Fallback strategy {strategy.name} failed: {e}")
                self._record_failure(strategy.name, str(e))
                continue
        
        # All strategies failed
        self._record_failure("all_strategies", "All fallback strategies failed")
        raise create_model_failure_error("All fallback strategies exhausted")
    
    def _update_degradation_level(self, current_performance: float):
        """Update degradation level based on current performance."""
        previous_level = self.current_degradation_level
        
        for level, threshold in self.degradation_thresholds.items():
            if current_performance >= threshold:
                self.current_degradation_level = level
                break
        
        # Log level changes
        if previous_level != self.current_degradation_level:
            logging.info(f"Degradation level changed from {previous_level.value} to {self.current_degradation_level.value}")
    
    def _is_strategy_appropriate(self, strategy: FallbackStrategy) -> bool:
        """Check if strategy is appropriate for current degradation level."""
        strategy_type = type(strategy).__name__
        
        if self.current_degradation_level == DegradationLevel.FAILSAFE_MODE:
            # Only allow safety-critical strategies
            return strategy_type in ["RuleBasedFallbackStrategy"]
        
        elif self.current_degradation_level == DegradationLevel.EMERGENCY_MODE:
            # Allow basic rule-based and safe default strategies
            return strategy_type in ["RuleBasedFallbackStrategy"]
        
        elif self.current_degradation_level == DegradationLevel.BASIC_FUNCTIONALITY:
            # Allow model and rule-based strategies
            return strategy_type in ["ModelFallbackStrategy", "RuleBasedFallbackStrategy"]
        
        else:
            # Allow all strategies
            return True
    
    def _record_success(self, strategy_name: str, performance: float):
        """Record successful execution."""
        self.fallback_audit.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "success",
            "strategy": strategy_name,
            "performance": performance,
            "degradation_level": self.current_degradation_level.value
        })
    
    def _record_failure(self, strategy_name: str, error: str):
        """Record failed execution."""
        self.fallback_audit.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "failure",
            "strategy": strategy_name,
            "error": error,
            "degradation_level": self.current_degradation_level.value
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get degradation manager status."""
        return {
            "current_degradation_level": self.current_degradation_level.value,
            "available_strategies": [s.get_status() for s in self.strategies if s.is_available()],
            "degradation_thresholds": {k.value: v for k, v in self.degradation_thresholds.items()},
            "performance_history": self.performance_history[-100:],  # Last 100 entries
            "fallback_audit": self.fallback_audit[-100:],  # Last 100 entries
            "strategy_count": len(self.strategies)
        }
    
    def enable_strategy(self, strategy_name: str):
        """Enable a specific strategy."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enabled = True
                break
    
    def disable_strategy(self, strategy_name: str):
        """Disable a specific strategy."""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enabled = False
                break


# Global degradation manager
degradation_manager = DegradationManager()


def with_fallback(
    model_fallback=None,
    rule_fallback=None,
    performance_threshold=0.8
) -> Callable:
    """Decorator for automatic fallback functionality."""
    
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Set up circuit breaker for the function
            circuit_breaker = circuit_breaker_registry.create(
                f"{func.__module__}.{func.__name__}",
                patient_safety_mode=True
            )
            
            try:
                # Try primary function
                return await circuit_breaker(func, *args, **kwargs)
                
            except Exception as e:
                # Handle fallback through degradation manager
                current_performance = 0.5  # Assume degraded performance
                
                return await degradation_manager.execute_with_fallback(
                    func, args[0] if args else {}, current_performance, *args[1:], **kwargs
                )
        
        def sync_wrapper(*args, **kwargs):
            # Sync version would need similar logic
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Pre-configured fallback setups for common medical scenarios
def setup_critical_fallbacks():
    """Setup fallbacks for critical medical operations."""
    # High-priority model fallback
    if hasattr(degradation_manager, 'primary_model') and hasattr(degradation_manager, 'fallback_model'):
        model_fallback = ModelFallbackStrategy(
            degradation_manager.primary_model,
            degradation_manager.fallback_model
        )
        degradation_manager.add_strategy(model_fallback)
    
    # Rule-based fallback for safety
    safety_rules = {
        "critical_assessment": lambda data: {"status": "manual_review_required"},
        "emergency_protocol": lambda data: {"action": "escalate_to_human"}
    }
    
    rule_fallback = RuleBasedFallbackStrategy(safety_rules)
    degradation_manager.add_strategy(rule_fallback)


def setup_routine_fallbacks():
    """Setup fallbacks for routine medical operations."""
    # Basic model fallback
    rule_fallback = RuleBasedFallbackStrategy({
        "routine_assessment": lambda data: {"confidence": "low", "recommendation": "standard_protocol"}
    })
    degradation_manager.add_strategy(rule_fallback)