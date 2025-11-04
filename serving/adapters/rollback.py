"""
Rollback and Fallback Management System

Provides comprehensive rollback mechanisms for production safety,
fallback strategies, and recovery procedures.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

from .registry import AdapterRegistry, AdapterMetadata, AdapterStatus
from .manager import AdapterManager, AdapterInstance
from .hot_swap import SwapOperation, SwapStatus

logger = logging.getLogger(__name__)


class RollbackTrigger(Enum):
    """Rollback trigger conditions."""
    MANUAL = "manual"                    # User-initiated
    HEALTH_CHECK_FAILED = "health_failed"
    PERFORMANCE_DEGRADATION = "performance_degraded"
    ERROR_RATE_HIGH = "error_rate_high"
    VALIDATION_FAILED = "validation_failed"
    SYSTEM_ERROR = "system_error"
    EMERGENCY = "emergency"


class FallbackStrategy(Enum):
    """Fallback strategies."""
    REVERT_TO_PREVIOUS = "revert_to_previous"
    LOAD_STABLE_VERSION = "load_stable_version"
    USE_BASE_MODEL = "use_base_model"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRADUAL_FALLBACK = "gradual_fallback"


@dataclass
class RollbackOperation:
    """Represents a rollback operation."""
    operation_id: str
    trigger: RollbackTrigger
    from_adapter_id: str
    to_adapter_id: str
    reason: str
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Results
    success: bool = False
    error_message: Optional[str] = None
    rollback_duration_seconds: float = 0.0
    
    # State tracking
    steps_completed: List[str] = field(default_factory=list)
    fallback_applied: Optional['FallbackAdapter'] = None
    
    def add_step(self, step: str):
        """Add completed step."""
        self.steps_completed.append(step)
        logger.info(f"Rollback {self.operation_id}: {step}")
    
    def mark_started(self):
        """Mark rollback as started."""
        self.started_at = time.time()
    
    def mark_completed(self, success: bool, error_msg: Optional[str] = None):
        """Mark rollback as completed."""
        self.completed_at = time.time()
        self.success = success
        self.error_message = error_msg
        self.rollback_duration_seconds = self.completed_at - self.started_at
        
        status = "completed successfully" if success else f"failed: {error_msg}"
        self.add_step(f"Rollback {status}")
    
    def get_duration_seconds(self) -> float:
        """Get rollback duration."""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or time.time()
        return end_time - self.started_at


@dataclass
class FallbackAdapter:
    """Represents a fallback adapter configuration."""
    adapter_id: str
    version_id: str
    strategy: FallbackStrategy
    priority: int  # 1 = highest priority
    health_score: float = 1.0
    validation_status: str = "unknown"
    last_tested: Optional[float] = None
    
    # Medical compliance
    medical_compliant: bool = False
    compliance_level: str = "none"
    
    # Performance characteristics
    avg_load_time: float = 0.0
    memory_usage_mb: float = 0.0
    success_rate: float = 0.0
    
    def is_available(self) -> bool:
        """Check if fallback adapter is available."""
        return self.validation_status == "validated" and self.health_score >= 0.7
    
    def get_readiness_score(self) -> float:
        """Get overall readiness score."""
        score = 0.0
        
        # Health score
        score += self.health_score * 0.3
        
        # Validation status
        if self.validation_status == "validated":
            score += 0.25
        elif self.validation_status == "tested":
            score += 0.15
        
        # Success rate
        score += self.success_rate * 0.2
        
        # Medical compliance
        if self.medical_compliant:
            score += 0.15
        else:
            score += 0.1
        
        # Priority (lower is better, so invert)
        priority_score = max(0, 1 - (self.priority - 1) * 0.1)
        score += priority_score * 0.1
        
        return min(score, 1.0)


class HealthMonitor:
    """Monitors system health for rollback decisions."""
    
    def __init__(self, manager: AdapterManager):
        self.manager = manager
        self.health_history: Dict[str, List[Dict[str, Any]]] = {}
        self.thresholds = {
            "error_rate_threshold": 0.05,        # 5% error rate
            "latency_threshold_ms": 1000,        # 1 second latency
            "memory_usage_threshold": 0.9,       # 90% memory usage
            "health_check_interval": 30.0        # 30 seconds
        }
        
    async def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            # Get manager health
            manager_health = await self.manager.health_check()
            
            # Check adapter health
            adapter_health = {}
            active_adapters = self.manager.get_active_adapters()
            
            for adapter_id, instance in active_adapters.items():
                adapter_health[adapter_id] = await self._check_adapter_health(instance)
            
            # Check overall system metrics
            system_metrics = await self._get_system_metrics()
            
            # Determine overall health status
            is_healthy = (
                manager_health.get("status") == "healthy" and
                all(h.get("healthy", False) for h in adapter_health.values()) and
                system_metrics.get("error_rate", 0) < self.thresholds["error_rate_threshold"]
            )
            
            health_result = {
                "timestamp": time.time(),
                "healthy": is_healthy,
                "manager_health": manager_health,
                "adapter_health": adapter_health,
                "system_metrics": system_metrics,
                "triggers": self._check_rollback_triggers(manager_health, adapter_health, system_metrics)
            }
            
            # Store in history
            self._store_health_record(health_result)
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "timestamp": time.time(),
                "healthy": False,
                "error": str(e)
            }
    
    async def _check_adapter_health(self, instance: AdapterInstance) -> Dict[str, Any]:
        """Check health of specific adapter instance."""
        try:
            # Basic state checks
            state_healthy = instance.state.value in ["loaded", "active"]
            
            # Error check
            no_errors = instance.error_message is None
            
            # Memory usage check
            memory_healthy = instance.memory_usage_mb < 8192  # 8GB threshold
            
            # Recent activity check
            recent_access = (time.time() - instance.last_accessed) < 3600  # 1 hour
            
            is_healthy = state_healthy and no_errors and memory_healthy
            
            return {
                "healthy": is_healthy,
                "state": instance.state.value,
                "has_errors": instance.error_message is not None,
                "memory_usage_mb": instance.memory_usage_mb,
                "recent_access": recent_access,
                "load_count": instance.load_count
            }
            
        except Exception as e:
            logger.error(f"Adapter health check failed for {instance.adapter_id}: {e}")
            return {
                "healthy": False,
                "error": str(e)
            }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            # Get manager stats
            stats = self.manager.get_manager_stats()
            
            # Extract key metrics
            memory_usage = stats.get("memory_usage", {})
            
            return {
                "error_rate": self._calculate_error_rate(),
                "avg_latency_ms": stats.get("avg_inference_latency_ms", 0),
                "memory_usage_percent": memory_usage.get("percent", 0),
                "active_adapters": stats.get("active_adapters", 0),
                "total_operations": stats.get("completed_operations", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        try:
            # This would typically query metrics database
            # For now, return a placeholder
            return 0.02  # 2% error rate
            
        except Exception as e:
            logger.error(f"Failed to calculate error rate: {e}")
            return 0.0
    
    def _check_rollback_triggers(self, 
                               manager_health: Dict[str, Any],
                               adapter_health: Dict[str, Any],
                               system_metrics: Dict[str, Any]) -> List[RollbackTrigger]:
        """Check if any rollback triggers are activated."""
        triggers = []
        
        # Manager health check
        if manager_health.get("status") != "healthy":
            triggers.append(RollbackTrigger.HEALTH_CHECK_FAILED)
        
        # Adapter health checks
        for adapter_id, health in adapter_health.items():
            if not health.get("healthy", False):
                triggers.append(RollbackTrigger.HEALTH_CHECK_FAILED)
                break
        
        # Performance degradation
        latency = system_metrics.get("avg_latency_ms", 0)
        if latency > self.thresholds["latency_threshold_ms"]:
            triggers.append(RollbackTrigger.PERFORMANCE_DEGRADATION)
        
        # Error rate check
        error_rate = system_metrics.get("error_rate", 0)
        if error_rate > self.thresholds["error_rate_threshold"]:
            triggers.append(RollbackTrigger.ERROR_RATE_HIGH)
        
        return list(set(triggers))  # Remove duplicates
    
    def _store_health_record(self, health_result: Dict[str, Any]):
        """Store health record in history."""
        timestamp = health_result["timestamp"]
        
        if "health_history" not in self.__dict__:
            self.health_history = {}
        
        # Keep only recent records (last 100)
        for key in list(self.health_history.keys()):
            history = self.health_history[key]
            if len(history) >= 100:
                self.health_history[key] = history[-50:]
        
        logger.debug(f"Health check: {'HEALTHY' if health_result['healthy'] else 'UNHEALTHY'}")


class FallbackManager:
    """Manages fallback adapters and strategies."""
    
    def __init__(self, registry: AdapterRegistry, manager: AdapterManager):
        self.registry = registry
        self.manager = manager
        self.fallback_adapters: List[FallbackAdapter] = []
        self.fallback_history: List[RollbackOperation] = []
        
    async def register_fallback_adapter(self, fallback: FallbackAdapter):
        """Register a fallback adapter."""
        try:
            # Validate adapter exists and is healthy
            metadata = self.registry.get_adapter(fallback.adapter_id)
            if not metadata:
                raise ValueError(f"Adapter {fallback.adapter_id} not found in registry")
            
            # Update fallback properties
            fallback.last_tested = time.time()
            fallback.health_score = await self._calculate_health_score(fallback)
            fallback.validation_status = metadata.validation_status
            fallback.medical_compliant = self._check_medical_compliance(metadata)
            
            # Add to fallback list
            self.fallback_adapters.append(fallback)
            
            # Sort by priority
            self.fallback_adapters.sort(key=lambda x: x.priority)
            
            logger.info(f"Registered fallback adapter: {fallback.adapter_id} "
                       f"(priority: {fallback.priority}, score: {fallback.get_readiness_score():.2f})")
            
        except Exception as e:
            logger.error(f"Failed to register fallback adapter {fallback.adapter_id}: {e}")
            raise
    
    async def get_best_fallback(self, exclude_adapter_id: Optional[str] = None) -> Optional[FallbackAdapter]:
        """Get the best available fallback adapter."""
        available_fallbacks = [
            f for f in self.fallback_adapters 
            if f.is_available() and f.adapter_id != exclude_adapter_id
        ]
        
        if not available_fallbacks:
            return None
        
        # Sort by readiness score
        available_fallbacks.sort(key=lambda x: x.get_readiness_score(), reverse=True)
        
        best_fallback = available_fallbacks[0]
        logger.info(f"Selected fallback adapter: {best_fallback.adapter_id} "
                   f"(score: {best_fallback.get_readiness_score():.2f})")
        
        return best_fallback
    
    async def test_fallback_readiness(self, fallback: FallbackAdapter) -> Dict[str, Any]:
        """Test fallback adapter readiness."""
        try:
            # Test loading the adapter
            instance = await self.manager.load_adapter(
                fallback.adapter_id,
                fallback.version_id,
                validate=True
            )
            
            # Check if it loads successfully
            success = instance is not None and instance.state.value in ["loaded", "active"]
            
            # Update health score
            if success:
                fallback.health_score = min(fallback.health_score + 0.1, 1.0)
                fallback.last_tested = time.time()
            else:
                fallback.health_score = max(fallback.health_score - 0.2, 0.0)
            
            return {
                "success": success,
                "health_score": fallback.health_score,
                "ready": fallback.is_available()
            }
            
        except Exception as e:
            logger.error(f"Fallback test failed for {fallback.adapter_id}: {e}")
            fallback.health_score = max(fallback.health_score - 0.3, 0.0)
            return {
                "success": False,
                "error": str(e),
                "health_score": fallback.health_score,
                "ready": False
            }
    
    async def _calculate_health_score(self, fallback: FallbackAdapter) -> float:
        """Calculate health score for fallback adapter."""
        try:
            # Get registry statistics
            metadata = self.registry.get_adapter(fallback.adapter_id)
            if not metadata:
                return 0.0
            
            # Base score from success rate
            base_score = metadata.get_success_rate()
            
            # Adjust for medical compliance
            if metadata.adapter_type.value in ["medical_lora", "clinical_lora"]:
                if metadata.safety_score >= 0.8 and metadata.validation_status == "validated":
                    base_score += 0.2
            
            # Adjust for performance
            if metadata.avg_load_time < 5.0:  # Good load time
                base_score += 0.1
            
            if metadata.inference_latency_ms < 100:  # Good latency
                base_score += 0.1
            
            return min(base_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate health score for {fallback.adapter_id}: {e}")
            return 0.0
    
    def _check_medical_compliance(self, metadata: AdapterMetadata) -> bool:
        """Check if adapter meets medical compliance requirements."""
        try:
            # Check for required compliance flags
            required_flags = ["validated"]
            
            has_required = all(flag in metadata.compliance_flags for flag in required_flags)
            
            # Check safety score
            safety_ok = metadata.safety_score >= 0.7
            
            # Check test coverage
            test_coverage_ok = metadata.test_coverage >= 0.8
            
            return has_required and safety_ok and test_coverage_ok
            
        except Exception as e:
            logger.error(f"Medical compliance check failed for {metadata.adapter_id}: {e}")
            return False
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback system statistics."""
        total_fallbacks = len(self.fallback_adapters)
        available_fallbacks = sum(1 for f in self.fallback_adapters if f.is_available())
        
        # Calculate average readiness score
        readiness_scores = [f.get_readiness_score() for f in self.fallback_adapters]
        avg_readiness = sum(readiness_scores) / len(readiness_scores) if readiness_scores else 0.0
        
        # Calculate success rate
        total_tests = len(self.fallback_history)
        successful_tests = sum(1 for op in self.fallback_history if op.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0.0
        
        return {
            "total_registered_fallbacks": total_fallbacks,
            "available_fallbacks": available_fallbacks,
            "avg_readiness_score": avg_readiness,
            "recent_test_success_rate": success_rate,
            "fallback_history_count": len(self.fallback_history)
        }


class RollbackManager:
    """
    Production-grade rollback manager with fallback capabilities.
    
    Features:
    - Automatic rollback on health failures
    - Multiple fallback strategies
    - Rollback operation tracking
    - Recovery procedures
    - Emergency rollback procedures
    """
    
    def __init__(self,
                 manager: AdapterManager,
                 registry: AdapterRegistry,
                 hot_swap_manager=None):
        
        self.manager = manager
        self.registry = registry
        self.hot_swap_manager = hot_swap_manager
        
        # Core components
        self.health_monitor = HealthMonitor(manager)
        self.fallback_manager = FallbackManager(registry, manager)
        
        # State management
        self.active_rollbacks: Dict[str, RollbackOperation] = {}
        self.rollback_history: List[RollbackOperation] = []
        self._lock = asyncio.Lock()
        
        # Configuration
        self.auto_rollback_enabled = True
        self.monitoring_interval = 30.0
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Rollback strategies
        self.strategy_handlers = {
            FallbackStrategy.REVERT_TO_PREVIOUS: self._revert_to_previous,
            FallbackStrategy.LOAD_STABLE_VERSION: self._load_stable_version,
            FallbackStrategy.USE_BASE_MODEL: self._use_base_model,
            FallbackStrategy.CIRCUIT_BREAKER: self._circuit_breaker,
            FallbackStrategy.GRADUAL_FALLBACK: self._gradual_fallback
        }
        
        logger.info("RollbackManager initialized")
    
    async def start_monitoring(self):
        """Start automatic rollback monitoring."""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Rollback monitoring started")
    
    async def stop_monitoring(self):
        """Stop automatic rollback monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Rollback monitoring stopped")
    
    async def execute_manual_rollback(self, 
                                    from_adapter_id: str,
                                    to_adapter_id: Optional[str] = None,
                                    reason: str = "Manual rollback") -> str:
        """Execute manual rollback operation."""
        
        operation_id = f"manual_rollback_{from_adapter_id}_{int(time.time())}"
        
        # Determine target adapter
        if to_adapter_id is None:
            to_adapter_id = await self._get_previous_version(from_adapter_id)
            if not to_adapter_id:
                raise ValueError("No previous version found for rollback")
        
        operation = RollbackOperation(
            operation_id=operation_id,
            trigger=RollbackTrigger.MANUAL,
            from_adapter_id=from_adapter_id,
            to_adapter_id=to_adapter_id,
            reason=reason
        )
        
        async with self._lock:
            self.active_rollbacks[operation_id] = operation
        
        # Execute rollback
        asyncio.create_task(self._execute_rollback(operation))
        
        logger.info(f"Started manual rollback: {from_adapter_id} -> {to_adapter_id}")
        return operation_id
    
    async def execute_emergency_rollback(self, adapter_id: str, reason: str) -> str:
        """Execute emergency rollback with highest priority."""
        
        operation_id = f"emergency_rollback_{adapter_id}_{int(time.time())}"
        
        operation = RollbackOperation(
            operation_id=operation_id,
            trigger=RollbackTrigger.EMERGENCY,
            from_adapter_id=adapter_id,
            to_adapter_id="base_model",  # Default to base model
            reason=reason
        )
        
        async with self._lock:
            self.active_rollbacks[operation_id] = operation
        
        # Execute emergency rollback
        asyncio.create_task(self._execute_emergency_rollback(operation))
        
        logger.warning(f"Emergency rollback started for adapter: {adapter_id}")
        return operation_id
    
    async def register_fallback_adapter(self, 
                                      adapter_id: str,
                                      version_id: str,
                                      strategy: FallbackStrategy,
                                      priority: int = 1):
        """Register a fallback adapter."""
        fallback = FallbackAdapter(
            adapter_id=adapter_id,
            version_id=version_id,
            strategy=strategy,
            priority=priority
        )
        
        await self.fallback_manager.register_fallback_adapter(fallback)
    
    async def get_rollback_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of rollback operation."""
        operation = self.active_rollbacks.get(operation_id)
        if operation:
            return {
                "operation_id": operation.operation_id,
                "trigger": operation.trigger.value,
                "from_adapter": operation.from_adapter_id,
                "to_adapter": operation.to_adapter_id,
                "status": "active",
                "reason": operation.reason,
                "duration_seconds": operation.get_duration_seconds(),
                "steps_completed": operation.steps_completed,
                "success": operation.success,
                "error_message": operation.error_message
            }
        
        # Check history
        for op in self.rollback_history:
            if op.operation_id == operation_id:
                return {
                    "operation_id": op.operation_id,
                    "trigger": op.trigger.value,
                    "from_adapter": op.from_adapter_id,
                    "to_adapter": op.to_adapter_id,
                    "status": "completed",
                    "reason": op.reason,
                    "duration_seconds": op.get_duration_seconds(),
                    "steps_completed": op.steps_completed,
                    "success": op.success,
                    "error_message": op.error_message
                }
        
        return None
    
    async def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback system statistics."""
        total_rollbacks = len(self.rollback_history)
        successful_rollbacks = sum(1 for op in self.rollback_history if op.success)
        success_rate = successful_rollbacks / total_rollbacks if total_rollbacks > 0 else 0.0
        
        # Calculate average rollback time
        completed_rollbacks = [op for op in self.rollback_history if op.completed_at]
        avg_time = sum(op.get_duration_seconds() for op in completed_rollbacks) / len(completed_rollbacks) if completed_rollbacks else 0.0
        
        return {
            "total_rollbacks": total_rollbacks,
            "successful_rollbacks": successful_rollbacks,
            "success_rate": success_rate,
            "avg_rollback_time_seconds": avg_time,
            "active_rollbacks": len(self.active_rollbacks),
            "fallback_statistics": self.fallback_manager.get_fallback_statistics(),
            "monitoring_active": self.monitoring_task is not None and not self.monitoring_task.done()
        }
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop for automatic rollbacks."""
        try:
            while True:
                if not self.auto_rollback_enabled:
                    await asyncio.sleep(self.monitoring_interval)
                    continue
                
                try:
                    # Check system health
                    health_result = await self.health_monitor.check_system_health()
                    
                    # Check for rollback triggers
                    if health_result.get("triggers"):
                        for trigger in health_result["triggers"]:
                            await self._handle_rollback_trigger(trigger, health_result)
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            logger.info("Rollback monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")
    
    async def _handle_rollback_trigger(self, trigger: RollbackTrigger, health_result: Dict[str, Any]):
        """Handle automatic rollback trigger."""
        try:
            # Find adapters that need rollback
            problematic_adapters = []
            
            for adapter_id, health in health_result.get("adapter_health", {}).items():
                if not health.get("healthy", False):
                    problematic_adapters.append(adapter_id)
            
            # Execute rollback for each problematic adapter
            for adapter_id in problematic_adapters:
                await self.execute_emergency_rollback(
                    adapter_id,
                    f"Auto rollback due to {trigger.value}"
                )
                
        except Exception as e:
            logger.error(f"Failed to handle rollback trigger {trigger.value}: {e}")
    
    async def _execute_rollback(self, operation: RollbackOperation):
        """Execute rollback operation."""
        try:
            operation.mark_started()
            
            # Step 1: Select fallback strategy
            fallback = await self.fallback_manager.get_best_fallback(operation.from_adapter_id)
            if not fallback:
                operation.mark_completed(False, "No fallback adapter available")
                return
            
            operation.add_step(f"Selected fallback: {fallback.adapter_id} (strategy: {fallback.strategy.value})")
            
            # Step 2: Execute strategy
            handler = self.strategy_handlers.get(fallback.strategy)
            if not handler:
                operation.mark_completed(False, f"Unknown fallback strategy: {fallback.strategy}")
                return
            
            success = await handler(operation, fallback)
            
            if success:
                operation.fallback_applied = fallback
                operation.mark_completed(True)
                logger.info(f"Rollback completed successfully: {operation.operation_id}")
            else:
                operation.mark_completed(False, "Rollback strategy execution failed")
                
        except Exception as e:
            logger.error(f"Rollback execution failed: {e}")
            operation.mark_completed(False, f"Rollback error: {str(e)}")
        
        finally:
            # Move to history
            async with self._lock:
                if operation.operation_id in self.active_rollbacks:
                    del self.active_rollbacks[operation.operation_id]
                self.rollback_history.append(operation)
                
                # Keep only recent history
                if len(self.rollback_history) > 100:
                    self.rollback_history = self.rollback_history[-50:]
    
    async def _execute_emergency_rollback(self, operation: RollbackOperation):
        """Execute emergency rollback (simplified for critical situations)."""
        try:
            operation.mark_started()
            operation.add_step("Emergency rollback: using base model")
            
            # Emergency rollback to base model
            success = self.manager.set_active_adapter("base_model")
            
            if success:
                operation.mark_completed(True)
            else:
                operation.mark_completed(False, "Failed to revert to base model")
                
        except Exception as e:
            operation.mark_completed(False, f"Emergency rollback failed: {str(e)}")
    
    # Fallback strategy handlers
    
    async def _revert_to_previous(self, operation: RollbackOperation, fallback: FallbackAdapter) -> bool:
        """Revert to previous adapter version."""
        try:
            operation.add_step("Reverting to previous version")
            
            success = self.manager.set_active_adapter(
                fallback.adapter_id,
                fallback.version_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Revert to previous failed: {e}")
            return False
    
    async def _load_stable_version(self, operation: RollbackOperation, fallback: FallbackAdapter) -> bool:
        """Load stable fallback version."""
        try:
            operation.add_step("Loading stable version")
            
            # Load the fallback adapter
            instance = await self.manager.load_adapter(
                fallback.adapter_id,
                fallback.version_id,
                validate=True
            )
            
            if instance:
                success = self.manager.set_active_adapter(
                    fallback.adapter_id,
                    fallback.version_id
                )
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Load stable version failed: {e}")
            return False
    
    async def _use_base_model(self, operation: RollbackOperation, fallback: FallbackAdapter) -> bool:
        """Use base model as fallback."""
        try:
            operation.add_step("Switching to base model")
            
            # This would involve disabling adapters and using base model directly
            # Implementation depends on how the serving system handles base model fallback
            
            return True  # Simplified for now
            
        except Exception as e:
            logger.error(f"Use base model failed: {e}")
            return False
    
    async def _circuit_breaker(self, operation: RollbackOperation, fallback: FallbackAdapter) -> bool:
        """Implement circuit breaker pattern."""
        try:
            operation.add_step("Activating circuit breaker")
            
            # Circuit breaker would temporarily disable the problematic adapter
            # and route to fallback
            
            # For now, just switch to fallback
            success = self.manager.set_active_adapter(
                fallback.adapter_id,
                fallback.version_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Circuit breaker failed: {e}")
            return False
    
    async def _gradual_fallback(self, operation: RollbackOperation, fallback: FallbackAdapter) -> bool:
        """Gradual fallback with traffic shifting."""
        try:
            operation.add_step("Starting gradual fallback")
            
            # Gradual fallback would involve shifting traffic percentages
            # For now, simplified to immediate switch
            
            success = self.manager.set_active_adapter(
                fallback.adapter_id,
                fallback.version_id
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Gradual fallback failed: {e}")
            return False
    
    async def _get_previous_version(self, adapter_id: str) -> Optional[str]:
        """Get previous version of adapter for rollback."""
        try:
            metadata = self.registry.get_adapter(adapter_id)
            if not metadata or len(metadata.versions) < 2:
                return None
            
            # Sort versions by creation time (newest first)
            sorted_versions = sorted(metadata.versions, key=lambda v: v.created_at, reverse=True)
            
            # Return second newest version
            return sorted_versions[1].version_id if len(sorted_versions) > 1 else None
            
        except Exception as e:
            logger.error(f"Failed to get previous version for {adapter_id}: {e}")
            return None


# Utility functions
async def create_rollback_manager(manager: AdapterManager,
                                registry: AdapterRegistry,
                                hot_swap_manager=None) -> RollbackManager:
    """Factory function to create rollback manager."""
    return RollbackManager(manager, registry, hot_swap_manager)


if __name__ == "__main__":
    # Example usage
    async def main():
        from .registry import AdapterRegistry, AdapterType
        from .manager import create_adapter_manager
        from .hot_swap import SwapOperation, SwapStatus
        
        # Create components
        registry = AdapterRegistry("./test_rollback_registry.db")
        # manager = await create_adapter_manager("microsoft/DialoGPT-medium", registry)
        
        # Create rollback manager
        # rollback_manager = await create_rollback_manager(manager, registry)
        
        # Start monitoring
        # await rollback_manager.start_monitoring()
        
        # Register fallback
        # await rollback_manager.register_fallback_adapter(
        #     adapter_id="stable_adapter",
        #     version_id="v1.0.0",
        #     strategy=FallbackStrategy.REVERT_TO_PREVIOUS,
        #     priority=1
        # )
        
        # Execute manual rollback
        # operation_id = await rollback_manager.execute_manual_rollback(
        #     from_adapter_id="problematic_adapter",
        #     reason="Performance degradation detected"
        # )
        
        print("Rollback manager example")
    
    # asyncio.run(main())