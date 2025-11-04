"""
Adapter Lifecycle Manager

Main orchestrator for the complete adapter management system,
integrating registry, manager, validator, cache, hot-swap, rollback, and metrics.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator, Callable

from .registry import AdapterRegistry, AdapterMetadata, AdapterType
from .manager import AdapterManager, AdapterInstance, AdapterState
from .validator import AdapterValidator, ValidationResult
from .cache import AdapterCache, MedicalAdapterCache, CacheStrategy, MemoryOptimizationLevel
from .hot_swap import HotSwapManager, SwapOperation, SwapStrategy
from .rollback import RollbackManager, RollbackOperation, RollbackTrigger
from .metrics import AdapterMetrics, OperationType, MetricType

logger = logging.getLogger(__name__)


class LifecycleState(Enum):
    """Adapter lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class SystemConfiguration:
    """System-wide configuration."""
    # Base model
    base_model_id: str
    base_model_path: Optional[str] = None
    
    # Registry
    registry_path: str = "./adapter_registry.db"
    enable_medical_compliance: bool = True
    
    # Cache configuration
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    memory_optimization: MemoryOptimizationLevel = MemoryOptimizationLevel.BALANCED
    max_cache_size: int = 10
    max_memory_mb: int = 16384
    enable_gpu_optimization: bool = True
    
    # Performance
    max_concurrent_loads: int = 2
    max_concurrent_swaps: int = 3
    
    # Monitoring
    enable_metrics: bool = True
    metrics_storage_path: str = "./adapter_metrics.db"
    monitoring_interval: int = 60
    
    # Safety
    auto_rollback_enabled: bool = True
    health_check_interval: int = 30
    emergency_swap_timeout: float = 60.0
    
    # Medical AI specific
    require_medical_validation: bool = True
    hipaa_compliance_required: bool = True
    clinical_trial_mode: bool = False


@dataclass
class SystemStatus:
    """Overall system status."""
    state: LifecycleState
    initialized_at: float
    last_health_check: Optional[float] = None
    active_adapters: int = 0
    total_adapters: int = 0
    active_operations: int = 0
    system_health: str = "unknown"
    
    # Statistics
    total_loads: int = 0
    successful_loads: int = 0
    total_swaps: int = 0
    successful_swaps: int = 0
    total_rollbacks: int = 0
    successful_rollbacks: int = 0
    
    # Performance
    avg_load_time_ms: float = 0.0
    avg_swap_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Medical compliance
    compliant_adapters: int = 0
    total_compliance_checks: int = 0


class AdapterLifecycleManager:
    """
    Complete adapter lifecycle management system.
    
    This is the main orchestrator that integrates all components:
    - Registry for adapter management
    - Manager for loading and lifecycle operations
    - Validator for compatibility checking
    - Cache for memory optimization
    - Hot-swap for zero-downtime updates
    - Rollback for production safety
    - Metrics for monitoring and analytics
    """
    
    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.state = LifecycleState.INITIALIZING
        self.status = SystemStatus(
            state=LifecycleState.INITIALIZING,
            initialized_at=time.time()
        )
        
        # Core components (initialized in startup)
        self.registry: Optional[AdapterRegistry] = None
        self.manager: Optional[AdapterManager] = None
        self.validator: Optional[AdapterValidator] = None
        self.cache: Optional[MedicalAdapterCache] = None
        self.hot_swap: Optional[HotSwapManager] = None
        self.rollback: Optional[RollbackManager] = None
        self.metrics: Optional[AdapterMetrics] = None
        
        # State management
        self._startup_complete = False
        self._shutdown_requested = False
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "adapter_loaded": [],
            "adapter_unloaded": [],
            "swap_started": [],
            "swap_completed": [],
            "rollback_executed": [],
            "health_check_failed": [],
            "system_error": []
        }
        
        logger.info("AdapterLifecycleManager initialized with configuration")
    
    async def startup(self):
        """Initialize all system components."""
        try:
            logger.info("Starting Adapter Lifecycle Manager...")
            
            # Initialize registry
            self.registry = AdapterRegistry(self.config.registry_path)
            logger.info("Registry initialized")
            
            # Initialize validator
            self.validator = AdapterValidator()
            logger.info("Validator initialized")
            
            # Initialize metrics system
            if self.config.enable_metrics:
                self.metrics = AdapterMetrics(self.config.metrics_storage_path)
                await self.metrics.start_monitoring(self.config.monitoring_interval)
                logger.info("Metrics system initialized")
            
            # Initialize adapter manager
            self.manager = AdapterManager(
                base_model_id=self.config.base_model_id,
                registry=self.registry,
                max_memory_mb=self.config.max_memory_mb,
                max_concurrent_loads=self.config.max_concurrent_loads,
                enable_gpu_optimization=self.config.enable_gpu_optimization
            )
            await self.manager.initialize()
            logger.info("Adapter manager initialized")
            
            # Initialize cache
            self.cache = MedicalAdapterCache(
                max_size=self.config.max_cache_size,
                strategy=self.config.cache_strategy,
                optimization_level=self.config.memory_optimization,
                max_memory_mb=self.config.max_memory_mb,
                enable_gpu_optimization=self.config.enable_gpu_optimization
            )
            logger.info("Cache system initialized")
            
            # Initialize hot-swap manager
            self.hot_swap = HotSwapManager(
                manager=self.manager,
                registry=self.registry,
                validator=self.validator
            )
            logger.info("Hot-swap manager initialized")
            
            # Initialize rollback manager
            self.rollback = RollbackManager(
                manager=self.manager,
                registry=self.registry,
                hot_swap_manager=self.hot_swap
            )
            
            if self.config.auto_rollback_enabled:
                await self.rollback.start_monitoring()
            logger.info("Rollback manager initialized")
            
            # Update system status
            self.state = LifecycleState.READY
            self.status.state = LifecycleState.READY
            self._startup_complete = True
            
            # Perform initial health check
            await self._perform_initial_health_check()
            
            logger.info("Adapter Lifecycle Manager startup completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Adapter Lifecycle Manager: {e}")
            self.state = LifecycleState.ERROR
            self.status.state = LifecycleState.ERROR
            raise
    
    async def shutdown(self):
        """Gracefully shutdown all system components."""
        try:
            logger.info("Shutting down Adapter Lifecycle Manager...")
            
            self._shutdown_requested = True
            self.state = LifecycleState.SHUTTING_DOWN
            self.status.state = LifecycleState.SHUTTING_DOWN
            
            # Stop rollback monitoring
            if self.rollback:
                await self.rollback.stop_monitoring()
            
            # Stop metrics monitoring
            if self.metrics:
                await self.metrics.stop_monitoring()
            
            # Cleanup adapter manager
            if self.manager:
                await self.manager.cleanup()
            
            # Clear cache
            if self.cache:
                self.cache.clear()
            
            logger.info("Adapter Lifecycle Manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    # Core lifecycle operations
    
    async def load_adapter(self,
                          adapter_id: str,
                          version_id: Optional[str] = None,
                          validate: bool = True,
                          cache: bool = True) -> AdapterInstance:
        """Load adapter with full lifecycle management."""
        operation_id = self.metrics.start_operation_tracking(
            OperationType.LOAD, adapter_id
        ) if self.metrics else f"load_{adapter_id}_{int(time.time())}"
        
        try:
            logger.info(f"Loading adapter: {adapter_id} (version: {version_id or 'latest'})")
            
            # Validate adapter if requested
            if validate and self.validator:
                metadata = self.registry.get_adapter(adapter_id)
                if metadata:
                    version = metadata.get_active_version()
                    if version:
                        validation_result = self.validator.validate_adapter(
                            adapter_path=version.model_path,
                            base_model_id=self.config.base_model_id,
                            adapter_id=adapter_id
                        )
                        
                        if not validation_result.is_compatible:
                            raise ValueError(f"Adapter validation failed: {validation_result.issues}")
            
            # Load adapter using manager
            instance = await self.manager.load_adapter(
                adapter_id, version_id, validate=False  # Already validated
            )
            
            # Cache adapter if requested and cache is available
            if cache and self.cache and instance:
                cache_key = f"{adapter_id}:{instance.version_id}"
                metadata_dict = {
                    "adapter_id": adapter_id,
                    "version_id": instance.version_id,
                    "file_size": instance.metadata.file_size if instance.metadata else 0,
                    "compliance_flags": instance.metadata.compliance_flags if instance.metadata else []
                }
                
                self.cache.put(
                    cache_key, adapter_id, instance.version_id,
                    instance.adapter_model, metadata_dict
                )
            
            # Record metrics
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=True)
                self.metrics.record_metric(
                    "adapter_load_success", 1.0, adapter_id, MetricType.COUNTER
                )
            
            # Emit event
            await self._emit_event("adapter_loaded", {
                "adapter_id": adapter_id,
                "version_id": instance.version_id if instance else version_id,
                "operation_id": operation_id
            })
            
            # Update status
            self.status.total_loads += 1
            self.status.successful_loads += 1
            
            logger.info(f"Successfully loaded adapter: {adapter_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_id}: {e}")
            
            # Record metrics
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=False, error_message=str(e))
                self.metrics.record_metric(
                    "adapter_load_failure", 1.0, adapter_id, MetricType.COUNTER
                )
            
            # Update status
            self.status.total_loads += 1
            
            # Emit event
            await self._emit_event("system_error", {
                "operation": "load_adapter",
                "adapter_id": adapter_id,
                "error": str(e)
            })
            
            raise
    
    async def unload_adapter(self, adapter_id: str, 
                           version_id: Optional[str] = None,
                           force: bool = False) -> bool:
        """Unload adapter and free resources."""
        operation_id = self.metrics.start_operation_tracking(
            OperationType.UNLOAD, adapter_id
        ) if self.metrics else f"unload_{adapter_id}_{int(time.time())}"
        
        try:
            logger.info(f"Unloading adapter: {adapter_id}")
            
            # Unload using manager
            success = await self.manager.unload_adapter(adapter_id, version_id, force)
            
            # Clear from cache if present
            if success and self.cache:
                cache_key = f"{adapter_id}:{version_id or 'latest'}"
                self.cache.remove(cache_key)
            
            # Record metrics
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=success)
                self.metrics.record_metric(
                    "adapter_unload_success" if success else "adapter_unload_failure",
                    1.0, adapter_id, MetricType.COUNTER
                )
            
            # Emit event
            if success:
                await self._emit_event("adapter_unloaded", {
                    "adapter_id": adapter_id,
                    "version_id": version_id,
                    "operation_id": operation_id
                })
            
            logger.info(f"Unloaded adapter: {adapter_id} (success: {success})")
            return success
            
        except Exception as e:
            logger.error(f"Failed to unload adapter {adapter_id}: {e}")
            
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=False, error_message=str(e))
            
            return False
    
    async def hot_swap_adapters(self,
                               from_adapter_id: str,
                               from_version_id: Optional[str],
                               to_adapter_id: str,
                               to_version_id: Optional[str],
                               strategy: SwapStrategy = SwapStrategy.BLUE_GREEN,
                               timeout: Optional[float] = None) -> str:
        """Perform hot-swap between adapters."""
        if not self.hot_swap:
            raise RuntimeError("Hot-swap manager not initialized")
        
        operation_id = self.metrics.start_operation_tracking(
            OperationType.HOT_SWAP, to_adapter_id
        ) if self.metrics else f"hot_swap_{to_adapter_id}_{int(time.time())}"
        
        try:
            logger.info(f"Starting hot-swap: {from_adapter_id} -> {to_adapter_id}")
            
            # Start hot-swap operation
            swap_id = await self.hot_swap.start_swap(
                from_adapter_id, from_version_id,
                to_adapter_id, to_version_id,
                strategy, timeout
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_metric(
                    "hot_swap_started", 1.0, to_adapter_id, MetricType.COUNTER,
                    tags={"strategy": strategy.value}
                )
            
            # Emit event
            await self._emit_event("swap_started", {
                "swap_id": swap_id,
                "from_adapter": from_adapter_id,
                "to_adapter": to_adapter_id,
                "strategy": strategy.value,
                "operation_id": operation_id
            })
            
            self.status.total_swaps += 1
            logger.info(f"Hot-swap started: {swap_id}")
            return swap_id
            
        except Exception as e:
            logger.error(f"Failed to start hot-swap: {e}")
            
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=False, error_message=str(e))
            
            raise
    
    async def rollback_adapter(self, adapter_id: str,
                              reason: str = "Manual rollback") -> str:
        """Rollback adapter to previous version."""
        if not self.rollback:
            raise RuntimeError("Rollback manager not initialized")
        
        operation_id = self.metrics.start_operation_tracking(
            OperationType.ROLLBACK, adapter_id
        ) if self.metrics else f"rollback_{adapter_id}_{int(time.time())}"
        
        try:
            logger.warning(f"Rolling back adapter: {adapter_id} - {reason}")
            
            # Execute rollback
            rollback_id = await self.rollback.execute_manual_rollback(
                adapter_id, reason=reason
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_metric(
                    "rollback_executed", 1.0, adapter_id, MetricType.COUNTER,
                    tags={"trigger": "manual"}
                )
            
            # Emit event
            await self._emit_event("rollback_executed", {
                "rollback_id": rollback_id,
                "adapter_id": adapter_id,
                "reason": reason,
                "operation_id": operation_id
            })
            
            self.status.total_rollbacks += 1
            logger.info(f"Rollback initiated: {rollback_id}")
            return rollback_id
            
        except Exception as e:
            logger.error(f"Failed to rollback adapter {adapter_id}: {e}")
            
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=False, error_message=str(e))
            
            raise
    
    async def emergency_rollback(self, adapter_id: str, reason: str) -> str:
        """Execute emergency rollback with highest priority."""
        if not self.rollback:
            raise RuntimeError("Rollback manager not initialized")
        
        operation_id = self.metrics.start_operation_tracking(
            OperationType.ROLLBACK, adapter_id
        ) if self.metrics else f"emergency_rollback_{adapter_id}_{int(time.time())}"
        
        try:
            logger.critical(f"EMERGENCY ROLLBACK for adapter: {adapter_id} - {reason}")
            
            # Execute emergency rollback
            rollback_id = await self.rollback.execute_emergency_rollback(
                adapter_id, reason
            )
            
            # Record metrics
            if self.metrics:
                self.metrics.record_metric(
                    "emergency_rollback", 1.0, adapter_id, MetricType.COUNTER,
                    tags={"trigger": "emergency"}
                )
            
            self.status.total_rollbacks += 1
            self.status.successful_rollbacks += 1
            logger.warning(f"Emergency rollback completed: {rollback_id}")
            return rollback_id
            
        except Exception as e:
            logger.critical(f"EMERGENCY ROLLBACK FAILED for {adapter_id}: {e}")
            
            if self.metrics:
                self.metrics.complete_operation_tracking(operation_id, success=False, error_message=str(e))
            
            raise
    
    # Registry operations
    
    async def register_adapter(self, metadata: AdapterMetadata) -> bool:
        """Register new adapter in registry."""
        try:
            success = self.registry.register_adapter(metadata)
            
            if success:
                logger.info(f"Registered adapter: {metadata.adapter_id}")
                
                if self.metrics:
                    self.metrics.record_metric(
                        "adapter_registered", 1.0, metadata.adapter_id, MetricType.COUNTER
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to register adapter {metadata.adapter_id}: {e}")
            return False
    
    async def list_adapters(self, 
                          adapter_type: Optional[AdapterType] = None,
                          medical_domain: Optional[str] = None) -> List[AdapterMetadata]:
        """List adapters with optional filtering."""
        return self.registry.list_adapters(adapter_type, medical_domain)
    
    async def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata."""
        return self.registry.get_adapter(adapter_id)
    
    # System operations
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        try:
            health_status = {
                "system_state": self.state.value,
                "timestamp": time.time(),
                "components": {},
                "overall_health": "healthy",
                "issues": []
            }
            
            # Check adapter manager
            if self.manager:
                manager_health = await self.manager.health_check()
                health_status["components"]["manager"] = manager_health
                
                if manager_health.get("status") != "healthy":
                    health_status["issues"].append("Adapter manager unhealthy")
            
            # Check registry
            if self.registry:
                registry_stats = self.registry.get_registry_stats()
                health_status["components"]["registry"] = {
                    "status": "healthy",
                    "stats": registry_stats
                }
            
            # Check cache
            if self.cache:
                cache_stats = self.cache.get_stats()
                health_status["components"]["cache"] = {
                    "status": "healthy" if cache_stats.get("cached_items", 0) >= 0 else "degraded",
                    "stats": cache_stats
                }
            
            # Check metrics system
            if self.metrics:
                system_overview = self.metrics.get_system_overview()
                health_status["components"]["metrics"] = {
                    "status": "healthy",
                    "overview": system_overview
                }
            
            # Determine overall health
            if health_status["issues"]:
                health_status["overall_health"] = "degraded"
            
            # Update status
            self.status.last_health_check = time.time()
            self.status.system_health = health_status["overall_health"]
            
            if health_status["overall_health"] != "healthy":
                await self._emit_event("health_check_failed", health_status)
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "system_state": self.state.value,
                "timestamp": time.time(),
                "overall_health": "error",
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Update current statistics
            if self.manager:
                stats = self.manager.get_manager_stats()
                self.status.active_adapters = stats.get("active_adapters", 0)
                self.status.memory_usage_mb = stats.get("memory_usage", {}).get("rss_mb", 0.0)
            
            if self.registry:
                registry_stats = self.registry.get_registry_stats()
                self.status.total_adapters = registry_stats.get("total_adapters", 0)
                
                # Medical compliance statistics
                validation_stats = registry_stats.get("validation_status", {})
                self.status.compliant_adapters = validation_stats.get("validated", 0)
                self.status.total_compliance_checks = sum(validation_stats.values())
            
            if self.hot_swap:
                active_swaps = await self.hot_swap.list_active_operations()
                self.status.active_operations = len(active_swaps)
            
            # Calculate success rates
            self.status.successful_swaps = self.status.total_swaps - len([
                op for op in getattr(self.hot_swap, 'operation_history', [])
                if not op.success
            ]) if hasattr(self.hot_swap, 'operation_history') else 0
            
            # Convert to dictionary
            status_dict = {
                "state": self.state.value,
                "initialized_at": self.status.initialized_at,
                "last_health_check": self.status.last_health_check,
                "active_adapters": self.status.active_adapters,
                "total_adapters": self.status.total_adapters,
                "active_operations": self.status.active_operations,
                "system_health": self.status.system_health,
                "statistics": {
                    "loads": {
                        "total": self.status.total_loads,
                        "successful": self.status.successful_loads,
                        "success_rate": self.status.successful_loads / self.status.total_loads if self.status.total_loads > 0 else 0.0,
                        "avg_time_ms": self.status.avg_load_time_ms
                    },
                    "swaps": {
                        "total": self.status.total_swaps,
                        "successful": self.status.successful_swaps,
                        "success_rate": self.status.successful_swaps / self.status.total_swaps if self.status.total_swaps > 0 else 0.0,
                        "avg_time_ms": self.status.avg_swap_time_ms
                    },
                    "rollbacks": {
                        "total": self.status.total_rollbacks,
                        "successful": self.status.successful_rollbacks,
                        "success_rate": self.status.successful_rollbacks / self.status.total_rollbacks if self.status.total_rollbacks > 0 else 0.0
                    },
                    "medical_compliance": {
                        "compliant_adapters": self.status.compliant_adapters,
                        "total_compliance_checks": self.status.total_compliance_checks,
                        "compliance_rate": self.status.compliant_adapters / self.status.total_adapters if self.status.total_adapters > 0 else 0.0
                    }
                },
                "memory_usage_mb": self.status.memory_usage_mb,
                "configuration": {
                    "base_model_id": self.config.base_model_id,
                    "max_memory_mb": self.config.max_memory_mb,
                    "enable_gpu_optimization": self.config.enable_gpu_optimization,
                    "auto_rollback_enabled": self.config.auto_rollback_enabled,
                    "enable_metrics": self.config.enable_metrics
                }
            }
            
            return status_dict
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {"error": str(e), "state": self.state.value}
    
    # Event system
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            logger.warning(f"Unknown event type: {event_type}")
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    logger.error(f"Event handler failed for {event_type}: {e}")
    
    # Private methods
    
    async def _perform_initial_health_check(self):
        """Perform initial health check after startup."""
        try:
            health_result = await self.health_check()
            
            if health_result.get("overall_health") != "healthy":
                logger.warning("Initial health check found issues")
                self.state = LifecycleState.DEGRADED
                self.status.state = LifecycleState.DEGRADED
            else:
                logger.info("Initial health check passed")
                
        except Exception as e:
            logger.error(f"Initial health check failed: {e}")
            self.state = LifecycleState.DEGRADED
            self.status.state = LifecycleState.DEGRADED


# Factory function and context manager

async def create_lifecycle_manager(config: SystemConfiguration) -> AdapterLifecycleManager:
    """Factory function to create lifecycle manager."""
    manager = AdapterLifecycleManager(config)
    await manager.startup()
    return manager


@asynccontextmanager
async def lifecycle_manager_context(config: SystemConfiguration) -> AsyncGenerator[AdapterLifecycleManager, None]:
    """Context manager for lifecycle manager."""
    manager = None
    try:
        manager = await create_lifecycle_manager(config)
        yield manager
    finally:
        if manager:
            await manager.shutdown()


# Utility functions

def create_default_config(base_model_id: str, 
                         medical_mode: bool = True) -> SystemConfiguration:
    """Create default system configuration."""
    config = SystemConfiguration(
        base_model_id=base_model_id,
        enable_medical_compliance=medical_mode,
        require_medical_validation=medical_mode,
        hipaa_compliance_required=medical_mode,
        clinical_trial_mode=medical_mode
    )
    
    if medical_mode:
        # More conservative settings for medical AI
        config.max_cache_size = 5
        config.max_memory_mb = 8192
        config.memory_optimization = MemoryOptimizationLevel.MEDICAL_STRICT
        config.auto_rollback_enabled = True
        config.enable_metrics = True
    
    return config


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create configuration
        config = create_default_config(
            base_model_id="microsoft/DialoGPT-medium",
            medical_mode=True
        )
        
        # Use context manager
        async with lifecycle_manager_context(config) as lifecycle:
            # Register an adapter
            from .registry import create_adapter_metadata, AdapterType
            metadata = create_adapter_metadata(
                adapter_id="medical_diagnosis_v1",
                name="Medical Diagnosis Assistant",
                description="LoRA adapter for medical diagnosis",
                adapter_type=AdapterType.MEDICAL_LORA,
                medical_domain="diagnostic_medicine"
            )
            
            await lifecycle.register_adapter(metadata)
            
            # Load adapter
            instance = await lifecycle.load_adapter("medical_diagnosis_v1")
            print(f"Loaded adapter: {instance.adapter_id}")
            
            # Get system status
            status = await lifecycle.get_system_status()
            print(f"System status: {status['state']}")
            
            # Health check
            health = await lifecycle.health_check()
            print(f"Health status: {health['overall_health']}")
    
    # asyncio.run(main())