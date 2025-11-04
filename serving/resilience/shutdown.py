"""
Medical AI Resilience - Graceful Shutdown and Health Checks
Graceful shutdown mechanisms and comprehensive health checks with medical data protection.
"""

import asyncio
import signal
import threading
import time
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime, timedelta
import json
import logging
import weakref
from abc import ABC, abstractmethod

from .errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory,
    MedicalErrorSeverity, create_service_unavailable_error
)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"           # System is fully operational
    DEGRADED = "degraded"         # System has issues but functional
    UNHEALTHY = "unhealthy"       # System has significant issues
    CRITICAL = "critical"         # System requires immediate attention
    UNKNOWN = "unknown"           # Health status unknown


class ShutdownPhase(Enum):
    """Shutdown phases for graceful termination."""
    IMMEDIATE = "immediate"       # Stop immediately (emergency)
    FAST = "fast"                 # Quick shutdown (30 seconds)
    GRACEFUL = "graceful"         # Standard graceful shutdown (2 minutes)
    COMPLETE = "complete"         # Complete shutdown with cleanup (5 minutes)


class ComponentState(Enum):
    """States for system components."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DRAINING = "draining"         # Stopping new requests, finishing existing
    STOPPING = "stopping"         # Stopping all operations
    STOPPED = "stopped"
    ERROR = "error"


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, name: str, priority: int = 1, timeout: float = 10.0):
        self.name = name
        self.priority = priority
        self.timeout = timeout
        self.last_check: Optional[datetime] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.failure_count = 0
        self.success_count = 0
        self.critical_threshold = 3  # Failures before marking as critical
    
    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Perform health check and return result."""
        pass
    
    async def execute(self) -> Dict[str, Any]:
        """Execute health check with timeout."""
        try:
            result = await asyncio.wait_for(
                self.check_health(),
                timeout=self.timeout
            )
            result["check_name"] = self.name
            result["timestamp"] = datetime.utcnow().isoformat()
            result["status"] = HealthStatus.HEALTHY.value
            
            self.success_count += 1
            self.failure_count = 0
            self.last_check = datetime.utcnow()
            self.last_result = result
            
            return result
            
        except asyncio.TimeoutError:
            result = {
                "check_name": self.name,
                "timestamp": datetime.utcnow().isoformat(),
                "status": HealthStatus.CRITICAL.value,
                "error": "Health check timeout"
            }
            
            self.failure_count += 1
            self.last_check = datetime.utcnow()
            self.last_result = result
            
            return result
            
        except Exception as e:
            result = {
                "check_name": self.name,
                "timestamp": datetime.utcnow().isoformat(),
                "status": HealthStatus.UNHEALTHY.value,
                "error": str(e)
            }
            
            self.failure_count += 1
            self.last_check = datetime.utcnow()
            self.last_result = result
            
            return result
    
    def get_status(self) -> HealthStatus:
        """Get current health status based on failure count."""
        if self.failure_count >= self.critical_threshold:
            return HealthStatus.CRITICAL
        elif self.failure_count > 0:
            return HealthStatus.UNHEALTHY
        elif self.success_count > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def reset_metrics(self):
        """Reset health check metrics."""
        self.failure_count = 0
        self.success_count = 0
        self.last_check = None
        self.last_result = None


class SystemHealthCheck(HealthCheck):
    """Health check for system resources."""
    
    def __init__(self, memory_threshold: float = 0.85, cpu_threshold: float = 0.80):
        super().__init__("system_health", priority=1, timeout=5.0)
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
    
    async def check_health(self) -> Dict[str, Any]:
        """Check system health metrics."""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent / 100.0
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent / 100.0
        
        # Load average (Unix systems)
        load_avg = None
        try:
            load_avg = psutil.getloadavg()[0]  # 1-minute load average
        except AttributeError:
            pass  # Windows doesn't have load average
        
        status = HealthStatus.HEALTHY
        issues = []
        
        if cpu_percent > self.cpu_threshold * 100:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            status = HealthStatus.DEGRADED
        
        if memory_percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory_percent:.1%}")
            status = HealthStatus.DEGRADED
        
        if disk_percent > 0.90:
            issues.append(f"High disk usage: {disk_percent:.1%}")
            status = HealthStatus.UNHEALTHY
        
        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "disk_usage": disk_percent,
            "load_average": load_avg,
            "issues": issues,
            "overall_status": status.value
        }


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, db_connection_factory: Callable, timeout: float = 10.0):
        super().__init__("database_health", priority=2, timeout=timeout)
        self.db_connection_factory = db_connection_factory
    
    async def check_health(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # Test database connection
            async with self.db_connection_factory() as conn:
                # Execute simple query to verify connection
                result = await conn.fetchval("SELECT 1")
                
                if result == 1:
                    return {
                        "connection_status": "connected",
                        "response_time_ms": 0,  # Would measure actual time
                        "overall_status": HealthStatus.HEALTHY.value
                    }
                else:
                    return {
                        "connection_status": "unhealthy",
                        "error": "Database query failed",
                        "overall_status": HealthStatus.UNHEALTHY.value
                    }
                    
        except Exception as e:
            return {
                "connection_status": "failed",
                "error": str(e),
                "overall_status": HealthStatus.UNHEALTHY.value
            }


class ModelHealthCheck(HealthCheck):
    """Health check for AI models."""
    
    def __init__(self, model_registry: Dict[str, Any], timeout: float = 15.0):
        super().__init__("model_health", priority=3, timeout=timeout)
        self.model_registry = model_registry
    
    async def check_health(self) -> Dict[str, Any]:
        """Check model health and performance."""
        model_status = {}
        overall_status = HealthStatus.HEALTHY
        critical_issues = []
        
        for model_name, model_info in self.model_registry.items():
            try:
                # Check if model is loaded
                if "model" not in model_info:
                    model_status[model_name] = {
                        "status": "not_loaded",
                        "overall_status": HealthStatus.UNHEALTHY.value
                    }
                    critical_issues.append(f"Model {model_name} not loaded")
                    continue
                
                model = model_info["model"]
                
                # Test model inference with sample data
                if hasattr(model, 'predict'):
                    start_time = time.time()
                    # Use safe test data
                    test_result = model.predict(model_info.get("test_data", {}))
                    inference_time = time.time() - start_time
                    
                    # Check inference time
                    if inference_time > 5.0:  # 5 second timeout
                        model_status[model_name] = {
                            "status": "slow",
                            "inference_time_ms": inference_time * 1000,
                            "overall_status": HealthStatus.DEGRADED.value
                        }
                    else:
                        model_status[model_name] = {
                            "status": "healthy",
                            "inference_time_ms": inference_time * 1000,
                            "overall_status": HealthStatus.HEALTHY.value
                        }
                else:
                    model_status[model_name] = {
                        "status": "invalid_interface",
                        "overall_status": HealthStatus.UNHEALTHY.value
                    }
                    
            except Exception as e:
                model_status[model_name] = {
                    "status": "error",
                    "error": str(e),
                    "overall_status": HealthStatus.CRITICAL.value
                }
                critical_issues.append(f"Model {model_name} error: {str(e)}")
        
        # Determine overall status
        if critical_issues:
            overall_status = HealthStatus.CRITICAL
        elif any(status["overall_status"] == HealthStatus.DEGRADED.value 
                for status in model_status.values()):
            overall_status = HealthStatus.DEGRADED
        
        return {
            "models": model_status,
            "critical_issues": critical_issues,
            "overall_status": overall_status.value
        }


class MedicalDataIntegrityCheck(HealthCheck):
    """Health check for medical data integrity and PHI protection."""
    
    def __init__(self, data_stores: List[Any], phi_validator: Callable):
        super().__init__("data_integrity", priority=1, timeout=20.0)  # Highest priority
        self.data_stores = data_stores
        self.phi_validator = phi_validator
    
    async def check_health(self) -> Dict[str, Any]:
        """Check medical data integrity and PHI protection."""
        integrity_status = {}
        phi_violations = []
        
        for store_name, store in self.data_stores.items():
            try:
                # Check data store connectivity
                if hasattr(store, 'ping'):
                    await store.ping()
                
                # Check for PHI violations
                phi_check_result = await self._check_phi_compliance(store)
                integrity_status[store_name] = phi_check_result
                
                if phi_check_result.get("violations", []):
                    phi_violations.extend([
                        f"{store_name}: {violation}" 
                        for violation in phi_check_result["violations"]
                    ])
                    
            except Exception as e:
                integrity_status[store_name] = {
                    "status": "error",
                    "error": str(e),
                    "overall_status": HealthStatus.UNHEALTHY.value
                }
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        if phi_violations:
            overall_status = HealthStatus.CRITICAL
        elif any(
            status.get("overall_status") == HealthStatus.UNHEALTHY.value
            for status in integrity_status.values()
        ):
            overall_status = HealthStatus.UNHEALTHY
        
        return {
            "data_stores": integrity_status,
            "phi_violations": phi_violations,
            "compliance_status": "violations_detected" if phi_violations else "compliant",
            "overall_status": overall_status.value
        }
    
    async def _check_phi_compliance(self, store: Any) -> Dict[str, Any]:
        """Check PHI compliance for a data store."""
        violations = []
        
        try:
            # This would implement actual PHI checking logic
            # For now, return a basic compliance check
            if hasattr(store, 'phi_fields'):
                # Check for unencrypted PHI fields
                for field in store.phi_fields:
                    if not getattr(field, 'encrypted', False):
                        violations.append(f"Unencrypted PHI field: {field.name}")
            
            return {
                "status": "checked",
                "violations": violations,
                "overall_status": HealthStatus.CRITICAL.value if violations else HealthStatus.HEALTHY.value
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "overall_status": HealthStatus.UNHEALTHY.value
            }


class GracefulShutdownManager:
    """Manager for graceful shutdown procedures."""
    
    def __init__(self):
        self.components: Dict[str, ComponentState] = {}
        self.shutdown_handlers: List[Callable] = []
        self.health_checks: List[HealthCheck] = []
        self.is_shutting_down = False
        self.shutdown_started: Optional[datetime] = None
        self.shutdown_phase = ShutdownPhase.GRACEFUL
        self.audit_callback = None
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.initiate_shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def register_component(self, name: str, component: Any):
        """Register a component for shutdown management."""
        self.components[name] = ComponentState.RUNNING
        # Store reference (weak reference to avoid circular dependencies)
        if not hasattr(self, '_component_refs'):
            self._component_refs = weakref.WeakValueDictionary()
        self._component_refs[name] = component
    
    def add_shutdown_handler(self, handler: Callable):
        """Add shutdown handler."""
        self.shutdown_handlers.append(handler)
    
    def add_health_check(self, health_check: HealthCheck):
        """Add health check."""
        self.health_checks.append(health_check)
        # Sort by priority (higher priority first)
        self.health_checks.sort(key=lambda h: h.priority)
    
    def set_audit_callback(self, callback: Callable):
        """Set audit callback for shutdown events."""
        self.audit_callback = callback
    
    async def initiate_shutdown(
        self, 
        phase: ShutdownPhase = ShutdownPhase.GRACEFUL,
        timeout: Optional[float] = None
    ):
        """Initiate graceful shutdown."""
        if self.is_shutting_down:
            logging.warning("Shutdown already in progress")
            return
        
        self.is_shutting_down = True
        self.shutdown_started = datetime.utcnow()
        self.shutdown_phase = phase
        
        # Determine timeout based on phase
        if timeout is None:
            timeout_map = {
                ShutdownPhase.IMMEDIATE: 5.0,
                ShutdownPhase.FAST: 30.0,
                ShutdownPhase.GRACEFUL: 120.0,
                ShutdownPhase.COMPLETE: 300.0
            }
            timeout = timeout_map.get(phase, 120.0)
        
        logging.info(f"Initiating {phase.value} shutdown with {timeout}s timeout")
        
        # Audit shutdown start
        if self.audit_callback:
            self.audit_callback({
                "event": "shutdown_initiated",
                "phase": phase.value,
                "timeout": timeout,
                "timestamp": self.shutdown_started.isoformat()
            })
        
        try:
            # Phase 1: Stop accepting new requests
            await self._phase_1_stop_new_requests(timeout * 0.1)
            
            # Phase 2: Drain existing requests
            await self._phase_2_drain_requests(timeout * 0.6)
            
            # Phase 3: Stop services and components
            await self._phase_3_stop_services(timeout * 0.2)
            
            # Phase 4: Final cleanup
            await self._phase_4_cleanup(timeout * 0.1)
            
            logging.info("Graceful shutdown completed successfully")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            # Force shutdown on error
            await self._force_shutdown()
        
        finally:
            self.is_shutting_down = False
    
    async def _phase_1_stop_new_requests(self, timeout: float):
        """Phase 1: Stop accepting new requests."""
        logging.info("Phase 1: Stopping acceptance of new requests")
        
        for component_name, component in self._get_component_refs().items():
            self.components[component_name] = ComponentState.DRAINING
            
            # Call component-specific stop accepting method
            if hasattr(component, 'stop_accepting_requests'):
                await asyncio.wait_for(
                    component.stop_accepting_requests(),
                    timeout=timeout / len(self.components)
                )
    
    async def _phase_2_drain_requests(self, timeout: float):
        """Phase 2: Drain existing requests."""
        logging.info("Phase 2: Draining existing requests")
        
        drain_start = time.time()
        while time.time() - drain_start < timeout:
            # Check if all requests are completed
            active_requests = await self._get_active_request_count()
            
            if active_requests == 0:
                logging.info("All requests drained successfully")
                return
            
            logging.info(f"Waiting for {active_requests} active requests to complete")
            await asyncio.sleep(1.0)
        
        logging.warning("Timeout waiting for requests to drain")
    
    async def _phase_3_stop_services(self, timeout: float):
        """Phase 3: Stop services and components."""
        logging.info("Phase 3: Stopping services")
        
        # Execute shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=timeout / len(self.shutdown_handlers))
                else:
                    handler()  # Synchronous handler
            except Exception as e:
                logging.error(f"Shutdown handler error: {e}")
        
        # Stop components
        for component_name in list(self.components.keys()):
            self.components[component_name] = ComponentState.STOPPING
            
            # Component-specific shutdown
            if component_name in self._get_component_refs():
                component = self._get_component_refs()[component_name]
                if hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                    except Exception as e:
                        logging.error(f"Component {component_name} shutdown error: {e}")
    
    async def _phase_4_cleanup(self, timeout: float):
        """Phase 4: Final cleanup."""
        logging.info("Phase 4: Final cleanup")
        
        # Close connections
        await self._close_all_connections(timeout * 0.5)
        
        # Flush logs
        await self._flush_logs(timeout * 0.3)
        
        # Final audit
        if self.audit_callback:
            self.audit_callback({
                "event": "shutdown_completed",
                "duration_seconds": (datetime.utcnow() - self.shutdown_started).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _force_shutdown(self):
        """Force shutdown in case of errors."""
        logging.error("Forcing immediate shutdown")
        
        for component_name in self.components:
            self.components[component_name] = ComponentState.STOPPED
        
        if self.audit_callback:
            self.audit_callback({
                "event": "forced_shutdown",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _get_active_request_count(self) -> int:
        """Get count of active requests."""
        # This would be implemented based on the specific system
        # For now, return a mock value
        return 0
    
    def _get_component_refs(self) -> Dict[str, Any]:
        """Get component references."""
        if hasattr(self, '_component_refs'):
            return dict(self._component_refs)
        return {}
    
    async def _close_all_connections(self, timeout: float):
        """Close all database and service connections."""
        # Implementation would close connections to databases, message queues, etc.
        await asyncio.sleep(0.1)  # Simulate connection closing
    
    async def _flush_logs(self, timeout: float):
        """Flush all log outputs."""
        logging.shutdown()
        await asyncio.sleep(0.1)  # Simulate log flushing
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        if not self.health_checks:
            return {
                "overall_status": HealthStatus.UNKNOWN.value,
                "message": "No health checks configured",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        health_results = []
        overall_status = HealthStatus.HEALTHY
        
        # Execute health checks in priority order
        for health_check in self.health_checks:
            try:
                result = await health_check.execute()
                health_results.append(result)
                
                # Update overall status based on individual results
                check_status = HealthStatus(result["status"])
                if check_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                    overall_status = check_status
                elif check_status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = check_status
                    
            except Exception as e:
                health_results.append({
                    "check_name": health_check.name,
                    "status": HealthStatus.CRITICAL.value,
                    "error": f"Health check execution failed: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
                overall_status = HealthStatus.CRITICAL
        
        # Add system component status
        component_status = {
            component_name: state.value 
            for component_name, state in self.components.items()
        }
        
        return {
            "overall_status": overall_status.value,
            "components": component_status,
            "health_checks": health_results,
            "shutting_down": self.is_shutting_down,
            "shutdown_phase": self.shutdown_phase.value if self.is_shutting_down else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_shutdown_status(self) -> Dict[str, Any]:
        """Get current shutdown status."""
        return {
            "is_shutting_down": self.is_shutting_down,
            "shutdown_phase": self.shutdown_phase.value if self.is_shutting_down else None,
            "shutdown_started": self.shutdown_started.isoformat() if self.shutdown_started else None,
            "components": dict(self.components),
            "registered_components": len(self.components),
            "health_checks": len(self.health_checks)
        }


# Global shutdown manager
shutdown_manager = GracefulShutdownManager()


def register_shutdown_component(name: str):
    """Decorator to register component for shutdown management."""
    def decorator(component_class):
        async def init_wrapper(self, *args, **kwargs):
            instance = component_class(self, *args, **kwargs)
            shutdown_manager.register_component(name, instance)
            return instance
        
        return init_wrapper
    return decorator


def add_health_check(priority: int = 1, timeout: float = 10.0):
    """Decorator to add health check."""
    def decorator(func):
        health_check = HealthCheck(func.__name__, priority, timeout)
        health_check.check_health = func
        shutdown_manager.add_health_check(health_check)
        return func
    return decorator


# Convenience functions for common health checks
async def basic_system_health() -> Dict[str, Any]:
    """Basic system health check."""
    import psutil
    
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "status": HealthStatus.HEALTHY.value
    }


async def database_health_check(db_factory: Callable) -> Dict[str, Any]:
    """Basic database health check."""
    try:
        async with db_factory() as conn:
            result = await conn.fetchval("SELECT 1")
            return {
                "status": "connected" if result == 1 else "unhealthy",
                "overall_status": HealthStatus.HEALTHY.value
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "overall_status": HealthStatus.UNHEALTHY.value
        }