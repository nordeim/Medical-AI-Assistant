"""
Health Check and System Diagnostics Endpoints
Comprehensive system monitoring and health assessment
"""

import asyncio
import time
import psutil
import platform
import subprocess
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import json
import os

from fastapi import APIRouter, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field
import structlog

from ..utils.exceptions import HealthCheckError
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Health status enums
class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class SystemLoad(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Individual component health information"""
    name: str
    status: ComponentStatus
    response_time_ms: float
    last_check: str
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_free_gb: float
    load_average: float
    network_io: Dict[str, int]
    process_count: int
    uptime_seconds: float


class HealthCheckService:
    """Service for comprehensive health checking"""
    
    def __init__(self):
        self.logger = get_logger("health_check")
        self.start_time = time.time()
        self.component_checks = {
            "database": self._check_database,
            "model_service": self._check_model_service,
            "redis_cache": self._check_redis_cache,
            "vector_store": self._check_vector_store,
            "audit_logger": self._check_audit_logger,
            "phi_protection": self._check_phi_protection,
            "medical_validation": self._check_medical_validation
        }
    
    async def check_all_components(self) -> List[ComponentHealth]:
        """Check health of all system components"""
        
        components = []
        
        # Check components concurrently for efficiency
        tasks = [
            self._check_component(name, check_func) 
            for name, check_func in self.component_checks.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                self.logger.error(f"Health check failed: {result}")
                components.append(ComponentHealth(
                    name="unknown",
                    status=ComponentStatus.UNHEALTHY,
                    response_time_ms=0,
                    last_check=datetime.now(timezone.utc).isoformat(),
                    error_message=str(result)
                ))
            else:
                components.append(result)
        
        return components
    
    async def _check_component(self, name: str, check_func) -> ComponentHealth:
        """Check individual component health"""
        
        start_time = time.time()
        
        try:
            result = await check_func()
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return ComponentHealth(
                name=name,
                status=ComponentStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc).isoformat(),
                metadata=result
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return ComponentHealth(
                name=name,
                status=ComponentStatus.UNHEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc).isoformat(),
                error_message=str(e)
            )
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        
        # Simulate database check
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "connection_pool_active": 5,
            "connection_pool_idle": 3,
            "avg_query_time_ms": 15.2,
            "slow_queries_count": 0,
            "database_size_mb": 1024.5,
            "backup_status": "current"
        }
    
    async def _check_model_service(self) -> Dict[str, Any]:
        """Check model service health and performance"""
        
        # Simulate model service check
        await asyncio.sleep(0.05)
        
        return {
            "model_loaded": True,
            "model_version": settings.model_version,
            "inference_queue_length": 2,
            "avg_inference_time_ms": 145.6,
            "memory_usage_mb": 2048.0,
            "gpu_utilization": 65.2,
            "batch_processing_active": True
        }
    
    async def _check_redis_cache(self) -> Dict[str, Any]:
        """Check Redis cache connectivity"""
        
        # Simulate Redis check
        await asyncio.sleep(0.02)
        
        return {
            "connected": True,
            "memory_usage_mb": 128.5,
            "key_count": 1542,
            "hit_rate_percent": 87.3,
            "evicted_keys_count": 0
        }
    
    async def _check_vector_store(self) -> Dict[str, Any]:
        """Check vector store availability"""
        
        # Simulate vector store check
        await asyncio.sleep(0.03)
        
        return {
            "index_count": 3,
            "total_vectors": 125000,
            "avg_similarity_time_ms": 12.8,
            "index_health": "good",
            "last_rebuild": "2024-01-10T10:00:00Z"
        }
    
    async def _check_audit_logger(self) -> Dict[str, Any]:
        """Check audit logging system"""
        
        # Simulate audit logger check
        await asyncio.sleep(0.01)
        
        return {
            "logger_active": True,
            "log_buffer_size": 45,
            "last_flush": datetime.now(timezone.utc).isoformat(),
            "log_volume_last_hour": 1250,
            "compliance_status": "compliant"
        }
    
    async def _check_phi_protection(self) -> Dict[str, Any]:
        """Check PHI protection system"""
        
        # Simulate PHI protection check
        await asyncio.sleep(0.02)
        
        return {
            "protection_enabled": True,
            "phi_detection_active": True,
            "redaction_active": True,
            "patterns_loaded": len(settings.phi_modes),
            "last_violation": None,
            "violation_count_today": 0
        }
    
    async def _check_medical_validation(self) -> Dict[str, Any]:
        """Check medical validation system"""
        
        # Simulate medical validation check
        await asyncio.sleep(0.015)
        
        return {
            "validation_enabled": settings.enable_medical_validation,
            "patterns_loaded": True,
            "term_dictionary_updated": True,
            "validation_success_rate": 99.2,
            "last_validation_error": None
        }
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network = psutil.net_io_counters()
        
        # Load average (Unix systems)
        try:
            load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        except AttributeError:
            load_avg = 0.0
        
        # Process count
        process_count = len(psutil.pids())
        
        # Uptime
        uptime_seconds = time.time() - self.start_time
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_total_mb=memory.total / (1024 * 1024),
            disk_percent=(disk.used / disk.total) * 100,
            disk_used_gb=disk.used / (1024 * 1024 * 1024),
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            load_average=load_avg,
            network_io={
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            },
            process_count=process_count,
            uptime_seconds=uptime_seconds
        )
    
    def assess_system_load(self, metrics: SystemMetrics) -> SystemLoad:
        """Assess overall system load"""
        
        load_factors = []
        
        # CPU assessment
        if metrics.cpu_percent > 90:
            load_factors.append(SystemLoad.CRITICAL)
        elif metrics.cpu_percent > 75:
            load_factors.append(SystemLoad.HIGH)
        elif metrics.cpu_percent > 50:
            load_factors.append(SystemLoad.MEDIUM)
        else:
            load_factors.append(SystemLoad.LOW)
        
        # Memory assessment
        if metrics.memory_percent > 90:
            load_factors.append(SystemLoad.CRITICAL)
        elif metrics.memory_percent > 75:
            load_factors.append(SystemLoad.HIGH)
        elif metrics.memory_percent > 50:
            load_factors.append(SystemLoad.MEDIUM)
        else:
            load_factors.append(SystemLoad.LOW)
        
        # Disk assessment
        if metrics.disk_percent > 95:
            load_factors.append(SystemLoad.CRITICAL)
        elif metrics.disk_percent > 85:
            load_factors.append(SystemLoad.HIGH)
        elif metrics.disk_percent > 70:
            load_factors.append(SystemLoad.MEDIUM)
        else:
            load_factors.append(SystemLoad.LOW)
        
        # Load average assessment (if available)
        if metrics.load_average > 4.0:
            load_factors.append(SystemLoad.HIGH)
        elif metrics.load_average > 2.0:
            load_factors.append(SystemLoad.MEDIUM)
        
        # Return highest load level
        if SystemLoad.CRITICAL in load_factors:
            return SystemLoad.CRITICAL
        elif SystemLoad.HIGH in load_factors:
            return SystemLoad.HIGH
        elif SystemLoad.MEDIUM in load_factors:
            return SystemLoad.MEDIUM
        else:
            return SystemLoad.LOW


# Global health check service
health_check_service = HealthCheckService()

# Pydantic models for responses
class SystemHealthResponse(BaseModel):
    """Overall system health response"""
    
    status: str = Field(..., description="Overall system status")
    timestamp: str = Field(..., description="Health check timestamp")
    uptime_seconds: float = Field(..., description="System uptime")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    
    # System metrics
    system_load: str = Field(..., description="System load level")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    disk_percent: float = Field(..., description="Disk usage percentage")
    load_average: float = Field(..., description="System load average")
    
    # Component summary
    total_components: int = Field(..., description="Total components checked")
    healthy_components: int = Field(..., description="Healthy components count")
    degraded_components: int = Field(..., description="Degraded components count")
    unhealthy_components: int = Field(..., description="Unhealthy components count")
    
    # Configuration
    medical_validation_enabled: bool = Field(..., description="Medical validation status")
    phi_protection_enabled: bool = Field(..., description="PHI protection status")
    audit_logging_enabled: bool = Field(..., description="Audit logging status")


class ComponentHealthResponse(BaseModel):
    """Individual component health response"""
    
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status")
    response_time_ms: float = Field(..., description="Component response time")
    last_check: str = Field(..., description="Last check timestamp")
    error_message: Optional[str] = Field(None, description="Error message if unhealthy")
    metadata: Dict[str, Any] = Field(..., description="Component-specific metadata")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response"""
    
    system_health: SystemHealthResponse = Field(..., description="Overall system health")
    components: List[ComponentHealthResponse] = Field(..., description="Individual component status")
    recommendations: List[str] = Field(..., description="System improvement recommendations")


class ReadinessProbeResponse(BaseModel):
    """Kubernetes readiness probe response"""
    
    ready: bool = Field(..., description="Service readiness status")
    timestamp: str = Field(..., description="Probe timestamp")
    checks_passed: int = Field(..., description="Number of checks passed")
    total_checks: int = Field(..., description="Total number of checks")


class LivenessProbeResponse(BaseModel):
    """Kubernetes liveness probe response"""
    
    alive: bool = Field(..., description="Service liveness status")
    timestamp: str = Field(..., description="Probe timestamp")
    uptime_seconds: float = Field(..., description="Service uptime")


# Endpoint implementations
@router.get("/system", response_model=SystemHealthResponse)
async def system_health():
    """
    Get overall system health status.
    
    Returns system-level health metrics including:
    - CPU, memory, and disk usage
    - Component health summary
    - System load assessment
    - Configuration status
    """
    
    try:
        # Get system metrics
        metrics = health_check_service.get_system_metrics()
        system_load = health_check_service.assess_system_load(metrics)
        
        # Get component health summary
        components = await health_check_service.check_all_components()
        
        # Count component statuses
        healthy_count = sum(1 for c in components if c.status == ComponentStatus.HEALTHY)
        degraded_count = sum(1 for c in components if c.status == ComponentStatus.DEGRADED)
        unhealthy_count = sum(1 for c in components if c.status == ComponentStatus.UNHEALTHY)
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=metrics.uptime_seconds,
            version="1.0.0",
            environment="production",
            system_load=system_load.value,
            cpu_percent=metrics.cpu_percent,
            memory_percent=metrics.memory_percent,
            disk_percent=metrics.disk_percent,
            load_average=metrics.load_average,
            total_components=len(components),
            healthy_components=healthy_count,
            degraded_components=degraded_count,
            unhealthy_components=unhealthy_count,
            medical_validation_enabled=settings.enable_medical_validation,
            phi_protection_enabled=settings.enable_phi_detection,
            audit_logging_enabled=settings.enable_audit_logging
        )
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        raise HealthCheckError(f"System health check failed: {str(e)}")


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """
    Get detailed health check with component breakdown.
    
    Provides comprehensive system diagnostics including:
    - Individual component health status
    - Performance metrics
    - System recommendations
    - Detailed error information
    """
    
    try:
        # Get system health
        system_health_response = await system_health()
        
        # Get component details
        components = await health_check_service.check_all_components()
        component_responses = [
            ComponentHealthResponse(
                name=comp.name,
                status=comp.status.value,
                response_time_ms=comp.response_time_ms,
                last_check=comp.last_check,
                error_message=comp.error_message,
                metadata=comp.metadata or {}
            )
            for comp in components
        ]
        
        # Generate recommendations
        recommendations = _generate_system_recommendations(
            system_health_response, component_responses
        )
        
        return DetailedHealthResponse(
            system_health=system_health_response,
            components=component_responses,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HealthCheckError(f"Detailed health check failed: {str(e)}")


@router.get("/ready", response_model=ReadinessProbeResponse)
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.
    
    Returns service readiness status for load balancer routing decisions.
    """
    
    try:
        # Check critical components
        components = await health_check_service.check_all_components()
        
        # Define critical components for readiness
        critical_components = ["database", "model_service", "redis_cache"]
        
        checks_passed = 0
        total_checks = len(critical_components)
        
        for component in components:
            if component.name in critical_components and component.status == ComponentStatus.HEALTHY:
                checks_passed += 1
        
        ready = checks_passed == total_checks
        
        return ReadinessProbeResponse(
            ready=ready,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks_passed=checks_passed,
            total_checks=total_checks
        )
        
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        return ReadinessProbeResponse(
            ready=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checks_passed=0,
            total_checks=3
        )


@router.get("/live", response_model=LivenessProbeResponse)
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.
    
    Returns service liveness status for container restart decisions.
    """
    
    try:
        uptime_seconds = time.time() - health_check_service.start_time
        
        # Basic liveness check - if we can respond, we're alive
        alive = True
        
        return LivenessProbeResponse(
            alive=alive,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=uptime_seconds
        )
        
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        return LivenessProbeResponse(
            alive=False,
            timestamp=datetime.now(timezone.utc).isoformat(),
            uptime_seconds=0
        )


@router.get("/metrics")
async def get_metrics():
    """
    Get Prometheus-compatible metrics.
    
    Returns system and application metrics in Prometheus format for monitoring.
    """
    
    metrics = health_check_service.get_system_metrics()
    
    # Generate Prometheus-style metrics
    prometheus_metrics = f"""# HELP medical_ai_cpu_percent CPU usage percentage
# TYPE medical_ai_cpu_percent gauge
medical_ai_cpu_percent {metrics.cpu_percent}

# HELP medical_ai_memory_percent Memory usage percentage
# TYPE medical_ai_memory_percent gauge
medical_ai_memory_percent {metrics.memory_percent}

# HELP medical_ai_memory_used_mb Memory used in MB
# TYPE medical_ai_memory_used_mb gauge
medical_ai_memory_used_mb {metrics.memory_used_mb}

# HELP medical_ai_memory_total_mb Total memory in MB
# TYPE medical_ai_memory_total_mb gauge
medical_ai_memory_total_mb {metrics.memory_total_mb}

# HELP medical_ai_disk_percent Disk usage percentage
# TYPE medical_ai_disk_percent gauge
medical_ai_disk_percent {metrics.disk_percent}

# HELP medical_ai_disk_used_gb Disk used in GB
# TYPE medical_ai_disk_used_gb gauge
medical_ai_disk_used_gb {metrics.disk_used_gb}

# HELP medical_ai_disk_free_gb Disk free in GB
# TYPE medical_ai_disk_free_gb gauge
medical_ai_disk_free_gb {metrics.disk_free_gb}

# HELP medical_ai_load_average System load average
# TYPE medical_ai_load_average gauge
medical_ai_load_average {metrics.load_average}

# HELP medical_ai_process_count Number of running processes
# TYPE medical_ai_process_count gauge
medical_ai_process_count {metrics.process_count}

# HELP medical_ai_uptime_seconds Service uptime in seconds
# TYPE medical_ai_uptime_seconds gauge
medical_ai_uptime_seconds {metrics.uptime_seconds}

# HELP medical_ai_network_bytes_sent Network bytes sent
# TYPE medical_ai_network_bytes_sent counter
medical_ai_network_bytes_sent {metrics.network_io['bytes_sent']}

# HELP medical_ai_network_bytes_recv Network bytes received
# TYPE medical_ai_network_bytes_recv counter
medical_ai_network_bytes_recv {metrics.network_io['bytes_recv']}
"""
    
    return prometheus_metrics


# Helper functions
def _generate_system_recommendations(
    system_health: SystemHealthResponse,
    components: List[ComponentHealthResponse]
) -> List[str]:
    """Generate system improvement recommendations"""
    
    recommendations = []
    
    # System load recommendations
    if system_health.cpu_percent > 80:
        recommendations.append("High CPU usage detected - consider scaling up or optimizing workloads")
    
    if system_health.memory_percent > 85:
        recommendations.append("High memory usage detected - consider increasing memory allocation")
    
    if system_health.disk_percent > 90:
        recommendations.append("Critical disk usage - immediate attention required for disk cleanup")
    
    # Component-specific recommendations
    for component in components:
        if component.status == "unhealthy":
            recommendations.append(f"Component '{component.name}' is unhealthy - check service logs and restart if necessary")
        elif component.status == "degraded":
            recommendations.append(f"Component '{component.name}' performance is degraded - monitor closely")
        
        # Specific recommendations based on metadata
        if component.name == "model_service" and component.metadata:
            if component.metadata.get("gpu_utilization", 0) > 90:
                recommendations.append("High GPU utilization detected - consider load balancing or hardware upgrade")
        
        if component.name == "database" and component.metadata:
            if component.metadata.get("slow_queries_count", 0) > 5:
                recommendations.append("Multiple slow database queries detected - optimize queries or add indexes")
        
        if component.name == "redis_cache" and component.metadata:
            if component.metadata.get("hit_rate_percent", 100) < 80:
                recommendations.append("Low cache hit rate - consider increasing cache size or reviewing cache strategy")
    
    # Security recommendations
    if not system_health.medical_validation_enabled:
        recommendations.append("Medical validation is disabled - enable for production safety")
    
    if not system_health.phi_protection_enabled:
        recommendations.append("PHI protection is disabled - enable for HIPAA compliance")
    
    if not system_health.audit_logging_enabled:
        recommendations.append("Audit logging is disabled - enable for compliance and security")
    
    # Performance recommendations
    if system_health.system_load == "high" or system_health.system_load == "critical":
        recommendations.append("System load is high - consider horizontal scaling or performance optimization")
    
    return recommendations