"""
Connection pooling and optimization for medical system scalability.
Provides efficient connection management for medical AI assistant with
resource optimization, health monitoring, and medical compliance.
"""

import asyncio
import time
import weakref
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, Any, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import structlog

from ...config.settings import get_settings


# Configuration
settings = get_settings()
logger = structlog.get_logger("connection_pool")


class ConnectionType(str, Enum):
    """Types of connections in the pool."""
    WEBSOCKET = "websocket"
    HTTP_CLIENT = "http_client"
    DATABASE = "database"
    REDIS = "redis"
    MODEL_CLIENT = "model_client"
    MEDICAL_API = "medical_api"


class PoolStatus(str, Enum):
    """Connection pool status."""
    ACTIVE = "active"
    IDLE = "idle"
    EXHAUSTED = "exhausted"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass
class ConnectionMetrics:
    """Metrics for connection performance."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    total_response_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_health_check: Optional[datetime] = None
    uptime_seconds: float = 0.0


@dataclass
class PoolConfig:
    """Configuration for connection pool."""
    max_connections: int = 100
    min_connections: int = 10
    max_idle_time: int = 300  # seconds
    health_check_interval: int = 60  # seconds
    connection_timeout: int = 30  # seconds
    request_timeout: int = 120  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0
    rate_limit: int = 1000  # requests per minute
    enable_metrics: bool = True
    enable_health_checks: bool = True
    enable_auto_scaling: bool = True
    medical_priority: bool = True
    compliance_mode: bool = True


@dataclass
class ConnectionInfo:
    """Information about a connection in the pool."""
    connection_id: str
    connection_type: ConnectionType
    created_at: datetime
    last_used: datetime
    last_health_check: datetime
    request_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    is_healthy: bool = True
    priority_level: int = 1  # 1-10, 10 being highest priority
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalConnectionPool:
    """Medical-grade connection pool with compliance and monitoring."""
    
    def __init__(self, pool_config: PoolConfig = None):
        self.config = pool_config or PoolConfig()
        self.pools: Dict[ConnectionType, Dict[str, Any]] = {}
        self.metrics: Dict[ConnectionType, ConnectionMetrics] = {}
        self.connection_map: Dict[str, Tuple[ConnectionType, str]] = {}
        self.active_connections: Dict[str, ConnectionInfo] = {}
        self.idle_connections: Dict[ConnectionType, deque] = defaultdict(deque)
        self.health_check_tasks: Dict[ConnectionType, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        self.logger = structlog.get_logger("pool.manager")
        
        # Initialize pools for each connection type
        self._initialize_pools()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_pools(self):
        """Initialize connection pools for all types."""
        for conn_type in ConnectionType:
            self.pools[conn_type] = {
                "connections": {},
                "available": deque(),
                "busy": set(),
                "config": self._get_type_specific_config(conn_type),
                "status": PoolStatus.ACTIVE
            }
            
            self.metrics[conn_type] = ConnectionMetrics()
            
            # Initialize minimum connections
            asyncio.create_task(self._warm_up_pool(conn_type))
    
    def _get_type_specific_config(self, conn_type: ConnectionType) -> Dict[str, Any]:
        """Get type-specific configuration for connection pool."""
        base_config = {
            "max_connections": self.config.max_connections,
            "min_connections": self.config.min_connections,
            "connection_timeout": self.config.connection_timeout,
            "request_timeout": self.config.request_timeout
        }
        
        # Type-specific optimizations
        if conn_type == ConnectionType.WEBSOCKET:
            base_config.update({
                "max_connections": 50,  # WebSockets are resource-intensive
                "min_connections": 5,
                "connection_timeout": 10,
                "enable_heartbeat": True,
                "heartbeat_interval": 30
            })
        elif conn_type == ConnectionType.DATABASE:
            base_config.update({
                "max_connections": 20,
                "min_connections": 5,
                "connection_timeout": 30,
                "enable_transaction_pooling": True,
                "max_transaction_time": 300
            })
        elif conn_type == ConnectionType.MODEL_CLIENT:
            base_config.update({
                "max_connections": 10,  # Model clients are expensive
                "min_connections": 2,
                "connection_timeout": 60,
                "enable_model_warmup": True,
                "max_model_requests": 5
            })
        elif conn_type == ConnectionType.REDIS:
            base_config.update({
                "max_connections": 30,
                "min_connections": 5,
                "connection_timeout": 5,
                "enable_cluster": True,
                "retry_delay": 0.1
            })
        elif conn_type == ConnectionType.MEDICAL_API:
            base_config.update({
                "max_connections": 25,
                "min_connections": 3,
                "connection_timeout": 15,
                "enable_api_key_rotation": True,
                "rate_limit": 500
            })
        
        return base_config
    
    async def _warm_up_pool(self, conn_type: ConnectionType):
        """Warm up connection pool with minimum connections."""
        pool_config = self.pools[conn_type]["config"]
        
        for i in range(pool_config["min_connections"]):
            try:
                connection = await self._create_connection(conn_type)
                if connection:
                    self._add_to_pool(conn_type, connection)
                    self.logger.debug(f"Warmed up {conn_type} connection {i+1}")
            except Exception as e:
                self.logger.error(f"Failed to warm up {conn_type} connection {i+1}: {e}")
        
        self.logger.info(f"Pool warmed up for {conn_type}")
    
    def _add_to_pool(self, conn_type: ConnectionType, connection: Any):
        """Add connection to appropriate pool."""
        connection_id = str(id(connection))
        connection_info = ConnectionInfo(
            connection_id=connection_id,
            connection_type=conn_type,
            created_at=datetime.utcnow(),
            last_used=datetime.utcnow(),
            last_health_check=datetime.utcnow(),
            priority_level=self._get_connection_priority(conn_type)
        )
        
        self.pools[conn_type]["connections"][connection_id] = {
            "connection": connection,
            "info": connection_info
        }
        
        self.idle_connections[conn_type].append(connection_id)
        self.connection_map[connection_id] = (conn_type, connection_info.connection_id)
        
        # Update metrics
        self.metrics[conn_type].total_connections += 1
        self.metrics[conn_type].idle_connections += 1
    
    async def get_connection(
        self, 
        conn_type: ConnectionType, 
        priority: int = 1,
        timeout: Optional[float] = None
    ) -> Optional[ConnectionInfo]:
        """Get connection from pool with medical priority handling."""
        
        if timeout is None:
            timeout = self.pools[conn_type]["config"]["connection_timeout"]
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for available connections
            if self.idle_connections[conn_type]:
                connection_id = self.idle_connections[conn_type].popleft()
                
                # Validate connection
                if self._validate_connection(conn_type, connection_id):
                    # Mark as busy
                    self.pools[conn_type]["busy"].add(connection_id)
                    
                    connection_info = self.pools[conn_type]["connections"][connection_id]["info"]
                    connection_info.last_used = datetime.utcnow()
                    
                    # Update metrics
                    self.metrics[conn_type].active_connections += 1
                    self.metrics[conn_type].idle_connections -= 1
                    
                    self.active_connections[connection_id] = connection_info
                    
                    self.logger.debug(
                        "Connection acquired from pool",
                        conn_type=conn_type.value,
                        connection_id=connection_id,
                        priority=priority
                    )
                    
                    return connection_info
            
            # If no connections available, check if we can create new ones
            current_connections = len(self.pools[conn_type]["connections"])
            max_allowed = self.pools[conn_type]["config"]["max_connections"]
            
            if current_connections < max_allowed:
                # Try to create new connection
                try:
                    connection = await self._create_connection(conn_type)
                    if connection:
                        self._add_to_pool(conn_type, connection)
                        # Continue loop to try getting connection again
                        continue
                except Exception as e:
                    self.logger.error(f"Failed to create connection: {e}")
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        # Timeout reached
        self.metrics[conn_type].failed_connections += 1
        self.logger.warning(
            "Connection pool timeout",
            conn_type=conn_type.value,
            timeout=timeout,
            available_connections=len(self.idle_connections[conn_type])
        )
        return None
    
    async def return_connection(self, connection_info: ConnectionInfo):
        """Return connection to pool."""
        conn_type = connection_info.connection_type
        connection_id = connection_info.connection_id
        
        if connection_id not in self.active_connections:
            self.logger.warning(f"Connection {connection_id} not in active connections")
            return
        
        # Remove from active connections
        del self.active_connections[connection_id]
        
        # Check connection health
        if connection_info.is_healthy and self._should_keep_connection(conn_type, connection_info):
            # Return to pool
            self.pools[conn_type]["busy"].discard(connection_id)
            self.idle_connections[conn_type].append(connection_id)
            
            # Update metrics
            self.metrics[conn_type].active_connections -= 1
            self.metrics[conn_type].idle_connections += 1
            
            self.logger.debug(
                "Connection returned to pool",
                conn_type=conn_type.value,
                connection_id=connection_id,
                total_requests=connection_info.request_count
            )
        else:
            # Remove unhealthy or unnecessary connections
            await self._remove_connection(conn_type, connection_id)
    
    def _validate_connection(self, conn_type: ConnectionType, connection_id: str) -> bool:
        """Validate connection health and usability."""
        if connection_id not in self.pools[conn_type]["connections"]:
            return False
        
        connection_info = self.pools[conn_type]["connections"][connection_id]["info"]
        
        # Check if connection is still healthy
        if not connection_info.is_healthy:
            return False
        
        # Check if connection is too old
        max_age = timedelta(seconds=self.config.max_idle_time)
        if datetime.utcnow() - connection_info.last_used > max_age:
            return False
        
        # Type-specific validation
        if conn_type == ConnectionType.WEBSOCKET:
            # Check WebSocket is still connected
            connection = self.pools[conn_type]["connections"][connection_id]["connection"]
            if hasattr(connection, 'readyState'):
                return connection.readyState in [1, 2]  # OPEN or CLOSING
        
        return True
    
    def _should_keep_connection(self, conn_type: ConnectionType, connection_info: ConnectionInfo) -> bool:
        """Determine if connection should be kept in pool."""
        # Medical priority: always keep critical connections
        if connection_info.priority_level >= 8:
            return True
        
        # Keep connections that have been recently used
        recent_usage = datetime.utcnow() - connection_info.last_used
        if recent_usage < timedelta(minutes=2):
            return True
        
        # Keep connections for frequently used types
        if conn_type in [ConnectionType.MODEL_CLIENT, ConnectionType.DATABASE]:
            return connection_info.request_count > 5
        
        # Remove if we have too many idle connections
        idle_count = len(self.idle_connections[conn_type])
        max_idle = self.pools[conn_type]["config"]["min_connections"] + 5
        
        return idle_count <= max_idle
    
    async def _create_connection(self, conn_type: ConnectionType) -> Optional[Any]:
        """Create new connection of specified type."""
        try:
            if conn_type == ConnectionType.WEBSOCKET:
                # Mock WebSocket connection
                return {"type": "websocket", "id": str(uuid.uuid4())}
            
            elif conn_type == ConnectionType.DATABASE:
                # Mock database connection
                return {"type": "database", "id": str(uuid.uuid4())}
            
            elif conn_type == ConnectionType.REDIS:
                # Mock Redis connection
                return {"type": "redis", "id": str(uuid.uuid4())}
            
            elif conn_type == ConnectionType.MODEL_CLIENT:
                # Mock model client connection
                return {"type": "model_client", "id": str(uuid.uuid4())}
            
            elif conn_type == ConnectionType.MEDICAL_API:
                # Mock medical API connection
                return {"type": "medical_api", "id": str(uuid.uuid4())}
            
            elif conn_type == ConnectionType.HTTP_CLIENT:
                # Mock HTTP client connection
                return {"type": "http_client", "id": str(uuid.uuid4())}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to create {conn_type} connection: {e}")
            return None
    
    async def _remove_connection(self, conn_type: ConnectionType, connection_id: str):
        """Remove connection from pool and clean up."""
        if connection_id not in self.pools[conn_type]["connections"]:
            return
        
        # Clean up resources
        connection_info = self.pools[conn_type]["connections"][connection_id]["info"]
        
        try:
            # Type-specific cleanup
            if conn_type == ConnectionType.WEBSOCKET:
                connection = self.pools[conn_type]["connections"][connection_id]["connection"]
                if hasattr(connection, 'close'):
                    connection.close()
            # Add other connection type cleanups as needed
            
        except Exception as e:
            self.logger.error(f"Error during connection cleanup: {e}")
        
        # Remove from pools
        self.pools[conn_type]["connections"].pop(connection_id, None)
        self.pools[conn_type]["busy"].discard(connection_id)
        
        # Remove from idle connections if present
        try:
            self.idle_connections[conn_type].remove(connection_id)
        except ValueError:
            pass  # Not in idle list
        
        # Remove from connection map
        self.connection_map.pop(connection_id, None)
        
        # Remove from active connections if present
        self.active_connections.pop(connection_id, None)
        
        # Update metrics
        self.metrics[conn_type].total_connections -= 1
        if connection_info.is_healthy:
            self.metrics[conn_type].failed_connections += 1
        
        self.logger.debug(
            "Connection removed from pool",
            conn_type=conn_type.value,
            connection_id=connection_id,
            reason="cleanup"
        )
    
    def _get_connection_priority(self, conn_type: ConnectionType) -> int:
        """Get priority level for connection type."""
        priorities = {
            ConnectionType.MODEL_CLIENT: 10,  # Critical for medical AI
            ConnectionType.MEDICAL_API: 9,    # Important for medical data
            ConnectionType.DATABASE: 8,       # Important for data persistence
            ConnectionType.REDIS: 7,          # Important for caching
            ConnectionType.WEBSOCKET: 6,      # Important for real-time
            ConnectionType.HTTP_CLIENT: 5     # Lower priority
        }
        return priorities.get(conn_type, 1)
    
    def _start_background_tasks(self):
        """Start background tasks for pool management."""
        # Health check task
        health_task = asyncio.create_task(self._health_check_loop())
        self.health_check_tasks["main"] = health_task
        
        # Auto-scaling task
        scaling_task = asyncio.create_task(self._auto_scaling_loop())
        self.health_check_tasks["scaling"] = scaling_task
        
        # Metrics collection task
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.health_check_tasks["metrics"] = metrics_task
        
        self.logger.info("Background tasks started for connection pool")
    
    async def _health_check_loop(self):
        """Periodic health check for all connections."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_checks(self):
        """Perform health checks on all connection types."""
        for conn_type in ConnectionType:
            try:
                pool = self.pools[conn_type]
                connections = pool["connections"]
                
                unhealthy_connections = []
                
                for connection_id, connection_data in connections.items():
                    connection_info = connection_data["info"]
                    
                    # Check if connection needs health check
                    time_since_check = datetime.utcnow() - connection_info.last_health_check
                    if time_since_check > timedelta(seconds=30):
                        is_healthy = await self._check_connection_health(conn_type, connection_id)
                        connection_info.is_healthy = is_healthy
                        connection_info.last_health_check = datetime.utcnow()
                        
                        if not is_healthy:
                            unhealthy_connections.append(connection_id)
                
                # Remove unhealthy connections
                for connection_id in unhealthy_connections:
                    await self._remove_connection(conn_type, connection_id)
                
                # Update pool status
                total_connections = len(connections)
                healthy_connections = sum(
                    1 for conn_data in connections.values()
                    if conn_data["info"].is_healthy
                )
                
                if healthy_connections == 0:
                    pool["status"] = PoolStatus.OFFLINE
                elif healthy_connections < total_connections * 0.5:
                    pool["status"] = PoolStatus.DEGRADED
                elif healthy_connections < total_connections * 0.8:
                    pool["status"] = PoolStatus.IDLE
                else:
                    pool["status"] = PoolStatus.ACTIVE
                
                # Update metrics
                self.metrics[conn_type].last_health_check = datetime.utcnow()
                
            except Exception as e:
                self.logger.error(f"Health check failed for {conn_type}: {e}")
    
    async def _check_connection_health(self, conn_type: ConnectionType, connection_id: str) -> bool:
        """Check health of specific connection."""
        try:
            if connection_id not in self.pools[conn_type]["connections"]:
                return False
            
            connection = self.pools[conn_type]["connections"][connection_id]["connection"]
            
            if conn_type == ConnectionType.WEBSOCKET:
                # Check WebSocket health
                if hasattr(connection, 'readyState'):
                    return connection.readyState == 1  # OPEN
            
            elif conn_type == ConnectionType.DATABASE:
                # Check database connection
                if hasattr(connection, 'ping'):
                    return await connection.ping()
            
            elif conn_type == ConnectionType.REDIS:
                # Check Redis connection
                if hasattr(connection, 'ping'):
                    return await connection.ping()
            
            elif conn_type == ConnectionType.MODEL_CLIENT:
                # Check model client
                if hasattr(connection, 'is_alive'):
                    return connection.is_alive()
            
            # For other types, assume healthy if no errors
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed for {conn_type} connection {connection_id}: {e}")
            return False
    
    async def _auto_scaling_loop(self):
        """Auto-scale connection pools based on load."""
        while not self._shutdown_event.is_set():
            try:
                await self._auto_scale_pools()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)
    
    async def _auto_scale_pools(self):
        """Auto-scale pools based on current load."""
        if not self.config.enable_auto_scaling:
            return
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        for conn_type in ConnectionType:
            try:
                pool = self.pools[conn_type]
                metrics = self.metrics[conn_type]
                
                # Calculate load factor
                total_connections = len(pool["connections"])
                active_connections = len(pool["busy"])
                load_factor = active_connections / max(total_connections, 1)
                
                # Scale up if high load
                if (load_factor > 0.8 or active_connections > pool["config"]["max_connections"] * 0.8) and \
                   total_connections < pool["config"]["max_connections"] * 1.5 and \
                   cpu_percent < 80 and memory_percent < 80:
                    
                    # Add connections
                    connections_to_add = min(
                        5,
                        pool["config"]["max_connections"] - total_connections
                    )
                    
                    for _ in range(connections_to_add):
                        connection = await self._create_connection(conn_type)
                        if connection:
                            self._add_to_pool(conn_type, connection)
                    
                    self.logger.info(
                        "Pool scaled up",
                        conn_type=conn_type.value,
                        connections_added=connections_to_add,
                        load_factor=load_factor
                    )
                
                # Scale down if low load
                elif load_factor < 0.3 and total_connections > pool["config"]["min_connections"]:
                    # Remove excess connections
                    connections_to_remove = min(
                        5,
                        total_connections - pool["config"]["min_connections"]
                    )
                    
                    # Remove from idle connections first
                    removed = 0
                    while removed < connections_to_remove and self.idle_connections[conn_type]:
                        connection_id = self.idle_connections[conn_type].popleft()
                        await self._remove_connection(conn_type, connection_id)
                        removed += 1
                    
                    if removed > 0:
                        self.logger.info(
                            "Pool scaled down",
                            conn_type=conn_type.value,
                            connections_removed=removed,
                            load_factor=load_factor
                        )
            
            except Exception as e:
                self.logger.error(f"Auto-scaling failed for {conn_type}: {e}")
    
    async def _metrics_collection_loop(self):
        """Collect and update metrics for all pools."""
        while not self._shutdown_event.is_set():
            try:
                await self._update_metrics()
                await asyncio.sleep(10)  # Update metrics every 10 seconds
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _update_metrics(self):
        """Update metrics for all connection types."""
        # Update system metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        for conn_type in ConnectionType:
            metrics = self.metrics[conn_type]
            pool = self.pools[conn_type]
            
            # Update connection counts
            metrics.total_connections = len(pool["connections"])
            metrics.active_connections = len(pool["busy"])
            metrics.idle_connections = len(self.idle_connections[conn_type])
            
            # Update system metrics
            metrics.memory_usage_mb = memory_info.used / 1024 / 1024
            metrics.cpu_usage_percent = cpu_percent
            
            # Calculate uptime
            if hasattr(self, '_start_time'):
                metrics.uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
    
    async def get_pool_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all connection pools."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "pools": {},
            "overall_health": "healthy",
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024
            }
        }
        
        total_healthy = 0
        total_connections = 0
        
        for conn_type in ConnectionType:
            pool = self.pools[conn_type]
            metrics = self.metrics[conn_type]
            
            healthy_connections = sum(
                1 for conn_data in pool["connections"].values()
                if conn_data["info"].is_healthy
            )
            
            total_healthy += healthy_connections
            total_connections += len(pool["connections"])
            
            status["pools"][conn_type.value] = {
                "status": pool["status"].value,
                "total_connections": len(pool["connections"]),
                "active_connections": len(pool["busy"]),
                "idle_connections": len(self.idle_connections[conn_type]),
                "healthy_connections": healthy_connections,
                "failed_connections": metrics.failed_connections,
                "avg_response_time": metrics.avg_response_time,
                "request_count": metrics.total_requests,
                "success_rate": metrics.successful_requests / max(metrics.total_requests, 1),
                "last_health_check": metrics.last_health_check.isoformat() if metrics.last_health_check else None
            }
        
        # Calculate overall health
        if total_connections == 0:
            status["overall_health"] = "offline"
        elif total_healthy / total_connections < 0.5:
            status["overall_health"] = "critical"
        elif total_healthy / total_connections < 0.8:
            status["overall_health"] = "degraded"
        else:
            status["overall_health"] = "healthy"
        
        return status
    
    async def shutdown(self):
        """Gracefully shutdown connection pool."""
        self.logger.info("Shutting down connection pool...")
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self.health_check_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.health_check_tasks.values(), return_exceptions=True)
        
        # Close all connections
        for conn_type in ConnectionType:
            connection_ids = list(self.pools[conn_type]["connections"].keys())
            for connection_id in connection_ids:
                await self._remove_connection(conn_type, connection_id)
        
        self.logger.info("Connection pool shutdown complete")


# Global connection pool instance
connection_pool = MedicalConnectionPool()


# Medical priority connection manager
class MedicalPriorityConnectionManager:
    """High-priority connection manager for medical operations."""
    
    def __init__(self):
        self.pool = connection_pool
        self.medical_queue: asyncio.Queue = asyncio.Queue()
        self.priority_weights = {
            "emergency": 10,
            "critical": 9,
            "urgent": 8,
            "high": 7,
            "normal": 5,
            "low": 3
        }
    
    async def get_medical_connection(
        self,
        conn_type: ConnectionType,
        priority: str = "normal",
        timeout: float = 30.0
    ) -> Optional[ConnectionInfo]:
        """Get connection with medical priority handling."""
        priority_level = self.priority_weights.get(priority, 5)
        
        # For emergency cases, use shorter timeout and higher priority
        if priority == "emergency":
            timeout = min(timeout, 10.0)
            priority_level = 10
        
        connection_info = await self.pool.get_connection(
            conn_type=conn_type,
            priority=priority_level,
            timeout=timeout
        )
        
        if connection_info:
            connection_info.priority_level = priority_level
            connection_info.metadata["medical_priority"] = priority
            connection_info.metadata["requested_at"] = datetime.utcnow().isoformat()
        
        return connection_info
    
    async def record_medical_request(
        self,
        connection_info: ConnectionInfo,
        response_time: float,
        success: bool
    ):
        """Record medical request metrics."""
        connection_info.request_count += 1
        connection_info.total_response_time += response_time
        
        if not success:
            connection_info.error_count += 1
        
        # Update pool metrics
        conn_type = connection_info.connection_type
        pool_metrics = self.pool.metrics[conn_type]
        
        pool_metrics.total_requests += 1
        pool_metrics.total_response_time += response_time
        pool_metrics.avg_response_time = pool_metrics.total_response_time / pool_metrics.total_requests
        
        if success:
            pool_metrics.successful_requests += 1
        else:
            pool_metrics.failed_requests += 1
        
        # Log medical request
        logger.info(
            "Medical request completed",
            connection_type=conn_type.value,
            connection_id=connection_info.connection_id,
            priority=connection_info.metadata.get("medical_priority"),
            response_time=response_time,
            success=success,
            total_requests=connection_info.request_count
        )


# Global medical priority manager
medical_connection_manager = MedicalPriorityConnectionManager()


# Export classes and functions
__all__ = [
    "connection_pool",
    "medical_connection_manager",
    "MedicalConnectionPool",
    "MedicalPriorityConnectionManager",
    "ConnectionType",
    "PoolStatus",
    "ConnectionMetrics",
    "PoolConfig",
    "ConnectionInfo"
]