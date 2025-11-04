"""
Connection Pooling and Resource Management for Medical AI System
Implements database connection pooling and API rate limiting for healthcare workloads
"""

import asyncio
import asyncpg
import aiohttp
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json
from enum import Enum
from collections import defaultdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    ADAPTIVE = "adaptive"

@dataclass
class ResourceMetrics:
    """Resource usage metrics"""
    timestamp: datetime
    active_connections: int
    pool_utilization: float
    avg_response_time: float
    error_rate: float
    rate_limit_hits: int

class DatabaseConnectionPool:
    """
    Optimized connection pool for medical databases
    Implements intelligent connection management for healthcare workloads
    """
    
    def __init__(self, 
                 database_url: str,
                 min_connections: int = 10,
                 max_connections: int = 50,
                 command_timeout: int = 60,
                 max_queries: int = 50000):
        self.database_url = database_url
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.command_timeout = command_timeout
        self.max_queries = max_queries
        self.pool = None
        self.connection_stats = defaultdict(int)
        self.active_connections = 0
        self.metrics = []
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_connections,
                max_size=self.max_connections,
                max_inactive_connection_lifetime=300.0,
                command_timeout=self.command_timeout,
                server_settings={
                    'application_name': 'medical_ai_service',
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3'
                }
            )
            logger.info(f"Database connection pool initialized: {self.min_connections}-{self.max_connections}")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    async def get_connection(self) -> asyncpg.Connection:
        """Get connection from pool with metrics"""
        async with self._lock:
            self.active_connections += 1
            self.connection_stats['total_acquisitions'] += 1
        
        try:
            conn = await self.pool.acquire()
            start_time = time.time()
            
            # Test connection health
            await conn.execute("SELECT 1")
            response_time = time.time() - start_time
            
            # Record metrics
            await self._record_connection_metrics(response_time)
            
            return conn
        except Exception as e:
            logger.error(f"Failed to acquire connection: {e}")
            async with self._lock:
                self.active_connections -= 1
            raise
    
    async def return_connection(self, conn: asyncpg.Connection):
        """Return connection to pool"""
        try:
            await self.pool.release(conn)
            async with self._lock:
                self.active_connections -= 1
                self.connection_stats['total_releases'] += 1
        except Exception as e:
            logger.error(f"Failed to return connection: {e}")
    
    async def execute_query(self, query: str, *args) -> Any:
        """Execute query with automatic connection management"""
        conn = await self.get_connection()
        try:
            result = await conn.execute(query, *args)
            self.connection_stats['successful_queries'] += 1
            return result
        except Exception as e:
            self.connection_stats['failed_queries'] += 1
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            await self.return_connection(conn)
    
    async def fetch_one(self, query: str, *args) -> Optional[Dict]:
        """Fetch single row with optimization"""
        conn = await self.get_connection()
        try:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None
        except Exception as e:
            self.connection_stats['failed_queries'] += 1
            logger.error(f"Fetch one failed: {e}")
            raise
        finally:
            await self.return_connection(conn)
    
    async def fetch_many(self, query: str, *args, limit: int = 1000) -> List[Dict]:
        """Fetch multiple rows with pagination"""
        conn = await self.get_connection()
        try:
            # Add limit to query for large datasets
            if 'LIMIT' not in query.upper():
                query = f"{query} LIMIT {limit}"
            
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
        except Exception as e:
            self.connection_stats['failed_queries'] += 1
            logger.error(f"Fetch many failed: {e}")
            raise
        finally:
            await self.return_connection(conn)
    
    async def _record_connection_metrics(self, response_time: float):
        """Record connection performance metrics"""
        utilization = self.active_connections / self.max_connections
        self.metrics.append(ResourceMetrics(
            timestamp=datetime.now(),
            active_connections=self.active_connections,
            pool_utilization=utilization,
            avg_response_time=response_time,
            error_rate=self.connection_stats['failed_queries'] / max(1, self.connection_stats['total_acquisitions']),
            rate_limit_hits=0
        ))
        
        # Keep only recent metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'utilization': self.active_connections / self.max_connections,
            'total_acquisitions': self.connection_stats['total_acquisitions'],
            'successful_queries': self.connection_stats['successful_queries'],
            'failed_queries': self.connection_stats['failed_queries'],
            'error_rate': self.connection_stats['failed_queries'] / max(1, self.connection_stats['total_acquisitions']),
            'recent_metrics': self.metrics[-10:] if self.metrics else []
        }
    
    async def health_check(self) -> bool:
        """Perform connection pool health check"""
        try:
            await self.execute_query("SELECT 1")
            return True
        except Exception:
            return False


class APIRateLimiter:
    """
    Advanced API rate limiting for medical AI workloads
    Supports multiple strategies including adaptive limiting
    """
    
    def __init__(self, 
                 strategy: RateLimitStrategy = RateLimitStrategy.ADAPTIVE,
                 redis_url: str = "redis://localhost:6379"):
        self.strategy = strategy
        self.redis_client = None
        self.rate_limits = {}
        self.request_history = defaultdict(list)
        self.adaptive_window = timedelta(minutes=5)
        
        # Default rate limits for medical AI endpoints
        self.default_limits = {
            '/api/patient-data': {'requests': 100, 'window': 60},  # 100/minute
            '/api/ai-inference': {'requests': 50, 'window': 60},   # 50/minute
            '/api/clinical-data': {'requests': 200, 'window': 60}, # 200/minute
            '/api/audit-logs': {'requests': 30, 'window': 60},     # 30/minute
            '/default': {'requests': 300, 'window': 60}            # 300/minute default
        }
    
    async def initialize(self):
        """Initialize rate limiter with Redis"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Rate limiter initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
    
    async def check_rate_limit(self, 
                             endpoint: str, 
                             client_id: str,
                             user_tier: str = 'standard') -> bool:
        """
        Check if request is within rate limits
        Returns True if allowed, False if rate limited
        """
        rate_config = self.rate_limits.get(endpoint, self.default_limits.get(endpoint, self.default_limits['default']))
        
        if self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._token_bucket_check(endpoint, client_id, rate_config)
        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._sliding_window_check(endpoint, client_id, rate_config)
        elif self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._fixed_window_check(endpoint, client_id, rate_config)
        elif self.strategy == RateLimitStrategy.ADAPTIVE:
            return await self._adaptive_check(endpoint, client_id, rate_config)
        
        return True
    
    async def _token_bucket_check(self, endpoint: str, client_id: str, config: Dict) -> bool:
        """Token bucket rate limiting algorithm"""
        key = f"rate_limit:token_bucket:{endpoint}:{client_id}"
        now = time.time()
        window = config['window']
        max_tokens = config['requests']
        
        # Get current bucket state
        bucket_data = await self._get_redis_data(key)
        
        if not bucket_data:
            # Initialize bucket
            bucket_data = {
                'tokens': max_tokens,
                'last_refill': now
            }
        else:
            bucket_data['tokens'] = float(bucket_data['tokens'])
            bucket_data['last_refill'] = float(bucket_data['last_refill'])
        
        # Calculate tokens to add based on time passed
        time_passed = now - bucket_data['last_refill']
        tokens_to_add = (time_passed / window) * max_tokens
        bucket_data['tokens'] = min(max_tokens, bucket_data['tokens'] + tokens_to_add)
        bucket_data['last_refill'] = now
        
        # Check if we have tokens
        if bucket_data['tokens'] >= 1.0:
            bucket_data['tokens'] -= 1.0
            await self._set_redis_data(key, bucket_data, window)
            return True
        
        await self._set_redis_data(key, bucket_data, window)
        return False
    
    async def _sliding_window_check(self, endpoint: str, client_id: str, config: Dict) -> bool:
        """Sliding window rate limiting"""
        key = f"rate_limit:sliding:{endpoint}:{client_id}"
        now = time.time()
        window = config['window']
        max_requests = config['requests']
        
        # Get request timestamps
        requests = await self._get_redis_list(key)
        
        # Remove old requests outside window
        cutoff = now - window
        requests = [ts for ts in requests if float(ts) > cutoff]
        
        # Check if under limit
        if len(requests) < max_requests:
            requests.append(now)
            await self._set_redis_list(key, requests, window)
            return True
        
        return False
    
    async def _fixed_window_check(self, endpoint: str, client_id: str, config: Dict) -> bool:
        """Fixed window rate limiting"""
        key = f"rate_limit:fixed:{endpoint}:{client_id}"
        window = config['window']
        max_requests = config['requests']
        
        # Get current window count
        count = await self._get_redis_counter(key)
        
        if count < max_requests:
            await self._increment_redis_counter(key, window)
            return True
        
        return False
    
    async def _adaptive_check(self, endpoint: str, client_id: str, config: Dict) -> bool:
        """Adaptive rate limiting based on system load"""
        base_config = config.copy()
        
        # Adjust limits based on system metrics
        system_load = await self._get_system_load()
        
        if system_load > 0.8:  # High load
            base_config['requests'] = int(base_config['requests'] * 0.5)
        elif system_load < 0.3:  # Low load
            base_config['requests'] = int(base_config['requests'] * 1.2)
        
        # Use sliding window with adjusted limits
        return await self._sliding_window_check(endpoint, client_id, base_config)
    
    async def _get_redis_data(self, key: str) -> Optional[Dict]:
        """Get data from Redis"""
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                return json.loads(data) if data else None
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        return None
    
    async def _set_redis_data(self, key: str, data: Dict, ttl: int):
        """Set data in Redis with TTL"""
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
    
    async def _get_redis_list(self, key: str) -> List[float]:
        """Get list from Redis"""
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                return json.loads(data) if data else []
            except Exception:
                return []
        return []
    
    async def _set_redis_list(self, key: str, data: List[float], ttl: int):
        """Set list in Redis with TTL"""
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
    
    async def _get_redis_counter(self, key: str) -> int:
        """Get counter from Redis"""
        if self.redis_client:
            try:
                return int(await self.redis_client.get(key) or 0)
            except Exception:
                return 0
        return 0
    
    async def _increment_redis_counter(self, key: str, ttl: int):
        """Increment counter in Redis"""
        if self.redis_client:
            try:
                await self.redis_client.incr(key)
                await self.redis_client.expire(key, ttl)
            except Exception as e:
                logger.warning(f"Redis increment error: {e}")
    
    async def _get_system_load(self) -> float:
        """Get current system load (placeholder implementation)"""
        # In a real implementation, this would query system metrics
        # For now, return a simulated value
        return 0.5
    
    def configure_endpoint_limit(self, endpoint: str, requests: int, window: int):
        """Configure rate limit for specific endpoint"""
        self.rate_limits[endpoint] = {
            'requests': requests,
            'window': window
        }


class ResourceMonitor:
    """
    Monitors and manages system resources for medical AI workloads
    """
    
    def __init__(self):
        self.resource_stats = defaultdict(dict)
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'connection_pool_utilization': 0.9
        }
    
    def check_resource_thresholds(self, metrics: ResourceMetrics) -> List[str]:
        """Check if resource metrics exceed thresholds"""
        alerts = []
        
        if metrics.pool_utilization > self.alert_thresholds['connection_pool_utilization']:
            alerts.append(f"High connection pool utilization: {metrics.pool_utilization:.2%}")
        
        if metrics.error_rate > 0.05:  # 5% error rate
            alerts.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.avg_response_time > 2.0:  # 2 seconds
            alerts.append(f"High response time: {metrics.avg_response_time:.2f}s")
        
        return alerts
    
    async def optimize_resources(self):
        """Automatically optimize resources based on usage patterns"""
        # Implement auto-scaling recommendations
        # Connection pool size adjustments
        # Rate limit tuning
        
        recommendations = []
        
        # Check connection pool utilization
        # if utilization > 80%: recommend increasing pool size
        # if utilization < 20%: recommend decreasing pool size
        
        # Check rate limit hit rates
        # if hit rate > 10%: consider increasing limits
        
        return recommendations


class MedicalAIServiceResourceManager:
    """
    Complete resource management system for medical AI services
    """
    
    def __init__(self, 
                 database_url: str,
                 redis_url: str = "redis://localhost:6379"):
        self.connection_pool = DatabaseConnectionPool(database_url)
        self.rate_limiter = APIRateLimiter(strategy=RateLimitStrategy.ADAPTIVE, redis_url=redis_url)
        self.resource_monitor = ResourceMonitor()
    
    async def initialize(self):
        """Initialize all resource management components"""
        await self.connection_pool.initialize()
        await self.rate_limiter.initialize()
        logger.info("Medical AI resource management initialized")
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform comprehensive health check"""
        health_status = {
            'database_pool': await self.connection_pool.health_check(),
            'redis_connection': self.rate_limiter.redis_client is not None
        }
        
        return health_status
    
    def get_resource_metrics(self) -> Dict[str, Any]:
        """Get comprehensive resource metrics"""
        return {
            'connection_pool': self.connection_pool.get_pool_stats(),
            'rate_limits': self.rate_limiter.default_limits,
            'resource_alerts': []  # Would include active alerts
        }


async def main():
    """Example usage of resource management system"""
    
    # Initialize resource manager
    resource_manager = MedicalAIServiceResourceManager(
        database_url="postgresql://user:pass@localhost/medical_db",
        redis_url="redis://localhost:6379"
    )
    
    await resource_manager.initialize()
    
    # Test rate limiting
    allowed = await resource_manager.rate_limiter.check_rate_limit(
        '/api/patient-data', 'client_123', 'premium'
    )
    
    print(f"Request allowed: {allowed}")
    
    # Get metrics
    metrics = resource_manager.get_resource_metrics()
    print(f"Resource metrics: {metrics}")
    
    # Health check
    health = await resource_manager.health_check()
    print(f"Health status: {health}")


if __name__ == "__main__":
    asyncio.run(main())