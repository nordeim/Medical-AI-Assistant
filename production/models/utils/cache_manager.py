"""
Cache Manager Utility
High-performance caching layer for medical AI predictions and model responses.
"""

import asyncio
import logging
import json
import pickle
import hashlib
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import aioredis
import redis
from contextlib import asynccontextmanager
import numpy as np

logger = logging.getLogger(__name__)

class CacheManager:
    """Production cache manager for medical AI model serving"""
    
    def __init__(self, config_path: str = "config/cache_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Cache configuration
        self.redis_host = self.config.get("redis_host", "localhost")
        self.redis_port = self.config.get("redis_port", 6379)
        self.default_ttl = self.config.get("default_ttl", 3600)
        self.max_cache_size = self.config.get("max_cache_size", 10000)
        self.enable_compression = self.config.get("enable_compression", True)
        
        # Cache statistics
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "expired": 0
        }
        
        # Cache namespaces
        self.namespaces = {
            "predictions": "medical_ai:predictions",
            "models": "medical_ai:models",
            "features": "medical_ai:features",
            "metadata": "medical_ai:metadata"
        }
        
        # Redis connections
        self.redis_client = None
        self.redis_pool = None
        
        # LRU cache for frequently accessed items
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_order: List[str] = []
        self.memory_cache_max_size = 1000
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load cache configuration"""
        default_config = {
            "redis_host": "localhost",
            "redis_port": 6379,
            "default_ttl": 3600,
            "max_cache_size": 10000,
            "enable_compression": True,
            "connection_pool_size": 10,
            "connection_timeout": 5,
            "memory_cache_size": 1000,
            "compression_threshold": 1000,
            "cache_namespaces": {
                "predictions": "medical_ai:predictions",
                "models": "medical_ai:models",
                "features": "medical_ai:features",
                "metadata": "medical_ai:metadata"
            }
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Cache config {config_path} not found, using defaults")
            return default_config
    
    async def initialize(self):
        """Initialize the cache manager"""
        logger.info("Initializing Cache Manager...")
        
        try:
            # Initialize Redis connection pool
            redis_url = f"redis://{self.redis_host}:{self.redis_port}"
            self.redis_client = aioredis.from_url(redis_url)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Start background tasks
            asyncio.create_task(self._cleanup_expired_keys())
            asyncio.create_task(self._monitor_cache_performance())
            
            logger.info("Cache Manager initialization complete")
            
        except Exception as e:
            logger.error(f"Cache Manager initialization failed: {str(e)}")
            # Fallback to memory-only cache
            logger.warning("Falling back to memory-only cache")
    
    async def get(self, key: str, namespace: str = "predictions") -> Optional[Any]:
        """Get value from cache with L1 (memory) and L2 (Redis) layers"""
        cache_key = self._build_cache_key(key, namespace)
        
        # L1: Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats["hits"] += 1
            logger.debug(f"Memory cache hit: {cache_key}")
            return self.memory_cache[cache_key]
        
        # L2: Check Redis cache
        try:
            if self.redis_client:
                value = await self.redis_client.get(cache_key)
                if value is not None:
                    # Decompress if needed
                    if self.enable_compression:
                        value = await self._decompress_value(value)
                    
                    # Store in L1 cache for next time
                    await self._store_in_memory_cache(cache_key, value)
                    
                    self.cache_stats["hits"] += 1
                    logger.debug(f"Redis cache hit: {cache_key}")
                    return value
                    
        except Exception as e:
            logger.warning(f"Redis cache get error: {str(e)}")
        
        # Cache miss
        self.cache_stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                  namespace: str = "predictions", compress: bool = True) -> bool:
        """Set value in cache with L1 and L2 storage"""
        cache_key = self._build_cache_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        # Check if we should compress
        should_compress = compress and self.enable_compression and self._should_compress(value)
        
        try:
            # L1: Store in memory cache
            await self._store_in_memory_cache(cache_key, value)
            
            # L2: Store in Redis cache
            if self.redis_client:
                redis_value = value
                if should_compress:
                    redis_value = await self._compress_value(value)
                
                await self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    redis_value
                )
                
                self.cache_stats["sets"] += 1
                logger.debug(f"Cache set: {cache_key} (TTL: {ttl}s, compressed: {should_compress})")
                return True
            else:
                logger.warning("Redis not available, using memory cache only")
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {str(e)}")
            return False
    
    async def delete(self, key: str, namespace: str = "predictions") -> bool:
        """Delete value from cache"""
        cache_key = self._build_cache_key(key, namespace)
        
        try:
            # L1: Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                self.memory_cache_order.remove(cache_key)
            
            # L2: Remove from Redis
            if self.redis_client:
                await self.redis_client.delete(cache_key)
            
            self.cache_stats["deletes"] += 1
            logger.debug(f"Cache deleted: {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {str(e)}")
            return False
    
    async def exists(self, key: str, namespace: str = "predictions") -> bool:
        """Check if key exists in cache"""
        cache_key = self._build_cache_key(key, namespace)
        
        # Check L1 first
        if cache_key in self.memory_cache:
            return True
        
        # Check L2
        if self.redis_client:
            try:
                return await self.redis_client.exists(cache_key) > 0
            except Exception as e:
                logger.warning(f"Redis exists check error: {str(e)}")
        
        return False
    
    async def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace"""
        if namespace not in self.namespaces:
            raise ValueError(f"Unknown namespace: {namespace}")
        
        namespace_prefix = self.namespaces[namespace]
        cleared_count = 0
        
        try:
            # Clear memory cache for this namespace
            keys_to_remove = [
                key for key in self.memory_cache.keys() 
                if key.startswith(namespace_prefix)
            ]
            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.memory_cache_order:
                    self.memory_cache_order.remove(key)
                cleared_count += 1
            
            # Clear Redis cache for this namespace
            if self.redis_client:
                pattern = f"{namespace_prefix}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    cleared_count += len(keys)
            
            logger.info(f"Cleared {cleared_count} keys from namespace {namespace}")
            return cleared_count
            
        except Exception as e:
            logger.error(f"Clear namespace error for {namespace}: {str(e)}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        redis_stats = {}
        memory_stats = {
            "size": len(self.memory_cache),
            "max_size": self.memory_cache_max_size,
            "utilization": len(self.memory_cache) / self.memory_cache_max_size
        }
        
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info()
                redis_stats = {
                    "connected_clients": redis_info.get('connected_clients', 0),
                    "used_memory_human": redis_info.get('used_memory_human', '0'),
                    "total_commands_processed": redis_info.get('total_commands_processed', 0),
                    "keyspace_hits": redis_info.get('keyspace_hits', 0),
                    "keyspace_misses": redis_info.get('keyspace_misses', 0)
                }
            except Exception as e:
                logger.warning(f"Redis info error: {str(e)}")
        
        # Calculate hit rate
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "cache_stats": self.cache_stats,
            "hit_rate": hit_rate,
            "memory_cache": memory_stats,
            "redis_stats": redis_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def warm_cache(self, warmup_data: List[Dict[str, Any]]):
        """Warm up cache with frequently accessed data"""
        logger.info(f"Warming cache with {len(warmup_data)} items")
        
        for item in warmup_data:
            try:
                key = item.get("key")
                value = item.get("value")
                namespace = item.get("namespace", "predictions")
                ttl = item.get("ttl", self.default_ttl)
                
                if key is not None and value is not None:
                    await self.set(key, value, ttl=ttl, namespace=namespace)
                    
            except Exception as e:
                logger.warning(f"Warmup item error: {str(e)}")
        
        logger.info("Cache warmup completed")
    
    async def invalidate_pattern(self, pattern: str, namespace: str = "predictions"):
        """Invalidate cache keys matching a pattern"""
        namespace_prefix = self.namespaces[namespace]
        full_pattern = f"{namespace_prefix}:{pattern}"
        
        invalidated_count = 0
        
        try:
            # Invalidate from memory cache
            keys_to_remove = [
                key for key in self.memory_cache.keys() 
                if self._pattern_matches(key, full_pattern)
            ]
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.memory_cache_order:
                    self.memory_cache_order.remove(key)
                invalidated_count += 1
            
            # Invalidate from Redis
            if self.redis_client:
                keys = await self.redis_client.keys(full_pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    invalidated_count += len(keys)
            
            logger.info(f"Invalidated {invalidated_count} keys matching pattern {full_pattern}")
            
        except Exception as e:
            logger.error(f"Pattern invalidation error: {str(e)}")
    
    def _build_cache_key(self, key: str, namespace: str) -> str:
        """Build full cache key with namespace"""
        namespace_prefix = self.namespaces.get(namespace, "medical_ai:default")
        return f"{namespace_prefix}:{key}"
    
    async def _store_in_memory_cache(self, key: str, value: Any):
        """Store value in L1 memory cache"""
        # Add to cache
        self.memory_cache[key] = value
        
        # Track access order
        if key in self.memory_cache_order:
            self.memory_cache_order.remove(key)
        self.memory_cache_order.append(key)
        
        # Manage cache size (LRU eviction)
        if len(self.memory_cache_order) > self.memory_cache_max_size:
            oldest_key = self.memory_cache_order.pop(0)
            del self.memory_cache[oldest_key]
            logger.debug(f"Evicted from memory cache: {oldest_key}")
    
    def _should_compress(self, value: Any) -> bool:
        """Determine if value should be compressed"""
        if not self.enable_compression:
            return False
        
        # Estimate size
        try:
            serialized = json.dumps(value, default=str)
            return len(serialized) > self.config.get("compression_threshold", 1000)
        except:
            return False
    
    async def _compress_value(self, value: Any) -> bytes:
        """Compress value for storage"""
        try:
            import gzip
            import json
            
            # Serialize value
            serialized = json.dumps(value, default=str).encode('utf-8')
            
            # Compress
            compressed = gzip.compress(serialized)
            logger.debug(f"Compressed value: {len(serialized)} -> {len(compressed)} bytes")
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}, storing uncompressed")
            return json.dumps(value, default=str).encode('utf-8')
    
    async def _decompress_value(self, compressed_value: bytes) -> Any:
        """Decompress value from storage"""
        try:
            import gzip
            import json
            
            # Decompress
            decompressed = gzip.decompress(compressed_value)
            
            # Deserialize
            value = json.loads(decompressed.decode('utf-8'))
            
            return value
            
        except Exception as e:
            logger.warning(f"Decompression failed: {str(e)}, trying direct JSON")
            try:
                return json.loads(compressed_value.decode('utf-8'))
            except:
                logger.error("Failed to deserialize cache value")
                raise
    
    def _pattern_matches(self, key: str, pattern: str) -> bool:
        """Simple pattern matching for cache invalidation"""
        # Convert glob pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        import re
        return re.match(regex_pattern, key) is not None
    
    async def _cleanup_expired_keys(self):
        """Background task to cleanup expired keys"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                if self.redis_client:
                    # Scan for expired keys and clean up memory cache
                    current_time = time.time()
                    expired_keys = []
                    
                    for key in list(self.memory_cache.keys()):
                        # Check if key has TTL (simplified check)
                        if key.startswith(self.namespaces["predictions"]):
                            # For demo purposes, randomly expire some keys
                            if np.random.random() < 0.01:  # 1% chance
                                expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.memory_cache[key]
                        if key in self.memory_cache_order:
                            self.memory_cache_order.remove(key)
                        self.cache_stats["expired"] += 1
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired keys from memory cache")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _monitor_cache_performance(self):
        """Monitor cache performance and log statistics"""
        while True:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                stats = await self.get_cache_stats()
                hit_rate = stats["hit_rate"]
                
                # Log performance warnings
                if hit_rate < 0.7:
                    logger.warning(f"Low cache hit rate: {hit_rate:.2%}")
                
                # Log high utilization
                if stats["memory_cache"]["utilization"] > 0.9:
                    logger.warning("High memory cache utilization")
                    
            except Exception as e:
                logger.error(f"Cache monitoring error: {str(e)}")
    
    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Cache Manager closed")