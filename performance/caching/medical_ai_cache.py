"""
Comprehensive Caching Strategy for Medical AI System
Implements Redis caching and application-level caching for improved response times
"""

import asyncio
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable
from functools import wraps
import redis.asyncio as redis
from datetime import datetime, timedelta
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"      # In-memory cache (fastest)
    L2_REDIS = "l2_redis"        # Redis cache (fast)
    L3_DATABASE = "l3_database"  # Database cache (slowest)

class MedicalAICache:
    """
    Multi-level caching system optimized for medical AI workloads
    - L1: In-memory cache for frequently accessed data
    - L2: Redis cache for shared data across services
    - L3: Database caching for persistence
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.l1_cache = {}  # In-memory cache
        self.l1_max_size = 1000
        self.cache_stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'l3_hits': 0,
            'l3_misses': 0
        }
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            await self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, namespace: str, *args, **kwargs) -> str:
        """Generate deterministic cache key"""
        key_data = f"{namespace}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for caching"""
        try:
            if isinstance(data, (dict, list, str, int, float, bool)):
                return json.dumps(data).encode()
            else:
                return pickle.dumps(data)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return json.dumps(str(data)).encode()
    
    def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize cached data"""
        try:
            # Try JSON first
            try:
                return json.loads(data.decode())
            except json.JSONDecodeError:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    async def get(self, key: str, cache_level: CacheLevel = CacheLevel.L2_REDIS) -> Optional[Any]:
        """Get data from cache with fallback between levels"""
        
        # Try L1 cache first
        if cache_level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]:
            if key in self.l1_cache:
                self.cache_stats['l1_hits'] += 1
                return self.l1_cache[key]['value']
            self.cache_stats['l1_misses'] += 1
        
        # Try L2 Redis cache
        if cache_level in [CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE] and self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    self.cache_stats['l2_hits'] += 1
                    value = self._deserialize_data(data)
                    
                    # Also store in L1 for future hits
                    await self._store_l1(key, value)
                    return value
                self.cache_stats['l2_misses'] += 1
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # L3 would typically be database query
        self.cache_stats['l3_misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, 
                  ttl: int = 3600, cache_level: CacheLevel = CacheLevel.L2_REDIS):
        """Set data in cache across multiple levels"""
        
        # Store in all appropriate cache levels
        if cache_level in [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE]:
            await self._store_l1(key, value, ttl)
        
        if cache_level in [CacheLevel.L2_REDIS, CacheLevel.L3_DATABASE] and self.redis_client:
            await self._store_l2(key, value, ttl)
    
    async def _store_l1(self, key: str, value: Any, ttl: int = 3600):
        """Store in L1 in-memory cache"""
        # Implement LRU eviction if cache is full
        if len(self.l1_cache) >= self.l1_max_size:
            # Remove oldest entry
            oldest_key = min(self.l1_cache.keys(), 
                           key=lambda k: self.l1_cache[k]['timestamp'])
            del self.l1_cache[oldest_key]
        
        self.l1_cache[key] = {
            'value': value,
            'timestamp': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
    
    async def _store_l2(self, key: str, value: Any, ttl: int = 3600):
        """Store in L2 Redis cache"""
        try:
            serialized_data = self._serialize_data(value)
            await self.redis_client.setex(key, ttl, serialized_data)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if self.redis_client:
            try:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache entries for pattern: {pattern}")
            except Exception as e:
                logger.error(f"Cache invalidation error: {e}")
    
    def cache_decorator(self, ttl: int = 3600, namespace: str = "default"):
        """
        Decorator for caching function results
        Use @cache_decorator(ttl=1800, namespace="patient_data")
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_cache_key(namespace, *args, **kwargs)
                
                # Try to get from cache first
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_result
                
                # Execute function and cache result
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                logger.debug(f"Cache miss for {func.__name__}, cached result")
                
                return result
            return wrapper
        return decorator
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics"""
        total_requests = sum(self.cache_stats.values())
        if total_requests > 0:
            hit_rate = (self.cache_stats['l1_hits'] + self.cache_stats['l2_hits'] + 
                       self.cache_stats['l3_hits']) / total_requests
        else:
            hit_rate = 0
        
        return {
            **self.cache_stats,
            'hit_rate': round(hit_rate, 3),
            'l1_size': len(self.l1_cache)
        }


class PatientDataCache:
    """
    Specialized caching for patient data with medical compliance considerations
    """
    
    def __init__(self, cache: MedicalAICache):
        self.cache = cache
        self.patient_ttl = 1800  # 30 minutes for patient data
        self.clinical_ttl = 900   # 15 minutes for clinical data
        self.audit_ttl = 7200    # 2 hours for audit logs
    
    async def cache_patient_record(self, patient_data: Dict):
        """Cache patient record with appropriate TTL"""
        patient_id = patient_data.get('patient_id')
        if patient_id:
            key = f"patient:{patient_id}"
            await self.cache.set(key, patient_data, self.patient_ttl)
    
    async def get_patient_record(self, patient_id: int) -> Optional[Dict]:
        """Get cached patient record"""
        key = f"patient:{patient_id}"
        return await self.cache.get(key)
    
    async def cache_clinical_data(self, patient_id: int, clinical_data: Dict):
        """Cache clinical data results"""
        key = f"clinical:{patient_id}:{clinical_data.get('data_type')}"
        await self.cache.set(key, clinical_data, self.clinical_ttl)
    
    async def cache_ai_inference(self, prompt_hash: str, result: Dict):
        """Cache AI inference results"""
        key = f"ai_inference:{prompt_hash}"
        await self.cache.set(key, result, ttl=3600)  # 1 hour for AI results


class ModelInferenceCache:
    """
    Caching system for model inference results
    Optimized for medical AI model outputs
    """
    
    def __init__(self, cache: MedicalAICache):
        self.cache = cache
        self.similarity_threshold = 0.95
    
    async def cache_inference_result(self, model_name: str, input_data: Dict, result: Dict):
        """Cache model inference result"""
        # Create semantic cache key based on input similarity
        input_hash = self._create_semantic_hash(input_data)
        key = f"inference:{model_name}:{input_hash}"
        await self.cache.set(key, result, ttl=7200)  # 2 hours for AI results
    
    async def get_inference_result(self, model_name: str, input_data: Dict) -> Optional[Dict]:
        """Get cached inference result"""
        input_hash = self._create_semantic_hash(input_data)
        key = f"inference:{model_name}:{input_hash}"
        return await self.cache.get(key)
    
    def _create_semantic_hash(self, data: Dict) -> str:
        """Create semantic hash for similar inputs"""
        # Normalize data for consistent hashing
        normalized = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


async def main():
    """Example usage of caching system"""
    
    # Initialize cache
    cache = MedicalAICache(redis_url="redis://localhost:6379")
    await cache.initialize()
    
    # Use patient data cache
    patient_cache = PatientDataCache(cache)
    
    # Example caching decorator
    @cache.cache_decorator(ttl=1800, namespace="medical_queries")
    async def get_patient_diagnoses(patient_id: int):
        """Simulated database query"""
        await asyncio.sleep(0.1)  # Simulate DB delay
        return [{"diagnosis": "Hypertension", "date": "2024-01-15"}]
    
    # Test caching
    result = await get_patient_diagnoses(123)
    cached_result = await get_patient_diagnoses(123)  # Should be cached
    
    # Print cache statistics
    stats = cache.get_cache_stats()
    print(f"Cache Statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(main())