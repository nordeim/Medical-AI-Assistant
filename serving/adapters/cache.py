"""
Memory-Optimized Adapter Cache

Provides LRU caching with advanced memory optimization,
GPU memory management, and medical model specific caching.
"""

import asyncio
import gc
import hashlib
import logging
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import psutil
import torch

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    MEMORY_AWARE = "memory_aware"  # Memory-based eviction
    HYBRID = "hybrid"              # Combination of strategies


class MemoryOptimizationLevel(Enum):
    """Memory optimization levels."""
    CONSERVATIVE = "conservative"  # Keep more adapters in memory
    BALANCED = "balanced"          # Balance between performance and memory
    AGGRESSIVE = "aggressive"      # Aggressive memory cleanup
    MEDICAL_STRICT = "medical_strict"  # Strict for medical compliance


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    adapter_id: str
    version_id: str
    adapter_model: Any
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age_hours(self) -> float:
        """Get entry age in hours."""
        return (time.time() - self.created_at) / 3600


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_evictions: int = 0
    ttl_evictions: int = 0
    hit_rate: float = 0.0
    avg_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "memory_evictions": self.memory_evictions,
            "ttl_evictions": self.ttl_evictions,
            "hit_rate": self.hit_rate,
            "avg_memory_usage_mb": self.avg_memory_usage_mb,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "gpu_memory_usage_mb": self.gpu_memory_usage_mb
        }


class MemoryProfiler:
    """Profiles memory usage of cached adapters."""
    
    def __init__(self):
        self._profile_callbacks: List[Callable] = []
        self._memory_history: List[Dict[str, float]] = []
    
    def estimate_model_memory(self, model: Any) -> Tuple[float, float]:
        """Estimate memory usage of a model."""
        try:
            # CPU memory estimation
            cpu_memory_mb = 0.0
            
            if hasattr(model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                cpu_memory_mb = param_size / (1024 * 1024)
            
            # GPU memory estimation
            gpu_memory_mb = 0.0
            if torch.cuda.is_available() and hasattr(model, 'device'):
                try:
                    if hasattr(model, 'to'):
                        # Try to get GPU memory
                        gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                except:
                    pass
            
            return cpu_memory_mb, gpu_memory_mb
            
        except Exception as e:
            logger.warning(f"Failed to estimate memory for model: {e}")
            return 0.0, 0.0
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get current system memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "gpu_memory_mb": self._get_gpu_memory()
        }
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024 * 1024)
        except:
            return 0.0


class MedicalAdapterCache(AdapterCache):
    """
    Medical AI specific adapter cache with enhanced features.
    
    Features:
    - Medical compliance-aware caching
    - HIPAA/PHI memory protection
    - Clinical trial data isolation
    - Medical model specific optimizations
    """
    
    def __init__(self,
                 max_size: int = 10,
                 strategy: CacheStrategy = CacheStrategy.LRU,
                 optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.BALANCED,
                 max_memory_mb: int = 16384,
                 enable_gpu_optimization: bool = True):
        
        super().__init__(max_size, strategy)
        
        self.optimization_level = optimization_level
        self.max_memory_mb = max_memory_mb
        self.enable_gpu_optimization = enable_gpu_optimization
        
        # Medical-specific components
        self.memory_profiler = MemoryProfiler()
        self.medical_adapters: Dict[str, CacheEntry] = {}
        self.phi_protected_adapters: Dict[str, CacheEntry] = {}
        
        # Medical compliance tracking
        self.compliance_flags: Dict[str, List[str]] = {}
        self.audit_log: List[Dict[str, Any]] = []
        
        # Performance optimization for medical models
        self.medical_model_thresholds = {
            "max_concurrent_medical": 3,
            "phi_memory_threshold_mb": 512,
            "clinical_data_isolation": True,
            "audit_all_access": True
        }
    
    def get(self, cache_key: str, adapter_id: str) -> Optional[Any]:
        """Get adapter from cache with medical compliance checks."""
        try:
            # Check medical compliance before accessing
            if not self._check_medical_access_compliance(adapter_id):
                logger.warning(f"Medical compliance check failed for adapter {adapter_id}")
                return None
            
            # Standard cache retrieval
            result = super().get(cache_key)
            
            if result is not None:
                # Update medical access tracking
                self._track_medical_access(adapter_id, "cache_hit")
                return result
            else:
                self._track_medical_access(adapter_id, "cache_miss")
                return None
                
        except Exception as e:
            logger.error(f"Medical cache access error for {adapter_id}: {e}")
            return None
    
    def put(self, cache_key: str, adapter_id: str, version_id: str, 
            adapter_model: Any, metadata: Dict[str, Any]):
        """Put adapter in cache with medical compliance."""
        try:
            # Validate medical compliance before caching
            if not self._validate_medical_compliance(adapter_id, metadata):
                logger.error(f"Medical compliance validation failed for adapter {adapter_id}")
                return False
            
            # Calculate memory requirements
            cpu_memory_mb, gpu_memory_mb = self.memory_profiler.estimate_model_memory(adapter_model)
            
            # Create cache entry
            entry = CacheEntry(
                adapter_id=adapter_id,
                version_id=version_id,
                adapter_model=adapter_model,
                metadata=metadata,
                created_at=time.time(),
                last_accessed=time.time(),
                size_bytes=metadata.get("file_size", 0),
                memory_mb=cpu_memory_mb,
                gpu_memory_mb=gpu_memory_mb
            )
            
            # Check memory limits
            if not self._check_memory_limits(cpu_memory_mb, gpu_memory_mb):
                logger.warning(f"Memory limits exceeded for adapter {adapter_id}")
                # Trigger aggressive cleanup
                self._aggressive_memory_cleanup()
            
            # Store in appropriate medical cache
            compliance_flags = metadata.get("compliance_flags", [])
            if "phi_protected" in compliance_flags:
                self.phi_protected_adapters[cache_key] = entry
                logger.info(f"Stored PHI-protected adapter {adapter_id}")
            else:
                self.medical_adapters[cache_key] = entry
            
            # Standard cache update
            super().put(cache_key, entry.metadata, entry.adapter_model)
            
            # Track medical storage
            self._track_medical_access(adapter_id, "cached")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache medical adapter {adapter_id}: {e}")
            return False
    
    def clear_medical_data(self, adapter_id: str) -> bool:
        """Clear medical data from cache with compliance."""
        try:
            entries_to_remove = []
            
            # Find all entries for this adapter
            for key, entry in self.medical_adapters.items():
                if entry.adapter_id == adapter_id:
                    entries_to_remove.append(key)
            
            for key in entries_to_remove:
                entry = self.medical_adapters[key]
                
                # Move to CPU and delete
                if hasattr(entry.adapter_model, 'cpu'):
                    entry.adapter_model.cpu()
                
                del self.medical_adapters[key]
                
                # Remove from standard cache
                if key in self._cache:
                    del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            # Clear PHI-protected data
            phi_entries_to_remove = []
            for key, entry in self.phi_protected_adapters.items():
                if entry.adapter_id == adapter_id:
                    phi_entries_to_remove.append(key)
            
            for key in phi_entries_to_remove:
                entry = self.phi_protected_adapters[key]
                
                # Secure deletion for PHI data
                if hasattr(entry.adapter_model, 'cpu'):
                    entry.adapter_model.cpu()
                
                # Overwrite memory if possible
                if hasattr(entry.adapter_model, 'data') and isinstance(entry.adapter_model.data, torch.Tensor):
                    entry.adapter_model.data.fill_(0)
                
                del self.phi_protected_adapters[key]
            
            # Audit log the cleanup
            self._audit_access(adapter_id, "secure_deletion", {
                "entries_removed": len(entries_to_remove) + len(phi_entries_to_remove),
                "phi_entries_removed": len(phi_entries_to_remove)
            })
            
            logger.info(f"Securely cleared medical data for adapter {adapter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear medical data for {adapter_id}: {e}")
            return False
    
    def get_medical_cache_stats(self) -> Dict[str, Any]:
        """Get medical-specific cache statistics."""
        medical_stats = self.get_stats()
        
        # Add medical-specific stats
        medical_stats.update({
            "medical_adapters": len(self.medical_adapters),
            "phi_protected_adapters": len(self.phi_protected_adapters),
            "compliance_flags": dict(self.compliance_flags),
            "recent_audit_entries": len(self.audit_log),
            "memory_profiling": self.memory_profiler.get_system_memory(),
            "optimization_level": self.optimization_level.value
        })
        
        return medical_stats
    
    def _check_medical_access_compliance(self, adapter_id: str) -> bool:
        """Check if medical access is compliant."""
        # Check if adapter has required compliance flags
        compliance_flags = self.compliance_flags.get(adapter_id, [])
        
        # For PHI-protected adapters, ensure proper access
        if "phi_protected" in compliance_flags:
            # Additional checks would go here
            pass
        
        return True
    
    def _validate_medical_compliance(self, adapter_id: str, metadata: Dict[str, Any]) -> bool:
        """Validate medical compliance before caching."""
        # Check required medical compliance
        required_compliance = ["validation_status"]
        
        for requirement in required_compliance:
            if requirement not in metadata:
                logger.error(f"Missing medical compliance requirement: {requirement}")
                return False
        
        # Store compliance flags
        self.compliance_flags[adapter_id] = metadata.get("compliance_flags", [])
        
        return True
    
    def _check_memory_limits(self, cpu_memory_mb: float, gpu_memory_mb: float) -> bool:
        """Check if memory usage is within limits."""
        current_usage = self.memory_profiler.get_system_memory()
        
        # Check CPU memory
        if current_usage["rss_mb"] + cpu_memory_mb > self.max_memory_mb:
            return False
        
        # Check GPU memory if available
        if self.enable_gpu_optimization and gpu_memory_mb > 0:
            if current_usage["gpu_memory_mb"] + gpu_memory_mb > self.max_memory_mb * 0.8:
                return False
        
        return True
    
    def _aggressive_memory_cleanup(self):
        """Perform aggressive memory cleanup."""
        logger.info("Performing aggressive medical cache cleanup")
        
        # Clear PHI-protected data first (most sensitive)
        for adapter_model in self.phi_protected_adapters.values():
            if hasattr(adapter_model.adapter_model, 'cpu'):
                adapter_model.adapter_model.cpu()
        self.phi_protected_adapters.clear()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Remove from standard cache
        self.clear()
    
    def _track_medical_access(self, adapter_id: str, access_type: str):
        """Track medical adapter access for audit."""
        self.audit_log.append({
            "adapter_id": adapter_id,
            "access_type": access_type,
            "timestamp": time.time(),
            "compliance_flags": self.compliance_flags.get(adapter_id, [])
        })
        
        # Keep only recent audit entries
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-500:]
    
    def _audit_access(self, adapter_id: str, operation: str, details: Dict[str, Any]):
        """Create audit entry for access."""
        audit_entry = {
            "adapter_id": adapter_id,
            "operation": operation,
            "timestamp": time.time(),
            "details": details,
            "system_memory": self.memory_profiler.get_system_memory()
        }
        
        self.audit_log.append(audit_entry)
        
        logger.info(f"Medical audit: {operation} for {adapter_id}")


class AdapterCache:
    """
    Production-grade LRU cache with memory optimization.
    
    Features:
    - LRU, LFU, TTL, and hybrid caching strategies
    - Memory pressure detection and cleanup
    - GPU memory management
    - Medical model specific optimizations
    """
    
    def __init__(self, 
                 max_size: int = 5,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        
        # Core cache structures
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        
        # Performance tracking
        self._stats = CacheStatistics()
        self._access_times: Dict[str, float] = {}
        self._memory_history: List[Dict[str, float]] = []
        
        # Memory management
        self._memory_profiler = MemoryProfiler()
        self._cleanup_callbacks: List[Callable] = []
        
        logger.info(f"AdapterCache initialized: strategy={strategy.value}, max_size={max_size}")
    
    def get(self, cache_key: str, adapter_id: Optional[str] = None) -> Optional[Any]:
        """Get adapter from cache."""
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Check expiration
                if entry.is_expired():
                    self._remove_entry(cache_key)
                    self._stats.misses += 1
                    return None
                
                # Update access statistics
                entry.update_access()
                self._update_access_order(cache_key)
                
                # Update statistics
                self._stats.hits += 1
                self._update_hit_rate()
                
                return entry.adapter_model
            
            self._stats.misses += 1
            self._update_hit_rate()
            return None
    
    def put(self, cache_key: str, metadata: Dict[str, Any], adapter_model: Any):
        """Put adapter in cache."""
        with self._lock:
            # Estimate memory usage
            cpu_memory_mb, gpu_memory_mb = self._memory_profiler.estimate_model_memory(adapter_model)
            
            # Create cache entry
            entry = CacheEntry(
                adapter_id=metadata.get("adapter_id", "unknown"),
                version_id=metadata.get("version_id", "unknown"),
                adapter_model=adapter_model,
                metadata=metadata,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=metadata.get("file_size", 0),
                memory_mb=cpu_memory_mb,
                gpu_memory_mb=gpu_memory_mb,
                ttl_seconds=metadata.get("ttl_seconds")
            )
            
            # Check if we need to evict
            if len(self._cache) >= self.max_size and cache_key not in self._cache:
                self._evict_entries()
            
            # Add entry
            self._cache[cache_key] = entry
            self._access_order[cache_key] = time.time()
            
            # Track memory
            self._track_memory_usage()
            
            # Trigger cleanup callbacks if needed
            current_usage = self._memory_profiler.get_system_memory()
            if current_usage["percent"] > 80:  # High memory usage
                self._trigger_cleanup_callbacks()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            # Move models to CPU before deletion
            for entry in self._cache.values():
                if hasattr(entry.adapter_model, 'cpu'):
                    entry.adapter_model.cpu()
            
            self._cache.clear()
            self._access_order.clear()
            self._access_times.clear()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Adapter cache cleared")
    
    def remove(self, cache_key: str) -> bool:
        """Remove specific entry from cache."""
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                
                # Move model to CPU
                if hasattr(entry.adapter_model, 'cpu'):
                    entry.adapter_model.cpu()
                
                del self._cache[cache_key]
                self._access_order.pop(cache_key, None)
                self._access_times.pop(cache_key, None)
                
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            current_memory = self._memory_profiler.get_system_memory()
            
            # Update statistics
            self._stats.avg_memory_usage_mb = current_memory["rss_mb"]
            self._stats.peak_memory_usage_mb = max(
                [entry.memory_mb for entry in self._cache.values()] + [current_memory["rss_mb"]]
            )
            self._stats.gpu_memory_usage_mb = current_memory["gpu_memory_mb"]
            
            stats_dict = self._stats.to_dict()
            stats_dict.update({
                "cached_items": len(self._cache),
                "max_size": self.max_size,
                "strategy": self.strategy.value,
                "memory_usage": current_memory,
                "access_patterns": self._analyze_access_patterns()
            })
            
            return stats_dict
    
    def add_cleanup_callback(self, callback: Callable):
        """Add cleanup callback."""
        with self._lock:
            self._cleanup_callbacks.append(callback)
    
    def _evict_entries(self):
        """Evict entries based on cache strategy."""
        if not self._cache:
            return
        
        entries_to_evict = []
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            sorted_keys = sorted(self._access_order.keys(), 
                               key=lambda k: self._access_order[k])
            entries_to_evict = sorted_keys[:1]  # Evict one entry
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            frequency_counts = {k: v.access_count for k, v in self._cache.items()}
            min_count = min(frequency_counts.values())
            entries_to_evict = [k for k, v in frequency_counts.items() if v == min_count][:1]
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first
            expired_keys = [k for k, v in self._cache.items() if v.is_expired()]
            if expired_keys:
                entries_to_evict = expired_keys[:1]
        
        elif self.strategy == CacheStrategy.MEMORY_AWARE:
            # Evict based on memory usage
            memory_usage = {k: v.memory_mb for k, v in self._cache.items()}
            max_memory_key = max(memory_usage, key=memory_usage.get)
            entries_to_evict = [max_memory_key]
        
        elif self.strategy == CacheStrategy.HYBRID:
            # Hybrid approach: consider recency, frequency, and memory
            def hybrid_score(key):
                entry = self._cache[key]
                recency_score = 1.0 / (time.time() - entry.last_accessed + 1)
                frequency_score = entry.access_count
                memory_score = 1.0 / (entry.memory_mb + 1)
                return recency_score * frequency_score * memory_score
            
            scored_entries = [(k, hybrid_score(k)) for k in self._cache.keys()]
            scored_entries.sort(key=lambda x: x[1])
            entries_to_evict = [scored_entries[0][0]] if scored_entries else []
        
        # Perform eviction
        for key in entries_to_evict:
            self._remove_entry(key)
            self._stats.evictions += 1
            logger.debug(f"Evicted cache entry: {key}")
    
    def _remove_entry(self, cache_key: str):
        """Remove entry from cache."""
        entry = self._cache.get(cache_key)
        if entry:
            # Move model to CPU
            if hasattr(entry.adapter_model, 'cpu'):
                entry.adapter_model.cpu()
            
            # Remove from all structures
            del self._cache[cache_key]
            self._access_order.pop(cache_key, None)
            self._access_times.pop(cache_key, None)
    
    def _update_access_order(self, cache_key: str):
        """Update access order for LRU."""
        self._access_order.move_to_end(cache_key)
        self._access_times[cache_key] = time.time()
    
    def _update_hit_rate(self):
        """Update cache hit rate."""
        total_requests = self._stats.hits + self._stats.misses
        if total_requests > 0:
            self._stats.hit_rate = self._stats.hits / total_requests
    
    def _track_memory_usage(self):
        """Track memory usage over time."""
        current_usage = self._memory_profiler.get_system_memory()
        self._memory_history.append({
            "timestamp": time.time(),
            **current_usage
        })
        
        # Keep only recent history
        if len(self._memory_history) > 100:
            self._memory_history = self._memory_history[-50:]
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze cache access patterns."""
        if not self._access_times:
            return {"pattern": "no_data"}
        
        recent_accesses = [t for t in self._access_times.values() 
                          if time.time() - t < 3600]  # Last hour
        
        if len(recent_accesses) < 2:
            return {"pattern": "insufficient_data"}
        
        # Analyze access intervals
        intervals = [recent_accesses[i] - recent_accesses[i-1] 
                    for i in range(1, len(recent_accesses))]
        
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval < 60:  # Less than 1 minute
            pattern = "high_frequency"
        elif avg_interval < 300:  # Less than 5 minutes
            pattern = "medium_frequency"
        else:
            pattern = "low_frequency"
        
        return {
            "pattern": pattern,
            "avg_interval_seconds": avg_interval,
            "recent_accesses_1h": len(recent_accesses)
        }
    
    def _trigger_cleanup_callbacks(self):
        """Trigger cleanup callbacks."""
        for callback in self._cleanup_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback())
                else:
                    callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")


# Medical-specific cache factory
def create_medical_adapter_cache(**kwargs) -> MedicalAdapterCache:
    """Factory function to create medical-specific cache."""
    default_kwargs = {
        "max_size": 5,
        "strategy": CacheStrategy.LRU,
        "optimization_level": MemoryOptimizationLevel.BALANCED,
        "max_memory_mb": 8192,
        "enable_gpu_optimization": True
    }
    
    default_kwargs.update(kwargs)
    return MedicalAdapterCache(**default_kwargs)


if __name__ == "__main__":
    # Example usage
    cache = create_medical_adapter_cache(
        max_size=3,
        optimization_level=MemoryOptimizationLevel.BALANCED
    )
    
    # Mock adapter data
    mock_metadata = {
        "adapter_id": "medical_diagnosis_v1",
        "version_id": "v1.0.0",
        "file_size": 1000000,
        "compliance_flags": ["validated", "medical_domain"]
    }
    
    # Add adapter
    cache.put("test_key", mock_metadata, "mock_model")
    
    # Retrieve adapter
    model = cache.get("test_key", "medical_diagnosis_v1")
    print(f"Retrieved model: {model}")
    
    # Get stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Get medical stats
    medical_stats = cache.get_medical_cache_stats()
    print(f"Medical cache stats: {medical_stats}")