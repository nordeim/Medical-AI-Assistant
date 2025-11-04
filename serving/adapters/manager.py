"""
Adapter Manager with Lifecycle Management

Provides comprehensive adapter lifecycle management including
load, unload, reload, and caching operations.
"""

import asyncio
import gc
import hashlib
import json
import logging
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple, AsyncGenerator

import psutil
import torch
from peft import PeftConfig, PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel, AutoConfig

from .registry import AdapterRegistry, AdapterMetadata, AdapterType
from .cache import AdapterCache, MemoryOptimizedCache
from .validator import AdapterValidator, ValidationResult

logger = logging.getLogger(__name__)


class AdapterState(Enum):
    """Adapter lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    UNLOADING = "unloading"
    ERROR = "error"


@dataclass
class AdapterInstance:
    """Represents an active adapter instance."""
    adapter_id: str
    version_id: str
    state: AdapterState
    adapter_model: Optional[Any] = None
    metadata: Optional[AdapterMetadata] = None
    load_time: Optional[float] = None
    memory_usage_mb: float = 0.0
    load_count: int = 0
    error_message: Optional[str] = None
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "adapter_id": self.adapter_id,
            "version_id": self.version_id,
            "state": self.state.value,
            "load_time": self.load_time,
            "memory_usage_mb": self.memory_usage_mb,
            "load_count": self.load_count,
            "error_message": self.error_message,
            "last_accessed": self.last_accessed
        }


@dataclass
class LoadOperation:
    """Represents an adapter load operation."""
    operation_id: str
    adapter_id: str
    version_id: str
    status: str = "pending"  # pending, loading, completed, failed, cancelled
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[AdapterInstance] = None
    
    def complete(self, success: bool, error_msg: Optional[str] = None):
        """Mark operation as completed."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "completed" if success else "failed"
        if error_msg:
            self.error_message = error_msg
    
    def cancel(self):
        """Cancel the operation."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = "cancelled"


class MemoryMonitor:
    """Monitors memory usage and provides cleanup capabilities."""
    
    def __init__(self, max_memory_mb: int = 16384):
        self.max_memory_mb = max_memory_mb
        self._lock = threading.RLock()
        self._memory_warnings = []
        self._cleanup_callbacks: List[Callable] = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "gpu_memory_mb": self._get_gpu_memory_usage()
        }
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage if available."""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / 1024 / 1024
        except:
            return 0.0
    
    def check_memory_limit(self) -> Tuple[bool, List[str]]:
        """Check if memory usage exceeds limits."""
        usage = self.get_memory_usage()
        warnings = []
        
        # Check RAM usage
        if usage["rss_mb"] > self.max_memory_mb:
            warnings.append(f"RAM usage {usage['rss_mb']:.1f}MB exceeds limit {self.max_memory_mb}MB")
        
        # Check available memory
        if usage["available_mb"] < 1024:  # Less than 1GB available
            warnings.append(f"Low available memory: {usage['available_mb']:.1f}MB")
        
        # Check GPU memory if available
        if usage["gpu_memory_mb"] > 0 and usage["gpu_memory_mb"] > self.max_memory_mb * 0.8:
            warnings.append(f"High GPU memory usage: {usage['gpu_memory_mb']:.1f}MB")
        
        return len(warnings) == 0, warnings
    
    async def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup."""
        with self._lock:
            logger.info("Starting memory cleanup")
            
            # Log current usage
            usage_before = self.get_memory_usage()
            logger.info(f"Memory before cleanup: {usage_before}")
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Execute cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Additional cleanup if aggressive
            if aggressive:
                # Force more aggressive cleanup
                for obj in gc.get_objects():
                    if isinstance(obj, torch.Tensor):
                        obj.cpu()
                gc.collect()
            
            # Log new usage
            usage_after = self.get_memory_usage()
            logger.info(f"Memory after cleanup: {usage_after}")
            
            return {
                "before": usage_before,
                "after": usage_after,
                "freed_mb": usage_before["rss_mb"] - usage_after["rss_mb"]
            }
    
    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback."""
        with self._lock:
            self._cleanup_callbacks.append(callback)


class AdapterManager:
    """
    Production-grade adapter manager with lifecycle management.
    
    Features:
    - Asynchronous adapter loading and management
    - Memory optimization and monitoring
    - Lifecycle state management
    - Error handling and recovery
    - Integration with registry and validation systems
    """
    
    def __init__(self,
                 base_model_id: str,
                 registry: AdapterRegistry,
                 max_memory_mb: int = 16384,
                 max_concurrent_loads: int = 2,
                 enable_gpu_optimization: bool = True):
        
        self.base_model_id = base_model_id
        self.registry = registry
        self.max_memory_mb = max_memory_mb
        self.max_concurrent_loads = max_concurrent_loads
        self.enable_gpu_optimization = enable_gpu_optimization
        
        # Core components
        self.memory_monitor = MemoryMonitor(max_memory_mb)
        self.validator = AdapterValidator()
        
        # State management
        self._lock = threading.RLock()
        self._base_model: Optional[PreTrainedModel] = None
        self._base_tokenizer: Optional[Any] = None
        self._active_adapters: Dict[str, AdapterInstance] = {}
        self._load_operations: Dict[str, LoadOperation] = {}
        self._operation_semaphore = asyncio.Semaphore(max_concurrent_loads)
        
        # Performance tracking
        self._load_history: List[Dict[str, Any]] = []
        self._performance_callbacks: List[Callable] = []
        
        logger.info(f"AdapterManager initialized for base model: {base_model_id}")
    
    async def initialize(self):
        """Initialize the manager asynchronously."""
        try:
            # Load base model and tokenizer
            await self._load_base_model_async()
            
            # Setup cleanup callbacks
            self.memory_monitor.add_cleanup_callback(self._cleanup_gpu_cache)
            
            logger.info("AdapterManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdapterManager: {e}")
            raise
    
    def initialize_sync(self):
        """Initialize the manager synchronously."""
        try:
            # Load base model and tokenizer
            self._load_base_model_sync()
            
            logger.info("AdapterManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AdapterManager: {e}")
            raise
    
    def _load_base_model_sync(self):
        """Load base model synchronously."""
        try:
            # Load tokenizer
            self._base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=True
            )
            
            # Auto-detect and load model
            with torch.no_grad():
                config = AutoConfig.from_pretrained(self.base_model_id)
                model_type = config.model_type
                
                if model_type in ["llama", "mistral", "gpt2", "gpt_neo", "gptj"]:
                    self._base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.enable_gpu_optimization else None,
                        trust_remote_code=True
                    )
                elif model_type in ["t5", "bart"]:
                    self._base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.enable_gpu_optimization else None,
                        trust_remote_code=True
                    )
                else:
                    # Default to causal LM
                    self._base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.enable_gpu_optimization else None,
                        trust_remote_code=True
                    )
            
            # Set tokenizer padding token if missing
            if self._base_tokenizer.pad_token is None:
                self._base_tokenizer.pad_token = self._base_tokenizer.eos_token
            
            logger.info(f"Base model loaded: {self.base_model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    async def _load_base_model_async(self):
        """Load base model asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, self._load_base_model_sync)
    
    async def load_adapter(self, 
                          adapter_id: str,
                          version_id: Optional[str] = None,
                          validate: bool = True,
                          timeout: float = 120.0) -> AdapterInstance:
        """
        Load adapter with lifecycle management.
        
        Args:
            adapter_id: Adapter identifier
            version_id: Specific version to load (latest if None)
            validate: Whether to validate before loading
            timeout: Maximum load time
            
        Returns:
            AdapterInstance with loaded adapter
        """
        operation_id = f"{adapter_id}_{version_id or 'latest'}_{int(time.time())}"
        
        async with self._operation_semaphore:
            # Create load operation
            operation = LoadOperation(
                operation_id=operation_id,
                adapter_id=adapter_id,
                version_id=version_id or "latest"
            )
            
            with self._lock:
                self._load_operations[operation_id] = operation
            
            try:
                # Get adapter metadata
                metadata = self.registry.get_adapter(adapter_id)
                if not metadata:
                    raise ValueError(f"Adapter {adapter_id} not found in registry")
                
                # Get specific version
                if version_id:
                    version = next((v for v in metadata.versions if v.version_id == version_id), None)
                    if not version:
                        raise ValueError(f"Version {version_id} not found for adapter {adapter_id}")
                else:
                    version = metadata.get_active_version()
                    if not version:
                        raise ValueError(f"No versions available for adapter {adapter_id}")
                
                # Check if already loaded
                existing_instance = self._get_loaded_instance(adapter_id, version.version_id)
                if existing_instance:
                    logger.info(f"Adapter {adapter_id} version {version.version_id} already loaded")
                    operation.complete(True)
                    self._update_load_stats(operation, existing_instance)
                    return existing_instance
                
                # Validate adapter if requested
                if validate:
                    validation_result = await self._validate_adapter(adapter_id, version, metadata)
                    if not validation_result.is_compatible:
                        raise ValueError(f"Adapter validation failed: {validation_result.issues}")
                
                # Create adapter instance
                instance = AdapterInstance(
                    adapter_id=adapter_id,
                    version_id=version.version_id,
                    state=AdapterState.LOADING
                )
                
                # Load adapter with timeout
                loaded_model = await asyncio.wait_for(
                    self._load_adapter_model(adapter_id, version),
                    timeout=timeout
                )
                
                # Update instance
                instance.adapter_model = loaded_model
                instance.state = AdapterState.LOADED
                instance.load_time = time.time()
                instance.metadata = metadata
                instance.load_count = 1
                
                # Register instance
                with self._lock:
                    self._active_adapters[f"{adapter_id}:{version.version_id}"] = instance
                
                # Update registry stats
                self.registry.update_usage_stats(
                    adapter_id, "load", operation.duration_ms, True
                )
                
                # Complete operation
                operation.complete(True)
                operation.result = instance
                
                # Check memory usage
                await self._check_memory_pressure()
                
                logger.info(f"Successfully loaded adapter {adapter_id} v{version.version}")
                return instance
                
            except asyncio.TimeoutError:
                operation.complete(False, "Load timeout")
                logger.error(f"Load timeout for adapter {adapter_id}")
                raise
                
            except Exception as e:
                operation.complete(False, str(e))
                logger.error(f"Failed to load adapter {adapter_id}: {e}")
                
                # Create error instance
                instance = AdapterInstance(
                    adapter_id=adapter_id,
                    version_id=version_id or "unknown",
                    state=AdapterState.ERROR,
                    error_message=str(e)
                )
                operation.result = instance
                
                # Update registry stats
                self.registry.update_usage_stats(
                    adapter_id, "load", operation.duration_ms, False
                )
                
                raise
    
    async def unload_adapter(self, adapter_id: str, 
                           version_id: Optional[str] = None,
                           force: bool = False) -> bool:
        """Unload adapter and free resources."""
        try:
            # Determine which instance to unload
            target_key = f"{adapter_id}:{version_id or 'latest'}"
            
            with self._lock:
                instance = self._active_adapters.get(target_key)
                if not instance:
                    logger.warning(f"Adapter {adapter_id} not loaded")
                    return False
                
                if instance.state == AdapterState.ACTIVE and not force:
                    logger.warning(f"Cannot unload active adapter {adapter_id} without force=True")
                    return False
                
                instance.state = AdapterState.UNLOADING
            
            # Perform unload
            await self._unload_adapter_instance(instance)
            
            # Remove from active adapters
            with self._lock:
                if target_key in self._active_adapters:
                    del self._active_adapters[target_key]
            
            # Update registry
            self.registry.update_usage_stats(adapter_id, "unload", success=True)
            
            logger.info(f"Unloaded adapter {adapter_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload adapter {adapter_id}: {e}")
            return False
    
    async def reload_adapter(self, adapter_id: str,
                           version_id: Optional[str] = None,
                           validate: bool = True) -> AdapterInstance:
        """Reload adapter (unload then load)."""
        try:
            # Unload current version
            await self.unload_adapter(adapter_id, version_id)
            
            # Load new version
            instance = await self.load_adapter(
                adapter_id, version_id, validate
            )
            
            logger.info(f"Reloaded adapter {adapter_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to reload adapter {adapter_id}: {e}")
            raise
    
    def get_adapter_instance(self, adapter_id: str, 
                           version_id: Optional[str] = None) -> Optional[AdapterInstance]:
        """Get loaded adapter instance."""
        target_key = f"{adapter_id}:{version_id or 'latest'}"
        
        with self._lock:
            return self._active_adapters.get(target_key)
    
    def get_active_adapters(self) -> Dict[str, AdapterInstance]:
        """Get all active adapter instances."""
        with self._lock:
            return dict(self._active_adapters)
    
    def set_active_adapter(self, adapter_id: str, 
                          version_id: Optional[str] = None) -> bool:
        """Set adapter as active."""
        try:
            target_key = f"{adapter_id}:{version_id or 'latest'}"
            
            with self._lock:
                if target_key not in self._active_adapters:
                    logger.error(f"Adapter {adapter_id} not loaded")
                    return False
                
                instance = self._active_adapters[target_key]
                
                if instance.state != AdapterState.LOADED:
                    logger.error(f"Adapter {adapter_id} not in LOADED state")
                    return False
                
                instance.state = AdapterState.ACTIVE
                instance.last_accessed = time.time()
                
                logger.info(f"Set active adapter: {adapter_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to set active adapter {adapter_id}: {e}")
            return False
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get comprehensive manager statistics."""
        with self._lock:
            memory_usage = self.memory_monitor.get_memory_usage()
            
            # Adapter state distribution
            state_counts = {}
            for instance in self._active_adapters.values():
                state = instance.state.value
                state_counts[state] = state_counts.get(state, 0) + 1
            
            # Load operation stats
            active_operations = len(self._load_operations)
            completed_operations = sum(
                1 for op in self._load_operations.values() 
                if op.status in ["completed", "failed", "cancelled"]
            )
            
            return {
                "base_model_id": self.base_model_id,
                "active_adapters": len(self._active_adapters),
                "adapter_states": state_counts,
                "active_load_operations": active_operations,
                "completed_operations": completed_operations,
                "memory_usage": memory_usage,
                "registry_stats": self.registry.get_registry_stats()
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "status": "healthy",
            "checks": {},
            "issues": []
        }
        
        # Check memory
        memory_ok, memory_warnings = self.memory_monitor.check_memory_limit()
        health["checks"]["memory"] = memory_ok
        if not memory_ok:
            health["issues"].extend(memory_warnings)
        
        # Check base model
        health["checks"]["base_model"] = self._base_model is not None
        
        # Check adapter instances
        for key, instance in self._active_adapters.items():
            if instance.state == AdapterState.ERROR:
                health["issues"].append(f"Adapter {instance.adapter_id} in error state")
        
        health["checks"]["adapters"] = len(self._active_adapters) > 0
        
        # Overall status
        if health["issues"]:
            health["status"] = "degraded" if health["checks"]["memory"] else "unhealthy"
        
        return health
    
    async def cleanup(self):
        """Cleanup all resources."""
        logger.info("Starting AdapterManager cleanup")
        
        # Unload all adapters
        for key in list(self._active_adapters.keys()):
            adapter_id = key.split(":")[0]
            await self.unload_adapter(adapter_id, force=True)
        
        # Unload base model
        if self._base_model:
            if hasattr(self._base_model, 'cpu'):
                self._base_model.cpu()
            del self._base_model
            self._base_model = None
        
        if self._base_tokenizer:
            del self._base_tokenizer
            self._base_tokenizer = None
        
        # Final memory cleanup
        await self.memory_monitor.cleanup_memory(aggressive=True)
        
        logger.info("AdapterManager cleanup completed")
    
    # Private methods
    
    def _get_loaded_instance(self, adapter_id: str, version_id: str) -> Optional[AdapterInstance]:
        """Get loaded adapter instance."""
        target_key = f"{adapter_id}:{version_id}"
        return self._active_adapters.get(target_key)
    
    async def _validate_adapter(self, adapter_id: str, version, metadata) -> ValidationResult:
        """Validate adapter before loading."""
        try:
            # Create validation request
            validation_result = self.validator.validate_adapter(
                adapter_path=version.model_path,
                base_model_id=self.base_model_id,
                adapter_id=adapter_id
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Validation failed for {adapter_id}: {e}")
            # Return failed validation result
            from .validator import ValidationResult, CompatibilityIssue
            return ValidationResult(
                adapter_id=adapter_id,
                is_compatible=False,
                issues=[CompatibilityIssue(
                    severity="error",
                    category="validation",
                    message=f"Validation failed: {str(e)}"
                )]
            )
    
    async def _load_adapter_model(self, adapter_id: str, version) -> Any:
        """Load PEFT adapter model."""
        loop = asyncio.get_event_loop()
        
        def load_model():
            try:
                # Determine PEFT model class based on base model
                if isinstance(self._base_model, AutoModelForCausalLM):
                    adapter_model = PeftModelForCausalLM.from_pretrained(
                        self._base_model,
                        version.model_path
                    )
                elif isinstance(self._base_model, AutoModelForSeq2SeqLM):
                    adapter_model = PeftModelForSeq2SeqLM.from_pretrained(
                        self._base_model,
                        version.model_path
                    )
                else:
                    # Generic PEFT model
                    adapter_model = PeftModel.from_pretrained(
                        self._base_model,
                        version.model_path
                    )
                
                return adapter_model
                
            except Exception as e:
                logger.error(f"Failed to load model for {adapter_id}: {e}")
                raise
        
        return await loop.run_in_executor(executor=None, func=load_model)
    
    async def _unload_adapter_instance(self, instance: AdapterInstance):
        """Unload a specific adapter instance."""
        try:
            # Move model to CPU to free GPU memory
            if instance.adapter_model and hasattr(instance.adapter_model, 'cpu'):
                instance.adapter_model.cpu()
            
            # Clear references
            instance.adapter_model = None
            instance.state = AdapterState.UNLOADED
            
        except Exception as e:
            logger.warning(f"Error during adapter unload for {instance.adapter_id}: {e}")
    
    async def _check_memory_pressure(self):
        """Check for memory pressure and trigger cleanup if needed."""
        memory_ok, warnings = self.memory_monitor.check_memory_limit()
        
        if not memory_ok or warnings:
            logger.warning(f"Memory pressure detected: {warnings}")
            
            # Aggressive cleanup if memory critical
            if not memory_ok:
                await self.memory_monitor.cleanup_memory(aggressive=True)
                
                # Consider unloading least recently used adapters
                await self._unload_least_recently_used()
    
    async def _unload_least_recently_used(self, keep_count: int = 2):
        """Unload least recently used adapters."""
        with self._lock:
            # Sort by last accessed time
            sorted_adapters = sorted(
                self._active_adapters.items(),
                key=lambda x: x[1].last_accessed
            )
            
            # Unload oldest adapters
            adapters_to_unload = sorted_adapters[:-keep_count]
            
            for key, instance in adapters_to_unload:
                if instance.state != AdapterState.ACTIVE:
                    logger.info(f"Unloading LRU adapter: {instance.adapter_id}")
                    asyncio.create_task(self.unload_adapter(instance.adapter_id, instance.version_id))
    
    def _update_load_stats(self, operation: LoadOperation, instance: AdapterInstance):
        """Update load operation statistics."""
        operation.complete(True)
        operation.result = instance
        
        # Add to performance history
        if operation.duration_ms:
            self._load_history.append({
                "operation_id": operation.operation_id,
                "adapter_id": operation.adapter_id,
                "duration_ms": operation.duration_ms,
                "timestamp": time.time()
            })
            
            # Keep only recent history
            if len(self._load_history) > 1000:
                self._load_history = self._load_history[-500:]
    
    def _cleanup_gpu_cache(self):
        """Cleanup GPU cache."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass


# Context manager for AdapterManager
@asynccontextmanager
async def create_adapter_manager(base_model_id: str,
                                registry: AdapterRegistry,
                                **kwargs) -> AsyncGenerator[AdapterManager, None]:
    """Create and manage adapter manager lifecycle."""
    manager = AdapterManager(base_model_id, registry, **kwargs)
    
    try:
        await manager.initialize()
        yield manager
    finally:
        await manager.cleanup()


if __name__ == "__main__":
    # Example usage
    async def main():
        from .registry import AdapterRegistry, AdapterType
        
        # Create registry
        registry = AdapterRegistry("./test_manager_registry.db")
        
        # Create manager
        async with create_adapter_manager(
            base_model_id="microsoft/DialoGPT-medium",
            registry=registry,
            max_memory_mb=8192,
            max_concurrent_loads=2
        ) as manager:
            
            # Health check
            health = await manager.health_check()
            print(f"Manager health: {health}")
            
            # Get stats
            stats = manager.get_manager_stats()
            print(f"Manager stats: {stats}")
            
    # asyncio.run(main())