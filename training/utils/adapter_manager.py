"""
Adapter Management System for PEFT Models

This module provides comprehensive adapter loading, hot-swapping, and memory management
for PEFT (Parameter-Efficient Fine-Tuning) models in production environments.
"""

import asyncio
import gc
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from threading import RLock
import psutil
import weakref

import torch
from peft import PeftConfig, PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedModel
from huggingface_hub import snapshot_download, list_repo_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdapterMetadata:
    """Metadata for adapter instances."""
    adapter_id: str
    adapter_type: str
    model_path: str
    base_model_id: str
    version: str
    timestamp: float
    file_size: int
    description: Optional[str] = None
    tags: List[str] = None
    performance_metrics: Dict[str, float] = None
    validation_status: str = "pending"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    @property
    def cache_key(self) -> str:
        """Generate cache key for adapter."""
        return hashlib.md5(
            f"{self.adapter_id}:{self.version}:{self.model_path}".encode()
        ).hexdigest()


@dataclass
class AdapterPerformanceMetrics:
    """Performance metrics for adapter operations."""
    load_time: float
    memory_usage_mb: float
    inference_latency_ms: float
    throughput_tokens_per_sec: float
    validation_accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryManager:
    """Manages memory allocation and cleanup for adapters."""
    
    def __init__(self, max_memory_mb: int = 8192):
        self.max_memory_mb = max_memory_mb
        self._lock = RLock()
        self._memory_warnings = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage exceeds limit."""
        usage = self.get_memory_usage()
        return usage["rss_mb"] < self.max_memory_mb
    
    def cleanup_unused_adapters(self, active_adapter_ids: List[str]):
        """Force garbage collection and cleanup unused memory."""
        with self._lock:
            # Log current memory usage
            usage = self.get_memory_usage()
            logger.info(f"Memory usage before cleanup: {usage}")
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Check memory after cleanup
            new_usage = self.get_memory_usage()
            logger.info(f"Memory usage after cleanup: {new_usage}")
            
            return new_usage["rss_mb"] < usage["rss_mb"]


class AdapterValidator:
    """Validates adapter compatibility and integrity."""
    
    def __init__(self):
        self._supported_types = {
            "peft_lora": "lora",
            "peft_adalora": "adalora", 
            "peft_ia3": "ia3",
            "peft_prefix_tuning": "prefix_tuning",
            "peft_p_tuning": "p_tuning"
        }
    
    def validate_adapter_path(self, adapter_path: str) -> Tuple[bool, str]:
        """Validate adapter path and structure."""
        try:
            path = Path(adapter_path)
            if not path.exists():
                return False, "Adapter path does not exist"
            
            # Check for required files
            required_files = ["adapter_config.json"]
            missing_files = [f for f in required_files if not (path / f).exists()]
            
            if missing_files:
                return False, f"Missing required files: {missing_files}"
            
            # Validate config
            config_path = path / "adapter_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if "peft_type" not in config:
                return False, "Invalid adapter config: missing peft_type"
            
            peft_type = config["peft_type"]
            if peft_type not in self._supported_types.values():
                return False, f"Unsupported PEFT type: {peft_type}"
            
            return True, "Valid adapter"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def validate_base_model_compatibility(self, adapter_config: Dict, base_model_id: str) -> Tuple[bool, str]:
        """Validate adapter compatibility with base model."""
        try:
            # Check model architecture compatibility
            peft_type = adapter_config.get("peft_type")
            
            # For now, basic validation - can be extended
            if peft_type in ["lora", "adalora", "ia3"]:
                return True, "Compatible"
            
            return False, f"Unknown compatibility for PEFT type: {peft_type}"
            
        except Exception as e:
            return False, f"Compatibility check failed: {str(e)}"


class AdapterCache:
    """Manages adapter caching with LRU strategy."""
    
    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self._cache: Dict[str, Tuple[AdapterMetadata, Any]] = {}
        self._access_order: List[str] = []
        self._lock = RLock()
        
    def get(self, cache_key: str) -> Optional[Any]:
        """Get adapter from cache."""
        with self._lock:
            if cache_key in self._cache:
                # Update access order (move to end)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return self._cache[cache_key][1]
            return None
    
    def put(self, cache_key: str, metadata: AdapterMetadata, adapter: Any):
        """Put adapter in cache."""
        with self._lock:
            if cache_key in self._cache:
                # Update existing entry
                self._cache[cache_key] = (metadata, adapter)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
            else:
                # Add new entry
                self._cache[cache_key] = (metadata, adapter)
                self._access_order.append(cache_key)
                
                # Evict if necessary
                if len(self._cache) > self.max_size:
                    self._evict_oldest()
    
    def _evict_oldest(self):
        """Remove oldest accessed item from cache."""
        if self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                # Clean up GPU memory if needed
                adapter = self._cache[oldest_key][1]
                if hasattr(adapter, 'cpu'):
                    adapter.cpu()
                del self._cache[oldest_key]
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            for _, adapter in self._cache.values():
                if hasattr(adapter, 'cpu'):
                    adapter.cpu()
            self._cache.clear()
            self._access_order.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "cached_items": len(self._cache),
                "max_size": self.max_size,
                "hit_rate": getattr(self, '_hit_rate', 0.0)
            }


class HotSwapManager:
    """Manages hot-swapping of adapters with zero downtime."""
    
    def __init__(self, manager: 'AdapterManager'):
        self.manager = manager
        self._swap_lock = asyncio.Lock()
        self._swap_queue: asyncio.Queue = asyncio.Queue()
        self._active_swaps: Dict[str, asyncio.Task] = {}
        
    async def schedule_swap(self, from_adapter: str, to_adapter: str, 
                          timeout: float = 30.0) -> bool:
        """Schedule adapter hot-swap."""
        swap_id = f"{from_adapter}->{to_adapter}_{int(time.time())}"
        
        try:
            async with self._swap_lock:
                logger.info(f"Starting hot-swap: {swap_id}")
                
                # Validate target adapter
                target_metadata = self.manager.get_adapter_metadata(to_adapter)
                if not target_metadata:
                    logger.error(f"Target adapter {to_adapter} not found")
                    return False
                
                # Load new adapter in background
                load_task = asyncio.create_task(
                    self.manager.load_adapter_async(to_adapter)
                )
                
                # Wait for load with timeout
                try:
                    new_adapter = await asyncio.wait_for(load_task, timeout=timeout)
                except asyncio.TimeoutError:
                    logger.error(f"Adapter load timeout for {to_adapter}")
                    return False
                
                # Test new adapter
                if not await self._test_adapter(new_adapter):
                    logger.error(f"Adapter validation failed for {to_adapter}")
                    return False
                
                # Perform swap
                success = await self._perform_swap(from_adapter, to_adapter, new_adapter)
                
                if success:
                    logger.info(f"Hot-swap completed successfully: {swap_id}")
                    # Cleanup old adapter
                    if from_adapter != to_adapter:
                        await self.manager.unload_adapter(from_adapter)
                else:
                    logger.error(f"Hot-swap failed: {swap_id}")
                    
                return success
                
        except Exception as e:
            logger.error(f"Hot-swap error: {str(e)}")
            return False
    
    async def _test_adapter(self, adapter: Any) -> bool:
        """Test adapter functionality."""
        try:
            # Basic test - this would be adapter-specific
            if hasattr(adapter, 'eval'):
                adapter.eval()
            return True
        except Exception as e:
            logger.error(f"Adapter test failed: {str(e)}")
            return False
    
    async def _perform_swap(self, from_adapter: str, to_adapter: str, new_adapter: Any) -> bool:
        """Perform the actual adapter swap."""
        try:
            # Update active adapter
            self.manager._active_adapter_id = to_adapter
            self.manager._active_adapter = new_adapter
            return True
        except Exception as e:
            logger.error(f"Swap execution failed: {str(e)}")
            return False
    
    async def rollback_swap(self, to_adapter: str, previous_adapter: str) -> bool:
        """Rollback to previous adapter."""
        try:
            logger.info(f"Rolling back to adapter: {previous_adapter}")
            previous_metadata = self.manager.get_adapter_metadata(previous_adapter)
            if previous_metadata:
                previous_adapter = await self.manager.load_adapter_async(previous_adapter)
                await self._perform_swap(to_adapter, previous_adapter, previous_adapter)
                return True
            return False
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False


class AdapterManager:
    """
    Main adapter management class that orchestrates all adapter operations.
    
    Features:
    - PEFT adapter loading and unloading
    - Dynamic adapter switching  
    - Memory-efficient management
    - Hot-swapping with zero downtime
    - Caching and performance optimization
    """
    
    def __init__(self, 
                 base_model_id: str,
                 cache_dir: Optional[str] = None,
                 max_cache_size: int = 5,
                 max_memory_mb: int = 8192):
        
        self.base_model_id = base_model_id
        self.cache_dir = Path(cache_dir or "~/.cache/adapters").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.validator = AdapterValidator()
        self.memory_manager = MemoryManager(max_memory_mb)
        self.cache = AdapterCache(max_cache_size)
        self.hot_swap_manager = HotSwapManager(self)
        
        # State management
        self._lock = RLock()
        self._base_model: Optional[PreTrainedModel] = None
        self._base_tokenizer: Optional[Any] = None
        self._active_adapter_id: Optional[str] = None
        self._active_adapter: Optional[Any] = None
        self._adapter_metadata: Dict[str, AdapterMetadata] = {}
        
        # Performance tracking
        self._performance_history: List[AdapterPerformanceMetrics] = []
        
        logger.info(f"AdapterManager initialized for base model: {base_model_id}")
    
    async def initialize_async(self):
        """Initialize manager asynchronously."""
        try:
            # Load base model and tokenizer
            await self._load_base_model_async()
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize manager: {str(e)}")
            raise
    
    def initialize(self):
        """Initialize manager synchronously."""
        try:
            # Load base model and tokenizer
            self._load_base_model()
            logger.info("Base model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize manager: {str(e)}")
            raise
    
    def _load_base_model(self):
        """Load base model synchronously."""
        try:
            self._base_tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                trust_remote_code=True
            )
            
            # Auto-detect model type
            with torch.no_grad():
                test_config = AutoConfig.from_pretrained(self.base_model_id)
                model_type = test_config.model_type
                
                if model_type in ["llama", "mistral", "gpt2", "gpt_neo", "gptj"]:
                    self._base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                elif model_type in ["t5", "bart"]:
                    self._base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                else:
                    # Default to causal LM
                    self._base_model = AutoModelForCausalLM.from_pretrained(
                        self.base_model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                
            logger.info(f"Loaded base model: {self.base_model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {str(e)}")
            raise
    
    async def _load_base_model_async(self):
        """Load base model asynchronously."""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, self._load_base_model)
    
    async def load_adapter_async(self, adapter_path: str, 
                                adapter_id: Optional[str] = None) -> Any:
        """Load adapter asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor=None,  # Use default thread pool
            func=lambda: self.load_adapter(adapter_path, adapter_id)
        )
    
    def load_adapter(self, adapter_path: str, 
                    adapter_id: Optional[str] = None) -> Any:
        """
        Load adapter with validation and caching.
        
        Args:
            adapter_path: Path to adapter or HF model ID
            adapter_id: Optional custom identifier for the adapter
            
        Returns:
            Loaded adapter model
        """
        adapter_id = adapter_id or adapter_path
        
        with self._lock:
            # Check cache first
            metadata = self._get_or_create_metadata(adapter_path, adapter_id)
            cache_key = metadata.cache_key
            
            cached_adapter = self.cache.get(cache_key)
            if cached_adapter is not None:
                logger.info(f"Adapter {adapter_id} loaded from cache")
                return cached_adapter
            
            # Validate adapter
            valid, message = self.validator.validate_adapter_path(adapter_path)
            if not valid:
                raise ValueError(f"Invalid adapter: {message}")
            
            # Load adapter
            start_time = time.time()
            try:
                # Load PEFT adapter
                if hasattr(PeftModelForCausalLM, 'from_pretrained'):
                    adapter_model = PeftModelForCausalLM.from_pretrained(
                        self._base_model, adapter_path
                    )
                else:
                    adapter_model = PeftModel.from_pretrained(
                        self._base_model, adapter_path
                    )
                
                load_time = time.time() - start_time
                
                # Create performance metrics
                memory_usage = self.memory_manager.get_memory_usage()["rss_mb"]
                metrics = AdapterPerformanceMetrics(
                    load_time=load_time,
                    memory_usage_mb=memory_usage,
                    inference_latency_ms=0.0,  # Will be measured during inference
                    throughput_tokens_per_sec=0.0
                )
                
                # Cache adapter
                self.cache.put(cache_key, metadata, adapter_model)
                self._adapter_metadata[adapter_id] = metadata
                
                # Update performance history
                self._performance_history.append(metrics)
                
                logger.info(f"Adapter {adapter_id} loaded successfully in {load_time:.2f}s")
                return adapter_model
                
            except Exception as e:
                logger.error(f"Failed to load adapter {adapter_id}: {str(e)}")
                raise
    
    def _get_or_create_metadata(self, adapter_path: str, adapter_id: str) -> AdapterMetadata:
        """Get or create adapter metadata."""
        if adapter_id in self._adapter_metadata:
            return self._adapter_metadata[adapter_id]
        
        # Get file size
        file_size = 0
        try:
            path = Path(adapter_path)
            if path.exists():
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        file_size += file_path.stat().st_size
        except:
            file_size = 0
        
        # Create metadata
        metadata = AdapterMetadata(
            adapter_id=adapter_id,
            adapter_type="peft",
            model_path=adapter_path,
            base_model_id=self.base_model_id,
            version="1.0.0",
            timestamp=time.time(),
            file_size=file_size,
            description=f"Adapter loaded from {adapter_path}"
        )
        
        return metadata
    
    def switch_adapter(self, adapter_id: str) -> bool:
        """Switch to a loaded adapter."""
        with self._lock:
            if adapter_id not in self._adapter_metadata:
                logger.error(f"Adapter {adapter_id} not loaded")
                return False
            
            try:
                metadata = self._adapter_metadata[adapter_id]
                cache_key = metadata.cache_key
                
                new_adapter = self.cache.get(cache_key)
                if new_adapter is None:
                    logger.error(f"Adapter {adapter_id} not in cache")
                    return False
                
                # Set as active
                self._active_adapter_id = adapter_id
                self._active_adapter = new_adapter
                
                logger.info(f"Switched to adapter: {adapter_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to switch adapter: {str(e)}")
                return False
    
    async def hot_swap_adapter(self, new_adapter_path: str, 
                              new_adapter_id: Optional[str] = None,
                              timeout: float = 30.0) -> bool:
        """
        Perform hot-swap to new adapter with zero downtime.
        
        Args:
            new_adapter_path: Path to new adapter
            new_adapter_id: Optional identifier for new adapter
            timeout: Maximum time for swap operation
            
        Returns:
            True if swap successful, False otherwise
        """
        current_adapter = self._active_adapter_id
        
        try:
            return await self.hot_swap_manager.schedule_swap(
                from_adapter=current_adapter or "",
                to_adapter=new_adapter_id or new_adapter_path,
                timeout=timeout
            )
        except Exception as e:
            logger.error(f"Hot-swap failed: {str(e)}")
            return False
    
    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload adapter and free memory."""
        with self._lock:
            if adapter_id not in self._adapter_metadata:
                logger.warning(f"Adapter {adapter_id} not found")
                return False
            
            try:
                # Remove from cache
                metadata = self._adapter_metadata[adapter_id]
                cache_key = metadata.cache_key
                
                # Clear from cache
                if cache_key in self.cache._cache:
                    adapter = self.cache._cache[cache_key][1]
                    if hasattr(adapter, 'cpu'):
                        adapter.cpu()
                    del self.cache._cache[cache_key]
                
                # Remove from access order
                if cache_key in self.cache._access_order:
                    self.cache._access_order.remove(cache_key)
                
                # Remove metadata
                del self._adapter_metadata[adapter_id]
                
                # Clear active adapter if this was it
                if self._active_adapter_id == adapter_id:
                    self._active_adapter_id = None
                    self._active_adapter = None
                
                # Force cleanup
                self.memory_manager.cleanup_unused_adapters(
                    list(self._adapter_metadata.keys())
                )
                
                logger.info(f"Unloaded adapter: {adapter_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload adapter {adapter_id}: {str(e)}")
                return False
    
    def get_active_adapter(self) -> Optional[Tuple[str, Any]]:
        """Get currently active adapter."""
        with self._lock:
            if self._active_adapter_id and self._active_adapter:
                return (self._active_adapter_id, self._active_adapter)
            return None
    
    def get_adapter_metadata(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata."""
        return self._adapter_metadata.get(adapter_id)
    
    def list_adapters(self) -> Dict[str, AdapterMetadata]:
        """List all loaded adapters."""
        with self._lock:
            return dict(self._adapter_metadata)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            if not self._performance_history:
                return {"message": "No performance data available"}
            
            return {
                "total_adapters_loaded": len(self._performance_history),
                "avg_load_time": sum(p.load_time for p in self._performance_history) / len(self._performance_history),
                "avg_memory_usage": sum(p.memory_usage_mb for p in self._performance_history) / len(self._performance_history),
                "memory_stats": self.memory_manager.get_memory_usage(),
                "cache_stats": self.cache.get_stats()
            }
    
    async def benchmark_adapter(self, adapter_id: str, 
                              num_iterations: int = 100) -> Optional[AdapterPerformanceMetrics]:
        """Benchmark adapter performance."""
        try:
            if adapter_id not in self._adapter_metadata:
                raise ValueError(f"Adapter {adapter_id} not found")
            
            # Get adapter from cache
            metadata = self._adapter_metadata[adapter_id]
            adapter = self.cache.get(metadata.cache_key)
            
            if adapter is None:
                raise ValueError(f"Adapter {adapter_id} not in cache")
            
            # Benchmark inference
            start_time = time.time()
            test_input = "Hello, how are you?"  # Simple test prompt
            
            with torch.no_grad():
                inputs = self._base_tokenizer.encode(test_input, return_tensors="pt")
                if hasattr(adapter, 'to'):
                    inputs = inputs.to(adapter.device)
                
                for _ in range(num_iterations):
                    _ = adapter.generate(inputs, max_length=10)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            avg_latency_ms = (total_time / num_iterations) * 1000
            throughput = (num_iterations * 10) / total_time  # tokens per second
            
            metrics = AdapterPerformanceMetrics(
                load_time=0.0,  # Already loaded
                memory_usage_mb=self.memory_manager.get_memory_usage()["rss_mb"],
                inference_latency_ms=avg_latency_ms,
                throughput_tokens_per_sec=throughput
            )
            
            logger.info(f"Benchmark completed for {adapter_id}: {avg_latency_ms:.2f}ms latency, {throughput:.1f} tok/s")
            return metrics
            
        except Exception as e:
            logger.error(f"Benchmark failed for {adapter_id}: {str(e)}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        with self._lock:
            try:
                # Clear all adapters
                self.cache.clear()
                self._adapter_metadata.clear()
                
                # Clear active adapter
                self._active_adapter_id = None
                self._active_adapter = None
                
                # Unload base model
                if self._base_model:
                    if hasattr(self._base_model, 'cpu'):
                        self._base_model.cpu()
                    del self._base_model
                    self._base_model = None
                
                if self._base_tokenizer:
                    del self._base_tokenizer
                    self._base_tokenizer = None
                
                # Force cleanup
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                logger.info("AdapterManager cleanup completed")
                
            except Exception as e:
                logger.error(f"Cleanup error: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Utility functions

def create_adapter_manager(base_model_id: str, 
                          cache_dir: Optional[str] = None,
                          **kwargs) -> AdapterManager:
    """Factory function to create adapter manager."""
    return AdapterManager(
        base_model_id=base_model_id,
        cache_dir=cache_dir,
        **kwargs
    )


async def create_adapter_manager_async(base_model_id: str,
                                     cache_dir: Optional[str] = None,
                                     **kwargs) -> AdapterManager:
    """Factory function to create adapter manager asynchronously."""
    manager = AdapterManager(
        base_model_id=base_model_id,
        cache_dir=cache_dir,
        **kwargs
    )
    await manager.initialize_async()
    return manager


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create manager
        manager = await create_adapter_manager_async(
            base_model_id="microsoft/DialoGPT-medium",
            cache_dir="./adapter_cache",
            max_cache_size=3,
            max_memory_mb=4096
        )
        
        try:
            # Load adapter
            adapter = await manager.load_adapter_async("./sample_adapter")
            
            # Switch to adapter
            manager.switch_adapter("sample_adapter")
            
            # Benchmark
            metrics = await manager.benchmark_adapter("sample_adapter")
            print(f"Performance metrics: {metrics}")
            
            # Hot swap example (would need actual adapter)
            # await manager.hot_swap_adapter("./new_adapter", timeout=30.0)
            
        finally:
            manager.cleanup()
    
    # Run example
    # asyncio.run(main())