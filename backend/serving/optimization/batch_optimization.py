"""
Batch processing optimization for improved throughput and efficiency.
"""

import torch
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import BatchConfig, InferenceMode


logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Strategies for batch processing."""
    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    TIMEOUT_BASED = "timeout_based"
    LATENCY_AWARE = "latency_aware"
    MEMORY_AWARE = "memory_aware"


@dataclass
class BatchRequest:
    """A single request for batch processing."""
    request_id: str
    input_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    timeout_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.timeout_seconds is None:
            self.timeout_seconds = 300.0  # 5 minutes default


@dataclass
class BatchResult:
    """Result of batch processing."""
    request_id: str
    output_data: Any
    processing_time_ms: float
    batch_id: Optional[str] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "output_data": self.output_data,
            "processing_time_ms": self.processing_time_ms,
            "batch_id": self.batch_id,
            "error_message": self.error_message,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata
        }


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance."""
    total_requests: int = 0
    total_batches: int = 0
    average_batch_size: float = 0.0
    average_processing_time_ms: float = 0.0
    throughput_requests_per_second: float = 0.0
    throughput_tokens_per_second: float = 0.0
    queue_wait_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": self.average_batch_size,
            "average_processing_time_ms": self.average_processing_time_ms,
            "throughput_requests_per_second": self.throughput_requests_per_second,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "queue_wait_time_ms": self.queue_wait_time_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "cache_hit_rate": self.cache_hit_rate,
            "error_rate": self.error_rate,
        }


class BatchProcessor:
    """
    Advanced batch processor for optimized throughput and resource utilization.
    Supports dynamic batching, multiple strategies, and performance monitoring.
    """
    
    def __init__(self, config: BatchConfig, inference_function: Optional[Callable] = None):
        self.config = config
        self.inference_function = inference_function
        
        # Request management
        self.request_queue = queue.PriorityQueue()
        self.pending_requests = {}
        self.active_batches = {}
        self.completed_batches = {}
        
        # Performance tracking
        self.metrics = BatchMetrics()
        self.processing_times = []
        self.batch_history = []
        
        # Dynamic batching state
        self.current_batch = []
        self.last_batch_time = time.time()
        self.batch_lock = threading.Lock()
        
        # Caching
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Control flags
        self.is_running = False
        self.is_processing = False
        
        # Background processing thread
        self.processing_thread = None
        self.metrics_thread = None
        
        # Performance optimization settings
        self.optimization_level = InferenceMode.BALANCED
        
        logger.info("Batch processor initialized")
    
    def start(self):
        """Start the batch processor."""
        if self.is_running:
            logger.warning("Batch processor is already running")
            return
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        # Start metrics collection thread
        self.metrics_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        self.metrics_thread.start()
        
        logger.info("Batch processor started")
    
    def stop(self):
        """Stop the batch processor."""
        self.is_running = False
        
        # Wait for threads to complete
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=2.0)
        
        # Process remaining requests
        self._process_remaining_batches()
        
        logger.info("Batch processor stopped")
    
    def submit_request(self, request: BatchRequest) -> str:
        """Submit a request for batch processing."""
        if not self.is_running:
            logger.error("Batch processor is not running")
            raise RuntimeError("Batch processor is not running")
        
        # Add to queue with priority (higher priority = lower number)
        priority = -request.priority  # Negative for max-heap behavior
        self.request_queue.put((priority, request.timestamp, request))
        
        # Track pending request
        self.pending_requests[request.request_id] = request
        
        logger.debug(f"Request {request.request_id} submitted to batch queue")
        return request.request_id
    
    def submit_batch(self, requests: List[BatchRequest]) -> List[str]:
        """Submit multiple requests as a batch."""
        request_ids = []
        
        for request in requests:
            request_id = self.submit_request(request)
            request_ids.append(request_id)
        
        logger.info(f"Submitted batch of {len(requests)} requests")
        return request_ids
    
    def get_result(self, request_id: str, timeout: Optional[float] = None) -> Optional[BatchResult]:
        """Get the result of a submitted request."""
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while True:
            # Check if result is available
            for batch_id, batch in self.completed_batches.items():
                for result in batch["results"]:
                    if result.request_id == request_id:
                        # Clean up old completed batches
                        if time.time() - batch["timestamp"] > 3600:  # 1 hour
                            del self.completed_batches[batch_id]
                        return result
            
            # Check for timeout
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for result of request {request_id}")
                return None
            
            # Check if processor is still running
            if not self.is_running:
                logger.warning("Batch processor stopped while waiting for result")
                return None
            
            time.sleep(check_interval)
    
    async def submit_async_request(self, request: BatchRequest) -> BatchResult:
        """Submit an async request and return the result."""
        request_id = self.submit_request(request)
        
        # Wait for result with timeout
        result = self.get_result(request_id, timeout=request.timeout_seconds)
        
        if result is None:
            raise TimeoutError(f"Request {request_id} timed out")
        
        return result
    
    def _processing_loop(self):
        """Main processing loop for batched requests."""
        while self.is_running:
            try:
                # Check if we should create a batch
                should_process = self._should_create_batch()
                
                if should_process:
                    self._process_batch()
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _should_create_batch(self) -> bool:
        """Determine if we should create and process a batch."""
        # Check queue size
        queue_size = self.request_queue.qsize()
        
        # Check time-based conditions
        time_since_last_batch = time.time() - self.last_batch_time
        
        # Different strategies
        if self.config.enable_dynamic_batching:
            return self._should_create_dynamic_batch(queue_size, time_since_last_batch)
        else:
            return queue_size >= self.config.max_batch_size or (
                queue_size > 0 and time_since_last_batch >= self.config.batch_timeout
            )
    
    def _should_create_dynamic_batch(self, queue_size: int, time_since_last_batch: float) -> bool:
        """Determine if we should create a batch using dynamic strategy."""
        # Adaptive batch size based on current performance
        current_latency = np.mean(self.processing_times[-10:]) if self.processing_times else 0.0
        target_latency = self.config.target_latency_ms
        
        # Increase batch size if latency is low, decrease if high
        if target_latency and current_latency < target_latency * 0.5:
            # Latency is very low, we can increase batch size
            adaptive_batch_size = min(self.config.max_batch_size * 1.5, self.config.max_batch_size)
        elif target_latency and current_latency > target_latency:
            # Latency is high, decrease batch size
            adaptive_batch_size = max(1, self.config.max_batch_size * 0.7)
        else:
            adaptive_batch_size = self.config.max_batch_size
        
        # Create batch conditions
        return (queue_size >= int(adaptive_batch_size) or
                (queue_size > 0 and time_since_last_batch >= self.config.batch_timeout))
    
    def _process_batch(self):
        """Process the current batch of requests."""
        if self.is_processing:
            return
        
        self.is_processing = True
        batch_start_time = time.time()
        batch_id = f"batch_{int(batch_start_time)}"
        
        try:
            # Collect requests for this batch
            batch_requests = []
            batch_timeout_reached = False
            
            # Always collect at least one request
            if not self.request_queue.empty():
                priority, timestamp, request = self.request_queue.get()
                batch_requests.append(request)
                del self.pending_requests[request.request_id]
            
            # Collect additional requests up to batch size
            while (len(batch_requests) < self.config.max_batch_size and 
                   not self.request_queue.empty()):
                try:
                    priority, timestamp, request = self.request_queue.get_nowait()
                    batch_requests.append(request)
                    del self.pending_requests[request.request_id]
                except queue.Empty:
                    break
            
            # Check timeout condition
            time_since_first = batch_start_time - batch_requests[0].timestamp
            if time_since_first >= self.config.batch_timeout:
                batch_timeout_reached = True
            
            if not batch_requests:
                self.is_processing = False
                return
            
            logger.info(f"Processing batch {batch_id} with {len(batch_requests)} requests")
            
            # Process the batch
            batch_results = self._execute_batch(batch_requests)
            
            # Store results
            self.completed_batches[batch_id] = {
                "results": batch_results,
                "timestamp": batch_start_time,
                "size": len(batch_requests),
                "processing_time": time.time() - batch_start_time
            }
            
            # Update metrics
            self._update_batch_metrics(batch_requests, batch_results, time.time() - batch_start_time)
            
            self.last_batch_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            
            # Create error results for all requests in the batch
            error_result = BatchResult(
                request_id="batch_error",
                output_data=None,
                processing_time_ms=0.0,
                batch_id=batch_id,
                error_message=str(e)
            )
            
            # Store error results
            self.completed_batches[batch_id] = {
                "results": [error_result],
                "timestamp": batch_start_time,
                "size": len(batch_requests),
                "processing_time": time.time() - batch_start_time
            }
        
        finally:
            self.is_processing = False
    
    def _execute_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Execute a batch of requests."""
        batch_start_time = time.time()
        results = []
        
        try:
            # Check cache first
            uncached_requests = []
            for request in requests:
                cache_key = self._get_cache_key(request)
                if cache_key in self.result_cache:
                    cached_result = self.result_cache[cache_key]
                    cached_result.request_id = request.request_id
                    cached_result.batch_id = f"cached_{int(batch_start_time)}"
                    results.append(cached_result)
                    self.cache_hits += 1
                else:
                    uncached_requests.append(request)
                    self.cache_misses += 1
            
            if not uncached_requests:
                return results
            
            # Process uncached requests
            if self.inference_function:
                # Use provided inference function
                batch_outputs = self.inference_function([req.input_data for req in uncached_requests])
                
                for i, (request, output) in enumerate(zip(uncached_requests, batch_outputs)):
                    result = BatchResult(
                        request_id=request.request_id,
                        output_data=output,
                        processing_time_ms=(time.time() - batch_start_time) * 1000 / len(uncached_requests),
                        batch_id=f"batch_{int(batch_start_time)}"
                    )
                    results.append(result)
                    
                    # Cache the result
                    cache_key = self._get_cache_key(request)
                    cached_result = BatchResult(
                        request_id=cache_key,
                        output_data=output,
                        processing_time_ms=result.processing_time_ms,
                        batch_id="cache"
                    )
                    self.result_cache[cache_key] = cached_result
            
            else:
                # Process requests individually if no batch function provided
                for request in uncached_requests:
                    try:
                        # Simulate processing
                        start_time = time.time()
                        time.sleep(0.01)  # Simulate processing time
                        processing_time = (time.time() - start_time) * 1000
                        
                        result = BatchResult(
                            request_id=request.request_id,
                            output_data=f"Processed: {request.input_data}",
                            processing_time_ms=processing_time,
                            batch_id=f"batch_{int(batch_start_time)}"
                        )
                        results.append(result)
                        
                        # Cache the result
                        cache_key = self._get_cache_key(request)
                        self.result_cache[cache_key] = BatchResult(
                            request_id=cache_key,
                            output_data=result.output_data,
                            processing_time_ms=processing_time,
                            batch_id="cache"
                        )
                        
                    except Exception as e:
                        error_result = BatchResult(
                            request_id=request.request_id,
                            output_data=None,
                            processing_time_ms=0.0,
                            batch_id=f"batch_{int(batch_start_time)}",
                            error_message=str(e)
                        )
                        results.append(error_result)
            
            # Limit cache size
            if len(self.result_cache) > 1000:
                # Remove oldest entries
                cache_items = list(self.result_cache.items())
                cache_items.sort(key=lambda x: x[1].processing_time_ms)
                self.result_cache = dict(cache_items[:500])
            
        except Exception as e:
            logger.error(f"Error executing batch: {e}")
            
            # Create error results for all uncached requests
            for request in uncached_requests:
                error_result = BatchResult(
                    request_id=request.request_id,
                    output_data=None,
                    processing_time_ms=0.0,
                    batch_id=f"batch_{int(batch_start_time)}",
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def _get_cache_key(self, request: BatchRequest) -> str:
        """Generate a cache key for a request."""
        import hashlib
        
        # Create a hash based on input data and relevant metadata
        data_str = str(request.input_data)
        metadata_str = str(sorted(request.metadata.items()))
        
        combined = f"{data_str}|{metadata_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _update_batch_metrics(self, 
                             requests: List[BatchRequest],
                             results: List[BatchResult],
                             processing_time_seconds: float):
        """Update batch processing metrics."""
        self.metrics.total_requests += len(requests)
        self.metrics.total_batches += 1
        
        # Update average metrics
        batch_size = len(requests)
        processing_time_ms = processing_time_seconds * 1000
        
        self.metrics.average_batch_size = (
            (self.metrics.average_batch_size * (self.metrics.total_batches - 1) + batch_size) 
            / self.metrics.total_batches
        )
        
        self.metrics.average_processing_time_ms = (
            (self.metrics.average_processing_time_ms * (self.metrics.total_batches - 1) + processing_time_ms)
            / self.metrics.total_batches
        )
        
        # Calculate throughput
        if processing_time_seconds > 0:
            throughput_rps = len(requests) / processing_time_seconds
            self.metrics.throughput_requests_per_second = (
                (self.metrics.throughput_requests_per_second * (self.metrics.total_batches - 1) + throughput_rps)
                / self.metrics.total_batches
            )
        
        # Track processing times
        self.processing_times.append(processing_time_ms)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # Calculate error rate
        error_count = sum(1 for result in results if result.error_message)
        error_rate = error_count / len(results) if results else 0.0
        self.metrics.error_rate = (
            (self.metrics.error_rate * (self.metrics.total_batches - 1) + error_rate)
            / self.metrics.total_batches
        )
        
        # Calculate cache hit rate
        if self.cache_hits + self.cache_misses > 0:
            self.metrics.cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
    
    def _metrics_collection_loop(self):
        """Background loop for collecting additional metrics."""
        while self.is_running:
            try:
                # Update memory usage
                if torch.cuda.is_available():
                    self.metrics.memory_usage_mb = torch.cuda.memory_allocated() / (1024**2)
                
                # Clean up old completed batches
                current_time = time.time()
                expired_batches = [
                    batch_id for batch_id, batch in self.completed_batches.items()
                    if current_time - batch["timestamp"] > 3600  # 1 hour
                ]
                
                for batch_id in expired_batches:
                    del self.completed_batches[batch_id]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(5)
    
    def _process_remaining_batches(self):
        """Process any remaining batches before shutdown."""
        logger.info("Processing remaining batches...")
        
        while not self.request_queue.empty():
            self._process_batch()
            time.sleep(0.1)  # Small delay between batches
    
    def optimize_batch_size(self, target_latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """Optimize batch size based on performance history."""
        if not self.processing_times:
            return {"recommendation": "insufficient_data"}
        
        current_avg_latency = np.mean(self.processing_times)
        
        # If no target specified, use current latency as baseline
        target = target_latency_ms or current_avg_latency
        
        # Calculate recommended batch size based on latency ratio
        latency_ratio = target / current_avg_latency if current_avg_latency > 0 else 1.0
        
        # Adjust batch size recommendation
        if latency_ratio < 0.5:
            # Target latency is much lower, reduce batch size
            recommended_batch_size = max(1, int(self.config.max_batch_size * 0.7))
            recommendation = "reduce_batch_size"
        elif latency_ratio > 1.5:
            # Target latency is much higher, increase batch size
            recommended_batch_size = min(self.config.max_batch_size * 2, 64)
            recommendation = "increase_batch_size"
        else:
            recommended_batch_size = self.config.max_batch_size
            recommendation = "maintain_batch_size"
        
        return {
            "recommendation": recommendation,
            "recommended_batch_size": recommended_batch_size,
            "current_avg_latency_ms": current_avg_latency,
            "target_latency_ms": target,
            "latency_ratio": latency_ratio,
            "current_batch_size": self.config.max_batch_size
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        
        report = {
            "metrics": self.metrics.to_dict(),
            "cache_stats": {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.result_cache)
            },
            "queue_status": {
                "queue_size": self.request_queue.qsize(),
                "pending_requests": len(self.pending_requests),
                "active_batches": len(self.active_batches),
                "completed_batches": len(self.completed_batches)
            },
            "performance_analysis": {
                "is_processing": self.is_processing,
                "recent_avg_latency_ms": np.mean(self.processing_times[-10:]) if self.processing_times else 0.0,
                "latency_trend": "stable" if len(self.processing_times) < 10 else (
                    "improving" if self.processing_times[-1] < self.processing_times[0] else "degrading"
                ),
                "throughput_trend": "stable"
            },
            "optimization_suggestions": []
        }
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions()
        report["optimization_suggestions"] = suggestions
        
        return report
    
    def _generate_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on current performance."""
        suggestions = []
        
        # Cache performance
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        
        if cache_hit_rate < 0.5:
            suggestions.append("Increase cache size or implement better cache key strategy")
        
        # Latency performance
        if self.processing_times:
            recent_avg = np.mean(self.processing_times[-10:])
            if recent_avg > 1000:  # >1 second average latency
                suggestions.append("Consider reducing batch size or optimizing inference function")
        
        # Error rate
        if self.metrics.error_rate > 0.1:  # >10% error rate
            suggestions.append("High error rate detected - check error logs and inference function")
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.memory_allocated() / (1024**2)
            if memory_usage_mb > 8000:  # >8GB GPU memory
                suggestions.append("High memory usage - consider reducing batch size or enabling memory optimization")
        
        # Queue management
        queue_size = self.request_queue.qsize()
        if queue_size > 100:
            suggestions.append("Large queue size detected - consider increasing processing resources")
        
        if not suggestions:
            suggestions.append("Performance is within acceptable parameters")
        
        return suggestions
    
    def reset_metrics(self):
        """Reset all performance metrics."""
        self.metrics = BatchMetrics()
        self.processing_times.clear()
        self.batch_history.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.result_cache.clear()
        
        logger.info("Batch processor metrics reset")


class AsyncBatchProcessor(BatchProcessor):
    """Asynchronous version of batch processor for high-concurrency scenarios."""
    
    def __init__(self, config: BatchConfig, async_inference_function: Optional[Callable] = None):
        super().__init__(config, None)
        self.async_inference_function = async_inference_function
        self.loop = None
    
    async def _execute_async_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Execute a batch of requests asynchronously."""
        if self.async_inference_function:
            batch_outputs = await self.async_inference_function([req.input_data for req in requests])
            
            results = []
            for request, output in zip(requests, batch_outputs):
                result = BatchResult(
                    request_id=request.request_id,
                    output_data=output,
                    processing_time_ms=0.0,  # Would need async timing
                    batch_id=f"async_batch_{int(time.time())}"
                )
                results.append(result)
            
            return results
        else:
            # Fallback to sync processing
            return self._execute_batch(requests)


class ChunkedBatchProcessor(BatchProcessor):
    """Batch processor that handles large inputs by chunking."""
    
    def __init__(self, config: BatchConfig, chunk_size: int = None):
        super().__init__(config)
        self.chunk_size = chunk_size or config.chunk_size
    
    def _execute_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Execute batch with chunking for large inputs."""
        # Check if any requests need chunking
        chunked_requests = []
        regular_requests = []
        
        for request in requests:
            if self._needs_chunking(request):
                chunked_requests.append(request)
            else:
                regular_requests.append(request)
        
        all_results = []
        
        # Process regular requests normally
        if regular_requests:
            regular_results = super()._execute_batch(regular_requests)
            all_results.extend(regular_results)
        
        # Process chunked requests
        for request in chunked_requests:
            chunk_results = self._process_chunked_request(request)
            all_results.extend(chunk_results)
        
        return all_results
    
    def _needs_chunking(self, request: BatchRequest) -> bool:
        """Check if a request needs chunking."""
        # Simple heuristic - check if input is too large
        if isinstance(request.input_data, (list, tuple)):
            return len(request.input_data) > self.chunk_size
        elif isinstance(request.input_data, str):
            return len(request.input_data) > self.chunk_size * 10  # Rough estimate
        else:
            return False
    
    def _process_chunked_request(self, request: BatchRequest) -> List[BatchResult]:
        """Process a single request by chunking it."""
        chunks = self._create_chunks(request)
        results = []
        
        for i, chunk in enumerate(chunks):
            # Create chunk request
            chunk_request = BatchRequest(
                request_id=f"{request.request_id}_chunk_{i}",
                input_data=chunk,
                metadata=request.metadata.copy()
            )
            
            # Process chunk
            chunk_results = super()._execute_batch([chunk_request])
            
            # Mark as part of original request
            for result in chunk_results:
                result.metadata["original_request_id"] = request.request_id
                result.metadata["chunk_index"] = i
                result.metadata["total_chunks"] = len(chunks)
            
            results.extend(chunk_results)
        
        return results
    
    def _create_chunks(self, request: BatchRequest) -> List[Any]:
        """Create chunks from request input."""
        if isinstance(request.input_data, (list, tuple)):
            return [request.input_data[i:i + self.chunk_size] 
                   for i in range(0, len(request.input_data), self.chunk_size)]
        elif isinstance(request.input_data, str):
            # Simple text chunking
            words = request.input_data.split()
            return [' '.join(words[i:i + self.chunk_size]) 
                   for i in range(0, len(words), self.chunk_size)]
        else:
            return [request.input_data]  # No chunking needed