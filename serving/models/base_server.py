"""
Base classes and interfaces for the model serving infrastructure.
Provides common functionality for all model servers.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import uuid
from datetime import datetime, timedelta

import torch
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException
import structlog

from ..config.logging_config import get_logger, get_model_logger
from ..config.settings import get_settings


class ModelStatus(Enum):
    """Model status enumeration."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"


class PredictionType(Enum):
    """Types of model predictions."""
    TEXT_GENERATION = "text_generation"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    QUESTION_ANSWERING = "question_answering"
    CONVERSATION = "conversation"


@dataclass
class ModelMetadata:
    """Metadata for a loaded model."""
    model_id: str
    name: str
    version: str
    prediction_type: PredictionType
    input_shape: Optional[tuple] = None
    output_shape: Optional[tuple] = None
    max_length: int = 512
    load_time: Optional[float] = None
    memory_usage: Optional[int] = None
    device: Optional[str] = None
    quantization: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: Optional[str] = None


@dataclass
class PredictionRequest:
    """Request for model prediction."""
    request_id: str
    model_id: str
    inputs: Union[str, List[str], Dict[str, Any]]
    parameters: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class PredictionResponse:
    """Response from model prediction."""
    request_id: str
    model_id: str
    outputs: Union[str, List[str], Dict[str, Any]]
    confidence: Optional[float] = None
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Metrics for a model server."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    active_requests: int = 0
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0


class ModelCache:
    """Cache for model predictions and responses."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.logger = structlog.get_logger("model_cache")
    
    def _generate_key(self, model_id: str, inputs: str, 
                     parameters: Dict[str, Any]) -> str:
        """Generate cache key."""
        import hashlib
        import json
        
        cache_data = {
            "model_id": model_id,
            "inputs": inputs,
            "parameters": parameters
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get(self, model_id: str, inputs: str, 
            parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached response."""
        key = self._generate_key(model_id, inputs, parameters)
        
        if key in self.cache:
            # Check if entry is still valid
            cache_time = self.access_times.get(key)
            if cache_time and datetime.utcnow() - cache_time < timedelta(seconds=self.ttl):
                # Update access time and return
                self.access_times[key] = datetime.utcnow()
                return self.cache[key]
            else:
                # Remove expired entry
                self._remove_key(key)
        
        return None
    
    def set(self, model_id: str, inputs: str, parameters: Dict[str, Any], 
            response: Dict[str, Any]) -> None:
        """Cache response."""
        key = self._generate_key(model_id, inputs, parameters)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = response
        self.access_times[key] = datetime.utcnow()
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
    
    def _evict_oldest(self) -> None:
        """Remove oldest accessed item."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: self.access_times[k])
        self._remove_key(oldest_key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0
        }


class BaseModelServer(ABC):
    """Base class for all model servers."""
    
    def __init__(self, model_id: str, name: str, prediction_type: PredictionType):
        self.model_id = model_id
        self.name = name
        self.prediction_type = prediction_type
        self.status = ModelStatus.LOADING
        self.metadata = None
        self.metrics = ModelMetrics()
        self.cache = ModelCache()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.model_logger = get_model_logger(self.name)
        self.settings = get_settings()
        self._model = None
        self._tokenizer = None
        self._device = None
    
    @abstractmethod
    async def load_model(self) -> None:
        """Load the ML model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make a prediction. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelMetadata:
        """Get model metadata. Must be implemented by subclasses."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the model server."""
        try:
            self.logger.info(f"Initializing model server: {self.name}")
            start_time = time.time()
            
            # Load the model
            await self.load_model()
            
            # Set status to ready
            load_time = time.time() - start_time
            self.status = ModelStatus.READY
            self.metadata = self.get_model_info()
            self.metadata.load_time = load_time
            
            self.logger.info(
                f"Model server initialized successfully: {self.name}",
                model_id=self.model_id,
                load_time=load_time
            )
            self.model_logger.log_model_load(self.metadata.name, True, load_time)
            
        except Exception as e:
            self.status = ModelStatus.ERROR
            self.logger.error(
                f"Failed to initialize model server: {self.name}",
                error=str(e),
                exc_info=True
            )
            self.model_logger.log_model_load(self.metadata.name if self.metadata else "unknown", 
                                           False, error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": self.status.value,
            "model_id": self.model_id,
            "name": self.name,
            "prediction_type": self.prediction_type.value,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.successful_requests / max(self.metrics.total_requests, 1),
                "average_response_time": self.metrics.average_response_time,
                "active_requests": self.metrics.active_requests
            },
            "cache": self.cache.get_stats(),
            "memory_usage_mb": self.metrics.memory_usage_mb
        }
    
    def update_metrics(self, processing_time: float, success: bool) -> None:
        """Update model metrics."""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
        
        # Update average response time
        alpha = 0.1  # Exponential moving average
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = processing_time
        else:
            self.metrics.average_response_time = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics.average_response_time
            )
        
        self.metrics.last_request_time = datetime.utcnow()
    
    def get_cached_response(self, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Get cached response if available."""
        if not self.settings.cache.enable_model_cache:
            return None
        
        inputs_str = str(request.inputs)
        cached_response = self.cache.get(
            self.model_id, 
            inputs_str, 
            request.parameters
        )
        
        if cached_response:
            self.metrics.cache_hits += 1
            return PredictionResponse(**cached_response)
        
        self.metrics.cache_misses += 1
        return None
    
    def cache_response(self, request: PredictionRequest, response: PredictionResponse) -> None:
        """Cache response."""
        if not self.settings.cache.enable_model_cache:
            return
        
        inputs_str = str(request.inputs)
        response_dict = {
            "request_id": response.request_id,
            "model_id": response.model_id,
            "outputs": response.outputs,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "timestamp": response.timestamp.isoformat(),
            "metadata": response.metadata
        }
        
        self.cache.set(self.model_id, inputs_str, request.parameters, response_dict)


class ConcurrentModelServer(BaseModelServer):
    """Model server with concurrency control."""
    
    def __init__(self, model_id: str, name: str, prediction_type: PredictionType,
                 max_concurrent_requests: int = 10):
        super().__init__(model_id, name, prediction_type)
        self.max_concurrent_requests = max_concurrent_requests
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.active_requests: Dict[str, asyncio.Task] = {}
    
    async def process_with_concurrency(self, request: PredictionRequest, 
                                      prediction_func) -> PredictionResponse:
        """Process request with concurrency control."""
        # Check concurrency limit
        if len(self.active_requests) >= self.max_concurrent_requests:
            raise HTTPException(
                status_code=503, 
                detail="Server is at maximum capacity"
            )
        
        # Acquire semaphore
        async with self.semaphore:
            request_task = asyncio.current_task()
            if request_task:
                self.active_requests[request.request_id] = request_task
            
            try:
                return await prediction_func(request)
            finally:
                # Clean up active request
                self.active_requests.pop(request.request_id, None)
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request."""
        task = self.active_requests.get(request_id)
        if task and not task.done():
            task.cancel()
            self.active_requests.pop(request_id, None)
            return True
        return False
    
    def get_active_requests(self) -> List[str]:
        """Get list of active request IDs."""
        return list(self.active_requests.keys())


class ModelRegistry:
    """Registry for managing multiple model servers."""
    
    def __init__(self):
        self.models: Dict[str, BaseModelServer] = {}
        self.logger = get_logger("model_registry")
    
    def register_model(self, model: BaseModelServer) -> None:
        """Register a model server."""
        self.models[model.model_id] = model
        self.logger.info(f"Registered model: {model.name}", model_id=model.model_id)
    
    def get_model(self, model_id: str) -> Optional[BaseModelServer]:
        """Get model server by ID."""
        return self.models.get(model_id)
    
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models."""
        return [model.get_model_info() for model in self.models.values()]
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check for all models."""
        results = {}
        for model_id, model in self.models.items():
            try:
                results[model_id] = await model.health_check()
            except Exception as e:
                results[model_id] = {
                    "status": "error",
                    "error": str(e)
                }
        return results


# Global model registry instance
model_registry = ModelRegistry()