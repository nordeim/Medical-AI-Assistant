"""
Model Serving Module with Adapter Support

This module provides a FastAPI-based model serving interface with adapter management,
load balancing, performance monitoring, and production-ready features.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import weakref

import torch
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import uvicorn

from adapter_manager import AdapterManager, create_adapter_manager_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('model_requests_total', 'Total model requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('model_request_duration_seconds', 'Request latency', ['method', 'endpoint'])
ACTIVE_ADAPTERS = Gauge('active_adapters_count', 'Number of active adapters')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency', ['adapter_id'])
THROUGHPUT = Counter('tokens_processed_total', 'Total tokens processed', ['adapter_id'])


class AdapterRequest(BaseModel):
    """Request model for adapter operations."""
    prompt: str = Field(..., description="Input text prompt", max_length=4096)
    max_tokens: int = Field(default=100, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    adapter_id: Optional[str] = Field(None, description="Specific adapter to use")
    stream: bool = Field(default=False, description="Whether to stream response")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms.",
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "adapter_id": "medical_assistant_v1"
            }
        }


class AdapterResponse(BaseModel):
    """Response model for adapter operations."""
    request_id: str
    response: str
    adapter_used: str
    tokens_generated: int
    latency_ms: float
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    adapters_loaded: int
    active_adapter: Optional[str]
    memory_usage_mb: float
    gpu_available: bool


class AdapterInfo(BaseModel):
    """Information about a loaded adapter."""
    adapter_id: str
    description: Optional[str]
    version: str
    loaded_at: datetime
    file_size_mb: float
    tags: List[str]
    performance_metrics: Dict[str, float]


class ServingMetrics:
    """Metrics collection and monitoring."""
    
    def __init__(self):
        self.request_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.adapter_usage: Dict[str, int] = {}
        
    def record_request(self, endpoint: str, latency: float, status_code: int):
        """Record request metrics."""
        method = "POST"  # Most operations are POST
        
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status_code)).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)
        
        if status_code >= 400:
            self.error_counts[endpoint] = self.error_counts.get(endpoint, 0) + 1
    
    def record_inference(self, adapter_id: str, latency: float, tokens: int):
        """Record inference metrics."""
        INFERENCE_LATENCY.labels(adapter_id=adapter_id).observe(latency)
        THROUGHPUT.labels(adapter_id=adapter_id).inc(tokens)
        self.adapter_usage[adapter_id] = self.adapter_usage.get(adapter_id, 0) + 1


class LoadBalancer:
    """Simple load balancer for multiple adapter instances."""
    
    def __init__(self):
        self.adapter_instances: Dict[str, Any] = {}
        self.current_index: Dict[str, int] = {}
        self.lock = asyncio.Lock()
        
    async def register_adapter(self, adapter_id: str, adapter_instance: Any):
        """Register an adapter instance."""
        async with self.lock:
            if adapter_id not in self.adapter_instances:
                self.adapter_instances[adapter_id] = []
                self.current_index[adapter_id] = 0
            self.adapter_instances[adapter_id].append(adapter_instance)
    
    async def get_adapter(self, adapter_id: str) -> Optional[Any]:
        """Get adapter instance using round-robin."""
        async with self.lock:
            if adapter_id not in self.adapter_instances:
                return None
            
            instances = self.adapter_instances[adapter_id]
            if not instances:
                return None
            
            index = self.current_index[adapter_id] % len(instances)
            self.current_index[adapter_id] = (index + 1) % len(instances)
            return instances[index]


class ModelServingManager:
    """
    Main model serving manager that orchestrates FastAPI app and adapter operations.
    """
    
    def __init__(self, 
                 base_model_id: str,
                 adapter_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 max_memory_mb: int = 8192):
        
        self.base_model_id = base_model_id
        self.adapter_configs = adapter_configs or {}
        self.host = host
        self.port = port
        
        # Core components
        self.adapter_manager: Optional[AdapterManager] = None
        self.metrics = ServingMetrics()
        self.load_balancer = LoadBalancer()
        
        # State
        self.is_running = False
        self.app: Optional[FastAPI] = None
        
        logger.info(f"ModelServingManager initialized for {base_model_id}")
    
    async def initialize(self):
        """Initialize the serving manager."""
        try:
            # Create adapter manager
            self.adapter_manager = await create_adapter_manager_async(
                base_model_id=self.base_model_id,
                max_memory_mb=max_memory_mb
            )
            
            # Load configured adapters
            for adapter_id, config in self.adapter_configs.items():
                adapter_path = config.get("path")
                if adapter_path:
                    try:
                        adapter = await self.adapter_manager.load_adapter_async(
                            adapter_path, adapter_id
                        )
                        await self.load_balancer.register_adapter(adapter_id, adapter)
                        logger.info(f"Loaded adapter: {adapter_id}")
                    except Exception as e:
                        logger.error(f"Failed to load adapter {adapter_id}: {str(e)}")
            
            # Set default active adapter
            if self.adapter_configs:
                first_adapter = next(iter(self.adapter_configs.keys()))
                self.adapter_manager.switch_adapter(first_adapter)
            
            # Update metrics
            adapters_count = len(self.adapter_manager.list_adapters())
            ACTIVE_ADAPTERS.set(adapters_count)
            
            logger.info("Model serving manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize serving manager: {str(e)}")
            raise
    
    async def generate_response(self, request: AdapterRequest) -> AdapterResponse:
        """Generate response using current or specified adapter."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            # Get active adapter or specified one
            adapter_id = request.adapter_id or self.adapter_manager._active_adapter_id
            if not adapter_id:
                raise HTTPException(status_code=400, detail="No adapter active")
            
            # Get adapter instance from load balancer
            adapter_instance = await self.load_balancer.get_adapter(adapter_id)
            if adapter_instance is None:
                # Use adapter manager's adapter
                active_adapter = self.adapter_manager.get_active_adapter()
                if not active_adapter or active_adapter[0] != adapter_id:
                    raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
                adapter_instance = active_adapter[1]
            
            # Generate response
            with torch.no_grad():
                # Tokenize input
                inputs = self.adapter_manager._base_tokenizer.encode(
                    request.prompt, return_tensors="pt"
                )
                
                if hasattr(adapter_instance, 'to'):
                    inputs = inputs.to(adapter_instance.device)
                
                # Generate
                outputs = adapter_instance.generate(
                    inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=True,
                    pad_token_id=self.adapter_manager._base_tokenizer.eos_token_id
                )
                
                # Decode response
                response = self.adapter_manager._base_tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # Extract generated portion
                prompt_len = len(self.adapter_manager._base_tokenizer.encode(request.prompt))
                generated_tokens = outputs[0][prompt_len:]
                response = self.adapter_manager._base_tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.metrics.record_inference(adapter_id, time.time() - start_time, len(generated_tokens))
            
            return AdapterResponse(
                request_id=request_id,
                response=response.strip(),
                adapter_used=adapter_id,
                tokens_generated=len(generated_tokens),
                latency_ms=latency_ms,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def list_adapters(self) -> List[AdapterInfo]:
        """List all loaded adapters with their information."""
        if not self.adapter_manager:
            return []
        
        adapters_info = []
        metadata_dict = self.adapter_manager.list_adapters()
        
        for adapter_id, metadata in metadata_dict.items():
            info = AdapterInfo(
                adapter_id=adapter_id,
                description=metadata.description,
                version=metadata.version,
                loaded_at=datetime.fromtimestamp(metadata.timestamp),
                file_size_mb=metadata.file_size / (1024 * 1024),
                tags=metadata.tags,
                performance_metrics=metadata.performance_metrics
            )
            adapters_info.append(info)
        
        return adapters_info
    
    async def hot_swap_adapter(self, new_adapter_id: str, timeout: float = 30.0) -> bool:
        """Perform hot-swap to new adapter."""
        if not self.adapter_manager:
            return False
        
        try:
            success = await self.adapter_manager.hot_swap_adapter(
                new_adapter_id, timeout=timeout
            )
            
            if success:
                ACTIVE_ADAPTERS.set(len(self.adapter_manager.list_adapters()))
                logger.info(f"Hot-swapped to adapter: {new_adapter_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Hot-swap failed: {str(e)}")
            return False
    
    def get_health_status(self) -> HealthResponse:
        """Get system health status."""
        if not self.adapter_manager:
            return HealthResponse(
                status="unhealthy",
                timestamp=datetime.now(),
                version="1.0.0",
                adapters_loaded=0,
                active_adapter=None,
                memory_usage_mb=0.0,
                gpu_available=torch.cuda.is_available()
            )
        
        # Get memory usage
        memory_stats = self.adapter_manager.memory_manager.get_memory_usage()
        active_adapter = self.adapter_manager._active_adapter_id
        
        return HealthResponse(
            status="healthy" if self.is_running else "unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            adapters_loaded=len(self.adapter_manager.list_adapters()),
            active_adapter=active_adapter,
            memory_usage_mb=memory_stats["rss_mb"],
            gpu_available=torch.cuda.is_available()
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if not self.adapter_manager:
            return {"error": "Manager not initialized"}
        
        return {
            "adapter_manager_stats": self.adapter_manager.get_performance_stats(),
            "serving_metrics": {
                "error_counts": dict(self.metrics.error_counts),
                "adapter_usage": dict(self.metrics.adapter_usage),
            },
            "system_info": {
                "gpu_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "memory_stats": self.adapter_manager.memory_manager.get_memory_usage()
            }
        }


# FastAPI application factory
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting model serving application...")
    
    # Initialize serving manager
    serving_manager = app.state.serving_manager
    await serving_manager.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down model serving application...")
    if serving_manager.adapter_manager:
        serving_manager.adapter_manager.cleanup()


def create_fastapi_app(manager: ModelServingManager) -> FastAPI:
    """Create FastAPI application with all routes and middleware."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await manager.initialize()
        app.state.serving_manager = manager
        manager.is_running = True
        yield
        manager.is_running = False
        if manager.adapter_manager:
            manager.adapter_manager.cleanup()
    
    app = FastAPI(
        title="Adapter-Powered Model Serving",
        description="Production-ready model serving with adapter management",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*"]  # Configure appropriately for production
    )
    
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add processing time header."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Routes
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check(manager: ModelServingManager = Depends(get_manager)):
        """Health check endpoint."""
        return manager.get_health_status()
    
    @app.post("/generate", response_model=AdapterResponse)
    async def generate_text(
        request: AdapterRequest,
        background_tasks: BackgroundTasks,
        manager: ModelServingManager = Depends(get_manager)
    ):
        """Generate text using active adapter."""
        start_time = time.time()
        
        try:
            response = await manager.generate_response(request)
            
            # Record metrics
            latency = time.time() - start_time
            manager.metrics.record_request("/generate", latency, 200)
            
            return response
            
        except HTTPException:
            manager.metrics.record_request("/generate", time.time() - start_time, 500)
            raise
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            manager.metrics.record_request("/generate", time.time() - start_time, 500)
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/adapters", response_model=List[AdapterInfo])
    async def list_adapters(manager: ModelServingManager = Depends(get_manager)):
        """List all loaded adapters."""
        return await manager.list_adapters()
    
    @app.post("/adapters/{adapter_id}/switch")
    async def switch_adapter(
        adapter_id: str,
        manager: ModelServingManager = Depends(get_manager)
    ):
        """Switch to a different adapter."""
        try:
            success = manager.adapter_manager.switch_adapter(adapter_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Adapter {adapter_id} not found")
            
            return {"message": f"Switched to adapter: {adapter_id}"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/adapters/{adapter_id}/hot-swap")
    async def hot_swap_adapter_endpoint(
        adapter_id: str,
        timeout: float = 30.0,
        manager: ModelServingManager = Depends(get_manager)
    ):
        """Perform hot-swap to new adapter."""
        try:
            success = await manager.hot_swap_adapter(adapter_id, timeout)
            if not success:
                raise HTTPException(status_code=400, detail="Hot-swap failed")
            
            return {"message": f"Hot-swapped to adapter: {adapter_id}"}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics")
    async def get_metrics(manager: ModelServingManager = Depends(get_manager)):
        """Get performance metrics."""
        return manager.get_performance_stats()
    
    @app.get("/metrics/prometheus")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    
    @app.get("/info")
    async def get_info(manager: ModelServingManager = Depends(get_manager)):
        """Get server information."""
        return {
            "model_id": manager.base_model_id,
            "version": "1.0.0",
            "adapters_loaded": len(manager.adapter_manager.list_adapters()) if manager.adapter_manager else 0,
            "active_adapter": manager.adapter_manager._active_adapter_id if manager.adapter_manager else None,
            "timestamp": datetime.now().isoformat()
        }
    
    return app


# Dependency for getting serving manager
async def get_manager(request: Request) -> ModelServingManager:
    """Get serving manager from request state."""
    return request.app.state.serving_manager


class ServingLauncher:
    """Production-ready serving launcher."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.app: Optional[FastAPI] = None
        self.manager: Optional[ModelServingManager] = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load serving configuration."""
        import yaml
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    async def setup(self):
        """Setup serving infrastructure."""
        config = self.load_config()
        
        # Create manager
        self.manager = ModelServingManager(
            base_model_id=config["model"]["base_model_id"],
            adapter_configs=config.get("adapters", {}),
            host=config.get("server", {}).get("host", "0.0.0.0"),
            port=config.get("server", {}).get("port", 8000),
            max_memory_mb=config.get("resources", {}).get("max_memory_mb", 8192)
        )
        
        # Create FastAPI app
        self.app = create_fastapi_app(self.manager)
    
    def run(self, 
            host: Optional[str] = None, 
            port: Optional[int] = None,
            workers: int = 1,
            reload: bool = False):
        """Run the serving application."""
        if not self.app:
            raise RuntimeError("Application not setup. Call setup() first.")
        
        uvicorn.run(
            self.app,
            host=host or self.manager.host,
            port=port or self.manager.port,
            workers=workers,
            reload=reload,
            log_level="info"
        )


# Utility functions
def create_serving_launcher(config_path: str) -> ServingLauncher:
    """Create serving launcher from config."""
    return ServingLauncher(config_path)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Sample config
    config = {
        "model": {
            "base_model_id": "microsoft/DialoGPT-medium"
        },
        "adapters": {
            "assistant_v1": {
                "path": "./adapters/assistant_v1",
                "description": "General assistant adapter"
            },
            "medical_v1": {
                "path": "./adapters/medical_v1", 
                "description": "Medical domain adapter"
            }
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "resources": {
            "max_memory_mb": 8192
        }
    }
    
    # Save sample config
    with open("serving_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create launcher
    launcher = create_serving_launcher("serving_config.yaml")
    
    # Run server
    # launcher.run(reload=True)  # Development mode
    # launcher.run(workers=4)    # Production mode