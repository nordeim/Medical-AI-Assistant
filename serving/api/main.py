"""
FastAPI application for the model serving infrastructure.
Provides RESTful API endpoints with proper security and medical data protection.
"""

from typing import Dict, List, Optional, Any, Union
import time
import uuid
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, HTTPException, Request, Response, Depends, 
    status, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import structlog

from ..config.settings import get_settings
from ..config.logging_config import (
    get_logger, get_audit_logger, LoggingContextManager,
    request_id_var, user_id_var, session_id_var
)
from ..models.base_server import model_registry, PredictionRequest, PredictionResponse


# Security
security = HTTPBearer(auto_error=False)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0.0", description="API version")
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    uptime_seconds: float = Field(default=0.0)


class PredictionRequestModel(BaseModel):
    """API request model for predictions."""
    inputs: Union[str, List[str], Dict[str, Any]] = Field(
        ..., 
        description="Model inputs"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model parameters"
    )
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    @validator('inputs')
    def validate_inputs(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("Input text cannot be empty")
        if isinstance(v, list) and not v:
            raise ValueError("Input list cannot be empty")
        return v


class PredictionResponseModel(BaseModel):
    """API response model for predictions."""
    request_id: str
    model_id: str
    outputs: Union[str, List[str], Dict[str, Any]]
    confidence: Optional[float] = None
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MetricsResponse(BaseModel):
    """Metrics response model."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    average_response_time: float = 0.0
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


# Global state
start_time = time.time()
settings = get_settings()
logger = get_logger("api")
audit_logger = get_audit_logger()


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for authentication."""
    if not settings.serving.api_key:
        return None
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    if credentials.credentials != settings.serving.api_key:
        audit_logger.log_access(
            user_id="unknown",
            action="auth_failed",
            resource="api",
            details={"reason": "invalid_api_key"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials


async def rate_limit_check(request: Request):
    """Rate limiting middleware."""
    # Simple in-memory rate limiting (use Redis in production)
    client_ip = request.client.host
    current_time = time.time()
    
    # This is a simplified implementation
    # In production, use a proper rate limiting solution like Redis
    if not hasattr(rate_limit_check, "requests"):
        rate_limit_check.requests = {}
    
    if client_ip not in rate_limit_check.requests:
        rate_limit_check.requests[client_ip] = []
    
    # Remove old requests (older than 1 minute)
    rate_limit_check.requests[client_ip] = [
        req_time for req_time in rate_limit_check.requests[client_ip]
        if current_time - req_time < 60
    ]
    
    # Check rate limit
    if len(rate_limit_check.requests[client_ip]) >= settings.serving.rate_limit_per_minute:
        audit_logger.log_access(
            user_id=client_ip,
            action="rate_limit_exceeded",
            resource="api",
            details={"requests_count": len(rate_limit_check.requests[client_ip])}
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Add current request
    rate_limit_check.requests[client_ip].append(current_time)


async def medical_data_validation(request_data: Dict[str, Any]):
    """Validate and sanitize medical data."""
    if not settings.medical.phi_redaction:
        return request_data
    
    # Redact sensitive patterns
    sensitive_patterns = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    }
    
    import re
    
    for key, value in request_data.items():
        if isinstance(value, str):
            for pattern_name, pattern in sensitive_patterns.items():
                request_data[key] = re.sub(pattern, '[REDACTED]', value, flags=re.IGNORECASE)
    
    return request_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting model serving API")
    
    # Initialize models (would be done in a separate initialization step)
    # For now, just log that we're ready
    yield
    
    # Shutdown
    logger.info("Shutting down model serving API")


# Create FastAPI app
app = FastAPI(
    title="Medical AI Model Serving API",
    description="Secure API for serving ML models with medical data compliance",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.serving.allowed_origins,
    allow_credentials=settings.serving.cors_allow_credentials,
    allow_methods=settings.serving.cors_allow_methods,
    allow_headers=settings.serving.cors_allow_headers,
)

if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1"]  # Configure based on deployment
    )


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request/response logging middleware."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Set request context
    with LoggingContextManager(request_id=request_id):
        # Log request
        logger.info(
            "Request received",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent")
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            processing_time = time.time() - start_time
            logger.info(
                "Request completed",
                status_code=response.status_code,
                processing_time_ms=round(processing_time * 1000, 2)
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = str(round(processing_time * 1000, 2))
            
            return response
        
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                "Request failed",
                error=str(e),
                processing_time_ms=round(processing_time * 1000, 2),
                exc_info=True
            )
            raise


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check model status
        model_status = await model_registry.health_check_all()
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            models=model_status,
            uptime_seconds=time.time() - start_time
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Metrics"])
async def get_metrics():
    """Get system metrics."""
    try:
        metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "average_response_time": 0.0,
            "models": {}
        }
        
        # Aggregate metrics from all models
        for model_id, model in model_registry.models.items():
            model_metrics = await model.health_check()
            metrics["models"][model_id] = model_metrics["metrics"]
            
            # Aggregate totals
            model_health = model_metrics["metrics"]
            metrics["requests_total"] += model_health["total_requests"]
            metrics["requests_successful"] += model_health["total_requests"] * model_health.get("success_rate", 0)
            metrics["requests_failed"] += model_health["total_requests"] * (1 - model_health.get("success_rate", 0))
            metrics["average_response_time"] += model_health["average_response_time"]
        
        # Calculate averages
        if len(model_registry.models) > 0:
            metrics["average_response_time"] /= len(model_registry.models)
        
        return MetricsResponse(**metrics)
    
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/predict", 
          response_model=PredictionResponseModel,
          tags=["Predictions"])
async def predict(
    model_id: str,
    request: PredictionRequestModel,
    background_tasks: BackgroundTasks,
    credentials: Optional[str] = Depends(verify_api_key)
):
    """
    Make a prediction using the specified model.
    
    This endpoint processes input data through the specified ML model
    and returns the generated predictions.
    """
    # Rate limiting
    await rate_limit_check(request)
    
    # Get model
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    if model.status.value != "ready":
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' is not ready"
        )
    
    # Validate request size
    import json
    request_size = len(json.dumps(request.dict()))
    if request_size > settings.serving.max_request_size:
        raise HTTPException(
            status_code=413,
            detail="Request too large"
        )
    
    # Validate medical data
    request_data = await medical_data_validation(request.dict())
    
    # Create prediction request
    prediction_request = PredictionRequest(
        request_id=str(uuid.uuid4()),
        model_id=model_id,
        inputs=request.inputs,
        parameters=request.parameters or {},
        user_id=request.user_id,
        session_id=request.session_id
    )
    
    # Set request context
    with LoggingContextManager(
        request_id=prediction_request.request_id,
        user_id=request.user_id,
        session_id=request.session_id
    ):
        try:
            # Audit log for predictions
            if settings.medical.enable_audit_log:
                background_tasks.add_task(
                    audit_logger.log_access,
                    user_id=request.user_id or "anonymous",
                    action="prediction_request",
                    resource=f"model:{model_id}",
                    details={
                        "input_type": type(request.inputs).__name__,
                        "parameter_count": len(request.parameters or {})
                    }
                )
            
            # Make prediction
            response = await model.predict(prediction_request)
            
            # Convert to API response format
            return PredictionResponseModel(
                request_id=response.request_id,
                model_id=response.model_id,
                outputs=response.outputs,
                confidence=response.confidence,
                processing_time=response.processing_time,
                timestamp=response.timestamp,
                metadata=response.metadata
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Prediction failed",
                model_id=model_id,
                request_id=prediction_request.request_id,
                error=str(e)
            )
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )


@app.get("/models", response_model=List[Dict[str, Any]], tags=["Models"])
async def list_models():
    """List all available models."""
    try:
        models = []
        for model_metadata in model_registry.list_models():
            models.append({
                "model_id": model_metadata.model_id,
                "name": model_metadata.name,
                "version": model_metadata.version,
                "prediction_type": model_metadata.prediction_type.value,
                "max_length": model_metadata.max_length,
                "device": model_metadata.device,
                "quantization": model_metadata.quantization,
                "tags": model_metadata.tags,
                "description": model_metadata.description
            })
        
        return models
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}/info", tags=["Models"])
async def get_model_info(model_id: str):
    """Get detailed information about a specific model."""
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    try:
        metadata = model.get_model_info()
        health = await model.health_check()
        
        return {
            "metadata": {
                "model_id": metadata.model_id,
                "name": metadata.name,
                "version": metadata.version,
                "prediction_type": metadata.prediction_type.value,
                "max_length": metadata.max_length,
                "device": metadata.device,
                "quantization": metadata.quantization,
                "tags": metadata.tags,
                "description": metadata.description,
                "load_time": metadata.load_time,
                "memory_usage": metadata.memory_usage
            },
            "health": health,
            "metrics": model.metrics
        }
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}/health", tags=["Models"])
async def get_model_health(model_id: str):
    """Get health status for a specific model."""
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    try:
        return await model.health_check()
    
    except Exception as e:
        logger.error(f"Failed to get model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}/cache", tags=["Cache"])
async def clear_model_cache(
    model_id: str,
    credentials: Optional[str] = Depends(verify_api_key)
):
    """Clear cache for a specific model."""
    model = model_registry.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found"
        )
    
    try:
        model.cache.clear()
        
        logger.info(
            f"Cache cleared for model",
            model_id=model_id,
            user_id=credentials
        )
        
        return {"message": f"Cache cleared for model {model_id}"}
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "detail": exc.detail,
            "request_id": request_id_var.get(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        path=request.url.path,
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "request_id": request_id_var.get(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.serving.host,
        port=settings.serving.port,
        reload=settings.environment == "development",
        workers=settings.serving.workers if settings.environment == "production" else 1
    )
