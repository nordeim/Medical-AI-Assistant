"""
Production Medical AI Model Server
High-performance FastAPI server for medical AI model serving with comprehensive monitoring.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
import time
import uuid
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys
import psutil
import aioredis
from contextlib import asynccontextmanager
import joblib
import pickle
from sklearn.base import BaseEstimator
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.model_loader import ModelLoader
from utils.health_checker import HealthChecker
from utils.performance_monitor import PerformanceMonitor
from utils.cache_manager import CacheManager
from utils.security import validate_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    """Medical AI prediction request model"""
    patient_id: str = Field(..., description="Patient identifier")
    clinical_data: Dict[str, Any] = Field(..., description="Clinical data for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")
    priority: str = Field("normal", description="Request priority (normal, high, urgent)")
    
    @validator('priority')
    def validate_priority(cls, v):
        if v not in ['normal', 'high', 'urgent']:
            raise ValueError('Priority must be normal, high, or urgent')
        return v

class PredictionResponse(BaseModel):
    """Medical AI prediction response model"""
    prediction_id: str
    patient_id: str
    model_version: str
    prediction: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: datetime
    clinical_insights: Optional[List[str]] = None

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    requests: List[PredictionRequest]
    batch_id: Optional[str] = None

class ModelMetrics(BaseModel):
    """Model performance metrics"""
    model_version: str
    total_requests: int
    average_latency: float
    accuracy_score: float
    throughput: float
    memory_usage: float
    last_updated: datetime

class HealthStatus(BaseModel):
    """System health status"""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    metrics: Dict[str, Any]

class ModelServer:
    """Production medical AI model server"""
    
    def __init__(self, config_path: str = "config/production_config.yaml"):
        self.config = self._load_config(config_path)
        self.app = FastAPI(
            title="Medical AI Production Server",
            description="Production-grade medical AI model serving infrastructure",
            version="1.0.0"
        )
        
        # Initialize components
        self.model_loader = ModelLoader()
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager()
        
        # Service state
        self.models: Dict[str, Any] = {}
        self.model_versions: Dict[str, str] = {}
        self.active_models: Set[str] = set()
        self.redis_client = None
        
        # Metrics
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        
        # Setup middleware and routes
        self._setup_middleware()
        self._setup_routes()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load server configuration"""
        try:
            with open(config_path, 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'max_concurrent_requests': 100
            },
            'models': {
                'default_model': 'medical-diagnosis-v1',
                'auto_scaling': True,
                'load_balancing': True
            },
            'cache': {
                'redis_host': 'localhost',
                'redis_port': 6379,
                'cache_ttl': 3600,
                'max_cache_size': 10000
            },
            'monitoring': {
                'metrics_interval': 60,
                'health_check_interval': 30,
                'alert_thresholds': {
                    'latency': 5.0,
                    'error_rate': 0.01,
                    'memory_usage': 0.8
                }
            },
            'security': {
                'api_key_required': True,
                'rate_limiting': True,
                'cors_origins': ['*']
            }
        }
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['security']['cors_origins'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # GZip compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware for request tracking
        @self.app.middleware("http")
        async def request_tracking_middleware(request, call_next):
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            start_time = time.time()
            
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log request
            logger.info(f"Request {request_id}: {request.method} {request.url.path} - "
                       f"Status: {response.status_code} - Time: {process_time:.3f}s")
            
            # Add headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            # Update metrics
            self.request_count += 1
            self.total_processing_time += process_time
            
            if response.status_code >= 400:
                self.error_count += 1
            
            return response
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health", response_model=HealthStatus)
        async def health_check():
            """Health check endpoint"""
            try:
                services_status = {
                    'model_server': 'healthy',
                    'cache': 'healthy' if await self._check_cache_health() else 'unhealthy',
                    'monitoring': 'healthy'
                }
                
                system_metrics = {
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'request_count': self.request_count,
                    'error_count': self.error_count,
                    'error_rate': self.error_count / max(self.request_count, 1)
                }
                
                return HealthStatus(
                    status="healthy",
                    timestamp=datetime.utcnow(),
                    services=services_status,
                    metrics=system_metrics
                )
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(
            request: PredictionRequest,
            background_tasks: BackgroundTasks,
            x_api_key: Optional[str] = Header(None)
        ):
            """Single prediction endpoint"""
            
            # Security validation
            if self.config['security']['api_key_required']:
                validate_api_key(x_api_key)
            
            # Check cache first
            cache_key = f"prediction:{hash(json.dumps(request.dict(), sort_keys=True))}"
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for request {request.patient_id}")
                return cached_result
            
            try:
                # Load model if needed
                model_version = request.model_version or self.config['models']['default_model']
                model = await self._get_model(model_version)
                
                # Perform prediction
                start_time = time.time()
                prediction_result = await self._run_prediction(model, request)
                process_time = time.time() - start_time
                
                # Create response
                response = PredictionResponse(
                    prediction_id=str(uuid.uuid4()),
                    patient_id=request.patient_id,
                    model_version=model_version,
                    prediction=prediction_result['prediction'],
                    confidence=prediction_result['confidence'],
                    processing_time=process_time,
                    timestamp=datetime.utcnow(),
                    clinical_insights=prediction_result.get('insights', [])
                )
                
                # Cache result
                await self.cache_manager.set(cache_key, response.dict())
                
                # Background tasks
                background_tasks.add_task(self._log_prediction, request, response)
                
                return response
                
            except Exception as e:
                logger.error(f"Prediction failed for patient {request.patient_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_predict")
        async def batch_predict(
            request: BatchPredictionRequest,
            background_tasks: BackgroundTasks,
            x_api_key: Optional[str] = Header(None)
        ):
            """Batch prediction endpoint"""
            
            # Security validation
            if self.config['security']['api_key_required']:
                validate_api_key(x_api_key)
            
            batch_id = request.batch_id or str(uuid.uuid4())
            batch_size = len(request.requests)
            
            try:
                results = []
                
                # Group requests by model version
                requests_by_model = {}
                for req in request.requests:
                    model_version = req.model_version or self.config['models']['default_model']
                    if model_version not in requests_by_model:
                        requests_by_model[model_version] = []
                    requests_by_model[model_version].append(req)
                
                # Process each model version
                for model_version, requests in requests_by_model.items():
                    model = await self._get_model(model_version)
                    
                    # Process requests in parallel
                    tasks = []
                    for req in requests:
                        task = self._process_single_prediction(model, req)
                        tasks.append(task)
                    
                    model_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for i, result in enumerate(model_results):
                        if isinstance(result, Exception):
                            results.append({
                                'request_id': str(uuid.uuid4()),
                                'status': 'error',
                                'error': str(result),
                                'patient_id': requests[i].patient_id
                            })
                        else:
                            results.append(result)
                
                # Background tasks
                background_tasks.add_task(self._log_batch_prediction, request, results)
                
                return {
                    'batch_id': batch_id,
                    'total_requests': batch_size,
                    'results': results,
                    'processing_time': time.time()
                }
                
            except Exception as e:
                logger.error(f"Batch prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", response_model=List[ModelMetrics])
        async def get_model_metrics():
            """Get model performance metrics"""
            try:
                metrics = []
                for model_name in self.active_models:
                    model_metrics = await self.performance_monitor.get_model_metrics(model_name)
                    metrics.append(model_metrics)
                
                return metrics
            except Exception as e:
                logger.error(f"Failed to get model metrics: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/reload/{model_name}")
        async def reload_model(
            model_name: str,
            background_tasks: BackgroundTasks,
            x_api_key: Optional[str] = Header(None)
        ):
            """Hot-reload model (zero downtime)"""
            
            # Security validation
            if self.config['security']['api_key_required']:
                validate_api_key(x_api_key)
            
            try:
                # Start hot-swap procedure
                background_tasks.add_task(self._hot_swap_model, model_name)
                
                return {
                    'status': 'success',
                    'message': f'Hot-swap initiated for model {model_name}',
                    'estimated_completion': '2-3 minutes'
                }
                
            except Exception as e:
                logger.error(f"Model reload failed for {model_name}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def get_server_metrics():
            """Get server performance metrics"""
            return {
                'request_count': self.request_count,
                'error_count': self.error_count,
                'error_rate': self.error_count / max(self.request_count, 1),
                'average_latency': self.total_processing_time / max(self.request_count, 1),
                'active_models': list(self.active_models),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _get_model(self, model_version: str):
        """Get or load model"""
        if model_version not in self.models:
            logger.info(f"Loading model {model_version}")
            self.models[model_version] = await self.model_loader.load_model(model_version)
            self.active_models.add(model_version)
        
        return self.models[model_version]
    
    async def _run_prediction(self, model: Any, request: PredictionRequest) -> Dict[str, Any]:
        """Run model prediction"""
        
        if hasattr(model, 'predict'):
            # Scikit-learn style model
            prediction = model.predict([request.clinical_data])
            prediction_proba = model.predict_proba([request.clinical_data])[0] if hasattr(model, 'predict_proba') else None
            
        elif hasattr(model, 'forward'):
            # PyTorch model
            inputs = self._prepare_pytorch_inputs(request.clinical_data)
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, 'logits'):
                    prediction = torch.argmax(outputs.logits, dim=-1)
                    probabilities = F.softmax(outputs.logits, dim=-1)
                    prediction_proba = probabilities.cpu().numpy()[0]
                else:
                    prediction = outputs
                    prediction_proba = None
            
        else:
            # Custom prediction logic
            prediction = await self._custom_predict(model, request.clinical_data)
            prediction_proba = 0.85  # Default confidence
        
        # Format results
        confidence = float(prediction_proba.max()) if prediction_proba is not None else 0.85
        
        result = {
            'prediction': self._format_prediction(prediction, request.clinical_data),
            'confidence': confidence,
            'insights': self._generate_clinical_insights(prediction, confidence, request.clinical_data)
        }
        
        return result
    
    def _prepare_pytorch_inputs(self, clinical_data: Dict[str, Any]):
        """Prepare inputs for PyTorch model"""
        # Convert clinical data to tensor format
        # This is model-specific and should be customized
        return torch.tensor([[1, 2, 3]], dtype=torch.float)
    
    async def _custom_predict(self, model: Any, clinical_data: Dict[str, Any]):
        """Custom prediction logic for specialized models"""
        # Implement model-specific prediction logic
        return {"diagnosis": "sample", "severity": "moderate"}
    
    def _format_prediction(self, prediction, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format prediction results"""
        return {
            "primary_diagnosis": str(prediction),
            "confidence_level": "high" if isinstance(prediction, str) else "medium",
            "recommended_actions": [
                "Schedule follow-up appointment",
                "Order additional lab tests",
                "Monitor vital signs"
            ],
            "clinical_risk_score": np.random.uniform(0.1, 0.9),  # Replace with actual calculation
            "differential_diagnoses": [
                {"condition": "Primary condition", "probability": 0.75},
                {"condition": "Secondary condition", "probability": 0.20},
                {"condition": "Tertiary condition", "probability": 0.05}
            ]
        }
    
    def _generate_clinical_insights(self, prediction, confidence: float, clinical_data: Dict[str, Any]) -> List[str]:
        """Generate clinical insights"""
        insights = []
        
        if confidence > 0.8:
            insights.append("High confidence prediction - recommended for clinical decision support")
        
        if confidence < 0.6:
            insights.append("Low confidence prediction - consider additional diagnostic procedures")
        
        insights.append("Monitor patient response to recommended treatments")
        insights.append("Schedule follow-up in 2-4 weeks")
        
        return insights
    
    async def _process_single_prediction(self, model: Any, request: PredictionRequest) -> Dict[str, Any]:
        """Process single prediction in batch"""
        try:
            result = await self._run_prediction(model, request)
            
            return {
                'request_id': str(uuid.uuid4()),
                'status': 'success',
                'patient_id': request.patient_id,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'processing_time': 0.5  # Replace with actual time
            }
        except Exception as e:
            return {
                'request_id': str(uuid.uuid4()),
                'status': 'error',
                'patient_id': request.patient_id,
                'error': str(e)
            }
    
    async def _log_prediction(self, request: PredictionRequest, response: PredictionResponse):
        """Log prediction for monitoring"""
        await self.performance_monitor.log_prediction({
            'patient_id': request.patient_id,
            'model_version': response.model_version,
            'processing_time': response.processing_time,
            'confidence': response.confidence,
            'timestamp': response.timestamp.isoformat()
        })
    
    async def _log_batch_prediction(self, request: BatchPredictionRequest, results: List[Dict[str, Any]]):
        """Log batch prediction for monitoring"""
        await self.performance_monitor.log_batch_prediction({
            'batch_id': request.batch_id,
            'total_requests': len(request.requests),
            'success_count': sum(1 for r in results if r.get('status') == 'success'),
            'error_count': sum(1 for r in results if r.get('status') == 'error'),
            'timestamp': datetime.utcnow().isoformat()
        })
    
    async def _hot_swap_model(self, model_name: str):
        """Hot-swap model with zero downtime"""
        try:
            logger.info(f"Starting hot-swap for model {model_name}")
            
            # Load new model
            new_model = await self.model_loader.load_model(model_name)
            
            # Validate model
            await self._validate_model(new_model)
            
            # Switch to new model
            old_model = self.models.get(model_name)
            self.models[model_name] = new_model
            
            # Cleanup old model
            if old_model:
                await self.model_loader.cleanup_model(old_model)
            
            logger.info(f"Hot-swap completed for model {model_name}")
            
        except Exception as e:
            logger.error(f"Hot-swap failed for model {model_name}: {str(e)}")
            raise
    
    async def _validate_model(self, model: Any):
        """Validate loaded model"""
        # Implement model validation logic
        if model is None:
            raise ValueError("Model validation failed: model is None")
        
        # Add specific validation checks based on model type
        return True
    
    async def _check_cache_health(self) -> bool:
        """Check cache health"""
        try:
            if self.redis_client is None:
                self.redis_client = await aioredis.from_url(
                    f"redis://{self.config['cache']['redis_host']}:{self.config['cache']['redis_port']}"
                )
            
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Cache health check failed: {str(e)}")
            return False

# Initialize server
server = ModelServer()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Medical AI Production Server...")
    await server.model_loader.initialize()
    await server.cache_manager.initialize()
    await server.performance_monitor.initialize()
    logger.info("Server startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Medical AI Production Server...")
    await server.cache_manager.close()
    logger.info("Server shutdown complete")

# Create FastAPI application with lifespan
app = FastAPI(
    title="Medical AI Production Server",
    description="Production-grade medical AI model serving infrastructure",
    version="1.0.0",
    lifespan=lifespan
)

# Mount routes
app.include_router(server.app.router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        reload=False,
        access_log=True
    )