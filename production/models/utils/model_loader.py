"""
Model Loader Utility
Handles loading, caching, and management of medical AI models with versioning support.
"""

import os
import asyncio
import logging
import pickle
import hashlib
import aiofiles
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import joblib
import numpy as np
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelLoader:
    """Production model loader with caching and versioning"""
    
    def __init__(self, model_cache_dir: str = "/tmp/model_cache", 
                 max_cache_size: int = 10, cache_ttl: int = 3600):
        self.model_cache_dir = model_cache_dir
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl
        self.loaded_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {"hits": 0, "misses": 0, "loads": 0}
        
        # Ensure cache directory exists
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # Model registry configuration
        self.model_registry = {
            "medical-diagnosis-v1": {
                "path": "models/medical_diagnosis_v1",
                "type": "transformers",
                "framework": "pytorch",
                "description": "Primary medical diagnosis model",
                "version": "1.0.0",
                "accuracy": 0.89,
                "latency_ms": 150,
                "memory_mb": 512
            },
            "clinical-risk-v1": {
                "path": "models/clinical_risk_v1",
                "type": "sklearn",
                "framework": "scikit-learn",
                "description": "Clinical risk assessment model",
                "version": "1.1.0",
                "accuracy": 0.92,
                "latency_ms": 50,
                "memory_mb": 128
            },
            "drug-interaction-v1": {
                "path": "models/drug_interaction_v1",
                "type": "neural_network",
                "framework": "pytorch",
                "description": "Drug interaction prediction model",
                "version": "2.0.0",
                "accuracy": 0.95,
                "latency_ms": 200,
                "memory_mb": 1024
            },
            "symptom-analyzer-v1": {
                "path": "models/symptom_analyzer_v1",
                "type": "ensemble",
                "framework": "mixed",
                "description": "Multi-modal symptom analysis",
                "version": "1.2.0",
                "accuracy": 0.87,
                "latency_ms": 300,
                "memory_mb": 768
            }
        }
    
    async def initialize(self):
        """Initialize the model loader"""
        logger.info("Initializing Model Loader...")
        
        # Preload critical models
        critical_models = ["medical-diagnosis-v1", "clinical-risk-v1"]
        for model_name in critical_models:
            try:
                await self.load_model(model_name)
                logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {str(e)}")
        
        # Start cache cleanup task
        asyncio.create_task(self._periodic_cache_cleanup())
        
        logger.info("Model Loader initialization complete")
    
    async def load_model(self, model_name: str) -> Any:
        """Load a model with caching"""
        
        # Check if model is already loaded and valid
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            if await self._is_model_valid(model_name, model):
                self.cache_stats["hits"] += 1
                logger.debug(f"Model cache hit: {model_name}")
                return model
        
        # Cache miss or invalid model
        self.cache_stats["misses"] += 1
        
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found in registry")
        
        model_config = self.model_registry[model_name]
        logger.info(f"Loading model {model_name} (version {model_config['version']})")
        
        try:
            # Load model based on framework
            if model_config["framework"] == "pytorch":
                model = await self._load_pytorch_model(model_name, model_config)
            elif model_config["framework"] == "scikit-learn":
                model = await self._load_sklearn_model(model_name, model_config)
            elif model_config["framework"] == "mixed":
                model = await self._load_mixed_model(model_name, model_config)
            else:
                raise ValueError(f"Unsupported framework: {model_config['framework']}")
            
            # Cache the model
            self.loaded_models[model_name] = model
            self.model_metadata[model_name] = {
                "loaded_at": datetime.utcnow().isoformat(),
                "version": model_config["version"],
                "type": model_config["type"],
                "framework": model_config["framework"],
                "size_mb": model_config["memory_mb"]
            }
            
            self.cache_stats["loads"] += 1
            logger.info(f"Model loaded successfully: {model_name}")
            
            # Manage cache size
            await self._manage_cache_size()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def _load_pytorch_model(self, model_name: str, model_config: Dict[str, Any]) -> Any:
        """Load PyTorch-based model"""
        
        # Create synthetic model for demonstration
        # In production, this would load actual model files
        class MockPyTorchModel:
            def __init__(self):
                self.model_name = model_name
                self.version = model_config["version"]
                
            def forward(self, **kwargs):
                # Mock forward pass
                class MockOutput:
                    def __init__(self):
                        self.logits = torch.randn(1, 10)
                
                return MockOutput()
            
            def predict(self, input_data):
                # Mock prediction
                return {"diagnosis": "sample_diagnosis", "confidence": 0.85}
        
        return MockPyTorchModel()
    
    async def _load_sklearn_model(self, model_name: str, model_config: Dict[str, Any]) -> Any:
        """Load scikit-learn model"""
        
        # Create synthetic sklearn model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create a simple classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Mock training data (in production, load actual model)
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        try:
            model.fit(X, y)
            logger.info(f"Sklearn model trained: {model_name}")
        except Exception as e:
            logger.warning(f"Model training failed, using untrained model: {str(e)}")
        
        return model
    
    async def _load_mixed_model(self, model_name: str, model_config: Dict[str, Any]) -> Any:
        """Load mixed framework model"""
        
        class MockEnsembleModel:
            def __init__(self):
                self.models = {
                    "transformers": MockPyTorchModel(),
                    "sklearn": RandomForestClassifier(n_estimators=50)
                }
                self.weights = [0.6, 0.4]
                
            def predict(self, input_data):
                # Combine predictions from different models
                pred1 = self.models["transformers"].predict(input_data)
                pred2 = self.models["sklearn"].predict([input_data] if not isinstance(input_data, list) else input_data)
                
                return {
                    "primary": pred1["diagnosis"],
                    "confidence": max(pred1["confidence"], 0.8),
                    "ensemble_contribution": {
                        "transformers": pred1,
                        "sklearn": pred2.tolist() if hasattr(pred2, 'tolist') else pred2
                    }
                }
        
        return MockEnsembleModel()
    
    async def _is_model_valid(self, model_name: str, model: Any) -> bool:
        """Check if loaded model is still valid"""
        if model_name not in self.model_metadata:
            return False
        
        metadata = self.model_metadata[model_name]
        loaded_at = datetime.fromisoformat(metadata["loaded_at"])
        
        # Check if model has expired
        if datetime.utcnow() - loaded_at > timedelta(seconds=self.cache_ttl):
            logger.debug(f"Model {model_name} has expired, will reload")
            return False
        
        return True
    
    async def _manage_cache_size(self):
        """Manage model cache size by removing oldest/unused models"""
        if len(self.loaded_models) <= self.max_cache_size:
            return
        
        # Sort models by last access time (simplified - using loaded_at for demo)
        models_by_age = sorted(
            self.model_metadata.items(),
            key=lambda x: x[1]["loaded_at"]
        )
        
        # Remove oldest models
        models_to_remove = len(self.loaded_models) - self.max_cache_size
        for model_name, _ in models_by_age[:models_to_remove]:
            if model_name != "medical-diagnosis-v1":  # Don't remove critical models
                await self.cleanup_model(self.loaded_models[model_name])
                del self.loaded_models[model_name]
                del self.model_metadata[model_name]
                logger.info(f"Removed model from cache: {model_name}")
    
    async def cleanup_model(self, model: Any):
        """Cleanup model resources"""
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
            
            if hasattr(model, 'save_pretrained'):
                # Clear transformers cache
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            logger.debug("Model resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Model cleanup warning: {str(e)}")
    
    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Remove expired models
                expired_models = []
                for model_name, metadata in self.model_metadata.items():
                    loaded_at = datetime.fromisoformat(metadata["loaded_at"])
                    if datetime.utcnow() - loaded_at > timedelta(seconds=self.cache_ttl):
                        expired_models.append(model_name)
                
                for model_name in expired_models:
                    if model_name in self.loaded_models:
                        await self.cleanup_model(self.loaded_models[model_name])
                        del self.loaded_models[model_name]
                        del self.model_metadata[model_name]
                        logger.info(f"Expired model removed: {model_name}")
                
                # Clear cache files
                await self._cleanup_cache_files()
                
            except Exception as e:
                logger.error(f"Cache cleanup task error: {str(e)}")
    
    async def _cleanup_cache_files(self):
        """Clean up old cache files"""
        try:
            current_time = datetime.utcnow().timestamp()
            cache_files = os.listdir(self.model_cache_dir)
            
            for filename in cache_files:
                file_path = os.path.join(self.model_cache_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > self.cache_ttl:
                        os.remove(file_path)
                        logger.debug(f"Removed old cache file: {filename}")
                        
        except Exception as e:
            logger.warning(f"Cache file cleanup error: {str(e)}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        return self.model_registry.get(model_name, {})
    
    def get_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Get all available models"""
        return self.model_registry.copy()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return self.cache_stats.copy()
    
    async def preload_models(self, model_names: List[str]):
        """Preload specific models"""
        tasks = [self.load_model(name) for name in model_names]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Preloaded models: {model_names}")
    
    def get_model_metadata(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for loaded model"""
        return self.model_metadata.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if model is currently loaded"""
        return model_name in self.loaded_models
    
    async def unload_model(self, model_name: str):
        """Unload a specific model"""
        if model_name in self.loaded_models:
            await self.cleanup_model(self.loaded_models[model_name])
            del self.loaded_models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            logger.info(f"Model unloaded: {model_name}")
    
    def get_cache_directory(self) -> str:
        """Get the cache directory path"""
        return self.model_cache_dir