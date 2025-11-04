"""
Model adapters for different model types and formats.
Provides standardized interfaces for various ML frameworks and model formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import os
import json
import pickle
import time
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import structlog

from ..config.logging_config import get_logger
from ..config.settings import get_settings


class ModelAdapter(ABC):
    """Base adapter interface for model implementations."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")
        self._model = None
        self._tokenizer = None
    
    @abstractmethod
    async def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    async def predict(self, inputs: Any, **kwargs) -> Any:
        """Make a prediction with the model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not self._model:
            return {"cpu_mb": 0.0, "gpu_mb": 0.0}
        
        cpu_memory = 0.0
        gpu_memory = 0.0
        
        # Calculate CPU memory
        for param in self._model.parameters():
            cpu_memory += param.numel() * param.element_size()
        
        # Calculate GPU memory if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
        
        return {
            "cpu_mb": cpu_memory / 1024 / 1024,
            "gpu_mb": gpu_memory
        }


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for Hugging Face models."""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None, **kwargs):
        super().__init__(model_path or model_name)
        self.model_name = model_name
        self.kwargs = kwargs
    
    async def load(self) -> None:
        """Load Hugging Face model and tokenizer."""
        try:
            settings = get_settings()
            
            # Load tokenizer
            tokenizer_kwargs = {"trust_remote_code": True}
            if self.model_path:
                tokenizer_kwargs["local_files_only"] = True
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path or self.model_name,
                revision=settings.model.model_revision,
                **tokenizer_kwargs
            )
            
            # Add padding token if missing
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": getattr(torch, settings.model.torch_dtype)
            }
            
            if settings.model.device_map:
                model_kwargs["device_map"] = settings.model.device_map
            
            if self.model_path:
                model_kwargs["local_files_only"] = True
            
            self._model = AutoModel.from_pretrained(
                self.model_path or self.model_name,
                revision=settings.model.model_revision,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if not settings.model.device_map:
                self._model = self._model.to(self.device)
            
            self.logger.info(
                f"HuggingFace model loaded successfully",
                model_name=self.model_name,
                device=self.device
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load HuggingFace model: {e}")
            raise
    
    async def predict(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """Make prediction with HuggingFace model."""
        if not self._model or not self._tokenizer:
            raise RuntimeError("Model not loaded")
        
        # Handle different input types
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # Tokenize
        encoded = self._tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=kwargs.get("max_length", get_settings().model.max_length),
            return_tensors="pt"
        )
        
        # Move to device
        if hasattr(encoded, 'to'):
            encoded = encoded.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self._model(**encoded)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to list
        embeddings_list = embeddings.cpu().numpy().tolist()
        
        # Clean up
        del encoded, outputs, embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embeddings_list[0] if len(embeddings_list) == 1 else embeddings_list
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "adapter_type": "huggingface",
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "parameters": sum(p.numel() for p in self._model.parameters()) if self._model else 0,
            "memory_usage": self.get_memory_usage()
        }


class LocalModelAdapter(ModelAdapter):
    """Adapter for locally saved models."""
    
    def __init__(self, model_path: str, model_type: str = "pytorch", **kwargs):
        super().__init__(model_path)
        self.model_type = model_type
        self.kwargs = kwargs
    
    async def load(self) -> None:
        """Load local model."""
        try:
            if self.model_type == "pytorch":
                self._model = torch.load(
                    os.path.join(self.model_path, "model.pt"),
                    map_location=self.device
                )
                self._model.eval()
                
                # Load tokenizer if available
                tokenizer_path = os.path.join(self.model_path, "tokenizer")
                if os.path.exists(tokenizer_path):
                    self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            elif self.model_type == "pickle":
                with open(os.path.join(self.model_path, "model.pkl"), "rb") as f:
                    self._model = pickle.load(f)
            
            elif self.model_type == "onnx":
                import onnxruntime as ort
                self._model = ort.InferenceSession(
                    os.path.join(self.model_path, "model.onnx"),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
            
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(
                f"Local model loaded successfully",
                model_path=self.model_path,
                model_type=self.model_type
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            raise
    
    async def predict(self, inputs: Any, **kwargs) -> Any:
        """Make prediction with local model."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        if self.model_type == "pytorch":
            if isinstance(inputs, str):
                inputs = self._tokenizer(inputs, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self._model(**inputs)
            return outputs
        
        elif self.model_type == "onnx":
            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: inputs})
            return outputs[0]
        
        elif self.model_type == "pickle":
            return self._model.predict(inputs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            "adapter_type": "local",
            "model_path": self.model_path,
            "model_type": self.model_type,
            "device": self.device,
            "memory_usage": self.get_memory_usage()
        }
        
        if self._model and hasattr(self._model, 'parameters'):
            info["parameters"] = sum(p.numel() for p in self._model.parameters())
        
        return info


class ModelAdapterFactory:
    """Factory for creating appropriate model adapters."""
    
    @staticmethod
    def create_adapter(model_path: str, **kwargs) -> ModelAdapter:
        """Create appropriate adapter based on model path and configuration."""
        settings = get_settings()
        
        # Check if it's a Hugging Face model
        if "/" in model_path and not os.path.exists(model_path):
            return HuggingFaceAdapter(model_path, **kwargs)
        
        # Check local model type
        if os.path.exists(model_path):
            model_files = os.listdir(model_path)
            
            if "model.pt" in model_files:
                return LocalModelAdapter(model_path, "pytorch", **kwargs)
            elif "model.onnx" in model_files:
                return LocalModelAdapter(model_path, "onnx", **kwargs)
            elif "model.pkl" in model_files:
                return LocalModelAdapter(model_path, "pickle", **kwargs)
        
        # Default to HuggingFace adapter
        return HuggingFaceAdapter(model_path, **kwargs)
    
    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported model formats."""
        return [
            "huggingface",
            "pytorch",
            "onnx",
            "pickle",
            "tensorflow",
            "scikit-learn"
        ]


class ModelCache:
    """Cache for model adapters and predictions."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, ModelAdapter] = {}
        self.access_times: Dict[str, datetime] = {}
        self.logger = structlog.get_logger("model_cache")
    
    def get(self, model_id: str) -> Optional[ModelAdapter]:
        """Get cached model adapter."""
        if model_id in self.cache:
            self.access_times[model_id] = datetime.utcnow()
            return self.cache[model_id]
        return None
    
    def set(self, model_id: str, adapter: ModelAdapter) -> None:
        """Cache model adapter."""
        # Implement LRU eviction
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[model_id] = adapter
        self.access_times[model_id] = datetime.utcnow()
    
    def _evict_oldest(self) -> None:
        """Remove oldest accessed adapter."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), 
                        key=lambda k: self.access_times[k])
        self.remove(oldest_key)
    
    def remove(self, model_id: str) -> None:
        """Remove model adapter from cache."""
        if model_id in self.cache:
            # Clean up model resources
            adapter = self.cache[model_id]
            if hasattr(adapter, '_model') and adapter._model:
                del adapter._model
            if hasattr(adapter, '_tokenizer') and adapter._tokenizer:
                del adapter._tokenizer
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            del self.cache[model_id]
            self.access_times.pop(model_id, None)
    
    def clear(self) -> None:
        """Clear all cached adapters."""
        for model_id in list(self.cache.keys()):
            self.remove(model_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_memory = 0.0
        for adapter in self.cache.values():
            memory_usage = adapter.get_memory_usage()
            total_memory += memory_usage.get("cpu_mb", 0) + memory_usage.get("gpu_mb", 0)
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "total_memory_mb": total_memory,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0
        }