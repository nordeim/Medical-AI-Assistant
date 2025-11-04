"""
Model Service for Medical AI Inference
Handles model loading, inference, and management
"""

import asyncio
import time
import torch
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging
from dataclasses import dataclass

from ..utils.logger import get_logger
from ..utils.exceptions import ModelUnavailableError
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    total_inferences: int = 0
    successful_inferences: int = 0
    failed_inferences: int = 0
    average_latency_ms: float = 0.0
    last_inference_time: Optional[str] = None
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0
    model_loaded: bool = False
    model_version: Optional[str] = None


class ModelService:
    """Service for managing medical AI models"""
    
    def __init__(self):
        self.logger = get_logger("model_service")
        self.model = None
        self.tokenizer = None
        self.metrics = ModelMetrics()
        self.is_initialized = False
        self.loading_start_time = None
        
        # Model configuration
        self.model_name = settings.model_name
        self.model_version = settings.model_version
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        
        # Performance tracking
        self.inference_times = []
        self.max_inference_history = 1000
    
    async def initialize(self):
        """Initialize the model service"""
        
        if self.is_initialized:
            return
        
        self.loading_start_time = time.time()
        self.logger.info("Starting model service initialization")
        
        try:
            # In production, this would load the actual medical AI model
            # For now, we'll simulate model loading
            await self._load_model()
            await self._load_tokenizer()
            await self._warm_up_model()
            
            # Update metrics
            self.metrics.model_loaded = True
            self.metrics.model_version = self.model_version
            self.is_initialized = True
            
            load_time = time.time() - self.loading_start_time
            self.logger.info(
                "Model service initialized successfully",
                load_time=load_time,
                model_name=self.model_name,
                model_version=self.model_version
            )
            
        except Exception as e:
            self.logger.error(f"Model service initialization failed: {e}")
            raise ModelUnavailableError(f"Failed to initialize model: {str(e)}")
    
    async def _load_model(self):
        """Load the medical AI model"""
        
        # Simulate model loading
        await asyncio.sleep(2)  # Simulate loading time
        
        # In production, this would be:
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     torch_dtype=torch.float16,
        #     device_map="auto"
        # )
        
        self.logger.info("Medical AI model loaded")
    
    async def _load_tokenizer(self):
        """Load the tokenizer"""
        
        # Simulate tokenizer loading
        await asyncio.sleep(0.5)
        
        # In production, this would be:
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.logger.info("Tokenizer loaded")
    
    async def _warm_up_model(self):
        """Warm up the model with sample inputs"""
        
        # Perform warm-up inferences
        warm_up_prompts = [
            "Patient presents with chest pain and shortness of breath.",
            "Symptoms include fever, cough, and fatigue.",
            "Medical history includes diabetes and hypertension."
        ]
        
        for prompt in warm_up_prompts:
            try:
                await self._simulate_inference(prompt)
            except Exception as e:
                self.logger.warning(f"Warm-up inference failed: {e}")
        
        self.logger.info("Model warm-up completed")
    
    async def inference(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform inference with the medical AI model"""
        
        if not self.is_initialized:
            raise ModelUnavailableError("Model service not initialized")
        
        start_time = time.time()
        self.metrics.total_inferences += 1
        
        try:
            # Prepare input
            input_text = self._prepare_input(prompt, context)
            
            # Perform inference
            response, confidence = await self._generate_response(
                input_text, max_new_tokens, temperature, **kwargs
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, success=True)
            
            # Update last inference time
            self.metrics.last_inference_time = datetime.now(timezone.utc).isoformat()
            
            self.logger.debug(
                "Inference completed successfully",
                latency_ms=latency_ms,
                confidence=confidence,
                prompt_length=len(prompt)
            )
            
            return {
                "response": response,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "model_version": self.model_version,
                "tokens_used": len(response.split()) * 1.3,  # Rough estimation
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_metrics(latency_ms, success=False)
            
            self.logger.error(f"Inference failed: {e}")
            raise ModelUnavailableError(f"Inference failed: {str(e)}")
    
    async def batch_inference(
        self,
        prompts: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform batch inference for multiple prompts"""
        
        if not self.is_initialized:
            raise ModelUnavailableError("Model service not initialized")
        
        # Prepare context list
        if contexts is None:
            contexts = [None] * len(prompts)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_inference(prompt, context):
            async with semaphore:
                return await self.inference(prompt, context, **kwargs)
        
        # Process all prompts concurrently
        tasks = [
            process_single_inference(prompt, context)
            for prompt, context in zip(prompts, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                failed_count += 1
                successful_results.append({
                    "error": str(result),
                    "success": False
                })
            else:
                result["success"] = True
                successful_results.append(result)
        
        self.logger.info(
            "Batch inference completed",
            total_prompts=len(prompts),
            successful=len(successful_results) - failed_count,
            failed=failed_count
        )
        
        return successful_results
    
    def get_metrics(self) -> ModelMetrics:
        """Get current model metrics"""
        return self.metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status"""
        
        return {
            "status": "healthy" if self.is_initialized else "unhealthy",
            "model_loaded": self.metrics.model_loaded,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "total_inferences": self.metrics.total_inferences,
            "success_rate": (
                self.metrics.successful_inferences / max(self.metrics.total_inferences, 1)
            ),
            "average_latency_ms": self.metrics.average_latency_ms,
            "last_inference": self.metrics.last_inference_time,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "gpu_utilization": self.metrics.gpu_utilization,
            "load_time": time.time() - self.loading_start_time if self.loading_start_time else None
        }
    
    async def cleanup(self):
        """Clean up model resources"""
        
        self.logger.info("Cleaning up model service")
        
        try:
            if self.model:
                # In production: torch.cuda.empty_cache() if using GPU
                pass
            
            self.is_initialized = False
            self.logger.info("Model service cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Model service cleanup failed: {e}")
    
    def _prepare_input(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Prepare input text for model"""
        
        # Add medical context if provided
        if context:
            medical_context = []
            
            if context.get("medical_domain"):
                medical_context.append(f"Medical domain: {context['medical_domain']}")
            
            if context.get("patient_info"):
                medical_context.append(f"Patient info: {context['patient_info']}")
            
            if context.get("urgency_level"):
                medical_context.append(f"Urgency: {context['urgency_level']}")
            
            if medical_context:
                prompt = f"Context: {'; '.join(medical_context)}\n\nQuestion: {prompt}"
        
        return prompt
    
    async def _generate_response(
        self,
        input_text: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> tuple[str, float]:
        """Generate response from model"""
        
        # Use provided parameters or defaults
        max_tokens = max_new_tokens or self.max_tokens
        temp = temperature or self.temperature
        
        # In production, this would call the actual model
        # For now, simulate response generation
        await asyncio.sleep(0.1)  # Simulate inference time
        
        # Simulate response based on input
        response = await self._simulate_inference(input_text)
        
        # Calculate confidence based on input characteristics
        confidence = self._calculate_confidence(input_text, response)
        
        return response, confidence
    
    async def _simulate_inference(self, input_text: str) -> str:
        """Simulate model inference for demonstration"""
        
        # Basic medical response simulation
        responses = [
            "Based on the symptoms described, I recommend consulting with a healthcare provider for proper evaluation and diagnosis.",
            "This appears to require medical attention. Please schedule an appointment with your doctor.",
            "Given the symptoms mentioned, it's important to monitor them closely and seek medical advice if they worsen.",
            "These symptoms could indicate various conditions. A thorough medical examination would be necessary for accurate diagnosis.",
            "I understand your concern. While I can provide general information, please consult with a healthcare professional for personalized medical advice."
        ]
        
        # Simple hash-based selection for consistency
        hash_val = hash(input_text.lower()) % len(responses)
        return responses[hash_val]
    
    def _calculate_confidence(self, input_text: str, response: str) -> float:
        """Calculate confidence score for response"""
        
        # Base confidence
        base_confidence = 0.75
        
        # Adjust based on input characteristics
        if len(input_text) < 50:
            base_confidence -= 0.1  # Less context = lower confidence
        elif len(input_text) > 200:
            base_confidence -= 0.05  # Very long input = more uncertainty
        
        # Check for medical keywords
        medical_keywords = [
            "symptoms", "pain", "fever", "cough", "diagnosis", "treatment",
            "medication", "doctor", "hospital", "emergency"
        ]
        
        medical_score = sum(1 for keyword in medical_keywords 
                          if keyword in input_text.lower())
        
        if medical_score > 3:
            base_confidence += 0.1  # More medical context = higher confidence
        elif medical_score == 0:
            base_confidence -= 0.1  # No medical context = lower confidence
        
        return max(0.0, min(1.0, base_confidence))
    
    def _update_metrics(self, latency_ms: float, success: bool):
        """Update performance metrics"""
        
        # Update inference times
        self.inference_times.append(latency_ms)
        if len(self.inference_times) > self.max_inference_history:
            self.inference_times = self.inference_times[-self.max_inference_history:]
        
        # Update metrics
        if success:
            self.metrics.successful_inferences += 1
        else:
            self.metrics.failed_inferences += 1
        
        # Update average latency
        self.metrics.average_latency_ms = sum(self.inference_times) / len(self.inference_times)
        
        # Update resource usage (mock values)
        self.metrics.memory_usage_mb = 2048.0  # Simulated memory usage
        self.metrics.gpu_utilization = 65.2 if torch.cuda.is_available() else 0.0


# Global model service instance
model_service = ModelService()