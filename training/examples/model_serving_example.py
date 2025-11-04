#!/usr/bin/env python3
"""
Model Serving Example for Medical AI Training Pipeline

This example demonstrates production-ready model serving:
- FastAPI-based REST API
- Model loading and optimization
- Input validation and PHI protection
- Rate limiting and security
- Health checks and monitoring
- Container deployment
- Load testing and performance optimization

Usage:
    # Development server
    python examples/model_serving_example.py --model_path ./outputs/trained_model --port 8000
    
    # Production server
    uvicorn examples.model_serving_example:app --host 0.0.0.0 --port 8000 --workers 4
    
    # With custom configuration
    python examples/model_serving_example.py --config configs/serving_config.yaml
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import argparse
import hashlib
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# FastAPI and serving components
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    import numpy as np
    from scipy.special import softmax
except ImportError as e:
    print(f"Warning: Some serving dependencies not available: {e}")
    print("Please install with: pip install fastapi uvicorn transformers torch scipy")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for API
class MedicalQuery(BaseModel):
    """Input model for medical queries"""
    text: str = Field(..., min_length=1, max_length=10000, description="Medical question or text")
    patient_context: Optional[str] = Field(None, max_length=2000, description="Additional patient context")
    max_length: Optional[int] = Field(512, ge=50, le=2048, description="Maximum response length")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Top-p sampling")
    do_sample: Optional[bool] = Field(True, description="Whether to use sampling")
    request_id: Optional[str] = Field(None, description="Client request ID")
    
    @validator('text')
    def validate_text(cls, v):
        # Basic medical content validation
        if len(v.strip()) < 5:
            raise ValueError("Text must be at least 5 characters long")
        return v.strip()

class MedicalResponse(BaseModel):
    """Output model for medical responses"""
    response: str = Field(..., description="Generated medical response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    request_id: str = Field(..., description="Request identifier")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    safety_checks: Dict[str, Any] = Field(..., description="Safety check results")
    phi_protected: bool = Field(..., description="Whether PHI protection was applied")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")

class ModelInfo(BaseModel):
    """Model information response"""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    parameters: int = Field(..., description="Number of parameters")
    max_length: int = Field(..., description="Maximum sequence length")
    vocab_size: int = Field(..., description="Vocabulary size")
    device: str = Field(..., description="Device used for inference")
    quantization: Optional[str] = Field(None, description="Quantization method")

class MedicalAIServer:
    """Production medical AI server with comprehensive features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._setup_device()
        
        # Server state
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.avg_response_time = 0.0
        
        # PHI protection
        self.phi_redactor = PHIRedactor()
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
        # Cache for responses
        self.response_cache = {}
        self.cache_size_limit = self.config.get('cache_size', 1000)
        
        # Safety patterns
        self.safety_patterns = self._load_safety_patterns()
        
        logger.info(f"Medical AI Server initialized on device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup computation device"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {device_count} GPUs")
            
            # Use specified GPU or auto-select
            gpu_id = self.config.get('gpu_id', 0)
            if gpu_id < device_count:
                torch.cuda.set_device(gpu_id)
                return f"cuda:{gpu_id}"
        
        logger.info("Using CPU for inference")
        return "cpu"
    
    def _load_safety_patterns(self) -> Dict[str, List[str]]:
        """Load medical safety patterns for content filtering"""
        return {
            'emergency_keywords': [
                'chest pain', 'heart attack', 'stroke', 'difficulty breathing',
                'severe bleeding', 'unconscious', 'seizure', 'severe injury'
            ],
            'medication_warnings': [
                'stop medication', 'don\'t take', 'avoid prescribed',
                'alternative to', 'replace with'
            ],
            'diagnosis_patterns': [
                'i diagnose', 'you have', 'diagnosis is', 'you are suffering from'
            ],
            'professional_advice': [
                'consult a doctor', 'see your physician', 'seek medical attention',
                'contact your healthcare provider', 'professional medical advice'
            ]
        }
    
    def load_model(self):
        """Load model and tokenizer for serving"""
        
        model_path = self.config.get('model_path', './outputs/trained_model')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        logger.info(f"Loading model from: {model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading configuration
        model_kwargs = {
            'torch_dtype': torch.float16 if self.config.get('fp16', True) else torch.float32,
            'device_map': 'auto' if self.device.startswith('cuda') else None,
            'trust_remote_code': self.config.get('trust_remote_code', False)
        }
        
        # Load with quantization if enabled
        if self.config.get('load_in_8bit', False):
            model_kwargs.update({
                'load_in_8bit': True,
                'device_map': 'auto'
            })
        elif self.config.get('load_in_4bit', False):
            model_kwargs.update({
                'load_in_4bit': True,
                'device_map': 'auto'
            })
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Create pipeline for easier inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if self.device.startswith('cuda') else -1,
            torch_dtype=torch.float16 if self.config.get('fp16', True) else torch.float32,
            do_sample=self.config.get('do_sample', True),
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Get model info
        self.model_info = self._get_model_info()
        
        logger.info("Model loaded successfully")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.config.get('model_name', 'medical_ai_model'),
            'model_type': 'LoRA' if hasattr(self.model, 'peft_config') else 'Full',
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'max_length': self.tokenizer.model_max_length,
            'vocab_size': len(self.tokenizer),
            'device': self.device,
            'quantization': '8bit' if self.config.get('load_in_8bit', False) else 
                          '4bit' if self.config.get('load_in_4bit', False) else 'fp16' if self.config.get('fp16', False) else 'fp32'
        }
    
    def generate_response(self, query: MedicalQuery) -> MedicalResponse:
        """Generate medical AI response"""
        
        start_time = time.time()
        
        try:
            # Rate limiting check
            if not self.rate_limiter.check_rate_limit(query.request_id or "anonymous"):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            # Input validation and preprocessing
            processed_text = self._preprocess_input(query.text, query.patient_context)
            
            # Check cache first
            cache_key = self._get_cache_key(processed_text, query)
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key].copy()
                cached_response['processing_time'] = time.time() - start_time
                return MedicalResponse(**cached_response)
            
            # PHI protection
            phi_protected, clean_text = self.phi_redactor.protect_phi(processed_text)
            
            # Safety checks
            safety_results = self._perform_safety_checks(clean_text)
            
            # Generate response
            generated_text = self._generate_text(clean_text, query)
            
            # Post-process response
            final_response = self._postprocess_response(generated_text, query)
            
            # Calculate confidence (simplified)
            confidence = self._calculate_confidence(final_response)
            
            # Prepare response
            response_data = {
                'response': final_response,
                'confidence': confidence,
                'request_id': query.request_id or f"req_{int(time.time())}",
                'processing_time': time.time() - start_time,
                'model_info': self.model_info,
                'safety_checks': safety_results,
                'phi_protected': phi_protected
            }
            
            # Cache response
            if len(self.response_cache) < self.cache_size_limit:
                self.response_cache[cache_key] = response_data.copy()
            
            # Update metrics
            self._update_metrics(time.time() - start_time)
            
            return MedicalResponse(**response_data)
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error generating response: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    def _preprocess_input(self, text: str, context: Optional[str] = None) -> str:
        """Preprocess input text"""
        
        # Combine text and context
        if context:
            processed = f"Medical Question: {text}\nContext: {context}\nMedical Answer:"
        else:
            processed = f"Medical Question: {text}\nMedical Answer:"
        
        # Clean text
        processed = processed.strip()
        
        # Basic content filtering
        if len(processed) > 5000:  # Limit input size
            processed = processed[:5000] + "..."
        
        return processed
    
    def _generate_text(self, prompt: str, query: MedicalQuery) -> str:
        """Generate text using the model"""
        
        # Prepare generation parameters
        generation_params = {
            'max_new_tokens': query.max_length or 512,
            'do_sample': query.do_sample,
            'temperature': query.temperature,
            'top_p': query.top_p,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'return_full_text': False,
            'clean_up_tokenization_spaces': True
        }
        
        # Generate response
        response = self.pipeline(
            prompt,
            **generation_params
        )
        
        # Extract generated text
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0]['generated_text']
        else:
            generated_text = str(response)
        
        # Clean and post-process
        generated_text = self._clean_generated_text(generated_text)
        
        return generated_text
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text"""
        
        # Remove common artifacts
        text = text.replace('\n\n\n', '\n\n')  # Multiple newlines
        text = text.replace('  ', ' ')  # Multiple spaces
        
        # Remove incomplete sentences at the end
        sentences = text.split('.')
        if len(sentences) > 1:
            # Keep sentences that are reasonably complete
            complete_sentences = []
            for sentence in sentences[:-1]:
                if len(sentence.strip()) > 10:  # Minimum sentence length
                    complete_sentences.append(sentence.strip())
            
            if complete_sentences:
                text = '. '.join(complete_sentences) + '.'
        
        return text.strip()
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate response confidence (simplified)"""
        
        # Simple confidence calculation based on response characteristics
        confidence_factors = []
        
        # Length factor (longer responses generally more informative)
        length_factor = min(len(response) / 200, 1.0)  # Normalize to 200 chars
        confidence_factors.append(length_factor)
        
        # Completion factor (responses ending with proper punctuation)
        completion_factor = 1.0 if response.rstrip().endswith(('.', '?', '!')) else 0.8
        confidence_factors.append(completion_factor)
        
        # Medical content factor (presence of medical terms)
        medical_terms = ['symptoms', 'treatment', 'diagnosis', 'medication', 'condition', 'disease']
        medical_count = sum(1 for term in medical_terms if term.lower() in response.lower())
        medical_factor = min(medical_count / 3, 1.0)  # Normalize to 3 medical terms
        confidence_factors.append(medical_factor)
        
        # Safety factor (absence of dangerous content)
        safety_factor = 1.0 if not any(warning in response.lower() for warning in 
                                     ['stop medication', 'don\'t take', 'alternative to']) else 0.7
        confidence_factors.append(safety_factor)
        
        # Calculate overall confidence
        confidence = np.mean(confidence_factors)
        
        return max(0.0, min(1.0, confidence))
    
    def _perform_safety_checks(self, text: str) -> Dict[str, Any]:
        """Perform safety checks on generated content"""
        
        safety_results = {
            'emergency_content_detected': False,
            'medication_warnings': [],
            'diagnosis_statements': [],
            'professional_advice_given': False,
            'safety_score': 1.0,
            'warnings': []
        }
        
        text_lower = text.lower()
        
        # Check for emergency content
        emergency_detected = any(keyword in text_lower for keyword in self.safety_patterns['emergency_keywords'])
        safety_results['emergency_content_detected'] = emergency_detected
        
        if emergency_detected:
            safety_results['warnings'].append("Emergency medical content detected")
        
        # Check for medication warnings
        for warning in self.safety_patterns['medication_warnings']:
            if warning in text_lower:
                safety_results['medication_warnings'].append(warning)
                safety_results['warnings'].append(f"Medication warning: {warning}")
                safety_results['safety_score'] -= 0.2
        
        # Check for diagnosis statements
        for pattern in self.safety_patterns['diagnosis_patterns']:
            if pattern in text_lower:
                safety_results['diagnosis_statements'].append(pattern)
                safety_results['warnings'].append(f"Diagnosis statement: {pattern}")
                safety_results['safety_score'] -= 0.3
        
        # Check for professional advice
        professional_advice = any(advice in text_lower for advice in self.safety_patterns['professional_advice'])
        safety_results['professional_advice_given'] = professional_advice
        
        # Ensure safety score is between 0 and 1
        safety_results['safety_score'] = max(0.0, min(1.0, safety_results['safety_score']))
        
        return safety_results
    
    def _postprocess_response(self, response: str, query: MedicalQuery) -> str:
        """Post-process generated response"""
        
        # Add medical disclaimer if needed
        if not any(disc in response.lower() for disc in ['not medical advice', 'consult your', 'seek professional']):
            disclaimer = " This information is for educational purposes only. Please consult with a healthcare professional for medical advice."
            response += disclaimer
        
        # Ensure response is not too long
        max_response_length = (query.max_length or 512) * 2  # Allow some buffer
        if len(response) > max_response_length:
            response = response[:max_response_length] + "..."
        
        return response
    
    def _get_cache_key(self, text: str, query: MedicalQuery) -> str:
        """Generate cache key for response"""
        
        cache_data = f"{text}_{query.temperature}_{query.top_p}_{query.do_sample}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _update_metrics(self, response_time: float):
        """Update server metrics"""
        
        self.request_count += 1
        
        # Update average response time
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (self.avg_response_time + response_time) / 2

class PHIRedactor:
    """PHI (Protected Health Information) protection service"""
    
    def __init__(self):
        self.phi_patterns = {
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'name': r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            'patient_id': r'\b[A-Z]{2,3}\d{6,8}\b'
        }
        
        logger.info("PHI Protection initialized")
    
    def protect_phi(self, text: str) -> tuple[bool, str]:
        """Protect PHI in text"""
        
        protected_text = text
        phi_detected = False
        
        # Apply PHI protection patterns
        for phi_type, pattern in self.phi_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                phi_detected = True
                # Replace with generic terms
                replacement_map = {
                    'phone': '[PHONE_NUMBER]',
                    'email': '[EMAIL_ADDRESS]',
                    'ssn': '[SSN]',
                    'date': '[DATE]',
                    'name': '[PATIENT_NAME]',
                    'patient_id': '[PATIENT_ID]'
                }
                protected_text = re.sub(pattern, replacement_map[phi_type], protected_text, flags=re.IGNORECASE)
        
        return phi_detected, protected_text

class RateLimiter:
    """Simple rate limiter implementation"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_history = {}
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limit"""
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        if client_id in self.request_history:
            self.request_history[client_id] = [
                req_time for req_time in self.request_history[client_id] 
                if req_time > minute_ago
            ]
        
        # Check current request count
        current_requests = len(self.request_history.get(client_id, []))
        
        if current_requests >= self.requests_per_minute:
            return False
        
        # Add current request
        if client_id not in self.request_history:
            self.request_history[client_id] = []
        self.request_history[client_id].append(current_time)
        
        return True

# FastAPI application
def create_fastAPI_app(server: MedicalAIServer) -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Medical AI Assistant API",
        description="HIPAA-compliant medical AI model serving API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server.config.get('cors_origins', ["*"]),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=server.config.get('allowed_hosts', ["*"])
    )
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            model_loaded=server.model is not None,
            version="1.0.0",
            uptime=time.time() - server.start_time
        )
    
    # Model info endpoint
    @app.get("/model/info", response_model=ModelInfo)
    async def get_model_info():
        """Get model information"""
        if server.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        return ModelInfo(**server.model_info)
    
    # Prediction endpoint
    @app.post("/predict", response_model=MedicalResponse)
    async def predict(
        query: MedicalQuery,
        request: Request,
        background_tasks: BackgroundTasks
    ):
        """Generate medical AI response"""
        
        # Add request to background tasks for logging
        background_tasks.add_task(log_request, query, request.client.host if request.client else "unknown")
        
        # Generate response
        response = server.generate_response(query)
        
        return response
    
    # Statistics endpoint
    @app.get("/stats")
    async def get_statistics():
        """Get server statistics"""
        return {
            "total_requests": server.request_count,
            "error_count": server.error_count,
            "error_rate": server.error_count / max(server.request_count, 1),
            "average_response_time": server.avg_response_time,
            "uptime": time.time() - server.start_time,
            "cache_size": len(server.response_cache),
            "model_info": server.model_info
        }
    
    # Clear cache endpoint
    @app.post("/admin/clear-cache")
    async def clear_cache():
        """Clear response cache (admin only)"""
        server.response_cache.clear()
        return {"message": "Cache cleared successfully"}
    
    return app

def log_request(query: MedicalQuery, client_ip: str):
    """Log incoming requests (background task)"""
    
    logger.info(f"Request from {client_ip}: {query.text[:100]}...")

def main():
    """Main function for model serving example"""
    
    parser = argparse.ArgumentParser(description="Medical AI Model Serving Example")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind server to")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to serving configuration file")
    parser.add_argument("--gpu_id", type=int, default=0,
                       help="GPU ID to use (0 for auto)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit quantized mode")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit quantized mode")
    parser.add_argument("--cache_size", type=int, default=1000,
                       help="Response cache size")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        import yaml
        with open(args.config, 'r') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    else:
        config = {
            'model_path': args.model_path,
            'port': args.port,
            'host': args.host,
            'gpu_id': args.gpu_id,
            'load_in_8bit': args.load_in_8bit,
            'load_in_4bit': args.load_in_4bit,
            'cache_size': args.cache_size,
            'fp16': True,
            'do_sample': True,
            'cors_origins': ["*"],
            'allowed_hosts': ["*"]
        }
    
    # Create server
    server = MedicalAIServer(config)
    
    try:
        # Load model
        logger.info("Loading model for serving...")
        server.load_model()
        
        # Create FastAPI app
        app = create_fastAPI_app(server)
        
        # Start server
        logger.info(f"Starting Medical AI Server on {args.host}:{args.port}")
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers if args.workers > 1 else 1,
            log_level="info",
            access_log=True
        )
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
