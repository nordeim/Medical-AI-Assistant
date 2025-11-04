"""
Concrete implementations of model servers for different types of ML models.
Includes text generation, embeddings, and conversation models.
"""

import os
import torch
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio
import time
from datetime import datetime
import json
import pickle

import transformers
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel, 
    BitsAndBytesConfig, pipeline
)
import structlog
from peft import PeftModel, PeftConfig

from .base_server import (
    BaseModelServer, ConcurrentModelServer, PredictionRequest, PredictionResponse,
    ModelMetadata, ModelStatus, PredictionType
)
from ..config.logging_config import get_model_logger
from ..config.settings import get_settings


class TextGenerationServer(ConcurrentModelServer):
    """Model server for text generation tasks."""
    
    def __init__(self, model_id: str, name: str, 
                 max_concurrent_requests: int = 5,
                 enable_chat: bool = False):
        super().__init__(model_id, name, PredictionType.TEXT_GENERATION, 
                        max_concurrent_requests)
        self.enable_chat = enable_chat
        self.model_logger = get_model_logger(f"{name}_text_gen")
    
    async def load_model(self) -> None:
        """Load text generation model."""
        self.logger.info(f"Loading text generation model: {self.name}")
        settings = get_settings()
        
        try:
            # Configure quantization
            quantization_config = None
            if settings.model.use_quantization:
                if settings.model.quantization_type == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif settings.model.quantization_type == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer
            tokenizer_kwargs = {"trust_remote_code": True}
            if settings.model.model_path:
                tokenizer_kwargs["local_files_only"] = True
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model.model_path or settings.model.model_name,
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
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if settings.model.device_map:
                model_kwargs["device_map"] = settings.model.device_map
            
            if settings.model.model_path:
                model_kwargs["local_files_only"] = True
            
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model.model_path or settings.model.model_name,
                revision=settings.model.model_revision,
                **model_kwargs
            )
            
            # Set device
            if hasattr(self._model, 'device'):
                self._device = str(self._model.device)
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Move to device if not using device_map
            if not settings.model.device_map:
                self._model = self._model.to(self._device)
            
            self.logger.info(
                f"Text generation model loaded successfully",
                model_name=settings.model.model_name,
                device=self._device,
                parameters=sum(p.numel() for p in self._model.parameters())
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load text generation model: {e}")
            raise
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate text from input prompt."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = self.get_cached_response(request)
            if cached_response:
                return cached_response
            
            # Process input
            if isinstance(request.inputs, str):
                prompt = request.inputs
            elif isinstance(request.inputs, dict):
                prompt = request.inputs.get("prompt", "")
            else:
                raise ValueError("Invalid input format for text generation")
            
            # Default parameters
            params = {
                "max_length": get_settings().model.max_length,
                "temperature": get_settings().model.temperature,
                "top_p": get_settings().model.top_p,
                "top_k": get_settings().model.top_k,
                "do_sample": True,
                "pad_token_id": self._tokenizer.eos_token_id
            }
            
            # Update with request parameters
            params.update(request.parameters)
            
            # Tokenize input
            inputs = self._tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=params["max_length"] // 2  # Leave room for generation
            ).to(self._device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=params.get("max_new_tokens", params["max_length"] // 2),
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    top_k=params["top_k"],
                    do_sample=params["do_sample"],
                    pad_token_id=params["pad_token_id"],
                    eos_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # Decode output
            generated_text = self._tokenizer.decode(
                outputs.sequences[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean up
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            
            # Create response
            response = PredictionResponse(
                request_id=request.request_id,
                model_id=self.model_id,
                outputs=generated_text.strip(),
                processing_time=processing_time,
                metadata={
                    "input_length": len(inputs["input_ids"][0]),
                    "output_length": len(generated_text.split()),
                    "tokens_generated": len(outputs.sequences[0]) - len(inputs["input_ids"][0]) if 'outputs' in locals() else 0
                }
            )
            
            # Cache response
            self.cache_response(request, response)
            
            # Update metrics
            self.update_metrics(processing_time, True)
            
            self.model_logger.log_prediction(
                self.name, 
                len(prompt), 
                len(generated_text), 
                processing_time
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            
            self.logger.error(
                f"Text generation failed",
                request_id=request.request_id,
                error=str(e)
            )
            raise
    
    def get_model_info(self) -> ModelMetadata:
        """Get model metadata."""
        settings = get_settings()
        
        return ModelMetadata(
            model_id=self.model_id,
            name=self.name,
            version=settings.model.model_revision,
            prediction_type=PredictionType.TEXT_GENERATION,
            max_length=settings.model.max_length,
            device=self._device,
            quantization=settings.model.quantization_type if settings.model.use_quantization else None,
            tags=["text-generation", "causal-lm"],
            description="Text generation model for completing prompts and generating responses"
        )


class EmbeddingServer(BaseModelServer):
    """Model server for embedding generation."""
    
    def __init__(self, model_id: str, name: str):
        super().__init__(model_id, name, PredictionType.EMBEDDING)
        self.model_logger = get_model_logger(f"{name}_embedding")
    
    async def load_model(self) -> None:
        """Load embedding model."""
        self.logger.info(f"Loading embedding model: {self.name}")
        settings = get_settings()
        
        try:
            # Load model and tokenizer
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": getattr(torch, settings.model.torch_dtype)
            }
            
            if settings.model.device_map:
                model_kwargs["device_map"] = settings.model.device_map
            
            if settings.model.model_path:
                model_kwargs["local_files_only"] = True
            
            self._model = AutoModel.from_pretrained(
                settings.model.model_path or settings.model.model_name,
                revision=settings.model.model_revision,
                **model_kwargs
            )
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model.model_path or settings.model.model_name,
                revision=settings.model.model_revision,
                trust_remote_code=True
            )
            
            # Set device
            if hasattr(self._model, 'device'):
                self._device = str(self._model.device)
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if not settings.model.device_map:
                self._model = self._model.to(self._device)
            
            self.logger.info(
                f"Embedding model loaded successfully",
                model_name=settings.model.model_name,
                device=self._device
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate embeddings for input text(s)."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_response = self.get_cached_response(request)
            if cached_response:
                return cached_response
            
            # Process input
            if isinstance(request.inputs, str):
                texts = [request.inputs]
            elif isinstance(request.inputs, list):
                texts = request.inputs
            elif isinstance(request.inputs, dict):
                texts = request.inputs.get("texts", [])
                if isinstance(texts, str):
                    texts = [texts]
            else:
                raise ValueError("Invalid input format for embeddings")
            
            # Tokenize texts
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=get_settings().model.max_length,
                return_tensors="pt"
            ).to(self._device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self._model(**encoded)
                # Use mean pooling for sentence embeddings
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            # Convert to list
            embeddings_list = embeddings.cpu().numpy().tolist()
            
            # Clean up
            del encoded, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            
            # Create response
            if len(texts) == 1:
                output = embeddings_list[0]
            else:
                output = embeddings_list
            
            response = PredictionResponse(
                request_id=request.request_id,
                model_id=self.model_id,
                outputs=output,
                processing_time=processing_time,
                metadata={
                    "num_texts": len(texts),
                    "embedding_dim": len(output[0]) if isinstance(output[0], list) else len(output)
                }
            )
            
            # Cache response
            self.cache_response(request, response)
            
            # Update metrics
            self.update_metrics(processing_time, True)
            
            self.model_logger.log_prediction(
                self.name,
                sum(len(text) for text in texts),
                len(output) * len(output[0]) if isinstance(output[0], list) else len(output),
                processing_time
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            
            self.logger.error(
                f"Embedding generation failed",
                request_id=request.request_id,
                error=str(e)
            )
            raise
    
    def get_model_info(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            model_id=self.model_id,
            name=self.name,
            version=get_settings().model.model_revision,
            prediction_type=PredictionType.EMBEDDING,
            max_length=get_settings().model.max_length,
            device=self._device,
            tags=["embeddings", "sentence-transformers"],
            description="Embedding model for generating text representations"
        )


class ConversationServer(ConcurrentModelServer):
    """Model server for conversational AI."""
    
    def __init__(self, model_id: str, name: str, max_concurrent_requests: int = 5):
        super().__init__(model_id, name, PredictionType.CONVERSATION, max_concurrent_requests)
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self.max_history_length = 10
        self.model_logger = get_model_logger(f"{name}_conversation")
    
    async def load_model(self) -> None:
        """Load conversation model."""
        self.logger.info(f"Loading conversation model: {self.name}")
        
        # Use text generation server functionality
        settings = get_settings()
        
        try:
            # Configure quantization
            quantization_config = None
            if settings.model.use_quantization:
                if settings.model.quantization_type == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model.model_path or settings.model.model_name,
                revision=settings.model.model_revision,
                trust_remote_code=True
            )
            
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": getattr(torch, settings.model.torch_dtype)
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if settings.model.device_map:
                model_kwargs["device_map"] = settings.model.device_map
            
            if settings.model.model_path:
                model_kwargs["local_files_only"] = True
            
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model.model_path or settings.model.model_name,
                revision=settings.model.model_revision,
                **model_kwargs
            )
            
            if hasattr(self._model, 'device'):
                self._device = str(self._model.device)
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if not settings.model.device_map:
                self._model = self._model.to(self._device)
            
            self.logger.info(
                f"Conversation model loaded successfully",
                model_name=settings.model.model_name,
                device=self._device
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load conversation model: {e}")
            raise
    
    def _format_conversation(self, session_id: str, current_message: str) -> str:
        """Format conversation history with current message."""
        history = self.conversation_history.get(session_id, [])
        
        # Format as chat format (simple version)
        conversation = ""
        for msg in history[-self.max_history_length:]:
            if msg["role"] == "user":
                conversation += f"Human: {msg['content']}\n"
            elif msg["role"] == "assistant":
                conversation += f"Assistant: {msg['content']}\n"
        
        # Add current message
        conversation += f"Human: {current_message}\nAssistant:"
        
        return conversation
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate conversational response."""
        start_time = time.time()
        
        try:
            # Process input
            if isinstance(request.inputs, str):
                message = request.inputs
                session_id = request.session_id or request.user_id or request.request_id
            elif isinstance(request.inputs, dict):
                message = request.inputs.get("message", "")
                session_id = request.inputs.get("session_id") or request.session_id or request.user_id or request.request_id
            else:
                raise ValueError("Invalid input format for conversation")
            
            # Format conversation
            prompt = self._format_conversation(session_id, message)
            
            # Check cache
            cache_key = f"{session_id}:{message}"
            cached_response = self.cache.get(self.model_id, cache_key, {})
            if cached_response:
                return PredictionResponse(**cached_response)
            
            # Generate response
            settings = get_settings()
            params = {
                "max_new_tokens": 150,
                "temperature": settings.model.temperature,
                "top_p": settings.model.top_p,
                "top_k": settings.model.top_k,
                "do_sample": True,
                "pad_token_id": self._tokenizer.eos_token_id
            }
            
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.model.max_length
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    **params,
                    eos_token_id=self._tokenizer.eos_token_id,
                    return_dict_in_generate=True
                )
            
            # Extract response
            response_text = self._tokenizer.decode(
                outputs.sequences[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Update conversation history
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            self.conversation_history[session_id].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text}
            ])
            
            # Keep history manageable
            if len(self.conversation_history[session_id]) > self.max_history_length * 2:
                self.conversation_history[session_id] = self.conversation_history[session_id][-self.max_history_length:]
            
            # Clean up
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            
            response = PredictionResponse(
                request_id=request.request_id,
                model_id=self.model_id,
                outputs=response_text,
                processing_time=processing_time,
                metadata={
                    "session_id": session_id,
                    "conversation_length": len(self.conversation_history[session_id]),
                    "message_history_truncated": len(self.conversation_history[session_id]) == self.max_history_length * 2
                }
            )
            
            # Cache response
            self.cache.set(self.model_id, cache_key, {}, {
                "request_id": response.request_id,
                "model_id": response.model_id,
                "outputs": response.outputs,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "timestamp": response.timestamp.isoformat(),
                "metadata": response.metadata
            })
            
            # Update metrics
            self.update_metrics(processing_time, True)
            
            self.model_logger.log_prediction(
                self.name,
                len(message),
                len(response_text),
                processing_time
            )
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            
            self.logger.error(
                f"Conversation generation failed",
                request_id=request.request_id,
                error=str(e)
            )
            raise
    
    def get_model_info(self) -> ModelMetadata:
        """Get model metadata."""
        return ModelMetadata(
            model_id=self.model_id,
            name=self.name,
            version=get_settings().model.model_revision,
            prediction_type=PredictionType.CONVERSATION,
            max_length=get_settings().model.max_length,
            device=self._device,
            tags=["conversation", "chat", "dialogue"],
            description="Conversation model for interactive dialogue"
        )
    
    def clear_conversation_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.conversation_history.pop(session_id, None)
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get conversation summary for a session."""
        history = self.conversation_history.get(session_id, [])
        return {
            "session_id": session_id,
            "message_count": len(history) // 2,  # Round trip messages
            "last_message": history[-1]["content"] if history else None,
            "active_sessions": len(self.conversation_history)
        }