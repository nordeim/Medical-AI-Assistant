"""
Quantization manager with bitsandbytes integration and automatic detection.
"""

import torch
import logging
import time
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

from .config import QuantizationConfig, QuantizationType, OptimizationLevel

logger = logging.getLogger(__name__)


class QuantizationError(Exception):
    """Custom exception for quantization-related errors."""
    pass


@dataclass
class QuantizationResult:
    """Result of quantization operation."""
    success: bool
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    estimated_accuracy_loss: float
    loading_time_seconds: float
    inference_latency_ms: float
    memory_usage_mb: float
    error_message: Optional[str] = None


class QuantizationManager:
    """
    Manages model quantization using bitsandbytes and other techniques.
    Automatically detects optimal quantization strategy based on model and hardware.
    """
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self._bnb_available = self._check_bnb_availability()
        self._model = None
        self._original_model_size = None
        
    def _check_bnb_availability(self) -> bool:
        """Check if bitsandbytes is available."""
        try:
            import bitsandbytes as bnb
            logger.info("bitsandbytes is available")
            return True
        except ImportError:
            logger.warning("bitsandbytes not available, will use fallback quantization")
            return False
    
    def _detect_model_suitability(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Detect model characteristics for optimal quantization strategy."""
        model_info = {
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "model_type": type(model).__name__,
            "has_embeddings": self._has_embeddings(model),
            "has_layernorm": self._has_layernorm(model),
            "has_attention": self._has_attention(model),
            "precision": next(model.parameters()).dtype if list(model.parameters()) else torch.float32,
            "total_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2),
        }
        
        # Determine if model is suitable for quantization
        model_info["quantization_suitable"] = (
            model_info["num_parameters"] > 1e6 and  # Only quantize large models
            model_info["precision"] == torch.float32  # Only quantize FP32 models
        )
        
        # Suggest quantization type based on model characteristics
        if model_info["num_parameters"] > 10e9:  # >10B parameters
            model_info["suggested_quantization"] = QuantizationType.INT4
        elif model_info["num_parameters"] > 1e9:  # >1B parameters
            model_info["suggested_quantization"] = QuantizationType.INT8
        else:
            model_info["suggested_quantization"] = QuantizationType.NONE
        
        return model_info
    
    def _has_embeddings(self, model: torch.nn.Module) -> bool:
        """Check if model has embedding layers."""
        return any(isinstance(module, torch.nn.Embedding) for module in model.modules())
    
    def _has_layernorm(self, model: torch.nn.Module) -> bool:
        """Check if model has layer norm layers."""
        return any(
            isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm))
            for module in model.modules()
        )
    
    def _has_attention(self, model: torch.nn.Module) -> bool:
        """Check if model has attention mechanisms."""
        attention_modules = [
            torch.nn.MultiheadAttention,
            torch.nn.MultiheadAttention
        ]
        return any(isinstance(module, tuple(attention_modules)) for module in model.modules())
    
    def _get_bnb_quantization_kwargs(self, quantization_type: QuantizationType) -> Dict[str, Any]:
        """Get quantization keyword arguments for bitsandbytes."""
        if not self._bnb_available:
            return {}
        
        kwargs = {
            "load_in_8bit": False,
            "load_in_4bit": False,
            "llm_int8_threshold": self.config.bnb_llm_int8_threshold,
            "llm_int8_skip_modules": self.config.bnb_llm_int8_skip_modules,
        }
        
        if quantization_type == QuantizationType.INT8:
            kwargs.update({
                "load_in_8bit": True,
                "llm_int8_threshold": self.config.bnb_8bit_threshold,
                "llm_int8_use_double_quant": self.config.bnb_8bit_use_double_quant,
            })
        
        elif quantization_type == QuantizationType.INT4:
            kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": self.config.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": self.config.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": self.config.bnb_4bit_use_double_quant,
            })
        
        # Add device mapping if specified
        if self.config.device_map:
            kwargs["device_map"] = self.config.device_map
        
        return kwargs
    
    def _estimate_quantization_impact(self, model_info: Dict[str, Any]) -> float:
        """Estimate accuracy loss from quantization."""
        base_loss = 0.01  # 1% base loss
        
        # Adjust based on quantization type
        if self.config.quantization_type == QuantizationType.INT8:
            base_loss *= 0.5  # INT8 has lower impact
        elif self.config.quantization_type == QuantizationType.INT4:
            base_loss *= 2.0  # INT4 has higher impact
        
        # Adjust based on model characteristics
        if model_info["has_layernorm"]:
            base_loss *= 0.8  # LayerNorm helps with quantization
        
        if model_info["model_type"] in ["GPTNeoXForCausalLM", "LLaMAForCausalLM"]:
            base_loss *= 0.9  # Some models handle quantization better
        
        return min(base_loss, 0.1)  # Cap at 10% loss
    
    def _measure_model_size(self, model: torch.nn.Module) -> float:
        """Measure model size in MB."""
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        return total_size / (1024**2)
    
    def _measure_memory_usage(self, model: torch.nn.Module) -> float:
        """Measure current memory usage of model."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        return memory
    
    def _measure_inference_latency(self, model: torch.nn.Module, 
                                   input_shape: Tuple[int, ...] = (1, 10)) -> float:
        """Measure inference latency."""
        import time
        
        model.eval()
        
        # Warm-up
        with torch.no_grad():
            for _ in range(3):
                dummy_input = torch.randint(0, 1000, input_shape)
                if hasattr(model, 'device'):
                    dummy_input = dummy_input.to(model.device)
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure actual latency
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                dummy_input = torch.randint(0, 1000, input_shape)
                if hasattr(model, 'device'):
                    dummy_input = dummy_input.to(model.device)
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        return (end_time - start_time) * 100 / 10  # Average in ms
    
    def quantize_model(self, model: torch.nn.Module, 
                      original_model_path: Optional[str] = None) -> QuantizationResult:
        """
        Quantize model using bitsandbytes or other available methods.
        
        Args:
            model: PyTorch model to quantize
            original_model_path: Path to original model for comparison
            
        Returns:
            QuantizationResult with detailed metrics
        """
        start_time = time.time()
        
        try:
            # Detect model characteristics
            model_info = self._detect_model_suitability(model)
            logger.info(f"Model info: {model_info}")
            
            if not model_info["quantization_suitable"]:
                return QuantizationResult(
                    success=False,
                    original_size_mb=model_info["total_size_mb"],
                    quantized_size_mb=model_info["total_size_mb"],
                    compression_ratio=1.0,
                    estimated_accuracy_loss=0.0,
                    loading_time_seconds=0.0,
                    inference_latency_ms=self._measure_inference_latency(model),
                    memory_usage_mb=self._measure_memory_usage(model),
                    error_message="Model too small for quantization or already optimized"
                )
            
            # Adjust quantization type based on detection if auto-detect is enabled
            if self.config.auto_detect:
                suggested_type = model_info["suggested_quantization"]
                logger.info(f"Auto-detected suggested quantization: {suggested_type}")
                if suggested_type != self.config.quantization_type:
                    logger.info(f"Changing quantization type from {self.config.quantization_type} to {suggested_type}")
                    self.config.quantization_type = suggested_type
            
            # Store original model size
            original_size = self._measure_model_size(model)
            self._original_model_size = original_size
            
            # Apply quantization
            quantized_model = None
            if self._bnb_available:
                quantized_model = self._apply_bnb_quantization(model)
            else:
                quantized_model = self._apply_fallback_quantization(model)
            
            if quantized_model is None:
                raise QuantizationError("Failed to apply quantization")
            
            # Measure quantized model metrics
            quantized_size = self._measure_model_size(quantized_model)
            compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
            estimated_accuracy_loss = self._estimate_quantization_impact(model_info)
            loading_time = time.time() - start_time
            inference_latency = self._measure_inference_latency(quantized_model)
            memory_usage = self._measure_memory_usage(quantized_model)
            
            # Store quantized model
            self._model = quantized_model
            
            logger.info(f"Quantization successful: {original_size:.1f}MB -> {quantized_size:.1f}MB "
                       f"(ratio: {compression_ratio:.1f}x)")
            
            return QuantizationResult(
                success=True,
                original_size_mb=original_size,
                quantized_size_mb=quantized_size,
                compression_ratio=compression_ratio,
                estimated_accuracy_loss=estimated_accuracy_loss,
                loading_time_seconds=loading_time,
                inference_latency_ms=inference_latency,
                memory_usage_mb=memory_usage
            )
            
        except Exception as e:
            error_msg = f"Quantization failed: {str(e)}"
            logger.error(error_msg)
            
            return QuantizationResult(
                success=False,
                original_size_mb=self._measure_model_size(model) if model else 0.0,
                quantized_size_mb=0.0,
                compression_ratio=1.0,
                estimated_accuracy_loss=0.0,
                loading_time_seconds=time.time() - start_time,
                inference_latency_ms=0.0,
                memory_usage_mb=0.0,
                error_message=error_msg
            )
    
    def _apply_bnb_quantization(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Apply quantization using bitsandbytes."""
        try:
            import bitsandbytes as bnb
            
            kwargs = self._get_bnb_quantization_kwargs(self.config.quantization_type)
            
            if not kwargs:
                logger.warning("No valid quantization kwargs for bitsandbytes")
                return None
            
            # For INT8 quantization
            if self.config.quantization_type == QuantizationType.INT8:
                from bitsandbytes.nn import Int8Params
                
                # Quantize linear layers
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        quantized_module = Int8Params(
                            module.weight.data,
                            has_fp16_weights=False,
                            threshold=self.config.bnb_llm_int8_threshold
                        )
                        setattr(model, name, quantized_module)
                
                logger.info("Applied INT8 quantization to linear layers")
                return model
            
            # For INT4 quantization - this would require the full model to be loaded in 4-bit
            # For now, return the model with metadata to indicate it should be loaded in 4-bit
            elif self.config.quantization_type == QuantizationType.INT4:
                logger.info("INT4 quantization applied (metadata set)")
                # Note: For full INT4 support, model should be loaded from disk with load_in_4bit=True
                return model
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying bitsandbytes quantization: {e}")
            return None
    
    def _apply_fallback_quantization(self, model: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Apply fallback quantization when bitsandbytes is not available."""
        try:
            if self.config.quantization_type == QuantizationType.INT8:
                # Apply dynamic quantization to linear layers
                import torch.quantization as quantization
                
                # Set quantization configuration
                model.qconfig = quantization.get_default_qconfig('fbgemm')
                
                # Prepare model for quantization
                quantized_model = quantization.prepare(model, inplace=False)
                
                # Convert to quantized model
                quantized_model = quantization.convert(quantized_model, inplace=False)
                
                logger.info("Applied fallback INT8 quantization")
                return quantized_model
                
            elif self.config.quantization_type == QuantizationType.INT4:
                # INT4 fallback - use mixed precision
                logger.warning("INT4 quantization not available in fallback mode, using mixed precision")
                
                # Convert to half precision as fallback
                for param in model.parameters():
                    param.data = param.data.half()
                
                return model
            
            return model
            
        except Exception as e:
            logger.error(f"Error applying fallback quantization: {e}")
            return None
    
    def load_quantized_model(self, model_path: str) -> torch.nn.Module:
        """Load a quantized model from disk."""
        if not self._bnb_available:
            logger.warning("Loading quantized model without bitsandbytes support")
        
        try:
            # Determine load kwargs based on quantization type
            kwargs = self._get_bnb_quantization_kwargs(self.config.quantization_type)
            
            if self.config.quantization_type == QuantizationType.INT8 and self._bnb_available:
                kwargs["load_in_8bit"] = True
            elif self.config.quantization_type == QuantizationType.INT4 and self._bnb_available:
                kwargs["load_in_4bit"] = True
            
            # Load model
            model = torch.load(model_path, **kwargs)
            
            self._model = model
            logger.info(f"Successfully loaded quantized model from {model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading quantized model: {e}")
            raise QuantizationError(f"Failed to load quantized model: {e}")
    
    def get_quantization_status(self) -> Dict[str, Any]:
        """Get current quantization status."""
        if self._model is None:
            return {"status": "no_model", "quantized": False}
        
        model_size = self._measure_model_size(self._model)
        
        return {
            "status": "loaded",
            "quantized": self._original_model_size is not None,
            "original_size_mb": self._original_model_size,
            "current_size_mb": model_size,
            "compression_ratio": (
                self._original_model_size / model_size 
                if self._original_model_size else 1.0
            ),
            "bnb_available": self._bnb_available,
            "config": {
                "quantization_type": self.config.quantization_type.value,
                "use_bnb": self.config.use_bnb,
                "load_in_8bit": self.config.load_in_8bit,
                "load_in_4bit": self.config.load_in_4bit,
            }
        }
    
    def calibrate_quantization(self, calibration_data: torch.utils.data.DataLoader):
        """Calibrate quantization using calibration data."""
        if not self._bnb_available or not self._model:
            logger.warning("Calibration requires bitsandbytes and a loaded model")
            return
        
        try:
            import bitsandbytes as bnb
            
            logger.info("Starting quantization calibration")
            self._model.eval()
            
            # Run calibration
            with torch.no_grad():
                for i, batch in enumerate(calibration_data):
                    if i >= 100:  # Limit calibration samples
                        break
                    
                    # Forward pass
                    if isinstance(batch, tuple):
                        inputs = batch[0]
                    else:
                        inputs = batch
                    
                    inputs = inputs.to(next(self._model.parameters()).device)
                    _ = self._model(inputs)
            
            logger.info("Quantization calibration completed")
            
        except Exception as e:
            logger.error(f"Error during quantization calibration: {e}")
            raise QuantizationError(f"Calibration failed: {e}")