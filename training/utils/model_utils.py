"""
Model Utilities for LoRA/PEFT Training
Comprehensive utility functions for model loading, saving, and management.
"""

import os
import json
import logging
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import pickle
import warnings

# PEFT and transformers imports
from peft import (
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
    AdaLoraConfig,
    IA3Config,
    PromptTuningConfig,
    PromptTuningInit,
    PeftConfig,
    PeftType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    cast_mixed_precision_params
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from transformers.modeling_utils import (
    load_sharded_checkpoint,
    shard_checkpoint
)

# HuggingFace Hub
try:
    from huggingface_hub import (
        HfApi,
        ModelFilter,
        ModelCard,
        login,
        snapshot_download,
        upload_folder
    )
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

# Quantization libraries
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Container for model information."""
    model_name: str
    model_type: str
    num_parameters: Optional[int] = None
    num_trainable_parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    quantization_info: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    device: Optional[str] = None
    dtype: Optional[str] = None

@dataclass
class CheckpointInfo:
    """Container for checkpoint information."""
    checkpoint_path: str
    epoch: Optional[int] = None
    step: Optional[int] = None
    global_step: Optional[int] = None
    metrics: Optional[Dict[str, float]] = None
    training_args: Optional[Dict[str, Any]] = None
    lora_config: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

class ModelLoadingError(Exception):
    """Exception raised for model loading errors."""
    pass

class ModelSavingError(Exception):
    """Exception raised for model saving errors."""
    pass

class QuantizationError(Exception):
    """Exception raised for quantization errors."""
    pass

class ModelInfoCollector:
    """Utility class for collecting model information."""
    
    @staticmethod
    def get_model_info(model: nn.Module, tokenizer: Optional[AutoTokenizer] = None) -> ModelInfo:
        """Get comprehensive model information."""
        model_name = getattr(model, 'name_or_path', 'unknown')
        model_type = getattr(model, 'config', {}).get('model_type', 'unknown') if hasattr(model, 'config') else 'unknown'
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size
        model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        # Get device info
        device = next(model.parameters()).device.type if next(model.parameters(), None) else None
        
        # Get dtype info
        dtype = str(next(model.parameters()).dtype) if next(model.parameters(), None) else None
        
        # Get quantization info
        quantization_info = ModelInfoCollector._get_quantization_info(model)
        
        # Get LoRA config
        lora_config = ModelInfoCollector._get_lora_config(model)
        
        return ModelInfo(
            model_name=model_name,
            model_type=model_type,
            num_parameters=total_params,
            num_trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            quantization_info=quantization_info,
            lora_config=lora_config,
            device=device,
            dtype=dtype
        )
    
    @staticmethod
    def _get_quantization_info(model: nn.Module) -> Optional[Dict[str, Any]]:
        """Extract quantization information from model."""
        quantization_info = {}
        
        # Check for quantization attributes
        if hasattr(model, 'is_quantized'):
            quantization_info['is_quantized'] = model.is_quantized
        
        # Check for quantization configuration
        if hasattr(model, ' quantization_config'):
            quantization_info['quantization_config'] = model.quantization_config
        
        # Check for BitsAndBytesConfig
        if hasattr(model, 'bnb_config'):
            config = model.bnb_config
            if config:
                quantization_info.update({
                    'load_in_8bit': config.load_in_8bit,
                    'load_in_4bit': config.load_in_4bit,
                    'bnb_4bit_quant_type': getattr(config, 'bnb_4bit_quant_type', None),
                    'bnb_4bit_use_double_quant': getattr(config, 'bnb_4bit_use_double_quant', None)
                })
        
        return quantization_info if quantization_info else None
    
    @staticmethod
    def _get_lora_config(model: nn.Module) -> Optional[Dict[str, Any]]:
        """Extract LoRA configuration from model."""
        if hasattr(model, 'peft_config'):
            config = model.peft_config
            if config:
                return {
                    'peft_type': str(config.peft_type) if hasattr(config, 'peft_type') else None,
                    'base_model_name_or_path': getattr(config, 'base_model_name_or_path', None),
                    'task_type': str(config.task_type) if hasattr(config, 'task_type') else None,
                    'inference_mode': getattr(config, 'inference_mode', None)
                }
        
        return None
    
    @staticmethod
    def print_model_summary(model: nn.Module, tokenizer: Optional[AutoTokenizer] = None) -> None:
        """Print comprehensive model summary."""
        info = ModelInfoCollector.get_model_info(model, tokenizer)
        
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print(f"Model Name: {info.model_name}")
        print(f"Model Type: {info.model_type}")
        print(f"Total Parameters: {info.num_parameters:,}")
        print(f"Trainable Parameters: {info.num_trainable_parameters:,}")
        print(f"Model Size: {info.model_size_mb:.2f} MB")
        print(f"Device: {info.device}")
        print(f"Dtype: {info.dtype}")
        
        if info.quantization_info:
            print("\nQuantization Info:")
            for key, value in info.quantization_info.items():
                print(f"  {key}: {value}")
        
        if info.lora_config:
            print("\nLoRA Config:")
            for key, value in info.lora_config.items():
                print(f"  {key}: {value}")
        
        print("="*50 + "\n")

class ModelLoader:
    """Utility class for loading models with various configurations."""
    
    @staticmethod
    def load_base_model(
        model_name: str,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        quantization_config: Optional[BitsAndBytesConfig] = None
    ) -> PreTrainedModel:
        """Load base model with various optimization options."""
        logger.info(f"Loading base model: {model_name}")
        
        try:
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
                "use_auth_token": use_auth_token,
                "cache_dir": cache_dir,
                "torch_dtype": torch_dtype
            }
            
            if device_map:
                model_kwargs["device_map"] = device_map
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
                
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            logger.info(f"Base model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise ModelLoadingError(f"Failed to load base model: {e}")
    
    @staticmethod
    def load_tokenizer(
        model_name: str,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        padding_side: str = "right"
    ) -> AutoTokenizer:
        """Load tokenizer with proper configuration."""
        logger.info(f"Loading tokenizer for: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
                cache_dir=cache_dir
            )
            
            # Configure tokenizer
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            tokenizer.padding_side = padding_side
            
            logger.info("Tokenizer loaded successfully")
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise ModelLoadingError(f"Failed to load tokenizer: {e}")
    
    @staticmethod
    def load_peft_model(
        base_model: PreTrainedModel,
        peft_model_path: str
    ) -> PeftModelForCausalLM:
        """Load PEFT model from saved path."""
        logger.info(f"Loading PEFT model from: {peft_model_path}")
        
        try:
            peft_model = PeftModelForCausalLM.from_pretrained(base_model, peft_model_path)
            logger.info("PEFT model loaded successfully")
            return peft_model
            
        except Exception as e:
            logger.error(f"Failed to load PEFT model: {e}")
            raise ModelLoadingError(f"Failed to load PEFT model: {e}")

class ModelSaver:
    """Utility class for saving models with various options."""
    
    @staticmethod
    def save_model(
        model: nn.Module,
        tokenizer: AutoTokenizer,
        save_directory: str,
        save_safetensors: bool = True,
        save_peft_format: bool = True,
        use_auth_token: Optional[str] = None,
        commit_message: Optional[str] = None,
        private: bool = False,
        create_pr: bool = False
    ) -> None:
        """Save model with comprehensive options."""
        logger.info(f"Saving model to: {save_directory}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_directory, exist_ok=True)
            
            # Save model
            if save_peft_format and hasattr(model, 'save_pretrained'):
                model.save_pretrained(save_directory, safe_serialization=save_safetensors)
            elif hasattr(model, 'save_pretrained'):
                model.save_pretrained(save_directory, safe_serialization=save_safetensors)
            else:
                torch.save(model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
            
            # Save tokenizer
            tokenizer.save_pretrained(save_directory)
            
            # Upload to HuggingFace Hub if specified
            if use_auth_token and commit_message:
                ModelSaver._upload_to_hub(
                    save_directory=save_directory,
                    use_auth_token=use_auth_token,
                    commit_message=commit_message,
                    private=private,
                    create_pr=create_pr
                )
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise ModelSavingError(f"Failed to save model: {e}")
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        tokenizer: AutoTokenizer,
        checkpoint_dir: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        global_step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        training_args: Optional[Dict[str, Any]] = None,
        lora_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save training checkpoint with metadata."""
        logger.info(f"Saving checkpoint to: {checkpoint_dir}")
        
        try:
            # Create checkpoint directory
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            if hasattr(model, 'save_pretrained'):
                model.save_pretrained(checkpoint_dir)
            else:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
            
            # Save tokenizer
            tokenizer.save_pretrained(checkpoint_dir)
            
            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                checkpoint_path=checkpoint_dir,
                epoch=epoch,
                step=step,
                global_step=global_step,
                metrics=metrics,
                training_args=training_args,
                lora_config=lora_config,
                timestamp=str(torch.timestamp() if torch.timestamp else "unknown")
            )
            
            # Save checkpoint metadata
            with open(os.path.join(checkpoint_dir, "checkpoint_info.json"), 'w') as f:
                json.dump(asdict(checkpoint_info), f, indent=2, default=str)
            
            # Save training arguments if provided
            if training_args:
                with open(os.path.join(checkpoint_dir, "training_args.json"), 'w') as f:
                    json.dump(training_args, f, indent=2, default=str)
            
            # Save LoRA config if provided
            if lora_config:
                with open(os.path.join(checkpoint_dir, "lora_config.json"), 'w') as f:
                    json.dump(lora_config, f, indent=2)
            
            logger.info("Checkpoint saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise ModelSavingError(f"Failed to save checkpoint: {e}")
    
    @staticmethod
    def _upload_to_hub(
        save_directory: str,
        use_auth_token: str,
        commit_message: str,
        private: bool = False,
        create_pr: bool = False
    ) -> None:
        """Upload model to HuggingFace Hub."""
        if not HUGGINGFACE_HUB_AVAILABLE:
            logger.warning("HuggingFace Hub not available, skipping upload")
            return
        
        try:
            # Login to HuggingFace
            login(token=use_auth_token)
            
            # Create API instance
            api = HfApi()
            
            # Extract model name from path
            model_name = os.path.basename(save_directory)
            
            # Upload folder
            api.upload_folder(
                folder_path=save_directory,
                repo_id=model_name,
                commit_message=commit_message,
                private=private,
                create_pr=create_pr
            )
            
            logger.info(f"Model uploaded to Hub: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to upload to HuggingFace Hub: {e}")

class ModelConverter:
    """Utility class for model conversion and optimization."""
    
    @staticmethod
    def convert_to_8bit(
        model: PreTrainedModel,
        bnb_config: Optional[BitsAndBytesConfig] = None
    ) -> PreTrainedModel:
        """Convert model to 8-bit quantization."""
        if not BNB_AVAILABLE:
            raise QuantizationError("bitsandbytes not available for 8-bit conversion")
        
        logger.info("Converting model to 8-bit quantization")
        
        try:
            if bnb_config is None:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Prepare model for quantization
            model = prepare_model_for_kbit_training(model)
            
            # Convert to 8-bit
            model = model.bnb_quantization_config = bnb_config
            
            logger.info("Model converted to 8-bit successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to convert to 8-bit: {e}")
            raise QuantizationError(f"Failed to convert to 8-bit: {e}")
    
    @staticmethod
    def convert_to_4bit(
        model: PreTrainedModel,
        bnb_config: Optional[BitsAndBytesConfig] = None
    ) -> PreTrainedModel:
        """Convert model to 4-bit quantization."""
        if not BNB_AVAILABLE:
            raise QuantizationError("bitsandbytes not available for 4-bit conversion")
        
        logger.info("Converting model to 4-bit quantization")
        
        try:
            if bnb_config is None:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            # Prepare model for quantization
            model = prepare_model_for_kbit_training(model)
            
            # Convert to 4-bit
            model = model.bnb_quantization_config = bnb_config
            
            logger.info("Model converted to 4-bit successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to convert to 4-bit: {e}")
            raise QuantizationError(f"Failed to convert to 4-bit: {e}")
    
    @staticmethod
    def merge_and_unload(
        model: PeftModelForCausalLM
    ) -> PreTrainedModel:
        """Merge LoRA weights into base model and unload LoRA adapters."""
        logger.info("Merging LoRA adapters and unloading")
        
        try:
            # Merge LoRA weights
            merged_model = model.merge_and_unload()
            
            logger.info("LoRA adapters merged and unloaded successfully")
            return merged_model
            
        except Exception as e:
            logger.error(f"Failed to merge and unload: {e}")
            raise ModelSavingError(f"Failed to merge and unload: {e}")
    
    @staticmethod
    def save_merged_model(
        model: PeftModelForCausalLM,
        tokenizer: AutoTokenizer,
        save_directory: str,
        save_safetensors: bool = True
    ) -> None:
        """Save merged model (base + LoRA) as a single model."""
        logger.info("Saving merged model")
        
        try:
            # Merge LoRA weights
            merged_model = ModelConverter.merge_and_unload(model)
            
            # Save merged model
            ModelSaver.save_model(
                model=merged_model,
                tokenizer=tokenizer,
                save_directory=save_directory,
                save_safetensors=save_safetensors,
                save_peft_format=False
            )
            
            logger.info("Merged model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save merged model: {e}")
            raise ModelSavingError(f"Failed to save merged model: {e}")

class ModelValidator:
    """Utility class for model validation and inspection."""
    
    @staticmethod
    def validate_model_integrity(model: nn.Module) -> bool:
        """Validate model integrity and structure."""
        logger.info("Validating model integrity")
        
        try:
            # Check if model has parameters
            if not list(model.parameters()):
                logger.warning("Model has no parameters")
                return False
            
            # Check for NaN values in parameters
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    logger.warning(f"NaN values found in parameter: {name}")
                    return False
            
            # Test forward pass with dummy input
            model.eval()
            with torch.no_grad():
                batch_size, seq_length = 2, 10
                input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=next(model.parameters()).device)
                attention_mask = torch.ones(batch_size, seq_length, device=next(model.parameters()).device)
                
                try:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    if torch.isnan(outputs.logits).any():
                        logger.warning("NaN values in model output")
                        return False
                except Exception as e:
                    logger.warning(f"Forward pass failed: {e}")
                    return False
            
            logger.info("Model integrity validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Model integrity validation failed: {e}")
            return False
    
    @staticmethod
    def check_gpu_memory() -> Dict[str, float]:
        """Check GPU memory usage."""
        memory_info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.cuda.device(i)
                device_name = torch.cuda.get_device_name(i)
                
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                memory_info[f"device_{i}_{device_name}"] = {
                    "allocated_gb": memory_allocated,
                    "reserved_gb": memory_reserved,
                    "total_gb": memory_total,
                    "free_gb": memory_total - memory_reserved
                }
        
        return memory_info
    
    @staticmethod
    def get_model_architecture_summary(model: nn.Module) -> Dict[str, Any]:
        """Get detailed model architecture summary."""
        summary = {
            "model_type": type(model).__name__,
            "total_layers": 0,
            "layer_types": {},
            "parameter_distribution": {},
            "trainable_parameters": 0,
            "total_parameters": 0
        }
        
        # Count parameters by layer type
        for name, param in model.named_parameters():
            summary["total_parameters"] += param.numel()
            if param.requires_grad:
                summary["trainable_parameters"] += param.numel()
            
            # Extract layer type from parameter name
            layer_type = name.split('.')[0]
            if layer_type not in summary["layer_types"]:
                summary["layer_types"][layer_type] = 0
            summary["layer_types"][layer_type] += param.numel()
        
        return summary

class ModelManager:
    """Main model management class combining all utilities."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self.model_info_collector = ModelInfoCollector()
        self.model_loader = ModelLoader()
        self.model_saver = ModelSaver()
        self.model_converter = ModelConverter()
        self.model_validator = ModelValidator()
    
    def setup_model(
        self,
        model_name: str,
        lora_config: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Dict[str, Any]] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None
    ) -> Tuple[PreTrainedModel, AutoTokenizer, Optional[PeftModel]]:
        """Setup complete model pipeline with LoRA and quantization."""
        logger.info("Setting up model pipeline")
        
        try:
            # Load tokenizer
            tokenizer = self.model_loader.load_tokenizer(
                model_name=model_name,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
                cache_dir=self.cache_dir
            )
            
            # Load base model
            base_model = self.model_loader.load_base_model(
                model_name=model_name,
                trust_remote_code=trust_remote_code,
                use_auth_token=use_auth_token,
                cache_dir=self.cache_dir,
                device_map=device_map
            )
            
            # Add LoRA if configured
            peft_model = None
            if lora_config:
                logger.info("Adding LoRA adapters")
                
                # Create LoRA config
                from peft import LoraConfig
                peft_config = LoraConfig(**lora_config)
                
                # Add LoRA to model
                peft_model = get_peft_model(base_model, peft_config)
            
            # Apply quantization if configured
            if quantization_config:
                logger.info("Applying quantization")
                from transformers import BitsAndBytesConfig
                
                if quantization_config.get("use_4bit"):
                    base_model = self.model_converter.convert_to_4bit(
                        base_model,
                        BitsAndBytesConfig(**quantization_config)
                    )
                elif quantization_config.get("use_8bit"):
                    base_model = self.model_converter.convert_to_8bit(
                        base_model,
                        BitsAndBytesConfig(**quantization_config)
                    )
            
            # Final model selection
            model = peft_model if peft_model else base_model
            
            # Validate model
            self.model_validator.validate_model_integrity(model)
            
            # Print model summary
            self.model_info_collector.print_model_summary(model, tokenizer)
            
            logger.info("Model pipeline setup completed successfully")
            return model, tokenizer, peft_model
            
        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            raise

# Export main classes and utilities
__all__ = [
    "ModelInfo",
    "CheckpointInfo",
    "ModelInfoCollector",
    "ModelLoader", 
    "ModelSaver",
    "ModelConverter",
    "ModelValidator",
    "ModelManager",
    "ModelLoadingError",
    "ModelSavingError",
    "QuantizationError"
]