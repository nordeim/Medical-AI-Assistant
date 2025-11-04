"""
Model Inference Performance Optimization for Medical AI
Implements 4-bit/8-bit quantization and batch processing for optimized model performance
"""

import torch
import torch.nn as nn
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from peft import PeftModel, PeftConfig
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import time
import gc
import psutil
import threading

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformanceMetrics:
    """Metrics for model inference performance"""
    model_name: str
    inference_time: float
    throughput: float  # tokens per second
    memory_usage: float  # GB
    batch_size: int
    quantization_level: str
    accuracy_impact: Optional[float] = None

class QuantizedModelManager:
    """
    Manages quantized models for optimal performance
    Supports 4-bit and 8-bit quantization for medical AI models
    """
    
    def __init__(self, device_map: str = "auto", torch_dtype: str = "float16"):
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.loaded_models = {}
        self.performance_metrics = {}
        self.gpu_available = torch.cuda.is_available()
        self.quantization_configs = {
            '4bit': {
                'load_in_4bit': True,
                'load_in_8bit': False,
                'bnb_4bit_compute_dtype': torch_dtype,
                'bnb_4bit_use_double_quant': True,
                'bnb_4bit_quant_type': "nf4"
            },
            '8bit': {
                'load_in_4bit': False,
                'load_in_8bit': True,
                'bnb_8bit_use_double_quant': False,
                'bnb_8bit_quant_type': "dynamic"
            },
            'fp16': {
                'torch_dtype': torch_dtype,
                'device_map': device_map
            }
        }
    
    async def load_quantized_model(self, model_name: str, 
                                  quantization_level: str = '4bit',
                                  lora_adapter_path: Optional[str] = None) -> nn.Module:
        """
        Load model with specified quantization level
        """
        try:
            logger.info(f"Loading {model_name} with {quantization_level} quantization")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization
            config = self.quantization_configs[quantization_level]
            
            if quantization_level in ['4bit', '8bit']:
                from accelerate import load_checkpoint_and_dispatch
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **config,
                        trust_remote_code=True
                    )
                model = load_checkpoint_and_dispatch(
                    model, 
                    device_map=self.device_map,
                    no_split_module_classes=["LlamaDecoderLayer"]
                )
            else:
                # Load base model for fp16
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **config,
                    trust_remote_code=True
                )
            
            # Load LoRA adapter if provided
            if lora_adapter_path:
                model = PeftModel.from_pretrained(model, lora_adapter_path)
                logger.info(f"Loaded LoRA adapter from {lora_adapter_path}")
            
            # Store loaded model
            model_key = f"{model_name}_{quantization_level}"
            self.loaded_models[model_key] = {
                'model': model,
                'tokenizer': tokenizer,
                'quantization_level': quantization_level,
                'loaded_at': datetime.now(),
                'memory_usage': self._get_model_memory_usage(model)
            }
            
            logger.info(f"Successfully loaded {model_name} with {quantization_level} quantization")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load quantized model {model_name}: {e}")
            raise
    
    async def inference_with_metrics(self, model_name: str, 
                                   prompt: Union[str, List[str]],
                                   quantization_level: str = '4bit',
                                   batch_size: int = 1,
                                   max_tokens: int = 512) -> Tuple[Union[str, List[str]], ModelPerformanceMetrics]:
        """
        Perform inference with performance metrics
        """
        model_key = f"{model_name}_{quantization_level}"
        
        if model_key not in self.loaded_models:
            await self.load_quantized_model(model_name, quantization_level)
        
        model_info = self.loaded_models[model_key]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Measure inference time
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Handle batch inference
            if isinstance(prompt, list):
                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=batch_size
                )
                
                # Decode outputs
                results = []
                for output in outputs:
                    decoded = tokenizer.decode(output, skip_special_tokens=True)
                    results.append(decoded)
            else:
                # Single inference
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
                
                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            inference_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            # Calculate throughput
            total_tokens = len(tokenizer.encode(result if isinstance(result, str) else result[0]))
            throughput = total_tokens / inference_time if inference_time > 0 else 0
            
            # Create metrics
            metrics = ModelPerformanceMetrics(
                model_name=model_name,
                inference_time=inference_time,
                throughput=throughput,
                memory_usage=memory_used,
                batch_size=batch_size,
                quantization_level=quantization_level
            )
            
            # Store metrics
            self.performance_metrics[model_key] = metrics
            
            logger.debug(f"Inference completed in {inference_time:.2f}s, "
                        f"throughput: {throughput:.1f} tokens/s")
            
            return result, metrics
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
        finally:
            # Clean up memory
            if self.gpu_available:
                torch.cuda.empty_cache()
            gc.collect()
    
    def _get_model_memory_usage(self, model: nn.Module) -> float:
        """Get memory usage of loaded model"""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            return psutil.Process().memory_info().rss / (1024**3)  # GB
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        if self.gpu_available:
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            return psutil.Process().memory_info().rss / (1024**3)  # GB
    
    async def optimize_batch_processing(self, model_name: str, 
                                       prompts: List[str],
                                       batch_size: int = 4,
                                       quantization_level: str = '4bit') -> List[Tuple[str, ModelPerformanceMetrics]]:
        """
        Process multiple prompts in optimized batches
        """
        model_key = f"{model_name}_{quantization_level}"
        
        # Ensure model is loaded
        if model_key not in self.loaded_models:
            await self.load_quantized_model(model_name, quantization_level)
        
        results = []
        model_info = self.loaded_models[model_key]
        model = model_info['model']
        tokenizer = model_info['tokenizer']
        
        # Process in batches for optimal performance
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            
            try:
                # Batch inference
                start_time = time.time()
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                
                # Move to device
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode results
                batch_results = []
                for output in outputs:
                    decoded = tokenizer.decode(output, skip_special_tokens=True)
                    batch_results.append(decoded)
                
                end_time = time.time()
                
                # Create metrics for batch
                batch_metrics = ModelPerformanceMetrics(
                    model_name=model_name,
                    inference_time=(end_time - start_time) / len(batch),
                    throughput=sum(len(tokenizer.encode(r)) for r in batch_results) / (end_time - start_time),
                    memory_usage=self._get_model_memory_usage(model),
                    batch_size=len(batch),
                    quantization_level=quantization_level
                )
                
                results.extend([(result, batch_metrics) for result in batch_results])
                
            except Exception as e:
                logger.error(f"Batch processing failed for batch {i//batch_size}: {e}")
                continue
        
        return results


class ModelPerformanceOptimizer:
    """
    Optimizes model performance with dynamic batch sizing and quantization selection
    """
    
    def __init__(self):
        self.quantization_manager = QuantizedModelManager()
        self.performance_history = []
    
    async def auto_optimize_quantization(self, model_name: str, 
                                       test_prompts: List[str]) -> Dict[str, float]:
        """
        Automatically determine optimal quantization level
        """
        quantization_levels = ['4bit', '8bit', 'fp16']
        results = {}
        
        for ql in quantization_levels:
            try:
                # Test each quantization level
                times = []
                for prompt in test_prompts[:5]:  # Test with 5 prompts
                    _, metrics = await self.quantization_manager.inference_with_metrics(
                        model_name, prompt, quantization_level=ql
                    )
                    times.append(metrics.inference_time)
                
                avg_time = np.mean(times)
                results[ql] = avg_time
                logger.info(f"{ql} quantization: {avg_time:.2f}s average inference time")
                
            except Exception as e:
                logger.error(f"Failed to test {ql} quantization: {e}")
                results[ql] = float('inf')
        
        # Select best quantization level
        best_level = min(results, key=results.get)
        logger.info(f"Selected optimal quantization: {best_level}")
        
        return results
    
    async def dynamic_batch_optimization(self, model_name: str, 
                                       prompts: List[str],
                                       max_batch_size: int = 8) -> List[Tuple[str, ModelPerformanceMetrics]]:
        """
        Find optimal batch size for processing
        """
        best_batch_size = 1
        best_throughput = 0
        
        for batch_size in range(1, min(max_batch_size + 1, len(prompts) + 1)):
            try:
                # Test with this batch size
                test_prompts = prompts[:batch_size * 2]  # Test with 2x batch_size
                results = await self.quantization_manager.optimize_batch_processing(
                    model_name, test_prompts, batch_size
                )
                
                # Calculate throughput
                avg_throughput = np.mean([r[1].throughput for r in results])
                
                if avg_throughput > best_throughput:
                    best_throughput = avg_throughput
                    best_batch_size = batch_size
                
                logger.info(f"Batch size {batch_size}: {avg_throughput:.1f} tokens/s throughput")
                
            except Exception as e:
                logger.error(f"Batch size {batch_size} failed: {e}")
                continue
        
        logger.info(f"Optimal batch size: {best_batch_size}")
        
        # Process with optimal batch size
        return await self.quantization_manager.optimize_batch_processing(
            model_name, prompts, best_batch_size
        )


class MedicalModelRouter:
    """
    Routes medical AI requests to appropriate optimized models
    """
    
    def __init__(self):
        self.optimizer = ModelPerformanceOptimizer()
        self.model_mappings = {
            'diagnosis': 'medical-llama-7b',
            'treatment': 'clinical-bert-base',
            'recommendation': 'medical-gpt-3b',
            'general': 'biomedical-llama-7b'
        }
    
    async def route_request(self, request_type: str, prompt: str) -> Tuple[str, ModelPerformanceMetrics]:
        """Route medical AI request to optimized model"""
        
        model_name = self.model_mappings.get(request_type, 'general')
        
        # Use optimized inference
        result, metrics = await self.optimizer.quantization_manager.inference_with_metrics(
            model_name, prompt, quantization_level='4bit'
        )
        
        return result, metrics


async def main():
    """Example usage of model inference optimization"""
    
    # Initialize performance optimizer
    optimizer = ModelPerformanceOptimizer()
    
    # Test prompts
    test_prompts = [
        "Patient presents with chest pain and shortness of breath.",
        "What are the differential diagnoses for acute abdominal pain?",
        "Recommend treatment plan for Type 2 diabetes management."
    ]
    
    # Test different quantization levels
    results = await optimizer.auto_optimize_quantization(
        'medical-llama-7b', test_prompts
    )
    
    print("Quantization Performance:", results)
    
    # Process batch with optimization
    batch_results = await optimizer.dynamic_batch_optimization(
        'medical-llama-7b', test_prompts
    )
    
    print(f"Processed {len(batch_results)} requests with optimization")


if __name__ == "__main__":
    asyncio.run(main())