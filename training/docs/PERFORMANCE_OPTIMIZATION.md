# Performance Optimization Guide

Comprehensive guide for optimizing the Medical AI Training Pipeline for maximum performance and efficiency.

## Table of Contents

1. [Performance Optimization Overview](#performance-optimization-overview)
2. [Hardware Optimization](#hardware-optimization)
3. [DeepSpeed Optimization](#deepspeed-optimization)
4. [LoRA Optimization](#lora-optimization)
5. [Memory Optimization](#memory-optimization)
6. [Communication Optimization](#communication-optimization)
7. [Data Pipeline Optimization](#data-pipeline-optimization)
8. [Model Optimization](#model-optimization)
9. [Profiling and Benchmarking](#profiling-and-benchmarking)
10. [Scaling Strategies](#scaling-strategies)

## Performance Optimization Overview

### Performance Goals
- **Throughput**: Maximize samples processed per second
- **Memory Efficiency**: Minimize GPU/CPU memory usage
- **Scalability**: Efficient scaling across multiple GPUs/nodes
- **Accuracy**: Maintain medical accuracy while optimizing performance
- **Cost Efficiency**: Optimize resource utilization

### Optimization Principles
1. **Measure First**: Always benchmark before optimizing
2. **Identify Bottlenecks**: Focus on the most significant limiting factors
3. **Iterative Optimization**: Make incremental improvements
4. **Medical Accuracy**: Never compromise on medical accuracy
5. **Resource Constraints**: Work within memory and compute limits

## Hardware Optimization

### GPU Optimization

#### NVIDIA GPU Configuration
```bash
# Set GPU frequency to maximum performance
nvidia-smi -ac 877,1380  # For NVIDIA A100 (adjust for your GPU)

# Enable persistence mode
nvidia-smi -pm 1

# Set GPU power limit to maximum
nvidia-smi -pl 400  # Adjust based on your GPU's TDP

# Configure GPU memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### GPU Memory Management
```python
import torch
import gc

# Clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# Monitor memory usage
def print_memory_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
        print(f"GPU Memory Free: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated():.2f}GB free")
```

#### Multi-GPU Setup
```bash
# Set CUDA_VISIBLE_DEVICES for optimal GPU selection
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configure NCCL settings for optimal communication
export NCCL_IB_DISABLE=1
export NCCL_MIN_NRINGS=8
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

# For InfiniBand networks
export NCCL_IB_HCA=mlx5
export NCCL_IB_SL=5
export NCCL_IB_TC=136
export NCCL_NET_GDR_LEVEL=4
```

### CPU Optimization

#### CPU Configuration
```bash
# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Enable all CPU cores
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export TOKENIZERS_PARALLELISM=true

# Set process affinity for better NUMA locality
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

#### Memory Bandwidth Optimization
```python
import os

# Optimize memory allocation
os.environ['MALLOC_CONF'] = 'background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000'

# Enable memory mapping for large datasets
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Optimize for high bandwidth memory systems
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.enable_fallback_to_cpu = False
```

## DeepSpeed Optimization

### ZeRO Optimization Stages

#### Stage 1: Optimizer State Partitioning
```json
{
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": false,
        "offload_param": false,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "gather_16bit_weights_on_model_save": false
    }
}
```

**Optimization Tips:**
- Use for models < 1B parameters
- Larger bucket sizes (500MB-1GB) reduce communication overhead
- Enable `gather_16bit_weights_on_model_save` for checkpoint efficiency

#### Stage 2: Optimizer + Gradient Partitioning
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": false,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "sub_group_size": 1e9,
        "prefetch_bucket_size": 5e7,
        "param_persistence_threshold": 4e5,
        "gather_16bit_weights_on_model_save": false
    }
}
```

**Optimization Tips:**
- Use for models 1-10B parameters
- CPU pin memory significantly improves offloading performance
- Tune `prefetch_bucket_size` (50MB-500MB) based on memory bandwidth
- Lower `sub_group_size` (100MB-1GB) for better CPU utilization

#### Stage 3: Complete Model Parallelism
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "gather_16bit_weights_on_model_save": true,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "sub_group_size": 1e9,
        "prefetch_bucket_size": 5e7,
        "param_persistence_threshold": 4e5,
        "stage3_prefetch_stream_reserve_memory": 5e7,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "gather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    }
}
```

**Optimization Tips:**
- Use for models > 10B parameters
- Enable parameter offloading for very large models
- Tune `stage3_prefetch_stream_reserve_memory` (50MB-500MB)
- Use higher `param_persistence_threshold` for better parameter reuse

### Mixed Precision Optimization

#### BF16 Configuration
```json
{
    "bfloat16": {
        "enabled": true
    },
    "fp16": {
        "enabled": false
    }
}
```

**Benefits:**
- Better numerical stability than FP16
- Minimal accuracy loss for medical AI tasks
- Faster than FP32 on modern GPUs

#### Optimizer Settings
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "CosineWarmupScheduler",
        "params": {
            "warmup_steps": 1000,
            "total_steps": 10000
        }
    }
}
```

### Communication Optimization

#### NCCL Settings
```bash
# For high-bandwidth networks
export NCCL_IB_DISABLE=1  # Disable InfiniBand for Ethernet
export NCCL_MIN_NRINGS=8  # Increase number of rings
export NCCL_IB_TC=136    # Set traffic class
export NCCL_IB_SL=5      # Set service level
export NCCL_NET_GDR_LEVEL=4  # Set GDR level

# For InfiniBand networks
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_IB_HCA=mlx5   # Use MLX5 HCA
export NCCL_IB_SL=5       # Set service level
export NCCL_IB_TC=136     # Set traffic class
export NCCL_NET_GDR_LEVEL=4  # Set GDR level
```

#### Communication Buckets
```json
{
    "zero_optimization": {
        "allgather_bucket_size": 2e8,     # 200MB
        "reduce_bucket_size": 2e8,        # 200MB
        "reduce_scatter_bucket_size": 2e8, # 200MB
        "prefetch_bucket_size": 5e7       # 50MB
    }
}
```

**Tuning Guidelines:**
- Larger buckets (200MB-1GB) reduce communication overhead
- Smaller `prefetch_bucket_size` (50MB-200MB) improves memory efficiency
- Tune based on network bandwidth and memory constraints

## LoRA Optimization

### Parameter-Efficient Training

#### LoRA Configuration for Different Model Sizes
```yaml
# Small Models (< 1B parameters)
lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

# Medium Models (1-5B parameters) 
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05

# Large Models (> 5B parameters)
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
```

#### Target Module Selection
```yaml
# For Causal Language Models
lora_target_modules: [
    "q_proj", "v_proj", "k_proj", "o_proj",  # Attention projections
    "gate_proj", "up_proj", "down_proj",     # MLP projections
    "embed_tokens",                           # Embedding layer
    "lm_head"                                 # Output layer
]

# For Encoder Models (like BERT)
lora_target_modules: [
    "query", "value", "key", "output.dense", # Attention
    "intermediate.dense", "output.dense"     # MLP
]
```

### LoRA Optimization Settings

#### Quantization Settings
```yaml
# 8-bit quantization for memory efficiency
load_in_8bit: true
llm_int8_threshold: 6.0
llm_int8_skip_modules: ["lm_head"]
llm_int8_enable_fp32_cpu_offload: true

# 4-bit quantization for extreme memory efficiency
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: "bfloat16"
```

#### Training Optimization
```yaml
# Gradient Checkpointing
gradient_checkpointing: true

# Mixed Precision
fp16: false
bf16: true

# Gradient Accumulation
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Learning Rate Scheduling
learning_rate: 0.0002
warmup_steps: 1000
lr_scheduler_type: "cosine"

# Optimizer Settings
weight_decay: 0.001
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1e-8
max_grad_norm: 1.0
```

## Memory Optimization

### Memory Management Strategies

#### GPU Memory Optimization
```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

class MemoryOptimizedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Use gradient checkpointing for memory efficiency
        def custom_forward(module, inputs):
            return module(inputs)
        
        # Checkpoint critical sections
        return checkpoint_sequential(
            custom_forward, 
            chunks=2,  # Split into 2 chunks
            module=self.model,
            inputs=x
        )
```

#### CPU Offloading
```json
{
    "zero_optimization": {
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu", 
            "pin_memory": true
        }
    }
}
```

### Memory Monitoring and Optimization

#### Memory Profiling
```python
from utils.performance_benchmark import MemoryProfiler

class OptimizedTrainingMonitor:
    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.peak_memory = 0
        
    def monitor_training_step(self, step, batch_size, sequence_length):
        # Profile memory usage
        memory_info = self.memory_profiler.profile_memory()
        
        # Calculate memory efficiency
        theoretical_memory = batch_size * sequence_length * 4  # 4 bytes per token
        actual_memory = memory_info['gpu_allocated'] / 1e9  # GB
        
        memory_efficiency = theoretical_memory / actual_memory if actual_memory > 0 else 0
        
        # Update peak memory
        self.peak_memory = max(self.peak_memory, actual_memory)
        
        # Log optimization metrics
        if step % 100 == 0:
            print(f"Step {step}:")
            print(f"  Peak Memory: {self.peak_memory:.2f}GB")
            print(f"  Memory Efficiency: {memory_efficiency:.4f}")
            print(f"  GPU Utilization: {memory_info['gpu_utilization']:.1f}%")
            
        return {
            "peak_memory": self.peak_memory,
            "memory_efficiency": memory_efficiency,
            "gpu_utilization": memory_info['gpu_utilization']
        }
```

#### Memory Optimization Techniques
```python
# 1. Dynamic Loss Scaling for Mixed Precision
class DynamicLossScaler:
    def __init__(self, init_scale=2.**32, scale_factor=2., scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.iter = 0
        self.unskipped = 0
        
    def update_scale(self, found_inf):
        if found_inf:
            self.scale = self.scale / self.scale_factor
            self.unskipped = 0
        else:
            self.unskipped += 1
            if self.unskipped == self.scale_window:
                self.scale = self.scale * self.scale_factor
                self.unskipped = 0

# 2. Memory-Efficient Attention
class MemoryEfficientAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Use memory-efficient attention implementation
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
            kdim=self.head_dim,
            vdim=self.head_dim
        )
        
    def forward(self, query, key, value, mask=None):
        # Use PyTorch's memory-efficient attention
        attn_output, _ = self.attention(
            query, key, value, 
            attn_mask=mask,
            key_padding_mask=None,
            need_weights=False
        )
        return attn_output

# 3. Gradient Accumulation with Memory Cleanup
class MemoryOptimizedTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.accumulation_steps = config.gradient_accumulation_steps
        
    def training_step(self, batch):
        for i in range(self.accumulation_steps):
            # Clear cache before each accumulation step
            if i > 0:
                torch.cuda.empty_cache()
                
            # Forward pass
            outputs = self.model(batch[i])
            loss = outputs.loss / self.accumulation_steps
            
            # Backward pass
            loss.backward()
            
        # Step optimizer once after accumulation
        self.optimizer.step()
        self.optimizer.zero_grad()
```

## Communication Optimization

### Network Configuration

#### For High-Bandwidth Networks (InfiniBand)
```bash
# Optimize for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_SL=5
export NCCL_IB_TC=136
export NCCL_NET_GDR_LEVEL=4

# Increase buffer sizes
export NCCL_IB_BUFSIZE=524288  # 512KB
export NCCL_IB_GIDINDEX=3

# Enable RDMA
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_NET_GDR_LEVEL=4
```

#### For Standard Ethernet
```bash
# Optimize for Ethernet
export NCCL_IB_DISABLE=1
export NCCL_MIN_NRINGS=8
export NCCL_IB_TC=136
export NCCL_DEBUG=INFO

# Enable TCP optimization
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
```

### Communication Efficiency

#### DeepSpeed Communication Settings
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        
        // Communication optimization
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "sub_group_size": 1e9,
        "prefetch_bucket_size": 5e7,
        
        // Advanced communication settings
        "stage3_prefetch_stream_reserve_memory": 5e7,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    
    "communication": {
        "backend": "nccl",
        "timeout": 1800,
        "max_train_batch_size": 32,
        "scatter_gather_tensors_to_gpu": true
    }
}
```

#### Communication Profiling
```python
from utils.performance_benchmark import CommunicationProfiler

class CommunicationOptimizer:
    def __init__(self, world_size):
        self.world_size = world_size
        self.profiler = CommunicationProfiler()
        
    def benchmark_communication(self, test_sizes=[1e6, 1e7, 1e8]):
        """Benchmark communication performance with different tensor sizes"""
        
        results = {}
        
        for size in test_sizes:
            print(f"Benchmarking communication for tensor size: {size}")
            
            # Test all-reduce
            all_reduce_time = self.profiler.benchmark_all_reduce(
                tensor_size=size,
                world_size=self.world_size
            )
            
            # Test reduce-scatter
            reduce_scatter_time = self.profiler.benchmark_reduce_scatter(
                tensor_size=size,
                world_size=self.world_size
            )
            
            # Test all-gather
            all_gather_time = self.profiler.benchmark_all_gather(
                tensor_size=size,
                world_size=self.world_size
            )
            
            results[size] = {
                "all_reduce": all_reduce_time,
                "reduce_scatter": reduce_scatter_time,
                "all_gather": all_gather_time
            }
            
        return results
        
    def optimize_bucket_sizes(self, benchmark_results):
        """Optimize bucket sizes based on benchmark results"""
        
        # Find optimal bucket size for each operation
        optimal_sizes = {}
        
        for operation in ["all_reduce", "reduce_scatter", "all_gather"]:
            # Find size with best throughput
            best_size = max(
                benchmark_results.keys(),
                key=lambda x: x / benchmark_results[x][operation]
            )
            
            optimal_sizes[operation] = best_size
            print(f"Optimal {operation} bucket size: {best_size}")
            
        return optimal_sizes
```

## Data Pipeline Optimization

### Data Loading Optimization

#### Efficient Data Loading
```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class OptimizedDataLoader:
    def __init__(self, dataset, tokenizer, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.config = config
        
        # Optimized collator
        self.collator = OptimizedDataCollator(tokenizer)
        
    def create_loader(self, batch_size, num_workers=None):
        num_workers = num_workers or min(8, os.cpu_count())
        
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else 2,
            collate_fn=self.collator
        )
        
        return loader

class OptimizedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, examples):
        # Efficient batching
        batch = self.tokenizer.pad(
            examples,
            padding=True,
            max_length=None,
            pad_to_multiple_of=8,  # Optimize for GPU memory
            return_tensors="pt"
        )
        
        return batch
```

#### Data Preprocessing Optimization
```python
def optimize_data_preprocessing(dataset, tokenizer, num_processes=None):
    """Optimize data preprocessing for maximum throughput"""
    
    num_processes = num_processes or min(8, os.cpu_count())
    
    # Use vectorized operations
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=512
        ),
        batched=True,
        num_proc=num_processes,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )
    
    # Filter out invalid examples
    filtered_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        desc="Filtering empty sequences"
    )
    
    return filtered_dataset
```

### Memory-Efficient Data Handling

#### Streaming Data Loading
```python
import json
from torch.utils.data import IterableDataset

class StreamingMedicalDataset(IterableDataset):
    def __init__(self, file_paths, tokenizer, max_length=512):
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __iter__(self):
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                for line in f:
                    # Process one line at a time
                    example = json.loads(line)
                    
                    # Tokenize on the fly
                    tokens = self.tokenizer(
                        example["text"],
                        truncation=True,
                        max_length=self.max_length,
                        padding=False,
                        return_tensors="pt"
                    )
                    
                    yield {
                        "input_ids": tokens["input_ids"].squeeze(),
                        "attention_mask": tokens["attention_mask"].squeeze(),
                        "labels": example.get("label", -1)
                    }
```

## Model Optimization

### Architecture Optimization

#### Model Size Selection
```python
MODELS = {
    "small": {
        "name": "microsoft/DialoGPT-small",
        "parameters": "117M",
        "memory_gb": 2,
        "throughput_samples_per_sec": 100,
        "medical_accuracy": 0.85
    },
    "medium": {
        "name": "microsoft/DialoGPT-medium",
        "parameters": "345M", 
        "memory_gb": 6,
        "throughput_samples_per_sec": 50,
        "medical_accuracy": 0.88
    },
    "large": {
        "name": "microsoft/DialoGPT-large",
        "parameters": "762M",
        "memory_gb": 14,
        "throughput_samples_per_sec": 25,
        "medical_accuracy": 0.91
    }
}

def select_optimal_model(requirements):
    """Select optimal model based on requirements"""
    
    constraints = requirements.get("constraints", {})
    targets = requirements.get("targets", {})
    
    # Filter models by constraints
    available_models = {}
    for name, config in MODELS.items():
        if "memory_gb" in constraints and config["memory_gb"] > constraints["memory_gb"]:
            continue
        available_models[name] = config
    
    # Select based on targets
    if "throughput" in targets:
        # Prioritize throughput
        best_model = max(
            available_models.keys(),
            key=lambda x: available_models[x]["throughput_samples_per_sec"]
        )
    elif "accuracy" in targets:
        # Prioritize accuracy
        best_model = max(
            available_models.keys(),
            key=lambda x: available_models[x]["medical_accuracy"]
        )
    else:
        # Balance throughput and accuracy
        best_model = min(
            available_models.keys(),
            key=lambda x: (
                -available_models[x]["throughput_samples_per_sec"] +
                available_models[x]["medical_accuracy"] * 100
            )
        )
    
    return MODELS[best_model]
```

#### Model Compilation
```python
import torch

def optimize_model_for_inference(model):
    """Optimize model for inference using TorchScript"""
    
    # Enable compilation
    if hasattr(torch, 'compile'):
        # Use PyTorch 2.0 compilation
        model = torch.compile(
            model,
            mode="default",  # "default", "reduce-overhead", "max-autotune"
            fullgraph=False
        )
    
    # Enable memory-efficient attention
    model = model.to(memory_format=torch.channels_last)
    
    return model
```

### Quantization Optimization

#### 8-bit Quantization
```python
from transformers import BitsAndBytesConfig
import torch

def setup_8bit_quantization():
    """Setup 8-bit quantization for memory efficiency"""
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=["lm_head"],
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    return quantization_config

# Usage in model loading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=setup_8bit_quantization(),
    device_map="auto"
)
```

#### 4-bit Quantization
```python
def setup_4bit_quantization():
    """Setup 4-bit quantization for extreme memory efficiency"""
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # or "fp4"
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    return quantization_config
```

## Profiling and Benchmarking

### Performance Profiling

#### Comprehensive Benchmark Suite
```python
from utils.performance_benchmark import PerformanceBenchmark
import time
import psutil

class ComprehensiveProfiler:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.benchmark = PerformanceBenchmark()
        
    def profile_training_performance(self, dataset):
        """Profile complete training pipeline"""
        
        results = {}
        
        # 1. Data Loading Performance
        print("Profiling data loading...")
        data_load_metrics = self.benchmark.benchmark_data_loading(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=4
        )
        results["data_loading"] = data_load_metrics
        
        # 2. Forward Pass Performance
        print("Profiling forward pass...")
        forward_metrics = self.benchmark.benchmark_forward_pass(
            model=self.model,
            batch_size=self.config.batch_size,
            sequence_length=512
        )
        results["forward_pass"] = forward_metrics
        
        # 3. Backward Pass Performance
        print("Profiling backward pass...")
        backward_metrics = self.benchmark.benchmark_backward_pass(
            model=self.model,
            batch_size=self.config.batch_size,
            sequence_length=512
        )
        results["backward_pass"] = backward_metrics
        
        # 4. Communication Performance
        print("Profiling communication...")
        comm_metrics = self.benchmark.benchmark_communication(
            world_size=torch.cuda.device_count()
        )
        results["communication"] = comm_metrics
        
        # 5. Memory Performance
        print("Profiling memory usage...")
        memory_metrics = self.benchmark.benchmark_memory_usage()
        results["memory"] = memory_metrics
        
        # 6. End-to-end Training Step
        print("Profiling end-to-end step...")
        step_metrics = self.benchmark.benchmark_training_step(
            model=self.model,
            dataset=dataset,
            steps=10
        )
        results["end_to_end"] = step_metrics
        
        return results
        
    def generate_optimization_recommendations(self, benchmark_results):
        """Generate optimization recommendations based on benchmarks"""
        
        recommendations = []
        
        # Check data loading bottleneck
        data_throughput = benchmark_results["data_loading"]["throughput_samples_per_sec"]
        model_throughput = benchmark_results["end_to_end"]["throughput_samples_per_sec"]
        
        if data_throughput < model_throughput * 0.5:
            recommendations.append({
                "issue": "Data loading bottleneck",
                "recommendation": "Increase num_workers or optimize data preprocessing",
                "priority": "high"
            })
        
        # Check memory efficiency
        peak_memory = benchmark_results["memory"]["peak_gpu_memory_gb"]
        total_memory = benchmark_results["memory"]["total_gpu_memory_gb"]
        memory_usage = peak_memory / total_memory
        
        if memory_usage > 0.9:
            recommendations.append({
                "issue": "High memory usage",
                "recommendation": "Use gradient checkpointing or ZeRO optimization",
                "priority": "high"
            })
        
        # Check communication efficiency
        comm_overhead = benchmark_results["communication"]["overhead_percentage"]
        if comm_overhead > 20:
            recommendations.append({
                "issue": "High communication overhead",
                "recommendation": "Optimize NCCL settings or use larger batch sizes",
                "priority": "medium"
            })
        
        return recommendations
```

### Custom Performance Metrics

#### Training Efficiency Metrics
```python
class TrainingEfficiencyMetrics:
    def __init__(self):
        self.metrics = {}
        
    def calculate_throughput(self, samples_processed, time_elapsed):
        """Calculate training throughput"""
        return samples_processed / time_elapsed
        
    def calculate_memory_efficiency(self, theoretical_memory, actual_memory):
        """Calculate memory efficiency ratio"""
        return theoretical_memory / actual_memory if actual_memory > 0 else 0
        
    def calculate_communication_efficiency(self, computation_time, communication_time):
        """Calculate communication overhead"""
        total_time = computation_time + communication_time
        return communication_time / total_time if total_time > 0 else 0
        
    def calculate_cost_efficiency(self, cloud_cost_per_hour, throughput):
        """Calculate cost per sample"""
        return cloud_cost_per_hour / (throughput * 3600) if throughput > 0 else float('inf')
        
    def calculate_convergence_speed(self, initial_loss, final_loss, steps):
        """Calculate convergence rate"""
        return (initial_loss - final_loss) / steps if steps > 0 else 0
        
    def calculate_medical_accuracy_efficiency(self, accuracy, throughput):
        """Calculate accuracy per unit throughput"""
        return accuracy / throughput if throughput > 0 else 0
```

## Scaling Strategies

### Horizontal Scaling

#### Multi-Node Scaling
```bash
# Node discovery and configuration
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29500
export WORLD_SIZE=8

# Node-specific configuration
export NODE_RANK=0
export LOCAL_RANK=0

# Launch distributed training
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_distributed.py \
    --config configs/multi_node_config.json \
    --model_name $MODEL_NAME \
    --dataset_path $DATASET_PATH
```

#### Elastic Training
```python
def setup_elastic_training():
    """Setup elastic training for dynamic scaling"""
    
    # Use DeepSpeed elastic training
    elastic_config = {
        "elastic_training": {
            "enabled": True,
            "min_elastic_nodes": 1,
            "max_elastic_nodes": 8,
            "elastic_checkpoint_interval": 100
        },
        "zero_optimization": {
            "stage": 3,
            "elastic_checkpointing": True
        }
    }
    
    return elastic_config
```

### Vertical Scaling

#### GPU Scaling Strategy
```python
def optimal_gpu_configuration(model_size_gb, available_memory_gb):
    """Determine optimal GPU configuration"""
    
    # Memory requirements for different components
    model_memory = model_size_gb
    optimizer_memory = model_memory * 2  # Adam optimizer needs 2x model memory
    gradient_memory = model_memory  # Gradients need same memory as model
    activation_memory = model_memory * 4  # Activations need 4x model memory
    
    total_required = model_memory + optimizer_memory + gradient_memory + activation_memory
    
    if total_required <= available_memory_gb * 0.8:  # 80% utilization target
        return {
            "gpu_count": 1,
            "batch_size_per_gpu": calculate_optimal_batch_size(available_memory_gb),
            "use_optimizer_state_sharding": False,
            "use_gradient_checkpointing": False
        }
    else:
        # Scale across multiple GPUs
        scale_factor = total_required / (available_memory_gb * 0.8)
        return {
            "gpu_count": int(scale_factor),
            "batch_size_per_gpu": calculate_optimal_batch_size(available_memory_gb // int(scale_factor)),
            "use_optimizer_state_sharding": True,
            "use_gradient_checkpointing": True
        }
```

### Auto-scaling Implementation

#### Dynamic Resource Allocation
```python
class AutoScalingTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scaling_manager = ResourceScalingManager()
        
    def adaptive_training_loop(self):
        """Training loop with automatic resource scaling"""
        
        while not self.training_complete():
            # Monitor current performance
            metrics = self.get_performance_metrics()
            
            # Check if scaling is needed
            scaling_decision = self.scaling_manager.should_scale(metrics)
            
            if scaling_decision.action == "scale_up":
                # Add more resources
                self.scaling_manager.scale_up(scaling_decision.target_resources)
            elif scaling_decision.action == "scale_down":
                # Reduce resources
                self.scaling_manager.scale_down(scaling_decision.target_resources)
            
            # Continue training with current resources
            self.training_step()
```

This completes the comprehensive Performance Optimization Guide. The guide provides detailed strategies for optimizing all aspects of the Medical AI Training Pipeline, from hardware configuration to advanced scaling strategies, ensuring maximum performance while maintaining medical accuracy and compliance requirements.
