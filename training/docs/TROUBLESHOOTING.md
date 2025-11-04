# Troubleshooting Guide

Comprehensive troubleshooting guide for common issues in the Medical AI Training Pipeline.

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Common Installation Issues](#common-installation-issues)
3. [Hardware and System Issues](#hardware-and-system-issues)
4. [DeepSpeed Configuration Issues](#deepspeed-configuration-issues)
5. [Training Issues](#training-issues)
6. [Memory Issues (OOM)](#memory-issues-oom)
7. [Communication and Distributed Training Issues](#communication-and-distributed-training-issues)
8. [Data and PHI Protection Issues](#data-and-phi-protection-issues)
9. [Model Loading and LoRA Issues](#model-loading-and-lora-issues)
10. [Performance Issues](#performance-issues)
11. [Model Serving Issues](#model-serving-issues)

## Quick Diagnostic Tools

### Environment Check Script

```python
#!/usr/bin/env python3
# diagnostics.py - Quick environment and setup check

import sys
import os
import torch
import platform

def check_system_environment():
    """Check system environment and dependencies"""
    
    print("=== System Environment Check ===")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # CUDA Check
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    
    # PyTorch Check
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"PyTorch CUDA Support: {torch.cuda.is_available()}")
    
    # DeepSpeed Check
    try:
        import deepspeed
        print(f"DeepSpeed Version: {deepspeed.__version__}")
    except ImportError:
        print("❌ DeepSpeed not installed")
    
    # Transformers Check
    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
    
    # Dataset Check
    try:
        import datasets
        print(f"Datasets Version: {datasets.__version__}")
    except ImportError:
        print("❌ Datasets not installed")

def check_gpu_memory():
    """Check GPU memory status"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    print("\n=== GPU Memory Status ===")
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"GPU {i}:")
        print(f"  Total: {total_memory:.2f}GB")
        print(f"  Allocated: {allocated:.2f}GB")
        print(f"  Reserved: {reserved:.2f}GB")
        print(f"  Free: {total_memory - reserved:.2f}GB")

def test_deepspeed():
    """Test DeepSpeed installation and basic functionality"""
    
    print("\n=== DeepSpeed Test ===")
    try:
        import deepspeed
        from deepspeed.ops.op_builder import OpBuilder
        
        # Test if ops can be compiled
        try:
            builder = OpBuilder()
            print("✓ DeepSpeed ops builder available")
        except Exception as e:
            print(f"❌ DeepSpeed ops builder failed: {e}")
        
        # Test distributed initialization
        try:
            import torch.distributed as dist
            print(f"✓ PyTorch distributed available")
            
            # Check if we can initialize
            if not dist.is_initialized():
                print("  Distributed not initialized (normal for single node)")
            else:
                print(f"  Distributed initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
        except Exception as e:
            print(f"❌ Distributed initialization failed: {e}")
            
    except ImportError as e:
        print(f"❌ DeepSpeed import failed: {e}")

def check_environment_variables():
    """Check important environment variables"""
    
    print("\n=== Environment Variables ===")
    
    important_vars = [
        'CUDA_VISIBLE_DEVICES',
        'NCCL_DEBUG',
        'MASTER_ADDR',
        'MASTER_PORT',
        'WORLD_SIZE',
        'RANK',
        'LOCAL_RANK',
        'PYTORCH_CUDA_ALLOC_CONF'
    ]
    
    for var in important_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
        else:
            print(f"{var}: (not set)")

def run_comprehensive_check():
    """Run all diagnostic checks"""
    
    print("Medical AI Training Pipeline - Diagnostic Check")
    print("=" * 50)
    
    try:
        check_system_environment()
        check_gpu_memory()
        check_environment_variables()
        test_deepspeed()
        
        print("\n=== Summary ===")
        print("✓ Environment check completed")
        print("Please review any ❌ marked items above")
        
    except Exception as e:
        print(f"\n❌ Diagnostic check failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    run_comprehensive_check()
```

### Performance Benchmark Script

```python
#!/usr/bin/env python3
# performance_check.py - Quick performance benchmark

import time
import torch
import torch.distributed as dist
from transformers import AutoModel, AutoTokenizer

def benchmark_model_loading():
    """Benchmark model loading performance"""
    
    print("=== Model Loading Benchmark ===")
    
    model_names = [
        "distilbert-base-uncased",
        "bert-base-uncased", 
        "bert-large-uncased"
    ]
    
    for model_name in model_names:
        try:
            start_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            load_time = time.time() - start_time
            
            # Estimate memory usage
            memory_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
            
            print(f"{model_name}:")
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Memory: {memory_mb:.1f}MB")
            
        except Exception as e:
            print(f"{model_name}: FAILED - {e}")

def benchmark_inference():
    """Benchmark inference performance"""
    
    print("\n=== Inference Benchmark ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModel.from_pretrained("bert-base-uncased")
        model.eval()
        
        # Test different batch sizes and sequence lengths
        test_cases = [
            (1, 128),
            (4, 128),
            (1, 512),
            (4, 512)
        ]
        
        for batch_size, seq_len in test_cases:
            # Prepare input
            text = "This is a sample medical text for testing." * (seq_len // 30)
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=seq_len)
            
            # Expand batch
            inputs = {k: v.repeat(batch_size, 1) for k, v in inputs.items()}
            
            # Benchmark
            with torch.no_grad():
                start_time = time.time()
                outputs = model(**inputs)
                inference_time = time.time() - start_time
            
            throughput = batch_size / inference_time
            
            print(f"Batch {batch_size}, Seq {seq_len}: {inference_time:.3f}s ({throughput:.1f} samples/sec)")
    
    except Exception as e:
        print(f"Inference benchmark failed: {e}")

def benchmark_distributed():
    """Benchmark distributed setup"""
    
    print("\n=== Distributed Setup Benchmark ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping distributed benchmark")
        return
    
    try:
        # Test if distributed is properly configured
        if dist.is_available():
            print(f"✓ PyTorch distributed available")
            
            if not dist.is_initialized():
                print("Distributed not initialized (normal for single node)")
            else:
                print(f"✓ Distributed initialized: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
        else:
            print("❌ PyTorch distributed not available")
            
    except Exception as e:
        print(f"Distributed benchmark failed: {e}")

if __name__ == "__main__":
    benchmark_model_loading()
    benchmark_inference() 
    benchmark_distributed()
```

## Common Installation Issues

### DeepSpeed Installation Issues

#### Issue: DeepSpeed compilation failures
```bash
# Error: PyTorch and CUDA version mismatch
ERROR: torch distributed not available

# Solution: Install compatible versions
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed
```

#### Issue: DeepSpeed ops compilation fails
```bash
# Error: Could not build wheels for deepspeed
# Solution: Install system dependencies
sudo apt-get update
sudo apt-get install build-essential
sudo apt-get install ninja-build

# Then reinstall DeepSpeed
pip uninstall deepspeed
pip install deepspeed --no-cache-dir

# Or use pre-compiled wheels
pip install deepspeed==0.12.4 --find-links https://download.pytorch.org/whl/torch_stable.html
```

#### Issue: CUDA out of memory during compilation
```bash
# Error: CUDA out of memory during setup
# Solution: Use minimal compilation
export DS_BUILD_AIO=0
export DS_BUILD_CPU_ADAM=0
export DS_BUILD_TRANSFORMER=0
export DS_BUILD_STOCHASTIC_TRANSFORMER=0
export DS_BUILD_MEMORY_EFFICIENT_ATTENTION=0

pip install deepspeed --no-cache-dir
```

### CUDA Version Compatibility

#### Issue: CUDA version mismatch
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Install matching PyTorch version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Transformers and Datasets Issues

#### Issue: Import errors or version conflicts
```bash
# Check installed versions
pip list | grep -E "(transformers|datasets|torch)"

# Update to latest versions
pip install --upgrade transformers datasets accelerate

# If conflicts persist, create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate
pip install -r requirements.txt
```

## Hardware and System Issues

### GPU Driver Issues

#### Issue: GPU not detected
```bash
# Check GPU status
nvidia-smi

# Check NVIDIA driver
nvidia-settings --version

# Reinstall NVIDIA drivers if needed
# Ubuntu/Debian:
sudo apt-get install --reinstall nvidia-driver-525

# Check GPU visibility for PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Issue: Multiple GPUs not visible
```bash
# Check GPU visibility
nvidia-smi --list-gpus

# Set CUDA devices explicitly
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Test GPU availability
python -c "
import torch
print(f'Devices: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_properties(i).name}')
"
```

### Memory Issues

#### Issue: Insufficient system memory
```bash
# Check system memory
free -h
cat /proc/meminfo | grep MemAvailable

# Monitor memory during training
watch -n 1 'free -h'

# Increase swap if needed
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Issue: NUMA topology issues
```bash
# Check NUMA topology
numactl --hardware

# Bind processes to specific NUMA nodes
numactl --cpubind=0 --membind=0 python training_script.py

# Set NUMA memory policy
export NUMA_BALANCING=1
export NUMA_MEMORY_POLICY=interleave
```

## DeepSpeed Configuration Issues

### ZeRO Optimization Issues

#### Issue: Stage 3 parameter offloading errors
```json
{
    "error": "Parameter offloading failed - insufficient memory"
}

# Solution: Adjust offloading settings
{
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "param_persistence_threshold": 1e6,  // Increase threshold
        "stage3_max_live_parameters": 2e9     // Reduce live parameters
    }
}
```

#### Issue: Communication timeouts with large bucket sizes
```bash
# Error: NCCL communication timeout
# Solution: Reduce bucket sizes
{
    "zero_optimization": {
        "allgather_bucket_size": 1e8,     // Reduce from 5e8
        "reduce_bucket_size": 1e8          // Reduce from 5e8
    },
    "communication": {
        "timeout": 3600                    // Increase timeout
    }
}
```

#### Issue: CPU memory insufficient for optimizer offloading
```bash
# Error: CPU out of memory during optimizer offloading
# Solution: Disable optimizer offloading or reduce model size
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": false,        // Disable offloading
        "offload_param": false
    }
}
```

### Mixed Precision Issues

#### Issue: BF16 not supported on GPU
```python
# Error: "bfloat16 is not supported"
# Solution: Check GPU support or use FP16
{
    "bfloat16": {
        "enabled": false                   // Disable BF16
    },
    "fp16": {
        "enabled": true,                   // Use FP16 instead
        "initial_scale_power": 16
    }
}
```

```bash
# Check GPU BF16 support
python -c "
import torch
print('BF16 support:', torch.cuda.is_bf16_supported())
"
```

#### Issue: Loss scaling with mixed precision
```python
# Error: "Gradient overflow detected"
# Solution: Adjust loss scaling
{
    "fp16": {
        "enabled": true,
        "initial_scale_power": 32,         // Increase initial scale
        "loss_scale_window": 2000,         // Increase window
        "min_loss_scale": 1.0,
        "hysteresis": 2
    }
}
```

## Training Issues

### Convergence Issues

#### Issue: Training loss not decreasing
```python
# Diagnose convergence problems
def diagnose_convergence(model, dataloader, optimizer):
    """Diagnose why training loss isn't converging"""
    
    print("=== Convergence Diagnosis ===")
    
    # 1. Check learning rate
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
    
    # 2. Check gradient norms
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2) ** 2
    total_norm = total_norm ** (1. / 2)
    
    print(f"Gradient norm: {total_norm}")
    
    if total_norm < 1e-6:
        print("❌ Gradient norm too small - learning rate may be too low")
    elif total_norm > 10:
        print("❌ Gradient norm too large - learning rate may be too high")
    
    # 3. Check data quality
    print("\nData quality check:")
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Check first 5 batches
            break
        
        labels = batch.get('labels', batch.get('label', None))
        if labels is not None:
            print(f"Batch {i} - Label distribution: {torch.bincount(labels).tolist()}")
    
    # 4. Check model parameters
    print("\nModel parameter check:")
    for name, param in model.named_parameters():
        if 'bias' in name:
            continue
        
        mean_val = param.data.mean().item()
        std_val = param.data.std().item()
        
        if abs(mean_val) > 1:
            print(f"❌ {name}: high mean={mean_val:.4f}")
        if std_val > 2:
            print(f"❌ {name}: high std={std_val:.4f}")
```

#### Issue: Validation loss much higher than training loss
```python
# Overfitting diagnosis
def diagnose_overfitting(train_loss, val_loss):
    """Diagnose overfitting issues"""
    
    gap = val_loss - train_loss
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Gap: {gap:.4f}")
    
    if gap > 0.5:
        print("❌ Significant overfitting detected")
        print("Solutions:")
        print("1. Increase regularization (weight decay)")
        print("2. Add dropout")
        print("3. Reduce model size")
        print("4. Increase training data")
        print("5. Use early stopping")
        return "overfitting"
    elif gap < -0.1:
        print("❌ Validation loss lower than training loss")
        print("Possible issues:")
        print("1. Data leakage between train/val sets")
        print("2. Insufficient training data")
        print("3. Learning rate too high")
        return "underfitting"
    else:
        print("✓ Normal overfitting level")
        return "normal"
```

### Data Issues

#### Issue: Data loading bottleneck
```python
# Diagnose data loading performance
def diagnose_data_loading(dataloader):
    """Diagnose data loading performance issues"""
    
    import time
    
    print("=== Data Loading Performance ===")
    
    # Time several batches
    times = []
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Time first 10 batches
            break
        
        start_time = time.time()
        # Simulate processing (remove actual processing)
        batch_size = batch['input_ids'].shape[0]
        process_time = batch_size * 0.001  # Simulated processing
        
        total_time = time.time() - start_time
        times.append((total_time, process_time))
    
    avg_load_time = sum(t[0] for t in times) / len(times)
    avg_process_time = sum(t[1] for t in times) / len(times)
    
    print(f"Average batch load time: {avg_load_time:.3f}s")
    print(f"Average processing time: {avg_process_time:.3f}s")
    print(f"I/O bottleneck: {avg_load_time > avg_process_time}")
    
    if avg_load_time > avg_process_time:
        print("❌ I/O bottleneck detected")
        print("Solutions:")
        print("1. Increase num_workers")
        print("2. Use faster storage (NVMe SSD)")
        print("3. Preprocess data to cache")
        print("4. Increase prefetch_factor")
```

## Memory Issues (OOM)

### GPU Memory Issues

#### Issue: CUDA out of memory (OOM)
```python
# Comprehensive OOM debugging
def diagnose_gpu_oom():
    """Diagnose GPU out of memory issues"""
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("=== GPU Memory Analysis ===")
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"GPU {i} ({props.name}):")
        print(f"  Total memory: {total_memory:.2f}GB")
        print(f"  Allocated: {allocated:.2f}GB ({allocated/total_memory*100:.1f}%)")
        print(f"  Reserved: {reserved:.2f}GB ({reserved/total_memory*100:.1f}%)")
        print(f"  Free: {total_memory - reserved:.2f}GB")
        
        # Analyze memory usage
        if reserved > total_memory * 0.9:
            print(f"  ❌ High memory usage: {reserved/total_memory*100:.1f}%")
        else:
            print(f"  ✓ Normal memory usage: {reserved/total_memory*100:.1f}%")
```

#### Solutions for GPU OOM
```python
# 1. Gradient Checkpointing
model.gradient_checkpointing_enable()

# 2. Reduce batch size
train_batch_size = 4  # Reduce from 8

# 3. Enable ZeRO optimization
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}

# 4. Use mixed precision
{
    "bfloat16": {"enabled": true}
}

# 5. CPU offloading
{
    "zero_optimization": {
        "offload_param": {"device": "cpu"},
        "offload_optimizer": {"device": "cpu"}
    }
}
```

#### Memory optimization script
```python
class MemoryOptimizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def optimize_for_memory(self):
        """Apply memory optimizations"""
        
        optimizations = []
        
        # 1. Enable gradient checkpointing
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            optimizations.append("Gradient checkpointing")
        
        # 2. Clear cache
        torch.cuda.empty_cache()
        optimizations.append("Cache cleared")
        
        # 3. Check for unused parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if trainable_params < total_params:
            optimizations.append(f"Non-trainable params: {total_params - trainable_params}")
        
        return optimizations
```

### CPU Memory Issues

#### Issue: System running out of RAM
```bash
# Monitor memory usage
watch -n 1 'free -h && ps aux --sort=-%mem | head -10'

# Check for memory leaks
import tracemalloc
tracemalloc.start()

# ... your code here ...

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024**2:.1f} MB")
print(f"Peak memory usage: {peak / 1024**2:.1f} MB")
tracemalloc.stop()
```

#### Solutions for CPU Memory Issues
```python
# 1. Use generators for large datasets
def large_dataset_generator(file_path):
    with open(file_path) as f:
        for line in f:
            yield json.loads(line)

# 2. Clear intermediate results
def memory_efficient_processing(data):
    results = []
    for item in data:
        # Process item
        result = process_item(item)
        results.append(result)
        
        # Clear intermediate results periodically
        if len(results) % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return results

# 3. Use memory mapping for large arrays
import numpy as np

# Create memory-mapped array instead of loading into RAM
mmap_array = np.memmap('large_data.dat', dtype='float32', mode='r', shape=(1000000, 1024))
```

## Communication and Distributed Training Issues

### Process Group Issues

#### Issue: Process group initialization fails
```python
def debug_process_group():
    """Debug process group initialization"""
    
    import torch.distributed as dist
    
    print("=== Process Group Debug ===")
    
    # Check environment variables
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"RANK: {os.environ.get('RANK')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    
    # Check if already initialized
    if dist.is_initialized():
        print(f"✓ Process group initialized")
        print(f"Rank: {dist.get_rank()}")
        print(f"World size: {dist.get_world_size()}")
        print(f"Backend: {dist.get_backend()}")
    else:
        print("❌ Process group not initialized")
        
    # Try to initialize
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=datetime.timedelta(seconds=300)
            )
        print("✓ Process group initialized successfully")
    except Exception as e:
        print(f"❌ Process group initialization failed: {e}")
```

#### Issue: Process group timeout
```bash
# Error: "Timed out waiting for all processes"
# Solution: Increase timeout and debug connectivity

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Check network connectivity
ping -c 4 $MASTER_ADDR
telnet $MASTER_ADDR $MASTER_PORT

# Increase timeout in DeepSpeed config
{
    "communication": {
        "timeout": 3600  // Increase from 1800
    }
}
```

### Network Connectivity Issues

#### Issue: NCCL communication timeout
```bash
# Debug NCCL issues
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Test communication between nodes
python -c "
import torch
import torch.distributed as dist
import os

os.environ['MASTER_ADDR'] = '192.168.1.100'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '2'

try:
    dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=300))
    print('✓ NCCL initialization successful')
    
    # Test basic communication
    tensor = torch.ones(1, device=f'cuda:{dist.get_rank()}')
    dist.all_reduce(tensor)
    print('✓ All-reduce successful')
    
except Exception as e:
    print(f'❌ NCCL test failed: {e}')
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
"
```

#### Issue: Inter-node communication fails
```bash
# Check firewall settings
sudo ufw status
sudo iptables -L

# Check if master port is accessible
telnet master_node_ip 29500

# Check SSH connectivity
ssh -v user@worker_node_ip

# Test with simple Python script
# test_comm.py on all nodes:
import torch
import torch.distributed as dist
import socket

print(f"Node hostname: {socket.gethostname()}")
print(f"Local IP: {socket.gethostbyname(socket.gethostname())}")
```

## Data and PHI Protection Issues

### PHI Detection Issues

#### Issue: False positives in PHI detection
```python
from utils.phi_validator import PHIValidator

# Test PHI detection on sample data
def debug_phi_detection():
    """Debug PHI detection issues"""
    
    validator = PHIValidator()
    
    # Test cases
    test_cases = [
        "Patient John Doe presented with symptoms",
        "The doctor recommended medication",
        "Hospital protocol was followed",
        "Mr. Smith was discharged yesterday"
    ]
    
    for text in test_cases:
        result = validator.detect_phi(text)
        print(f"Text: {text}")
        print(f"PHI detected: {result}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print("---")
```

#### Solutions for PHI Issues
```python
# 1. Adjust PHI detection sensitivity
phi_config = {
    "phi_protection": {
        "enabled": True,
        "strict_mode": False,  # Reduce false positives
        "validation": {
            "validation_level": "moderate",  # Use moderate instead of strict
            "confidence_threshold": 0.7     # Lower confidence threshold
        }
    }
}

# 2. Custom PHI patterns
custom_patterns = [
    {
        "name": "medical_id",
        "patterns": ["MED-[0-9]{6}"],
        "method": "mask",
        "mask_format": "MED-######"
    }
]

# 3. Whitelist certain patterns
whitelist_patterns = [
    r"Dr\.\s+[A-Za-z]+",  # Doctor names
    r"Hospital\s+[A-Za-z]+"  # Hospital names
]
```

### Data Format Issues

#### Issue: Invalid JSON format
```python
def validate_json_data(file_path):
    """Validate JSON data format"""
    
    import json
    
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Validate required fields
                    if 'text' not in data:
                        print(f"Line {line_num}: Missing 'text' field")
                    if 'label' not in data:
                        print(f"Line {line_num}: Missing 'label' field")
                except json.JSONDecodeError as e:
                    print(f"Line {line_num}: Invalid JSON - {e}")
                    return False
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False
    
    return True
```

#### Issue: Encoding problems
```python
def fix_encoding_issues(input_file, output_file):
    """Fix encoding issues in data files"""
    
    import codecs
    
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with codecs.open(input_file, 'r', encoding=encoding) as f_in:
                with codecs.open(output_file, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        f_out.write(line)
            print(f"✓ Successfully converted using {encoding}")
            return True
        except UnicodeDecodeError:
            continue
    
    print("❌ Could not decode file with any encoding")
    return False
```

## Model Loading and LoRA Issues

### Model Loading Issues

#### Issue: Model not loading due to memory
```python
def memory_efficient_model_loading(model_name):
    """Load model with memory optimizations"""
    
    from transformers import AutoModel, AutoTokenizer
    import torch
    
    # Enable memory efficient loading
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    
    try:
        # Load with device map
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
        
    except torch.cuda.OutOfMemoryError:
        print("❌ GPU OOM - trying CPU offloading")
        
        # Try with CPU offloading
        model = AutoModel.from_pretrained(
            model_name,
            device_map="cpu",
            offload_folder="./offload",
            torch_dtype=torch.float32
        )
        
        return model, tokenizer
```

#### Issue: LoRA adapter not loading
```python
def debug_lora_loading(base_model_path, adapter_path):
    """Debug LoRA adapter loading issues"""
    
    from peft import PeftModel, PeftConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    
    try:
        # Load LoRA config
        config = PeftConfig.from_pretrained(adapter_path)
        print(f"✓ LoRA config loaded from {adapter_path}")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16
        )
        print(f"✓ Base model loaded: {config.base_model_name_or_path}")
        
        # Load LoRA model
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"✓ LoRA adapter loaded")
        
        return model, config
        
    except Exception as e:
        print(f"❌ LoRA loading failed: {e}")
        
        # Common fixes
        print("Possible solutions:")
        print("1. Check if base model path is correct")
        print("2. Check if adapter path exists")
        print("3. Ensure compatible PEFT version")
        print("4. Clear cache and retry")
```

### LoRA Training Issues

#### Issue: LoRA training memory blowup
```python
def optimize_lora_training():
    """Optimize LoRA training for memory efficiency"""
    
    config = {
        "lora_r": 8,              # Reduce rank
        "lora_alpha": 16,         # Reduce alpha
        "lora_dropout": 0.1,      # Keep dropout
        "per_device_train_batch_size": 1,  # Reduce batch size
        "gradient_accumulation_steps": 16,  # Accumulate gradients
        "gradient_checkpointing": True,     # Enable checkpointing
        "bf16": True,             # Use BF16
        "load_in_8bit": True,     # 8-bit quantization
        "lora_target_modules": ["q_proj", "v_proj"]  # Reduce target modules
    }
    
    return config
```

## Performance Issues

### Slow Training Speed

#### Issue: Training much slower than expected
```python
def diagnose_training_speed():
    """Diagnose slow training performance"""
    
    import time
    import psutil
    
    print("=== Training Speed Diagnosis ===")
    
    # 1. Check CPU utilization
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU utilization: {cpu_percent:.1f}%")
    
    # 2. Check GPU utilization
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_util = torch.cuda.utilization(i)  # Custom function
            print(f"GPU {i} utilization: {gpu_util:.1f}%")
    
    # 3. Check data loading performance
    start_time = time.time()
    # ... load and process a batch ...
    load_time = time.time() - start_time
    print(f"Batch loading time: {load_time:.3f}s")
    
    # 4. Check memory bandwidth
    # ... run memory benchmark ...
```

#### Solutions for slow training
```python
# 1. Enable optimizations
{
    "bfloat16": {"enabled": true},
    "zero_optimization": {"stage": 1},
    "gradient_checkpointing": true
}

# 2. Optimize data loading
dataloader_config = {
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 2
}

# 3. Use mixed precision
fp16_config = {
    "enabled": True,
    "opt_level": "O2",
    "loss_scale": "dynamic"
}

# 4. Optimize batch size
batch_size_optimization = {
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 32
}
```

### Communication Bottlenecks

#### Issue: High communication overhead
```python
def analyze_communication_overhead():
    """Analyze communication overhead in distributed training"""
    
    import torch.distributed as dist
    
    if not dist.is_initialized():
        print("Distributed not initialized")
        return
    
    # Measure all-reduce time
    tensor = torch.ones(1000000, device=f'cuda:{dist.get_rank()}')
    
    # Warmup
    for _ in range(10):
        dist.all_reduce(tensor)
    
    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    end_time = time.time()
    
    all_reduce_time = end_time - start_time
    print(f"All-reduce time: {all_reduce_time:.4f}s")
    
    # Calculate bandwidth
    tensor_size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
    bandwidth_mb_s = tensor_size_mb / all_reduce_time
    print(f"Effective bandwidth: {bandwidth_mb_s:.1f} MB/s")
```

#### Solutions for communication bottlenecks
```json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_bucket_size": 2e8,
        "reduce_bucket_size": 2e8,
        "prefetch_bucket_size": 5e7,
        "sub_group_size": 1e9
    },
    "communication": {
        "backend": "nccl",
        "timeout": 1800
    }
}
```

## Model Serving Issues

### FastAPI Serving Issues

#### Issue: Model serving crashes on startup
```python
def debug_model_serving():
    """Debug model serving issues"""
    
    try:
        from utils.model_serving import ModelServer
        
        # Test model loading
        server = ModelServer()
        
        # Test with minimal configuration
        server.config = {
            "model_path": "./models/test_model",
            "max_length": 512,
            "device": "cpu"  # Start with CPU
        }
        
        # Try loading model
        print("Testing model loading...")
        server.load_model()
        
        print("✓ Model loading successful")
        
        # Test inference
        test_input = "Test medical query"
        print(f"Testing inference with: {test_input}")
        
        result = server.generate(test_input)
        print(f"Inference result: {result}")
        
    except Exception as e:
        print(f"❌ Model serving debug failed: {e}")
        
        # Common fixes
        print("Common solutions:")
        print("1. Check model path exists")
        print("2. Verify model format")
        print("3. Try CPU device first")
        print("4. Check available memory")
```

#### Issue: Slow inference performance
```python
def optimize_inference_performance():
    """Optimize model serving performance"""
    
    serving_config = {
        "model_loading": {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": "float16"
        },
        "generation": {
            "max_length": 512,
            "do_sample": False,  # Faster for deterministic tasks
            "num_beams": 1,
            "use_cache": True
        },
        "server": {
            "workers": 1,  # Single worker for GPU serving
            "max_connections": 100
        },
        "caching": {
            "enabled": True,
            "cache_size": 1000,
            "cache_ttl": 3600
        }
    }
    
    return serving_config
```

This completes the comprehensive Troubleshooting Guide. Each section provides detailed diagnosis and solution steps for common issues in the Medical AI Training Pipeline, from installation problems to performance optimization.
