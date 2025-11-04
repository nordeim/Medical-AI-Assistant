# DeepSpeed Distributed Training Troubleshooting Guide

## Common Issues and Solutions

### 1. Process Group Initialization Errors

#### Problem: `torch.distributed.ReducedConsistencyError`
```bash
Error: Reduced consistency error during initialization
```

**Solutions:**
- Check that all processes are using the same `MASTER_ADDR` and `MASTER_PORT`
- Ensure firewalls allow communication between nodes
- Verify network connectivity between all nodes

```bash
# Test network connectivity
nc -zv <master_addr> <master_port>

# Check if processes are using same environment variables
echo $MASTER_ADDR $MASTER_PORT $WORLD_SIZE $RANK
```

#### Problem: `RuntimeError: NCCLcommInitRank failed`
```bash
Error: NCCLcommInitRank failed: invalid device ordinal
```

**Solutions:**
- Check that `LOCAL_RANK` is within the range of available GPUs
- Ensure `torch.cuda.set_device(local_rank)` is called before NCCL initialization

```python
# Correct initialization
torch.cuda.set_device(local_rank)
deepspeed.init_distributed(dist_backend='nccl')
```

### 2. Memory-Related Errors

#### Problem: `RuntimeError: CUDA out of memory`
```bash
Error: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
- Enable ZeRO optimization (Stage 1, 2, or 3)
- Enable CPU offloading in ZeRO configuration
- Reduce batch size or enable gradient accumulation

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
        }
    }
}
```

#### Problem: `RuntimeError: NCCL remote error`
```bash
Error: NCCL remote error encountered
```

**Solutions:**
- Increase communication timeout values
- Enable CUDA IPC for faster GPU communication
- Check for bad network connections

```json
{
    "communication_backend": {
        "name": "nccl",
        "timeout": 600
    }
}
```

### 3. Communication Bottlenecks

#### Problem: Slow training throughput
```bash
Warning: Low throughput detected - communication bottleneck
```

**Solutions:**
- Optimize NCCL settings
- Use gradient accumulation instead of small batch sizes
- Enable communication compression

```python
# Set NCCL environment variables
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_MIN_NRINGS"] = "4"

# Enable compression in config
{
    "communication": {
        "compression": {
            "method": "fp16",
            "config": {"enabled": true}
        }
    }
}
```

#### Problem: `TimeoutError` during training
```bash
Error: Grad sync timeout - communication taking too long
```

**Solutions:**
- Increase timeout values
- Reduce communication frequency
- Optimize batch sizes and gradient accumulation

```json
{
    "timeout": {
        "grad_sync_timeout": 1200,
        "param_sync_timeout": 1200
    }
}
```

### 4. Checkpoint Issues

#### Problem: `Failed to load checkpoint`
```bash
Error: Checkpoint loading failed - incompatible versions
```

**Solutions:**
- Ensure DeepSpeed version compatibility
- Check checkpoint integrity
- Verify model state dict compatibility

```bash
# Check checkpoint files
ls -la checkpoints/checkpoint_*/manager.py

# Verify checkpoint structure
python -c "import torch; print(torch.load('checkpoint/checkpoint_*/distrib_description.pt').keys())"
```

#### Problem: Checkpoint size too large
```bash
Warning: Checkpoint size exceeds disk capacity
```

**Solutions:**
- Enable 16-bit weight storage
- Reduce checkpoint frequency
- Use incremental checkpointing

```json
{
    "zero_optimization": {
        "gather_16bit_weights_on_model_save": true
    },
    "checkpointing": {
        "save_checkpoint_limit": 3
    }
}
```

### 5. Model Compatibility Issues

#### Problem: `TypeError: unsupported type`
```bash
Error: Data type not supported for distributed training
```

**Solutions:**
- Ensure model supports the data type
- Check for unsupported operations in distributed mode
- Use compatible layers and activations

```python
# Check model compatibility
from utils.deepspeed_utils import ModelValidator
validation_results = ModelValidator.validate_model_for_distributed_training(model)
print(validation_results)
```

#### Problem: Embedding layers not working with ZeRO
```bash
Error: Embedding layers require special handling with ZeRO
```

**Solutions:**
- Configure ZeRO to exclude embedding layers
- Use different partitioning strategy

```json
{
    "zero_optimization": {
        "override_module_apply": true,
        "sub_group_size": 1e9
    }
}
```

### 6. Performance Issues

#### Problem: Slow communication between nodes
```bash
Warning: Communication overhead > 50% of total time
```

**Solutions:**
- Optimize NCCL algorithm settings
- Use faster network protocols
- Increase batch sizes appropriately

```bash
# Optimize NCCL settings
export NCCL_NET_GDR_LEVEL=4
export NCCL_IB_HCA=mlx5
export NCCL_IB_SL=5
export NCCL_IB_TC=136
```

#### Problem: Inconsistent training across nodes
```bash
Warning: Metrics inconsistency detected across nodes
```

**Solutions:**
- Ensure all nodes use identical configuration
- Synchronize random seeds
- Use distributed samplers

```python
# Set consistent seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)

# Use distributed sampler
sampler = DistributedSampler(dataset, shuffle=True)
```

## Diagnostic Commands

### 1. System Information
```bash
# Check GPU information
nvidia-smi

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Check DeepSpeed installation
python -c "import deepspeed; print(deepspeed.__version__)"
```

### 2. Network Diagnostics
```bash
# Test network bandwidth between nodes
iperf3 -c <master_addr> -p <master_port>

# Check network connectivity
ping -c 5 <master_addr>

# Verify firewall settings
firewall-cmd --list-ports
```

### 3. Performance Monitoring
```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Monitor network traffic
iftop -i eth0

# Check system resources
htop
```

## Environment Variables

### Key DeepSpeed Environment Variables
```bash
# DeepSpeed specific
export DS_ACCELERATOR=cuda
export DS_ACCELERATORATOR=cuda

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# CUDA settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Communication settings
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=4
```

## Debugging Tips

### 1. Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in DeepSpeed config
{
    "logging": {
        "level": "DEBUG",
        "version": 1
    }
}
```

### 2. Use Memory Profiling
```python
from utils.deepspeed_utils import MemoryProfiler
memory_profiler = MemoryProfiler()
memory_profiler.log_memory_usage("Debug checkpoint")
memory_profiler.print_summary()
```

### 3. Validate Model
```python
from utils.deepspeed_utils import ModelValidator
validation_results = ModelValidator.validate_model_for_distributed_training(model)
print(f"Model compatible: {validation_results['compatible']}")
if validation_results['warnings']:
    print(f"Warnings: {validation_results['warnings']}")
```

### 4. Test Communication
```python
from utils.deepspeed_utils import CommunicationOptimizer
results = CommunicationOptimizer.benchmark_communication(world_size)
print(f"Communication benchmark results: {results}")
```

## Common Configuration Patterns

### Small Models (< 1B parameters)
```json
{
    "zero_optimization": {
        "stage": 1,
        "cpu_offload": false
    },
    "train_micro_batch_size_per_gpu": 8
}
```

### Medium Models (1-10B parameters)
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    },
    "train_micro_batch_size_per_gpu": 4
}
```

### Large Models (> 10B parameters)
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "train_micro_batch_size_per_gpu": 2
}
```

## Getting Help

If you continue to experience issues:

1. Check the DeepSpeed GitHub issues: https://github.com/microsoft/DeepSpeed/issues
2. Consult the DeepSpeed documentation: https://www.deepspeed.ai/getting-started/
3. Use the performance monitoring tools to gather diagnostic information
4. Provide a minimal reproducible example with:
   - System configuration
   - DeepSpeed version
   - Configuration file
   - Error logs
   - Memory/network diagnostics

## Performance Optimization Checklist

- [ ] Enable appropriate ZeRO stage based on model size
- [ ] Configure CPU offloading for large models
- [ ] Set optimal NCCL parameters
- [ ] Enable mixed precision training (bf16/fp16)
- [ ] Configure communication compression
- [ ] Set appropriate timeout values
- [ ] Enable memory profiling and monitoring
- [ ] Use distributed samplers for datasets
- [ ] Configure checkpoint saving strategy
- [ ] Optimize batch sizes and gradient accumulation
- [ ] Enable activation checkpointing for memory efficiency
- [ ] Configure TensorBoard/WandB for monitoring