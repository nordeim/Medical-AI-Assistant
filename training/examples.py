#!/usr/bin/env python3
"""
Example usage of DeepSpeed distributed training configuration
Demonstrates different training scenarios and configurations.
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add the scripts and utils directories to path
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'utils')

def setup_logging():
    """Setup logging for examples."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def example_basic_training():
    """Example of basic single-node training."""
    logger = logging.getLogger(__name__)
    
    logger.info("Example 1: Basic Single-Node Training")
    logger.info("-" * 40)
    
    # Import after adding to path
    from train_distributed import main
    from utils.deepspeed_utils import DeepSpeedUtils, MemoryProfiler
    
    # Set environment for single-node training
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["RANK"] = "0"
    
    # Example command line arguments for basic training
    example_args = [
        "--config", "configs/single_node_config.json",
        "--model_name", "bert-base-uncased",
        "--dataset_path", "data/train.jsonl",
        "--epochs", "3",
        "--output_dir", "./examples/basic_training"
    ]
    
    logger.info("Command:")
    logger.info(f"torchrun --nproc_per_node=2 train_distributed.py {' '.join(example_args)}")
    
    logger.info("\nConfiguration Highlights:")
    logger.info("- ZeRO Stage 1 (basic optimizer state partitioning)")
    logger.info("- BF16 mixed precision")
    logger.info("- Small batch sizes (8)")
    logger.info("- Standard monitoring")

def example_multi_node_training():
    """Example of multi-node training setup."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 2: Multi-Node Training")
    logger.info("-" * 40)
    
    # Master node setup
    master_commands = {
        "environment": [
            "export MASTER_ADDR=192.168.1.100",
            "export NCCL_DEBUG=INFO",
            "export NCCL_MIN_NRINGS=4"
        ],
        "master_launch": [
            "torchrun \\",
            "  --nproc_per_node=4 \\",
            "  --nnodes=2 \\",
            "  --node_rank=0 \\",
            "  --master_addr=192.168.1.100 \\",
            "  --master_port=29500 \\",
            "  train_distributed.py \\",
            "  --config configs/multi_node_config.json \\",
            "  --model_name bert-large-uncased \\",
            "  --dataset_path /data/large_dataset \\",
            "  --epochs 10"
        ]
    }
    
    # Worker node setup
    worker_commands = {
        "environment": [
            "export MASTER_ADDR=192.168.1.100",
            "export NCCL_DEBUG=INFO"
        ],
        "worker_launch": [
            "torchrun \\",
            "  --nproc_per_node=4 \\",
            "  --nnodes=2 \\",
            "  --node_rank=1 \\",
            "  --master_addr=192.168.1.100 \\",
            "  --master_port=29500 \\",
            "  train_distributed.py \\",
            "  --config configs/multi_node_config.json \\",
            "  --model_name bert-large-uncased \\",
            "  --dataset_path /data/large_dataset \\",
            "  --epochs 10"
        ]
    }
    
    logger.info("Master Node (192.168.1.100):")
    for cmd in master_commands["environment"]:
        logger.info(f"  {cmd}")
    for cmd in master_commands["master_launch"]:
        logger.info(f"  {cmd}")
    
    logger.info("\nWorker Node:")
    for cmd in worker_commands["environment"]:
        logger.info(f"  {cmd}")
    for cmd in worker_commands["worker_launch"]:
        logger.info(f"  {cmd}")
    
    logger.info("\nConfiguration Highlights:")
    logger.info("- ZeRO Stage 2 with CPU offloading")
    logger.info("- Optimized network communication")
    logger.info("- Enhanced monitoring and logging")

def example_large_model_training():
    """Example of large model training with ZeRO Stage 3."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 3: Large Model Training (ZeRO Stage 3)")
    logger.info("-" * 40)
    
    # Command for large model training
    large_model_command = [
        "torchrun \\",
        "  --nproc_per_node=8 \\",
        "  --nnodes=4 \\",
        "  --master_addr=192.168.1.100 \\",
        "  train_distributed.py \\",
        "  --config configs/large_model_stage3_config.json \\",
        "  --model_name microsoft/dept-base \\",
        "  --dataset_path /data/huge_dataset \\",
        "  --epochs 20 \\",
        "  --output_dir ./models/large_model"
    ]
    
    logger.info("Command:")
    for cmd in large_model_command:
        logger.info(f"  {cmd}")
    
    logger.info("\nConfiguration Highlights:")
    logger.info("- ZeRO Stage 3 (complete model parallelism)")
    logger.info("- CPU offloading for optimizer and parameters")
    logger.info("- Small batch sizes (2)")
    logger.info("- Aggressive memory optimization")
    logger.info("- Communication compression")

def example_resume_training():
    """Example of resuming training from checkpoint."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 4: Resuming Training from Checkpoint")
    logger.info("-" * 40)
    
    resume_command = [
        "torchrun \\",
        "  --nproc_per_node=4 \\",
        "  train_distributed.py \\",
        "  --config configs/single_node_config.json \\",
        "  --model_name bert-base-uncased \\",
        "  --dataset_path /data/train.jsonl \\",
        "  --resume_from ./checkpoints/checkpoint-5000 \\",
        "  --epochs 10"
    ]
    
    logger.info("Command:")
    for cmd in resume_command:
        logger.info(f"  {cmd}")
    
    logger.info("\nFeatures:")
    logger.info("- Automatic checkpoint loading")
    logger.info("- Resume training from specific step")
    logger.info("- Maintains training state and optimizer")

def example_performance_benchmarking():
    """Example of running performance benchmarks."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 5: Performance Benchmarking")
    logger.info("-" * 40)
    
    # Comprehensive benchmark
    benchmark_commands = {
        "comprehensive": [
            "python scripts/benchmark_performance.py \\",
            "  --config configs/single_node_config.json \\",
            "  --benchmark_type all \\",
            "  --output_dir benchmark_results"
        ],
        "step_time_only": [
            "python scripts/benchmark_performance.py \\",
            "  --config configs/single_node_config.json \\",
            "  --benchmark_type step_time"
        ],
        "memory_profiling": [
            "python scripts/benchmark_performance.py \\",
            "  --config configs/single_node_config.json \\",
            "  --benchmark_type memory"
        ],
        "zero_comparison": [
            "python scripts/benchmark_performance.py \\",
            "  --config configs/single_node_config.json \\",
            "  --benchmark_type zero"
        ]
    }
    
    for benchmark_type, command in benchmark_commands.items():
        logger.info(f"\n{benchmark_type.title()} Benchmark:")
        for cmd in command:
            logger.info(f"  {cmd}")

def example_utility_usage():
    """Example of using utility functions."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 6: Using Utility Functions")
    logger.info("-" * 40)
    
    utility_examples = """
# Memory Profiling
from utils.deepspeed_utils import MemoryProfiler

memory_profiler = MemoryProfiler()
memory_info = memory_profiler.profile_memory("Training step")
memory_profiler.log_memory_usage("Current state")

# Performance Monitoring
from utils.deepspeed_utils import PerformanceMonitor

perf_monitor = PerformanceMonitor()
perf_monitor.log_step(global_step=100, batch_idx=5, batch_size=32)
perf_monitor.print_summary()

# Model Validation
from utils.deepspeed_utils import ModelValidator

validation_results = ModelValidator.validate_model_for_distributed_training(model)
print(f"Model compatible: {validation_results['compatible']}")
if validation_results['warnings']:
    print(f"Warnings: {validation_results['warnings']}")

# Checkpoint Management
from utils.deepspeed_utils import CheckpointManager

checkpoint_manager = CheckpointManager(save_dir="checkpoints")
checkpoint_manager.save_checkpoint(
    engine=engine,
    epoch=5,
    step=1000,
    metrics={"loss": 0.1, "accuracy": 0.95}
)

# Communication Optimization
from utils.deepspeed_utils import CommunicationOptimizer

CommunicationOptimizer.optimize_communication_settings()
results = CommunicationOptimizer.benchmark_communication(world_size=4)
"""
    
    logger.info("Utility Function Examples:")
    logger.info(utility_examples)

def example_troubleshooting():
    """Example troubleshooting scenarios."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 7: Common Troubleshooting Scenarios")
    logger.info("-" * 40)
    
    scenarios = [
        {
            "problem": "CUDA out of memory",
            "solution": "Use ZeRO Stage 2 or 3 with CPU offloading",
            "config": "configs/large_model_stage3_config.json"
        },
        {
            "problem": "Slow communication between nodes",
            "solution": "Optimize NCCL settings and check network",
            "environment": "export NCCL_MIN_NRINGS=4\nexport NCCL_ASYNC_ERROR_HANDLING=1"
        },
        {
            "problem": "Process group initialization failed",
            "solution": "Check MASTER_ADDR and port, ensure all nodes can communicate",
            "check": "ping $MASTER_ADDR\nnc -zv $MASTER_ADDR $MASTER_PORT"
        },
        {
            "problem": "Inconsistent training results",
            "solution": "Synchronize random seeds and use distributed samplers",
            "code": "torch.manual_seed(42)\ntorch.cuda.manual_seed_all(42)\nsampler = DistributedSampler(dataset)"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        logger.info(f"\nScenario {i}: {scenario['problem']}")
        logger.info(f"  Solution: {scenario['solution']}")
        if 'config' in scenario:
            logger.info(f"  Config: {scenario['config']}")
        if 'environment' in scenario:
            logger.info(f"  Environment: {scenario['environment']}")
        if 'check' in scenario:
            logger.info(f"  Check: {scenario['check']}")
        if 'code' in scenario:
            logger.info(f"  Code: {scenario['code']}")

def example_script_usage():
    """Example of using launch scripts."""
    logger = logging.getLogger(__name__)
    
    logger.info("\nExample 8: Launch Script Usage")
    logger.info("-" * 40)
    
    script_examples = {
        "single_node": [
            "# Basic single-node training",
            "./scripts/launch_single_node.sh \\",
            "  --model_name bert-base-uncased \\",
            "  --dataset_path /data/train.jsonl \\",
            "  --epochs 10"
        ],
        "multi_node": [
            "# Multi-node training",
            "./scripts/launch_multi_node.sh \\",
            "  --master_addr 192.168.1.100 \\",
            "  --node_rank 0 \\",
            "  --nnodes 2 \\",
            "  --gpus_per_node 4 \\",
            "  --model_name bert-large-uncased \\",
            "  --dataset_path /data/large_dataset"
        ],
        "slurm": [
            "# SLURM cluster training",
            "sbatch \\",
            "  --nodes=4 \\",
            "  --ntasks-per-node=8 \\",
            "  --gpus-per-task=1 \\",
            "  --export=MODEL_NAME=bert-base-uncased,DATASET_PATH=/data/train.jsonl \\",
            "  scripts/launch_slurm.sbatch"
        ]
    }
    
    for script_type, commands in script_examples.items():
        logger.info(f"\n{script_type.title()} Script:")
        for cmd in commands:
            logger.info(f"  {cmd}")

def main():
    """Run all examples."""
    logger = setup_logging()
    
    logger.info("DeepSpeed Distributed Training Examples")
    logger.info("=" * 60)
    logger.info("This file demonstrates various training scenarios and configurations.")
    logger.info("Each example shows command-line usage and key features.")
    logger.info("=" * 60)
    
    examples = [
        example_basic_training,
        example_multi_node_training,
        example_large_model_training,
        example_resume_training,
        example_performance_benchmarking,
        example_utility_usage,
        example_troubleshooting,
        example_script_usage
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            logger.error(f"Error in example {example_func.__name__}: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Examples completed!")
    logger.info("For more detailed information, see:")
    logger.info("- README.md for comprehensive usage guide")
    logger.info("- TROUBLESHOOTING.md for issue resolution")
    logger.info("- test_setup.py for system validation")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()