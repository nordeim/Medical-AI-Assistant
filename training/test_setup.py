#!/usr/bin/env python3
"""
Test script for DeepSpeed distributed training setup
Validates configurations, dependencies, and basic functionality.
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

def setup_logging():
    """Setup logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    logger = logging.getLogger(__name__)
    
    if sys.version_info < (3.8, 0):
        logger.error(f"Python 3.8+ required, found {sys.version}")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    logger = logging.getLogger(__name__)
    required_packages = [
        'torch',
        'deepspeed',
        'transformers',
        'datasets',
        'numpy',
        'psutil',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_cuda_availability():
    """Check CUDA availability."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {device_name}")
            return True
        else:
            logger.warning("! CUDA not available - training will use CPU (slow)")
            return False
    
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return False

def check_deepspeed_version():
    """Check DeepSpeed version and configuration."""
    logger = logging.getLogger(__name__)
    
    try:
        import deepspeed
        
        version = deepspeed.__version__
        logger.info(f"✓ DeepSpeed version: {version}")
        
        # Check if DeepSpeed is properly configured
        try:
            # Try to create a simple engine
            import torch.nn as nn
            import torch
            
            model = nn.Linear(10, 5)
            config = {
                "train_micro_batch_size_per_gpu": 1,
                "bf16": {"enabled": True}
            }
            
            engine, _, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=config
            )
            
            logger.info("✓ DeepSpeed engine initialization successful")
            return True
        
        except Exception as e:
            logger.error(f"✗ DeepSpeed engine initialization failed: {e}")
            return False
    
    except Exception as e:
        logger.error(f"Error checking DeepSpeed: {e}")
        return False

def test_config_files():
    """Test if configuration files are valid."""
    logger = logging.getLogger(__name__)
    config_files = [
        "deepspeed_config.json",
        "configs/single_node_config.json",
        "configs/multi_node_config.json",
        "configs/large_model_stage3_config.json"
    ]
    
    all_valid = True
    
    for config_file in config_files:
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.error(f"✗ Configuration file missing: {config_file}")
            all_valid = False
            continue
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Basic validation
            if "train_micro_batch_size_per_gpu" not in config:
                logger.warning(f"! {config_file} missing 'train_micro_batch_size_per_gpu'")
            
            logger.info(f"✓ Configuration file valid: {config_file}")
        
        except json.JSONDecodeError as e:
            logger.error(f"✗ Invalid JSON in {config_file}: {e}")
            all_valid = False
        except Exception as e:
            logger.error(f"✗ Error reading {config_file}: {e}")
            all_valid = False
    
    return all_valid

def test_utility_functions():
    """Test utility functions."""
    logger = logging.getLogger(__name__)
    
    try:
        # Add utils to path
        sys.path.insert(0, 'utils')
        from deepspeed_utils import DeepSpeedUtils, ModelValidator
        import torch.nn as nn
        
        logger.info("✓ Successfully imported utility functions")
        
        # Test model validation
        model = nn.Linear(10, 5)
        validation_result = ModelValidator.validate_model_for_distributed_training(model)
        
        if validation_result['compatible']:
            logger.info("✓ Model validation successful")
        else:
            logger.warning(f"! Model validation warnings: {validation_result.get('warnings', [])}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Utility function test failed: {e}")
        return False

def test_script_imports():
    """Test if training scripts can be imported."""
    logger = logging.getLogger(__name__)
    
    try:
        sys.path.insert(0, 'scripts')
        
        # Test benchmark script import
        import benchmark_performance
        logger.info("✓ Benchmark script can be imported")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Script import test failed: {e}")
        return False

def test_distributed_setup():
    """Test distributed training setup."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        import torch.distributed as dist
        
        # Check if distributed is available
        logger.info("✓ Distributed package available")
        
        # Test basic process group functionality (if available)
        if not dist.is_initialized():
            logger.info("! Distributed training not initialized (normal for single process)")
        else:
            logger.info("✓ Distributed training already initialized")
            logger.info(f"  World size: {dist.get_world_size()}")
            logger.info(f"  Rank: {dist.get_rank()}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Distributed setup test failed: {e}")
        return False

def run_system_info():
    """Run comprehensive system information."""
    logger = logging.getLogger(__name__)
    
    try:
        import psutil
        
        logger.info("System Information:")
        
        # CPU info
        logger.info(f"  CPU cores: {psutil.cpu_count()}")
        logger.info(f"  CPU usage: {psutil.cpu_percent(interval=1)}%")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"  Total RAM: {memory.total / 1024**3:.1f} GB")
        logger.info(f"  Available RAM: {memory.available / 1024**3:.1f} GB")
        logger.info(f"  Memory usage: {memory.percent}%")
        
        # GPU info
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.memory_stats(i)
                allocated = mem_info.get('allocated_bytes.all.current', 0)
                reserved = mem_info.get('reserved_bytes.all.current', 0)
                logger.info(f"  GPU {i} - Allocated: {allocated/1024**3:.2f}GB, Reserved: {reserved/1024**3:.2f}GB")
        
        return True
    
    except Exception as e:
        logger.error(f"Error gathering system info: {e}")
        return False

def run_benchmark_test():
    """Run a quick benchmark test."""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        import torch.nn as nn
        import time
        
        # Create a simple model
        model = nn.Linear(1024, 512).cuda()
        config = {
            "train_micro_batch_size_per_gpu": 2,
            "bf16": {"enabled": True}
        }
        
        if torch.cuda.is_available():
            engine, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=config
            )
            
            # Test forward/backward pass
            inputs = torch.randn(2, 1024).cuda()
            targets = torch.randn(2, 512).cuda()
            
            start_time = time.time()
            
            # Forward pass
            outputs = engine(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            
            # Backward pass
            engine.backward(loss)
            engine.step()
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"✓ Benchmark test completed in {elapsed_time:.4f}s")
            return True
        else:
            logger.warning("! Skipping GPU benchmark - no CUDA available")
            return True
    
    except Exception as e:
        logger.error(f"✗ Benchmark test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger = setup_logging()
    
    logger.info("DeepSpeed Distributed Training Setup Test")
    logger.info("=" * 50)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA Availability", check_cuda_availability),
        ("DeepSpeed Setup", check_deepspeed_version),
        ("Configuration Files", test_config_files),
        ("Utility Functions", test_utility_functions),
        ("Script Imports", test_script_imports),
        ("Distributed Setup", test_distributed_setup),
        ("System Information", run_system_info),
        ("Benchmark Test", run_benchmark_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("Test Summary:")
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✓ All tests passed! DeepSpeed setup is ready for distributed training.")
    else:
        logger.error("✗ Some tests failed. Please fix the issues before starting distributed training.")
        sys.exit(1)

if __name__ == "__main__":
    main()