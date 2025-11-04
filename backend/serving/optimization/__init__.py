"""
Medical AI Assistant - Optimization Framework

This module provides comprehensive quantization and optimization support for Phase 6,
including bitsandbytes integration, memory optimization, device management, and validation.

Components:
- Quantization: 8-bit/4-bit with automatic detection
- Memory Optimization: Gradient checkpointing, model offloading
- Device Optimization: GPU/CPU auto-detection and inference optimization
- Batch Processing: Throughput optimization
- Model Reduction: Pruning and distillation awareness
- Validation: Accuracy benchmarks and testing
"""

from .config import OptimizationConfig, OptimizationLevel
from .quantization import QuantizationManager, QuantizationConfig, QuantizationType
from .memory_optimization import MemoryOptimizer, MemoryProfile, OffloadStrategy
from .device_optimization import DeviceManager, DeviceType, InferenceMode
from .batch_optimization import BatchProcessor, BatchConfig
from .model_reduction import ModelReducer, ReductionType, PruningConfig
from .validation import QuantizationValidator, ValidationResult
from .utils import OptimizationUtils, SystemProfiler

__all__ = [
    "OptimizationConfig",
    "OptimizationLevel",
    "QuantizationManager",
    "QuantizationConfig",
    "QuantizationType",
    "MemoryOptimizer",
    "MemoryProfile",
    "OffloadStrategy",
    "DeviceManager",
    "DeviceType",
    "InferenceMode",
    "BatchProcessor",
    "BatchConfig",
    "ModelReducer",
    "ReductionType",
    "PruningConfig",
    "QuantizationValidator",
    "ValidationResult",
    "OptimizationUtils",
    "SystemProfiler",
]

__version__ = "1.0.0"