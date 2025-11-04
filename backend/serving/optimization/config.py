"""
Configuration management for optimization framework.
"""

import os
import json
from enum import Enum
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict


class OptimizationLevel(Enum):
    """Optimization levels for different use cases."""
    MINIMAL = "minimal"           # No optimization, maximum accuracy
    BALANCED = "balanced"         # Balanced optimization and accuracy
    AGGRESSIVE = "aggressive"     # Maximum optimization, acceptable accuracy loss
    MEDICAL_CRITICAL = "medical_critical"  # Preserve medical accuracy


class QuantizationType(Enum):
    """Types of quantization supported."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    DYNAMIC = "dynamic"
    STATIC = "static"


class MemoryProfile(Enum):
    """Memory usage profiles."""
    LOW = "low"           # < 4GB
    MEDIUM = "medium"     # 4-8GB
    HIGH = "high"         # 8-16GB
    UNLIMITED = "unlimited"  # > 16GB


class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


class ReductionType(Enum):
    """Model reduction types."""
    NONE = "none"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    BOTH = "both"


@dataclass
class QuantizationConfig:
    """Configuration for quantization settings."""
    quantization_type: QuantizationType = QuantizationType.INT8
    use_bnb: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_8bit_quant_type: str = "dynamic"
    bnb_8bit_threshold: float = 6.0
    bnb_8bit_use_double_quant: bool = True
    bnb_llm_int8_threshold: float = 6.0
    bnb_4bit_use_double_quant: bool = True
    bnb_llm_int8_skip_modules: list = None
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    device_map: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    def __post_init__(self):
        if self.bnb_llm_int8_skip_modules is None:
            self.bnb_llm_int8_skip_modules = []


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    enable_gradient_checkpointing: bool = False
    enable_cpu_offload: bool = False
    enable_disk_offload: bool = False
    offload_threshold: float = 0.8
    max_memory_gb: Optional[float] = None
    offload_strategy: str = "mixed"
    check_for_nan: bool = False
    
    
@dataclass
class DeviceConfig:
    """Configuration for device management."""
    preferred_device: DeviceType = DeviceType.AUTO
    fallback_device: DeviceType = DeviceType.CPU
    device_memory_fraction: float = 0.8
    allow_cpu_mapped: bool = True
    device_specific_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.device_specific_kwargs is None:
            self.device_specific_kwargs = {}


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 32
    batch_timeout: float = 0.1
    enable_chunking: bool = True
    chunk_size: int = 1024
    enable_dynamic_batching: bool = True
    target_latency_ms: Optional[float] = None
    enable_early_exit: bool = False
    
    
@dataclass
class ReductionConfig:
    """Configuration for model reduction."""
    reduction_type: ReductionType = ReductionType.NONE
    prune_ratio: float = 0.0
    prune_method: str = "magnitude"
    distill_student_size_ratio: float = 0.5
    distill_temperature: float = 3.0
    distill_alpha: float = 0.5
    enable_structured_pruning: bool = True
    enable_granular_pruning: bool = False


@dataclass
class ValidationConfig:
    """Configuration for validation and testing."""
    enable_validation: bool = True
    benchmark_datasets: list = None
    accuracy_threshold: float = 0.95
    latency_threshold_ms: float = 1000.0
    memory_threshold_gb: float = 8.0
    enable_performance_profiling: bool = True
    validation_batch_size: int = 16
    warmup_iterations: int = 5
    
    def __post_init__(self):
        if self.benchmark_datasets is None:
            self.benchmark_datasets = ["medical_qa", "clinical_reasoning"]


@dataclass
class OptimizationConfig:
    """Main configuration class for optimization settings."""
    level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Core optimizations
    quantization: QuantizationConfig = None
    memory: MemoryConfig = None
    device: DeviceConfig = None
    batch: BatchConfig = None
    reduction: ReductionConfig = None
    validation: ValidationConfig = None
    
    # Auto-detection settings
    auto_detect_system: bool = True
    auto_adjust_for_medical: bool = True
    preserve_medical_accuracy: bool = True
    
    def __post_init__(self):
        if self.quantization is None:
            self.quantization = QuantizationConfig()
        if self.memory is None:
            self.memory = MemoryConfig()
        if self.device is None:
            self.device = DeviceConfig()
        if self.batch is None:
            self.batch = BatchConfig()
        if self.reduction is None:
            self.reduction = ReductionConfig()
        if self.validation is None:
            self.validation = ValidationConfig()
    
    @classmethod
    def from_env(cls, prefix: str = "OPTIMIZATION_") -> "OptimizationConfig":
        """Create config from environment variables."""
        config = cls()
        
        # Parse optimization level
        level_str = os.getenv(f"{prefix}LEVEL")
        if level_str:
            config.level = OptimizationLevel(level_str)
        
        # Parse quantization settings
        quant_type = os.getenv(f"{prefix}QUANTIZATION_TYPE")
        if quant_type:
            config.quantization.quantization_type = QuantizationType(quant_type)
        
        # Parse memory settings
        if os.getenv(f"{prefix}GRADIENT_CHECKPOINTING"):
            config.memory.enable_gradient_checkpointing = True
        
        # Parse device settings
        device_type = os.getenv(f"{prefix}PREFERRED_DEVICE")
        if device_type:
            config.device.preferred_device = DeviceType(device_type)
        
        # Parse batch settings
        max_batch = os.getenv(f"{prefix}MAX_BATCH_SIZE")
        if max_batch:
            config.batch.max_batch_size = int(max_batch)
        
        # Parse validation settings
        if os.getenv(f"{prefix}ENABLE_VALIDATION"):
            config.validation.enable_validation = True
        
        return config
    
    @classmethod
    def from_file(cls, config_path: str) -> "OptimizationConfig":
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert nested dictionaries to dataclass instances
        if 'quantization' in config_dict:
            config_dict['quantization'] = QuantizationConfig(**config_dict['quantization'])
        if 'memory' in config_dict:
            config_dict['memory'] = MemoryConfig(**config_dict['memory'])
        if 'device' in config_dict:
            config_dict['device'] = DeviceConfig(**config_dict['device'])
        if 'batch' in config_dict:
            config_dict['batch'] = BatchConfig(**config_dict['batch'])
        if 'reduction' in config_dict:
            config_dict['reduction'] = ReductionConfig(**config_dict['reduction'])
        if 'validation' in config_dict:
            config_dict['validation'] = ValidationConfig(**config_dict['validation'])
        
        return cls(**config_dict)
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def auto_adjust_for_medical_requirements(self):
        """Auto-adjust configuration for medical accuracy requirements."""
        if not self.auto_adjust_for_medical:
            return
        
        # For medical-critical tasks, be more conservative
        if self.preserve_medical_accuracy:
            if self.quantization.quantization_type == QuantizationType.INT4:
                self.quantization.quantization_type = QuantizationType.INT8
            
            # Enable validation for medical tasks
            self.validation.enable_validation = True
            self.validation.accuracy_threshold = 0.98
            
            # Limit memory optimization for stability
            self.memory.enable_disk_offload = False
        
        # Adjust based on optimization level
        if self.level == OptimizationLevel.MEDICAL_CRITICAL:
            self.quantization.quantization_type = QuantizationType.INT8
            self.reduction.reduction_type = ReductionType.NONE
            self.validation.accuracy_threshold = 0.99
        
        elif self.level == OptimizationLevel.MINIMAL:
            self.quantization.quantization_type = QuantizationType.NONE
            self.reduction.reduction_type = ReductionType.NONE
        
        elif self.level == OptimizationLevel.AGGRESSIVE:
            self.quantization.quantization_type = QuantizationType.INT4
            self.reduction.reduction_type = ReductionType.BOTH
            self.memory.enable_gradient_checkpointing = True
            self.memory.enable_cpu_offload = True
    
    def get_memory_profile(self) -> MemoryProfile:
        """Determine memory profile based on available system memory."""
        import psutil
        
        available_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_gb < 4:
            return MemoryProfile.LOW
        elif available_gb < 8:
            return MemoryProfile.MEDIUM
        elif available_gb < 16:
            return MemoryProfile.HIGH
        else:
            return MemoryProfile.UNLIMITED
    
    def adapt_to_memory_profile(self):
        """Adapt configuration based on memory profile."""
        profile = self.get_memory_profile()
        
        if profile == MemoryProfile.LOW:
            # Use aggressive optimization for low memory
            self.quantization.quantization_type = QuantizationType.INT4
            self.memory.enable_gradient_checkpointing = True
            self.memory.enable_cpu_offload = True
            self.batch.max_batch_size = 8
            
        elif profile == MemoryProfile.MEDIUM:
            # Moderate optimization
            self.quantization.quantization_type = QuantizationType.INT8
            self.memory.enable_gradient_checkpointing = True
            self.batch.max_batch_size = 16
            
        elif profile == MemoryProfile.HIGH:
            # Light optimization
            self.quantization.quantization_type = QuantizationType.INT8
            self.batch.max_batch_size = 24
            
        # UNLIMITED profile keeps default settings