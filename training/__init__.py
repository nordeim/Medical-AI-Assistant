"""
LoRA/PEFT Training Package
Comprehensive package for training large language models with LoRA and other PEFT techniques.
"""

from .scripts.train_lora import (
    ModelConfig,
    LoRAConfig,
    QuantizationConfig,
    TrainingConfig,
    OptimizationConfig,
    DataConfig,
    LoRATrainer,
    CustomDataCollator,
    MemoryMonitoringCallback,
    ModelValidator,
    load_config_from_yaml
)

from .utils.model_utils import (
    ModelInfo,
    CheckpointInfo,
    ModelInfoCollector,
    ModelLoader,
    ModelSaver,
    ModelConverter,
    ModelValidator as UtilsModelValidator,
    ModelManager,
    ModelLoadingError,
    ModelSavingError,
    QuantizationError
)

from .utils.data_utils import (
    DataPreprocessingConfig,
    DataValidator,
    DataPreprocessor,
    ChatMLProcessor,
    DataAugmentation,
    DataStatistics
)

__version__ = "1.0.0"
__author__ = "LoRA Training Team"

__all__ = [
    # Training classes
    "ModelConfig",
    "LoRAConfig", 
    "QuantizationConfig",
    "TrainingConfig",
    "OptimizationConfig",
    "DataConfig",
    "LoRATrainer",
    "CustomDataCollator",
    "MemoryMonitoringCallback",
    "ModelValidator",
    "load_config_from_yaml",
    
    # Model utilities
    "ModelInfo",
    "CheckpointInfo",
    "ModelInfoCollector",
    "ModelLoader",
    "ModelSaver",
    "ModelConverter",
    "UtilsModelValidator",
    "ModelManager",
    "ModelLoadingError",
    "ModelSavingError",
    "QuantizationError",
    
    # Data utilities
    "DataPreprocessingConfig",
    "DataValidator",
    "DataPreprocessor",
    "ChatMLProcessor",
    "DataAugmentation",
    "DataStatistics",
    
    # Version info
    "__version__",
    "__author__"
]