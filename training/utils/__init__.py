"""
Training Utilities Package

This package provides comprehensive utilities for LoRA/PEFT training and serving of large language models.

Main Components:
- ModelManager: Model loading, saving, and conversion utilities
- DataPreprocessor: Data loading and preprocessing utilities
- ChatMLProcessor: ChatML format conversation processing
- ModelConverter: Model quantization and conversion tools
- AdapterManager: PEFT adapter loading, hot-swapping, and management
- ModelServing: FastAPI-based model serving with adapter support
- AdapterValidator: Adapter compatibility validation
- PerformanceBenchmark: Adapter performance benchmarking tools

Usage - Training:
    from training.utils import ModelManager, DataPreprocessor, ChatMLProcessor
    
    # Initialize components
    model_manager = ModelManager()
    data_preprocessor = DataPreprocessor(config)
    chatml_processor = ChatMLProcessor()
    
    # Load and process data
    dataset = data_preprocessor.prepare_dataset("data.jsonl")
    
    # Load model with LoRA
    model, tokenizer, peft_model = model_manager.setup_model(
        model_name="llama-7b",
        lora_config=lora_config
    )

Usage - Serving:
    from training.utils import create_adapter_manager_async, create_fastapi_app
    from training.scripts.serve_model import ServingLauncher
    
    # Create adapter manager
    manager = await create_adapter_manager_async("model_id")
    await manager.load_adapter_async("adapter_path", "adapter_id")
    
    # Create FastAPI app
    app = create_fastapi_app(manager)
    
    # Or use CLI
    from training.scripts.serve_model import ServingCLI
    cli = ServingCLI()
    await cli.run_server("config.yaml")

Author: LoRA Training Team
Version: 2.0.0
"""

from .model_utils import (
    ModelInfo,
    CheckpointInfo,
    ModelInfoCollector,
    ModelLoader,
    ModelSaver,
    ModelConverter,
    UtilsModelValidator,
    ModelManager,
    ModelLoadingError,
    ModelSavingError,
    QuantizationError
)

from .data_utils import (
    DataPreprocessingConfig,
    DataValidator,
    DataPreprocessor,
    ChatMLProcessor,
    DataAugmentation,
    DataStatistics
)

# Adapter System Imports
try:
    from .adapter_manager import (
        AdapterManager,
        AdapterMetadata,
        AdapterPerformanceMetrics,
        MemoryManager,
        AdapterValidator,
        AdapterCache,
        HotSwapManager,
        create_adapter_manager,
        create_adapter_manager_async
    )
    ADAPTER_SYSTEM_AVAILABLE = True
except ImportError as e:
    ADAPTER_SYSTEM_AVAILABLE = False
    AdapterManager = None
    # Set placeholder for missing dependencies
    adapter_import_error = str(e)

try:
    from .model_serving import (
        ModelServingManager,
        AdapterRequest,
        AdapterResponse,
        HealthResponse,
        ServingMetrics,
        LoadBalancer,
        create_fastapi_app,
        ServingLauncher
    )
    MODEL_SERVING_AVAILABLE = True
except ImportError as e:
    MODEL_SERVING_AVAILABLE = False

try:
    from .adapter_validation import (
        AdapterCompatibilityValidator,
        ValidationResult,
        CompatibilityLevel,
        CompatibilityIssue,
        validate_adapter_compatibility
    )
    ADAPTER_VALIDATION_AVAILABLE = True
except ImportError as e:
    ADAPTER_VALIDATION_AVAILABLE = False

try:
    from .performance_benchmark import (
        PerformanceBenchmark,
        BenchmarkConfig,
        BenchmarkMetrics,
        BenchmarkSuite,
        benchmark_adapter,
        compare_adapters
    )
    PERFORMANCE_BENCHMARK_AVAILABLE = True
except ImportError as e:
    PERFORMANCE_BENCHMARK_AVAILABLE = False

__version__ = "2.0.0"
__author__ = "LoRA Training Team"

__all__ = [
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
    
    # Adapter System (only if available)
    "AdapterManager",
    "AdapterMetadata", 
    "AdapterPerformanceMetrics",
    "MemoryManager",
    "AdapterValidator",
    "AdapterCache",
    "HotSwapManager",
    "create_adapter_manager",
    "create_adapter_manager_async",
    "ModelServingManager",
    "AdapterRequest",
    "AdapterResponse", 
    "HealthResponse",
    "ServingMetrics",
    "LoadBalancer",
    "create_fastapi_app",
    "ServingLauncher",
    "AdapterCompatibilityValidator",
    "ValidationResult",
    "CompatibilityLevel",
    "CompatibilityIssue",
    "validate_adapter_compatibility",
    "PerformanceBenchmark",
    "BenchmarkConfig",
    "BenchmarkMetrics", 
    "BenchmarkSuite",
    "benchmark_adapter",
    "compare_adapters"
]

# Available features
FEATURES = {
    "adapter_system": ADAPTER_SYSTEM_AVAILABLE,
    "model_serving": MODEL_SERVING_AVAILABLE,
    "adapter_validation": ADAPTER_VALIDATION_AVAILABLE,
    "performance_benchmark": PERFORMANCE_BENCHMARK_AVAILABLE
}

# Feature availability check
def check_feature(feature_name: str) -> bool:
    """Check if a feature is available."""
    return FEATURES.get(feature_name, False)

def get_system_info() -> dict:
    """Get information about available system features."""
    return {
        "version": __version__,
        "features": FEATURES,
        "missing_dependencies": {
            "adapter_system": not ADAPTER_SYSTEM_AVAILABLE,
            "model_serving": not MODEL_SERVING_AVAILABLE, 
            "adapter_validation": not ADAPTER_VALIDATION_AVAILABLE,
            "performance_benchmark": not PERFORMANCE_BENCHMARK_AVAILABLE
        } if not all(FEATURES.values()) else None
    }