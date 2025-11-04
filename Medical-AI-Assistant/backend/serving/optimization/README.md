# Medical AI Assistant - Optimization Framework

A comprehensive quantization and optimization framework designed specifically for Phase 6 of the Medical AI Assistant project. This framework provides advanced optimization techniques while maintaining the high accuracy standards required for medical applications.

## 🚀 Features

### Core Optimization Components

1. **Quantization Management**
   - 8-bit and 4-bit quantization using bitsandbytes
   - Automatic quantization strategy detection
   - Medical model accuracy preservation
   - Dynamic quantization switching based on system resources

2. **Memory Optimization**
   - Gradient checkpointing for reduced memory usage
   - CPU/GPU model offloading strategies
   - Intelligent memory management with monitoring
   - Emergency cleanup for memory pressure situations

3. **Device Optimization**
   - Automatic GPU/CPU detection and selection
   - Device-specific optimization strategies
   - Multi-GPU support with intelligent device mapping
   - Performance benchmarking and monitoring

4. **Batch Processing**
   - Dynamic batch sizing for optimal throughput
   - Multiple batching strategies (timeout, latency-aware, memory-aware)
   - Async batch processing for high-concurrency scenarios
   - Chunked processing for large inputs

5. **Model Reduction**
   - Neural network pruning (magnitude, structured, gradual)
   - Knowledge distillation for model compression
   - Medical-specific reduction strategies
   - Accuracy impact assessment

6. **Validation & Testing**
   - Comprehensive accuracy benchmarks
   - Medical-specific validation metrics
   - Performance regression testing
   - Detailed validation reports with visualizations

## 📋 Requirements

### System Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (recommended)
- 8GB+ system RAM (16GB+ recommended for large models)

### Optional Dependencies
```bash
pip install bitsandbytes  # For advanced quantization
pip install pynvml       # For detailed GPU monitoring
pip install tensorrt     # For high-performance inference
pip install accelerate   # For distributed training
```

## 🏗️ Architecture

```
serving/optimization/
├── __init__.py              # Module initialization
├── config.py               # Configuration management
├── quantization.py         # Quantization engine
├── memory_optimization.py  # Memory management
├── device_optimization.py  # Device management
├── batch_optimization.py   # Batch processing
├── model_reduction.py      # Model reduction
├── validation.py           # Validation & testing
├── utils.py               # Utility functions
├── example_usage.py       # Usage examples
└── README.md              # Documentation
```

## 🚦 Quick Start

### Basic Usage

```python
from serving.optimization import (
    OptimizationConfig, QuantizationManager, 
    ValidationConfig, SystemProfiler
)

# 1. System analysis
profiler = SystemProfiler()
system_info = profiler.get_optimization_recommendations()

# 2. Configuration
config = OptimizationConfig(
    level=OptimizationLevel.BALANCED,
    preserve_medical_accuracy=True
)

# 3. Model quantization
quant_manager = QuantizationManager(QuantizationConfig())
result = quant_manager.quantize_model(your_model)

# 4. Validation
validator = QuantizationValidator(ValidationConfig())
validation_report = validator.validate_quantization(
    original_model, quantized_model
)

print(f"Compression: {result.compression_ratio:.1f}x")
print(f"Accuracy: {validation_report.overall_score:.3f}")
```

### Medical-Specific Configuration

```python
from serving.optimization import (
    OptimizationConfig, OptimizationLevel,
    QuantizationType
)

# Configure for medical accuracy preservation
config = OptimizationConfig(
    level=OptimizationLevel.MEDICAL_CRITICAL,
    preserve_medical_accuracy=True,
    auto_adjust_for_medical=True
)

# Conservative quantization for medical models
quant_config = QuantizationConfig(
    quantization_type=QuantizationType.INT8,  # More conservative than INT4
    load_in_8bit=True
)
```

### Advanced Memory Optimization

```python
from serving.optimization import MemoryOptimizer, MemoryConfig

memory_config = MemoryConfig(
    enable_gradient_checkpointing=True,
    enable_cpu_offload=True,
    offload_threshold=0.8
)

memory_optimizer = MemoryOptimizer(memory_config)

# Enable gradient checkpointing
checkpoint_state = memory_optimizer.enable_gradient_checkpointing(model)

# Optimize memory layout
optimization_results = memory_optimizer.optimize_memory_layout(model)
```

### Batch Processing Setup

```python
from serving.optimization import BatchProcessor, BatchConfig, BatchRequest

batch_config = BatchConfig(
    max_batch_size=32,
    enable_dynamic_batching=True,
    target_latency_ms=100.0
)

def inference_function(batch_inputs):
    # Your model inference logic
    return [model(input) for input in batch_inputs]

batch_processor = BatchProcessor(batch_config, inference_function)
batch_processor.start()

# Submit requests
request = BatchRequest(
    request_id="req_001",
    input_data=your_input_data,
    priority=1
)

request_id = batch_processor.submit_request(request)
result = batch_processor.get_result(request_id, timeout=30.0)
```

## 🔧 Configuration

### Optimization Levels

- **MINIMAL**: No optimization, maximum accuracy
- **BALANCED**: Balanced optimization and accuracy
- **AGGRESSIVE**: Maximum optimization, acceptable accuracy loss
- **MEDICAL_CRITICAL**: Preserve medical accuracy standards

### Medical Compliance

The framework includes medical-specific safeguards:

```python
config = OptimizationConfig(
    preserve_medical_accuracy=True,
    auto_adjust_for_medical=True
)

# Automatically adjusts:
# - Uses INT8 instead of INT4 for critical models
# - Higher accuracy thresholds (98% vs 95%)
# - Mandatory validation for medical models
# - Conservative pruning ratios
```

## 📊 Validation & Testing

### Medical-Specific Metrics

```python
validation_config = ValidationConfig(
    accuracy_threshold=0.98,        # Higher for medical
    medical_compliance_required=True
)

validator = QuantizationValidator(validation_config)

# Metrics tested:
# - Medical accuracy
# - Clinical relevance
# - Safety score
# - Bias detection
# - Regulatory compliance
```

### Validation Report

```python
report = validator.validate_quantization(original_model, quantized_model)

# Report includes:
# - Overall compliance score
# - Individual metric scores
# - Threshold comparisons
# - Medical compliance status
# - Performance impact analysis
# - Recommendations for improvement
```

## 🎯 Model Reduction

### Pruning

```python
from serving.optimization import ModelReducer, PruningMethod

model_reducer = ModelReducer(ReductionConfig(
    prune_ratio=0.3,
    prune_method="magnitude"
))

# Available methods:
# - MAGNITUDE: Remove smallest weights
# - STRUCTURED: Remove entire neurons/channels
# - GRADUAL: Gradual pruning schedule
# - ACTIVATION: Based on activation patterns
```

### Knowledge Distillation

```python
# Create student model (smaller)
student_model = create_student_model()

# Apply distillation
distillation_result = model_reducer.apply_distillation(
    teacher_model=teacher_model,
    student_model=student_model,
    distillation_data=your_training_data
)

print(f"Compression: {distillation_result.compression_ratio:.1f}x")
print(f"Accuracy retention: {distillation_result.accuracy_retention:.3f}")
```

## 🔍 System Analysis

### Automatic System Detection

```python
profiler = SystemProfiler()

# Get system specifications
specs = profiler.specs

# Check optimization compatibility
compatibility = profiler.check_optimization_compatibility()

# Get recommendations
recommendations = profiler.get_optimization_recommendations()

# Benchmark performance
benchmark = profiler.benchmark_system_performance()
```

### Custom Validation

```python
# Define custom validation function
def medical_accuracy_validator(model, test_data):
    # Your medical-specific accuracy calculation
    return accuracy_score

# Use in optimization
pruning_result = model_reducer.apply_pruning(
    model,
    pruning_ratio=0.2,
    validation_function=medical_accuracy_validator
)
```

## 📈 Performance Monitoring

### Real-time Metrics

```python
# Memory monitoring
profile = memory_optimizer.get_memory_profile()
print(f"Memory usage: {profile.usage_percent:.1f}%")

# Batch processing metrics
batch_report = batch_processor.get_performance_report()
print(f"Throughput: {batch_report.metrics.throughput_requests_per_second:.1f} req/s")

# Device utilization
device_status = device_manager.get_device_status()
print(f"GPU utilization: {device_status['devices'][0]['utilization_percent']:.1f}%")
```

## 🛠️ Advanced Features

### Custom Optimization Strategies

```python
class CustomQuantizationManager(QuantizationManager):
    def _custom_quantization_logic(self, model):
        # Implement your custom quantization
        pass

# Use custom manager
custom_manager = CustomQuantizationManager(config)
```

### Async Processing

```python
async def async_inference(batch_inputs):
    # Async inference implementation
    return await your_async_model(batch_inputs)

async_batch_processor = AsyncBatchProcessor(batch_config, async_inference)
result = await async_batch_processor.submit_async_request(request)
```

### Model Compatibility Checking

```python
# Check if model is suitable for optimization
compatibility = OptimizationUtils.check_model_compatibility(
    model, 
    optimization_level="aggressive"
)

print(f"Compatible: {compatibility['compatible']}")
print(f"Warnings: {compatibility['warnings']}")
```

## 📚 Examples

See `example_usage.py` for comprehensive examples:

1. **Basic Optimization**: Simple quantization and validation
2. **Medical Model**: Complete medical model optimization pipeline
3. **Memory Optimization**: Advanced memory management
4. **Batch Processing**: High-throughput batch inference
5. **Model Reduction**: Pruning and distillation

## 🔒 Medical Compliance

### Accuracy Standards

- **Medical Accuracy**: ≥98% (vs 95% for general AI)
- **Safety Score**: ≥99%
- **Clinical Relevance**: ≥90%
- **Bias Score**: ≥85%

### Regulatory Considerations

The framework addresses:
- HIPAA compliance requirements
- FDA guidelines for medical AI
- Medical device regulations
- Clinical validation standards

### Bias Detection

```python
# Automatic bias testing
validation_report = validator.validate_quantization(
    original_model, quantized_model,
    custom_metrics=[ValidationMetric.BIAS_SCORE]
)

print(f"Bias score: {validation_report.medical_compliance}")
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Enable gradient checkpointing
   memory_config.enable_gradient_checkpointing = True
   
   # Enable CPU offloading
   memory_config.enable_cpu_offload = True
   ```

2. **Accuracy Loss After Quantization**
   ```python
   # Use INT8 instead of INT4 for medical models
   config.quantization.quantization_type = QuantizationType.INT8
   
   # Increase validation threshold
   config.validation.accuracy_threshold = 0.98
   ```

3. **Slow Batch Processing**
   ```python
   # Optimize batch size
   batch_config.max_batch_size = 64
   
   # Enable dynamic batching
   batch_config.enable_dynamic_batching = True
   ```

### Performance Tuning

1. **For Small Models (<100MB)**:
   - Use minimal optimization
   - Focus on accuracy preservation
   - Skip quantization

2. **For Large Models (>1GB)**:
   - Use aggressive optimization
   - Enable all memory optimizations
   - Apply model reduction

3. **For Medical Models**:
   - Always use MEDICAL_CRITICAL level
   - Validate all optimizations
   - Maintain conservative settings

## 📖 API Reference

### Main Classes

- `OptimizationConfig`: Configuration management
- `QuantizationManager`: Model quantization
- `MemoryOptimizer`: Memory optimization
- `DeviceManager`: Device management
- `BatchProcessor`: Batch processing
- `ModelReducer`: Model reduction
- `QuantizationValidator`: Validation and testing
- `SystemProfiler`: System analysis

### Key Methods

- `quantize_model()`: Apply quantization to model
- `enable_gradient_checkpointing()`: Reduce memory usage
- `select_optimal_device()`: Choose best device
- `submit_batch()`: Process batch requests
- `apply_pruning()`: Reduce model size
- `validate_quantization()`: Test optimized model

## 🤝 Contributing

When contributing to the optimization framework:

1. **Medical Accuracy**: All optimizations must maintain medical standards
2. **Validation**: New features require validation tests
3. **Documentation**: Update docs for new features
4. **Testing**: Include medical-specific test cases

## 📄 License

This optimization framework is part of the Medical AI Assistant project and follows the same licensing terms.

---

**Note**: This framework is designed specifically for medical AI applications where accuracy and safety are paramount. Always validate optimized models before deployment in medical contexts.