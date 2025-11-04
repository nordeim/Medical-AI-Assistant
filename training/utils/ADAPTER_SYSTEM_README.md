# Adapter Loading and Hot-Swap Utilities

This directory contains a comprehensive adapter management system for PEFT (Parameter-Efficient Fine-Tuning) models, providing production-ready features for loading, hot-swapping, serving, and monitoring adapters in real-time.

## 🚀 Overview

The adapter system provides:

- **Adapter Management**: PEFT adapter loading, unloading, and dynamic switching
- **Hot-Swap Functionality**: Zero-downtime adapter updates with validation
- **Model Serving**: FastAPI-based serving with adapter support
- **Performance Monitoring**: Comprehensive metrics and benchmarking
- **Resource Optimization**: Memory-efficient adapter caching and management
- **Compatibility Validation**: Automated adapter compatibility checking

## 📁 Components

### Core Adapter Management (`adapter_manager.py`)

The main `AdapterManager` class provides:

- **PEFT Adapter Loading**: Support for LoRA, AdaLoRA, IA3, Prefix Tuning, and P-Tuning
- **Dynamic Adapter Switching**: Real-time adapter changes without service interruption
- **Memory Management**: Intelligent caching and garbage collection
- **Performance Tracking**: Load times, memory usage, and inference metrics

```python
from training.utils import create_adapter_manager_async

# Create and initialize manager
manager = await create_adapter_manager_async(
    base_model_id="microsoft/DialoGPT-medium",
    max_memory_mb=8192,
    max_cache_size=5
)

# Load adapter
await manager.load_adapter_async("./adapters/general_assistant", "general")

# Switch adapter
manager.switch_adapter("general")

# Hot-swap to new adapter
await manager.hot_swap_adapter("./adapters/medical_specialist", timeout=30.0)

# Get performance metrics
metrics = await manager.benchmark_adapter("general")
print(f"Average latency: {metrics.inference_latency_ms:.1f}ms")
```

### Model Serving (`model_serving.py`)

Production-ready FastAPI serving with:

- **Adapter Support**: Multiple adapters served through single endpoint
- **Load Balancing**: Round-robin distribution across adapter instances
- **Performance Monitoring**: Prometheus metrics and health checks
- **Request Validation**: Input/output validation and filtering

```python
from training.utils import create_fastapi_app, ModelServingManager

# Create serving manager
manager = ModelServingManager(
    base_model_id="microsoft/DialoGPT-medium",
    adapter_configs={
        "general": {"path": "./adapters/general"},
        "medical": {"path": "./adapters/medical"}
    }
)

# Create FastAPI app
app = create_fastapi_app(manager)

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### CLI Interface (`scripts/serve_model.py`)

Command-line interface for:

- Server startup and management
- Health checks and monitoring
- Adapter switching and hot-swapping
- Configuration validation

```bash
# Start server
python -m training.scripts.serve_model start --config serving_config.yaml

# Check health
python -m training.scripts.serve_model health --config serving_config.yaml

# Hot-swap adapter
python -m training.scripts.serve_model hot-swap --adapter medical_v2 --config serving_config.yaml

# Get metrics
python -m training.scripts.serve_model metrics --config serving_config.yaml
```

### Configuration (`configs/serving_config.yaml`)

Comprehensive configuration covering:

- Model and adapter settings
- Server configuration
- Resource allocation
- Security and authentication
- Performance tuning
- Monitoring and alerting

### Adapter Validation (`adapter_validation.py`)

Automated compatibility validation:

- **Architecture Compatibility**: Model architecture and PEFT type matching
- **Configuration Validation**: PEFT config schema validation
- **Tokenizer Compatibility**: Special tokens and vocab size checks
- **Performance Assessment**: Load time and memory usage analysis
- **Deployment Readiness**: File validation and deployment checks

```python
from training.utils import validate_adapter_compatibility

result = validate_adapter_compatibility(
    adapter_path="./adapters/medical_v1",
    base_model_id="llama-7b-hf"
)

print(f"Compatibility: {result.compatibility_level.value}")
print(f"Score: {result.overall_score:.2f}")

if result.issues:
    for issue in result.issues:
        print(f"[{issue.severity}] {issue.category}: {issue.message}")
```

### Performance Benchmarking (`performance_benchmark.py`)

Comprehensive benchmarking suite:

- **Latency Benchmarks**: Single request latency with percentiles
- **Throughput Testing**: Requests and tokens per second
- **Memory Profiling**: CPU and GPU memory usage patterns
- **Quality Assessment**: Response quality scoring
- **Stress Testing**: Sustained load testing
- **Comparative Analysis**: Multi-adapter comparison

```python
from training.utils import PerformanceBenchmark, BenchmarkConfig

# Configure benchmark
config = BenchmarkConfig(
    num_benchmark_requests=100,
    prompt_lengths=[10, 50, 100, 200],
    enable_quality_check=True,
    concurrent_requests=4
)

# Run benchmark
benchmark = PerformanceBenchmark(manager, config)
results = await benchmark.run_comprehensive_benchmark("medical_v1")

print(results.summary())
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│  Serving Manager │────│  Adapter Mgr    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   Hot Swap Mgr  │────│  Memory Manager │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Metrics Client │────│  Validation Svc  │────│   Cache Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚦 Getting Started

### 1. Install Dependencies

```bash
pip install -r training/utils/requirements.txt
```

### 2. Prepare Configuration

Copy and modify the sample configuration:

```bash
cp training/configs/serving_config.yaml my_config.yaml
# Edit my_config.yaml with your settings
```

### 3. Load Adapters

```python
import asyncio
from training.utils import create_adapter_manager_async

async def load_adapters():
    manager = await create_adapter_manager_async(
        base_model_id="your-model-id"
    )
    
    # Load your adapters
    await manager.load_adapter_async("./adapters/general", "general")
    await manager.load_adapter_async("./adapters/medical", "medical")
    
    return manager

manager = asyncio.run(load_adapters())
```

### 4. Start Server

```bash
# Using CLI
python -m training.scripts.serve_model start --config my_config.yaml

# Or programmatically
from training.utils.model_serving import ServingLauncher
launcher = ServingLauncher("my_config.yaml")
launcher.run(workers=4)
```

### 5. Test Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Generate text
curl -X POST http://localhost:8000/generate \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "Explain quantum computing",
    "adapter_id": "general",
    "max_tokens": 100
  }'

# List adapters
curl http://localhost:8000/adapters

# Get metrics
curl http://localhost:8000/metrics
```

## 🔧 API Endpoints

### Health and Info

- `GET /health` - Server health status
- `GET /info` - Server information
- `GET /metrics/prometheus` - Prometheus metrics

### Adapter Management

- `GET /adapters` - List loaded adapters
- `POST /adapters/{id}/switch` - Switch to adapter
- `POST /adapters/{id}/hot-swap` - Hot-swap to adapter

### Text Generation

- `POST /generate` - Generate text using current/specified adapter

### Performance

- `GET /metrics` - Detailed performance metrics

## 📊 Monitoring

### Prometheus Metrics

The system exposes metrics at `/metrics/prometheus`:

- `model_requests_total` - Total requests processed
- `model_request_duration_seconds` - Request latency
- `active_adapters_count` - Number of loaded adapters
- `memory_usage_bytes` - Memory usage
- `inference_latency_seconds` - Inference latency by adapter
- `tokens_processed_total` - Total tokens processed

### Health Checks

- **Liveness**: Server process status
- **Readiness**: Adapter availability
- **Memory**: Memory usage thresholds
- **Performance**: Latency and throughput

## 🛡️ Security Features

### Input Validation

- Prompt length limits
- Token generation limits
- Content filtering options

### Rate Limiting

- Request rate limiting
- Burst protection
- Per-adapter limits

### Authentication

- Bearer token support
- API key management
- OAuth integration ready

## ⚡ Performance Optimizations

### Memory Management

- **Adapter Caching**: LRU cache with configurable size
- **Lazy Loading**: Load adapters on demand
- **Memory Monitoring**: Automatic cleanup and warnings
- **GPU Memory**: Efficient GPU memory management

### Serving Optimizations

- **Connection Pooling**: Efficient connection handling
- **Response Caching**: Optional response caching
- **Batch Processing**: Support for batch requests
- **Streaming**: Real-time response streaming

## 🔄 Hot-Swap Process

The hot-swap process ensures zero downtime:

1. **Pre-validation**: New adapter compatibility check
2. **Background Loading**: Load new adapter in background
3. **Validation Testing**: Test loaded adapter functionality
4. **Atomic Switch**: Replace active adapter atomically
5. **Cleanup**: Unload old adapter and free memory
6. **Rollback**: Automatic rollback on failure

## 🧪 Testing

### Adapter Validation

```python
from training.utils.adapter_validation import AdapterCompatibilityValidator

validator = AdapterCompatibilityValidator()
result = validator.validate_adapter(
    adapter_path="./adapters/medical",
    base_model_id="llama-7b-hf"
)

if result.compatibility_level.value != "full":
    print("Validation issues found:")
    for issue in result.issues:
        print(f"- {issue.message}")
```

### Performance Benchmarking

```python
from training.utils.performance_benchmark import benchmark_adapter

results = await benchmark_adapter(
    adapter_path="./adapters/general",
    base_model_id="llama-7b-hf",
    config=BenchmarkConfig(num_requests=100)
)

print(f"Latency: {results.avg_latency_ms:.1f}ms")
print(f"Throughput: {results.tokens_per_second:.1f} tok/s")
```

### Stress Testing

```python
from training.utils.performance_benchmark import BenchmarkSuite

suite = BenchmarkSuite(manager)
stress_results = await suite.benchmark_stress_test(
    adapter_id="medical",
    duration_seconds=300
)

print(f"Requests/sec: {stress_results['requests_per_second']:.1f}")
print(f"Error rate: {stress_results['error_rate']:.2%}")
```

## 📈 Scaling

### Horizontal Scaling

- **Load Balancing**: Multiple server instances
- **Adapter Distribution**: Distribute adapters across instances
- **Session Affinity**: Consistent adapter selection

### Vertical Scaling

- **GPU Resources**: Multi-GPU support
- **Memory Tuning**: Configurable memory limits
- **CPU Optimization**: Multi-threading support

## 🔧 Troubleshooting

### Common Issues

1. **Adapter Loading Failures**
   - Check adapter path and permissions
   - Validate adapter configuration
   - Ensure base model compatibility

2. **Memory Issues**
   - Reduce adapter cache size
   - Enable memory monitoring
   - Use adapter unloading

3. **Performance Degradation**
   - Check memory usage
   - Validate adapter quality
   - Monitor system resources

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start with reload for development
python -m training.scripts.serve_model start --config debug_config.yaml --reload
```

## 📚 Examples

See the `examples/` directory for complete usage examples:

- `basic_serving.py` - Simple serving setup
- `multi_adapter.py` - Multiple adapter management
- `hot_swap_demo.py` - Hot-swapping demonstration
- `benchmarking.py` - Performance benchmarking
- `custom_validation.py` - Custom validation rules

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This adapter system is licensed under the same license as the main project.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review the examples directory
3. Create an issue with detailed logs and configuration

---

**Ready to deploy production-ready adapter serving!** 🚀