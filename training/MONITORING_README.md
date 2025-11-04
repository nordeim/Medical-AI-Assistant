# Performance Monitoring and Metrics Tracking System

A comprehensive performance monitoring and metrics tracking system for machine learning training, providing real-time insights, automated reporting, and performance optimization recommendations.

## 🚀 Features

### Training Metrics
- **Training Progress**: Loss curves, accuracy tracking, learning rate schedules
- **Gradient Monitoring**: Gradient norms, exploding/vanishing gradient detection
- **Memory Tracking**: CPU/GPU memory usage, memory leak detection
- **Timing Analysis**: Batch processing times, data loading performance

### Model Performance
- **Inference Benchmarks**: Latency, throughput, memory footprint analysis
- **Model Optimization**: Quantization impact assessment, size optimization
- **Quality Metrics**: Model consistency and inference quality tracking

### System Monitoring
- **Resource Utilization**: CPU, GPU, memory, disk I/O, network traffic
- **Hardware Monitoring**: Temperature, power consumption, clock speeds
- **Process Tracking**: Per-process CPU and memory usage
- **Alert System**: Automated alerts for performance issues

### Real-time Visualization
- **Web Dashboard**: Interactive monitoring interface
- **Plotly Charts**: Real-time training progress visualization
- **TensorBoard Integration**: Compatible with TensorBoard logging
- **Export Capabilities**: JSON, CSV, and interactive HTML reports

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA-compatible GPU (optional, for GPU monitoring)

### Install Dependencies
```bash
cd training/
pip install -r requirements-monitoring.txt
```

### Optional GPU Monitoring
For full GPU monitoring capabilities:
```bash
pip install pynvml
```

## 🛠️ Quick Start

### 1. Basic Usage

```python
import sys
sys.path.append('../utils')

from monitor_training import TrainingMonitor
from performance_monitor import TrainingMetrics

# Create configuration
config = {
    'save_dir': './monitoring_logs',
    'enable_dashboard': True,
    'dashboard_port': 8080,
    'enable_alerts': True
}

# Create and start monitor
monitor = TrainingMonitor(config)
monitor.start_monitoring()

# In your training loop:
for step in range(num_steps):
    # ... your training code ...
    
    # Log metrics
    monitor.log_training_step(
        model=model,
        optimizer=optimizer,
        loss=loss.item(),
        accuracy=accuracy,
        learning_rate=lr,
        custom_metrics={'batch_size': batch_size}
    )
```

### 2. Using Configuration File

Create a `monitoring_config.yaml` file:
```yaml
dashboard:
  enabled: true
  host: "127.0.0.1"
  port: 8080

alert_system:
  enabled: true
  thresholds:
    cpu_usage_percent: 85.0
    memory_usage_percent: 80.0
```

Then use it:
```python
from monitor_training import TrainingMonitor

monitor = TrainingMonitor('monitoring_config.yaml')
```

### 3. Standalone Monitoring Script

Run the monitoring dashboard:
```bash
python scripts/monitor_training.py \
    --config configs/monitoring_config.yaml \
    --enable_dashboard \
    --dashboard_port 8080
```

Or with simulation:
```bash
python scripts/monitor_training.py --simulate --simulate_steps 100
```

## 📊 Dashboard Interface

Access the web-based monitoring dashboard at `http://127.0.0.1:8080`:

### Features:
- **Real-time Training Metrics**: Loss, accuracy, learning rate
- **System Resource Monitoring**: CPU, memory, GPU utilization
- **Alert Panel**: Active alerts and notifications
- **Interactive Charts**: Zoomable, filterable performance plots
- **Export Tools**: Download metrics data and reports

### Navigation:
- **Overview**: Main dashboard with key metrics
- **Training**: Detailed training progress charts
- **System**: Resource utilization over time
- **Alerts**: Alert history and management
- **Export**: Data export and reporting tools

## 🔧 Integration Examples

### PyTorch Training Integration

See `examples/integration_example.py` for a complete integration example:

```python
def train_model_with_monitoring():
    # Create model and optimizer
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create monitor
    monitor = TrainingMonitor(config)
    monitor.start_monitoring()
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Training steps
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Log to monitor
            monitor.log_training_step(
                model=model,
                optimizer=optimizer,
                loss=loss.item(),
                accuracy=accuracy,
                learning_rate=optimizer.param_groups[0]['lr']
            )
```

### Custom Metrics

Add custom metrics to your training:

```python
custom_metrics = {
    'tokens_per_second': tokens_processed / batch_time,
    'throughput': batch_size / batch_time,
    'memory_efficiency': effective_batch_size / peak_memory_mb
}

monitor.log_training_step(
    model=model,
    optimizer=optimizer,
    loss=loss.item(),
    custom_metrics=custom_metrics
)
```

### Alert Configuration

Set up custom alerts:

```python
# Add custom alert rule
monitor.alert_system.add_alert_rule(
    name="High Loss Spike",
    condition=lambda metrics: metrics.loss > threshold,
    severity="warning",
    message="Loss spike detected: {loss:.2f}",
    cooldown_seconds=300
)
```

## 📈 Metrics Reference

### Training Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `loss` | Training/validation loss | - |
| `accuracy` | Model accuracy | 0-1 |
| `learning_rate` | Current learning rate | - |
| `grad_norm` | Total gradient norm | - |
| `batch_time` | Time per batch | seconds |
| `cpu_memory_mb` | CPU memory usage | MB |
| `gpu_memory_mb` | GPU memory usage | MB |

### System Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `cpu_usage_percent` | CPU utilization | % |
| `memory_usage_percent` | System memory usage | % |
| `gpu_utilization` | GPU compute utilization | % |
| `gpu_temperature` | GPU temperature | °C |
| `gpu_power_watts` | GPU power consumption | W |
| `disk_read_mb_s` | Disk read rate | MB/s |

### Model Performance Metrics
| Metric | Description | Unit |
|--------|-------------|------|
| `avg_latency_ms` | Average inference latency | ms |
| `samples_per_second` | Inference throughput | samples/s |
| `model_size_mb` | Model size | MB |
| `quantization_ratio` | Size reduction from quantization | % |

## ⚙️ Configuration Options

### Dashboard Settings
```yaml
dashboard:
  enabled: true
  host: "0.0.0.0"  # Bind to all interfaces
  port: 8080
  refresh_interval: 5  # seconds
```

### Alert Thresholds
```yaml
alert_system:
  thresholds:
    cpu_usage_percent: 90.0
    memory_usage_percent: 85.0
    grad_norm_exploded: 10.0
    learning_rate_too_low: 1e-8
```

### Performance Optimization
```yaml
performance_tuning:
  high_frequency_mode: false
  buffer_size: 1000
  flush_interval: 10.0
```

## 🚨 Alert System

### Built-in Alerts
- **High CPU/Memory Usage**: System resource constraints
- **Gradient Explosion**: Training instability detection
- **Learning Rate Issues**: Learning rate too high/low
- **GPU Temperature**: Hardware safety monitoring
- **Training Anomalies**: Loss spikes, unusual patterns

### Custom Alerts
Create custom alerts based on your specific requirements:

```python
def custom_condition(metrics):
    return metrics.loss > target_loss * 2

monitor.alert_system.add_alert_rule(
    name="Target Loss Not Met",
    condition=custom_condition,
    severity="warning",
    message="Loss is {loss:.2f}, target was {target_loss:.2f}"
)
```

## 📊 Visualization and Reporting

### Generated Reports
- **Performance Summary**: Overall training statistics
- **Training Progress**: Loss/accuracy curves over time
- **Resource Utilization**: System resource usage patterns
- **Optimization Recommendations**: Automated performance suggestions

### Export Formats
- **JSON**: Machine-readable metrics data
- **CSV**: Spreadsheet-compatible data
- **HTML**: Interactive web reports
- **PNG**: Static chart images

### TensorBoard Integration
```python
config = {
    'tensorboard_dir': './logs',
    'enable_performance_monitor': True
}
```
Metrics are automatically logged to TensorBoard for advanced visualization.

## 🔧 Advanced Features

### Model Profiling
Benchmark model inference performance:

```python
# Profile model inference
dummy_input = torch.randn(1, 3, 224, 224).to(device)
monitor.log_model_inference_benchmark(model, device, dummy_input)
```

### Gradient Analysis
Detailed gradient monitoring:

```python
# Enable gradient monitoring
config = {
    'track_gradients': True,
    'gradient_monitoring': {
        'track_layers': ['encoder', 'decoder', 'classifier']
    }
}
```

### Distributed Training Support
Monitor multi-GPU/distributed training:

```python
# Configure for distributed training
config = {
    'distributed_monitoring': True,
    'track_rank_0_only': True  # Prevent log duplication
}
```

## 🔍 Troubleshooting

### Common Issues

**Dashboard not accessible**
```bash
# Check if port is in use
lsof -i :8080

# Use different port
python scripts/monitor_training.py --dashboard_port 8081
```

**High memory usage in monitoring**
```yaml
# Reduce buffer size and flush frequency
metrics_collection:
  buffer_size: 500
  flush_interval: 5.0
```

**Missing GPU metrics**
```bash
# Install pynvml for full GPU monitoring
pip install pynvml

# Or use fallback mode (limited metrics)
monitor.system_monitor = SystemMonitor(update_interval=5.0)
```

### Performance Optimization

**For fast training loops (>100 steps/sec):**
```yaml
performance_tuning:
  high_frequency_mode: true
  batch_metrics_collection: true
  max_events_per_second: 1000
```

**For large datasets:**
```yaml
performance_tuning:
  use_memory_maps: true
  auto_cleanup: true
  max_database_size_gb: 100.0
```

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd training/
pip install -r requirements-monitoring.txt
pip install -r requirements-dev.txt
```

### Running Tests
```bash
python -m pytest tests/test_performance_monitor.py
python -m pytest tests/test_metrics_collector.py
```

### Code Style
```bash
black training/utils/
flake8 training/
```

## 📄 License

This monitoring system is part of the Medical AI Assistant project. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and configuration examples
- **Issues**: Report bugs and feature requests via GitHub issues
- **Community**: Join discussions in project forums

## 🔄 Changelog

### v1.0.0 (Current)
- ✅ Complete training metrics monitoring
- ✅ Real-time web dashboard
- ✅ Alert system with custom rules
- ✅ Data visualization and export
- ✅ PyTorch integration
- ✅ GPU monitoring support
- ✅ TensorBoard compatibility

### Planned Features
- 🔄 MLflow integration
- 🔄 Weights & Biases compatibility
- 🔄 Kubernetes cluster monitoring
- 🔄 Advanced anomaly detection
- 🔄 Auto-scaling recommendations
- 🔄 Cost optimization insights