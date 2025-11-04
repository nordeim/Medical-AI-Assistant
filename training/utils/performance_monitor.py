"""
Performance Monitoring System for ML Training

This module provides comprehensive performance monitoring capabilities for machine learning
training, including training metrics, model performance, and system resource monitoring.
"""

import asyncio
import gc
import json
import logging
import math
import os
import psutil
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training-specific metrics."""
    # Basic training metrics
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    phase: str = "train"  # train, val, test
    
    # Loss and accuracy
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    momentum: float = 0.0
    
    # Gradient metrics
    grad_norm: float = 0.0
    grad_norm_clipped: float = 0.0
    param_norm: float = 0.0
    
    # Timing metrics
    batch_time: float = 0.0
    data_load_time: float = 0.0
    forward_pass_time: float = 0.0
    backward_pass_time: float = 0.0
    optimization_time: float = 0.0
    
    # Memory metrics
    cpu_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_utilization: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result.pop('custom_metrics', None)  # Handle separately
        result.update(self.custom_metrics)
        return result


@dataclass
class ModelPerformanceMetrics:
    """Container for model inference performance metrics."""
    # Latency metrics
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    
    # Throughput metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Memory metrics
    model_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Model size and optimization
    model_size_mb: float = 0.0
    quantization_ratio: float = 0.0
    compression_ratio: float = 0.0
    
    # Quality metrics
    inference_quality_score: float = 0.0
    consistency_score: float = 0.0
    
    # Metadata
    batch_size: int = 0
    sequence_length: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SystemMetrics:
    """Container for system resource metrics."""
    # CPU metrics
    cpu_usage_percent: float = 0.0
    cpu_temperature: float = 0.0
    cpu_freq: float = 0.0
    
    # Memory metrics
    memory_usage_percent: float = 0.0
    memory_available_gb: float = 0.0
    swap_usage_percent: float = 0.0
    
    # GPU metrics
    gpu_utilization: float = 0.0
    gpu_memory_utilization: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_watts: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Disk I/O metrics
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    disk_usage_percent: float = 0.0
    
    # Network metrics
    network_sent_mb_s: float = 0.0
    network_recv_mb_s: float = 0.0
    
    # Process metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class SystemMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, device_ids: Optional[List[int]] = None, update_interval: float = 1.0):
        self.device_ids = device_ids or list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.update_interval = update_interval
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._latest_metrics: Optional[SystemMetrics] = None
        
        # I/O tracking
        self._last_disk_io = psutil.disk_io_counters()
        self._last_network_io = psutil.net_io_counters()
        self._last_time = time.time()
        
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("System monitoring stopped")
    
    def get_latest_metrics(self) -> Optional[SystemMetrics]:
        """Get the latest system metrics."""
        return self._latest_metrics
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._latest_metrics = self._collect_system_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_temp = self._get_cpu_temperature()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024
        process_cpu = process.cpu_percent()
        
        # GPU metrics
        gpu_metrics = self._get_gpu_metrics()
        
        # Disk I/O metrics
        disk_read, disk_write = self._get_disk_io_rate()
        
        # Network metrics
        network_sent, network_recv = self._get_network_io_rate()
        
        return SystemMetrics(
            cpu_usage_percent=cpu_percent,
            cpu_temperature=cpu_temp,
            cpu_freq=cpu_freq.current if cpu_freq else 0.0,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory.available / 1024 / 1024 / 1024,
            swap_usage_percent=swap.percent,
            gpu_utilization=gpu_metrics.get('utilization', 0.0),
            gpu_memory_utilization=gpu_metrics.get('memory_utilization', 0.0),
            gpu_temperature=gpu_metrics.get('temperature', 0.0),
            gpu_power_watts=gpu_metrics.get('power', 0.0),
            gpu_memory_mb=gpu_metrics.get('memory_mb', 0.0),
            disk_read_mb_s=disk_read,
            disk_write_mb_s=disk_write,
            disk_usage_percent=psutil.disk_usage('/').percent,
            network_sent_mb_s=network_sent,
            network_recv_mb_s=network_recv,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory
        )
    
    def _get_cpu_temperature(self) -> float:
        """Get CPU temperature."""
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                return temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                return temps['cpu_thermal'][0].current
        except:
            pass
        return 0.0
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics using nvidia-ml-py if available."""
        if not torch.cuda.is_available() or not self.device_ids:
            return {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            
            metrics = {}
            for device_id in self.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['utilization'] = util.gpu
                
                # Memory utilization
                metrics['memory_utilization'] = util.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics['temperature'] = temp
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    metrics['power'] = power
                except:
                    metrics['power'] = 0.0
                
                # Memory usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['memory_mb'] = mem_info.used / 1024 / 1024
            
            pynvml.nvmlShutdown()
            return metrics
            
        except ImportError:
            # Fallback to torch.cuda if nvidia-ml-py not available
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024
                memory_cached = torch.cuda.memory_reserved(device) / 1024 / 1024
                return {
                    'memory_mb': memory_allocated,
                    'memory_utilization': memory_cached / (torch.cuda.get_device_properties(device).total_memory / 1024 / 1024) * 100
                }
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
        
        return {}
    
    def _get_disk_io_rate(self) -> Tuple[float, float]:
        """Get disk I/O rate in MB/s."""
        try:
            current_io = psutil.disk_io_counters()
            current_time = time.time()
            
            if self._last_disk_io is None:
                self._last_disk_io = current_io
                self._last_time = current_time
                return 0.0, 0.0
            
            time_diff = current_time - self._last_time
            
            read_bytes = current_io.read_bytes - self._last_disk_io.read_bytes
            write_bytes = current_io.write_bytes - self._last_disk_io.write_bytes
            
            read_rate = (read_bytes / time_diff) / (1024 * 1024)  # MB/s
            write_rate = (write_bytes / time_diff) / (1024 * 1024)  # MB/s
            
            self._last_disk_io = current_io
            self._last_time = current_time
            
            return max(0, read_rate), max(0, write_rate)
            
        except Exception as e:
            logger.warning(f"Failed to collect disk I/O metrics: {e}")
            return 0.0, 0.0
    
    def _get_network_io_rate(self) -> Tuple[float, float]:
        """Get network I/O rate in MB/s."""
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            if self._last_network_io is None:
                self._last_network_io = current_io
                self._last_time = current_time
                return 0.0, 0.0
            
            time_diff = current_time - self._last_time
            
            sent_bytes = current_io.bytes_sent - self._last_network_io.bytes_sent
            recv_bytes = current_io.bytes_recv - self._last_network_io.bytes_recv
            
            sent_rate = (sent_bytes / time_diff) / (1024 * 1024)  # MB/s
            recv_rate = (recv_bytes / time_diff) / (1024 * 1024)  # MB/s
            
            self._last_network_io = current_io
            self._last_time = current_time
            
            return max(0, sent_rate), max(0, recv_rate)
            
        except Exception as e:
            logger.warning(f"Failed to collect network I/O metrics: {e}")
            return 0.0, 0.0


class GradientMonitor:
    """Monitor gradients during training."""
    
    def __init__(self, track_layers: Optional[List[str]] = None):
        self.track_layers = track_layers or []
        self.gradient_norms: Deque[float] = deque(maxlen=100)
        self.layer_grad_norms: Dict[str, List[float]] = defaultdict(list)
        self.gradient_stats: Dict[str, float] = {}
        
    def monitor_step(self, model: torch.nn.Module) -> Dict[str, float]:
        """Monitor gradients after a training step."""
        total_norm = 0.0
        param_count = 0
        layer_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                # Track specific layers if specified
                if not self.track_layers or any(layer in name for layer in self.track_layers):
                    self.layer_grad_norms[name].append(param_norm.item())
                    
                    # Keep only recent values
                    if len(self.layer_grad_norms[name]) > 100:
                        self.layer_grad_norms[name].pop(0)
        
        total_norm = total_norm ** (1.0 / 2) if param_count > 0 else 0.0
        
        self.gradient_norms.append(total_norm)
        self.gradient_stats = {
            'grad_norm': total_norm,
            'avg_grad_norm': np.mean(self.gradient_norms) if self.gradient_norms else 0.0,
            'max_grad_norm': max(self.gradient_norms) if self.gradient_norms else 0.0,
            'min_grad_norm': min(self.gradient_norms) if self.gradient_norms else 0.0,
            'grad_norm_std': np.std(self.gradient_norms) if len(self.gradient_norms) > 1 else 0.0
        }
        
        return self.gradient_stats
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        return self.gradient_stats.copy()
    
    def get_layer_stats(self, layer_name: str) -> Dict[str, float]:
        """Get statistics for a specific layer."""
        if layer_name not in self.layer_grad_norms:
            return {}
        
        norms = self.layer_grad_norms[layer_name]
        if not norms:
            return {}
        
        return {
            'avg': np.mean(norms),
            'max': max(norms),
            'min': min(norms),
            'std': np.std(norms),
            'count': len(norms)
        }


class ModelProfiler:
    """Profile model inference performance."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
        # Model size analysis
        self.model_size_mb = self._calculate_model_size()
        
        # Inference benchmarks
        self.benchmark_results: List[ModelPerformanceMetrics] = []
        
    def _calculate_model_size(self) -> float:
        """Calculate model size in MB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        total_size_bytes = total_params * 4  # Assuming float32
        return total_size_bytes / (1024 * 1024)
    
    def benchmark_inference(self, 
                           input_data: torch.Tensor,
                           num_runs: int = 100,
                           warmup_runs: int = 10) -> ModelPerformanceMetrics:
        """Benchmark model inference performance."""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_data.to(self.device))
        
        # Benchmark
        latencies = []
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                iter_start = time.time()
                _ = self.model(input_data.to(self.device))
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                iter_end = time.time()
                latencies.append((iter_end - iter_start) * 1000)  # Convert to ms
        
        total_time = time.time() - start_time
        num_samples = input_data.shape[0] if input_data.dim() > 0 else 1
        
        # Calculate metrics
        latencies = sorted(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        results = ModelPerformanceMetrics(
            avg_latency_ms=np.mean(latencies),
            median_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            latency_std_ms=np.std(latencies),
            samples_per_second=(num_runs * num_samples) / total_time,
            model_size_mb=self.model_size_mb,
            batch_size=num_samples,
            sequence_length=input_data.shape[-1] if input_data.dim() > 1 else 0
        )
        
        self.benchmark_results.append(results)
        return results
    
    def profile_memory_usage(self, input_data: torch.Tensor) -> Dict[str, float]:
        """Profile memory usage during inference."""
        if not torch.cuda.is_available():
            return {}
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        baseline_memory = torch.cuda.memory_allocated()
        
        # Run inference
        with torch.no_grad():
            _ = self.model(input_data.to(self.device))
        
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            'baseline_memory_mb': baseline_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'memory_delta_mb': (peak_memory - baseline_memory) / 1024 / 1024
        }
    
    def analyze_quantization_impact(self, 
                                  original_model: torch.nn.Module,
                                  quantized_model: torch.nn.Module,
                                  input_data: torch.Tensor) -> Dict[str, float]:
        """Analyze the impact of quantization on model performance."""
        
        # Benchmark original model
        original_metrics = self.benchmark_inference(input_data, num_runs=50, warmup_runs=5)
        
        # Benchmark quantized model
        self.model = quantized_model
        quantized_metrics = self.benchmark_inference(input_data, num_runs=50, warmup_runs=5)
        
        # Calculate impact
        size_reduction = (original_metrics.model_size_mb - quantized_metrics.model_size_mb) / original_metrics.model_size_mb
        latency_improvement = (original_metrics.avg_latency_ms - quantized_metrics.avg_latency_ms) / original_metrics.avg_latency_ms
        
        return {
            'size_reduction_ratio': size_reduction,
            'latency_improvement_ratio': latency_improvement,
            'original_size_mb': original_metrics.model_size_mb,
            'quantized_size_mb': quantized_metrics.model_size_mb,
            'original_latency_ms': original_metrics.avg_latency_ms,
            'quantized_latency_ms': quantized_metrics.avg_latency_ms,
            'throughput_improvement_ratio': quantized_metrics.samples_per_second / original_metrics.samples_per_second
        }


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, 
                 save_dir: str = "./monitoring_logs",
                 tensorboard_dir: Optional[str] = None,
                 monitor_system: bool = True,
                 track_gradients: bool = True):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring components
        self.system_monitor = SystemMonitor() if monitor_system else None
        self.gradient_monitor = GradientMonitor() if track_gradients else None
        self.model_profiler: Optional[ModelProfiler] = None
        
        # TensorBoard logging
        self.tensorboard_writer: Optional[SummaryWriter] = None
        if tensorboard_dir:
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)
        
        # Metrics storage
        self.training_history: List[TrainingMetrics] = []
        self.system_history: List[SystemMetrics] = []
        self.model_performance_history: List[ModelPerformanceMetrics] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0,
            'temperature': 80.0,
            'grad_norm_exploded': 10.0,
            'learning_rate_zero': 1e-10
        }
        
        # Current training state
        self.current_epoch = 0
        self.current_step = 0
        self.current_phase = "train"
        
        logger.info(f"PerformanceMonitor initialized. Logs saved to: {self.save_dir}")
    
    def set_model(self, model: torch.nn.Module, device: torch.device):
        """Set model for profiling."""
        self.model_profiler = ModelProfiler(model, device)
    
    def start_monitoring(self):
        """Start all monitoring components."""
        if self.system_monitor:
            self.system_monitor.start_monitoring()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info("Performance monitoring stopped")
    
    def log_training_step(self, 
                         metrics: TrainingMetrics,
                         custom_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Log training step metrics."""
        
        # Add timestamp
        metrics.timestamp = time.time()
        
        # Update current state
        self.current_epoch = metrics.epoch
        self.current_step = metrics.step
        self.current_phase = metrics.phase
        
        # Collect system metrics
        system_metrics = None
        if self.system_monitor:
            system_metrics = self.system_monitor.get_latest_metrics()
            if system_metrics:
                self.system_history.append(system_metrics)
        
        # Monitor gradients if configured
        grad_metrics = {}
        if self.gradient_monitor and metrics.grad_norm > 0:
            grad_metrics = self.gradient_monitor.monitor_step(metrics.get('model', None))
        
        # Combine metrics
        combined_metrics = {
            'training': metrics.to_dict(),
            'system': system_metrics.to_dict() if system_metrics else {},
            'gradients': grad_metrics,
            'custom': custom_metrics or {}
        }
        
        # Check for alerts
        alerts = self._check_alerts(metrics, system_metrics, grad_metrics)
        if alerts:
            logger.warning(f"Performance alerts: {alerts}")
        
        # Store in history
        self.training_history.append(metrics)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self._log_to_tensorboard(metrics, system_metrics, grad_metrics)
        
        # Auto-save periodically
        if metrics.step % 100 == 0:
            self.save_metrics()
        
        return {
            'metrics': combined_metrics,
            'alerts': alerts
        }
    
    def _check_alerts(self, 
                     training_metrics: TrainingMetrics,
                     system_metrics: Optional[SystemMetrics],
                     grad_metrics: Dict[str, float]) -> List[str]:
        """Check for performance alerts."""
        alerts = []
        
        # Check training metrics
        if training_metrics.grad_norm > self.alert_thresholds['grad_norm_exploded']:
            alerts.append(f"Gradient norm exploded: {training_metrics.grad_norm:.2f}")
        
        if training_metrics.learning_rate < self.alert_thresholds['learning_rate_zero']:
            alerts.append(f"Learning rate too low: {training_metrics.learning_rate:.2e}")
        
        # Check system metrics
        if system_metrics:
            if system_metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage']:
                alerts.append(f"High CPU usage: {system_metrics.cpu_usage_percent:.1f}%")
            
            if system_metrics.memory_usage_percent > self.alert_thresholds['memory_usage']:
                alerts.append(f"High memory usage: {system_metrics.memory_usage_percent:.1f}%")
            
            if system_metrics.gpu_memory_utilization > self.alert_thresholds['gpu_memory_usage']:
                alerts.append(f"High GPU memory usage: {system_metrics.gpu_memory_utilization:.1f}%")
            
            if system_metrics.gpu_temperature > self.alert_thresholds['temperature']:
                alerts.append(f"High GPU temperature: {system_metrics.gpu_temperature:.1f}°C")
        
        return alerts
    
    def _log_to_tensorboard(self, 
                           training_metrics: TrainingMetrics,
                           system_metrics: Optional[SystemMetrics],
                           grad_metrics: Dict[str, float]):
        """Log metrics to TensorBoard."""
        if not self.tensorboard_writer:
            return
        
        step = training_metrics.global_step
        
        # Training metrics
        self.tensorboard_writer.add_scalar('Loss/train', training_metrics.loss, step)
        if training_metrics.accuracy > 0:
            self.tensorboard_writer.add_scalar('Accuracy/train', training_metrics.accuracy, step)
        
        self.tensorboard_writer.add_scalar('Learning_Rate', training_metrics.learning_rate, step)
        
        # Gradient metrics
        if grad_metrics:
            self.tensorboard_writer.add_scalar('Gradient_Norm/total', grad_metrics.get('grad_norm', 0), step)
        
        # System metrics
        if system_metrics:
            self.tensorboard_writer.add_scalar('System/CPU_Usage', system_metrics.cpu_usage_percent, step)
            self.tensorboard_writer.add_scalar('System/Memory_Usage', system_metrics.memory_usage_percent, step)
            if system_metrics.gpu_memory_utilization > 0:
                self.tensorboard_writer.add_scalar('System/GPU_Memory_Usage', system_metrics.gpu_memory_utilization, step)
        
        # Timing metrics
        if training_metrics.batch_time > 0:
            self.tensorboard_writer.add_scalar('Timing/Batch_Time', training_metrics.batch_time * 1000, step)
    
    def save_metrics(self):
        """Save all collected metrics to disk."""
        timestamp = int(time.time())
        
        # Save training metrics
        training_file = self.save_dir / f"training_metrics_{timestamp}.json"
        with open(training_file, 'w') as f:
            json.dump([m.to_dict() for m in self.training_history], f, indent=2)
        
        # Save system metrics
        if self.system_history:
            system_file = self.save_dir / f"system_metrics_{timestamp}.json"
            with open(system_file, 'w') as f:
                json.dump([m.to_dict() for m in self.system_history], f, indent=2)
        
        # Save model performance metrics
        if self.model_performance_history:
            performance_file = self.save_dir / f"model_performance_{timestamp}.json"
            with open(performance_file, 'w') as f:
                json.dump([m.to_dict() for m in self.model_performance_history], f, indent=2)
        
        logger.info(f"Metrics saved to {self.save_dir}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected performance metrics."""
        summary = {
            'monitoring_duration': 0,
            'training_steps': len(self.training_history),
            'system_samples': len(self.system_history),
            'model_performance_tests': len(self.model_performance_history),
            'latest_training_metrics': None,
            'latest_system_metrics': None,
            'gradient_stats': None,
            'performance_recommendations': []
        }
        
        if self.training_history:
            latest_training = self.training_history[-1]
            summary['latest_training_metrics'] = latest_training.to_dict()
            summary['monitoring_duration'] = latest_training.timestamp - (self.training_history[0].timestamp if self.training_history else latest_training.timestamp)
        
        if self.system_history:
            summary['latest_system_metrics'] = self.system_history[-1].to_dict()
        
        if self.gradient_monitor:
            summary['gradient_stats'] = self.gradient_monitor.get_gradient_stats()
        
        # Generate recommendations
        summary['performance_recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.system_history:
            return ["Insufficient data for recommendations"]
        
        recent_system = self.system_history[-10:] if len(self.system_history) >= 10 else self.system_history
        
        avg_cpu = np.mean([s.cpu_usage_percent for s in recent_system])
        avg_memory = np.mean([s.memory_usage_percent for s in recent_system])
        avg_gpu_mem = np.mean([s.gpu_memory_utilization for s in recent_system if s.gpu_memory_utilization > 0])
        
        if avg_cpu < 50:
            recommendations.append("CPU utilization is low. Consider increasing batch size or using more workers.")
        elif avg_cpu > 90:
            recommendations.append("High CPU usage detected. Consider reducing batch size or optimizing data loading.")
        
        if avg_memory > 80:
            recommendations.append("High memory usage. Consider gradient accumulation or reducing batch size.")
        
        if avg_gpu_mem > 80:
            recommendations.append("High GPU memory usage. Consider model parallelism or gradient checkpointing.")
        
        if self.gradient_monitor and self.gradient_monitor.gradient_stats:
            avg_grad_norm = self.gradient_monitor.gradient_stats.get('avg_grad_norm', 0)
            if avg_grad_norm > 5.0:
                recommendations.append("High gradient norms detected. Consider gradient clipping or reducing learning rate.")
            elif avg_grad_norm < 0.001:
                recommendations.append("Very low gradient norms. Consider increasing learning rate or checking model initialization.")
        
        return recommendations


@contextmanager
def monitor_training_step(monitor: PerformanceMonitor, phase: str = "train"):
    """Context manager for monitoring individual training steps."""
    
    start_time = time.time()
    
    # System metrics before
    system_before = monitor.system_monitor.get_latest_metrics() if monitor.system_monitor else None
    
    # Memory before
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    try:
        yield
    finally:
        # Calculate timing
        batch_time = time.time() - start_time
        
        # Memory after
        gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        # System metrics after
        system_after = monitor.system_monitor.get_latest_metrics() if monitor.system_monitor else None
        
        # Update metrics
        timing_metrics = TrainingMetrics(
            batch_time=batch_time,
            cpu_memory_mb=system_before.process_memory_mb if system_before else 0,
            gpu_memory_mb=gpu_memory,
            phase=phase
        )
        
        if system_before and system_after:
            # Calculate data load time (approximation)
            timing_metrics.data_load_time = max(0, batch_time - (system_after.cpu_usage_percent - system_before.cpu_usage_percent) * 0.01)
        
        monitor.log_training_step(timing_metrics)


# Utility functions
def create_performance_monitor(save_dir: str = "./monitoring_logs",
                             tensorboard_dir: Optional[str] = None,
                             monitor_system: bool = True,
                             track_gradients: bool = True) -> PerformanceMonitor:
    """Create and initialize a performance monitor."""
    return PerformanceMonitor(
        save_dir=save_dir,
        tensorboard_dir=tensorboard_dir,
        monitor_system=monitor_system,
        track_gradients=track_gradients
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitor Demo")
    parser.add_argument("--monitor_system", action="store_true", help="Monitor system resources")
    parser.add_argument("--tensorboard_dir", type=str, help="TensorBoard log directory")
    parser.add_argument("--save_dir", type=str, default="./monitoring_logs", help="Save directory")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = create_performance_monitor(
        save_dir=args.save_dir,
        tensorboard_dir=args.tensorboard_dir,
        monitor_system=args.monitor_system
    )
    
    try:
        monitor.start_monitoring()
        
        # Simulate training steps
        for step in range(100):
            metrics = TrainingMetrics(
                epoch=1,
                step=step,
                global_step=step,
                loss=2.0 - step * 0.02 + np.random.normal(0, 0.1),
                accuracy=0.1 + step * 0.008 + np.random.normal(0, 0.05),
                learning_rate=1e-4 * (0.95 ** (step // 10)),
                grad_norm=np.random.uniform(0.1, 5.0),
                phase="train"
            )
            
            result = monitor.log_training_step(metrics)
            
            if step % 20 == 0:
                print(f"Step {step}: Loss={metrics.loss:.3f}, LR={metrics.learning_rate:.2e}")
                if result['alerts']:
                    print(f"  Alerts: {result['alerts']}")
            
            time.sleep(0.1)
        
        # Print summary
        summary = monitor.get_performance_summary()
        print(f"\nMonitoring Summary:")
        print(f"Training steps: {summary['training_steps']}")
        print(f"System samples: {summary['system_samples']}")
        print(f"Recommendations: {summary['performance_recommendations']}")
        
    finally:
        monitor.stop_monitoring()
        monitor.save_metrics()