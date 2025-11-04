"""
Core Metrics Collection System

Provides comprehensive real-time metrics collection for inference, system resources,
model performance, and medical outcome tracking with Prometheus integration.
"""

import asyncio
import json
import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
import structlog

import torch
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest

logger = structlog.get_logger("metrics_collector")


@dataclass
class InferenceMetrics:
    """Real-time inference metrics."""
    model_id: str
    request_id: str
    
    # Latency metrics
    total_latency_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0
    
    # Throughput metrics
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Success metrics
    status_code: int = 200
    error_message: Optional[str] = None
    cache_hit: bool = False
    
    # Quality metrics
    confidence_score: float = 0.0
    clinical_relevance_score: float = 0.0
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    clinical_context: Optional[Dict[str, Any]] = None


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float = field(default_factory=time.time)
    uptime_seconds: float = 0.0
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_frequency_mhz: float = 0.0
    cpu_temperature_c: Optional[float] = None
    load_average_1m: float = 0.0
    
    # Memory metrics
    memory_total_gb: float = 0.0
    memory_available_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_usage_percent: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_usage_percent: float = 0.0
    
    # GPU metrics
    gpu_available: bool = False
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_usage_percent: float = 0.0
    gpu_utilization_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_watts: float = 0.0
    
    # Disk metrics
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_read_mb_s: float = 0.0
    disk_write_mb_s: float = 0.0
    
    # Network metrics
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    network_connections: int = 0
    
    # Process metrics
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_thread_count: int = 0
    
    # Container metrics (if available)
    container_cpu_percent: Optional[float] = None
    container_memory_mb: Optional[float] = None


@dataclass
class ModelMetrics:
    """Model-specific performance metrics."""
    model_id: str
    model_version: str
    model_type: str
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    
    # Accuracy metrics
    accuracy_score: float = 0.0
    precision_score: float = 0.0
    recall_score: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    
    # Clinical metrics
    clinical_effectiveness_score: float = 0.0
    medical_relevance_score: float = 0.0
    safety_score: float = 0.0
    bias_score: float = 0.0
    
    # Drift detection
    data_drift_score: float = 0.0
    concept_drift_score: float = 0.0
    performance_drift_score: float = 0.0
    
    # Resource metrics
    memory_usage_mb: float = 0.0
    gpu_memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Requests metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    requests_rate_per_minute: float = 0.0
    
    # Model metadata
    model_size_mb: float = 0.0
    parameters_count: int = 0
    last_updated: float = field(default_factory=time.time)
    timestamp: float = field(default_factory=time.time)


class SLATracker:
    """SLA (Service Level Agreement) tracking for medical AI systems."""
    
    def __init__(self):
        self.sla_thresholds = {
            "response_time_ms": 2000,  # 2 seconds
            "availability_percent": 99.9,
            "accuracy_score": 0.95,
            "clinical_effectiveness": 0.90,
            "error_rate": 0.01
        }
        
        self.violations = deque(maxlen=1000)
        self.performance_history = deque(maxlen=10000)
        
    def check_sla_violation(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if metrics violate SLA thresholds."""
        violations = {}
        
        for metric, value in metrics.items():
            threshold = self.sla_thresholds.get(metric)
            if threshold is not None:
                # For "higher is better" metrics
                if metric in ["accuracy_score", "clinical_effectiveness", "availability_percent"]:
                    violations[metric] = value < threshold
                # For "lower is better" metrics
                else:
                    violations[metric] = value > threshold
                
                if violations[metric]:
                    self.violations.append({
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "timestamp": time.time()
                    })
        
        return violations


class MetricsCollector:
    """Comprehensive metrics collection service."""
    
    def __init__(self, 
                 collection_interval: float = 1.0,
                 retention_days: int = 30,
                 enable_prometheus: bool = True,
                 prometheus_registry: Optional[CollectorRegistry] = None,
                 save_directory: str = "./monitoring_data"):
        
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        self.enable_prometheus = enable_prometheus
        self.prometheus_registry = prometheus_registry or CollectorRegistry()
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        
        self.logger = structlog.get_logger("metrics_collector")
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        
        # Metrics storage
        self.inference_metrics: deque = deque(maxlen=100000)
        self.system_metrics_history: deque = deque(maxlen=10000)
        self.model_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        
        # Performance aggregators
        self.latency_aggregator = LatencyAggregator()
        self.throughput_aggregator = ThroughputAggregator()
        self.error_aggregator = ErrorAggregator()
        
        # SLA tracking
        self.sla_tracker = SLATracker()
        
        # Thread pool for system metrics collection
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Custom metrics hooks
        self.custom_metrics_hooks: List[Callable] = []
        
        # Initialize Prometheus metrics if enabled
        if self.enable_prometheus:
            self._init_prometheus_metrics()
            
        self.logger.info("MetricsCollector initialized", 
                        collection_interval=collection_interval,
                        retention_days=retention_days,
                        enable_prometheus=enable_prometheus)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, Info
            
            # Inference metrics
            self.prom_inference_requests_total = Counter(
                'medical_ai_inference_requests_total',
                'Total number of inference requests',
                ['model_id', 'status_code'],
                registry=self.prometheus_registry
            )
            
            self.prom_inference_duration_seconds = Histogram(
                'medical_ai_inference_duration_seconds',
                'Inference request duration',
                ['model_id'],
                registry=self.prometheus_registry
            )
            
            self.prom_inference_tokens_total = Counter(
                'medical_ai_inference_tokens_total',
                'Total tokens processed',
                ['model_id', 'type'],  # type: input/output
                registry=self.prometheus_registry
            )
            
            # System metrics
            self.prom_cpu_usage_percent = Gauge(
                'medical_ai_system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.prometheus_registry
            )
            
            self.prom_memory_usage_percent = Gauge(
                'medical_ai_system_memory_usage_percent',
                'System memory usage percentage',
                registry=self.prometheus_registry
            )
            
            self.prom_gpu_memory_usage_percent = Gauge(
                'medical_ai_system_gpu_memory_usage_percent',
                'System GPU memory usage percentage',
                registry=self.prometheus_registry
            )
            
            # Model accuracy metrics
            self.prom_model_accuracy = Gauge(
                'medical_ai_model_accuracy_score',
                'Model accuracy score',
                ['model_id'],
                registry=self.prometheus_registry
            )
            
            self.prom_model_clinical_effectiveness = Gauge(
                'medical_ai_model_clinical_effectiveness_score',
                'Model clinical effectiveness score',
                ['model_id'],
                registry=self.prometheus_registry
            )
            
            # SLA metrics
            self.prom_sla_violations_total = Counter(
                'medical_ai_sla_violations_total',
                'Total SLA violations',
                ['sla_type'],
                registry=self.prometheus_registry
            )
            
            # Model info
            self.prom_model_info = Info(
                'medical_ai_model_info',
                'Model information',
                ['model_id', 'model_type', 'version'],
                registry=self.prometheus_registry
            )
            
        except ImportError:
            self.logger.warning("Prometheus client not available, metrics collection disabled")
            self.enable_prometheus = False
    
    def add_inference_metric(self, metrics: InferenceMetrics):
        """Add inference metrics to collection."""
        try:
            self.inference_metrics.append(metrics)
            
            # Update aggregators
            self.latency_aggregator.add_latency(metrics.model_id, metrics.total_latency_ms)
            self.throughput_aggregator.add_request(metrics.model_id)
            
            if metrics.status_code >= 400:
                self.error_aggregator.add_error(metrics.model_id, metrics.status_code)
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self._update_prometheus_metrics(metrics)
            
            self.logger.debug("Inference metric recorded", 
                            model_id=metrics.model_id,
                            latency_ms=metrics.total_latency_ms,
                            status_code=metrics.status_code)
                            
        except Exception as e:
            self.logger.error("Failed to record inference metric", error=str(e))
    
    def _update_prometheus_metrics(self, metrics: InferenceMetrics):
        """Update Prometheus metrics."""
        try:
            # Request counts
            self.prom_inference_requests_total.labels(
                model_id=metrics.model_id,
                status_code=metrics.status_code
            ).inc()
            
            # Duration
            self.prom_inference_duration_seconds.labels(
                model_id=metrics.model_id
            ).observe(metrics.total_latency_ms / 1000.0)
            
            # Token counts
            if metrics.input_tokens > 0:
                self.prom_inference_tokens_total.labels(
                    model_id=metrics.model_id,
                    type='input'
                ).inc(metrics.input_tokens)
            
            if metrics.output_tokens > 0:
                self.prom_inference_tokens_total.labels(
                    model_id=metrics.model_id,
                    type='output'
                ).inc(metrics.output_tokens)
                
        except Exception as e:
            self.logger.warning("Failed to update Prometheus metrics", error=str(e))
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        loop = asyncio.get_event_loop()
        
        try:
            # Run in thread pool to avoid blocking
            system_metrics = await loop.run_in_executor(
                self.thread_pool, 
                self._collect_system_metrics_sync
            )
            
            self.system_metrics_history.append(system_metrics)
            
            # Update Prometheus system metrics
            if self.enable_prometheus:
                self.prom_cpu_usage_percent.set(system_metrics.cpu_percent)
                self.prom_memory_usage_percent.set(system_metrics.memory_usage_percent)
                if system_metrics.gpu_available:
                    self.prom_gpu_memory_usage_percent.set(system_metrics.gpu_memory_usage_percent)
            
            return system_metrics
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
            raise
    
    def _collect_system_metrics_sync(self) -> SystemMetrics:
        """Synchronous system metrics collection."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
        
        # Get CPU temperature
        cpu_temp = None
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                cpu_temp = temps['cpu_thermal'][0].current
        except:
            pass
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # GPU metrics
        gpu_available = torch.cuda.is_available()
        gpu_memory_total_gb = 0.0
        gpu_memory_used_gb = 0.0
        gpu_memory_usage_percent = 0.0
        gpu_utilization_percent = 0.0
        gpu_temperature_c = 0.0
        gpu_power_watts = 0.0
        
        if gpu_available:
            try:
                # Use torch for basic GPU metrics
                gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_usage_percent = (gpu_memory_used_gb / gpu_memory_total_gb) * 100
                
                # Try to get additional GPU metrics using nvidia-ml-py
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization_percent = util.gpu
                    
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature_c = temp
                    
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                        gpu_power_watts = power
                    except:
                        pass
                    
                    pynvml.nvmlShutdown()
                except ImportError:
                    pass  # nvidia-ml-py not available
                    
            except Exception as e:
                self.logger.warning("Failed to collect GPU metrics", error=str(e))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        network_connections = len(psutil.net_connections())
        
        # Process metrics
        process = psutil.Process()
        process_memory_mb = process.memory_info().rss / (1024 * 1024)
        process_cpu_percent = process.cpu_percent()
        process_thread_count = process.num_threads()
        
        # Calculate uptime
        uptime_seconds = time.time() - getattr(self, '_start_time', time.time())
        
        return SystemMetrics(
            uptime_seconds=uptime_seconds,
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_frequency_mhz=cpu_freq.current if cpu_freq else 0.0,
            cpu_temperature_c=cpu_temp,
            load_average_1m=load_avg,
            memory_total_gb=memory.total / (1024**3),
            memory_available_gb=memory.available / (1024**3),
            memory_used_gb=memory.used / (1024**3),
            memory_usage_percent=memory.percent,
            swap_total_gb=swap.total / (1024**3),
            swap_used_gb=swap.used / (1024**3),
            swap_usage_percent=swap.percent,
            gpu_available=gpu_available,
            gpu_count=torch.cuda.device_count() if gpu_available else 0,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_usage_percent=gpu_memory_usage_percent,
            gpu_utilization_percent=gpu_utilization_percent,
            gpu_temperature_c=gpu_temperature_c,
            gpu_power_watts=gpu_power_watts,
            disk_total_gb=disk.total / (1024**3),
            disk_free_gb=disk.free / (1024**3),
            disk_used_gb=disk.used / (1024**3),
            disk_usage_percent=(disk.used / disk.total) * 100,
            disk_read_mb_s=disk_io.read_bytes / (1024 * 1024) if disk_io else 0.0,
            disk_write_mb_s=disk_io.write_bytes / (1024 * 1024) if disk_io else 0.0,
            network_bytes_sent=network_io.bytes_sent if network_io else 0,
            network_bytes_recv=network_io.bytes_recv if network_io else 0,
            network_packets_sent=network_io.packets_sent if network_io else 0,
            network_packets_recv=network_io.packets_recv if network_io else 0,
            network_connections=network_connections,
            process_cpu_percent=process_cpu_percent,
            process_memory_mb=process_memory_mb,
            process_thread_count=process_thread_count
        )
    
    async def start_collection(self):
        """Start the metrics collection service."""
        if self.is_running:
            self.logger.warning("Metrics collection already running")
            return
        
        self.is_running = True
        self._start_time = time.time()
        
        self.logger.info("Starting metrics collection service")
        
        # Start the collection task
        self.collection_task = asyncio.create_task(self._collection_loop())
        
        self.logger.info("Metrics collection service started")
    
    async def stop_collection(self):
        """Stop the metrics collection service."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        self.thread_pool.shutdown(wait=True)
        
        # Save current metrics
        await self.save_metrics()
        
        self.logger.info("Metrics collection service stopped")
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.is_running:
            try:
                await self.collect_system_metrics()
                
                # Run custom metrics hooks
                for hook in self.custom_metrics_hooks:
                    try:
                        await hook()
                    except Exception as e:
                        self.logger.warning("Custom metrics hook failed", error=str(e))
                
                # Auto-save metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    await self.save_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    async def save_metrics(self):
        """Save metrics to disk."""
        try:
            timestamp = int(time.time())
            
            # Save system metrics
            if self.system_metrics_history:
                system_file = self.save_directory / f"system_metrics_{timestamp}.json"
                with open(system_file, 'w') as f:
                    json.dump([asdict(m) for m in self.system_metrics_history], f)
            
            # Save inference metrics (sample for large datasets)
            if len(self.inference_metrics) > 1000:
                inference_file = self.save_directory / f"inference_metrics_sample_{timestamp}.json"
                sample_metrics = list(self.inference_metrics)[-1000:]  # Last 1000
                with open(inference_file, 'w') as f:
                    json.dump([asdict(m) for m in sample_metrics], f)
            
            # Save model metrics
            for model_id, metrics_history in self.model_metrics_history.items():
                if metrics_history:
                    model_file = self.save_directory / f"model_metrics_{model_id}_{timestamp}.json"
                    with open(model_file, 'w') as f:
                        json.dump([asdict(m) for m in metrics_history], f)
            
            self.logger.debug("Metrics saved to disk", 
                            timestamp=timestamp,
                            system_metrics_count=len(self.system_metrics_history),
                            inference_metrics_count=len(self.inference_metrics))
            
        except Exception as e:
            self.logger.error("Failed to save metrics", error=str(e))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        try:
            # Calculate recent statistics
            now = time.time()
            last_5_minutes = [m for m in self.inference_metrics if now - m.timestamp <= 300]
            last_1_hour = [m for m in self.inference_metrics if now - m.timestamp <= 3600]
            
            # Recent system metrics
            recent_system = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            # Model summaries
            model_summaries = {}
            for model_id, history in self.model_metrics_history.items():
                if history:
                    recent_model_metrics = [m for m in history if now - m.timestamp <= 3600]
                    if recent_model_metrics:
                        model_summaries[model_id] = asdict(recent_model_metrics[-1])
            
            return {
                "timestamp": now,
                "uptime_seconds": now - getattr(self, '_start_time', now),
                "inference_metrics": {
                    "total_count": len(self.inference_metrics),
                    "recent_5min_count": len(last_5_minutes),
                    "recent_1h_count": len(last_1_hour),
                    "avg_latency_5min": np.mean([m.total_latency_ms for m in last_5_minutes]) if last_5_minutes else 0,
                    "avg_latency_1h": np.mean([m.total_latency_ms for m in last_1_hour]) if last_1_hour else 0,
                    "error_rate_5min": len([m for m in last_5_minutes if m.status_code >= 400]) / len(last_5_minutes) if last_5_minutes else 0
                },
                "system_metrics": asdict(recent_system) if recent_system else {},
                "model_summaries": model_summaries,
                "sla_status": self.sla_tracker.get_sla_status(),
                "collection_interval": self.collection_interval,
                "storage_usage": self._get_storage_usage()
            }
            
        except Exception as e:
            self.logger.error("Failed to get metrics summary", error=str(e))
            return {"error": str(e)}
    
    def _get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage information."""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.save_directory.glob("**/*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            return {
                "total_size_bytes": total_size,
                "total_size_gb": total_size / (1024**3),
                "file_count": file_count
            }
        except Exception as e:
            self.logger.warning("Failed to get storage usage", error=str(e))
            return {}


class LatencyAggregator:
    """Aggregates latency metrics for analysis."""
    
    def __init__(self):
        self.model_latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
    
    def add_latency(self, model_id: str, latency_ms: float):
        """Add latency measurement."""
        self.model_latencies[model_id].append({
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
    
    def get_percentiles(self, model_id: str, percentiles: List[float] = [50, 95, 99]) -> Dict[str, float]:
        """Get latency percentiles for a model."""
        if model_id not in self.model_latencies:
            return {f'p{p}': 0.0 for p in percentiles}
        
        latencies = [m['latency_ms'] for m in self.model_latencies[model_id]]
        if not latencies:
            return {f'p{p}': 0.0 for p in percentiles}
        
        latencies.sort()
        return {
            f'p{int(p*100)}': np.percentile(latencies, p * 100)
            for p in percentiles
        }


class ThroughputAggregator:
    """Aggregates throughput metrics."""
    
    def __init__(self):
        self.request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
    
    def add_request(self, model_id: str):
        """Add request timestamp."""
        self.request_times[model_id].append(time.time())
    
    def get_throughput(self, model_id: str, window_seconds: int = 60) -> float:
        """Get requests per second for a model."""
        if model_id not in self.request_times:
            return 0.0
        
        now = time.time()
        recent_requests = [
            t for t in self.request_times[model_id] 
            if now - t <= window_seconds
        ]
        
        return len(recent_requests) / window_seconds


class ErrorAggregator:
    """Aggregates error metrics."""
    
    def __init__(self):
        self.error_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.total_counts: Dict[str, int] = defaultdict(int)
    
    def add_error(self, model_id: str, status_code: int):
        """Add error count."""
        self.error_counts[model_id][status_code] += 1
        self.total_counts[model_id] += 1
    
    def get_error_rate(self, model_id: str) -> float:
        """Get error rate for a model."""
        if model_id not in self.error_counts:
            return 0.0
        
        total = self.total_counts[model_id]
        if total == 0:
            return 0.0
        
        errors = sum(self.error_counts[model_id].values())
        return errors / total