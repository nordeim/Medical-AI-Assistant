"""
Performance Optimization Recommendation Engine

This module analyzes collected metrics and provides actionable recommendations
for optimizing machine learning training performance.
"""

import json
import logging
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union

from .performance_monitor import TrainingMetrics, SystemMetrics, ModelPerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRecommendation:
    """Container for optimization recommendations."""
    
    category: str  # training, system, model, resource
    priority: str  # high, medium, low
    title: str
    description: str
    rationale: str
    impact_estimate: str
    implementation_effort: str  # low, medium, high
    code_examples: Optional[List[str]] = None
    config_changes: Optional[Dict[str, Any]] = None
    metrics_affected: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'rationale': self.rationale,
            'impact_estimate': self.impact_estimate,
            'implementation_effort': self.implementation_effort,
            'code_examples': self.code_examples or [],
            'config_changes': self.config_changes or {},
            'metrics_affected': self.metrics_affected or []
        }


class PerformanceAnalyzer:
    """Analyze performance metrics and generate optimization recommendations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.analysis_cache: Dict[str, Any] = {}
        
    def analyze_performance(self,
                          training_metrics: List[TrainingMetrics],
                          system_metrics: List[SystemMetrics],
                          model_metrics: List[ModelPerformanceMetrics],
                          time_window_hours: float = 24.0) -> List[OptimizationRecommendation]:
        """Analyze performance and generate recommendations."""
        
        recommendations = []
        
        # Analyze training performance
        training_recs = self._analyze_training_performance(training_metrics, time_window_hours)
        recommendations.extend(training_recs)
        
        # Analyze system performance
        system_recs = self._analyze_system_performance(system_metrics, time_window_hours)
        recommendations.extend(system_recs)
        
        # Analyze model performance
        model_recs = self._analyze_model_performance(model_metrics, time_window_hours)
        recommendations.extend(model_recs)
        
        # Analyze resource utilization patterns
        resource_recs = self._analyze_resource_utilization(system_metrics, training_metrics)
        recommendations.extend(resource_recs)
        
        # Analyze convergence patterns
        convergence_recs = self._analyze_convergence_patterns(training_metrics)
        recommendations.extend(convergence_recs)
        
        # Analyze bottlenecks
        bottleneck_recs = self._analyze_bottlenecks(training_metrics, system_metrics)
        recommendations.extend(bottleneck_recs)
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return recommendations
    
    def _analyze_training_performance(self, 
                                    training_metrics: List[TrainingMetrics],
                                    time_window_hours: float) -> List[OptimizationRecommendation]:
        """Analyze training-specific performance issues."""
        
        recommendations = []
        
        if not training_metrics:
            return recommendations
        
        # Filter recent metrics
        current_time = training_metrics[-1].timestamp if training_metrics else 0
        start_time = current_time - (time_window_hours * 3600)
        recent_metrics = [m for m in training_metrics if m.timestamp >= start_time]
        
        if not recent_metrics:
            return recommendations
        
        # Analyze loss progression
        loss_values = [m.loss for m in recent_metrics if m.loss > 0]
        if len(loss_values) > 10:
            # Check for training instability
            loss_variance = np.var(loss_values)
            loss_mean = np.mean(loss_values)
            
            if loss_variance > loss_mean * 0.5:
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="high",
                    title="Training Instability Detected",
                    description="High loss variance indicates training instability",
                    rationale=f"Loss variance ({loss_variance:.3f}) is high relative to mean ({loss_mean:.3f})",
                    impact_estimate="15-30% improvement in convergence stability",
                    implementation_effort="low",
                    code_examples=[
                        "# Reduce learning rate",
                        "optimizer.param_groups[0]['lr'] *= 0.5",
                        "",
                        "# Add gradient clipping",
                        "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
                    ],
                    config_changes={
                        "learning_rate": "reduce by 50%",
                        "gradient_clipping": "max_norm=1.0"
                    },
                    metrics_affected=["loss", "grad_norm"]
                ))
            
            # Check for slow convergence
            recent_loss = np.mean(loss_values[-10:]) if len(loss_values) >= 10 else loss_values[-1]
            initial_loss = np.mean(loss_values[:10]) if len(loss_values) >= 10 else loss_values[0]
            
            improvement_rate = (initial_loss - recent_loss) / initial_loss if initial_loss > 0 else 0
            steps_taken = len(loss_values)
            expected_improvement = min(0.7, steps_taken / 100)  # 70% improvement expected in 100 steps
            
            if improvement_rate < expected_improvement * 0.5:
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="medium",
                    title="Slow Convergence Detected",
                    description="Training is converging slower than expected",
                    rationale=f"Only {improvement_rate:.1%} improvement after {steps_taken} steps",
                    impact_estimate="20-40% faster convergence",
                    implementation_effort="medium",
                    code_examples=[
                        "# Use learning rate schedule",
                        "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)",
                        "",
                        "# Add momentum",
                        "optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)"
                    ],
                    config_changes={
                        "learning_rate_schedule": "StepLR with gamma=0.1",
                        "optimizer": "SGD with momentum=0.9"
                    },
                    metrics_affected=["loss", "learning_rate"]
                ))
        
        # Analyze gradient patterns
        grad_norms = [m.grad_norm for m in recent_metrics if m.grad_norm > 0]
        if grad_norms:
            avg_grad_norm = np.mean(grad_norms)
            max_grad_norm = max(grad_norms)
            
            if avg_grad_norm > 5.0:
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="high",
                    title="High Gradient Norms",
                    description="Gradients are consistently high, indicating potential instability",
                    rationale=f"Average gradient norm: {avg_grad_norm:.2f}, Max: {max_grad_norm:.2f}",
                    impact_estimate="25-50% improvement in training stability",
                    implementation_effort="low",
                    code_examples=[
                        "# Add gradient clipping",
                        "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)",
                        "",
                        "# Reduce learning rate",
                        "optimizer.param_groups[0]['lr'] *= 0.1"
                    ],
                    config_changes={
                        "gradient_clipping": "max_norm=1.0",
                        "learning_rate": "multiply by 0.1"
                    },
                    metrics_affected=["grad_norm", "loss"]
                ))
            
            elif avg_grad_norm < 0.001:
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="medium",
                    title="Very Low Gradient Norms",
                    description="Gradients are very small, indicating potential dead neurons",
                    rationale=f"Average gradient norm: {avg_grad_norm:.4f}",
                    impact_estimate="10-20% improvement in model capacity",
                    implementation_effort="medium",
                    code_examples=[
                        "# Increase learning rate",
                        "optimizer.param_groups[0]['lr'] *= 10",
                        "",
                        "# Check initialization",
                        "for p in model.parameters():",
                        "    if p.requires_grad:",
                        "        torch.nn.init.xavier_uniform_(p)"
                    ],
                    config_changes={
                        "learning_rate": "multiply by 10",
                        "weight_initialization": "xavier_uniform"
                    },
                    metrics_affected=["grad_norm", "param_norm"]
                ))
        
        return recommendations
    
    def _analyze_system_performance(self, 
                                  system_metrics: List[SystemMetrics],
                                  time_window_hours: float) -> List[OptimizationRecommendation]:
        """Analyze system resource performance."""
        
        recommendations = []
        
        if not system_metrics:
            return recommendations
        
        # Filter recent metrics
        current_time = system_metrics[-1].timestamp if system_metrics else 0
        start_time = current_time - (time_window_hours * 3600)
        recent_metrics = [m for m in system_metrics if m.timestamp >= start_time]
        
        if not recent_metrics:
            return recommendations
        
        # Analyze CPU usage
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        avg_cpu = np.mean(cpu_usage)
        
        if avg_cpu < 30:
            recommendations.append(OptimizationRecommendation(
                category="system",
                priority="medium",
                title="Low CPU Utilization",
                description="CPU is underutilized, suggesting potential for increased workload",
                rationale=f"Average CPU usage: {avg_cpu:.1f}%",
                impact_estimate="30-60% throughput improvement",
                implementation_effort="low",
                code_examples=[
                    "# Increase batch size",
                    "batch_size = batch_size * 2",
                    "",
                    "# Add more data loading workers",
                    "DataLoader(dataset, num_workers=4, pin_memory=True)"
                ],
                config_changes={
                    "batch_size": "increase by 2x",
                    "num_workers": "set to 4-8",
                    "pin_memory": "True"
                },
                metrics_affected=["throughput", "batch_time"]
            ))
        
        elif avg_cpu > 90:
            recommendations.append(OptimizationRecommendation(
                category="system",
                priority="high",
                title="High CPU Usage",
                description="CPU is heavily loaded, may bottleneck training",
                rationale=f"Average CPU usage: {avg_cpu:.1f}%",
                impact_estimate="10-25% performance improvement",
                implementation_effort="medium",
                code_examples=[
                    "# Reduce batch size",
                    "batch_size = batch_size // 2",
                    "",
                    "# Optimize data loading",
                    "DataLoader(dataset, num_workers=2, pin_memory=False)",
                    "",
                    "# Use mixed precision",
                    "scaler = torch.cuda.amp.GradScaler()"
                ],
                config_changes={
                    "batch_size": "reduce by 50%",
                    "num_workers": "reduce to 2",
                    "mixed_precision": "enabled"
                },
                metrics_affected=["batch_time", "throughput"]
            ))
        
        # Analyze memory usage
        memory_usage = [m.memory_usage_percent for m in recent_metrics]
        avg_memory = np.mean(memory_usage)
        max_memory = max(memory_usage)
        
        if avg_memory > 85:
            recommendations.append(OptimizationRecommendation(
                category="system",
                priority="high",
                title="High Memory Usage",
                description="System memory is heavily loaded, risk of OOM",
                rationale=f"Average memory usage: {avg_memory:.1f}%, Peak: {max_memory:.1f}%",
                impact_estimate="Prevent OOM errors, 5-15% stability improvement",
                implementation_effort="medium",
                code_examples=[
                    "# Use gradient accumulation",
                    "accumulation_steps = 4",
                    "optimizer.zero_grad()",
                    "for i, (data, target) in enumerate(dataloader):",
                    "    loss = model(data).mean()",
                    "    loss.backward()",
                    "    if (i + 1) % accumulation_steps == 0:",
                    "        optimizer.step()",
                    "        optimizer.zero_grad()"
                ],
                config_changes={
                    "gradient_accumulation_steps": "4-8",
                    "batch_size": "reduce by 50%"
                },
                metrics_affected=["memory_usage_percent"]
            ))
        
        # Analyze GPU utilization
        gpu_usage = [m.gpu_utilization for m in recent_metrics if m.gpu_utilization > 0]
        if gpu_usage:
            avg_gpu = np.mean(gpu_usage)
            
            if avg_gpu < 50:
                recommendations.append(OptimizationRecommendation(
                    category="system",
                    priority="medium",
                    title="Low GPU Utilization",
                    description="GPU is underutilized, potential for better hardware usage",
                    rationale=f"Average GPU utilization: {avg_gpu:.1f}%",
                    impact_estimate="50-100% training speed improvement",
                    implementation_effort="low",
                    code_examples=[
                        "# Increase batch size",
                        "batch_size = batch_size * 4",
                        "",
                        "# Use mixed precision training",
                        "with torch.cuda.amp.autocast():",
                        "    output = model(input)",
                        "    loss = criterion(output, target)",
                        "scaler.scale(loss).backward()",
                        "scaler.step(optimizer)",
                        "scaler.update()"
                    ],
                    config_changes={
                        "batch_size": "increase by 4x",
                        "mixed_precision": "enabled",
                        "pin_memory": "True"
                    },
                    metrics_affected=["gpu_utilization", "throughput"]
                ))
        
        return recommendations
    
    def _analyze_model_performance(self, 
                                 model_metrics: List[ModelPerformanceMetrics],
                                 time_window_hours: float) -> List[OptimizationRecommendation]:
        """Analyze model inference performance."""
        
        recommendations = []
        
        if not model_metrics:
            return recommendations
        
        # Filter recent metrics
        current_time = model_metrics[-1].timestamp if model_metrics else 0
        start_time = current_time - (time_window_hours * 3600)
        recent_metrics = [m for m in model_metrics if m.timestamp >= start_time]
        
        if not recent_metrics:
            return recommendations
        
        # Analyze latency
        latencies = [m.avg_latency_ms for m in recent_metrics]
        if latencies:
            avg_latency = np.mean(latencies)
            
            if avg_latency > 100:
                recommendations.append(OptimizationRecommendation(
                    category="model",
                    priority="medium",
                    title="High Inference Latency",
                    description="Model inference is slower than desired",
                    rationale=f"Average latency: {avg_latency:.1f}ms",
                    impact_estimate="40-70% latency reduction",
                    implementation_effort="medium",
                    code_examples=[
                        "# Use model quantization",
                        "model = torch.quantization.quantize_dynamic(",
                        "    model, {torch.nn.Linear}, dtype=torch.qint8",
                        ")",
                        "",
                        "# Enable TensorRT optimization",
                        "model = torch.jit.trace(model, dummy_input)"
                    ],
                    config_changes={
                        "quantization": "dynamic quantization",
                        "optimization": "TensorRT or JIT"
                    },
                    metrics_affected=["avg_latency_ms", "model_size_mb"]
                ))
        
        # Analyze throughput
        throughputs = [m.samples_per_second for m in recent_metrics]
        if throughputs:
            avg_throughput = np.mean(throughputs)
            
            if avg_throughput < 10:
                recommendations.append(OptimizationRecommendation(
                    category="model",
                    priority="medium",
                    title="Low Inference Throughput",
                    description="Model is not processing samples efficiently",
                    rationale=f"Average throughput: {avg_throughput:.1f} samples/sec",
                    impact_estimate="100-300% throughput improvement",
                    implementation_effort="medium",
                    code_examples=[
                        "# Enable batching",
                        "batch_size = 32",
                        "",
                        "# Use model compilation",
                        "model = torch.compile(model)",
                        "",
                        "# Optimize with ONNX",
                        "# Export to ONNX and use optimized runtime"
                    ],
                    config_changes={
                        "batch_inference": "enabled",
                        "model_compilation": "enabled"
                    },
                    metrics_affected=["samples_per_second", "avg_latency_ms"]
                ))
        
        return recommendations
    
    def _analyze_resource_utilization(self,
                                    system_metrics: List[SystemMetrics],
                                    training_metrics: List[TrainingMetrics]) -> List[OptimizationRecommendation]:
        """Analyze resource utilization patterns and inefficiencies."""
        
        recommendations = []
        
        if not system_metrics or not training_metrics:
            return recommendations
        
        # Analyze CPU-GPU utilization ratio
        recent_sys = system_metrics[-10:] if len(system_metrics) >= 10 else system_metrics
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_sys])
        gpu_utils = [m.gpu_utilization for m in recent_sys if m.gpu_utilization > 0]
        
        if gpu_utils and avg_cpu > 80 and np.mean(gpu_utils) < 60:
            recommendations.append(OptimizationRecommendation(
                category="resource",
                priority="high",
                title="CPU-GPU Imbalance",
                description="High CPU load with low GPU utilization suggests data pipeline bottleneck",
                rationale=f"CPU: {avg_cpu:.1f}%, GPU: {np.mean(gpu_utils):.1f}%",
                impact_estimate="50-100% training speed improvement",
                implementation_effort="medium",
                code_examples=[
                    "# Optimize data loading",
                    "DataLoader(dataset, num_workers=8, pin_memory=True)",
                    "",
                    "# Use asynchronous data loading",
                    "from torch.utils.data import prefetch_iterator",
                    "",
                    "# Preload data to GPU",
                    "input = input.pin_memory().cuda(non_blocking=True)"
                ],
                config_changes={
                    "num_workers": "increase to 8-16",
                    "pin_memory": "True",
                    "async_data_loading": "enabled"
                },
                metrics_affected=["cpu_usage_percent", "gpu_utilization", "batch_time"]
            ))
        
        # Analyze memory patterns for leaks
        memory_values = [m.memory_usage_percent for m in system_metrics[-50:] if m.memory_usage_percent > 0]
        if len(memory_values) >= 20:
            # Linear regression to detect memory growth trend
            x = np.arange(len(memory_values))
            slope = np.polyfit(x, memory_values, 1)[0]
            
            if slope > 0.1:  # Growing more than 0.1% per sample
                recommendations.append(OptimizationRecommendation(
                    category="resource",
                    priority="high",
                    title="Potential Memory Leak",
                    description="System memory usage is consistently increasing",
                    rationale=f"Memory growth rate: {slope:.3f}% per sample",
                    impact_estimate="Prevent out-of-memory crashes",
                    implementation_effort="medium",
                    code_examples=[
                        "# Add explicit garbage collection",
                        "import gc",
                        "gc.collect()",
                        "if torch.cuda.is_available():",
                        "    torch.cuda.empty_cache()",
                        "",
                        "# Clear cache after validation",
                        "if phase == 'val':",
                        "    torch.cuda.empty_cache()"
                    ],
                    config_changes={
                        "gc_frequency": "after each epoch",
                        "cache_clearing": "enabled"
                    },
                    metrics_affected=["memory_usage_percent"]
                ))
        
        return recommendations
    
    def _analyze_convergence_patterns(self, 
                                    training_metrics: List[TrainingMetrics]) -> List[OptimizationRecommendation]:
        """Analyze training convergence patterns."""
        
        recommendations = []
        
        if len(training_metrics) < 20:
            return recommendations
        
        # Analyze learning rate effectiveness
        recent_metrics = training_metrics[-20:]
        learning_rates = [m.learning_rate for m in recent_metrics if m.learning_rate > 0]
        
        if learning_rates:
            # Check if learning rate is too high (loss oscillating)
            loss_values = [m.loss for m in recent_metrics]
            loss_changes = np.diff(loss_values)
            
            # Count oscillations (consecutive opposite changes)
            oscillations = 0
            for i in range(1, len(loss_changes) - 1):
                if loss_changes[i] * loss_changes[i+1] < 0:  # Sign change
                    oscillations += 1
            
            oscillation_ratio = oscillations / len(loss_changes) if loss_changes.size > 0 else 0
            
            if oscillation_ratio > 0.3:  # More than 30% oscillations
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="medium",
                    title="High Learning Rate",
                    description="Learning rate appears too high based on loss oscillations",
                    rationale=f"Loss oscillation ratio: {oscillation_ratio:.1%}",
                    impact_estimate="15-30% improvement in convergence",
                    implementation_effort="low",
                    code_examples=[
                        "# Reduce learning rate",
                        "optimizer.param_groups[0]['lr'] *= 0.5",
                        "",
                        "# Use learning rate scheduler",
                        "scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(",
                        "    optimizer, mode='min', factor=0.5, patience=5",
                        ")"
                    ],
                    config_changes={
                        "learning_rate": "reduce by 50%",
                        "lr_scheduler": "ReduceLROnPlateau"
                    },
                    metrics_affected=["loss", "learning_rate"]
                ))
        
        # Analyze plateau detection
        recent_loss = [m.loss for m in training_metrics[-30:] if m.loss > 0]
        if len(recent_loss) >= 20:
            # Check for plateau in recent loss
            loss_std = np.std(recent_loss[-10:]) if len(recent_loss) >= 10 else 0
            loss_mean = np.mean(recent_loss[-10:]) if len(recent_loss) >= 10 else recent_loss[-1]
            
            if loss_std < loss_mean * 0.01:  # Very low variation
                recommendations.append(OptimizationRecommendation(
                    category="training",
                    priority="medium",
                    title="Training Plateau",
                    description="Loss has plateaued, consider learning rate adjustment",
                    rationale=f"Loss variation in last 10 steps: {loss_std/loss_mean:.1%} of mean",
                    impact_estimate="20-40% additional improvement possible",
                    implementation_effort="low",
                    code_examples=[
                        "# Increase learning rate",
                        "optimizer.param_groups[0]['lr'] *= 1.5",
                        "",
                        "# Or use different optimizer",
                        "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)",
                        "",
                        "# Add regularization",
                        "weight_decay = 1e-4"
                    ],
                    config_changes={
                        "learning_rate": "increase by 50%",
                        "optimizer": "AdamW",
                        "weight_decay": "1e-4"
                    },
                    metrics_affected=["loss", "learning_rate"]
                ))
        
        return recommendations
    
    def _analyze_bottlenecks(self,
                           training_metrics: List[TrainingMetrics],
                           system_metrics: List[SystemMetrics]) -> List[OptimizationRecommendation]:
        """Analyze performance bottlenecks."""
        
        recommendations = []
        
        if not training_metrics or not system_metrics:
            return recommendations
        
        # Analyze batch processing time components
        recent_training = training_metrics[-20:]
        recent_system = system_metrics[-20:]
        
        if recent_training:
            # Check data loading bottleneck
            batch_times = [m.batch_time for m in recent_training if m.batch_time > 0]
            data_load_times = [m.data_load_time for m in recent_training if m.data_load_time > 0]
            
            if batch_times and data_load_times:
                avg_batch_time = np.mean(batch_times)
                avg_data_load_time = np.mean(data_load_times)
                
                if avg_data_load_time / avg_batch_time > 0.5:  # Data loading takes >50% of batch time
                    recommendations.append(OptimizationRecommendation(
                        category="system",
                        priority="high",
                        title="Data Loading Bottleneck",
                        description="Data loading is consuming significant training time",
                        rationale=f"Data loading: {avg_data_load_time*1000:.1f}ms, Batch time: {avg_batch_time*1000:.1f}ms",
                        impact_estimate="30-60% training speed improvement",
                        implementation_effort="medium",
                        code_examples=[
                            "# Increase data loading workers",
                            "DataLoader(dataset, num_workers=8, pin_memory=True)",
                            "",
                            "# Use prefetching",
                            "DataLoader(dataset, prefetch_factor=4)",
                            "",
                            "# Cache frequently used data",
                            "dataset = torch.utils.data.TensorDataset(*cached_data)"
                        ],
                        config_changes={
                            "num_workers": "increase to 8-16",
                            "prefetch_factor": "4-8",
                            "pin_memory": "True"
                        },
                        metrics_affected=["batch_time", "data_load_time"]
                    ))
        
        # Check for I/O bottlenecks
        if recent_system:
            disk_reads = [m.disk_read_mb_s for m in recent_system if m.disk_read_mb_s > 0]
            disk_writes = [m.disk_write_mb_s for m in recent_system if m.disk_write_mb_s > 0]
            
            if disk_reads and np.mean(disk_reads) > 100:  # High disk read rate
                recommendations.append(OptimizationRecommendation(
                    category="system",
                    priority="medium",
                    title="Disk I/O Bottleneck",
                    description="High disk I/O usage detected",
                    rationale=f"Average disk read rate: {np.mean(disk_reads):.1f} MB/s",
                    impact_estimate="20-40% I/O performance improvement",
                    implementation_effort="medium",
                    code_examples=[
                        "# Use faster storage for data",
                        "# Copy data to RAM disk or SSD",
                        "",
                        "# Implement data caching",
                        "from functools import lru_cache",
                        "@lru_cache(maxsize=1000)",
                        "def load_data样本(self, idx):",
                        "    return dataset[idx]"
                    ],
                    config_changes={
                        "storage": "SSD or RAM disk",
                        "data_caching": "enabled"
                    },
                    metrics_affected=["disk_read_mb_s", "batch_time"]
                ))
        
        return recommendations
    
    def generate_optimization_report(self,
                                   recommendations: List[OptimizationRecommendation],
                                   include_code_examples: bool = True,
                                   output_format: str = "markdown") -> str:
        """Generate a formatted optimization report."""
        
        if output_format == "markdown":
            return self._generate_markdown_report(recommendations, include_code_examples)
        elif output_format == "json":
            return self._generate_json_report(recommendations)
        elif output_format == "html":
            return self._generate_html_report(recommendations)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _generate_markdown_report(self, 
                                recommendations: List[OptimizationRecommendation],
                                include_code_examples: bool) -> str:
        """Generate markdown report."""
        
        report = "# Performance Optimization Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Group by priority
        by_priority = defaultdict(list)
        for rec in recommendations:
            by_priority[rec.priority].append(rec)
        
        for priority in ['high', 'medium', 'low']:
            if priority in by_priority:
                report += f"## {priority.title()} Priority Recommendations\n\n"
                
                for rec in by_priority[priority]:
                    report += f"### {rec.title}\n\n"
                    report += f"**Category:** {rec.category.title()}\n"
                    report += f"**Impact:** {rec.impact_estimate}\n"
                    report += f"**Effort:** {rec.implementation_effort.title()}\n\n"
                    
                    report += f"**Description:** {rec.description}\n\n"
                    report += f"**Rationale:** {rec.rationale}\n\n"
                    
                    if include_code_examples and rec.code_examples:
                        report += "**Implementation:**\n```python\n"
                        report += "\n".join(rec.code_examples)
                        report += "\n```\n\n"
                    
                    if rec.config_changes:
                        report += "**Configuration Changes:**\n"
                        for key, value in rec.config_changes.items():
                            report += f"- {key}: {value}\n"
                        report += "\n"
                    
                    if rec.metrics_affected:
                        report += f"**Affected Metrics:** {', '.join(rec.metrics_affected)}\n\n"
                    
                    report += "---\n\n"
        
        return report
    
    def _generate_json_report(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Generate JSON report."""
        data = {
            'generated_at': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'recommendations': [rec.to_dict() for rec in recommendations]
        }
        return json.dumps(data, indent=2)
    
    def _generate_html_report(self, recommendations: List[OptimizationRecommendation]) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Optimization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .recommendation { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }
                .high { border-left: 5px solid #dc3545; }
                .medium { border-left: 5px solid #ffc107; }
                .low { border-left: 5px solid #28a745; }
                .code { background: #f8f9fa; padding: 15px; border-radius: 3px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>Performance Optimization Report</h1>
            <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        """
        
        for rec in recommendations:
            html += f"""
            <div class="recommendation {rec.priority}">
                <h2>{rec.title}</h2>
                <p><strong>Category:</strong> {rec.category.title()}</p>
                <p><strong>Impact:</strong> {rec.impact_estimate}</p>
                <p><strong>Effort:</strong> {rec.implementation_effort.title()}</p>
                <p><strong>Description:</strong> {rec.description}</p>
                <p><strong>Rationale:</strong> {rec.rationale}</p>
            """
            
            if rec.code_examples:
                html += f"""
                <h3>Implementation</h3>
                <pre class="code"><code>{chr(10).join(rec.code_examples)}</code></pre>
                """
            
            if rec.config_changes:
                html += "<h3>Configuration Changes</h3><ul>"
                for key, value in rec.config_changes.items():
                    html += f"<li>{key}: {value}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += "</body></html>"
        return html


def generate_performance_recommendations(training_metrics: List[TrainingMetrics],
                                       system_metrics: List[SystemMetrics],
                                       model_metrics: List[ModelPerformanceMetrics] = None,
                                       config: Optional[Dict[str, Any]] = None) -> List[OptimizationRecommendation]:
    """Generate performance optimization recommendations."""
    
    analyzer = PerformanceAnalyzer(config)
    return analyzer.analyze_performance(training_metrics, system_metrics, model_metrics or [])


def save_recommendations_report(recommendations: List[OptimizationRecommendation],
                              output_path: str,
                              format: str = "markdown",
                              include_code_examples: bool = True):
    """Save recommendations report to file."""
    
    analyzer = PerformanceAnalyzer()
    report = analyzer.generate_optimization_report(
        recommendations, 
        include_code_examples=include_code_examples,
        output_format=format
    )
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Performance recommendations report saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../utils')
    
    from performance_monitor import TrainingMetrics, SystemMetrics
    
    # Create sample metrics
    training_metrics = [
        TrainingMetrics(
            step=i,
            global_step=i,
            loss=2.0 - i * 0.01 + np.random.normal(0, 0.1),
            learning_rate=1e-4,
            grad_norm=np.random.uniform(0.1, 2.0),
            batch_time=0.5,
            timestamp=time.time() + i
        )
        for i in range(100)
    ]
    
    system_metrics = [
        SystemMetrics(
            cpu_usage_percent=np.random.uniform(20, 80),
            memory_usage_percent=np.random.uniform(30, 70),
            gpu_utilization=np.random.uniform(40, 90),
            gpu_memory_utilization=np.random.uniform(50, 85),
            timestamp=time.time() + i
        )
        for i in range(100)
    ]
    
    # Generate recommendations
    recommendations = generate_performance_recommendations(training_metrics, system_metrics)
    
    # Print summary
    print(f"Generated {len(recommendations)} recommendations:")
    for rec in recommendations[:5]:  # Show first 5
        print(f"- {rec.title} ({rec.priority} priority)")
    
    # Save report
    save_recommendations_report(recommendations, "optimization_report.md")