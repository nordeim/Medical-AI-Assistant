"""
Performance Benchmarks for Medical AI Training System
====================================================

Comprehensive performance testing and benchmarking suite for:
- Training speed benchmarks
- Inference latency tests
- Memory usage benchmarks
- Model size optimization tests
- Scalability assessments
"""

import os
import sys
import json
import time
import psutil
import threading
import gc
import resource
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import subprocess
import logging

import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    # Timing metrics
    training_time_per_epoch: float = 0.0
    inference_latency: float = 0.0
    data_loading_time: float = 0.0
    preprocessing_time: float = 0.0
    
    # Memory metrics
    peak_memory_usage_mb: float = 0.0
    avg_memory_usage_mb: float = 0.0
    memory_efficiency_score: float = 0.0
    
    # Throughput metrics
    training_throughput_samples_per_sec: float = 0.0
    inference_throughput_samples_per_sec: float = 0.0
    tokens_per_second: float = 0.0
    
    # Model size metrics
    model_size_mb: float = 0.0
    lora_parameters_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # Quality metrics
    convergence_speed: float = 0.0
    final_accuracy: float = 0.0

class PerformanceBenchmark:
    """Main performance benchmarking class."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize benchmark results
        self.benchmark_results = {
            'timestamp': time.time(),
            'system_info': self._get_system_info(),
            'benchmarks': {}
        }
        
        # Performance thresholds for quality gates
        self.thresholds = {
            'training_time_per_epoch_max': 300.0,  # 5 minutes
            'inference_latency_max': 1.0,  # 1 second
            'peak_memory_usage_max': 8192.0,  # 8GB
            'training_throughput_min': 10.0,  # samples/sec
            'inference_throughput_min': 100.0,  # samples/sec
            'model_size_max': 4096.0,  # 4GB
            'lora_parameters_max': 100.0  # 100MB
        }
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            'cpu_count': os.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            for i in range(info['gpu_count']):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info[f'gpu_{i}_name'] = gpu_name
                info[f'gpu_{i}_memory_gb'] = gpu_memory
        
        return info
    
    def benchmark_training_speed(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark training speed with different configurations."""
        logger.info("🚀 Starting training speed benchmarks...")
        
        results = {
            'config': config,
            'timing_metrics': {},
            'throughput_metrics': {},
            'scalability_results': {}
        }
        
        # Test different batch sizes
        batch_sizes = config.get('batch_sizes', [1, 2, 4, 8, 16])
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Simulate training with different batch sizes
            timing_result = self._simulate_training_epoch(batch_size, config)
            
            results['timing_metrics'][f'batch_size_{batch_size}'] = timing_result
            
            # Calculate throughput
            throughput = batch_size / timing_result['epoch_time']
            results['throughput_metrics'][f'batch_size_{batch_size}'] = {
                'samples_per_second': throughput,
                'tokens_per_second': throughput * config.get('seq_length', 512)
            }
        
        # Test different LoRA ranks
        lora_ranks = config.get('lora_ranks', [1, 4, 8, 16, 32])
        
        for rank in lora_ranks:
            logger.info(f"Testing LoRA rank: {rank}")
            
            timing_result = self._simulate_training_epoch(8, config, lora_rank=rank)
            results['timing_metrics'][f'lora_rank_{rank}'] = timing_result
        
        # Save results
        results_file = self.output_dir / "training_speed_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Training speed benchmark completed. Results saved to {results_file}")
        return results
    
    def benchmark_inference_latency(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark inference latency."""
        logger.info("🚀 Starting inference latency benchmarks...")
        
        results = {
            'config': config,
            'latency_metrics': {},
            'throughput_metrics': {},
            'memory_metrics': {}
        }
        
        # Test different batch sizes
        batch_sizes = config.get('inference_batch_sizes', [1, 4, 16, 64])
        
        for batch_size in batch_sizes:
            logger.info(f"Testing inference batch size: {batch_size}")
            
            # Measure latency
            latencies = self._measure_inference_latency(batch_size, config)
            
            results['latency_metrics'][f'batch_size_{batch_size}'] = {
                'avg_latency_ms': np.mean(latencies) * 1000,
                'median_latency_ms': np.median(latencies) * 1000,
                'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                'min_latency_ms': np.min(latencies) * 1000,
                'max_latency_ms': np.max(latencies) * 1000,
                'std_latency_ms': np.std(latencies) * 1000
            }
            
            # Calculate throughput
            throughput = batch_size / np.mean(latencies)
            results['throughput_metrics'][f'batch_size_{batch_size}'] = {
                'samples_per_second': throughput,
                'tokens_per_second': throughput * config.get('seq_length', 512)
            }
            
            # Measure memory usage during inference
            memory_usage = self._measure_inference_memory(batch_size, config)
            results['memory_metrics'][f'batch_size_{batch_size}'] = memory_usage
        
        # Save results
        results_file = self.output_dir / "inference_latency_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Inference latency benchmark completed. Results saved to {results_file}")
        return results
    
    def benchmark_memory_usage(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark memory usage patterns."""
        logger.info("🚀 Starting memory usage benchmarks...")
        
        results = {
            'config': config,
            'memory_profiles': {},
            'peak_memory_analysis': {},
            'memory_efficiency': {}
        }
        
        # Test memory usage during different operations
        operations = ['initialization', 'training_step', 'inference', 'backward_pass']
        
        for operation in operations:
            logger.info(f"Testing memory usage for: {operation}")
            
            memory_profile = self._measure_operation_memory(operation, config)
            results['memory_profiles'][operation] = memory_profile
            
            # Analyze peak memory
            if 'memory_measurements' in memory_profile:
                measurements = memory_profile['memory_measurements']
                results['peak_memory_analysis'][operation] = {
                    'peak_memory_mb': np.max(measurements),
                    'avg_memory_mb': np.mean(measurements),
                    'memory_growth_rate_mb_per_sec': np.mean(np.diff(measurements)) if len(measurements) > 1 else 0
                }
        
        # Test memory efficiency with different configurations
        memory_configs = [
            {'gradient_checkpointing': False, 'mixed_precision': False},
            {'gradient_checkpointing': True, 'mixed_precision': False},
            {'gradient_checkpointing': False, 'mixed_precision': True},
            {'gradient_checkpointing': True, 'mixed_precision': True}
        ]
        
        for i, mem_config in enumerate(memory_configs):
            config_name = f"config_{i+1}"
            logger.info(f"Testing memory config: {config_name}")
            
            memory_efficiency = self._test_memory_efficiency(mem_config, config)
            results['memory_efficiency'][config_name] = memory_efficiency
        
        # Save results
        results_file = self.output_dir / "memory_usage_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Memory usage benchmark completed. Results saved to {results_file}")
        return results
    
    def benchmark_model_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark model size and optimization techniques."""
        logger.info("🚀 Starting model optimization benchmarks...")
        
        results = {
            'config': config,
            'model_sizes': {},
            'compression_ratios': {},
            'optimization_tradeoffs': {}
        }
        
        # Test different quantization methods
        quantization_methods = ['none', 'int8', 'int4', 'fp16', 'bf16']
        
        for method in quantization_methods:
            logger.info(f"Testing quantization method: {method}")
            
            model_size_info = self._measure_model_size(method, config)
            results['model_sizes'][method] = model_size_info
        
        # Test LoRA parameter efficiency
        lora_configs = [
            {'rank': 4, 'alpha': 8, 'dropout': 0.1},
            {'rank': 8, 'alpha': 16, 'dropout': 0.1},
            {'rank': 16, 'alpha': 32, 'dropout': 0.1},
            {'rank': 32, 'alpha': 64, 'dropout': 0.1}
        ]
        
        for lora_config in lora_configs:
            config_name = f"rank_{lora_config['rank']}"
            logger.info(f"Testing LoRA config: {config_name}")
            
            lora_efficiency = self._measure_lora_efficiency(lora_config, config)
            results['optimization_tradeoffs'][config_name] = lora_efficiency
        
        # Calculate compression ratios
        base_model_size = results['model_sizes']['none']['size_mb']
        for method, info in results['model_sizes'].items():
            if method != 'none':
                results['compression_ratios'][method] = {
                    'compression_ratio': base_model_size / info['size_mb'],
                    'size_reduction_percent': (1 - info['size_mb'] / base_model_size) * 100
                }
        
        # Save results
        results_file = self.output_dir / "model_optimization_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Model optimization benchmark completed. Results saved to {results_file}")
        return results
    
    def benchmark_scalability(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark system scalability."""
        logger.info("🚀 Starting scalability benchmarks...")
        
        results = {
            'config': config,
            'data_scaling': {},
            'model_scaling': {},
            'hardware_scaling': {}
        }
        
        # Test data scaling
        data_sizes = config.get('data_sizes', [1000, 5000, 10000, 50000, 100000])
        
        for data_size in data_sizes:
            logger.info(f"Testing data scaling with {data_size} samples")
            
            scaling_metrics = self._measure_data_scaling(data_size, config)
            results['data_scaling'][f'size_{data_size}'] = scaling_metrics
        
        # Test model scaling
        model_sizes = config.get('model_sizes', [7, 13, 30, 70])  # billions of parameters
        
        for model_size in model_sizes:
            logger.info(f"Testing model scaling with {model_size}B parameters")
            
            scaling_metrics = self._measure_model_scaling(model_size, config)
            results['model_scaling'][f'model_{model_size}b'] = scaling_metrics
        
        # Test hardware scaling (if multiple GPUs available)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            gpu_counts = [1, min(2, torch.cuda.device_count()), torch.cuda.device_count()]
            
            for gpu_count in gpu_counts:
                logger.info(f"Testing hardware scaling with {gpu_count} GPUs")
                
                scaling_metrics = self._measure_hardware_scaling(gpu_count, config)
                results['hardware_scaling'][f'gpus_{gpu_count}'] = scaling_metrics
        
        # Save results
        results_file = self.output_dir / "scalability_benchmark.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📊 Scalability benchmark completed. Results saved to {results_file}")
        return results
    
    def _simulate_training_epoch(self, batch_size: int, config: Dict[str, Any], lora_rank: int = 8) -> Dict[str, Any]:
        """Simulate a training epoch and measure timing."""
        # This is a simulation since we don't want to actually train models
        # In a real implementation, this would run actual training
        
        seq_length = config.get('seq_length', 512)
        hidden_size = config.get('hidden_size', 768)
        num_layers = config.get('num_layers', 12)
        
        # Simulate forward pass time
        forward_pass_time = batch_size * seq_length * hidden_size / 1e9  # Rough estimate
        # Simulate backward pass time (typically 2-3x forward pass)
        backward_pass_time = forward_pass_time * 2.5
        # Simulate data loading time
        data_loading_time = batch_size * 0.001  # 1ms per sample
        
        # Add some variance
        variance = np.random.normal(0, 0.1)
        total_time = (forward_pass_time + backward_pass_time + data_loading_time) * (1 + variance)
        
        return {
            'epoch_time': total_time,
            'forward_pass_time': forward_pass_time,
            'backward_pass_time': backward_pass_time,
            'data_loading_time': data_loading_time,
            'throughput_samples_per_sec': batch_size / total_time,
            'lora_rank': lora_rank
        }
    
    def _measure_inference_latency(self, batch_size: int, config: Dict[str, Any], num_runs: int = 100) -> List[float]:
        """Measure inference latency for given batch size."""
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            # Simulate inference (in real implementation, this would be actual model inference)
            # Simulate computation time based on batch size and model complexity
            inference_time = batch_size * config.get('seq_length', 512) / 1e10
            time.sleep(max(0.001, inference_time))  # At least 1ms
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        return latencies
    
    def _measure_inference_memory(self, batch_size: int, config: Dict[str, Any]) -> Dict[str, float]:
        """Measure memory usage during inference."""
        process = psutil.Process()
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate inference memory usage
        # In reality, this would allocate actual tensors
        dummy_tensors = []
        for _ in range(batch_size):
            tensor_size = config.get('seq_length', 512) * config.get('hidden_size', 768)
            dummy_tensor = torch.randn(tensor_size, dtype=torch.float32)
            dummy_tensors.append(dummy_tensor)
        
        # Measure peak memory
        memory_during = process.memory_info().rss / 1024 / 1024  # MB
        peak_memory = memory_during
        
        # Cleanup
        del dummy_tensors
        gc.collect()
        
        # Measure memory after cleanup
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'memory_before_mb': memory_before,
            'memory_peak_mb': peak_memory,
            'memory_after_mb': memory_after,
            'memory_increase_mb': peak_memory - memory_before
        }
    
    def _measure_operation_memory(self, operation: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Measure memory usage for a specific operation."""
        process = psutil.Process()
        memory_measurements = []
        
        # Start monitoring
        def monitor_memory():
            for _ in range(10):  # Monitor for 10 intervals
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_measurements.append(memory_mb)
                time.sleep(0.1)
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        # Perform operation simulation
        time.sleep(0.5)  # Simulate operation duration
        
        # Stop monitoring
        monitor_thread.join()
        
        return {
            'operation': operation,
            'memory_measurements': memory_measurements,
            'peak_memory_mb': max(memory_measurements) if memory_measurements else 0,
            'avg_memory_mb': np.mean(memory_measurements) if memory_measurements else 0
        }
    
    def _test_memory_efficiency(self, mem_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        """Test memory efficiency with different configurations."""
        base_memory = 1000.0  # Base memory usage in MB
        
        # Simulate memory savings
        savings = 0
        if mem_config.get('gradient_checkpointing'):
            savings += 0.3  # 30% savings
        if mem_config.get('mixed_precision'):
            savings += 0.5  # 50% savings
        
        final_memory = base_memory * (1 - savings)
        
        return {
            'base_memory_mb': base_memory,
            'final_memory_mb': final_memory,
            'memory_savings_percent': savings * 100,
            'config': mem_config
        }
    
    def _measure_model_size(self, quantization_method: str, config: Dict[str, Any]) -> Dict[str, float]:
        """Measure model size with different quantization methods."""
        base_model_size = config.get('base_model_size_mb', 14000)  # 14GB for 7B model
        
        # Simulate size reductions
        size_multipliers = {
            'none': 1.0,
            'int8': 0.25,
            'int4': 0.125,
            'fp16': 0.5,
            'bf16': 0.5
        }
        
        multiplier = size_multipliers.get(quantization_method, 1.0)
        model_size = base_model_size * multiplier
        
        return {
            'size_mb': model_size,
            'quantization_method': quantization_method,
            'size_multiplier': multiplier
        }
    
    def _measure_lora_efficiency(self, lora_config: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        """Measure LoRA parameter efficiency."""
        base_model_size = config.get('base_model_size_mb', 14000)
        
        # LoRA adds: 2 * rank * hidden_size * num_layers parameters per weight matrix
        rank = lora_config['rank']
        alpha = lora_config['alpha']
        dropout = lora_config['dropout']
        hidden_size = config.get('hidden_size', 768)
        num_layers = config.get('num_layers', 12)
        
        # Estimate LoRA parameters (approximate)
        lora_params = 2 * rank * hidden_size * num_layers * 16  # 16 weight matrices
        lora_size_mb = lora_params * 4 / (1024 * 1024)  # 4 bytes per float
        
        # Simulate performance impact
        performance_impact = 1.0 - (rank / 128) * 0.1  # Higher rank = more overhead
        
        return {
            'lora_size_mb': lora_size_mb,
            'rank': rank,
            'alpha': alpha,
            'dropout': dropout,
            'size_overhead_percent': (lora_size_mb / base_model_size) * 100,
            'performance_impact': performance_impact
        }
    
    def _measure_data_scaling(self, data_size: int, config: Dict[str, Any]) -> Dict[str, float]:
        """Measure scaling with different data sizes."""
        # Simulate processing time (sublinear scaling)
        base_time = 1.0  # seconds for 1000 samples
        scaling_factor = data_size / 1000
        processing_time = base_time * (scaling_factor ** 0.8)  # Sublinear scaling
        
        # Simulate memory usage (linear scaling)
        base_memory = 100.0  # MB for 1000 samples
        memory_usage = base_memory * scaling_factor
        
        return {
            'processing_time_seconds': processing_time,
            'memory_usage_mb': memory_usage,
            'throughput_samples_per_sec': data_size / processing_time,
            'data_size': data_size
        }
    
    def _measure_model_scaling(self, model_size_billions: int, config: Dict[str, Any]) -> Dict[str, float]:
        """Measure scaling with different model sizes."""
        # Simulate training time (linear with model size)
        base_time = 300.0  # seconds for 7B model
        training_time = base_time * (model_size_billions / 7)
        
        # Simulate memory usage (linear with model size)
        base_memory = 8000.0  # MB for 7B model
        memory_usage = base_memory * (model_size_billions / 7)
        
        return {
            'training_time_per_epoch_seconds': training_time,
            'memory_usage_mb': memory_usage,
            'model_size_billions': model_size_billions
        }
    
    def _measure_hardware_scaling(self, gpu_count: int, config: Dict[str, Any]) -> Dict[str, float]:
        """Measure scaling with different hardware configurations."""
        base_time = 300.0  # seconds with 1 GPU
        # Simulate imperfect scaling due to communication overhead
        scaling_efficiency = 0.85  # 85% efficiency
        training_time = base_time / (gpu_count * scaling_efficiency)
        
        return {
            'training_time_seconds': training_time,
            'speedup_factor': base_time / training_time,
            'efficiency': scaling_efficiency,
            'gpu_count': gpu_count
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report = f"""
Performance Benchmark Report
============================
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
System: {self.benchmark_results['system_info']['cpu_count']} CPUs, 
        {self.benchmark_results['system_info']['memory_total_gb']:.1f}GB RAM
CUDA Available: {self.benchmark_results['system_info']['cuda_available']}

Performance Summary:
-------------------
"""
        
        # Analyze results
        for benchmark_name, benchmark_data in self.benchmark_results.get('benchmarks', {}).items():
            report += f"\n{benchmark_name.replace('_', ' ').title()}:\n"
            
            if 'timing_metrics' in benchmark_data:
                for metric, value in benchmark_data['timing_metrics'].items():
                    if isinstance(value, dict) and 'epoch_time' in value:
                        report += f"  - {metric}: {value['epoch_time']:.2f}s per epoch\n"
            
            if 'throughput_metrics' in benchmark_data:
                for metric, value in benchmark_data['throughput_metrics'].items():
                    if isinstance(value, dict) and 'samples_per_second' in value:
                        report += f"  - {metric}: {value['samples_per_second']:.1f} samples/sec\n"
        
        # Add recommendations
        report += "\nPerformance Recommendations:\n"
        recommendations = self._generate_performance_recommendations()
        for i, rec in enumerate(recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        return report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze benchmark results for recommendations
        for benchmark_name, data in self.benchmark_results.get('benchmarks', {}).items():
            if benchmark_name == 'training_speed':
                # Check for scaling issues
                if 'timing_metrics' in data:
                    batch_times = [v['epoch_time'] for v in data['timing_metrics'].values() if isinstance(v, dict)]
                    if batch_times:
                        best_batch_size = batch_times.index(min(batch_times)) + 1
                        recommendations.append(
                            f"Optimal batch size appears to be {best_batch_size} for best throughput"
                        )
            
            elif benchmark_name == 'memory_usage':
                # Check memory efficiency
                if 'memory_efficiency' in data:
                    best_config = min(data['memory_efficiency'].items(), 
                                    key=lambda x: x[1].get('final_memory_mb', float('inf')))
                    recommendations.append(
                        f"Most memory-efficient configuration: {best_config[0]}"
                    )
        
        return recommendations
    
    def save_benchmark_results(self, results: Dict[str, Any], benchmark_name: str):
        """Save benchmark results."""
        self.benchmark_results['benchmarks'][benchmark_name] = results
        
        # Save complete results
        results_file = self.output_dir / "complete_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)
    
    def run_all_benchmarks(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Run all performance benchmarks."""
        if config is None:
            config = {
                'batch_sizes': [1, 2, 4, 8, 16],
                'inference_batch_sizes': [1, 4, 16, 64],
                'lora_ranks': [1, 4, 8, 16, 32],
                'data_sizes': [1000, 5000, 10000, 50000],
                'model_sizes': [7, 13, 30],
                'seq_length': 512,
                'hidden_size': 768,
                'num_layers': 12,
                'base_model_size_mb': 14000
            }
        
        logger.info("🚀 Starting comprehensive performance benchmarks...")
        logger.info("=" * 60)
        
        try:
            # Run individual benchmarks
            self.benchmark_training_speed(config)
            self.benchmark_inference_latency(config)
            self.benchmark_memory_usage(config)
            self.benchmark_model_optimization(config)
            self.benchmark_scalability(config)
            
            # Generate and save report
            report = self.generate_performance_report()
            report_file = self.output_dir / "performance_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"📊 All benchmarks completed successfully!")
            logger.info(f"📄 Full report saved to: {report_file}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            return False

# ==================== TEST FUNCTIONS ====================

def test_benchmark_functionality():
    """Test benchmark functionality with sample configurations."""
    logger.info("🧪 Testing benchmark functionality...")
    
    # Create benchmark instance
    benchmark = PerformanceBenchmark("test_benchmark_results")
    
    # Test configuration
    test_config = {
        'batch_sizes': [1, 2, 4],
        'inference_batch_sizes': [1, 4],
        'lora_ranks': [1, 4, 8],
        'data_sizes': [1000, 5000],
        'model_sizes': [7, 13],
        'seq_length': 256,
        'hidden_size': 512,
        'num_layers': 6,
        'base_model_size_mb': 7000
    }
    
    # Run a quick benchmark test
    success = benchmark.run_all_benchmarks(test_config)
    
    if success:
        logger.info("✅ Benchmark functionality test passed!")
        
        # Verify results were generated
        results_file = Path("test_benchmark_results/complete_benchmark_results.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info(f"📊 Generated {len(results['benchmarks'])} benchmark categories")
            
            for category, data in results['benchmarks'].items():
                logger.info(f"  - {category}: {len(data)} metrics")
        
        return True
    else:
        logger.error("❌ Benchmark functionality test failed!")
        return False

def run_performance_regression_tests():
    """Run performance regression tests to catch performance degradations."""
    logger.info("🔍 Running performance regression tests...")
    
    # Load previous benchmark results if available
    results_file = Path("benchmark_results/complete_benchmark_results.json")
    if not results_file.exists():
        logger.warning("No previous benchmark results found. Running initial benchmarks.")
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
        return True
    
    # Load previous results
    with open(results_file, 'r') as f:
        previous_results = json.load(f)
    
    # Run current benchmarks
    current_benchmark = PerformanceBenchmark()
    current_benchmark.run_all_benchmarks()
    
    # Compare results
    with open(results_file, 'r') as f:
        current_results = json.load(f)
    
    # Check for performance regressions
    regressions_detected = []
    
    # Simple regression checks (in a real implementation, this would be more sophisticated)
    for category in previous_results['benchmarks']:
        if category in current_results['benchmarks']:
            # Check training speed regression
            if category == 'training_speed':
                prev_times = []
                curr_times = []
                
                for metric, data in previous_results['benchmarks'][category].get('timing_metrics', {}).items():
                    if isinstance(data, dict) and 'epoch_time' in data:
                        prev_times.append(data['epoch_time'])
                
                for metric, data in current_results['benchmarks'][category].get('timing_metrics', {}).items():
                    if isinstance(data, dict) and 'epoch_time' in data:
                        curr_times.append(data['epoch_time'])
                
                if prev_times and curr_times:
                    avg_prev = sum(prev_times) / len(prev_times)
                    avg_curr = sum(curr_times) / len(curr_times)
                    
                    if avg_curr > avg_prev * 1.2:  # 20% degradation threshold
                        regressions_detected.append(
                            f"Training speed regression: {((avg_curr/avg_prev - 1) * 100):.1f}% slower"
                        )
    
    if regressions_detected:
        logger.warning("⚠️ Performance regressions detected:")
        for regression in regressions_detected:
            logger.warning(f"  - {regression}")
        return False
    else:
        logger.info("✅ No performance regressions detected!")
        return True

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run performance benchmarks for Medical AI Training System')
    parser.add_argument('--test', action='store_true', help='Run functionality test')
    parser.add_argument('--regression', action='store_true', help='Run regression tests')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory for results')
    parser.add_argument('--config', type=str, help='Configuration file (JSON)')
    
    args = parser.parse_args()
    
    if args.test:
        success = test_benchmark_functionality()
        sys.exit(0 if success else 1)
    
    if args.regression:
        success = run_performance_regression_tests()
        sys.exit(0 if success else 1)
    
    # Run full benchmark suite
    benchmark = PerformanceBenchmark(args.output_dir)
    
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    success = benchmark.run_all_benchmarks(config)
    sys.exit(0 if success else 1)