#!/usr/bin/env python3
"""
DeepSpeed Performance Benchmarking Utilities
Comprehensive benchmarking suite for DeepSpeed distributed training performance.
"""

import os
import sys
import time
import json
import argparse
import logging
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
from deepspeed import get_accelerator

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DeepSpeedBenchmark:
    """Comprehensive benchmarking suite for DeepSpeed performance."""
    
    def __init__(self, config_path: str, local_rank: int = 0):
        """
        Initialize benchmarking suite.
        
        Args:
            config_path: Path to DeepSpeed configuration file
            local_rank: Local GPU rank
        """
        self.config_path = config_path
        self.local_rank = local_rank
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Benchmark settings
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.results = {}
        
        # Setup
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Failed to load config: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for benchmarking."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - Rank {self.rank} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_dummy_model(self, model_size: str = "medium") -> nn.Module:
        """Create dummy model for benchmarking."""
        class DummyModel(nn.Module):
            def __init__(self, size):
                super().__init__()
                if size == "small":
                    self.fc = nn.Linear(1024, 512)
                elif size == "medium":
                    self.fc1 = nn.Linear(4096, 2048)
                    self.fc2 = nn.Linear(2048, 1024)
                    self.fc3 = nn.Linear(1024, 512)
                elif size == "large":
                    self.fc1 = nn.Linear(8192, 4096)
                    self.fc2 = nn.Linear(4096, 2048)
                    self.fc3 = nn.Linear(2048, 1024)
                    self.fc4 = nn.Linear(1024, 512)
                
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.activation(x)
                x = self.dropout(x)
                x = self.fc2(x)
                x = self.activation(x)
                x = self.dropout(x)
                if hasattr(self, 'fc3'):
                    x = self.fc3(x)
                    x = self.activation(x)
                    x = self.dropout(x)
                if hasattr(self, 'fc4'):
                    x = self.fc4(x)
                    x = self.activation(x)
                    x = self.dropout(x)
                return x
        
        return DummyModel(model_size)
    
    def _create_dummy_dataset(self, num_samples: int = 1000, seq_len: int = 1024) -> TensorDataset:
        """Create dummy dataset for benchmarking."""
        # Create random input data
        inputs = torch.randn(num_samples, seq_len)
        targets = torch.randn(num_samples, 512)
        
        dataset = TensorDataset(inputs, targets)
        return dataset
    
    def benchmark_step_time(self, model_sizes: List[str] = None, batch_sizes: List[int] = None,
                          sequence_lengths: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark step time across different configurations.
        
        Args:
            model_sizes: List of model sizes to test
            batch_sizes: List of batch sizes to test
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            Benchmark results dictionary
        """
        if model_sizes is None:
            model_sizes = ["small", "medium", "large"]
        if batch_sizes is None:
            batch_sizes = [8, 16, 32]
        if sequence_lengths is None:
            sequence_lengths = [512, 1024, 2048]
        
        self.logger.info("Starting step time benchmarking...")
        
        results = {}
        config_list = []
        
        # Generate all combinations
        for model_size in model_sizes:
            for batch_size in batch_sizes:
                for seq_len in sequence_lengths:
                    config_list.append({
                        "model_size": model_size,
                        "batch_size": batch_size,
                        "sequence_length": seq_len
                    })
        
        for config in config_list:
            try:
                # Create model and dataset
                model = self._create_dummy_model(config["model_size"])
                dataset = self._create_dummy_dataset(
                    num_samples=100,
                    seq_len=config["sequence_length"]
                )
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=config["batch_size"],
                    shuffle=False,
                    num_workers=2
                )
                
                # Initialize DeepSpeed
                engine, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    config_params=self.config,
                    model_parameters=self._get_model_parameters(model)
                )
                
                # Warmup
                self._warmup_benchmark(engine, dataloader, warmup_steps=10)
                
                # Benchmark
                step_times = self._benchmark_step_time(engine, dataloader, benchmark_steps=100)
                
                # Calculate statistics
                stats = self._calculate_step_time_stats(step_times)
                
                results[f"{config['model_size']}_batch_{config['batch_size']}_seq_{config['sequence_length']}"] = {
                    "model_size": config["model_size"],
                    "batch_size": config["batch_size"],
                    "sequence_length": config["sequence_length"],
                    "step_times": stats,
                    "throughput_samples_per_sec": self._calculate_throughput(
                        stats["mean_step_time"], config["batch_size"]
                    )
                }
                
                self.logger.info(
                    f"Benchmarked {config['model_size']} model: "
                    f"mean={stats['mean_step_time']:.4f}s, "
                    f"std={stats['std_step_time']:.4f}s, "
                    f"throughput={self._calculate_throughput(stats['mean_step_time'], config['batch_size']):.2f} samples/sec"
                )
                
                # Cleanup
                del engine, optimizer, model, dataset
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for config {config}: {e}")
                continue
        
        self.results["step_time_benchmark"] = results
        return results
    
    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Get model parameters for DeepSpeed."""
        return {
            'params': model.parameters(),
            'weight_decay': 1e-2
        }
    
    def _warmup_benchmark(self, engine, dataloader, warmup_steps: int = 10):
        """Warmup benchmark."""
        engine.train()
        warmup_times = []
        
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= warmup_steps:
                break
            
            start_time = time.time()
            
            # Forward pass
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = engine(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            
            # Backward pass
            engine.backward(loss)
            engine.step()
            
            warmup_times.append(time.time() - start_time)
            
            # Clear gradients
            engine.zero_grad()
        
        self.logger.info(f"Warmup completed. Mean warmup time: {statistics.mean(warmup_times):.4f}s")
    
    def _benchmark_step_time(self, engine, dataloader, benchmark_steps: int = 100) -> List[float]:
        """Benchmark step time."""
        step_times = []
        
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= benchmark_steps:
                break
            
            start_time = time.time()
            
            # Forward pass
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            outputs = engine(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            
            # Backward pass
            engine.backward(loss)
            engine.step()
            
            step_times.append(time.time() - start_time)
            
            # Clear gradients
            engine.zero_grad()
        
        return step_times
    
    def _calculate_step_time_stats(self, step_times: List[float]) -> Dict[str, float]:
        """Calculate statistics for step times."""
        if not step_times:
            return {}
        
        return {
            "mean_step_time": statistics.mean(step_times),
            "median_step_time": statistics.median(step_times),
            "std_step_time": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
            "min_step_time": min(step_times),
            "max_step_time": max(step_times),
            "p95_step_time": self._calculate_percentile(step_times, 95),
            "p99_step_time": self._calculate_percentile(step_times, 99)
        }
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _calculate_throughput(self, step_time: float, batch_size: int) -> float:
        """Calculate throughput in samples per second."""
        return batch_size / step_time if step_time > 0 else 0.0
    
    def benchmark_memory_usage(self, model_sizes: List[str] = None) -> Dict[str, Any]:
        """
        Benchmark memory usage across different model sizes.
        
        Args:
            model_sizes: List of model sizes to test
            
        Returns:
            Memory usage results
        """
        if model_sizes is None:
            model_sizes = ["small", "medium", "large"]
        
        self.logger.info("Starting memory usage benchmarking...")
        
        results = {}
        
        for model_size in model_sizes:
            try:
                # Create model
                model = self._create_dummy_model(model_size)
                
                # Get model parameters
                param_count = sum(p.numel() for p in model.parameters())
                
                # Initialize with DeepSpeed
                engine, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    config_params=self.config,
                    model_parameters=self._get_model_parameters(model)
                )
                
                # Create sample data
                batch_size = 8
                seq_len = 1024
                inputs = torch.randn(batch_size, seq_len)
                targets = torch.randn(batch_size, 512)
                
                # Profile memory
                memory_profile = self._profile_memory_detailed(engine, inputs, targets)
                
                results[model_size] = {
                    "parameter_count": param_count,
                    "memory_usage": memory_profile,
                    "memory_efficiency": self._calculate_memory_efficiency(param_count, memory_profile)
                }
                
                self.logger.info(
                    f"Memory profile for {model_size} model: "
                    f"{memory_profile['total_gpu_memory_gb']:.2f} GB GPU, "
                    f"{memory_profile['total_cpu_memory_gb']:.2f} GB CPU"
                )
                
                # Cleanup
                del engine, optimizer, model
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"Memory benchmark failed for {model_size} model: {e}")
                continue
        
        self.results["memory_usage_benchmark"] = results
        return results
    
    def _profile_memory_detailed(self, engine, inputs, targets) -> Dict[str, float]:
        """Detailed memory profiling."""
        memory_info = {}
        
        # Baseline memory
        torch.cuda.synchronize()
        baseline_gpu = torch.cuda.memory_allocated()
        baseline_cpu = self._get_cpu_memory_usage()
        
        # Move data to device
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        torch.cuda.synchronize()
        data_gpu = torch.cuda.memory_allocated()
        
        # Forward pass
        with torch.no_grad():
            outputs = engine(inputs)
        
        torch.cuda.synchronize()
        forward_gpu = torch.cuda.memory_allocated()
        
        # Backward pass
        loss = nn.functional.mse_loss(outputs, targets)
        engine.backward(loss)
        engine.step()
        
        torch.cuda.synchronize()
        peak_gpu = torch.cuda.max_memory_allocated()
        
        # Final memory
        final_gpu = torch.cuda.memory_allocated()
        final_cpu = self._get_cpu_memory_usage()
        
        # Calculate memory deltas
        memory_info = {
            "baseline_gpu_memory_mb": baseline_gpu / 1024 / 1024,
            "data_gpu_memory_mb": (data_gpu - baseline_gpu) / 1024 / 1024,
            "forward_gpu_memory_mb": (forward_gpu - data_gpu) / 1024 / 1024,
            "peak_gpu_memory_mb": peak_gpu / 1024 / 1024,
            "final_gpu_memory_mb": final_gpu / 1024 / 1024,
            "total_gpu_memory_mb": (peak_gpu - baseline_gpu) / 1024 / 1024,
            "cpu_memory_delta_mb": (final_cpu - baseline_cpu) / 1024 / 1024,
            "total_cpu_memory_gb": final_cpu / 1024 / 1024 / 1024
        }
        
        return memory_info
    
    def _get_cpu_memory_usage(self) -> int:
        """Get current CPU memory usage."""
        import psutil
        return psutil.Process().memory_info().rss
    
    def _calculate_memory_efficiency(self, param_count: int, memory_info: Dict[str, float]) -> float:
        """Calculate memory efficiency (parameters per GB of memory)."""
        total_memory_gb = memory_info["total_gpu_memory_mb"] / 1024
        if total_memory_gb > 0:
            return param_count / total_memory_gb / 1e9  # Parameters per GB in billions
        return 0.0
    
    def benchmark_communication_overhead(self) -> Dict[str, Any]:
        """Benchmark communication overhead in distributed training."""
        if not dist.is_initialized():
            self.logger.warning("Distributed training not initialized - skipping communication benchmark")
            return {}
        
        self.logger.info("Starting communication overhead benchmarking...")
        
        results = {}
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Test different tensor sizes
        tensor_sizes = [1024, 4096, 16384, 65536, 262144]
        
        for size in tensor_sizes:
            try:
                # Test all-reduce
                all_reduce_times = self._benchmark_all_reduce(size, num_tests=10)
                
                # Test broadcast
                broadcast_times = self._benchmark_broadcast(size, num_tests=10)
                
                # Test gather
                gather_times = self._benchmark_gather(size, num_tests=10)
                
                results[f"tensor_size_{size}"] = {
                    "all_reduce": self._calculate_communication_stats(all_reduce_times),
                    "broadcast": self._calculate_communication_stats(broadcast_times),
                    "gather": self._calculate_communication_stats(gather_times)
                }
                
                if rank == 0:
                    self.logger.info(
                        f"Communication benchmark for tensor size {size}: "
                        f"all-reduce: {all_reduce_times[0]:.4f}s, "
                        f"broadcast: {broadcast_times[0]:.4f}s, "
                        f"gather: {gather_times[0]:.4f}s"
                    )
                
            except Exception as e:
                self.logger.error(f"Communication benchmark failed for size {size}: {e}")
                continue
        
        self.results["communication_overhead_benchmark"] = results
        return results
    
    def _benchmark_all_reduce(self, tensor_size: int, num_tests: int = 10) -> List[float]:
        """Benchmark all-reduce operation."""
        times = []
        
        for _ in range(num_tests):
            tensor = torch.randn(tensor_size, dtype=torch.float32).cuda()
            
            start_time = time.time()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        return times
    
    def _benchmark_broadcast(self, tensor_size: int, num_tests: int = 10) -> List[float]:
        """Benchmark broadcast operation."""
        times = []
        
        for _ in range(num_tests):
            tensor = torch.randn(tensor_size, dtype=torch.float32).cuda()
            
            start_time = time.time()
            dist.broadcast(tensor, src=0)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        return times
    
    def _benchmark_gather(self, tensor_size: int, num_tests: int = 10) -> List[float]:
        """Benchmark gather operation."""
        times = []
        
        for _ in range(num_tests):
            tensor = torch.randn(tensor_size, dtype=torch.float32).cuda()
            
            start_time = time.time()
            if dist.get_rank() == 0:
                gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
                dist.gather(tensor, gather_list=gathered_tensors)
            else:
                dist.gather(tensor)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
        
        return times
    
    def _calculate_communication_stats(self, times: List[float]) -> Dict[str, float]:
        """Calculate communication statistics."""
        if not times:
            return {}
        
        return {
            "mean_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min_time": min(times),
            "max_time": max(times)
        }
    
    def benchmark_zero_optimization(self, zero_stages: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark different ZeRO optimization stages.
        
        Args:
            zero_stages: List of ZeRO stages to test
            
        Returns:
            ZeRO optimization results
        """
        if zero_stages is None:
            zero_stages = [1, 2, 3]
        
        self.logger.info("Starting ZeRO optimization benchmarking...")
        
        results = {}
        model = self._create_dummy_model("medium")
        
        for stage in zero_stages:
            try:
                # Create config for this ZeRO stage
                stage_config = self._get_zero_stage_config(stage)
                
                # Initialize DeepSpeed with specific ZeRO stage
                engine, optimizer, _, _ = deepspeed.initialize(
                    model=model,
                    config_params=stage_config,
                    model_parameters=self._get_model_parameters(model)
                )
                
                # Benchmark memory usage
                batch_size = 4
                inputs = torch.randn(batch_size, 1024)
                targets = torch.randn(batch_size, 512)
                
                memory_profile = self._profile_memory_detailed(engine, inputs, targets)
                
                # Benchmark step time
                dataset = self._create_dummy_dataset(num_samples=50)
                dataloader = DataLoader(dataset, batch_size=batch_size)
                
                # Warmup
                self._warmup_benchmark(engine, dataloader, warmup_steps=5)
                
                # Benchmark
                step_times = self._benchmark_step_time(engine, dataloader, benchmark_steps=20)
                step_time_stats = self._calculate_step_time_stats(step_times)
                
                results[f"zero_stage_{stage}"] = {
                    "memory_profile": memory_profile,
                    "step_time_stats": step_time_stats,
                    "throughput": self._calculate_throughput(step_time_stats["mean_step_time"], batch_size)
                }
                
                self.logger.info(
                    f"ZeRO Stage {stage}: "
                    f"memory={memory_profile['total_gpu_memory_mb']:.1f} MB, "
                    f"step_time={step_time_stats['mean_step_time']:.4f}s"
                )
                
                # Cleanup
                del engine, optimizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                self.logger.error(f"ZeRO benchmark failed for stage {stage}: {e}")
                continue
        
        self.results["zero_optimization_benchmark"] = results
        return results
    
    def _get_zero_stage_config(self, stage: int) -> Dict[str, Any]:
        """Get configuration for specific ZeRO stage."""
        base_config = {
            "bf16": {"enabled": True},
            "train_micro_batch_size_per_gpu": 4
        }
        
        if stage == 1:
            base_config["zero_optimization"] = {
                "stage": 1,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "cpu_offload": False
            }
        elif stage == 2:
            base_config["zero_optimization"] = {
                "stage": 2,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8
            }
        elif stage == 3:
            base_config["zero_optimization"] = {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": 1e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 1e8,
                "gather_16bit_weights_on_model_save": True
            }
        
        return base_config
    
    def run_comprehensive_benchmark(self, output_dir: str = "benchmark_results") -> Dict[str, Any]:
        """Run comprehensive benchmark suite."""
        self.logger.info("Starting comprehensive DeepSpeed benchmark suite...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Initialize benchmarking
        if dist.is_initialized():
            self.benchmark_communication_overhead()
        
        # Run benchmarks
        self.benchmark_step_time()
        self.benchmark_memory_usage()
        self.benchmark_zero_optimization()
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        # Save detailed results
        self._save_results(output_dir)
        
        self.logger.info(f"Benchmark completed. Results saved to {output_dir}")
        return self.results
    
    def _generate_summary_report(self, output_dir: Path):
        """Generate summary report."""
        report_lines = []
        report_lines.append("# DeepSpeed Performance Benchmark Report")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Step time summary
        if "step_time_benchmark" in self.results:
            report_lines.append("## Step Time Benchmark Summary")
            step_results = self.results["step_time_benchmark"]
            for config_name, results in step_results.items():
                throughput = results.get("throughput_samples_per_sec", 0)
                report_lines.append(f"- {config_name}: {throughput:.2f} samples/sec")
            report_lines.append("")
        
        # Memory usage summary
        if "memory_usage_benchmark" in self.results:
            report_lines.append("## Memory Usage Summary")
            memory_results = self.results["memory_usage_benchmark"]
            for model_size, results in memory_results.items():
                gpu_memory = results["memory_usage"]["total_gpu_memory_gb"]
                param_count = results["parameter_count"]
                report_lines.append(f"- {model_size} model: {gpu_memory:.2f} GB GPU, {param_count:,} parameters")
            report_lines.append("")
        
        # ZeRO optimization comparison
        if "zero_optimization_benchmark" in self.results:
            report_lines.append("## ZeRO Optimization Comparison")
            zero_results = self.results["zero_optimization_benchmark"]
            for stage, results in zero_results.items():
                memory_mb = results["memory_profile"]["total_gpu_memory_mb"]
                throughput = results["throughput"]
                report_lines.append(f"- {stage}: {memory_mb:.1f} MB, {throughput:.2f} samples/sec")
            report_lines.append("")
        
        # Write report
        report_path = output_dir / "benchmark_summary.md"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
    
    def _save_results(self, output_dir: Path):
        """Save detailed benchmark results."""
        # Save raw results
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save results for each rank (only if distributed)
        if dist.is_initialized():
            rank = dist.get_rank()
            rank_results_path = output_dir / f"benchmark_results_rank_{rank}.json"
            with open(rank_results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="DeepSpeed Performance Benchmarking")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to DeepSpeed configuration file")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Output directory for benchmark results")
    parser.add_argument("--local_rank", type=int, default=0,
                       help="Local GPU rank")
    parser.add_argument("--benchmark_type", type=str, default="all",
                       choices=["step_time", "memory", "communication", "zero", "all"],
                       help="Type of benchmark to run")
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    # Initialize benchmark
    benchmark = DeepSpeedBenchmark(args.config, args.local_rank)
    
    try:
        # Run appropriate benchmarks
        if args.benchmark_type in ["step_time", "all"]:
            benchmark.benchmark_step_time()
        
        if args.benchmark_type in ["memory", "all"]:
            benchmark.benchmark_memory_usage()
        
        if args.benchmark_type in ["communication", "all"] and dist.is_initialized():
            benchmark.benchmark_communication_overhead()
        
        if args.benchmark_type in ["zero", "all"]:
            benchmark.benchmark_zero_optimization()
        
        # Generate comprehensive report
        results = benchmark.run_comprehensive_benchmark(args.output_dir)
        
        if benchmark.rank == 0:
            print(f"Benchmark completed successfully!")
            print(f"Results saved to: {args.output_dir}")
    
    except Exception as e:
        print(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()