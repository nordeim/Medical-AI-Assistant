"""
Performance Benchmarking Tools for Adapters

This module provides comprehensive benchmarking tools to measure adapter performance
across various metrics including latency, throughput, memory usage, and quality.
"""

import asyncio
import gc
import json
import logging
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
import psutil
import torch
import numpy as np
from transformers import AutoTokenizer

from adapter_manager import AdapterManager, create_adapter_manager_async

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results."""
    # Latency metrics
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    
    # Throughput metrics
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    memory_variance: float = 0.0
    
    # Quality metrics
    average_score: float = 0.0
    quality_std: float = 0.0
    
    # System metrics
    cpu_usage_percent: float = 0.0
    gpu_memory_usage_mb: Optional[float] = None
    
    # Benchmark metadata
    total_tokens: int = 0
    total_requests: int = 0
    total_time_seconds: float = 0.0
    adapter_id: str = ""
    benchmark_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Adapter: {self.adapter_id}
Latency: {self.avg_latency_ms:.1f}ms (p95: {self.p95_latency_ms:.1f}ms)
Throughput: {self.tokens_per_second:.1f} tok/s, {self.requests_per_second:.1f} req/s
Memory: Peak {self.peak_memory_mb:.1f}MB, Avg {self.avg_memory_mb:.1f}MB
Quality: {self.average_score:.2f} ± {self.quality_std:.2f}
Requests: {self.total_requests}, Tokens: {self.total_tokens}
        """.strip()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    # Benchmark parameters
    num_warmup_requests: int = 5
    num_benchmark_requests: int = 100
    concurrent_requests: int = 1
    batch_size: int = 1
    
    # Prompt configuration
    prompt_lengths: List[int] = field(default_factory=lambda: [10, 50, 100, 200, 500])
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    
    # System configuration
    device: str = "auto"
    dtype: str = "float16"
    use_cache: bool = True
    
    # Quality assessment
    enable_quality_check: bool = False
    quality_evaluator: Optional[Callable] = None
    
    # Memory management
    clear_cache_between_tests: bool = True
    gc_frequency: int = 10
    
    # Output configuration
    save_results: bool = True
    results_dir: str = "./benchmark_results"
    verbose: bool = True
    
    def __post_init__(self):
        """Ensure results directory exists."""
        if self.save_results:
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)


class MemoryProfiler:
    """Memory profiling utilities for benchmark."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.measurements: List[Dict[str, float]] = []
    
    @contextmanager
    def profile(self):
        """Context manager for memory profiling."""
        start_memory = self.get_memory_usage()
        
        # Record initial state
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        self.measurements.append({
            "timestamp": time.time(),
            "rss_mb": start_memory["rss_mb"],
            "vms_mb": start_memory["vms_mb"],
            "gpu_memory_mb": self.get_gpu_memory() or 0
        })
        
        try:
            yield
        finally:
            end_memory = self.get_memory_usage()
            self.measurements.append({
                "timestamp": time.time(),
                "rss_mb": end_memory["rss_mb"],
                "vms_mb": end_memory["vms_mb"],
                "gpu_memory_mb": self.get_gpu_memory() or 0
            })
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": self.process.memory_percent()
        }
    
    def get_gpu_memory(self) -> Optional[float]:
        """Get GPU memory usage if available."""
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    
    def get_peak_memory(self) -> Tuple[float, Optional[float]]:
        """Get peak memory usage."""
        if not self.measurements:
            return 0.0, None
        
        cpu_memory = max(m["rss_mb"] for m in self.measurements)
        gpu_memory = max(m["gpu_memory_mb"] for m in self.measurements if m["gpu_memory_mb"] > 0)
        
        return cpu_memory, gpu_memory if gpu_memory > 0 else None
    
    def reset(self):
        """Reset measurements."""
        self.measurements.clear()


class QualityEvaluator:
    """Base class for quality evaluation."""
    
    def __init__(self, evaluator_type: str = "simple"):
        self.evaluator_type = evaluator_type
    
    def evaluate(self, prompt: str, response: str) -> float:
        """Evaluate response quality. Returns score between 0.0 and 1.0."""
        if self.evaluator_type == "simple":
            return self._simple_evaluation(prompt, response)
        elif self.evaluator_type == "length_based":
            return self._length_based_evaluation(response)
        elif self.evaluator_type == "semantic":
            return self._semantic_evaluation(prompt, response)
        else:
            return 0.5  # Default neutral score
    
    def _simple_evaluation(self, prompt: str, response: str) -> float:
        """Simple heuristic quality evaluation."""
        if not response or not response.strip():
            return 0.0
        
        # Basic quality checks
        score = 0.5  # Base score
        
        # Length appropriateness (not too short, not too long)
        length_ratio = len(response) / len(prompt)
        if 0.5 <= length_ratio <= 5.0:
            score += 0.2
        
        # Contains meaningful content
        if len(response.split()) >= 3:
            score += 0.2
        
        # Not repetitive
        words = response.lower().split()
        if len(set(words)) / len(words) > 0.6:
            score += 0.1
        
        return min(1.0, score)
    
    def _length_based_evaluation(self, response: str) -> float:
        """Length-based quality evaluation."""
        if not response:
            return 0.0
        
        word_count = len(response.split())
        char_count = len(response)
        
        # Optimal ranges
        if 10 <= word_count <= 100 and 50 <= char_count <= 1000:
            return 1.0
        elif 5 <= word_count <= 200 and 25 <= char_count <= 2000:
            return 0.8
        else:
            return 0.5
    
    def _semantic_evaluation(self, prompt: str, response: str) -> float:
        """Semantic quality evaluation (placeholder for more sophisticated methods)."""
        # This would typically use embeddings or a trained quality model
        # For now, use simple heuristics
        return self._simple_evaluation(prompt, response)


class PromptGenerator:
    """Generates test prompts for benchmarking."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def generate_prompt(self, target_length: int, category: str = "general") -> str:
        """Generate a prompt of approximately target_length tokens."""
        templates = {
            "general": [
                "Explain the concept of {topic} in simple terms.",
                "What are the main benefits of {topic}?",
                "How does {topic} work in practice?",
                "Compare {topic} to other similar approaches.",
                "Describe the history and development of {topic}."
            ],
            "technical": [
                "Explain the algorithm behind {topic} implementation.",
                "What are the computational complexities of {topic}?",
                "Describe the mathematical foundation of {topic}.",
                "How can {topic} be optimized for performance?",
                "What are the limitations and challenges of {topic}?"
            ],
            "creative": [
                "Write a short story about {topic} in a futuristic setting.",
                "Describe a day in the life of someone working with {topic}.",
                "Create a dialogue between experts discussing {topic}.",
                "Imagine if {topic} could think - what would it say?",
                "Write a poem about the beauty of {topic}."
            ],
            "analytical": [
                "Analyze the pros and cons of {topic}.",
                "What are the potential risks associated with {topic}?",
                "How might {topic} evolve in the next decade?",
                "What evidence supports the effectiveness of {topic}?",
                "Compare different approaches to solving problems with {topic}."
            ]
        }
        
        topics = ["machine learning", "quantum computing", "renewable energy", 
                 "artificial intelligence", "climate change", "space exploration",
                 "biotechnology", "neural networks", "blockchain", "robotics"]
        
        template = random.choice(templates.get(category, templates["general"]))
        topic = random.choice(topics)
        
        prompt = template.format(topic=topic)
        
        # Extend to target length
        while len(self.tokenizer.encode(prompt)) < target_length:
            extensions = [
                " Please provide detailed explanations with examples.",
                " Consider both theoretical and practical aspects.",
                " Include recent developments and future trends.",
                " Discuss the implications for society and industry.",
                " Provide a comprehensive analysis with supporting evidence."
            ]
            prompt += random.choice(extensions)
        
        return prompt
    
    def generate_batch(self, count: int, target_length: int, category: str = "general") -> List[str]:
        """Generate a batch of prompts."""
        return [self.generate_prompt(target_length, category) for _ in range(count)]


class PerformanceBenchmark:
    """Main benchmark class for adapter performance testing."""
    
    def __init__(self, adapter_manager: AdapterManager, config: Optional[BenchmarkConfig] = None):
        self.adapter_manager = adapter_manager
        self.config = config or BenchmarkConfig()
        self.memory_profiler = MemoryProfiler()
        self.quality_evaluator = QualityEvaluator()
        
        # Get tokenizer from adapter manager
        self.tokenizer = adapter_manager._base_tokenizer
        
        # Generate prompts
        self.prompt_generator = PromptGenerator(self.tokenizer)
    
    async def run_comprehensive_benchmark(self, adapter_id: str) -> BenchmarkMetrics:
        """Run comprehensive benchmark suite."""
        logger.info(f"Starting comprehensive benchmark for adapter: {adapter_id}")
        
        # Ensure adapter is loaded and active
        if not self.adapter_manager.switch_adapter(adapter_id):
            raise ValueError(f"Failed to switch to adapter: {adapter_id}")
        
        results = BenchmarkMetrics(adapter_id=adapter_id)
        
        try:
            with self.memory_profiler.profile():
                # Warmup
                await self._warmup()
                
                # Single request latency benchmark
                latency_results = await self._benchmark_latency()
                results.avg_latency_ms = statistics.mean(latency_results)
                results.median_latency_ms = statistics.median(latency_results)
                results.p95_latency_ms = self._percentile(latency_results, 95)
                results.p99_latency_ms = self._percentile(latency_results, 99)
                results.min_latency_ms = min(latency_results)
                results.max_latency_ms = max(latency_results)
                results.latency_std_ms = statistics.stdev(latency_results)
                
                # Throughput benchmark
                throughput_results = await self._benchmark_throughput()
                results.tokens_per_second = throughput_results["tokens_per_second"]
                results.requests_per_second = throughput_results["requests_per_second"]
                results.total_tokens = throughput_results["total_tokens"]
                results.total_requests = throughput_results["total_requests"]
                
                # Memory benchmark
                cpu_memory, gpu_memory = self.memory_profiler.get_peak_memory()
                results.peak_memory_mb = cpu_memory
                results.gpu_memory_usage_mb = gpu_memory
                
                # Quality benchmark
                if self.config.enable_quality_check:
                    quality_results = await self._benchmark_quality()
                    results.average_score = statistics.mean(quality_results)
                    results.quality_std = statistics.stdev(quality_results)
                
                # System metrics
                results.cpu_usage_percent = psutil.cpu_percent()
                
                # Benchmark config
                results.benchmark_config = asdict(self.config)
                
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            raise
        
        # Save results if configured
        if self.config.save_results:
            await self._save_results(results)
        
        return results
    
    async def _warmup(self):
        """Warmup the adapter with a few requests."""
        logger.info("Warming up adapter...")
        
        warmup_prompt = "Hello, how are you?"
        
        for _ in range(self.config.num_warmup_requests):
            await self._generate_single(warmup_prompt)
            
            if self.config.clear_cache_between_tests and _ % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Warmup complete")
    
    async def _benchmark_latency(self) -> List[float]:
        """Benchmark single request latency."""
        logger.info("Running latency benchmark...")
        
        latencies = []
        
        for prompt_length in self.config.prompt_lengths:
            prompt = self.prompt_generator.generate_prompt(prompt_length)
            
            # Test multiple prompts for this length
            latencies_for_length = []
            
            for _ in range(10):  # Test multiple requests per length
                start_time = time.time()
                await self._generate_single(prompt)
                latency = (time.time() - start_time) * 1000  # Convert to ms
                latencies_for_length.append(latency)
            
            logger.info(f"Length {prompt_length}: {statistics.mean(latencies_for_length):.1f}ms avg")
            latencies.extend(latencies_for_length)
        
        return latencies
    
    async def _benchmark_throughput(self) -> Dict[str, Any]:
        """Benchmark throughput with concurrent requests."""
        logger.info("Running throughput benchmark...")
        
        total_tokens = 0
        total_requests = 0
        start_time = time.time()
        
        if self.config.concurrent_requests == 1:
            # Sequential requests
            for _ in range(self.config.num_benchmark_requests):
                prompt = self.prompt_generator.generate_prompt(random.choice(self.config.prompt_lengths))
                result = await self._generate_single(prompt)
                total_tokens += len(self.tokenizer.encode(result))
                total_requests += 1
        else:
            # Concurrent requests
            tasks = []
            for _ in range(self.config.num_benchmark_requests):
                prompt = self.prompt_generator.generate_prompt(random.choice(self.config.prompt_lengths))
                task = self._generate_single(prompt)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, str):
                    total_tokens += len(self.tokenizer.encode(result))
                    total_requests += 1
        
        total_time = time.time() - start_time
        
        return {
            "tokens_per_second": total_tokens / total_time,
            "requests_per_second": total_requests / total_time,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "total_time_seconds": total_time
        }
    
    async def _benchmark_quality(self) -> List[float]:
        """Benchmark response quality."""
        logger.info("Running quality benchmark...")
        
        quality_scores = []
        
        for prompt_length in self.config.prompt_lengths:
            prompt = self.prompt_generator.generate_prompt(prompt_length)
            
            for _ in range(5):  # Test multiple responses per length
                response = await self._generate_single(prompt)
                score = self.quality_evaluator.evaluate(prompt, response)
                quality_scores.append(score)
        
        logger.info(f"Quality score: {statistics.mean(quality_scores):.2f} ± {statistics.stdev(quality_scores):.2f}")
        return quality_scores
    
    async def _generate_single(self, prompt: str) -> str:
        """Generate a single response."""
        try:
            active_adapter = self.adapter_manager.get_active_adapter()
            if not active_adapter:
                raise ValueError("No active adapter found")
            
            adapter_instance = active_adapter[1]
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if hasattr(adapter_instance, 'to'):
                inputs = inputs.to(adapter_instance.device)
            
            # Generate
            with torch.no_grad():
                outputs = adapter_instance.generate(
                    inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            prompt_len = inputs.shape[1]
            generated_tokens = outputs[0][prompt_len:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return ""  # Return empty string on failure
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def _save_results(self, results: BenchmarkMetrics):
        """Save benchmark results to file."""
        timestamp = int(time.time())
        adapter_name = results.adapter_id.replace("/", "_")
        filename = f"benchmark_{adapter_name}_{timestamp}.json"
        filepath = Path(self.config.results_dir) / filename
        
        result_dict = results.to_dict()
        result_dict["timestamp"] = time.time()
        result_dict["system_info"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")


class BenchmarkSuite:
    """Collection of benchmark methods for different scenarios."""
    
    def __init__(self, adapter_manager: AdapterManager):
        self.adapter_manager = adapter_manager
    
    async def benchmark_stress_test(self, adapter_id: str, duration_seconds: int = 300) -> Dict[str, Any]:
        """Stress test adapter under sustained load."""
        logger.info(f"Starting stress test for {duration_seconds} seconds...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        request_count = 0
        error_count = 0
        latencies = []
        
        async def stress_worker():
            nonlocal request_count, error_count, latencies
            
            while time.time() < end_time:
                try:
                    # Get active adapter
                    active_adapter = self.adapter_manager.get_active_adapter()
                    if not active_adapter:
                        error_count += 1
                        continue
                    
                    adapter_instance = active_adapter[1]
                    
                    # Generate random prompt
                    prompt = f"Test prompt {random.randint(1, 1000)}: " + " ".join(
                        ["word"] * random.randint(5, 20)
                    )
                    
                    # Measure latency
                    start = time.time()
                    with torch.no_grad():
                        inputs = self.adapter_manager._base_tokenizer.encode(prompt, return_tensors="pt")
                        if hasattr(adapter_instance, 'to'):
                            inputs = inputs.to(adapter_instance.device)
                        
                        outputs = adapter_instance.generate(inputs, max_new_tokens=10)
                    
                    latency = (time.time() - start) * 1000
                    latencies.append(latency)
                    request_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Stress test error: {e}")
                
                # Small delay between requests
                await asyncio.sleep(0.01)
        
        # Run multiple workers
        num_workers = 4
        tasks = [stress_worker() for _ in range(num_workers)]
        await asyncio.gather(*tasks)
        
        # Calculate results
        total_time = time.time() - start_time
        results = {
            "duration_seconds": total_time,
            "total_requests": request_count,
            "total_errors": error_count,
            "requests_per_second": request_count / total_time,
            "error_rate": error_count / (request_count + error_count) if (request_count + error_count) > 0 else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": self._percentile(latencies, 95) if latencies else 0
        }
        
        logger.info(f"Stress test completed: {results}")
        return results
    
    async def benchmark_memory_patterns(self, adapter_id: str) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        logger.info("Analyzing memory patterns...")
        
        # Switch to adapter
        if not self.adapter_manager.switch_adapter(adapter_id):
            raise ValueError(f"Failed to switch to adapter: {adapter_id}")
        
        memory_samples = []
        
        # Collect memory samples over time
        sample_duration = 30  # seconds
        sample_interval = 1   # seconds
        
        start_time = time.time()
        while time.time() - start_time < sample_duration:
            memory_info = psutil.Process().memory_info()
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            
            memory_samples.append({
                "timestamp": time.time(),
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "gpu_memory_mb": gpu_memory
            })
            
            await asyncio.sleep(sample_interval)
        
        # Analyze patterns
        rss_values = [s["rss_mb"] for s in memory_samples]
        gpu_values = [s["gpu_memory_mb"] for s in memory_samples]
        
        results = {
            "memory_samples": len(memory_samples),
            "rss_stats": {
                "mean": statistics.mean(rss_values),
                "std": statistics.stdev(rss_values) if len(rss_values) > 1 else 0,
                "min": min(rss_values),
                "max": max(rss_values)
            },
            "gpu_memory_stats": {
                "mean": statistics.mean(gpu_values) if gpu_values else 0,
                "std": statistics.stdev(gpu_values) if len(gpu_values) > 1 else 0,
                "min": min(gpu_values) if gpu_values else 0,
                "max": max(gpu_values) if gpu_values else 0
            } if torch.cuda.is_available() else None
        }
        
        return results
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


# Utility functions
async def benchmark_adapter(adapter_path: str, 
                          base_model_id: str,
                          adapter_id: Optional[str] = None,
                          config: Optional[BenchmarkConfig] = None) -> BenchmarkMetrics:
    """Quick benchmark function for single adapter."""
    
    # Create manager
    manager = await create_adapter_manager_async(base_model_id)
    
    try:
        # Load adapter
        await manager.load_adapter_async(adapter_path, adapter_id)
        
        # Run benchmark
        benchmark = PerformanceBenchmark(manager, config)
        results = await benchmark.run_comprehensive_benchmark(adapter_id or adapter_path)
        
        return results
        
    finally:
        manager.cleanup()


async def compare_adapters(adapter_configs: List[Dict[str, str]], 
                         base_model_id: str) -> Dict[str, BenchmarkMetrics]:
    """Compare multiple adapters and return results."""
    
    results = {}
    
    for config in adapter_configs:
        adapter_path = config["path"]
        adapter_id = config.get("id", Path(adapter_path).name)
        
        logger.info(f"Benchmarking adapter: {adapter_id}")
        
        try:
            result = await benchmark_adapter(adapter_path, base_model_id, adapter_id)
            results[adapter_id] = result
        except Exception as e:
            logger.error(f"Failed to benchmark {adapter_id}: {e}")
            results[adapter_id] = BenchmarkMetrics(adapter_id=adapter_id)  # Empty result
    
    return results


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create sample config
        config = BenchmarkConfig(
            num_warmup_requests=2,
            num_benchmark_requests=20,
            prompt_lengths=[10, 50, 100],
            enable_quality_check=True,
            save_results=True
        )
        
        # Benchmark single adapter
        results = await benchmark_adapter(
            adapter_path="./sample_adapter",
            base_model_id="microsoft/DialoGPT-medium",
            adapter_id="sample_adapter",
            config=config
        )
        
        print("Benchmark Results:")
        print(results.summary())
    
    # Run example
    # asyncio.run(main())