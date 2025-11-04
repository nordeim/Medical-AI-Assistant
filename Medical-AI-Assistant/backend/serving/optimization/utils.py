"""
Utility functions for the optimization framework.
"""

import torch
import torch.nn as nn
import psutil
import GPUtil
import logging
import time
import json
import os
import shutil
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import numpy as np
import platform
import sys


logger = logging.getLogger(__name__)


@dataclass
class SystemSpecs:
    """System specifications and capabilities."""
    cpu_model: str
    cpu_cores: int
    cpu_threads: int
    total_memory_gb: float
    available_memory_gb: float
    gpu_count: int
    gpu_models: List[str]
    gpu_memory_gb: List[float]
    cuda_available: bool
    cuda_version: Optional[str]
    torch_version: str
    platform: str
    python_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_model": self.cpu_model,
            "cpu_cores": self.cpu_cores,
            "cpu_threads": self.cpu_threads,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "gpu_count": self.gpu_count,
            "gpu_models": self.gpu_models,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "platform": self.platform,
            "python_version": self.python_version
        }


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_usage_percent: Dict[int, float]
    gpu_memory_usage_percent: Dict[int, float]
    disk_usage_percent: float
    network_io_mbps: float
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_percent": self.memory_usage_percent,
            "gpu_usage_percent": self.gpu_usage_percent,
            "gpu_memory_usage_percent": self.gpu_memory_usage_percent,
            "disk_usage_percent": self.disk_usage_percent,
            "network_io_mbps": self.network_io_mbps,
            "timestamp": self.timestamp
        }


class SystemProfiler:
    """Advanced system profiling and capability detection."""
    
    def __init__(self):
        self.specs = self._detect_system_specs()
        self.performance_history = []
        self._monitoring_active = False
        
        logger.info(f"System profiler initialized - {self.specs.cpu_model} "
                   f"with {self.specs.gpu_count} GPU(s)")
    
    def _detect_system_specs(self) -> SystemSpecs:
        """Detect comprehensive system specifications."""
        # CPU information
        cpu_model = platform.processor() or "Unknown CPU"
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_threads = psutil.cpu_count(logical=True) or 1
        
        # Memory information
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # GPU information
        gpu_count = 0
        gpu_models = []
        gpu_memory_gb = []
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            for i in range(gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    gpu_models.append(props.name)
                    gpu_memory_gb.append(props.total_memory / (1024**3))
                except Exception as e:
                    logger.warning(f"Could not get GPU {i} properties: {e}")
                    gpu_models.append(f"GPU_{i}")
                    gpu_memory_gb.append(0.0)
        
        # CUDA version
        cuda_version = None
        if torch.cuda.is_available():
            try:
                cuda_version = torch.version.cuda
            except:
                pass
        
        return SystemSpecs(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
            gpu_memory_gb=gpu_memory_gb,
            cuda_available=torch.cuda.is_available(),
            cuda_version=cuda_version,
            torch_version=torch.__version__,
            platform=platform.system(),
            python_version=platform.python_version()
        )
    
    def get_current_performance(self) -> PerformanceMetrics:
        """Get current system performance metrics."""
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_usage = {}
        gpu_memory_usage = {}
        
        if self.specs.cuda_available:
            try:
                # GPU utilization
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_usage[gpu.id] = gpu.load * 100
                    gpu_memory_usage[gpu.id] = gpu.memoryUtil * 100
            except Exception as e:
                logger.debug(f"Could not get GPU metrics: {e}")
                # Fallback to CUDA statistics
                for i in range(self.specs.gpu_count):
                    try:
                        allocated = torch.cuda.memory_allocated(i)
                        reserved = torch.cuda.memory_reserved(i)
                        total = torch.cuda.get_device_properties(i).total_memory
                        
                        gpu_usage[i] = 0.0  # Can't get utilization without nvidia-ml
                        gpu_memory_usage[i] = (reserved / total) * 100 if total > 0 else 0.0
                    except Exception:
                        gpu_usage[i] = 0.0
                        gpu_memory_usage[i] = 0.0
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network I/O (simplified)
        net_io = psutil.net_io_counters()
        network_io_mbps = 0.0  # Would need delta calculation for real value
        
        return PerformanceMetrics(
            cpu_usage_percent=cpu_usage,
            memory_usage_percent=memory_usage,
            gpu_usage_percent=gpu_usage,
            gpu_memory_usage_percent=gpu_memory_usage,
            disk_usage_percent=disk_usage,
            network_io_mbps=network_io_mbps,
            timestamp=time.time()
        )
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate optimization recommendations based on system specs."""
        recommendations = {
            "general": [],
            "quantization": [],
            "memory": [],
            "device": [],
            "batch_processing": []
        }
        
        # Memory-based recommendations
        memory_gb = self.specs.total_memory_gb
        available_gb = self.specs.available_memory_gb
        
        if memory_gb < 8:
            recommendations["general"].append("System has limited memory - use aggressive optimization")
            recommendations["memory"].append("Enable gradient checkpointing and CPU offloading")
            recommendations["quantization"].append("Use INT4 quantization to save memory")
            recommendations["batch_processing"].append("Use smaller batch sizes (8-16)")
        elif memory_gb < 16:
            recommendations["general"].append("Moderate memory available - use balanced optimization")
            recommendations["memory"].append("Consider enabling gradient checkpointing for large models")
            recommendations["quantization"].append("INT8 quantization is recommended")
        else:
            recommendations["general"].append("Sufficient memory for most optimizations")
            recommendations["memory"].append("Memory optimization not critical")
        
        # GPU-based recommendations
        if self.specs.gpu_count > 0:
            recommendations["device"].append("GPU available - use CUDA acceleration")
            
            if self.specs.cuda_available:
                recommendations["device"].append("CUDA available - enable GPU optimizations")
                
                # GPU memory recommendations
                gpu_memory = self.specs.gpu_memory_gb[0] if self.specs.gpu_memory_gb else 0
                if gpu_memory < 8:
                    recommendations["device"].append("Limited GPU memory - use model parallelism")
                    recommendations["quantization"].append("INT4 quantization essential for large models")
                elif gpu_memory < 16:
                    recommendations["device"].append("Moderate GPU memory - use balanced settings")
                else:
                    recommendations["device"].append("Sufficient GPU memory for most models")
        else:
            recommendations["device"].append("No GPU detected - use CPU-optimized inference")
            recommendations["batch_processing"].append("Enable larger batches for CPU efficiency")
        
        # CPU-based recommendations
        if self.specs.cpu_cores >= 8:
            recommendations["general"].append("Multi-core CPU available - enable parallel processing")
            recommendations["batch_processing"].append("Use larger batch sizes (32-64) for CPU efficiency")
        else:
            recommendations["general"].append("Limited CPU cores - focus on single-threaded optimization")
        
        # Platform-specific recommendations
        if self.specs.platform == "Linux":
            recommendations["general"].append("Linux detected - optimal for ML workloads")
        elif self.specs.platform == "Windows":
            recommendations["general"].append("Windows detected - some optimizations may be limited")
        elif self.specs.platform == "Darwin":  # macOS
            recommendations["general"].append("macOS detected - use Metal Performance Shaders if available")
        
        return recommendations
    
    def check_optimization_compatibility(self) -> Dict[str, Any]:
        """Check which optimizations are compatible with the current system."""
        compatibility = {
            "quantization": {},
            "memory_optimization": {},
            "device_acceleration": {},
            "batch_processing": {}
        }
        
        # Quantization compatibility
        compatibility["quantization"]["int8"] = True  # Always available with PyTorch
        compatibility["quantization"]["int4"] = self.specs.cuda_available and any("RTX" in model or "A100" in model for model in self.specs.gpu_models)
        compatibility["quantization"]["bnb"] = self._check_bitsandbytes_availability()
        
        # Memory optimization compatibility
        compatibility["memory_optimization"]["gradient_checkpointing"] = True
        compatibility["memory_optimization"]["cpu_offload"] = True
        compatibility["memory_optimization"]["disk_offload"] = self.specs.platform != "Darwin"  # macOS has issues with disk offloading
        
        # Device acceleration compatibility
        compatibility["device_acceleration"]["cuda"] = self.specs.cuda_available
        compatibility["device_acceleration"]["tensorrt"] = self.specs.cuda_available and self._check_tensorrt_availability()
        compatibility["device_acceleration"]["mps"] = self.specs.platform == "Darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Batch processing compatibility
        compatibility["batch_processing"]["dynamic_batching"] = True
        compatibility["batch_processing"]["async_processing"] = sys.version_info >= (3, 7)  # Requires asyncio
        
        return compatibility
    
    def _check_bitsandbytes_availability(self) -> bool:
        """Check if bitsandbytes is available."""
        try:
            import bitsandbytes
            return True
        except ImportError:
            return False
    
    def _check_tensorrt_availability(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt
            return True
        except ImportError:
            return False
    
    def benchmark_system_performance(self, model_size_mb: float = 100) -> Dict[str, Any]:
        """Benchmark system performance for ML workloads."""
        logger.info("Starting system performance benchmark...")
        
        benchmark_results = {
            "cpu_benchmark": self._benchmark_cpu(),
            "memory_benchmark": self._benchmark_memory(),
            "gpu_benchmark": self._benchmark_gpu() if self.specs.gpu_count > 0 else None,
            "overall_score": 0.0
        }
        
        # Calculate overall score
        scores = []
        if benchmark_results["cpu_benchmark"]:
            scores.append(benchmark_results["cpu_benchmark"]["score"])
        if benchmark_results["memory_benchmark"]:
            scores.append(benchmark_results["memory_benchmark"]["score"])
        if benchmark_results["gpu_benchmark"]:
            scores.append(benchmark_results["gpu_benchmark"]["score"])
        
        if scores:
            benchmark_results["overall_score"] = np.mean(scores)
        
        logger.info(f"Benchmark completed. Overall score: {benchmark_results['overall_score']:.2f}")
        return benchmark_results
    
    def _benchmark_cpu(self) -> Optional[Dict[str, Any]]:
        """Benchmark CPU performance."""
        try:
            # Simple CPU benchmark - matrix multiplication
            size = 500
            start_time = time.time()
            
            # Perform multiple matrix multiplications
            for _ in range(5):
                a = torch.randn(size, size)
                b = torch.randn(size, size)
                c = torch.mm(a, b)
            
            cpu_time = time.time() - start_time
            
            # Calculate score (higher is better)
            score = max(0, 100 - cpu_time * 10)
            
            return {
                "score": score,
                "time_seconds": cpu_time,
                "operations_per_second": 5 / cpu_time,
                "gflops": (5 * size**3) / (cpu_time * 1e9)
            }
            
        except Exception as e:
            logger.warning(f"CPU benchmark failed: {e}")
            return None
    
    def _benchmark_memory(self) -> Optional[Dict[str, Any]]:
        """Benchmark memory performance."""
        try:
            # Memory bandwidth benchmark
            size_mb = 100
            data_size = size_mb * 1024 * 1024 // 8  # Convert to float64 elements
            
            # Memory write test
            start_time = time.time()
            data = torch.zeros(data_size)
            for i in range(0, data_size, 1000000):
                end_idx = min(i + 1000000, data_size)
                data[i:end_idx] = torch.randn(end_idx - i)
            write_time = time.time() - start_time
            
            # Memory read test
            start_time = time.time()
            _ = torch.sum(data)
            read_time = time.time() - start_time
            
            # Calculate bandwidth
            total_bytes = size_mb * 2 * 1024 * 1024  # Write + read
            bandwidth_gbps = total_bytes / ((write_time + read_time) * 1e9)
            
            # Calculate score
            score = min(100, bandwidth_gbps * 10)
            
            return {
                "score": score,
                "write_time_seconds": write_time,
                "read_time_seconds": read_time,
                "total_time_seconds": write_time + read_time,
                "bandwidth_gbps": bandwidth_gbps
            }
            
        except Exception as e:
            logger.warning(f"Memory benchmark failed: {e}")
            return None
    
    def _benchmark_gpu(self) -> Optional[Dict[str, Any]]:
        """Benchmark GPU performance."""
        if not self.specs.cuda_available:
            return None
        
        try:
            # GPU benchmark - matrix multiplication with CUDA
            size = 2048
            device = torch.device('cuda')
            
            # Warm up
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            _ = torch.mm(a, b)
            torch.cuda.synchronize()
            
            # Actual benchmark
            start_time = time.time()
            for _ in range(10):
                c = torch.mm(a, b)
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # Calculate performance
            gflops = (10 * size**3) / (gpu_time * 1e9)
            
            # Calculate score based on theoretical max
            # Assume modern GPU can do ~10 TFLOPS
            theoretical_max = 10000  # TFLOPS
            score = min(100, (gflops / theoretical_max) * 100)
            
            return {
                "score": score,
                "time_seconds": gpu_time,
                "gflops": gflops,
                "operations_per_second": 10 / gpu_time,
                "memory_usage_mb": torch.cuda.memory_allocated() / (1024**2)
            }
            
        except Exception as e:
            logger.warning(f"GPU benchmark failed: {e}")
            return None
    
    def save_system_profile(self, filepath: str):
        """Save system profile to file."""
        profile_data = {
            "specs": self.specs.to_dict(),
            "compatibility": self.check_optimization_compatibility(),
            "recommendations": self.get_optimization_recommendations(),
            "benchmark": self.benchmark_system_performance(),
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"System profile saved to {filepath}")


class OptimizationUtils:
    """Utility functions for optimization operations."""
    
    @staticmethod
    def check_model_compatibility(model: nn.Module, optimization_level: str) -> Dict[str, Any]:
        """Check if model is compatible with specified optimization level."""
        compatibility = {
            "compatible": True,
            "warnings": [],
            "recommendations": [],
            "model_analysis": {}
        }
        
        # Analyze model
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        compatibility["model_analysis"] = {
            "total_parameters": total_params,
            "model_size_mb": model_size_mb,
            "model_type": type(model).__name__,
            "requires_grad": sum(p.requires_grad for p in model.parameters())
        }
        
        # Check compatibility based on optimization level
        if optimization_level == "aggressive" and total_params > 1e9:
            compatibility["warnings"].append("Large model may not benefit from aggressive optimization")
            compatibility["recommendations"].append("Consider using balanced optimization level")
        
        if optimization_level == "minimal" and total_params > 1e8:
            compatibility["warnings"].append("Large model may benefit from optimization")
            compatibility["recommendations"].append("Consider using balanced or aggressive optimization")
        
        # Check for quantization compatibility
        has_float32 = any(p.dtype == torch.float32 for p in model.parameters())
        if not has_float32:
            compatibility["warnings"].append("Model already uses non-float32 precision")
            compatibility["recommendations"].append("Quantization may not provide additional benefits")
        
        return compatibility
    
    @staticmethod
    def estimate_optimization_impact(original_model: nn.Module) -> Dict[str, Any]:
        """Estimate the potential impact of optimization techniques."""
        total_params = sum(p.numel() for p in original_model.parameters())
        model_size_mb = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024**2)
        
        estimates = {
            "quantization_impact": {},
            "pruning_impact": {},
            "distillation_impact": {},
            "memory_optimization_impact": {}
        }
        
        # Quantization impact
        estimates["quantization_impact"] = {
            "int8_compression": 4.0,  # Typical INT8 compression
            "int4_compression": 8.0,  # Typical INT4 compression
            "accuracy_loss_estimate": {
                "int8": 0.01,  # 1% accuracy loss
                "int4": 0.05   # 5% accuracy loss
            }
        }
        
        # Pruning impact
        if total_params > 1e7:
            estimates["pruning_impact"] = {
                "potential_sparsity": 0.5,  # 50% sparsity potential
                "compression_ratio": 2.0,
                "speedup_estimate": 1.5,
                "accuracy_impact": 0.02  # 2% accuracy impact
            }
        else:
            estimates["pruning_impact"] = {
                "potential_sparsity": 0.3,
                "compression_ratio": 1.4,
                "speedup_estimate": 1.2,
                "accuracy_impact": 0.01
            }
        
        # Distillation impact
        if total_params > 1e8:
            estimates["distillation_impact"] = {
                "compression_ratio": 3.0,
                "size_reduction_mb": model_size_mb * 0.7,
                "accuracy_retention": 0.95,
                "training_time_hours": 24.0
            }
        else:
            estimates["distillation_impact"] = {
                "compression_ratio": 2.0,
                "size_reduction_mb": model_size_mb * 0.5,
                "accuracy_retention": 0.90,
                "training_time_hours": 8.0
            }
        
        # Memory optimization impact
        estimates["memory_optimization_impact"] = {
            "gradient_checkpointing_savings": min(0.6, model_size_mb / 1000),  # Up to 60% savings
            "cpu_offload_savings": min(0.8, model_size_mb / 500),  # Up to 80% savings
            "performance_impact": 0.2  # 20% performance overhead
        }
        
        return estimates
    
    @staticmethod
    def create_optimization_config_template(output_path: str):
        """Create a template configuration file for optimization."""
        template = {
            "optimization_level": "balanced",
            "preserve_medical_accuracy": True,
            "quantization": {
                "quantization_type": "int8",
                "use_bnb": True,
                "load_in_8bit": False,
                "load_in_4bit": False
            },
            "memory": {
                "enable_gradient_checkpointing": False,
                "enable_cpu_offload": False,
                "enable_disk_offload": False
            },
            "device": {
                "preferred_device": "auto",
                "device_memory_fraction": 0.8
            },
            "batch": {
                "max_batch_size": 32,
                "enable_dynamic_batching": True,
                "batch_timeout": 0.1
            },
            "reduction": {
                "reduction_type": "none",
                "prune_ratio": 0.0
            },
            "validation": {
                "enable_validation": True,
                "accuracy_threshold": 0.95,
                "medical_compliance_required": True
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=2)
        
        logger.info(f"Optimization config template created at {output_path}")
    
    @staticmethod
    def get_system_requirements(model_size_mb: float, optimization_level: str) -> Dict[str, Any]:
        """Calculate system requirements for given model and optimization level."""
        # Base requirements
        base_memory_gb = model_size_mb / 1024  # Convert to GB
        
        # Optimization level multipliers
        level_multipliers = {
            "minimal": {"memory": 1.0, "compute": 1.0},
            "balanced": {"memory": 0.7, "compute": 1.2},
            "aggressive": {"memory": 0.5, "compute": 1.5}
        }
        
        multipliers = level_multipliers.get(optimization_level, level_multipliers["balanced"])
        
        requirements = {
            "minimum_memory_gb": base_memory_gb * multipliers["memory"],
            "recommended_memory_gb": base_memory_gb * multipliers["memory"] * 1.5,
            "gpu_memory_gb": base_memory_gb * multipliers["memory"] * 0.8,  # GPUs need less overhead
            "cpu_cores": 4 if optimization_level == "aggressive" else 2,
            "storage_gb": model_size_mb / 1024 * 2,  # Model + working space
            "network_bandwidth_mbps": 100 if optimization_level == "aggressive" else 50
        }
        
        # Add recommendations
        recommendations = []
        
        if requirements["minimum_memory_gb"] > 16:
            recommendations.append("Consider using model parallelism or aggressive quantization")
        if optimization_level == "aggressive":
            recommendations.append("Use high-performance GPU with ample VRAM")
            recommendations.append("Enable all optimization techniques")
        elif optimization_level == "minimal":
            recommendations.append("Focus on accuracy preservation over performance")
        
        requirements["recommendations"] = recommendations
        
        return requirements
    
    @staticmethod
    def validate_environment() -> Dict[str, Any]:
        """Validate the optimization environment and dependencies."""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "dependencies": {},
            "recommendations": []
        }
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            validation["errors"].append("Python 3.8+ required")
            validation["valid"] = False
        elif python_version < (3, 9):
            validation["warnings"].append("Python 3.9+ recommended for best performance")
        
        # Check PyTorch
        try:
            import torch
            validation["dependencies"]["torch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
            
            if not torch.cuda.is_available():
                validation["warnings"].append("CUDA not available - GPU acceleration disabled")
        except ImportError:
            validation["errors"].append("PyTorch not installed")
            validation["valid"] = False
        
        # Check optional dependencies
        optional_deps = {
            "bitsandbytes": "Advanced quantization support",
            "tensorrt": "High-performance inference optimization",
            "transformers": "Hugging Face model support",
            "accelerate": "Distributed training support"
        }
        
        for dep_name, description in optional_deps.items():
            try:
                dep = __import__(dep_name)
                validation["dependencies"][dep_name] = {"available": True}
            except ImportError:
                validation["dependencies"][dep_name] = {
                    "available": False, 
                    "description": description,
                    "install_command": f"pip install {dep_name}"
                }
                validation["recommendations"].append(f"Install {dep_name} for {description}")
        
        # System-specific recommendations
        system = platform.system()
        if system == "Darwin":
            validation["recommendations"].append("Consider using Metal Performance Shaders on macOS")
        elif system == "Windows":
            validation["recommendations"].append("Some optimizations may have limited support on Windows")
        
        return validation
    
    @staticmethod
    def create_backup(original_path: str, backup_dir: str = None) -> str:
        """Create backup of model or configuration file."""
        if backup_dir is None:
            backup_dir = os.path.dirname(original_path)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.splitext(os.path.basename(original_path))[0]}_backup_{timestamp}{os.path.splitext(original_path)[1]}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        shutil.copy2(original_path, backup_path)
        logger.info(f"Backup created: {backup_path}")
        
        return backup_path
    
    @staticmethod
    def cleanup_temp_files(temp_dir: str):
        """Clean up temporary files and cache."""
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")
    
    @staticmethod
    def format_file_size(size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        
        return f"{s} {size_names[i]}"
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"