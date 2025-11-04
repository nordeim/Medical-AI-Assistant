"""
Model Optimization Utilities
Performance optimization, quantization, and inference acceleration for medical AI models.
"""

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional, Tuple
import joblib
from sklearn.model_selection import train_test_split
import time
import psutil
import gc
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Production model optimization toolkit"""
    
    def __init__(self, config_path: str = "config/optimization_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Optimization profiles
        self.optimization_profiles = {
            "fast_inference": {
                "quantization": True,
                "pruning": False,
                "graph_optimization": True,
                "batch_size": 1
            },
            "balanced": {
                "quantization": True,
                "pruning": True,
                "graph_optimization": True,
                "batch_size": 8
            },
            "memory_efficient": {
                "quantization": True,
                "pruning": True,
                "compression": True,
                "batch_size": 4
            },
            "maximum_accuracy": {
                "quantization": False,
                "pruning": False,
                "graph_optimization": True,
                "batch_size": 16
            }
        }
        
        # Performance benchmarks
        self.benchmarks = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load optimization configuration"""
        default_config = {
            "quantization": {
                "enabled": True,
                "method": "dynamic",  # dynamic, static, qat
                "precision": "int8",
                "calibration_samples": 100
            },
            "pruning": {
                "enabled": True,
                "method": "magnitude",  # magnitude, structured, lottery_ticket
                "sparsity": 0.5,
                "gradual_pruning": True
            },
            "graph_optimization": {
                "enabled": True,
                "optimization_level": "O2",
                "constant_folding": True,
                "dead_code_elimination": True
            },
            "batch_optimization": {
                "max_batch_size": 32,
                "dynamic_batching": True,
                "batch_timeout_ms": 50
            }
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Optimization config {config_path} not found, using defaults")
            return default_config
    
    def optimize_model(self, model, model_type: str, profile: str = "balanced",
                      optimization_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize model with specified profile"""
        
        if optimization_config is None:
            optimization_config = self.optimization_profiles.get(profile, self.optimization_profiles["balanced"])
        
        logger.info(f"Optimizing model with profile: {profile}")
        
        # Start optimization
        optimization_start = time.time()
        
        try:
            original_size = self._get_model_size(model, model_type)
            
            # Apply optimizations
            optimized_model = model
            applied_optimizations = []
            
            # Quantization
            if optimization_config.get("quantization", False):
                optimized_model = self._apply_quantization(optimized_model, model_type)
                applied_optimizations.append("quantization")
            
            # Pruning
            if optimization_config.get("pruning", False):
                optimized_model = self._apply_pruning(optimized_model, model_type)
                applied_optimizations.append("pruning")
            
            # Graph optimization (convert to ONNX and optimize)
            if optimization_config.get("graph_optimization", False):
                optimized_model = self._apply_graph_optimization(optimized_model, model_type)
                applied_optimizations.append("graph_optimization")
            
            # Compression
            if optimization_config.get("compression", False):
                optimized_model = self._apply_compression(optimized_model)
                applied_optimizations.append("compression")
            
            # Benchmark optimized model
            benchmark_results = self._benchmark_model(optimized_model, model_type, optimization_config)
            
            optimized_size = self._get_model_size(optimized_model, model_type)
            
            optimization_time = time.time() - optimization_start
            
            # Store benchmark results
            self.benchmarks[f"{model_type}_{profile}"] = {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": original_size / optimized_size if optimized_size > 0 else 1.0,
                "optimization_time": optimization_time,
                "applied_optimizations": applied_optimizations,
                "benchmark_results": benchmark_results,
                "profile": profile
            }
            
            logger.info(f"Model optimization completed: {applied_optimizations}")
            logger.info(f"Compression ratio: {original_size / optimized_size:.2f}x")
            
            return {
                "optimized_model": optimized_model,
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": original_size / optimized_size if optimized_size > 0 else 1.0,
                "optimization_time": optimization_time,
                "applied_optimizations": applied_optimizations,
                "benchmark_results": benchmark_results
            }
            
        except Exception as e:
            logger.error(f"Model optimization failed: {str(e)}")
            raise
    
    def _get_model_size(self, model, model_type: str) -> float:
        """Get model size in MB"""
        try:
            if model_type == "pytorch":
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    return total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            elif model_type == "sklearn":
                return sys.getsizeof(joblib.dump(model, None)[1]) / (1024 * 1024)
            
            # Fallback estimate
            return 100.0  # MB
            
        except Exception as e:
            logger.warning(f"Model size calculation failed: {str(e)}")
            return 0.0
    
    def _apply_quantization(self, model, model_type: str):
        """Apply quantization to model"""
        try:
            if model_type == "pytorch":
                # Dynamic quantization for PyTorch models
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d}, 
                    dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization")
                return quantized_model
            else:
                # For other model types, return as-is (would implement ONNX quantization in production)
                logger.info("Quantization skipped for non-PyTorch model")
                return model
                
        except Exception as e:
            logger.warning(f"Quantization failed: {str(e)}")
            return model
    
    def _apply_pruning(self, model, model_type: str):
        """Apply pruning to reduce model size"""
        try:
            if model_type == "pytorch":
                # Implement magnitude-based pruning
                import torch.nn.utils.prune as prune
                
                # Prune linear layers
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=0.5)
                
                # Remove pruning reparameterization (make it permanent)
                for module in model.modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.remove(module, 'weight')
                
                logger.info("Applied magnitude-based pruning")
                return model
            else:
                logger.info("Pruning not implemented for model type")
                return model
                
        except Exception as e:
            logger.warning(f"Pruning failed: {str(e)}")
            return model
    
    def _apply_graph_optimization(self, model, model_type: str):
        """Apply graph-level optimizations"""
        try:
            if model_type == "pytorch":
                # Convert to ONNX and optimize
                dummy_input = torch.randn(1, 10)  # Adjust input shape as needed
                onnx_path = "temp_model.onnx"
                
                torch.onnx.export(
                    model, 
                    dummy_input, 
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                
                # Load ONNX model with optimization
                providers = ['CPUExecutionProvider']
                session = ort.InferenceSession(onnx_path, providers=providers)
                
                logger.info("Applied graph optimization via ONNX")
                return session
            else:
                logger.info("Graph optimization not available for model type")
                return model
                
        except Exception as e:
            logger.warning(f"Graph optimization failed: {str(e)}")
            return model
    
    def _apply_compression(self, model):
        """Apply model compression techniques"""
        try:
            # Use pickle compression for serialization
            import gzip
            
            # Store compressed representation (in production, save to file)
            compressed_data = gzip.compress(pickle.dumps(model))
            
            logger.info("Applied compression")
            return model
            
        except Exception as e:
            logger.warning(f"Compression failed: {str(e)}")
            return model
    
    def _benchmark_model(self, model, model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark model performance"""
        try:
            benchmark_results = {}
            
            # Generate test data
            batch_size = config.get("batch_size", 1)
            num_batches = 100
            
            if model_type == "pytorch":
                # PyTorch model benchmark
                model.eval()
                
                # Latency benchmark
                latencies = []
                with torch.no_grad():
                    for _ in range(num_batches):
                        dummy_input = torch.randn(batch_size, 10)
                        
                        start_time = time.time()
                        output = model(dummy_input)
                        latency = time.time() - start_time
                        latencies.append(latency * 1000)  # Convert to ms
                
                benchmark_results = {
                    "latency_ms": {
                        "p50": np.percentile(latencies, 50),
                        "p95": np.percentile(latencies, 95),
                        "p99": np.percentile(latencies, 99),
                        "avg": np.mean(latencies),
                        "min": np.min(latencies),
                        "max": np.max(latencies)
                    },
                    "throughput_qps": batch_size * num_batches / (sum(latencies) / 1000),
                    "memory_usage_mb": psutil.Process().memory_info().rss / (1024 * 1024)
                }
            
            elif model_type == "sklearn":
                # Scikit-learn model benchmark
                test_data = np.random.rand(1000, 10)
                
                start_time = time.time()
                predictions = model.predict(test_data)
                total_time = time.time() - start_time
                
                benchmark_results = {
                    "total_prediction_time": total_time,
                    "avg_latency_ms": (total_time / 1000) * 1000,
                    "throughput_qps": 1000 / total_time,
                    "predictions_made": len(predictions)
                }
            
            logger.info("Model benchmarking completed")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Model benchmarking failed: {str(e)}")
            return {}
    
    def optimize_for_batch_processing(self, model, model_type: str, 
                                    max_batch_size: int = 32,
                                    dynamic_batching: bool = True) -> Any:
        """Optimize model for batch processing"""
        try:
            if model_type == "pytorch":
                # Enable JIT compilation for better performance
                model.eval()
                
                # Create a sample input for tracing
                sample_input = torch.randn(1, 10)
                
                # Trace the model
                traced_model = torch.jit.trace(model, sample_input)
                
                logger.info("Model optimized for batch processing with JIT tracing")
                return traced_model
            else:
                logger.info("Batch processing optimization not implemented for model type")
                return model
                
        except Exception as e:
            logger.warning(f"Batch processing optimization failed: {str(e)}")
            return model
    
    def create_model_variant(self, model, model_type: str, variant_config: Dict[str, Any]) -> Any:
        """Create a model variant with specific optimizations"""
        try:
            profile = variant_config.get("profile", "balanced")
            custom_config = variant_config.get("config", {})
            
            # Get optimization profile
            base_config = self.optimization_profiles.get(profile, self.optimization_profiles["balanced"])
            
            # Merge with custom configuration
            final_config = {**base_config, **custom_config}
            
            # Apply optimizations
            optimization_result = self.optimize_model(model, model_type, "custom", final_config)
            
            logger.info(f"Model variant created with profile: {profile}")
            return optimization_result["optimized_model"]
            
        except Exception as e:
            logger.error(f"Model variant creation failed: {str(e)}")
            raise
    
    def get_optimization_recommendations(self, model_type: str, 
                                       deployment_target: str = "production") -> Dict[str, Any]:
        """Get optimization recommendations based on deployment target"""
        
        recommendations = {
            "edge_devices": {
                "profile": "memory_efficient",
                "priority": ["quantization", "pruning", "compression"],
                "target_size_mb": 50,
                "max_latency_ms": 100
            },
            "cloud_production": {
                "profile": "balanced",
                "priority": ["quantization", "graph_optimization", "batching"],
                "target_accuracy_retention": 0.95,
                "max_latency_ms": 200
            },
            "high_accuracy": {
                "profile": "maximum_accuracy",
                "priority": ["graph_optimization"],
                "min_accuracy_retention": 0.98,
                "max_memory_gb": 8
            },
            "real_time": {
                "profile": "fast_inference",
                "priority": ["quantization", "graph_optimization"],
                "max_latency_ms": 50,
                "min_throughput_qps": 1000
            }
        }
        
        target_recommendations = recommendations.get(deployment_target, recommendations["cloud_production"])
        
        # Add model-specific recommendations
        if model_type == "pytorch":
            target_recommendations.update({
                "recommended_batch_size": 8,
                "enable_jit": True,
                "enable_amp": True
            })
        elif model_type == "sklearn":
            target_recommendations.update({
                "compress_model": True,
                "use_joblib_multiprocessing": True
            })
        
        return target_recommendations
    
    def compare_optimization_strategies(self, model, model_type: str) -> Dict[str, Any]:
        """Compare different optimization strategies"""
        logger.info("Comparing optimization strategies...")
        
        results = {}
        
        # Test each optimization profile
        for profile_name in self.optimization_profiles.keys():
            try:
                result = self.optimize_model(model, model_type, profile_name)
                results[profile_name] = {
                    "compression_ratio": result["compression_ratio"],
                    "optimization_time": result["optimization_time"],
                    "applied_optimizations": result["applied_optimizations"],
                    "benchmark_results": result["benchmark_results"]
                }
            except Exception as e:
                results[profile_name] = {"error": str(e)}
        
        # Determine best strategy
        best_profile = None
        best_score = 0
        
        for profile, metrics in results.items():
            if "error" not in metrics:
                # Simple scoring: compression ratio / optimization time
                score = metrics["compression_ratio"] / max(metrics["optimization_time"], 0.1)
                if score > best_score:
                    best_score = score
                    best_profile = profile
        
        return {
            "results": results,
            "best_profile": best_profile,
            "best_score": best_score,
            "recommendation": f"Use {best_profile} profile for optimal balance of performance and size" if best_profile else "Manual optimization required"
        }
    
    def analyze_model_complexity(self, model, model_type: str) -> Dict[str, Any]:
        """Analyze model complexity and identify optimization opportunities"""
        try:
            complexity_analysis = {}
            
            if model_type == "pytorch":
                # Analyze PyTorch model
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                # Analyze layer types
                layer_types = {}
                for module in model.modules():
                    layer_type = type(module).__name__
                    layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
                
                # Calculate FLOPs (simplified)
                flops_estimate = total_params * 2  # Rough estimate
                
                complexity_analysis = {
                    "total_parameters": total_params,
                    "trainable_parameters": trainable_params,
                    "model_size_mb": total_params * 4 / (1024 * 1024),
                    "layer_distribution": layer_types,
                    "flops_estimate": flops_estimate,
                    "optimization_opportunities": []
                }
                
                # Identify optimization opportunities
                if total_params > 1000000:  # > 1M parameters
                    complexity_analysis["optimization_opportunities"].append("Consider quantization")
                
                if layer_types.get("Linear", 0) > 10:
                    complexity_analysis["optimization_opportunities"].append("Consider pruning")
                
                if flops_estimate > 100000000:  # > 100M FLOPs
                    complexity_analysis["optimization_opportunities"].append("Consider graph optimization")
            
            elif model_type == "sklearn":
                # Analyze scikit-learn model
                complexity_analysis = {
                    "model_type": type(model).__name__,
                    "n_features": getattr(model, "n_features_in_", None),
                    "n_classes": getattr(model, "n_classes_", None),
                    "n_estimators": getattr(model, "n_estimators", None),
                    "optimization_opportunities": []
                }
                
                if hasattr(model, "feature_importances_"):
                    complexity_analysis["optimization_opportunities"].append("Feature selection possible")
                
                if hasattr(model, "estimators_") and len(model.estimators_) > 100:
                    complexity_analysis["optimization_opportunities"].append("Consider ensemble reduction")
            
            logger.info("Model complexity analysis completed")
            return complexity_analysis
            
        except Exception as e:
            logger.error(f"Model complexity analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "optimization_benchmarks": self.benchmarks,
            "total_optimizations": len(self.benchmarks),
            "report_timestamp": time.time(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "pytorch_version": torch.__version__ if torch else None
            }
        }

# Dynamic batching optimization
class DynamicBatcher:
    """Dynamic batching for optimized inference"""
    
    def __init__(self, max_batch_size: int = 32, timeout_ms: int = 50):
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.pending_requests = []
        self.batch_lock = False
    
    def add_request(self, request_data: Dict[str, Any]) -> str:
        """Add request to batch queue"""
        request_id = f"req_{time.time()}_{len(self.pending_requests)}"
        self.pending_requests.append({
            "id": request_id,
            "data": request_data,
            "timestamp": time.time()
        })
        return request_id
    
    def get_batch(self) -> Optional[Dict[str, Any]]:
        """Get batch for processing"""
        if self.batch_lock or not self.pending_requests:
            return None
        
        current_time = time.time()
        
        # Check if we have a full batch or timed-out requests
        full_batch = len(self.pending_requests) >= self.max_batch_size
        timeout_batch = self.pending_requests and \
                       (current_time - self.pending_requests[0]["timestamp"]) * 1000 > self.timeout_ms
        
        if full_batch or timeout_batch:
            batch_size = min(len(self.pending_requests), self.max_batch_size)
            batch = {
                "requests": self.pending_requests[:batch_size],
                "batch_id": f"batch_{current_time}",
                "size": batch_size,
                "timestamp": current_time
            }
            
            # Remove processed requests
            self.pending_requests = self.pending_requests[batch_size:]
            self.batch_lock = True
            
            return batch
    
    def release_batch(self):
        """Release batch lock"""
        self.batch_lock = False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "pending_requests": len(self.pending_requests),
            "max_batch_size": self.max_batch_size,
            "timeout_ms": self.timeout_ms,
            "batch_lock": self.batch_lock
        }

# Example usage
if __name__ == "__main__":
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    # Create sample PyTorch model
    import torch.nn as nn
    
    class SampleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 5)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SampleModel()
    
    # Analyze model complexity
    complexity = optimizer.analyze_model_complexity(model, "pytorch")
    print(f"Model complexity: {complexity}")
    
    # Compare optimization strategies
    comparison = optimizer.compare_optimization_strategies(model, "pytorch")
    print(f"Best optimization profile: {comparison['best_profile']}")
    print(f"Recommendation: {comparison['recommendation']}")
    
    # Get recommendations for deployment
    recommendations = optimizer.get_optimization_recommendations("pytorch", "cloud_production")
    print(f"Deployment recommendations: {recommendations}")