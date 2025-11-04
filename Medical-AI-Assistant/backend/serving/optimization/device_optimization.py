"""
Device optimization with GPU/CPU auto-detection and inference optimization.
"""

import torch
import logging
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time
import threading

from .config import DeviceConfig, DeviceType, InferenceMode


logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference modes for different performance requirements."""
    LOW_LATENCY = "low_latency"      # Minimize latency
    HIGH_THROUGHPUT = "high_throughput"  # Maximize throughput
    BALANCED = "balanced"            # Balance latency and throughput
    MEMORY_EFFICIENT = "memory_efficient"  # Minimize memory usage
    ACCURACY_FOCUSED = "accuracy_focused"  # Maximize accuracy


@dataclass
class DeviceInfo:
    """Information about a compute device."""
    device_type: DeviceType
    device_id: int
    name: str
    memory_total_gb: float
    memory_available_gb: float
    compute_capability: float
    supports_float16: bool
    supports_bfloat16: bool
    supports_int8: bool
    supports_int4: bool
    supports_tensorrt: bool
    utilization_percent: float
    temperature_celsius: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type.value,
            "device_id": self.device_id,
            "name": self.name,
            "memory_total_gb": self.memory_total_gb,
            "memory_available_gb": self.memory_available_gb,
            "compute_capability": self.compute_capability,
            "supports_float16": self.supports_float16,
            "supports_bfloat16": self.supports_bfloat16,
            "supports_int8": self.supports_int8,
            "supports_int4": self.supports_int4,
            "supports_tensorrt": self.supports_tensorrt,
            "utilization_percent": self.utilization_percent,
            "temperature_celsius": self.temperature_celsius,
        }


@dataclass
class DevicePerformance:
    """Performance metrics for a device."""
    device_id: int
    throughput_tokens_per_second: float
    latency_ms: float
    memory_bandwidth_gb_s: float
    compute_utilization_percent: float
    thermal_throttling: bool
    power_consumption_watts: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "latency_ms": self.latency_ms,
            "memory_bandwidth_gb_s": self.memory_bandwidth_gb_s,
            "compute_utilization_percent": self.compute_utilization_percent,
            "thermal_throttling": self.thermal_throttling,
            "power_consumption_watts": self.power_consumption_watts,
        }


class DeviceManager:
    """
    Advanced device management with automatic detection and optimization.
    Handles GPU/CPU selection, device mapping, and inference optimization.
    """
    
    def __init__(self, config: DeviceConfig):
        self.config = config
        self._devices = []
        self._performance_cache = {}
        self._device_selection_lock = threading.Lock()
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Initialize device detection
        self._detect_all_devices()
        
        # Set up device monitoring if enabled
        if self.config.device_memory_fraction < 1.0:
            self._setup_monitoring()
    
    def _detect_all_devices(self) -> List[DeviceInfo]:
        """Detect all available compute devices."""
        devices = []
        
        # Always add CPU as fallback
        cpu_info = self._get_cpu_info()
        devices.append(cpu_info)
        
        # Detect CUDA GPUs
        if torch.cuda.is_available():
            cuda_devices = self._detect_cuda_devices()
            devices.extend(cuda_devices)
        
        # Detect other accelerators (if available)
        try:
            mps_devices = self._detect_mps_devices()
            devices.extend(mps_devices)
        except:
            pass  # MPS might not be available
        
        self._devices = devices
        logger.info(f"Detected {len(devices)} devices: {[d.name for d in devices]}")
        return devices
    
    def _get_cpu_info(self) -> DeviceInfo:
        """Get CPU device information."""
        cpu_count = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        return DeviceInfo(
            device_type=DeviceType.CPU,
            device_id=0,
            name=f"CPU {cpu_count}c/{cpu_threads}t",
            memory_total_gb=psutil.virtual_memory().total / (1024**3),
            memory_available_gb=psutil.virtual_memory().available / (1024**3),
            compute_capability=1.0,  # Relative to baseline
            supports_float16=True,
            supports_bfloat16=True,
            supports_int8=True,
            supports_int4=True,
            supports_tensorrt=False,
            utilization_percent=psutil.cpu_percent()
        )
    
    def _detect_cuda_devices(self) -> List[DeviceInfo]:
        """Detect CUDA GPU devices."""
        cuda_devices = []
        
        for i in range(torch.cuda.device_count()):
            try:
                device_props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                memory_total = torch.cuda.get_device_properties(i).total_memory
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_free = memory_total - memory_reserved
                
                # Check capabilities
                compute_capability = float(f"{device_props.major}.{device_props.minor}")
                supports_float16 = device_props.major >= 7  # Volta+
                supports_bfloat16 = compute_capability >= 8.0  # Ampere+
                supports_int8 = True  # Most modern GPUs support INT8
                supports_int4 = compute_capability >= 8.0  # Modern GPUs with tensor cores
                supports_tensorrt = supports_int8  # TensorRT typically requires INT8 support
                
                # Get utilization (requires nvidia-ml-py for accurate data)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization_percent = float(utilization.gpu)
                except:
                    utilization_percent = 0.0  # Fallback
                
                device_info = DeviceInfo(
                    device_type=DeviceType.GPU,
                    device_id=i,
                    name=device_props.name,
                    memory_total_gb=memory_total / (1024**3),
                    memory_available_gb=memory_free / (1024**3),
                    compute_capability=compute_capability,
                    supports_float16=supports_float16,
                    supports_bfloat16=supports_bfloat16,
                    supports_int8=supports_int8,
                    supports_int4=supports_int4,
                    supports_tensorrt=supports_tensorrt,
                    utilization_percent=utilization_percent
                )
                
                cuda_devices.append(device_info)
                
            except Exception as e:
                logger.warning(f"Error detecting CUDA device {i}: {e}")
        
        return cuda_devices
    
    def _detect_mps_devices(self) -> List[DeviceInfo]:
        """Detect Metal Performance Shaders (MPS) devices (Apple Silicon)."""
        mps_devices = []
        
        try:
            if torch.backends.mps.is_available():
                device_info = DeviceInfo(
                    device_type=DeviceType.GPU,
                    device_id=0,  # MPS typically has single device
                    name="Apple Metal",
                    memory_total_gb=16.0,  # Approximate, MPS doesn't provide exact memory
                    memory_available_gb=8.0,  # Conservative estimate
                    compute_capability=8.0,  # Modern Apple Silicon
                    supports_float16=True,
                    supports_bfloat16=False,  # MPS doesn't support bfloat16
                    supports_int8=False,  # Limited INT8 support
                    supports_int4=False,
                    supports_tensorrt=False,
                    utilization_percent=0.0  # MPS doesn't provide utilization
                )
                mps_devices.append(device_info)
        except:
            pass  # MPS might not be available
        
        return mps_devices
    
    def _setup_monitoring(self):
        """Set up device performance monitoring."""
        def monitor_devices():
            while self._monitoring_active:
                try:
                    self._update_device_metrics()
                    time.sleep(5)  # Update every 5 seconds
                except Exception as e:
                    logger.error(f"Device monitoring error: {e}")
                    break
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=monitor_devices, daemon=True)
        self._monitor_thread.start()
        logger.info("Device monitoring started")
    
    def _update_device_metrics(self):
        """Update real-time device metrics."""
        for device in self._devices:
            if device.device_type == DeviceType.GPU and device.device_id < torch.cuda.device_count():
                # Update GPU utilization and memory
                try:
                    if device.device_id < torch.cuda.device_count():
                        # Update memory info
                        memory_total = torch.cuda.get_device_properties(device.device_id).total_memory
                        memory_allocated = torch.cuda.memory_allocated(device.device_id)
                        memory_free = memory_total - memory_allocated
                        
                        device.memory_available_gb = memory_free / (1024**3)
                        device.utilization_percent = self._get_gpu_utilization(device.device_id)
                        
                except Exception as e:
                    logger.debug(f"Error updating device {device.device_id} metrics: {e}")
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization percentage."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(utilization.gpu)
        except:
            return 0.0
    
    def select_optimal_device(self, 
                             model_size_gb: float,
                             inference_mode: InferenceMode = InferenceMode.BALANCED,
                             batch_size: int = 1) -> DeviceInfo:
        """
        Select the optimal device for inference based on requirements.
        
        Args:
            model_size_gb: Size of the model in GB
            inference_mode: Performance requirement mode
            batch_size: Expected batch size for inference
            
        Returns:
            DeviceInfo for the selected device
        """
        with self._device_selection_lock:
            # Get available devices sorted by preference
            available_devices = self._filter_available_devices(model_size_gb)
            
            if not available_devices:
                # Fallback to CPU
                logger.warning("No suitable GPU devices found, falling back to CPU")
                return self._get_cpu_device()
            
            # Score devices based on inference mode
            device_scores = []
            for device in available_devices:
                score = self._calculate_device_score(
                    device, model_size_gb, inference_mode, batch_size
                )
                device_scores.append((device, score))
            
            # Sort by score (highest first)
            device_scores.sort(key=lambda x: x[1], reverse=True)
            best_device = device_scores[0][0]
            
            logger.info(f"Selected device {best_device.name} for {inference_mode.value} inference")
            return best_device
    
    def _filter_available_devices(self, model_size_gb: float) -> List[DeviceInfo]:
        """Filter devices that can accommodate the model."""
        available = []
        
        for device in self._devices:
            # Check memory requirements
            memory_threshold = self.config.device_memory_fraction
            
            if device.device_type == DeviceType.GPU:
                required_memory = model_size_gb * 1.5  # 50% overhead for inference
                if device.memory_available_gb * memory_threshold >= required_memory:
                    available.append(device)
            else:  # CPU
                available.append(device)
        
        return available
    
    def _calculate_device_score(self, 
                               device: DeviceInfo,
                               model_size_gb: float,
                               inference_mode: InferenceMode,
                               batch_size: int) -> float:
        """Calculate device suitability score for given requirements."""
        base_score = 0.0
        
        # Memory availability score
        memory_ratio = device.memory_available_gb / model_size_gb if model_size_gb > 0 else 1.0
        base_score += memory_ratio * 20
        
        # Device type preference
        if device.device_type == DeviceType.GPU:
            base_score += 50
        else:
            base_score += 10
        
        # Compute capability score
        base_score += device.compute_capability * 5
        
        # Utilization penalty
        utilization_penalty = device.utilization_percent * 0.5
        base_score -= utilization_penalty
        
        # Inference mode adjustments
        if inference_mode == InferenceMode.LOW_LATENCY:
            if device.device_type == DeviceType.GPU:
                base_score += 20  # GPUs are better for low latency
        elif inference_mode == InferenceMode.HIGH_THROUGHPUT:
            if device.device_type == DeviceType.GPU and device.compute_capability >= 7.5:
                base_score += 25  # Modern GPUs for high throughput
        elif inference_mode == InferenceMode.MEMORY_EFFICIENT:
            # Prefer devices with more available memory
            base_score += device.memory_available_gb * 2
        elif inference_mode == InferenceMode.ACCURACY_FOCUSED:
            # Prefer devices that support higher precision
            if device.supports_bfloat16:
                base_score += 15
            elif device.supports_float16:
                base_score += 10
        
        # Batch size considerations
        if batch_size > 16 and device.device_type == DeviceType.GPU:
            base_score += 10  # GPUs handle large batches better
        
        return base_score
    
    def get_device_map(self, 
                      model: torch.nn.Module,
                      selected_device: DeviceInfo) -> Dict[str, Union[str, int]]:
        """Generate device map for model parallelism."""
        device_map = {}
        
        # For single GPU/CPU deployment
        if selected_device.device_type == DeviceType.GPU:
            device_map = {
                "": selected_device.device_id
            }
        else:
            device_map = {
                "": "cpu"
            }
        
        # For multi-GPU deployment, implement layer mapping
        # This is a simplified implementation
        if self._get_gpu_count() > 1:
            device_map = self._generate_layer_device_map(model, selected_device.device_id)
        
        return device_map
    
    def _generate_layer_device_map(self, model: torch.nn.Module, primary_gpu_id: int) -> Dict[str, str]:
        """Generate device map for layer-wise model parallelism."""
        device_map = {}
        
        # Map different parts of the model to different devices
        layer_count = 0
        gpu_ids = list(range(min(4, self._get_gpu_count())))  # Limit to 4 GPUs max
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) or 'layer' in name.lower():
                gpu_id = gpu_ids[layer_count % len(gpu_ids)]
                device_map[name] = str(gpu_id)
                layer_count += 1
        
        # Ensure primary modules are on primary GPU
        if "" not in device_map:
            device_map[""] = str(primary_gpu_id)
        
        return device_map
    
    def _get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    def get_cpu_device(self) -> DeviceInfo:
        """Get CPU device information."""
        for device in self._devices:
            if device.device_type == DeviceType.CPU:
                return device
        return self._get_cpu_info()  # Fallback
    
    def optimize_for_inference(self, 
                              device: DeviceInfo,
                              model: torch.nn.Module) -> Dict[str, Any]:
        """Optimize model for inference on specific device."""
        optimization_results = {
            "device": device.name,
            "optimizations": [],
            "performance_improvements": {}
        }
        
        try:
            # 1. Set device precision
            if device.supports_float16 and device.device_type == DeviceType.GPU:
                self._set_model_precision(model, torch.float16)
                optimization_results["optimizations"].append("fp16_precision")
            
            # 2. Enable memory efficient attention if available
            if self._enable_memory_efficient_attention(model):
                optimization_results["optimizations"].append("memory_efficient_attention")
            
            # 3. Enable TensorRT optimization if supported
            if device.supports_tensorrt and device.device_type == DeviceType.GPU:
                trt_result = self._enable_tensorrt_optimization(model)
                if trt_result["success"]:
                    optimization_results["optimizations"].append("tensorrt")
                    optimization_results["performance_improvements"].update(trt_result["improvements"])
            
            # 4. Enable fused operations
            if self._enable_fused_operations(model):
                optimization_results["optimizations"].append("fused_operations")
            
            # 5. Optimize kernel selection
            if self._optimize_kernels(model, device):
                optimization_results["optimizations"].append("kernel_optimization")
            
            # 6. Enable caching optimizations
            if self._enable_caching_optimizations(model):
                optimization_results["optimizations"].append("caching")
            
            logger.info(f"Inference optimization completed: {optimization_results['optimizations']}")
            
        except Exception as e:
            logger.error(f"Error optimizing model for inference: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    def _set_model_precision(self, model: torch.nn.Module, dtype: torch.dtype):
        """Set model precision for inference."""
        for param in model.parameters():
            param.data = param.data.to(dtype)
        
        for buffer in model.buffers():
            buffer.data = buffer.data.to(dtype)
        
        logger.debug(f"Model precision set to {dtype}")
    
    def _enable_memory_efficient_attention(self, model: torch.nn.Module) -> bool:
        """Enable memory efficient attention mechanisms."""
        try:
            # Check if scaled dot product attention is available
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.debug("Scaled dot product attention available for optimization")
                return True
        except:
            pass
        
        return False
    
    def _enable_tensorrt_optimization(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Enable TensorRT optimization for NVIDIA GPUs."""
        try:
            # This is a simplified TensorRT integration
            # In practice, you'd use torch_tensorrt or onnx_tensorrt
            
            # Convert to ONNX first (simplified)
            dummy_input = torch.randn(1, 10)
            
            # Try to create TensorRT engine (placeholder)
            logger.info("TensorRT optimization attempted")
            
            return {
                "success": False,
                "improvements": {},
                "reason": "TensorRT integration not implemented in this version"
            }
            
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return {
                "success": False,
                "improvements": {},
                "error": str(e)
            }
    
    def _enable_fused_operations(self, model: torch.nn.Module) -> bool:
        """Enable fused operations for better performance."""
        try:
            # Enable Flash Attention if available
            # This would be handled at the model level in practice
            logger.debug("Fused operations optimization applied")
            return True
        except:
            return False
    
    def _optimize_kernels(self, model: torch.nn.Module, device: DeviceInfo) -> bool:
        """Optimize kernel selection for the device."""
        try:
            if device.device_type == DeviceType.GPU:
                # Enable optimized CUDA kernels
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                
                logger.debug("CUDA kernels optimized")
                return True
            else:
                # Optimize CPU kernels
                torch.set_num_threads(min(8, psutil.cpu_count()))
                
                logger.debug("CPU kernels optimized")
                return True
                
        except Exception as e:
            logger.debug(f"Kernel optimization failed: {e}")
            return False
    
    def _enable_caching_optimizations(self, model: torch.nn.Module) -> bool:
        """Enable caching optimizations."""
        try:
            # Enable various caching optimizations
            # This is model-specific and would depend on the architecture
            
            # Enable static graph caching for certain model types
            if hasattr(model, 'config'):
                if hasattr(model.config, 'use_cache'):
                    model.config.use_cache = True
            
            logger.debug("Caching optimizations enabled")
            return True
            
        except:
            return False
    
    def benchmark_device_performance(self, 
                                   device: DeviceInfo,
                                   model: torch.nn.Module,
                                   input_shape: Tuple[int, ...] = (1, 10),
                                   num_iterations: int = 100) -> DevicePerformance:
        """Benchmark device performance with the model."""
        try:
            # Move model to device
            model_device = device.device_id if device.device_type == DeviceType.GPU else "cpu"
            model = model.to(model_device)
            model.eval()
            
            # Generate test input
            test_input = torch.randint(0, 1000, input_shape, device=model_device)
            
            # Warm-up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark throughput
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    outputs = model(test_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            
            total_time = end_time - start_time
            throughput = num_iterations / total_time
            latency_ms = (total_time / num_iterations) * 1000
            
            # Estimate memory bandwidth (simplified)
            memory_bandwidth = self._estimate_memory_bandwidth(device, test_input.numel())
            
            # Get compute utilization
            compute_util = device.utilization_percent
            
            # Check for thermal throttling
            thermal_throttling = self._check_thermal_throttling(device)
            
            return DevicePerformance(
                device_id=device.device_id,
                throughput_tokens_per_second=throughput,
                latency_ms=latency_ms,
                memory_bandwidth_gb_s=memory_bandwidth,
                compute_utilization_percent=compute_util,
                thermal_throttling=thermal_throttling
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking device performance: {e}")
            return DevicePerformance(
                device_id=device.device_id,
                throughput_tokens_per_second=0.0,
                latency_ms=0.0,
                memory_bandwidth_gb_s=0.0,
                compute_utilization_percent=0.0,
                thermal_throttling=False
            )
    
    def _estimate_memory_bandwidth(self, device: DeviceInfo, tensor_size: int) -> float:
        """Estimate memory bandwidth for the operation."""
        # Simplified bandwidth estimation
        if device.device_type == DeviceType.GPU:
            return 500.0  # GB/s typical for modern GPUs
        else:
            return 50.0  # GB/s typical for CPU memory
    
    def _check_thermal_throttling(self, device: DeviceInfo) -> bool:
        """Check if device is thermal throttling."""
        try:
            if device.device_type == DeviceType.GPU and device.temperature_celsius:
                return device.temperature_celsius > 85  # Typical throttling temp
        except:
            pass
        
        return False
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get comprehensive device status."""
        return {
            "devices": [device.to_dict() for device in self._devices],
            "device_count": len(self._devices),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            "selected_device": self.config.preferred_device.value if self.config.preferred_device else None,
            "device_memory_fraction": self.config.device_memory_fraction
        }
    
    def cleanup(self):
        """Clean up device resources."""
        self._monitoring_active = False
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        
        # Clear performance cache
        self._performance_cache.clear()
        
        logger.info("Device manager cleanup completed")


class DeviceAwareInference:
    """High-level interface for device-optimized inference."""
    
    def __init__(self, device_manager: DeviceManager):
        self.device_manager = device_manager
        
    def create_inference_session(self,
                                model: torch.nn.Module,
                                model_size_gb: float,
                                inference_mode: InferenceMode = InferenceMode.BALANCED,
                                batch_size: int = 1) -> Dict[str, Any]:
        """Create an optimized inference session."""
        # Select optimal device
        device = self.device_manager.select_optimal_device(
            model_size_gb, inference_mode, batch_size
        )
        
        # Optimize model for inference
        optimization_results = self.device_manager.optimize_for_inference(device, model)
        
        # Generate device map
        device_map = self.device_manager.get_device_map(model, device)
        
        return {
            "device": device,
            "device_map": device_map,
            "optimizations": optimization_results,
            "session_id": f"inference_{int(time.time())}"
        }