"""
DeepSpeed Training Utilities
Comprehensive utilities for DeepSpeed distributed training including initialization,
monitoring, checkpoint management, and performance profiling.
"""

import os
import json
import time
import psutil
import logging
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import torch
import torch.distributed as dist
from deepspeed import get_accelerator

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DeepSpeedUtils:
    """Main utility class for DeepSpeed operations."""
    
    @staticmethod
    def setup_distributed_environment(rank: int = 0, world_size: int = 1, master_addr: str = "localhost", 
                                     master_port: str = "29500", backend: str = "nccl") -> bool:
        """
        Setup distributed training environment.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            master_addr: Master node address
            master_port: Master node port
            backend: Communication backend
            
        Returns:
            True if setup successful
        """
        try:
            # Set environment variables
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(rank)
            
            # Initialize distributed training
            if not dist.is_initialized():
                torch.cuda.set_device(rank)
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    world_size=world_size,
                    rank=rank
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup distributed environment: {e}")
            return False
    
    @staticmethod
    def cleanup_distributed():
        """Cleanup distributed training environment."""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logging.error(f"Error during distributed cleanup: {e}")
    
    @staticmethod
    def get_world_info() -> Dict[str, int]:
        """Get distributed training world information."""
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        return {
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "is_distributed": dist.is_initialized()
        }
    
    @staticmethod
    def save_zero_state_dict(engine, save_path: str):
        """Save ZeRO state dict for optimization."""
        try:
            # Get zero partition states
            zero_stage = engine.config.get('zero_optimization', {}).get('stage', 0)
            
            if zero_stage > 0:
                # Save optimizer states
                state_dict = engine.optimizer.state_dict()
                
                # Save additional zero states
                if hasattr(engine, '_zero_grad'):
                    state_dict['zero_grad'] = engine._zero_grad
                
                torch.save(state_dict, save_path)
                
        except Exception as e:
            logging.error(f"Failed to save zero state dict: {e}")
    
    @staticmethod
    def broadcast_object(obj: Any, src: int = 0) -> Any:
        """Broadcast object from src to all processes."""
        if not dist.is_initialized():
            return obj
        
        try:
            # Serialize object
            import pickle
            data = pickle.dumps(obj)
            shape = torch.tensor([len(data)], dtype=torch.int64, device='cuda')
            
            if dist.get_rank() == src:
                # Source process sends data
                dist.broadcast(shape, src=src)
                tensor = torch.from_numpy(bytearray(data))
                dist.broadcast(tensor, src=src)
            else:
                # Other processes receive data
                dist.broadcast(shape, src=src)
                tensor = torch.empty(shape.item(), dtype=torch.uint8, device='cuda')
                dist.broadcast(tensor, src=src)
                data = bytes(tensor.cpu().numpy())
            
            return pickle.loads(data) if dist.get_rank() != src else obj
            
        except Exception as e:
            logging.error(f"Failed to broadcast object: {e}")
            return obj


class MemoryProfiler:
    """Memory profiling utilities for distributed training."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.memory_history = []
        
    def profile_memory(self, prefix: str = "") -> Dict[str, float]:
        """Profile current memory usage."""
        memory_info = {}
        
        try:
            # GPU memory
            if torch.cuda.is_available():
                memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_info['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3   # GB
                memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3
            
            # CPU memory
            process = psutil.Process(os.getpid())
            memory_info['cpu_rss'] = process.memory_info().rss / 1024**3  # GB
            memory_info['cpu_vms'] = process.memory_info().vms / 1024**3   # GB
            
            # Record history
            self.memory_history.append({
                'timestamp': time.time(),
                **memory_info
            })
            
            # Log memory usage
            if prefix:
                self.logger.info(f"{prefix} - GPU allocated: {memory_info.get('gpu_allocated', 0):.2f}GB, "
                               f"CPU RSS: {memory_info.get('cpu_rss', 0):.2f}GB")
            
        except Exception as e:
            self.logger.error(f"Memory profiling failed: {e}")
        
        return memory_info
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        memory_info = self.profile_memory()
        
        if memory_info:
            world_info = DeepSpeedUtils.get_world_info()
            rank = world_info['rank']
            
            if rank == 0:
                self.logger.info(f"{context} - Memory Usage:")
                for key, value in memory_info.items():
                    self.logger.info(f"  {key}: {value:.2f}")
    
    def monitor_memory_usage(self, interval: int = 60, duration: int = 3600):
        """Monitor memory usage for specified duration."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.profile_memory("Memory Monitor")
            time.sleep(interval)
    
    def print_summary(self):
        """Print memory usage summary."""
        if not self.memory_history:
            return
        
        try:
            # Calculate statistics
            recent_data = self.memory_history[-100:]  # Last 100 samples
            
            stats = {}
            for key in ['gpu_allocated', 'cpu_rss']:
                values = [d.get(key, 0) for d in recent_data if key in d]
                if values:
                    stats[key] = {
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values)
                    }
            
            # Log summary
            self.logger.info("Memory Usage Summary:")
            for key, stat in stats.items():
                self.logger.info(f"  {key}: min={stat['min']:.2f}GB, "
                               f"max={stat['max']:.2f}GB, mean={stat['mean']:.2f}GB")
            
        except Exception as e:
            self.logger.error(f"Failed to print memory summary: {e}")
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            self.logger.info("Memory cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")


class PerformanceMonitor:
    """Performance monitoring utilities."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.step_times = []
        self.throughput_history = []
        self.start_time = None
        self.total_samples = 0
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.logger.info("Performance monitoring started")
    
    def log_step(self, global_step: int, batch_idx: int, batch_size: int = None):
        """Log performance metrics for current step."""
        try:
            current_time = time.time()
            
            # Calculate step time
            if len(self.step_times) > 0:
                step_time = current_time - self.step_times[-1]
                
                # Calculate throughput
                if batch_size and step_time > 0:
                    samples_per_sec = batch_size / step_time
                    self.throughput_history.append({
                        'step': global_step,
                        'samples_per_sec': samples_per_sec,
                        'timestamp': current_time
                    })
                
                # Log every 100 steps
                if global_step % 100 == 0 and batch_size:
                    self.logger.info(f"Step {global_step}: "
                                   f"throughput={samples_per_sec:.2f} samples/sec, "
                                   f"step_time={step_time:.3f}s")
            
            self.step_times.append(current_time)
            
        except Exception as e:
            self.logger.error(f"Failed to log step performance: {e}")
    
    def calculate_throughput(self, total_samples: int, duration: float) -> float:
        """Calculate overall throughput."""
        if duration > 0:
            return total_samples / duration
        return 0.0
    
    def benchmark_step_time(self, warmup_steps: int = 10, benchmark_steps: int = 100) -> Dict[str, float]:
        """Benchmark step timing."""
        self.logger.info(f"Running step time benchmark: {warmup_steps} warmup, {benchmark_steps} benchmark steps")
        
        # Warmup
        for _ in range(warmup_steps):
            time.sleep(0.001)  # Simulate training step
        
        # Benchmark
        step_times = []
        for _ in range(benchmark_steps):
            start = time.time()
            time.sleep(0.001)  # Simulate training step
            step_times.append(time.time() - start)
        
        # Calculate statistics
        mean_time = sum(step_times) / len(step_times)
        min_time = min(step_times)
        max_time = max(step_times)
        
        results = {
            'mean_step_time': mean_time,
            'min_step_time': min_time,
            'max_step_time': max_time,
            'steps_per_second': 1.0 / mean_time
        }
        
        self.logger.info(f"Benchmark results: {results}")
        return results
    
    def monitor_communication_overhead(self):
        """Monitor communication overhead in distributed training."""
        if not dist.is_initialized():
            self.logger.warning("Distributed training not initialized")
            return
        
        try:
            # Test all-reduce performance
            start_time = time.time()
            tensor = torch.randn(1000, 1000).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            all_reduce_time = time.time() - start_time
            
            # Test broadcast performance
            start_time = time.time()
            dist.broadcast(tensor, src=0)
            broadcast_time = time.time() - start_time
            
            self.logger.info(f"Communication overhead - All-reduce: {all_reduce_time:.4f}s, "
                           f"Broadcast: {broadcast_time:.4f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to monitor communication overhead: {e}")
    
    def print_summary(self):
        """Print performance summary."""
        try:
            if not self.throughput_history:
                self.logger.warning("No performance data available")
                return
            
            # Calculate throughput statistics
            throughputs = [d['samples_per_sec'] for d in self.throughput_history]
            
            stats = {
                'mean_throughput': sum(throughputs) / len(throughputs),
                'max_throughput': max(throughputs),
                'min_throughput': min(throughputs)
            }
            
            self.logger.info("Performance Summary:")
            self.logger.info(f"  Mean throughput: {stats['mean_throughput']:.2f} samples/sec")
            self.logger.info(f"  Max throughput: {stats['max_throughput']:.2f} samples/sec")
            self.logger.info(f"  Min throughput: {stats['min_throughput']:.2f} samples/sec")
            
        except Exception as e:
            self.logger.error(f"Failed to print performance summary: {e}")
    
    def save_performance_report(self, filename: str = "performance_report.json"):
        """Save detailed performance report."""
        try:
            report = {
                'throughput_history': self.throughput_history,
                'step_times': self.step_times,
                'total_samples': self.total_samples
            }
            
            report_path = self.log_dir / filename
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Performance report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance report: {e}")


class CheckpointManager:
    """Advanced checkpoint management utilities."""
    
    def __init__(self, save_dir: str = "checkpoints", max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.max_checkpoints = max_checkpoints
        self.checkpoint_history = []
    
    def save_checkpoint(self, engine, epoch: int, step: int, metrics: Dict[str, Any], 
                       additional_state: Dict[str, Any] = None, save_name: str = None):
        """
        Save comprehensive checkpoint.
        
        Args:
            engine: DeepSpeed engine
            epoch: Training epoch
            step: Global step
            metrics: Training metrics
            additional_state: Additional state to save
            save_name: Custom checkpoint name
        """
        try:
            if save_name is None:
                save_name = f"checkpoint_epoch{epoch}_step{step}"
            
            checkpoint_path = self.save_dir / save_name
            checkpoint_path.mkdir(exist_ok=True)
            
            # Save DeepSpeed checkpoint
            engine.save_checkpoint(checkpoint_path)
            
            # Save additional state
            state_dict = {
                'epoch': epoch,
                'step': step,
                'metrics': metrics,
                'timestamp': time.time(),
                'world_info': DeepSpeedUtils.get_world_info(),
                'config': engine.config
            }
            
            if additional_state:
                state_dict.update(additional_state)
            
            # Save state file
            state_file = checkpoint_path / "training_state.json"
            with open(state_file, 'w') as f:
                json.dump(state_dict, f, indent=2, default=str)
            
            # Save optimizer states separately
            optimizer_file = checkpoint_path / "optimizer_state.pt"
            torch.save(engine.optimizer.state_dict(), optimizer_file)
            
            # Update checkpoint history
            self._update_checkpoint_history(checkpoint_path, state_dict)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, engine, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint.
        
        Args:
            engine: DeepSpeed engine
            checkpoint_path: Path to checkpoint
            
        Returns:
            Loaded state information
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            # Load DeepSpeed checkpoint
            engine.load_checkpoint(checkpoint_path)
            
            # Load additional state
            state_file = checkpoint_path / "training_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_dict = json.load(f)
                
                self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
                return state_dict
            else:
                self.logger.warning("No training state file found in checkpoint")
                return {}
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to maintain disk space."""
        try:
            if len(self.checkpoint_history) > self.max_checkpoints:
                # Sort by timestamp and remove oldest
                sorted_history = sorted(self.checkpoint_history, 
                                      key=lambda x: x['timestamp'])
                
                checkpoints_to_remove = sorted_history[:-self.max_checkpoints]
                
                for checkpoint_info in checkpoints_to_remove:
                    checkpoint_path = Path(checkpoint_info['path'])
                    if checkpoint_path.exists():
                        import shutil
                        shutil.rmtree(checkpoint_path)
                        self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
                
                self.checkpoint_history = sorted_history[-self.max_checkpoints:]
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    def _update_checkpoint_history(self, checkpoint_path: Path, state_dict: Dict[str, Any]):
        """Update checkpoint history tracking."""
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': state_dict.get('epoch'),
            'step': state_dict.get('step'),
            'timestamp': state_dict.get('timestamp', time.time()),
            'metrics': state_dict.get('metrics', {})
        }
        
        self.checkpoint_history.append(checkpoint_info)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.checkpoint_history.copy()
    
    def find_best_checkpoint(self, metric_name: str, ascending: bool = True) -> Optional[Dict[str, Any]]:
        """Find best checkpoint based on metric."""
        if not self.checkpoint_history:
            return None
        
        # Filter checkpoints with the metric
        valid_checkpoints = [cp for cp in self.checkpoint_history 
                           if metric_name in cp.get('metrics', {})]
        
        if not valid_checkpoints:
            return None
        
        # Sort by metric value
        if ascending:
            best_cp = min(valid_checkpoints, 
                         key=lambda x: x['metrics'][metric_name])
        else:
            best_cp = max(valid_checkpoints, 
                         key=lambda x: x['metrics'][metric_name])
        
        return best_cp


class ModelValidator:
    """Utility for validating models before distributed training."""
    
    @staticmethod
    def validate_model_for_distributed_training(model: torch.nn.Module) -> Dict[str, Any]:
        """Validate model compatibility with distributed training."""
        validation_results = {
            'compatible': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        try:
            # Check for unsupported modules
            unsupported_modules = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.LSTM):
                    unsupported_modules.append(name)
            
            if unsupported_modules:
                validation_results['warnings'].append(
                    f"Found RNN modules that may not work optimally with ZeRO: {unsupported_modules}"
                )
            
            # Check parameter count
            param_count = sum(p.numel() for p in model.parameters())
            validation_results['parameter_count'] = param_count
            
            if param_count > 1e9:  # More than 1B parameters
                validation_results['recommendations'].append(
                    "Consider using ZeRO Stage 3 for models with >1B parameters"
                )
            
            # Check for batch norm layers
            bn_layers = sum(1 for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
            if bn_layers > 0:
                validation_results['recommendations'].append(
                    "BatchNorm layers may need special attention with mixed precision training"
                )
            
            # Test forward pass
            try:
                dummy_input = torch.randn(2, 10)
                with torch.no_grad():
                    output = model(dummy_input)
                validation_results['forward_pass_test'] = 'success'
            except Exception as e:
                validation_results['errors'].append(f"Forward pass test failed: {e}")
                validation_results['compatible'] = False
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
            validation_results['compatible'] = False
        
        return validation_results


class CommunicationOptimizer:
    """Utilities for optimizing communication in distributed training."""
    
    @staticmethod
    def optimize_communication_settings():
        """Apply optimal communication settings."""
        try:
            # Set environment variables for optimized communication
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            os.environ["NCCL_MIN_NRINGS"] = "4"
            os.environ["NCCL_DEBUG"] = "WARN"
            
            # Set CUDA settings
            if torch.cuda.is_available():
                # Enable peer-to-peer access
                for i in range(torch.cuda.device_count()):
                    for j in range(torch.cuda.device_count()):
                        if i != j:
                            torch.cuda.enable_peer_to_peer(i, j)
            
            logging.info("Communication settings optimized")
            
        except Exception as e:
            logging.error(f"Failed to optimize communication settings: {e}")
    
    @staticmethod
    def benchmark_communication(world_size: int) -> Dict[str, float]:
        """Benchmark communication performance."""
        results = {}
        
        try:
            if not dist.is_initialized():
                logging.warning("Distributed training not initialized")
                return results
            
            rank = dist.get_rank()
            
            # Test all-reduce
            sizes = [1024, 4096, 16384, 65536]  # Different tensor sizes
            
            for size in sizes:
                tensor = torch.randn(size, dtype=torch.float32).cuda()
                
                # Warmup
                for _ in range(5):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize()
                
                # Benchmark
                times = []
                for _ in range(10):
                    start = time.time()
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    torch.cuda.synchronize()
                    times.append(time.time() - start)
                
                results[f"all_reduce_{size}"] = {
                    'mean_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
            
            if rank == 0:
                logging.info("Communication benchmark results:")
                for test, stats in results.items():
                    logging.info(f"  {test}: {stats['mean_time']:.4f}s (min: {stats['min_time']:.4f}s)")
            
        except Exception as e:
            logging.error(f"Communication benchmark failed: {e}")
        
        return results