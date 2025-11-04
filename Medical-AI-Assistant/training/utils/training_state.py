"""
Training State Management System

This module provides comprehensive training state management including:
- Training state serialization and deserialization
- State recovery mechanisms
- Configuration management
- Metrics tracking and history
- Version compatibility checking
- Incremental training support

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import json
import os
import pickle
import shutil
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import yaml

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int = 0
    step: int = 0
    phase: str = "training"  # training, validation, test
    
    # Loss metrics
    loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    
    # Accuracy metrics
    accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    
    # Learning rate
    learning_rate: float = 0.0
    
    # Additional metrics
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TrainingConfiguration:
    """Training configuration with versioning"""
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Model configuration
    model_name: str = ""
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    max_steps: int = 0
    
    # Optimization
    optimizer: str = "adam"
    optimizer_config: Dict[str, Any] = field(default_factory=dict)
    scheduler: str = ""
    scheduler_config: Dict[str, Any] = field(default_factory=dict)
    
    # Regularization
    weight_decay: float = 0.0
    dropout: float = 0.0
    gradient_clip_val: float = 0.0
    
    # Data configuration
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware configuration
    device: str = "cuda"
    num_gpus: int = 1
    mixed_precision: bool = False
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_every_n_epochs: int = 5
    save_every_n_steps: int = 1000
    
    # Random seeds for reproducibility
    random_seed: int = 42
    
    # Environment information
    python_version: str = ""
    pytorch_version: str = ""
    
    def update_version(self) -> None:
        """Update configuration version"""
        self.version = f"{float(self.version.split('.')[0]) + 1}.0"
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingConfiguration':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class EnvironmentState:
    """System environment state"""
    python_version: str
    pytorch_version: str
    cuda_version: Optional[str]
    gpu_info: List[Dict[str, Any]]
    system_info: Dict[str, Any]
    dependencies: Dict[str, str]
    environment_variables: Dict[str, str]
    
    @classmethod
    def capture_current_state(cls) -> 'EnvironmentState':
        """Capture current environment state"""
        import platform
        import sys
        
        # GPU info
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'capability': torch.cuda.get_device_capability(i)
                })
        
        # System info
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total if 'psutil' in sys.modules else 0
        }
        
        # Dependencies
        dependencies = {}
        try:
            import pkg_resources
            for package in pkg_resources.working_set:
                dependencies[package.key] = package.version
        except ImportError:
            pass
        
        # Environment variables (filtered)
        env_vars = {}
        for key, value in os.environ.items():
            if any(prefix in key.lower() for prefix in ['path', 'home', 'user', 'tmp', 'cache']):
                env_vars[key] = value[:100] + "..." if len(value) > 100 else value
        
        return cls(
            python_version=sys.version,
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            gpu_info=gpu_info,
            system_info=system_info,
            dependencies=dependencies,
            environment_variables=env_vars
        )


class TrainingState:
    """Comprehensive training state management"""
    
    def __init__(
        self,
        state_dir: Union[str, Path],
        experiment_name: str = "default",
        max_history: int = 1000
    ):
        self.state_dir = Path(state_dir)
        self.experiment_name = experiment_name
        self.max_history = max_history
        
        # Create state directory
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "configs").mkdir(exist_ok=True)
        (self.state_dir / "metrics").mkdir(exist_ok=True)
        (self.state_dir / "history").mkdir(exist_ok=True)
        (self.state_dir / "snapshots").mkdir(exist_ok=True)
        
        # State files
        self.current_state_file = self.state_dir / "current_state.json"
        self.config_history_file = self.state_dir / "configs" / "config_history.json"
        self.metrics_history_file = self.state_dir / "metrics" / "metrics_history.json"
        self.snapshots_dir = self.state_dir / "snapshots"
        
        # Initialize components
        self.current_config = self._load_config()
        self.current_metrics = self._load_current_metrics()
        self.metrics_history = self._load_metrics_history()
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.state_dir / "logs")
        
        # Version compatibility
        self.compatibility_handlers: Dict[str, Callable] = {}
        self._register_compatibility_handlers()
        
        logger.info(f"TrainingState initialized for experiment: {experiment_name}")
    
    def _load_config(self) -> TrainingConfiguration:
        """Load current configuration"""
        config_file = self.state_dir / "configs" / "current_config.yaml"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = yaml.safe_load(f)
                return TrainingConfiguration.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        # Create default config
        config = TrainingConfiguration()
        config.python_version = torch.__version__
        
        # Capture environment state
        try:
            env_state = EnvironmentState.capture_current_state()
            config.pytorch_version = env_state.pytorch_version
        except Exception:
            pass
        
        self._save_config(config)
        return config
    
    def _load_current_metrics(self) -> TrainingMetrics:
        """Load current training metrics"""
        metrics_file = self.state_dir / "current_metrics.json"
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                return TrainingMetrics.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
        
        return TrainingMetrics()
    
    def _load_metrics_history(self) -> List[TrainingMetrics]:
        """Load metrics history"""
        if self.metrics_history_file.exists():
            try:
                with open(self.metrics_history_file, 'r') as f:
                    data = json.load(f)
                return [TrainingMetrics.from_dict(item) for item in data]
            except Exception as e:
                logger.warning(f"Failed to load metrics history: {e}")
        
        return []
    
    def _save_config(self, config: TrainingConfiguration) -> None:
        """Save configuration"""
        config_file = self.state_dir / "configs" / "current_config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        
        # Update history
        self._update_config_history(config)
    
    def _save_current_metrics(self, metrics: TrainingMetrics) -> None:
        """Save current metrics"""
        metrics_file = self.state_dir / "current_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def _save_metrics_history(self) -> None:
        """Save metrics history"""
        with open(self.metrics_history_file, 'w') as f:
            json.dump([m.to_dict() for m in self.metrics_history], f, indent=2)
    
    def _update_config_history(self, config: TrainingConfiguration) -> None:
        """Update configuration history"""
        history_file = self.config_history_file
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(config.to_dict())
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def _register_compatibility_handlers(self) -> None:
        """Register version compatibility handlers"""
        self.compatibility_handlers["1.0"] = self._handle_v1_0
        # Add more handlers as versions evolve
    
    def _handle_v1_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle version 1.0 configuration"""
        # Migration logic for version 1.0
        if 'max_steps' not in config_data:
            config_data['max_steps'] = 0
        
        if 'mixed_precision' not in config_data:
            config_data['mixed_precision'] = False
        
        return config_data
    
    def update_config(
        self,
        config_updates: Dict[str, Any],
        require_version_bump: bool = True
    ) -> bool:
        """
        Update training configuration
        
        Args:
            config_updates: Dictionary of configuration updates
            require_version_bump: Whether to bump version after update
        
        Returns:
            Success status
        """
        with self._lock:
            try:
                # Create updated config
                updated_config = self.current_config
                for key, value in config_updates.items():
                    if hasattr(updated_config, key):
                        setattr(updated_config, key, value)
                    else:
                        # Add to model_config or data_config
                        if key in ['model_config', 'optimizer_config', 'scheduler_config', 'data_config']:
                            getattr(updated_config, key).update(value)
                        else:
                            updated_config.model_config[key] = value
                
                # Bump version if required
                if require_version_bump:
                    updated_config.update_version()
                
                # Validate configuration
                if not self._validate_config(updated_config):
                    logger.error("Configuration validation failed")
                    return False
                
                # Save configuration
                self.current_config = updated_config
                self._save_config(updated_config)
                
                logger.info("Configuration updated successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update configuration: {e}")
                return False
    
    def update_metrics(self, metrics: Union[TrainingMetrics, Dict[str, Any]]) -> None:
        """
        Update training metrics
        
        Args:
            metrics: Training metrics to update
        """
        with self._lock:
            try:
                if isinstance(metrics, dict):
                    metrics = TrainingMetrics.from_dict(metrics)
                
                # Update current metrics
                self.current_metrics = metrics
                self._save_current_metrics(metrics)
                
                # Add to history
                self.metrics_history.append(metrics)
                
                # Maintain history limit
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                # Save history periodically
                if len(self.metrics_history) % 10 == 0:
                    self._save_metrics_history()
                
                # Log to TensorBoard
                self._log_to_tensorboard(metrics)
                
            except Exception as e:
                logger.error(f"Failed to update metrics: {e}")
    
    def _validate_config(self, config: TrainingConfiguration) -> bool:
        """Validate configuration"""
        try:
            # Basic validation
            if config.batch_size <= 0:
                logger.error("Batch size must be positive")
                return False
            
            if config.learning_rate <= 0:
                logger.error("Learning rate must be positive")
                return False
            
            if config.num_epochs <= 0:
                logger.error("Number of epochs must be positive")
                return False
            
            # Check device availability
            if config.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                config.device = "cpu"
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def _log_to_tensorboard(self, metrics: TrainingMetrics) -> None:
        """Log metrics to TensorBoard"""
        try:
            # Loss metrics
            if metrics.loss > 0:
                self.writer.add_scalar('loss/training', metrics.loss, metrics.step)
            if metrics.val_loss > 0:
                self.writer.add_scalar('loss/validation', metrics.val_loss, metrics.epoch)
            
            # Accuracy metrics
            if metrics.accuracy > 0:
                self.writer.add_scalar('accuracy/training', metrics.accuracy, metrics.step)
            if metrics.val_accuracy > 0:
                self.writer.add_scalar('accuracy/validation', metrics.val_accuracy, metrics.epoch)
            
            # Learning rate
            if metrics.learning_rate > 0:
                self.writer.add_scalar('learning_rate', metrics.learning_rate, metrics.step)
            
            # Additional metrics
            for metric_name, value in metrics.additional_metrics.items():
                self.writer.add_scalar(f'additional/{metric_name}', value, metrics.step)
            
        except Exception as e:
            logger.warning(f"Failed to log to TensorBoard: {e}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current training state"""
        return {
            'experiment_name': self.experiment_name,
            'config': self.current_config.to_dict(),
            'metrics': self.current_metrics.to_dict(),
            'metrics_history_length': len(self.metrics_history),
            'has_snapshots': len(self.snapshots) > 0
        }
    
    def get_metrics_history(
        self,
        phase: Optional[str] = None,
        metric_name: Optional[str] = None,
        start_epoch: Optional[int] = None,
        end_epoch: Optional[int] = None
    ) -> List[TrainingMetrics]:
        """Get filtered metrics history"""
        filtered_history = self.metrics_history
        
        if phase:
            filtered_history = [m for m in filtered_history if m.phase == phase]
        
        if start_epoch is not None:
            filtered_history = [m for m in filtered_history if m.epoch >= start_epoch]
        
        if end_epoch is not None:
            filtered_history = [m for m in filtered_history if m.epoch <= end_epoch]
        
        if metric_name:
            filtered_history = [
                m for m in filtered_history 
                if metric_name in m.additional_metrics or hasattr(m, metric_name)
            ]
        
        return filtered_history
    
    def get_best_metrics(
        self,
        metric_name: str,
        mode: str = "max",
        phase: str = "validation",
        min_epoch: int = 0
    ) -> Tuple[Optional[TrainingMetrics], Optional[float]]:
        """Get best metrics based on specified criteria"""
        filtered_history = self.get_metrics_history(
            phase=phase,
            start_epoch=min_epoch
        )
        
        if not filtered_history:
            return None, None
        
        best_metrics = None
        best_value = float('-inf') if mode == "max" else float('inf')
        
        for metrics in filtered_history:
            # Get metric value
            value = None
            if metric_name == "loss":
                value = metrics.val_loss if phase == "validation" else metrics.loss
            elif metric_name == "accuracy":
                value = metrics.val_accuracy if phase == "validation" else metrics.accuracy
            elif metric_name in metrics.additional_metrics:
                value = metrics.additional_metrics[metric_name]
            
            if value is not None:
                if (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
                    best_value = value
                    best_metrics = metrics
        
        return best_metrics, best_value
    
    def create_snapshot(
        self,
        snapshot_name: str,
        include_history: bool = False,
        compression: bool = True
    ) -> bool:
        """
        Create training state snapshot
        
        Args:
            snapshot_name: Name of snapshot
            include_history: Include metrics history
            compression: Compress snapshot files
        
        Returns:
            Success status
        """
        try:
            snapshot_dir = self.snapshots_dir / snapshot_name
            snapshot_dir.mkdir(exist_ok=True)
            
            # Save current state
            state_data = {
                'experiment_name': self.experiment_name,
                'snapshot_time': datetime.now().isoformat(),
                'config': self.current_config.to_dict(),
                'current_metrics': self.current_metrics.to_dict(),
                'version': '1.0'
            }
            
            state_file = snapshot_dir / "state.json"
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Save config history
            if self.config_history_file.exists():
                shutil.copy2(self.config_history_file, snapshot_dir / "config_history.json")
            
            # Save metrics history if requested
            if include_history and self.metrics_history_file.exists():
                shutil.copy2(self.metrics_history_file, snapshot_dir / "metrics_history.json")
            
            # Save model checkpoints if they exist
            model_dir = Path(self.state_dir).parent / "models" / "checkpoints"
            if model_dir.exists():
                snapshot_model_dir = snapshot_dir / "checkpoints"
                shutil.copytree(model_dir, snapshot_model_dir, dirs_exist_ok=True)
            
            # Save TensorBoard logs
            log_dir = self.state_dir / "logs"
            if log_dir.exists():
                snapshot_log_dir = snapshot_dir / "logs"
                shutil.copytree(log_dir, snapshot_log_dir, dirs_exist_ok=True)
            
            # Compress if requested
            if compression:
                snapshot_file = snapshot_dir.with_suffix('.tar.gz')
                shutil.make_archive(
                    str(snapshot_dir),
                    'gzip',
                    root_dir=str(snapshot_dir)
                )
                shutil.rmtree(snapshot_dir)
            
            # Store snapshot metadata
            self.snapshots[snapshot_name] = {
                'created_at': datetime.now().isoformat(),
                'include_history': include_history,
                'compressed': compression,
                'path': str(snapshot_file if compression else snapshot_dir)
            }
            
            logger.info(f"Snapshot created: {snapshot_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return False
    
    def restore_snapshot(
        self,
        snapshot_name: str,
        restore_history: bool = False,
        restore_checkpoints: bool = False
    ) -> bool:
        """
        Restore training state from snapshot
        
        Args:
            snapshot_name: Name of snapshot to restore
            restore_history: Restore metrics history
            restore_checkpoints: Restore model checkpoints
        
        Returns:
            Success status
        """
        try:
            if snapshot_name not in self.snapshots:
                logger.error(f"Snapshot {snapshot_name} not found")
                return False
            
            snapshot_info = self.snapshots[snapshot_name]
            snapshot_path = Path(snapshot_info['path'])
            
            # Extract if compressed
            extract_dir = None
            if snapshot_path.suffix == '.gz':
                extract_dir = self.snapshots_dir / f"temp_{snapshot_name}"
                extract_dir.mkdir(exist_ok=True)
                
                import tarfile
                with tarfile.open(snapshot_path, 'r:gz') as tar:
                    tar.extractall(extract_dir)
                snapshot_path = extract_dir
            
            # Load state
            state_file = snapshot_path / "state.json"
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore configuration
            self.current_config = TrainingConfiguration.from_dict(state_data['config'])
            self._save_config(self.current_config)
            
            # Restore current metrics
            self.current_metrics = TrainingMetrics.from_dict(state_data['current_metrics'])
            self._save_current_metrics(self.current_metrics)
            
            # Restore history if requested
            if restore_history:
                metrics_history_file = snapshot_path / "metrics_history.json"
                if metrics_history_file.exists():
                    shutil.copy2(metrics_history_file, self.metrics_history_file)
                    self.metrics_history = self._load_metrics_history()
            
            # Restore checkpoints if requested
            if restore_checkpoints:
                snapshot_checkpoints_dir = snapshot_path / "checkpoints"
                if snapshot_checkpoints_dir.exists():
                    model_dir = Path(self.state_dir).parent / "models" / "checkpoints"
                    model_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(snapshot_checkpoints_dir, model_dir, dirs_exist_ok=True)
            
            # Restore TensorBoard logs
            snapshot_log_dir = snapshot_path / "logs"
            if snapshot_log_dir.exists():
                log_dir = self.state_dir / "logs"
                shutil.copytree(snapshot_log_dir, log_dir, dirs_exist_ok=True)
            
            # Cleanup temporary extraction
            if extract_dir:
                shutil.rmtree(extract_dir)
            
            logger.info(f"Snapshot restored: {snapshot_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}")
            return False
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all snapshots"""
        return [
            {
                'name': name,
                **info
            }
            for name, info in self.snapshots.items()
        ]
    
    def delete_snapshot(self, snapshot_name: str) -> bool:
        """Delete snapshot"""
        try:
            if snapshot_name not in self.snapshots:
                logger.error(f"Snapshot {snapshot_name} not found")
                return False
            
            snapshot_info = self.snapshots[snapshot_name]
            snapshot_path = Path(snapshot_info['path'])
            
            # Remove files
            if snapshot_path.exists():
                if snapshot_path.is_file():
                    snapshot_path.unlink()
                else:
                    shutil.rmtree(snapshot_path)
            
            # Remove from registry
            del self.snapshots[snapshot_name]
            
            logger.info(f"Snapshot deleted: {snapshot_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete snapshot: {e}")
            return False
    
    def check_compatibility(self, target_version: str) -> bool:
        """Check if current state is compatible with target version"""
        current_version = self.current_config.version
        
        # Parse versions
        try:
            current_major = int(current_version.split('.')[0])
            target_major = int(target_version.split('.')[0])
            
            # Simple compatibility: same major version
            return current_major == target_major
            
        except Exception:
            return False
    
    def migrate(self, target_version: str) -> bool:
        """Migrate state to target version"""
        try:
            current_version = self.current_config.version
            
            # Find migration path
            if current_version == target_version:
                return True
            
            # Apply migrations sequentially
            current_major = int(current_version.split('.')[0])
            target_major = int(target_version.split('.')[0])
            
            for version in range(current_major + 1, target_major + 1):
                version_str = f"{version}.0"
                if version_str in self.compatibility_handlers:
                    # Apply migration
                    config_data = self.current_config.to_dict()
                    migrated_data = self.compatibility_handlers[version_str](config_data)
                    
                    # Update version
                    migrated_data['version'] = version_str
                    self.current_config = TrainingConfiguration.from_dict(migrated_data)
                    
                    logger.info(f"Migrated to version {version_str}")
                else:
                    logger.warning(f"No migration handler for version {version_str}")
            
            # Save migrated configuration
            self._save_config(self.current_config)
            
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def export_state(
        self,
        export_path: Union[str, Path],
        format: str = "json",
        include_history: bool = True
    ) -> bool:
        """Export training state to file"""
        try:
            export_path = Path(export_path)
            
            if format == "json":
                export_data = {
                    'experiment_name': self.experiment_name,
                    'export_time': datetime.now().isoformat(),
                    'config': self.current_config.to_dict(),
                    'current_metrics': self.current_metrics.to_dict(),
                    'snapshots': self.snapshots,
                    'version': self.current_config.version
                }
                
                if include_history:
                    export_data['metrics_history'] = [
                        m.to_dict() for m in self.metrics_history
                    ]
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format == "yaml":
                export_data = {
                    'experiment_name': self.experiment_name,
                    'export_time': datetime.now().isoformat(),
                    'config': self.current_config.to_dict(),
                    'current_metrics': self.current_metrics.to_dict(),
                    'version': self.current_config.version
                }
                
                with open(export_path, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"State exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def get_state_hash(self) -> str:
        """Get hash of current state for change detection"""
        state_data = {
            'config': self.current_config.to_dict(),
            'current_metrics': self.current_metrics.to_dict(),
            'experiment_name': self.experiment_name
        }
        
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def close(self) -> None:
        """Clean up resources"""
        if self.writer:
            self.writer.close()
        
        self._save_current_metrics(self.current_metrics)
        self._save_metrics_history()
        self._save_config(self.current_config)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()