"""
Advanced Model Checkpoint Management System

This module provides comprehensive checkpointing functionality including:
- Automatic checkpoint creation during training
- Checkpoint validation and integrity checking
- Checkpoint compression and storage optimization
- Checkpoint cleanup and retention policies
- Cloud storage integration
- Version compatibility checking

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import os
import json
import shutil
import hashlib
import pickle
import gzip
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

try:
    import boto3
    import azure.storage.blob
    import google.cloud.storage
    CLOUD_STORAGE_AVAILABLE = True
except ImportError:
    CLOUD_STORAGE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""
    checkpoint_id: str
    timestamp: str
    epoch: int
    step: int
    model_name: str
    model_version: str
    framework_version: str
    file_size: int
    file_hash: str
    compression: str
    storage_path: str
    training_config: Dict[str, Any]
    metrics: Dict[str, float]
    state_size: Dict[str, int]
    cloud_backup: bool = False
    retention_policy: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return asdict(self)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    # Checkpoint settings
    save_every_n_epochs: int = 5
    save_every_n_steps: int = 1000
    max_checkpoints_local: int = 10
    max_checkpoints_cloud: int = 50
    
    # Retention policies
    auto_cleanup: bool = True
    retention_days: int = 30
    keep_best_models: int = 3
    
    # Compression and optimization
    compress_checkpoints: bool = True
    compression_level: int = 6
    optimize_storage: bool = True
    
    # Cloud storage
    use_cloud_backup: bool = False
    cloud_provider: str = "aws"  # aws, azure, gcp
    cloud_bucket: str = ""
    
    # Validation
    validate_checkpoints: bool = True
    integrity_check: bool = True
    
    # Performance
    parallel_uploads: int = 2
    upload_timeout: int = 3600
    
    # Monitoring
    analytics_enabled: bool = True
    alert_on_failure: bool = True


class CheckpointManager:
    """Advanced checkpoint management system"""
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        config: CheckpointConfig,
        experiment_name: str = "default"
    ):
        self.save_dir = Path(save_dir)
        self.config = config
        self.experiment_name = experiment_name
        
        # Create directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "checkpoints").mkdir(exist_ok=True)
        (self.save_dir / "metadata").mkdir(exist_ok=True)
        (self.save_dir / "logs").mkdir(exist_ok=True)
        (self.save_dir / "analytics").mkdir(exist_ok=True)
        
        # Initialize components
        self.metadata_file = self.save_dir / "metadata" / "checkpoint_metadata.json"
        self.metadata: Dict[str, CheckpointMetadata] = self._load_metadata()
        self.writer = SummaryWriter(self.save_dir / "logs")
        
        # Thread safety
        self._lock = threading.RLock()
        self._backup_thread = None
        
        # Cloud storage clients
        self.cloud_client = None
        if config.use_cloud_backup and CLOUD_STORAGE_AVAILABLE:
            self._init_cloud_client()
        
        # Analytics
        if config.analytics_enabled:
            self.analytics_file = self.save_dir / "analytics" / "checkpoint_analytics.json"
            self.analytics = self._load_analytics()
        else:
            self.analytics = {}
        
        logger.info(f"CheckpointManager initialized for experiment: {experiment_name}")
    
    def _load_metadata(self) -> Dict[str, CheckpointMetadata]:
        """Load checkpoint metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {
                    k: CheckpointMetadata(**v) 
                    for k, v in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to file"""
        try:
            data = {
                k: v.to_dict() 
                for k, v in self.metadata.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _load_analytics(self) -> Dict[str, Any]:
        """Load checkpoint analytics data"""
        if self.analytics_file and self.analytics_file.exists():
            try:
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load analytics: {e}")
        return {
            "total_checkpoints": 0,
            "total_size_mb": 0.0,
            "successful_saves": 0,
            "failed_saves": 0,
            "cloud_uploads": 0,
            "cleanup_operations": 0,
            "validation_failures": 0
        }
    
    def _save_analytics(self) -> None:
        """Save analytics data"""
        if self.analytics_file and self.config.analytics_enabled:
            try:
                with open(self.analytics_file, 'w') as f:
                    json.dump(self.analytics, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save analytics: {e}")
    
    def _init_cloud_client(self) -> None:
        """Initialize cloud storage client"""
        try:
            if self.config.cloud_provider == "aws":
                self.cloud_client = boto3.client('s3')
            elif self.config.cloud_provider == "azure":
                self.cloud_client = azure.storage.blob.BlobServiceClient(
                    self.config.cloud_bucket
                )
            elif self.config.cloud_provider == "gcp":
                self.cloud_client = google.cloud.storage.Client()
            logger.info(f"Cloud client initialized for {self.config.cloud_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize cloud client: {e}")
            self.cloud_client = None
    
    def should_save_checkpoint(
        self,
        epoch: int,
        step: int,
        force_save: bool = False
    ) -> bool:
        """Check if checkpoint should be saved"""
        if force_save:
            return True
        
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            return True
        
        if step > 0 and step % self.config.save_every_n_steps == 0:
            return True
        
        return False
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        checkpoint_id: Optional[str] = None,
        save_best: bool = False
    ) -> str:
        """
        Save model checkpoint with comprehensive metadata
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            training_config: Training configuration
            additional_state: Additional state to save
            checkpoint_id: Custom checkpoint ID
            save_best: Mark as best model checkpoint
        
        Returns:
            Checkpoint ID
        """
        checkpoint_id = checkpoint_id or f"{self.experiment_name}_epoch_{epoch}_step_{step}"
        
        with self._lock:
            try:
                # Prepare checkpoint data
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'metrics': metrics or {},
                    'timestamp': datetime.now().isoformat(),
                    'experiment_name': self.experiment_name
                }
                
                # Add optimizer state
                if optimizer is not None:
                    checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
                
                # Add scheduler state
                if scheduler is not None:
                    checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
                
                # Add additional state
                if additional_state:
                    checkpoint_data.update(additional_state)
                
                # Add training config
                if training_config:
                    checkpoint_data['training_config'] = training_config
                
                # Generate checkpoint path
                checkpoint_file = (
                    self.save_dir / "checkpoints" / f"{checkpoint_id}.pth"
                )
                
                # Serialize and compress checkpoint
                if self.config.compress_checkpoints:
                    checkpoint_file = checkpoint_file.with_suffix('.pth.gz')
                    with gzip.open(checkpoint_file, 'wb', compresslevel=self.config.compression_level) as f:
                        pickle.dump(checkpoint_data, f)
                else:
                    torch.save(checkpoint_data, checkpoint_file)
                
                # Calculate file hash for integrity checking
                file_hash = self._calculate_file_hash(checkpoint_file)
                file_size = checkpoint_file.stat().st_size
                
                # Create metadata
                metadata = CheckpointMetadata(
                    checkpoint_id=checkpoint_id,
                    timestamp=datetime.now().isoformat(),
                    epoch=epoch,
                    step=step,
                    model_name=model.__class__.__name__,
                    model_version="1.0",
                    framework_version=torch.__version__,
                    file_size=file_size,
                    file_hash=file_hash,
                    compression="gzip" if self.config.compress_checkpoints else "none",
                    storage_path=str(checkpoint_file),
                    training_config=training_config or {},
                    metrics=metrics or {},
                    state_size={
                        'model_params': sum(p.numel() for p in model.parameters()),
                        'optimizer_states': len(optimizer.state_dict() if optimizer else {}),
                        'scheduler_states': len(scheduler.state_dict() if scheduler else {})
                    }
                )
                
                # Validate checkpoint if enabled
                if self.config.validate_checkpoints:
                    if not self._validate_checkpoint(checkpoint_file, metadata):
                        raise ValueError("Checkpoint validation failed")
                
                # Upload to cloud if enabled
                cloud_backup_successful = False
                if self.config.use_cloud_backup and self.cloud_client:
                    cloud_backup_successful = self._upload_to_cloud(checkpoint_file, checkpoint_id)
                    metadata.cloud_backup = cloud_backup_successful
                
                # Save metadata
                self.metadata[checkpoint_id] = metadata
                self._save_metadata()
                
                # Update analytics
                self._update_analytics('save', file_size, cloud_backup_successful)
                
                # Log to TensorBoard
                if metrics:
                    for metric_name, value in metrics.items():
                        self.writer.add_scalar(f'metrics/{metric_name}', value, epoch)
                    self.writer.add_scalar('checkpoint/size_mb', file_size / 1024 / 1024, epoch)
                
                # Cleanup old checkpoints
                if self.config.auto_cleanup:
                    self._cleanup_old_checkpoints()
                
                logger.info(
                    f"Checkpoint saved successfully: {checkpoint_id} "
                    f"(size: {file_size / 1024 / 1024:.2f} MB)"
                )
                
                if cloud_backup_successful:
                    logger.info(f"Cloud backup completed for {checkpoint_id}")
                
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
                self._update_analytics('save_fail', 0, False)
                raise
    
    def _validate_checkpoint(
        self,
        checkpoint_file: Path,
        metadata: CheckpointMetadata
    ) -> bool:
        """Validate checkpoint integrity"""
        try:
            # Check file hash
            current_hash = self._calculate_file_hash(checkpoint_file)
            if current_hash != metadata.file_hash:
                logger.warning(f"Hash mismatch for checkpoint {metadata.checkpoint_id}")
                self.analytics['validation_failures'] += 1
                return False
            
            # Try to load checkpoint
            if metadata.compression == "gzip":
                with gzip.open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            else:
                checkpoint_data = torch.load(checkpoint_file)
            
            # Verify required keys
            required_keys = ['model_state_dict', 'epoch', 'step']
            for key in required_keys:
                if key not in checkpoint_data:
                    logger.warning(f"Missing key {key} in checkpoint {metadata.checkpoint_id}")
                    return False
            
            logger.debug(f"Checkpoint {metadata.checkpoint_id} validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation error: {e}")
            self.analytics['validation_failures'] += 1
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _upload_to_cloud(self, checkpoint_file: Path, checkpoint_id: str) -> bool:
        """Upload checkpoint to cloud storage"""
        try:
            cloud_path = f"{self.experiment_name}/checkpoints/{checkpoint_id}.pth"
            
            if self.config.cloud_provider == "aws":
                return self._upload_to_s3(checkpoint_file, cloud_path)
            elif self.config.cloud_provider == "azure":
                return self._upload_to_azure(checkpoint_file, cloud_path)
            elif self.config.cloud_provider == "gcp":
                return self._upload_to_gcp(checkpoint_file, cloud_path)
            
            return False
            
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
            return False
    
    def _upload_to_s3(self, checkpoint_file: Path, cloud_path: str) -> bool:
        """Upload to AWS S3"""
        try:
            self.cloud_client.upload_file(
                str(checkpoint_file),
                self.config.cloud_bucket,
                cloud_path
            )
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
    
    def _upload_to_azure(self, checkpoint_file: Path, cloud_path: str) -> bool:
        """Upload to Azure Blob Storage"""
        try:
            blob_client = self.cloud_client.get_blob_client(cloud_path)
            with open(checkpoint_file, 'rb') as f:
                blob_client.upload_blob(f)
            return True
        except Exception as e:
            logger.error(f"Azure upload failed: {e}")
            return False
    
    def _upload_to_gcp(self, checkpoint_file: Path, cloud_path: str) -> bool:
        """Upload to Google Cloud Storage"""
        try:
            bucket = self.cloud_client.bucket(self.config.cloud_bucket)
            blob = bucket.blob(cloud_path)
            blob.upload_from_filename(str(checkpoint_file))
            return True
        except Exception as e:
            logger.error(f"GCP upload failed: {e}")
            return False
    
    def _update_analytics(self, operation: str, file_size: int, cloud_success: bool) -> None:
        """Update analytics counters"""
        with self._lock:
            if operation == 'save':
                self.analytics['total_checkpoints'] += 1
                self.analytics['total_size_mb'] += file_size / 1024 / 1024
                self.analytics['successful_saves'] += 1
                if cloud_success:
                    self.analytics['cloud_uploads'] += 1
            elif operation == 'save_fail':
                self.analytics['failed_saves'] += 1
            elif operation == 'cleanup':
                self.analytics['cleanup_operations'] += 1
            
            self._save_analytics()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Cleanup old checkpoints based on retention policy"""
        try:
            # Sort checkpoints by timestamp (oldest first)
            sorted_checkpoints = sorted(
                self.metadata.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Remove old local checkpoints
            if len(sorted_checkpoints) > self.config.max_checkpoints_local:
                checkpoints_to_remove = (
                    sorted_checkpoints[:len(sorted_checkpoints) - self.config.max_checkpoints_local]
                )
                
                for checkpoint_id, metadata in checkpoints_to_remove:
                    # Don't remove if marked as best or recently created
                    if metadata.retention_policy == "keep_forever":
                        continue
                    
                    # Check if checkpoint is too old
                    checkpoint_time = datetime.fromisoformat(metadata.timestamp)
                    age_days = (datetime.now() - checkpoint_time).days
                    
                    if age_days >= self.config.retention_days:
                        self._remove_checkpoint(checkpoint_id, metadata)
            
            # Clean up cloud storage if over limit
            cloud_checkpoints = [
                (k, v) for k, v in self.metadata.items() 
                if v.cloud_backup
            ]
            
            if len(cloud_checkpoints) > self.config.max_checkpoints_cloud:
                cloud_checkpoints.sort(key=lambda x: x[1].timestamp)
                checkpoints_to_remove = cloud_checkpoints[
                    :-self.config.max_checkpoints_cloud
                ]
                
                for checkpoint_id, metadata in checkpoints_to_remove:
                    self._remove_cloud_checkpoint(checkpoint_id, metadata)
            
            logger.debug("Checkpoint cleanup completed")
            
        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")
    
    def _remove_checkpoint(self, checkpoint_id: str, metadata: CheckpointMetadata) -> None:
        """Remove checkpoint file"""
        try:
            checkpoint_file = Path(metadata.storage_path)
            if checkpoint_file.exists():
                checkpoint_file.unlink()
            
            del self.metadata[checkpoint_id]
            logger.debug(f"Removed checkpoint: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove checkpoint {checkpoint_id}: {e}")
    
    def _remove_cloud_checkpoint(self, checkpoint_id: str, metadata: CheckpointMetadata) -> None:
        """Remove checkpoint from cloud storage"""
        try:
            cloud_path = f"{self.experiment_name}/checkpoints/{checkpoint_id}.pth"
            
            if self.config.cloud_provider == "aws":
                self.cloud_client.delete_object(
                    Bucket=self.config.cloud_bucket,
                    Key=cloud_path
                )
            elif self.config.cloud_provider == "azure":
                blob_client = self.cloud_client.get_blob_client(cloud_path)
                blob_client.delete_blob()
            elif self.config.cloud_provider == "gcp":
                bucket = self.cloud_client.bucket(self.config.cloud_bucket)
                blob = bucket.blob(cloud_path)
                blob.delete()
            
            metadata.cloud_backup = False
            logger.debug(f"Removed cloud checkpoint: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove cloud checkpoint {checkpoint_id}: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load checkpoint with comprehensive recovery
        
        Args:
            checkpoint_id: Specific checkpoint ID to load
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            map_location: Device mapping for loading
            strict: Whether to enforce strict key matching
        
        Returns:
            Checkpoint data dictionary
        """
        # Determine checkpoint to load
        if checkpoint_id is None:
            checkpoint_id = self.get_latest_checkpoint_id()
        
        if checkpoint_id not in self.metadata:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        metadata = self.metadata[checkpoint_id]
        checkpoint_file = Path(metadata.storage_path)
        
        if not checkpoint_file.exists():
            # Try to download from cloud
            if metadata.cloud_backup and self.cloud_client:
                if not self._download_from_cloud(checkpoint_file, checkpoint_id):
                    raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found locally or in cloud")
            else:
                raise FileNotFoundError(f"Checkpoint file {checkpoint_file} not found")
        
        try:
            # Load checkpoint data
            if metadata.compression == "gzip":
                with gzip.open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
            else:
                checkpoint_data = torch.load(checkpoint_file, map_location=map_location)
            
            # Load model state
            if model is not None and 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
            
            # Load optimizer state
            if optimizer is not None and 'optimizer_state_dict' in checkpoint_data:
                optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            # Load scheduler state
            if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            # Remove loaded keys from data to avoid confusion
            loaded_data = {
                'checkpoint_data': checkpoint_data,
                'metadata': metadata,
                'loaded_epoch': checkpoint_data.get('epoch', 0),
                'loaded_step': checkpoint_data.get('step', 0),
                'loaded_metrics': checkpoint_data.get('metrics', {})
            }
            
            logger.info(f"Checkpoint loaded successfully: {checkpoint_id}")
            return loaded_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise
    
    def _download_from_cloud(self, checkpoint_file: Path, checkpoint_id: str) -> bool:
        """Download checkpoint from cloud storage"""
        try:
            cloud_path = f"{self.experiment_name}/checkpoints/{checkpoint_id}.pth"
            
            if self.config.cloud_provider == "aws":
                return self._download_from_s3(checkpoint_file, cloud_path)
            elif self.config.cloud_provider == "azure":
                return self._download_from_azure(checkpoint_file, cloud_path)
            elif self.config.cloud_provider == "gcp":
                return self._download_from_gcp(checkpoint_file, cloud_path)
            
            return False
            
        except Exception as e:
            logger.error(f"Cloud download failed: {e}")
            return False
    
    def _download_from_s3(self, checkpoint_file: Path, cloud_path: str) -> bool:
        """Download from AWS S3"""
        try:
            self.cloud_client.download_file(
                self.config.cloud_bucket,
                cloud_path,
                str(checkpoint_file)
            )
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False
    
    def _download_from_azure(self, checkpoint_file: Path, cloud_path: str) -> bool:
        """Download from Azure Blob Storage"""
        try:
            blob_client = self.cloud_client.get_blob_client(cloud_path)
            with open(checkpoint_file, 'wb') as f:
                blob_client.download_blob().readinto(f)
            return True
        except Exception as e:
            logger.error(f"Azure download failed: {e}")
            return False
    
    def _download_from_gcp(self, checkpoint_file: Path, cloud_path: str) -> bool:
        """Download from Google Cloud Storage"""
        try:
            bucket = self.cloud_client.bucket(self.config.cloud_bucket)
            blob = bucket.blob(cloud_path)
            blob.download_to_filename(str(checkpoint_file))
            return True
        except Exception as e:
            logger.error(f"GCP download failed: {e}")
            return False
    
    def get_latest_checkpoint_id(self) -> Optional[str]:
        """Get the ID of the latest checkpoint"""
        if not self.metadata:
            return None
        
        latest_checkpoint = max(
            self.metadata.items(),
            key=lambda x: x[1].timestamp
        )
        return latest_checkpoint[0]
    
    def get_best_checkpoint_id(self, metric: str = "accuracy", mode: str = "max") -> Optional[str]:
        """Get checkpoint ID with best metric value"""
        if not self.metadata:
            return None
        
        best_checkpoint = None
        best_value = float('-inf') if mode == "max" else float('inf')
        
        for checkpoint_id, metadata in self.metadata.items():
            if metric in metadata.metrics:
                value = metadata.metrics[metric]
                
                if (mode == "max" and value > best_value) or (mode == "min" and value < best_value):
                    best_value = value
                    best_checkpoint = checkpoint_id
        
        return best_checkpoint
    
    def list_checkpoints(
        self,
        include_metadata: bool = True,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """List all checkpoints with optional filtering"""
        checkpoints = []
        
        for checkpoint_id, metadata in self.metadata.items():
            if filter_fn and not filter_fn(metadata):
                continue
            
            checkpoint_info = {
                'checkpoint_id': checkpoint_id,
                'timestamp': metadata.timestamp,
                'epoch': metadata.epoch,
                'step': metadata.step,
                'file_size_mb': metadata.file_size / 1024 / 1024,
                'has_cloud_backup': metadata.cloud_backup
            }
            
            if include_metadata:
                checkpoint_info.update({
                    'metrics': metadata.metrics,
                    'model_name': metadata.model_name,
                    'compression': metadata.compression,
                    'training_config': metadata.training_config
                })
            
            checkpoints.append(checkpoint_info)
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get checkpoint analytics"""
        return self.analytics.copy()
    
    def export_checkpoint(
        self,
        checkpoint_id: str,
        export_path: Union[str, Path],
        format: str = "pytorch"
    ) -> bool:
        """
        Export checkpoint to different format
        
        Args:
            checkpoint_id: Checkpoint to export
            export_path: Path to export to
            format: Export format (pytorch, onnx, tensorflow)
        
        Returns:
            Success status
        """
        try:
            checkpoint_data = self.load_checkpoint(checkpoint_id)
            
            if format == "pytorch":
                torch.save(checkpoint_data['checkpoint_data'], export_path)
            elif format == "onnx":
                # Convert to ONNX if model is available
                # Implementation would depend on model type
                raise NotImplementedError("ONNX export not implemented")
            elif format == "tensorflow":
                # Convert to TensorFlow format
                raise NotImplementedError("TensorFlow export not implemented")
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Checkpoint exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def optimize_storage(self) -> None:
        """Optimize storage usage by compressing and deduplicating"""
        if not self.config.optimize_storage:
            return
        
        logger.info("Starting storage optimization...")
        
        try:
            # Compress large uncompressed checkpoints
            for checkpoint_id, metadata in list(self.metadata.items()):
                if metadata.compression == "none":
                    checkpoint_file = Path(metadata.storage_path)
                    if checkpoint_file.exists():
                        compressed_file = checkpoint_file.with_suffix('.pth.gz')
                        
                        with open(checkpoint_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb', compresslevel=self.config.compression_level) as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Remove original and update metadata
                        checkpoint_file.unlink()
                        metadata.storage_path = str(compressed_file)
                        metadata.compression = "gzip"
                        metadata.file_size = compressed_file.stat().st_size
                        metadata.file_hash = self._calculate_file_hash(compressed_file)
            
            # Remove duplicate checkpoints based on hash
            hash_to_checkpoints = {}
            for checkpoint_id, metadata in self.metadata.items():
                file_hash = metadata.file_hash
                if file_hash not in hash_to_checkpoints:
                    hash_to_checkpoints[file_hash] = []
                hash_to_checkpoints[file_hash].append((checkpoint_id, metadata))
            
            # Keep only one checkpoint per hash
            for file_hash, checkpoints in hash_to_checkpoints.items():
                if len(checkpoints) > 1:
                    # Keep the latest one
                    checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)
                    for checkpoint_id, metadata in checkpoints[1:]:
                        self._remove_checkpoint(checkpoint_id, metadata)
            
            # Save updated metadata
            self._save_metadata()
            
            logger.info("Storage optimization completed")
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
    
    def close(self) -> None:
        """Clean up resources"""
        if self.writer:
            self.writer.close()
        
        self._save_metadata()
        self._save_analytics()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()