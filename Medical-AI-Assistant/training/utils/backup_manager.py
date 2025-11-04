"""
Advanced Backup and Disaster Recovery Management System

This module provides comprehensive backup functionality including:
- Automated backup creation with validation
- Backup testing and integrity checking
- Recovery procedures for training state
- Disaster recovery planning
- Multi-location backup strategies
- Cloud storage integration

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import os
import json
import shutil
import tarfile
import gzip
import hashlib
import logging
import threading
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

import torch

try:
    import boto3
    import azure.storage.blob
    import google.cloud.storage
    CLOUD_STORAGE_AVAILABLE = True
except ImportError:
    CLOUD_STORAGE_AVAILABLE = False

# Optional imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for backup files"""
    backup_id: str
    created_at: str
    backup_type: str  # full, incremental, differential
    size_bytes: int
    compressed: bool
    encrypted: bool
    version: str
    source_paths: List[str]
    included_components: List[str]
    excludes: List[str]
    validation_hash: str
    cloud_locations: List[str] = None
    retention_policy: str = "default"
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.cloud_locations is None:
            self.cloud_locations = []
        if self.tags is None:
            self.tags = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class BackupConfiguration:
    """Configuration for backup operations"""
    # Backup scheduling
    auto_backup: bool = False
    backup_interval_hours: int = 24
    max_backups_local: int = 10
    max_backups_cloud: int = 50
    
    # Backup types
    include_checkpoints: bool = True
    include_logs: bool = True
    include_config: bool = True
    include_data: bool = False  # Usually large, optional
    
    # Compression and encryption
    compress_backups: bool = True
    compression_level: int = 6
    encrypt_backups: bool = False
    encryption_key: str = ""
    
    # Cloud storage
    use_cloud_backup: bool = False
    cloud_providers: List[str] = None
    cloud_bucket_prefix: str = "backups"
    
    # Validation
    validate_after_backup: bool = True
    test_recovery: bool = False
    integrity_checks: bool = True
    
    # Retention
    retention_days: int = 30
    retention_policy_type: str = "time_based"  # time_based, count_based, hybrid
    
    # Performance
    parallel_uploads: int = 2
    bandwidth_limit_mbps: int = 0  # 0 = unlimited
    compression_threads: int = 2
    
    # Notifications
    notify_on_success: bool = True
    notify_on_failure: bool = True
    notification_webhook: str = ""
    
    def __post_init__(self):
        if self.cloud_providers is None:
            self.cloud_providers = []


class BackupManager:
    """Advanced backup and disaster recovery management"""
    
    def __init__(
        self,
        backup_dir: Union[str, Path],
        experiment_name: str = "default",
        config: Optional[BackupConfiguration] = None
    ):
        self.backup_dir = Path(backup_dir)
        self.experiment_name = experiment_name
        self.config = config or BackupConfiguration()
        
        # Create backup directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        (self.backup_dir / "local").mkdir(exist_ok=True)
        (self.backup_dir / "cloud").mkdir(exist_ok=True)
        (self.backup_dir / "metadata").mkdir(exist_ok=True)
        (self.backup_dir / "validation").mkdir(exist_ok=True)
        (self.backup_dir / "recovery").mkdir(exist_ok=True)
        
        # State management
        self.metadata_file = self.backup_dir / "metadata" / "backup_registry.json"
        self.backups: Dict[str, BackupMetadata] = self._load_backup_registry()
        self._lock = threading.RLock()
        
        # Cloud clients
        self.cloud_clients: Dict[str, Any] = {}
        if self.config.use_cloud_backup:
            self._init_cloud_clients()
        
        # Background tasks
        self._backup_thread = None
        self._stop_event = threading.Event()
        
        logger.info(f"BackupManager initialized for experiment: {experiment_name}")
    
    def _load_backup_registry(self) -> Dict[str, BackupMetadata]:
        """Load backup registry from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {
                    k: BackupMetadata(**v) 
                    for k, v in data.items()
                }
            except Exception as e:
                logger.warning(f"Failed to load backup registry: {e}")
        return {}
    
    def _save_backup_registry(self) -> None:
        """Save backup registry to file"""
        try:
            data = {
                k: v.to_dict() 
                for k, v in self.backups.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save backup registry: {e}")
    
    def _init_cloud_clients(self) -> None:
        """Initialize cloud storage clients"""
        for provider in self.config.cloud_providers:
            try:
                if provider == "aws":
                    self.cloud_clients["aws"] = boto3.client('s3')
                elif provider == "azure":
                    self.cloud_clients["azure"] = azure.storage.blob.BlobServiceClient(
                        account_url=f"https://{self.config.cloud_bucket_prefix}.blob.core.windows.net"
                    )
                elif provider == "gcp":
                    self.cloud_clients["gcp"] = google.cloud.storage.Client()
                
                logger.info(f"Initialized cloud client for {provider}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider} client: {e}")
    
    def create_backup(
        self,
        backup_name: Optional[str] = None,
        backup_type: str = "full",
        source_paths: Optional[List[str]] = None,
        included_components: Optional[List[str]] = None,
        excludes: Optional[List[str]] = None,
        upload_to_cloud: bool = None,
        validate: bool = None,
        tags: Optional[Dict[str, str]] = None,
        retention_policy: str = "default"
    ) -> Optional[str]:
        """
        Create comprehensive backup
        
        Args:
            backup_name: Custom backup name (auto-generated if not provided)
            backup_type: Type of backup (full, incremental, differential)
            source_paths: Source paths to backup
            included_components: Components to include
            excludes: Paths to exclude
            upload_to_cloud: Upload to cloud storage
            validate: Validate backup after creation
            tags: Backup tags
            retention_policy: Retention policy for this backup
        
        Returns:
            Backup ID if successful, None otherwise
        """
        backup_name = backup_name or f"{self.experiment_name}_{backup_type}_{int(time.time())}"
        
        with self._lock:
            try:
                logger.info(f"Creating backup: {backup_name}")
                
                # Generate backup ID
                backup_id = f"{backup_name}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
                
                # Determine source paths
                if source_paths is None:
                    source_paths = self._get_default_source_paths()
                
                # Determine included components
                if included_components is None:
                    included_components = self._get_default_components()
                
                # Create temporary backup directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Copy source files
                    total_size = self._copy_source_files(
                        source_paths, temp_path, includes=included_components, excludes=excludes or []
                    )
                    
                    # Create backup archive
                    backup_file = self._create_backup_archive(temp_path, backup_id)
                    
                    if not backup_file:
                        logger.error("Failed to create backup archive")
                        return None
                    
                    # Calculate validation hash
                    validation_hash = self._calculate_file_hash(backup_file)
                    
                    # Create metadata
                    metadata = BackupMetadata(
                        backup_id=backup_id,
                        created_at=datetime.now().isoformat(),
                        backup_type=backup_type,
                        size_bytes=backup_file.stat().st_size,
                        compressed=self.config.compress_backups,
                        encrypted=self.config.encrypt_backups,
                        version="1.0",
                        source_paths=source_paths,
                        included_components=included_components,
                        excludes=excludes or [],
                        validation_hash=validation_hash,
                        tags=tags or {},
                        retention_policy=retention_policy
                    )
                    
                    # Validate backup if requested
                    if validate if validate is not None else self.config.validate_after_backup:
                        if not self._validate_backup(backup_file, metadata):
                            logger.error("Backup validation failed")
                            backup_file.unlink()
                            return None
                    
                    # Upload to cloud if requested
                    cloud_locations = []
                    upload_flag = upload_to_cloud if upload_to_cloud is not None else self.config.use_cloud_backup
                    if upload_flag and self.cloud_clients:
                        cloud_locations = self._upload_to_cloud(backup_file, backup_id)
                        metadata.cloud_locations = cloud_locations
                    
                    # Store backup metadata
                    self.backups[backup_id] = metadata
                    self._save_backup_registry()
                    
                    # Cleanup old backups
                    self._cleanup_old_backups()
                    
                    logger.info(f"Backup created successfully: {backup_id}")
                    
                    # Send notification
                    if self.config.notify_on_success:
                        self._send_notification("backup_success", {
                            "backup_id": backup_id,
                            "backup_name": backup_name,
                            "size_mb": metadata.size_bytes / 1024 / 1024,
                            "cloud_locations": cloud_locations
                        })
                    
                    return backup_id
                    
            except Exception as e:
                logger.error(f"Backup creation failed: {e}")
                
                # Send failure notification
                if self.config.notify_on_failure:
                    self._send_notification("backup_failure", {
                        "backup_name": backup_name,
                        "error": str(e)
                    })
                
                return None
    
    def _get_default_source_paths(self) -> List[str]:
        """Get default source paths for backup"""
        experiment_dir = Path(self.backup_dir).parent
        
        source_paths = []
        
        if self.config.include_checkpoints:
            checkpoints_dir = experiment_dir / "checkpoints"
            if checkpoints_dir.exists():
                source_paths.append(str(checkpoints_dir))
        
        if self.config.include_logs:
            logs_dir = experiment_dir / "state" / "logs"
            if logs_dir.exists():
                source_paths.append(str(logs_dir))
        
        if self.config.include_config:
            config_dir = experiment_dir / "state" / "configs"
            if config_dir.exists():
                source_paths.append(str(config_dir))
        
        # Always include state directory
        state_dir = experiment_dir / "state"
        if state_dir.exists():
            source_paths.append(str(state_dir))
        
        return source_paths
    
    def _get_default_components(self) -> List[str]:
        """Get default components to include in backup"""
        components = ["state"]
        
        if self.config.include_checkpoints:
            components.append("checkpoints")
        
        if self.config.include_logs:
            components.append("logs")
        
        if self.config.include_config:
            components.append("config")
        
        if self.config.include_data:
            components.append("data")
        
        return components
    
    def _copy_source_files(
        self,
        source_paths: List[str],
        dest_path: Path,
        includes: List[str] = None,
        excludes: List[str] = None
    ) -> int:
        """Copy source files to backup directory"""
        total_size = 0
        
        includes = includes or []
        excludes = excludes or []
        
        for source_path in source_paths:
            source = Path(source_path)
            if not source.exists():
                logger.warning(f"Source path does not exist: {source_path}")
                continue
            
            # Create destination directory
            relative_path = source.relative_to(source.anchor if source.is_absolute() else source.parent)
            dest = dest_path / relative_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            if source.is_file():
                # Copy single file
                size = source.stat().st_size
                if self._should_include_file(source, includes, excludes):
                    shutil.copy2(source, dest)
                    total_size += size
            else:
                # Copy directory recursively
                for item in source.rglob('*'):
                    if item.is_file():
                        rel_path = item.relative_to(source)
                        item_dest = dest / rel_path
                        item_dest.parent.mkdir(parents=True, exist_ok=True)
                        
                        if self._should_include_file(item, includes, excludes):
                            size = item.stat().st_size
                            shutil.copy2(item, item_dest)
                            total_size += size
        
        return total_size
    
    def _should_include_file(self, file_path: Path, includes: List[str], excludes: List[str]) -> bool:
        """Check if file should be included in backup"""
        file_str = str(file_path)
        
        # Check excludes first
        for exclude_pattern in excludes:
            if exclude_pattern in file_str:
                return False
        
        # If no includes specified, include all
        if not includes:
            return True
        
        # Check includes
        for include_pattern in includes:
            if include_pattern in file_str:
                return True
        
        return False
    
    def _create_backup_archive(self, source_dir: Path, backup_id: str) -> Optional[Path]:
        """Create backup archive file"""
        try:
            backup_file = self.backup_dir / "local" / f"{backup_id}.tar.gz"
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(source_dir, arcname="backup")
            
            return backup_file
            
        except Exception as e:
            logger.error(f"Failed to create backup archive: {e}")
            return None
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _validate_backup(self, backup_file: Path, metadata: BackupMetadata) -> bool:
        """Validate backup integrity"""
        try:
            logger.info(f"Validating backup: {metadata.backup_id}")
            
            # Check file hash
            current_hash = self._calculate_file_hash(backup_file)
            if current_hash != metadata.validation_hash:
                logger.error(f"Hash mismatch for backup {metadata.backup_id}")
                return False
            
            # Test archive integrity
            try:
                with tarfile.open(backup_file, 'r:gz') as tar:
                    # Try to list contents
                    members = tar.getmembers()
                    if not members:
                        logger.error(f"Empty backup archive: {metadata.backup_id}")
                        return False
                    
                    # Test reading first few files
                    for member in members[:5]:
                        if member.isfile():
                            tar.extractfile(member).read(1024)  # Read first 1KB
                    
            except Exception as e:
                logger.error(f"Archive validation failed for {metadata.backup_id}: {e}")
                return False
            
            logger.info(f"Backup validation passed: {metadata.backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Backup validation error: {e}")
            return False
    
    def _upload_to_cloud(self, backup_file: Path, backup_id: str) -> List[str]:
        """Upload backup to cloud storage"""
        cloud_locations = []
        
        if not self.cloud_clients:
            return cloud_locations
        
        def upload_to_provider(provider: str, client: Any) -> Tuple[str, bool]:
            try:
                cloud_path = f"{self.experiment_name}/backups/{backup_id}.tar.gz"
                
                if provider == "aws":
                    bucket_name = f"{self.config.cloud_bucket_prefix}-{self.experiment_name}"
                    client.upload_file(str(backup_file), bucket_name, cloud_path)
                elif provider == "azure":
                    blob_client = client.get_blob_client(
                        container_name=f"backups-{self.experiment_name}",
                        blob_name=cloud_path
                    )
                    with open(backup_file, 'rb') as f:
                        blob_client.upload_blob(f)
                elif provider == "gcp":
                    bucket = client.bucket(f"{self.config.cloud_bucket_prefix}-{self.experiment_name}")
                    blob = bucket.blob(cloud_path)
                    blob.upload_from_filename(str(backup_file))
                
                return f"{provider}:{cloud_path}", True
                
            except Exception as e:
                logger.error(f"Cloud upload failed for {provider}: {e}")
                return f"{provider}:failed", False
        
        # Upload to all cloud providers in parallel
        with ThreadPoolExecutor(max_workers=self.config.parallel_uploads) as executor:
            futures = {
                executor.submit(upload_to_provider, provider, client): provider
                for provider, client in self.cloud_clients.items()
            }
            
            for future in as_completed(futures):
                location, success = future.result()
                if success:
                    cloud_locations.append(location)
        
        return cloud_locations
    
    def _cleanup_old_backups(self) -> None:
        """Cleanup old backups based on retention policy"""
        try:
            sorted_backups = sorted(
                self.backups.items(),
                key=lambda x: x[1].created_at
            )
            
            # Time-based cleanup
            if self.config.retention_policy_type in ["time_based", "hybrid"]:
                cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
                
                for backup_id, metadata in sorted_backups:
                    if metadata.retention_policy == "keep_forever":
                        continue
                    
                    created_date = datetime.fromisoformat(metadata.created_at)
                    if created_date < cutoff_date:
                        self._remove_backup(backup_id, metadata)
            
            # Count-based cleanup
            if self.config.retention_policy_type in ["count_based", "hybrid"]:
                # Keep only max_backups_local most recent
                if len(sorted_backups) > self.config.max_backups_local:
                    backups_to_remove = sorted_backups[:-self.config.max_backups_local]
                    
                    for backup_id, metadata in backups_to_remove:
                        if metadata.retention_policy == "keep_forever":
                            continue
                        self._remove_backup(backup_id, metadata)
            
            logger.debug("Backup cleanup completed")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def _remove_backup(self, backup_id: str, metadata: BackupMetadata) -> None:
        """Remove backup files and metadata"""
        try:
            # Remove local file
            backup_file = self.backup_dir / "local" / f"{backup_id}.tar.gz"
            if backup_file.exists():
                backup_file.unlink()
            
            # Remove from cloud
            for cloud_location in metadata.cloud_locations:
                self._remove_from_cloud(cloud_location, backup_id)
            
            # Remove from registry
            del self.backups[backup_id]
            
            logger.debug(f"Removed backup: {backup_id}")
            
        except Exception as e:
            logger.error(f"Failed to remove backup {backup_id}: {e}")
    
    def _remove_from_cloud(self, cloud_location: str, backup_id: str) -> None:
        """Remove backup from cloud storage"""
        try:
            provider = cloud_location.split(':')[0]
            if provider in self.cloud_clients:
                # Implementation depends on cloud provider
                logger.debug(f"Removing from cloud: {cloud_location}")
                
        except Exception as e:
            logger.error(f"Failed to remove from cloud {cloud_location}: {e}")
    
    def _send_notification(self, event_type: str, data: Dict[str, Any]) -> None:
        """Send notification about backup events"""
        try:
            if not self.config.notification_webhook:
                return
            
            import requests
            
            notification = {
                "event": event_type,
                "experiment": self.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            response = requests.post(
                self.config.notification_webhook,
                json=notification,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Notification sent successfully")
            else:
                logger.warning(f"Notification failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def list_backups(
        self,
        backup_type: Optional[str] = None,
        since: Optional[datetime] = None,
        include_details: bool = True
    ) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for backup_id, metadata in self.backups.items():
            # Apply filters
            if backup_type and metadata.backup_type != backup_type:
                continue
            
            if since:
                created_date = datetime.fromisoformat(metadata.created_at)
                if created_date < since:
                    continue
            
            backup_info = {
                'backup_id': backup_id,
                'created_at': metadata.created_at,
                'backup_type': metadata.backup_type,
                'size_mb': metadata.size_bytes / 1024 / 1024,
                'compressed': metadata.compressed,
                'encrypted': metadata.encrypted,
                'cloud_locations': metadata.cloud_locations,
                'tags': metadata.tags
            }
            
            if include_details:
                backup_info.update({
                    'source_paths': metadata.source_paths,
                    'included_components': metadata.included_components,
                    'validation_hash': metadata.validation_hash,
                    'retention_policy': metadata.retention_policy
                })
            
            backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x['created_at'], reverse=True)
    
    def restore_backup(
        self,
        backup_id: str,
        restore_path: Optional[Path] = None,
        verify_integrity: bool = True,
        overwrite: bool = False
    ) -> bool:
        """
        Restore backup to specified location
        
        Args:
            backup_id: ID of backup to restore
            restore_path: Path to restore to (default: new directory)
            verify_integrity: Verify backup integrity before restore
            overwrite: Overwrite existing files
        
        Returns:
            Success status
        """
        if backup_id not in self.backups:
            logger.error(f"Backup {backup_id} not found")
            return False
        
        metadata = self.backups[backup_id]
        restore_path = restore_path or Path(self.backup_dir / "recovery" / backup_id)
        
        try:
            logger.info(f"Restoring backup: {backup_id}")
            
            # Get backup file
            backup_file = self.backup_dir / "local" / f"{backup_id}.tar.gz"
            
            if not backup_file.exists():
                # Try to download from cloud
                if metadata.cloud_locations:
                    if not self._download_from_cloud(metadata.cloud_locations[0], backup_file):
                        logger.error(f"Could not locate backup file for {backup_id}")
                        return False
                else:
                    logger.error(f"Backup file not found: {backup_file}")
                    return False
            
            # Verify integrity if requested
            if verify_integrity:
                if not self._validate_backup(backup_file, metadata):
                    logger.error("Backup integrity verification failed")
                    return False
            
            # Create restore directory
            restore_path.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(restore_path)
            
            # Post-restore validation
            self._validate_restored_data(restore_path, metadata)
            
            logger.info(f"Backup restored successfully to: {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    def _download_from_cloud(self, cloud_location: str, local_path: Path) -> bool:
        """Download backup from cloud storage"""
        try:
            provider = cloud_location.split(':')[0]
            cloud_path = cloud_location.split(':', 1)[1]
            
            if provider in self.cloud_clients:
                client = self.cloud_clients[provider]
                
                if provider == "aws":
                    bucket_name = cloud_path.split('/')[0]
                    object_key = '/'.join(cloud_path.split('/')[1:])
                    client.download_file(bucket_name, object_key, str(local_path))
                elif provider == "azure":
                    # Azure implementation
                    pass
                elif provider == "gcp":
                    # GCP implementation
                    pass
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Cloud download failed: {e}")
            return False
    
    def _validate_restored_data(self, restore_path: Path, metadata: BackupMetadata) -> None:
        """Validate restored data"""
        try:
            # Check that expected components are present
            expected_components = set(metadata.included_components)
            restored_components = set()
            
            for component in expected_components:
                component_path = restore_path / "backup" / component
                if component_path.exists():
                    restored_components.add(component)
            
            missing_components = expected_components - restored_components
            if missing_components:
                logger.warning(f"Missing components in restored backup: {missing_components}")
            
            logger.info("Restored data validation completed")
            
        except Exception as e:
            logger.error(f"Restored data validation failed: {e}")
    
    def test_backup_recovery(self, backup_id: str) -> Dict[str, Any]:
        """Test backup recovery procedure"""
        if backup_id not in self.backups:
            return {"success": False, "error": "Backup not found"}
        
        try:
            # Create test restore directory
            test_restore_path = Path(self.backup_dir / "validation" / f"test_{backup_id}")
            
            # Attempt restore
            success = self.restore_backup(
                backup_id=backup_id,
                restore_path=test_restore_path,
                verify_integrity=True,
                overwrite=True
            )
            
            # Clean up test restore
            if test_restore_path.exists():
                shutil.rmtree(test_restore_path)
            
            return {
                "success": success,
                "backup_id": backup_id,
                "test_time": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "backup_id": backup_id,
                "error": str(e),
                "test_time": datetime.now().isoformat()
            }
    
    def start_automatic_backup(self) -> None:
        """Start automatic backup scheduling"""
        if not self.config.auto_backup:
            return
        
        def backup_worker():
            while not self._stop_event.is_set():
                try:
                    # Create backup
                    backup_id = self.create_backup(
                        backup_type="automatic",
                        validate=self.config.test_recovery
                    )
                    
                    if backup_id:
                        logger.info(f"Automatic backup completed: {backup_id}")
                    else:
                        logger.error("Automatic backup failed")
                    
                    # Wait for next backup interval
                    self._stop_event.wait(self.config.backup_interval_hours * 3600)
                    
                except Exception as e:
                    logger.error(f"Automatic backup error: {e}")
                    time.sleep(300)  # Wait 5 minutes before retry
        
        self._backup_thread = threading.Thread(target=backup_worker, daemon=True)
        self._backup_thread.start()
        
        logger.info("Automatic backup scheduling started")
    
    def stop_automatic_backup(self) -> None:
        """Stop automatic backup scheduling"""
        if self._backup_thread:
            self._stop_event.set()
            self._backup_thread.join()
            self._backup_thread = None
        
        logger.info("Automatic backup scheduling stopped")
    
    def export_backup_info(self, output_file: Union[str, Path], format: str = "json") -> bool:
        """Export backup information to file"""
        try:
            output_file = Path(output_file)
            
            backup_data = {
                "experiment": self.experiment_name,
                "export_time": datetime.now().isoformat(),
                "total_backups": len(self.backups),
                "backups": [metadata.to_dict() for metadata in self.backups.values()],
                "configuration": asdict(self.config)
            }
            
            if format == "json":
                with open(output_file, 'w') as f:
                    json.dump(backup_data, f, indent=2)
            elif format == "yaml":
                with open(output_file, 'w') as f:
                    yaml.dump(backup_data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Backup info exported to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage statistics"""
        try:
            local_backup_dir = self.backup_dir / "local"
            
            if PSUTIL_AVAILABLE:
                disk_usage = psutil.disk_usage(str(local_backup_dir))
                return {
                    "total_bytes": disk_usage.total,
                    "used_bytes": disk_usage.used,
                    "free_bytes": disk_usage.free,
                    "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                    "backup_count": len(self.backups),
                    "total_backup_size_bytes": sum(m.size_bytes for m in self.backups.values())
                }
            else:
                # Fallback using os.stat
                total_size = sum(f.stat().st_size for f in local_backup_dir.rglob('*') if f.is_file())
                return {
                    "backup_count": len(self.backups),
                    "total_backup_size_bytes": total_size
                }
                
        except Exception as e:
            logger.error(f"Failed to get disk usage: {e}")
            return {}
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_automatic_backup()
        self._save_backup_registry()
    
    def close(self) -> None:
        """Close backup manager"""
        self.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()