"""
Demo Backup and Recovery System
Provides automated backup and recovery procedures for demonstration reliability
"""

import os
import shutil
import json
import sqlite3
import gzip
import tarfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import logging

@dataclass
class BackupMetadata:
    """Backup file metadata"""
    backup_id: str
    timestamp: datetime
    backup_type: str  # 'full', 'incremental', 'demo_reset'
    size_bytes: int
    checksum: str
    source_path: str
    description: str
    components: List[str]  # 'database', 'models', 'config', 'analytics'

@dataclass
class RecoveryPoint:
    """Recovery point information"""
    recovery_id: str
    timestamp: datetime
    backup_id: str
    description: str
    demo_state: str  # 'fresh', 'configured', 'active'
    verification_status: str  # 'verified', 'corrupted', 'pending'

class DemoBackupManager:
    """Manages backup and recovery for demo environment"""
    
    def __init__(self, backup_dir: str = "./backups", 
                 demo_data_dir: str = "./demo"):
        self.backup_dir = backup_dir
        self.demo_data_dir = demo_data_dir
        self.metadata_file = os.path.join(backup_dir, "backup_metadata.json")
        
        # Ensure backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for backup operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.backup_dir, 'backup.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_full_backup(self, description: str = "Full demo environment backup") -> BackupMetadata:
        """Create a complete backup of demo environment"""
        backup_id = f"full_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now()
        
        self.logger.info(f"Starting full backup: {backup_id}")
        
        # Backup database
        db_backup_path = self._backup_database()
        
        # Backup configuration files
        config_backup_path = self._backup_config_files()
        
        # Backup analytics data
        analytics_backup_path = self._backup_analytics_data()
        
        # Backup models and trained data
        models_backup_path = self._backup_models()
        
        # Create backup manifest
        manifest = {
            "backup_id": backup_id,
            "timestamp": timestamp.isoformat(),
            "backup_type": "full",
            "description": description,
            "components": ["database", "config", "analytics", "models"],
            "files": {
                "database": db_backup_path,
                "config": config_backup_path,
                "analytics": analytics_backup_path,
                "models": models_backup_path
            }
        }
        
        # Create compressed archive
        backup_archive = os.path.join(self.backup_dir, f"{backup_id}.tar.gz")
        
        with tarfile.open(backup_archive, "w:gz") as tar:
            if db_backup_path and os.path.exists(db_backup_path):
                tar.add(db_backup_path, arcname="database.db")
            if config_backup_path and os.path.exists(config_backup_path):
                tar.add(config_backup_path, arcname="config.json")
            if analytics_backup_path and os.path.exists(analytics_backup_path):
                tar.add(analytics_backup_path, arcname="analytics.db")
            if models_backup_path and os.path.exists(models_backup_path):
                tar.add(models_backup_path, arcname="models/")
            
            # Add manifest
            manifest_content = json.dumps(manifest, indent=2)
            tar.addfile(tarfile.TarInfo("manifest.json"), 
                       fileobj=StringIO(manifest_content))
        
        # Calculate size and checksum
        size_bytes = os.path.getsize(backup_archive)
        checksum = self._calculate_checksum(backup_archive)
        
        # Create metadata
        metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=timestamp,
            backup_type="full",
            size_bytes=size_bytes,
            checksum=checksum,
            source_path=backup_archive,
            description=description,
            components=["database", "config", "analytics", "models"]
        )
        
        # Save metadata
        self._save_backup_metadata(metadata)
        
        self.logger.info(f"Full backup completed: {backup_id} ({size_bytes} bytes)")
        return metadata
        
    def create_demo_reset_backup(self) -> BackupMetadata:
        """Create backup before demo reset"""
        return self.create_full_backup("Pre-reset demo state backup")
        
    def restore_backup(self, backup_id: str, 
                      components: Optional[List[str]] = None) -> bool:
        """Restore from backup"""
        self.logger.info(f"Starting restore from backup: {backup_id}")
        
        # Load backup metadata
        metadata = self._load_backup_metadata(backup_id)
        if not metadata:
            self.logger.error(f"Backup metadata not found: {backup_id}")
            return False
            
        # Verify backup integrity
        if not self._verify_backup_integrity(backup_id):
            self.logger.error(f"Backup integrity check failed: {backup_id}")
            return False
            
        # Create backup before restore
        pre_restore_backup = self.create_demo_reset_backup()
        self.logger.info(f"Created pre-restore backup: {pre_restore_backup.backup_id}")
        
        backup_archive = metadata.source_path
        
        # Extract backup archive
        extract_dir = os.path.join(self.backup_dir, f"restore_{backup_id}")
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(backup_archive, "r:gz") as tar:
            tar.extractall(extract_dir)
            
        # Restore components
        success = True
        restore_components = components or ["database", "config", "analytics", "models"]
        
        for component in restore_components:
            if not self._restore_component(component, extract_dir):
                self.logger.error(f"Failed to restore component: {component}")
                success = False
                
        # Cleanup extraction directory
        shutil.rmtree(extract_dir)
        
        if success:
            self.logger.info(f"Backup restore completed successfully: {backup_id}")
            # Create recovery point
            self._create_recovery_point(backup_id, "Restore from backup")
        else:
            self.logger.error(f"Backup restore failed: {backup_id}")
            
        return success
        
    def restore_to_fresh_demo(self) -> bool:
        """Restore to fresh demo state"""
        self.logger.info("Restoring to fresh demo state")
        
        # Get the most recent fresh demo backup
        fresh_backups = self._get_backups_by_type("full")
        if not fresh_backups:
            self.logger.error("No fresh demo backup found")
            return False
            
        latest_backup = fresh_backups[0]  # Most recent
        return self.restore_backup(latest_backup.backup_id)
        
    def schedule_auto_backup(self):
        """Schedule automatic backups"""
        # Check for backups older than 24 hours
        latest_backup = self._get_latest_backup()
        
        if not latest_backup or self._is_backup_stale(latest_backup):
            backup = self.create_full_backup("Auto backup - scheduled")
            self.logger.info(f"Auto backup created: {backup.backup_id}")
            
    def verify_demo_state(self) -> Dict[str, bool]:
        """Verify demo environment state"""
        verification = {
            "database_accessible": False,
            "config_valid": False,
            "analytics_working": False,
            "models_loaded": False,
            "all_tests_passed": False
        }
        
        # Test database
        try:
            conn = sqlite3.connect("demo.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            verification["database_accessible"] = True
            conn.close()
        except Exception as e:
            self.logger.error(f"Database verification failed: {e}")
            
        # Test configuration
        try:
            if os.path.exists("demo/config/demo_settings.py"):
                verification["config_valid"] = True
        except Exception as e:
            self.logger.error(f"Config verification failed: {e}")
            
        # Test analytics
        try:
            if os.path.exists("demo_analytics.db"):
                conn = sqlite3.connect("demo_analytics.db")
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                verification["analytics_working"] = True
                conn.close()
        except Exception as e:
            self.logger.error(f"Analytics verification failed: {e}")
            
        # Test models
        try:
            if os.path.exists("models/") or os.path.exists("training/outputs/"):
                verification["models_loaded"] = True
        except Exception as e:
            self.logger.error(f"Models verification failed: {e}")
            
        verification["all_tests_passed"] = all(verification.values())
        
        return verification
        
    def cleanup_old_backups(self, retention_days: int = 7):
        """Clean up old backup files"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        metadata_list = self._load_all_backup_metadata()
        
        for backup_id, metadata in metadata_list.items():
            if metadata.timestamp < cutoff_date:
                # Remove backup file
                if os.path.exists(metadata.source_path):
                    os.remove(metadata.source_path)
                    self.logger.info(f"Removed old backup: {backup_id}")
                    
                # Remove metadata entry
                self._remove_backup_metadata(backup_id)
                
    def get_demo_readiness_report(self) -> Dict:
        """Generate demo readiness report"""
        verification = self.verify_demo_state()
        latest_backup = self._get_latest_backup()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "demo_state": {
                "ready": verification["all_tests_passed"],
                "verification_results": verification
            },
            "backup_status": {
                "latest_backup": latest_backup.backup_id if latest_backup else None,
                "backup_age_hours": self._get_backup_age_hours(latest_backup) if latest_backup else None
            },
            "recommendations": self._generate_recommendations(verification, latest_backup)
        }
        
    def _backup_database(self) -> Optional[str]:
        """Backup demo database"""
        db_path = "demo.db"
        if not os.path.exists(db_path):
            return None
            
        backup_path = os.path.join(self.backup_dir, f"database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy2(db_path, backup_path)
        return backup_path
        
    def _backup_config_files(self) -> Optional[str]:
        """Backup configuration files"""
        config_files = [
            "demo/config/demo_settings.py",
            ".env",
            "config.yaml"
        ]
        
        config_backup = os.path.join(self.backup_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        config_data = {}
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        config_data[config_file] = f.read()
                except Exception as e:
                    self.logger.warning(f"Could not backup config file {config_file}: {e}")
                    
        with open(config_backup, 'w') as f:
            json.dump(config_data, f, indent=2)
            
        return config_backup
        
    def _backup_analytics_data(self) -> Optional[str]:
        """Backup analytics database"""
        analytics_path = "demo_analytics.db"
        if not os.path.exists(analytics_path):
            return None
            
        backup_path = os.path.join(self.backup_dir, f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        shutil.copy2(analytics_path, backup_path)
        return backup_path
        
    def _backup_models(self) -> Optional[str]:
        """Backup trained models"""
        model_dirs = ["models/", "training/outputs/", "training/checkpoints/"]
        
        models_backup = os.path.join(self.backup_dir, f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if not any(os.path.exists(d) for d in model_dirs):
            return None
            
        os.makedirs(models_backup, exist_ok=True)
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                dest_dir = os.path.join(models_backup, os.path.basename(model_dir))
                shutil.copytree(model_dir, dest_dir, dirs_exist_ok=True)
                
        return models_backup
        
    def _restore_component(self, component: str, extract_dir: str) -> bool:
        """Restore a specific component"""
        try:
            if component == "database":
                src = os.path.join(extract_dir, "database.db")
                if os.path.exists(src):
                    shutil.copy2(src, "demo.db")
                    return True
                    
            elif component == "config":
                src = os.path.join(extract_dir, "config.json")
                if os.path.exists(src):
                    with open(src, 'r') as f:
                        config_data = json.load(f)
                    for config_file, content in config_data.items():
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, 'w') as f:
                            f.write(content)
                    return True
                    
            elif component == "analytics":
                src = os.path.join(extract_dir, "analytics.db")
                if os.path.exists(src):
                    shutil.copy2(src, "demo_analytics.db")
                    return True
                    
            elif component == "models":
                src = os.path.join(extract_dir, "models")
                if os.path.exists(src):
                    if os.path.exists("models/"):
                        shutil.rmtree("models/")
                    shutil.copytree(src, "models/")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error restoring {component}: {e}")
            
        return False
        
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum"""
        import hashlib
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def _verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify backup file integrity"""
        metadata = self._load_backup_metadata(backup_id)
        if not metadata:
            return False
            
        if not os.path.exists(metadata.source_path):
            return False
            
        # Verify checksum
        current_checksum = self._calculate_checksum(metadata.source_path)
        return current_checksum == metadata.checksum
        
    def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata"""
        metadata_list = self._load_all_backup_metadata()
        metadata_list[metadata.backup_id] = metadata
        
        # Convert to serializable format
        serializable_data = {}
        for backup_id, meta in metadata_list.items():
            serializable_data[backup_id] = {
                "backup_id": meta.backup_id,
                "timestamp": meta.timestamp.isoformat(),
                "backup_type": meta.backup_type,
                "size_bytes": meta.size_bytes,
                "checksum": meta.checksum,
                "source_path": meta.source_path,
                "description": meta.description,
                "components": meta.components
            }
            
        with open(self.metadata_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load specific backup metadata"""
        metadata_list = self._load_all_backup_metadata()
        
        if backup_id in metadata_list:
            meta_data = metadata_list[backup_id]
            return BackupMetadata(
                backup_id=meta_data["backup_id"],
                timestamp=datetime.fromisoformat(meta_data["timestamp"]),
                backup_type=meta_data["backup_type"],
                size_bytes=meta_data["size_bytes"],
                checksum=meta_data["checksum"],
                source_path=meta_data["source_path"],
                description=meta_data["description"],
                components=meta_data["components"]
            )
        return None
        
    def _load_all_backup_metadata(self) -> Dict:
        """Load all backup metadata"""
        if not os.path.exists(self.metadata_file):
            return {}
            
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
            
    def _get_latest_backup(self) -> Optional[BackupMetadata]:
        """Get the most recent backup"""
        metadata_list = self._load_all_backup_metadata()
        
        if not metadata_list:
            return None
            
        # Sort by timestamp
        sorted_backups = sorted(
            metadata_list.items(),
            key=lambda x: datetime.fromisoformat(x[1]["timestamp"]),
            reverse=True
        )
        
        latest_data = sorted_backups[0][1]
        return BackupMetadata(
            backup_id=latest_data["backup_id"],
            timestamp=datetime.fromisoformat(latest_data["timestamp"]),
            backup_type=latest_data["backup_type"],
            size_bytes=latest_data["size_bytes"],
            checksum=latest_data["checksum"],
            source_path=latest_data["source_path"],
            description=latest_data["description"],
            components=latest_data["components"]
        )
        
    def _create_recovery_point(self, backup_id: str, description: str):
        """Create a recovery point record"""
        recovery_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        recovery_point = {
            "recovery_id": recovery_id,
            "timestamp": datetime.now().isoformat(),
            "backup_id": backup_id,
            "description": description,
            "demo_state": "active",
            "verification_status": "verified"
        }
        
        recovery_file = os.path.join(self.backup_dir, "recovery_points.json")
        
        if os.path.exists(recovery_file):
            with open(recovery_file, 'r') as f:
                recovery_data = json.load(f)
        else:
            recovery_data = []
            
        recovery_data.append(recovery_point)
        
        with open(recovery_file, 'w') as f:
            json.dump(recovery_data, f, indent=2)
            
    def _generate_recommendations(self, verification: Dict, latest_backup: Optional[BackupMetadata]) -> List[str]:
        """Generate demo readiness recommendations"""
        recommendations = []
        
        if not verification["database_accessible"]:
            recommendations.append("Database connection failed - restore from backup")
            
        if not verification["config_valid"]:
            recommendations.append("Configuration files missing or invalid")
            
        if not verification["analytics_working"]:
            recommendations.append("Analytics database not accessible")
            
        if latest_backup and self._get_backup_age_hours(latest_backup) > 48:
            recommendations.append("Last backup is older than 48 hours - create new backup")
            
        if verification["all_tests_passed"]:
            recommendations.append("Demo environment is ready for presentations")
            
        return recommendations

# Additional utilities
class StringIO:
    """Simple StringIO replacement for Python compatibility"""
    def __init__(self, content=""):
        self.content = content
        self.position = 0
        
    def write(self, data):
        self.content = data
        return len(data)
        
    def read(self):
        result = self.content[self.position:]
        self.position = len(self.content)
        return result
        
    def readline(self):
        lines = self.content.split('\n')
        if self.position < len(lines):
            line = lines[self.position]
            self.position += 1
            return line + '\n'
        return ""

if __name__ == "__main__":
    # Test backup system
    backup_manager = DemoBackupManager()
    
    print("Demo Backup System Test")
    print("=" * 30)
    
    # Create backup
    backup = backup_manager.create_full_backup("Test backup")
    print(f"Backup created: {backup.backup_id}")
    
    # Verify demo state
    readiness = backup_manager.get_demo_readiness_report()
    print(f"Demo ready: {readiness['demo_state']['ready']}")
    print(f"Recommendations: {readiness['recommendations']}")