"""
Adapter Registry and Versioning System

Provides comprehensive registry with metadata storage, versioning,
and production-ready adapter management.
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import uuid4

import psutil
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class AdapterStatus(Enum):
    """Adapter deployment status."""
    PENDING = "pending"
    VALIDATING = "validating"
    READY = "ready"
    LOADING = "loading"
    ACTIVE = "active"
    DEACTIVATED = "deactivated"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class AdapterType(Enum):
    """Supported adapter types."""
    LORA = "lora"
    ADALORA = "adalora"
    IA3 = "ia3"
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"
    MEDICAL_LORA = "medical_lora"
    CLINICAL_LORA = "clinical_lora"


@dataclass
class AdapterVersion:
    """Represents an adapter version."""
    version_id: str
    version: str
    adapter_type: AdapterType
    base_model_id: str
    created_at: datetime
    created_by: str
    description: str
    tags: List[str] = field(default_factory=list)
    file_size: int = 0
    checksum: str = ""
    
    @property
    def display_name(self) -> str:
        """Get display name for version."""
        return f"{self.adapter_type.value}_{self.version}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['adapter_type'] = self.adapter_type.value
        data['created_at'] = self.created_at.isoformat()
        return data


@dataclass
class AdapterMetadata:
    """Comprehensive adapter metadata."""
    adapter_id: str
    name: str
    description: str
    adapter_type: AdapterType
    versions: List[AdapterVersion] = field(default_factory=list)
    
    # Medical model specific
    medical_domain: Optional[str] = None
    clinical_use_case: Optional[str] = None
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    safety_score: float = 0.0
    compliance_flags: List[str] = field(default_factory=list)
    
    # Performance metrics
    avg_load_time: float = 0.0
    memory_footprint: float = 0.0
    inference_latency_ms: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    
    # Usage statistics
    total_loads: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    last_used: Optional[datetime] = None
    
    # Compatibility
    compatible_models: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    
    # Quality assurance
    validation_status: str = "unvalidated"
    test_coverage: float = 0.0
    security_scan: str = "not_scanned"
    
    def get_active_version(self) -> Optional[AdapterVersion]:
        """Get currently active version."""
        # For now, return latest version
        return max(self.versions, key=lambda v: v.created_at) if self.versions else None
    
    def get_version_count(self) -> int:
        """Get number of versions."""
        return len(self.versions)
    
    def get_success_rate(self) -> float:
        """Get successful load rate."""
        if self.total_loads == 0:
            return 0.0
        return self.successful_loads / self.total_loads
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['adapter_type'] = self.adapter_type.value
        data['created_at'] = data.get('last_used')
        if data.get('last_used'):
            data['last_used'] = data['last_used'].isoformat() if isinstance(data['last_used'], datetime) else data['last_used']
        return data


class AdapterRegistry:
    """
    Production-grade adapter registry with SQLite backend.
    
    Features:
    - SQLite-backed storage for reliability
    - Thread-safe operations
    - Version management
    - Usage tracking
    - Medical compliance validation
    """
    
    def __init__(self, registry_path: str = "./adapter_registry.db"):
        self.registry_path = Path(registry_path)
        self._lock = threading.RLock()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.registry_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adapters (
                    adapter_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    adapter_type TEXT NOT NULL,
                    medical_domain TEXT,
                    clinical_use_case TEXT,
                    safety_score REAL DEFAULT 0.0,
                    compliance_flags TEXT DEFAULT '[]',
                    avg_load_time REAL DEFAULT 0.0,
                    memory_footprint REAL DEFAULT 0.0,
                    inference_latency_ms REAL DEFAULT 0.0,
                    throughput_tokens_per_sec REAL DEFAULT 0.0,
                    total_loads INTEGER DEFAULT 0,
                    successful_loads INTEGER DEFAULT 0,
                    failed_loads INTEGER DEFAULT 0,
                    last_used TEXT,
                    compatible_models TEXT DEFAULT '[]',
                    required_resources TEXT DEFAULT '{}',
                    validation_status TEXT DEFAULT 'unvalidated',
                    test_coverage REAL DEFAULT 0.0,
                    security_scan TEXT DEFAULT 'not_scanned',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adapter_versions (
                    version_id TEXT PRIMARY KEY,
                    adapter_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    base_model_id TEXT NOT NULL,
                    description TEXT,
                    tags TEXT DEFAULT '[]',
                    file_size INTEGER DEFAULT 0,
                    checksum TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    FOREIGN KEY (adapter_id) REFERENCES adapters (adapter_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adapter_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (adapter_id) REFERENCES adapters (adapter_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adapter_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    duration_ms REAL,
                    success BOOLEAN,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (adapter_id) REFERENCES adapters (adapter_id)
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_adapter_type ON adapters (adapter_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_version_adapter ON adapter_versions (adapter_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_validation_adapter ON validation_metrics (adapter_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_adapter ON usage_statistics (adapter_id)")
            
            conn.commit()
            
        logger.info(f"Adapter registry initialized: {self.registry_path}")
    
    def register_adapter(self, metadata: AdapterMetadata) -> bool:
        """Register a new adapter."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute("""
                        INSERT OR REPLACE INTO adapters (
                            adapter_id, name, description, adapter_type,
                            medical_domain, clinical_use_case, safety_score,
                            compliance_flags, compatible_models, required_resources,
                            validation_status, test_coverage, security_scan,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        metadata.adapter_id, metadata.name, metadata.description,
                        metadata.adapter_type.value, metadata.medical_domain,
                        metadata.clinical_use_case, metadata.safety_score,
                        json.dumps(metadata.compliance_flags),
                        json.dumps(metadata.compatible_models),
                        json.dumps(metadata.required_resources),
                        metadata.validation_status, metadata.test_coverage,
                        metadata.security_scan, now, now
                    ))
                    conn.commit()
                    
                    logger.info(f"Registered adapter: {metadata.adapter_id}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to register adapter {metadata.adapter_id}: {e}")
                return False
    
    def add_version(self, adapter_id: str, version: AdapterVersion) -> bool:
        """Add a new version to an adapter."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO adapter_versions (
                            version_id, adapter_id, version, base_model_id,
                            description, tags, file_size, checksum,
                            created_at, created_by, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        version.version_id, adapter_id, version.version,
                        version.base_model_id, version.description,
                        json.dumps(version.tags), version.file_size,
                        version.checksum, version.created_at.isoformat(),
                        version.created_by, AdapterStatus.PENDING.value
                    ))
                    conn.commit()
                    
                    # Update adapter metadata
                    self._update_adapter_stats(adapter_id)
                    
                    logger.info(f"Added version {version.version} for adapter {adapter_id}")
                    return True
                    
            except Exception as e:
                logger.error(f"Failed to add version for adapter {adapter_id}: {e}")
                return False
    
    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get adapter metadata."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    # Get adapter data
                    cursor = conn.execute("""
                        SELECT * FROM adapters WHERE adapter_id = ?
                    """, (adapter_id,))
                    row = cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    # Build adapter metadata
                    metadata = AdapterMetadata(
                        adapter_id=row[0],
                        name=row[1],
                        description=row[2],
                        adapter_type=AdapterType(row[3]),
                        medical_domain=row[4],
                        clinical_use_case=row[5],
                        safety_score=row[6],
                        compliance_flags=json.loads(row[7] or '[]'),
                        avg_load_time=row[8],
                        memory_footprint=row[9],
                        inference_latency_ms=row[10],
                        throughput_tokens_per_sec=row[11],
                        total_loads=row[12],
                        successful_loads=row[13],
                        failed_loads=row[14],
                        last_used=datetime.fromisoformat(row[15]) if row[15] else None,
                        compatible_models=json.loads(row[16] or '[]'),
                        required_resources=json.loads(row[17] or '{}'),
                        validation_status=row[18],
                        test_coverage=row[19],
                        security_scan=row[20]
                    )
                    
                    # Get versions
                    cursor = conn.execute("""
                        SELECT * FROM adapter_versions WHERE adapter_id = ?
                        ORDER BY created_at DESC
                    """, (adapter_id,))
                    
                    versions = []
                    for row in cursor.fetchall():
                        version = AdapterVersion(
                            version_id=row[0],
                            version=row[2],
                            adapter_type=AdapterType(metadata.adapter_type.value),
                            base_model_id=row[3],
                            created_at=datetime.fromisoformat(row[8]),
                            created_by=row[9],
                            description=row[4],
                            tags=json.loads(row[6] or '[]'),
                            file_size=row[7],
                            checksum=row[8]
                        )
                        versions.append(version)
                    
                    metadata.versions = versions
                    return metadata
                    
            except Exception as e:
                logger.error(f"Failed to get adapter {adapter_id}: {e}")
                return None
    
    def list_adapters(self, adapter_type: Optional[AdapterType] = None,
                     medical_domain: Optional[str] = None) -> List[AdapterMetadata]:
        """List all adapters with optional filtering."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    query = "SELECT * FROM adapters WHERE 1=1"
                    params = []
                    
                    if adapter_type:
                        query += " AND adapter_type = ?"
                        params.append(adapter_type.value)
                    
                    if medical_domain:
                        query += " AND medical_domain = ?"
                        params.append(medical_domain)
                    
                    query += " ORDER BY name"
                    
                    cursor = conn.execute(query, params)
                    adapters = []
                    
                    for row in cursor.fetchall():
                        metadata = AdapterMetadata(
                            adapter_id=row[0],
                            name=row[1],
                            description=row[2],
                            adapter_type=AdapterType(row[3]),
                            medical_domain=row[4],
                            clinical_use_case=row[5],
                            safety_score=row[6],
                            compliance_flags=json.loads(row[7] or '[]'),
                            avg_load_time=row[8],
                            memory_footprint=row[9],
                            inference_latency_ms=row[10],
                            throughput_tokens_per_sec=row[11],
                            total_loads=row[12],
                            successful_loads=row[13],
                            failed_loads=row[14],
                            last_used=datetime.fromisoformat(row[15]) if row[15] else None,
                            compatible_models=json.loads(row[16] or '[]'),
                            required_resources=json.loads(row[17] or '{}'),
                            validation_status=row[18],
                            test_coverage=row[19],
                            security_scan=row[20]
                        )
                        adapters.append(metadata)
                    
                    return adapters
                    
            except Exception as e:
                logger.error(f"Failed to list adapters: {e}")
                return []
    
    def update_usage_stats(self, adapter_id: str, operation_type: str,
                          duration_ms: Optional[float] = None,
                          success: bool = True,
                          metadata: Optional[Dict[str, Any]] = None):
        """Update usage statistics."""
        with self._lock:
            try:
                now = datetime.now(timezone.utc).isoformat()
                
                with sqlite3.connect(self.registry_path) as conn:
                    # Log usage event
                    conn.execute("""
                        INSERT INTO usage_statistics (
                            adapter_id, operation_type, timestamp, duration_ms,
                            success, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        adapter_id, operation_type, now, duration_ms,
                        success, json.dumps(metadata or {})
                    ))
                    
                    # Update aggregate stats
                    if operation_type == "load":
                        conn.execute("""
                            UPDATE adapters SET
                                total_loads = total_loads + 1,
                                successful_loads = CASE WHEN ? THEN successful_loads + 1 ELSE successful_loads END,
                                failed_loads = CASE WHEN NOT ? THEN failed_loads + 1 ELSE failed_loads END,
                                last_used = ?
                            WHERE adapter_id = ?
                        """, (success, success, now, adapter_id))
                    
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Failed to update usage stats for {adapter_id}: {e}")
    
    def _update_adapter_stats(self, adapter_id: str):
        """Update adapter performance statistics."""
        try:
            with sqlite3.connect(self.registry_path) as conn:
                # Calculate average load time from usage statistics
                cursor = conn.execute("""
                    SELECT AVG(duration_ms) FROM usage_statistics
                    WHERE adapter_id = ? AND operation_type = 'load' AND success = 1
                """, (adapter_id,))
                avg_load_time = cursor.fetchone()[0] or 0.0
                
                # Update adapter stats
                conn.execute("""
                    UPDATE adapters SET
                        avg_load_time = ?
                    WHERE adapter_id = ?
                """, (avg_load_time, adapter_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update adapter stats for {adapter_id}: {e}")
    
    def get_usage_statistics(self, adapter_id: str, 
                           hours: int = 24) -> Dict[str, Any]:
        """Get usage statistics for an adapter."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    # Get recent usage data
                    cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
                    cutoff_str = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
                    
                    cursor = conn.execute("""
                        SELECT operation_type, COUNT(*), AVG(duration_ms), success
                        FROM usage_statistics
                        WHERE adapter_id = ? AND timestamp > ?
                        GROUP BY operation_type, success
                    """, (adapter_id, cutoff_str))
                    
                    stats = {
                        "adapter_id": adapter_id,
                        "period_hours": hours,
                        "operations": {},
                        "total_operations": 0,
                        "success_rate": 0.0
                    }
                    
                    total_ops = 0
                    successful_ops = 0
                    
                    for op_type, count, avg_duration, success in cursor.fetchall():
                        key = f"{op_type}_{'success' if success else 'failed'}"
                        stats["operations"][key] = {
                            "count": count,
                            "avg_duration_ms": avg_duration or 0.0
                        }
                        total_ops += count
                        if success:
                            successful_ops += count
                    
                    if total_ops > 0:
                        stats["success_rate"] = successful_ops / total_ops
                    
                    stats["total_operations"] = total_ops
                    
                    return stats
                    
            except Exception as e:
                logger.error(f"Failed to get usage stats for {adapter_id}: {e}")
                return {}
    
    def validate_medical_compliance(self, adapter_id: str) -> Dict[str, Any]:
        """Validate medical AI compliance for adapter."""
        metadata = self.get_adapter(adapter_id)
        if not metadata:
            return {"valid": False, "errors": ["Adapter not found"]}
        
        compliance_results = {
            "valid": True,
            "checks": {},
            "errors": [],
            "warnings": []
        }
        
        # Basic validation checks
        checks = [
            ("safety_score_threshold", metadata.safety_score >= 0.7, 
             f"Safety score {metadata.safety_score:.2f} below threshold 0.7"),
            ("domain_specified", metadata.medical_domain is not None,
             "Medical domain not specified"),
            ("validation_status", metadata.validation_status == "validated",
             "Adapter not properly validated"),
            ("test_coverage", metadata.test_coverage >= 0.8,
             f"Test coverage {metadata.test_coverage:.1%} below 80%")
        ]
        
        for check_name, passed, message in checks:
            compliance_results["checks"][check_name] = passed
            if not passed:
                compliance_results["valid"] = False
                compliance_results["errors"].append(message)
        
        # Medical-specific compliance checks
        if metadata.adapter_type in [AdapterType.MEDICAL_LORA, AdapterType.CLINICAL_LORA]:
            clinical_checks = [
                ("clinical_use_case", metadata.clinical_use_case is not None,
                 "Clinical use case not specified"),
                ("compliance_flags", len(metadata.compliance_flags) >= 1,
                 "Missing compliance documentation")
            ]
            
            for check_name, passed, message in clinical_checks:
                compliance_results["checks"][check_name] = passed
                if not passed:
                    compliance_results["valid"] = False
                    compliance_results["errors"].append(message)
        
        # HIPAA/PHI protection warnings
        if metadata.compliance_flags and "phi_protection" not in metadata.compliance_flags:
            compliance_results["warnings"].append("Missing PHI protection compliance flag")
        
        return compliance_results
    
    def cleanup_old_versions(self, adapter_id: str, keep_versions: int = 3) -> int:
        """Clean up old adapter versions, keeping most recent."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    # Get versions ordered by creation date (newest first)
                    cursor = conn.execute("""
                        SELECT version_id FROM adapter_versions
                        WHERE adapter_id = ?
                        ORDER BY created_at DESC
                    """, (adapter_id,))
                    
                    versions = [row[0] for row in cursor.fetchall()]
                    
                    # Keep most recent versions
                    versions_to_delete = versions[keep_versions:]
                    
                    for version_id in versions_to_delete:
                        conn.execute("""
                            DELETE FROM adapter_versions WHERE version_id = ?
                        """, (version_id,))
                    
                    conn.commit()
                    
                    deleted_count = len(versions_to_delete)
                    if deleted_count > 0:
                        logger.info(f"Cleaned up {deleted_count} old versions for adapter {adapter_id}")
                    
                    return deleted_count
                    
            except Exception as e:
                logger.error(f"Failed to cleanup old versions for {adapter_id}: {e}")
                return 0
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        with self._lock:
            try:
                with sqlite3.connect(self.registry_path) as conn:
                    stats = {}
                    
                    # Basic counts
                    cursor = conn.execute("SELECT COUNT(*) FROM adapters")
                    stats["total_adapters"] = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(*) FROM adapter_versions")
                    stats["total_versions"] = cursor.fetchone()[0]
                    
                    # By type
                    cursor = conn.execute("""
                        SELECT adapter_type, COUNT(*) 
                        FROM adapters GROUP BY adapter_type
                    """)
                    stats["by_type"] = dict(cursor.fetchall())
                    
                    # By status
                    cursor = conn.execute("""
                        SELECT status, COUNT(*) 
                        FROM adapter_versions GROUP BY status
                    """)
                    stats["by_status"] = dict(cursor.fetchall())
                    
                    # Medical compliance
                    cursor = conn.execute("""
                        SELECT validation_status, COUNT(*) 
                        FROM adapters GROUP BY validation_status
                    """)
                    stats["validation_status"] = dict(cursor.fetchall())
                    
                    # Recent activity
                    cursor = conn.execute("""
                        SELECT COUNT(*) FROM usage_statistics 
                        WHERE timestamp > datetime('now', '-24 hours')
                    """)
                    stats["operations_last_24h"] = cursor.fetchone()[0]
                    
                    return stats
                    
            except Exception as e:
                logger.error(f"Failed to get registry stats: {e}")
                return {}


# Utility functions for the registry
def create_adapter_metadata(adapter_id: str, name: str, description: str,
                           adapter_type: AdapterType,
                           medical_domain: Optional[str] = None,
                           clinical_use_case: Optional[str] = None) -> AdapterMetadata:
    """Factory function to create adapter metadata."""
    return AdapterMetadata(
        adapter_id=adapter_id,
        name=name,
        description=description,
        adapter_type=adapter_type,
        medical_domain=medical_domain,
        clinical_use_case=clinical_use_case
    )


def create_adapter_version(adapter_id: str, version: str,
                          base_model_id: str, description: str,
                          created_by: str,
                          tags: Optional[List[str]] = None) -> AdapterVersion:
    """Factory function to create adapter version."""
    return AdapterVersion(
        version_id=str(uuid4()),
        version=version,
        adapter_type=AdapterType.LORA,  # Will be updated when linked to adapter
        base_model_id=base_model_id,
        created_at=datetime.now(timezone.utc),
        created_by=created_by,
        description=description,
        tags=tags or []
    )


if __name__ == "__main__":
    # Example usage
    registry = AdapterRegistry("./test_registry.db")
    
    # Create and register an adapter
    metadata = create_adapter_metadata(
        adapter_id="medical_diagnosis_v1",
        name="Medical Diagnosis Assistant",
        description="LoRA adapter for medical diagnosis assistance",
        adapter_type=AdapterType.MEDICAL_LORA,
        medical_domain="diagnostic_medicine",
        clinical_use_case="differential_diagnosis"
    )
    
    registry.register_adapter(metadata)
    
    # Add a version
    version = create_adapter_version(
        adapter_id="medical_diagnosis_v1",
        version="1.0.0",
        base_model_id="microsoft/DialoGPT-medium",
        description="Initial version with basic diagnosis capabilities",
        created_by="medical_ai_team"
    )
    
    registry.add_version("medical_diagnosis_v1", version)
    
    # Get and display adapter
    retrieved = registry.get_adapter("medical_diagnosis_v1")
    if retrieved:
        print(f"Adapter: {retrieved.name}")
        print(f"Type: {retrieved.adapter_type.value}")
        print(f"Versions: {retrieved.get_version_count()}")
        print(f"Success Rate: {retrieved.get_success_rate():.1%}")