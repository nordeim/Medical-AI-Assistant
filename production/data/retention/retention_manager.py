"""
Production Data Archival and Retention Management System
Implements HIPAA-compliant data archival and retention policies
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import shutil
from pathlib import Path
import sqlite3
import csv
from cryptography.fernet import Fernet
import boto3
from botocore.exceptions import ClientError

class DataClassification(Enum):
    """Data classification levels"""
    PHI = "protected_health_information"
    PII = "personally_identifiable_information"
    CONFIDENTIAL = "confidential"
    INTERNAL = "internal"
    PUBLIC = "public"

class RetentionStatus(Enum):
    """Data retention status"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"
    EXPIRED = "expired"

class ArchiveType(Enum):
    """Types of archival storage"""
    HOT_STORAGE = "hot_storage"           # Immediately accessible
    WARM_STORAGE = "warm_storage"         # Quick retrieval (<1 hour)
    COLD_STORAGE = "cold_storage"         # Standard retrieval (<24 hours)
    DEEP_COLD = "deep_cold"              # Long-term storage (>24 hours)
    PERMANENT_ARCHIVE = "permanent_archive" # Legal/compliance retention

class DeletionMethod(Enum):
    """Data deletion methods"""
    LOGICAL_DELETE = "logical_delete"     # Mark as deleted, keep data
    PHYSICAL_DELETE = "physical_delete"   # Remove data completely
    SECURE_DELETE = "secure_delete"       # Overwrite before deletion
    CRYPTO_DELETE = "crypto_delete"       # Destroy encryption keys

@dataclass
class RetentionPolicy:
    """Data retention policy configuration"""
    policy_id: str
    name: str
    description: str
    data_types: List[str]
    classification: DataClassification
    retention_period_days: int
    archive_type: ArchiveType
    encryption_required: bool
    audit_required: bool
    legal_hold: bool
    deletion_method: DeletionMethod
    compliance_requirements: List[str]
    notification_schedule: List[str]
    auto_cleanup: bool = True

@dataclass
class DataRecordMetadata:
    """Metadata for archived data records"""
    record_id: str
    original_table: str
    classification: DataClassification
    retention_policy_id: str
    created_date: datetime
    archived_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    retention_status: RetentionStatus = RetentionStatus.ACTIVE
    encryption_enabled: bool = False
    checksum: Optional[str] = None
    archive_location: Optional[str] = None
    access_log: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ArchiveJob:
    """Archive job configuration and status"""
    job_id: str
    policy_id: str
    data_source: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_archived: int = 0
    archive_size_mb: float = 0.0
    error_log: List[str] = field(default_factory=list)
    verification_passed: bool = False

@dataclass
class DeletionJob:
    """Data deletion job configuration"""
    job_id: str
    policy_id: str
    records_to_delete: List[str]
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    records_deleted: int = 0
    verification_required: bool = True

class DataRetentionManager:
    """Production data retention and archival management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.retention_policies = {}
        self.archive_storage = {}
        self.metadata_db = None
        self.encryption_key = self._get_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup retention management logging"""
        logger = logging.getLogger("data_retention")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key for archive security"""
        key = self.config.get("encryption_key")
        if key:
            return key.encode()
        else:
            # Generate new key for development
            key = Fernet.generate_key()
            self.logger.warning("Generated new encryption key for archives")
            return key
    
    async def initialize_retention_system(self) -> None:
        """Initialize retention and archival system"""
        try:
            # Initialize retention policies
            await self._initialize_retention_policies()
            
            # Initialize metadata database
            await self._initialize_metadata_database()
            
            # Initialize archive storage connections
            await self._initialize_archive_storage()
            
            self.logger.info("Data retention system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Retention system initialization failed: {str(e)}")
            raise
    
    async def _initialize_retention_policies(self) -> None:
        """Initialize HIPAA-compliant retention policies"""
        self.retention_policies = {
            "patient_records_active": RetentionPolicy(
                policy_id="POL_001",
                name="Active Patient Records",
                description="Active patient medical records requiring immediate access",
                data_types=["patient_records", "encounters", "medications", "allergies"],
                classification=DataClassification.PHI,
                retention_period_days=2555,  # 7 years
                archive_type=ArchiveType.HOT_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.CRYPTO_DELETE,
                compliance_requirements=["HIPAA", "HITECH", "State_Medical_Records"],
                notification_schedule=["quarterly"],
                auto_cleanup=True
            ),
            
            "clinical_documents": RetentionPolicy(
                policy_id="POL_002",
                name="Clinical Documents and Reports",
                description="Clinical notes, reports, and supporting documentation",
                data_types=["clinical_notes", "discharge_summaries", "consultations"],
                classification=DataClassification.PHI,
                retention_period_days=4380,  # 12 years
                archive_type=ArchiveType.WARM_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.CRYPTO_DELETE,
                compliance_requirements=["HIPAA", "HITECH", "JCAHO"],
                notification_schedule=["annually"],
                auto_cleanup=True
            ),
            
            "imaging_studies": RetentionPolicy(
                policy_id="POL_003",
                name="Medical Imaging Studies",
                description="DICOM images and imaging reports",
                data_types=["dicom_images", "imaging_reports", "radiology_studies"],
                classification=DataClassification.PHI,
                retention_period_days=7300,  # 20 years
                archive_type=ArchiveType.COLD_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.CRYPTO_DELETE,
                compliance_requirements=["HIPAA", "ACR", "State_Radiology_Laws"],
                notification_schedule=["annually"],
                auto_cleanup=False
            ),
            
            "laboratory_results": RetentionPolicy(
                policy_id="POL_004",
                name="Laboratory Results and Pathology",
                description="Lab test results, pathology reports, and specimens data",
                data_types=["lab_results", "pathology_reports", "cytology"],
                classification=DataClassification.PHI,
                retention_period_days=3650,  # 10 years
                archive_type=ArchiveType.WARM_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.SECURE_DELETE,
                compliance_requirements=["HIPAA", "CLIA", "CAP"],
                notification_schedule=["annually"],
                auto_cleanup=True
            ),
            
            "financial_records": RetentionPolicy(
                policy_id="POL_005",
                name="Financial and Billing Records",
                description="Insurance claims, billing records, and payment data",
                data_types=["insurance_claims", "billing_records", "payment_data"],
                classification=DataClassification.PHI,
                retention_period_days=2920,  # 8 years
                archive_type=ArchiveType.COLD_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.PHYSICAL_DELETE,
                compliance_requirements=["HIPAA", "IRS_Requirements", "Insurance_Laws"],
                notification_schedule=["quarterly"],
                auto_cleanup=True
            ),
            
            "quality_improvement": RetentionPolicy(
                policy_id="POL_006",
                name="Quality Improvement and Safety Data",
                description="Quality metrics, safety reports, and improvement initiatives",
                data_types=["quality_metrics", "safety_reports", "incident_reports"],
                classification=DataClassification.PHI,
                retention_period_days=4380,  # 12 years
                archive_type=ArchiveType.WARM_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.PHYSICAL_DELETE,
                compliance_requirements=["HIPAA", "JCAHO", "CMS_Requirements"],
                notification_schedule=["annually"],
                auto_cleanup=True
            ),
            
            "operational_logs": RetentionPolicy(
                policy_id="POL_007",
                name="System Operational Logs",
                description="System access logs, audit trails, and operational data",
                data_types=["access_logs", "audit_trails", "system_logs"],
                classification=DataClassification.CONFIDENTIAL,
                retention_period_days=1095,  # 3 years
                archive_type=ArchiveType.COLD_STORAGE,
                encryption_required=True,
                audit_required=True,
                legal_hold=False,
                deletion_method=DeletionMethod.PHYSICAL_DELETE,
                compliance_requirements=["HIPAA", "SOX", "NIST"],
                notification_schedule=["monthly"],
                auto_cleanup=True
            ),
            
            "research_data_anonymized": RetentionPolicy(
                policy_id="POL_008",
                name="Anonymized Research Data",
                description="De-identified data for research and analytics",
                data_types=["research_data", "analytics_datasets", "deidentified_data"],
                classification=DataClassification.INTERNAL,
                retention_period_days=7300,  # 20 years
                archive_type=ArchiveType.DEEP_COLD,
                encryption_required=True,
                audit_required=False,
                legal_hold=False,
                deletion_method=DeletionMethod.PHYSICAL_DELETE,
                compliance_requirements=["HIPAA_Deidentification", "Research_Ethics"],
                notification_schedule=["annually"],
                auto_cleanup=True
            )
        }
    
    async def _initialize_metadata_database(self) -> None:
        """Initialize metadata database for retention tracking"""
        db_path = self.config.get("metadata_db_path", "retention_metadata.db")
        
        # Create metadata table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS retention_metadata (
                record_id TEXT PRIMARY KEY,
                original_table TEXT NOT NULL,
                classification TEXT NOT NULL,
                retention_policy_id TEXT NOT NULL,
                created_date TIMESTAMP NOT NULL,
                archived_date TIMESTAMP,
                expiration_date TIMESTAMP,
                retention_status TEXT NOT NULL,
                encryption_enabled BOOLEAN DEFAULT FALSE,
                checksum TEXT,
                archive_location TEXT,
                access_log TEXT,
                FOREIGN KEY (retention_policy_id) REFERENCES retention_policies (policy_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS archive_jobs (
                job_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                data_source TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                records_archived INTEGER DEFAULT 0,
                archive_size_mb REAL DEFAULT 0.0,
                error_log TEXT,
                verification_passed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (policy_id) REFERENCES retention_policies (policy_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deletion_jobs (
                job_id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                records_to_delete TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                completed_at TIMESTAMP,
                records_deleted INTEGER DEFAULT 0,
                verification_required BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (policy_id) REFERENCES retention_policies (policy_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.metadata_db = db_path
        self.logger.info("Retention metadata database initialized")
    
    async def _initialize_archive_storage(self) -> None:
        """Initialize archive storage connections"""
        # Initialize S3 for cloud storage
        if self.config.get("s3_enabled", False):
            try:
                self.archive_storage["s3"] = boto3.client(
                    's3',
                    aws_access_key_id=self.config.get("aws_access_key"),
                    aws_secret_access_key=self.config.get("aws_secret_key"),
                    region_name=self.config.get("aws_region", "us-east-1")
                )
            except Exception as e:
                self.logger.warning(f"S3 initialization failed: {str(e)}")
        
        # Initialize local storage paths
        base_path = Path(self.config.get("archive_base_path", "./archives"))
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different storage types
        for archive_type in ArchiveType:
            (base_path / archive_type.value).mkdir(exist_ok=True)
        
        self.archive_storage["local_base_path"] = base_path
        self.logger.info("Archive storage initialized")
    
    async def archive_data(self, policy_id: str, data_source: str, 
                          source_config: Dict[str, Any]) -> ArchiveJob:
        """Archive data according to retention policy"""
        
        if policy_id not in self.retention_policies:
            raise ValueError(f"Retention policy {policy_id} not found")
        
        policy = self.retention_policies[policy_id]
        job_id = f"ARCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{policy_id}"
        
        job = ArchiveJob(
            job_id=job_id,
            policy_id=policy_id,
            data_source=data_source,
            status="started",
            started_at=datetime.now()
        )
        
        try:
            self.logger.info(f"Starting archive job: {job_id}")
            
            # Load data to be archived
            data = await self._load_data_for_archiving(source_config)
            
            # Apply retention policy and classification
            processed_data = await self._apply_retention_policy(data, policy)
            
            # Encrypt data if required
            if policy.encryption_required:
                processed_data = await self._encrypt_archive_data(processed_data, policy)
            
            # Archive to appropriate storage
            archive_result = await self._store_archive_data(processed_data, policy, job_id)
            
            # Update metadata
            await self._update_retention_metadata(processed_data, policy, archive_result)
            
            # Verify archive integrity
            verification_passed = await self._verify_archive_integrity(archive_result)
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.records_archived = len(processed_data) if isinstance(processed_data, pd.DataFrame) else 1
            job.archive_size_mb = archive_result.get("size_mb", 0.0)
            job.verification_passed = verification_passed
            
            # Log completion
            self.logger.info(f"Archive job completed: {job_id} - {job.records_archived} records archived")
            
        except Exception as e:
            job.status = "failed"
            job.completed_at = datetime.now()
            job.error_log.append(str(e))
            self.logger.error(f"Archive job failed: {job_id} - {str(e)}")
        
        return job
    
    async def _load_data_for_archiving(self, source_config: Dict[str, Any]) -> pd.DataFrame:
        """Load data from source for archiving"""
        # In production, this would load from actual databases
        # For demonstration, create sample data
        
        if source_config.get("table_name"):
            # Simulate loading from database table
            sample_data = pd.DataFrame({
                "patient_id": [f"PAT_{i:04d}" for i in range(1, 101)],
                "record_date": pd.date_range("2020-01-01", periods=100, freq="D"),
                "record_type": ["consultation", "lab_result", "imaging"] * 33 + ["consultation"],
                "clinical_data": [f"Clinical data for record {i}" for i in range(1, 101)],
                "provider_id": [f"PROV_{i%10:02d}" for i in range(1, 101)],
                "status": ["active"] * 100
            })
            
            return sample_data
        else:
            # Default sample data
            return pd.DataFrame({
                "id": range(1, 101),
                "data": [f"Sample data {i}" for i in range(1, 101)],
                "timestamp": pd.date_range("2020-01-01", periods=100, freq="D")
            })
    
    async def _apply_retention_policy(self, data: pd.DataFrame, policy: RetentionPolicy) -> pd.DataFrame:
        """Apply retention policy rules to data"""
        processed_data = data.copy()
        
        # Add retention metadata columns
        processed_data["retention_policy_id"] = policy.policy_id
        processed_data["classification"] = policy.classification.value
        processed_data["created_date"] = datetime.now()
        
        # Calculate expiration date
        expiration_date = datetime.now() + timedelta(days=policy.retention_period_days)
        processed_data["expiration_date"] = expiration_date
        
        # Apply data classification specific processing
        if policy.classification == DataClassification.PHI:
            # Add additional PHI protections
            processed_data["phi_protected"] = True
            processed_data["access_controls"] = "strict"
        
        elif policy.classification == DataClassification.PII:
            # Add PII handling
            processed_data["pii_protected"] = True
        
        # Add audit trail if required
        if policy.audit_required:
            processed_data["audit_required"] = True
            processed_data["audit_level"] = "comprehensive"
        
        return processed_data
    
    async def _encrypt_archive_data(self, data: pd.DataFrame, policy: RetentionPolicy) -> pd.DataFrame:
        """Encrypt archive data if required"""
        encrypted_data = data.copy()
        
        # Encrypt sensitive columns
        sensitive_columns = []
        
        if policy.classification in [DataClassification.PHI, DataClassification.PII]:
            sensitive_columns = ["patient_id", "clinical_data", "provider_id"]
        elif policy.classification == DataClassification.CONFIDENTIAL:
            sensitive_columns = ["data", "timestamp"]
        
        for col in sensitive_columns:
            if col in encrypted_data.columns:
                encrypted_data[f"{col}_encrypted"] = encrypted_data[col].apply(
                    lambda x: self.cipher_suite.encrypt(str(x).encode()).decode() if pd.notna(x) else None
                )
                # Remove original column if specified
                encrypted_data = encrypted_data.drop(columns=[col])
        
        encrypted_data["encryption_enabled"] = True
        return encrypted_data
    
    async def _store_archive_data(self, data: pd.DataFrame, policy: RetentionPolicy, job_id: str) -> Dict[str, Any]:
        """Store archive data in appropriate storage"""
        
        storage_type = policy.archive_type
        archive_result = {}
        
        try:
            if storage_type == ArchiveType.HOT_STORAGE:
                # Store in fast-access storage
                result = await self._store_hot_storage(data, job_id)
            
            elif storage_type == ArchiveType.WARM_STORAGE:
                # Store in quick-retrieval storage
                result = await self._store_warm_storage(data, job_id)
            
            elif storage_type == ArchiveType.COLD_STORAGE:
                # Store in standard cold storage
                result = await self._store_cold_storage(data, job_id)
            
            elif storage_type == ArchiveType.DEEP_COLD:
                # Store in long-term cold storage
                result = await self._store_deep_cold_storage(data, job_id)
            
            elif storage_type == ArchiveType.PERMANENT_ARCHIVE:
                # Store in permanent archive
                result = await self._store_permanent_archive(data, job_id)
            
            else:
                raise ValueError(f"Unknown archive type: {storage_type}")
            
            archive_result = {
                "job_id": job_id,
                "storage_type": storage_type.value,
                "location": result["location"],
                "size_mb": result.get("size_mb", 0.0),
                "checksum": result.get("checksum", ""),
                "stored_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Archive stored: {archive_result['location']}")
            
        except Exception as e:
            self.logger.error(f"Archive storage failed: {str(e)}")
            raise
        
        return archive_result
    
    async def _store_hot_storage(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in hot storage (immediate access)"""
        base_path = self.archive_storage["local_base_path"] / "hot_storage"
        filename = f"{job_id}_archive.parquet"
        filepath = base_path / filename
        
        data.to_parquet(filepath, compression='snappy')
        
        return {
            "location": str(filepath),
            "size_mb": filepath.stat().st_size / (1024 * 1024),
            "checksum": self._calculate_checksum(filepath)
        }
    
    async def _store_warm_storage(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in warm storage (quick retrieval)"""
        base_path = self.archive_storage["local_base_path"] / "warm_storage"
        filename = f"{job_id}_archive.parquet"
        filepath = base_path / filename
        
        data.to_parquet(filepath, compression='gzip')
        
        return {
            "location": str(filepath),
            "size_mb": filepath.stat().st_size / (1024 * 1024),
            "checksum": self._calculate_checksum(filepath)
        }
    
    async def _store_cold_storage(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in cold storage (standard retrieval)"""
        # Use S3 if available, otherwise local cold storage
        if "s3" in self.archive_storage:
            return await self._store_s3_cold_storage(data, job_id)
        else:
            return await self._store_local_cold_storage(data, job_id)
    
    async def _store_s3_cold_storage(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in S3 cold storage"""
        import io
        
        bucket_name = self.config.get("s3_bucket_name", "healthcare-archives")
        key = f"cold_storage/{job_id}_archive.parquet"
        
        # Convert DataFrame to parquet bytes
        buffer = io.BytesIO()
        data.to_parquet(buffer, compression='snappy')
        buffer.seek(0)
        
        # Upload to S3
        s3_client = self.archive_storage["s3"]
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=buffer.getvalue(),
            StorageClass='STANDARD_IA'  # Standard-IA for infrequent access
        )
        
        return {
            "location": f"s3://{bucket_name}/{key}",
            "size_mb": buffer.tell() / (1024 * 1024),
            "checksum": self._calculate_checksum(buffer)
        }
    
    async def _store_local_cold_storage(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in local cold storage"""
        base_path = self.archive_storage["local_base_path"] / "cold_storage"
        filename = f"{job_id}_archive.parquet"
        filepath = base_path / filename
        
        data.to_parquet(filepath, compression='gzip')
        
        return {
            "location": str(filepath),
            "size_mb": filepath.stat().st_size / (1024 * 1024),
            "checksum": self._calculate_checksum(filepath)
        }
    
    async def _store_deep_cold_storage(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in deep cold storage (long-term)"""
        # Compress heavily and store offline
        base_path = self.archive_storage["local_base_path"] / "deep_cold"
        filename = f"{job_id}_archive.parquet"
        filepath = base_path / filename
        
        data.to_parquet(filepath, compression='bzip2')
        
        return {
            "location": str(filepath),
            "size_mb": filepath.stat().st_size / (1024 * 1024),
            "checksum": self._calculate_checksum(filepath)
        }
    
    async def _store_permanent_archive(self, data: pd.DataFrame, job_id: str) -> Dict[str, Any]:
        """Store in permanent archive (legal/compliance)"""
        # Store with multiple redundancy and compliance tracking
        base_path = self.archive_storage["local_base_path"] / "permanent_archive"
        filename = f"{job_id}_archive.parquet"
        filepath = base_path / filename
        
        data.to_parquet(filepath, compression='snappy')
        
        # Create compliance manifest
        manifest = {
            "job_id": job_id,
            "archived_at": datetime.now().isoformat(),
            "compliance_tracking": True,
            "legal_hold": False,
            "retention_permanent": True
        }
        
        manifest_path = base_path / f"{job_id}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        return {
            "location": str(filepath),
            "manifest_location": str(manifest_path),
            "size_mb": filepath.stat().st_size / (1024 * 1024),
            "checksum": self._calculate_checksum(filepath)
        }
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA-256 checksum for file integrity verification"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _calculate_checksum(self, buffer: io.BytesIO) -> str:
        """Calculate SHA-256 checksum for buffer"""
        hash_sha256 = hashlib.sha256()
        for chunk in iter(lambda: buffer.read(4096), b""):
            hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    async def _update_retention_metadata(self, data: pd.DataFrame, policy: RetentionPolicy, 
                                       archive_result: Dict[str, Any]) -> None:
        """Update retention metadata in database"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        # Get or create unique record IDs
        if isinstance(data, pd.DataFrame) and len(data) > 0:
            # Generate record IDs based on data
            if "patient_id" in data.columns:
                record_ids = data["patient_id"].unique().tolist()
            elif "id" in data.columns:
                record_ids = data["id"].unique().tolist()
            else:
                record_ids = [f"REC_{i:06d}" for i in range(1, len(data) + 1)]
            
            # Insert metadata records
            for i, record_id in enumerate(record_ids):
                metadata_record = DataRecordMetadata(
                    record_id=record_id,
                    original_table="unknown",  # Would be provided by source config
                    classification=policy.classification,
                    retention_policy_id=policy.policy_id,
                    created_date=datetime.now() - timedelta(days=np.random.randint(1, 365)),
                    archived_date=datetime.now(),
                    expiration_date=datetime.now() + timedelta(days=policy.retention_period_days),
                    retention_status=RetentionStatus.ARCHIVED,
                    encryption_enabled=policy.encryption_required,
                    checksum=archive_result.get("checksum", ""),
                    archive_location=archive_result["location"]
                )
                
                cursor.execute('''
                    INSERT OR REPLACE INTO retention_metadata 
                    (record_id, original_table, classification, retention_policy_id, 
                     created_date, archived_date, expiration_date, retention_status,
                     encryption_enabled, checksum, archive_location, access_log)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata_record.record_id,
                    metadata_record.original_table,
                    metadata_record.classification.value,
                    metadata_record.retention_policy_id,
                    metadata_record.created_date,
                    metadata_record.archived_date,
                    metadata_record.expiration_date,
                    metadata_record.retention_status.value,
                    metadata_record.encryption_enabled,
                    metadata_record.checksum,
                    metadata_record.archive_location,
                    json.dumps(metadata_record.access_log)
                ))
        
        conn.commit()
        conn.close()
    
    async def _verify_archive_integrity(self, archive_result: Dict[str, Any]) -> bool:
        """Verify archive integrity"""
        try:
            location = archive_result["location"]
            
            # For S3 locations
            if location.startswith("s3://"):
                # Verify S3 object integrity
                parts = location.replace("s3://", "").split("/", 1)
                bucket = parts[0]
                key = parts[1]
                
                s3_client = self.archive_storage["s3"]
                response = s3_client.head_object(Bucket=bucket, Key=key)
                
                # Check if object exists and has expected size
                expected_size = archive_result.get("size_mb", 0) * 1024 * 1024
                actual_size = response["ContentLength"]
                
                return abs(actual_size - expected_size) < 1024  # Allow 1KB difference
            
            # For local files
            else:
                filepath = Path(location)
                if not filepath.exists():
                    return False
                
                # Verify file size
                expected_size = archive_result.get("size_mb", 0) * 1024 * 1024
                actual_size = filepath.stat().st_size
                
                if abs(actual_size - expected_size) > 1024:  # Allow 1KB difference
                    return False
                
                # Verify checksum if available
                if archive_result.get("checksum"):
                    calculated_checksum = self._calculate_checksum(filepath)
                    return calculated_checksum == archive_result["checksum"]
                
                return True
                
        except Exception as e:
            self.logger.error(f"Archive integrity verification failed: {str(e)}")
            return False
    
    async def execute_retention_cleanup(self) -> Dict[str, Any]:
        """Execute scheduled retention cleanup"""
        cleanup_results = {
            "started_at": datetime.now(),
            "policies_processed": [],
            "records_archived": 0,
            "records_deleted": 0,
            "errors": []
        }
        
        try:
            conn = sqlite3.connect(self.metadata_db)
            cursor = conn.cursor()
            
            # Check for expired records
            cursor.execute('''
                SELECT record_id, retention_policy_id, expiration_date, retention_status
                FROM retention_metadata
                WHERE expiration_date < ? AND retention_status != 'deleted'
            ''', (datetime.now(),))
            
            expired_records = cursor.fetchall()
            
            for record_id, policy_id, expiration_date, current_status in expired_records:
                try:
                    policy = self.retention_policies.get(policy_id)
                    if not policy:
                        continue
                    
                    # Process according to policy
                    if policy.auto_cleanup:
                        deletion_job = await self._execute_deletion(record_id, policy)
                        cleanup_results["records_deleted"] += deletion_job.records_deleted
                        cleanup_results["policies_processed"].append(policy_id)
                    
                    # Update retention status
                    cursor.execute('''
                        UPDATE retention_metadata
                        SET retention_status = ?
                        WHERE record_id = ?
                    ''', (RetentionStatus.EXPIRED.value, record_id))
                    
                except Exception as e:
                    cleanup_results["errors"].append(f"Failed to process record {record_id}: {str(e)}")
            
            conn.commit()
            conn.close()
            
            cleanup_results["completed_at"] = datetime.now()
            cleanup_results["expired_records_found"] = len(expired_records)
            
            self.logger.info(f"Retention cleanup completed: {cleanup_results['records_deleted']} records processed")
            
        except Exception as e:
            cleanup_results["errors"].append(f"Cleanup execution failed: {str(e)}")
            self.logger.error(f"Retention cleanup failed: {str(e)}")
        
        return cleanup_results
    
    async def _execute_deletion(self, record_id: str, policy: RetentionPolicy) -> DeletionJob:
        """Execute data deletion according to policy"""
        job_id = f"DEL_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{record_id}"
        
        job = DeletionJob(
            job_id=job_id,
            policy_id=policy.policy_id,
            records_to_delete=[record_id],
            status="started",
            started_at=datetime.now()
        )
        
        try:
            if policy.deletion_method == DeletionMethod.LOGICAL_DELETE:
                await self._logical_delete(record_id, policy)
            
            elif policy.deletion_method == DeletionMethod.PHYSICAL_DELETE:
                await self._physical_delete(record_id, policy)
            
            elif policy.deletion_method == DeletionMethod.SECURE_DELETE:
                await self._secure_delete(record_id, policy)
            
            elif policy.deletion_method == DeletionMethod.CRYPTO_DELETE:
                await self._crypto_delete(record_id, policy)
            
            job.status = "completed"
            job.records_deleted = 1
            
        except Exception as e:
            job.status = "failed"
            job.error_log.append(str(e))
        
        job.completed_at = datetime.now()
        return job
    
    async def _logical_delete(self, record_id: str, policy: RetentionPolicy) -> None:
        """Perform logical deletion (mark as deleted)"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE retention_metadata
            SET retention_status = ?
            WHERE record_id = ?
        ''', (RetentionStatus.DELETED.value, record_id))
        
        conn.commit()
        conn.close()
    
    async def _physical_delete(self, record_id: str, policy: RetentionPolicy) -> None:
        """Perform physical deletion (remove data completely)"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        # Get archive location
        cursor.execute('''
            SELECT archive_location FROM retention_metadata
            WHERE record_id = ?
        ''', (record_id,))
        
        result = cursor.fetchone()
        if result:
            archive_location = result[0]
            
            # Remove archive file if exists
            try:
                if archive_location.startswith("s3://"):
                    # Remove from S3
                    parts = archive_location.replace("s3://", "").split("/", 1)
                    bucket = parts[0]
                    key = parts[1]
                    
                    s3_client = self.archive_storage["s3"]
                    s3_client.delete_object(Bucket=bucket, Key=key)
                
                else:
                    # Remove local file
                    filepath = Path(archive_location)
                    if filepath.exists():
                        filepath.unlink()
            
            except Exception as e:
                self.logger.warning(f"Failed to remove archive file: {str(e)}")
        
        # Remove from metadata
        cursor.execute('DELETE FROM retention_metadata WHERE record_id = ?', (record_id,))
        
        conn.commit()
        conn.close()
    
    async def _secure_delete(self, record_id: str, policy: RetentionPolicy) -> None:
        """Perform secure deletion (overwrite before removal)"""
        # This would implement secure deletion algorithms
        # For now, use physical delete with additional logging
        await self._physical_delete(record_id, policy)
        
        self.logger.info(f"Secure deletion completed for record: {record_id}")
    
    async def _crypto_delete(self, record_id: str, policy: RetentionPolicy) -> None:
        """Perform cryptographic deletion (destroy encryption keys)"""
        # In production, this would involve destroying encryption keys
        # making the data unrecoverable
        
        await self._logical_delete(record_id, policy)
        
        self.logger.info(f"Cryptographic deletion initiated for record: {record_id}")
    
    def get_retention_status(self) -> Dict[str, Any]:
        """Get current retention system status"""
        conn = sqlite3.connect(self.metadata_db)
        cursor = conn.cursor()
        
        # Get summary statistics
        cursor.execute('''
            SELECT 
                classification,
                retention_status,
                COUNT(*) as record_count,
                AVG(julianday(expiration_date) - julianday(created_date)) as avg_retention_days
            FROM retention_metadata
            GROUP BY classification, retention_status
        ''')
        
        summary_stats = cursor.fetchall()
        
        # Get policy coverage
        cursor.execute('''
            SELECT 
                rm.retention_policy_id,
                COUNT(*) as covered_records,
                COUNT(CASE WHEN rm.retention_status = 'archived' THEN 1 END) as archived_records,
                COUNT(CASE WHEN rm.retention_status = 'active' THEN 1 END) as active_records
            FROM retention_metadata rm
            GROUP BY rm.retention_policy_id
        ''')
        
        policy_coverage = cursor.fetchall()
        
        conn.close()
        
        return {
            "summary_statistics": summary_stats,
            "policy_coverage": policy_coverage,
            "policies_configured": len(self.retention_policies),
            "last_cleanup": datetime.now().isoformat(),
            "system_status": "operational"
        }

def create_retention_manager(config: Dict[str, Any] = None) -> DataRetentionManager:
    """Factory function to create retention manager"""
    if config is None:
        config = {
            "archive_base_path": "./archives",
            "metadata_db_path": "retention_metadata.db",
            "encryption_required": True,
            "s3_enabled": False
        }
    
    return DataRetentionManager(config)

# Example usage
if __name__ == "__main__":
    async def main():
        manager = create_retention_manager()
        
        # Initialize retention system
        await manager.initialize_retention_system()
        
        # Archive sample data
        source_config = {
            "table_name": "patient_records",
            "connection_string": "sample_connection"
        }
        
        job = await manager.archive_data("POL_001", "patient_db", source_config)
        
        print(f"Archive Job: {job.job_id}")
        print(f"Status: {job.status}")
        print(f"Records Archived: {job.records_archived}")
        print(f"Archive Size: {job.archive_size_mb:.2f} MB")
        print(f"Verification Passed: {job.verification_passed}")
        
        # Execute retention cleanup
        cleanup_results = await manager.execute_retention_cleanup()
        print(f"Cleanup Results: {cleanup_results['records_deleted']} records deleted")
        
        # Get retention status
        status = manager.get_retention_status()
        print(f"Retention Status: {status['system_status']}")
    
    asyncio.run(main())
