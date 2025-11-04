"""
Production Data Management Configuration
Comprehensive configuration for healthcare data pipeline and analytics
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

class DataSource(Enum):
    """Healthcare data sources"""
    EHR = "electronic_health_records"
    LAB_RESULTS = "laboratory_results"
    IMAGING = "medical_imaging"
    MEDICATION = "medication_records"
    VITAL_SIGNS = "vital_signs"
    CLINICAL_NOTES = "clinical_notes"
    CLAIMS = "insurance_claims"
    GENOMIC = "genomic_data"

class DataQuality(Enum):
    """Data quality levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RetentionPeriod(Enum):
    """HIPAA-compliant retention periods"""
    ACTIVE = "active"
    SHORT_TERM = "short_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-7 years
    LONG_TERM = "long_term"  # 7-25 years
    PERMANENT = "permanent"

@dataclass
class DataSourceConfig:
    """Configuration for healthcare data sources"""
    name: str
    source_type: DataSource
    connection_string: str
    schema: str
    tables: List[str]
    quality_level: DataQuality
    retention_period: RetentionPeriod
    hipaa_classification: str
    encryption_enabled: bool = True
    audit_enabled: bool = True
    batch_size: int = 1000
    refresh_frequency: str = "hourly"
    validation_rules: List[str] = field(default_factory=list)

@dataclass
class ETLConfig:
    """ETL pipeline configuration"""
    pipeline_name: str
    source_config: DataSourceConfig
    transformation_rules: List[Dict[str, Any]]
    data_quality_checks: List[Dict[str, Any]]
    error_handling: str = "reject"
    batch_processing: bool = True
    real_time_processing: bool = False
    parallel_processes: int = 4
    timeout_minutes: int = 60

@dataclass
class AnalyticsConfig:
    """Analytics system configuration"""
    dashboard_type: str
    update_frequency: str = "real_time"
    kpi_metrics: List[Dict[str, Any]]
    alert_thresholds: Dict[str, float]
    visualization_types: List[str]
    export_formats: List[str]

@dataclass
class RetentionConfig:
    """Data retention configuration"""
    policy_name: str
    data_types: List[str]
    retention_period: RetentionPeriod
    archival_location: str
    access_controls: Dict[str, Any]
    compliance_requirements: List[str]
    automated_cleanup: bool = True

# Production configuration constants
PRODUCTION_CONFIG = {
    "environment": "production",
    "compliance_framework": "HIPAA",
    "encryption_algorithm": "AES-256",
    "audit_retention_days": 2555,  # 7 years
    "data_classification_levels": ["PHI", "PII", "Internal", "Public"],
    
    "data_sources": [
        DataSourceConfig(
            name="Epic EHR",
            source_type=DataSource.EHR,
            connection_string=os.getenv("EHR_DB_CONNECTION"),
            schema="epic_production",
            tables=["patients", "encounters", "observations", "procedures"],
            quality_level=DataQuality.CRITICAL,
            retention_period=RetentionPeriod.PERMANENT,
            hipaa_classification="PHI",
            validation_rules=["patient_id_format", "date_validity", "required_fields"]
        ),
        DataSourceConfig(
            name="Lab Results",
            source_type=DataSource.LAB_RESULTS,
            connection_string=os.getenv("LAB_DB_CONNECTION"),
            schema="lab_production",
            tables=["test_results", "specimens", "reference_ranges"],
            quality_level=DataQuality.HIGH,
            retention_period=RetentionPeriod.LONG_TERM,
            hipaa_classification="PHI",
            validation_rules=["test_codes", "reference_ranges", "unit_conversion"]
        ),
        DataSourceConfig(
            name="Medical Imaging",
            source_type=DataSource.IMAGING,
            connection_string=os.getenv("IMAGING_STORAGE_PATH"),
            schema="imaging_metadata",
            tables=["studies", "series", "instances", "reports"],
            quality_level=DataQuality.HIGH,
            retention_period=RetentionPeriod.PERMANENT,
            hipaa_classification="PHI",
            validation_rules=["dicom_compliance", "metadata_integrity", "file_integrity"]
        )
    ],
    
    "etl_pipelines": [
        ETLConfig(
            pipeline_name="patient_data_sync",
            source_config=DataSourceConfig(
                name="Patient Sync",
                source_type=DataSource.EHR,
                connection_string="",
                schema="",
                tables=[],
                quality_level=DataQuality.CRITICAL,
                retention_period=RetentionPeriod.ACTIVE,
                hipaa_classification="PHI"
            ),
            transformation_rules=[
                {"type": "phi_anonymization", "fields": ["name", "ssn", "address"]},
                {"type": "date_standardization", "fields": ["birth_date", "admission_date"]},
                {"type": "code_normalization", "fields": ["diagnosis_codes", "procedure_codes"]}
            ],
            data_quality_checks=[
                {"rule": "completeness", "threshold": 0.95, "fields": ["patient_id", "birth_date"]},
                {"rule": "validity", "threshold": 0.99, "fields": ["diagnosis_codes"]},
                {"rule": "consistency", "threshold": 0.98, "fields": ["admission_date", "discharge_date"]}
            ]
        )
    ],
    
    "analytics_config": AnalyticsConfig(
        dashboard_type="healthcare_analytics",
        update_frequency="real_time",
        kpi_metrics=[
            {"name": "patient_safety_score", "target": 95.0, "category": "safety"},
            {"name": "clinical_quality_index", "target": 90.0, "category": "quality"},
            {"name": "operational_efficiency", "target": 85.0, "category": "efficiency"},
            {"name": "cost_per_encounter", "target": 1500.0, "category": "cost"},
            {"name": "readmission_rate", "target": 0.12, "category": "outcome"},
            {"name": "patient_satisfaction", "target": 4.5, "category": "satisfaction"}
        ],
        alert_thresholds={
            "patient_safety_score": 90.0,
            "clinical_quality_index": 85.0,
            "readmission_rate": 0.15,
            "cost_per_encounter": 2000.0
        },
        visualization_types=["line_charts", "bar_charts", "heatmaps", "scatter_plots"],
        export_formats=["pdf", "csv", "xlsx", "json"]
    ),
    
    "retention_policies": [
        RetentionConfig(
            policy_name="phi_active_storage",
            data_types=["patient_records", "encounters", "medications"],
            retention_period=RetentionPeriod.ACTIVE,
            archival_location="encrypted_s3_bucket",
            access_controls={"role_based": True, "audit_required": True},
            compliance_requirements=["HIPAA", "HITECH"],
            automated_cleanup=True
        ),
        RetentionConfig(
            policy_name="clinical_archive",
            data_types=["imaging_studies", "lab_results", "clinical_notes"],
            retention_period=RetentionPeriod.LONG_TERM,
            archival_location="cold_storage_archive",
            access_controls={"role_based": True, "audit_required": True, "approval_required": True},
            compliance_requirements=["HIPAA", "HITECH", "State_Laws"],
            automated_cleanup=False
        )
    ]
}

def get_data_source_config(source_name: str) -> DataSourceConfig:
    """Get configuration for a specific data source"""
    for source in PRODUCTION_CONFIG["data_sources"]:
        if source.name == source_name:
            return source
    raise ValueError(f"Data source {source_name} not found")

def get_etl_config(pipeline_name: str) -> ETLConfig:
    """Get configuration for a specific ETL pipeline"""
    for pipeline in PRODUCTION_CONFIG["etl_pipelines"]:
        if pipeline.pipeline_name == pipeline_name:
            return pipeline
    raise ValueError(f"ETL pipeline {pipeline_name} not found")

def get_retention_policy(policy_name: str) -> RetentionConfig:
    """Get configuration for a specific retention policy"""
    for policy in PRODUCTION_CONFIG["retention_policies"]:
        if policy.policy_name == policy_name:
            return policy
    raise ValueError(f"Retention policy {policy_name} not found")

def validate_config() -> Dict[str, Any]:
    """Validate production configuration"""
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required environment variables
    required_env_vars = ["EHR_DB_CONNECTION", "LAB_DB_CONNECTION", "ENCRYPTION_KEY"]
    for var in required_env_vars:
        if not os.getenv(var):
            validation_results["errors"].append(f"Missing required environment variable: {var}")
            validation_results["valid"] = False
    
    # Check data source configurations
    for source in PRODUCTION_CONFIG["data_sources"]:
        if not source.connection_string:
            validation_results["warnings"].append(f"No connection string for {source.name}")
    
    return validation_results