"""
Configuration management for Model Version Tracking System.

Provides centralized configuration for all components with environment-specific
settings and medical compliance requirements.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class DatabaseConfig:
    """Database configuration for metadata storage."""
    url: str = "sqlite:///./version_tracking.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


@dataclass
class RegistryConfig:
    """External registry integration configuration."""
    mlflow_tracking_uri: str = ""
    mlflow_registry_uri: str = ""
    mlflow_username: str = ""
    mlflow_password: str = ""
    
    wandb_project: str = "medical-ai-models"
    wandb_entity: str = ""
    wandb_api_key: str = ""
    
    sync_enabled: bool = True
    sync_interval_minutes: int = 60


@dataclass
class ComplianceConfig:
    """Medical compliance and regulatory configuration."""
    regulatory_authority: str = "FDA"
    compliance_framework: str = "FDA_21CFR820"
    audit_retention_days: int = 2555  # 7 years
    validation_required: bool = True
    irb_approval_required: bool = True
    
    # Clinical thresholds
    min_medical_accuracy: float = 0.85
    min_diagnostic_accuracy: float = 0.90
    min_clinical_sensitivity: float = 0.95
    min_clinical_specificity: float = 0.90
    min_auc_roc: float = 0.85
    max_latency_ms: float = 1000.0
    max_error_rate: float = 0.01
    
    # Risk management
    risk_assessment_required: bool = True
    clinical_benefit_documentation_required: bool = True
    post_market_surveillance_required: bool = True


@dataclass
class DeploymentConfig:
    """Deployment and rollout configuration."""
    default_deployment_type: str = "canary"
    default_rollout_percentage: float = 10.0
    health_check_interval_seconds: int = 60
    rollback_threshold: float = 0.95
    auto_rollback_enabled: bool = True
    max_rollback_time_hours: int = 24
    
    # Canary settings
    canary_duration_minutes: int = 60
    canary_success_threshold: float = 0.90
    
    # Blue-green settings
    green_environment_prefix: str = "green-"
    blue_environment_prefix: str = "blue-"
    
    # Rolling deployment settings
    rolling_batch_size: int = 4
    rolling_pause_seconds: int = 30


@dataclass
class TestingConfig:
    """A/B testing and experiment configuration."""
    default_significance_level: float = 0.05
    default_statistical_power: float = 0.80
    default_minimum_sample_size: int = 1000
    default_maximum_duration_days: int = 30
    default_minimum_detectable_effect: float = 0.05
    
    # Safety monitoring
    safety_monitoring_interval_minutes: int = 15
    high_severity_alert_threshold: int = 1
    experiment_auto_stop_enabled: bool = True
    
    # Clinical testing requirements
    clinical_approval_required: bool = True
    irb_approval_required: bool = True
    principal_investigator_required: bool = True


@dataclass
class SecurityConfig:
    """Security and audit configuration."""
    audit_log_encryption: bool = True
    audit_log_retention_days: int = 2555  # 7 years
    sensitive_data_encryption: bool = True
    api_key_rotation_days: int = 90
    
    # Access control
    require_authentication: bool = True
    require_authorization: bool = True
    role_based_access: bool = True
    
    # Audit trail
    detailed_audit_trail: bool = True
    audit_trail_compression: bool = True
    audit_trail_backup: bool = True
    
    # PHI protection
    phi_redaction_enabled: bool = True
    phi_encryption_algorithm: str = "AES-256"
    phi_key_rotation_days: int = 30


@dataclass
class VersionTrackingConfig:
    """Main configuration class."""
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    log_level: str = "INFO"
    
    # Storage paths
    registry_path: str = "./registry"
    metadata_path: str = "./metadata"
    audit_log_path: str = "./logs"
    backup_path: str = "./backup"
    
    # Component configurations
    database: DatabaseConfig = None
    registry: RegistryConfig = None
    compliance: ComplianceConfig = None
    deployment: DeploymentConfig = None
    testing: TestingConfig = None
    security: SecurityConfig = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.database is None:
            self.database = DatabaseConfig()
        if self.registry is None:
            self.registry = RegistryConfig()
        if self.compliance is None:
            self.compliance = ComplianceConfig()
        if self.deployment is None:
            self.deployment = DeploymentConfig()
        if self.testing is None:
            self.testing = TestingConfig()
        if self.security is None:
            self.security = SecurityConfig()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"
    
    def get_registry_paths(self) -> Dict[str, Path]:
        """Get all registry paths as Path objects."""
        return {
            "registry": Path(self.registry_path),
            "metadata": Path(self.metadata_path),
            "audit_log": Path(self.audit_log_path),
            "backup": Path(self.backup_path)
        }
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Environment validation
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            errors.append(f"Invalid environment '{self.environment}'. Must be one of {valid_environments}")
        
        # Compliance thresholds validation
        thresholds = [
            ("min_medical_accuracy", self.compliance.min_medical_accuracy, 0.0, 1.0),
            ("min_diagnostic_accuracy", self.compliance.min_diagnostic_accuracy, 0.0, 1.0),
            ("min_clinical_sensitivity", self.compliance.min_clinical_sensitivity, 0.0, 1.0),
            ("min_clinical_specificity", self.compliance.min_clinical_specificity, 0.0, 1.0),
            ("min_auc_roc", self.compliance.min_auc_roc, 0.0, 1.0),
            ("max_latency_ms", self.compliance.max_latency_ms, 0.0, float('inf')),
            ("max_error_rate", self.compliance.max_error_rate, 0.0, 1.0)
        ]
        
        for name, value, min_val, max_val in thresholds:
            if not (min_val <= value <= max_val):
                errors.append(f"Invalid {name}: {value} (must be between {min_val} and {max_val})")
        
        # Path validation
        for name, path in self.get_registry_paths().items():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create {name} path '{path}': {e}")
        
        # Registry validation
        if self.registry.sync_enabled:
            if not self.registry.mlflow_tracking_uri and not self.registry.wandb_api_key:
                errors.append("Registry sync enabled but no registry credentials provided")
        
        return errors
    
    @classmethod
    def from_environment(cls, env_file: str = None) -> 'VersionTrackingConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Environment settings
        config.environment = os.getenv('VERSION_TRACKING_ENV', 'development')
        config.debug = os.getenv('VERSION_TRACKING_DEBUG', 'false').lower() == 'true'
        config.log_level = os.getenv('VERSION_TRACKING_LOG_LEVEL', 'INFO')
        
        # Path settings
        config.registry_path = os.getenv('VERSION_TRACKING_REGISTRY_PATH', './registry')
        config.metadata_path = os.getenv('VERSION_TRACKING_METADATA_PATH', './metadata')
        config.audit_log_path = os.getenv('VERSION_TRACKING_AUDIT_LOG_PATH', './logs')
        config.backup_path = os.getenv('VERSION_TRACKING_BACKUP_PATH', './backup')
        
        # Database configuration
        config.database = DatabaseConfig(
            url=os.getenv('DATABASE_URL', 'sqlite:///./version_tracking.db'),
            echo=os.getenv('DATABASE_ECHO', 'false').lower() == 'true'
        )
        
        # Registry configuration
        config.registry = RegistryConfig(
            mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI', ''),
            mlflow_registry_uri=os.getenv('MLFLOW_REGISTRY_URI', ''),
            wandb_project=os.getenv('WANDB_PROJECT', 'medical-ai-models'),
            wandb_api_key=os.getenv('WANDB_API_KEY', ''),
            sync_enabled=os.getenv('REGISTRY_SYNC_ENABLED', 'true').lower() == 'true'
        )
        
        # Compliance configuration
        config.compliance = ComplianceConfig(
            regulatory_authority=os.getenv('COMPLIANCE_REGULATORY_AUTHORITY', 'FDA'),
            min_medical_accuracy=float(os.getenv('COMPLIANCE_MIN_MEDICAL_ACCURACY', '0.85')),
            min_diagnostic_accuracy=float(os.getenv('COMPLIANCE_MIN_DIAGNOSTIC_ACCURACY', '0.90')),
            min_clinical_sensitivity=float(os.getenv('COMPLIANCE_MIN_CLINICAL_SENSITIVITY', '0.95')),
            min_clinical_specificity=float(os.getenv('COMPLIANCE_MIN_CLINICAL_SPECIFICITY', '0.90')),
            min_auc_roc=float(os.getenv('COMPLIANCE_MIN_AUC_ROC', '0.85')),
            max_latency_ms=float(os.getenv('COMPLIANCE_MAX_LATENCY_MS', '1000.0')),
            max_error_rate=float(os.getenv('COMPLIANCE_MAX_ERROR_RATE', '0.01'))
        )
        
        # Security configuration
        config.security = SecurityConfig(
            audit_log_encryption=os.getenv('SECURITY_AUDIT_LOG_ENCRYPTION', 'true').lower() == 'true',
            phi_redaction_enabled=os.getenv('SECURITY_PHI_REDACTION', 'true').lower() == 'true',
            require_authentication=os.getenv('SECURITY_REQUIRE_AUTH', 'true').lower() == 'true'
        )
        
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'VersionTrackingConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'log_level': self.log_level,
            'registry_path': self.registry_path,
            'metadata_path': self.metadata_path,
            'audit_log_path': self.audit_log_path,
            'backup_path': self.backup_path,
            'database': self.database.__dict__,
            'registry': self.registry.__dict__,
            'compliance': self.compliance.__dict__,
            'deployment': self.deployment.__dict__,
            'testing': self.testing.__dict__,
            'security': self.security.__dict__
        }


# Global configuration instance
config = VersionTrackingConfig.from_environment()


def get_config() -> VersionTrackingConfig:
    """Get the global configuration instance."""
    return config


def set_config(new_config: VersionTrackingConfig):
    """Set the global configuration instance."""
    global config
    config = new_config


# Configuration validators
def validate_compliance_thresholds(thresholds: Dict[str, float]) -> List[str]:
    """Validate clinical thresholds."""
    errors = []
    
    required_thresholds = [
        'min_medical_accuracy',
        'min_diagnostic_accuracy', 
        'min_clinical_sensitivity',
        'min_clinical_specificity',
        'min_auc_roc'
    ]
    
    for threshold in required_thresholds:
        if threshold not in thresholds:
            errors.append(f"Missing required threshold: {threshold}")
        elif not (0.0 <= thresholds[threshold] <= 1.0):
            errors.append(f"Invalid {threshold}: must be between 0.0 and 1.0")
    
    # Operational thresholds
    operational_thresholds = [
        ('max_latency_ms', thresholds.get('max_latency_ms', 1000.0), 0.0, float('inf')),
        ('max_error_rate', thresholds.get('max_error_rate', 0.01), 0.0, 1.0)
    ]
    
    for name, value, min_val, max_val in operational_thresholds:
        if not (min_val <= value <= max_val):
            errors.append(f"Invalid {name}: {value} (must be between {min_val} and {max_val})")
    
    return errors


def get_default_health_checks() -> List[Dict[str, Any]]:
    """Get default health check configuration."""
    return [
        {
            "name": "latency",
            "check_type": "latency",
            "threshold": 1000.0,
            "window_minutes": 5,
            "failure_threshold": 3,
            "severity": "error"
        },
        {
            "name": "error_rate", 
            "check_type": "error_rate",
            "threshold": 0.05,
            "window_minutes": 5,
            "failure_threshold": 2,
            "severity": "critical"
        },
        {
            "name": "accuracy",
            "check_type": "accuracy", 
            "threshold": 0.85,
            "window_minutes": 10,
            "failure_threshold": 3,
            "severity": "critical"
        }
    ]


def get_environment_specific_config(env: str) -> VersionTrackingConfig:
    """Get environment-specific configuration."""
    env_configs = {
        'development': {
            'debug': True,
            'log_level': 'DEBUG',
            'database': {'echo': True},
            'security': {
                'audit_log_encryption': False,
                'require_authentication': False
            }
        },
        'staging': {
            'debug': False,
            'log_level': 'INFO',
            'database': {'echo': False},
            'security': {
                'audit_log_encryption': True,
                'require_authentication': True
            }
        },
        'production': {
            'debug': False,
            'log_level': 'WARNING',
            'database': {'echo': False},
            'security': {
                'audit_log_encryption': True,
                'require_authentication': True,
                'phi_redaction_enabled': True
            },
            'compliance': {
                'validation_required': True,
                'irb_approval_required': True
            }
        }
    }
    
    if env not in env_configs:
        raise ValueError(f"Unknown environment: {env}")
    
    base_config = VersionTrackingConfig.from_environment()
    env_overrides = env_configs[env]
    
    # Apply environment-specific overrides
    for section, overrides in env_overrides.items():
        if hasattr(base_config, section):
            section_obj = getattr(base_config, section)
            for key, value in overrides.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
    
    return base_config