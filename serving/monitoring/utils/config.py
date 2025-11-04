"""
Configuration management for monitoring system.

Provides comprehensive configuration loading, validation, and management
for all monitoring components.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import timedelta

try:
    from pydantic import BaseSettings, Field, validator
except ImportError:
    # Fallback for environments without pydantic
    from typing import Any
    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def Field(default=None, **kwargs):
        class FieldContainer:
            def __init__(self, default, **kwargs):
                self.default = default
                self.kwargs = kwargs
        return FieldContainer(default, **kwargs)


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    collection_interval: float = 1.0
    retention_days: int = 30
    enable_prometheus: bool = True
    enable_prometheus_pushgateway: bool = False
    prometheus_pushgateway_url: str = "http://localhost:9091"
    metrics_endpoint: str = "/metrics"
    save_directory: str = "./monitoring_data"
    max_metrics_memory: int = 1000  # Maximum metrics to keep in memory


@dataclass
class SystemMonitoringConfig:
    """System monitoring configuration."""
    enabled: bool = True
    check_interval: int = 60
    enable_gpu_monitoring: bool = True
    enable_disk_io_monitoring: bool = True
    enable_network_monitoring: bool = True
    cpu_alert_threshold: float = 85.0
    memory_alert_threshold: float = 90.0
    gpu_memory_alert_threshold: float = 95.0
    disk_usage_alert_threshold: float = 90.0


@dataclass
class ModelMonitoringConfig:
    """Model-specific monitoring configuration."""
    enabled: bool = True
    accuracy_monitoring_enabled: bool = True
    drift_detection_enabled: bool = True
    performance_tracking_enabled: bool = True
    min_samples_for_evaluation: int = 100
    evaluation_interval: int = 3600  # 1 hour
    drift_threshold: float = 0.3
    accuracy_threshold: float = 0.85
    latency_threshold_ms: float = 2000.0
    error_rate_threshold: float = 0.05


@dataclass
class ClinicalOutcomeConfig:
    """Clinical outcome tracking configuration."""
    enabled: bool = True
    tracking_window_days: int = 90
    validation_delay_days: int = 30
    min_outcomes_for_analysis: int = 50
    enable_regulatory_compliance: bool = True
    compliance_frameworks: List[str] = field(default_factory=lambda: ["hipaa", "fda", "gdpr"])
    outcome_tracking_endpoints: List[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    """Alerting configuration."""
    enabled: bool = True
    escalation_enabled: bool = True
    max_escalations: int = 3
    escalation_interval: int = 900  # 15 minutes
    notification_cooldown: int = 1800  # 30 minutes
    suppress_on_maintenance: bool = True
    
    # SLA thresholds
    sla_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "response_time_ms": 2000.0,
        "availability_percent": 99.9,
        "accuracy_score": 0.95,
        "clinical_effectiveness": 0.90,
        "error_rate": 0.01
    })


@dataclass
class NotificationConfig:
    """Notification system configuration."""
    enabled: bool = True
    
    # Email configuration
    email_enabled: bool = True
    smtp_host: str = "localhost"
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""
    from_address: str = "alerts@medical-ai.com"
    
    # Email recipients by severity
    recipients: Dict[str, List[str]] = field(default_factory=lambda: {
        "emergency": [],
        "critical": [],
        "error": [],
        "warning": ["admin@medical-ai.com"],
        "info": []
    })
    
    # Slack configuration
    slack_enabled: bool = True
    slack_webhook_url: str = ""
    slack_default_channel: str = "#alerts"
    
    # Webhook configuration
    webhook_enabled: bool = True
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})
    webhook_auth_type: str = "bearer"  # bearer, basic, none
    webhook_auth_token: str = ""
    
    # SMS configuration
    sms_enabled: bool = True
    sms_provider: str = "twilio"  # twilio, aws_sns, etc.
    sms_account_sid: str = ""
    sms_auth_token: str = ""
    sms_from_number: str = ""
    sms_recipients: Dict[str, List[str]] = field(default_factory=lambda: {
        "emergency": [],
        "critical": []
    })


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enabled: bool = True
    grafana_url: str = "http://localhost:3000"
    grafana_api_key: str = ""
    grafana_username: str = ""
    grafana_password: str = ""
    dashboard_folder: str = "Medical AI"
    refresh_interval: str = "30s"
    auto_deploy_dashboards: bool = True
    dashboard_templates: List[str] = field(default_factory=lambda: [
        "system_overview",
        "model_performance", 
        "clinical_outcomes",
        "alert_management",
        "regulatory_compliance",
        "executive_dashboard"
    ])


@dataclass
class StorageConfig:
    """Storage configuration."""
    type: str = "local"  # local, s3, azure, gcp
    base_path: str = "./monitoring_data"
    
    # S3 configuration
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    
    # Database configuration
    database_url: str = ""
    database_type: str = "postgresql"  # postgresql, mysql, sqlite
    
    # Retention policies
    metrics_retention_days: int = 30
    logs_retention_days: int = 90
    audit_logs_retention_days: int = 2555  # 7 years for medical records
    
    # Backup configuration
    backup_enabled: bool = True
    backup_interval: int = 86400  # 24 hours
    backup_retention_days: int = 365


@dataclass
class SecurityConfig:
    """Security and compliance configuration."""
    enable_phi_protection: bool = True
    enable_data_encryption: bool = True
    enable_audit_logging: bool = True
    enable_access_control: bool = True
    
    # Encryption
    encryption_key: str = ""
    encryption_algorithm: str = "AES-256"
    
    # Audit logging
    audit_log_file: str = "./logs/audit.log"
    audit_log_level: str = "INFO"
    audit_include_requests: bool = True
    audit_include_responses: bool = False
    
    # Access control
    require_api_key: bool = False
    api_keys: List[str] = field(default_factory=list)
    allowed_ips: List[str] = field(default_factory=list)
    
    # Data retention
    data_retention_days: int = 30
    phi_redaction_enabled: bool = True


@dataclass
class MonitoringConfig:
    """Main monitoring configuration."""
    
    # Core components
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    system_monitoring: SystemMonitoringConfig = field(default_factory=SystemMonitoringConfig)
    model_monitoring: ModelMonitoringConfig = field(default_factory=ModelMonitoringConfig)
    clinical_outcomes: ClinicalOutcomeConfig = field(default_factory=ClinicalOutcomeConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    dashboards: DashboardConfig = field(default_factory=DashboardConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Global settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Component enable/disable
    components_enabled: Dict[str, bool] = field(default_factory=lambda: {
        "metrics_collector": True,
        "system_monitor": True,
        "model_monitor": True,
        "clinical_tracker": True,
        "alert_manager": True,
        "notification_system": True,
        "dashboard_manager": True,
        "drift_detector": True,
        "accuracy_monitor": True,
        "compliance_monitor": True
    })


class ConfigManager:
    """Configuration management for monitoring system."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._find_config_file()
        self.config: Optional[MonitoringConfig] = None
        self._load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations."""
        possible_paths = [
            "./config/monitoring_config.yaml",
            "./config/monitoring_config.yml",
            "./monitoring_config.yaml",
            "./monitoring_config.yml",
            "./config.yaml",
            "./config.yml",
            os.getenv("MONITORING_CONFIG_FILE", "")
        ]
        
        for path in possible_paths:
            if path and Path(path).exists():
                return path
        
        # Return default path if none found
        return "./config/monitoring_config.yaml"
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if Path(self.config_file).exists():
                with open(self.config_file, 'r') as f:
                    if self.config_file.endswith(('.yaml', '.yml')):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                self.config = self._parse_config(config_data)
            else:
                # Create default configuration
                self.config = MonitoringConfig()
                self._save_config()  # Save default config
                
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            self.config = MonitoringConfig()  # Fallback to defaults
    
    def _parse_config(self, data: Dict[str, Any]) -> MonitoringConfig:
        """Parse configuration dictionary into MonitoringConfig object."""
        
        # Parse nested configurations
        metrics_config = MetricsConfig(**data.get('metrics', {}))
        
        system_monitoring_config = SystemMonitoringConfig(**data.get('system_monitoring', {}))
        
        model_monitoring_config = ModelMonitoringConfig(**data.get('model_monitoring', {}))
        
        clinical_outcomes_config = ClinicalOutcomeConfig(**data.get('clinical_outcomes', {}))
        
        alerts_config = AlertConfig(**data.get('alerts', {}))
        
        notifications_config = NotificationConfig(**data.get('notifications', {}))
        
        dashboards_config = DashboardConfig(**data.get('dashboards', {}))
        
        storage_config = StorageConfig(**data.get('storage', {}))
        
        security_config = SecurityConfig(**data.get('security', {}))
        
        # Parse global settings
        return MonitoringConfig(
            metrics=metrics_config,
            system_monitoring=system_monitoring_config,
            model_monitoring=model_monitoring_config,
            clinical_outcomes=clinical_outcomes_config,
            alerts=alerts_config,
            notifications=notifications_config,
            dashboards=dashboards_config,
            storage=storage_config,
            security=security_config,
            environment=data.get('environment', 'development'),
            debug=data.get('debug', False),
            log_level=data.get('log_level', 'INFO'),
            log_format=data.get('log_format', 'json'),
            components_enabled=data.get('components_enabled', {})
        )
    
    def _save_config(self):
        """Save current configuration to file."""
        try:
            # Ensure directory exists
            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            config_dict = asdict(self.config)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                if self.config_file.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
                    
        except Exception as e:
            print(f"Failed to save configuration: {e}")
    
    def get_config(self) -> MonitoringConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if not self.config:
            return
        
        # Update nested configurations
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(value, dict) and hasattr(getattr(self.config, key), '__dict__'):
                    # Update nested config object
                    nested_config = getattr(self.config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    # Update direct attribute
                    setattr(self.config, key, value)
        
        # Save updated configuration
        self._save_config()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        if not self.config:
            issues.append("Configuration not loaded")
            return issues
        
        # Validate metrics configuration
        if self.config.metrics.collection_interval <= 0:
            issues.append("Metrics collection interval must be positive")
        
        if self.config.metrics.retention_days <= 0:
            issues.append("Metrics retention days must be positive")
        
        # Validate system monitoring
        if self.config.system_monitoring.check_interval <= 0:
            issues.append("System monitoring check interval must be positive")
        
        # Validate model monitoring
        if self.config.model_monitoring.min_samples_for_evaluation <= 0:
            issues.append("Minimum samples for evaluation must be positive")
        
        if self.config.model_monitoring.drift_threshold < 0 or self.config.model_monitoring.drift_threshold > 1:
            issues.append("Drift threshold must be between 0 and 1")
        
        if self.config.model_monitoring.accuracy_threshold < 0 or self.config.model_monitoring.accuracy_threshold > 1:
            issues.append("Accuracy threshold must be between 0 and 1")
        
        # Validate clinical outcomes
        if self.config.clinical_outcomes.tracking_window_days <= 0:
            issues.append("Clinical outcomes tracking window must be positive")
        
        # Validate alerts
        if self.config.alerts.escalation_interval <= 0:
            issues.append("Alert escalation interval must be positive")
        
        # Validate notifications
        if self.config.notifications.email_enabled:
            if not self.config.notifications.smtp_host:
                issues.append("SMTP host required when email is enabled")
            
            if not self.config.notifications.recipients:
                issues.append("No email recipients configured")
        
        if self.config.notifications.slack_enabled:
            if not self.config.notifications.slack_webhook_url:
                issues.append("Slack webhook URL required when Slack is enabled")
        
        # Validate dashboards
        if self.config.dashboards.enabled:
            if not self.config.dashboards.grafana_url:
                issues.append("Grafana URL required when dashboards are enabled")
        
        # Validate storage
        if self.config.storage.type == "s3":
            if not self.config.storage.s3_bucket:
                issues.append("S3 bucket required for S3 storage")
        
        # Validate security
        if self.config.security.enable_audit_logging:
            if not self.config.security.audit_log_file:
                issues.append("Audit log file path required when audit logging is enabled")
        
        return issues
    
    def export_config(self, format: str = "yaml") -> str:
        """Export configuration as string."""
        if not self.config:
            return ""
        
        config_dict = asdict(self.config)
        
        if format.lower() == "json":
            return json.dumps(config_dict, indent=2)
        elif format.lower() in ["yaml", "yml"]:
            return yaml.dump(config_dict, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def reload_config(self):
        """Reload configuration from file."""
        self._load_config()


# Default configuration template
DEFAULT_CONFIG_TEMPLATE = """
# Medical AI Monitoring System Configuration

# Global Settings
environment: development
debug: false
log_level: INFO
log_format: json

# Metrics Collection
metrics:
  collection_interval: 1.0
  retention_days: 30
  enable_prometheus: true
  enable_prometheus_pushgateway: false
  prometheus_pushgateway_url: "http://localhost:9091"
  metrics_endpoint: "/metrics"
  save_directory: "./monitoring_data"
  max_metrics_memory: 1000

# System Monitoring
system_monitoring:
  enabled: true
  check_interval: 60
  enable_gpu_monitoring: true
  enable_disk_io_monitoring: true
  enable_network_monitoring: true
  cpu_alert_threshold: 85.0
  memory_alert_threshold: 90.0
  gpu_memory_alert_threshold: 95.0
  disk_usage_alert_threshold: 90.0

# Model Monitoring
model_monitoring:
  enabled: true
  accuracy_monitoring_enabled: true
  drift_detection_enabled: true
  performance_tracking_enabled: true
  min_samples_for_evaluation: 100
  evaluation_interval: 3600
  drift_threshold: 0.3
  accuracy_threshold: 0.85
  latency_threshold_ms: 2000.0
  error_rate_threshold: 0.05

# Clinical Outcomes Tracking
clinical_outcomes:
  enabled: true
  tracking_window_days: 90
  validation_delay_days: 30
  min_outcomes_for_analysis: 50
  enable_regulatory_compliance: true
  compliance_frameworks: ["hipaa", "fda", "gdpr"]

# Alerting
alerts:
  enabled: true
  escalation_enabled: true
  max_escalations: 3
  escalation_interval: 900
  notification_cooldown: 1800
  suppress_on_maintenance: true
  sla_thresholds:
    response_time_ms: 2000.0
    availability_percent: 99.9
    accuracy_score: 0.95
    clinical_effectiveness: 0.90
    error_rate: 0.01

# Notifications
notifications:
  enabled: true
  
  # Email Configuration
  email_enabled: true
  smtp_host: "localhost"
  smtp_port: 587
  use_tls: true
  username: ""
  password: ""
  from_address: "alerts@medical-ai.com"
  recipients:
    emergency: []
    critical: []
    error: []
    warning: ["admin@medical-ai.com"]
    info: []
  
  # Slack Configuration
  slack_enabled: true
  slack_webhook_url: ""
  slack_default_channel: "#alerts"
  
  # Webhook Configuration
  webhook_enabled: true
  webhook_url: ""
  webhook_headers:
    Content-Type: "application/json"
  webhook_auth_type: "bearer"
  webhook_auth_token: ""
  
  # SMS Configuration
  sms_enabled: true
  sms_provider: "twilio"
  sms_account_sid: ""
  sms_auth_token: ""
  sms_from_number: ""
  sms_recipients:
    emergency: []
    critical: []

# Dashboard Configuration
dashboards:
  enabled: true
  grafana_url: "http://localhost:3000"
  grafana_api_key: ""
  grafana_username: ""
  grafana_password: ""
  dashboard_folder: "Medical AI"
  refresh_interval: "30s"
  auto_deploy_dashboards: true
  dashboard_templates:
    - "system_overview"
    - "model_performance"
    - "clinical_outcomes"
    - "alert_management"
    - "regulatory_compliance"
    - "executive_dashboard"

# Storage Configuration
storage:
  type: "local"
  base_path: "./monitoring_data"
  
  # S3 Configuration (if using S3)
  s3_bucket: ""
  s3_region: "us-east-1"
  s3_access_key: ""
  s3_secret_key: ""
  
  # Database Configuration
  database_url: ""
  database_type: "postgresql"
  
  # Retention Policies
  metrics_retention_days: 30
  logs_retention_days: 90
  audit_logs_retention_days: 2555
  
  # Backup Configuration
  backup_enabled: true
  backup_interval: 86400
  backup_retention_days: 365

# Security Configuration
security:
  enable_phi_protection: true
  enable_data_encryption: true
  enable_audit_logging: true
  enable_access_control: true
  
  # Encryption
  encryption_key: ""
  encryption_algorithm: "AES-256"
  
  # Audit Logging
  audit_log_file: "./logs/audit.log"
  audit_log_level: "INFO"
  audit_include_requests: true
  audit_include_responses: false
  
  # Access Control
  require_api_key: false
  api_keys: []
  allowed_ips: []
  
  # Data Retention
  data_retention_days: 30
  phi_redaction_enabled: true

# Component Enable/Disable
components_enabled:
  metrics_collector: true
  system_monitor: true
  model_monitor: true
  clinical_tracker: true
  alert_manager: true
  notification_system: true
  dashboard_manager: true
  drift_detector: true
  accuracy_monitor: true
  compliance_monitor: true
"""


def create_default_config(config_file: str = "./config/monitoring_config.yaml"):
    """Create default configuration file."""
    config_path = Path(config_file)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(DEFAULT_CONFIG_TEMPLATE)
    
    print(f"Default configuration created at: {config_path}")


def load_config(config_file: Optional[str] = None) -> MonitoringConfig:
    """Load monitoring configuration."""
    config_manager = ConfigManager(config_file)
    return config_manager.get_config()