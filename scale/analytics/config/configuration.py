"""
Analytics Platform Configuration Management
Centralized configuration for all analytics modules
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class AnalyticsConfig:
    """Core analytics platform configuration"""
    
    # Platform settings
    platform_name: str = "Advanced Analytics Platform"
    version: str = "1.0.0"
    environment: str = "production"
    debug_mode: bool = False
    
    # Database configuration
    database_url: str = "postgresql://user:password@localhost:5432/analytics"
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    # Machine Learning settings
    ml_confidence_threshold: float = 0.75
    auto_model_training: bool = True
    model_retrain_interval: int = 24  # hours
    prediction_batch_size: int = 1000
    
    # Analytics settings
    insight_confidence_threshold: float = 0.7
    insight_impact_threshold: float = 0.6
    data_quality_threshold: float = 0.8
    
    # Performance settings
    max_concurrent_analyses: int = 10
    timeout_seconds: int = 300
    memory_limit_mb: int = 4096
    
    # Security settings
    encryption_enabled: bool = True
    audit_logging: bool = True
    data_retention_days: int = 365
    
    # Integration settings
    api_rate_limit: int = 1000
    external_apis_enabled: bool = True
    webhook_callbacks: bool = True
    
    # Notification settings
    email_notifications: bool = True
    dashboard_alerts: bool = True
    threshold_violation_alerts: bool = True

class ConfigManager:
    """Configuration manager for analytics platform"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "analytics_config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> AnalyticsConfig:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                return AnalyticsConfig(**config_data)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_file}: {e}")
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> AnalyticsConfig:
        """Create and save default configuration"""
        config = AnalyticsConfig()
        self.save_config(config)
        return config
    
    def save_config(self, config: AnalyticsConfig) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(asdict(config), f, indent=2, default=str)
        except Exception as e:
            raise Exception(f"Error saving configuration: {e}")
    
    def get_config(self) -> AnalyticsConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs) -> AnalyticsConfig:
        """Update configuration with new values"""
        current_config = asdict(self.config)
        current_config.update(kwargs)
        self.config = AnalyticsConfig(**current_config)
        self.save_config(self.config)
        return self.config
    
    def validate_config(self) -> Dict[str, bool]:
        """Validate configuration settings"""
        validation_results = {}
        
        # Validate ML settings
        validation_results['ml_confidence_threshold'] = 0.0 <= self.config.ml_confidence_threshold <= 1.0
        validation_results['auto_model_training'] = isinstance(self.config.auto_model_training, bool)
        
        # Validate performance settings
        validation_results['max_concurrent_analyses'] = self.config.max_concurrent_analyses > 0
        validation_results['timeout_seconds'] = self.config.timeout_seconds > 0
        validation_results['memory_limit_mb'] = self.config.memory_limit_mb > 0
        
        # Validate security settings
        validation_results['encryption_enabled'] = isinstance(self.config.encryption_enabled, bool)
        validation_results['audit_logging'] = isinstance(self.config.audit_logging, bool)
        
        return validation_results

# Environment-specific configurations
def get_production_config() -> AnalyticsConfig:
    """Get production environment configuration"""
    return AnalyticsConfig(
        environment="production",
        debug_mode=False,
        cache_enabled=True,
        encryption_enabled=True,
        audit_logging=True,
        external_apis_enabled=True,
        max_concurrent_analyses=20,
        memory_limit_mb=8192
    )

def get_development_config() -> AnalyticsConfig:
    """Get development environment configuration"""
    return AnalyticsConfig(
        environment="development",
        debug_mode=True,
        cache_enabled=False,
        encryption_enabled=False,
        audit_logging=False,
        external_apis_enabled=False,
        max_concurrent_analyses=5,
        memory_limit_mb=2048
    )

def get_testing_config() -> AnalyticsConfig:
    """Get testing environment configuration"""
    return AnalyticsConfig(
        environment="testing",
        debug_mode=True,
        cache_enabled=False,
        encryption_enabled=False,
        audit_logging=False,
        external_apis_enabled=False,
        max_concurrent_analyses=2,
        memory_limit_mb=1024
    )

# Configuration presets for different use cases
def get_enterprise_config() -> AnalyticsConfig:
    """Get enterprise-grade configuration"""
    return AnalyticsConfig(
        environment="production",
        platform_name="Enterprise Analytics Platform",
        max_concurrent_analyses=50,
        memory_limit_mb=16384,
        encryption_enabled=True,
        audit_logging=True,
        data_retention_days=730,
        api_rate_limit=5000,
        email_notifications=True,
        dashboard_alerts=True
    )

def get_startup_config() -> AnalyticsConfig:
    """Get startup-friendly configuration"""
    return AnalyticsConfig(
        environment="development",
        platform_name="Startup Analytics Platform",
        max_concurrent_analyses=5,
        memory_limit_mb=2048,
        encryption_enabled=False,
        audit_logging=False,
        data_retention_days=90,
        api_rate_limit=100,
        email_notifications=False,
        dashboard_alerts=True
    )

if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    
    # Validate current configuration
    validation = config_manager.validate_config()
    print("Configuration Validation Results:")
    for key, is_valid in validation.items():
        print(f"  {key}: {'✓' if is_valid else '✗'}")
    
    # Update configuration
    config_manager.update_config(
        ml_confidence_threshold=0.8,
        max_concurrent_analyses=15
    )
    
    print(f"\nUpdated configuration: {config_manager.get_config().ml_confidence_threshold}")