"""
Production-Grade Performance Configuration for Medical AI Assistant
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum

class EnvironmentType(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ProductionPerformanceConfig:
    """Production-grade performance configuration"""
    
    # Environment settings
    environment: EnvironmentType = EnvironmentType.PRODUCTION
    debug: bool = False
    
    # Database configuration
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/medical_ai")
    max_db_connections: int = 50
    min_db_connections: int = 10
    db_connection_timeout: int = 30
    db_pool_recycle: int = 3600
    
    # Redis caching configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_max_connections: int = 100
    redis_connection_timeout: int = 5
    
    # Cache TTL configuration (in seconds)
    cache_ttl_config: Dict[str, int] = None
    
    # Auto-scaling configuration
    min_replicas: int = 3
    max_replicas: int = 100
    cpu_target_percentage: int = 70
    memory_target_percentage: int = 80
    scale_up_stabilization: int = 60  # seconds
    scale_down_stabilization: int = 300  # seconds
    
    # Healthcare-specific scaling patterns
    healthcare_scaling_config: Dict[str, Any] = None
    
    # Load testing configuration
    load_test_config: Dict[str, Any] = None
    
    # Frontend optimization
    frontend_config: Dict[str, Any] = None
    
    # Resource management
    resource_limits: Dict[str, Any] = None
    
    # Performance targets
    performance_targets: Dict[str, float] = None
    
    # Monitoring configuration
    monitoring_config: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default configurations"""
        if self.cache_ttl_config is None:
            self.cache_ttl_config = {
                "patient_data": 1800,      # 30 minutes
                "clinical_data": 900,      # 15 minutes
                "ai_inference": 3600,      # 1 hour
                "audit_logs": 7200,        # 2 hours
                "vital_signs": 600,        # 10 minutes
                "medications": 1200,       # 20 minutes
                "lab_results": 1800,       # 30 minutes
                "appointments": 3600,      # 1 hour
                "medical_history": 7200,   # 2 hours
                "emergency_data": 300      # 5 minutes
            }
        
        if self.healthcare_scaling_config is None:
            self.healthcare_scaling_config = {
                "morning_rounds": {
                    "start": "06:00",
                    "end": "09:00",
                    "scale_factor": 1.5,
                    "priority": "high"
                },
                "afternoon_rounds": {
                    "start": "14:00",
                    "end": "16:00",
                    "scale_factor": 1.3,
                    "priority": "medium"
                },
                "evening_transitions": {
                    "start": "16:00",
                    "end": "20:00",
                    "scale_factor": 1.1,
                    "priority": "medium"
                },
                "night_hours": {
                    "start": "22:00",
                    "end": "06:00",
                    "scale_factor": 0.7,
                    "priority": "low"
                },
                "emergency_periods": {
                    "scale_factor": 2.0,
                    "priority": "critical",
                    "auto_scale_up": True
                }
            }
        
        if self.load_test_config is None:
            self.load_test_config = {
                "concurrent_users": {
                    "light_load": 10,
                    "normal_load": 50,
                    "heavy_load": 100,
                    "stress_test": 500,
                    "spike_test": 200
                },
                "duration_minutes": {
                    "load_test": 30,
                    "stress_test": 15,
                    "spike_test": 5,
                    "endurance_test": 120,
                    "volume_test": 60
                },
                "medical_scenarios": [
                    "patient_lookup",
                    "clinical_data_retrieval",
                    "ai_inference",
                    "medical_history_access",
                    "vital_signs_monitoring",
                    "medication_management",
                    "appointment_scheduling",
                    "emergency_response"
                ]
            }
        
        if self.frontend_config is None:
            self.frontend_config = {
                "bundle_size_limit": 500,  # KB
                "code_splitting_enabled": True,
                "lazy_loading_enabled": True,
                "performance_targets": {
                    "first_paint": 1500,    # ms
                    "interactive": 3000,    # ms
                    "lcp": 2500,           # ms
                    "fid": 100,            # ms
                    "cls": 0.1              # score
                },
                "optimization_strategies": [
                    "code_splitting",
                    "lazy_loading",
                    "tree_shaking",
                    "compression",
                    "caching",
                    "preloading"
                ]
            }
        
        if self.resource_limits is None:
            self.resource_limits = {
                "cpu": {
                    "request": "500m",
                    "limit": "2000m"
                },
                "memory": {
                    "request": "1Gi",
                    "limit": "4Gi"
                },
                "storage": {
                    "request": "10Gi",
                    "limit": "100Gi"
                },
                "connections": {
                    "max_per_service": 1000,
                    "pool_timeout": 30
                }
            }
        
        if self.performance_targets is None:
            self.performance_targets = {
                "response_time_p95": 2.0,     # seconds
                "response_time_p99": 3.0,     # seconds
                "response_time_avg": 1.0,     # seconds
                "cached_response_time": 0.2,  # seconds
                "throughput_min": 100,        # requests/second
                "throughput_peak": 500,       # requests/second
                "cache_hit_rate": 0.85,       # 85%
                "cpu_utilization": 0.70,      # 70%
                "memory_utilization": 0.80,   # 80%
                "error_rate": 0.01,           # 1%
                "availability": 0.999         # 99.9%
            }
        
        if self.monitoring_config is None:
            self.monitoring_config = {
                "metrics_collection_interval": 15,  # seconds
                "alert_thresholds": {
                    "response_time": 2.0,
                    "error_rate": 0.05,
                    "cpu_usage": 0.80,
                    "memory_usage": 0.85,
                    "cache_hit_rate": 0.80,
                    "connection_pool": 0.90
                },
                "dashboard_refresh_interval": 30,  # seconds
                "regression_detection": {
                    "enabled": True,
                    "significance_threshold": 0.05,
                    "performance_degradation_threshold": 0.10,
                    "baseline_window": 7  # days
                },
                "alerting": {
                    "enabled": True,
                    "channels": ["email", "slack", "pagerduty"],
                    "escalation_levels": 3
                }
            }

# Global configuration instance
config = ProductionPerformanceConfig()

# Utility functions
def get_config() -> ProductionPerformanceConfig:
    """Get the global configuration instance"""
    return config

def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values"""
    global config
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)

def get_performance_target(metric: str) -> float:
    """Get performance target for a specific metric"""
    return config.performance_targets.get(metric, 0.0)

def is_production_environment() -> bool:
    """Check if running in production environment"""
    return config.environment == EnvironmentType.PRODUCTION

def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return {
        "url": config.database_url,
        "max_connections": config.max_db_connections,
        "min_connections": config.min_db_connections,
        "timeout": config.db_connection_timeout,
        "pool_recycle": config.db_pool_recycle
    }

def get_redis_config() -> Dict[str, Any]:
    """Get Redis configuration"""
    return {
        "url": config.redis_url,
        "max_connections": config.redis_max_connections,
        "connection_timeout": config.redis_connection_timeout
    }

def get_scaling_config() -> Dict[str, Any]:
    """Get auto-scaling configuration"""
    return {
        "min_replicas": config.min_replicas,
        "max_replicas": config.max_replicas,
        "cpu_target_percentage": config.cpu_target_percentage,
        "memory_target_percentage": config.memory_target_percentage,
        "scale_up_stabilization": config.scale_up_stabilization,
        "scale_down_stabilization": config.scale_down_stabilization,
        "healthcare_patterns": config.healthcare_scaling_config
    }

def get_monitoring_config() -> Dict[str, Any]:
    """Get monitoring configuration"""
    return config.monitoring_config