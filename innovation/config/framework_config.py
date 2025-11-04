"""
Continuous Innovation Framework Configuration
Central configuration management for all innovation systems
"""

import os
from typing import Dict, Any

class InnovationFrameworkConfig:
    """Configuration for the continuous innovation framework"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self._load_config()
    
    def _load_config(self):
        """Load configuration based on environment"""
        if self.environment == "production":
            self._load_production_config()
        elif self.environment == "staging":
            self._load_staging_config()
        else:
            self._load_development_config()
    
    def _load_development_config(self):
        """Development environment configuration"""
        self.config = {
            # Framework Core
            "framework": {
                "name": "Continuous Innovation Framework",
                "version": "1.0.0",
                "environment": self.environment,
                "debug": True,
                "log_level": "INFO"
            },
            
            # AI Feature Engine Configuration
            "ai_feature": {
                "enabled": True,
                "models": {
                    "feature_generation": "gpt-4",
                    "code_generation": "code-t5",
                    "testing_generation": "test-gpt",
                    "quality_assessment": "quality-ai"
                },
                "automation_level": "semi_automated",
                "batch_size": 5,
                "confidence_threshold": 0.8
            },
            
            # Customer Feedback Integration
            "feedback": {
                "enabled": True,
                "sources": {
                    "survey": True,
                    "support_ticket": True,
                    "user_interview": True,
                    "app_store_review": True,
                    "social_media": True,
                    "usage_analytics": True,
                    "chat_support": True,
                    "email_feedback": True
                },
                "processing_frequency": "daily",
                "sentiment_analysis": True,
                "trend_analysis": True,
                "nlp_processing": True
            },
            
            # Rapid Prototyping Engine
            "prototyping": {
                "enabled": True,
                "agile_framework": "scrum",
                "sprint_duration": "2 weeks",
                "devops": {
                    "platform": "kubernetes",
                    "ci_cd": "github_actions",
                    "containerization": "docker"
                },
                "testing": {
                    "framework": "pytest",
                    "coverage_threshold": 80,
                    "test_automation": True
                },
                "deployment_environments": ["development", "staging", "production"]
            },
            
            # Competitive Analysis Engine
            "competitive": {
                "enabled": True,
                "analysis_frequency": "weekly",
                "depth": "comprehensive",
                "sources": [
                    "web_scanning",
                    "social_media",
                    "news",
                    "patent_databases",
                    "market_reports"
                ],
                "gap_analysis": {
                    "sensitivity_threshold": 0.6,
                    "opportunity_scoring": True,
                    "trend_prediction": True
                }
            },
            
            # Product Roadmap Optimizer
            "roadmap": {
                "enabled": True,
                "optimization": {
                    "algorithm": "genetic_algorithm",
                    "objective": "balanced_approach",
                    "iterations": 1000,
                    "population_size": 100
                },
                "strategic_planning": {
                    "time_horizon": "3 years",
                    "goal_alignment": True,
                    "resource_optimization": True
                },
                "resource_constraints": {
                    "developers": {"total": 120, "cost_per_unit": 15000},
                    "qa": {"total": 40, "cost_per_unit": 12000},
                    "designers": {"total": 30, "cost_per_unit": 13000},
                    "infrastructure": {"total": 100000, "cost_per_unit": 1.0}
                }
            },
            
            # Innovation Labs
            "innovation_lab": {
                "enabled": True,
                "labs": [
                    "ai_research",
                    "digital_health",
                    "biomedical_engineering",
                    "quantum_computing",
                    "neurotechnology",
                    "robotics",
                    "iot_sensors",
                    "blockchain",
                    "edge_computing",
                    "augmented_reality"
                ],
                "research_budget": 10000000,
                "max_projects": 25,
                "collaboration": {
                    "universities": True,
                    "research_institutes": True,
                    "industry_partners": True,
                    "startups": True
                }
            },
            
            # Database Configuration
            "database": {
                "type": "postgresql",
                "host": os.getenv("DB_HOST", "localhost"),
                "port": int(os.getenv("DB_PORT", "5432")),
                "name": os.getenv("DB_NAME", "innovation_framework"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "password")
            },
            
            # Cache Configuration
            "cache": {
                "type": "redis",
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "ttl": 3600
            },
            
            # Message Queue Configuration
            "message_queue": {
                "type": "rabbitmq",
                "host": os.getenv("RABBITMQ_HOST", "localhost"),
                "port": int(os.getenv("RABBITMQ_PORT", "5672")),
                "exchange": "innovation_framework"
            },
            
            # Monitoring and Analytics
            "monitoring": {
                "enabled": True,
                "metrics_collection": True,
                "alerting": True,
                "dashboard": True,
                "retention_period": "90 days"
            },
            
            # Security Configuration
            "security": {
                "api_key_encryption": True,
                "access_control": "rbac",
                "audit_logging": True,
                "data_encryption": True
            },
            
            # Integration Configuration
            "integrations": {
                "ehr_systems": {
                    "epic": True,
                    "cerner": True,
                    "allscripts": True
                },
                "cloud_providers": {
                    "aws": True,
                    "azure": True,
                    "gcp": True
                },
                "development_tools": {
                    "github": True,
                    "gitlab": True,
                    "jira": True,
                    "slack": True
                }
            },
            
            # Performance Configuration
            "performance": {
                "max_concurrent_processes": 10,
                "timeout_seconds": 300,
                "retry_attempts": 3,
                "batch_processing": True
            }
        }
    
    def _load_staging_config(self):
        """Staging environment configuration"""
        self._load_development_config()
        # Override with staging-specific settings
        self.config["framework"]["debug"] = False
        self.config["monitoring"]["alerting"] = True
        self.config["security"]["audit_logging"] = True
    
    def _load_production_config(self):
        """Production environment configuration"""
        self._load_development_config()
        # Override with production-specific settings
        self.config["framework"]["debug"] = False
        self.config["framework"]["log_level"] = "WARNING"
        
        # Enhanced monitoring in production
        self.config["monitoring"]["enabled"] = True
        self.config["monitoring"]["metrics_collection"] = True
        self.config["monitoring"]["alerting"] = True
        self.config["monitoring"]["dashboard"] = True
        self.config["monitoring"]["retention_period"] = "365 days"
        
        # Enhanced security in production
        self.config["security"]["api_key_encryption"] = True
        self.config["security"]["access_control"] = "rbac"
        self.config["security"]["audit_logging"] = True
        self.config["security"]["data_encryption"] = True
        
        # Resource constraints for production
        self.config["performance"]["max_concurrent_processes"] = 20
        self.config["performance"]["timeout_seconds"] = 600
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self.get("database")
    
    def get_ai_feature_config(self) -> Dict[str, Any]:
        """Get AI feature engine configuration"""
        return self.get("ai_feature")
    
    def get_feedback_config(self) -> Dict[str, Any]:
        """Get feedback integration configuration"""
        return self.get("feedback")
    
    def get_prototyping_config(self) -> Dict[str, Any]:
        """Get prototyping configuration"""
        return self.get("prototyping")
    
    def get_competitive_config(self) -> Dict[str, Any]:
        """Get competitive analysis configuration"""
        return self.get("competitive")
    
    def get_roadmap_config(self) -> Dict[str, Any]:
        """Get roadmap optimizer configuration"""
        return self.get("roadmap")
    
    def get_innovation_lab_config(self) -> Dict[str, Any]:
        """Get innovation lab configuration"""
        return self.get("innovation_lab")
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        return self.get(f"{feature_name}.enabled", False)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration"""
        return self.config.copy()
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        required_sections = [
            "framework",
            "ai_feature",
            "feedback", 
            "prototyping",
            "competitive",
            "roadmap",
            "innovation_lab"
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return True
    
    def export_config(self, filepath: str):
        """Export configuration to file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get required environment variables"""
        return {
            "DB_HOST": "Database host",
            "DB_PORT": "Database port", 
            "DB_NAME": "Database name",
            "DB_USER": "Database user",
            "DB_PASSWORD": "Database password",
            "REDIS_HOST": "Redis host",
            "REDIS_PORT": "Redis port",
            "RABBITMQ_HOST": "RabbitMQ host",
            "RABBITMQ_PORT": "RabbitMQ port"
        }

# Environment-specific configurations
DEVELOPMENT_CONFIG = InnovationFrameworkConfig("development")
STAGING_CONFIG = InnovationFrameworkConfig("staging") 
PRODUCTION_CONFIG = InnovationFrameworkConfig("production")

def get_config(environment: str = None) -> InnovationFrameworkConfig:
    """Get configuration for specified environment"""
    if environment is None:
        environment = os.getenv("INNOVATION_ENV", "development")
    
    if environment.lower() == "production":
        return PRODUCTION_CONFIG
    elif environment.lower() == "staging":
        return STAGING_CONFIG
    else:
        return DEVELOPMENT_CONFIG

# Configuration presets for different use cases
PRESET_CONFIGS = {
    "healthcare_ai": {
        "ai_feature": {"models": {"feature_generation": "healthcare-gpt"}},
        "innovation_lab": {"labs": ["ai_research", "digital_health", "biomedical_engineering"]},
        "integrations": {"ehr_systems": {"epic": True, "cerner": True}}
    },
    
    "rapid_prototyping": {
        "prototyping": {"sprint_duration": "1 week", "automation_level": "full"},
        "ai_feature": {"automation_level": "fully_automated"},
        "performance": {"max_concurrent_processes": 20}
    },
    
    "enterprise_scale": {
        "innovation_lab": {"research_budget": 50000000, "max_projects": 100},
        "monitoring": {"retention_period": "2 years"},
        "security": {"access_control": "zero_trust"}
    }
}

def apply_preset_config(config: InnovationFrameworkConfig, preset_name: str):
    """Apply configuration preset"""
    if preset_name in PRESET_CONFIGS:
        preset = PRESET_CONFIGS[preset_name]
        for key, value in preset.items():
            config.set(key, value)
    else:
        raise ValueError(f"Unknown preset: {preset_name}")

if __name__ == "__main__":
    # Example usage
    config = get_config("development")
    print(f"Environment: {config.get('framework.environment')}")
    print(f"AI Feature Enabled: {config.is_feature_enabled('ai_feature')}")
    print(f"Database Config: {config.get_database_config()}")
    
    # Validate configuration
    if config.validate_config():
        print("Configuration is valid")