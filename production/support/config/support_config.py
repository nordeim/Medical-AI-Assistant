"""
Production Support System Configuration
Healthcare-focused customer support configuration settings
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class PriorityLevel(Enum):
    CRITICAL_MEDICAL = "critical_medical"
    HIGH_MEDICAL = "high_medical"
    STANDARD_MEDICAL = "standard_medical"
    ADMINISTRATIVE = "administrative"
    EMERGENCY = "emergency"

class SupportTier(Enum):
    CRITICAL = "critical"
    STANDARD = "standard"
    ADMINISTRATIVE = "administrative"
    TRAINING = "training"

@dataclass
class MedicalSLAResponse:
    """SLA requirements for medical support cases"""
    priority: PriorityLevel
    response_time_hours: int
    resolution_time_hours: int
    escalation_threshold: int
    medical_specialist_required: bool
    emergency_notification: bool

@dataclass
class HealthcareConfig:
    """Healthcare-specific configuration settings"""
    hipaa_compliance: bool = True
    fda_regulatory_mode: bool = True
    medical_device_integration: bool = True
    ehr_system_integration: bool = True
    clinical_workflow_support: bool = True
    patient_safety_priority: bool = True

class SupportConfig:
    """Main configuration class for support systems"""
    
    # Healthcare Organization Settings
    HEALTHCARE_CONFIG = HealthcareConfig()
    
    # Medical SLA Requirements
    MEDICAL_SLA_REQUIREMENTS = {
        PriorityLevel.EMERGENCY: MedicalSLAResponse(
            priority=PriorityLevel.EMERGENCY,
            response_time_hours=0,
            resolution_time_hours=1,
            escalation_threshold=15,
            medical_specialist_required=True,
            emergency_notification=True
        ),
        PriorityLevel.CRITICAL_MEDICAL: MedicalSLAResponse(
            priority=PriorityLevel.CRITICAL_MEDICAL,
            response_time_hours=1,
            resolution_time_hours=4,
            escalation_threshold=30,
            medical_specialist_required=True,
            emergency_notification=True
        ),
        PriorityLevel.HIGH_MEDICAL: MedicalSLAResponse(
            priority=PriorityLevel.HIGH_MEDICAL,
            response_time_hours=2,
            resolution_time_hours=12,
            escalation_threshold=60,
            medical_specialist_required=True,
            emergency_notification=False
        ),
        PriorityLevel.STANDARD_MEDICAL: MedicalSLAResponse(
            priority=PriorityLevel.STANDARD_MEDICAL,
            response_time_hours=8,
            resolution_time_hours=48,
            escalation_threshold=120,
            medical_specialist_required=False,
            emergency_notification=False
        ),
        PriorityLevel.ADMINISTRATIVE: MedicalSLAResponse(
            priority=PriorityLevel.ADMINISTRATIVE,
            response_time_hours=24,
            resolution_time_hours=72,
            escalation_threshold=240,
            medical_specialist_required=False,
            emergency_notification=False
        )
    }
    
    # Support Tier Configuration
    SUPPORT_TIERS = {
        SupportTier.CRITICAL: {
            "name": "Critical Medical Support",
            "availability": "24/7",
            "response_team": "medical_specialists",
            "escalation_path": "emergency_medical_team",
            "technology_support": "full_technical_support"
        },
        SupportTier.STANDARD: {
            "name": "Standard Healthcare Support",
            "availability": "business_hours",
            "response_team": "healthcare_support",
            "escalation_path": "medical_specialists",
            "technology_support": "standard_technical_support"
        },
        SupportTier.ADMINISTRATIVE: {
            "name": "Administrative Support",
            "availability": "business_hours",
            "response_team": "admin_support",
            "escalation_path": "healthcare_support",
            "technology_support": "basic_technical_support"
        },
        SupportTier.TRAINING: {
            "name": "Training & Certification Support",
            "availability": "scheduled_sessions",
            "response_team": "training_specialists",
            "escalation_path": "medical_specialists",
            "technology_support": "training_technical_support"
        }
    }
    
    # Feedback Collection Settings
    FEEDBACK_CONFIG = {
        "sentiment_analysis": {
            "enabled": True,
            "medical_context_awareness": True,
            "real_time_processing": True,
            "confidence_threshold": 0.8
        },
        "collection_channels": [
            "post_interaction_survey",
            "periodic_satisfaction_survey",
            "medical_outcome_correlation",
            "user_behavior_analytics",
            "clinical_feedback_sessions"
        ],
        "privacy_settings": {
            "anonymize_feedback": True,
            "patient_consent_required": True,
            "hipaa_compliant_storage": True,
            "audit_trail_maintained": True
        }
    }
    
    # Health Monitoring Configuration
    MONITORING_CONFIG = {
        "uptime_targets": {
            "critical_systems": 99.99,
            "standard_systems": 99.9,
            "administrative_systems": 99.5
        },
        "health_checks": {
            "api_endpoints": {
                "interval_seconds": 30,
                "timeout_seconds": 10,
                "retries": 3
            },
            "database_connections": {
                "interval_seconds": 60,
                "timeout_seconds": 5,
                "retries": 2
            },
            "third_party_integrations": {
                "interval_seconds": 120,
                "timeout_seconds": 15,
                "retries": 3
            }
        },
        "alerting": {
            "critical_thresholds": {
                "cpu_usage": 90,
                "memory_usage": 85,
                "disk_usage": 90,
                "response_time": 5000
            },
            "notification_channels": [
                "email",
                "sms",
                "slack",
                "pagerduty"
            ]
        }
    }
    
    # Knowledge Base Configuration
    KNOWLEDGE_BASE_CONFIG = {
        "medical_categories": [
            "clinical_workflows",
            "patient_care_protocols",
            "medical_terminology",
            "system_administration",
            "compliance_guidelines",
            "emergency_procedures",
            "troubleshooting",
            "best_practices"
        ],
        "content_types": [
            "medical_documentation",
            "video_tutorials",
            "interactive_guides",
            "faq",
            "case_studies",
            "protocols"
        ],
        "search_features": {
            "semantic_search": True,
            "medical_code_search": True,
            "image_search": True,
            "multi_language_support": True
        }
    }
    
    # Training & Certification Configuration
    TRAINING_CONFIG = {
        "certification_tracks": [
            {
                "name": "Healthcare Professional Certification",
                "duration_weeks": 8,
                "modules": [
                    "medical_ai_fundamentals",
                    "clinical_integration",
                    "patient_safety",
                    "regulatory_compliance"
                ],
                "assessment_types": [
                    "practical_simulation",
                    "written_examination",
                    "case_study_analysis"
                ]
            },
            {
                "name": "Administrator Certification",
                "duration_weeks": 4,
                "modules": [
                    "system_administration",
                    "user_management",
                    "compliance_monitoring",
                    "performance_optimization"
                ],
                "assessment_types": [
                    "practical_lab",
                    "scenario_testing",
                    "compliance_audit"
                ]
            }
        ],
        "continuing_education": {
            "credit_hours_required": 20,
            "renewal_period_months": 12,
            " accredited_providers": True,
            "competency_assessments": True
        }
    }
    
    # Integration Configuration
    INTEGRATION_CONFIG = {
        "ehr_systems": [
            "epic",
            "cerner",
            "allscripts",
            "athenahealth",
            "nextgen"
        ],
        "hospital_systems": [
            "meditech",
            "sunquest",
            "siemens",
            "ge_healthcare",
            "philips"
        ],
        "communication_platforms": [
            "slack",
            "microsoft_teams",
            "zoom",
            "phone_systems",
            "secure_messaging"
        ]
    }
    
    # Security & Compliance Configuration
    SECURITY_CONFIG = {
        "encryption": {
            "data_at_rest": "AES-256",
            "data_in_transit": "TLS 1.3",
            "key_management": "HSM",
            "rotation_policy": "quarterly"
        },
        "access_control": {
            "rbac_enabled": True,
            "mfa_required": True,
            "session_timeout_minutes": 30,
            "password_policy": "strict"
        },
        "audit_logging": {
            "log_retention_days": 2555,  # 7 years for healthcare
            "immutable_logs": True,
            "real_time_monitoring": True,
            "compliance_reporting": True
        }
    }
    
    @classmethod
    def get_sla_for_priority(cls, priority: PriorityLevel) -> MedicalSLAResponse:
        """Get SLA requirements for a specific priority level"""
        return cls.MEDICAL_SLA_REQUIREMENTS.get(priority)
    
    @classmethod
    def get_support_tier_config(cls, tier: SupportTier) -> Dict[str, Any]:
        """Get configuration for a specific support tier"""
        return cls.SUPPORT_TIERS.get(tier)
    
    @classmethod
    def is_medical_emergency(cls, priority: PriorityLevel) -> bool:
        """Check if a priority level represents a medical emergency"""
        return priority in [PriorityLevel.EMERGENCY, PriorityLevel.CRITICAL_MEDICAL]
    
    @classmethod
    def requires_medical_specialist(cls, priority: PriorityLevel) -> bool:
        """Check if a priority level requires medical specialist involvement"""
        sla = cls.get_sla_for_priority(priority)
        return sla.medical_specialist_required if sla else False

# Environment-specific configurations
DEVELOPMENT_CONFIG = {
    "debug_mode": True,
    "log_level": "DEBUG",
    "database_url": os.getenv("DEV_DATABASE_URL", "sqlite:///support_dev.db"),
    "cache_enabled": False
}

STAGING_CONFIG = {
    "debug_mode": True,
    "log_level": "INFO",
    "database_url": os.getenv("STAGING_DATABASE_URL", "postgresql://staging_user:password@localhost/support_staging"),
    "cache_enabled": True
}

PRODUCTION_CONFIG = {
    "debug_mode": False,
    "log_level": "WARNING",
    "database_url": os.getenv("PRODUCTION_DATABASE_URL"),
    "cache_enabled": True,
    "load_balancer_health_checks": True,
    "auto_scaling_enabled": True
}

# Current environment configuration
CURRENT_CONFIG = PRODUCTION_CONFIG

def get_config() -> Dict[str, Any]:
    """Get current environment configuration"""
    return CURRENT_CONFIG

def get_medical_sla_requirements() -> Dict[PriorityLevel, MedicalSLAResponse]:
    """Get all medical SLA requirements"""
    return SupportConfig.MEDICAL_SLA_REQUIREMENTS

def get_healthcare_config() -> HealthcareConfig:
    """Get healthcare-specific configuration"""
    return SupportConfig.HEALTHCARE_CONFIG