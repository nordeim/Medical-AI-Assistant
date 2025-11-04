"""
Healthcare Customer Onboarding Framework - Configuration Management
Centralized configuration for all framework components
"""

import json
from typing import Dict, Any, List
from datetime import datetime

class HealthcareOnboardingConfig:
    """Configuration manager for the healthcare onboarding framework"""
    
    def __init__(self):
        self.config = self._load_default_configuration()
    
    def _load_default_configuration(self) -> Dict[str, Any]:
        """Load default framework configuration"""
        return {
            "framework": {
                "name": "Healthcare Customer Onboarding Automation Framework",
                "version": "1.0.0",
                "environment": "production",
                "last_updated": datetime.now().isoformat(),
                "supported_healthcare_types": [
                    "hospital",
                    "clinic", 
                    "health_system",
                    "specialty_center",
                    "research_institution"
                ]
            },
            
            "onboarding_workflows": {
                "default_timeline_multipliers": {
                    "small": 0.8,
                    "medium": 1.0,
                    "large": 1.3,
                    "enterprise": 1.6
                },
                
                "milestone_definitions": {
                    "pre_assessment": {
                        "duration_days": 14,
                        "critical": True,
                        "dependencies": []
                    },
                    "compliance_validation": {
                        "duration_days": 21,
                        "critical": True,
                        "dependencies": ["pre_assessment"]
                    },
                    "technical_integration": {
                        "duration_days": 28,
                        "critical": True,
                        "dependencies": ["compliance_validation"]
                    },
                    "clinical_workflow_integration": {
                        "duration_days": 35,
                        "critical": False,
                        "dependencies": ["technical_integration"]
                    },
                    "staff_training": {
                        "duration_days": 42,
                        "critical": True,
                        "dependencies": ["clinical_workflow_integration"]
                    },
                    "pilot_deployment": {
                        "duration_days": 21,
                        "critical": False,
                        "dependencies": ["staff_training"]
                    },
                    "full_deployment": {
                        "duration_days": 14,
                        "critical": True,
                        "dependencies": ["pilot_deployment"]
                    },
                    "optimization": {
                        "duration_days": 28,
                        "critical": False,
                        "dependencies": ["full_deployment"]
                    }
                },
                
                "healthcare_specific_customizations": {
                    "hospital": {
                        "additional_milestones": [
                            "Multi-department coordination",
                            "Emergency department integration",
                            "Surgical services integration"
                        ],
                        "training_hours_multiplier": 1.2
                    },
                    "clinic": {
                        "additional_milestones": [
                            "Outpatient workflow optimization",
                            "Appointment system integration"
                        ],
                        "training_hours_multiplier": 0.9
                    },
                    "health_system": {
                        "additional_milestones": [
                            "Multi-facility standardization",
                            "Enterprise master patient index",
                            "Regional care coordination"
                        ],
                        "training_hours_multiplier": 1.4
                    }
                }
            },
            
            "training_certification": {
                "cme_accreditation": {
                    "provider": "Accreditation Council for Continuing Medical Education",
                    "credit_structure": "AMA PRA Category 1 Credit(s)™",
                    "maintenance_requirements": {
                        "recertification_period_months": 24,
                        "minimum_annual_hours": 8,
                        "competency_assessment_required": True
                    }
                },
                
                "module_definitions": {
                    "FUND_001": {
                        "title": "AI in Healthcare Fundamentals",
                        "duration_hours": 4.0,
                        "cme_credits": 4.0,
                        "target_roles": ["clinician", "physician", "nurse", "administrator"],
                        "prerequisites": []
                    },
                    "CLIN_001": {
                        "title": "Clinical Decision Support Integration",
                        "duration_hours": 6.0,
                        "cme_credits": 6.0,
                        "target_roles": ["clinician", "physician"],
                        "prerequisites": ["FUND_001"]
                    },
                    "COMP_001": {
                        "title": "HIPAA Compliance and Data Security",
                        "duration_hours": 4.0,
                        "cme_credits": 4.0,
                        "target_roles": ["clinician", "physician", "nurse", "administrator"],
                        "prerequisites": []
                    },
                    "TECH_001": {
                        "title": "System Administration and Operations",
                        "duration_hours": 8.0,
                        "cme_credits": 0,
                        "target_roles": ["it_staff", "administrator"],
                        "prerequisites": ["FUND_001"]
                    },
                    "ADV_001": {
                        "title": "Advanced AI Features and Customization",
                        "duration_hours": 5.0,
                        "cme_credits": 5.0,
                        "target_roles": ["clinician", "physician", "administrator"],
                        "prerequisites": ["CLIN_001"]
                    }
                },
                
                "certification_paths": {
                    "clinician_basic": {
                        "required_modules": ["FUND_001"],
                        "total_cme_credits": 4.0,
                        "certification_level": "Basic"
                    },
                    "clinician_intermediate": {
                        "required_modules": ["FUND_001", "CLIN_001", "COMP_001"],
                        "total_cme_credits": 14.0,
                        "certification_level": "Intermediate"
                    },
                    "physician_expert": {
                        "required_modules": ["FUND_001", "CLIN_001", "COMP_001", "ADV_001"],
                        "total_cme_credits": 19.0,
                        "certification_level": "Expert"
                    }
                }
            },
            
            "success_monitoring": {
                "health_metrics": {
                    "adoption_metrics": {
                        "daily_active_users": {
                            "target": 80.0,
                            "warning_threshold": 60.0,
                            "critical_threshold": 40.0,
                            "weight": 0.15
                        },
                        "weekly_active_users": {
                            "target": 90.0,
                            "warning_threshold": 75.0,
                            "critical_threshold": 50.0,
                            "weight": 0.15
                        },
                        "feature_adoption_rate": {
                            "target": 70.0,
                            "warning_threshold": 50.0,
                            "critical_threshold": 30.0,
                            "weight": 0.10
                        }
                    },
                    
                    "engagement_metrics": {
                        "session_duration": {
                            "target": 15.0,
                            "warning_threshold": 10.0,
                            "critical_threshold": 5.0,
                            "weight": 0.08
                        },
                        "clinical_decisions_supported": {
                            "target": 50.0,
                            "warning_threshold": 30.0,
                            "critical_threshold": 20.0,
                            "weight": 0.12
                        },
                        "user_satisfaction_score": {
                            "target": 4.5,
                            "warning_threshold": 3.5,
                            "critical_threshold": 2.5,
                            "weight": 0.15
                        }
                    },
                    
                    "performance_metrics": {
                        "system_uptime": {
                            "target": 99.5,
                            "warning_threshold": 99.0,
                            "critical_threshold": 98.0,
                            "weight": 0.10
                        },
                        "response_time": {
                            "target": 2.0,
                            "warning_threshold": 3.0,
                            "critical_threshold": 5.0,
                            "weight": 0.08
                        },
                        "clinical_accuracy": {
                            "target": 95.0,
                            "warning_threshold": 90.0,
                            "critical_threshold": 85.0,
                            "weight": 0.15
                        }
                    },
                    
                    "business_metrics": {
                        "roi_achievement": {
                            "target": 150.0,
                            "warning_threshold": 100.0,
                            "critical_threshold": 75.0,
                            "weight": 0.20
                        },
                        "time_savings": {
                            "target": 10.0,
                            "warning_threshold": 5.0,
                            "critical_threshold": 2.0,
                            "weight": 0.12
                        }
                    }
                },
                
                "health_scoring": {
                    "categories": {
                        "adoption": 0.30,
                        "engagement": 0.25,
                        "performance": 0.20,
                        "support": 0.10,
                        "business_value": 0.10,
                        "clinical_outcomes": 0.05
                    },
                    
                    "status_thresholds": {
                        "thriving": 85.0,
                        "healthy": 70.0,
                        "needs_attention": 55.0,
                        "at_risk": 40.0,
                        "critical": 0.0
                    }
                }
            },
            
            "proactive_support": {
                "sla_definitions": {
                    "critical": {
                        "first_response_minutes": 5,
                        "resolution_hours": 1,
                        "communication_frequency_minutes": 15,
                        "escalation_minutes": 10,
                        "availability": "24/7",
                        "resource_allocation": "immediate_dedicated"
                    },
                    "high": {
                        "first_response_minutes": 15,
                        "resolution_hours": 4,
                        "communication_frequency_minutes": 60,
                        "escalation_minutes": 60,
                        "availability": "24/7",
                        "resource_allocation": "dedicated_team"
                    },
                    "medium": {
                        "first_response_minutes": 60,
                        "resolution_hours": 8,
                        "communication_frequency_minutes": 240,
                        "escalation_minutes": 120,
                        "availability": "business_hours_extended",
                        "resource_allocation": "shared_team"
                    },
                    "low": {
                        "first_response_minutes": 240,
                        "resolution_hours": 24,
                        "communication_frequency_minutes": 480,
                        "escalation_minutes": 480,
                        "availability": "business_hours",
                        "resource_allocation": "queue_based"
                    }
                },
                
                "emergency_protocols": {
                    "patient_safety_emergency": {
                        "trigger_conditions": [
                            "AI system providing incorrect clinical recommendations",
                            "System failure during critical clinical decision",
                            "Data integrity issue affecting patient care"
                        ],
                        "response_time_minutes": 5,
                        "escalation_chain": [
                            {"role": "Clinical Lead", "response_time_minutes": 5},
                            {"role": "Medical Director", "response_time_minutes": 10},
                            {"role": "Chief Medical Officer", "response_time_minutes": 15}
                        ]
                    },
                    
                    "system_wide_outage": {
                        "trigger_conditions": [
                            "Complete system unavailability",
                            "Data center failure",
                            "Network connectivity loss"
                        ],
                        "response_time_minutes": 10,
                        "escalation_chain": [
                            {"role": "Senior Technical Lead", "response_time_minutes": 10},
                            {"role": "Engineering Director", "response_time_minutes": 15},
                            {"role": "CTO", "response_time_minutes": 20}
                        ]
                    }
                },
                
                "support_team_structure": {
                    "clinical_specialists": {
                        "availability": "24/7 for critical issues",
                        "response_time_critical_minutes": 5,
                        "specializations": ["emergency_medicine", "clinical_workflow"]
                    },
                    "technical_specialists": {
                        "availability": "24/7 for technical issues",
                        "response_time_high_minutes": 15,
                        "specializations": ["ehr_integration", "system_performance"]
                    },
                    "compliance_specialists": {
                        "availability": "24/7 for compliance emergencies",
                        "response_time_critical_minutes": 10,
                        "specializations": ["hipaa_compliance", "security_incident_response"]
                    }
                }
            },
            
            "optimization_engine": {
                "workflow_analysis": {
                    "target_workflows": [
                        "clinical_documentation",
                        "diagnostic_process",
                        "treatment_planning",
                        "clinical_encounter"
                    ],
                    
                    "efficiency_metrics": {
                        "time_savings_percentage": {
                            "target_improvement": 25.0,
                            "benchmark_comparison": "industry_standard"
                        },
                        "accuracy_improvement": {
                            "target_improvement": 15.0,
                            "benchmark_comparison": "clinical_guidelines"
                        },
                        "user_satisfaction": {
                            "target_improvement": 20.0,
                            "benchmark_comparison": "current_satisfaction"
                        }
                    }
                },
                
                "optimization_patterns": {
                    "reduce_documentation_time": {
                        "expected_improvements": {
                            "time_savings": "30-50%",
                            "accuracy_improvement": "15-25%",
                            "satisfaction_boost": "25-35%"
                        }
                    },
                    "accelerate_diagnosis": {
                        "expected_improvements": {
                            "time_to_diagnosis": "20-40%",
                            "accuracy_improvement": "10-20%",
                            "cost_reduction": "15-25%"
                        }
                    }
                }
            },
            
            "advocacy_program": {
                "reference_tiers": {
                    "strategic": {
                        "criteria": [
                            "Large-scale implementation (1000+ users)",
                            "Multi-facility deployment",
                            "High visibility in industry"
                        ],
                        "benefits": [
                            "Exclusive customer advisory board",
                            "Co-marketing opportunities",
                            "Early access to features"
                        ]
                    },
                    "developmental": {
                        "criteria": [
                            "Early adopter of new features",
                            "Innovation-focused implementation",
                            "Collaborative development approach"
                        ],
                        "benefits": [
                            "Beta feature access",
                            "Product roadmap influence",
                            "Direct development team access"
                        ]
                    },
                    "standard": {
                        "criteria": [
                            "Successful implementation",
                            "Good adoption rates",
                            "Positive customer satisfaction"
                        ],
                        "benefits": [
                            "Reference call participation",
                            "Case study opportunity",
                            "Community participation"
                        ]
                    }
                },
                
                "success_story_development": {
                    "story_types": [
                        "clinical_efficiency_transformation",
                        "implementation_success",
                        "clinical_innovation_adoption",
                        "patient_experience_enhancement"
                    ],
                    "development_timeline_months": 6,
                    "approval_process": ["customer_review", "legal_review", "executive_approval"]
                }
            },
            
            "compliance_framework": {
                "regulatory_requirements": {
                    "hipaa": {
                        "title": "Health Insurance Portability and Accountability Act",
                        "applicable_sections": ["Privacy Rule", "Security Rule", "Breach Notification Rule"],
                        "compliance_monitoring": "continuous"
                    },
                    "hitech": {
                        "title": "Health Information Technology for Economic and Clinical Health Act",
                        "applicable_sections": ["Meaningful Use", "Breach Notification"],
                        "compliance_monitoring": "continuous"
                    },
                    "joint_commission": {
                        "title": "Joint Commission Standards",
                        "applicable_sections": ["Patient Safety", "Information Management"],
                        "compliance_monitoring": "annual_assessment"
                    },
                    "fda_regulations": {
                        "title": "FDA 21 CFR Part 11",
                        "applicable_sections": ["Electronic Records", "Electronic Signatures"],
                        "compliance_monitoring": "implementation_validation"
                    }
                },
                
                "audit_capabilities": {
                    "audit_trail": "complete_interaction_tracking",
                    "compliance_reporting": "automated_generation",
                    "risk_assessment": "continuous_monitoring",
                    "regulatory_change_management": "automated_updates"
                }
            }
        }
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """Get configuration section"""
        if section is None:
            return self.config
        
        return self.config.get(section, {})
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section"""
        if section in self.config:
            self.config[section].update(updates)
        else:
            self.config[section] = updates
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, filepath: str) -> None:
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            self.config = json.load(f)

def create_default_config_file(filepath: str) -> None:
    """Create default configuration file"""
    config_manager = HealthcareOnboardingConfig()
    config_manager.save_config(filepath)
    print(f"Default configuration saved to: {filepath}")

if __name__ == "__main__":
    # Create default configuration
    create_default_config_file("healthcare_onboarding_config.json")
    print("Configuration file created successfully")