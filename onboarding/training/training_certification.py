"""
Healthcare Staff Training and Certification Programs
CME-accredited training modules with compliance requirements
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class UserRole(Enum):
    """Healthcare user roles"""
    CLINICIAN = "clinician"
    NURSE = "nurse"
    PHYSICIAN = "physician"
    TECHNICIAN = "technician"
    ADMINISTRATOR = "administrator"
    IT_STAFF = "it_staff"
    COMPLIANCE_OFFICER = "compliance_officer"

class TrainingModuleType(Enum):
    """Types of training modules"""
    FUNDAMENTALS = "fundamentals"
    CLINICAL_APPLICATION = "clinical_application"
    TECHNICAL_OPERATION = "technical_operation"
    COMPLIANCE_SECURITY = "compliance_security"
    ADVANCED_FEATURES = "advanced_features"
    TROUBLESHOOTING = "troubleshooting"

class CertificationLevel(Enum):
    """Certification levels"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"
    TRAINER = "trainer"

@dataclass
class TrainingModule:
    """Individual training module"""
    module_id: str
    title: str
    description: str
    module_type: TrainingModuleType
    target_roles: List[UserRole]
    duration_hours: float
    learning_objectives: List[str]
    content_sections: List[Dict[str, Any]]
    assessments: List[Dict[str, Any]]
    cme_credits: float
    prerequisites: List[str]
    certification_levels: List[CertificationLevel]

@dataclass
class UserProgress:
    """User training progress tracking"""
    user_id: str
    role: UserRole
    organization_id: str
    modules_completed: List[str]
    modules_in_progress: List[str]
    assessment_scores: Dict[str, float]
    certification_level: CertificationLevel
    cme_credits_earned: float
    last_activity_date: str
    competency_assessment_date: str

class HealthcareTrainingManager:
    """Training and certification management system for healthcare staff"""
    
    def __init__(self):
        self.training_modules = self._initialize_training_modules()
        self.certification_paths = self._initialize_certification_paths()
        self.assessment_bank = self._initialize_assessment_bank()
    
    def _initialize_training_modules(self) -> Dict[str, TrainingModule]:
        """Initialize comprehensive training modules"""
        return {
            # Fundamentals Module
            "FUND_001": TrainingModule(
                module_id="FUND_001",
                title="AI in Healthcare Fundamentals",
                description="Introduction to artificial intelligence applications in healthcare",
                module_type=TrainingModuleType.FUNDAMENTALS,
                target_roles=[UserRole.CLINICIAN, UserRole.NURSE, UserRole.PHYSICIAN, 
                             UserRole.ADMINISTRATOR, UserRole.IT_STAFF],
                duration_hours=4.0,
                learning_objectives=[
                    "Understand AI concepts and applications in healthcare",
                    "Recognize potential benefits and limitations of AI systems",
                    "Identify appropriate use cases for AI-assisted clinical decision making",
                    "Develop trust and confidence in AI recommendations"
                ],
                content_sections=[
                    {
                        "title": "AI Technology Overview",
                        "duration_hours": 1.5,
                        "content_type": "video_presentation",
                        "key_points": [
                            "Machine learning basics",
                            "Natural language processing",
                            "Clinical decision support systems",
                            "AI vs. traditional software"
                        ]
                    },
                    {
                        "title": "Healthcare AI Applications",
                        "duration_hours": 1.5,
                        "content_type": "case_studies",
                        "key_points": [
                            "Diagnostic assistance",
                            "Treatment recommendations",
                            "Workflow optimization",
                            "Patient monitoring"
                        ]
                    },
                    {
                        "title": "Benefits and Limitations",
                        "duration_hours": 1.0,
                        "content_type": "interactive_content",
                        "key_points": [
                            "Accuracy and reliability",
                            "Human-AI collaboration",
                            "Bias and fairness considerations",
                            "Continuous learning"
                        ]
                    }
                ],
                assessments=[
                    {
                        "assessment_type": "knowledge_check",
                        "questions": 20,
                        "passing_score": 80,
                        "time_limit_minutes": 30
                    },
                    {
                        "assessment_type": "scenario_analysis",
                        "scenarios": 5,
                        "passing_score": 85,
                        "time_limit_minutes": 45
                    }
                ],
                cme_credits=4.0,
                prerequisites=[],
                certification_levels=[CertificationLevel.BASIC]
            ),
            
            # Clinical Application Module
            "CLIN_001": TrainingModule(
                module_id="CLIN_001",
                title="Clinical Decision Support Integration",
                description="Integration of AI systems with clinical workflows",
                module_type=TrainingModuleType.CLINICAL_APPLICATION,
                target_roles=[UserRole.CLINICIAN, UserRole.PHYSICIAN, UserRole.NURSE],
                duration_hours=6.0,
                learning_objectives=[
                    "Integrate AI recommendations into clinical decision making",
                    "Validate AI suggestions against clinical knowledge",
                    "Document AI-assisted decisions appropriately",
                    "Maintain clinical accountability while using AI tools"
                ],
                content_sections=[
                    {
                        "title": "Clinical Workflow Integration",
                        "duration_hours": 2.0,
                        "content_type": "workflow_demonstration",
                        "key_points": [
                            "EHR integration points",
                            "Clinical documentation workflows",
                            "Order entry processes",
                            "Clinical documentation best practices"
                        ]
                    },
                    {
                        "title": "AI Recommendation Validation",
                        "duration_hours": 2.0,
                        "content_type": "hands_on_practice",
                        "key_points": [
                            "Critical evaluation of AI suggestions",
                            "Correlation with clinical guidelines",
                            "Patient-specific considerations",
                            "When to trust vs. override AI recommendations"
                        ]
                    },
                    {
                        "title": "Clinical Accountability",
                        "duration_hours": 2.0,
                        "content_type": "case_based_learning",
                        "key_points": [
                            "Legal and ethical considerations",
                            "Documentation requirements",
                            "Patient communication about AI use",
                            "Medical record maintenance"
                        ]
                    }
                ],
                assessments=[
                    {
                        "assessment_type": "clinical_scenarios",
                        "scenarios": 10,
                        "passing_score": 90,
                        "time_limit_minutes": 90
                    },
                    {
                        "assessment_type": "competency_demonstration",
                        "tasks": 5,
                        "passing_score": 85,
                        "time_limit_minutes": 60
                    }
                ],
                cme_credits=6.0,
                prerequisites=["FUND_001"],
                certification_levels=[CertificationLevel.INTERMEDIATE, CertificationLevel.ADVANCED]
            ),
            
            # Compliance and Security Module
            "COMP_001": TrainingModule(
                module_id="COMP_001",
                title="HIPAA Compliance and Data Security",
                description="HIPAA compliance requirements and data security best practices",
                module_type=TrainingModuleType.COMPLIANCE_SECURITY,
                target_roles=[UserRole.CLINICIAN, UserRole.NURSE, UserRole.ADMINISTRATOR, 
                             UserRole.COMPLIANCE_OFFICER, UserRole.IT_STAFF],
                duration_hours=4.0,
                learning_objectives=[
                    "Understand HIPAA Privacy and Security Rules",
                    "Implement proper data handling procedures",
                    "Recognize and respond to security incidents",
                    "Maintain compliance in AI-assisted workflows"
                ],
                content_sections=[
                    {
                        "title": "HIPAA Privacy Rule",
                        "duration_hours": 1.5,
                        "content_type": "regulatory_content",
                        "key_points": [
                            "Protected health information (PHI) definitions",
                            "Minimum necessary standard",
                            "Patient rights and consent",
                            "Breach notification requirements"
                        ]
                    },
                    {
                        "title": "HIPAA Security Rule",
                        "duration_hours": 1.5,
                        "content_type": "technical_content",
                        "key_points": [
                            "Administrative safeguards",
                            "Physical safeguards",
                            "Technical safeguards",
                            "Audit controls and monitoring"
                        ]
                    },
                    {
                        "title": "AI-Specific Security Considerations",
                        "duration_hours": 1.0,
                        "content_type": "security_focus",
                        "key_points": [
                            "AI model security",
                            "Data encryption in AI workflows",
                            "Access controls for AI systems",
                            "Audit trails for AI interactions"
                        ]
                    }
                ],
                assessments=[
                    {
                        "assessment_type": "compliance_quiz",
                        "questions": 25,
                        "passing_score": 95,
                        "time_limit_minutes": 45
                    },
                    {
                        "assessment_type": "security_scenarios",
                        "scenarios": 8,
                        "passing_score": 90,
                        "time_limit_minutes": 60
                    }
                ],
                cme_credits=4.0,
                prerequisites=[],
                certification_levels=[CertificationLevel.BASIC, CertificationLevel.INTERMEDIATE]
            ),
            
            # Technical Operations Module
            "TECH_001": TrainingModule(
                module_id="TECH_001",
                title="System Administration and Operations",
                description="Technical administration and system operations for IT staff",
                module_type=TrainingModuleType.TECHNICAL_OPERATION,
                target_roles=[UserRole.IT_STAFF, UserRole.ADMINISTRATOR],
                duration_hours=8.0,
                learning_objectives=[
                    "Configure and maintain AI system infrastructure",
                    "Monitor system performance and health",
                    "Manage user access and permissions",
                    "Troubleshoot common technical issues"
                ],
                content_sections=[
                    {
                        "title": "System Configuration",
                        "duration_hours": 2.5,
                        "content_type": "hands_on_lab",
                        "key_points": [
                            "Installation and setup procedures",
                            "Configuration management",
                            "Integration with existing systems",
                            "Performance tuning"
                        ]
                    },
                    {
                        "title": "User Management",
                        "duration_hours": 2.0,
                        "content_type": "administration_tools",
                        "key_points": [
                            "User account management",
                            "Role-based access control",
                            "Permission management",
                            "Audit trail configuration"
                        ]
                    },
                    {
                        "title": "Monitoring and Maintenance",
                        "duration_hours": 2.0,
                        "content_type": "monitoring_dashboards",
                        "key_points": [
                            "System health monitoring",
                            "Performance metrics",
                            "Alert configuration",
                            "Maintenance procedures"
                        ]
                    },
                    {
                        "title": "Troubleshooting",
                        "duration_hours": 1.5,
                        "content_type": "problem_solving",
                        "key_points": [
                            "Common issues and solutions",
                            "Log analysis",
                            "Diagnostic procedures",
                            "Escalation procedures"
                        ]
                    }
                ],
                assessments=[
                    {
                        "assessment_type": "technical_demonstration",
                        "tasks": 12,
                        "passing_score": 90,
                        "time_limit_minutes": 120
                    },
                    {
                        "assessment_type": "troubleshooting_scenarios",
                        "scenarios": 8,
                        "passing_score": 85,
                        "time_limit_minutes": 90
                    }
                ],
                cme_credits=0,  # Technical training doesn't qualify for CME
                prerequisites=["FUND_001"],
                certification_levels=[CertificationLevel.INTERMEDIATE, CertificationLevel.ADVANCED]
            ),
            
            # Advanced Features Module
            "ADV_001": TrainingModule(
                module_id="ADV_001",
                title="Advanced AI Features and Customization",
                description="Advanced features and customization options",
                module_type=TrainingModuleType.ADVANCED_FEATURES,
                target_roles=[UserRole.CLINICIAN, UserRole.PHYSICIAN, UserRole.ADMINISTRATOR],
                duration_hours=5.0,
                learning_objectives=[
                    "Utilize advanced AI features for complex cases",
                    "Customize AI parameters for specific specialties",
                    "Analyze AI performance metrics",
                    "Optimize AI workflow for maximum benefit"
                ],
                content_sections=[
                    {
                        "title": "Advanced Clinical Features",
                        "duration_hours": 2.0,
                        "content_type": "feature_demonstration",
                        "key_points": [
                            "Multi-modal AI capabilities",
                            "Complex case analysis",
                            "Population health insights",
                            "Clinical research applications"
                        ]
                    },
                    {
                        "title": "Workflow Customization",
                        "duration_hours": 2.0,
                        "content_type": "configuration_lab",
                        "key_points": [
                            "Clinical specialty customization",
                            "Workflow optimization",
                            "Integration enhancement",
                            "Performance tuning"
                        ]
                    },
                    {
                        "title": "Analytics and Reporting",
                        "duration_hours": 1.0,
                        "content_type": "analytics_workshop",
                        "key_points": [
                            "AI performance metrics",
                            "Usage analytics",
                            "Outcome measurement",
                            "ROI analysis"
                        ]
                    }
                ],
                assessments=[
                    {
                        "assessment_type": "advanced_scenarios",
                        "scenarios": 6,
                        "passing_score": 90,
                        "time_limit_minutes": 75
                    },
                    {
                        "assessment_type": "customization_project",
                        "project_type": "workflow_optimization",
                        "passing_score": 85,
                        "time_limit_minutes": 120
                    }
                ],
                cme_credits=5.0,
                prerequisites=["CLIN_001"],
                certification_levels=[CertificationLevel.ADVANCED, CertificationLevel.EXPERT]
            )
        }
    
    def _initialize_certification_paths(self) -> Dict[UserRole, Dict[CertificationLevel, Dict[str, Any]]]:
        """Initialize certification pathways for different user roles"""
        return {
            UserRole.CLINICIAN: {
                CertificationLevel.BASIC: {
                    "required_modules": ["FUND_001"],
                    "total_hours": 4.0,
                    "cme_credits": 4.0,
                    "validity_period_months": 24,
                    "competency_requirements": [
                        "AI literacy demonstration",
                        "Basic system navigation",
                        "Understanding of AI limitations"
                    ]
                },
                CertificationLevel.INTERMEDIATE: {
                    "required_modules": ["FUND_001", "CLIN_001", "COMP_001"],
                    "total_hours": 14.0,
                    "cme_credits": 14.0,
                    "validity_period_months": 24,
                    "competency_requirements": [
                        "Clinical workflow integration",
                        "AI recommendation validation",
                        "HIPAA compliance knowledge",
                        "Documentation proficiency"
                    ]
                },
                CertificationLevel.ADVANCED: {
                    "required_modules": ["FUND_001", "CLIN_001", "COMP_001", "ADV_001"],
                    "total_hours": 19.0,
                    "cme_credits": 19.0,
                    "validity_period_months": 24,
                    "competency_requirements": [
                        "Advanced feature utilization",
                        "Workflow optimization",
                        "Performance analysis",
                        "Mentoring capabilities"
                    ]
                }
            },
            
            UserRole.PHYSICIAN: {
                CertificationLevel.BASIC: {
                    "required_modules": ["FUND_001"],
                    "total_hours": 4.0,
                    "cme_credits": 4.0,
                    "validity_period_months": 24,
                    "competency_requirements": [
                        "AI physician interface understanding",
                        "Decision support recognition",
                        "Clinical accountability awareness"
                    ]
                },
                CertificationLevel.INTERMEDIATE: {
                    "required_modules": ["FUND_001", "CLIN_001", "COMP_001"],
                    "total_hours": 14.0,
                    "cme_credits": 14.0,
                    "validity_period_months": 24,
                    "competency_requirements": [
                        "Advanced clinical integration",
                        "Independent AI validation",
                        "Medical record documentation",
                        "Patient communication about AI"
                    ]
                },
                CertificationLevel.EXPERT: {
                    "required_modules": ["FUND_001", "CLIN_001", "COMP_001", "ADV_001"],
                    "total_hours": 19.0,
                    "cme_credits": 19.0,
                    "validity_period_months": 24,
                    "competency_requirements": [
                        "Complex case management",
                        "AI system optimization",
                        "Clinical research applications",
                        "Teaching and mentoring"
                    ]
                }
            },
            
            UserRole.IT_STAFF: {
                CertificationLevel.INTERMEDIATE: {
                    "required_modules": ["FUND_001", "TECH_001"],
                    "total_hours": 12.0,
                    "cme_credits": 0,
                    "validity_period_months": 36,
                    "competency_requirements": [
                        "System administration",
                        "User management",
                        "Monitoring and troubleshooting",
                        "Security implementation"
                    ]
                },
                CertificationLevel.EXPERT: {
                    "required_modules": ["FUND_001", "TECH_001", "COMP_001", "ADV_001"],
                    "total_hours": 16.0,
                    "cme_credits": 0,
                    "validity_period_months": 36,
                    "competency_requirements": [
                        "Advanced system configuration",
                        "Performance optimization",
                        "Advanced security management",
                        "Training delivery capability"
                    ]
                }
            }
        }
    
    def _initialize_assessment_bank(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize assessment question bank"""
        return {
            "knowledge_checks": [
                {
                    "question_id": "KC_001",
                    "question": "What is the primary benefit of AI-assisted clinical decision making?",
                    "options": [
                        "Replacing human judgment entirely",
                        "Enhancing clinical decision accuracy and efficiency",
                        "Eliminating the need for clinical training",
                        "Reducing healthcare costs automatically"
                    ],
                    "correct_answer": 1,
                    "explanation": "AI assists clinicians by providing data-driven insights while maintaining human oversight and accountability.",
                    "difficulty": "basic"
                },
                {
                    "question_id": "KC_002",
                    "question": "According to HIPAA, what constitutes Protected Health Information (PHI)?",
                    "options": [
                        "Only medical diagnoses",
                        "Any information that can identify a patient and relates to their health",
                        "Only financial information",
                        "Only demographic information"
                    ],
                    "correct_answer": 1,
                    "explanation": "PHI includes any information that can identify a patient and relates to their health status, care, or payment.",
                    "difficulty": "intermediate"
                }
            ],
            "clinical_scenarios": [
                {
                    "scenario_id": "CS_001",
                    "title": "AI Recommendation Override Decision",
                    "description": "A 65-year-old patient presents with chest pain. The AI system suggests a diagnosis of acid reflux, but clinical indicators suggest possible cardiac involvement.",
                    "questions": [
                        {
                            "question": "What should be your primary consideration?",
                            "options": [
                                "Trust the AI recommendation and discharge patient",
                                "Immediately order cardiac tests",
                                "Use clinical judgment to reconcile AI output with physical examination",
                                "Consult with another physician before making any decisions"
                            ],
                            "correct_answer": 2,
                            "explanation": "AI should supplement, not replace, clinical judgment. Use all available information."
                        }
                    ]
                }
            ]
        }
    
    def generate_training_plan(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized training plan for healthcare user"""
        role = UserRole(user_profile["role"])
        current_level = user_profile.get("current_certification_level", "basic")
        experience_level = user_profile.get("experience_level", "novice")
        
        training_plan = {
            "user_profile": user_profile,
            "recommended_certification_level": current_level,
            "training_path": [],
            "estimated_completion_weeks": 0,
            "total_training_hours": 0,
            "total_cme_credits": 0,
            "prerequisite_check": {},
            "assessment_schedule": [],
            "milestone_schedule": []
        }
        
        # Get certification path for role
        if role in self.certification_paths:
            cert_path = self.certification_paths[role]
            
            # Determine target certification level based on role and experience
            target_levels = self._determine_target_certification_levels(role, experience_level)
            
            for target_level in target_levels:
                if target_level in cert_path:
                    level_requirements = cert_path[target_level]
                    
                    # Check prerequisites
                    missing_prerequisites = []
                    for prereq in level_requirements["required_modules"]:
                        if prereq not in user_profile.get("completed_modules", []):
                            missing_prerequisites.append(prereq)
                    
                    training_plan["prerequisite_check"][target_level.value] = missing_prerequisites
                    
                    # Add modules to training path
                    for module_id in level_requirements["required_modules"]:
                        if module_id in self.training_modules:
                            module = self.training_modules[module_id]
                            
                            # Check if prerequisites are met
                            prereqs_met = all(
                                prereq in user_profile.get("completed_modules", [])
                                for prereq in module.prerequisites
                            )
                            
                            if not missing_prerequisites or prereqs_met:
                                training_plan["training_path"].append({
                                    "module_id": module.module_id,
                                    "module_info": asdict(module),
                                    "target_certification_level": target_level.value,
                                    "estimated_completion_hours": module.duration_hours
                                })
                                
                                training_plan["total_training_hours"] += module.duration_hours
                                training_plan["total_cme_credits"] += module.cme_credits
            
            # Calculate estimated completion time
            weekly_capacity = self._calculate_weekly_training_capacity(user_profile)
            if weekly_capacity > 0:
                training_plan["estimated_completion_weeks"] = int(
                    training_plan["total_training_hours"] / weekly_capacity
                )
        
        # Generate assessment schedule
        training_plan["assessment_schedule"] = self._generate_assessment_schedule(training_plan["training_path"])
        
        # Generate milestone schedule
        training_plan["milestone_schedule"] = self._generate_milestone_schedule(
            training_plan["training_path"], weekly_capacity
        )
        
        return training_plan
    
    def _determine_target_certification_levels(self, role: UserRole, 
                                             experience_level: str) -> List[CertificationLevel]:
        """Determine appropriate target certification levels"""
        if role in [UserRole.CLINICIAN, UserRole.PHYSICIAN]:
            if experience_level == "expert":
                return [CertificationLevel.ADVANCED, CertificationLevel.EXPERT]
            elif experience_level == "intermediate":
                return [CertificationLevel.INTERMEDIATE]
            else:
                return [CertificationLevel.BASIC, CertificationLevel.INTERMEDIATE]
        elif role == UserRole.IT_STAFF:
            return [CertificationLevel.INTERMEDIATE, CertificationLevel.EXPERT]
        else:
            return [CertificationLevel.BASIC]
    
    def _calculate_weekly_training_capacity(self, user_profile: Dict[str, Any]) -> float:
        """Calculate weekly training capacity for user"""
        base_capacity = 2.0  # hours per week
        
        # Adjust based on role
        role = user_profile.get("role")
        if role == "physician":
            # Physicians have limited time for training
            return base_capacity * 0.7
        elif role == "nurse":
            return base_capacity * 1.2  # Nurses may have more flexibility
        elif role == "it_staff":
            return base_capacity * 1.5  # IT staff often dedicated to training
        else:
            return base_capacity
    
    def _generate_assessment_schedule(self, training_path: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate assessment schedule"""
        schedule = []
        
        for module_item in training_path:
            module = module_item["module_info"]
            
            for assessment in module["assessments"]:
                schedule.append({
                    "assessment_id": f"{module['module_id']}_{assessment['assessment_type']}",
                    "module_id": module["module_id"],
                    "assessment_type": assessment["assessment_type"],
                    "questions": assessment.get("questions", 0),
                    "passing_score": assessment["passing_score"],
                    "time_limit_minutes": assessment.get("time_limit_minutes", 0),
                    "estimated_duration_minutes": 30
                })
        
        return schedule
    
    def _generate_milestone_schedule(self, training_path: List[Dict[str, Any]], 
                                   weekly_capacity: float) -> List[Dict[str, Any]]:
        """Generate training milestone schedule"""
        schedule = []
        current_date = datetime.now()
        
        for i, module_item in enumerate(training_path):
            module = module_item["module_info"]
            
            # Calculate start date
            if i == 0:
                start_date = current_date
            else:
                # Previous module completion date
                prev_duration_weeks = training_path[i-1]["estimated_completion_hours"] / weekly_capacity
                start_date = current_date + timedelta(weeks=sum(
                    training_path[j]["estimated_completion_hours"] / weekly_capacity
                    for j in range(i)
                ))
            
            # Calculate completion date
            duration_weeks = module["duration_hours"] / weekly_capacity
            completion_date = start_date + timedelta(weeks=duration_weeks)
            
            schedule.append({
                "milestone_id": f"MS_{module['module_id']}",
                "module_id": module["module_id"],
                "name": f"Complete {module['title']}",
                "start_date": start_date.isoformat(),
                "target_completion_date": completion_date.isoformat(),
                "estimated_hours": module["duration_hours"],
                "cme_credits": module["cme_credits"],
                "critical": module["module_type"] in ["clinical_application", "compliance_security"]
            })
        
        return schedule
    
    def track_user_progress(self, user_id: str, organization_id: str) -> UserProgress:
        """Track user training progress (simplified for demo)"""
        # In a real implementation, this would query actual user data
        return UserProgress(
            user_id=user_id,
            role=UserRole.CLINICIAN,
            organization_id=organization_id,
            modules_completed=["FUND_001"],
            modules_in_progress=["CLIN_001"],
            assessment_scores={"FUND_001": 92.5},
            certification_level=CertificationLevel.BASIC,
            cme_credits_earned=4.0,
            last_activity_date=datetime.now().isoformat(),
            competency_assessment_date=datetime.now().isoformat()
        )
    
    def generate_certification_report(self, progress: UserProgress) -> Dict[str, Any]:
        """Generate certification progress report"""
        report = {
            "user_progress": asdict(progress),
            "certification_status": {},
            "recommendations": [],
            "next_steps": []
        }
        
        # Determine current certification status
        role = progress.role
        if role in self.certification_paths:
            cert_path = self.certification_paths[role]
            
            for level, requirements in cert_path.items():
                completed_modules = set(progress.modules_completed)
                required_modules = set(requirements["required_modules"])
                
                if completed_modules >= required_modules:
                    report["certification_status"][level.value] = {
                        "status": "eligible",
                        "requirements_met": True,
                        "missing_requirements": [],
                        "valid_until": self._calculate_certification_expiry(requirements["validity_period_months"])
                    }
                else:
                    missing = list(required_modules - completed_modules)
                    report["certification_status"][level.value] = {
                        "status": "in_progress",
                        "requirements_met": False,
                        "missing_requirements": missing,
                        "completion_percentage": (len(completed_modules) / len(required_modules)) * 100
                    }
        
        # Generate recommendations
        if progress.certification_level == CertificationLevel.BASIC:
            report["recommendations"].append("Consider advancing to intermediate certification for enhanced capabilities")
            report["next_steps"].append("Complete clinical application training module")
        
        return report
    
    def _calculate_certification_expiry(self, validity_months: int) -> str:
        """Calculate certification expiry date"""
        expiry_date = datetime.now() + timedelta(days=validity_months * 30)
        return expiry_date.isoformat()
    
    def export_training_plan(self, training_plan: Dict[str, Any], output_path: str) -> None:
        """Export training plan to file"""
        with open(output_path, 'w') as f:
            json.dump(training_plan, f, indent=2, default=str)
    
    def validate_training_completion(self, user_progress: UserProgress) -> Dict[str, Any]:
        """Validate training completion requirements"""
        validation_results = {
            "is_certified": False,
            "current_certification_level": user_progress.certification_level,
            "eligible_certifications": [],
            "completion_percentage": 0.0,
            "missing_requirements": [],
            "recommendations": []
        }
        
        role = user_progress.role
        if role in self.certification_paths:
            cert_path = self.certification_paths[role]
            
            total_required_modules = 0
            completed_modules = len(user_progress.modules_completed)
            
            for level, requirements in cert_path.items():
                required_modules = set(requirements["required_modules"])
                completed_set = set(user_progress.modules_completed)
                
                total_required_modules += len(required_modules)
                
                if completed_set >= required_modules:
                    validation_results["eligible_certifications"].append(level.value)
                    if level == user_progress.certification_level:
                        validation_results["is_certified"] = True
            
            if total_required_modules > 0:
                validation_results["completion_percentage"] = (completed_modules / total_required_modules) * 100
                
                # Identify missing requirements
                all_required = set()
                for requirements in cert_path.values():
                    all_required.update(requirements["required_modules"])
                
                missing = list(all_required - set(user_progress.modules_completed))
                validation_results["missing_requirements"] = missing
        
        return validation_results