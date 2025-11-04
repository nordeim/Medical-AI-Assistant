"""
Healthcare Implementation Timeline and Project Management System
Phase-based project planning with medical workflow integration scheduling
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ProjectPhase(Enum):
    """Implementation project phases"""
    PLANNING = "planning"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"
    TESTING = "testing"
    TRAINING = "training"
    DEPLOYMENT = "deployment"
    OPTIMIZATION = "optimization"
    CLOSURE = "closure"

class ResourceType(Enum):
    """Types of project resources"""
    TECHNICAL = "technical"
    CLINICAL = "clinical"
    PROJECT_MANAGEMENT = "project_management"
    COMPLIANCE = "compliance"
    TRAINING = "training"

class DependencyType(Enum):
    """Types of task dependencies"""
    FINISH_TO_START = "finish_to_start"
    START_TO_START = "start_to_start"
    FINISH_TO_FINISH = "finish_to_finish"
    START_TO_FINISH = "start_to_finish"

@dataclass
class TimelineTask:
    """Individual task in the implementation timeline"""
    task_id: str
    name: str
    description: str
    phase: ProjectPhase
    estimated_duration_hours: int
    dependencies: List[str]
    assigned_resources: List[str]
    deliverables: List[str]
    success_criteria: Dict[str, Any]
    risk_level: str  # low, medium, high, critical
    medical_workflow_impact: str
    clinical_validation_required: bool
    compliance_checkpoints: List[str]

@dataclass
class ResourceAllocation:
    """Resource allocation for project tasks"""
    resource_type: ResourceType
    name: str
    availability_percentage: int
    skills_required: List[str]
    cost_per_hour: float
    availability_calendar: Dict[str, int]  # date -> hours available

class HealthcareProjectManager:
    """Project management system for healthcare implementations"""
    
    def __init__(self):
        self.tasks = {}
        self.phases = {}
        self.resource_allocations = {}
        self._initialize_standard_phases()
    
    def _initialize_standard_phases(self):
        """Initialize standard project phases for healthcare implementations"""
        self.phases = {
            ProjectPhase.PLANNING: {
                "name": "Project Planning & Assessment",
                "description": "Initial project planning, stakeholder alignment, and technical assessment",
                "typical_duration_days": 14,
                "key_activities": [
                    "Stakeholder kickoff meeting",
                    "Technical infrastructure assessment",
                    "Clinical workflow analysis",
                    "Resource planning and allocation",
                    "Risk assessment and mitigation planning"
                ],
                "deliverables": [
                    "Project charter and scope document",
                    "Technical assessment report",
                    "Clinical workflow integration plan",
                    "Resource allocation plan",
                    "Risk management plan"
                ],
                "exit_criteria": [
                    "Stakeholder approval of project plan",
                    "Technical feasibility confirmed",
                    "Resources allocated and committed",
                    "Risk mitigation strategies defined"
                ]
            },
            
            ProjectPhase.INFRASTRUCTURE: {
                "name": "Infrastructure Setup & Security",
                "description": "Infrastructure preparation, security implementation, and compliance setup",
                "typical_duration_days": 21,
                "key_activities": [
                    "Server and network infrastructure setup",
                    "Security controls implementation",
                    "HIPAA compliance configuration",
                    "Backup and disaster recovery setup",
                    "Performance monitoring deployment"
                ],
                "deliverables": [
                    "Infrastructure deployment completed",
                    "Security controls validated",
                    "HIPAA compliance verification",
                    "Disaster recovery procedures",
                    "Performance baseline established"
                ],
                "exit_criteria": [
                    "Infrastructure passes all tests",
                    "Security controls verified",
                    "Compliance requirements met",
                    "Performance meets baseline"
                ]
            },
            
            ProjectPhase.INTEGRATION: {
                "name": "System Integration & Data Migration",
                "description": "Integration with existing systems and data migration",
                "typical_duration_days": 28,
                "key_activities": [
                    "EHR/EMR integration development",
                    "API integrations with existing systems",
                    "Data migration and validation",
                    "Interoperability testing",
                    "Clinical workflow integration"
                ],
                "deliverables": [
                    "Integration specifications completed",
                    "EHR/EMR integration tested",
                    "Data migration validated",
                    "Clinical workflows integrated",
                    "Interoperability confirmed"
                ],
                "exit_criteria": [
                    "All integrations functioning correctly",
                    "Data integrity verified",
                    "Clinical workflows operational",
                    "Performance benchmarks met"
                ]
            },
            
            ProjectPhase.TESTING: {
                "name": "Testing & Quality Assurance",
                "description": "Comprehensive testing including clinical validation",
                "typical_duration_days": 21,
                "key_activities": [
                    "Functional testing",
                    "Performance testing",
                    "Security testing",
                    "Clinical workflow validation",
                    "User acceptance testing"
                ],
                "deliverables": [
                    "Test plans and procedures",
                    "Functional test results",
                    "Performance test results",
                    "Clinical validation reports",
                    "UAT sign-off documentation"
                ],
                "exit_criteria": [
                    "All tests passed successfully",
                    "Clinical validation completed",
                    "Performance meets requirements",
                    "User acceptance achieved"
                ]
            },
            
            ProjectPhase.TRAINING: {
                "name": "Staff Training & Certification",
                "description": "Comprehensive training program for healthcare staff",
                "typical_duration_days": 14,
                "key_activities": [
                    "Training material development",
                    "Train-the-trainer sessions",
                    "End-user training delivery",
                    "Competency assessments",
                    "Ongoing education planning"
                ],
                "deliverables": [
                    "Training curriculum completed",
                    "Trainers certified",
                    "Staff training completed",
                    "Competency assessments passed",
                    "Ongoing education program"
                ],
                "exit_criteria": [
                    "All required staff trained",
                    "Competency requirements met",
                    "Training effectiveness validated",
                    "Support procedures established"
                ]
            },
            
            ProjectPhase.DEPLOYMENT: {
                "name": "Production Deployment & Go-Live",
                "description": "Production deployment and initial support",
                "typical_duration_days": 7,
                "key_activities": [
                    "Production environment setup",
                    "Go-live deployment",
                    "Initial user support",
                    "Performance monitoring",
                    "Issue resolution"
                ],
                "deliverables": [
                    "Production deployment completed",
                    "Go-live support delivered",
                    "Performance monitoring active",
                    "Issue tracking system",
                    "Support procedures operational"
                ],
                "exit_criteria": [
                    "Successful production deployment",
                    "System stability confirmed",
                    "User support effective",
                    "Performance monitoring active"
                ]
            },
            
            ProjectPhase.OPTIMIZATION: {
                "name": "Optimization & Refinement",
                "description": "System optimization and continuous improvement",
                "typical_duration_days": 28,
                "key_activities": [
                    "Performance optimization",
                    "User experience enhancement",
                    "Workflow refinement",
                    "Success metrics analysis",
                    "Continuous improvement implementation"
                ],
                "deliverables": [
                    "Optimization recommendations",
                    "Performance improvements implemented",
                    "User experience enhancements",
                    "Success metrics dashboard",
                    "Improvement roadmap"
                ],
                "exit_criteria": [
                    "Performance targets achieved",
                    "User satisfaction high",
                    "Workflow efficiency improved",
                    "Success metrics positive"
                ]
            },
            
            ProjectPhase.CLOSURE: {
                "name": "Project Closure & Transition",
                "description": "Project closure and transition to ongoing support",
                "typical_duration_days": 7,
                "key_activities": [
                    "Final documentation completion",
                    "Knowledge transfer",
                    "Support transition",
                    "Lessons learned analysis",
                    "Success celebration"
                ],
                "deliverables": [
                    "Final project documentation",
                    "Knowledge transfer completed",
                    "Support transition plan",
                    "Lessons learned report",
                    "Success documentation"
                ],
                "exit_criteria": [
                    "All deliverables completed",
                    "Knowledge transferred successfully",
                    "Support transition completed",
                    "Project formally closed"
                ]
            }
        }
    
    def create_implementation_timeline(self, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation timeline for healthcare customer"""
        timeline = {
            "customer_profile": customer_profile,
            "project_start_date": datetime.now().isoformat(),
            "estimated_completion_date": None,
            "total_duration_days": 0,
            "phases": [],
            "critical_path": [],
            "resource_schedule": {},
            "milestone_schedule": [],
            "risk_mitigation_schedule": [],
            "communication_schedule": []
        }
        
        # Calculate base timeline
        total_days = 0
        critical_path_tasks = []
        
        for phase_enum, phase_info in self.phases.items():
            # Customize phase based on customer profile
            customized_phase = self._customize_phase_for_customer(phase_info, customer_profile)
            
            # Generate tasks for this phase
            phase_tasks = self._generate_phase_tasks(phase_enum, customized_phase, customer_profile)
            
            # Add to timeline
            timeline["phases"].append({
                "phase": phase_enum.value,
                "info": customized_phase,
                "tasks": phase_tasks,
                "estimated_duration_days": customized_phase["typical_duration_days"]
            })
            
            total_days += customized_phase["typical_duration_days"]
            
            # Identify critical path tasks
            critical_path_tasks.extend([task for task in phase_tasks if task.risk_level in ["high", "critical"]])
        
        timeline["total_duration_days"] = total_days
        
        # Calculate completion date
        start_date = datetime.now()
        completion_date = start_date + timedelta(days=total_days)
        timeline["estimated_completion_date"] = completion_date.isoformat()
        
        # Generate resource schedule
        timeline["resource_schedule"] = self._generate_resource_schedule(customer_profile, total_days)
        
        # Generate milestone schedule
        timeline["milestone_schedule"] = self._generate_milestone_schedule(timeline["phases"])
        
        # Generate risk mitigation schedule
        timeline["risk_mitigation_schedule"] = self._generate_risk_mitigation_schedule(customer_profile)
        
        # Generate communication schedule
        timeline["communication_schedule"] = self._generate_communication_schedule(total_days)
        
        return timeline
    
    def _customize_phase_for_customer(self, phase_info: Dict[str, Any], 
                                    customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Customize phase based on customer profile"""
        customized = phase_info.copy()
        
        # Adjust timeline based on organization size
        size_category = customer_profile.get("size_category", "medium")
        size_multipliers = {
            "small": 0.8,
            "medium": 1.0,
            "large": 1.3,
            "enterprise": 1.6
        }
        
        multiplier = size_multipliers.get(size_category, 1.0)
        customized["typical_duration_days"] = int(customized["typical_duration_days"] * multiplier)
        
        # Add customer-specific requirements
        provider_type = customer_profile.get("provider_type", "hospital")
        if provider_type == "hospital" and phase_info["phase"] == ProjectPhase.INTEGRATION:
            customized["key_activities"].extend([
                "Multi-department coordination",
                "Hospital administration integration",
                "Bed management system integration"
            ])
        elif provider_type == "clinic" and phase_info["phase"] == ProjectPhase.PLANNING:
            customized["key_activities"].extend([
                "Clinic-specific workflow analysis",
                "Provider adoption strategy development"
            ])
        
        # Add compliance-specific requirements
        compliance_requirements = customer_profile.get("compliance_requirements", [])
        if "HIPAA" in compliance_requirements and phase_info["phase"] == ProjectPhase.INFRASTRUCTURE:
            customized["deliverables"].extend([
                "HIPAA risk assessment",
                "Business Associate Agreement",
                "Privacy and security procedures"
            ])
        
        return customized
    
    def _generate_phase_tasks(self, phase_enum: ProjectPhase, phase_info: Dict[str, Any],
                            customer_profile: Dict[str, Any]) -> List[TimelineTask]:
        """Generate detailed tasks for a project phase"""
        tasks = []
        
        # Base tasks for each phase
        base_tasks = {
            ProjectPhase.PLANNING: [
                TimelineTask(
                    task_id="PLN_001",
                    name="Stakeholder Kickoff Meeting",
                    description="Initial stakeholder alignment and project charter approval",
                    phase=phase_enum,
                    estimated_duration_hours=4,
                    dependencies=[],
                    assigned_resources=["project_manager", "customer_executive"],
                    deliverables=["Project charter", "Stakeholder register"],
                    success_criteria={"attendance": "100%", "charter_approved": True},
                    risk_level="low",
                    medical_workflow_impact="Executive alignment for clinical transformation",
                    clinical_validation_required=False,
                    compliance_checkpoints=[]
                ),
                TimelineTask(
                    task_id="PLN_002",
                    name="Clinical Workflow Assessment",
                    description="Comprehensive analysis of current clinical workflows",
                    phase=phase_enum,
                    estimated_duration_hours=16,
                    dependencies=["PLN_001"],
                    assigned_resources=["clinical_analyst", "healthcare_consultant"],
                    deliverables=["Current state workflow documentation", "Improvement opportunities"],
                    success_criteria={"coverage": "All key workflows documented", "stakeholder_validation": True},
                    risk_level="medium",
                    medical_workflow_impact="Foundation for workflow optimization",
                    clinical_validation_required=True,
                    compliance_checkpoints=["Clinical workflow compliance review"]
                )
            ],
            
            ProjectPhase.INFRASTRUCTURE: [
                TimelineTask(
                    task_id="INF_001",
                    name="Security Infrastructure Setup",
                    description="Implementation of security controls and access management",
                    phase=phase_enum,
                    estimated_duration_hours=32,
                    dependencies=["PLN_001"],
                    assigned_resources=["security_engineer", "system_administrator"],
                    deliverables=["Security controls", "Access management system"],
                    success_criteria={"security_controls": "100% implemented", "testing": "Passed"},
                    risk_level="high",
                    medical_workflow_impact="Secure foundation for clinical data processing",
                    clinical_validation_required=False,
                    compliance_checkpoints=["HIPAA Security Rule compliance", "Security audit"]
                ),
                TimelineTask(
                    task_id="INF_002",
                    name="HIPAA Compliance Configuration",
                    description="Configuration of HIPAA-compliant data handling procedures",
                    phase=phase_enum,
                    estimated_duration_hours=24,
                    dependencies=["INF_001"],
                    assigned_resources=["compliance_specialist", "privacy_officer"],
                    deliverables=["HIPAA procedures", "Privacy policies", "BAAs"],
                    success_criteria={"compliance": "100%", "documentation": "Complete"},
                    risk_level="high",
                    medical_workflow_impact="Regulatory compliant clinical data management",
                    clinical_validation_required=False,
                    compliance_checkpoints=["HIPAA compliance validation", "Privacy impact assessment"]
                )
            ],
            
            ProjectPhase.INTEGRATION: [
                TimelineTask(
                    task_id="INT_001",
                    name="EHR/EMR Integration Development",
                    description="Integration with existing electronic health records system",
                    phase=phase_enum,
                    estimated_duration_hours=40,
                    dependencies=["INF_002"],
                    assigned_resources=["integration_architect", "ehr_specialist", "developer"],
                    deliverables=["Integration code", "API documentation", "Test results"],
                    success_criteria={"integration_success": True, "performance": "Meets SLA"},
                    risk_level="high",
                    medical_workflow_impact="Seamless clinical data flow",
                    clinical_validation_required=True,
                    compliance_checkpoints=["Data integrity validation", "Interoperability standards"]
                )
            ]
        }
        
        # Get base tasks or create minimal task set
        if phase_enum in base_tasks:
            tasks.extend(base_tasks[phase_enum])
        else:
            # Generate generic task
            tasks.append(TimelineTask(
                task_id=f"{phase_enum.value.upper()}_001",
                name=f"Core {phase_enum.value.title()} Activities",
                description=f"Core activities for {phase_enum.value} phase",
                phase=phase_enum,
                estimated_duration_hours=24,
                dependencies=[],
                assigned_resources=["project_manager"],
                deliverables=["Phase deliverables"],
                success_criteria={"completion": "100%"},
                risk_level="medium",
                medical_workflow_impact="Healthcare implementation support",
                clinical_validation_required=True,
                compliance_checkpoints=[]
            ))
        
        return tasks
    
    def _generate_resource_schedule(self, customer_profile: Dict[str, Any], 
                                  total_days: int) -> Dict[str, Any]:
        """Generate resource allocation schedule"""
        base_resources = {
            "project_manager": {"allocation": 100, "cost_per_hour": 150},
            "technical_lead": {"allocation": 75, "cost_per_hour": 175},
            "clinical_specialist": {"allocation": 50, "cost_per_hour": 200},
            "compliance_specialist": {"allocation": 25, "cost_per_hour": 180},
            "integration_architect": {"allocation": 60, "cost_per_hour": 190},
            "training_specialist": {"allocation": 40, "cost_per_hour": 160}
        }
        
        # Scale based on organization size
        size_multiplier = {
            "small": 0.7,
            "medium": 1.0,
            "large": 1.3,
            "enterprise": 1.6
        }.get(customer_profile.get("size_category", "medium"), 1.0)
        
        # Apply scaling
        for resource, details in base_resources.items():
            details["allocation"] = int(details["allocation"] * size_multiplier)
            details["estimated_cost"] = details["allocation"] / 100 * 8 * total_days * details["cost_per_hour"]
        
        return {
            "resource_allocation": base_resources,
            "total_estimated_cost": sum(r["estimated_cost"] for r in base_resources.values()),
            "critical_resources": ["project_manager", "technical_lead", "clinical_specialist"],
            "resource_conflicts": []
        }
    
    def _generate_milestone_schedule(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate milestone schedule"""
        milestones = []
        current_date = datetime.now()
        
        for phase in phases:
            # Calculate phase dates
            phase_duration = timedelta(days=phase["estimated_duration_days"])
            phase_start = current_date
            phase_end = current_date + phase_duration
            
            # Add phase milestone
            milestones.append({
                "milestone_id": f"MIL_{phase['phase'].upper()}",
                "name": f"{phase['phase'].title()} Phase Complete",
                "date": phase_end.isoformat(),
                "phase": phase["phase"],
                "success_criteria": phase["info"]["exit_criteria"],
                "deliverables": phase["info"]["deliverables"],
                "critical": phase["phase"] in ["integration", "deployment"]
            })
            
            current_date = phase_end
        
        return milestones
    
    def _generate_risk_mitigation_schedule(self, customer_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk mitigation schedule"""
        risks = [
            {
                "risk_id": "R001",
                "risk": "Integration complexity with legacy systems",
                "probability": "Medium",
                "impact": "High",
                "mitigation": "Early technical assessment and proof of concept",
                "timeline": "Planning phase",
                "owner": "Integration Architect"
            },
            {
                "risk_id": "R002",
                "risk": "Clinical workflow disruption during deployment",
                "probability": "Medium",
                "impact": "Critical",
                "mitigation": "Phased rollout with clinical validation",
                "timeline": "Integration through deployment",
                "owner": "Clinical Specialist"
            },
            {
                "risk_id": "R003",
                "risk": "HIPAA compliance issues",
                "probability": "Low",
                "impact": "Critical",
                "mitigation": "Early compliance assessment and continuous monitoring",
                "timeline": "Infrastructure through deployment",
                "owner": "Compliance Specialist"
            },
            {
                "risk_id": "R004",
                "risk": "User adoption challenges",
                "probability": "Medium",
                "impact": "Medium",
                "mitigation": "Comprehensive training and change management",
                "timeline": "Training through optimization",
                "owner": "Training Specialist"
            }
        ]
        
        return risks
    
    def _generate_communication_schedule(self, total_days: int) -> List[Dict[str, Any]]:
        """Generate stakeholder communication schedule"""
        communications = [
            {
                "communication_id": "COMM_001",
                "type": "Stakeholder Meeting",
                "frequency": "Weekly",
                "audience": "Executive stakeholders",
                "purpose": "Project status and decision making",
                "duration_hours": 1,
                "distribution": ["Email summary", "Presentation"]
            },
            {
                "communication_id": "COMM_002",
                "type": "Clinical Team Update",
                "frequency": "Bi-weekly",
                "audience": "Clinical staff and department heads",
                "purpose": "Workflow integration updates",
                "duration_hours": 0.5,
                "distribution": ["Email", "Team meeting"]
            },
            {
                "communication_id": "COMM_003",
                "type": "Technical Team Sync",
                "frequency": "Daily",
                "audience": "Technical team",
                "purpose": "Development progress and blockers",
                "duration_hours": 0.5,
                "distribution": ["Standup notes"]
            },
            {
                "communication_id": "COMM_004",
                "type": "Go-Live Announcement",
                "frequency": "One-time",
                "audience": "All users",
                "purpose": "System deployment notification",
                "duration_hours": 0,
                "distribution": ["Email", "Posters", "Training sessions"]
            }
        ]
        
        return communications
    
    def export_timeline(self, timeline: Dict[str, Any], output_path: str) -> None:
        """Export implementation timeline to file"""
        # Convert dataclasses to dictionaries for JSON serialization
        export_data = timeline.copy()
        
        for phase in export_data["phases"]:
            if "tasks" in phase:
                phase["tasks"] = [asdict(task) for task in phase["tasks"]]
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def calculate_timeline_variance(self, planned_timeline: Dict[str, Any], 
                                  actual_timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate timeline variance analysis"""
        variance_analysis = {
            "overall_variance_days": 0,
            "phase_variances": [],
            "critical_path_impact": [],
            "resource_utilization_variance": {},
            "recommendations": []
        }
        
        planned_total = planned_timeline["total_duration_days"]
        actual_total = actual_timeline.get("total_duration_days", planned_total)
        
        variance_analysis["overall_variance_days"] = actual_total - planned_total
        
        # Analyze phase variances
        for phase in planned_timeline["phases"]:
            planned_duration = phase["estimated_duration_days"]
            actual_duration = self._get_actual_phase_duration(phase["phase"], actual_timeline)
            variance = actual_duration - planned_duration
            
            variance_analysis["phase_variances"].append({
                "phase": phase["phase"],
                "planned_duration": planned_duration,
                "actual_duration": actual_duration,
                "variance_days": variance,
                "variance_percentage": (variance / planned_duration) * 100 if planned_duration > 0 else 0
            })
        
        # Generate recommendations
        if variance_analysis["overall_variance_days"] > 7:
            variance_analysis["recommendations"].append("Critical timeline variance - escalate to executive stakeholders")
        if variance_analysis["overall_variance_days"] > 14:
            variance_analysis["recommendations"].append("Major project delay - consider timeline replanning")
        
        return variance_analysis
    
    def _get_actual_phase_duration(self, phase: str, actual_timeline: Dict[str, Any]) -> int:
        """Get actual duration for a phase (simplified for demo)"""
        # In a real implementation, this would query actual project data
        return actual_timeline.get("total_duration_days", 0) // len(actual_timeline.get("phases", []))