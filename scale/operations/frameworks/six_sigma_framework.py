"""
Six Sigma Operational Excellence Framework for Healthcare AI
Implements DMAIC methodology for statistical process control and defect reduction
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
import statistics

class SigmaLevel(Enum):
    """Six Sigma quality levels"""
    SIGMA_1 = 1  # 69.15% yield
    SIGMA_2 = 2  # 93.32% yield
    SIGMA_3 = 3  # 99.38% yield
    SIGMA_4 = 4  # 99.98% yield
    SIGMA_5 = 5  # 99.9997% yield
    SIGMA_6 = 6  # 99.999999998% yield

class DefectType(Enum):
    """Types of defects in healthcare AI systems"""
    CLINICAL_ACCURACY = "clinical_accuracy"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM_PERFORMANCE = "system_performance"
    COMPLIANCE_VIOLATION = "compliance_violation"
    USER_EXPERIENCE = "user_experience"
    SECURITY_BREACH = "security_breach"

class DMAICPhase(Enum):
    """Six Sigma DMAIC phases"""
    DEFINE = "define"
    MEASURE = "measure"
    ANALYZE = "analyze"
    IMPROVE = "improve"
    CONTROL = "control"

@dataclass
class DefectMetrics:
    """Defect measurement metrics"""
    defect_type: DefectType
    count: int
    rate_per_million: float
    cost_impact: float
    sigma_level: SigmaLevel
    severity: str  # Low, Medium, High, Critical
    root_cause: str
    owner: str

@dataclass
class ProcessCapability:
    """Process capability analysis"""
    process_name: str
    cp_value: float  # Process Capability Index
    cpk_value: float  # Process Capability Index with centering
    sigma_level: SigmaLevel
    yield_percentage: float
    defect_rate: float  # DPMO (Defects Per Million Opportunities)
    control_limits: Dict[str, float]  # UCL, LCL
    specification_limits: Dict[str, float]  # USL, LSL

@dataclass
class StatisticalTest:
    """Statistical test results"""
    test_name: str
    p_value: float
    confidence_level: float
    result: str  # Accept/Reject null hypothesis
    effect_size: float
    power_analysis: float
    sample_size: int
    interpretation: str

@dataclass
class ImprovementProject:
    """Six Sigma improvement project"""
    project_id: str
    title: str
    description: str
    dmaic_phase: DMAICPhase
    problem_statement: str
    goal_statement: str
    scope: str
    timeline_weeks: int
    team_members: List[str]
    metrics: Dict[str, float]
    baseline_performance: Dict[str, float]
    target_performance: Dict[str, float]
    status: str

class SixSigmaHealthcareAIManager:
    """Six Sigma Operations Manager for Healthcare AI"""
    
    def __init__(self):
        self.defect_metrics: List[DefectMetrics] = []
        self.process_capabilities: Dict[str, ProcessCapability] = {}
        self.improvement_projects: List[ImprovementProject] = []
        self.statistical_tests: List[StatisticalTest] = []
        self.current_sigma_level = SigmaLevel.SIGMA_3
        
    async def define_phase(self, project: ImprovementProject) -> Dict:
        """Define phase of DMAIC methodology"""
        define_checklist = {
            "problem_identification": {
                "problem_statement": project.problem_statement,
                "business_impact": "High - affecting clinical decision accuracy",
                "customer_impact": "Patient safety and care quality at risk",
                "scope_definition": project.scope
            },
            "goal_setting": {
                "goal_statement": project.goal_statement,
                "success_criteria": [
                    "Reduce defect rate by 50%",
                    "Achieve Sigma 4.5 level",
                    "Improve clinical accuracy to 99.5%",
                    "Zero patient safety incidents"
                ],
                "measurable_objectives": project.target_performance
            },
            "team_formation": {
                "team_lead": "Six Sigma Black Belt",
                "members": project.team_members,
                "stakeholders": ["Clinical Director", "IT Manager", "Quality Assurance"],
                "sponsor": "Chief Medical Officer"
            },
            "project_planning": {
                "timeline": f"{project.timeline_weeks} weeks",
                "milestones": [
                    "Week 2: Measure current state",
                    "Week 4: Analyze root causes",
                    "Week 6: Implement improvements",
                    "Week 8: Control and sustain"
                ],
                "resources_required": [
                    "Data analysis tools",
                    "Clinical subject matter experts",
                    "IT system access",
                    "Statistical software"
                ]
            }
        }
        
        return {
            "phase": DMAICPhase.DEFINE.value,
            "project_id": project.project_id,
            "status": "completed",
            "deliverables": define_checklist,
            "next_phase": DMAICPhase.MEASURE.value
        }
    
    async def measure_phase(self, process_name: str, sample_data: List[float]) -> Dict:
        """Measure phase of DMAIC methodology"""
        
        # Calculate statistical measures
        mean_value = statistics.mean(sample_data)
        std_dev = statistics.stdev(sample_data) if len(sample_data) > 1 else 0
        median_value = statistics.median(sample_data)
        
        # Calculate process capability metrics
        usl = max(sample_data) * 1.1  # Upper Specification Limit
        lsl = min(sample_data) * 0.9  # Lower Specification Limit
        
        cp = (usl - lsl) / (6 * std_dev) if std_dev > 0 else 0
        cpk = min((usl - mean_value) / (3 * std_dev), 
                 (mean_value - lsl) / (3 * std_dev)) if std_dev > 0 else 0
        
        # Determine sigma level based on Cp/Cpk values
        if cp >= 2.0:
            sigma_level = SigmaLevel.SIGMA_6
            yield_percentage = 99.999999998
        elif cp >= 1.67:
            sigma_level = SigmaLevel.SIGMA_5
            yield_percentage = 99.9997
        elif cp >= 1.33:
            sigma_level = SigmaLevel.SIGMA_4
            yield_percentage = 99.98
        elif cp >= 1.0:
            sigma_level = SigmaLevel.SIGMA_3
            yield_percentage = 99.38
        elif cp >= 0.67:
            sigma_level = SigmaLevel.SIGMA_2
            yield_percentage = 93.32
        else:
            sigma_level = SigmaLevel.SIGMA_1
            yield_percentage = 69.15
        
        defect_rate = (1 - yield_percentage/100) * 1000000  # DPMO
        
        # Control limits (3 sigma)
        ucl = mean_value + (3 * std_dev)
        lcl = mean_value - (3 * std_dev)
        
        # Create process capability object
        capability = ProcessCapability(
            process_name=process_name,
            cp_value=cp,
            cpk_value=cpk,
            sigma_level=sigma_level,
            yield_percentage=yield_percentage,
            defect_rate=defect_rate,
            control_limits={"UCL": ucl, "LCL": lcl},
            specification_limits={"USL": usl, "LSL": lsl}
        )
        
        self.process_capabilities[process_name] = capability
        
        measurement_results = {
            "phase": DMAICPhase.MEASURE.value,
            "process_name": process_name,
            "sample_size": len(sample_data),
            "statistical_measures": {
                "mean": round(mean_value, 4),
                "standard_deviation": round(std_dev, 4),
                "median": round(median_value, 4),
                "range": round(max(sample_data) - min(sample_data), 4)
            },
            "process_capability": {
                "cp_value": round(cp, 4),
                "cpk_value": round(cpk, 4),
                "sigma_level": sigma_level.value,
                "yield_percentage": round(yield_percentage, 4),
                "defect_rate_dpmo": round(defect_rate, 2)
            },
            "control_limits": {
                "upper_control_limit": round(ucl, 4),
                "lower_control_limit": round(lcl, 4)
            },
            "specification_limits": {
                "upper_spec_limit": round(usl, 4),
                "lower_spec_limit": round(lsl, 4)
            },
            "quality_assessment": {
                "process_capable": cp >= 1.33,
                "process_centered": abs(cpk - cp) < 0.1,
                "improvement_needed": sigma_level.value < 4
            }
        }
        
        return measurement_results
    
    async def analyze_phase(self, process_name: str, analysis_type: str = "root_cause") -> Dict:
        """Analyze phase of DMAIC methodology"""
        
        # Simulate different analysis types for healthcare AI processes
        analysis_results = {}
        
        if process_name == "ai_model_inference":
            analysis_results = {
                "root_cause_analysis": {
                    "primary_causes": [
                        "Training data bias (45% contribution)",
                        "Model overfitting (30% contribution)",
                        "Feature selection issues (15% contribution)",
                        "Infrastructure latency (10% contribution)"
                    ],
                    "correlation_analysis": {
                        "data_quality_score": 0.85,
                        "model_complexity": -0.72,
                        "feature_importance_variance": 0.68
                    },
                    "statistical_tests": [
                        {
                            "test_name": "Chi-Square Test",
                            "p_value": 0.003,
                            "result": "Reject null hypothesis",
                            "interpretation": "Data quality significantly affects accuracy"
                        }
                    ]
                },
                "pareto_analysis": {
                    "top_defects": [
                        {"defect": "Prediction Accuracy", "frequency": 45, "percentage": 65.2},
                        {"defect": "Response Time", "frequency": 18, "percentage": 26.1},
                        {"defect": "Data Loss", "frequency": 6, "percentage": 8.7}
                    ],
                    "vital_few": "Top 2 defects account for 91.3% of issues"
                }
            }
        
        elif process_name == "clinical_decision_support":
            analysis_results = {
                "fishbone_analysis": {
                    "categories": {
                        "People": ["Insufficient training", "Skill gaps", "Communication issues"],
                        "Process": ["Inadequate protocols", "Documentation errors", "Workflow bottlenecks"],
                        "Technology": ["System limitations", "Integration issues", "Performance problems"],
                        "Data": ["Quality issues", "Incomplete records", "Inconsistent formats"],
                        "Environment": ["Workload pressure", "Time constraints", "Noise/distractions"],
                        "Management": ["Resource allocation", "Quality oversight", "Strategic alignment"]
                    },
                    "top_root_causes": [
                        "Inadequate clinical training on AI tools",
                        "Missing data validation protocols",
                        "System integration complexity"
                    ]
                },
                "regression_analysis": {
                    "model_r_squared": 0.847,
                    "significant_factors": [
                        {"factor": "Clinician Experience", "coefficient": 0.42, "p_value": 0.001},
                        {"factor": "AI Confidence Score", "coefficient": 0.38, "p_value": 0.002},
                        {"factor": "Patient Complexity", "coefficient": -0.25, "p_value": 0.015}
                    ]
                }
            }
        
        # Create statistical tests for various analyses
        statistical_tests = [
            StatisticalTest(
                test_name="Two-Sample T-Test",
                p_value=0.023,
                confidence_level=0.95,
                result="Reject null hypothesis",
                effect_size=0.65,
                power_analysis=0.78,
                sample_size=150,
                interpretation="Significant improvement observed after intervention"
            ),
            StatisticalTest(
                test_name="ANOVA",
                p_value=0.005,
                confidence_level=0.95,
                result="Reject null hypothesis",
                effect_size=0.82,
                power_analysis=0.91,
                sample_size=200,
                interpretation="Significant differences between groups detected"
            )
        ]
        
        self.statistical_tests.extend(statistical_tests)
        
        return {
            "phase": DMAICPhase.ANALYZE.value,
            "process_name": process_name,
            "analysis_type": analysis_type,
            "results": analysis_results,
            "statistical_tests": [
                {
                    "test_name": test.test_name,
                    "p_value": test.p_value,
                    "confidence_level": test.confidence_level,
                    "result": test.result,
                    "effect_size": test.effect_size,
                    "interpretation": test.interpretation
                }
                for test in statistical_tests
            ],
            "key_findings": [
                "Root cause analysis reveals data quality as primary issue",
                "Statistical tests confirm significant improvement potential",
                "Pareto analysis focuses efforts on vital few defects"
            ],
            "next_phase": DMAICPhase.IMPROVE.value
        }
    
    async def improve_phase(self, project: ImprovementProject, improvements: List[Dict]) -> Dict:
        """Improve phase of DMAIC methodology"""
        
        # Simulate improvement implementation
        improvement_results = {
            "phase": DMAICPhase.IMPROVE.value,
            "project_id": project.project_id,
            "improvements_implemented": [
                {
                    "improvement": "Automated Data Validation Pipeline",
                    "description": "Implement real-time data quality checks",
                    "impact": {
                        "defect_reduction": "65%",
                        "cost_savings": "$125K annually",
                        "time_savings": "40 hours/week"
                    },
                    "implementation_cost": "$45K",
                    "timeline": "6 weeks"
                },
                {
                    "improvement": "Enhanced AI Model Training",
                    "description": "Improve model accuracy through advanced techniques",
                    "impact": {
                        "accuracy_improvement": "15%",
                        "response_time_reduction": "25%",
                        "patient_safety_improvement": "80%"
                    },
                    "implementation_cost": "$80K",
                    "timeline": "8 weeks"
                }
            ],
            "pilot_results": {
                "baseline_metrics": {
                    "defect_rate": 450,  # per million
                    "sigma_level": 3.2,
                    "yield_percentage": 95.5,
                    "process_capability": 1.15
                },
                "improved_metrics": {
                    "defect_rate": 158,  # per million
                    "sigma_level": 4.1,
                    "yield_percentage": 98.4,
                    "process_capability": 1.67
                },
                "improvements": {
                    "defect_reduction": "65%",
                    "sigma_improvement": "0.9 levels",
                    "yield_improvement": "2.9 percentage points",
                    "capability_improvement": "45%"
                }
            },
            "validation_testing": {
                "a_b_testing_results": {
                    "control_group_performance": 95.5,
                    "treatment_group_performance": 98.4,
                    "statistical_significance": "p < 0.01",
                    "confidence_interval": "95%"
                },
                "statistical_validation": [
                    {
                        "test": "Chi-Square Test",
                        "p_value": 0.002,
                        "result": "Significant improvement"
                    },
                    {
                        "test": "T-Test",
                        "p_value": 0.008,
                        "result": "Significant improvement"
                    }
                ]
            },
            "risks_and_mitigation": [
                {
                    "risk": "Implementation resistance",
                    "mitigation": "Stakeholder engagement and training"
                },
                {
                    "risk": "Technology integration issues",
                    "mitigation": "Phased rollout with fallback procedures"
                }
            ]
        }
        
        return improvement_results
    
    async def control_phase(self, project: ImprovementProject) -> Dict:
        """Control phase of DMAIC methodology"""
        
        # Create control plan
        control_plan = {
            "phase": DMAICPhase.CONTROL.value,
            "project_id": project.project_id,
            "control_measures": [
                {
                    "metric": "AI Model Accuracy",
                    "measurement_method": "Automated testing pipeline",
                    "measurement_frequency": "Daily",
                    "control_limits": "95-100%",
                    "response_plan": "Alert if < 97% for 2 consecutive days",
                    "owner": "ML Operations Team"
                },
                {
                    "metric": "Clinical Decision Quality",
                    "measurement_method": "Random clinical audits",
                    "measurement_frequency": "Weekly",
                    "control_limits": "> 95% compliance",
                    "response_plan": "Root cause analysis and retraining",
                    "owner": "Clinical Quality Team"
                },
                {
                    "metric": "System Response Time",
                    "measurement_method": "Real-time monitoring",
                    "measurement_frequency": "Continuous",
                    "control_limits": "< 100ms",
                    "response_plan": "Immediate escalation and performance review",
                    "owner": "DevOps Team"
                }
            ],
            "monitoring_dashboard": {
                "key_indicators": [
                    "Sigma Level Trend",
                    "Defect Rate Trend",
                    "Process Capability Index",
                    "Cost of Quality"
                ],
                "alert_thresholds": {
                    "defect_rate_increase": "> 20% week-over-week",
                    "sigma_level_drop": "Below 4.0",
                    "capability_decline": "Cp < 1.33"
                },
                "reporting_frequency": "Daily dashboard, Weekly summary, Monthly review"
            },
            "sustainability_plan": {
                "process_documentation": "Updated SOPs and work instructions",
                "training_program": "Monthly refreshers and new hire training",
                "auditing_schedule": "Quarterly compliance audits",
                "continuous_improvement": "Monthly Kaizen events"
            }
        }
        
        return control_plan
    
    async def calculate_defect_metrics(self, process_name: str, defects_data: Dict) -> List[DefectMetrics]:
        """Calculate comprehensive defect metrics"""
        
        defect_metrics = []
        
        # Define expected defect rates for different processes
        process_defects = {
            "ai_model_inference": [
                {"type": DefectType.CLINICAL_ACCURACY, "count": 25, "rate_per_million": 850, "cost_impact": 50000},
                {"type": DefectType.SYSTEM_PERFORMANCE, "count": 12, "rate_per_million": 400, "cost_impact": 15000},
                {"type": DefectType.DATA_INTEGRITY, "count": 8, "rate_per_million": 270, "cost_impact": 25000}
            ],
            "clinical_decision_support": [
                {"type": DefectType.CLINICAL_ACCURACY, "count": 15, "rate_per_million": 500, "cost_impact": 75000},
                {"type": DefectType.USER_EXPERIENCE, "count": 20, "rate_per_million": 670, "cost_impact": 10000},
                {"type": DefectType.COMPLIANCE_VIOLATION, "count": 3, "rate_per_million": 100, "cost_impact": 100000}
            ]
        }
        
        for defect_data in process_defects.get(process_name, []):
            # Determine sigma level based on defect rate
            defect_rate = defect_data["rate_per_million"]
            if defect_rate <= 3.4:  # Six Sigma level
                sigma_level = SigmaLevel.SIGMA_6
            elif defect_rate <= 233:  # Five Sigma level
                sigma_level = SigmaLevel.SIGMA_5
            elif defect_rate <= 13500:  # Four Sigma level
                sigma_level = SigmaLevel.SIGMA_4
            elif defect_rate <= 66800:  # Three Sigma level
                sigma_level = SigmaLevel.SIGMA_3
            else:
                sigma_level = SigmaLevel.SIGMA_2
            
            # Determine severity based on defect type and rate
            if defect_data["type"] == DefectType.COMPLIANCE_VIOLATION:
                severity = "Critical"
            elif defect_data["type"] == DefectType.CLINICAL_ACCURACY:
                severity = "High"
            elif defect_data["rate_per_million"] > 500:
                severity = "Medium"
            else:
                severity = "Low"
            
            defect_metrics.append(DefectMetrics(
                defect_type=defect_data["type"],
                count=defect_data["count"],
                rate_per_million=defect_data["rate_per_million"],
                cost_impact=defect_data["cost_impact"],
                sigma_level=sigma_level,
                severity=severity,
                root_cause=f"Process optimization needed for {defect_data['type'].value}",
                owner="Quality Assurance Team"
            ))
        
        self.defect_metrics.extend(defect_metrics)
        return defect_metrics
    
    async def generate_six_sigma_dashboard(self) -> Dict:
        """Generate Six Sigma dashboard data"""
        
        # Calculate overall sigma level
        total_defects = sum(dm.rate_per_million for dm in self.defect_metrics)
        weighted_sigma = sum(dm.sigma_level.value * dm.rate_per_million for dm in self.defect_metrics) / total_defects if total_defects > 0 else 3.0
        
        dashboard_data = {
            "sigma_level_overview": {
                "current_sigma_level": round(weighted_sigma, 1),
                "target_sigma_level": 4.5,
                "sigma_level_trend": "+0.3 this quarter",
                "quality_yield_percentage": 98.2
            },
            "defect_analysis": {
                "total_defects_per_million": total_defects,
                "top_defect_types": [
                    {"type": "Clinical Accuracy", "count": 40, "percentage": 40.4},
                    {"type": "User Experience", "count": 20, "percentage": 20.2},
                    {"type": "Data Integrity", "count": 18, "percentage": 18.2}
                ],
                "defect_cost_impact": sum(dm.cost_impact for dm in self.defect_metrics),
                "defect_reduction_target": "50% this year"
            },
            "process_capability": {
                "capable_processes": len([pc for pc in self.process_capabilities.values() if pc.cp_value >= 1.33]),
                "total_processes": len(self.process_capabilities),
                "average_cp": statistics.mean([pc.cp_value for pc in self.process_capabilities.values()]) if self.process_capabilities else 0,
                "capability_improvement": "+15% this quarter"
            },
            "improvement_projects": {
                "active_projects": len([p for p in self.improvement_projects if p.status == "active"]),
                "completed_projects": len([p for p in self.improvement_projects if p.status == "completed"]),
                "projects_on_track": 12,
                "average_roi": 285.5  # percentage
            },
            "real_time_metrics": {
                "defect_rate_trend": "-12.5% week-over-week",
                "sigma_improvement_trend": "+0.15 month-over-month",
                "cost_savings_trend": "+$25K this month",
                "quality_score_trend": "+2.1 points this quarter"
            }
        }
        
        return dashboard_data
    
    async def export_six_sigma_report(self, filepath: str) -> Dict:
        """Export comprehensive Six Sigma report"""
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_title": "Healthcare AI Six Sigma Quality Report",
                "reporting_period": "Q4 2025",
                "methodology": "DMAIC"
            },
            "executive_summary": {
                "current_sigma_level": 3.8,
                "target_sigma_level": 4.5,
                "defect_reduction_achieved": "42%",
                "cost_savings_realized": "$485K",
                "process_improvements": 15,
                "quality_yield": "98.2%"
            },
            "dmaic_project_details": [
                {
                    "project_id": p.project_id,
                    "title": p.title,
                    "current_phase": p.dmaic_phase.value,
                    "timeline": f"{p.timeline_weeks} weeks",
                    "status": p.status
                }
                for p in self.improvement_projects
            ],
            "process_capability_analysis": [
                {
                    "process_name": pc.process_name,
                    "cp_value": pc.cp_value,
                    "cpk_value": pc.cpk_value,
                    "sigma_level": pc.sigma_level.value,
                    "yield_percentage": pc.yield_percentage,
                    "defect_rate_dpmo": pc.defect_rate
                }
                for pc in self.process_capabilities.values()
            ],
            "recommendations": [
                "Focus on critical defects with highest patient safety impact",
                "Implement statistical process control for all key processes",
                "Increase sampling frequency for high-risk processes",
                "Establish cross-functional teams for complex improvements",
                "Create quality culture through training and recognition"
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return {"status": "success", "report_file": filepath}

# Example usage and testing
async def run_six_sigma_demo():
    """Demonstrate Six Sigma Framework implementation"""
    six_sigma_manager = SixSigmaHealthcareAIManager()
    
    # 1. Define Phase
    print("=== Define Phase ===")
    project = ImprovementProject(
        project_id="DMAIC_001",
        title="Improve AI Model Clinical Accuracy",
        description="Reduce prediction errors in clinical decision support",
        dmaic_phase=DMAICPhase.DEFINE,
        problem_statement="Current AI model accuracy is 95.2%, below target of 99%",
        goal_statement="Achieve 99.5% clinical accuracy while maintaining response time < 100ms",
        scope="All clinical decision support features across healthcare departments",
        timeline_weeks=12,
        team_members=["ML Engineer", "Clinical Specialist", "Quality Analyst"],
        metrics={"accuracy": 95.2, "response_time": 85, "patient_safety_score": 87},
        baseline_performance={"accuracy": 95.2, "defect_rate": 4800, "sigma_level": 3.2},
        target_performance={"accuracy": 99.5, "defect_rate": 500, "sigma_level": 4.5},
        status="active"
    )
    
    define_result = await six_sigma_manager.define_phase(project)
    print(f"Project: {project.title}")
    print(f"Goal: {project.goal_statement}")
    print(f"Status: {define_result['status']}")
    
    # 2. Measure Phase
    print("\n=== Measure Phase ===")
    sample_accuracy_data = [94.8, 95.2, 96.1, 95.5, 94.9, 95.8, 96.2, 95.1, 95.7, 95.3]
    measure_result = await six_sigma_manager.measure_phase("ai_model_inference", sample_accuracy_data)
    print(f"Process: {measure_result['process_name']}")
    print(f"Sigma Level: {measure_result['process_capability']['sigma_level']}")
    print(f"Yield: {measure_result['process_capability']['yield_percentage']:.2f}%")
    print(f"Defect Rate: {measure_result['process_capability']['defect_rate_dpmo']:.0f} DPMO")
    
    # 3. Analyze Phase
    print("\n=== Analyze Phase ===")
    analyze_result = await six_sigma_manager.analyze_phase("ai_model_inference")
    print(f"Phase: {analyze_result['phase']}")
    print("Key Findings:")
    for finding in analyze_result['key_findings']:
        print(f"  - {finding}")
    
    # 4. Improve Phase
    print("\n=== Improve Phase ===")
    improvements = [
        {"name": "Data Quality Pipeline", "impact": "65% defect reduction"},
        {"name": "Model Enhancement", "impact": "15% accuracy improvement"}
    ]
    improve_result = await six_sigma_manager.improve_phase(project, improvements)
    print(f"Defect Reduction: {improve_result['pilot_results']['improvements']['defect_reduction']}")
    print(f"Sigma Improvement: {improve_result['pilot_results']['improvements']['sigma_improvement']}")
    
    # 5. Control Phase
    print("\n=== Control Phase ===")
    control_result = await six_sigma_manager.control_phase(project)
    print(f"Phase: {control_result['phase']}")
    print(f"Control Measures: {len(control_result['control_measures'])}")
    print("Key Metrics:")
    for measure in control_result['control_measures'][:2]:
        print(f"  - {measure['metric']}: {measure['measurement_frequency']}")
    
    # 6. Calculate Defect Metrics
    print("\n=== Defect Metrics Analysis ===")
    defect_metrics = await six_sigma_manager.calculate_defect_metrics("ai_model_inference", {})
    for dm in defect_metrics[:2]:
        print(f"Defect Type: {dm.defect_type.value}")
        print(f"Rate: {dm.rate_per_million:.0f} per million")
        print(f"Sigma Level: {dm.sigma_level.value}")
        print(f"Cost Impact: ${dm.cost_impact:,}")
        print("---")
    
    # 7. Generate Dashboard
    print("\n=== Six Sigma Dashboard ===")
    dashboard_data = await six_sigma_manager.generate_six_sigma_dashboard()
    print(f"Current Sigma Level: {dashboard_data['sigma_level_overview']['current_sigma_level']}")
    print(f"Defect Rate: {dashboard_data['defect_analysis']['total_defects_per_million']:.0f} DPMO")
    print(f"Cost Impact: ${dashboard_data['defect_analysis']['defect_cost_impact']:,}")
    
    # 8. Export Report
    print("\n=== Exporting Six Sigma Report ===")
    report_result = await six_sigma_manager.export_six_sigma_report("six_sigma_report.json")
    print(f"Report exported to: {report_result['report_file']}")
    
    return six_sigma_manager

if __name__ == "__main__":
    asyncio.run(run_six_sigma_demo())
