"""
Clinical Assessment Integration Example

Comprehensive example showing how to integrate all clinical assessment tools:
- Clinical Assessor for medical accuracy evaluation
- Medical Expert System for professional review workflows  
- Clinical Benchmark Suite for standardized evaluation

This example demonstrates a complete workflow from model development to clinical validation.

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import json
import logging
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the training directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import all clinical assessment components
try:
    from utils.clinical_assessor import (
        ClinicalAssessor, ClinicalAssessment, RiskLevel, ClinicalDomain
    )
    from utils.medical_expert import (
        ExpertReviewSystem, ExpertRole, QualityLevel, ReviewStatus
    )
    from evaluation.clinical_benchmarks import (
        ClinicalBenchmarkSuite, BenchmarkCategory, DifficultyLevel, DatasetType
    )
    print("✓ All clinical assessment modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure all dependencies are installed: numpy, pandas")
    exit(1)


class MedicalAIWorkflow:
    """Complete medical AI workflow with clinical assessment integration"""
    
    def __init__(self, output_dir: str = "./clinical_assessment_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logger()
        
        # Initialize all assessment components
        self.clinical_assessor = ClinicalAssessor()
        self.expert_system = ExpertReviewSystem()
        self.benchmark_suite = ClinicalBenchmarkSuite()
        
        self.logger.info("Medical AI Workflow initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the workflow"""
        logger = logging.getLogger("MedicalAIWorkflow")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_sample_clinical_cases(self, num_cases: int = 10) -> List[Dict[str, Any]]:
        """Create sample clinical cases for testing"""
        
        sample_cases = [
            {
                "case_id": f"cardiac_case_{i:03d}",
                "symptoms": ["chest_pain", "shortness_of_breath", "diaphoresis"],
                "diagnosis": "myocardial_infarction",
                "explanation": "Classic presentation of acute coronary syndrome with chest pain, dyspnea, and diaphoresis",
                "treatments": ["aspirin", "nitroglycerin", "oxygen", "heparin"],
                "medications": ["aspirin", "metformin", "lisinopril"],
                "patient_context": {
                    "conditions": ["diabetes", "hypertension"],
                    "age": 65,
                    "gender": "male"
                },
                "high_risk_indicators": ["elderly", "cardiac_symptoms", "diabetes"],
                "scenario_type": "emergency"
            }
            for i in range(num_cases // 2)
        ] + [
            {
                "case_id": f"respiratory_case_{i:03d}",
                "symptoms": ["cough", "fever", "chest_pain"],
                "diagnosis": "pneumonia",
                "explanation": "Presentation consistent with community-acquired pneumonia",
                "treatments": ["antibiotics", "oxygen", "hydration"],
                "medications": ["amoxicillin", "ibuprofen"],
                "patient_context": {
                    "conditions": ["asthma"],
                    "age": 45,
                    "gender": "female"
                },
                "high_risk_indicators": ["fever", "chest_pain"],
                "scenario_type": "primary_care"
            }
            for i in range(num_cases // 2, num_cases)
        ]
        
        self.logger.info(f"Created {len(sample_cases)} sample clinical cases")
        return sample_cases
    
    def run_clinical_assessment(self, clinical_cases: List[Dict[str, Any]]) -> List[ClinicalAssessment]:
        """Run comprehensive clinical assessment on cases"""
        
        self.logger.info("Starting clinical assessment...")
        start_time = time.time()
        
        assessments = []
        for case in clinical_cases:
            assessment = self.clinical_assessor.comprehensive_assessment(case)
            assessments.append(assessment)
        
        # Save individual assessments
        for assessment in assessments:
            assessment_path = self.output_dir / f"assessment_{assessment.case_id}.json"
            self.clinical_assessor.save_assessment(assessment, str(assessment_path))
        
        # Generate summary report
        self._generate_assessment_summary(assessments)
        
        assessment_time = time.time() - start_time
        self.logger.info(f"Clinical assessment completed in {assessment_time:.2f}s")
        
        return assessments
    
    def _generate_assessment_summary(self, assessments: List[ClinicalAssessment]):
        """Generate summary report of clinical assessments"""
        
        if not assessments:
            return
        
        # Calculate aggregate statistics
        total_cases = len(assessments)
        avg_overall_score = sum(a.overall_score for a in assessments) / total_cases
        
        risk_distribution = {}
        for assessment in assessments:
            risk_level = assessment.risk_level.value
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        # Calculate metric averages
        metric_averages = {}
        if assessments[0].metrics:
            for metric_name in [m.name for m in assessments[0].metrics]:
                metric_scores = [m.score for a in assessments for m in a.metrics if m.name == metric_name]
                metric_averages[metric_name] = sum(metric_scores) / len(metric_scores) if metric_scores else 0
        
        # Identify common issues
        all_warnings = []
        all_recommendations = []
        for assessment in assessments:
            all_warnings.extend(assessment.warnings)
            all_recommendations.extend(assessment.recommendations)
        
        summary = {
            "assessment_summary": {
                "total_cases": total_cases,
                "average_overall_score": avg_overall_score,
                "assessment_timestamp": datetime.now().isoformat(),
                "risk_distribution": risk_distribution,
                "metric_averages": metric_averages
            },
            "common_issues": {
                "frequent_warnings": list(set(all_warnings)),
                "common_recommendations": list(set(all_recommendations))
            },
            "quality_metrics": {
                "cases_above_threshold": sum(1 for a in assessments if a.overall_score >= 0.7),
                "cases_needing_improvement": sum(1 for a in assessments if a.overall_score < 0.5),
                "high_risk_cases": sum(1 for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
            }
        }
        
        # Save summary
        summary_path = self.output_dir / "clinical_assessment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Assessment summary saved to {summary_path}")
        return summary
    
    def setup_expert_review_workflow(self, clinical_cases: List[Dict[str, Any]]):
        """Setup expert review workflow for selected cases"""
        
        self.logger.info("Setting up expert review workflow...")
        
        # Create sample experts for demonstration
        self.expert_system.create_sample_experts()
        
        # Submit high-risk cases for expert review
        high_risk_cases = []
        for case in clinical_cases:
            if case.get("high_risk_indicators"):
                high_risk_cases.append(case)
        
        workflow_ids = []
        for case in high_risk_cases[:3]:  # Limit to first 3 cases for demo
            # Submit for expert review
            submission_id = self.expert_system.submit_case_for_review(
                case_data=case,
                submitted_by="AutomatedWorkflow",
                required_expert_roles=[ExpertRole.CARDIOLOGIST, ExpertRole.EMERGENCY_PHYSICIAN],
                priority="high",
                consensus_required=True,
                minimum_reviews=2
            )
            
            workflow_id = list(self.expert_system.active_workflows.keys())[0] if self.expert_system.active_workflows else None
            if workflow_id:
                workflow_ids.append((submission_id, workflow_id))
        
        self.logger.info(f"Submitted {len(workflow_ids)} cases for expert review")
        return workflow_ids
    
    def simulate_expert_reviews(self, workflow_ids: List[tuple]):
        """Simulate expert reviews (in real scenario, experts would complete these)"""
        
        self.logger.info("Simulating expert reviews...")
        
        for submission_id, workflow_id in workflow_ids:
            # Get workflow status
            status = self.expert_system.get_workflow_status(workflow_id)
            
            if not status or not status.get("assigned_experts"):
                continue
            
            # Simulate reviews from assigned experts
            for expert_id in status["assigned_experts"][:2]:  # Limit to 2 experts for demo
                review_id = self.expert_system.start_expert_review(
                    workflow_id, expert_id, submission_id
                )
                
                if review_id:
                    # Simulate expert assessment
                    success = self.expert_system.submit_expert_review(
                        review_id=review_id,
                        quality_assessment=QualityLevel.GOOD,
                        clinical_accuracy=0.85 + (hash(expert_id) % 100) / 1000,  # Vary scores
                        safety_score=0.90 + (hash(review_id) % 50) / 1000,
                        completeness_score=0.80 + (hash(expert_id + review_id) % 100) / 1000,
                        clarity_score=0.85,
                        guidelines_adherence=0.88,
                        diagnostic_assessment={
                            "diagnosis_accuracy": "good",
                            "differential_consideration": "adequate",
                            "risk_stratification": "appropriate"
                        },
                        treatment_recommendations={
                            "appropriateness": "good",
                            "evidence_based": "yes",
                            "safety_considerations": "adequate"
                        },
                        safety_concerns=[
                            "Monitor for drug interactions",
                            "Consider patient allergies"
                        ],
                        improvements_suggested=[
                            "Add more detailed monitoring plan",
                            "Consider alternative treatments"
                        ],
                        strengths_identified=[
                            "Comprehensive assessment",
                            "Appropriate risk stratification",
                            "Evidence-based recommendations"
                        ],
                        professional_comments="Good clinical reasoning with appropriate safety considerations.",
                        confidential_notes="Expert satisfied with assessment quality"
                    )
                    
                    if success:
                        self.logger.info(f"Expert {expert_id} completed review {review_id}")
        
        # Generate final report
        for _, workflow_id in workflow_ids:
            report = self.expert_system.generate_review_report(workflow_id)
            if report:
                report_path = self.output_dir / f"expert_review_report_{workflow_id}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                self.logger.info(f"Expert review report saved to {report_path}")
    
    def run_benchmark_evaluation(self, model_func, model_name: str) -> Dict[str, Any]:
        """Run benchmark evaluation on a model"""
        
        self.logger.info("Running benchmark evaluation...")
        
        # Create benchmark dataset
        dataset_id = self.benchmark_suite.create_benchmark_dataset(
            name="Comprehensive Clinical Evaluation Set",
            category=BenchmarkCategory.DIAGNOSTIC_ACCURACY,
            num_cases=30,
            difficulty_levels=[DifficultyLevel.BASIC, DifficultyLevel.INTERMEDIATE],
            dataset_type=DatasetType.SYNTHETIC
        )
        
        # Evaluate model
        evaluation_result = self.benchmark_suite.evaluate_model(
            model_name=model_name,
            model_function=model_func,
            dataset_id=dataset_id,
            output_dir=str(self.output_dir / "benchmark_results")
        )
        
        # Generate performance report
        performance_report = self.benchmark_suite.generate_performance_report(dataset_id)
        
        # Compare with baseline (if available)
        comparison_result = self.benchmark_suite.compare_models(
            model_names=[model_name, "baseline_model"],
            dataset_id=dataset_id
        )
        
        benchmark_summary = {
            "dataset_id": dataset_id,
            "evaluation_result": {
                "overall_score": evaluation_result.overall_score,
                "evaluation_duration": evaluation_result.evaluation_duration,
                "case_results_count": len(evaluation_result.case_results)
            },
            "performance_report": performance_report,
            "comparison": comparison_result,
            "recommendations": evaluation_result.recommendations
        }
        
        # Save benchmark results
        benchmark_path = self.output_dir / f"benchmark_evaluation_{model_name}.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_summary, f, indent=2, default=str)
        
        self.logger.info(f"Benchmark evaluation completed: {model_name} scored {evaluation_result.overall_score:.3f}")
        
        return benchmark_summary
    
    def generate_comprehensive_report(self, 
                                    clinical_assessments: List[ClinicalAssessment],
                                    benchmark_summary: Dict[str, Any],
                                    expert_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive clinical assessment report"""
        
        self.logger.info("Generating comprehensive clinical assessment report...")
        
        comprehensive_report = {
            "report_metadata": {
                "generation_timestamp": datetime.now().isoformat(),
                "workflow_version": "1.0.0",
                "total_assessment_duration": "calculated_on_completion"
            },
            "executive_summary": {
                "clinical_accuracy_overall": sum(a.overall_score for a in clinical_assessments) / len(clinical_assessments),
                "benchmark_performance": benchmark_summary.get("evaluation_result", {}).get("overall_score", 0),
                "expert_review_consensus": self._calculate_expert_consensus(expert_reports),
                "key_findings": self._generate_key_findings(clinical_assessments, benchmark_summary)
            },
            "detailed_results": {
                "clinical_assessments": {
                    "total_cases": len(clinical_assessments),
                    "average_scores": {
                        metric.name: sum(m.score for a in clinical_assessments for m in a.metrics if m.name == metric.name) / 
                                     sum(1 for a in clinical_assessments for m in a.metrics if m.name == metric.name)
                        for metric in clinical_assessments[0].metrics if clinical_assessments
                    },
                    "risk_distribution": {
                        level.value: sum(1 for a in clinical_assessments if a.risk_level == level)
                        for level in RiskLevel
                    }
                },
                "benchmark_evaluation": benchmark_summary,
                "expert_reviews": expert_reports
            },
            "recommendations": self._generate_comprehensive_recommendations(
                clinical_assessments, benchmark_summary, expert_reports
            ),
            "quality_assurance": {
                "validation_status": "completed",
                "expert_reviewed": len(expert_reports) > 0,
                "benchmark_evaluated": True,
                "safety_assessments_completed": True
            }
        }
        
        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_clinical_assessment_report.json"
        with open(report_path, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        
        return comprehensive_report
    
    def _calculate_expert_consensus(self, expert_reports: List[Dict[str, Any]]) -> float:
        """Calculate consensus among expert reviews"""
        if not expert_reports:
            return 0.0
        
        consensus_scores = []
        for report in expert_reports:
            consensus_analysis = report.get("consensus_analysis", {})
            consensus_score = consensus_analysis.get("consensus_score", 0.5)
            consensus_scores.append(consensus_score)
        
        return sum(consensus_scores) / len(consensus_scores)
    
    def _generate_key_findings(self, 
                             assessments: List[ClinicalAssessment], 
                             benchmark_summary: Dict[str, Any]) -> List[str]:
        """Generate key findings from all assessments"""
        
        findings = []
        
        # Clinical assessment findings
        avg_score = sum(a.overall_score for a in assessments) / len(assessments)
        if avg_score >= 0.8:
            findings.append("Model demonstrates strong clinical accuracy")
        elif avg_score >= 0.6:
            findings.append("Model shows acceptable clinical performance with room for improvement")
        else:
            findings.append("Model requires significant clinical accuracy improvements")
        
        # Risk assessment findings
        high_risk_cases = sum(1 for a in assessments if a.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL])
        if high_risk_cases > len(assessments) * 0.3:
            findings.append("High proportion of high-risk cases identified - enhanced monitoring recommended")
        
        # Benchmark findings
        benchmark_score = benchmark_summary.get("evaluation_result", {}).get("overall_score", 0)
        if benchmark_score >= 0.8:
            findings.append("Benchmark evaluation shows competitive performance")
        
        return findings
    
    def _generate_comprehensive_recommendations(self,
                                              assessments: List[ClinicalAssessment],
                                              benchmark_summary: Dict[str, Any],
                                              expert_reports: List[Dict[str, Any]]) -> List[str]:
        """Generate comprehensive recommendations"""
        
        recommendations = []
        
        # Based on clinical assessments
        avg_clinical_score = sum(a.overall_score for a in assessments) / len(assessments)
        if avg_clinical_score < 0.7:
            recommendations.append("Improve clinical training data and model architecture for better accuracy")
        
        # Based on benchmark evaluation
        benchmark_recommendations = benchmark_summary.get("recommendations", [])
        recommendations.extend(benchmark_recommendations)
        
        # Based on expert reviews
        if expert_reports:
            recommendations.append("Address expert-identified concerns in next model iteration")
        
        # Safety recommendations
        safety_concerns = []
        for assessment in assessments:
            safety_concerns.extend(assessment.warnings)
        
        if safety_concerns:
            recommendations.append("Implement enhanced safety checks for identified contraindications")
        
        # Quality assurance
        recommendations.append("Establish regular clinical assessment pipeline for ongoing quality monitoring")
        
        return recommendations


def mock_medical_model(clinical_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mock medical AI model for demonstration purposes"""
    
    import random
    
    # Simulate processing time
    time.sleep(random.uniform(0.1, 0.5))
    
    # Generate realistic model output based on input
    primary_diagnosis = clinical_data.get("primary_diagnosis", clinical_data.get("diagnosis", "unknown"))
    
    # Add some realistic variation
    confidence = 0.7 + random.random() * 0.25  # 0.7 to 0.95
    
    output = {
        "primary_diagnosis": primary_diagnosis,
        "differential_diagnosis": clinical_data.get("differential_diagnosis", [primary_diagnosis]),
        "confidence_score": confidence,
        "reasoning": f"Based on clinical presentation including {', '.join(clinical_data.get('symptoms', []))}",
        "treatment_recommendations": clinical_data.get("treatments", ["supportive_care"]),
        "risk_assessment": {
            "level": "moderate" if confidence > 0.8 else "high",
            "factors": clinical_data.get("high_risk_indicators", [])
        },
        "next_steps": [
            "monitor_symptoms",
            "follow_up_in_24_hours",
            "consider_specialist_consultation"
        ]
    }
    
    return output


def main():
    """Main demonstration workflow"""
    
    print("=" * 80)
    print("CLINICAL ASSESSMENT INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Initialize workflow
    workflow = MedicalAIWorkflow(output_dir="./clinical_assessment_demo_output")
    
    # Step 1: Create sample clinical cases
    print("Step 1: Creating sample clinical cases...")
    clinical_cases = workflow.create_sample_clinical_cases(num_cases=12)
    print(f"✓ Created {len(clinical_cases)} clinical cases")
    
    # Step 2: Run clinical assessment
    print("\nStep 2: Running clinical assessment...")
    assessments = workflow.run_clinical_assessment(clinical_cases)
    print(f"✓ Completed assessment of {len(assessments)} cases")
    print(f"  Average clinical accuracy: {sum(a.overall_score for a in assessments) / len(assessments):.3f}")
    
    # Step 3: Setup expert review workflow
    print("\nStep 3: Setting up expert review workflow...")
    workflow_ids = workflow.setup_expert_review_workflow(clinical_cases)
    print(f"✓ Submitted {len(workflow_ids)} cases for expert review")
    
    # Step 4: Simulate expert reviews
    print("\nStep 4: Simulating expert reviews...")
    workflow.simulate_expert_reviews(workflow_ids)
    print("✓ Expert reviews completed")
    
    # Step 5: Run benchmark evaluation
    print("\nStep 5: Running benchmark evaluation...")
    benchmark_summary = workflow.run_benchmark_evaluation(
        model_func=mock_medical_model,
        model_name="DemoMedicalAI_v1.0"
    )
    print(f"✓ Benchmark evaluation completed")
    print(f"  Model performance: {benchmark_summary['evaluation_result']['overall_score']:.3f}")
    
    # Step 6: Generate comprehensive report
    print("\nStep 6: Generating comprehensive report...")
    expert_reports = []
    for submission_id, workflow_id in workflow_ids:
        report = workflow.expert_system.generate_review_report(workflow_id)
        if report:
            expert_reports.append(report)
    
    comprehensive_report = workflow.generate_comprehensive_report(
        assessments, benchmark_summary, expert_reports
    )
    print("✓ Comprehensive report generated")
    
    # Final summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("SUMMARY RESULTS:")
    print(f"• Clinical Cases Assessed: {len(clinical_cases)}")
    print(f"• Average Clinical Accuracy: {comprehensive_report['executive_summary']['clinical_accuracy_overall']:.3f}")
    print(f"• Benchmark Performance: {comprehensive_report['executive_summary']['benchmark_performance']:.3f}")
    print(f"• Expert Review Consensus: {comprehensive_report['executive_summary']['expert_review_consensus']:.3f}")
    print(f"• High-Risk Cases: {comprehensive_report['detailed_results']['clinical_assessments']['risk_distribution']['high']}")
    print()
    print("OUTPUT FILES:")
    print(f"• All results saved to: {workflow.output_dir}")
    print("• Comprehensive report: comprehensive_clinical_assessment_report.json")
    print("• Clinical assessments: assessment_*.json")
    print("• Expert reviews: expert_review_report_*.json")
    print("• Benchmark results: benchmark_evaluation_*.json")
    print()
    print("KEY RECOMMENDATIONS:")
    for i, rec in enumerate(comprehensive_report['recommendations'][:5], 1):
        print(f"{i}. {rec}")
    print()
    print("✓ Clinical Assessment Integration Demonstration Complete!")


if __name__ == "__main__":
    # Run the comprehensive demonstration
    main()
