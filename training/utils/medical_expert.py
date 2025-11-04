"""
Medical Expert Review Interface

Professional interface for medical expert review and quality assurance workflows:
- Expert review interface
- Clinical case evaluation 
- Professional feedback integration
- Quality assurance workflows

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import uuid
from pathlib import Path


class ExpertRole(Enum):
    """Medical expert roles and specializations"""
    PHYSICIAN = "physician"
    SPECIALIST = "specialist"
    NURSE_PRACTITIONER = "nurse_practitioner"
    PHARMACIST = "pharmacist"
    SURGEON = "surgeon"
    RADIOLOGIST = "radiologist"
    PATHOLOGIST = "pathologist"
    EMERGENCY_PHYSICIAN = "emergency_physician"
    CARDIOLOGIST = "cardiologist"
    NEUROLOGIST = "neurologist"
    ONCOLOGIST = "oncologist"
    PEDIATRICIAN = "pediatrician"
    GERIATRICIAN = "geriatrician"


class ReviewStatus(Enum):
    """Review workflow status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    REVISION_REQUIRED = "revision_required"
    APPROVED = "approved"


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    UNACCEPTABLE = "unacceptable"


@dataclass
class ExpertProfile:
    """Medical expert profile information"""
    expert_id: str
    name: str
    role: ExpertRole
    specialization: Optional[str]
    years_experience: int
    credentials: List[str]
    certifications: List[str]
    affiliation: Optional[str]
    contact_email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass
class ClinicalCaseSubmission:
    """Clinical case submission for expert review"""
    submission_id: str
    case_data: Dict[str, Any]
    submitted_by: str
    submission_type: str
    priority: str
    special_instructions: Optional[str]
    submission_date: datetime
    target_expert_roles: List[ExpertRole]
    deadline: Optional[datetime]
    metadata: Dict[str, Any]


@dataclass
class ExpertReview:
    """Individual expert review of a clinical case"""
    review_id: str
    case_id: str
    expert_id: str
    review_status: ReviewStatus
    overall_quality: QualityLevel
    clinical_accuracy_score: float
    safety_score: float
    completeness_score: float
    clarity_score: float
    adherence_to_guidelines: float
    
    # Detailed assessments
    diagnostic_assessment: Dict[str, Any]
    treatment_recommendations: Dict[str, Any]
    safety_concerns: List[str]
    improvements_suggested: List[str]
    strengths_identified: List[str]
    
    # Professional feedback
    professional_comments: str
    confidential_notes: Optional[str]
    
    # Workflow tracking
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    revision_count: int
    time_spent_minutes: Optional[int]
    
    # Metadata
    review_version: int
    created_at: datetime
    updated_at: datetime


@dataclass
class ReviewWorkflow:
    """Complete review workflow for a clinical case"""
    workflow_id: str
    case_id: str
    current_status: ReviewStatus
    assigned_experts: List[str]
    completed_reviews: List[str]
    
    # Workflow phases
    phases: List[Dict[str, Any]]
    current_phase: str
    
    # Quality assurance
    consensus_required: bool
    minimum_reviews: int
    approval_threshold: float
    
    # Final outcome
    final_decision: Optional[ReviewStatus]
    final_quality_assessment: Optional[QualityLevel]
    aggregated_feedback: Dict[str, Any]
    
    # Tracking
    created_at: datetime
    updated_at: datetime
    deadline: Optional[datetime]


class MedicalExpertDatabase:
    """Database for managing medical expert information"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.experts: Dict[str, ExpertProfile] = {}
        self.role_specializations = {
            ExpertRole.CARDIOLOGIST: ["cardiology", "interventional_cardiology", "electrophysiology"],
            ExpertRole.NEUROLOGIST: ["neurology", "stroke_medicine", "epilepsy"],
            ExpertRole.ONCOLOGIST: ["medical_oncology", "radiation_oncology", "surgical_oncology"],
            ExpertRole.EMERGENCY_PHYSICIAN: ["emergency_medicine", "trauma", "critical_care"],
            ExpertRole.SURGEON: ["general_surgery", "orthopedic_surgery", "neurosurgery"],
            ExpertRole.PHARMACIST: ["clinical_pharmacy", "pharmacotherapy", "drug_interactions"],
            ExpertRole.PEDIATRICIAN: ["general_pediatrics", "pediatric_emergency", "neonatology"],
            ExpertRole.GERIATRICIAN: ["geriatric_medicine", "long_term_care", "palliative_care"]
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for expert database"""
        logger = logging.getLogger(f"{__name__}.MedicalExpertDatabase")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def add_expert(self, expert_profile: ExpertProfile) -> bool:
        """Add a new medical expert to the database"""
        try:
            self.experts[expert_profile.expert_id] = expert_profile
            self.logger.info(f"Added expert: {expert_profile.name} ({expert_profile.role.value})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add expert: {str(e)}")
            return False
    
    def get_expert(self, expert_id: str) -> Optional[ExpertProfile]:
        """Get expert profile by ID"""
        return self.experts.get(expert_id)
    
    def find_experts_by_role(self, role: ExpertRole) -> List[ExpertProfile]:
        """Find experts by medical role"""
        return [expert for expert in self.experts.values() 
                if expert.role == role and expert.is_active]
    
    def find_experts_by_specialization(self, specialization: str) -> List[ExpertProfile]:
        """Find experts by specialization"""
        return [expert for expert in self.experts.values()
                if expert.specialization and 
                specialization.lower() in expert.specialization.lower() and 
                expert.is_active]
    
    def get_available_experts(self) -> List[ExpertProfile]:
        """Get all active experts available for review"""
        return [expert for expert in self.experts.values() if expert.is_active]
    
    def assign_experts_to_case(self, 
                             case_requirements: Dict[str, Any],
                             required_roles: List[ExpertRole],
                             preferred_specializations: Optional[List[str]] = None) -> List[str]:
        """Assign appropriate experts to a clinical case"""
        
        assigned_experts = []
        
        # Get available experts for each required role
        for role in required_roles:
            role_experts = self.find_experts_by_role(role)
            
            if not role_experts:
                self.logger.warning(f"No experts available for role: {role.value}")
                continue
            
            # Filter by specialization if specified
            if preferred_specializations:
                specialized_experts = []
                for spec in preferred_specializations:
                    specialized_experts.extend(
                        self.find_experts_by_specialization(spec)
                    )
                
                # Use specialized experts if available, otherwise use role experts
                if specialized_experts:
                    role_experts = specialized_experts
            
            # Sort by experience (most experienced first)
            role_experts.sort(key=lambda x: x.years_experience, reverse=True)
            
            # Assign the most experienced expert
            if role_experts:
                assigned_experts.append(role_experts[0].expert_id)
        
        self.logger.info(f"Assigned {len(assigned_experts)} experts to case")
        return assigned_experts


class ExpertReviewSystem:
    """Main expert review system for clinical quality assurance"""
    
    def __init__(self, expert_db: Optional[MedicalExpertDatabase] = None):
        self.logger = self._setup_logger()
        self.expert_db = expert_db or MedicalExpertDatabase()
        self.active_workflows: Dict[str, ReviewWorkflow] = {}
        self.reviews: Dict[str, ExpertReview] = {}
        self.submissions: Dict[str, ClinicalCaseSubmission] = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for expert review system"""
        logger = logging.getLogger(f"{__name__}.ExpertReviewSystem")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def submit_case_for_review(self, 
                             case_data: Dict[str, Any],
                             submitted_by: str,
                             required_expert_roles: List[ExpertRole],
                             submission_type: str = "clinical_assessment",
                             priority: str = "normal",
                             special_instructions: Optional[str] = None,
                             deadline: Optional[datetime] = None,
                             consensus_required: bool = True,
                             minimum_reviews: int = 2) -> str:
        """Submit a clinical case for expert review"""
        
        # Create submission
        submission = ClinicalCaseSubmission(
            submission_id=str(uuid.uuid4()),
            case_data=case_data,
            submitted_by=submitted_by,
            submission_type=submission_type,
            priority=priority,
            special_instructions=special_instructions,
            submission_date=datetime.now(),
            target_expert_roles=required_expert_roles,
            deadline=deadline,
            metadata={
                "consensus_required": consensus_required,
                "minimum_reviews": minimum_reviews,
                "workflow_type": "expert_review"
            }
        )
        
        self.submissions[submission.submission_id] = submission
        
        # Create workflow
        workflow = self._create_review_workflow(
            submission.submission_id,
            required_expert_roles,
            consensus_required,
            minimum_reviews
        )
        
        self.active_workflows[workflow.workflow_id] = workflow
        
        self.logger.info(f"Submitted case {submission.submission_id} for review")
        return submission.submission_id
    
    def _create_review_workflow(self,
                              submission_id: str,
                              required_roles: List[ExpertRole],
                              consensus_required: bool,
                              minimum_reviews: int) -> ReviewWorkflow:
        """Create a review workflow for a submitted case"""
        
        # Assign experts
        assigned_experts = self.expert_db.assign_experts_to_case(
            case_requirements={"roles": required_roles},
            required_roles=required_roles
        )
        
        # Define workflow phases
        phases = [
            {
                "phase_name": "initial_review",
                "description": "Initial expert review and assessment",
                "required_experts": assigned_experts[:minimum_reviews] if len(assigned_experts) >= minimum_reviews else assigned_experts,
                "estimated_duration_hours": 24
            },
            {
                "phase_name": "consensus_discussion", 
                "description": "Expert consensus and discussion (if required)",
                "required_experts": assigned_experts,
                "estimated_duration_hours": 48 if consensus_required else 0
            },
            {
                "phase_name": "final_review",
                "description": "Final quality assurance review",
                "required_experts": [assigned_experts[0]] if assigned_experts else [],
                "estimated_duration_hours": 12
            }
        ]
        
        # Filter out zero-duration phases
        phases = [p for p in phases if p["estimated_duration_hours"] > 0]
        
        workflow = ReviewWorkflow(
            workflow_id=str(uuid.uuid4()),
            case_id=submission_id,
            current_status=ReviewStatus.PENDING,
            assigned_experts=assigned_experts,
            completed_reviews=[],
            phases=phases,
            current_phase=phases[0]["phase_name"] if phases else "final_review",
            consensus_required=consensus_required,
            minimum_reviews=minimum_reviews,
            approval_threshold=0.8,  # 80% agreement required
            final_decision=None,
            final_quality_assessment=None,
            aggregated_feedback={},
            created_at=datetime.now(),
            updated_at=datetime.now(),
            deadline=datetime.now()  # Default deadline
        )
        
        return workflow
    
    def start_expert_review(self,
                          workflow_id: str,
                          expert_id: str,
                          case_id: str) -> Optional[str]:
        """Start a review by an expert"""
        
        if workflow_id not in self.active_workflows:
            self.logger.error(f"Workflow {workflow_id} not found")
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        if expert_id not in workflow.assigned_experts:
            self.logger.error(f"Expert {expert_id} not assigned to workflow {workflow_id}")
            return None
        
        # Create review record
        review = ExpertReview(
            review_id=str(uuid.uuid4()),
            case_id=case_id,
            expert_id=expert_id,
            review_status=ReviewStatus.IN_PROGRESS,
            overall_quality=QualityLevel.ACCEPTABLE,  # Default
            clinical_accuracy_score=0.5,
            safety_score=0.5,
            completeness_score=0.5,
            clarity_score=0.5,
            adherence_to_guidelines=0.5,
            diagnostic_assessment={},
            treatment_recommendations={},
            safety_concerns=[],
            improvements_suggested=[],
            strengths_identified=[],
            professional_comments="",
            confidential_notes=None,
            started_at=datetime.now(),
            completed_at=None,
            revision_count=0,
            time_spent_minutes=None,
            review_version=1,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.reviews[review.review_id] = review
        workflow.completed_reviews.append(review.review_id)
        workflow.updated_at = datetime.now()
        
        self.logger.info(f"Expert {expert_id} started review {review.review_id}")
        return review.review_id
    
    def submit_expert_review(self,
                           review_id: str,
                           quality_assessment: QualityLevel,
                           clinical_accuracy: float,
                           safety_score: float,
                           completeness_score: float,
                           clarity_score: float,
                           guidelines_adherence: float,
                           diagnostic_assessment: Dict[str, Any],
                           treatment_recommendations: Dict[str, Any],
                           safety_concerns: List[str],
                           improvements_suggested: List[str],
                           strengths_identified: List[str],
                           professional_comments: str,
                           confidential_notes: Optional[str] = None) -> bool:
        """Submit completed expert review"""
        
        if review_id not in self.reviews:
            self.logger.error(f"Review {review_id} not found")
            return False
        
        review = self.reviews[review_id]
        
        # Update review with assessment
        review.overall_quality = quality_assessment
        review.clinical_accuracy_score = clinical_accuracy
        review.safety_score = safety_score
        review.completeness_score = completeness_score
        review.clarity_score = clarity_score
        review.adherence_to_guidelines = guidelines_adherence
        
        review.diagnostic_assessment = diagnostic_assessment
        review.treatment_recommendations = treatment_recommendations
        review.safety_concerns = safety_concerns
        review.improvements_suggested = improvements_suggested
        review.strengths_identified = strengths_identified
        
        review.professional_comments = professional_comments
        review.confidential_notes = confidential_notes
        
        review.review_status = ReviewStatus.COMPLETED
        review.completed_at = datetime.now()
        review.updated_at = datetime.now()
        
        # Update workflow status
        self._update_workflow_status(review.case_id)
        
        self.logger.info(f"Submitted expert review {review_id}")
        return True
    
    def _update_workflow_status(self, case_id: str):
        """Update workflow status based on completed reviews"""
        
        # Find workflow for this case
        workflow = None
        for w in self.active_workflows.values():
            if w.case_id == case_id:
                workflow = w
                break
        
        if not workflow:
            return
        
        # Get completed reviews for this workflow
        completed_reviews = [
            self.reviews[review_id] for review_id in workflow.completed_reviews
            if review_id in self.reviews
        ]
        
        completed_reviews = [r for r in completed_reviews if r.review_status == ReviewStatus.COMPLETED]
        
        # Check if minimum reviews completed
        if len(completed_reviews) >= workflow.minimum_reviews:
            
            # Calculate consensus
            if workflow.consensus_required and len(completed_reviews) > 1:
                consensus_score = self._calculate_consensus(completed_reviews)
                
                if consensus_score >= workflow.approval_threshold:
                    workflow.final_decision = ReviewStatus.APPROVED
                    workflow.current_status = ReviewStatus.COMPLETED
                else:
                    workflow.final_decision = ReviewStatus.REJECTED
                    workflow.current_status = ReviewStatus.COMPLETED
            else:
                # Single review or no consensus required
                if completed_reviews:
                    avg_quality = self._calculate_average_quality(completed_reviews)
                    workflow.final_quality_assessment = avg_quality
                    
                    if avg_quality in [QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.ACCEPTABLE]:
                        workflow.final_decision = ReviewStatus.APPROVED
                    else:
                        workflow.final_decision = ReviewStatus.REJECTED
                    
                    workflow.current_status = ReviewStatus.COMPLETED
            
            # Aggregate feedback
            workflow.aggregated_feedback = self._aggregate_feedback(completed_reviews)
        
        workflow.updated_at = datetime.now()
    
    def _calculate_consensus(self, reviews: List[ExpertReview]) -> float:
        """Calculate consensus score among reviews"""
        if not reviews:
            return 0.0
        
        # Calculate agreement on overall quality
        quality_scores = {
            QualityLevel.EXCELLENT: 5,
            QualityLevel.GOOD: 4,
            QualityLevel.ACCEPTABLE: 3,
            QualityLevel.NEEDS_IMPROVEMENT: 2,
            QualityLevel.UNACCEPTABLE: 1
        }
        
        scores = [quality_scores[review.overall_quality] for review in reviews]
        
        # Calculate variance (lower variance = higher consensus)
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Convert to consensus score (0-1)
        consensus = max(0, 1 - (variance / 4))  # Max variance is 4
        
        return consensus
    
    def _calculate_average_quality(self, reviews: List[ExpertReview]) -> QualityLevel:
        """Calculate average quality level"""
        quality_scores = {
            QualityLevel.EXCELLENT: 5,
            QualityLevel.GOOD: 4,
            QualityLevel.ACCEPTABLE: 3,
            QualityLevel.NEEDS_IMPROVEMENT: 2,
            QualityLevel.UNACCEPTABLE: 1
        }
        
        if not reviews:
            return QualityLevel.ACCEPTABLE
        
        avg_score = np.mean([quality_scores[review.overall_quality] for review in reviews])
        
        if avg_score >= 4.5:
            return QualityLevel.EXCELLENT
        elif avg_score >= 3.5:
            return QualityLevel.GOOD
        elif avg_score >= 2.5:
            return QualityLevel.ACCEPTABLE
        elif avg_score >= 1.5:
            return QualityLevel.NEEDS_IMPROVEMENT
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _aggregate_feedback(self, reviews: List[ExpertReview]) -> Dict[str, Any]:
        """Aggregate feedback from multiple expert reviews"""
        
        # Aggregate safety concerns
        all_safety_concerns = []
        for review in reviews:
            all_safety_concerns.extend(review.safety_concerns)
        
        # Aggregate improvements suggested
        all_improvements = []
        for review in reviews:
            all_improvements.extend(review.improvements_suggested)
        
        # Aggregate strengths identified
        all_strengths = []
        for review in reviews:
            all_strengths.extend(review.strengths_identified)
        
        # Aggregate diagnostic assessments
        aggregated_diagnostics = {}
        for review in reviews:
            for key, value in review.diagnostic_assessment.items():
                if key not in aggregated_diagnostics:
                    aggregated_diagnostics[key] = []
                aggregated_diagnostics[key].append(value)
        
        # Calculate average scores
        avg_clinical_accuracy = np.mean([r.clinical_accuracy_score for r in reviews])
        avg_safety_score = np.mean([r.safety_score for r in reviews])
        avg_completeness = np.mean([r.completeness_score for r in reviews])
        avg_clarity = np.mean([r.clarity_score for r in reviews])
        avg_guidelines = np.mean([r.adherence_to_guidelines for r in reviews])
        
        return {
            "aggregate_quality": self._calculate_average_quality(reviews).value,
            "consensus_score": self._calculate_consensus(reviews),
            "total_reviews": len(reviews),
            "safety_concerns": all_safety_concerns,
            "improvements_suggested": all_improvements,
            "strengths_identified": all_strengths,
            "aggregated_diagnostics": aggregated_diagnostics,
            "average_scores": {
                "clinical_accuracy": avg_clinical_accuracy,
                "safety": avg_safety_score,
                "completeness": avg_completeness,
                "clarity": avg_clarity,
                "guidelines_adherence": avg_guidelines
            },
            "consensus_metrics": {
                "agreement_on_quality": self._calculate_consensus(reviews),
                "agreement_on_safety": 1 - np.var([r.safety_score for r in reviews]),
                "agreement_on_diagnostics": self._calculate_diagnostic_consensus(reviews)
            }
        }
    
    def _calculate_diagnostic_consensus(self, reviews: List[ExpertReview]) -> float:
        """Calculate consensus on diagnostic assessments"""
        if not reviews:
            return 0.0
        
        # Simple consensus based on agreement in diagnostic assessments
        # This could be enhanced with more sophisticated similarity measures
        diagnostic_keys = set()
        for review in reviews:
            diagnostic_keys.update(review.diagnostic_assessment.keys())
        
        if not diagnostic_keys:
            return 1.0
        
        # Calculate agreement across common diagnostic keys
        agreement_scores = []
        for key in diagnostic_keys:
            values = [review.diagnostic_assessment.get(key) for review in reviews]
            if len(set(values)) == 1:  # Perfect agreement
                agreement_scores.append(1.0)
            else:  # Disagreement
                agreement_scores.append(0.0)
        
        return np.mean(agreement_scores) if agreement_scores else 0.0
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current workflow status"""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow.workflow_id,
            "case_id": workflow.case_id,
            "status": workflow.current_status.value,
            "current_phase": workflow.current_phase,
            "assigned_experts": workflow.assigned_experts,
            "completed_reviews": workflow.completed_reviews,
            "consensus_required": workflow.consensus_required,
            "minimum_reviews": workflow.minimum_reviews,
            "final_decision": workflow.final_decision.value if workflow.final_decision else None,
            "final_quality": workflow.final_quality_assessment.value if workflow.final_quality_assessment else None,
            "aggregated_feedback": workflow.aggregated_feedback,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "deadline": workflow.deadline.isoformat() if workflow.deadline else None
        }
    
    def get_expert_workload(self, expert_id: str) -> Dict[str, Any]:
        """Get current workload for an expert"""
        
        active_reviews = []
        for review in self.reviews.values():
            if (review.expert_id == expert_id and 
                review.review_status in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]):
                active_reviews.append(review.review_id)
        
        total_reviews = sum(1 for review in self.reviews.values() 
                          if review.expert_id == expert_id)
        
        avg_completion_time = self._calculate_average_completion_time(expert_id)
        
        return {
            "expert_id": expert_id,
            "active_reviews": len(active_reviews),
            "total_reviews_completed": total_reviews,
            "average_completion_time_hours": avg_completion_time,
            "review_ids": active_reviews
        }
    
    def _calculate_average_completion_time(self, expert_id: str) -> Optional[float]:
        """Calculate average review completion time for an expert"""
        
        completed_reviews = [
            review for review in self.reviews.values()
            if (review.expert_id == expert_id and 
                review.completed_at and 
                review.started_at)
        ]
        
        if not completed_reviews:
            return None
        
        completion_times = [
            (review.completed_at - review.started_at).total_seconds() / 3600
            for review in completed_reviews
        ]
        
        return np.mean(completion_times)
    
    def generate_review_report(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive review report"""
        
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        
        # Get all reviews for this workflow
        reviews = [
            self.reviews[review_id] for review_id in workflow.completed_reviews
            if review_id in self.reviews
        ]
        
        if not reviews:
            return None
        
        # Generate detailed report
        report = {
            "workflow_summary": self.get_workflow_status(workflow_id),
            "expert_profiles": [],
            "detailed_assessments": [],
            "consensus_analysis": workflow.aggregated_feedback,
            "recommendations": self._generate_review_recommendations(reviews, workflow),
            "quality_trends": self._analyze_quality_trends(reviews),
            "report_generated_at": datetime.now().isoformat()
        }
        
        # Add expert profiles
        for expert_id in workflow.assigned_experts:
            expert_profile = self.expert_db.get_expert(expert_id)
            if expert_profile:
                report["expert_profiles"].append(asdict(expert_profile))
        
        # Add detailed assessments
        for review in reviews:
            report["detailed_assessments"].append(asdict(review))
        
        return report
    
    def _generate_review_recommendations(self, 
                                       reviews: List[ExpertReview],
                                       workflow: ReviewWorkflow) -> List[str]:
        """Generate recommendations based on review results"""
        
        recommendations = []
        
        # Based on consensus
        if workflow.final_decision == ReviewStatus.APPROVED:
            recommendations.append("Case approved - clinical quality meets standards")
        elif workflow.final_decision == ReviewStatus.REJECTED:
            recommendations.append("Case rejected - significant quality issues identified")
        
        # Based on common safety concerns
        all_safety_concerns = []
        for review in reviews:
            all_safety_concerns.extend(review.safety_concerns)
        
        if all_safety_concerns:
            recommendations.append("Review safety protocols - multiple concerns identified")
        
        # Based on quality improvements
        if any(review.overall_quality in [QualityLevel.NEEDS_IMPROVEMENT, QualityLevel.UNACCEPTABLE] 
               for review in reviews):
            recommendations.append("Quality improvement plan required")
        
        return recommendations
    
    def _analyze_quality_trends(self, reviews: List[ExpertReview]) -> Dict[str, Any]:
        """Analyze quality trends in the reviews"""
        
        if len(reviews) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend in quality scores
        scores = [review.clinical_accuracy_score for review in reviews]
        
        if len(scores) >= 2:
            recent_avg = np.mean(scores[-2:])
            early_avg = np.mean(scores[:-2]) if len(scores) > 2 else scores[0]
            
            if recent_avg > early_avg + 0.1:
                trend = "improving"
            elif recent_avg < early_avg - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "recent_average": np.mean(scores[-2:]) if len(scores) >= 2 else scores[0],
            "overall_average": np.mean(scores),
            "score_variance": np.var(scores),
            "total_reviews": len(scores)
        }
    
    def export_workflow_data(self, workflow_id: str, output_path: str) -> bool:
        """Export complete workflow data to file"""
        
        workflow_data = self.generate_review_report(workflow_id)
        
        if not workflow_data:
            return False
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(workflow_data, f, indent=2, default=str)
            
            self.logger.info(f"Workflow data exported to {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to export workflow data: {str(e)}")
            return False
    
    def create_sample_experts(self):
        """Create sample expert profiles for testing"""
        
        sample_experts = [
            ExpertProfile(
                expert_id="EXP001",
                name="Dr. Sarah Chen",
                role=ExpertRole.CARDIOLOGIST,
                specialization="Interventional Cardiology",
                years_experience=15,
                credentials=["MD", "FACC"],
                certifications=["Board Certified in Cardiology", "Interventional Cardiology Fellowship"],
                affiliation="Metro General Hospital",
                contact_email="sarah.chen@metrogeneral.org",
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ExpertProfile(
                expert_id="EXP002",
                name="Dr. Michael Rodriguez",
                role=ExpertRole.EMERGENCY_PHYSICIAN,
                specialization="Emergency Medicine",
                years_experience=12,
                credentials=["MD", "FACEP"],
                certifications=["Board Certified in Emergency Medicine"],
                affiliation="City Emergency Center",
                contact_email="michael.rodriguez@cityemergency.org",
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            ),
            ExpertProfile(
                expert_id="EXP003",
                name="Dr. Lisa Thompson",
                role=ExpertRole.PHARMACIST,
                specialization="Clinical Pharmacy",
                years_experience=10,
                credentials=["PharmD", "BCPS"],
                certifications=["Board Certified Pharmacotherapy Specialist"],
                affiliation="University Medical Center",
                contact_email="lisa.thompson@universitymed.org",
                is_active=True,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        ]
        
        for expert in sample_experts:
            self.expert_db.add_expert(expert)
        
        self.logger.info("Sample experts created")


# Example usage and testing
def example_expert_review_workflow():
    """Example of expert review workflow"""
    
    # Initialize system
    expert_db = MedicalExpertDatabase()
    expert_system = ExpertReviewSystem(expert_db)
    
    # Create sample experts
    expert_system.create_sample_experts()
    
    # Example clinical case
    clinical_case = {
        "case_id": "cardiac_001",
        "patient_info": {
            "age": 65,
            "gender": "male",
            "chief_complaint": "chest_pain"
        },
        "presentation": {
            "symptoms": ["chest_pain", "shortness_of_breath", "diaphoresis"],
            "vital_signs": {
                "blood_pressure": "160/95",
                "heart_rate": "110",
                "oxygen_saturation": "94%"
            }
        },
        "assessment": {
            "differential_diagnosis": ["myocardial_infarction", "unstable_angina", "pulmonary_embolism"],
            "recommended_tests": ["ECG", "cardiac_enzymes", "chest_xray"],
            "initial_treatment": ["aspirin", "nitroglycerin", "oxygen"]
        }
    }
    
    # Submit case for review
    submission_id = expert_system.submit_case_for_review(
        case_data=clinical_case,
        submitted_by="AI_System",
        required_expert_roles=[ExpertRole.CARDIOLOGIST, ExpertRole.EMERGENCY_PHYSICIAN],
        priority="high",
        consensus_required=True,
        minimum_reviews=2
    )
    
    print(f"Submitted case: {submission_id}")
    
    # Get workflow status
    workflow_id = list(expert_system.active_workflows.keys())[0]
    status = expert_system.get_workflow_status(workflow_id)
    print(f"Workflow status: {status['status']}")
    
    # Simulate expert reviews
    assigned_experts = status['assigned_experts']
    
    for expert_id in assigned_experts[:2]:  # Limit to 2 experts for demo
        review_id = expert_system.start_expert_review(workflow_id, expert_id, submission_id)
        
        if review_id:
            # Simulate expert assessment
            success = expert_system.submit_expert_review(
                review_id=review_id,
                quality_assessment=QualityLevel.GOOD,
                clinical_accuracy=0.85,
                safety_score=0.90,
                completeness_score=0.80,
                clarity_score=0.85,
                guidelines_adherence=0.88,
                diagnostic_assessment={
                    "diagnosis_accuracy": "high",
                    "differential_consideration": "adequate",
                    "risk_stratification": "appropriate"
                },
                treatment_recommendations={
                    "appropriateness": "good",
                    "evidence_based": "yes",
                    "safety_considerations": "adequate"
                },
                safety_concerns=[
                    "Monitor for bleeding risk with dual antiplatelet therapy",
                    "Consider renal function before contrast administration"
                ],
                improvements_suggested=[
                    "Add specific time frames for re-evaluation",
                    "Consider family history in risk assessment"
                ],
                strengths_identified=[
                    "Comprehensive symptom assessment",
                    "Appropriate initial diagnostic workup",
                    "Evidence-based treatment recommendations"
                ],
                professional_comments="Well-structured case with appropriate clinical reasoning. ECG findings and cardiac enzymes would help confirm diagnosis.",
                confidential_notes="Expert comfortable with assessment level"
            )
            
            print(f"Review {review_id} submitted: {success}")
    
    # Generate final report
    final_report = expert_system.generate_review_report(workflow_id)
    print(f"Final report generated: {final_report is not None}")
    
    return final_report


if __name__ == "__main__":
    # Run example
    example_expert_review_workflow()