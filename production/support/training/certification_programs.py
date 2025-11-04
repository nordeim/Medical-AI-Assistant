"""
Training and Certification Programs for Healthcare Professionals
Comprehensive certification system with CME credits and competency assessments
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import statistics
import logging

from config.support_config import SupportConfig

logger = logging.getLogger(__name__)

class CertificationTrack(Enum):
    HEALTHCARE_PROFESSIONAL = "healthcare_professional"
    ADMINISTRATOR_CERTIFICATION = "administrator_certification"
    IT_SUPPORT_SPECIALIST = "it_support_specialist"
    MEDICAL_DIRECTOR = "medical_director"
    NURSE_SUPERVISOR = "nurse_supervisor"

class TrainingModule(Enum):
    MEDICAL_AI_FUNDAMENTALS = "medical_ai_fundamentals"
    CLINICAL_INTEGRATION = "clinical_integration"
    PATIENT_SAFETY = "patient_safety"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    SYSTEM_ADMINISTRATION = "system_administration"
    USER_MANAGEMENT = "user_management"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    EMERGENCY_PROCEDURES = "emergency_procedures"
    DATA_PROTECTION = "data_protection"

class AssessmentType(Enum):
    PRACTICAL_SIMULATION = "practical_simulation"
    WRITTEN_EXAMINATION = "written_examination"
    CASE_STUDY_ANALYSIS = "case_study_analysis"
    HANDS_ON_LAB = "hands_on_lab"
    COMPETENCY_DEMONSTRATION = "competency_demonstration"
    PEER_EVALUATION = "peer_evaluation"

class CompletionStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    EXPIRED = "expired"

class CompetencyLevel(Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class MedicalCEUCredits:
    """Continuing Education Units for medical professionals"""
    clinical_training: float
    safety_training: float
    regulatory_training: float
    technology_training: float
    total_credits: float
    accreditation_body: str
    valid_until: datetime

@dataclass
class LearningObjective:
    """Learning objective for training modules"""
    id: str
    description: str
    competency_level: CompetencyLevel
    assessment_criteria: List[str]
    practical_application: str
    medical_context: str

@dataclass
class TrainingModuleContent:
    """Individual training module content"""
    id: str
    module_type: TrainingModule
    title: str
    description: str
    duration_hours: float
    difficulty_level: int  # 1-5
    learning_objectives: List[LearningObjective]
    prerequisites: List[str]
    content_sections: List[Dict[str, Any]]
    assessment_questions: List[Dict[str, Any]]
    practical_exercises: List[Dict[str, Any]]
    medical_specialties: List[str]
    target_roles: List[str]
    cme_eligible: bool
    last_updated: datetime

@dataclass
class CertificationTrack:
    """Complete certification track"""
    id: str
    name: str
    description: str
    target_audience: str
    modules: List[TrainingModule]
    total_duration_hours: float
    validity_period_months: int
    continuing_education_required: bool
    assessment_requirements: List[AssessmentType]
    passing_score: float
    created_date: datetime
    version: str

@dataclass
class UserProgress:
    """User training progress tracking"""
    user_id: str
    user_name: str
    user_facility: str
    user_role: str
    enrollment_date: datetime
    certification_tracks: Dict[str, Dict[str, Any]]  # track_id -> progress_data
    completed_modules: List[str]
    assessment_scores: Dict[str, float]
    competency_levels: Dict[str, CompetencyLevel]
    ceu_credits: MedicalCEUCredits
    certificates_earned: List[str]
    last_activity: datetime
    total_training_hours: float

@dataclass
class AssessmentResult:
    """Assessment result for training modules"""
    id: str
    user_id: str
    module_id: str
    assessment_type: AssessmentType
    score: float
    max_score: float
    passed: bool
    completed_at: datetime
    time_taken_minutes: int
    feedback_provided: bool
    competency_demonstrated: List[str]

class MedicalEducationSystem:
    """Medical education and certification management system"""
    
    def __init__(self):
        self.training_modules: Dict[str, TrainingModuleContent] = {}
        self.certification_tracks: Dict[str, CertificationTrack] = {}
        self.user_progress: Dict[str, UserProgress] = {}
        self.assessment_results: List[AssessmentResult] = []
        self.accreditation_body = "American Medical Informatics Association (AMIA)"
        self.cme_credit_conversion = {
            TrainingModule.MEDICAL_AI_FUNDAMENTALS: 2.0,
            TrainingModule.CLINICAL_INTEGRATION: 3.0,
            TrainingModule.PATIENT_SAFETY: 2.5,
            TrainingModule.REGULATORY_COMPLIANCE: 2.0,
            TrainingModule.SYSTEM_ADMINISTRATION: 1.5,
            TrainingModule.EMERGENCY_PROCEDURES: 1.0,
            TrainingModule.DATA_PROTECTION: 1.5
        }
        
        # Initialize certification tracks
        self._initialize_certification_tracks()
        self._initialize_training_modules()
    
    async def enroll_user_in_track(
        self,
        user_id: str,
        user_name: str,
        user_facility: str,
        user_role: str,
        certification_track_id: str
    ) -> UserProgress:
        """Enroll user in a certification track"""
        
        if certification_track_id not in self.certification_tracks:
            raise ValueError(f"Certification track {certification_track_id} not found")
        
        if user_id in self.user_progress:
            user_progress = self.user_progress[user_id]
            
            # Add new track if not already enrolled
            if certification_track_id not in user_progress.certification_tracks:
                user_progress.certification_tracks[certification_track_id] = {
                    "enrolled_date": datetime.now().isoformat(),
                    "status": CompletionStatus.NOT_STARTED.value,
                    "modules_completed": [],
                    "assessments_attempted": [],
                    "current_module": None,
                    "completion_percentage": 0.0
                }
        else:
            # Create new user progress
            track = self.certification_tracks[certification_track_id]
            
            user_progress = UserProgress(
                user_id=user_id,
                user_name=user_name,
                user_facility=user_facility,
                user_role=user_role,
                enrollment_date=datetime.now(),
                certification_tracks={
                    certification_track_id: {
                        "enrolled_date": datetime.now().isoformat(),
                        "status": CompletionStatus.NOT_STARTED.value,
                        "modules_completed": [],
                        "assessments_attempted": [],
                        "current_module": None,
                        "completion_percentage": 0.0
                    }
                },
                completed_modules=[],
                assessment_scores={},
                competency_levels={},
                ceu_credits=MedicalCEUCredits(
                    clinical_training=0.0,
                    safety_training=0.0,
                    regulatory_training=0.0,
                    technology_training=0.0,
                    total_credits=0.0,
                    accreditation_body=self.accreditation_body,
                    valid_until=datetime.now() + timedelta(days=730)  # 2 years
                ),
                certificates_earned=[],
                last_activity=datetime.now(),
                total_training_hours=0.0
            )
        
        self.user_progress[user_id] = user_progress
        
        logger.info(f"Enrolled user {user_name} in certification track {certification_track_id}")
        return user_progress
    
    async def start_module(
        self,
        user_id: str,
        certification_track_id: str,
        module_id: str
    ) -> Dict[str, Any]:
        """Start a training module"""
        
        if user_id not in self.user_progress:
            raise ValueError(f"User {user_id} not found")
        
        user_progress = self.user_progress[user_id]
        
        if certification_track_id not in user_progress.certification_tracks:
            raise ValueError(f"User not enrolled in track {certification_track_id}")
        
        if module_id not in self.training_modules:
            raise ValueError(f"Training module {module_id} not found")
        
        # Check prerequisites
        module = self.training_modules[module_id]
        if module.prerequisites:
            track_progress = user_progress.certification_tracks[certification_track_id]
            completed_modules = track_progress["modules_completed"]
            
            for prereq in module.prerequisites:
                if prereq not in completed_modules:
                    raise ValueError(f"Prerequisite module {prereq} not completed")
        
        # Update user progress
        track_progress = user_progress.certification_tracks[certification_track_id]
        track_progress["current_module"] = module_id
        track_progress["status"] = CompletionStatus.IN_PROGRESS.value
        user_progress.last_activity = datetime.now()
        
        logger.info(f"User {user_progress.user_name} started module {module_id}")
        return {
            "module_id": module_id,
            "module_title": module.title,
            "duration_hours": module.duration_hours,
            "status": "started",
            "started_at": datetime.now().isoformat()
        }
    
    async def complete_module(
        self,
        user_id: str,
        certification_track_id: str,
        module_id: str,
        completion_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete a training module"""
        
        if user_id not in self.user_progress:
            raise ValueError(f"User {user_id} not found")
        
        user_progress = self.user_progress[user_id]
        module = self.training_modules[module_id]
        
        # Update completion status
        track_progress = user_progress.certification_tracks[certification_track_id]
        
        if module_id not in track_progress["modules_completed"]:
            track_progress["modules_completed"].append(module_id)
            user_progress.completed_modules.append(module_id)
            user_progress.total_training_hours += module.duration_hours
        
        track_progress["current_module"] = None
        
        # Update completion percentage
        track = self.certification_tracks[certification_track_id]
        completion_percentage = (len(track_progress["modules_completed"]) / len(track.modules)) * 100
        track_progress["completion_percentage"] = completion_percentage
        
        # Award CME credits if applicable
        if module.cme_eligible:
            cme_credits = self.cme_credit_conversion.get(module.module_type, 0.0)
            await self._award_cme_credits(user_id, module, cme_credits)
        
        # Check if track is completed
        if completion_percentage >= 100:
            track_progress["status"] = CompletionStatus.COMPLETED.value
            await self._complete_certification_track(user_id, certification_track_id)
        
        user_progress.last_activity = datetime.now()
        
        logger.info(f"User {user_progress.user_name} completed module {module_id}")
        return {
            "module_id": module_id,
            "completion_percentage": completion_percentage,
            "track_status": track_progress["status"],
            "total_training_hours": user_progress.total_training_hours
        }
    
    async def submit_assessment(
        self,
        user_id: str,
        module_id: str,
        assessment_type: AssessmentType,
        answers: Dict[str, Any],
        time_taken_minutes: int
    ) -> AssessmentResult:
        """Submit assessment for a module"""
        
        if user_id not in self.user_progress:
            raise ValueError(f"User {user_id} not found")
        
        if module_id not in self.training_modules:
            raise ValueError(f"Training module {module_id} not found")
        
        module = self.training_modules[module_id]
        
        # Calculate score based on assessment type
        if assessment_type == AssessmentType.WRITTEN_EXAMINATION:
            score = await self._grade_written_exam(module, answers)
        elif assessment_type == AssessmentType.PRACTICAL_SIMULATION:
            score = await self._grade_practical_simulation(module, answers)
        elif assessment_type == AssessmentType.CASE_STUDY_ANALYSIS:
            score = await self._grade_case_study(module, answers)
        else:
            score = await self._grade_assessment(module, assessment_type, answers)
        
        # Determine if passed
        track = self._get_user_track_for_module(user_id, module_id)
        passing_score = track.passing_score if track else 80.0
        passed = score >= passing_score
        
        # Create assessment result
        result = AssessmentResult(
            id=f"ASMT-{user_id}-{module_id}-{len([r for r in self.assessment_results if r.user_id == user_id and r.module_id == module_id]) + 1}",
            user_id=user_id,
            module_id=module_id,
            assessment_type=assessment_type,
            score=score,
            max_score=100.0,
            passed=passed,
            completed_at=datetime.now(),
            time_taken_minutes=time_taken_minutes,
            feedback_provided=True,
            competency_demonstrated=await self._assess_competencies(module, score, answers)
        )
        
        self.assessment_results.append(result)
        
        # Update user progress
        user_progress = self.user_progress[user_id]
        user_progress.assessment_scores[f"{module_id}_{assessment_type.value}"] = score
        
        # Update competency levels
        for competency in result.competency_demonstrated:
            user_progress.competency_levels[competency] = CompetencyLevel.INTERMEDIATE
        
        logger.info(f"Assessment submitted for user {user_progress.user_name}: {score:.1f}% ({'passed' if passed else 'failed'})")
        return result
    
    async def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user training dashboard"""
        
        if user_id not in self.user_progress:
            return {"message": "User not found"}
        
        user_progress = self.user_progress[user_id]
        
        # Current enrollments
        current_enrollments = []
        for track_id, progress in user_progress.certification_tracks.items():
            track = self.certification_tracks[track_id]
            current_enrollments.append({
                "track_id": track_id,
                "track_name": track.name,
                "status": progress["status"],
                "completion_percentage": progress["completion_percentage"],
                "current_module": progress["current_module"],
                "enrolled_date": progress["enrolled_date"]
            })
        
        # Recent activity
        recent_activity = []
        for result in sorted(
            [r for r in self.assessment_results if r.user_id == user_id],
            key=lambda x: x.completed_at,
            reverse=True
        )[:10]:
            recent_activity.append({
                "module_id": result.module_id,
                "module_title": self.training_modules[result.module_id].title if result.module_id in self.training_modules else "Unknown",
                "assessment_type": result.assessment_type.value,
                "score": result.score,
                "passed": result.passed,
                "completed_at": result.completed_at.isoformat()
            })
        
        # Competency overview
        competency_overview = {
            level.value: len([c for c, l in user_progress.competency_levels.items() if l == level])
            for level in CompetencyLevel
        }
        
        # CME credits summary
        ceu_summary = asdict(user_progress.ceu_credits)
        
        return {
            "user_info": {
                "user_id": user_progress.user_id,
                "name": user_progress.user_name,
                "facility": user_progress.user_facility,
                "role": user_progress.user_role,
                "enrollment_date": user_progress.enrollment_date.isoformat()
            },
            "current_enrollments": current_enrollments,
            "training_summary": {
                "total_training_hours": user_progress.total_training_hours,
                "modules_completed": len(user_progress.completed_modules),
                "assessments_taken": len(user_progress.assessment_scores),
                "certificates_earned": len(user_progress.certificates_earned)
            },
            "competency_overview": competency_overview,
            "ceu_credits": ceu_summary,
            "recent_activity": recent_activity,
            "recommendations": await self._generate_training_recommendations(user_progress)
        }
    
    async def generate_certification_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive certification report"""
        
        # Filter users by enrollment date
        period_users = [
            user for user in self.user_progress.values()
            if start_date <= user.enrollment_date <= end_date
        ]
        
        # Calculate metrics
        total_enrolled = len(period_users)
        completed_tracks = sum(
            1 for user in period_users
            for progress in user.certification_tracks.values()
            if progress["status"] == CompletionStatus.COMPLETED.value
        )
        
        # Track completion rates
        track_completion_rates = {}
        for track_id, track in self.certification_tracks.items():
            track_users = [
                user for user in period_users
                if track_id in user.certification_tracks
            ]
            
            completed_users = [
                user for user in track_users
                if user.certification_tracks[track_id]["status"] == CompletionStatus.COMPLETED.value
            ]
            
            completion_rate = (len(completed_users) / len(track_users) * 100) if track_users else 0
            track_completion_rates[track_id] = {
                "track_name": track.name,
                "enrolled": len(track_users),
                "completed": len(completed_users),
                "completion_rate": completion_rate
            }
        
        # Assessment statistics
        period_assessments = [
            result for result in self.assessment_results
            if start_date <= result.completed_at <= end_date
        ]
        
        assessment_stats = {
            "total_assessments": len(period_assessments),
            "pass_rate": len([r for r in period_assessments if r.passed]) / len(period_assessments) * 100 if period_assessments else 0,
            "average_score": statistics.mean([r.score for r in period_assessments]) if period_assessments else 0,
            "by_assessment_type": self._analyze_assessment_by_type(period_assessments)
        }
        
        # CME credit distribution
        total_cme_credits = sum(
            user.ceu_credits.total_credits for user in period_users
        )
        
        # Facility breakdown
        facility_stats = self._analyze_by_facility(period_users)
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_enrolled": total_enrolled,
                "completed_certifications": completed_tracks
            },
            "certification_tracks": track_completion_rates,
            "assessment_statistics": assessment_stats,
            "cme_credits": {
                "total_awarded": total_cme_credits,
                "average_per_user": total_cme_credits / total_enrolled if total_enrolled > 0 else 0
            },
            "facility_breakdown": facility_stats,
            "trends": {
                "enrollment_trend": await self._calculate_enrollment_trend(period_users),
                "completion_trend": await self._calculate_completion_trend(period_users),
                "popular_tracks": self._identify_popular_tracks(period_users)
            }
        }
    
    def _initialize_certification_tracks(self) -> None:
        """Initialize certification tracks"""
        
        # Healthcare Professional Certification
        healthcare_track = CertificationTrack(
            id="HC_PROF_001",
            name="Healthcare Professional Certification",
            description="Comprehensive certification for healthcare professionals using medical AI systems",
            target_audience="Physicians, Nurses, and Clinical Staff",
            modules=[
                TrainingModule.MEDICAL_AI_FUNDAMENTALS,
                TrainingModule.CLINICAL_INTEGRATION,
                TrainingModule.PATIENT_SAFETY,
                TrainingModule.REGULATORY_COMPLIANCE,
                TrainingModule.EMERGENCY_PROCEDURES
            ],
            total_duration_hours=40.0,
            validity_period_months=24,
            continuing_education_required=True,
            assessment_requirements=[
                AssessmentType.PRACTICAL_SIMULATION,
                AssessmentType.WRITTEN_EXAMINATION,
                AssessmentType.CASE_STUDY_ANALYSIS
            ],
            passing_score=85.0,
            created_date=datetime.now(),
            version="1.0"
        )
        
        # Administrator Certification
        admin_track = CertificationTrack(
            id="ADMIN_001",
            name="Healthcare Administrator Certification",
            description="Certification for healthcare administrators and IT support staff",
            target_audience="Administrators, IT Support, and Technical Staff",
            modules=[
                TrainingModule.SYSTEM_ADMINISTRATION,
                TrainingModule.USER_MANAGEMENT,
                TrainingModule.COMPLIANCE_MONITORING,
                TrainingModule.PERFORMANCE_OPTIMIZATION,
                TrainingModule.DATA_PROTECTION
            ],
            total_duration_hours=32.0,
            validity_period_months=24,
            continuing_education_required=True,
            assessment_requirements=[
                AssessmentType.HANDS_ON_LAB,
                AssessmentType.COMPETENCY_DEMONSTRATION,
                AssessmentType.PEER_EVALUATION
            ],
            passing_score=80.0,
            created_date=datetime.now(),
            version="1.0"
        )
        
        self.certification_tracks = {
            healthcare_track.id: healthcare_track,
            admin_track.id: admin_track
        }
    
    def _initialize_training_modules(self) -> None:
        """Initialize training modules"""
        
        # Medical AI Fundamentals Module
        ai_fundamentals = TrainingModuleContent(
            id="AI_FUND_001",
            module_type=TrainingModule.MEDICAL_AI_FUNDAMENTALS,
            title="Medical AI Fundamentals",
            description="Introduction to artificial intelligence in healthcare applications",
            duration_hours=8.0,
            difficulty_level=2,
            learning_objectives=[
                LearningObjective(
                    id="AI_01",
                    description="Understand basic AI concepts in healthcare",
                    competency_level=CompetencyLevel.BEGINNER,
                    assessment_criteria=["Define AI terminology", "Identify AI applications"],
                    practical_application="Apply AI concepts to clinical scenarios",
                    medical_context="General medical practice"
                ),
                LearningObjective(
                    id="AI_02",
                    description="Recognize AI benefits and limitations",
                    competency_level=CompetencyLevel.INTERMEDIATE,
                    assessment_criteria=["List AI benefits", "Identify limitations"],
                    practical_application="Make informed decisions about AI usage",
                    medical_context="Patient care scenarios"
                )
            ],
            prerequisites=[],
            content_sections=[
                {"title": "Introduction to AI", "duration_hours": 2},
                {"title": "AI in Healthcare", "duration_hours": 3},
                {"title": "Ethical Considerations", "duration_hours": 2},
                {"title": "AI Safety", "duration_hours": 1}
            ],
            assessment_questions=[
                {
                    "type": "multiple_choice",
                    "question": "What is artificial intelligence in healthcare?",
                    "options": ["A) Advanced calculator", "B) Machine learning for medical decisions", "C) Database system", "D) Network protocol"],
                    "correct_answer": "B"
                }
            ],
            practical_exercises=[
                {"title": "AI Use Case Analysis", "description": "Analyze real-world AI applications"}
            ],
            medical_specialties=["general_medicine"],
            target_roles=["physician", "nurse"],
            cme_eligible=True,
            last_updated=datetime.now()
        )
        
        # Patient Safety Module
        patient_safety = TrainingModuleContent(
            id="SAFE_001",
            module_type=TrainingModule.PATIENT_SAFETY,
            title="Patient Safety with Medical AI",
            description="Ensuring patient safety when using AI-powered medical systems",
            duration_hours=6.0,
            difficulty_level=3,
            learning_objectives=[
                LearningObjective(
                    id="SAFE_01",
                    description="Identify patient safety risks in AI systems",
                    competency_level=CompetencyLevel.INTERMEDIATE,
                    assessment_criteria=["List safety risks", "Assess risk levels"],
                    practical_application="Mitigate safety concerns in clinical practice",
                    medical_context="Emergency medicine, critical care"
                )
            ],
            prerequisites=["AI_FUND_001"],
            content_sections=[
                {"title": "AI Safety Principles", "duration_hours": 2},
                {"title": "Risk Assessment", "duration_hours": 2},
                {"title": "Emergency Protocols", "duration_hours": 2}
            ],
            assessment_questions=[],
            practical_exercises=[
                {"title": "Safety Scenario Simulation", "description": "Practice emergency response procedures"}
            ],
            medical_specialties=["emergency_medicine", "critical_care"],
            target_roles=["physician", "nurse"],
            cme_eligible=True,
            last_updated=datetime.now()
        )
        
        self.training_modules = {
            ai_fundamentals.id: ai_fundamentals,
            patient_safety.id: patient_safety
        }
    
    async def _award_cme_credits(self, user_id: str, module: TrainingModuleContent, credits: float) -> None:
        """Award CME credits for completed module"""
        
        user_progress = self.user_progress[user_id]
        
        # Categorize credits based on module type
        if module.module_type in [TrainingModule.CLINICAL_INTEGRATION, TrainingModule.MEDICAL_AI_FUNDAMENTALS]:
            user_progress.ceu_credits.clinical_training += credits
        elif module.module_type == TrainingModule.PATIENT_SAFETY:
            user_progress.ceu_credits.safety_training += credits
        elif module.module_type == TrainingModule.REGULATORY_COMPLIANCE:
            user_progress.ceu_credits.regulatory_training += credits
        elif module.module_type in [TrainingModule.SYSTEM_ADMINISTRATION, TrainingModule.DATA_PROTECTION]:
            user_progress.ceu_credits.technology_training += credits
        
        # Update total
        user_progress.ceu_credits.total_credits = (
            user_progress.ceu_credits.clinical_training +
            user_progress.ceu_credits.safety_training +
            user_progress.ceu_credits.regulatory_training +
            user_progress.ceu_credits.technology_training
        )
    
    async def _complete_certification_track(self, user_id: str, track_id: str) -> None:
        """Mark certification track as completed and award certificate"""
        
        user_progress = self.user_progress[user_id]
        track_progress = user_progress.certification_tracks[track_id]
        
        track_progress["status"] = CompletionStatus.PASSED.value
        
        # Generate certificate
        certificate_id = f"CERT-{user_id}-{track_id}-{datetime.now().strftime('%Y%m%d')}"
        user_progress.certificates_earned.append(certificate_id)
        
        logger.info(f"User {user_progress.user_name} completed certification track {track_id}")
    
    def _get_user_track_for_module(self, user_id: str, module_id: str) -> Optional[CertificationTrack]:
        """Get the certification track that contains the specified module"""
        
        user_progress = self.user_progress[user_id]
        
        for track_id, progress in user_progress.certification_tracks.items():
            track = self.certification_tracks[track_id]
            if module_id in [self.training_modules[mid].title for mid in track.modules]:
                return track
        
        return None
    
    async def _grade_written_exam(self, module: TrainingModuleContent, answers: Dict[str, Any]) -> float:
        """Grade written examination"""
        
        correct_answers = 0
        total_questions = len(module.assessment_questions)
        
        for question in module.assessment_questions:
            question_id = question.get("id", "")
            if question_id in answers and answers[question_id] == question.get("correct_answer"):
                correct_answers += 1
        
        return (correct_answers / total_questions * 100) if total_questions > 0 else 0.0
    
    async def _grade_practical_simulation(self, module: TrainingModuleContent, answers: Dict[str, Any]) -> float:
        """Grade practical simulation assessment"""
        
        # Simplified grading for practical simulations
        # In production, this would involve more sophisticated evaluation
        
        base_score = 70.0
        performance_bonus = answers.get("performance_rating", 0) * 5
        safety_bonus = answers.get("safety_compliance", False) * 10
        
        return min(base_score + performance_bonus + safety_bonus, 100.0)
    
    async def _grade_case_study(self, module: TrainingModuleContent, answers: Dict[str, Any]) -> float:
        """Grade case study analysis"""
        
        # Evaluate case study responses
        analysis_quality = answers.get("analysis_quality", 0)  # 0-1 scale
        decision_rationale = answers.get("decision_rationale", 0)  # 0-1 scale
        safety_consideration = answers.get("safety_consideration", 0)  # 0-1 scale
        
        score = (analysis_quality + decision_rationale + safety_consideration) * 33.33
        return min(score, 100.0)
    
    async def _grade_assessment(self, module: TrainingModuleContent, assessment_type: AssessmentType, answers: Dict[str, Any]) -> float:
        """Generic assessment grading"""
        
        # Default grading for other assessment types
        completion_rate = answers.get("completion_percentage", 0)  # 0-100
        quality_rating = answers.get("quality_rating", 0)  # 0-1
        
        return min(completion_rate * 0.7 + quality_rating * 30, 100.0)
    
    async def _assess_competencies(self, module: TrainingModuleContent, score: float, answers: Dict[str, Any]) -> List[str]:
        """Assess demonstrated competencies"""
        
        competencies = []
        
        # High score indicates competency achievement
        if score >= 80:
            for objective in module.learning_objectives:
                competencies.append(objective.id)
        
        return competencies
    
    async def _generate_training_recommendations(self, user_progress: UserProgress) -> List[str]:
        """Generate personalized training recommendations"""
        
        recommendations = []
        
        # Recommend based on competency levels
        if CompetencyLevel.BEGINNER in user_progress.competency_levels.values():
            recommendations.append("Consider taking advanced modules to build expertise")
        
        # Recommend based on incomplete tracks
        for track_id, progress in user_progress.certification_tracks.items():
            if progress["status"] == CompletionStatus.IN_PROGRESS.value:
                if progress["completion_percentage"] < 50:
                    recommendations.append(f"Continue your {self.certification_tracks[track_id].name} certification")
        
        # Recommend continuing education
        ceu_credits = user_progress.ceu_credits.total_credits
        if ceu_credits < 20:  # Less than typical annual requirement
            recommendations.append("Complete additional CME-eligible modules to meet annual requirements")
        
        return recommendations
    
    def _analyze_assessment_by_type(self, assessments: List[AssessmentResult]) -> Dict[str, Dict[str, Any]]:
        """Analyze assessment performance by type"""
        
        analysis = {}
        
        for assessment_type in AssessmentType:
            type_assessments = [a for a in assessments if a.assessment_type == assessment_type]
            
            if type_assessments:
                analysis[assessment_type.value] = {
                    "count": len(type_assessments),
                    "pass_rate": len([a for a in type_assessments if a.passed]) / len(type_assessments) * 100,
                    "average_score": statistics.mean([a.score for a in type_assessments])
                }
        
        return analysis
    
    def _analyze_by_facility(self, users: List[UserProgress]) -> Dict[str, Dict[str, Any]]:
        """Analyze certification metrics by healthcare facility"""
        
        facility_stats = defaultdict(lambda: {"enrolled": 0, "completed": 0, "total_hours": 0})
        
        for user in users:
            facility = user.user_facility
            facility_stats[facility]["enrolled"] += 1
            facility_stats[facility]["total_hours"] += user.total_training_hours
            
            # Count completed certifications
            for progress in user.certification_tracks.values():
                if progress["status"] == CompletionStatus.COMPLETED.value:
                    facility_stats[facility]["completed"] += 1
        
        return dict(facility_stats)
    
    async def _calculate_enrollment_trend(self, users: List[UserProgress]) -> Dict[str, Any]:
        """Calculate enrollment trends over time"""
        
        # Group by month
        monthly_enrollment = defaultdict(int)
        
        for user in users:
            month_key = user.enrollment_date.strftime("%Y-%m")
            monthly_enrollment[month_key] += 1
        
        return dict(monthly_enrollment)
    
    async def _calculate_completion_trend(self, users: List[UserProgress]) -> Dict[str, Any]:
        """Calculate completion trends over time"""
        
        monthly_completion = defaultdict(int)
        
        for user in users:
            for progress in user.certification_tracks.values():
                if progress["status"] == CompletionStatus.COMPLETED.value:
                    month_key = progress["enrolled_date"][:7]  # Extract YYYY-MM
                    monthly_completion[month_key] += 1
        
        return dict(monthly_completion)
    
    def _identify_popular_tracks(self, users: List[UserProgress]) -> List[Dict[str, Any]]:
        """Identify most popular certification tracks"""
        
        track_popularity = defaultdict(int)
        
        for user in users:
            for track_id in user.certification_tracks.keys():
                track_popularity[track_id] += 1
        
        return [
            {
                "track_id": track_id,
                "track_name": self.certification_tracks[track_id].name,
                "enrolled_count": count
            }
            for track_id, count in sorted(track_popularity.items(), key=lambda x: x[1], reverse=True)
        ]

# Global training system instance
training_system = MedicalEducationSystem()

# Example usage and testing functions
async def setup_sample_training():
    """Set up sample training system"""
    
    # Enroll a sample user
    user_progress = await training_system.enroll_user_in_track(
        user_id="dr_smith_001",
        user_name="Dr. Sarah Smith",
        user_facility="General Hospital",
        user_role="Physician",
        certification_track_id="HC_PROF_001"
    )
    
    print(f"Enrolled Dr. Sarah Smith in Healthcare Professional Certification")
    
    # Start first module
    start_result = await training_system.start_module(
        user_id="dr_smith_001",
        certification_track_id="HC_PROF_001",
        module_id="AI_FUND_001"
    )
    
    print(f"Started module: {start_result['module_title']}")
    
    # Complete module
    await training_system.complete_module(
        user_id="dr_smith_001",
        certification_track_id="HC_PROF_001",
        module_id="AI_FUND_001",
        completion_data={"time_spent": 8.0, "engagement_score": 0.85}
    )
    
    # Submit assessment
    assessment_result = await training_system.submit_assessment(
        user_id="dr_smith_001",
        module_id="AI_FUND_001",
        assessment_type=AssessmentType.WRITTEN_EXAMINATION,
        answers={"AI_01": "B", "comprehension": 0.8},
        time_taken_minutes=45
    )
    
    print(f"Assessment result: {assessment_result.score:.1f}% ({'passed' if assessment_result.passed else 'failed'})")
    
    # Get user dashboard
    dashboard = await training_system.get_user_dashboard("dr_smith_001")
    print(f"User dashboard: {dashboard['training_summary']}")

if __name__ == "__main__":
    asyncio.run(setup_sample_training())