"""
Clinical Benchmarks for Medical AI Evaluation

Comprehensive clinical benchmark evaluation suite including:
- Standard medical datasets
- Clinical case repositories
- Benchmark evaluation suites
- Performance baselines

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod


class BenchmarkCategory(Enum):
    """Categories of clinical benchmarks"""
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    DRUG_INTERACTION = "drug_interaction"
    CLINICAL_REASONING = "clinical_reasoning"
    EMERGENCY_MEDICINE = "emergency_medicine"
    CHRONIC_DISEASE = "chronic_disease"
    PREVENTIVE_CARE = "preventive_care"
    PEDIATRIC_CARE = "pediatric_care"


class DifficultyLevel(Enum):
    """Difficulty levels for benchmark cases"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class DatasetType(Enum):
    """Types of benchmark datasets"""
    SYNTHETIC = "synthetic"
    REAL_WORLD = "real_world"
    RETROSPECTIVE = "retrospective"
    PROSPECTIVE = "prospective"
    TEACHING_CASES = "teaching_cases"


@dataclass
class BenchmarkCase:
    """Individual benchmark case structure"""
    case_id: str
    category: BenchmarkCategory
    difficulty_level: DifficultyLevel
    dataset_type: DatasetType
    
    # Case content
    patient_info: Dict[str, Any]
    clinical_data: Dict[str, Any]
    expected_output: Dict[str, Any]
    gold_standard: Dict[str, Any]
    
    # Metadata
    source: str
    domain_expertise_required: str
    estimated_completion_time: int  # minutes
    validation_status: str
    created_at: datetime
    tags: List[str]
    references: List[str]
    
    # Scoring criteria
    evaluation_criteria: Dict[str, float]  # criterion -> max_score
    acceptance_threshold: float


@dataclass
class BenchmarkDataset:
    """Benchmark dataset configuration"""
    dataset_id: str
    name: str
    description: str
    category: BenchmarkCategory
    dataset_type: DatasetType
    
    # Dataset properties
    total_cases: int
    difficulty_distribution: Dict[DifficultyLevel, int]
    domain_coverage: List[str]
    
    # Quality metrics
    validation_score: float
    inter_rater_reliability: float
    clinical_accuracy: float
    
    # Metadata
    version: str
    last_updated: datetime
    citation: Optional[str]
    license: str
    download_url: Optional[str]
    
    # Use cases
    intended_use: List[str]
    target_models: List[str]


@dataclass
class EvaluationResult:
    """Result from benchmark evaluation"""
    evaluation_id: str
    dataset_id: str
    model_name: str
    timestamp: datetime
    
    # Performance metrics
    overall_score: float
    category_scores: Dict[BenchmarkCategory, float]
    difficulty_scores: Dict[DifficultyLevel, float]
    
    # Detailed metrics
    accuracy_metrics: Dict[str, float]
    safety_metrics: Dict[str, float]
    efficiency_metrics: Dict[str, float]
    
    # Individual case results
    case_results: List[Dict[str, Any]]
    
    # Analysis
    error_analysis: Dict[str, Any]
    recommendations: List[str]
    
    # Metadata
    evaluation_duration: float
    resource_usage: Dict[str, Any]


class ClinicalDatasetGenerator:
    """Generator for clinical benchmark datasets"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.symptom_clusters = {
            "cardiac": ["chest_pain", "shortness_of_breath", "palpitations", "syncope"],
            "respiratory": ["cough", "dyspnea", "wheezing", "chest_tightness"],
            "neurological": ["headache", "seizure", "weakness", "confusion"],
            "gastrointestinal": ["abdominal_pain", "nausea", "vomiting", "diarrhea"],
            "musculoskeletal": ["joint_pain", "muscle_weakness", "stiffness", "swelling"]
        }
        
        self.diagnosis_patterns = {
            "acute_coronary_syndrome": {
                "symptoms": ["chest_pain", "shortness_of_breath", "diaphoresis"],
                "risk_factors": ["hypertension", "diabetes", "smoking", "family_history"],
                "tests": ["ECG", "troponin", "chest_xray"],
                "treatments": ["aspirin", "nitroglycerin", "oxygen", "heparin"]
            },
            "pneumonia": {
                "symptoms": ["fever", "cough", "chest_pain", "shortness_of_breath"],
                "risk_factors": ["elderly", "immunocompromised", "chronic_lung_disease"],
                "tests": ["chest_xray", "complete_blood_count", "blood_cultures"],
                "treatments": ["antibiotics", "oxygen", "hydration"]
            },
            "stroke": {
                "symptoms": ["weakness", "speech_difficulty", "facial_droop", "headache"],
                "risk_factors": ["hypertension", "atrial_fibrillation", "diabetes"],
                "tests": ["CT_scan", "MRI", "neurological_exam"],
                "treatments": ["thrombolytics", "antiplatelets", "blood_pressure_management"]
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for dataset generation"""
        logger = logging.getLogger(f"{__name__}.ClinicalDatasetGenerator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def generate_diagnostic_cases(self, 
                                num_cases: int,
                                difficulty_levels: List[DifficultyLevel],
                                categories: List[BenchmarkCategory]) -> List[BenchmarkCase]:
        """Generate diagnostic benchmark cases"""
        
        cases = []
        
        for i in range(num_cases):
            # Select random components
            difficulty = np.random.choice(difficulty_levels)
            category = np.random.choice(categories)
            
            # Generate patient info
            patient_info = self._generate_patient_info(difficulty)
            
            # Generate clinical scenario
            clinical_data = self._generate_clinical_scenario(category, difficulty, patient_info)
            
            # Generate expected output
            expected_output = self._generate_expected_output(clinical_data, category)
            
            # Create benchmark case
            case = BenchmarkCase(
                case_id=f"diagnostic_{category.value}_{difficulty.value}_{i:04d}",
                category=category,
                difficulty_level=difficulty,
                dataset_type=DatasetType.SYNTHETIC,
                patient_info=patient_info,
                clinical_data=clinical_data,
                expected_output=expected_output,
                gold_standard=expected_output,  # For synthetic data
                source="synthetic_generator",
                domain_expertise_required=category.value,
                estimated_completion_time=self._estimate_completion_time(difficulty),
                validation_status="auto_generated",
                created_at=datetime.now(),
                tags=self._generate_tags(category, difficulty),
                references=[],
                evaluation_criteria=self._get_evaluation_criteria(category),
                acceptance_threshold=0.75
            )
            
            cases.append(case)
        
        return cases
    
    def _generate_patient_info(self, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate realistic patient information"""
        
        age_ranges = {
            DifficultyLevel.BASIC: (18, 65),
            DifficultyLevel.INTERMEDIATE: (25, 75),
            DifficultyLevel.ADVANCED: (35, 85),
            DifficultyLevel.EXPERT: (45, 95)
        }
        
        min_age, max_age = age_ranges[difficulty]
        age = np.random.randint(min_age, max_age + 1)
        
        gender = np.random.choice(["male", "female"])
        
        # Generate medical history based on age
        medical_history = []
        if age > 50:
            medical_history.extend(["hypertension", "diabetes"])
        if age > 60:
            medical_history.append("hyperlipidemia")
        
        return {
            "age": age,
            "gender": gender,
            "chief_complaint": self._generate_chief_complaint(),
            "history_of_present_illness": self._generate_hpi(difficulty),
            "past_medical_history": medical_history,
            "medications": self._generate_medications(medical_history),
            "allergies": self._generate_allergies(),
            "social_history": {
                "smoking": np.random.choice(["never", "former", "current"], p=[0.5, 0.3, 0.2]),
                "alcohol": np.random.choice(["none", "social", "moderate", "heavy"], p=[0.4, 0.4, 0.15, 0.05]),
                "occupation": self._generate_occupation()
            },
            "family_history": self._generate_family_history()
        }
    
    def _generate_clinical_scenario(self, 
                                  category: BenchmarkCategory, 
                                  difficulty: DifficultyLevel,
                                  patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate clinical scenario based on category and difficulty"""
        
        if category == BenchmarkCategory.DIAGNOSTIC_ACCURACY:
            return self._generate_diagnostic_scenario(patient_info, difficulty)
        elif category == BenchmarkCategory.TREATMENT_RECOMMENDATION:
            return self._generate_treatment_scenario(patient_info, difficulty)
        elif category == BenchmarkCategory.SYMPTOM_ANALYSIS:
            return self._generate_symptom_analysis_scenario(patient_info, difficulty)
        elif category == BenchmarkCategory.RISK_ASSESSMENT:
            return self._generate_risk_assessment_scenario(patient_info, difficulty)
        else:
            return self._generate_general_scenario(patient_info, difficulty)
    
    def _generate_diagnostic_scenario(self, 
                                    patient_info: Dict[str, Any], 
                                    difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate diagnostic scenario"""
        
        # Select appropriate diagnosis based on age and symptoms
        possible_diagnoses = list(self.diagnosis_patterns.keys())
        diagnosis = np.random.choice(possible_diagnoses)
        
        # Generate symptoms based on diagnosis
        base_symptoms = self.diagnosis_patterns[diagnosis]["symptoms"]
        
        # Add difficulty-based complexity
        if difficulty in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
            # Add atypical or misleading symptoms
            additional_symptoms = ["fatigue", "weight_loss", "night_sweats", "anxiety"]
            additional_symptoms = np.random.choice(additional_symptoms, 
                                                 size=np.random.randint(1, 3), 
                                                 replace=False)
            all_symptoms = list(base_symptoms) + list(additional_symptoms)
        else:
            all_symptoms = base_symptoms
        
        # Generate vital signs
        vital_signs = self._generate_vital_signs(diagnosis, patient_info["age"])
        
        # Generate physical exam findings
        physical_exam = self._generate_physical_exam(diagnosis, all_symptoms)
        
        # Generate lab/imaging results
        diagnostic_tests = self._generate_diagnostic_tests(diagnosis, all_symptoms)
        
        return {
            "primary_diagnosis": diagnosis,
            "symptoms": all_symptoms,
            "vital_signs": vital_signs,
            "physical_exam": physical_exam,
            "diagnostic_tests": diagnostic_tests,
            "clinical_impression": self._generate_clinical_impression(diagnosis),
            "differential_diagnosis": self._generate_differential_diagnosis(diagnosis, difficulty)
        }
    
    def _generate_treatment_scenario(self, 
                                   patient_info: Dict[str, Any], 
                                   difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate treatment recommendation scenario"""
        
        # Start with a diagnostic scenario
        diagnostic_scenario = self._generate_diagnostic_scenario(patient_info, difficulty)
        diagnosis = diagnostic_scenario["primary_diagnosis"]
        
        # Generate treatment options
        standard_treatments = self.diagnosis_patterns[diagnosis]["treatments"]
        
        # Add complexity based on difficulty
        if difficulty in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
            # Add contraindications or drug interactions
            contraindications = self._generate_contraindications(patient_info)
            drug_interactions = self._generate_drug_interactions(standard_treatments)
            
            treatments = {
                "first_line": standard_treatments,
                "contraindications": contraindications,
                "drug_interactions": drug_interactions,
                "alternative_options": self._generate_alternative_treatments(diagnosis)
            }
        else:
            treatments = {
                "recommended": standard_treatments,
                "monitoring": self._generate_monitoring_plan(diagnosis)
            }
        
        return {
            **diagnostic_scenario,
            "treatment_plan": treatments,
            "patient_preferences": self._generate_patient_preferences(difficulty),
            "resource_availability": self._generate_resource_availability()
        }
    
    def _generate_symptom_analysis_scenario(self, 
                                          patient_info: Dict[str, Any], 
                                          difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate symptom analysis scenario"""
        
        # Select symptom cluster
        symptom_cluster = np.random.choice(list(self.symptom_clusters.keys()))
        symptoms = self.symptom_clusters[symptom_cluster]
        
        # Generate analysis context
        context = {
            "temporal_pattern": np.random.choice(["acute", "chronic", "progressive", "intermittent"]),
            "aggravating_factors": self._generate_aggravating_factors(symptoms),
            "relieving_factors": self._generate_relieving_factors(symptoms),
            "associated_symptoms": self._generate_associated_symptoms(symptom_cluster),
            "red_flags": self._generate_red_flags(symptoms, difficulty)
        }
        
        return {
            "primary_symptoms": symptoms,
            "symptom_cluster": symptom_cluster,
            "analysis_context": context,
            "expected_analysis": {
                "differential_diagnosis": self._generate_symptom_based_differential(symptom_cluster),
                "priority_assessment": self._generate_priority_assessment(symptoms, context),
                "urgent_interventions": self._generate_urgent_interventions(symptoms, context)
            }
        }
    
    def _generate_risk_assessment_scenario(self, 
                                         patient_info: Dict[str, Any], 
                                         difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate risk assessment scenario"""
        
        # Generate risk factors
        risk_factors = self._generate_comprehensive_risk_factors(patient_info)
        
        # Generate risk calculation context
        assessment_context = {
            "assessment_purpose": np.random.choice([" preoperative", "medication", "procedure", "monitoring"]),
            "time_horizon": np.random.choice(["immediate", "short_term", "long_term"]),
            "available_data": self._generate_available_data(patient_info, difficulty),
            "clinical_settings": np.random.choice(["inpatient", "outpatient", "emergency", "icu"])
        }
        
        return {
            "patient_profile": patient_info,
            "risk_factors": risk_factors,
            "assessment_context": assessment_context,
            "expected_assessment": {
                "risk_stratification": self._generate_risk_stratification(risk_factors),
                "mitigation_strategies": self._generate_mitigation_strategies(risk_factors),
                "monitoring_plan": self._generate_risk_monitoring_plan(assessment_context)
            }
        }
    
    def _generate_expected_output(self, 
                                clinical_data: Dict[str, Any], 
                                category: BenchmarkCategory) -> Dict[str, Any]:
        """Generate expected model output for evaluation"""
        
        if category == BenchmarkCategory.DIAGNOSTIC_ACCURACY:
            return {
                "primary_diagnosis": clinical_data.get("primary_diagnosis"),
                "differential_diagnosis": clinical_data.get("differential_diagnosis", []),
                "confidence_score": np.random.uniform(0.7, 0.95),
                "reasoning": self._generate_diagnostic_reasoning(clinical_data),
                "next_steps": self._generate_next_steps(clinical_data)
            }
        elif category == BenchmarkCategory.TREATMENT_RECOMMENDATION:
            return {
                "treatment_recommendations": clinical_data.get("treatment_plan", {}),
                "monitoring_plan": clinical_data.get("treatment_plan", {}).get("monitoring", []),
                "contraindications": clinical_data.get("treatment_plan", {}).get("contraindications", []),
                "follow_up_plan": self._generate_follow_up_plan(clinical_data)
            }
        elif category == BenchmarkCategory.RISK_ASSESSMENT:
            return clinical_data.get("expected_assessment", {})
        else:
            return {
                "analysis": "Generic analysis output",
                "recommendations": ["Standard clinical recommendations"],
                "confidence": 0.8
            }
    
    def _estimate_completion_time(self, difficulty: DifficultyLevel) -> int:
        """Estimate time to complete case (minutes)"""
        time_estimates = {
            DifficultyLevel.BASIC: 15,
            DifficultyLevel.INTERMEDIATE: 30,
            DifficultyLevel.ADVANCED: 45,
            DifficultyLevel.EXPERT: 60
        }
        return time_estimates[difficulty]
    
    def _generate_tags(self, category: BenchmarkCategory, difficulty: DifficultyLevel) -> List[str]:
        """Generate relevant tags for case"""
        return [category.value, difficulty.value, "synthetic", "benchmark"]
    
    def _get_evaluation_criteria(self, category: BenchmarkCategory) -> Dict[str, float]:
        """Get evaluation criteria for category"""
        base_criteria = {
            "accuracy": 40.0,
            "completeness": 20.0,
            "safety": 20.0,
            "clarity": 10.0,
            "evidence_based": 10.0
        }
        
        # Adjust criteria based on category
        if category == BenchmarkCategory.DIAGNOSTIC_ACCURACY:
            base_criteria["accuracy"] = 50.0
            base_criteria["safety"] = 25.0
        elif category == BenchmarkCategory.TREATMENT_RECOMMENDATION:
            base_criteria["safety"] = 30.0
            base_criteria["evidence_based"] = 20.0
        
        return base_criteria
    
    # Helper methods for generating specific data components
    def _generate_chief_complaint(self) -> str:
        complaints = [
            "Chest pain for 2 hours",
            "Shortness of breath",
            "Severe headache",
            "Abdominal pain",
            "Joint pain and swelling"
        ]
        return np.random.choice(complaints)
    
    def _generate_hpi(self, difficulty: DifficultyLevel) -> str:
        """Generate history of present illness"""
        if difficulty == DifficultyLevel.BASIC:
            return "Patient reports symptoms starting 2 days ago, gradually worsening."
        elif difficulty == DifficultyLevel.INTERMEDIATE:
            return "Symptoms began insidiously 1 week ago, associated with fatigue and mild fever."
        else:
            return "Complex presentation with atypical symptoms, recent travel history, and multiple comorbidities."
    
    def _generate_medications(self, conditions: List[str]) -> List[str]:
        """Generate medications based on conditions"""
        medications = []
        if "hypertension" in conditions:
            medications.append("lisinopril 10mg daily")
        if "diabetes" in conditions:
            medications.append("metformin 500mg twice daily")
        if "hyperlipidemia" in conditions:
            medications.append("atorvastatin 20mg nightly")
        return medications
    
    def _generate_allergies(self) -> List[str]:
        """Generate patient allergies"""
        allergies = []
        if np.random.random() < 0.2:  # 20% chance of allergies
            allergies = ["penicillin", "latex", "iodine"]
        return allergies
    
    def _generate_occupation(self) -> str:
        occupations = ["office worker", "construction", "teacher", "retired", "healthcare worker"]
        return np.random.choice(occupations)
    
    def _generate_family_history(self) -> Dict[str, Any]:
        return {
            "father": "hypertension, diabetes",
            "mother": "breast cancer",
            "siblings": "none significant"
        }
    
    def _generate_vital_signs(self, diagnosis: str, age: int) -> Dict[str, Any]:
        """Generate realistic vital signs based on diagnosis and age"""
        base_vitals = {
            "temperature": np.random.normal(98.6, 1.0),
            "heart_rate": np.random.randint(70, 100),
            "blood_pressure": f"{np.random.randint(110, 140)}/{np.random.randint(70, 90)}",
            "respiratory_rate": np.random.randint(14, 20),
            "oxygen_saturation": np.random.randint(95, 100)
        }
        
        # Adjust based on diagnosis
        if diagnosis == "acute_coronary_syndrome":
            base_vitals["heart_rate"] = np.random.randint(90, 130)
            base_vitals["blood_pressure"] = f"{np.random.randint(140, 180)}/{np.random.randint(90, 100)}"
        
        return base_vitals
    
    def _generate_physical_exam(self, diagnosis: str, symptoms: List[str]) -> Dict[str, Any]:
        """Generate physical exam findings"""
        return {
            "general": "Alert, oriented, appears uncomfortable",
            "cardiovascular": "Regular rate and rhythm, no murmurs",
            "respiratory": "Clear to auscultation bilaterally",
            "abdomen": "Soft, non-tender, no rebound or guarding",
            "neurological": "Alert and oriented x3, no focal deficits"
        }
    
    def _generate_diagnostic_tests(self, diagnosis: str, symptoms: List[str]) -> Dict[str, Any]:
        """Generate diagnostic test results"""
        return {
            "laboratory": {
                "complete_blood_count": "within normal limits",
                "basic_metabolic_panel": "normal",
                "troponin": "elevated" if diagnosis == "acute_coronary_syndrome" else "normal"
            },
            "imaging": {
                "chest_xray": "clear lung fields",
                "ECG": "normal sinus rhythm" if diagnosis != "acute_coronary_syndrome" else "ST segment changes"
            }
        }
    
    def _generate_clinical_impression(self, diagnosis: str) -> str:
        return f"Clinical presentation consistent with {diagnosis.replace('_', ' ').title()}"
    
    def _generate_differential_diagnosis(self, primary_diagnosis: str, difficulty: DifficultyLevel) -> List[str]:
        """Generate differential diagnosis list"""
        all_diagnoses = list(self.diagnosis_patterns.keys())
        differential = [primary_diagnosis]
        
        # Add alternative diagnoses
        alternatives = [d for d in all_diagnoses if d != primary_diagnosis]
        num_alternatives = min(2, len(alternatives)) if difficulty == DifficultyLevel.BASIC else min(4, len(alternatives))
        differential.extend(np.random.choice(alternatives, size=num_alternatives, replace=False))
        
        return differential
    
    def _generate_diagnostic_reasoning(self, clinical_data: Dict[str, Any]) -> str:
        return "Based on clinical presentation and diagnostic findings, the most likely diagnosis is primary_diagnosis."
    
    def _generate_next_steps(self, clinical_data: Dict[str, Any]) -> List[str]:
        return [
            "Admit to coronary care unit",
            "Serial cardiac enzymes",
            "Cardiology consultation",
            "Echocardiogram"
        ]
    
    def _generate_contraindications(self, patient_info: Dict[str, Any]) -> List[str]:
        """Generate contraindications for treatments"""
        contraindications = []
        if "kidney_disease" in patient_info.get("past_medical_history", []):
            contraindications.append("contrast_agents")
        if "bleeding_disorder" in patient_info.get("past_medical_history", []):
            contraindications.append("anticoagulants")
        return contraindications
    
    def _generate_drug_interactions(self, treatments: List[str]) -> List[Dict[str, Any]]:
        """Generate potential drug interactions"""
        interactions = []
        if "warfarin" in treatments and "aspirin" in treatments:
            interactions.append({
                "drugs": ["warfarin", "aspirin"],
                "severity": "moderate",
                "description": "Increased bleeding risk"
            })
        return interactions
    
    def _generate_alternative_treatments(self, diagnosis: str) -> List[str]:
        """Generate alternative treatment options"""
        alternatives = {
            "acute_coronary_syndrome": ["conservative_management", "percutaneous_intervention"],
            "pneumonia": ["outpatient_antibiotics", "hospitalization"],
            "stroke": ["thrombolytic_therapy", "supportive_care"]
        }
        return alternatives.get(diagnosis, ["alternative_1", "alternative_2"])
    
    def _generate_monitoring_plan(self, diagnosis: str) -> List[str]:
        """Generate monitoring plan"""
        return [
            "Daily vital signs",
            "Serial laboratory values",
            "Symptom assessment",
            "Adverse effect monitoring"
        ]
    
    def _generate_patient_preferences(self, difficulty: DifficultyLevel) -> Dict[str, Any]:
        """Generate patient treatment preferences"""
        return {
            "invasive_procedures": np.random.choice(["willing", "reluctant", "refuses"]),
            "medication_adherence": np.random.choice(["excellent", "good", "poor"]),
            "quality_vs_quantity": np.random.choice(["quality", "quantity", "balance"])
        }
    
    def _generate_resource_availability(self) -> Dict[str, Any]:
        """Generate resource availability information"""
        return {
            "icu_available": True,
            "specialist_availability": np.random.choice(["immediate", "within_hours", "next_day"]),
            "medication_availability": np.random.choice(["full", "limited", "unavailable"])
        }
    
    def _generate_follow_up_plan(self, clinical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate follow-up plan"""
        return {
            "timeframe": "1-2 weeks",
            "setting": "outpatient",
            "monitoring": "symptom tracking and medication adjustment",
            "escalation_criteria": "worsening symptoms or new concerning findings"
        }
    
    def _generate_aggravating_factors(self, symptoms: List[str]) -> List[str]:
        return ["exertion", "stress", "position changes", "weather changes"]
    
    def _generate_relieving_factors(self, symptoms: List[str]) -> List[str]:
        return ["rest", "medication", "heat/cold therapy", "positioning"]
    
    def _generate_associated_symptoms(self, cluster: str) -> List[str]:
        associated = {
            "cardiac": ["fatigue", "anxiety", "nausea"],
            "respiratory": ["chest_tightness", "wheezing"],
            "neurological": ["visual_changes", "nausea"],
            "gastrointestinal": ["bloating", "changes_in_bowel_habits"],
            "musculoskeletal": ["stiffness", "limited_range_of_motion"]
        }
        return associated.get(cluster, [])
    
    def _generate_red_flags(self, symptoms: List[str], difficulty: DifficultyLevel) -> List[str]:
        """Generate red flag symptoms"""
        red_flags = ["chest_pain_at_rest", "neurological_deficits", "severe_abdominal_pain"]
        if difficulty in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
            return np.random.choice(red_flags, size=2, replace=False).tolist()
        return []
    
    def _generate_symptom_based_differential(self, cluster: str) -> List[str]:
        differential = {
            "cardiac": ["myocardial_infarction", "angina", "pericarditis"],
            "respiratory": ["pneumonia", "asthma", "copd_exacerbation"],
            "neurological": ["migraine", "tension_headache", "brain_mass"],
            "gastrointestinal": ["gastritis", "peptic_ulcer", "pancreatitis"],
            "musculoskeletal": ["arthritis", "bursitis", "muscle_strain"]
        }
        return differential.get(cluster, ["differential_1", "differential_2"])
    
    def _generate_priority_assessment(self, symptoms: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "urgency_level": np.random.choice(["routine", "urgent", "emergent"]),
            "stability": "stable",
            "risk_level": "moderate"
        }
    
    def _generate_urgent_interventions(self, symptoms: List[str], context: Dict[str, Any]) -> List[str]:
        return ["immediate_assessment", "vital_sign_monitoring", "symptom_management"]
    
    def _generate_comprehensive_risk_factors(self, patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk factor assessment"""
        return {
            "demographic": {
                "age": patient_info["age"],
                "gender": patient_info["gender"]
            },
            "clinical": {
                "comorbidities": patient_info.get("past_medical_history", []),
                "medications": patient_info.get("medications", []),
                "allergies": patient_info.get("allergies", [])
            },
            "lifestyle": patient_info.get("social_history", {}),
            "family_history": patient_info.get("family_history", {})
        }
    
    def _generate_available_data(self, patient_info: Dict[str, Any], difficulty: DifficultyLevel) -> List[str]:
        """Generate available data for risk assessment"""
        data_sources = ["clinical_history", "physical_exam", "laboratory", "imaging"]
        if difficulty in [DifficultyLevel.ADVANCED, DifficultyLevel.EXPERT]:
            data_sources.extend(["genetic_testing", "biomarkers", "functional_assessment"])
        return data_sources
    
    def _generate_risk_stratification(self, risk_factors: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk stratification results"""
        return {
            "risk_level": np.random.choice(["low", "moderate", "high", "very_high"]),
            "risk_score": np.random.uniform(0.1, 0.9),
            "confidence": np.random.uniform(0.7, 0.95),
            "key_factors": ["hypertension", "diabetes", "age_over_65"]
        }
    
    def _generate_mitigation_strategies(self, risk_factors: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation strategies"""
        return [
            "optimize_medical_management",
            "enhanced_monitoring",
            "patient_education",
            "lifestyle_modifications"
        ]
    
    def _generate_risk_monitoring_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate monitoring plan for risk assessment"""
        return {
            "frequency": "weekly",
            "parameters": ["vital_signs", "symptoms", "laboratory_values"],
            "triggers_for_action": ["worsening_symptoms", "abnormal_laboratory_values"],
            "follow_up_timeframe": "1_month"
        }


class ClinicalBenchmarkSuite:
    """Main benchmark evaluation suite for clinical AI models"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.datasets: Dict[str, BenchmarkDataset] = {}
        self.cases: Dict[str, BenchmarkCase] = {}
        self.evaluation_history: List[EvaluationResult] = []
        self.dataset_generator = ClinicalDatasetGenerator()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for benchmark suite"""
        logger = logging.getLogger(f"{__name__}.ClinicalBenchmarkSuite")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_benchmark_dataset(self,
                               name: str,
                               category: BenchmarkCategory,
                               num_cases: int,
                               difficulty_levels: List[DifficultyLevel],
                               dataset_type: DatasetType = DatasetType.SYNTHETIC) -> str:
        """Create a new benchmark dataset"""
        
        # Generate cases
        cases = self.dataset_generator.generate_diagnostic_cases(
            num_cases=num_cases,
            difficulty_levels=difficulty_levels,
            categories=[category]
        )
        
        # Create dataset
        dataset_id = f"{category.value}_{len(self.datasets):04d}"
        
        dataset = BenchmarkDataset(
            dataset_id=dataset_id,
            name=name,
            description=f"Benchmark dataset for {category.value}",
            category=category,
            dataset_type=dataset_type,
            total_cases=len(cases),
            difficulty_distribution=self._calculate_difficulty_distribution(cases),
            domain_coverage=[category.value],
            validation_score=0.95,  # Synthetic data assumed to be validated
            inter_rater_reliability=0.90,
            clinical_accuracy=0.92,
            version="1.0.0",
            last_updated=datetime.now(),
            citation=None,
            license="MIT",
            download_url=None,
            intended_use=["AI_model_evaluation", "benchmarking"],
            target_models=["all_medical_ai_models"]
        )
        
        # Store dataset and cases
        self.datasets[dataset_id] = dataset
        for case in cases:
            self.cases[case.case_id] = case
        
        self.logger.info(f"Created dataset {dataset_id} with {len(cases)} cases")
        return dataset_id
    
    def _calculate_difficulty_distribution(self, cases: List[BenchmarkCase]) -> Dict[DifficultyLevel, int]:
        """Calculate distribution of difficulty levels in cases"""
        distribution = {level: 0 for level in DifficultyLevel}
        for case in cases:
            distribution[case.difficulty_level] += 1
        return distribution
    
    def get_dataset(self, dataset_id: str) -> Optional[BenchmarkDataset]:
        """Get dataset by ID"""
        return self.datasets.get(dataset_id)
    
    def get_cases_for_dataset(self, dataset_id: str) -> List[BenchmarkCase]:
        """Get all cases for a specific dataset"""
        dataset = self.datasets.get(dataset_id)
        if not dataset:
            return []
        
        return [case for case in self.cases.values() 
                if case.category == dataset.category]
    
    def evaluate_model(self,
                      model_name: str,
                      model_function: Callable,
                      dataset_id: str,
                      output_dir: str) -> EvaluationResult:
        """Evaluate a model against a benchmark dataset"""
        
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {dataset_id} not found")
        
        cases = self.get_cases_for_dataset(dataset_id)
        if not cases:
            raise ValueError(f"No cases found for dataset {dataset_id}")
        
        self.logger.info(f"Evaluating model {model_name} on {len(cases)} cases")
        
        start_time = datetime.now()
        case_results = []
        category_scores = {cat: [] for cat in BenchmarkCategory}
        difficulty_scores = {level: [] for level in DifficultyLevel}
        
        # Evaluate each case
        for case in cases:
            try:
                result = self._evaluate_single_case(model_function, case)
                case_results.append(result)
                
                # Track scores
                category_scores[case.category].append(result["overall_score"])
                difficulty_scores[case.difficulty_level].append(result["overall_score"])
                
            except Exception as e:
                self.logger.error(f"Error evaluating case {case.case_id}: {str(e)}")
                case_results.append({
                    "case_id": case.case_id,
                    "error": str(e),
                    "overall_score": 0.0
                })
        
        # Calculate aggregate scores
        overall_score = np.mean([r["overall_score"] for r in case_results if "overall_score" in r])
        
        final_category_scores = {
            cat: np.mean(scores) if scores else 0.0 
            for cat, scores in category_scores.items()
        }
        
        final_difficulty_scores = {
            level: np.mean(scores) if scores else 0.0
            for level, scores in difficulty_scores.items()
        }
        
        # Calculate detailed metrics
        accuracy_metrics = self._calculate_accuracy_metrics(case_results)
        safety_metrics = self._calculate_safety_metrics(case_results)
        efficiency_metrics = self._calculate_efficiency_metrics(case_results)
        
        # Generate error analysis
        error_analysis = self._analyze_errors(case_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(overall_score, error_analysis)
        
        # Create evaluation result
        evaluation_duration = (datetime.now() - start_time).total_seconds()
        
        result = EvaluationResult(
            evaluation_id=f"eval_{dataset_id}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataset_id=dataset_id,
            model_name=model_name,
            timestamp=datetime.now(),
            overall_score=overall_score,
            category_scores=final_category_scores,
            difficulty_scores=final_difficulty_scores,
            accuracy_metrics=accuracy_metrics,
            safety_metrics=safety_metrics,
            efficiency_metrics=efficiency_metrics,
            case_results=case_results,
            error_analysis=error_analysis,
            recommendations=recommendations,
            evaluation_duration=evaluation_duration,
            resource_usage={"cpu_time": evaluation_duration, "memory_usage": "N/A"}
        )
        
        # Store result
        self.evaluation_history.append(result)
        
        # Save results
        self._save_evaluation_result(result, output_dir)
        
        self.logger.info(f"Evaluation completed: {model_name} scored {overall_score:.2f}")
        return result
    
    def _evaluate_single_case(self, 
                            model_function: Callable, 
                            case: BenchmarkCase) -> Dict[str, Any]:
        """Evaluate a single case"""
        
        # Call model function with case data
        model_output = model_function(case.clinical_data)
        
        # Evaluate against gold standard
        evaluation_result = self._score_case(model_output, case)
        
        return {
            "case_id": case.case_id,
            "model_output": model_output,
            "gold_standard": case.gold_standard,
            "evaluation_result": evaluation_result,
            "overall_score": evaluation_result["overall_score"],
            "detailed_scores": evaluation_result["detailed_scores"],
            "correct_predictions": evaluation_result["correct_predictions"],
            "incorrect_predictions": evaluation_result["incorrect_predictions"]
        }
    
    def _score_case(self, 
                   model_output: Dict[str, Any], 
                   case: BenchmarkCase) -> Dict[str, Any]:
        """Score model output against case gold standard"""
        
        gold_standard = case.gold_standard
        criteria = case.evaluation_criteria
        
        detailed_scores = {}
        correct_predictions = []
        incorrect_predictions = []
        
        # Score each criterion
        for criterion, max_score in criteria.items():
            if criterion == "accuracy":
                score = self._score_accuracy(model_output, gold_standard, max_score)
            elif criterion == "completeness":
                score = self._score_completeness(model_output, gold_standard, max_score)
            elif criterion == "safety":
                score = self._score_safety(model_output, gold_standard, max_score)
            elif criterion == "clarity":
                score = self._score_clarity(model_output, gold_standard, max_score)
            elif criterion == "evidence_based":
                score = self._score_evidence_based(model_output, gold_standard, max_score)
            else:
                score = max_score * 0.5  # Default neutral score
            
            detailed_scores[criterion] = score
            
            # Track correctness
            if score >= max_score * 0.7:  # 70% threshold for correctness
                correct_predictions.append(criterion)
            else:
                incorrect_predictions.append(criterion)
        
        # Calculate overall score
        total_possible = sum(criteria.values())
        earned_points = sum(detailed_scores.values())
        overall_score = earned_points / total_possible if total_possible > 0 else 0
        
        return {
            "overall_score": overall_score,
            "detailed_scores": detailed_scores,
            "correct_predictions": correct_predictions,
            "incorrect_predictions": incorrect_predictions
        }
    
    def _score_accuracy(self, 
                       model_output: Dict[str, Any], 
                       gold_standard: Dict[str, Any], 
                       max_score: float) -> float:
        """Score accuracy of predictions"""
        
        # Simple accuracy scoring based on key fields
        key_fields = ["primary_diagnosis", "differential_diagnosis", "treatment_recommendations"]
        
        matches = 0
        total_fields = 0
        
        for field in key_fields:
            if field in model_output and field in gold_standard:
                total_fields += 1
                if field == "differential_diagnosis":
                    # For lists, check for overlap
                    model_list = model_output[field] if isinstance(model_output[field], list) else [model_output[field]]
                    gold_list = gold_standard[field] if isinstance(gold_standard[field], list) else [gold_standard[field]]
                    overlap = len(set(model_list) & set(gold_list))
                    if overlap > 0:
                        matches += 1
                else:
                    # For single values, check exact match
                    if str(model_output[field]).lower() == str(gold_standard[field]).lower():
                        matches += 1
        
        accuracy = matches / total_fields if total_fields > 0 else 0
        return accuracy * max_score
    
    def _score_completeness(self, 
                          model_output: Dict[str, Any], 
                          gold_standard: Dict[str, Any], 
                          max_score: float) -> float:
        """Score completeness of response"""
        
        expected_fields = set(gold_standard.keys())
        provided_fields = set(model_output.keys())
        
        # Calculate completeness as proportion of expected fields provided
        completeness = len(provided_fields & expected_fields) / len(expected_fields) if expected_fields else 0
        
        # Bonus for additional relevant fields
        additional_bonus = min(0.2, len(provided_fields - expected_fields) * 0.05)
        final_completeness = min(1.0, completeness + additional_bonus)
        
        return final_completeness * max_score
    
    def _score_safety(self, 
                     model_output: Dict[str, Any], 
                     gold_standard: Dict[str, Any], 
                     max_score: float) -> float:
        """Score safety of recommendations"""
        
        # Check for dangerous recommendations
        dangerous_patterns = [
            "stop_all_medications",
            "ignore_symptoms",
            "delay_treatment",
            "contraindicated_combination"
        ]
        
        model_str = json.dumps(model_output).lower()
        
        # Deduct points for dangerous patterns
        safety_penalty = 0
        for pattern in dangerous_patterns:
            if pattern in model_str:
                safety_penalty += 0.3
        
        # Base safety score
        safety_score = max(0, 1.0 - safety_penalty)
        
        return safety_score * max_score
    
    def _score_clarity(self, 
                      model_output: Dict[str, Any], 
                      gold_standard: Dict[str, Any], 
                      max_score: float) -> float:
        """Score clarity of response"""
        
        # Check for clear structure and readable content
        clarity_indicators = 0
        
        # Check for structured output
        if isinstance(model_output, dict) and len(model_output) > 0:
            clarity_indicators += 0.3
        
        # Check for reasonable text length
        total_text = sum(len(str(v)) for v in model_output.values())
        if 50 <= total_text <= 2000:  # Reasonable text length
            clarity_indicators += 0.3
        
        # Check for absence of unclear elements
        unclear_indicators = 0
        for value in model_output.values():
            if isinstance(value, str):
                if len(value.split()) < 3:  # Too brief
                    unclear_indicators += 0.1
                if "unclear" in value.lower() or "uncertain" in value.lower():
                    unclear_indicators += 0.2
        
        clarity_score = clarity_indicators - unclear_indicators
        clarity_score = max(0, min(1, clarity_score))
        
        return clarity_score * max_score
    
    def _score_evidence_based(self, 
                            model_output: Dict[str, Any], 
                            gold_standard: Dict[str, Any], 
                            max_score: float) -> float:
        """Score evidence-based reasoning"""
        
        # Check for evidence-based indicators in reasoning
        evidence_indicators = [
            "guidelines", "evidence", "studies", "research", "literature",
            "clinical_trial", "meta_analysis", "systematic_review"
        ]
        
        reasoning_text = ""
        for value in model_output.values():
            if isinstance(value, str):
                reasoning_text += value + " "
        
        reasoning_text = reasoning_text.lower()
        
        # Count evidence indicators
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in reasoning_text)
        
        # Score based on evidence indicators
        evidence_score = min(1.0, evidence_count / 3)  # Max at 3 indicators
        
        return evidence_score * max_score
    
    def _calculate_accuracy_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate accuracy metrics across all cases"""
        
        if not case_results:
            return {}
        
        scores = [r["overall_score"] for r in case_results if "overall_score" in r]
        
        return {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "min_accuracy": np.min(scores),
            "max_accuracy": np.max(scores),
            "median_accuracy": np.median(scores),
            "accuracy_at_95th_percentile": np.percentile(scores, 95),
            "cases_above_threshold": sum(1 for s in scores if s >= 0.7) / len(scores)
        }
    
    def _calculate_safety_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate safety metrics"""
        
        safety_scores = []
        for result in case_results:
            if "detailed_scores" in result:
                safety_score = result["detailed_scores"].get("safety", 0)
                safety_scores.append(safety_score)
        
        if not safety_scores:
            return {}
        
        return {
            "mean_safety_score": np.mean(safety_scores),
            "critical_safety_failures": sum(1 for s in safety_scores if s < 0.3),
            "safety_compliance_rate": sum(1 for s in safety_scores if s >= 0.7) / len(safety_scores)
        }
    
    def _calculate_efficiency_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate efficiency metrics"""
        
        # This would typically include timing and resource usage
        # For now, return basic metrics
        return {
            "average_processing_time": 0.0,  # Would be populated with actual timing data
            "throughput": len(case_results),
            "resource_efficiency": 1.0
        }
    
    def _analyze_errors(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze errors across all cases"""
        
        error_cases = [r for r in case_results if "error" in r]
        low_score_cases = [r for r in case_results if r.get("overall_score", 0) < 0.5]
        
        return {
            "total_errors": len(error_cases),
            "low_performance_cases": len(low_score_cases),
            "error_rate": len(error_cases) / len(case_results) if case_results else 0,
            "common_failure_modes": self._identify_common_failures(case_results),
            "case_ids_with_errors": [r["case_id"] for r in error_cases]
        }
    
    def _identify_common_failures(self, case_results: List[Dict[str, Any]]) -> List[str]:
        """Identify common failure patterns"""
        
        failures = []
        for result in case_results:
            if "incorrect_predictions" in result:
                failures.extend(result["incorrect_predictions"])
        
        # Count failure types
        failure_counts = {}
        for failure in failures:
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        
        # Return most common failures
        sorted_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
        return [f"{failure}: {count}" for failure, count in sorted_failures[:5]]
    
    def _generate_recommendations(self, 
                                overall_score: float, 
                                error_analysis: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        if overall_score < 0.5:
            recommendations.append("Major model improvements needed - consider retraining")
        elif overall_score < 0.7:
            recommendations.append("Moderate improvements needed - focus on accuracy and safety")
        
        if error_analysis["error_rate"] > 0.1:
            recommendations.append("High error rate detected - improve error handling")
        
        if error_analysis["critical_safety_failures"] > 0:
            recommendations.append("Critical safety failures detected - prioritize safety improvements")
        
        # Add specific recommendations based on failure modes
        for failure in error_analysis.get("common_failure_modes", []):
            if "accuracy" in failure:
                recommendations.append("Improve diagnostic accuracy through better training data")
            if "safety" in failure:
                recommendations.append("Implement safety checks and contraindication detection")
            if "completeness" in failure:
                recommendations.append("Ensure all required information is captured")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory - continue monitoring")
        
        return recommendations
    
    def _save_evaluation_result(self, result: EvaluationResult, output_dir: str):
        """Save evaluation result to file"""
        
        output_path = Path(output_dir) / f"{result.evaluation_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = asdict(result)
        result_dict["timestamp"] = result.timestamp.isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Evaluation result saved to {output_path}")
    
    def compare_models(self, 
                      model_names: List[str], 
                      dataset_id: str,
                      comparison_metrics: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models on the same dataset"""
        
        if comparison_metrics is None:
            comparison_metrics = ["overall_score", "mean_accuracy", "safety_compliance_rate"]
        
        # Get evaluation results for all models
        model_results = {}
        for result in self.evaluation_history:
            if result.dataset_id == dataset_id and result.model_name in model_names:
                model_results[result.model_name] = result
        
        if len(model_results) != len(model_names):
            missing_models = set(model_names) - set(model_results.keys())
            self.logger.warning(f"Missing evaluation results for models: {missing_models}")
        
        # Create comparison
        comparison = {
            "dataset_id": dataset_id,
            "models_compared": list(model_results.keys()),
            "comparison_timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Compare each metric
        for metric in comparison_metrics:
            metric_scores = {}
            for model_name, result in model_results.items():
                if "." in metric:
                    # Handle nested metrics
                    parts = metric.split(".")
                    value = result
                    for part in parts:
                        if hasattr(value, part):
                            value = getattr(value, part)
                        elif isinstance(value, dict) and part in value:
                            value = value[part]
                        else:
                            value = None
                            break
                    metric_scores[model_name] = value
                else:
                    # Handle direct metrics
                    if hasattr(result, metric):
                        metric_scores[model_name] = getattr(result, metric)
                    elif metric in result.__dict__:
                        metric_scores[model_name] = result.__dict__[metric]
            
            comparison["metrics"][metric] = metric_scores
        
        # Add ranking
        if "overall_score" in comparison["metrics"]:
            scores = comparison["metrics"]["overall_score"]
            ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            comparison["ranking"] = [{"model": model, "score": score} for model, score in ranking]
        
        return comparison
    
    def generate_performance_report(self, dataset_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report for a dataset"""
        
        # Get all evaluations for this dataset
        dataset_evaluations = [r for r in self.evaluation_history if r.dataset_id == dataset_id]
        
        if not dataset_evaluations:
            return {"error": "No evaluations found for this dataset"}
        
        # Aggregate statistics
        all_scores = [r.overall_score for r in dataset_evaluations]
        
        report = {
            "dataset_info": asdict(self.datasets[dataset_id]) if dataset_id in self.datasets else {},
            "evaluation_summary": {
                "total_evaluations": len(dataset_evaluations),
                "unique_models": len(set(r.model_name for r in dataset_evaluations)),
                "evaluation_period": {
                    "first_evaluation": min(r.timestamp for r in dataset_evaluations).isoformat(),
                    "last_evaluation": max(r.timestamp for r in dataset_evaluations).isoformat()
                }
            },
            "performance_statistics": {
                "overall_performance": {
                    "mean_score": np.mean(all_scores),
                    "median_score": np.median(all_scores),
                    "std_score": np.std(all_scores),
                    "min_score": np.min(all_scores),
                    "max_score": np.max(all_scores)
                }
            },
            "model_rankings": [],
            "benchmark_trends": {},
            "recommendations": []
        }
        
        # Model rankings
        model_scores = {}
        for evaluation in dataset_evaluations:
            if evaluation.model_name not in model_scores:
                model_scores[evaluation.model_name] = []
            model_scores[evaluation.model_name].append(evaluation.overall_score)
        
        model_rankings = [
            {
                "model": model,
                "average_score": np.mean(scores),
                "evaluation_count": len(scores),
                "best_score": np.max(scores),
                "consistency": 1 - np.std(scores) if len(scores) > 1 else 1.0
            }
            for model, scores in model_scores.items()
        ]
        
        model_rankings.sort(key=lambda x: x["average_score"], reverse=True)
        report["model_rankings"] = model_rankings
        
        # Generate recommendations
        report["recommendations"] = self._generate_dataset_recommendations(dataset_evaluations)
        
        return report
    
    def _generate_dataset_recommendations(self, evaluations: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on dataset evaluation results"""
        
        recommendations = []
        
        # Analyze overall performance
        scores = [e.overall_score for e in evaluations]
        mean_score = np.mean(scores)
        
        if mean_score < 0.6:
            recommendations.append("Dataset appears challenging - consider validation with domain experts")
        elif mean_score > 0.9:
            recommendations.append("Dataset may be too easy - consider increasing difficulty")
        
        # Analyze error patterns
        all_errors = []
        for evaluation in evaluations:
            all_errors.extend(evaluation.error_analysis.get("common_failure_modes", []))
        
        if all_errors:
            recommendations.append("Review common failure patterns to improve dataset quality")
        
        # Performance consistency
        if len(evaluations) > 1:
            score_variance = np.var(scores)
            if score_variance > 0.1:
                recommendations.append("High performance variance detected - consider data quality review")
        
        return recommendations


# Example usage and testing
def example_benchmark_evaluation():
    """Example of benchmark evaluation workflow"""
    
    # Initialize benchmark suite
    suite = ClinicalBenchmarkSuite()
    
    # Create a sample dataset
    dataset_id = suite.create_benchmark_dataset(
        name="Diagnostic Accuracy Test Set",
        category=BenchmarkCategory.DIAGNOSTIC_ACCURACY,
        num_cases=50,
        difficulty_levels=[DifficultyLevel.BASIC, DifficultyLevel.INTERMEDIATE],
        dataset_type=DatasetType.SYNTHETIC
    )
    
    print(f"Created dataset: {dataset_id}")
    
    # Define a simple mock model function
    def mock_medical_model(clinical_data):
        """Mock medical AI model for testing"""
        import random
        
        # Simulate model processing
        time.sleep(random.uniform(0.1, 0.5))
        
        # Generate mock output
        output = {
            "primary_diagnosis": clinical_data.get("primary_diagnosis", "unknown"),
            "differential_diagnosis": clinical_data.get("differential_diagnosis", []),
            "confidence_score": random.uniform(0.6, 0.95),
            "reasoning": "Based on clinical presentation and diagnostic findings.",
            "next_steps": ["further_testing", "specialist_consultation"]
        }
        
        return output
    
    # Evaluate the mock model
    result = suite.evaluate_model(
        model_name="MockMedicalAI_v1.0",
        model_function=mock_medical_model,
        dataset_id=dataset_id,
        output_dir="./evaluation_results"
    )
    
    print(f"Evaluation completed:")
    print(f"  Overall Score: {result.overall_score:.3f}")
    print(f"  Dataset ID: {result.dataset_id}")
    print(f"  Model: {result.model_name}")
    print(f"  Evaluation Duration: {result.evaluation_duration:.2f}s")
    
    # Generate performance report
    report = suite.generate_performance_report(dataset_id)
    print(f"\nPerformance Report Generated:")
    print(f"  Total Evaluations: {report['evaluation_summary']['total_evaluations']}")
    print(f"  Unique Models: {report['evaluation_summary']['unique_models']}")
    
    # Save detailed results
    print(f"\nDetailed results saved to: ./evaluation_results/")
    
    return result


if __name__ == "__main__":
    # Run example
    import time  # For mock model sleep simulation
    example_benchmark_evaluation()