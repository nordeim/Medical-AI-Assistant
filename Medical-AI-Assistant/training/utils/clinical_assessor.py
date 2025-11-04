"""
Clinical Accuracy Assessor

Comprehensive clinical accuracy assessment tools for medical AI models including:
- Medical terminology accuracy
- Symptom-diagnosis consistency  
- Treatment appropriateness
- Contraindication detection
- Drug interaction safety
- Clinical knowledge validation
- Risk assessment

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path


class RiskLevel(Enum):
    """Risk level classification for clinical scenarios"""
    LOW = "low"
    MODERATE = "moderate" 
    HIGH = "high"
    CRITICAL = "critical"


class ClinicalDomain(Enum):
    """Clinical domains for specialized assessment"""
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    EMERGENCY = "emergency"
    PRIMARY_CARE = "primary_care"
    SURGERY = "surgery"
    PSYCHIATRY = "psychiatry"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    RADIOLOGY = "radiology"


@dataclass
class ClinicalMetric:
    """Individual clinical assessment metric"""
    name: str
    score: float
    max_score: float
    weight: float
    domain: str
    description: str
    details: Dict[str, Any]
    timestamp: datetime


@dataclass 
class ClinicalAssessment:
    """Complete clinical assessment results"""
    case_id: str
    overall_score: float
    risk_level: RiskLevel
    metrics: List[ClinicalMetric]
    recommendations: List[str]
    warnings: List[str]
    compliance_status: Dict[str, bool]
    timestamp: datetime


class MedicalKnowledgeBase:
    """Medical knowledge base for validation"""
    
    def __init__(self):
        self.drug_interactions = self._load_drug_interactions()
        self.contraindications = self._load_contraindications()
        self.dosage_guidelines = self._load_dosage_guidelines()
        self.symptom_diagnosis_map = self._load_symptom_diagnosis_map()
        self.treatment_protocols = self._load_treatment_protocols()
        
    def _load_drug_interactions(self) -> Dict[str, List[str]]:
        """Load drug interaction data"""
        return {
            "warfarin": ["aspirin", "ibuprofen", "amiodarone"],
            "metformin": ["contrast_dye", "alcohol"],
            "lisinopril": ["potassium", "spironolactone"],
            "simvastatin": ["grapefruit", "clarithromycin"],
            "digoxin": ["verapamil", "amiodarone", "quinidine"]
        }
    
    def _load_contraindications(self) -> Dict[str, List[str]]:
        """Load contraindication data"""
        return {
            "aspirin": ["bleeding_disorders", "peptic_ulcer", "pregnancy"],
            "metformin": ["kidney_disease", "liver_disease", "alcohol_abuse"],
            "ace_inhibitors": ["pregnancy", "angioedema_history", "bilateral_renal_artery_stenosis"],
            "beta_blockers": ["asthma", "copd", "heart_block"],
            "statins": ["liver_disease", "pregnancy", "breastfeeding"]
        }
    
    def _load_dosage_guidelines(self) -> Dict[str, Dict]:
        """Load dosage guideline data"""
        return {
            "warfarin": {
                "initial_dose": "5mg",
                "max_dose": "10mg",
                "monitoring": "INR"
            },
            "metformin": {
                "initial_dose": "500mg",
                "max_dose": "2000mg",
                "monitoring": "renal_function"
            },
            "lisinopril": {
                "initial_dose": "10mg",
                "max_dose": "40mg",
                "monitoring": "blood_pressure, renal_function"
            }
        }
    
    def _load_symptom_diagnosis_map(self) -> Dict[str, List[str]]:
        """Load symptom to diagnosis mapping"""
        return {
            "chest_pain": ["myocardial_infarction", "angina", "pulmonary_embolism", "aortic_dissection"],
            "shortness_of_breath": ["heart_failure", "pneumonia", "copd", "asthma"],
            "abdominal_pain": ["appendicitis", "gastritis", "pancreatitis", "bowel_obstruction"],
            "headache": ["migraine", "tension_headache", "cluster_headache", "brain_tumor"],
            "fever": ["infection", "inflammation", "malignancy", "autoimmune_disease"]
        }
    
    def _load_treatment_protocols(self) -> Dict[str, Dict]:
        """Load treatment protocol data"""
        return {
            "myocardial_infarction": {
                "emergency": ["aspirin", "nitroglycerin", "oxygen"],
                "acute": ["thrombolytics", "beta_blockers", "ace_inhibitors"],
                "follow_up": ["statin", "aspirin", "lifestyle_modification"]
            },
            "heart_failure": {
                "initial": ["ace_inhibitor", "beta_blocker", "diuretic"],
                "optimization": ["spironolactone", "digoxin"],
                "monitoring": ["weight", "blood_pressure", "electrolytes"]
            }
        }


class ClinicalAssessor:
    """Main clinical accuracy assessment engine"""
    
    def __init__(self, knowledge_base: Optional[MedicalKnowledgeBase] = None):
        self.logger = self._setup_logger()
        self.knowledge_base = knowledge_base or MedicalKnowledgeBase()
        self.risk_weights = {
            "contraindication": 10.0,
            "drug_interaction": 8.0,
            "dosage_error": 7.0,
            "diagnosis_miss": 9.0,
            "treatment_inappropriate": 6.0,
            "missing_urgent_treatment": 10.0
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for clinical assessment"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def assess_medical_terminology(self, 
                                  predicted_text: str, 
                                  reference_text: str) -> ClinicalMetric:
        """Assess accuracy of medical terminology usage"""
        
        # Extract medical terms using regex patterns
        predicted_terms = self._extract_medical_terms(predicted_text)
        reference_terms = self._extract_medical_terms(reference_text)
        
        # Calculate precision and recall
        true_positives = len(predicted_terms.intersection(reference_terms))
        precision = true_positives / len(predicted_terms) if predicted_terms else 0
        recall = true_positives / len(reference_terms) if reference_terms else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        details = {
            "predicted_terms": list(predicted_terms),
            "reference_terms": list(reference_terms),
            "true_positives": true_positives,
            "precision": precision,
            "recall": recall
        }
        
        return ClinicalMetric(
            name="medical_terminology_accuracy",
            score=f1_score,
            max_score=1.0,
            weight=0.15,
            domain="general",
            description="Accuracy of medical terminology usage",
            details=details,
            timestamp=datetime.now()
        )
    
    def _extract_medical_terms(self, text: str) -> set:
        """Extract medical terms from text using patterns"""
        # Common medical term patterns
        patterns = [
            r'\b(?:hypertension|hypotension|diabetes|cancer|stroke|heart\s+attack)\b',
            r'\b(?:aspirin|warfarin|metformin|lisinopril|atorvastatin)\b',
            r'\b(?:myocardial|infarction|arrhythmia|tachycardia|bradycardia)\b',
            r'\b(?:fever|chill|sweat|headache|nausea|vomiting)\b'
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            terms.update(matches)
            
        return terms
    
    def assess_symptom_diagnosis_consistency(self, 
                                           symptoms: List[str], 
                                           diagnosis: str,
                                           explanation: str) -> ClinicalMetric:
        """Assess consistency between symptoms and diagnosis"""
        
        # Get potential diagnoses for symptoms
        potential_diagnoses = set()
        for symptom in symptoms:
            if symptom in self.knowledge_base.symptom_diagnosis_map:
                potential_diagnoses.update(
                    self.knowledge_base.symptom_diagnosis_map[symptom]
                )
        
        # Check if diagnosis is consistent
        is_consistent = diagnosis.lower() in [d.lower() for d in potential_diagnoses]
        
        # Assess reasoning quality
        reasoning_quality = self._assess_reasoning_quality(symptoms, diagnosis, explanation)
        
        # Combined score
        consistency_score = 1.0 if is_consistent else 0.0
        final_score = (consistency_score + reasoning_quality) / 2
        
        details = {
            "symptoms": symptoms,
            "diagnosis": diagnosis,
            "potential_diagnoses": list(potential_diagnoses),
            "is_consistent": is_consistent,
            "reasoning_quality": reasoning_quality
        }
        
        return ClinicalMetric(
            name="symptom_diagnosis_consistency",
            score=final_score,
            max_score=1.0,
            weight=0.25,
            domain="diagnostic",
            description="Consistency between symptoms and diagnosis",
            details=details,
            timestamp=datetime.now()
        )
    
    def _assess_reasoning_quality(self, symptoms: List[str], diagnosis: str, explanation: str) -> float:
        """Assess quality of diagnostic reasoning"""
        if not explanation:
            return 0.0
            
        # Check for key reasoning components
        reasoning_components = [
            "based on", "consistent with", "likely", "suggests", 
            "evidence of", "correlates with"
        ]
        
        explanation_lower = explanation.lower()
        component_count = sum(1 for comp in reasoning_components if comp in explanation_lower)
        
        # Check for symptom mentions in explanation
        symptom_mentions = sum(1 for symptom in symptoms if symptom.lower() in explanation_lower)
        symptom_coverage = symptom_mentions / len(symptoms) if symptoms else 0
        
        return min(1.0, (component_count * 0.3 + symptom_coverage * 0.7))
    
    def assess_treatment_appropriateness(self, 
                                       diagnosis: str,
                                       recommended_treatments: List[str],
                                       patient_context: Dict[str, Any]) -> ClinicalMetric:
        """Assess appropriateness of recommended treatments"""
        
        # Get standard treatments for diagnosis
        standard_treatments = self.knowledge_base.treatment_protocols.get(
            diagnosis.lower(), {}
        ).get("initial", [])
        
        # Check treatment appropriateness
        appropriate_treatments = 0
        total_treatments = len(recommended_treatments) if recommended_treatments else 0
        
        for treatment in recommended_treatments:
            if treatment.lower() in [t.lower() for t in standard_treatments]:
                appropriate_treatments += 1
        
        # Check for contraindications
        contraindications_found = self._check_contraindications(
            recommended_treatments, patient_context
        )
        
        # Calculate appropriateness score
        if total_treatments > 0:
            coverage_score = appropriate_treatments / total_treatments
            contraindication_penalty = len(contraindications_found) * 0.2
            appropriateness_score = max(0, coverage_score - contraindication_penalty)
        else:
            appropriateness_score = 0.0
        
        details = {
            "diagnosis": diagnosis,
            "recommended_treatments": recommended_treatments,
            "standard_treatments": standard_treatments,
            "appropriate_treatments": appropriate_treatments,
            "contraindications_found": contraindications_found,
            "coverage_score": coverage_score if 'coverage_score' in locals() else 0
        }
        
        return ClinicalMetric(
            name="treatment_appropriateness",
            score=appropriateness_score,
            max_score=1.0,
            weight=0.20,
            domain="therapeutic",
            description="Appropriateness of recommended treatments",
            details=details,
            timestamp=datetime.now()
        )
    
    def assess_contraindications(self, 
                               medications: List[str],
                               patient_history: Dict[str, Any],
                               current_conditions: List[str]) -> ClinicalMetric:
        """Detect contraindications in treatment plan"""
        
        contraindications_found = []
        severity_scores = []
        
        for medication in medications:
            med_contraindications = self.knowledge_base.contraindications.get(
                medication.lower(), []
            )
            
            # Check against patient history
            patient_risk_factors = []
            for condition in current_conditions:
                if condition in med_contraindications:
                    patient_risk_factors.append(condition)
            
            if patient_risk_factors:
                contraindications_found.append({
                    "medication": medication,
                    "risk_factors": patient_risk_factors
                })
                severity_scores.append(len(patient_risk_factors))
        
        # Calculate risk score
        if not contraindications_found:
            risk_score = 1.0  # No contraindications found
        else:
            max_severity = max(severity_scores) if severity_scores else 1
            risk_score = max(0, 1.0 - (max_severity * 0.3))
        
        details = {
            "medications": medications,
            "contraindications_found": contraindications_found,
            "total_risk_factors": sum(len(c["risk_factors"]) for c in contraindications_found),
            "max_severity": max_severity if severity_scores else 0
        }
        
        return ClinicalMetric(
            name="contraindication_detection",
            score=risk_score,
            max_score=1.0,
            weight=0.15,
            domain="safety",
            description="Detection of medication contraindications",
            details=details,
            timestamp=datetime.now()
        )
    
    def _check_contraindications(self, 
                               medications: List[str], 
                               patient_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for specific contraindications"""
        contraindications = []
        patient_conditions = patient_context.get("conditions", [])
        patient_history = patient_context.get("history", [])
        
        for medication in medications:
            med_contraindications = self.knowledge_base.contraindications.get(
                medication.lower(), []
            )
            
            conflicts = []
            for condition in patient_conditions + patient_history:
                if condition in med_contraindications:
                    conflicts.append(condition)
            
            if conflicts:
                contraindications.append({
                    "medication": medication,
                    "conflicts": conflicts
                })
        
        return contraindications
    
    def assess_drug_interactions(self, 
                               medications: List[str]) -> ClinicalMetric:
        """Assess drug interaction safety"""
        
        interactions_found = []
        interaction_severity = []
        
        # Check pairwise interactions
        for i, med1 in enumerate(medications):
            for j, med2 in enumerate(medications[i+1:], i+1):
                known_interactions = self.knowledge_base.drug_interactions.get(
                    med1.lower(), []
                )
                
                if med2.lower() in known_interactions:
                    interactions_found.append({
                        "medication_1": med1,
                        "medication_2": med2,
                        "severity": "moderate"  # Default severity
                    })
                    interaction_severity.append(0.5)  # Moderate severity score
        
        # Check for triple interactions (higher risk)
        for i, med1 in enumerate(medications):
            for j, med2 in enumerate(medications[i+1:], i+1):
                for k, med3 in enumerate(medications[j+1:], j+1):
                    # Simplified triple interaction check
                    if (med1.lower() in self.knowledge_base.drug_interactions.get(med2.lower(), []) or
                        med2.lower() in self.knowledge_base.drug_interactions.get(med1.lower(), [])):
                        if any(med3.lower() in self.knowledge_base.drug_interactions.get(m.lower(), []) 
                              for m in [med1, med2]):
                            interactions_found.append({
                                "medication_1": med1,
                                "medication_2": med2, 
                                "medication_3": med3,
                                "severity": "high"
                            })
                            interaction_severity.append(0.8)  # High severity score
        
        # Calculate safety score
        if not interactions_found:
            safety_score = 1.0
        else:
            max_severity = max(interaction_severity) if interaction_severity else 0.5
            safety_score = max(0, 1.0 - max_severity)
        
        details = {
            "medications": medications,
            "interactions_found": interactions_found,
            "total_interactions": len(interactions_found),
            "max_severity_score": max(interaction_severity) if interaction_severity else 0,
            "high_risk_interactions": len([i for i in interactions_found if i.get("severity") == "high"])
        }
        
        return ClinicalMetric(
            name="drug_interaction_safety",
            score=safety_score,
            max_score=1.0,
            weight=0.15,
            domain="safety",
            description="Drug interaction safety assessment",
            details=details,
            timestamp=datetime.now()
        )
    
    def assess_clinical_guidelines(self, 
                                 clinical_scenario: Dict[str, Any]) -> ClinicalMetric:
        """Assess adherence to clinical guidelines"""
        
        scenario_type = clinical_scenario.get("scenario_type", "")
        recommendations = clinical_scenario.get("recommendations", [])
        context = clinical_scenario.get("context", {})
        
        # Get guideline requirements for scenario type
        guideline_compliance = self._check_guideline_compliance(
            scenario_type, recommendations, context
        )
        
        # Calculate compliance score
        total_requirements = len(guideline_compliance)
        met_requirements = sum(1 for req in guideline_compliance.values() if req["met"])
        
        compliance_score = met_requirements / total_requirements if total_requirements > 0 else 0
        
        details = {
            "scenario_type": scenario_type,
            "recommendations": recommendations,
            "guideline_compliance": guideline_compliance,
            "met_requirements": met_requirements,
            "total_requirements": total_requirements,
            "compliance_percentage": compliance_score * 100
        }
        
        return ClinicalMetric(
            name="clinical_guidelines_compliance",
            score=compliance_score,
            max_score=1.0,
            weight=0.10,
            domain="compliance",
            description="Adherence to clinical guidelines",
            details=details,
            timestamp=datetime.now()
        )
    
    def _check_guideline_compliance(self, 
                                  scenario_type: str,
                                  recommendations: List[str],
                                  context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Check compliance with specific clinical guidelines"""
        
        compliance_results = {}
        
        # Emergency medicine guidelines
        if scenario_type.lower() in ["emergency", "acute", "urgent"]:
            compliance_results.update({
                "triage_assessment": {"met": "triage" in " ".join(recommendations).lower()},
                "vital_signs": {"met": any(term in " ".join(recommendations).lower() 
                                       for term in ["bp", "pulse", "oxygen", "temperature"])},
                "immediate_treatment": {"met": len(recommendations) > 0}
            })
        
        # Chronic disease management
        if scenario_type.lower() in ["chronic", "long_term", "maintenance"]:
            compliance_results.update({
                "monitoring_plan": {"met": "monitor" in " ".join(recommendations).lower()},
                "patient_education": {"met": any(term in " ".join(recommendations).lower()
                                              for term in ["educate", "counsel", "inform"])},
                "follow_up": {"met": "follow" in " ".join(recommendations).lower()}
            })
        
        # Diagnostic guidelines
        if scenario_type.lower() in ["diagnostic", "workup"]:
            compliance_results.update({
                "appropriate_tests": {"met": any(term in " ".join(recommendations).lower()
                                              for term in ["test", "lab", "imaging", "study"])},
                "differential_diagnosis": {"met": "differential" in " ".join(recommendations).lower()},
                "risk_assessment": {"met": any(term in " ".join(recommendations).lower()
                                            for term in ["risk", "evaluate", "assess"])}
            })
        
        return compliance_results
    
    def assess_risk_level(self, 
                         clinical_case: Dict[str, Any]) -> Tuple[RiskLevel, ClinicalMetric]:
        """Assess overall risk level for clinical scenario"""
        
        risk_factors = []
        severity_scores = []
        
        # Check for high-risk indicators
        high_risk_indicators = clinical_case.get("high_risk_indicators", [])
        urgent_symptoms = clinical_case.get("urgent_symptoms", [])
        critical_medications = clinical_case.get("critical_medications", [])
        
        # Evaluate risk factors
        if high_risk_indicators:
            risk_factors.extend(high_risk_indicators)
            severity_scores.extend([0.8] * len(high_risk_indicators))
        
        if urgent_symptoms:
            risk_factors.extend(urgent_symptoms)
            severity_scores.extend([0.9] * len(urgent_symptoms))
        
        if critical_medications:
            risk_factors.extend([f"critical_med: {med}" for med in critical_medications])
            severity_scores.extend([0.7] * len(critical_medications))
        
        # Calculate overall risk
        if not risk_factors:
            risk_level = RiskLevel.LOW
            risk_score = 1.0
        else:
            max_severity = max(severity_scores) if severity_scores else 0.5
            avg_severity = np.mean(severity_scores) if severity_scores else 0.5
            
            if max_severity >= 0.8 or avg_severity >= 0.7:
                risk_level = RiskLevel.CRITICAL
                risk_score = 0.1
            elif max_severity >= 0.6 or avg_severity >= 0.5:
                risk_level = RiskLevel.HIGH  
                risk_score = 0.3
            elif max_severity >= 0.4 or avg_severity >= 0.3:
                risk_level = RiskLevel.MODERATE
                risk_score = 0.6
            else:
                risk_level = RiskLevel.LOW
                risk_score = 0.8
        
        details = {
            "risk_factors": risk_factors,
            "severity_scores": severity_scores,
            "max_severity": max(severity_scores) if severity_scores else 0,
            "avg_severity": np.mean(severity_scores) if severity_scores else 0,
            "total_risk_factors": len(risk_factors)
        }
        
        metric = ClinicalMetric(
            name="risk_assessment",
            score=risk_score,
            max_score=1.0,
            weight=0.20,
            domain="risk",
            description="Overall risk level assessment",
            details=details,
            timestamp=datetime.now()
        )
        
        return risk_level, metric
    
    def comprehensive_assessment(self, 
                               clinical_case: Dict[str, Any]) -> ClinicalAssessment:
        """Perform comprehensive clinical assessment"""
        
        case_id = clinical_case.get("case_id", f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        metrics = []
        recommendations = []
        warnings = []
        
        # Medical terminology assessment
        if "predicted_text" in clinical_case and "reference_text" in clinical_case:
            term_metric = self.assess_medical_terminology(
                clinical_case["predicted_text"],
                clinical_case["reference_text"]
            )
            metrics.append(term_metric)
        
        # Symptom-diagnosis consistency
        if "symptoms" in clinical_case and "diagnosis" in clinical_case:
            consistency_metric = self.assess_symptom_diagnosis_consistency(
                clinical_case["symptoms"],
                clinical_case["diagnosis"],
                clinical_case.get("explanation", "")
            )
            metrics.append(consistency_metric)
        
        # Treatment appropriateness
        if "diagnosis" in clinical_case and "treatments" in clinical_case:
            treatment_metric = self.assess_treatment_appropriateness(
                clinical_case["diagnosis"],
                clinical_case["treatments"],
                clinical_case.get("patient_context", {})
            )
            metrics.append(treatment_metric)
        
        # Contraindications assessment
        if "medications" in clinical_case:
            contraindication_metric = self.assess_contraindications(
                clinical_case["medications"],
                clinical_case.get("patient_history", {}),
                clinical_case.get("conditions", [])
            )
            metrics.append(contraindication_metric)
            
            # Add warnings for contraindications
            for contra in contraindication_metric.details.get("contraindications_found", []):
                warnings.append(f"Contraindication: {contra['medication']} with {contra['risk_factors']}")
        
        # Drug interactions
        if "medications" in clinical_case and len(clinical_case["medications"]) > 1:
            interaction_metric = self.assess_drug_interactions(
                clinical_case["medications"]
            )
            metrics.append(interaction_metric)
            
            # Add warnings for interactions
            for interaction in interaction_metric.details.get("interactions_found", []):
                warnings.append(f"Drug interaction: {interaction['medication_1']} - {interaction['medication_2']}")
        
        # Clinical guidelines compliance
        guidelines_metric = self.assess_clinical_guidelines(clinical_case)
        metrics.append(guidelines_metric)
        
        # Risk assessment
        risk_level, risk_metric = self.assess_risk_level(clinical_case)
        metrics.append(risk_metric)
        
        # Calculate overall score
        weighted_sum = sum(metric.score * metric.weight for metric in metrics)
        total_weight = sum(metric.weight for metric in metrics)
        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, risk_level, warnings)
        
        # Compliance status
        compliance_status = {
            "guidelines_adherent": guidelines_metric.score >= 0.8,
            "no_critical_contraindications": risk_level != RiskLevel.CRITICAL,
            "appropriate_risk_level": risk_level in [RiskLevel.LOW, RiskLevel.MODERATE],
            "safe_drug_combinations": not any("interaction" in w.lower() for w in warnings)
        }
        
        return ClinicalAssessment(
            case_id=case_id,
            overall_score=overall_score,
            risk_level=risk_level,
            metrics=metrics,
            recommendations=recommendations,
            warnings=warnings,
            compliance_status=compliance_status,
            timestamp=datetime.now()
        )
    
    def _generate_recommendations(self, 
                                metrics: List[ClinicalMetric],
                                risk_level: RiskLevel,
                                warnings: List[str]) -> List[str]:
        """Generate recommendations based on assessment results"""
        
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("URGENT: Immediate medical attention required")
            recommendations.append("Consider emergency protocols")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Close monitoring required")
            recommendations.append("Consider specialist consultation")
        
        # Metric-based recommendations
        for metric in metrics:
            if metric.name == "symptom_diagnosis_consistency" and metric.score < 0.7:
                recommendations.append("Review diagnostic reasoning - consider alternative diagnoses")
            
            if metric.name == "treatment_appropriateness" and metric.score < 0.6:
                recommendations.append("Re-evaluate treatment plan for appropriateness")
            
            if metric.name == "contraindication_detection" and metric.score < 0.8:
                recommendations.append("Review medication list for contraindications")
            
            if metric.name == "drug_interaction_safety" and metric.score < 0.8:
                recommendations.append("Check for drug interactions and adjust medications")
        
        # Warning-based recommendations
        for warning in warnings:
            if "contraindication" in warning.lower():
                recommendations.append("Discontinue conflicting medications")
            elif "interaction" in warning.lower():
                recommendations.append("Modify medication regimen to avoid interactions")
        
        return recommendations
    
    def save_assessment(self, assessment: ClinicalAssessment, output_path: str):
        """Save assessment results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        assessment_dict = {
            **asdict(assessment),
            "timestamp": assessment.timestamp.isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(assessment_dict, f, indent=2, default=str)
        
        self.logger.info(f"Assessment saved to {output_path}")
    
    def batch_assess(self, clinical_cases: List[Dict[str, Any]]) -> List[ClinicalAssessment]:
        """Assess multiple clinical cases in batch"""
        
        assessments = []
        for i, case in enumerate(clinical_cases):
            try:
                self.logger.info(f"Assessing case {i+1}/{len(clinical_cases)}")
                assessment = self.comprehensive_assessment(case)
                assessments.append(assessment)
            except Exception as e:
                self.logger.error(f"Error assessing case {i+1}: {str(e)}")
                # Create error assessment
                error_assessment = ClinicalAssessment(
                    case_id=case.get("case_id", f"case_{i}"),
                    overall_score=0.0,
                    risk_level=RiskLevel.HIGH,
                    metrics=[],
                    recommendations=[f"Assessment failed: {str(e)}"],
                    warnings=["Assessment error"],
                    compliance_status={},
                    timestamp=datetime.now()
                )
                assessments.append(error_assessment)
        
        return assessments


# Example usage and testing
def example_assessment():
    """Example of clinical assessment usage"""
    
    # Initialize assessor
    assessor = ClinicalAssessor()
    
    # Example clinical case
    clinical_case = {
        "case_id": "example_001",
        "symptoms": ["chest_pain", "shortness_of_breath"],
        "diagnosis": "myocardial_infarction",
        "explanation": "Based on chest pain and shortness of breath, this is consistent with myocardial infarction",
        "treatments": ["aspirin", "nitroglycerin", "oxygen"],
        "medications": ["aspirin", "warfarin"],
        "patient_context": {
            "conditions": ["hypertension"],
            "age": 65
        },
        "high_risk_indicators": ["elderly", "cardiac_symptoms"],
        "scenario_type": "emergency"
    }
    
    # Perform assessment
    assessment = assessor.comprehensive_assessment(clinical_case)
    
    # Print results
    print(f"Case ID: {assessment.case_id}")
    print(f"Overall Score: {assessment.overall_score:.2f}")
    print(f"Risk Level: {assessment.risk_level.value}")
    print(f"Recommendations: {assessment.recommendations}")
    print(f"Warnings: {assessment.warnings}")
    
    # Save assessment
    assessor.save_assessment(assessment, "clinical_assessment_example.json")
    
    return assessment


if __name__ == "__main__":
    # Run example
    example_assessment()