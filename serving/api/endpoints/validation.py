"""
Medical Data Validation Endpoints
HIPAA compliance, PHI detection, and medical accuracy validation
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timezone
import re

from fastapi import APIRouter, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field, validator
import structlog

from ..utils.exceptions import ValidationError, MedicalValidationError, PHIProtectionError
from ..utils.security import SecurityValidator
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Medical validation result types
class ValidationResultType(Enum):
    MEDICAL_ACCURACY = "medical_accuracy"
    PHI_DETECTION = "phi_detection"
    HIPAA_COMPLIANCE = "hipaa_compliance"
    MEDICATION_SAFETY = "medication_safety"
    SYMPTOM_VALIDATION = "symptom_validation"
    DOSAGE_VALIDATION = "dosage_validation"
    CONTRAINDICATION_CHECK = "contraindication_check"

# Validation severity levels
class ValidationSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Validation status
class ValidationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"

class MedicalValidationEngine:
    """Comprehensive medical data validation engine"""
    
    def __init__(self):
        self.logger = get_logger("medical_validation")
        self.validation_rules = self._load_validation_rules()
        self.medical_terms = self._load_medical_terms()
        self.dosage_guidelines = self._load_dosage_guidelines()
        self.contraindications = self._load_contraindications()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load medical validation rules"""
        return {
            "vital_signs": {
                "blood_pressure": {
                    "systolic": {"min": 70, "max": 250},
                    "diastolic": {"min": 40, "max": 150}
                },
                "heart_rate": {"min": 30, "max": 200},
                "temperature": {"min": 35.0, "max": 42.0},
                "respiratory_rate": {"min": 8, "max": 40},
                "oxygen_saturation": {"min": 70, "max": 100}
            },
            "laboratory_values": {
                "glucose": {"min": 50, "max": 600, "unit": "mg/dl"},
                "hemoglobin": {"min": 8, "max": 20, "unit": "g/dl"},
                "white_blood_cells": {"min": 3, "max": 20, "unit": "10^3/μL"},
                "platelet_count": {"min": 50, "max": 500, "unit": "10^3/μL"}
            }
        }
    
    def _load_medical_terms(self) -> Dict[str, List[str]]:
        """Load medical terminology dictionary"""
        return {
            "anatomical_terms": [
                "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine",
                "spleen", "pancreas", "thyroid", "adrenal", "prostate", "uterus"
            ],
            "medical_conditions": [
                "diabetes", "hypertension", "asthma", "pneumonia", "fracture",
                "infection", "inflammation", "tumor", "cancer", "anemia"
            ],
            "medications": [
                "aspirin", "ibuprofen", "acetaminophen", "antibiotic", "insulin",
                "metformin", "lisinopril", "amlodipine", "atorvastatin"
            ],
            "symptoms": [
                "pain", "fever", "cough", "shortness of breath", "nausea",
                "vomiting", "diarrhea", "headache", "dizziness", "fatigue"
            ]
        }
    
    def _load_dosage_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """Load medication dosage guidelines"""
        return {
            "acetaminophen": {
                "max_daily_dose": 4000,  # mg
                "max_single_dose": 1000,
                "dosing_interval": 6,  # hours
                "warnings": ["liver toxicity", "overdose risk"]
            },
            "ibuprofen": {
                "max_daily_dose": 3200,
                "max_single_dose": 800,
                "dosing_interval": 6,
                "warnings": ["GI bleeding", "renal impairment"]
            },
            "aspirin": {
                "max_daily_dose": 4000,
                "max_single_dose": 1000,
                "dosing_interval": 4,
                "warnings": ["bleeding risk", "Reye syndrome"]
            }
        }
    
    def _load_contraindications(self) -> Dict[str, Dict[str, Any]]:
        """Load medication contraindications"""
        return {
            "ace_inhibitors": {
                "absolute": ["pregnancy", "history of angioedema"],
                "relative": ["bilateral renal artery stenosis", "hyperkalemia"],
                "monitoring": ["renal function", "electrolytes"]
            },
            "beta_blockers": {
                "absolute": ["severe bradycardia", "heart block", "cardiogenic shock"],
                "relative": ["asthma", "COPD", "diabetes"],
                "monitoring": ["heart rate", "blood pressure", "blood glucose"]
            }
        }
    
    async def validate_medical_data(
        self,
        data: Dict[str, Any],
        validation_types: List[ValidationResultType]
    ) -> Dict[str, Any]:
        """Comprehensive medical data validation"""
        
        validation_id = str(uuid.uuid4())
        validation_results = {
            "validation_id": validation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_types": [vt.value for vt in validation_types],
            "overall_status": ValidationStatus.PASS.value,
            "results": {},
            "summary": {
                "total_checks": 0,
                "passed_checks": 0,
                "failed_checks": 0,
                "warnings": 0,
                "critical_issues": 0
            }
        }
        
        try:
            # Run validation types
            for validation_type in validation_types:
                if validation_type == ValidationResultType.MEDICAL_ACCURACY:
                    result = await self._validate_medical_accuracy(data)
                    validation_results["results"]["medical_accuracy"] = result
                
                elif validation_type == ValidationResultType.PHI_DETECTION:
                    result = await self._detect_phi(data)
                    validation_results["results"]["phi_detection"] = result
                
                elif validation_type == ValidationResultType.HIPAA_COMPLIANCE:
                    result = await self._check_hipaa_compliance(data)
                    validation_results["results"]["hipaa_compliance"] = result
                
                elif validation_type == ValidationResultType.MEDICATION_SAFETY:
                    result = await self._validate_medication_safety(data)
                    validation_results["results"]["medication_safety"] = result
                
                elif validation_type == ValidationResultType.SYMPTOM_VALIDATION:
                    result = await self._validate_symptoms(data)
                    validation_results["results"]["symptom_validation"] = result
                
                elif validation_type == ValidationResultType.DOSAGE_VALIDATION:
                    result = await self._validate_dosages(data)
                    validation_results["results"]["dosage_validation"] = result
                
                elif validation_type == ValidationResultType.CONTRAINDICATION_CHECK:
                    result = await self._check_contraindications(data)
                    validation_results["results"]["contraindication_check"] = result
            
            # Calculate summary statistics
            self._calculate_validation_summary(validation_results)
            
            # Determine overall status
            validation_results["overall_status"] = self._determine_overall_status(validation_results)
            
            logger.info(
                "Medical data validation completed",
                validation_id=validation_id,
                overall_status=validation_results["overall_status"],
                total_checks=validation_results["summary"]["total_checks"]
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Medical validation failed: {e}", validation_id=validation_id)
            raise MedicalValidationError(f"Validation failed: {str(e)}")
    
    async def _validate_medical_accuracy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical data for accuracy and consistency"""
        
        result = {
            "validation_type": "medical_accuracy",
            "status": ValidationStatus.PASS.value,
            "checks": [],
            "warnings": [],
            "errors": []
        }
        
        # Check vital signs consistency
        if "vital_signs" in data:
            vital_signs = data["vital_signs"]
            
            # Blood pressure validation
            if "blood_pressure" in vital_signs:
                bp = vital_signs["blood_pressure"]
                if "systolic" in bp and "diastolic" in bp:
                    systolic = bp["systolic"]
                    diastolic = bp["diastolic"]
                    
                    # Systolic should be higher than diastolic
                    if systolic <= diastolic:
                        result["errors"].append("Systolic BP must be higher than diastolic BP")
                        result["status"] = ValidationStatus.FAIL.value
                    
                    # Check against normal ranges
                    if not (70 <= systolic <= 250):
                        result["warnings"].append(f"Systolic BP {systolic} is outside normal range")
                    if not (40 <= diastolic <= 150):
                        result["warnings"].append(f"Diastolic BP {diastolic} is outside normal range")
            
            # Heart rate validation
            if "heart_rate" in vital_signs:
                hr = vital_signs["heart_rate"]
                if not (30 <= hr <= 200):
                    result["errors"].append(f"Heart rate {hr} is outside physiologically plausible range")
                    result["status"] = ValidationStatus.FAIL.value
        
        # Check laboratory values
        if "laboratory_values" in data:
            lab_values = data["laboratory_values"]
            
            for test, value in lab_values.items():
                if test in self.validation_rules["laboratory_values"]:
                    rules = self.validation_rules["laboratory_values"][test]
                    if not (rules["min"] <= value <= rules["max"]):
                        result["warnings"].append(f"{test} value {value} is outside normal range")
        
        result["checks"].append({
            "check_name": "vital_signs_consistency",
            "status": "passed" if not result["errors"] else "failed"
        })
        
        result["checks"].append({
            "check_name": "laboratory_ranges",
            "status": "passed" if not result["warnings"] else "warning"
        })
        
        return result
    
    async def _detect_phi(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect PHI in medical data"""
        
        result = {
            "validation_type": "phi_detection",
            "status": ValidationStatus.PASS.value,
            "phi_detected": False,
            "phi_instances": [],
            "phi_types": [],
            "risk_level": "low"
        }
        
        # Convert data to string for analysis
        data_str = json.dumps(data)
        
        # Use security validator for PHI detection
        phi_instances = SecurityValidator.detect_phi(data_str)
        
        if phi_instances:
            result["phi_detected"] = True
            result["phi_instances"] = phi_instances
            result["phi_types"] = list(set(instance["type"] for instance in phi_instances))
            
            # Determine risk level
            high_risk_types = {"social_security_numbers", "biometric_identifiers", "photos"}
            medium_risk_types = {"names", "dates", "telephone_numbers", "email_addresses"}
            
            if any(phi_type in high_risk_types for phi_type in result["phi_types"]):
                result["risk_level"] = "high"
                result["status"] = ValidationStatus.REVIEW_REQUIRED.value
            elif any(phi_type in medium_risk_types for phi_type in result["phi_types"]):
                result["risk_level"] = "medium"
                result["status"] = ValidationStatus.WARNING.value
            else:
                result["risk_level"] = "low"
        
        return result
    
    async def _check_hipaa_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check HIPAA compliance of medical data"""
        
        result = {
            "validation_type": "hipaa_compliance",
            "status": ValidationStatus.PASS.value,
            "compliance_checks": [],
            "violations": [],
            "recommendations": []
        }
        
        # Check for proper identifiers
        data_str = json.dumps(data).lower()
        
        # Check for unencrypted identifiers
        unencrypted_identifiers = [
            "patient_name", "social_security", "medical_record_number"
        ]
        
        for identifier in unencrypted_identifiers:
            if identifier in data_str:
                result["violations"].append(f"Unencrypted {identifier.replace('_', ' ')} detected")
                result["status"] = ValidationStatus.FAIL.value
        
        # Check for proper data handling
        if "data_retention" not in data:
            result["recommendations"].append("Specify data retention policy")
        
        if "authorized_access" not in data:
            result["recommendations"].append("Define authorized access controls")
        
        # Check for audit trail requirements
        if "audit_required" not in data:
            result["compliance_checks"].append({
                "check": "audit_trail",
                "status": "missing",
                "recommendation": "Enable comprehensive audit logging"
            })
        
        result["compliance_checks"].append({
            "check": "phi_protection",
            "status": "passed" if result["status"] != ValidationStatus.FAIL.value else "failed"
        })
        
        return result
    
    async def _validate_medication_safety(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medication safety"""
        
        result = {
            "validation_type": "medication_safety",
            "status": ValidationStatus.PASS.value,
            "medications_checked": [],
            "safety_issues": [],
            "interactions": [],
            "warnings": []
        }
        
        if "medications" not in data:
            result["medications_checked"] = []
            return result
        
        medications = data["medications"]
        
        for medication in medications:
            med_name = medication.get("name", "").lower()
            result["medications_checked"].append(med_name)
            
            # Check against known safety issues
            for drug, info in self.dosage_guidelines.items():
                if drug in med_name:
                    # Check dosage if provided
                    if "dosage" in medication:
                        dosage = medication["dosage"]
                        if dosage > info["max_daily_dose"]:
                            result["safety_issues"].append({
                                "medication": med_name,
                                "issue": f"Dosage {dosage} exceeds maximum daily dose of {info['max_daily_dose']}mg",
                                "severity": "high"
                            })
                            result["status"] = ValidationStatus.FAIL.value
                    
                    # Add warnings
                    for warning in info["warnings"]:
                        result["warnings"].append({
                            "medication": med_name,
                            "warning": warning
                        })
        
        return result
    
    async def _validate_symptoms(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical symptoms"""
        
        result = {
            "validation_type": "symptom_validation",
            "status": ValidationStatus.PASS.value,
            "symptoms_analyzed": [],
            "medical_terms_found": [],
            "inconsistencies": [],
            "red_flags": []
        }
        
        if "symptoms" not in data:
            return result
        
        symptoms = data["symptoms"]
        result["symptoms_analyzed"] = symptoms
        
        # Check for medical terminology
        all_text = " ".join(str(s) for s in symptoms).lower()
        
        for category, terms in self.medical_terms.items():
            found_terms = [term for term in terms if term in all_text]
            if found_terms:
                result["medical_terms_found"].extend([
                    {"category": category, "terms": found_terms}
                ])
        
        # Check for red flag symptoms
        red_flag_symptoms = [
            "chest pain", "severe headache", "difficulty breathing",
            "loss of consciousness", "severe bleeding"
        ]
        
        for symptom in symptoms:
            symptom_lower = str(symptom).lower()
            for red_flag in red_flag_symptoms:
                if red_flag in symptom_lower:
                    result["red_flags"].append(symptom)
                    result["status"] = ValidationStatus.REVIEW_REQUIRED.value
        
        return result
    
    async def _validate_dosages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medication dosages"""
        
        result = {
            "validation_type": "dosage_validation",
            "status": ValidationStatus.PASS.value,
            "dosages_checked": [],
            "errors": [],
            "warnings": []
        }
        
        if "medications" not in data:
            return result
        
        medications = data["medications"]
        
        for medication in medications:
            if "name" in medication and "dosage" in medication:
                med_name = medication["name"].lower()
                dosage = medication["dosage"]
                result["dosages_checked"].append({
                    "medication": med_name,
                    "dosage": dosage
                })
                
                # Check against guidelines
                for drug, guidelines in self.dosage_guidelines.items():
                    if drug in med_name:
                        if dosage > guidelines["max_single_dose"]:
                            result["errors"].append({
                                "medication": med_name,
                                "dosage": dosage,
                                "max_allowed": guidelines["max_single_dose"],
                                "issue": "Single dose exceeds maximum"
                            })
                            result["status"] = ValidationStatus.FAIL.value
                        elif dosage > guidelines["max_daily_dose"] * 0.8:  # 80% threshold for warning
                            result["warnings"].append({
                                "medication": med_name,
                                "dosage": dosage,
                                "max_daily": guidelines["max_daily_dose"],
                                "warning": "Dosage approaching maximum daily limit"
                            })
        
        return result
    
    async def _check_contraindications(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for medication contraindications"""
        
        result = {
            "validation_type": "contraindication_check",
            "status": ValidationStatus.PASS.value,
            "contraindications_found": [],
            "absolute_contraindications": [],
            "relative_contraindications": [],
            "monitoring_required": []
        }
        
        if "medications" not in data or "patient_conditions" not in data:
            return result
        
        medications = data["medications"]
        patient_conditions = data["patient_conditions"]
        
        for medication in medications:
            med_name = medication.get("name", "").lower()
            
            # Check contraindications
            for drug, contraindications in self.contraindications.items():
                if drug in med_name:
                    # Check absolute contraindications
                    for condition in contraindications["absolute"]:
                        if any(cond in str(patient_conditions).lower() for cond in [condition]):
                            result["absolute_contraindications"].append({
                                "medication": med_name,
                                "contraindication": condition,
                                "severity": "absolute"
                            })
                            result["status"] = ValidationStatus.FAIL.value
                    
                    # Check relative contraindications
                    for condition in contraindications["relative"]:
                        if any(cond in str(patient_conditions).lower() for cond in [condition]):
                            result["relative_contraindications"].append({
                                "medication": med_name,
                                "contraindication": condition,
                                "severity": "relative"
                            })
                            result["status"] = ValidationStatus.WARNING.value
                    
                    # Check monitoring requirements
                    for monitor_item in contraindications["monitoring"]:
                        result["monitoring_required"].append({
                            "medication": med_name,
                            "monitor": monitor_item
                        })
        
        return result
    
    def _calculate_validation_summary(self, results: Dict[str, Any]):
        """Calculate validation summary statistics"""
        
        summary = results["summary"]
        summary["total_checks"] = 0
        summary["passed_checks"] = 0
        summary["failed_checks"] = 0
        summary["warnings"] = 0
        summary["critical_issues"] = 0
        
        for validation_result in results["results"].values():
            if isinstance(validation_result, dict):
                # Count checks
                if "checks" in validation_result:
                    for check in validation_result["checks"]:
                        summary["total_checks"] += 1
                        if check.get("status") == "passed":
                            summary["passed_checks"] += 1
                        elif check.get("status") == "failed":
                            summary["failed_checks"] += 1
                
                # Count errors
                if "errors" in validation_result:
                    summary["failed_checks"] += len(validation_result["errors"])
                
                # Count warnings
                if "warnings" in validation_result:
                    summary["warnings"] += len(validation_result["warnings"])
                
                # Count critical issues
                if "safety_issues" in validation_result:
                    critical_issues = [issue for issue in validation_result["safety_issues"] 
                                     if issue.get("severity") == "high"]
                    summary["critical_issues"] += len(critical_issues)
                
                # Count absolute contraindications
                if "absolute_contraindications" in validation_result:
                    summary["critical_issues"] += len(validation_result["absolute_contraindications"])
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> str:
        """Determine overall validation status"""
        
        summary = results["summary"]
        
        # Critical issues override everything
        if summary["critical_issues"] > 0:
            return ValidationStatus.FAIL.value
        
        # Failed checks
        if summary["failed_checks"] > 0:
            return ValidationStatus.FAIL.value
        
        # High number of warnings
        if summary["warnings"] > 5:
            return ValidationStatus.REVIEW_REQUIRED.value
        
        # Some warnings
        if summary["warnings"] > 0:
            return ValidationStatus.WARNING.value
        
        return ValidationStatus.PASS.value


# Global validation engine
validation_engine = MedicalValidationEngine()

# Pydantic models
class ValidationRequest(BaseModel):
    """Medical data validation request"""
    
    data: Dict[str, Any] = Field(..., description="Medical data to validate")
    validation_types: List[str] = Field(
        default_factory=lambda: ["medical_accuracy", "phi_detection", "hipaa_compliance"],
        description="Types of validation to perform"
    )
    strict_mode: bool = Field(False, description="Enable strict validation mode")
    include_recommendations: bool = Field(True, description="Include improvement recommendations")
    
    @validator('validation_types')
    def validate_types(cls, v):
        valid_types = [
            "medical_accuracy", "phi_detection", "hipaa_compliance",
            "medication_safety", "symptom_validation", "dosage_validation",
            "contraindication_check"
        ]
        
        for validation_type in v:
            if validation_type not in valid_types:
                raise ValueError(f"Invalid validation type: {validation_type}")
        
        return v


class ValidationResponse(BaseModel):
    """Medical data validation response"""
    
    validation_id: str = Field(..., description="Unique validation identifier")
    timestamp: str = Field(..., description="Validation timestamp")
    overall_status: str = Field(..., description="Overall validation status")
    validation_types: List[str] = Field(..., description="Validation types performed")
    results: Dict[str, Any] = Field(..., description="Detailed validation results")
    summary: Dict[str, Any] = Field(..., description="Validation summary statistics")
    recommendations: List[str] = Field([], description="Improvement recommendations")
    compliance_score: float = Field(..., ge=0.0, le=1.0, description="Overall compliance score")
    
    class Config:
        schema_extra = {
            "example": {
                "validation_id": "validation-uuid",
                "timestamp": "2024-01-15T10:30:00Z",
                "overall_status": "pass",
                "validation_types": ["medical_accuracy", "phi_detection"],
                "results": {
                    "medical_accuracy": {
                        "status": "pass",
                        "checks": [
                            {"check_name": "vital_signs_consistency", "status": "passed"}
                        ]
                    },
                    "phi_detection": {
                        "status": "pass",
                        "phi_detected": False
                    }
                },
                "summary": {
                    "total_checks": 2,
                    "passed_checks": 2,
                    "failed_checks": 0,
                    "warnings": 0,
                    "critical_issues": 0
                },
                "compliance_score": 1.0
            }
        }


class PHIAnalysisRequest(BaseModel):
    """PHI analysis request"""
    
    data: Dict[str, Any] = Field(..., description="Data to analyze for PHI")
    protection_mode: Literal["mask", "anonymize", "hash"] = Field("mask", description="PHI protection mode")
    preserve_structure: bool = Field(True, description="Preserve data structure when redacting")
    
    @validator('data')
    def validate_data(cls, v):
        if not v or not isinstance(v, dict):
            raise ValueError("Data must be a non-empty dictionary")
        return v


class PHIAnalysisResponse(BaseModel):
    """PHI analysis response"""
    
    analysis_id: str = Field(..., description="Unique analysis identifier")
    phi_detected: bool = Field(..., description="Whether PHI was detected")
    phi_types: List[str] = Field([], description="Types of PHI detected")
    phi_instances: List[Dict[str, Any]] = Field([], description="Specific PHI instances")
    risk_level: str = Field(..., description="PHI risk level")
    protected_data: Dict[str, Any] = Field(..., description="PHI-protected version of data")
    compliance_status: str = Field(..., description="HIPAA compliance status")
    timestamp: str = Field(..., description="Analysis timestamp")


class MedicalAccuracyRequest(BaseModel):
    """Medical accuracy validation request"""
    
    medical_data: Dict[str, Any] = Field(..., description="Medical data for accuracy validation")
    reference_guidelines: List[str] = Field(default_factory=list, description="Reference guidelines to check against")
    strict_validation: bool = Field(False, description="Enable strict medical validation")
    
    @validator('medical_data')
    def validate_medical_data(cls, v):
        required_sections = ["vital_signs", "symptoms", "diagnoses"]
        if not any(section in v for section in required_sections):
            raise ValueError("Medical data must contain at least one of: vital_signs, symptoms, diagnoses")
        return v


class MedicalAccuracyResponse(BaseModel):
    """Medical accuracy validation response"""
    
    accuracy_id: str = Field(..., description="Unique accuracy validation identifier")
    accuracy_score: float = Field(..., ge=0.0, le=1.0, description="Medical accuracy score")
    validation_status: str = Field(..., description="Validation status")
    accuracy_issues: List[Dict[str, Any]] = Field([], description="Identified accuracy issues")
    medical_consistency: Dict[str, Any] = Field(..., description="Medical consistency analysis")
    guideline_compliance: Dict[str, Any] = Field(..., description="Medical guideline compliance")
    recommendations: List[str] = Field([], description="Accuracy improvement recommendations")
    timestamp: str = Field(..., description="Validation timestamp")


# Endpoint implementations
@router.post("/validate", response_model=ValidationResponse)
async def validate_medical_data(request: ValidationRequest):
    """
    Comprehensive medical data validation.
    
    Performs multiple types of validation including:
    - Medical accuracy and consistency
    - PHI detection and protection
    - HIPAA compliance checking
    - Medication safety validation
    - Symptom validation
    - Dosage validation
    - Contraindication checking
    """
    
    validation_id = str(uuid.uuid4())
    
    logger.info(
        "Starting medical data validation",
        validation_id=validation_id,
        validation_types=request.validation_types,
        strict_mode=request.strict_mode,
        client_ip=None  # Would be extracted from request
    )
    
    try:
        # Convert validation types to enum
        validation_types = [ValidationResultType(vt) for vt in request.validation_types]
        
        # Perform validation
        validation_results = await validation_engine.validate_medical_data(
            data=request.data,
            validation_types=validation_types
        )
        
        # Generate recommendations
        recommendations = []
        if request.include_recommendations:
            recommendations = _generate_validation_recommendations(validation_results)
        
        # Calculate compliance score
        compliance_score = _calculate_compliance_score(validation_results)
        
        return ValidationResponse(
            validation_id=validation_id,
            timestamp=validation_results["timestamp"],
            overall_status=validation_results["overall_status"],
            validation_types=request.validation_types,
            results=validation_results["results"],
            summary=validation_results["summary"],
            recommendations=recommendations,
            compliance_score=compliance_score
        )
        
    except Exception as e:
        logger.error(f"Medical validation failed: {e}", validation_id=validation_id)
        raise MedicalValidationError(f"Validation failed: {str(e)}")


@router.post("/phi/analyze", response_model=PHIAnalysisResponse)
async def analyze_phi(data: PHIAnalysisRequest):
    """
    Comprehensive PHI detection and analysis.
    
    Analyzes data for PHI content and provides:
    - PHI type detection
    - Risk level assessment
    - Automated PHI protection
    - HIPAA compliance checking
    - Protection mode recommendations
    """
    
    analysis_id = str(uuid.uuid4())
    
    logger.info(
        "Starting PHI analysis",
        analysis_id=analysis_id,
        protection_mode=data.protection_mode,
        client_ip=None
    )
    
    try:
        # Detect PHI
        phi_instances = SecurityValidator.detect_phi(json.dumps(data.data))
        phi_detected = len(phi_instances) > 0
        
        # Determine PHI types
        phi_types = list(set(instance["type"] for instance in phi_instances))
        
        # Calculate risk level
        risk_level = _calculate_phi_risk_level(phi_types, phi_instances)
        
        # Apply PHI protection
        protected_data = data.data.copy()
        if phi_detected:
            protected_data = _apply_phi_protection(protected_data, data.protection_mode)
        
        # Determine compliance status
        compliance_status = _assess_hipaa_compliance(phi_types, risk_level)
        
        return PHIAnalysisResponse(
            analysis_id=analysis_id,
            phi_detected=phi_detected,
            phi_types=phi_types,
            phi_instances=phi_instances,
            risk_level=risk_level,
            protected_data=protected_data,
            compliance_status=compliance_status,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"PHI analysis failed: {e}", analysis_id=analysis_id)
        raise PHIProtectionError(f"PHI analysis failed: {str(e)}")


@router.post("/accuracy/validate", response_model=MedicalAccuracyResponse)
async def validate_medical_accuracy(request: MedicalAccuracyRequest):
    """
    Medical accuracy validation against clinical guidelines.
    
    Validates medical data for:
    - Clinical guideline compliance
    - Medical consistency
    - Evidence-based recommendations
    - Safety considerations
    - Best practice alignment
    """
    
    accuracy_id = str(uuid.uuid4())
    
    logger.info(
        "Starting medical accuracy validation",
        accuracy_id=accuracy_id,
        strict_validation=request.strict_validation,
        client_ip=None
    )
    
    try:
        # Perform medical accuracy validation
        validation_results = await validation_engine.validate_medical_data(
            data=request.medical_data,
            validation_types=[ValidationResultType.MEDICAL_ACCURACY]
        )
        
        # Extract accuracy-specific results
        accuracy_result = validation_results["results"]["medical_accuracy"]
        
        # Calculate accuracy score
        accuracy_score = _calculate_medical_accuracy_score(accuracy_result)
        
        # Analyze medical consistency
        consistency_analysis = _analyze_medical_consistency(request.medical_data)
        
        # Check guideline compliance
        guideline_compliance = _check_medical_guidelines(request.medical_data, request.reference_guidelines)
        
        # Generate recommendations
        recommendations = _generate_accuracy_recommendations(accuracy_result, consistency_analysis)
        
        return MedicalAccuracyResponse(
            accuracy_id=accuracy_id,
            accuracy_score=accuracy_score,
            validation_status=accuracy_result["status"],
            accuracy_issues=accuracy_result.get("errors", []),
            medical_consistency=consistency_analysis,
            guideline_compliance=guideline_compliance,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Medical accuracy validation failed: {e}", accuracy_id=accuracy_id)
        raise MedicalValidationError(f"Accuracy validation failed: {str(e)}")


@router.get("/compliance/hipaa")
async def get_hipaa_compliance_guidelines():
    """
    Get HIPAA compliance guidelines and best practices.
    
    Returns comprehensive HIPAA compliance information including:
    - Safe Harbor provisions
    - Expert determination guidelines
    - Technical safeguards
    - Administrative safeguards
    - Physical safeguards
    """
    
    guidelines = {
        "hipaa_version": "45 CFR Parts 160, 162, and 164",
        "last_updated": "2024-01-15",
        "safe_harbor_provisions": {
            "deidentified_data": [
                "Names",
                "Geographic subdivisions smaller than state",
                "All elements of dates (except year)",
                "Telephone numbers",
                "Fax numbers",
                "Email addresses",
                "Social Security numbers",
                "Medical record numbers",
                "Health plan beneficiary numbers",
                "Account numbers",
                "Certificate/license numbers",
                "Vehicle identifiers",
                "Device identifiers",
                "Web URLs",
                "Biometric identifiers",
                "Full-face photos",
                "Any other unique identifying numbers"
            ],
            "safe Harbor exceptions": [
                "Education records covered by FERPA",
                "Employment records held by covered entity",
                "Deceased persons information"
            ]
        },
        "expert_determination": {
            "requirements": [
                "Qualified statistician determination",
                "Documented methodology",
                "Documented testing and validation",
                "Ongoing monitoring and validation"
            ]
        },
        "technical_safeguards": {
            "access_control": [
                "Unique user identification",
                "Emergency access procedure",
                "Automatic logoff",
                "Encryption and decryption"
            ],
            "audit_controls": [
                "Hardware, software, and procedural mechanisms",
                "Recording access and modifications",
                "Regular audit review"
            ],
            "integrity": [
                "Data alteration protection",
                "Data corruption prevention",
                "Data authenticity verification"
            ],
            "person_or_entity_authentication": [
                "User identity verification",
                "Multi-factor authentication options"
            ],
            "transmission_security": [
                "End-to-end encryption",
                "Integrity controls",
                "Secure transmission protocols"
            ]
        },
        "administrative_safeguards": {
            "security_officer": "Designated security official responsibility",
            "workforce_training": "Regular security awareness training",
            "access_management": "User access authorization and management",
            "incident_response": "Security incident procedures",
            "contingency_plan": "Data backup and disaster recovery"
        },
        "physical_safeguards": {
            "facility_access": "Physical access to systems and equipment",
            "workstation_use": "Workstation use restrictions",
            "device_media": "Device and media controls"
        }
    }
    
    return guidelines


@router.post("/validation/guidelines/check")
async def check_against_guidelines(
    medical_data: Dict[str, Any],
    guidelines: List[str] = None
):
    """
    Check medical data against specific clinical guidelines.
    
    Validates medical decisions and data against:
    - Evidence-based clinical guidelines
    - Professional medical standards
    - Best practice recommendations
    - Safety protocols
    """
    
    if not guidelines:
        guidelines = ["general_medical_practice"]
    
    # Mock guideline checking (in production, this would query medical databases)
    compliance_results = {
        "guidelines_checked": guidelines,
        "compliance_status": "compliant",
        "violations": [],
        "recommendations": [],
        "evidence_level": "high",
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    # Basic compliance checks
    if "emergency_symptoms" in medical_data:
        emergency_symptoms = medical_data["emergency_symptoms"]
        if emergency_symptoms:
            compliance_results["recommendations"].append(
                "Emergency symptoms detected - follow emergency protocols"
            )
    
    if "medications" in medical_data:
        medications = medical_data["medications"]
        for medication in medications:
            if "dosage" in medication:
                # Basic dosage validation
                dosage = medication.get("dosage", 0)
                if dosage > 10000:  # Very high threshold
                    compliance_results["violations"].append(
                        f"Extremely high dosage detected: {dosage}"
                    )
    
    # Determine overall compliance
    if compliance_results["violations"]:
        compliance_results["compliance_status"] = "non_compliant"
    elif compliance_results["recommendations"]:
        compliance_results["compliance_status"] = "warning"
    
    return compliance_results


# Helper functions
def _generate_validation_recommendations(validation_results: Dict[str, Any]) -> List[str]:
    """Generate validation improvement recommendations"""
    
    recommendations = []
    
    # Analyze each validation result
    for validation_type, result in validation_results["results"].items():
        if isinstance(result, dict):
            # PHI recommendations
            if validation_type == "phi_detection":
                if result.get("phi_detected"):
                    recommendations.append("Implement PHI encryption for sensitive data")
                    recommendations.append("Review and minimize PHI collection")
                else:
                    recommendations.append("Continue current PHI protection measures")
            
            # Medical accuracy recommendations
            elif validation_type == "medical_accuracy":
                if result.get("errors"):
                    recommendations.append("Review and correct medical data inconsistencies")
                if result.get("warnings"):
                    recommendations.append("Address medical data warnings to improve accuracy")
            
            # HIPAA compliance recommendations
            elif validation_type == "hipaa_compliance":
                if result.get("violations"):
                    recommendations.append("Address HIPAA compliance violations immediately")
                recommendations.append("Implement comprehensive audit logging")
            
            # Medication safety recommendations
            elif validation_type == "medication_safety":
                if result.get("safety_issues"):
                    recommendations.append("Review medication dosages and interactions")
                if result.get("warnings"):
                    recommendations.append("Monitor patients for medication warnings")
    
    return recommendations


def _calculate_compliance_score(validation_results: Dict[str, Any]) -> float:
    """Calculate overall compliance score"""
    
    summary = validation_results["summary"]
    
    if summary["total_checks"] == 0:
        return 1.0
    
    # Base score from passed checks
    base_score = summary["passed_checks"] / summary["total_checks"]
    
    # Penalties for issues
    warning_penalty = summary["warnings"] * 0.05
    critical_penalty = summary["critical_issues"] * 0.2
    
    # Calculate final score
    compliance_score = max(0.0, base_score - warning_penalty - critical_penalty)
    
    return round(compliance_score, 3)


def _calculate_phi_risk_level(phi_types: List[str], phi_instances: List[Dict[str, Any]]) -> str:
    """Calculate PHI risk level"""
    
    high_risk_types = {"social_security_numbers", "biometric_identifiers", "photos"}
    medium_risk_types = {"names", "dates", "telephone_numbers", "email_addresses"}
    
    if any(phi_type in high_risk_types for phi_type in phi_types):
        return "high"
    elif any(phi_type in medium_risk_types for phi_type in phi_types):
        return "medium"
    elif phi_types:
        return "low"
    else:
        return "none"


def _assess_hipaa_compliance(phi_types: List[str], risk_level: str) -> str:
    """Assess HIPAA compliance status"""
    
    if risk_level == "high":
        return "requires_immediate_attention"
    elif risk_level == "medium":
        return "requires_review"
    elif risk_level == "low":
        return "compliant_with_monitoring"
    else:
        return "fully_compliant"


def _apply_phi_protection(data: Dict[str, Any], protection_mode: str) -> Dict[str, Any]:
    """Apply PHI protection to data"""
    
    protected_data = data.copy()
    data_str = json.dumps(data)
    
    # Apply protection based on mode
    if protection_mode == "mask":
        protected_data = json.loads(
            SecurityValidator.redact_phi(data_str, mode="mask")
        )
    elif protection_mode == "anonymize":
        protected_data = json.loads(
            SecurityValidator.redact_phi(data_str, mode="anonymize")
        )
    elif protection_mode == "hash":
        protected_data = json.loads(
            SecurityValidator.redact_phi(data_str, mode="hash")
        )
    
    return protected_data


def _calculate_medical_accuracy_score(accuracy_result: Dict[str, Any]) -> float:
    """Calculate medical accuracy score"""
    
    if not accuracy_result.get("checks"):
        return 0.5  # Neutral score if no checks performed
    
    total_checks = len(accuracy_result["checks"])
    passed_checks = sum(1 for check in accuracy_result["checks"] if check.get("status") == "passed")
    
    base_score = passed_checks / total_checks
    
    # Penalties for errors and warnings
    error_penalty = len(accuracy_result.get("errors", [])) * 0.2
    warning_penalty = len(accuracy_result.get("warnings", [])) * 0.1
    
    accuracy_score = max(0.0, base_score - error_penalty - warning_penalty)
    
    return round(accuracy_score, 3)


def _analyze_medical_consistency(medical_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze medical data consistency"""
    
    consistency_analysis = {
        "overall_consistency": "good",
        "inconsistencies": [],
        "cross_references": [],
        "confidence_level": 0.8
    }
    
    # Check vital signs consistency
    if "vital_signs" in medical_data:
        vital_signs = medical_data["vital_signs"]
        
        # Blood pressure consistency
        if "blood_pressure" in vital_signs:
            bp = vital_signs["blood_pressure"]
            if "systolic" in bp and "diastolic" in bp:
                if bp["systolic"] <= bp["diastolic"]:
                    consistency_analysis["inconsistencies"].append(
                        "Systolic BP must be higher than diastolic BP"
                    )
                    consistency_analysis["overall_consistency"] = "poor"
    
    # Check for logical relationships
    if "symptoms" in medical_data and "diagnoses" in medical_data:
        symptoms = [s.lower() for s in medical_data["symptoms"]]
        diagnoses = [d.lower() for d in medical_data["diagnoses"]]
        
        # Check for symptom-diagnosis alignment
        for diagnosis in diagnoses:
            if "pain" in diagnosis and "pain" not in symptoms:
                consistency_analysis["cross_references"].append(
                    f"Diagnosis '{diagnosis}' without corresponding symptom"
                )
    
    return consistency_analysis


def _check_medical_guidelines(medical_data: Dict[str, Any], guidelines: List[str]) -> Dict[str, Any]:
    """Check medical data against clinical guidelines"""
    
    compliance = {
        "guidelines_compliant": True,
        "violations": [],
        "recommendations": [],
        "evidence_level": "moderate",
        "confidence_score": 0.8
    }
    
    # Mock guideline checking
    for guideline in guidelines:
        if guideline == "vital_signs_ranges":
            if "vital_signs" in medical_data:
                vs = medical_data["vital_signs"]
                
                if "heart_rate" in vs:
                    hr = vs["heart_rate"]
                    if not (30 <= hr <= 200):
                        compliance["violations"].append(f"Heart rate {hr} outside normal range")
                        compliance["guidelines_compliant"] = False
        
        elif guideline == "medication_dosage_limits":
            if "medications" in medical_data:
                for med in medical_data["medications"]:
                    if "dosage" in med and med["dosage"] > 5000:  # High dosage threshold
                        compliance["violations"].append(
                            f"High medication dosage: {med.get('name', 'Unknown')} - {med['dosage']}"
                        )
                        compliance["guidelines_compliant"] = False
    
    return compliance


def _generate_accuracy_recommendations(
    accuracy_result: Dict[str, Any],
    consistency_analysis: Dict[str, Any]
) -> List[str]:
    """Generate medical accuracy improvement recommendations"""
    
    recommendations = []
    
    # Address errors
    if accuracy_result.get("errors"):
        recommendations.append("Correct identified medical data errors")
        recommendations.append("Implement data validation checks")
    
    # Address warnings
    if accuracy_result.get("warnings"):
        recommendations.append("Review medical data warnings")
        recommendations.append("Consider additional clinical validation")
    
    # Address inconsistencies
    if consistency_analysis.get("inconsistencies"):
        recommendations.append("Resolve medical data inconsistencies")
        recommendations.append("Implement cross-reference validation")
    
    # General recommendations
    recommendations.extend([
        "Implement regular data quality audits",
        "Establish medical data governance policies",
        "Train staff on medical data entry standards"
    ])
    
    return recommendations