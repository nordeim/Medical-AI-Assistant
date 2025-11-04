"""
Medical Data Validation Middleware
HIPAA-compliant validation with medical accuracy checks
"""

import re
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from ..utils.exceptions import MedicalValidationError, PHIProtectionError
from ..utils.security import SecurityValidator
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Medical terms validation
MEDICAL_TERMS = {
    "anatomy": [
        "heart", "lung", "liver", "kidney", "brain", "stomach", "intestine", 
        "spleen", "pancreas", "thyroid", "adrenal", "prostate", "uterus", 
        "ovary", "testis", "skin", "muscle", "bone", "joint", "cartilage"
    ],
    "conditions": [
        "diabetes", "hypertension", "asthma", "pneumonia", "fracture", 
        "infection", "inflammation", "tumor", "cancer", "anemia", "arrhythmia",
        "stroke", "heart attack", "kidney failure", "liver disease", 
        "pulmonary embolism", "sepsis", "meningitis", "appendicitis"
    ],
    "medications": [
        "aspirin", "ibuprofen", "acetaminophen", "antibiotic", "insulin",
        "metformin", "lisinopril", "amlodipine", "atorvastatin", "omeprazole",
        "warfarin", "metoprolol", "furosemide", "levothyroxine", "gabapentin"
    ],
    "procedures": [
        "surgery", "biopsy", "endoscopy", "ct scan", "mri", "x-ray", 
        "ultrasound", "blood test", "urine test", "ecg", "eeg", "angioplasty",
        "stent", "dialysis", "transplant", "chemotherapy", "radiation therapy"
    ],
    "symptoms": [
        "pain", "fever", "cough", "shortness of breath", "nausea", "vomiting",
        "diarrhea", "headache", "dizziness", "fatigue", "weight loss",
        "blood pressure", "palpitations", "swelling", "rash", "infection"
    ],
    "measurements": [
        "blood pressure", "heart rate", "temperature", "oxygen saturation",
        "glucose level", "cholesterol", "hemoglobin", "white blood cell count",
        "platelet count", "creatinine", "alt", "ast", "bilirubin"
    ]
}

# Valid medical units
MEDICAL_UNITS = {
    "weight": ["kg", "lbs", "pounds"],
    "height": ["cm", "inches", "feet"],
    "temperature": ["celsius", "fahrenheit", "°c", "°f"],
    "pressure": ["mmhg", "kpa", "psi"],
    "heart_rate": ["bpm", "beats per minute"],
    "glucose": ["mg/dl", "mmol/l"],
    "cholesterol": ["mg/dl", "mmol/l"],
    "oxygen": ["%", "percent"]
}

# Medical validation rules
MEDICAL_VALIDATION_RULES = {
    "blood_pressure": {
        "pattern": r"\b(\d{2,3})[/\-](\d{2,3})\s*(mmhg)?\b",
        "systolic_range": (80, 250),
        "diastolic_range": (50, 150)
    },
    "heart_rate": {
        "pattern": r"\b(\d{2,3})\s*(bpm|beats per minute)?\b",
        "range": (30, 200)
    },
    "temperature": {
        "pattern": r"\b(\d{2,3}(\.\d+)?)[°\s]*(celsius|fahrenheit|c|f)?\b",
        "celsius_range": (35.0, 42.0),
        "fahrenheit_range": (95.0, 107.6)
    },
    "weight": {
        "pattern": r"\b(\d{2,3}(\.\d+)?)\s*(kg|lbs|pounds)?\b",
        "range": (30, 300)
    },
    "height": {
        "pattern": r"\b(\d{2,3}(\.\d+)?)\s*(cm|inches|feet)?\b",
        "cm_range": (100, 250),
        "inches_range": (39, 98)
    },
    "glucose": {
        "pattern": r"\b(\d{2,4}(\.\d+)?)\s*(mg/dl|mmol/l)?\b",
        "mgdl_range": (50, 600),
        "mmoll_range": (2.8, 33.3)
    }
}


class MedicalValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive medical data validation"""
    
    def __init__(self, app, call_next):
        super().__init__(app)
        self.call_next = call_next
        self.logger = get_logger("medical_validation")
    
    async def dispatch(self, request: Request, call_next):
        """Process request through medical validation pipeline"""
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            # Extract request data
            request_data = await self._extract_request_data(request)
            
            # Validate medical data
            validation_result = await self._validate_medical_data(request_data)
            
            if not validation_result["is_valid"]:
                self.logger.log_validation_failure(
                    validation_type=validation_result["validation_type"],
                    failure_reason=validation_result["error"],
                    data_sample=str(request_data)[:100],
                    user_id=self._get_user_id(request)
                )
                
                raise MedicalValidationError(
                    detail=validation_result["error"],
                    validation_type=validation_result["validation_type"]
                )
            
            # Add validated data to request state
            request.state.validated_data = validation_result["validated_data"]
            
            # Process response
            response = await call_next(request)
            
            # Log successful validation
            processing_time = time.time() - start_time
            self.logger.log_medical_operation(
                operation="medical_validation",
                success=True,
                details={
                    "processing_time": processing_time,
                    "validation_type": validation_result["validation_type"],
                    "client_ip": client_ip
                }
            )
            
            return response
            
        except (MedicalValidationError, PHIProtectionError):
            # Re-raise medical and PHI errors
            raise
        except Exception as e:
            # Log unexpected errors
            processing_time = time.time() - start_time
            self.logger.log_medical_operation(
                operation="medical_validation",
                success=False,
                details={
                    "error": str(e),
                    "processing_time": processing_time,
                    "client_ip": client_ip
                }
            )
            raise
        
        finally:
            # Add processing time header
            processing_time = time.time() - start_time
            request.state.processing_time = processing_time
    
    async def _extract_request_data(self, request: Request) -> Dict[str, Any]:
        """Extract and parse request data"""
        
        try:
            if request.method in ["POST", "PUT", "PATCH"]:
                body = await request.body()
                if body:
                    return json.loads(body.decode())
            elif request.method == "GET":
                return dict(request.query_params)
            
            return {}
            
        except json.JSONDecodeError:
            raise MedicalValidationError(
                detail="Invalid JSON in request body",
                validation_type="json_parse"
            )
        except Exception as e:
            raise MedicalValidationError(
                detail=f"Failed to extract request data: {str(e)}",
                validation_type="data_extraction"
            )
    
    async def _validate_medical_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive medical data validation"""
        
        validation_results = {
            "is_valid": True,
            "validation_type": "comprehensive",
            "error": None,
            "validated_data": data
        }
        
        try:
            # Validate text content for medical context
            if self._has_medical_content(data):
                medical_validation = self._validate_medical_content(data)
                if not medical_validation["is_valid"]:
                    return medical_validation
            
            # Validate medical measurements
            measurements_validation = self._validate_medical_measurements(data)
            if not measurements_validation["is_valid"]:
                return measurements_validation
            
            # Validate medical terminology
            terminology_validation = self._validate_medical_terminology(data)
            if not terminology_validation["is_valid"]:
                return terminology_validation
            
            # Validate dosage and medication data
            medication_validation = self._validate_medication_data(data)
            if not medication_validation["is_valid"]:
                return medication_validation
            
            return validation_results
            
        except Exception as e:
            return {
                "is_valid": False,
                "validation_type": "exception",
                "error": f"Validation exception: {str(e)}",
                "validated_data": data
            }
    
    def _has_medical_content(self, data: Dict[str, Any]) -> bool:
        """Check if data contains medical content"""
        
        medical_keywords = [
            "patient", "diagnosis", "treatment", "medication", "symptom",
            "condition", "procedure", "test", "result", "doctor", "nurse"
        ]
        
        def check_dict(d):
            for key, value in d.items():
                if isinstance(value, str) and any(keyword in value.lower() for keyword in medical_keywords):
                    return True
                elif isinstance(value, dict):
                    if check_dict(value):
                        return True
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and check_dict(item):
                            return True
                        elif isinstance(item, str) and any(keyword in item.lower() for keyword in medical_keywords):
                            return True
            return False
        
        return check_dict(data)
    
    def _validate_medical_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical content for proper context"""
        
        # Check for dangerous or inappropriate medical content
        dangerous_patterns = [
            r"self[- ]?harm",
            r"suicide",
            r"kill myself",
            r"end my life",
            r"illegal drugs",
            r"overdose"
        ]
        
        content_str = str(data).lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content_str):
                return {
                    "is_valid": False,
                    "validation_type": "medical_content",
                    "error": "Potentially dangerous medical content detected",
                    "validated_data": data
                }
        
        return {"is_valid": True, "validated_data": data}
    
    def _validate_medical_measurements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical measurements and vital signs"""
        
        for field, rules in MEDICAL_VALIDATION_RULES.items():
            if field in data:
                value = str(data[field])
                match = re.search(rules["pattern"], value, re.IGNORECASE)
                
                if match:
                    numeric_values = [float(g) for g in match.groups() if g and g.replace(".", "").isdigit()]
                    
                    if field == "blood_pressure" and len(numeric_values) >= 2:
                        systolic, diastolic = numeric_values[0], numeric_values[1]
                        if not (rules["systolic_range"][0] <= systolic <= rules["systolic_range"][1]):
                            return {
                                "is_valid": False,
                                "validation_type": "blood_pressure",
                                "error": f"Systolic pressure {systolic} outside valid range {rules['systolic_range']}",
                                "validated_data": data
                            }
                        if not (rules["diastolic_range"][0] <= diastolic <= rules["diastolic_range"][1]):
                            return {
                                "is_valid": False,
                                "validation_type": "blood_pressure", 
                                "error": f"Diastolic pressure {diastolic} outside valid range {rules['diastolic_range']}",
                                "validated_data": data
                            }
                    
                    elif field != "blood_pressure" and numeric_values:
                        value_range = rules.get("range") or rules.get("celsius_range") or rules.get("mgdl_range")
                        if value_range:
                            value = numeric_values[0]
                            if not (value_range[0] <= value <= value_range[1]):
                                return {
                                    "is_valid": False,
                                    "validation_type": field,
                                    "error": f"{field.replace('_', ' ').title()} value {value} outside valid range {value_range}",
                                    "validated_data": data
                                }
        
        return {"is_valid": True, "validated_data": data}
    
    def _validate_medical_terminology(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medical terminology usage"""
        
        content_str = str(data).lower()
        
        # Check for misspelled medical terms (basic check)
        medical_words = []
        for category, terms in MEDICAL_TERMS.items():
            medical_words.extend(terms)
        
        # Simple check for common misspellings
        common_misspellings = {
            "diabeties": "diabetes",
            "hypertention": "hypertension", 
            "pneumia": "pneumonia",
            "arithmia": "arrhythmia",
            "appendicitis": "appendicitis"
        }
        
        for misspelled, correct in common_misspellings.items():
            if misspelled in content_str:
                return {
                    "is_valid": False,
                    "validation_type": "medical_terminology",
                    "error": f"Possible misspelling detected: '{misspelled}' should be '{correct}'",
                    "validated_data": data
                }
        
        return {"is_valid": True, "validated_data": data}
    
    def _validate_medication_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate medication dosage and frequency"""
        
        # Check for dosage patterns
        dosage_patterns = [
            r"(\d+)\s*(mg|g|ml|units)\s*(once|twice|daily|bid|tid|qid)",
            r"(\d+)\s*(mg|g|ml)\s*every\s*(\d+)\s*(hours?|h)",
            r"take\s*(\d+)\s*(mg|g|ml)"
        ]
        
        content_str = str(data).lower()
        
        for pattern in dosage_patterns:
            matches = re.findall(pattern, content_str)
            for match in matches:
                # Basic validation for reasonable dosage ranges
                try:
                    dosage = float(match[0]) if match[0].replace(".", "").isdigit() else 0
                    
                    # Check common medication dosages
                    if "mg" in content_str and dosage > 1000:  # > 1g seems high
                        return {
                            "is_valid": False,
                            "validation_type": "medication_dosage",
                            "error": f"Potentially high dosage detected: {dosage}mg",
                            "validated_data": data
                        }
                except (ValueError, IndexError):
                    continue
        
        return {"is_valid": True, "validated_data": data}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        return request.client.host if request.client else "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # This would typically extract from JWT token or session
        return request.headers.get("x-user-id")