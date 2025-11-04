"""
Medical AI Resilience - Input Validation and Sanitization
Comprehensive input validation and sanitization with medical data integrity protection.
"""

import re
import json
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Type
from datetime import datetime, date
import logging
from abc import ABC, abstractmethod

from .errors import (
    MedicalError, MedicalErrorCode, MedicalErrorCategory,
    MedicalErrorSeverity, create_phi_violation_error
)


class ValidationLevel(Enum):
    """Validation levels for medical data."""
    BASIC = "basic"              # Basic format validation
    STANDARD = "standard"        # Standard medical validation
    STRICT = "strict"            # Strict validation with PHI checks
    CRITICAL = "critical"        # Critical validation with full audit
    SECURITY = "security"        # Security-focused validation


class DataType(Enum):
    """Supported data types for medical validation."""
    PATIENT_ID = "patient_id"
    MEDICAL_RECORD = "medical_record"
    DIAGNOSIS_CODE = "diagnosis_code"
    MEDICATION_ORDER = "medication_order"
    CLINICAL_NOTE = "clinical_note"
    LAB_RESULT = "lab_result"
    VITAL_SIGN = "vital_sign"
    PROCEDURE_CODE = "procedure_code"
    PHI_DATA = "phi_data"
    GENERIC_TEXT = "generic_text"
    JSON_DATA = "json_data"


class PHIField(Enum):
    """Protected Health Information fields that require special handling."""
    SSN = "social_security_number"
    MEDICAL_RECORD_NUMBER = "medical_record_number"
    BIOMETRIC_DATA = "biometric_data"
    FACE_IMAGE = "face_image"
    FINGERPRINT = "fingerprint"
    ADDRESS = "full_address"
    PHONE = "phone_number"
    EMAIL = "email_address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"
    DEVICE_ID = "device_identifier"


class ValidationResult:
    """Result of validation operation."""
    
    def __init__(
        self,
        is_valid: bool,
        errors: List[str] = None,
        warnings: List[str] = None,
        sanitized_data: Any = None,
        phi_detected: List[str] = None,
        audit_info: Dict[str, Any] = None
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.sanitized_data = sanitized_data
        self.phi_detected = phi_detected or []
        self.audit_info = audit_info or {}
        self.timestamp = datetime.utcnow()
    
    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "phi_detected": self.phi_detected,
            "has_phi": len(self.phi_detected) > 0,
            "timestamp": self.timestamp.isoformat()
        }


class BaseValidator(ABC):
    """Base validator class for medical data."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.audit_callback = None
        self.phi_callback = None
    
    def set_audit_callback(self, callback: Callable):
        """Set callback for audit events."""
        self.audit_callback = callback
    
    def set_phi_callback(self, callback: Callable):
        """Set callback for PHI detection events."""
        self.phi_callback = callback
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate input data."""
        pass
    
    def _audit_event(self, event_type: str, data: Any, result: ValidationResult):
        """Audit validation event."""
        if self.audit_callback:
            self.audit_callback({
                "event_type": event_type,
                "validation_level": self.validation_level.value,
                "data_type": type(data).__name__,
                "result": result.get_summary(),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    def _detect_phi(self, data: Any) -> List[str]:
        """Detect PHI in data."""
        phi_fields = []
        data_str = str(data).lower()
        
        # PHI patterns
        phi_patterns = {
            PHIField.SSN: [r'\b\d{3}-?\d{2}-?\d{4}\b', r'\bssn[:\s]*\d{3}-?\d{2}-?\d{4}\b'],
            PHIField.MEDICAL_RECORD_NUMBER: [r'\bmrn[:\s]*\d+\b', r'\bmedical[:\s]*record[:\s]*\d+\b'],
            PHIField.PHONE: [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'],
            PHIField.EMAIL: [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'],
            PHIField.IP_ADDRESS: [r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'],
            PHIField.DATE_OF_BIRTH: [r'\b(0[1-9]|1[0-2])[-/\.](0[1-9]|[12]\d|3[01])[-/\.](19|20)\d\d\b']
        }
        
        for field, patterns in phi_patterns.items():
            for pattern in patterns:
                if re.search(pattern, data_str):
                    phi_fields.append(field.value)
                    break
        
        if phi_fields and self.phi_callback:
            self.phi_callback({
                "phi_detected": phi_fields,
                "data_snippet": str(data)[:100],  # First 100 chars
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return phi_fields


class PatientIDValidator(BaseValidator):
    """Validator for patient IDs."""
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate patient ID."""
        result = ValidationResult(is_valid=True)
        
        # Basic validation
        if not isinstance(data, str):
            result.add_error("Patient ID must be a string")
            return result
        
        # Length check
        if len(data) < 3 or len(data) > 20:
            result.add_error("Patient ID must be between 3 and 20 characters")
        
        # Character check
        if not re.match(r'^[A-Za-z0-9_-]+$', data):
            result.add_error("Patient ID contains invalid characters")
        
        # PHI detection
        phi_detected = self._detect_phi(data)
        if phi_detected:
            result.add_warning(f"Potential PHI detected: {phi_detected}")
        
        # Security validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.CRITICAL]:
            if len(data) < 8:
                result.add_error("Patient ID too short for security requirements")
        
        result.sanitized_data = data.strip()
        result.phi_detected = phi_detected
        self._audit_event("patient_id_validation", data, result)
        
        return result


class MedicalRecordValidator(BaseValidator):
    """Validator for medical records."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        self.required_fields = ["patient_id", "record_date", "record_type"]
        self.phi_patterns = self._load_phi_patterns()
    
    def _load_phi_patterns(self) -> Dict[str, List[str]]:
        """Load PHI detection patterns."""
        return {
            "names": [r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', r'\bDr\.\s+[A-Z][a-z]+ [A-Z][a-z]+\b'],
            "addresses": [r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)\b'],
            "phone": [r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'],
            "email": [r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b']
        }
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate medical record."""
        result = ValidationResult(is_valid=True)
        
        # Basic validation
        if not isinstance(data, dict):
            result.add_error("Medical record must be a dictionary")
            return result
        
        # Required fields check
        for field in self.required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")
        
        # Validate each field
        for field_name, field_value in data.items():
            field_result = self._validate_field(field_name, field_value)
            result.errors.extend(field_result.errors)
            result.warnings.extend(field_result.warnings)
            result.phi_detected.extend(field_result.phi_detected)
            
            if not field_result.is_valid:
                result.is_valid = False
        
        # PHI detection
        data_str = json.dumps(data).lower()
        for phi_type, patterns in self.phi_patterns.items():
            for pattern in patterns:
                if re.search(pattern, data_str, re.IGNORECASE):
                    result.phi_detected.append(f"medical_record.{phi_type}")
        
        if result.phi_detected:
            result.add_warning(f"PHI detected in medical record: {result.phi_detected}")
        
        result.sanitized_data = self._sanitize_record(data, result.phi_detected)
        self._audit_event("medical_record_validation", data, result)
        
        return result
    
    def _validate_field(self, field_name: str, field_value: Any) -> ValidationResult:
        """Validate individual field."""
        result = ValidationResult(is_valid=True)
        
        # Field-specific validation
        if field_name == "patient_id":
            patient_validator = PatientIDValidator(self.validation_level)
            field_result = patient_validator.validate(field_value)
            result.errors.extend(field_result.errors)
            result.warnings.extend(field_result.warnings)
            result.phi_detected.extend(field_result.phi_detected)
            if not field_result.is_valid:
                result.is_valid = False
        
        elif field_name == "record_date":
            if not self._is_valid_date(field_value):
                result.add_error("Invalid record date format")
        
        elif field_name in ["clinical_notes", "notes"]:
            # Text content validation
            if isinstance(field_value, str):
                if len(field_value) > 10000:  # 10KB limit
                    result.add_error("Clinical notes too long")
                if self.validation_level == ValidationLevel.CRITICAL:
                    if any(word in field_value.lower() for word in ["confidential", "secret", "private"]):
                        result.add_warning("Potentially sensitive content detected")
        
        # PHI detection for field
        phi_detected = self._detect_phi(field_value)
        result.phi_detected.extend(phi_detected)
        
        return result
    
    def _is_valid_date(self, date_value: Any) -> bool:
        """Validate date format."""
        if isinstance(date_value, (date, datetime)):
            return True
        
        if isinstance(date_value, str):
            try:
                datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return True
            except ValueError:
                return False
        
        return False
    
    def _sanitize_record(self, data: Dict[str, Any], phi_detected: List[str]) -> Dict[str, Any]:
        """Sanitize medical record."""
        sanitized = data.copy()
        
        # Remove or mask PHI if detected
        for phi_field in phi_detected:
            if "." in phi_field:
                field_path = phi_field.split(".")
                if len(field_path) == 2:
                    field_name = field_path[1]
                    if field_name in sanitized:
                        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.CRITICAL]:
                            sanitized[field_name] = "[REDACTED_PHI]"
                        else:
                            sanitized[field_name] = self._mask_phi(sanitized[field_name])
        
        return sanitized
    
    def _mask_phi(self, phi_value: Any) -> Any:
        """Mask PHI in value."""
        if isinstance(phi_value, str):
            if len(phi_value) <= 4:
                return "*" * len(phi_value)
            else:
                return phi_value[:2] + "*" * (len(phi_value) - 4) + phi_value[-2:]
        return str(phi_value)


class DiagnosisCodeValidator(BaseValidator):
    """Validator for medical diagnosis codes."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        # Common diagnosis code patterns
        self.icd10_pattern = re.compile(r'^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$')
        self.icd9_pattern = re.compile(r'^\d{3}\.?\d{0,2}$')
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate diagnosis code."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(data, str):
            result.add_error("Diagnosis code must be a string")
            return result
        
        code = data.strip().upper()
        
        # Pattern matching
        is_icd10 = bool(self.icd10_pattern.match(code))
        is_icd9 = bool(self.icd9_pattern.match(code))
        
        if not (is_icd10 or is_icd9):
            result.add_error("Invalid diagnosis code format (must be ICD-10 or ICD-9)")
        
        # Length validation
        if is_icd10 and len(code) > 7:
            result.add_warning("ICD-10 code longer than expected")
        elif is_icd9 and len(code) > 5:
            result.add_warning("ICD-9 code longer than expected")
        
        result.sanitized_data = code
        self._audit_event("diagnosis_code_validation", data, result)
        
        return result


class ClinicalTextValidator(BaseValidator):
    """Validator for clinical text content."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        super().__init__(validation_level)
        # Dangerous content patterns
        self.dangerous_patterns = [
            r'\b(kill|murder|harm|violence|weapon|bomb|explosive)\b',
            r'\b(suicide|self.?harm|end.?life|die|death)\b',
            r'\b(drug|medication|overdose|poison|toxic)\b'
        ]
        
        # PHI patterns in text
        self.phi_patterns = [
            r'\b\d{3}-?\d{2}-?\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone alternative
        ]
    
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Validate clinical text."""
        result = ValidationResult(is_valid=True)
        
        if not isinstance(data, str):
            result.add_error("Clinical text must be a string")
            return result
        
        text = data.strip()
        
        # Length validation
        if len(text) == 0:
            result.add_error("Clinical text cannot be empty")
        elif len(text) > 50000:  # 50KB limit
            result.add_error("Clinical text too long")
        
        # Dangerous content detection
        dangerous_matches = []
        for pattern in self.dangerous_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dangerous_matches.extend(matches)
        
        if dangerous_matches:
            if self.validation_level == ValidationLevel.CRITICAL:
                result.add_error(f"Potentially dangerous content detected: {dangerous_matches}")
            else:
                result.add_warning(f"Potentially sensitive content detected: {dangerous_matches}")
        
        # PHI detection
        phi_matches = []
        for pattern in self.phi_patterns:
            matches = re.findall(pattern, text)
            phi_matches.extend(matches)
        
        if phi_matches:
            result.phi_detected.extend(["text.ssn", "text.email", "text.phone"])
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.CRITICAL]:
                result.add_warning(f"PHI detected in text content")
        
        # Content sanitization
        result.sanitized_data = self._sanitize_text(text, dangerous_matches, phi_matches)
        self._audit_event("clinical_text_validation", data, result)
        
        return result
    
    def _sanitize_text(self, text: str, dangerous: List[str], phi: List[str]) -> str:
        """Sanitize clinical text."""
        sanitized = text
        
        # Replace dangerous content with safe alternatives
        if dangerous:
            sanitized = re.sub(r'\b(kill|murder|harm|violence|weapon|bomb|explosive)\b', '[CONTENT_REMOVED]', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'\b(suicide|self.?harm|end.?life|die|death)\b', '[MENTAL_HEALTH_REFERRAL]', sanitized, flags=re.IGNORECASE)
        
        # Mask PHI if in strict mode
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.CRITICAL]:
            sanitized = re.sub(r'\b\d{3}-?\d{2}-?\d{4}\b', '[SSN_MASKED]', sanitized)
            sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_MASKED]', sanitized)
            sanitized = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_MASKED]', sanitized)
        
        return sanitized


class DataValidator:
    """Central data validation manager."""
    
    def __init__(self):
        self.validators: Dict[DataType, BaseValidator] = {
            DataType.PATIENT_ID: PatientIDValidator(),
            DataType.MEDICAL_RECORD: MedicalRecordValidator(),
            DataType.DIAGNOSIS_CODE: DiagnosisCodeValidator(),
            DataType.CLINICAL_NOTE: ClinicalTextValidator(),
            DataType.GENERIC_TEXT: ClinicalTextValidator(),
            DataType.JSON_DATA: MedicalRecordValidator()  # Reuse for JSON
        }
        self.validation_cache: Dict[str, ValidationResult] = {}
        self.max_cache_size = 1000
    
    def register_validator(self, data_type: DataType, validator: BaseValidator):
        """Register custom validator."""
        self.validators[data_type] = validator
    
    def validate(
        self,
        data: Any,
        data_type: DataType,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        use_cache: bool = True,
        **kwargs
    ) -> ValidationResult:
        """Validate data with specified type and level."""
        
        # Check cache
        cache_key = self._generate_cache_key(data, data_type, validation_level)
        if use_cache and cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Get validator
        validator = self.validators.get(data_type)
        if not validator:
            raise ValueError(f"No validator registered for data type: {data_type}")
        
        # Set validation level
        validator.validation_level = validation_level
        
        # Validate
        result = validator.validate(data, **kwargs)
        
        # Cache result
        if use_cache:
            self._cache_result(cache_key, result)
        
        return result
    
    def batch_validate(
        self,
        data_items: List[Tuple[Any, DataType]],
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        **kwargs
    ) -> List[ValidationResult]:
        """Validate multiple data items."""
        results = []
        
        for data, data_type in data_items:
            try:
                result = self.validate(data, data_type, validation_level, **kwargs)
                results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    is_valid=False,
                    errors=[f"Validation failed: {str(e)}"]
                )
                results.append(error_result)
        
        return results
    
    def validate_with_safety_checks(
        self,
        data: Any,
        data_type: DataType,
        patient_id: Optional[str] = None,
        clinical_priority: str = "normal"
    ) -> ValidationResult:
        """Validate with additional safety checks for medical data."""
        
        # Determine validation level based on clinical priority
        if clinical_priority == "critical":
            validation_level = ValidationLevel.CRITICAL
        elif clinical_priority == "high":
            validation_level = ValidationLevel.STRICT
        else:
            validation_level = ValidationLevel.STANDARD
        
        result = self.validate(data, data_type, validation_level)
        
        # Add medical-specific checks
        if not result.is_valid and clinical_priority in ["critical", "high"]:
            # For critical data, try to salvage valid portions
            result.sanitized_data = self._attempt_data_recovery(data, data_type)
            result.is_valid = True
            result.warnings.append("Data recovered through salvage process")
        
        # Audit validation
        if patient_id:
            result.audit_info["patient_id"] = patient_id
            result.audit_info["clinical_priority"] = clinical_priority
        
        return result
    
    def _generate_cache_key(self, data: Any, data_type: DataType, validation_level: ValidationLevel) -> str:
        """Generate cache key for validation result."""
        data_hash = str(hash(str(data)))  # Simple hash
        return f"{data_type.value}_{validation_level.value}_{data_hash}"
    
    def _cache_result(self, cache_key: str, result: ValidationResult):
        """Cache validation result."""
        if len(self.validation_cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.validation_cache))
            del self.validation_cache[oldest_key]
        
        self.validation_cache[cache_key] = result
    
    def _attempt_data_recovery(self, data: Any, data_type: DataType) -> Any:
        """Attempt to recover usable data from invalid input."""
        if isinstance(data, str):
            # Remove obviously invalid characters
            cleaned = re.sub(r'[^\w\s\-.,;:()/\[\]{}]', '', data)
            cleaned = ' '.join(cleaned.split())  # Normalize whitespace
            return cleaned if cleaned else "[INVALID_DATA]"
        elif isinstance(data, dict):
            # Keep only valid-looking fields
            valid_fields = {}
            for key, value in data.items():
                if isinstance(key, str) and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                    valid_fields[key] = str(value) if not isinstance(value, (dict, list)) else "[COMPLEX_DATA]"
            return valid_fields if valid_fields else {"recovered": False}
        else:
            return str(data)
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total_validations = len(self.validation_cache)
        valid_count = sum(1 for result in self.validation_cache.values() if result.is_valid)
        
        phi_detected_count = sum(
            len(result.phi_detected) for result in self.validation_cache.values()
        )
        
        return {
            "total_validations": total_validations,
            "valid_results": valid_count,
            "invalid_results": total_validations - valid_count,
            "success_rate": valid_count / max(total_validations, 1),
            "phi_detections": phi_detected_count,
            "cached_results": total_validations,
            "validator_count": len(self.validators),
            "validation_types": list(self.validators.keys())
        }


# Global data validator instance
data_validator = DataValidator()


def validate_input(
    data_type: DataType,
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    patient_id: Optional[str] = None,
    clinical_priority: str = "normal",
    use_cache: bool = True
) -> Callable:
    """Decorator for automatic input validation."""
    
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            # Validate first argument (typically input data)
            if args:
                result = data_validator.validate_with_safety_checks(
                    args[0],
                    data_type,
                    patient_id,
                    clinical_priority
                )
                
                if not result.is_valid:
                    raise create_phi_violation_error(
                        f"Input validation failed: {'; '.join(result.errors)}",
                        patient_context={"patient_id": patient_id} if patient_id else None
                    )
                
                # Replace first argument with sanitized data
                args = (result.sanitized_data,) + args[1:]
            
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            if args:
                result = data_validator.validate_with_safety_checks(
                    args[0],
                    data_type,
                    patient_id,
                    clinical_priority
                )
                
                if not result.is_valid:
                    raise create_phi_violation_error(
                        f"Input validation failed: {'; '.join(result.errors)}",
                        patient_context={"patient_id": patient_id} if patient_id else None
                    )
                
                args = (result.sanitized_data,) + args[1:]
            
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Specific validation decorators for common medical data types
def validate_patient_id(patient_id_param: str = "patient_id"):
    """Decorator to validate patient ID parameters."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Find patient_id in kwargs or args
            patient_id = kwargs.get(patient_id_param)
            if patient_id is None and args:
                # Assume first arg is data dict with patient_id
                if isinstance(args[0], dict) and patient_id_param in args[0]:
                    patient_id = args[0][patient_id_param]
            
            result = data_validator.validate_with_safety_checks(
                patient_id,
                DataType.PATIENT_ID,
                patient_id,
                "high"  # Patient IDs are always high priority
            )
            
            if not result.is_valid:
                raise create_phi_violation_error(
                    f"Patient ID validation failed: {'; '.join(result.errors)}"
                )
            
            # Update the parameter
            if patient_id in kwargs:
                kwargs[patient_id_param] = result.sanitized_data
            elif isinstance(args[0], dict) and patient_id_param in args[0]:
                new_args = list(args)
                new_args[0] = result.sanitized_data
                args = tuple(new_args)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator