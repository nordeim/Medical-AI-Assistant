"""
Custom Exceptions for Medical AI Inference API
Specialized exception handling for medical and security contexts
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class MedicalAIException(Exception):
    """Base exception for Medical AI API"""
    
    def __init__(self, detail: str, code: Optional[str] = None):
        self.detail = detail
        self.code = code or "MEDICAL_AI_ERROR"
        super().__init__(self.detail)


class MedicalValidationError(MedicalAIException):
    """Raised when medical data validation fails"""
    
    def __init__(self, detail: str, validation_type: Optional[str] = None):
        self.validation_type = validation_type
        super().__init__(detail, "MEDICAL_VALIDATION_ERROR")


class PHIProtectionError(MedicalAIException):
    """Raised when PHI protection is violated"""
    
    def __init__(self, detail: str, phi_type: Optional[str] = None):
        self.phi_type = phi_type
        super().__init__(detail, "PHI_PROTECTION_ERROR")


class ModelUnavailableError(MedicalAIException):
    """Raised when model service is unavailable"""
    
    def __init__(self, detail: str, model_name: Optional[str] = None):
        self.model_name = model_name
        super().__init__(detail, "MODEL_UNAVAILABLE_ERROR")


class RateLimitExceededError(MedicalAIException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, detail: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(detail, "RATE_LIMIT_EXCEEDED")


class AuthenticationError(MedicalAIException):
    """Raised when authentication fails"""
    
    def __init__(self, detail: str):
        super().__init__(detail, "AUTHENTICATION_ERROR")


class AuthorizationError(MedicalAIException):
    """Raised when authorization fails"""
    
    def __init__(self, detail: str):
        super().__init__(detail, "AUTHORIZATION_ERROR")


class ClinicalDecisionError(MedicalAIException):
    """Raised when clinical decision support fails"""
    
    def __init__(self, detail: str, confidence: Optional[float] = None):
        self.confidence = confidence
        super().__init__(detail, "CLINICAL_DECISION_ERROR")


class DataCorruptionError(MedicalAIException):
    """Raised when data corruption is detected"""
    
    def __init__(self, detail: str, data_type: Optional[str] = None):
        self.data_type = data_type
        super().__init__(detail, "DATA_CORRUPTION_ERROR")


class StreamingError(MedicalAIException):
    """Raised when streaming fails"""
    
    def __init__(self, detail: str, stream_id: Optional[str] = None):
        self.stream_id = stream_id
        super().__init__(detail, "STREAMING_ERROR")


class BatchProcessingError(MedicalAIException):
    """Raised when batch processing fails"""
    
    def __init__(self, detail: str, batch_id: Optional[str] = None):
        self.batch_id = batch_id
        super().__init__(detail, "BATCH_PROCESSING_ERROR")


class ConversationError(MedicalAIException):
    """Raised when conversation processing fails"""
    
    def __init__(self, detail: str, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        super().__init__(detail, "CONVERSATION_ERROR")


class ValidationError(MedicalAIException):
    """Raised when input validation fails"""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        self.field = field
        super().__init__(detail, "VALIDATION_ERROR")


class SecurityError(MedicalAIException):
    """Raised when security validation fails"""
    
    def __init__(self, detail: str, security_type: Optional[str] = None):
        self.security_type = security_type
        super().__init__(detail, "SECURITY_ERROR")


class HealthCheckError(MedicalAIException):
    """Raised when health check fails"""
    
    def __init__(self, detail: str, component: Optional[str] = None):
        self.component = component
        super().__init__(detail, "HEALTH_CHECK_ERROR")


class AuditLogError(MedicalAIException):
    """Raised when audit logging fails"""
    
    def __init__(self, detail: str, operation: Optional[str] = None):
        self.operation = operation
        super().__init__(detail, "AUDIT_LOG_ERROR")


class CacheError(MedicalAIException):
    """Raised when cache operations fail"""
    
    def __init__(self, detail: str, cache_type: Optional[str] = None):
        self.cache_type = cache_type
        super().__init__(detail, "CACHE_ERROR")


# HTTP Exception Mappers
EXCEPTION_MAPPING = {
    MedicalValidationError: (status.HTTP_400_BAD_REQUEST, "Medical Validation Failed"),
    PHIProtectionError: (status.HTTP_403_FORBIDDEN, "PHI Protection Violation"),
    ModelUnavailableError: (status.HTTP_503_SERVICE_UNAVAILABLE, "Model Unavailable"),
    RateLimitExceededError: (status.HTTP_429_TOO_MANY_REQUESTS, "Rate Limit Exceeded"),
    AuthenticationError: (status.HTTP_401_UNAUTHORIZED, "Authentication Failed"),
    AuthorizationError: (status.HTTP_403_FORBIDDEN, "Authorization Failed"),
    ClinicalDecisionError: (status.HTTP_422_UNPROCESSABLE_ENTITY, "Clinical Decision Failed"),
    DataCorruptionError: (status.HTTP_422_UNPROCESSABLE_ENTITY, "Data Corruption Detected"),
    StreamingError: (status.HTTP_503_SERVICE_UNAVAILABLE, "Streaming Failed"),
    BatchProcessingError: (status.HTTP_503_SERVICE_UNAVAILABLE, "Batch Processing Failed"),
    ConversationError: (status.HTTP_422_UNPROCESSABLE_ENTITY, "Conversation Processing Failed"),
    ValidationError: (status.HTTP_422_UNPROCESSABLE_ENTITY, "Validation Failed"),
    SecurityError: (status.HTTP_403_FORBIDDEN, "Security Validation Failed"),
    HealthCheckError: (status.HTTP_503_SERVICE_UNAVAILABLE, "Health Check Failed"),
    AuditLogError: (status.HTTP_500_INTERNAL_SERVER_ERROR, "Audit Logging Failed"),
    CacheError: (status.HTTP_503_SERVICE_UNAVAILABLE, "Cache Operation Failed"),
}


def to_http_exception(exc: MedicalAIException) -> HTTPException:
    """Convert MedicalAIException to HTTPException"""
    
    exception_class = type(exc)
    if exception_class in EXCEPTION_MAPPING:
        status_code, message = EXCEPTION_MAPPING[exception_class]
        
        response_data = {
            "error": message,
            "detail": exc.detail,
            "code": exc.code,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
        
        # Add specific attributes if present
        for attr in ["validation_type", "phi_type", "model_name", "confidence", 
                     "data_type", "stream_id", "batch_id", "conversation_id",
                     "field", "security_type", "component", "operation", "cache_type"]:
            if hasattr(exc, attr):
                response_data[attr] = getattr(exc, attr)
        
        return HTTPException(
            status_code=status_code,
            detail=response_data
        )
    
    # Default case
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "Internal Server Error",
            "detail": exc.detail,
            "code": exc.code,
            "timestamp": __import__("datetime").datetime.now().isoformat()
        }
    )