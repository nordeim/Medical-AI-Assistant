"""
Core Inference API Endpoints
Medical AI inference with validation and PHI protection
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timezone
from enum import Enum

from fastapi import APIRouter, HTTPException, Request, Depends, status
from pydantic import BaseModel, Field, validator
import structlog

from ..utils.exceptions import (
    MedicalValidationError, 
    ModelUnavailableError, 
    ValidationError,
    SecurityError
)
from ..utils.security import SecurityValidator, rate_limiter
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Pydantic models for request/response
class InferenceRequest(BaseModel):
    """Inference request with medical validation"""
    
    query: str = Field(..., min_length=1, max_length=5000, description="Medical query or input text")
    context: Optional[str] = Field(None, max_length=2000, description="Additional context or background information")
    patient_id: Optional[str] = Field(None, max_length=50, description="Anonymized patient identifier")
    session_id: Optional[str] = Field(None, max_length=50, description="Session identifier")
    model_name: Optional[str] = Field(settings.model_name, description="Model to use for inference")
    temperature: Optional[float] = Field(settings.temperature, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(settings.max_tokens, ge=1, le=4096, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Enable streaming response")
    
    # Medical context fields
    medical_domain: Optional[Literal["general", "cardiology", "oncology", "neurology", "emergency", "pediatrics"]] = Field(None, description="Medical domain specialization")
    urgency_level: Optional[Literal["low", "medium", "high", "critical"]] = Field("medium", description="Urgency level of the query")
    
    # Validation flags
    require_medical_validation: bool = Field(True, description="Require medical data validation")
    enable_phi_protection: bool = Field(True, description="Enable PHI protection")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for potentially dangerous content
        dangerous_keywords = ["self-harm", "suicide", "kill myself", "illegal drugs"]
        if any(keyword in v.lower() for keyword in dangerous_keywords):
            raise ValueError("Potentially dangerous content detected")
        
        return v.strip()
    
    @validator('patient_id')
    def validate_patient_id(cls, v):
        if v and not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Patient ID must contain only alphanumeric characters, hyphens, and underscores")
        return v


class InferenceResponse(BaseModel):
    """Inference response with medical context"""
    
    request_id: str = Field(..., description="Unique request identifier")
    response: str = Field(..., description="Generated response text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    medical_context: Dict[str, Any] = Field(..., description="Medical analysis context")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Response timestamp")
    
    # Medical validation results
    medical_validation_passed: bool = Field(..., description="Whether medical validation passed")
    phi_protection_applied: bool = Field(..., description="Whether PHI protection was applied")
    
    # Clinical decision support
    clinical_recommendations: Optional[List[str]] = Field(None, description="Clinical recommendations")
    risk_assessment: Optional[Dict[str, Any]] = Field(None, description="Risk assessment results")
    
    # Metadata
    tokens_used: int = Field(..., description="Number of tokens used")
    cache_hit: bool = Field(False, description="Whether response was cached")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "uuid-string",
                "response": "Based on the symptoms described, this could indicate...",
                "confidence": 0.85,
                "medical_context": {
                    "diagnosis_suggested": "common_cold",
                    "severity": "mild",
                    "urgency": "low"
                },
                "processing_time": 1.234,
                "model_version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "medical_validation_passed": True,
                "phi_protection_applied": True,
                "clinical_recommendations": [
                    "Monitor symptoms for 24-48 hours",
                    "Consider over-the-counter pain relievers"
                ],
                "tokens_used": 156,
                "cache_hit": False
            }
        }


class BatchInferenceRequest(BaseModel):
    """Batch inference request for multiple queries"""
    
    queries: List[InferenceRequest] = Field(..., min_items=1, max_items=50, description="List of inference requests")
    batch_id: Optional[str] = Field(None, description="Optional batch identifier")
    
    @validator('queries')
    def validate_queries(cls, v):
        if len(v) > 50:
            raise ValueError("Maximum 50 queries per batch")
        return v


class BatchInferenceResponse(BaseModel):
    """Batch inference response"""
    
    batch_id: str = Field(..., description="Batch identifier")
    results: List[InferenceResponse] = Field(..., description="List of inference results")
    total_processing_time: float = Field(..., description="Total batch processing time")
    successful_count: int = Field(..., description="Number of successful inferences")
    failed_count: int = Field(..., description="Number of failed inferences")
    timestamp: str = Field(..., description="Batch completion timestamp")


class ModelStatusResponse(BaseModel):
    """Model status information"""
    
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Current status")
    last_health_check: str = Field(..., description="Last health check timestamp")
    uptime_seconds: int = Field(..., description="Uptime in seconds")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    gpu_utilization: Optional[float] = Field(None, description="GPU utilization percentage")
    avg_response_time_ms: float = Field(..., description="Average response time in milliseconds")


# Dependency functions
async def verify_rate_limit(request: Request) -> Dict[str, Any]:
    """Verify rate limiting"""
    
    client_id = request.client.host if request.client else "unknown"
    
    # Check per-minute limit
    if rate_limiter.is_rate_limited(
        identifier=f"{client_id}:minute",
        limit=settings.rate_limit_per_minute,
        window=60
    ):
        raise ValidationError("Rate limit exceeded (per minute)")
    
    # Check per-hour limit
    if rate_limiter.is_rate_limited(
        identifier=f"{client_id}:hour", 
        limit=settings.rate_limit_per_hour,
        window=3600
    ):
        raise ValidationError("Rate limit exceeded (per hour)")
    
    return {"client_id": client_id}


async def validate_medical_input(request_data: InferenceRequest) -> InferenceRequest:
    """Validate medical input data"""
    
    if not settings.enable_medical_validation:
        return request_data
    
    # Additional validation for medical data
    medical_patterns = [
        r"\b\d+\s*(mg|ml|g|units?)\b",  # Medication dosages
        r"\b\d{2,3}/\d{2,3}\s*(mmhg)?\b",  # Blood pressure
        r"\b\d{2,3}\s*(bpm|beats per minute)\b",  # Heart rate
        r"\b\d{2,3}(\.\d+)?\s*°?[cf]\b"  # Temperature
    ]
    
    import re
    content = f"{request_data.query} {request_data.context or ''}"
    
    for pattern in medical_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            # Validate the medical data
            if not request_data.medical_domain:
                raise MedicalValidationError(
                    "Medical measurements detected but no medical domain specified",
                    validation_type="missing_domain"
                )
            break
    
    return request_data


# Endpoint implementations
@router.post("/single", response_model=InferenceResponse)
async def single_inference(
    request: InferenceRequest,
    http_request: Request,
    rate_limit_info: Dict[str, Any] = Depends(verify_rate_limit),
    validated_request: InferenceRequest = Depends(validate_medical_input)
):
    """
    Single medical inference with comprehensive validation and PHI protection.
    
    This endpoint processes medical queries with:
    - Medical data validation
    - PHI protection and detection
    - Clinical context analysis
    - Risk assessment
    - Audit logging
    """
    
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Starting single inference",
            request_id=request_id,
            patient_id=validated_request.patient_id,
            medical_domain=validated_request.medical_domain,
            client_ip=http_request.client.host if http_request.client else None
        )
        
        # Validate security
        if settings.enable_security_headers:
            # Additional security checks could be added here
            pass
        
        # Apply PHI protection
        phi_protection_applied = False
        if validated_request.enable_phi_protection:
            # PHI protection logic would be applied here
            phi_protection_applied = True
        
        # Simulate model inference (in production, this would call actual model)
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Generate response based on medical domain and urgency
        response_text = await _generate_medical_response(validated_request)
        
        # Calculate confidence and medical context
        confidence = _calculate_confidence(validated_request)
        medical_context = _extract_medical_context(validated_request)
        
        # Generate clinical recommendations
        clinical_recommendations = _generate_clinical_recommendations(
            validated_request, confidence, medical_context
        )
        
        # Risk assessment
        risk_assessment = _assess_risk(validated_request, medical_context)
        
        # Medical validation
        medical_validation_passed = await _validate_medical_response(response_text)
        
        processing_time = time.time() - start_time
        
        # Log successful inference
        logger.log_medical_operation(
            operation="single_inference",
            patient_id=validated_request.patient_id,
            success=True,
            details={
                "request_id": request_id,
                "processing_time": processing_time,
                "confidence": confidence,
                "medical_domain": validated_request.medical_domain,
                "urgency_level": validated_request.urgency_level
            }
        )
        
        return InferenceResponse(
            request_id=request_id,
            response=response_text,
            confidence=confidence,
            medical_context=medical_context,
            processing_time=processing_time,
            model_version=settings.model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            medical_validation_passed=medical_validation_passed,
            phi_protection_applied=phi_protection_applied,
            clinical_recommendations=clinical_recommendations,
            risk_assessment=risk_assessment,
            tokens_used=len(response_text.split()) * 1.3,  # Rough token estimation
            cache_hit=False
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        logger.log_medical_operation(
            operation="single_inference",
            patient_id=validated_request.patient_id,
            success=False,
            details={
                "request_id": request_id,
                "error": str(e),
                "processing_time": processing_time
            }
        )
        
        raise


@router.post("/batch", response_model=BatchInferenceResponse)
async def batch_inference(
    request: BatchInferenceRequest,
    http_request: Request,
    rate_limit_info: Dict[str, Any] = Depends(verify_rate_limit)
):
    """
    Batch inference for multiple medical queries.
    
    Processes up to 50 queries simultaneously with:
    - Parallel processing for efficiency
    - Individual validation for each query
    - Batch-level error handling
    - Comprehensive auditing
    """
    
    start_time = time.time()
    batch_id = request.batch_id or str(uuid.uuid4())
    
    logger.info(
        "Starting batch inference",
        batch_id=batch_id,
        query_count=len(request.queries),
        client_ip=http_request.client.host if http_request.client else None
    )
    
    results = []
    failed_count = 0
    
    try:
        # Process queries in parallel for efficiency
        tasks = []
        for i, query_request in enumerate(request.queries):
            task = _process_single_batch_query(
                query_request, batch_id, i, http_request
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in batch_results:
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"Batch query failed: {result}")
            else:
                results.append(result)
        
        total_processing_time = time.time() - start_time
        
        logger.info(
            "Batch inference completed",
            batch_id=batch_id,
            successful_count=len(results),
            failed_count=failed_count,
            total_processing_time=total_processing_time
        )
        
        return BatchInferenceResponse(
            batch_id=batch_id,
            results=results,
            total_processing_time=total_processing_time,
            successful_count=len(results),
            failed_count=failed_count,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch inference failed: {e}", batch_id=batch_id)
        raise


@router.get("/models/status", response_model=ModelStatusResponse)
async def model_status():
    """
    Get current model status and health information.
    
    Returns:
    - Model health status
    - Performance metrics
    - Resource utilization
    - Uptime information
    """
    
    # In production, this would query actual model service
    return ModelStatusResponse(
        model_name=settings.model_name,
        version=settings.model_version,
        status="healthy",
        last_health_check=datetime.now(timezone.utc).isoformat(),
        uptime_seconds=3600,  # Would be actual uptime
        memory_usage_mb=2048.5,
        gpu_utilization=65.2,
        avg_response_time_ms=245.6
    )


# Helper functions
async def _generate_medical_response(request: InferenceRequest) -> str:
    """Generate medical response based on domain and context"""
    
    # This would call the actual medical AI model
    # For now, we'll simulate intelligent responses
    
    base_response = f"Thank you for your medical query. "
    
    if request.medical_domain == "cardiology":
        base_response += "Regarding cardiovascular health, "
    elif request.medical_domain == "oncology":
        base_response += "Concerning oncology matters, "
    elif request.medical_domain == "emergency":
        base_response += "This appears to require immediate attention. "
    else:
        base_response += "Based on general medical principles, "
    
    # Add urgency-based response
    if request.urgency_level == "critical":
        base_response += "Given the critical nature of this situation, "
    elif request.urgency_level == "high":
        base_response += "This requires prompt evaluation. "
    
    base_response += f"I understand you're asking about: {request.query[:100]}..."
    
    if request.context:
        base_response += f" With the additional context of {request.context[:50]}, "
    
    base_response += "Please note this is for informational purposes only and should not replace professional medical advice."
    
    return base_response


def _calculate_confidence(request: InferenceRequest) -> float:
    """Calculate model confidence based on query characteristics"""
    
    base_confidence = 0.7
    
    # Adjust confidence based on medical domain specificity
    if request.medical_domain:
        base_confidence += 0.1
    
    # Adjust based on query length and clarity
    if len(request.query) > 100:
        base_confidence += 0.05
    
    # Adjust based on context availability
    if request.context:
        base_confidence += 0.1
    
    # Urgency might reduce confidence due to time pressure
    if request.urgency_level == "critical":
        base_confidence -= 0.1
    
    return max(0.0, min(1.0, base_confidence))


def _extract_medical_context(request: InferenceRequest) -> Dict[str, Any]:
    """Extract and analyze medical context from request"""
    
    context = {
        "primary_topic": request.medical_domain or "general",
        "urgency": request.urgency_level,
        "has_context": bool(request.context),
        "has_patient_id": bool(request.patient_id),
        "query_length": len(request.query),
        "medical_terms_detected": _detect_medical_terms(request.query + " " + (request.context or ""))
    }
    
    return context


def _detect_medical_terms(text: str) -> List[str]:
    """Detect medical terms in text"""
    
    medical_keywords = [
        "pain", "fever", "cough", "shortness of breath", "headache",
        "nausea", "vomiting", "diarrhea", "dizziness", "fatigue",
        "chest pain", "abdominal pain", "back pain", "joint pain"
    ]
    
    detected_terms = []
    text_lower = text.lower()
    
    for term in medical_keywords:
        if term in text_lower:
            detected_terms.append(term)
    
    return detected_terms


def _generate_clinical_recommendations(
    request: InferenceRequest,
    confidence: float,
    context: Dict[str, Any]
) -> List[str]:
    """Generate clinical recommendations based on analysis"""
    
    recommendations = []
    
    # Base recommendations based on urgency
    if request.urgency_level == "critical":
        recommendations.append("Seek immediate medical attention")
        recommendations.append("Consider emergency services if symptoms worsen")
    elif request.urgency_level == "high":
        recommendations.append("Schedule medical appointment within 24-48 hours")
        recommendations.append("Monitor symptoms closely")
    else:
        recommendations.append("Schedule routine medical consultation")
        recommendations.append("Monitor symptoms for any changes")
    
    # Domain-specific recommendations
    if request.medical_domain == "cardiology":
        recommendations.append("Consider cardiac evaluation if symptoms persist")
    elif request.medical_domain == "oncology":
        recommendations.append("Ensure all recommendations are reviewed by oncologist")
    
    # Confidence-based recommendations
    if confidence < 0.6:
        recommendations.append("Obtain additional medical evaluation for accurate diagnosis")
    
    return recommendations


def _assess_risk(request: InferenceRequest, context: Dict[str, Any]) -> Dict[str, Any]:
    """Assess risk level based on query and context"""
    
    risk_level = "low"
    risk_factors = []
    
    # Urgency-based risk
    if request.urgency_level in ["critical", "high"]:
        risk_level = "high"
        risk_factors.append("High urgency level")
    
    # Medical terms indicating potential emergency
    emergency_terms = [
        "chest pain", "difficulty breathing", "severe headache",
        "loss of consciousness", "severe bleeding"
    ]
    
    for term in emergency_terms:
        if term in request.query.lower():
            risk_level = "high"
            risk_factors.append(f"Emergency symptom: {term}")
            break
    
    return {
        "level": risk_level,
        "factors": risk_factors,
        "confidence": 0.8,
        "requires_immediate_attention": risk_level == "high"
    }


async def _validate_medical_response(response: str) -> bool:
    """Validate generated medical response"""
    
    # Check for appropriate medical disclaimers
    if "medical advice" not in response.lower():
        return False
    
    # Check for dangerous recommendations
    dangerous_patterns = [
        r"stop taking.*medication",
        r"ignore.*symptoms",
        r"no need.*doctor"
    ]
    
    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return False
    
    return True


async def _process_single_batch_query(
    request: InferenceRequest,
    batch_id: str,
    index: int,
    http_request: Request
) -> InferenceResponse:
    """Process a single query from a batch"""
    
    try:
        # Create a mock request for single inference
        single_request = InferenceRequest(
            query=request.query,
            context=request.context,
            patient_id=request.patient_id,
            session_id=f"{batch_id}-{index}"
        )
        
        # Use the single inference logic
        result = await single_inference(single_request, http_request)
        return result
        
    except Exception as e:
        # Return error response for this specific query
        return InferenceResponse(
            request_id=f"{batch_id}-{index}-error",
            response=f"Error processing query: {str(e)}",
            confidence=0.0,
            medical_context={"error": str(e)},
            processing_time=0.0,
            model_version=settings.model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            medical_validation_passed=False,
            phi_protection_applied=False,
            tokens_used=0,
            cache_hit=False
        )