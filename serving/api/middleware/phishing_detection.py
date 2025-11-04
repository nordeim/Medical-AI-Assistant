"""
PHI Protection Middleware
Comprehensive PHI detection, validation, and protection for HIPAA compliance
"""

import re
import json
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from ..utils.exceptions import PHIProtectionError
from ..utils.security import SecurityValidator
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# HIPAA Safe Harbor PHI identifiers
HIPAA_PHI_IDENTIFIERS = {
    "names": [
        r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # Full names
        r"\b[A-Z]\. [A-Z][a-z]+\b"       # First initial + last name
    ],
    "geographic_subdivisions": [
        r"\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way)\b",
        r"\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}\b"  # City, State ZIP
    ],
    "dates": [
        r"\b(19|20)\d{2}[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b",  # Birth dates
        r"\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])[-/](19|20)\d{2}\b"   # MM/DD/YYYY
    ],
    "telephone_numbers": [
        r"\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b"
    ],
    "fax_numbers": [
        r"\bfax[:\s]*(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
    ],
    "email_addresses": [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ],
    "social_security_numbers": [
        r"\b\d{3}-?\d{2}-?\d{4}\b"
    ],
    "medical_record_numbers": [
        r"\b(MRN|mrn|medical record|patient id|chart)\s*[:#]?\s*[A-Z0-9]{6,12}\b"
    ],
    "health_plan_beneficiary_numbers": [
        r"\b(insurance|policy|member|group)\s*(id|number|#)\s*[A-Z0-9]{6,15}\b"
    ],
    "account_numbers": [
        r"\b(account|acct)\s*(id|number|#)\s*[A-Z0-9]{6,15}\b"
    ],
    "certificate_license_numbers": [
        r"\b(license|lic|dr|npi|dea)\s*(id|number|#)\s*[A-Z0-9]{6,15}\b"
    ],
    "vehicle_identifiers": [
        r"\b(vin|vehicle|license plate)\s*[:#]?\s*[A-HJ-NPR-Z0-9]{11,17}\b"
    ],
    "device_identifiers": [
        r"\b(device|implant|serial|id)\s*[:#]?\s*[A-Z0-9]{6,20}\b"
    ],
    "web_urls": [
        r"\bhttps?://[^\s]+\b"
    ],
    "biometric_identifiers": [
        r"\b(biometric|fingerprint|retina|iris|voiceprint|handprint)\s*[:#]?\s*[A-Z0-9]{6,20}\b"
    ],
    "photos": [
        r"\b(photo|image|picture|face)\s*[:#]?\s*[A-Za-z0-9_-]+\.(jpg|jpeg|png|gif)\b"
    ],
    "any_unique_identifying_number": [
        r"\b(id|identifier|number|#)\s*[:#]?\s*[A-Z0-9]{6,20}\b"
    ]
}

# Medical record patterns specific to healthcare
MEDICAL_RECORD_PATTERNS = {
    "patient_id": r"\b(patient|pt)\s*(id|number|record)\s*[:#]?\s*[A-Z0-9]{6,12}\b",
    "doctor_id": r"\b(doctor|dr|physician)\s*(id|number|license)\s*[:#]?\s*[A-Z0-9]{6,15}\b",
    "hospital_id": r"\b(hospital|hosp|facility)\s*(id|number|code)\s*[:#]?\s*[A-Z0-9]{6,15}\b",
    "appointment_id": r"\b(appt|appointment|visit)\s*(id|number)\s*[:#]?\s*[A-Z0-9]{6,15}\b"
}


class PHIProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive PHI protection"""
    
    def __init__(self, app, call_next):
        super().__init__(app)
        self.call_next = call_next
        self.logger = get_logger("phi_protection")
        self.session_phi_data = {}  # Track PHI data per session
    
    async def dispatch(self, request: Request, call_next):
        """Process request through PHI protection pipeline"""
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        session_id = self._get_session_id(request)
        
        try:
            # Extract request data
            request_data = await self._extract_request_data(request)
            
            # PHI Detection and Analysis
            phi_analysis = self._analyze_phi_content(request_data)
            
            # Store PHI metadata for session
            if phi_analysis["phi_detected"]:
                self._update_session_phi_data(session_id, phi_analysis)
                
                self.logger.log_phi_access(
                    operation="phi_detection",
                    phi_type="multiple",
                    success=True,
                    user_id=self._get_user_id(request)
                )
            
            # PHI Protection Actions
            protected_data = await self._apply_phi_protection(request_data, phi_analysis)
            
            # Add PHI metadata to request state
            request.state.phi_analysis = phi_analysis
            request.state.protected_data = protected_data
            
            # Process response
            response = await call_next(request)
            
            # PHI Protection for Response
            protected_response = await self._protect_response(response, phi_analysis)
            
            # Log successful PHI protection
            processing_time = time.time() - start_time
            self.logger.log_phi_access(
                operation="phi_protection",
                phi_type="response_protection",
                success=True,
                user_id=self._get_user_id(request)
            )
            
            return protected_response
            
        except PHIProtectionError:
            raise
        except Exception as e:
            # Log PHI protection error
            processing_time = time.time() - start_time
            self.logger.log_security_event(
                event_type="phi_protection_error",
                severity="high",
                description=f"PHI protection failed: {str(e)}",
                user_id=self._get_user_id(request),
                ip_address=client_ip
            )
            raise PHIProtectionError(detail=f"PHI protection failed: {str(e)}")
        
        finally:
            request.state.processing_time = time.time() - start_time
    
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
            raise PHIProtectionError(detail="Invalid JSON in request body")
        except Exception as e:
            raise PHIProtectionError(detail=f"Failed to extract request data: {str(e)}")
    
    def _analyze_phi_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive PHI content analysis"""
        
        analysis_result = {
            "phi_detected": False,
            "phi_types": set(),
            "phi_instances": [],
            "hipaa_violations": [],
            "risk_level": "low",
            "requires_redaction": False
        }
        
        # Convert data to string for analysis
        content_str = str(data)
        
        # Check HIPAA PHI identifiers
        for phi_type, patterns in HIPAA_PHI_IDENTIFIERS.items():
            for pattern in patterns:
                matches = re.findall(pattern, content_str, re.IGNORECASE)
                if matches:
                    analysis_result["phi_detected"] = True
                    analysis_result["phi_types"].add(phi_type)
                    
                    for match in matches:
                        analysis_result["phi_instances"].append({
                            "type": phi_type,
                            "value": match,
                            "pattern": pattern,
                            "position": content_str.find(str(match)) if isinstance(match, str) else None
                        })
        
        # Check medical record patterns
        for record_type, pattern in MEDICAL_RECORD_PATTERNS.items():
            matches = re.findall(pattern, content_str, re.IGNORECASE)
            if matches:
                analysis_result["phi_detected"] = True
                analysis_result["phi_types"].add("medical_records")
                
                for match in matches:
                    analysis_result["phi_instances"].append({
                        "type": "medical_records",
                        "subtype": record_type,
                        "value": match,
                        "pattern": pattern
                    })
        
        # Determine risk level
        high_risk_types = {"social_security_numbers", "biometric_identifiers", "photos"}
        medium_risk_types = {"names", "dates", "telephone_numbers", "email_addresses"}
        
        if analysis_result["phi_types"] & high_risk_types:
            analysis_result["risk_level"] = "high"
            analysis_result["requires_redaction"] = True
        elif analysis_result["phi_types"] & medium_risk_types:
            analysis_result["risk_level"] = "medium"
            analysis_result["requires_redaction"] = True
        
        # Check for HIPAA violations (explicit consent, retention periods, etc.)
        hipaa_violations = self._check_hipaa_compliance(data)
        analysis_result["hipaa_violations"] = hipaa_violations
        
        if hipaa_violations:
            analysis_result["risk_level"] = "critical"
            analysis_result["requires_redaction"] = True
        
        return analysis_result
    
    def _check_hipaa_compliance(self, data: Dict[str, Any]) -> List[str]:
        """Check for HIPAA compliance violations"""
        
        violations = []
        content_str = str(data).lower()
        
        # Check for improper data retention periods
        if "retention" in content_str:
            retention_match = re.search(r"retention[:\s]*(\d+)\s*(years?|months?)", content_str)
            if retention_match:
                retention_period = int(retention_match.group(1))
                unit = retention_match.group(2)
                
                # HIPAA requires minimum 6 years for records
                if "year" in unit and retention_period < 6:
                    violations.append("Insufficient data retention period (minimum 6 years required)")
        
        # Check for improper sharing disclosure
        sharing_keywords = ["share with", "disclose to", "provide to", "send to"]
        for keyword in sharing_keywords:
            if keyword in content_str:
                # Check if proper authorization is mentioned
                if "authorization" not in content_str and "consent" not in content_str:
                    violations.append(f"Potential unauthorized disclosure: {keyword}")
        
        # Check for business associate requirements
        if "third party" in content_str or "vendor" in content_str:
            if "business associate agreement" not in content_str:
                violations.append("Third party involvement without business associate agreement")
        
        return violations
    
    async def _apply_phi_protection(self, data: Dict[str, Any], phi_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PHI protection measures"""
        
        if not phi_analysis["phi_detected"] or not settings.phi_redaction:
            return data
        
        protected_data = self._deep_copy_dict(data)
        protection_applied = False
        
        # Apply different protection modes based on PHI types
        for phi_instance in phi_analysis["phi_instances"]:
            phi_type = phi_instance["type"]
            value = phi_instance["value"]
            
            # Determine protection mode
            protection_mode = self._get_protection_mode(phi_type, phi_analysis["risk_level"])
            
            # Apply protection
            protected_value = SecurityValidator.redact_phi(str(value), mode=protection_mode)
            
            # Replace in protected data
            protected_data = self._replace_in_data(protected_data, str(value), protected_value)
            protection_applied = True
        
        if protection_applied:
            self.logger.log_phi_access(
                operation="phi_redaction",
                phi_type="response_redaction",
                success=True,
                user_id=self._get_user_id(None)
            )
        
        return protected_data
    
    async def _protect_response(self, response: Response, phi_analysis: Dict[str, Any]) -> Response:
        """Protect response data from PHI leakage"""
        
        # Add PHI protection headers
        if hasattr(response, 'headers'):
            response.headers["X-PHI-Protected"] = "true"
            response.headers["X-Risk-Level"] = phi_analysis["risk_level"]
            
            if phi_analysis["phi_detected"]:
                response.headers["X-PHI-Types"] = ",".join(phi_analysis["phi_types"])
        
        return response
    
    def _get_protection_mode(self, phi_type: str, risk_level: str) -> str:
        """Determine appropriate protection mode based on PHI type and risk"""
        
        # High-risk PHI types require masking
        high_risk_types = {"social_security_numbers", "biometric_identifiers", "photos"}
        
        if risk_level == "critical" or phi_type in high_risk_types:
            return "mask"
        elif phi_type in {"names", "dates"}:
            return "anonymize"
        else:
            return "hash"
    
    def _deep_copy_dict(self, data: Any) -> Any:
        """Create deep copy of data structure"""
        if isinstance(data, dict):
            return {k: self._deep_copy_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy_dict(item) for item in data]
        else:
            return data
    
    def _replace_in_data(self, data: Any, old_value: str, new_value: str) -> Any:
        """Replace value recursively in data structure"""
        if isinstance(data, dict):
            return {k: self._replace_in_data(v, old_value, new_value) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_in_data(item, old_value, new_value) for item in data]
        elif isinstance(data, str) and old_value in data:
            return data.replace(old_value, new_value)
        else:
            return data
    
    def _update_session_phi_data(self, session_id: str, phi_analysis: Dict[str, Any]):
        """Update session PHI data tracking"""
        
        if session_id not in self.session_phi_data:
            self.session_phi_data[session_id] = {
                "phi_encounters": 0,
                "phi_types_seen": set(),
                "risk_levels": [],
                "last_phi_access": datetime.now(timezone.utc)
            }
        
        session_data = self.session_phi_data[session_id]
        session_data["phi_encounters"] += 1
        session_data["phi_types_seen"].update(phi_analysis["phi_types"])
        session_data["risk_levels"].append(phi_analysis["risk_level"])
        session_data["last_phi_access"] = datetime.now(timezone.utc)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        return request.client.host if request.client else "unknown"
    
    def _get_session_id(self, request: Request) -> str:
        """Extract or generate session ID"""
        return request.headers.get("x-session-id", "anonymous")
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        if request:
            return request.headers.get("x-user-id")
        return None