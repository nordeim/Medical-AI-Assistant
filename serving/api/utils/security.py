"""
Security Utilities for Medical AI Inference API
Enterprise-grade security validation and authentication
"""

import re
import hashlib
import hmac
import secrets
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import jwt as pyjwt

from .exceptions import AuthenticationError, AuthorizationError, SecurityError
from .logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# Security context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# PHI patterns for detection
PHI_PATTERNS = {
    "ssn": r'\b\d{3}-?\d{2}-?\d{4}\b',
    "phone": r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    "mrn": r'\b(MRN|mrn)\s*[:#]?\s*[A-Z0-9]{6,12}\b',
    "name": r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',
    "address": r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way)\b'
}

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
}

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|EXECUTE)\b)",
    r"(\b(UNION|OR|AND)\b\s+['\"]?\d+['\"]?\s*[=<>])",
    r"(['\";].*(--|\bOR\b|\bAND\b).*['\";])",
    r"(\bOR\b.*=.*['\"]?\d+['\"]?.*OR\b)",
    r"(\bUNION\b.*\bSELECT\b)",
    r"(['\"].*\bDROP\b.*['\"].*\bTABLE\b)",
    r"(\bEXEC\b\s*\(\s*['\"].*['\"]\s*\))"
]

# XSS patterns
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
    r"<form[^>]*action[^>]*>.*?</form>"
]


class SecurityValidator:
    """Comprehensive security validation"""
    
    @staticmethod
    def validate_input(text: str) -> Dict[str, Any]:
        """Validate input for security threats"""
        
        validation_result = {
            "is_safe": True,
            "threats": [],
            "phi_detected": [],
            "sanitized": text
        }
        
        if not text:
            return validation_result
        
        # Check for SQL injection
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                validation_result["threats"].append("sql_injection")
                validation_result["is_safe"] = False
        
        # Check for XSS
        for pattern in XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                validation_result["threats"].append("xss")
                validation_result["is_safe"] = False
        
        # Check for PHI
        for phi_type, pattern in PHI_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                validation_result["phi_detected"].append({
                    "type": phi_type,
                    "matches": matches
                })
        
        # Sanitize input if threats detected
        if validation_result["threats"]:
            validation_result["sanitized"] = SecurityValidator._sanitize_input(text)
        
        return validation_result
    
    @staticmethod
    def _sanitize_input(text: str) -> str:
        """Sanitize potentially malicious input"""
        
        # Remove script tags and javascript: protocols
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)
        
        # Remove event handlers
        text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)
        
        # Remove iframe, object, embed tags
        for tag in ["iframe", "object", "embed"]:
            text = re.sub(f"<{tag}[^>]*>.*?</{tag}>", "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Escape HTML characters
        html_escape_map = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#x27;"
        }
        
        for char, escape in html_escape_map.items():
            text = text.replace(char, escape)
        
        return text.strip()
    
    @staticmethod
    def detect_phi(text: str) -> List[Dict[str, Any]]:
        """Detect PHI in text"""
        
        phi_instances = []
        
        for phi_type, pattern in PHI_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                phi_instances.append({
                    "type": phi_type,
                    "value": match,
                    "position": text.find(match) if isinstance(match, str) else None
                })
        
        return phi_instances
    
    @staticmethod
    def redact_phi(text: str, mode: str = "mask") -> str:
        """Redact PHI from text"""
        
        redacted_text = text
        
        for phi_type, pattern in PHI_PATTERNS.items():
            if mode == "mask":
                # Replace with asterisks, preserving length
                redacted_text = re.sub(pattern, lambda m: "*" * len(m.group()), redacted_text, flags=re.IGNORECASE)
            elif mode == "anonymize":
                # Replace with generic placeholders
                redacted_text = re.sub(pattern, f"[{phi_type.upper()}_REDACTED]", redacted_text, flags=re.IGNORECASE)
            elif mode == "hash":
                # Replace with hash
                redacted_text = re.sub(pattern, lambda m: hashlib.md5(m.group().encode()).hexdigest()[:8], redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    @staticmethod
    def validate_security_headers(headers: Dict[str, str]) -> Dict[str, Any]:
        """Validate security headers"""
        
        missing_headers = []
        for header, expected_value in SECURITY_HEADERS.items():
            if header not in headers:
                missing_headers.append(header)
        
        return {
            "valid": len(missing_headers) == 0,
            "missing_headers": missing_headers,
            "required_headers": list(SECURITY_HEADERS.keys())
        }
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_jwt_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT token"""
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode = data.copy()
        to_encode.update({"exp": expire})
        
        return jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    
    @staticmethod
    def verify_jwt_token(token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")


class TokenValidator:
    """Token validation and management"""
    
    @staticmethod
    def validate_token(credentials: str) -> Dict[str, Any]:
        """Validate bearer token"""
        
        if not credentials:
            raise AuthenticationError("No credentials provided")
        
        try:
            # Remove "Bearer " prefix if present
            if credentials.startswith("Bearer "):
                credentials = credentials[7:]
            
            # Verify JWT token
            payload = SecurityValidator.verify_jwt_token(credentials)
            
            # Check token expiration
            if "exp" in payload:
                if datetime.fromtimestamp(payload["exp"]) < datetime.utcnow():
                    raise AuthenticationError("Token expired")
            
            return {
                "valid": True,
                "user_id": payload.get("user_id"),
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", [])
            }
            
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
    
    @staticmethod
    def check_permissions(user_info: Dict[str, Any], required_permissions: List[str]) -> bool:
        """Check if user has required permissions"""
        
        user_permissions = set(user_info.get("permissions", []))
        required_permissions = set(required_permissions)
        
        return required_permissions.issubset(user_permissions)
    
    @staticmethod
    def check_role(user_info: Dict[str, Any], required_roles: List[str]) -> bool:
        """Check if user has required role"""
        
        user_roles = set(user_info.get("roles", []))
        required_roles = set(required_roles)
        
        return len(user_roles.intersection(required_roles)) > 0


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self):
        self.attempts = {}
    
    def is_rate_limited(self, identifier: str, limit: int, window: int) -> bool:
        """Check if identifier is rate limited"""
        
        now = datetime.now()
        key = f"{identifier}:{now.hour if window == 3600 else now.minute}"
        
        if key not in self.attempts:
            self.attempts[key] = []
        
        # Clean old attempts
        cutoff = now.timestamp() - window
        self.attempts[key] = [ts for ts in self.attempts[key] if ts > cutoff]
        
        # Check limit
        if len(self.attempts[key]) >= limit:
            return True
        
        # Add current attempt
        self.attempts[key].append(now.timestamp())
        return False


# Global rate limiter instance
rate_limiter = RateLimiter()