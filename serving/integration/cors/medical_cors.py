"""
Medical-grade CORS configuration with domain restrictions and security.
Provides secure cross-origin resource sharing for medical applications
with HIPAA compliance and security best practices.
"""

import re
import ipaddress
from typing import List, Dict, Set, Optional, Any
from datetime import datetime, timedelta
from urllib.parse import urlparse

from fastapi import (
    Request,
    Response,
    HTTPException,
    status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from ...config.settings import get_settings


# Configuration
settings = get_settings()
logger = structlog.get_logger("cors")


# Security Models
class MedicalDomainConfig(BaseModel):
    """Medical domain configuration for CORS."""
    domain: str
    protocol: str = "https"
    port: Optional[int] = None
    is_wildcard: bool = False
    is_ip: bool = False
    allowed_paths: List[str] = []
    blocked_paths: List[str] = []
    max_age_seconds: int = 86400  # 24 hours
    allow_credentials: bool = True
    allowed_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    allowed_headers: List[str] = [
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "X-Request-ID",
        "X-User-ID",
        "X-Session-ID"
    ]
    security_headers: Dict[str, str] = {}


class CORSSecurityPolicy(BaseModel):
    """CORS security policy configuration."""
    allowed_origins: List[MedicalDomainConfig]
    blocked_origins: List[str]
    allowed_ip_ranges: List[str]
    blocked_ip_ranges: List[str]
    max_request_size: int = 10485760  # 10MB
    rate_limit_per_origin: int = 1000  # requests per hour
    enable_strict_transport_security: bool = True
    enable_content_security_policy: bool = True
    enable_request_logging: bool = True
    medical_compliance_mode: bool = True


class MedicalCORSManager:
    """Medical-grade CORS manager with security and compliance."""
    
    def __init__(self):
        self.security_policy = self._initialize_security_policy()
        self.request_counts: Dict[str, List[datetime]] = {}
        self.blocked_origins: Set[str] = set()
        self.allowed_origins_cache: Set[str] = set()
        self.logger = structlog.get_logger("cors.manager")
    
    def _initialize_security_policy(self) -> CORSSecurityPolicy:
        """Initialize medical-grade CORS security policy."""
        
        # Production medical domain configurations
        production_domains = [
            MedicalDomainConfig(
                domain="app.medical-ai.health",
                protocol="https",
                port=443,
                allowed_paths=["/api/", "/ws/", "/dashboard/", "/patient/"],
                max_age_seconds=3600,
                security_headers={
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "DENY",
                    "X-XSS-Protection": "1; mode=block"
                }
            ),
            MedicalDomainConfig(
                domain="dashboard.medical-ai.health",
                protocol="https",
                port=443,
                allowed_paths=["/api/", "/nurse/", "/admin/", "/metrics/"],
                max_age_seconds=1800,  # Shorter for admin dashboard
                security_headers={
                    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    "X-Content-Type-Options": "nosniff",
                    "X-Frame-Options": "SAMEORIGIN"
                }
            ),
            MedicalDomainConfig(
                domain="*.medical-ai.health",
                protocol="https",
                is_wildcard=True,
                max_age_seconds=1800,
                allowed_methods=["GET", "POST"],
                allowed_headers=["Content-Type", "Authorization"]
            )
        ]
        
        # Development/staging domains
        dev_domains = [
            MedicalDomainConfig(
                domain="localhost",
                protocol="http",
                port=3000,
                allowed_paths=["/"],
                max_age_seconds=600,
                allow_credentials=True
            ),
            MedicalDomainConfig(
                domain="localhost",
                protocol="http",
                port=5173,  # Vite dev server
                allowed_paths=["/"],
                max_age_seconds=600,
                allow_credentials=True
            ),
            MedicalDomainConfig(
                domain="127.0.0.1",
                protocol="http",
                port=3000,
                allowed_paths=["/"],
                max_age_seconds=600,
                allow_credentials=True,
                is_ip=True
            )
        ]
        
        # Select domains based on environment
        if settings.environment == "production":
            allowed_origins = production_domains
        elif settings.environment == "staging":
            allowed_origins = production_domains + dev_domains
        else:  # development, testing
            allowed_origins = dev_domains + production_domains
        
        return CORSSecurityPolicy(
            allowed_origins=allowed_origins,
            blocked_origins=[
                "*.evil.com",
                "malicious-site.net",
                "phishing.healthcare.gov"
            ],
            allowed_ip_ranges=[
                "10.0.0.0/8",      # Private networks
                "172.16.0.0/12",   # Private networks
                "192.168.0.0/16",  # Private networks
                "127.0.0.0/8",     # Localhost
                "::1/128"          # IPv6 localhost
            ],
            blocked_ip_ranges=[
                "0.0.0.0/8",
                "169.254.0.0/16",  # Link-local
                "224.0.0.0/4",     # Multicast
                "240.0.0.0/4"      # Reserved
            ],
            max_request_size=settings.serving.max_request_size,
            rate_limit_per_origin=1000,
            enable_strict_transport_security=settings.environment == "production",
            enable_content_security_policy=True,
            enable_request_logging=True,
            medical_compliance_mode=True
        )
    
    def parse_origin(self, origin: str) -> Optional[MedicalDomainConfig]:
        """Parse and validate origin string."""
        if not origin:
            return None
        
        try:
            # Parse URL
            parsed = urlparse(origin)
            
            if not parsed.scheme or not parsed.netloc:
                return None
            
            domain = parsed.hostname
            protocol = parsed.scheme
            port = parsed.port
            
            if not domain:
                return None
            
            # Check if it's an IP address
            try:
                ipaddress.ip_address(domain)
                is_ip = True
            except ValueError:
                is_ip = False
            
            # Find matching domain config
            for domain_config in self.security_policy.allowed_origins:
                if self._domain_matches(domain, domain_config, is_ip):
                    # Check path restrictions
                    path = parsed.path or "/"
                    if self._path_allowed(path, domain_config):
                        return MedicalDomainConfig(
                            domain=domain,
                            protocol=protocol,
                            port=port,
                            is_wildcard=domain_config.is_wildcard,
                            is_ip=is_ip,
                            allowed_paths=domain_config.allowed_paths,
                            blocked_paths=domain_config.blocked_paths,
                            max_age_seconds=domain_config.max_age_seconds,
                            allow_credentials=domain_config.allow_credentials,
                            allowed_methods=domain_config.allowed_methods,
                            allowed_headers=domain_config.allowed_headers,
                            security_headers=domain_config.security_headers
                        )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to parse origin: {origin}, error: {e}")
            return None
    
    def _domain_matches(self, domain: str, config: MedicalDomainConfig, is_ip: bool) -> bool:
        """Check if domain matches configuration."""
        if config.is_wildcard and "*" in config.domain:
            # Handle wildcard domains
            wildcard_pattern = config.domain.replace("*.", "")
            return domain.endswith(wildcard_pattern)
        elif is_ip and config.is_ip:
            return domain == config.domain
        else:
            return domain == config.domain
    
    def _path_allowed(self, path: str, config: MedicalDomainConfig) -> bool:
        """Check if path is allowed for domain."""
        if config.blocked_paths:
            for blocked in config.blocked_paths:
                if path.startswith(blocked):
                    return False
        
        if config.allowed_paths:
            for allowed in config.allowed_paths:
                if path.startswith(allowed):
                    return True
            return False  # No match in allowed paths
        
        return True  # No restrictions
    
    def validate_origin(self, origin: str, request: Request) -> tuple[bool, Optional[str]]:
        """Validate origin and return validation result and reason."""
        if not origin:
            return False, "No origin provided"
        
        # Check if origin is in blocked list
        if origin.lower() in [blocked.lower() for blocked in self.security_policy.blocked_origins]:
            return False, "Origin is explicitly blocked"
        
        # Parse and validate domain
        domain_config = self.parse_origin(origin)
        if not domain_config:
            return False, "Origin not in allowed domain list"
        
        # Check if request is coming from expected source
        client_ip = self._get_client_ip(request)
        if not self._validate_source_ip(client_ip, origin):
            return False, "Request source validation failed"
        
        # Rate limiting per origin
        if not self._check_rate_limit(origin):
            return False, "Rate limit exceeded for origin"
        
        return True, None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers first (production setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        return request.client.host if request.client else "unknown"
    
    def _validate_source_ip(self, client_ip: str, origin: str) -> bool:
        """Validate that request is coming from expected source."""
        try:
            # Check if IP is in allowed ranges
            client_ip_obj = ipaddress.ip_address(client_ip)
            
            for allowed_range in self.security_policy.allowed_ip_ranges:
                if client_ip_obj in ipaddress.ip_network(allowed_range):
                    return True
            
            # Check if IP is in blocked ranges
            for blocked_range in self.security_policy.blocked_ip_ranges:
                if client_ip_obj in ipaddress.ip_network(blocked_range):
                    return False
            
            # For production, be more restrictive about IP sources
            if settings.environment == "production":
                # In production, we expect specific IP ranges or private networks
                return client_ip_obj.is_private or client_ip_obj.is_loopback
            
            return True  # Development/testing is more permissive
            
        except Exception as e:
            self.logger.error(f"Failed to validate source IP {client_ip}: {e}")
            return False
    
    def _check_rate_limit(self, origin: str) -> bool:
        """Check rate limit for origin."""
        now = datetime.utcnow()
        origin_requests = self.request_counts.get(origin, [])
        
        # Remove old requests (older than 1 hour)
        cutoff_time = now - timedelta(hours=1)
        self.request_counts[origin] = [
            req_time for req_time in origin_requests
            if req_time > cutoff_time
        ]
        
        # Check rate limit
        current_requests = len(self.request_counts[origin])
        if current_requests >= self.security_policy.rate_limit_per_origin:
            return False
        
        # Add current request
        self.request_counts[origin].append(now)
        return True
    
    def get_cors_headers(
        self,
        origin: str,
        request: Request,
        allow_credentials: bool = True
    ) -> Dict[str, str]:
        """Generate CORS headers based on origin and security policy."""
        domain_config = self.parse_origin(origin)
        
        if not domain_config:
            # Reject request - return minimal CORS headers
            return {
                "Access-Control-Allow-Origin": "",
                "Vary": "Origin",
                "Access-Control-Allow-Credentials": "false"
            }
        
        headers = {
            "Access-Control-Allow-Origin": origin,
            "Vary": "Origin",
            "Access-Control-Allow-Credentials": str(domain_config.allow_credentials).lower(),
            "Access-Control-Max-Age": str(domain_config.max_age_seconds),
            "Access-Control-Allow-Methods": ",".join(domain_config.allowed_methods),
            "Access-Control-Allow-Headers": ",".join(domain_config.allowed_headers)
        }
        
        # Add security headers if configured
        if domain_config.security_headers:
            headers.update(domain_config.security_headers)
        
        # Add medical compliance headers
        if self.security_policy.medical_compliance_mode:
            headers.update({
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Referrer-Policy": "strict-origin-when-cross-origin"
            })
        
        # Add HSTS in production
        if self.security_policy.enable_strict_transport_security and settings.environment == "production":
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Add Content Security Policy
        if self.security_policy.enable_content_security_policy:
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self' wss: https:; "
                "media-src 'self'; "
                "object-src 'none'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
            headers["Content-Security-Policy"] = csp
        
        return headers
    
    def log_cors_request(
        self,
        origin: str,
        request: Request,
        is_allowed: bool,
        reason: Optional[str] = None
    ):
        """Log CORS request for security monitoring."""
        if not self.security_policy.enable_request_logging:
            return
        
        client_ip = self._get_client_ip(request)
        
        self.logger.info(
            "CORS request processed",
            origin=origin,
            client_ip=client_ip,
            method=request.method,
            url=str(request.url.path),
            is_allowed=is_allowed,
            reason=reason,
            user_agent=request.headers.get("user-agent", ""),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def get_allowed_origins(self) -> List[str]:
        """Get list of currently allowed origins."""
        return [config.domain for config in self.security_policy.allowed_origins]
    
    def add_temporary_origin(
        self,
        origin: str,
        duration_seconds: int = 3600,
        allowed_paths: List[str] = None
    ):
        """Add temporary origin to allowed list (for emergency access)."""
        temp_config = MedicalDomainConfig(
            domain=origin,
            allowed_paths=allowed_paths or ["/"],
            max_age_seconds=duration_seconds
        )
        
        self.security_policy.allowed_origins.append(temp_config)
        self.allowed_origins_cache.add(origin)
        
        # Schedule removal
        def remove_temp_origin():
            import time
            time.sleep(duration_seconds)
            if origin in self.allowed_origins_cache:
                self.allowed_origins_cache.remove(origin)
                if temp_config in self.security_policy.allowed_origins:
                    self.security_policy.allowed_origins.remove(temp_config)
        
        import threading
        thread = threading.Thread(target=remove_temp_origin, daemon=True)
        thread.start()
        
        self.logger.info("Temporary origin added", origin=origin, duration_seconds=duration_seconds)


# Global CORS manager instance
cors_manager = MedicalCORSManager()


class MedicalCORSMiddleware:
    """FastAPI middleware for medical-grade CORS handling."""
    
    def __init__(self):
        self.logger = structlog.get_logger("cors.middleware")
    
    async def __call__(self, request: Request, call_next):
        """Process request with medical-grade CORS handling."""
        
        # Handle preflight requests
        if request.method == "OPTIONS":
            return await self._handle_preflight(request)
        
        # Process actual request
        response = await call_next(request)
        
        # Add CORS headers to response
        origin = request.headers.get("origin")
        if origin:
            is_allowed, reason = cors_manager.validate_origin(origin, request)
            
            # Add CORS headers
            cors_headers = cors_manager.get_cors_headers(origin, request)
            for header, value in cors_headers.items():
                response.headers[header] = value
            
            # Log request
            cors_manager.log_cors_request(origin, request, is_allowed, reason)
            
            # Reject if origin not allowed
            if not is_allowed:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "CORS policy violation",
                        "detail": reason or "Origin not allowed",
                        "allowed_origins": cors_manager.get_allowed_origins()
                    },
                    headers=cors_headers
                )
        
        return response
    
    async def _handle_preflight(self, request: Request) -> Response:
        """Handle OPTIONS preflight request."""
        origin = request.headers.get("origin")
        
        if not origin:
            return JSONResponse(
                status_code=400,
                content={"error": "Origin header required for preflight"}
            )
        
        is_allowed, reason = cors_manager.validate_origin(origin, request)
        
        cors_headers = cors_manager.get_cors_headers(origin, request)
        
        # Log preflight request
        cors_manager.log_cors_request(origin, request, is_allowed, reason)
        
        if not is_allowed:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "CORS preflight failed",
                    "detail": reason or "Origin not allowed"
                },
                headers=cors_headers
            )
        
        # Return successful preflight response
        return Response(
            status_code=200,
            headers=cors_headers
        )


# FastAPI middleware factory
def create_medical_cors_middleware() -> MedicalCORSMiddleware:
    """Create medical-grade CORS middleware instance."""
    return MedicalCORSMiddleware()


# Utility functions
def get_cors_configuration() -> Dict[str, Any]:
    """Get current CORS configuration for documentation."""
    return {
        "allowed_origins": cors_manager.get_allowed_origins(),
        "security_policy": {
            "medical_compliance_mode": cors_manager.security_policy.medical_compliance_mode,
            "max_request_size": cors_manager.security_policy.max_request_size,
            "rate_limit_per_origin": cors_manager.security_policy.rate_limit_per_origin,
            "enable_strict_transport_security": cors_manager.security_policy.enable_strict_transport_security,
            "enable_content_security_policy": cors_manager.security_policy.enable_content_security_policy
        },
        "environment": settings.environment
    }


def validate_medical_origin(origin: str) -> Dict[str, Any]:
    """Validate origin for medical compliance."""
    if not origin:
        return {
            "valid": False,
            "reason": "No origin provided",
            "compliance_score": 0.0
        }
    
    domain_config = cors_manager.parse_origin(origin)
    
    if not domain_config:
        return {
            "valid": False,
            "reason": "Origin not in allowed medical domains",
            "compliance_score": 0.0,
            "allowed_origins": cors_manager.get_allowed_origins()
        }
    
    # Calculate compliance score
    score = 0.0
    
    # HTTPS bonus
    if domain_config.protocol == "https":
        score += 30
    
    # Medical domain bonus
    if "medical" in domain_config.domain.lower():
        score += 25
    
    # Port security (443 for HTTPS, standard for HTTP)
    if (domain_config.protocol == "https" and domain_config.port == 443) or \
       (domain_config.protocol == "http" and domain_config.port == 80):
        score += 15
    
    # Credentials allowed (if appropriate)
    if domain_config.allow_credentials:
        score += 10
    
    # Security headers present
    if domain_config.security_headers:
        score += 20
    
    return {
        "valid": True,
        "compliance_score": min(score, 100.0),
        "domain_config": domain_config.dict(),
        "recommendations": [
            "Use HTTPS for production",
            "Implement proper security headers",
            "Restrict to medical domains",
            "Enable credential validation"
        ]
    }


# Export functions and classes
__all__ = [
    "cors_manager",
    "MedicalCORSMiddleware",
    "MedicalDomainConfig",
    "CORSSecurityPolicy",
    "create_medical_cors_middleware",
    "get_cors_configuration",
    "validate_medical_origin"
]