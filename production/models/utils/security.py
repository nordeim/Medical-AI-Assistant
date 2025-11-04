"""
Security Utility
API key validation, authentication, and access control for medical AI serving infrastructure.
"""

import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import jwt
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

class SecurityManager:
    """Security management for medical AI production serving"""
    
    def __init__(self, config_path: str = "config/security_config.yaml"):
        self.config = self._load_config(config_path)
        
        # API key storage (in production, use secure key management service)
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        
        # JWT settings
        self.jwt_secret = self.config.get("jwt_secret", "your-secret-key")
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = timedelta(hours=24)
        
        # Rate limiting
        self.default_rate_limit = self.config.get("default_rate_limit", 100)  # requests per minute
        self.rate_limit_window = timedelta(minutes=1)
        
        # Load configuration
        self._load_security_config()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load security configuration"""
        default_config = {
            "jwt_secret": "production-secret-key-change-in-deployment",
            "jwt_expiration_hours": 24,
            "default_rate_limit": 100,
            "enable_rate_limiting": True,
            "enable_api_key_validation": True,
            "allowed_origins": ["*"],
            "token_refresh_threshold": 3600,  # 1 hour
            "audit_logging": True
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Security config {config_path} not found, using defaults")
            return default_config
    
    def _load_security_config(self):
        """Load and initialize security configuration"""
        # Initialize with demo API keys
        demo_keys = {
            "demo_client_001": {
                "name": "Medical Center A",
                "permissions": ["read", "predict", "batch_predict"],
                "rate_limit": 500,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            },
            "demo_client_002": {
                "name": "Research Institute B",
                "permissions": ["read", "predict"],
                "rate_limit": 200,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            },
            "demo_admin_001": {
                "name": "System Administrator",
                "permissions": ["read", "predict", "admin"],
                "rate_limit": 1000,
                "created_at": datetime.utcnow().isoformat(),
                "active": True
            }
        }
        
        # Generate secure API keys
        for key_id, config in demo_keys.items():
            api_key = self._generate_api_key(key_id)
            config["api_key"] = api_key
            self.api_keys[api_key] = config
        
        logger.info(f"Loaded {len(self.api_keys)} API keys")
    
    def _generate_api_key(self, key_id: str) -> str:
        """Generate a secure API key"""
        # Create key components
        timestamp = str(int(time.time()))
        random_component = secrets.token_urlsafe(32)
        
        # Combine and hash
        key_data = f"{key_id}:{timestamp}:{random_component}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Create final key
        api_key = f"med_ai_{key_hash[:32]}"
        
        return api_key
    
    def validate_api_key(self, api_key: Optional[str]) -> Dict[str, Any]:
        """Validate API key and return client info"""
        if not api_key:
            if self.config.get("enable_api_key_validation", True):
                raise SecurityError("API key required", "MISSING_API_KEY")
            return {"anonymous": True, "permissions": ["read"]}
        
        if api_key not in self.api_keys:
            raise SecurityError("Invalid API key", "INVALID_API_KEY")
        
        client_info = self.api_keys[api_key]
        
        if not client_info.get("active", False):
            raise SecurityError("API key is inactive", "INACTIVE_API_KEY")
        
        # Update rate limit tracking
        self._update_rate_limit_tracking(api_key)
        
        return client_info
    
    def _update_rate_limit_tracking(self, api_key: str):
        """Update rate limit tracking for an API key"""
        current_time = datetime.utcnow()
        
        if api_key not in self.rate_limits:
            self.rate_limits[api_key] = {
                "requests": [],
                "current_count": 0,
                "window_start": current_time
            }
        
        limit_info = self.rate_limits[api_key]
        window_start = limit_info["window_start"]
        
        # Reset window if expired
        if current_time - window_start > self.rate_limit_window:
            limit_info["requests"] = []
            limit_info["current_count"] = 0
            limit_info["window_start"] = current_time
        
        # Add current request
        limit_info["requests"].append(current_time)
        limit_info["current_count"] += 1
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Check if API key is within rate limits"""
        if not self.config.get("enable_rate_limiting", True):
            return True
        
        if api_key not in self.rate_limits:
            return True
        
        limit_info = self.rate_limits[api_key]
        client_config = self.api_keys.get(api_key, {})
        
        # Get rate limit for this client
        rate_limit = client_config.get("rate_limit", self.default_rate_limit)
        
        # Check if within limit
        current_count = limit_info["current_count"]
        
        if current_count > rate_limit:
            logger.warning(f"Rate limit exceeded for {api_key}: {current_count}/{rate_limit}")
            return False
        
        return True
    
    def generate_jwt_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "issued_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + self.jwt_expiration).isoformat()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        return token
    
    def validate_jwt_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check expiration
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.utcnow() > expires_at:
                raise SecurityError("Token expired", "TOKEN_EXPIRED")
            
            return payload
            
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {str(e)}", "INVALID_TOKEN")
    
    def has_permission(self, client_info: Dict[str, Any], required_permission: str) -> bool:
        """Check if client has required permission"""
        permissions = client_info.get("permissions", [])
        return required_permission in permissions
    
    def get_client_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Get client information by API key"""
        return self.api_keys.get(api_key)
    
    def get_rate_limit_status(self, api_key: str) -> Dict[str, Any]:
        """Get current rate limit status for API key"""
        if api_key not in self.rate_limits:
            return {"requests": 0, "limit": "unknown", "remaining": "unknown"}
        
        limit_info = self.rate_limits[api_key]
        client_config = self.api_keys.get(api_key, {})
        rate_limit = client_config.get("rate_limit", self.default_rate_limit)
        
        current_count = limit_info["current_count"]
        remaining = max(0, rate_limit - current_count)
        
        return {
            "requests": current_count,
            "limit": rate_limit,
            "remaining": remaining,
            "window_start": limit_info["window_start"].isoformat()
        }
    
    def create_audit_log(self, api_key: str, action: str, resource: str, 
                        result: str, details: Optional[Dict[str, Any]] = None):
        """Create audit log entry"""
        if not self.config.get("audit_logging", True):
            return
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "api_key": api_key[:8] + "...",  # Partial key for security
            "action": action,
            "resource": resource,
            "result": result,
            "details": details or {}
        }
        
        # In production, this would write to a secure audit log service
        logger.info(f"AUDIT: {log_entry}")
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key]["active"] = False
            self.create_audit_log(api_key, "revoke", "api_key", "success")
            logger.info(f"API key revoked: {api_key[:8]}...")
            return True
        return False
    
    def create_api_key(self, name: str, permissions: List[str], 
                      rate_limit: Optional[int] = None) -> str:
        """Create a new API key"""
        key_id = f"key_{secrets.token_hex(8)}"
        rate_limit = rate_limit or self.default_rate_limit
        
        api_key = self._generate_api_key(key_id)
        
        self.api_keys[api_key] = {
            "name": name,
            "permissions": permissions,
            "rate_limit": rate_limit,
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        self.create_audit_log(api_key, "create", "api_key", "success", {
            "name": name,
            "permissions": permissions
        })
        
        return api_key

# Security decorator for API endpoints
def require_api_key(required_permission: Optional[str] = None):
    """Decorator to require API key authentication"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get API key from headers (this would be passed from FastAPI)
            api_key = kwargs.get('x_api_key')
            
            # Initialize security manager
            security = SecurityManager()
            
            try:
                # Validate API key
                client_info = security.validate_api_key(api_key)
                
                # Check required permission
                if required_permission:
                    if not security.has_permission(client_info, required_permission):
                        security.create_audit_log(
                            api_key or "anonymous", 
                            "deny", 
                            f"permission:{required_permission}", 
                            "denied"
                        )
                        raise SecurityError("Insufficient permissions", "INSUFFICIENT_PERMISSIONS")
                
                # Check rate limit
                if api_key and not security.check_rate_limit(api_key):
                    security.create_audit_log(
                        api_key, "rate_limit", "request", "denied"
                    )
                    raise SecurityError("Rate limit exceeded", "RATE_LIMIT_EXCEEDED")
                
                # Audit successful access
                if api_key:
                    security.create_audit_log(
                        api_key, "access", "api_endpoint", "success",
                        {"endpoint": func.__name__}
                    )
                
                # Add client info to kwargs for use in endpoint
                kwargs['client_info'] = client_info
                return await func(*args, **kwargs)
                
            except SecurityError:
                raise
            except Exception as e:
                logger.error(f"Security validation error: {str(e)}")
                raise SecurityError("Security validation failed", "SECURITY_ERROR")
        
        return wrapper
    return decorator

class SecurityError(Exception):
    """Security-related exceptions"""
    
    def __init__(self, message: str, error_code: str):
        self.message = message
        self.error_code = error_code
        super().__init__(message)

# Utility functions
def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(length)

def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """Hash sensitive data with optional salt"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    combined = f"{data}:{salt}"
    return hashlib.sha256(combined.encode()).hexdigest()

def validate_input_safety(input_data: Any) -> bool:
    """Basic input safety validation"""
    if isinstance(input_data, str):
        # Check for potentially malicious patterns
        dangerous_patterns = [
            "<script", "javascript:", "vbscript:",
            "onload=", "onerror=", "eval(",
            "system(", "exec(", "shell_"
        ]
        
        input_lower = input_data.lower()
        return not any(pattern in input_lower for pattern in dangerous_patterns)
    
    return True

# Global security manager instance
security_manager = SecurityManager()

# Validation function for FastAPI integration
def validate_api_key(api_key: Optional[str] = None):
    """Validate API key (for FastAPI dependency injection)"""
    return security_manager.validate_api_key(api_key)

# Example usage and testing
if __name__ == "__main__":
    # Initialize security manager
    security = SecurityManager()
    
    # Test API key validation
    for api_key, info in list(security.api_keys.items())[:2]:
        print(f"API Key: {api_key[:20]}...")
        print(f"Client: {info['name']}")
        print(f"Permissions: {info['permissions']}")
        
        try:
            validated_info = security.validate_api_key(api_key)
            print(f"Validation: Success")
            
            # Test rate limiting
            for _ in range(3):
                security.check_rate_limit(api_key)
            rate_status = security.get_rate_limit_status(api_key)
            print(f"Rate Limit Status: {rate_status}")
            
        except Exception as e:
            print(f"Validation: Failed - {e}")
        
        print("-" * 50)
    
    # Test JWT token generation
    test_token = security.generate_jwt_token("test_user", ["read", "predict"])
    print(f"JWT Token: {test_token[:50]}...")
    
    try:
        token_payload = security.validate_jwt_token(test_token)
        print(f"Token Payload: {token_payload}")
    except Exception as e:
        print(f"Token Validation: Failed - {e}")