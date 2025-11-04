# Production API Security Configuration
# OAuth2, JWT, and Rate Limiting Security Implementation

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import secrets
import hashlib
import hmac
import jwt
from cryptography.fernet import Fernet
import aiohttp
import aiofiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    BASIC_AUTH = "basic_auth"
    CLIENT_CERTIFICATE = "client_certificate"

class Permission(Enum):
    """API permissions"""
    # Patient data permissions
    PATIENT_READ = "patient:read"
    PATIENT_WRITE = "patient:write"
    PATIENT_DELETE = "patient:delete"
    
    # Clinical data permissions
    OBSERVATION_READ = "observation:read"
    OBSERVATION_WRITE = "observation:write"
    CONDITION_READ = "condition:read"
    CONDITION_WRITE = "condition:write"
    
    # FHIR permissions
    FHIR_READ = "fhir:read"
    FHIR_WRITE = "fhir:write"
    FHIR_ADMIN = "fhir:admin"
    
    # System permissions
    ANALYTICS_READ = "analytics:read"
    WEBHOOK_MANAGE = "webhook:manage"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"

class RateLimitType(Enum):
    """Rate limiting types"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"

@dataclass
class APIKey:
    """API key configuration"""
    key_id: str
    user_id: str
    name: str
    permissions: List[Permission]
    rate_limit: int  # requests per minute
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    status: str = "active"
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        if self.last_used:
            data['last_used'] = self.last_used.isoformat()
        data['permissions'] = [p.value for p in self.permissions]
        return data

@dataclass
class OAuthClient:
    """OAuth2 client configuration"""
    client_id: str
    client_secret_hash: str
    client_name: str
    redirect_uris: List[str]
    scopes: List[Permission]
    grant_types: List[str]
    response_types: List[str]
    created_at: datetime
    status: str = "active"

@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    rule_type: RateLimitType
    limit: int  # requests allowed
    window: int  # time window in seconds
    burst_limit: int  # burst capacity
    user_identifiers: List[str]  # ip, user_id, api_key, etc.
    whitelist_ips: List[str] = None
    blacklist_ips: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.whitelist_ips is None:
            self.whitelist_ips = []
        if self.blacklist_ips is None:
            self.blacklist_ips = []

class SecurityManager:
    """Core security manager for healthcare API"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jwt_secret = config.get("jwt_secret", secrets.token_urlsafe(32))
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Storage
        self.api_keys: Dict[str, APIKey] = {}
        self.oauth_clients: Dict[str, OAuthClient] = {}
        self.jwt_tokens: Dict[str, Dict[str, Any]] = {}
        self.rate_limit_rules: List[RateLimitRule] = []
        
        # Initialize default configurations
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules"""
        
        # Clinical API rate limits
        self.rate_limit_rules.append(
            RateLimitRule(
                name="clinical_api",
                rule_type=RateLimitType.SLIDING_WINDOW,
                limit=100,  # 100 requests
                window=60,   # per minute
                burst_limit=150,
                user_identifiers=["api_key", "user_id"],
                whitelist_ips=["10.0.0.0/8", "172.16.0.0/12"],  # Internal networks
                blacklist_ips=["192.168.1.100"]  # Known problematic IP
            )
        )
        
        # FHIR API rate limits
        self.rate_limit_rules.append(
            RateLimitRule(
                name="fhir_api",
                rule_type=RateLimitType.TOKEN_BUCKET,
                limit=50,
                window=60,
                burst_limit=75,
                user_identifiers=["api_key"],
                whitelist_ips=["10.0.0.0/8"]
            )
        )
        
        # Analytics API rate limits
        self.rate_limit_rules.append(
            RateLimitRule(
                name="analytics_api",
                rule_type=RateLimitType.LEAKY_BUCKET,
                limit=200,
                window=60,
                burst_limit=250,
                user_identifiers=["api_key", "ip"]
            )
        )
        
        # Emergency override (for critical healthcare operations)
        self.rate_limit_rules.append(
            RateLimitRule(
                name="emergency_override",
                rule_type=RateLimitType.FIXED_WINDOW,
                limit=1000,
                window=60,
                burst_limit=2000,
                user_identifiers=["ip"],
                whitelist_ips=["10.0.0.0/8"],  # Only internal
                enabled=True
            )
        )
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[Permission],
        rate_limit: int = 100,
        expires_in_days: int = 365
    ) -> APIKey:
        """Create new API key"""
        
        # Generate secure API key
        key_id = secrets.token_urlsafe(16)
        raw_key = secrets.token_urlsafe(32)
        
        # Hash the key for storage
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        api_key = APIKey(
            key_id=key_id,
            user_id=user_id,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        )
        
        # Store key (in production, store hash)
        self.api_keys[key_id] = api_key
        self.api_keys[f"{key_id}_hash"] = key_hash
        
        logger.info(f"Created API key {key_id} for user {user_id}")
        
        # Return the actual key (only shown once)
        api_key_dict = api_key.to_dict()
        api_key_dict["api_key"] = raw_key
        
        return api_key
    
    async def validate_api_key(self, provided_key: str) -> Optional[APIKey]:
        """Validate API key"""
        
        key_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        
        # Find matching key
        for key_id, stored_hash in self.api_keys.items():
            if isinstance(stored_hash, str) and stored_hash == key_hash:
                api_key = self.api_keys[key_id.rstrip("_hash")]
                
                # Check expiration
                if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
                    logger.warning(f"API key {key_id} has expired")
                    return None
                
                # Update last used
                api_key.last_used = datetime.now(timezone.utc)
                
                return api_key
        
        return None
    
    async def generate_jwt_token(
        self,
        user_id: str,
        permissions: List[Permission],
        expires_in_minutes: int = 60,
        additional_claims: Dict[str, Any] = None
    ) -> str:
        """Generate JWT token"""
        
        now = datetime.now(timezone.utc)
        expiration = now + timedelta(minutes=expires_in_minutes)
        
        payload = {
            "iss": "healthcare-api",
            "sub": user_id,
            "aud": "healthcare-api",
            "exp": expiration,
            "iat": now,
            "jti": secrets.token_urlsafe(16),
            "permissions": [p.value for p in permissions],
            "token_type": "access_token"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        
        # Store token for validation
        self.jwt_tokens[payload["jti"]] = {
            "user_id": user_id,
            "expires_at": expiration,
            "permissions": [p.value for p in permissions]
        }
        
        logger.info(f"Generated JWT token for user {user_id}")
        return token
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check token ID
            jti = payload.get("jti")
            if jti not in self.jwt_tokens:
                return None
            
            token_info = self.jwt_tokens[jti]
            
            # Verify payload matches stored info
            if (payload["sub"] != token_info["user_id"] or 
                payload["exp"] != token_info["expires_at"]):
                return None
            
            # Check expiration
            if datetime.now(timezone.utc) > token_info["expires_at"]:
                del self.jwt_tokens[jti]
                return None
            
            return {
                "user_id": payload["sub"],
                "permissions": payload["permissions"],
                "expires_at": payload["exp"],
                "jti": jti
            }
        
        except jwt.InvalidTokenError:
            return None
    
    async def check_permission(
        self,
        user_permissions: List[Permission],
        required_permission: Permission
    ) -> bool:
        """Check if user has required permission"""
        return required_permission in user_permissions
    
    async def register_oauth_client(
        self,
        client_name: str,
        redirect_uris: List[str],
        scopes: List[Permission],
        grant_types: List[str] = None
    ) -> Dict[str, str]:
        """Register new OAuth2 client"""
        
        if grant_types is None:
            grant_types = ["client_credentials"]
        
        client_id = secrets.token_urlsafe(16)
        client_secret = secrets.token_urlsafe(32)
        
        # Hash secret for storage
        secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()
        
        oauth_client = OAuthClient(
            client_id=client_id,
            client_secret_hash=secret_hash,
            client_name=client_name,
            redirect_uris=redirect_uris,
            scopes=scopes,
            grant_types=grant_types,
            response_types=["code"],
            created_at=datetime.now(timezone.utc)
        )
        
        self.oauth_clients[client_id] = oauth_client
        
        logger.info(f"Registered OAuth client {client_id}")
        
        return {
            "client_id": client_id,
            "client_secret": client_secret
        }
    
    async def generate_oauth_token(
        self,
        client_id: str,
        client_secret: str,
        grant_type: str,
        scopes: List[Permission] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """Generate OAuth2 token"""
        
        # Validate client
        client = self.oauth_clients.get(client_id)
        if not client:
            raise ValueError("Invalid client")
        
        secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()
        if secret_hash != client.client_secret_hash:
            raise ValueError("Invalid client secret")
        
        if grant_type not in client.grant_types:
            raise ValueError(f"Grant type {grant_type} not supported")
        
        # Generate access token
        access_token = await self.generate_jwt_token(
            user_id or client_id,
            scopes or client.scopes,
            expires_in_minutes=60
        )
        
        # Generate refresh token
        refresh_token = secrets.token_urlsafe(32)
        refresh_expires = datetime.now(timezone.utc) + timedelta(days=30)
        
        # Store refresh token
        refresh_tokens = {
            refresh_token: {
                "client_id": client_id,
                "user_id": user_id,
                "scopes": [s.value for s in (scopes or client.scopes)],
                "expires_at": refresh_expires
            }
        }
        
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": refresh_token,
            "scope": " ".join([s.value for s in (scopes or client.scopes)])
        }

class RateLimiter:
    """Production-grade rate limiter with multiple algorithms"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.request_counts: Dict[str, deque] = {}
        self.token_buckets: Dict[str, Dict[str, float]] = {}
    
    async def check_rate_limit(
        self,
        identifier: str,
        rule: RateLimitRule
    ) -> Dict[str, Any]:
        """Check rate limit for identifier using specified rule"""
        
        # Skip if rule disabled
        if not rule.enabled:
            return {"allowed": True, "remaining": rule.limit, "reset_time": None}
        
        # Check whitelist/blacklist
        if identifier in rule.blacklist_ips:
            return {"allowed": False, "reason": "IP blacklisted"}
        
        if rule.whitelist_ips and not any(
            identifier.startswith(whitelist_ip) for whitelist_ip in rule.whitelist_ips
        ):
            return {"allowed": False, "reason": "IP not whitelisted"}
        
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=rule.window)
        
        if rule.rule_type == RateLimitType.SLIDING_WINDOW:
            return await self._check_sliding_window(identifier, rule, window_start)
        elif rule.rule_type == RateLimitType.FIXED_WINDOW:
            return await self._check_fixed_window(identifier, rule, window_start)
        elif rule.rule_type == RateLimitType.TOKEN_BUCKET:
            return await self._check_token_bucket(identifier, rule)
        elif rule.rule_type == RateLimitType.LEAKY_BUCKET:
            return await self._check_leaky_bucket(identifier, rule)
        else:
            return {"allowed": False, "reason": "Unknown rate limit algorithm"}
    
    async def _check_sliding_window(
        self,
        identifier: str,
        rule: RateLimitRule,
        window_start: datetime
    ) -> Dict[str, Any]:
        """Check rate limit using sliding window algorithm"""
        
        if identifier not in self.request_counts:
            self.request_counts[identifier] = deque()
        
        # Remove old requests
        while self.request_counts[identifier]:
            earliest_request = self.request_counts[identifier][0]
            if earliest_request < window_start:
                self.request_counts[identifier].popleft()
            else:
                break
        
        current_count = len(self.request_counts[identifier])
        
        if current_count < rule.limit:
            # Allow request and record it
            self.request_counts[identifier].append(datetime.now(timezone.utc))
            return {
                "allowed": True,
                "remaining": rule.limit - current_count - 1,
                "reset_time": window_start + timedelta(seconds=rule.window)
            }
        else:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "remaining": 0,
                "reset_time": window_start + timedelta(seconds=rule.window)
            }
    
    async def _check_fixed_window(
        self,
        identifier: str,
        rule: RateLimitRule,
        window_start: datetime
    ) -> Dict[str, Any]:
        """Check rate limit using fixed window algorithm"""
        
        # Simple implementation - in production would use more efficient storage
        current_minute = int(datetime.now(timezone.utc).timestamp() // 60)
        counter_key = f"{identifier}:{current_minute}"
        
        current_count = self.request_counts.get(counter_key, 0)
        
        if current_count < rule.limit:
            self.request_counts[counter_key] = current_count + 1
            return {
                "allowed": True,
                "remaining": rule.limit - current_count - 1,
                "reset_time": window_start + timedelta(seconds=rule.window)
            }
        else:
            return {
                "allowed": False,
                "reason": "Rate limit exceeded",
                "remaining": 0,
                "reset_time": window_start + timedelta(seconds=rule.window)
            }
    
    async def _check_token_bucket(
        self,
        identifier: str,
        rule: RateLimitRule
    ) -> Dict[str, Any]:
        """Check rate limit using token bucket algorithm"""
        
        current_time = time.time()
        
        if identifier not in self.token_buckets:
            self.token_buckets[identifier] = {
                "tokens": rule.limit,
                "last_refill": current_time
            }
        
        bucket = self.token_buckets[identifier]
        
        # Calculate tokens to add
        time_passed = current_time - bucket["last_refill"]
        tokens_to_add = time_passed * (rule.limit / rule.window)
        
        # Refill bucket
        bucket["tokens"] = min(rule.limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = current_time
        
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return {
                "allowed": True,
                "remaining": int(bucket["tokens"]),
                "reset_time": None
            }
        else:
            return {
                "allowed": False,
                "reason": "Token bucket empty",
                "remaining": 0,
                "reset_time": datetime.fromtimestamp(current_time + (1 / (rule.limit / rule.window)))
            }
    
    async def _check_leaky_bucket(
        self,
        identifier: str,
        rule: RateLimitRule
    ) -> Dict[str, Any]:
        """Check rate limit using leaky bucket algorithm"""
        
        current_time = time.time()
        
        if identifier not in self.token_buckets:
            self.token_buckets[identifier] = {
                "water_level": 0.0,
                "last_check": current_time
            }
        
        bucket = self.token_buckets[identifier]
        
        # Calculate water level after leak
        time_passed = current_time - bucket["last_check"]
        leaked_water = time_passed * (rule.limit / rule.window)
        bucket["water_level"] = max(0, bucket["water_level"] - leaked_water)
        
        # Check if request would overflow
        if bucket["water_level"] + 1.0 <= rule.limit:
            bucket["water_level"] += 1.0
            return {
                "allowed": True,
                "remaining": int(rule.limit - bucket["water_level"]),
                "reset_time": None
            }
        else:
            return {
                "allowed": False,
                "reason": "Leaky bucket overflow",
                "remaining": 0,
                "reset_time": datetime.fromtimestamp(
                    current_time + ((1.0 - (rule.limit - bucket["water_level"])) * rule.window / rule.limit)
                )
            }
    
    async def get_best_matching_rule(
        self,
        endpoint: str,
        user_identifiers: Dict[str, str]
    ) -> Optional[RateLimitRule]:
        """Find the best matching rate limit rule for endpoint and user"""
        
        # Simple matching - in production would use more sophisticated rules
        for rule in self.security_manager.rate_limit_rules:
            if self._rule_matches(rule, endpoint, user_identifiers):
                return rule
        
        return None
    
    def _rule_matches(
        self,
        rule: RateLimitRule,
        endpoint: str,
        user_identifiers: Dict[str, str]
    ) -> bool:
        """Check if rate limit rule matches the request"""
        
        # Check if any user identifier matches
        for identifier_type in rule.user_identifiers:
            if identifier_type in user_identifiers:
                identifier_value = user_identifiers[identifier_type]
                
                # Check whitelist
                if rule.whitelist_ips:
                    if any(identifier_value.startswith(wl_ip) for wl_ip in rule.whitelist_ips):
                        return True
                
                # Check blacklist
                if rule.blacklist_ips:
                    if any(identifier_value == bl_ip for bl_ip in rule.blacklist_ips):
                        return False
                
                return True
        
        return False

class SecurityMiddleware:
    """Security middleware for API requests"""
    
    def __init__(self, security_manager: SecurityManager, rate_limiter: RateLimiter):
        self.security_manager = security_manager
        self.rate_limiter = rate_limiter
    
    async def authenticate_request(
        self,
        headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Authenticate API request"""
        
        # Check for API key
        api_key_header = headers.get("X-API-Key")
        if api_key_header:
            api_key = await self.security_manager.validate_api_key(api_key_header)
            if api_key:
                return {
                    "auth_method": "api_key",
                    "user_id": api_key.user_id,
                    "permissions": api_key.permissions,
                    "rate_limit": api_key.rate_limit,
                    "key_id": api_key.key_id
                }
        
        # Check for Bearer token
        auth_header = headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            token_info = await self.security_manager.validate_jwt_token(token)
            if token_info:
                return {
                    "auth_method": "jwt",
                    "user_id": token_info["user_id"],
                    "permissions": token_info["permissions"],
                    "rate_limit": 1000,  # Default for JWT
                    "token_jti": token_info["jti"]
                }
        
        return None
    
    async def authorize_request(
        self,
        user_permissions: List[Permission],
        required_permission: Permission
    ) -> bool:
        """Authorize request based on permissions"""
        return await self.security_manager.check_permission(user_permissions, required_permission)
    
    async def check_rate_limit(
        self,
        endpoint: str,
        user_identifiers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Check rate limit for request"""
        
        rule = await self.rate_limiter.get_best_matching_rule(endpoint, user_identifiers)
        if not rule:
            return {"allowed": True, "remaining": 1000, "reset_time": None}
        
        # Use primary identifier
        primary_identifier = user_identifiers.get("api_key") or user_identifiers.get("user_id") or user_identifiers.get("ip", "unknown")
        
        return await self.rate_limiter.check_rate_limit(primary_identifier, rule)

# Example usage and configuration
if __name__ == "__main__":
    async def main():
        # Initialize security system
        config = {
            "jwt_secret": "your-super-secret-jwt-key",
            "environment": "production"
        }
        
        security_manager = SecurityManager(config)
        rate_limiter = RateLimiter(security_manager)
        middleware = SecurityMiddleware(security_manager, rate_limiter)
        
        # Create API key
        api_key = await security_manager.create_api_key(
            user_id="hospital_001",
            name="Main Hospital System",
            permissions=[Permission.PATIENT_READ, Permission.OBSERVATION_WRITE],
            rate_limit=100
        )
        
        print(f"Created API key: {api_key.key_id}")
        print("API Key (store securely):", [k for k in api_key.to_dict().keys() if "api_key" in k])
        
        # Validate API key
        # validated_key = await security_manager.validate_api_key("provided_api_key")
        # if validated_key:
        #     print(f"Validated API key for user: {validated_key.user_id}")
        
        # Generate JWT token
        jwt_token = await security_manager.generate_jwt_token(
            user_id="hospital_001",
            permissions=[Permission.PATIENT_READ, Permission.OBSERVATION_WRITE],
            expires_in_minutes=60
        )
        
        print(f"Generated JWT token: {jwt_token[:50]}...")
        
        # Register OAuth client
        oauth_creds = await security_manager.register_oauth_client(
            client_name="Healthcare Mobile App",
            redirect_uris=["https://app.healthcare.org/oauth/callback"],
            scopes=[Permission.PATIENT_READ, Permission.OBSERVATION_READ]
        )
        
        print(f"Registered OAuth client: {oauth_creds['client_id']}")
        
        # Test rate limiting
        rate_check = await middleware.check_rate_limit(
            endpoint="/api/v1/patients",
            user_identifiers={"api_key": api_key.key_id, "ip": "192.168.1.100"}
        )
        
        print(f"Rate limit check: {rate_check}")
        
        print("Security system initialized successfully!")
    
    asyncio.run(main())