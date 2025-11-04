"""
Configuration management for the model serving infrastructure.
Supports multiple environments with proper validation and security.
"""

from typing import Optional, Dict, Any
import os
from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for ML models and serving."""
    
    model_name: str = Field(default="microsoft/DialoGPT-medium", description="Primary model identifier")
    model_path: Optional[str] = Field(default=None, description="Local path to model if not using Hugging Face Hub")
    model_revision: Optional[str] = Field(default="main", description="Model revision/tag")
    
    # Model serving settings
    max_length: int = Field(default=512, description="Maximum sequence length")
    temperature: float = Field(default=0.7, description="Sampling temperature", ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter", ge=0.1, le=1.0)
    top_k: int = Field(default=50, description="Top-k sampling parameter", ge=1, le=100)
    
    # Quantization settings
    use_quantization: bool = Field(default=True, description="Enable model quantization")
    quantization_type: str = Field(default="4bit", description="Quantization type", pattern="^(4bit|8bit)$")
    load_in_8bit: bool = Field(default=False, description="Load model in 8-bit precision")
    load_in_4bit: bool = Field(default=True, description="Load model in 4-bit precision")
    
    # Device settings
    device_map: Optional[Dict[str, str]] = Field(default=None, description="Custom device mapping")
    torch_dtype: str = Field(default="float16", description="Torch data type")
    
    model_config = SettingsConfigDict(
        env_prefix="MODEL_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )


class ServingConfig(BaseSettings):
    """Configuration for the serving infrastructure."""
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, description="Server port", ge=1, le=65535)
    workers: int = Field(default=4, description="Number of Gunicorn workers", ge=1)
    
    # Security settings
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    allowed_origins: list[str] = Field(default=["http://localhost:3000"], description="CORS allowed origins")
    cors_allow_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_allow_methods: list[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS allowed methods")
    cors_allow_headers: list[str] = Field(default=["*"], description="CORS allowed headers")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=60, description="Requests per minute per client")
    rate_limit_burst: int = Field(default=10, description="Burst requests allowed")
    
    # Request limits
    max_request_size: int = Field(default=10485760, description="Maximum request size in bytes (10MB)")
    max_concurrent_requests: int = Field(default=10, description="Maximum concurrent requests per model")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    
    model_config = SettingsConfigDict(
        env_prefix="SERVING_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )


class CacheConfig(BaseSettings):
    """Configuration for caching system."""
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_ssl: bool = Field(default=False, description="Use SSL for Redis connection")
    
    # Cache settings
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    max_cache_size: int = Field(default=1000, description="Maximum cached items")
    enable_model_cache: bool = Field(default=True, description="Enable model result caching")
    enable_response_cache: bool = Field(default=True, description="Enable response caching")
    
    # Disk cache
    disk_cache_dir: str = Field(default="./cache", description="Disk cache directory")
    disk_cache_size: int = Field(default=10737418240, description="Disk cache size in bytes (10GB)")
    
    model_config = SettingsConfigDict(
        env_prefix="CACHE_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )


class LoggingConfig(BaseSettings):
    """Configuration for logging infrastructure."""
    
    # Log levels
    log_level: str = Field(default="INFO", description="Logging level", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field(default="json", description="Log format", pattern="^(json|text)$")
    
    # File logging
    log_file: Optional[str] = Field(default="./logs/serving.log", description="Log file path")
    log_file_max_size: int = Field(default=104857600, description="Maximum log file size (100MB)")
    log_file_backup_count: int = Field(default=5, description="Number of backup log files")
    enable_file_logging: bool = Field(default=True, description="Enable file logging")
    
    # Console logging
    enable_console_logging: bool = Field(default=True, description="Enable console logging")
    console_log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Console log format")
    
    # Structured logging
    enable_structured_logging: bool = Field(default=True, description="Enable structured JSON logging")
    include_request_id: bool = Field(default=True, description="Include request ID in logs")
    include_user_id: bool = Field(default=True, description="Include user ID in logs (if available)")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")
    
    # Sentry
    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )


class MedicalComplianceConfig(BaseSettings):
    """Configuration for medical data compliance and security."""
    
    # Data encryption
    enable_encryption: bool = Field(default=True, description="Enable data encryption")
    encryption_key: Optional[str] = Field(default=None, description="Encryption key")
    
    # PHI handling
    phi_redaction: bool = Field(default=True, description="Enable PHI redaction")
    allowed_phi_fields: list[str] = Field(default=["patient_id", "timestamp"], description="Allowed PHI fields")
    phi_logging: bool = Field(default=False, description="Enable PHI logging (for compliance)")
    
    # Data retention
    data_retention_days: int = Field(default=30, description="Data retention period in days")
    auto_delete: bool = Field(default=True, description="Auto delete data after retention period")
    
    # Audit logging
    enable_audit_log: bool = Field(default=True, description="Enable audit logging")
    audit_log_file: str = Field(default="./logs/audit.log", description="Audit log file path")
    
    # Access control
    enable_rbac: bool = Field(default=True, description="Enable role-based access control")
    default_role: str = Field(default="user", description="Default user role")
    
    model_config = SettingsConfigDict(
        env_prefix="MEDICAL_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )


class DatabaseConfig(BaseSettings):
    """Configuration for database connections."""
    
    # Database URL
    database_url: str = Field(default="postgresql://user:password@localhost:5432/medical_ai", description="Database connection URL")
    
    # Connection settings
    database_echo: bool = Field(default=False, description="Enable SQL query logging")
    database_pool_size: int = Field(default=5, description="Database connection pool size")
    database_max_overflow: int = Field(default=10, description="Database connection pool max overflow")
    
    # Migration
    enable_alembic: bool = Field(default=True, description="Enable database migrations")
    alembic_path: str = Field(default="alembic", description="Alembic migration directory")
    
    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8"
    )


class Settings(BaseSettings):
    """Main application settings combining all configurations."""
    
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )
    
    # Environment
    environment: str = Field(default="development", description="Application environment", pattern="^(development|testing|staging|production)$")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    medical: MedicalComplianceConfig = Field(default_factory=MedicalComplianceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment setting."""
        if v not in ["development", "testing", "staging", "production"]:
            raise ValueError("Environment must be one of: development, testing, staging, production")
        return v
    
    @validator("debug")
    def validate_debug(cls, v, values):
        """Validate debug setting based on environment."""
        if v and values.get("environment") == "production":
            raise ValueError("Debug mode cannot be enabled in production environment")
        return v


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def validate_config() -> Dict[str, Any]:
    """Validate current configuration and return validation results."""
    try:
        settings_dict = settings.dict()
        return {
            "valid": True,
            "environment": settings.environment,
            "model_name": settings.model.model_name,
            "port": settings.serving.port,
            "cache_enabled": settings.cache.enable_model_cache,
            "encryption_enabled": settings.medical.enable_encryption,
            "audit_logging": settings.medical.enable_audit_log
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }