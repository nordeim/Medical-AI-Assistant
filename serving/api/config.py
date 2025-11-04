"""
Configuration Management for Medical AI Inference API
Enterprise-grade settings with environment variable support
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation"""
    
    # Server configuration
    host: str = Field("0.0.0.0", description="Server host address")
    port: int = Field(8000, description="Server port number")
    debug: bool = Field(False, description="Enable debug mode")
    workers: int = Field(4, description="Number of worker processes")
    
    # Security settings
    secret_key: str = Field(..., description="Secret key for JWT signing")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, description="Token expiration time")
    
    # CORS settings
    allowed_origins: List[str] = Field(
        ["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    
    # Database settings
    database_url: str = Field(
        "postgresql://user:password@localhost:5432/medical_ai",
        description="Database connection URL"
    )
    redis_url: str = Field(
        "redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # Model configuration
    model_name: str = Field(
        "medical-ai-assistant",
        description="Name of the medical AI model"
    )
    model_version: str = Field(
        "1.0.0",
        description="Model version"
    )
    max_tokens: int = Field(
        2048,
        description="Maximum tokens for model output"
    )
    temperature: float = Field(
        0.7,
        description="Model temperature setting",
        ge=0.0,
        le=2.0
    )
    
    # RAG configuration
    vector_store_type: str = Field(
        "pgvector",
        description="Vector store type"
    )
    embedding_model: str = Field(
        "all-MiniLM-L6-v2",
        description="Embedding model name"
    )
    chunk_size: int = Field(
        1000,
        description="Text chunk size for RAG"
    )
    chunk_overlap: int = Field(
        200,
        description="Chunk overlap for RAG"
    )
    similarity_threshold: float = Field(
        0.7,
        description="Similarity threshold for retrieval",
        ge=0.0,
        le=1.0
    )
    
    # Medical validation settings
    enable_medical_validation: bool = Field(
        True,
        description="Enable medical data validation"
    )
    strict_mode: bool = Field(
        False,
        description="Enable strict medical validation mode"
    )
    medical_terms: Optional[List[str]] = Field(
        None,
        description="Custom medical terms for validation"
    )
    
    # PHI protection settings
    enable_phi_detection: bool = Field(
        True,
        description="Enable PHI detection"
    )
    phi_redaction: bool = Field(
        True,
        description="Enable PHI redaction"
    )
    phi_modes: List[str] = Field(
        ["mask", "anonymize"],
        description="PHI protection modes"
    )
    
    # Rate limiting
    enable_rate_limiting: bool = Field(
        True,
        description="Enable rate limiting"
    )
    rate_limit_per_minute: int = Field(
        60,
        description="Requests per minute limit"
    )
    rate_limit_per_hour: int = Field(
        3600,
        description="Requests per hour limit"
    )
    
    # Logging
    log_level: str = Field(
        "INFO",
        description="Logging level"
    )
    log_format: str = Field(
        "json",
        description="Log format (json/text)"
    )
    enable_audit_logging: bool = Field(
        True,
        description="Enable audit logging"
    )
    
    # Monitoring
    enable_metrics: bool = Field(
        True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        9090,
        description="Metrics port"
    )
    
    # Security headers
    enable_security_headers: bool = Field(
        True,
        description="Enable security headers"
    )
    
    # SSL settings
    ssl_keyfile: Optional[str] = Field(
        None,
        description="SSL key file path"
    )
    ssl_certfile: Optional[str] = Field(
        None,
        description="SSL certificate file path"
    )
    
    # Model serving
    model_cache_size: int = Field(
        100,
        description="Model cache size"
    )
    batch_size: int = Field(
        32,
        description="Default batch size"
    )
    max_concurrent_requests: int = Field(
        100,
        description="Maximum concurrent requests"
    )
    
    # Clinical decision support
    enable_clinical_validation: bool = Field(
        True,
        description="Enable clinical decision support"
    )
    clinical_confidence_threshold: float = Field(
        0.8,
        description="Clinical confidence threshold",
        ge=0.0,
        le=1.0
    )
    
    # Streaming settings
    enable_streaming: bool = Field(
        True,
        description="Enable streaming responses"
    )
    streaming_chunk_size: int = Field(
        512,
        description="Streaming chunk size"
    )
    
    # Conversation settings
    max_conversation_length: int = Field(
        50,
        description="Maximum conversation turns"
    )
    conversation_retention_days: int = Field(
        30,
        description="Conversation retention period in days"
    )
    
    # Cache settings
    cache_ttl: int = Field(
        3600,
        description="Cache TTL in seconds"
    )
    cache_size: int = Field(
        10000,
        description="Cache size limit"
    )
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if not v or len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'sqlite:///')):
            raise ValueError('Database URL must be PostgreSQL or SQLite')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        if not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()