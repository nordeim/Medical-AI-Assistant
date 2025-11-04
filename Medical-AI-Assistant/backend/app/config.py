"""
Application Configuration

Centralized configuration management using Pydantic Settings.
All configuration values are loaded from environment variables.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application
    APP_NAME: str = "Medical AI Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # API
    API_V1_PREFIX: str = "/api"
    ALLOWED_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL connection string")
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_ECHO: bool = False
    
    # Security
    SECRET_KEY: str = Field(..., description="JWT secret key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Password
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_BCRYPT_ROUNDS: int = 12
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    
    # Model Configuration
    MODEL_PATH: str = Field(..., description="Path to LLM model")
    MODEL_DEVICE: str = "cuda"
    MODEL_MAX_LENGTH: int = 2048
    MODEL_TEMPERATURE: float = 0.7
    MODEL_TOP_P: float = 0.9
    USE_8BIT: bool = True
    USE_FLASH_ATTENTION: bool = False
    
    # Vector Store
    VECTOR_STORE_TYPE: str = "chroma"
    VECTOR_STORE_PATH: str = Field(..., description="Path to vector store data")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    
    # RAG Configuration
    RAG_TOP_K: int = 5
    RAG_SIMILARITY_THRESHOLD: float = 0.7
    RAG_USE_RERANKER: bool = True
    RAG_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Agent Configuration
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_STREAM_TOKENS: bool = True
    AGENT_VERBOSE: bool = False
    
    # Safety Filters
    SAFETY_ENABLED: bool = True
    SAFETY_BLOCK_DIAGNOSIS: bool = True
    SAFETY_BLOCK_PRESCRIPTION: bool = True
    SAFETY_LOG_ALL_FILTERS: bool = True
    
    # Redis (for caching and rate limiting)
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "/app/logs/app.log"
    LOG_ROTATION: str = "500 MB"
    LOG_RETENTION: str = "30 days"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # LangSmith (optional observability)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: Optional[str] = None
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "medical-ai-assistant"
    
    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_origins(cls, v):
        """Parse comma-separated origins string into list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("MODEL_DEVICE")
    @classmethod
    def validate_device(cls, v):
        """Validate device selection"""
        if v not in ["cuda", "cpu", "mps"]:
            raise ValueError("MODEL_DEVICE must be one of: cuda, cpu, mps")
        return v
    
    @field_validator("VECTOR_STORE_TYPE")
    @classmethod
    def validate_vector_store(cls, v):
        """Validate vector store type"""
        if v not in ["chroma", "qdrant", "milvus"]:
            raise ValueError("VECTOR_STORE_TYPE must be one of: chroma, qdrant, milvus")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This function is cached to ensure we only load settings once
    and reuse the same instance throughout the application lifecycle.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Convenience access
settings = get_settings()
