"""
Structured logging infrastructure for the model serving system.
Provides centralized logging with proper medical data compliance.
"""

import logging
import logging.handlers
import structlog
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import os
import uuid
import asyncio
from datetime import datetime
from contextvars import ContextVar

from .settings import get_settings


# Context variables for request tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


class RequestContextFilter(logging.Filter):
    """Filter to add request context to log records."""
    
    def filter(self, record):
        record.request_id = request_id_var.get()
        record.user_id = user_id_var.get()
        record.session_id = session_id_var.get()
        return True


class MedicalDataFilter(logging.Filter):
    """Filter to redact sensitive medical data from logs."""
    
    PHI_KEYWORDS = [
        'ssn', 'social_security', 'patient_id', 'medical_record', 'dob',
        'birth_date', 'phone', 'email', 'address', 'diagnosis', 'prescription',
        'treatment', 'symptom', 'condition', 'allergy', 'medication'
    ]
    
    def filter(self, record):
        if not self.settings.medical.phi_redaction:
            return True
        
        # Redact PHI from log message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            message = record.msg
            for keyword in self.PHI_KEYWORDS:
                if keyword in message.lower():
                    # Replace potential PHI patterns
                    message = self._redact_phi(message, keyword)
            record.msg = message
        
        return True
    
    def _redact_phi(self, text: str, keyword: str) -> str:
        """Redact PHI patterns from text."""
        import re
        
        # Common PHI patterns
        patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'patient_id': r'patient[_\-]?id[:\s]*[\w\-]+',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        if keyword in patterns:
            text = re.sub(patterns[keyword], '[REDACTED]', text, flags=re.IGNORECASE)
        
        return text


class MedicalAuditLogger:
    """Specialized logger for medical audit trails."""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("medical_audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_file,
            maxBytes=104857600,  # 100MB
            backupCount=5
        )
        
        # Custom formatter for audit logs
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_access(self, user_id: str, action: str, resource: str, 
                   details: Optional[Dict[str, Any]] = None):
        """Log data access for compliance."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "details": details or {}
        }
        
        self.logger.info(json.dumps(audit_entry))
    
    def log_phi_access(self, user_id: str, phi_type: str, action: str):
        """Log PHI access for compliance reporting."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "phi_type": phi_type,
            "request_id": request_id_var.get(),
            "audit_type": "phi_access"
        }
        
        self.logger.info(json.dumps(audit_entry))


class ModelServingLogger:
    """Logger for model serving operations."""
    
    def __init__(self, name: str = "model_serving"):
        self.logger = structlog.get_logger(name)
        self.settings = get_settings()
    
    def log_model_load(self, model_name: str, success: bool, 
                      load_time: Optional[float] = None, 
                      error: Optional[str] = None):
        """Log model loading events."""
        event = {
            "event": "model_load",
            "model_name": model_name,
            "success": success,
            "load_time_seconds": load_time,
            "error": error
        }
        
        if success:
            self.logger.info("Model loaded successfully", **event)
        else:
            self.logger.error("Model load failed", **event)
    
    def log_prediction(self, model_name: str, input_size: int, 
                      output_size: int, processing_time: float,
                      confidence: Optional[float] = None):
        """Log prediction requests and responses."""
        event = {
            "event": "prediction",
            "model_name": model_name,
            "input_size": input_size,
            "output_size": output_size,
            "processing_time_ms": round(processing_time * 1000, 2),
            "confidence": confidence
        }
        
        self.logger.info("Prediction completed", **event)
    
    def log_cache_operation(self, operation: str, key: str, 
                           hit: bool, size: Optional[int] = None):
        """Log cache operations."""
        event = {
            "event": "cache_operation",
            "operation": operation,
            "key": key,
            "hit": hit,
            "size_bytes": size
        }
        
        self.logger.debug("Cache operation", **event)


class MetricsLogger:
    """Logger for performance metrics."""
    
    def __init__(self):
        self.logger = structlog.get_logger("metrics")
    
    def log_request_metrics(self, endpoint: str, method: str, 
                          status_code: int, response_time: float,
                          request_size: int, response_size: int):
        """Log HTTP request metrics."""
        event = {
            "event": "http_request",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": round(response_time * 1000, 2),
            "request_size_bytes": request_size,
            "response_size_bytes": response_size
        }
        
        self.logger.info("HTTP request metrics", **event)
    
    def log_resource_usage(self, cpu_percent: float, memory_mb: float,
                          gpu_memory_mb: Optional[float] = None):
        """Log system resource usage."""
        event = {
            "event": "resource_usage",
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "gpu_memory_mb": gpu_memory_mb
        }
        
        self.logger.debug("Resource usage", **event)


def setup_structured_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure Python logging
    logging.basicConfig(
        format="%(message)s",
        stream=os.sys.stdout,
        level=getattr(logging, settings.logging.log_level.upper())
    )
    
    # Remove default handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure structlog
    if settings.logging.enable_structured_logging:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Text logging format
        logging.basicConfig(
            format=settings.logging.console_log_format,
            level=getattr(logging, settings.logging.log_level.upper())
        )
    
    # File logging configuration
    if settings.logging.enable_file_logging and settings.logging.log_file:
        log_file = Path(settings.logging.log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=settings.logging.log_file_max_size,
            backupCount=settings.logging.log_file_backup_count
        )
        
        if settings.logging.log_format == "json":
            formatter = structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer()
            )
        else:
            formatter = logging.Formatter(settings.logging.console_log_format)
        
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add context filters
    context_filter = RequestContextFilter()
    medical_filter = MedicalDataFilter()
    medical_filter.settings = settings
    
    for logger_name in ['model_serving', 'api', 'cache', 'medical_audit']:
        logger = logging.getLogger(logger_name)
        logger.addFilter(context_filter)
        logger.addFilter(medical_filter)


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def get_audit_logger() -> MedicalAuditLogger:
    """Get the medical audit logger."""
    settings = get_settings()
    return MedicalAuditLogger(settings.medical.audit_log_file)


def get_model_logger(name: str = "model_serving") -> ModelServingLogger:
    """Get a model serving logger."""
    return ModelServingLogger(name)


def get_metrics_logger() -> MetricsLogger:
    """Get a metrics logger."""
    return MetricsLogger()


class LoggingContextManager:
    """Context manager for logging contexts."""
    
    def __init__(self, request_id: Optional[str] = None, 
                 user_id: Optional[str] = None, 
                 session_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
    
    def __enter__(self):
        request_id_var.set(self.request_id)
        if self.user_id:
            user_id_var.set(self.user_id)
        if self.session_id:
            session_id_var.set(self.session_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_var.set('')
        user_id_var.set('')
        session_id_var.set('')


async def log_async_operation(operation: str, coro, *args, **kwargs):
    """Log async operations with timing."""
    start_time = datetime.utcnow()
    logger = get_logger("async_operations")
    
    try:
        result = await coro(*args, **kwargs)
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(
            f"Async operation completed",
            operation=operation,
            duration_seconds=duration,
            success=True
        )
        return result
    
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.error(
            f"Async operation failed",
            operation=operation,
            duration_seconds=duration,
            success=False,
            error=str(e)
        )
        raise


# Initialize logging on module import
setup_structured_logging()