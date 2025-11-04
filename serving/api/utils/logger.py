"""
Logging Configuration for Medical AI Inference API
Structured logging with medical and security contexts
"""

import logging
import sys
from typing import Any, Dict, Optional
from datetime import datetime
import json
from functools import lru_cache
import structlog

from ..config import get_settings


class MedicalLogger:
    """Specialized logger for medical AI operations"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
        self.settings = get_settings()
    
    def log_medical_operation(self, 
                            operation: str, 
                            patient_id: Optional[str] = None,
                            success: bool = True,
                            details: Optional[Dict[str, Any]] = None) -> None:
        """Log medical operation with proper context"""
        
        log_data = {
            "operation": operation,
            "patient_id": self._anonymize_patient_id(patient_id),
            "success": success,
            "timestamp": datetime.now().isoformat(),
            **(details or {})
        }
        
        level = "info" if success else "error"
        getattr(self.logger, level)("medical_operation", **log_data)
    
    def log_phi_access(self, 
                      operation: str, 
                      phi_type: str,
                      success: bool = True,
                      user_id: Optional[str] = None) -> None:
        """Log PHI access attempts"""
        
        log_data = {
            "operation": operation,
            "phi_type": phi_type,
            "user_id": user_id,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "requires_audit": True
        }
        
        level = "info" if success else "warning"
        getattr(self.logger, level)("phi_access", **log_data)
    
    def log_clinical_decision(self,
                            decision_type: str,
                            confidence: float,
                            recommendation: str,
                            patient_id: Optional[str] = None) -> None:
        """Log clinical decision support events"""
        
        log_data = {
            "decision_type": decision_type,
            "confidence": confidence,
            "recommendation": recommendation,
            "patient_id": self._anonymize_patient_id(patient_id),
            "timestamp": datetime.now().isoformat(),
            "requires_review": confidence < self.settings.clinical_confidence_threshold
        }
        
        self.logger.info("clinical_decision", **log_data)
    
    def log_security_event(self,
                          event_type: str,
                          severity: str,
                          description: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None) -> None:
        """Log security events"""
        
        log_data = {
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.now().isoformat()
        }
        
        level_map = {
            "low": "info",
            "medium": "warning", 
            "high": "error",
            "critical": "critical"
        }
        
        level = level_map.get(severity.lower(), "info")
        getattr(self.logger, level)("security_event", **log_data)
    
    def log_model_performance(self,
                            operation: str,
                            latency_ms: float,
                            success: bool,
                            model_version: str,
                            details: Optional[Dict[str, Any]] = None) -> None:
        """Log model performance metrics"""
        
        log_data = {
            "operation": operation,
            "latency_ms": latency_ms,
            "success": success,
            "model_version": model_version,
            "timestamp": datetime.now().isoformat(),
            **(details or {})
        }
        
        level = "info" if success else "warning"
        getattr(self.logger, level)("model_performance", **log_data)
    
    def log_validation_failure(self,
                             validation_type: str,
                             failure_reason: str,
                             data_sample: Optional[str] = None,
                             user_id: Optional[str] = None) -> None:
        """Log validation failures"""
        
        log_data = {
            "validation_type": validation_type,
            "failure_reason": failure_reason,
            "data_sample": data_sample[:100] if data_sample else None,  # Truncate for privacy
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.warning("validation_failure", **log_data)
    
    def _anonymize_patient_id(self, patient_id: Optional[str]) -> Optional[str]:
        """Anonymize patient ID for logging"""
        if not patient_id:
            return None
        
        # Simple anonymization - in production, use proper hashing
        if len(patient_id) > 4:
            return f"{patient_id[:2]}***{patient_id[-2:]}"
        return "***"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def configure_logging():
    """Configure structured logging"""
    
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.log_format == "json" 
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure Python logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",  # structlog handles formatting
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # Set specific log levels for third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)


@lru_cache()
def get_logger(name: str) -> MedicalLogger:
    """Get cached logger instance"""
    return MedicalLogger(name)