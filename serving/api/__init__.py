"""
Medical AI Assistant - Serving API
Phase 6: Comprehensive Inference API Endpoints

Enterprise-grade API endpoints with medical validation, PHI protection,
and high availability for healthcare applications.
"""

__version__ = "1.0.0"
__author__ = "Medical AI Assistant Team"
__email__ = "team@medical-ai-assistant.com"

from .main import app
from .middleware import MedicalValidationMiddleware
from .endpoints import (
    inference,
    streaming,
    batch,
    health,
    conversation,
    clinical_decision_support,
    validation
)

__all__ = [
    "app",
    "MedicalValidationMiddleware",
    "inference",
    "streaming", 
    "batch",
    "health",
    "conversation",
    "clinical_decision_support",
    "validation"
]