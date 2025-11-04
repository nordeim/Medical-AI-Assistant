"""
Phase 7 Integration Testing Framework

Comprehensive end-to-end integration testing for Medical AI Assistant Phase 7.
Tests complete system integration including frontend, backend, training, serving,
patient chat flows, nurse dashboard workflows, real-time communication, performance,
and system reliability.
"""

__version__ = "1.0.0"

from .test_complete_system_integration import *
from .test_patient_chat_flow import *
from .test_nurse_dashboard_workflow import *
from .test_training_serving_integration import *
from .test_websocket_communication import *
from .test_model_serving_performance import *
from .test_system_reliability import *