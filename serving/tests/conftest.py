"""
Comprehensive test configuration and fixtures for medical AI serving system.

This file provides shared test fixtures, mock data generators, and configuration
for all test suites including HIPAA-compliant synthetic medical data.
"""

import asyncio
import pytest
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import logging
from unittest.mock import Mock, patch, AsyncMock

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Medical AI specific constants
MEDICAL_AI_MODELS = {
    "clinical_text_generation": "meta-llama/Llama-2-7b-chat-hf",
    "medical_qa": "microsoft/DialoGPT-medium",
    "symptom_analyzer": "distilbert-base-cased",
    "diagnostic_assistant": "google/flan-t5-large",
    "treatment_recommender": "allenai/biomed_roberta_base"
}

CLINICAL_ACCURACY_THRESHOLDS = {
    "symptom_analysis": 0.85,
    "diagnosis_suggestion": 0.80,
    "treatment_recommendation": 0.75,
    "medical_qa": 0.90,
    "clinical_text_generation": 0.85
}

PHI_PROTECTION_REQUIREMENTS = {
    "ssn_redaction": True,
    "dob_redaction": True,
    "phone_redaction": True,
    "email_redaction": True,
    "address_redaction": True,
    "mrn_redaction": True
}

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "load: mark test as a load/performance test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "compliance: mark test as a compliance test"
    )
    config.addinivalue_line(
        "markers", "medical: mark test as a medical-specific test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "phi: mark test as involving PHI data handling"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "environment": "test",
        "debug": True,
        "database_url": "sqlite:///:memory:",
        "model_cache_size": 5,
        "max_request_timeout": 30,
        "rate_limit_per_minute": 1000,
        "enable_phi_protection": True,
        "enable_clinical_validation": True,
        "log_level": "DEBUG",
        "test_mode": True
    }


@pytest.fixture
def mock_medical_data():
    """Generate HIPAA-compliant synthetic medical data."""
    return generate_synthetic_medical_data()


@pytest.fixture
def sample_patient_cases():
    """Provide sample patient cases for testing."""
    return generate_sample_patient_cases()


@pytest.fixture
def clinical_accuracy_metrics():
    """Provide clinical accuracy test metrics."""
    return CLINICAL_ACCURACY_THRESHOLDS.copy()


@pytest.fixture
def phi_protection_config():
    """Provide PHI protection configuration."""
    return PHI_PROTECTION_REQUIREMENTS.copy()


@pytest.fixture
def mock_api_client():
    """Create mock API client for testing."""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    return client


@pytest.fixture
def medical_ai_models():
    """Provide list of medical AI models for testing."""
    return list(MEDICAL_AI_MODELS.keys())


@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def performance_thresholds():
    """Provide performance testing thresholds."""
    return {
        "max_response_time_ms": 2000,
        "min_throughput_rps": 100,
        "max_memory_usage_mb": 2048,
        "max_cpu_usage_percent": 80,
        "max_error_rate": 0.01,
        "availability_target": 0.999
    }


def generate_synthetic_medical_data():
    """Generate HIPAA-compliant synthetic medical data for testing."""
    
    # Sample patient demographics (anonymized)
    patients = [
        {
            "patient_id": f"P{1000 + i}",
            "age_group": ["18-30", "31-45", "46-60", "61-75"][i % 4],
            "gender": ["M", "F", "O"][i % 3],
            "ethnicity": ["Caucasian", "Hispanic", "African American", "Asian", "Other"][i % 5],
            "zip_code": f"{10000 + i * 100}",
        }
        for i in range(100)
    ]
    
    # Medical conditions (generalized)
    conditions = [
        "Type 2 Diabetes", "Hypertension", "Asthma", "Depression", "Anxiety",
        "Arthritis", "Heart Disease", "Chronic Pain", "Sleep Apnea", "Obesity"
    ]
    
    # Symptoms (anonymized)
    symptoms = [
        "Increased thirst", "Frequent urination", "Fatigue", "Headaches",
        "Shortness of breath", "Chest pain", "Joint stiffness", "Mood changes",
        "Sleep disturbances", "Weight gain"
    ]
    
    # Medications (generic names only)
    medications = [
        "Metformin", "Lisinopril", "Albuterol", "Sertraline", "Ibuprofen",
        "Atorvastatin", "Amlodipine", "Omeprazole", "Gabapentin", "Levothyroxine"
    ]
    
    # Test cases
    test_cases = []
    for i in range(50):
        case = {
            "case_id": f"C{2000 + i}",
            "patient": patients[i % len(patients)],
            "presenting_complaint": symptoms[i % len(symptoms)],
            "differential_diagnosis": [conditions[j % len(conditions)] for j in range(3)],
            "recommended_tests": [
                "Complete Blood Count", "Basic Metabolic Panel", "Lipid Panel",
                "HbA1c", "Thyroid Function Tests", "Vitamin D Level"
            ],
            "treatment_plan": [
                f"Prescribe {medications[i % len(medications)]}",
                "Lifestyle modifications",
                "Regular follow-up in 4 weeks"
            ],
            "clinical_notes": f"Patient presents with {symptoms[i % len(symptoms)]}. "
                             f"Consider {conditions[i % len(conditions)]} in differential diagnosis.",
            "timestamp": (datetime.now() - timedelta(days=i)).isoformat()
        }
        test_cases.append(case)
    
    return {
        "patients": patients,
        "conditions": conditions,
        "symptoms": symptoms,
        "medications": medications,
        "test_cases": test_cases,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_patients": len(patients),
            "total_conditions": len(conditions),
            "total_symptoms": len(symptoms),
            "total_test_cases": len(test_cases),
            "phi_compliant": True,
            "synthetic": True
        }
    }


def generate_sample_patient_cases():
    """Generate sample patient cases for end-to-end testing."""
    
    cases = [
        {
            "case_id": "E2E_001",
            "scenario": "diabetes_management",
            "patient_profile": {
                "age": 45,
                "condition": "Type 2 Diabetes",
                "current_medications": ["Metformin 500mg BID"],
                "recent_readings": {"glucose": 180, "bp": "130/85"}
            },
            "interaction_steps": [
                {
                    "step": 1,
                    "user_input": "My glucose readings have been high lately",
                    "expected_response_type": "symptom_acknowledgment",
                    "required_fields": ["glucose_levels", "dietary_impact"]
                },
                {
                    "step": 2,
                    "user_input": "I'm eating a lot of carbs",
                    "expected_response_type": "dietary_guidance",
                    "required_fields": ["carb_limit", "meal_planning"]
                }
            ],
            "expected_outcomes": {
                "accuracy_threshold": 0.85,
                "response_time_target": "2.0s",
                "phi_protection": True,
                "clinical_validation": True
            }
        },
        {
            "case_id": "E2E_002",
            "scenario": "hypertension_monitoring",
            "patient_profile": {
                "age": 62,
                "condition": "Hypertension",
                "current_medications": ["Lisinopril 10mg daily"],
                "recent_readings": {"bp": "140/90", "pulse": 72}
            },
            "interaction_steps": [
                {
                    "step": 1,
                    "user_input": "My blood pressure readings are still elevated",
                    "expected_response_type": "assessment_guidance",
                    "required_fields": ["reading_confirmation", "lifestyle_factors"]
                }
            ],
            "expected_outcomes": {
                "accuracy_threshold": 0.85,
                "response_time_target": "2.0s",
                "phi_protection": True,
                "clinical_validation": True
            }
        }
    ]
    
    return cases


# Utility functions for test data generation
def generate_test_conversation(conversation_id: str, num_turns: int = 5):
    """Generate test conversation for chat-based models."""
    
    turns = []
    for i in range(num_turns):
        turn = {
            "turn_id": f"{conversation_id}_turn_{i}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Test message {i}" if i % 2 == 0 else f"Test response {i}",
            "timestamp": (datetime.now() - timedelta(minutes=num_turns-i)).isoformat(),
            "metadata": {
                "session_id": conversation_id,
                "turn_number": i,
                "phi_indicators": [] if i % 2 == 0 else ["medical_advice"]
            }
        }
        turns.append(turn)
    
    return {
        "conversation_id": conversation_id,
        "turns": turns,
        "total_turns": num_turns,
        "started_at": turns[0]["timestamp"],
        "ended_at": turns[-1]["timestamp"]
    }


def generate_load_test_scenarios():
    """Generate scenarios for load testing."""
    
    return {
        "text_generation_load": {
            "concurrent_users": 50,
            "requests_per_second": 10,
            "test_duration_minutes": 15,
            "model": "clinical_text_generation",
            "input_data": [
                "Patient presents with symptoms of",
                "Treatment options include",
                "Differential diagnosis considerations are",
                "Clinical recommendations suggest",
                "Medical assessment indicates"
            ]
        },
        "qa_load": {
            "concurrent_users": 25,
            "requests_per_second": 5,
            "test_duration_minutes": 10,
            "model": "medical_qa",
            "input_data": [
                "What are the symptoms of diabetes?",
                "How should hypertension be managed?",
                "What tests are needed for chest pain?",
                "When should I see a doctor for fatigue?",
                "What lifestyle changes help with weight loss?"
            ]
        },
        "symptom_analysis_load": {
            "concurrent_users": 30,
            "requests_per_second": 8,
            "test_duration_minutes": 12,
            "model": "symptom_analyzer",
            "input_data": [
                "Patient reports fatigue and weight loss",
                "Chest pain with shortness of breath",
                "Frequent urination and excessive thirst",
                "Joint pain and morning stiffness",
                "Headaches and visual changes"
            ]
        }
    }


def create_test_metrics():
    """Create standard test metrics structure."""
    return {
        "performance": {
            "response_time_ms": 0,
            "throughput_rps": 0,
            "memory_usage_mb": 0,
            "cpu_usage_percent": 0,
            "error_rate": 0.0
        },
        "accuracy": {
            "clinical_accuracy": 0.0,
            "semantic_similarity": 0.0,
            "factual_correctness": 0.0,
            "completeness_score": 0.0
        },
        "compliance": {
            "phi_protection_score": 0.0,
            "regulatory_compliance": True,
            "audit_trail_complete": True,
            "data_encryption_valid": True
        },
        "reliability": {
            "availability": 0.0,
            "consistency": 0.0,
            "fault_tolerance": True,
            "recovery_time_ms": 0
        }
    }