"""
Test Configuration and Fixtures for Phase 7 Integration Testing

Provides comprehensive test fixtures, mock services, and configuration
for end-to-end integration testing of all system components.
"""

import asyncio
import pytest
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock, MagicMock
import aiohttp
import websockets
import structlog

# Test Configuration
TEST_CONFIG = {
    "base_url": "http://localhost:8000",
    "websocket_url": "ws://localhost:8000/ws",
    "frontend_url": "http://localhost:3000",
    "timeout": 30,
    "max_retries": 3,
    "test_data_dir": "/workspace/testing/integration/test_data",
    "mock_services": {
        "patient_service": True,
        "nurse_service": True,
        "model_service": True,
        "training_service": True
    }
}

# Mock Data
MOCK_PATIENT_DATA = {
    "patient_001": {
        "id": "patient_001",
        "session_id": "sess_001",
        "demographics": {
            "age": 45,
            "gender": "F",
            "name": "Test Patient 1"
        },
        "chief_complaint": "Chest pain and shortness of breath",
        "symptoms": ["chest_pain", "shortness_of_breath", "diaphoresis"],
        "severity": "severe",
        "duration": "2 hours",
        "risk_factors": ["diabetes", "hypertension", "family_history"],
        "created_at": datetime.utcnow().isoformat(),
        "status": "waiting_for_assessment"
    },
    "patient_002": {
        "id": "patient_002", 
        "session_id": "sess_002",
        "demographics": {
            "age": 32,
            "gender": "M", 
            "name": "Test Patient 2"
        },
        "chief_complaint": "Persistent headache",
        "symptoms": ["headache", "fatigue", "nausea"],
        "severity": "moderate",
        "duration": "3 days",
        "risk_factors": ["stress", "poor_sleep"],
        "created_at": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        "status": "in_assessment"
    },
    "patient_003": {
        "id": "patient_003",
        "session_id": "sess_003", 
        "demographics": {
            "age": 58,
            "gender": "F",
            "name": "Test Patient 3"
        },
        "chief_complaint": "Abdominal pain",
        "symptoms": ["severe_abdominal_pain", "nausea", "vomiting"],
        "severity": "severe",
        "duration": "6 hours",
        "risk_factors": ["previous_abdominal_surgery"],
        "created_at": datetime.utcnow().isoformat(),
        "status": "escalated_to_nurse"
    }
}

MOCK_NURSE_DATA = {
    "nurse_001": {
        "id": "nurse_001",
        "name": "Nurse Test 1",
        "specialization": "Emergency Medicine",
        "experience_years": 8,
        "active": True,
        "current_cases": 2,
        "max_cases": 5,
        "skills": ["triage", "emergency_care", "patient_communication"]
    },
    "nurse_002": {
        "id": "nurse_002", 
        "name": "Nurse Test 2",
        "specialization": "Internal Medicine",
        "experience_years": 12,
        "active": True,
        "current_cases": 1,
        "max_cases": 6,
        "skills": ["chronic_care", "medication_management", "patient_education"]
    }
}

MOCK_MODEL_DATA = {
    "clinical_assessment_v1": {
        "model_id": "clinical_assessment_v1",
        "version": "1.0.0",
        "type": "clinical_analysis",
        "accuracy": 0.92,
        "training_data_size": 100000,
        "last_updated": datetime.utcnow().isoformat(),
        "deployment_status": "active",
        "serving_endpoint": "/models/clinical_assessment_v1/predict"
    },
    "symptom_analyzer_v2": {
        "model_id": "symptom_analyzer_v2",
        "version": "2.0.1", 
        "type": "symptom_analysis",
        "accuracy": 0.88,
        "training_data_size": 75000,
        "last_updated": datetime.utcnow().isoformat(),
        "deployment_status": "active",
        "serving_endpoint": "/models/symptom_analyzer_v2/predict"
    }
}

class MockWebSocketManager:
    """Mock WebSocket manager for testing."""
    
    def __init__(self):
        self.connections = {}
        self.message_history = []
        self.logger = structlog.get_logger("mock.websocket")
    
    async def connect(self, session_id: str, user_type: str) -> Mock:
        """Mock WebSocket connection."""
        websocket = Mock()
        websocket.session_id = session_id
        websocket.user_type = user_type
        websocket.is_closed = False
        
        self.connections[session_id] = {
            "websocket": websocket,
            "user_type": user_type,
            "connected_at": datetime.utcnow(),
            "messages_sent": 0,
            "messages_received": 0
        }
        
        return websocket
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Mock sending message through WebSocket."""
        if session_id in self.connections:
            connection = self.connections[session_id]
            connection["messages_sent"] += 1
            self.message_history.append({
                "session_id": session_id,
                "message": message,
                "timestamp": datetime.utcnow(),
                "direction": "outbound"
            })
    
    async def receive_message(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Mock receiving message through WebSocket."""
        if session_id in self.connections:
            connection = self.connections[session_id]
            connection["messages_received"] += 1
            
            # Mock AI response based on previous message
            last_message = None
            for msg in reversed(self.message_history):
                if msg["session_id"] == session_id and msg["direction"] == "inbound":
                    last_message = msg["message"]
                    break
            
            if last_message:
                content = last_message.get("content", "").lower()
                if "chest pain" in content or "heart" in content:
                    response = {
                        "type": "ai_response",
                        "content": "I'm concerned about your chest pain. This could be serious. I'm connecting you with a nurse immediately.",
                        "urgency": "high",
                        "red_flags": ["chest_pain"],
                        "confidence": 0.95
                    }
                elif "headache" in content:
                    response = {
                        "type": "ai_response", 
                        "content": "I understand you're experiencing a headache. Can you describe the pain characteristics and duration?",
                        "urgency": "low",
                        "confidence": 0.80
                    }
                else:
                    response = {
                        "type": "ai_response",
                        "content": "Thank you for that information. Could you provide more details about your symptoms?",
                        "urgency": "normal",
                        "confidence": 0.75
                    }
            else:
                response = {
                    "type": "ai_response",
                    "content": "Hello! I'm here to help assess your symptoms. Please describe what brings you in today.",
                    "urgency": "normal",
                    "confidence": 0.80
                }
            
            self.message_history.append({
                "session_id": session_id,
                "message": response,
                "timestamp": datetime.utcnow(),
                "direction": "outbound"
            })
            
            return response
        
        return None
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.connections),
            "total_messages": len(self.message_history),
            "active_sessions": list(self.connections.keys())
        }

class MockTrainingService:
    """Mock training service for testing."""
    
    def __init__(self):
        self.training_jobs = {}
        self.model_registry = MOCK_MODEL_DATA.copy()
        self.logger = structlog.get_logger("mock.training")
    
    async def start_training_job(
        self,
        model_config: Dict[str, Any],
        training_data: Dict[str, Any]
    ) -> str:
        """Mock starting training job."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        self.training_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "model_config": model_config,
            "training_data_info": {
                "samples": training_data.get("sample_count", 1000),
                "validation_split": 0.2,
                "features": training_data.get("features", [])
            },
            "started_at": datetime.utcnow(),
            "progress": 0.0,
            "metrics": {}
        }
        
        # Simulate training progress
        asyncio.create_task(self._simulate_training_progress(job_id))
        
        return job_id
    
    async def _simulate_training_progress(self, job_id: str):
        """Simulate training progress updates."""
        for progress in [0.25, 0.50, 0.75, 1.0]:
            await asyncio.sleep(1)  # Simulate training time
            if job_id in self.training_jobs:
                self.training_jobs[job_id]["progress"] = progress
                
                if progress == 1.0:
                    # Training completed
                    self.training_jobs[job_id]["status"] = "completed"
                    self.training_jobs[job_id]["completed_at"] = datetime.utcnow()
                    self.training_jobs[job_id]["metrics"] = {
                        "accuracy": 0.90 + (progress * 0.05),
                        "loss": 0.5 - (progress * 0.1),
                        "validation_accuracy": 0.88,
                        "training_time_minutes": 15
                    }
    
    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status."""
        if job_id not in self.training_jobs:
            raise ValueError(f"Training job {job_id} not found")
        
        return self.training_jobs[job_id]
    
    async def deploy_model(self, job_id: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock model deployment."""
        if job_id not in self.training_jobs:
            raise ValueError(f"Training job {job_id} not found")
        
        job = self.training_jobs[job_id]
        
        # Create new model entry
        model_id = f"{deployment_config.get('model_name', 'new_model')}_v{deployment_config.get('version', '1.0')}"
        
        self.model_registry[model_id] = {
            "model_id": model_id,
            "version": deployment_config.get("version", "1.0"),
            "type": deployment_config.get("model_type", "clinical_analysis"),
            "accuracy": job["metrics"].get("accuracy", 0.85),
            "training_data_size": job["training_data_info"]["samples"],
            "last_updated": datetime.utcnow().isoformat(),
            "deployment_status": "active",
            "serving_endpoint": f"/models/{model_id}/predict",
            "deployment_config": deployment_config
        }
        
        return {
            "model_id": model_id,
            "deployment_status": "deployed",
            "serving_endpoint": f"/models/{model_id}/predict"
        }

# Global mock instances
mock_websocket_manager = MockWebSocketManager()
mock_training_service = MockTrainingService()

# Pytest Fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_patient_data():
    """Provide mock patient data."""
    return MOCK_PATIENT_DATA

@pytest.fixture
def mock_nurse_data():
    """Provide mock nurse data."""
    return MOCK_NURSE_DATA

@pytest.fixture
def mock_model_data():
    """Provide mock model data."""
    return MOCK_MODEL_DATA

@pytest.fixture
async def mock_websocket_manager():
    """Provide mock WebSocket manager."""
    await mock_websocket_manager.connections.clear()
    mock_websocket_manager.message_history.clear()
    return mock_websocket_manager

@pytest.fixture
async def mock_training_service():
    """Provide mock training service."""
    # Reset training jobs
    mock_training_service.training_jobs.clear()
    # Reset model registry to base state
    mock_training_service.model_registry = MOCK_MODEL_DATA.copy()
    return mock_training_service

@pytest.fixture
async def http_client():
    """Create async HTTP client for testing."""
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=TEST_CONFIG["timeout"])
    ) as session:
        yield session

@pytest.fixture
async def websocket_session():
    """Create WebSocket session for testing."""
    async with websockets.connect(
        TEST_CONFIG["websocket_url"],
        timeout=TEST_CONFIG["timeout"]
    ) as websocket:
        yield websocket

@pytest.fixture
async def patient_session():
    """Create patient chat session."""
    session_id = f"test_session_{uuid.uuid4().hex[:8]}"
    return {
        "session_id": session_id,
        "user_type": "patient",
        "started_at": datetime.utcnow()
    }

@pytest.fixture
async def nurse_session():
    """Create nurse dashboard session."""
    session_id = f"nurse_session_{uuid.uuid4().hex[:8]}"
    return {
        "session_id": session_id,
        "user_type": "nurse",
        "nurse_id": "nurse_001",
        "started_at": datetime.utcnow()
    }

@pytest.fixture
def test_measurements():
    """Provide test measurement utilities."""
    class TestMeasurements:
        def __init__(self):
            self.measurements = {}
        
        def start_timer(self, name: str):
            self.measurements[name] = {
                "start_time": time.time(),
                "start_datetime": datetime.utcnow()
            }
        
        def end_timer(self, name: str) -> Dict[str, Any]:
            if name not in self.measurements:
                raise ValueError(f"Timer {name} not started")
            
            end_time = time.time()
            measurement = self.measurements[name]
            
            result = {
                "duration_seconds": end_time - measurement["start_time"],
                "start_time": measurement["start_datetime"],
                "end_time": datetime.utcnow()
            }
            
            del self.measurements[name]
            return result
        
        def record_metric(self, name: str, value: float, unit: str = None):
            if not hasattr(self, '_metrics'):
                self._metrics = {}
            
            self._metrics[name] = {
                "value": value,
                "unit": unit,
                "timestamp": datetime.utcnow()
            }
        
        def get_metrics(self) -> Dict[str, Any]:
            return getattr(self, '_metrics', {})
    
    return TestMeasurements()

@pytest.fixture
def assert_helpers():
    """Provide assertion helper utilities."""
    class AssertHelpers:
        @staticmethod
        def assert_response_time(measurement: Dict[str, Any], max_seconds: float):
            duration = measurement["duration_seconds"]
            assert duration <= max_seconds, f"Response time {duration:.2f}s exceeds {max_seconds}s"
        
        @staticmethod
        def assert_urgency_escalation(response: Dict[str, Any], min_urgency: str = "high"):
            urgency_levels = {"low": 1, "normal": 2, "moderate": 3, "high": 4, "emergency": 5}
            
            actual_urgency = response.get("urgency", "low")
            min_level = urgency_levels.get(min_urgency, 4)
            actual_level = urgency_levels.get(actual_urgency, 1)
            
            assert actual_level >= min_level, f"Urgency {actual_urgency} below required {min_urgency}"
        
        @staticmethod
        def assert_red_flags_detected(response: Dict[str, Any], required_flags: List[str] = None):
            if required_flags is None:
                required_flags = ["chest_pain", "shortness_of_breath"]
            
            red_flags = response.get("red_flags", [])
            
            if any(flag in str(response.get("content", "")).lower() for flag in required_flags):
                # If red flag mentioned in content, should be detected
                assert len(red_flags) > 0, "Red flags should be detected when emergency keywords present"
    
    return AssertHelpers()

# Pytest Configuration
pytest_plugins = ["pytest_asyncio"]

# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "reliability: mark test as reliability test"
    )
    config.addinivalue_line(
        "markers", "websocket: mark test as WebSocket test"
    )
    config.addinivalue_line(
        "markers", "training: mark test as training integration test"
    )
    config.addinivalue_line(
        "markers", "serving: mark test as model serving test"
    )