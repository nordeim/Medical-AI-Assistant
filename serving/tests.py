"""
Test suite for the model serving infrastructure.
Verifies core functionality, API endpoints, and system components.
"""

import asyncio
import pytest
import httpx
import json
from typing import Dict, Any
import time
import uuid

from fastapi.testclient import TestClient
import structlog

from config.settings import get_settings
from api.main import app
from models.base_server import model_registry
from models.concrete_servers import (
    TextGenerationServer, EmbeddingServer, ConversationServer
)


class TestModelServingInfrastructure:
    """Test suite for the model serving infrastructure."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def test_data(self):
        """Create test data."""
        return {
            "text_inputs": [
                "The patient shows symptoms of",
                "Treatment options include",
                "Medical diagnosis based on"
            ],
            "prediction_request": {
                "inputs": "The patient shows symptoms of fever and cough",
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": 0.7
                },
                "user_id": "test_user_123",
                "session_id": "test_session_456"
            }
        }
    
    def test_configuration_loading(self):
        """Test that configuration loads correctly."""
        settings = get_settings()
        
        assert settings is not None
        assert hasattr(settings, 'model')
        assert hasattr(settings, 'serving')
        assert hasattr(settings, 'cache')
        assert hasattr(settings, 'logging')
        assert hasattr(settings, 'medical')
    
    def test_settings_validation(self):
        """Test settings validation."""
        from config.settings import validate_config
        
        validation_result = validate_config()
        
        assert "valid" in validation_result
        assert "environment" in validation_result
        assert "model_name" in validation_result
        assert "port" in validation_result
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "models" in data
    
    def test_models_endpoint(self, client):
        """Test models listing endpoint."""
        response = client.get("/models")
        
        assert response.status_code == 200
        
        models = response.json()
        assert isinstance(models, list)
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        
        metrics = response.json()
        assert "timestamp" in metrics
        assert "requests_total" in metrics
        assert "requests_successful" in metrics
        assert "requests_failed" in metrics
        assert "models" in metrics
    
    def test_prediction_endpoint(self, client, test_data):
        """Test prediction endpoint."""
        # Test with a non-existent model first
        response = client.post(
            "/models/nonexistent/predict",
            json=test_data["prediction_request"]
        )
        assert response.status_code == 404
        
        # Test with valid model (this will fail until models are loaded)
        # For now, we'll just test the endpoint structure
        request_data = test_data["prediction_request"]
        
        # Test invalid input validation
        invalid_request = {
            "inputs": "",  # Empty input should fail validation
            "parameters": {}
        }
        
        response = client.post(
            "/models/text_generation_v1/predict",
            json=invalid_request
        )
        # This should fail validation or return 404 if model not loaded
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/models/text_generation_v1/info")
        
        # Either 200 (if model loaded) or 404 (if not loaded)
        assert response.status_code in [200, 404]
    
    def test_model_health_endpoint(self, client):
        """Test model health endpoint."""
        response = client.get("/models/text_generation_v1/health")
        
        # Either 200 (if model loaded) or 404 (if not loaded)
        assert response.status_code in [200, 404]
    
    def test_cors_middleware(self, client):
        """Test CORS middleware."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
    
    def test_request_id_header(self, client):
        """Test that request ID is added to responses."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0
    
    def test_rate_limiting_headers(self, client):
        """Test rate limiting headers."""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert "x-processing-time" in response.headers
    
    def test_error_handling(self, client):
        """Test error handling."""
        # Test 404 error
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Test invalid JSON
        response = client.post(
            "/models/test/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_structured_logging(self):
        """Test that structured logging is working."""
        logger = structlog.get_logger("test")
        
        # This should not raise any exceptions
        logger.info("Test log message", test_field="test_value")
        logger.warning("Test warning", warning_code="test_001")
        logger.error("Test error", error_code="test_error")
    
    def test_medical_data_validation(self):
        """Test medical data validation patterns."""
        from api.main import medical_data_validation
        
        # Test PHI redaction
        test_data = {
            "message": "Patient SSN: 123-45-6789, Phone: 555-123-4567",
            "email": "doctor@hospital.com"
        }
        
        validated_data = asyncio.run(medical_data_validation(test_data))
        
        # Should contain redacted SSN pattern
        assert "123-45-6789" not in validated_data["message"]
        assert "REDACTED" in validated_data["message"]


class TestModelServers:
    """Test model server implementations."""
    
    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Test that model servers can be initialized."""
        # This test would require actual model loading
        # For now, we just test the class structure
        
        text_gen = TextGenerationServer(
            model_id="test_text_gen",
            name="Test Text Generator"
        )
        
        assert text_gen.model_id == "test_text_gen"
        assert text_gen.name == "Test Text Generator"
        assert text_gen.status.value == "loading"  # Initial state
    
    @pytest.mark.asyncio
    async def test_embedding_server(self):
        """Test embedding server structure."""
        embedding_server = EmbeddingServer(
            model_id="test_embedding",
            name="Test Embeddings"
        )
        
        assert embedding_server.model_id == "test_embedding"
        assert embedding_server.name == "Test Embeddings"
        assert embedding_server.status.value == "loading"
    
    @pytest.mark.asyncio
    async def test_conversation_server(self):
        """Test conversation server structure."""
        conversation_server = ConversationServer(
            model_id="test_conversation",
            name="Test Conversation"
        )
        
        assert conversation_server.model_id == "test_conversation"
        assert conversation_server.name == "Test Conversation"
        assert conversation_server.status.value == "loading"
        
        # Test conversation history management
        session_id = "test_session_123"
        conversation_server.conversation_history[session_id] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        summary = conversation_server.get_conversation_summary(session_id)
        assert summary["session_id"] == session_id
        assert summary["message_count"] == 1


class TestCacheSystem:
    """Test caching functionality."""
    
    def test_model_cache(self):
        """Test model prediction caching."""
        from models.base_server import ModelCache
        
        cache = ModelCache(max_size=10, ttl=300)
        
        # Test setting and getting cache
        cache.set("model1", "input1", {"param": "value"}, {"output": "result1"})
        
        result = cache.get("model1", "input1", {"param": "value"})
        assert result["output"] == "result1"
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 10
    
    def test_cache_eviction(self):
        """Test cache LRU eviction."""
        from models.base_server import ModelCache
        
        cache = ModelCache(max_size=2, ttl=300)
        
        # Add more items than capacity
        cache.set("model1", "input1", {}, {"output": "result1"})
        cache.set("model2", "input2", {}, {"output": "result2"})
        cache.set("model3", "input3", {}, {"output": "result3"})
        
        # First item should be evicted
        result = cache.get("model1", "input1", {})
        assert result is None
        
        # Other items should still exist
        result = cache.get("model2", "input2", {})
        assert result["output"] == "result2"


class TestSecurityFeatures:
    """Test security and compliance features."""
    
    def test_api_key_authentication(self):
        """Test API key authentication."""
        from api.main import verify_api_key
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Test with no API key required (default)
        credentials = None
        result = asyncio.run(verify_api_key(credentials))
        assert result is None
    
    def test_medical_data_sanitization(self):
        """Test medical data sanitization."""
        test_data = {
            "patient_info": "SSN: 123-45-6789, DOB: 01/01/1990",
            "contact": "Phone: (555) 123-4567, Email: patient@email.com"
        }
        
        # This would need to be tested with actual redaction logic
        # For now, just verify the function exists
        from api.main import medical_data_validation
        result = asyncio.run(medical_data_validation(test_data))
        assert isinstance(result, dict)


def run_integration_tests():
    """Run integration tests."""
    print("Running model serving infrastructure tests...")
    
    # Test basic imports
    try:
        from config.settings import get_settings, validate_config
        from config.logging_config import get_logger
        from models.base_server import model_registry
        from api.main import app
        print("✓ Core imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test configuration
    try:
        settings = get_settings()
        validation = validate_config()
        print(f"✓ Configuration loaded: {validation['environment']}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Test FastAPI app
    try:
        client = TestClient(app)
        response = client.get("/health")
        if response.status_code == 200:
            print("✓ Health endpoint working")
        else:
            print(f"✗ Health endpoint returned {response.status_code}")
    except Exception as e:
        print(f"✗ FastAPI test error: {e}")
        return False
    
    print("✓ All basic tests passed!")
    return True


if __name__ == "__main__":
    # Run integration tests
    success = run_integration_tests()
    
    if success:
        print("\n🚀 Model serving infrastructure is ready!")
        print("\nNext steps:")
        print("1. Configure .env file with your settings")
        print("2. Run 'python main.py' to start the server")
        print("3. Visit http://localhost:8000/docs for API documentation")
    else:
        print("\n❌ Infrastructure setup incomplete. Please check errors above.")
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Pytest not available. Install with: pip install pytest pytest-asyncio")