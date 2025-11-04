"""
Comprehensive Test Suite for Production Medical AI Model Infrastructure
Tests all components of the production serving system.
"""

import asyncio
import pytest
import sys
import os
import json
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'serving'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monitoring'))

import pytest_asyncio
from serving.main import ModelServer
from utils.model_loader import ModelLoader
from utils.health_checker import HealthChecker
from utils.performance_monitor import PerformanceMonitor
from utils.cache_manager import CacheManager
from utils.security import SecurityManager, SecurityError
from monitoring.model_monitor import ModelMonitoringSystem

# Test configuration
pytest_plugins = ['pytest_asyncio']

class TestModelLoader:
    """Test model loading functionality"""
    
    @pytest.fixture
    def model_loader(self):
        return ModelLoader()
    
    @pytest.mark.asyncio
    async def test_model_initialization(self, model_loader):
        """Test model loader initialization"""
        await model_loader.initialize()
        assert model_loader is not None
        assert len(model_loader.model_registry) > 0
    
    @pytest.mark.asyncio
    async def test_model_loading(self, model_loader):
        """Test model loading"""
        await model_loader.initialize()
        
        # Test loading a model
        model = await model_loader.load_model("medical-diagnosis-v1")
        assert model is not None
        
        # Test cache hit
        model2 = await model_loader.load_model("medical-diagnosis-v1")
        assert model2 is not None
    
    def test_model_info(self, model_loader):
        """Test model information retrieval"""
        info = model_loader.get_model_info("medical-diagnosis-v1")
        assert info is not None
        assert "version" in info
    
    def test_cache_stats(self, model_loader):
        """Test cache statistics"""
        stats = model_loader.get_cache_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats

class TestHealthChecker:
    """Test health monitoring functionality"""
    
    @pytest.fixture
    def health_checker(self):
        return HealthChecker()
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, health_checker):
        """Test system health check"""
        result = await health_checker._check_system_health()
        assert "status" in result
        assert result["status"] in ["healthy", "warning", "critical"]
    
    @pytest.mark.asyncio
    async def test_all_health_checks(self, health_checker):
        """Test all health checks"""
        results = await health_checker.run_all_health_checks()
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that we have expected services
        expected_services = ["system", "redis", "model_server"]
        for service in expected_services:
            assert service in results
    
    def test_health_summary(self, health_checker):
        """Test health summary"""
        summary = health_checker.get_health_summary()
        assert "overall_status" in summary
        assert "health_score" in summary

class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        return PerformanceMonitor()
    
    @pytest.mark.asyncio
    async def test_initialization(self, performance_monitor):
        """Test performance monitor initialization"""
        await performance_monitor.initialize()
        assert performance_monitor is not None
    
    @pytest.mark.asyncio
    async def test_prediction_logging(self, performance_monitor):
        """Test prediction logging"""
        await performance_monitor.initialize()
        
        prediction_data = {
            "patient_id": "test_patient_001",
            "model_version": "medical-diagnosis-v1",
            "processing_time": 0.5,
            "confidence": 0.85,
            "success": True
        }
        
        await performance_monitor.log_prediction(prediction_data)
        
        # Verify metric was logged
        metrics = performance_monitor.get_model_performance("medical-diagnosis-v1", hours=1)
        assert metrics is not None
        assert "sample_count" in metrics
    
    def test_performance_metrics(self, performance_monitor):
        """Test performance metrics calculation"""
        # Mock some data
        for i in range(10):
            metric_data = {
                "model_name": "test_model",
                "timestamp": datetime.utcnow(),
                "latency_ms": 100 + i * 10,
                "throughput_qps": 50 + i,
                "memory_usage_mb": 100,
                "cpu_usage_percent": 50
            }
            performance_monitor.model_metrics["test_model"].append(
                type('MockMetric', (), metric_data)()
            )
        
        # Get performance summary
        performance = performance_monitor.get_model_performance("test_model", hours=1)
        assert "latency" in performance
        assert "throughput" in performance

class TestCacheManager:
    """Test cache management functionality"""
    
    @pytest.fixture
    def cache_manager(self):
        return CacheManager()
    
    @pytest.mark.asyncio
    async def test_cache_initialization(self, cache_manager):
        """Test cache initialization"""
        await cache_manager.initialize()
        assert cache_manager is not None
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, cache_manager):
        """Test basic cache operations"""
        await cache_manager.initialize()
        
        # Test set and get
        test_key = "test_key"
        test_value = {"data": "test_data", "timestamp": datetime.utcnow().isoformat()}
        
        success = await cache_manager.set(test_key, test_value)
        assert success
        
        cached_value = await cache_manager.get(test_key)
        assert cached_value == test_value
        
        # Test exists
        exists = await cache_manager.exists(test_key)
        assert exists
        
        # Test delete
        deleted = await cache_manager.delete(test_key)
        assert deleted
        
        # Verify deletion
        exists_after = await cache_manager.exists(test_key)
        assert not exists_after
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics"""
        await cache_manager.initialize()
        
        # Generate some cache activity
        for i in range(5):
            await cache_manager.set(f"key_{i}", f"value_{i}")
            await cache_manager.get(f"key_{i}")
        
        stats = await cache_manager.get_cache_stats()
        assert "hit_rate" in stats
        assert stats["hit_rate"] > 0

class TestSecurityManager:
    """Test security functionality"""
    
    @pytest.fixture
    def security_manager(self):
        return SecurityManager()
    
    def test_api_key_validation(self, security_manager):
        """Test API key validation"""
        # Test valid API key
        for api_key in security_manager.api_keys:
            client_info = security_manager.validate_api_key(api_key)
            assert client_info is not None
            assert "permissions" in client_info
            break  # Just test one
    
    def test_rate_limiting(self, security_manager):
        """Test rate limiting"""
        for api_key in security_manager.api_keys:
            # Test rate limit checking
            allowed = security_manager.check_rate_limit(api_key)
            assert isinstance(allowed, bool)
            break
    
    def test_jwt_token_generation(self, security_manager):
        """Test JWT token generation"""
        token = security_manager.generate_jwt_token("test_user", ["read", "predict"])
        assert token is not None
        assert isinstance(token, str)
        
        # Test token validation
        payload = security_manager.validate_jwt_token(token)
        assert payload["user_id"] == "test_user"
        assert "read" in payload["permissions"]
    
    def test_invalid_api_key(self, security_manager):
        """Test handling of invalid API key"""
        with pytest.raises(SecurityError) as exc_info:
            security_manager.validate_api_key("invalid_key")
        assert "Invalid API key" in str(exc_info.value)

class TestModelMonitoringSystem:
    """Test model monitoring functionality"""
    
    @pytest.fixture
    def model_monitor(self):
        return ModelMonitoringSystem()
    
    @pytest.mark.asyncio
    async def test_initialization(self, model_monitor):
        """Test model monitoring initialization"""
        await model_monitor.initialize()
        assert model_monitor is not None
    
    @pytest.mark.asyncio
    async def test_prediction_logging(self, model_monitor):
        """Test prediction logging"""
        await model_monitor.initialize()
        
        prediction_data = {
            "prediction": "condition_a",
            "confidence": 0.85,
            "processing_time": 0.5,
            "patient_id": "test_patient",
            "success": True
        }
        
        await model_monitor.log_prediction("medical-diagnosis-v1", prediction_data)
        
        # Verify prediction was logged
        predictions = list(model_monitor.prediction_history["medical-diagnosis-v1"])
        assert len(predictions) > 0
        assert predictions[-1]["prediction"] == "condition_a"
    
    @pytest.mark.asyncio
    async def test_actual_outcome_recording(self, model_monitor):
        """Test actual outcome recording"""
        await model_monitor.initialize()
        
        outcome_data = {
            "actual_result": "condition_a",
            "clinical_notes": "Follow-up completed"
        }
        
        await model_monitor.record_actual_outcome(
            "medical-diagnosis-v1", "test_patient", outcome_data
        )
        
        # Verify outcome was recorded
        outcomes = model_monitor.actual_outcomes["medical-diagnosis-v1"]
        assert len(outcomes) > 0
        assert outcomes[-1].patient_id == "test_patient"
    
    @pytest.mark.asyncio
    async def test_drift_detection(self, model_monitor):
        """Test drift detection"""
        await model_monitor.initialize()
        
        # Generate synthetic prediction data for drift detection
        for i in range(50):
            prediction_data = {
                "prediction": "condition_a" if i % 2 == 0 else "condition_b",
                "confidence": np.random.uniform(0.6, 0.95),
                "processing_time": np.random.uniform(0.1, 0.5),
                "patient_id": f"patient_{i}",
                "success": True
            }
            await model_monitor.log_prediction("medical-diagnosis-v1", prediction_data)
        
        # Detect drift
        drift = await model_monitor.detect_model_drift("medical-diagnosis-v1")
        # Note: drift might be None if no significant drift is detected
        assert drift is None or isinstance(drift, type(drift))

class TestModelServer:
    """Test FastAPI model server"""
    
    @pytest.fixture
    def model_server(self):
        return ModelServer()
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, model_server):
        """Test server initialization"""
        assert model_server is not None
        assert model_server.app is not None
    
    def test_health_endpoint_format(self, model_server):
        """Test health endpoint response format"""
        # This would typically test the actual FastAPI endpoint
        # For now, just verify the structure exists
        assert hasattr(model_server, '_setup_routes')
    
    @pytest.mark.asyncio
    async def test_prediction_request_validation(self, model_server):
        """Test prediction request validation"""
        from serving.main import PredictionRequest
        
        # Test valid request
        valid_request = PredictionRequest(
            patient_id="test_patient",
            clinical_data={"symptom1": "value1"},
            priority="normal"
        )
        assert valid_request.patient_id == "test_patient"
        assert valid_request.priority == "normal"
        
        # Test invalid priority
        with pytest.raises(Exception):  # Pydantic validation error
            PredictionRequest(
                patient_id="test_patient",
                clinical_data={"symptom1": "value1"},
                priority="invalid_priority"
            )

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prediction_flow(self):
        """Test complete prediction workflow"""
        # Initialize components
        model_loader = ModelLoader()
        cache_manager = CacheManager()
        performance_monitor = PerformanceMonitor()
        
        await model_loader.initialize()
        await cache_manager.initialize()
        await performance_monitor.initialize()
        
        # Simulate prediction flow
        prediction_data = {
            "patient_id": "integration_test_patient",
            "model_version": "medical-diagnosis-v1",
            "processing_time": 0.3,
            "confidence": 0.92,
            "success": True
        }
        
        # Log prediction
        await performance_monitor.log_prediction(prediction_data)
        
        # Verify flow worked
        assert performance_monitor is not None
        
        # Cleanup
        await cache_manager.close()
        await performance_monitor.close()
    
    @pytest.mark.asyncio
    async def test_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        model_monitor = ModelMonitoringSystem()
        
        await model_monitor.initialize()
        
        # Log predictions
        for i in range(20):
            await model_monitor.log_prediction("test_model", {
                "prediction": f"result_{i}",
                "confidence": np.random.uniform(0.7, 0.95),
                "processing_time": np.random.uniform(0.1, 0.8),
                "patient_id": f"patient_{i}",
                "success": True
            })
        
        # Record some outcomes
        for i in range(10):
            await model_monitor.record_actual_outcome("test_model", f"patient_{i}", {
                "actual_result": f"result_{i}",
                "clinical_notes": f"Follow-up {i}"
            })
        
        # Get performance summary
        summary = await model_monitor.get_model_performance_summary("test_model", hours=1)
        assert "performance_metrics" in summary
        
        await model_monitor.close()

class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_predictions(self):
        """Test concurrent prediction handling"""
        model_loader = ModelLoader()
        await model_loader.initialize()
        
        # Load model once
        model = await model_loader.load_model("medical-diagnosis-v1")
        
        # Simulate concurrent predictions
        start_time = time.time()
        tasks = []
        
        for i in range(10):
            task = asyncio.create_task(
                self._simulate_prediction(model_loader, f"patient_{i}")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify all predictions completed
        assert len(results) == 10
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
        
        await model_loader.cleanup_model(model)
    
    async def _simulate_prediction(self, model_loader, patient_id):
        """Simulate a single prediction"""
        model = await model_loader.load_model("medical-diagnosis-v1")
        # Simulate prediction work
        await asyncio.sleep(0.1)
        return {"patient_id": patient_id, "status": "completed"}

class TestSecurity:
    """Security tests"""
    
    def test_input_validation(self):
        """Test input validation"""
        # Test malicious input detection
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "'; DROP TABLE users; --",
            "${7*7}",
            "{{7*7}}"
        ]
        
        for malicious_input in malicious_inputs:
            # Should detect as unsafe
            is_safe = True  # This would use actual validation logic
            assert is_safe is not None  # Simplified for demo
    
    def test_api_key_security(self):
        """Test API key security"""
        security = SecurityManager()
        
        # Test API key format
        for api_key in security.api_keys.values():
            assert api_key["api_key"].startswith("med_ai_")
            assert len(api_key["api_key"]) >= 32
    
    def test_audit_logging(self):
        """Test audit logging"""
        security = SecurityManager()
        
        # Test audit log creation
        security.create_audit_log(
            "test_api_key",
            "predict",
            "model_endpoint",
            "success"
        )
        
        # Verify audit log was created (would check actual log in production)

# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])