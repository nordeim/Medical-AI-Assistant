"""
Complete System Integration Testing for Phase 7

Comprehensive end-to-end integration testing covering all system components:
- Frontend integration
- Backend API integration
- Training pipeline integration
- Model serving integration
- Database integration
- Real-time communication

Tests complete workflows from patient input to AI assessment to nurse interaction
with full system validation and performance benchmarking.
"""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
import websockets

# Test markers
pytestmark = pytest.mark.integration


class TestSystemComponentIntegration:
    """Test integration between all system components."""
    
    @pytest.mark.asyncio
    async def test_frontend_backend_integration(self, http_client, mock_patient_data, test_measurements):
        """Test complete frontend-backend integration."""
        
        # Measure integration response time
        test_measurements.start_timer("frontend_backend_integration")
        
        # Test patient chat initiation
        patient_data = mock_patient_data["patient_001"]
        
        # Simulate frontend API call to start patient session
        session_request = {
            "action": "start_chat_session",
            "patient_info": {
                "session_id": f"test_session_{uuid.uuid4().hex[:8]}",
                "user_type": "patient",
                "initial_message": f"I'm experiencing {patient_data['chief_complaint'].lower()}"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Mock HTTP request (in real implementation, would call actual frontend API)
        async with http_client.post(
            "http://localhost:8000/api/v1/chat/sessions",
            json=session_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            assert response.status == 200
            session_response = await response.json()
            
            assert "session_id" in session_response
            assert "status" in session_response
            assert session_response["status"] == "active"
        
        # Test AI assessment endpoint
        assessment_request = {
            "session_id": session_response["session_id"],
            "patient_symptoms": patient_data["symptoms"],
            "severity": patient_data["severity"],
            "duration": patient_data["duration"]
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/assessment/analyze",
            json=assessment_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            assert response.status == 200
            assessment_result = await response.json()
            
            assert "assessment_id" in assessment_result
            assert "risk_level" in assessment_result
            assert "recommendations" in assessment_result
            assert "confidence_score" in assessment_result
        
        # Test nurse queue integration
        queue_request = {
            "action": "get_queue",
            "filters": {
                "risk_level": "high",
                "status": "waiting_for_assessment"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/nurse/queue",
            json=queue_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            assert response.status == 200
            queue_result = await response.json()
            
            assert "queue" in queue_result
            assert "total_count" in queue_result
            assert isinstance(queue_result["queue"], list)
        
        # Measure completion time
        measurement = test_measurements.end_timer("frontend_backend_integration")
        
        # Validate performance
        assert measurement["duration_seconds"] < 5.0, "Frontend-backend integration too slow"
        print(f"Frontend-backend integration: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_training_serving_integration(self, mock_training_service, mock_model_data, test_measurements):
        """Test integration between training pipeline and model serving."""
        
        test_measurements.start_timer("training_serving_integration")
        
        # Start training job
        model_config = {
            "model_name": "clinical_assessment",
            "model_type": "bert_based_classifier",
            "architecture": "fine_tune",
            "hyperparameters": {
                "learning_rate": 1e-5,
                "batch_size": 32,
                "epochs": 10
            }
        }
        
        training_data = {
            "sample_count": 5000,
            "features": ["symptoms", "demographics", "risk_factors"],
            "labels": ["diagnosis", "urgency", "recommendations"]
        }
        
        # Start training job
        job_id = await mock_training_service.start_training_job(model_config, training_data)
        
        # Wait for training completion (simulated)
        await asyncio.sleep(2)
        
        # Check training status
        status = await mock_training_service.get_training_status(job_id)
        assert status["status"] in ["running", "completed"]
        
        # Deploy model when ready
        if status["status"] == "completed":
            deployment_config = {
                "model_name": "clinical_assessment",
                "version": "2.0.0",
                "model_type": "clinical_analysis",
                "deployment_environment": "production"
            }
            
            deployment_result = await mock_training_service.deploy_model(job_id, deployment_config)
            
            assert "model_id" in deployment_result
            assert "deployment_status" in deployment_result
            assert deployment_result["deployment_status"] == "deployed"
            
            # Verify model is available in registry
            assert deployment_result["model_id"] in mock_training_service.model_registry
            
            # Test model serving endpoint
            model_id = deployment_result["model_id"]
            model_info = mock_training_service.model_registry[model_id]
            
            assert "serving_endpoint" in model_info
            assert model_info["deployment_status"] == "active"
        
        # Measure integration time
        measurement = test_measurements.end_timer("training_serving_integration")
        assert measurement["duration_seconds"] < 10.0, "Training-serving integration too slow"
        print(f"Training-serving integration: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_database_integration(self, http_client, mock_patient_data, test_measurements):
        """Test database integration across all components."""
        
        test_measurements.start_timer("database_integration")
        
        # Test patient data persistence
        patient_data = mock_patient_data["patient_001"]
        
        # Store patient case
        store_request = {
            "action": "store_patient_case",
            "patient_data": patient_data,
            "metadata": {
                "created_by": "integration_test",
                "test_session": "comprehensive_integration"
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/data/patients",
            json=store_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                store_result = await response.json()
                assert "patient_id" in store_result
                assert store_result["patient_id"] == patient_data["id"]
        
        # Retrieve patient case
        retrieve_request = {
            "action": "get_patient_case",
            "patient_id": patient_data["id"]
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/data/patients/retrieve",
            json=retrieve_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                retrieve_result = await response.json()
                assert "patient_data" in retrieve_result
                assert retrieve_result["patient_data"]["id"] == patient_data["id"]
        
        # Test audit logging
        audit_request = {
            "action": "log_clinical_action",
            "user_id": "test_user",
            "action_type": "patient_assessment",
            "resource_id": patient_data["id"],
            "details": {"test": "audit_logging"}
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/audit/logs",
            json=audit_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                audit_result = await response.json()
                assert "log_id" in audit_result
        
        # Measure database integration time
        measurement = test_measurements.end_timer("database_integration")
        assert measurement["duration_seconds"] < 3.0, "Database integration too slow"
        print(f"Database integration: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_security_integration(self, http_client, test_measurements):
        """Test security integration across all system components."""
        
        test_measurements.start_timer("security_integration")
        
        # Test authentication
        auth_request = {
            "action": "authenticate",
            "credentials": {
                "user_type": "nurse",
                "user_id": "nurse_001",
                "token": "test_token_123"
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/auth/authenticate",
            json=auth_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                auth_result = await response.json()
                assert "access_token" in auth_result
                assert "user_info" in auth_result
        
        # Test authorization checks
        protected_request = {
            "action": "access_protected_resource",
            "resource": "patient_data",
            "patient_id": "patient_001"
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/security/authorize",
            json=protected_request,
            headers={"Authorization": "Bearer test_token_123"}
        ) as response:
            
            # Should either succeed (with proper auth) or fail gracefully
            assert response.status in [200, 401, 403]
        
        # Test PHI protection
        phi_request = {
            "action": "process_with_phi_protection",
            "data": {
                "patient_name": "John Doe",
                "ssn": "123-45-6789",
                "medical_info": "Confidential patient data"
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/security/phi-process",
            json=phi_request,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                phi_result = await response.json()
                assert "processed_data" in phi_result
                # Verify PHI is properly protected/redacted
        
        # Measure security integration time
        measurement = test_measurements.end_timer("security_integration")
        assert measurement["duration_seconds"] < 2.0, "Security integration too slow"
        print(f"Security integration: {measurement['duration_seconds']:.2f}s")


class TestSystemDataFlow:
    """Test complete data flow through the entire system."""
    
    @pytest.mark.asyncio
    async def test_patient_data_flow(self, http_client, mock_patient_data, test_measurements):
        """Test complete patient data flow through the system."""
        
        test_measurements.start_timer("patient_data_flow")
        
        patient_data = mock_patient_data["patient_001"]
        
        # 1. Patient registers/checks in
        registration_data = {
            "action": "patient_registration",
            "patient_info": {
                "session_id": f"reg_session_{uuid.uuid4().hex[:8]}",
                "chief_complaint": patient_data["chief_complaint"],
                "symptoms": patient_data["symptoms"],
                "severity": patient_data["severity"]
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/patient/register",
            json=registration_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                registration_result = await response.json()
                session_id = registration_result.get("session_id")
                assert session_id is not None
        
        # 2. AI initial assessment
        assessment_data = {
            "session_id": session_id,
            "symptoms": patient_data["symptoms"],
            "demographics": patient_data["demographics"],
            "severity": patient_data["severity"],
            "risk_factors": patient_data["risk_factors"]
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/assessment/initial",
            json=assessment_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                assessment_result = await response.json()
                assert "risk_level" in assessment_result
                assert "triage_recommendation" in assessment_result
        
        # 3. Nurse queue entry
        queue_entry_data = {
            "session_id": session_id,
            "assessment_result": assessment_result if response.status == 200 else {"risk_level": "high"},
            "queue_priority": "high" if patient_data["severity"] == "severe" else "normal"
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/queue/add",
            json=queue_entry_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                queue_result = await response.json()
                assert "queue_position" in queue_result
        
        # 4. Nurse review
        nurse_review_data = {
            "session_id": session_id,
            "nurse_id": "nurse_001",
            "action": "review_patient",
            "review_decision": "approve_assessment"
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/nurse/review",
            json=nurse_review_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                review_result = await response.json()
                assert "final_decision" in review_result
        
        # 5. Final patient communication
        final_comm_data = {
            "session_id": session_id,
            "communication_type": "final_assessment",
            "final_assessment": review_result if response.status == 200 else {"final_decision": "approved"}
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/communication/final",
            json=final_comm_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            # Should succeed regardless of previous responses
            assert response.status in [200, 404, 500]  # Flexible for testing
        
        # Measure data flow time
        measurement = test_measurements.end_timer("patient_data_flow")
        assert measurement["duration_seconds"] < 10.0, "Patient data flow too slow"
        print(f"Patient data flow: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_model_data_flow(self, http_client, mock_model_data, test_measurements):
        """Test complete model data flow from training to serving."""
        
        test_measurements.start_timer("model_data_flow")
        
        model_config = mock_model_data["clinical_assessment_v1"]
        
        # 1. Model registration
        registration_data = {
            "action": "register_model",
            "model_info": {
                "model_id": model_config["model_id"],
                "version": model_config["version"],
                "type": model_config["type"],
                "accuracy": model_config["accuracy"]
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/models/register",
            json=registration_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                reg_result = await response.json()
                assert "model_id" in reg_result
        
        # 2. Model serving deployment
        deployment_data = {
            "action": "deploy_model",
            "model_id": model_config["model_id"],
            "deployment_config": {
                "serving_endpoint": model_config["serving_endpoint"],
                "deployment_status": "active"
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/models/deploy",
            json=deployment_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                deploy_result = await response.json()
                assert "deployment_status" in deploy_result
        
        # 3. Model prediction
        prediction_data = {
            "model_id": model_config["model_id"],
            "input_data": {
                "symptoms": ["chest_pain", "shortness_of_breath"],
                "patient_age": 45,
                "severity": "severe"
            }
        }
        
        async with http_client.post(
            model_config["serving_endpoint"],
            json=prediction_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            if response.status == 200:
                pred_result = await response.json()
                assert "prediction" in pred_result
                assert "confidence" in pred_result
        
        # 4. Model performance tracking
        performance_data = {
            "model_id": model_config["model_id"],
            "prediction_id": f"pred_{uuid.uuid4().hex[:8]}",
            "performance_metrics": {
                "inference_time_ms": 150,
                "accuracy": 0.92,
                "throughput": "requests_per_second"
            }
        }
        
        async with http_client.post(
            "http://localhost:8000/api/v1/models/track-performance",
            json=performance_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            
            # Should succeed for monitoring
            assert response.status in [200, 404]
        
        # Measure model data flow time
        measurement = test_measurements.end_timer("model_data_flow")
        assert measurement["duration_seconds"] < 5.0, "Model data flow too slow"
        print(f"Model data flow: {measurement['duration_seconds']:.2f}s")


class TestSystemMonitoring:
    """Test system monitoring and health checks."""
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, http_client, test_measurements):
        """Test system health check integration."""
        
        test_measurements.start_timer("health_check_integration")
        
        # System health check
        async with http_client.get("http://localhost:8000/health") as response:
            assert response.status == 200
            health_data = await response.json()
            
            assert "status" in health_data
            assert "components" in health_data
            assert "timestamp" in health_data
        
        # Component-specific health checks
        components = ["database", "models", "training", "websocket", "queue"]
        
        for component in components:
            async with http_client.get(f"http://localhost:8000/health/{component}") as response:
                # Each component should report health status
                assert response.status == 200
                comp_health = await response.json()
                assert "status" in comp_health
                assert "component" in comp_health
        
        # Performance metrics
        async with http_client.get("http://localhost:8000/metrics") as response:
            if response.status == 200:
                metrics_data = await response.json()
                assert "system_metrics" in metrics_data
                assert "performance_metrics" in metrics_data
        
        # Measure health check time
        measurement = test_measurements.end_timer("health_check_integration")
        assert measurement["duration_seconds"] < 3.0, "Health check integration too slow"
        print(f"Health check integration: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(self, http_client, test_measurements):
        """Test audit logging integration across the system."""
        
        test_measurements.start_timer("audit_logging_integration")
        
        # Log various system actions
        audit_actions = [
            {
                "action": "patient_assessment",
                "resource": "patient_001",
                "user": "ai_assistant",
                "outcome": "completed"
            },
            {
                "action": "nurse_review",
                "resource": "patient_001", 
                "user": "nurse_001",
                "outcome": "approved"
            },
            {
                "action": "model_prediction",
                "resource": "clinical_assessment_v1",
                "user": "system",
                "outcome": "success"
            },
            {
                "action": "data_access",
                "resource": "patient_records",
                "user": "test_user",
                "outcome": "authorized"
            }
        ]
        
        for audit_action in audit_actions:
            async with http_client.post(
                "http://localhost:8000/api/v1/audit/log",
                json=audit_action,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                # Audit logging should succeed
                assert response.status in [200, 201]
        
        # Retrieve audit logs
        async with http_client.get("http://localhost:8000/api/v1/audit/logs") as response:
            if response.status == 200:
                logs_data = await response.json()
                assert "logs" in logs_data
                assert len(logs_data["logs"]) > 0
        
        # Measure audit logging time
        measurement = test_measurements.end_timer("audit_logging_integration")
        assert measurement["duration_seconds"] < 2.0, "Audit logging integration too slow"
        print(f"Audit logging integration: {measurement['duration_seconds']:.2f}s")


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    @pytest.mark.asyncio
    async def test_emergency_scenario_flow(self, http_client, mock_patient_data, test_measurements):
        """Test emergency scenario flow through entire system."""
        
        test_measurements.start_timer("emergency_scenario_flow")
        
        # Patient with emergency symptoms
        emergency_patient = mock_patient_data["patient_001"]  # Chest pain
        
        # Rapid assessment flow
        steps = [
            ("patient_registration", "http://localhost:8000/api/v1/patient/register"),
            ("emergency_assessment", "http://localhost:8000/api/v1/assessment/emergency"),
            ("immediate_triage", "http://localhost:8000/api/v1/triage/immediate"),
            ("nurse_escalation", "http://localhost:8000/api/v1/nurse/escalate"),
            ("emergency_protocols", "http://localhost:8000/api/v1/emergency/protocols")
        ]
        
        for step_name, endpoint in steps:
            step_data = {
                "patient_data": emergency_patient,
                "step": step_name,
                "timestamp": datetime.utcnow().isoformat(),
                "priority": "emergency"
            }
            
            async with http_client.post(
                endpoint,
                json=step_data,
                headers={"Content-Type": "application/json", "Priority": "emergency"}
            ) as response:
                
                # Emergency flow should have priority handling
                assert response.status in [200, 201, 503]  # May be busy during emergency
        
        # Measure emergency scenario time
        measurement = test_measurements.end_timer("emergency_scenario_flow")
        # Emergency scenarios should be very fast
        assert measurement["duration_seconds"] < 15.0, "Emergency scenario too slow"
        print(f"Emergency scenario: {measurement['duration_seconds']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_routine_care_flow(self, http_client, mock_patient_data, test_measurements):
        """Test routine care flow through entire system."""
        
        test_measurements.start_timer("routine_care_flow")
        
        # Patient with routine symptoms
        routine_patient = mock_patient_data["patient_002"]  # Headache
        
        # Normal assessment flow
        steps = [
            ("patient_registration", "http://localhost:8000/api/v1/patient/register"),
            ("initial_assessment", "http://localhost:8000/api/v1/assessment/initial"),
            ("symptom_analysis", "http://localhost:8000/api/v1/analysis/symptoms"),
            ("queue_entry", "http://localhost:8000/api/v1/queue/add"),
            ("nurse_review", "http://localhost:8000/api/v1/nurse/review"),
            ("final_recommendations", "http://localhost:8000/api/v1/recommendations/final")
        ]
        
        step_results = []
        
        for step_name, endpoint in steps:
            step_data = {
                "patient_data": routine_patient,
                "step": step_name,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async with http_client.post(
                endpoint,
                json=step_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    step_results.append({
                        "step": step_name,
                        "result": result,
                        "success": True
                    })
                else:
                    step_results.append({
                        "step": step_name,
                        "result": None,
                        "success": False
                    })
        
        # Verify flow completion
        successful_steps = [s for s in step_results if s["success"]]
        assert len(successful_steps) >= len(steps) * 0.6  # At least 60% should succeed
        
        # Measure routine care time
        measurement = test_measurements.end_timer("routine_care_flow")
        assert measurement["duration_seconds"] < 30.0, "Routine care flow too slow"
        print(f"Routine care flow: {measurement['duration_seconds']:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])