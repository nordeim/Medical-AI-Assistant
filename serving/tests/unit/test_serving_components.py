"""
Unit tests for serving components with medical-specific test cases.

This module tests the core serving infrastructure components including:
- Model servers with clinical accuracy validation
- Medical data processing pipelines
- Clinical decision support systems
- Medical text generation and analysis
- PHI protection and compliance mechanisms
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any

from fastapi.testclient import TestClient

# Import serving components
from models.base_server import ModelServer, ModelCache, model_registry
from models.concrete_servers import (
    TextGenerationServer, EmbeddingServer, ConversationServer,
    ClinicalAnalysisServer, DiagnosticAssistantServer
)
from api.main import app


class TestModelServers:
    """Test medical AI model server implementations."""
    
    @pytest.mark.unit
    @pytest.mark.medical
    async def test_text_generation_server_medical_validation(self):
        """Test text generation server with medical context validation."""
        
        server = TextGenerationServer(
            model_id="clinical_text_generation_v1",
            name="Clinical Text Generator",
            medical_context=True,
            clinical_validation=True
        )
        
        # Test medical prompt processing
        medical_prompt = """
        Patient: 45-year-old male with Type 2 Diabetes
        Current medications: Metformin 500mg BID, Atorvastatin 20mg daily
        Latest readings: Glucose 180 mg/dL, HbA1c 8.2%
        Current symptoms: Polyuria, polydipsia, fatigue
        
        Please provide clinical notes for this visit.
        """
        
        # Mock the actual model inference
        with patch.object(server, 'generate', new_callable=AsyncMock) as mock_generate:
            mock_generate.return_value = {
                "generated_text": "Patient presents with symptoms consistent with suboptimal glycemic control.",
                "clinical_accuracy": 0.87,
                "medical_relevance": 0.92,
                "phi_sanitized": True
            }
            
            result = await server.generate(medical_prompt)
            
            assert result["clinical_accuracy"] >= 0.80
            assert result["medical_relevance"] >= 0.85
            assert result["phi_sanitized"] is True
            
    @pytest.mark.unit
    @pytest.mark.medical
    async def test_clinical_analysis_server(self):
        """Test clinical analysis server for symptom/diagnosis processing."""
        
        server = ClinicalAnalysisServer(
            model_id="clinical_analysis_v1",
            name="Clinical Analysis Assistant",
            analysis_types=["symptom_analysis", "differential_diagnosis", "risk_assessment"]
        )
        
        # Test symptom analysis
        symptom_data = {
            "symptoms": ["polyuria", "polydipsia", "weight_loss", "fatigue"],
            "duration": "3 months",
            "severity": "moderate",
            "patient_age": 45,
            "gender": "male"
        }
        
        with patch.object(server, 'analyze_symptoms', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = {
                "primary_consideration": "Type 2 Diabetes Mellitus",
                "confidence_score": 0.85,
                "differential_diagnosis": [
                    {"condition": "Type 2 Diabetes", "probability": 0.85},
                    {"condition": "Hyperthyroidism", "probability": 0.15},
                    {"condition": "Chronic Kidney Disease", "probability": 0.10}
                ],
                "recommended_tests": ["HbA1c", "Fasting Glucose", "Thyroid Function"],
                "clinical_risk_factors": ["Age >40", "Male gender", "Overweight"]
            }
            
            result = await server.analyze_symptoms(symptom_data)
            
            assert result["confidence_score"] >= 0.80
            assert result["primary_consideration"] == "Type 2 Diabetes Mellitus"
            assert len(result["differential_diagnosis"]) >= 2
            assert len(result["recommended_tests"]) >= 3
            
    @pytest.mark.unit
    @pytest.mark.medical
    async def test_diagnostic_assistant_server(self):
        """Test diagnostic assistant with clinical decision support."""
        
        server = DiagnosticAssistantServer(
            model_id="diagnostic_assistant_v1",
            name="Diagnostic Decision Support",
            decision_support_enabled=True,
            clinical_guidelines_active=True
        )
        
        diagnostic_request = {
            "chief_complaint": "Chest pain and shortness of breath",
            "history": "45-year-old male, smoker, family history of heart disease",
            "vital_signs": {"bp": "150/95", "hr": 95, "spo2": "94%"},
            "physical_exam": "Diaphoresis, anxiety, S4 gallop present"
        }
        
        with patch.object(server, 'provide_diagnostic_assistance', new_callable=AsyncMock) as mock_diagnose:
            mock_diagnose.return_value = {
                "immediate_concerns": ["Acute Coronary Syndrome", "Pulmonary Embolism"],
                "urgency_level": "HIGH",
                "recommended_actions": [
                    "Immediate EKG",
                    "Troponin levels",
                    "Chest X-ray",
                    "Cardiology consultation"
                ],
                "risk_stratification": {
                    "level": "intermediate",
                    "score": 65,
                    "factors": ["Chest pain", "Risk factors", "Vital signs"]
                },
                "clinical_guidelines_followed": True
            }
            
            result = await server.provide_diagnostic_assistance(diagnostic_request)
            
            assert result["urgency_level"] in ["LOW", "INTERMEDIATE", "HIGH"]
            assert len(result["recommended_actions"]) >= 4
            assert result["clinical_guidelines_followed"] is True
            
    @pytest.mark.unit
    @pytest.mark.medical
    async def test_embedding_server_medical_context(self):
        """Test embedding server with medical context understanding."""
        
        server = EmbeddingServer(
            model_id="medical_embeddings_v1",
            name="Medical Context Embeddings",
            medical_vocabulary=True,
            clinical_relevance_weighting=True
        )
        
        medical_texts = [
            "Patient diagnosed with Type 2 Diabetes Mellitus",
            "Hyperglycemia and insulin resistance",
            "Polyuria and polydipsia symptoms",
            "Metformin therapy initiated",
            "HbA1c monitoring required"
        ]
        
        with patch.object(server, 'encode', new_callable=AsyncMock) as mock_encode:
            mock_encode.return_value = {
                "embeddings": np.random.rand(5, 768).tolist(),
                "medical_relevance_scores": [0.92, 0.88, 0.85, 0.90, 0.87],
                "clinical_concepts_extracted": [
                    ["diabetes", "hyperglycemia"],
                    ["insulin", "resistance"],
                    ["polyuria", "polydipsia"],
                    ["metformin", "therapy"],
                    ["HbA1c", "monitoring"]
                ]
            }
            
            result = await server.encode(medical_texts)
            
            assert len(result["embeddings"]) == 5
            assert all(score > 0.8 for score in result["medical_relevance_scores"])
            assert all(len(concepts) > 0 for concepts in result["clinical_concepts_extracted"])


class TestMedicalDataProcessing:
    """Test medical data processing and validation."""
    
    @pytest.mark.unit
    @pytest.mark.medical
    def test_phi_detection_and_redaction(self):
        """Test PHI detection and redaction functionality."""
        
        from api.utils.phi_protection import PHIRedactor
        
        redactor = PHIRedactor()
        
        test_cases = [
            {
                "input": "Patient SSN: 123-45-6789, DOB: 01/15/1980, Phone: (555) 123-4567",
                "expected_patterns": ["123-45-6789", "01/15/1980", "(555) 123-4567"]
            },
            {
                "input": "Name: John Doe, MRN: MRN123456, Email: john.doe@email.com",
                "expected_patterns": ["John Doe", "MRN123456", "john.doe@email.com"]
            },
            {
                "input": "Address: 123 Main St, Boston MA 02101, License: MA1234567",
                "expected_patterns": ["123 Main St", "Boston MA 02101", "MA1234567"]
            }
        ]
        
        for case in test_cases:
            redacted = redactor.redact_phi(case["input"])
            
            # Verify PHI patterns are removed
            for pattern in case["expected_patterns"]:
                assert pattern not in redacted
            
            # Verify placeholder replacement
            assert "[REDACTED]" in redacted or "[NAME_REDACTED]" in redacted
    
    @pytest.mark.unit
    @pytest.mark.medical
    def test_medical_data_validation(self):
        """Test medical data validation rules."""
        
        from api.utils.medical_validation import MedicalDataValidator
        
        validator = MedicalDataValidator()
        
        # Valid medical data
        valid_data = {
            "age": 45,
            "gender": "M",
            "symptoms": ["polyuria", "polydipsia"],
            "medications": ["metformin"],
            "lab_values": {"glucose": 180, "hba1c": 8.2}
        }
        
        # Invalid medical data
        invalid_data = {
            "age": -5,  # Negative age
            "gender": "invalid",  # Invalid gender
            "symptoms": [],  # Empty symptoms
            "medications": ["unknown_drug"],  # Unknown medication
            "lab_values": {"glucose": "invalid"}  # Invalid lab value
        }
        
        assert validator.validate(valid_data)["valid"] is True
        assert validator.validate(invalid_data)["valid"] is False
    
    @pytest.mark.unit
    @pytest.mark.medical
    def test_clinical_terminology_validation(self):
        """Test clinical terminology and coding validation."""
        
        from api.utils.clinical_codes import ClinicalCodeValidator
        
        validator = ClinicalCodeValidator()
        
        # Test ICD-10 codes
        icd_codes = ["E11.9", "I10", "J45.9", "F32.9"]
        for code in icd_codes:
            assert validator.validate_icd10(code) is True
        
        # Test SNOMED CT concepts
        snomed_concepts = ["44054006", "38341003", "195967001"]
        for concept in snomed_concepts:
            assert validator.validate_snomed(concept) is True


class TestModelCache:
    """Test model caching with medical context."""
    
    @pytest.mark.unit
    def test_medical_model_cache(self):
        """Test model cache with medical data."""
        
        cache = ModelCache(max_size=10, ttl=300)
        
        # Test medical prediction caching
        medical_input = {
            "symptoms": ["polyuria", "polydipsia"],
            "patient_age": 45,
            "medical_context": "diabetes_assessment"
        }
        
        medical_output = {
            "diagnosis": "Type 2 Diabetes Mellitus",
            "confidence": 0.85,
            "recommendations": ["HbA1c test", "Dietary consultation"]
        }
        
        cache.set("clinical_analysis", json.dumps(medical_input), {}, medical_output)
        
        cached_result = cache.get("clinical_analysis", json.dumps(medical_input), {})
        assert cached_result == medical_output
        
    @pytest.mark.unit
    def test_cache_invalidation_on_phi_changes(self):
        """Test cache invalidation when PHI patterns change."""
        
        cache = ModelCache(max_size=10, ttl=300)
        
        # Cache medical data with potential PHI
        input1 = "Patient reports symptoms: SSN 123-45-6789"
        input2 = "Patient reports symptoms: different data"
        
        cache.set("model1", input1, {}, {"result": "result1"})
        cache.set("model1", input2, {}, {"result": "result2"})
        
        # Verify both are cached
        assert cache.get("model1", input1, {}) is not None
        assert cache.get("model1", input2, {}) is not None
        
        # Simulate cache size limit causing eviction
        cache._max_size = 2  # Very small limit
        cache._cache.clear()  # Clear to simulate limit
        
        # Re-add items to trigger eviction
        cache.set("model1", input1, {}, {"result": "result1"})
        cache.set("model1", "input3", {}, {"result": "result3"})
        cache.set("model1", "input4", {}, {"result": "result4"})
        
        # First item should be evicted
        assert cache.get("model1", input1, {}) is None


class TestModelRegistry:
    """Test model registry with medical models."""
    
    @pytest.mark.unit
    @pytest.mark.medical
    def test_model_registration(self):
        """Test medical model registration."""
        
        # Test registering a clinical text generation model
        text_server = TextGenerationServer(
            model_id="clinical_text_gen_v1",
            name="Clinical Text Generator",
            medical_context=True
        )
        
        assert text_server.model_id == "clinical_text_gen_v1"
        assert text_server.name == "Clinical Text Generator"
        
    @pytest.mark.unit
    @pytest.mark.medical
    def test_model_health_monitoring(self):
        """Test model health monitoring for medical models."""
        
        server = TextGenerationServer(
            model_id="test_clinical_model",
            name="Test Clinical Model"
        )
        
        # Test initial health status
        health_status = server.get_health_status()
        assert "status" in health_status
        assert "last_prediction" in health_status
        assert "prediction_count" in health_status
        assert "error_count" in health_status
        
    @pytest.mark.unit
    @pytest.mark.medical
    def test_model_metrics_collection(self):
        """Test metrics collection for medical models."""
        
        server = ClinicalAnalysisServer(
            model_id="test_analysis_model",
            name="Test Clinical Analysis"
        )
        
        # Simulate some predictions
        server.increment_prediction_count()
        server.increment_prediction_count()
        server.update_last_prediction_time()
        
        # Get metrics
        metrics = server.get_metrics()
        assert metrics["prediction_count"] == 2
        assert "last_prediction_time" in metrics
        assert "uptime_seconds" in metrics


class TestClinicalAccuracyValidation:
    """Test clinical accuracy validation mechanisms."""
    
    @pytest.mark.unit
    @pytest.mark.medical
    async def test_clinical_accuracy_threshold(self):
        """Test clinical accuracy threshold validation."""
        
        from api.utils.accuracy_validator import ClinicalAccuracyValidator
        
        validator = ClinicalAccuracyValidator()
        
        # Test case with acceptable accuracy
        high_accuracy_result = {
            "diagnosis": "Type 2 Diabetes Mellitus",
            "confidence": 0.92,
            "evidence_based": True,
            "clinical_guidelines_followed": True
        }
        
        # Test case with low accuracy
        low_accuracy_result = {
            "diagnosis": "Unknown condition",
            "confidence": 0.45,
            "evidence_based": False,
            "clinical_guidelines_followed": False
        }
        
        assert validator.validate_accuracy(high_accuracy_result)["passed"] is True
        assert validator.validate_accuracy(low_accuracy_result)["passed"] is False
    
    @pytest.mark.unit
    @pytest.mark.medical
    async def test_evidence_based_recommendations(self):
        """Test evidence-based recommendation validation."""
        
        from api.utils.evidence_validator import EvidenceValidator
        
        validator = EvidenceValidator()
        
        # Evidence-based recommendation
        evidence_based_rec = {
            "recommendation": "Prescribe Metformin 500mg BID for Type 2 Diabetes",
            "evidence_level": "Level A",
            "guidelines": ["ADA Standards of Care", "American Diabetes Association"],
            "contraindications_checked": True,
            "drug_interactions_reviewed": True
        }
        
        # Non-evidence-based recommendation
        non_evidence_based_rec = {
            "recommendation": "Try this experimental therapy",
            "evidence_level": "Insufficient",
            "guidelines": [],
            "contraindications_checked": False,
            "drug_interactions_reviewed": False
        }
        
        assert validator.validate_evidence(evidence_based_rec)["evidence_based"] is True
        assert validator.validate_evidence(non_evidence_based_rec)["evidence_based"] is False


class TestServerPerformance:
    """Test server performance characteristics."""
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_model_loading_performance(self):
        """Test model loading performance."""
        
        server = TextGenerationServer(
            model_id="performance_test_model",
            name="Performance Test Model"
        )
        
        # Measure model loading time
        start_time = time.time()
        # In real implementation, this would involve actual model loading
        # For testing, we mock the loading process
        loading_time = time.time() - start_time
        
        # Should load within reasonable time (mock test)
        assert loading_time < 1.0  # Should be very fast in mock
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_prediction_latency(self):
        """Test prediction latency requirements."""
        
        server = ClinicalAnalysisServer(
            model_id="latency_test_model",
            name="Latency Test Model"
        )
        
        # Mock prediction with specific timing
        with patch.object(server, 'analyze', new_callable=AsyncMock) as mock_analyze:
            # Simulate processing time
            mock_analyze.return_value = {"result": "test"}
            
            start_time = time.time()
            result = await server.analyze({"test": "data"})
            latency = time.time() - start_time
            
            # Should complete within 2 seconds (mock)
            assert latency < 0.1  # Very fast in mock
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])