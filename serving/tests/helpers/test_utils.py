"""
Test utilities and helper functions for medical AI testing framework.

This module provides common utilities used across different test categories
including data generators, validators, and testing helpers.
"""

import json
import hashlib
import secrets
import string
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np
import random


class MedicalDataGenerator:
    """Generate synthetic medical data for testing."""
    
    @staticmethod
    def generate_patient_profile(age_range: tuple = (18, 90), 
                                gender_options: List[str] = None) -> Dict[str, Any]:
        """Generate synthetic patient profile."""
        
        if gender_options is None:
            gender_options = ["M", "F", "Other"]
        
        return {
            "patient_id": f"PT_{secrets.token_hex(4).upper()}",
            "age": random.randint(age_range[0], age_range[1]),
            "gender": random.choice(gender_options),
            "ethnicity": random.choice([
                "Caucasian", "Hispanic", "African American", 
                "Asian", "Native American", "Other"
            ]),
            "blood_type": random.choice(["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
            "height_cm": random.randint(150, 200),
            "weight_kg": random.randint(45, 120),
            "created_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def generate_clinical_symptoms(condition: str = None) -> Dict[str, Any]:
        """Generate clinical symptoms based on condition."""
        
        symptom_sets = {
            "diabetes": ["polyuria", "polydipsia", "weight_loss", "fatigue", "blurred_vision"],
            "hypertension": ["headache", "dizziness", "chest_pain", "shortness_of_breath"],
            "chest_pain": ["chest_pain", "shortness_of_breath", "diaphoresis", "nausea"],
            "infection": ["fever", "chills", "fatigue", "pain", "swelling"],
            "anxiety": ["restlessness", "rapid_heartbeat", "sweating", "trembling"]
        }
        
        if condition and condition in symptom_sets:
            symptoms = random.sample(symptom_sets[condition], 
                                   random.randint(2, len(symptom_sets[condition])))
        else:
            symptoms = ["general_discomfort"]
        
        return {
            "symptoms": symptoms,
            "severity": random.choice(["mild", "moderate", "severe"]),
            "duration": random.choice(["hours", "days", "weeks", "months"]),
            "onset": random.choice(["gradual", "sudden", "progressive"])
        }
    
    @staticmethod
    def generate_lab_values(condition: str = None) -> Dict[str, Any]:
        """Generate laboratory test values."""
        
        base_labs = {
            "glucose": random.randint(70, 300),
            "hba1c": round(random.uniform(5.0, 12.0), 1),
            "blood_pressure": f"{random.randint(110, 160)}/{random.randint(70, 100)}",
            "heart_rate": random.randint(60, 120),
            "temperature": round(random.uniform(96.0, 101.0), 1),
            "hemoglobin": round(random.uniform(10.0, 17.0), 1),
            "white_blood_cells": random.randint(4, 15),
            "platelet_count": random.randint(150, 450)
        }
        
        if condition == "diabetes":
            base_labs["glucose"] = random.randint(150, 400)
            base_labs["hba1c"] = round(random.uniform(7.0, 14.0), 1)
        elif condition == "hypertension":
            systolic = random.randint(140, 180)
            diastolic = random.randint(90, 110)
            base_labs["blood_pressure"] = f"{systolic}/{diastolic}"
        
        return base_labs
    
    @staticmethod
    def generate_medications(condition: str = None) -> List[Dict[str, Any]]:
        """Generate medication list."""
        
        medication_options = {
            "diabetes": [
                {"name": "Metformin", "dose": "500mg", "frequency": "BID"},
                {"name": "Insulin", "dose": "10 units", "frequency": "daily"}
            ],
            "hypertension": [
                {"name": "Lisinopril", "dose": "10mg", "frequency": "daily"},
                {"name": "Amlodipine", "dose": "5mg", "frequency": "daily"}
            ],
            "general": [
                {"name": "Aspirin", "dose": "81mg", "frequency": "daily"},
                {"name": "Vitamin D", "dose": "1000 IU", "frequency": "daily"}
            ]
        }
        
        if condition in medication_options:
            return random.sample(medication_options[condition], 
                               random.randint(1, len(medication_options[condition])))
        else:
            return random.sample(medication_options["general"], 1)


class PHIProtectionValidator:
    """Validate PHI protection and redaction."""
    
    PHI_PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
        'phone': r'\b\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'dob': r'\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](19|20)\d{2}\b',
        'mrn': r'\bMRN\s*\d{6,10}\b',
        'address': r'\b\d+\s+[A-Za-z\s]+\b',
        'name': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
    }
    
    @staticmethod
    def scan_for_phi(text: str) -> Dict[str, List[str]]:
        """Scan text for PHI patterns."""
        
        import re
        findings = {}
        
        for pattern_name, pattern in PHIProtectionValidator.PHI_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings[pattern_name] = matches
        
        return findings
    
    @staticmethod
    def validate_redaction(original_text: str, redacted_text: str) -> Dict[str, Any]:
        """Validate effectiveness of PHI redaction."""
        
        original_phi = PHIProtectionValidator.scan_for_phi(original_text)
        redacted_phi = PHIProtectionValidator.scan_for_phi(redacted_text)
        
        original_count = sum(len(matches) for matches in original_phi.values())
        redacted_count = sum(len(matches) for matches in redacted_phi.values())
        
        return {
            "original_phi_count": original_count,
            "redacted_phi_count": redacted_count,
            "redaction_effective": redacted_count == 0 and original_count > 0,
            "redaction_percentage": ((original_count - redacted_count) / original_count * 100) if original_count > 0 else 100
        }


class ClinicalAccuracyValidator:
    """Validate clinical accuracy of AI responses."""
    
    @staticmethod
    def validate_diagnosis_accuracy(ai_diagnosis: str, expected_diagnosis: str) -> float:
        """Validate diagnostic accuracy using text similarity."""
        
        ai_lower = ai_diagnosis.lower()
        expected_lower = expected_diagnosis.lower()
        
        # Extract medical concepts (simplified)
        medical_concepts = {
            "diabetes": ["diabetes", "dm", "diabetic"],
            "hypertension": ["hypertension", "htn", "high_blood_pressure"],
            "chest_pain": ["chest_pain", "angina", "cardiac"],
            "infection": ["infection", "sepsis", "pneumonia"],
            "obesity": ["obesity", "overweight"],
            "anxiety": ["anxiety", "anxious"]
        }
        
        ai_concepts = set()
        expected_concepts = set()
        
        for concept, keywords in medical_concepts.items():
            if any(keyword in ai_lower for keyword in keywords):
                ai_concepts.add(concept)
            if any(keyword in expected_lower for keyword in keywords):
                expected_concepts.add(concept)
        
        if not expected_concepts:
            return 0.5
        
        overlap = len(ai_concepts & expected_concepts)
        return overlap / len(expected_concepts)
    
    @staticmethod
    def validate_treatment_accuracy(ai_treatments: List[str], 
                                  expected_treatments: List[str]) -> float:
        """Validate treatment recommendation accuracy."""
        
        if not ai_treatments or not expected_treatments:
            return 0.0
        
        # Treatment keywords mapping
        treatment_keywords = {
            "metformin": ["metformin", "biguanide"],
            "insulin": ["insulin", "insulin_therapy"],
            "ace_inhibitor": ["ace_inhibitor", "lisinopril", "captopril"],
            "lifestyle": ["lifestyle", "diet", "exercise"],
            "monitoring": ["monitoring", "follow_up", "surveillance"]
        }
        
        matches = 0
        for expected in expected_treatments:
            expected_lower = expected.lower()
            expected_keywords = treatment_keywords.get(expected_lower, [expected_lower])
            
            for ai_treatment in ai_treatments:
                ai_lower = ai_treatment.lower()
                if any(keyword in ai_lower for keyword in expected_keywords):
                    matches += 1
                    break
        
        return min(matches / len(expected_treatments), 1.0)
    
    @staticmethod
    def calculate_overall_accuracy(ai_response: Dict, expected_response: Dict) -> float:
        """Calculate overall clinical accuracy score."""
        
        scores = []
        
        # Diagnosis accuracy
        if "diagnosis" in ai_response and "diagnosis" in expected_response:
            diag_score = ClinicalAccuracyValidator.validate_diagnosis_accuracy(
                ai_response["diagnosis"],
                expected_response["diagnosis"]
            )
            scores.append(diag_score * 0.4)  # 40% weight
        
        # Treatment accuracy
        if "treatment_recommendations" in ai_response and "treatment_recommendations" in expected_response:
            treat_score = ClinicalAccuracyValidator.validate_treatment_accuracy(
                ai_response["treatment_recommendations"],
                expected_response["treatment_recommendations"]
            )
            scores.append(treat_score * 0.4)  # 40% weight
        
        # Clinical reasoning accuracy
        if "clinical_reasoning" in ai_response:
            reasoning_score = 0.8 if len(ai_response["clinical_reasoning"]) > 50 else 0.5
            scores.append(reasoning_score * 0.2)  # 20% weight
        
        return sum(scores) if scores else 0.5


class PerformanceBenchmark:
    """Performance benchmarking utilities."""
    
    @staticmethod
    def measure_response_time(func, *args, **kwargs) -> tuple:
        """Measure function execution time."""
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        return result, response_time
    
    @staticmethod
    def measure_throughput(requests_per_second: int, duration_seconds: int) -> Dict[str, float]:
        """Measure system throughput."""
        
        return {
            "target_rps": requests_per_second,
            "duration_seconds": duration_seconds,
            "total_requests": requests_per_second * duration_seconds
        }
    
    @staticmethod
    def calculate_percentiles(response_times: List[float]) -> Dict[str, float]:
        """Calculate response time percentiles."""
        
        if not response_times:
            return {"p50": 0, "p95": 0, "p99": 0, "p999": 0}
        
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        
        return {
            "p50": np.percentile(sorted_times, 50),
            "p95": np.percentile(sorted_times, 95),
            "p99": np.percentile(sorted_times, 99),
            "p999": np.percentile(sorted_times, 99.9)
        }


class SecurityTestHelper:
    """Security testing helper utilities."""
    
    @staticmethod
    def generate_sql_injection_payloads() -> List[str]:
        """Generate SQL injection test payloads."""
        
        return [
            "' OR '1'='1",
            "'; DROP TABLE patients; --",
            "' UNION SELECT * FROM users --",
            "admin'--",
            "1' OR '1'='1' --",
            "' OR 1=1--",
            "' UNION SELECT password FROM users--"
        ]
    
    @staticmethod
    def generate_xss_payloads() -> List[str]:
        """Generate XSS test payloads."""
        
        return [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<svg/onload=alert('XSS')>",
            "'><script>alert('XSS')</script>"
        ]
    
    @staticmethod
    def generate_authentication_bypass_headers() -> List[Dict[str, str]]:
        """Generate authentication bypass test headers."""
        
        return [
            {"Authorization": "Bearer admin"},
            {"Authorization": "Bearer 123456"},
            {"X-User-ID": "admin"},
            {"X-User-ID": "0"},
            {"X-Admin": "true"},
            {"User-Id": "administrator"}
        ]
    
    @staticmethod
    def validate_encryption(data: str, encrypted_data: str, decrypt_func) -> bool:
        """Validate that encryption/decryption works correctly."""
        
        try:
            decrypted_data = decrypt_func(encrypted_data)
            return data == decrypted_data
        except Exception:
            return False


class ComplianceValidator:
    """Medical compliance validation utilities."""
    
    @staticmethod
    def check_hipaa_compliance(security_config: Dict) -> Dict[str, Any]:
        """Check HIPAA compliance."""
        
        findings = []
        score = 0.0
        max_score = 4.0
        
        # Administrative safeguards
        if security_config.get("administrative_safeguards", {}).get("security_officer"):
            findings.append("✅ Security Officer assigned")
            score += 1.0
        else:
            findings.append("❌ Security Officer not assigned")
        
        # Physical safeguards
        if security_config.get("physical_safeguards", {}).get("facility_access_controls"):
            findings.append("✅ Facility access controls implemented")
            score += 1.0
        else:
            findings.append("❌ Facility access controls missing")
        
        # Technical safeguards
        technical = security_config.get("technical_safeguards", {})
        if technical.get("access_control") and technical.get("audit_logs"):
            findings.append("✅ Access control and audit logs implemented")
            score += 1.0
        else:
            findings.append("❌ Technical safeguards incomplete")
        
        # Policies and procedures
        if security_config.get("policies", {}).get("incident_response"):
            findings.append("✅ Incident response procedures in place")
            score += 1.0
        else:
            findings.append("❌ Incident response procedures missing")
        
        return {
            "compliant": score >= 3.0,
            "score": score / max_score,
            "findings": findings,
            "percentage": (score / max_score) * 100
        }
    
    @staticmethod
    def validate_audit_trail_completeness(audit_entries: List[Dict]) -> Dict[str, Any]:
        """Validate completeness of audit trail."""
        
        required_fields = ["timestamp", "user_id", "action", "resource_accessed", "ip_address"]
        optional_fields = ["session_id", "user_agent", "result"]
        
        completeness_scores = []
        
        for entry in audit_entries:
            field_score = 0
            for field in required_fields:
                if field in entry and entry[field]:
                    field_score += 1
            completeness_scores.append(field_score / len(required_fields))
        
        overall_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        return {
            "complete": overall_score >= 0.9,
            "score": overall_score,
            "total_entries": len(audit_entries),
            "required_fields_present": sum(1 for entry in audit_entries 
                                         if all(field in entry for field in required_fields))
        }


class TestDataManager:
    """Manage test data lifecycle."""
    
    def __init__(self, test_data_dir: str = "tests/data"):
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(exist_ok=True)
    
    def save_test_data(self, data: Dict, filename: str) -> str:
        """Save test data to file."""
        
        file_path = self.test_data_dir / f"{filename}.json"
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(file_path)
    
    def load_test_data(self, filename: str) -> Dict:
        """Load test data from file."""
        
        file_path = self.test_data_dir / f"{filename}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def generate_test_data_id(self) -> str:
        """Generate unique test data ID."""
        
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(4)
        return f"{timestamp}_{random_suffix}"
    
    def cleanup_test_data(self, older_than_hours: int = 24):
        """Clean up old test data files."""
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        for file_path in self.test_data_dir.glob("*.json"):
            if file_path.stat().st_mtime < cutoff_time.timestamp():
                file_path.unlink()


class MockResponseGenerator:
    """Generate mock responses for testing."""
    
    @staticmethod
    def generate_clinical_analysis_response(case_type: str, accuracy: float = 0.85) -> Dict[str, Any]:
        """Generate mock clinical analysis response."""
        
        responses = {
            "diabetes": {
                "diagnosis": "Type 2 Diabetes Mellitus",
                "differential_diagnosis": [
                    {"condition": "Type 2 Diabetes", "probability": 0.85},
                    {"condition": "Type 1 Diabetes", "probability": 0.10},
                    {"condition": "Secondary diabetes", "probability": 0.05}
                ],
                "treatment_recommendations": ["Metformin 500mg BID", "Lifestyle modifications"],
                "clinical_reasoning": "Based on patient's symptoms and lab results...",
                "evidence_level": "A"
            },
            "hypertension": {
                "diagnosis": "Stage 2 Hypertension",
                "differential_diagnosis": [
                    {"condition": "Essential Hypertension", "probability": 0.90},
                    {"condition": "Secondary Hypertension", "probability": 0.10}
                ],
                "treatment_recommendations": ["Lisinopril 10mg daily", "Dietary changes"],
                "clinical_reasoning": "Consistent with elevated blood pressure readings...",
                "evidence_level": "A"
            }
        }
        
        base_response = responses.get(case_type, responses["diabetes"])
        base_response["clinical_accuracy"] = accuracy
        base_response["timestamp"] = datetime.now().isoformat()
        
        return base_response
    
    @staticmethod
    def generate_medical_qa_response(question: str, confidence: float = 0.88) -> Dict[str, Any]:
        """Generate mock medical Q&A response."""
        
        return {
            "question": question,
            "answer": "Based on current medical guidelines, the recommended approach is...",
            "confidence_score": confidence,
            "evidence_level": "A",
            "references": [
                "American Diabetes Association Standards of Care",
                "UpToDate Clinical Decision Support"
            ],
            "related_questions": [
                "What are the side effects of metformin?",
                "How often should HbA1c be monitored?"
            ],
            "timestamp": datetime.now().isoformat()
        }


# Utility functions for common test operations
def create_test_session() -> Dict[str, Any]:
    """Create a test session for workflow testing."""
    
    return {
        "session_id": str(uuid.uuid4()),
        "user_id": f"test_user_{secrets.token_hex(4)}",
        "start_time": datetime.now().isoformat(),
        "test_mode": True
    }


def generate_test_conversation(num_turns: int = 5) -> List[Dict[str, Any]]:
    """Generate test conversation data."""
    
    conversation = []
    for i in range(num_turns):
        turn = {
            "turn_id": f"turn_{i+1}",
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Test message {i+1}" if i % 2 == 0 else f"Test response {i+1}",
            "timestamp": (datetime.now() - timedelta(minutes=num_turns-i)).isoformat()
        }
        conversation.append(turn)
    
    return conversation


def calculate_test_coverage(test_results: List[Dict]) -> Dict[str, float]:
    """Calculate test coverage metrics."""
    
    if not test_results:
        return {"total_tests": 0, "passed": 0, "failed": 0, "coverage": 0.0}
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.get("passed", False))
    failed_tests = total_tests - passed_tests
    coverage = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    return {
        "total_tests": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "coverage": coverage
    }