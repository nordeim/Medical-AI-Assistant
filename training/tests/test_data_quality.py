"""
Data Quality Testing Module
===========================

Comprehensive data validation and quality assurance for medical AI training data.
"""

import os
import sys
import json
import logging
import tempfile
import shutil
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict
import statistics
from dataclasses import dataclass
import hashlib

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from phi_redactor import PHIRedactor
from phi_validator import PHIValidator
from compliance_checker import ComplianceChecker

@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    missing_fields: int = 0
    duplicate_records: int = 0
    phi_violations: int = 0
    data_consistency_score: float = 0.0
    medical_accuracy_score: float = 0.0
    safety_compliance_score: float = 0.0

class MedicalDataValidator:
    """Validates medical data quality and compliance."""
    
    def __init__(self):
        self.phi_redactor = PHIRedactor()
        self.phi_validator = PHIValidator()
        self.compliance_checker = ComplianceChecker()
        self.quality_metrics = DataQualityMetrics()
    
    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """Validate text quality for medical data."""
        issues = []
        
        # Check for empty or None text
        if not text or not isinstance(text, str):
            issues.append("Empty or invalid text")
            return {"valid": False, "issues": issues}
        
        # Check minimum length
        if len(text.strip()) < 10:
            issues.append("Text too short (< 10 characters)")
        
        # Check maximum length
        if len(text) > 10000:
            issues.append("Text too long (> 10,000 characters)")
        
        # Check for excessive whitespace
        if '  ' in text:
            issues.append("Multiple consecutive spaces detected")
        
        # Check for excessive special characters
        special_char_count = len(re.findall(r'[^\w\s\.,;:\!\?\-\(\)]', text))
        if special_char_count > len(text) * 0.1:
            issues.append("Excessive special characters")
        
        # Check for medical terminology presence (basic check)
        common_medical_terms = [
            'patient', 'doctor', 'hospital', 'medication', 'treatment',
            'diagnosis', 'symptoms', 'condition', 'therapy', 'clinic'
        ]
        has_medical_terms = any(term.lower() in text.lower() for term in common_medical_terms)
        if not has_medical_terms:
            issues.append("No apparent medical content detected")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "length": len(text),
            "word_count": len(text.split()),
            "has_medical_terms": has_medical_terms
        }
    
    def validate_field_completeness(self, record: Dict[str, Any], required_fields: List[str]) -> Dict[str, Any]:
        """Validate field completeness in records."""
        missing_fields = []
        present_fields = []
        
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == "":
                missing_fields.append(field)
            else:
                present_fields.append(field)
        
        # Check for additional unexpected fields
        unexpected_fields = [field for field in record.keys() if field not in required_fields]
        
        return {
            "complete": len(missing_fields) == 0,
            "missing_fields": missing_fields,
            "present_fields": present_fields,
            "unexpected_fields": unexpected_fields,
            "completeness_score": len(present_fields) / len(required_fields)
        }
    
    def validate_data_types(self, record: Dict[str, Any], schema: Dict[str, type]) -> Dict[str, Any]:
        """Validate data types according to schema."""
        type_violations = []
        
        for field, expected_type in schema.items():
            if field in record and record[field] is not None:
                actual_value = record[field]
                
                # Special handling for different types
                if expected_type == str and not isinstance(actual_value, str):
                    type_violations.append(f"Field '{field}': expected string, got {type(actual_value)}")
                elif expected_type == int and not isinstance(actual_value, int):
                    if isinstance(actual_value, float) and actual_value == int(actual_value):
                        # Allow float that is exactly an integer
                        continue
                    type_violations.append(f"Field '{field}': expected int, got {type(actual_value)}")
                elif expected_type == float and not isinstance(actual_value, (int, float)):
                    type_violations.append(f"Field '{field}': expected float, got {type(actual_value)}")
                elif expected_type == bool and not isinstance(actual_value, bool):
                    type_violations.append(f"Field '{field}': expected bool, got {type(actual_value)}")
                elif isinstance(expected_type, list) and not isinstance(actual_value, expected_type):
                    type_violations.append(f"Field '{field}': expected {expected_type}, got {type(actual_value)}")
        
        return {
            "valid": len(type_violations) == 0,
            "type_violations": type_violations
        }
    
    def detect_duplicates(self, records: List[Dict[str, Any]], key_fields: List[str]) -> Dict[str, Any]:
        """Detect duplicate records based on key fields."""
        seen_records = defaultdict(list)
        duplicates = []
        
        for idx, record in enumerate(records):
            # Create key from specified fields
            key_fields_values = tuple(record.get(field) for field in key_fields)
            key_hash = hashlib.md5(str(key_fields_values).encode()).hexdigest()
            
            if key_hash in seen_records:
                duplicates.append({
                    "duplicate_index": idx,
                    "original_index": seen_records[key_hash][0],
                    "key_fields": dict(zip(key_fields, key_fields_values))
                })
            else:
                seen_records[key_hash].append(idx)
        
        return {
            "has_duplicates": len(duplicates) > 0,
            "duplicate_count": len(duplicates),
            "duplicates": duplicates,
            "unique_count": len(records) - len(duplicates)
        }

class PHIComplianceValidator:
    """Validates PHI (Protected Health Information) compliance."""
    
    def __init__(self):
        self.phi_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b\d{3}-?\d{3}-?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'medical_record': r'\b(MR|Medical Record|Record #?)\s*:?\s*\d+\b',
            'date_of_birth': r'\b(01/02/1990|January 2, 1990|1990-01-02)\b',
            'address': r'\b\d+\s+([A-Za-z0-9]+\s+){1,4}(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd)\b'
        }
    
    def scan_for_phi_patterns(self, text: str) -> Dict[str, List[str]]:
        """Scan text for PHI patterns."""
        phi_matches = {}
        
        for pattern_name, pattern in self.phi_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            found_matches = [match.group() for match in matches]
            
            if found_matches:
                phi_matches[pattern_name] = found_matches
        
        return phi_matches
    
    def validate_phi_compliance(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate PHI compliance across all records."""
        total_phi_violations = 0
        compliance_details = []
        
        for idx, record in enumerate(records):
            text_fields = [v for k, v in record.items() if isinstance(v, str)]
            combined_text = ' '.join(text_fields)
            
            phi_patterns_found = self.scan_for_phi_patterns(combined_text)
            
            if phi_patterns_found:
                total_phi_violations += 1
                compliance_details.append({
                    "record_index": idx,
                    "phi_patterns_found": phi_patterns_found,
                    "severity": "high" if any(p in phi_patterns_found for p in ['ssn', 'email', 'phone']) else "medium"
                })
        
        compliance_rate = (len(records) - total_phi_violations) / len(records) if records else 1.0
        
        return {
            "compliance_rate": compliance_rate,
            "total_violations": total_phi_violations,
            "violation_details": compliance_details,
            "compliant_records": len(records) - total_phi_violations
        }

class MedicalAccuracyValidator:
    """Validates medical accuracy and safety of content."""
    
    def __init__(self):
        self.medical_dictionaries = {
            'medications': [
                'aspirin', 'ibuprofen', 'acetaminophen', 'antibiotic', 'analgesic',
                'antihistamine', 'insulin', 'metformin', 'lisinopril', 'amlodipine'
            ],
            'conditions': [
                'hypertension', 'diabetes', 'asthma', 'arthritis', 'depression',
                'anxiety', 'infection', 'inflammation', 'allergy', 'obesity'
            ],
            'body_parts': [
                'heart', 'lungs', 'liver', 'kidneys', 'brain', 'stomach',
                'intestines', 'pancreas', 'thyroid', 'blood vessels'
            ],
            'procedures': [
                'surgery', 'biopsy', 'endoscopy', 'imaging', 'blood test',
                'therapy', 'injection', 'examination', 'consultation', 'diagnosis'
            ]
        }
        
        self.dangerous_content_patterns = [
            r'how to self-harm',
            r'suicide.*(method|ways|help)',
            r'illegal drug.*(manufacture|make)',
            r'weapon.*(create|make)',
            r'poison.*(create|make)'
        ]
    
    def validate_medical_content(self, text: str) -> Dict[str, Any]:
        """Validate medical content for accuracy and safety."""
        issues = []
        medical_terms_found = []
        
        # Check for medical terminology
        text_lower = text.lower()
        for category, terms in self.medical_dictionaries.items():
            found_terms = [term for term in terms if term in text_lower]
            medical_terms_found.extend(found_terms)
        
        if not medical_terms_found:
            issues.append("No medical terminology detected - content may not be medical")
        
        # Check for potentially dangerous content
        for pattern in self.dangerous_content_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                issues.append("Potentially dangerous content detected")
                break
        
        # Check for appropriate medical language
        inappropriate_terms = ['cure guaranteed', 'miracle drug', 'instant healing']
        found_inappropriate = [term for term in inappropriate_terms if term in text_lower]
        if found_inappropriate:
            issues.append(f"Inappropriate medical claims: {', '.join(found_inappropriate)}")
        
        # Check for medical disclaimer presence
        has_disclaimer = any(disclaimer in text_lower for disclaimer in [
            'not a substitute for professional advice',
            'consult your doctor',
            'medical professional'
        ])
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "medical_terms_found": medical_terms_found,
            "has_disclaimer": has_disclaimer,
            "text_category": "medical" if medical_terms_found else "non-medical"
        }
    
    def validate_consistency_across_records(self, records: List[Dict[str, Any]], consistency_fields: List[str]) -> Dict[str, Any]:
        """Validate consistency of medical information across records."""
        field_values = defaultdict(list)
        
        for record in records:
            for field in consistency_fields:
                if field in record and record[field]:
                    field_values[field].append(record[field])
        
        consistency_scores = {}
        for field, values in field_values.items():
            if len(values) > 1:
                # Calculate consistency as 1 - (unique_values / total_values)
                unique_values = len(set(values))
                consistency_score = 1 - (unique_values / len(values))
                consistency_scores[field] = consistency_score
        
        return {
            "consistency_scores": consistency_scores,
            "average_consistency": statistics.mean(consistency_scores.values()) if consistency_scores else 1.0,
            "inconsistent_fields": [field for field, score in consistency_scores.items() if score < 0.8]
        }

class DataQualityTester:
    """Main data quality testing class."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.validator = MedicalDataValidator()
        self.phi_validator = PHIComplianceValidator()
        self.accuracy_validator = MedicalAccuracyTester()
        
        # Load configuration if provided
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load testing configuration."""
        default_config = {
            "required_fields": ["text"],
            "schema": {"text": str},
            "key_fields_for_duplicates": ["text"],
            "consistency_fields": ["medical_category"],
            "quality_thresholds": {
                "min_text_length": 10,
                "max_text_length": 10000,
                "min_completeness": 0.8,
                "min_phi_compliance": 0.99,
                "min_medical_accuracy": 0.7
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def load_test_data(self, data_path: str, data_type: str = "jsonl") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Load test data from file."""
        records = []
        
        if data_type == "jsonl":
            with open(data_path, 'r') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        
        elif data_type == "json":
            with open(data_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]
        
        elif data_type == "csv":
            df = pd.read_csv(data_path)
            records = df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        return records, {"total_loaded": len(records)}
    
    def run_complete_validation(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run complete data quality validation."""
        logger.info(f"Starting validation of {len(records)} records")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_records": len(records),
            "validation_results": {},
            "quality_metrics": {},
            "recommendations": []
        }
        
        # 1. Text quality validation
        logger.info("Validating text quality...")
        text_quality_results = []
        for idx, record in enumerate(records):
            if 'text' in record:
                validation = self.validator.validate_text_quality(record['text'])
                validation['record_index'] = idx
                text_quality_results.append(validation)
        
        valid_texts = sum(1 for r in text_quality_results if r['valid'])
        results["validation_results"]["text_quality"] = {
            "valid_count": valid_texts,
            "invalid_count": len(text_quality_results) - valid_texts,
            "details": text_quality_results[:10]  # First 10 for inspection
        }
        
        # 2. Field completeness validation
        logger.info("Validating field completeness...")
        completeness_results = []
        for idx, record in enumerate(records):
            validation = self.validator.validate_field_completeness(record, self.config["required_fields"])
            validation['record_index'] = idx
            completeness_results.append(validation)
        
        complete_records = sum(1 for r in completeness_results if r['complete'])
        results["validation_results"]["completeness"] = {
            "complete_count": complete_records,
            "incomplete_count": len(completeness_results) - complete_records,
            "details": completeness_results[:10]
        }
        
        # 3. Data type validation
        logger.info("Validating data types...")
        type_results = []
        for idx, record in enumerate(records):
            validation = self.validator.validate_data_types(record, self.config["schema"])
            validation['record_index'] = idx
            type_results.append(validation)
        
        valid_types = sum(1 for r in type_results if r['valid'])
        results["validation_results"]["data_types"] = {
            "valid_count": valid_types,
            "invalid_count": len(type_results) - valid_types,
            "details": type_results[:10]
        }
        
        # 4. Duplicate detection
        logger.info("Detecting duplicates...")
        duplicate_results = self.validator.detect_duplicates(records, self.config["key_fields_for_duplicates"])
        results["validation_results"]["duplicates"] = duplicate_results
        
        # 5. PHI compliance validation
        logger.info("Validating PHI compliance...")
        phi_results = self.phi_validator.validate_phi_compliance(records)
        results["validation_results"]["phi_compliance"] = phi_results
        
        # 6. Medical accuracy validation
        logger.info("Validating medical accuracy...")
        accuracy_results = []
        for idx, record in enumerate(records):
            if 'text' in record:
                validation = self.accuracy_validator.validate_medical_content(record['text'])
                validation['record_index'] = idx
                accuracy_results.append(validation)
        
        accurate_content = sum(1 for r in accuracy_results if r['valid'])
        results["validation_results"]["medical_accuracy"] = {
            "accurate_count": accurate_content,
            "inaccurate_count": len(accuracy_results) - accurate_content,
            "details": accuracy_results[:10]
        }
        
        # 7. Consistency validation
        logger.info("Validating consistency...")
        consistency_results = self.accuracy_validator.validate_consistency_across_records(
            records, self.config["consistency_fields"]
        )
        results["validation_results"]["consistency"] = consistency_results
        
        # Calculate overall quality metrics
        results["quality_metrics"] = self._calculate_quality_metrics(results)
        
        # Generate recommendations
        results["recommendations"] = self._generate_recommendations(results)
        
        logger.info("Data quality validation completed")
        return results
    
    def _calculate_quality_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        metrics = DataQualityMetrics()
        
        total = results["total_records"]
        metrics.total_records = total
        
        # Text quality
        metrics.valid_records = results["validation_results"]["text_quality"]["valid_count"]
        
        # Completeness
        metrics.missing_fields = sum(
            len(r["missing_fields"]) for r in results["validation_results"]["completeness"]["details"]
        )
        
        # Duplicates
        metrics.duplicate_records = results["validation_results"]["duplicates"]["duplicate_count"]
        
        # PHI compliance
        metrics.phi_violations = results["validation_results"]["phi_compliance"]["total_violations"]
        
        # Scores
        if total > 0:
            metrics.data_consistency_score = results["validation_results"]["consistency"]["average_consistency"]
            
            if "medical_accuracy" in results["validation_results"]:
                metrics.medical_accuracy_score = (
                    results["validation_results"]["medical_accuracy"]["accurate_count"] / total
                )
            
            metrics.safety_compliance_score = results["validation_results"]["phi_compliance"]["compliance_rate"]
        
        return {
            "data_consistency_score": metrics.data_consistency_score,
            "medical_accuracy_score": metrics.medical_accuracy_score,
            "safety_compliance_score": metrics.safety_compliance_score,
            "completeness_rate": (total - metrics.missing_fields) / total if total > 0 else 0,
            "duplicate_rate": metrics.duplicate_records / total if total > 0 else 0,
            "phi_compliance_rate": metrics.safety_compliance_score
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        thresholds = self.config["quality_thresholds"]
        metrics = results["quality_metrics"]
        
        # Completeness recommendations
        completeness_rate = metrics["completeness_rate"]
        if completeness_rate < thresholds["min_completeness"]:
            recommendations.append(
                f"Data completeness is {completeness_rate:.1%}, below threshold of {thresholds['min_completeness']:.1%}. "
                "Consider adding missing fields or filtering incomplete records."
            )
        
        # PHI compliance recommendations
        phi_rate = metrics["phi_compliance_rate"]
        if phi_rate < thresholds["min_phi_compliance"]:
            recommendations.append(
                f"PHI compliance is {phi_rate:.1%}, below threshold of {thresholds['min_phi_compliance']:.1%}. "
                "Remove or anonymize all PHI before training."
            )
        
        # Medical accuracy recommendations
        accuracy_rate = metrics["medical_accuracy_score"]
        if accuracy_rate < thresholds["min_medical_accuracy"]:
            recommendations.append(
                f"Medical accuracy is {accuracy_rate:.1%}, below threshold of {thresholds['min_medical_accuracy']:.1%}. "
                "Review medical content for accuracy and appropriate language."
            )
        
        # Duplicate recommendations
        duplicate_rate = metrics["duplicate_rate"]
        if duplicate_rate > 0.05:
            recommendations.append(
                f"Duplicate rate is {duplicate_rate:.1%}. Consider removing duplicate records to improve training quality."
            )
        
        # Consistency recommendations
        consistency_score = metrics["data_consistency_score"]
        if consistency_score < 0.8:
            recommendations.append(
                f"Data consistency is {consistency_score:.1%}. Review inconsistent fields and standardize data format."
            )
        
        return recommendations

# ==================== TEST EXECUTION FUNCTIONS ====================

def test_data_quality_with_sample_data():
    """Test data quality with sample medical data."""
    # Create sample data
    sample_data = [
        {
            "text": "Patient presents with headache and fever. Advised rest and hydration.",
            "medical_category": "general",
            "timestamp": "2025-01-01"
        },
        {
            "text": "Patient has diabetes and hypertension. Medication adjusted.",
            "medical_category": "chronic",
            "timestamp": "2025-01-02"
        },
        {
            "text": "Short",  # Too short
            "medical_category": "",
            "timestamp": None  # Missing timestamp
        }
    ]
    
    tester = DataQualityTester()
    results = tester.run_complete_validation(sample_data)
    
    # Verify results structure
    assert "total_records" in results
    assert "validation_results" in results
    assert "quality_metrics" in results
    assert "recommendations" in results
    
    print("✅ Data quality test completed successfully")
    print(f"📊 Quality Metrics: {results['quality_metrics']}")
    
    return results

def test_phi_compliance_validation():
    """Test PHI compliance validation."""
    sample_data = [
        {
            "text": "Patient John Doe with SSN 123-45-6789 presents with symptoms.",
            "medical_category": "general"
        },
        {
            "text": "Patient presents with common cold. Recommended rest.",
            "medical_category": "general"
        }
    ]
    
    validator = PHIComplianceValidator()
    results = validator.validate_phi_compliance(sample_data)
    
    assert "compliance_rate" in results
    assert "total_violations" in results
    assert results["total_violations"] >= 1  # Should detect PHI violation
    
    print("✅ PHI compliance test completed successfully")
    print(f"📊 PHI Compliance Rate: {results['compliance_rate']:.1%}")
    
    return results

def test_medical_accuracy_validation():
    """Test medical accuracy validation."""
    validator = MedicalAccuracyValidator()
    
    # Test valid medical content
    valid_text = "Patient presents with hypertension. Prescribed lisinopril 10mg daily."
    validation = validator.validate_medical_content(valid_text)
    
    assert "valid" in validation
    assert "issues" in validation
    
    # Test content that may need review
    questionable_text = "Miracle cure guaranteed! This drug will cure all diseases instantly!"
    validation = validator.validate_medical_content(questionable_text)
    
    assert not validation["valid"]  # Should be invalid
    
    print("✅ Medical accuracy test completed successfully")
    print(f"📊 Valid content: {validation['valid']}")
    
    return validation

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    logger.info("Starting Data Quality Tests")
    logger.info("=" * 50)
    
    try:
        # Run individual tests
        test_data_quality_with_sample_data()
        test_phi_compliance_validation()
        test_medical_accuracy_validation()
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 All data quality tests passed!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Data quality tests failed: {e}")
        sys.exit(1)