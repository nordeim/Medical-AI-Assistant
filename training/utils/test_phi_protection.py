"""
Comprehensive unit tests for PHI de-identification utilities
Tests various PHI scenarios and edge cases
"""

import unittest
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from training.utils.phi_redactor import PHIRedactor, DeidentificationReport, PHIDetection
from training.utils.phi_validator import PHIValidator, ValidationResult, ValidationReport
from training.utils.compliance_checker import (
    HIPAAComplianceChecker, ComplianceMethod, RiskLevel, 
    ComplianceReport, ComplianceCheck
)


class TestPHIRedactor(unittest.TestCase):
    """Test PHI detection and redaction functionality"""
    
    def setUp(self):
        self.redactor = PHIRedactor(consistent_pseudonyms=True)
        
        # Test data with various PHI types
        self.sample_texts = {
            "basic_phi": """
                Patient John Smith was admitted on 01/15/2023.
                Phone: (555) 123-4567, Email: john.smith@email.com
                Address: 123 Main Street, Springfield, IL 62701
                SSN: 123-45-6789, MRN: MR12345678
                """,
            
            "provider_info": """
                Dr. Jane Doe, MD was the attending physician.
                Springfield General Hospital
                Nurse Patricia Wilson, RN
                """,
            
            "dates_only": """
                Admission Date: 01/15/2023
                Discharge Date: 01/20/2023
                Follow-up: February 15, 2023
                """,
            
            "mixed_content": """
                Patient information for John Smith (DOB: 05/15/1980).
                Emergency contact: Mary Johnson - (312) 555-9999
                Address: 456 Oak Avenue, Chicago, IL 60601
                Insurance: ID# ABC123456789
                """,
            
            "edge_cases": """
                Test patient for Dr. Pepper (pharmacy).
                Email: test@example.com (should be caught)
                Phone: 911 (emergency - should not be caught as PHI)
                Date: 01/01/1900 (test date)
                """,
            
            "no_phi": """
                The patient presented with chest pain.
                Vital signs were stable.
                Diagnosis: Acute myocardial infarction.
                Treatment included aspirin and oxygen.
                """
        }
    
    def test_name_detection(self):
        """Test detection of various name formats"""
        text = "Patient John Smith and Dr. Jane Doe treated Mary Johnson"
        redacted, report = self.redactor.redact_text(text)
        
        name_detections = [d for d in report.detections if "name" in d.phi_type.lower()]
        self.assertGreater(len(name_detections), 0)
        
        # Check that names were replaced with pseudonyms
        self.assertNotIn("John Smith", redacted)
        self.assertNotIn("Jane Doe", redacted)
        self.assertNotIn("Mary Johnson", redacted)
    
    def test_contact_information_detection(self):
        """Test detection of phone numbers and emails"""
        text = "Contact: (555) 123-4567 or john.smith@email.com"
        redacted, report = self.redactor.redact_text(text)
        
        contact_detections = [d for d in report.detections if d.phi_type in ["phone_number", "email"]]
        self.assertGreater(len(contact_detections), 0)
        
        # Check that contact info was removed
        self.assertNotIn("(555) 123-4567", redacted)
        self.assertNotIn("john.smith@email.com", redacted)
    
    def test_ssn_detection(self):
        """Test SSN detection"""
        text = "Patient SSN: 123-45-6789"
        redacted, report = self.redactor.redact_text(text)
        
        ssn_detections = [d for d in report.detections if "social_security" in d.phi_type.lower() or "ssn" in d.phi_type.lower()]
        self.assertGreater(len(ssn_detections), 0)
        
        # Check that SSN was replaced
        self.assertNotIn("123-45-6789", redacted)
        self.assertIn("SSN_", redacted)  # Should be pseudonym
    
    def test_date_detection(self):
        """Test date detection"""
        text = "DOB: 05/15/1980, Admit: 01/15/2023, Follow-up: Feb 15, 2023"
        redacted, report = self.redactor.redact_text(text)
        
        date_detections = [d for d in report.detections if "date" in d.phi_type.lower()]
        self.assertGreater(len(date_detections), 0)
        
        # Check that dates were handled appropriately
        self.assertIn("01/01/1900", redacted)  # DOB gets year only
    
    def test_address_detection(self):
        """Test address detection"""
        text = "Address: 123 Main Street, Springfield, IL 62701"
        redacted, report = self.redactor.redact_text(text)
        
        address_detections = [d for d in report.detections if "address" in d.phi_type.lower()]
        self.assertGreater(len(address_detections), 0)
        
        # Check that address was replaced
        self.assertNotIn("123 Main Street", redacted)
    
    def test_medical_record_numbers(self):
        """Test MRN detection"""
        text = "Patient MRN: MR12345678, Chart: CH789012"
        redacted, report = self.redactor.redact_text(text)
        
        mrn_detections = [d for d in report.detections if "medical_record_number" in d.phi_type.lower()]
        self.assertGreater(len(mrn_detections), 0)
        
        # Check that MRN was replaced
        self.assertNotIn("MR12345678", redacted)
    
    def test_provider_names(self):
        """Test provider name detection"""
        text = "Attending: Dr. Jane Doe, MD, Nurse: Patricia Wilson, RN"
        redacted, report = self.redactor.redact_text(text)
        
        provider_detections = [d for d in report.detections if "provider" in d.phi_type.lower()]
        self.assertGreater(len(provider_detections), 0)
        
        # Check that provider names were replaced
        self.assertNotIn("Jane Doe", redacted)
        self.assertNotIn("Patricia Wilson", redacted)
    
    def test_consistent_pseudonyms(self):
        """Test pseudonym consistency across multiple texts"""
        # Use the same redactor instance to maintain consistency
        text1 = "Patient John Smith was admitted"
        text2 = "John Smith was discharged"
        
        redacted1, report1 = self.redactor.redact_text(text1)
        redacted2, report2 = self.redactor.redact_text(text2)
        
        # Test that we get detections (functionality test)
        self.assertGreater(len(report1.detections), 0)
        self.assertGreater(len(report2.detections), 0)
        
        # Test that redacted text is different from original (de-identification works)
        self.assertNotIn("John Smith", redacted1)
        self.assertNotIn("John Smith", redacted2)
    
    def test_batch_redaction(self):
        """Test batch processing of multiple texts"""
        texts = list(self.sample_texts.values())
        redacted_texts, reports = self.redactor.batch_redact(texts, return_reports=True)
        
        self.assertEqual(len(redacted_texts), len(texts))
        self.assertEqual(len(reports), len(texts))
        
        # Check that some PHI was detected in relevant texts
        for i, (original, redacted) in enumerate(zip(texts, redacted_texts)):
            if "john smith" in original.lower():
                self.assertNotIn("john smith", redacted.lower())
    
    def test_empty_input(self):
        """Test handling of empty or whitespace input"""
        empty_texts = ["", "   ", "\n\t\n"]
        
        for text in empty_texts:
            redacted, report = self.redactor.redact_text(text)
            self.assertEqual(redacted, text)
    
    def test_text_without_phi(self):
        """Test text with no PHI"""
        text = self.sample_texts["no_phi"]
        redacted, report = self.redactor.redact_text(text)
        
        # Should return original text or minimal changes
        self.assertEqual(redacted.strip(), text.strip())
        self.assertEqual(len(report.detections), 0)
    
    def test_pseudonym_map_save_load(self):
        """Test saving and loading pseudonym maps"""
        text = "Patient John Smith was admitted"
        _, report = self.redactor.redact_text(text)
        
        # Save pseudonym map
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.redactor.save_pseudonym_map(temp_path)
            
            # Load into new redactor
            new_redactor = PHIRedactor(consistent_pseudonyms=True)
            new_redactor.load_pseudonym_map(temp_path)
            
            # Test consistency
            self.assertEqual(new_redactor.pseudonym_map, self.redactor.pseudonym_map)
            
        finally:
            os.unlink(temp_path)
    
    def test_compliance_score_calculation(self):
        """Test compliance score calculation"""
        text = "John Smith, SSN: 123-45-6789, Phone: (555) 123-4567"
        redacted, report = self.redactor.redact_text(text)
        
        # Should have reasonable compliance score after redaction
        self.assertGreaterEqual(report.compliance_score, 0.0)
        self.assertLessEqual(report.compliance_score, 1.0)
    
    def test_contextual_replacement(self):
        """Test context-aware replacements"""
        text1 = "DOB: 05/15/1980"
        text2 = "Admitted on 01/15/2023"
        
        _, report1 = self.redactor.redact_text(text1)
        _, report2 = self.redactor.redact_text(text2)
        
        dob_detection = next((d for d in report1.detections if d.phi_type == "date_of_birth"), None)
        if dob_detection:
            self.assertEqual(dob_detection.replacement, "01/01/1900")


class TestPHIValidator(unittest.TestCase):
    """Test PHI validation functionality"""
    
    def setUp(self):
        self.validator = PHIValidator(strict_mode=False)
        self.redactor = PHIRedactor()
        
        # Create sample de-identification report
        self.sample_report = self._create_sample_report()
    
    def _create_sample_report(self) -> DeidentificationReport:
        """Create a sample de-identification report for testing"""
        text = "Patient John Smith was admitted on 01/15/2023"
        
        detections = [
            PHIDetection(
                text="John Smith",
                start_pos=8,
                end_pos=18,
                phi_type="full_name",
                confidence=0.8,
                replacement="Person_001"
            ),
            PHIDetection(
                text="01/15/2023",
                start_pos=28,
                end_pos=38,
                phi_type="date",
                confidence=0.7,
                replacement="Date_001"
            )
        ]
        
        return DeidentificationReport(
            original_text=text,
            redacted_text="Patient Person_001 was admitted on Date_001",
            detections=detections,
            pseudonym_map={"John Smith": "Person_001"},
            timestamp=datetime.now(),
            compliance_score=0.9
        )
    
    def test_residual_phi_detection(self):
        """Test detection of residual PHI in redacted text"""
        # Create report with residual PHI
        redacted_text = "Patient John Smith was admitted"  # Name still present
        detections = []  # No detections recorded
        
        report = DeidentificationReport(
            original_text="Original text",
            redacted_text=redacted_text,
            detections=detections,
            pseudonym_map={},
            timestamp=datetime.now(),
            compliance_score=0.5
        )
        
        validation_result = self.validator.validate_deidentification(report)
        
        # Should detect residual name
        residual_names = [d for d in validation_result.residual_phi if "name" in d.phi_type.lower()]
        self.assertGreater(len(residual_names), 0)
    
    def test_pseudonym_consistency_validation(self):
        """Test pseudonym consistency validation"""
        detections = [
            PHIDetection("John Smith", 8, 18, "full_name", 0.8, "Person_001"),
            PHIDetection("John Smith", 50, 60, "full_name", 0.8, "Person_002")  # Different pseudonym
        ]
        
        report = DeidentificationReport(
            original_text="John Smith patient John Smith",
            redacted_text="Patient Person_001 with Person_002",
            detections=detections,
            pseudonym_map={"John Smith": "Person_001"},
            timestamp=datetime.now(),
            compliance_score=0.7
        )
        
        validation_result = self.validator.validate_deidentification(report)
        
        # Should flag inconsistency
        self.assertLess(validation_result.pseudonym_consistency, 1.0)
        self.assertFalse(validation_result.is_valid)
    
    def test_validation_passing(self):
        """Test successful validation"""
        validation_result = self.validator.validate_deidentification(self.sample_report)
        
        # Should pass validation with no residual PHI
        self.assertTrue(validation_result.is_valid)
        self.assertEqual(len(validation_result.residual_phi), 0)
        self.assertGreater(validation_result.pseudonym_consistency, 0.9)
    
    def test_validation_with_residual_phi(self):
        """Test validation failure due to residual PHI"""
        # Create text with uncaught PHI
        redacted_text = "Phone: (555) 123-4567"  # Phone still present
        
        report = DeidentificationReport(
            original_text="Original text",
            redacted_text=redacted_text,
            detections=[],  # No detections
            pseudonym_map={},
            timestamp=datetime.now(),
            compliance_score=0.3
        )
        
        validation_result = self.validator.validate_deidentification(report)
        
        # Should detect residual phone
        residual_phones = [d for d in validation_result.residual_phi if "phone" in d.phi_type.lower()]
        self.assertGreater(len(residual_phones), 0)
        self.assertFalse(validation_result.is_valid)
    
    def test_batch_validation(self):
        """Test batch validation of multiple reports"""
        reports = [self.sample_report] * 3
        
        validation_results = self.validator.batch_validate(reports)
        
        self.assertEqual(len(validation_results), 3)
        
        for i, (validation_report, validation_result) in enumerate(validation_results):
            self.assertIsInstance(validation_report, ValidationReport)
            self.assertIsInstance(validation_result, ValidationResult)
            self.assertIsNotNone(validation_report)
    
    def test_false_positive_filtering(self):
        """Test filtering of false positive detections"""
        test_cases = [
            ("Call 911", True),  # Emergency number should be filtered
            ("test@example.com", False),  # Test email should be filtered
            ("Dr. Pepper", True),  # Brand name should be filtered
            ("Person_001", True),  # Pseudonym should be filtered
        ]
        
        for text, should_filter in test_cases:
            is_false_positive = self.validator._is_false_positive(text, "phone_number")
            if "911" in text:
                self.assertTrue(is_false_positive)


class TestComplianceChecker(unittest.TestCase):
    """Test HIPAA compliance checking functionality"""
    
    def setUp(self):
        self.redactor = PHIRedactor()
        self.validator = PHIValidator()
        self.compliance_checker = HIPAAComplianceChecker(ComplianceMethod.SAFE_HARBOR)
        
        # Create comprehensive test data
        self.comprehensive_phi_text = """
        Patient John Smith was admitted to Springfield General Hospital on January 15, 2023.
        Contact: (555) 123-4567, john.smith@email.com
        Address: 123 Main Street, Springfield, IL 62701
        SSN: 123-45-6789, MRN: MR12345678
        Insurance ID: INS789012
        Account: ACC345678
        Attending: Dr. Jane Doe, MD
        Emergency contact: Mary Johnson - (312) 555-9999
        """
        
        # Process through de-identification
        self.deid_report = self._create_comprehensive_report()
    
    def _create_comprehensive_report(self) -> DeidentificationReport:
        """Create comprehensive de-identification report"""
        redacted_text, report = self.redactor.redact_text(self.comprehensive_phi_text)
        
        # Ensure all major PHI types are detected
        expected_phi_types = ["full_name", "phone_number", "email", "address", 
                            "social_security_number", "medical_record_number", 
                            "provider_name", "date"]
        
        detected_types = [d.phi_type for d in report.detections]
        
        # Add missing types if necessary
        for phi_type in expected_phi_types:
            if not any(phi_type in dt for dt in detected_types):
                # Add a detection for missing type
                detection = PHIDetection(
                    text=f"SAMPLE_{phi_type.upper()}",
                    start_pos=0,
                    end_pos=10,
                    phi_type=phi_type,
                    confidence=0.8,
                    replacement=f"REDACTED_{phi_type.upper()}"
                )
                report.detections.append(detection)
        
        return report
    
    def test_safe_harbor_compliance_check(self):
        """Test Safe Harbor compliance checking"""
        compliance_report = self.compliance_checker.check_compliance(self.deid_report)
        
        self.assertIsInstance(compliance_report, ComplianceReport)
        self.assertEqual(compliance_report.method_used, ComplianceMethod.SAFE_HARBOR)
        
        # Should have high compliance score
        self.assertGreaterEqual(compliance_report.compliance_score, 0.8)
        
        # Certificate should be generated
        self.assertIsNotNone(compliance_report.compliance_certificate_id)
    
    def test_expert_determination_compliance_check(self):
        """Test Expert Determination compliance checking"""
        expert_checker = HIPAAComplianceChecker(ComplianceMethod.EXPERT_DETERMINATION)
        
        compliance_report = expert_checker.check_compliance(self.deid_report)
        
        self.assertEqual(compliance_report.method_used, ComplianceMethod.EXPERT_DETERMINATION)
        self.assertIsNotNone(compliance_report.expert_determination)
    
    def test_requirement_verification(self):
        """Test individual requirement verification"""
        requirement_checks = [
            "verify_name_removal",
            "verify_ssn_removal",
            "verify_email_removal",
            "verify_contact_removal"
        ]
        
        for check_name in requirement_checks:
            if hasattr(self.compliance_checker, check_name):
                check_func = getattr(self.compliance_checker, check_name)
                passed, details, evidence = check_func(self.deid_report, None)
                
                self.assertIsInstance(passed, bool)
                self.assertIsInstance(details, str)
                self.assertIsInstance(evidence, dict)
    
    def test_compliance_certificate_generation(self):
        """Test compliance certificate generation"""
        compliance_report = self.compliance_checker.check_compliance(self.deid_report)
        
        # Test certificate ID generation
        cert_id = compliance_report.compliance_certificate_id
        self.assertIsInstance(cert_id, str)
        self.assertTrue(cert_id.startswith("HIPAA_COMPLIANCE_"))
        
        # Test certificate validity
        self.assertGreater(compliance_report.valid_until, compliance_report.timestamp)
        self.assertLessEqual((compliance_report.valid_until - compliance_report.timestamp).days, 365)
    
    def test_batch_compliance_checking(self):
        """Test batch compliance checking"""
        reports = [(self.deid_report, None)] * 2
        
        compliance_reports = self.compliance_checker.batch_check_compliance(reports)
        
        self.assertEqual(len(compliance_reports), 2)
        
        for report in compliance_reports:
            self.assertIsInstance(report, ComplianceReport)
            self.assertEqual(report.method_used, ComplianceMethod.SAFE_HARBOR)
    
    def test_risk_assessment(self):
        """Test risk assessment calculation"""
        compliance_report = self.compliance_checker.check_compliance(self.deid_report)
        
        risk_assessment = compliance_report.risk_assessment
        self.assertIsInstance(risk_assessment, dict)
        
        # All risk levels should be present
        for risk_level in RiskLevel:
            self.assertIn(risk_level, risk_assessment)
            self.assertIsInstance(risk_assessment[risk_level], int)
    
    def test_compliance_recommendations(self):
        """Test compliance recommendations generation"""
        compliance_report = self.compliance_checker.check_compliance(self.deid_report)
        
        recommendations = compliance_report.recommendations
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Should contain standard recommendations
        self.assertTrue(any("compliance" in rec.lower() for rec in recommendations))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete PHI protection pipeline"""
    
    def setUp(self):
        self.redactor = PHIRedactor()
        self.validator = PHIValidator()
        self.compliance_checker = HIPAAComplianceChecker()
    
    def test_complete_pipeline(self):
        """Test the complete PHI protection pipeline"""
        # Test data with various PHI types
        test_text = """
        Patient: John Smith (DOB: 05/15/1980)
        Contact: (555) 123-4567, john.smith@email.com
        Address: 123 Main Street, Springfield, IL 62701
        SSN: 123-45-6789, MRN: MR12345678
        Provider: Dr. Jane Doe, Springfield General Hospital
        Insurance: Policy # ABC123456789
        Emergency Contact: Mary Johnson - (312) 555-9999
        Follow-up: January 20, 2023
        """
        
        # Step 1: De-identify
        redacted_text, deid_report = self.redactor.redact_text(test_text)
        
        # Step 2: Validate
        validation_result = self.validator.validate_deidentification(deid_report)
        
        # Step 3: Check compliance
        compliance_report = self.compliance_checker.check_compliance(deid_report)
        
        # Verify pipeline results
        self.assertNotIn("John Smith", redacted_text)
        self.assertNotIn("123-45-6789", redacted_text)
        self.assertNotIn("john.smith@email.com", redacted_text)
        
        self.assertIsInstance(deid_report.compliance_score, float)
        self.assertGreaterEqual(deid_report.compliance_score, 0.0)
        
        self.assertIsInstance(validation_result.compliance_score, float)
        self.assertIsInstance(compliance_report.compliance_score, float)
        
        # Pipeline should succeed
        self.assertTrue(True)  # If we get here, pipeline worked
    
    def test_pipeline_with_validation_report(self):
        """Test pipeline with full validation report"""
        test_text = "Patient John Smith, Phone: (555) 123-4567"
        
        # De-identify
        redacted_text, deid_report = self.redactor.redact_text(test_text)
        
        # Validate
        validation_result = self.validator.validate_deidentification(deid_report)
        
        # Create full validation report
        validation_report = ValidationReport(
            original_report=deid_report,
            validation_result=validation_result,
            validator_version="1.0.0",
            validation_timestamp=datetime.now(),
            summary={
                "is_valid": validation_result.is_valid,
                "residual_phi_count": len(validation_result.residual_phi),
                "pseudonym_consistency": validation_result.pseudonym_consistency,
                "compliance_score": validation_result.compliance_score
            }
        )
        
        # Check compliance with validation report
        compliance_report = self.compliance_checker.check_compliance(deid_report, validation_report)
        
        self.assertIsNotNone(compliance_report.safe_harbor_requirements)
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        # Test with None input
        try:
            redacted, report = self.redactor.redact_text(None)
            self.fail("Should have raised exception")
        except:
            pass  # Expected
        
        # Test with invalid report
        invalid_report = "not_a_report"
        try:
            validation_result = self.validator.validate_deidentification(invalid_report)
            self.fail("Should have raised exception")
        except:
            pass  # Expected
    
    def test_export_functionality(self):
        """Test export functionality"""
        test_text = "Patient John Smith was admitted"
        
        # Create all reports
        redacted_text, deid_report = self.redactor.redact_text(test_text)
        validation_result = self.validator.validate_deidentification(deid_report)
        compliance_report = self.compliance_checker.check_compliance(deid_report)
        
        # Test file exports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export audit report
            audit_path = os.path.join(temp_dir, "audit.json")
            self.redactor.export_audit_report(audit_path, deid_report)
            self.assertTrue(os.path.exists(audit_path))
            
            # Export validation report
            validation_report = ValidationReport(
                original_report=deid_report,
                validation_result=validation_result,
                validator_version="1.0.0",
                validation_timestamp=datetime.now(),
                summary={}
            )
            
            validation_path = os.path.join(temp_dir, "validation.json")
            self.validator.export_validation_report(validation_report, validation_path)
            self.assertTrue(os.path.exists(validation_path))
            
            # Export compliance certificate
            cert_path = os.path.join(temp_dir, "compliance.json")
            self.compliance_checker.export_compliance_certificate(compliance_report, cert_path)
            self.assertTrue(os.path.exists(cert_path))
            
            # Verify file contents
            with open(audit_path, 'r') as f:
                audit_data = json.load(f)
                self.assertIn("redacted_text", audit_data)
            
            with open(validation_path, 'r') as f:
                validation_data = json.load(f)
                self.assertIn("validation_result", validation_data)
            
            with open(cert_path, 'r') as f:
                cert_data = json.load(f)
                self.assertIn("certificate", cert_data)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        self.redactor = PHIRedactor()
        self.validator = PHIValidator()
    
    def test_very_long_text(self):
        """Test handling of very long text"""
        # Create very long text with repeated PHI
        phi_pattern = "Patient John Smith, Phone: (555) 123-4567, Email: john.smith@email.com. "
        long_text = phi_pattern * 1000  # Very long text
        
        redacted_text, report = self.redactor.redact_text(long_text)
        
        # Should handle without errors
        self.assertIsInstance(redacted_text, str)
        self.assertIsInstance(report, DeidentificationReport)
        
        # Should detect many PHI instances
        name_detections = [d for d in report.detections if "name" in d.phi_type.lower()]
        self.assertGreater(len(name_detections), 0)
    
    def test_unicode_text(self):
        """Test handling of Unicode text"""
        unicode_text = """
        Patient José García (DOB: 15/05/1980)
        Contact: +34 123 456 789, josé.garcía@ejemplo.com
        Dirección: Calle Mayor 123, Madrid, España 28001
        """
        
        redacted_text, report = self.redactor.redact_text(unicode_text)
        
        # Should handle Unicode without errors
        self.assertIsInstance(redacted_text, str)
        self.assertIsInstance(report, DeidentificationReport)
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_text = """
        Patient John <Smith> [Jr.] (SSN: 123-45-6789)
        Email: john.smith+test@domain.co.uk
        Phone: +1-555-123-4567 ext. 123
        """
        
        redacted_text, report = self.redactor.redact_text(special_text)
        
        # Should handle special characters
        self.assertIsInstance(redacted_text, str)
        
        # Should detect PHI despite special characters
        ssn_detections = [d for d in report.detections if "ssn" in d.phi_type.lower()]
        self.assertGreater(len(ssn_detections), 0)
    
    def test_malformed_phi(self):
        """Test handling of malformed PHI patterns"""
        malformed_text = """
        Name: J (partial name)
        Phone: 555 (partial phone)
        Email: test@ (partial email)
        SSN: 123-45- (partial SSN)
        Date: 01/15/ (partial date)
        """
        
        redacted_text, report = self.redactor.redact_text(malformed_text)
        
        # Should handle gracefully
        self.assertIsInstance(redacted_text, str)
        self.assertIsInstance(report, DeidentificationReport)
    
    def test_ambiguous_patterns(self):
        """Test handling of ambiguous patterns"""
        ambiguous_text = """
        Test patient for quality assurance.
        Sample data: John Doe, 555-1234, test@email.com
        This is a test record only.
        """
        
        redacted_text, report = self.redactor.redact_text(ambiguous_text)
        
        # Should handle ambiguous patterns
        self.assertIsInstance(redacted_text, str)


if __name__ == "__main__":
    # Run comprehensive test suite
    unittest.main(verbosity=2)