"""Comprehensive unit tests for data validation and quality assurance utilities."""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the modules to test
from training.utils.data_validator import (
    DataValidator, MedicalDataValidator, ValidationConfig, ValidationResult
)
from training.utils.validation_reporter import ValidationReporter, BatchValidationReporter


class TestValidationConfig(unittest.TestCase):
    """Test validation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ValidationConfig()
        
        self.assertIn('conversation_id', config.required_fields)
        self.assertIn('user_input', config.required_fields)
        self.assertEqual(config.age_range, (0, 150))
        self.assertIn('emergency', config.valid_triage_levels)
        self.assertEqual(config.min_text_length, 10)
        self.assertEqual(config.duplicate_similarity_threshold, 0.95)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ValidationConfig(
            required_fields=['id', 'text'],
            age_range=(18, 100),
            min_text_length=20
        )
        
        self.assertEqual(config.required_fields, ['id', 'text'])
        self.assertEqual(config.age_range, (18, 100))
        self.assertEqual(config.min_text_length, 20)


class TestDataValidator(unittest.TestCase):
    """Test the main data validator."""
    
    def setUp(self):
        """Set up test data."""
        self.config = ValidationConfig()
        self.validator = DataValidator(self.config)
        
        # Sample valid data
        self.valid_data = [
            {
                'conversation_id': 'conv_001',
                'user_input': 'I have a severe headache and fever.',
                'assistant_response': 'I understand you are experiencing headache and fever. This could indicate several conditions. Have you measured your temperature?',
                'timestamp': '2023-01-01T10:00:00Z',
                'age': 30,
                'gender': 'female',
                'triage_level': 'urgent',
                'symptoms': 'severe headache, fever, nausea'
            },
            {
                'conversation_id': 'conv_002',
                'user_input': 'Chest pain after exercise',
                'assistant_response': 'Chest pain after physical activity warrants immediate attention. Please seek emergency care immediately.',
                'timestamp': '2023-01-01T11:00:00Z',
                'age': 45,
                'gender': 'male',
                'triage_level': 'emergency',
                'symptoms': 'chest pain, shortness of breath'
            }
        ]
        
        # Sample invalid data
        self.invalid_data = [
            {
                'conversation_id': 'conv_003',
                'user_input': '',  # Empty text
                'assistant_response': 'Response',
                'timestamp': '2023-01-01T12:00:00Z',
                'age': 200,  # Invalid age
                'gender': 'invalid',
                'triage_level': 'invalid_level',
                'symptoms': 'symptoms'
            }
        ]
    
    def test_validate_dataset_valid(self):
        """Test validation of valid dataset."""
        result = self.validator.validate_dataset(self.valid_data)
        
        self.assertTrue(result.is_valid)
        self.assertGreater(result.score, 0.7)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_dataset_invalid(self):
        """Test validation of invalid dataset."""
        result = self.validator.validate_dataset(self.invalid_data)
        
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)
    
    def test_validate_data_integrity(self):
        """Test data integrity validation."""
        df = pd.DataFrame(self.valid_data)
        result = ValidationResult(is_valid=True)
        
        self.validator._validate_data_integrity(df, result)
        
        # Should not have errors for valid data
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_medical_data(self):
        """Test medical data validation."""
        df = pd.DataFrame(self.valid_data)
        result = ValidationResult(is_valid=True)
        
        self.validator._validate_medical_data(df, result)
        
        # Should not have errors for valid medical data
        self.assertEqual(len(result.errors), 0)
    
    def test_detect_duplicates(self):
        """Test duplicate detection."""
        data_with_duplicates = self.valid_data + [
            self.valid_data[0].copy()  # Exact duplicate
        ]
        
        df = pd.DataFrame(data_with_duplicates)
        duplicates = self.validator._detect_duplicates(df)
        
        self.assertGreater(len(duplicates), 0)
    
    def test_detect_phi(self):
        """Test PHI detection."""
        data_with_phi = self.valid_data + [
            {
                'conversation_id': 'conv_phi',
                'user_input': 'My SSN is 123-45-6789 and email is test@example.com',
                'assistant_response': 'Response',
                'timestamp': '2023-01-01T13:00:00Z',
                'age': 30,
                'gender': 'female',
                'triage_level': 'non-urgent',
                'symptoms': 'headache'
            }
        ]
        
        df = pd.DataFrame(data_with_phi)
        phi_detections = self.validator._detect_phi(df)
        
        self.assertGreater(len(phi_detections), 0)
    
    def test_outlier_detection(self):
        """Test outlier detection."""
        data_with_outliers = self.valid_data + [
            {
                'conversation_id': 'conv_outlier',
                'user_input': 'Normal input',
                'assistant_response': 'Normal response',
                'timestamp': '2023-01-01T14:00:00Z',
                'age': 999,  # Outlier
                'gender': 'female',
                'triage_level': 'non-urgent',
                'symptoms': 'symptoms'
            }
        ]
        
        df = pd.DataFrame(data_with_outliers)
        outliers = self.validator._detect_outliers(df[['age']])
        
        self.assertIn('age', outliers)
    
    def test_text_quality_calculation(self):
        """Test text quality calculation."""
        good_text = "This is a well-structured medical consultation with appropriate details and clear communication."
        bad_text = "x" * 5  # Too short
        
        good_score = self.validator._calculate_text_score(good_text)
        bad_score = self.validator._calculate_text_score(bad_text)
        
        self.assertGreater(good_score, bad_score)
        self.assertGreater(good_score, 0.5)
    
    def test_readability_calculation(self):
        """Test readability calculation."""
        simple_text = "The cat sat on the mat."
        complex_text = "The feline quadruped established a resting position upon the woven floor covering."
        
        simple_readability = self.validator._calculate_readability(simple_text)
        complex_readability = self.validator._calculate_readability(complex_text)
        
        # Simple text should have higher readability score
        self.assertGreater(simple_readability, complex_readability)
    
    def test_text_similarity(self):
        """Test text similarity calculation."""
        text1 = "I have a headache and fever"
        text2 = "I am experiencing headache and high temperature"
        text3 = "The weather is sunny today"
        
        similarity_same = self.validator._calculate_text_similarity(text1, text1)
        similarity_similar = self.validator._calculate_text_similarity(text1, text2)
        similarity_different = self.validator._calculate_text_similarity(text1, text3)
        
        self.assertEqual(similarity_same, 1.0)
        self.assertGreater(similarity_similar, 0.5)
        self.assertLess(similarity_different, 0.3)
    
    def test_class_balance_analysis(self):
        """Test class balance analysis."""
        # Create imbalanced data
        data = [
            {'triage_level': 'emergency'} for _ in range(10)
        ] + [
            {'triage_level': 'non-urgent'} for _ in range(1)
        ]
        
        df = pd.DataFrame(data)
        balance = self.validator._analyze_class_balance(df, 'triage_level')
        
        self.assertEqual(balance['total_classes'], 2)
        self.assertGreater(balance['imbalance_ratio'], 5.0)
    
    def test_missing_data_analysis(self):
        """Test missing data pattern analysis."""
        data = [
            {'field1': 'value1', 'field2': None, 'field3': 'value3'},
            {'field1': None, 'field2': 'value2', 'field3': None},
            {'field1': 'value1', 'field2': 'value2', 'field3': 'value3'}
        ]
        
        df = pd.DataFrame(data)
        missing_patterns = self.validator._analyze_missing_data(df)
        
        self.assertIn('total_missing_percentage', missing_patterns)
        self.assertGreater(missing_patterns['total_missing_percentage'], 0)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        # Perfect result
        perfect_result = ValidationResult(is_valid=True, score=1.0)
        perfect_result.errors = []
        perfect_result.warnings = []
        
        # Poor result
        poor_result = ValidationResult(is_valid=False, score=0.3)
        poor_result.errors = ['Critical error 1', 'Critical error 2']
        poor_result.warnings = ['Warning 1', 'Warning 2', 'Warning 3']
        
        perfect_score = self.validator._calculate_overall_score(perfect_result)
        poor_score = self.validator._calculate_overall_score(poor_result)
        
        self.assertGreater(perfect_score, poor_score)
        self.assertLessEqual(perfect_score, 1.0)
        self.assertGreaterEqual(poor_score, 0.0)
    
    def test_validate_single_record(self):
        """Test single record validation."""
        result = self.validator.validate_single_record(self.valid_data[0])
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
    
    def test_batch_validate(self):
        """Test batch validation."""
        batches = [
            self.valid_data[:1],
            self.valid_data[1:2]
        ]
        
        results = self.validator.batch_validate(batches)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ValidationResult)


class TestMedicalDataValidator(unittest.TestCase):
    """Test the medical-specific data validator."""
    
    def setUp(self):
        """Set up medical validator."""
        self.medical_validator = MedicalDataValidator()
    
    def test_medical_abbreviation_recognition(self):
        """Test medical abbreviation handling."""
        data = [
            {
                'conversation_id': 'conv_med',
                'user_input': 'I have high BP and elevated HR',
                'assistant_response': 'Your blood pressure and heart rate are elevated. We should monitor your O2 levels.',
                'timestamp': '2023-01-01T15:00:00Z',
                'age': 50,
                'gender': 'male',
                'triage_level': 'urgent',
                'symptoms': 'high blood pressure, rapid heart rate'
            }
        ]
        
        result = self.medical_validator.validate_dataset(data)
        
        # Should have warnings about abbreviations
        self.assertGreater(len(result.warnings), 0)
    
    def test_medical_accuracy_calculation(self):
        """Test enhanced medical accuracy calculation."""
        good_medical_response = "I recommend you seek immediate medical attention for your chest pain symptoms. This could be serious."
        
        df = pd.DataFrame([{
            'assistant_response': good_medical_response,
            'user_input': 'I have chest pain'
        }])
        
        accuracy = self.medical_validator._calculate_medical_accuracy(df)
        
        self.assertIn('avg_medical_term_usage', accuracy)
        self.assertGreater(accuracy['avg_medical_term_usage'], 0)


class TestValidationReporter(unittest.TestCase):
    """Test validation reporting functionality."""
    
    def setUp(self):
        """Set up reporter."""
        self.reporter = ValidationReporter()
        
        # Create a sample validation result
        self.sample_result = ValidationResult(
            is_valid=True,
            score=0.85,
            errors=[],
            warnings=['Sample warning'],
            metrics={
                'quality': {
                    'avg_text_quality': 0.8,
                    'coherence': {'avg_coherence': 0.75}
                },
                'class_balance': {
                    'imbalance_ratio': 2.5,
                    'entropy': 1.2
                }
            }
        )
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_html_report_generation(self):
        """Test HTML report generation."""
        output_path = os.path.join(self.temp_dir, 'test_report.html')
        
        result = self.reporter.generate_html_report(
            self.sample_result, 
            output_path, 
            dataset_name='Test Dataset'
        )
        
        self.assertTrue(os.path.exists(result))
        
        # Check that file contains expected content
        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Test Dataset', content)
            self.assertIn('85.0', content)  # Score
            self.assertIn('Sample warning', content)
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        output_path = os.path.join(self.temp_dir, 'test_report.json')
        
        result = self.reporter.generate_json_report(
            self.sample_result, 
            output_path,
            {'record_count': 100}
        )
        
        self.assertTrue(os.path.exists(result))
        
        # Check JSON content
        with open(result, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn('summary', data)
            self.assertEqual(data['summary']['is_valid'], True)
            self.assertEqual(data['summary']['score'], 0.85)
    
    def test_csv_summary_generation(self):
        """Test CSV summary generation."""
        output_path = os.path.join(self.temp_dir, 'test_summary.csv')
        
        result = self.reporter.generate_csv_summary(
            self.sample_result, 
            output_path, 
            record_count=100
        )
        
        self.assertTrue(os.path.exists(result))
        
        # Check CSV content
        df = pd.read_csv(output_path)
        self.assertEqual(len(df), 6)  # 6 metrics
        self.assertIn('Overall Score', df['Metric'].values)
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        # Test with errors
        error_result = ValidationResult(is_valid=False, score=0.5)
        error_result.errors = ['Missing required fields']
        
        recommendations, has_critical = self.reporter._generate_recommendations(error_result)
        
        self.assertGreater(len(recommendations), 0)
        self.assertTrue(has_critical)
        
        # Test with warnings
        warning_result = ValidationResult(is_valid=True, score=0.7)
        warning_result.warnings = ['High missing data percentage']
        
        recommendations, has_critical = self.reporter._generate_recommendations(warning_result)
        
        self.assertGreater(len(recommendations), 0)
        self.assertFalse(has_critical)
    
    def test_quality_grade_calculation(self):
        """Test quality grade calculation."""
        self.assertEqual(self.reporter._get_quality_grade(0.95), 'A+')
        self.assertEqual(self.reporter._get_quality_grade(0.90), 'A')
        self.assertEqual(self.reporter._get_quality_grade(0.75), 'C+')
        self.assertEqual(self.reporter._get_quality_grade(0.60), 'D')
        self.assertEqual(self.reporter._get_quality_grade(0.50), 'F')


class TestBatchValidationReporter(unittest.TestCase):
    """Test batch validation reporting."""
    
    def setUp(self):
        """Set up batch reporter."""
        self.batch_reporter = BatchValidationReporter()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_summary_report(self):
        """Test batch summary report generation."""
        # Create sample results
        results = [
            ValidationResult(is_valid=True, score=0.9),
            ValidationResult(is_valid=True, score=0.85),
            ValidationResult(is_valid=False, score=0.6)
        ]
        
        dataset_names = ['Dataset1', 'Dataset2', 'Dataset3']
        output_path = os.path.join(self.temp_dir, 'batch_report.html')
        
        result = self.batch_reporter.generate_batch_summary_report(
            results, output_path, dataset_names
        )
        
        self.assertTrue(os.path.exists(result))
        
        # Check content
        with open(result, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('Batch Validation Summary', content)
            self.assertIn('3', content)  # Total datasets


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete validation pipeline."""
    
    def setUp(self):
        """Set up integration test data."""
        self.config = ValidationConfig()
        self.validator = MedicalDataValidator(self.config)
        self.reporter = ValidationReporter(self.config)
        
        # Create comprehensive test dataset
        self.test_data = []
        
        # Add valid records
        for i in range(50):
            self.test_data.append({
                'conversation_id': f'conv_{i:03d}',
                'user_input': f'I have been experiencing {["headache", "fever", "cough", "pain"][i % 4]} for the past day.',
                'assistant_response': f'I understand your concerns about your symptoms. Based on what you described, I recommend you monitor your condition and consider seeking medical attention if symptoms worsen.',
                'timestamp': f'2023-01-{(i % 30) + 1:02d}T{(i % 24):02d}:00:00Z',
                'age': 20 + (i % 50),
                'gender': ['male', 'female', 'other'][i % 3],
                'triage_level': ['emergency', 'urgent', 'non-urgent', 'advisory'][i % 4],
                'symptoms': f'Symptoms include {["headache", "fever", "cough", "nausea"][i % 4]} and fatigue'
            })
        
        # Add some problematic records
        self.test_data.extend([
            {
                'conversation_id': 'problem_1',
                'user_input': '',  # Empty
                'assistant_response': 'OK',
                'timestamp': '2023-01-01T10:00:00Z',
                'age': 999,  # Invalid
                'gender': 'unknown',
                'triage_level': 'invalid',
                'symptoms': 'x'
            },
            {
                'conversation_id': 'problem_2',
                'user_input': 'My SSN is 123-45-6789 and I need help',
                'assistant_response': 'I can help you with your medical concerns.',
                'timestamp': '2023-01-01T11:00:00Z',
                'age': 30,
                'gender': 'female',
                'triage_level': 'non-urgent',
                'symptoms': 'general concern'
            }
        ])
    
    def test_full_validation_pipeline(self):
        """Test the complete validation and reporting pipeline."""
        # Run validation
        result = self.validator.validate_dataset(self.test_data)
        
        # Verify validation worked
        self.assertIsInstance(result, ValidationResult)
        self.assertGreater(result.score, 0)
        
        # Generate reports
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # HTML report
            html_path = os.path.join(temp_dir, 'integration_test.html')
            self.reporter.generate_html_report(result, html_path, 'Integration Test Dataset')
            self.assertTrue(os.path.exists(html_path))
            
            # JSON report
            json_path = os.path.join(temp_dir, 'integration_test.json')
            self.reporter.generate_json_report(result, json_path)
            self.assertTrue(os.path.exists(json_path))
            
            # CSV summary
            csv_path = os.path.join(temp_dir, 'integration_test.csv')
            self.reporter.generate_csv_summary(result, csv_path, len(self.test_data))
            self.assertTrue(os.path.exists(csv_path))
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_validation_performance(self):
        """Test validation performance on larger datasets."""
        # Create a larger dataset
        large_data = []
        for i in range(1000):
            large_data.append({
                'conversation_id': f'large_conv_{i}',
                'user_input': f'Patient reports symptoms {i % 10}',
                'assistant_response': f'Response to symptoms {i % 10}',
                'timestamp': f'2023-01-01T10:00:00Z',
                'age': 30,
                'gender': 'female',
                'triage_level': 'non-urgent',
                'symptoms': f'Symptoms {i % 10}'
            })
        
        # Measure validation time
        start_time = datetime.now()
        result = self.validator.validate_dataset(large_data)
        end_time = datetime.now()
        
        # Should complete within reasonable time (adjust threshold as needed)
        duration = (end_time - start_time).total_seconds()
        self.assertLess(duration, 10)  # Should complete within 10 seconds
        
        # Verify result quality
        self.assertIsInstance(result, ValidationResult)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up edge case tests."""
        self.validator = DataValidator()
    
    def test_empty_dataset(self):
        """Test validation of empty dataset."""
        result = self.validator.validate_dataset([])
        
        self.assertIsInstance(result, ValidationResult)
        # Empty dataset might be valid or invalid depending on requirements
    
    def test_dataset_with_missing_fields(self):
        """Test validation when required fields are missing."""
        incomplete_data = [{'conversation_id': 'conv_001'}]
        
        result = self.validator.validate_dataset(incomplete_data)
        
        # Should have errors about missing fields
        self.assertGreater(len(result.errors), 0)
    
    def test_invalid_data_types(self):
        """Test validation with invalid data types."""
        invalid_data = [
            {
                'conversation_id': 'conv_001',
                'user_input': 'Normal text',
                'assistant_response': 'Normal response',
                'timestamp': '2023-01-01T10:00:00Z',
                'age': 'not_a_number',  # Invalid type
                'gender': 'female',
                'triage_level': 'urgent',
                'symptoms': 'symptoms'
            }
        ]
        
        result = self.validator.validate_dataset(invalid_data)
        
        # Should handle type errors gracefully
        self.assertIsInstance(result, ValidationResult)
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        unicode_data = [
            {
                'conversation_id': 'conv_unicode',
                'user_input': 'Patient reports: Tosse, febre e dor de cabeça 🚑',
                'assistant_response': 'Entendo seus sintomas. Recomendo procurar atendimento médico.',
                'timestamp': '2023-01-01T10:00:00Z',
                'age': 30,
                'gender': 'female',
                'triage_level': 'urgent',
                'symptoms': 'tosse, febre, dor de cabeça'
            }
        ]
        
        result = self.validator.validate_dataset(unicode_data)
        
        # Should handle unicode gracefully
        self.assertIsInstance(result, ValidationResult)
        self.assertGreaterEqual(result.score, 0)
    
    def test_very_long_text(self):
        """Test handling of very long text fields."""
        long_text_data = [
            {
                'conversation_id': 'conv_long',
                'user_input': 'Very long description: ' + 'x' * 10000,
                'assistant_response': 'Very long response: ' + 'y' * 10000,
                'timestamp': '2023-01-01T10:00:00Z',
                'age': 30,
                'gender': 'female',
                'triage_level': 'non-urgent',
                'symptoms': 'Very long symptom description: ' + 'z' * 5000
            }
        ]
        
        result = self.validator.validate_dataset(long_text_data)
        
        # Should handle long text gracefully
        self.assertIsInstance(result, ValidationResult)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)