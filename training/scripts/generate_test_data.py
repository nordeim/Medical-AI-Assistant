#!/usr/bin/env python3
"""
Test Data Generation for Quality Assurance
==========================================

Generates synthetic test data for comprehensive testing of the Medical AI Training System.
Features:
- Medical dialogue generation
- PHI anonymization testing
- Data quality validation
- Performance testing datasets
- Edge case scenarios
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalTestDataGenerator:
    """Generate synthetic medical test data for QA purposes."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
        # Medical terminology and patterns
        self.medical_terms = {
            'conditions': [
                'hypertension', 'diabetes mellitus', 'asthma', 'pneumonia', 'influenza',
                'migraine', 'arthritis', 'depression', 'anxiety', 'obesity',
                'hyperlipidemia', 'hypothyroidism', 'gastroesophageal reflux disease',
                'chronic obstructive pulmonary disease', 'coronary artery disease'
            ],
            'medications': [
                'metformin', 'lisinopril', 'amlodipine', 'atorvastatin', 'omeprazole',
                'albuterol', 'warfarin', 'insulin glargine', 'levothyroxine',
                'sertraline', 'lorazepam', 'hydrochlorothiazide'
            ],
            'procedures': [
                'physical examination', 'blood test', 'chest X-ray', 'ECG',
                'endoscopy', 'biopsy', 'MRI', 'CT scan', 'ultrasound',
                'stress test', 'echocardiogram', 'colonoscopy'
            ],
            'symptoms': [
                'chest pain', 'shortness of breath', 'fatigue', 'fever',
                'headache', 'cough', 'nausea', 'vomiting', 'diarrhea',
                'abdominal pain', 'joint pain', 'rash'
            ],
            'body_parts': [
                'heart', 'lungs', 'liver', 'kidneys', 'brain', 'stomach',
                'intestines', 'pancreas', 'thyroid', 'blood vessels',
                'joints', 'muscles', 'skin'
            ]
        }
        
        # PHI patterns for testing
        self.phi_patterns = {
            'names': ['John Doe', 'Jane Smith', 'Robert Johnson', 'Mary Wilson'],
            'ssn': ['123-45-6789', '987-65-4321', '111-22-3333'],
            'phone': ['555-123-4567', '555-987-6543', '555-555-5555'],
            'email': ['john.doe@email.com', 'jane.smith@hospital.org'],
            'addresses': ['123 Main St, City, State 12345', '456 Oak Ave, Town, State 67890'],
            'mrn': ['MRN123456', 'MRN789012', 'MRN345678'],
            'dob': ['01/15/1980', '03/22/1975', '07/08/1990']
        }
        
        # Dialogue templates
        self.dialogue_templates = [
            "Patient {name} presents with {symptom}. {age} year old {gender}.",
            "Chief complaint: {symptom}. Patient has history of {condition}.",
            "HPI: {symptom} for {duration}. Associated with {associated_symptom}.",
            "PMH: Significant for {condition}. Current medications: {medication}.",
            "Assessment: {primary_diagnosis}. Plan includes {treatment}.",
            "Follow-up in {followup_time}. Monitor for {monitoring_point}.",
            "Patient denies {negative_symptom}. Vital signs stable.",
            "Review of systems positive for {ros_symptom}. Negative for {negative_ros}.",
            "Physical examination reveals {exam_findings}.",
            "Diagnostic testing ordered: {tests}. Results pending."
        ]
    
    def generate_basic_medical_dialogues(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate basic medical dialogue data."""
        dialogues = []
        genders = ['male', 'female']
        
        for i in range(count):
            # Random selection from categories
            condition = random.choice(self.medical_terms['conditions'])
            symptom = random.choice(self.medical_terms['symptoms'])
            medication = random.choice(self.medical_terms['medications'])
            procedure = random.choice(self.medical_terms['procedures'])
            body_part = random.choice(self.medical_terms['body_parts'])
            
            # Generate patient info
            name = f"Patient_{i:03d}"
            age = random.randint(18, 90)
            gender = random.choice(genders)
            
            # Create dialogue
            dialogue_parts = [
                f"Patient {name} is a {age} year old {gender}.",
                f"Presenting with {symptom} for 2 days.",
                f"History of {condition}.",
                f"Current medication: {medication}.",
                f"Physical examination of {body_part} within normal limits.",
                f"Ordered {procedure} for further evaluation."
            ]
            
            dialogue_text = " ".join(dialogue_parts)
            
            dialogues.append({
                'id': f"dialogue_{i:03d}",
                'text': dialogue_text,
                'medical_category': self._classify_medical_category(condition),
                'symptoms': [symptom],
                'conditions': [condition],
                'medications': [medication],
                'procedures': [procedure],
                'patient_age': age,
                'patient_gender': gender,
                'created_at': datetime.now().isoformat(),
                'data_source': 'synthetic_generation',
                'quality_grade': random.choice(['A', 'B', 'C'])
            })
        
        return dialogues
    
    def generate_phi_test_data(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate test data with PHI for compliance testing."""
        phi_dialogues = []
        
        for i in range(count):
            # Create PHI-containing text
            phi_pattern = random.choice(list(self.phi_patterns.keys()))
            phi_value = random.choice(self.phi_patterns[phi_pattern])
            
            # Generate medical text with embedded PHI
            symptom = random.choice(self.medical_terms['symptoms'])
            condition = random.choice(self.medical_terms['conditions'])
            
            dialogue_text = f"""
            Patient {phi_value} presents with {symptom}.
            This is a test case for PHI detection and anonymization.
            Medical condition: {condition}.
            """.strip()
            
            phi_dialogues.append({
                'id': f"phi_test_{i:03d}",
                'text': dialogue_text,
                'phi_type': phi_pattern,
                'phi_value': phi_value,
                'medical_category': 'phi_compliance_test',
                'should_contain_phi': True,
                'anonymization_required': True,
                'compliance_test': True
            })
        
        return phi_dialogues
    
    def generate_edge_case_data(self, count: int = 30) -> List[Dict[str, Any]]:
        """Generate edge case data for stress testing."""
        edge_cases = []
        
        for i in range(count):
            case_type = random.choice([
                'empty_text', 'very_long_text', 'special_characters',
                'non_medical_text', 'mixed_languages', 'duplicate_content',
                'invalid_format', 'missing_fields', 'very_short_text'
            ])
            
            if case_type == 'empty_text':
                text = ""
            elif case_type == 'very_long_text':
                text = "This is a very long medical dialogue. " * 500  # ~10,000 characters
            elif case_type == 'special_characters':
                text = "Patient @#$%^&*() presents with !@#$% symptoms."
            elif case_type == 'non_medical_text':
                text = "The weather is nice today. I had coffee for breakfast."
            elif case_type == 'mixed_languages':
                text = "Patient presents with síntomas médicos. La condición es importante."
            elif case_type == 'duplicate_content':
                text = "Duplicate test case for duplicate detection."
            elif case_type == 'invalid_format':
                text = "{\"invalid\": json \"format\"}"
            elif case_type == 'missing_fields':
                text = "Partial text"
            elif case_type == 'very_short_text':
                text = "Short"
            
            edge_cases.append({
                'id': f"edge_case_{i:03d}",
                'text': text,
                'case_type': case_type,
                'medical_category': 'edge_case',
                'stress_test': True,
                'expected_validation': case_type in ['very_long_text', 'special_characters', 'non_medical_text']
            })
        
        return edge_cases
    
    def generate_performance_test_data(self, size: str = 'medium') -> List[Dict[str, Any]]:
        """Generate large datasets for performance testing."""
        size_configs = {
            'small': 1000,
            'medium': 10000,
            'large': 50000,
            'xlarge': 100000
        }
        
        count = size_configs.get(size, 10000)
        logger.info(f"Generating {count} records for performance testing")
        
        performance_data = []
        
        for i in range(count):
            # Generate varied-length texts
            num_sentences = random.randint(1, 10)
            sentences = []
            
            for _ in range(num_sentences):
                symptom = random.choice(self.medical_terms['symptoms'])
                condition = random.choice(self.medical_terms['conditions'])
                sentence = f"Patient presents with {symptom}. Diagnosed with {condition}."
                sentences.append(sentence)
            
            text = " ".join(sentences)
            
            performance_data.append({
                'id': f"perf_{i:08d}",
                'text': text,
                'medical_category': random.choice(['general', 'cardiology', 'neurology', 'internal']),
                'performance_test': True,
                'record_size': len(text),
                'created_for': 'performance_benchmarking'
            })
        
        return performance_data
    
    def generate_consistency_test_data(self, count: int = 100) -> List[Dict[str, Any]]:
        """Generate data with known patterns for consistency testing."""
        consistency_data = []
        
        # Create consistent patterns across multiple records
        base_conditions = ['diabetes mellitus', 'hypertension', 'asthma']
        base_medications = ['metformin', 'lisinopril', 'albuterol']
        
        for i in range(count):
            # Ensure some consistency in medical categories
            condition = random.choice(base_conditions)
            medication = base_medications[base_conditions.index(condition)]  # Consistent pairing
            
            text = f"Patient presents with diabetes-related symptoms. Medication: {medication}."
            
            consistency_data.append({
                'id': f"consistency_{i:03d}",
                'text': text,
                'medical_category': 'consistency_test',
                'condition': condition,
                'medication': medication,
                'consistency_test': True,
                'expected_consistency': True
            })
        
        return consistency_data
    
    def _classify_medical_category(self, condition: str) -> str:
        """Classify medical condition into broader categories."""
        category_mapping = {
            'hypertension': 'cardiology',
            'diabetes mellitus': 'endocrinology',
            'asthma': 'pulmonology',
            'migraine': 'neurology',
            'arthritis': 'rheumatology',
            'depression': 'psychiatry',
            'obesity': 'internal'
        }
        
        return category_mapping.get(condition, 'general')
    
    def save_data(self, data: List[Dict[str, Any]], output_path: str, format: str = 'jsonl'):
        """Save generated data to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'jsonl':
            with open(output_file, 'w') as f:
                for record in data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        elif format == 'json':
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
        
        logger.info(f"Saved {len(data)} records to {output_file}")
    
    def generate_comprehensive_test_dataset(self, output_dir: str, size: str = 'medium') -> Dict[str, str]:
        """Generate a comprehensive test dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate different types of test data
        logger.info("Generating comprehensive test dataset...")
        
        # Basic medical dialogues
        basic_data = self.generate_basic_medical_dialogues(500)
        basic_file = output_path / "basic_medical_dialogues.jsonl"
        self.save_data(basic_data, str(basic_file))
        
        # PHI test data
        phi_data = self.generate_phi_test_data(100)
        phi_file = output_path / "phi_compliance_test_data.jsonl"
        self.save_data(phi_data, str(phi_file))
        
        # Edge cases
        edge_data = self.generate_edge_case_data(50)
        edge_file = output_path / "edge_case_data.jsonl"
        self.save_data(edge_data, str(edge_file))
        
        # Performance test data
        perf_data = self.generate_performance_test_data(size)
        perf_file = output_path / f"performance_test_data_{size}.jsonl"
        self.save_data(perf_data, str(perf_file))
        
        # Consistency test data
        consistency_data = self.generate_consistency_test_data(200)
        consistency_file = output_path / "consistency_test_data.jsonl"
        self.save_data(consistency_data, str(consistency_file))
        
        # Create summary file
        summary = {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0',
            'seed': self.seed,
            'dataset_files': {
                'basic_medical_dialogues': str(basic_file),
                'phi_compliance_test_data': str(phi_file),
                'edge_case_data': str(edge_file),
                'performance_test_data': str(perf_file),
                'consistency_test_data': str(consistency_file)
            },
            'record_counts': {
                'basic_medical_dialogues': len(basic_data),
                'phi_compliance_test_data': len(phi_data),
                'edge_case_data': len(edge_data),
                'performance_test_data': len(perf_data),
                'consistency_test_data': len(consistency_data)
            },
            'total_records': len(basic_data) + len(phi_data) + len(edge_data) + len(perf_data) + len(consistency_data)
        }
        
        summary_file = output_path / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated comprehensive test dataset with {summary['total_records']} total records")
        
        return {
            'output_directory': str(output_path),
            'summary_file': str(summary_file),
            'total_records': summary['total_records']
        }

class QADataValidator:
    """Validate generated test data quality."""
    
    def __init__(self):
        self.validation_results = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': [],
            'phi_violations': 0,
            'data_quality_issues': []
        }
    
    def validate_test_data(self, data_file: str) -> Dict[str, Any]:
        """Validate test data quality."""
        logger.info(f"Validating test data: {data_file}")
        
        with open(data_file, 'r') as f:
            records = [json.loads(line) for line in f]
        
        self.validation_results['total_records'] = len(records)
        
        for i, record in enumerate(records):
            # Basic validation
            if not self._validate_record_structure(record):
                self.validation_results['invalid_records'] += 1
                self.validation_results['validation_errors'].append(f"Record {i}: Invalid structure")
                continue
            
            # PHI validation
            if record.get('phi_test', False):
                if not self._validate_phi_presence(record):
                    self.validation_results['phi_violations'] += 1
            
            # Data quality validation
            if not self._validate_data_quality(record):
                self.validation_results['data_quality_issues'].append(f"Record {i}: Quality issues")
            
            self.validation_results['valid_records'] += 1
        
        return self.validation_results
    
    def _validate_record_structure(self, record: Dict[str, Any]) -> bool:
        """Validate record has required structure."""
        required_fields = ['id', 'text']
        
        for field in required_fields:
            if field not in record:
                return False
        
        return True
    
    def _validate_phi_presence(self, record: Dict[str, Any]) -> bool:
        """Validate PHI test records contain expected PHI."""
        if not record.get('should_contain_phi', False):
            return True
        
        text = record.get('text', '')
        phi_value = record.get('phi_value', '')
        
        # Check if PHI is actually in the text
        return phi_value in text
    
    def _validate_data_quality(self, record: Dict[str, Any]) -> bool:
        """Validate data quality of record."""
        text = record.get('text', '')
        
        # Check for empty text (except for edge cases)
        if not text and not record.get('edge_case', False):
            return False
        
        # Check for extremely long text
        if len(text) > 50000:  # 50k characters
            return False
        
        return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Generate test data for Medical AI Training System QA')
    parser.add_argument('--output-dir', default='test_data', help='Output directory for generated data')
    parser.add_argument('--size', choices=['small', 'medium', 'large', 'xlarge'], default='medium', 
                       help='Size of performance test dataset')
    parser.add_argument('--type', choices=['basic', 'phi', 'edge', 'performance', 'consistency', 'comprehensive'],
                       default='comprehensive', help='Type of test data to generate')
    parser.add_argument('--count', type=int, help='Number of records to generate (for basic type)')
    parser.add_argument('--validate', action='store_true', help='Validate generated data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create generator
    generator = MedicalTestDataGenerator(seed=args.seed)
    
    if args.type == 'comprehensive':
        # Generate comprehensive test dataset
        result = generator.generate_comprehensive_test_dataset(args.output_dir, args.size)
        
        print(f"✅ Generated comprehensive test dataset")
        print(f"📁 Output directory: {result['output_directory']}")
        print(f"📊 Total records: {result['total_records']}")
        
        if args.validate:
            validator = QADataValidator()
            for file_path in Path(result['output_directory']).glob("*.jsonl"):
                validation_result = validator.validate_test_data(str(file_path))
                print(f"📋 Validation results for {file_path.name}:")
                print(f"   Valid records: {validation_result['valid_records']}/{validation_result['total_records']}")
    
    else:
        # Generate specific type of data
        output_file = Path(args.output_dir) / f"{args.type}_test_data.jsonl"
        
        if args.type == 'basic':
            count = args.count or 100
            data = generator.generate_basic_medical_dialogues(count)
        elif args.type == 'phi':
            count = args.count or 50
            data = generator.generate_phi_test_data(count)
        elif args.type == 'edge':
            count = args.count or 30
            data = generator.generate_edge_case_data(count)
        elif args.type == 'performance':
            data = generator.generate_performance_test_data(args.size)
        elif args.type == 'consistency':
            count = args.count or 100
            data = generator.generate_consistency_test_data(count)
        
        generator.save_data(data, str(output_file))
        print(f"✅ Generated {len(data)} {args.type} test records")
        print(f"📁 Saved to: {output_file}")
    
    print(f"\n🎯 Test data generation completed successfully!")

if __name__ == "__main__":
    main()