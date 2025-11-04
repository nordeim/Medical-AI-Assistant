#!/usr/bin/env python3
"""Example usage of data validation and quality assurance utilities."""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from training.utils.data_validator import DataValidator, MedicalDataValidator, ValidationConfig
from training.utils.validation_reporter import ValidationReporter


def create_sample_medical_data():
    """Create sample medical training data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Sample data generator
    symptoms_list = [
        'severe headache and fever',
        'chest pain and shortness of breath',
        'persistent cough and fatigue',
        'nausea and vomiting',
        'back pain and leg numbness',
        'abdominal pain and diarrhea',
        'dizziness and blurred vision',
        'sore throat and congestion'
    ]
    
    triage_levels = ['emergency', 'urgent', 'non-urgent', 'advisory']
    genders = ['male', 'female', 'other']
    
    data = []
    
    for i in range(50):
        # Generate realistic conversation
        conversation_id = f"conv_{i:03d}"
        
        # User input with medical context
        user_input = f"I've been experiencing {symptoms_list[i % len(symptoms_list)]} for the past day."
        
        # Assistant response with medical appropriateness
        if 'emergency' in user_input.lower() or 'severe' in user_input.lower():
            assistant_response = "This sounds like it requires immediate medical attention. Please go to the emergency room right away or call 911."
        elif 'urgent' in user_input.lower():
            assistant_response = "Your symptoms warrant prompt medical evaluation. I recommend contacting your healthcare provider today or going to an urgent care center."
        else:
            assistant_response = "I understand your concerns. While these symptoms may not require immediate emergency care, you should monitor them closely and consider scheduling an appointment with your healthcare provider."
        
        # Generate metadata
        timestamp = (datetime.now() - timedelta(days=i)).isoformat() + "Z"
        age = np.random.randint(18, 85)
        gender = np.random.choice(genders)
        triage_level = np.random.choice(triage_levels)
        
        data.append({
            'conversation_id': conversation_id,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'timestamp': timestamp,
            'age': age,
            'gender': gender,
            'triage_level': triage_level,
            'symptoms': user_input.replace("I've been experiencing ", "").replace(" for the past day.", "")
        })
    
    # Add some problematic data for testing
    data.extend([
        {
            'conversation_id': 'problem_001',
            'user_input': '',  # Empty user input
            'assistant_response': 'OK',
            'timestamp': datetime.now().isoformat() + "Z",
            'age': 999,  # Invalid age
            'gender': 'unknown',
            'triage_level': 'invalid_level',
            'symptoms': 'problem'
        },
        {
            'conversation_id': 'phi_001',
            'user_input': 'My phone is 555-123-4567 and SSN is 123-45-6789',
            'assistant_response': 'I can help with medical concerns.',
            'timestamp': datetime.now().isoformat() + "Z",
            'age': 30,
            'gender': 'female',
            'triage_level': 'non-urgent',
            'symptoms': 'general concern'
        }
    ])
    
    return data


def demonstrate_basic_validation():
    """Demonstrate basic validation functionality."""
    print("="*60)
    print("BASIC DATA VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    print("\n1. Creating sample medical training data...")
    data = create_sample_medical_data()
    print(f"   Generated {len(data)} training records")
    
    # Create validator
    print("\n2. Setting up validation configuration...")
    config = ValidationConfig(
        min_text_length=10,
        max_text_length=2000,
        duplicate_similarity_threshold=0.9,
        age_range=(0, 150)
    )
    
    validator = MedicalDataValidator(config)
    print("   ✓ Medical Data Validator initialized")
    
    # Run validation
    print("\n3. Running data validation...")
    result = validator.validate_dataset(data)
    
    # Print results
    print(f"\n4. Validation Results:")
    print(f"   Status: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
    print(f"   Score: {result.score:.1%}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    if result.errors:
        print(f"\n   Errors found:")
        for i, error in enumerate(result.errors[:5], 1):
            print(f"     {i}. {error}")
        if len(result.errors) > 5:
            print(f"     ... and {len(result.errors) - 5} more errors")
    
    if result.warnings:
        print(f"\n   Warnings found:")
        for i, warning in enumerate(result.warnings[:5], 1):
            print(f"     {i}. {warning}")
        if len(result.warnings) > 5:
            print(f"     ... and {len(result.warnings) - 5} more warnings")
    
    return result


def demonstrate_reporting():
    """Demonstrate report generation."""
    print("\n" + "="*60)
    print("REPORT GENERATION DEMONSTRATION")
    print("="*60)
    
    # Use the data and result from previous demonstration
    data = create_sample_medical_data()
    config = ValidationConfig()
    validator = MedicalDataValidator(config)
    result = validator.validate_dataset(data)
    
    # Create output directory
    output_dir = "example_validation_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate reports
    print(f"\n1. Generating reports in '{output_dir}' directory...")
    
    reporter = ValidationReporter(config)
    
    # Generate HTML report
    html_path = os.path.join(output_dir, "medical_data_validation_report.html")
    reporter.generate_html_report(result, html_path, "Sample Medical Dataset")
    print(f"   ✓ HTML report: {html_path}")
    
    # Generate JSON report
    json_path = os.path.join(output_dir, "medical_data_validation_report.json")
    reporter.generate_json_report(result, json_path)
    print(f"   ✓ JSON report: {json_path}")
    
    # Generate CSV summary
    csv_path = os.path.join(output_dir, "medical_data_validation_summary.csv")
    reporter.generate_csv_summary(result, csv_path, len(data))
    print(f"   ✓ CSV summary: {csv_path}")
    
    print(f"\n2. Reports generated successfully!")
    print(f"   Open {html_path} in your browser to view the detailed report")
    
    return output_dir


def demonstrate_batch_validation():
    """Demonstrate batch validation functionality."""
    print("\n" + "="*60)
    print("BATCH VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create multiple datasets
    print("\n1. Creating multiple datasets...")
    datasets = []
    
    for i in range(3):
        # Create dataset with different characteristics
        dataset = create_sample_medical_data()
        
        # Modify characteristics for variety
        if i == 1:
            # More problems
            dataset.extend([
                {
                    'conversation_id': f'problem_{i}_001',
                    'user_input': '',  # Empty
                    'assistant_response': 'Response',
                    'timestamp': datetime.now().isoformat() + "Z",
                    'age': -5,  # Invalid
                    'gender': 'invalid',
                    'triage_level': 'invalid',
                    'symptoms': 'symptoms'
                }
            ])
        elif i == 2:
            # Better quality data (less issues)
            for j in range(min(10, len(dataset))):
                dataset[j]['triage_level'] = np.random.choice(['non-urgent', 'advisory'])
        
        datasets.append(dataset)
        print(f"   Dataset {i+1}: {len(dataset)} records")
    
    # Create validator
    config = ValidationConfig()
    validator = MedicalDataValidator(config)
    
    # Batch validation
    print("\n2. Running batch validation...")
    results = validator.batch_validate(datasets)
    
    # Generate batch report
    output_dir = "example_batch_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    from training.utils.validation_reporter import BatchValidationReporter
    
    batch_reporter = BatchValidationReporter(config)
    batch_report_path = os.path.join(output_dir, "batch_validation_summary.html")
    
    batch_reporter.generate_batch_summary_report(
        results, 
        batch_report_path, 
        [f"Dataset_{i+1}" for i in range(len(datasets))]
    )
    
    print(f"   ✓ Batch report: {batch_report_path}")
    
    # Print summary
    print(f"\n3. Batch Results Summary:")
    passed = sum(1 for r in results if r.is_valid)
    failed = len(results) - passed
    avg_score = np.mean([r.score for r in results])
    
    print(f"   Total datasets: {len(results)}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Average score: {avg_score:.1%}")
    
    return output_dir


def demonstrate_custom_configuration():
    """Demonstrate custom validation configuration."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    print("\n1. Creating custom validation configuration...")
    
    # Custom configuration
    custom_config = ValidationConfig(
        # Strict text quality requirements
        min_text_length=20,
        max_text_length=1000,
        min_readability_score=8.0,
        
        # Stricter duplicate detection
        duplicate_similarity_threshold=0.8,
        
        # More restrictive age range
        age_range=(18, 100),
        
        # Additional required fields
        required_fields=[
            'conversation_id', 'user_input', 'assistant_response', 
            'timestamp', 'age', 'gender', 'triage_level', 'symptoms',
            'patient_id', 'visit_reason'
        ],
        
        # Additional PHI patterns
        phi_patterns=ValidationConfig().phi_patterns + [
            r'\b[A-Z]{2}\d{6}\b',  # Medical record numbers
            r'\b\d{2}/\d{2}/\d{4}\b'  # Dates
        ],
        
        # Additional medical terms
        medical_terms=ValidationConfig().medical_terms + [
            'hypertension', 'diabetes', 'infection', 'inflammation',
            'fracture', 'laceration', 'contusion', 'burn'
        ]
    )
    
    print("   ✓ Custom configuration created:")
    print(f"     - Min text length: {custom_config.min_text_length}")
    print(f"     - Age range: {custom_config.age_range}")
    print(f"     - Required fields: {len(custom_config.required_fields)}")
    print(f"     - Medical terms: {len(custom_config.medical_terms)}")
    
    # Test with sample data
    print("\n2. Testing with custom configuration...")
    data = create_sample_medical_data()
    
    # Add custom required fields to test data
    for i, record in enumerate(data):
        record['patient_id'] = f"PAT_{i:04d}"
        record['visit_reason'] = "General consultation"
    
    validator = MedicalDataValidator(custom_config)
    result = validator.validate_dataset(data)
    
    print(f"\n3. Results with custom configuration:")
    print(f"   Score: {result.score:.1%}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    # Generate report with custom config
    output_dir = "example_custom_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    reporter = ValidationReporter(custom_config)
    report_path = os.path.join(output_dir, "custom_validation_report.html")
    reporter.generate_html_report(result, report_path, "Custom Configuration Test")
    
    print(f"   ✓ Custom report: {report_path}")
    
    return output_dir


def main():
    """Run all demonstration examples."""
    print("🏥 Medical AI Assistant - Data Validation Examples")
    print("=" * 80)
    
    try:
        # Run demonstrations
        basic_result = demonstrate_basic_validation()
        basic_reports_dir = demonstrate_reporting()
        batch_reports_dir = demonstrate_batch_validation()
        custom_reports_dir = demonstrate_custom_configuration()
        
        print("\n" + "="*80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\nGenerated Reports:")
        print(f"  • Basic validation: {basic_reports_dir}")
        print(f"  • Batch validation: {batch_reports_dir}")
        print(f"  • Custom config: {custom_reports_dir}")
        
        print(f"\nOpen any of the HTML reports in your browser to view detailed results.")
        print(f"Check the JSON reports for machine-readable data and the CSV summaries for quick analysis.")
        
        print(f"\nTo run validation on your own data:")
        print(f"  python training/scripts/validate_data.py file your_data.json --medical")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()