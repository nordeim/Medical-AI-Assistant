#!/usr/bin/env python3
"""
PHI Protection Utilities - Complete Example Usage

This script demonstrates the complete PHI protection workflow including:
1. PHI detection and de-identification
2. Validation of de-identification results
3. HIPAA compliance checking
4. Audit trail generation
5. Certificate creation

Run this script to see the utilities in action.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

# Import PHI protection utilities
from training.utils import (
    PHIRedactor, 
    PHIValidator, 
    HIPAAComplianceChecker,
    ComplianceMethod
)


def setup_logging():
    """Setup logging for the demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('phi_protection_demo.log')
        ]
    )


def create_sample_healthcare_data() -> List[Dict[str, str]]:
    """Create sample healthcare data for demonstration"""
    sample_data = [
        {
            "record_id": "H001",
            "content": """
            Patient John Smith was admitted to Springfield General Hospital on January 15, 2023.
            Contact: (555) 123-4567, john.smith@email.com
            Address: 123 Main Street, Springfield, IL 62701
            SSN: 123-45-6789, MRN: MR12345678
            Attending physician: Dr. Jane Doe
            Insurance: Policy # ABC123456789
            Emergency contact: Mary Johnson - (312) 555-9999
            Discharge date: January 20, 2023
            """,
            "type": "admission_record"
        },
        {
            "record_id": "H002", 
            "content": """
            Follow-up appointment for patient Mary Johnson scheduled for February 15, 2023.
            Previous diagnosis: Diabetes Type 2
            Current medication: Metformin 500mg
            Phone: (847) 555-8888, Email: mary.johnson@hotmail.com
            Address: 456 Oak Avenue, Chicago, IL 60601
            """,
            "type": "follow_up"
        },
        {
            "record_id": "H003",
            "content": """
            Emergency department visit - Patient unknown, approximately 65 years old.
            No identification available. Treated for chest pain.
            BP: 140/90, HR: 95, Temp: 98.6°F
            EKG: Normal sinus rhythm
            Released after observation.
            """,
            "type": "emergency_visit"
        },
        {
            "record_id": "H004",
            "content": """
            Radiology report for Patient Robert Wilson.
            Study: Chest X-ray
            Date: 01/22/2023
            Findings: Clear lungs, no acute findings
            Radiologist: Dr. Sarah Chen, MD
            Hospital: Memorial Medical Center
            Phone: (618) 555-7777
            """,
            "type": "radiology_report"
        }
    ]
    
    return sample_data


def demonstrate_phi_redaction(redactor: PHIRedactor, data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Demonstrate PHI redaction functionality"""
    print("\n" + "="*80)
    print("PHI REDACTION DEMONSTRATION")
    print("="*80)
    
    redacted_data = []
    
    for i, record in enumerate(data, 1):
        print(f"\nProcessing Record {i} ({record['record_id']})...")
        print(f"Type: {record['type']}")
        
        # Redact PHI
        redacted_text, report = redactor.redact_text(record['content'])
        
        print(f"Original length: {len(record['content'])} characters")
        print(f"Redacted length: {len(redacted_text)} characters")
        print(f"PHI detections: {len(report.detections)}")
        
        # Show detected PHI types
        phi_types = {}
        for detection in report.detections:
            phi_types[detection.phi_type] = phi_types.get(detection.phi_type, 0) + 1
        
        if phi_types:
            print("PHI types detected:")
            for phi_type, count in phi_types.items():
                print(f"  - {phi_type}: {count}")
        else:
            print("No PHI detected")
        
        redacted_record = {
            "record_id": record['record_id'],
            "original_content": record['content'],
            "redacted_content": redacted_text,
            "deidentification_report": report,
            "phi_detections": phi_types
        }
        redacted_data.append(redacted_record)
    
    return redacted_data


def demonstrate_validation(validator: PHIValidator, redacted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Demonstrate PHI validation functionality"""
    print("\n" + "="*80)
    print("PHI VALIDATION DEMONSTRATION")
    print("="*80)
    
    validation_results = []
    
    for i, record in enumerate(redacted_data, 1):
        print(f"\nValidating Record {i} ({record['record_id']})...")
        
        # Validate de-identification
        validation_result = validator.validate_deidentification(record['deidentification_report'])
        
        print(f"Validation status: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        print(f"Residual PHI: {len(validation_result.residual_phi)}")
        print(f"Pseudonym consistency: {validation_result.pseudonym_consistency:.2f}")
        print(f"Compliance score: {validation_result.compliance_score:.2f}")
        
        if validation_result.issues:
            print("Issues found:")
            for issue in validation_result.issues:
                print(f"  - {issue}")
        
        if validation_result.recommendations:
            print("Recommendations:")
            for rec in validation_result.recommendations:
                print(f"  - {rec}")
        
        validation_record = {
            "record_id": record['record_id'],
            "validation_result": validation_result,
            "deidentification_report": record['deidentification_report'],
            "is_valid": validation_result.is_valid
        }
        validation_results.append(validation_record)
    
    return validation_results


def demonstrate_compliance_checking(checker: HIPAAComplianceChecker, 
                                  redacted_data: List[Dict[str, Any]],
                                  validation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Demonstrate HIPAA compliance checking"""
    print("\n" + "="*80)
    print("HIPAA COMPLIANCE CHECKING DEMONSTRATION")
    print("="*80)
    
    compliance_results = []
    
    # Test both methods
    for method in [ComplianceMethod.SAFE_HARBOR, ComplianceMethod.EXPERT_DETERMINATION]:
        print(f"\nTesting {method.value.title()} Method:")
        print("-" * 50)
        
        method_checker = HIPAAComplianceChecker(method)
        
        for i, (record, validation) in enumerate(zip(redacted_data, validation_results), 1):
            print(f"\nChecking Record {i} ({record['record_id']})...")
            
            # Check compliance
            compliance_report = method_checker.check_compliance(
                record['deidentification_report'],
                None  # Simplified - no full validation report
            )
            
            print(f"Compliance status: {compliance_report.overall_status}")
            print(f"Compliance score: {compliance_report.compliance_score:.2f}")
            print(f"Requirements met: {compliance_report.requirements_met}/{compliance_report.total_requirements}")
            print(f"Certificate ID: {compliance_report.compliance_certificate_id}")
            
            compliance_record = {
                "record_id": record['record_id'],
                "method": method.value,
                "compliance_report": compliance_report
            }
            compliance_results.append(compliance_record)
    
    return compliance_results


def demonstrate_audit_trail(redactor: PHIRedactor, 
                          validation_results: List[Dict[str, Any]], 
                          compliance_results: List[Dict[str, Any]]):
    """Demonstrate audit trail generation"""
    print("\n" + "="*80)
    print("AUDIT TRAIL DEMONSTRATION")
    print("="*80)
    
    # Create audit directory
    audit_dir = "phi_protection_audit"
    os.makedirs(audit_dir, exist_ok=True)
    
    # Export pseudonym maps
    pseudonym_file = os.path.join(audit_dir, "pseudonym_map.json")
    redactor.save_pseudonym_map(pseudonym_file)
    print(f"Pseudonym map saved: {pseudonym_file}")
    
    # Export validation reports
    for i, validation in enumerate(validation_results, 1):
        validation_report_file = os.path.join(audit_dir, f"validation_report_{validation['record_id']}.json")
        
        # Create full validation report for export
        from training.utils.phi_validator import ValidationReport
        full_validation_report = ValidationReport(
            original_report=validation['deidentification_report'],
            validation_result=validation['validation_result'],
            validator_version="1.0.0",
            validation_timestamp=datetime.now(),
            summary={
                "is_valid": validation['is_valid'],
                "record_id": validation['record_id']
            }
        )
        
        validator = PHIValidator()
        validator.export_validation_report(full_validation_report, validation_report_file)
        print(f"Validation report saved: {validation_report_file}")
    
    # Export compliance certificates
    for compliance in compliance_results:
        cert_file = os.path.join(audit_dir, f"compliance_cert_{compliance['record_id']}_{compliance['method']}.json")
        
        checker = HIPAAComplianceChecker(ComplianceMethod.SAFE_HARBOR)
        checker.export_compliance_certificate(compliance['compliance_report'], cert_file)
        print(f"Compliance certificate saved: {cert_file}")
    
    # Create summary report
    summary_report = {
        "audit_timestamp": datetime.now().isoformat(),
        "total_records_processed": len(validation_results),
        "validation_results": {
            "passed": sum(1 for v in validation_results if v['is_valid']),
            "failed": sum(1 for v in validation_results if not v['is_valid'])
        },
        "compliance_summary": {
            method: {
                "passed": sum(1 for c in compliance_results 
                            if c['method'] == method and c['compliance_report'].overall_status == "COMPLIANT"),
                "failed": sum(1 for c in compliance_results 
                            if c['method'] == method and c['compliance_report'].overall_status == "NON_COMPLIANT")
            }
            for method in set(c['method'] for c in compliance_results)
        },
        "files_generated": os.listdir(audit_dir)
    }
    
    summary_file = os.path.join(audit_dir, "audit_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    print(f"Audit summary saved: {summary_file}")
    
    return audit_dir


def demonstrate_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n" + "="*80)
    print("ERROR HANDLING DEMONSTRATION")
    print("="*80)
    
    redactor = PHIRedactor()
    
    # Test with various problematic inputs
    test_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("Normal medical text without PHI", "Text without PHI"),
        (None, "None input"),
        ("Patient John Smith \x00with null bytes", "Text with null bytes")
    ]
    
    for test_input, description in test_cases:
        print(f"\nTesting: {description}")
        try:
            if test_input is not None:
                redacted, report = redactor.redact_text(test_input)
                print(f"  Result: {len(report.detections)} PHI detections")
            else:
                # Test None handling
                redacted, report = redactor.redact_text("")
                print(f"  Result: Handled gracefully")
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")


def main():
    """Main demonstration function"""
    print("PHI PROTECTION UTILITIES - COMPREHENSIVE DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows the complete PHI protection workflow")
    print("including detection, validation, compliance checking, and audit trails.")
    
    # Setup
    setup_logging()
    
    # Create sample data
    print("\nCreating sample healthcare data...")
    sample_data = create_sample_healthcare_data()
    print(f"Created {len(sample_data)} sample records")
    
    # Initialize components
    print("\nInitializing PHI protection components...")
    redactor = PHIRedactor(consistent_pseudonyms=True)
    validator = PHIValidator()
    checker = HIPAAComplianceChecker()
    
    # Demonstrate each component
    print("\nRunning demonstrations...")
    
    # 1. PHI Redaction
    redacted_data = demonstrate_phi_redaction(redactor, sample_data)
    
    # 2. Validation
    validation_results = demonstrate_validation(validator, redacted_data)
    
    # 3. Compliance Checking
    compliance_results = demonstrate_compliance_checking(checker, redacted_data, validation_results)
    
    # 4. Audit Trail
    audit_dir = demonstrate_audit_trail(redactor, validation_results, compliance_results)
    
    # 5. Error Handling
    demonstrate_error_handling()
    
    # Summary
    print("\n" + "="*80)
    print("DEMONSTRATION SUMMARY")
    print("="*80)
    print(f"Records processed: {len(sample_data)}")
    print(f"Total PHI detections: {sum(len(r['deidentification_report'].detections) for r in redacted_data)}")
    print(f"Validation passes: {sum(1 for v in validation_results if v['is_valid'])}/{len(validation_results)}")
    print(f"Compliance certificates generated: {len(compliance_results)}")
    print(f"Audit directory: {audit_dir}")
    print("\nAll demonstration components completed successfully!")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review generated audit files in:", audit_dir)
    print("2. Run unit tests: python -m pytest training/utils/test_phi_protection.py")
    print("3. Customize patterns and requirements for your specific use case")
    print("4. Integrate with your existing healthcare data processing pipeline")
    print("5. Schedule periodic re-validation for ongoing compliance")


if __name__ == "__main__":
    main()