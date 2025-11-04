# PHI Protection Utilities

Comprehensive Protected Health Information (PHI) de-identification and compliance checking utilities for healthcare data protection.

## Overview

This package provides robust tools for detecting, removing, and validating Protected Health Information (PHI) in healthcare data, ensuring HIPAA compliance and protecting patient privacy.

## Features

### 🔍 PHI Detection
- **Regex-based pattern matching** for common PHI types
- **Named Entity Recognition (NER)** using spaCy
- **Dictionary-based detection** for names and medical terms
- **Context-aware validation** to reduce false positives

### 🛡️ De-identification
- **Safe Harbor method** (HIPAA-compliant removal of 18 identifiers)
- **Expert Determination method** (statistical risk assessment)
- **Consistent pseudonyms** across datasets
- **Contextual replacements** (e.g., DOB → 01/01/1900)

### ✅ Validation
- **Residual PHI detection** after de-identification
- **Pseudonym consistency verification**
- **Compliance scoring** and risk assessment
- **Detailed issue reporting**

### 📋 Compliance
- **HIPAA Safe Harbor implementation**
- **Expert Determination support**
- **Compliance certificate generation**
- **Audit trail maintenance**

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Optional: Install spaCy model
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from training.utils import PHIRedactor, PHIValidator, HIPAAComplianceChecker

# Initialize components
redactor = PHIRedactor(consistent_pseudonyms=True)
validator = PHIValidator()
checker = HIPAAComplianceChecker()

# Sample healthcare data
text = """
Patient John Smith was admitted on 01/15/2023.
Phone: (555) 123-4567, Email: john.smith@email.com
Address: 123 Main Street, Springfield, IL 62701
SSN: 123-45-6789, MRN: MR12345678
"""

# 1. De-identify PHI
redacted_text, report = redactor.redact_text(text)
print("Redacted:", redacted_text)

# 2. Validate results
validation_result = validator.validate_deidentification(report)
print("Valid:", validation_result.is_valid)

# 3. Check compliance
compliance_report = checker.check_compliance(report)
print("Compliance:", compliance_report.overall_status)
```

## API Reference

### PHIRedactor

Main class for PHI detection and removal.

```python
redactor = PHIRedactor(
    method="safe_harbor",           # "safe_harbor" or "expert_determination"
    consistent_pseudonyms=True      # Use consistent pseudonyms
)
```

#### Methods

- `redact_text(text, return_report=True)` - De-identify single text
- `batch_redact(texts, return_reports=False)` - Process multiple texts
- `save_pseudonym_map(filepath)` - Save pseudonym mapping
- `load_pseudonym_map(filepath)` - Load pseudonym mapping

### PHIValidator

Validates de-identification results.

```python
validator = PHIValidator(strict_mode=False)
```

#### Methods

- `validate_deidentification(report)` - Validate single report
- `batch_validate(reports)` - Validate multiple reports
- `export_validation_report(report, filepath)` - Export detailed report

### HIPAAComplianceChecker

Checks HIPAA compliance.

```python
checker = HIPAAComplianceChecker(method=ComplianceMethod.SAFE_HARBOR)
```

#### Methods

- `check_compliance(deid_report, validation_report=None)` - Check compliance
- `batch_check_compliance(reports)` - Check multiple reports
- `export_compliance_certificate(report, filepath)` - Export certificate

## Detected PHI Types

### Names
- Full names: `John Smith`
- Names with titles: `Dr. Jane Doe`, `Mr. Johnson`

### Contact Information
- Phone numbers: `(555) 123-4567`, `555-123-4567`
- Email addresses: `john.smith@email.com`

### Address Information
- Street addresses: `123 Main Street`
- City, State, Zip: `Springfield, IL 62701`

### Medical Identifiers
- Social Security Numbers: `123-45-6789`
- Medical Record Numbers: `MR12345678`
- Insurance IDs: `ABC123456789`

### Dates
- Date of Birth: `05/15/1980`
- Admission/Discharge dates: `01/15/2023`
- General dates: `February 15, 2023`

### Provider Information
- Provider names: `Dr. Jane Doe, MD`
- Hospital names: `Springfield General Hospital`
- Medical facilities: `Memorial Medical Center`

## De-identification Strategies

### Safe Harbor Method
Removes 18 HIPAA-specified identifiers:
1. Names
2. Geographic subdivisions smaller than state
3. Dates (except year)
4. Contact numbers
5. Email addresses
6. Social Security numbers
7. Medical record numbers
8. Health plan beneficiary numbers
9. Account numbers
10. Certificate/license numbers
11. Vehicle identifiers
12. Device identifiers
13. Web URLs
14. IP addresses
15. Biometric identifiers
16. Full-face photos
17. Any unique identifying number

### Expert Determination Method
- Statistical risk assessment
- Qualified expert certification
- Context-aware de-identification
- Lower re-identification risk

## Configuration

### Custom Patterns
```python
# Add custom detection patterns
redactor.detection_patterns["custom"] = [
    {
        "pattern": r'\bCUSTOM_ID_[0-9]{6}\b',
        "type": "custom_identifier",
        "confidence": 0.9
    }
]
```

### Pseudonym Generation
```python
# Customize pseudonym format
class CustomRedactor(PHIRedactor):
    def _generate_pseudonym(self, entity_type, original):
        return f"CUSTOM_{entity_type.upper()}_{hash(original) % 1000:03d}"
```

## Validation

### Compliance Scoring
- **0.9 - 1.0**: Excellent compliance
- **0.7 - 0.9**: Good compliance (review recommended)
- **0.5 - 0.7**: Fair compliance (action required)
- **< 0.5**: Poor compliance (immediate action needed)

### Risk Levels
- **CRITICAL**: SSN, MRN, Email, Phone
- **HIGH**: Names, Addresses, Dates
- **MEDIUM**: Provider information
- **LOW**: Generalized location data

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest training/utils/test_phi_protection.py -v

# Run with coverage
python -m pytest training/utils/test_phi_protection.py --cov=training.utils

# Run specific test classes
python -m pytest training/utils/test_phi_protection.py::TestPHIRedactor -v
```

## Example Usage

### Batch Processing
```python
# Process multiple documents
texts = [
    "Patient John Smith was admitted...",
    "Follow-up appointment for Mary Johnson...",
    "Emergency visit - patient unknown..."
]

redacted_texts, reports = redactor.batch_redact(texts, return_reports=True)

# Validate all results
validation_results = validator.batch_validate(reports)

# Check compliance
compliance_results = checker.batch_check_compliance(
    [(r, None) for r in reports]
)
```

### Audit Trail
```python
# Export audit information
for i, (report, validation, compliance) in enumerate(zip(reports, validation_results, compliance_results)):
    # Export pseudonym map
    redactor.save_pseudonym_map(f"pseudonym_map_{i}.json")
    
    # Export validation report
    validator.export_validation_report(validation, f"validation_{i}.json")
    
    # Export compliance certificate
    checker.export_compliance_certificate(compliance, f"compliance_{i}.json")
```

## Error Handling

The utilities handle various error conditions gracefully:

- Empty or None input
- Malformed PHI patterns
- Unicode characters
- Very long texts
- Special characters
- System resource limitations

## Performance Considerations

### Memory Usage
- Process large texts in chunks for memory efficiency
- Use batch processing for multiple documents
- Consider streaming for real-time processing

### Processing Speed
- Regex patterns are optimized for performance
- NER processing may be slower but more accurate
- Use strict mode for faster validation with fewer false positives

## Security Best Practices

### Data Protection
- Store pseudonym maps securely
- Encrypt audit logs in production
- Implement access controls for PHI utilities
- Regular security audits

### Compliance
- Maintain audit trails for all processing
- Regular re-validation of detection patterns
- Document custom configurations
- Train staff on PHI handling procedures

## Troubleshooting

### Common Issues

**No PHI detected when PHI is present**
- Check if patterns match your data format
- Enable NER for better name detection
- Review custom pattern configuration

**Too many false positives**
- Use strict mode in validator
- Add medical terms to exclusion list
- Adjust confidence thresholds

**Compliance failures**
- Review residual PHI detection results
- Check pseudonym consistency
- Verify all 18 Safe Harbor identifiers are handled

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Review the example usage script
- Check the test cases for implementation examples

## Changelog

### Version 1.0.0
- Initial release
- Safe Harbor and Expert Determination methods
- Comprehensive PHI detection
- Validation and compliance checking
- Audit trail generation
- Unit test coverage