"""
PHI Validator: Validation and compliance checking for de-identified healthcare data
Provides comprehensive validation of PHI de-identification processes and results.
"""

import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib

from .phi_redactor import PHIRedactor, DeidentificationReport, PHIDetection


@dataclass
class ValidationResult:
    """Result of PHI validation"""
    is_valid: bool
    residual_phi: List[PHIDetection]
    pseudonym_consistency: float
    compliance_score: float
    issues: List[str]
    recommendations: List[str]
    validation_timestamp: datetime


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    original_report: DeidentificationReport
    validation_result: ValidationResult
    validator_version: str
    validation_timestamp: datetime
    summary: Dict[str, Any]


class PHIValidator:
    """
    Validates PHI de-identification results and ensures compliance
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize PHI Validator
        
        Args:
            strict_mode: Whether to use strict validation (fewer false negatives but more false positives)
        """
        self.strict_mode = strict_mode
        self.validation_patterns = self._initialize_validation_patterns()
        self.reserved_patterns = self._initialize_reserved_patterns()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation process"""
        logger = logging.getLogger(f"phi_validator_{datetime.now().strftime('%Y%m%d')}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_validation_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize patterns for detecting residual PHI"""
        patterns = {
            # Names - more aggressive patterns
            "names": [
                {
                    "pattern": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                    "type": "potential_name",
                    "confidence": 0.6
                },
                {
                    "pattern": r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+\b',
                    "type": "name_with_title",
                    "confidence": 0.8
                }
            ],
            
            # Contact information
            "phone": [
                {
                    "pattern": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                    "type": "phone_number",
                    "confidence": 0.95
                }
            ],
            
            "email": [
                {
                    "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    "type": "email",
                    "confidence": 0.95
                }
            ],
            
            # Dates - very sensitive
            "dates": [
                {
                    "pattern": r'\b[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}\b',
                    "type": "date",
                    "confidence": 0.7
                },
                {
                    "pattern": r'\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b',
                    "type": "date",
                    "confidence": 0.7
                }
            ],
            
            # Addresses
            "address": [
                {
                    "pattern": r'\b[0-9]+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',
                    "type": "address",
                    "confidence": 0.8
                }
            ],
            
            # Medical numbers
            "medical": [
                {
                    "pattern": r'\b[A-Z]{2,3}[0-9]{4,8}\b',
                    "type": "medical_identifier",
                    "confidence": 0.6
                },
                {
                    "pattern": r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b',
                    "type": "ssn",
                    "confidence": 0.95
                }
            ]
        }
        
        return patterns
    
    def _initialize_reserved_patterns(self) -> Dict[str, str]:
        """Initialize patterns that indicate potential PHI even if pattern doesn't match exactly"""
        return {
            "date_indicators": r"\b(?:born|birthdate|dob|date of birth|admission|discharge|admit|discharge)\b",
            "name_indicators": r"\b(?:patient|person|individual|subject|client)\b.*?\b[A-Z][a-z]+\b",
            "location_indicators": r"\b(?:lives in|resides at|address:)\b",
            "contact_indicators": r"\b(?:phone|cell|email|contact|reach|call)\b.*?\d",
            "medical_indicators": r"\b(?:chart|mrn|medical record|patient id)\b.*?\d"
        }
    
    def validate_deidentification(self, report: DeidentificationReport) -> ValidationResult:
        """
        Validate de-identification results
        
        Args:
            report: DeidentificationReport from PHIRedactor
            
        Returns:
            ValidationResult with validation outcomes
        """
        self.logger.info(f"Starting validation for report from {report.timestamp}")
        
        # Check for residual PHI
        residual_phi = self._detect_residual_phi(report.redacted_text)
        
        # Validate pseudonym consistency
        pseudonym_consistency = self._validate_pseudonym_consistency(report)
        
        # Calculate compliance score
        compliance_score = self._calculate_validation_compliance_score(residual_phi, pseudonym_consistency)
        
        # Identify issues and generate recommendations
        issues, recommendations = self._analyze_validation_issues(residual_phi, report, pseudonym_consistency)
        
        # Determine if validation passed
        is_valid = (
            len(residual_phi) == 0 and 
            pseudonym_consistency >= 0.8 and
            compliance_score >= 0.9 and
            len([i for i in issues if i.startswith("CRITICAL")]) == 0
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            residual_phi=residual_phi,
            pseudonym_consistency=pseudonym_consistency,
            compliance_score=compliance_score,
            issues=issues,
            recommendations=recommendations,
            validation_timestamp=datetime.now()
        )
        
        self.logger.info(f"Validation completed: {'PASSED' if is_valid else 'FAILED'}")
        self.logger.info(f"Residual PHI: {len(residual_phi)}, Pseudonym consistency: {pseudonym_consistency:.2f}")
        
        return result
    
    def _detect_residual_phi(self, text: str) -> List[PHIDetection]:
        """Detect any remaining PHI in de-identified text"""
        residual_phi = []
        
        # Apply validation patterns
        for category, patterns in self.validation_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                phi_type = pattern_info["type"]
                confidence = pattern_info["confidence"]
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    detection = PHIDetection(
                        text=text[match.start():match.end()],
                        start_pos=match.start(),
                        end_pos=match.end(),
                        phi_type=phi_type,
                        confidence=confidence,
                        replacement=""
                    )
                    
                    # Filter out false positives
                    if self._is_false_positive(detection.text, phi_type):
                        continue
                    
                    # Mark as residual PHI (was not caught in original detection)
                    detection.phi_type = f"RESIDUAL_{phi_type}"
                    residual_phi.append(detection)
        
        # Check for indicators of missed PHI
        residual_phi.extend(self._detect_phi_indicators(text))
        
        return residual_phi
    
    def _is_false_positive(self, text: str, phi_type: str) -> bool:
        """Determine if detection is likely a false positive"""
        text_lower = text.lower().strip()
        
        # Common false positive patterns
        false_positives = {
            "phone_number": ["123", "911", "411"],
            "email": ["test@example.com", "noreply@", "admin@"],
            "date": ["01/01/2023", "12/25/2023", "04/01/2023"],  # Common test dates
            "name_with_title": ["Dr. Pepper", "Mr. Clean"],  # Brand names
        }
        
        if phi_type in false_positives:
            for fp in false_positives[phi_type]:
                if fp in text_lower:
                    return True
        
        # Check if it's a placeholder or pseudonym
        placeholder_patterns = [
            r"^[A-Z]+_\d{3,}$",  # LIKE_PERSON_001
            r"^[A-Z_]+$",        # ALL_CAPS_WITH_UNDERSCORES
            r"^\[.*\]$",         # [BRACKETED_TEXT]
        ]
        
        for pattern in placeholder_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _detect_phi_indicators(self, text: str) -> List[PHIDetection]:
        """Detect indicators of potentially missed PHI"""
        detections = []
        
        for indicator_name, pattern in self.reserved_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detection = PHIDetection(
                    text=text[match.start():match.end()],
                    start_pos=match.start(),
                    end_pos=match.end(),
                    phi_type=f"PHI_INDICATOR_{indicator_name.upper()}",
                    confidence=0.5,
                    replacement=""
                )
                detections.append(detection)
        
        return detections
    
    def _validate_pseudonym_consistency(self, report: DeidentificationReport) -> float:
        """Validate that pseudonyms are used consistently"""
        if not report.detections:
            return 1.0
        
        # Group detections by original text
        entity_groups = defaultdict(list)
        for detection in report.detections:
            entity_groups[detection.text].append(detection)
        
        # Check consistency within each entity group
        total_consistent = 0
        total_entities = 0
        
        for original_text, detections in entity_groups.items():
            total_entities += 1
            
            # Check if all occurrences of this entity have the same pseudonym
            pseudonyms = set(detection.replacement for detection in detections)
            
            if len(pseudonyms) == 1:
                total_consistent += 1
            else:
                self.logger.warning(f"Inconsistent pseudonyms for '{original_text}': {pseudonyms}")
        
        consistency_score = total_consistent / total_entities if total_entities > 0 else 1.0
        
        return consistency_score
    
    def _calculate_validation_compliance_score(self, residual_phi: List[PHIDetection], pseudonym_consistency: float) -> float:
        """Calculate overall compliance score"""
        if not residual_phi:
            base_score = 1.0
        else:
            # Penalties for different types of residual PHI
            critical_penalty = sum(1 for d in residual_phi if d.phi_type.startswith("RESIDUAL_") and 
                                 any(crit in d.phi_type for crit in ["SSN", "MRN", "EMAIL", "PHONE"]))
            high_penalty = sum(1 for d in residual_phi if any(high in d.phi_type for high in ["NAME", "ADDRESS", "DATE"]))
            low_penalty = len(residual_phi) - critical_penalty - high_penalty
            
            base_score = max(0.0, 1.0 - (critical_penalty * 0.5 + high_penalty * 0.3 + low_penalty * 0.1))
        
        # Weight pseudonym consistency
        weighted_score = base_score * 0.7 + pseudonym_consistency * 0.3
        
        return weighted_score
    
    def _analyze_validation_issues(self, residual_phi: List[PHIDetection], 
                                 report: DeidentificationReport, 
                                 pseudonym_consistency: float) -> Tuple[List[str], List[str]]:
        """Analyze validation results and identify issues"""
        issues = []
        recommendations = []
        
        # Check for residual PHI
        if residual_phi:
            critical_phi = [d for d in residual_phi if any(crit in d.phi_type for crit in ["SSN", "MRN", "EMAIL", "PHONE"])]
            high_risk_phi = [d for d in residual_phi if any(high in d.phi_type for high in ["NAME", "ADDRESS", "DATE"])]
            
            if critical_phi:
                issues.append(f"CRITICAL: {len(critical_phi)} critical PHI elements remain (SSN, MRN, Email, Phone)")
                recommendations.append("IMMEDIATE ACTION REQUIRED: Review and manually remove critical PHI elements")
            
            if high_risk_phi:
                issues.append(f"HIGH: {len(high_risk_phi)} high-risk PHI elements remain (Names, Addresses, Dates)")
                recommendations.append("Review remaining PHI and consider enhanced detection methods")
            
            # Detailed issue reporting
            phi_type_counts = Counter(d.phi_type for d in residual_phi)
            for phi_type, count in phi_type_counts.items():
                issues.append(f"{phi_type}: {count} instances detected")
        
        # Check pseudonym consistency
        if pseudonym_consistency < 1.0:
            issues.append(f"PSEUDONYM_INCONSISTENCY: {pseudonym_consistency:.2f} consistency score")
            recommendations.append("Ensure consistent pseudonym mapping across all text segments")
        elif pseudonym_consistency < 0.8:
            issues.append(f"CRITICAL: Low pseudonym consistency ({pseudonym_consistency:.2f})")
            recommendations.append("REVIEW REQUIRED: Pseudonym consistency is too low - may indicate data integrity issues")
        
        # Check coverage
        detection_rate = len(report.detections) / max(len(report.original_text.split()), 1) * 1000
        if detection_rate < 0.1:  # Very few detections
            recommendations.append("Consider if PHI detection is too aggressive or not aggressive enough")
        
        # Method-specific checks
        if report.compliance_score < 0.5:
            issues.append("LOW_COMPLIANCE_SCORE: Original de-identification has low compliance score")
            recommendations.append("Review de-identification method and patterns")
        
        # Generate general recommendations
        if not issues:
            recommendations.append("Validation passed successfully")
            recommendations.append("Consider periodic re-validation as detection patterns evolve")
        else:
            recommendations.append("Consider implementing additional validation layers")
            recommendations.append("Document any acceptable PHI omissions for compliance review")
        
        return issues, recommendations
    
    def batch_validate(self, reports: List[DeidentificationReport]) -> List[Tuple[ValidationReport, ValidationResult]]:
        """Validate multiple de-identification reports"""
        results = []
        
        for i, report in enumerate(reports):
            self.logger.info(f"Validating report {i+1}/{len(reports)}")
            
            try:
                validation_result = self.validate_deidentification(report)
                
                validation_report = ValidationReport(
                    original_report=report,
                    validation_result=validation_result,
                    validator_version="1.0.0",
                    validation_timestamp=datetime.now(),
                    summary={
                        "is_valid": validation_result.is_valid,
                        "residual_phi_count": len(validation_result.residual_phi),
                        "pseudonym_consistency": validation_result.pseudonym_consistency,
                        "compliance_score": validation_result.compliance_score,
                        "critical_issues": len([i for i in validation_result.issues if i.startswith("CRITICAL")])
                    }
                )
                
                results.append((validation_report, validation_result))
                
            except Exception as e:
                self.logger.error(f"Error validating report {i}: {e}")
                # Create error result
                error_result = ValidationResult(
                    is_valid=False,
                    residual_phi=[],
                    pseudonym_consistency=0.0,
                    compliance_score=0.0,
                    issues=[f"VALIDATION_ERROR: {str(e)}"],
                    recommendations=["Fix validation error and re-run"],
                    validation_timestamp=datetime.now()
                )
                results.append((None, error_result))
        
        return results
    
    def export_validation_report(self, validation_report: ValidationReport, filepath: str):
        """Export detailed validation report"""
        report_dict = {
            "validation_metadata": {
                "timestamp": validation_report.validation_timestamp.isoformat(),
                "validator_version": validation_report.validator_version,
                "original_timestamp": validation_report.original_report.timestamp.isoformat()
            },
            "validation_result": {
                "is_valid": validation_report.validation_result.is_valid,
                "residual_phi_count": len(validation_report.validation_result.residual_phi),
                "pseudonym_consistency": validation_report.validation_result.pseudonym_consistency,
                "compliance_score": validation_report.validation_result.compliance_score,
                "issues": validation_report.validation_result.issues,
                "recommendations": validation_report.validation_result.recommendations
            },
            "original_deidentification": {
                "compliance_score": validation_report.original_report.compliance_score,
                "detections_count": len(validation_report.original_report.detections),
                "pseudonym_mapping_size": len(validation_report.original_report.pseudonym_map)
            },
            "summary": validation_report.summary,
            "redacted_text_sample": validation_report.original_report.redacted_text[:500] + "..." if len(validation_report.original_report.redacted_text) > 500 else validation_report.original_report.redacted_text
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Validation report exported to {filepath}")
    
    def generate_compliance_certificate(self, validation_reports: List[ValidationReport], 
                                      output_path: str) -> str:
        """Generate compliance certificate for validated data"""
        all_valid = all(vr.validation_result.is_valid for vr in validation_reports)
        avg_compliance = sum(vr.validation_result.compliance_score for vr in validation_reports) / len(validation_reports)
        
        certificate = {
            "certificate_id": f"PHI_COMPLIANCE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "issued_date": datetime.now().isoformat(),
            "status": "COMPLIANT" if all_valid else "NON_COMPLIANT",
            "validation_summary": {
                "total_documents": len(validation_reports),
                "compliant_documents": sum(1 for vr in validation_reports if vr.validation_result.is_valid),
                "average_compliance_score": avg_compliance,
                "total_residual_phi": sum(len(vr.validation_result.residual_phi) for vr in validation_reports)
            },
            "compliance_statement": "This data has been validated for HIPAA compliance" if all_valid else "This data requires review for HIPAA compliance",
            "recommendations": [
                "Store this certificate with the de-identified data",
                "Re-validate periodically or when detection patterns change",
                "Maintain audit logs for compliance review"
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(certificate, f, indent=2)
        
        self.logger.info(f"Compliance certificate generated: {output_path}")
        return certificate["certificate_id"]


def main():
    """Example usage and testing"""
    # Create sample de-identification report for testing
    from training.utils.phi_redactor import PHIRedactor
    
    # Sample text with PHI
    sample_text = """
    Patient John Smith was admitted on 01/15/2023.
    Phone: (555) 123-4567, Email: john.smith@email.com
    Address: 123 Main Street, Springfield, IL 62701
    SSN: 123-45-6789
    Dr. Jane Doe was the attending physician.
    """
    
    # De-identify
    redactor = PHIRedactor()
    redacted_text, report = redactor.redact_text(sample_text)
    
    print("Original text:")
    print(sample_text)
    print("\nRedacted text:")
    print(redacted_text)
    
    # Validate
    validator = PHIValidator()
    validation_result = validator.validate_deidentification(report)
    
    print(f"\nValidation Result: {'PASSED' if validation_result.is_valid else 'FAILED'}")
    print(f"Residual PHI: {len(validation_result.residual_phi)}")
    print(f"Pseudonym Consistency: {validation_result.pseudonym_consistency:.2f}")
    print(f"Compliance Score: {validation_result.compliance_score:.2f}")
    
    if validation_result.issues:
        print("\nIssues:")
        for issue in validation_result.issues:
            print(f"  - {issue}")
    
    if validation_result.recommendations:
        print("\nRecommendations:")
        for rec in validation_result.recommendations:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()