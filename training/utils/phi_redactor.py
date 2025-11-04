"""
PHI (Protected Health Information) De-identification Utilities

This module provides comprehensive tools for de-identifying medical data
to ensure compliance with HIPAA and other privacy regulations.
"""

import re
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class PHIRedactionReport:
    """Report containing PHI de-identification statistics"""
    total_entities: int
    entities_redacted: int
    redacted_entity_types: Dict[str, int]
    timestamp: str
    confidence_score: float


class PHIRedactor:
    """Main class for PHI de-identification in medical data"""
    
    def __init__(self):
        self.phi_patterns = self._initialize_patterns()
        self.redaction_count = 0
        
    def _initialize_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for PHI detection"""
        return {
            'names': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Full names
                r'\b[A-Z]\. [A-Z][a-z]+\b',     # First initial + last name
            ],
            'emails': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'phone_numbers': [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                r'\(\d{3}\) \d{3}-\d{4}'
            ],
            'ssn': [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b'
            ],
            'dates': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
                r'\b\d{4}-\d{2}-\d{2}\b'
            ],
            'addresses': [
                r'\b\d+ [A-Z][a-z]+ (Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln)\b',
                r'\b[A-Z][a-z]+, [A-Z]{2} \d{5}\b'
            ],
            'medical_record_numbers': [
                r'\bMRN[:#]?\s*\d+\b',
                r'\bPatient[:#]?\s*\d+\b'
            ],
            'ip_addresses': [
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ]
        }
    
    def detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """Detect PHI entities in text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PHI entities with their positions and types
        """
        entities = []
        
        for phi_type, patterns in self.phi_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        'type': phi_type,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': self._calculate_confidence(phi_type, match.group())
                    })
        
        return entities
    
    def _calculate_confidence(self, phi_type: str, text: str) -> float:
        """Calculate confidence score for detected PHI entity"""
        base_confidence = 0.7
        
        # Increase confidence based on context
        if phi_type == 'names':
            # Names are more confident if they follow medical titles
            if re.search(r'\b(Dr\.|Patient|Patient:|Patient Name)\b', text, re.IGNORECASE):
                base_confidence += 0.2
        
        elif phi_type == 'emails':
            base_confidence = 0.95  # Email patterns are very reliable
        
        elif phi_type == 'phone_numbers':
            base_confidence = 0.85  # Phone patterns are reliable
        
        elif phi_type == 'ssn':
            base_confidence = 0.90  # SSN patterns are very specific
        
        elif phi_type == 'dates':
            # Lower confidence for dates as they could be medical dates
            base_confidence = 0.6
            
        return min(base_confidence, 1.0)
    
    def redact_phi(self, text: str, preserve_format: bool = True) -> Tuple[str, PHIRedactionReport]:
        """Redact PHI from text
        
        Args:
            text: Input text to redact
            preserve_format: Whether to preserve text length format
            
        Returns:
            Tuple of (redacted_text, report)
        """
        entities = self.detect_phi(text)
        redacted_text = text
        redacted_entity_types = {}
        
        # Sort entities by start position (descending) to avoid position shifts
        entities.sort(key=lambda x: x['start'], reverse=True)
        
        for entity in entities:
            redacted_entity_types[entity['type']] = redacted_entity_types.get(entity['type'], 0) + 1
            
            if preserve_format:
                replacement = self._get_format_preserving_replacement(entity)
            else:
                replacement = f"[{entity['type'].upper()}_REDACTED]"
            
            redacted_text = (
                redacted_text[:entity['start']] + 
                replacement + 
                redacted_text[entity['end']:]
            )
        
        report = PHIRedactionReport(
            total_entities=len(entities),
            entities_redacted=len(entities),
            redacted_entity_types=redacted_entity_types,
            timestamp=datetime.now().isoformat(),
            confidence_score=self._calculate_overall_confidence(entities)
        )
        
        return redacted_text, report
    
    def _get_format_preserving_replacement(self, entity: Dict[str, Any]) -> str:
        """Generate format-preserving replacement for PHI entity"""
        entity_type = entity['type']
        original_text = entity['text']
        
        if entity_type == 'names':
            return "John Doe"  # Generic name
        elif entity_type == 'emails':
            return "email@redacted.com"
        elif entity_type == 'phone_numbers':
            return re.sub(r'\d', 'X', original_text)
        elif entity_type == 'ssn':
            return "XXX-XX-XXXX"
        elif entity_type == 'dates':
            # Preserve date format
            if re.match(r'\d{4}-\d{2}-\d{2}', original_text):
                return "YYYY-MM-DD"
            elif re.match(r'\d{1,2}/\d{1,2}/\d{2,4}', original_text):
                return "MM/DD/YYYY"
            else:
                return "REDACTED_DATE"
        elif entity_type == 'addresses':
            return "REDACTED_ADDRESS"
        elif entity_type == 'medical_record_numbers':
            return "MRN_XXXXXXX"
        elif entity_type == 'ip_addresses':
            return "XXX.XXX.XXX.XXX"
        else:
            return "[REDACTED]"
    
    def _calculate_overall_confidence(self, entities: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score for the de-identification"""
        if not entities:
            return 1.0
        
        total_confidence = sum(entity['confidence'] for entity in entities)
        return total_confidence / len(entities)
    
    def batch_redact(self, data: List[Dict[str, Any]], text_fields: List[str]) -> List[Dict[str, Any]]:
        """Batch process multiple records for PHI de-identification
        
        Args:
            data: List of dictionaries containing medical records
            text_fields: List of field names containing text to redact
            
        Returns:
            List of de-identified records
        """
        redacted_data = []
        total_reports = []
        
        for record in data:
            redacted_record = record.copy()
            record_reports = {}
            
            for field in text_fields:
                if field in record and isinstance(record[field], str):
                    redacted_text, report = self.redact_phi(record[field])
                    redacted_record[field] = redacted_text
                    record_reports[field] = report
            
            redacted_data.append(redacted_record)
            total_reports.append(record_reports)
        
        return redacted_data, total_reports
    
    def validate_redaction(self, original_text: str, redacted_text: str) -> Dict[str, Any]:
        """Validate the effectiveness of PHI de-identification
        
        Args:
            original_text: Original text before redaction
            redacted_text: Redacted text after processing
            
        Returns:
            Validation results including effectiveness metrics
        """
        original_entities = self.detect_phi(original_text)
        remaining_entities = self.detect_phi(redacted_text)
        
        effectiveness = {
            'original_entities': len(original_entities),
            'remaining_entities': len(remaining_entities),
            'effectiveness_rate': (
                (len(original_entities) - len(remaining_entities)) / 
                max(len(original_entities), 1)
            ),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        if remaining_entities:
            effectiveness['remaining_entity_details'] = remaining_entities
            effectiveness['recommendation'] = "Manual review required - PHI still present"
        else:
            effectiveness['recommendation'] = "De-identification appears successful"
        
        return effectiveness


def create_sample_phi_data() -> List[Dict[str, Any]]:
    """Create sample medical data with PHI for testing"""
    return [
        {
            'id': 1,
            'patient_name': 'John Smith',
            'dob': '01/15/1980',
            'email': 'john.smith@email.com',
            'phone': '555-123-4567',
            'ssn': '123-45-6789',
            'address': '123 Main Street, New York, NY 10001',
            'mrn': 'MRN 123456',
            'chief_complaint': 'Patient John Smith presents with chest pain. Contact: john.smith@email.com',
            'notes': 'Follow-up scheduled for 02/15/2023 with Dr. Williams.'
        },
        {
            'id': 2,
            'patient_name': 'Mary Johnson',
            'dob': '03/22/1975',
            'email': 'mary.j@hospital.org',
            'phone': '(555) 987-6543',
            'ssn': '987654321',
            'address': '456 Oak Avenue, Chicago, IL 60601',
            'mrn': 'Patient# 789012',
            'chief_complaint': 'Mary Johnson reports persistent headache since 01/10/2023.',
            'notes': 'Previous history with Dr. Brown. IP: 192.168.1.100'
        }
    ]


if __name__ == "__main__":
    # Demo usage
    redactor = PHIRedactor()
    sample_data = create_sample_phi_data()
    
    print("PHI De-identification Demo")
    print("=" * 50)
    
    for i, record in enumerate(sample_data, 1):
        print(f"\nRecord {i}:")
        print(f"Original: {record['chief_complaint']}")
        
        redacted_text, report = redactor.redact_phi(record['chief_complaint'])
        print(f"Redacted: {redacted_text}")
        print(f"Report: {report}")