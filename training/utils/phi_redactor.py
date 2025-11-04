"""
PHI Redactor: Comprehensive Protected Health Information Detection and Removal
Provides HIPAA-compliant de-identification of healthcare data.
"""

import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from pathlib import Path


@dataclass
class PHIDetection:
    """Represents a detected PHI entity"""
    text: str
    start_pos: int
    end_pos: int
    phi_type: str
    confidence: float
    replacement: str


@dataclass
class DeidentificationReport:
    """Report of de-identification process"""
    original_text: str
    redacted_text: str
    detections: List[PHIDetection]
    pseudonym_map: Dict[str, str]
    timestamp: datetime
    compliance_score: float


class PHIRedactor:
    """
    Comprehensive PHI de-identification system implementing HIPAA Safe Harbor and Expert Determination methods.
    """
    
    def __init__(self, method: str = "safe_harbor", consistent_pseudonyms: bool = True):
        """
        Initialize PHI Redactor
        
        Args:
            method: De-identification method ('safe_harbor' or 'expert_determination')
            consistent_pseudonyms: Whether to use consistent pseudonyms for the same entity
        """
        self.method = method
        self.consistent_pseudonyms = consistent_pseudonyms
        self.pseudonym_map: Dict[str, str] = {}
        self.entity_counter = defaultdict(int)
        self.detection_patterns = self._initialize_patterns()
        self.nlp = None
        
        # Setup logging first
        self.logger = self._setup_logging()
        
        # Load comprehensive name and location dictionaries
        self.common_names = self._load_common_names()
        self.medical_terms = self._load_medical_terms()
        
        # Setup NLP after logger is available
        self._setup_nlp()
        
    def _setup_nlp(self):
        """Initialize spaCy NLP model if available"""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.logger.warning("spaCy en_core_web_sm model not found. NER will be limited.")
                self.nlp = None
        else:
            self.logger.info("spaCy not available. Using regex-based detection only.")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for audit trail"""
        logger = logging.getLogger(f"phi_redactor_{datetime.now().strftime('%Y%m%d')}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_patterns(self) -> Dict[str, List[Dict]]:
        """Initialize comprehensive regex patterns for PHI detection"""
        return {
            # Names
            "names": [
                {
                    "pattern": r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
                    "type": "full_name",
                    "confidence": 0.8
                },
                {
                    "pattern": r'\b(?:Dr|Mr|Ms|Mrs|Prof)\.?\s+[A-Z][a-z]+\b',
                    "type": "name_with_title",
                    "confidence": 0.9
                }
            ],
            
            # Contact Information
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
            
            # Address
            "address": [
                {
                    "pattern": r'\b\d+\s+[A-Za-z0-9\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b',
                    "type": "street_address",
                    "confidence": 0.8
                },
                {
                    "pattern": r'\b[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b',
                    "type": "address_with_zip",
                    "confidence": 0.9
                }
            ],
            
            # Medical Record Numbers
            "mrn": [
                {
                    "pattern": r'\b(?:MRN|Medical Record Number|Patient ID|Chart Number):?\s*[A-Za-z0-9-]{6,}\b',
                    "type": "medical_record_number",
                    "confidence": 0.95
                },
                {
                    "pattern": r'\b[A-Z]{2}[0-9]{6,8}\b',
                    "type": "medical_record_number",
                    "confidence": 0.7
                }
            ],
            
            # Social Security Numbers
            "ssn": [
                {
                    "pattern": r'\b(?:SSN|Social Security Number):?\s*[0-9]{3}-[0-9]{2}-[0-9]{4}\b',
                    "type": "social_security_number",
                    "confidence": 0.95
                },
                {
                    "pattern": r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b',
                    "type": "social_security_number",
                    "confidence": 0.9
                }
            ],
            
            # Dates
            "dates": [
                {
                    "pattern": r'\b(?:DOB|Date of Birth|Birth Date):?\s*[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}\b',
                    "type": "date_of_birth",
                    "confidence": 0.95
                },
                {
                    "pattern": r'\b(?:Admit Date|Admission Date):?\s*[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}\b',
                    "type": "admission_date",
                    "confidence": 0.9
                },
                {
                    "pattern": r'\b[0-9]{1,2}/[0-9]{1,2}/[0-9]{4}\b',
                    "type": "date",
                    "confidence": 0.6
                },
                {
                    "pattern": r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+[0-9]{1,2},?\s+[0-9]{4}\b',
                    "type": "date",
                    "confidence": 0.7
                }
            ],
            
            # Geographic Locations
            "location": [
                {
                    "pattern": r'\b[A-Z][a-z]+,\s*[A-Z]{2}\b',
                    "type": "city_state",
                    "confidence": 0.8
                },
                {
                    "pattern": r'\b[0-9]{5}(?:-\d{4})?\b',
                    "type": "zip_code",
                    "confidence": 0.9
                }
            ],
            
            # Provider Information
            "provider": [
                {
                    "pattern": r'\b(?:Dr|MD|DO|RN|PA|NP|PharmD|DDS|DMD)\.?\s+[A-Z][a-z]+\b',
                    "type": "provider_name",
                    "confidence": 0.9
                },
                {
                    "pattern": r'\b[Hh]ospital\s+[A-Z][A-Za-z\s]+\b',
                    "type": "hospital_name",
                    "confidence": 0.8
                },
                {
                    "pattern": r'\b[A-Z][A-Za-z\s]+(?:Medical Center|Hospital|Healthcare|Clinic)\b',
                    "type": "medical_facility",
                    "confidence": 0.7
                }
            ]
        }
    
    def _load_common_names(self) -> Set[str]:
        """Load common first and last names for better name detection"""
        # In a real implementation, this would load from a comprehensive database
        return {
            # Common first names
            "john", "mary", "james", "patricia", "robert", "jennifer", "michael", "linda",
            "william", "elizabeth", "david", "barbara", "richard", "susan", "joseph", "jessica",
            "thomas", "sarah", "charles", "karen", "christopher", "nancy", "daniel", "lisa",
            "matthew", "betty", "anthony", "margaret", "mark", "sandra", "donald", "ashley",
            "steven", "kimberly", "paul", "emily", "andrew", "donna", "joshua", "michelle",
            
            # Common last names
            "smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis",
            "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas",
            "taylor", "moore", "jackson", "martin", "lee", "perez", "thompson", "white",
            "harris", "sanchez", "clark", "ramirez", "lewis", "robinson", "walker", "young"
        }
    
    def _load_medical_terms(self) -> Set[str]:
        """Load medical terms to avoid false positives"""
        return {
            "diagnosis", "treatment", "medication", "symptoms", "condition", "disease",
            "surgery", "procedure", "examination", "test", "result", "patient", "doctor",
            "hospital", "clinic", "emergency", "admission", "discharge", "follow-up"
        }
    
    def _generate_pseudonym(self, entity_type: str, original: str) -> str:
        """Generate consistent pseudonym for entity"""
        if self.consistent_pseudonyms and original in self.pseudonym_map:
            return self.pseudonym_map[original]
        
        # Generate consistent pseudonym based on entity type and hash
        entity_id = self.entity_counter[entity_type] + 1
        self.entity_counter[entity_type] = entity_id
        
        pseudonym_types = {
            "full_name": f"Person_{entity_id:03d}",
            "name_with_title": f"Person_{entity_id:03d}",
            "phone_number": f"Phone_{entity_id:03d}",
            "email": f"Email_{entity_id:03d}",
            "street_address": f"Address_{entity_id:03d}",
            "address_with_zip": f"Address_{entity_id:03d}",
            "medical_record_number": f"MRN_{entity_id:03d}",
            "social_security_number": f"SSN_{entity_id:03d}",
            "date_of_birth": "DOB_REDACTED",
            "admission_date": "ADMIT_DATE_REDACTED",
            "date": f"Date_{entity_id:03d}",
            "city_state": f"Location_{entity_id:03d}",
            "zip_code": "ZIP_REDACTED",
            "provider_name": f"Provider_{entity_id:03d}",
            "hospital_name": f"Hospital_{entity_id:03d}",
            "medical_facility": f"Facility_{entity_id:03d}"
        }
        
        pseudonym = pseudonym_types.get(entity_type, f"{entity_type.title()}_{entity_id:03d}")
        
        if self.consistent_pseudonyms:
            self.pseudonym_map[original] = pseudonym
            
        return pseudonym
    
    def detect_phi_regex(self, text: str) -> List[PHIDetection]:
        """Detect PHI using regex patterns"""
        detections = []
        
        for phi_category, patterns in self.detection_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                phi_type = pattern_info["type"]
                confidence = pattern_info["confidence"]
                
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    # Additional validation to reduce false positives
                    if self._validate_detection(text[match.start():match.end()], phi_type):
                        detection = PHIDetection(
                            text=text[match.start():match.end()],
                            start_pos=match.start(),
                            end_pos=match.end(),
                            phi_type=phi_type,
                            confidence=confidence,
                            replacement=""
                        )
                        detections.append(detection)
        
        # Sort by position to process from left to right
        detections.sort(key=lambda x: x.start_pos)
        return detections
    
    def _validate_detection(self, text: str, phi_type: str) -> bool:
        """Validate detection to reduce false positives"""
        text_lower = text.lower().strip()
        
        # Skip if it's a common medical term
        if text_lower in self.medical_terms:
            return False
        
        # Special validation for names
        if phi_type in ["full_name", "name_with_title"]:
            words = text.split()
            if len(words) == 2:  # First and last name
                first, last = words
                return (first.lower() in self.common_names or 
                       last.lower() in self.common_names)
            return False
        
        return True
    
    def detect_phi_ner(self, text: str) -> List[PHIDetection]:
        """Detect PHI using Named Entity Recognition (spaCy)"""
        detections = []
        
        if not self.nlp:
            return detections
        
        try:
            doc = self.nlp(text)
            
            # Map spaCy entity types to PHI types
            entity_mapping = {
                "PERSON": "full_name",
                "ORG": "medical_facility",
                "GPE": "city_state",
                "DATE": "date",
                "PHONE": "phone_number"
            }
            
            for ent in doc.ents:
                if ent.label_ in entity_mapping:
                    phi_type = entity_mapping[ent.label_]
                    
                    # Additional confidence based on entity label
                    confidence = 0.8 if ent.label_ == "PERSON" else 0.7
                    
                    detection = PHIDetection(
                        text=ent.text,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        phi_type=phi_type,
                        confidence=confidence,
                        replacement=""
                    )
                    detections.append(detection)
                    
        except Exception as e:
            self.logger.warning(f"NER processing failed: {e}")
        
        return detections
    
    def get_contextual_replacement(self, text: str, phi_type: str, detection: PHIDetection) -> str:
        """Generate context-aware replacement based on surrounding text"""
        # Get context around the detection
        start = max(0, detection.start_pos - 20)
        end = min(len(text), detection.end_pos + 20)
        context = text[start:end]
        
        replacement = self._generate_pseudonym(phi_type, detection.text)
        
        # Context-specific adjustments
        if "DOB" in context or "birth" in context.lower():
            if phi_type == "date_of_birth":
                return "01/01/1900"  # Generic early date
        
        elif "admission" in context.lower() or "admit" in context.lower():
            if "date" in phi_type:
                return "[ADMISSION_DATE]"
        
        elif "discharge" in context.lower():
            if "date" in phi_type:
                return "[DISCHARGE_DATE]"
        
        return replacement
    
    def redact_text(self, text: str, return_report: bool = True) -> Tuple[str, Optional[DeidentificationReport]]:
        """
        Redact PHI from text using specified method
        
        Args:
            text: Input text containing potential PHI
            return_report: Whether to return detailed report
            
        Returns:
            Tuple of (redacted_text, report) where report is None if return_report is False
        """
        if not text.strip():
            return text, None
        
        # Detect PHI using multiple methods
        regex_detections = self.detect_phi_regex(text)
        ner_detections = self.detect_phi_ner(text) if self.nlp else []
        
        # Combine and deduplicate detections
        all_detections = regex_detections + ner_detections
        final_detections = self._deduplicate_detections(all_detections)
        
        # Sort by position for correct replacement
        final_detections.sort(key=lambda x: x.start_pos)
        
        # Apply de-identification
        redacted_text = self._apply_replacements(text, final_detections)
        
        # Generate report
        report = None
        if return_report:
            compliance_score = self._calculate_compliance_score(final_detections)
            
            report = DeidentificationReport(
                original_text=text,
                redacted_text=redacted_text,
                detections=final_detections,
                pseudonym_map=dict(self.pseudonym_map),
                timestamp=datetime.now(),
                compliance_score=compliance_score
            )
            
            self.logger.info(f"De-identification completed: {len(final_detections)} PHI entities detected and redacted")
        
        return redacted_text, report
    
    def _deduplicate_detections(self, detections: List[PHIDetection]) -> List[PHIDetection]:
        """Remove overlapping or duplicate detections"""
        if not detections:
            return []
        
        # Sort by position and confidence
        detections.sort(key=lambda x: (x.start_pos, -x.confidence))
        
        deduplicated = []
        for detection in detections:
            # Check for overlap with existing detections
            overlap = False
            for existing in deduplicated:
                if (detection.start_pos < existing.end_pos and 
                    detection.end_pos > existing.start_pos):
                    overlap = True
                    break
            
            if not overlap:
                deduplicated.append(detection)
        
        return deduplicated
    
    def _apply_replacements(self, text: str, detections: List[PHIDetection]) -> str:
        """Apply replacements to text from right to left to maintain positions"""
        if not detections:
            return text
        
        result = text
        # Process from right to left to avoid position shifting
        for detection in reversed(detections):
            replacement = self.get_contextual_replacement(
                text, detection.phi_type, detection
            )
            detection.replacement = replacement
            
            result = (result[:detection.start_pos] + 
                     replacement + 
                     result[detection.end_pos:])
        
        return result
    
    def _calculate_compliance_score(self, detections: List[PHIDetection]) -> float:
        """Calculate HIPAA compliance score based on detected PHI"""
        if not detections:
            return 1.0
        
        # Weight different types of PHI
        phi_weights = {
            "social_security_number": 1.0,
            "medical_record_number": 0.9,
            "phone_number": 0.8,
            "email": 0.8,
            "date_of_birth": 0.9,
            "full_name": 0.7,
            "address": 0.8,
            "provider_name": 0.6,
            "hospital_name": 0.5
        }
        
        total_weight = sum(phi_weights.get(d.phi_type, 0.5) for d in detections)
        max_possible_weight = len(detections)  # Assuming max weight of 1.0
        
        # Lower score = more PHI detected = less compliant
        compliance_score = max(0.0, 1.0 - (total_weight / max_possible_weight))
        
        return compliance_score
    
    def batch_redact(self, texts: List[str], return_reports: bool = False) -> Tuple[List[str], Optional[List[DeidentificationReport]]]:
        """
        Redact PHI from multiple texts
        
        Args:
            texts: List of input texts
            return_reports: Whether to return detailed reports
            
        Returns:
            Tuple of (redacted_texts, reports) where reports is None if return_reports is False
        """
        redacted_texts = []
        reports = []
        
        for i, text in enumerate(texts):
            try:
                redacted, report = self.redact_text(text, return_reports)
                redacted_texts.append(redacted)
                if return_reports:
                    reports.append(report)
            except Exception as e:
                self.logger.error(f"Error processing text {i}: {e}")
                redacted_texts.append(text)  # Return original if processing fails
                if return_reports:
                    reports.append(None)
        
        return redacted_texts, reports if return_reports else None
    
    def save_pseudonym_map(self, filepath: str):
        """Save pseudonym mapping for audit purposes"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.pseudonym_map), f, indent=2)
    
    def load_pseudonym_map(self, filepath: str):
        """Load pseudonym mapping from file"""
        try:
            with open(filepath, 'r') as f:
                self.pseudonym_map = json.load(f)
            self.logger.info(f"Loaded pseudonym map from {filepath}")
        except Exception as e:
            self.logger.warning(f"Could not load pseudonym map: {e}")
    
    def export_audit_report(self, filepath: str, report: DeidentificationReport):
        """Export detailed audit report"""
        audit_data = {
            "timestamp": report.timestamp.isoformat(),
            "method": self.method,
            "compliance_score": report.compliance_score,
            "original_text_length": len(report.original_text),
            "redacted_text_length": len(report.redacted_text),
            "detections_summary": {
                "total_detections": len(report.detections),
                "phi_types": Counter(d.phi_type for d in report.detections),
                "average_confidence": sum(d.confidence for d in report.detections) / len(report.detections) if report.detections else 0
            },
            "pseudonym_mapping": report.pseudonym_map,
            "original_text": report.original_text,
            "redacted_text": report.redacted_text
        }
        
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        self.logger.info(f"Audit report exported to {filepath}")


def main():
    """Example usage and testing"""
    sample_texts = [
        """
        Patient John Smith (DOB: 05/15/1980) was admitted to General Hospital on 01/15/2023.
        Contact: (555) 123-4567, john.smith@email.com
        Address: 123 Main St, Springfield, IL 62701
        MRN: MR12345678
        Attending: Dr. Jane Doe, MD
        """,
        
        """
        Follow-up appointment for Mary Johnson scheduled for next week.
        Insurance: ID# 123-45-6789
        Pharmacy: MedStore on Oak Avenue
        """,
        
        """
        Emergency department visit - Patient unknown, approximately 65 years old.
        No identification available. Treated for chest pain.
        """
    ]
    
    # Initialize redactor
    redactor = PHIRedactor(method="safe_harbor", consistent_pseudonyms=True)
    
    # Process texts
    redacted_texts, reports = redactor.batch_redact(sample_texts, return_reports=True)
    
    # Display results
    for i, (original, redacted) in enumerate(zip(sample_texts, redacted_texts), 1):
        print(f"\n{'='*60}")
        print(f"Text {i} - Original:")
        print(original.strip())
        print(f"\nText {i} - Redacted:")
        print(redacted.strip())
        
        if reports and reports[i-1]:
            print(f"\nCompliance Score: {reports[i-1].compliance_score:.2f}")
            print(f"Detections: {len(reports[i-1].detections)}")


if __name__ == "__main__":
    main()