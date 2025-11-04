"""Content Filter for Safety Checks"""
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContentFilter:
    """Filters AI responses for prohibited content"""
    
    DIAGNOSIS_PATTERNS = [
        r"you have (a|an)?\s*\w+",
        r"this is (a|an)?\s*\w+",
        r"diagnosed with",
        r"you are suffering from"
    ]
    
    PRESCRIPTION_PATTERNS = [
        r"take \d+mg",
        r"you should take",
        r"start.*medication",
        r"prescribe"
    ]
    
    def __init__(self, config):
        self.config = config
        logger.info("Content filter initialized")
    
    def check(self, text: str) -> List[Dict[str, Any]]:
        """
        Check text for safety violations.
        
        Returns:
            List of violations found
        """
        violations = []
        
        if self.config.block_diagnosis:
            for pattern in self.DIAGNOSIS_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({
                        "type": "diagnosis",
                        "pattern": pattern,
                        "severity": "high"
                    })
        
        if self.config.block_prescription:
            for pattern in self.PRESCRIPTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    violations.append({
                        "type": "prescription",
                        "pattern": pattern,
                        "severity": "high"
                    })
        
        return violations
    
    def correct(self, text: str, violations: List[Dict]) -> str:
        """Generate corrected version of text"""
        # TODO: Implement smart correction
        return text
