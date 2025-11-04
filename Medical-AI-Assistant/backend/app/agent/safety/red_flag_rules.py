"""Red Flag Rules for Emergency Detection"""
import logging
from typing import List

logger = logging.getLogger(__name__)

class RedFlagRules:
    """Rules for detecting emergency red flag symptoms"""
    
    RED_FLAGS = {
        "chest_pain": ["chest pain", "chest pressure", "pain radiating"],
        "breathing": ["difficulty breathing", "can't breathe", "shortness of breath"],
        "neurological": ["worst headache", "confusion", "altered mental", "stroke"],
        "bleeding": ["severe bleeding", "heavy bleeding", "hemorrhage"],
        "consciousness": ["loss of consciousness", "passed out", "fainted"],
        "psychiatric": ["suicidal", "harm myself", "end my life"]
    }
    
    def __init__(self):
        logger.info("Red flag rules initialized")
    
    def check(self, text: str) -> List[str]:
        """
        Check text for red flag symptoms.
        
        Returns:
            List of detected red flags
        """
        detected = []
        text_lower = text.lower()
        
        for category, patterns in self.RED_FLAGS.items():
            for pattern in patterns:
                if pattern in text_lower:
                    detected.append(category)
                    break
        
        return detected
    
    def get_urgency(self, red_flags: List[str]) -> str:
        """Determine urgency level based on red flags"""
        if red_flags:
            return "immediate"
        return "routine"
