"""
Data Augmentation Module for Medical AI Training

This module provides comprehensive text augmentation techniques for expanding training datasets
while maintaining medical accuracy, context, and safety.
"""

import re
import random
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict, Counter, deque
import unicodedata
import hashlib
from pathlib import Path
import concurrent.futures
from threading import Lock
import time


@dataclass
class AugmentationConfig:
    """Enhanced configuration for data augmentation"""
    # Basic augmentation probabilities
    synonym_probability: float = 0.3
    paraphrase_probability: float = 0.4
    back_translation_probability: float = 0.2
    masked_lm_probability: float = 0.15
    style_transfer_probability: float = 0.25
    
    # Medical-specific augmentation
    symptom_variation_probability: float = 0.4
    demographic_diversity_probability: float = 0.3
    scenario_augmentation_probability: float = 0.5
    conversation_flow_probability: float = 0.35
    emergency_routine_balance_probability: float = 0.4
    
    # Quality control
    diversity_threshold: float = 0.8
    semantic_similarity_threshold: float = 0.7
    medical_accuracy_threshold: float = 0.95
    safety_constraint_probability: float = 0.9
    coherence_check_probability: float = 0.8
    
    # Dataset balance
    max_augmentations: int = 5
    preserve_medical_terms: bool = True
    context_aware: bool = True
    medical_term_preservation_rate: float = 0.8
    
    # Emergency vs Routine balancing
    emergency_case_target_ratio: float = 0.3
    routine_case_target_ratio: float = 0.7
    
    # Patient demographics
    age_group_diversity: bool = True
    gender_diversity: bool = True
    cultural_diversity: bool = True
    socioeconomic_diversity: bool = True
    
    # Quality monitoring
    enable_quality_checks: bool = True
    enable_safety_validation: bool = True
    enable_coherence_tracking: bool = True
    min_text_length: int = 10
    max_text_length: int = 1000
    
    # Performance optimization
    batch_size: int = 100
    max_workers: int = 4
    enable_caching: bool = True
    cache_dir: str = ".augmentation_cache"


class MedicalTerminologyReplacer:
    """Handles replacement of medical terms with appropriate alternatives"""
    
    def __init__(self):
        self.medical_synonyms = self._initialize_medical_synonyms()
        self.symptom_descriptions = self._initialize_symptom_descriptions()
        self.body_parts = self._initialize_body_parts()
        self.medical_units = self._initialize_medical_units()
    
    def _initialize_medical_synonyms(self) -> Dict[str, List[str]]:
        """Initialize medical term synonyms"""
        return {
            # Symptoms
            "pain": ["ache", "discomfort", "soreness", "tenderness"],
            "fever": ["elevated temperature", "high temperature", "body heat"],
            "cough": ["tickle in throat", "throat clearing", "bronchial irritation"],
            "headache": ["head pain", "cephalalgia", "migraine", "head pressure"],
            "nausea": ["queasiness", "stomach upset", "motion sickness feeling"],
            "dizziness": ["lightheadedness", "vertigo", "balance issues"],
            "fatigue": ["tiredness", "exhaustion", "weakness", "low energy"],
            "shortness of breath": ["breathing difficulty", "dyspnea", "air hunger"],
            
            # Severity levels
            "severe": ["intense", "extreme", "serious", "bad"],
            "mild": ["slight", "minor", "light", "gentle"],
            "moderate": ["medium", "intermediate", "average", "medium level"],
            
            # Duration terms
            "acute": ["sudden", "recent", "immediate", "sharp"],
            "chronic": ["ongoing", "persistent", "long-term", "recurring"],
            "sudden": ["abrupt", "immediate", "rapid", "quick"],
            "gradual": ["slow", "progressive", "incremental", "step by step"],
            
            # Body parts
            "chest": ["torso", "upper body", "rib cage area"],
            "stomach": ["abdomen", "belly", "abdominal area"],
            "head": ["skull", "cranium", "brain area"],
            "back": ["spine", "vertebral column", "lumbar region"],
            "arm": ["upper limb", "shoulder to hand", "brachial area"],
            
            # Medical actions
            "examine": ["check", "inspect", "assess", "evaluate"],
            "diagnose": ["identify", "determine", "assess", "conclude"],
            "treat": ["manage", "address", "handle", "care for"],
            "monitor": ["watch", "observe", "track", "supervise"],
            
            # Emergency terms
            "emergency": ["urgent situation", "critical condition", "immediate need"],
            "urgent": ["critical", "pressing", "immediate", "serious"],
            "stable": ["steady", "controlled", "consistent", "normal"]
        }
    
    def _initialize_symptom_descriptions(self) -> Dict[str, List[str]]:
        """Initialize symptom description variations"""
        return {
            "chest pain": [
                "tightness in the chest",
                "pressure across the chest",
                "sharp pain in the chest area",
                "discomfort in the chest",
                "squeezing sensation in chest"
            ],
            "shortness of breath": [
                "difficulty catching my breath",
                "feeling out of breath",
                "can't seem to get enough air",
                "breathing feels labored",
                "gasping for air"
            ],
            "stomach pain": [
                "cramping in my stomach",
                "abdominal discomfort",
                "belly ache",
                "stomach cramps",
                "pain in my abdomen"
            ],
            "headache": [
                "throbbing in my head",
                "pressure in my skull",
                "head pain",
                "temple pain",
                "behind my eyes hurts"
            ]
        }
    
    def _initialize_body_parts(self) -> Dict[str, Dict[str, str]]:
        """Initialize body part terminology"""
        return {
            "chest": {"formal": "thorax", "informal": "chest", "clinical": "pectoral region"},
            "stomach": {"formal": "abdomen", "informal": "stomach", "clinical": "abdominal cavity"},
            "head": {"formal": "cranium", "informal": "head", "clinical": "cephalic region"},
            "back": {"formal": "spine", "informal": "back", "clinical": "vertebral column"}
        }
    
    def _initialize_medical_units(self) -> Dict[str, List[str]]:
        """Initialize medical measurement units"""
        return {
            "temperature": ["°F", "°C", "degrees Fahrenheit", "degrees Celsius"],
            "blood pressure": ["mmHg", "millimeters of mercury"],
            "heart rate": ["bpm", "beats per minute"],
            "weight": ["lbs", "pounds", "kg", "kilograms"],
            "height": ["ft", "feet", "cm", "centimeters"]
        }
    
    def replace_medical_term(self, text: str, term: str, preserve_probability: float = 0.5) -> str:
        """Replace medical term with appropriate alternative"""
        if random.random() < preserve_probability:
            return text
        
        # Find exact matches
        if term in self.medical_synonyms:
            replacement = random.choice(self.medical_synonyms[term])
            text = re.sub(r'\b' + re.escape(term) + r'\b', replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def replace_symptom_description(self, text: str) -> str:
        """Replace symptom with alternative description"""
        for symptom, alternatives in self.symptom_descriptions.items():
            if symptom.lower() in text.lower():
                replacement = random.choice(alternatives)
                text = re.sub(re.escape(symptom), replacement, text, flags=re.IGNORECASE)
        
        return text


class ParaphraseGenerator:
    """Generates paraphrases of medical text while preserving meaning"""
    
    def __init__(self):
        self.phrase_patterns = self._initialize_phrase_patterns()
        self.question_variations = self._initialize_question_variations()
        
    def _initialize_phrase_patterns(self) -> Dict[str, List[str]]:
        """Initialize common phrase patterns for paraphrase"""
        return {
            "symptom_introduction": [
                "I'm experiencing",
                "I'm having",
                "I've been dealing with",
                "I'm suffering from",
                "I feel",
                "I'm noticing"
            ],
            "time_descriptors": [
                "for the past few days",
                "since this morning",
                "over the last week",
                "for several hours",
                "since yesterday",
                "throughout the day"
            ],
            "severity_expressions": [
                "quite severe",
                "really bad",
                "extremely uncomfortable",
                "quite distressing",
                "really concerning",
                "extremely bothersome"
            ],
            "question_starters": [
                "Can you help me understand",
                "Could you explain",
                "What does this mean",
                "Is this normal",
                "Should I be worried about"
            ]
        }
    
    def _initialize_question_variations(self) -> Dict[str, List[str]]:
        """Initialize question variations"""
        return {
            "pain_level": [
                "How would you rate your pain on a scale of 1 to 10?",
                "On a scale from 1 to 10, how severe is your pain?",
                "Can you describe your pain level from 1 to 10?",
                "How intense is your pain, rated 1 through 10?"
            ],
            "duration": [
                "How long have you been experiencing these symptoms?",
                "When did you first notice these symptoms?",
                "How long has this been going on?",
                "Since when have you been feeling this way?"
            ],
            "medications": [
                "Are you currently taking any medications?",
                "Do you take any prescription drugs?",
                "What medications are you on right now?",
                "Are you taking any treatments or drugs?"
            ],
            "allergies": [
                "Do you have any known allergies?",
                "Are you allergic to anything?",
                "Have you had any allergic reactions before?",
                "Do you have any drug allergies?"
            ]
        }
    
    def paraphrase_text(self, text: str, preserve_medical_terms: bool = True) -> str:
        """Generate paraphrase of text"""
        # Simple word substitution based on patterns
        words = text.split()
        
        # Apply phrase patterns
        for i, word in enumerate(words):
            if word.lower() in self.phrase_patterns:
                if random.random() < 0.3:  # 30% chance to replace
                    replacement = random.choice(self.phrase_patterns[word.lower()])
                    words[i] = replacement
        
        # Join and clean up
        paraphrased = ' '.join(words)
        
        # Basic sentence restructuring
        if "I'm" in paraphrased and random.random() < 0.2:
            paraphrased = paraphrased.replace("I'm", "I am")
        if "I've" in paraphrased and random.random() < 0.2:
            paraphrased = paraphrased.replace("I've", "I have")
        
        return paraphrased
    
    def vary_question(self, question: str) -> str:
        """Vary the phrasing of questions"""
        question_lower = question.lower()
        
        for key, variations in self.question_variations.items():
            if key in question_lower:
                return random.choice(variations)
        
        # Generic question variations
        if "how" in question_lower:
            return question.replace("How", "In what way")
        elif "what" in question_lower:
            return question.replace("What", "Which")
        elif "when" in question_lower:
            return question.replace("When", "At what time")
        
        return question


class AdversarialExampleGenerator:
    """Generates adversarial examples for robustness testing"""
    
    def __init__(self):
        self.medical_errors = self._initialize_medical_errors()
        self.confusing_terms = self._initialize_confusing_terms()
        self.contradictory_statements = self._initialize_contradictory_statements()
    
    def _initialize_medical_errors(self) -> Dict[str, List[str]]:
        """Initialize common medical misconceptions"""
        return {
            "symptom_duration": [
                "I've had this for months now",
                "This just started yesterday",
                "It's been going on for years"
            ],
            "pain_description": [
                "It's like a dull ache",
                "It's sharp and stabbing",
                "It's a burning sensation",
                "It's throbbing"
            ],
            "symptom_severity": [
                "It's tolerable",
                "It's unbearable",
                "It's manageable",
                "It's extremely severe"
            ]
        }
    
    def _initialize_confusing_terms(self) -> Dict[str, List[str]]:
        """Initialize confusing medical terminology"""
        return {
            "pressure": ["compression", "constriction", "squeezing"],
            "swelling": ["edema", "inflammation", "enlargement"],
            "infection": ["inflammation", "irritation", "contamination"]
        }
    
    def _initialize_contradictory_statements(self) -> List[str]:
        """Initialize contradictory statements"""
        return [
            "On second thought, it's not that bad",
            "Actually, I think I was wrong about that",
            "Wait, let me correct myself",
            "I may have misstated that"
        ]
    
    def introduce_medical_error(self, text: str) -> str:
        """Introduce subtle medical errors"""
        if random.random() > 0.3:  # 30% chance
            return text
        
        # Add contradictory statements
        if random.random() < 0.5:
            contradiction = random.choice(self.contradictory_statements)
            text = f"{text} {contradiction}."
        
        # Alter symptom severity
        severity_words = ["severe", "mild", "moderate"]
        for severity in severity_words:
            if severity in text.lower():
                if random.random() < 0.3:
                    # Replace with different severity
                    other_severities = [s for s in severity_words if s != severity]
                    replacement = random.choice(other_severities)
                    text = re.sub(r'\b' + re.escape(severity) + r'\b', replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def add_confusing_terminology(self, text: str) -> str:
        """Add confusing medical terms"""
        for standard, alternatives in self.confusing_terms.items():
            if standard in text.lower() and random.random() < 0.2:
                replacement = random.choice(alternatives)
                text = re.sub(r'\b' + re.escape(standard) + r'\b', replacement, text, flags=re.IGNORECASE)
        
        return text


class BackTranslationGenerator:
    """Generates back-translation variations for text diversity"""
    
    def __init__(self):
        self.translation_cache = {}
        self.supported_languages = ['fr', 'es', 'de', 'it', 'pt', 'nl']
        
    def back_translate(self, text: str, target_languages: List[str] = None) -> str:
        """Simulate back-translation through multiple languages"""
        if target_languages is None:
            target_languages = random.sample(self.supported_languages, 2)
        
        translated = text
        for lang in target_languages:
            # Simulate translation (in practice, would use actual translation API)
            translated = self._simulate_translation(translated, lang)
        
        return translated
    
    def _simulate_translation(self, text: str, lang: str) -> str:
        """Simulate translation - in practice would use actual translation service"""
        # For demonstration, we'll use simple word substitutions
        common_phrases = {
            'pain': {'fr': 'douleur', 'es': 'dolor', 'de': 'schmerz'},
            'headache': {'fr': 'mal de tête', 'es': 'dolor de cabeza', 'de': 'kopfschmerzen'},
            'fever': {'fr': 'fièvre', 'es': 'fiebre', 'de': 'fieber'},
            'cough': {'fr': 'toux', 'es': 'tos', 'de': 'husten'}
        }
        
        translated = text.lower()
        for english, translations in common_phrases.items():
            if english in translated and lang in translations:
                translated = translated.replace(english, translations[lang])
        
        return translated.capitalize()


class StyleTransferGenerator:
    """Performs style transfer for medical text variations"""
    
    def __init__(self):
        self.formal_to_informal = {
            "experience": "feel",
            "demonstrate": "show",
            "approximately": "about",
            "therefore": "so",
            "however": "but",
            "furthermore": "also",
            "utilize": "use",
            "commence": "start"
        }
        
        self.informal_to_formal = {v: k for k, v in self.formal_to_informal.items()}
        
        self.tone_variations = {
            "empathetic": ["I understand", "I can see", "I hear", "That sounds"],
            "professional": ["Based on", "Our assessment indicates", "The results suggest"],
            "urgent": ["This requires immediate", "Urgent attention needed", "Emergency action required"],
            "calming": ["Don't worry", "This is normal", "Take it easy", "You'll be fine"]
        }
    
    def transfer_style(self, text: str, source_style: str, target_style: str) -> str:
        """Transfer text from one style to another"""
        
        # Convert formality level
        if source_style == "formal" and target_style == "informal":
            for formal, informal in self.formal_to_informal.items():
                text = re.sub(r'\b' + re.escape(formal) + r'\b', informal, text, flags=re.IGNORECASE)
        elif source_style == "informal" and target_style == "formal":
            for informal, formal in self.informal_to_formal.items():
                text = re.sub(r'\b' + re.escape(informal) + r'\b', formal, text, flags=re.IGNORECASE)
        
        # Apply tone variations
        if target_style in self.tone_variations:
            prefix = random.choice(self.tone_variations[target_style])
            if random.random() < 0.3:
                text = f"{prefix}, {text.lower()}"
        
        return text
    
    def vary_tone(self, text: str) -> str:
        """Vary the tone of medical responses"""
        tone_types = ["empathetic", "professional", "urgent", "calming"]
        
        if random.random() < 0.4:  # 40% chance to change tone
            new_tone = random.choice(tone_types)
            text = self.transfer_style(text, "neutral", new_tone)
        
        return text


class MaskedLanguageModel:
    """Simulates masked language modeling for contextual augmentation"""
    
    def __init__(self):
        self.medical_tokens = set([
            "symptom", "diagnosis", "treatment", "medication", "dosage", "pain",
            "fever", "cough", "headache", "nausea", "dizziness", "fatigue"
        ])
        
        self.context_predictions = {
            "I have severe _____ in my chest": ["pain", "discomfort", "tightness"],
            "My _____ has been elevated": ["temperature", "fever", "blood pressure"],
            "I need to take this _____ twice daily": ["medication", "pill", "medicine"],
            "The pain is _____ and constant": ["sharp", "dull", "throbbing"]
        }
    
    def mask_and_predict(self, text: str) -> str:
        """Apply masked language modeling to text"""
        
        # Find context where masking would be appropriate
        for pattern, predictions in self.context_predictions.items():
            if "_____" in pattern:
                # Replace blank with actual word from text
                base_pattern = pattern.replace("_____", "(.+)")
                match = re.search(base_pattern, text, re.IGNORECASE)
                if match:
                    original_word = match.group(1)
                    # Mask and provide alternative
                    if random.random() < 0.3:
                        new_word = random.choice(predictions)
                        text = text.replace(original_word, new_word, 1)
        
        return text


class QualityControlValidator:
    """Validates quality of augmented data"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.safety_patterns = self._initialize_safety_patterns()
        self.medical_terms = self._initialize_medical_terms()
        self.quality_cache = {}
    
    def _initialize_safety_patterns(self) -> List[str]:
        """Initialize safety constraint patterns"""
        return [
            r"\b(?:suicide|kill myself|end my life)\b",
            r"\b(?:overdose|pill|medication)\s+(?:\d+|many|lots)\b",
            r"\b(?:emergency|urgent|critical)\s+(?:care|treatment|help)\b",
            r"\bsevere|intense|extreme\s+pain\b"
        ]
    
    def _initialize_medical_terms(self) -> Dict[str, List[str]]:
        """Initialize medical terminology for accuracy checking"""
        return {
            "cardiovascular": ["heart", "blood pressure", "pulse", "cardiac"],
            "respiratory": ["breathing", "lung", "airway", "respiratory"],
            "neurological": ["brain", "nerve", "neurological", "cognitive"],
            "gastrointestinal": ["stomach", "digestive", "abdominal", "intestinal"]
        }
    
    def validate_semantic_similarity(self, original: str, augmented: str) -> float:
        """Calculate semantic similarity between texts"""
        
        def jaccard_similarity(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0
        
        def cosine_similarity(text1: str, text2: str) -> float:
            # Simple word frequency comparison
            words1 = text1.lower().split()
            words2 = text2.lower().split()
            
            vocab = set(words1 + words2)
            vec1 = [words1.count(word) for word in vocab]
            vec2 = [words2.count(word) for word in vocab]
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5
            
            return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0
        
        jaccard = jaccard_similarity(original, augmented)
        cosine = cosine_similarity(original, augmented)
        
        return (jaccard + cosine) / 2
    
    def validate_medical_accuracy(self, text: str) -> float:
        """Validate medical accuracy of text"""
        medical_score = 0
        
        # Check for medical terminology usage
        words = text.lower().split()
        medical_word_count = sum(1 for word in words if word in self.medical_terms)
        medical_score += min(medical_word_count * 0.1, 0.3)
        
        # Check for appropriate medical context
        if any(term in text.lower() for category_terms in self.medical_terms.values() 
               for term in category_terms):
            medical_score += 0.4
        
        # Check for medical accuracy patterns
        accuracy_patterns = [
            r"\d+\s*(?:mg|mcg|g|ml|cc)\b",  # Dosage patterns
            r"\d+\s*(?:°F|°C|beats? per minute|mmHg)\b",  # Vital signs
            r"(?:heart rate|blood pressure|temperature)\s+is\s+\d+"  # Vital signs statements
        ]
        
        for pattern in accuracy_patterns:
            if re.search(pattern, text.lower()):
                medical_score += 0.2
        
        return min(medical_score, 1.0)
    
    def validate_safety_constraints(self, text: str) -> Tuple[bool, List[str]]:
        """Check safety constraints"""
        violations = []
        
        for pattern in self.safety_patterns:
            if re.search(pattern, text.lower()):
                violations.append(f"Safety pattern match: {pattern}")
        
        # Check for inappropriate medical advice
        inappropriate_advice = [
            r"stop taking.*medication.*immediately",
            r"ignore.*doctor.*advice",
            r"self-medicate.*without.*supervision"
        ]
        
        for pattern in inappropriate_advice:
            if re.search(pattern, text.lower()):
                violations.append(f"Inappropriate advice: {pattern}")
        
        return len(violations) == 0, violations
    
    def validate_coherence(self, text: str) -> float:
        """Validate text coherence"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check sentence length consistency
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        coherence_score = max(0, 1 - variance / (avg_length ** 2))
        
        # Check for logical connectors
        connectors = ["however", "therefore", "furthermore", "additionally", "in contrast"]
        connector_score = sum(1 for s in sentences for connector in connectors 
                             if connector in s.lower()) / len(sentences)
        
        return (coherence_score + min(connector_score, 0.5)) / 1.5


class DemographicAugmentor:
    """Augments patient demographics for diversity"""
    
    def __init__(self):
        self.age_groups = {
            "child": {"range": (0, 17), "terms": ["my child", "little one", "kiddo"]},
            "adult": {"range": (18, 64), "terms": ["myself", "I", "adult"]},
            "elderly": {"range": (65, 120), "terms": ["elderly", "senior", "aging"]}
        }
        
        self.cultural_variations = {
            "direct": ["I have", "I'm experiencing", "I feel"],
            "indirect": ["It seems like", "I might be having", "I could be feeling"],
            "formal": ["The patient reports", "Observed symptoms include", "Experiencing"],
            "informal": ["Doc, I got", "Hey, I've been", "So I've got"]
        }
    
    def vary_demographics(self, text: str, target_demographic: Dict[str, Any]) -> str:
        """Vary text based on patient demographics"""
        
        # Age-based variations
        age_group = target_demographic.get("age_group", "adult")
        if age_group in self.age_groups:
            terms = self.age_groups[age_group]["terms"]
            if random.random() < 0.3:
                # Replace first person pronouns based on age
                if age_group == "child":
                    text = text.replace("I ", "my child ")
                elif age_group == "elderly":
                    text = text.replace("I feel", "I am feeling")
        
        # Cultural style variations
        cultural_style = target_demographic.get("cultural_style", "direct")
        if cultural_style in self.cultural_variations:
            variations = self.cultural_variations[cultural_style]
            if "I have" in text and random.random() < 0.4:
                text = text.replace("I have", random.choice(variations))
        
        return text


class DataAugmentor:
    """Enhanced main data augmentation class"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.terminology_replacer = MedicalTerminologyReplacer()
        self.paraphrase_generator = ParaphraseGenerator()
        self.adversarial_generator = AdversarialExampleGenerator()
        self.back_translator = BackTranslationGenerator()
        self.style_transfer = StyleTransferGenerator()
        self.masked_lm = MaskedLanguageModel()
        self.quality_validator = QualityControlValidator(self.config)
        self.demographic_augmentor = DemographicAugmentor()
        
        self.generated_texts = []
        self.quality_scores = []
        self.safety_violations = []
        self.cache = {}
        self._cache_lock = Lock() if self.config.enable_caching else None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def augment_conversation(self, conversation: List[Dict[str, Any]], 
                           target_demographics: List[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        """Enhanced conversation augmentation with comprehensive strategies"""
        augmented_conversations = []
        num_augmentations = random.randint(1, self.config.max_augmentations)
        
        for i in range(num_augmentations):
            augmented_conv = []
            demographics = target_demographics[i] if target_demographics and i < len(target_demographics) else {}
            
            for turn in conversation:
                augmented_turn = turn.copy()
                
                if turn["speaker"] == "patient":
                    augmented_turn["text"] = self.augment_patient_text_comprehensive(
                        augmented_turn["text"], demographics
                    )
                elif turn["speaker"] == "ai":
                    augmented_turn["text"] = self.augment_ai_response_comprehensive(
                        augmented_turn["text"]
                    )
                
                # Quality validation
                if self.config.enable_quality_checks:
                    quality_score = self.quality_validator.validate_semantic_similarity(
                        turn["text"], augmented_turn["text"]
                    )
                    if quality_score >= self.config.semantic_similarity_threshold:
                        augmented_conv.append(augmented_turn)
                    else:
                        augmented_conv.append(turn)  # Keep original if quality is poor
                else:
                    augmented_conv.append(augmented_turn)
            
            augmented_conversations.append(augmented_conv)
            self.generated_texts.extend([turn["text"] for turn in augmented_conv])
        
        return augmented_conversations
    
    def augment_patient_text_comprehensive(self, text: str, demographics: Dict[str, Any] = None) -> str:
        """Comprehensive patient text augmentation"""
        
        # Apply demographic variations
        if demographics and self.config.demographic_diversity_probability:
            text = self.demographic_augmentor.vary_demographics(text, demographics)
        
        # Apply symptom variations
        if random.random() < self.config.symptom_variation_probability:
            text = self.terminology_replacer.replace_symptom_description(text)
        
        # Apply synonym replacement
        if random.random() < self.config.synonym_probability:
            text = self.augment_patient_text(text)
        
        # Apply back-translation
        if random.random() < self.config.back_translation_probability:
            text = self.back_translator.back_translate(text)
        
        # Apply masked language modeling
        if random.random() < self.config.masked_lm_probability:
            text = self.masked_lm.mask_and_predict(text)
        
        # Apply paraphrasing
        if random.random() < self.config.paraphrase_probability:
            text = self.paraphrase_generator.paraphrase_text(text)
        
        return text
    
    def augment_ai_response_comprehensive(self, text: str) -> str:
        """Comprehensive AI response augmentation"""
        
        # Apply style transfer
        if random.random() < self.config.style_transfer_probability:
            text = self.style_transfer.vary_tone(text)
        
        # Apply synonym replacement (conservative for medical terms)
        if random.random() < self.config.synonym_probability * 0.5:
            text = self.augment_ai_response(text)
        
        # Apply paraphrasing for questions
        if random.random() < self.config.paraphrase_probability * 0.6:
            if "?" in text:
                text = self.paraphrase_generator.vary_question(text)
        
        return text
    
    def augment_patient_text(self, text: str) -> str:
        """Augment patient speech patterns"""
        # Replace medical terminology
        for term in self.terminology_replacer.medical_synonyms:
            if self.config.preserve_medical_terms:
                text = self.terminology_replacer.replace_medical_term(
                    text, term, preserve_probability=0.7  # 70% chance to preserve
                )
            else:
                text = self.terminology_replacer.replace_medical_term(text, term, preserve_probability=0.3)
        
        # Replace symptom descriptions
        text = self.terminology_replacer.replace_symptom_description(text)
        
        return text
    
    def augment_ai_response(self, text: str) -> str:
        """Augment AI response patterns"""
        # Replace medical terminology with clinical terms
        for term, synonyms in self.terminology_replacer.medical_synonyms.items():
            if random.random() < 0.3:  # 30% chance
                clinical_term = synonyms[0] if synonyms else term
                text = re.sub(r'\b' + re.escape(term) + r'\b', clinical_term, text, flags=re.IGNORECASE)
        
        return text
    
    def generate_emergency_routine_balance(self, conversations: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        """Generate balanced emergency vs routine cases"""
        
        emergency_indicators = [
            "emergency", "urgent", "severe pain", "chest pain", "difficulty breathing",
            "heart attack", "stroke", "unconscious", "bleeding heavily"
        ]
        
        routine_indicators = [
            "routine", "check-up", "follow-up", "annual", "preventive",
            "minor symptoms", "general inquiry", "prescription refill"
        ]
        
        emergency_conversations = []
        routine_conversations = []
        
        for conv in conversations:
            full_text = " ".join(turn["text"].lower() for turn in conv)
            
            if any(indicator in full_text for indicator in emergency_indicators):
                emergency_conversations.append(conv)
            elif any(indicator in full_text for indicator in routine_indicators):
                routine_conversations.append(conv)
            else:
                # Classify based on content
                if "pain" in full_text and random.random() < 0.3:
                    emergency_conversations.append(conv)
                else:
                    routine_conversations.append(conv)
        
        # Balance the dataset
        target_emergency_count = int(len(conversations) * self.config.emergency_case_target_ratio)
        target_routine_count = int(len(conversations) * self.config.routine_case_target_ratio)
        
        balanced_conversations = []
        
        # Add emergency cases
        if emergency_conversations:
            emergency_to_add = min(len(emergency_conversations), target_emergency_count)
            balanced_conversations.extend(emergency_conversations[:emergency_to_add])
            
            # Augment to reach target if needed
            while len([c for c in balanced_conversations if c in emergency_conversations]) < target_emergency_count:
                if emergency_conversations:
                    base_conv = random.choice(emergency_conversations)
                    augmented = self.augment_conversation(base_conv)
                    balanced_conversations.extend(augmented)
        
        # Add routine cases
        if routine_conversations:
            routine_to_add = min(len(routine_conversations), target_routine_count)
            balanced_conversations.extend(routine_conversations[:routine_to_add])
            
            # Augment to reach target if needed
            while len([c for c in balanced_conversations if c in routine_conversations]) < target_routine_count:
                if routine_conversations:
                    base_conv = random.choice(routine_conversations)
                    augmented = self.augment_conversation(base_conv)
                    balanced_conversations.extend(augmented)
        
        self.logger.info(f"Balanced dataset: {len([c for c in balanced_conversations if c in emergency_conversations])} emergency, "
                        f"{len([c for c in balanced_conversations if c in routine_conversations])} routine")
        
        return balanced_conversations[:len(conversations)]  # Limit to original size
    
    def vary_conversation_flow(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Vary conversation flow patterns"""
        
        if random.random() > self.config.conversation_flow_probability:
            return conversation
        
        varied_conversation = conversation.copy()
        
        # Randomly reorder some exchanges while maintaining medical logic
        ai_turns = [(i, turn) for i, turn in enumerate(varied_conversation) if turn["speaker"] == "ai"]
        patient_turns = [(i, turn) for i, turn in enumerate(varied_conversation) if turn["speaker"] == "patient"]
        
        # Sometimes add clarifying questions earlier
        if len(ai_turns) > 1 and random.random() < 0.3:
            # Move a clarifying question from later to earlier
            later_questions = [turn for _, turn in ai_turns[1:] if "?" in turn["text"]]
            if later_questions:
                question = later_questions[0]
                # Find its position and move it earlier
                original_index = varied_conversation.index(question)
                new_index = min(original_index - 1, len(ai_turns) - 1)
                
                if new_index < original_index:
                    varied_conversation.remove(question)
                    varied_conversation.insert(new_index, question)
        
        # Sometimes add follow-up questions
        if random.random() < 0.4:
            follow_up_templates = [
                "Can you tell me more about that?",
                "How long has this been going on?",
                "On a scale of 1 to 10, how would you rate this?",
                "Are you currently taking any medications?"
            ]
            
            # Insert follow-up after patient response
            for i, turn in enumerate(varied_conversation[:-1]):
                if turn["speaker"] == "patient" and varied_conversation[i + 1]["speaker"] == "ai":
                    if random.random() < 0.2:  # 20% chance to add follow-up
                        follow_up = {
                            "speaker": "ai",
                            "text": random.choice(follow_up_templates),
                            "timestamp": turn.get("timestamp", 0) + 0.1
                        }
                        varied_conversation.insert(i + 1, follow_up)
                    break
        
        return varied_conversation
    
    def optimize_augmentation_strategy(self, 
                                     original_data: List[Dict[str, Any]], 
                                     validation_split: float = 0.2) -> Dict[str, Any]:
        """Optimize augmentation strategy based on performance"""
        
        strategies = [
            {"synonym_probability": 0.2, "paraphrase_probability": 0.3, "back_translation_probability": 0.2},
            {"synonym_probability": 0.4, "paraphrase_probability": 0.4, "back_translation_probability": 0.1},
            {"synonym_probability": 0.3, "paraphrase_probability": 0.2, "back_translation_probability": 0.3},
            {"synonym_probability": 0.5, "paraphrase_probability": 0.5, "back_translation_probability": 0.0}
        ]
        
        best_strategy = strategies[0]
        best_score = 0
        
        # Split data for validation
        split_point = int(len(original_data) * (1 - validation_split))
        train_data = original_data[:split_point]
        val_data = original_data[split_point:]
        
        for strategy in strategies:
            # Apply strategy
            config = AugmentationConfig()
            config.synonym_probability = strategy["synonym_probability"]
            config.paraphrase_probability = strategy["paraphrase_probability"]
            config.back_translation_probability = strategy["back_translation_probability"]
            
            augmentor = DataAugmentor(config)
            augmented_results = apply_augmentation_pipeline(train_data, augmentor.config)
            
            # Evaluate on validation set
            val_results = apply_augmentation_pipeline(val_data, augmentor.config)
            
            # Calculate strategy score
            strategy_score = self._evaluate_strategy_performance(augmented_results, val_results)
            
            if strategy_score > best_score:
                best_score = strategy_score
                best_strategy = strategy
        
        return {
            "best_strategy": best_strategy,
            "best_score": best_score,
            "all_scores": {str(i): self._evaluate_strategy_performance(results, val_data) 
                          for i, results in enumerate([apply_augmentation_pipeline(original_data, AugmentationConfig(**strategy)) for strategy in strategies])}
        }
    
    def _evaluate_strategy_performance(self, train_results: Dict, val_data: List[Dict]) -> float:
        """Evaluate strategy performance"""
        # Simplified performance metric
        quality_score = train_results.get("quality_metrics", {}).get("semantic_similarity", 0)
        diversity_score = train_results.get("quality_metrics", {}).get("diversity_score", 0)
        accuracy_score = train_results.get("quality_metrics", {}).get("medical_accuracy", 0)
        
        return (quality_score + diversity_score + accuracy_score) / 3

    def validate_augmentation_quality_comprehensive(self, 
                                                 original_texts: List[str], 
                                                 augmented_texts: List[str]) -> Dict[str, float]:
        """Comprehensive quality validation"""
        
        if not original_texts or not augmented_texts:
            return {"diversity_score": 0.0, "semantic_similarity": 0.0, "medical_accuracy": 0.0, 
                   "safety_score": 0.0, "coherence_score": 0.0}
        
        # Calculate individual metrics
        diversity_score = self._calculate_diversity_score(original_texts, augmented_texts)
        semantic_similarity = self._calculate_semantic_similarity(original_texts, augmented_texts)
        medical_accuracy = self._calculate_medical_accuracy(augmented_texts)
        
        # Safety and coherence validation
        safety_scores = []
        coherence_scores = []
        
        for text in augmented_texts:
            safety_valid, _ = self.quality_validator.validate_safety_constraints(text)
            coherence_score = self.quality_validator.validate_coherence(text)
            
            safety_scores.append(1.0 if safety_valid else 0.0)
            coherence_scores.append(coherence_score)
        
        safety_score = np.mean(safety_scores) if safety_scores else 0.0
        coherence_score = np.mean(coherence_scores) if coherence_scores else 0.0
        
        # Store scores for monitoring
        self.quality_scores.append({
            "diversity_score": diversity_score,
            "semantic_similarity": semantic_similarity,
            "medical_accuracy": medical_accuracy,
            "safety_score": safety_score,
            "coherence_score": coherence_score,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "diversity_score": diversity_score,
            "semantic_similarity": semantic_similarity,
            "medical_accuracy": medical_accuracy,
            "safety_score": safety_score,
            "coherence_score": coherence_score
        }
    
    def _calculate_diversity_score(self, original: List[str], augmented: List[str]) -> float:
        """Calculate diversity score"""
        if not original or not augmented:
            return 0.0
        
        original_vocab = set(' '.join(original).lower().split())
        augmented_vocab = set(' '.join(augmented).lower().split())
        new_words = augmented_vocab - original_vocab
        
        return len(new_words) / len(augmented_vocab) if augmented_vocab else 0
    
    def _calculate_semantic_similarity(self, original: List[str], augmented: List[str]) -> float:
        """Calculate semantic similarity"""
        if len(original) != len(augmented):
            return 0.0
        
        similarities = []
        for orig, aug in zip(original, augmented):
            similarity = self.quality_validator.validate_semantic_similarity(orig, aug)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0
    
    def _calculate_medical_accuracy(self, texts: List[str]) -> float:
        """Calculate medical accuracy"""
        if not texts:
            return 0.0
        
        accuracies = []
        for text in texts:
            accuracy = self.quality_validator.validate_medical_accuracy(text)
            accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0

    def save_quality_report(self, filepath: str):
        """Save quality monitoring report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_texts_generated": len(self.generated_texts),
            "quality_scores": self.quality_scores,
            "safety_violations": self.safety_violations,
            "average_scores": {
                metric: np.mean([score[metric] for score in self.quality_scores])
                for metric in ["diversity_score", "semantic_similarity", "medical_accuracy", 
                             "safety_score", "coherence_score"] if self.quality_scores
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def balance_dataset(self, 
                       scenarios: List[Dict[str, Any]], 
                       target_distribution: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Balance dataset to improve training"""
        
        # Analyze current distribution
        specialty_counts = Counter()
        triage_counts = Counter()
        
        for scenario in scenarios:
            specialty_counts[scenario.get("specialty", "unknown")] += 1
            triage_counts[scenario.get("triage_level", "unknown")] += 1
        
        print(f"Original specialty distribution: {dict(specialty_counts)}")
        print(f"Original triage distribution: {dict(triage_counts)}")
        
        # Apply balancing if target distribution specified
        if target_distribution:
            balanced_scenarios = []
            for scenario in scenarios:
                specialty = scenario.get("specialty", "unknown")
                if specialty in target_distribution:
                    # Add based on target probability
                    if random.random() < target_distribution[specialty]:
                        balanced_scenarios.append(scenario)
                else:
                    balanced_scenarios.append(scenario)
            
            scenarios = balanced_scenarios
        
        return scenarios
    
    def validate_augmentation_quality(self, 
                                    original_texts: List[str], 
                                    augmented_texts: List[str]) -> Dict[str, float]:
        """Validate quality of augmented data"""
        
        if not original_texts or not augmented_texts:
            return {"diversity_score": 0.0, "semantic_similarity": 0.0, "medical_accuracy": 0.0}
        
        # Calculate diversity score
        original_vocab = set(' '.join(original_texts).lower().split())
        augmented_vocab = set(' '.join(augmented_texts).lower().split())
        new_words = augmented_vocab - original_vocab
        
        diversity_score = len(new_words) / len(augmented_vocab) if augmented_vocab else 0
        
        # Calculate semantic similarity (simplified)
        def jaccard_similarity(text1, text2):
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0
        
        similarities = []
        for orig, aug in zip(original_texts, augmented_texts):
            similarities.append(jaccard_similarity(orig, aug))
        
        avg_similarity = np.mean(similarities) if similarities else 0
        
        # Medical accuracy check (basic)
        medical_terms = set()
        for synonyms in self.terminology_replacer.medical_synonyms.values():
            medical_terms.update(synonyms)
        medical_terms.update(self.terminology_replacer.medical_synonyms.keys())
        
        medical_accuracy_score = 0.9  # Simplified - assume high accuracy with proper term handling
        
        return {
            "diversity_score": diversity_score,
            "semantic_similarity": avg_similarity,
            "medical_accuracy": medical_accuracy_score
        }
    
    def generate_adversarial_dataset(self, 
                                   scenarios: List[Dict[str, Any]], 
                                   attack_rate: float = 0.1) -> List[Dict[str, Any]]:
        """Generate adversarial examples for robustness testing"""
        
        adversarial_scenarios = []
        
        for scenario in scenarios:
            if random.random() < attack_rate:
                # Create adversarial version
                adversarial_scenario = scenario.copy()
                
                # Attack conversation
                if "conversation" in adversarial_scenario:
                    adversarial_scenario["conversation"] = self._attack_conversation(
                        adversarial_scenario["conversation"]
                    )
                
                # Add metadata
                adversarial_scenario["is_adversarial"] = True
                adversarial_scenario["attack_type"] = "medical_error"
                
                adversarial_scenarios.append(adversarial_scenario)
        
        return adversarial_scenarios
    
    def _attack_conversation(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply adversarial attacks to conversation"""
        
        attacked_conv = []
        
        for turn in conversation:
            attacked_turn = turn.copy()
            
            if turn["speaker"] == "patient":
                # Introduce medical errors
                attacked_turn["text"] = self.adversarial_generator.introduce_medical_error(
                    attacked_turn["text"]
                )
                attacked_turn["text"] = self.adversarial_generator.add_confusing_terminology(
                    attacked_turn["text"]
                )
            
            attacked_conv.append(attacked_turn)
        
        return attacked_conv


def load_conversations_from_json(filepath: str) -> List[Dict[str, Any]]:
    """Load conversations from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    conversations = []
    for scenario in data.get("scenarios", []):
        if "conversation" in scenario:
            conversations.append(scenario["conversation"])
    
    return conversations


def save_augmented_conversations(conversations: List[List[Dict[str, Any]]], filepath: str):
    """Save augmented conversations to JSON file"""
    data = {
        "generated_at": datetime.now().isoformat(),
        "num_conversations": len(conversations),
        "conversations": conversations
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def apply_augmentation_pipeline(original_data: List[Dict[str, Any]], 
                              config: Optional[AugmentationConfig] = None,
                              augmentor: Optional[DataAugmentor] = None) -> Dict[str, Any]:
    """Apply complete augmentation pipeline with enhanced features"""
    
    if config is None:
        config = AugmentationConfig()
    
    if augmentor is None:
        augmentor = DataAugmentor(config)
    
    # Track results
    results = {
        "original_conversations": len(original_data),
        "augmented_conversations": [],
        "original_texts": [],
        "augmented_texts": [],
        "quality_metrics": {},
        "demographic_variations": 0,
        "emergency_routine_balance": {},
        "flow_variations": 0,
        "safety_validations": 0,
        "timestamp": datetime.now().isoformat()
    }
    
    # Generate demographic variations
    target_demographics = []
    if config.demographic_diversity:
        for _ in range(len(original_data)):
            demographic = {
                "age_group": random.choice(["child", "adult", "elderly"]),
                "gender": random.choice(["male", "female", "other"]),
                "cultural_style": random.choice(["direct", "indirect", "formal", "informal"])
            }
            target_demographics.append(demographic)
    
    # Process each conversation
    for scenario in original_data:
        if "conversation" in scenario:
            original_conversation = scenario["conversation"]
            
            # Apply conversation flow variations
            if config.conversation_flow_probability:
                varied_conversation = augmentor.vary_conversation_flow(original_conversation)
                if varied_conversation != original_conversation:
                    results["flow_variations"] += 1
            
            # Generate augmentations with demographics
            augmented_conversations = augmentor.augment_conversation(
                varied_conversation if config.conversation_flow_probability else original_conversation,
                target_demographics
            )
            
            results["augmented_conversations"].extend(augmented_conversations)
            results["original_texts"].extend([turn["text"] for turn in original_conversation])
            
            for aug_conv in augmented_conversations:
                results["augmented_texts"].extend([turn["text"] for turn in aug_conv])
    
    # Balance emergency vs routine cases
    if config.emergency_routine_balance_probability:
        balanced_conversations = augmentor.generate_emergency_routine_balance(
            results["augmented_conversations"]
        )
        results["augmented_conversations"] = balanced_conversations
        
        # Count emergency vs routine
        emergency_count = 0
        routine_count = 0
        for conv in balanced_conversations:
            full_text = " ".join(turn["text"].lower() for turn in conv)
            if any(indicator in full_text for indicator in ["emergency", "urgent", "severe"]):
                emergency_count += 1
            else:
                routine_count += 1
        
        results["emergency_routine_balance"] = {
            "emergency": emergency_count,
            "routine": routine_count,
            "ratio": emergency_count / (emergency_count + routine_count) if (emergency_count + routine_count) > 0 else 0
        }
    
    # Calculate comprehensive quality metrics
    results["quality_metrics"] = augmentor.validate_augmentation_quality_comprehensive(
        results["original_texts"], results["augmented_texts"]
    )
    
    # Validate safety constraints
    if config.enable_safety_validation:
        safety_valid_count = 0
        for text in results["augmented_texts"]:
            is_safe, violations = augmentor.quality_validator.validate_safety_constraints(text)
            if is_safe:
                safety_valid_count += 1
            else:
                augmentor.safety_violations.extend(violations)
        
        results["safety_validations"] = {
            "total_texts": len(results["augmented_texts"]),
            "safe_texts": safety_valid_count,
            "safety_rate": safety_valid_count / len(results["augmented_texts"]) if results["augmented_texts"] else 0
        }
    
    # Generate adversarial examples
    adversarial_scenarios = augmentor.generate_adversarial_dataset(original_data)
    results["adversarial_count"] = len(adversarial_scenarios)
    
    # Save quality report
    quality_report_path = f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    augmentor.save_quality_report(quality_report_path)
    results["quality_report_path"] = quality_report_path
    
    return results


if __name__ == "__main__":
    from datetime import datetime
    
    # Example usage
    config = AugmentationConfig(
        synonym_probability=0.4,
        paraphrase_probability=0.3,
        adversarial_probability=0.2,
        max_augmentations=3
    )
    
    print("Data Augmentation Module for Medical AI Training")
    print(f"Configuration: {config}")