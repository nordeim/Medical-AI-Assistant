"""
Data Quality Assessment Tools for Medical AI Training

This module provides comprehensive quality assessment capabilities for training data:
- Semantic similarity analysis
- Medical accuracy validation
- Safety constraint enforcement  
- Coherence maintenance checks
- Diversity metrics
- Statistical analysis
"""

import re
import json
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
import math
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .data_augmentation import QualityControlValidator
except ImportError:
    from data_augmentation import QualityControlValidator


@dataclass
class QualityMetrics:
    """Container for quality metrics"""
    # Semantic metrics
    semantic_similarity: float = 0.0
    semantic_consistency: float = 0.0
    semantic_coherence: float = 0.0
    
    # Medical accuracy metrics
    medical_accuracy: float = 0.0
    medical_term_usage: float = 0.0
    symptom_recognition: float = 0.0
    diagnostic_logic: float = 0.0
    
    # Safety metrics
    safety_score: float = 0.0
    inappropriate_content_rate: float = 0.0
    medical_advice_safety: float = 0.0
    phi_protection: float = 0.0
    
    # Coherence metrics
    conversation_coherence: float = 0.0
    logical_flow: float = 0.0
    contextual_relevance: float = 0.0
    
    # Diversity metrics
    vocabulary_diversity: float = 0.0
    syntactic_diversity: float = 0.0
    content_diversity: float = 0.0
    demographic_diversity: float = 0.0
    
    # Statistical metrics
    length_distribution: float = 0.0
    speaker_distribution: float = 0.0
    response_time_patterns: float = 0.0
    
    # Overall quality
    overall_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class SemanticAnalyzer:
    """Analyzes semantic properties of medical conversations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Medical terminology for context analysis
        self.medical_categories = {
            "cardiovascular": ["heart", "blood pressure", "pulse", "cardiac", "chest pain"],
            "respiratory": ["breathing", "lung", "airway", "cough", "shortness of breath"],
            "neurological": ["brain", "nerve", "headache", "dizziness", "confusion"],
            "gastrointestinal": ["stomach", "abdominal", "nausea", "vomiting", "diarrhea"],
            "musculoskeletal": ["joint", "muscle", "bone", "back pain", "arthritis"],
            "dermatological": ["skin", "rash", "itching", "lesion", "dermatitis"]
        }
        
        # Symptom keywords
        self.symptom_keywords = [
            "pain", "ache", "sore", "hurt", "discomfort", "tenderness",
            "fever", "temperature", "hot", "chills", "sweating",
            "cough", "sneeze", "runny nose", "congestion",
            "headache", "dizzy", "lightheaded", "vertigo",
            "nausea", "vomiting", "stomach upset", "indigestion",
            "fatigue", "tired", "exhausted", "weakness",
            "shortness of breath", "difficulty breathing", "wheezing"
        ]
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        
        # Word-level Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union)
        
        # Medical context similarity
        context_similarity = self._calculate_context_similarity(text1, text2)
        
        # Weighted combination
        return (jaccard + context_similarity) / 2
    
    def _calculate_context_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on medical context"""
        
        def get_medical_context(text):
            context_scores = {}
            for category, terms in self.medical_categories.items():
                score = sum(1 for term in terms if term in text.lower())
                context_scores[category] = score
            return context_scores
        
        context1 = get_medical_context(text1)
        context2 = get_medical_context(text2)
        
        # Calculate cosine similarity of context vectors
        all_categories = set(context1.keys()) | set(context2.keys())
        
        vec1 = [context1.get(cat, 0) for cat in all_categories]
        vec2 = [context2.get(cat, 0) for cat in all_categories]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def analyze_conversation_coherence(self, conversation: List[Dict[str, Any]]) -> float:
        """Analyze coherence within a conversation"""
        
        if len(conversation) < 2:
            return 1.0
        
        # Check speaker alternation
        speakers = [turn.get("speaker") for turn in conversation]
        alternation_score = self._calculate_alternation_score(speakers)
        
        # Check topic consistency
        topic_consistency_score = self._calculate_topic_consistency(conversation)
        
        # Check response appropriateness
        response_score = self._calculate_response_appropriateness(conversation)
        
        # Weighted combination
        return (alternation_score + topic_consistency_score + response_score) / 3
    
    def _calculate_alternation_score(self, speakers: List[str]) -> float:
        """Calculate score based on speaker alternation"""
        if len(speakers) <= 1:
            return 0.0
        
        alternations = 0
        for i in range(1, len(speakers)):
            if speakers[i] != speakers[i-1]:
                alternations += 1
        
        max_alternations = len(speakers) - 1
        return alternations / max_alternations if max_alternations > 0 else 0
    
    def _calculate_topic_consistency(self, conversation: List[Dict[str, Any]]) -> float:
        """Calculate topic consistency across conversation"""
        
        texts = [turn.get("text", "").lower() for turn in conversation]
        
        # Extract medical contexts for each turn
        contexts = []
        for text in texts:
            context = set()
            for category, terms in self.medical_categories.items():
                if any(term in text for term in terms):
                    context.add(category)
            contexts.append(context)
        
        # Calculate consistency
        if len(contexts) <= 1:
            return 1.0
        
        consistency_scores = []
        for i in range(1, len(contexts)):
            intersection = contexts[i] & contexts[i-1]
            union = contexts[i] | contexts[i-1]
            score = len(intersection) / len(union) if union else 0
            consistency_scores.append(score)
        
        return np.mean(consistency_scores)
    
    def _calculate_response_appropriateness(self, conversation: List[Dict[str, Any]]) -> float:
        """Calculate appropriateness of responses"""
        
        appropriate_responses = 0
        total_responses = 0
        
        for i in range(1, len(conversation)):
            current_turn = conversation[i]
            previous_turn = conversation[i-1]
            
            # Skip if current turn is not a response
            if current_turn.get("speaker") == previous_turn.get("speaker"):
                continue
            
            total_responses += 1
            
            # Check if response addresses previous message
            if self._addresses_previous_message(previous_turn, current_turn):
                appropriate_responses += 1
        
        return appropriate_responses / total_responses if total_responses > 0 else 0
    
    def _addresses_previous_message(self, previous: Dict, current: Dict) -> bool:
        """Check if current message addresses previous message"""
        
        prev_text = previous.get("text", "").lower()
        curr_text = current.get("text", "").lower()
        
        # Check for question-answer patterns
        prev_is_question = "?" in prev_text
        curr_is_answer = any(word in curr_text for word in ["yes", "no", "because", "since", "when", "where"])
        
        if prev_is_question and curr_is_answer:
            return True
        
        # Check for symptom-response patterns
        prev_has_symptom = any(symptom in prev_text for symptom in self.symptom_keywords)
        curr_has_medical_response = any(term in curr_text for terms in self.medical_categories.values() 
                                      for term in terms)
        
        if prev_has_symptom and curr_has_medical_response:
            return True
        
        return False


class MedicalAccuracyValidator:
    """Validates medical accuracy of conversations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Medical knowledge base
        self.medical_knowledge = {
            # Common symptom-diagnosis associations (simplified)
            "chest pain": {
                "possible_causes": ["muscle strain", "heartburn", "anxiety", "heart attack", "pneumonia"],
                "requires_urgent_care": True,
                "typical_questions": ["How long have you had this pain?", "Does it radiate to your arm?", "Any shortness of breath?"]
            },
            "shortness of breath": {
                "possible_causes": ["asthma", "heart failure", "pneumonia", "anxiety", "lung disease"],
                "requires_urgent_care": True,
                "typical_questions": ["When does this occur?", "Any chest pain?", "History of heart or lung disease?"]
            },
            "headache": {
                "possible_causes": ["tension", "migraine", "dehydration", "stress", "sinus infection"],
                "requires_urgent_care": False,
                "typical_questions": ["Where is the pain?", "How severe is it?", "Any nausea or sensitivity to light?"]
            }
        }
        
        # Medical terminology accuracy patterns
        self.accuracy_patterns = {
            "dosage_patterns": re.compile(r'\b\d+\s*(?:mg|mcg|g|ml|cc)\b', re.IGNORECASE),
            "vital_signs_patterns": re.compile(r'\b\d+\s*(?:°F|°C|mmHg|bpm|beats? per minute)\b', re.IGNORECASE),
            "medical_conditions": re.compile(r'\b(?:diabetes|hypertension|asthma|copd|heart disease)\b', re.IGNORECASE),
            "medications": re.compile(r'\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril)\b', re.IGNORECASE)
        }
        
        # Inappropriate medical advice patterns
        self.inappropriate_patterns = [
            r"\bstop taking.*medication.*immediately\b",
            r"\bignore.*doctor.*advice\b",
            r"\bself-medicate.*without.*supervision\b",
            r"\bthis is definitely.*cancer\b",
            r"\byou definitely have.*disease\b"
        ]
    
    def validate_medical_accuracy(self, conversation: List[Dict[str, Any]]) -> float:
        """Validate medical accuracy of conversation"""
        
        if not conversation:
            return 0.0
        
        accuracy_scores = []
        
        for turn in conversation:
            text = turn.get("text", "")
            score = self._validate_single_turn(text)
            accuracy_scores.append(score)
        
        # Check conversation-level accuracy
        conversation_score = self._validate_conversation_level(conversation)
        accuracy_scores.append(conversation_score)
        
        return np.mean(accuracy_scores)
    
    def _validate_single_turn(self, text: str) -> float:
        """Validate accuracy of single turn"""
        
        score = 0.0
        
        # Check for appropriate medical terminology
        terminology_score = self._check_medical_terminology(text)
        score += terminology_score * 0.3
        
        # Check for inappropriate advice
        inappropriate_penalty = self._check_inappropriate_advice(text)
        score -= inappropriate_penalty * 0.4
        
        # Check for medical context consistency
        context_score = self._check_medical_context(text)
        score += context_score * 0.3
        
        return max(0.0, min(1.0, score))
    
    def _check_medical_terminology(self, text: str) -> float:
        """Check usage of appropriate medical terminology"""
        
        medical_terms_found = 0
        total_possible_terms = 0
        
        for category, pattern in self.accuracy_patterns.items():
            matches = pattern.findall(text)
            medical_terms_found += len(matches)
            total_possible_terms += 1  # Each category is a potential match
        
        # Bonus for specific medical details
        if self.accuracy_patterns["dosage_patterns"].search(text):
            medical_terms_found += 0.5
        
        if self.accuracy_patterns["vital_signs_patterns"].search(text):
            medical_terms_found += 0.5
        
        return min(medical_terms_found / max(total_possible_terms * 2, 1), 1.0)
    
    def _check_inappropriate_advice(self, text: str) -> float:
        """Check for inappropriate medical advice"""
        
        penalty = 0.0
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text.lower()):
                penalty += 1.0
        
        return min(penalty, 1.0)
    
    def _check_medical_context(self, text: str) -> float:
        """Check medical context appropriateness"""
        
        # Check for question patterns typical in medical consultations
        question_words = ["what", "when", "where", "how", "why", "which", "who"]
        has_questions = any(word in text.lower() for word in question_words)
        
        # Check for empathy and professional language
        empathetic_terms = ["understand", "sorry", "help", "concern", "care"]
        has_empathy = any(term in text.lower() for term in empathetic_terms)
        
        # Check for medical follow-up patterns
        follow_up_patterns = ["tell me more", "can you describe", "when did this start"]
        has_follow_up = any(pattern in text.lower() for pattern in follow_up_patterns)
        
        score = 0.0
        if has_questions:
            score += 0.3
        if has_empathy:
            score += 0.3
        if has_follow_up:
            score += 0.4
        
        return score
    
    def _validate_conversation_level(self, conversation: List[Dict[str, Any]]) -> float:
        """Validate conversation-level medical accuracy"""
        
        if len(conversation) < 2:
            return 1.0
        
        # Check for appropriate medical questioning sequence
        questioning_sequence_score = self._check_questioning_sequence(conversation)
        
        # Check for logical medical progression
        logical_progression_score = self._check_logical_progression(conversation)
        
        return (questioning_sequence_score + logical_progression_score) / 2
    
    def _check_questioning_sequence(self, conversation: List[Dict[str, Any]]) -> float:
        """Check if medical questions follow logical sequence"""
        
        questions = []
        for turn in conversation:
            if turn.get("speaker") == "ai" and "?" in turn.get("text", ""):
                questions.append(turn.get("text", ""))
        
        if len(questions) <= 1:
            return 1.0
        
        # Simple scoring based on question diversity
        question_types = set()
        for question in questions:
            if "pain" in question.lower():
                question_types.add("pain_inquiry")
            if "when" in question.lower():
                question_types.add("timing_inquiry")
            if "how" in question.lower():
                question_types.add("severity_inquiry")
        
        return min(len(question_types) / 3, 1.0)  # Expect at least 3 different question types
    
    def _check_logical_progression(self, conversation: List[Dict[str, Any]]) -> float:
        """Check logical progression of medical consultation"""
        
        # Look for progression from general to specific questions
        general_keywords = ["what", "problem", "issue", "concern"]
        specific_keywords = ["pain", "fever", "cough", "duration", "severity"]
        
        general_questions = 0
        specific_questions = 0
        
        for turn in conversation:
            if turn.get("speaker") == "ai" and "?" in turn.get("text", ""):
                text = turn.get("text", "").lower()
                if any(keyword in text for keyword in general_keywords):
                    general_questions += 1
                if any(keyword in text for keyword in specific_keywords):
                    specific_questions += 1
        
        if general_questions == 0:
            return 0.5  # No general questions found
        
        # Good progression should have more specific questions
        progression_score = min(specific_questions / max(general_questions, 1), 2.0)
        return min(progression_score / 2.0, 1.0)  # Normalize to 0-1


class SafetyValidator:
    """Validates safety constraints and PHI protection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # PHI patterns
        self.phi_patterns = {
            "names": re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),
            "phone_numbers": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            "emails": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "addresses": re.compile(r'\b\d+\s+[A-Za-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b', re.IGNORECASE),
            "ssn": re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            "dates": re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
        }
        
        # Safety violation patterns
        self.safety_violations = [
            r"\b(suicide|kill myself|end my life|harm myself)\b",
            r"\b(overdose|poison)\b.*\b(medication|drug|pill)\b",
            r"\b(ignore|disregard)\b.*\b(doctor|physician|medical)\b.*\b(advice|instruction)\b",
            r"\b(self-medicate|treat myself)\b.*\b(without|without)\b.*\b(doctor|medical supervision)\b",
            r"\bdangerous|life-threatening|emergency\b.*\b(no|no need to|no need for)\b.*\b(medical attention|doctor|emergency room)\b"
        ]
        
        # Medical advice appropriateness patterns
        self.inappropriate_advice = [
            r"\bstop.*medication.*immediately\b",
            r"\bdon't.*see.*doctor\b",
            r"\bthis.*is.*not.*serious\b",
            r"\byou.*definitely.*have.*\b(cancer|heart attack|stroke)\b",
            r"\btake.*(any|every).*medication.*without.*checking\b"
        ]
    
    def validate_safety(self, conversation: List[Dict[str, Any]]) -> Tuple[float, List[str]]:
        """Validate safety of conversation"""
        
        all_text = " ".join(turn.get("text", "") for turn in conversation)
        violations = []
        
        # Check for PHI
        phi_score = self._check_phi_protection(all_text)
        
        # Check for safety violations
        safety_score, safety_violations = self._check_safety_violations(all_text)
        
        # Check for inappropriate medical advice
        advice_score, advice_violations = self._check_inappropriate_advice(all_text)
        
        violations.extend(safety_violations)
        violations.extend(advice_violations)
        
        # Overall safety score (weighted average)
        overall_score = (phi_score * 0.4 + safety_score * 0.4 + advice_score * 0.2)
        
        return overall_score, violations
    
    def _check_phi_protection(self, text: str) -> float:
        """Check for PHI protection"""
        
        phi_found = 0
        total_checks = 0
        
        for phi_type, pattern in self.phi_patterns.items():
            matches = pattern.findall(text)
            if matches:
                phi_found += len(matches)
            total_checks += 1
        
        # Lower score indicates more PHI (worse protection)
        phi_ratio = phi_found / max(total_checks, 1)
        return max(0.0, 1.0 - phi_ratio)
    
    def _check_safety_violations(self, text: str) -> Tuple[float, List[str]]:
        """Check for safety violations"""
        
        violations = []
        violation_count = 0
        
        for pattern in self.safety_violations:
            matches = re.findall(pattern, text.lower())
            if matches:
                violation_count += len(matches)
                violations.append(f"Safety violation: {pattern}")
        
        safety_score = max(0.0, 1.0 - (violation_count * 0.2))  # Each violation reduces score by 20%
        return safety_score, violations
    
    def _check_inappropriate_advice(self, text: str) -> Tuple[float, List[str]]:
        """Check for inappropriate medical advice"""
        
        violations = []
        violation_count = 0
        
        for pattern in self.inappropriate_advice:
            matches = re.findall(pattern, text.lower())
            if matches:
                violation_count += len(matches)
                violations.append(f"Inappropriate advice: {pattern}")
        
        advice_score = max(0.0, 1.0 - (violation_count * 0.3))  # Advice violations are more serious
        return advice_score, violations


class DiversityAnalyzer:
    """Analyzes diversity in medical conversations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Demographic categories for diversity analysis
        self.demographic_categories = {
            "age_groups": ["child", "adult", "elderly"],
            "genders": ["male", "female", "other"],
            "cultural_backgrounds": ["direct", "indirect", "formal", "informal"],
            "medical_conditions": ["chronic", "acute", "preventive"]
        }
    
    def analyze_vocabulary_diversity(self, conversations: List[List[Dict[str, Any]]]) -> float:
        """Analyze vocabulary diversity across conversations"""
        
        all_words = []
        conversation_vocabularies = []
        
        for conversation in conversations:
            vocab = set()
            for turn in conversation:
                words = turn.get("text", "").lower().split()
                vocab.update(words)
                all_words.extend(words)
            conversation_vocabularies.append(vocab)
        
        if not all_words:
            return 0.0
        
        # Calculate Type-Token Ratio (TTR)
        unique_words = set(all_words)
        ttr = len(unique_words) / len(all_words)
        
        # Calculate vocabulary size per conversation
        vocab_sizes = [len(vocab) for vocab in conversation_vocabularies]
        avg_vocab_size = np.mean(vocab_sizes)
        
        # Calculate vocabulary growth rate
        sorted_vocabs = sorted(vocab_sizes)
        vocab_growth = (sorted_vocabs[-1] - sorted_vocabs[0]) / max(sorted_vocabs[0], 1)
        
        # Combined diversity score
        diversity_score = (ttr + (avg_vocab_size / 1000) + vocab_growth) / 3
        return min(diversity_score, 1.0)
    
    def analyze_syntactic_diversity(self, conversations: List[List[Dict[str, Any]]]) -> float:
        """Analyze syntactic diversity"""
        
        sentence_structures = []
        
        for conversation in conversations:
            for turn in conversation:
                text = turn.get("text", "")
                
                # Analyze sentence patterns
                sentences = re.split(r'[.!?]+', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                for sentence in sentences:
                    # Analyze complexity patterns
                    word_count = len(sentence.split())
                    has_question = "?" in sentence
                    has_conditional = any(word in sentence.lower() for word in ["if", "when", "because", "since"])
                    has_passive = "is" in sentence.lower() and "was" in sentence.lower()
                    
                    structure_features = {
                        "word_count": word_count,
                        "is_question": has_question,
                        "is_conditional": has_conditional,
                        "is_passive": has_passive
                    }
                    
                    sentence_structures.append(structure_features)
        
        if not sentence_structures:
            return 0.0
        
        # Analyze diversity in sentence structures
        question_ratio = sum(1 for s in sentence_structures if s["is_question"]) / len(sentence_structures)
        conditional_ratio = sum(1 for s in sentence_structures if s["is_conditional"]) / len(sentence_structures)
        passive_ratio = sum(1 for s in sentence_structures if s["is_passive"]) / len(sentence_structures)
        
        # Calculate complexity distribution
        complexity_scores = []
        for s in sentence_structures:
            complexity = 0
            if s["word_count"] > 10:
                complexity += 0.3
            if s["is_conditional"]:
                complexity += 0.3
            if s["is_passive"]:
                complexity += 0.2
            if s["is_question"]:
                complexity += 0.2
            complexity_scores.append(complexity)
        
        complexity_variance = np.var(complexity_scores)
        
        # Combined syntactic diversity score
        diversity_score = (question_ratio + conditional_ratio + passive_ratio + min(complexity_variance, 0.5)) / 4
        return min(diversity_score, 1.0)
    
    def analyze_content_diversity(self, conversations: List[List[Dict[str, Any]]]) -> float:
        """Analyze content diversity"""
        
        # Categorize conversations by medical topics
        topic_categories = {
            "cardiovascular": 0,
            "respiratory": 0,
            "neurological": 0,
            "gastrointestinal": 0,
            "musculoskeletal": 0,
            "dermatological": 0,
            "general": 0
        }
        
        medical_keywords = {
            "cardiovascular": ["chest pain", "heart", "blood pressure", "palpitations"],
            "respiratory": ["cough", "breathing", "shortness of breath", "wheezing"],
            "neurological": ["headache", "dizziness", "memory", "confusion"],
            "gastrointestinal": ["stomach pain", "nausea", "vomiting", "diarrhea"],
            "musculoskeletal": ["back pain", "joint pain", "muscle", "stiffness"],
            "dermatological": ["rash", "skin", "itching", "lesion"]
        }
        
        for conversation in conversations:
            conversation_text = " ".join(turn.get("text", "") for turn in conversation).lower()
            categorized = False
            
            for category, keywords in medical_keywords.items():
                if any(keyword in conversation_text for keyword in keywords):
                    topic_categories[category] += 1
                    categorized = True
                    break
            
            if not categorized:
                topic_categories["general"] += 1
        
        # Calculate category distribution entropy
        total_conversations = sum(topic_categories.values())
        if total_conversations == 0:
            return 0.0
        
        entropies = []
        for count in topic_categories.values():
            if count > 0:
                probability = count / total_conversations
                entropy = -probability * np.log2(probability)
                entropies.append(entropy)
        
        max_entropy = np.log2(len(topic_categories))
        diversity_score = sum(entropies) / max_entropy if max_entropy > 0 else 0
        
        return diversity_score
    
    def analyze_demographic_diversity(self, conversations: List[List[Dict[str, Any]]]) -> float:
        """Analyze demographic diversity indicators"""
        
        # Analyze linguistic patterns that might indicate demographic diversity
        formality_indicators = {
            "formal": ["doctor", "physician", "patient", "medical", "consultation"],
            "informal": ["hey", "gonna", "wanna", "yeah", "okay"],
            "direct": ["I have", "my", "I'm experiencing"],
            "indirect": ["it seems", "might be", "could be", "perhaps"]
        }
        
        diversity_scores = []
        
        for conversation in conversations:
            conversation_text = " ".join(turn.get("text", "") for turn in conversation).lower()
            
            formality_scores = {}
            for style, indicators in formality_indicators.items():
                score = sum(1 for indicator in indicators if indicator in conversation_text)
                formality_scores[style] = score
            
            # Calculate diversity within this conversation
            max_score = max(formality_scores.values()) if formality_scores.values() else 0
            if max_score > 0:
                normalized_scores = [score / max_score for score in formality_scores.values()]
                conversation_diversity = np.std(normalized_scores)
                diversity_scores.append(conversation_diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0


class DataQualityAssessment:
    """Main data quality assessment orchestrator"""
    
    def __init__(self):
        self.semantic_analyzer = SemanticAnalyzer()
        self.medical_validator = MedicalAccuracyValidator()
        self.safety_validator = SafetyValidator()
        self.diversity_analyzer = DiversityAnalyzer()
        self.quality_validator = QualityControlValidator(AugmentationConfig())
        
        self.logger = logging.getLogger(__name__)
    
    def assess_data_quality(self, conversations: List[List[Dict[str, Any]]]) -> QualityMetrics:
        """Comprehensive quality assessment of medical conversation data"""
        
        self.logger.info(f"Starting quality assessment of {len(conversations)} conversations...")
        
        if not conversations:
            return QualityMetrics()
        
        metrics = QualityMetrics()
        
        # Semantic analysis
        semantic_scores = self._assess_semantic_quality(conversations)
        metrics.semantic_similarity = semantic_scores["similarity"]
        metrics.semantic_consistency = semantic_scores["consistency"]
        metrics.semantic_coherence = semantic_scores["coherence"]
        
        # Medical accuracy
        medical_scores = self._assess_medical_accuracy(conversations)
        metrics.medical_accuracy = medical_scores["overall"]
        metrics.medical_term_usage = medical_scores["terminology"]
        metrics.symptom_recognition = medical_scores["symptoms"]
        metrics.diagnostic_logic = medical_scores["logic"]
        
        # Safety validation
        safety_scores = self._assess_safety(conversations)
        metrics.safety_score = safety_scores["overall"]
        metrics.inappropriate_content_rate = 1.0 - safety_scores["content_safety"]
        metrics.medical_advice_safety = safety_scores["advice_safety"]
        metrics.phi_protection = safety_scores["phi"]
        
        # Coherence analysis
        coherence_scores = self._assess_coherence(conversations)
        metrics.conversation_coherence = coherence_scores["conversation"]
        metrics.logical_flow = coherence_scores["flow"]
        metrics.contextual_relevance = coherence_scores["relevance"]
        
        # Diversity analysis
        diversity_scores = self._assess_diversity(conversations)
        metrics.vocabulary_diversity = diversity_scores["vocabulary"]
        metrics.syntactic_diversity = diversity_scores["syntactic"]
        metrics.content_diversity = diversity_scores["content"]
        metrics.demographic_diversity = diversity_scores["demographic"]
        
        # Statistical analysis
        stats_scores = self._assess_statistics(conversations)
        metrics.length_distribution = stats_scores["length"]
        metrics.speaker_distribution = stats_scores["speaker"]
        metrics.response_time_patterns = stats_scores["timing"]
        
        # Calculate overall quality score
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        self.logger.info(f"Quality assessment complete. Overall score: {metrics.overall_score:.3f}")
        
        return metrics
    
    def _assess_semantic_quality(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess semantic quality metrics"""
        
        similarity_scores = []
        consistency_scores = []
        coherence_scores = []
        
        # Sample conversations for efficiency
        sample_size = min(len(conversations), 100)
        sampled_conversations = np.random.choice(conversations, sample_size, replace=False)
        
        for conversation in sampled_conversations:
            if len(conversation) >= 2:
                # Calculate similarity between adjacent turns
                for i in range(len(conversation) - 1):
                    text1 = conversation[i].get("text", "")
                    text2 = conversation[i + 1].get("text", "")
                    similarity = self.semantic_analyzer.calculate_semantic_similarity(text1, text2)
                    similarity_scores.append(similarity)
                
                # Calculate conversation coherence
                coherence = self.semantic_analyzer.analyze_conversation_coherence(conversation)
                coherence_scores.append(coherence)
        
        return {
            "similarity": np.mean(similarity_scores) if similarity_scores else 0.0,
            "consistency": np.mean(similarity_scores) if similarity_scores else 0.0,  # Simplified
            "coherence": np.mean(coherence_scores) if coherence_scores else 0.0
        }
    
    def _assess_medical_accuracy(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess medical accuracy metrics"""
        
        accuracy_scores = []
        terminology_scores = []
        symptom_scores = []
        logic_scores = []
        
        # Sample conversations
        sample_size = min(len(conversations), 50)
        sampled_conversations = np.random.choice(conversations, sample_size, replace=False)
        
        for conversation in sampled_conversations:
            accuracy = self.medical_validator.validate_medical_accuracy(conversation)
            accuracy_scores.append(accuracy)
            
            # Individual turn analysis
            terminology_score = 0
            symptom_score = 0
            logic_score = 0
            
            for turn in conversation:
                text = turn.get("text", "")
                
                # Check terminology
                terminology = self.medical_validator._check_medical_terminology(text)
                terminology_score += terminology
                
                # Check symptoms
                symptom_indicators = len(re.findall(r'\b(?:pain|fever|cough|headache|nausea)\b', text.lower()))
                symptom_score += min(symptom_indicators / 5, 1.0)
                
                # Check logic (simplified)
                logic_indicators = len(re.findall(r'\b(?:because|since|therefore|thus)\b', text.lower()))
                logic_score += min(logic_indicators / 2, 1.0)
            
            if conversation:
                terminology_scores.append(terminology_score / len(conversation))
                symptom_scores.append(symptom_score / len(conversation))
                logic_scores.append(logic_score / len(conversation))
        
        return {
            "overall": np.mean(accuracy_scores) if accuracy_scores else 0.0,
            "terminology": np.mean(terminology_scores) if terminology_scores else 0.0,
            "symptoms": np.mean(symptom_scores) if symptom_scores else 0.0,
            "logic": np.mean(logic_scores) if logic_scores else 0.0
        }
    
    def _assess_safety(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess safety metrics"""
        
        overall_scores = []
        content_safety_scores = []
        advice_safety_scores = []
        phi_scores = []
        
        # Sample conversations
        sample_size = min(len(conversations), 50)
        sampled_conversations = np.random.choice(conversations, sample_size, replace=False)
        
        for conversation in sampled_conversations:
            overall, violations = self.safety_validator.validate_safety(conversation)
            overall_scores.append(overall)
            
            # Individual checks
            content_safety = 1.0 - (len([v for v in violations if "Safety violation" in v]) / 10)
            advice_safety = 1.0 - (len([v for v in violations if "Inappropriate advice" in v]) / 10)
            phi_score = self.quality_validator.validate_medical_accuracy(
                " ".join(turn.get("text", "") for turn in conversation)
            )
            
            content_safety_scores.append(max(content_safety, 0.0))
            advice_safety_scores.append(max(advice_safety, 0.0))
            phi_scores.append(max(phi_score, 0.0))
        
        return {
            "overall": np.mean(overall_scores) if overall_scores else 0.0,
            "content_safety": np.mean(content_safety_scores) if content_safety_scores else 1.0,
            "advice_safety": np.mean(advice_safety_scores) if advice_safety_scores else 1.0,
            "phi": np.mean(phi_scores) if phi_scores else 1.0
        }
    
    def _assess_coherence(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess coherence metrics"""
        
        conversation_coherence = []
        logical_flow = []
        contextual_relevance = []
        
        # Sample conversations
        sample_size = min(len(conversations), 100)
        sampled_conversations = np.random.choice(conversations, sample_size, replace=False)
        
        for conversation in sampled_conversations:
            coherence = self.semantic_analyzer.analyze_conversation_coherence(conversation)
            conversation_coherence.append(coherence)
            
            # Simplified flow and relevance scores
            flow_score = min(len(conversation) / 10, 1.0)  # More turns = better flow
            relevance_score = 0.8 if conversation else 0.0  # Basic relevance
            
            logical_flow.append(flow_score)
            contextual_relevance.append(relevance_score)
        
        return {
            "conversation": np.mean(conversation_coherence) if conversation_coherence else 0.0,
            "flow": np.mean(logical_flow) if logical_flow else 0.0,
            "relevance": np.mean(contextual_relevance) if contextual_relevance else 0.0
        }
    
    def _assess_diversity(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess diversity metrics"""
        
        vocab_diversity = self.diversity_analyzer.analyze_vocabulary_diversity(conversations)
        syntactic_diversity = self.diversity_analyzer.analyze_syntactic_diversity(conversations)
        content_diversity = self.diversity_analyzer.analyze_content_diversity(conversations)
        demographic_diversity = self.diversity_analyzer.analyze_demographic_diversity(conversations)
        
        return {
            "vocabulary": vocab_diversity,
            "syntactic": syntactic_diversity,
            "content": content_diversity,
            "demographic": demographic_diversity
        }
    
    def _assess_statistics(self, conversations: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Assess statistical distribution metrics"""
        
        # Length distribution
        lengths = []
        for conversation in conversations:
            for turn in conversation:
                text = turn.get("text", "")
                lengths.append(len(text))
        
        length_std = np.std(lengths) if lengths else 0
        length_cv = length_std / np.mean(lengths) if lengths and np.mean(lengths) > 0 else 0
        length_score = min(length_cv, 1.0)  # Higher coefficient of variation = more diverse
        
        # Speaker distribution
        speaker_counts = Counter()
        for conversation in conversations:
            for turn in conversation:
                speaker = turn.get("speaker", "unknown")
                speaker_counts[speaker] += 1
        
        if len(speaker_counts) > 1:
            total_turns = sum(speaker_counts.values())
            distribution_entropy = 0
            for count in speaker_counts.values():
                p = count / total_turns
                distribution_entropy -= p * np.log2(p)
            max_entropy = np.log2(len(speaker_counts))
            speaker_score = distribution_entropy / max_entropy if max_entropy > 0 else 0
        else:
            speaker_score = 0.0
        
        # Response timing (simplified - based on turn order)
        timing_score = 0.8  # Placeholder for now
        
        return {
            "length": length_score,
            "speaker": speaker_score,
            "timing": timing_score
        }
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score"""
        
        weights = {
            "semantic_similarity": 0.15,
            "semantic_consistency": 0.10,
            "semantic_coherence": 0.10,
            "medical_accuracy": 0.20,
            "medical_term_usage": 0.05,
            "symptom_recognition": 0.05,
            "diagnostic_logic": 0.05,
            "safety_score": 0.15,
            "conversation_coherence": 0.10,
            "vocabulary_diversity": 0.05
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            if hasattr(metrics, metric_name):
                value = getattr(metrics, metric_name)
                total_score += value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def generate_quality_report(self, metrics: QualityMetrics, output_file: str = None) -> Dict[str, Any]:
        """Generate detailed quality report"""
        
        report = {
            "assessment_timestamp": datetime.now().isoformat(),
            "overall_quality_score": metrics.overall_score,
            "detailed_metrics": {
                "semantic_quality": {
                    "similarity": metrics.semantic_similarity,
                    "consistency": metrics.semantic_consistency,
                    "coherence": metrics.semantic_coherence
                },
                "medical_accuracy": {
                    "overall": metrics.medical_accuracy,
                    "terminology_usage": metrics.medical_term_usage,
                    "symptom_recognition": metrics.symptom_recognition,
                    "diagnostic_logic": metrics.diagnostic_logic
                },
                "safety": {
                    "overall_score": metrics.safety_score,
                    "inappropriate_content_rate": metrics.inappropriate_content_rate,
                    "medical_advice_safety": metrics.medical_advice_safety,
                    "phi_protection": metrics.phi_protection
                },
                "coherence": {
                    "conversation_coherence": metrics.conversation_coherence,
                    "logical_flow": metrics.logical_flow,
                    "contextual_relevance": metrics.contextual_relevance
                },
                "diversity": {
                    "vocabulary": metrics.vocabulary_diversity,
                    "syntactic": metrics.syntactic_diversity,
                    "content": metrics.content_diversity,
                    "demographic": metrics.demographic_diversity
                },
                "statistics": {
                    "length_distribution": metrics.length_distribution,
                    "speaker_distribution": metrics.speaker_distribution,
                    "response_time_patterns": metrics.response_time_patterns
                }
            },
            "quality_assessment": self._interpret_quality_score(metrics.overall_score),
            "recommendations": self._generate_recommendations(metrics)
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _interpret_quality_score(self, score: float) -> str:
        """Interpret quality score with descriptive labels"""
        
        if score >= 0.9:
            return "Excellent - Data meets highest quality standards"
        elif score >= 0.8:
            return "Good - Data is of high quality with minor areas for improvement"
        elif score >= 0.7:
            return "Satisfactory - Data is acceptable but could benefit from refinement"
        elif score >= 0.6:
            return "Fair - Data needs improvement in several areas"
        else:
            return "Poor - Data requires significant improvement before use"
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations based on metrics"""
        
        recommendations = []
        
        # Semantic quality recommendations
        if metrics.semantic_similarity < 0.7:
            recommendations.append("Improve semantic consistency between conversation turns")
        
        if metrics.semantic_coherence < 0.7:
            recommendations.append("Enhance conversation flow and logical progression")
        
        # Medical accuracy recommendations
        if metrics.medical_accuracy < 0.8:
            recommendations.append("Review and improve medical terminology accuracy")
        
        if metrics.symptom_recognition < 0.7:
            recommendations.append("Strengthen symptom recognition and description patterns")
        
        # Safety recommendations
        if metrics.safety_score < 0.9:
            recommendations.append("Implement stricter safety validation and PHI protection")
        
        if metrics.phi_protection < 0.95:
            recommendations.append("Enhance PHI detection and removal procedures")
        
        # Diversity recommendations
        if metrics.vocabulary_diversity < 0.6:
            recommendations.append("Increase vocabulary diversity through data augmentation")
        
        if metrics.content_diversity < 0.6:
            recommendations.append("Diversify medical topics and scenarios in the dataset")
        
        # Overall recommendations
        if metrics.overall_score < 0.7:
            recommendations.append("Consider comprehensive data quality improvement process")
        
        if not recommendations:
            recommendations.append("Data quality is good. Continue monitoring and minor optimizations.")
        
        return recommendations


def assess_medical_conversation_quality(data_file: str, 
                                      output_file: str = None) -> Dict[str, Any]:
    """Convenience function to assess quality of medical conversation data"""
    
    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract conversations
    conversations = []
    if isinstance(data, dict):
        if "conversations" in data:
            conversations = data["conversations"]
        elif "scenarios" in data:
            for scenario in data["scenarios"]:
                if "conversation" in scenario:
                    conversations.append(scenario["conversation"])
    
    # Assess quality
    assessor = DataQualityAssessment()
    metrics = assessor.assess_data_quality(conversations)
    
    # Generate report
    report = assessor.generate_quality_report(metrics, output_file)
    
    return {
        "quality_metrics": metrics,
        "quality_report": report,
        "data_summary": {
            "total_conversations": len(conversations),
            "assessment_timestamp": datetime.now().isoformat()
        }
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Data Quality Assessment Tools for Medical AI Training")
    
    # Create sample data for testing
    sample_conversations = [
        [
            {"speaker": "patient", "text": "I have chest pain since yesterday"},
            {"speaker": "ai", "text": "I understand you're experiencing chest pain. Can you describe the pain - is it sharp or dull?"},
            {"speaker": "patient", "text": "It's a sharp pain that gets worse when I breathe"}
        ]
    ]
    
    # Assess quality
    assessor = DataQualityAssessment()
    metrics = assessor.assess_data_quality(sample_conversations)
    
    print(f"Quality assessment complete. Overall score: {metrics.overall_score:.3f}")