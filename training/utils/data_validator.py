"""Comprehensive data validation and quality assurance utilities for medical training data."""

import re
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import statistics
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    
    # Data integrity checks
    required_fields: List[str] = field(default_factory=lambda: [
        'conversation_id', 'user_input', 'assistant_response', 'timestamp',
        'age', 'gender', 'triage_level', 'symptoms'
    ])
    
    # Medical data specific
    valid_triage_levels: List[str] = field(default_factory=lambda: [
        'emergency', 'urgent', 'non-urgent', 'advisory'
    ])
    
    age_range: Tuple[int, int] = (0, 150)
    
    # PHI detection patterns
    phi_patterns: List[str] = field(default_factory=lambda: [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{10,11}\b',         # Phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr)\b',  # Address
        r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
    ])
    
    # Medical terms for validation
    medical_terms: List[str] = field(default_factory=lambda: [
        'pain', 'fever', 'nausea', 'vomiting', 'diarrhea', 'headache',
        'chest pain', 'shortness of breath', 'dizziness', 'fatigue',
        'cough', 'sore throat', 'congestion', 'rash', 'swelling',
        'bleeding', 'fracture', 'injury', 'infection', 'inflammation'
    ])
    
    # Text quality thresholds
    min_text_length: int = 10
    max_text_length: int = 5000
    min_readability_score: float = 5.0
    
    # Statistical thresholds
    outlier_std_multiplier: float = 3.0
    duplicate_similarity_threshold: float = 0.95
    max_missing_data_percentage: float = 20.0
    
    # Logging
    log_level: str = "INFO"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class DataValidator:
    """Comprehensive data validation and quality assurance."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize the data validator."""
        self.config = config or ValidationConfig()
        self.logger = self._setup_logging()
        self.phi_detector = re.compile('|'.join(self.config.phi_patterns))
        
        # Initialize statistics tracking
        self._text_cache = {}
        self._duplicate_hashes = set()
        self._similarity_matrix = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def validate_dataset(self, data: Union[List[Dict], pd.DataFrame]) -> ValidationResult:
        """Validate an entire dataset."""
        self.logger.info(f"Validating dataset with {len(data)} records...")
        
        result = ValidationResult(is_valid=True, metrics={})
        
        try:
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Run all validation checks
            self._validate_data_integrity(df, result)
            self._validate_medical_data(df, result)
            self._perform_statistical_validation(df, result)
            self._calculate_quality_metrics(df, result)
            
            # Calculate overall score
            result.score = self._calculate_overall_score(result)
            
            # Determine if validation passed
            result.is_valid = result.score >= 0.8 and len(result.errors) == 0
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Dataset validation failed: {str(e)}")
            self.logger.error(f"Dataset validation error: {e}")
        
        self.logger.info(f"Validation completed. Score: {result.score:.2f}, "
                        f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
        
        return result
    
    def _validate_data_integrity(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate data integrity."""
        self.logger.info("Performing data integrity checks...")
        
        # Check required fields
        missing_fields = []
        for field in self.config.required_fields:
            if field not in df.columns:
                missing_fields.append(field)
        
        if missing_fields:
            result.errors.append(f"Missing required fields: {missing_fields}")
        
        # Check data types and formats
        if 'age' in df.columns:
            invalid_ages = df[~df['age'].between(*self.config.age_range)]['age'].tolist()
            if invalid_ages:
                result.warnings.append(f"Found {len(invalid_ages)} invalid age values")
        
        # Check encoding and text quality
        text_fields = ['user_input', 'assistant_response', 'symptoms']
        for field in text_fields:
            if field in df.columns:
                self._validate_text_field(df, field, result)
        
        # Check for duplicates
        duplicates = self._detect_duplicates(df)
        if duplicates:
            result.warnings.append(f"Found {len(duplicates)} potential duplicate records")
            result.metrics['duplicates'] = duplicates
        
    def _validate_medical_data(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate medical data specific checks."""
        self.logger.info("Performing medical data validation...")
        
        # Validate triage levels
        if 'triage_level' in df.columns:
            invalid_triage = df[~df['triage_level'].isin(self.config.valid_triage_levels)]['triage_level'].tolist()
            if invalid_triage:
                result.errors.append(f"Found {len(invalid_triage)} invalid triage levels")
        
        # Check demographic data consistency
        if 'gender' in df.columns:
            valid_genders = ['male', 'female', 'other', 'M', 'F', 'm', 'f']
            invalid_gender = df[~df['gender'].str.lower().isin(valid_genders)]['gender'].tolist()
            if invalid_gender:
                result.warnings.append(f"Found {len(invalid_gender)} invalid gender values")
        
        # Validate symptom descriptions
        if 'symptoms' in df.columns:
            self._validate_symptoms(df, result)
        
        # Detect PHI patterns
        phi_detections = self._detect_phi(df)
        if phi_detections:
            result.warnings.append(f"Found {len(phi_detections)} potential PHI patterns")
            result.metrics['phi_detections'] = phi_detections
        
        # Check medical terminology consistency
        medical_consistency = self._check_medical_consistency(df)
        if medical_consistency:
            result.warnings.append(f"Medical consistency issues: {medical_consistency}")
    
    def _perform_statistical_validation(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Perform statistical validation."""
        self.logger.info("Performing statistical validation...")
        
        # Distribution analysis
        distributions = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            distributions[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        result.metrics['distributions'] = distributions
        
        # Outlier detection
        outliers = self._detect_outliers(df[numeric_columns])
        if outliers:
            result.metrics['outliers'] = outliers
            result.warnings.append(f"Found outliers in {len(outliers)} columns")
        
        # Class balance analysis
        if 'triage_level' in df.columns:
            balance_analysis = self._analyze_class_balance(df, 'triage_level')
            result.metrics['class_balance'] = balance_analysis
            if balance_analysis['imbalance_ratio'] > 3.0:
                result.warnings.append("Significant class imbalance detected")
        
        # Missing data patterns
        missing_patterns = self._analyze_missing_data(df)
        result.metrics['missing_patterns'] = missing_patterns
        
        total_missing = missing_patterns['total_missing_percentage']
        if total_missing > self.config.max_missing_data_percentage:
            result.warnings.append(f"High missing data percentage: {total_missing:.1f}%")
    
    def _calculate_quality_metrics(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Calculate quality metrics."""
        self.logger.info("Calculating quality metrics...")
        
        quality_metrics = {}
        
        # Text quality scores
        text_quality = self._calculate_text_quality(df)
        quality_metrics.update(text_quality)
        
        # Conversation coherence metrics
        coherence_metrics = self._calculate_coherence_metrics(df)
        quality_metrics['coherence'] = coherence_metrics
        
        # Medical accuracy indicators
        medical_accuracy = self._calculate_medical_accuracy(df)
        quality_metrics['medical_accuracy'] = medical_accuracy
        
        # User satisfaction proxies
        satisfaction_proxies = self._calculate_satisfaction_proxies(df)
        quality_metrics['satisfaction'] = satisfaction_proxies
        
        result.metrics['quality'] = quality_metrics
        
        # Overall quality assessment
        if quality_metrics['avg_text_quality'] < 0.7:
            result.warnings.append("Low text quality detected")
        
        if quality_metrics['avg_coherence'] < 0.6:
            result.warnings.append("Low conversation coherence detected")
    
    def _validate_text_field(self, df: pd.DataFrame, field: str, result: ValidationResult) -> None:
        """Validate a text field for quality and format."""
        field_data = df[field].dropna().astype(str)
        
        # Check length constraints
        too_short = field_data.str.len() < self.config.min_text_length
        too_long = field_data.str.len() > self.config.max_text_length
        
        if too_short.any():
            result.warnings.append(f"Field '{field}': {too_short.sum()} records too short")
        
        if too_long.any():
            result.warnings.append(f"Field '{field}': {too_long.sum()} records too long")
        
        # Check encoding issues
        encoding_issues = field_data.str.contains(r'[^\x00-\x7F]', na=False)
        if encoding_issues.any():
            result.warnings.append(f"Field '{field}': {encoding_issues.sum()} records with potential encoding issues")
        
        # Check for repetitive content
        repetitive = field_data.str.contains(r'(.)\1{4,}', na=False)
        if repetitive.any():
            result.warnings.append(f"Field '{field}': {repetitive.sum()} records with repetitive content")
    
    def _detect_duplicates(self, df: pd.DataFrame) -> List[Dict]:
        """Detect duplicate and near-duplicate records."""
        duplicates = []
        
        # Exact duplicates
        duplicates_df = df.duplicated(keep=False)
        if duplicates_df.any():
            exact_dups = df[duplicates_df].to_dict('records')
            duplicates.extend([{'type': 'exact', 'record': rec} for rec in exact_dups])
        
        # Near-duplicate detection using text similarity
        text_columns = ['user_input', 'assistant_response']
        existing_texts = []
        
        for idx, row in df.iterrows():
            combined_text = ' '.join([str(row.get(col, '')) for col in text_columns if col in df.columns])
            
            # Check against cached texts
            for cached_idx, cached_text in existing_texts:
                similarity = self._calculate_text_similarity(combined_text, cached_text)
                if similarity > self.config.duplicate_similarity_threshold:
                    duplicates.append({
                        'type': 'near_duplicate',
                        'similarity': similarity,
                        'record1_index': cached_idx,
                        'record2_index': idx
                    })
            
            existing_texts.append((idx, combined_text))
        
        return duplicates
    
    def _validate_symptoms(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Validate symptom descriptions."""
        if 'symptoms' not in df.columns:
            return
        
        symptoms = df['symptoms'].dropna().astype(str)
        invalid_symptoms = []
        
        for symptom in symptoms:
            # Check for medical terminology
            has_medical_term = any(term.lower() in symptom.lower() for term in self.config.medical_terms)
            
            # Check for completeness (should have descriptive content)
            if len(symptom.strip()) < 5:
                invalid_symptoms.append(f"Too short symptom description: '{symptom}'")
            
            # Check for excessive repetition
            if symptom.count(' ') < 2 and len(symptom) < 20:
                invalid_symptoms.append(f"Insufficiently descriptive symptom: '{symptom}'")
        
        if invalid_symptoms:
            result.warnings.append(f"Symptom validation issues: {len(invalid_symptoms)} found")
    
    def _detect_phi(self, df: pd.DataFrame) -> List[Dict]:
        """Detect potential PHI (Protected Health Information) patterns."""
        phi_detections = []
        
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            for idx, text in df[col].dropna().astype(str).items():
                matches = self.phi_detector.findall(text)
                if matches:
                    phi_detections.append({
                        'column': col,
                        'record_index': idx,
                        'pattern_matches': matches,
                        'context': text[:100]  # First 100 chars for context
                    })
        
        return phi_detections
    
    def _check_medical_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check for medical consistency issues."""
        issues = []
        
        # Check age-triage consistency
        if 'age' in df.columns and 'triage_level' in df.columns:
            pediatric_emergency = df[(df['age'] < 18) & (df['triage_level'] == 'non-urgent')]
            if len(pediatric_emergency) > len(df) * 0.1:  # More than 10% seems suspicious
                issues.append("Unusually high non-urgent pediatric cases")
        
        # Check symptom-triage consistency
        if 'symptoms' in df.columns and 'triage_level' in df.columns:
            serious_symptoms = ['chest pain', 'severe bleeding', 'difficulty breathing']
            non_urgent_serious = df[
                df['symptoms'].str.contains('|'.join(serious_symptoms), case=False, na=False) &
                (df['triage_level'] == 'non-urgent')
            ]
            if len(non_urgent_serious) > 0:
                issues.append("Serious symptoms marked as non-urgent")
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers using IQR method."""
        outliers = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.config.outlier_std_multiplier * IQR
                upper_bound = Q3 + self.config.outlier_std_multiplier * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers
    
    def _analyze_class_balance(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Analyze class balance in categorical data."""
        value_counts = df[column].value_counts()
        total = len(df)
        
        balance_metrics = {
            'distribution': value_counts.to_dict(),
            'total_classes': len(value_counts),
            'imbalance_ratio': value_counts.max() / value_counts.min() if value_counts.min() > 0 else float('inf'),
            'entropy': -sum((count/total) * np.log2(count/total) for count in value_counts if count > 0)
        }
        
        return balance_metrics
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_stats = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_stats[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_percentage
            }
        
        # Calculate overall missing data percentage
        total_cells = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        overall_percentage = (total_missing / total_cells) * 100
        
        missing_stats['total_missing_percentage'] = overall_percentage
        
        return missing_stats
    
    def _calculate_text_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate text quality metrics."""
        text_fields = ['user_input', 'assistant_response', 'symptoms']
        quality_scores = {}
        
        for field in text_fields:
            if field in df.columns:
                texts = df[field].dropna().astype(str).tolist()
                if texts:
                    scores = [self._calculate_text_score(text) for text in texts]
                    quality_scores[f'{field}_quality'] = np.mean(scores)
        
        # Overall quality
        avg_quality = np.mean([score for score in quality_scores.values() if not np.isnan(score)])
        quality_scores['avg_text_quality'] = avg_quality
        
        return quality_scores
    
    def _calculate_text_score(self, text: str) -> float:
        """Calculate quality score for a single text."""
        if not text or len(text.strip()) < 5:
            return 0.0
        
        score = 0.0
        
        # Length score (normalized)
        length = len(text)
        if 20 <= length <= 500:
            score += 0.3
        elif length > 0:
            score += 0.1
        
        # Readability score
        readability = self._calculate_readability(text)
        score += min(readability / 10.0, 0.3)  # Cap at 0.3
        
        # Structure score (sentences, proper case)
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            score += 0.2
        
        # Grammar/structure indicators
        has_proper_casing = any(word[0].isupper() for word in text.split() if word)
        if has_proper_casing:
            score += 0.1
        
        # Medical terminology usage
        medical_term_count = sum(1 for term in self.config.medical_terms if term.lower() in text.lower())
        score += min(medical_term_count * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate basic readability score."""
        # Simple readability based on sentence length and word complexity
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Flesch-like score (simplified)
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 6))
        return max(0, score)
    
    def _calculate_coherence_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate conversation coherence metrics."""
        coherence_scores = []
        
        if 'user_input' in df.columns and 'assistant_response' in df.columns:
            for _, row in df.iterrows():
                user_text = str(row.get('user_input', ''))
                response_text = str(row.get('assistant_response', ''))
                
                if user_text and response_text:
                    similarity = self._calculate_text_similarity(user_text, response_text)
                    coherence_scores.append(similarity)
        
        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        return {
            'avg_coherence': avg_coherence,
            'coherence_scores': coherence_scores
        }
    
    def _calculate_medical_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate medical accuracy indicators."""
        accuracy_metrics = {}
        
        # Check for appropriate medical terminology
        medical_accuracy_scores = []
        
        if 'assistant_response' in df.columns:
            for response in df['assistant_response'].dropna():
                response_text = str(response)
                medical_terms_found = sum(1 for term in self.config.medical_terms 
                                        if term.lower() in response_text.lower())
                # Normalize by text length
                term_density = medical_terms_found / max(len(response_text.split()), 1)
                medical_accuracy_scores.append(min(term_density * 100, 1.0))
        
        accuracy_metrics['avg_medical_term_usage'] = np.mean(medical_accuracy_scores) if medical_accuracy_scores else 0.0
        
        # Check for safety indicators
        safety_indicators = ['recommend', 'suggest', 'advise', 'caution', 'emergency', 'urgent']
        safety_scores = []
        
        if 'assistant_response' in df.columns:
            for response in df['assistant_response'].dropna():
                response_text = str(response).lower()
                safety_score = sum(1 for indicator in safety_indicators if indicator in response_text)
                safety_scores.append(min(safety_score / len(safety_indicators), 1.0))
        
        accuracy_metrics['avg_safety_indicators'] = np.mean(safety_scores) if safety_scores else 0.0
        
        return accuracy_metrics
    
    def _calculate_satisfaction_proxies(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate user satisfaction proxies."""
        # These are heuristic measures based on text characteristics
        satisfaction_indicators = {
            'response_length_appropriateness': 0.0,
            'question_follow_through': 0.0,
            'conversation_completeness': 0.0
        }
        
        if 'assistant_response' in df.columns and 'user_input' in df.columns:
            response_lengths = df['assistant_response'].dropna().astype(str).str.len()
            user_lengths = df['user_input'].dropna().astype(str).str.len()
            
            # Appropriate response length (not too short, not too long)
            if len(response_lengths) > 0 and len(user_lengths) > 0:
                avg_response_length = response_lengths.mean()
                avg_user_length = user_lengths.mean()
                
                # Ideal response should be 1-3x the user input length
                if 0.8 <= avg_response_length / max(avg_user_length, 1) <= 3.0:
                    satisfaction_indicators['response_length_appropriateness'] = 0.8
                elif avg_response_length > avg_user_length:
                    satisfaction_indicators['response_length_appropriateness'] = 0.6
                else:
                    satisfaction_indicators['response_length_appropriateness'] = 0.4
        
        # Calculate overall satisfaction proxy
        overall_satisfaction = np.mean(list(satisfaction_indicators.values()))
        satisfaction_indicators['overall_proxy'] = overall_satisfaction
        
        return satisfaction_indicators
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF and cosine similarity."""
        if not text1 or not text2:
            return 0.0
        
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            
            if tfidf_matrix.shape[0] == 2:
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return similarity
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_overall_score(self, result: ValidationResult) -> float:
        """Calculate overall validation score."""
        score = 1.0
        
        # Penalize errors heavily
        score -= len(result.errors) * 0.2
        
        # Penalize warnings
        score -= len(result.warnings) * 0.05
        
        # Bonus for high quality metrics
        if 'quality' in result.metrics:
            quality = result.metrics['quality']
            if 'avg_text_quality' in quality:
                score += quality['avg_text_quality'] * 0.1
            if 'coherence' in quality and 'avg_coherence' in quality['coherence']:
                score += quality['coherence']['avg_coherence'] * 0.1
        
        # Penalty for poor quality metrics
        if 'missing_patterns' in result.metrics:
            missing_pct = result.metrics['missing_patterns'].get('total_missing_percentage', 0)
            if missing_pct > 10:
                score -= (missing_pct - 10) * 0.01
        
        return max(0.0, min(1.0, score))
    
    def validate_single_record(self, record: Dict[str, Any]) -> ValidationResult:
        """Validate a single record."""
        # For single record validation, convert to DataFrame and run validation
        df = pd.DataFrame([record])
        return self.validate_dataset(df)
    
    def batch_validate(self, data_batches: List[Union[List[Dict], pd.DataFrame]]) -> List[ValidationResult]:
        """Validate multiple data batches."""
        results = []
        for i, batch in enumerate(data_batches):
            self.logger.info(f"Validating batch {i+1}/{len(data_batches)}")
            result = self.validate_dataset(batch)
            results.append(result)
        
        return results


class MedicalDataValidator(DataValidator):
    """Specialized validator for medical training data."""
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        # Add medical-specific patterns
        self.medical_abbreviations = {
            'BP': 'blood pressure',
            'HR': 'heart rate',
            'Temp': 'temperature',
            'O2': 'oxygen',
            'SOB': 'shortness of breath',
            'N/V': 'nausea/vomiting',
            'LOC': 'loss of consciousness'
        }
    
    def _validate_symptoms(self, df: pd.DataFrame, result: ValidationResult) -> None:
        """Enhanced symptom validation for medical data."""
        super()._validate_symptoms(df, result)
        
        # Additional medical symptom validation
        if 'symptoms' not in df.columns:
            return
        
        medical_symptoms = df['symptoms'].dropna().astype(str)
        
        # Check for medical abbreviations
        abbreviations_found = []
        for symptom in medical_symptoms:
            for abbrev, full_form in self.medical_abbreviations.items():
                if abbrev in symptom and full_form.lower() not in symptom.lower():
                    abbreviations_found.append(f"Abbreviation '{abbrev}' used without expansion")
        
        if abbreviations_found:
            result.warnings.append(f"Medical abbreviations without expansion: {len(abbreviations_found)} found")
        
        # Check for symptom severity indicators
        severity_indicators = ['severe', 'mild', 'moderate', 'intense', 'sharp', 'dull']
        severity_coverage = 0
        
        for symptom in medical_symptoms:
            if any(indicator in symptom.lower() for indicator in severity_indicators):
                severity_coverage += 1
        
        severity_percentage = (severity_coverage / len(medical_symptoms)) * 100 if medical_symptoms else 0
        if severity_percentage < 30:
            result.warnings.append(f"Low symptom severity indication: {severity_percentage:.1f}%")
    
    def _calculate_medical_accuracy(self, df: pd.DataFrame) -> Dict[str, float]:
        """Enhanced medical accuracy calculation."""
        base_accuracy = super()._calculate_medical_accuracy(df)
        
        # Add medical-specific accuracy metrics
        medical_accuracy = base_accuracy.copy()
        
        # Check for proper medical formatting
        if 'assistant_response' in df.columns:
            formatting_score = 0
            total_responses = 0
            
            for response in df['assistant_response'].dropna():
                response_text = str(response).lower()
                total_responses += 1
                
                # Check for appropriate medical disclaimers
                if any(disclaimer in response_text for disclaimer in 
                      ['not a substitute', 'professional advice', 'emergency']):
                    formatting_score += 1
                
                # Check for appropriate response structure
                if any(phrase in response_text for phrase in 
                      ['i understand', 'i recommend', 'you should']):
                    formatting_score += 1
            
            if total_responses > 0:
                medical_accuracy['medical_formatting_score'] = formatting_score / (total_responses * 2)
            else:
                medical_accuracy['medical_formatting_score'] = 0.0
        
        return medical_accuracy