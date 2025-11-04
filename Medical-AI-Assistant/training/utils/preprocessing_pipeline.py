"""
Preprocessing Pipeline for Medical AI Training Data

This module provides comprehensive data preprocessing capabilities including:
- Multi-stage data preprocessing
- Custom transformation pipelines  
- Batch processing capabilities
- Memory optimization techniques
"""

import re
import json
import logging
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
import concurrent.futures
from threading import Lock
import gzip
import pickle
import time
import sys

# Add training utils to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .data_augmentation import AugmentationConfig, DataAugmentor
except ImportError:
    # Fallback for direct execution
    from data_augmentation import AugmentationConfig, DataAugmentor


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    # Data cleaning
    remove_empty_conversations: bool = True
    min_conversation_length: int = 2
    max_conversation_length: int = 50
    min_text_length: int = 5
    max_text_length: int = 1000
    
    # Text normalization
    normalize_whitespace: bool = True
    remove_special_characters: bool = False
    standardize_medical_terms: bool = True
    fix_encoding_errors: bool = True
    
    # Quality filtering
    enable_quality_filters: bool = True
    filter_inappropriate_content: bool = True
    filter_medical_inaccuracies: bool = True
    validate_conversation_flow: bool = True
    
    # Batch processing
    batch_size: int = 1000
    max_workers: int = 4
    chunk_size: int = 10000
    
    # Memory optimization
    enable_streaming: bool = True
    cache_intermediate_results: bool = True
    compress_cache: bool = True
    max_memory_usage_mb: int = 1024
    
    # Output configuration
    output_format: str = "json"  # json, csv, parquet
    include_metadata: bool = True
    add_processing_info: bool = True


class DataCleaner:
    """Handles data cleaning operations"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cleaning patterns
        self.cleaning_patterns = {
            "whitespace": re.compile(r'\s+'),
            "special_chars": re.compile(r'[^\w\s\.\!\?\;\:\,\-\(\)]'),
            "encoding_errors": re.compile(r'[^\x00-\x7F]+'),
            "empty_lines": re.compile(r'^\s*$', re.MULTILINE)
        }
        
        # Medical term standardization
        self.medical_terms_map = self._initialize_medical_terms_map()
        
        # Inappropriate content patterns
        self.inappropriate_patterns = self._initialize_inappropriate_patterns()
    
    def _initialize_medical_terms_map(self) -> Dict[str, str]:
        """Initialize medical term standardization mapping"""
        return {
            "dr": "doctor",
            "md": "medical doctor",
            "er": "emergency room",
            "rx": "prescription",
            "bp": "blood pressure",
            "hr": "heart rate",
            "temp": "temperature",
            "vs": "vital signs",
            "po": "by mouth",
            "iv": "intravenous",
            "im": "intramuscular",
            "sc": "subcutaneous",
            "prn": "as needed",
            "tid": "three times daily",
            "bid": "twice daily",
            "qid": "four times daily"
        }
    
    def _initialize_inappropriate_patterns(self) -> List[str]:
        """Initialize patterns for inappropriate content filtering"""
        return [
            r"\b(suicide|kill myself|end my life)\b",
            r"\b(overdose|poison|harm)\b.*\b(medication|drug)\b",
            r"\b(ignore|disregard)\b.*\b(doctor|physician|medical)\b.*\b(advice|instruction)\b",
            r"\b(self-medicate|treat myself)\b.*\b(without|without)\b.*\b(doctor|medical supervision)\b"
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean individual text"""
        if not text:
            return ""
        
        # Fix encoding errors
        if self.config.fix_encoding_errors:
            try:
                text = text.encode('ascii', errors='ignore').decode('ascii')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        
        # Remove special characters (if enabled)
        if self.config.remove_special_characters:
            text = self.cleaning_patterns["special_chars"].sub(' ', text)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.cleaning_patterns["whitespace"].sub(' ', text)
            text = text.strip()
        
        # Standardize medical terms
        if self.config.standardize_medical_terms:
            text = self._standardize_medical_terms(text)
        
        return text
    
    def _standardize_medical_terms(self, text: str) -> str:
        """Standardize medical abbreviations and terms"""
        words = text.split()
        standardized_words = []
        
        for word in words:
            # Remove common punctuation for comparison
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.medical_terms_map:
                replacement = self.medical_terms_map[clean_word]
                # Preserve original capitalization
                if word.isupper():
                    replacement = replacement.upper()
                elif word.istitle():
                    replacement = replacement.capitalize()
                standardized_words.append(replacement)
            else:
                standardized_words.append(word)
        
        return ' '.join(standardized_words)
    
    def filter_inappropriate_content(self, text: str) -> bool:
        """Check if text contains inappropriate content"""
        if not self.config.filter_inappropriate_content:
            return True
        
        text_lower = text.lower()
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, text_lower):
                return False
        
        return True
    
    def validate_conversation_structure(self, conversation: List[Dict[str, Any]]) -> bool:
        """Validate conversation structure and flow"""
        if not self.config.validate_conversation_flow:
            return True
        
        if len(conversation) < self.config.min_conversation_length:
            return False
        
        if len(conversation) > self.config.max_conversation_length:
            return False
        
        # Check for alternating speakers
        speakers = [turn.get("speaker") for turn in conversation]
        if not speakers:
            return False
        
        # Basic validation: ensure we have both patient and AI turns
        unique_speakers = set(speakers)
        if len(unique_speakers) < 1:  # At least one type of speaker
            return False
        
        # Check for reasonable turn lengths
        for turn in conversation:
            text = turn.get("text", "")
            if len(text) < self.config.min_text_length:
                return False
            if len(text) > self.config.max_text_length:
                return False
        
        return True
    
    def process_conversation(self, conversation: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Process and clean a single conversation"""
        
        # Validate structure
        if not self.validate_conversation_structure(conversation):
            return None
        
        cleaned_conversation = []
        
        for turn in conversation:
            # Clean text
            text = self.clean_text(turn.get("text", ""))
            
            # Filter inappropriate content
            if not self.filter_inappropriate_content(text):
                continue
            
            # Create cleaned turn
            cleaned_turn = {
                "speaker": turn.get("speaker", "unknown"),
                "text": text,
                "timestamp": turn.get("timestamp"),
                "metadata": turn.get("metadata", {})
            }
            
            # Add original index for tracking
            if "index" in turn:
                cleaned_turn["original_index"] = turn["index"]
            
            cleaned_conversation.append(cleaned_turn)
        
        return cleaned_conversation if cleaned_conversation else None


class DataTransformer:
    """Handles data transformation operations"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize transformers
        self.transformers = {
            "add_conversation_stats": self._add_conversation_stats,
            "extract_symptoms": self._extract_symptoms,
            "categorize_intent": self._categorize_intent,
            "add_sentiment": self._add_sentiment,
            "standardize_format": self._standardize_format
        }
    
    def _add_conversation_stats(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add statistical information to conversation"""
        stats = {
            "num_turns": len(conversation),
            "patient_turns": sum(1 for turn in conversation if turn.get("speaker") == "patient"),
            "ai_turns": sum(1 for turn in conversation if turn.get("speaker") == "ai"),
            "total_characters": sum(len(turn.get("text", "")) for turn in conversation),
            "avg_turn_length": np.mean([len(turn.get("text", "")) for turn in conversation]) if conversation else 0,
            "conversation_duration": self._estimate_conversation_duration(conversation)
        }
        
        # Add to each turn's metadata
        for turn in conversation:
            if "metadata" not in turn:
                turn["metadata"] = {}
            turn["metadata"]["conversation_stats"] = stats
        
        return stats
    
    def _estimate_conversation_duration(self, conversation: List[Dict[str, Any]]) -> float:
        """Estimate conversation duration based on content"""
        if not conversation:
            return 0.0
        
        # Simple heuristic based on number of turns
        base_time = len(conversation) * 0.5  # 30 seconds per turn on average
        return base_time
    
    def _extract_symptoms(self, conversation: List[Dict[str, Any]]) -> List[str]:
        """Extract mentioned symptoms from conversation"""
        symptoms = []
        symptom_keywords = [
            "pain", "fever", "cough", "headache", "nausea", "dizziness", "fatigue",
            "shortness of breath", "chest pain", "stomach pain", "back pain",
            "sore throat", "runny nose", "fever", "chills", "sweating"
        ]
        
        full_text = " ".join(turn.get("text", "").lower() for turn in conversation)
        
        for symptom in symptom_keywords:
            if symptom in full_text:
                symptoms.append(symptom)
        
        return list(set(symptoms))
    
    def _categorize_intent(self, conversation: List[Dict[str, Any]]) -> str:
        """Categorize the intent of the conversation"""
        full_text = " ".join(turn.get("text", "").lower() for turn in conversation)
        
        # Intent categories
        if any(word in full_text for word in ["emergency", "urgent", "severe", "chest pain"]):
            return "emergency"
        elif any(word in full_text for word in ["checkup", "routine", "annual", "follow-up"]):
            return "routine_checkup"
        elif any(word in full_text for word in ["symptom", "pain", "problem"]):
            return "symptom_inquiry"
        elif any(word in full_text for word in ["medication", "prescription", "drug"]):
            return "medication_inquiry"
        elif any(word in full_text for word in ["test", "result", "lab"]):
            return "test_results"
        else:
            return "general_inquiry"
    
    def _add_sentiment(self, conversation: List[Dict[str, Any]]) -> Dict[str, float]:
        """Add basic sentiment analysis"""
        # Simple sentiment analysis based on keywords
        positive_words = ["good", "better", "improved", "fine", "okay", "great"]
        negative_words = ["bad", "worse", "terrible", "awful", "severe", "pain"]
        urgency_words = ["urgent", "emergency", "immediate", "now", "asap"]
        
        full_text = " ".join(turn.get("text", "").lower() for turn in conversation)
        
        positive_count = sum(1 for word in positive_words if word in full_text)
        negative_count = sum(1 for word in negative_words if word in full_text)
        urgency_count = sum(1 for word in urgency_words if word in full_text)
        
        total_words = len(full_text.split())
        
        sentiment = {
            "positive_score": positive_count / total_words if total_words > 0 else 0,
            "negative_score": negative_count / total_words if total_words > 0 else 0,
            "urgency_score": urgency_count / total_words if total_words > 0 else 0,
            "overall_sentiment": "neutral"
        }
        
        if positive_count > negative_count:
            sentiment["overall_sentiment"] = "positive"
        elif negative_count > positive_count:
            sentiment["overall_sentiment"] = "negative"
        
        if urgency_count > 0:
            sentiment["overall_sentiment"] = "urgent"
        
        return sentiment
    
    def _standardize_format(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize the format of conversation data"""
        standardized = []
        
        for i, turn in enumerate(conversation):
            standardized_turn = {
                "id": f"turn_{i}",
                "speaker": turn.get("speaker", "unknown").lower().strip(),
                "text": turn.get("text", "").strip(),
                "sequence": i
            }
            
            # Add optional fields
            if turn.get("timestamp"):
                standardized_turn["timestamp"] = turn["timestamp"]
            
            if turn.get("metadata"):
                standardized_turn["metadata"] = turn["metadata"]
            
            standardized.append(standardized_turn)
        
        return standardized
    
    def transform_conversation(self, conversation: List[Dict[str, Any]], 
                             transformations: List[str] = None) -> List[Dict[str, Any]]:
        """Apply specified transformations to conversation"""
        
        if transformations is None:
            transformations = ["standardize_format", "add_conversation_stats"]
        
        transformed = conversation.copy()
        
        for transform_name in transformations:
            if transform_name in self.transformers:
                transform_func = self.transformers[transform_name]
                
                if transform_name == "standardize_format":
                    transformed = transform_func(transformed)
                else:
                    # Functions that add metadata
                    result = transform_func(transformed)
                    if isinstance(result, dict):
                        # Add to conversation metadata
                        for turn in transformed:
                            if "metadata" not in turn:
                                turn["metadata"] = {}
                            turn["metadata"][transform_name] = result
        
        return transformed


class MemoryOptimizedProcessor:
    """Handles memory-optimized processing of large datasets"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._memory_lock = Lock()
        self._memory_usage = 0
        
        # Cache for intermediate results
        self._cache = {}
        if config.cache_intermediate_results:
            self._setup_cache()
    
    def _setup_cache(self):
        """Setup caching directory and compression"""
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback estimation
            return self._memory_usage
    
    def check_memory_limit(self) -> bool:
        """Check if memory usage is within limits"""
        return self.get_memory_usage() < self.config.max_memory_usage_mb
    
    def cache_result(self, key: str, data: Any):
        """Cache result with memory management"""
        if not self.config.cache_intermediate_results:
            return
        
        with self._memory_lock:
            # Calculate size
            try:
                size = sys.getsizeof(pickle.dumps(data))
                if self.check_memory_limit() or key not in self._cache:
                    # Compress if enabled
                    if self.config.compress_cache:
                        compressed_data = gzip.compress(pickle.dumps(data))
                        self._cache[key] = compressed_data
                        self._memory_usage += len(compressed_data)
                    else:
                        self._cache[key] = pickle.dumps(data)
                        self._memory_usage += size
            except Exception as e:
                self.logger.warning(f"Failed to cache result {key}: {e}")
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Retrieve cached result"""
        if not self.config.cache_intermediate_results:
            return None
        
        with self._memory_lock:
            if key in self._cache:
                try:
                    data = self._cache[key]
                    if self.config.compress_cache:
                        return pickle.loads(gzip.decompress(data))
                    else:
                        return pickle.loads(data)
                except Exception as e:
                    self.logger.warning(f"Failed to retrieve cached result {key}: {e}")
                    return None
        return None
    
    def process_in_chunks(self, data: Iterator[Any], processor_func: Callable, 
                         chunk_size: int = None) -> Iterator[Any]:
        """Process data in chunks to manage memory"""
        
        if chunk_size is None:
            chunk_size = self.config.chunk_size
        
        chunk = []
        chunk_count = 0
        
        for item in data:
            chunk.append(item)
            
            if len(chunk) >= chunk_size:
                self.logger.info(f"Processing chunk {chunk_count + 1}...")
                
                # Process chunk
                processed_chunk = processor_func(chunk)
                
                # Yield results
                for result in processed_chunk:
                    yield result
                
                # Clear chunk to free memory
                chunk = []
                chunk_count += 1
                
                # Memory check
                if not self.check_memory_limit():
                    self.logger.warning("Memory usage high, forcing garbage collection")
                    import gc
                    gc.collect()
        
        # Process remaining items
        if chunk:
            processed_chunk = processor_func(chunk)
            for result in processed_chunk:
                yield result


class PreprocessingPipeline:
    """Main preprocessing pipeline orchestrator"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        
        # Initialize components
        self.data_cleaner = DataCleaner(self.config)
        self.data_transformer = DataTransformer(self.config)
        self.memory_processor = MemoryOptimizedProcessor(self.config)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            "total_conversations": 0,
            "cleaned_conversations": 0,
            "filtered_conversations": 0,
            "processing_time": 0,
            "memory_peak": 0
        }
    
    def preprocess_dataset(self, 
                          data: Union[List[Dict], Iterator[Dict]], 
                          transformations: List[str] = None) -> Dict[str, Any]:
        """Preprocess complete dataset"""
        
        start_time = time.time()
        self.logger.info("Starting dataset preprocessing...")
        
        # Ensure we have data
        if isinstance(data, Iterator):
            data = list(data)  # Convert iterator to list for processing
        
        self.stats["total_conversations"] = len(data)
        
        # Process conversations
        processed_conversations = []
        filtered_count = 0
        
        if self.config.enable_streaming and isinstance(data, list) and len(data) > self.config.chunk_size:
            # Use chunked processing for large datasets
            processed_conversations = list(self._process_in_batches(data))
        else:
            # Process all at once for smaller datasets
            for i, conversation_data in enumerate(data):
                if i % 1000 == 0:
                    self.logger.info(f"Processed {i}/{len(data)} conversations...")
                
                conversation = conversation_data.get("conversation", [])
                
                # Clean conversation
                cleaned_conversation = self.data_cleaner.process_conversation(conversation)
                
                if cleaned_conversation is None:
                    filtered_count += 1
                    continue
                
                # Transform conversation
                transformed_conversation = self.data_transformer.transform_conversation(
                    cleaned_conversation, transformations
                )
                
                # Add original data metadata
                processed_conversation = {
                    "conversation": transformed_conversation,
                    "original_data": conversation_data,
                    "processing_info": {
                        "processed_at": datetime.now().isoformat(),
                        "pipeline_version": "1.0",
                        "config": self.config.__dict__
                    }
                }
                
                processed_conversations.append(processed_conversation)
        
        self.stats["cleaned_conversations"] = len(processed_conversations)
        self.stats["filtered_conversations"] = filtered_count
        self.stats["processing_time"] = time.time() - start_time
        self.stats["memory_peak"] = self.memory_processor.get_memory_usage()
        
        self.logger.info(f"Preprocessing complete. Processed {len(processed_conversations)} conversations, "
                        f"filtered {filtered_count}. Time: {self.stats['processing_time']:.2f}s")
        
        return {
            "conversations": processed_conversations,
            "statistics": self.stats,
            "config": self.config.__dict__,
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def _process_in_batches(self, data: List[Dict]) -> Iterator[Dict]:
        """Process data in batches"""
        
        def process_batch(batch: List[Dict]) -> List[Dict]:
            processed_batch = []
            
            for conversation_data in batch:
                conversation = conversation_data.get("conversation", [])
                
                # Clean conversation
                cleaned_conversation = self.data_cleaner.process_conversation(conversation)
                
                if cleaned_conversation is None:
                    continue
                
                # Transform conversation
                transformed_conversation = self.data_transformer.transform_conversation(
                    cleaned_conversation
                )
                
                processed_conversation = {
                    "conversation": transformed_conversation,
                    "original_data": conversation_data,
                    "processing_info": {
                        "processed_at": datetime.now().isoformat(),
                        "pipeline_version": "1.0"
                    }
                }
                
                processed_batch.append(processed_conversation)
            
            return processed_batch
        
        # Use memory-optimized processor
        yield from self.memory_processor.process_in_chunks(
            data, process_batch, self.config.batch_size
        )
    
    def save_preprocessed_data(self, 
                             preprocessed_data: Dict[str, Any], 
                             output_path: str):
        """Save preprocessed data to file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.output_format == "json":
            with open(output_file, 'w') as f:
                json.dump(preprocessed_data, f, indent=2, default=str)
        elif self.config.output_format == "pickle":
            with open(output_file, 'wb') as f:
                pickle.dump(preprocessed_data, f)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
        
        self.logger.info(f"Saved preprocessed data to {output_path}")
    
    def generate_processing_report(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed processing report"""
        
        conversations = preprocessed_data.get("conversations", [])
        stats = preprocessed_data.get("statistics", {})
        
        # Analyze conversation characteristics
        total_turns = sum(len(conv.get("conversation", [])) for conv in conversations)
        avg_turns_per_conversation = total_turns / len(conversations) if conversations else 0
        
        # Analyze text characteristics
        all_texts = []
        for conv in conversations:
            for turn in conv.get("conversation", []):
                all_texts.append(turn.get("text", ""))
        
        avg_text_length = np.mean([len(text) for text in all_texts]) if all_texts else 0
        
        # Analyze speaker distribution
        speaker_counts = Counter()
        for conv in conversations:
            for turn in conv.get("conversation", []):
                speaker_counts[turn.get("speaker", "unknown")] += 1
        
        report = {
            "processing_summary": {
                "total_conversations_processed": len(conversations),
                "total_turns": total_turns,
                "average_turns_per_conversation": avg_turns_per_conversation,
                "average_text_length": avg_text_length,
                "speaker_distribution": dict(speaker_counts),
                "processing_time_seconds": stats.get("processing_time", 0),
                "memory_peak_mb": stats.get("memory_peak", 0)
            },
            "quality_metrics": {
                "filtering_rate": stats.get("filtered_conversations", 0) / stats.get("total_conversations", 1),
                "data_quality_score": self._calculate_quality_score(conversations)
            },
            "configuration": preprocessed_data.get("config", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_quality_score(self, conversations: List[Dict]) -> float:
        """Calculate overall data quality score"""
        if not conversations:
            return 0.0
        
        quality_factors = []
        
        # Factor 1: Conversation completeness
        complete_conversations = sum(
            1 for conv in conversations 
            if len(conv.get("conversation", [])) >= 2
        )
        quality_factors.append(complete_conversations / len(conversations))
        
        # Factor 2: Text quality (length distribution)
        text_lengths = []
        for conv in conversations:
            for turn in conv.get("conversation", []):
                text_lengths.append(len(turn.get("text", "")))
        
        if text_lengths:
            # Quality is higher when text lengths are reasonable (not too short/long)
            reasonable_lengths = sum(
                1 for length in text_lengths 
                if 10 <= length <= 500
            )
            quality_factors.append(reasonable_lengths / len(text_lengths))
        
        # Factor 3: Speaker diversity
        speaker_diversity_scores = []
        for conv in conversations:
            speakers = set(turn.get("speaker") for turn in conv.get("conversation", []))
            speaker_diversity_scores.append(len(speakers) / 2)  # Expected 2 speakers max
        
        quality_factors.append(np.mean(speaker_diversity_scores) if speaker_diversity_scores else 0)
        
        return np.mean(quality_factors)


def create_preprocessing_pipeline(config_dict: Dict[str, Any] = None) -> PreprocessingPipeline:
    """Create preprocessing pipeline with configuration"""
    
    if config_dict:
        config = PreprocessingConfig(**config_dict)
    else:
        config = PreprocessingConfig()
    
    return PreprocessingPipeline(config)


def preprocess_medical_conversations(input_file: str, 
                                   output_file: str, 
                                   config_dict: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience function to preprocess medical conversations"""
    
    # Load input data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create pipeline
    pipeline = create_preprocessing_pipeline(config_dict)
    
    # Process data
    preprocessed_data = pipeline.preprocess_dataset(data)
    
    # Save results
    pipeline.save_preprocessed_data(preprocessed_data, output_file)
    
    # Generate report
    report = pipeline.generate_processing_report(preprocessed_data)
    
    return {
        "preprocessed_data_path": output_file,
        "processing_report": report,
        "pipeline_stats": pipeline.stats
    }


if __name__ == "__main__":
    # Example usage
    config = PreprocessingConfig(
        batch_size=500,
        enable_streaming=True,
        output_format="json"
    )
    
    pipeline = PreprocessingPipeline(config)
    print("Preprocessing Pipeline Initialized")
    print(f"Configuration: {config}")