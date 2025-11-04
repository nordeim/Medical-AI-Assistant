"""
Data preprocessing utilities for LoRA training.
Handles various data formats and preprocessing tasks.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import hashlib

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing."""
    max_length: int = 2048
    min_length: int = 10
    remove_empty: bool = True
    remove_duplicates: bool = True
    filter_by_length: bool = True
    normalize_whitespace: bool = True
    remove_special_tokens: bool = False
    add_system_prompt: bool = False
    system_prompt: str = "You are a helpful assistant."
    conversation_format: str = "chatml"  # chatml, sharegpt, alpaca
    response_key: str = "response"
    prompt_key: str = "prompt"
    text_key: str = "text"

class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_json_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean JSON data."""
        valid_data = []
        
        for i, item in enumerate(data):
            try:
                # Ensure item is a dictionary
                if not isinstance(item, dict):
                    logger.warning(f"Item {i} is not a dictionary, skipping")
                    continue
                
                # Check for required fields
                if not any(key in item for key in ['text', 'prompt', 'response']):
                    logger.warning(f"Item {i} missing required text fields, skipping")
                    continue
                
                # Clean empty values
                cleaned_item = {k: v for k, v in item.items() if v is not None and v != ""}
                
                valid_data.append(cleaned_item)
                
            except Exception as e:
                logger.warning(f"Error processing item {i}: {e}")
                continue
        
        logger.info(f"Validated {len(valid_data)}/{len(data)} items")
        return valid_data
    
    @staticmethod
    def validate_text_quality(text: str) -> bool:
        """Validate text quality."""
        if not text or not isinstance(text, str):
            return False
        
        # Check minimum length
        if len(text.strip()) < 10:
            return False
        
        # Check for excessive whitespace
        if text.count(' ') / len(text) > 0.5:  # More than 50% whitespace
            return False
        
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in '.,!?;:-')
        if special_chars / len(text) > 0.3:  # More than 30% special characters
            return False
        
        return True

class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.validator = DataValidator()
        
    def load_data(self, data_path: str, data_type: str = "jsonl") -> Dataset:
        """Load data from various sources."""
        logger.info(f"Loading data from {data_path} (type: {data_type})")
        
        try:
            if data_type == "jsonl":
                data = []
                with open(data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return Dataset.from_list(data)
                
            elif data_type == "json":
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return Dataset.from_list(data)
                
            elif data_type == "csv":
                df = pd.read_csv(data_path)
                return Dataset.from_pandas(df)
                
            elif data_type == "txt":
                with open(data_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f if line.strip()]
                return Dataset.from_dict({"text": texts})
                
            elif data_type == "datasets":
                return load_dataset(data_path)
                
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
                
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess individual text."""
        if not text:
            return ""
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special tokens if specified
        if self.config.remove_special_tokens:
            text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def format_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data according to conversation format."""
        if self.config.conversation_format == "chatml":
            return self._format_chatml(data)
        elif self.config.conversation_format == "sharegpt":
            return self._format_sharegpt(data)
        elif self.config.conversation_format == "alpaca":
            return self._format_alpaca(data)
        else:
            return data
    
    def _format_chatml(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data in ChatML style."""
        if self.config.prompt_key in data and self.config.response_key in data:
            prompt = data[self.config.prompt_key]
            response = data[self.config.response_key]
            
            if self.config.add_system_prompt:
                text = f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            
            data['text'] = text
            data['input_ids'] = prompt + response  # For compatibility
        
        return data
    
    def _format_sharegpt(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data in ShareGPT style."""
        if 'conversations' in data:
            conversations = data['conversations']
            formatted_parts = []
            
            if self.config.add_system_prompt:
                formatted_parts.append(f"<|im_start|>system\n{self.config.system_prompt}<|im_end|>")
            
            for conv in conversations:
                role = conv.get('from', 'human')
                value = conv.get('value', '')
                
                if role == 'human':
                    formatted_parts.append(f"<|im_start|>user\n{value}<|im_end|>")
                elif role == 'gpt':
                    formatted_parts.append(f"<|im_start|>assistant\n{value}<|im_end|>")
            
            data['text'] = '\n'.join(formatted_parts)
        
        return data
    
    def _format_alpaca(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data in Alpaca style."""
        if 'instruction' in data and 'input' in data and 'output' in data:
            instruction = data['instruction']
            input_text = data['input']
            output = data['output']
            
            if input_text.strip():
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
            data['text'] = prompt
        
        return data
    
    def filter_and_clean(self, dataset: Dataset) -> Dataset:
        """Filter and clean dataset."""
        logger.info("Filtering and cleaning dataset")
        
        original_size = len(dataset)
        
        def filter_function(example):
            # Get text content
            text = ""
            if self.config.text_key in example:
                text = example[self.config.text_key]
            elif self.config.prompt_key in example and self.config.response_key in example:
                text = example[self.config.prompt_key] + " " + example[self.config.response_key]
            
            # Remove empty if specified
            if self.config.remove_empty and (not text or not text.strip()):
                return False
            
            # Length filtering
            if self.config.filter_by_length:
                text_len = len(text.strip())
                if text_len < self.config.min_length or text_len > self.config.max_length:
                    return False
            
            # Quality validation
            if not self.validator.validate_text_quality(text):
                return False
            
            # Preprocess text
            text = self.preprocess_text(text)
            
            # Update example
            if self.config.text_key in example:
                example[self.config.text_key] = text
            
            return True
        
        # Apply filtering
        filtered_dataset = dataset.filter(filter_function, batched=False)
        
        logger.info(f"Dataset filtered: {len(filtered_dataset)}/{original_size} samples")
        
        # Remove duplicates if specified
        if self.config.remove_duplicates and len(filtered_dataset) > 0:
            filtered_dataset = self._remove_duplicates(filtered_dataset)
        
        return filtered_dataset
    
    def _remove_duplicates(self, dataset: Dataset) -> Dataset:
        """Remove duplicate samples from dataset."""
        logger.info("Removing duplicates")
        
        # Create hash for each sample
        def hash_example(example):
            # Create hash based on text content
            text = ""
            if self.config.text_key in example:
                text = example[self.config.text_key]
            elif self.config.prompt_key in example and self.config.response_key in example:
                text = example[self.config.prompt_key] + example[self.config.response_key]
            
            hash_obj = hashlib.md5(text.encode('utf-8'))
            return {'hash': hash_obj.hexdigest()}
        
        # Add hash column
        hashed_dataset = dataset.map(hash_example)
        
        # Remove duplicates based on hash
        deduplicated = hashed_dataset.remove_duplicates(subset=['hash'], keep='first')
        
        # Remove hash column
        final_dataset = deduplicated.remove_columns(['hash'])
        
        logger.info(f"Duplicates removed: {len(final_dataset)} samples")
        return final_dataset
    
    def convert_to_standard_format(self, dataset: Dataset) -> Dataset:
        """Convert dataset to standard format for training."""
        logger.info("Converting to standard format")
        
        def convert_function(example):
            # Extract text based on available keys
            text = ""
            if self.config.text_key in example:
                text = example[self.config.text_key]
            elif self.config.prompt_key in example and self.config.response_key in example:
                text = example[self.config.prompt_key] + " " + example[self.config.response_key]
            else:
                # Try to find any text-like field
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 10:
                        text = value
                        break
            
            if not text:
                return None
            
            # Format conversation if needed
            formatted_example = {
                'text': text,
                'original_example': example
            }
            
            # Add other fields
            for key, value in example.items():
                if key not in [self.config.text_key, self.config.prompt_key, self.config.response_key]:
                    formatted_example[key] = value
            
            return formatted_example
        
        # Apply conversion
        converted_dataset = dataset.map(
            convert_function,
            remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else []
        )
        
        # Filter out None values
        converted_dataset = converted_dataset.filter(lambda x: x is not None)
        
        return converted_dataset
    
    def prepare_dataset(
        self,
        data_path: str,
        data_type: str = "jsonl",
        save_path: Optional[str] = None
    ) -> Dataset:
        """Complete data preparation pipeline."""
        logger.info("Starting data preparation pipeline")
        
        # Load data
        dataset = self.load_data(data_path, data_type)
        
        # Validate data
        if isinstance(dataset, list):
            validated_data = self.validator.validate_json_data(dataset)
            dataset = Dataset.from_list(validated_data)
        
        # Convert to standard format
        dataset = self.convert_to_standard_format(dataset)
        
        # Filter and clean
        dataset = self.filter_and_clean(dataset)
        
        # Format conversations
        if self.config.conversation_format:
            def format_example(example):
                return self.format_conversation(example)
            
            dataset = dataset.map(format_example)
        
        # Save if path provided
        if save_path:
            dataset.save_to_disk(save_path)
            logger.info(f"Processed dataset saved to {save_path}")
        
        logger.info(f"Data preparation completed: {len(dataset)} samples")
        return dataset

class ChatMLProcessor:
    """Specialized processor for ChatML format data."""
    
    @staticmethod
    def parse_conversations(text: str) -> List[Dict[str, str]]:
        """Parse ChatML format conversations."""
        pattern = r'<\|im_start\|>([^|]+)\n([^<]*?)<\|im_end\|>'
        matches = re.findall(pattern, text)
        
        conversations = []
        for role, content in matches:
            role = role.strip()
            content = content.strip()
            if role and content:
                conversations.append({
                    'role': role,
                    'content': content
                })
        
        return conversations
    
    @staticmethod
    def create_conversation(prompt: str, response: str, system: Optional[str] = None) -> str:
        """Create ChatML format conversation."""
        parts = []
        
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append(f"<|im_start|>assistant\n{response}<|im_end|>")
        
        return '\n'.join(parts)
    
    @staticmethod
    def extract_prompt_response(conversation: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract prompt and response from ChatML conversation."""
        conversations = ChatMLProcessor.parse_conversations(conversation)
        
        user_text = ""
        assistant_text = ""
        
        for conv in conversations:
            if conv['role'] == 'user':
                user_text = conv['content']
            elif conv['role'] == 'assistant':
                assistant_text = conv['content']
        
        return user_text or None, assistant_text or None

class DataAugmentation:
    """Data augmentation utilities for training data."""
    
    @staticmethod
    def add_noise(text: str, noise_prob: float = 0.1) -> str:
        """Add random noise to text."""
        import random
        
        if random.random() > noise_prob:
            return text
        
        words = text.split()
        if len(words) < 2:
            return text
        
        # Randomly shuffle some words
        if random.random() > 0.5:
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        
        # Randomly remove some words
        if random.random() > 0.3:
            words = [w for w in words if random.random() > 0.1]
        
        return ' '.join(words)
    
    @staticmethod
    def paraphrase_simple(text: str) -> str:
        """Simple paraphrasing by replacing common words."""
        replacements = {
            'good': 'excellent',
            'bad': 'poor',
            'big': 'large',
            'small': 'tiny',
            'fast': 'quick',
            'slow': 'gradual'
        }
        
        words = text.split()
        paraphrased = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in replacements:
                # Preserve original capitalization
                replacement = replacements[clean_word]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                paraphrased.append(word.replace(clean_word, replacement))
            else:
                paraphrased.append(word)
        
        return ' '.join(paraphrased)

class DataStatistics:
    """Utility class for analyzing data statistics."""
    
    @staticmethod
    def get_dataset_stats(dataset: Dataset) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            'total_samples': len(dataset),
            'columns': dataset.column_names if hasattr(dataset, 'column_names') else [],
            'avg_length': 0,
            'length_distribution': {},
            'sample_quality': 0
        }
        
        if len(dataset) == 0:
            return stats
        
        # Calculate text lengths
        lengths = []
        for example in dataset:
            text = ""
            if 'text' in example:
                text = example['text']
            elif 'prompt' in example and 'response' in example:
                text = example['prompt'] + " " + example['response']
            
            lengths.append(len(text.strip()))
        
        if lengths:
            stats['avg_length'] = sum(lengths) / len(lengths)
            stats['min_length'] = min(lengths)
            stats['max_length'] = max(lengths)
            
            # Length distribution
            bins = [0, 100, 500, 1000, 2000, 5000, float('inf')]
            for i in range(len(bins) - 1):
                count = sum(1 for length in lengths if bins[i] <= length < bins[i+1])
                stats['length_distribution'][f"{bins[i]}-{bins[i+1]}"] = count
        
        # Quality metrics
        quality_scores = []
        for example in dataset:
            text = ""
            if 'text' in example:
                text = example['text']
            
            if text:
                # Simple quality score based on various factors
                score = DataStatistics._calculate_quality_score(text)
                quality_scores.append(score)
        
        if quality_scores:
            stats['avg_quality'] = sum(quality_scores) / len(quality_scores)
        
        return stats
    
    @staticmethod
    def _calculate_quality_score(text: str) -> float:
        """Calculate a simple quality score for text."""
        if not text:
            return 0.0
        
        score = 1.0
        
        # Length penalty/bonus
        if len(text) < 10:
            score *= 0.5
        elif len(text) > 5000:
            score *= 0.8
        
        # Whitespace penalty
        whitespace_ratio = text.count(' ') / len(text)
        if whitespace_ratio > 0.5:
            score *= 0.8
        
        # Punctuation bonus
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        if punctuation_count > 0:
            score *= 1.1
        
        return min(score, 1.0)
    
    @staticmethod
    def print_dataset_summary(dataset: Dataset) -> None:
        """Print comprehensive dataset summary."""
        stats = DataStatistics.get_dataset_stats(dataset)
        
        print("\n" + "=" * 50)
        print("DATASET SUMMARY")
        print("=" * 50)
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Average Length: {stats['avg_length']:.1f} characters")
        print(f"Min Length: {stats['min_length']}")
        print(f"Max Length: {stats['max_length']}")
        
        if 'avg_quality' in stats:
            print(f"Average Quality Score: {stats['avg_quality']:.2f}")
        
        print(f"Columns: {', '.join(stats['columns'])}")
        
        if stats['length_distribution']:
            print("\nLength Distribution:")
            for range_str, count in stats['length_distribution'].items():
                print(f"  {range_str}: {count:,}")
        
        print("=" * 50 + "\n")

# Export main classes
__all__ = [
    "DataPreprocessingConfig",
    "DataValidator",
    "DataPreprocessor",
    "ChatMLProcessor",
    "DataAugmentation",
    "DataStatistics"
]