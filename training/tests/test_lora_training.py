"""
Unit Tests for LoRA Training System
Comprehensive test suite for all core functionality.
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import torch
import torch.nn as nn
import numpy as np
import yaml

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

from train_lora import (
    ModelConfig,
    LoRAConfig,
    QuantizationConfig,
    TrainingConfig,
    OptimizationConfig,
    DataConfig,
    CustomDataCollator,
    MemoryMonitoringCallback,
    ModelValidator,
    LoRATrainer
)
from model_utils import (
    ModelInfoCollector,
    ModelLoader,
    ModelSaver,
    ModelConverter,
    ModelValidator as UtilsModelValidator,
    ModelManager,
    ModelLoadingError,
    ModelSavingError,
    QuantizationError
)

class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""
    
    def test_model_config_creation(self):
        """Test creation of ModelConfig."""
        config = ModelConfig(
            model_name="test-model",
            model_type="llama",
            max_length=1024
        )
        
        self.assertEqual(config.model_name, "test-model")
        self.assertEqual(config.model_type, "llama")
        self.assertEqual(config.max_length, 1024)
    
    def test_model_config_defaults(self):
        """Test ModelConfig defaults."""
        config = ModelConfig()
        
        self.assertEqual(config.model_name, "meta-llama/Llama-2-7b-hf")
        self.assertEqual(config.model_type, "llama")
        self.assertEqual(config.trust_remote_code, False)
        self.assertEqual(config.use_auth_token, None)
    
    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Invalid model type should raise error
        with self.assertRaises(ValueError):
            config = ModelConfig(model_type="invalid_type")

class TestLoRAConfig(unittest.TestCase):
    """Test LoRAConfig dataclass."""
    
    def test_lora_config_creation(self):
        """Test creation of LoRAConfig."""
        config = LoRAConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        
        self.assertEqual(config.r, 8)
        self.assertEqual(config.lora_alpha, 16)
        self.assertEqual(config.lora_dropout, 0.1)
        self.assertEqual(config.target_modules, ["q_proj", "v_proj"])
    
    def test_lora_config_validation(self):
        """Test LoRAConfig validation."""
        # Invalid rank
        with self.assertRaises(ValueError):
            config = LoRAConfig(r=-1)
        
        # Invalid alpha
        with self.assertRaises(ValueError):
            config = LoRAConfig(lora_alpha=0)
        
        # Invalid dropout
        with self.assertRaises(ValueError):
            config = LoRAConfig(lora_dropout=1.5)
        
        # Invalid bias value
        with self.assertRaises(ValueError):
            config = LoRAConfig(bias="invalid_bias")

class TestQuantizationConfig(unittest.TestCase):
    """Test QuantizationConfig dataclass."""
    
    def test_quantization_config_creation(self):
        """Test creation of QuantizationConfig."""
        config = QuantizationConfig(
            use_4bit=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.assertEqual(config.use_4bit, True)
        self.assertEqual(config.bnb_4bit_quant_type, "nf4")
    
    def test_quantization_config_validation(self):
        """Test QuantizationConfig validation."""
        # Cannot use both 4-bit and 8-bit
        with self.assertRaises(ValueError):
            config = QuantizationConfig(use_4bit=True, use_8bit=True)
        
        # Cannot load both 4-bit and 8-bit
        with self.assertRaises(ValueError):
            config = QuantizationConfig(load_in_4bit=True, load_in_8bit=True)

class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig dataclass."""
    
    def test_training_config_creation(self):
        """Test creation of TrainingConfig."""
        config = TrainingConfig(
            output_dir="./test_output",
            num_epochs=5,
            learning_rate=1e-3
        )
        
        self.assertEqual(config.output_dir, "./test_output")
        self.assertEqual(config.num_epochs, 5)
        self.assertEqual(config.learning_rate, 1e-3)
    
    def test_training_config_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()
        
        self.assertEqual(config.output_dir, "./output")
        self.assertEqual(config.num_epochs, 3)
        self.assertEqual(config.fp16, False)
        self.assertEqual(config.bf16, False)

class TestCustomDataCollator(unittest.TestCase):
    """Test CustomDataCollator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        self.collator = CustomDataCollator(
            tokenizer=self.mock_tokenizer,
            max_length=100,
            padding=True,
            truncation=True
        )
    
    def test_collator_creation(self):
        """Test DataCollator creation."""
        self.assertIsNotNone(self.collator)
        self.assertEqual(self.collator.max_length, 100)
        self.assertEqual(self.collator.padding, True)
    
    def test_collator_call_with_input_ids(self):
        """Test DataCollator with input_ids."""
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1]
            },
            {
                "input_ids": [1, 2, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1]
            }
        ]
        
        batch = self.collator(features)
        
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)
        
        # Check batch dimensions
        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(batch["input_ids"].shape[0], 2)
    
    def test_collator_call_with_raw_text(self):
        """Test DataCollator with raw text input."""
        features = [{"text": "Hello world"}, {"text": "This is a test"}]
        
        batch = self.collator(features)
        
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("labels", batch)
    
    def test_collator_labels_handling(self):
        """Test DataCollator labels handling."""
        features = [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [2, 3, 4]
            }
        ]
        
        batch = self.collator(features)
        
        self.assertEqual(batch["labels"].tolist()[0], [2, 3, 4])

class TestModelValidator(unittest.TestCase):
    """Test ModelValidator class."""
    
    def test_validate_model_config(self):
        """Test ModelConfig validation."""
        # Valid config should not raise
        valid_config = ModelConfig(model_type="llama")
        ModelValidator.validate_model_config(valid_config)
        
        # Invalid config should raise
        with self.assertRaises(ValueError):
            invalid_config = ModelConfig(model_type="invalid_type")
            ModelValidator.validate_model_config(invalid_config)
    
    def test_validate_lora_config(self):
        """Test LoRAConfig validation."""
        # Valid config should not raise
        valid_config = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        ModelValidator.validate_lora_config(valid_config)
        
        # Invalid configs should raise
        with self.assertRaises(ValueError):
            invalid_r = LoRAConfig(r=-1)
            ModelValidator.validate_lora_config(invalid_r)
        
        with self.assertRaises(ValueError):
            invalid_dropout = LoRAConfig(lora_dropout=2.0)
            ModelValidator.validate_lora_config(invalid_dropout)
    
    def test_validate_quantization_config(self):
        """Test QuantizationConfig validation."""
        # Valid config should not raise
        valid_config = QuantizationConfig(use_4bit=True)
        ModelValidator.validate_quantization_config(valid_config)
        
        # Invalid config should raise
        with self.assertRaises(ValueError):
            invalid_config = QuantizationConfig(use_4bit=True, use_8bit=True)
            ModelValidator.validate_quantization_config(invalid_config)

class TestLoRATrainer(unittest.TestCase):
    """Test LoRATrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        self.model_config = ModelConfig(
            model_name="gpt2",
            model_type="gpt2",
            max_length=128
        )
        
        self.lora_config = LoRAConfig(r=4, lora_alpha=8)
        
        self.quantization_config = QuantizationConfig()
        
        self.training_config = TrainingConfig(
            output_dir=self.temp_dir,
            num_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            evaluation_strategy="no",
            report_to=[]
        )
        
        self.optimization_config = OptimizationConfig(
            gradient_checkpointing=False
        )
        
        self.data_config = DataConfig(
            data_path="tests/data/test_data.jsonl",
            data_type="jsonl",
            max_samples=10
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_trainer_creation(self):
        """Test LoRATrainer creation."""
        trainer = LoRATrainer(
            model_config=self.model_config,
            lora_config=self.lora_config,
            quantization_config=self.quantization_config,
            training_config=self.training_config,
            optimization_config=self.optimization_config,
            data_config=self.data_config
        )
        
        self.assertIsNotNone(trainer)
        self.assertEqual(trainer.model_config.model_name, "gpt2")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_tokenizer(self, mock_model, mock_tokenizer):
        """Test tokenizer loading."""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        trainer = LoRATrainer(
            model_config=self.model_config,
            lora_config=self.lora_config,
            quantization_config=self.quantization_config,
            training_config=self.training_config,
            optimization_config=self.optimization_config,
            data_config=self.data_config
        )
        
        trainer._load_and_prepare_tokenizer()
        
        self.assertIsNotNone(trainer.tokenizer)
        mock_tokenizer.assert_called_once()
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('train_lora.get_peft_model')
    def test_model_loading(self, mock_get_peft, mock_model):
        """Test model loading."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_peft_model = Mock()
        mock_get_peft.return_value = mock_peft_model
        
        trainer = LoRATrainer(
            model_config=self.model_config,
            lora_config=self.lora_config,
            quantization_config=self.quantization_config,
            training_config=self.training_config,
            optimization_config=self.optimization_config,
            data_config=self.data_config
        )
        
        trainer.tokenizer = Mock()  # Mock tokenizer
        trainer._load_and_prepare_model()
        
        self.assertIsNotNone(trainer.model)
        mock_get_peft.assert_called_once()

class TestModelInfoCollector(unittest.TestCase):
    """Test ModelInfoCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
    
    def test_get_model_info(self):
        """Test model info collection."""
        info = ModelInfoCollector.get_model_info(self.mock_model)
        
        self.assertIsNotNone(info.model_name)
        self.assertIsNotNone(info.model_type)
        self.assertGreater(info.num_parameters, 0)
        self.assertGreater(info.model_size_mb, 0)
    
    def test_print_model_summary(self):
        """Test model summary printing."""
        # Should not raise any exceptions
        ModelInfoCollector.print_model_summary(self.mock_model)
    
    def test_get_quantization_info(self):
        """Test quantization info extraction."""
        # Model without quantization
        info = ModelInfoCollector._get_quantization_info(self.mock_model)
        self.assertIsNone(info)
    
    def test_get_lora_config(self):
        """Test LoRA config extraction."""
        # Model without LoRA
        info = ModelInfoCollector._get_lora_config(self.mock_model)
        self.assertIsNone(info)

class TestModelLoader(unittest.TestCase):
    """Test ModelLoader class."""
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_load_tokenizer(self, mock_tokenizer):
        """Test tokenizer loading."""
        mock_tokenizer.return_value = Mock()
        
        tokenizer = ModelLoader.load_tokenizer("gpt2")
        
        self.assertIsNotNone(tokenizer)
        mock_tokenizer.assert_called_once()
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_load_base_model(self, mock_model):
        """Test base model loading."""
        mock_model.return_value = Mock()
        
        model = ModelLoader.load_base_model("gpt2")
        
        self.assertIsNotNone(model)
        mock_model.assert_called_once()
    
    def test_load_base_model_error(self):
        """Test base model loading error handling."""
        with patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("Test error")):
            with self.assertRaises(ModelLoadingError):
                ModelLoader.load_base_model("invalid-model")

class TestModelSaver(unittest.TestCase):
    """Test ModelSaver class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Mock model save method
        self.mock_model.save_pretrained = Mock()
        self.mock_tokenizer.save_pretrained = Mock()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_model(self):
        """Test model saving."""
        ModelSaver.save_model(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            save_directory=self.temp_dir
        )
        
        self.mock_model.save_pretrained.assert_called_once()
        self.mock_tokenizer.save_pretrained.assert_called_once()
        
        # Check if directory was created
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_save_model_error(self):
        """Test model saving error handling."""
        # Mock model without save_pretrained method
        bad_model = Mock(spec=[])
        
        with self.assertRaises(ModelSavingError):
            ModelSaver.save_model(
                model=bad_model,
                tokenizer=self.mock_tokenizer,
                save_directory=self.temp_dir
            )
    
    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        ModelSaver.save_checkpoint(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
            checkpoint_dir=self.temp_dir,
            epoch=1,
            step=100,
            metrics={"loss": 0.5}
        )
        
        # Check if checkpoint files were created
        checkpoint_info_path = os.path.join(self.temp_dir, "checkpoint_info.json")
        self.assertTrue(os.path.exists(checkpoint_info_path))
        
        # Check checkpoint info content
        with open(checkpoint_info_path, 'r') as f:
            checkpoint_info = json.load(f)
        
        self.assertEqual(checkpoint_info["epoch"], 1)
        self.assertEqual(checkpoint_info["step"], 100)
        self.assertEqual(checkpoint_info["metrics"]["loss"], 0.5)

class TestModelConverter(unittest.TestCase):
    """Test ModelConverter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=['bnb_quantization_config'])
    
    @patch('training.utils.model_utils.BNB_AVAILABLE', False)
    def test_convert_to_8bit_error(self):
        """Test 8-bit conversion error when BNB not available."""
        with self.assertRaises(QuantizationError):
            ModelConverter.convert_to_8bit(self.mock_model)
    
    @patch('training.utils.model_utils.BNB_AVAILABLE', False)
    def test_convert_to_4bit_error(self):
        """Test 4-bit conversion error when BNB not available."""
        with self.assertRaises(QuantizationError):
            ModelConverter.convert_to_4bit(self.mock_model)
    
    def test_merge_and_unload(self):
        """Test LoRA merging and unloading."""
        # Create a mock PEFT model
        mock_peft_model = Mock()
        mock_peft_model.merge_and_unload.return_value = Mock()
        
        merged_model = ModelConverter.merge_and_unload(mock_peft_model)
        
        mock_peft_model.merge_and_unload.assert_called_once()
        self.assertIsNotNone(merged_model)

class TestUtilsModelValidator(unittest.TestCase):
    """Test UtilsModelValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_model = nn.Sequential(
            nn.Linear(10, 5)
        )
    
    def test_validate_model_integrity(self):
        """Test model integrity validation."""
        # Valid model should pass
        result = UtilsModelValidator.validate_model_integrity(self.simple_model)
        self.assertTrue(result)
    
    def test_model_with_nan(self):
        """Test model with NaN values."""
        # Create model with NaN parameters
        model_with_nan = nn.Sequential(
            nn.Linear(10, 5)
        )
        model_with_nan[0].weight.data[0, 0] = float('nan')
        
        result = UtilsModelValidator.validate_model_integrity(model_with_nan)
        self.assertFalse(result)
    
    def test_check_gpu_memory(self):
        """Test GPU memory checking."""
        memory_info = UtilsModelValidator.check_gpu_memory()
        
        self.assertIsInstance(memory_info, dict)
        
        if torch.cuda.is_available():
            # Should have device information
            self.assertGreater(len(memory_info), 0)
        else:
            # Should be empty if no GPU
            self.assertEqual(len(memory_info), 0)
    
    def test_get_model_architecture_summary(self):
        """Test model architecture summary."""
        summary = UtilsModelValidator.get_model_architecture_summary(self.simple_model)
        
        self.assertIsInstance(summary, dict)
        self.assertIn("model_type", summary)
        self.assertIn("total_parameters", summary)
        self.assertIn("trainable_parameters", summary)
        self.assertGreater(summary["total_parameters"], 0)

class TestModelManager(unittest.TestCase):
    """Test ModelManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manager_creation(self):
        """Test ModelManager creation."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.cache_dir, self.temp_dir)

class TestConfigurationLoading(unittest.TestCase):
    """Test configuration loading functionality."""
    
    def test_load_config_from_yaml(self):
        """Test loading YAML configuration."""
        from train_lora import load_config_from_yaml
        
        config_content = """
        model:
          model_name: "test-model"
          model_type: "llama"
        training:
          num_epochs: 5
          learning_rate: 1e-3
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            f.flush()
            
            config = load_config_from_yaml(f.name)
            
            self.assertIn("model", config)
            self.assertIn("training", config)
            self.assertEqual(config["model"]["model_name"], "test-model")
            self.assertEqual(config["training"]["num_epochs"], 5)
            
            os.unlink(f.name)
    
    def test_load_config_from_json(self):
        """Test loading JSON configuration."""
        config_content = {
            "model": {
                "model_name": "test-model",
                "model_type": "llama"
            },
            "training": {
                "num_epochs": 5,
                "learning_rate": 1e-3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_content, f)
            f.flush()
            
            with open(f.name, 'r') as json_f:
                loaded_config = json.load(json_f)
            
            self.assertEqual(loaded_config["model"]["model_name"], "test-model")
            self.assertEqual(loaded_config["training"]["num_epochs"], 5)
            
            os.unlink(f.name)

class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        test_data = [
            {"text": "This is a test sentence."},
            {"text": "Another test sentence for training."},
            {"text": "Third test sentence."}
        ]
        
        with open(os.path.join(self.temp_dir, "test_data.jsonl"), 'w') as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_complete_training_pipeline(self, mock_model, mock_tokenizer):
        """Test the complete training pipeline."""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Create trainer
        model_config = ModelConfig(
            model_name="gpt2",
            model_type="gpt2",
            max_length=128
        )
        
        lora_config = LoRAConfig(r=4, lora_alpha=8)
        quantization_config = QuantizationConfig()
        training_config = TrainingConfig(
            output_dir=self.temp_dir,
            num_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            evaluation_strategy="no",
            report_to=[]
        )
        optimization_config = OptimizationConfig(gradient_checkpointing=False)
        data_config = DataConfig(
            data_path=os.path.join(self.temp_dir, "test_data.jsonl"),
            data_type="jsonl",
            max_samples=5
        )
        
        trainer = LoRATrainer(
            model_config=model_config,
            lora_config=lora_config,
            quantization_config=quantization_config,
            training_config=training_config,
            optimization_config=optimization_config,
            data_config=data_config
        )
        
        # Test individual components
        trainer._validate_configurations()
        trainer._load_and_prepare_tokenizer()
        trainer._load_and_prepare_model()
        
        self.assertIsNotNone(trainer.tokenizer)
        self.assertIsNotNone(trainer.model)

def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelConfig,
        TestLoRAConfig,
        TestQuantizationConfig,
        TestTrainingConfig,
        TestCustomDataCollator,
        TestModelValidator,
        TestLoRATrainer,
        TestModelInfoCollector,
        TestModelLoader,
        TestModelSaver,
        TestModelConverter,
        TestUtilsModelValidator,
        TestModelManager,
        TestConfigurationLoading,
        IntegrationTests
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running LoRA Training System Tests")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        print("=" * 50)
        sys.exit(0)
    else:
        print("\n" + "=" * 50)
        print("Some tests failed!")
        print("=" * 50)
        sys.exit(1)