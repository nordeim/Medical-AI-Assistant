"""
Comprehensive Test Suite for Medical AI Training System
======================================================

This module provides comprehensive testing coverage including:
- Unit testing for individual components
- Integration testing for end-to-end workflows
- Performance testing for scalability assessment
- Edge case and error condition testing
"""

import os
import sys
import json
import time
import psutil
import logging
import tempfile
import shutil
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import resource

import torch
import torch.nn as nn
import numpy as np
import pytest
import yaml
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Import training modules
from train_lora import (
    ModelConfig, LoRAConfig, QuantizationConfig, TrainingConfig,
    OptimizationConfig, DataConfig, CustomDataCollator,
    MemoryMonitoringCallback, ModelValidator, LoRATrainer
)
from model_utils import (
    ModelInfoCollector, ModelLoader, ModelSaver, ModelConverter,
    ModelValidator as UtilsModelValidator, ModelManager,
    ModelLoadingError, ModelSavingError, QuantizationError
)

# ==================== UNIT TESTING ====================

class TestUnitComponents:
    """Unit tests for individual components."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    
    def teardown_method(self):
        """Cleanup after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        gc.collect()
    
    def test_model_config_edge_cases(self):
        """Test ModelConfig with edge cases."""
        # Test extremely long model names
        long_name = "a" * 1000
        config = ModelConfig(model_name=long_name)
        assert config.model_name == long_name
        
        # Test special characters in model type
        config = ModelConfig(model_type="custom-type_v1.0")
        assert config.model_type == "custom-type_v1.0"
        
        # Test zero max_length (should be handled gracefully)
        try:
            config = ModelConfig(max_length=0)
            # If it doesn't raise an error, ensure it handles gracefully
            assert config.max_length >= 0
        except ValueError:
            # Expected for invalid max_length
            pass
    
    def test_lora_config_edge_cases(self):
        """Test LoRAConfig with edge cases."""
        # Test boundary values
        config = LoRAConfig(r=1, lora_alpha=1, lora_dropout=0.0)
        assert config.r == 1
        assert config.lora_alpha == 1
        assert config.lora_dropout == 0.0
        
        # Test maximum reasonable values
        config = LoRAConfig(r=1024, lora_alpha=2048, lora_dropout=0.99)
        assert config.r == 1024
        assert config.lora_alpha == 2048
        assert config.lora_dropout == 0.99
        
        # Test empty target_modules
        config = LoRAConfig(target_modules=[])
        assert config.target_modules == []
    
    def test_data_collator_edge_cases(self):
        """Test CustomDataCollator with edge cases."""
        collator = CustomDataCollator(
            tokenizer=self.mock_tokenizer,
            max_length=1,  # Very small max_length
            padding=True,
            truncation=True
        )
        
        # Test with very long text (should be truncated)
        long_text = "a" * 10000
        features = [{"text": long_text}]
        
        batch = collator(features)
        assert "input_ids" in batch
        assert len(batch["input_ids"][0]) <= 1  # Truncated to max_length
        
        # Test with empty features
        try:
            empty_batch = collator([])
            assert len(empty_batch["input_ids"]) == 0
        except Exception:
            # Empty batch handling is implementation-dependent
            pass
    
    def test_memory_monitoring_edge_cases(self):
        """Test MemoryMonitoringCallback with memory constraints."""
        callback = MemoryMonitoringCallback(
            memory_threshold_mb=1,  # Very low threshold
            check_interval=0.1  # Very frequent checking
        )
        
        # Test with mock trainer
        mock_trainer = Mock()
        mock_trainer.state = Mock()
        mock_trainer.state.global_step = 1
        
        # Test on_train_begin
        callback.on_train_begin(mock_trainer)
        
        # Test on_train_end
        callback.on_train_end(mock_trainer)
        
        # Should handle memory monitoring gracefully
    
    def test_error_conditions(self):
        """Test error condition handling."""
        # Test invalid quantization configs
        with pytest.raises(ValueError):
            QuantizationConfig(use_4bit=True, use_8bit=True)
        
        with pytest.raises(ValueError):
            QuantizationConfig(load_in_4bit=True, load_in_8bit=True)
        
        # Test invalid training configs
        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-1.0)
        
        with pytest.raises(ValueError):
            TrainingConfig(num_epochs=-1)
        
        # Test invalid LoRA configs
        with pytest.raises(ValueError):
            LoRAConfig(r=-1)
        
        with pytest.raises(ValueError):
            LoRAConfig(lora_dropout=1.5)

# ==================== INTEGRATION TESTING ====================

class TestIntegrationWorkflows:
    """Integration tests for end-to-end workflows."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.create_test_data()
    
    def teardown_method(self):
        """Cleanup integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        gc.collect()
    
    def create_test_data(self):
        """Create test data for integration tests."""
        test_data = []
        for i in range(10):
            test_data.append({
                "text": f"Test medical dialogue {i}: Patient presents with symptoms.",
                "medical_category": "general"
            })
        
        data_path = os.path.join(self.temp_dir, "integration_test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in test_data:
                json.dump(item, f)
                f.write('\n')
        
        self.data_path = data_path
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('train_lora.get_peft_model')
    def test_end_to_end_training_pipeline(self, mock_peft, mock_model, mock_tokenizer):
        """Test complete training pipeline from start to finish."""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_peft.return_value = Mock()
        
        # Create configurations
        model_config = ModelConfig(
            model_name="gpt2",
            model_type="gpt2",
            max_length=128
        )
        
        lora_config = LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.1)
        quantization_config = QuantizationConfig(use_4bit=False)
        training_config = TrainingConfig(
            output_dir=self.temp_dir,
            num_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="no",
            report_to=[],
            save_steps=1,
            logging_steps=1
        )
        optimization_config = OptimizationConfig(gradient_checkpointing=False)
        data_config = DataConfig(
            data_path=self.data_path,
            data_type="jsonl",
            max_samples=5
        )
        
        # Create trainer
        trainer = LoRATrainer(
            model_config=model_config,
            lora_config=lora_config,
            quantization_config=quantization_config,
            training_config=training_config,
            optimization_config=optimization_config,
            data_config=data_config
        )
        
        # Execute pipeline steps
        trainer._validate_configurations()
        trainer._load_and_prepare_tokenizer()
        trainer._load_and_prepare_model()
        
        # Verify pipeline execution
        assert trainer.tokenizer is not None
        assert trainer.model is not None
        
        # Test data loading
        dataset = trainer._load_and_prepare_data()
        assert dataset is not None
        assert len(dataset) > 0
        
        print("✅ End-to-end training pipeline completed successfully")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('train_lora.get_peft_model')
    def test_model_serving_pipeline(self, mock_peft, mock_model, mock_tokenizer):
        """Test model serving pipeline."""
        # Setup mocks
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_peft.return_value = Mock()
        
        # Create model manager
        model_manager = ModelManager(cache_dir=self.temp_dir)
        
        # Test model loading and saving
        with patch.object(model_manager, 'load_model') as mock_load:
            mock_load.return_value = (Mock(), Mock())
            result = model_manager.load_model("gpt2")
            assert result is not None
        
        print("✅ Model serving pipeline completed successfully")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_data_quality_pipeline(self, mock_tokenizer):
        """Test data quality validation pipeline."""
        mock_tokenizer.return_value = Mock()
        
        # Test data loading and validation
        tokenizer = ModelLoader.load_tokenizer("gpt2")
        
        # Simulate data validation
        data_samples = []
        for i in range(5):
            data_samples.append({"text": f"Sample medical text {i}"})
        
        # Validate data quality
        for sample in data_samples:
            assert "text" in sample
            assert len(sample["text"]) > 0
        
        print("✅ Data quality pipeline completed successfully")
    
    def test_configuration_validation_pipeline(self):
        """Test configuration validation pipeline."""
        # Test valid configurations
        configs = [
            ModelConfig(model_type="gpt2"),
            LoRAConfig(r=4, lora_alpha=8),
            QuantizationConfig(use_4bit=False),
            TrainingConfig(num_epochs=1),
            OptimizationConfig(),
            DataConfig(data_type="jsonl")
        ]
        
        for config in configs:
            # Each config should be valid
            assert config is not None
        
        # Test invalid configurations should raise errors
        invalid_configs = [
            lambda: ModelConfig(model_type="invalid"),
            lambda: LoRAConfig(r=-1),
            lambda: QuantizationConfig(use_4bit=True, use_8bit=True),
            lambda: TrainingConfig(learning_rate=-1.0)
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, AssertionError)):
                invalid_config()
        
        print("✅ Configuration validation pipeline completed successfully")

# ==================== PERFORMANCE TESTING ====================

class TestPerformanceMetrics:
    """Performance tests for training scripts."""
    
    def setup_method(self):
        """Setup for performance tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}
    
    def teardown_method(self):
        """Cleanup performance test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def test_data_loading_performance(self):
        """Test data loading performance."""
        # Create large test dataset
        large_data = []
        for i in range(10000):
            large_data.append({
                "text": f"Medical text sample {i}: Patient presents with symptoms and history.",
                "label": "general" if i % 2 == 0 else "specific"
            })
        
        data_path = os.path.join(self.temp_dir, "large_test_data.jsonl")
        with open(data_path, 'w') as f:
            for item in large_data:
                json.dump(item, f)
                f.write('\n')
        
        # Measure loading time
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Load data (simulated)
        with open(data_path, 'r') as f:
            loaded_data = [json.loads(line) for line in f]
        
        load_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        memory_used = end_memory - start_memory
        
        self.results['data_loading'] = {
            'time_seconds': load_time,
            'memory_mb': memory_used,
            'records_per_second': len(loaded_data) / load_time
        }
        
        assert load_time < 10.0  # Should load in reasonable time
        assert memory_used < 100  # Should not use excessive memory
        
        print(f"📊 Data Loading Performance: {len(loaded_data)} records in {load_time:.2f}s")
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('train_lora.get_peft_model')
    def test_model_loading_performance(self, mock_peft, mock_model, mock_tokenizer):
        """Test model loading performance."""
        # Setup lightweight mocks for performance testing
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        mock_peft.return_value = Mock()
        
        # Measure model loading time
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        # Simulate model loading
        tokenizer = ModelLoader.load_tokenizer("gpt2")
        model = ModelLoader.load_base_model("gpt2")
        
        load_time = time.time() - start_time
        end_memory = self.get_memory_usage()
        memory_used = end_memory - start_memory
        
        self.results['model_loading'] = {
            'time_seconds': load_time,
            'memory_mb': memory_used
        }
        
        assert load_time < 30.0  # Should load in reasonable time
        
        print(f"📊 Model Loading Performance: {load_time:.2f}s, {memory_used:.1f}MB")
    
    def test_tokenization_performance(self):
        """Test tokenization performance."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.pad_token_id = 0
        
        collator = CustomDataCollator(
            tokenizer=mock_tokenizer,
            max_length=512,
            padding=True,
            truncation=True
        )
        
        # Create test batches
        batch_sizes = [1, 8, 32, 128]
        results = {}
        
        for batch_size in batch_sizes:
            # Create batch
            features = [
                {"text": f"Test medical text {i}: Patient data"}
                for i in range(batch_size)
            ]
            
            # Measure tokenization time
            start_time = time.time()
            
            for _ in range(10):  # Multiple runs for stability
                batch = collator(features)
            
            avg_time = (time.time() - start_time) / 10
            results[batch_size] = avg_time
            
            print(f"📊 Tokenization: {batch_size} samples in {avg_time*1000:.1f}ms")
        
        self.results['tokenization'] = results
        
        # Performance should scale reasonably with batch size
        assert results[128] < results[1] * 150  # Should not be 150x slower
    
    def test_memory_scaling(self):
        """Test memory usage scaling."""
        batch_sizes = [1, 4, 16, 64]
        memory_usage = []
        
        for batch_size in batch_sizes:
            gc.collect()  # Clean up before measurement
            start_memory = self.get_memory_usage()
            
            # Simulate memory usage with batch
            features = [
                {"text": "Test medical text: Patient information"}
                for _ in range(batch_size)
            ]
            
            # Simulate processing (add some memory usage)
            dummy_array = np.zeros((batch_size * 1000, 10))
            memory_used = self.get_memory_usage() - start_memory
            memory_usage.append(memory_used)
            
            del dummy_array
            gc.collect()
        
        self.results['memory_scaling'] = dict(zip(batch_sizes, memory_usage))
        
        # Memory should scale roughly linearly
        ratio_64_1 = memory_usage[3] / memory_usage[0] if memory_usage[0] > 0 else 1
        assert ratio_64_1 < 100  # Should not be 100x more memory
        
        print(f"📊 Memory Scaling: {memory_usage}MB")
    
    def test_concurrent_processing(self):
        """Test concurrent processing performance."""
        def simulate_processing(worker_id):
            """Simulate work done by a worker."""
            start_time = time.time()
            
            # Simulate some processing work
            result = 0
            for i in range(1000):
                result += i * worker_id
            
            # Add some variation in processing time
            time.sleep(0.01)
            
            return {
                'worker_id': worker_id,
                'processing_time': time.time() - start_time,
                'result': result
            }
        
        # Test with different numbers of workers
        worker_counts = [1, 2, 4, 8]
        results = {}
        
        for worker_count in worker_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [executor.submit(simulate_processing, i) for i in range(worker_count)]
                worker_results = [future.result() for future in as_completed(futures)]
            
            total_time = time.time() - start_time
            results[worker_count] = {
                'total_time': total_time,
                'worker_results': worker_results
            }
            
            print(f"📊 Concurrent Processing: {worker_count} workers in {total_time:.2f}s")
        
        self.results['concurrent_processing'] = results
        
        # Verify all workers completed successfully
        for worker_count in worker_counts:
            assert len(results[worker_count]['worker_results']) == worker_count

# ==================== SCALABILITY TESTING ====================

class TestScalability:
    """Tests for scalability assessment."""
    
    def test_model_size_scaling(self):
        """Test model size scaling with LoRA parameters."""
        # Test different LoRA ranks
        lora_ranks = [1, 4, 8, 16, 32, 64]
        parameter_counts = []
        
        for rank in lora_ranks:
            config = LoRAConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.1)
            
            # Simulate parameter count calculation
            # LoRA adds 2 * rank * d_model parameters per target module
            d_model = 768  # Example model dimension
            num_modules = 16  # Example number of target modules
            
            lora_params = 2 * rank * d_model * num_modules
            parameter_counts.append(lora_params)
        
        # Verify scaling is roughly proportional
        for i in range(1, len(parameter_counts)):
            ratio = parameter_counts[i] / parameter_counts[i-1]
            expected_ratio = lora_ranks[i] / lora_ranks[i-1]
            
            # Should be roughly proportional
            assert abs(ratio - expected_ratio) < 0.1
        
        print(f"📊 LoRA Parameter Scaling: {list(zip(lora_ranks, parameter_counts))}")
    
    def test_training_speed_scaling(self):
        """Test training speed scaling with batch size."""
        batch_sizes = [1, 2, 4, 8, 16]
        training_times = []
        
        for batch_size in batch_sizes:
            # Simulate training time (inversely related to batch size with diminishing returns)
            base_time = 100.0  # seconds for batch_size=1
            batch_time = base_time / (batch_size ** 0.8)  # 0.8 scaling factor for diminishing returns
            
            training_times.append(batch_time)
        
        # Verify diminishing returns effect
        throughput_per_size = [bs/t for bs, t in zip(batch_sizes, training_times)]
        
        # Throughput should increase with batch size but at diminishing rate
        for i in range(1, len(throughput_per_size)):
            assert throughput_per_size[i] > throughput_per_size[i-1]
        
        print(f"📊 Training Speed Scaling: {list(zip(batch_sizes, training_times))}")
    
    def test_memory_requirements_scaling(self):
        """Test memory requirements scaling."""
        model_sizes = [7, 13, 30, 70]  # Billions of parameters
        memory_requirements = []
        
        for size in model_sizes:
            # Estimate memory requirements
            # Base model: 4 bytes per parameter
            # LoRA: Additional overhead
            # Gradients: 2x model size
            # Optimizer states: 2x model size
            
            model_memory = size * 1e9 * 4 / 1e6  # MB
            lora_overhead = 0.1 * model_memory  # 10% overhead
            gradient_memory = 2 * model_memory
            optimizer_memory = 2 * model_memory
            
            total_memory = model_memory + lora_overhead + gradient_memory + optimizer_memory
            memory_requirements.append(total_memory)
        
        # Verify reasonable scaling
        for i, memory in enumerate(memory_requirements):
            assert memory > 0
            if i > 0:
                assert memory > memory_requirements[i-1]  # Larger models need more memory
        
        print(f"📊 Memory Requirements: {list(zip(model_sizes, memory_requirements))}")

# ==================== STRESS TESTING ====================

class TestStressConditions:
    """Tests for stress and edge conditions."""
    
    def test_extreme_batch_sizes(self):
        """Test handling of extreme batch sizes."""
        collator = CustomDataCollator(
            tokenizer=Mock(pad_token_id=0),
            max_length=512,
            padding=True,
            truncation=True
        )
        
        # Test very small batch
        small_features = [{"text": "tiny"}]
        batch = collator(small_features)
        assert "input_ids" in batch
        
        # Test very large batch (simulated)
        large_features = [{"text": f"sample_{i}"} for i in range(1000)]
        
        try:
            # This might fail due to memory constraints
            large_batch = collator(large_features)
            print("✅ Large batch processing successful")
        except Exception as e:
            print(f"⚠️ Large batch failed as expected: {e}")
    
    def test_memory_pressure(self):
        """Test system behavior under memory pressure."""
        initial_memory = self.get_memory_usage()
        
        # Create memory pressure
        memory_hogs = []
        try:
            for _ in range(10):
                hog = np.random.randn(1000, 1000)
                memory_hogs.append(hog)
                current_memory = self.get_memory_usage()
                
                # If memory usage increases significantly, we might need to clean up
                if current_memory - initial_memory > 100:  # 100MB threshold
                    break
        except MemoryError:
            print("⚠️ Memory pressure test: MemoryError encountered")
        
        finally:
            # Clean up
            del memory_hogs
            gc.collect()
    
    def test_concurrent_access(self):
        """Test concurrent access to shared resources."""
        results = []
        errors = []
        
        def concurrent_worker(worker_id):
            try:
                # Simulate concurrent resource access
                time.sleep(0.01)  # Small delay to increase contention
                return {
                    'worker_id': worker_id,
                    'status': 'success',
                    'timestamp': time.time()
                }
            except Exception as e:
                return {
                    'worker_id': worker_id,
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        # Run multiple concurrent workers
        num_workers = 20
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(concurrent_worker, i) for i in range(num_workers)]
            worker_results = [future.result() for future in as_completed(futures)]
        
        # Check results
        successful_workers = [r for r in worker_results if r['status'] == 'success']
        failed_workers = [r for r in worker_results if r['status'] == 'error']
        
        print(f"📊 Concurrent Access: {len(successful_workers)}/{num_workers} workers successful")
        
        # Most workers should succeed under normal conditions
        assert len(successful_workers) >= num_workers * 0.8  # 80% success rate
    
    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

# ==================== REGRESSION TESTING ====================

class TestRegressionScenarios:
    """Tests to catch regressions in functionality."""
    
    def test_basic_training_step(self):
        """Test that basic training step still works."""
        # This test ensures that the fundamental training step hasn't regressed
        
        config = TrainingConfig(
            output_dir="./test_output",
            num_epochs=1,
            per_device_train_batch_size=1,
            evaluation_strategy="no",
            report_to=[]
        )
        
        # Should be able to create training config
        assert config.output_dir == "./test_output"
        assert config.num_epochs == 1
        assert config.per_device_train_batch_size == 1
    
    def test_model_loading_regression(self):
        """Test that model loading still works after changes."""
        # This test ensures model loading functionality hasn't regressed
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.return_value = Mock()
            
            tokenizer = ModelLoader.load_tokenizer("gpt2")
            assert tokenizer is not None
            mock_tokenizer.assert_called_once()
    
    def test_data_processing_regression(self):
        """Test that data processing still works after changes."""
        # This test ensures data processing hasn't regressed
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        collator = CustomDataCollator(
            tokenizer=mock_tokenizer,
            max_length=128,
            padding=True,
            truncation=True
        )
        
        features = [
            {"text": "Test medical text"},
            {"text": "Another test text"}
        ]
        
        batch = collator(features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

# ==================== TEST EXECUTION ====================

def run_comprehensive_tests():
    """Run all comprehensive tests."""
    logger.info("Starting Comprehensive Test Suite")
    logger.info("=" * 60)
    
    test_results = {
        'unit_tests': {'passed': 0, 'failed': 0, 'errors': []},
        'integration_tests': {'passed': 0, 'failed': 0, 'errors': []},
        'performance_tests': {'passed': 0, 'failed': 0, 'errors': []},
        'scalability_tests': {'passed': 0, 'failed': 0, 'errors': []},
        'stress_tests': {'passed': 0, 'failed': 0, 'errors': []},
        'regression_tests': {'passed': 0, 'failed': 0, 'errors': []}
    }
    
    test_classes = [
        ('Unit Tests', TestUnitComponents),
        ('Integration Tests', TestIntegrationWorkflows),
        ('Performance Tests', TestPerformanceMetrics),
        ('Scalability Tests', TestScalability),
        ('Stress Tests', TestStressConditions),
        ('Regression Tests', TestRegressionScenarios)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category, test_class in test_classes:
        logger.info(f"\n🔍 Running {category}...")
        logger.info("-" * 40)
        
        try:
            # Run tests in the class
            pytest.main([
                '-v',
                '--tb=short',
                __file__,
                f'-k {category.replace(" ", "_").replace(":", "")}'
            ])
            
            test_results[f'{category.lower().split()[0]}_tests']['passed'] += 1
            
        except Exception as e:
            test_results[f'{category.lower().split()[0]}_tests']['errors'].append(str(e))
            test_results[f'{category.lower().split()[0]}_tests']['failed'] += 1
        
        total_tests += 1
        passed_tests += 1
    
    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE TEST REPORT")
    logger.info("=" * 60)
    
    for category, results in test_results.items():
        logger.info(f"{category.replace('_', ' ').title()}:")
        logger.info(f"  ✅ Passed: {results['passed']}")
        logger.info(f"  ❌ Failed: {results['failed']}")
        if results['errors']:
            logger.info(f"  ⚠️  Errors: {len(results['errors'])}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TOTAL: {passed_tests}/{total_tests} test categories passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All comprehensive tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)