"""
Example usage of the optimization framework for Phase 6.

This script demonstrates how to use the quantization and optimization system
for medical AI models with accuracy preservation requirements.
"""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any

from config import OptimizationConfig, OptimizationLevel, QuantizationType
from quantization import QuantizationManager, QuantizationConfig
from memory_optimization import MemoryOptimizer, MemoryConfig
from device_optimization import DeviceManager, DeviceConfig, InferenceMode
from batch_optimization import BatchProcessor, BatchConfig
from model_reduction import ModelReducer, ReductionConfig
from validation import QuantizationValidator, ValidationConfig
from utils import SystemProfiler, OptimizationUtils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalModelExample(nn.Module):
    """Example medical model for demonstration purposes."""
    
    def __init__(self, input_size=768, hidden_size=1024, num_classes=10):
        super(MedicalModelExample, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.diagnosis_head = nn.Linear(hidden_size, 5)  # 5 medical conditions
        
    def forward(self, x):
        features = self.encoder(x)
        classification = self.classifier(features)
        diagnosis = self.diagnosis_head(features)
        return classification, diagnosis


def create_sample_data():
    """Create sample data for testing."""
    # Training data
    train_data = []
    for i in range(1000):
        x = torch.randn(768)  # Input features
        y_class = torch.randint(0, 10, (1,))  # Classification labels
        y_diagnosis = torch.randint(0, 5, (1,))  # Diagnosis labels
        train_data.append((x, y_class, y_diagnosis))
    
    # Test data
    test_data = []
    for i in range(100):
        x = torch.randn(768)
        y_class = torch.randint(0, 10, (1,))
        y_diagnosis = torch.randint(0, 5, (1,))
        test_data.append((x, y_class, y_diagnosis))
    
    return train_data, test_data


def simple_accuracy_validator(model, test_data=None):
    """Simple accuracy validator for demonstration."""
    if test_data is None:
        return 0.95  # Placeholder accuracy
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_data[:50]:  # Limit for demo
            if len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
                outputs, _ = model(inputs.unsqueeze(0))
                
                if outputs.shape[-1] > 1:  # Multi-class
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
    
    accuracy = correct / total if total > 0 else 0.95
    return accuracy


def run_complete_optimization_example():
    """Run a complete optimization example."""
    logger.info("Starting complete optimization example for medical model")
    
    # Step 1: Create system profile and check compatibility
    logger.info("Step 1: Analyzing system and creating profile...")
    system_profiler = SystemProfiler()
    
    compatibility = system_profiler.check_optimization_compatibility()
    recommendations = system_profiler.get_optimization_recommendations()
    
    logger.info(f"System compatibility: {compatibility}")
    logger.info(f"Optimization recommendations: {recommendations}")
    
    # Step 2: Create and analyze model
    logger.info("Step 2: Creating and analyzing medical model...")
    model = MedicalModelExample()
    
    model_analysis = ModelReducer(ReductionConfig()).analyze_model(model)
    logger.info(f"Model analysis: {model_analysis}")
    
    # Step 3: Setup optimization configuration
    logger.info("Step 3: Setting up optimization configuration...")
    
    # Create medical-focused configuration
    config = OptimizationConfig(
        level=OptimizationLevel.MEDICAL_CRITICAL,
        preserve_medical_accuracy=True,
        auto_adjust_for_medical=True
    )
    
    # Adjust for medical requirements
    config.auto_adjust_for_medical_requirements()
    
    logger.info(f"Optimized configuration: {config}")
    
    # Step 4: Initialize optimization components
    logger.info("Step 4: Initializing optimization components...")
    
    # Quantization manager
    quant_config = QuantizationConfig(
        quantization_type=QuantizationType.INT8,  # Conservative for medical
        use_bnb=True,
        load_in_8bit=False
    )
    quant_manager = QuantizationManager(quant_config)
    
    # Memory optimizer
    memory_config = MemoryConfig(
        enable_gradient_checkpointing=True,
        enable_cpu_offload=True
    )
    memory_optimizer = MemoryOptimizer(memory_config)
    
    # Device manager
    device_config = DeviceConfig(
        preferred_device="auto",
        device_memory_fraction=0.8
    )
    device_manager = DeviceManager(device_config)
    
    # Batch processor
    batch_config = BatchConfig(
        max_batch_size=16,  # Smaller for medical accuracy
        batch_timeout=0.2,
        enable_dynamic_batching=True
    )
    
    # Model reducer
    reduction_config = ReductionConfig(
        prune_ratio=0.1,  # Conservative pruning for medical models
        reduction_type="pruning"
    )
    model_reducer = ModelReducer(reduction_config)
    
    # Validation system
    validation_config = ValidationConfig(
        accuracy_threshold=0.98,  # High threshold for medical
        enable_validation=True
    )
    validator = QuantizationValidator(validation_config)
    
    # Step 5: Apply optimizations step by step
    logger.info("Step 5: Applying optimizations...")
    
    # 5.1: Memory optimization
    logger.info("5.1: Applying memory optimization...")
    checkpoint_state = memory_optimizer.enable_gradient_checkpointing(model)
    logger.info(f"Gradient checkpointing: {checkpoint_state}")
    
    # 5.2: Device optimization
    logger.info("5.2: Setting up device optimization...")
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    optimal_device = device_manager.select_optimal_device(
        model_size_gb=model_size_mb / 1024,
        inference_mode=InferenceMode.ACCURACY_FOCUSED
    )
    logger.info(f"Selected device: {optimal_device.name}")
    
    device_optimizations = device_manager.optimize_for_inference(optimal_device, model)
    logger.info(f"Device optimizations: {device_optimizations}")
    
    # 5.3: Quantization
    logger.info("5.3: Applying quantization...")
    quant_result = quant_manager.quantize_model(model)
    logger.info(f"Quantization result: {quant_result}")
    
    # 5.4: Model reduction (conservative for medical)
    logger.info("5.4: Applying model reduction...")
    if model_size_mb > 50:  # Only reduce large models
        pruning_result = model_reducer.apply_pruning(
            model,
            pruning_ratio=0.1,  # Conservative 10% pruning
            method="magnitude",
            validation_function=lambda m: simple_accuracy_validator(m)
        )
        logger.info(f"Pruning result: {pruning_result}")
    else:
        logger.info("Model too small for reduction")
    
    # Step 6: Validation
    logger.info("Step 6: Validating optimized model...")
    
    # Create test data
    train_data, test_data = create_sample_data()
    
    # Validate quantized model
    validation_report = validator.validate_quantization(
        original_model=MedicalModelExample(),  # Fresh original model
        quantized_model=model,
        test_dataset=test_data[:20],  # Small test set for demo
        custom_metrics=["accuracy", "medical_accuracy", "safety_score"]
    )
    
    logger.info(f"Validation report: {validation_report.to_dict()}")
    
    # Step 7: Batch processing setup
    logger.info("Step 7: Setting up batch processing...")
    
    def inference_function(batch_inputs):
        """Example inference function for batch processing."""
        model.eval()
        results = []
        
        with torch.no_grad():
            for inputs in batch_inputs:
                inputs = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                class_out, diag_out = model(inputs)
                results.append({
                    'classification': class_out.squeeze().tolist(),
                    'diagnosis': diag_out.squeeze().tolist()
                })
        
        return results
    
    batch_processor = BatchProcessor(batch_config, inference_function)
    
    # Submit some test requests
    logger.info("Step 8: Testing batch processing...")
    batch_processor.start()
    
    from batch_optimization import BatchRequest
    import uuid
    
    test_requests = []
    for i in range(5):
        request = BatchRequest(
            request_id=str(uuid.uuid4()),
            input_data=torch.randn(768),
            priority=1
        )
        test_requests.append(request)
    
    # Submit batch
    request_ids = batch_processor.submit_batch(test_requests)
    logger.info(f"Submitted batch with {len(request_ids)} requests")
    
    # Wait for results
    results = []
    for request_id in request_ids:
        result = batch_processor.get_result(request_id, timeout=10.0)
        if result:
            results.append(result)
    
    logger.info(f"Received {len(results)} results")
    
    # Step 9: Performance analysis
    logger.info("Step 9: Performance analysis...")
    
    batch_report = batch_processor.get_performance_report()
    logger.info(f"Batch processing report: {batch_report}")
    
    # System benchmark
    benchmark_results = system_profiler.benchmark_system_performance(model_size_mb)
    logger.info(f"System benchmark: {benchmark_results}")
    
    # Step 10: Generate final report
    logger.info("Step 10: Generating optimization report...")
    
    final_report = {
        "optimization_summary": {
            "original_model_size_mb": model_size_mb,
            "quantization_applied": quant_result.success,
            "memory_optimization": checkpoint_state.enabled,
            "device_optimization": optimal_device.name,
            "model_reduction": "applied" if model_size_mb > 50 else "skipped",
            "batch_processing": len(results) > 0
        },
        "validation_results": {
            "overall_score": validation_report.overall_score,
            "passing_tests": validation_report.passing_tests,
            "failing_tests": validation_report.failing_tests,
            "medical_compliant": validation_report.medical_compliance.get("overall_compliant", False)
        },
        "performance_metrics": {
            "batch_throughput": batch_report.get("metrics", {}).get("throughput_requests_per_second", 0),
            "system_benchmark_score": benchmark_results.get("overall_score", 0),
            "compression_achieved": quant_result.compression_ratio if quant_result.success else 1.0
        },
        "recommendations": validation_report.recommendations
    }
    
    logger.info(f"Final optimization report: {final_report}")
    
    # Cleanup
    batch_processor.stop()
    memory_optimizer.cleanup()
    device_manager.cleanup()
    
    logger.info("Complete optimization example finished successfully!")
    
    return final_report


def run_quick_optimization_example():
    """Run a quick optimization example for testing."""
    logger.info("Starting quick optimization example...")
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Quick optimization
    config = OptimizationConfig(level=OptimizationLevel.BALANCED)
    
    # Quantize model
    quant_manager = QuantizationManager(QuantizationConfig())
    quant_result = quant_manager.quantize_model(model)
    
    logger.info(f"Quick quantization result: {quant_result}")
    
    # Validate
    validator = QuantizationValidator(ValidationConfig())
    validation_report = validator.validate_quantization(
        original_model=nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ),
        quantized_model=model
    )
    
    logger.info(f"Quick validation result: {validation_report.overall_score}")
    
    return {
        "quantization_success": quant_result.success,
        "compression_ratio": quant_result.compression_ratio,
        "validation_score": validation_report.overall_score
    }


def main():
    """Main function to run examples."""
    print("=" * 80)
    print("Medical AI Assistant - Optimization Framework Example")
    print("=" * 80)
    
    try:
        # Run quick example first
        print("\n1. Running Quick Example...")
        quick_result = run_quick_optimization_example()
        print(f"Quick example result: {quick_result}")
        
        # Run complete example
        print("\n2. Running Complete Example...")
        complete_result = run_complete_optimization_example()
        print(f"Complete example result: {complete_result}")
        
        print("\n" + "=" * 80)
        print("All optimization examples completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"Example failed with error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)