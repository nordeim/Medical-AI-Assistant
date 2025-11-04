#!/usr/bin/env python3
"""
Test script for the enhanced data augmentation and preprocessing pipeline

This script tests the key components to ensure they're working correctly.
"""

import sys
import json
from pathlib import Path

# Add the training directory to the path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported successfully"""
    print("Testing module imports...")
    
    try:
        # Test data augmentation imports
        from utils.data_augmentation import DataAugmentor, AugmentationConfig, apply_augmentation_pipeline
        print("✅ Data augmentation imports successful")
        
        # Test preprocessing imports
        from utils.preprocessing_pipeline import PreprocessingPipeline, PreprocessingConfig, create_preprocessing_pipeline
        print("✅ Preprocessing pipeline imports successful")
        
        # Test quality assessment imports
        from utils.data_quality_assessment import DataQualityAssessment, QualityMetrics, assess_medical_conversation_quality
        print("✅ Quality assessment imports successful")
        
        # Test optimization imports
        from utils.augmentation_optimizer import OptimizationConfig, optimize_augmentation_strategy
        print("✅ Optimization imports successful")
        
        # Test __init__ imports - only test specific classes that should be available
        from utils import create_preprocessing_pipeline
        print("✅ Main package imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test data augmentation
        config = AugmentationConfig(
            synonym_probability=0.3,
            max_augmentations=2,
            preserve_medical_terms=True
        )
        augmentor = DataAugmentor(config)
        print("✅ DataAugmentor initialization successful")
        
        # Test preprocessing
        preprocessor_config = PreprocessingConfig(
            batch_size=10,
            enable_streaming=True
        )
        pipeline = PreprocessingPipeline(preprocessor_config)
        print("✅ PreprocessingPipeline initialization successful")
        
        # Test quality assessment
        assessor = DataQualityAssessment()
        print("✅ DataQualityAssessment initialization successful")
        
        # Test optimization
        opt_config = OptimizationConfig(
            algorithm="random",
            population_size=3,
            generations=2
        )
        print("✅ OptimizationConfig initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        return False

def test_sample_data_processing():
    """Test processing of sample medical conversation data"""
    print("\nTesting sample data processing...")
    
    # Create sample data
    sample_conversations = [
        {
            "conversation": [
                {"speaker": "patient", "text": "I have been experiencing chest pain for the past two days"},
                {"speaker": "ai", "text": "I understand you're having chest pain. Can you describe the pain - is it sharp, dull, or pressure-like?"},
                {"speaker": "patient", "text": "It's a sharp, stabbing pain that gets worse when I breathe deeply"}
            ]
        },
        {
            "conversation": [
                {"speaker": "patient", "text": "I've had a headache since this morning"},
                {"speaker": "ai", "text": "I'm sorry to hear about your headache. On a scale of 1 to 10, how would you rate the pain?"},
                {"speaker": "patient", "text": "I'd say it's about a 7 out of 10"}
            ]
        }
    ]
    
    try:
        # Test data augmentation
        config = AugmentationConfig(
            synonym_probability=0.5,
            max_augmentations=1,
            preserve_medical_terms=True
        )
        augmentor = DataAugmentor(config)
        
        # Augment first conversation
        conversation = sample_conversations[0]["conversation"]
        augmented = augmentor.augment_conversation(conversation)
        print(f"✅ Data augmentation successful - generated {len(augmented)} variations")
        
        # Test quality assessment
        assessor = DataQualityAssessment()
        quality_metrics = assessor.assess_data_quality([conversation])
        print(f"✅ Quality assessment successful - overall score: {quality_metrics.overall_score:.3f}")
        
        # Test preprocessing
        preprocessor_config = PreprocessingConfig(
            min_conversation_length=2,
            max_conversation_length=10
        )
        pipeline = PreprocessingPipeline(preprocessor_config)
        processed = pipeline.preprocess_dataset(sample_conversations)
        print(f"✅ Preprocessing successful - processed {len(processed['conversations'])} conversations")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data processing test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_preprocessing_script():
    """Test the preprocessing CLI script"""
    print("\nTesting preprocessing script...")
    
    try:
        # Test that the script can be imported and executed
        script_path = Path(__file__).parent / "scripts" / "preprocess_data.py"
        
        if script_path.exists():
            print(f"✅ Preprocessing script found at {script_path}")
            
            # Test script help (without actually running it)
            import subprocess
            result = subprocess.run([
                sys.executable, str(script_path), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Preprocessing script help command successful")
                return True
            else:
                print(f"❌ Preprocessing script help failed: {result.stderr}")
                return False
        else:
            print(f"❌ Preprocessing script not found at {script_path}")
            return False
            
    except Exception as e:
        print(f"❌ Preprocessing script test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests and provide a summary"""
    print("="*60)
    print("COMPREHENSIVE DATA AUGMENTATION AND PREPROCESSING TEST")
    print("="*60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Sample Data Processing", test_sample_data_processing),
        ("Preprocessing Script", test_preprocessing_script)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✅" if result else "❌"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The data augmentation and preprocessing system is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)