#!/usr/bin/env python3
"""
Simple test of clinical assessment tools functionality
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_clinical_assessor():
    """Test the clinical assessor functionality"""
    print("Testing ClinicalAssessor...")
    
    from utils.clinical_assessor import ClinicalAssessor, RiskLevel
    
    # Create assessor
    assessor = ClinicalAssessor()
    
    # Create test case
    test_case = {
        "case_id": "test_001",
        "symptoms": ["chest_pain", "shortness_of_breath"],
        "diagnosis": "myocardial_infarction",
        "explanation": "Classic cardiac symptoms suggesting acute coronary syndrome",
        "treatments": ["aspirin", "nitroglycerin", "oxygen"],
        "medications": ["warfarin", "aspirin"],
        "patient_context": {
            "conditions": ["hypertension", "diabetes"],
            "age": 65
        },
        "high_risk_indicators": ["elderly", "cardiac_symptoms", "diabetes"],
        "scenario_type": "emergency"
    }
    
    # Run assessment
    try:
        assessment = assessor.comprehensive_assessment(test_case)
        print(f"✓ Assessment completed successfully")
        print(f"  Overall Score: {assessment.overall_score:.3f}")
        print(f"  Risk Level: {assessment.risk_level.value}")
        print(f"  Recommendations: {len(assessment.recommendations)}")
        return True
    except Exception as e:
        print(f"✗ Assessment failed: {e}")
        return False

def test_expert_system():
    """Test the expert review system"""
    print("\nTesting ExpertReviewSystem...")
    
    from utils.medical_expert import ExpertReviewSystem, ExpertRole
    
    # Create expert system
    expert_system = ExpertReviewSystem()
    
    # Create sample experts
    try:
        expert_system.create_sample_experts()
        print("✓ Sample experts created")
        
        # Test case
        test_case = {
            "case_id": "expert_test_001",
            "symptoms": ["chest_pain"],
            "diagnosis": "myocardial_infarction"
        }
        
        # Submit for review
        submission_id = expert_system.submit_case_for_review(
            case_data=test_case,
            submitted_by="TestSystem",
            required_expert_roles=[ExpertRole.CARDIOLOGIST],
            priority="high"
        )
        
        print(f"✓ Case submitted for review: {submission_id}")
        
        # Check workflow status
        if expert_system.active_workflows:
            workflow_id = list(expert_system.active_workflows.keys())[0]
            status = expert_system.get_workflow_status(workflow_id)
            print(f"✓ Workflow status: {status['status']}")
        
        return True
    except Exception as e:
        print(f"✗ Expert system test failed: {e}")
        return False

def test_benchmark_suite():
    """Test the benchmark suite"""
    print("\nTesting ClinicalBenchmarkSuite...")
    
    from evaluation.clinical_benchmarks import ClinicalBenchmarkSuite, BenchmarkCategory, DifficultyLevel
    
    # Create benchmark suite
    suite = ClinicalBenchmarkSuite()
    
    try:
        # Create small test dataset
        dataset_id = suite.create_benchmark_dataset(
            name="Test Dataset",
            category=BenchmarkCategory.DIAGNOSTIC_ACCURACY,
            num_cases=5,
            difficulty_levels=[DifficultyLevel.BASIC]
        )
        
        print(f"✓ Benchmark dataset created: {dataset_id}")
        
        # Test mock model function
        def mock_model(clinical_data):
            return {
                "primary_diagnosis": clinical_data.get("primary_diagnosis", "test_diagnosis"),
                "confidence_score": 0.8
            }
        
        # Run evaluation
        result = suite.evaluate_model(
            model_name="TestModel",
            model_function=mock_model,
            dataset_id=dataset_id,
            output_dir="./test_results"
        )
        
        print(f"✓ Benchmark evaluation completed")
        print(f"  Model Score: {result.overall_score:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("CLINICAL ASSESSMENT TOOLS FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Clinical Assessor", test_clinical_assessor),
        ("Expert Review System", test_expert_system), 
        ("Benchmark Suite", test_benchmark_suite)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("✓ All tests passed! Clinical assessment tools are working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())
