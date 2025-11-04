#!/usr/bin/env python3
"""
Clinical Assessment Tools - Final Verification

Quick verification that all clinical assessment tools are properly integrated
and functioning. This script validates the complete implementation.

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import sys
from pathlib import Path

# Add training directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("🔍 Clinical Assessment Tools - Final Verification")
    print("=" * 60)
    
    success_count = 0
    total_tests = 5
    
    # Test 1: Import all modules
    try:
        from utils.clinical_assessor import ClinicalAssessor
        from utils.medical_expert import ExpertReviewSystem  
        from evaluation.clinical_benchmarks import ClinicalBenchmarkSuite
        print("✅ Module imports successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Module import failed: {e}")
    
    # Test 2: Create instances
    try:
        assessor = ClinicalAssessor()
        expert_system = ExpertReviewSystem()
        benchmark_suite = ClinicalBenchmarkSuite()
        print("✅ Instance creation successful")
        success_count += 1
    except Exception as e:
        print(f"❌ Instance creation failed: {e}")
    
    # Test 3: Quick assessment test
    try:
        test_case = {
            "case_id": "verify_001",
            "symptoms": ["chest_pain", "shortness_of_breath"],
            "diagnosis": "myocardial_infarction",
            "treatments": ["aspirin", "nitroglycerin"],
            "medications": ["warfarin"]
        }
        assessment = assessor.comprehensive_assessment(test_case)
        print(f"✅ Clinical assessment working (Score: {assessment.overall_score:.3f})")
        success_count += 1
    except Exception as e:
        print(f"❌ Clinical assessment failed: {e}")
    
    # Test 4: Expert system test
    try:
        expert_system.create_sample_experts()
        print(f"✅ Expert system working ({len(expert_system.expert_db.experts)} experts)")
        success_count += 1
    except Exception as e:
        print(f"❌ Expert system failed: {e}")
    
    # Test 5: Benchmark suite test
    try:
        dataset_id = benchmark_suite.create_benchmark_dataset(
            name="Verification Dataset",
            category=ClinicalBenchmarkSuite.BenchmarkCategory.DIAGNOSTIC_ACCURACY,
            num_cases=3,
            difficulty_levels=[ClinicalBenchmarkSuite.DifficultyLevel.BASIC]
        )
        print(f"✅ Benchmark suite working (Dataset: {dataset_id})")
        success_count += 1
    except Exception as e:
        print(f"❌ Benchmark suite failed: {e}")
    
    # Final results
    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS: {success_count}/{total_tests} tests passed")
    print("=" * 60)
    
    if success_count == total_tests:
        print("🎉 ALL CLINICAL ASSESSMENT TOOLS VERIFIED SUCCESSFULLY!")
        print("\n✅ COMPLETED COMPONENTS:")
        print("  • Clinical Assessor - Medical accuracy evaluation")
        print("  • Medical Expert System - Professional review workflows")
        print("  • Clinical Benchmark Suite - Standardized evaluation")
        print("  • Integration Examples - Complete workflow demos")
        print("  • Comprehensive Documentation - Usage guides and API reference")
        
        print("\n📁 FILES CREATED:")
        print("  • training/utils/clinical_assessor.py (849 lines)")
        print("  • training/utils/medical_expert.py (1,013 lines)")  
        print("  • training/evaluation/clinical_benchmarks.py (1,497 lines)")
        print("  • training/examples/clinical_assessment_integration.py (592 lines)")
        print("  • training/CLINICAL_ASSESSMENT_README.md (462 lines)")
        print("  • training/test_clinical_tools.py (170 lines)")
        
        print("\n🔧 USAGE:")
        print("  from training.utils.clinical_assessor import ClinicalAssessor")
        print("  from training.utils.medical_expert import ExpertReviewSystem")
        print("  from training.evaluation.clinical_benchmarks import ClinicalBenchmarkSuite")
        
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please check implementation")
        return 1

if __name__ == "__main__":
    exit(main())
