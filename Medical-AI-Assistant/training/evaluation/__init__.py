"""
Training Evaluation Package

This package provides comprehensive evaluation tools for medical AI models including:
- Clinical benchmarks and datasets
- Model evaluation and comparison
- Performance analytics and reporting
- Quality assurance workflows

Main Components:
- ClinicalBenchmarkSuite: Comprehensive clinical benchmark evaluation
- BenchmarkDataset: Dataset management and configuration
- EvaluationResult: Standardized evaluation result format

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

try:
    from .clinical_benchmarks import (
        ClinicalBenchmarkSuite,
        BenchmarkDataset,
        BenchmarkCase,
        EvaluationResult,
        ClinicalDatasetGenerator,
        BenchmarkCategory,
        DifficultyLevel,
        DatasetType
    )
except ImportError:
    # ClinicalBenchmarkSuite requires numpy, provide stubs
    class ClinicalBenchmarkSuite:
        pass
    class BenchmarkDataset:
        pass
    class BenchmarkCase:
        pass
    class EvaluationResult:
        pass
    class ClinicalDatasetGenerator:
        pass
    class BenchmarkCategory:
        pass
    class DifficultyLevel:
        pass
    class DatasetType:
        pass

# Version information
__version__ = "1.0.0"
__author__ = "Medical AI Assistant Team"
__date__ = "2025-11-04"

# Package metadata
__all__ = [
    "create_benchmark_suite",
    "get_version_info",
    "validate_evaluation_environment"
]

# Factory function
def create_benchmark_suite():
    """
    Factory function to create a ClinicalBenchmarkSuite
    
    Returns:
        Configured ClinicalBenchmarkSuite instance
    """
    try:
        from .clinical_benchmarks import ClinicalBenchmarkSuite
        return ClinicalBenchmarkSuite()
    except ImportError:
        raise ImportError("ClinicalBenchmarkSuite requires numpy. Install with: pip install numpy")

def get_version_info() -> dict:
    """Get version and package information"""
    return {
        "version": __version__,
        "author": __author__,
        "date": __date__,
        "components": {
            "clinical_benchmarks": ClinicalBenchmarkSuite.__module__ if ClinicalBenchmarkSuite != type(None) else None
        }
    }

def validate_evaluation_environment() -> dict:
    """Validate evaluation environment and dependencies"""
    import sys
    import platform
    
    validation_result = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_available": False,
            "clinical_benchmarks_available": False,
            "pandas_available": False
        }
    }
    
    # Check NumPy
    try:
        import numpy
        validation_result["environment"]["numpy_available"] = True
        validation_result["clinical_benchmarks_available"] = True
    except ImportError:
        validation_result["errors"].append("NumPy not installed - evaluation tools limited")
        validation_result["valid"] = False
    
    # Check Pandas
    try:
        import pandas
        validation_result["environment"]["pandas_available"] = True
    except ImportError:
        validation_result["warnings"].append("Pandas not installed - some data processing features limited")
    
    # Check other dependencies
    optional_deps = {
        "matplotlib": "Plotting and visualization",
        "seaborn": "Statistical visualization",
        "scikit_learn": "Machine learning evaluation metrics",
        "jupyter": "Interactive evaluation notebooks"
    }
    
    missing_optional = []
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(f"{dep}: {description}")
    
    if missing_optional:
        validation_result["warnings"].append(
            f"Optional dependencies not found: {', '.join(missing_optional)}"
        )
    
    return validation_result

# Example usage
def example_evaluation_workflow():
    """Example of evaluation workflow"""
    
    # Create benchmark suite
    suite = create_benchmark_suite()
    
    # Create a test dataset
    dataset_id = suite.create_benchmark_dataset(
        name="Sample Diagnostic Dataset",
        category=BenchmarkCategory.DIAGNOSTIC_ACCURACY,
        num_cases=20,
        difficulty_levels=[DifficultyLevel.BASIC],
        dataset_type=DatasetType.SYNTHETIC
    )
    
    # Define a mock model for testing
    def mock_model(clinical_data):
        return {
            "primary_diagnosis": clinical_data.get("primary_diagnosis", "unknown"),
            "differential_diagnosis": clinical_data.get("differential_diagnosis", []),
            "confidence_score": 0.8
        }
    
    # Evaluate the model
    result = suite.evaluate_model(
        model_name="MockModel_v1.0",
        model_function=mock_model,
        dataset_id=dataset_id,
        output_dir="./evaluation_results"
    )
    
    # Generate report
    report = suite.generate_performance_report(dataset_id)
    
    return {
        "dataset_id": dataset_id,
        "evaluation_result": result,
        "performance_report": report
    }

if __name__ == "__main__":
    # Show package information
    print(f"Training Evaluation Package v{__version__}")
    print(f"Author: {__author__}")
    print(f"Date: {__date__}")
    
    # Validate environment
    validation = validate_evaluation_environment()
    print(f"\nEnvironment Validation:")
    print(f"Valid: {validation['valid']}")
    print(f"Clinical Benchmarks Available: {validation['environment']['clinical_benchmarks_available']}")
    print(f"Pandas Available: {validation['environment']['pandas_available']}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # Example workflow
    print("\nRun example evaluation workflow:")
    print("example_evaluation_workflow()")
