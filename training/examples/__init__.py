"""
Training Examples Package

This package contains example scripts and tutorials for using the training and
evaluation tools for medical AI models.

Examples include:
- Clinical assessment integration
- Model evaluation workflows
- Benchmark suite usage
- Expert review system setup

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

__version__ = "1.0.0"
__author__ = "Medical AI Assistant Team"
__date__ = "2025-11-04"

# Import available examples
try:
    from .clinical_assessment_integration import (
        MedicalAIWorkflow,
        mock_medical_model,
        main as run_clinical_integration_demo
    )
    __all__ = [
        "MedicalAIWorkflow",
        "mock_medical_model", 
        "run_clinical_integration_demo"
    ]
except ImportError:
    __all__ = []

# Package information
__all__.extend([
    "get_example_list",
    "run_example"
])

def get_example_list() -> list:
    """Get list of available examples"""
    examples = [
        {
            "name": "Clinical Assessment Integration",
            "description": "Complete workflow for clinical assessment of medical AI models",
            "script": "clinical_assessment_integration.py",
            "function": "run_clinical_integration_demo",
            "complexity": "advanced",
            "estimated_time": "10-15 minutes"
        }
    ]
    return examples

def run_example(example_name: str):
    """Run a specific example by name"""
    examples = {ex["name"].lower(): ex for ex in get_example_list()}
    
    if example_name.lower() not in examples:
        available = [ex["name"] for ex in examples.values()]
        raise ValueError(f"Example '{example_name}' not found. Available examples: {available}")
    
    example = examples[example_name.lower()]
    
    print(f"Running example: {example['name']}")
    print(f"Description: {example['description']}")
    print(f"Estimated time: {example['estimated_time']}")
    print()
    
    # Execute the example
    try:
        if example["function"] == "run_clinical_integration_demo":
            run_clinical_integration_demo()
        else:
            raise NotImplementedError(f"Example function '{example['function']}' not implemented")
    except Exception as e:
        print(f"Error running example: {e}")
        raise

def list_examples():
    """List all available examples with descriptions"""
    examples = get_example_list()
    
    print("Available Training Examples:")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Description: {example['description']}")
        print(f"   Complexity: {example['complexity']}")
        print(f"   Time: {example['estimated_time']}")
        print()
    
    print("To run an example:")
    print("run_example('Example Name')")
    print()
    print("Example: run_example('Clinical Assessment Integration')")

if __name__ == "__main__":
    print("Training Examples Package")
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print()
    list_examples()
