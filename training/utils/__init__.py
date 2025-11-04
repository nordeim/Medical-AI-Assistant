"""
Training Utilities Package

This package provides comprehensive utilities for model training, checkpoint management,
training state management, backup operations, analytics, clinical assessment, and data processing.

Main Components:
- CheckpointManager: Advanced checkpoint management with cloud storage
- TrainingState: Comprehensive training state serialization and recovery
- BackupManager: Automated backup creation and disaster recovery
- CheckpointAnalytics: Performance analytics and reporting
- ResumeTraining CLI: Command-line interface for resuming training
- ClinicalAssessor: Clinical accuracy assessment for medical AI models
- MedicalExpertSystem: Professional medical expert review workflows
- ClinicalBenchmarkSuite: Comprehensive clinical benchmark evaluation
- DataAugmentor: Comprehensive data augmentation for medical conversations
- PreprocessingPipeline: Multi-stage data preprocessing with optimization
- DataQualityAssessment: Comprehensive quality assessment and validation
- AugmentationOptimizer: Strategy optimization for data augmentation

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

try:
    from .checkpoint_manager import (
        CheckpointManager,
        CheckpointConfig,
        CheckpointMetadata
    )
except ImportError:
    # CheckpointManager requires torch, provide stubs
    class CheckpointManager:
        pass
    class CheckpointConfig:
        pass
    class CheckpointMetadata:
        pass

try:
    from .training_state import (
        TrainingState,
        TrainingConfiguration,
        TrainingMetrics,
        EnvironmentState
    )
except ImportError:
    # TrainingState requires torch, provide stubs
    class TrainingState:
        pass
    class TrainingConfiguration:
        pass
    class TrainingMetrics:
        pass
    class EnvironmentState:
        pass

try:
    from .backup_manager import (
        BackupManager,
        BackupConfig,
        BackupMetadata
    )
except ImportError:
    # BackupManager requires torch, provide stubs
    class BackupManager:
        pass
    class BackupConfig:
        pass
    class BackupMetadata:
        pass

try:
    from .analytics import (
        CheckpointAnalytics,
        AnalyticsConfig
    )
except ImportError:
    # Analytics requires torch, provide stubs
    class CheckpointAnalytics:
        pass
    class AnalyticsConfig:
        pass

# Clinical assessment components (always available)
try:
    from .clinical_assessor import (
        ClinicalAssessor,
        ClinicalAssessment,
        ClinicalMetric,
        MedicalKnowledgeBase,
        RiskLevel,
        ClinicalDomain
    )
except ImportError:
    # ClinicalAssessor requires numpy, provide stubs
    class ClinicalAssessor:
        pass
    class ClinicalAssessment:
        pass
    class ClinicalMetric:
        pass
    class MedicalKnowledgeBase:
        pass
    class RiskLevel:
        pass
    class ClinicalDomain:
        pass

try:
    from .medical_expert import (
        MedicalExpertSystem,
        ExpertReview,
        ReviewWorkflow,
        ExpertProfile,
        MedicalExpertDatabase,
        ExpertRole,
        ReviewStatus,
        QualityLevel
    )
except ImportError:
    # MedicalExpertSystem requires numpy, provide stubs
    class MedicalExpertSystem:
        pass
    class ExpertReview:
        pass
    class ReviewWorkflow:
        pass
    class ExpertProfile:
        pass
    class MedicalExpertDatabase:
        pass
    class ExpertRole:
        pass
    class ReviewStatus:
        pass
    class QualityLevel:
        pass

# Data augmentation and preprocessing components (always available)
try:
    from .data_augmentation import (
        DataAugmentor,
        AugmentationConfig,
        MedicalTerminologyReplacer,
        ParaphraseGenerator,
        AdversarialExampleGenerator,
        BackTranslationGenerator,
        StyleTransferGenerator,
        MaskedLanguageModel,
        QualityControlValidator,
        DemographicAugmentor,
        apply_augmentation_pipeline
    )
except ImportError:
    # Data augmentation components require basic dependencies
    class DataAugmentor:
        pass
    class AugmentationConfig:
        pass
    class MedicalTerminologyReplacer:
        pass
    class ParaphraseGenerator:
        pass
    class AdversarialExampleGenerator:
        pass
    class BackTranslationGenerator:
        pass
    class StyleTransferGenerator:
        pass
    class MaskedLanguageModel:
        pass
    class QualityControlValidator:
        pass
    class DemographicAugmentor:
        pass
    def apply_augmentation_pipeline(*args, **kwargs):
        return {}

try:
    from .preprocessing_pipeline import (
        PreprocessingPipeline,
        PreprocessingConfig,
        DataCleaner,
        DataTransformer,
        MemoryOptimizedProcessor,
        create_preprocessing_pipeline,
        preprocess_medical_conversations
    )
except ImportError:
    class PreprocessingPipeline:
        pass
    class PreprocessingConfig:
        pass
    class DataCleaner:
        pass
    class DataTransformer:
        pass
    class MemoryOptimizedProcessor:
        pass
    def create_preprocessing_pipeline(*args, **kwargs):
        return None
    def preprocess_medical_conversations(*args, **kwargs):
        return {}

try:
    from .data_quality_assessment import (
        DataQualityAssessment,
        QualityMetrics,
        SemanticAnalyzer,
        MedicalAccuracyValidator,
        SafetyValidator,
        DiversityAnalyzer,
        assess_medical_conversation_quality
    )
except ImportError:
    class DataQualityAssessment:
        pass
    class QualityMetrics:
        pass
    class SemanticAnalyzer:
        pass
    class MedicalAccuracyValidator:
        pass
    class SafetyValidator:
        pass
    class DiversityAnalyzer:
        pass
    def assess_medical_conversation_quality(*args, **kwargs):
        return {}

try:
    from .augmentation_optimizer import (
        OptimizationOrchestrator,
        OptimizationConfig,
        StrategyEvaluator,
        GeneticOptimizer,
        GridSearchOptimizer,
        RandomOptimizer,
        OptimizationResult,
        optimize_augmentation_strategy
    )
except ImportError:
    class OptimizationOrchestrator:
        pass
    class OptimizationConfig:
        pass
    class StrategyEvaluator:
        pass
    class GeneticOptimizer:
        pass
    class GridSearchOptimizer:
        pass
    class RandomOptimizer:
        pass
    class OptimizationResult:
        pass
    def optimize_augmentation_strategy(*args, **kwargs):
        return {}

# Version information
__version__ = "1.0.0"
__author__ = "Medical AI Assistant Team"
__date__ = "2025-11-04"

# Package metadata
__all__ = [
    # Utility functions
    "create_checkpoint_manager",
    "create_training_state", 
    "create_backup_manager",
    "get_version_info",
    "validate_environment",
    "integrate_with_existing_training_loop",
    "create_clinical_assessor",
    "create_medical_expert_system",
    
    # Data processing utilities
    "create_preprocessing_pipeline",
    "preprocess_medical_conversations",
    "assess_medical_conversation_quality",
    "optimize_augmentation_strategy"
]

# Try to get CheckpointManager classes if available
try:
    from .checkpoint_manager import CheckpointManager, CheckpointConfig, CheckpointMetadata
    __all__.extend([
        "CheckpointManager",
        "CheckpointConfig", 
        "CheckpointMetadata"
    ])
except ImportError:
    pass

# Try to get ClinicalAssessment classes if available
try:
    from .clinical_assessor import (
        ClinicalAssessor, ClinicalAssessment, ClinicalMetric,
        MedicalKnowledgeBase, RiskLevel, ClinicalDomain
    )
    __all__.extend([
        "ClinicalAssessor",
        "ClinicalAssessment",
        "ClinicalMetric", 
        "MedicalKnowledgeBase",
        "RiskLevel",
        "ClinicalDomain"
    ])
except ImportError:
    pass

# Try to get MedicalExpertSystem classes if available
try:
    from .medical_expert import (
        MedicalExpertSystem, ExpertReview, ReviewWorkflow,
        ExpertProfile, MedicalExpertDatabase, ExpertRole,
        ReviewStatus, QualityLevel
    )
    __all__.extend([
        "MedicalExpertSystem",
        "ExpertReview",
        "ReviewWorkflow",
        "ExpertProfile",
        "MedicalExpertDatabase",
        "ExpertRole", 
        "ReviewStatus",
        "QualityLevel"
    ])
except ImportError:
    pass

# Try to get DataAugmentation classes if available
try:
    from .data_augmentation import (
        DataAugmentor, AugmentationConfig, MedicalTerminologyReplacer,
        ParaphraseGenerator, AdversarialExampleGenerator, BackTranslationGenerator,
        StyleTransferGenerator, MaskedLanguageModel, QualityControlValidator,
        DemographicAugmentor, apply_augmentation_pipeline
    )
    __all__.extend([
        "DataAugmentor",
        "AugmentationConfig",
        "MedicalTerminologyReplacer",
        "ParaphraseGenerator",
        "AdversarialExampleGenerator",
        "BackTranslationGenerator",
        "StyleTransferGenerator",
        "MaskedLanguageModel",
        "QualityControlValidator",
        "DemographicAugmentor",
        "apply_augmentation_pipeline"
    ])
except ImportError:
    pass

# Try to get PreprocessingPipeline classes if available
try:
    from .preprocessing_pipeline import (
        PreprocessingPipeline, PreprocessingConfig, DataCleaner,
        DataTransformer, MemoryOptimizedProcessor, create_preprocessing_pipeline,
        preprocess_medical_conversations
    )
    __all__.extend([
        "PreprocessingPipeline",
        "PreprocessingConfig",
        "DataCleaner",
        "DataTransformer",
        "MemoryOptimizedProcessor",
        "create_preprocessing_pipeline",
        "preprocess_medical_conversations"
    ])
except ImportError:
    pass

# Try to get DataQualityAssessment classes if available
try:
    from .data_quality_assessment import (
        DataQualityAssessment, QualityMetrics, SemanticAnalyzer,
        MedicalAccuracyValidator, SafetyValidator, DiversityAnalyzer,
        assess_medical_conversation_quality
    )
    __all__.extend([
        "DataQualityAssessment",
        "QualityMetrics",
        "SemanticAnalyzer",
        "MedicalAccuracyValidator",
        "SafetyValidator",
        "DiversityAnalyzer",
        "assess_medical_conversation_quality"
    ])
except ImportError:
    pass

# Try to get AugmentationOptimizer classes if available
try:
    from .augmentation_optimizer import (
        OptimizationOrchestrator, OptimizationConfig, StrategyEvaluator,
        GeneticOptimizer, GridSearchOptimizer, RandomOptimizer, OptimizationResult,
        optimize_augmentation_strategy
    )
    __all__.extend([
        "OptimizationOrchestrator",
        "OptimizationConfig",
        "StrategyEvaluator",
        "GeneticOptimizer",
        "GridSearchOptimizer",
        "RandomOptimizer",
        "OptimizationResult",
        "optimize_augmentation_strategy"
    ])
except ImportError:
    pass

# Utility functions for common operations
def create_checkpoint_manager(
    save_dir: str,
    experiment_name: str = "default",
    config: dict = None,
    **kwargs
):
    """
    Factory function to create a CheckpointManager with default or custom configuration
    
    Args:
        save_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        config: Optional configuration dictionary
        **kwargs: Additional configuration parameters
    
    Returns:
        Configured CheckpointManager instance
    """
    try:
        from .checkpoint_manager import CheckpointConfig
        CheckpointManager = globals()['CheckpointManager']
        
        # Create default config
        checkpoint_config = CheckpointConfig()
        
        # Update with provided config
        if config:
            for key, value in config.items():
                if hasattr(checkpoint_config, key):
                    setattr(checkpoint_config, key, value)
        
        # Update with kwargs
        for key, value in kwargs.items():
            if hasattr(checkpoint_config, key):
                setattr(checkpoint_config, key, value)
        
        return CheckpointManager(
            save_dir=save_dir,
            config=checkpoint_config,
            experiment_name=experiment_name
        )
    except ImportError:
        raise ImportError("CheckpointManager requires PyTorch. Install with: pip install torch")


def create_training_state(
    state_dir: str,
    experiment_name: str = "default",
    max_history: int = 1000
):
    """
    Factory function to create a TrainingState manager
    
    Args:
        state_dir: Directory to store training state
        experiment_name: Name of the experiment
        max_history: Maximum number of metrics to keep in history
    
    Returns:
        Configured TrainingState instance
    """
    try:
        from .training_state import TrainingState
        return TrainingState(
            state_dir=state_dir,
            experiment_name=experiment_name,
            max_history=max_history
        )
    except ImportError:
        raise ImportError("TrainingState requires PyTorch. Install with: pip install torch")


def create_backup_manager(
    backup_dir: str,
    experiment_name: str = "default",
    config: dict = None
) -> BackupManager:
    """
    Factory function to create a BackupManager with default or custom configuration
    
    Args:
        backup_dir: Directory to store backups
        experiment_name: Name of the experiment
        config: Optional configuration dictionary
    
    Returns:
        Configured BackupManager instance
    """
    from .backup_manager import BackupConfiguration
    
    # Create default config
    backup_config = BackupConfiguration()
    
    # Update with provided config
    if config:
        for key, value in config.items():
            if hasattr(backup_config, key):
                setattr(backup_config, key, value)
    
    return BackupManager(
        backup_dir=backup_dir,
        experiment_name=experiment_name,
        config=backup_config
    )


def create_clinical_assessor(knowledge_base_path: str = None):
    """
    Factory function to create a ClinicalAssessor with optional knowledge base
    
    Args:
        knowledge_base_path: Optional path to custom knowledge base
    
    Returns:
        Configured ClinicalAssessor instance
    """
    try:
        from .clinical_assessor import ClinicalAssessor, MedicalKnowledgeBase
        
        if knowledge_base_path:
            # Load custom knowledge base (implementation depends on format)
            knowledge_base = MedicalKnowledgeBase()  # Simplified - could load from file
        else:
            knowledge_base = None
            
        return ClinicalAssessor(knowledge_base=knowledge_base)
    except ImportError:
        raise ImportError("ClinicalAssessor requires numpy. Install with: pip install numpy")


def create_medical_expert_system(expert_db_path: str = None):
    """
    Factory function to create a MedicalExpertSystem with optional expert database
    
    Args:
        expert_db_path: Optional path to expert database
    
    Returns:
        Configured MedicalExpertSystem instance
    """
    try:
        from .medical_expert import MedicalExpertSystem, MedicalExpertDatabase
        
        if expert_db_path:
            # Load custom expert database (implementation depends on format)
            expert_db = MedicalExpertDatabase()  # Simplified - could load from file
        else:
            expert_db = MedicalExpertDatabase()
            
        return MedicalExpertSystem(expert_db=expert_db)
    except ImportError:
        raise ImportError("MedicalExpertSystem requires numpy. Install with: pip install numpy")


def get_version_info() -> dict:
    """Get version and package information"""
    return {
        "version": __version__,
        "author": __author__,
        "date": __date__,
        "components": {
            "checkpoint_manager": CheckpointManager.__module__,
            "training_state": TrainingState.__module__,
            "backup_manager": BackupManager.__module__,
            "analytics": CheckpointAnalytics.__module__,
            "clinical_assessor": ClinicalAssessor.__module__ if ClinicalAssessor != type(None) else None,
            "medical_expert_system": MedicalExpertSystem.__module__ if MedicalExpertSystem != type(None) else None
        }
    }


def validate_environment() -> dict:
    """Validate training environment and dependencies"""
    import sys
    import platform
    
    validation_result = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "pytorch_version": "Not installed",
            "cuda_available": False,
            "numpy_available": False,
            "clinical_assessment_available": False,
            "expert_review_available": False
        }
    }
    
    # Check PyTorch
    try:
        import torch
        validation_result["environment"]["pytorch_version"] = torch.__version__
        validation_result["environment"]["cuda_available"] = torch.cuda.is_available()
        if not hasattr(torch, 'version') or not torch.version:
            validation_result["warnings"].append("PyTorch version information not available")
    except ImportError:
        validation_result["errors"].append("PyTorch not installed")
        validation_result["valid"] = False
    
    # Check NumPy
    try:
        import numpy
        validation_result["environment"]["numpy_available"] = True
        validation_result["clinical_assessment_available"] = True
        validation_result["expert_review_available"] = True
    except ImportError:
        validation_result["warnings"].append("NumPy not installed - clinical assessment tools limited")
        validation_result["environment"]["clinical_assessment_available"] = False
        validation_result["environment"]["expert_review_available"] = False
    
    # Check optional dependencies
    optional_deps = {
        "boto3": "AWS S3 support",
        "azure.storage.blob": "Azure Blob Storage support", 
        "google.cloud.storage": "Google Cloud Storage support",
        "matplotlib": "Plotting and visualization",
        "pandas": "Data analysis",
        "seaborn": "Statistical visualization",
        "psutil": "System monitoring"
    }
    
    missing_optional = []
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace(".", "_"))
        except ImportError:
            missing_optional.append(f"{dep}: {description}")
    
    if missing_optional:
        validation_result["warnings"].append(
            f"Optional dependencies not found: {', '.join(missing_optional)}"
        )
    
    return validation_result


# Integration helpers
def integrate_with_existing_training_loop(
    model,
    optimizer,
    scheduler=None,
    checkpoint_manager: CheckpointManager = None,
    training_state: TrainingState = None,
    save_frequency: int = 1000
):
    """
    Helper function to integrate checkpointing into existing training loops
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        checkpoint_manager: CheckpointManager instance
        training_state: TrainingState instance
        save_frequency: Save checkpoint every N steps
    
    Returns:
        Dictionary with integration helpers
    """
    step_counter = 0
    
    def save_checkpoint_if_needed(epoch, metrics=None, force_save=False):
        nonlocal step_counter
        step_counter += 1
        
        if checkpoint_manager and (force_save or step_counter % save_frequency == 0):
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step_counter,
                metrics=metrics
            )
    
    def update_metrics(epoch, metrics_dict):
        if training_state:
            metrics = TrainingMetrics(
                epoch=epoch,
                step=step_counter,
                **metrics_dict
            )
            training_state.update_metrics(metrics)
    
    return {
        "save_checkpoint_if_needed": save_checkpoint_if_needed,
        "update_metrics": update_metrics,
        "get_step": lambda: step_counter
    }


# Clinical assessment integration helpers
def integrate_clinical_assessment(model, assessor: ClinicalAssessor = None, frequency: int = 100):
    """
    Helper function to integrate clinical assessment into training loops
    
    Args:
        model: Model to assess (should have predict or forward method)
        assessor: ClinicalAssessor instance
        frequency: Assess model every N training steps
    
    Returns:
        Dictionary with clinical assessment helpers
    """
    step_counter = 0
    
    def assess_model_if_needed(clinical_cases=None, force_assess=False):
        nonlocal step_counter
        step_counter += 1
        
        if assessor and (force_assess or step_counter % frequency == 0):
            if clinical_cases:
                return assessor.batch_assess(clinical_cases)
            else:
                return None  # No cases provided
    
    def assess_single_case(case_data):
        if assessor:
            return assessor.comprehensive_assessment(case_data)
        return None
    
    return {
        "assess_model_if_needed": assess_model_if_needed,
        "assess_single_case": assess_single_case,
        "get_step": lambda: step_counter
    }


# Example usage
def example_usage():
    """Example of how to use the training utilities"""
    
    # Create managers
    checkpoint_manager = create_checkpoint_manager(
        save_dir="./checkpoints",
        experiment_name="my_experiment",
        config={
            "save_every_n_steps": 500,
            "compress_checkpoints": True,
            "use_cloud_backup": False
        }
    )
    
    training_state = create_training_state(
        state_dir="./state",
        experiment_name="my_experiment"
    )
    
    # Create clinical assessment components
    clinical_assessor = create_clinical_assessor()
    expert_system = create_medical_expert_system()
    
    # Simulate training loop with clinical assessment
    integration = integrate_with_existing_training_loop(
        model=None,  # Replace with actual model
        optimizer=None,  # Replace with actual optimizer
        checkpoint_manager=checkpoint_manager,
        training_state=training_state
    )
    
    clinical_integration = integrate_clinical_assessment(
        model=None,  # Replace with actual model
        assessor=clinical_assessor
    )
    
    # Example metrics
    for epoch in range(10):
        # Simulate training step
        metrics = {
            "loss": 1.0 / (epoch + 1),
            "accuracy": min(0.9, 0.5 + epoch * 0.05),
            "learning_rate": 1e-3 / (epoch + 1)
        }
        
        # Update state and potentially save checkpoint
        integration["update_metrics"](epoch, metrics)
        integration["save_checkpoint_if_needed"](epoch, metrics)
        
        # Perform clinical assessment periodically
        clinical_assessment = clinical_integration["assess_model_if_needed"]()
        if clinical_assessment:
            print(f"Clinical assessment completed for epoch {epoch}")
    
    # Generate analytics
    analytics = CheckpointAnalytics(
        checkpoint_manager=checkpoint_manager,
        training_state=training_state,
        output_dir="./analytics"
    )
    
    analytics_data = analytics.generate_comprehensive_analytics()
    print("Analytics generated successfully!")


if __name__ == "__main__":
    # Show package information
    print(f"Training Utilities Package v{__version__}")
    print(f"Author: {__author__}")
    print(f"Date: {__date__}")
    
    # Validate environment
    validation = validate_environment()
    print(f"\nEnvironment Validation:")
    print(f"Valid: {validation['valid']}")
    print(f"Clinical Assessment Available: {validation['environment']['clinical_assessment_available']}")
    print(f"Expert Review Available: {validation['environment']['expert_review_available']}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  - {error}")
    
    # Example usage
    print("\nRun example usage (with actual model/optimizer):")
    print("example_usage()")