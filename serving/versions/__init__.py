"""
Model Version Tracking System for Medical AI Assistant

Enterprise-grade model lifecycle management with medical compliance,
audit trails, and production safety mechanisms for Phase 6.

Components:
- Semantic versioning with medical compliance tracking
- Model registry integration (MLflow/W&B support)
- Version compatibility checking
- A/B testing infrastructure
- Rollout and rollback mechanisms
- Performance comparison utilities
- Version metadata and documentation tracking
"""

from .core import ModelVersion, VersionRegistry, VersionManager
from .registry import MLflowRegistry, WandbRegistry, RegistryAdapter
from .compatibility import CompatibilityChecker, VersionCompatibility
from .testing import ABTestingManager, ExperimentConfig
from .deployment import RolloutManager, RollbackManager, DeploymentStatus
from .comparison import PerformanceComparator, MedicalMetrics
from .metadata import VersionMetadata, ClinicalValidation

__version__ = "1.0.0"
__all__ = [
    "ModelVersion",
    "VersionRegistry", 
    "VersionManager",
    "MLflowRegistry",
    "WandbRegistry", 
    "RegistryAdapter",
    "CompatibilityChecker",
    "VersionCompatibility",
    "ABTestingManager",
    "ExperimentConfig",
    "RolloutManager",
    "RollbackManager",
    "DeploymentStatus",
    "PerformanceComparator",
    "MedicalMetrics",
    "VersionMetadata",
    "ClinicalValidation"
]