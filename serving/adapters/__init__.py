"""
LoRA Adapter Management System for Medical AI Serving

This package provides comprehensive LoRA adapter management for production
medical AI serving environments with zero-downtime updates, validation,
and rollback capabilities.
"""

from .registry import AdapterRegistry, AdapterVersion, AdapterMetadata
from .manager import AdapterManager, AdapterLifecycleManager
from .validator import AdapterValidator, ModelCompatibilityChecker, ValidationResult
from .cache import AdapterCache, MemoryOptimizedCache
from .hot_swap import HotSwapManager, SwapOperation
from .rollback import RollbackManager, FallbackAdapter
from .metrics import AdapterMetrics, UsageStatistics

__all__ = [
    'AdapterRegistry',
    'AdapterVersion', 
    'AdapterMetadata',
    'AdapterManager',
    'AdapterLifecycleManager',
    'AdapterValidator',
    'ModelCompatibilityChecker',
    'ValidationResult',
    'AdapterCache',
    'MemoryOptimizedCache',
    'HotSwapManager',
    'SwapOperation',
    'RollbackManager',
    'FallbackAdapter',
    'AdapterMetrics',
    'UsageStatistics'
]

__version__ = "1.0.0"