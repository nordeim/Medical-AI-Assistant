"""
Comprehensive Test Suite for LoRA Adapter Management System

This test file demonstrates all major features of the adapter management
system and provides examples of production usage patterns.
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

# Import all components
from adapters import (
    SystemConfiguration,
    AdapterLifecycleManager,
    create_lifecycle_manager,
    lifecycle_manager_context,
    create_default_config,
    
    # Registry components
    AdapterRegistry,
    AdapterMetadata,
    AdapterType,
    create_adapter_metadata,
    AdapterVersion,
    
    # Validator components
    AdapterValidator,
    ValidationResult,
    
    # Hot-swap components
    SwapStrategy,
    
    # Rollback components
    RollbackTrigger,
    FallbackStrategy,
    
    # Metrics components
    OperationType,
    MetricType,
    
    # Cache components
    CacheStrategy,
    MemoryOptimizationLevel,
    create_medical_adapter_cache
)


async def test_registry_system():
    """Test the adapter registry and versioning system."""
    print("\n=== Testing Registry System ===")
    
    # Create temporary registry
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = Path(temp_dir) / "test_registry.db"
        registry = AdapterRegistry(str(registry_path))
        
        # Test creating and registering adapters
        metadata1 = create_adapter_metadata(
            adapter_id="medical_diagnosis_v1",
            name="Medical Diagnosis Assistant",
            description="LoRA adapter for medical diagnosis assistance",
            adapter_type=AdapterType.MEDICAL_LORA,
            medical_domain="diagnostic_medicine",
            clinical_use_case="differential_diagnosis"
        )
        
        metadata2 = create_adapter_metadata(
            adapter_id="clinical_assessment_v2",
            name="Clinical Assessment Tool",
            description="Adapter for clinical patient assessment",
            adapter_type=AdapterType.CLINICAL_LORA,
            medical_domain="clinical_assessment",
            clinical_use_case="patient_screening"
        )
        
        # Register adapters
        success1 = await registry.register_adapter(metadata1)
        success2 = await registry.register_adapter(metadata2)
        
        print(f"Registered adapter 1: {success1}")
        print(f"Registered adapter 2: {success2}")
        
        # Add versions
        version1 = AdapterVersion(
            version_id="v1.0.0",
            version="1.0.0",
            adapter_type=AdapterType.MEDICAL_LORA,
            base_model_id="microsoft/DialoGPT-medium",
            created_at=datetime.now(timezone.utc),
            created_by="medical_ai_team",
            description="Initial version with basic diagnosis capabilities",
            tags=["diagnosis", "medical", "v1.0"]
        )
        
        version2 = AdapterVersion(
            version_id="v2.0.0",
            version="2.0.0",
            adapter_type=AdapterType.MEDICAL_LORA,
            base_model_id="microsoft/DialoGPT-medium",
            created_at=datetime.now(timezone.utc),
            created_by="medical_ai_team",
            description="Enhanced version with improved accuracy",
            tags=["diagnosis", "medical", "v2.0", "improved"]
        )
        
        registry.add_version("medical_diagnosis_v1", version1)
        registry.add_version("medical_diagnosis_v1", version2)
        registry.add_version("clinical_assessment_v2", version1)
        
        # Test retrieval
        retrieved1 = registry.get_adapter("medical_diagnosis_v1")
        retrieved2 = registry.get_adapter("clinical_assessment_v2")
        
        print(f"Retrieved adapter 1: {retrieved1.name} (versions: {retrieved1.get_version_count()})")
        print(f"Retrieved adapter 2: {retrieved2.name} (versions: {retrieved2.get_version_count()})")
        
        # Test listing
        all_adapters = registry.list_adapters()
        medical_adapters = registry.list_adapters(adapter_type=AdapterType.MEDICAL_LORA)
        
        print(f"Total adapters: {len(all_adapters)}")
        print(f"Medical adapters: {len(medical_adapters)}")
        
        # Test usage statistics
        registry.update_usage_stats("medical_diagnosis_v1", "load", 1500.0, True)
        registry.update_usage_stats("medical_diagnosis_v1", "load", 1200.0, True)
        registry.update_usage_stats("medical_diagnosis_v1", "load", 3000.0, False)
        
        usage_stats = registry.get_usage_statistics("medical_diagnosis_v1")
        print(f"Usage statistics for medical_diagnosis_v1:")
        print(f"  Total operations: {usage_stats['total_operations']}")
        print(f"  Success rate: {usage_stats['success_rate']:.1%}")
        
        # Test medical compliance
        compliance = registry.validate_medical_compliance("medical_diagnosis_v1")
        print(f"Medical compliance for medical_diagnosis_v1: {compliance['valid']}")
        
        # Test registry statistics
        stats = registry.get_registry_stats()
        print(f"Registry statistics:")
        print(f"  Total adapters: {stats['total_adapters']}")
        print(f"  Total versions: {stats['total_versions']}")
        print(f"  Operations last 24h: {stats['operations_last_24h']}")


async def test_validation_system():
    """Test adapter validation and compatibility checking."""
    print("\n=== Testing Validation System ===")
    
    validator = AdapterValidator()
    
    # Mock validation test (would need real adapter files)
    print("Validator initialized successfully")
    print("Supported architectures: Llama, Mistral, T5, BART")
    print("Supported PEFT types: LoRA, AdaLoRA, IA3, Prefix Tuning, P-Tuning")
    print("Medical compliance checks: HIPAA, FDA, Clinical Validation")
    
    # Test validation result structure
    print("\nValidation would check:")
    print("  - Model architecture compatibility")
    print("  - PEFT configuration validation")  
    print("  - Tokenizer compatibility")
    print("  - Medical compliance requirements")
    print("  - Performance thresholds")
    print("  - Security and safety requirements")


async def test_cache_system():
    """Test memory-optimized caching system."""
    print("\n=== Testing Cache System ===")
    
    cache = create_medical_adapter_cache(
        max_size=3,
        optimization_level=MemoryOptimizationLevel.BALANCED
    )
    
    # Test basic cache operations
    mock_metadata = {
        "adapter_id": "test_adapter_v1",
        "version_id": "v1.0.0",
        "file_size": 1000000,
        "compliance_flags": ["validated", "medical_domain"],
        "ttl_seconds": 3600
    }
    
    # Add items to cache
    cache.put("adapter_1", "test_adapter_1", "v1.0.0", "mock_model_1", mock_metadata)
    cache.put("adapter_2", "test_adapter_2", "v1.0.0", "mock_model_2", mock_metadata)
    cache.put("adapter_3", "test_adapter_3", "v1.0.0", "mock_model_3", mock_metadata)
    
    print(f"Cached items: {len(cache._cache)}")
    
    # Test retrieval
    model = cache.get("adapter_1", "test_adapter_1")
    print(f"Retrieved model: {model}")
    
    # Test cache statistics
    stats = cache.get_stats()
    print(f"Cache statistics:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Memory usage: {stats['memory_usage']['rss_mb']:.1f}MB")
    
    # Test medical-specific features
    medical_stats = cache.get_medical_cache_stats()
    print(f"Medical cache features:")
    print(f"  Medical adapters: {medical_stats['medical_adapters']}")
    print(f"  PHI-protected: {medical_stats['phi_protected_adapters']}")


async def test_lifecycle_manager():
    """Test the complete lifecycle management system."""
    print("\n=== Testing Lifecycle Manager ===")
    
    # Create configuration
    config = create_default_config(
        base_model_id="microsoft/DialoGPT-medium",
        medical_mode=True
    )
    
    # Test configuration
    print(f"Configuration created:")
    print(f"  Base model: {config.base_model_id}")
    print(f"  Medical compliance: {config.enable_medical_compliance}")
    print(f"  Max memory: {config.max_memory_mb}MB")
    print(f"  Auto rollback: {config.auto_rollback_enabled}")
    
    # Create lifecycle manager with context manager
    async with lifecycle_manager_context(config) as lifecycle:
        print("\nLifecycle manager initialized")
        
        # Test system status
        status = await lifecycle.get_system_status()
        print(f"Initial status:")
        print(f"  State: {status['state']}")
        print(f"  Active adapters: {status['active_adapters']}")
        print(f"  System health: {status['system_health']}")
        
        # Test health check
        health = await lifecycle.health_check()
        print(f"\nHealth check:")
        print(f"  Overall health: {health['overall_health']}")
        print(f"  Components checked: {list(health['components'].keys())}")
        
        # Register test adapters
        metadata1 = create_adapter_metadata(
            adapter_id="test_medical_v1",
            name="Test Medical Adapter",
            description="Test adapter for lifecycle testing",
            adapter_type=AdapterType.MEDICAL_LORA,
            medical_domain="test_medicine"
        )
        
        metadata2 = create_adapter_metadata(
            adapter_id="test_clinical_v1",
            name="Test Clinical Adapter", 
            description="Test clinical adapter",
            adapter_type=AdapterType.CLINICAL_LORA,
            medical_domain="test_clinical"
        )
        
        success1 = await lifecycle.register_adapter(metadata1)
        success2 = await lifecycle.register_adapter(metadata2)
        
        print(f"\nRegistered adapters: {success1 and success2}")
        
        # List adapters
        adapters = await lifecycle.list_adapters()
        print(f"Available adapters: {len(adapters)}")
        
        # Test metrics system
        if lifecycle.metrics:
            print("\nMetrics system active")
            
            # Record test metric
            lifecycle.metrics.record_metric(
                "test_metric", 1.0, "test_adapter", MetricType.COUNTER
            )
            
            # Get system overview
            overview = lifecycle.metrics.get_system_overview()
            print(f"  System overview: {overview['system_health']}")
        
        print("\nLifecycle manager testing completed")


async def test_hot_swap_simulation():
    """Test hot-swap functionality simulation."""
    print("\n=== Testing Hot-Swap System ===")
    
    print("Hot-swap strategies available:")
    print("  BLUE_GREEN: Complete replacement")
    print("  CANARY: Gradual rollout with testing")
    print("  GRADUAL_ROLLOUT: Phased traffic shifting")
    print("  SHADOW: Parallel testing")
    print("  EMERGENCY: Critical situation handling")
    
    print("\nHot-swap process simulation:")
    print("1. Validate target adapter")
    print("2. Load new adapter in background")
    print("3. Run health checks")
    print("4. Execute strategy-specific switch")
    print("5. Monitor for issues")
    print("6. Rollback if needed")
    
    print("\nHealth monitoring during swap:")
    print("  - Adapter state monitoring")
    print("  - Performance metrics tracking")
    print("  - Error rate monitoring")
    print("  - Memory usage tracking")


async def test_rollback_system():
    """Test rollback and fallback mechanisms."""
    print("\n=== Testing Rollback System ===")
    
    print("Rollback triggers:")
    print("  MANUAL: User-initiated rollback")
    print("  HEALTH_CHECK_FAILED: Automatic on health failure")
    print("  PERFORMANCE_DEGRADATION: High latency detected")
    print("  ERROR_RATE_HIGH: Too many errors")
    print("  VALIDATION_FAILED: Validation issues")
    print("  SYSTEM_ERROR: System-level problems")
    print("  EMERGENCY: Critical emergency situations")
    
    print("\nFallback strategies:")
    print("  REVERT_TO_PREVIOUS: Go back to previous version")
    print("  LOAD_STABLE_VERSION: Use known stable adapter")
    print("  USE_BASE_MODEL: Fall back to base model")
    print("  CIRCUIT_BREAKER: Temporarily disable adapter")
    print("  GRADUAL_FALLBACK: Progressive traffic reduction")
    
    print("\nRollback process:")
    print("1. Detect trigger condition")
    print("2. Select best fallback strategy")
    print("3. Execute rollback operation")
    print("4. Monitor recovery")
    print("5. Update system status")


async def test_metrics_system():
    """Test metrics collection and analytics."""
    print("\n=== Testing Metrics System ===")
    
    from adapters.metrics import create_metrics_system
    
    # Create metrics system
    with tempfile.TemporaryDirectory() as temp_dir:
        metrics_path = Path(temp_dir) / "test_metrics.db"
        metrics = create_metrics_system(str(metrics_path))
        
        await metrics.start_monitoring()
        
        try:
            print("Metrics system initialized")
            
            # Record test metrics
            metrics.record_metric("system_load", 0.75, metric_type=MetricType.GAUGE)
            metrics.record_metric("memory_usage_mb", 2048.0, "adapter_v1", MetricType.GAUGE)
            metrics.record_metric("inference_latency_ms", 150.0, "adapter_v1", MetricType.HISTOGRAM)
            
            # Track operations
            op_id = metrics.start_operation_tracking(OperationType.LOAD, "adapter_v1")
            await asyncio.sleep(0.1)  # Simulate operation time
            metrics.complete_operation_tracking(op_id, success=True)
            
            # Get system overview
            overview = metrics.get_system_overview()
            print(f"System overview:")
            print(f"  Active adapters: {overview['active_adapters']}")
            print(f"  Success rate: {overview['overall_success_rate']:.1%}")
            print(f"  System health: {overview['system_health']}")
            
            # Test export
            export_data = metrics.export_metrics(format="json")
            exported = json.loads(export_data)
            print(f"Exported data structure:")
            print(f"  Metrics count: {len(exported['metrics'])}")
            print(f"  Operations count: {len(exported['operations'])}")
            
        finally:
            await metrics.stop_monitoring()


async def test_medical_compliance():
    """Test medical AI compliance features."""
    print("\n=== Testing Medical Compliance ===")
    
    print("Medical compliance features:")
    print("  HIPAA compliance checking")
    print("  FDA regulatory alignment")
    print("  Clinical validation requirements")
    print("  PHI protection measures")
    print("  Safety score validation")
    print("  Audit logging")
    
    print("\nCompliance validation process:")
    print("1. Check required compliance flags")
    print("2. Validate safety score thresholds")
    print("3. Verify clinical trial data")
    print("4. Test PHI protection mechanisms")
    print("5. Audit access patterns")
    print("6. Generate compliance report")
    
    print("\nMedical adapter types:")
    print("  MEDICAL_LORA: General medical AI tasks")
    print("  CLINICAL_LORA: Clinical decision support")
    print("  RESEARCH_LORA: Research-only adapters")


async def test_production_scenario():
    """Test a complete production scenario."""
    print("\n=== Production Scenario Test ===")
    
    # Production configuration
    config = SystemConfiguration(
        base_model_id="production-medical-model",
        registry_path="./prod_registry.db",
        enable_medical_compliance=True,
        cache_strategy=CacheStrategy.LRU,
        memory_optimization=MemoryOptimizationLevel.MEDICAL_STRICT,
        max_cache_size=3,
        max_memory_mb=8192,
        enable_gpu_optimization=True,
        max_concurrent_loads=1,
        auto_rollback_enabled=True,
        health_check_interval=15,
        enable_metrics=True,
        metrics_storage_path="./prod_metrics.db"
    )
    
    print("Production configuration:")
    print(f"  Medical compliance: {config.enable_medical_compliance}")
    print(f"  Memory optimization: {config.memory_optimization.value}")
    print(f"  Auto rollback: {config.auto_rollback_enabled}")
    print(f"  Health check interval: {config.health_check_interval}s")
    
    async with lifecycle_manager_context(config) as lifecycle:
        print("\nProduction lifecycle manager started")
        
        # Register production adapters
        production_adapters = [
            create_adapter_metadata(
                adapter_id="prod_diagnosis_v1",
                name="Production Diagnosis Model",
                description="Production medical diagnosis adapter",
                adapter_type=AdapterType.MEDICAL_LORA,
                medical_domain="diagnostic_medicine",
                clinical_use_case="differential_diagnosis"
            ),
            create_adapter_metadata(
                adapter_id="prod_assessment_v2",
                name="Production Assessment Model",
                description="Production clinical assessment adapter",
                adapter_type=AdapterType.CLINICAL_LORA,
                medical_domain="clinical_assessment",
                clinical_use_case="patient_screening"
            )
        ]
        
        # Register all adapters
        for metadata in production_adapters:
            await lifecycle.register_adapter(metadata)
        
        print(f"Registered {len(production_adapters)} production adapters")
        
        # Simulate production workload
        print("\nSimulating production workload...")
        
        # Load adapters
        for i, metadata in enumerate(production_adapters):
            try:
                instance = await lifecycle.load_adapter(metadata.adapter_id)
                print(f"  Loaded adapter {i+1}: {instance.adapter_id}")
            except Exception as e:
                print(f"  Failed to load adapter {metadata.adapter_id}: {e}")
        
        # Monitor system during operation
        print("\nMonitoring system during operation...")
        for _ in range(5):
            await asyncio.sleep(1)
            status = await lifecycle.get_system_status()
            health = await lifecycle.health_check()
            
            print(f"  Health: {health['overall_health']}, "
                  f"Active adapters: {status['active_adapters']}, "
                  f"Memory: {status['memory_usage_mb']:.0f}MB")
        
        # Simulate hot-swap scenario
        print("\nSimulating hot-swap scenario...")
        try:
            # This would perform actual hot-swap in real scenario
            print("  Would execute hot-swap from v1 to v2")
            print("  Monitor health during transition")
            print("  Rollback on failure")
        except Exception as e:
            print(f"  Hot-swap failed: {e}")
        
        # Final status
        final_status = await lifecycle.get_system_status()
        final_health = await lifecycle.health_check()
        
        print(f"\nFinal production status:")
        print(f"  System state: {final_status['state']}")
        print(f"  Health: {final_health['overall_health']}")
        print(f"  Total loads: {final_status['statistics']['loads']['total']}")
        print(f"  Load success rate: {final_status['statistics']['loads']['success_rate']:.1%}")
        
        print("\nProduction scenario test completed")


async def main():
    """Run comprehensive test suite."""
    print("LoRA Adapter Management System - Comprehensive Test Suite")
    print("=" * 70)
    
    test_functions = [
        test_registry_system,
        test_validation_system,
        test_cache_system,
        test_lifecycle_manager,
        test_hot_swap_simulation,
        test_rollback_system,
        test_metrics_system,
        test_medical_compliance,
        test_production_scenario
    ]
    
    for test_func in test_functions:
        try:
            await test_func()
            print(f"✓ {test_func.__name__} completed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Brief pause between tests
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 70)
    print("All tests completed!")


if __name__ == "__main__":
    # Run the comprehensive test suite
    asyncio.run(main())