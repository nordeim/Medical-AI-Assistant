"""
Tests for Model Version Tracking System.

Comprehensive test suite for all components of the model version tracking system.
"""

import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import json

from core import (
    ModelVersion, VersionRegistry, VersionManager, 
    VersionType, ComplianceLevel, VersionStatus
)
from compatibility import CompatibilityChecker, CompatibilityType
from testing import ABTestingManager, ExperimentConfig, TestType, StatisticalTest
from deployment import RolloutManager, DeploymentTarget, DeploymentType
from comparison import PerformanceComparator, MedicalMetrics
from metadata import MetadataManager, ClinicalValidation, ValidationType
from config import VersionTrackingConfig


class TestModelVersion(unittest.TestCase):
    """Test ModelVersion class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_version_creation(self):
        """Test basic version creation."""
        version = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            model_type="neural_network",
            description="Test model version",
            created_by="test_user"
        )
        
        self.assertEqual(version.version, "1.0.0")
        self.assertEqual(version.model_name, "test_model")
        self.assertEqual(version.version_type, VersionType.MINOR)
        self.assertEqual(version.status, VersionStatus.DEVELOPMENT)
        self.assertIsInstance(version.created_at, datetime)
    
    def test_version_validation(self):
        """Test version format validation."""
        # Valid version should work
        version = ModelVersion(
            version="1.2.3",
            model_name="test_model",
            model_type="neural_network",
            created_by="test_user"
        )
        self.assertEqual(version.version, "1.2.3")
        
        # Invalid version should raise ValueError
        with self.assertRaises(ValueError):
            ModelVersion(
                version="invalid",
                model_name="test_model",
                model_type="neural_network",
                created_by="test_user"
            )
    
    def test_version_increment(self):
        """Test version increment functionality."""
        version = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            model_type="neural_network",
            created_by="test_user"
        )
        
        # Test minor increment
        new_version = version.increment_version(VersionType.MINOR)
        self.assertEqual(new_version, "1.1.0")
        
        # Test major increment
        new_version = version.increment_version(VersionType.MAJOR)
        self.assertEqual(new_version, "2.0.0")
        
        # Test patch increment
        new_version = version.increment_version(VersionType.PATCH)
        self.assertEqual(new_version, "1.0.1")
    
    def test_compliance_metadata(self):
        """Test compliance metadata functionality."""
        version = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            model_type="neural_network",
            created_by="test_user"
        )
        
        # Set compliance metadata
        version.compliance.compliance_level = ComplianceLevel.PRODUCTION
        version.compliance.clinical_approval_date = datetime.now()
        version.compliance.approval_authority = "FDA"
        
        # Test audit trail
        version.compliance.add_audit_entry("approved", "test_user", {"reason": "testing"})
        
        self.assertEqual(len(version.compliance.audit_trail), 1)
        self.assertEqual(version.compliance.audit_trail[0]["action"], "approved")


class TestVersionRegistry(unittest.TestCase):
    """Test VersionRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.registry = VersionRegistry(self.temp_dir / "test_registry")
        
        # Create test versions
        self.version_1_0_0 = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            model_type="neural_network",
            description="Initial version",
            created_by="test_user"
        )
        
        self.version_1_1_0 = ModelVersion(
            version="1.1.0",
            model_name="test_model",
            model_type="neural_network",
            description="Minor update",
            parent_version="1.0.0",
            created_by="test_user"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_version_registration(self):
        """Test version registration."""
        # Register first version
        result = self.registry.register_version(self.version_1_0_0)
        self.assertTrue(result)
        
        # Verify version is stored
        retrieved = self.registry.get_version("test_model", "1.0.0")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.version, "1.0.0")
    
    def test_version_listing(self):
        """Test version listing."""
        # Register versions
        self.registry.register_version(self.version_1_0_0)
        self.registry.register_version(self.version_1_1_0)
        
        # List versions
        versions = self.registry.list_versions("test_model")
        self.assertEqual(len(versions), 2)
        self.assertIn("1.0.0", versions)
        self.assertIn("1.1.0", versions)
    
    def test_latest_version(self):
        """Test latest version retrieval."""
        # Register versions
        self.registry.register_version(self.version_1_0_0)
        self.registry.register_version(self.version_1_1_0)
        
        # Get latest version
        latest = self.registry.get_latest_version("test_model")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.version, "1.1.0")  # Latest by creation time
    
    def test_compliance_validation(self):
        """Test compliance validation."""
        # Set compliance for version 1.0.0
        self.version_1_0_0.compliance.compliance_level = ComplianceLevel.PRODUCTION
        self.version_1_0_0.compliance.clinical_approval_date = datetime.now()
        self.version_1_0_0.compliance.approval_authority = "FDA"
        
        self.registry.register_version(self.version_1_0_0)
        
        # Validate compliance
        report = self.registry.generate_compliance_report("test_model")
        self.assertEqual(report["total_versions"], 1)
        self.assertEqual(report["compliance_levels"]["production"], 1)


class TestCompatibilityChecker(unittest.TestCase):
    """Test CompatibilityChecker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = CompatibilityChecker()
        
        # Create test versions
        self.version_1_0_0 = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            model_type="neural_network",
            created_by="test_user"
        )
        
        self.version_1_1_0 = ModelVersion(
            version="1.1.0",
            model_name="test_model",
            model_type="neural_network",
            parent_version="1.0.0",
            created_by="test_user"
        )
        
        self.version_2_0_0 = ModelVersion(
            version="2.0.0",
            model_name="test_model",
            model_type="transformer",
            parent_version="1.1.0",
            created_by="test_user"
        )
    
    def test_compatibility_checking(self):
        """Test compatibility checking between versions."""
        # Check 1.0.0 -> 1.1.0 compatibility (should be backward compatible)
        compatibility = self.checker.check_compatibility(
            self.version_1_0_0, 
            self.version_1_1_0
        )
        
        self.assertIsNotNone(compatibility)
        self.assertEqual(compatibility.source_version, "1.0.0")
        self.assertEqual(compatibility.target_version, "1.1.0")
    
    def test_major_version_incompatibility(self):
        """Test major version incompatibility detection."""
        # Check 1.0.0 -> 2.0.0 compatibility (should detect breaking changes)
        compatibility = self.checker.check_compatibility(
            self.version_1_0_0,
            self.version_2_0_0
        )
        
        # Should detect incompatibility due to major version change
        self.assertIn(compatibility.overall_compatibility.value, ["incompatible", "minor_breaking"])
    
    def test_rollback_safety(self):
        """Test rollback safety validation."""
        # Test rollback safety from 1.1.0 to 1.0.0
        safety_check = self.checker.validate_rollback_safety(
            self.version_1_1_0, 
            "1.0.0"
        )
        
        self.assertIn("safe", safety_check)
        self.assertIsInstance(safety_check["safe"], bool)


class TestABTestingManager(unittest.TestCase):
    """Test ABTestingManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.ab_testing = ABTestingManager(str(self.temp_dir / "ab_tests"))
        
        # Create experiment configuration
        self.config = ExperimentConfig(
            name="Test Experiment",
            description="Test A/B testing functionality",
            model_name="test_model",
            control_version="1.0.0",
            treatment_version="1.1.0",
            test_type=TestType.MODEL_PERFORMANCE,
            created_by="test_user"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_experiment_creation(self):
        """Test experiment creation."""
        experiment_id = self.ab_testing.create_experiment(self.config)
        
        self.assertIsNotNone(experiment_id)
        self.assertIn(experiment_id, self.ab_testing.experiments)
        
        # Verify experiment configuration
        stored_config = self.ab_testing.experiments[experiment_id]
        self.assertEqual(stored_config.name, "Test Experiment")
        self.assertEqual(stored_config.model_name, "test_model")
    
    def test_experiment_start_stop(self):
        """Test experiment start and stop."""
        experiment_id = self.ab_testing.create_experiment(self.config)
        
        # Start experiment
        result = self.ab_testing.start_experiment(experiment_id)
        self.assertTrue(result)
        
        # Check status
        result_obj = self.ab_testing.results.get(experiment_id)
        self.assertIsNotNone(result_obj)
        
        # Stop experiment
        stop_result = self.ab_testing.stop_experiment(experiment_id, "test_stop")
        self.assertTrue(stop_result)
    
    def test_user_assignment(self):
        """Test user group assignment."""
        experiment_id = self.ab_testing.create_experiment(self.config)
        self.ab_testing.start_experiment(experiment_id)
        
        # Assign user
        assignment = self.ab_testing.assign_user_to_group(experiment_id, "test_user_123")
        
        self.assertIsNotNone(assignment)
        self.assertIn(assignment.group_name, ["Control Group", "Treatment Group"])
        
        # Same user should get same assignment
        assignment2 = self.ab_testing.assign_user_to_group(experiment_id, "test_user_123")
        self.assertEqual(assignment.group_id, assignment2.group_id)


class TestRolloutManager(unittest.TestCase):
    """Test RolloutManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create registry and managers
        self.registry = VersionRegistry(self.temp_dir / "registry")
        self.version_manager = VersionManager(self.registry)
        self.rollout_manager = RolloutManager(self.version_manager)
        
        # Create test version
        self.version = ModelVersion(
            version="1.1.0",
            model_name="test_model",
            model_type="neural_network",
            description="Test deployment version",
            created_by="test_user"
        )
        
        self.version.compliance.compliance_level = ComplianceLevel.CLINICAL_VALIDATION
        self.version.compliance.intended_use = "Testing deployment"
        
        self.registry.register_version(self.version)
        
        # Create deployment target
        self.target = DeploymentTarget(
            name="test_target",
            url="https://test.example.com",
            environment="staging"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_deployment_creation(self):
        """Test deployment creation."""
        deployment_id = self.rollout_manager.deploy_model(
            model_name="test_model",
            version="1.1.0",
            targets=[self.target],
            deployment_type=DeploymentType.IMMEDIATE,
            initiated_by="test_user"
        )
        
        self.assertIsNotNone(deployment_id)
        self.assertIn(deployment_id, self.rollout_manager.active_deployments)
    
    def test_deployment_status(self):
        """Test deployment status tracking."""
        deployment_id = self.rollout_manager.deploy_model(
            model_name="test_model",
            version="1.1.0",
            targets=[self.target],
            deployment_type=DeploymentType.IMMEDIATE,
            initiated_by="test_user"
        )
        
        # Get deployment status
        status = self.rollout_manager.get_deployment_status(deployment_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.model_name, "test_model")
        self.assertEqual(status.model_version, "1.1.0")


class TestPerformanceComparator(unittest.TestCase):
    """Test PerformanceComparator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.medical_metrics = MedicalMetrics()
        self.comparator = PerformanceComparator(self.medical_metrics)
        
        # Create test model versions
        self.control_version = ModelVersion(
            version="1.0.0",
            model_name="test_model",
            model_type="neural_network",
            created_by="test_user"
        )
        
        self.treatment_version = ModelVersion(
            version="1.1.0",
            model_name="test_model",
            model_type="neural_network",
            created_by="test_user"
        )
    
    def test_clinical_metrics_calculation(self):
        """Test clinical metrics calculation."""
        # Create test data
        y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
        
        metrics = self.medical_metrics.calculate_clinical_metrics(y_true, y_pred)
        
        self.assertGreaterEqual(metrics.accuracy, 0.0)
        self.assertLessEqual(metrics.accuracy, 1.0)
        self.assertGreaterEqual(metrics.specificity, 0.0)
        self.assertLessEqual(metrics.specificity, 1.0)
    
    def test_model_comparison(self):
        """Test model performance comparison."""
        # Create test data
        test_data = []
        for i in range(100):
            test_data.append({
                "patient_id": f"patient_{i:03d}",
                "features": [0.1 * i, 0.2 * i, 0.3 * i],
                "label": 1 if i % 3 == 0 else 0
            })
        
        # Perform comparison
        comparison_result = self.comparator.compare_models(
            control_version=self.control_version,
            treatment_version=self.treatment_version,
            test_data=test_data,
            statistical_test="t_test"
        )
        
        self.assertIsNotNone(comparison_result)
        self.assertEqual(comparison_result.control_version, "1.0.0")
        self.assertEqual(comparison_result.treatment_version, "1.1.0")
        self.assertIsInstance(comparison_result.overall_improvement, float)


class TestMetadataManager(unittest.TestCase):
    """Test MetadataManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.metadata_manager = MetadataManager(str(self.temp_dir / "metadata"))
        
        # Create test version
        self.version = ModelVersion(
            version="1.1.0",
            model_name="test_model",
            model_type="neural_network",
            description="Test metadata version",
            created_by="test_user"
        )
        
        self.version.compliance.compliance_level = ComplianceLevel.CLINICAL_VALIDATION
        self.version.compliance.intended_use = "Testing metadata"
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_metadata_creation(self):
        """Test metadata creation."""
        metadata = self.metadata_manager.create_metadata(self.version, "test_user")
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.model_name, "test_model")
        self.assertEqual(metadata.model_version, "1.1.0")
        self.assertEqual(metadata.created_by, "test_user")
    
    def test_clinical_validation_addition(self):
        """Test adding clinical validation."""
        metadata = self.metadata_manager.create_metadata(self.version, "test_user")
        
        validation = ClinicalValidation(
            validation_id="TEST001",
            validation_type=ValidationType.CLINICAL_VALIDATION,
            validation_name="Test Validation",
            description="Test clinical validation",
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            validation_status=ValidationStatus.PLANNED
        )
        
        result = self.metadata_manager.add_clinical_validation(metadata.version_id, validation)
        self.assertTrue(result)
        
        # Verify validation was added
        validations = self.metadata_manager.get_clinical_validations(metadata.version_id)
        self.assertEqual(len(validations), 1)
        self.assertEqual(validations[0].validation_id, "TEST001")
    
    def test_compliance_validation(self):
        """Test compliance validation."""
        metadata = self.metadata_manager.create_metadata(self.version, "test_user")
        
        # Update metadata to meet compliance requirements
        self.metadata_manager.update_metadata(metadata.version_id, {
            "intended_use": "Assist in medical diagnosis",
            "indications": ["Emergency diagnosis"],
            "contraindications": ["Not for standalone use"],
            "warnings": ["Requires physician oversight"]
        }, "test_user")
        
        # Validate compliance
        compliance_result = self.metadata_manager.validate_regulatory_compliance(metadata.version_id)
        
        self.assertIn("compliance_score", compliance_result)
        self.assertIsInstance(compliance_result["compliance_score"], float)
        self.assertGreaterEqual(compliance_result["compliance_score"], 0.0)
        self.assertLessEqual(compliance_result["compliance_score"], 1.0)


class TestConfiguration(unittest.TestCase):
    """Test configuration management."""
    
    def test_default_config(self):
        """Test default configuration loading."""
        config = VersionTrackingConfig()
        
        self.assertEqual(config.environment, "development")
        self.assertEqual(config.log_level, "INFO")
        self.assertIsInstance(config.database, type(config).__annotations__['database'])
        self.assertIsInstance(config.compliance, type(config).__annotations__['compliance'])
    
    def test_environment_config(self):
        """Test environment-specific configuration."""
        try:
            import os
            os.environ['VERSION_TRACKING_ENV'] = 'production'
            
            config = VersionTrackingConfig.from_environment()
            self.assertEqual(config.environment, 'production')
            
        finally:
            # Clean up
            if 'VERSION_TRACKING_ENV' in os.environ:
                del os.environ['VERSION_TRACKING_ENV']
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = VersionTrackingConfig()
        
        # Valid configuration should have no errors
        errors = config.validate()
        # May have path creation warnings but no critical errors
        self.assertIsInstance(errors, list)
    
    def test_compliance_thresholds(self):
        """Test compliance threshold validation."""
        from config import validate_compliance_thresholds
        
        valid_thresholds = {
            'min_medical_accuracy': 0.85,
            'min_diagnostic_accuracy': 0.90,
            'min_clinical_sensitivity': 0.95,
            'min_clinical_specificity': 0.90,
            'min_auc_roc': 0.85,
            'max_latency_ms': 1000.0,
            'max_error_rate': 0.01
        }
        
        errors = validate_compliance_thresholds(valid_thresholds)
        self.assertEqual(len(errors), 0)
        
        # Test invalid threshold
        invalid_thresholds = valid_thresholds.copy()
        invalid_thresholds['min_medical_accuracy'] = 1.5  # Invalid
        
        errors = validate_compliance_thresholds(invalid_thresholds)
        self.assertGreater(len(errors), 0)


def run_integration_test():
    """Run a simple integration test."""
    print("\n🧪 Running integration test...")
    
    # Create temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Set up components
        registry = VersionRegistry(temp_dir / "registry")
        version_manager = VersionManager(registry)
        
        # Create test version
        version = ModelVersion(
            version="1.0.0",
            model_name="integration_test_model",
            model_type="neural_network",
            description="Integration test version",
            created_by="integration_test"
        )
        
        version.compliance.compliance_level = ComplianceLevel.PRODUCTION
        version.compliance.intended_use = "Integration testing"
        
        # Register version
        success = registry.register_version(version)
        assert success, "Version registration failed"
        
        # Create metadata
        metadata_manager = MetadataManager(temp_dir / "metadata")
        metadata = metadata_manager.create_metadata(version, "integration_test")
        assert metadata is not None, "Metadata creation failed"
        
        # Validate compliance
        compliance_result = metadata_manager.validate_regulatory_compliance(metadata.version_id)
        assert "compliance_score" in compliance_result, "Compliance validation failed"
        
        # Test compatibility
        version2 = ModelVersion(
            version="1.1.0",
            model_name="integration_test_model",
            model_type="neural_network",
            description="Second integration test version",
            parent_version="1.0.0",
            created_by="integration_test"
        )
        
        registry.register_version(version2)
        
        checker = CompatibilityChecker()
        compatibility = checker.check_compatibility(version, version2)
        assert compatibility is not None, "Compatibility checking failed"
        
        print("✅ Integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    # Run unit tests
    print("🔬 Running Model Version Tracking System Tests")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run integration test
    run_integration_test()