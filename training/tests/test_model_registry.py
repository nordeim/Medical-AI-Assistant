"""
Model Registry Tests

Comprehensive test suite for the model registry system including:
- Unit tests for core functionality
- Integration tests for MLflow and wandb
- Performance tests
- End-to-end workflow tests
"""

import pytest
import tempfile
import shutil
import json
import yaml
import sqlite3
import joblib
import pickle
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_registry import (
    ModelRegistry, ModelMetadata, ModelStage, ModelStatus,
    ModelComparison, register_sklearn_model, get_best_model
)
from utils.versioning import (
    SemanticVersion, VersionTracker, GitIntegration,
    ModelComparator, ChangeTracker
)


class TestModelRegistry:
    """Test ModelRegistry class"""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path"""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Create sample data
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        return model
    
    @pytest.fixture
    def registry(self, temp_registry_path):
        """Create registry instance"""
        return ModelRegistry(temp_registry_path)
    
    def test_registry_initialization(self, temp_registry_path):
        """Test registry initialization"""
        registry = ModelRegistry(temp_registry_path)
        
        assert registry.registry_path.exists()
        assert (registry.registry_path / "artifacts").exists()
        assert registry.db_path.exists()
        
        # Check database initialization
        with registry._get_db_connection() as conn:
            result = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [row[0] for row in result]
            assert 'models' in table_names
    
    def test_model_registration(self, registry, sample_model):
        """Test model registration"""
        model_id = registry.register_model(
            model=sample_model,
            name="test_model",
            version="1.0.0",
            description="Test model",
            performance_metrics={"accuracy": 0.85, "f1_score": 0.82},
            hyperparams={"n_estimators": 10}
        )
        
        assert model_id is not None
        assert model_id.startswith("test_model")
        
        # Check metadata was stored
        metadata = registry.get_model(model_id)
        assert metadata is not None
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test model"
        assert metadata.performance_metrics["accuracy"] == 0.85
        assert metadata.hyperparams["n_estimators"] == 10
        assert metadata.framework == "scikit-learn"
    
    def test_model_registration_with_auto_version(self, registry, sample_model):
        """Test model registration with auto-generated version"""
        model_id = registry.register_model(
            model=sample_model,
            name="test_model_v2",
            description="Test model with auto version"
        )
        
        metadata = registry.get_model(model_id)
        assert metadata.version is not None
        assert metadata.version == "0.1.0"  # Default initial version
    
    def test_get_latest_version(self, registry, sample_model):
        """Test getting latest version"""
        # Register multiple versions
        id1 = registry.register_model(sample_model, name="test_model")
        id2 = registry.register_model(sample_model, name="test_model")
        id3 = registry.register_model(sample_model, name="test_model")
        
        latest = registry.get_latest_version("test_model")
        assert latest is not None
        
        # Check that all versions are different
        metadata1 = registry.get_model(id1)
        metadata2 = registry.get_model(id2)
        metadata3 = registry.get_model(id3)
        
        versions = {metadata1.version, metadata2.version, metadata3.version}
        assert len(versions) == 3
    
    def test_list_models(self, registry, sample_model):
        """Test listing models"""
        # Register some models
        id1 = registry.register_model(sample_model, name="model1", stage=ModelStage.DEVELOPMENT)
        id2 = registry.register_model(sample_model, name="model2", stage=ModelStage.PRODUCTION)
        id3 = registry.register_model(sample_model, name="model3", stage=ModelStage.STAGING)
        
        # List all models
        all_models = registry.list_models()
        assert len(all_models) == 3
        
        # Filter by stage
        prod_models = registry.list_models(stage=ModelStage.PRODUCTION)
        assert len(prod_models) == 1
        assert prod_models[0].model_id == id2
        
        # Filter by name
        model1_models = registry.list_models(name="model1")
        assert len(model1_models) == 1
        assert model1_models[0].model_id == id1
        
        # Limit results
        limited_models = registry.list_models(limit=2)
        assert len(limited_models) == 2
    
    def test_load_model(self, registry, sample_model):
        """Test loading model from registry"""
        model_id = registry.register_model(
            model=sample_model,
            name="test_model"
        )
        
        # Load model
        loaded_model = registry.load_model(model_id)
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
        
        # Test prediction
        X_test = np.random.rand(10, 10)
        predictions_original = sample_model.predict(X_test)
        predictions_loaded = loaded_model.predict(X_test)
        
        np.testing.assert_array_equal(predictions_original, predictions_loaded)
    
    def test_update_model_stage(self, registry, sample_model):
        """Test updating model stage"""
        model_id = registry.register_model(sample_model, name="test_model")
        
        # Update stage
        success = registry.update_model_stage(model_id, ModelStage.PRODUCTION)
        assert success
        
        # Check stage was updated
        metadata = registry.get_model(model_id)
        assert metadata.stage == ModelStage.PRODUCTION
    
    def test_update_model_metrics(self, registry, sample_model):
        """Test updating model metrics"""
        model_id = registry.register_model(sample_model, name="test_model")
        
        # Update metrics
        new_metrics = {"accuracy": 0.90, "precision": 0.88}
        success = registry.update_model_metrics(model_id, new_metrics)
        assert success
        
        # Check metrics were updated
        metadata = registry.get_model(model_id)
        assert metadata.performance_metrics["accuracy"] == 0.90
        assert metadata.performance_metrics["precision"] == 0.88
    
    def test_compare_models(self, registry, sample_model):
        """Test model comparison"""
        # Register two models with different performance
        id1 = registry.register_model(
            sample_model, 
            name="model_a",
            performance_metrics={"accuracy": 0.85, "f1_score": 0.82}
        )
        
        # Create another model with better performance
        from sklearn.ensemble import RandomForestClassifier
        X, y = make_classification(n_samples=100, n_features=10, random_state=123)
        model_b = RandomForestClassifier(n_estimators=20, random_state=123)
        model_b.fit(X, y)
        
        id2 = registry.register_model(
            model_b,
            name="model_b", 
            performance_metrics={"accuracy": 0.92, "f1_score": 0.89}
        )
        
        # Compare models
        comparison = registry.compare_models(id1, id2)
        assert comparison is not None
        assert comparison.model_a_id == id1
        assert comparison.model_b_id == id2
        assert len(comparison.metric_comparisons) > 0
        
        # Check that model B is better
        assert comparison.winner == id2
        assert comparison.confidence_score > 0
    
    def test_promote_model(self, registry, sample_model):
        """Test model promotion"""
        model_id = registry.register_model(
            sample_model,
            name="test_model",
            performance_metrics={"accuracy": 0.90, "f1_score": 0.85}
        )
        
        # Promote to staging
        success = registry.promote_model(
            model_id,
            ModelStage.STAGING,
            requirements={"min_accuracy": 0.85}
        )
        assert success
        
        # Check promotion
        metadata = registry.get_model(model_id)
        assert metadata.stage == ModelStage.STAGING
        
        # Promote to production
        success = registry.promote_model(
            model_id,
            ModelStage.PRODUCTION,
            requirements={"min_accuracy": 0.85}
        )
        assert success
        
        metadata = registry.get_model(model_id)
        assert metadata.stage == ModelStage.PRODUCTION
        assert metadata.status == ModelStatus.DEPLOYED
    
    def test_promotion_requirements(self, registry, sample_model):
        """Test model promotion with requirements"""
        model_id = registry.register_model(
            sample_model,
            name="test_model",
            performance_metrics={"accuracy": 0.70}  # Low accuracy
        )
        
        # Try to promote with high accuracy requirement
        success = registry.promote_model(
            model_id,
            ModelStage.PRODUCTION,
            requirements={"min_accuracy": 0.85}
        )
        assert not success  # Should fail due to low accuracy
        
        metadata = registry.get_model(model_id)
        assert metadata.stage != ModelStage.PRODUCTION
    
    def test_rollback_model(self, registry, sample_model):
        """Test model rollback"""
        # Register a production model
        id1 = registry.register_model(
            sample_model,
            name="model_v1",
            performance_metrics={"accuracy": 0.85},
            stage=ModelStage.PRODUCTION
        )
        
        # Register a new model
        id2 = registry.register_model(
            sample_model,
            name="model_v2",
            performance_metrics={"accuracy": 0.90},
            stage=ModelStage.PRODUCTION
        )
        
        # Rollback to v1
        success = registry.rollback_model(id2)
        assert success
        
        # Check that v1 is production and v2 is archived
        metadata1 = registry.get_model(id1)
        metadata2 = registry.get_model(id2)
        
        assert metadata1.stage == ModelStage.PRODUCTION
        assert metadata2.stage == ModelStage.ARCHIVED
    
    def test_archive_model(self, registry, sample_model):
        """Test model archiving"""
        model_id = registry.register_model(sample_model, name="test_model")
        
        success = registry.archive_model(model_id)
        assert success
        
        metadata = registry.get_model(model_id)
        assert metadata.stage == ModelStage.ARCHIVED
    
    def test_delete_model(self, registry, sample_model):
        """Test model deletion"""
        model_id = registry.register_model(sample_model, name="test_model")
        
        # Delete non-production model
        success = registry.delete_model(model_id)
        assert success
        
        # Check model is deleted
        metadata = registry.get_model(model_id)
        assert metadata is None
    
    def test_delete_production_model(self, registry, sample_model):
        """Test preventing deletion of production models"""
        model_id = registry.register_model(
            sample_model, 
            name="test_model",
            stage=ModelStage.PRODUCTION
        )
        
        # Try to delete without force
        success = registry.delete_model(model_id, force=False)
        assert not success
        
        # Delete with force
        success = registry.delete_model(model_id, force=True)
        assert success
    
    def test_search_models(self, registry, sample_model):
        """Test model search"""
        # Register models with descriptions
        id1 = registry.register_model(
            sample_model,
            name="fraud_detector",
            description="Model for detecting fraudulent transactions"
        )
        id2 = registry.register_model(
            sample_model,
            name="sentiment_analyzer",
            description="Model for analyzing customer sentiment"
        )
        id3 = registry.register_model(
            sample_model,
            name="risk_assessor",
            description="Model for assessing credit risk"
        )
        
        # Search for "fraud"
        results = registry.search_models("fraud")
        assert len(results) == 1
        assert results[0].model_id == id1
        
        # Search for "sentiment"
        results = registry.search_models("sentiment")
        assert len(results) == 1
        assert results[0].model_id == id2
        
        # Search for "model"
        results = registry.search_models("model")
        assert len(results) == 3
    
    def test_export_registry(self, registry, sample_model):
        """Test registry export"""
        # Register some models
        registry.register_model(sample_model, name="model1")
        registry.register_model(sample_model, name="model2")
        
        # Export to temporary file
        export_path = Path(registry.registry_path) / "export.json"
        success = registry.export_registry(str(export_path))
        assert success
        assert export_path.exists()
        
        # Verify export content
        with open(export_path) as f:
            export_data = json.load(f)
        
        assert 'export_timestamp' in export_data
        assert 'models' in export_data
        assert len(export_data['models']) == 2
    
    def test_get_registry_stats(self, registry, sample_model):
        """Test registry statistics"""
        # Register models in different stages
        registry.register_model(sample_model, name="model1", stage=ModelStage.DEVELOPMENT)
        registry.register_model(sample_model, name="model2", stage=ModelStage.PRODUCTION)
        registry.register_model(sample_model, name="model3", stage=ModelStage.STAGING)
        
        stats = registry.get_registry_stats()
        
        assert stats['total_models'] == 3
        assert stats['by_stage']['development'] == 1
        assert stats['by_stage']['production'] == 1
        assert stats['by_stage']['staging'] == 1
        assert stats['by_status']['registered'] == 3
        assert stats['by_framework']['scikit-learn'] == 3


class TestSemanticVersion:
    """Test SemanticVersion class"""
    
    def test_valid_version_parsing(self):
        """Test parsing valid semantic versions"""
        versions = [
            "1.0.0",
            "1.2.3",
            "10.20.30",
            "1.0.0-alpha",
            "1.0.0-alpha.1",
            "1.0.0-0.3.7",
            "1.0.0-x.7.z.92",
            "1.0.0+20130313144700",
            "1.0.0-beta+exp.sha.5114f85",
            "1.0.0+20130313144700+exp.sha.5114f85"
        ]
        
        for version_str in versions:
            version = SemanticVersion(version_str)
            assert str(version) == version_str
    
    def test_invalid_version_parsing(self):
        """Test parsing invalid semantic versions"""
        invalid_versions = [
            "1.0",
            "1",
            "v1.0.0",
            "1.0.0-",
            "1.0.0+",
            "1.0.0- +build"
        ]
        
        for version_str in invalid_versions:
            with pytest.raises(ValueError):
                SemanticVersion(version_str)
    
    def test_version_comparison(self):
        """Test version comparison operators"""
        v1 = SemanticVersion("1.0.0")
        v2 = SemanticVersion("1.0.1")
        v3 = SemanticVersion("1.1.0")
        v4 = SemanticVersion("2.0.0")
        v5 = SemanticVersion("1.0.0")
        
        assert v1 < v2
        assert v1 < v3
        assert v1 < v4
        assert v1 == v5
        assert v2 > v1
        assert v3 > v1
        assert v4 > v1
    
    def test_prerelease_comparison(self):
        """Test prerelease version comparison"""
        v1 = SemanticVersion("1.0.0-alpha")
        v2 = SemanticVersion("1.0.0-alpha.1")
        v3 = SemanticVersion("1.0.0-beta")
        v4 = SemanticVersion("1.0.0")
        
        assert v1 < v2
        assert v2 < v3
        assert v3 < v4
        assert v4 > v3
    
    def test_version_bumping(self):
        """Test version bumping methods"""
        v = SemanticVersion("1.2.3")
        
        assert str(v.bump_major()) == "2.0.0"
        assert str(v.bump_minor()) == "1.3.0"
        assert str(v.bump_patch()) == "1.2.4"
        
        v2 = v.add_prerelease("beta")
        assert str(v2) == "1.2.3-beta"
        
        v3 = v.add_build("build.123")
        assert str(v3) == "1.2.3+build.123"


class TestVersionTracker:
    """Test VersionTracker class"""
    
    @pytest.fixture
    def version_tracker(self):
        """Create version tracker"""
        return VersionTracker()
    
    def test_parse_version(self, version_tracker):
        """Test version parsing"""
        version = version_tracker.parse_version("1.2.3")
        assert version is not None
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
    
    def test_increment_version(self, version_tracker):
        """Test version incrementing"""
        # No current version
        new_version = version_tracker.increment_version(None)
        assert new_version == "0.1.0"
        
        # Increment patch
        new_version = version_tracker.increment_version("1.2.3", "patch")
        assert new_version == "1.2.4"
        
        # Increment minor
        new_version = version_tracker.increment_version("1.2.3", "minor")
        assert new_version == "1.3.0"
        
        # Increment major
        new_version = version_tracker.increment_version("1.2.3", "major")
        assert new_version == "2.0.0"
        
        # Invalid increment type
        new_version = version_tracker.increment_version("1.2.3", "invalid")
        assert new_version == "1.2.3"
    
    def test_compare_versions(self, version_tracker):
        """Test version comparison"""
        assert version_tracker.compare_versions("1.0.0", "1.0.1") == -1
        assert version_tracker.compare_versions("1.0.1", "1.0.0") == 1
        assert version_tracker.compare_versions("1.0.0", "1.0.0") == 0
    
    def test_sort_versions(self, version_tracker):
        """Test version sorting"""
        versions = ["1.2.3", "1.0.0", "1.1.0", "2.0.0"]
        sorted_versions = version_tracker.sort_versions(versions)
        
        assert sorted_versions == ["1.0.0", "1.1.0", "1.2.3", "2.0.0"]
        
        # Test reverse sorting
        sorted_versions = version_tracker.sort_versions(versions, reverse=True)
        assert sorted_versions == ["2.0.0", "1.2.3", "1.1.0", "1.0.0"]


class TestModelComparator:
    """Test ModelComparator class"""
    
    @pytest.fixture
    def comparator(self):
        """Create model comparator"""
        return ModelComparator()
    
    def test_compare_versions(self, comparator):
        """Test version comparison"""
        comparison = comparator.compare_versions("1.0.0", "2.0.0")
        
        assert comparison['version_a'] == "1.0.0"
        assert comparison['version_b'] == "2.0.0"
        assert comparison['relationship'] == "older"
        assert comparison['changes']['major_changed'] == True
        assert comparison['changes']['change_type'] == "major"
    
    def test_compatibility_check(self, comparator):
        """Test compatibility checking"""
        # Backward compatible
        assert comparator.is_compatible_change("1.0.0", "1.1.0", "backward") == True
        assert comparator.is_compatible_change("1.0.0", "1.0.1", "backward") == True
        
        # Not backward compatible
        assert comparator.is_compatible_change("1.0.0", "2.0.0", "backward") == False
        
        # Fully compatible
        assert comparator.is_compatible_change("1.0.0", "1.0.0", "both") == True
        assert comparator.is_compatible_change("1.0.0", "1.0.1", "both") == False
    
    def test_suggest_version_bump(self, comparator):
        """Test version bump suggestions"""
        # Breaking changes
        increment, version = comparator.suggest_version_bump(
            ["Remove backward compatibility"], "1.0.0"
        )
        assert increment == "major"
        assert version == "2.0.0"
        
        # Feature additions
        increment, version = comparator.suggest_version_bump(
            ["Add new feature"], "1.0.0"
        )
        assert increment == "minor"
        assert version == "1.1.0"
        
        # Bug fixes
        increment, version = comparator.suggest_version_bump(
            ["Fix bug in prediction"], "1.0.0"
        )
        assert increment == "patch"
        assert version == "1.0.1"


class TestChangeTracker:
    """Test ChangeTracker class"""
    
    @pytest.fixture
    def temp_tracker_path(self):
        """Create temporary change tracker path"""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    @pytest.fixture
    def change_tracker(self, temp_tracker_path):
        """Create change tracker"""
        return ChangeTracker(temp_tracker_path)
    
    def test_record_change(self, change_tracker):
        """Test recording changes"""
        change_tracker.record_change(
            change_type="feature",
            description="Added new model type",
            version="1.1.0",
            metadata={"author": "test_user"}
        )
        
        changes = change_tracker.get_changes()
        assert len(changes) == 1
        assert changes[0]['type'] == "feature"
        assert changes[0]['description'] == "Added new model type"
        assert changes[0]['version'] == "1.1.0"
        assert changes[0]['metadata']['author'] == "test_user"
    
    def test_get_changes_by_version(self, change_tracker):
        """Test getting changes by version"""
        change_tracker.record_change("feature", "Feature A", "1.0.0")
        change_tracker.record_change("bugfix", "Bug fix B", "1.0.1")
        change_tracker.record_change("feature", "Feature C", "1.1.0")
        
        v1_changes = change_tracker.get_changes("1.0.0")
        assert len(v1_changes) == 1
        assert v1_changes[0]['description'] == "Feature A"
        
        v1_1_changes = change_tracker.get_changes("1.1.0")
        assert len(v1_1_changes) == 1
        assert v1_1_changes[0]['description'] == "Feature C"
    
    def test_get_change_summary(self, change_tracker):
        """Test getting change summary"""
        change_tracker.record_change("feature", "Feature A", "1.0.0")
        change_tracker.record_change("bugfix", "Bug fix B", "1.0.1")
        change_tracker.record_change("feature", "Feature C", "1.0.1")
        change_tracker.record_change("feature", "Feature D", "1.1.0")
        
        summary = change_tracker.get_change_summary()
        
        assert summary['total_changes'] == 4
        assert summary['by_version']['1.0.0'] == 1
        assert summary['by_version']['1.0.1'] == 2
        assert summary['by_version']['1.1.0'] == 1
        assert summary['by_type']['feature'] == 3
        assert summary['by_type']['bugfix'] == 1


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path"""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    def test_register_sklearn_model(self, temp_registry_path, sample_sklearn_model):
        """Test sklearn model registration convenience function"""
        model_id = register_sklearn_model(
            model=sample_sklearn_model,
            name="test_sklearn_model",
            registry_path=temp_registry_path
        )
        
        assert model_id is not None
        assert model_id.startswith("test_sklearn_model")
    
    def test_get_best_model(self, temp_registry_path, sample_sklearn_model):
        """Test getting best model convenience function"""
        # Register models with different performance
        register_sklearn_model(
            sample_sklearn_model,
            name="test_model",
            registry_path=temp_registry_path,
            performance_metrics={"accuracy": 0.85}
        )
        
        register_sklearn_model(
            sample_sklearn_model,
            name="test_model",
            registry_path=temp_registry_path,
            performance_metrics={"accuracy": 0.92}
        )
        
        # Get best model
        best_model_id = get_best_model(
            name="test_model",
            metric="accuracy",
            registry_path=temp_registry_path
        )
        
        assert best_model_id is not None
        
        # Load and verify it's the better model
        registry = ModelRegistry(temp_registry_path)
        metadata = registry.get_model(best_model_id)
        assert metadata.performance_metrics["accuracy"] == 0.92


class TestIntegration:
    """Integration tests"""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path"""
        path = tempfile.mkdtemp()
        yield path
        shutil.rmtree(path)
    
    def test_full_model_lifecycle(self, temp_registry_path, sample_model):
        """Test complete model lifecycle"""
        registry = ModelRegistry(temp_registry_path)
        
        # 1. Register model
        model_id = registry.register_model(
            sample_model,
            name="lifecycle_test_model",
            version="0.1.0",
            performance_metrics={"accuracy": 0.75}
        )
        
        # 2. Update metrics
        registry.update_model_metrics(model_id, {"accuracy": 0.80})
        
        # 3. Promote to staging
        registry.promote_model(
            model_id,
            ModelStage.STAGING,
            requirements={"min_accuracy": 0.75}
        )
        
        # 4. Validate and promote to production
        registry.promote_model(
            model_id,
            ModelStage.PRODUCTION,
            requirements={"min_accuracy": 0.75}
        )
        
        # 5. Register new version
        new_model_id = registry.register_model(
            sample_model,
            name="lifecycle_test_model",
            version="0.2.0",
            performance_metrics={"accuracy": 0.85}
        )
        
        # 6. Promote new version
        registry.promote_model(
            new_model_id,
            ModelStage.PRODUCTION,
            requirements={"min_accuracy": 0.75}
        )
        
        # 7. Rollback to old version
        registry.rollback_model(new_model_id)
        
        # Verify final state
        metadata = registry.get_model(model_id)
        assert metadata.stage == ModelStage.PRODUCTION
        
        new_metadata = registry.get_model(new_model_id)
        assert new_metadata.stage == ModelStage.ARCHIVED
    
    def test_model_comparison_workflow(self, temp_registry_path):
        """Test model comparison workflow"""
        registry = ModelRegistry(temp_registry_path)
        
        # Create models with different performance
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=200, n_features=20, random_state=42)
        
        rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
        rf_model.fit(X, y)
        
        gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb_model.fit(X, y)
        
        # Register models
        rf_id = registry.register_model(
            rf_model,
            name="comparison_model",
            performance_metrics={"accuracy": 0.82, "f1_score": 0.80}
        )
        
        gb_id = registry.register_model(
            gb_model,
            name="comparison_model",
            performance_metrics={"accuracy": 0.85, "f1_score": 0.83}
        )
        
        # Compare models
        comparison = registry.compare_models(rf_id, gb_id)
        
        assert comparison is not None
        assert comparison.winner == gb_id
        assert comparison.confidence_score > 0
        
        # Promote better model
        registry.promote_model(gb_id, ModelStage.PRODUCTION)
        
        # Verify promotion
        gb_metadata = registry.get_model(gb_id)
        assert gb_metadata.stage == ModelStage.PRODUCTION


# Utility function to create sample sklearn model
def make_sample_sklearn_model():
    """Create sample sklearn model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


# Pytest fixture for sample model
@pytest.fixture
def sample_sklearn_model():
    """Create sample sklearn model"""
    return make_sample_sklearn_model()


if __name__ == "__main__":
    pytest.main([__file__])