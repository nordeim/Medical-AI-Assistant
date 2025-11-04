#!/usr/bin/env python3
"""
Model Registry Example and Validation Script

Demonstrates comprehensive usage of the model registry system including:
- Model registration and versioning
- Performance tracking and comparison
- Model promotion workflows
- Integration with MLflow and wandb
- A/B testing support
- Rollback capabilities
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import tempfile
import shutil
import warnings
import sys
warnings.filterwarnings('ignore')

# Add parent directory to path for direct imports
sys.path.append(str(Path(__file__).parent.parent))

# Import directly to avoid torch dependency issues
from utils.model_registry import (
    ModelRegistry, ModelStage, ModelStatus, 
    register_sklearn_model, get_best_model
)
from utils.versioning import SemanticVersion, VersionTracker


class ModelRegistryExample:
    """Comprehensive example and validation of model registry system"""
    
    def __init__(self, registry_path: str, config_path: str = None):
        """Initialize example with registry"""
        self.registry_path = registry_path
        self.config_path = config_path
        self.registry = None
        self.version_tracker = VersionTracker()
        self.examples_dir = Path(registry_path) / "examples"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
    
    def initialize_registry(self) -> bool:
        """Initialize the model registry"""
        try:
            print("🚀 Initializing Model Registry...")
            
            # Create registry
            config = None
            if self.config_path and Path(self.config_path).exists():
                import yaml
                with open(self.config_path) as f:
                    config = yaml.safe_load(f)
            
            self.registry = ModelRegistry(self.registry_path, config)
            
            print(f"✅ Registry initialized at: {self.registry_path}")
            print(f"   Database: {self.registry.db_path}")
            print(f"   Artifacts: {self.registry.registry_path / 'artifacts'}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize registry: {e}")
            return False
    
    def create_sample_models(self) -> Dict[str, str]:
        """Create sample models for demonstration"""
        print("\n🔬 Creating Sample Models...")
        
        # Import required libraries
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.datasets import make_classification
        
        models_created = {}
        
        # Create different datasets for different models
        datasets = {
            'balanced': (200, 20, 42),
            'imbalanced': (150, 15, 123),
            'high_dim': (100, 50, 456)
        }
        
        for dataset_name, (n_samples, n_features, random_state) in datasets.items():
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=min(n_features - 2, 10),
                n_redundant=2,
                random_state=random_state
            )
            
            # Random Forest Model
            rf_model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10,
                random_state=random_state
            )
            rf_model.fit(X, y)
            
            # Gradient Boosting Model
            gb_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=6,
                random_state=random_state
            )
            gb_model.fit(X, y)
            
            # Logistic Regression Model
            lr_model = LogisticRegression(
                max_iter=1000,
                random_state=random_state
            )
            lr_model.fit(X, y)
            
            # Store models and their test predictions
            models_created.update({
                f'rf_{dataset_name}': rf_model,
                f'gb_{dataset_name}': gb_model,
                f'lr_{dataset_name}': lr_model
            })
        
        print(f"✅ Created {len(models_created)} sample models")
        return models_created
    
    def register_models(self, models: Dict[str, Any]) -> Dict[str, str]:
        """Register models in the registry"""
        print("\n📋 Registering Models in Registry...")
        
        model_ids = {}
        
        for model_name, model in models.items():
            # Calculate performance metrics
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            # Get predictions for metrics calculation
            try:
                if hasattr(model, 'predict_proba'):
                    # For classifiers with predict_proba
                    X_pred = model.feature_names_in_[:100] if hasattr(model, 'feature_names_in_') else None
                else:
                    X_pred = None
                
                # Simple performance estimation
                # In practice, you'd use proper validation
                performance_metrics = {
                    "accuracy": np.random.uniform(0.75, 0.95),
                    "f1_score": np.random.uniform(0.70, 0.92),
                    "precision": np.random.uniform(0.70, 0.94),
                    "recall": np.random.uniform(0.68, 0.93)
                }
                
                # Add some variation for different model types
                if 'rf' in model_name:
                    performance_metrics["accuracy"] += 0.02
                elif 'gb' in model_name:
                    performance_metrics["f1_score"] += 0.01
                elif 'lr' in model_name:
                    performance_metrics["precision"] += 0.01
                
                # Generate hyperparameters
                hyperparams = {}
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    hyperparams = {k: v for k, v in list(params.items())[:5]}
                
                # Generate version
                base_version = "1.0.0" if "balanced" in model_name else "0.9.0"
                
                # Register model
                model_id = self.registry.register_model(
                    model=model,
                    name=f"medical_{model_name}",
                    version=base_version,
                    stage=ModelStage.DEVELOPMENT,
                    description=f"Medical AI model for {model_name.replace('_', ' ')} dataset",
                    tags={
                        "domain": "medical",
                        "model_type": model_name.split('_')[0],
                        "dataset": model_name.split('_')[1]
                    },
                    performance_metrics=performance_metrics,
                    hyperparams=hyperparams,
                    created_by="example_script",
                    data_lineage={
                        "dataset_type": model_name.split('_')[1],
                        "training_date": time.strftime("%Y-%m-%d"),
                        "data_version": "v1.0"
                    }
                )
                
                model_ids[model_name] = model_id
                print(f"   Registered: {model_name} -> {model_id[:20]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to register {model_name}: {e}")
        
        print(f"✅ Registered {len(model_ids)} models successfully")
        return model_ids
    
    def demonstrate_versioning(self, model_ids: Dict[str, str]) -> Dict[str, str]:
        """Demonstrate versioning functionality"""
        print("\n🔢 Demonstrating Versioning...")
        
        # Create new versions of some models
        new_versions = {}
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        for base_model_name in ['balanced_rf', 'imbalanced_gb']:
            if base_model_name in model_ids:
                # Create improved version
                X, y = make_classification(
                    n_samples=200,
                    n_features=20 if "balanced" in base_model_name else 15,
                    random_state=789
                )
                
                improved_model = RandomForestClassifier(
                    n_estimators=75,  # Increased from 50
                    max_depth=12,     # Increased from 10
                    random_state=789
                )
                improved_model.fit(X, y)
                
                # Generate next version
                latest_version = self.registry.get_latest_version(f"medical_{base_model_name}")
                next_version = self.version_tracker.increment_version(latest_version, 'minor')
                
                # Register improved version
                new_model_id = self.registry.register_model(
                    model=improved_model,
                    name=f"medical_{base_model_name}",
                    version=next_version,
                    stage=ModelStage.DEVELOPMENT,
                    description=f"Improved version of {base_model_name}",
                    performance_metrics={
                        "accuracy": np.random.uniform(0.82, 0.97),
                        "f1_score": np.random.uniform(0.78, 0.95),
                        "precision": np.random.uniform(0.80, 0.96),
                        "recall": np.random.uniform(0.79, 0.94)
                    },
                    hyperparams={
                        "n_estimators": 75,
                        "max_depth": 12,
                        "random_state": 789
                    },
                    created_by="example_script"
                )
                
                new_versions[base_model_name] = new_model_id
                print(f"   Created new version: {base_model_name} -> {next_version}")
        
        print(f"✅ Created {len(new_versions)} new versions")
        return new_versions
    
    def demonstrate_model_comparison(self, model_ids: Dict[str, str]) -> None:
        """Demonstrate model comparison functionality"""
        print("\n⚖️  Demonstrating Model Comparison...")
        
        # Find models that can be compared
        balanced_models = [name for name in model_ids.keys() if "balanced" in name]
        
        if len(balanced_models) >= 2:
            model_a_name = balanced_models[0]
            model_b_name = balanced_models[1]
            
            model_a_id = model_ids[model_a_name]
            model_b_id = model_ids[model_b_name]
            
            # Compare models
            comparison = self.registry.compare_models(model_a_id, model_b_id)
            
            if comparison:
                print(f"   Comparison: {model_a_name} vs {model_b_name}")
                print(f"   Winner: {comparison.winner}")
                print(f"   Confidence: {comparison.confidence_score:.3f}")
                print("   Metrics compared:")
                
                for metric, data in comparison.metric_comparisons.items():
                    improvement = data['improvement']
                    winner_symbol = "🏆" if improvement > 0 else "📉"
                    print(f"     {winner_symbol} {metric}: {data['model_a']:.3f} -> {data['model_b']:.3f} ({improvement:+.1f}%)")
            else:
                print("   ❌ Comparison failed")
        else:
            print("   ⚠️  Not enough models for comparison")
    
    def demonstrate_promotion_workflow(self, model_ids: Dict[str, str]) -> Dict[str, str]:
        """Demonstrate model promotion workflow"""
        print("\n🚀 Demonstrating Model Promotion Workflow...")
        
        promoted_models = {}
        
        # Select models for promotion
        models_to_promote = list(model_ids.items())[:3]  # Take first 3 models
        
        for model_name, model_id in models_to_promote:
            print(f"   Promoting: {model_name}")
            
            # Get current metadata
            metadata = self.registry.get_model(model_id)
            if not metadata:
                continue
            
            # Update metrics for promotion readiness
            performance_metrics = {
                "accuracy": metadata.performance_metrics.get("accuracy", 0.8) + 0.05,
                "f1_score": metadata.performance_metrics.get("f1_score", 0.75) + 0.03,
                "precision": metadata.performance_metrics.get("precision", 0.78) + 0.02,
                "recall": metadata.performance_metrics.get("recall", 0.76) + 0.03
            }
            
            # Update metrics
            self.registry.update_model_metrics(model_id, performance_metrics)
            
            # Promote to Staging
            staging_success = self.registry.promote_model(
                model_id,
                ModelStage.STAGING,
                requirements={
                    "min_accuracy": 0.80,
                    "min_f1": 0.75,
                    "validation_required": True
                }
            )
            
            if staging_success:
                print(f"     ✅ Promoted to STAGING")
                
                # Simulate validation period
                time.sleep(0.5)
                
                # Promote to Production
                production_success = self.registry.promote_model(
                    model_id,
                    ModelStage.PRODUCTION,
                    requirements={
                        "min_accuracy": 0.80,
                        "min_f1": 0.75,
                        "validation_required": True
                    }
                )
                
                if production_success:
                    print(f"     ✅ Promoted to PRODUCTION")
                    promoted_models[model_name] = model_id
                else:
                    print(f"     ❌ Failed to promote to PRODUCTION")
            else:
                print(f"     ❌ Failed to promote to STAGING")
        
        print(f"✅ {len(promoted_models)} models promoted to production")
        return promoted_models
    
    def demonstrate_ab_testing(self, model_ids: Dict[str, str]) -> None:
        """Demonstrate A/B testing workflow"""
        print("\n🧪 Demonstrating A/B Testing...")
        
        # Find production models for A/B testing
        production_models = []
        for model_name, model_id in model_ids.items():
            metadata = self.registry.get_model(model_id)
            if metadata and metadata.stage == ModelStage.PRODUCTION:
                production_models.append((model_name, model_id, metadata))
        
        if len(production_models) >= 2:
            print("   Setting up A/B test between production models...")
            
            # Select two models for A/B test
            model_a = production_models[0]
            model_b = production_models[1]
            
            print(f"   Control: {model_a[0]} (80% traffic)")
            print(f"   Treatment: {model_b[0]} (20% traffic)")
            
            # Simulate A/B test results
            test_duration = 7  # days
            sample_size_a = 10000
            sample_size_b = 2500
            
            accuracy_a = model_a[2].performance_metrics.get("accuracy", 0.85)
            accuracy_b = model_b[2].performance_metrics.get("accuracy", 0.87)
            
            print(f"   Test Results after {test_duration} days:")
            print(f"     Control model: {accuracy_a:.3f} accuracy ({sample_size_a} samples)")
            print(f"     Treatment model: {accuracy_b:.3f} accuracy ({sample_size_b} samples)")
            
            improvement = accuracy_b - accuracy_a
            if improvement > 0.01:  # 1% improvement threshold
                print(f"   🎉 Treatment model wins! ({improvement:+.1%} improvement)")
                
                # Promote treatment model
                success = self.registry.promote_model(
                    model_b[1],
                    ModelStage.PRODUCTION  # Already in production, but could update status
                )
                
                if success:
                    print("   ✅ Treatment model confirmed as production winner")
            else:
                print("   📊 No significant improvement detected")
        
        else:
            print("   ⚠️  Not enough production models for A/B testing")
    
    def demonstrate_search_and_filtering(self) -> None:
        """Demonstrate search and filtering capabilities"""
        print("\n🔍 Demonstrating Search and Filtering...")
        
        # Search for medical models
        print("   Searching for 'medical' models...")
        medical_models = self.registry.search_models("medical")
        print(f"   Found {len(medical_models)} medical models")
        
        # List all models by stage
        print("   Models by stage:")
        for stage in ModelStage:
            models_in_stage = self.registry.list_models(stage=stage)
            if models_in_stage:
                print(f"     {stage.value}: {len(models_in_stage)} models")
        
        # List models by framework
        print("   Models by framework:")
        all_models = self.registry.list_models()
        frameworks = {}
        for model in all_models:
            framework = model.framework
            if framework not in frameworks:
                frameworks[framework] = 0
            frameworks[framework] += 1
        
        for framework, count in frameworks.items():
            print(f"     {framework}: {count} models")
        
        # Filter by performance threshold
        print("   Models with accuracy > 0.85:")
        high_performance_models = [
            model for model in all_models
            if model.performance_metrics and 
            model.performance_metrics.get("accuracy", 0) > 0.85
        ]
        print(f"   Found {len(high_performance_models)} high-performance models")
    
    def demonstrate_rollback(self, model_ids: Dict[str, str]) -> None:
        """Demonstrate rollback functionality"""
        print("\n🔄 Demonstrating Model Rollback...")
        
        # Find a model with multiple versions
        model_versions = {}
        for model_name, model_id in model_ids.items():
            metadata = self.registry.get_model(model_id)
            if metadata:
                base_name = model_name
                if base_name not in model_versions:
                    model_versions[base_name] = []
                model_versions[base_name].append((metadata.version, model_id))
        
        # Find model with multiple versions
        multi_version_model = None
        for name, versions in model_versions.items():
            if len(versions) > 1:
                multi_version_model = (name, versions)
                break
        
        if multi_version_model:
            model_name, versions = multi_version_model
            print(f"   Found multi-version model: {model_name}")
            print(f"   Versions: {[v[0] for v in versions]}")
            
            # Rollback to previous version
            current_version = versions[-1][1]  # Latest version
            rollback_success = self.registry.rollback_model(current_version)
            
            if rollback_success:
                print("   ✅ Rollback successful")
                
                # Verify rollback
                current_metadata = self.registry.get_model(current_version)
                if current_metadata:
                    print(f"   Current stage: {current_metadata.stage.value}")
            else:
                print("   ❌ Rollback failed")
        else:
            print("   ⚠️  No multi-version models found for rollback demo")
    
    def demonstrate_best_model_selection(self) -> None:
        """Demonstrate best model selection"""
        print("\n🏆 Demonstrating Best Model Selection...")
        
        # Get best model for different metrics
        metrics = ["accuracy", "f1_score", "precision", "recall"]
        
        for metric in metrics:
            best_model_id = get_best_model(
                name="medical_balanced_rf",
                metric=metric,
                registry_path=self.registry_path
            )
            
            if best_model_id:
                metadata = self.registry.get_model(best_model_id)
                if metadata:
                    print(f"   Best by {metric}: {metadata.version} ({metadata.performance_metrics.get(metric, 0):.3f})")
            else:
                print(f"   Best by {metric}: Not found")
    
    def generate_statistics(self) -> Dict[str, Any]:
        """Generate and display registry statistics"""
        print("\n📊 Registry Statistics...")
        
        stats = self.registry.get_registry_stats()
        
        print(f"   Total Models: {stats['total_models']}")
        print(f"   Recent Models (7 days): {stats['recent_models']}")
        
        print("   Models by Stage:")
        for stage, count in stats['by_stage'].items():
            print(f"     {stage}: {count}")
        
        print("   Models by Status:")
        for status, count in stats['by_status'].items():
            print(f"     {status}: {count}")
        
        print("   Models by Framework:")
        for framework, count in stats['by_framework'].items():
            print(f"     {framework}: {count}")
        
        return stats
    
    def export_example_data(self) -> str:
        """Export example data for documentation"""
        print("\n💾 Exporting Example Data...")
        
        export_path = self.examples_dir / "example_export.json"
        success = self.registry.export_registry(str(export_path))
        
        if success:
            print(f"   ✅ Exported to: {export_path}")
        else:
            print("   ❌ Export failed")
        
        return str(export_path) if success else None
    
    def save_example_script(self) -> str:
        """Save this example script for reference"""
        script_path = self.examples_dir / "example_usage.py"
        
        example_code = '''#!/usr/bin/env python3
"""
Example Usage of Model Registry

This script demonstrates how to use the model registry system.
"""

from utils.model_registry import ModelRegistry, register_sklearn_model, get_best_model
from utils.model_registry import ModelStage, ModelStatus
from utils.versioning import VersionTracker
import joblib

# Initialize registry
registry = ModelRegistry("./model_registry")

# Create and register a model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=100, n_features=10, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Register model
model_id = registry.register_model(
    model=model,
    name="example_model",
    version="1.0.0",
    description="Example model",
    performance_metrics={"accuracy": 0.85},
    hyperparams={"n_estimators": 10}
)

print(f"Model registered with ID: {model_id}")

# Get model metadata
metadata = registry.get_model(model_id)
print(f"Model: {metadata.name} v{metadata.version}")

# Load model
loaded_model = registry.load_model(model_id)

# Make predictions
predictions = loaded_model.predict(X[:10])
print(f"Predictions: {predictions}")

# List all models
models = registry.list_models()
print(f"Total models in registry: {len(models)}")

# Promote model
from utils.model_registry import ModelStage
registry.promote_model(
    model_id,
    ModelStage.PRODUCTION,
    requirements={"min_accuracy": 0.80}
)

# Get best model
best_model_id = get_best_model(
    name="example_model",
    metric="accuracy",
    registry_path="./model_registry"
)

# Compare models (register another version first)
model_id_2 = registry.register_model(
    model,
    name="example_model",
    version="1.1.0"
)

comparison = registry.compare_models(model_id, model_id_2)
if comparison:
    print(f"Winner: {comparison.winner}")
'''
        
        with open(script_path, 'w') as f:
            f.write(example_code)
        
        print(f"   ✅ Example script saved to: {script_path}")
        return str(script_path)
    
    def run_complete_example(self) -> bool:
        """Run the complete example workflow"""
        print("🚀 Model Registry Complete Example")
        print("=" * 60)
        
        try:
            # 1. Initialize registry
            if not self.initialize_registry():
                return False
            
            # 2. Create sample models
            models = self.create_sample_models()
            
            # 3. Register models
            model_ids = self.register_models(models)
            
            if not model_ids:
                print("❌ No models were registered successfully")
                return False
            
            # 4. Demonstrate versioning
            new_versions = self.demonstrate_versioning(model_ids)
            
            # 5. Compare models
            self.demonstrate_model_comparison(model_ids)
            
            # 6. Demonstrate promotion workflow
            promoted_models = self.demonstrate_promotion_workflow(model_ids)
            
            # 7. Demonstrate A/B testing
            self.demonstrate_ab_testing(model_ids)
            
            # 8. Search and filtering
            self.demonstrate_search_and_filtering()
            
            # 9. Demonstrate rollback
            self.demonstrate_rollback(model_ids)
            
            # 10. Best model selection
            self.demonstrate_best_model_selection()
            
            # 11. Generate statistics
            stats = self.generate_statistics()
            
            # 12. Export data
            export_path = self.export_example_data()
            
            # 13. Save example script
            script_path = self.save_example_script()
            
            print("\n" + "=" * 60)
            print("✅ COMPLETE EXAMPLE FINISHED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Registry Location: {self.registry_path}")
            print(f"Total Models: {stats['total_models']}")
            print(f"Production Models: {stats['by_stage'].get('production', 0)}")
            
            if export_path:
                print(f"Export File: {export_path}")
            if script_path:
                print(f"Example Script: {script_path}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Example failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Model Registry Example and Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete example
  python model_registry_example.py
  
  # Run with custom registry path
  python model_registry_example.py --registry-path ./my_registry
  
  # Run with configuration
  python model_registry_example.py --config ./configs/registry_config.yaml
  
  # Clean registry before running
  python model_registry_example.py --clean
        """
    )
    
    parser.add_argument(
        "--registry-path", "-r",
        default="./model_registry_example",
        help="Path for the model registry"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to registry configuration file"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean registry before running (remove existing registry)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Clean registry if requested
    if args.clean:
        registry_path = Path(args.registry_path)
        if registry_path.exists():
            print(f"🧹 Cleaning existing registry at {registry_path}")
            shutil.rmtree(registry_path, ignore_errors=True)
    
    # Create example runner
    example = ModelRegistryExample(
        registry_path=args.registry_path,
        config_path=args.config
    )
    
    # Run the example
    success = example.run_complete_example()
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()