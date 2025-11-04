#!/usr/bin/env python3
"""
Model Registry Test Runner

Comprehensive test runner for the model registry system with:
- Unit tests
- Integration tests
- Performance benchmarks
- Validation tests
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import tempfile
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_registry import ModelRegistry, register_sklearn_model, get_best_model
from utils.versioning import SemanticVersion, VersionTracker


class ModelRegistryTester:
    """Test runner for model registry system"""
    
    def __init__(self, verbose: bool = False):
        """Initialize test runner"""
        self.verbose = verbose
        self.temp_dirs = []
    
    def __del__(self):
        """Cleanup temporary directories"""
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def create_temp_registry(self) -> str:
        """Create temporary registry for testing"""
        temp_dir = tempfile.mkdtemp(prefix="registry_test_")
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        print("Running unit tests...")
        start_time = time.time()
        
        try:
            # Run pytest
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(Path(__file__).parent / ".." / "tests" / "test_model_registry.py"),
                "-v", "--tb=short"
            ], capture_output=True, text=True)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "status": "passed",
                    "elapsed": elapsed,
                    "output": result.stdout,
                    "tests_run": self._extract_test_count(result.stdout)
                }
            else:
                return {
                    "status": "failed",
                    "elapsed": elapsed,
                    "error": result.stderr,
                    "output": result.stdout
                }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _extract_test_count(self, output: str) -> int:
        """Extract number of tests run from pytest output"""
        import re
        # Look for patterns like "X passed" or "X failed"
        patterns = [
            r"(\d+) passed",
            r"(\d+) failed",
            r"(\d+) warnings"
        ]
        
        total = 0
        for pattern in patterns:
            matches = re.findall(pattern, output)
            total += sum(int(m) for m in matches)
        
        return total
    
    def run_functional_tests(self) -> Dict[str, Any]:
        """Run functional tests"""
        print("Running functional tests...")
        start_time = time.time()
        
        test_results = {
            "model_registration": self._test_model_registration(),
            "versioning": self._test_versioning(),
            "model_promotion": self._test_model_promotion(),
            "model_comparison": self._test_model_comparison(),
            "model_search": self._test_model_search()
        }
        
        elapsed = time.time() - start_time
        
        passed_tests = sum(1 for result in test_results.values() if result.get("passed", False))
        total_tests = len(test_results)
        
        return {
            "status": "passed" if passed_tests == total_tests else "failed",
            "elapsed": elapsed,
            "tests_passed": passed_tests,
            "tests_total": total_tests,
            "results": test_results
        }
    
    def _test_model_registration(self) -> Dict[str, Any]:
        """Test model registration"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Create sample model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Register model
            model_id = registry.register_model(
                model=model,
                name="test_model",
                version="1.0.0",
                description="Test model",
                performance_metrics={"accuracy": 0.85},
                hyperparams={"n_estimators": 10}
            )
            
            if not model_id:
                return {"passed": False, "error": "Model registration failed"}
            
            # Verify registration
            metadata = registry.get_model(model_id)
            if not metadata:
                return {"passed": False, "error": "Model not found after registration"}
            
            if metadata.name != "test_model":
                return {"passed": False, "error": "Model metadata mismatch"}
            
            # Test model loading
            loaded_model = registry.load_model(model_id)
            if loaded_model is None:
                return {"passed": False, "error": "Model loading failed"}
            
            # Test prediction
            predictions = loaded_model.predict(X[:10])
            if len(predictions) != 10:
                return {"passed": False, "error": "Model prediction failed"}
            
            return {
                "passed": True,
                "model_id": model_id,
                "version": metadata.version,
                "performance_metrics": metadata.performance_metrics
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_versioning(self) -> Dict[str, Any]:
        """Test versioning functionality"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            tracker = VersionTracker()
            
            # Create sample model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Register multiple versions
            versions = []
            for i in range(3):
                model_id = registry.register_model(
                    model=model,
                    name="versioned_model"
                )
                versions.append(model_id)
            
            if len(versions) != 3:
                return {"passed": False, "error": "Multiple version registration failed"}
            
            # Check version uniqueness
            metadata_list = [registry.get_model(vid) for vid in versions]
            version_set = set(m.version for m in metadata_list)
            
            if len(version_set) != 3:
                return {"passed": False, "error": "Versions not unique"}
            
            # Test latest version retrieval
            latest = registry.get_latest_version("versioned_model")
            if latest is None:
                return {"passed": False, "error": "Latest version retrieval failed"}
            
            # Test semantic versioning
            v1 = tracker.parse_version("1.2.3")
            v2 = tracker.parse_version("1.2.4")
            v3 = tracker.parse_version("2.0.0")
            
            if not (v1 < v2 < v3):
                return {"passed": False, "error": "Semantic versioning comparison failed"}
            
            return {
                "passed": True,
                "versions": [registry.get_model(vid).version for vid in versions],
                "latest": latest,
                "semantic_versioning": "1.2.3 < 1.2.4 < 2.0.0"
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_model_promotion(self) -> Dict[str, Any]:
        """Test model promotion workflow"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Create sample model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Register model
            model_id = registry.register_model(
                model=model,
                name="promotion_test_model",
                performance_metrics={"accuracy": 0.85}
            )
            
            # Test promotion through stages
            from utils.model_registry import ModelStage
            
            # Development -> Staging
            success = registry.promote_model(
                model_id,
                ModelStage.STAGING,
                requirements={"min_accuracy": 0.80}
            )
            if not success:
                return {"passed": False, "error": "Promotion to staging failed"}
            
            metadata = registry.get_model(model_id)
            if metadata.stage != ModelStage.STAGING:
                return {"passed": False, "error": "Stage not updated to staging"}
            
            # Staging -> Production
            success = registry.promote_model(
                model_id,
                ModelStage.PRODUCTION,
                requirements={"min_accuracy": 0.80}
            )
            if not success:
                return {"passed": False, "error": "Promotion to production failed"}
            
            metadata = registry.get_model(model_id)
            if metadata.stage != ModelStage.PRODUCTION:
                return {"passed": False, "error": "Stage not updated to production"}
            
            return {
                "passed": True,
                "model_id": model_id,
                "final_stage": "production",
                "status": metadata.status.value
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_model_comparison(self) -> Dict[str, Any]:
        """Test model comparison functionality"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Create models with different performance
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=200, n_features=20, random_state=42)
            
            # Model A - Random Forest
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_model.fit(X, y)
            
            # Model B - Gradient Boosting
            gb_model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            gb_model.fit(X, y)
            
            # Register models
            rf_id = registry.register_model(
                rf_model,
                name="comparison_test_model",
                performance_metrics={"accuracy": 0.82, "f1_score": 0.80}
            )
            
            gb_id = registry.register_model(
                gb_model,
                name="comparison_test_model",
                performance_metrics={"accuracy": 0.85, "f1_score": 0.83}
            )
            
            # Compare models
            comparison = registry.compare_models(rf_id, gb_id)
            
            if comparison is None:
                return {"passed": False, "error": "Model comparison failed"}
            
            if comparison.winner != gb_id:
                return {"passed": False, "error": "Incorrect winner identified"}
            
            if comparison.confidence_score <= 0:
                return {"passed": False, "error": "Confidence score not positive"}
            
            return {
                "passed": True,
                "model_a_id": rf_id,
                "model_b_id": gb_id,
                "winner": comparison.winner,
                "confidence_score": comparison.confidence_score,
                "metrics_compared": list(comparison.metric_comparisons.keys())
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _test_model_search(self) -> Dict[str, Any]:
        """Test model search functionality"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Create and register models with descriptions
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Register models with different descriptions
            registry.register_model(
                model,
                name="fraud_model",
                description="Model for detecting fraudulent transactions"
            )
            
            registry.register_model(
                model,
                name="sentiment_model",
                description="Model for analyzing customer sentiment in reviews"
            )
            
            registry.register_model(
                model,
                name="risk_model",
                description="Model for assessing credit risk and loan approval"
            )
            
            # Test search
            fraud_results = registry.search_models("fraud")
            if len(fraud_results) != 1:
                return {"passed": False, "error": "Search for 'fraud' returned wrong number of results"}
            
            sentiment_results = registry.search_models("sentiment")
            if len(sentiment_results) != 1:
                return {"passed": False, "error": "Search for 'sentiment' returned wrong number of results"}
            
            all_results = registry.search_models("model")
            if len(all_results) != 3:
                return {"passed": False, "error": "Search for 'model' did not return all models"}
            
            return {
                "passed": True,
                "fraud_search_results": len(fraud_results),
                "sentiment_search_results": len(sentiment_results),
                "model_search_results": len(all_results)
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("Running performance tests...")
        start_time = time.time()
        
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Create sample model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Benchmark model registration
            registration_times = []
            for i in range(10):
                reg_start = time.time()
                model_id = registry.register_model(
                    model,
                    name=f"perf_test_model_{i}",
                    version=f"1.0.{i}"
                )
                registration_times.append(time.time() - reg_start)
            
            # Benchmark model retrieval
            retrieval_times = []
            for i in range(10):
                reg_start = time.time()
                metadata = registry.get_model(f"perf_test_model_{i}_1.0.{i}_latest")
                retrieval_times.append(time.time() - reg_start)
            
            # Benchmark model listing
            listing_times = []
            for i in range(10):
                list_start = time.time()
                models = registry.list_models()
                listing_times.append(time.time() - list_start)
            
            elapsed = time.time() - start_time
            
            return {
                "status": "completed",
                "elapsed": elapsed,
                "registration": {
                    "avg_time": sum(registration_times) / len(registration_times),
                    "min_time": min(registration_times),
                    "max_time": max(registration_times)
                },
                "retrieval": {
                    "avg_time": sum(retrieval_times) / len(retrieval_times),
                    "min_time": min(retrieval_times),
                    "max_time": max(retrieval_times)
                },
                "listing": {
                    "avg_time": sum(listing_times) / len(listing_times),
                    "min_time": min(listing_times),
                    "max_time": max(listing_times)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_validation_test(self) -> Dict[str, Any]:
        """Run system validation test"""
        print("Running system validation...")
        start_time = time.time()
        
        validation_results = {
            "dependencies": self._validate_dependencies(),
            "configuration": self._validate_configuration(),
            "registry_creation": self._validate_registry_creation(),
            "model_workflow": self._validate_model_workflow()
        }
        
        elapsed = time.time() - start_time
        
        passed_validations = sum(1 for result in validation_results.values() if result.get("passed", False))
        total_validations = len(validation_results)
        
        return {
            "status": "passed" if passed_validations == total_validations else "failed",
            "elapsed": elapsed,
            "validations_passed": passed_validations,
            "validations_total": total_validations,
            "results": validation_results
        }
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate required dependencies"""
        try:
            dependencies = {
                "sklearn": self._check_import("sklearn"),
                "numpy": self._check_import("numpy"),
                "sqlite3": self._check_import("sqlite3"),
                "json": self._check_import("json"),
                "pathlib": self._check_import("pathlib"),
                "datetime": self._check_import("datetime")
            }
            
            missing_deps = [dep for dep, available in dependencies.items() if not available]
            
            if missing_deps:
                return {
                    "passed": False,
                    "error": f"Missing dependencies: {missing_deps}"
                }
            
            return {
                "passed": True,
                "dependencies": dependencies
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _check_import(self, module_name: str) -> bool:
        """Check if module can be imported"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration file"""
        try:
            config_path = Path(__file__).parent.parent / "configs" / "registry_config.yaml"
            
            if not config_path.exists():
                return {"passed": False, "error": "Configuration file not found"}
            
            # Try to load configuration
            with open(config_path) as f:
                import yaml
                config = yaml.safe_load(f)
            
            required_sections = ["registry", "storage", "versioning", "mlflow", "wandb"]
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                return {
                    "passed": False,
                    "error": f"Missing configuration sections: {missing_sections}"
                }
            
            return {
                "passed": True,
                "config_sections": list(config.keys())
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _validate_registry_creation(self) -> Dict[str, Any]:
        """Validate registry creation"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Check directories
            dirs_created = [
                registry.registry_path.exists(),
                (registry.registry_path / "artifacts").exists(),
                registry.db_path.exists()
            ]
            
            if not all(dirs_created):
                return {"passed": False, "error": "Registry directories not created"}
            
            # Check database
            with registry._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
            
            if "models" not in tables:
                return {"passed": False, "error": "Models table not created"}
            
            return {
                "passed": True,
                "registry_path": registry_path,
                "database_tables": tables
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _validate_model_workflow(self) -> Dict[str, Any]:
        """Validate complete model workflow"""
        try:
            registry_path = self.create_temp_registry()
            registry = ModelRegistry(registry_path)
            
            # Create model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            from utils.model_registry import ModelStage
            
            X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Register model
            model_id = registry.register_model(
                model,
                name="validation_test_model",
                version="1.0.0",
                performance_metrics={"accuracy": 0.85}
            )
            
            if not model_id:
                return {"passed": False, "error": "Model registration failed"}
            
            # Load model
            loaded_model = registry.load_model(model_id)
            if loaded_model is None:
                return {"passed": False, "error": "Model loading failed"}
            
            # Promote model
            success = registry.promote_model(
                model_id,
                ModelStage.PRODUCTION,
                requirements={"min_accuracy": 0.80}
            )
            if not success:
                return {"passed": False, "error": "Model promotion failed"}
            
            # Get stats
            stats = registry.get_registry_stats()
            if stats["total_models"] != 1:
                return {"passed": False, "error": "Registry stats incorrect"}
            
            return {
                "passed": True,
                "workflow_steps": [
                    "model_registration",
                    "model_loading",
                    "model_promotion",
                    "stats_retrieval"
                ],
                "model_id": model_id
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Model Registry Test Runner")
    parser.add_argument("--test-type", choices=["all", "unit", "functional", "performance", "validation"],
                       default="all", help="Type of tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    print("Model Registry Test Runner")
    print("=" * 50)
    
    tester = ModelRegistryTester(verbose=args.verbose)
    
    results = {}
    
    if args.test_type in ["all", "unit"]:
        print("\n🔍 Running Unit Tests")
        results["unit_tests"] = tester.run_unit_tests()
    
    if args.test_type in ["all", "functional"]:
        print("\n🔧 Running Functional Tests")
        results["functional_tests"] = tester.run_functional_tests()
    
    if args.test_type in ["all", "performance"]:
        print("\n⚡ Running Performance Tests")
        results["performance_tests"] = tester.run_performance_tests()
    
    if args.test_type in ["all", "validation"]:
        print("\n✅ Running System Validation")
        results["validation"] = tester.run_validation_test()
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_passed = 0
    total_failed = 0
    
    for test_name, result in results.items():
        status = result.get("status", "unknown")
        print(f"{test_name}: {status.upper()}")
        
        if status == "passed":
            total_passed += 1
        else:
            total_failed += 1
            if args.verbose and "error" in result:
                print(f"  Error: {result['error']}")
    
    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit code
    exit_code = 0 if total_failed == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()