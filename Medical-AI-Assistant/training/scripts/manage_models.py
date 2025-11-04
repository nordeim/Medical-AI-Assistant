#!/usr/bin/env python3
"""
Model Management CLI

Command-line interface for managing machine learning models in the registry.
Supports bulk operations, automated workflows, and status monitoring.
"""

import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model_registry import ModelRegistry, ModelStage, ModelStatus, register_sklearn_model
from utils.versioning import SemanticVersion, VersionTracker, generate_version_from_git


class ModelManagerCLI:
    """Command-line interface for model management"""
    
    def __init__(self, registry_path: str, config_path: Optional[str] = None):
        """Initialize CLI"""
        self.registry = ModelRegistry(registry_path, self._load_config(config_path))
        self.version_tracker = VersionTracker()
        self.logger = self._setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        if not config_path:
            return {}
        
        try:
            with open(config_path) as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("ModelManagerCLI")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def register_model(
        self,
        model_path: str,
        name: str,
        version: Optional[str] = None,
        stage: str = 'development',
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        framework: Optional[str] = None,
        **kwargs
    ) -> str:
        """Register a model"""
        try:
            # Load model from file
            model = self._load_model_from_path(model_path, framework)
            if model is None:
                self.logger.error(f"Failed to load model from {model_path}")
                return ""
            
            # Generate version if not provided
            if version is None:
                version = self.version_tracker.generate_version_from_git('patch')
            
            # Register model
            model_id = self.registry.register_model(
                model=model,
                name=name,
                version=version,
                stage=ModelStage(stage),
                description=description,
                tags=tags,
                performance_metrics=performance_metrics,
                hyperparams=hyperparams,
                **kwargs
            )
            
            self.logger.info(f"Model registered successfully: {model_id}")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Failed to register model: {e}")
            return ""
    
    def _load_model_from_path(self, model_path: str, framework: Optional[str]) -> Any:
        """Load model from file path"""
        import joblib
        
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                self.logger.error(f"Model file not found: {model_path}")
                return None
            
            # Try to load with joblib
            if model_file.suffix == '.joblib':
                return joblib.load(model_file)
            else:
                # Assume pickle format
                import pickle
                with open(model_file, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None
    
    def list_models(
        self,
        name: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        output_format: str = 'table'
    ) -> bool:
        """List models"""
        try:
            # Convert string parameters to enums
            stage_enum = ModelStage(stage) if stage else None
            status_enum = ModelStatus(status) if status else None
            
            models = self.registry.list_models(
                name=name,
                stage=stage_enum,
                status=status_enum,
                limit=limit
            )
            
            if not models:
                print("No models found matching criteria.")
                return True
            
            if output_format == 'table':
                self._print_models_table(models)
            elif output_format == 'json':
                print(json.dumps([{
                    'model_id': m.model_id,
                    'name': m.name,
                    'version': m.version,
                    'stage': m.stage.value,
                    'status': m.status.value,
                    'framework': m.framework,
                    'created_at': m.created_at.isoformat(),
                    'description': m.description
                } for m in models], indent=2))
            elif output_format == 'csv':
                self._print_models_csv(models)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return False
    
    def _print_models_table(self, models: List):
        """Print models in table format"""
        print(f"\n{'Model ID':<20} {'Name':<15} {'Version':<10} {'Stage':<12} {'Status':<10} {'Framework':<12} {'Created':<20}")
        print("-" * 120)
        
        for model in models:
            created_str = model.created_at.strftime('%Y-%m-%d %H:%M')
            print(f"{model.model_id:<20} {model.name:<15} {model.version:<10} "
                  f"{model.stage.value:<12} {model.status.value:<10} "
                  f"{model.framework:<12} {created_str:<20}")
    
    def _print_models_csv(self, models: List):
        """Print models in CSV format"""
        print("model_id,name,version,stage,status,framework,created_at,description")
        
        for model in models:
            print(f"{model.model_id},{model.name},{model.version},{model.stage.value},"
                  f"{model.status.value},{model.framework},{model.created_at.isoformat()},"
                  f'"{model.description or ""}"')
    
    def get_model(self, model_id: str, output_format: str = 'json') -> bool:
        """Get model details"""
        try:
            model = self.registry.get_model(model_id)
            if not model:
                print(f"Model not found: {model_id}")
                return False
            
            if output_format == 'json':
                model_dict = {
                    'model_id': model.model_id,
                    'name': model.name,
                    'version': model.version,
                    'stage': model.stage.value,
                    'status': model.status.value,
                    'framework': model.framework,
                    'created_at': model.created_at.isoformat(),
                    'updated_at': model.updated_at.isoformat(),
                    'created_by': model.created_by,
                    'description': model.description,
                    'tags': model.tags,
                    'performance_metrics': model.performance_metrics,
                    'hyperparams': model.hyperparams,
                    'artifact_path': model.artifact_path,
                    'model_size_bytes': model.model_size_bytes
                }
                print(json.dumps(model_dict, indent=2))
            elif output_format == 'yaml':
                model_dict = {
                    'model_id': model.model_id,
                    'name': model.name,
                    'version': model.version,
                    'stage': model.stage.value,
                    'status': model.status.value,
                    'framework': model.framework,
                    'created_at': model.created_at.isoformat(),
                    'updated_at': model.updated_at.isoformat(),
                    'created_by': model.created_by,
                    'description': model.description,
                    'tags': model.tags,
                    'performance_metrics': model.performance_metrics,
                    'hyperparams': model.hyperparams,
                    'artifact_path': model.artifact_path,
                    'model_size_bytes': model.model_size_bytes
                }
                print(yaml.dump(model_dict, default_flow_style=False))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to get model: {e}")
            return False
    
    def load_model(self, model_id: str, output_path: str) -> bool:
        """Load and save model from registry"""
        try:
            model = self.registry.load_model(model_id)
            if model is None:
                print(f"Model not found: {model_id}")
                return False
            
            output_file = Path(output_path)
            import joblib
            
            try:
                joblib.dump(model, output_file)
            except Exception:
                import pickle
                with open(output_file, 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"Model saved to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def update_stage(self, model_id: str, new_stage: str) -> bool:
        """Update model stage"""
        try:
            stage = ModelStage(new_stage)
            success = self.registry.update_model_stage(model_id, stage)
            
            if success:
                print(f"Model {model_id} stage updated to {new_stage}")
            else:
                print(f"Failed to update model {model_id} stage")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update stage: {e}")
            return False
    
    def update_metrics(self, model_id: str, metrics_file: str) -> bool:
        """Update model performance metrics"""
        try:
            with open(metrics_file) as f:
                if metrics_file.endswith('.yaml') or metrics_file.endswith('.yml'):
                    metrics = yaml.safe_load(f)
                else:
                    metrics = json.load(f)
            
            success = self.registry.update_model_metrics(model_id, metrics)
            
            if success:
                print(f"Model {model_id} metrics updated")
            else:
                print(f"Failed to update model {model_id} metrics")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
            return False
    
    def promote_model(
        self,
        model_id: str,
        target_stage: str,
        requirements_file: Optional[str] = None
    ) -> bool:
        """Promote model to target stage"""
        try:
            requirements = None
            if requirements_file:
                with open(requirements_file) as f:
                    if requirements_file.endswith('.yaml') or requirements_file.endswith('.yml'):
                        requirements = yaml.safe_load(f)
                    else:
                        requirements = json.load(f)
            
            success = self.registry.promote_model(
                model_id, ModelStage(target_stage), requirements
            )
            
            if success:
                print(f"Model {model_id} promoted to {target_stage}")
            else:
                print(f"Failed to promote model {model_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to promote model: {e}")
            return False
    
    def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
        metrics_file: Optional[str] = None
    ) -> bool:
        """Compare two models"""
        try:
            metrics_to_compare = None
            if metrics_file:
                with open(metrics_file) as f:
                    if metrics_file.endswith('.yaml') or metrics_file.endswith('.yml'):
                        metrics_to_compare = yaml.safe_load(f)
                    else:
                        metrics_to_compare = json.load(f)
            
            comparison = self.registry.compare_models(
                model_a_id, model_b_id, metrics_to_compare
            )
            
            if comparison is None:
                print("Failed to compare models")
                return False
            
            print(f"\nModel Comparison: {model_a_id} vs {model_b_id}")
            print("-" * 60)
            
            for metric, comp in comparison.metric_comparisons.items():
                print(f"{metric}:")
                print(f"  Model A: {comp['model_a']:.4f}")
                print(f"  Model B: {comp['model_b']:.4f}")
                print(f"  Difference: {comp['difference']:.4f}")
                print(f"  Improvement: {comp['improvement']:.2f}%")
                print()
            
            if comparison.winner:
                print(f"Winner: {comparison.winner}")
                print(f"Confidence: {comparison.confidence_score:.2f}")
            else:
                print("No clear winner")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compare models: {e}")
            return False
    
    def rollback_model(self, model_id: str, target_version: Optional[str] = None) -> bool:
        """Rollback model"""
        try:
            success = self.registry.rollback_model(model_id, target_version)
            
            if success:
                if target_version:
                    print(f"Model {model_id} rolled back to version {target_version}")
                else:
                    print(f"Model {model_id} rolled back to previous production version")
            else:
                print(f"Failed to rollback model {model_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to rollback model: {e}")
            return False
    
    def archive_model(self, model_id: str) -> bool:
        """Archive model"""
        try:
            success = self.registry.archive_model(model_id)
            
            if success:
                print(f"Model {model_id} archived")
            else:
                print(f"Failed to archive model {model_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to archive model: {e}")
            return False
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete model"""
        try:
            if not force:
                confirm = input(f"Are you sure you want to delete model {model_id}? [y/N]: ")
                if confirm.lower() != 'y':
                    print("Deletion cancelled")
                    return True
            
            success = self.registry.delete_model(model_id, force)
            
            if success:
                print(f"Model {model_id} deleted")
            else:
                print(f"Failed to delete model {model_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete model: {e}")
            return False
    
    def search_models(self, query: str, limit: Optional[int] = None) -> bool:
        """Search models"""
        try:
            models = self.registry.search_models(query)
            
            if limit:
                models = models[:limit]
            
            if not models:
                print(f"No models found matching query: {query}")
                return True
            
            print(f"\nFound {len(models)} models matching '{query}':")
            print("-" * 80)
            
            for model in models:
                print(f"{model.model_id} - {model.name} v{model.version} "
                      f"({model.stage.value}, {model.status.value})")
                if model.description:
                    print(f"  {model.description}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to search models: {e}")
            return False
    
    def get_stats(self) -> bool:
        """Get registry statistics"""
        try:
            stats = self.registry.get_registry_stats()
            
            print("Registry Statistics:")
            print("=" * 50)
            print(f"Total Models: {stats['total_models']}")
            print(f"Recent Models (7 days): {stats['recent_models']}")
            
            print("\nBy Stage:")
            for stage, count in stats['by_stage'].items():
                print(f"  {stage}: {count}")
            
            print("\nBy Status:")
            for status, count in stats['by_status'].items():
                print(f"  {status}: {count}")
            
            print("\nBy Framework:")
            for framework, count in stats['by_framework'].items():
                print(f"  {framework}: {count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return False
    
    def export_registry(self, output_path: str) -> bool:
        """Export registry"""
        try:
            success = self.registry.export_registry(output_path)
            
            if success:
                print(f"Registry exported to: {output_path}")
            else:
                print("Failed to export registry")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to export registry: {e}")
            return False
    
    def bulk_register(self, models_config: str) -> bool:
        """Bulk register models from configuration"""
        try:
            with open(models_config) as f:
                if models_config.endswith('.yaml') or models_config.endswith('.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            models = config.get('models', [])
            results = []
            
            for model_config in models:
                try:
                    model_id = self.register_model(**model_config)
                    results.append({
                        'config': model_config,
                        'success': bool(model_id),
                        'model_id': model_id
                    })
                    time.sleep(0.1)  # Small delay between registrations
                except Exception as e:
                    results.append({
                        'config': model_config,
                        'success': False,
                        'error': str(e)
                    })
            
            # Print results
            print(f"\nBulk Registration Results ({len(results)} models):")
            print("-" * 60)
            
            success_count = sum(1 for r in results if r['success'])
            print(f"Successful: {success_count}")
            print(f"Failed: {len(results) - success_count}")
            
            for result in results:
                status = "✓" if result['success'] else "✗"
                print(f"{status} {result['config'].get('name', 'Unknown')} - {result.get('model_id', result.get('error', 'Unknown error'))}")
            
            return success_count == len(results)
            
        except Exception as e:
            self.logger.error(f"Failed to bulk register models: {e}")
            return False
    
    def monitor_models(self, interval: int = 30, duration: int = 300) -> bool:
        """Monitor model status"""
        try:
            print(f"Monitoring models every {interval} seconds for {duration} seconds...")
            print("Press Ctrl+C to stop")
            print("-" * 60)
            
            start_time = time.time()
            last_stats = None
            
            while time.time() - start_time < duration:
                try:
                    stats = self.registry.get_registry_stats()
                    
                    # Clear screen (works on most terminals)
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"Model Registry Monitor - {timestamp}")
                    print("=" * 60)
                    print(f"Total Models: {stats['total_models']}")
                    print(f"Production Models: {stats['by_stage'].get('production', 0)}")
                    print(f"Training Models: {stats['by_stage'].get('development', 0)}")
                    
                    # Show status changes
                    if last_stats:
                        for status, count in stats['by_status'].items():
                            last_count = last_stats['by_status'].get(status, 0)
                            if count != last_count:
                                change = count - last_count
                                print(f"{status}: {count} ({change:+d})")
                    
                    last_stats = stats
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    print("\nMonitoring stopped")
                    break
                except Exception as e:
                    self.logger.error(f"Monitor error: {e}")
                    time.sleep(interval)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to monitor models: {e}")
            return False


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Model Registry Management CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register a model
  python manage_models.py register --model-path ./model.joblib --name my-model --version 1.0.0
  
  # List all models
  python manage_models.py list
  
  # List production models
  python manage_models.py list --stage production
  
  # Get model details
  python manage_models.py get model-id-123
  
  # Promote model to staging
  python manage_models.py promote model-id-123 --target-stage staging
  
  # Compare two models
  python manage_models.py compare model-a-id model-b-id
  
  # Export registry
  python manage_models.py export ./registry_export.json
  
  # Bulk register models
  python manage_models.py bulk-register ./models_config.json
  
  # Monitor registry
  python manage_models.py monitor --interval 30
        """
    )
    
    # Global arguments
    parser.add_argument('--registry-path', default='./model_registry',
                       help='Path to model registry directory')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Register command
    register_parser = subparsers.add_parser('register', help='Register a model')
    register_parser.add_argument('--model-path', required=True,
                                help='Path to model file')
    register_parser.add_argument('--name', required=True,
                                help='Model name')
    register_parser.add_argument('--version', help='Model version')
    register_parser.add_argument('--stage', default='development',
                                choices=['development', 'staging', 'production', 'archived'],
                                help='Model stage')
    register_parser.add_argument('--description', help='Model description')
    register_parser.add_argument('--tags', help='JSON string of tags')
    register_parser.add_argument('--metrics', help='JSON string of performance metrics')
    register_parser.add_argument('--hyperparams', help='JSON string of hyperparameters')
    register_parser.add_argument('--framework', help='ML framework name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--name', help='Filter by model name')
    list_parser.add_argument('--stage', choices=['development', 'staging', 'production', 'archived'],
                           help='Filter by stage')
    list_parser.add_argument('--status', choices=['registered', 'training', 'validating', 'deployed', 'failed', 'deprecated'],
                           help='Filter by status')
    list_parser.add_argument('--limit', type=int, help='Limit number of results')
    list_parser.add_argument('--output', choices=['table', 'json', 'csv'], default='table',
                           help='Output format')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get model details')
    get_parser.add_argument('model_id', help='Model ID')
    get_parser.add_argument('--output', choices=['json', 'yaml'], default='json',
                          help='Output format')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load model from registry')
    load_parser.add_argument('model_id', help='Model ID')
    load_parser.add_argument('--output', required=True, help='Output path')
    
    # Update commands
    update_stage_parser = subparsers.add_parser('update-stage', help='Update model stage')
    update_stage_parser.add_argument('model_id', help='Model ID')
    update_stage_parser.add_argument('--stage', required=True,
                                   choices=['development', 'staging', 'production', 'archived'],
                                   help='New stage')
    
    update_metrics_parser = subparsers.add_parser('update-metrics', help='Update model metrics')
    update_metrics_parser.add_argument('model_id', help='Model ID')
    update_metrics_parser.add_argument('--metrics-file', required=True,
                                     help='Path to metrics file (JSON or YAML)')
    
    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote model')
    promote_parser.add_argument('model_id', help='Model ID')
    promote_parser.add_argument('--target-stage', required=True,
                              choices=['development', 'staging', 'production', 'archived'],
                              help='Target stage')
    promote_parser.add_argument('--requirements', help='Requirements file (JSON or YAML)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('model_a_id', help='First model ID')
    compare_parser.add_argument('model_b_id', help='Second model ID')
    compare_parser.add_argument('--metrics', help='Metrics file (JSON or YAML)')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback model')
    rollback_parser.add_argument('model_id', help='Model ID')
    rollback_parser.add_argument('--version', help='Target version')
    
    # Archive command
    archive_parser = subparsers.add_parser('archive', help='Archive model')
    archive_parser.add_argument('model_id', help='Model ID')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete model')
    delete_parser.add_argument('model_id', help='Model ID')
    delete_parser.add_argument('--force', action='store_true', help='Force deletion')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search models')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--limit', type=int, help='Limit results')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get registry statistics')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export registry')
    export_parser.add_argument('output_path', help='Output file path')
    
    # Bulk register command
    bulk_register_parser = subparsers.add_parser('bulk-register', help='Bulk register models')
    bulk_register_parser.add_argument('config', help='Configuration file (JSON or YAML)')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Monitor registry')
    monitor_parser.add_argument('--interval', type=int, default=30,
                              help='Monitoring interval in seconds')
    monitor_parser.add_argument('--duration', type=int, default=300,
                              help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = ModelManagerCLI(args.registry_path, args.config)
    
    if args.verbose:
        cli.logger.setLevel(logging.DEBUG)
    
    # Execute command
    try:
        success = False
        
        if args.command == 'register':
            tags = json.loads(args.tags) if args.tags else None
            metrics = json.loads(args.metrics) if args.metrics else None
            hyperparams = json.loads(args.hyperparams) if args.hyperparams else None
            
            result = cli.register_model(
                model_path=args.model_path,
                name=args.name,
                version=args.version,
                stage=args.stage,
                description=args.description,
                tags=tags,
                performance_metrics=metrics,
                hyperparams=hyperparams,
                framework=args.framework
            )
            success = bool(result)
            
        elif args.command == 'list':
            success = cli.list_models(
                name=args.name,
                stage=args.stage,
                status=args.status,
                limit=args.limit,
                output_format=args.output
            )
            
        elif args.command == 'get':
            success = cli.get_model(args.model_id, args.output)
            
        elif args.command == 'load':
            success = cli.load_model(args.model_id, args.output)
            
        elif args.command == 'update-stage':
            success = cli.update_stage(args.model_id, args.stage)
            
        elif args.command == 'update-metrics':
            success = cli.update_metrics(args.model_id, args.metrics_file)
            
        elif args.command == 'promote':
            success = cli.promote_model(
                args.model_id, args.target_stage, args.requirements
            )
            
        elif args.command == 'compare':
            success = cli.compare_models(
                args.model_a_id, args.model_b_id, args.metrics
            )
            
        elif args.command == 'rollback':
            success = cli.rollback_model(args.model_id, args.version)
            
        elif args.command == 'archive':
            success = cli.archive_model(args.model_id)
            
        elif args.command == 'delete':
            success = cli.delete_model(args.model_id, args.force)
            
        elif args.command == 'search':
            success = cli.search_models(args.query, args.limit)
            
        elif args.command == 'stats':
            success = cli.get_stats()
            
        elif args.command == 'export':
            success = cli.export_registry(args.output_path)
            
        elif args.command == 'bulk-register':
            success = cli.bulk_register(args.config)
            
        elif args.command == 'monitor':
            success = cli.monitor_models(args.interval, args.duration)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        cli.logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()