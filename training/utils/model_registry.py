"""
Model Registry Module

Provides comprehensive model registry functionality including:
- Model metadata storage and retrieval
- Version control and tracking
- Model artifact management
- Performance metrics storage
- Integration with MLflow and Weights & Biases
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import logging
from contextlib import contextmanager
import hashlib
import pickle
import joblib

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .versioning import SemanticVersion, VersionTracker


class ModelStage(Enum):
    """Model deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """Model operational status"""
    REGISTERED = "registered"
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    version: str
    stage: ModelStage
    status: ModelStatus
    framework: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    parent_model_id: Optional[str] = None
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    dependencies: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    artifact_path: Optional[str] = None
    model_size_bytes: Optional[int] = None
    training_time_seconds: Optional[float] = None
    hyperparams: Optional[Dict[str, Any]] = None
    schema_info: Optional[Dict[str, Any]] = None
    data_lineage: Optional[Dict[str, Any]] = None


@dataclass
class ModelComparison:
    """Model comparison results"""
    model_a_id: str
    model_b_id: str
    metric_comparisons: Dict[str, Dict[str, float]]
    winner: Optional[str] = None
    confidence_score: Optional[float] = None
    notes: Optional[str] = None


class ModelRegistry:
    """
    Comprehensive Model Registry System
    
    Features:
    - Version control with semantic versioning
    - Git-based tracking
    - Performance metrics storage
    - Model lineage tracking
    - Integration with MLflow and wandb
    - A/B testing support
    - Rollback capabilities
    """
    
    def __init__(self, registry_path: str, config: Optional[Dict] = None):
        """
        Initialize Model Registry
        
        Args:
            registry_path: Path to registry directory
            config: Optional configuration dictionary
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.config = config or {}
        self.logger = self._setup_logging()
        self.version_tracker = VersionTracker()
        
        # Initialize database
        self.db_path = self.registry_path / "registry.db"
        self._init_database()
        
        # Initialize MLflow integration
        if MLFLOW_AVAILABLE and self.config.get("mlflow", {}).get("enabled", True):
            self._init_mlflow()
        
        # Initialize wandb integration
        if WANDB_AVAILABLE and self.config.get("wandb", {}).get("enabled", True):
            self._init_wandb()
        
        self.logger.info(f"ModelRegistry initialized at {self.registry_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f"ModelRegistry.{id(self)}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _init_database(self):
        """Initialize SQLite database for metadata"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    tags TEXT,
                    parent_model_id TEXT,
                    git_commit TEXT,
                    git_branch TEXT,
                    dependencies TEXT,
                    performance_metrics TEXT,
                    artifact_path TEXT,
                    model_size_bytes INTEGER,
                    training_time_seconds REAL,
                    hyperparams TEXT,
                    schema_info TEXT,
                    data_lineage TEXT,
                    FOREIGN KEY (parent_model_id) REFERENCES models(model_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_name_version 
                ON models(name, version)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_stage 
                ON models(stage)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_models_status 
                ON models(status)
            """)
    
    def _init_mlflow(self):
        """Initialize MLflow integration"""
        if not MLFLOW_AVAILABLE:
            return
        
        mlflow_config = self.config.get("mlflow", {})
        tracking_uri = mlflow_config.get("tracking_uri", "sqlite:///mlflow.db")
        
        try:
            mlflow.set_tracking_uri(tracking_uri)
            self.mlflow_experiment = mlflow_config.get("experiment_name", "model_registry")
            mlflow.set_experiment(self.mlflow_experiment)
            self.logger.info("MLflow integration initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MLflow: {e}")
            self.mlflow_experiment = None
    
    def _init_wandb(self):
        """Initialize wandb integration"""
        if not WANDB_AVAILABLE:
            return
        
        wandb_config = self.config.get("wandb", {})
        project_name = wandb_config.get("project", "model_registry")
        
        try:
            # Initialize wandb with project name (no explicit init in constructor)
            self.wandb_project = project_name
            self.logger.info("WandB integration initialized")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_project = None
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def register_model(
        self,
        model: Any,
        name: str,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        created_by: str = "system",
        **kwargs
    ) -> str:
        """
        Register a model in the registry
        
        Args:
            model: The model object to register
            model_id: Optional custom model ID
            name: Model name
            version: Optional semantic version (auto-generated if not provided)
            stage: Model deployment stage
            description: Optional description
            tags: Optional tags
            performance_metrics: Optional performance metrics
            hyperparams: Optional hyperparameters
            created_by: Creator identifier
            **kwargs: Additional metadata
            
        Returns:
            Model ID
        """
        # Generate model ID if not provided
        if model_id is None:
            model_id = self._generate_model_id(name, version)
        
        # Generate version if not provided
        if version is None:
            latest_version = self.get_latest_version(name)
            version = self.version_tracker.increment_version(latest_version)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version,
            stage=stage,
            status=ModelStatus.REGISTERED,
            framework=self._detect_framework(model),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            created_by=created_by,
            description=description,
            tags=tags or {},
            performance_metrics=performance_metrics,
            hyperparams=hyperparams,
            **kwargs
        )
        
        # Store model artifact
        artifact_path = self._store_artifact(model, model_id)
        metadata.artifact_path = str(artifact_path)
        metadata.model_size_bytes = artifact_path.stat().st_size if artifact_path.exists() else None
        
        # Store in database
        self._store_metadata(metadata)
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and hasattr(self, 'mlflow_experiment'):
            self._log_to_mlflow(model, metadata)
        
        # Log to wandb
        if WANDB_AVAILABLE and hasattr(self, 'wandb_project'):
            self._log_to_wandb(model, metadata)
        
        self.logger.info(f"Model registered: {model_id} (v{version})")
        return model_id
    
    def _generate_model_id(self, name: str, version: Optional[str] = None) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_str = version or "latest"
        return f"{name}_{version_str}_{timestamp}"
    
    def _detect_framework(self, model: Any) -> str:
        """Detect ML framework from model type"""
        model_type = type(model).__module__
        
        framework_map = {
            'sklearn': 'scikit-learn',
            'torch': 'pytorch',
            'tensorflow': 'tensorflow',
            'keras': 'keras',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm'
        }
        
        for key, framework in framework_map.items():
            if key in model_type.lower():
                return framework
        
        return 'unknown'
    
    def _store_artifact(self, model: Any, model_id: str) -> Path:
        """Store model artifact"""
        artifacts_dir = self.registry_path / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        artifact_path = artifacts_dir / f"{model_id}.joblib"
        
        try:
            joblib.dump(model, artifact_path)
            return artifact_path
        except Exception:
            # Fallback to pickle
            artifact_path = artifacts_dir / f"{model_id}.pkl"
            with open(artifact_path, 'wb') as f:
                pickle.dump(model, f)
            return artifact_path
    
    def _store_metadata(self, metadata: ModelMetadata):
        """Store model metadata in database"""
        with self._get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO models (
                    model_id, name, version, stage, status, framework,
                    created_at, updated_at, created_by, description, tags,
                    parent_model_id, git_commit, git_branch, dependencies,
                    performance_metrics, artifact_path, model_size_bytes,
                    training_time_seconds, hyperparams, schema_info, data_lineage
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.model_id, metadata.name, metadata.version, 
                metadata.stage.value, metadata.status.value, metadata.framework,
                metadata.created_at, metadata.updated_at, metadata.created_by,
                metadata.description, json.dumps(metadata.tags or {}),
                metadata.parent_model_id, metadata.git_commit, metadata.git_branch,
                json.dumps(metadata.dependencies or {}),
                json.dumps(metadata.performance_metrics or {}),
                metadata.artifact_path, metadata.model_size_bytes,
                metadata.training_time_seconds,
                json.dumps(metadata.hyperparams or {}),
                json.dumps(metadata.schema_info or {}),
                json.dumps(metadata.data_lineage or {})
            ))
    
    def _log_to_mlflow(self, model: Any, metadata: ModelMetadata):
        """Log model to MLflow"""
        if not MLFLOW_AVAILABLE:
            return
        
        try:
            with mlflow.start_run(experiment_id=mlflow.get_experiment_by_name(
                self.mlflow_experiment).experiment_id, run_name=metadata.model_id):
                
                # Log parameters
                if metadata.hyperparams:
                    mlflow.log_params(metadata.hyperparams)
                
                # Log metrics
                if metadata.performance_metrics:
                    mlflow.log_metrics(metadata.performance_metrics)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                # Log tags
                mlflow.set_tags({
                    "model_id": metadata.model_id,
                    "stage": metadata.stage.value,
                    "framework": metadata.framework
                })
                
        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {e}")
    
    def _log_to_wandb(self, model: Any, metadata: ModelMetadata):
        """Log model to wandb"""
        if not WANDB_AVAILABLE:
            return
        
        try:
            with wandb.init(project=self.wandb_project, name=metadata.model_id):
                # Log parameters
                if metadata.hyperparams:
                    wandb.config.update(metadata.hyperparams)
                
                # Log metrics
                if metadata.performance_metrics:
                    for metric, value in metadata.performance_metrics.items():
                        wandb.log({metric: value})
                
                # Log model artifact
                artifact = wandb.Artifact(metadata.model_id, type="model")
                if hasattr(model, 'feature_names_in_'):
                    artifact.metadata["features"] = model.feature_names_in_.tolist()
                wandb.log_artifact(artifact)
                
        except Exception as e:
            self.logger.warning(f"Failed to log to wandb: {e}")
    
    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata by ID"""
        with self._get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            
            if row:
                return self._row_to_metadata(row)
            return None
    
    def get_model_by_name_version(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Retrieve model by name and version"""
        with self._get_db_connection() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE name = ? AND version = ?",
                (name, version)
            ).fetchone()
            
            if row:
                return self._row_to_metadata(row)
            return None
    
    def get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of a model"""
        with self._get_db_connection() as conn:
            row = conn.execute("""
                SELECT version FROM models 
                WHERE name = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (name,)).fetchone()
            
            return row['version'] if row else None
    
    def list_models(
        self,
        name: Optional[str] = None,
        stage: Optional[ModelStage] = None,
        status: Optional[ModelStatus] = None,
        limit: Optional[int] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering"""
        query = "SELECT * FROM models WHERE 1=1"
        params = []
        
        if name:
            query += " AND name = ?"
            params.append(name)
        
        if stage:
            query += " AND stage = ?"
            params.append(stage.value)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        with self._get_db_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_metadata(row) for row in rows]
    
    def _row_to_metadata(self, row) -> ModelMetadata:
        """Convert database row to ModelMetadata"""
        return ModelMetadata(
            model_id=row['model_id'],
            name=row['name'],
            version=row['version'],
            stage=ModelStage(row['stage']),
            status=ModelStatus(row['status']),
            framework=row['framework'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            created_by=row['created_by'],
            description=row['description'],
            tags=json.loads(row['tags']) if row['tags'] else None,
            parent_model_id=row['parent_model_id'],
            git_commit=row['git_commit'],
            git_branch=row['git_branch'],
            dependencies=json.loads(row['dependencies']) if row['dependencies'] else None,
            performance_metrics=json.loads(row['performance_metrics']) if row['performance_metrics'] else None,
            artifact_path=row['artifact_path'],
            model_size_bytes=row['model_size_bytes'],
            training_time_seconds=row['training_time_seconds'],
            hyperparams=json.loads(row['hyperparams']) if row['hyperparams'] else None,
            schema_info=json.loads(row['schema_info']) if row['schema_info'] else None,
            data_lineage=json.loads(row['data_lineage']) if row['data_lineage'] else None
        )
    
    def load_model(self, model_id: str) -> Optional[Any]:
        """Load model from registry"""
        metadata = self.get_model(model_id)
        if not metadata or not metadata.artifact_path:
            return None
        
        artifact_path = Path(metadata.artifact_path)
        if not artifact_path.exists():
            return None
        
        # Try joblib first, then pickle
        if artifact_path.suffix == '.joblib':
            try:
                return joblib.load(artifact_path)
            except Exception:
                pass
        
        with open(artifact_path, 'rb') as f:
            return pickle.load(f)
    
    def update_model_stage(
        self,
        model_id: str,
        new_stage: ModelStage,
        notes: Optional[str] = None
    ) -> bool:
        """Update model deployment stage"""
        metadata = self.get_model(model_id)
        if not metadata:
            return False
        
        metadata.stage = new_stage
        metadata.updated_at = datetime.now(timezone.utc)
        
        self._store_metadata(metadata)
        
        self.logger.info(f"Model {model_id} stage updated to {new_stage.value}")
        return True
    
    def update_model_metrics(
        self,
        model_id: str,
        metrics: Dict[str, float]
    ) -> bool:
        """Update model performance metrics"""
        metadata = self.get_model(model_id)
        if not metadata:
            return False
        
        metadata.performance_metrics = {
            **(metadata.performance_metrics or {}),
            **metrics
        }
        metadata.updated_at = datetime.now(timezone.utc)
        
        self._store_metadata(metadata)
        return True
    
    def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
        metrics_to_compare: Optional[List[str]] = None
    ) -> Optional[ModelComparison]:
        """Compare two models"""
        metadata_a = self.get_model(model_a_id)
        metadata_b = self.get_model(model_b_id)
        
        if not metadata_a or not metadata_b:
            return None
        
        metric_comparisons = {}
        
        if metrics_to_compare is None:
            # Use all common metrics
            metrics_a = set(metadata_a.performance_metrics or {})
            metrics_b = set(metadata_b.performance_metrics or {})
            metrics_to_compare = list(metrics_a.intersection(metrics_b))
        
        for metric in metrics_to_compare:
            val_a = metadata_a.performance_metrics.get(metric, 0)
            val_b = metadata_b.performance_metrics.get(metric, 0)
            
            metric_comparisons[metric] = {
                'model_a': val_a,
                'model_b': val_b,
                'difference': val_b - val_a,
                'improvement': ((val_b - val_a) / max(abs(val_a), 1e-10)) * 100
            }
        
        # Determine winner (simple majority)
        improvements = sum(1 for comp in metric_comparisons.values() 
                          if comp['improvement'] > 0)
        total_metrics = len(metric_comparisons)
        
        winner = None
        if improvements > total_metrics / 2:
            winner = model_b_id
        elif improvements < total_metrics / 2:
            winner = model_a_id
        
        confidence_score = abs(improvements - total_metrics / 2) / total_metrics
        
        return ModelComparison(
            model_a_id=model_a_id,
            model_b_id=model_b_id,
            metric_comparisons=metric_comparisons,
            winner=winner,
            confidence_score=confidence_score
        )
    
    def promote_model(
        self,
        model_id: str,
        target_stage: ModelStage,
        requirements: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Promote model to target stage"""
        metadata = self.get_model(model_id)
        if not metadata:
            return False
        
        # Check requirements
        if requirements:
            if not self._check_promotion_requirements(metadata, requirements):
                return False
        
        # Update stage
        metadata.stage = target_stage
        metadata.updated_at = datetime.now(timezone.utc)
        
        # Update status based on stage
        if target_stage == ModelStage.PRODUCTION:
            metadata.status = ModelStatus.DEPLOYED
        elif target_stage == ModelStage.STAGING:
            metadata.status = ModelStatus.VALIDATING
        else:
            metadata.status = ModelStatus.REGISTERED
        
        self._store_metadata(metadata)
        
        self.logger.info(f"Model {model_id} promoted to {target_stage.value}")
        return True
    
    def _check_promotion_requirements(
        self,
        metadata: ModelMetadata,
        requirements: Dict[str, Any]
    ) -> bool:
        """Check if model meets promotion requirements"""
        # Check minimum performance requirements
        if 'min_accuracy' in requirements:
            accuracy = metadata.performance_metrics.get('accuracy', 0)
            if accuracy < requirements['min_accuracy']:
                return False
        
        if 'min_f1' in requirements:
            f1 = metadata.performance_metrics.get('f1_score', 0)
            if f1 < requirements['min_f1']:
                return False
        
        # Check if model has been validated
        if 'validation_required' in requirements and requirements['validation_required']:
            if metadata.status not in [ModelStatus.VALIDATING, ModelStatus.DEPLOYED]:
                return False
        
        return True
    
    def rollback_model(
        self,
        model_id: str,
        target_version: Optional[str] = None
    ) -> bool:
        """Rollback model to previous version"""
        if target_version:
            # Rollback to specific version
            target_metadata = self.get_model_by_name_version(
                self.get_model(model_id).name, target_version
            )
        else:
            # Rollback to latest production version
            metadata = self.get_model(model_id)
            production_models = self.list_models(
                name=metadata.name,
                stage=ModelStage.PRODUCTION
            )
            
            if not production_models:
                return False
            
            # Get latest production model that's not the current one
            production_models = [m for m in production_models if m.model_id != model_id]
            if not production_models:
                return False
            
            target_metadata = production_models[0]
        
        if not target_metadata:
            return False
        
        # Update current model stage
        self.update_model_stage(model_id, ModelStage.ARCHIVED)
        
        # Promote target model to production
        return self.promote_model(target_metadata.model_id, ModelStage.PRODUCTION)
    
    def archive_model(self, model_id: str) -> bool:
        """Archive a model"""
        metadata = self.get_model(model_id)
        if not metadata:
            return False
        
        metadata.stage = ModelStage.ARCHIVED
        metadata.updated_at = datetime.now(timezone.utc)
        
        self._store_metadata(metadata)
        
        self.logger.info(f"Model {model_id} archived")
        return True
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """Delete model from registry"""
        metadata = self.get_model(model_id)
        if not metadata:
            return False
        
        # Prevent deletion of production models unless forced
        if metadata.stage == ModelStage.PRODUCTION and not force:
            self.logger.error(f"Cannot delete production model {model_id} without force=True")
            return False
        
        # Delete artifact file
        if metadata.artifact_path:
            artifact_path = Path(metadata.artifact_path)
            if artifact_path.exists():
                artifact_path.unlink()
        
        # Delete from database
        with self._get_db_connection() as conn:
            conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
        
        self.logger.info(f"Model {model_id} deleted")
        return True
    
    def get_model_lineage(self, model_id: str) -> List[ModelMetadata]:
        """Get model lineage (parent models)"""
        lineage = []
        current_id = model_id
        
        while current_id:
            metadata = self.get_model(current_id)
            if not metadata:
                break
            
            lineage.append(metadata)
            current_id = metadata.parent_model_id
        
        return lineage
    
    def search_models(self, query: str) -> List[ModelMetadata]:
        """Search models by name, description, or tags"""
        search_pattern = f"%{query}%"
        
        with self._get_db_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM models 
                WHERE name LIKE ? OR description LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC
            """, (search_pattern, search_pattern, search_pattern)).fetchall()
            
            return [self._row_to_metadata(row) for row in rows]
    
    def export_registry(self, output_path: str) -> bool:
        """Export registry metadata to JSON"""
        models = self.list_models()
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'registry_path': str(self.registry_path),
            'models': [asdict(model) for model in models]
        }
        
        # Convert datetime objects to strings for JSON serialization
        for model_data in export_data['models']:
            model_data['created_at'] = model_data['created_at'].isoformat()
            model_data['updated_at'] = model_data['updated_at'].isoformat()
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return True
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._get_db_connection() as conn:
            stats = {}
            
            # Total models
            stats['total_models'] = conn.execute(
                "SELECT COUNT(*) FROM models"
            ).fetchone()[0]
            
            # Models by stage
            stage_counts = conn.execute("""
                SELECT stage, COUNT(*) FROM models GROUP BY stage
            """).fetchall()
            stats['by_stage'] = {row[0]: row[1] for row in stage_counts}
            
            # Models by status
            status_counts = conn.execute("""
                SELECT status, COUNT(*) FROM models GROUP BY status
            """).fetchall()
            stats['by_status'] = {row[0]: row[1] for row in status_counts}
            
            # Models by framework
            framework_counts = conn.execute("""
                SELECT framework, COUNT(*) FROM models GROUP BY framework
            """).fetchall()
            stats['by_framework'] = {row[0]: row[1] for row in framework_counts}
            
            # Recent models (last 7 days)
            stats['recent_models'] = conn.execute("""
                SELECT COUNT(*) FROM models 
                WHERE created_at > datetime('now', '-7 days')
            """).fetchone()[0]
            
            return stats


def create_registry(config_path: Optional[str] = None) -> ModelRegistry:
    """
    Factory function to create ModelRegistry instance
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ModelRegistry instance
    """
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    registry_path = config.get('registry_path', './model_registry')
    
    return ModelRegistry(registry_path, config)


# Utility functions for common operations
def register_sklearn_model(
    model: Any,
    name: str,
    registry_path: str = "./model_registry",
    **kwargs
) -> str:
    """Convenience function to register scikit-learn models"""
    from sklearn.base import BaseEstimator
    
    if not isinstance(model, BaseEstimator):
        raise ValueError("Model must be a scikit-learn estimator")
    
    registry = ModelRegistry(registry_path)
    return registry.register_model(model, name=name, framework='scikit-learn', **kwargs)


def register_pytorch_model(
    model: Any,
    name: str,
    registry_path: str = "./model_registry",
    **kwargs
) -> str:
    """Convenience function to register PyTorch models"""
    try:
        import torch.nn
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")
    except ImportError:
        raise ValueError("PyTorch not available")
    
    registry = ModelRegistry(registry_path)
    return registry.register_model(model, name=name, framework='pytorch', **kwargs)


def get_best_model(
    name: str,
    metric: str = "accuracy",
    registry_path: str = "./model_registry"
) -> Optional[str]:
    """
    Get the best performing model for a given metric
    
    Args:
        name: Model name
        metric: Performance metric to optimize
        registry_path: Registry path
        
    Returns:
        Model ID of best performing model
    """
    registry = ModelRegistry(registry_path)
    models = registry.list_models(name=name)
    
    best_model = None
    best_score = float('-inf')
    
    for model in models:
        score = model.performance_metrics.get(metric, 0)
        if score > best_score:
            best_score = score
            best_model = model.model_id
    
    return best_model