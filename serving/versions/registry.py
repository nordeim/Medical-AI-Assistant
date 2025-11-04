"""
Model registry integration with MLflow and Weights & Biases support.

Provides adapters for integrating with external model registries
while maintaining internal version tracking consistency.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .core import ModelVersion, VersionRegistry


class RegistryAdapter(ABC):
    """Abstract base class for registry adapters."""
    
    @abstractmethod
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to external registry."""
        pass
    
    @abstractmethod
    def upload_model(self, version: ModelVersion, model_path: str) -> bool:
        """Upload model to external registry."""
        pass
    
    @abstractmethod
    def download_model(self, model_name: str, version: str) -> Optional[str]:
        """Download model from external registry."""
        pass
    
    @abstractmethod
    def list_versions(self, model_name: str) -> List[str]:
        """List available versions in external registry."""
        pass
    
    @abstractmethod
    def get_model_info(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model metadata from external registry."""
        pass
    
    @abstractmethod
    def create_experiment(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Create experiment for A/B testing."""
        pass


class MLflowRegistry(RegistryAdapter):
    """MLflow registry adapter."""
    
    def __init__(self, tracking_uri: str = None):
        self.tracking_uri = tracking_uri
        self.client = None
        self.connected = False
    
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to MLflow tracking server."""
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow not available. Install with: pip install mlflow")
            return False
        
        try:
            # Set tracking URI
            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            
            # Get MLflow client
            self.client = mlflow.MlflowClient()
            
            # Test connection
            self.client.search_experiments()
            
            self.connected = True
            logger.info(f"Connected to MLflow at {mlflow.get_tracking_uri()}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            return False
    
    def upload_model(self, version: ModelVersion, model_path: str) -> bool:
        """Upload model to MLflow registry."""
        if not self.connected:
            logger.error("Not connected to MLflow")
            return False
        
        try:
            # Create MLflow experiment for the model if it doesn't exist
            experiment_name = f"model_{version.model_name}"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
                mlflow.set_experiment(experiment_name)
            
            # Set tags for medical compliance tracking
            tags = {
                "model_name": version.model_name,
                "version": version.version,
                "version_type": version.version_type.value,
                "compliance_level": version.compliance.compliance_level.value,
                "medical_device_class": version.compliance.medical_device_class,
                "created_by": version.created_by
            }
            
            # Log model with MLflow
            with mlflow.start_run(experiment_id=experiment_id):
                # Log parameters
                mlflow.log_params({
                    "model_type": version.model_type,
                    "created_by": version.created_by,
                    "compliance_level": version.compliance.compliance_level.value,
                    "medical_device_class": version.compliance.medical_device_class
                })
                
                # Log metrics
                for metric_name, metric_value in version.performance_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.log_artifact(model_path)
                
                # Log version metadata as JSON
                version_metadata = json.dumps(version.to_dict(), indent=2)
                mlflow.log_text(version_metadata, "version_metadata.json")
                
                # Set tags
                mlflow.set_tags(tags)
            
            logger.info(f"Uploaded model {version.model_name} v{version.version} to MLflow")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model to MLflow: {e}")
            return False
    
    def download_model(self, model_name: str, version: str) -> Optional[str]:
        """Download model from MLflow registry."""
        if not self.connected:
            logger.error("Not connected to MLflow")
            return None
        
        try:
            experiment_name = f"model_{model_name}"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if not experiment:
                logger.error(f"Experiment {experiment_name} not found")
                return None
            
            # Find the run for this version
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                    filter_string=f"tags.version = '{version}'")
            
            if runs.empty:
                logger.error(f"Version {version} not found for model {model_name}")
                return None
            
            # Get the latest run for this version
            latest_run = runs.iloc[0]
            run_id = latest_run.run_id
            
            # Download artifacts
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="model",
                dst_path=f"/tmp/mlflow_downloads/{model_name}/{version}"
            )
            
            logger.info(f"Downloaded model {model_name} v{version} from MLflow")
            return local_path
            
        except Exception as e:
            logger.error(f"Failed to download model from MLflow: {e}")
            return None
    
    def list_versions(self, model_name: str) -> List[str]:
        """List available versions in MLflow."""
        if not self.connected:
            logger.error("Not connected to MLflow")
            return []
        
        try:
            experiment_name = f"model_{model_name}"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if not experiment:
                return []
            
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            if runs.empty:
                return []
            
            # Extract versions from tags
            versions = []
            for _, run in runs.iterrows():
                if 'tags.version' in run:
                    version = run['tags.version']
                    if version and version not in versions:
                        versions.append(version)
            
            return sorted(versions)
            
        except Exception as e:
            logger.error(f"Failed to list versions from MLflow: {e}")
            return []
    
    def get_model_info(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model metadata from MLflow."""
        if not self.connected:
            logger.error("Not connected to MLflow")
            return {}
        
        try:
            experiment_name = f"model_{model_name}"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if not experiment:
                return {}
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.version = '{version}'"
            )
            
            if runs.empty:
                return {}
            
            run = runs.iloc[0]
            
            return {
                "model_name": model_name,
                "version": version,
                "run_id": run.run_id,
                "experiment_id": experiment.experiment_id,
                "status": run.get('status', 'unknown'),
                "start_time": datetime.fromtimestamp(run.get('start_time', 0) / 1000).isoformat(),
                "end_time": datetime.fromtimestamp(run.get('end_time', 0) / 1000).isoformat() if run.get('end_time') else None,
                "metrics": {k: v for k, v in run.items() if k.startswith('metrics.')},
                "tags": run.get('tags', {}),
                "params": {k: v for k, v in run.items() if k.startswith('params.')}
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info from MLflow: {e}")
            return {}
    
    def create_experiment(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Create MLflow experiment for A/B testing."""
        if not self.connected:
            logger.error("Not connected to MLflow")
            return False
        
        try:
            experiment_name = f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with mlflow.start_run():
                mlflow.log_params(config)
                mlflow.log_text(json.dumps(config, indent=2), "experiment_config.json")
                mlflow.set_tags({
                    "experiment_type": "ab_testing",
                    "model_name": model_name,
                    "created_at": datetime.now().isoformat()
                })
            
            logger.info(f"Created MLflow experiment: {experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create MLflow experiment: {e}")
            return False


class WandbRegistry(RegistryAdapter):
    """Weights & Biases registry adapter."""
    
    def __init__(self, project: str = None):
        self.project = project
        self.run = None
        self.connected = False
    
    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to Weights & Biases."""
        if not WANDB_AVAILABLE:
            logger.error("Weights & Biases not available. Install with: pip install wandb")
            return False
        
        try:
            # Initialize wandb
            wandb.init(
                project=self.project or credentials.get('project', 'medical-ai-models'),
                config=credentials
            )
            
            self.connected = True
            logger.info(f"Connected to Weights & Biases project: {self.project}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to W&B: {e}")
            return False
    
    def upload_model(self, version: ModelVersion, model_path: str) -> bool:
        """Upload model to W&B."""
        if not self.connected:
            logger.error("Not connected to W&B")
            return False
        
        try:
            # Create wandb artifact for model
            artifact = wandb.Artifact(
                name=f"{version.model_name}_v{version.version}",
                type="model",
                description=f"Medical AI model {version.model_name} version {version.version}"
            )
            
            # Add compliance metadata
            artifact.metadata = {
                "model_name": version.model_name,
                "version": version.version,
                "model_type": version.model_type,
                "compliance_level": version.compliance.compliance_level.value,
                "medical_device_class": version.compliance.medical_device_class,
                "created_by": version.created_by,
                "created_at": version.created_at.isoformat(),
                "performance_metrics": version.performance_metrics,
                "changelog": version.changelog
            }
            
            # Add model file
            artifact.add_file(model_path)
            
            # Add version metadata
            metadata_file = json.dumps(version.to_dict(), indent=2)
            with open("/tmp/version_metadata.json", "w") as f:
                f.write(metadata_file)
            artifact.add_file("/tmp/version_metadata.json", name="version_metadata.json")
            
            # Log artifact
            wandb.log_artifact(artifact)
            
            # Log metrics and parameters
            wandb.config.update({
                "model_name": version.model_name,
                "version": version.version,
                "model_type": version.model_type,
                "compliance_level": version.compliance.compliance_level.value,
                "medical_device_class": version.compliance.medical_device_class
            })
            
            for metric_name, metric_value in version.performance_metrics.items():
                wandb.log({metric_name: metric_value})
            
            logger.info(f"Uploaded model {version.model_name} v{version.version} to W&B")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload model to W&B: {e}")
            return False
    
    def download_model(self, model_name: str, version: str) -> Optional[str]:
        """Download model from W&B."""
        if not self.connected:
            logger.error("Not connected to W&B")
            return None
        
        try:
            # Get artifact
            artifact_name = f"{model_name}_v{version}:latest"
            
            artifact = wandb.run.use_artifact(artifact_name, type='model')
            artifact_dir = artifact.download(root=f"/tmp/wandb_downloads/{model_name}/{version}")
            
            logger.info(f"Downloaded model {model_name} v{version} from W&B")
            return artifact_dir
            
        except Exception as e:
            logger.error(f"Failed to download model from W&B: {e}")
            return None
    
    def list_versions(self, model_name: str) -> List[str]:
        """List available versions in W&B."""
        if not self.connected:
            logger.error("Not connected to W&B")
            return []
        
        try:
            # List all artifacts of type 'model'
            api = wandb.Api()
            
            versions = []
            for artifact in api.artifacts(type="model", name=f"{model_name}_v*"):
                # Extract version from artifact name
                name_parts = artifact.name.split('_v')
                if len(name_parts) == 2:
                    version = name_parts[1].split(':')[0]  # Remove :latest suffix
                    if version not in versions:
                        versions.append(version)
            
            return sorted(versions)
            
        except Exception as e:
            logger.error(f"Failed to list versions from W&B: {e}")
            return []
    
    def get_model_info(self, model_name: str, version: str) -> Dict[str, Any]:
        """Get model metadata from W&B."""
        if not self.connected:
            logger.error("Not connected to W&B")
            return {}
        
        try:
            artifact_name = f"{model_name}_v{version}:latest"
            artifact = wandb.run.use_artifact(artifact_name, type='model')
            
            return {
                "model_name": model_name,
                "version": version,
                "artifact_name": artifact.name,
                "description": artifact.description,
                "metadata": artifact.metadata,
                "created_at": artifact.created_at,
                "size": artifact.size,
                "files": len(list(artifact.logged_by().logged_artifacts()))
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info from W&B: {e}")
            return {}
    
    def create_experiment(self, model_name: str, config: Dict[str, Any]) -> bool:
        """Create W&B experiment for A/B testing."""
        if not self.connected:
            logger.error("Not connected to W&B")
            return False
        
        try:
            experiment_name = f"experiment_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create new run for experiment
            with wandb.init(project=self.project, name=experiment_name, 
                          tags=["ab_testing", model_name]):
                wandb.config.update(config)
                wandb.log({"experiment_created": datetime.now().isoformat()})
            
            logger.info(f"Created W&B experiment: {experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create W&B experiment: {e}")
            return False


class RegistryManager:
    """Manager for multiple registry adapters."""
    
    def __init__(self):
        self.registries: Dict[str, RegistryAdapter] = {}
        self.internal_registry = VersionRegistry(Path("/tmp/version_registry"))
    
    def register_adapter(self, name: str, adapter: RegistryAdapter):
        """Register a new adapter."""
        self.registries[name] = adapter
        logger.info(f"Registered registry adapter: {name}")
    
    def connect_all(self, credentials_map: Dict[str, Dict[str, Any]]):
        """Connect to all registered registries."""
        connected = []
        failed = []
        
        for name, adapter in self.registries.items():
            creds = credentials_map.get(name, {})
            if adapter.connect(creds):
                connected.append(name)
            else:
                failed.append(name)
        
        logger.info(f"Connected to registries: {connected}")
        if failed:
            logger.warning(f"Failed to connect to registries: {failed}")
    
    def sync_to_registries(self, version: ModelVersion, model_path: str) -> Dict[str, bool]:
        """Sync model version to all connected registries."""
        results = {}
        
        for name, adapter in self.registries.items():
            if adapter.connected:
                try:
                    results[name] = adapter.upload_model(version, model_path)
                except Exception as e:
                    logger.error(f"Failed to sync to {name}: {e}")
                    results[name] = False
            else:
                logger.warning(f"Registry {name} not connected, skipping sync")
                results[name] = False
        
        return results
    
    def get_available_versions(self, model_name: str) -> Dict[str, List[str]]:
        """Get available versions from all registries."""
        all_versions = {}
        
        for name, adapter in self.registries.items():
            if adapter.connected:
                try:
                    versions = adapter.list_versions(model_name)
                    all_versions[name] = versions
                except Exception as e:
                    logger.error(f"Failed to get versions from {name}: {e}")
                    all_versions[name] = []
        
        return all_versions
    
    def get_registry_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registries."""
        status = {}
        
        for name, adapter in self.registries.items():
            status[name] = {
                "connected": adapter.connected,
                "type": type(adapter).__name__,
                "available": hasattr(adapter, 'connected') and adapter.connected
            }
        
        return status
    
    def find_version_across_registries(self, model_name: str, version: str) -> Dict[str, Any]:
        """Find version information across all registries."""
        results = {}
        
        for name, adapter in self.registries.items():
            if adapter.connected:
                try:
                    model_info = adapter.get_model_info(model_name, version)
                    if model_info:
                        results[name] = model_info
                except Exception as e:
                    logger.error(f"Failed to get version info from {name}: {e}")
        
        return results
    
    def create_ab_testing_experiments(self, model_name: str, 
                                    experiment_configs: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Create A/B testing experiments in all registries."""
        results = {}
        
        for name, adapter in self.registries.items():
            if adapter.connected:
                try:
                    success_count = 0
                    for config in experiment_configs:
                        if adapter.create_experiment(model_name, config):
                            success_count += 1
                    results[name] = success_count == len(experiment_configs)
                except Exception as e:
                    logger.error(f"Failed to create experiments in {name}: {e}")
                    results[name] = False
        
        return results