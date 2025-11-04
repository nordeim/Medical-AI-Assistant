"""
MLflow Model Registry with A/B Testing Framework
Production-grade model versioning, deployment, and performance comparison.
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version import ModelVersion
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

@dataclass
class ABTestConfig:
    """A/B Test configuration"""
    test_id: str
    name: str
    control_model: str  # Model version
    treatment_model: str  # Model version
    traffic_split: float  # 0.5 = 50/50 split
    duration_hours: int
    min_sample_size: int
    success_metrics: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "active"  # active, completed, paused, cancelled

@dataclass
class ABTestResult:
    """A/B Test result"""
    test_id: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]
    statistical_significance: bool
    p_value: float
    sample_size_control: int
    sample_size_treatment: int
    winner: str  # control, treatment, or inconclusive
    confidence_level: float
    recommendations: List[str]

class ModelRegistry:
    """MLflow-based model registry with A/B testing"""
    
    def __init__(self, mlflow_uri: str = "sqlite:///models.db", 
                 experiment_name: str = "medical_ai_models"):
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        self.client = None
        
        # A/B Testing
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, ABTestResult] = {}
        self.model_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Model deployment tracking
        self.deployed_models: Dict[str, Dict[str, Any]] = {}
        
        # Initialize MLflow
        self._initialize_mlflow()
        
    def _initialize_mlflow(self):
        """Initialize MLflow client and experiment"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            self.client = MlflowClient()
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(self.experiment_name)
                else:
                    experiment_id = experiment.experiment_id
            except Exception as e:
                logger.warning(f"Experiment creation error: {str(e)}, using default")
                experiment_id = mlflow.create_experiment(self.experiment_name)
            
            logger.info(f"MLflow initialized with experiment: {self.experiment_name}")
            
        except Exception as e:
            logger.error(f"MLflow initialization failed: {str(e)}")
            raise
    
    def register_model(self, model, model_name: str, model_version: str,
                      model_metadata: Dict[str, Any], 
                      performance_metrics: Dict[str, float]) -> str:
        """Register a model in MLflow registry"""
        try:
            # Create experiment run
            with mlflow.start_run(experiment_name=self.experiment_name):
                
                # Log model parameters and metrics
                mlflow.log_params(model_metadata.get("parameters", {}))
                mlflow.log_metrics(performance_metrics)
                
                # Log model based on type
                if hasattr(model, 'predict'):
                    # Scikit-learn model
                    if model_metadata.get("framework") == "scikit-learn":
                        mlflow.sklearn.log_model(model, f"{model_name}_v{model_version}")
                    # PyTorch model
                    elif model_metadata.get("framework") == "pytorch":
                        mlflow.pytorch.log_model(model, f"{model_name}_v{model_version}")
                
                # Add tags for organization
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("model_version", model_version)
                mlflow.set_tag("framework", model_metadata.get("framework", "unknown"))
                mlflow.set_tag("deployment_status", "staging")
                mlflow.set_tag("created_by", "medical_ai_system")
                
                run_id = mlflow.last_active_run().info.run_id
                
                # Create model version
                model_uri = f"runs:/{run_id}/{model_name}_v{model_version}"
                
                # Try to create registered model, fall back to add version
                try:
                    self.client.create_registered_model(model_name)
                except mlflow.exceptions.MlflowException:
                    pass  # Model already exists
                
                model_version_info = self.client.create_model_version(
                    name=model_name,
                    source=model_uri,
                    run_id=run_id,
                    tags={
                        "version": model_version,
                        "framework": model_metadata.get("framework", "unknown"),
                        "deployment_status": "staging"
                    }
                )
                
                logger.info(f"Model registered: {model_name} v{model_version}")
                return model_version_info.version
                
        except Exception as e:
            logger.error(f"Model registration failed: {str(e)}")
            raise
    
    def create_ab_test(self, test_config: ABTestConfig) -> str:
        """Create a new A/B test"""
        try:
            # Validate model versions exist
            control_exists = self._model_version_exists(test_config.control_model)
            treatment_exists = self._model_version_exists(test_config.treatment_model)
            
            if not control_exists or not treatment_exists:
                raise ValueError("One or both model versions do not exist")
            
            # Store test configuration
            self.active_tests[test_config.test_id] = test_config
            
            # Update model deployment status
            self._update_model_deployment_status(test_config.control_model, "ab_testing")
            self._update_model_deployment_status(test_config.treatment_model, "ab_testing")
            
            # Log test creation
            logger.info(f"A/B Test created: {test_config.name} (ID: {test_config.test_id})")
            
            return test_config.test_id
            
        except Exception as e:
            logger.error(f"A/B Test creation failed: {str(e)}")
            raise
    
    def record_prediction_result(self, test_id: str, model_version: str, 
                                prediction_data: Dict[str, Any]):
        """Record prediction result for A/B testing"""
        try:
            if test_id not in self.active_tests:
                return  # Test doesn't exist
            
            test_config = self.active_tests[test_id]
            
            # Ensure we're tracking this model version
            if model_version not in self.model_performance_history:
                self.model_performance_history[model_version] = []
            
            # Add prediction data
            result_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "model_version": model_version,
                "patient_id": prediction_data.get("patient_id"),
                "prediction": prediction_data.get("prediction"),
                "confidence": prediction_data.get("confidence", 0.0),
                "processing_time": prediction_data.get("processing_time", 0.0),
                "actual_outcome": prediction_data.get("actual_outcome"),
                "accuracy": prediction_data.get("accuracy")
            }
            
            self.model_performance_history[model_version].append(result_entry)
            
            # Periodically evaluate test
            if len(self.model_performance_history[model_version]) % 100 == 0:
                asyncio.create_task(self._evaluate_ab_test(test_id))
                
        except Exception as e:
            logger.error(f"Prediction result recording failed: {str(e)}")
    
    async def _evaluate_ab_test(self, test_id: str):
        """Evaluate A/B test results"""
        try:
            test_config = self.active_tests[test_id]
            
            # Get performance data for both models
            control_data = self.model_performance_history.get(test_config.control_model, [])
            treatment_data = self.model_performance_history.get(test_config.treatment_model, [])
            
            # Check if we have enough data
            min_samples = test_config.min_sample_size
            
            if len(control_data) < min_samples or len(treatment_data) < min_samples:
                logger.debug(f"A/B test {test_id}: Insufficient data ({len(control_data)}, {len(treatment_data)})")
                return
            
            # Calculate metrics
            control_metrics = self._calculate_model_metrics(control_data)
            treatment_metrics = self._calculate_model_metrics(treatment_data)
            
            # Perform statistical significance test
            significance_result = self._perform_significance_test(
                control_data, treatment_data, 
                test_config.success_metrics
            )
            
            # Determine winner
            winner = self._determine_test_winner(
                control_metrics, treatment_metrics, 
                significance_result, test_config.success_metrics
            )
            
            # Create test result
            test_result = ABTestResult(
                test_id=test_id,
                control_metrics=control_metrics,
                treatment_metrics=treatment_metrics,
                statistical_significance=significance_result["significant"],
                p_value=significance_result["p_value"],
                sample_size_control=len(control_data),
                sample_size_treatment=len(treatment_data),
                winner=winner,
                confidence_level=0.95,
                recommendations=self._generate_test_recommendations(
                    control_metrics, treatment_metrics, 
                    significance_result, winner
                )
            )
            
            self.test_results[test_id] = test_result
            
            # Check if test should be completed
            if self._should_complete_test(test_config, test_result):
                await self._complete_ab_test(test_id)
            
            logger.info(f"A/B test {test_id} evaluated: Winner = {winner}")
            
        except Exception as e:
            logger.error(f"A/B test evaluation failed: {str(e)}")
    
    def _calculate_model_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from prediction data"""
        if not data:
            return {}
        
        confidences = [d.get("confidence", 0.0) for d in data if d.get("confidence") is not None]
        processing_times = [d.get("processing_time", 0.0) for d in data]
        accuracies = [d.get("accuracy") for d in data if d.get("accuracy") is not None]
        
        metrics = {
            "sample_size": len(data),
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "avg_processing_time": np.mean(processing_times) if processing_times else 0.0,
            "p95_processing_time": np.percentile(processing_times, 95) if processing_times else 0.0
        }
        
        if accuracies:
            metrics.update({
                "accuracy": np.mean(accuracies),
                "precision": np.mean(accuracies),  # Simplified for demo
                "recall": np.mean(accuracies),     # Simplified for demo
                "f1_score": np.mean(accuracies)    # Simplified for demo
            })
        
        return metrics
    
    def _perform_significance_test(self, control_data: List[Dict[str, Any]], 
                                  treatment_data: List[Dict[str, Any]], 
                                  metrics: List[str]) -> Dict[str, Any]:
        """Perform statistical significance test"""
        # Extract confidence scores for comparison
        control_confidences = [d.get("confidence", 0.0) for d in control_data if d.get("confidence")]
        treatment_confidences = [d.get("confidence", 0.0) for d in treatment_data if d.get("confidence")]
        
        if not control_confidences or not treatment_confidences:
            return {"significant": False, "p_value": 1.0, "test_type": "insufficient_data"}
        
        # Perform t-test
        try:
            statistic, p_value = stats.ttest_ind(control_confidences, treatment_confidences)
            
            return {
                "significant": p_value < 0.05,
                "p_value": p_value,
                "test_type": "t-test",
                "statistic": statistic
            }
            
        except Exception as e:
            logger.warning(f"Statistical test failed: {str(e)}")
            return {"significant": False, "p_value": 1.0, "test_type": "failed"}
    
    def _determine_test_winner(self, control_metrics: Dict[str, float], 
                              treatment_metrics: Dict[str, float],
                              significance_result: Dict[str, Any], 
                              success_metrics: List[str]) -> str:
        """Determine the winner of the A/B test"""
        
        if not significance_result["significant"]:
            return "inconclusive"
        
        # Compare primary success metric (usually accuracy)
        primary_metric = success_metrics[0] if success_metrics else "accuracy"
        
        control_score = control_metrics.get(primary_metric, 0.0)
        treatment_score = treatment_metrics.get(primary_metric, 0.0)
        
        # Determine winner based on score
        if treatment_score > control_score:
            return "treatment"
        elif control_score > treatment_score:
            return "control"
        else:
            return "inconclusive"
    
    def _generate_test_recommendations(self, control_metrics: Dict[str, float],
                                     treatment_metrics: Dict[str, float],
                                     significance_result: Dict[str, Any],
                                     winner: str) -> List[str]:
        """Generate recommendations based on A/B test results"""
        recommendations = []
        
        if winner == "treatment":
            recommendations.append("Deploy treatment model to production")
            recommendations.append("Gradually increase traffic allocation to treatment model")
        elif winner == "control":
            recommendations.append("Continue with current control model")
            recommendations.append("Consider retraining treatment model with new data")
        else:
            recommendations.append("Results inconclusive - extend test duration")
            recommendations.append("Consider increasing sample size or adjusting metrics")
        
        # Performance insights
        if significance_result["significant"]:
            confidence_improvement = treatment_metrics.get("avg_confidence", 0) - control_metrics.get("avg_confidence", 0)
            if confidence_improvement > 0.05:
                recommendations.append(f"Treatment model shows {confidence_improvement:.3f} confidence improvement")
            
            latency_change = treatment_metrics.get("avg_processing_time", 0) - control_metrics.get("avg_processing_time", 0)
            if latency_change > 0.1:
                recommendations.append("Treatment model has higher latency - consider optimization")
        
        return recommendations
    
    def _should_complete_test(self, test_config: ABTestConfig, 
                            test_result: ABTestResult) -> bool:
        """Determine if test should be completed"""
        # Check duration
        elapsed_time = datetime.utcnow() - test_config.start_time
        if elapsed_time > timedelta(hours=test_config.duration_hours):
            return True
        
        # Check statistical significance
        if test_result.statistical_significance and test_result.sample_size_control > 1000:
            return True
        
        return False
    
    async def _complete_ab_test(self, test_id: str):
        """Complete an A/B test and deploy winning model"""
        try:
            test_config = self.active_tests[test_id]
            test_result = self.test_results[test_id]
            
            # Update test status
            test_config.status = "completed"
            test_config.end_time = datetime.utcnow()
            
            # Deploy winning model if significant
            if test_result.statistical_significance and test_result.winner != "inconclusive":
                winning_model = (test_config.treatment_model if test_result.winner == "treatment" 
                               else test_config.control_model)
                
                self.deploy_model(winning_model, "production")
                
                logger.info(f"A/B test {test_id} completed: {test_result.winner} model deployed")
            
        except Exception as e:
            logger.error(f"A/B test completion failed: {str(e)}")
    
    def deploy_model(self, model_version: str, stage: str = "production") -> bool:
        """Deploy model to specific stage"""
        try:
            # Update model status in MLflow
            client = self.client
            
            # Get model versions
            versions = client.get_latest_versions(model_version, stages=[stage])
            
            if versions:
                # Transition existing version
                client.transition_model_version_stage(
                    name=model_version,
                    version=versions[0].version,
                    stage=stage
                )
            else:
                # Create new production version
                try:
                    latest_versions = client.get_latest_versions(model_version)
                    if latest_versions:
                        client.transition_model_version_stage(
                            name=model_version,
                            version=latest_versions[0].version,
                            stage=stage
                        )
                except Exception as e:
                    logger.warning(f"No versions available for {model_version}: {str(e)}")
                    return False
            
            # Track deployment
            self.deployed_models[model_version] = {
                "stage": stage,
                "deployed_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            logger.info(f"Model {model_version} deployed to {stage}")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {str(e)}")
            return False
    
    def rollback_model(self, model_version: str, target_stage: str = "staging") -> bool:
        """Rollback model to previous stage"""
        try:
            client = self.client
            
            # Get current production versions
            prod_versions = client.get_latest_versions(model_version, stages=["production"])
            
            if prod_versions:
                # Rollback to staging
                client.transition_model_version_stage(
                    name=model_version,
                    version=prod_versions[0].version,
                    stage=target_stage
                )
                
                logger.info(f"Model {model_version} rolled back to {target_stage}")
                return True
            else:
                logger.warning(f"No production version found for {model_version}")
                return False
                
        except Exception as e:
            logger.error(f"Model rollback failed: {str(e)}")
            return False
    
    def _model_version_exists(self, model_version: str) -> bool:
        """Check if model version exists"""
        try:
            client = self.client
            versions = client.get_latest_versions(model_version)
            return len(versions) > 0
        except:
            return False
    
    def _update_model_deployment_status(self, model_version: str, status: str):
        """Update model deployment status"""
        try:
            client = self.client
            versions = client.get_latest_versions(model_version)
            
            if versions:
                # Update tags (simplified approach)
                pass  # MLflow doesn't directly support tag updates via API
                
        except Exception as e:
            logger.warning(f"Status update failed for {model_version}: {str(e)}")
    
    def get_ab_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test status"""
        if test_id not in self.active_tests:
            return None
        
        test_config = self.active_tests[test_id]
        result = self.test_results.get(test_id)
        
        return {
            "config": asdict(test_config),
            "result": asdict(result) if result else None,
            "current_sample_sizes": {
                "control": len(self.model_performance_history.get(test_config.control_model, [])),
                "treatment": len(self.model_performance_history.get(test_config.treatment_model, []))
            }
        }
    
    def get_all_ab_tests(self) -> List[Dict[str, Any]]:
        """Get all A/B tests"""
        tests = []
        
        for test_id in self.active_tests:
            status = self.get_ab_test_status(test_id)
            if status:
                tests.append(status)
        
        return tests
    
    def list_deployed_models(self) -> Dict[str, Dict[str, Any]]:
        """List all deployed models"""
        return self.deployed_models.copy()
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific model"""
        try:
            client = self.client
            versions = client.get_latest_versions(model_name)
            
            return [
                {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "created_at": v.creation_timestamp,
                    "last_updated": v.last_updated_timestamp,
                    "tags": v.tags
                }
                for v in versions
            ]
            
        except Exception as e:
            logger.error(f"Failed to get model versions: {str(e)}")
            return []
    
    def delete_ab_test(self, test_id: str) -> bool:
        """Delete an A/B test"""
        try:
            if test_id in self.active_tests:
                test_config = self.active_tests[test_id]
                
                # Update model statuses back to normal
                self._update_model_deployment_status(test_config.control_model, "staging")
                self._update_model_deployment_status(test_config.treatment_model, "staging")
                
                # Remove test data
                del self.active_tests[test_id]
                if test_id in self.test_results:
                    del self.test_results[test_id]
                
                logger.info(f"A/B test {test_id} deleted")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"A/B test deletion failed: {str(e)}")
            return False

# Example usage and A/B test management
if __name__ == "__main__":
    # Initialize registry
    registry = ModelRegistry()
    
    # Create A/B test
    test_config = ABTestConfig(
        test_id=str(uuid.uuid4()),
        name="Medical Diagnosis Model v1 vs v2",
        control_model="medical-diagnosis-v1",
        treatment_model="medical-diagnosis-v2", 
        traffic_split=0.5,
        duration_hours=24,
        min_sample_size=500,
        success_metrics=["accuracy", "precision"],
        start_time=datetime.utcnow()
    )
    
    test_id = registry.create_ab_test(test_config)
    print(f"A/B Test created: {test_id}")
    
    # Simulate prediction results
    for i in range(1000):
        prediction_data = {
            "patient_id": f"patient_{i}",
            "prediction": {"diagnosis": "condition_a"},
            "confidence": np.random.uniform(0.7, 0.95),
            "processing_time": np.random.uniform(0.1, 0.5),
            "accuracy": np.random.uniform(0.8, 0.95)
        }
        
        # Randomly assign to control or treatment
        model_version = "medical-diagnosis-v2" if i % 2 == 0 else "medical-diagnosis-v1"
        registry.record_prediction_result(test_id, model_version, prediction_data)
    
    print("Prediction results recorded")
    
    # Check test status
    status = registry.get_ab_test_status(test_id)
    if status and status["result"]:
        print(f"Test Result: Winner = {status['result']['winner']}")
        print(f"Statistical Significance: {status['result']['statistical_significance']}")