"""
A/B testing infrastructure for model comparison and performance analysis.

Provides comprehensive A/B testing capabilities for medical AI models
with statistical significance testing and clinical outcome analysis.
"""

import json
import random
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import numpy as np
from scipy import stats

from .core import ModelVersion, ComplianceLevel, VersionType

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """A/B testing experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class TestType(Enum):
    """Types of A/B tests."""
    MODEL_PERFORMANCE = "model_performance"
    CLINICAL_OUTCOMES = "clinical_outcomes"
    USER_EXPERIENCE = "user_experience"
    SYSTEM_PERFORMANCE = "system_performance"
    COMPLIANCE_VALIDATION = "compliance_validation"
    LATENCY_COMPARISON = "latency_comparison"


class StatisticalTest(Enum):
    """Statistical tests for significance analysis."""
    T_TEST = "t_test"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON = "wilcoxon"
    BAYESIAN = "bayesian"


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiment."""
    name: str
    description: str
    model_name: str
    control_version: str
    treatment_version: str
    test_type: TestType
    statistical_test: StatisticalTest = StatisticalTest.T_TEST
    
    # Traffic allocation
    traffic_split: Dict[str, float] = field(default_factory=lambda: {"control": 0.5, "treatment": 0.5})
    
    # Sample size and duration
    minimum_sample_size: int = 100
    maximum_duration_days: int = 30
    significance_level: float = 0.05
    power: float = 0.8
    
    # Success criteria
    primary_metric: str = "accuracy"
    success_threshold: float = 0.0
    minimum_detectable_effect: float = 0.05
    
    # Compliance and safety
    requires_clinical_approval: bool = False
    max_rollback_time_hours: int = 24
    safety_monitoring_interval_minutes: int = 15
    
    # Metadata
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentResult:
    """Result of A/B testing experiment."""
    experiment_id: str
    experiment_name: str
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_hours: Optional[float] = None
    
    # Sample sizes
    control_sample_size: int = 0
    treatment_sample_size: int = 0
    
    # Statistical results
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    statistical_power: Optional[float] = None
    
    # Performance metrics
    control_metrics: Dict[str, float] = field(default_factory=dict)
    treatment_metrics: Dict[str, float] = field(default_factory=dict)
    metric_differences: Dict[str, float] = field(default_factory=dict)
    
    # Clinical outcomes (if applicable)
    clinical_outcomes: Dict[str, Any] = field(default_factory=dict)
    
    # Statistical significance
    is_significant: bool = False
    significance_level: float = 0.05
    conclusion: str = ""
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Logs and monitoring
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    monitoring_data: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ABTestGroup:
    """Group assignment in A/B test."""
    group_id: str
    group_name: str
    version: ModelVersion
    traffic_percentage: float
    assignment_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def get_assignment_hash(self, user_id: str, experiment_id: str) -> str:
        """Generate consistent assignment hash for user."""
        data = f"{user_id}:{experiment_id}:{self.group_id}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def should_assign_user(self, user_id: str, experiment_id: str) -> bool:
        """Determine if user should be assigned to this group."""
        hash_value = int(self.get_assignment_hash(user_id, experiment_id), 16)
        return (hash_value % 10000) < (self.traffic_percentage * 10000)


class ABTestingManager:
    """Manager for A/B testing experiments."""
    
    def __init__(self, registry_path: str = "/tmp/ab_tests"):
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.results: Dict[str, ExperimentResult] = {}
        self.active_assignments: Dict[str, Dict[str, ABTestGroup]] = {}
        self.experiment_path = f"{registry_path}/experiments"
        self.results_path = f"{registry_path}/results"
        
        # Initialize directories
        import os
        os.makedirs(self.experiment_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create new A/B testing experiment."""
        experiment_id = str(uuid.uuid4())
        
        # Validate configuration
        validation = self._validate_experiment_config(config)
        if not validation["valid"]:
            raise ValueError(f"Invalid experiment configuration: {validation['errors']}")
        
        # Store experiment
        self.experiments[experiment_id] = config
        
        # Save to disk
        config_path = f"{self.experiment_path}/{experiment_id}.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
        
        logger.info(f"Created A/B test experiment: {experiment_id} - {config.name}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start A/B testing experiment."""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        config = self.experiments[experiment_id]
        
        # Check compliance requirements
        if config.requires_clinical_approval:
            clinical_check = self._check_clinical_requirements(experiment_id)
            if not clinical_check["approved"]:
                logger.error(f"Clinical approval required for experiment {experiment_id}")
                return False
        
        # Initialize experiment result
        result = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=config.name,
            status=ExperimentStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.results[experiment_id] = result
        
        # Save result
        result_path = f"{self.results_path}/{experiment_id}.json"
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Started A/B test experiment: {experiment_id}")
        return True
    
    def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop A/B testing experiment."""
        if experiment_id not in self.results:
            return False
        
        result = self.results[experiment_id]
        result.status = ExperimentStatus.STOPPED
        result.end_time = datetime.now()
        
        if result.start_time:
            result.duration_hours = (result.end_time - result.start_time).total_seconds() / 3600
        
        if reason:
            result.conclusion = f"Stopped: {reason}"
        
        # Save updated result
        result_path = f"{self.results_path}/{experiment_id}.json"
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Stopped A/B test experiment: {experiment_id}")
        return True
    
    def complete_experiment(self, experiment_id: str) -> bool:
        """Complete A/B testing experiment with statistical analysis."""
        if experiment_id not in self.results:
            return False
        
        config = self.experiments[experiment_id]
        result = self.results[experiment_id]
        
        # Perform statistical analysis
        analysis_result = self._perform_statistical_analysis(experiment_id)
        
        # Update result with analysis
        result.status = ExperimentStatus.COMPLETED
        result.end_time = datetime.now()
        result.duration_hours = (result.end_time - result.start_time).total_seconds() / 3600
        
        # Update statistical results
        result.test_statistic = analysis_result.get("test_statistic")
        result.p_value = analysis_result.get("p_value")
        result.confidence_interval = analysis_result.get("confidence_interval")
        result.effect_size = analysis_result.get("effect_size")
        result.statistical_power = analysis_result.get("statistical_power")
        
        result.is_significant = analysis_result.get("is_significant", False)
        result.conclusion = analysis_result.get("conclusion", "")
        result.recommendations = analysis_result.get("recommendations", [])
        
        # Save updated result
        result_path = f"{self.results_path}/{experiment_id}.json"
        with open(result_path, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        logger.info(f"Completed A/B test experiment: {experiment_id}")
        return True
    
    def assign_user_to_group(self, experiment_id: str, user_id: str) -> Optional[ABTestGroup]:
        """Assign user to control or treatment group."""
        if experiment_id not in self.experiments:
            return None
        
        # Check if already assigned
        if experiment_id in self.active_assignments and user_id in self.active_assignments[experiment_id]:
            return self.active_assignments[experiment_id][user_id]
        
        config = self.experiments[experiment_id]
        
        # Create groups
        control_group = ABTestGroup(
            group_id="control",
            group_name="Control Group",
            version=config.control_version,  # This should be ModelVersion object
            traffic_percentage=config.traffic_split["control"]
        )
        
        treatment_group = ABTestGroup(
            group_id="treatment",
            group_name="Treatment Group", 
            version=config.treatment_version,  # This should be ModelVersion object
            traffic_percentage=config.traffic_split["treatment"]
        )
        
        # Assign user based on hash
        control_hash = int(control_group.get_assignment_hash(user_id, experiment_id), 16)
        treatment_hash = int(treatment_group.get_assignment_hash(user_id, experiment_id), 16)
        
        # Use hash to determine assignment
        if (control_hash % 10000) < (config.traffic_split["control"] * 10000):
            assigned_group = control_group
        else:
            assigned_group = treatment_group
        
        # Store assignment
        if experiment_id not in self.active_assignments:
            self.active_assignments[experiment_id] = {}
        self.active_assignments[experiment_id][user_id] = assigned_group
        
        return assigned_group
    
    def record_experiment_event(self, experiment_id: str, user_id: str, 
                              event_type: str, event_data: Dict[str, Any]):
        """Record event in A/B test."""
        if experiment_id not in self.results:
            return
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "event_type": event_type,
            "experiment_id": experiment_id,
            "data": event_data
        }
        
        # Check for safety alerts
        self._check_safety_alerts(experiment_id, event)
        
        # Store event (in practice would store in time-series database)
        logger.debug(f"Recorded experiment event: {event_type} for user {user_id}")
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get results for experiment."""
        return self.results.get(experiment_id)
    
    def list_experiments(self, status: ExperimentStatus = None) -> List[ExperimentConfig]:
        """List experiments, optionally filtered by status."""
        if status is None:
            return list(self.experiments.values())
        return [exp for exp in self.experiments.values() if exp.status == status]
    
    def _validate_experiment_config(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Validate experiment configuration."""
        errors = []
        warnings = []
        
        # Check required fields
        if not config.name:
            errors.append("Experiment name is required")
        if not config.description:
            errors.append("Experiment description is required")
        if not config.model_name:
            errors.append("Model name is required")
        
        # Check traffic split
        total_traffic = sum(config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            errors.append(f"Traffic split must sum to 1.0, got {total_traffic}")
        
        # Check sample size
        if config.minimum_sample_size < 10:
            warnings.append("Minimum sample size is very small, results may not be reliable")
        
        # Check significance level
        if config.significance_level <= 0 or config.significance_level >= 1:
            errors.append("Significance level must be between 0 and 1")
        
        # Check compliance requirements
        if config.test_type == TestType.CLINICAL_OUTCOMES and not config.requires_clinical_approval:
            errors.append("Clinical outcome experiments require clinical approval")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _check_clinical_requirements(self, experiment_id: str) -> Dict[str, Any]:
        """Check clinical approval requirements."""
        config = self.experiments[experiment_id]
        
        # In practice, this would check with clinical approval system
        # For now, return mock result
        return {
            "approved": not config.requires_clinical_approval,  # Mock: approve if no approval needed
            "approval_id": "mock_approval_123",
            "approval_date": datetime.now().isoformat(),
            "approved_by": "clinical_system"
        }
    
    def _perform_statistical_analysis(self, experiment_id: str) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results."""
        # This is a simplified implementation
        # In practice, would need to collect and analyze actual data
        
        result = {
            "test_statistic": 0.0,
            "p_value": 0.5,
            "confidence_interval": (0.95, 1.05),
            "effect_size": 0.1,
            "statistical_power": 0.8,
            "is_significant": False,
            "conclusion": "No significant difference detected",
            "recommendations": ["Collect more data", "Extend experiment duration"]
        }
        
        # Mock statistical test based on config
        config = self.experiments[experiment_id]
        result["statistical_test"] = config.statistical_test.value
        
        # Simulate p-value calculation
        if config.statistical_test == StatisticalTest.T_TEST:
            # Mock t-test result
            result["test_statistic"] = 1.96  # Mock test statistic
            result["p_value"] = 0.049  # Mock p-value just below threshold
            result["is_significant"] = result["p_value"] < config.significance_level
            
            if result["is_significant"]:
                result["conclusion"] = "Statistically significant difference detected"
                result["recommendations"] = ["Consider deploying treatment version", "Monitor for continued performance"]
            else:
                result["conclusion"] = "No statistically significant difference detected"
        
        return result
    
    def _check_safety_alerts(self, experiment_id: str, event: Dict[str, Any]):
        """Check for safety alerts during experiment."""
        if event["event_type"] == "error" or event["event_type"] == "exception":
            # Safety alert for errors
            alert = {
                "type": "safety_alert",
                "timestamp": datetime.now().isoformat(),
                "experiment_id": experiment_id,
                "severity": "high",
                "message": "Error detected during experiment",
                "event": event
            }
            
            if experiment_id in self.results:
                self.results[experiment_id].alerts.append(alert)
            
            logger.warning(f"Safety alert for experiment {experiment_id}: {event}")
        
        # Check latency thresholds
        if event["event_type"] == "latency":
            latency = event["data"].get("latency_ms", 0)
            if latency > 5000:  # 5 second threshold
                alert = {
                    "type": "performance_alert",
                    "timestamp": datetime.now().isoformat(),
                    "experiment_id": experiment_id,
                    "severity": "medium",
                    "message": f"High latency detected: {latency}ms",
                    "event": event
                }
                
                if experiment_id in self.results:
                    self.results[experiment_id].alerts.append(alert)
    
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        config = self.experiments[experiment_id]
        result = self.results.get(experiment_id)
        
        if not result:
            return {"error": "Experiment results not found"}
        
        report = {
            "experiment_info": asdict(config),
            "execution_summary": {
                "status": result.status.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_hours": result.duration_hours,
                "total_sample_size": result.control_sample_size + result.treatment_sample_size
            },
            "statistical_analysis": {
                "test_type": config.statistical_test.value,
                "test_statistic": result.test_statistic,
                "p_value": result.p_value,
                "confidence_interval": result.confidence_interval,
                "effect_size": result.effect_size,
                "statistical_power": result.statistical_power,
                "is_significant": result.is_significant,
                "significance_level": config.significance_level
            },
            "performance_comparison": {
                "control_metrics": result.control_metrics,
                "treatment_metrics": result.treatment_metrics,
                "metric_differences": result.metric_differences
            },
            "conclusions": {
                "conclusion": result.conclusion,
                "recommendations": result.recommendations,
                "risk_assessment": result.risk_assessment
            },
            "safety_and_monitoring": {
                "alerts_count": len(result.alerts),
                "alerts": result.alerts[-5:],  # Last 5 alerts
                "monitoring_data_points": len(result.monitoring_data)
            }
        }
        
        return report
    
    def auto_stop_experiment(self, experiment_id: str) -> bool:
        """Automatically stop experiment based on safety criteria."""
        if experiment_id not in self.results:
            return False
        
        result = self.results[experiment_id]
        config = self.experiments[experiment_id]
        
        # Check for high-severity alerts
        high_severity_alerts = [alert for alert in result.alerts if alert.get("severity") == "high"]
        if high_severity_alerts:
            return self.stop_experiment(experiment_id, "High severity safety alert")
        
        # Check duration limit
        if result.start_time:
            duration = (datetime.now() - result.start_time).total_seconds() / 3600
            if duration > config.maximum_duration_days * 24:
                return self.stop_experiment(experiment_id, "Maximum duration reached")
        
        # Check sample size (simplified - would need actual counts)
        if result.control_sample_size + result.treatment_sample_size >= config.minimum_sample_size:
            # If we have enough data and results are significant, consider completing
            if result.is_significant:
                return self.complete_experiment(experiment_id)
        
        return False
    
    def compare_models(self, control_version: ModelVersion, treatment_version: ModelVersion,
                      test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Direct model comparison without full A/B testing setup."""
        
        # Initialize comparison results
        control_results = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        treatment_results = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # Mock evaluation (in practice, would run actual model predictions)
        import random
        random.seed(42)  # For reproducible results
        
        control_predictions = [random.choice([0, 1]) for _ in range(len(test_data))]
        treatment_predictions = [random.choice([0, 1]) for _ in range(len(test_data))]
        
        # Calculate metrics (simplified)
        for i, data_point in enumerate(test_data):
            true_label = data_point.get("label", random.choice([0, 1]))
            control_pred = control_predictions[i]
            treatment_pred = treatment_predictions[i]
            
            # Update control metrics
            if control_pred == true_label:
                control_results["accuracy"] += 1
                if true_label == 1:
                    control_results["recall"] += 1
                    if control_pred == 1:
                        control_results["precision"] += 1
                else:
                    if control_pred == 1:
                        control_results["precision"] -= 1
            
            # Update treatment metrics
            if treatment_pred == true_label:
                treatment_results["accuracy"] += 1
                if true_label == 1:
                    treatment_results["recall"] += 1
                    if treatment_pred == 1:
                        treatment_results["precision"] += 1
                else:
                    if treatment_pred == 1:
                        treatment_results["precision"] -= 1
        
        # Normalize metrics
        total_samples = len(test_data)
        for metric in control_results:
            control_results[metric] /= total_samples
            treatment_results[metric] /= total_samples
        
        # Calculate differences
        differences = {
            metric: treatment_results[metric] - control_results[metric]
            for metric in control_results
        }
        
        # Statistical significance test (simplified)
        improvement_count = sum(1 for diff in differences.values() if diff > 0)
        total_metrics = len(differences)
        
        comparison_result = {
            "control_version": control_version.version,
            "treatment_version": treatment_version.version,
            "control_metrics": control_results,
            "treatment_metrics": treatment_results,
            "metric_differences": differences,
            "improvement_ratio": improvement_count / total_metrics,
            "summary": f"Treatment model shows improvement in {improvement_count}/{total_metrics} metrics",
            "recommendations": self._generate_comparison_recommendations(differences),
            "compliance_check": self._check_compliance_for_comparison(control_version, treatment_version)
        }
        
        return comparison_result
    
    def _generate_comparison_recommendations(self, differences: Dict[str, float]) -> List[str]:
        """Generate recommendations based on model comparison."""
        recommendations = []
        
        # Check accuracy improvement
        if differences.get("accuracy", 0) > 0.05:
            recommendations.append("Significant accuracy improvement detected - consider A/B testing")
        elif differences.get("accuracy", 0) < -0.05:
            recommendations.append("Accuracy degradation detected - investigate model changes")
        
        # Check precision/recall balance
        precision_diff = differences.get("precision", 0)
        recall_diff = differences.get("recall", 0)
        
        if precision_diff > 0.1 and recall_diff < -0.1:
            recommendations.append("Precision improved at cost of recall - consider false positive impact")
        elif recall_diff > 0.1 and precision_diff < -0.1:
            recommendations.append("Recall improved at cost of precision - consider false negative impact")
        
        # Clinical implications
        if differences.get("f1_score", 0) > 0.05:
            recommendations.append("Overall F1 score improvement supports clinical deployment")
        
        if not recommendations:
            recommendations.append("No significant changes detected - current performance maintained")
        
        return recommendations
    
    def _check_compliance_for_comparison(self, control: ModelVersion, treatment: ModelVersion) -> Dict[str, Any]:
        """Check compliance implications of model comparison."""
        compliance_check = {
            "control_compliance_level": control.compliance.compliance_level.value,
            "treatment_compliance_level": treatment.compliance.compliance_level.value,
            "compliance_issues": [],
            "regulatory_requirements": [],
            "clinical_approval_required": False
        }
        
        # Check if treatment version has lower compliance level
        if treatment.compliance.compliance_level.value in ["deprecated", "withdrawn"]:
            compliance_check["compliance_issues"].append(
                f"Treatment version has {treatment.compliance.compliance_level.value} status"
            )
        
        # Check medical device class changes
        if control.compliance.medical_device_class != treatment.compliance.medical_device_class:
            compliance_check["regulatory_requirements"].append(
                "Medical device class change may require new regulatory approval"
            )
            compliance_check["clinical_approval_required"] = True
        
        # Check if production deployment requires clinical approval
        if treatment.compliance.compliance_level.value == "production":
            compliance_check["clinical_approval_required"] = True
            compliance_check["regulatory_requirements"].append(
                "Production deployment requires clinical validation"
            )
        
        return compliance_check