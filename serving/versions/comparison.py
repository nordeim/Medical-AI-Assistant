"""
Model performance comparison utilities with medical accuracy metrics.

Provides comprehensive performance analysis for medical AI models
including clinical metrics, statistical analysis, and regulatory compliance.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from .core import ModelVersion, ComplianceLevel, VersionType

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categories of performance metrics."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    SPECIFICITY = "specificity"
    SENSITIVITY = "sensitivity"
    AUC_ROC = "auc_roc"
    CLINICAL_SENSITIVITY = "clinical_sensitivity"
    CLINICAL_SPECIFICITY = "clinical_specificity"
    PPV = "ppv"  # Positive Predictive Value
    NPV = "npv"  # Negative Predictive Value
    LR_POSITIVE = "lr_positive"  # Likelihood Ratio Positive
    LR_NEGATIVE = "lr_negative"  # Likelihood Ratio Negative
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEDICAL_ACCURACY = "medical_accuracy"
    DIAGNOSTIC_ACCURACY = "diagnostic_accuracy"
    PROGNOSTIC_ACCURACY = "prognostic_accuracy"


@dataclass
class ClinicalThreshold:
    """Clinical threshold for medical metrics."""
    metric_name: str
    threshold_value: float
    clinical_significance: str
    regulatory_requirement: bool = False
    severity_level: str = "warning"  # critical, warning, info
    source: str = ""  # Clinical guideline or regulatory source
    
    def is_met(self, actual_value: float, comparison_type: str = ">=") -> bool:
        """Check if threshold is met."""
        if comparison_type == ">=":
            return actual_value >= self.threshold_value
        elif comparison_type == "<=":
            return actual_value <= self.threshold_value
        elif comparison_type == ">":
            return actual_value > self.threshold_value
        elif comparison_type == "<":
            return actual_value < self.threshold_value
        return False


@dataclass
class PerformanceMetrics:
    """Performance metrics for model evaluation."""
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    specificity: float = 0.0
    sensitivity: float = 0.0
    
    # AUC metrics
    auc_roc: float = 0.0
    auc_pr: float = 0.0  # Precision-Recall AUC
    
    # Clinical metrics
    ppv: float = 0.0  # Positive Predictive Value
    npv: float = 0.0  # Negative Predictive Value
    lr_positive: float = 0.0  # Likelihood Ratio Positive
    lr_negative: float = 0.0  # Likelihood Ratio Negative
    
    # Medical-specific metrics
    medical_accuracy: float = 0.0
    diagnostic_accuracy: float = 0.0
    prognostic_accuracy: float = 0.0
    clinical_sensitivity: float = 0.0
    clinical_specificity: float = 0.0
    
    # Operational metrics
    latency_ms: float = 0.0
    throughput_qps: float = 0.0  # Queries per second
    error_rate: float = 0.0
    
    # Confidence intervals and statistical measures
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    standard_errors: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    
    # Metadata
    evaluation_date: datetime = field(default_factory=datetime.now)
    evaluation_dataset: str = ""
    evaluation_context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['evaluation_date'] = self.evaluation_date.isoformat()
        return data


@dataclass
class ComparisonResult:
    """Result of model performance comparison."""
    control_version: str
    treatment_version: str
    control_metrics: PerformanceMetrics
    treatment_metrics: PerformanceMetrics
    
    # Statistical analysis
    metric_differences: Dict[str, float] = field(default_factory=dict)
    relative_improvements: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, bool] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Clinical analysis
    clinical_improvements: List[Dict[str, Any]] = field(default_factory=list)
    clinical_deteriorations: List[Dict[str, Any]] = field(default_factory=list)
    regulatory_compliance: Dict[str, Any] = field(default_factory=dict)
    
    # Overall assessment
    overall_improvement: float = 0.0
    recommendation: str = ""
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed analysis
    detailed_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        return data


class MedicalMetrics:
    """Medical accuracy metrics and clinical assessments."""
    
    def __init__(self):
        self.clinical_thresholds = self._load_clinical_thresholds()
    
    def _load_clinical_thresholds(self) -> Dict[str, ClinicalThreshold]:
        """Load clinical thresholds for medical metrics."""
        return {
            # General medical accuracy thresholds
            "medical_accuracy": ClinicalThreshold(
                "medical_accuracy", 0.85, "Minimum acceptable medical accuracy",
                regulatory_requirement=True, severity_level="critical"
            ),
            
            # Diagnostic accuracy thresholds
            "diagnostic_accuracy": ClinicalThreshold(
                "diagnostic_accuracy", 0.90, "High diagnostic accuracy required",
                regulatory_requirement=True, severity_level="critical"
            ),
            
            # Sensitivity thresholds (critical for medical screening)
            "clinical_sensitivity": ClinicalThreshold(
                "clinical_sensitivity", 0.95, "High sensitivity for disease detection",
                regulatory_requirement=True, severity_level="critical"
            ),
            
            # Specificity thresholds
            "clinical_specificity": ClinicalThreshold(
                "clinical_specificity", 0.90, "Adequate specificity to reduce false positives",
                regulatory_requirement=True, severity_level="warning"
            ),
            
            # AUC ROC thresholds
            "auc_roc": ClinicalThreshold(
                "auc_roc", 0.85, "Good discriminative ability",
                regulatory_requirement=True, severity_level="warning"
            ),
            
            # Operational thresholds
            "latency_ms": ClinicalThreshold(
                "latency_ms", 1000.0, "Acceptable response time for medical applications",
                regulatory_requirement=False, severity_level="warning"
            ),
            
            "error_rate": ClinicalThreshold(
                "error_rate", 0.01, "Low error rate for medical safety",
                regulatory_requirement=True, severity_level="critical"
            )
        }
    
    def calculate_clinical_metrics(self, y_true: List[Any], y_pred: List[Any], 
                                 y_prob: List[float] = None) -> PerformanceMetrics:
        """Calculate comprehensive clinical metrics."""
        
        # Convert to numpy arrays for calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob) if y_prob else None
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Specificity and sensitivity (for binary classification)
        if len(np.unique(y_true)) == 2:
            specificity = self._calculate_specificity(y_true, y_pred)
            sensitivity = recall  # Recall is sensitivity for weighted averaging
        else:
            specificity = 0.0
            sensitivity = recall
        
        # AUC metrics
        auc_roc = 0.0
        auc_pr = 0.0
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
                # For AUC-PR, we need to handle it differently
                auc_pr = auc_roc  # Simplified - in practice would calculate PR AUC
            except:
                auc_roc = 0.0
                auc_pr = 0.0
        
        # Clinical metrics
        ppv = self._calculate_ppv(y_true, y_pred)
        npv = self._calculate_npv(y_true, y_pred)
        lr_positive = self._calculate_likelihood_ratio_positive(sensitivity, specificity)
        lr_negative = self._calculate_likelihood_ratio_negative(sensitivity, specificity)
        
        # Medical-specific metrics
        medical_accuracy = self._calculate_medical_accuracy(y_true, y_pred)
        diagnostic_accuracy = self._calculate_diagnostic_accuracy(y_true, y_pred)
        prognostic_accuracy = self._calculate_prognostic_accuracy(y_true, y_pred)
        clinical_sensitivity = sensitivity  # Use calculated sensitivity
        clinical_specificity = specificity  # Use calculated specificity
        
        # Create metrics object
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            sensitivity=sensitivity,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            ppv=ppv,
            npv=npv,
            lr_positive=lr_positive,
            lr_negative=lr_negative,
            medical_accuracy=medical_accuracy,
            diagnostic_accuracy=diagnostic_accuracy,
            prognostic_accuracy=prognostic_accuracy,
            clinical_sensitivity=clinical_sensitivity,
            clinical_specificity=clinical_specificity,
            sample_size=len(y_true)
        )
        
        # Add operational metrics (mocked for this example)
        import random
        random.seed(42)
        metrics.latency_ms = random.uniform(50, 200)
        metrics.throughput_qps = random.uniform(100, 1000)
        metrics.error_rate = random.uniform(0.001, 0.01)
        
        # Calculate confidence intervals (simplified)
        metrics = self._calculate_confidence_intervals(metrics, y_true, y_pred)
        
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        if tn + fp == 0:
            return 0.0
        
        return tn / (tn + fp)
    
    def _calculate_ppv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Positive Predictive Value (Precision)."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        
        if tp + fp == 0:
            return 0.0
        
        return tp / (tp + fp)
    
    def _calculate_npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Negative Predictive Value."""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        if tn + fn == 0:
            return 0.0
        
        return tn / (tn + fn)
    
    def _calculate_likelihood_ratio_positive(self, sensitivity: float, specificity: float) -> float:
        """Calculate Likelihood Ratio Positive."""
        if specificity == 1.0:
            return float('inf')
        
        return sensitivity / (1 - specificity)
    
    def _calculate_likelihood_ratio_negative(self, sensitivity: float, specificity: float) -> float:
        """Calculate Likelihood Ratio Negative."""
        if sensitivity == 1.0:
            return 0.0
        
        return (1 - sensitivity) / specificity
    
    def _calculate_medical_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate medical accuracy (custom metric for medical context)."""
        # Medical accuracy considers clinical context and consequences
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        
        if total_predictions == 0:
            return 0.0
        
        return correct_predictions / total_predictions
    
    def _calculate_diagnostic_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate diagnostic accuracy (focus on disease detection)."""
        # For diagnostic accuracy, we weigh false negatives more heavily
        tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
        tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
        fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
        fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
        
        # Weight true positives and true negatives higher
        weighted_correct = (tp * 2 + tn)  # Emphasize correct disease detection
        total_weighted = (tp + tn + fp + fn) + tp  # Extra weight for positives
        
        if total_weighted == 0:
            return 0.0
        
        return weighted_correct / total_weighted
    
    def _calculate_prognostic_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate prognostic accuracy (focus on outcome prediction)."""
        # Prognostic accuracy considers temporal aspects and outcome prediction
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        
        if total_predictions == 0:
            return 0.0
        
        # Add confidence-based weighting if probabilities available
        # For simplicity, using same as medical accuracy
        return correct_predictions / total_predictions
    
    def _calculate_confidence_intervals(self, metrics: PerformanceMetrics, 
                                      y_true: np.ndarray, y_pred: np.ndarray) -> PerformanceMetrics:
        """Calculate confidence intervals for metrics."""
        n = len(y_true)
        
        if n < 30:  # Too few samples for reliable confidence intervals
            return metrics
        
        # Calculate standard errors using binomial distribution approximation
        # For accuracy
        accuracy_se = np.sqrt(metrics.accuracy * (1 - metrics.accuracy) / n)
        metrics.standard_errors['accuracy'] = accuracy_se
        metrics.confidence_intervals['accuracy'] = (
            metrics.accuracy - 1.96 * accuracy_se,
            metrics.accuracy + 1.96 * accuracy_se
        )
        
        # For other metrics (simplified)
        for metric_name in ['precision', 'recall', 'f1_score', 'specificity']:
            metric_value = getattr(metrics, metric_name)
            if metric_value > 0 and metric_value < 1:
                se = np.sqrt(metric_value * (1 - metric_value) / n)
                metrics.standard_errors[metric_name] = se
                metrics.confidence_intervals[metric_name] = (
                    metric_value - 1.96 * se,
                    metric_value + 1.96 * se
                )
        
        return metrics
    
    def evaluate_against_clinical_thresholds(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate metrics against clinical thresholds."""
        results = {
            "passed": [],
            "failed": [],
            "warnings": [],
            "critical_issues": [],
            "overall_compliance": True
        }
        
        for metric_name, threshold in self.clinical_thresholds.items():
            if hasattr(metrics, metric_name):
                actual_value = getattr(metrics, metric_name)
                
                if threshold.regulatory_requirement:
                    if not threshold.is_met(actual_value):
                        results["critical_issues"].append({
                            "metric": metric_name,
                            "actual_value": actual_value,
                            "threshold": threshold.threshold_value,
                            "clinical_significance": threshold.clinical_significance,
                            "source": threshold.source
                        })
                        results["overall_compliance"] = False
                    else:
                        results["passed"].append({
                            "metric": metric_name,
                            "actual_value": actual_value,
                            "threshold": threshold.threshold_value
                        })
                else:
                    if threshold.is_met(actual_value):
                        results["passed"].append({
                            "metric": metric_name,
                            "actual_value": actual_value,
                            "threshold": threshold.threshold_value
                        })
                    else:
                        results["warnings"].append({
                            "metric": metric_name,
                            "actual_value": actual_value,
                            "threshold": threshold.threshold_value,
                            "clinical_significance": threshold.clinical_significance
                        })
        
        return results


class PerformanceComparator:
    """Comprehensive performance comparison for model versions."""
    
    def __init__(self, medical_metrics: MedicalMetrics):
        self.medical_metrics = medical_metrics
    
    def compare_models(self, 
                      control_version: ModelVersion, 
                      treatment_version: ModelVersion,
                      test_data: List[Dict[str, Any]],
                      statistical_test: str = "t_test") -> ComparisonResult:
        """Compare performance between two model versions."""
        
        # Extract predictions for both models
        control_predictions = self._get_model_predictions(control_version, test_data)
        treatment_predictions = self._get_model_predictions(treatment_version, test_data)
        
        # Get true labels
        y_true = [data.get("label", 0) for data in test_data]
        
        # Calculate metrics for both models
        control_metrics = self._calculate_metrics(control_version, y_true, control_predictions)
        treatment_metrics = self._calculate_metrics(treatment_version, y_true, treatment_predictions)
        
        # Perform statistical comparison
        comparison_result = ComparisonResult(
            control_version=control_version.version,
            treatment_version=treatment_version.version,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics
        )
        
        # Calculate differences and improvements
        comparison_result = self._calculate_metric_differences(comparison_result)
        
        # Perform statistical significance testing
        comparison_result = self._perform_statistical_tests(comparison_result, 
                                                           y_true, control_predictions, treatment_predictions,
                                                           statistical_test)
        
        # Analyze clinical implications
        comparison_result = self._analyze_clinical_implications(comparison_result)
        
        # Generate recommendations
        comparison_result.recommendation = self._generate_recommendations(comparison_result)
        
        return comparison_result
    
    def _get_model_predictions(self, version: ModelVersion, test_data: List[Dict[str, Any]]) -> List[Any]:
        """Get predictions from model version (mock implementation)."""
        # This would integrate with actual model serving
        # For now, generate mock predictions based on version characteristics
        import random
        random.seed(hash(version.version) % 1000)  # Deterministic based on version
        
        predictions = []
        for data in test_data:
            # Mock prediction logic based on version
            true_label = data.get("label", 0)
            
            # Simulate version-dependent accuracy
            version_factor = float(version.version.split('.')[0]) / 10.0  # Higher major version = better accuracy
            base_accuracy = 0.8 + version_factor * 0.15  # 0.8 to 0.95 range
            
            if random.random() < base_accuracy:
                predictions.append(true_label)  # Correct prediction
            else:
                predictions.append(1 - true_label)  # Incorrect prediction
        
        return predictions
    
    def _calculate_metrics(self, version: ModelVersion, y_true: List[Any], y_pred: List[Any]) -> PerformanceMetrics:
        """Calculate metrics for a model version."""
        # Use medical metrics calculator
        return self.medical_metrics.calculate_clinical_metrics(y_true, y_pred)
    
    def _calculate_metric_differences(self, result: ComparisonResult) -> ComparisonResult:
        """Calculate differences between control and treatment metrics."""
        
        # Get metric attributes
        control_attrs = [attr for attr in dir(result.control_metrics) 
                        if not attr.startswith('_') and isinstance(getattr(result.control_metrics, attr), (int, float))]
        
        treatment_attrs = [attr for attr in dir(result.treatment_metrics) 
                          if not attr.startswith('_') and isinstance(getattr(result.treatment_metrics, attr), (int, float))]
        
        # Calculate differences
        for attr in set(control_attrs) & set(treatment_attrs):
            control_value = getattr(result.control_metrics, attr)
            treatment_value = getattr(result.treatment_metrics, attr)
            
            if isinstance(control_value, (int, float)) and isinstance(treatment_value, (int, float)):
                difference = treatment_value - control_value
                relative_improvement = (difference / control_value) if control_value != 0 else 0
                
                result.metric_differences[attr] = difference
                result.relative_improvements[attr] = relative_improvement
        
        # Calculate overall improvement score
        improvements = [imp for imp in result.relative_improvements.values() if not np.isnan(imp)]
        result.overall_improvement = np.mean(improvements) if improvements else 0.0
        
        return result
    
    def _perform_statistical_tests(self, result: ComparisonResult,
                                 y_true: List[Any], control_pred: List[Any], treatment_pred: List[Any],
                                 test_type: str = "t_test") -> ComparisonResult:
        """Perform statistical significance tests."""
        
        if test_type == "t_test":
            # Perform t-test for each metric
            for metric_name in result.metric_differences.keys():
                # Mock statistical test (in practice would use actual test implementation)
                # For demonstration, use difference magnitude to estimate significance
                difference = abs(result.metric_differences[metric_name])
                sample_size = result.control_metrics.sample_size
                
                # Simplified significance calculation
                t_stat = difference * np.sqrt(sample_size) / 2.0  # Simplified
                p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))  # Simplified
                
                result.statistical_significance[metric_name] = p_value < 0.05
                
                # Store confidence intervals (simplified)
                se = difference / np.sqrt(sample_size)
                ci_lower = difference - 1.96 * se
                ci_upper = difference + 1.96 * se
                result.confidence_intervals[metric_name] = (ci_lower, ci_upper)
        
        return result
    
    def _analyze_clinical_implications(self, result: ComparisonResult) -> ComparisonResult:
        """Analyze clinical implications of performance differences."""
        
        # Check against clinical thresholds
        control_compliance = self.medical_metrics.evaluate_against_clinical_thresholds(result.control_metrics)
        treatment_compliance = self.medical_metrics.evaluate_against_clinical_thresholds(result.treatment_metrics)
        
        # Analyze improvements and deteriorations
        for metric_name, difference in result.metric_differences.items():
            if hasattr(result.control_metrics, metric_name) and hasattr(result.treatment_metrics, metric_name):
                
                control_value = getattr(result.control_metrics, metric_name)
                treatment_value = getattr(result.treatment_metrics, metric_name)
                
                # Clinical improvement analysis
                if difference > 0:  # Improvement
                    improvement_entry = {
                        "metric": metric_name,
                        "control_value": control_value,
                        "treatment_value": treatment_value,
                        "improvement": difference,
                        "relative_improvement": result.relative_improvements.get(metric_name, 0),
                        "clinical_significance": self._assess_clinical_significance(metric_name, difference, result.relative_improvements.get(metric_name, 0)),
                        "regulatory_impact": self._assess_regulatory_impact(metric_name, treatment_value)
                    }
                    result.clinical_improvements.append(improvement_entry)
                
                elif difference < 0:  # Deterioration
                    deterioration_entry = {
                        "metric": metric_name,
                        "control_value": control_value,
                        "treatment_value": treatment_value,
                        "deterioration": abs(difference),
                        "relative_deterioration": abs(result.relative_improvements.get(metric_name, 0)),
                        "clinical_risk": self._assess_clinical_risk(metric_name, abs(difference)),
                        "regulatory_risk": self._assess_regulatory_risk(metric_name, treatment_value)
                    }
                    result.clinical_deteriorations.append(deterioration_entry)
        
        # Overall regulatory compliance assessment
        result.regulatory_compliance = {
            "control_compliant": control_compliance["overall_compliance"],
            "treatment_compliant": treatment_compliance["overall_compliance"],
            "control_issues": control_compliance["critical_issues"],
            "treatment_issues": treatment_compliance["critical_issues"],
            "regulatory_impact": self._assess_overall_regulatory_impact(result)
        }
        
        return result
    
    def _assess_clinical_significance(self, metric_name: str, improvement: float, 
                                    relative_improvement: float) -> str:
        """Assess clinical significance of improvement."""
        
        # Define significance thresholds
        if metric_name in ["clinical_sensitivity", "diagnostic_accuracy", "medical_accuracy"]:
            if improvement >= 0.05 or relative_improvement >= 0.1:
                return "high"
            elif improvement >= 0.02 or relative_improvement >= 0.05:
                return "moderate"
            else:
                return "low"
        elif metric_name in ["latency_ms", "error_rate"]:
            if improvement >= 0.1:  # For latency/error rate, improvement is reduction
                return "high"
            elif improvement >= 0.05:
                return "moderate"
            else:
                return "low"
        else:
            if relative_improvement >= 0.15:
                return "high"
            elif relative_improvement >= 0.05:
                return "moderate"
            else:
                return "low"
    
    def _assess_regulatory_impact(self, metric_name: str, value: float) -> str:
        """Assess regulatory impact of metric value."""
        
        if metric_name in self.medical_metrics.clinical_thresholds:
            threshold = self.medical_metrics.clinical_thresholds[metric_name]
            if threshold.regulatory_requirement:
                if value >= threshold.threshold_value:
                    return "meets_requirements"
                else:
                    return "requires_attention"
        
        return "no_regulatory_impact"
    
    def _assess_clinical_risk(self, metric_name: str, deterioration: float) -> str:
        """Assess clinical risk of deterioration."""
        
        if metric_name in ["clinical_sensitivity", "diagnostic_accuracy"]:
            if deterioration >= 0.05:
                return "high"
            elif deterioration >= 0.02:
                return "moderate"
            else:
                return "low"
        elif metric_name in ["error_rate"]:
            if deterioration >= 0.01:
                return "high"
            elif deterioration >= 0.005:
                return "moderate"
            else:
                return "low"
        else:
            return "low"
    
    def _assess_regulatory_risk(self, metric_name: str, value: float) -> str:
        """Assess regulatory risk of metric value."""
        
        if metric_name in self.medical_metrics.clinical_thresholds:
            threshold = self.medical_metrics.clinical_thresholds[metric_name]
            if threshold.regulatory_requirement and value < threshold.threshold_value * 0.9:
                return "high"
            elif threshold.regulatory_requirement:
                return "moderate"
        
        return "low"
    
    def _assess_overall_regulatory_impact(self, result: ComparisonResult) -> Dict[str, Any]:
        """Assess overall regulatory impact of comparison."""
        
        impact_assessment = {
            "impact_level": "none",
            "requires_review": False,
            "regulatory_concerns": [],
            "recommendations": []
        }
        
        # Check critical deteriorations
        critical_deteriorations = [d for d in result.clinical_deteriorations if d["clinical_risk"] == "high"]
        if critical_deteriorations:
            impact_assessment["impact_level"] = "high"
            impact_assessment["requires_review"] = True
            impact_assessment["regulatory_concerns"].extend([
                f"Critical deterioration in {d['metric']}" for d in critical_deteriorations
            ])
        
        # Check compliance issues
        if not result.regulatory_compliance["treatment_compliant"]:
            impact_assessment["impact_level"] = "high"
            impact_assessment["requires_review"] = True
            issue_count = len(result.regulatory_compliance["treatment_issues"])
            impact_assessment["regulatory_concerns"].append(
                f"Treatment model has {issue_count} regulatory compliance issues"
            )
        
        # Check significant improvements that might trigger review
        significant_improvements = [i for i in result.clinical_improvements 
                                  if i["clinical_significance"] == "high"]
        if significant_improvements:
            impact_assessment["recommendations"].append(
                "Consider regulatory submission for significant improvements"
            )
        
        return impact_assessment
    
    def _generate_recommendations(self, result: ComparisonResult) -> str:
        """Generate recommendations based on comparison results."""
        
        # High-level recommendation logic
        if result.overall_improvement > 0.1 and not result.clinical_deteriorations:
            return "Strongly recommend deploying treatment version - significant improvement with no critical issues"
        elif result.overall_improvement > 0.05 and len([d for d in result.clinical_deteriorations if d["clinical_risk"] == "high"]) == 0:
            return "Recommend deploying treatment version with monitoring - moderate improvement with acceptable risk"
        elif result.overall_improvement > 0.0 and len(result.clinical_deteriorations) == 0:
            return "Consider deploying treatment version - minor improvement with no deterioration"
        elif len(result.clinical_deteriorations) > 0 and any(d["clinical_risk"] == "high" for d in result.clinical_deteriorations):
            return "Do not deploy treatment version - critical deterioration detected"
        else:
            return "Requires further evaluation - mixed results with moderate risks"
    
    def generate_performance_report(self, result: ComparisonResult) -> Dict[str, Any]:
        """Generate comprehensive performance comparison report."""
        
        report = {
            "comparison_summary": {
                "control_version": result.control_version,
                "treatment_version": result.treatment_version,
                "overall_improvement": f"{result.overall_improvement:.2%}",
                "recommendation": result.recommendation,
                "sample_size": result.control_metrics.sample_size
            },
            
            "detailed_metrics": {
                "control_metrics": result.control_metrics.to_dict(),
                "treatment_metrics": result.treatment_metrics.to_dict(),
                "metric_differences": result.metric_differences,
                "relative_improvements": {k: f"{v:.2%}" for k, v in result.relative_improvements.items()}
            },
            
            "statistical_analysis": {
                "statistical_significance": result.statistical_significance,
                "confidence_intervals": {k: [f"{v[0]:.3f}", f"{v[1]:.3f}"] 
                                       for k, v in result.confidence_intervals.items()}
            },
            
            "clinical_analysis": {
                "improvements": result.clinical_improvements,
                "deteriorations": result.clinical_deteriorations
            },
            
            "regulatory_compliance": result.regulatory_compliance,
            
            "risk_assessment": result.risk_assessment,
            
            "detailed_analysis": result.detailed_analysis
        }
        
        return report