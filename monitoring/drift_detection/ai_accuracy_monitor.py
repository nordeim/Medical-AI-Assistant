"""
AI Accuracy Monitoring and Model Drift Detection System
Provides comprehensive monitoring for medical AI model performance, drift detection,
and clinical validation with statistical tests and fairness metrics.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPerformanceMetrics:
    """Container for model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    sensitivity: float
    specificity: float
    timestamp: datetime
    sample_size: int
    confidence_interval_95: Tuple[float, float]

class ModelDriftDetector:
    """Advanced model drift detection with multiple statistical tests"""
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 baseline_window: int = 1000,
                 drift_threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            significance_level: Statistical significance level for tests
            baseline_window: Size of baseline window for comparison
            drift_threshold: Threshold for declaring drift
        """
        self.significance_level = significance_level
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.baseline_performance = {}
        self.drift_history = []
        
    def detect_data_drift(self, 
                         baseline_data: np.ndarray, 
                         current_data: np.ndarray,
                         test_type: str = 'ks') -> Dict[str, Any]:
        """
        Detect data distribution drift using statistical tests
        
        Args:
            baseline_data: Baseline data distribution
            current_data: Current data distribution
            test_type: Type of test ('ks', 'chi2', 'ks_weighted')
            
        Returns:
            Dict containing drift detection results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'test_type': test_type,
            'drift_detected': False,
            'test_statistics': {},
            'p_values': {},
            'effect_sizes': {}
        }
        
        try:
            if test_type == 'ks':
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(baseline_data, current_data)
                effect_size = statistic
                
            elif test_type == 'chi2':
                # Chi-square test for categorical data
                # Create bins for continuous data
                bins = np.linspace(min(baseline_data.min(), current_data.min()),
                                 max(baseline_data.max(), current_data.max()),
                                 20)
                
                baseline_hist, _ = np.histogram(baseline_data, bins=bins)
                current_hist, _ = np.histogram(current_data, bins=bins)
                
                statistic, p_value = stats.chisquare(current_hist, baseline_hist)
                effect_size = statistic / len(current_hist)
                
            elif test_type == 'ks_weighted':
                # Weighted KS test for time series
                weights_baseline = np.exp(-0.001 * np.arange(len(baseline_data)))
                weights_current = np.ones(len(current_data))
                
                statistic, p_value = stats.ks_2samp(baseline_data, current_data)
                effect_size = statistic * np.sum(weights_baseline)
                
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            results['test_statistics'] = {'statistic': float(statistic)}
            results['p_values'] = {'p_value': float(p_value)}
            results['effect_sizes'] = {'effect_size': float(effect_size)}
            
            # Determine drift based on p-value and effect size
            if p_value < self.significance_level and effect_size > self.drift_threshold:
                results['drift_detected'] = True
                results['severity'] = 'high' if effect_size > 2 * self.drift_threshold else 'medium'
            else:
                results['severity'] = 'none'
                
        except Exception as e:
            logging.error(f"Error in data drift detection: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def detect_performance_drift(self,
                                baseline_predictions: np.ndarray,
                                baseline_labels: np.ndarray,
                                current_predictions: np.ndarray,
                                current_labels: np.ndarray,
                                task_type: str = 'classification') -> Dict[str, Any]:
        """
        Detect performance drift in model predictions
        
        Args:
            baseline_predictions: Baseline model predictions
            baseline_labels: Ground truth labels for baseline
            current_predictions: Current model predictions
            current_labels: Ground truth labels for current
            task_type: Type of task ('classification', 'regression')
            
        Returns:
            Dict containing performance drift analysis
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'task_type': task_type,
            'drift_detected': False,
            'performance_metrics': {},
            'statistical_tests': {}
        }
        
        try:
            # Calculate performance metrics for both periods
            if task_type == 'classification':
                baseline_metrics = self._calculate_classification_metrics(
                    baseline_predictions, baseline_labels)
                current_metrics = self._calculate_classification_metrics(
                    current_predictions, current_labels)
            else:
                baseline_metrics = self._calculate_regression_metrics(
                    baseline_predictions, baseline_labels)
                current_metrics = self._calculate_regression_metrics(
                    current_predictions, current_labels)
            
            results['performance_metrics'] = {
                'baseline': baseline_metrics,
                'current': current_metrics
            }
            
            # Statistical tests for performance differences
            performance_diffs = []
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in baseline_metrics and metric in current_metrics:
                    diff = current_metrics[metric] - baseline_metrics[metric]
                    performance_diffs.append(diff)
            
            # Paired t-test for performance differences
            if len(performance_diffs) > 1:
                t_stat, p_value = stats.ttest_1samp(performance_diffs, 0)
                results['statistical_tests']['t_test'] = {
                    'statistic': float(t_stat),
                    'p_value': float(p_value)
                }
                
                # Bootstrap confidence intervals
                confidence_interval = self._bootstrap_confidence_interval(
                    current_predictions, current_labels, task_type)
                results['statistical_tests']['confidence_interval'] = confidence_interval
            
            # Determine if significant drift occurred
            if p_value < self.significance_level:
                results['drift_detected'] = True
                results['severity'] = 'high' if abs(t_stat) > 2.58 else 'medium'
            else:
                results['severity'] = 'none'
                
        except Exception as e:
            logging.error(f"Error in performance drift detection: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def _calculate_classification_metrics(self, 
                                        predictions: np.ndarray, 
                                        labels: np.ndarray) -> Dict[str, float]:
        """Calculate classification performance metrics"""
        try:
            return {
                'accuracy': float(accuracy_score(labels, predictions)),
                'precision': float(precision_score(labels, predictions, average='weighted', zero_division=0)),
                'recall': float(recall_score(labels, predictions, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(labels, predictions, average='weighted', zero_division=0))
            }
        except Exception:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    def _calculate_regression_metrics(self,
                                   predictions: np.ndarray,
                                   labels: np.ndarray) -> Dict[str, float]:
        """Calculate regression performance metrics"""
        try:
            mse = np.mean((predictions - labels) ** 2)
            mae = np.mean(np.abs(predictions - labels))
            r2 = 1 - (np.sum((labels - predictions) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2)
            }
        except Exception:
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': 0.0}
    
    def _bootstrap_confidence_interval(self,
                                     predictions: np.ndarray,
                                     labels: np.ndarray,
                                     task_type: str,
                                     n_bootstrap: int = 1000) -> Dict[str, float]:
        """Calculate bootstrap confidence intervals for performance metrics"""
        try:
            bootstrap_scores = []
            n_samples = len(predictions)
            
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(n_samples, n_samples, replace=True)
                boot_preds = predictions[indices]
                boot_labels = labels[indices]
                
                # Calculate performance
                if task_type == 'classification':
                    score = accuracy_score(boot_labels, boot_preds)
                else:
                    score = 1 - np.mean((boot_preds - boot_labels) ** 2)  # Pseudo R²
                
                bootstrap_scores.append(score)
            
            # Calculate confidence interval
            lower = np.percentile(bootstrap_scores, 2.5)
            upper = np.percentile(bootstrap_scores, 97.5)
            
            return {
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'mean': float(np.mean(bootstrap_scores))
            }
        except Exception:
            return {'lower_bound': 0.0, 'upper_bound': 1.0, 'mean': 0.5}
    
    def detect_concept_drift(self,
                           predictions_proba: np.ndarray,
                           timestamps: List[datetime],
                           window_size: int = 100) -> Dict[str, Any]:
        """
        Detect concept drift in prediction probabilities over time
        
        Args:
            predictions_proba: Model prediction probabilities
            timestamps: Timestamps for predictions
            window_size: Size of sliding window for analysis
            
        Returns:
            Dict containing concept drift analysis
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'concept_drift_detected': False,
            'drift_locations': [],
            'drift_magnitude': [],
            'adaptive_metrics': {}
        }
        
        try:
            if len(predictions_proba) < 2 * window_size:
                results['error'] = "Insufficient data for concept drift detection"
                return results
            
            # Calculate entropy over time (measure of uncertainty)
            entropies = []
            for i in range(window_size, len(predictions_proba)):
                window_probs = predictions_proba[i-window_size:i]
                # Assume binary classification, use max probability
                max_probs = np.max(window_probs, axis=1)
                entropy = -np.sum(max_probs * np.log(max_probs + 1e-10))
                entropies.append(entropy)
            
            results['adaptive_metrics']['entropy_series'] = entropies
            
            # Detect significant changes in entropy
            if len(entropies) > 1:
                entropy_changes = np.diff(entropies)
                threshold = np.std(entropy_changes) * 2
                
                drift_indices = np.where(np.abs(entropy_changes) > threshold)[0]
                results['drift_locations'] = [timestamps[i + window_size] for i in drift_indices]
                results['drift_magnitude'] = entropy_changes[drift_indices].tolist()
                
                if len(drift_indices) > 0:
                    results['concept_drift_detected'] = True
                    results['drift_count'] = len(drift_indices)
            
        except Exception as e:
            logging.error(f"Error in concept drift detection: {str(e)}")
            results['error'] = str(e)
            
        return results

class BiasDetectionSystem:
    """System for detecting and measuring bias in medical AI models"""
    
    def __init__(self, protected_attributes: List[str]):
        """
        Initialize bias detection system
        
        Args:
            protected_attributes: List of protected attribute names
        """
        self.protected_attributes = protected_attributes
        
    def calculate_fairness_metrics(self,
                                 predictions: np.ndarray,
                                 labels: np.ndarray,
                                 protected_attributes: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate fairness metrics across protected groups
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            protected_attributes: Dict mapping attribute names to attribute values
            
        Returns:
            Dict containing fairness metrics
        """
        fairness_metrics = {}
        
        try:
            for attr_name, attr_values in protected_attributes.items():
                unique_groups = np.unique(attr_values)
                
                if len(unique_groups) < 2:
                    continue
                
                # Calculate metrics for each group
                group_metrics = {}
                for group in unique_groups:
                    group_mask = attr_values == group
                    group_preds = predictions[group_mask]
                    group_labels = labels[group_mask]
                    
                    if len(group_preds) > 0:
                        group_metrics[group] = {
                            'accuracy': accuracy_score(group_labels, group_preds),
                            'precision': precision_score(group_labels, group_preds, zero_division=0),
                            'recall': recall_score(group_labels, group_preds, zero_division=0)
                        }
                
                # Calculate fairness metrics
                if len(group_metrics) >= 2:
                    accuracies = [m['accuracy'] for m in group_metrics.values()]
                    precisions = [m['precision'] for m in group_metrics.values()]
                    recalls = [m['recall'] for m in group_metrics.values()]
                    
                    fairness_metrics[attr_name] = {
                        'demographic_parity': self._calculate_demographic_parity(predictions, attr_values),
                        'equalized_odds': self._calculate_equalized_odds(predictions, labels, attr_values),
                        'accuracy_difference': max(accuracies) - min(accuracies),
                        'precision_difference': max(precisions) - min(precisions),
                        'recall_difference': max(recalls) - min(recalls)
                    }
                    
        except Exception as e:
            logging.error(f"Error calculating fairness metrics: {str(e)}")
            fairness_metrics['error'] = str(e)
            
        return fairness_metrics
    
    def _calculate_demographic_parity(self,
                                    predictions: np.ndarray,
                                    protected_attr: np.ndarray) -> float:
        """Calculate demographic parity (equal positive prediction rates)"""
        try:
            unique_groups = np.unique(protected_attr)
            positive_rates = []
            
            for group in unique_groups:
                group_mask = protected_attr == group
                group_preds = predictions[group_mask]
                
                if len(group_preds) > 0:
                    positive_rate = np.mean(group_preds)
                    positive_rates.append(positive_rate)
            
            if len(positive_rates) > 1:
                return max(positive_rates) - min(positive_rates)
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_equalized_odds(self,
                                predictions: np.ndarray,
                                labels: np.ndarray,
                                protected_attr: np.ndarray) -> float:
        """Calculate equalized odds (equal true positive and false positive rates)"""
        try:
            unique_groups = np.unique(protected_attr)
            tpr_diffs = []
            fpr_diffs = []
            
            for group in unique_groups:
                group_mask = protected_attr == group
                group_preds = predictions[group_mask]
                group_labels = labels[group_mask]
                
                if len(group_preds) > 0 and len(group_labels) > 0:
                    # Calculate TPR and FPR for this group
                    tp = np.sum((group_preds == 1) & (group_labels == 1))
                    fp = np.sum((group_preds == 1) & (group_labels == 0))
                    tn = np.sum((group_preds == 0) & (group_labels == 0))
                    fn = np.sum((group_preds == 0) & (group_labels == 1))
                    
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                    
                    tpr_diffs.append(tpr)
                    fpr_diffs.append(fpr)
            
            if len(tpr_diffs) > 1:
                tpr_diff = max(tpr_diffs) - min(tpr_diffs)
                fpr_diff = max(fpr_diffs) - min(fpr_diffs)
                return max(tpr_diff, fpr_diff)
            return 0.0
        except Exception:
            return 0.0

class ClinicalValidationSystem:
    """System for clinical validation of AI model performance"""
    
    def __init__(self, validation_horizon_days: int = 30):
        """
        Initialize clinical validation system
        
        Args:
            validation_horizon_days: Days for prospective validation
        """
        self.validation_horizon_days = validation_horizon_days
        
    def validate_clinical_utility(self,
                                ai_recommendations: np.ndarray,
                                clinician_actions: np.ndarray,
                                patient_outcomes: np.ndarray,
                                cost_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate clinical utility of AI recommendations
        
        Args:
            ai_recommendations: AI system recommendations
            clinician_actions: Actual clinician actions taken
            patient_outcomes: Patient outcomes (binary or continuous)
            cost_data: Cost data for cost-effectiveness analysis
            
        Returns:
            Dict containing clinical validation results
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'clinical_utility_score': 0.0,
            'inter_rater_agreement': 0.0,
            'outcome_improvement': 0.0,
            'cost_effectiveness': None,
            'recommendations': []
        }
        
        try:
            # Calculate inter-rater agreement between AI and clinicians
            if len(ai_recommendations) > 0 and len(clinician_actions) > 0:
                validation_results['inter_rater_agreement'] = float(
                    accuracy_score(clinician_actions, ai_recommendations))
            
            # Measure outcome improvement
            if len(patient_outcomes) > 0:
                validation_results['outcome_improvement'] = float(np.mean(patient_outcomes))
            
            # Calculate overall clinical utility score
            utility_components = [
                validation_results['inter_rater_agreement'],
                min(validation_results['outcome_improvement'], 1.0)
            ]
            
            if cost_data is not None:
                # Cost-effectiveness component
                cost_effectiveness = self._calculate_cost_effectiveness(
                    ai_recommendations, patient_outcomes, cost_data)
                validation_results['cost_effectiveness'] = cost_effectiveness
                utility_components.append(cost_effectiveness)
            
            validation_results['clinical_utility_score'] = float(np.mean(utility_components))
            
            # Generate recommendations based on results
            validation_results['recommendations'] = self._generate_recommendations(validation_results)
            
        except Exception as e:
            logging.error(f"Error in clinical validation: {str(e)}")
            validation_results['error'] = str(e)
            
        return validation_results
    
    def _calculate_cost_effectiveness(self,
                                    recommendations: np.ndarray,
                                    outcomes: np.ndarray,
                                    costs: np.ndarray) -> float:
        """Calculate cost-effectiveness ratio"""
        try:
            if len(recommendations) == len(outcomes) == len(costs):
                # Simple cost-effectiveness calculation
                total_cost = np.sum(costs)
                total_outcome = np.sum(outcomes)
                
                if total_outcome > 0:
                    return float(total_outcome / total_cost)
            return 0.0
        except Exception:
            return 0.0
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        try:
            utility_score = results.get('clinical_utility_score', 0)
            agreement = results.get('inter_rater_agreement', 0)
            
            if utility_score < 0.6:
                recommendations.append("Model requires retraining - clinical utility below threshold")
            
            if agreement < 0.7:
                recommendations.append("Improve model alignment with clinical practices")
            
            if results.get('outcome_improvement', 0) < 0.5:
                recommendations.append("Investigate factors affecting patient outcomes")
            
            cost_eff = results.get('cost_effectiveness', 0)
            if cost_eff is not None and cost_eff < 0.1:
                recommendations.append("Review cost-effectiveness of AI recommendations")
            
            if not recommendations:
                recommendations.append("Model performance acceptable - continue monitoring")
                
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate specific recommendations")
            
        return recommendations

class ModelMonitoringOrchestrator:
    """Orchestrates all model monitoring and drift detection activities"""
    
    def __init__(self,
                 model_name: str,
                 protected_attributes: Optional[List[str]] = None,
                 validation_horizon_days: int = 30):
        """
        Initialize model monitoring orchestrator
        
        Args:
            model_name: Name of the model being monitored
            protected_attributes: Protected attributes for bias detection
            validation_horizon_days: Days for clinical validation
        """
        self.model_name = model_name
        self.drift_detector = ModelDriftDetector()
        self.bias_detector = BiasDetectionSystem(protected_attributes or [])
        self.clinical_validator = ClinicalValidationSystem(validation_horizon_days)
        
        self.performance_history = []
        self.drift_history = []
        self.bias_history = []
        
    def run_comprehensive_monitoring(self,
                                   baseline_data: Optional[np.ndarray] = None,
                                   current_data: Optional[np.ndarray] = None,
                                   predictions: Optional[np.ndarray] = None,
                                   labels: Optional[np.ndarray] = None,
                                   protected_attrs: Optional[Dict[str, np.ndarray]] = None,
                                   timestamps: Optional[List[datetime]] = None) -> Dict[str, Any]:
        """
        Run comprehensive model monitoring and analysis
        
        Args:
            baseline_data: Baseline data for drift detection
            current_data: Current data for drift detection
            predictions: Model predictions for performance analysis
            labels: Ground truth labels
            protected_attrs: Protected attributes for bias analysis
            timestamps: Timestamps for time-series analysis
            
        Returns:
            Dict containing comprehensive monitoring results
        """
        monitoring_results = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'drift_detection': {},
            'performance_analysis': {},
            'bias_analysis': {},
            'clinical_validation': {},
            'overall_health_score': 0.0,
            'alerts': []
        }
        
        try:
            # Data drift detection
            if baseline_data is not None and current_data is not None:
                monitoring_results['drift_detection'] = self.drift_detector.detect_data_drift(
                    baseline_data, current_data)
                self.drift_history.append(monitoring_results['drift_detection'])
            
            # Performance drift detection
            if predictions is not None and labels is not None:
                monitoring_results['performance_analysis'] = self.drift_detector.detect_performance_drift(
                    predictions[:len(predictions)//2], labels[:len(labels)//2],
                    predictions[len(predictions)//2:], labels[len(labels)//2:])
                self.performance_history.append(monitoring_results['performance_analysis'])
            
            # Concept drift detection
            if predictions is not None and timestamps is not None:
                if len(predictions) > 0 and predictions.shape[1] > 1:  # Probability predictions
                    monitoring_results['drift_detection']['concept_drift'] = \
                        self.drift_detector.detect_concept_drift(predictions, timestamps)
            
            # Bias detection
            if protected_attrs is not None:
                monitoring_results['bias_analysis'] = self.bias_detector.calculate_fairness_metrics(
                    predictions, labels, protected_attrs)
                self.bias_history.append(monitoring_results['bias_analysis'])
            
            # Clinical validation (simplified for demonstration)
            if predictions is not None and labels is not None:
                # Simulate clinician actions and outcomes for validation
                clinician_actions = np.random.choice([0, 1], size=len(predictions))
                patient_outcomes = labels + np.random.normal(0, 0.1, len(labels))
                costs = np.random.exponential(100, len(predictions))
                
                monitoring_results['clinical_validation'] = self.clinical_validator.validate_clinical_utility(
                    predictions, clinician_actions, patient_outcomes, costs)
            
            # Calculate overall health score
            monitoring_results['overall_health_score'] = self._calculate_overall_health_score(monitoring_results)
            
            # Generate alerts
            monitoring_results['alerts'] = self._generate_alerts(monitoring_results)
            
        except Exception as e:
            logging.error(f"Error in comprehensive monitoring: {str(e)}")
            monitoring_results['error'] = str(e)
            
        return monitoring_results
    
    def _calculate_overall_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall model health score"""
        health_components = []
        
        try:
            # Performance health
            perf_analysis = results.get('performance_analysis', {})
            if 'drift_detected' in perf_analysis and not perf_analysis['drift_detected']:
                health_components.append(0.9)
            else:
                health_components.append(0.3)
            
            # Bias health
            bias_analysis = results.get('bias_analysis', {})
            bias_scores = []
            for attr_metrics in bias_analysis.values():
                if isinstance(attr_metrics, dict):
                    accuracy_diff = attr_metrics.get('accuracy_difference', 1.0)
                    bias_scores.append(max(0, 1 - accuracy_diff))
            
            if bias_scores:
                health_components.append(np.mean(bias_scores))
            
            # Clinical utility health
            clinical_validation = results.get('clinical_validation', {})
            utility_score = clinical_validation.get('clinical_utility_score', 0)
            health_components.append(utility_score)
            
            return float(np.mean(health_components)) if health_components else 0.0
            
        except Exception:
            return 0.0
    
    def _generate_alerts(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on monitoring results"""
        alerts = []
        
        try:
            health_score = results.get('overall_health_score', 0)
            
            if health_score < 0.5:
                alerts.append({
                    'severity': 'critical',
                    'message': 'Model health score below critical threshold',
                    'recommendation': 'Immediate model review and potential retraining required'
                })
            elif health_score < 0.7:
                alerts.append({
                    'severity': 'warning',
                    'message': 'Model health score below optimal threshold',
                    'recommendation': 'Monitor closely and consider model updates'
                })
            
            # Check for specific issues
            drift_detection = results.get('drift_detection', {})
            if drift_detection.get('drift_detected', False):
                alerts.append({
                    'severity': 'warning',
                    'message': f"Data drift detected - severity: {drift_detection.get('severity', 'unknown')}",
                    'recommendation': 'Investigate data source changes and consider model adaptation'
                })
            
            bias_analysis = results.get('bias_analysis', {})
            for attr, metrics in bias_analysis.items():
                if isinstance(metrics, dict):
                    accuracy_diff = metrics.get('accuracy_difference', 0)
                    if accuracy_diff > 0.1:
                        alerts.append({
                            'severity': 'warning',
                            'message': f'Bias detected in {attr} - accuracy difference: {accuracy_diff:.3f}',
                            'recommendation': 'Review model training data and fairness constraints'
                        })
            
        except Exception as e:
            logging.error(f"Error generating alerts: {str(e)}")
            
        return alerts

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize monitoring orchestrator
    orchestrator = ModelMonitoringOrchestrator(
        model_name="medical_diagnosis_model",
        protected_attributes=["age_group", "gender", "ethnicity"],
        validation_horizon_days=30
    )
    
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Sample baseline and current data
    baseline_data = np.random.normal(0, 1, n_samples)
    current_data = np.random.normal(0.2, 1.2, n_samples)  # Slightly different distribution
    
    # Sample predictions and labels
    predictions = np.random.choice([0, 1], size=n_samples)
    labels = np.random.choice([0, 1], size=n_samples)
    
    # Sample protected attributes
    protected_attributes = {
        'age_group': np.random.choice(['young', 'middle', 'elderly'], size=n_samples),
        'gender': np.random.choice(['male', 'female'], size=n_samples),
        'ethnicity': np.random.choice(['group_a', 'group_b', 'group_c'], size=n_samples)
    }
    
    # Sample timestamps
    timestamps = [datetime.now() - timedelta(hours=i) for i in range(n_samples)]
    
    # Run comprehensive monitoring
    results = orchestrator.run_comprehensive_monitoring(
        baseline_data=baseline_data,
        current_data=current_data,
        predictions=predictions,
        labels=labels,
        protected_attrs=protected_attributes,
        timestamps=timestamps
    )
    
    # Print results
    print("=== Model Monitoring Results ===")
    print(f"Overall Health Score: {results['overall_health_score']:.3f}")
    
    if results['drift_detection']:
        print(f"Data Drift Detected: {results['drift_detection']['drift_detected']}")
    
    if results['bias_analysis']:
        print("Bias Analysis Results:")
        for attr, metrics in results['bias_analysis'].items():
            if isinstance(metrics, dict):
                print(f"  {attr}: Accuracy Difference = {metrics.get('accuracy_difference', 0):.3f}")
    
    print(f"Number of Alerts: {len(results['alerts'])}")
    for alert in results['alerts']:
        print(f"  [{alert['severity'].upper()}] {alert['message']}")