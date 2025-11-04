"""
Model Drift Detection and Accuracy Monitoring

Provides comprehensive model accuracy monitoring, drift detection, and quality metrics
for medical AI systems with clinical validation capabilities.
"""

import asyncio
import json
import numpy as np
import time
import torch
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import structlog

from ...config.logging_config import get_logger

logger = structlog.get_logger("model_monitoring")


@dataclass
class AccuracyMetrics:
    """Model accuracy metrics container."""
    model_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Basic accuracy metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    
    # Medical-specific metrics
    clinical_accuracy: float = 0.0
    medical_relevance: float = 0.0
    diagnostic_sensitivity: float = 0.0  # True positive rate
    diagnostic_specificity: float = 0.0  # True negative rate
    
    # Safety metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    harmful_prediction_rate: float = 0.0
    
    # Confidence metrics
    avg_confidence: float = 0.0
    confidence_calibration_error: float = 0.0
    
    # Sample information
    total_samples: int = 0
    positive_samples: int = 0
    negative_samples: int = 0
    
    # Validation context
    validation_dataset: str = ""
    medical_specialty: str = ""
    clinical_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftMetrics:
    """Model drift detection metrics container."""
    model_id: str
    timestamp: float = field(default_factory=time.time)
    
    # Data drift metrics
    data_drift_score: float = 0.0
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    distribution_shift_detected: bool = False
    drift_magnitude: float = 0.0
    
    # Concept drift metrics
    concept_drift_score: float = 0.0
    decision_boundary_shift: float = 0.0
    prediction_pattern_change: float = 0.0
    
    # Performance drift metrics
    performance_drift_score: float = 0.0
    accuracy_degradation: float = 0.0
    latency_increase: float = 0.0
    error_rate_increase: float = 0.0
    
    # Clinical drift metrics
    clinical_relevance_drift: float = 0.0
    medical_safety_drift: float = 0.0
    bias_drift_scores: Dict[str, float] = field(default_factory=dict)
    
    # Drift detection details
    drift_direction: str = "stable"  # increasing, decreasing, stable
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    statistical_significance: float = 0.0
    
    # Alert information
    drift_alert_level: str = "normal"  # normal, warning, critical
    recommended_actions: List[str] = field(default_factory=list)


class StatisticalDriftDetector:
    """Statistical drift detection using various statistical tests."""
    
    def __init__(self, 
                 significance_level: float = 0.05,
                 min_samples: int = 100):
        self.significance_level = significance_level
        self.min_samples = min_samples
        
        # Historical data storage
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_stats: Dict[str, Dict[str, float]] = {}
        
        # Drift detection methods
        self.detection_methods = {
            'ks_test': self._kolmogorov_smirnov_test,
            'psi': self._population_stability_index,
            'jensen_shannon': self._jensen_shannon_divergence,
            'isolation_forest': self._isolation_forest_detection,
            'pca_reconstruction': self._pca_reconstruction_error
        }
    
    def set_reference_data(self, feature_name: str, reference_data: np.ndarray):
        """Set reference data for a feature."""
        self.reference_data[feature_name] = reference_data
        
        # Calculate reference statistics
        self.reference_stats[feature_name] = {
            'mean': np.mean(reference_data),
            'std': np.std(reference_data),
            'min': np.min(reference_data),
            'max': np.max(reference_data),
            'median': np.median(reference_data),
            'q25': np.percentile(reference_data, 25),
            'q75': np.percentile(reference_data, 75)
        }
    
    def detect_drift(self, 
                    current_data: Dict[str, np.ndarray],
                    methods: List[str] = None) -> Dict[str, float]:
        """Detect drift in current data compared to reference."""
        if methods is None:
            methods = ['ks_test', 'psi', 'jensen_shannon']
        
        drift_scores = {}
        
        for feature_name, current_values in current_data.items():
            if feature_name not in self.reference_data:
                continue
            
            reference_values = self.reference_data[feature_name]
            
            if len(current_values) < self.min_samples or len(reference_values) < self.min_samples:
                continue
            
            feature_drift_scores = []
            
            for method in methods:
                if method in self.detection_methods:
                    try:
                        score = self.detection_methods[method](feature_name, current_values, reference_values)
                        feature_drift_scores.append(score)
                    except Exception as e:
                        logger.warning(f"Drift detection method {method} failed", 
                                     feature=feature_name, error=str(e))
            
            # Combine drift scores
            if feature_drift_scores:
                drift_scores[feature_name] = np.mean(feature_drift_scores)
        
        return drift_scores
    
    def _kolmogorov_smirnov_test(self, feature_name: str, 
                                current: np.ndarray, reference: np.ndarray) -> float:
        """KS test for distribution comparison."""
        try:
            statistic, p_value = stats.ks_2samp(reference, current)
            # Convert to drift score (1 - p_value, higher means more drift)
            drift_score = 1 - p_value
            return min(drift_score, 1.0)
        except:
            return 0.0
    
    def _population_stability_index(self, feature_name: str,
                                   current: np.ndarray, reference: np.ndarray) -> float:
        """Population Stability Index (PSI) calculation."""
        try:
            # Create bins based on reference data
            bins = np.quantile(reference, np.linspace(0, 1, 21))  # 20 bins
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            ref_hist, _ = np.histogram(reference, bins=bins)
            curr_hist, _ = np.histogram(current, bins=bins)
            
            # Normalize to percentages
            ref_pct = ref_hist / len(reference)
            curr_pct = curr_hist / len(current)
            
            # Avoid log(0)
            ref_pct = np.where(ref_pct == 0, 0.001, ref_pct)
            curr_pct = np.where(curr_pct == 0, 0.001, curr_pct)
            
            # Calculate PSI
            psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
            
            return min(psi, 1.0)  # Cap at 1.0
        except:
            return 0.0
    
    def _jensen_shannon_divergence(self, feature_name: str,
                                  current: np.ndarray, reference: np.ndarray) -> float:
        """Jensen-Shannon divergence."""
        try:
            # Create histograms
            bins = 50
            ref_hist, _ = np.histogram(reference, bins=bins, density=True)
            curr_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Normalize
            ref_hist = ref_hist / np.sum(ref_hist)
            curr_hist = curr_hist / np.sum(curr_hist)
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_hist += epsilon
            curr_hist += epsilon
            
            # Calculate JSD
            m = 0.5 * (ref_hist + curr_hist)
            jsd = 0.5 * stats.entropy(ref_hist, m) + 0.5 * stats.entropy(curr_hist, m)
            
            return min(jsd, 1.0)  # Cap at 1.0
        except:
            return 0.0
    
    def _isolation_forest_detection(self, feature_name: str,
                                   current: np.ndarray, reference: np.ndarray) -> float:
        """Isolation Forest for anomaly detection."""
        try:
            # Combine reference and current data
            combined_data = np.concatenate([reference, current])
            labels = np.concatenate([np.zeros(len(reference)), np.ones(len(current))])
            
            # Train isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(combined_data.reshape(-1, 1))
            
            # Score current data
            anomaly_scores = iso_forest.decision_function(current.reshape(-1, 1))
            
            # Convert to drift score
            drift_score = 1 - np.mean(anomaly_scores)  # Higher anomaly score = more drift
            return max(0.0, min(drift_score, 1.0))
        except:
            return 0.0
    
    def _pca_reconstruction_error(self, feature_name: str,
                                 current: np.ndarray, reference: np.ndarray) -> float:
        """PCA reconstruction error for drift detection."""
        try:
            if len(current) < 10 or len(reference) < 10:
                return 0.0
            
            # Prepare data
            reference_reshaped = reference.reshape(-1, 1)
            current_reshaped = current.reshape(-1, 1)
            
            # Standardize data
            scaler = StandardScaler()
            combined_data = scaler.fit_transform(np.concatenate([reference_reshaped, current_reshaped]))
            
            ref_data = combined_data[:len(reference)]
            curr_data = combined_data[len(reference):]
            
            # Fit PCA
            pca = PCA(n_components=1)
            pca.fit(ref_data)
            
            # Calculate reconstruction error for current data
            reconstructed = pca.inverse_transform(pca.transform(curr_data))
            reconstruction_error = np.mean((curr_data - reconstructed) ** 2)
            
            # Normalize by variance
            normalized_error = reconstruction_error / np.var(ref_data)
            
            return min(normalized_error, 1.0)
        except:
            return 0.0


class ConceptDriftDetector:
    """Detects concept drift in model decision boundaries."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.prediction_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
        
    def add_prediction(self, 
                      predictions: np.ndarray, 
                      confidences: np.ndarray,
                      metadata: Dict[str, Any] = None):
        """Add prediction results for drift detection."""
        if len(predictions) != len(confidences):
            raise ValueError("Predictions and confidences must have same length")
        
        for pred, conf in zip(predictions, confidences):
            self.prediction_history.append({
                'prediction': pred,
                'confidence': conf,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
    
    def detect_concept_drift(self) -> Dict[str, float]:
        """Detect concept drift in predictions."""
        if len(self.prediction_history) < 100:
            return {'concept_drift_score': 0.0, 'drift_magnitude': 0.0}
        
        # Split history into reference and current windows
        reference_size = len(self.prediction_history) // 2
        reference_predictions = list(self.prediction_history)[:reference_size]
        current_predictions = list(self.prediction_history)[reference_size:]
        
        # Extract predictions and confidences
        ref_preds = np.array([p['prediction'] for p in reference_predictions])
        curr_preds = np.array([p['prediction'] for p in current_predictions])
        
        ref_confs = np.array([p['confidence'] for p in reference_predictions])
        curr_confs = np.array([p['confidence'] for p in current_predictions])
        
        # Calculate drift metrics
        prediction_drift = self._calculate_prediction_drift(ref_preds, curr_preds)
        confidence_drift = self._calculate_confidence_drift(ref_confs, curr_confs)
        
        # Combine drift scores
        concept_drift_score = 0.6 * prediction_drift + 0.4 * confidence_drift
        drift_magnitude = np.sqrt(prediction_drift**2 + confidence_drift**2)
        
        return {
            'concept_drift_score': concept_drift_score,
            'prediction_drift': prediction_drift,
            'confidence_drift': confidence_drift,
            'drift_magnitude': drift_magnitude
        }
    
    def _calculate_prediction_drift(self, ref_preds: np.ndarray, curr_preds: np.ndarray) -> float:
        """Calculate prediction pattern drift."""
        try:
            # Calculate prediction proportions
            ref_unique, ref_counts = np.unique(ref_preds, return_counts=True)
            curr_unique, curr_counts = np.unique(curr_preds, return_counts=True)
            
            ref_props = ref_counts / len(ref_preds)
            curr_props = curr_counts / len(curr_preds)
            
            # Calculate JS divergence between prediction distributions
            all_classes = sorted(set(ref_unique) | set(curr_unique))
            
            ref_dist = np.zeros(len(all_classes))
            curr_dist = np.zeros(len(all_classes))
            
            for i, class_label in enumerate(all_classes):
                if class_label in ref_unique:
                    idx = np.where(ref_unique == class_label)[0][0]
                    ref_dist[i] = ref_props[idx]
                
                if class_label in curr_unique:
                    idx = np.where(curr_unique == class_label)[0][0]
                    curr_dist[i] = curr_props[idx]
            
            # Add epsilon to avoid log(0)
            epsilon = 1e-10
            ref_dist += epsilon
            curr_dist += epsilon
            
            # Calculate JS divergence
            m = 0.5 * (ref_dist + curr_dist)
            js_div = 0.5 * stats.entropy(ref_dist, m) + 0.5 * stats.entropy(curr_dist, m)
            
            return min(js_div, 1.0)
        except:
            return 0.0
    
    def _calculate_confidence_drift(self, ref_confs: np.ndarray, curr_confs: np.ndarray) -> float:
        """Calculate confidence distribution drift."""
        try:
            # Use KS test for confidence distributions
            statistic, p_value = stats.ks_2samp(ref_confs, curr_confs)
            
            # Convert to drift score
            drift_score = 1 - p_value
            return min(drift_score, 1.0)
        except:
            return 0.0


class AccuracyMonitor:
    """Comprehensive accuracy monitoring with medical AI specific metrics."""
    
    def __init__(self, 
                 evaluation_interval: int = 3600,  # 1 hour
                 reference_window: int = 10000,
                 min_samples_for_evaluation: int = 100):
        
        self.evaluation_interval = evaluation_interval
        self.reference_window = reference_window
        self.min_samples_for_evaluation = min_samples_for_evaluation
        
        self.logger = structlog.get_logger("accuracy_monitor")
        
        # Data storage
        self.predictions_history: deque = deque(maxlen=reference_window)
        self.confidences_history: deque = deque(maxlen=reference_window)
        self.true_labels_history: deque = deque(maxlen=reference_window)
        self.metadata_history: deque = deque(maxlen=reference_window)
        
        # Current evaluation results
        self.current_accuracy_metrics: Optional[AccuracyMetrics] = None
        
        # Calibration data
        self.calibration_data: Dict[float, List[float]] = defaultdict(list)  # confidence -> accuracy pairs
        
        self.logger.info("AccuracyMonitor initialized")
    
    def add_evaluation_data(self,
                           predictions: np.ndarray,
                           true_labels: np.ndarray,
                           confidences: np.ndarray,
                           metadata: Dict[str, Any] = None):
        """Add evaluation data for accuracy monitoring."""
        if len(predictions) != len(true_labels) != len(confidences):
            raise ValueError("All arrays must have the same length")
        
        for pred, true_label, conf in zip(predictions, true_labels, confidences):
            self.predictions_history.append(pred)
            self.true_labels_history.append(true_label)
            self.confidences_history.append(conf)
            self.metadata_history.append(metadata or {})
    
    def evaluate_accuracy(self, 
                         medical_context: Dict[str, Any] = None) -> AccuracyMetrics:
        """Evaluate current model accuracy."""
        if len(self.predictions_history) < self.min_samples_for_evaluation:
            raise ValueError(f"Insufficient data: need {self.min_samples_for_evaluation}, have {len(self.predictions_history)}")
        
        # Convert to numpy arrays
        predictions = np.array(list(self.predictions_history))
        true_labels = np.array(list(self.true_labels_history))
        confidences = np.array(list(self.confidences_history))
        
        # Handle different prediction types
        is_binary = len(np.unique(predictions)) == 2
        is_probabilistic = np.all((predictions >= 0) & (predictions <= 1))
        
        metrics = AccuracyMetrics(
            model_id="current_model",
            total_samples=len(predictions),
            positive_samples=int(np.sum(true_labels == 1)) if is_binary else 0,
            negative_samples=int(np.sum(true_labels == 0)) if is_binary else 0,
            medical_specialty=medical_context.get('specialty', 'general') if medical_context else 'general',
            clinical_context=medical_context or {},
            validation_dataset=medical_context.get('dataset', 'inference') if medical_context else 'inference'
        )
        
        try:
            # Basic accuracy metrics
            if is_binary:
                # Binary classification metrics
                predictions_binary = (predictions > 0.5).astype(int)
                
                metrics.accuracy = accuracy_score(true_labels, predictions_binary)
                metrics.precision = precision_score(true_labels, predictions_binary, zero_division=0)
                metrics.recall = recall_score(true_labels, predictions_binary, zero_division=0)
                metrics.f1_score = f1_score(true_labels, predictions_binary, zero_division=0)
                
                # Sensitivity and specificity
                tn = np.sum((predictions_binary == 0) & (true_labels == 0))
                fp = np.sum((predictions_binary == 1) & (true_labels == 0))
                fn = np.sum((predictions_binary == 0) & (true_labels == 1))
                tp = np.sum((predictions_binary == 1) & (true_labels == 1))
                
                metrics.diagnostic_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                metrics.diagnostic_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                # False positive/negative rates
                metrics.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                metrics.false_negative_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
                
                # AUC score if probabilistic predictions
                if is_probabilistic:
                    try:
                        metrics.auc_score = roc_auc_score(true_labels, predictions)
                    except:
                        metrics.auc_score = 0.0
            
            else:
                # Multi-class or regression metrics
                if is_probabilistic:
                    # For multi-class probabilistic predictions
                    try:
                        if predictions.shape[1] > 1:  # Multi-class
                            pred_classes = np.argmax(predictions, axis=1)
                            metrics.accuracy = accuracy_score(true_labels, pred_classes)
                            metrics.precision = precision_score(true_labels, pred_classes, average='macro', zero_division=0)
                            metrics.recall = recall_score(true_labels, pred_classes, average='macro', zero_division=0)
                            metrics.f1_score = f1_score(true_labels, pred_classes, average='macro', zero_division=0)
                        else:
                            # Regression or single probability
                            if np.all((true_labels >= 0) & (true_labels <= 1)):
                                # Binary classification with single probability
                                predictions_binary = (predictions > 0.5).astype(int)
                                metrics.accuracy = accuracy_score(true_labels, predictions_binary)
                            else:
                                # Regression
                                mae = np.mean(np.abs(predictions - true_labels))
                                mse = np.mean((predictions - true_labels) ** 2)
                                metrics.accuracy = 1.0 / (1.0 + mae)  # Convert MAE to accuracy-like score
                    except:
                        metrics.accuracy = 0.0
                else:
                    # Classification with class labels
                    try:
                        metrics.accuracy = accuracy_score(true_labels, predictions)
                        metrics.precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
                        metrics.recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
                        metrics.f1_score = f1_score(true_labels, predictions, average='macro', zero_division=0)
                    except:
                        metrics.accuracy = 0.0
            
            # Medical-specific metrics
            if medical_context:
                metrics.clinical_accuracy = self._calculate_clinical_accuracy(
                    predictions, true_labels, medical_context
                )
                metrics.medical_relevance = self._calculate_medical_relevance(
                    predictions, true_labels, medical_context
                )
            
            # Confidence metrics
            metrics.avg_confidence = float(np.mean(confidences))
            metrics.confidence_calibration_error = self._calculate_calibration_error(
                predictions, true_labels, confidences
            )
            
            # Safety metrics
            metrics.harmful_prediction_rate = self._calculate_harmful_prediction_rate(
                predictions, true_labels, medical_context
            )
            
            self.current_accuracy_metrics = metrics
            
            self.logger.info("Accuracy evaluation completed",
                           accuracy=metrics.accuracy,
                           f1_score=metrics.f1_score,
                           total_samples=metrics.total_samples)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Accuracy evaluation failed", error=str(e))
            raise
    
    def _calculate_clinical_accuracy(self,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   medical_context: Dict[str, Any]) -> float:
        """Calculate clinically-weighted accuracy."""
        try:
            specialty = medical_context.get('specialty', 'general')
            
            # Get clinical importance weights
            clinical_weights = self._get_clinical_weights(specialty)
            
            # Calculate weighted accuracy
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
            
            correct = (pred_classes == true_labels).astype(float)
            
            # Apply clinical weights based on importance
            weighted_accuracy = np.average(correct, weights=clinical_weights)
            
            return weighted_accuracy
        except:
            return 0.0
    
    def _calculate_medical_relevance(self,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   medical_context: Dict[str, Any]) -> float:
        """Calculate medical relevance score."""
        try:
            specialty = medical_context.get('specialty', 'general')
            condition_severity = medical_context.get('condition_severity', 'medium')
            
            # Base relevance (all predictions should be relevant for medical AI)
            base_relevance = 1.0
            
            # Adjust based on condition severity
            severity_multiplier = {
                'critical': 1.2,
                'high': 1.1,
                'medium': 1.0,
                'low': 0.9
            }.get(condition_severity, 1.0)
            
            # Specialty-specific adjustments
            specialty_multiplier = {
                'cardiology': 1.1,  # High stakes
                'oncology': 1.1,    # Critical for treatment
                'radiology': 1.05,  # Important for diagnosis
                'emergency': 1.15,  # Critical care
                'general': 1.0
            }.get(specialty, 1.0)
            
            relevance_score = base_relevance * severity_multiplier * specialty_multiplier
            return min(relevance_score, 1.0)
            
        except:
            return 0.0
    
    def _calculate_calibration_error(self,
                                   predictions: np.ndarray,
                                   true_labels: np.ndarray,
                                   confidences: np.ndarray) -> float:
        """Calculate expected calibration error."""
        try:
            # Bin predictions by confidence
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            total_samples = len(predictions)
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this confidence bin
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    # Accuracy in this bin
                    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                        pred_in_bin = np.argmax(predictions[in_bin], axis=1)
                    else:
                        pred_in_bin = (predictions[in_bin] > 0.5).astype(int)
                    
                    accuracy_in_bin = (pred_in_bin == true_labels[in_bin]).mean()
                    
                    # Average confidence in this bin
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    
                    # Add to ECE
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
            
        except:
            return 0.0
    
    def _calculate_harmful_prediction_rate(self,
                                         predictions: np.ndarray,
                                         true_labels: np.ndarray,
                                         medical_context: Dict[str, Any]) -> float:
        """Calculate rate of potentially harmful predictions."""
        try:
            specialty = medical_context.get('specialty', 'general')
            
            # Define harmful prediction patterns by specialty
            harmful_patterns = self._get_harmful_patterns(specialty)
            
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
            
            harmful_count = 0
            
            for i, (pred, true_label) in enumerate(zip(pred_classes, true_labels)):
                # Check if prediction matches harmful patterns
                if (pred, true_label) in harmful_patterns:
                    harmful_count += 1
            
            return harmful_count / len(predictions) if len(predictions) > 0 else 0.0
            
        except:
            return 0.0
    
    def _get_clinical_weights(self, specialty: str) -> np.ndarray:
        """Get clinical importance weights for predictions."""
        # This is a simplified version - in practice, this would be more sophisticated
        # and based on actual medical importance of correct vs incorrect predictions
        
        # For now, return uniform weights
        return np.ones(1000)  # Placeholder
    
    def _get_harmful_patterns(self, specialty: str) -> List[Tuple[int, int]]:
        """Get patterns considered harmful for medical AI."""
        # Examples of harmful patterns (prediction, true_label):
        harmful_patterns = []
        
        if specialty in ['cardiology', 'emergency']:
            # Harmful to miss critical conditions
            harmful_patterns.extend([
                (0, 1),  # False negative for critical conditions
                (1, 0)   # False positive leading to unnecessary intervention
            ])
        
        return harmful_patterns


class ModelDriftDetector:
    """Comprehensive model drift detection combining multiple approaches."""
    
    def __init__(self, 
                 model_id: str,
                 reference_data: Optional[Dict[str, np.ndarray]] = None,
                 reference_labels: Optional[np.ndarray] = None):
        
        self.model_id = model_id
        self.logger = structlog.get_logger("drift_detector")
        
        # Initialize detectors
        self.statistical_detector = StatisticalDriftDetector()
        self.concept_detector = ConceptDriftDetector()
        self.accuracy_monitor = AccuracyMonitor()
        
        # Set reference data if provided
        if reference_data:
            for feature_name, data in reference_data.items():
                self.statistical_detector.set_reference_data(feature_name, data)
        
        # Historical drift metrics
        self.drift_history: deque = deque(maxlen=1000)
        
        self.logger.info("ModelDriftDetector initialized", model_id=model_id)
    
    def add_data_point(self,
                      features: Dict[str, np.ndarray],
                      predictions: np.ndarray,
                      confidences: np.ndarray,
                      metadata: Dict[str, Any] = None):
        """Add new data point for drift detection."""
        # Add to statistical detector
        current_data = {name: values for name, values in features.items()}
        
        # Add to concept detector
        self.concept_detector.add_prediction(predictions, confidences, metadata)
        
        # If we have true labels, add to accuracy monitor
        if metadata and 'true_labels' in metadata:
            self.accuracy_monitor.add_evaluation_data(
                predictions,
                metadata['true_labels'],
                confidences,
                metadata.get('medical_context')
            )
    
    def detect_drift(self, 
                    current_features: Dict[str, np.ndarray],
                    threshold: float = 0.1) -> DriftMetrics:
        """Detect all types of drift."""
        
        drift_metrics = DriftMetrics(model_id=self.model_id)
        
        try:
            # Statistical drift detection
            feature_drift_scores = self.statistical_detector.detect_drift(current_features)
            
            if feature_drift_scores:
                drift_metrics.feature_drift_scores = feature_drift_scores
                drift_metrics.data_drift_score = np.mean(list(feature_drift_scores.values()))
                drift_metrics.drift_magnitude = np.max(list(feature_drift_scores.values()))
            
            # Concept drift detection
            concept_drift_results = self.concept_detector.detect_concept_drift()
            drift_metrics.concept_drift_score = concept_drift_results.get('concept_drift_score', 0.0)
            drift_metrics.decision_boundary_shift = concept_drift_results.get('prediction_drift', 0.0)
            
            # Performance drift (if we have accuracy data)
            if self.accuracy_monitor.current_accuracy_metrics:
                accuracy_metrics = self.accuracy_monitor.current_accuracy_metrics
                
                # Calculate performance drift compared to reference
                ref_accuracy = 0.95  # This should come from model validation
                drift_metrics.performance_drift_score = max(0, ref_accuracy - accuracy_metrics.accuracy)
                drift_metrics.accuracy_degradation = drift_metrics.performance_drift_score
            
            # Clinical drift metrics
            if self.accuracy_monitor.current_accuracy_metrics:
                accuracy_metrics = self.accuracy_monitor.current_accuracy_metrics
                
                # Clinical relevance drift
                ref_clinical_relevance = 0.9  # Reference clinical relevance
                drift_metrics.clinical_relevance_drift = max(0, ref_clinical_relevance - accuracy_metrics.medical_relevance)
                
                # Medical safety drift
                ref_safety_score = 0.95  # Reference safety score
                drift_metrics.medical_safety_drift = max(0, ref_safety_score - (1.0 - accuracy_metrics.harmful_prediction_rate))
            
            # Determine drift direction
            total_drift = (drift_metrics.data_drift_score + 
                          drift_metrics.concept_drift_score + 
                          drift_metrics.performance_drift_score) / 3
            
            if total_drift > threshold * 1.5:
                drift_metrics.drift_direction = "increasing"
            elif total_drift > threshold:
                drift_metrics.drift_direction = "moderate"
            else:
                drift_metrics.drift_direction = "stable"
            
            # Set alert level
            if total_drift > threshold * 2:
                drift_metrics.drift_alert_level = "critical"
                drift_metrics.recommended_actions = [
                    "Immediate model retraining required",
                    "Investigate data pipeline for issues",
                    "Review recent deployments or data changes"
                ]
            elif total_drift > threshold * 1.5:
                drift_metrics.drift_alert_level = "warning"
                drift_metrics.recommended_actions = [
                    "Schedule model evaluation",
                    "Monitor drift trends closely",
                    "Consider retraining if trend continues"
                ]
            else:
                drift_metrics.drift_alert_level = "normal"
                drift_metrics.recommended_actions = ["Continue monitoring"]
            
            # Store in history
            self.drift_history.append(drift_metrics)
            
            self.logger.info("Drift detection completed",
                           model_id=self.model_id,
                           data_drift_score=drift_metrics.data_drift_score,
                           concept_drift_score=drift_metrics.concept_drift_score,
                           alert_level=drift_metrics.drift_alert_level)
            
            return drift_metrics
            
        except Exception as e:
            self.logger.error("Drift detection failed", error=str(e))
            drift_metrics.drift_alert_level = "error"
            drift_metrics.recommended_actions = [f"Drift detection error: {str(e)}"]
            return drift_metrics
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        if not self.drift_history:
            return {"status": "no_data"}
        
        recent_drift = list(self.drift_history)[-10:]  # Last 10 measurements
        
        return {
            "model_id": self.model_id,
            "total_measurements": len(self.drift_history),
            "recent_measurements": len(recent_drift),
            "latest_drift_metrics": asdict(self.drift_history[-1]) if self.drift_history else {},
            "drift_trends": {
                "data_drift_avg": np.mean([d.data_drift_score for d in recent_drift]),
                "concept_drift_avg": np.mean([d.concept_drift_score for d in recent_drift]),
                "performance_drift_avg": np.mean([d.performance_drift_score for d in recent_drift])
            },
            "alert_summary": {
                "normal": len([d for d in recent_drift if d.drift_alert_level == "normal"]),
                "warning": len([d for d in recent_drift if d.drift_alert_level == "warning"]),
                "critical": len([d for d in recent_drift if d.drift_alert_level == "critical"])
            }
        }