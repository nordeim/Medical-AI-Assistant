"""
Model Monitoring System
Real-time monitoring of model accuracy, drift detection, and clinical outcome tracking.
"""

import os
import sys
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import aioredis
from collections import deque, defaultdict
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class ModelDrift:
    """Model drift detection result"""
    model_name: str
    drift_type: str  # data_drift, concept_drift, performance_drift
    drift_score: float
    threshold: float
    confidence_level: float
    timestamp: datetime
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    alert_level: str  # low, medium, high, critical

@dataclass
class ClinicalValidation:
    """Clinical validation result"""
    patient_id: str
    model_prediction: Dict[str, Any]
    actual_outcome: Dict[str, Any]
    prediction_time: datetime
    validation_time: datetime
    accuracy_score: float
    clinical_significance: str  # excellent, good, fair, poor
    validation_status: str = "pending"

class ModelMonitoringSystem:
    """Production model monitoring with drift detection and clinical validation"""
    
    def __init__(self, config_path: str = "config/monitoring_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Monitoring storage
        self.prediction_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.actual_outcomes: Dict[str, List[ClinicalValidation]] = defaultdict(list)
        self.drift_alerts: List[ModelDrift] = []
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # Drift detection parameters
        self.drift_thresholds = self.config.get("drift_thresholds", {})
        self.baseline_window_size = self.config.get("baseline_window_size", 1000)
        self.drift_window_size = self.config.get("drift_window_size", 100)
        
        # Clinical validation
        self.clinical_validation_rules = self.config.get("clinical_validation_rules", {})
        self.required_clinical_metrics = self.config.get("required_clinical_metrics", [])
        
        # Performance monitoring integration
        self.performance_monitor = PerformanceMonitor()
        
        # Redis for distributed monitoring
        self.redis_client = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "drift_thresholds": {
                "data_drift": 0.1,
                "concept_drift": 0.15,
                "performance_drift": 0.2
            },
            "baseline_window_size": 1000,
            "drift_window_size": 100,
            "clinical_validation_rules": {
                "accuracy_threshold": 0.85,
                "precision_threshold": 0.80,
                "recall_threshold": 0.75,
                "auc_threshold": 0.80
            },
            "required_clinical_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "monitoring_interval": 300,  # 5 minutes
            "alert_thresholds": {
                "accuracy_degradation": 0.05,
                "precision_degradation": 0.05,
                "latency_increase": 0.5,  # 50% increase
                "error_rate_increase": 0.02
            }
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Monitoring config {config_path} not found, using defaults")
            return default_config
    
    async def initialize(self):
        """Initialize the monitoring system"""
        logger.info("Initializing Model Monitoring System...")
        
        try:
            # Initialize performance monitor
            await self.performance_monitor.initialize()
            
            # Initialize Redis connection
            redis_host = self.config.get("redis_host", "localhost")
            redis_port = self.config.get("redis_port", 6379)
            self.redis_client = await aioredis.from_url(f"redis://{redis_host}:{redis_port}")
            
            # Load historical baselines
            await self._load_performance_baselines()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitoring_loop())
            asyncio.create_task(self._drift_detection_loop())
            asyncio.create_task(self._clinical_validation_loop())
            asyncio.create_task(self._alert_processing_loop())
            
            logger.info("Model Monitoring System initialized")
            
        except Exception as e:
            logger.error(f"Monitoring system initialization failed: {str(e)}")
            raise
    
    async def _load_performance_baselines(self):
        """Load historical performance baselines"""
        # In production, load from persistent storage
        logger.info("Performance baselines loaded")
    
    async def log_prediction(self, model_name: str, prediction_data: Dict[str, Any]):
        """Log model prediction for monitoring"""
        try:
            timestamp = datetime.utcnow()
            
            # Enrich prediction data
            enriched_data = {
                "timestamp": timestamp.isoformat(),
                "model_name": model_name,
                "prediction": prediction_data.get("prediction"),
                "confidence": prediction_data.get("confidence", 0.0),
                "processing_time": prediction_data.get("processing_time", 0.0),
                "patient_id": prediction_data.get("patient_id"),
                "input_features": prediction_data.get("input_features"),
                "success": prediction_data.get("success", True),
                "error": prediction_data.get("error")
            }
            
            # Store in history
            self.prediction_history[model_name].append(enriched_data)
            
            # Store in Redis for distributed monitoring
            if self.redis_client:
                await self.redis_client.lpush(
                    "medical_ai:predictions",
                    json.dumps(enriched_data)
                )
                # Keep only recent entries
                await self.redis_client.ltrim("medical_ai:predictions", 0, 9999)
            
            logger.debug(f"Prediction logged for model {model_name}")
            
        except Exception as e:
            logger.error(f"Prediction logging failed: {str(e)}")
    
    async def record_actual_outcome(self, model_name: str, patient_id: str, 
                                  actual_outcome: Dict[str, Any]):
        """Record actual clinical outcome"""
        try:
            # Find corresponding prediction
            prediction_data = None
            for pred in reversed(self.prediction_history[model_name]):
                if pred.get("patient_id") == patient_id:
                    prediction_data = pred
                    break
            
            if not prediction_data:
                logger.warning(f"No prediction found for patient {patient_id}")
                return
            
            # Create validation record
            validation = ClinicalValidation(
                patient_id=patient_id,
                model_prediction=prediction_data,
                actual_outcome=actual_outcome,
                prediction_time=datetime.fromisoformat(prediction_data["timestamp"]),
                validation_time=datetime.utcnow(),
                accuracy_score=0.0,  # Will calculate below
                clinical_significance="pending"
            )
            
            # Calculate accuracy metrics
            validation.accuracy_score = self._calculate_accuracy_score(
                prediction_data, actual_outcome
            )
            
            # Determine clinical significance
            validation.clinical_significance = self._determine_clinical_significance(
                validation.accuracy_score
            )
            
            # Store validation
            self.actual_outcomes[model_name].append(validation)
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.lpush(
                    f"medical_ai:outcomes:{model_name}",
                    json.dumps(asdict(validation), default=str)
                )
            
            logger.info(f"Clinical outcome recorded for patient {patient_id}, model {model_name}")
            
        except Exception as e:
            logger.error(f"Actual outcome recording failed: {str(e)}")
    
    def _calculate_accuracy_score(self, prediction_data: Dict[str, Any], 
                                actual_outcome: Dict[str, Any]) -> float:
        """Calculate accuracy score comparing prediction to actual outcome"""
        try:
            # Simple accuracy calculation - in production would be more sophisticated
            predicted_value = prediction_data.get("prediction")
            actual_value = actual_outcome.get("actual_result")
            
            if predicted_value == actual_value:
                return 1.0
            else:
                # Calculate confidence-adjusted accuracy
                confidence = prediction_data.get("confidence", 0.5)
                return confidence * 0.8  # Penalize based on confidence
            
        except Exception as e:
            logger.warning(f"Accuracy calculation failed: {str(e)}")
            return 0.0
    
    def _determine_clinical_significance(self, accuracy_score: float) -> str:
        """Determine clinical significance level"""
        if accuracy_score >= 0.9:
            return "excellent"
        elif accuracy_score >= 0.8:
            return "good"
        elif accuracy_score >= 0.7:
            return "fair"
        else:
            return "poor"
    
    async def detect_model_drift(self, model_name: str) -> Optional[ModelDrift]:
        """Detect model drift using statistical tests"""
        try:
            if model_name not in self.prediction_history:
                return None
            
            predictions = list(self.prediction_history[model_name])
            if len(predictions) < self.drift_window_size * 2:
                logger.debug(f"Insufficient data for drift detection: {len(predictions)} < {self.drift_window_size * 2}")
                return None
            
            # Split data into baseline and current windows
            baseline_data = predictions[-self.drift_window_size*2:-self.drift_window_size]
            current_data = predictions[-self.drift_window_size:]
            
            # Detect different types of drift
            
            # 1. Data drift (changes in input distribution)
            data_drift = self._detect_data_drift(baseline_data, current_data)
            
            # 2. Concept drift (changes in prediction patterns)
            concept_drift = self._detect_concept_drift(baseline_data, current_data)
            
            # 3. Performance drift (changes in accuracy/confidence)
            performance_drift = self._detect_performance_drift(baseline_data, current_data)
            
            # Determine most significant drift
            drift_types = [
                ("data_drift", data_drift),
                ("concept_drift", concept_drift),
                ("performance_drift", performance_drift)
            ]
            
            significant_drifts = []
            for drift_type, drift_score in drift_types:
                threshold = self.drift_thresholds.get(drift_type, 0.1)
                if drift_score > threshold:
                    alert_level = self._determine_alert_level(drift_score, threshold)
                    significant_drifts.append({
                        "type": drift_type,
                        "score": drift_score,
                        "threshold": threshold,
                        "alert_level": alert_level
                    })
            
            if not significant_drifts:
                return None
            
            # Return the most significant drift
            most_significant = max(significant_drifts, key=lambda x: x["score"])
            
            # Calculate metrics before and after
            metrics_before = self._calculate_metrics(baseline_data)
            metrics_after = self._calculate_metrics(current_data)
            
            drift_alert = ModelDrift(
                model_name=model_name,
                drift_type=most_significant["type"],
                drift_score=most_significant["score"],
                threshold=most_significant["threshold"],
                confidence_level=0.95,
                timestamp=datetime.utcnow(),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                alert_level=most_significant["alert_level"]
            )
            
            # Store alert
            self.drift_alerts.append(drift_alert)
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.lpush(
                    "medical_ai:drift_alerts",
                    json.dumps(asdict(drift_alert), default=str)
                )
            
            logger.warning(f"Model drift detected for {model_name}: {most_significant['type']} "
                         f"(score: {most_significant['score']:.3f})")
            
            return drift_alert
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return None
    
    def _detect_data_drift(self, baseline_data: List[Dict[str, Any]], 
                          current_data: List[Dict[str, Any]]) -> float:
        """Detect data drift using statistical tests"""
        try:
            # Extract feature vectors (simplified - using confidence scores as proxy)
            baseline_features = [d.get("confidence", 0.5) for d in baseline_data]
            current_features = [d.get("confidence", 0.5) for d in current_data]
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(baseline_features, current_features)
            
            # Return drift score (1 - p_value for significance)
            drift_score = 1 - p_value
            
            return min(drift_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Data drift detection failed: {str(e)}")
            return 0.0
    
    def _detect_concept_drift(self, baseline_data: List[Dict[str, Any]], 
                            current_data: List[Dict[str, Any]]) -> float:
        """Detect concept drift (changes in prediction patterns)"""
        try:
            # Extract predictions
            baseline_predictions = [str(d.get("prediction", "")) for d in baseline_data]
            current_predictions = [str(d.get("prediction", "")) for d in current_data]
            
            # Calculate prediction distribution changes
            from collections import Counter
            
            baseline_dist = Counter(baseline_predictions)
            current_dist = Counter(current_predictions)
            
            # Calculate Jensen-Shannon divergence
            all_predictions = set(baseline_predictions + current_predictions)
            
            baseline_probs = np.array([baseline_dist.get(pred, 0) / len(baseline_predictions) 
                                     for pred in all_predictions])
            current_probs = np.array([current_dist.get(pred, 0) / len(current_predictions) 
                                    for pred in all_predictions])
            
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            baseline_probs = baseline_probs + epsilon
            current_probs = current_probs + epsilon
            baseline_probs = baseline_probs / baseline_probs.sum()
            current_probs = current_probs / current_probs.sum()
            
            # Jensen-Shannon divergence
            m = 0.5 * (baseline_probs + current_probs)
            js_divergence = 0.5 * (stats.entropy(baseline_probs, m) + stats.entropy(current_probs, m))
            
            return js_divergence
            
        except Exception as e:
            logger.warning(f"Concept drift detection failed: {str(e)}")
            return 0.0
    
    def _detect_performance_drift(self, baseline_data: List[Dict[str, Any]], 
                                current_data: List[Dict[str, Any]]) -> float:
        """Detect performance drift (changes in model performance)"""
        try:
            # Compare confidence scores and success rates
            baseline_confidences = [d.get("confidence", 0.5) for d in baseline_data]
            current_confidences = [d.get("confidence", 0.5) for d in current_data]
            
            baseline_success_rate = np.mean([1 if d.get("success", True) else 0 for d in baseline_data])
            current_success_rate = np.mean([1 if d.get("success", True) else 0 for d in current_data])
            
            # Calculate performance metrics
            baseline_avg_confidence = np.mean(baseline_confidences)
            current_avg_confidence = np.mean(current_confidences)
            
            # Calculate performance drift score
            confidence_change = abs(current_avg_confidence - baseline_avg_confidence) / baseline_avg_confidence
            success_rate_change = abs(current_success_rate - baseline_success_rate)
            
            performance_drift = (confidence_change + success_rate_change) / 2
            
            return performance_drift
            
        except Exception as e:
            logger.warning(f"Performance drift detection failed: {str(e)}")
            return 0.0
    
    def _calculate_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics for a dataset"""
        if not data:
            return {}
        
        confidences = [d.get("confidence", 0.0) for d in data]
        success_rate = np.mean([1 if d.get("success", True) else 0 for d in data])
        processing_times = [d.get("processing_time", 0.0) for d in data]
        
        return {
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
            "success_rate": success_rate,
            "avg_processing_time": np.mean(processing_times),
            "sample_size": len(data)
        }
    
    def _determine_alert_level(self, drift_score: float, threshold: float) -> str:
        """Determine alert level based on drift score and threshold"""
        ratio = drift_score / threshold
        
        if ratio < 1.2:
            return "low"
        elif ratio < 1.5:
            return "medium"
        elif ratio < 2.0:
            return "high"
        else:
            return "critical"
    
    async def get_model_performance_summary(self, model_name: str, 
                                          hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive model performance summary"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Get recent predictions
            recent_predictions = [
                pred for pred in self.prediction_history[model_name]
                if datetime.fromisoformat(pred["timestamp"]) >= cutoff_time
            ]
            
            if not recent_predictions:
                return {"message": "No recent data available"}
            
            # Calculate performance metrics
            metrics = self._calculate_metrics(recent_predictions)
            
            # Get clinical validation results
            clinical_results = [
                outcome for outcome in self.actual_outcomes.get(model_name, [])
                if outcome.validation_time >= cutoff_time
            ]
            
            # Calculate clinical metrics
            if clinical_results:
                clinical_accuracy = np.mean([outcome.accuracy_score for outcome in clinical_results])
                clinical_significance_counts = defaultdict(int)
                for outcome in clinical_results:
                    clinical_significance_counts[outcome.clinical_significance] += 1
            else:
                clinical_accuracy = 0.0
                clinical_significance_counts = {}
            
            # Get recent drift alerts
            recent_drifts = [
                drift for drift in self.drift_alerts
                if drift.model_name == model_name and drift.timestamp >= cutoff_time
            ]
            
            return {
                "model_name": model_name,
                "time_range_hours": hours,
                "performance_metrics": metrics,
                "clinical_validation": {
                    "total_validations": len(clinical_results),
                    "accuracy": clinical_accuracy,
                    "significance_distribution": dict(clinical_significance_counts),
                    "validation_rate": len(clinical_results) / len(recent_predictions) if recent_predictions else 0
                },
                "drift_alerts": {
                    "count": len(recent_drifts),
                    "latest_alert": asdict(recent_drifts[-1]) if recent_drifts else None,
                    "alert_levels": [drift.alert_level for drift in recent_drifts]
                },
                "recommendations": self._generate_recommendations(metrics, clinical_accuracy, recent_drifts),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance summary generation failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, performance_metrics: Dict[str, float],
                                clinical_accuracy: float, drift_alerts: List[ModelDrift]) -> List[str]:
        """Generate actionable recommendations based on monitoring data"""
        recommendations = []
        
        # Performance-based recommendations
        success_rate = performance_metrics.get("success_rate", 1.0)
        if success_rate < 0.95:
            recommendations.append("Low success rate detected - investigate model performance")
        
        avg_confidence = performance_metrics.get("avg_confidence", 0.0)
        if avg_confidence < 0.7:
            recommendations.append("Low average confidence - consider model retraining")
        
        avg_processing_time = performance_metrics.get("avg_processing_time", 0.0)
        if avg_processing_time > 1.0:
            recommendations.append("High processing time - consider model optimization")
        
        # Clinical validation recommendations
        if clinical_accuracy < 0.8:
            recommendations.append("Poor clinical validation accuracy - immediate attention required")
        elif clinical_accuracy < 0.9:
            recommendations.append("Moderate clinical accuracy - schedule model review")
        
        # Drift-based recommendations
        if drift_alerts:
            latest_drift = drift_alerts[-1]
            if latest_drift.alert_level in ["high", "critical"]:
                recommendations.append(f"Significant drift detected ({latest_drift.drift_type}) - initiate retraining pipeline")
            elif latest_drift.alert_level == "medium":
                recommendations.append("Moderate drift detected - monitor closely and prepare for retraining")
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append("Model performance is within acceptable parameters")
        
        return recommendations
    
    async def get_all_models_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of all monitored models"""
        summaries = {}
        
        for model_name in self.prediction_history.keys():
            summary = await self.get_model_performance_summary(model_name, hours)
            summaries[model_name] = summary
        
        # Calculate overall system health
        total_models = len(summaries)
        healthy_models = 0
        critical_models = 0
        
        for model_name, summary in summaries.items():
            if "error" not in summary:
                if summary.get("clinical_validation", {}).get("accuracy", 0) > 0.8:
                    healthy_models += 1
                else:
                    critical_models += 1
        
        return {
            "overall_health": {
                "total_models": total_models,
                "healthy_models": healthy_models,
                "critical_models": critical_models,
                "health_score": healthy_models / total_models if total_models > 0 else 0
            },
            "model_summaries": summaries,
            "report_timestamp": datetime.utcnow().isoformat()
        }
    
    async def export_monitoring_data(self, model_name: str, format: str = "json") -> str:
        """Export monitoring data for analysis"""
        try:
            data = {
                "predictions": list(self.prediction_history[model_name]),
                "clinical_validations": [asdict(outcome) for outcome in self.actual_outcomes[model_name]],
                "drift_alerts": [asdict(drift) for drift in self.drift_alerts if drift.model_name == model_name],
                "export_timestamp": datetime.utcnow().isoformat()
            }
            
            if format.lower() == "json":
                return json.dumps(data, indent=2, default=str)
            elif format.lower() == "csv":
                # Convert to DataFrame and export as CSV
                df = pd.DataFrame(data["predictions"])
                return df.to_csv(index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Data export failed: {str(e)}")
            raise
    
    # Background monitoring tasks
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while True:
            try:
                await self.performance_monitor._metrics_collection_loop()
                await asyncio.sleep(self.config.get("monitoring_interval", 300))
            except Exception as e:
                logger.error(f"Monitoring loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _drift_detection_loop(self):
        """Continuous drift detection loop"""
        while True:
            try:
                for model_name in self.prediction_history.keys():
                    await self.detect_model_drift(model_name)
                
                await asyncio.sleep(600)  # Run every 10 minutes
            except Exception as e:
                logger.error(f"Drift detection loop error: {str(e)}")
                await asyncio.sleep(120)
    
    async def _clinical_validation_loop(self):
        """Clinical validation processing loop"""
        while True:
            try:
                # Process pending clinical validations
                for model_name, outcomes in self.actual_outcomes.items():
                    pending_validations = [o for o in outcomes if o.validation_status == "pending"]
                    
                    for validation in pending_validations:
                        # Process validation (could trigger alerts, retraining, etc.)
                        if validation.clinical_significance == "poor":
                            logger.warning(f"Poor clinical validation for patient {validation.patient_id}")
                
                await asyncio.sleep(1800)  # Run every 30 minutes
            except Exception as e:
                logger.error(f"Clinical validation loop error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _alert_processing_loop(self):
        """Alert processing and notification loop"""
        while True:
            try:
                # Get recent alerts from Redis
                if self.redis_client:
                    alerts = await self.redis_client.lrange("medical_ai:drift_alerts", 0, -1)
                    
                    for alert_json in alerts:
                        alert_data = json.loads(alert_json)
                        
                        # Process alert (send notifications, trigger actions)
                        await self._process_alert(alert_data)
                
                await asyncio.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Alert processing loop error: {str(e)}")
                await asyncio.sleep(120)
    
    async def _process_alert(self, alert_data: Dict[str, Any]):
        """Process a monitoring alert"""
        try:
            alert_type = alert_data.get("drift_type")
            model_name = alert_data.get("model_name")
            alert_level = alert_data.get("alert_level")
            
            logger.warning(f"ALERT: {alert_level} {alert_type} drift detected for {model_name}")
            
            # In production, this would:
            # - Send notifications to on-call team
            # - Trigger automated actions
            # - Update dashboards
            # - Create incident tickets
            
        except Exception as e:
            logger.error(f"Alert processing failed: {str(e)}")
    
    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        await self.performance_monitor.close()
        logger.info("Model Monitoring System closed")

# Example usage
if __name__ == "__main__":
    async def run_monitoring_example():
        # Initialize monitoring system
        monitoring = ModelMonitoringSystem()
        await monitoring.initialize()
        
        # Log sample predictions
        for i in range(50):
            await monitoring.log_prediction("medical-diagnosis-v1", {
                "prediction": "condition_a" if i % 2 == 0 else "condition_b",
                "confidence": np.random.uniform(0.6, 0.95),
                "processing_time": np.random.uniform(0.1, 0.5),
                "patient_id": f"patient_{i}",
                "success": True
            })
        
        # Record some clinical outcomes
        for i in range(20):
            await monitoring.record_actual_outcome("medical-diagnosis-v1", f"patient_{i}", {
                "actual_result": "condition_a" if i % 3 == 0 else "condition_b",
                "clinical_notes": "Follow-up completed"
            })
        
        # Detect drift
        drift = await monitoring.detect_model_drift("medical-diagnosis-v1")
        if drift:
            print(f"Drift detected: {drift.drift_type} - {drift.drift_score:.3f}")
        
        # Get performance summary
        summary = await monitoring.get_model_performance_summary("medical-diagnosis-v1", hours=1)
        print(f"Performance summary: {summary['recommendations']}")
        
        # Get overall health
        health = await monitoring.get_all_models_summary(hours=1)
        print(f"Overall health score: {health['overall_health']['health_score']:.2f}")
    
    # Run example
    asyncio.run(run_monitoring_example())