"""
Performance Monitor Utility
Real-time monitoring and analytics for medical AI model performance.
"""

import asyncio
import logging
import time
import psutil
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json
import aioredis
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class ModelMetric:
    """Model performance metric"""
    model_name: str
    timestamp: datetime
    latency_ms: float
    throughput_qps: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy_score: Optional[float] = None
    confidence_score: Optional[float] = None
    error_count: int = 0
    total_requests: int = 0

@dataclass
class ClinicalOutcome:
    """Clinical outcome tracking"""
    patient_id: str
    model_prediction: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]] = None
    prediction_time: Optional[datetime] = None
    outcome_time: Optional[datetime] = None
    accuracy_score: Optional[float] = None
    validation_status: str = "pending"

class PerformanceMonitor:
    """Production performance monitoring for medical AI models"""
    
    def __init__(self, config_path: str = "config/performance_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Metrics storage
        self.model_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.clinical_outcomes: List[ClinicalOutcome] = []
        self.request_history: deque = deque(maxlen=10000)
        self.system_metrics: deque = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = self.config.get("thresholds", {})
        self.alert_handlers = []
        
        # Redis connection for distributed monitoring
        self.redis_client = None
        self.metrics_redis_key = "medical_ai:metrics"
        self.outcomes_redis_key = "medical_ai:outcomes"
        
        # Performance calculation windows
        self.latency_window_size = self.config.get("latency_window_size", 100)
        self.throughput_window_size = self.config.get("throughput_window_size", 1000)
        
        # Drift detection parameters
        self.drift_threshold = self.config.get("drift_threshold", 0.1)
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load performance monitor configuration"""
        default_config = {
            "metrics_interval": 60,
            "thresholds": {
                "latency_p95": 2000,  # 2 seconds
                "latency_p99": 5000,  # 5 seconds
                "throughput_min": 10,  # QPS
                "memory_usage_max": 80,  # percentage
                "cpu_usage_max": 80,  # percentage
                "accuracy_min": 0.85,
                "error_rate_max": 0.01
            },
            "latency_window_size": 100,
            "throughput_window_size": 1000,
            "drift_threshold": 0.1,
            "enable_distributed_monitoring": True,
            "redis_host": "localhost",
            "redis_port": 6379
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Performance config {config_path} not found, using defaults")
            return default_config
    
    async def initialize(self):
        """Initialize the performance monitor"""
        logger.info("Initializing Performance Monitor...")
        
        try:
            # Initialize Redis connection if enabled
            if self.config.get("enable_distributed_monitoring", True):
                self.redis_client = await aioredis.from_url(
                    f"redis://{self.config['redis_host']}:{self.config['redis_port']}"
                )
                logger.info("Distributed monitoring enabled")
            
            # Start monitoring tasks
            asyncio.create_task(self._metrics_collection_loop())
            asyncio.create_task(self._drift_detection_loop())
            asyncio.create_task(self._alerts_processing_loop())
            
            logger.info("Performance Monitor initialization complete")
            
        except Exception as e:
            logger.error(f"Performance Monitor initialization failed: {str(e)}")
            raise
    
    async def _metrics_collection_loop(self):
        """Continuous metrics collection loop"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._process_pending_metrics()
                
                await asyncio.sleep(self.config.get("metrics_interval", 60))
                
            except Exception as e:
                logger.error(f"Metrics collection error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metric = {
                "timestamp": timestamp.isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
                "network_io": dict(psutil.net_io_counters()._asdict()) if psutil.net_io_counters() else {},
                "process_count": len(psutil.pids())
            }
            
            self.system_metrics.append(system_metric)
            
            # Store in Redis if available
            if self.redis_client:
                await self.redis_client.lpush(
                    self.metrics_redis_key + ":system",
                    json.dumps(system_metric)
                )
                # Keep only recent entries
                await self.redis_client.ltrim(
                    self.metrics_redis_key + ":system", 0, 999
                )
            
        except Exception as e:
            logger.error(f"System metrics collection error: {str(e)}")
    
    async def _process_pending_metrics(self):
        """Process any pending metrics from Redis"""
        if not self.redis_client:
            return
        
        try:
            # Get pending model metrics
            model_metrics_json = await self.redis_client.lrange(
                self.metrics_redis_key + ":pending", 0, -1
            )
            
            for metric_json in model_metrics_json:
                try:
                    metric_data = json.loads(metric_json)
                    metric = ModelMetric(**metric_data)
                    
                    model_name = metric.model_name
                    self.model_metrics[model_name].append(metric)
                    
                except Exception as e:
                    logger.warning(f"Failed to process metric: {str(e)}")
            
            # Clear processed metrics
            if model_metrics_json:
                await self.redis_client.delete(self.metrics_redis_key + ":pending")
            
        except Exception as e:
            logger.warning(f"Pending metrics processing error: {str(e)}")
    
    async def log_prediction(self, prediction_data: Dict[str, Any]):
        """Log a model prediction for performance tracking"""
        try:
            timestamp = datetime.utcnow()
            
            # Extract metrics from prediction data
            model_name = prediction_data.get("model_version", "unknown")
            latency_ms = prediction_data.get("processing_time", 0) * 1000
            confidence = prediction_data.get("confidence", 0.0)
            patient_id = prediction_data.get("patient_id", "unknown")
            
            # Create metric
            metric = ModelMetric(
                model_name=model_name,
                timestamp=timestamp,
                latency_ms=latency_ms,
                throughput_qps=0,  # Will be calculated later
                memory_usage_mb=0,  # Will be updated by system metrics
                cpu_usage_percent=psutil.cpu_percent(),
                accuracy_score=prediction_data.get("accuracy_score"),
                confidence_score=confidence,
                total_requests=1
            )
            
            # Store metric
            self.model_metrics[model_name].append(metric)
            
            # Store in Redis for distributed monitoring
            if self.redis_client:
                await self.redis_client.lpush(
                    self.metrics_redis_key + ":pending",
                    json.dumps(asdict(metric))
                )
            
            # Store clinical outcome if available
            if prediction_data.get("actual_outcome"):
                outcome = ClinicalOutcome(
                    patient_id=patient_id,
                    model_prediction=prediction_data,
                    actual_outcome=prediction_data["actual_outcome"],
                    prediction_time=timestamp,
                    outcome_time=datetime.utcnow()
                )
                self.clinical_outcomes.append(outcome)
            
        except Exception as e:
            logger.error(f"Prediction logging error: {str(e)}")
    
    async def log_batch_prediction(self, batch_data: Dict[str, Any]):
        """Log batch prediction metrics"""
        try:
            timestamp = datetime.utcnow()
            batch_id = batch_data.get("batch_id", "unknown")
            total_requests = batch_data.get("total_requests", 0)
            success_count = batch_data.get("success_count", 0)
            error_count = batch_data.get("error_count", 0)
            
            # Calculate batch metrics
            processing_time = batch_data.get("processing_time", 0)
            throughput = total_requests / processing_time if processing_time > 0 else 0
            
            # Log as aggregate metric
            metric = ModelMetric(
                model_name="batch_processor",
                timestamp=timestamp,
                latency_ms=processing_time * 1000,
                throughput_qps=throughput,
                memory_usage_mb=0,
                cpu_usage_percent=psutil.cpu_percent(),
                total_requests=total_requests,
                error_count=error_count
            )
            
            self.model_metrics["batch_processor"].append(metric)
            
        except Exception as e:
            logger.error(f"Batch prediction logging error: {str(e)}")
    
    def get_model_performance(self, model_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get performance metrics for a specific model"""
        if model_name not in self.model_metrics:
            return {}
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = [
            m for m in self.model_metrics[model_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if not metrics:
            return {"error": "No metrics available for the specified time range"}
        
        # Calculate statistics
        latencies = [m.latency_ms for m in metrics]
        throughputs = [m.throughput_qps for m in metrics]
        confidences = [m.confidence_score for m in metrics if m.confidence_score is not None]
        
        return {
            "model_name": model_name,
            "time_range_hours": hours,
            "sample_count": len(metrics),
            "latency": {
                "p50": np.percentile(latencies, 50) if latencies else 0,
                "p95": np.percentile(latencies, 95) if latencies else 0,
                "p99": np.percentile(latencies, 99) if latencies else 0,
                "avg": np.mean(latencies) if latencies else 0,
                "min": min(latencies) if latencies else 0,
                "max": max(latencies) if latencies else 0
            },
            "throughput": {
                "avg": np.mean(throughputs) if throughputs else 0,
                "max": max(throughputs) if throughputs else 0,
                "min": min(throughputs) if throughputs else 0
            },
            "accuracy": {
                "avg_confidence": np.mean(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0
            },
            "reliability": {
                "error_rate": sum(m.error_count for m in metrics) / max(sum(m.total_requests for m in metrics), 1),
                "total_requests": sum(m.total_requests for m in metrics),
                "total_errors": sum(m.error_count for m in metrics)
            }
        }
    
    def get_model_metrics(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model metrics in the format expected by the API"""
        performance = self.get_model_performance(model_name, hours=1)
        
        if "error" in performance:
            return None
        
        return {
            "model_version": model_name,
            "total_requests": performance["reliability"]["total_requests"],
            "average_latency": performance["latency"]["avg"],
            "accuracy_score": performance["accuracy"]["avg_confidence"],
            "throughput": performance["throughput"]["avg"],
            "memory_usage": 0,  # Would be populated from system metrics
            "last_updated": datetime.utcnow().isoformat()
        }
    
    async def _drift_detection_loop(self):
        """Continuous drift detection"""
        while True:
            try:
                await self._check_model_drift()
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Drift detection error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _check_model_drift(self):
        """Check for model performance drift"""
        for model_name, metrics in self.model_metrics.items():
            if len(metrics) < 10:  # Need minimum samples
                continue
            
            recent_metrics = list(metrics)[-50:]  # Last 50 predictions
            
            # Calculate current performance
            current_latency = np.mean([m.latency_ms for m in recent_metrics])
            current_confidence = np.mean([m.confidence_score for m in recent_metrics if m.confidence_score])
            
            # Compare with baseline
            if model_name in self.baseline_metrics:
                baseline = self.baseline_metrics[model_name]
                
                # Check latency drift
                baseline_latency = baseline.get("latency_avg", current_latency)
                latency_drift = abs(current_latency - baseline_latency) / baseline_latency
                
                # Check confidence drift
                baseline_confidence = baseline.get("confidence_avg", current_confidence)
                confidence_drift = abs(current_confidence - baseline_confidence) / baseline_confidence
                
                # Alert if drift exceeds threshold
                if latency_drift > self.drift_threshold or confidence_drift > self.drift_threshold:
                    logger.warning(
                        f"Model drift detected for {model_name}: "
                        f"latency_drift={latency_drift:.3f}, confidence_drift={confidence_drift:.3f}"
                    )
                    
                    # Create alert
                    await self._create_drift_alert(
                        model_name, latency_drift, confidence_drift,
                        current_latency, current_confidence
                    )
    
    async def _create_drift_alert(self, model_name: str, latency_drift: float, 
                                confidence_drift: float, current_latency: float, 
                                current_confidence: float):
        """Create drift detection alert"""
        alert = {
            "type": "model_drift",
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_drift": latency_drift,
            "confidence_drift": confidence_drift,
            "current_latency": current_latency,
            "current_confidence": current_confidence,
            "threshold": self.drift_threshold,
            "severity": "warning" if max(latency_drift, confidence_drift) < 0.2 else "critical"
        }
        
        # Store alert
        if self.redis_client:
            await self.redis_client.lpush(
                self.metrics_redis_key + ":alerts",
                json.dumps(alert)
            )
    
    async def _alerts_processing_loop(self):
        """Process performance alerts"""
        while True:
            try:
                if self.redis_client:
                    alerts = await self.redis_client.lrange(
                        self.metrics_redis_key + ":alerts", 0, -1
                    )
                    
                    for alert_json in alerts:
                        try:
                            alert = json.loads(alert_json)
                            await self._process_alert(alert)
                        except Exception as e:
                            logger.warning(f"Alert processing error: {str(e)}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Alert processing loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _process_alert(self, alert: Dict[str, Any]):
        """Process a performance alert"""
        alert_type = alert.get("type")
        
        if alert_type == "model_drift":
            logger.warning(
                f"MODEL DRIFT ALERT: {alert['model_name']} - "
                f"Latency drift: {alert['latency_drift']:.3f}, "
                f"Confidence drift: {alert['confidence_drift']:.3f}"
            )
            
            # In production, this would trigger:
            # - Auto-scaling
            # - Model retraining pipeline
            # - Notification to ML ops team
            # - Automatic model rollback if necessary
    
    def establish_baseline(self, model_name: str, hours: int = 24):
        """Establish performance baseline for a model"""
        if model_name not in self.model_metrics:
            return False
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        metrics = [
            m for m in self.model_metrics[model_name] 
            if m.timestamp >= cutoff_time
        ]
        
        if len(metrics) < 10:
            logger.warning(f"Insufficient data to establish baseline for {model_name}")
            return False
        
        # Calculate baseline metrics
        latencies = [m.latency_ms for m in metrics]
        confidences = [m.confidence_score for m in metrics if m.confidence_score is not None]
        
        self.baseline_metrics[model_name] = {
            "latency_avg": np.mean(latencies),
            "latency_std": np.std(latencies),
            "confidence_avg": np.mean(confidences) if confidences else 0.0,
            "confidence_std": np.std(confidences) if confidences else 0.0,
            "established_at": datetime.utcnow().isoformat(),
            "sample_count": len(metrics)
        }
        
        logger.info(f"Performance baseline established for {model_name}")
        return True
    
    def get_clinical_outcomes_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get clinical outcomes summary"""
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_outcomes = [
            o for o in self.clinical_outcomes 
            if o.prediction_time and o.prediction_time >= cutoff_time
        ]
        
        if not recent_outcomes:
            return {"message": "No recent clinical outcomes available"}
        
        # Calculate accuracy metrics
        validated_outcomes = [o for o in recent_outcomes if o.accuracy_score is not None]
        
        return {
            "time_range_days": days,
            "total_predictions": len(recent_outcomes),
            "validated_outcomes": len(validated_outcomes),
            "validation_rate": len(validated_outcomes) / len(recent_outcomes),
            "average_accuracy": np.mean([o.accuracy_score for o in validated_outcomes]) if validated_outcomes else 0,
            "accuracy_distribution": {
                "excellent": sum(1 for o in validated_outcomes if o.accuracy_score >= 0.9),
                "good": sum(1 for o in validated_outcomes if 0.8 <= o.accuracy_score < 0.9),
                "fair": sum(1 for o in validated_outcomes if 0.7 <= o.accuracy_score < 0.8),
                "poor": sum(1 for o in validated_outcomes if o.accuracy_score < 0.7)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Performance Monitor closed")