"""
A/B Testing Framework for Medical AI Models
Traffic splitting, performance comparison, and statistical analysis.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import asyncio
import json
import hashlib
from collections import defaultdict, deque
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'registry'))
from model_registry import ModelRegistry, ABTestConfig, ABTestResult

logger = logging.getLogger(__name__)

@dataclass
class TrafficAssignment:
    """Traffic assignment for A/B test"""
    test_id: str
    patient_id: str
    model_version: str
    assignment_time: datetime
    traffic_percentage: float

@dataclass
class TestMetrics:
    """Real-time test metrics"""
    test_id: str
    model_version: str
    timestamp: datetime
    requests: int
    successes: int
    failures: int
    avg_latency: float
    p95_latency: float
    avg_confidence: float
    accuracy: Optional[float] = None

class ABTestFramework:
    """Production A/B testing framework for medical AI models"""
    
    def __init__(self, config_path: str = "config/ab_testing_config.yaml"):
        self.config = self._load_config(config_path)
        
        # A/B Test management
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.traffic_assignments: Dict[str, TrafficAssignment] = {}
        self.test_metrics: Dict[str, List[TestMetrics]] = defaultdict(list)
        
        # Traffic routing
        self.routing_cache: Dict[str, str] = {}  # patient_id -> model_version
        self.routing_ttl = timedelta(hours=24)
        
        # Model registry integration
        self.model_registry = ModelRegistry()
        
        # Performance tracking
        self.prediction_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.clinical_outcomes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Statistical analysis
        self.significance_threshold = self.config.get("significance_threshold", 0.05)
        self.min_sample_size = self.config.get("min_sample_size", 100)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load A/B testing configuration"""
        default_config = {
            "significance_threshold": 0.05,
            "min_sample_size": 100,
            "min_test_duration_hours": 24,
            "traffic_routing_cache_ttl": 24,
            "metrics_collection_interval": 60,
            "auto_stop_conditions": {
                "max_duration_hours": 168,  # 1 week
                "min_improvement_threshold": 0.02,
                "max_sample_size": 10000
            },
            "clinical_validation": {
                "require_clinical_outcomes": True,
                "min_clinical_samples": 50,
                "clinical_significance_threshold": 0.1
            }
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"A/B testing config {config_path} not found, using defaults")
            return default_config
    
    async def initialize(self):
        """Initialize the A/B testing framework"""
        logger.info("Initializing A/B Testing Framework...")
        
        try:
            # Initialize model registry
            await self._initialize_model_registry()
            
            # Load existing tests
            await self._load_existing_tests()
            
            # Start background tasks
            asyncio.create_task(self._metrics_collection_loop())
            asyncio.create_task(self._test_evaluation_loop())
            asyncio.create_task(self._traffic_cleanup_loop())
            
            logger.info("A/B Testing Framework initialization complete")
            
        except Exception as e:
            logger.error(f"A/B Testing Framework initialization failed: {str(e)}")
            raise
    
    async def _initialize_model_registry(self):
        """Initialize model registry connection"""
        # This would connect to actual MLflow in production
        logger.info("Model registry initialized")
    
    async def _load_existing_tests(self):
        """Load existing A/B tests from storage"""
        # In production, load from persistent storage
        logger.info("Loaded existing A/B tests")
    
    async def start_ab_test(self, test_config: ABTestConfig) -> str:
        """Start a new A/B test"""
        try:
            # Validate test configuration
            self._validate_test_config(test_config)
            
            # Create test in model registry
            registry_test_id = self.model_registry.create_ab_test(test_config)
            
            # Store test configuration
            self.active_tests[test_config.test_id] = test_config
            
            # Initialize metrics tracking
            self.test_metrics[test_config.test_id] = []
            self.prediction_history[test_config.control_model] = []
            self.prediction_history[test_config.treatment_model] = []
            
            # Log test start
            logger.info(f"A/B Test started: {test_config.name} ({test_config.test_id})")
            
            return test_config.test_id
            
        except Exception as e:
            logger.error(f"Failed to start A/B test: {str(e)}")
            raise
    
    async def route_request(self, patient_id: str, test_id: Optional[str] = None) -> str:
        """Route request to appropriate model version"""
        
        # Check cache first
        if patient_id in self.routing_cache:
            cached_assignment = self.routing_cache[patient_id]
            # Check if assignment is still valid
            assignment_time = cached_assignment.get("assignment_time")
            if assignment_time:
                assign_time = datetime.fromisoformat(assignment_time)
                if datetime.utcnow() - assign_time < self.routing_ttl:
                    return cached_assignment["model_version"]
        
        # Find applicable test
        applicable_test = None
        if test_id and test_id in self.active_tests:
            applicable_test = self.active_tests[test_id]
        else:
            # Find test by patient characteristics (simplified)
            applicable_test = self._find_applicable_test(patient_id)
        
        if not applicable_test:
            # No A/B test applies, return default model
            return "medical-diagnosis-v1"
        
        # Assign to model based on traffic split
        model_version = self._assign_model_version(patient_id, applicable_test)
        
        # Cache assignment
        self.routing_cache[patient_id] = {
            "model_version": model_version,
            "test_id": applicable_test.test_id,
            "assignment_time": datetime.utcnow().isoformat()
        }
        
        # Store traffic assignment
        traffic_assignment = TrafficAssignment(
            test_id=applicable_test.test_id,
            patient_id=patient_id,
            model_version=model_version,
            assignment_time=datetime.utcnow(),
            traffic_percentage=applicable_test.traffic_split
        )
        
        self.traffic_assignments[f"{applicable_test.test_id}:{patient_id}"] = traffic_assignment
        
        return model_version
    
    def _find_applicable_test(self, patient_id: str) -> Optional[ABTestConfig]:
        """Find A/B test applicable to patient"""
        # Simplified test selection - in production would use patient characteristics
        active_tests = [t for t in self.active_tests.values() if t.status == "active"]
        
        if not active_tests:
            return None
        
        # Return first active test for simplicity
        return active_tests[0]
    
    def _assign_model_version(self, patient_id: str, test_config: ABTestConfig) -> str:
        """Assign patient to control or treatment model"""
        # Use consistent hashing for deterministic assignment
        hash_value = int(hashlib.md5(patient_id.encode()).hexdigest(), 16)
        assignment_threshold = int(test_config.traffic_split * 100)
        
        if hash_value % 100 < assignment_threshold:
            return test_config.treatment_model
        else:
            return test_config.control_model
    
    async def record_prediction(self, test_id: str, patient_id: str, 
                              model_version: str, prediction_data: Dict[str, Any]):
        """Record prediction for A/B test analysis"""
        try:
            if test_id not in self.active_tests:
                return
            
            # Add test context to prediction data
            enriched_data = {
                "test_id": test_id,
                "patient_id": patient_id,
                "model_version": model_version,
                "timestamp": datetime.utcnow().isoformat(),
                **prediction_data
            }
            
            # Store prediction history
            self.prediction_history[model_version].append(enriched_data)
            
            # Update real-time metrics
            await self._update_test_metrics(test_id, model_version, enriched_data)
            
            # Record in model registry
            self.model_registry.record_prediction_result(test_id, model_version, prediction_data)
            
            logger.debug(f"Prediction recorded for test {test_id}, model {model_version}")
            
        except Exception as e:
            logger.error(f"Prediction recording failed: {str(e)}")
    
    async def _update_test_metrics(self, test_id: str, model_version: str, 
                                  prediction_data: Dict[str, Any]):
        """Update real-time test metrics"""
        try:
            # Get or create metrics entry for current hour
            current_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            
            metrics_list = self.test_metrics[test_id]
            
            # Find existing metrics for this model and hour
            existing_metrics = None
            for metrics in metrics_list:
                if (metrics.model_version == model_version and 
                    metrics.timestamp == current_time):
                    existing_metrics = metrics
                    break
            
            if existing_metrics:
                # Update existing metrics
                existing_metrics.requests += 1
                if prediction_data.get("success", True):
                    existing_metrics.successes += 1
                else:
                    existing_metrics.failures += 1
                
                # Update latency
                processing_time = prediction_data.get("processing_time", 0)
                existing_metrics.avg_latency = (
                    (existing_metrics.avg_latency * (existing_metrics.requests - 1) + processing_time) /
                    existing_metrics.requests
                )
                
                # Update confidence
                confidence = prediction_data.get("confidence", 0)
                existing_metrics.avg_confidence = (
                    (existing_metrics.avg_confidence * (existing_metrics.requests - 1) + confidence) /
                    existing_metrics.requests
                )
                
            else:
                # Create new metrics entry
                metrics = TestMetrics(
                    test_id=test_id,
                    model_version=model_version,
                    timestamp=current_time,
                    requests=1,
                    successes=1 if prediction_data.get("success", True) else 0,
                    failures=0 if prediction_data.get("success", True) else 1,
                    avg_latency=prediction_data.get("processing_time", 0),
                    p95_latency=prediction_data.get("processing_time", 0),
                    avg_confidence=prediction_data.get("confidence", 0)
                )
                metrics_list.append(metrics)
                
                # Keep only recent metrics (last 48 hours)
                if len(metrics_list) > 48:
                    metrics_list.pop(0)
                    
        except Exception as e:
            logger.error(f"Metrics update failed: {str(e)}")
    
    async def record_clinical_outcome(self, test_id: str, patient_id: str, 
                                    outcome_data: Dict[str, Any]):
        """Record clinical outcome for A/B test validation"""
        try:
            if test_id not in self.active_tests:
                return
            
            outcome_record = {
                "test_id": test_id,
                "patient_id": patient_id,
                "timestamp": datetime.utcnow().isoformat(),
                **outcome_data
            }
            
            self.clinical_outcomes[test_id].append(outcome_record)
            
            logger.debug(f"Clinical outcome recorded for test {test_id}, patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Clinical outcome recording failed: {str(e)}")
    
    async def evaluate_test(self, test_id: str) -> Optional[ABTestResult]:
        """Evaluate A/B test results"""
        try:
            if test_id not in self.active_tests:
                return None
            
            test_config = self.active_tests[test_id]
            
            # Get performance data
            control_data = self.prediction_history.get(test_config.control_model, [])
            treatment_data = self.prediction_history.get(test_config.treatment_model, [])
            
            # Check if test meets completion criteria
            if not self._should_complete_test(test_config, control_data, treatment_data):
                return None
            
            # Calculate metrics
            control_metrics = self._calculate_comprehensive_metrics(control_data)
            treatment_metrics = self._calculate_comprehensive_metrics(treatment_data)
            
            # Perform statistical analysis
            significance_result = self._perform_statistical_analysis(
                control_data, treatment_data, test_config.success_metrics
            )
            
            # Determine winner
            winner = self._determine_winner(
                control_metrics, treatment_metrics, significance_result, test_config
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                control_metrics, treatment_metrics, significance_result, winner, test_config
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
                recommendations=recommendations
            )
            
            # Store result in model registry
            # In production, this would be done through the registry
            
            logger.info(f"A/B test {test_id} evaluated: Winner = {winner}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test evaluation failed: {str(e)}")
            return None
    
    def _calculate_comprehensive_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not data:
            return {}
        
        # Basic metrics
        sample_size = len(data)
        
        # Latency metrics
        processing_times = [d.get("processing_time", 0) for d in data]
        latencies = [t * 1000 for t in processing_times]  # Convert to milliseconds
        
        # Confidence metrics
        confidences = [d.get("confidence", 0) for d in data if d.get("confidence") is not None]
        
        # Accuracy metrics
        accuracies = [d.get("accuracy") for d in data if d.get("accuracy") is not None]
        
        # Success rate
        successes = sum(1 for d in data if d.get("success", True))
        success_rate = successes / sample_size
        
        # Error rate
        error_rate = 1 - success_rate
        
        metrics = {
            "sample_size": sample_size,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "p50_latency_ms": np.percentile(latencies, 50) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            "max_latency_ms": np.max(latencies) if latencies else 0,
            "avg_confidence": np.mean(confidences) if confidences else 0,
            "min_confidence": np.min(confidences) if confidences else 0,
            "max_confidence": np.max(confidences) if confidences else 0
        }
        
        if accuracies:
            metrics.update({
                "accuracy": np.mean(accuracies),
                "precision": np.mean(accuracies),  # Simplified for demo
                "recall": np.mean(accuracies),     # Simplified for demo
                "f1_score": np.mean(accuracies)    # Simplified for demo
            })
        
        return metrics
    
    def _perform_statistical_analysis(self, control_data: List[Dict[str, Any]],
                                    treatment_data: List[Dict[str, Any]],
                                    success_metrics: List[str]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        
        # Extract confidence scores for primary comparison
        control_confidences = [d.get("confidence", 0) for d in control_data if d.get("confidence")]
        treatment_confidences = [d.get("confidence", 0) for d in treatment_data if d.get("confidence")]
        
        # Primary statistical test (t-test for confidence scores)
        primary_result = self._perform_t_test(control_confidences, treatment_confidences)
        
        # Secondary tests
        secondary_results = {}
        
        # Test success rates
        control_success_rates = [1 if d.get("success", True) else 0 for d in control_data]
        treatment_success_rates = [1 if d.get("success", True) else 0 for d in treatment_data]
        secondary_results["success_rate"] = self._perform_t_test(control_success_rates, treatment_success_rates)
        
        # Test latency distributions
        control_latencies = [d.get("processing_time", 0) * 1000 for d in control_data]
        treatment_latencies = [d.get("processing_time", 0) * 1000 for d in treatment_data]
        secondary_results["latency"] = self._perform_t_test(control_latencies, treatment_latencies)
        
        # Effect size calculation (Cohen's d)
        effect_size = self._calculate_cohens_d(control_confidences, treatment_confidences)
        
        return {
            "primary_test": primary_result,
            "secondary_tests": secondary_results,
            "effect_size": effect_size,
            "significant": primary_result["significant"],
            "p_value": primary_result["p_value"]
        }
    
    def _perform_t_test(self, control_values: List[float], 
                       treatment_values: List[float]) -> Dict[str, Any]:
        """Perform t-test on two samples"""
        if not control_values or not treatment_values:
            return {"significant": False, "p_value": 1.0, "statistic": 0.0}
        
        try:
            statistic, p_value = stats.ttest_ind(control_values, treatment_values)
            
            return {
                "significant": p_value < self.significance_threshold,
                "p_value": p_value,
                "statistic": statistic,
                "control_mean": np.mean(control_values),
                "treatment_mean": np.mean(treatment_values),
                "control_std": np.std(control_values),
                "treatment_std": np.std(treatment_values)
            }
        except Exception as e:
            logger.warning(f"T-test failed: {str(e)}")
            return {"significant": False, "p_value": 1.0, "statistic": 0.0}
    
    def _calculate_cohens_d(self, control_values: List[float], 
                           treatment_values: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        if not control_values or not treatment_values:
            return 0.0
        
        try:
            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(
                (np.var(control_values) + np.var(treatment_values)) / 2
            )
            
            if pooled_std == 0:
                return 0.0
            
            cohens_d = (treatment_mean - control_mean) / pooled_std
            
            return cohens_d
            
        except Exception as e:
            logger.warning(f"Cohen's d calculation failed: {str(e)}")
            return 0.0
    
    def _determine_winner(self, control_metrics: Dict[str, float],
                         treatment_metrics: Dict[str, float],
                         significance_result: Dict[str, Any],
                         test_config: ABTestConfig) -> str:
        """Determine the winner based on statistical analysis and clinical criteria"""
        
        if not significance_result["significant"]:
            return "inconclusive"
        
        # Primary metric comparison (usually accuracy or confidence)
        primary_metric = test_config.success_metrics[0] if test_config.success_metrics else "accuracy"
        
        control_score = control_metrics.get(primary_metric, 0.0)
        treatment_score = treatment_metrics.get(primary_metric, 0.0)
        
        # Clinical significance check
        improvement_threshold = self.config.get("auto_stop_conditions", {}).get("min_improvement_threshold", 0.02)
        score_difference = treatment_score - control_score
        
        if score_difference < improvement_threshold:
            return "control"  # Not clinically significant improvement
        
        # Performance impact assessment
        latency_impact = treatment_metrics.get("avg_latency_ms", 0) - control_metrics.get("avg_latency_ms", 0)
        
        # Consider latency penalties
        if latency_impact > 100:  # 100ms penalty
            if score_difference < 0.05:  # Less than 5% improvement
                return "control"
        
        # Determine winner
        if treatment_score > control_score:
            return "treatment"
        elif control_score > treatment_score:
            return "control"
        else:
            return "inconclusive"
    
    def _generate_recommendations(self, control_metrics: Dict[str, float],
                                treatment_metrics: Dict[str, float],
                                significance_result: Dict[str, Any],
                                winner: str, test_config: ABTestConfig) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if winner == "treatment":
            recommendations.append("Deploy treatment model to production")
            recommendations.append("Begin gradual rollout to larger patient populations")
            
            # Performance insights
            confidence_improvement = treatment_metrics.get("avg_confidence", 0) - control_metrics.get("avg_confidence", 0)
            if confidence_improvement > 0.05:
                recommendations.append(f"Treatment model shows {confidence_improvement:.3f} confidence improvement")
            
            latency_impact = treatment_metrics.get("avg_latency_ms", 0) - control_metrics.get("avg_latency_ms", 0)
            if latency_impact > 50:
                recommendations.append("Consider latency optimization for treatment model")
                
        elif winner == "control":
            recommendations.append("Continue with current control model")
            recommendations.append("Investigate treatment model performance issues")
            recommendations.append("Consider additional training data for treatment model")
            
        else:
            recommendations.append("Results inconclusive - extend test duration")
            recommendations.append("Consider increasing sample size")
            recommendations.append("Review metric selection and thresholds")
        
        # Statistical insights
        if significance_result["significant"]:
            p_value = significance_result["p_value"]
            effect_size = significance_result.get("effect_size", 0)
            
            recommendations.append(f"Results are statistically significant (p={p_value:.4f})")
            
            if abs(effect_size) > 0.8:
                recommendations.append("Large effect size detected - results are practically significant")
            elif abs(effect_size) > 0.5:
                recommendations.append("Medium effect size detected")
            else:
                recommendations.append("Small effect size detected - consider clinical significance")
        
        # Clinical validation recommendations
        clinical_samples = len(self.clinical_outcomes.get(test_config.test_id, []))
        min_clinical = self.config.get("clinical_validation", {}).get("min_clinical_samples", 50)
        
        if clinical_samples < min_clinical:
            recommendations.append(f"Collect more clinical outcomes (current: {clinical_samples}, required: {min_clinical})")
        
        return recommendations
    
    def _should_complete_test(self, test_config: ABTestConfig, 
                            control_data: List[Dict[str, Any]], 
                            treatment_data: List[Dict[str, Any]]) -> bool:
        """Determine if test should be completed"""
        
        # Check minimum sample size
        min_samples = test_config.min_sample_size
        if len(control_data) < min_samples or len(treatment_data) < min_samples:
            return False
        
        # Check minimum duration
        elapsed_time = datetime.utcnow() - test_config.start_time
        min_duration = timedelta(hours=test_config.duration_hours)
        if elapsed_time < min_duration:
            return False
        
        # Check auto-stop conditions
        auto_conditions = self.config.get("auto_stop_conditions", {})
        
        # Maximum duration check
        max_duration = timedelta(hours=auto_conditions.get("max_duration_hours", 168))
        if elapsed_time > max_duration:
            return True
        
        # Statistical significance achieved with sufficient power
        if len(control_data) > 1000 and len(treatment_data) > 1000:
            return True
        
        return False
    
    def _validate_test_config(self, test_config: ABTestConfig):
        """Validate A/B test configuration"""
        if not test_config.control_model or not test_config.treatment_model:
            raise ValueError("Both control and treatment models must be specified")
        
        if test_config.control_model == test_config.treatment_model:
            raise ValueError("Control and treatment models must be different")
        
        if not 0 < test_config.traffic_split < 1:
            raise ValueError("Traffic split must be between 0 and 1")
        
        if test_config.min_sample_size < 10:
            raise ValueError("Minimum sample size must be at least 10")
        
        if test_config.duration_hours < 1:
            raise ValueError("Test duration must be at least 1 hour")
    
    # Background tasks
    
    async def _metrics_collection_loop(self):
        """Background task for continuous metrics collection"""
        while True:
            try:
                # Aggregate and store metrics periodically
                for test_id in self.active_tests:
                    if self.active_tests[test_id].status == "active":
                        await self._collect_aggregated_metrics(test_id)
                
                await asyncio.sleep(self.config.get("metrics_collection_interval", 60))
                
            except Exception as e:
                logger.error(f"Metrics collection loop error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _test_evaluation_loop(self):
        """Background task for continuous test evaluation"""
        while True:
            try:
                for test_id, test_config in self.active_tests.items():
                    if test_config.status == "active":
                        await self.evaluate_test(test_id)
                
                await asyncio.sleep(300)  # Evaluate every 5 minutes
                
            except Exception as e:
                logger.error(f"Test evaluation loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _traffic_cleanup_loop(self):
        """Background task to cleanup expired traffic assignments"""
        while True:
            try:
                cutoff_time = datetime.utcnow() - self.routing_ttl
                
                expired_keys = []
                for key, assignment in self.traffic_assignments.items():
                    if assignment.assignment_time < cutoff_time:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.traffic_assignments[key]
                
                # Cleanup routing cache
                expired_cache_keys = []
                for patient_id, assignment in self.routing_cache.items():
                    assign_time = datetime.fromisoformat(assignment["assignment_time"])
                    if assign_time < cutoff_time:
                        expired_cache_keys.append(patient_id)
                
                for patient_id in expired_cache_keys:
                    del self.routing_cache[patient_id]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired traffic assignments")
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except Exception as e:
                logger.error(f"Traffic cleanup loop error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _collect_aggregated_metrics(self, test_id: str):
        """Collect aggregated metrics for a test"""
        # This would aggregate data and send to monitoring systems
        logger.debug(f"Collecting aggregated metrics for test {test_id}")
    
    async def stop_test(self, test_id: str, reason: str = "manual_stop") -> bool:
        """Manually stop an A/B test"""
        try:
            if test_id not in self.active_tests:
                return False
            
            test_config = self.active_tests[test_id]
            test_config.status = "stopped"
            test_config.end_time = datetime.utcnow()
            
            # Record stop in model registry
            logger.info(f"A/B test {test_id} stopped: {reason}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop test {test_id}: {str(e)}")
            return False
    
    def get_test_status(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive test status"""
        if test_id not in self.active_tests:
            return None
        
        test_config = self.active_tests[test_id]
        
        # Get current metrics
        current_metrics = self.test_metrics.get(test_id, [])
        latest_metrics = current_metrics[-1] if current_metrics else None
        
        # Calculate current sample sizes
        control_size = len(self.prediction_history.get(test_config.control_model, []))
        treatment_size = len(self.prediction_history.get(test_config.treatment_model, []))
        
        # Calculate elapsed time
        elapsed_time = datetime.utcnow() - test_config.start_time
        
        return {
            "test_config": asdict(test_config),
            "current_status": {
                "elapsed_time_hours": elapsed_time.total_seconds() / 3600,
                "sample_sizes": {
                    "control": control_size,
                    "treatment": treatment_size,
                    "total": control_size + treatment_size
                },
                "min_sample_size": test_config.min_sample_size,
                "sample_size_met": control_size >= test_config.min_sample_size and treatment_size >= test_config.min_sample_size,
                "duration_met": elapsed_time.total_seconds() >= (test_config.duration_hours * 3600)
            },
            "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
            "clinical_outcomes_count": len(self.clinical_outcomes.get(test_id, []))
        }
    
    def get_all_tests(self) -> List[Dict[str, Any]]:
        """Get status of all tests"""
        return [self.get_test_status(test_id) for test_id in self.active_tests]
    
    def get_traffic_distribution(self, test_id: str) -> Dict[str, Any]:
        """Get current traffic distribution for a test"""
        if test_id not in self.active_tests:
            return {}
        
        test_config = self.active_tests[test_id]
        
        # Count assignments by model
        control_count = 0
        treatment_count = 0
        
        for assignment in self.traffic_assignments.values():
            if assignment.test_id == test_id:
                if assignment.model_version == test_config.control_model:
                    control_count += 1
                elif assignment.model_version == test_config.treatment_model:
                    treatment_count += 1
        
        total_count = control_count + treatment_count
        actual_split = treatment_count / total_count if total_count > 0 else 0
        
        return {
            "control_count": control_count,
            "treatment_count": treatment_count,
            "total_count": total_count,
            "actual_traffic_split": actual_split,
            "configured_split": test_config.traffic_split,
            "split_deviation": abs(actual_split - test_config.traffic_split)
        }

# Example usage
if __name__ == "__main__":
    import uuid
    from datetime import datetime
    
    # Initialize A/B testing framework
    framework = ABTestFramework()
    
    # Create test configuration
    test_config = ABTestConfig(
        test_id=str(uuid.uuid4()),
        name="Medical Diagnosis Model A/B Test",
        control_model="medical-diagnosis-v1",
        treatment_model="medical-diagnosis-v2",
        traffic_split=0.5,
        duration_hours=24,
        min_sample_size=500,
        success_metrics=["accuracy", "confidence"],
        start_time=datetime.utcnow()
    )
    
    # Start test
    import asyncio
    
    async def run_example():
        await framework.initialize()
        
        test_id = await framework.start_ab_test(test_config)
        print(f"A/B Test started: {test_id}")
        
        # Simulate traffic
        for i in range(1000):
            patient_id = f"patient_{i}"
            
            # Route request
            model_version = await framework.route_request(patient_id, test_id)
            
            # Simulate prediction
            prediction_data = {
                "prediction": {"diagnosis": "condition"},
                "confidence": np.random.uniform(0.7, 0.95),
                "processing_time": np.random.uniform(0.1, 0.5),
                "success": True,
                "accuracy": np.random.uniform(0.8, 0.95)
            }
            
            await framework.record_prediction(test_id, patient_id, model_version, prediction_data)
        
        # Get test status
        status = framework.get_test_status(test_id)
        print(f"Test Status: {status['current_status']}")
        
        # Evaluate test
        result = await framework.evaluate_test(test_id)
        if result:
            print(f"Test Result: Winner = {result.winner}")
            print(f"Recommendations: {result.recommendations}")
    
    # Run example
    asyncio.run(run_example())