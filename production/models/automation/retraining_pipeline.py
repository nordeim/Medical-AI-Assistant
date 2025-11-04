"""
Automated Retraining Pipeline System
Production-grade automated model retraining based on performance metrics and drift detection.
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
from collections import defaultdict
import subprocess
import pickle
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'monitoring'))
from model_monitor import ModelMonitoringSystem
from utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

@dataclass
class RetrainingTrigger:
    """Retraining trigger configuration"""
    trigger_id: str
    model_name: str
    trigger_type: str  # performance_degradation, drift_detected, schedule, manual
    severity: str  # low, medium, high, critical
    triggered_at: datetime
    threshold_values: Dict[str, float]
    current_values: Dict[str, float]
    status: str = "pending"  # pending, approved, running, completed, failed

@dataclass
class TrainingJob:
    """Training job definition"""
    job_id: str
    model_name: str
    trigger_id: str
    training_config: Dict[str, Any]
    data_source: Dict[str, Any]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "queued"  # queued, running, completed, failed
    progress: float = 0.0
    metrics: Dict[str, Any] = None
    artifacts: List[str] = None

@dataclass
class TrainingResult:
    """Training job result"""
    job_id: str
    model_name: str
    success: bool
    performance_improvement: float
    new_model_version: str
    validation_metrics: Dict[str, float]
    deployment_recommendation: str
    training_time_seconds: float
    artifact_paths: List[str]

class AutomatedRetrainingSystem:
    """Production automated retraining pipeline system"""
    
    def __init__(self, config_path: str = "config/retraining_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Component initialization
        self.model_monitor = ModelMonitoringSystem()
        self.performance_monitor = PerformanceMonitor()
        
        # Retraining triggers and jobs
        self.active_triggers: Dict[str, RetrainingTrigger] = {}
        self.training_queue: List[TrainingJob] = []
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        
        # Trigger thresholds
        self.trigger_thresholds = self.config.get("trigger_thresholds", {})
        self.retraining_schedule = self.config.get("retraining_schedule", {})
        
        # Training environment
        self.training_workspace = self.config.get("training_workspace", "/tmp/model_training")
        self.model_registry_path = self.config.get("model_registry_path", "/tmp/model_registry")
        
        # Redis for distributed coordination
        self.redis_client = None
        
        # Pipeline status
        self.pipeline_status = {
            "total_triggers": 0,
            "total_jobs": 0,
            "success_rate": 0.0,
            "avg_training_time": 0.0
        }
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load retraining configuration"""
        default_config = {
            "trigger_thresholds": {
                "accuracy_degradation": 0.05,  # 5% drop
                "precision_degradation": 0.05,
                "recall_degradation": 0.05,
                "confidence_drop": 0.1,  # 10% drop in avg confidence
                "drift_score_threshold": 0.3,
                "error_rate_increase": 0.02  # 2% increase in error rate
            },
            "retraining_schedule": {
                "enabled": True,
                "frequency_hours": 168,  # Weekly
                "max_queue_size": 5,
                "concurrent_jobs": 2
            },
            "training_workspace": "/tmp/model_training",
            "model_registry_path": "/tmp/model_registry",
            "data_sources": {
                "production_data": {
                    "type": "database",
                    "connection": "postgresql://user:pass@localhost/medical_data",
                    "table": "predictions_with_outcomes"
                },
                "historical_data": {
                    "type": "file",
                    "path": "/data/historical_training_data.csv"
                }
            },
            "training_config": {
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "max_epochs": 100,
                "learning_rate": 0.001,
                "batch_size": 32,
                "architecture_search": False
            },
            "validation_criteria": {
                "min_accuracy_improvement": 0.02,
                "max_training_time_hours": 24,
                "required_metrics": ["accuracy", "precision", "recall", "f1_score"]
            },
            "deployment_policy": {
                "auto_deploy": False,
                "require_approval": True,
                "staging_duration_hours": 24
            }
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Retraining config {config_path} not found, using defaults")
            return default_config
    
    async def initialize(self):
        """Initialize the automated retraining system"""
        logger.info("Initializing Automated Retraining System...")
        
        try:
            # Create necessary directories
            os.makedirs(self.training_workspace, exist_ok=True)
            os.makedirs(self.model_registry_path, exist_ok=True)
            
            # Initialize monitoring components
            await self.model_monitor.initialize()
            await self.performance_monitor.initialize()
            
            # Initialize Redis connection
            redis_host = self.config.get("redis_host", "localhost")
            redis_port = self.config.get("redis_port", 6379)
            self.redis_client = await aioredis.from_url(f"redis://{redis_host}:{redis_port}")
            
            # Load existing triggers and jobs
            await self._load_pipeline_state()
            
            # Start background tasks
            asyncio.create_task(self._trigger_monitoring_loop())
            asyncio.create_task(self._training_queue_processor())
            asyncio.create_task(self._job_monitoring_loop())
            asyncio.create_task(self._schedule_monitoring_loop())
            
            logger.info("Automated Retraining System initialized")
            
        except Exception as e:
            logger.error(f"Retraining system initialization failed: {str(e)}")
            raise
    
    async def _load_pipeline_state(self):
        """Load existing pipeline state from storage"""
        # In production, load from persistent storage
        logger.info("Pipeline state loaded")
    
    async def check_retraining_triggers(self, model_name: str) -> List[RetrainingTrigger]:
        """Check if any retraining triggers are met for a model"""
        try:
            triggers = []
            
            # Get model performance summary
            performance_summary = await self.model_monitor.get_model_performance_summary(model_name, hours=24)
            
            if "error" in performance_summary:
                logger.warning(f"Cannot check triggers for {model_name}: {performance_summary['error']}")
                return triggers
            
            # Check performance degradation triggers
            performance_triggers = await self._check_performance_triggers(model_name, performance_summary)
            triggers.extend(performance_triggers)
            
            # Check drift-based triggers
            drift_triggers = await self._check_drift_triggers(model_name)
            triggers.extend(drift_triggers)
            
            # Check schedule-based triggers
            schedule_triggers = await self._check_schedule_triggers(model_name)
            triggers.extend(schedule_triggers)
            
            # Store triggers
            for trigger in triggers:
                self.active_triggers[trigger.trigger_id] = trigger
                self.pipeline_status["total_triggers"] += 1
            
            # Store triggers in Redis for distributed processing
            if self.redis_client:
                for trigger in triggers:
                    await self.redis_client.lpush(
                        "medical_ai:retraining_triggers",
                        json.dumps(asdict(trigger), default=str)
                    )
            
            if triggers:
                logger.info(f"Created {len(triggers)} retraining triggers for {model_name}")
            
            return triggers
            
        except Exception as e:
            logger.error(f"Trigger checking failed for {model_name}: {str(e)}")
            return []
    
    async def _check_performance_triggers(self, model_name: str, 
                                        performance_summary: Dict[str, Any]) -> List[RetrainingTrigger]:
        """Check performance-based retraining triggers"""
        triggers = []
        
        try:
            perf_metrics = performance_summary.get("performance_metrics", {})
            clinical_metrics = performance_summary.get("clinical_validation", {})
            
            current_values = {}
            if perf_metrics:
                current_values["success_rate"] = perf_metrics.get("success_rate", 1.0)
                current_values["avg_confidence"] = perf_metrics.get("avg_confidence", 0.0)
                current_values["avg_processing_time"] = perf_metrics.get("avg_processing_time", 0.0)
            
            if clinical_metrics:
                current_values["clinical_accuracy"] = clinical_metrics.get("accuracy", 0.0)
            
            # Compare with thresholds
            for metric, current_value in current_values.items():
                threshold_key = f"{metric}_degradation" if metric != "avg_confidence" else "confidence_drop"
                threshold = self.trigger_thresholds.get(threshold_key)
                
                if threshold is not None and self._evaluate_threshold(metric, current_value, threshold):
                    trigger = RetrainingTrigger(
                        trigger_id=f"perf_{model_name}_{metric}_{int(datetime.utcnow().timestamp())}",
                        model_name=model_name,
                        trigger_type="performance_degradation",
                        severity=self._determine_trigger_severity(metric, current_value, threshold),
                        triggered_at=datetime.utcnow(),
                        threshold_values={threshold_key: threshold},
                        current_values={metric: current_value}
                    )
                    triggers.append(trigger)
            
        except Exception as e:
            logger.warning(f"Performance trigger checking failed: {str(e)}")
        
        return triggers
    
    async def _check_drift_triggers(self, model_name: str) -> List[RetrainingTrigger]:
        """Check drift-based retraining triggers"""
        triggers = []
        
        try:
            # Get recent drift alerts
            health_summary = await self.model_monitor.get_all_models_summary(hours=24)
            model_summary = health_summary.get("model_summaries", {}).get(model_name, {})
            
            drift_alerts = model_summary.get("drift_alerts", {})
            if drift_alerts.get("count", 0) > 0:
                latest_alert = drift_alerts.get("latest_alert")
                if latest_alert:
                    drift_score = latest_alert.get("drift_score", 0.0)
                    threshold = self.trigger_thresholds.get("drift_score_threshold", 0.3)
                    
                    if drift_score > threshold:
                        trigger = RetrainingTrigger(
                            trigger_id=f"drift_{model_name}_{int(datetime.utcnow().timestamp())}",
                            model_name=model_name,
                            trigger_type="drift_detected",
                            severity=self._determine_drift_severity(drift_score, threshold),
                            triggered_at=datetime.utcnow(),
                            threshold_values={"drift_score_threshold": threshold},
                            current_values={"drift_score": drift_score}
                        )
                        triggers.append(trigger)
        
        except Exception as e:
            logger.warning(f"Drift trigger checking failed: {str(e)}")
        
        return triggers
    
    async def _check_schedule_triggers(self, model_name: str) -> List[RetrainingTrigger]:
        """Check schedule-based retraining triggers"""
        triggers = []
        
        if not self.retraining_schedule.get("enabled", True):
            return triggers
        
        try:
            # Check if it's time for scheduled retraining
            frequency_hours = self.retraining_schedule.get("frequency_hours", 168)
            
            # Find last retraining time for this model
            last_training_time = await self._get_last_training_time(model_name)
            
            if last_training_time is None or \
               (datetime.utcnow() - last_training_time).total_seconds() > (frequency_hours * 3600):
                
                trigger = RetrainingTrigger(
                    trigger_id=f"sched_{model_name}_{int(datetime.utcnow().timestamp())}",
                    model_name=model_name,
                    trigger_type="schedule",
                    severity="medium",
                    triggered_at=datetime.utcnow(),
                    threshold_values={"frequency_hours": frequency_hours},
                    current_values={"hours_since_last_training": (datetime.utcnow() - (last_training_time or datetime.utcnow())).total_seconds() / 3600}
                )
                triggers.append(trigger)
        
        except Exception as e:
            logger.warning(f"Schedule trigger checking failed: {str(e)}")
        
        return triggers
    
    def _evaluate_threshold(self, metric: str, current_value: float, threshold: float) -> bool:
        """Evaluate if a metric exceeds threshold"""
        if metric in ["success_rate", "clinical_accuracy"]:
            return current_value < (1.0 - threshold)
        elif metric == "avg_confidence":
            return current_value < threshold
        elif metric == "avg_processing_time":
            return current_value > threshold
        else:
            return False
    
    def _determine_trigger_severity(self, metric: str, current_value: float, threshold: float) -> str:
        """Determine trigger severity based on how much threshold is exceeded"""
        if metric in ["success_rate", "clinical_accuracy"]:
            deficit = (1.0 - threshold) - current_value
        elif metric == "avg_confidence":
            deficit = threshold - current_value
        else:
            deficit = current_value - threshold
        
        if deficit > threshold * 2:
            return "critical"
        elif deficit > threshold * 1.5:
            return "high"
        elif deficit > threshold:
            return "medium"
        else:
            return "low"
    
    def _determine_drift_severity(self, drift_score: float, threshold: float) -> str:
        """Determine drift severity"""
        ratio = drift_score / threshold
        if ratio > 2.0:
            return "critical"
        elif ratio > 1.5:
            return "high"
        elif ratio > 1.2:
            return "medium"
        else:
            return "low"
    
    async def _get_last_training_time(self, model_name: str) -> Optional[datetime]:
        """Get the last training time for a model"""
        # Check completed jobs
        for job in self.completed_jobs.values():
            if job.model_name == model_name and job.end_time:
                return job.end_time
        
        return None
    
    async def trigger_retraining(self, trigger: RetrainingTrigger) -> Optional[str]:
        """Trigger retraining based on a retraining trigger"""
        try:
            # Check if we can accept new jobs
            if len(self.training_queue) >= self.retraining_schedule.get("max_queue_size", 5):
                logger.warning(f"Training queue full, cannot trigger retraining for {trigger.model_name}")
                return None
            
            # Create training job
            job_id = f"job_{trigger.model_name}_{int(datetime.utcnow().timestamp())}"
            
            training_job = TrainingJob(
                job_id=job_id,
                model_name=trigger.model_name,
                trigger_id=trigger.trigger_id,
                training_config=self.config.get("training_config", {}),
                data_source=self.config.get("data_sources", {}),
                status="queued",
                progress=0.0
            )
            
            # Add to queue
            self.training_queue.append(training_job)
            self.running_jobs[job_id] = training_job
            
            # Update trigger status
            trigger.status = "approved"
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.lpush(
                    "medical_ai:training_queue",
                    json.dumps(asdict(training_job), default=str)
                )
            
            self.pipeline_status["total_jobs"] += 1
            
            logger.info(f"Retraining triggered for {trigger.model_name} (job: {job_id})")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Retraining trigger failed: {str(e)}")
            return None
    
    async def process_training_queue(self):
        """Process training queue and start jobs"""
        while True:
            try:
                # Check if we can start new jobs
                max_concurrent = self.retraining_schedule.get("concurrent_jobs", 2)
                current_running = len([job for job in self.running_jobs.values() if job.status == "running"])
                
                if current_running < max_concurrent and self.training_queue:
                    # Get next job from queue
                    job = self.training_queue.pop(0)
                    
                    # Start training job
                    await self._start_training_job(job)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Training queue processing error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _start_training_job(self, job: TrainingJob):
        """Start a training job"""
        try:
            logger.info(f"Starting training job {job.job_id} for {job.model_name}")
            
            # Update job status
            job.status = "running"
            job.start_time = datetime.utcnow()
            job.progress = 0.0
            
            # Start training in background
            asyncio.create_task(self._execute_training_job(job))
            
        except Exception as e:
            logger.error(f"Failed to start training job {job.job_id}: {str(e)}")
            job.status = "failed"
    
    async def _execute_training_job(self, job: TrainingJob):
        """Execute the actual training job"""
        try:
            logger.info(f"Executing training job {job.job_id}")
            
            # Create training workspace
            workspace_dir = os.path.join(self.training_workspace, job.job_id)
            os.makedirs(workspace_dir, exist_ok=True)
            
            # Prepare training data
            training_data = await self._prepare_training_data(job)
            
            # Execute training
            training_start = datetime.utcnow()
            job.progress = 0.1
            
            # Mock training process (in production, would use actual ML training)
            result = await self._run_mock_training(job, training_data, workspace_dir)
            
            job.progress = 1.0
            job.end_time = datetime.utcnow()
            job.status = "completed"
            job.metrics = result.validation_metrics
            job.artifacts = result.artifact_paths
            
            # Move to completed jobs
            self.completed_jobs[job.job_id] = job
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
            
            # Update pipeline statistics
            await self._update_pipeline_stats()
            
            logger.info(f"Training job {job.job_id} completed successfully")
            
            # Trigger deployment pipeline if configured
            if self.config.get("deployment_policy", {}).get("auto_deploy", False):
                await self._initiate_deployment_pipeline(result)
            
        except Exception as e:
            logger.error(f"Training job {job.job_id} failed: {str(e)}")
            job.status = "failed"
            job.end_time = datetime.utcnow()
            
            if job.job_id in self.running_jobs:
                del self.running_jobs[job.job_id]
    
    async def _prepare_training_data(self, job: TrainingJob) -> Dict[str, Any]:
        """Prepare training data for the job"""
        try:
            # In production, this would:
            # 1. Query production database for recent data
            # 2. Apply data quality checks
            # 3. Split into train/validation sets
            # 4. Apply feature engineering
            
            # Mock data preparation
            data_size = 10000
            features = np.random.rand(data_size, 20)
            labels = np.random.randint(0, 5, data_size)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            logger.info(f"Prepared training data: {len(X_train)} train, {len(X_val)} validation samples")
            
            return {
                "X_train": X_train,
                "X_val": X_val,
                "y_train": y_train,
                "y_val": y_val,
                "data_summary": {
                    "total_samples": data_size,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "features": X_train.shape[1],
                    "classes": len(np.unique(labels))
                }
            }
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {str(e)}")
            raise
    
    async def _run_mock_training(self, job: TrainingJob, training_data: Dict[str, Any], 
                                workspace_dir: str) -> TrainingResult:
        """Run mock training (replace with actual training pipeline)"""
        try:
            # Mock training progress updates
            for progress in [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]:
                job.progress = progress
                await asyncio.sleep(2)  # Simulate training time
                logger.debug(f"Training progress: {progress * 100:.0f}%")
            
            # Calculate mock metrics
            val_accuracy = np.random.uniform(0.85, 0.95)
            val_precision = np.random.uniform(0.80, 0.90)
            val_recall = np.random.uniform(0.80, 0.90)
            val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall)
            
            validation_metrics = {
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1_score": val_f1
            }
            
            # Create mock model artifact
            model_path = os.path.join(workspace_dir, f"{job.model_name}_trained.joblib")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(training_data["X_train"], training_data["y_train"])
            joblib.dump(model, model_path)
            
            # Calculate performance improvement
            baseline_accuracy = 0.82  # Mock baseline
            performance_improvement = val_accuracy - baseline_accuracy
            
            # Determine deployment recommendation
            min_improvement = self.config.get("validation_criteria", {}).get("min_accuracy_improvement", 0.02)
            if performance_improvement > min_improvement:
                recommendation = "deploy_to_staging"
            elif performance_improvement > 0:
                recommendation = "review_and_approve"
            else:
                recommendation = "do_not_deploy"
            
            training_time = (datetime.utcnow() - job.start_time).total_seconds()
            
            return TrainingResult(
                job_id=job.job_id,
                model_name=job.model_name,
                success=True,
                performance_improvement=performance_improvement,
                new_model_version=f"v{int(datetime.utcnow().timestamp())}",
                validation_metrics=validation_metrics,
                deployment_recommendation=recommendation,
                training_time_seconds=training_time,
                artifact_paths=[model_path]
            )
            
        except Exception as e:
            logger.error(f"Mock training failed: {str(e)}")
            raise
    
    async def _initiate_deployment_pipeline(self, result: TrainingResult):
        """Initiate deployment pipeline for trained model"""
        try:
            logger.info(f"Initiating deployment pipeline for {result.model_name}")
            
            # In production, this would:
            # 1. Deploy to staging environment
            # 2. Run A/B tests
            # 3. Monitor performance
            # 4. Deploy to production if successful
            
            # Mock deployment
            await asyncio.sleep(5)
            
            logger.info(f"Deployment pipeline initiated for {result.model_name}")
            
        except Exception as e:
            logger.error(f"Deployment pipeline initiation failed: {str(e)}")
    
    async def _update_pipeline_stats(self):
        """Update pipeline statistics"""
        try:
            # Calculate success rate
            completed_jobs = [job for job in self.completed_jobs.values()]
            if completed_jobs:
                successful_jobs = [job for job in completed_jobs if job.status == "completed"]
                self.pipeline_status["success_rate"] = len(successful_jobs) / len(completed_jobs)
                
                # Calculate average training time
                training_times = [
                    (job.end_time - job.start_time).total_seconds() 
                    for job in successful_jobs 
                    if job.start_time and job.end_time
                ]
                if training_times:
                    self.pipeline_status["avg_training_time"] = np.mean(training_times)
        
        except Exception as e:
            logger.warning(f"Pipeline stats update failed: {str(e)}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "pipeline_stats": self.pipeline_status,
            "active_triggers": len(self.active_triggers),
            "queued_jobs": len(self.training_queue),
            "running_jobs": len([job for job in self.running_jobs.values() if job.status == "running"]),
            "completed_jobs": len(self.completed_jobs),
            "recent_triggers": [
                asdict(trigger) for trigger in list(self.active_triggers.values())[-5:]
            ],
            "recent_jobs": [
                {
                    "job_id": job.job_id,
                    "model_name": job.model_name,
                    "status": job.status,
                    "progress": job.progress,
                    "duration": (job.end_time - job.start_time).total_seconds() if job.start_time and job.end_time else None
                }
                for job in list(self.completed_jobs.values())[-5:]
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def cancel_training_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        try:
            # Cancel queued job
            for i, job in enumerate(self.training_queue):
                if job.job_id == job_id:
                    job.status = "cancelled"
                    self.training_queue.pop(i)
                    logger.info(f"Cancelled queued training job {job_id}")
                    return True
            
            # Cancel running job
            if job_id in self.running_jobs:
                job = self.running_jobs[job_id]
                job.status = "cancelled"
                del self.running_jobs[job_id]
                logger.info(f"Cancelled running training job {job_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Job cancellation failed: {str(e)}")
            return False
    
    # Background tasks
    
    async def _trigger_monitoring_loop(self):
        """Monitor for retraining triggers"""
        while True:
            try:
                # Get all models from monitoring system
                health_summary = await self.model_monitor.get_all_models_summary(hours=1)
                model_summaries = health_summary.get("model_summaries", {})
                
                for model_name in model_summaries.keys():
                    await self.check_retraining_triggers(model_name)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Trigger monitoring loop error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _training_queue_processor(self):
        """Process training queue"""
        await self.process_training_queue()
    
    async def _job_monitoring_loop(self):
        """Monitor running training jobs"""
        while True:
            try:
                # Check job health and update status
                for job_id, job in list(self.running_jobs.items()):
                    if job.status == "running":
                        # Check if job has timed out
                        if job.start_time:
                            max_duration = self.config.get("validation_criteria", {}).get("max_training_time_hours", 24) * 3600
                            elapsed = (datetime.utcnow() - job.start_time).total_seconds()
                            
                            if elapsed > max_duration:
                                logger.warning(f"Training job {job_id} timed out")
                                job.status = "failed"
                                job.end_time = datetime.utcnow()
                                del self.running_jobs[job_id]
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Job monitoring loop error: {str(e)}")
                await asyncio.sleep(120)
    
    async def _schedule_monitoring_loop(self):
        """Monitor scheduled retraining"""
        while True:
            try:
                if self.retraining_schedule.get("enabled", True):
                    # Check all models for scheduled retraining
                    health_summary = await self.model_monitor.get_all_models_summary(hours=24)
                    model_summaries = health_summary.get("model_summaries", {})
                    
                    for model_name in model_summaries.keys():
                        schedule_triggers = await self._check_schedule_triggers(model_name)
                        for trigger in schedule_triggers:
                            await self.trigger_retraining(trigger)
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Schedule monitoring loop error: {str(e)}")
                await asyncio.sleep(600)
    
    async def close(self):
        """Clean up resources"""
        if self.redis_client:
            await self.redis_client.close()
        await self.model_monitor.close()
        await self.performance_monitor.close()
        logger.info("Automated Retraining System closed")

# Example usage
if __name__ == "__main__":
    async def run_retraining_example():
        # Initialize system
        retraining_system = AutomatedRetrainingSystem()
        await retraining_system.initialize()
        
        # Mock some performance degradation
        print("Simulating performance degradation...")
        
        # Check for triggers
        triggers = await retraining_system.check_retraining_triggers("medical-diagnosis-v1")
        print(f"Found {len(triggers)} triggers")
        
        # Trigger retraining for each trigger
        for trigger in triggers:
            job_id = await retraining_system.trigger_retraining(trigger)
            if job_id:
                print(f"Triggered retraining job: {job_id}")
        
        # Get pipeline status
        status = await retraining_system.get_pipeline_status()
        print(f"Pipeline status: {status['pipeline_stats']}")
        
        # Wait for training to complete
        print("Waiting for training to complete...")
        await asyncio.sleep(15)
        
        # Get updated status
        final_status = await retraining_system.get_pipeline_status()
        print(f"Final status: {final_status}")
    
    # Run example
    asyncio.run(run_retraining_example())