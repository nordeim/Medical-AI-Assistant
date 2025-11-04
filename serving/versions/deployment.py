"""
Model rollout and rollback mechanisms with automated health checks.

Provides comprehensive deployment management for medical AI models
with safety controls, monitoring, and automated rollback capabilities.
"""

import json
import time
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import threading
import requests

from .core import ModelVersion, ComplianceLevel, VersionStatus, VersionManager
from .testing import ABTestingManager

logger = logging.getLogger(__name__)


class DeploymentType(Enum):
    """Types of deployment strategies."""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    GRADUAL_ROLLOUT = "gradual_rollout"
    IMMEDIATE = "immediate"


class DeploymentStatus(Enum):
    """Status of deployment operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class HealthCheckStatus(Enum):
    """Health check result status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckConfig:
    """Configuration for health checks."""
    name: str
    check_type: str  # latency, accuracy, error_rate, custom
    threshold: float
    window_minutes: int = 5
    failure_threshold: int = 3
    success_threshold: int = 2
    timeout_seconds: int = 30
    enabled: bool = True
    severity: str = "error"  # error, warning, critical
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    config_name: str
    check_type: str
    status: HealthCheckStatus
    value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class DeploymentTarget:
    """Target for model deployment."""
    name: str
    url: str
    environment: str  # dev, staging, production
    capacity: int = 100
    current_load: int = 0
    health_endpoint: str = "/health"
    status_endpoint: str = "/status"
    deployment_id: Optional[str] = None
    last_health_check: Optional[datetime] = None
    health_status: HealthCheckStatus = HealthCheckStatus.UNKNOWN
    
    def get_full_health_url(self) -> str:
        return f"{self.url.rstrip('/')}{self.health_endpoint}"
    
    def get_full_status_url(self) -> str:
        return f"{self.url.rstrip('/')}{self.status_endpoint}"


@dataclass
class DeploymentOperation:
    """Deployment operation tracking."""
    deployment_id: str
    model_name: str
    model_version: str
    deployment_type: DeploymentType
    status: DeploymentStatus
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    
    # Deployment configuration
    targets: List[DeploymentTarget] = field(default_factory=list)
    rollout_percentage: float = 100.0
    health_checks: List[HealthCheckConfig] = field(default_factory=list)
    rollback_threshold: float = 0.95  # Health score below which to rollback
    
    # Progress tracking
    current_step: str = "initializing"
    steps_completed: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    # Health monitoring
    health_check_results: List[HealthCheckResult] = field(default_factory=list)
    recent_health_scores: List[float] = field(default_factory=list)
    alert_triggered: bool = False
    auto_rollback_enabled: bool = True
    
    # Rollback tracking
    rollback_reason: str = ""
    rollback_completed: bool = False
    
    # Compliance and audit
    initiated_by: str = ""
    compliance_checks_passed: bool = False
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_audit_entry(self, action: str, user: str, details: Dict[str, Any] = None):
        """Add entry to deployment audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details or {}
        }
        self.audit_trail.append(entry)


@dataclass
class RollbackOperation:
    """Rollback operation tracking."""
    rollback_id: str
    original_deployment_id: str
    model_name: str
    target_version: str
    rollback_type: str = "manual"  # manual, automatic, scheduled
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Rollback configuration
    targets: List[DeploymentTarget] = field(default_factory=list)
    health_checks: List[HealthCheckConfig] = field(default_factory=list)
    
    # Reason and justification
    reason: str = ""
    triggered_by: str = ""
    triggered_at: Optional[datetime] = None
    
    # Progress tracking
    steps_completed: List[str] = field(default_factory=list)
    progress_percentage: float = 0.0
    
    # Audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


class RolloutManager:
    """Manager for model rollout operations."""
    
    def __init__(self, version_manager: VersionManager, ab_testing: ABTestingManager = None):
        self.version_manager = version_manager
        self.ab_testing = ab_testing
        self.active_deployments: Dict[str, DeploymentOperation] = {}
        self.rollback_history: Dict[str, List[RollbackOperation]] = {}
        self.health_check_monitor = None
        self.monitoring_active = False
        
        # Deployment configurations
        self.default_health_checks = [
            HealthCheckConfig("latency", "latency", 1000.0, window_minutes=5, failure_threshold=3),
            HealthCheckConfig("error_rate", "error_rate", 0.05, window_minutes=5, failure_threshold=2),
            HealthCheckConfig("accuracy", "accuracy", 0.85, window_minutes=10, failure_threshold=3)
        ]
    
    def deploy_model(self, 
                    model_name: str, 
                    version: str, 
                    targets: List[DeploymentTarget],
                    deployment_type: DeploymentType = DeploymentType.CANARY,
                    rollout_percentage: float = 10.0,
                    health_checks: List[HealthCheckConfig] = None,
                    initiated_by: str = "system") -> Optional[str]:
        """Deploy model to targets."""
        
        # Create deployment operation
        deployment_id = str(uuid.uuid4())
        
        # Get model version
        version_obj = self.version_manager.registry.get_version(model_name, version)
        if not version_obj:
            logger.error(f"Model version {version} not found for {model_name}")
            return None
        
        # Validate compliance requirements
        compliance_check = self._validate_compliance_for_deployment(version_obj, targets)
        if not compliance_check["compliant"]:
            logger.error(f"Compliance validation failed: {compliance_check['issues']}")
            return None
        
        # Create deployment operation
        deployment = DeploymentOperation(
            deployment_id=deployment_id,
            model_name=model_name,
            model_version=version,
            deployment_type=deployment_type,
            status=DeploymentStatus.PENDING,
            targets=targets,
            rollout_percentage=rollout_percentage,
            health_checks=health_checks or self.default_health_checks,
            initiated_by=initiated_by,
            compliance_checks_passed=True
        )
        
        # Add audit entry
        deployment.add_audit_entry("deployment_created", initiated_by, {
            "model_name": model_name,
            "version": version,
            "deployment_type": deployment_type.value,
            "targets": [t.name for t in targets],
            "rollout_percentage": rollout_percentage
        })
        
        # Store deployment
        self.active_deployments[deployment_id] = deployment
        
        # Start deployment
        return self._start_deployment(deployment)
    
    def _start_deployment(self, deployment: DeploymentOperation) -> str:
        """Start deployment execution."""
        deployment.status = DeploymentStatus.IN_PROGRESS
        deployment.started_at = datetime.now()
        
        # Determine deployment strategy
        if deployment.deployment_type == DeploymentType.BLUE_GREEN:
            return self._execute_blue_green_deployment(deployment)
        elif deployment.deployment_type == DeploymentType.ROLLING:
            return self._execute_rolling_deployment(deployment)
        elif deployment.deployment_type == DeploymentType.CANARY:
            return self._execute_canary_deployment(deployment)
        elif deployment.deployment_type == DeploymentType.GRADUAL_ROLLOUT:
            return self._execute_gradual_rollout(deployment)
        else:
            return self._execute_immediate_deployment(deployment)
    
    def _execute_blue_green_deployment(self, deployment: DeploymentOperation) -> str:
        """Execute blue-green deployment strategy."""
        def deployment_thread():
            try:
                deployment.current_step = "deploying_to_green_environment"
                deployment.add_audit_entry("deployment_started", "system")
                
                # Deploy to green environment (new version)
                green_targets = [t for t in deployment.targets if "green" in t.name.lower()]
                
                # Simulate deployment
                for target in green_targets:
                    deployment.current_step = f"deploying_to_{target.name}"
                    self._deploy_to_target(deployment, target)
                    time.sleep(1)  # Simulate deployment time
                
                deployment.current_step = "validating_green_environment"
                deployment.steps_completed.append("green_deployment")
                
                # Validate green environment with health checks
                if self._run_health_checks(deployment, green_targets):
                    deployment.current_step = "switching_traffic"
                    
                    # Switch traffic from blue to green
                    for blue_target in [t for t in deployment.targets if "blue" in t.name.lower()]:
                        self._switch_traffic(blue_target, deployment)
                    
                    deployment.current_step = "deployment_completed"
                    deployment.status = DeploymentStatus.SUCCESS
                    deployment.completed_at = datetime.now()
                    deployment.progress_percentage = 100.0
                    
                    deployment.add_audit_entry("deployment_completed", "system")
                    logger.info(f"Blue-green deployment completed: {deployment.deployment_id}")
                    
                else:
                    # Health check failed - rollback to blue
                    deployment.current_step = "health_check_failed_rolling_back"
                    self._execute_rollback(deployment, "Health check failed")
                    return
                
            except Exception as e:
                logger.error(f"Blue-green deployment failed: {e}")
                deployment.status = DeploymentStatus.FAILED
                deployment.add_audit_entry("deployment_failed", "system", {"error": str(e)})
        
        # Start deployment thread
        thread = threading.Thread(target=deployment_thread)
        thread.daemon = True
        thread.start()
        
        # Start health monitoring
        self._start_health_monitoring(deployment)
        
        return deployment.deployment_id
    
    def _execute_canary_deployment(self, deployment: DeploymentOperation) -> str:
        """Execute canary deployment strategy."""
        def deployment_thread():
            try:
                deployment.current_step = "deploying_canary_version"
                deployment.add_audit_entry("deployment_started", "system")
                
                # Deploy to subset of targets (canary)
                canary_targets = deployment.targets[:int(len(deployment.targets) * 0.1)]  # 10% canary
                
                for target in canary_targets:
                    self._deploy_to_target(deployment, target)
                    time.sleep(0.5)
                
                deployment.current_step = "monitoring_canary"
                deployment.steps_completed.append("canary_deployment")
                
                # Monitor canary for health
                canary_monitor_duration = 60 * 10  # 10 minutes
                monitor_start = time.time()
                
                while time.time() - monitor_start < canary_monitor_duration:
                    if self._run_health_checks(deployment, canary_targets):
                        time.sleep(30)  # Check every 30 seconds
                    else:
                        # Health check failed
                        deployment.current_step = "canary_health_check_failed"
                        self._execute_rollback(deployment, "Canary health check failed")
                        return
                
                # Canary passed - expand rollout
                deployment.current_step = "expanding_rollout"
                remaining_targets = [t for t in deployment.targets if t not in canary_targets]
                
                for target in remaining_targets:
                    self._deploy_to_target(deployment, target)
                    time.sleep(0.5)
                
                deployment.current_step = "deployment_completed"
                deployment.status = DeploymentStatus.SUCCESS
                deployment.completed_at = datetime.now()
                deployment.progress_percentage = 100.0
                
                deployment.add_audit_entry("deployment_completed", "system")
                logger.info(f"Canary deployment completed: {deployment.deployment_id}")
                
            except Exception as e:
                logger.error(f"Canary deployment failed: {e}")
                deployment.status = DeploymentStatus.FAILED
                deployment.add_audit_entry("deployment_failed", "system", {"error": str(e)})
        
        thread = threading.Thread(target=deployment_thread)
        thread.daemon = True
        thread.start()
        
        self._start_health_monitoring(deployment)
        return deployment.deployment_id
    
    def _execute_rolling_deployment(self, deployment: DeploymentOperation) -> str:
        """Execute rolling deployment strategy."""
        def deployment_thread():
            try:
                deployment.add_audit_entry("deployment_started", "system")
                
                target_count = len(deployment.targets)
                batch_size = max(1, target_count // 4)  # 4 batches
                
                for i in range(0, target_count, batch_size):
                    batch = deployment.targets[i:i + batch_size]
                    deployment.current_step = f"deploying_batch_{i//batch_size + 1}"
                    
                    # Deploy batch
                    for target in batch:
                        self._deploy_to_target(deployment, target)
                        time.sleep(0.5)
                    
                    deployment.progress_percentage = ((i + batch_size) / target_count) * 100
                    deployment.steps_completed.append(f"batch_{i//batch_size + 1}")
                    
                    # Health check batch
                    if not self._run_health_checks(deployment, batch):
                        deployment.current_step = f"batch_{i//batch_size + 1}_health_check_failed"
                        self._execute_rollback(deployment, f"Batch {i//batch_size + 1} health check failed")
                        return
                    
                    # Brief pause between batches
                    time.sleep(2)
                
                deployment.current_step = "deployment_completed"
                deployment.status = DeploymentStatus.SUCCESS
                deployment.completed_at = datetime.now()
                deployment.progress_percentage = 100.0
                
                deployment.add_audit_entry("deployment_completed", "system")
                logger.info(f"Rolling deployment completed: {deployment.deployment_id}")
                
            except Exception as e:
                logger.error(f"Rolling deployment failed: {e}")
                deployment.status = DeploymentStatus.FAILED
                deployment.add_audit_entry("deployment_failed", "system", {"error": str(e)})
        
        thread = threading.Thread(target=deployment_thread)
        thread.daemon = True
        thread.start()
        
        self._start_health_monitoring(deployment)
        return deployment.deployment_id
    
    def _execute_gradual_rollout(self, deployment: DeploymentOperation) -> str:
        """Execute gradual rollout strategy."""
        def deployment_thread():
            try:
                deployment.add_audit_entry("deployment_started", "system")
                
                rollout_stages = [5, 10, 25, 50, 100]  # Gradual rollout percentages
                
                for stage_percentage in rollout_stages:
                    deployment.current_step = f"rollout_stage_{stage_percentage}%"
                    
                    # Calculate number of targets for this stage
                    stage_targets = deployment.targets[:int(len(deployment.targets) * stage_percentage / 100)]
                    previous_stage_targets = deployment.targets[:int(len(deployment.targets) * (stage_percentage - 5) / 100)]
                    
                    # Deploy new targets
                    new_targets = [t for t in stage_targets if t not in previous_stage_targets]
                    for target in new_targets:
                        self._deploy_to_target(deployment, target)
                    
                    deployment.rollout_percentage = stage_percentage
                    deployment.progress_percentage = stage_percentage
                    deployment.steps_completed.append(f"stage_{stage_percentage}%")
                    
                    # Monitor for health
                    if not self._run_health_checks(deployment, stage_targets):
                        deployment.current_step = f"stage_{stage_percentage}%_health_check_failed"
                        self._execute_rollback(deployment, f"Stage {stage_percentage}% health check failed")
                        return
                    
                    # Wait before next stage
                    time.sleep(30)
                
                deployment.current_step = "deployment_completed"
                deployment.status = DeploymentStatus.SUCCESS
                deployment.completed_at = datetime.now()
                deployment.progress_percentage = 100.0
                
                deployment.add_audit_entry("deployment_completed", "system")
                logger.info(f"Gradual rollout completed: {deployment.deployment_id}")
                
            except Exception as e:
                logger.error(f"Gradual rollout failed: {e}")
                deployment.status = DeploymentStatus.FAILED
                deployment.add_audit_entry("deployment_failed", "system", {"error": str(e)})
        
        thread = threading.Thread(target=deployment_thread)
        thread.daemon = True
        thread.start()
        
        self._start_health_monitoring(deployment)
        return deployment.deployment_id
    
    def _execute_immediate_deployment(self, deployment: DeploymentOperation) -> str:
        """Execute immediate deployment strategy."""
        def deployment_thread():
            try:
                deployment.add_audit_entry("deployment_started", "system")
                
                # Deploy to all targets immediately
                for i, target in enumerate(deployment.targets):
                    deployment.current_step = f"deploying_to_{target.name}"
                    self._deploy_to_target(deployment, target)
                    deployment.progress_percentage = ((i + 1) / len(deployment.targets)) * 100
                    time.sleep(0.2)
                
                # Validate all targets
                deployment.current_step = "validating_deployment"
                if self._run_health_checks(deployment, deployment.targets):
                    deployment.current_step = "deployment_completed"
                    deployment.status = DeploymentStatus.SUCCESS
                    deployment.completed_at = datetime.now()
                    deployment.progress_percentage = 100.0
                    
                    deployment.add_audit_entry("deployment_completed", "system")
                    logger.info(f"Immediate deployment completed: {deployment.deployment_id}")
                else:
                    deployment.current_step = "health_check_failed"
                    self._execute_rollback(deployment, "Health check failed")
                
            except Exception as e:
                logger.error(f"Immediate deployment failed: {e}")
                deployment.status = DeploymentStatus.FAILED
                deployment.add_audit_entry("deployment_failed", "system", {"error": str(e)})
        
        thread = threading.Thread(target=deployment_thread)
        thread.daemon = True
        thread.start()
        
        self._start_health_monitoring(deployment)
        return deployment.deployment_id
    
    def _deploy_to_target(self, deployment: DeploymentOperation, target: DeploymentTarget):
        """Deploy model to specific target."""
        # Simulate deployment (in practice, would use deployment API)
        logger.info(f"Deploying {deployment.model_name} v{deployment.model_version} to {target.name}")
        
        # Update target status
        target.deployment_id = deployment.deployment_id
        target.health_status = HealthCheckStatus.HEALTHY
        
        # Add audit entry
        deployment.add_audit_entry("target_deployed", "system", {
            "target": target.name,
            "environment": target.environment
        })
    
    def _switch_traffic(self, target: DeploymentTarget, deployment: DeploymentOperation):
        """Switch traffic from blue to green environment."""
        # Simulate traffic switching
        logger.info(f"Switching traffic to {target.name}")
        target.health_status = HealthCheckStatus.HEALTHY
        
        deployment.add_audit_entry("traffic_switched", "system", {
            "target": target.name,
            "version": deployment.model_version
        })
    
    def _run_health_checks(self, deployment: DeploymentOperation, targets: List[DeploymentTarget]) -> bool:
        """Run health checks on deployment targets."""
        all_healthy = True
        health_results = []
        
        for target in targets:
            for check_config in deployment.health_checks:
                if not check_config.enabled:
                    continue
                
                result = self._perform_health_check(target, check_config)
                health_results.append(result)
                deployment.health_check_results.append(result)
                
                # Update target health status
                if result.status in [HealthCheckStatus.UNHEALTHY, HealthCheckStatus.CRITICAL]:
                    all_healthy = False
                    target.health_status = result.status
                elif result.status == HealthCheckStatus.WARNING and target.health_status == HealthCheckStatus.UNKNOWN:
                    target.health_status = HealthCheckStatus.WARNING
                elif result.status == HealthCheckStatus.HEALTHY and target.health_status == HealthCheckStatus.UNKNOWN:
                    target.health_status = HealthCheckStatus.HEALTHY
                
                target.last_health_check = datetime.now()
        
        # Calculate overall health score
        health_score = self._calculate_health_score(health_results)
        deployment.recent_health_scores.append(health_score)
        
        # Check if rollback threshold is reached
        if health_score < deployment.rollback_threshold and len(deployment.recent_health_scores) >= 3:
            deployment.alert_triggered = True
            if deployment.auto_rollback_enabled:
                self._execute_rollback(deployment, f"Health score {health_score:.2f} below threshold {deployment.rollback_threshold}")
        
        return all_healthy and health_score >= deployment.rollback_threshold
    
    def _perform_health_check(self, target: DeploymentTarget, config: HealthCheckConfig) -> HealthCheckResult:
        """Perform individual health check."""
        try:
            if config.check_type == "latency":
                return self._check_latency(target, config)
            elif config.check_type == "error_rate":
                return self._check_error_rate(target, config)
            elif config.check_type == "accuracy":
                return self._check_accuracy(target, config)
            else:
                return HealthCheckResult(
                    config_name=config.name,
                    check_type=config.check_type,
                    status=HealthCheckStatus.UNKNOWN,
                    value=0.0,
                    threshold=config.threshold,
                    message=f"Unknown check type: {config.check_type}"
                )
        except Exception as e:
            return HealthCheckResult(
                config_name=config.name,
                check_type=config.check_type,
                status=HealthCheckStatus.CRITICAL,
                value=0.0,
                threshold=config.threshold,
                message=f"Health check failed: {str(e)}"
            )
    
    def _check_latency(self, target: DeploymentTarget, config: HealthCheckConfig) -> HealthCheckResult:
        """Check response latency."""
        try:
            start_time = time.time()
            response = requests.get(target.get_full_health_url(), timeout=config.timeout_seconds)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            status = HealthCheckStatus.HEALTHY
            if latency > config.threshold * 2:
                status = HealthCheckStatus.CRITICAL
            elif latency > config.threshold:
                status = HealthCheckStatus.WARNING
            
            message = f"Latency: {latency:.2f}ms"
            
            return HealthCheckResult(
                config_name=config.name,
                check_type="latency",
                status=status,
                value=latency,
                threshold=config.threshold,
                message=message,
                details={"response_code": response.status_code}
            )
        except Exception as e:
            return HealthCheckResult(
                config_name=config.name,
                check_type="latency",
                status=HealthCheckStatus.CRITICAL,
                value=0.0,
                threshold=config.threshold,
                message=f"Latency check failed: {str(e)}"
            )
    
    def _check_error_rate(self, target: DeploymentTarget, config: HealthCheckConfig) -> HealthCheckResult:
        """Check error rate."""
        # Simplified error rate check (would need actual metrics in practice)
        import random
        random.seed(int(time.time() / 60))  # Seed changes every minute for variation
        
        error_rate = random.uniform(0.0, 0.1)  # Mock error rate between 0% and 10%
        
        status = HealthCheckStatus.HEALTHY
        if error_rate > config.threshold * 2:
            status = HealthCheckStatus.CRITICAL
        elif error_rate > config.threshold:
            status = HealthCheckStatus.WARNING
        
        message = f"Error rate: {error_rate:.1%}"
        
        return HealthCheckResult(
            config_name=config.name,
            check_type="error_rate",
            status=status,
            value=error_rate,
            threshold=config.threshold,
            message=message
        )
    
    def _check_accuracy(self, target: DeploymentTarget, config: HealthCheckConfig) -> HealthCheckResult:
        """Check model accuracy."""
        # Mock accuracy check (would need actual model performance metrics)
        import random
        random.seed(int(time.time() / 300))  # Seed changes every 5 minutes
        
        accuracy = random.uniform(0.7, 0.95)  # Mock accuracy between 70% and 95%
        
        status = HealthCheckStatus.HEALTHY
        if accuracy < config.threshold * 0.8:
            status = HealthCheckStatus.CRITICAL
        elif accuracy < config.threshold:
            status = HealthCheckStatus.WARNING
        
        message = f"Accuracy: {accuracy:.1%}"
        
        return HealthCheckResult(
            config_name=config.name,
            check_type="accuracy",
            status=status,
            value=accuracy,
            threshold=config.threshold,
            message=message
        )
    
    def _calculate_health_score(self, health_results: List[HealthCheckResult]) -> float:
        """Calculate overall health score from health check results."""
        if not health_results:
            return 1.0
        
        scores = []
        for result in health_results:
            if result.status == HealthCheckStatus.HEALTHY:
                scores.append(1.0)
            elif result.status == HealthCheckStatus.WARNING:
                scores.append(0.7)
            elif result.status == HealthCheckStatus.CRITICAL:
                scores.append(0.0)
            else:
                scores.append(0.5)
        
        return sum(scores) / len(scores)
    
    def _start_health_monitoring(self, deployment: DeploymentOperation):
        """Start continuous health monitoring for deployment."""
        if self.health_check_monitor is None:
            self.health_check_monitor = threading.Thread(target=self._health_monitoring_loop)
            self.health_check_monitor.daemon = True
            self.health_check_monitor.start()
        
        self.monitoring_active = True
    
    def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for deployment in self.active_deployments.values():
                    if deployment.status in [DeploymentStatus.IN_PROGRESS, DeploymentStatus.HEALTHY]:
                        # Check if it's time for health checks
                        needs_check = False
                        for target in deployment.targets:
                            if (not target.last_health_check or 
                                (current_time - target.last_health_check).total_seconds() > 300):  # 5 minutes
                                needs_check = True
                                break
                        
                        if needs_check:
                            self._run_health_checks(deployment, deployment.targets)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(60)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentOperation]:
        """Get status of deployment operation."""
        return self.active_deployments.get(deployment_id)
    
    def cancel_deployment(self, deployment_id: str, reason: str = "") -> bool:
        """Cancel deployment operation."""
        if deployment_id not in self.active_deployments:
            return False
        
        deployment = self.active_deployments[deployment_id]
        if deployment.status in [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK]:
            return False
        
        deployment.status = DeploymentStatus.CANCELLED
        deployment.completed_at = datetime.now()
        deployment.add_audit_entry("deployment_cancelled", "system", {"reason": reason})
        
        logger.info(f"Deployment cancelled: {deployment_id}")
        return True


class RollbackManager:
    """Manager for rollback operations."""
    
    def __init__(self, rollout_manager: RolloutManager, version_manager: VersionManager):
        self.rollout_manager = rollout_manager
        self.version_manager = version_manager
        self.rollback_history: Dict[str, List[RollbackOperation]] = {}
    
    def rollback_deployment(self, deployment_id: str, target_version: str = None, 
                          reason: str = "", triggered_by: str = "system") -> Optional[str]:
        """Rollback deployment to previous version."""
        
        if deployment_id not in self.rollout_manager.active_deployments:
            logger.error(f"Deployment {deployment_id} not found")
            return None
        
        deployment = self.rollout_manager.active_deployments[deployment_id]
        
        # Determine target version for rollback
        if not target_version:
            target_version = self._find_rollback_version(deployment)
        
        if not target_version:
            logger.error("No suitable rollback version found")
            return None
        
        # Create rollback operation
        rollback_id = str(uuid.uuid4())
        rollback = RollbackOperation(
            rollback_id=rollback_id,
            original_deployment_id=deployment_id,
            model_name=deployment.model_name,
            target_version=target_version,
            reason=reason or "Manual rollback",
            triggered_by=triggered_by,
            targets=deployment.targets.copy(),
            health_checks=deployment.health_checks.copy()
        )
        
        rollback.triggered_at = datetime.now()
        rollback.add_audit_entry("rollback_initiated", triggered_by, {
            "original_deployment": deployment_id,
            "target_version": target_version,
            "reason": reason
        })
        
        # Store rollback
        if deployment_id not in self.rollback_history:
            self.rollback_history[deployment_id] = []
        self.rollback_history[deployment_id].append(rollback)
        
        # Execute rollback
        return self._execute_rollback_operation(rollback)
    
    def _find_rollback_version(self, deployment: DeploymentOperation) -> Optional[str]:
        """Find appropriate rollback version."""
        # Get latest version in registry that is not the current deployment version
        version_obj = self.version_manager.registry.get_version(deployment.model_name)
        if not version_obj:
            return None
        
        # For simplicity, rollback to previous version
        versions = self.version_manager.registry.list_versions(deployment.model_name)
        current_index = versions.index(deployment.model_version) if deployment.model_version in versions else -1
        
        if current_index > 0:
            return versions[current_index - 1]
        elif len(versions) > 1:
            return versions[1]  # Second latest version
        
        return None
    
    def _execute_rollback_operation(self, rollback: RollbackOperation) -> str:
        """Execute rollback operation."""
        def rollback_thread():
            try:
                rollback.status = DeploymentStatus.IN_PROGRESS
                rollback.started_at = datetime.now()
                
                # Stop accepting traffic to failed deployment
                deployment = self.rollout_manager.active_deployments[rollback.original_deployment_id]
                deployment.status = DeploymentStatus.CRITICAL
                deployment.rollback_reason = rollback.reason
                
                rollback.current_step = "stopping_traffic_to_failed_deployment"
                
                # Deploy rollback version to all targets
                for i, target in enumerate(rollback.targets):
                    rollback.current_step = f"rolling_back_to_{target.name}"
                    
                    # In practice, would deploy actual rollback version
                    logger.info(f"Rolling back {rollback.model_name} to v{rollback.target_version} on {target.name}")
                    
                    target.health_status = HealthCheckStatus.HEALTHY
                    rollback.steps_completed.append(f"rollback_target_{i+1}")
                    rollback.progress_percentage = ((i + 1) / len(rollback.targets)) * 100
                    
                    time.sleep(0.5)
                
                # Validate rollback with health checks
                rollback.current_step = "validating_rollback"
                if self._validate_rollback(rollback):
                    rollback.current_step = "rollback_completed"
                    rollback.status = DeploymentStatus.SUCCESS
                    rollback.completed_at = datetime.now()
                    rollback.rollback_completed = True
                    
                    # Update deployment status
                    deployment.status = DeploymentStatus.ROLLED_BACK
                    deployment.completed_at = datetime.now()
                    
                    rollback.add_audit_entry("rollback_completed", "system")
                    logger.info(f"Rollback completed: {rollback.rollback_id}")
                else:
                    rollback.current_step = "rollback_validation_failed"
                    rollback.status = DeploymentStatus.FAILED
                    rollback.add_audit_entry("rollback_failed", "system")
                
            except Exception as e:
                logger.error(f"Rollback operation failed: {e}")
                rollback.status = DeploymentStatus.FAILED
                rollback.add_audit_entry("rollback_failed", "system", {"error": str(e)})
        
        thread = threading.Thread(target=rollback_thread)
        thread.daemon = True
        thread.start()
        
        return rollback.rollback_id
    
    def _validate_rollback(self, rollback: RollbackOperation) -> bool:
        """Validate rollback with health checks."""
        # Run health checks on rollback targets
        health_results = []
        
        for target in rollback.targets:
            for check_config in rollback.health_checks:
                result = self.rollout_manager._perform_health_check(target, check_config)
                health_results.append(result)
        
        # Calculate health score
        health_score = self.rollout_manager._calculate_health_score(health_results)
        
        # Rollback is successful if health score is above threshold
        return health_score >= 0.9  # 90% health score threshold for rollback
    
    def get_rollback_history(self, deployment_id: str) -> List[RollbackOperation]:
        """Get rollback history for deployment."""
        return self.rollback_history.get(deployment_id, [])
    
    def schedule_rollback(self, deployment_id: str, target_version: str, 
                        schedule_time: datetime, reason: str) -> bool:
        """Schedule automatic rollback."""
        # Simplified scheduling - in practice would use proper scheduling system
        logger.info(f"Scheduled rollback of {deployment_id} to v{target_version} at {schedule_time}")
        return True
    
    def _validate_compliance_for_deployment(self, version: ModelVersion, targets: List[DeploymentTarget]) -> Dict[str, Any]:
        """Validate compliance requirements for deployment."""
        issues = []
        
        # Check compliance level requirements
        if version.compliance.compliance_level == ComplianceLevel.PRODUCTION:
            # Production deployment requires clinical approval
            if not version.compliance.clinical_approval_date:
                issues.append("Production deployment requires clinical approval date")
            
            # Check target environment
            for target in targets:
                if target.environment not in ["staging", "production"]:
                    issues.append(f"Cannot deploy production model to {target.environment} environment")
        
        return {
            "compliant": len(issues) == 0,
            "issues": issues
        }