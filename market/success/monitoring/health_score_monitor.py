"""
Health Score Monitoring and Intervention Workflows
Real-time customer health monitoring with predictive analytics for healthcare AI clients
"""

import asyncio
import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from config.framework_config import HealthcareKPI, HealthScoreStatus, InterventionLevel

class MetricType(Enum):
    USAGE = "usage"
    CLINICAL = "clinical"
    FINANCIAL = "financial"
    SATISFACTION = "satisfaction"
    SUPPORT = "support"
    ENGAGEMENT = "engagement"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class InterventionType(Enum):
    AUTOMATED = "automated"
    HUMAN = "human"
    ESCALATION = "escalation"
    EMERGENCY = "emergency"

@dataclass
class HealthMetric:
    """Individual health metric for customer"""
    metric_id: str
    customer_id: str
    metric_type: MetricType
    metric_name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    weight: float
    last_updated: datetime.datetime
    data_source: str
    measurement_frequency: str  # daily, weekly, monthly
    historical_data: List[Tuple[datetime.datetime, float]] = field(default_factory=list)

@dataclass
class HealthAlert:
    """Health alert generated from metric monitoring"""
    alert_id: str
    customer_id: str
    metric_id: str
    severity: AlertSeverity
    message: str
    description: str
    created_at: datetime.datetime
    acknowledged: bool = False
    acknowledged_by: str = ""
    acknowledged_at: Optional[datetime.datetime] = None
    resolved: bool = False
    resolved_by: str = ""
    resolved_at: Optional[datetime.datetime] = None
    intervention_required: bool = False
    intervention_type: Optional[InterventionType] = None

@dataclass
class InterventionWorkflow:
    """Intervention workflow for health issues"""
    workflow_id: str
    customer_id: str
    trigger_metric: str
    intervention_level: InterventionLevel
    title: str
    description: str
    priority: str  # P1, P2, P3, P4
    assigned_to: str
    status: str  # created, in_progress, completed, cancelled
    created_at: datetime.datetime
    due_date: datetime.datetime
    completed_at: Optional[datetime.datetime] = None
    actions_taken: List[str] = field(default_factory=list)
    outcome: str = ""
    next_steps: List[str] = field(default_factory=list)

@dataclass
class PredictiveInsight:
    """Predictive insight about customer health"""
    insight_id: str
    customer_id: str
    prediction_type: str  # churn_risk, expansion_opportunity, health_decline, etc.
    probability: float
    timeframe: str
    confidence: float
    factors: List[str]
    recommendations: List[str]
    created_at: datetime.datetime
    status: str = "active"  # active, expired, acted_upon

class HealthScoreMonitor:
    """Real-time health score monitoring system for healthcare customers"""
    
    def __init__(self):
        self.metrics: Dict[str, HealthMetric] = {}
        self.alerts: Dict[str, HealthAlert] = {}
        self.interventions: Dict[str, InterventionWorkflow] = {}
        self.predictive_insights: Dict[str, PredictiveInsight] = {}
        self.monitoring_rules: Dict[str, Callable] = {}
        self.escalation_policies: Dict[InterventionLevel, Dict] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitoring rules and policies
        self._initialize_monitoring_rules()
        self._initialize_escalation_policies()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _initialize_monitoring_rules(self):
        """Initialize monitoring rules for different metric types"""
        self.monitoring_rules = {
            "usage_decline": self._check_usage_decline,
            "clinical_performance_drop": self._check_clinical_performance,
            "support_volume_spike": self._check_support_volume,
            "nps_decline": self._check_nps_decline,
            "engagement_drop": self._check_engagement_drop,
            "financial_risk": self._check_financial_risk,
            "competitive_threat": self._check_competitive_threat
        }
    
    def _initialize_escalation_policies(self):
        """Initialize escalation policies for different intervention levels"""
        self.escalation_policies = {
            InterventionLevel.INFO: {
                "notification_recipients": ["csm"],
                "response_time_hours": 24,
                "follow_up_frequency": "weekly",
                "escalation_after_days": 7
            },
            InterventionLevel.WARNING: {
                "notification_recipients": ["csm", "team_lead"],
                "response_time_hours": 4,
                "follow_up_frequency": "every_2_days",
                "escalation_after_days": 3
            },
            InterventionLevel.CRITICAL: {
                "notification_recipients": ["csm", "team_lead", "director"],
                "response_time_hours": 1,
                "follow_up_frequency": "daily",
                "escalation_after_days": 1
            },
            InterventionLevel.EMERGENCY: {
                "notification_recipients": ["csm", "team_lead", "director", "vp"],
                "response_time_hours": 0.5,
                "follow_up_frequency": "twice_daily",
                "escalation_after_hours": 4
            }
        }
    
    def _start_background_monitoring(self):
        """Start background monitoring tasks"""
        # In a real implementation, this would start background threads/tasks
        self.logger.info("Started background health monitoring")
    
    def register_metric(self, metric: HealthMetric) -> bool:
        """Register a health metric for monitoring"""
        try:
            self.metrics[metric.metric_id] = metric
            self.logger.info(f"Registered metric {metric.metric_name} for customer {metric.customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register metric: {e}")
            return False
    
    def update_metric_value(self, metric_id: str, new_value: float, 
                          timestamp: Optional[datetime.datetime] = None) -> bool:
        """Update metric value and trigger monitoring checks"""
        if metric_id not in self.metrics:
            return False
        
        metric = self.metrics[metric_id]
        
        # Update value
        current_time = timestamp or datetime.datetime.now()
        old_value = metric.current_value
        metric.current_value = new_value
        metric.last_updated = current_time
        
        # Add to historical data
        metric.historical_data.append((current_time, new_value))
        
        # Keep only last 30 data points
        if len(metric.historical_data) > 30:
            metric.historical_data = metric.historical_data[-30:]
        
        # Check for alerts
        self._check_metric_thresholds(metric, old_value)
        
        # Run monitoring rules
        self._run_monitoring_checks(metric)
        
        self.logger.info(f"Updated metric {metric_id} from {old_value} to {new_value}")
        return True
    
    def _check_metric_thresholds(self, metric: HealthMetric, old_value: float):
        """Check if metric crosses threshold levels"""
        
        # Check warning threshold
        if metric.current_value < metric.threshold_warning and old_value >= metric.threshold_warning:
            self._create_alert(
                customer_id=metric.customer_id,
                metric_id=metric.metric_id,
                severity=AlertSeverity.MEDIUM,
                message=f"Warning threshold breached for {metric.metric_name}",
                description=f"Current value {metric.current_value} below warning threshold {metric.threshold_warning}",
                intervention_required=True
            )
        
        # Check critical threshold
        if metric.current_value < metric.threshold_critical and old_value >= metric.threshold_critical:
            self._create_alert(
                customer_id=metric.customer_id,
                metric_id=metric.metric_id,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical threshold breached for {metric.metric_name}",
                description=f"Current value {metric.current_value} below critical threshold {metric.threshold_critical}",
                intervention_required=True
            )
    
    def _create_alert(self, customer_id: str, metric_id: str, severity: AlertSeverity,
                     message: str, description: str, intervention_required: bool = False):
        """Create a new alert"""
        alert = HealthAlert(
            alert_id=f"alert_{customer_id}_{metric_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=customer_id,
            metric_id=metric_id,
            severity=severity,
            message=message,
            description=description,
            created_at=datetime.datetime.now(),
            intervention_required=intervention_required
        )
        
        # Set intervention type based on severity
        if intervention_required:
            if severity == AlertSeverity.CRITICAL:
                alert.intervention_type = InterventionType.ESCALATION
            elif severity == AlertSeverity.HIGH:
                alert.intervention_type = InterventionType.HUMAN
            else:
                alert.intervention_type = InterventionType.AUTOMATED
        
        self.alerts[alert.alert_id] = alert
        
        # Trigger intervention workflow if needed
        if intervention_required:
            self._trigger_intervention_workflow(customer_id, metric_id, alert)
        
        self.logger.warning(f"Created {severity.value} alert for customer {customer_id}: {message}")
    
    def _trigger_intervention_workflow(self, customer_id: str, metric_id: str, alert: HealthAlert):
        """Trigger intervention workflow based on alert"""
        
        # Determine intervention level based on alert severity
        if alert.severity == AlertSeverity.CRITICAL:
            intervention_level = InterventionLevel.EMERGENCY
        elif alert.severity == AlertSeverity.HIGH:
            intervention_level = InterventionLevel.CRITICAL
        elif alert.severity == AlertSeverity.MEDIUM:
            intervention_level = InterventionLevel.WARNING
        else:
            intervention_level = InterventionLevel.INFO
        
        # Create intervention workflow
        workflow = InterventionWorkflow(
            workflow_id=f"intervention_{customer_id}_{metric_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=customer_id,
            trigger_metric=metric_id,
            intervention_level=intervention_level,
            title=f"Health Alert Intervention - {alert.message}",
            description=f"Automated intervention triggered by {alert.severity.value} alert",
            priority=self._get_priority_from_intervention_level(intervention_level),
            assigned_to="csm",  # Would be assigned based on escalation policy
            status="created",
            created_at=datetime.datetime.now(),
            due_date=datetime.datetime.now() + datetime.timedelta(
                hours=self.escalation_policies[intervention_level]["response_time_hours"]
            )
        )
        
        self.interventions[workflow.workflow_id] = workflow
        
        self.logger.info(f"Created intervention workflow {workflow.workflow_id} for customer {customer_id}")
    
    def _get_priority_from_intervention_level(self, level: InterventionLevel) -> str:
        """Map intervention level to priority level"""
        priority_map = {
            InterventionLevel.EMERGENCY: "P1",
            InterventionLevel.CRITICAL: "P2",
            InterventionLevel.WARNING: "P3",
            InterventionLevel.INFO: "P4"
        }
        return priority_map.get(level, "P4")
    
    def _run_monitoring_checks(self, metric: HealthMetric):
        """Run all applicable monitoring checks for a metric"""
        for rule_name, rule_func in self.monitoring_rules.items():
            try:
                result = rule_func(metric)
                if result:
                    self._handle_monitoring_rule_result(rule_name, metric, result)
            except Exception as e:
                self.logger.error(f"Error in monitoring rule {rule_name}: {e}")
    
    def _handle_monitoring_rule_result(self, rule_name: str, metric: HealthMetric, result: Dict):
        """Handle the result of a monitoring rule"""
        if result.get("trigger_alert", False):
            self._create_alert(
                customer_id=metric.customer_id,
                metric_id=metric.metric_id,
                severity=result.get("severity", AlertSeverity.MEDIUM),
                message=result.get("message", f"Alert from monitoring rule {rule_name}"),
                description=result.get("description", ""),
                intervention_required=result.get("intervention_required", False)
            )
    
    def _check_usage_decline(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for significant usage decline"""
        if metric.metric_type != MetricType.USAGE or len(metric.historical_data) < 7:
            return None
        
        # Calculate recent trend (last 7 vs previous 7 data points)
        recent_data = metric.historical_data[-7:]
        previous_data = metric.historical_data[-14:-7] if len(metric.historical_data) >= 14 else metric.historical_data[:-7]
        
        if not previous_data:
            return None
        
        recent_avg = sum(point[1] for point in recent_data) / len(recent_data)
        previous_avg = sum(point[1] for point in previous_data) / len(previous_data)
        
        decline_percentage = (previous_avg - recent_avg) / previous_avg if previous_avg > 0 else 0
        
        if decline_percentage > 0.20:  # 20% decline
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.HIGH if decline_percentage > 0.35 else AlertSeverity.MEDIUM,
                "message": f"Significant usage decline detected: {decline_percentage:.1%}",
                "description": f"Usage decreased by {decline_percentage:.1%} over the past period",
                "intervention_required": True
            }
        
        return None
    
    def _check_clinical_performance(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for clinical performance degradation"""
        if metric.metric_type != MetricType.CLINICAL or len(metric.historical_data) < 5:
            return None
        
        # Check if performance is consistently below target
        recent_data = metric.historical_data[-5:]
        below_target_count = sum(1 for point in recent_data if point[1] < metric.target_value)
        
        if below_target_count >= 4:  # 4 out of 5 measurements below target
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.CRITICAL,
                "message": "Clinical performance consistently below target",
                "description": f"{below_target_count} out of {len(recent_data)} recent measurements below target value",
                "intervention_required": True
            }
        
        return None
    
    def _check_support_volume(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for support volume spike"""
        if metric.metric_type != MetricType.SUPPORT or len(metric.historical_data) < 7:
            return None
        
        # Calculate baseline vs current support volume
        baseline_data = metric.historical_data[:-7] if len(metric.historical_data) > 7 else []
        current_data = metric.historical_data[-7:]
        
        if not baseline_data:
            return None
        
        baseline_avg = sum(point[1] for point in baseline_data) / len(baseline_data)
        current_avg = sum(point[1] for point in current_data) / len(current_data)
        
        increase_percentage = (current_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
        
        if increase_percentage > 0.50:  # 50% increase
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.HIGH if increase_percentage > 1.0 else AlertSeverity.MEDIUM,
                "message": f"Support volume spike: {increase_percentage:.1%} increase",
                "description": f"Support tickets increased by {increase_percentage:.1%} compared to baseline",
                "intervention_required": True
            }
        
        return None
    
    def _check_nps_decline(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for NPS score decline"""
        if metric.metric_type != MetricType.SATISFACTION or len(metric.historical_data) < 3:
            return None
        
        # Check for declining NPS trend
        recent_trend = metric.historical_data[-3:]
        is_declining = all(
            recent_trend[i][1] > recent_trend[i+1][1] 
            for i in range(len(recent_trend)-1)
        )
        
        current_score = metric.current_value
        if is_declining and current_score < 7:  # NPS below 7
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.HIGH,
                "message": "NPS score showing declining trend",
                "description": f"NPS score has been declining and is currently {current_score}",
                "intervention_required": True
            }
        
        return None
    
    def _check_engagement_drop(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for engagement level drop"""
        if metric.metric_type != MetricType.ENGAGEMENT or len(metric.historical_data) < 5:
            return None
        
        # Check for sustained engagement drop
        recent_data = metric.historical_data[-5:]
        low_engagement_count = sum(1 for point in recent_data if point[1] < 0.6)  # 60% threshold
        
        if low_engagement_count >= 3:  # 3 out of 5 measurements show low engagement
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.MEDIUM,
                "message": "Sustained low engagement detected",
                "description": f"{low_engagement_count} out of {len(recent_data)} recent measurements show low engagement",
                "intervention_required": True
            }
        
        return None
    
    def _check_financial_risk(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for financial risk indicators"""
        if metric.metric_type != MetricType.FINANCIAL or len(metric.historical_data) < 3:
            return None
        
        # Check for financial health decline
        recent_data = metric.historical_data[-3:]
        declining_trend = all(
            recent_data[i][1] > recent_data[i+1][1] 
            for i in range(len(recent_data)-1)
        )
        
        if declining_trend and metric.current_value < 0.5:  # Financial health below 50%
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.HIGH,
                "message": "Financial health showing declining trend",
                "description": f"Financial health metrics declining and currently at {metric.current_value:.2f}",
                "intervention_required": True
            }
        
        return None
    
    def _check_competitive_threat(self, metric: HealthMetric) -> Optional[Dict]:
        """Check for competitive threat indicators"""
        # This would typically check for competitive intelligence data
        # For now, we'll simulate based on specific metrics
        
        if metric.metric_name == "competitive_pressure" and metric.current_value > 0.7:
            return {
                "trigger_alert": True,
                "severity": AlertSeverity.HIGH,
                "message": "High competitive pressure detected",
                "description": f"Competitive pressure score at {metric.current_value:.2f}",
                "intervention_required": True
            }
        
        return None
    
    def generate_predictive_insight(self, customer_id: str, prediction_type: str, 
                                  factors: List[str], probability: float,
                                  recommendations: List[str]) -> PredictiveInsight:
        """Generate predictive insight about customer health"""
        insight = PredictiveInsight(
            insight_id=f"insight_{customer_id}_{prediction_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=customer_id,
            prediction_type=prediction_type,
            probability=probability,
            timeframe="90_days",  # Default timeframe
            confidence=0.75,  # Would be calculated based on model accuracy
            factors=factors,
            recommendations=recommendations,
            created_at=datetime.datetime.now()
        )
        
        self.predictive_insights[insight.insight_id] = insight
        self.logger.info(f"Generated predictive insight: {prediction_type} for customer {customer_id}")
        
        return insight
    
    def get_customer_health_dashboard(self, customer_id: str) -> Dict:
        """Get comprehensive health dashboard for a customer"""
        
        # Get customer metrics
        customer_metrics = [
            metric for metric in self.metrics.values() 
            if metric.customer_id == customer_id
        ]
        
        # Get active alerts
        customer_alerts = [
            alert for alert in self.alerts.values()
            if alert.customer_id == customer_id and not alert.resolved
        ]
        
        # Get active interventions
        customer_interventions = [
            workflow for workflow in self.interventions.values()
            if workflow.customer_id == customer_id and workflow.status != "completed"
        ]
        
        # Get recent insights
        customer_insights = [
            insight for insight in self.predictive_insights.values()
            if insight.customer_id == customer_id and insight.status == "active"
        ]
        
        # Calculate overall health score
        health_score = self._calculate_overall_health_score(customer_metrics)
        
        # Generate alerts summary
        alerts_summary = self._generate_alerts_summary(customer_alerts)
        
        return {
            "customer_id": customer_id,
            "overall_health_score": health_score,
            "metrics_summary": {
                "total_metrics": len(customer_metrics),
                "healthy_metrics": len([m for m in customer_metrics if m.current_value >= m.target_value]),
                "warning_metrics": len([m for m in customer_metrics if m.threshold_warning <= m.current_value < m.target_value]),
                "critical_metrics": len([m for m in customer_metrics if m.current_value < m.threshold_critical])
            },
            "alerts_summary": alerts_summary,
            "active_interventions": len(customer_interventions),
            "predictive_insights": len(customer_insights),
            "metric_details": [
                {
                    "metric_id": m.metric_id,
                    "name": m.metric_name,
                    "current_value": m.current_value,
                    "target_value": m.target_value,
                    "status": self._get_metric_status(m),
                    "last_updated": m.last_updated
                }
                for m in customer_metrics
            ],
            "recent_insights": [
                {
                    "insight_id": i.insight_id,
                    "prediction_type": i.prediction_type,
                    "probability": i.probability,
                    "recommendations": i.recommendations[:3]  # Top 3 recommendations
                }
                for i in customer_insights[-5:]  # Last 5 insights
            ],
            "intervention_priority": self._get_intervention_priority(customer_interventions)
        }
    
    def _calculate_overall_health_score(self, metrics: List[HealthMetric]) -> float:
        """Calculate overall health score from metrics"""
        if not metrics:
            return 75.0  # Default score
        
        weighted_scores = []
        total_weight = 0
        
        for metric in metrics:
            # Normalize score to 0-100 scale (assuming metric values are already normalized)
            score = (metric.current_value / metric.target_value * 100) if metric.target_value > 0 else 50
            score = max(0, min(100, score))  # Clamp to 0-100
            
            weighted_scores.append(score * metric.weight)
            total_weight += metric.weight
        
        return sum(weighted_scores) / total_weight if total_weight > 0 else 75.0
    
    def _get_metric_status(self, metric: HealthMetric) -> str:
        """Get status string for metric"""
        if metric.current_value >= metric.target_value:
            return "healthy"
        elif metric.current_value >= metric.threshold_warning:
            return "warning"
        elif metric.current_value >= metric.threshold_critical:
            return "critical"
        else:
            return "critical"
    
    def _generate_alerts_summary(self, alerts: List[HealthAlert]) -> Dict:
        """Generate summary of alerts"""
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        intervention_required = 0
        
        for alert in alerts:
            severity_counts[alert.severity.value] += 1
            if alert.intervention_required:
                intervention_required += 1
        
        return {
            "total_alerts": len(alerts),
            "severity_breakdown": severity_counts,
            "intervention_required": intervention_required,
            "acknowledged_alerts": len([a for a in alerts if a.acknowledged]),
            "unacknowledged_alerts": len([a for a in alerts if not a.acknowledged])
        }
    
    def _get_intervention_priority(self, interventions: List[InterventionWorkflow]) -> str:
        """Get highest intervention priority"""
        if not interventions:
            return "none"
        
        priority_levels = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
        highest_priority = min(interventions, key=lambda x: priority_levels.get(x.priority, 4))
        return highest_priority.priority
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.datetime.now()
        
        self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.datetime.now()
        
        self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True
    
    def update_intervention_status(self, workflow_id: str, status: str, 
                                 actions_taken: List[str] = None) -> bool:
        """Update intervention workflow status"""
        if workflow_id not in self.interventions:
            return False
        
        workflow = self.interventions[workflow_id]
        workflow.status = status
        
        if actions_taken:
            workflow.actions_taken.extend(actions_taken)
        
        if status == "completed":
            workflow.completed_at = datetime.datetime.now()
        
        self.logger.info(f"Updated intervention {workflow_id} to status: {status}")
        return True