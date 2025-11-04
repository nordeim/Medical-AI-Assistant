"""
Advanced Alerting and Notification System

Provides comprehensive alerting with configurable thresholds, escalation policies,
and multi-channel notification for medical AI serving platform.
"""

import asyncio
import json
import smtplib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
import structlog
import aiohttp
import aiofiles

from ...config.logging_config import get_logger

logger = structlog.get_logger("alerting_system")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status tracking."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SMS_TWILIO = "sms_twilio"
    DASHBOARD = "dashboard"
    AUDIT_LOG = "audit_log"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "greater_than", "less_than", "equals", "outside_range"
    threshold_value: float
    threshold_upper: Optional[float] = None  # For range conditions
    time_window: int = 300  # seconds
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True
    
    # Escalation settings
    escalation_enabled: bool = True
    escalation_interval: int = 900  # 15 minutes
    max_escalations: int = 3
    
    # Notification settings
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    notification_cooldown: int = 1800  # 30 minutes
    
    # Medical AI specific settings
    clinical_impact_assessment: bool = True
    regulatory_compliance_check: bool = True
    
    # Metadata
    created_timestamp: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    
    # Alert content
    title: str
    description: str
    message: str
    
    # Metrics and context
    metric_name: str
    current_value: float
    threshold_value: float
    metric_context: Dict[str, Any]
    
    # Timing
    triggered_timestamp: float
    acknowledged_timestamp: Optional[float] = None
    resolved_timestamp: Optional[float] = None
    
    # Escalation tracking
    escalation_count: int = 0
    escalation_timestamp: Optional[float] = None
    escalation_levels: List[Dict[str, Any]] = field(default_factory=list)
    
    # Notifications sent
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    
    # Medical AI context
    clinical_impact_score: float = 0.0
    regulatory_compliance_issue: bool = False
    patient_safety_risk: bool = False
    
    # Audit trail
    created_by: str = "system"
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThresholdManager:
    """Manages dynamic thresholds with adaptive learning."""
    
    def __init__(self, learning_window: int = 86400):  # 24 hours
        self.learning_window = learning_window
        self.baseline_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.threshold_adjustments: Dict[str, float] = {}
        
        self.logger = structlog.get_logger("threshold_manager")
    
    def add_baseline_data(self, metric_name: str, value: float, timestamp: float = None):
        """Add baseline data point for metric."""
        if timestamp is None:
            timestamp = time.time()
        
        self.baseline_data[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
    
    def calculate_adaptive_threshold(self, metric_name: str, base_threshold: float, 
                                   percentile: float = 95.0) -> float:
        """Calculate adaptive threshold based on historical data."""
        
        if metric_name not in self.baseline_data:
            return base_threshold
        
        # Get recent data (within learning window)
        cutoff_time = time.time() - self.learning_window
        recent_data = [
            d['value'] for d in self.baseline_data[metric_name]
            if d['timestamp'] >= cutoff_time
        ]
        
        if len(recent_data) < 50:  # Need minimum data points
            return base_threshold
        
        # Calculate percentile-based threshold
        try:
            import numpy as np
            adaptive_threshold = np.percentile(recent_data, percentile)
            
            # Apply learning adjustment
            if metric_name in self.threshold_adjustments:
                adjustment = self.threshold_adjustments[metric_name]
                adaptive_threshold *= (1.0 + adjustment)
            
            return adaptive_threshold
            
        except Exception as e:
            self.logger.warning("Failed to calculate adaptive threshold", 
                              metric=metric_name, error=str(e))
            return base_threshold
    
    def update_threshold_adjustment(self, metric_name: str, 
                                  alert_frequency: float, 
                                  target_frequency: float = 0.01):
        """Update threshold adjustment based on alert frequency."""
        
        if metric_name not in self.baseline_data:
            return
        
        # Calculate adjustment based on alert frequency
        if alert_frequency > target_frequency * 2:
            # Too many alerts - increase threshold
            adjustment_increase = min(0.1, alert_frequency - target_frequency)
            self.threshold_adjustments[metric_name] = \
                self.threshold_adjustments.get(metric_name, 0.0) + adjustment_increase
        elif alert_frequency < target_frequency * 0.5:
            # Too few alerts - decrease threshold
            adjustment_decrease = min(0.1, target_frequency - alert_frequency)
            self.threshold_adjustments[metric_name] = \
                self.threshold_adjustments.get(metric_name, 0.0) - adjustment_decrease
        
        # Clamp adjustments
        if metric_name in self.threshold_adjustments:
            self.threshold_adjustments[metric_name] = max(-0.5, min(0.5, self.threshold_adjustments[metric_name]))


class NotificationSystem:
    """Multi-channel notification system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger("notification_system")
        
        # Channel configurations
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.webhook_config = config.get('webhook', {})
        self.sms_config = config.get('sms', {})
        
        # Notification rate limiting
        self.rate_limits: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Template system
        self.alert_templates = self._load_alert_templates()
        
        self.logger.info("NotificationSystem initialized")
    
    def _load_alert_templates(self) -> Dict[str, Dict[str, str]]:
        """Load notification templates."""
        return {
            'critical': {
                'email_subject': '🚨 CRITICAL: Medical AI System Alert - {alert_title}',
                'email_body': self._get_critical_email_template(),
                'slack': ':rotating_light: *CRITICAL ALERT* - {alert_title}\n{alert_message}',
                'sms': 'CRITICAL Medical AI Alert: {alert_title}'
            },
            'warning': {
                'email_subject': '⚠️ WARNING: Medical AI System Alert - {alert_title}',
                'email_body': self._get_warning_email_template(),
                'slack': ':warning: *WARNING* - {alert_title}\n{alert_message}',
                'sms': 'Medical AI Warning: {alert_title}'
            },
            'error': {
                'email_subject': '❌ ERROR: Medical AI System Alert - {alert_title}',
                'email_body': self._get_error_email_template(),
                'slack': ':x: *ERROR* - {alert_title}\n{alert_message}',
                'sms': 'Medical AI Error: {alert_title}'
            }
        }
    
    async def send_notification(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send notification via specified channel."""
        
        try:
            # Rate limiting check
            if not self._check_rate_limit(channel.value, alert.rule_id):
                self.logger.debug("Notification rate limited", 
                                channel=channel.value, 
                                rule_id=alert.rule_id)
                return False
            
            success = False
            
            if channel == NotificationChannel.EMAIL:
                success = await self._send_email_notification(alert)
            elif channel == NotificationChannel.SLACK:
                success = await self._send_slack_notification(alert)
            elif channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook_notification(alert)
            elif channel == NotificationChannel.SMS:
                success = await self._send_sms_notification(alert)
            elif channel == NotificationChannel.AUDIT_LOG:
                success = await self._send_audit_log_notification(alert)
            elif channel == NotificationChannel.DASHBOARD:
                success = await self._send_dashboard_notification(alert)
            
            if success:
                # Update rate limiting
                self._update_rate_limit(channel.value, alert.rule_id)
                
                # Log notification
                self.logger.info("Notification sent successfully",
                               channel=channel.value,
                               alert_id=alert.alert_id,
                               severity=alert.severity.value)
            
            return success
            
        except Exception as e:
            self.logger.error("Notification failed", 
                            channel=channel.value,
                            error=str(e))
            return False
    
    def _check_rate_limit(self, channel: str, rule_id: str) -> bool:
        """Check if notification is within rate limits."""
        last_sent = self.rate_limits[channel][rule_id]
        cooldown = 1800  # 30 minutes default cooldown
        
        return time.time() - last_sent > cooldown
    
    def _update_rate_limit(self, channel: str, rule_id: str):
        """Update rate limiting timestamp."""
        self.rate_limits[channel][rule_id] = time.time()
    
    async def _send_email_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        
        if not self.email_config:
            return False
        
        try:
            # Get template
            template = self.alert_templates.get(alert.severity.value, {}).get('email_body', '')
            
            # Format message
            formatted_message = template.format(
                alert_title=alert.title,
                alert_message=alert.message,
                metric_name=alert.metric_name,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                severity=alert.severity.value.upper(),
                timestamp=datetime.fromtimestamp(alert.triggered_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Create email
            msg = MimeMultipart()
            msg['From'] = self.email_config.get('from_address', 'alerts@medical-ai.com')
            msg['To'] = ', '.join(self.email_config.get('to_addresses', []))
            
            template_info = self.alert_templates.get(alert.severity.value, {}).get('email_subject', '')
            msg['Subject'] = template_info.format(alert_title=alert.title)
            
            msg.attach(MimeText(formatted_message, 'html'))
            
            # Send email
            if self.email_config.get('smtp_host'):
                server = smtplib.SMTP(
                    self.email_config['smtp_host'],
                    self.email_config.get('smtp_port', 587)
                )
                server.starttls()
                server.login(
                    self.email_config.get('username'),
                    self.email_config.get('password')
                )
                server.send_message(msg)
                server.quit()
                
                return True
            
        except Exception as e:
            self.logger.error("Email notification failed", error=str(e))
        
        return False
    
    async def _send_slack_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        
        if not self.slack_config:
            return False
        
        try:
            webhook_url = self.slack_config.get('webhook_url')
            if not webhook_url:
                return False
            
            # Get template
            template = self.alert_templates.get(alert.severity.value, {}).get('slack', '')
            
            # Format message
            formatted_message = template.format(
                alert_title=alert.title,
                alert_message=alert.message
            )
            
            # Create payload
            payload = {
                'text': formatted_message,
                'username': 'Medical AI Alerts',
                'icon_emoji': ':medical_symbol:'
            }
            
            # Send to Slack
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
            
        except Exception as e:
            self.logger.error("Slack notification failed", error=str(e))
        
        return False
    
    async def _send_webhook_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        
        if not self.webhook_config:
            return False
        
        try:
            webhook_url = self.webhook_config.get('url')
            if not webhook_url:
                return False
            
            # Create payload
            payload = {
                'alert': asdict(alert),
                'timestamp': time.time(),
                'source': 'medical_ai_monitoring'
            }
            
            # Send webhook
            async with aiohttp.ClientSession() as session:
                headers = self.webhook_config.get('headers', {'Content-Type': 'application/json'})
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    return response.status < 400
            
        except Exception as e:
            self.logger.error("Webhook notification failed", error=str(e))
        
        return False
    
    async def _send_sms_notification(self, alert: Alert) -> bool:
        """Send SMS notification."""
        
        if not self.sms_config:
            return False
        
        try:
            # This would integrate with SMS providers like Twilio
            # For now, just log the attempt
            template = self.alert_templates.get(alert.severity.value, {}).get('sms', '')
            formatted_message = template.format(alert_title=alert.title)
            
            self.logger.info("SMS notification would be sent",
                           message=formatted_message,
                           phone_numbers=self.sms_config.get('phone_numbers', []))
            
            # Return True for simulation
            return True
            
        except Exception as e:
            self.logger.error("SMS notification failed", error=str(e))
        
        return False
    
    async def _send_audit_log_notification(self, alert: Alert) -> bool:
        """Send audit log notification."""
        
        try:
            audit_log_path = Path('./logs/audit.log')
            audit_log_path.parent.mkdir(exist_ok=True)
            
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'alert_id': alert.alert_id,
                'severity': alert.severity.value,
                'title': alert.title,
                'description': alert.description,
                'clinical_impact': alert.clinical_impact_score,
                'regulatory_compliance': alert.regulatory_compliance_issue,
                'patient_safety_risk': alert.patient_safety_risk
            }
            
            async with aiofiles.open(audit_log_path, 'a') as f:
                await f.write(json.dumps(audit_entry) + '\n')
            
            return True
            
        except Exception as e:
            self.logger.error("Audit log notification failed", error=str(e))
        
        return False
    
    async def _send_dashboard_notification(self, alert: Alert) -> bool:
        """Send dashboard notification."""
        
        try:
            # This would update a real-time dashboard
            dashboard_config = self.config.get('dashboard', {})
            dashboard_url = dashboard_config.get('api_url')
            
            if not dashboard_url:
                return False
            
            payload = {
                'type': 'alert',
                'data': asdict(alert)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(dashboard_url, json=payload) as response:
                    return response.status < 400
            
        except Exception as e:
            self.logger.error("Dashboard notification failed", error=str(e))
        
        return False
    
    def _get_critical_email_template(self) -> str:
        """Get critical alert email template."""
        return """
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="background-color: #ffebee; border: 2px solid #f44336; padding: 20px; margin-bottom: 20px;">
                <h2 style="color: #f44336; margin: 0;">🚨 CRITICAL MEDICAL AI ALERT</h2>
            </div>
            
            <h3>{alert_title}</h3>
            
            <p><strong>Severity:</strong> {severity}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Metric:</strong> {metric_name}</p>
            <p><strong>Current Value:</strong> {current_value}</p>
            <p><strong>Threshold:</strong> {threshold_value}</p>
            
            <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0;">
                <h4>Alert Details:</h4>
                <p>{alert_message}</p>
            </div>
            
            <div style="background-color: #fff3e0; border: 1px solid #ff9800; padding: 15px; margin: 15px 0;">
                <h4>⚠️ Medical AI Specific Warnings:</h4>
                <ul>
                    <li>This alert may impact patient care</li>
                    <li>Review model performance and clinical outcomes</li>
                    <li>Consider regulatory compliance requirements</li>
                    <li>Document all actions taken</li>
                </ul>
            </div>
            
            <p style="font-size: 12px; color: #666; margin-top: 30px;">
                This is an automated alert from the Medical AI Monitoring System.
                Please take immediate action and document your response.
            </p>
        </body>
        </html>
        """
    
    def _get_warning_email_template(self) -> str:
        """Get warning alert email template."""
        return """
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="background-color: #fff3e0; border: 2px solid #ff9800; padding: 20px; margin-bottom: 20px;">
                <h2 style="color: #ff9800; margin: 0;">⚠️ WARNING: Medical AI System Alert</h2>
            </div>
            
            <h3>{alert_title}</h3>
            
            <p><strong>Severity:</strong> {severity}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Metric:</strong> {metric_name}</p>
            <p><strong>Current Value:</strong> {current_value}</p>
            <p><strong>Threshold:</strong> {threshold_value}</p>
            
            <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0;">
                <h4>Alert Details:</h4>
                <p>{alert_message}</p>
            </div>
            
            <div style="background-color: #e8f5e8; border: 1px solid #4caf50; padding: 15px; margin: 15px 0;">
                <h4>📊 Recommended Actions:</h4>
                <ul>
                    <li>Monitor the metric closely</li>
                    <li>Check recent model deployments</li>
                    <li>Review system resource usage</li>
                    <li>Consider preventive maintenance</li>
                </ul>
            </div>
            
            <p style="font-size: 12px; color: #666; margin-top: 30px;">
                This is an automated alert from the Medical AI Monitoring System.
            </p>
        </body>
        </html>
        """
    
    def _get_error_email_template(self) -> str:
        """Get error alert email template."""
        return """
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="background-color: #ffebee; border: 2px solid #e91e63; padding: 20px; margin-bottom: 20px;">
                <h2 style="color: #e91e63; margin: 0;">❌ ERROR: Medical AI System Alert</h2>
            </div>
            
            <h3>{alert_title}</h3>
            
            <p><strong>Severity:</strong> {severity}</p>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Metric:</strong> {metric_name}</p>
            <p><strong>Current Value:</strong> {current_value}</p>
            <p><strong>Threshold:</strong> {threshold_value}</p>
            
            <div style="background-color: #f5f5f5; padding: 15px; margin: 15px 0;">
                <h4>Error Details:</h4>
                <p>{alert_message}</p>
            </div>
            
            <div style="background-color: #fce4ec; border: 1px solid #e91e63; padding: 15px; margin: 15px 0;">
                <h4>🔧 Technical Actions Required:</h4>
                <ul>
                    <li>Check system logs for detailed error information</li>
                    <li>Verify model service status</li>
                    <li>Review recent configuration changes</li>
                    <li>Consider system restart if necessary</li>
                </ul>
            </div>
            
            <p style="font-size: 12px; color: #666; margin-top: 30px;">
                This is an automated alert from the Medical AI Monitoring System.
            </p>
        </body>
        </html>
        """


class AlertManager:
    """Main alert management system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger("alert_manager")
        
        # Components
        self.threshold_manager = ThresholdManager()
        self.notification_system = NotificationSystem(config.get('notifications', {}))
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.suppressed_alerts: Dict[str, float] = {}  # rule_id -> suppression_end_time
        
        # Metric tracking
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_alert_times: Dict[str, float] = {}
        
        # Performance tracking
        self.escalation_stats: Dict[str, int] = defaultdict(int)
        self.resolution_times: List[float] = []
        
        # Load default alert rules
        self._load_default_rules()
        
        self.logger.info("AlertManager initialized")
    
    def _load_default_rules(self):
        """Load default alert rules for medical AI systems."""
        
        default_rules = [
            # System performance alerts
            AlertRule(
                rule_id="cpu_high",
                name="High CPU Usage",
                description="CPU usage exceeds safe threshold",
                metric_name="system.cpu_percent",
                condition="greater_than",
                threshold_value=85.0,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                tags=["system", "performance"]
            ),
            AlertRule(
                rule_id="memory_high",
                name="High Memory Usage",
                description="Memory usage exceeds safe threshold",
                metric_name="system.memory_usage_percent",
                condition="greater_than",
                threshold_value=90.0,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                tags=["system", "memory"]
            ),
            AlertRule(
                rule_id="gpu_memory_critical",
                name="Critical GPU Memory Usage",
                description="GPU memory usage at critical level",
                metric_name="system.gpu_memory_usage_percent",
                condition="greater_than",
                threshold_value=95.0,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS],
                escalation_enabled=True,
                tags=["system", "gpu", "critical"]
            ),
            
            # Model performance alerts
            AlertRule(
                rule_id="model_accuracy_low",
                name="Low Model Accuracy",
                description="Model accuracy has dropped below acceptable threshold",
                metric_name="model.accuracy_score",
                condition="less_than",
                threshold_value=0.85,
                severity=AlertSeverity.ERROR,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                clinical_impact_assessment=True,
                tags=["model", "accuracy", "performance"]
            ),
            AlertRule(
                rule_id="inference_latency_high",
                name="High Inference Latency",
                description="Model inference latency exceeds SLA threshold",
                metric_name="model.avg_latency_ms",
                condition="greater_than",
                threshold_value=2000.0,  # 2 seconds
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.EMAIL],
                tags=["model", "latency", "performance"]
            ),
            AlertRule(
                rule_id="error_rate_high",
                name="High Error Rate",
                description="Model error rate is elevated",
                metric_name="model.error_rate",
                condition="greater_than",
                threshold_value=0.05,  # 5%
                severity=AlertSeverity.ERROR,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                escalation_enabled=True,
                tags=["model", "errors", "reliability"]
            ),
            
            # Clinical outcome alerts
            AlertRule(
                rule_id="clinical_effectiveness_low",
                name="Low Clinical Effectiveness",
                description="Clinical effectiveness score has dropped",
                metric_name="clinical.effectiveness_score",
                condition="less_than",
                threshold_value=0.80,
                severity=AlertSeverity.ERROR,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                clinical_impact_assessment=True,
                regulatory_compliance_check=True,
                tags=["clinical", "effectiveness", "safety"]
            ),
            AlertRule(
                rule_id="adverse_event_rate_high",
                name="High Adverse Event Rate",
                description="Rate of adverse events exceeds safety threshold",
                metric_name="clinical.adverse_event_rate",
                condition="greater_than",
                threshold_value=0.02,  # 2%
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.SMS],
                escalation_enabled=True,
                clinical_impact_assessment=True,
                regulatory_compliance_check=True,
                tags=["clinical", "safety", "critical"]
            ),
            
            # Drift detection alerts
            AlertRule(
                rule_id="data_drift_detected",
                name="Data Drift Detected",
                description="Significant data drift has been detected",
                metric_name="model.data_drift_score",
                condition="greater_than",
                threshold_value=0.3,
                severity=AlertSeverity.WARNING,
                notification_channels=[NotificationChannel.EMAIL],
                tags=["model", "drift", "quality"]
            ),
            AlertRule(
                rule_id="concept_drift_critical",
                name="Critical Concept Drift",
                description="Critical concept drift detected - model retraining required",
                metric_name="model.concept_drift_score",
                condition="greater_than",
                threshold_value=0.5,
                severity=AlertSeverity.CRITICAL,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                escalation_enabled=True,
                tags=["model", "drift", "critical"]
            ),
            
            # Regulatory compliance alerts
            AlertRule(
                rule_id="compliance_violation",
                name="Regulatory Compliance Violation",
                description="Regulatory compliance issue detected",
                metric_name="regulatory.compliance_score",
                condition="less_than",
                threshold_value=0.9,
                severity=AlertSeverity.ERROR,
                notification_channels=[NotificationChannel.EMAIL],
                regulatory_compliance_check=True,
                tags=["compliance", "regulatory", "audit"]
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        self.logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule."""
        
        try:
            self.alert_rules[rule.rule_id] = rule
            rule.last_modified = time.time()
            
            self.logger.info("Alert rule added", 
                           rule_id=rule.rule_id,
                           name=rule.name,
                           severity=rule.severity.value)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to add alert rule", error=str(e))
            return False
    
    def evaluate_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None):
        """Evaluate a metric against alert rules."""
        
        # Store metric value
        self.metric_values[metric_name].append({
            'value': value,
            'timestamp': time.time(),
            'context': context or {}
        })
        
        # Check applicable rules
        for rule in self.alert_rules.values():
            if rule.metric_name != metric_name or not rule.enabled:
                continue
            
            # Check suppression
            if self._is_rule_suppressed(rule.rule_id):
                continue
            
            # Evaluate condition
            if self._evaluate_condition(rule, value):
                # Check cooldown period
                if self._is_in_cooldown(rule.rule_id, metric_name):
                    continue
                
                # Create alert
                alert = self._create_alert(rule, value, context)
                
                # Send notifications
                asyncio.create_task(self._process_alert(alert))
        
        # Update threshold manager with baseline data
        self.threshold_manager.add_baseline_data(metric_name, value)
    
    def _evaluate_condition(self, rule: AlertRule, value: float) -> bool:
        """Evaluate if metric value violates alert rule condition."""
        
        if rule.condition == "greater_than":
            return value > rule.threshold_value
        elif rule.condition == "less_than":
            return value < rule.threshold_value
        elif rule.condition == "equals":
            return abs(value - rule.threshold_value) < 0.001
        elif rule.condition == "outside_range":
            if rule.threshold_upper is None:
                return False
            return value < rule.threshold_value or value > rule.threshold_upper
        
        return False
    
    def _is_rule_suppressed(self, rule_id: str) -> bool:
        """Check if rule is currently suppressed."""
        
        if rule_id not in self.suppressed_alerts:
            return False
        
        suppression_end = self.suppressed_alerts[rule_id]
        return time.time() < suppression_end
    
    def _is_in_cooldown(self, rule_id: str, metric_name: str) -> bool:
        """Check if rule is in cooldown period."""
        
        key = f"{rule_id}:{metric_name}"
        last_alert = self.last_alert_times.get(key, 0)
        cooldown_period = next(
            (rule.notification_cooldown for rule in self.alert_rules.values() 
             if rule.rule_id == rule_id), 
            1800  # 30 minutes default
        )
        
        return time.time() - last_alert < cooldown_period
    
    def _create_alert(self, rule: AlertRule, value: float, context: Dict[str, Any]) -> Alert:
        """Create alert from rule violation."""
        
        alert_id = f"{rule.rule_id}_{int(time.time() * 1000)}"
        
        # Generate alert content
        title = f"{rule.name}"
        
        if rule.condition == "greater_than":
            message = f"{rule.metric_name} is {value:.2f}, which exceeds the threshold of {rule.threshold_value:.2f}"
        elif rule.condition == "less_than":
            message = f"{rule.metric_name} is {value:.2f}, which is below the threshold of {rule.threshold_value:.2f}"
        else:
            message = f"{rule.metric_name} value {value:.2f} triggered {rule.name}"
        
        description = rule.description
        
        # Assess clinical impact for medical AI metrics
        clinical_impact_score = 0.0
        regulatory_compliance_issue = False
        patient_safety_risk = False
        
        if rule.clinical_impact_assessment:
            clinical_impact_score = self._assess_clinical_impact(rule, value, context)
            
            if clinical_impact_score > 0.8:
                patient_safety_risk = True
            
            # Check for compliance issues
            if "compliance" in rule.metric_name or "regulatory" in rule.metric_name:
                regulatory_compliance_issue = True
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=title,
            description=description,
            message=message,
            metric_name=rule.metric_name,
            current_value=value,
            threshold_value=rule.threshold_value,
            metric_context=context or {},
            triggered_timestamp=time.time(),
            clinical_impact_score=clinical_impact_score,
            regulatory_compliance_issue=regulatory_compliance_issue,
            patient_safety_risk=patient_safety_risk,
            created_by="system"
        )
        
        return alert
    
    def _assess_clinical_impact(self, rule: AlertRule, value: float, context: Dict[str, Any]) -> float:
        """Assess clinical impact of alert."""
        
        impact_score = 0.0
        
        # Base impact by severity
        severity_multipliers = {
            AlertSeverity.EMERGENCY: 1.0,
            AlertSeverity.CRITICAL: 0.9,
            AlertSeverity.ERROR: 0.7,
            AlertSeverity.WARNING: 0.5,
            AlertSeverity.INFO: 0.3
        }
        
        impact_score = severity_multipliers.get(rule.severity, 0.5)
        
        # Adjust for metric type
        if "accuracy" in rule.metric_name:
            impact_score *= 1.2  # Accuracy issues have high clinical impact
        
        if "safety" in rule.metric_name or "adverse" in rule.metric_name:
            impact_score *= 1.5  # Safety issues have maximum impact
        
        if "effectiveness" in rule.metric_name:
            impact_score *= 1.1  # Effectiveness has clinical impact
        
        # Adjust for context
        medical_specialty = context.get('medical_specialty', 'general')
        if medical_specialty in ['emergency', 'cardiology', 'oncology']:
            impact_score *= 1.3
        
        # Adjust for patient population
        patient_population = context.get('patient_population', 'general')
        if patient_population in ['pediatric', 'elderly', 'critical']:
            impact_score *= 1.2
        
        return min(1.0, impact_score)
    
    async def _process_alert(self, alert: Alert):
        """Process an alert (send notifications, handle escalation)."""
        
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update last alert time
            key = f"{alert.rule_id}:{alert.metric_name}"
            self.last_alert_times[key] = alert.triggered_timestamp
            
            # Send notifications
            rule = self.alert_rules[alert.rule_id]
            
            notification_tasks = []
            for channel in rule.notification_channels:
                task = asyncio.create_task(
                    self.notification_system.send_notification(alert, channel)
                )
                notification_tasks.append(task)
            
            # Wait for notifications with timeout
            if notification_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*notification_tasks, return_exceptions=True),
                    timeout=30.0
                )
            
            # Log alert
            self.logger.warning("Alert triggered",
                              alert_id=alert.alert_id,
                              rule_id=alert.rule_id,
                              severity=alert.severity.value,
                              clinical_impact=alert.clinical_impact_score)
            
            # Schedule escalation if enabled
            if rule.escalation_enabled and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
                asyncio.create_task(self._schedule_escalation(alert))
            
        except Exception as e:
            self.logger.error("Alert processing failed", 
                            alert_id=alert.alert_id,
                            error=str(e))
    
    async def _schedule_escalation(self, alert: Alert):
        """Schedule alert escalation."""
        
        rule = self.alert_rules[alert.rule_id]
        max_escalations = rule.max_escalations
        escalation_interval = rule.escalation_interval
        
        for escalation_level in range(1, max_escalations + 1):
            await asyncio.sleep(escalation_interval)
            
            # Check if alert is still active
            if alert.alert_id not in self.active_alerts:
                break
            
            current_alert = self.active_alerts[alert.alert_id]
            if current_alert.status != AlertStatus.ACTIVE:
                break
            
            # Escalate
            await self._escalate_alert(current_alert, escalation_level)
    
    async def _escalate_alert(self, alert: Alert, escalation_level: int):
        """Escalate an alert to next level."""
        
        try:
            alert.escalation_count = escalation_level
            alert.escalation_timestamp = time.time()
            
            # Add to escalation trail
            escalation_info = {
                'level': escalation_level,
                'timestamp': alert.escalation_timestamp,
                'reason': f'Automatic escalation level {escalation_level}',
                'channels_added': ['email', 'sms'] if escalation_level > 1 else []
            }
            alert.escalation_levels.append(escalation_info)
            
            # Update severity for critical escalations
            if escalation_level >= 2 and alert.severity != AlertSeverity.EMERGENCY:
                alert.severity = AlertSeverity.EMERGENCY
            
            # Send additional notifications
            additional_channels = [
                NotificationChannel.SMS,
                NotificationChannel.WEBHOOK
            ]
            
            for channel in additional_channels:
                await self.notification_system.send_notification(alert, channel)
            
            self.escalation_stats[alert.rule_id] += 1
            
            self.logger.warning("Alert escalated",
                              alert_id=alert.alert_id,
                              escalation_level=escalation_level,
                              severity=alert.severity.value)
            
        except Exception as e:
            self.logger.error("Alert escalation failed", 
                            alert_id=alert.alert_id,
                            error=str(e))
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: str = None) -> bool:
        """Acknowledge an alert."""
        
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_timestamp = time.time()
        
        # Add to audit trail
        audit_entry = {
            'action': 'acknowledged',
            'timestamp': alert.acknowledged_timestamp,
            'user': acknowledged_by,
            'notes': notes
        }
        alert.audit_trail.append(audit_entry)
        
        self.logger.info("Alert acknowledged",
                       alert_id=alert_id,
                       acknowledged_by=acknowledged_by)
        
        return True
    
    def resolve_alert(self, alert_id: str, resolved_by: str, notes: str = None) -> bool:
        """Resolve an alert."""
        
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_timestamp = time.time()
        
        # Calculate resolution time
        resolution_time = alert.resolved_timestamp - alert.triggered_timestamp
        self.resolution_times.append(resolution_time)
        
        # Add to audit trail
        audit_entry = {
            'action': 'resolved',
            'timestamp': alert.resolved_timestamp,
            'user': resolved_by,
            'notes': notes,
            'resolution_time_seconds': resolution_time
        }
        alert.audit_trail.append(audit_entry)
        
        # Remove from active alerts
        del self.active_alerts[alert_id]
        
        self.logger.info("Alert resolved",
                       alert_id=alert_id,
                       resolved_by=resolved_by,
                       resolution_time_seconds=resolution_time)
        
        return True
    
    def suppress_alerts(self, rule_ids: List[str], duration_minutes: int = 60) -> bool:
        """Suppress alerts for specified rules."""
        
        suppression_end = time.time() + (duration_minutes * 60)
        
        for rule_id in rule_ids:
            if rule_id in self.alert_rules:
                self.suppressed_alerts[rule_id] = suppression_end
        
        self.logger.info("Alerts suppressed",
                       rule_ids=rule_ids,
                       duration_minutes=duration_minutes)
        
        return True
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        
        now = time.time()
        last_24h = now - 86400
        last_7d = now - 604800
        
        # Count alerts by status and severity
        status_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for alert in self.active_alerts.values():
            status_counts[alert.status.value] += 1
            severity_counts[alert.severity.value] += 1
        
        # Calculate metrics for last 24h
        recent_alerts = [a for a in self.alert_history if a.triggered_timestamp >= last_24h]
        
        resolution_times_24h = [
            a.resolved_timestamp - a.triggered_timestamp 
            for a in recent_alerts 
            if a.resolved_timestamp is not None
        ]
        
        avg_resolution_time_24h = np.mean(resolution_times_24h) if resolution_times_24h else 0
        
        return {
            'active_alerts': len(self.active_alerts),
            'total_alerts_24h': len(recent_alerts),
            'total_alerts_7d': len([a for a in self.alert_history if a.triggered_timestamp >= last_7d]),
            'status_distribution': dict(status_counts),
            'severity_distribution': dict(severity_counts),
            'average_resolution_time_24h': avg_resolution_time_24h,
            'escalation_statistics': dict(self.escalation_stats),
            'suppressed_rules': list(self.suppressed_alerts.keys()),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled])
        }