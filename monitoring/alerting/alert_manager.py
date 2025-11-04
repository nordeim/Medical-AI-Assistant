"""
Automated Anomaly Detection and Alerting System
Provides intelligent alerting with escalation procedures and smart notification
management for medical AI systems with healthcare-specific alert handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import requests
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import yaml

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertStatus(Enum):
    """Alert status tracking"""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    ASSIGNED = "assigned"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"

@dataclass
class AlertRule:
    """Configuration for alert rules"""
    name: str
    description: str
    query: str
    severity: AlertSeverity
    threshold: float
    duration_seconds: int
    notification_channels: List[NotificationChannel]
    escalation_rules: List[Dict[str, Any]]
    runbook_url: Optional[str] = None
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}

@dataclass
class Alert:
    """Alert instance with tracking"""
    id: str
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    notifications_sent: List[Dict[str, Any]] = None
    escalation_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}
        if self.notifications_sent is None:
            self.notifications_sent = []
        if self.escalation_history is None:
            self.escalation_history = []

class MedicalAlertRuleEngine:
    """Rule engine for evaluating alert conditions"""
    
    def __init__(self, 
                 evaluation_interval_seconds: int = 30,
                 rule_timeout_seconds: int = 300):
        """
        Initialize alert rule engine
        
        Args:
            evaluation_interval_seconds: How often to evaluate rules
            rule_timeout_seconds: Timeout for rule evaluation
        """
        self.evaluation_interval_seconds = evaluation_interval_seconds
        self.rule_timeout_seconds = rule_timeout_seconds
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)  # Keep last 10k alerts
        
        # Medical AI specific thresholds and patterns
        self.medical_thresholds = {
            'clinical_decision_accuracy': 0.85,
            'model_bias_threshold': 0.1,
            'phi_exposure_threshold': 0,
            'audit_log_gap_hours': 24,
            'backup_failure_hours': 1,
            'critical_service_down_seconds': 60,
            'high_error_rate_percent': 5.0,
            'database_latency_seconds': 5.0,
            'cpu_utilization_percent': 85.0,
            'memory_utilization_percent': 90.0,
            'disk_utilization_percent': 85.0
        }
        
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule"""
        self.rules[rule.name] = rule
        logging.info(f"Added alert rule: {rule.name}")
    
    def load_rules_from_config(self, config_path: str) -> None:
        """Load alert rules from YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'alert_rules' in config:
                for rule_config in config['alert_rules']:
                    rule = self._create_rule_from_config(rule_config)
                    self.add_rule(rule)
                    
            logging.info(f"Loaded {len(self.rules)} alert rules from {config_path}")
            
        except Exception as e:
            logging.error(f"Error loading rules from {config_path}: {str(e)}")
    
    def _create_rule_from_config(self, rule_config: Dict[str, Any]) -> AlertRule:
        """Create AlertRule from configuration"""
        return AlertRule(
            name=rule_config['name'],
            description=rule_config.get('description', ''),
            query=rule_config['query'],
            severity=AlertSeverity(rule_config['severity']),
            threshold=rule_config['threshold'],
            duration_seconds=rule_config['duration_seconds'],
            notification_channels=[NotificationChannel(ch) for ch in rule_config.get('notification_channels', [])],
            escalation_rules=rule_config.get('escalation_rules', []),
            runbook_url=rule_config.get('runbook_url'),
            labels=rule_config.get('labels', {}),
            annotations=rule_config.get('annotations', {})
        )
    
    def evaluate_rules(self, metrics_data: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all rules against current metrics
        
        Args:
            metrics_data: Current system metrics
            
        Returns:
            List of triggered alerts
        """
        new_alerts = []
        
        try:
            for rule_name, rule in self.rules.items():
                try:
                    alert = self._evaluate_single_rule(rule, metrics_data)
                    if alert:
                        new_alerts.append(alert)
                except Exception as e:
                    logging.error(f"Error evaluating rule {rule_name}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Error in rule evaluation: {str(e)}")
            
        return new_alerts
    
    def _evaluate_single_rule(self, rule: AlertRule, metrics_data: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate a single alert rule"""
        try:
            # Simple query evaluation (in production, this would be more sophisticated)
            value = self._extract_metric_value(rule.query, metrics_data)
            
            if value is None:
                return None
            
            # Check if threshold is exceeded
            threshold_exceeded = value > rule.threshold if rule.severity != AlertSeverity.INFO else True
            
            # Check duration requirement
            rule_key = f"{rule.name}_{rule.query}"
            
            if threshold_exceeded:
                # Update or create alert
                if rule_key in self.active_alerts:
                    # Existing alert - check if duration requirement is met
                    existing_alert = self.active_alerts[rule_key]
                    duration = (datetime.now() - existing_alert.created_at).total_seconds()
                    
                    if duration >= rule.duration_seconds and existing_alert.status == AlertStatus.NEW:
                        # Duration requirement met, trigger alert
                        return existing_alert
                else:
                    # New potential alert - create and check duration
                    alert = self._create_alert(rule, value)
                    self.active_alerts[rule_key] = alert
                    
                    # Check if duration requirement is already met
                    if rule.duration_seconds <= 0:
                        return alert
            else:
                # Threshold not exceeded - resolve existing alert if any
                if rule_key in self.active_alerts:
                    resolved_alert = self._resolve_alert(self.active_alerts[rule_key])
                    if resolved_alert:
                        self.alert_history.append(resolved_alert)
                        del self.active_alerts[rule_key]
                        
            return None
            
        except Exception as e:
            logging.error(f"Error evaluating rule {rule.name}: {str(e)}")
            return None
    
    def _extract_metric_value(self, query: str, metrics_data: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from query"""
        try:
            # Simple extraction logic (would be more sophisticated in production)
            # Handle common metric paths
            if '.' in query:
                parts = query.split('.')
                value = metrics_data
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    elif isinstance(value, list) and part.isdigit():
                        idx = int(part)
                        if idx < len(value):
                            value = value[idx]
                        else:
                            return None
                    else:
                        return None
                
                # Convert to float if possible
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass
                        
            return None
            
        except Exception as e:
            logging.error(f"Error extracting metric value for query '{query}': {str(e)}")
            return None
    
    def _create_alert(self, rule: AlertRule, metric_value: float) -> Alert:
        """Create alert from rule and metric value"""
        alert_id = self._generate_alert_id(rule.name)
        
        # Generate alert title and description
        title = f"{rule.severity.value.upper()}: {rule.name}"
        description = f"{rule.description}\n\nMetric value: {metric_value}\nThreshold: {rule.threshold}\nQuery: {rule.query}"
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            title=title,
            description=description,
            severity=rule.severity,
            status=AlertStatus.NEW,
            created_at=datetime.now(),
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy()
        )
        
        # Add timestamp and value to annotations
        alert.annotations['timestamp'] = datetime.now().isoformat()
        alert.annotations['metric_value'] = str(metric_value)
        alert.annotations['threshold'] = str(rule.threshold)
        alert.annotations['query'] = rule.query
        
        return alert
    
    def _resolve_alert(self, alert: Alert) -> Optional[Alert]:
        """Resolve an alert"""
        if alert.status in [AlertStatus.RESOLVED, AlertStatus.CLOSED]:
            return None
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Add resolution annotation
        alert.annotations['resolved_at'] = alert.resolved_at.isoformat()
        alert.annotations['resolution_note'] = 'Threshold condition no longer met'
        
        return alert
    
    def _generate_alert_id(self, rule_name: str) -> str:
        """Generate unique alert ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        hash_input = f"{rule_name}_{timestamp}_{np.random.randint(1000, 9999)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]

class NotificationManager:
    """Manages notifications across multiple channels"""
    
    def __init__(self, 
                 smtp_config: Optional[Dict[str, Any]] = None,
                 slack_config: Optional[Dict[str, Any]] = None,
                 pagerduty_config: Optional[Dict[str, Any]] = None):
        """
        Initialize notification manager
        
        Args:
            smtp_config: SMTP server configuration
            slack_config: Slack webhook configuration
            pagerduty_config: PagerDuty integration configuration
        """
        self.smtp_config = smtp_config or {}
        self.slack_config = slack_config or {}
        self.pagerduty_config = pagerduty_config or {}
        
        self.notification_queue = deque()
        self.notification_workers = []
        self.max_workers = 5
        
    def send_notifications(self, alert: Alert, channels: List[NotificationChannel]) -> List[Dict[str, Any]]:
        """
        Send notifications for an alert
        
        Args:
            alert: Alert to send notifications for
            channels: List of notification channels
            
        Returns:
            List of notification results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for channel in channels:
                future = executor.submit(self._send_notification, alert, channel)
                futures.append((channel, future))
            
            for channel, future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout per notification
                    results.append({
                        'channel': channel.value,
                        'status': 'success',
                        'result': result
                    })
                    
                    # Add to alert's notification history
                    alert.notifications_sent.append({
                        'channel': channel.value,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'sent'
                    })
                    
                except Exception as e:
                    error_result = {
                        'channel': channel.value,
                        'status': 'error',
                        'error': str(e)
                    }
                    results.append(error_result)
                    
                    # Add failed notification to history
                    alert.notifications_sent.append({
                        'channel': channel.value,
                        'timestamp': datetime.now().isoformat(),
                        'status': 'failed',
                        'error': str(e)
                    })
                    
                    logging.error(f"Failed to send {channel.value} notification for alert {alert.id}: {str(e)}")
        
        return results
    
    def _send_notification(self, alert: Alert, channel: NotificationChannel) -> Any:
        """Send notification to specific channel"""
        if channel == NotificationChannel.EMAIL:
            return self._send_email_notification(alert)
        elif channel == NotificationChannel.SLACK:
            return self._send_slack_notification(alert)
        elif channel == NotificationChannel.PAGERDUTY:
            return self._send_pagerduty_notification(alert)
        elif channel == NotificationChannel.WEBHOOK:
            return self._send_webhook_notification(alert)
        else:
            logging.warning(f"Unsupported notification channel: {channel}")
            return None
    
    def _send_email_notification(self, alert: Alert) -> str:
        """Send email notification"""
        if not self.smtp_config:
            raise Exception("SMTP configuration not provided")
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = self.smtp_config.get('from', 'alerts@medical-ai.local')
        msg['To'] = self.smtp_config.get('to', 'admin@medical-ai.local')
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        
        # Email body
        body = f"""
        Medical AI System Alert
        
        Alert ID: {alert.id}
        Severity: {alert.severity.value.upper()}
        Rule: {alert.rule_name}
        Status: {alert.status.value}
        Created: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}
        
        Description:
        {alert.description}
        
        Labels:
        {json.dumps(alert.labels, indent=2)}
        
        Annotations:
        {json.dumps(alert.annotations, indent=2)}
        
        This is an automated alert from the Medical AI Monitoring System.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
        if self.smtp_config.get('use_tls', True):
            server.starttls()
        if self.smtp_config.get('username') and self.smtp_config.get('password'):
            server.login(self.smtp_config['username'], self.smtp_config['password'])
        
        text = msg.as_string()
        server.sendmail(msg['From'], msg['To'].split(','), text)
        server.quit()
        
        return f"Email sent to {msg['To']}"
    
    def _send_slack_notification(self, alert: Alert) -> str:
        """Send Slack notification"""
        if not self.slack_config.get('webhook_url'):
            raise Exception("Slack webhook URL not configured")
        
        # Prepare Slack message
        color_map = {
            AlertSeverity.INFO: 'good',
            AlertSeverity.WARNING: 'warning',
            AlertSeverity.ERROR: 'danger',
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.EMERGENCY: '#ff0000'
        }
        
        payload = {
            "text": f"Medical AI Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.severity, 'warning'),
                    "fields": [
                        {"title": "Alert ID", "value": alert.id, "short": True},
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Rule", "value": alert.rule_name, "short": True},
                        {"title": "Status", "value": alert.status.value, "short": True},
                        {"title": "Created", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S'), "short": False},
                        {"title": "Description", "value": alert.description, "short": False}
                    ]
                }
            ]
        }
        
        # Send webhook
        response = requests.post(
            self.slack_config['webhook_url'],
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code != 200:
            raise Exception(f"Slack API returned {response.status_code}: {response.text}")
        
        return f"Slack notification sent successfully"
    
    def _send_pagerduty_notification(self, alert: Alert) -> str:
        """Send PagerDuty notification"""
        if not self.pagerduty_config.get('integration_key'):
            raise Exception("PagerDuty integration key not configured")
        
        # Map severity to PagerDuty
        severity_map = {
            AlertSeverity.INFO: 'info',
            AlertSeverity.WARNING: 'warning',
            AlertSeverity.ERROR: 'error',
            AlertSeverity.CRITICAL: 'critical',
            AlertSeverity.EMERGENCY: 'critical'
        }
        
        payload = {
            'routing_key': self.pagerduty_config['integration_key'],
            'event_action': 'trigger',
            'payload': {
                'summary': alert.title,
                'source': 'Medical AI Monitoring System',
                'severity': severity_map.get(alert.severity, 'warning'),
                'timestamp': alert.created_at.isoformat(),
                'custom_details': {
                    'alert_id': alert.id,
                    'rule_name': alert.rule_name,
                    'description': alert.description,
                    'labels': alert.labels,
                    'annotations': alert.annotations
                }
            }
        }
        
        # Send to PagerDuty
        response = requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code != 202:
            raise Exception(f"PagerDuty API returned {response.status_code}: {response.text}")
        
        return "PagerDuty notification sent successfully"
    
    def _send_webhook_notification(self, alert: Alert) -> str:
        """Send generic webhook notification"""
        webhook_url = os.getenv('ALERT_WEBHOOK_URL')
        if not webhook_url:
            raise Exception("Webhook URL not configured")
        
        payload = {
            'alert': asdict(alert),
            'timestamp': datetime.now().isoformat(),
            'source': 'medical_ai_monitoring_system'
        }
        
        response = requests.post(
            webhook_url,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code >= 400:
            raise Exception(f"Webhook returned {response.status_code}: {response.text}")
        
        return "Webhook notification sent successfully"

class EscalationManager:
    """Manages alert escalation based on rules and timing"""
    
    def __init__(self, escalation_check_interval: int = 60):
        """
        Initialize escalation manager
        
        Args:
            escalation_check_interval: How often to check for escalations (seconds)
        """
        self.escalation_check_interval = escalation_check_interval
        self.running = False
        self.escalation_thread = None
        
    def start_escalation_monitoring(self, 
                                  active_alerts: Dict[str, Alert],
                                  rule_engine: MedicalAlertRuleEngine,
                                  notification_manager: NotificationManager) -> None:
        """
        Start monitoring alerts for escalation
        
        Args:
            active_alerts: Dictionary of active alerts
            rule_engine: Alert rule engine with escalation rules
            notification_manager: Notification manager
        """
        self.running = True
        self.active_alerts = active_alerts
        self.rule_engine = rule_engine
        self.notification_manager = notification_manager
        
        self.escalation_thread = threading.Thread(target=self._escalation_monitoring_loop, daemon=True)
        self.escalation_thread.start()
        
        logging.info("Started escalation monitoring")
    
    def stop_escalation_monitoring(self) -> None:
        """Stop escalation monitoring"""
        self.running = False
        if self.escalation_thread:
            self.escalation_thread.join(timeout=5)
        
        logging.info("Stopped escalation monitoring")
    
    def _escalation_monitoring_loop(self) -> None:
        """Main escalation monitoring loop"""
        while self.running:
            try:
                self._check_escalations()
                time.sleep(self.escalation_check_interval)
            except Exception as e:
                logging.error(f"Error in escalation monitoring loop: {str(e)}")
                time.sleep(self.escalation_check_interval)
    
    def _check_escalations(self) -> None:
        """Check for alerts that need escalation"""
        current_time = datetime.now()
        
        for alert_key, alert in self.active_alerts.items():
            try:
                # Get rule for this alert
                rule = self.rule_engine.rules.get(alert.rule_name)
                if not rule:
                    continue
                
                # Check each escalation rule
                for escalation_rule in rule.escalation_rules:
                    self._evaluate_escalation_rule(alert, escalation_rule, current_time, rule)
                    
            except Exception as e:
                logging.error(f"Error checking escalation for alert {alert.id}: {str(e)}")
    
    def _evaluate_escalation_rule(self,
                                alert: Alert,
                                escalation_rule: Dict[str, Any],
                                current_time: datetime,
                                rule: AlertRule) -> None:
        """Evaluate a single escalation rule"""
        try:
            escalation_type = escalation_rule.get('type', 'time_based')
            conditions = escalation_rule.get('conditions', {})
            
            if escalation_type == 'time_based':
                self._handle_time_based_escalation(alert, conditions, current_time, rule)
            elif escalation_type == 'severity_based':
                self._handle_severity_based_escalation(alert, conditions, rule)
            elif escalation_type == 'notification_based':
                self._handle_notification_based_escalation(alert, conditions, rule)
                
        except Exception as e:
            logging.error(f"Error evaluating escalation rule: {str(e)}")
    
    def _handle_time_based_escalation(self,
                                    alert: Alert,
                                    conditions: Dict[str, Any],
                                    current_time: datetime,
                                    rule: AlertRule) -> None:
        """Handle time-based escalation"""
        if not alert.escalated_at:  # Don't escalate already escalated alerts
            escalation_minutes = conditions.get('minutes', 30)
            elapsed_time = (current_time - alert.created_at).total_seconds() / 60
            
            if elapsed_time >= escalation_minutes:
                self._escalate_alert(alert, rule, f"Time-based escalation after {escalation_minutes} minutes")
    
    def _handle_severity_based_escalation(self,
                                        alert: Alert,
                                        conditions: Dict[str, Any],
                                        rule: AlertRule) -> None:
        """Handle severity-based escalation"""
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] and not alert.escalated_at:
            self._escalate_alert(alert, rule, "Immediate escalation due to critical/emergency severity")
    
    def _handle_notification_based_escalation(self,
                                            alert: Alert,
                                            conditions: Dict[str, Any],
                                            rule: AlertRule) -> None:
        """Handle escalation based on notification failures"""
        failed_notifications = [
            notif for notif in alert.notifications_sent 
            if notif.get('status') == 'failed'
        ]
        
        max_failures = conditions.get('max_failures', 2)
        
        if len(failed_notifications) >= max_failures and not alert.escalated_at:
            self._escalate_alert(alert, rule, f"Escalation after {len(failed_notifications)} notification failures")
    
    def _escalate_alert(self,
                       alert: Alert,
                       rule: AlertRule,
                       reason: str) -> None:
        """Actually escalate an alert"""
        # Update alert status
        alert.status = AlertStatus.ESCALATED
        alert.escalated_at = datetime.now()
        
        # Add escalation annotation
        alert.annotations['escalated_at'] = alert.escalated_at.isoformat()
        alert.annotations['escalation_reason'] = reason
        
        # Add to escalation history
        alert.escalation_history.append({
            'timestamp': alert.escalated_at.isoformat(),
            'reason': reason,
            'escalated_to': rule.notification_channels
        })
        
        # Send escalation notifications
        try:
            escalation_channels = rule.notification_channels
            if escalation_channels:
                self.notification_manager.send_notifications(alert, escalation_channels)
                
                logging.info(f"Escalated alert {alert.id}: {reason}")
        except Exception as e:
            logging.error(f"Failed to send escalation notifications for alert {alert.id}: {str(e)}")

class AlertManager:
    """Main orchestrator for the alerting system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize alert manager
        
        Args:
            config_path: Path to alerting configuration file
        """
        self.rule_engine = MedicalAlertRuleEngine()
        self.notification_manager = NotificationManager()
        self.escalation_manager = EscalationManager()
        
        # Load configuration if provided
        if config_path:
            self.load_configuration(config_path)
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'resolved_alerts': 0,
            'escalated_alerts': 0,
            'notifications_sent': 0,
            'notifications_failed': 0
        }
    
    def load_configuration(self, config_path: str) -> None:
        """Load complete alerting configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load alert rules
            if 'alert_rules' in config:
                self.rule_engine.load_rules_from_config(config_path)
            
            # Configure notification channels
            if 'notifications' in config:
                notification_config = config['notifications']
                
                if 'email' in notification_config:
                    self.notification_manager.smtp_config = notification_config['email']
                
                if 'slack' in notification_config:
                    self.notification_manager.slack_config = notification_config['slack']
                
                if 'pagerduty' in notification_config:
                    self.notification_manager.pagerduty_config = notification_config['pagerduty']
            
            # Load medical-specific thresholds
            if 'medical_thresholds' in config:
                self.rule_engine.medical_thresholds.update(config['medical_thresholds'])
            
            logging.info(f"Loaded alerting configuration from {config_path}")
            
        except Exception as e:
            logging.error(f"Error loading configuration from {config_path}: {str(e)}")
            raise
    
    def add_medical_alert_rules(self) -> None:
        """Add standard medical AI alert rules"""
        rules = [
            # Clinical Safety Rules
            AlertRule(
                name="clinical_decision_accuracy_low",
                description="Clinical decision support system accuracy below acceptable threshold",
                query="metrics.clinical_decision_accuracy",
                severity=AlertSeverity.CRITICAL,
                threshold=0.85,
                duration_seconds=300,  # 5 minutes
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                escalation_rules=[
                    {"type": "time_based", "conditions": {"minutes": 15}},
                    {"type": "severity_based", "conditions": {}}
                ],
                runbook_url="https://runbooks.medical-ai.example.com/clinical-accuracy",
                labels={"category": "clinical_safety", "component": "decision_support"}
            ),
            
            # HIPAA Compliance Rules
            AlertRule(
                name="phi_data_exposure",
                description="Protected Health Information (PHI) exposure detected",
                query="metrics.phi_exposure_count",
                severity=AlertSeverity.EMERGENCY,
                threshold=0,
                duration_seconds=0,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY, NotificationChannel.WEBHOOK],
                escalation_rules=[
                    {"type": "severity_based", "conditions": {}},
                    {"type": "time_based", "conditions": {"minutes": 5}}
                ],
                runbook_url="https://runbooks.medical-ai.example.com/phi-exposure",
                labels={"category": "compliance", "regulation": "hipaa", "component": "data_protection"}
            ),
            
            # Model Performance Rules
            AlertRule(
                name="model_bias_detected",
                description="AI model bias exceeding acceptable threshold",
                query="metrics.model_bias_score",
                severity=AlertSeverity.WARNING,
                threshold=0.1,
                duration_seconds=600,  # 10 minutes
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                escalation_rules=[
                    {"type": "time_based", "conditions": {"minutes": 30}},
                    {"type": "notification_based", "conditions": {"max_failures": 1}}
                ],
                runbook_url="https://runbooks.medical-ai.example.com/model-bias",
                labels={"category": "model_performance", "component": "ai_ml"}
            ),
            
            # System Health Rules
            AlertRule(
                name="high_cpu_utilization",
                description="System CPU utilization above threshold",
                query="metrics.cpu_utilization",
                severity=AlertSeverity.WARNING,
                threshold=85.0,
                duration_seconds=300,
                notification_channels=[NotificationChannel.EMAIL],
                escalation_rules=[
                    {"type": "time_based", "conditions": {"minutes": 60}}
                ],
                labels={"category": "system_health", "component": "infrastructure"}
            ),
            
            # Database Rules
            AlertRule(
                name="database_high_latency",
                description="Database query latency above acceptable threshold",
                query="metrics.database_latency",
                severity=AlertSeverity.ERROR,
                threshold=5.0,
                duration_seconds=180,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                escalation_rules=[
                    {"type": "time_based", "conditions": {"minutes": 20}}
                ],
                labels={"category": "database", "component": "infrastructure"}
            ),
            
            # Security Rules
            AlertRule(
                name="unauthorized_access_attempts",
                description="Multiple unauthorized access attempts detected",
                query="metrics.auth_failures_5min",
                severity=AlertSeverity.CRITICAL,
                threshold=10,
                duration_seconds=120,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                escalation_rules=[
                    {"type": "time_based", "conditions": {"minutes": 10}},
                    {"type": "notification_based", "conditions": {"max_failures": 1}}
                ],
                runbook_url="https://runbooks.medical-ai.example.com/unauthorized-access",
                labels={"category": "security", "regulation": "hipaa"}
            ),
            
            # Audit and Compliance Rules
            AlertRule(
                name="audit_log_gap",
                description="No audit log entries for extended period",
                query="metrics.hours_since_last_audit_log",
                severity=AlertSeverity.ERROR,
                threshold=24,
                duration_seconds=0,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                escalation_rules=[
                    {"type": "time_based", "conditions": {"minutes": 30}}
                ],
                labels={"category": "audit", "regulation": "hipaa"}
            ),
            
            # Backup and DR Rules
            AlertRule(
                name="backup_failure",
                description="Database backup operation failed",
                query="metrics.backup_status",
                severity=AlertSeverity.CRITICAL,
                threshold=1,  # 1 means failure
                duration_seconds=0,
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY],
                escalation_rules=[
                    {"type": "severity_based", "conditions": {}},
                    {"type": "time_based", "conditions": {"minutes": 15}}
                ],
                runbook_url="https://runbooks.medical-ai.example.com/backup-failure",
                labels={"category": "disaster_recovery", "regulation": "hipaa"}
            )
        ]
        
        for rule in rules:
            self.rule_engine.add_rule(rule)
        
        logging.info(f"Added {len(rules)} medical AI alert rules")
    
    def process_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process current metrics and generate alerts
        
        Args:
            metrics_data: Current system metrics
            
        Returns:
            Dict containing processing results and generated alerts
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'metrics_processed': len(metrics_data),
            'alerts_generated': [],
            'alerts_resolved': [],
            'escalations_performed': [],
            'statistics': self.stats.copy()
        }
        
        try:
            # Evaluate rules
            new_alerts = self.rule_engine.evaluate_rules(metrics_data)
            
            # Process new alerts
            for alert in new_alerts:
                try:
                    # Send notifications
                    rule = self.rule_engine.rules[alert.rule_name]
                    notification_results = self.notification_manager.send_notifications(
                        alert, rule.notification_channels
                    )
                    
                    # Update statistics
                    self.stats['total_alerts'] += 1
                    self.stats['active_alerts'] += 1
                    
                    # Count successful and failed notifications
                    for result in notification_results:
                        if result['status'] == 'success':
                            self.stats['notifications_sent'] += 1
                        else:
                            self.stats['notifications_failed'] += 1
                    
                    results['alerts_generated'].append({
                        'alert_id': alert.id,
                        'rule_name': alert.rule_name,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'notifications_sent': len([r for r in notification_results if r['status'] == 'success'])
                    })
                    
                    logging.info(f"Generated alert {alert.id}: {alert.title}")
                    
                except Exception as e:
                    logging.error(f"Error processing alert {alert.id}: {str(e)}")
            
            # Check for resolved alerts
            resolved_alerts = []
            for alert_key, alert in self.rule_engine.active_alerts.items():
                # Check if this alert was already resolved in evaluation
                if alert.status == AlertStatus.RESOLVED:
                    resolved_alerts.append(alert)
                    del self.rule_engine.active_alerts[alert_key]
            
            for alert in resolved_alerts:
                results['alerts_resolved'].append({
                    'alert_id': alert.id,
                    'rule_name': alert.rule_name,
                    'resolved_at': alert.resolved_at.isoformat()
                })
                self.stats['active_alerts'] -= 1
                self.stats['resolved_alerts'] += 1
                
                logging.info(f"Resolved alert {alert.id}")
            
            # Update overall statistics
            results['statistics'] = self.stats.copy()
            
        except Exception as e:
            logging.error(f"Error processing metrics: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def get_active_alerts(self, severity_filter: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        alerts = []
        
        for alert in self.rule_engine.active_alerts.values():
            if severity_filter is None or alert.severity == severity_filter:
                alerts.append({
                    'id': alert.id,
                    'rule_name': alert.rule_name,
                    'title': alert.title,
                    'severity': alert.severity.value,
                    'status': alert.status.value,
                    'created_at': alert.created_at.isoformat(),
                    'labels': alert.labels,
                    'annotations': alert.annotations
                })
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.rule_engine.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.now()
                alert.assigned_to = acknowledged_by
                
                logging.info(f"Acknowledged alert {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_note: str = "") -> bool:
        """Manually resolve an alert"""
        for alert_key, alert in self.rule_engine.active_alerts.items():
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                alert.annotations['resolved_by'] = resolved_by
                alert.annotations['resolution_note'] = resolution_note
                
                # Move to history
                self.rule_engine.alert_history.append(alert)
                del self.rule_engine.active_alerts[alert_key]
                
                # Update statistics
                self.stats['active_alerts'] -= 1
                self.stats['resolved_alerts'] += 1
                
                logging.info(f"Manually resolved alert {alert_id} by {resolved_by}")
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        return self.stats.copy()
    
    def start_monitoring(self) -> None:
        """Start the complete monitoring system"""
        # Start escalation monitoring
        self.escalation_manager.start_escalation_monitoring(
            self.rule_engine.active_alerts,
            self.rule_engine,
            self.notification_manager
        )
        
        logging.info("Started complete alerting system")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        self.escalation_manager.stop_escalation_monitoring()
        logging.info("Stopped alerting system")

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Add medical AI alert rules
    alert_manager.add_medical_alert_rules()
    
    # Start monitoring
    alert_manager.start_monitoring()
    
    # Example metrics data
    sample_metrics = {
        'metrics': {
            'clinical_decision_accuracy': 0.82,  # Below threshold
            'phi_exposure_count': 0,  # Good
            'model_bias_score': 0.12,  # Above threshold
            'cpu_utilization': 87.5,  # Above threshold
            'database_latency': 6.2,  # Above threshold
            'auth_failures_5min': 15,  # Above threshold
            'hours_since_last_audit_log': 25,  # Above threshold
            'backup_status': 1  # Failure
        }
    }
    
    # Process metrics
    results = alert_manager.process_metrics(sample_metrics)
    
    print("=== Alert Processing Results ===")
    print(f"Metrics Processed: {results['metrics_processed']}")
    print(f"Alerts Generated: {len(results['alerts_generated'])}")
    print(f"Alerts Resolved: {len(results['alerts_resolved'])}")
    
    print("\n=== Generated Alerts ===")
    for alert in results['alerts_generated']:
        print(f"- {alert['severity'].upper()}: {alert['title']}")
        print(f"  Rule: {alert['rule_name']}")
        print(f"  Notifications Sent: {alert['notifications_sent']}")
    
    print(f"\n=== Current Active Alerts ===")
    active_alerts = alert_manager.get_active_alerts()
    for alert in active_alerts:
        print(f"- [{alert['severity'].upper()}] {alert['title']}")
        print(f"  Status: {alert['status']}")
        print(f"  Created: {alert['created_at']}")
    
    print(f"\n=== Alert Statistics ===")
    stats = alert_manager.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Stop monitoring
    time.sleep(5)  # Give time for escalation checks
    alert_manager.stop_monitoring()