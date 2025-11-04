"""
Advanced Notification System

Provides comprehensive notification capabilities with multi-channel support,
template management, and delivery tracking for medical AI monitoring alerts.
"""

import asyncio
import json
import smtplib
import ssl
import time
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.image import MimeImage
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import aiohttp
import aiofiles
import structlog
import base64

from ...config.logging_config import get_logger

logger = structlog.get_logger("notification_system")


class NotificationChannel:
    """Base class for notification channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rate_limiter = RateLimiter()
        self.delivery_tracker = DeliveryTracker()
    
    async def send(self, alert_data: Dict[str, Any], template: str = None) -> bool:
        """Send notification. Override in subclasses."""
        raise NotImplementedError
    
    def can_send(self, recipient: str, alert_type: str) -> bool:
        """Check if notification can be sent based on rate limits."""
        return self.rate_limiter.can_send(recipient, alert_type)


class RateLimiter:
    """Rate limiter for notifications."""
    
    def __init__(self):
        self.message_history: Dict[str, List[float]] = {}
        self.rate_limits = {
            'critical': 0,      # No limit for critical alerts
            'emergency': 0,     # No limit for emergency alerts  
            'error': 300,       # 5 minutes
            'warning': 600,     # 10 minutes
            'info': 1800        # 30 minutes
        }
    
    def can_send(self, recipient: str, alert_type: str) -> bool:
        """Check if message can be sent based on rate limits."""
        current_time = time.time()
        
        if recipient not in self.message_history:
            self.message_history[recipient] = []
        
        # Get rate limit for alert type
        rate_limit = self.rate_limits.get(alert_type, 1800)
        
        if rate_limit == 0:  # No limit
            return True
        
        # Clean old messages
        cutoff_time = current_time - rate_limit
        self.message_history[recipient] = [
            msg_time for msg_time in self.message_history[recipient]
            if msg_time > cutoff_time
        ]
        
        # Check if within rate limit
        return len(self.message_history[recipient]) < 10  # Max 10 messages per window
    
    def record_message(self, recipient: str):
        """Record that a message was sent."""
        current_time = time.time()
        if recipient not in self.message_history:
            self.message_history[recipient] = []
        self.message_history[recipient].append(current_time)


class DeliveryTracker:
    """Tracks notification delivery status."""
    
    def __init__(self, storage_path: str = "./logs/notifications.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(exist_ok=True)
        self.delivery_history: List[Dict[str, Any]] = []
    
    def record_delivery(self, 
                       notification_id: str,
                       channel: str,
                       recipient: str,
                       status: str,
                       response_time: float,
                       error_message: str = None):
        """Record notification delivery attempt."""
        delivery_record = {
            'notification_id': notification_id,
            'channel': channel,
            'recipient': recipient,
            'status': status,  # 'sent', 'failed', 'retry'
            'response_time': response_time,
            'timestamp': time.time(),
            'error_message': error_message
        }
        
        self.delivery_history.append(delivery_record)
        
        # Persist to disk
        asyncio.create_task(self._persist_delivery_record(delivery_record))
    
    async def _persist_delivery_record(self, record: Dict[str, Any]):
        """Persist delivery record to disk."""
        try:
            # Load existing records
            if self.storage_path.exists():
                async with aiofiles.open(self.storage_path, 'r') as f:
                    content = await f.read()
                    existing_records = json.loads(content) if content else []
            else:
                existing_records = []
            
            # Add new record
            existing_records.append(record)
            
            # Keep only last 1000 records
            existing_records = existing_records[-1000:]
            
            # Save back
            async with aiofiles.open(self.storage_path, 'w') as f:
                await f.write(json.dumps(existing_records, indent=2))
                
        except Exception as e:
            logger.error("Failed to persist delivery record", error=str(e))


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel with HTML templates."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.smtp_server = config.get('smtp_server', 'localhost')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_address = config.get('from_address', 'alerts@medical-ai.com')
        self.use_tls = config.get('use_tls', True)
        self.use_ssl = config.get('use_ssl', False)
        
        # Recipients by alert severity
        self.recipients = config.get('recipients', {
            'emergency': [],
            'critical': [],
            'error': [],
            'warning': [],
            'info': []
        })
        
        # Email templates
        self.templates = self._load_email_templates()
    
    def _load_email_templates(self) -> Dict[str, str]:
        """Load email templates for different alert types."""
        return {
            'emergency': self._get_emergency_template(),
            'critical': self._get_critical_template(),
            'error': self._get_error_template(),
            'warning': self._get_warning_template(),
            'info': self._get_info_template()
        }
    
    async def send(self, alert_data: Dict[str, Any], template: str = None) -> bool:
        """Send email notification."""
        try:
            severity = alert_data.get('severity', 'info')
            recipients = self._get_recipients_for_severity(severity)
            
            if not recipients:
                logger.warning("No recipients configured for severity", severity=severity)
                return False
            
            # Check rate limiting
            for recipient in recipients:
                if not self.can_send(recipient, severity):
                    logger.debug("Email rate limited", recipient=recipient, severity=severity)
                    continue
            
            # Get template
            template_content = self.templates.get(severity, self.templates['info'])
            if template:
                template_content = template
            
            # Format email content
            email_subject, email_body = self._format_email(alert_data, template_content)
            
            # Send email
            success = await self._send_email(recipients, email_subject, email_body)
            
            # Record delivery
            notification_id = f"email_{int(time.time() * 1000)}"
            for recipient in recipients:
                status = 'sent' if success else 'failed'
                self.delivery_tracker.record_delivery(
                    notification_id, 'email', recipient, status, 0.0
                )
                if success:
                    self.rate_limiter.record_message(recipient)
            
            return success
            
        except Exception as e:
            logger.error("Email notification failed", error=str(e))
            return False
    
    def _get_recipients_for_severity(self, severity: str) -> List[str]:
        """Get recipients for specific severity level."""
        recipients = set()
        
        # Get recipients for specific severity
        recipients.update(self.recipients.get(severity, []))
        
        # Add higher severity recipients for critical emergencies
        if severity in ['error', 'warning', 'info']:
            recipients.update(self.recipients.get('critical', []))
            recipients.update(self.recipients.get('emergency', []))
        
        return list(recipients)
    
    def _format_email(self, alert_data: Dict[str, Any], template: str) -> tuple:
        """Format email subject and body from template and alert data."""
        
        # Extract alert information
        title = alert_data.get('title', 'Medical AI Alert')
        message = alert_data.get('message', '')
        severity = alert_data.get('severity', 'info').upper()
        timestamp = alert_data.get('timestamp', time.time())
        metric_name = alert_data.get('metric_name', 'Unknown')
        current_value = alert_data.get('current_value', 'Unknown')
        threshold = alert_data.get('threshold_value', 'Unknown')
        model_id = alert_data.get('model_id', 'Unknown')
        
        # Format timestamp
        formatted_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        
        # Clinical impact information
        clinical_impact = alert_data.get('clinical_impact_score', 0.0)
        patient_safety_risk = alert_data.get('patient_safety_risk', False)
        regulatory_issue = alert_data.get('regulatory_compliance_issue', False)
        
        # Format subject
        severity_emoji = {
            'EMERGENCY': '🚨',
            'CRITICAL': '⚠️',
            'ERROR': '❌',
            'WARNING': '⚠️',
            'INFO': 'ℹ️'
        }.get(severity, '📧')
        
        subject = f"{severity_emoji} Medical AI {severity} Alert: {title}"
        
        # Format body
        body = template.format(
            title=title,
            message=message,
            severity=severity,
            timestamp=formatted_timestamp,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold,
            model_id=model_id,
            clinical_impact=clinical_impact,
            patient_safety_risk=patient_safety_risk,
            regulatory_issue=regulatory_issue,
            **alert_data.get('additional_context', {})
        )
        
        return subject, body
    
    async def _send_email(self, recipients: List[str], subject: str, body: str) -> bool:
        """Send email via SMTP."""
        try:
            # Create message
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_address
            msg['To'] = ', '.join(recipients)
            
            # Add HTML body
            html_part = MimeText(body, 'html', 'utf-8')
            msg.attach(html_part)
            
            # Send email
            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                    if self.username and self.password:
                        server.login(self.username, self.password)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    if self.username and self.password:
                        server.login(self.username, self.password)
                    server.send_message(msg)
            
            logger.info("Email sent successfully",
                       recipients=len(recipients),
                       subject=subject)
            
            return True
            
        except Exception as e:
            logger.error("Email sending failed", error=str(e))
            return False
    
    def _get_emergency_template(self) -> str:
        """Emergency alert email template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }
                .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: #dc3545; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
                .header h1 { margin: 0; font-size: 24px; }
                .header p { margin: 5px 0 0 0; opacity: 0.9; }
                .content { padding: 20px; }
                .alert-box { background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; border-radius: 4px; margin: 15px 0; }
                .alert-box h3 { margin: 0 0 10px 0; color: #721c24; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                .metrics-table th, .metrics-table td { padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }
                .metrics-table th { background-color: #f8f9fa; font-weight: 600; }
                .clinical-warning { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; margin: 15px 0; }
                .clinical-warning h3 { margin: 0 0 10px 0; color: #856404; }
                .footer { padding: 20px; background: #f8f9fa; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 EMERGENCY: Medical AI System Alert</h1>
                    <p>Immediate attention required - Patient safety may be affected</p>
                </div>
                
                <div class="content">
                    <h2>{title}</h2>
                    
                    <div class="alert-box">
                        <h3>Emergency Situation</h3>
                        <p><strong>Alert:</strong> {message}</p>
                        <p><strong>Severity:</strong> {severity}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                    </div>
                    
                    <h3>Alert Details</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Current Value</th><th>Threshold</th></tr>
                        <tr><td>{metric_name}</td><td>{current_value}</td><td>{threshold_value}</td></tr>
                    </table>
                    
                    <p><strong>Model:</strong> {model_id}</p>
                    
                    {clinical_warning_section}
                    
                    <h3>Required Actions</h3>
                    <ol>
                        <li><strong>Immediate Response Required</strong> - This alert requires immediate attention</li>
                        <li>Review patient safety implications</li>
                        <li>Document all actions taken</li>
                        <li>Consider regulatory compliance requirements</li>
                        <li>Notify clinical leadership</li>
                    </ol>
                    
                    <p><em>This is an automated emergency alert from the Medical AI Monitoring System. 
                    Immediate action is required.</em></p>
                </div>
                
                <div class="footer">
                    Medical AI Monitoring System | Emergency Alert Notification<br>
                    Time Generated: {timestamp} | System: Medical AI Platform
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_critical_template(self) -> str:
        """Critical alert email template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }
                .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: #fd7e14; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
                .header h1 { margin: 0; font-size: 24px; }
                .content { padding: 20px; }
                .alert-box { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 4px; margin: 15px 0; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                .metrics-table th, .metrics-table td { padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }
                .metrics-table th { background-color: #f8f9fa; font-weight: 600; }
                .footer { padding: 20px; background: #f8f9fa; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ CRITICAL: Medical AI System Alert</h1>
                </div>
                
                <div class="content">
                    <h2>{title}</h2>
                    
                    <div class="alert-box">
                        <p><strong>Alert:</strong> {message}</p>
                        <p><strong>Severity:</strong> {severity}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                    </div>
                    
                    <h3>Alert Details</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Current Value</th><th>Threshold</th></tr>
                        <tr><td>{metric_name}</td><td>{current_value}</td><td>{threshold_value}</td></tr>
                    </table>
                    
                    <p><strong>Model:</strong> {model_id}</p>
                    
                    <h3>Recommended Actions</h3>
                    <ol>
                        <li>Review the alert and assess impact</li>
                        <li>Monitor the affected system closely</li>
                        <li>Consider preventive measures</li>
                        <li>Document response actions</li>
                    </ol>
                </div>
                
                <div class="footer">
                    Medical AI Monitoring System | Critical Alert Notification<br>
                    Time Generated: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_error_template(self) -> str:
        """Error alert email template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }
                .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: #6c757d; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
                .header h1 { margin: 0; font-size: 24px; }
                .content { padding: 20px; }
                .alert-box { background: #e9ecef; border: 1px solid #ced4da; padding: 15px; border-radius: 4px; margin: 15px 0; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                .metrics-table th, .metrics-table td { padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }
                .metrics-table th { background-color: #f8f9fa; font-weight: 600; }
                .footer { padding: 20px; background: #f8f9fa; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>❌ ERROR: Medical AI System Alert</h1>
                </div>
                
                <div class="content">
                    <h2>{title}</h2>
                    
                    <div class="alert-box">
                        <p><strong>Error:</strong> {message}</p>
                        <p><strong>Severity:</strong> {severity}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                    </div>
                    
                    <h3>Error Details</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Current Value</th><th>Threshold</th></tr>
                        <tr><td>{metric_name}</td><td>{current_value}</td><td>{threshold_value}</td></tr>
                    </table>
                    
                    <p><strong>Model:</strong> {model_id}</p>
                    
                    <h3>Resolution Steps</h3>
                    <ol>
                        <li>Check system logs for detailed error information</li>
                        <li>Verify service status and connectivity</li>
                        <li>Review recent configuration changes</li>
                        <li>Implement appropriate fix</li>
                    </ol>
                </div>
                
                <div class="footer">
                    Medical AI Monitoring System | Error Alert Notification<br>
                    Time Generated: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_warning_template(self) -> str:
        """Warning alert email template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }
                .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: #28a745; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
                .header h1 { margin: 0; font-size: 24px; }
                .content { padding: 20px; }
                .alert-box { background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 4px; margin: 15px 0; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                .metrics-table th, .metrics-table td { padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }
                .metrics-table th { background-color: #f8f9fa; font-weight: 600; }
                .footer { padding: 20px; background: #f8f9fa; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>⚠️ WARNING: Medical AI System Alert</h1>
                </div>
                
                <div class="content">
                    <h2>{title}</h2>
                    
                    <div class="alert-box">
                        <p><strong>Warning:</strong> {message}</p>
                        <p><strong>Severity:</strong> {severity}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                    </div>
                    
                    <h3>Warning Details</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Current Value</th><th>Threshold</th></tr>
                        <tr><td>{metric_name}</td><td>{current_value}</td><td>{threshold_value}</td></tr>
                    </table>
                    
                    <p><strong>Model:</strong> {model_id}</p>
                    
                    <h3>Recommended Actions</h3>
                    <ol>
                        <li>Monitor the metric for trend changes</li>
                        <li>Review system performance periodically</li>
                        <li>Consider capacity planning</li>
                        <li>Schedule maintenance if needed</li>
                    </ol>
                </div>
                
                <div class="footer">
                    Medical AI Monitoring System | Warning Alert Notification<br>
                    Time Generated: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """
    
    def _get_info_template(self) -> str:
        """Info alert email template."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f8f9fa; }
                .container { max-width: 800px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: #17a2b8; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
                .header h1 { margin: 0; font-size: 24px; }
                .content { padding: 20px; }
                .alert-box { background: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 4px; margin: 15px 0; }
                .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
                .metrics-table th, .metrics-table td { padding: 8px 12px; border: 1px solid #dee2e6; text-align: left; }
                .metrics-table th { background-color: #f8f9fa; font-weight: 600; }
                .footer { padding: 20px; background: #f8f9fa; border-radius: 0 0 8px 8px; font-size: 12px; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ℹ️ INFO: Medical AI System Notification</h1>
                </div>
                
                <div class="content">
                    <h2>{title}</h2>
                    
                    <div class="alert-box">
                        <p><strong>Information:</strong> {message}</p>
                        <p><strong>Type:</strong> {severity}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                    </div>
                    
                    <h3>Details</h3>
                    <table class="metrics-table">
                        <tr><th>Metric</th><th>Value</th><th>Threshold</th></tr>
                        <tr><td>{metric_name}</td><td>{current_value}</td><td>{threshold_value}</td></tr>
                    </table>
                    
                    <p><strong>Model:</strong> {model_id}</p>
                </div>
                
                <div class="footer">
                    Medical AI Monitoring System | Info Notification<br>
                    Time Generated: {timestamp}
                </div>
            </div>
        </body>
        </html>
        """


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('webhook_url')
        self.bot_token = config.get('bot_token')
        self.default_channel = config.get('default_channel', '#alerts')
        
        # Channel routing by severity
        self.channel_routing = config.get('channel_routing', {
            'emergency': '#emergency-alerts',
            'critical': '#critical-alerts',
            'error': '#errors',
            'warning': '#warnings',
            'info': '#info'
        })
    
    async def send(self, alert_data: Dict[str, Any], template: str = None) -> bool:
        """Send Slack notification."""
        try:
            if not self.webhook_url and not self.bot_token:
                logger.warning("No Slack webhook or bot token configured")
                return False
            
            severity = alert_data.get('severity', 'info')
            channel = self.channel_routing.get(severity, self.default_channel)
            
            # Format message
            slack_message = self._format_slack_message(alert_data, channel)
            
            # Send via webhook
            if self.webhook_url:
                success = await self._send_via_webhook(slack_message, channel)
            else:
                success = await self._send_via_bot_api(slack_message, channel)
            
            # Record delivery
            if success:
                notification_id = f"slack_{int(time.time() * 1000)}"
                self.delivery_tracker.record_delivery(
                    notification_id, 'slack', channel, 'sent', 0.0
                )
                self.rate_limiter.record_message(channel)
            
            return success
            
        except Exception as e:
            logger.error("Slack notification failed", error=str(e))
            return False
    
    def _format_slack_message(self, alert_data: Dict[str, Any], channel: str) -> Dict[str, Any]:
        """Format Slack message."""
        
        severity = alert_data.get('severity', 'info').lower()
        title = alert_data.get('title', 'Medical AI Alert')
        message = alert_data.get('message', '')
        timestamp = alert_data.get('timestamp', time.time())
        model_id = alert_data.get('model_id', 'Unknown')
        
        # Color by severity
        colors = {
            'emergency': 'danger',
            'critical': 'danger', 
            'error': 'warning',
            'warning': 'warning',
            'info': 'good'
        }
        
        color = colors.get(severity, 'good')
        
        # Severity emoji
        emoji_map = {
            'emergency': ':rotating_light:',
            'critical': ':warning:',
            'error': ':x:',
            'warning': ':warning:',
            'info': ':information_source:'
        }
        
        emoji = emoji_map.get(severity, ':information_source:')
        
        # Clinical impact warning
        clinical_impact = alert_data.get('clinical_impact_score', 0.0)
        patient_safety = alert_data.get('patient_safety_risk', False)
        
        clinical_warning = ""
        if patient_safety:
            clinical_warning = "\n:warning: *Patient Safety Impact Detected*"
        elif clinical_impact > 0.7:
            clinical_warning = f"\n:medical_symbol: Clinical Impact Score: {clinical_impact:.1%}"
        
        formatted_timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        
        return {
            'channel': channel,
            'username': 'Medical AI Alerts',
            'icon_emoji': ':medical_symbol:',
            'attachments': [
                {
                    'color': color,
                    'title': f"{emoji} {title}",
                    'text': message,
                    'fields': [
                        {
                            'title': 'Severity',
                            'value': severity.upper(),
                            'short': True
                        },
                        {
                            'title': 'Model',
                            'value': model_id,
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': formatted_timestamp,
                            'short': True
                        },
                        {
                            'title': 'Clinical Impact',
                            'value': f"{clinical_impact:.1%}",
                            'short': True
                        }
                    ],
                    'footer': 'Medical AI Monitoring System',
                    'ts': int(timestamp),
                    'mrkdwn_in': ['text', 'fields']
                }
            ],
            'text': f"Medical AI {severity.upper()} Alert: {title}{clinical_warning}"
        }
    
    async def _send_via_webhook(self, message: Dict[str, Any], channel: str) -> bool:
        """Send message via webhook."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=message) as response:
                    success = response.status == 200
                    
                    if not success:
                        response_text = await response.text()
                        logger.warning("Slack webhook failed",
                                     status=response.status,
                                     response=response_text)
                    
                    return success
                    
        except Exception as e:
            logger.error("Slack webhook error", error=str(e))
            return False
    
    async def _send_via_bot_api(self, message: Dict[str, Any], channel: str) -> bool:
        """Send message via Slack Bot API."""
        try:
            headers = {
                'Authorization': f'Bearer {self.bot_token}',
                'Content-Type': 'application/json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://slack.com/api/chat.postMessage',
                    headers=headers,
                    json=message
                ) as response:
                    result = await response.json()
                    return result.get('ok', False)
                    
        except Exception as e:
            logger.error("Slack API error", error=str(e))
            return False


class WebhookNotificationChannel(NotificationChannel):
    """Generic webhook notification channel."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.webhook_url = config.get('url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.authentication = config.get('authentication')  # 'bearer', 'basic', etc.
        self.auth_token = config.get('auth_token')
        self.username = config.get('username')
        self.password = config.get('password')
        
        # Retry configuration
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)  # seconds
    
    async def send(self, alert_data: Dict[str, Any], template: str = None) -> bool:
        """Send webhook notification."""
        try:
            if not self.webhook_url:
                logger.warning("No webhook URL configured")
                return False
            
            # Prepare payload
            payload = {
                'alert': alert_data,
                'timestamp': time.time(),
                'source': 'medical_ai_monitoring',
                'webhook_version': '1.0'
            }
            
            # Send with retries
            for attempt in range(self.max_retries):
                success = await self._send_webhook_request(payload)
                
                if success:
                    # Record successful delivery
                    notification_id = f"webhook_{int(time.time() * 1000)}"
                    self.delivery_tracker.record_delivery(
                        notification_id, 'webhook', self.webhook_url, 'sent', 0.0
                    )
                    return True
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
            
            # Record failed delivery
            notification_id = f"webhook_{int(time.time() * 1000)}"
            self.delivery_tracker.record_delivery(
                notification_id, 'webhook', self.webhook_url, 'failed', 0.0
            )
            
            return False
            
        except Exception as e:
            logger.error("Webhook notification failed", error=str(e))
            return False
    
    async def _send_webhook_request(self, payload: Dict[str, Any]) -> bool:
        """Send webhook request with authentication."""
        try:
            headers = self.headers.copy()
            
            # Add authentication
            if self.authentication == 'bearer' and self.auth_token:
                headers['Authorization'] = f'Bearer {self.auth_token}'
            elif self.authentication == 'basic' and self.username and self.password:
                import base64
                credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
                headers['Authorization'] = f'Basic {credentials}'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    success = response.status < 400
                    
                    if not success:
                        response_text = await response.text()
                        logger.warning("Webhook request failed",
                                     status=response.status,
                                     response=response_text[:500])
                    
                    return success
                    
        except asyncio.TimeoutError:
            logger.warning("Webhook request timed out")
            return False
        except Exception as e:
            logger.error("Webhook request error", error=str(e))
            return False


class SMSNotificationChannel(NotificationChannel):
    """SMS notification channel (placeholder for integration with SMS providers)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config.get('provider', 'twilio')  # twilio, aws_sns, etc.
        self.account_sid = config.get('account_sid')
        self.auth_token = config.get('auth_token')
        self.from_number = config.get('from_number')
        self.recipients = config.get('recipients', {})
        
        # Rate limiting for SMS (more restrictive)
        self.rate_limiter.rate_limits = {
            'emergency': 0,      # No limit for emergencies
            'critical': 300,     # 5 minutes
            'error': 600,        # 10 minutes  
            'warning': 1800,     # 30 minutes
            'info': 3600         # 1 hour
        }
    
    async def send(self, alert_data: Dict[str, Any], template: str = None) -> bool:
        """Send SMS notification."""
        try:
            severity = alert_data.get('severity', 'info')
            recipients = self._get_recipients_for_severity(severity)
            
            if not recipients:
                logger.warning("No SMS recipients configured for severity", severity=severity)
                return False
            
            # Check rate limiting
            for recipient in recipients:
                if not self.can_send(recipient, severity):
                    logger.debug("SMS rate limited", recipient=recipient, severity=severity)
                    continue
            
            # Format message
            message = self._format_sms_message(alert_data)
            
            # Send SMS
            if self.provider == 'twilio':
                success = await self._send_twilio_sms(recipients, message)
            else:
                logger.warning(f"SMS provider {self.provider} not implemented")
                return False
            
            # Record delivery
            if success:
                notification_id = f"sms_{int(time.time() * 1000)}"
                for recipient in recipients:
                    self.delivery_tracker.record_delivery(
                        notification_id, 'sms', recipient, 'sent', 0.0
                    )
                    self.rate_limiter.record_message(recipient)
            
            return success
            
        except Exception as e:
            logger.error("SMS notification failed", error=str(e))
            return False
    
    def _get_recipients_for_severity(self, severity: str) -> List[str]:
        """Get SMS recipients for specific severity level."""
        recipients = set()
        
        # Get recipients for specific severity
        recipients.update(self.recipients.get(severity, []))
        
        # Add emergency recipients for critical alerts
        if severity == 'critical':
            recipients.update(self.recipients.get('emergency', []))
        
        return list(recipients)
    
    def _format_sms_message(self, alert_data: Dict[str, Any]) -> str:
        """Format SMS message."""
        
        title = alert_data.get('title', 'Medical AI Alert')[:50]  # SMS length limit
        severity = alert_data.get('severity', 'info').upper()
        timestamp = time.strftime('%H:%M:%S', time.localtime(alert_data.get('time.time()', time.time())))
        
        # Severity prefix
        prefixes = {
            'EMERGENCY': '🚨 EMERGENCY: ',
            'CRITICAL': '⚠️ CRITICAL: ',
            'ERROR': '❌ ERROR: ',
            'WARNING': '⚠️ WARNING: ',
            'INFO': 'ℹ️ INFO: '
        }
        
        prefix = prefixes.get(severity, '')
        
        message = f"{prefix}{title} at {timestamp}. Check monitoring system for details."
        
        # Ensure message fits SMS limit (160 chars for single SMS)
        if len(message) > 160:
            message = message[:157] + "..."
        
        return message
    
    async def _send_twilio_sms(self, recipients: List[str], message: str) -> bool:
        """Send SMS via Twilio (placeholder implementation)."""
        try:
            # This would integrate with Twilio API
            # For now, just log the attempt
            logger.info("SMS would be sent via Twilio",
                       recipients=len(recipients),
                       message=message,
                       from_number=self.from_number)
            
            # Return True for simulation
            return True
            
        except Exception as e:
            logger.error("Twilio SMS failed", error=str(e))
            return False


class NotificationManager:
    """Main notification manager coordinating all channels."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = structlog.get_logger("notification_manager")
        
        # Initialize channels
        self.channels = self._initialize_channels()
        
        # Template management
        self.custom_templates: Dict[str, str] = {}
        
        # Delivery statistics
        self.delivery_stats = {
            'total_sent': 0,
            'total_failed': 0,
            'by_channel': defaultdict(int),
            'by_severity': defaultdict(int)
        }
        
        self.logger.info("NotificationManager initialized",
                        channels=list(self.channels.keys()))
    
    def _initialize_channels(self) -> Dict[str, NotificationChannel]:
        """Initialize all configured notification channels."""
        channels = {}
        
        # Email channel
        email_config = self.config.get('email', {})
        if email_config.get('enabled', True):
            channels['email'] = EmailNotificationChannel(email_config)
        
        # Slack channel
        slack_config = self.config.get('slack', {})
        if slack_config.get('enabled', True):
            channels['slack'] = SlackNotificationChannel(slack_config)
        
        # Webhook channel
        webhook_config = self.config.get('webhook', {})
        if webhook_config.get('enabled', True):
            channels['webhook'] = WebhookNotificationChannel(webhook_config)
        
        # SMS channel
        sms_config = self.config.get('sms', {})
        if sms_config.get('enabled', True):
            channels['sms'] = SMSNotificationChannel(sms_config)
        
        return channels
    
    async def send_alert_notification(self, 
                                    alert_data: Dict[str, Any],
                                    channels: List[str] = None,
                                    template: str = None) -> Dict[str, bool]:
        """Send alert notification through specified channels."""
        
        if channels is None:
            # Use default channels based on severity
            severity = alert_data.get('severity', 'info')
            channels = self._get_default_channels_for_severity(severity)
        
        results = {}
        
        # Send to each channel
        send_tasks = []
        for channel_name in channels:
            if channel_name in self.channels:
                channel = self.channels[channel_name]
                task = asyncio.create_task(channel.send(alert_data, template))
                send_tasks.append((channel_name, task))
        
        # Wait for all sends to complete
        for channel_name, task in send_tasks:
            try:
                success = await task
                results[channel_name] = success
                
                # Update statistics
                self.delivery_stats['total_sent' if success else 'total_failed'] += 1
                self.delivery_stats['by_channel'][channel_name] += 1
                
                severity = alert_data.get('severity', 'info')
                self.delivery_stats['by_severity'][severity] += 1
                
            except Exception as e:
                logger.error("Channel send failed",
                           channel=channel_name,
                           error=str(e))
                results[channel_name] = False
                self.delivery_stats['total_failed'] += 1
        
        return results
    
    def _get_default_channels_for_severity(self, severity: str) -> List[str]:
        """Get default notification channels for severity level."""
        
        default_channels = {
            'emergency': ['email', 'slack', 'sms'],
            'critical': ['email', 'slack'],
            'error': ['email', 'slack'],
            'warning': ['email'],
            'info': ['slack']
        }
        
        return default_channels.get(severity, ['email'])
    
    def add_custom_template(self, template_name: str, template_content: str) -> bool:
        """Add custom notification template."""
        try:
            self.custom_templates[template_name] = template_content
            logger.info("Custom template added", template_name=template_name)
            return True
        except Exception as e:
            logger.error("Failed to add custom template", error=str(e))
            return False
    
    def get_delivery_statistics(self) -> Dict[str, Any]:
        """Get notification delivery statistics."""
        
        total_attempts = self.delivery_stats['total_sent'] + self.delivery_stats['total_failed']
        success_rate = (
            self.delivery_stats['total_sent'] / total_attempts 
            if total_attempts > 0 else 0.0
        )
        
        return {
            'total_sent': self.delivery_stats['total_sent'],
            'total_failed': self.delivery_stats['total_failed'],
            'success_rate': success_rate,
            'by_channel': dict(self.delivery_stats['by_channel']),
            'by_severity': dict(self.delivery_stats['by_severity']),
            'available_channels': list(self.channels.keys())
        }
    
    def test_channel(self, channel_name: str, test_data: Dict[str, Any] = None) -> bool:
        """Test a specific notification channel."""
        
        if channel_name not in self.channels:
            logger.error("Channel not found", channel=channel_name)
            return False
        
        # Default test data
        if test_data is None:
            test_data = {
                'title': 'Test Notification',
                'message': 'This is a test notification from Medical AI Monitoring System',
                'severity': 'info',
                'timestamp': time.time(),
                'model_id': 'test_model',
                'metric_name': 'test_metric',
                'current_value': 'test_value',
                'threshold_value': 'test_threshold'
            }
        
        try:
            # Create asyncio task for sync channel
            result = asyncio.create_task(self.channels[channel_name].send(test_data))
            return asyncio.run(result)
        except Exception as e:
            logger.error("Channel test failed",
                       channel=channel_name,
                       error=str(e))
            return False