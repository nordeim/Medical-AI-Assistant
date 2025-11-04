"""
Response Automation System
Automated responses and workflows for healthcare support
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from config.support_config import SupportConfig, PriorityLevel
from ticketing.ticket_management import ticket_system, TicketCategory, MedicalContext
from feedback.feedback_collection import feedback_system, FeedbackType
from monitoring.health_checks import health_monitor, HealthStatus
from incident_management.emergency_response import incident_system, IncidentSeverity

logger = logging.getLogger(__name__)

class AutomationTrigger(Enum):
    TICKET_CREATED = "ticket_created"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    HEALTH_CHECK_FAILED = "health_check_failed"
    INCIDENT_ESCALATED = "incident_escalated"
    SENTIMENT_CRITICAL = "sentiment_critical"
    SLA_APPROACHING = "sla_approaching"
    EMERGENCY_DETECTED = "emergency_detected"

class AutomationAction(Enum):
    SEND_NOTIFICATION = "send_notification"
    CREATE_INCIDENT = "create_incident"
    ESCALATE_TICKET = "escalate_ticket"
    ASSIGN_AUTO = "assign_auto"
    SEND_SURVEY = "send_survey"
    UPDATE_STATUS = "update_status"
    ALERT_TEAM = "alert_team"
    CREATE_KB_ARTICLE = "create_kb_article"

@dataclass
class AutomationRule:
    """Automation rule definition"""
    id: str
    name: str
    description: str
    trigger: AutomationTrigger
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    enabled: bool
    priority: int
    last_executed: Optional[datetime] = None
    execution_count: int = 0

@dataclass
class AutomationContext:
    """Context data for automation execution"""
    trigger_data: Dict[str, Any]
    timestamp: datetime
    source_system: str
    priority_level: PriorityLevel
    facility_context: Dict[str, Any]
    user_context: Dict[str, Any]

class ResponseAutomationEngine:
    """Main automation engine for support responses"""
    
    def __init__(self):
        self.automation_rules: Dict[str, AutomationRule] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.notification_queue = []
        
        # Initialize default automation rules
        self._initialize_automation_rules()
    
    def _initialize_automation_rules(self) -> None:
        """Initialize default automation rules for healthcare support"""
        
        # Emergency medical ticket rule
        emergency_ticket_rule = AutomationRule(
            id="AUTO_001",
            name="Emergency Medical Ticket Response",
            description="Automatically respond to emergency medical tickets",
            trigger=AutomationTrigger.TICKET_CREATED,
            conditions={
                "priority": ["emergency", "critical_medical"],
                "category": ["medical_emergency", "clinical_issue"],
                "keywords": ["emergency", "urgent", "critical", "life-threatening"]
            },
            actions=[
                {
                    "type": AutomationAction.SEND_NOTIFICATION,
                    "recipients": ["emergency_team"],
                    "message": "Emergency medical ticket created: {ticket_id}",
                    "priority": "urgent"
                },
                {
                    "type": AutomationAction.ESCALATE_TICKET,
                    "escalation_level": "emergency_medical_team",
                    "reason": "Auto-escalated due to emergency keywords"
                },
                {
                    "type": AutomationAction.ALERT_TEAM,
                    "teams": ["emergency_medical_team"],
                    "alert_type": "medical_emergency"
                }
            ],
            enabled=True,
            priority=1
        )
        
        # Critical feedback response rule
        critical_feedback_rule = AutomationRule(
            id="AUTO_002",
            name="Critical Feedback Response",
            description="Respond to critical user feedback",
            trigger=AutomationTrigger.FEEDBACK_SUBMITTED,
            conditions={
                "patient_safety_mentioned": True,
                "sentiment": ["very_negative", "negative"],
                "urgency": ["high", "critical"]
            },
            actions=[
                {
                    "type": AutomationAction.CREATE_INCIDENT,
                    "incident_type": "patient_safety_concern",
                    "severity": "sev1_critical",
                    "title": "Patient Safety Concern from Feedback"
                },
                {
                    "type": AutomationAction.SEND_NOTIFICATION,
                    "recipients": ["safety_team", "management"],
                    "message": "Critical feedback mentions patient safety",
                    "priority": "critical"
                }
            ],
            enabled=True,
            priority=2
        )
        
        # Health check failure rule
        health_check_rule = AutomationRule(
            id="AUTO_003",
            name="Health Check Failure Response",
            description="Respond to health check failures",
            trigger=AutomationTrigger.HEALTH_CHECK_FAILED,
            conditions={
                "status": ["critical", "down"],
                "component_type": ["api_endpoint", "database", "medical_device"]
            },
            actions=[
                {
                    "type": AutomationAction.CREATE_INCIDENT,
                    "incident_type": "system_outage",
                    "severity": "sev2_high",
                    "title": "System Component Failure: {component_name}"
                },
                {
                    "type": AutomationAction.ALERT_TEAM,
                    "teams": ["operations_team"],
                    "alert_type": "system_failure"
                },
                {
                    "type": AutomationAction.SEND_NOTIFICATION,
                    "recipients": ["operations_team"],
                    "message": "Health check failed for {component_name}",
                    "priority": "high"
                }
            ],
            enabled=True,
            priority=3
        )
        
        # SLA approaching rule
        sla_rule = AutomationRule(
            id="AUTO_004",
            name="SLA Approaching Alert",
            description="Alert when SLA is approaching",
            trigger=AutomationTrigger.SLA_APPROACHING,
            conditions={
                "sla_remaining_minutes": {"max": 30},
                "priority": ["high_medical", "critical_medical"]
            },
            actions=[
                {
                    "type": AutomationAction.SEND_NOTIFICATION,
                    "recipients": ["assigned_agent", "team_lead"],
                    "message": "SLA approaching for ticket {ticket_id}",
                    "priority": "high"
                },
                {
                    "type": AutomationAction.ALERT_TEAM,
                    "teams": ["support_team"],
                    "alert_type": "sla_warning"
                }
            ],
            enabled=True,
            priority=4
        )
        
        # Sentiment analysis rule
        sentiment_rule = AutomationRule(
            id="AUTO_005",
            name="Critical Sentiment Response",
            description="Respond to critical negative sentiment",
            trigger=AutomationTrigger.SENTIMENT_CRITICAL,
            conditions={
                "sentiment": "very_negative",
                "action_required": True,
                "patient_safety_concern": True
            },
            actions=[
                {
                    "type": AutomationAction.CREATE_INCIDENT,
                    "incident_type": "patient_safety_risk",
                    "severity": "sev1_critical",
                    "title": "Critical Sentiment Analysis Alert"
                },
                {
                    "type": AutomationAction.ESCALATE_TICKET,
                    "escalation_level": "medical_specialist",
                    "reason": "Critical negative sentiment detected"
                }
            ],
            enabled=True,
            priority=1
        )
        
        # Auto-assignment rule
        auto_assign_rule = AutomationRule(
            id="AUTO_006",
            name="Auto-Assignment",
            description="Automatically assign tickets to appropriate teams",
            trigger=AutomationTrigger.TICKET_CREATED,
            conditions={
                "assigned_to": None
            },
            actions=[
                {
                    "type": AutomationAction.ASSIGN_AUTO,
                    "assignment_rules": {
                        "medical_emergency": "emergency_medical_team",
                        "technical_issue": "technical_support",
                        "compliance_issue": "compliance_team",
                        "integration_issue": "integration_team"
                    }
                }
            ],
            enabled=True,
            priority=5
        )
        
        # Knowledge base creation rule
        kb_rule = AutomationRule(
            id="AUTO_007",
            name="Knowledge Base Article Creation",
            description="Create KB articles from common ticket patterns",
            trigger=AutomationTrigger.TICKET_CREATED,
            conditions={
                "category": "technical_issue",
                "frequency_threshold": 5,  # If similar issue occurs 5+ times
                "resolution_exists": True
            },
            actions=[
                {
                    "type": AutomationAction.CREATE_KB_ARTICLE,
                    "template": "troubleshooting",
                    "auto_publish": False
                }
            ],
            enabled=True,
            priority=6
        )
        
        self.automation_rules = {
            rule.id: rule for rule in [
                emergency_ticket_rule,
                critical_feedback_rule,
                health_check_rule,
                sla_rule,
                sentiment_rule,
                auto_assign_rule,
                kb_rule
            ]
        }
    
    async def process_automation_trigger(
        self,
        trigger: AutomationTrigger,
        context_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process automation trigger and execute applicable rules"""
        
        automation_context = AutomationContext(
            trigger_data=context_data,
            timestamp=datetime.now(),
            source_system=trigger.value,
            priority_level=context_data.get("priority", PriorityLevel.STANDARD_MEDICAL),
            facility_context=context_data.get("facility_context", {}),
            user_context=context_data.get("user_context", {})
        )
        
        executed_rules = []
        
        # Find applicable rules
        applicable_rules = self._find_applicable_rules(trigger, automation_context)
        
        # Execute rules in priority order
        for rule in sorted(applicable_rules, key=lambda x: x.priority):
            try:
                await self._execute_automation_rule(rule, automation_context)
                executed_rules.append({
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "executed_at": datetime.now().isoformat(),
                    "actions_taken": len(rule.actions)
                })
                
                # Update rule execution history
                rule.last_executed = datetime.now()
                rule.execution_count += 1
                
            except Exception as e:
                logger.error(f"Error executing automation rule {rule.id}: {str(e)}")
                executed_rules.append({
                    "rule_id": rule.id,
                    "error": str(e),
                    "executed_at": datetime.now().isoformat()
                })
        
        # Log execution
        self.execution_history.append({
            "trigger": trigger.value,
            "context": asdict(automation_context),
            "executed_rules": executed_rules,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Automation processed for trigger {trigger.value}: {len(executed_rules)} rules executed")
        return executed_rules
    
    def _find_applicable_rules(
        self,
        trigger: AutomationTrigger,
        context: AutomationContext
    ) -> List[AutomationRule]:
        """Find automation rules applicable to the trigger and context"""
        
        applicable_rules = []
        
        for rule in self.automation_rules.values():
            if not rule.enabled or rule.trigger != trigger:
                continue
            
            # Check conditions
            if self._evaluate_conditions(rule.conditions, context):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        context: AutomationContext
    ) -> bool:
        """Evaluate if conditions are met for automation rule"""
        
        trigger_data = context.trigger_data
        
        # Priority condition
        if "priority" in conditions:
            priority_list = conditions["priority"]
            context_priority = context.priority_level.value
            if context_priority not in priority_list:
                return False
        
        # Category condition
        if "category" in conditions:
            category_list = conditions["category"]
            context_category = trigger_data.get("category", "")
            if context_category not in category_list:
                return False
        
        # Keywords condition
        if "keywords" in conditions:
            required_keywords = conditions["keywords"]
            content = trigger_data.get("content", "")
            if not any(keyword in content.lower() for keyword in required_keywords):
                return False
        
        # Boolean conditions
        boolean_conditions = [
            "patient_safety_mentioned", "emergency_situation", 
            "system_performance_issues", "assigned_to"
        ]
        
        for condition in boolean_conditions:
            if condition in conditions:
                context_value = trigger_data.get(condition, False)
                if context_value != conditions[condition]:
                    return False
        
        # Sentiment conditions
        if "sentiment" in conditions:
            sentiment_list = conditions["sentiment"]
            context_sentiment = trigger_data.get("sentiment", "")
            if context_sentiment not in sentiment_list:
                return False
        
        # Urgency conditions
        if "urgency" in conditions:
            urgency_list = conditions["urgency"]
            context_urgency = trigger_data.get("urgency", "")
            if context_urgency not in urgency_list:
                return False
        
        # Status conditions
        if "status" in conditions:
            status_list = conditions["status"]
            context_status = trigger_data.get("status", "")
            if context_status not in status_list:
                return False
        
        # Component type conditions
        if "component_type" in conditions:
            component_type_list = conditions["component_type"]
            context_component_type = trigger_data.get("component_type", "")
            if context_component_type not in component_type_list:
                return False
        
        return True
    
    async def _execute_automation_rule(
        self,
        rule: AutomationRule,
        context: AutomationContext
    ) -> None:
        """Execute automation rule actions"""
        
        for action_config in rule.actions:
            action_type = AutomationAction(action_config["type"])
            
            try:
                if action_type == AutomationAction.SEND_NOTIFICATION:
                    await self._send_notification(action_config, context)
                
                elif action_type == AutomationAction.CREATE_INCIDENT:
                    await self._create_incident(action_config, context)
                
                elif action_type == AutomationAction.ESCALATE_TICKET:
                    await self._escalate_ticket(action_config, context)
                
                elif action_type == AutomationAction.ASSIGN_AUTO:
                    await self._auto_assign_ticket(action_config, context)
                
                elif action_type == AutomationAction.ALERT_TEAM:
                    await self._alert_team(action_config, context)
                
                elif action_type == AutomationAction.UPDATE_STATUS:
                    await self._update_status(action_config, context)
                
                elif action_type == AutomationAction.SEND_SURVEY:
                    await self._send_survey(action_config, context)
                
                elif action_type == AutomationAction.CREATE_KB_ARTICLE:
                    await self._create_kb_article(action_config, context)
                
            except Exception as e:
                logger.error(f"Error executing action {action_type.value} for rule {rule.id}: {str(e)}")
    
    async def _send_notification(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Send notification based on action configuration"""
        
        recipients = action_config.get("recipients", [])
        message_template = action_config.get("message", "")
        priority = action_config.get("priority", "normal")
        
        # Format message with context data
        message = self._format_message(message_template, context.trigger_data)
        
        # Add to notification queue
        notification = {
            "id": f"NOTIF_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "recipients": recipients,
            "message": message,
            "priority": priority,
            "source": context.source_system,
            "timestamp": datetime.now().isoformat(),
            "context": context.trigger_data
        }
        
        self.notification_queue.append(notification)
        
        # In production, this would actually send notifications
        logger.info(f"Notification queued: {notification['id']} to {recipients}")
        
        # Simulate sending notification
        await self._process_notification_queue()
    
    async def _create_incident(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Create incident based on action configuration"""
        
        incident_type = action_config.get("incident_type", "general")
        severity = action_config.get("severity", "sev3_medium")
        title_template = action_config.get("title", "Automated Incident")
        
        # Format title with context
        title = self._format_message(title_template, context.trigger_data)
        description = f"Automatically created from {context.source_system} trigger"
        
        try:
            incident = await incident_system.create_incident(
                title=title,
                description=description,
                incident_type=incident_type,
                severity=severity,
                reporter_id="automation_system",
                reporter_name="Automation System",
                reporter_facility=context.facility_context.get("facility_name", "Unknown")
            )
            
            logger.info(f"Automatically created incident: {incident.id}")
            
        except Exception as e:
            logger.error(f"Error creating incident: {str(e)}")
    
    async def _escalate_ticket(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Escalate ticket based on action configuration"""
        
        escalation_level = action_config.get("escalation_level")
        reason = action_config.get("reason", "Automated escalation")
        
        ticket_id = context.trigger_data.get("ticket_id")
        if ticket_id and ticket_id in ticket_system.tickets:
            try:
                from incident_management.emergency_response import EscalationLevel
                level = EscalationLevel(escalation_level)
                
                await ticket_system.escalation_manager.escalate_ticket(
                    ticket_system.tickets[ticket_id],
                    reason
                )
                
                logger.info(f"Automatically escalated ticket {ticket_id} to {escalation_level}")
                
            except Exception as e:
                logger.error(f"Error escalating ticket: {str(e)}")
    
    async def _auto_assign_ticket(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Automatically assign ticket to appropriate team"""
        
        assignment_rules = action_config.get("assignment_rules", {})
        category = context.trigger_data.get("category", "")
        
        ticket_id = context.trigger_data.get("ticket_id")
        if ticket_id and ticket_id in ticket_system.tickets:
            ticket = ticket_system.tickets[ticket_id]
            
            # Find appropriate team based on category
            assigned_team = assignment_rules.get(category, "healthcare_support")
            
            ticket.assigned_team = assigned_team
            ticket.status = "assigned"
            ticket.updated_at = datetime.now()
            
            logger.info(f"Auto-assigned ticket {ticket_id} to {assigned_team}")
    
    async def _alert_team(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Alert specific teams"""
        
        teams = action_config.get("teams", [])
        alert_type = action_config.get("alert_type", "general")
        
        alert = {
            "id": f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "teams": teams,
            "alert_type": alert_type,
            "source": context.source_system,
            "context": context.trigger_data,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"Team alert triggered: {alert_type} for teams {teams}")
        
        # In production, this would send actual team alerts
        # (Slack, Teams, email, SMS, etc.)
    
    async def _update_status(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Update status based on action configuration"""
        
        # Implementation would depend on specific status update requirements
        logger.info(f"Status update action configured: {action_config}")
    
    async def _send_survey(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Send survey based on action configuration"""
        
        survey_type = action_config.get("survey_type", "satisfaction")
        trigger_type = context.trigger_data.get("trigger_type")
        
        # Create appropriate feedback collection
        feedback = await feedback_system.collect_feedback(
            feedback_type=FeedbackType.POST_INTERACTION,
            user_id=context.user_context.get("user_id", ""),
            user_name=context.user_context.get("user_name", ""),
            user_facility=context.facility_context.get("facility_name", ""),
            user_role=context.user_context.get("user_role", ""),
            content=f"Post-{trigger_type} satisfaction survey",
            medical_context=f"{trigger_type}_follow_up"
        )
        
        logger.info(f"Sent survey to user {feedback.user_name}")
    
    async def _create_kb_article(self, action_config: Dict[str, Any], context: AutomationContext) -> None:
        """Create knowledge base article based on automation"""
        
        template = action_config.get("template", "general")
        auto_publish = action_config.get("auto_publish", False)
        
        # This would analyze ticket patterns and create KB articles
        # For now, just log the action
        logger.info(f"Knowledge base article creation triggered with template: {template}")
    
    def _format_message(self, template: str, data: Dict[str, Any]) -> str:
        """Format message template with data"""
        
        try:
            return template.format(**data)
        except KeyError:
            # Return template with missing keys as placeholders
            return template
    
    async def _process_notification_queue(self) -> None:
        """Process queued notifications"""
        
        if not self.notification_queue:
            return
        
        notifications_to_process = self.notification_queue[:]
        self.notification_queue.clear()
        
        for notification in notifications_to_process:
            try:
                # In production, this would actually send notifications
                # via email, SMS, Slack, Teams, etc.
                
                logger.info(f"Processing notification: {notification['id']} - {notification['message'][:50]}...")
                
                # Simulate successful sending
                await asyncio.sleep(0.1)  # Simulate network delay
                
            except Exception as e:
                logger.error(f"Error processing notification {notification['id']}: {str(e)}")
                # Re-queue failed notifications
                self.notification_queue.append(notification)
    
    async def get_automation_stats(self) -> Dict[str, Any]:
        """Get automation statistics and performance metrics"""
        
        # Calculate recent execution stats
        last_24h = datetime.now() - timedelta(hours=24)
        recent_executions = [
            record for record in self.execution_history
            if datetime.fromisoformat(record["timestamp"]) >= last_24h
        ]
        
        # Rule performance
        rule_stats = {}
        for rule in self.automation_rules.values():
            rule_stats[rule.id] = {
                "name": rule.name,
                "enabled": rule.enabled,
                "execution_count": rule.execution_count,
                "last_executed": rule.last_executed.isoformat() if rule.last_executed else None
            }
        
        # Trigger statistics
        trigger_stats = {}
        for execution in recent_executions:
            trigger = execution["trigger"]
            trigger_stats[trigger] = trigger_stats.get(trigger, 0) + 1
        
        return {
            "total_rules": len(self.automation_rules),
            "enabled_rules": len([r for r in self.automation_rules.values() if r.enabled]),
            "recent_executions_24h": len(recent_executions),
            "rule_performance": rule_stats,
            "trigger_statistics": trigger_stats,
            "notification_queue_size": len(self.notification_queue)
        }
    
    def add_custom_rule(self, rule: AutomationRule) -> None:
        """Add custom automation rule"""
        self.automation_rules[rule.id] = rule
        logger.info(f"Added custom automation rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove automation rule"""
        if rule_id in self.automation_rules:
            rule_name = self.automation_rules[rule_id].name
            del self.automation_rules[rule_id]
            logger.info(f"Removed automation rule: {rule_name}")
            return True
        return False
    
    def toggle_rule(self, rule_id: str) -> bool:
        """Toggle automation rule enabled/disabled state"""
        if rule_id in self.automation_rules:
            rule = self.automation_rules[rule_id]
            rule.enabled = not rule.enabled
            logger.info(f"Toggled rule {rule.name}: {'enabled' if rule.enabled else 'disabled'}")
            return True
        return False

# Global automation engine instance
automation_engine = ResponseAutomationEngine()

# Example usage and testing functions
async def test_automation_engine():
    """Test the automation engine with sample triggers"""
    
    # Test emergency ticket creation
    emergency_context = {
        "ticket_id": "TKT-20241104-TEST",
        "priority": "emergency",
        "category": "medical_emergency",
        "content": "Emergency cardiac arrest situation",
        "facility_context": {"facility_name": "General Hospital"},
        "user_context": {"user_id": "dr_smith", "user_name": "Dr. Smith"}
    }
    
    await automation_engine.process_automation_trigger(
        AutomationTrigger.TICKET_CREATED,
        emergency_context
    )
    
    # Test critical feedback
    critical_feedback_context = {
        "feedback_id": "FB-20241104-TEST",
        "patient_safety_mentioned": True,
        "sentiment": "very_negative",
        "urgency": "critical",
        "content": "System failure caused patient safety risk",
        "facility_context": {"facility_name": "Heart Center"},
        "user_context": {"user_id": "nurse_jones", "user_name": "Nurse Jones"}
    }
    
    await automation_engine.process_automation_trigger(
        AutomationTrigger.FEEDBACK_SUBMITTED,
        critical_feedback_context
    )
    
    # Get automation stats
    stats = await automation_engine.get_automation_stats()
    print(f"Automation Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_automation_engine())