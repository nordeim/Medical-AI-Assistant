"""
Intervention Manager
Manages customer interventions and retention actions
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class InterventionStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InterventionPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class InterventionType(Enum):
    RETENTION = "retention"
    ENGAGEMENT = "engagement"
    EXPANSION = "expansion"
    SUPPORT = "support"
    ONBOARDING = "onboarding"


@dataclass
class Intervention:
    intervention_id: str
    customer_id: str
    intervention_type: InterventionType
    priority: InterventionPriority
    status: InterventionStatus
    trigger_reason: str
    recommended_actions: List[str]
    assigned_to: Optional[str]
    due_date: datetime
    created_at: datetime
    completed_at: Optional[datetime] = None
    success_metrics: Dict[str, Any] = None
    notes: List[str] = None


class InterventionManager:
    """
    Manages customer interventions and retention actions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Intervention management
        self.active_interventions = {}
        self.completed_interventions = {}
        self.intervention_playbooks = self.load_intervention_playbooks()
        self.intervention_templates = self.load_intervention_templates()
        
        # Performance tracking
        self.intervention_metrics = {
            "total_interventions": 0,
            "successful_interventions": 0,
            "average_resolution_time": 0,
            "success_rate": 0,
            "customer_satisfaction": 0
        }
        
        # Initialize intervention tracking
        self.initialize_intervention_tracking()
        
        self.logger.info("Intervention Manager initialized")
    
    def load_intervention_playbooks(self) -> Dict[str, Any]:
        """Load intervention playbooks and procedures"""
        return {
            "retention_interventions": {
                "high_churn_risk": {
                    "description": "Immediate intervention for high churn risk customers",
                    "trigger_conditions": ["churn_risk > 0.8", "health_score < 40"],
                    "intervention_steps": [
                        "assign_dedicated_success_manager",
                        "schedule_executive_call_within_24h",
                        "conduct_comprehensive_health_assessment",
                        "create_personalized_retention_plan",
                        "implement_immediate_solutions",
                        "schedule_daily_check_ins",
                        "monitor_health_score_improvement"
                    ],
                    "escalation_triggers": ["health_score_declines", "customer_unresponsive", "executive_approval_needed"],
                    "success_metrics": ["health_score_improvement", "churn_risk_reduction", "customer_satisfaction"],
                    "timeline": "immediate",
                    "resources_required": ["success_manager", "executive_sponsor", "solution_architect"]
                },
                "low_engagement": {
                    "description": "Re-engage customers with low platform usage",
                    "trigger_conditions": ["engagement_score < 60", "usage_decline > 30%"],
                    "intervention_steps": [
                        "analyze_usage_patterns",
                        "identify_engagement_barriers",
                        "send_personalized_re_engagement_email",
                        "provide_usage_optimization_tips",
                        "offer_additional_training",
                        "schedule_success_check_in",
                        "share_relevant_case_studies"
                    ],
                    "escalation_triggers": ["engagement_continues_declining", "customer_unreachable"],
                    "success_metrics": ["engagement_score_increase", "usage_frequency_improvement", "feature_adoption"],
                    "timeline": "1-3 days",
                    "resources_required": ["success_manager", "training_resources"]
                },
                "payment_issues": {
                    "description": "Address payment and billing issues",
                    "trigger_conditions": ["payment_delay > 7 days", "billing_complaint"],
                    "intervention_steps": [
                        "review_payment_history",
                        "contact_customer_about_payment",
                        "offer_payment_plan_if_needed",
                        "resolve_billing_dispute",
                        "provide_payment_confirmation",
                        "follow_up_on_payment_completion"
                    ],
                    "escalation_triggers": ["payment_unresponsive", "billing_dispute_escalation"],
                    "success_metrics": ["payment_recovery", "billing_satisfaction", "payment_delay_resolution"],
                    "timeline": "1-5 days",
                    "resources_required": ["billing_team", "success_manager"]
                }
            },
            "engagement_interventions": {
                "feature_underutilization": {
                    "description": "Increase feature adoption and usage",
                    "trigger_conditions": ["feature_adoption < 50%", "available_features_unused"],
                    "intervention_steps": [
                        "analyze_current_feature_usage",
                        "identify_high_value_unused_features",
                        "create_personalized_feature_tour",
                        "provide_feature_specific_training",
                        "demonstrate_feature_benefits",
                        "schedule_feature_adoption_follow_up"
                    ],
                    "escalation_triggers": ["feature_adoption_stagnant", "customer_training_needed"],
                    "success_metrics": ["feature_adoption_rate", "feature_usage_frequency", "productivity_improvement"],
                    "timeline": "1-2 weeks",
                    "resources_required": ["training_specialist", "product_specialist"]
                },
                "support_dependency": {
                    "description": "Reduce excessive support dependency",
                    "trigger_conditions": ["support_tickets > 5/month", "basic_questions_frequent"],
                    "intervention_steps": [
                        "analyze_support_ticket_patterns",
                        "identify_knowledge_gaps",
                        "provide_self_service_resources",
                        "offer_comprehensive_training",
                        "create_custom_documentation",
                        "establish_support_schedule"
                    ],
                    "escalation_triggers": ["support_tickets_increase", "customer_satisfaction_declines"],
                    "success_metrics": ["support_ticket_reduction", "self_service_usage", "knowledge_retention"],
                    "timeline": "2-4 weeks",
                    "resources_required": ["training_specialist", "documentation_team"]
                }
            },
            "expansion_interventions": {
                "upsell_opportunity": {
                    "description": "Present upsell opportunities to qualified customers",
                    "trigger_conditions": ["high_usage", "feature_requirements_exceeded", "expansion_ready"],
                    "intervention_steps": [
                        "analyze_current_usage_vs_plan",
                        "identify_specific_expansion_needs",
                        "create_customized_expansion_proposal",
                        "present_roi_demonstration",
                        "schedule_sales_discussion",
                        "negotiate_expansion_terms",
                        "implement_expanded_solution"
                    ],
                    "escalation_triggers": ["expansion_interest", "complex_requirements", "competitive_threat"],
                    "success_metrics": ["expansion_rate", "revenue_growth", "customer_satisfaction"],
                    "timeline": "2-6 weeks",
                    "resources_required": ["account_executive", "solution_consultant"]
                },
                "strategic_partnership": {
                    "description": "Develop strategic partnership opportunities",
                    "trigger_conditions": ["strategic_customer", "partnership_potential", "expansion_vision"],
                    "intervention_steps": [
                        "assess_partnership_potential",
                        "identify_partnership_opportunities",
                        "create_partnership_proposal",
                        "present_strategic_value",
                        "negotiate_partnership_terms",
                        "establish_partnership_governance"
                    ],
                    "escalation_triggers": ["executive_involvement", "legal_review", "competitive_partnerships"],
                    "success_metrics": ["partnership_establishment", "strategic_value_delivery", "long_term_engagement"],
                    "timeline": "1-3 months",
                    "resources_required": ["executive_team", "partnership_manager", "legal_team"]
                }
            },
            "support_interventions": {
                "technical_issues": {
                    "description": "Resolve ongoing technical issues",
                    "trigger_conditions": ["recurring_support_tickets", "system_performance_issues"],
                    "intervention_steps": [
                        "analyze_technical_issue_patterns",
                        "escalate_to_technical_team",
                        "create_resolution_plan",
                        "implement_solution",
                        "verify_issue_resolution",
                        "prevent_future_occurrences"
                    ],
                    "escalation_triggers": ["complex_technical_issue", "system_wide_impact", "customer_executive_involvement"],
                    "success_metrics": ["issue_resolution_time", "recurrence_prevention", "customer_satisfaction"],
                    "timeline": "1-7 days",
                    "resources_required": ["technical_team", "engineering_team"]
                },
                "user_experience": {
                    "description": "Improve user experience and satisfaction",
                    "trigger_conditions": ["negative_feedback", "user_experience_issues"],
                    "intervention_steps": [
                        "collect_detailed_feedback",
                        "analyze_user_journey",
                        "identify_experience_improvements",
                        "implement_quick_wins",
                        "plan_larger_improvements",
                        "follow_up_on_satisfaction"
                    ],
                    "escalation_triggers": ["executive_complaint", "public_feedback", "retention_risk"],
                    "success_metrics": ["satisfaction_score", "nps_improvement", "issue_resolution"],
                    "timeline": "1-4 weeks",
                    "resources_required": ["user_experience_team", "product_team"]
                }
            }
        }
    
    def load_intervention_templates(self) -> Dict[str, Any]:
        """Load intervention templates and communication scripts"""
        return {
            "communication_templates": {
                "initial_outreach": {
                    "email_subject": "Re: Important Update for Your Account - {company_name}",
                    "email_body": """Dear {customer_name},

I hope this email finds you well. I'm reaching out because I want to ensure that {company_name} continues to get maximum value from our platform.

Based on our recent analysis, I've identified some opportunities that could help improve your experience and results. I'd love to schedule a brief call to discuss these findings and how we can best support your continued success.

Would you have 30 minutes available this week for a quick conversation?

Best regards,
{success_manager}
Customer Success Manager""",
                    "phone_script": """Hi {customer_name}, this is {success_manager} from {company}. I wanted to personally reach out because I've been reviewing your account and I think there are some exciting opportunities we could explore together to help {company_name} achieve even better results. Do you have a few minutes to chat about this?"""
                },
                "intervention_follow_up": {
                    "email_subject": "Following Up on Our Recent Conversation - {company_name}",
                    "email_body": """Dear {customer_name},

Thank you for taking the time to speak with me recently. I wanted to follow up on the action items we discussed:

{action_items}

Please let me know if you have any questions or if there's anything else I can do to support your success.

I'm here to help ensure {company_name} continues to get the most value from our platform.

Best regards,
{success_manager}""",
                    "phone_script": """Hi {customer_name}, this is {success_manager}. I wanted to follow up on our recent conversation about {topic}. How are the action items coming along? Is there anything I can do to help you move forward?"""
                },
                "success_celebration": {
                    "email_subject": "Congratulations! {company_name} Success Milestone Reached 🎉",
                    "email_body": """Dear {customer_name},

Congratulations! {company_name} has reached an important milestone:

{milestone_description}

This achievement demonstrates the value {company_name} is getting from our platform, and I'm excited to see what else we can accomplish together.

Would you be interested in sharing your success story with other customers? It could be a great way to showcase your achievements and inspire others.

Thank you for being a valued customer!

Best regards,
{success_manager}""",
                    "phone_script": """Hi {customer_name}, congratulations on reaching {milestone}! I wanted to personally congratulate you and see if you'd be interested in sharing your success story. Other customers could really benefit from learning about your approach."""
                }
            },
            "escalation_templates": {
                "executive_escalation": {
                    "escalation_reason": "Customer satisfaction and retention risk identified",
                    "priority": "high",
                    "required_actions": [
                        "Schedule executive-level meeting within 48 hours",
                        "Prepare comprehensive account review",
                        "Develop retention proposal with incentives",
                        "Assign executive sponsor for relationship management"
                    ],
                    "success_criteria": [
                        "Customer satisfaction improvement",
                        "Retention commitment secured",
                        "Satisfaction plan implemented",
                        "Executive relationship established"
                    ]
                },
                "technical_escalation": {
                    "escalation_reason": "Technical issues affecting customer success",
                    "priority": "high",
                    "required_actions": [
                        "Immediate technical team involvement",
                        "Root cause analysis",
                        "Permanent solution implementation",
                        "Customer communication and expectation management"
                    ],
                    "success_criteria": [
                        "Technical issues resolved",
                        "Prevention measures implemented",
                        "Customer satisfaction restored",
                        "System stability verified"
                    ]
                }
            },
            "assessment_templates": {
                "health_assessment": {
                    "areas_to_assess": [
                        "Product Usage and Adoption",
                        "Customer Satisfaction",
                        "Business Value Realization",
                        "Stakeholder Engagement",
                        "Support Experience",
                        "Billing and Contract Management"
                    ],
                    "scoring_criteria": {
                        "usage": "How effectively is the customer using the product?",
                        "satisfaction": "How satisfied is the customer with the product and service?",
                        "value": "Is the customer realizing business value from the product?",
                        "engagement": "How engaged are the key stakeholders?",
                        "support": "Is the support experience meeting expectations?",
                        "billing": "Are billing and contract matters handled smoothly?"
                    }
                }
            }
        }
    
    def initialize_intervention_tracking(self):
        """Initialize intervention tracking database"""
        try:
            db_path = self.data_dir / "interventions.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create interventions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interventions (
                    intervention_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    intervention_type TEXT,
                    priority TEXT,
                    status TEXT,
                    trigger_reason TEXT,
                    recommended_actions TEXT,
                    assigned_to TEXT,
                    due_date TIMESTAMP,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    success_metrics TEXT,
                    notes TEXT
                )
            ''')
            
            # Create intervention_history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intervention_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intervention_id TEXT,
                    action TEXT,
                    timestamp TIMESTAMP,
                    user_id TEXT,
                    notes TEXT,
                    FOREIGN KEY (intervention_id) REFERENCES interventions (intervention_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Intervention tracking database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing intervention tracking: {str(e)}")
    
    def trigger_intervention(self, customer_id: str, intervention_type: InterventionType, 
                           trigger_reason: str, context: Dict[str, Any]) -> Optional[str]:
        """Trigger intervention for a customer"""
        try:
            # Generate intervention ID
            intervention_id = f"{intervention_type.value}_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Determine priority based on trigger
            priority = self.determine_intervention_priority(trigger_reason, context)
            
            # Get recommended actions
            recommended_actions = self.get_recommended_actions(intervention_type, trigger_reason, context)
            
            # Set due date based on priority
            due_date = self.calculate_due_date(priority)
            
            # Assign intervention
            assigned_to = self.assign_intervention_owner(customer_id, intervention_type, priority)
            
            # Create intervention
            intervention = Intervention(
                intervention_id=intervention_id,
                customer_id=customer_id,
                intervention_type=intervention_type,
                priority=priority,
                status=InterventionStatus.PENDING,
                trigger_reason=trigger_reason,
                recommended_actions=recommended_actions,
                assigned_to=assigned_to,
                due_date=due_date,
                created_at=datetime.now(),
                success_metrics={},
                notes=[]
            )
            
            # Store intervention
            self.active_interventions[intervention_id] = intervention
            self.save_intervention_to_db(intervention)
            
            # Log intervention trigger
            self.log_intervention_action(intervention_id, "INTERVENTION_TRIGGERED", 
                                       f"Intervention triggered: {trigger_reason}")
            
            # Execute immediate actions if critical
            if priority == InterventionPriority.CRITICAL:
                self.execute_immediate_actions(intervention)
            
            # Update metrics
            self.intervention_metrics["total_interventions"] += 1
            
            self.logger.info(f"Intervention {intervention_id} triggered for customer {customer_id}")
            return intervention_id
            
        except Exception as e:
            self.logger.error(f"Error triggering intervention: {str(e)}")
            return None
    
    def determine_intervention_priority(self, trigger_reason: str, context: Dict[str, Any]) -> InterventionPriority:
        """Determine intervention priority based on trigger and context"""
        # Critical priority triggers
        if any(keyword in trigger_reason.lower() for keyword in ["critical", "immediate", "churn risk > 0.8", "cancelling"]):
            return InterventionPriority.CRITICAL
        
        # High priority triggers
        if any(keyword in trigger_reason.lower() for keyword in ["high churn risk", "payment issue", "executive complaint"]):
            return InterventionPriority.HIGH
        
        # Medium priority triggers
        if any(keyword in trigger_reason.lower() for keyword in ["low engagement", "support issue", "feature underutilization"]):
            return InterventionPriority.MEDIUM
        
        # Default to medium
        return InterventionPriority.MEDIUM
    
    def get_recommended_actions(self, intervention_type: InterventionType, 
                              trigger_reason: str, context: Dict[str, Any]) -> List[str]:
        """Get recommended actions for intervention"""
        try:
            actions = []
            
            # Map intervention type to playbook category
            playbook_category_map = {
                InterventionType.RETENTION: "retention_interventions",
                InterventionType.ENGAGEMENT: "engagement_interventions",
                InterventionType.EXPANSION: "expansion_interventions",
                InterventionType.SUPPORT: "support_interventions",
                InterventionType.ONBOARDING: "onboarding_interventions"
            }
            
            playbook_category = playbook_category_map.get(intervention_type, "retention_interventions")
            playbooks = self.intervention_playbooks.get(playbook_category, {})
            
            # Find matching playbook
            for playbook_key, playbook in playbooks.items():
                trigger_conditions = playbook.get("trigger_conditions", [])
                
                # Check if trigger matches any conditions
                if any(condition in trigger_reason.lower() for condition in trigger_conditions):
                    actions.extend(playbook.get("intervention_steps", []))
                    break
            
            # Add generic actions if no specific playbook found
            if not actions:
                actions = self.get_generic_actions(intervention_type, trigger_reason)
            
            return actions[:10]  # Return top 10 actions
            
        except Exception as e:
            self.logger.error(f"Error getting recommended actions: {str(e)}")
            return ["Contact customer", "Assess situation", "Develop action plan"]
    
    def get_generic_actions(self, intervention_type: InterventionType, trigger_reason: str) -> List[str]:
        """Get generic actions when no specific playbook matches"""
        generic_actions = {
            InterventionType.RETENTION: [
                "Contact customer to understand concerns",
                "Conduct health assessment",
                "Develop retention plan",
                "Schedule follow-up meetings",
                "Monitor improvement metrics"
            ],
            InterventionType.ENGAGEMENT: [
                "Analyze engagement patterns",
                "Identify engagement barriers",
                "Create re-engagement plan",
                "Provide additional training",
                "Monitor engagement improvement"
            ],
            InterventionType.EXPANSION: [
                "Assess expansion readiness",
                "Identify expansion opportunities",
                "Create expansion proposal",
                "Schedule sales discussion",
                "Implement expansion if approved"
            ],
            InterventionType.SUPPORT: [
                "Analyze support needs",
                "Provide technical assistance",
                "Offer additional training",
                "Improve support experience",
                "Follow up on satisfaction"
            ],
            InterventionType.ONBOARDING: [
                "Review onboarding progress",
                "Provide additional guidance",
                "Accelerate time to value",
                "Ensure successful adoption",
                "Transition to success management"
            ]
        }
        
        return generic_actions.get(intervention_type, ["Contact customer", "Assess situation"])
    
    def calculate_due_date(self, priority: InterventionPriority) -> datetime:
        """Calculate due date based on priority"""
        current_time = datetime.now()
        
        if priority == InterventionPriority.CRITICAL:
            return current_time + timedelta(hours=24)
        elif priority == InterventionPriority.HIGH:
            return current_time + timedelta(days=3)
        elif priority == InterventionPriority.MEDIUM:
            return current_time + timedelta(days=7)
        else:
            return current_time + timedelta(days=14)
    
    def assign_intervention_owner(self, customer_id: str, intervention_type: InterventionType, 
                                priority: InterventionPriority) -> str:
        """Assign intervention owner based on customer and intervention characteristics"""
        # In real implementation, would use assignment logic
        # For demo, return generic assignment
        
        if priority == InterventionPriority.CRITICAL:
            return "senior_success_manager"
        elif intervention_type == InterventionType.EXPANSION:
            return "account_executive"
        elif intervention_type == InterventionType.SUPPORT:
            return "technical_success_manager"
        else:
            return "customer_success_manager"
    
    def execute_immediate_actions(self, intervention: Intervention):
        """Execute immediate actions for critical interventions"""
        try:
            # Log immediate action execution
            self.log_intervention_action(
                intervention.intervention_id, 
                "IMMEDIATE_ACTIONS_EXECUTED", 
                "Critical intervention - immediate actions initiated"
            )
            
            # In real implementation, this would:
            # - Send immediate alerts to assigned person
            # - Create calendar events for meetings
            # - Trigger notification workflows
            # - Start monitoring processes
            
            self.logger.info(f"Immediate actions executed for intervention {intervention.intervention_id}")
            
        except Exception as e:
            self.logger.error(f"Error executing immediate actions: {str(e)}")
    
    def process_pending_interventions(self) -> int:
        """Process all pending interventions"""
        try:
            processed_count = 0
            
            for intervention_id, intervention in list(self.active_interventions.items()):
                if intervention.status == InterventionStatus.PENDING:
                    # Update status
                    intervention.status = InterventionStatus.IN_PROGRESS
                    
                    # Log status change
                    self.log_intervention_action(
                        intervention_id, 
                        "STATUS_UPDATED", 
                        "Status changed to in-progress"
                    )
                    
                    processed_count += 1
            
            self.logger.info(f"Processed {processed_count} pending interventions")
            return processed_count
            
        except Exception as e:
            self.logger.error(f"Error processing pending interventions: {str(e)}")
            return 0
    
    def complete_intervention(self, intervention_id: str, success: bool, 
                            completion_notes: str = "", metrics: Dict[str, Any] = None) -> bool:
        """Mark intervention as completed"""
        try:
            if intervention_id not in self.active_interventions:
                self.logger.error(f"Intervention {intervention_id} not found")
                return False
            
            intervention = self.active_interventions[intervention_id]
            
            # Update intervention
            intervention.status = InterventionStatus.COMPLETED if success else InterventionStatus.FAILED
            intervention.completed_at = datetime.now()
            
            if completion_notes:
                intervention.notes.append(completion_notes)
            
            if metrics:
                intervention.success_metrics = metrics
            
            # Move to completed interventions
            self.completed_interventions[intervention_id] = intervention
            del self.active_interventions[intervention_id]
            
            # Update database
            self.update_intervention_in_db(intervention)
            
            # Log completion
            self.log_intervention_action(
                intervention_id,
                "INTERVENTION_COMPLETED" if success else "INTERVENTION_FAILED",
                completion_notes or "Intervention completed"
            )
            
            # Update metrics
            self.intervention_metrics["successful_interventions"] += 1 if success else 0
            
            self.logger.info(f"Intervention {intervention_id} completed with status: {intervention.status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing intervention: {str(e)}")
            return False
    
    def save_intervention_to_db(self, intervention: Intervention):
        """Save intervention to database"""
        try:
            db_path = self.data_dir / "interventions.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO interventions (
                    intervention_id, customer_id, intervention_type, priority, status,
                    trigger_reason, recommended_actions, assigned_to, due_date,
                    created_at, completed_at, success_metrics, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                intervention.intervention_id,
                intervention.customer_id,
                intervention.intervention_type.value,
                intervention.priority.value,
                intervention.status.value,
                intervention.trigger_reason,
                json.dumps(intervention.recommended_actions),
                intervention.assigned_to,
                intervention.due_date,
                intervention.created_at,
                intervention.completed_at,
                json.dumps(intervention.success_metrics) if intervention.success_metrics else None,
                json.dumps(intervention.notes) if intervention.notes else None
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error saving intervention to database: {str(e)}")
    
    def update_intervention_in_db(self, intervention: Intervention):
        """Update intervention in database"""
        self.save_intervention_to_db(intervention)  # Same as save for now
    
    def log_intervention_action(self, intervention_id: str, action: str, notes: str, user_id: str = None):
        """Log intervention action to history"""
        try:
            db_path = self.data_dir / "interventions.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO intervention_history (
                    intervention_id, action, timestamp, user_id, notes
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                intervention_id,
                action,
                datetime.now(),
                user_id,
                notes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging intervention action: {str(e)}")
    
    def get_intervention_status(self, intervention_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific intervention"""
        try:
            # Check active interventions
            if intervention_id in self.active_interventions:
                intervention = self.active_interventions[intervention_id]
                return self.format_intervention_status(intervention)
            
            # Check completed interventions
            if intervention_id in self.completed_interventions:
                intervention = self.completed_interventions[intervention_id]
                return self.format_intervention_status(intervention)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting intervention status: {str(e)}")
            return None
    
    def format_intervention_status(self, intervention: Intervention) -> Dict[str, Any]:
        """Format intervention status for display"""
        return {
            "intervention_id": intervention.intervention_id,
            "customer_id": intervention.customer_id,
            "intervention_type": intervention.intervention_type.value,
            "priority": intervention.priority.value,
            "status": intervention.status.value,
            "trigger_reason": intervention.trigger_reason,
            "recommended_actions": intervention.recommended_actions,
            "assigned_to": intervention.assigned_to,
            "due_date": intervention.due_date.isoformat(),
            "created_at": intervention.created_at.isoformat(),
            "completed_at": intervention.completed_at.isoformat() if intervention.completed_at else None,
            "success_metrics": intervention.success_metrics,
            "notes": intervention.notes,
            "days_remaining": (intervention.due_date - datetime.now()).days,
            "is_overdue": datetime.now() > intervention.due_date and intervention.status not in [InterventionStatus.COMPLETED, InterventionStatus.FAILED]
        }
    
    def get_active_interventions(self, customer_id: str = None) -> List[Dict[str, Any]]:
        """Get all active interventions"""
        try:
            interventions = []
            
            for intervention in self.active_interventions.values():
                if customer_id is None or intervention.customer_id == customer_id:
                    interventions.append(self.format_intervention_status(intervention))
            
            return interventions
            
        except Exception as e:
            self.logger.error(f"Error getting active interventions: {str(e)}")
            return []
    
    def get_overdue_interventions(self) -> List[Dict[str, Any]]:
        """Get all overdue interventions"""
        try:
            overdue = []
            current_time = datetime.now()
            
            for intervention in self.active_interventions.values():
                if current_time > intervention.due_date:
                    status = self.format_intervention_status(intervention)
                    status["days_overdue"] = (current_time - intervention.due_date).days
                    overdue.append(status)
            
            return overdue
            
        except Exception as e:
            self.logger.error(f"Error getting overdue interventions: {str(e)}")
            return []
    
    def get_intervention_summary(self) -> Dict[str, Any]:
        """Get summary of all interventions"""
        try:
            active_count = len(self.active_interventions)
            completed_count = len(self.completed_interventions)
            
            # Calculate success rate
            total_interventions = active_count + completed_count
            successful_interventions = len([
                i for i in self.completed_interventions.values()
                if i.status == InterventionStatus.COMPLETED
            ])
            
            success_rate = (successful_interventions / completed_count * 100) if completed_count > 0 else 0
            
            return {
                "total_interventions": total_interventions,
                "active_interventions": active_count,
                "completed_interventions": completed_count,
                "successful_interventions": successful_interventions,
                "success_rate": round(success_rate, 2),
                "overdue_interventions": len(self.get_overdue_interventions()),
                "interventions_by_priority": {
                    "critical": len([i for i in self.active_interventions.values() if i.priority == InterventionPriority.CRITICAL]),
                    "high": len([i for i in self.active_interventions.values() if i.priority == InterventionPriority.HIGH]),
                    "medium": len([i for i in self.active_interventions.values() if i.priority == InterventionPriority.MEDIUM]),
                    "low": len([i for i in self.active_interventions.values() if i.priority == InterventionPriority.LOW])
                },
                "interventions_by_type": {
                    "retention": len([i for i in self.active_interventions.values() if i.intervention_type == InterventionType.RETENTION]),
                    "engagement": len([i for i in self.active_interventions.values() if i.intervention_type == InterventionType.ENGAGEMENT]),
                    "expansion": len([i for i in self.active_interventions.values() if i.intervention_type == InterventionType.EXPANSION]),
                    "support": len([i for i in self.active_interventions.values() if i.intervention_type == InterventionType.SUPPORT]),
                    "onboarding": len([i for i in self.active_interventions.values() if i.intervention_type == InterventionType.ONBOARDING])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting intervention summary: {str(e)}")
            return {}
    
    def generate_intervention_report(self, timeframe: str = "30d") -> Dict[str, Any]:
        """Generate comprehensive intervention report"""
        try:
            # Calculate date range
            end_date = datetime.now()
            if timeframe == "7d":
                start_date = end_date - timedelta(days=7)
            elif timeframe == "30d":
                start_date = end_date - timedelta(days=30)
            elif timeframe == "90d":
                start_date = end_date - timedelta(days=90)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Filter interventions by date range
            period_interventions = []
            for intervention in self.completed_interventions.values():
                if intervention.created_at >= start_date:
                    period_interventions.append(intervention)
            
            # Calculate metrics
            total_interventions = len(period_interventions)
            successful_interventions = len([
                i for i in period_interventions
                if i.status == InterventionStatus.COMPLETED
            ])
            
            success_rate = (successful_interventions / total_interventions * 100) if total_interventions > 0 else 0
            
            # Calculate average resolution time
            resolution_times = []
            for intervention in period_interventions:
                if intervention.completed_at:
                    resolution_time = (intervention.completed_at - intervention.created_at).total_seconds() / 3600  # hours
                    resolution_times.append(resolution_time)
            
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            return {
                "report_period": timeframe,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "summary": {
                    "total_interventions": total_interventions,
                    "successful_interventions": successful_interventions,
                    "failed_interventions": total_interventions - successful_interventions,
                    "success_rate": round(success_rate, 2),
                    "average_resolution_time_hours": round(avg_resolution_time, 2)
                },
                "intervention_breakdown": {
                    "by_type": {
                        "retention": len([i for i in period_interventions if i.intervention_type == InterventionType.RETENTION]),
                        "engagement": len([i for i in period_interventions if i.intervention_type == InterventionType.ENGAGEMENT]),
                        "expansion": len([i for i in period_interventions if i.intervention_type == InterventionType.EXPANSION]),
                        "support": len([i for i in period_interventions if i.intervention_type == InterventionType.SUPPORT]),
                        "onboarding": len([i for i in period_interventions if i.intervention_type == InterventionType.ONBOARDING])
                    },
                    "by_priority": {
                        "critical": len([i for i in period_interventions if i.priority == InterventionPriority.CRITICAL]),
                        "high": len([i for i in period_interventions if i.priority == InterventionPriority.HIGH]),
                        "medium": len([i for i in period_interventions if i.priority == InterventionPriority.MEDIUM]),
                        "low": len([i for i in period_interventions if i.priority == InterventionPriority.LOW])
                    }
                },
                "top_triggers": [
                    "Low engagement score",
                    "High churn risk",
                    "Payment delays",
                    "Support ticket increase",
                    "Feature underutilization"
                ],  # Would be calculated from actual data in real implementation
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating intervention report: {str(e)}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get intervention manager status"""
        return {
            "status": "operational",
            "active_interventions": len(self.active_interventions),
            "completed_interventions": len(self.completed_interventions),
            "intervention_playbooks": len(self.intervention_playbooks),
            "success_rate": self.intervention_metrics.get("success_rate", 0),
            "overdue_interventions": len(self.get_overdue_interventions()),
            "last_update": datetime.now().isoformat()
        }
