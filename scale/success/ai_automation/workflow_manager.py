"""
Workflow Manager for Customer Success Automation
Handles automated workflow execution and management
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
from enum import Enum


class WorkflowStatus(Enum):
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class WorkflowStep:
    name: str
    description: str
    delay_hours: int
    prerequisites: List[str] = None
    automation_level: str = "automatic"
    timeout_hours: int = 24
    retry_count: int = 0
    max_retries: int = 3
    success_criteria: Dict[str, Any] = None


@dataclass
class WorkflowExecution:
    workflow_id: str
    customer_id: str
    workflow_type: str
    status: WorkflowStatus
    priority: WorkflowPriority
    steps: List[WorkflowStep]
    current_step_index: int
    context: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_log: List[str] = None
    execution_history: List[Dict[str, Any]] = None


class WorkflowManager:
    """
    Manages automated workflows for customer success operations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Active workflows
        self.active_workflows: Dict[str, WorkflowExecution] = {}
        self.completed_workflows: Dict[str, WorkflowExecution] = {}
        self.failed_workflows: Dict[str, WorkflowExecution] = {}
        
        # Workflow templates
        self.workflow_templates = self.load_workflow_templates()
        
        # Execution tracking
        self.execution_stats = {
            "total_initiated": 0,
            "total_completed": 0,
            "total_failed": 0,
            "average_completion_time": 0,
            "success_rate": 0
        }
        
        self.logger.info("Workflow Manager initialized")
    
    def load_workflow_templates(self) -> Dict[str, Any]:
        """Load predefined workflow templates"""
        return {
            "customer_onboarding": {
                "description": "Complete customer onboarding process",
                "priority": WorkflowPriority.HIGH,
                "estimated_duration_hours": 168,  # 7 days
                "steps": [
                    WorkflowStep(
                        name="send_welcome_email",
                        description="Send personalized welcome email with setup guide",
                        delay_hours=0,
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="schedule_setup_call",
                        description="Schedule initial setup and configuration call",
                        delay_hours=1,
                        prerequisites=["send_welcome_email"],
                        automation_level="semi-automatic"
                    ),
                    WorkflowStep(
                        name="provide_initial_training",
                        description="Provide basic training on key features",
                        delay_hours=24,
                        prerequisites=["schedule_setup_call"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="first_success_check",
                        description="Conduct first success check after 3 days",
                        delay_hours=72,
                        prerequisites=["provide_initial_training"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="week_one_review",
                        description="Review progress after first week",
                        delay_hours=168,
                        prerequisites=["first_success_check"],
                        automation_level="semi-automatic"
                    )
                ]
            },
            "retention_intervention": {
                "description": "Retention intervention for at-risk customers",
                "priority": WorkflowPriority.CRITICAL,
                "estimated_duration_hours": 72,
                "steps": [
                    WorkflowStep(
                        name="immediate_alert",
                        description="Send immediate alert to success team",
                        delay_hours=0,
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="risk_analysis",
                        description="Perform detailed risk analysis",
                        delay_hours=1,
                        prerequisites=["immediate_alert"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="personal_outreach",
                        description="Personal outreach by success manager",
                        delay_hours=4,
                        prerequisites=["risk_analysis"],
                        automation_level="manual"
                    ),
                    WorkflowStep(
                        name="solution_implementation",
                        description="Implement targeted solution",
                        delay_hours=24,
                        prerequisites=["personal_outreach"],
                        automation_level="semi-automatic"
                    ),
                    WorkflowStep(
                        name="follow_up_monitoring",
                        description="Monitor for 48 hours after intervention",
                        delay_hours=48,
                        prerequisites=["solution_implementation"],
                        automation_level="automatic"
                    )
                ]
            },
            "expansion_opportunity": {
                "description": "Expansion opportunity identification and pursuit",
                "priority": WorkflowPriority.MEDIUM,
                "estimated_duration_hours": 336,  # 14 days
                "steps": [
                    WorkflowStep(
                        name="value_analysis",
                        description="Analyze current value and potential",
                        delay_hours=0,
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="opportunity_identification",
                        description="Identify specific expansion opportunities",
                        delay_hours=24,
                        prerequisites=["value_analysis"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="proposal_preparation",
                        description="Prepare expansion proposal",
                        delay_hours=72,
                        prerequisites=["opportunity_identification"],
                        automation_level="semi-automatic"
                    ),
                    WorkflowStep(
                        name="sales_handoff",
                        description="Handoff to sales team",
                        delay_hours=168,
                        prerequisites=["proposal_preparation"],
                        automation_level="manual"
                    )
                ]
            },
            "engagement_recovery": {
                "description": "Recover customer engagement after decline",
                "priority": WorkflowPriority.HIGH,
                "estimated_duration_hours": 168,
                "steps": [
                    WorkflowStep(
                        name="engagement_analysis",
                        description="Analyze engagement decline patterns",
                        delay_hours=0,
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="personalized_content",
                        description="Send personalized engagement content",
                        delay_hours=24,
                        prerequisites=["engagement_analysis"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="training_invitation",
                        description="Invite to relevant training session",
                        delay_hours=72,
                        prerequisites=["personalized_content"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="success_story_sharing",
                        description="Share relevant success stories",
                        delay_hours=120,
                        prerequisites=["training_invitation"],
                        automation_level="automatic"
                    )
                ]
            },
            "advocacy_development": {
                "description": "Develop customer advocates",
                "priority": WorkflowPriority.MEDIUM,
                "estimated_duration_hours": 720,  # 30 days
                "steps": [
                    WorkflowStep(
                        name="advocacy_assessment",
                        description="Assess customer advocacy potential",
                        delay_hours=0,
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="content_collaboration",
                        description="Collaborate on case studies or content",
                        delay_hours=168,
                        prerequisites=["advocacy_assessment"],
                        automation_level="semi-automatic"
                    ),
                    WorkflowStep(
                        name="referral_program_invitation",
                        description="Invite to referral program",
                        delay_hours=336,
                        prerequisites=["content_collaboration"],
                        automation_level="automatic"
                    ),
                    WorkflowStep(
                        name="community_introduction",
                        description="Introduce to customer community leadership",
                        delay_hours=504,
                        prerequisites=["referral_program_invitation"],
                        automation_level="automatic"
                    )
                ]
            }
        }
    
    def initiate_workflow(
        self, 
        customer_id: str, 
        workflow_type: str, 
        context: Dict[str, Any],
        priority: Optional[WorkflowPriority] = None
    ) -> str:
        """Initiate a new workflow for a customer"""
        try:
            # Validate workflow type
            if workflow_type not in self.workflow_templates:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            template = self.workflow_templates[workflow_type]
            
            # Create workflow execution
            workflow_id = self.generate_workflow_id(workflow_type, customer_id)
            
            workflow_execution = WorkflowExecution(
                workflow_id=workflow_id,
                customer_id=customer_id,
                workflow_type=workflow_type,
                status=WorkflowStatus.INITIATED,
                priority=priority or template["priority"],
                steps=template["steps"].copy(),
                current_step_index=0,
                context=context,
                started_at=datetime.now(),
                error_log=[],
                execution_history=[]
            )
            
            # Store active workflow
            self.active_workflows[workflow_id] = workflow_execution
            
            # Update stats
            self.execution_stats["total_initiated"] += 1
            
            # Start execution
            asyncio.create_task(self.execute_workflow(workflow_id))
            
            self.logger.info(f"Initiated workflow '{workflow_type}' (ID: {workflow_id}) for customer {customer_id}")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Error initiating workflow: {str(e)}")
            raise
    
    async def execute_workflow(self, workflow_id: str):
        """Execute workflow steps"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                self.logger.error(f"Workflow {workflow_id} not found")
                return
            
            workflow.status = WorkflowStatus.IN_PROGRESS
            self.log_execution_step(workflow_id, "WORKFLOW_STARTED", "Workflow execution started")
            
            while workflow.current_step_index < len(workflow.steps):
                current_step = workflow.steps[workflow.current_step_index]
                
                # Check prerequisites
                if not self.check_prerequisites(workflow_id, current_step):
                    self.log_execution_step(
                        workflow_id, "STEP_SKIPPED", 
                        f"Step {current_step.name} prerequisites not met"
                    )
                    workflow.current_step_index += 1
                    continue
                
                # Wait for delay
                if current_step.delay_hours > 0:
                    self.log_execution_step(
                        workflow_id, "STEP_DELAYED", 
                        f"Waiting {current_step.delay_hours} hours for {current_step.name}"
                    )
                    await asyncio.sleep(current_step.delay_hours * 3600)
                
                # Execute step
                success = await self.execute_workflow_step(workflow_id, current_step)
                
                if success:
                    self.log_execution_step(
                        workflow_id, "STEP_COMPLETED", 
                        f"Step {current_step.name} completed successfully"
                    )
                    workflow.current_step_index += 1
                else:
                    # Handle failure
                    current_step.retry_count += 1
                    if current_step.retry_count >= current_step.max_retries:
                        self.fail_workflow(workflow_id, f"Max retries reached for step {current_step.name}")
                        return
                    else:
                        self.log_execution_step(
                            workflow_id, "STEP_RETRY", 
                            f"Step {current_step.name} failed, retrying (attempt {current_step.retry_count})"
                        )
                        await asyncio.sleep(3600)  # Wait 1 hour before retry
            
            # Workflow completed
            self.complete_workflow(workflow_id)
            
        except Exception as e:
            self.logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
            self.fail_workflow(workflow_id, str(e))
    
    def check_prerequisites(self, workflow_id: str, step: WorkflowStep) -> bool:
        """Check if step prerequisites are met"""
        if not step.prerequisites:
            return True
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return False
        
        # Check if prerequisite steps were completed
        completed_steps = set()
        for i in range(workflow.current_step_index):
            completed_steps.add(workflow.steps[i].name)
        
        return all(prereq in completed_steps for prereq in step.prerequisites)
    
    async def execute_workflow_step(self, workflow_id: str, step: WorkflowStep) -> bool:
        """Execute a specific workflow step"""
        try:
            workflow = self.active_workflows.get(workflow_id)
            if not workflow:
                return False
            
            self.log_execution_step(workflow_id, "STEP_STARTED", f"Executing {step.name}")
            
            # Route to appropriate handler based on step name
            success = False
            
            if step.name == "send_welcome_email":
                success = self.send_welcome_email(workflow.customer_id, workflow.context)
            elif step.name == "schedule_setup_call":
                success = self.schedule_setup_call(workflow.customer_id, workflow.context)
            elif step.name == "provide_initial_training":
                success = self.provide_initial_training(workflow.customer_id, workflow.context)
            elif step.name == "first_success_check":
                success = self.conduct_first_success_check(workflow.customer_id)
            elif step.name == "week_one_review":
                success = self.conduct_week_one_review(workflow.customer_id)
            elif step.name == "immediate_alert":
                success = self.send_immediate_alert(workflow.customer_id)
            elif step.name == "risk_analysis":
                success = self.perform_risk_analysis(workflow.customer_id)
            elif step.name == "personal_outreach":
                success = await self.attempt_personal_outreach(workflow.customer_id)
            elif step.name == "solution_implementation":
                success = self.implement_solution(workflow.customer_id, workflow.context)
            elif step.name == "follow_up_monitoring":
                success = await self.setup_follow_up_monitoring(workflow.customer_id)
            elif step.name == "value_analysis":
                success = self.analyze_customer_value(workflow.customer_id)
            elif step.name == "opportunity_identification":
                success = self.identify_expansion_opportunities(workflow.customer_id)
            elif step.name == "proposal_preparation":
                success = self.prepare_expansion_proposal(workflow.customer_id, workflow.context)
            elif step.name == "sales_handoff":
                success = await self.handoff_to_sales(workflow.customer_id, workflow.context)
            elif step.name == "engagement_analysis":
                success = self.analyze_engagement_decline(workflow.customer_id)
            elif step.name == "personalized_content":
                success = self.send_personalized_content(workflow.customer_id, workflow.context)
            elif step.name == "training_invitation":
                success = self.send_training_invitation(workflow.customer_id)
            elif step.name == "success_story_sharing":
                success = self.share_success_stories(workflow.customer_id)
            elif step.name == "advocacy_assessment":
                success = self.assess_advocacy_potential(workflow.customer_id)
            elif step.name == "content_collaboration":
                success = await self.initiate_content_collaboration(workflow.customer_id)
            elif step.name == "referral_program_invitation":
                success = self.invite_to_referral_program(workflow.customer_id)
            elif step.name == "community_introduction":
                success = await self.introduce_to_community(workflow.customer_id)
            
            if not success:
                self.log_execution_step(workflow_id, "STEP_FAILED", f"Step {step.name} failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error executing step {step.name}: {str(e)}")
            return False
    
    def send_welcome_email(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Send welcome email"""
        try:
            # Implementation would send actual email
            self.logger.info(f"Sending welcome email to customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending welcome email: {str(e)}")
            return False
    
    def schedule_setup_call(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Schedule setup call"""
        try:
            # Implementation would create calendar event
            self.logger.info(f"Scheduling setup call for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error scheduling setup call: {str(e)}")
            return False
    
    def provide_initial_training(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Provide initial training"""
        try:
            # Implementation would send training materials or schedule session
            self.logger.info(f"Providing initial training to customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error providing initial training: {str(e)}")
            return False
    
    def conduct_first_success_check(self, customer_id: str) -> bool:
        """Conduct first success check"""
        try:
            # Implementation would calculate and record success metrics
            self.logger.info(f"Conducting first success check for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error conducting first success check: {str(e)}")
            return False
    
    def conduct_week_one_review(self, customer_id: str) -> bool:
        """Conduct week one review"""
        try:
            # Implementation would generate review report
            self.logger.info(f"Conducting week one review for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error conducting week one review: {str(e)}")
            return False
    
    def send_immediate_alert(self, customer_id: str) -> bool:
        """Send immediate alert to success team"""
        try:
            # Implementation would send alert
            self.logger.info(f"Sending immediate alert for at-risk customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending immediate alert: {str(e)}")
            return False
    
    def perform_risk_analysis(self, customer_id: str) -> bool:
        """Perform detailed risk analysis"""
        try:
            # Implementation would analyze customer data and calculate risk
            self.logger.info(f"Performing risk analysis for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error performing risk analysis: {str(e)}")
            return False
    
    async def attempt_personal_outreach(self, customer_id: str) -> bool:
        """Attempt personal outreach"""
        try:
            # Implementation would make phone call or send personal message
            await asyncio.sleep(1)  # Simulate outreach time
            self.logger.info(f"Attempting personal outreach to customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error in personal outreach: {str(e)}")
            return False
    
    def implement_solution(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Implement targeted solution"""
        try:
            # Implementation would implement specific solution based on context
            self.logger.info(f"Implementing solution for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error implementing solution: {str(e)}")
            return False
    
    async def setup_follow_up_monitoring(self, customer_id: str) -> bool:
        """Setup follow-up monitoring"""
        try:
            # Implementation would set up monitoring alerts
            await asyncio.sleep(1)  # Simulate setup time
            self.logger.info(f"Setting up follow-up monitoring for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting up follow-up monitoring: {str(e)}")
            return False
    
    def analyze_customer_value(self, customer_id: str) -> bool:
        """Analyze customer value and potential"""
        try:
            # Implementation would calculate value metrics
            self.logger.info(f"Analyzing customer value for {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error analyzing customer value: {str(e)}")
            return False
    
    def identify_expansion_opportunities(self, customer_id: str) -> bool:
        """Identify expansion opportunities"""
        try:
            # Implementation would identify upsell/cross-sell opportunities
            self.logger.info(f"Identifying expansion opportunities for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error identifying expansion opportunities: {str(e)}")
            return False
    
    def prepare_expansion_proposal(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Prepare expansion proposal"""
        try:
            # Implementation would generate proposal document
            self.logger.info(f"Preparing expansion proposal for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error preparing expansion proposal: {str(e)}")
            return False
    
    async def handoff_to_sales(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Handoff to sales team"""
        try:
            # Implementation would create sales opportunity
            await asyncio.sleep(1)  # Simulate handoff process
            self.logger.info(f"Handoff to sales for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error in sales handoff: {str(e)}")
            return False
    
    def analyze_engagement_decline(self, customer_id: str) -> bool:
        """Analyze engagement decline patterns"""
        try:
            # Implementation would analyze engagement data
            self.logger.info(f"Analyzing engagement decline for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error analyzing engagement decline: {str(e)}")
            return False
    
    def send_personalized_content(self, customer_id: str, context: Dict[str, Any]) -> bool:
        """Send personalized engagement content"""
        try:
            # Implementation would send targeted content
            self.logger.info(f"Sending personalized content to customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending personalized content: {str(e)}")
            return False
    
    def send_training_invitation(self, customer_id: str) -> bool:
        """Send training invitation"""
        try:
            # Implementation would send training invitation
            self.logger.info(f"Sending training invitation to customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sending training invitation: {str(e)}")
            return False
    
    def share_success_stories(self, customer_id: str) -> bool:
        """Share relevant success stories"""
        try:
            # Implementation would share success stories
            self.logger.info(f"Sharing success stories with customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error sharing success stories: {str(e)}")
            return False
    
    def assess_advocacy_potential(self, customer_id: str) -> bool:
        """Assess customer advocacy potential"""
        try:
            # Implementation would calculate advocacy score
            self.logger.info(f"Assessing advocacy potential for customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error assessing advocacy potential: {str(e)}")
            return False
    
    async def initiate_content_collaboration(self, customer_id: str) -> bool:
        """Initiate content collaboration"""
        try:
            # Implementation would reach out for collaboration
            await asyncio.sleep(2)  # Simulate collaboration setup
            self.logger.info(f"Initiating content collaboration with customer {customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error initiating content collaboration: {str(e)}")
            return False
    
    def invite_to_referral_program(self, customer_id: str) -> bool:
        """Invite to referral program"""
        try:
            # Implementation would send referral program invitation
            self.logger.info(f"Inviting customer {customer_id} to referral program")
            return True
        except Exception as e:
            self.logger.error(f"Error inviting to referral program: {str(e)}")
            return False
    
    async def introduce_to_community(self, customer_id: str) -> bool:
        """Introduce to customer community"""
        try:
            # Implementation would add to community or introduce to community leaders
            await asyncio.sleep(1)  # Simulate introduction
            self.logger.info(f"Introducing customer {customer_id} to community")
            return True
        except Exception as e:
            self.logger.error(f"Error introducing to community: {str(e)}")
            return False
    
    def complete_workflow(self, workflow_id: str):
        """Mark workflow as completed"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = datetime.now()
            
            # Move to completed workflows
            self.completed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            # Update stats
            self.execution_stats["total_completed"] += 1
            self.update_success_rate()
            
            self.log_execution_step(workflow_id, "WORKFLOW_COMPLETED", "Workflow completed successfully")
            self.logger.info(f"Workflow {workflow_id} completed")
    
    def fail_workflow(self, workflow_id: str, error_message: str):
        """Mark workflow as failed"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.FAILED
            workflow.completed_at = datetime.now()
            workflow.error_log.append(error_message)
            
            # Move to failed workflows
            self.failed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            # Update stats
            self.execution_stats["total_failed"] += 1
            self.update_success_rate()
            
            self.log_execution_step(workflow_id, "WORKFLOW_FAILED", f"Workflow failed: {error_message}")
            self.logger.error(f"Workflow {workflow_id} failed: {error_message}")
    
    def log_execution_step(self, workflow_id: str, event: str, message: str):
        """Log workflow execution step"""
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "event": event,
            "message": message
        }
        
        # Add to workflow's execution history
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id].execution_history.append(log_entry)
        
        self.logger.info(f"[{workflow_id}] {event}: {message}")
    
    def generate_workflow_id(self, workflow_type: str, customer_id: str) -> str:
        """Generate unique workflow ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{workflow_type}_{customer_id}_{timestamp}"
    
    def update_success_rate(self):
        """Update workflow success rate statistics"""
        total_workflows = self.execution_stats["total_completed"] + self.execution_stats["total_failed"]
        if total_workflows > 0:
            self.execution_stats["success_rate"] = (
                self.execution_stats["total_completed"] / total_workflows * 100
            )
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status"""
        # Check active workflows
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            return self.format_workflow_status(workflow)
        
        # Check completed workflows
        if workflow_id in self.completed_workflows:
            workflow = self.completed_workflows[workflow_id]
            return self.format_workflow_status(workflow)
        
        # Check failed workflows
        if workflow_id in self.failed_workflows:
            workflow = self.failed_workflows[workflow_id]
            return self.format_workflow_status(workflow)
        
        return None
    
    def format_workflow_status(self, workflow: WorkflowExecution) -> Dict[str, Any]:
        """Format workflow status for display"""
        return {
            "workflow_id": workflow.workflow_id,
            "customer_id": workflow.customer_id,
            "workflow_type": workflow.workflow_type,
            "status": workflow.status.value,
            "priority": workflow.priority.value,
            "current_step": workflow.current_step_index + 1,
            "total_steps": len(workflow.steps),
            "started_at": workflow.started_at.isoformat(),
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "execution_history": workflow.execution_history[-5:],  # Last 5 events
            "error_log": workflow.error_log
        }
    
    def get_active_workflows(self) -> List[Dict[str, Any]]:
        """Get all active workflows"""
        return [self.format_workflow_status(w) for w in self.active_workflows.values()]
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        return {
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "failed_workflows": len(self.failed_workflows),
            "execution_stats": self.execution_stats
        }
    
    def cancel_workflow(self, workflow_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = datetime.now()
            workflow.error_log.append(f"Cancelled: {reason}")
            
            # Move to appropriate category
            self.failed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            self.log_execution_step(workflow_id, "WORKFLOW_CANCELLED", f"Workflow cancelled: {reason}")
            return True
        
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow manager status"""
        return {
            "status": "operational",
            "active_workflows": len(self.active_workflows),
            "completed_workflows": len(self.completed_workflows),
            "failed_workflows": len(self.failed_workflows),
            "available_templates": list(self.workflow_templates.keys()),
            "success_rate": self.execution_stats["success_rate"],
            "last_update": datetime.now().isoformat()
        }
