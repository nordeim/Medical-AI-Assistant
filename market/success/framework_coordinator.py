"""
Enterprise Customer Success Framework Coordinator
Integrates all customer success components for healthcare AI organizations
"""

import datetime
from typing import Dict, List, Optional, Any
import logging

from config.framework_config import HealthcareCSConfig, CustomerTier, HealthScoreStatus
from management.customer_success_manager import HealthcareCSManager, CustomerProfile
from retention.retention_strategies import HealthcareRetentionManager, RetentionStrategyType, RetentionRiskLevel
from expansion.expansion_strategies import ExpansionRevenueManager, ExpansionType, ExpansionTrigger
from monitoring.health_score_monitor import HealthScoreMonitor
from feedback.feedback_loops import FeedbackLoopManager, FeedbackType, FeedbackPriority
from reviews.annual_business_reviews import AnnualBusinessReviewManager, ReviewType
from community.customer_community import CommunityManager

class EnterpriseCustomerSuccessFramework:
    """Main coordinator for enterprise customer success framework"""
    
    def __init__(self):
        # Initialize all components
        self.csm_manager = HealthcareCSManager()
        self.retention_manager = HealthcareRetentionManager()
        self.expansion_manager = ExpansionRevenueManager()
        self.health_monitor = HealthScoreMonitor()
        self.feedback_manager = FeedbackLoopManager()
        self.review_manager = AnnualBusinessReviewManager()
        self.community_manager = CommunityManager()
        
        # Framework state
        self.framework_initialized = False
        self.integration_hooks = {}
        self.automated_workflows = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize automated workflows
        self._initialize_automated_workflows()
        
        self.logger.info("Enterprise Customer Success Framework initialized")
    
    def _initialize_automated_workflows(self):
        """Initialize automated workflows and triggers"""
        
        # Health Score Alert Workflow
        self.automated_workflows["health_score_alerts"] = {
            "trigger": "health_score_drop",
            "conditions": ["health_score_below_60", "rapid_decline_20_percent"],
            "actions": [
                "create_intervention_workflow",
                "notify_csm_team",
                "schedule_urgent_check_in",
                "deploy_retention_strategy"
            ],
            "enabled": True
        }
        
        # Feedback Processing Workflow
        self.automated_workflows["feedback_processing"] = {
            "trigger": "critical_feedback_received",
            "conditions": ["feedback_priority_critical", "customer_tier_enterprise"],
            "actions": [
                "immediate_team_notification",
                "create_bug_fix_request",
                "schedule_customer_call",
                "escalate_to_product_team"
            ],
            "enabled": True
        }
        
        # Expansion Opportunity Workflow
        self.automated_workflows["expansion_opportunity"] = {
            "trigger": "high_usage_detected",
            "conditions": ["usage_above_80_percent", "health_score_above_75"],
            "actions": [
                "identify_expansion_opportunities",
                "schedule_expansion_conversation",
                "prepare_roi_documentation",
                "create_expansion_proposal"
            ],
            "enabled": True
        }
        
        # Annual Review Workflow
        self.automated_workflows["annual_review_prep"] = {
            "trigger": "review_scheduled",
            "conditions": ["customer_tier_enterprise", "health_score_above_70"],
            "actions": [
                "prepare_business_metrics",
                "calculate_clinical_roi",
                "generate_strategic_objectives",
                "schedule_stakeholder_meetings"
            ],
            "enabled": True
        }
    
    def initialize_customer_onboarding(self, customer_data: Dict) -> str:
        """Initialize complete customer onboarding into success framework"""
        
        # Create customer profile
        customer_id = customer_data["customer_id"]
        
        # Create customer profile for CSM
        customer_profile = CustomerProfile(
            customer_id=customer_id,
            organization_name=customer_data["organization_name"],
            customer_tier=CustomerTier(customer_data["tier"]),
            segment=customer_data["segment"],
            primary_contact=customer_data.get("primary_contact", ""),
            email=customer_data.get("email", ""),
            csm_assigned=customer_data.get("csm_assigned", ""),
            contract_start_date=datetime.datetime.strptime(customer_data.get("contract_start", ""), "%Y-%m-%d").date(),
            contract_value=customer_data.get("contract_value", 0)
        )
        
        # Add customer to CSM system
        self.csm_manager.add_customer(customer_profile)
        
        # Initialize health monitoring metrics
        self._setup_customer_health_monitoring(customer_id, customer_data)
        
        # Join customer to community
        self._setup_customer_community_access(customer_id, customer_data)
        
        # Schedule initial review
        self._schedule_initial_customer_review(customer_id, customer_data)
        
        self.logger.info(f"Completed framework onboarding for customer {customer_id}")
        return customer_id
    
    def _setup_customer_health_monitoring(self, customer_id: str, customer_data: Dict):
        """Setup health monitoring for new customer"""
        
        # Register key health metrics
        metrics_to_register = [
            "usage_percentage",
            "clinical_outcome_improvement",
            "user_adoption_rate",
            "support_ticket_volume",
            "nps_score",
            "engagement_level"
        ]
        
        for metric_name in metrics_to_register:
            metric_config = self._get_metric_configuration(metric_name, customer_data)
            self.health_monitor.register_metric(metric_config)
    
    def _get_metric_configuration(self, metric_name: str, customer_data: Dict):
        """Get metric configuration for customer"""
        
        # This would return properly configured HealthMetric objects
        # Simplified for example purposes
        
        from monitoring.health_score_monitor import HealthMetric, MetricType
        
        return HealthMetric(
            metric_id=f"{customer_data['customer_id']}_{metric_name}",
            customer_id=customer_data['customer_id'],
            metric_type=MetricType.USAGE,
            metric_name=metric_name,
            current_value=75.0,
            target_value=85.0,
            threshold_warning=65.0,
            threshold_critical=50.0,
            weight=0.15,
            last_updated=datetime.datetime.now(),
            data_source="automated_system",
            measurement_frequency="daily"
        )
    
    def _setup_customer_community_access(self, customer_id: str, customer_data: Dict):
        """Setup community access for customer team members"""
        
        # Add key team members to community
        team_members = customer_data.get("team_members", [])
        
        for member_data in team_members:
            member = self.community_manager.join_community(
                customer_id=customer_id,
                name=member_data["name"],
                title=member_data["title"],
                organization=customer_data["organization_name"],
                specialty=member_data["specialty"],
                expertise_areas=member_data.get("expertise_areas", [])
            )
    
    def _schedule_initial_customer_review(self, customer_id: str, customer_data: Dict):
        """Schedule initial customer review"""
        
        # Schedule based on customer tier
        customer_tier = CustomerTier(customer_data["tier"])
        
        if customer_tier in [CustomerTier.ENTERPRISE, CustomerTier.STRATEGIC]:
            review_type = ReviewType.STRATEGIC_PLANNING
        else:
            review_type = ReviewType.QUARTERLY_BUSINESS_REVIEW
        
        # Schedule review in 30 days
        review_date = datetime.datetime.now() + datetime.timedelta(days=30)
        
        attendees = customer_data.get("key_stakeholders", [])
        
        self.review_manager.schedule_business_review(
            customer_id=customer_id,
            review_type=review_type,
            meeting_date=review_date,
            attendees=attendees
        )
    
    def process_customer_health_update(self, customer_id: str, health_metrics: Dict) -> Dict:
        """Process customer health update and trigger appropriate workflows"""
        
        results = {
            "customer_id": customer_id,
            "health_score_calculated": False,
            "alerts_triggered": 0,
            "workflows_started": [],
            "interventions_created": 0
        }
        
        try:
            # Update health metrics
            for metric_name, value in health_metrics.items():
                metric_id = f"{customer_id}_{metric_name}"
                self.health_monitor.update_metric_value(metric_id, value)
            
            # Get updated health dashboard
            health_dashboard = self.health_monitor.get_customer_health_dashboard(customer_id)
            results["health_score_calculated"] = True
            results["current_health_score"] = health_dashboard["overall_health_score"]
            
            # Check if alerts were triggered
            active_alerts = health_dashboard.get("alerts_summary", {}).get("total_alerts", 0)
            results["alerts_triggered"] = active_alerts
            
            # Process automated workflows
            if health_dashboard["overall_health_score"] < 60:
                workflow_result = self._trigger_health_workflow(customer_id, health_dashboard)
                results["workflows_started"].append(workflow_result)
            
            # Update CSM manager with health score
            customer_profile = self.csm_manager.customers.get(customer_id)
            if customer_profile:
                customer_profile.current_metrics.customer_health_score = health_dashboard["overall_health_score"]
            
            self.logger.info(f"Processed health update for customer {customer_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process health update for {customer_id}: {e}")
            results["error"] = str(e)
        
        return results
    
    def _trigger_health_workflow(self, customer_id: str, health_dashboard: Dict) -> str:
        """Trigger health-related automated workflow"""
        
        workflow_id = f"health_workflow_{customer_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create retention strategy if needed
        if health_dashboard["overall_health_score"] < 50:
            retention_strategy = self.retention_manager.create_retention_strategy(
                customer_id=customer_id,
                strategy_type=RetentionStrategyType.CLINICAL_OUTCOMES,
                risk_level=RetentionRiskLevel.HIGH
            )
        
        # Schedule urgent CSM check-in
        self.csm_manager.schedule_customer_activity(
            activity=self._create_urgent_checkin_activity(customer_id, health_dashboard)
        )
        
        return workflow_id
    
    def _create_urgent_checkin_activity(self, customer_id: str, health_dashboard: Dict):
        """Create urgent check-in activity for at-risk customer"""
        
        from management.customer_success_manager import CSMActivity
        
        return CSMActivity(
            activity_id=f"urgent_checkin_{customer_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=customer_id,
            csm_name=self.csm_manager.customers[customer_id].csm_assigned if customer_id in self.csm_manager.customers else "unassigned",
            activity_type="urgent_health_check",
            description=f"Urgent health check required - health score: {health_dashboard['overall_health_score']:.1f}",
            scheduled_date=datetime.datetime.now() + datetime.timedelta(hours=4),
            next_steps=[
                "Review health dashboard details",
                "Contact customer for immediate discussion",
                "Assess intervention requirements"
            ]
        )
    
    def process_customer_feedback(self, customer_id: str, feedback_data: Dict) -> Dict:
        """Process customer feedback and trigger appropriate actions"""
        
        results = {
            "feedback_submitted": False,
            "related_feedback_found": 0,
            "enhancement_request_created": False,
            "actions_taken": []
        }
        
        try:
            # Submit feedback
            feedback = self.feedback_manager.submit_customer_feedback(
                customer_id=customer_id,
                feedback_type=FeedbackType(feedback_data["type"]),
                title=feedback_data["title"],
                description=feedback_data["description"],
                submitted_by=feedback_data.get("submitted_by", "unknown"),
                priority=FeedbackPriority(feedback_data.get("priority", "medium"))
            )
            
            results["feedback_submitted"] = True
            results["feedback_id"] = feedback.feedback_id
            
            # Check for related feedback
            related_count = len(feedback.related_feedback)
            results["related_feedback_found"] = related_count
            
            # If critical feedback, trigger immediate actions
            if feedback.priority == FeedbackPriority.CRITICAL:
                results["actions_taken"].append("critical_feedback_escalation")
                # Would trigger immediate notifications and actions
            
            # If multiple similar feedback, consider creating enhancement request
            if related_count >= 3:
                enhancement = self.feedback_manager.create_enhancement_request(
                    feedback_ids=[feedback.feedback_id] + feedback.related_feedback,
                    title=f"Enhancement from customer feedback: {feedback_data['title']}",
                    description=feedback_data["description"],
                    business_value="Multiple customer requests for this feature"
                )
                results["enhancement_request_created"] = True
                results["enhancement_id"] = enhancement.request_id
            
            self.logger.info(f"Processed feedback from customer {customer_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to process feedback for {customer_id}: {e}")
            results["error"] = str(e)
        
        return results
    
    def identify_expansion_opportunities(self, customer_id: str) -> Dict:
        """Identify expansion opportunities for customer"""
        
        # Get customer data
        customer_profile = self.csm_manager.customers.get(customer_id)
        if not customer_profile:
            return {"error": "Customer not found"}
        
        # Analyze usage patterns and create opportunities
        customer_data = {
            "annual_contract_value": customer_profile.contract_value,
            "usage_metrics": {
                "overall_usage": customer_profile.current_metrics.engagement_level * 100,
                "feature_adoption": customer_profile.current_metrics.feature_adoption_rate * 100
            },
            "clinical_outcome_improvement": customer_profile.healthcare_kpis.clinical_outcome_improvement,
            "organization_size": customer_profile.segment.size_category,
            "customer_satisfaction": customer_profile.current_metrics.nps_score * 10
        }
        
        opportunities = self.expansion_manager.identify_expansion_opportunities(customer_id, customer_data)
        
        # Get pipeline value
        pipeline_value = self.expansion_manager.calculate_expansion_pipeline_value(customer_id)
        
        return {
            "customer_id": customer_id,
            "opportunities_identified": len(opportunities),
            "total_pipeline_value": pipeline_value["total_pipeline_value"],
            "weighted_pipeline_value": pipeline_value["weighted_pipeline_value"],
            "opportunities": [
                {
                    "opportunity_id": opp.opportunity_id,
                    "type": opp.expansion_type.value,
                    "product": opp.product_or_service,
                    "potential_value": opp.potential_additional_value,
                    "probability": opp.probability,
                    "expected_close_date": opp.expected_close_date
                }
                for opp in opportunities
            ]
        }
    
    def generate_customer_health_report(self, customer_id: str) -> Dict:
        """Generate comprehensive customer health report"""
        
        # Get data from all components
        health_dashboard = self.health_monitor.get_customer_health_dashboard(customer_id)
        csm_summary = self.csm_manager.generate_customer_summary(customer_id)
        
        # Check retention risk
        if customer_id in self.retention_manager.churn_predictions:
            retention_data = self.retention_manager.churn_predictions[customer_id]
        else:
            # Assess retention risk if not already done
            customer_data = self._get_customer_risk_data(customer_id)
            retention_prediction = self.retention_manager.assess_retention_risk(customer_id, customer_data)
            retention_data = retention_prediction
        
        # Get expansion opportunities
        expansion_data = self.identify_expansion_opportunities(customer_id)
        
        # Compile comprehensive report
        report = {
            "customer_id": customer_id,
            "report_date": datetime.datetime.now(),
            "executive_summary": self._generate_executive_health_summary(csm_summary, health_dashboard, retention_data),
            
            "health_score": {
                "current_score": health_dashboard["overall_health_score"],
                "status": "healthy" if health_dashboard["overall_health_score"] >= 75 else "at_risk",
                "trend": "improving",  # Would calculate from historical data
                "key_components": health_dashboard["metric_details"]
            },
            
            "retention_analysis": {
                "churn_risk": retention_data.churn_probability,
                "risk_level": "high" if retention_data.churn_probability > 0.6 else "medium" if retention_data.churn_probability > 0.3 else "low",
                "top_risk_factors": retention_data.risk_factors[:3],
                "retention_opportunities": retention_data.retention_opportunities,
                "recommended_actions": retention_data.recommended_actions
            },
            
            "expansion_potential": expansion_data,
            
            "customer_engagement": {
                "engagement_level": csm_summary["key_metrics"]["engagement_level"],
                "nps_score": csm_summary["key_metrics"]["nps_score"],
                "support_health": health_dashboard["alerts_summary"],
                "feature_adoption": csm_summary["key_metrics"]["engagement_level"] * 100
            },
            
            "clinical_outcomes": {
                "outcome_improvement": csm_summary["healthcare_kpis"]["clinical_outcome_improvement"],
                "efficiency_gain": csm_summary["healthcare_kpis"]["clinical_efficiency_gain"],
                "cost_reduction": csm_summary["healthcare_kpis"]["cost_reduction"],
                "compliance_score": csm_summary["healthcare_kpis"]["compliance_score"]
            },
            
            "upcoming_activities": csm_summary["recommendations"]["follow_up_activities"],
            
            "action_items": [
                {
                    "priority": "high" if health_dashboard["overall_health_score"] < 60 else "medium",
                    "action": "Schedule health check review",
                    "owner": "Customer Success Manager",
                    "due_date": datetime.date.today() + datetime.timedelta(days=7)
                }
            ]
        }
        
        return report
    
    def _get_customer_risk_data(self, customer_id: str) -> Dict:
        """Get customer data for risk assessment"""
        # This would gather relevant data for retention risk assessment
        customer_profile = self.csm_manager.customers.get(customer_id)
        
        if not customer_profile:
            return {}
        
        return {
            "clinical_outcome_trend": "stable",
            "roi_concerns": customer_profile.current_metrics.roi_delivery_score < 0.7,
            "workflow_adoption": customer_profile.current_metrics.engagement_level * 100,
            "competitive_pressure": 0.3,  # Would get from competitive intelligence
            "staff_satisfaction": customer_profile.healthcare_kpis.staff_satisfaction / 100
        }
    
    def _generate_executive_health_summary(self, csm_summary: Dict, health_dashboard: Dict, retention_data) -> str:
        """Generate executive summary of customer health"""
        
        health_score = health_dashboard["overall_health_score"]
        nps_score = csm_summary["key_metrics"]["nps_score"]
        churn_risk = retention_data.churn_probability
        
        if health_score >= 80 and churn_risk < 0.3:
            status = "Excellent health with strong partnership potential"
        elif health_score >= 60 and churn_risk < 0.5:
            status = "Good health with expansion opportunities"
        elif health_score >= 40:
            status = "Moderate health requiring attention"
        else:
            status = "Poor health requiring immediate intervention"
        
        summary = f"""
        Customer Health Overview:
        
        Health Score: {health_score:.1f}/100 ({status})
        NPS Score: {nps_score}/10
        Churn Risk: {churn_risk:.1%} probability
        
        Key Strengths:
        • Strong clinical outcome improvements ({csm_summary['healthcare_kpis']['clinical_outcome_improvement']:.1f}%)
        • Good user engagement ({csm_summary['key_metrics']['engagement_level']:.0%})
        • Solid ROI delivery ({csm_summary['key_metrics']['roi_delivery_score']:.0%})
        
        Areas of Focus:
        • {"Monitor health score trends" if health_score < 80 else "Continue current success patterns"}
        • {"Address retention risk factors" if churn_risk > 0.4 else "Leverage strong relationship for expansion"}
        • {"Increase engagement touchpoints" if csm_summary['key_metrics']['engagement_level'] < 0.7 else "Maintain high engagement levels"}
        
        Strategic Recommendations:
        • {"Deploy retention strategies" if churn_risk > 0.5 else "Explore expansion opportunities"}
        • {"Schedule executive review" if health_score < 70 else "Continue strategic partnership development"}
        • {"Focus on value demonstration" if csm_summary['key_metrics']['roi_delivery_score'] < 0.8 else "Leverage success for reference and growth"}
        """
        
        return summary.strip()
    
    def run_automated_health_checks(self) -> Dict:
        """Run automated health checks across all customers"""
        
        results = {
            "customers_checked": 0,
            "alerts_generated": 0,
            "interventions_triggered": 0,
            "expansion_opportunities_identified": 0,
            "customers_needing_attention": []
        }
        
        for customer_id in self.csm_manager.customers.keys():
            try:
                # Run health analysis
                health_dashboard = self.health_monitor.get_customer_health_dashboard(customer_id)
                results["customers_checked"] += 1
                
                health_score = health_dashboard["overall_health_score"]
                active_alerts = health_dashboard["alerts_summary"]["total_alerts"]
                
                if active_alerts > 0:
                    results["alerts_generated"] += active_alerts
                
                if health_score < 60 or active_alerts > 3:
                    results["customers_needing_attention"].append(customer_id)
                    results["interventions_triggered"] += 1
                
                # Check for expansion opportunities
                expansion_data = self.identify_expansion_opportunities(customer_id)
                if expansion_data.get("opportunities_identified", 0) > 0:
                    results["expansion_opportunities_identified"] += expansion_data["opportunities_identified"]
            
            except Exception as e:
                self.logger.error(f"Health check failed for customer {customer_id}: {e}")
        
        return results
    
    def generate_framework_dashboard(self) -> Dict:
        """Generate comprehensive framework dashboard"""
        
        # Get data from all components
        csm_data = self._get_csm_dashboard_data()
        retention_data = self.retention_manager.get_retention_dashboard_data()
        expansion_data = self._get_expansion_dashboard_data()
        feedback_data = self.feedback_manager.get_product_improvement_dashboard()
        community_data = self.community_manager.generate_community_dashboard()
        
        # Aggregate metrics
        total_customers = len(self.csm_manager.customers)
        at_risk_customers = len([c for c in self.csm_manager.customers.values() 
                               if c.current_metrics.churn_risk_score > 0.5])
        
        return {
            "framework_overview": {
                "total_customers": total_customers,
                "healthy_customers": total_customers - at_risk_customers,
                "at_risk_customers": at_risk_customers,
                "framework_health": "operational",
                "last_updated": datetime.datetime.now()
            },
            
            "customer_success_metrics": csm_data,
            "retention_performance": retention_data,
            "expansion_pipeline": expansion_data,
            "product_feedback": feedback_data,
            "community_engagement": community_data,
            
            "automated_workflows": {
                "active_workflows": len([w for w in self.automated_workflows.values() if w["enabled"]]),
                "workflow_triggers_today": 0,  # Would track actual triggers
                "automation_effectiveness": "high"
            },
            
            "strategic_priorities": [
                "Reduce customer churn through proactive interventions",
                "Increase expansion revenue through targeted campaigns",
                "Enhance customer community engagement",
                "Improve product based on customer feedback",
                "Scale automated success processes"
            ]
        }
    
    def _get_csm_dashboard_data(self) -> Dict:
        """Get CSM dashboard data"""
        # Aggregate CSM workload data
        total_customers = len(self.csm_manager.customers)
        csm_workload = {}
        
        for csm_name in self.csm_manager.csm_workload.keys():
            csm_workload[csm_name] = self.csm_manager.get_csm_workload_report(csm_name)
        
        return {
            "total_customers": total_customers,
            "csm_workload_summary": csm_workload,
            "average_health_score": sum(c.current_metrics.customer_health_score for c in self.csm_manager.customers.values()) / total_customers if total_customers > 0 else 0
        }
    
    def _get_expansion_dashboard_data(self) -> Dict:
        """Get expansion dashboard data"""
        total_pipeline = self.expansion_manager.calculate_expansion_pipeline_value()
        
        return {
            "total_pipeline_value": total_pipeline["total_pipeline_value"],
            "weighted_pipeline_value": total_pipeline["weighted_pipeline_value"],
            "opportunity_count": total_pipeline["opportunity_count"],
            "average_deal_size": total_pipeline["average_deal_size"]
        }
    
    def enable_automated_workflow(self, workflow_name: str) -> bool:
        """Enable specific automated workflow"""
        if workflow_name in self.automated_workflows:
            self.automated_workflows[workflow_name]["enabled"] = True
            self.logger.info(f"Enabled automated workflow: {workflow_name}")
            return True
        return False
    
    def disable_automated_workflow(self, workflow_name: str) -> bool:
        """Disable specific automated workflow"""
        if workflow_name in self.automated_workflows:
            self.automated_workflows[workflow_name]["enabled"] = False
            self.logger.info(f"Disabled automated workflow: {workflow_name}")
            return True
        return False
    
    def export_customer_data(self, customer_id: str, format: str = "json") -> Dict:
        """Export complete customer data from all components"""
        
        export_data = {
            "customer_id": customer_id,
            "export_date": datetime.datetime.now(),
            "components": {}
        }
        
        try:
            # CSM Data
            if customer_id in self.csm_manager.customers:
                export_data["components"]["customer_success"] = self.csm_manager.generate_customer_summary(customer_id)
            
            # Health Data
            export_data["components"]["health_monitoring"] = self.health_monitor.get_customer_health_dashboard(customer_id)
            
            # Retention Data
            if customer_id in self.retention_manager.churn_predictions:
                export_data["components"]["retention_analysis"] = self.retention_manager.churn_predictions[customer_id]
            
            # Expansion Data
            export_data["components"]["expansion_opportunities"] = self.identify_expansion_opportunities(customer_id)
            
            # Review Data
            customer_reviews = [r for r in self.review_manager.reviews.values() if r.customer_id == customer_id]
            export_data["components"]["business_reviews"] = [r.__dict__ for r in customer_reviews]
            
            # Community Data
            customer_members = [m for m in self.community_manager.members.values() if m.customer_id == customer_id]
            export_data["components"]["community_participation"] = [m.__dict__ for m in customer_members]
            
            self.logger.info(f"Exported comprehensive data for customer {customer_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to export data for customer {customer_id}: {e}")
            export_data["error"] = str(e)
        
        return export_data