"""
Expansion Revenue Strategies for Healthcare AI
Upselling and cross-selling strategies for medical organizations
"""

import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

class ExpansionType(Enum):
    UPSELL = "upsell"          # Increase existing product usage
    CROSS_SELL = "cross_sell"  # Add new products/services
    ADD_ON = "add_on"          # Additional features/modules
    MODULE_EXPANSION = "module_expansion"  # New AI modules
    WORKFLOW_EXPANSION = "workflow_expansion"  # New clinical workflows
    INTEGRATION_EXPANSION = "integration_expansion"  # Additional integrations

class ExpansionTrigger(Enum):
    HIGH_USAGE = "high_usage"              # Current product usage >80%
    FEATURE_SUCCESS = "feature_success"    # Successful feature adoption
    OUTCOME_IMPROVEMENT = "outcome_improvement"  # Clinical outcomes improved
    WORKFLOW_OPTIMIZATION = "workflow_optimization"  # Workflow efficiency gains
    COMPETITIVE_THREAT = "competitive_threat"  # Competitive pressure
    CONTRACT_RENEWAL = "contract_renewal"  # Renewal opportunity
    QUARTERLY_REVIEW = "quarterly_review"  # Regular review cycle
    BUDGET_AVAILABLE = "budget_available"  # Budget cycle/opportunity

class ExpansionStage(Enum):
    IDENTIFICATION = "identification"  # Opportunity identified
    QUALIFICATION = "qualification"    # Opportunity qualified
    PROPOSAL = "proposal"             # Proposal developed
    NEGOTIATION = "negotiation"       # In negotiations
    CLOSED_WON = "closed_won"         # Expansion won
    CLOSED_LOST = "closed_lost"       # Expansion lost

@dataclass
class ExpansionOpportunity:
    """Expansion opportunity for healthcare customer"""
    opportunity_id: str
    customer_id: str
    expansion_type: ExpansionType
    trigger: ExpansionTrigger
    product_or_service: str
    description: str
    
    # Financial Details
    current_annual_value: float
    potential_additional_value: float
    total_projected_value: float
    
    # Sales Process
    stage: ExpansionStage
    probability: float  # 0-1 scale
    expected_close_date: datetime.date
    
    # Analysis
    business_case: str
    roi_projection: float
    implementation_timeline: str
    
    # Tracking
    created_date: datetime.date = field(default_factory=datetime.date.today)
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    assigned_to: str = ""  # Sales rep or CSM
    notes: List[str] = field(default_factory=list)
    objections: List[str] = field(default_factory=list)
    supporting_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExpansionPlaybook:
    """Expansion playbook template for different scenarios"""
    playbook_id: str
    name: str
    target_customer_tier: str
    expansion_type: ExpansionType
    trigger_conditions: List[ExpansionTrigger]
    sales_messaging: str
    value_proposition: str
    objection_handling: Dict[str, str]
    expected_win_rate: float
    avg_deal_size_multiplier: float
    implementation_steps: List[str]
    success_metrics: List[str]

@dataclass
class ExpansionCampaign:
    """Targeted expansion campaign"""
    campaign_id: str
    name: str
    target_segment: str
    expansion_types: List[ExpansionType]
    triggers: List[ExpansionTrigger]
    start_date: datetime.date
    end_date: datetime.date
    target_customers: List[str]
    goals: Dict[str, Any]
    budget: float
    messaging_strategy: str
    campaigns: List[str]  # Email campaigns, webinars, etc.

class ExpansionRevenueManager:
    """Healthcare expansion revenue management system"""
    
    def __init__(self):
        self.opportunities: Dict[str, ExpansionOpportunity] = {}
        self.playbooks: Dict[str, ExpansionPlaybook] = {}
        self.campaigns: Dict[str, ExpansionCampaign] = {}
        self.expansion_pipeline: Dict[str, List[ExpansionOpportunity]] = {}  # customer_id -> opportunities
        self.logger = logging.getLogger(__name__)
        
        # Initialize expansion playbooks
        self._initialize_expansion_playbooks()
    
    def _initialize_expansion_playbooks(self):
        """Initialize expansion playbook templates"""
        
        # Clinical Analytics Module Expansion
        self.playbooks["clinical_analytics_upsell"] = ExpansionPlaybook(
            playbook_id="clinical_analytics_upsell",
            name="Clinical Analytics Module Expansion",
            target_customer_tier="premium",
            expansion_type=ExpansionType.UPSELL,
            trigger_conditions=[ExpansionTrigger.HIGH_USAGE, ExpansionTrigger.OUTCOME_IMPROVEMENT],
            sales_messaging="Expand your clinical insights with advanced AI analytics",
            value_proposition="Drive better patient outcomes with deeper clinical insights and predictive analytics",
            objection_handling={
                "budget": "Show ROI calculation and cost of not having advanced analytics",
                "complexity": "Emphasize seamless integration with existing workflows",
                "training": "Highlight comprehensive training and support included"
            },
            expected_win_rate=0.75,
            avg_deal_size_multiplier=1.5,
            implementation_steps=[
                "Assess current analytics needs",
                "Demonstrate advanced features",
                "Create custom ROI model",
                "Pilot program setup",
                "Full implementation"
            ],
            success_metrics=["clinical_outcome_improvement", "analytical_capability_score", "user_adoption_rate"]
        )
        
        # Workflow Automation Cross-sell
        self.playbooks["workflow_automation_crosssell"] = ExpansionPlaybook(
            playbook_id="workflow_automation_crosssell",
            name="Workflow Automation Add-on",
            target_customer_tier="enterprise",
            expansion_type=ExpansionType.CROSS_SELL,
            trigger_conditions=[ExpansionTrigger.WORKFLOW_OPTIMIZATION, ExpansionTrigger.HIGH_USAGE],
            sales_messaging="Automate routine tasks and streamline clinical workflows",
            value_proposition="Reduce manual work by 40% while improving accuracy and compliance",
            objection_handling={
                "integration": "Show proven integration track record and dedicated support",
                "reliability": "Emphasize 99.9% uptime SLA and monitoring",
                "staff_resistance": "Highlight quick wins and change management support"
            },
            expected_win_rate=0.65,
            avg_deal_size_multiplier=2.0,
            implementation_steps=[
                "Workflow analysis and mapping",
                "Automation opportunity assessment",
                "Pilot automation setup",
                "Staff training program",
                "Full rollout and optimization"
            ],
            success_metrics=["time_saved", "error_reduction", "staff_satisfaction"]
        )
        
        # Integration Package Expansion
        self.playbooks["integration_expansion"] = ExpansionPlaybook(
            playbook_id="integration_expansion",
            name="Enhanced Integration Package",
            target_customer_tier="standard",
            expansion_type=ExpansionType.INTEGRATION_EXPANSION,
            trigger_conditions=[ExpansionTrigger.HIGH_USAGE, ExpansionTrigger.COMPETITIVE_THREAT],
            sales_messaging="Connect all your healthcare systems for seamless data flow",
            value_proposition="Eliminate data silos and improve care coordination across your organization",
            objection_handling={
                "technical": "Provide detailed integration roadmap and technical support",
                "cost": "Show total cost of ownership comparison including manual processes",
                "timeline": "Offer phased implementation to spread costs and minimize disruption"
            },
            expected_win_rate=0.70,
            avg_deal_size_multiplier=1.8,
            implementation_steps=[
                "System integration assessment",
                "Integration architecture design",
                "Security and compliance review",
                "Phased integration deployment",
                "User training and support"
            ],
            success_metrics=["data_integration_score", "workflow_efficiency", "user_satisfaction"]
        )
        
        # AI Module Expansion (Multiple Modules)
        self.playbooks["ai_module_expansion"] = ExpansionPlaybook(
            playbook_id="ai_module_expansion",
            name="Advanced AI Module Suite",
            target_customer_tier="enterprise",
            expansion_type=ExpansionType.MODULE_EXPANSION,
            trigger_conditions=[ExpansionTrigger.OUTCOME_IMPROVEMENT, ExpansionTrigger.CONTRACT_RENEWAL],
            sales_messaging="Unlock the full potential of AI-powered healthcare with comprehensive module suite",
            value_proposition="Transform your entire healthcare delivery with industry-leading AI capabilities",
            objection_handling={
                "scope": "Start with highest-impact modules and expand gradually",
                "resources": "Provide dedicated implementation team and change management",
                "vendor_risk": "Highlight proven track record and comprehensive support"
            },
            expected_win_rate=0.60,
            avg_deal_size_multiplier=3.0,
            implementation_steps=[
                "AI readiness assessment",
                "Priority module selection",
                "Implementation roadmap",
                "Phased module deployment",
                "Performance optimization"
            ],
            success_metrics=["comprehensive_ai_score", "clinical_transformation", "competitive_advantage"]
        )
    
    def identify_expansion_opportunities(self, customer_id: str, customer_data: Dict) -> List[ExpansionOpportunity]:
        """Identify expansion opportunities for a customer"""
        opportunities = []
        
        # Analyze current usage and performance
        current_annual_value = customer_data.get("annual_contract_value", 0)
        usage_metrics = customer_data.get("usage_metrics", {})
        
        # Check for high usage expansion opportunities
        if usage_metrics.get("overall_usage", 0) > 80:
            opportunities.extend(self._identify_upsell_opportunities(customer_id, customer_data, current_annual_value))
        
        # Check for cross-sell opportunities
        if self._assess_cross_sell_potential(customer_data):
            opportunities.extend(self._identify_cross_sell_opportunities(customer_id, customer_data, current_annual_value))
        
        # Check for integration opportunities
        if self._assess_integration_potential(customer_data):
            opportunities.extend(self._identify_integration_opportunities(customer_id, customer_data, current_annual_value))
        
        # Check for module expansion opportunities
        if self._assess_module_expansion_potential(customer_data):
            opportunities.extend(self._identify_module_expansion_opportunities(customer_id, customer_data, current_annual_value))
        
        # Store opportunities in pipeline
        if customer_id not in self.expansion_pipeline:
            self.expansion_pipeline[customer_id] = []
        self.expansion_pipeline[customer_id].extend(opportunities)
        
        return opportunities
    
    def _identify_upsell_opportunities(self, customer_id: str, customer_data: Dict, current_value: float) -> List[ExpansionOpportunity]:
        """Identify upsell opportunities based on usage patterns"""
        opportunities = []
        
        # Clinical Analytics Expansion
        if customer_data.get("has_analytics_basic") and customer_data.get("analytics_usage", 0) > 75:
            opp = ExpansionOpportunity(
                opportunity_id=f"{customer_id}_upsell_analytics_{datetime.datetime.now().strftime('%Y%m%d')}",
                customer_id=customer_id,
                expansion_type=ExpansionType.UPSELL,
                trigger=ExpansionTrigger.HIGH_USAGE,
                product_or_service="Advanced Clinical Analytics Module",
                description="Upgrade to advanced analytics with predictive modeling and outcome prediction",
                current_annual_value=current_value,
                potential_additional_value=current_value * 0.6,
                total_projected_value=current_value * 1.6,
                stage=ExpansionStage.IDENTIFICATION,
                probability=0.75,
                expected_close_date=datetime.date.today() + datetime.timedelta(days=90),
                business_case="Customer showing high usage of basic analytics, ready for advanced features",
                roi_projection=250.0,
                implementation_timeline="3-4 months"
            )
            opportunities.append(opp)
        
        # Workflow Automation Upsell
        if customer_data.get("workflow_count", 0) > 3 and customer_data.get("automation_usage", 0) > 70:
            opp = ExpansionOpportunity(
                opportunity_id=f"{customer_id}_upsell_automation_{datetime.datetime.now().strftime('%Y%m%d')}",
                customer_id=customer_id,
                expansion_type=ExpansionType.UPSELL,
                trigger=ExpansionTrigger.WORKFLOW_OPTIMIZATION,
                product_or_service="Advanced Workflow Automation",
                description="Expand automation capabilities to cover additional clinical workflows",
                current_annual_value=current_value,
                potential_additional_value=current_value * 0.5,
                total_projected_value=current_value * 1.5,
                stage=ExpansionStage.IDENTIFICATION,
                probability=0.70,
                expected_close_date=datetime.date.today() + datetime.timedelta(days=120),
                business_case="Multiple workflows automated successfully, ready for expansion",
                roi_projection=200.0,
                implementation_timeline="4-5 months"
            )
            opportunities.append(opp)
        
        return opportunities
    
    def _identify_cross_sell_opportunities(self, customer_id: str, customer_data: Dict, current_value: float) -> List[ExpansionOpportunity]:
        """Identify cross-sell opportunities"""
        opportunities = []
        
        # Patient Engagement Platform
        if not customer_data.get("has_patient_engagement") and customer_data.get("patient_satisfaction_score", 0) > 75:
            opp = ExpansionOpportunity(
                opportunity_id=f"{customer_id}_crosssell_patient_engagement_{datetime.datetime.now().strftime('%Y%m%d')}",
                customer_id=customer_id,
                expansion_type=ExpansionType.CROSS_SELL,
                trigger=ExpansionTrigger.FEATURE_SUCCESS,
                product_or_service="Patient Engagement AI Platform",
                description="Enhance patient satisfaction and outcomes with AI-driven engagement",
                current_annual_value=current_value,
                potential_additional_value=current_value * 0.8,
                total_projected_value=current_value * 1.8,
                stage=ExpansionStage.IDENTIFICATION,
                probability=0.65,
                expected_close_date=datetime.date.today() + datetime.timedelta(days=150),
                business_case="High patient satisfaction scores indicate readiness for engagement tools",
                roi_projection=180.0,
                implementation_timeline="5-6 months"
            )
            opportunities.append(opp)
        
        # Population Health Management
        if customer_data.get("organization_size") in ["large", "enterprise"] and not customer_data.get("has_population_health"):
            opp = ExpansionOpportunity(
                opportunity_id=f"{customer_id}_crosssell_population_health_{datetime.datetime.now().strftime('%Y%m%d')}",
                customer_id=customer_id,
                expansion_type=ExpansionType.CROSS_SELL,
                trigger=ExpansionTrigger.OUTCOME_IMPROVEMENT,
                product_or_service="Population Health Management AI",
                description="Manage and improve outcomes for entire patient populations",
                current_annual_value=current_value,
                potential_additional_value=current_value * 1.2,
                total_projected_value=current_value * 2.2,
                stage=ExpansionStage.IDENTIFICATION,
                probability=0.55,
                expected_close_date=datetime.date.today() + datetime.timedelta(days=180),
                business_case="Proven clinical outcomes, organization size supports population health focus",
                roi_projection=300.0,
                implementation_timeline="6-8 months"
            )
            opportunities.append(opp)
        
        return opportunities
    
    def _identify_integration_opportunities(self, customer_id: str, customer_data: Dict, current_value: float) -> List[ExpansionOpportunity]:
        """Identify integration expansion opportunities"""
        opportunities = []
        
        # EMR Integration Expansion
        integration_count = customer_data.get("current_integrations", 0)
        if integration_count < 5:  # Assuming enterprise has 5+ integrations
            opp = ExpansionOpportunity(
                opportunity_id=f"{customer_id}_integration_expansion_{datetime.datetime.now().strftime('%Y%m%d')}",
                customer_id=customer_id,
                expansion_type=ExpansionType.INTEGRATION_EXPANSION,
                trigger=ExpansionTrigger.HIGH_USAGE,
                product_or_service="Enhanced Integration Package",
                description="Connect additional systems for seamless data flow and improved workflows",
                current_annual_value=current_value,
                potential_additional_value=current_value * 0.4,
                total_projected_value=current_value * 1.4,
                stage=ExpansionStage.IDENTIFICATION,
                probability=0.70,
                expected_close_date=datetime.date.today() + datetime.timedelta(days=90),
                business_case="High usage indicates need for more comprehensive integration",
                roi_projection=150.0,
                implementation_timeline="2-3 months"
            )
            opportunities.append(opp)
        
        return opportunities
    
    def _identify_module_expansion_opportunities(self, customer_id: str, customer_data: Dict, current_value: float) -> List[ExpansionOpportunity]:
        """Identify AI module expansion opportunities"""
        opportunities = []
        
        # Additional AI Modules
        current_modules = customer_data.get("current_ai_modules", [])
        available_modules = ["diagnosis_assistance", "treatment_optimization", "risk_prediction", "resource_optimization"]
        
        missing_modules = [m for m in available_modules if m not in current_modules]
        
        if missing_modules:
            opp = ExpansionOpportunity(
                opportunity_id=f"{customer_id}_module_expansion_{datetime.datetime.now().strftime('%Y%m%d')}",
                customer_id=customer_id,
                expansion_type=ExpansionType.MODULE_EXPANSION,
                trigger=ExpansionTrigger.OUTCOME_IMPROVEMENT,
                product_or_service="Additional AI Modules Suite",
                description=f"Expand AI capabilities with {', '.join(missing_modules[:2])}",
                current_annual_value=current_value,
                potential_additional_value=current_value * 0.9,
                total_projected_value=current_value * 1.9,
                stage=ExpansionStage.IDENTIFICATION,
                probability=0.60,
                expected_close_date=datetime.date.today() + datetime.timedelta(days=120),
                business_case="Successful current AI implementation indicates readiness for expansion",
                roi_projection=220.0,
                implementation_timeline="4-6 months"
            )
            opportunities.append(opp)
        
        return opportunities
    
    def _assess_cross_sell_potential(self, customer_data: Dict) -> bool:
        """Assess if customer has cross-sell potential"""
        # Check organizational readiness
        if customer_data.get("organization_size") not in ["medium", "large", "enterprise"]:
            return False
        
        # Check current performance
        if customer_data.get("customer_satisfaction", 0) < 70:
            return False
        
        # Check usage patterns
        if customer_data.get("usage_consistency", 0) < 0.8:
            return False
        
        return True
    
    def _assess_integration_potential(self, customer_data: Dict) -> bool:
        """Assess if customer needs integration expansion"""
        current_integrations = customer_data.get("current_integrations", 0)
        tech_sophistication = customer_data.get("tech_sophistication", "medium")
        
        # Enterprise customers should have more integrations
        expected_integrations = {
            "small": 2,
            "medium": 4,
            "large": 6,
            "enterprise": 8
        }
        
        org_size = customer_data.get("organization_size", "medium")
        return current_integrations < expected_integrations.get(org_size, 4) and tech_sophistication in ["medium", "high"]
    
    def _assess_module_expansion_potential(self, customer_data: Dict) -> bool:
        """Assess if customer is ready for module expansion"""
        # Check if they're using core modules successfully
        current_modules = customer_data.get("current_ai_modules", [])
        if len(current_modules) < 2:
            return False
        
        # Check performance metrics
        if customer_data.get("ai_performance_score", 0) < 75:
            return False
        
        # Check organizational maturity
        if customer_data.get("ai_maturity_level", 0) < 3:  # Assuming 1-5 scale
            return False
        
        return True
    
    def calculate_expansion_pipeline_value(self, customer_id: Optional[str] = None) -> Dict:
        """Calculate total pipeline value for expansion opportunities"""
        if customer_id:
            opportunities = self.expansion_pipeline.get(customer_id, [])
        else:
            opportunities = list(self.opportunities.values())
        
        if not opportunities:
            return {
                "total_pipeline_value": 0,
                "weighted_pipeline_value": 0,
                "opportunity_count": 0,
                "average_deal_size": 0
            }
        
        total_pipeline_value = sum(opp.potential_additional_value for opp in opportunities)
        weighted_pipeline_value = sum(
            opp.potential_additional_value * opp.probability for opp in opportunities
        )
        
        return {
            "total_pipeline_value": total_pipeline_value,
            "weighted_pipeline_value": weighted_pipeline_value,
            "opportunity_count": len(opportunities),
            "average_deal_size": total_pipeline_value / len(opportunities),
            "stage_breakdown": self._get_pipeline_stage_breakdown(opportunities),
            "type_breakdown": self._get_pipeline_type_breakdown(opportunities)
        }
    
    def _get_pipeline_stage_breakdown(self, opportunities: List[ExpansionOpportunity]) -> Dict:
        """Breakdown opportunities by stage"""
        stage_breakdown = {}
        for opp in opportunities:
            stage = opp.stage.value
            if stage not in stage_breakdown:
                stage_breakdown[stage] = {"count": 0, "value": 0}
            stage_breakdown[stage]["count"] += 1
            stage_breakdown[stage]["value"] += opp.potential_additional_value
        
        return stage_breakdown
    
    def _get_pipeline_type_breakdown(self, opportunities: List[ExpansionOpportunity]) -> Dict:
        """Breakdown opportunities by type"""
        type_breakdown = {}
        for opp in opportunities:
            exp_type = opp.expansion_type.value
            if exp_type not in type_breakdown:
                type_breakdown[exp_type] = {"count": 0, "value": 0}
            type_breakdown[exp_type]["count"] += 1
            type_breakdown[exp_type]["value"] += opp.potential_additional_value
        
        return type_breakdown
    
    def launch_expansion_campaign(self, campaign_config: Dict) -> ExpansionCampaign:
        """Launch targeted expansion campaign"""
        campaign = ExpansionCampaign(
            campaign_id=f"expansion_{campaign_config.get('name', 'campaign')}_{datetime.datetime.now().strftime('%Y%m%d')}",
            name=campaign_config.get("name", "Expansion Campaign"),
            target_segment=campaign_config.get("target_segment", "all"),
            expansion_types=campaign_config.get("expansion_types", [ExpansionType.UPSELL]),
            triggers=campaign_config.get("triggers", [ExpansionTrigger.HIGH_USAGE]),
            start_date=datetime.date.today(),
            end_date=datetime.date.today() + datetime.timedelta(days=90),
            target_customers=campaign_config.get("target_customers", []),
            goals=campaign_config.get("goals", {"revenue_target": 500000, "win_rate_target": 0.65}),
            budget=campaign_config.get("budget", 50000),
            messaging_strategy=campaign_config.get("messaging_strategy", "Value-focused expansion messaging")
        )
        
        self.campaigns[campaign.campaign_id] = campaign
        
        # Auto-create opportunities for target customers
        if campaign.target_customers:
            for customer_id in campaign.target_customers:
                # Would need actual customer data to create opportunities
                # For now, just store the campaign
                pass
        
        self.logger.info(f"Launched expansion campaign: {campaign.name}")
        return campaign
    
    def generate_expansion_forecast(self, timeframe_months: int = 12) -> Dict:
        """Generate expansion revenue forecast"""
        opportunities = list(self.opportunities.values())
        
        # Filter opportunities by expected close date
        forecast_end_date = datetime.date.today() + datetime.timedelta(days=timeframe_months * 30)
        relevant_opportunities = [
            opp for opp in opportunities 
            if opp.expected_close_date <= forecast_end_date
        ]
        
        # Group by quarter
        quarterly_forecast = {}
        for opp in relevant_opportunities:
            quarter = self._get_quarter_from_date(opp.expected_close_date)
            if quarter not in quarterly_forecast:
                quarterly_forecast[quarter] = {
                    "opportunities": [],
                    "total_value": 0,
                    "weighted_value": 0
                }
            
            quarterly_forecast[quarter]["opportunities"].append(opp.opportunity_id)
            quarterly_forecast[quarter]["total_value"] += opp.potential_additional_value
            quarterly_forecast[quarter]["weighted_value"] += opp.potential_additional_value * opp.probability
        
        # Calculate overall metrics
        total_pipeline_value = sum(opp.potential_additional_value for opp in relevant_opportunities)
        total_weighted_value = sum(opp.potential_additional_value * opp.probability for opp in relevant_opportunities)
        
        return {
            "forecast_period": {
                "start_date": datetime.date.today(),
                "end_date": forecast_end_date,
                "months": timeframe_months
            },
            "overall_metrics": {
                "total_pipeline_value": total_pipeline_value,
                "total_weighted_value": total_weighted_value,
                "opportunity_count": len(relevant_opportunities),
                "expected_win_rate": len(relevant_opportunities) / len(opportunities) if opportunities else 0
            },
            "quarterly_breakdown": quarterly_forecast,
            "type_breakdown": self._get_pipeline_type_breakdown(relevant_opportunities),
            "top_opportunities": self._get_top_opportunities(relevant_opportunities, limit=10)
        }
    
    def _get_quarter_from_date(self, date: datetime.date) -> str:
        """Get quarter string from date"""
        quarter = (date.month - 1) // 3 + 1
        return f"Q{quarter} {date.year}"
    
    def _get_top_opportunities(self, opportunities: List[ExpansionOpportunity], limit: int = 10) -> List[Dict]:
        """Get top opportunities by weighted value"""
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda opp: opp.potential_additional_value * opp.probability, 
            reverse=True
        )
        
        return [
            {
                "opportunity_id": opp.opportunity_id,
                "customer_id": opp.customer_id,
                "product_or_service": opp.product_or_service,
                "potential_value": opp.potential_additional_value,
                "weighted_value": opp.potential_additional_value * opp.probability,
                "probability": opp.probability,
                "expected_close_date": opp.expected_close_date,
                "stage": opp.stage.value
            }
            for opp in sorted_opportunities[:limit]
        ]
    
    def update_opportunity_stage(self, opportunity_id: str, new_stage: ExpansionStage, 
                               probability: Optional[float] = None) -> bool:
        """Update opportunity stage and probability"""
        if opportunity_id not in self.opportunities:
            return False
        
        opportunity = self.opportunities[opportunity_id]
        opportunity.stage = new_stage
        opportunity.last_updated = datetime.datetime.now()
        
        if probability is not None:
            opportunity.probability = probability
        
        # Update opportunity in pipeline
        customer_opportunities = self.expansion_pipeline.get(opportunity.customer_id, [])
        for i, opp in enumerate(customer_opportunities):
            if opp.opportunity_id == opportunity_id:
                customer_opportunities[i] = opportunity
                break
        
        self.logger.info(f"Updated opportunity {opportunity_id} to stage {new_stage.value}")
        return True