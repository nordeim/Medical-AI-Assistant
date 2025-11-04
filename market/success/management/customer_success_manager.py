"""
Customer Success Management System
Enterprise healthcare customer success management with medical AI focus
"""

import asyncio
import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from config.framework_config import (
    CustomerTier, HealthcareKPI, CustomerSegment, SuccessMetrics,
    HealthcareCSConfig, HealthScoreStatus, InterventionLevel
)

@dataclass
class CustomerProfile:
    """Comprehensive customer profile for healthcare organizations"""
    customer_id: str
    organization_name: str
    customer_tier: CustomerTier
    segment: CustomerSegment
    
    # Contact Information
    primary_contact: str = ""
    email: str = ""
    phone: str = ""
    csm_assigned: str = ""
    
    # Contract Information
    contract_start_date: datetime.date = field(default_factory=datetime.date.today)
    contract_end_date: Optional[datetime.date] = None
    contract_value: float = 0.0
    renewal_date: Optional[datetime.date] = None
    
    # Current Metrics
    current_metrics: SuccessMetrics = field(default_factory=lambda: SuccessMetrics(
        customer_health_score=75.0, nps_score=7.0, churn_risk_score=0.2,
        expansion_potential=0.6, engagement_level=0.7, clinical_impact_score=0.8,
        support_ticket_volume=1, feature_adoption_rate=0.6, roi_delivery_score=0.75
    ))
    
    # Healthcare-specific KPIs
    healthcare_kpis: HealthcareKPI = field(default_factory=lambda: HealthcareKPI(
        clinical_outcome_improvement=15.0, clinical_efficiency_gain=20.0,
        cost_reduction=12.0, compliance_score=95.0, staff_satisfaction=85.0,
        patient_satisfaction=88.0, roi_percentage=25.0, implementation_success_rate=90.0
    ))
    
    # Additional Information
    success_stories: List[str] = field(default_factory=list)
    pain_points: List[str] = field(default_factory=list)
    expansion_opportunities: List[str] = field(default_factory=list)
    last_health_check: Optional[datetime.datetime] = None

@dataclass
class CSMActivity:
    """Customer Success Manager activity tracking"""
    activity_id: str
    customer_id: str
    csm_name: str
    activity_type: str  # check_in, implementation_review, qbr, training, etc.
    description: str
    scheduled_date: datetime.datetime
    completed_date: Optional[datetime.datetime] = None
    outcome: str = ""
    notes: str = ""
    next_steps: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

@dataclass
class HealthScoreCalculation:
    """Health score calculation breakdown"""
    customer_id: str
    total_score: float
    component_scores: Dict[str, float]
    status: HealthScoreStatus
    last_updated: datetime.datetime
    factors_impacting_score: List[str]
    recommended_actions: List[str]

class HealthcareCSManager:
    """Healthcare Customer Success Manager"""
    
    def __init__(self):
        self.customers: Dict[str, CustomerProfile] = {}
        self.activities: Dict[str, CSMActivity] = {}
        self.health_calculations: Dict[str, HealthScoreCalculation] = {}
        self.csm_workload: Dict[str, List[str]] = {}  # CSM -> customer_ids
        self.logger = logging.getLogger(__name__)
    
    def add_customer(self, customer: CustomerProfile) -> bool:
        """Add a new customer to the system"""
        try:
            self.customers[customer.customer_id] = customer
            
            # Update CSM workload
            if customer.csm_assigned not in self.csm_workload:
                self.csm_workload[customer.csm_assigned] = []
            self.csm_workload[customer.csm_assigned].append(customer.customer_id)
            
            # Check if workload exceeds limits
            workload_limit = HealthcareCSConfig.CSM_WORKLOAD_LIMITS[customer.customer_tier]
            if len(self.csm_workload[customer.csm_assigned]) > workload_limit:
                self.logger.warning(
                    f"CSM {customer.csm_assigned} workload exceeds limit for {customer.customer_tier}"
                )
            
            self.logger.info(f"Added customer {customer.organization_name} to system")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add customer: {e}")
            return False
    
    def calculate_health_score(self, customer_id: str) -> HealthScoreCalculation:
        """Calculate comprehensive health score for customer"""
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        customer = self.customers[customer_id]
        metrics = customer.current_metrics
        
        # Calculate component scores
        component_scores = {
            "product_usage": self._calculate_usage_score(customer),
            "clinical_outcomes": self._calculate_clinical_score(customer),
            "financial_health": self._calculate_financial_score(customer),
            "customer_satisfaction": self._calculate_satisfaction_score(customer),
            "support_health": self._calculate_support_score(customer),
            "engagement_level": self._calculate_engagement_score(customer)
        }
        
        # Calculate weighted total score
        total_score = sum(
            component_scores[component] * weight
            for component, weight in HealthcareCSConfig.HEALTH_SCORE_WEIGHTS.items()
        )
        
        # Determine status
        status = HealthcareCSConfig.get_health_score_status(total_score)
        
        # Identify factors impacting score
        impacting_factors = []
        recommended_actions = []
        
        for component, score in component_scores.items():
            if score < 60:  # Threshold for concern
                impacting_factors.append(f"{component}: {score:.1f}%")
                recommended_actions.extend(self._get_recommended_actions(component, score))
        
        # Create health score calculation
        health_calc = HealthScoreCalculation(
            customer_id=customer_id,
            total_score=total_score,
            component_scores=component_scores,
            status=status,
            last_updated=datetime.datetime.now(),
            factors_impacting_score=impacting_factors,
            recommended_actions=recommended_actions
        )
        
        self.health_calculations[customer_id] = health_calc
        customer.last_health_check = health_calc.last_updated
        
        return health_calc
    
    def _calculate_usage_score(self, customer: CustomerProfile) -> float:
        """Calculate product usage score"""
        # Base score from feature adoption
        base_score = customer.current_metrics.feature_adoption_rate * 100
        
        # Adjust based on license utilization (would need actual usage data)
        # For now, use engagement level as proxy
        usage_adjustment = customer.current_metrics.engagement_level * 10
        
        return min(100, base_score + usage_adjustment)
    
    def _calculate_clinical_score(self, customer: CustomerProfile) -> float:
        """Calculate clinical outcome score"""
        # Weighted average of clinical KPIs
        clinical_score = (
            customer.healthcare_kpis.clinical_outcome_improvement * 0.3 +
            customer.healthcare_kpis.clinical_efficiency_gain * 0.3 +
            customer.healthcare_kpis.compliance_score * 0.2 +
            customer.healthcare_kpis.implementation_success_rate * 0.2
        )
        
        return min(100, clinical_score)
    
    def _calculate_financial_score(self, customer: CustomerProfile) -> float:
        """Calculate financial health score"""
        # Based on ROI delivery, churn risk, and expansion potential
        roi_score = customer.current_metrics.roi_delivery_score * 100
        churn_penalty = customer.current_metrics.churn_risk_score * 50
        expansion_bonus = customer.current_metrics.expansion_potential * 20
        
        return max(0, min(100, roi_score - churn_penalty + expansion_bonus))
    
    def _calculate_satisfaction_score(self, customer: CustomerProfile) -> float:
        """Calculate customer satisfaction score"""
        # Combine NPS and satisfaction scores
        nps_score = customer.current_metrics.nps_score * 10  # Convert to 0-100 scale
        staff_satisfaction = customer.healthcare_kpis.staff_satisfaction
        patient_satisfaction = customer.healthcare_kpis.patient_satisfaction
        
        return (nps_score + staff_satisfaction + patient_satisfaction) / 3
    
    def _calculate_support_score(self, customer: CustomerProfile) -> float:
        """Calculate support health score"""
        # Lower ticket volume = higher score
        ticket_volume = customer.current_metrics.support_ticket_volume
        
        if ticket_volume <= 2:
            return 100
        elif ticket_volume <= 5:
            return 80
        elif ticket_volume <= 10:
            return 60
        else:
            return 40
    
    def _calculate_engagement_score(self, customer: CustomerProfile) -> float:
        """Calculate engagement level score"""
        return customer.current_metrics.engagement_level * 100
    
    def _get_recommended_actions(self, component: str, score: float) -> List[str]:
        """Get recommended actions for low-scoring components"""
        actions = []
        
        if component == "product_usage":
            actions.extend([
                "Schedule product training session",
                "Identify unused features and provide demos",
                "Create custom usage reports"
            ])
        elif component == "clinical_outcomes":
            actions.extend([
                "Review clinical workflow optimization opportunities",
                "Analyze outcome improvement metrics",
                "Schedule clinical best practices review"
            ])
        elif component == "financial_health":
            actions.extend([
                "Review ROI calculation and demonstration",
                "Identify additional value opportunities",
                "Schedule financial review meeting"
            ])
        elif component == "customer_satisfaction":
            actions.extend([
                "Conduct satisfaction survey",
                "Address any reported pain points",
                "Increase engagement touchpoints"
            ])
        elif component == "support_health":
            actions.extend([
                "Review support ticket trends",
                "Provide proactive support outreach",
                "Offer additional training resources"
            ])
        elif component == "engagement_level":
            actions.extend([
                "Schedule regular check-ins",
                "Invite to customer events",
                "Share success stories and best practices"
            ])
        
        return actions
    
    def schedule_customer_activity(self, activity: CSMActivity) -> bool:
        """Schedule a customer success activity"""
        try:
            # Check for scheduling conflicts
            existing_activities = [
                a for a in self.activities.values() 
                if a.customer_id == activity.customer_id and 
                a.scheduled_date.date() == activity.scheduled_date.date()
            ]
            
            if len(existing_activities) >= 3:  # Max 3 activities per day
                self.logger.warning("Too many activities scheduled for this day")
                return False
            
            self.activities[activity.activity_id] = activity
            self.logger.info(f"Scheduled {activity.activity_type} for customer {activity.customer_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to schedule activity: {e}")
            return False
    
    def get_at_risk_customers(self) -> List[CustomerProfile]:
        """Identify customers at risk of churn"""
        at_risk_customers = []
        
        for customer in self.customers.values():
            # Check multiple risk factors
            risk_score = 0
            
            # Health score risk
            if customer.current_metrics.customer_health_score < 50:
                risk_score += 30
            
            # Churn risk score
            risk_score += customer.current_metrics.churn_risk_score * 40
            
            # NPS risk
            if customer.current_metrics.nps_score < 5:
                risk_score += 20
            
            # Support issues
            if customer.current_metrics.support_ticket_volume > 5:
                risk_score += 10
            
            if risk_score >= 50:  # Risk threshold
                at_risk_customers.append(customer)
        
        return sorted(at_risk_customers, key=lambda c: c.current_metrics.churn_risk_score, reverse=True)
    
    def get_expansion_opportunities(self) -> List[CustomerProfile]:
        """Identify customers with expansion opportunities"""
        expansion_customers = []
        
        for customer in self.customers.values():
            # Calculate expansion score
            expansion_data = {
                "user_licenses": customer.current_metrics.feature_adoption_rate * 100,
                "feature_adoption": customer.current_metrics.feature_adoption_rate * 100,
                "workflow_optimization": customer.current_metrics.engagement_level * 100,
                "integrations": 50  # Would need actual integration usage data
            }
            
            expansion_score = HealthcareCSConfig.calculate_expansion_score(expansion_data)
            
            if expansion_score > 0.6:  # Expansion threshold
                customer.current_metrics.expansion_potential = expansion_score
                expansion_customers.append(customer)
        
        return sorted(expansion_customers, key=lambda c: c.current_metrics.expansion_potential, reverse=True)
    
    def generate_customer_summary(self, customer_id: str) -> Dict:
        """Generate comprehensive customer summary report"""
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        
        customer = self.customers[customer_id]
        
        # Get latest health calculation
        health_calc = self.health_calculations.get(customer_id)
        if not health_calc:
            health_calc = self.calculate_health_score(customer_id)
        
        # Calculate days to renewal
        days_to_renewal = 0
        if customer.renewal_date:
            days_to_renewal = (customer.renewal_date - datetime.date.today()).days
        
        # Generate summary
        summary = {
            "customer_info": {
                "customer_id": customer.customer_id,
                "organization_name": customer.organization_name,
                "tier": customer.customer_tier.value,
                "segment": customer.segment.segment_name,
                "csm_assigned": customer.csm_assigned
            },
            "health_score": {
                "current_score": health_calc.total_score,
                "status": health_calc.status.value,
                "component_breakdown": health_calc.component_scores,
                "last_updated": health_calc.last_updated
            },
            "key_metrics": {
                "nps_score": customer.current_metrics.nps_score,
                "churn_risk": customer.current_metrics.churn_risk_score,
                "expansion_potential": customer.current_metrics.expansion_potential,
                "engagement_level": customer.current_metrics.engagement_level,
                "clinical_impact": customer.current_metrics.clinical_impact_score,
                "roi_delivery": customer.current_metrics.roi_delivery_score
            },
            "healthcare_kpis": {
                "clinical_outcome_improvement": customer.healthcare_kpis.clinical_outcome_improvement,
                "clinical_efficiency_gain": customer.healthcare_kpis.clinical_efficiency_gain,
                "cost_reduction": customer.healthcare_kpis.cost_reduction,
                "compliance_score": customer.healthcare_kpis.compliance_score,
                "staff_satisfaction": customer.healthcare_kpis.staff_satisfaction,
                "patient_satisfaction": customer.healthcare_kpis.patient_satisfaction
            },
            "renewal_status": {
                "renewal_date": customer.renewal_date,
                "days_to_renewal": days_to_renewal,
                "contract_value": customer.contract_value
            },
            "risk_assessment": {
                "is_at_risk": customer.current_metrics.churn_risk_score > 0.5,
                "has_expansion_opportunity": customer.current_metrics.expansion_potential > 0.6,
                "intervention_needed": health_calc.status in [HealthScoreStatus.RED, HealthScoreStatus.CRITICAL]
            },
            "recommendations": {
                "immediate_actions": health_calc.recommended_actions[:3],
                "expansion_opportunities": customer.expansion_opportunities,
                "follow_up_activities": [activity for activity in self.activities.values() 
                                      if activity.customer_id == customer_id and 
                                      activity.scheduled_date > datetime.datetime.now()]
            }
        }
        
        return summary
    
    def get_csm_workload_report(self, csm_name: str) -> Dict:
        """Generate workload report for a specific CSM"""
        if csm_name not in self.csm_workload:
            return {"error": f"CSM {csm_name} not found"}
        
        customer_ids = self.csm_workload[csm_name]
        customers = [self.customers[cid] for cid in customer_ids if cid in self.customers]
        
        # Calculate workload metrics
        total_customers = len(customers)
        at_risk_count = len([c for c in customers if c.current_metrics.churn_risk_score > 0.5])
        expansion_opportunities = len([c for c in customers if c.current_metrics.expansion_potential > 0.6])
        
        # Get upcoming activities
        upcoming_activities = [
            activity for activity in self.activities.values()
            if activity.csm_name == csm_name and activity.scheduled_date > datetime.datetime.now()
        ]
        
        return {
            "csm_name": csm_name,
            "total_customers": total_customers,
            "tier_breakdown": {
                "enterprise": len([c for c in customers if c.customer_tier == CustomerTier.ENTERPRISE]),
                "premium": len([c for c in customers if c.customer_tier == CustomerTier.PREMIUM]),
                "standard": len([c for c in customers if c.customer_tier == CustomerTier.STANDARD]),
                "basic": len([c for c in customers if c.customer_tier == CustomerTier.BASIC])
            },
            "customer_health": {
                "green": len([c for c in customers if c.current_metrics.customer_health_score >= 80]),
                "yellow": len([c for c in customers if 60 <= c.current_metrics.customer_health_score < 80]),
                "red": len([c for c in customers if c.current_metrics.customer_health_score < 60])
            },
            "at_risk_customers": at_risk_count,
            "expansion_opportunities": expansion_opportunities,
            "upcoming_activities": len(upcoming_activities),
            "activity_breakdown": {
                activity_type: len([a for a in upcoming_activities if a.activity_type == activity_type])
                for activity_type in ["check_in", "qbr", "training", "implementation_review", "clinical_review"]
            },
            "revenue_metrics": {
                "total_arr": sum(c.contract_value for c in customers),
                "renewing_90_days": len([c for c in customers if c.renewal_date and 
                                       0 <= (c.renewal_date - datetime.date.today()).days <= 90])
            }
        }