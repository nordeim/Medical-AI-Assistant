"""
Customer Data Models for Business Intelligence
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal
import json

@dataclass
class Customer:
    """Customer entity for BI analysis"""
    customer_id: str
    company_name: str
    industry: str
    employee_count: int
    annual_revenue: Decimal
    customer_tier: str  # 'Enterprise', 'Mid-Market', 'SMB'
    acquisition_date: date
    acquisition_channel: str
    acquisition_cost: Decimal
    contract_value: Decimal
    contract_start: date
    contract_end: Optional[date]
    monthly_recurring_revenue: Decimal
    status: str  # 'Active', 'Churned', 'Suspended'
    onboarding_completion_date: Optional[date]
    last_interaction_date: Optional[date]
    satisfaction_score: Optional[float]  # 1-10 scale
    nps_score: Optional[int]  # -100 to 100 scale
    support_tickets_count: int
    feature_usage_score: float  # 0-100 scale
    renewal_probability: float  # 0-1 scale
    tags: List[str]
    custom_fields: Dict[str, Any]
    
    def calculate_ltv(self) -> Decimal:
        """Calculate customer lifetime value"""
        if self.monthly_recurring_revenue <= 0:
            return Decimal('0')
        
        # LTV = (Monthly Revenue × Gross Margin × Average Customer Lifespan) - Acquisition Cost
        gross_margin = Decimal('0.75')  # 75% gross margin assumption
        avg_lifespan_months = Decimal('36')  # 3 years average
        
        ltv = (self.monthly_recurring_revenue * gross_margin * avg_lifespan_months) - self.acquisition_cost
        return max(ltv, Decimal('0'))
    
    def calculate_payback_period(self) -> Decimal:
        """Calculate customer payback period in months"""
        if self.acquisition_cost <= 0 or self.monthly_recurring_revenue <= 0:
            return Decimal('0')
        
        # Payback Period = Acquisition Cost / Monthly Revenue
        payback = self.acquisition_cost / self.monthly_recurring_revenue
        return payback
    
    def get_customer_age_days(self) -> int:
        """Get customer age in days"""
        return (date.today() - self.acquisition_date).days
    
    def get_customer_age_months(self) -> int:
        """Get customer age in months"""
        return int(self.get_customer_age_days() / 30.44)  # Average days per month

@dataclass
class CustomerCohort:
    """Customer cohort for retention and behavior analysis"""
    cohort_id: str
    cohort_month: date  # YYYY-MM format
    cohort_size: int
    acquisition_channels: List[str]
    industries: List[str]
    customer_tiers: List[str]
    total_revenue: Decimal
    total_acquisition_cost: Decimal
    retention_rates: Dict[int, float]  # month_number -> retention_rate
    revenue_by_month: Dict[int, Decimal]  # month_number -> revenue
    churn_rates: Dict[int, float]  # month_number -> churn_rate
    ltv_trends: Dict[int, Decimal]  # month_number -> ltv
    
    def get_monthly_revenue_retention(self, month_number: int) -> float:
        """Get revenue retention for specific month"""
        if month_number == 0:
            return 1.0
        
        current_month_revenue = self.revenue_by_month.get(month_number, Decimal('0'))
        initial_month_revenue = self.revenue_by_month.get(0, Decimal('1'))
        
        if initial_month_revenue <= 0:
            return 0.0
        
        return float(current_month_revenue / initial_month_revenue)
    
    def get_cumulative_ltv(self, month_number: int) -> Decimal:
        """Get cumulative LTV up to specific month"""
        cumulative_ltv = Decimal('0')
        for i in range(month_number + 1):
            cumulative_ltv += self.ltv_trends.get(i, Decimal('0'))
        return cumulative_ltv

@dataclass
class CustomerLifetimeValue:
    """Customer LTV analysis and tracking"""
    customer_id: str
    current_ltv: Decimal
    predicted_ltv: Decimal
    ltv_to_acquisition_cost_ratio: float
    payback_period_months: Decimal
    ltv_cohort_percentile: float  # 0-100 percentile
    risk_score: float  # 0-1 scale, higher = more risk
    expansion_revenue: Decimal
    contraction_revenue: Decimal
    churn_likelihood: float
    upgrade_probability: float
    downgrade_probability: float
    recommended_actions: List[str]
    ltv_components: Dict[str, Decimal]
    
    def get_ltv_health_score(self) -> float:
        """Calculate overall LTV health score (0-100)"""
        # Factors: LTV/CAC ratio, payback period, expansion potential, churn risk
        ratio_score = min(self.ltv_to_acquisition_cost_ratio / 3.0, 1.0) * 30  # Max 30 points
        payback_score = max(0, (24 - float(self.payback_period_months)) / 24) * 25  # Max 25 points
        expansion_score = min(float(self.expansion_revenue) / 10000, 1.0) * 25  # Max 25 points
        churn_score = (1 - self.churn_likelihood) * 20  # Max 20 points
        
        return min(ratio_score + payback_score + expansion_score + churn_score, 100)
    
    def get_expansion_opportunities(self) -> List[Dict[str, Any]]:
        """Identify expansion opportunities"""
        opportunities = []
        
        if self.expansion_revenue > 0:
            opportunities.append({
                'type': 'Upsell',
                'potential_revenue': self.expansion_revenue,
                'probability': self.upgrade_probability,
                'recommendation': 'Increase usage of premium features'
            })
        
        if self.churn_likelihood > 0.3:
            opportunities.append({
                'type': 'Retention',
                'potential_revenue': self.current_ltv,
                'probability': 1 - self.churn_likelihood,
                'recommendation': 'Proactive engagement to prevent churn'
            })
        
        return opportunities