"""
Sales Data Models for Business Intelligence
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum

class DealStage(Enum):
    """Sales deal stages"""
    LEAD = "lead"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"

class LeadSource(Enum):
    """Lead source tracking"""
    WEBSITE = "website"
    REFERRAL = "referral"
    COLD_OUTREACH = "cold_outreach"
    WEBINAR = "webinar"
    TRADE_SHOW = "trade_show"
    PARTNER = "partner"
    MARKETING_CAMPAIGN = "marketing_campaign"
    INBOUND_SALES = "inbound_sales"

@dataclass
class SalesMetrics:
    """Sales performance metrics"""
    period_start: date
    period_end: date
    total_leads: int
    qualified_leads: int
    proposals_sent: int
    deals_won: int
    deals_lost: int
    total_pipeline_value: Decimal
    closed_won_value: Decimal
    average_deal_size: Decimal
    average_sales_cycle_days: int
    lead_to_close_rate: float
    win_rate: float
    revenue: Decimal
    quota_attainment: float
    new_customers: int
    upsells: int
    cross_sells: int
    churned_customers: int
    net_revenue_retention: float
    
    def calculate_sales_velocity(self) -> Decimal:
        """Calculate sales velocity (revenue per day)"""
        days = (self.period_end - self.period_start).days
        if days <= 0:
            return Decimal('0')
        return self.revenue / days
    
    def get_conversion_funnel(self) -> Dict[str, Dict[str, Any]]:
        """Get conversion funnel metrics"""
        return {
            'lead_to_qualified': {
                'count': self.qualified_leads,
                'rate': self.qualified_leads / max(self.total_leads, 1),
                'conversion_rate': self.qualified_leads / max(self.total_leads, 1) * 100
            },
            'qualified_to_proposal': {
                'count': self.proposals_sent,
                'rate': self.proposals_sent / max(self.qualified_leads, 1),
                'conversion_rate': self.proposals_sent / max(self.qualified_leads, 1) * 100
            },
            'proposal_to_close': {
                'count': self.deals_won,
                'rate': self.deals_won / max(self.proposals_sent, 1),
                'conversion_rate': self.deals_won / max(self.proposals_sent, 1) * 100
            }
        }

@dataclass
class DealPipeline:
    """Sales pipeline tracking"""
    deal_id: str
    customer_name: str
    deal_value: Decimal
    stage: DealStage
    probability: float
    expected_close_date: date
    actual_close_date: Optional[date]
    sales_rep: str
    lead_source: LeadSource
    days_in_stage: int
    total_days_in_pipeline: int
    last_activity_date: date
    next_activity_date: Optional[date]
    notes: str
    decision_makers: List[str]
    competitors: List[str]
    custom_fields: Dict[str, Any]
    
    def get_stage_probability(self) -> float:
        """Get probability based on stage"""
        stage_probabilities = {
            DealStage.LEAD: 0.10,
            DealStage.QUALIFIED: 0.25,
            DealStage.PROPOSAL: 0.50,
            DealStage.NEGOTIATION: 0.75,
            DealStage.CLOSED_WON: 1.0,
            DealStage.CLOSED_LOST: 0.0
        }
        return stage_probabilities.get(self.stage, 0.0)
    
    def get_weighted_value(self) -> Decimal:
        """Get weighted deal value based on probability"""
        return self.deal_value * Decimal(str(self.get_stage_probability()))
    
    def is_stale(self, days_threshold: int = 30) -> bool:
        """Check if deal is stale"""
        return self.days_in_stage > days_threshold

@dataclass
class RevenueMetrics:
    """Revenue tracking and analysis"""
    period_start: date
    period_end: date
    total_revenue: Decimal
    recurring_revenue: Decimal
    one_time_revenue: Decimal
    expansion_revenue: Decimal
    contraction_revenue: Decimal
    new_revenue: Decimal
    churned_revenue: Decimal
    net_revenue: Decimal
    gross_revenue: Decimal
    discounts: Decimal
    refunds: Decimal
    taxes: Decimal
    
    # Revenue by customer segment
    enterprise_revenue: Decimal
    mid_market_revenue: Decimal
    smb_revenue: Decimal
    
    # Revenue by product/service
    core_product_revenue: Decimal
    add_ons_revenue: Decimal
    services_revenue: Decimal
    
    # Growth metrics
    period_over_period_growth: float
    year_over_year_growth: float
    quarter_over_quarter_growth: float
    
    def get_net_revenue_retention(self) -> float:
        """Calculate net revenue retention"""
        if self.recurring_revenue <= 0:
            return 0.0
        
        # NRR = (Starting MRR + Expansion - Churn) / Starting MRR
        # Using period metrics as proxy
        starting_mrr = self.recurring_revenue - self.new_revenue + self.churned_revenue
        if starting_mrr <= 0:
            return 0.0
        
        nrr = (starting_mrr + self.expansion_revenue - self.churned_revenue) / starting_mrr
        return nrr * 100
    
    def get_revenue_breakdown_by_segment(self) -> Dict[str, Dict[str, Any]]:
        """Get revenue breakdown by customer segment"""
        total = self.total_revenue
        if total <= 0:
            return {}
        
        return {
            'Enterprise': {
                'revenue': self.enterprise_revenue,
                'percentage': float(self.enterprise_revenue / total) * 100,
                'growth_rate': self.period_over_period_growth  # Simplified
            },
            'Mid-Market': {
                'revenue': self.mid_market_revenue,
                'percentage': float(self.mid_market_revenue / total) * 100,
                'growth_rate': self.period_over_period_growth
            },
            'SMB': {
                'revenue': self.smb_revenue,
                'percentage': float(self.smb_revenue / total) * 100,
                'growth_rate': self.period_over_period_growth
            }
        }