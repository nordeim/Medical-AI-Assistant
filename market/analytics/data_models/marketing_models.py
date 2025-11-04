"""
Marketing Data Models for Business Intelligence
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum

class CampaignType(Enum):
    """Marketing campaign types"""
    EMAIL = "email"
    PAID_SEARCH = "paid_search"
    SOCIAL_MEDIA = "social_media"
    CONTENT_MARKETING = "content_marketing"
    WEBINAR = "webinar"
    TRADE_SHOW = "trade_show"
    DIRECT_MAIL = "direct_mail"
    PARTNERSHIP = "partnership"
    EVENT = "event"
    INFLUENCER = "influencer"

class Channel(Enum):
    """Marketing channels"""
    WEBSITE = "website"
    BLOG = "blog"
    EMAIL = "email"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    GOOGLE_ADS = "google_ads"
    YOUTUBE = "youtube"
    WEBINAR = "webinar"
    PODCAST = "podcast"

@dataclass
class MarketingMetrics:
    """Marketing performance metrics"""
    period_start: date
    period_end: date
    
    # Traffic metrics
    website_visitors: int
    unique_visitors: int
    page_views: int
    bounce_rate: float
    average_session_duration: float
    pages_per_session: float
    return_visitor_rate: float
    
    # Lead generation
    leads_generated: int
    marketing_qualified_leads: int
    sales_qualified_leads: int
    lead_conversion_rate: float
    
    # Cost metrics
    total_marketing_spend: Decimal
    cost_per_lead: Decimal
    cost_per_qualified_lead: Decimal
    cost_per_acquisition: Decimal
    total_marketing_roi: float
    
    # Channel performance
    channel_performance: Dict[str, Dict[str, Any]]
    
    # Content metrics
    content_published: int
    content_engagement_score: float
    social_shares: int
    backlinks_acquired: int
    
    # Email marketing
    emails_sent: int
    email_open_rate: float
    email_click_rate: float
    email_unsubscribe_rate: float
    email_conversion_rate: float
    
    def get_marketing_roi(self) -> float:
        """Calculate marketing ROI"""
        if self.total_marketing_spend <= 0:
            return 0.0
        
        # Assuming revenue attribution - simplified calculation
        attributed_revenue = self.leads_generated * 500  # $500 avg per lead
        roi = (attributed_revenue - self.total_marketing_spend) / self.total_marketing_spend
        return roi * 100
    
    def get_lead_quality_score(self) -> float:
        """Calculate lead quality score"""
        if self.leads_generated <= 0:
            return 0.0
        
        qualified_ratio = self.marketing_qualified_leads / self.leads_generated
        conversion_ratio = self.sales_qualified_leads / self.marketing_qualified_leads
        
        # Weighted score (70% qualification rate, 30% conversion rate)
        quality_score = (qualified_ratio * 0.7) + (conversion_ratio * 0.3)
        return quality_score * 100

@dataclass
class CampaignPerformance:
    """Marketing campaign performance tracking"""
    campaign_id: str
    campaign_name: str
    campaign_type: CampaignType
    start_date: date
    end_date: Optional[date]
    budget: Decimal
    actual_spend: Decimal
    target_audience: str
    channels: List[Channel]
    
    # Performance metrics
    impressions: int
    clicks: int
    conversions: int
    cost_per_click: Decimal
    cost_per_conversion: Decimal
    conversion_rate: float
    click_through_rate: float
    
    # Lead metrics
    leads_generated: int
    qualified_leads: int
    opportunities_created: int
    revenue_attributed: Decimal
    
    # Engagement metrics
    engagement_rate: float
    share_rate: float
    comment_rate: float
    video_completion_rate: Optional[float]
    
    def get_campaign_roi(self) -> float:
        """Calculate campaign ROI"""
        if self.actual_spend <= 0:
            return 0.0
        
        roi = (self.revenue_attributed - self.actual_spend) / self.actual_spend
        return roi * 100
    
    def get_cost_efficiency(self) -> Decimal:
        """Get cost efficiency (lower is better)"""
        if self.conversions <= 0:
            return Decimal('999999')  # High penalty for no conversions
        
        return self.actual_spend / self.conversions
    
    def is_within_budget(self, tolerance: float = 0.1) -> bool:
        """Check if campaign is within budget"""
        return self.actual_spend <= self.budget * (1 + tolerance)

@dataclass
class CACAnalysis:
    """Customer Acquisition Cost analysis"""
    period_start: date
    period_end: date
    
    # Overall CAC metrics
    total_customer_acquisitions: int
    total_acquisition_cost: Decimal
    average_cac: Decimal
    
    # CAC by channel
    cac_by_channel: Dict[str, Decimal]
    
    # CAC by customer segment
    enterprise_cac: Decimal
    mid_market_cac: Decimal
    smb_cac: Decimal
    
    # LTV:CAC ratios
    overall_ltv_cac_ratio: float
    enterprise_ltv_cac_ratio: float
    mid_market_ltv_cac_ratio: float
    smb_ltv_cac_ratio: float
    
    # Payback periods
    average_payback_months: Decimal
    channel_payback_months: Dict[str, Decimal]
    
    # Efficiency metrics
    cac_trend: float  # % change period over period
    ltv_cac_ratio_trend: float
    
    def get_cac_health_score(self) -> float:
        """Calculate CAC health score (0-100)"""
        # Factors: LTV:CAC ratio, payback period, efficiency trends
        
        # LTV:CAC ratio score (max 40 points)
        ratio_score = min(self.overall_ltv_cac_ratio / 3.0, 1.0) * 40
        
        # Payback period score (max 30 points)
        payback_score = max(0, (24 - float(self.average_payback_months)) / 24) * 30
        
        # Efficiency trend score (max 30 points)
        if self.cac_trend < 0:  # CAC decreasing is good
            efficiency_score = min(abs(self.cac_trend) / 20, 1.0) * 30
        else:  # CAC increasing is bad
            efficiency_score = max(0, (20 - self.cac_trend) / 20) * 30
        
        return min(ratio_score + payback_score + efficiency_score, 100)
    
    def get_channel_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by channel"""
        performance = {}
        
        for channel, cac in self.cac_by_channel.items():
            payback = self.channel_payback_months.get(channel, Decimal('999'))
            
            performance[channel] = {
                'cac': cac,
                'payback_months': payback,
                'efficiency_score': self._calculate_channel_efficiency(cac, payback)
            }
        
        return performance
    
    def _calculate_channel_efficiency(self, cac: Decimal, payback: Decimal) -> float:
        """Calculate channel efficiency score"""
        # Lower CAC and shorter payback are better
        cac_score = max(0, (2000 - float(cac)) / 2000) * 50  # Assume $2000 is max good CAC
        payback_score = max(0, (24 - float(payback)) / 24) * 50
        
        return cac_score + payback_score