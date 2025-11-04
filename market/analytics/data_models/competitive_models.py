"""
Competitive Intelligence Data Models for Market Share and Analysis
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum

class CompetitorType(Enum):
    """Types of competitors"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    EMERGING = "emerging"
    SUBSTITUTE = "substitute"

class PricingModel(Enum):
    """Pricing models"""
    SUBSCRIPTION = "subscription"
    PER_USER = "per_user"
    PER_TRANSACTION = "per_transaction"
    ONE_TIME = "one_time"
    FREEMIUM = "freemium"
    ENTERPRISE = "enterprise"

@dataclass
class MarketShare:
    """Market share analysis"""
    period_start: date
    period_end: date
    market_name: str
    market_size: Decimal
    
    # Market share data
    our_market_share: float
    competitors: List[Dict[str, Any]]  # competitor_name -> share_percentage
    
    # Growth metrics
    our_growth_rate: float
    market_growth_rate: float
    growth_vs_market: float
    market_momentum: str  # 'Growing', 'Stable', 'Declining'
    
    # Geographic data
    geographic_shares: Dict[str, float]  # region -> share_percentage
    
    def get_competitive_position(self) -> str:
        """Get competitive position based on market share"""
        share = self.our_market_share
        if share >= 50:
            return "Market Leader"
        elif share >= 25:
            return "Strong Competitor"
        elif share >= 10:
            return "Significant Player"
        elif share >= 5:
            return "Emerging Player"
        else:
            return "Niche Player"
    
    def get_market_opportunity_score(self) -> float:
        """Calculate market opportunity score"""
        # Factors: market growth rate, competitive intensity, our position
        
        # Growth opportunity (max 30 points)
        growth_score = min(self.market_growth_rate / 20, 1.0) * 30
        
        # Competitive pressure (max 35 points) - lower competition is better
        competition_score = (100 - max(self.our_market_share, 50)) / 100 * 35
        
        # Position strength (max 35 points)
        position_score = min(self.our_market_share / 30, 1.0) * 35
        
        return growth_score + competition_score + position_score

@dataclass
class CompetitiveAnalysis:
    """Detailed competitive analysis"""
    competitor_id: str
    competitor_name: str
    competitor_type: CompetitorType
    market_share: float
    annual_revenue: Optional[Decimal]
    employee_count: Optional[int]
    
    # Product analysis
    products: List[Dict[str, Any]]
    product_strengths: List[str]
    product_weaknesses: List[str]
    feature_gaps: List[str]
    
    # Pricing analysis
    pricing_model: PricingModel
    pricing_range_min: Optional[Decimal]
    pricing_range_max: Optional[Decimal]
    pricing_competitiveness: float  # 0-100 scale
    
    # Market position
    market_positioning: str
    target_audience: str
    value_proposition: str
    
    # Performance metrics
    customer_satisfaction: Optional[float]
    employee_count_growth: Optional[float]
    funding_stage: Optional[str]
    latest_funding: Optional[Decimal]
    
    # SWOT analysis
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]
    
    # Recent activities
    recent_product_releases: List[str]
    recent_partnerships: List[str]
    recent_funding: Optional[str]
    recent_hires: List[str]
    
    def get_threat_level(self) -> str:
        """Assess competitive threat level"""
        threat_score = 0
        
        # Market share threat
        if self.market_share > 20:
            threat_score += 30
        elif self.market_share > 10:
            threat_score += 20
        elif self.market_share > 5:
            threat_score += 10
        
        # Feature gap threat
        threat_score += len(self.feature_gaps) * 5
        
        # Recent activity threat
        recent_activity_score = len(self.recent_product_releases) + len(self.recent_partnerships)
        threat_score += recent_activity_score * 5
        
        if threat_score >= 50:
            return "High"
        elif threat_score >= 30:
            return "Medium"
        else:
            return "Low"

@dataclass
class Benchmarking:
    """Performance benchmarking against competitors"""
    metric_name: str
    metric_value: Decimal
    metric_unit: str
    benchmark_date: date
    
    # Benchmark data
    our_value: Decimal
    industry_average: Decimal
    best_in_class: Decimal
    worst_in_class: Decimal
    
    # Performance assessment
    percentile_rank: float  # 0-100
    performance_vs_average: float  # % vs industry average
    performance_vs_best: float  # % vs best in class
    
    # Competitive gap analysis
    gap_to_best: Decimal
    gap_to_average: Decimal
    improvement_potential: float  # 0-100 scale
    
    def get_performance_rating(self) -> str:
        """Get performance rating"""
        if self.percentile_rank >= 90:
            return "Excellent"
        elif self.percentile_rank >= 70:
            return "Good"
        elif self.percentile_rank >= 50:
            return "Average"
        elif self.percentile_rank >= 30:
            return "Below Average"
        else:
            return "Poor"
    
    def get_impriority_priority(self) -> str:
        """Get improvement priority"""
        if self.percentile_rank >= 75:
            return "Maintain Excellence"
        elif self.percentile_rank >= 50:
            return "Steady Improvement"
        elif self.percentile_rank >= 25:
            return "Urgent Improvement"
        else:
            return "Critical Focus"
    
    def calculate_improvement_impact(self, target_percentile: float) -> Dict[str, Any]:
        """Calculate impact of improving to target percentile"""
        if self.metric_value <= 0:
            return {'impact': 'Cannot calculate', 'target_value': None}
        
        # Assuming linear relationship for simplicity
        percentile_improvement = target_percentile - self.percentile_rank
        performance_improvement = (percentile_improvement / 100) * 2  # Simplified scaling
        
        # For revenue metrics, assume direct impact
        # For cost metrics, improvement means reduction
        if 'cost' in self.metric_name.lower() or 'expense' in self.metric_name.lower():
            current_value = self.our_value
            target_value = current_value * (1 - performance_improvement)
            impact = "Cost Reduction"
        else:
            current_value = self.our_value
            target_value = current_value * (1 + performance_improvement)
            impact = "Revenue/Performance Gain"
        
        return {
            'impact': impact,
            'target_value': target_value,
            'improvement_percentage': performance_improvement * 100,
            'target_percentile': target_percentile
        }