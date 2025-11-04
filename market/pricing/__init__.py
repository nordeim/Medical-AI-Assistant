"""
Revenue Optimization and Pricing Framework

A comprehensive pricing framework for healthcare AI market segments including:
- Market segment analysis and optimization
- Value-based pricing strategies
- Revenue operations and forecasting
- ROI calculations and financial modeling
- Competitive pricing analysis
- Customer lifetime value calculations
"""

# Import all components with absolute imports
try:
    # For direct script execution
    from pricing_framework import (
        HealthcarePricingFramework,
        MarketSegment,
        PricingModel,
        CustomerTier,
        CustomerProfile,
        PricingStrategy,
        ClinicalOutcome,
        RevenueAttribution,
        ROI_Result
    )

    from market_segment_analysis import (
        MarketSegmentAnalyzer,
        MarketMetrics,
        CompetitiveLandscape,
        SegmentStrategy
    )

    from revenue_operations import (
        RevenueOperations,
        Deal,
        Customer,
        DealStage,
        RevenueType,
        MarketingAttribution,
        ForecastScenario
    )

    from roi_calculators import (
        HealthcareROICalculator,
        CostComponent,
        BenefitComponent,
        ClinicalMetric,
        OperationalMetric,
        OutcomeCategory,
        TimeHorizon,
        ROIAnalysis
    )
except ImportError:
    # For package imports
    from .pricing_framework import (
        HealthcarePricingFramework,
        MarketSegment,
        PricingModel,
        CustomerTier,
        CustomerProfile,
        PricingStrategy,
        ClinicalOutcome,
        RevenueAttribution,
        ROI_Result
    )

    from .market_segment_analysis import (
        MarketSegmentAnalyzer,
        MarketMetrics,
        CompetitiveLandscape,
        SegmentStrategy
    )

    from .revenue_operations import (
        RevenueOperations,
        Deal,
        Customer,
        DealStage,
        RevenueType,
        MarketingAttribution,
        ForecastScenario
    )

    from .roi_calculators import (
        HealthcareROICalculator,
        CostComponent,
        BenefitComponent,
        ClinicalMetric,
        OperationalMetric,
        OutcomeCategory,
        TimeHorizon,
        ROIAnalysis
    )

# Version information
__version__ = "1.0.0"
__author__ = "Healthcare AI Revenue Optimization Team"

# Package metadata
__all__ = [
    # Core framework
    "HealthcarePricingFramework",
    "MarketSegment",
    "PricingModel", 
    "CustomerTier",
    "CustomerProfile",
    "PricingStrategy",
    "ClinicalOutcome",
    "RevenueAttribution",
    "ROI_Result",
    
    # Market analysis
    "MarketSegmentAnalyzer",
    "MarketMetrics",
    "CompetitiveLandscape", 
    "SegmentStrategy",
    
    # Revenue operations
    "RevenueOperations",
    "Deal",
    "Customer",
    "DealStage",
    "RevenueType",
    "MarketingAttribution",
    "ForecastScenario",
    
    # ROI calculations
    "HealthcareROICalculator",
    "CostComponent",
    "BenefitComponent",
    "ClinicalMetric",
    "OperationalMetric",
    "OutcomeCategory",
    "TimeHorizon",
    "ROIAnalysis"
]

# Convenience factory functions
def create_pricing_framework() -> HealthcarePricingFramework:
    """Create and initialize a new pricing framework instance"""
    return HealthcarePricingFramework()

def create_market_analyzer() -> MarketSegmentAnalyzer:
    """Create and initialize a new market segment analyzer"""
    return MarketSegmentAnalyzer()

def create_revenue_operations() -> RevenueOperations:
    """Create and initialize a new revenue operations instance"""
    return RevenueOperations()

def create_roi_calculator() -> HealthcareROICalculator:
    """Create and initialize a new ROI calculator"""
    return HealthcareROICalculator()

# Configuration defaults
DEFAULT_CONFIG = {
    "market_segments": {
        "hospital_system": {
            "base_price": 500000,
            "win_rate": 0.25,
            "sales_cycle_days": 180,
            "avg_deal_size": 450000
        },
        "academic_medical_center": {
            "base_price": 300000,
            "win_rate": 0.30,
            "sales_cycle_days": 240,
            "avg_deal_size": 320000
        },
        "clinic": {
            "base_price": 120000,
            "win_rate": 0.35,
            "sales_cycle_days": 90,
            "avg_deal_size": 85000
        }
    },
    "pricing_models": {
        "subscription": {
            "annual_multiplier": 1.0,
            "discount_tiers": {
                "platinum": 1.0,
                "gold": 0.9,
                "silver": 0.8,
                "bronze": 0.7
            }
        },
        "enterprise_license": {
            "setup_fee_multiplier": 0.2,
            "annual_maintenance": 0.15
        }
    },
    "roi_parameters": {
        "default_discount_rate": 0.08,
        "default_time_horizon_months": 36,
        "confidence_threshold": 0.8
    }
}