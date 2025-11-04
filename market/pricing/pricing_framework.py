"""
Healthcare AI Revenue Optimization and Pricing Framework

This module implements a comprehensive pricing strategy system for healthcare AI solutions,
including value-based pricing, subscription models, enterprise licensing, and revenue optimization.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid

class MarketSegment(Enum):
    """Healthcare market segments"""
    HOSPITAL_SYSTEM = "hospital_system"
    AMC = "academic_medical_center"
    CLINIC = "clinic"
    IDN = "integrated_delivery_network"
    REGIONAL_HOSPITAL = "regional_hospital"
    SPECIALTY_CLINIC = "specialty_clinic"
    RURAL_HOSPITAL = "rural_hospital"

class PricingModel(Enum):
    """Pricing model types"""
    SUBSCRIPTION = "subscription"
    ENTERPRISE_LICENSE = "enterprise_license"
    PER_PATIENT = "per_patient"
    PER_SEAT = "per_seat"
    OUTCOMES_BASED = "outcomes_based"
    HYBRID = "hybrid"

class CustomerTier(Enum):
    """Customer tier classifications"""
    PLATINUM = "platinum"
    GOLD = "gold"
    SILVER = "silver"
    BRONZE = "bronze"

@dataclass
class ClinicalOutcome:
    """Clinical outcome metrics for value-based pricing"""
    metric_name: str
    improvement_percentage: float
    baseline_value: float
    current_value: float
    measurement_period_months: int
    cost_per_unit: float  # Cost saved per unit improvement

@dataclass
class CustomerProfile:
    """Customer profile for value-based pricing"""
    customer_id: str
    organization_name: str
    market_segment: MarketSegment
    tier: CustomerTier
    annual_revenue: float
    patient_volume: int
    clinical_specialties: List[str]
    technology_adoption_score: float
    budget_range: Dict[str, float]  # min, max
    decision_makers: List[str]
    current_ai_spend: float
    competitive_solutions: List[str]

@dataclass
class PricingStrategy:
    """Pricing strategy configuration"""
    strategy_id: str
    market_segment: MarketSegment
    pricing_model: PricingModel
    base_price: float
    price_factors: Dict[str, float]  # multipliers for different factors
    value_metrics: List[str]  # metrics used for value calculation
    discount_tiers: Dict[CustomerTier, float]
    contractual_terms: Dict[str, str]
    success_metrics: List[str]

@dataclass
class RevenueAttribution:
    """Revenue attribution tracking"""
    attribution_id: str
    customer_id: str
    marketing_channel: str
    campaign_id: Optional[str]
    touchpoint_date: datetime
    deal_value: float
    sales_stage: str
    attribution_weight: float  # 0-1 for multi-touch attribution
    conversion_time_days: int

@dataclass
class ROI_Result:
    """ROI calculation results"""
    customer_id: str
    investment_amount: float
    total_savings: float
    total_benefits: float
    roi_percentage: float
    payback_period_months: float
    net_present_value: float
    clinical_outcomes: List[ClinicalOutcome]
    calculation_date: datetime

class HealthcarePricingFramework:
    """Main pricing framework class"""
    
    def __init__(self):
        self.pricing_strategies: Dict[str, PricingStrategy] = {}
        self.customer_profiles: Dict[str, CustomerProfile] = {}
        self.clinical_outcomes_db: Dict[str, List[ClinicalOutcome]] = {}
        self.revenue_attribution: List[RevenueAttribution] = []
        self.roi_cache: Dict[str, ROI_Result] = {}
        
    def add_customer_profile(self, profile: CustomerProfile) -> None:
        """Add customer profile to the framework"""
        self.customer_profiles[profile.customer_id] = profile
        
    def create_pricing_strategy(self, strategy: PricingStrategy) -> None:
        """Create pricing strategy for market segment"""
        self.pricing_strategies[strategy.strategy_id] = strategy
        
    def calculate_value_based_price(self, customer_id: str, 
                                  clinical_outcomes: List[ClinicalOutcome]) -> Tuple[float, Dict]:
        """Calculate value-based pricing using clinical outcomes"""
        profile = self.customer_profiles.get(customer_id)
        if not profile:
            raise ValueError(f"Customer profile not found: {customer_id}")
            
        # Get appropriate pricing strategy
        strategy = self._get_pricing_strategy(profile.market_segment)
        
        # Calculate value metrics
        total_value = 0
        value_breakdown = {}
        
        for outcome in clinical_outcomes:
            # Calculate financial value of clinical improvement
            improvement_value = (
                outcome.improvement_percentage / 100 * 
                outcome.baseline_value * 
                outcome.cost_per_unit
            )
            
            # Annualize the value
            annual_value = improvement_value * (12 / outcome.measurement_period_months)
            total_value += annual_value
            value_breakdown[outcome.metric_name] = {
                'improvement_percentage': outcome.improvement_percentage,
                'annual_value': annual_value,
                'cost_savings': improvement_value
            }
            
        # Apply value capture rate (typically 15-30% for B2B SaaS)
        value_capture_rate = 0.20
        captured_value = total_value * value_capture_rate
        
        # Apply customer tier discounts
        base_price = captured_value * strategy.discount_tiers.get(profile.tier, 1.0)
        
        # Apply market segment adjustments
        segment_multiplier = strategy.price_factors.get('market_segment', 1.0)
        final_price = base_price * segment_multiplier
        
        return final_price, {
            'total_annual_value': total_value,
            'captured_value': captured_value,
            'value_breakdown': value_breakdown,
            'base_price': base_price,
            'segment_multiplier': segment_multiplier,
            'customer_tier': profile.tier.value
        }
        
    def create_subscription_model(self, customer_id: str, 
                                subscription_tier: CustomerTier) -> Dict:
        """Create subscription pricing model"""
        profile = self.customer_profiles[customer_id]
        
        # Base subscription prices by market segment
        base_prices = {
            MarketSegment.HOSPITAL_SYSTEM: 500000,
            MarketSegment.AMC: 300000,
            MarketSegment.CLINIC: 120000,
            MarketSegment.IDN: 800000,
            MarketSegment.REGIONAL_HOSPITAL: 200000,
            MarketSegment.SPECIALTY_CLINIC: 150000,
            MarketSegment.RURAL_HOSPITAL: 80000
        }
        
        base_price = base_prices.get(profile.market_segment, 100000)
        
        # Volume discounts based on patient volume
        volume_discount = self._calculate_volume_discount(profile.patient_volume)
        
        # Technology adoption premium/discount
        tech_discount = self._calculate_tech_adoption_discount(profile.technology_adoption_score)
        
        # Tier-based pricing
        tier_multipliers = {
            CustomerTier.PLATINUM: 2.0,
            CustomerTier.GOLD: 1.5,
            CustomerTier.SILVER: 1.0,
            CustomerTier.BRONZE: 0.7
        }
        
        tier_multiplier = tier_multipliers.get(subscription_tier, 1.0)
        
        annual_subscription = base_price * volume_discount * tech_discount * tier_multiplier
        
        # Add usage-based components
        usage_components = self._calculate_usage_components(profile)
        total_monthly = (annual_subscription + usage_components['annual_extra']) / 12
        
        return {
            'base_annual': annual_subscription,
            'monthly_total': total_monthly,
            'usage_components': usage_components,
            'volume_discount': volume_discount,
            'tech_discount': tech_discount,
            'tier_multiplier': tier_multiplier,
            'contract_terms': self._get_contract_terms(profile.market_segment, subscription_tier)
        }
        
    def optimize_pricing(self, customer_id: str, competitive_analysis: Dict) -> Dict:
        """Optimize pricing based on customer value and competition"""
        profile = self.customer_profiles[customer_id]
        
        # Get competitive pricing
        if isinstance(competitive_analysis, dict) and competitive_analysis:
            try:
                if isinstance(list(competitive_analysis.values())[0], dict):
                    prices = [comp.get('price', 0) for comp in competitive_analysis.values()]
                    prices = [p for p in prices if isinstance(p, (int, float)) and p > 0]
                    avg_competitive_price = np.mean(prices) if prices else 0
                else:
                    # competitive_analysis is a dict of numbers
                    prices = [v for v in competitive_analysis.values() if isinstance(v, (int, float)) and v > 0]
                    avg_competitive_price = np.mean(prices) if prices else 0
            except (IndexError, ValueError, TypeError):
                avg_competitive_price = 0
        else:
            avg_competitive_price = 0
        
        # Calculate willingness to pay based on customer profile
        wtp = self._calculate_willingness_to_pay(profile)
        
        # Get current pricing recommendation
        current_strategy = self._get_pricing_strategy(profile.market_segment)
        
        # Apply optimization algorithms
        optimal_price = self._calculate_optimal_price(
            wtp, avg_competitive_price, current_strategy, profile
        )
        
        return {
            'optimized_price': optimal_price,
            'willingness_to_pay': wtp,
            'competitive_average': avg_competitive_price,
            'price_positioning': self._calculate_price_positioning(
                optimal_price, avg_competitive_price
            ),
            'confidence_score': self._calculate_optimization_confidence(profile, competitive_analysis)
        }
        
    def forecast_revenue(self, forecast_periods: int = 12) -> Dict:
        """Forecast revenue based on current pipeline and strategies"""
        pipeline_data = self._get_pipeline_data()
        customer_metrics = self._get_customer_metrics()
        
        revenue_forecast = []
        total_pipeline = 0
        
        for month in range(forecast_periods):
            month_forecast = {
                'month': month + 1,
                'new_revenue': 0,
                'expansion_revenue': 0,
                'churn_risk': 0,
                'total': 0
            }
            
            # Calculate new customer revenue
            for customer in pipeline_data:
                if customer.get('expected_close_month') == month + 1:
                    customer_profile = self.customer_profiles.get(customer['customer_id'])
                    if customer_profile:
                        price = self._calculate_customer_price(customer_profile)
                        month_forecast['new_revenue'] += price
                        
            # Calculate expansion revenue
            for customer_id, profile in self.customer_profiles.items():
                expansion_revenue = self._calculate_expansion_revenue(profile, month)
                month_forecast['expansion_revenue'] += expansion_revenue
                
            # Calculate churn risk
            churn_risk = self._calculate_churn_risk(month)
            month_forecast['churn_risk'] = churn_risk
            
            month_forecast['total'] = (
                month_forecast['new_revenue'] + 
                month_forecast['expansion_revenue'] - 
                month_forecast['churn_risk']
            )
            
            revenue_forecast.append(month_forecast)
            total_pipeline += month_forecast['new_revenue']
            
        return {
            'monthly_forecast': revenue_forecast,
            'total_pipeline': total_pipeline,
            'forecast_date': datetime.now(),
            'methodology': 'pipeline_based_with_expansion_and_churn'
        }
        
    def calculate_roi(self, customer_id: str, implementation_cost: float,
                     clinical_outcomes: List[ClinicalOutcome]) -> ROI_Result:
        """Calculate ROI for customer implementation"""
        profile = self.customer_profiles[customer_id]
        
        # Calculate annual benefits from clinical outcomes
        total_annual_benefits = 0
        for outcome in clinical_outcomes:
            annual_benefit = (
                outcome.improvement_percentage / 100 * 
                outcome.baseline_value * 
                outcome.cost_per_unit *
                (12 / outcome.measurement_period_months)
            )
            total_annual_benefits += annual_benefit
            
        # Calculate total benefits over 3 years
        total_benefits = total_annual_benefits * 3
        
        # Calculate NPV (assuming 10% discount rate)
        discount_rate = 0.10
        npv = 0
        for year in range(1, 4):
            annual_benefit_year = total_annual_benefits
            discounted_benefit = annual_benefit_year / ((1 + discount_rate) ** year)
            npv += discounted_benefit
            
        npv -= implementation_cost
        
        # Calculate ROI percentage
        roi_percentage = ((total_benefits - implementation_cost) / implementation_cost) * 100
        
        # Calculate payback period
        payback_period = implementation_cost / total_annual_benefits * 12
        
        roi_result = ROI_Result(
            customer_id=customer_id,
            investment_amount=implementation_cost,
            total_savings=total_benefits * 0.7,  # Assuming 70% of benefits are savings
            total_benefits=total_benefits,
            roi_percentage=roi_percentage,
            payback_period_months=payback_period,
            net_present_value=npv,
            clinical_outcomes=clinical_outcomes,
            calculation_date=datetime.now()
        )
        
        self.roi_cache[customer_id] = roi_result
        return roi_result
        
    def track_revenue_attribution(self, attribution: RevenueAttribution) -> None:
        """Track revenue attribution for marketing ROI"""
        self.revenue_attribution.append(attribution)
        
    def calculate_marketing_roi(self, time_period_days: int = 90) -> Dict:
        """Calculate marketing ROI based on revenue attribution"""
        cutoff_date = datetime.now() - timedelta(days=time_period_days)
        recent_attributions = [
            attr for attr in self.revenue_attribution 
            if attr.touchpoint_date >= cutoff_date
        ]
        
        # Group by marketing channel
        channel_performance = {}
        for attr in recent_attributions:
            channel = attr.marketing_channel
            if channel not in channel_performance:
                channel_performance[channel] = {
                    'total_revenue': 0,
                    'total_attribution': 0,
                    'deal_count': 0,
                    'avg_deal_size': 0
                }
                
            channel_performance[channel]['total_revenue'] += attr.deal_value
            channel_performance[channel]['total_attribution'] += attr.attribution_weight
            channel_performance[channel]['deal_count'] += 1
            
        # Calculate metrics for each channel
        for channel, data in channel_performance.items():
            data['avg_deal_size'] = data['total_revenue'] / data['deal_count'] if data['deal_count'] > 0 else 0
            data['attribution_efficiency'] = data['total_revenue'] * data['total_attribution']
            
        return {
            'time_period_days': time_period_days,
            'total_revenue_tracked': sum(data['total_revenue'] for data in channel_performance.values()),
            'channel_performance': channel_performance,
            'top_channels': sorted(
                channel_performance.items(), 
                key=lambda x: x[1]['total_revenue'], 
                reverse=True
            )[:5],
            'calculation_date': datetime.now()
        }
        
    def _get_pricing_strategy(self, market_segment: MarketSegment) -> PricingStrategy:
        """Get pricing strategy for market segment"""
        strategy_id = f"strategy_{market_segment.value}"
        if strategy_id not in self.pricing_strategies:
            # Create default strategy if not exists
            default_strategy = PricingStrategy(
                strategy_id=strategy_id,
                market_segment=market_segment,
                pricing_model=PricingModel.SUBSCRIPTION,
                base_price=100000,
                price_factors={
                    'market_segment': 1.0,
                    'technology_adoption': 1.0,
                    'competitive_pressure': 1.0
                },
                value_metrics=['clinical_outcomes', 'efficiency_gains'],
                discount_tiers={
                    CustomerTier.PLATINUM: 1.0,
                    CustomerTier.GOLD: 0.9,
                    CustomerTier.SILVER: 0.8,
                    CustomerTier.BRONZE: 0.7
                },
                contractual_terms={'term_length': '3 years'},
                success_metrics=['implementation_success', 'user_adoption']
            )
            self.pricing_strategies[strategy_id] = default_strategy
            
        return self.pricing_strategies[strategy_id]
        
    def _calculate_volume_discount(self, patient_volume: int) -> float:
        """Calculate volume discount based on patient volume"""
        if patient_volume > 500000:
            return 0.8  # 20% discount
        elif patient_volume > 100000:
            return 0.85  # 15% discount
        elif patient_volume > 50000:
            return 0.9   # 10% discount
        elif patient_volume > 10000:
            return 0.95  # 5% discount
        else:
            return 1.0   # No discount
            
    def _calculate_tech_adoption_discount(self, tech_score: float) -> float:
        """Calculate discount based on technology adoption score"""
        if tech_score > 0.8:
            return 0.95  # 5% discount for high adoption
        elif tech_score > 0.6:
            return 1.0   # No discount
        else:
            return 1.1   # 10% premium for low adoption
            
    def _calculate_usage_components(self, profile: CustomerProfile) -> Dict:
        """Calculate usage-based pricing components"""
        base_usage = profile.patient_volume * 0.5  # $0.50 per patient
        
        # Complexity multiplier based on specialties
        complexity_multipliers = {
            'cardiology': 1.5,
            'oncology': 2.0,
            'radiology': 1.3,
            'pathology': 1.8,
            'emergency': 1.2,
            'default': 1.0
        }
        
        max_complexity = max([
            complexity_multipliers.get(spec, complexity_multipliers['default']) 
            for spec in profile.clinical_specialties
        ], default=1.0)
        
        adjusted_usage = base_usage * max_complexity
        
        return {
            'base_annual': adjusted_usage,
            'annual_extra': 0,  # Could add overage charges
            'complexity_multiplier': max_complexity
        }
        
    def _get_contract_terms(self, market_segment: MarketSegment, tier: CustomerTier) -> Dict:
        """Get contract terms based on segment and tier"""
        base_terms = {
            'term_length': '3 years',
            'payment_terms': 'Annual',
            'support_level': 'Standard'
        }
        
        if tier == CustomerTier.PLATINUM:
            base_terms.update({
                'term_length': '5 years',
                'payment_terms': 'Annual with 10% discount',
                'support_level': 'Premium 24/7',
                'custom_development': 'Included'
            })
        elif tier == CustomerTier.GOLD:
            base_terms.update({
                'support_level': 'Priority'
            })
            
        return base_terms
        
    def _calculate_willingness_to_pay(self, profile: CustomerProfile) -> float:
        """Calculate customer willingness to pay"""
        # Base WTP as percentage of annual revenue
        base_wtp_rate = 0.002  # 0.2% of annual revenue
        
        # Adjust for market segment
        segment_multipliers = {
            MarketSegment.HOSPITAL_SYSTEM: 1.5,
            MarketSegment.AMC: 1.2,
            MarketSegment.CLINIC: 0.8,
            MarketSegment.IDN: 1.8,
            MarketSegment.REGIONAL_HOSPITAL: 1.0,
            MarketSegment.SPECIALTY_CLINIC: 0.9,
            MarketSegment.RURAL_HOSPITAL: 0.6
        }
        
        segment_multiplier = segment_multipliers.get(profile.market_segment, 1.0)
        
        # Adjust for technology adoption
        tech_multiplier = 0.5 + (profile.technology_adoption_score * 0.5)  # 0.5 to 1.0
        
        # Calculate WTP
        wtp = profile.annual_revenue * base_wtp_rate * segment_multiplier * tech_multiplier
        
        return max(wtp, 50000)  # Minimum WTP of $50k
        
    def _calculate_optimal_price(self, wtp: float, competitive_price: float, 
                               strategy: PricingStrategy, profile: CustomerProfile) -> float:
        """Calculate optimal price using economic pricing models"""
        # Start with willingness to pay
        optimal_price = wtp * 0.8  # Capture 80% of WTP
        
        # Competitive positioning adjustment
        if competitive_price > 0:
            competitive_factor = 1.1 if optimal_price < competitive_price * 0.8 else 0.9
            optimal_price *= competitive_factor
            
        # Apply strategy constraints
        strategy_multiplier = strategy.price_factors.get('competitive_pressure', 1.0)
        optimal_price *= strategy_multiplier
        
        # Ensure price is within customer's budget range
        budget_max = profile.budget_range.get('max', optimal_price * 2)
        optimal_price = min(optimal_price, budget_max)
        
        return max(optimal_price, 10000)  # Minimum price of $10k
        
    def _calculate_price_positioning(self, our_price: float, competitive_price: float) -> str:
        """Calculate price positioning relative to competition"""
        if competitive_price == 0:
            return "no_competition"
            
        ratio = our_price / competitive_price
        
        if ratio < 0.7:
            return "low_price"
        elif ratio < 0.9:
            return "competitive"
        elif ratio < 1.1:
            return "market_rate"
        elif ratio < 1.3:
            return "premium"
        else:
            return "luxury"
            
    def _calculate_optimization_confidence(self, profile: CustomerProfile, 
                                         competitive_analysis: Dict) -> float:
        """Calculate confidence in pricing optimization"""
        confidence_factors = []
        
        # Factor 1: Data completeness
        data_completeness = (
            (1 if profile.annual_revenue > 0 else 0) +
            (1 if profile.patient_volume > 0 else 0) +
            (1 if len(profile.clinical_specialties) > 0 else 0) +
            (1 if profile.technology_adoption_score > 0 else 0)
        ) / 4
        confidence_factors.append(data_completeness)
        
        # Factor 2: Competitive analysis completeness
        comp_completeness = min(len(competitive_analysis) / 3, 1.0)
        confidence_factors.append(comp_completeness)
        
        # Factor 3: Market segment clarity
        segment_confidence = 1.0 if profile.market_segment else 0.5
        confidence_factors.append(segment_confidence)
        
        return np.mean(confidence_factors)
        
    def _get_pipeline_data(self) -> List[Dict]:
        """Get sales pipeline data"""
        # This would integrate with CRM system
        return [
            {
                'customer_id': 'sample_1',
                'expected_close_month': 1,
                'deal_value': 250000,
                'stage': 'proposal'
            },
            {
                'customer_id': 'sample_2', 
                'expected_close_month': 2,
                'deal_value': 180000,
                'stage': 'negotiation'
            }
        ]
        
    def _get_customer_metrics(self) -> Dict:
        """Get customer health and expansion metrics"""
        return {
            'expansion_rate': 0.25,
            'churn_rate': 0.05,
            'nrr': 1.25  # Net Revenue Retention
        }
        
    def _calculate_customer_price(self, profile: CustomerProfile) -> float:
        """Calculate price for a customer profile"""
        base_price = 150000  # Default base price
        return base_price
        
    def _calculate_expansion_revenue(self, profile: CustomerProfile, month: int) -> float:
        """Calculate expansion revenue for existing customers"""
        base_expansion_rate = 0.02  # 2% monthly expansion
        return profile.current_ai_spend * base_expansion_rate
        
    def _calculate_churn_risk(self, month: int) -> float:
        """Calculate potential churn risk"""
        base_churn_rate = 0.004  # 0.4% monthly churn
        return 100000 * base_churn_rate  # Assuming average customer value of $100k


if __name__ == "__main__":
    # Example usage
    framework = HealthcarePricingFramework()
    
    # Add sample customer profile
    customer = CustomerProfile(
        customer_id="customer_001",
        organization_name="General Hospital System",
        market_segment=MarketSegment.HOSPITAL_SYSTEM,
        tier=CustomerTier.GOLD,
        annual_revenue=500000000,
        patient_volume=200000,
        clinical_specialties=["cardiology", "oncology", "radiology"],
        technology_adoption_score=0.75,
        budget_range={"min": 300000, "max": 800000},
        decision_makers=["CMO", "CIO"],
        current_ai_spend=150000,
        competitive_solutions=["solution_a", "solution_b"]
    )
    
    framework.add_customer_profile(customer)
    
    # Create pricing strategy
    strategy = PricingStrategy(
        strategy_id="hospital_strategy",
        market_segment=MarketSegment.HOSPITAL_SYSTEM,
        pricing_model=PricingModel.SUBSCRIPTION,
        base_price=500000,
        price_factors={"market_segment": 1.2, "technology_adoption": 1.0},
        value_metrics=["reduced_readmissions", "improved_diagnosis_accuracy"],
        discount_tiers={
            CustomerTier.PLATINUM: 1.0,
            CustomerTier.GOLD: 0.9,
            CustomerTier.SILVER: 0.8,
            CustomerTier.BRONZE: 0.7
        },
        contractual_terms={"term_length": "3 years"},
        success_metrics=["implementation_success", "user_adoption"]
    )
    
    framework.create_pricing_strategy(strategy)
    
    # Calculate value-based pricing
    outcomes = [
        ClinicalOutcome(
            metric_name="readmission_reduction",
            improvement_percentage=15.0,
            baseline_value=5000000,
            current_value=4250000,
            measurement_period_months=12,
            cost_per_unit=0.2
        )
    ]
    
    price, breakdown = framework.calculate_value_based_price(customer.customer_id, outcomes)
    print(f"Value-based price: ${price:,.2f}")
    print(f"Value breakdown: {breakdown}")
    
    # Create subscription model
    subscription = framework.create_subscription_model(customer.customer_id, CustomerTier.GOLD)
    print(f"Subscription pricing: {subscription}")
    
    # Calculate ROI
    roi = framework.calculate_roi(customer.customer_id, 300000, outcomes)
    print(f"ROI: {roi.roi_percentage:.1f}%")
    print(f"Payback period: {roi.payback_period_months:.1f} months")