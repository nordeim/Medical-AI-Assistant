"""
Market Segment Analysis and Pricing Optimization

This module provides detailed analysis and optimization capabilities for different healthcare market segments,
including hospital systems, academic medical centers, clinics, and integrated delivery networks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from pricing_framework import MarketSegment, CustomerTier, CustomerProfile

@dataclass
class MarketMetrics:
    """Market segment metrics and benchmarks"""
    segment: MarketSegment
    total_addressable_market: float
    average_deal_size: float
    sales_cycle_days: int
    win_rate: float
    avg_annual_revenue: float
    avg_patient_volume: int
    typical_buying_committee_size: int
    decision_process_complexity: float  # 0-1 scale
    price_sensitivity_score: float      # 0-1 scale

@dataclass
class CompetitiveLandscape:
    """Competitive analysis for market segments"""
    segment: MarketSegment
    key_competitors: List[str]
    avg_market_price: float
    price_range_min: float
    price_range_max: float
    market_share_estimates: Dict[str, float]
    differentiation_factors: List[str]
    competitive_intensity: float  # 0-1 scale

@dataclass
class SegmentStrategy:
    """Strategy recommendations for market segments"""
    segment: MarketSegment
    recommended_pricing_model: str
    key_value_propositions: List[str]
    sales_messaging: List[str]
    success_factors: List[str]
    risk_factors: List[str]
    implementation_priorities: List[str]

class MarketSegmentAnalyzer:
    """Market segment analysis and optimization"""
    
    def __init__(self):
        self.market_metrics = self._initialize_market_metrics()
        self.competitive_landscapes = self._initialize_competitive_landscapes()
        self.segment_strategies = self._initialize_segment_strategies()
        
    def analyze_segment_opportunity(self, segment: MarketSegment) -> Dict:
        """Analyze opportunity for a specific market segment"""
        metrics = self.market_metrics[segment]
        competition = self.competitive_landscapes[segment]
        strategy = self.segment_strategies[segment]
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(metrics, competition)
        
        # Calculate pricing optimization potential
        pricing_potential = self._calculate_pricing_potential(metrics, competition)
        
        # Estimate market penetration timeline
        penetration_timeline = self._estimate_penetration_timeline(metrics)
        
        return {
            'segment': segment.value,
            'opportunity_score': opportunity_score,
            'total_addressable_market': metrics.total_addressable_market,
            'average_deal_size': metrics.average_deal_size,
            'sales_cycle': metrics.sales_cycle_days,
            'win_rate': metrics.win_rate,
            'competitive_intensity': competition.competitive_intensity,
            'pricing_potential': pricing_potential,
            'penetration_timeline': penetration_timeline,
            'recommended_approach': {
                'pricing_model': strategy.recommended_pricing_model,
                'value_props': strategy.key_value_propositions,
                'sales_messaging': strategy.sales_messaging,
                'success_factors': strategy.success_factors
            }
        }
        
    def optimize_pricing_for_segment(self, segment: MarketSegment, 
                                   customer_profile: CustomerProfile) -> Dict:
        """Optimize pricing for specific segment and customer"""
        metrics = self.market_metrics[segment]
        competition = self.competitive_landscapes[segment]
        
        # Base pricing using market benchmarks
        base_price = self._calculate_base_price(customer_profile, metrics)
        
        # Competitive positioning
        competitive_price = competition.avg_market_price
        positioning = self._determine_positioning(customer_profile, competitive_price, metrics)
        
        # Value-based adjustments
        value_adjustments = self._calculate_value_adjustments(customer_profile, metrics)
        
        # Risk-adjusted pricing
        final_price = self._apply_risk_adjustments(
            base_price, value_adjustments, metrics, customer_profile
        )
        
        return {
            'segment_optimized_price': final_price,
            'base_price': base_price,
            'competitive_benchmark': competitive_price,
            'positioning': positioning,
            'value_adjustments': value_adjustments,
            'confidence_interval': self._calculate_confidence_interval(final_price, metrics),
            'alternative_scenarios': {
                'conservative': final_price * 0.8,
                'aggressive': final_price * 1.2,
                'competitive': competitive_price
            }
        }
        
    def forecast_segment_performance(self, segment: MarketSegment, 
                                   months_ahead: int = 12) -> Dict:
        """Forecast performance for market segment"""
        metrics = self.market_metrics[segment]
        
        # Calculate baseline metrics
        monthly_leads = self._estimate_monthly_leads(segment)
        conversion_rate = metrics.win_rate
        avg_deal_size = metrics.average_deal_size
        
        # Generate monthly forecast
        forecast = []
        for month in range(1, months_ahead + 1):
            # Account for seasonality and growth trends
            seasonal_factor = self._calculate_seasonal_factor(month)
            growth_factor = 1 + (month * 0.02)  # 2% monthly growth assumption
            
            month_leads = monthly_leads * seasonal_factor * growth_factor
            month_opportunities = month_leads * conversion_rate
            month_revenue = month_opportunities * avg_deal_size
            
            forecast.append({
                'month': month,
                'leads': int(month_leads),
                'opportunities': int(month_opportunities),
                'deals': int(month_opportunities * 0.8),  # 80% of opportunities close
                'revenue': month_revenue,
                'conversion_rate': conversion_rate
            })
            
        # Calculate key metrics
        total_revenue = sum(month['revenue'] for month in forecast)
        total_opportunities = sum(month['opportunities'] for month in forecast)
        weighted_avg_deal_size = total_revenue / total_opportunities if total_opportunities > 0 else 0
        
        return {
            'segment': segment.value,
            'forecast_period': months_ahead,
            'monthly_forecast': forecast,
            'total_revenue': total_revenue,
            'total_opportunities': total_opportunities,
            'weighted_avg_deal_size': weighted_avg_deal_size,
            'forecast_confidence': self._calculate_forecast_confidence(segment, metrics)
        }
        
    def compare_segment_strategies(self, segments: List[MarketSegment]) -> Dict:
        """Compare strategies across multiple segments"""
        comparisons = []
        
        for segment in segments:
            analysis = self.analyze_segment_opportunity(segment)
            comparisons.append(analysis)
            
        # Sort by opportunity score
        comparisons.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Calculate portfolio insights
        total_market = sum(comp['total_addressable_market'] for comp in comparisons)
        weighted_avg_deal_size = sum(
            comp['average_deal_size'] * comp['total_addressable_market'] 
            for comp in comparisons
        ) / total_market if total_market > 0 else 0
        
        return {
            'segment_rankings': comparisons,
            'portfolio_metrics': {
                'total_tam': total_market,
                'weighted_avg_deal_size': weighted_avg_deal_size,
                'recommended_focus_segments': [comp['segment'] for comp in comparisons[:3]]
            },
            'resource_allocation': self._calculate_resource_allocation(comparisons)
        }
        
    def _initialize_market_metrics(self) -> Dict[MarketSegment, MarketMetrics]:
        """Initialize market metrics for all segments"""
        return {
            MarketSegment.HOSPITAL_SYSTEM: MarketMetrics(
                segment=MarketSegment.HOSPITAL_SYSTEM,
                total_addressable_market=2500000000,  # $2.5B
                average_deal_size=450000,
                sales_cycle_days=180,
                win_rate=0.25,
                avg_annual_revenue=800000000,
                avg_patient_volume=300000,
                typical_buying_committee_size=8,
                decision_process_complexity=0.8,
                price_sensitivity_score=0.6
            ),
            MarketSegment.AMC: MarketMetrics(
                segment=MarketSegment.AMC,
                total_addressable_market=1800000000,  # $1.8B
                average_deal_size=320000,
                sales_cycle_days=240,
                win_rate=0.30,
                avg_annual_revenue=600000000,
                avg_patient_volume=250000,
                typical_buying_committee_size=6,
                decision_process_complexity=0.7,
                price_sensitivity_score=0.5
            ),
            MarketSegment.CLINIC: MarketMetrics(
                segment=MarketSegment.CLINIC,
                total_addressable_market=1200000000,  # $1.2B
                average_deal_size=85000,
                sales_cycle_days=90,
                win_rate=0.35,
                avg_annual_revenue=50000000,
                avg_patient_volume=25000,
                typical_buying_committee_size=3,
                decision_process_complexity=0.4,
                price_sensitivity_score=0.8
            ),
            MarketSegment.IDN: MarketMetrics(
                segment=MarketSegment.IDN,
                total_addressable_market=3200000000,  # $3.2B
                average_deal_size=650000,
                sales_cycle_days=210,
                win_rate=0.22,
                avg_annual_revenue=1200000000,
                avg_patient_volume=450000,
                typical_buying_committee_size=10,
                decision_process_complexity=0.9,
                price_sensitivity_score=0.5
            ),
            MarketSegment.REGIONAL_HOSPITAL: MarketMetrics(
                segment=MarketSegment.REGIONAL_HOSPITAL,
                total_addressable_market=800000000,   # $800M
                average_deal_size=180000,
                sales_cycle_days=120,
                win_rate=0.28,
                avg_annual_revenue=150000000,
                avg_patient_volume=75000,
                typical_buying_committee_size=5,
                decision_process_complexity=0.6,
                price_sensitivity_score=0.7
            ),
            MarketSegment.SPECIALTY_CLINIC: MarketMetrics(
                segment=MarketSegment.SPECIALTY_CLINIC,
                total_addressable_market=600000000,   # $600M
                average_deal_size=120000,
                sales_cycle_days=100,
                win_rate=0.32,
                avg_annual_revenue=75000000,
                avg_patient_volume=40000,
                typical_buying_committee_size=4,
                decision_process_complexity=0.5,
                price_sensitivity_score=0.7
            ),
            MarketSegment.RURAL_HOSPITAL: MarketMetrics(
                segment=MarketSegment.RURAL_HOSPITAL,
                total_addressable_market=300000000,   # $300M
                average_deal_size=65000,
                sales_cycle_days=80,
                win_rate=0.40,
                avg_annual_revenue=40000000,
                avg_patient_volume=20000,
                typical_buying_committee_size=2,
                decision_process_complexity=0.3,
                price_sensitivity_score=0.9
            )
        }
        
    def _initialize_competitive_landscapes(self) -> Dict[MarketSegment, CompetitiveLandscape]:
        """Initialize competitive landscape data"""
        return {
            MarketSegment.HOSPITAL_SYSTEM: CompetitiveLandscape(
                segment=MarketSegment.HOSPITAL_SYSTEM,
                key_competitors=["Epic", "Cerner", "Allscripts", "athenahealth"],
                avg_market_price=500000,
                price_range_min=300000,
                price_range_max=800000,
                market_share_estimates={"Epic": 0.35, "Cerner": 0.28, "Others": 0.37},
                differentiation_factors=["integration", "clinical_workflow", "scalability"],
                competitive_intensity=0.8
            ),
            MarketSegment.AMC: CompetitiveLandscape(
                segment=MarketSegment.AMC,
                key_competitors=["Epic", "Cerner", " MEDITECH", "InterSystems"],
                avg_market_price=380000,
                price_range_min=250000,
                price_range_max=600000,
                market_share_estimates={"Epic": 0.40, "Cerner": 0.25, "Others": 0.35},
                differentiation_factors=["research_integration", "education_support", "innovation"],
                competitive_intensity=0.7
            ),
            MarketSegment.CLINIC: CompetitiveLandscape(
                segment=MarketSegment.CLINIC,
                key_competitors=["athenahealth", "eClinicalWorks", "NextGen", "DrChrono"],
                avg_market_price=95000,
                price_range_min=50000,
                price_range_max=150000,
                market_share_estimates={"athenahealth": 0.25, "eClinicalWorks": 0.20, "Others": 0.55},
                differentiation_factors=["ease_of_use", "cost_effectiveness", "implementation_speed"],
                competitive_intensity=0.6
            ),
            MarketSegment.IDN: CompetitiveLandscape(
                segment=MarketSegment.IDN,
                key_competitors=["Epic", "Cerner", "Allscripts", "MEDITECH"],
                avg_market_price=750000,
                price_range_min=500000,
                price_range_max=1200000,
                market_share_estimates={"Epic": 0.38, "Cerner": 0.30, "Others": 0.32},
                differentiation_factors=["enterprise_scale", "network_efficiency", "data_analytics"],
                competitive_intensity=0.9
            ),
            MarketSegment.REGIONAL_HOSPITAL: CompetitiveLandscape(
                segment=MarketSegment.REGIONAL_HOSPITAL,
                key_competitors=["MEDITECH", "Allscripts", "athenahealth", "Epic"],
                avg_market_price=200000,
                price_range_min=120000,
                price_range_max=350000,
                market_share_estimates={"MEDITECH": 0.30, "Allscripts": 0.25, "Others": 0.45},
                differentiation_factors=["regional_focus", "cost_optimization", "support"],
                competitive_intensity=0.6
            ),
            MarketSegment.SPECIALTY_CLINIC: CompetitiveLandscape(
                segment=MarketSegment.SPECIALTY_CLINIC,
                key_competitors=["NextGen", "DrChrono", "athenahealth", "eClinicalWorks"],
                avg_market_price=135000,
                price_range_min=80000,
                price_range_max=200000,
                market_share_estimates={"NextGen": 0.28, "DrChrono": 0.22, "Others": 0.50},
                differentiation_factors=["specialty_workflows", "efficiency", "ROI"],
                competitive_intensity=0.5
            ),
            MarketSegment.RURAL_HOSPITAL: CompetitiveLandscape(
                segment=MarketSegment.RURAL_HOSPITAL,
                key_competitors=["MEDITECH", "athenahealth", "eClinicalWorks"],
                avg_market_price=75000,
                price_range_min=40000,
                price_range_max=120000,
                market_share_estimates={"MEDITECH": 0.35, "athenahealth": 0.25, "Others": 0.40},
                differentiation_factors=["affordability", "simplicity", "support"],
                competitive_intensity=0.4
            )
        }
        
    def _initialize_segment_strategies(self) -> Dict[MarketSegment, SegmentStrategy]:
        """Initialize strategy recommendations for segments"""
        return {
            MarketSegment.HOSPITAL_SYSTEM: SegmentStrategy(
                segment=MarketSegment.HOSPITAL_SYSTEM,
                recommended_pricing_model="enterprise_subscription",
                key_value_propositions=[
                    "Improve patient outcomes through AI-powered clinical decisions",
                    "Reduce operational costs via workflow optimization",
                    "Enhance quality metrics and reporting capabilities",
                    "Seamless integration with existing EHR systems"
                ],
                sales_messaging=[
                    "Proven ROI in similar hospital systems",
                    "Clinical outcome improvements documented in peer-reviewed studies",
                    "Comprehensive implementation and support services"
                ],
                success_factors=[
                    "Strong clinical champion identification",
                    "Demonstrated EHR integration capabilities", 
                    "Comprehensive change management support",
                    "Clear ROI measurement framework"
                ],
                risk_factors=[
                    "Complex procurement processes",
                    "Multiple stakeholder alignment required",
                    "Long implementation timelines",
                    "High competition from established vendors"
                ],
                implementation_priorities=[
                    "Develop reference customers in segment",
                    "Create comprehensive integration documentation",
                    "Build strong partnerships with EHR vendors",
                    "Establish clinical advisory board"
                ]
            ),
            MarketSegment.AMC: SegmentStrategy(
                segment=MarketSegment.AMC,
                recommended_pricing_model="research_partnership",
                key_value_propositions=[
                    "Accelerate research through AI-powered insights",
                    "Enhance educational programs with cutting-edge technology",
                    "Improve clinical training with AI-assisted diagnostics",
                    "Generate publishable research outcomes"
                ],
                sales_messaging=[
                    "Partnership approach to innovation",
                    "Research collaboration opportunities",
                    "Educational institution pricing",
                    "Co-development potential"
                ],
                success_factors=[
                    "Research collaboration agreements",
                    "Faculty engagement and buy-in",
                    "Student/resident training integration",
                    "Publication and conference support"
                ],
                risk_factors=[
                    "Budget constraints in academic settings",
                    "Long decision cycles",
                    "Competing research priorities",
                    "IT infrastructure limitations"
                ],
                implementation_priorities=[
                    "Establish research partnerships",
                    "Create academic pricing programs",
                    "Develop educational content",
                    "Build faculty relationships"
                ]
            ),
            MarketSegment.CLINIC: SegmentStrategy(
                segment=MarketSegment.CLINIC,
                recommended_pricing_model="cloud_subscription",
                key_value_propositions=[
                    "Improve patient care quality and efficiency",
                    "Reduce administrative burden",
                    "Faster diagnosis and treatment decisions",
                    "Competitive advantage through technology"
                ],
                sales_messaging=[
                    "Quick implementation and ROI",
                    "Affordable monthly pricing",
                    "Minimal IT infrastructure requirements",
                    "Dedicated customer success support"
                ],
                success_factors=[
                    "Streamlined sales process",
                    "Fast implementation (< 30 days)",
                    "Clear ROI demonstration",
                    "Strong customer references"
                ],
                risk_factors=[
                    "Price sensitivity",
                    "Limited IT resources",
                    "Owner/physician decision making",
                    "Competition from low-cost alternatives"
                ],
                implementation_priorities=[
                    "Develop self-service onboarding",
                    "Create ROI calculator tools",
                    "Establish referral programs",
                    "Build cloud-first architecture"
                ]
            ),
            MarketSegment.IDN: SegmentStrategy(
                segment=MarketSegment.IDN,
                recommended_pricing_model="enterprise_license",
                key_value_propositions=[
                    "Network-wide standardization and efficiency",
                    "Population health management capabilities",
                    "Cost savings across multiple facilities",
                    "Unified data analytics and insights"
                ],
                sales_messaging=[
                    "Enterprise-scale solution",
                    "Cross-facility standardization",
                    "Comprehensive analytics platform",
                    "Strategic partnership approach"
                ],
                success_factors=[
                    "Executive sponsorship at network level",
                    "Cross-facility champion network",
                    "Unified implementation approach",
                    "Network-wide ROI demonstration"
                ],
                risk_factors=[
                    "Complex multi-facility negotiations",
                    "Varying technical capabilities across facilities",
                    "Political dynamics within network",
                    "Extended sales and implementation cycles"
                ],
                implementation_priorities=[
                    "Develop enterprise negotiation framework",
                    "Create multi-facility implementation playbook",
                    "Build network-level relationships",
                    "Establish governance structures"
                ]
            )
        }
        
    def _calculate_opportunity_score(self, metrics: MarketMetrics, 
                                   competition: CompetitiveLandscape) -> float:
        """Calculate opportunity score for market segment"""
        # Factors: TAM, deal size, win rate, competitive intensity (inverted)
        tam_factor = np.log10(metrics.total_addressable_market / 1000000) / 3  # Normalize to 0-2
        deal_size_factor = metrics.average_deal_size / 500000  # Normalize to 0-2
        win_rate_factor = metrics.win_rate * 5  # Normalize to 0-2
        competition_factor = (1 - competition.competitive_intensity) * 2  # Invert and normalize
        
        # Weighted average
        score = (tam_factor * 0.3 + deal_size_factor * 0.25 + 
                win_rate_factor * 0.25 + competition_factor * 0.2)
        
        return min(score, 10.0)  # Cap at 10.0
        
    def _calculate_pricing_potential(self, metrics: MarketMetrics, 
                                   competition: CompetitiveLandscape) -> Dict:
        """Calculate pricing optimization potential"""
        # Base pricing power based on market characteristics
        base_power = (1 - metrics.price_sensitivity_score) * 0.6
        complexity_power = metrics.decision_process_complexity * 0.4
        total_power = base_power + complexity_power
        
        # Competitive positioning
        competitive_pressure = competition.competitive_intensity
        
        return {
            'pricing_power_score': total_power,
            'competitive_pressure': competitive_pressure,
            'premium_potential': total_power * (1 - competitive_pressure) * 100,
            'recommendation': 'premium' if total_power > 0.6 else 'competitive'
        }
        
    def _estimate_penetration_timeline(self, metrics: MarketMetrics) -> Dict:
        """Estimate market penetration timeline"""
        # Base timeline factors
        cycle_factor = metrics.sales_cycle_days / 180  # Normalize to typical cycle
        complexity_factor = metrics.decision_process_complexity
        
        # Estimate phases
        market_entry_months = int(6 * cycle_factor * complexity_factor)
        significant_penetration_months = int(24 * cycle_factor * complexity_factor)
        mature_penetration_months = int(60 * cycle_factor * complexity_factor)
        
        return {
            'market_entry': market_entry_months,
            'significant_penetration': significant_penetration_months,
            'mature_penetration': mature_penetration_months,
            'assumptions': f"Based on {metrics.sales_cycle_days} day sales cycle and {metrics.decision_process_complexity:.1f} complexity"
        }
        
    def _calculate_base_price(self, profile: CustomerProfile, 
                            metrics: MarketMetrics) -> float:
        """Calculate base price for customer in segment"""
        # Start with segment average
        base_price = metrics.average_deal_size
        
        # Adjust for customer size (revenue and volume)
        revenue_multiplier = min(profile.annual_revenue / metrics.avg_annual_revenue, 2.0)
        volume_multiplier = min(profile.patient_volume / metrics.avg_patient_volume, 2.0)
        
        # Adjust for technology adoption
        tech_multiplier = 0.8 + (profile.technology_adoption_score * 0.4)  # 0.8 to 1.2
        
        adjusted_price = base_price * (revenue_multiplier + volume_multiplier) / 2 * tech_multiplier
        
        return max(adjusted_price, metrics.average_deal_size * 0.5)  # Minimum 50% of segment avg
        
    def _determine_positioning(self, profile: CustomerProfile, competitive_price: float,
                             metrics: MarketMetrics) -> Dict:
        """Determine competitive positioning strategy"""
        our_price = self._calculate_base_price(profile, metrics)
        
        positioning_ratio = our_price / competitive_price
        
        if positioning_ratio < 0.8:
            strategy = "value_leader"
            messaging = "Superior value at competitive price"
        elif positioning_ratio < 1.1:
            strategy = "competitive_parity"
            messaging = "Comparable features with better support"
        elif positioning_ratio < 1.3:
            strategy = "premium"
            messaging = "Premium solution with proven ROI"
        else:
            strategy = "luxury"
            messaging = "Best-in-class solution for leading organizations"
            
        return {
            'strategy': strategy,
            'positioning_ratio': positioning_ratio,
            'messaging': messaging,
            'price_difference': f"{((positioning_ratio - 1) * 100):.1f}% vs competition"
        }
        
    def _calculate_value_adjustments(self, profile: CustomerProfile, 
                                   metrics: MarketMetrics) -> Dict:
        """Calculate value-based pricing adjustments"""
        adjustments = {}
        
        # Clinical specialty premium
        specialty_premiums = {
            'cardiology': 1.2,
            'oncology': 1.5,
            'radiology': 1.3,
            'emergency': 1.1,
            'default': 1.0
        }
        
        max_specialty = max([
            specialty_premiums.get(spec, specialty_premiums['default'])
            for spec in profile.clinical_specialties
        ], default=1.0)
        
        adjustments['clinical_specialty'] = max_specialty
        
        # Technology adoption adjustment
        if profile.technology_adoption_score > 0.8:
            adjustments['tech_adoption'] = 1.1  # Premium for high adoption
        elif profile.technology_adoption_score < 0.4:
            adjustments['tech_adoption'] = 0.9  # Discount for low adoption
        else:
            adjustments['tech_adoption'] = 1.0
            
        # Budget alignment
        budget_mid = (profile.budget_range['min'] + profile.budget_range['max']) / 2
        if budget_mid > metrics.average_deal_size * 1.5:
            adjustments['budget_alignment'] = 1.2
        elif budget_mid < metrics.average_deal_size * 0.7:
            adjustments['budget_alignment'] = 0.8
        else:
            adjustments['budget_alignment'] = 1.0
            
        return adjustments
        
    def _apply_risk_adjustments(self, base_price: float, value_adjustments: Dict,
                              metrics: MarketMetrics, profile: CustomerProfile) -> float:
        """Apply risk-based pricing adjustments"""
        adjusted_price = base_price
        
        # Apply value adjustments
        for adjustment_type, multiplier in value_adjustments.items():
            adjusted_price *= multiplier
            
        # Risk factors
        if metrics.price_sensitivity_score > 0.8:
            adjusted_price *= 0.9  # Price-sensitive market
        if metrics.decision_process_complexity > 0.7:
            adjusted_price *= 0.95  # Complex decision process
            
        # Customer-specific risks
        if profile.technology_adoption_score < 0.3:
            adjusted_price *= 0.85  # High implementation risk
            
        return max(adjusted_price, 10000)  # Minimum price floor
        
    def _calculate_confidence_interval(self, price: float, metrics: MarketMetrics) -> Tuple[float, float]:
        """Calculate confidence interval for pricing"""
        # Base confidence on market data quality
        confidence_factor = 0.8  # Base confidence
        
        # Adjust for win rate (higher win rate = more confidence)
        confidence_factor += (metrics.win_rate - 0.25) * 0.4
        
        # Adjust for sales cycle (longer cycles = less confidence in current pricing)
        cycle_factor = max(0, 1 - (metrics.sales_cycle_days - 180) / 365)
        confidence_factor *= cycle_factor
        
        # Calculate confidence interval (±20% at full confidence, ±40% at low confidence)
        interval_size = 0.4 - (confidence_factor * 0.2)
        
        lower_bound = price * (1 - interval_size)
        upper_bound = price * (1 + interval_size)
        
        return (lower_bound, upper_bound)
        
    def _calculate_seasonal_factor(self, month: int) -> float:
        """Calculate seasonal adjustment factor"""
        # Healthcare tends to be slower in summer months
        seasonal_factors = {
            1: 1.1,   # January - budget flush
            2: 1.0,   # February
            3: 1.0,   # March
            4: 1.0,   # April
            5: 0.9,   # May
            6: 0.8,   # June - summer slowdown begins
            7: 0.7,   # July - summer slowdown
            8: 0.7,   # August - summer slowdown
            9: 0.9,   # September - back to school/business
            10: 1.0,  # October
            11: 1.1,  # November - year-end push
            12: 1.2   # December - year-end budget utilization
        }
        
        return seasonal_factors.get(month, 1.0)
        
    def _estimate_monthly_leads(self, segment: MarketSegment) -> float:
        """Estimate monthly leads for segment"""
        # This would be based on actual marketing and sales data
        metrics = self.market_metrics[segment]
        
        # Estimate based on TAM and market penetration assumptions
        annual_penetration_rate = 0.02  # 2% annual penetration assumption
        estimated_customers = metrics.total_addressable_market / metrics.average_deal_size
        
        monthly_new_leads = (estimated_customers * annual_penetration_rate) / 12
        
        return max(monthly_new_leads, 1.0)  # Minimum 1 lead per month
        
    def _calculate_forecast_confidence(self, segment: MarketSegment, 
                                     metrics: MarketMetrics) -> float:
        """Calculate confidence in forecast"""
        # Base confidence factors
        win_rate_confidence = metrics.win_rate * 2  # Normalize to 0-2
        cycle_confidence = max(0, 2 - (metrics.sales_cycle_days / 180))  # Shorter cycles = more confidence
        
        # Market maturity factor (newer markets have less predictable forecasts)
        maturity_factor = 0.8 if segment in [MarketSegment.HOSPITAL_SYSTEM, MarketSegment.CLINIC] else 0.6
        
        confidence = (win_rate_confidence + cycle_confidence) / 4 * maturity_factor
        
        return min(confidence, 0.9)  # Cap at 90% confidence
        
    def _calculate_resource_allocation(self, comparisons: List[Dict]) -> Dict:
        """Calculate recommended resource allocation across segments"""
        total_opportunity = sum(comp['opportunity_score'] for comp in comparisons)
        
        allocations = {}
        for comp in comparisons:
            allocation_percentage = (comp['opportunity_score'] / total_opportunity) * 100
            allocations[comp['segment']] = {
                'percentage': allocation_percentage,
                'priority': 'high' if allocation_percentage > 25 else 'medium' if allocation_percentage > 15 else 'low'
            }
            
        return allocations


if __name__ == "__main__":
    # Example usage
    analyzer = MarketSegmentAnalyzer()
    
    # Analyze hospital system segment
    hospital_analysis = analyzer.analyze_segment_opportunity(MarketSegment.HOSPITAL_SYSTEM)
    print("Hospital System Analysis:")
    print(json.dumps(hospital_analysis, indent=2, default=str))
    
    # Forecast performance
    forecast = analyzer.forecast_segment_performance(MarketSegment.HOSPITAL_SYSTEM, 6)
    print("\nHospital System Forecast:")
    print(json.dumps(forecast, indent=2, default=str))
    
    # Compare strategies
    segments = [MarketSegment.HOSPITAL_SYSTEM, MarketSegment.CLINIC, MarketSegment.AMC]
    comparison = analyzer.compare_segment_strategies(segments)
    print("\nSegment Comparison:")
    print(json.dumps(comparison, indent=2, default=str))