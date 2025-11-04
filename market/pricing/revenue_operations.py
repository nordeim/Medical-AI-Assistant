"""
Revenue Operations and Forecasting System

This module implements comprehensive revenue operations including forecasting, pipeline management,
revenue attribution, and performance analytics for healthcare AI pricing optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from collections import defaultdict

from pricing_framework import MarketSegment, CustomerTier, CustomerProfile

class DealStage(Enum):
    """Sales pipeline stages"""
    LEAD = "lead"
    QUALIFIED = "qualified"
    PROPOSAL = "proposal"
    NEGOTIATION = "negotiation"
    CLOSED_WON = "closed_won"
    CLOSED_LOST = "closed_lost"

class RevenueType(Enum):
    """Types of revenue"""
    NEW_BUSINESS = "new_business"
    EXPANSION = "expansion"
    RENEWAL = "renewal"
    CHURN = "churn"
    CONTRIBUTION = "contribution"

@dataclass
class Deal:
    """Individual deal in pipeline"""
    deal_id: str
    customer_id: str
    customer_name: str
    market_segment: MarketSegment
    stage: DealStage
    value: float
    probability: float
    expected_close_date: datetime
    created_date: datetime
    last_activity_date: datetime
    sales_rep: str
    product_category: str
    competitive_situation: str
    last_updated: datetime

@dataclass
class Customer:
    """Customer account record"""
    customer_id: str
    account_name: str
    market_segment: MarketSegment
    tier: CustomerTier
    annual_contract_value: float
    start_date: datetime
    renewal_date: datetime
    health_score: float  # 0-1 scale
    expansion_potential: float  # 0-1 scale
    churn_risk: float  # 0-1 scale
    product_usage: Dict[str, float]
    support_tickets: int
    satisfaction_score: float

@dataclass
class MarketingAttribution:
    """Marketing attribution data"""
    attribution_id: str
    customer_id: str
    touchpoint_date: datetime
    marketing_channel: str
    campaign_id: Optional[str]
    content_type: str
    engagement_score: float  # 0-1 scale
    attribution_weight: float  # 0-1 for multi-touch attribution

@dataclass
class ForecastScenario:
    """Revenue forecast scenario"""
    scenario_name: str
    assumptions: Dict[str, float]
    monthly_forecast: List[Dict]
    confidence_level: float
    risk_factors: List[str]
    opportunity_factors: List[str]

class RevenueOperations:
    """Revenue operations management system"""
    
    def __init__(self):
        self.deals: List[Deal] = []
        self.customers: List[Customer] = []
        self.attribution_data: List[MarketingAttribution] = []
        self.forecast_scenarios: Dict[str, ForecastScenario] = {}
        
        # Initialize with sample data
        self._initialize_sample_data()
        
    def add_deal(self, deal: Deal) -> None:
        """Add deal to pipeline"""
        self.deals.append(deal)
        
    def add_customer(self, customer: Customer) -> None:
        """Add customer to customer base"""
        self.customers.append(customer)
        
    def add_attribution(self, attribution: MarketingAttribution) -> None:
        """Add marketing attribution data"""
        self.attribution_data.append(attribution)
        
    def forecast_revenue(self, forecast_months: int = 12, 
                        scenario: str = "base_case") -> Dict:
        """Generate comprehensive revenue forecast"""
        if scenario not in self.forecast_scenarios:
            self._create_forecast_scenario(scenario)
            
        base_forecast = self._calculate_base_forecast(forecast_months)
        scenario_data = self.forecast_scenarios[scenario]
        
        # Apply scenario adjustments
        adjusted_forecast = self._apply_scenario_adjustments(
            base_forecast, scenario_data, forecast_months
        )
        
        # Calculate key metrics
        total_revenue = sum(month['total_revenue'] for month in adjusted_forecast)
        new_business_revenue = sum(month['new_business'] for month in adjusted_forecast)
        expansion_revenue = sum(month['expansion'] for month in adjusted_forecast)
        
        # Calculate confidence intervals
        confidence_analysis = self._calculate_confidence_analysis(adjusted_forecast)
        
        return {
            'scenario': scenario,
            'forecast_period': forecast_months,
            'monthly_forecast': adjusted_forecast,
            'summary_metrics': {
                'total_revenue': total_revenue,
                'new_business_revenue': new_business_revenue,
                'expansion_revenue': expansion_revenue,
                'average_monthly_revenue': total_revenue / forecast_months,
                'growth_rate': self._calculate_growth_rate(adjusted_forecast)
            },
            'confidence_analysis': confidence_analysis,
            'scenario_assumptions': scenario_data.assumptions,
            'forecast_date': datetime.now().isoformat()
        }
        
    def analyze_pipeline_health(self) -> Dict:
        """Analyze pipeline health and conversion metrics"""
        # Filter active deals (not closed)
        active_deals = [deal for deal in self.deals if deal.stage != DealStage.CLOSED_LOST]
        
        # Group by stage
        pipeline_by_stage = defaultdict(list)
        for deal in active_deals:
            pipeline_by_stage[deal.stage.value].append(deal)
            
        # Calculate stage metrics
        stage_analysis = {}
        for stage, deals in pipeline_by_stage.items():
            total_value = sum(deal.value for deal in deals)
            weighted_value = sum(deal.value * deal.probability for deal in deals)
            deal_count = len(deals)
            avg_deal_size = total_value / deal_count if deal_count > 0 else 0
            
            stage_analysis[stage] = {
                'deal_count': deal_count,
                'total_value': total_value,
                'weighted_value': weighted_value,
                'average_deal_size': avg_deal_size,
                'conversion_rate_to_next': self._estimate_stage_conversion(stage)
            }
            
        # Calculate overall pipeline metrics
        total_pipeline = sum(deal.value for deal in active_deals)
        weighted_pipeline = sum(deal.value * deal.probability for deal in active_deals)
        overall_conversion = self._calculate_overall_conversion_rate()
        
        # Identify pipeline risks
        pipeline_risks = self._identify_pipeline_risks()
        
        return {
            'pipeline_summary': {
                'total_deals': len(active_deals),
                'total_pipeline_value': total_pipeline,
                'weighted_pipeline_value': weighted_pipeline,
                'overall_conversion_rate': overall_conversion
            },
            'stage_analysis': stage_analysis,
            'pipeline_health_score': self._calculate_pipeline_health_score(stage_analysis),
            'pipeline_risks': pipeline_risks,
            'recommendations': self._generate_pipeline_recommendations(stage_analysis)
        }
        
    def calculate_customer_lifetime_value(self, customer_id: str) -> Dict:
        """Calculate customer lifetime value and metrics"""
        customer = next((c for c in self.customers if c.customer_id == customer_id), None)
        if not customer:
            raise ValueError(f"Customer not found: {customer_id}")
            
        # Get customer deals history
        customer_deals = [d for d in self.deals if d.customer_id == customer_id]
        
        # Calculate historical metrics
        total_revenue_to_date = sum(deal.value for deal in customer_deals 
                                  if deal.stage == DealStage.CLOSED_WON)
        total_contracts = len([d for d in customer_deals if deal.stage == DealStage.CLOSED_WON])
        
        # Calculate retention probability
        retention_probability = 1 - customer.churn_risk
        
        # Calculate expansion probability and rate
        expansion_probability = customer.expansion_potential
        avg_expansion_rate = 0.25  # 25% annual expansion rate assumption
        
        # Calculate lifetime value
        monthly_revenue = customer.annual_contract_value / 12
        expected_lifetime_months = 36  # Assume 3-year average lifetime
        
        # Present value calculation (assuming 10% discount rate)
        discount_rate = 0.10 / 12  # Monthly discount rate
        pv_factor = (1 - (1 + discount_rate) ** (-expected_lifetime_months)) / discount_rate
        present_value = monthly_revenue * pv_factor
        
        # Expansion value
        expansion_value = monthly_revenue * expansion_probability * avg_expansion_rate * pv_factor
        
        # Total LTV
        lifetime_value = present_value + expansion_value
        
        return {
            'customer_id': customer_id,
            'account_name': customer.account_name,
            'current_annual_value': customer.annual_contract_value,
            'total_revenue_to_date': total_revenue_to_date,
            'lifetime_value': lifetime_value,
            'present_value': present_value,
            'expansion_value': expansion_value,
            'expected_lifetime_months': expected_lifetime_months,
            'retention_probability': retention_probability,
            'expansion_probability': expansion_probability,
            'ltv_to_acv_ratio': lifetime_value / customer.annual_contract_value if customer.annual_contract_value > 0 else 0,
            'health_score': customer.health_score,
            'churn_risk': customer.churn_risk,
            'calculation_date': datetime.now().isoformat()
        }
        
    def analyze_marketing_roi(self, time_period_months: int = 12) -> Dict:
        """Analyze marketing ROI and attribution"""
        cutoff_date = datetime.now() - timedelta(days=time_period_months * 30)
        
        # Filter attribution data for time period
        relevant_attribution = [
            attr for attr in self.attribution_data 
            if attr.touchpoint_date >= cutoff_date
        ]
        
        # Group by marketing channel
        channel_performance = defaultdict(lambda: {
            'attribution_count': 0,
            'total_weight': 0,
            'revenue_attributed': 0,
            'avg_engagement': 0,
            'conversion_rate': 0
        })
        
        # Calculate channel metrics
        for attr in relevant_attribution:
            channel = attr.marketing_channel
            channel_performance[channel]['attribution_count'] += 1
            channel_performance[channel]['total_weight'] += attr.attribution_weight
            channel_performance[channel]['avg_engagement'] += attr.engagement_score
            
            # Attribute revenue (simplified - would need deal matching in real implementation)
            customer = next((c for c in self.customers if c.customer_id == attr.customer_id), None)
            if customer:
                channel_performance[channel]['revenue_attributed'] += (
                    customer.annual_contract_value * attr.attribution_weight
                )
                
        # Calculate averages and conversion rates
        for channel, data in channel_performance.items():
            if data['attribution_count'] > 0:
                data['avg_engagement'] /= data['attribution_count']
                data['conversion_rate'] = data['revenue_attributed'] / data['attribution_count'] if data['attribution_count'] > 0 else 0
                
        # Calculate overall marketing efficiency
        total_attributed_revenue = sum(data['revenue_attributed'] for data in channel_performance.values())
        total_marketing_touches = sum(data['attribution_count'] for data in channel_performance.values())
        
        # Rank channels by ROI
        channel_rankings = sorted(
            channel_performance.items(),
            key=lambda x: x[1]['revenue_attributed'],
            reverse=True
        )
        
        return {
            'analysis_period_months': time_period_months,
            'total_attributed_revenue': total_attributed_revenue,
            'total_marketing_touches': total_marketing_touches,
            'avg_revenue_per_touch': total_attributed_revenue / total_marketing_touches if total_marketing_touches > 0 else 0,
            'channel_performance': dict(channel_performance),
            'top_channels': channel_rankings[:5],
            'marketing_efficiency_score': self._calculate_marketing_efficiency_score(channel_performance),
            'recommendations': self._generate_marketing_recommendations(channel_performance)
        }
        
    def optimize_pricing_recommendations(self, customer_id: str, 
                                       competitive_data: Dict) -> Dict:
        """Generate pricing optimization recommendations"""
        customer = next((c for c in self.customers if c.customer_id == customer_id), None)
        if not customer:
            raise ValueError(f"Customer not found: {customer_id}")
            
        # Get customer deals
        customer_deals = [d for d in self.deals if d.customer_id == customer_id]
        active_deals = [d for d in customer_deals if d.stage not in [DealStage.CLOSED_WON, DealStage.CLOSED_LOST]]
        
        if not active_deals:
            return {'error': 'No active deals found for customer'}
            
        deal = active_deals[0]  # Use most recent active deal
        
        # Analyze current pricing position
        current_price = deal.value
        market_segment = customer.market_segment
        
        # Get competitive benchmarks (simplified)
        competitive_avg = competitive_data.get('average_price', current_price)
        price_position = self._calculate_price_position(current_price, competitive_avg)
        
        # Calculate optimization opportunities
        optimization_opportunities = self._identify_pricing_opportunities(
            customer, deal, competitive_data
        )
        
        # Generate recommendations
        recommendations = []
        if price_position['position'] == 'below_market':
            recommendations.append({
                'type': 'price_increase',
                'description': 'Increase price to market levels',
                'potential_uplift': competitive_avg - current_price,
                'confidence': 'high'
            })
        elif price_position['position'] == 'premium':
            recommendations.append({
                'type': 'value_demonstration',
                'description': 'Demonstrate value to justify premium pricing',
                'focus_areas': optimization_opportunities['value_drivers']
            })
            
        # Expansion opportunities
        if customer.expansion_potential > 0.5:
            recommendations.append({
                'type': 'expansion_pricing',
                'description': 'Offer expansion pricing with volume discounts',
                'potential_expansion': customer.annual_contract_value * 0.5,
                'discount_strategy': 'tiered_volume_discounts'
            })
            
        return {
            'customer_id': customer_id,
            'current_pricing': {
                'price': current_price,
                'market_position': price_position['position'],
                'competitive_gap': price_position['gap_percentage']
            },
            'optimization_opportunities': optimization_opportunities,
            'recommendations': recommendations,
            'expected_impact': self._estimate_pricing_optimization_impact(recommendations),
            'implementation_priority': self._prioritize_recommendations(recommendations)
        }
        
    def calculate_forecast_accuracy(self, historical_months: int = 6) -> Dict:
        """Calculate forecast accuracy metrics"""
        # Generate historical forecasts and compare to actual
        accuracy_metrics = []
        
        for months_ago in range(historical_months):
            forecast_date = datetime.now() - timedelta(days=(months_ago + 1) * 30)
            
            # Get actual revenue for the forecasted month
            actual_revenue = self._get_actual_revenue_for_month(forecast_date)
            
            # Get forecast made for that month
            forecast = self._get_forecast_for_date(forecast_date)
            
            if forecast and actual_revenue is not None:
                forecast_value = forecast.get('total_revenue', 0)
                accuracy = 1 - abs(forecast_value - actual_revenue) / max(forecast_value, actual_revenue)
                accuracy_metrics.append({
                    'forecast_date': forecast_date,
                    'forecasted_revenue': forecast_value,
                    'actual_revenue': actual_revenue,
                    'accuracy': accuracy,
                    'error_percentage': abs(forecast_value - actual_revenue) / actual_revenue * 100 if actual_revenue > 0 else 0
                })
                
        if not accuracy_metrics:
            return {'error': 'Insufficient historical data for accuracy calculation'}
            
        # Calculate aggregate metrics
        avg_accuracy = np.mean([m['accuracy'] for m in accuracy_metrics])
        avg_error = np.mean([m['error_percentage'] for m in accuracy_metrics])
        
        # Calculate forecast bias
        forecast_errors = [m['forecasted_revenue'] - m['actual_revenue'] for m in accuracy_metrics]
        bias = np.mean(forecast_errors)
        
        # Identify trends
        recent_accuracy = np.mean([m['accuracy'] for m in accuracy_metrics[-3:]])  # Last 3 months
        trend = 'improving' if recent_accuracy > avg_accuracy else 'declining' if recent_accuracy < avg_accuracy else 'stable'
        
        return {
            'analysis_period_months': historical_months,
            'average_accuracy': avg_accuracy,
            'average_error_percentage': avg_error,
            'forecast_bias': bias,
            'accuracy_trend': trend,
            'monthly_details': accuracy_metrics,
            'accuracy_grade': self._grade_forecast_accuracy(avg_accuracy)
        }
        
    def _initialize_sample_data(self) -> None:
        """Initialize with sample data for demonstration"""
        # Sample deals
        sample_deals = [
            Deal(
                deal_id="deal_001",
                customer_id="customer_001",
                customer_name="Metro Hospital System",
                market_segment=MarketSegment.HOSPITAL_SYSTEM,
                stage=DealStage.PROPOSAL,
                value=450000,
                probability=0.7,
                expected_close_date=datetime.now() + timedelta(days=45),
                created_date=datetime.now() - timedelta(days=60),
                last_activity_date=datetime.now() - timedelta(days=2),
                sales_rep="John Smith",
                product_category="Clinical AI Suite",
                competitive_situation="vs Epic",
                last_updated=datetime.now()
            ),
            Deal(
                deal_id="deal_002",
                customer_id="customer_002",
                customer_name="University Medical Center",
                market_segment=MarketSegment.AMC,
                stage=DealStage.NEGOTIATION,
                value=320000,
                probability=0.85,
                expected_close_date=datetime.now() + timedelta(days=30),
                created_date=datetime.now() - timedelta(days=90),
                last_activity_date=datetime.now() - timedelta(days=1),
                sales_rep="Sarah Johnson",
                product_category="Research AI Platform",
                competitive_situation="vs Cerner",
                last_updated=datetime.now()
            )
        ]
        
        # Sample customers
        sample_customers = [
            Customer(
                customer_id="customer_001",
                account_name="Metro Hospital System",
                market_segment=MarketSegment.HOSPITAL_SYSTEM,
                tier=CustomerTier.GOLD,
                annual_contract_value=450000,
                start_date=datetime.now() - timedelta(days=365),
                renewal_date=datetime.now() + timedelta(days=365),
                health_score=0.85,
                expansion_potential=0.6,
                churn_risk=0.15,
                product_usage={'clinical_ai': 0.8, 'analytics': 0.7},
                support_tickets=2,
                satisfaction_score=4.2
            )
        ]
        
        # Sample attribution data
        sample_attribution = [
            MarketingAttribution(
                attribution_id="attr_001",
                customer_id="customer_001",
                touchpoint_date=datetime.now() - timedelta(days=30),
                marketing_channel="webinar",
                campaign_id="webinar_clinical_ai",
                content_type="webinar",
                engagement_score=0.9,
                attribution_weight=0.3
            )
        ]
        
        self.deals.extend(sample_deals)
        self.customers.extend(sample_customers)
        self.attribution_data.extend(sample_attribution)
        
    def _create_forecast_scenario(self, scenario_name: str) -> None:
        """Create forecast scenario with assumptions"""
        base_assumptions = {
            'win_rate': 0.25,
            'average_deal_size': 150000,
            'sales_cycle_days': 120,
            'expansion_rate': 0.25,
            'churn_rate': 0.05
        }
        
        if scenario_name == "conservative":
            base_assumptions.update({
                'win_rate': 0.20,
                'average_deal_size': 130000,
                'expansion_rate': 0.15,
                'churn_rate': 0.08
            })
        elif scenario_name == "aggressive":
            base_assumptions.update({
                'win_rate': 0.35,
                'average_deal_size': 180000,
                'expansion_rate': 0.35,
                'churn_rate': 0.03
            })
            
        scenario = ForecastScenario(
            scenario_name=scenario_name,
            assumptions=base_assumptions,
            monthly_forecast=[],
            confidence_level=0.8 if scenario_name == "base_case" else 0.6,
            risk_factors=[],
            opportunity_factors=[]
        )
        
        self.forecast_scenarios[scenario_name] = scenario
        
    def _calculate_base_forecast(self, months: int) -> List[Dict]:
        """Calculate base revenue forecast"""
        forecast = []
        
        for month in range(1, months + 1):
            # Calculate new business revenue
            new_business = self._calculate_new_business_revenue(month)
            
            # Calculate expansion revenue
            expansion = self._calculate_expansion_revenue(month)
            
            # Calculate churn impact
            churn = self._calculate_churn_impact(month)
            
            total_revenue = new_business + expansion - churn
            
            forecast.append({
                'month': month,
                'new_business': new_business,
                'expansion': expansion,
                'churn': churn,
                'total_revenue': total_revenue,
                'cumulative_revenue': sum(m['total_revenue'] for m in forecast) + total_revenue
            })
            
        return forecast
        
    def _apply_scenario_adjustments(self, base_forecast: List[Dict], 
                                  scenario: ForecastScenario, months: int) -> List[Dict]:
        """Apply scenario-specific adjustments to forecast"""
        adjusted_forecast = []
        
        for i, month_data in enumerate(base_forecast):
            month_num = i + 1
            
            # Apply scenario multipliers
            win_rate_adj = scenario.assumptions['win_rate'] / 0.25  # Normalize to base case
            deal_size_adj = scenario.assumptions['average_deal_size'] / 150000
            
            adjusted_new_business = month_data['new_business'] * win_rate_adj * deal_size_adj
            adjusted_expansion = month_data['expansion'] * (1 + scenario.assumptions['expansion_rate'] - 0.25)
            adjusted_churn = month_data['churn'] * (1 + scenario.assumptions['churn_rate'] - 0.05)
            
            adjusted_total = adjusted_new_business + adjusted_expansion - adjusted_churn
            
            adjusted_forecast.append({
                'month': month_num,
                'new_business': adjusted_new_business,
                'expansion': adjusted_expansion,
                'churn': adjusted_churn,
                'total_revenue': adjusted_total,
                'cumulative_revenue': (adjusted_forecast[-1]['cumulative_revenue'] if adjusted_forecast else 0) + adjusted_total
            })
            
        return adjusted_forecast
        
    def _calculate_new_business_revenue(self, month: int) -> float:
        """Calculate new business revenue for month"""
        # Get deals expected to close in this month
        month_deals = [
            deal for deal in self.deals 
            if deal.expected_close_date.month == (datetime.now().month + month - 1) % 12 + 1
            and deal.stage in [DealStage.PROPOSAL, DealStage.NEGOTIATION]
        ]
        
        return sum(deal.value * deal.probability for deal in month_deals)
        
    def _calculate_expansion_revenue(self, month: int) -> float:
        """Calculate expansion revenue for month"""
        total_expansion = 0
        
        for customer in self.customers:
            # Calculate potential expansion based on health and expansion score
            if customer.health_score > 0.7 and customer.expansion_potential > 0.5:
                expansion_amount = customer.annual_contract_value * 0.25 / 12  # 25% annual expansion spread monthly
                total_expansion += expansion_amount
                
        return total_expansion
        
    def _calculate_churn_impact(self, month: int) -> float:
        """Calculate churn impact for month"""
        total_churn = 0
        
        for customer in self.customers:
            # Calculate churn based on churn risk score
            if customer.churn_risk > 0.3:
                churn_amount = customer.annual_contract_value * customer.churn_risk / 12
                total_churn += churn_amount
                
        return total_churn
        
    def _calculate_confidence_analysis(self, forecast: List[Dict]) -> Dict:
        """Calculate forecast confidence analysis"""
        # Calculate forecast volatility
        monthly_revenues = [m['total_revenue'] for m in forecast]
        revenue_volatility = np.std(monthly_revenues) / np.mean(monthly_revenues) if np.mean(monthly_revenues) > 0 else 0
        
        # Calculate confidence intervals (±20% at base confidence)
        confidence_level = 0.8
        interval_size = 0.2 / confidence_level
        
        confidence_intervals = []
        for month in forecast:
            revenue = month['total_revenue']
            lower_bound = revenue * (1 - interval_size)
            upper_bound = revenue * (1 + interval_size)
            confidence_intervals.append({
                'month': month['month'],
                'revenue': revenue,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
        return {
            'confidence_level': confidence_level,
            'forecast_volatility': revenue_volatility,
            'confidence_intervals': confidence_intervals,
            'confidence_grade': self._grade_forecast_confidence(confidence_level, revenue_volatility)
        }
        
    def _calculate_growth_rate(self, forecast: List[Dict]) -> float:
        """Calculate forecast growth rate"""
        if len(forecast) < 2:
            return 0.0
            
        first_month = forecast[0]['total_revenue']
        last_month = forecast[-1]['total_revenue']
        
        if first_month == 0:
            return 0.0
            
        # Calculate CAGR (Compound Annual Growth Rate)
        months = len(forecast)
        growth_rate = ((last_month / first_month) ** (12 / months) - 1) * 100
        
        return growth_rate
        
    def _estimate_stage_conversion(self, stage: str) -> float:
        """Estimate conversion rate to next stage"""
        conversion_rates = {
            'lead': 0.15,
            'qualified': 0.30,
            'proposal': 0.60,
            'negotiation': 0.80
        }
        return conversion_rates.get(stage, 0.25)
        
    def _calculate_overall_conversion_rate(self) -> float:
        """Calculate overall pipeline conversion rate"""
        total_pipeline = sum(deal.value for deal in self.deals)
        if total_pipeline == 0:
            return 0.0
            
        closed_won_value = sum(
            deal.value for deal in self.deals 
            if deal.stage == DealStage.CLOSED_WON
        )
        
        return closed_won_value / total_pipeline if total_pipeline > 0 else 0
        
    def _identify_pipeline_risks(self) -> List[Dict]:
        """Identify pipeline risks and issues"""
        risks = []
        
        # Check for deals in negotiation stage too long
        negotiation_deals = [d for d in self.deals if d.stage == DealStage.NEGOTIATION]
        for deal in negotiation_deals:
            days_in_stage = (datetime.now() - deal.last_activity_date).days
            if days_in_stage > 30:
                risks.append({
                    'type': 'stalled_deal',
                    'description': f"Deal {deal.deal_id} has been in negotiation for {days_in_stage} days",
                    'severity': 'high' if days_in_stage > 60 else 'medium',
                    'deal_value': deal.value
                })
                
        # Check for low-probability high-value deals
        high_value_low_prob = [
            d for d in self.deals 
            if d.value > 300000 and d.probability < 0.3 and d.stage in [DealStage.PROPOSAL, DealStage.NEGOTIATION]
        ]
        
        if high_value_low_prob:
            risks.append({
                'type': 'overoptimistic_forecasting',
                'description': f"{len(high_value_low_prob)} high-value deals with low probability",
                'severity': 'medium',
                'total_value': sum(d.value for d in high_value_low_prob)
            })
            
        return risks
        
    def _calculate_pipeline_health_score(self, stage_analysis: Dict) -> float:
        """Calculate overall pipeline health score"""
        if not stage_analysis:
            return 0.0
            
        # Weight factors for pipeline health
        weights = {
            'lead': 0.1,
            'qualified': 0.2,
            'proposal': 0.3,
            'negotiation': 0.4
        }
        
        health_score = 0.0
        total_weight = 0.0
        
        for stage, analysis in stage_analysis.items():
            if stage in weights:
                stage_health = analysis['conversion_rate_to_next']
                weight = weights[stage]
                health_score += stage_health * weight
                total_weight += weight
                
        return health_score / total_weight if total_weight > 0 else 0.0
        
    def _generate_pipeline_recommendations(self, stage_analysis: Dict) -> List[Dict]:
        """Generate pipeline improvement recommendations"""
        recommendations = []
        
        # Check lead generation
        if stage_analysis.get('lead', {}).get('deal_count', 0) < 20:
            recommendations.append({
                'type': 'lead_generation',
                'priority': 'high',
                'description': 'Increase lead generation efforts',
                'action': 'Implement targeted marketing campaigns'
            })
            
        # Check conversion rates
        for stage, analysis in stage_analysis.items():
            if analysis['conversion_rate_to_next'] < 0.2:
                recommendations.append({
                    'type': 'conversion_optimization',
                    'priority': 'high',
                    'description': f"Improve {stage} to next stage conversion",
                    'action': f"Focus on {stage} stage qualification and nurturing"
                })
                
        return recommendations
        
    def _calculate_price_position(self, our_price: float, competitive_price: float) -> Dict:
        """Calculate price positioning relative to competition"""
        if competitive_price == 0:
            return {'position': 'no_competition', 'gap_percentage': 0}
            
        ratio = our_price / competitive_price
        
        if ratio < 0.8:
            return {'position': 'below_market', 'gap_percentage': (1 - ratio) * 100}
        elif ratio < 1.1:
            return {'position': 'competitive', 'gap_percentage': (ratio - 1) * 100}
        elif ratio < 1.3:
            return {'position': 'premium', 'gap_percentage': (ratio - 1) * 100}
        else:
            return {'position': 'luxury', 'gap_percentage': (ratio - 1) * 100}
            
    def _identify_pricing_opportunities(self, customer: Customer, deal: Deal, 
                                      competitive_data: Dict) -> Dict:
        """Identify pricing optimization opportunities"""
        opportunities = {
            'value_drivers': [],
            'pricing_gaps': [],
            'expansion_opportunities': []
        }
        
        # Analyze value drivers
        if customer.health_score > 0.8:
            opportunities['value_drivers'].append('High customer satisfaction and engagement')
            
        if customer.expansion_potential > 0.6:
            opportunities['value_drivers'].append('High expansion potential')
            
        # Analyze pricing gaps
        segment_avg_deal_size = self._get_segment_avg_deal_size(customer.market_segment)
        if deal.value < segment_avg_deal_size * 0.8:
            opportunities['pricing_gaps'].append('Below segment average deal size')
            
        # Expansion opportunities
        if customer.product_usage.get('analytics', 0) < 0.5:
            opportunities['expansion_opportunities'].append('Analytics module upsell')
            
        return opportunities
        
    def _estimate_pricing_optimization_impact(self, recommendations: List[Dict]) -> Dict:
        """Estimate impact of pricing optimization"""
        total_uplift = 0
        affected_deals = 0
        
        for rec in recommendations:
            if rec.get('potential_uplift'):
                total_uplift += rec['potential_uplift']
                affected_deals += 1
            elif rec.get('potential_expansion'):
                total_uplift += rec['potential_expansion']
                affected_deals += 1
                
        return {
            'total_potential_uplift': total_uplift,
            'affected_deals': affected_deals,
            'average_uplift_per_deal': total_uplift / affected_deals if affected_deals > 0 else 0,
            'confidence': 'medium'
        }
        
    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize pricing recommendations"""
        priority_weights = {
            'price_increase': 3,
            'value_demonstration': 2,
            'expansion_pricing': 2
        }
        
        for rec in recommendations:
            rec['priority_score'] = priority_weights.get(rec['type'], 1)
            
        return sorted(recommendations, key=lambda x: x['priority_score'], reverse=True)
        
    def _calculate_marketing_efficiency_score(self, channel_performance: Dict) -> float:
        """Calculate overall marketing efficiency score"""
        if not channel_performance:
            return 0.0
            
        total_revenue = sum(data['revenue_attributed'] for data in channel_performance.values())
        total_touches = sum(data['attribution_count'] for data in channel_performance.values())
        
        if total_touches == 0:
            return 0.0
            
        avg_revenue_per_touch = total_revenue / total_touches
        
        # Normalize score (assuming $1000 per touch is good performance)
        efficiency_score = min(avg_revenue_per_touch / 1000, 1.0)
        
        return efficiency_score
        
    def _generate_marketing_recommendations(self, channel_performance: Dict) -> List[Dict]:
        """Generate marketing optimization recommendations"""
        recommendations = []
        
        # Find top performing channel
        if channel_performance:
            top_channel = max(channel_performance.items(), key=lambda x: x[1]['revenue_attributed'])
            recommendations.append({
                'type': 'increase_investment',
                'channel': top_channel[0],
                'description': f"Increase investment in {top_channel[0]} - top performing channel",
                'potential_impact': 'high'
            })
            
        # Find underperforming channels
        avg_performance = np.mean([data['revenue_attributed'] for data in channel_performance.values()])
        
        for channel, data in channel_performance.items():
            if data['revenue_attributed'] < avg_performance * 0.5:
                recommendations.append({
                    'type': 'optimize_or_reduce',
                    'channel': channel,
                    'description': f"Optimize or reduce investment in {channel}",
                    'potential_impact': 'medium'
                })
                
        return recommendations
        
    def _grade_forecast_accuracy(self, accuracy: float) -> str:
        """Grade forecast accuracy"""
        if accuracy >= 0.9:
            return 'A'
        elif accuracy >= 0.8:
            return 'B'
        elif accuracy >= 0.7:
            return 'C'
        elif accuracy >= 0.6:
            return 'D'
        else:
            return 'F'
            
    def _grade_forecast_confidence(self, confidence: float, volatility: float) -> str:
        """Grade forecast confidence"""
        if confidence >= 0.9 and volatility <= 0.3:
            return 'Very High'
        elif confidence >= 0.8 and volatility <= 0.4:
            return 'High'
        elif confidence >= 0.7 and volatility <= 0.5:
            return 'Medium'
        else:
            return 'Low'
            
    def _get_segment_avg_deal_size(self, segment: MarketSegment) -> float:
        """Get average deal size for market segment"""
        segment_avg_deals = {
            MarketSegment.HOSPITAL_SYSTEM: 450000,
            MarketSegment.AMC: 320000,
            MarketSegment.CLINIC: 85000,
            MarketSegment.IDN: 650000,
            MarketSegment.REGIONAL_HOSPITAL: 180000,
            MarketSegment.SPECIALTY_CLINIC: 120000,
            MarketSegment.RURAL_HOSPITAL: 65000
        }
        return segment_avg_deals.get(segment, 150000)
        
    def _get_actual_revenue_for_month(self, month_date: datetime) -> Optional[float]:
        """Get actual revenue for a specific month"""
        # This would integrate with actual billing/finance systems
        # For demo, return None to indicate missing data
        return None
        
    def _get_forecast_for_date(self, forecast_date: datetime) -> Optional[Dict]:
        """Get forecast made for a specific date"""
        # This would integrate with historical forecast data
        # For demo, return None to indicate missing data
        return None


if __name__ == "__main__":
    # Example usage
    rev_ops = RevenueOperations()
    
    # Generate forecast
    forecast = rev_ops.forecast_revenue(6, "base_case")
    print("Revenue Forecast:")
    print(json.dumps(forecast, indent=2, default=str))
    
    # Analyze pipeline health
    pipeline_health = rev_ops.analyze_pipeline_health()
    print("\nPipeline Health:")
    print(json.dumps(pipeline_health, indent=2, default=str))
    
    # Calculate customer LTV
    if rev_ops.customers:
        ltv = rev_ops.calculate_customer_lifetime_value(rev_ops.customers[0].customer_id)
        print("\nCustomer LTV:")
        print(json.dumps(ltv, indent=2, default=str))
    
    # Analyze marketing ROI
    marketing_roi = rev_ops.analyze_marketing_roi(6)
    print("\nMarketing ROI:")
    print(json.dumps(marketing_roi, indent=2, default=str))