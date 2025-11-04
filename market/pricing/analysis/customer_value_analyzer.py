"""
Customer Value Analysis and Pricing Optimization Engine
Advanced analytics for healthcare AI pricing optimization
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CustomerProfile:
    """Customer profile for value analysis"""
    organization_id: str
    organization_type: str
    revenue: float
    patient_volume: int
    technology_readiness: float  # 0-1 scale
    outcome_focus: float  # 0-1 scale
    cost_sensitivity: float  # 0-1 scale
    decision_speed: str  # fast, medium, slow
    competitive_position: str
    geographic_market: str
    current_solutions: List[str]
    pain_points: List[str]

@dataclass
class ValueDriver:
    """Value driver definition"""
    driver_name: str
    category: str  # clinical, operational, financial
    measurement_unit: str
    baseline_value: float
    improvement_potential: float
    time_to_value_months: int
    risk_level: str  # low, medium, high
    value_coefficient: float

@dataclass
class PricingOptimizationResult:
    """Pricing optimization result"""
    optimal_price: float
    confidence_score: float
    value_proposition: Dict
    competitive_positioning: Dict
    negotiation_leverage: Dict
    recommendations: List[str]

class CustomerValueAnalyzer:
    """
    Customer Value Analysis and Pricing Optimization Engine
    Analyzes customer value and optimizes pricing strategies
    """
    
    def __init__(self):
        self.value_drivers = self._initialize_value_drivers()
        self.pricing_models = self._initialize_pricing_models()
        self.optimization_engine = OptimizationEngine()
        self.analytics_engine = AnalyticsEngine()
        
    def _initialize_value_drivers(self) -> Dict[str, ValueDriver]:
        """Initialize value drivers for healthcare AI"""
        return {
            "mortality_reduction": ValueDriver(
                driver_name="Mortality Rate Reduction",
                category="clinical",
                measurement_unit="deaths_per_year",
                baseline_value=0.06,
                improvement_potential=0.15,
                time_to_value_months=12,
                risk_level="medium",
                value_coefficient=50000
            ),
            "readmission_reduction": ValueDriver(
                driver_name="Readmission Rate Reduction",
                category="clinical",
                measurement_unit="readmissions_per_year",
                baseline_value=0.16,
                improvement_potential=0.25,
                time_to_value_months=9,
                risk_level="low",
                value_coefficient=30000
            ),
            "length_of_stay_reduction": ValueDriver(
                driver_name="Average Length of Stay Reduction",
                category="operational",
                measurement_unit="days_per_patient",
                baseline_value=4.2,
                improvement_potential=0.15,
                time_to_value_months=6,
                risk_level="low",
                value_coefficient=2000
            ),
            "diagnostic_accuracy": ValueDriver(
                driver_name="Diagnostic Accuracy Improvement",
                category="clinical",
                measurement_unit="accuracy_percentage",
                baseline_value=0.85,
                improvement_potential=0.08,
                time_to_value_months=8,
                risk_level="medium",
                value_coefficient=25000
            ),
            "staff_productivity": ValueDriver(
                driver_name="Staff Productivity Enhancement",
                category="operational",
                measurement_unit="hours_per_week",
                baseline_value=40,
                improvement_potential=0.20,
                time_to_value_months=4,
                risk_level="low",
                value_coefficient=75
            ),
            "patient_satisfaction": ValueDriver(
                driver_name="Patient Satisfaction Score",
                category="clinical",
                measurement_unit="satisfaction_score",
                baseline_value=82,
                improvement_potential=0.08,
                time_to_value_months=6,
                risk_level="medium",
                value_coefficient=10000
            ),
            "quality_score_improvement": ValueDriver(
                driver_name="Quality Score Enhancement",
                category="financial",
                measurement_unit="quality_points",
                baseline_value=75,
                improvement_potential=0.15,
                time_to_value_months=15,
                risk_level="high",
                value_coefficient=150000
            ),
            "cost_per_case_reduction": ValueDriver(
                driver_name="Cost Per Case Reduction",
                category="financial",
                measurement_unit="dollars_per_case",
                baseline_value=8500,
                improvement_potential=0.12,
                time_to_value_months=10,
                risk_level="medium",
                value_coefficient=1200
            )
        }
    
    def _initialize_pricing_models(self) -> Dict:
        """Initialize pricing models for different customer segments"""
        return {
            "value_based_pricing": {
                "base_methodology": "tied_to_clinical_outcomes",
                "price_range": {"min": 200000, "max": 1200000},
                "volume_tiers": {
                    "small": {"multiplier": 0.7, "threshold": 1000},
                    "medium": {"multiplier": 1.0, "threshold": 5000},
                    "large": {"multiplier": 1.4, "threshold": 15000},
                    "enterprise": {"multiplier": 2.0, "threshold": 30000}
                }
            },
            "subscription_pricing": {
                "base_methodology": "per_bed_per_month",
                "price_range": {"min": 150, "max": 500},
                "volume_tiers": {
                    "tier_1": {"beds": 50, "rate": 150},
                    "tier_2": {"beds": 200, "rate": 250},
                    "tier_3": {"beds": 500, "rate": 350},
                    "tier_4": {"beds": 1000, "rate": 450}
                }
            },
            "hybrid_pricing": {
                "base_methodology": "fixed_plus_variable",
                "fixed_component": 0.6,
                "variable_component": 0.4,
                "performance_multiplier": 1.2
            }
        }
    
    def analyze_customer_value(self, customer: CustomerProfile) -> Dict:
        """
        Comprehensive customer value analysis
        
        Args:
            customer: Customer profile
        
        Returns:
            Customer value analysis results
        """
        
        # Calculate customer value drivers
        value_contributions = self._calculate_value_contributions(customer)
        
        # Assess willingness to pay
        willingness_to_pay = self._assess_willingness_to_pay(customer, value_contributions)
        
        # Analyze competitive landscape
        competitive_analysis = self._analyze_competitive_landscape(customer)
        
        # Calculate customer lifetime value
        clv_analysis = self._calculate_customer_lifetime_value(customer, value_contributions)
        
        # Assess price sensitivity
        price_sensitivity = self._assess_price_sensitivity(customer)
        
        # Generate value proposition
        value_proposition = self._generate_value_proposition(customer, value_contributions)
        
        return {
            "customer_profile": {
                "organization_id": customer.organization_id,
                "organization_type": customer.organization_type,
                "revenue": customer.revenue,
                "patient_volume": customer.patient_volume
            },
            "value_analysis": {
                "total_annual_value": sum(value_contributions.values()),
                "value_drivers": value_contributions,
                "value_maturity_score": self._calculate_value_maturity_score(customer, value_contributions)
            },
            "financial_analysis": {
                "willingness_to_pay": willingness_to_pay,
                "customer_lifetime_value": clv_analysis,
                "price_sensitivity_score": price_sensitivity,
                "optimal_price_range": self._calculate_optimal_price_range(willingness_to_pay, value_contributions)
            },
            "competitive_positioning": competitive_analysis,
            "value_proposition": value_proposition,
            "optimization_recommendations": self._generate_optimization_recommendations(
                customer, value_contributions, competitive_analysis
            )
        }
    
    def _calculate_value_contributions(self, customer: CustomerProfile) -> Dict:
        """Calculate value contributions from each value driver"""
        
        contributions = {}
        
        for driver_name, driver in self.value_drivers.items():
            # Base calculation
            annual_impact = self._calculate_annual_impact(customer, driver)
            
            # Adjust for customer characteristics
            customer_factor = self._calculate_customer_factor(customer, driver_name)
            
            # Risk adjustment
            risk_factor = self._calculate_risk_factor(driver, customer)
            
            # Time to value adjustment
            time_factor = self._calculate_time_factor(driver)
            
            # Final contribution
            contribution = annual_impact * customer_factor * risk_factor * time_factor
            contributions[driver_name] = contribution
        
        return contributions
    
    def _calculate_annual_impact(self, customer: CustomerProfile, driver: ValueDriver) -> float:
        """Calculate annual impact for a value driver"""
        
        if driver.category == "clinical":
            # Clinical impact based on patient volume
            impact_per_patient = driver.improvement_potential * driver.value_coefficient
            return impact_per_patient * customer.patient_volume
        
        elif driver.category == "operational":
            # Operational impact based on organization size
            if "staff" in driver.driver_name.lower():
                impact_per_staff = driver.improvement_potential * driver.value_coefficient * 52  # weeks
                staff_estimate = customer.patient_volume / 100  # rough estimate
                return impact_per_staff * staff_estimate
            else:
                # Bed days impact
                bed_days = customer.patient_volume * 4.2  # avg length of stay
                impact_per_bed_day = driver.improvement_potential * driver.value_coefficient
                return impact_per_bed_day * bed_days
        
        elif driver.category == "financial":
            # Financial impact based on revenue
            financial_impact = customer.revenue * driver.improvement_potential * driver.value_coefficient / 1000000
            return financial_impact
        
        return 0
    
    def _calculate_customer_factor(self, customer: CustomerProfile, driver_name: str) -> float:
        """Calculate customer-specific factor for value driver"""
        
        factors = {
            "mortality_reduction": min(1.5, max(0.5, customer.technology_readiness * 1.2)),
            "readmission_reduction": min(1.3, max(0.7, customer.outcome_focus * 1.1)),
            "length_of_stay_reduction": min(1.4, max(0.6, customer.technology_readiness * 1.3)),
            "diagnostic_accuracy": min(1.2, max(0.8, customer.technology_readiness * 1.0)),
            "staff_productivity": min(1.3, max(0.7, customer.outcome_focus * 1.2)),
            "patient_satisfaction": min(1.2, max(0.8, customer.outcome_focus * 1.1)),
            "quality_score_improvement": min(1.5, max(0.6, customer.outcome_focus * 1.4)),
            "cost_per_case_reduction": min(1.3, max(0.7, customer.cost_sensitivity * 1.2))
        }
        
        return factors.get(driver_name, 1.0)
    
    def _calculate_risk_factor(self, driver: ValueDriver, customer: CustomerProfile) -> float:
        """Calculate risk adjustment factor"""
        
        risk_multipliers = {
            "low": 0.95,
            "medium": 0.85,
            "high": 0.70
        }
        
        base_risk = risk_multipliers[driver.risk_level]
        
        # Adjust based on customer technology readiness
        tech_adjustment = 0.9 + (customer.technology_readiness * 0.2)
        
        return base_risk * tech_adjustment
    
    def _calculate_time_factor(self, driver: ValueDriver) -> float:
        """Calculate time to value adjustment factor"""
        
        # Faster time to value = higher factor
        if driver.time_to_value_months <= 6:
            return 1.0
        elif driver.time_to_value_months <= 12:
            return 0.9
        else:
            return 0.8
    
    def _assess_willingness_to_pay(self, customer: CustomerProfile, value_contributions: Dict) -> Dict:
        """Assess customer willingness to pay"""
        
        # Calculate maximum willingness to pay
        total_value = sum(value_contributions.values())
        
        # Industry benchmarks for willingness to pay
        wtp_percentages = {
            "academic_medical_center": 0.35,
            "large_hospital": 0.30,
            "community_hospital": 0.25,
            "clinic": 0.40,
            "idn": 0.32
        }
        
        base_wtp = wtp_percentages.get(customer.organization_type, 0.25)
        
        # Adjust for customer characteristics
        technology_factor = 1.0 + (customer.technology_readiness * 0.2)
        outcome_factor = 1.0 + (customer.outcome_focus * 0.15)
        cost_factor = 1.0 - (customer.cost_sensitivity * 0.1)  # Higher cost sensitivity = lower WTP
        
        adjusted_wtp = base_wtp * technology_factor * outcome_factor * cost_factor
        
        # Calculate range
        min_wtp = total_value * adjusted_wtp * 0.8
        max_wtp = total_value * adjusted_wtp * 1.2
        
        return {
            "total_value": total_value,
            "willingness_to_pay_percentage": adjusted_wtp,
            "min_willingness_to_pay": min_wtp,
            "max_willingness_to_pay": max_wtp,
            "optimal_willingness_to_pay": total_value * adjusted_wtp,
            "confidence_score": self._calculate_wtp_confidence(customer)
        }
    
    def _calculate_wtp_confidence(self, customer: CustomerProfile) -> float:
        """Calculate confidence score for WTP assessment"""
        
        confidence_factors = [
            customer.technology_readiness,  # Technology readiness
            customer.outcome_focus,         # Outcome focus
            1 - customer.cost_sensitivity,  # Inverse cost sensitivity
        ]
        
        # Adjust for decision speed
        if customer.decision_speed == "fast":
            confidence_factors.append(0.9)
        elif customer.decision_speed == "medium":
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors)
    
    def _calculate_customer_lifetime_value(self, customer: CustomerProfile, value_contributions: Dict) -> Dict:
        """Calculate customer lifetime value"""
        
        # Calculate annual value
        annual_value = sum(value_contributions.values())
        
        # Estimate retention period
        retention_period = self._estimate_retention_period(customer)
        
        # Calculate CLV components
        retention_multiplier = 1 + (0.1 * retention_period)  # 10% annual growth
        discount_rate = 0.08
        
        clv_components = []
        for year in range(1, retention_period + 1):
            discounted_value = annual_value * (retention_multiplier ** year) / (1 + discount_rate) ** year
            clv_components.append(discounted_value)
        
        total_clv = sum(clv_components)
        
        # Customer acquisition cost estimate
        cac = self._estimate_acquisition_cost(customer)
        
        return {
            "total_clv": total_clv,
            "annual_value": annual_value,
            "retention_period_years": retention_period,
            "acquisition_cost": cac,
            "clv_to_cac_ratio": total_clv / cac if cac > 0 else 0,
            "present_value": total_clv,
            "break_even_months": (cac / (annual_value / 12)) if annual_value > 0 else 0
        }
    
    def _estimate_retention_period(self, customer: CustomerProfile) -> int:
        """Estimate customer retention period"""
        
        base_retention = 5  # years
        
        # Adjust based on customer characteristics
        if customer.technology_readiness > 0.8:
            base_retention += 1
        if customer.outcome_focus > 0.7:
            base_retention += 1
        if customer.cost_sensitivity < 0.3:
            base_retention += 0.5
        
        return int(base_retention)
    
    def _estimate_acquisition_cost(self, customer: CustomerProfile) -> float:
        """Estimate customer acquisition cost"""
        
        base_cac = 50000  # Base CAC
        
        # Adjust for organization size
        if customer.organization_type == "academic_medical_center":
            base_cac *= 2.0
        elif customer.organization_type == "large_hospital":
            base_cac *= 1.5
        elif customer.organization_type == "clinic":
            base_cac *= 0.6
        
        # Adjust for market competitiveness
        if customer.competitive_position == "strong":
            base_cac *= 1.3
        elif customer.competitive_position == "weak":
            base_cac *= 0.8
        
        return base_cac
    
    def _assess_price_sensitivity(self, customer: CustomerProfile) -> Dict:
        """Assess customer price sensitivity"""
        
        # Base sensitivity scores by organization type
        base_sensitivity = {
            "academic_medical_center": 0.4,
            "large_hospital": 0.6,
            "community_hospital": 0.7,
            "clinic": 0.8,
            "idn": 0.5
        }
        
        base_score = base_sensitivity.get(customer.organization_type, 0.6)
        
        # Adjust for customer characteristics
        technology_adjustment = -customer.technology_readiness * 0.2  # Higher tech = lower sensitivity
        outcome_adjustment = -customer.outcome_focus * 0.3          # Higher outcome focus = lower sensitivity
        cost_adjustment = customer.cost_sensitivity * 0.4           # Higher cost sensitivity = higher sensitivity
        
        final_score = base_score + technology_adjustment + outcome_adjustment + cost_adjustment
        final_score = max(0.1, min(1.0, final_score))  # Clamp between 0.1 and 1.0
        
        return {
            "sensitivity_score": final_score,
            "sensitivity_category": self._categorize_sensitivity(final_score),
            "price_elasticity": self._estimate_price_elasticity(customer),
            "negotiation_leverage": self._assess_negotiation_leverage(final_score, customer)
        }
    
    def _categorize_sensitivity(self, score: float) -> str:
        """Categorize price sensitivity"""
        if score <= 0.3:
            return "low_sensitivity"
        elif score <= 0.6:
            return "moderate_sensitivity"
        else:
            return "high_sensitivity"
    
    def _estimate_price_elasticity(self, customer: CustomerProfile) -> float:
        """Estimate price elasticity of demand"""
        
        # Base elasticity by organization type
        base_elasticity = {
            "academic_medical_center": -0.8,
            "large_hospital": -1.2,
            "community_hospital": -1.5,
            "clinic": -2.0,
            "idn": -1.0
        }
        
        elasticity = base_elasticity.get(customer.organization_type, -1.2)
        
        # Adjust for customer characteristics
        if customer.outcome_focus > 0.7:
            elasticity *= 0.8  # Less elastic for outcome-focused customers
        if customer.cost_sensitivity > 0.7:
            elasticity *= 1.2  # More elastic for cost-sensitive customers
        
        return elasticity
    
    def _assess_negotiation_leverage(self, sensitivity_score: float, customer: CustomerProfile) -> Dict:
        """Assess negotiation leverage"""
        
        # Leverage calculation
        if sensitivity_score > 0.7:
            leverage_level = "customer_leverage"
            leverage_strength = "high"
        elif sensitivity_score > 0.5:
            leverage_level = "balanced"
            leverage_strength = "medium"
        else:
            leverage_level = "vendor_leverage"
            leverage_strength = "low"
        
        return {
            "leverage_level": leverage_level,
            "leverage_strength": leverage_strength,
            "negotiation_strategy": self._suggest_negotiation_strategy(leverage_level),
            "price_flexibility": 1 - sensitivity_score
        }
    
    def _suggest_negotiation_strategy(self, leverage_level: str) -> str:
        """Suggest negotiation strategy"""
        
        strategies = {
            "customer_leverage": "Value-based negotiation with risk-sharing terms",
            "balanced": "Collaborative approach with mutual value creation",
            "vendor_leverage": "Outcome-based pricing with performance guarantees"
        }
        
        return strategies.get(leverage_level, "Collaborative value-based negotiation")
    
    def _calculate_optimal_price_range(self, wtp_analysis: Dict, value_contributions: Dict) -> Dict:
        """Calculate optimal price range"""
        
        total_value = wtp_analysis["total_value"]
        optimal_wtp = wtp_analysis["optimal_willingness_to_pay"]
        
        # Define pricing tiers based on value sharing
        return {
            "conservative": total_value * 0.20,  # 20% of total value
            "balanced": total_value * 0.30,     # 30% of total value
            "aggressive": total_value * 0.40,    # 40% of total value
            "maximize": total_value * 0.50,      # 50% of total value
            "recommended_range": {
                "min": total_value * 0.25,
                "max": total_value * 0.35,
                "optimal": optimal_wtp
            }
        }
    
    def _analyze_competitive_landscape(self, customer: CustomerProfile) -> Dict:
        """Analyze competitive positioning"""
        
        # Simulated competitive data
        competitors = {
            "primary_competitor": {"market_share": 0.35, "pricing_level": "premium"},
            "secondary_competitor": {"market_share": 0.25, "pricing_level": "competitive"},
            "tertiary_competitor": {"market_share": 0.15, "pricing_level": "value"}
        }
        
        our_position = self._determine_competitive_position(customer)
        
        return {
            "market_position": our_position,
            "competitive_threat_level": self._assess_competitive_threat(customer),
            "differentiation_opportunities": self._identify_differentiation_opportunities(customer),
            "pricing_pressure": self._assess_pricing_pressure(competitors, our_position)
        }
    
    def _determine_competitive_position(self, customer: CustomerProfile) -> str:
        """Determine our competitive position"""
        
        if customer.technology_readiness > 0.8 and customer.outcome_focus > 0.7:
            return "innovation_leader"
        elif customer.technology_readiness > 0.6:
            return "technology_differentiation"
        elif customer.outcome_focus > 0.6:
            return "clinical_excellence"
        else:
            return "cost_competitor"
    
    def _assess_competitive_threat(self, customer: CustomerProfile) -> str:
        """Assess competitive threat level"""
        
        threat_score = 0
        
        # Factors that increase threat
        if customer.competitive_position == "weak":
            threat_score += 2
        if customer.technology_readiness < 0.5:
            threat_score += 1
        if customer.cost_sensitivity > 0.7:
            threat_score += 1
        
        if threat_score >= 3:
            return "high"
        elif threat_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _identify_differentiation_opportunities(self, customer: CustomerProfile) -> List[str]:
        """Identify differentiation opportunities"""
        
        opportunities = []
        
        if customer.technology_readiness > 0.7:
            opportunities.append("Advanced AI capabilities")
        
        if customer.outcome_focus > 0.7:
            opportunities.append("Clinical outcome guarantees")
        
        if customer.pain_points:
            opportunities.extend([f"Address {pain_point}" for pain_point in customer.pain_points])
        
        if customer.current_solutions:
            opportunities.append("Seamless integration with existing systems")
        
        return opportunities
    
    def _assess_pricing_pressure(self, competitors: Dict, position: str) -> str:
        """Assess pricing pressure"""
        
        pressure_factors = {
            "innovation_leader": "low",
            "technology_differentiation": "low",
            "clinical_excellence": "medium",
            "cost_competitor": "high"
        }
        
        return pressure_factors.get(position, "medium")
    
    def _generate_value_proposition(self, customer: CustomerProfile, value_contributions: Dict) -> Dict:
        """Generate customized value proposition"""
        
        top_drivers = sorted(value_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "value_summary": f"Generate ${sum(value_contributions.values()):,.0f} annual value through AI-powered healthcare optimization",
            "primary_value_drivers": [driver[0] for driver in top_drivers],
            "value_breakdown": {driver: f"${value:,.0f}" for driver, value in top_drivers},
            "roi_projection": f"{self._calculate_roi_projection(value_contributions):.1f}x ROI",
            "time_to_value": f"{min([self.value_drivers[d].time_to_value_months for d in value_contributions.keys()]):.0f} months",
            "risk_mitigation": self._generate_risk_mitigation_statement(customer)
        }
    
    def _calculate_roi_projection(self, value_contributions: Dict) -> float:
        """Calculate ROI projection"""
        
        # Assume average pricing based on value
        assumed_price = sum(value_contributions.values()) * 0.30
        annual_value = sum(value_contributions.values())
        
        return annual_value / assumed_price if assumed_price > 0 else 0
    
    def _generate_risk_mitigation_statement(self, customer: CustomerProfile) -> str:
        """Generate risk mitigation statement"""
        
        statements = [
            "Performance-based pricing with outcome guarantees",
            "Comprehensive implementation support and training",
            "Regular performance reviews and optimization",
            "Phased rollout to minimize disruption"
        ]
        
        return statements
    
    def _calculate_value_maturity_score(self, customer: CustomerProfile, value_contributions: Dict) -> float:
        """Calculate value maturity score"""
        
        # Factors indicating value readiness
        factors = {
            "technology_readiness": customer.technology_readiness,
            "outcome_focus": customer.outcome_focus,
            "organization_size": min(1.0, customer.revenue / 500000000),  # Normalize to 0-1
            "patient_volume": min(1.0, customer.patient_volume / 50000),  # Normalize to 0-1
            "value_realization": sum(value_contributions.values()) / customer.revenue  # Value as % of revenue
        }
        
        return np.mean(list(factors.values()))
    
    def _generate_optimization_recommendations(self, 
                                             customer: CustomerProfile,
                                             value_contributions: Dict,
                                             competitive_analysis: Dict) -> List[str]:
        """Generate pricing optimization recommendations"""
        
        recommendations = []
        
        # Value-based recommendations
        total_value = sum(value_contributions.values())
        optimal_price = total_value * 0.30
        
        recommendations.append(f"Set pricing at ${optimal_price:,.0f} (30% of total value)")
        
        # Strategy recommendations based on competitive position
        position = competitive_analysis["market_position"]
        
        if position == "innovation_leader":
            recommendations.append("Emphasize innovation and outcomes in value proposition")
            recommendations.append("Consider premium pricing strategy")
        elif position == "technology_differentiation":
            recommendations.append("Highlight technical advantages and integration capabilities")
            recommendations.append("Focus on efficiency gains in sales process")
        elif position == "clinical_excellence":
            recommendations.append("Lead with clinical outcome improvements")
            recommendations.append("Provide detailed ROI case studies")
        
        # Customer-specific recommendations
        if customer.technology_readiness > 0.8:
            recommendations.append("Offer advanced features as upsell opportunities")
        
        if customer.outcome_focus > 0.7:
            recommendations.append("Propose outcome-based pricing components")
        
        if customer.cost_sensitivity > 0.7:
            recommendations.append("Consider flexible payment terms")
            recommendations.append("Emphasize cost avoidance benefits")
        
        return recommendations

class OptimizationEngine:
    """Pricing optimization engine using machine learning"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def optimize_pricing(self, customer: CustomerProfile, value_analysis: Dict) -> PricingOptimizationResult:
        """Optimize pricing using ML models"""
        
        # This would normally use historical pricing data
        # For demo purposes, we'll use rule-based optimization
        
        total_value = value_analysis["financial_analysis"]["willingness_to_pay"]["total_value"]
        wtp_percentage = value_analysis["financial_analysis"]["willingness_to_pay"]["willingness_to_pay_percentage"]
        
        # Calculate optimal price
        optimal_price = total_value * wtp_percentage
        
        # Apply market adjustments
        market_adjustment = self._apply_market_adjustments(customer)
        optimized_price = optimal_price * market_adjustment
        
        # Calculate confidence
        confidence = self._calculate_optimization_confidence(customer, value_analysis)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            customer, optimized_price, value_analysis
        )
        
        return PricingOptimizationResult(
            optimal_price=optimized_price,
            confidence_score=confidence,
            value_proposition=self._generate_optimized_value_prop(customer, optimized_price),
            competitive_positioning=self._assess_optimized_positioning(customer, optimized_price),
            negotiation_leverage=self._assess_optimized_leverage(customer, optimized_price),
            recommendations=recommendations
        )
    
    def _apply_market_adjustments(self, customer: CustomerProfile) -> float:
        """Apply market-based pricing adjustments"""
        
        adjustment = 1.0
        
        # Geographic adjustments
        geographic_adjustments = {
            "Northeast": 1.1,
            "Southeast": 0.95,
            "Midwest": 0.9,
            "Southwest": 0.85,
            "West": 1.05
        }
        
        # Organization size adjustments
        if customer.revenue > 1000000000:
            adjustment *= 1.2  # Large organizations pay premium
        elif customer.revenue < 100000000:
            adjustment *= 0.8  # Smaller organizations get discount
        
        return adjustment
    
    def _calculate_optimization_confidence(self, customer: CustomerProfile, value_analysis: Dict) -> float:
        """Calculate confidence in optimization result"""
        
        confidence_factors = [
            customer.technology_readiness,
            customer.outcome_focus,
            value_analysis["financial_analysis"]["willingness_to_pay"]["confidence_score"],
            1 - customer.cost_sensitivity
        ]
        
        return np.mean(confidence_factors)
    
    def _generate_optimization_recommendations(self, 
                                             customer: CustomerProfile,
                                             price: float,
                                             value_analysis: Dict) -> List[str]:
        """Generate specific optimization recommendations"""
        
        recommendations = []
        
        # Price-specific recommendations
        if price > 800000:
            recommendations.append("Consider multi-year contracts for price stability")
            recommendations.append("Offer volume discounts for multiple facilities")
        elif price < 200000:
            recommendations.append("Bundle additional services to increase value")
            recommendations.append("Focus on quick wins and rapid ROI demonstration")
        
        # Customer-specific recommendations
        if customer.decision_speed == "slow":
            recommendations.append("Provide extensive pilot program options")
            recommendations.append("Develop detailed change management plan")
        
        if customer.competitive_position == "weak":
            recommendations.append("Emphasize competitive advantages heavily")
            recommendations.append("Provide risk-free trial period")
        
        return recommendations
    
    def _generate_optimized_value_prop(self, customer: CustomerProfile, price: float) -> Dict:
        """Generate optimized value proposition"""
        
        return {
            "primary_message": f"Transform healthcare delivery with ${price:,.0f} annual AI investment",
            "value_equation": "Clinical Excellence + Operational Efficiency + Financial Performance",
            "roi_guarantee": "Minimum 2.5x ROI within 18 months",
            "unique_differentiators": [
                "Proven clinical outcome improvements",
                "Seamless integration capabilities",
                "Comprehensive support services"
            ]
        }
    
    def _assess_optimized_positioning(self, customer: CustomerProfile, price: float) -> Dict:
        """Assess optimized competitive positioning"""
        
        price_percentile = "premium" if price > 600000 else "competitive" if price > 300000 else "value"
        
        return {
            "pricing_position": price_percentile,
            "market_approach": "differentiation" if price_percentile == "premium" else "value",
            "competitive_response": "price_protection" if price_percentile == "premium" else "value_based"
        }
    
    def _assess_optimized_leverage(self, customer: CustomerProfile, price: float) -> Dict:
        """Assess negotiation leverage after optimization"""
        
        leverage_score = 0.5  # Neutral starting point
        
        # Adjust based on customer characteristics
        if customer.technology_readiness > 0.7:
            leverage_score += 0.1
        if customer.outcome_focus > 0.7:
            leverage_score += 0.1
        if customer.cost_sensitivity > 0.7:
            leverage_score -= 0.1
        
        return {
            "leverage_score": leverage_score,
            "negotiation_approach": "firm_but_flexible" if leverage_score > 0.5 else "collaborative",
            "price_flexibility": max(0.05, 0.20 - (abs(leverage_score - 0.5) * 0.3))
        }

class AnalyticsEngine:
    """Advanced analytics for pricing optimization"""
    
    def segment_customers(self, customers: List[CustomerProfile]) -> Dict:
        """Segment customers using clustering analysis"""
        
        # Prepare data for clustering
        features = []
        for customer in customers:
            features.append([
                customer.revenue,
                customer.patient_volume,
                customer.technology_readiness,
                customer.outcome_focus,
                customer.cost_sensitivity
            ])
        
        features = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # Analyze clusters
        segments = {}
        for i in range(4):
            cluster_customers = [customers[j] for j in range(len(customers)) if clusters[j] == i]
            segments[f"segment_{i+1}"] = {
                "customers": [c.organization_id for c in cluster_customers],
                "size": len(cluster_customers),
                "characteristics": self._analyze_cluster_characteristics(cluster_customers),
                "recommended_strategy": self._recommend_segment_strategy(cluster_customers)
            }
        
        return segments
    
    def _analyze_cluster_characteristics(self, customers: List[CustomerProfile]) -> Dict:
        """Analyze characteristics of customer cluster"""
        
        if not customers:
            return {}
        
        return {
            "avg_revenue": np.mean([c.revenue for c in customers]),
            "avg_patient_volume": np.mean([c.patient_volume for c in customers]),
            "avg_technology_readiness": np.mean([c.technology_readiness for c in customers]),
            "avg_outcome_focus": np.mean([c.outcome_focus for c in customers]),
            "avg_cost_sensitivity": np.mean([c.cost_sensitivity for c in customers]),
            "dominant_organization_type": max(set([c.organization_type for c in customers]), 
                                            key=[c.organization_type for c in customers].count)
        }
    
    def _recommend_segment_strategy(self, customers: List[CustomerProfile]) -> Dict:
        """Recommend pricing strategy for customer segment"""
        
        if not customers:
            return {}
        
        avg_tech_readiness = np.mean([c.technology_readiness for c in customers])
        avg_outcome_focus = np.mean([c.outcome_focus for c in customers])
        avg_cost_sensitivity = np.mean([c.cost_sensitivity for c in customers])
        
        # Determine strategy based on averages
        if avg_tech_readiness > 0.7 and avg_outcome_focus > 0.6:
            pricing_strategy = "premium_value_based"
            sales_approach = "consultative_outcomes"
        elif avg_cost_sensitivity > 0.7:
            pricing_strategy = "competitive_value"
            sales_approach = "roi_focused"
        else:
            pricing_strategy = "balanced_value"
            sales_approach = "collaborative"
        
        return {
            "pricing_strategy": pricing_strategy,
            "sales_approach": sales_approach,
            "key_messages": self._generate_segment_messages(avg_tech_readiness, avg_outcome_focus, avg_cost_sensitivity),
            "success_factors": self._identify_segment_success_factors(customers)
        }
    
    def _generate_segment_messages(self, tech_readiness: float, outcome_focus: float, cost_sensitivity: float) -> List[str]:
        """Generate key messages for segment"""
        
        messages = []
        
        if tech_readiness > 0.7:
            messages.append("Advanced AI capabilities and seamless integration")
        
        if outcome_focus > 0.6:
            messages.append("Proven clinical outcomes and measurable improvements")
        
        if cost_sensitivity > 0.7:
            messages.append("Clear ROI and cost-effective implementation")
        
        messages.append("Comprehensive support and ongoing optimization")
        
        return messages
    
    def _identify_segment_success_factors(self, customers: List[CustomerProfile]) -> List[str]:
        """Identify success factors for segment"""
        
        return [
            "Strong executive sponsorship",
            "Clear success metrics defined",
            "Comprehensive change management",
            "Regular performance reviews"
        ]

# Example usage and testing
if __name__ == "__main__":
    # Create sample customers
    customers = [
        CustomerProfile(
            organization_id="ORG001",
            organization_type="academic_medical_center",
            revenue=800000000,
            patient_volume=25000,
            technology_readiness=0.9,
            outcome_focus=0.85,
            cost_sensitivity=0.3,
            decision_speed="medium",
            competitive_position="strong",
            geographic_market="Northeast",
            current_solutions=["EMR", "PACS"],
            pain_points=["research_integration", "outcome_tracking"]
        ),
        CustomerProfile(
            organization_id="ORG002",
            organization_type="community_hospital",
            revenue=150000000,
            patient_volume=8000,
            technology_readiness=0.6,
            outcome_focus=0.7,
            cost_sensitivity=0.8,
            decision_speed="slow",
            competitive_position="weak",
            geographic_market="Midwest",
            current_solutions=["EMR"],
            pain_points=["cost_reduction", "staff_efficiency"]
        )
    ]
    
    # Initialize analyzer
    analyzer = CustomerValueAnalyzer()
    
    # Analyze each customer
    for customer in customers:
        print(f"\nCustomer Value Analysis for {customer.organization_id}")
        print("=" * 60)
        
        value_analysis = analyzer.analyze_customer_value(customer)
        
        print(f"Total Annual Value: ${value_analysis['value_analysis']['total_annual_value']:,.0f}")
        print(f"Optimal Price Range: ${value_analysis['financial_analysis']['optimal_price_range']['recommended_range']['min']:,.0f} - ${value_analysis['financial_analysis']['optimal_price_range']['recommended_range']['max']:,.0f}")
        print(f"Recommended Price: ${value_analysis['financial_analysis']['optimal_price_range']['recommended_range']['optimal']:,.0f}")
        print(f"Price Sensitivity: {value_analysis['financial_analysis']['price_sensitivity_score']['sensitivity_category']}")
        print(f"Customer Lifetime Value: ${value_analysis['financial_analysis']['customer_lifetime_value']['total_clv']:,.0f}")
