"""
Main Revenue Optimization Manager

This module provides the main orchestrator class that integrates all components of the
healthcare AI revenue optimization and pricing framework.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import json
from datetime import datetime

from pricing_framework import (
    HealthcarePricingFramework, 
    MarketSegment, 
    CustomerTier, 
    CustomerProfile,
    ClinicalOutcome
)
from market_segment_analysis import MarketSegmentAnalyzer
from revenue_operations import RevenueOperations, Deal, Customer
from roi_calculators import HealthcareROICalculator, ClinicalMetric, OperationalMetric, CostComponent, BenefitComponent, TimeHorizon, OutcomeCategory
from config_manager import PricingConfigManager

class RevenueOptimizationManager:
    """Main manager for revenue optimization and pricing operations"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the revenue optimization manager"""
        self.config_manager = PricingConfigManager(config_dir)
        self.pricing_framework = HealthcarePricingFramework()
        self.market_analyzer = MarketSegmentAnalyzer()
        self.revenue_operations = RevenueOperations()
        self.roi_calculator = HealthcareROICalculator()
        
        # Initialize with default data
        self._initialize_default_data()
        
    def _initialize_default_data(self) -> None:
        """Initialize with sample data for demonstration"""
        # Add sample customers
        sample_customers = [
            CustomerProfile(
                customer_id="metro_hospital",
                organization_name="Metro Hospital System",
                market_segment=MarketSegment.HOSPITAL_SYSTEM,
                tier=CustomerTier.GOLD,
                annual_revenue=500000000,
                patient_volume=200000,
                clinical_specialties=["cardiology", "oncology", "radiology"],
                technology_adoption_score=0.75,
                budget_range={"min": 300000, "max": 800000},
                decision_makers=["CMO", "CIO", "CFO"],
                current_ai_spend=150000,
                competitive_solutions=["Epic", "Cerner"]
            ),
            CustomerProfile(
                customer_id="university_medical",
                organization_name="University Medical Center",
                market_segment=MarketSegment.AMC,
                tier=CustomerTier.PLATINUM,
                annual_revenue=600000000,
                patient_volume=250000,
                clinical_specialties=["research", "education", "clinical_care"],
                technology_adoption_score=0.85,
                budget_range={"min": 250000, "max": 600000},
                decision_makers=["Dean", "CIO", "Research_Director"],
                current_ai_spend=200000,
                competitive_solutions=["Custom Research Platform"]
            ),
            CustomerProfile(
                customer_id="family_clinic",
                organization_name="Family Care Clinic",
                market_segment=MarketSegment.CLINIC,
                tier=CustomerTier.SILVER,
                annual_revenue=50000000,
                patient_volume=25000,
                clinical_specialties=["primary_care", "family_medicine"],
                technology_adoption_score=0.65,
                budget_range={"min": 80000, "max": 150000},
                decision_makers=["Owner", "Practice_Manager"],
                current_ai_spend=25000,
                competitive_solutions=["athenahealth"]
            )
        ]
        
        for customer in sample_customers:
            self.pricing_framework.add_customer_profile(customer)
            
    def run_comprehensive_analysis(self, customer_id: str, 
                                 include_competitive_data: bool = True) -> Dict[str, Any]:
        """Run comprehensive revenue optimization analysis for a customer"""
        
        customer_profile = self.pricing_framework.customer_profiles.get(customer_id)
        if not customer_profile:
            return {"error": f"Customer {customer_id} not found"}
            
        # Get market segment analysis
        segment_analysis = self.market_analyzer.analyze_segment_opportunity(
            customer_profile.market_segment
        )
        
        # Optimize pricing for the customer
        competitive_data = {
            "average_price": segment_analysis.get("average_deal_size", 150000),
            "price_range": [segment_analysis.get("average_deal_size", 150000) * 0.8, 
                          segment_analysis.get("average_deal_size", 150000) * 1.2]
        } if include_competitive_data else {}
        
        pricing_optimization = self.pricing_framework.optimize_pricing(
            customer_id, competitive_data
        )
        
        # Create subscription model
        subscription_model = self.pricing_framework.create_subscription_model(
            customer_id, customer_profile.tier
        )
        
        # Calculate ROI with sample clinical outcomes
        clinical_outcomes = self._generate_sample_clinical_outcomes(customer_profile)
        roi_analysis = self.roi_calculator.calculate_comprehensive_roi(
            customer_id=customer_id,
            clinical_metrics=clinical_outcomes["clinical_metrics"],
            operational_metrics=clinical_outcomes["operational_metrics"],
            cost_components=clinical_outcomes["cost_components"],
            benefit_components=clinical_outcomes["benefit_components"]
        )
        
        # Generate comprehensive report
        roi_report = self.roi_calculator.generate_roi_report(roi_analysis)
        
        # Market segment forecast
        segment_forecast = self.market_analyzer.forecast_segment_performance(
            customer_profile.market_segment, 12
        )
        
        # Revenue operations analysis
        pipeline_analysis = self.revenue_operations.analyze_pipeline_health()
        ltv_analysis = self.revenue_operations.calculate_customer_lifetime_value(customer_id)
        
        return {
            "customer_profile": {
                "customer_id": customer_id,
                "organization_name": customer_profile.organization_name,
                "market_segment": customer_profile.market_segment.value,
                "tier": customer_profile.tier.value,
                "annual_revenue": customer_profile.annual_revenue,
                "patient_volume": customer_profile.patient_volume
            },
            "market_segment_analysis": segment_analysis,
            "pricing_optimization": pricing_optimization,
            "subscription_model": subscription_model,
            "roi_analysis": {
                "roi_percentage": roi_analysis.roi_percentage,
                "payback_period_months": roi_analysis.payback_period_months,
                "net_present_value": roi_analysis.net_present_value,
                "summary": roi_report["executive_summary"]
            },
            "market_forecast": segment_forecast,
            "revenue_operations": {
                "pipeline_health": pipeline_analysis,
                "customer_ltv": ltv_analysis
            },
            "recommendations": self._generate_comprehensive_recommendations(
                segment_analysis, pricing_optimization, roi_analysis
            ),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    def run_market_expansion_analysis(self, target_segments: List[str]) -> Dict[str, Any]:
        """Run market expansion analysis across multiple segments"""
        
        segments = [MarketSegment(seg) for seg in target_segments]
        
        # Compare segment strategies
        segment_comparison = self.market_analyzer.compare_segment_strategies(segments)
        
        # Generate forecasts for each segment
        segment_forecasts = {}
        for segment in segments:
            segment_forecasts[segment.value] = self.market_analyzer.forecast_segment_performance(
                segment, 12
            )
            
        # Calculate total market opportunity
        total_opportunity = sum(
            analysis["total_revenue"] for analysis in segment_forecasts.values()
        )
        
        # Resource allocation recommendations
        resource_allocation = segment_comparison.get("resource_allocation", {})
        
        # Prioritized entry strategy
        entry_strategy = self._generate_entry_strategy(segments, segment_comparison)
        
        return {
            "target_segments": target_segments,
            "segment_comparison": segment_comparison,
            "segment_forecasts": segment_forecasts,
            "total_market_opportunity": total_opportunity,
            "resource_allocation": resource_allocation,
            "entry_strategy": entry_strategy,
            "recommended_focus": segment_comparison.get("portfolio_metrics", {}),
            "analysis_date": datetime.now().isoformat()
        }
        
    def optimize_pricing_strategy(self, market_segment: str) -> Dict[str, Any]:
        """Optimize pricing strategy for a specific market segment"""
        
        segment = MarketSegment(market_segment)
        
        # Get segment analysis
        segment_analysis = self.market_analyzer.analyze_segment_opportunity(segment)
        
        # Get competitive landscape
        competitive_data = {
            "segment": market_segment,
            "competitive_intensity": segment_analysis.get("competitive_intensity", 0.5),
            "pricing_potential": segment_analysis.get("pricing_potential", {}),
            "recommended_approach": segment_analysis.get("recommended_approach", {})
        }
        
        # Get customer profiles in this segment
        segment_customers = [
            profile for profile in self.pricing_framework.customer_profiles.values()
            if profile.market_segment == segment
        ]
        
        # Analyze pricing patterns across customers
        pricing_analysis = self._analyze_pricing_patterns(segment_customers, competitive_data)
        
        # Generate pricing strategy recommendations
        pricing_strategy = self._generate_pricing_strategy(segment, segment_analysis, pricing_analysis)
        
        # Calculate financial projections
        financial_projections = self._calculate_pricing_projections(segment, pricing_strategy)
        
        return {
            "market_segment": market_segment,
            "segment_analysis": segment_analysis,
            "competitive_analysis": competitive_data,
            "customer_analysis": {
                "customer_count": len(segment_customers),
                "average_deal_size": pricing_analysis.get("average_deal_size", 0),
                "price_range": pricing_analysis.get("price_range", [0, 0])
            },
            "pricing_strategy": pricing_strategy,
            "financial_projections": financial_projections,
            "implementation_roadmap": self._create_pricing_implementation_roadmap(pricing_strategy),
            "success_metrics": self._define_success_metrics(pricing_strategy),
            "analysis_date": datetime.now().isoformat()
        }
        
    def generate_revenue_forecast(self, forecast_months: int = 12, 
                                scenario: str = "base_case") -> Dict[str, Any]:
        """Generate comprehensive revenue forecast"""
        
        # Get forecast from revenue operations
        revenue_forecast = self.revenue_operations.forecast_revenue(forecast_months, scenario)
        
        # Add market segment forecasts
        market_forecasts = {}
        for segment in MarketSegment:
            market_forecasts[segment.value] = self.market_analyzer.forecast_segment_performance(
                segment, forecast_months
            )
            
        # Combine forecasts
        combined_forecast = self._combine_forecasts(revenue_forecast, market_forecasts)
        
        # Calculate key metrics
        key_metrics = self._calculate_forecast_metrics(combined_forecast)
        
        # Identify risks and opportunities
        risk_analysis = self._analyze_forecast_risks(combined_forecast)
        
        return {
            "forecast_period": forecast_months,
            "scenario": scenario,
            "revenue_forecast": revenue_forecast,
            "market_segment_forecasts": market_forecasts,
            "combined_forecast": combined_forecast,
            "key_metrics": key_metrics,
            "risk_analysis": risk_analysis,
            "confidence_level": key_metrics.get("forecast_confidence", 0.8),
            "forecast_date": datetime.now().isoformat()
        }
        
    def benchmark_performance(self, benchmark_period_months: int = 12) -> Dict[str, Any]:
        """Benchmark performance against industry standards"""
        
        benchmarks = {}
        
        # Benchmark each market segment
        for segment in MarketSegment:
            segment_benchmark = self.roi_calculator.benchmark_roi_metrics(
                segment.value, peer_group_size=10
            )
            benchmarks[segment.value] = segment_benchmark
            
        # Calculate overall performance metrics
        overall_performance = self._calculate_overall_performance(benchmark_period_months)
        
        # Generate improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(benchmarks, overall_performance)
        
        return {
            "benchmark_period_months": benchmark_period_months,
            "segment_benchmarks": benchmarks,
            "overall_performance": overall_performance,
            "performance_grades": self._grade_performance(benchmarks, overall_performance),
            "improvement_recommendations": improvement_recommendations,
            "best_practices": self._identify_best_practices(benchmarks),
            "benchmark_date": datetime.now().isoformat()
        }
        
    def create_executive_dashboard(self, time_period: str = "current") -> Dict[str, Any]:
        """Create executive dashboard with key metrics and insights"""
        
        # Get current pipeline health
        pipeline_health = self.revenue_operations.analyze_pipeline_health()
        
        # Get recent customer LTVs
        customer_ltvs = {}
        for customer_id in list(self.pricing_framework.customer_profiles.keys())[:5]:
            try:
                customer_ltvs[customer_id] = self.revenue_operations.calculate_customer_lifetime_value(customer_id)
            except:
                continue
                
        # Get revenue forecast
        revenue_forecast = self.generate_revenue_forecast(6)
        
        # Get market opportunities
        market_opportunities = self.run_market_expansion_analysis([
            "hospital_system", "clinic", "academic_medical_center"
        ])
        
        # Calculate key performance indicators
        kpis = self._calculate_executive_kpis(pipeline_health, customer_ltvs, revenue_forecast)
        
        # Identify top priorities
        priorities = self._identify_executive_priorities(pipeline_health, market_opportunities, kpis)
        
        return {
            "dashboard_date": datetime.now().isoformat(),
            "time_period": time_period,
            "key_metrics": kpis,
            "pipeline_health": pipeline_health,
            "customer_performance": customer_ltvs,
            "revenue_forecast": revenue_forecast,
            "market_opportunities": market_opportunities,
            "executive_priorities": priorities,
            "alerts": self._generate_executive_alerts(pipeline_health, kpis),
            "recommendations": self._generate_executive_recommendations(priorities, kpis)
        }
        
    def export_analysis_results(self, analysis_type: str, 
                              customer_id: Optional[str] = None,
                              output_file: Optional[str] = None) -> Dict[str, Any]:
        """Export analysis results to file or return structured data"""
        
        if analysis_type == "customer_analysis":
            if not customer_id:
                return {"error": "Customer ID required for customer analysis"}
            results = self.run_comprehensive_analysis(customer_id)
        elif analysis_type == "market_expansion":
            results = self.run_market_expansion_analysis([
                "hospital_system", "clinic", "academic_medical_center"
            ])
        elif analysis_type == "pricing_strategy":
            results = self.optimize_pricing_strategy("hospital_system")
        elif analysis_type == "revenue_forecast":
            results = self.generate_revenue_forecast(12)
        elif analysis_type == "benchmark":
            results = self.benchmark_performance(12)
        elif analysis_type == "executive_dashboard":
            results = self.create_executive_dashboard()
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}
            
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            return {"message": f"Analysis results exported to {output_file}"}
        else:
            return results
            
    def _generate_sample_clinical_outcomes(self, customer_profile: CustomerProfile) -> Dict[str, Any]:
        """Generate sample clinical outcomes for ROI calculation"""
        
        # Clinical metrics based on customer profile
        clinical_metrics = [
            ClinicalMetric(
                metric_name="Diagnosis Accuracy",
                category="clinical_quality",
                current_performance=85,
                target_performance=95,
                measurement_unit="percentage",
                patient_volume_impacted=customer_profile.patient_volume // 4,
                financial_value_per_unit=50,
                regulatory_impact=True,
                quality_score_weight=0.3,
                confidence_level=0.85
            ),
            ClinicalMetric(
                metric_name="Readmission Rate",
                category="clinical_quality",
                current_performance=15,
                target_performance=12,
                measurement_unit="percentage",
                patient_volume_impacted=customer_profile.patient_volume // 10,
                financial_value_per_unit=15000,
                regulatory_impact=True,
                quality_score_weight=0.4,
                confidence_level=0.90
            )
        ]
        
        # Operational metrics
        operational_metrics = [
            OperationalMetric(
                metric_name="Administrative Efficiency",
                current_efficiency=60,
                target_efficiency=80,
                measurement_unit="percentage",
                baseline_cost=customer_profile.annual_revenue * 0.02,
                cost_reduction_percentage=25,
                time_savings_hours=2000,
                staff_hours_impacted=50,
                confidence_level=0.80
            )
        ]
        
        # Cost components
        cost_components = [
            CostComponent(
                cost_name="Software License",
                category="recurring",
                unit_cost=120000,
                quantity=1,
                frequency="annual",
                implementation_month=1,
                description="Annual software licensing fees"
            ),
            CostComponent(
                cost_name="Implementation Services",
                category="one_time",
                unit_cost=50000,
                quantity=1,
                frequency="one_time",
                implementation_month=1,
                description="Initial setup and configuration"
            )
        ]
        
        # Benefit components
        benefit_components = [
            BenefitComponent(
                benefit_name="Reduced Readmissions",
                category=OutcomeCategory.FINANCIAL_IMPACT,
                baseline_value=1000,
                improved_value=920,
                unit_value=15000,
                frequency="annual",
                measurement_period=12,
                confidence_level=0.85,
                description="Annual savings from reduced readmissions"
            ),
            BenefitComponent(
                benefit_name="Improved Patient Throughput",
                category=OutcomeCategory.OPERATIONAL_EFFICIENCY,
                baseline_value=100,
                improved_value=125,
                unit_value=200,
                frequency="monthly",
                measurement_period=1,
                confidence_level=0.9,
                description="Monthly revenue from increased patient volume"
            )
        ]
        
        return {
            "clinical_metrics": clinical_metrics,
            "operational_metrics": operational_metrics,
            "cost_components": cost_components,
            "benefit_components": benefit_components
        }
        
    def _generate_comprehensive_recommendations(self, segment_analysis: Dict, 
                                              pricing_optimization: Dict, 
                                              roi_analysis: Any) -> List[Dict]:
        """Generate comprehensive recommendations based on all analyses"""
        
        recommendations = []
        
        # Market strategy recommendations
        if segment_analysis.get("opportunity_score", 0) > 7:
            recommendations.append({
                "category": "market_strategy",
                "priority": "high",
                "recommendation": "High opportunity segment - increase investment",
                "rationale": f"Opportunity score: {segment_analysis.get('opportunity_score', 0):.1f}/10"
            })
            
        # Pricing recommendations
        if pricing_optimization.get("price_positioning", "") == "premium":
            recommendations.append({
                "category": "pricing",
                "priority": "medium",
                "recommendation": "Position as premium solution with value justification",
                "rationale": "Analysis shows premium pricing opportunity"
            })
        elif pricing_optimization.get("price_positioning", "") == "competitive":
            recommendations.append({
                "category": "pricing",
                "priority": "medium",
                "recommendation": "Maintain competitive pricing with value-add services",
                "rationale": "Market positioning suggests competitive approach"
            })
            
        # ROI recommendations
        if roi_analysis.roi_percentage < 100:
            recommendations.append({
                "category": "roi",
                "priority": "high",
                "recommendation": "Focus on high-impact use cases to improve ROI",
                "rationale": f"Current ROI of {roi_analysis.roi_percentage:.1f}% is below target"
            })
        elif roi_analysis.roi_percentage > 200:
            recommendations.append({
                "category": "roi",
                "priority": "low",
                "recommendation": "Strong ROI - expand similar implementations",
                "rationale": f"Excellent ROI of {roi_analysis.roi_percentage:.1f}% demonstrates value"
            })
            
        # Implementation recommendations
        if roi_analysis.payback_period_months > 24:
            recommendations.append({
                "category": "implementation",
                "priority": "medium",
                "recommendation": "Consider phased implementation to improve cash flow",
                "rationale": f"Payback period of {roi_analysis.payback_period_months:.1f} months is extended"
            })
            
        return recommendations
        
    def _generate_entry_strategy(self, segments: List[MarketSegment], 
                               segment_comparison: Dict) -> Dict[str, Any]:
        """Generate market entry strategy"""
        
        # Prioritize segments based on opportunity score
        ranked_segments = segment_comparison.get("segment_rankings", [])
        
        # Phase 1: High-opportunity segments
        phase1_segments = [seg["segment"] for seg in ranked_segments[:2]]
        
        # Phase 2: Medium-opportunity segments
        phase2_segments = [seg["segment"] for seg in ranked_segments[2:4]]
        
        # Phase 3: Remaining segments
        phase3_segments = [seg["segment"] for seg in ranked_segments[4:]]
        
        return {
            "phase_1_immediate": {
                "segments": phase1_segments,
                "rationale": "Highest opportunity scores - immediate focus",
                "timeline": "0-6 months",
                "resource_allocation": "60%"
            },
            "phase_2_expansion": {
                "segments": phase2_segments,
                "rationale": "Good opportunity scores - secondary focus",
                "timeline": "6-12 months",
                "resource_allocation": "30%"
            },
            "phase_3_long_term": {
                "segments": phase3_segments,
                "rationale": "Longer-term opportunities - future focus",
                "timeline": "12-24 months",
                "resource_allocation": "10%"
            }
        }
        
    def _analyze_pricing_patterns(self, customers: List[CustomerProfile], 
                                competitive_data: Dict) -> Dict[str, Any]:
        """Analyze pricing patterns across customers"""
        
        if not customers:
            return {"message": "No customers in segment"}
            
        # Calculate price statistics
        deal_sizes = [customer.current_ai_spend for customer in customers]
        
        return {
            "customer_count": len(customers),
            "average_deal_size": sum(deal_sizes) / len(deal_sizes),
            "price_range": [min(deal_sizes), max(deal_sizes)],
            "median_deal_size": sorted(deal_sizes)[len(deal_sizes) // 2],
            "competitive_positioning": competitive_data.get("competitive_intensity", 0.5)
        }
        
    def _generate_pricing_strategy(self, segment: MarketSegment, 
                                 segment_analysis: Dict, 
                                 pricing_analysis: Dict) -> Dict[str, Any]:
        """Generate pricing strategy recommendations"""
        
        # Base strategy on segment characteristics
        opportunity_score = segment_analysis.get("opportunity_score", 5)
        competitive_intensity = segment_analysis.get("competitive_intensity", 0.5)
        
        # Determine pricing approach
        if opportunity_score > 7 and competitive_intensity < 0.6:
            approach = "premium_value"
            rationale = "High opportunity with low competition"
        elif opportunity_score > 5:
            approach = "competitive_value"
            rationale = "Good opportunity with moderate competition"
        else:
            approach = "value_leader"
            rationale = "Focus on value leadership to differentiate"
            
        return {
            "recommended_approach": approach,
            "rationale": rationale,
            "target_positioning": "premium" if approach == "premium_value" else "competitive",
            "key_differentiators": segment_analysis.get("recommended_approach", {}).get("value_props", []),
            "pricing_model": "subscription",
            "discount_strategy": "tiered_volume_discounts",
            "value_communication": self._generate_value_communication(segment, approach)
        }
        
    def _calculate_pricing_projections(self, segment: MarketSegment, 
                                     pricing_strategy: Dict) -> Dict[str, Any]:
        """Calculate financial projections for pricing strategy"""
        
        # Get segment metrics
        segment_metrics = self.market_analyzer.market_metrics.get(segment)
        if not segment_metrics:
            return {"error": f"No metrics available for segment: {segment.value}"}
            
        # Calculate projected revenue
        annual_customers = segment_metrics.total_addressable_market / segment_metrics.average_deal_size
        market_penetration_rate = 0.02  # Assume 2% annual penetration
        
        projected_customers = annual_customers * market_penetration_rate
        projected_revenue = projected_customers * segment_metrics.average_deal_size
        
        return {
            "projected_annual_customers": int(projected_customers),
            "projected_annual_revenue": projected_revenue,
            "market_penetration_assumption": market_penetration_rate,
            "revenue_growth_projection": [
                projected_revenue * (1.1 ** year) for year in range(1, 4)
            ],
            "key_assumptions": [
                f"{market_penetration_rate:.1%} annual market penetration",
                f"Average deal size: ${segment_metrics.average_deal_size:,.0f}",
                "3-year projection period"
            ]
        }
        
    def _create_pricing_implementation_roadmap(self, pricing_strategy: Dict) -> Dict[str, Any]:
        """Create implementation roadmap for pricing strategy"""
        
        return {
            "phase_1_foundation": {
                "duration": "1-2 months",
                "activities": [
                    "Develop pricing documentation",
                    "Train sales team on new strategy",
                    "Create value proposition materials",
                    "Update CRM pricing rules"
                ]
            },
            "phase_2_pilot": {
                "duration": "2-3 months",
                "activities": [
                    "Test pricing with select customers",
                    "Gather feedback and adjust",
                    "Refine value messaging",
                    "Monitor competitive response"
                ]
            },
            "phase_3_rollout": {
                "duration": "3-6 months",
                "activities": [
                    "Full market rollout",
                    "Monitor performance metrics",
                    "Continuous optimization",
                    "Scale successful practices"
                ]
            }
        }
        
    def _define_success_metrics(self, pricing_strategy: Dict) -> List[Dict]:
        """Define success metrics for pricing strategy"""
        
        return [
            {
                "metric": "Average Deal Size",
                "target": "15% increase",
                "measurement": "monthly"
            },
            {
                "metric": "Win Rate",
                "target": "maintain or improve",
                "measurement": "monthly"
            },
            {
                "metric": "Sales Cycle Length",
                "target": "maintain or reduce",
                "measurement": "monthly"
            },
            {
                "metric": "Customer LTV",
                "target": "20% increase",
                "measurement": "quarterly"
            }
        ]
        
    def _combine_forecasts(self, revenue_forecast: Dict, 
                         market_forecasts: Dict) -> Dict[str, Any]:
        """Combine different forecast sources"""
        
        # Extract monthly data from revenue forecast
        revenue_monthly = revenue_forecast.get("monthly_forecast", [])
        
        # Calculate market segment contribution
        market_total = sum(
            forecast.get("total_revenue", 0) 
            for forecast in market_forecasts.values()
        )
        
        return {
            "combined_monthly_revenue": revenue_monthly,
            "market_segment_total": market_total,
            "revenue_operations_total": sum(
                month.get("total_revenue", 0) for month in revenue_monthly
            ),
            "forecast_accuracy_factors": {
                "historical_accuracy": 0.85,
                "market_volatility": 0.3,
                "competitive_pressure": 0.4
            }
        }
        
    def _calculate_forecast_metrics(self, combined_forecast: Dict) -> Dict[str, Any]:
        """Calculate key forecast metrics"""
        
        monthly_revenue = combined_forecast.get("combined_monthly_revenue", [])
        
        if not monthly_revenue:
            return {"error": "No forecast data available"}
            
        # Calculate growth rate
        first_month = monthly_revenue[0].get("total_revenue", 0)
        last_month = monthly_revenue[-1].get("total_revenue", 0)
        
        growth_rate = ((last_month / first_month) - 1) * 100 if first_month > 0 else 0
        
        # Calculate volatility
        revenues = [month.get("total_revenue", 0) for month in monthly_revenue]
        import statistics
        avg_revenue = statistics.mean(revenues)
        volatility = statistics.stdev(revenues) / avg_revenue if avg_revenue > 0 else 0
        
        return {
            "total_forecast_revenue": sum(revenues),
            "average_monthly_revenue": avg_revenue,
            "growth_rate": growth_rate,
            "revenue_volatility": volatility,
            "forecast_confidence": max(0.5, 1.0 - volatility)  # Higher volatility = lower confidence
        }
        
    def _analyze_forecast_risks(self, combined_forecast: Dict) -> Dict[str, Any]:
        """Analyze risks in revenue forecast"""
        
        risks = []
        
        # Check for high volatility
        monthly_revenue = combined_forecast.get("combined_monthly_revenue", [])
        if monthly_revenue:
            revenues = [month.get("total_revenue", 0) for month in monthly_revenue]
            if revenues:
                import statistics
                volatility = statistics.stdev(revenues) / statistics.mean(revenues) if statistics.mean(revenues) > 0 else 0
                if volatility > 0.5:
                    risks.append({
                        "risk_type": "high_volatility",
                        "severity": "medium",
                        "description": "Revenue forecast shows high volatility"
                    })
                    
        # Check for declining trend
        if len(monthly_revenue) > 3:
            recent_revenue = [month.get("total_revenue", 0) for month in monthly_revenue[-3:]]
            if all(recent_revenue[i] <= recent_revenue[i-1] for i in range(1, len(recent_revenue))):
                risks.append({
                    "risk_type": "declining_trend",
                    "severity": "high",
                    "description": "Revenue forecast shows declining trend"
                })
                
        return {
            "identified_risks": risks,
            "overall_risk_level": "high" if len(risks) >= 2 else "medium" if len(risks) == 1 else "low",
            "risk_mitigation": self._generate_risk_mitigation_strategies(risks)
        }
        
    def _calculate_overall_performance(self, period_months: int) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        
        # This would integrate with actual performance data
        # For now, return sample metrics
        return {
            "total_revenue": 2500000,
            "new_customers": 15,
            "average_deal_size": 175000,
            "win_rate": 0.28,
            "customer_satisfaction": 4.2,
            "market_share": 0.08,
            "competitive_win_rate": 0.65
        }
        
    def _generate_improvement_recommendations(self, benchmarks: Dict, 
                                            overall_performance: Dict) -> List[Dict]:
        """Generate improvement recommendations based on benchmarks"""
        
        recommendations = []
        
        # Analyze benchmark performance
        for segment, benchmark in benchmarks.items():
            peer_metrics = benchmark.get("benchmark_metrics", {})
            for metric_name, metric_data in peer_metrics.items():
                benchmark_value = metric_data.get("benchmark_value", 0)
                peer_average = metric_data.get("peer_average", 0)
                
                if benchmark_value < peer_average * 0.8:  # Below 80% of peer average
                    recommendations.append({
                        "category": "performance",
                        "segment": segment,
                        "metric": metric_name,
                        "recommendation": f"Improve {metric_name} to reach peer average",
                        "current_value": benchmark_value,
                        "target_value": peer_average,
                        "priority": "high"
                    })
                    
        return recommendations
        
    def _grade_performance(self, benchmarks: Dict, overall_performance: Dict) -> Dict[str, str]:
        """Grade performance against benchmarks"""
        
        grades = {}
        
        # Grade overall metrics
        if overall_performance.get("win_rate", 0) > 0.3:
            grades["win_rate"] = "A"
        elif overall_performance.get("win_rate", 0) > 0.25:
            grades["win_rate"] = "B"
        else:
            grades["win_rate"] = "C"
            
        if overall_performance.get("customer_satisfaction", 0) > 4.0:
            grades["customer_satisfaction"] = "A"
        elif overall_performance.get("customer_satisfaction", 0) > 3.5:
            grades["customer_satisfaction"] = "B"
        else:
            grades["customer_satisfaction"] = "C"
            
        return grades
        
    def _identify_best_practices(self, benchmarks: Dict) -> List[Dict]:
        """Identify best practices from benchmark analysis"""
        
        best_practices = []
        
        # Find segments with best performance
        for segment, benchmark in benchmarks.items():
            assessment = benchmark.get("overall_assessment", {})
            if assessment.get("level") in ["excellent", "good"]:
                best_practices.append({
                    "segment": segment,
                    "practice": f"Strong performance in {segment}",
                    "description": assessment.get("description", ""),
                    "applicable_segments": [segment]
                })
                
        return best_practices
        
    def _calculate_executive_kpis(self, pipeline_health: Dict, 
                                customer_ltvs: Dict, revenue_forecast: Dict) -> Dict[str, Any]:
        """Calculate key performance indicators for executive dashboard"""
        
        # Extract key metrics
        pipeline_value = pipeline_health.get("pipeline_summary", {}).get("total_pipeline_value", 0)
        weighted_pipeline = pipeline_health.get("pipeline_summary", {}).get("weighted_pipeline_value", 0)
        
        # Calculate average LTV
        ltv_values = [ltv.get("lifetime_value", 0) for ltv in customer_ltvs.values()]
        avg_ltv = sum(ltv_values) / len(ltv_values) if ltv_values else 0
        
        # Get forecast metrics
        forecast_metrics = revenue_forecast.get("key_metrics", {})
        
        return {
            "pipeline_value": pipeline_value,
            "weighted_pipeline_value": weighted_pipeline,
            "average_customer_ltv": avg_ltv,
            "forecasted_revenue": forecast_metrics.get("total_forecast_revenue", 0),
            "pipeline_health_score": pipeline_health.get("pipeline_health_score", 0),
            "customer_count": len(customer_ltvs),
            "forecast_confidence": forecast_metrics.get("forecast_confidence", 0.8)
        }
        
    def _identify_executive_priorities(self, pipeline_health: Dict, 
                                     market_opportunities: Dict, kpis: Dict) -> List[str]:
        """Identify executive priorities based on analysis"""
        
        priorities = []
        
        # Pipeline health priorities
        if pipeline_health.get("pipeline_health_score", 0) < 0.6:
            priorities.append("Improve pipeline health and conversion rates")
            
        # Revenue forecast priorities
        if kpis.get("forecast_confidence", 0) < 0.7:
            priorities.append("Increase forecast accuracy and reliability")
            
        # Market expansion priorities
        total_opportunity = market_opportunities.get("total_market_opportunity", 0)
        if total_opportunity > 5000000:
            priorities.append("Pursue high-value market expansion opportunities")
            
        # Customer LTV priorities
        avg_ltv = kpis.get("average_customer_ltv", 0)
        if avg_ltv < 500000:
            priorities.append("Increase customer lifetime value through expansion")
            
        return priorities[:5]  # Return top 5 priorities
        
    def _generate_executive_alerts(self, pipeline_health: Dict, kpis: Dict) -> List[Dict]:
        """Generate executive alerts for critical issues"""
        
        alerts = []
        
        # Pipeline alerts
        risks = pipeline_health.get("pipeline_risks", [])
        for risk in risks:
            if risk.get("severity") == "high":
                alerts.append({
                    "alert_type": "pipeline_risk",
                    "severity": "high",
                    "message": risk.get("description", ""),
                    "action_required": True
                })
                
        # KPI alerts
        if kpis.get("pipeline_health_score", 0) < 0.5:
            alerts.append({
                "alert_type": "pipeline_health",
                "severity": "high",
                "message": "Pipeline health score is critically low",
                "action_required": True
            })
            
        return alerts
        
    def _generate_executive_recommendations(self, priorities: List[str], kpis: Dict) -> List[Dict]:
        """Generate executive recommendations"""
        
        recommendations = []
        
        # Priority-based recommendations
        for priority in priorities[:3]:  # Top 3 priorities
            recommendations.append({
                "category": "strategic",
                "recommendation": priority,
                "expected_impact": "high",
                "timeline": "3-6 months"
            })
            
        # KPI-based recommendations
        if kpis.get("average_customer_ltv", 0) < 500000:
            recommendations.append({
                "category": "customer_value",
                "recommendation": "Implement customer expansion program to increase LTV",
                "expected_impact": "medium",
                "timeline": "6-12 months"
            })
            
        return recommendations
        
    def _generate_value_communication(self, segment: MarketSegment, approach: str) -> List[str]:
        """Generate value communication strategy"""
        
        if approach == "premium_value":
            return [
                "Demonstrate superior clinical outcomes",
                "Highlight ROI and payback period",
                "Showcase integration capabilities",
                "Provide reference customer testimonials"
            ]
        elif approach == "competitive_value":
            return [
                "Emphasize cost-effectiveness",
                "Highlight competitive advantages",
                "Demonstrate quick implementation",
                "Focus on customer support quality"
            ]
        else:  # value_leader
            return [
                "Lead with total cost of ownership",
                "Demonstrate efficiency gains",
                "Showcase innovation and future-proofing",
                "Highlight partnership approach"
            ]
            
    def _generate_risk_mitigation_strategies(self, risks: List[Dict]) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        for risk in risks:
            if risk.get("risk_type") == "high_volatility":
                strategies.append("Implement revenue diversification across segments")
            elif risk.get("risk_type") == "declining_trend":
                strategies.append("Increase marketing and sales investment to reverse trend")
            elif risk.get("risk_type") == "pipeline_risk":
                strategies.append("Accelerate deal closure processes and address bottlenecks")
                
        return strategies


if __name__ == "__main__":
    # Example usage
    manager = RevenueOptimizationManager()
    
    # Run comprehensive customer analysis
    print("Running comprehensive customer analysis...")
    analysis = manager.run_comprehensive_analysis("metro_hospital")
    print(f"Customer: {analysis['customer_profile']['organization_name']}")
    print(f"Market Segment: {analysis['customer_profile']['market_segment']}")
    print(f"ROI: {analysis['roi_analysis']['roi_percentage']:.1f}%")
    print(f"Recommendations: {len(analysis['recommendations'])} items")
    
    # Generate executive dashboard
    print("\nGenerating executive dashboard...")
    dashboard = manager.create_executive_dashboard()
    print(f"Pipeline Health Score: {dashboard['key_metrics']['pipeline_health_score']:.2f}")
    print(f"Customer LTV: ${dashboard['key_metrics']['average_customer_ltv']:,.0f}")
    print(f"Executive Priorities: {len(dashboard['executive_priorities'])} items")
    
    # Export results
    print("\nExporting analysis results...")
    export_result = manager.export_analysis_results("customer_analysis", "metro_hospital", "customer_analysis.json")
    print(export_result)