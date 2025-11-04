"""
Financial ROI Calculators for Healthcare AI Clients
Comprehensive ROI calculation models with clinical outcome metrics
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class HospitalProfile:
    """Hospital profile for ROI calculations"""
    name: str
    bed_count: int
    annual_patients: int
    avg_length_of_stay: float
    avg_cost_per_patient: float
    annual_revenue: float
    readmission_rate: float
    mortality_rate: float
    patient_satisfaction: float
    geographic_region: str

@dataclass
class ROIParameters:
    """ROI calculation parameters"""
    implementation_cost: float
    annual_subscription_cost: float
    expected_improvements: Dict[str, float]
    measurement_period_months: int
    discount_rate: float
    risk_adjustment_factor: float

class HealthcareROICalculator:
    """
    Healthcare AI ROI Calculator
    Calculates comprehensive ROI for healthcare AI investments
    """
    
    def __init__(self):
        self.clinical_benchmarks = self._load_clinical_benchmarks()
        self.cost_benchmarks = self._load_cost_benchmarks()
        self.roi_models = self._initialize_roi_models()
    
    def _load_clinical_benchmarks(self) -> Dict:
        """Load clinical benchmark data"""
        return {
            "mortality_rates": {
                "cardiac_surgery": {"baseline": 0.025, "top_quartile": 0.015},
                "stroke": {"baseline": 0.12, "top_quartile": 0.08},
                "sepsis": {"baseline": 0.18, "top_quartile": 0.12},
                "pneumonia": {"baseline": 0.08, "top_quartile": 0.05}
            },
            "readmission_rates": {
                "heart_failure": {"baseline": 0.25, "target": 0.18},
                "copd": {"baseline": 0.22, "target": 0.16},
                "pneumonia": {"baseline": 0.18, "target": 0.13},
                "joint_replacement": {"baseline": 0.08, "target": 0.05}
            },
            "diagnostic_accuracy": {
                "radiology": {"baseline": 0.85, "ai_enhanced": 0.94},
                "pathology": {"baseline": 0.82, "ai_enhanced": 0.92},
                "clinical_decision": {"baseline": 0.78, "ai_enhanced": 0.89}
            },
            "patient_satisfaction": {
                "overall": {"baseline": 82, "top_quartile": 90},
                "communication": {"baseline": 79, "top_quartile": 88},
                "care_coordination": {"baseline": 77, "top_quartile": 86}
            }
        }
    
    def _load_cost_benchmarks(self) -> Dict:
        """Load cost benchmark data"""
        return {
            "cost_per_case": {
                "cardiac_surgery": 45000,
                "stroke_care": 28000,
                "sepsis_treatment": 15000,
                "routine_procedure": 8000
            },
            "penalty_costs": {
                "readmission_penalty_per_case": 15000,
                "mortality_penalty_per_death": 25000,
                "quality_score_penalty": 500000
            },
            "operational_costs": {
                "cost_per_bed_day": 2000,
                "staff_cost_per_hour": 75,
                "admin_cost_per_case": 500
            }
        }
    
    def _initialize_roi_models(self) -> Dict:
        """Initialize ROI calculation models"""
        return {
            "clinical_outcomes_model": {
                "mortality_reduction": {
                    "value_per_death_prevented": 50000,
                    "quality_adjusted_life_years": 15,
                    "value_per_qaly": 50000
                },
                "readmission_reduction": {
                    "cost_per_readmission": 15000,
                    "quality_penalty": 15000
                },
                "diagnostic_accuracy": {
                    "cost_per_error": 25000,
                    "liability_cost": 100000
                }
            },
            "operational_efficiency_model": {
                "length_of_stay_reduction": {
                    "cost_per_bed_day": 2000,
                    "efficiency_multiplier": 1.5
                },
                "staff_productivity": {
                    "time_savings_per_hour": 75,
                    "annual_hours_saved": 2000
                },
                "workflow_optimization": {
                    "cost_per_visit": 500,
                    "volume_increase_percent": 0.10
                }
            },
            "financial_model": {
                "revenue_enhancement": {
                    "quality_score_improvement": 0.15,
                    "revenue_per_quality_point": 100000,
                    "reimbursement_increase": 0.08
                },
                "cost_avoidance": {
                    "unnecessary_test_reduction": 0.20,
                    "cost_per_test": 500,
                    "error_reduction_savings": 0.15
                }
            }
        }
    
    def calculate_comprehensive_roi(self, 
                                  hospital: HospitalProfile,
                                  ai_solution_type: str,
                                  parameters: ROIParameters) -> Dict:
        """
        Calculate comprehensive ROI for healthcare AI investment
        
        Args:
            hospital: Hospital profile
            ai_solution_type: Type of AI solution
            parameters: ROI calculation parameters
        
        Returns:
            Comprehensive ROI analysis
        """
        
        # Calculate baseline costs and revenues
        baseline_analysis = self._calculate_baseline_metrics(hospital)
        
        # Calculate improvement scenarios
        improvement_scenarios = self._calculate_improvement_scenarios(
            hospital, ai_solution_type, parameters
        )
        
        # Calculate financial benefits
        financial_benefits = self._calculate_financial_benefits(
            hospital, improvement_scenarios, baseline_analysis
        )
        
        # Calculate costs
        total_costs = self._calculate_total_costs(parameters)
        
        # Generate ROI analysis
        roi_analysis = self._generate_roi_analysis(
            total_costs, financial_benefits, parameters.measurement_period_months
        )
        
        # Create sensitivity analysis
        sensitivity_analysis = self._create_sensitivity_analysis(
            hospital, ai_solution_type, parameters
        )
        
        # Generate recommendations
        recommendations = self._generate_roi_recommendations(roi_analysis, sensitivity_analysis)
        
        return {
            "hospital_profile": {
                "name": hospital.name,
                "bed_count": hospital.bed_count,
                "annual_patients": hospital.annual_patients,
                "annual_revenue": hospital.annual_revenue
            },
            "ai_solution": ai_solution_type,
            "baseline_analysis": baseline_analysis,
            "improvement_scenarios": improvement_scenarios,
            "financial_benefits": financial_benefits,
            "cost_analysis": total_costs,
            "roi_metrics": roi_analysis,
            "sensitivity_analysis": sensitivity_analysis,
            "recommendations": recommendations,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _calculate_baseline_metrics(self, hospital: HospitalProfile) -> Dict:
        """Calculate baseline hospital metrics"""
        
        # Clinical metrics
        annual_mortality_cases = int(hospital.annual_patients * hospital.mortality_rate)
        annual_readmission_cases = int(hospital.annual_patients * hospital.readmission_rate)
        
        # Cost metrics
        annual_patient_costs = hospital.annual_patients * hospital.avg_cost_per_patient
        total_bed_days = hospital.bed_count * 365
        annual_bed_day_costs = total_bed_days * self.cost_benchmarks["operational_costs"]["cost_per_bed_day"]
        
        # Revenue metrics
        quality_score_penalty = self._estimate_quality_penalty(hospital.patient_satisfaction)
        readmission_penalties = annual_readmission_cases * self.cost_benchmarks["penalty_costs"]["readmission_penalty_per_case"]
        
        return {
            "clinical_metrics": {
                "annual_mortality_cases": annual_mortality_cases,
                "annual_readmission_cases": annual_readmission_cases,
                "avg_length_of_stay": hospital.avg_length_of_stay,
                "patient_satisfaction_score": hospital.patient_satisfaction
            },
            "financial_metrics": {
                "annual_patient_costs": annual_patient_costs,
                "annual_bed_day_costs": annual_bed_day_costs,
                "quality_penalties": quality_score_penalty,
                "readmission_penalties": readmission_penalties,
                "total_annual_costs": annual_patient_costs + annual_bed_day_costs + quality_score_penalty + readmission_penalties
            },
            "operational_metrics": {
                "annual_bed_days": total_bed_days,
                "bed_utilization_rate": min(1.0, hospital.annual_patients * hospital.avg_length_of_stay / total_bed_days),
                "staff_efficiency_score": 0.75  # Assumed baseline
            }
        }
    
    def _calculate_improvement_scenarios(self, 
                                       hospital: HospitalProfile,
                                       ai_solution_type: str,
                                       parameters: ROIParameters) -> Dict:
        """Calculate improvement scenarios based on AI solution"""
        
        # AI solution improvement multipliers
        improvement_multipliers = {
            "clinical_decision_support": {
                "mortality_reduction": 0.15,
                "readmission_reduction": 0.20,
                "diagnostic_accuracy": 0.08,
                "length_of_stay_reduction": 0.12
            },
            "predictive_analytics": {
                "mortality_reduction": 0.25,
                "readmission_reduction": 0.30,
                "diagnostic_accuracy": 0.05,
                "length_of_stay_reduction": 0.18
            },
            "workflow_optimization": {
                "mortality_reduction": 0.05,
                "readmission_reduction": 0.15,
                "diagnostic_accuracy": 0.03,
                "length_of_stay_reduction": 0.25,
                "staff_productivity": 0.20
            },
            "comprehensive_ai_platform": {
                "mortality_reduction": 0.30,
                "readmission_reduction": 0.35,
                "diagnostic_accuracy": 0.12,
                "length_of_stay_reduction": 0.22,
                "staff_productivity": 0.25
            }
        }
        
        multipliers = improvement_multipliers.get(ai_solution_type, improvement_multipliers["clinical_decision_support"])
        
        # Calculate improved metrics
        baseline = self._calculate_baseline_metrics(hospital)
        
        scenarios = {}
        
        for metric, multiplier in multipliers.items():
            if metric == "mortality_reduction":
                reduction = multiplier * baseline["clinical_metrics"]["annual_mortality_cases"]
                scenarios[metric] = {
                    "absolute_reduction": reduction,
                    "percentage_improvement": multiplier,
                    "cases_improved": int(reduction)
                }
            elif metric == "readmission_reduction":
                reduction = multiplier * baseline["clinical_metrics"]["annual_readmission_cases"]
                scenarios[metric] = {
                    "absolute_reduction": reduction,
                    "percentage_improvement": multiplier,
                    "cases_improved": int(reduction)
                }
            elif metric == "length_of_stay_reduction":
                reduction_days = multiplier * baseline["clinical_metrics"]["avg_length_of_stay"]
                scenarios[metric] = {
                    "days_reduced": reduction_days,
                    "percentage_improvement": multiplier,
                    "annual_days_saved": reduction_days * hospital.annual_patients
                }
            elif metric == "staff_productivity":
                scenarios[metric] = {
                    "productivity_improvement": multiplier,
                    "hours_saved_annually": 2000 * multiplier,
                    "cost_savings": 2000 * 75 * multiplier  # $75/hour
                }
        
        return scenarios
    
    def _calculate_financial_benefits(self, 
                                    hospital: HospitalProfile,
                                    scenarios: Dict,
                                    baseline: Dict) -> Dict:
        """Calculate financial benefits from improvements"""
        
        benefits = {}
        
        # Clinical outcome benefits
        if "mortality_reduction" in scenarios:
            mortality_cases = scenarios["mortality_reduction"]["cases_improved"]
            benefits["mortality_reduction"] = mortality_cases * 50000  # $50K per death prevented
        
        if "readmission_reduction" in scenarios:
            readmission_cases = scenarios["readmission_reduction"]["cases_improved"]
            benefits["readmission_reduction"] = readmission_cases * 30000  # $30K per readmission avoided
        
        # Operational efficiency benefits
        if "length_of_stay_reduction" in scenarios:
            days_saved = scenarios["length_of_stay_reduction"]["annual_days_saved"]
            benefits["length_of_stay_reduction"] = days_saved * 2000  # $2K per bed day
        
        if "staff_productivity" in scenarios:
            benefits["staff_productivity"] = scenarios["staff_productivity"]["cost_savings"]
        
        # Quality score improvements
        if "diagnostic_accuracy" in scenarios:
            quality_benefit = hospital.annual_revenue * 0.08  # 8% revenue increase from quality
            benefits["quality_improvement"] = quality_benefit
        
        # Calculate total benefits
        total_benefits = sum(benefits.values())
        
        # Annual vs multi-year benefits
        benefits["annual_total"] = total_benefits
        benefits["five_year_total"] = total_benefits * 5 * 0.95**np.arange(5)  # Discounting
        benefits["ten_year_total"] = total_benefits * 10 * 0.95**np.arange(10)  # Discounting
        
        return benefits
    
    def _calculate_total_costs(self, parameters: ROIParameters) -> Dict:
        """Calculate total costs of AI implementation"""
        
        implementation_cost = parameters.implementation_cost
        annual_subscription = parameters.annual_subscription_cost
        total_first_year = implementation_cost + annual_subscription
        
        # Multi-year costs
        years = parameters.measurement_period_months / 12
        total_subscription_cost = annual_subscription * years
        total_costs = implementation_cost + total_subscription_cost
        
        costs = {
            "implementation_cost": implementation_cost,
            "annual_subscription": annual_subscription,
            "total_first_year": total_first_year,
            "total_measurement_period": total_costs,
            "annual_subscription_breakdown": {
                "software_licensing": annual_subscription * 0.6,
                "support_and_maintenance": annual_subscription * 0.25,
                "training_and_onboarding": annual_subscription * 0.15
            }
        }
        
        return costs
    
    def _generate_roi_analysis(self, 
                             costs: Dict,
                             benefits: Dict,
                             period_months: int) -> Dict:
        """Generate comprehensive ROI analysis"""
        
        # Basic ROI calculations
        net_benefit = benefits["annual_total"] - costs["annual_subscription"]
        roi_ratio = net_benefit / costs["total_measurement_period"]
        
        # Payback period calculation
        payback_months = costs["total_first_year"] / (benefits["annual_total"] / 12)
        
        # NPV calculation
        discount_rate = 0.08  # 8% discount rate
        npv_benefits = 0
        for year in range(1, 6):
            npv_benefits += benefits["annual_total"] / (1 + discount_rate) ** year
        
        npv_costs = costs["total_measurement_period"]
        npv = npv_benefits - npv_costs
        
        # IRR calculation (simplified)
        cash_flows = [-costs["total_first_year"]] + [benefits["annual_total"]] * 5
        irr = self._calculate_irr(cash_flows)
        
        return {
            "roi_metrics": {
                "roi_ratio": roi_ratio,
                "roi_percentage": roi_ratio * 100,
                "net_present_value": npv,
                "internal_rate_of_return": irr,
                "payback_period_months": payback_months
            },
            "financial_summary": {
                "total_investment": costs["total_measurement_period"],
                "total_benefits": benefits["five_year_total"],
                "net_benefit": benefits["five_year_total"] - costs["total_measurement_period"],
                "benefit_cost_ratio": benefits["five_year_total"] / costs["total_measurement_period"]
            },
            "cash_flow_analysis": self._generate_cash_flow_analysis(costs, benefits),
            "risk_assessment": self._assess_investment_risk(roi_ratio, npv, payback_months)
        }
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return (simplified)"""
        # This is a simplified IRR calculation
        # In practice, you'd use more sophisticated methods
        try:
            npv_test = lambda rate: sum([cf / (1 + rate) ** i for i, cf in enumerate(cash_flows)])
            
            # Bisection method to find IRR
            low, high = 0.0, 1.0
            while high - low > 0.001:
                mid = (low + high) / 2
                if npv_test(mid) > 0:
                    low = mid
                else:
                    high = mid
            
            return (low + high) / 2
        except:
            return 0.15  # Default 15% IRR
    
    def _generate_cash_flow_analysis(self, costs: Dict, benefits: Dict) -> Dict:
        """Generate detailed cash flow analysis"""
        
        cash_flows = []
        for year in range(1, 6):
            year_cash_flow = {
                "year": year,
                "implementation_cost": costs["implementation_cost"] if year == 1 else 0,
                "annual_subscription": costs["annual_subscription"] if year >= 1 else 0,
                "annual_benefits": benefits["annual_total"],
                "net_cash_flow": benefits["annual_total"] - costs["annual_subscription"] - 
                               (costs["implementation_cost"] if year == 1 else 0),
                "cumulative_cash_flow": 0
            }
            
            if year == 1:
                year_cash_flow["cumulative_cash_flow"] = year_cash_flow["net_cash_flow"]
            else:
                year_cash_flow["cumulative_cash_flow"] = (cash_flows[-1]["cumulative_cash_flow"] + 
                                                        year_cash_flow["net_cash_flow"])
            
            cash_flows.append(year_cash_flow)
        
        return {"annual_cash_flows": cash_flows}
    
    def _assess_investment_risk(self, roi_ratio: float, npv: float, payback_months: float) -> Dict:
        """Assess investment risk profile"""
        
        risk_factors = []
        risk_score = 0
        
        # ROI-based risk
        if roi_ratio < 1.0:
            risk_factors.append("Negative ROI")
            risk_score += 3
        elif roi_ratio < 2.0:
            risk_factors.append("Low ROI")
            risk_score += 1
        
        # NPV-based risk
        if npv < 0:
            risk_factors.append("Negative NPV")
            risk_score += 2
        
        # Payback period risk
        if payback_months > 36:
            risk_factors.append("Long payback period")
            risk_score += 2
        elif payback_months > 24:
            risk_factors.append("Moderate payback period")
            risk_score += 1
        
        # Overall risk rating
        if risk_score >= 5:
            risk_rating = "high"
        elif risk_score >= 3:
            risk_rating = "moderate"
        else:
            risk_rating = "low"
        
        return {
            "risk_score": risk_score,
            "risk_rating": risk_rating,
            "risk_factors": risk_factors,
            "mitigation_strategies": self._suggest_risk_mitigation(risk_rating)
        }
    
    def _suggest_risk_mitigation(self, risk_rating: str) -> List[str]:
        """Suggest risk mitigation strategies"""
        
        strategies = {
            "high": [
                "Implement phased rollout approach",
                "Negotiate performance-based pricing",
                "Ensure comprehensive change management",
                "Establish clear success metrics"
            ],
            "moderate": [
                "Develop detailed implementation plan",
                "Plan for continuous optimization",
                "Establish regular review checkpoints",
                "Create contingency budgets"
            ],
            "low": [
                "Proceed with standard implementation",
                "Focus on change management excellence",
                "Plan for rapid scale-up",
                "Consider expansion opportunities"
            ]
        }
        
        return strategies.get(risk_rating, [])
    
    def _create_sensitivity_analysis(self, 
                                   hospital: HospitalProfile,
                                   ai_solution_type: str,
                                   parameters: ROIParameters) -> Dict:
        """Create sensitivity analysis for key parameters"""
        
        base_benefits = self._calculate_financial_benefits(
            hospital, self._calculate_improvement_scenarios(hospital, ai_solution_type, parameters),
            self._calculate_baseline_metrics(hospital)
        )["annual_total"]
        
        # Sensitivity scenarios
        scenarios = {
            "conservative": {
                "improvement_multiplier": 0.7,
                "description": "70% of projected improvements"
            },
            "base_case": {
                "improvement_multiplier": 1.0,
                "description": "Base case scenario"
            },
            "optimistic": {
                "improvement_multiplier": 1.3,
                "description": "130% of projected improvements"
            },
            "best_case": {
                "improvement_multiplier": 1.5,
                "description": "150% of projected improvements"
            }
        }
        
        sensitivity_results = {}
        
        for scenario_name, scenario in scenarios.items():
            adjusted_benefits = base_benefits * scenario["improvement_multiplier"]
            adjusted_costs = parameters.implementation_cost + parameters.annual_subscription_cost
            roi = (adjusted_benefits - parameters.annual_subscription_cost) / adjusted_costs
            
            sensitivity_results[scenario_name] = {
                "description": scenario["description"],
                "annual_benefits": adjusted_benefits,
                "roi_ratio": roi,
                "roi_percentage": roi * 100,
                "payback_months": adjusted_costs / (adjusted_benefits / 12)
            }
        
        return sensitivity_results
    
    def _estimate_quality_penalty(self, patient_satisfaction: float) -> float:
        """Estimate quality penalty based on patient satisfaction"""
        
        # Linear penalty for scores below 85
        if patient_satisfaction >= 85:
            return 0
        
        penalty_per_point = 25000  # $25K per point below 85
        return max(0, (85 - patient_satisfaction) * penalty_per_point)
    
    def _generate_roi_recommendations(self, roi_analysis: Dict, sensitivity: Dict) -> List[str]:
        """Generate ROI-based recommendations"""
        
        recommendations = []
        
        roi_ratio = roi_analysis["roi_metrics"]["roi_ratio"]
        payback_months = roi_analysis["roi_metrics"]["payback_period_months"]
        npv = roi_analysis["roi_metrics"]["net_present_value"]
        
        # ROI-based recommendations
        if roi_ratio > 3.0:
            recommendations.append("Strong ROI profile - proceed with full implementation")
        elif roi_ratio > 2.0:
            recommendations.append("Good ROI potential - consider pilot program first")
        elif roi_ratio > 1.0:
            recommendations.append("Positive but modest ROI - ensure operational excellence")
        else:
            recommendations.append("Poor ROI - consider alternative solutions or cost reduction")
        
        # Payback-based recommendations
        if payback_months < 12:
            recommendations.append("Fast payback - strong financial case for investment")
        elif payback_months < 24:
            recommendations.append("Reasonable payback period - align with strategic objectives")
        else:
            recommendations.append("Long payback - consider phased implementation")
        
        # NPV-based recommendations
        if npv > 500000:
            recommendations.append("Strong NPV - excellent long-term value creation")
        elif npv > 0:
            recommendations.append("Positive NPV - worthwhile investment")
        else:
            recommendations.append("Negative NPV - reconsider investment timing")
        
        return recommendations
    
    def generate_roi_report(self, roi_analysis: Dict, output_path: str) -> str:
        """Generate comprehensive ROI report"""
        
        report = f"""
# Healthcare AI Investment ROI Analysis Report

## Executive Summary
- **ROI Ratio**: {roi_analysis['roi_metrics']['roi_ratio']:.2f}x
- **Net Present Value**: ${roi_analysis['roi_metrics']['net_present_value']:,.0f}
- **Payback Period**: {roi_analysis['roi_metrics']['payback_period_months']:.1f} months
- **Internal Rate of Return**: {roi_analysis['roi_metrics']['internal_rate_of_return']:.1%}

## Financial Analysis

### Investment Summary
- **Total Investment**: ${roi_analysis['financial_summary']['total_investment']:,.0f}
- **Total Benefits (5-year)**: ${roi_analysis['financial_summary']['total_benefits']:,.0f}
- **Net Benefit**: ${roi_analysis['financial_summary']['net_benefit']:,.0f}
- **Benefit-Cost Ratio**: {roi_analysis['financial_summary']['benefit_cost_ratio']:.2f}x

### Risk Assessment
- **Risk Rating**: {roi_analysis['risk_assessment']['risk_rating'].upper()}
- **Risk Score**: {roi_analysis['risk_assessment']['risk_score']}/10

## Recommendations
{chr(10).join(['- ' + rec for rec in roi_analysis['recommendations']])}

## Cash Flow Analysis
"""
        
        for year_data in roi_analysis['cash_flow_analysis']['annual_cash_flows']:
            report += f"""
**Year {year_data['year']}:**
- Net Cash Flow: ${year_data['net_cash_flow']:,.0f}
- Cumulative Cash Flow: ${year_data['cumulative_cash_flow']:,.0f}
"""
        
        return report
    
    def create_roi_visualizations(self, roi_analysis: Dict, output_path: str) -> Dict:
        """Create ROI visualization charts"""
        
        visualizations = {}
        
        # Cash flow chart
        years = [year['year'] for year in roi_analysis['cash_flow_analysis']['annual_cash_flows']]
        net_cash_flows = [year['net_cash_flow'] for year in roi_analysis['cash_flow_analysis']['annual_cash_flows']]
        cumulative_cash_flows = [year['cumulative_cash_flow'] for year in roi_analysis['cash_flow_analysis']['annual_cash_flows']]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Annual cash flow chart
        ax1.bar(years, net_cash_flows, alpha=0.7, color='steelblue')
        ax1.set_title('Annual Net Cash Flow')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Cash Flow ($)')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative cash flow chart
        ax2.plot(years, cumulative_cash_flows, marker='o', linewidth=2, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_title('Cumulative Cash Flow')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Cash Flow ($)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/roi_cash_flow_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        visualizations['cash_flow_chart'] = f"{output_path}/roi_cash_flow_analysis.png"
        
        return visualizations

# Example usage and testing
if __name__ == "__main__":
    # Create sample hospital profile
    hospital = HospitalProfile(
        name="Memorial General Hospital",
        bed_count=400,
        annual_patients=15000,
        avg_length_of_stay=4.2,
        avg_cost_per_patient=8500,
        annual_revenue=180000000,
        readmission_rate=0.16,
        mortality_rate=0.06,
        patient_satisfaction=83,
        geographic_region="Midwest"
    )
    
    # Create ROI parameters
    parameters = ROIParameters(
        implementation_cost=150000,
        annual_subscription_cost=400000,
        expected_improvements={
            "mortality_reduction": 0.20,
            "readmission_reduction": 0.25,
            "length_of_stay_reduction": 0.15
        },
        measurement_period_months=36,
        discount_rate=0.08,
        risk_adjustment_factor=0.9
    )
    
    # Calculate ROI
    calculator = HealthcareROICalculator()
    roi_analysis = calculator.calculate_comprehensive_roi(
        hospital=hospital,
        ai_solution_type="comprehensive_ai_platform",
        parameters=parameters
    )
    
    print("Healthcare AI ROI Calculator Demo")
    print("=" * 50)
    print(f"Hospital: {hospital.name}")
    print(f"ROI Ratio: {roi_analysis['roi_metrics']['roi_metrics']['roi_ratio']:.2f}x")
    print(f"Payback Period: {roi_analysis['roi_metrics']['roi_metrics']['payback_period_months']:.1f} months")
    print(f"NPV: ${roi_analysis['roi_metrics']['roi_metrics']['net_present_value']:,.0f}")
