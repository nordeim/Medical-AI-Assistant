"""
ROI Calculators for Healthcare Clients

This module provides comprehensive ROI calculation tools specifically designed for healthcare AI implementations,
including clinical outcome measurement, cost-benefit analysis, and financial modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import math

class OutcomeCategory(Enum):
    """Categories of clinical and operational outcomes"""
    CLINICAL_QUALITY = "clinical_quality"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    FINANCIAL_IMPACT = "financial_impact"
    PATIENT_EXPERIENCE = "patient_experience"
    COMPLIANCE_RISK = "compliance_risk"

class TimeHorizon(Enum):
    """Time horizons for ROI calculations"""
    SHORT_TERM = 12    # 12 months
    MEDIUM_TERM = 36   # 3 years
    LONG_TERM = 60     # 5 years

@dataclass
class CostComponent:
    """Individual cost component"""
    cost_name: str
    category: str
    unit_cost: float
    quantity: int
    frequency: str  # "one_time", "monthly", "annual"
    implementation_month: int
    description: str

@dataclass
class BenefitComponent:
    """Individual benefit component"""
    benefit_name: str
    category: OutcomeCategory
    baseline_value: float
    improved_value: float
    unit_value: float
    frequency: str  # "monthly", "annual"
    measurement_period: int
    confidence_level: float  # 0-1 scale
    description: str

@dataclass
class ClinicalMetric:
    """Clinical quality metric"""
    metric_name: str
    category: OutcomeCategory
    current_performance: float
    target_performance: float
    measurement_unit: str
    patient_volume_impacted: int
    financial_value_per_unit: float
    regulatory_impact: bool
    quality_score_weight: float
    confidence_level: float  # 0-1 scale for measurement confidence

@dataclass
class OperationalMetric:
    """Operational efficiency metric"""
    metric_name: str
    current_efficiency: float
    target_efficiency: float
    measurement_unit: str
    baseline_cost: float
    cost_reduction_percentage: float
    time_savings_hours: float
    staff_hours_impacted: int
    confidence_level: float  # 0-1 scale for measurement confidence

@dataclass
class ROIAnalysis:
    """Complete ROI analysis results"""
    analysis_id: str
    customer_id: str
    analysis_date: datetime
    time_horizon_months: int
    total_investment: float
    total_benefits: float
    net_present_value: float
    internal_rate_of_return: float
    payback_period_months: float
    roi_percentage: float
    benefit_cost_ratio: float
    risk_adjusted_metrics: Dict[str, float]
    sensitivity_analysis: Dict[str, Dict]
    clinical_outcomes: List[ClinicalMetric]
    operational_outcomes: List[OperationalMetric]
    cost_breakdown: List[CostComponent]
    benefit_breakdown: List[BenefitComponent]

class HealthcareROICalculator:
    """ROI calculator specifically designed for healthcare AI implementations"""
    
    def __init__(self):
        self.roi_cache: Dict[str, ROIAnalysis] = {}
        self.benchmark_data = self._initialize_benchmark_data()
        
    def calculate_comprehensive_roi(self, customer_id: str, 
                                  clinical_metrics: List[ClinicalMetric],
                                  operational_metrics: List[OperationalMetric],
                                  cost_components: List[CostComponent],
                                  benefit_components: List[BenefitComponent],
                                  time_horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> ROIAnalysis:
        """Calculate comprehensive ROI for healthcare AI implementation"""
        
        analysis_id = f"roi_{customer_id}_{datetime.now().strftime('%Y%m%d')}"
        time_horizon_months = time_horizon.value
        
        # Calculate investment costs
        total_investment = self._calculate_total_investment(cost_components, time_horizon_months)
        
        # Calculate benefits over time
        total_benefits = self._calculate_total_benefits(
            clinical_metrics, operational_metrics, benefit_components, time_horizon_months
        )
        
        # Calculate NPV using discount rate
        discount_rate = 0.08  # 8% healthcare sector discount rate
        net_present_value = self._calculate_npv(total_benefits, total_investment, discount_rate, time_horizon_months)
        
        # Calculate IRR (simplified approximation)
        internal_rate_of_return = self._calculate_irr_approximation(total_benefits, total_investment, time_horizon_months)
        
        # Calculate payback period
        payback_period_months = self._calculate_payback_period(benefit_components, cost_components, time_horizon_months)
        
        # Calculate standard ROI percentage
        roi_percentage = ((total_benefits - total_investment) / total_investment) * 100
        
        # Calculate benefit-cost ratio
        benefit_cost_ratio = total_benefits / total_investment if total_investment > 0 else 0
        
        # Risk-adjusted metrics
        risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(
            total_benefits, total_investment, clinical_metrics, operational_metrics
        )
        
        # Sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(
            clinical_metrics, operational_metrics, cost_components, time_horizon_months
        )
        
        roi_analysis = ROIAnalysis(
            analysis_id=analysis_id,
            customer_id=customer_id,
            analysis_date=datetime.now(),
            time_horizon_months=time_horizon_months,
            total_investment=total_investment,
            total_benefits=total_benefits,
            net_present_value=net_present_value,
            internal_rate_of_return=internal_rate_of_return,
            payback_period_months=payback_period_months,
            roi_percentage=roi_percentage,
            benefit_cost_ratio=benefit_cost_ratio,
            risk_adjusted_metrics=risk_adjusted_metrics,
            sensitivity_analysis=sensitivity_analysis,
            clinical_outcomes=clinical_metrics,
            operational_outcomes=operational_metrics,
            cost_breakdown=cost_components,
            benefit_breakdown=benefit_components
        )
        
        self.roi_cache[analysis_id] = roi_analysis
        return roi_analysis
        
    def create_roi_template(self, customer_profile: Dict) -> Dict:
        """Create standardized ROI calculation template"""
        market_segment = customer_profile.get('market_segment', 'hospital_system')
        
        # Get benchmark data for segment
        benchmarks = self.benchmark_data.get(market_segment, {})
        
        # Create template based on segment characteristics
        template = {
            'clinical_metrics': self._create_clinical_metrics_template(market_segment, benchmarks),
            'operational_metrics': self._create_operational_metrics_template(market_segment, benchmarks),
            'cost_components': self._create_cost_components_template(customer_profile),
            'benefit_components': self._create_benefit_components_template(market_segment, benchmarks),
            'calculation_notes': self._get_segment_calculation_notes(market_segment)
        }
        
        return template
        
    def benchmark_roi_metrics(self, market_segment: str, 
                            peer_group_size: int = 10) -> Dict:
        """Benchmark ROI metrics against peer organizations"""
        if market_segment not in self.benchmark_data:
            return {'error': f'No benchmark data available for {market_segment}'}
            
        benchmarks = self.benchmark_data[market_segment]
        
        # Simulate peer group analysis
        peer_metrics = {}
        
        for metric_name, benchmark_value in benchmarks.items():
            # Generate peer group distribution around benchmark
            peer_values = np.random.normal(
                benchmark_value, 
                benchmark_value * 0.2,  # 20% standard deviation
                peer_group_size
            )
            
            peer_metrics[metric_name] = {
                'benchmark_value': benchmark_value,
                'peer_average': np.mean(peer_values),
                'peer_median': np.median(peer_values),
                'peer_range': [np.min(peer_values), np.max(peer_values)],
                'peer_quartiles': [np.percentile(peer_values, q) for q in [25, 75]],
                'percentile_rank': self._calculate_percentile_rank(benchmark_value, peer_values)
            }
            
        return {
            'market_segment': market_segment,
            'peer_group_size': peer_group_size,
            'benchmark_metrics': peer_metrics,
            'overall_assessment': self._generate_benchmark_assessment(peer_metrics)
        }
        
    def perform_sensitivity_analysis(self, roi_analysis: ROIAnalysis, 
                                   variables_to_test: List[str]) -> Dict:
        """Perform detailed sensitivity analysis on ROI components"""
        base_case = roi_analysis
        
        sensitivity_results = {}
        
        for variable in variables_to_test:
            if variable in ['clinical_performance', 'operational_efficiency', 'implementation_cost']:
                # Test different scenarios for each variable
                scenarios = {
                    'conservative': 0.8,  # 20% worse
                    'base_case': 1.0,     # No change
                    'optimistic': 1.2     # 20% better
                }
                
                variable_results = {}
                
                for scenario_name, multiplier in scenarios.items():
                    if variable == 'clinical_performance':
                        modified_metrics = self._adjust_clinical_metrics(
                            base_case.clinical_outcomes, multiplier
                        )
                        adjusted_benefits = self._calculate_benefits_from_metrics(
                            modified_metrics, base_case.operational_outcomes
                        )
                    elif variable == 'operational_efficiency':
                        modified_metrics = self._adjust_operational_metrics(
                            base_case.operational_outcomes, multiplier
                        )
                        adjusted_benefits = self._calculate_benefits_from_metrics(
                            base_case.clinical_outcomes, modified_metrics
                        )
                    elif variable == 'implementation_cost':
                        adjusted_costs = base_case.total_investment * multiplier
                        
                    # Recalculate ROI with modified variable
                    new_benefits = adjusted_benefits if 'adjusted_benefits' in locals() else base_case.total_benefits
                    new_costs = adjusted_costs if 'adjusted_costs' in locals() else base_case.total_investment
                    
                    new_roi = ((new_benefits - new_costs) / new_costs) * 100 if new_costs > 0 else 0
                    
                    variable_results[scenario_name] = {
                        'roi_percentage': new_roi,
                        'net_present_value': new_benefits - new_costs,
                        'payback_period': self._estimate_payback_from_benefits_costs(new_benefits, new_costs)
                    }
                    
                    # Clear temporary variables
                    if 'adjusted_benefits' in locals():
                        del adjusted_benefits
                    if 'adjusted_costs' in locals():
                        del adjusted_costs
                        
                sensitivity_results[variable] = variable_results
                
        return {
            'base_case_roi': base_case.roi_percentage,
            'sensitivity_analysis': sensitivity_results,
            'most_sensitive_variables': self._identify_most_sensitive_variables(sensitivity_results),
            'recommendations': self._generate_sensitivity_recommendations(sensitivity_results)
        }
        
    def generate_roi_report(self, roi_analysis: ROIAnalysis, 
                          include_charts: bool = False) -> Dict:
        """Generate comprehensive ROI report"""
        
        # Executive summary
        executive_summary = {
            'total_investment': roi_analysis.total_investment,
            'total_benefits': roi_analysis.total_benefits,
            'net_present_value': roi_analysis.net_present_value,
            'roi_percentage': roi_analysis.roi_percentage,
            'payback_period_months': roi_analysis.payback_period_months,
            'benefit_cost_ratio': roi_analysis.benefit_cost_ratio,
            'investment_grade': self._grade_investment(roi_analysis.roi_percentage),
            'payback_grade': self._grade_payback_period(roi_analysis.payback_period_months)
        }
        
        # Clinical outcomes summary
        clinical_summary = self._summarize_clinical_outcomes(roi_analysis.clinical_outcomes)
        
        # Operational outcomes summary
        operational_summary = self._summarize_operational_outcomes(roi_analysis.operational_outcomes)
        
        # Cost-benefit breakdown
        cost_benefit_breakdown = self._create_cost_benefit_breakdown(roi_analysis)
        
        # Risk assessment
        risk_assessment = self._assess_investment_risks(roi_analysis)
        
        # Recommendations
        recommendations = self._generate_roi_recommendations(roi_analysis)
        
        return {
            'report_metadata': {
                'analysis_id': roi_analysis.analysis_id,
                'customer_id': roi_analysis.customer_id,
                'analysis_date': roi_analysis.analysis_date.isoformat(),
                'time_horizon_months': roi_analysis.time_horizon_months,
                'report_version': '1.0'
            },
            'executive_summary': executive_summary,
            'clinical_outcomes': clinical_summary,
            'operational_outcomes': operational_summary,
            'financial_analysis': {
                'cost_benefit_breakdown': cost_benefit_breakdown,
                'cash_flow_timeline': self._create_cash_flow_timeline(roi_analysis),
                'key_financial_metrics': {
                    'npv': roi_analysis.net_present_value,
                    'irr': roi_analysis.internal_rate_of_return,
                    'roi': roi_analysis.roi_percentage,
                    'bc_ratio': roi_analysis.benefit_cost_ratio
                }
            },
            'risk_assessment': risk_assessment,
            'sensitivity_analysis': roi_analysis.sensitivity_analysis,
            'recommendations': recommendations,
            'implementation_roadmap': self._create_implementation_roadmap(roi_analysis),
            'monitoring_framework': self._create_monitoring_framework(roi_analysis)
        }
        
    def _calculate_total_investment(self, cost_components: List[CostComponent], 
                                  time_horizon_months: int) -> float:
        """Calculate total investment over time horizon"""
        total_investment = 0
        
        for cost in cost_components:
            if cost.frequency == 'one_time':
                total_investment += cost.unit_cost * cost.quantity
            elif cost.frequency == 'monthly':
                months_active = time_horizon_months - cost.implementation_month + 1
                if months_active > 0:
                    total_investment += cost.unit_cost * months_active
            elif cost.frequency == 'annual':
                years_active = (time_horizon_months - cost.implementation_month + 1) / 12
                if years_active > 0:
                    total_investment += cost.unit_cost * cost.quantity * math.ceil(years_active)
                    
        return total_investment
        
    def _calculate_total_benefits(self, clinical_metrics: List[ClinicalMetric],
                                operational_metrics: List[OperationalMetric],
                                benefit_components: List[BenefitComponent],
                                time_horizon_months: int) -> float:
        """Calculate total benefits over time horizon"""
        total_benefits = 0
        
        # Calculate clinical benefits
        for metric in clinical_metrics:
            annual_benefit = (
                (metric.target_performance - metric.current_performance) / 100 *
                metric.patient_volume_impacted *
                metric.financial_value_per_unit
            )
            total_benefits += annual_benefit * (time_horizon_months / 12)
            
        # Calculate operational benefits
        for metric in operational_metrics:
            annual_benefit = (
                metric.baseline_cost * 
                (metric.target_efficiency - metric.current_efficiency) / 100
            )
            total_benefits += annual_benefit * (time_horizon_months / 12)
            
        # Calculate other benefit components
        for benefit in benefit_components:
            if benefit.frequency == 'annual':
                total_benefits += benefit.unit_value * (time_horizon_months / 12) * benefit.confidence_level
            elif benefit.frequency == 'monthly':
                total_benefits += benefit.unit_value * time_horizon_months * benefit.confidence_level
                
        return total_benefits
        
    def _calculate_npv(self, total_benefits: float, total_investment: float,
                      discount_rate: float, time_horizon_months: int) -> float:
        """Calculate Net Present Value"""
        # Convert monthly discount rate to appropriate period
        annual_discount_rate = discount_rate
        periods = time_horizon_months / 12
        
        # Calculate NPV using standard formula
        npv = -total_investment
        
        # Assume benefits are realized evenly over the period
        annual_benefit = total_benefits / periods
        
        for year in range(1, int(periods) + 1):
            npv += annual_benefit / ((1 + annual_discount_rate) ** year)
            
        return npv
        
    def _calculate_irr_approximation(self, total_benefits: float, total_investment: float,
                                   time_horizon_months: int) -> float:
        """Calculate approximate Internal Rate of Return"""
        # Simplified IRR calculation using binary search
        net_cash_flow = total_benefits - total_investment
        years = time_horizon_months / 12
        
        if total_investment == 0 or net_cash_flow <= 0:
            return 0.0
            
        # Use approximation: IRR ≈ (net_cash_flow / investment)^(1/years) - 1
        irr = (net_cash_flow / total_investment) ** (1/years) - 1
        
        return irr * 100  # Return as percentage
        
    def _calculate_payback_period(self, benefit_components: List[BenefitComponent],
                                cost_components: List[CostComponent],
                                time_horizon_months: int) -> float:
        """Calculate payback period in months"""
        total_investment = sum(cost.unit_cost * cost.quantity for cost in cost_components)
        
        # Calculate monthly benefit
        monthly_benefit = 0
        for benefit in benefit_components:
            if benefit.frequency == 'annual':
                monthly_benefit += benefit.unit_value / 12 * benefit.confidence_level
            elif benefit.frequency == 'monthly':
                monthly_benefit += benefit.unit_value * benefit.confidence_level
                
        if monthly_benefit <= 0:
            return float('inf')  # Never pays back
            
        payback_months = total_investment / monthly_benefit
        
        return min(payback_months, time_horizon_months)  # Cap at time horizon
        
    def _calculate_risk_adjusted_metrics(self, total_benefits: float, total_investment: float,
                                       clinical_metrics: List[ClinicalMetric],
                                       operational_metrics: List[OperationalMetric]) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        # Calculate confidence-weighted benefits
        clinical_confidence = np.mean([m.confidence_level for m in clinical_metrics]) if clinical_metrics else 0.8
        operational_confidence = np.mean([m.confidence_level for m in operational_metrics]) if operational_metrics else 0.8
        
        overall_confidence = (clinical_confidence + operational_confidence) / 2
        
        risk_adjusted_benefits = total_benefits * overall_confidence
        risk_adjusted_roi = ((risk_adjusted_benefits - total_investment) / total_investment) * 100
        
        # Calculate downside scenarios (10th percentile)
        downside_multiplier = 0.7  # 30% reduction in benefits
        downside_roi = ((total_benefits * downside_multiplier - total_investment) / total_investment) * 100
        
        return {
            'risk_adjusted_benefits': risk_adjusted_benefits,
            'risk_adjusted_roi': risk_adjusted_roi,
            'downside_scenario_roi': downside_roi,
            'confidence_level': overall_confidence,
            'risk_adjusted_npv': risk_adjusted_benefits - total_investment
        }
        
    def _perform_sensitivity_analysis(self, clinical_metrics: List[ClinicalMetric],
                                    operational_metrics: List[OperationalMetric],
                                    cost_components: List[CostComponent],
                                    time_horizon_months: int) -> Dict[str, Dict]:
        """Perform sensitivity analysis on key variables"""
        # Base case
        base_benefits = self._calculate_total_benefits(clinical_metrics, operational_metrics, [], time_horizon_months)
        base_costs = self._calculate_total_investment(cost_components, time_horizon_months)
        base_roi = ((base_benefits - base_costs) / base_costs) * 100 if base_costs > 0 else 0
        
        sensitivity_results = {}
        
        # Test clinical performance variations
        for variation in [-20, -10, 10, 20]:  # percentage changes
            modified_metrics = []
            for metric in clinical_metrics:
                modified_metric = ClinicalMetric(
                    metric_name=metric.metric_name,
                    category=metric.category,
                    current_performance=metric.current_performance,
                    target_performance=metric.target_performance + (metric.target_performance - metric.current_performance) * variation / 100,
                    measurement_unit=metric.measurement_unit,
                    patient_volume_impacted=metric.patient_volume_impacted,
                    financial_value_per_unit=metric.financial_value_per_unit,
                    regulatory_impact=metric.regulatory_impact,
                    quality_score_weight=metric.quality_score_weight,
                    confidence_level=metric.confidence_level
                )
                modified_metrics.append(modified_metric)
                
            modified_benefits = self._calculate_total_benefits(modified_metrics, operational_metrics, [], time_horizon_months)
            modified_roi = ((modified_benefits - base_costs) / base_costs) * 100 if base_costs > 0 else 0
            
            sensitivity_results[f'clinical_performance_{variation}'] = {
                'roi': modified_roi,
                'benefit_change': ((modified_benefits - base_benefits) / base_benefits) * 100 if base_benefits > 0 else 0
            }
            
        return sensitivity_results
        
    def _initialize_benchmark_data(self) -> Dict:
        """Initialize benchmark data for different market segments"""
        return {
            'hospital_system': {
                'avg_roi_percentage': 285,
                'avg_payback_months': 18,
                'avg_npv_millions': 4.2,
                'clinical_improvement_percentage': 15,
                'operational_efficiency_gain': 25,
                'patient_satisfaction_improvement': 12,
                'readmission_reduction': 8
            },
            'academic_medical_center': {
                'avg_roi_percentage': 320,
                'avg_payback_months': 15,
                'avg_npv_millions': 5.8,
                'clinical_improvement_percentage': 18,
                'operational_efficiency_gain': 30,
                'research_productivity_gain': 40,
                'educational_effectiveness': 25
            },
            'clinic': {
                'avg_roi_percentage': 180,
                'avg_payback_months': 12,
                'avg_npv_millions': 0.8,
                'clinical_improvement_percentage': 20,
                'operational_efficiency_gain': 35,
                'patient_throughput_increase': 25,
                'provider_satisfaction': 30
            },
            'idn': {
                'avg_roi_percentage': 350,
                'avg_payback_months': 20,
                'avg_npv_millions': 8.5,
                'network_efficiency_gain': 45,
                'population_health_improvement': 20,
                'cost_reduction_percentage': 15,
                'quality_standardization': 60
            }
        }
        
    def _create_clinical_metrics_template(self, market_segment: str, 
                                        benchmarks: Dict) -> List[Dict]:
        """Create template for clinical metrics"""
        common_metrics = [
            {
                'metric_name': 'Diagnosis Accuracy',
                'category': 'clinical_quality',
                'current_performance': 85,
                'target_performance': 95,
                'measurement_unit': 'percentage',
                'patient_volume_impacted': 10000,
                'financial_value_per_unit': 50,
                'regulatory_impact': True,
                'quality_score_weight': 0.3
            },
            {
                'metric_name': 'Treatment Response Time',
                'category': 'operational_efficiency',
                'current_performance': 24,
                'target_performance': 12,
                'measurement_unit': 'hours',
                'patient_volume_impacted': 8000,
                'financial_value_per_unit': 25,
                'regulatory_impact': False,
                'quality_score_weight': 0.2
            }
        ]
        
        return common_metrics
        
    def _create_operational_metrics_template(self, market_segment: str, 
                                           benchmarks: Dict) -> List[Dict]:
        """Create template for operational metrics"""
        return [
            {
                'metric_name': 'Administrative Efficiency',
                'current_efficiency': 60,
                'target_efficiency': 85,
                'measurement_unit': 'percentage',
                'baseline_cost': 500000,
                'cost_reduction_percentage': 25,
                'time_savings_hours': 2000,
                'staff_hours_impacted': 50
            },
            {
                'metric_name': 'Resource Utilization',
                'current_efficiency': 70,
                'target_efficiency': 90,
                'measurement_unit': 'percentage',
                'baseline_cost': 1200000,
                'cost_reduction_percentage': 20,
                'time_savings_hours': 1500,
                'staff_hours_impacted': 75
            }
        ]
        
    def _create_cost_components_template(self, customer_profile: Dict) -> List[Dict]:
        """Create template for cost components"""
        return [
            {
                'cost_name': 'Software License',
                'category': 'recurring',
                'unit_cost': 120000,
                'quantity': 1,
                'frequency': 'annual',
                'implementation_month': 1,
                'description': 'Annual software licensing fees'
            },
            {
                'cost_name': 'Implementation Services',
                'category': 'one_time',
                'unit_cost': 50000,
                'quantity': 1,
                'frequency': 'one_time',
                'implementation_month': 1,
                'description': 'Initial setup and configuration'
            },
            {
                'cost_name': 'Training and Change Management',
                'category': 'one_time',
                'unit_cost': 25000,
                'quantity': 1,
                'frequency': 'one_time',
                'implementation_month': 2,
                'description': 'Staff training and adoption support'
            },
            {
                'cost_name': 'Ongoing Support',
                'category': 'recurring',
                'unit_cost': 15000,
                'quantity': 1,
                'frequency': 'monthly',
                'implementation_month': 3,
                'description': 'Monthly support and maintenance'
            }
        ]
        
    def _create_benefit_components_template(self, market_segment: str, 
                                          benchmarks: Dict) -> List[Dict]:
        """Create template for benefit components"""
        return [
            {
                'benefit_name': 'Reduced Readmissions',
                'category': 'financial_impact',
                'baseline_value': 1000,
                'improved_value': 920,
                'unit_value': 15000,
                'frequency': 'annual',
                'measurement_period': 12,
                'confidence_level': 0.85,
                'description': 'Annual savings from reduced readmissions'
            },
            {
                'benefit_name': 'Improved Patient Throughput',
                'category': 'operational_efficiency',
                'baseline_value': 100,
                'improved_value': 125,
                'unit_value': 200,
                'frequency': 'monthly',
                'measurement_period': 1,
                'confidence_level': 0.9,
                'description': 'Monthly revenue from increased patient volume'
            },
            {
                'benefit_name': 'Quality Score Improvement',
                'category': 'clinical_quality',
                'baseline_value': 85,
                'improved_value': 92,
                'unit_value': 50000,
                'frequency': 'annual',
                'measurement_period': 12,
                'confidence_level': 0.8,
                'description': 'Annual value from quality score improvements'
            }
        ]
        
    def _get_segment_calculation_notes(self, market_segment: str) -> List[str]:
        """Get calculation notes specific to market segment"""
        notes = {
            'hospital_system': [
                'Include quality bonus implications in financial calculations',
                'Account for regulatory compliance cost savings',
                'Consider patient satisfaction impact on revenue',
                'Include efficiency gains in multiple departments'
            ],
            'academic_medical_center': [
                'Include research productivity benefits',
                'Account for educational value and training efficiency',
                'Consider grant and funding opportunities',
                'Include reputation and ranking benefits'
            ],
            'clinic': [
                'Focus on provider efficiency and satisfaction',
                'Include patient satisfaction impact on retention',
                'Account for competitive advantage benefits',
                'Consider scalability for multi-location growth'
            ],
            'idn': [
                'Include network-wide standardization benefits',
                'Account for population health management value',
                'Consider enterprise-scale efficiency gains',
                'Include governance and oversight improvements'
            ]
        }
        
        return notes.get(market_segment, [
            'Use standard healthcare ROI calculation methodology',
            'Include both direct and indirect benefits',
            'Account for implementation timeline and ramp-up period'
        ])
        
    def _adjust_clinical_metrics(self, metrics: List[ClinicalMetric], multiplier: float) -> List[ClinicalMetric]:
        """Adjust clinical metrics by multiplier"""
        adjusted = []
        for metric in metrics:
            adjusted_metric = ClinicalMetric(
                metric_name=metric.metric_name,
                category=metric.category,
                current_performance=metric.current_performance,
                target_performance=metric.target_performance,
                measurement_unit=metric.measurement_unit,
                patient_volume_impacted=int(metric.patient_volume_impacted * multiplier),
                financial_value_per_unit=metric.financial_value_per_unit,
                regulatory_impact=metric.regulatory_impact,
                quality_score_weight=metric.quality_score_weight,
                confidence_level=metric.confidence_level
            )
            adjusted.append(adjusted_metric)
        return adjusted
        
    def _adjust_operational_metrics(self, metrics: List[OperationalMetric], multiplier: float) -> List[OperationalMetric]:
        """Adjust operational metrics by multiplier"""
        adjusted = []
        for metric in metrics:
            adjusted_metric = OperationalMetric(
                metric_name=metric.metric_name,
                current_efficiency=metric.current_efficiency,
                target_efficiency=min(metric.target_efficiency * multiplier, 100),  # Cap at 100%
                measurement_unit=metric.measurement_unit,
                baseline_cost=metric.baseline_cost,
                cost_reduction_percentage=min(metric.cost_reduction_percentage * multiplier, 50),  # Cap at 50%
                time_savings_hours=metric.time_savings_hours * multiplier,
                staff_hours_impacted=int(metric.staff_hours_impacted * multiplier),
                confidence_level=metric.confidence_level
            )
            adjusted.append(adjusted_metric)
        return adjusted
        
    def _calculate_benefits_from_metrics(self, clinical_metrics: List[ClinicalMetric],
                                       operational_metrics: List[OperationalMetric]) -> float:
        """Calculate benefits from clinical and operational metrics"""
        total_benefits = 0
        
        for metric in clinical_metrics:
            annual_benefit = (
                (metric.target_performance - metric.current_performance) / 100 *
                metric.patient_volume_impacted *
                metric.financial_value_per_unit
            )
            total_benefits += annual_benefit
            
        for metric in operational_metrics:
            annual_benefit = (
                metric.baseline_cost * 
                (metric.target_efficiency - metric.current_efficiency) / 100
            )
            total_benefits += annual_benefit
            
        return total_benefits
        
    def _estimate_payback_from_benefits_costs(self, benefits: float, costs: float) -> float:
        """Estimate payback period from benefits and costs"""
        if benefits <= 0:
            return float('inf')
        monthly_benefit = benefits / 36  # Assume 3-year horizon
        return costs / monthly_benefit if monthly_benefit > 0 else float('inf')
        
    def _identify_most_sensitive_variables(self, sensitivity_results: Dict) -> List[str]:
        """Identify most sensitive variables in the analysis"""
        sensitivity_scores = {}
        
        for variable, scenarios in sensitivity_results.items():
            if 'base_case' in scenarios and 'conservative' in scenarios:
                base_roi = scenarios['base_case']['roi_percentage']
                conservative_roi = scenarios['conservative']['roi_percentage']
                sensitivity_scores[variable] = abs(base_roi - conservative_roi)
                
        # Sort by sensitivity score
        sorted_sensitivity = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [var[0] for var in sorted_sensitivity[:3]]  # Top 3 most sensitive
        
    def _generate_sensitivity_recommendations(self, sensitivity_results: Dict) -> List[str]:
        """Generate recommendations based on sensitivity analysis"""
        recommendations = []
        
        # Find variables with high sensitivity
        high_sensitivity_threshold = 50  # ROI percentage points
        
        for variable, scenarios in sensitivity_results.items():
            if 'conservative' in scenarios and 'optimistic' in scenarios:
                roi_range = scenarios['optimistic']['roi_percentage'] - scenarios['conservative']['roi_percentage']
                if roi_range > high_sensitivity_threshold:
                    recommendations.append(
                        f"Consider risk mitigation strategies for {variable} - high impact on ROI"
                    )
                    
        if recommendations:
            recommendations.append("Focus on variables with highest sensitivity in implementation planning")
            
        return recommendations
        
    def _calculate_percentile_rank(self, value: float, peer_values: np.ndarray) -> float:
        """Calculate percentile rank of a value within peer group"""
        return (np.sum(peer_values <= value) / len(peer_values)) * 100
        
    def _generate_benchmark_assessment(self, peer_metrics: Dict) -> Dict:
        """Generate overall benchmark assessment"""
        above_average_count = 0
        total_metrics = len(peer_metrics)
        
        for metric_data in peer_metrics.values():
            if metric_data['benchmark_value'] > metric_data['peer_average']:
                above_average_count += 1
                
        performance_level = above_average_count / total_metrics
        
        if performance_level >= 0.7:
            return {'level': 'excellent', 'score': performance_level}
        elif performance_level >= 0.5:
            return {'level': 'good', 'score': performance_level}
        elif performance_level >= 0.3:
            return {'level': 'average', 'score': performance_level}
        else:
            return {'level': 'below_average', 'score': performance_level}
            
    def _grade_investment(self, roi_percentage: float) -> str:
        """Grade investment based on ROI percentage"""
        if roi_percentage >= 300:
            return 'A'
        elif roi_percentage >= 200:
            return 'B'
        elif roi_percentage >= 100:
            return 'C'
        elif roi_percentage >= 50:
            return 'D'
        else:
            return 'F'
            
    def _grade_payback_period(self, payback_months: float) -> str:
        """Grade investment based on payback period"""
        if payback_months <= 12:
            return 'A'
        elif payback_months <= 18:
            return 'B'
        elif payback_months <= 24:
            return 'C'
        elif payback_months <= 36:
            return 'D'
        else:
            return 'F'
            
    def _summarize_clinical_outcomes(self, clinical_metrics: List[ClinicalMetric]) -> Dict:
        """Summarize clinical outcomes"""
        if not clinical_metrics:
            return {'message': 'No clinical metrics provided'}
            
        total_improvement = sum(
            (m.target_performance - m.current_performance) / m.current_performance * 100
            for m in clinical_metrics
        )
        
        avg_improvement = total_improvement / len(clinical_metrics)
        
        return {
            'total_metrics': len(clinical_metrics),
            'average_improvement_percentage': avg_improvement,
            'top_performing_metric': max(clinical_metrics, key=lambda m: m.target_performance - m.current_performance).metric_name,
            'metrics_breakdown': [
                {
                    'metric': m.metric_name,
                    'improvement': m.target_performance - m.current_performance,
                    'percentage_improvement': (m.target_performance - m.current_performance) / m.current_performance * 100
                }
                for m in clinical_metrics
            ]
        }
        
    def _summarize_operational_outcomes(self, operational_metrics: List[OperationalMetric]) -> Dict:
        """Summarize operational outcomes"""
        if not operational_metrics:
            return {'message': 'No operational metrics provided'}
            
        total_efficiency_gain = sum(
            m.target_efficiency - m.current_efficiency for m in operational_metrics
        )
        
        avg_efficiency_gain = total_efficiency_gain / len(operational_metrics)
        
        total_cost_savings = sum(
            m.baseline_cost * m.cost_reduction_percentage / 100 for m in operational_metrics
        )
        
        return {
            'total_metrics': len(operational_metrics),
            'average_efficiency_gain': avg_efficiency_gain,
            'total_annual_cost_savings': total_cost_savings,
            'top_improvement_metric': max(operational_metrics, key=lambda m: m.target_efficiency - m.current_efficiency).metric_name,
            'metrics_breakdown': [
                {
                    'metric': m.metric_name,
                    'efficiency_gain': m.target_efficiency - m.current_efficiency,
                    'cost_savings': m.baseline_cost * m.cost_reduction_percentage / 100
                }
                for m in operational_metrics
            ]
        }
        
    def _create_cost_benefit_breakdown(self, roi_analysis: ROIAnalysis) -> Dict:
        """Create detailed cost-benefit breakdown"""
        total_costs = sum(cost.unit_cost * cost.quantity for cost in roi_analysis.cost_breakdown)
        total_benefits = sum(
            benefit.unit_value * benefit.confidence_level for benefit in roi_analysis.benefit_breakdown
        )
        
        return {
            'total_costs': total_costs,
            'total_benefits': total_benefits,
            'cost_breakdown_by_category': self._group_costs_by_category(roi_analysis.cost_breakdown),
            'benefit_breakdown_by_category': self._group_benefits_by_category(roi_analysis.benefit_breakdown)
        }
        
    def _group_costs_by_category(self, cost_components: List[CostComponent]) -> Dict:
        """Group costs by category"""
        categories = {}
        for cost in cost_components:
            if cost.category not in categories:
                categories[cost.category] = 0
            categories[cost.category] += cost.unit_cost * cost.quantity
        return categories
        
    def _group_benefits_by_category(self, benefit_components: List[BenefitComponent]) -> Dict:
        """Group benefits by category"""
        categories = {}
        for benefit in benefit_components:
            if benefit.category.value not in categories:
                categories[benefit.category.value] = 0
            categories[benefit.category.value] += benefit.unit_value * benefit.confidence_level
        return categories
        
    def _create_cash_flow_timeline(self, roi_analysis: ROIAnalysis) -> List[Dict]:
        """Create monthly cash flow timeline"""
        timeline = []
        
        for month in range(roi_analysis.time_horizon_months + 1):
            month_data = {'month': month, 'cash_inflow': 0, 'cash_outflow': 0, 'net_cash_flow': 0}
            
            # Calculate costs for this month
            for cost in roi_analysis.cost_breakdown:
                if cost.frequency == 'one_time' and cost.implementation_month == month:
                    month_data['cash_outflow'] += cost.unit_cost * cost.quantity
                elif cost.frequency == 'monthly' and month >= cost.implementation_month:
                    month_data['cash_outflow'] += cost.unit_cost
                elif cost.frequency == 'annual' and month % 12 == 0 and month >= cost.implementation_month:
                    month_data['cash_outflow'] += cost.unit_cost * cost.quantity
                    
            # Calculate benefits for this month
            for benefit in roi_analysis.benefit_breakdown:
                if month >= 1:  # Benefits start from month 1
                    if benefit.frequency == 'monthly':
                        month_data['cash_inflow'] += benefit.unit_value * benefit.confidence_level
                    elif benefit.frequency == 'annual' and month % 12 == 0:
                        month_data['cash_inflow'] += benefit.unit_value * benefit.confidence_level
                        
            month_data['net_cash_flow'] = month_data['cash_inflow'] - month_data['cash_outflow']
            timeline.append(month_data)
            
        return timeline
        
    def _assess_investment_risks(self, roi_analysis: ROIAnalysis) -> Dict:
        """Assess investment risks"""
        risks = []
        
        # ROI-related risks
        if roi_analysis.roi_percentage < 100:
            risks.append({
                'risk_type': 'low_roi',
                'severity': 'high',
                'description': 'ROI below 100% indicates potential value risk'
            })
            
        # Payback period risks
        if roi_analysis.payback_period_months > 24:
            risks.append({
                'risk_type': 'long_payback',
                'severity': 'medium',
                'description': 'Extended payback period increases investment risk'
            })
            
        # Confidence-related risks
        avg_confidence = np.mean([
            m.confidence_level for m in roi_analysis.clinical_outcomes + roi_analysis.operational_outcomes
        ]) if roi_analysis.clinical_outcomes or roi_analysis.operational_outcomes else 0.8
        
        if avg_confidence < 0.7:
            risks.append({
                'risk_type': 'low_confidence',
                'severity': 'medium',
                'description': 'Low confidence in outcome achievement'
            })
            
        return {
            'identified_risks': risks,
            'overall_risk_level': 'high' if len(risks) >= 2 else 'medium' if len(risks) == 1 else 'low',
            'mitigation_strategies': self._generate_risk_mitigation_strategies(risks)
        }
        
    def _generate_risk_mitigation_strategies(self, risks: List[Dict]) -> List[str]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        for risk in risks:
            if risk['risk_type'] == 'low_roi':
                strategies.append('Focus on high-impact use cases to improve ROI')
            elif risk['risk_type'] == 'long_payback':
                strategies.append('Consider phased implementation to improve cash flow')
            elif risk['risk_type'] == 'low_confidence':
                strategies.append('Implement pilot program to validate assumptions')
                
        return strategies
        
    def _generate_roi_recommendations(self, roi_analysis: ROIAnalysis) -> List[Dict]:
        """Generate recommendations based on ROI analysis"""
        recommendations = []
        
        # ROI-based recommendations
        if roi_analysis.roi_percentage < 150:
            recommendations.append({
                'priority': 'high',
                'category': 'roi_optimization',
                'recommendation': 'Consider additional use cases or cost optimization to improve ROI',
                'expected_impact': 'medium'
            })
            
        # Payback-based recommendations
        if roi_analysis.payback_period_months > 18:
            recommendations.append({
                'priority': 'medium',
                'category': 'payback_optimization',
                'recommendation': 'Implement quick wins to accelerate payback period',
                'expected_impact': 'high'
            })
            
        # Implementation recommendations
        recommendations.append({
            'priority': 'high',
            'category': 'implementation',
            'recommendation': 'Establish clear success metrics and monitoring framework',
            'expected_impact': 'high'
        })
        
        return recommendations
        
    def _create_implementation_roadmap(self, roi_analysis: ROIAnalysis) -> Dict:
        """Create implementation roadmap based on ROI analysis"""
        return {
            'phase_1_quick_wins': {
                'duration_months': 3,
                'focus': 'High-impact, low-risk implementations',
                'expected_benefits': roi_analysis.total_benefits * 0.3,
                'key_activities': ['Initial setup', 'Basic training', 'Pilot deployment']
            },
            'phase_2_scale': {
                'duration_months': 6,
                'focus': 'Full deployment and optimization',
                'expected_benefits': roi_analysis.total_benefits * 0.7,
                'key_activities': ['Complete rollout', 'Advanced training', 'Process optimization']
            },
            'phase_3_optimize': {
                'duration_months': 3,
                'focus': 'Continuous improvement and expansion',
                'expected_benefits': roi_analysis.total_benefits * 0.2,
                'key_activities': ['Performance tuning', 'Advanced features', 'ROI validation']
            }
        }
        
    def _create_monitoring_framework(self, roi_analysis: ROIAnalysis) -> Dict:
        """Create monitoring framework for ROI tracking"""
        return {
            'key_performance_indicators': [
                {
                    'kpi': 'Clinical Outcome Improvement',
                    'target': f"{np.mean([m.target_performance - m.current_performance for m in roi_analysis.clinical_outcomes]):.1f}%",
                    'measurement_frequency': 'monthly'
                },
                {
                    'kpi': 'Operational Efficiency Gain',
                    'target': f"{np.mean([m.target_efficiency - m.current_efficiency for m in roi_analysis.operational_outcomes]):.1f}%",
                    'measurement_frequency': 'monthly'
                },
                {
                    'kpi': 'Cumulative Benefit Realization',
                    'target': f"${roi_analysis.total_benefits:,.0f}",
                    'measurement_frequency': 'quarterly'
                }
            ],
            'reporting_schedule': 'monthly',
            'review_points': ['3_months', '6_months', '12_months', '24_months', '36_months'],
            'success_criteria': {
                'minimum_roi': 100,
                'target_roi': 200,
                'maximum_payback_months': 24
            }
        }


if __name__ == "__main__":
    # Example usage
    calculator = HealthcareROICalculator()
    
    # Create sample clinical metrics
    clinical_metrics = [
        ClinicalMetric(
            metric_name="Readmission Rate",
            category=OutcomeCategory.CLINICAL_QUALITY,
            current_performance=15.0,
            target_performance=12.0,
            measurement_unit="percentage",
            patient_volume_impacted=10000,
            financial_value_per_unit=15000,
            regulatory_impact=True,
            quality_score_weight=0.3,
            confidence_level=0.85
        )
    ]
    
    # Create sample operational metrics
    operational_metrics = [
        OperationalMetric(
            metric_name="Administrative Efficiency",
            current_efficiency=60.0,
            target_efficiency=80.0,
            measurement_unit="percentage",
            baseline_cost=500000,
            cost_reduction_percentage=25.0,
            time_savings_hours=2000,
            staff_hours_impacted=50,
            confidence_level=0.80
        )
    ]
    
    # Create sample cost components
    cost_components = [
        CostComponent(
            cost_name="Software License",
            category="recurring",
            unit_cost=120000,
            quantity=1,
            frequency="annual",
            implementation_month=1,
            description="Annual software licensing fees"
        )
    ]
    
    # Create sample benefit components
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
        )
    ]
    
    # Calculate ROI
    roi_analysis = calculator.calculate_comprehensive_roi(
        customer_id="test_customer",
        clinical_metrics=clinical_metrics,
        operational_metrics=operational_metrics,
        cost_components=cost_components,
        benefit_components=benefit_components,
        time_horizon=TimeHorizon.MEDIUM_TERM
    )
    
    print("ROI Analysis Results:")
    print(f"ROI: {roi_analysis.roi_percentage:.1f}%")
    print(f"Payback Period: {roi_analysis.payback_period_months:.1f} months")
    print(f"NPV: ${roi_analysis.net_present_value:,.2f}")
    
    # Generate report
    report = calculator.generate_roi_report(roi_analysis)
    print("\nROI Report Generated Successfully")
    print(json.dumps(report['executive_summary'], indent=2, default=str))