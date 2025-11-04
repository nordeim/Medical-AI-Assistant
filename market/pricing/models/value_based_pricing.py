"""
Value-Based Pricing Models for Healthcare AI
Outcome-tied pricing strategies and models
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

@dataclass
class ClinicalOutcome:
    """Clinical outcome definition for value-based pricing"""
    metric_name: str
    category: str  # mortality, readmission, length_of_stay, etc.
    measurement_unit: str
    baseline_value: float
    target_improvement: float
    measurement_frequency: str  # monthly, quarterly, annually
    risk_adjustment: float
    value_per_unit_improvement: float
    minimum_guarantee: float
    maximum_bonus: float

@dataclass
class ValueBasedContract:
    """Value-based contract structure"""
    contract_id: str
    customer_id: str
    base_fee: float
    outcome_tiers: List[Dict]
    measurement_period_months: int
    settlement_frequency: str  # quarterly, annually
    minimum_performance_threshold: float
    maximum_performance_cap: float
    risk_sharing_ratio: Tuple[float, float]  # (vendor_share, customer_share)
    payment_terms: str

@dataclass
class OutcomeMeasurement:
    """Outcome measurement and settlement"""
    measurement_id: str
    contract_id: str
    measurement_date: datetime
    outcome_metrics: Dict[str, float]
    calculated_improvement: float
    payout_amount: float
    measurement_status: str  # preliminary, verified, settled
    methodology_version: str

class ValueBasedPricingEngine:
    """
    Value-Based Pricing Engine for Healthcare AI
    Implements outcome-tied pricing strategies
    """
    
    def __init__(self):
        self.outcome_catalog = self._initialize_outcome_catalog()
        self.risk_models = self._initialize_risk_models()
        self.settlement_engine = SettlementEngine()
        
    def _initialize_outcome_catalog(self) -> Dict[str, ClinicalOutcome]:
        """Initialize clinical outcome catalog for value-based pricing"""
        
        return {
            "mortality_reduction_cardiac": ClinicalOutcome(
                metric_name="30-Day Mortality Rate - Cardiac Surgery",
                category="mortality",
                measurement_unit="deaths_per_100_cases",
                baseline_value=2.5,
                target_improvement=0.5,  # 0.5 deaths per 100 cases reduction
                measurement_frequency="quarterly",
                risk_adjustment=0.15,  # 15% risk adjustment
                value_per_unit_improvement=50000,  # $50K per death prevented
                minimum_guarantee=0.2,
                maximum_bonus=1.5
            ),
            "readmission_reduction_hf": ClinicalOutcome(
                metric_name="30-Day Readmission Rate - Heart Failure",
                category="readmission",
                measurement_unit="readmissions_per_100_cases",
                baseline_value=25.0,
                target_improvement=5.0,  # 5 readmissions per 100 cases reduction
                measurement_frequency="monthly",
                risk_adjustment=0.10,
                value_per_unit_improvement=30000,  # $30K per readmission avoided
                minimum_guarantee=0.3,
                maximum_bonus=1.2
            ),
            "length_of_stay_reduction": ClinicalOutcome(
                metric_name="Average Length of Stay Reduction",
                category="efficiency",
                measurement_unit="days",
                baseline_value=4.2,
                target_improvement=0.6,  # 0.6 days reduction
                measurement_frequency="monthly",
                risk_adjustment=0.05,
                value_per_unit_improvement=8000,  # $8K per day saved
                minimum_guarantee=0.4,
                maximum_bonus=1.3
            ),
            "diagnostic_accuracy_improvement": ClinicalOutcome(
                metric_name="Diagnostic Accuracy Score",
                category="quality",
                measurement_unit="accuracy_percentage",
                baseline_value=85.0,
                target_improvement=8.0,  # 8% accuracy improvement
                measurement_frequency="monthly",
                risk_adjustment=0.20,
                value_per_unit_improvement=12000,  # $12K per 1% accuracy improvement
                minimum_guarantee=0.25,
                maximum_bonus=1.4
            ),
            "patient_satisfaction_score": ClinicalOutcome(
                metric_name="Patient Satisfaction Score (HCAHPS)",
                category="patient_experience",
                measurement_unit="composite_score",
                baseline_value=82.0,
                target_improvement=6.0,  # 6 point improvement
                measurement_frequency="quarterly",
                risk_adjustment=0.08,
                value_per_unit_improvement=25000,  # $25K per point improvement
                minimum_guarantee=0.35,
                maximum_bonus=1.25
            ),
            "cost_per_case_reduction": ClinicalOutcome(
                metric_name="Cost Per Case Reduction",
                category="financial",
                measurement_unit="dollars",
                baseline_value=12500,
                target_improvement=1200,  # $1,200 reduction per case
                measurement_frequency="monthly",
                risk_adjustment=0.12,
                value_per_unit_improvement=40000,  # $40K per $100 reduction
                minimum_guarantee=0.30,
                maximum_bonus=1.35
            ),
            "quality_score_improvement": ClinicalOutcome(
                metric_name="CMS Quality Score Improvement",
                category="quality",
                measurement_unit="quality_points",
                baseline_value=75.0,
                target_improvement=12.0,  # 12 point improvement
                measurement_frequency="annually",
                risk_adjustment=0.25,
                value_per_unit_improvement=75000,  # $75K per point improvement
                minimum_guarantee=0.20,
                maximum_bonus=1.6
            )
        }
    
    def _initialize_risk_models(self) -> Dict:
        """Initialize risk models for outcome-based pricing"""
        
        return {
            "outcome_risk_factors": {
                "baseline_performance": {
                    "high_performer": {"risk_multiplier": 0.8, "description": "Low improvement potential"},
                    "average_performer": {"risk_multiplier": 1.0, "description": "Standard risk"},
                    "low_performer": {"risk_multiplier": 1.3, "description": "High improvement potential"}
                },
                "organization_readiness": {
                    "high_readiness": {"risk_multiplier": 0.9, "description": "Strong implementation capability"},
                    "medium_readiness": {"risk_multiplier": 1.0, "description": "Standard implementation"},
                    "low_readiness": {"risk_multiplier": 1.2, "description": "Higher implementation risk"}
                },
                "data_quality": {
                    "high_quality": {"risk_multiplier": 0.95, "description": "Reliable measurement data"},
                    "medium_quality": {"risk_multiplier": 1.0, "description": "Adequate measurement capability"},
                    "low_quality": {"risk_multiplier": 1.1, "description": "Measurement uncertainty"}
                }
            },
            "market_risk_factors": {
                "seasonal_variation": 0.05,  # 5% seasonal adjustment
                "regulatory_changes": 0.10,  # 10% regulatory risk
                "competitive_pressure": 0.08   # 8% competitive risk
            }
        }
    
    def design_value_based_contract(self,
                                  customer_profile: Dict,
                                  target_outcomes: List[str],
                                  contract_duration_months: int,
                                  base_fee: float) -> Dict:
        """
        Design value-based contract structure
        
        Args:
            customer_profile: Customer characteristics and readiness
            target_outcomes: List of outcome metrics to include
            contract_duration_months: Contract length in months
            base_fee: Base platform fee
        
        Returns:
            Contract design and pricing structure
        """
        
        # Analyze customer risk profile
        risk_profile = self._analyze_customer_risk(customer_profile)
        
        # Design outcome structure
        outcome_structure = self._design_outcome_structure(target_outcomes, risk_profile)
        
        # Calculate risk-adjusted pricing
        pricing_structure = self._calculate_risk_adjusted_pricing(
            base_fee, outcome_structure, risk_profile
        )
        
        # Generate contract terms
        contract_terms = self._generate_contract_terms(
            customer_profile, outcome_structure, pricing_structure, contract_duration_months
        )
        
        # Calculate expected value and risk
        value_analysis = self._calculate_expected_value_and_risk(
            outcome_structure, pricing_structure, risk_profile
        )
        
        return {
            "contract_design": {
                "base_fee": base_fee,
                "outcome_structure": outcome_structure,
                "pricing_structure": pricing_structure,
                "risk_profile": risk_profile,
                "contract_terms": contract_terms
            },
            "financial_projections": value_analysis,
            "implementation_plan": self._create_implementation_plan(outcome_structure),
            "measurement_framework": self._create_measurement_framework(outcome_structure),
            "risk_mitigation": self._design_risk_mitigation(risk_profile)
        }
    
    def _analyze_customer_risk(self, customer_profile: Dict) -> Dict:
        """Analyze customer risk profile for outcome-based pricing"""
        
        # Baseline performance analysis
        baseline_performance = customer_profile.get("baseline_performance", "average")
        if baseline_performance not in ["high", "average", "low"]:
            baseline_performance = "average"
        
        # Organization readiness assessment
        tech_readiness = customer_profile.get("technology_readiness", 0.7)
        outcome_focus = customer_profile.get("outcome_focus", 0.7)
        
        if tech_readiness > 0.8 and outcome_focus > 0.8:
            readiness_level = "high"
        elif tech_readiness > 0.6 and outcome_focus > 0.6:
            readiness_level = "medium"
        else:
            readiness_level = "low"
        
        # Data quality assessment
        data_quality = customer_profile.get("data_quality_score", 0.7)
        if data_quality > 0.85:
            data_quality_level = "high"
        elif data_quality > 0.7:
            data_quality_level = "medium"
        else:
            data_quality_level = "low"
        
        # Calculate composite risk score
        baseline_risk = self.risk_models["outcome_risk_factors"]["baseline_performance"][f"{baseline_performance}_performer"]["risk_multiplier"]
        readiness_risk = self.risk_models["outcome_risk_factors"]["organization_readiness"][f"{readiness_level}_readiness"]["risk_multiplier"]
        data_risk = self.risk_models["outcome_risk_factors"]["data_quality"][f"{data_quality}_quality"]["risk_multiplier"]
        
        composite_risk = (baseline_risk * 0.4 + readiness_risk * 0.35 + data_risk * 0.25)
        
        return {
            "baseline_performance": baseline_performance,
            "organization_readiness": readiness_level,
            "data_quality": data_quality_level,
            "composite_risk_score": composite_risk,
            "risk_tier": self._categorize_risk_tier(composite_risk),
            "risk_factors": {
                "baseline_risk": baseline_risk,
                "readiness_risk": readiness_risk,
                "data_risk": data_risk
            }
        }
    
    def _categorize_risk_tier(self, risk_score: float) -> str:
        """Categorize risk into tiers"""
        if risk_score <= 0.9:
            return "low_risk"
        elif risk_score <= 1.1:
            return "medium_risk"
        else:
            return "high_risk"
    
    def _design_outcome_structure(self, 
                                target_outcomes: List[str],
                                risk_profile: Dict) -> Dict:
        """Design outcome-based payment structure"""
        
        outcome_tiers = []
        
        for outcome_key in target_outcomes:
            if outcome_key not in self.outcome_catalog:
                continue
                
            outcome = self.outcome_catalog[outcome_key]
            risk_adjusted_improvement = outcome.target_improvement * risk_profile["composite_risk_score"]
            
            # Design tiered payment structure
            tier_1 = {
                "tier": "minimum",
                "improvement_threshold": risk_adjusted_improvement * outcome.minimum_guarantee,
                "payment_rate": 0.5,  # 50% of full value
                "description": "Minimum acceptable improvement"
            }
            
            tier_2 = {
                "tier": "target",
                "improvement_threshold": risk_adjusted_improvement,
                "payment_rate": 1.0,  # 100% of full value
                "description": "Target improvement level"
            }
            
            tier_3 = {
                "tier": "excellence",
                "improvement_threshold": risk_adjusted_improvement * outcome.maximum_bonus,
                "payment_rate": 1.2,  # 120% of full value for exceeding target
                "description": "Excellence performance bonus"
            }
            
            outcome_tiers.append({
                "outcome_metric": outcome.metric_name,
                "category": outcome.category,
                "baseline_value": outcome.baseline_value,
                "target_improvement": outcome.target_improvement,
                "risk_adjusted_improvement": risk_adjusted_improvement,
                "value_per_unit": outcome.value_per_unit_improvement,
                "measurement_frequency": outcome.measurement_frequency,
                "tiers": [tier_1, tier_2, tier_3],
                "maximum_potential_value": risk_adjusted_improvement * outcome.maximum_bonus * outcome.value_per_unit_improvement,
                "risk_adjustment_factor": risk_profile["composite_risk_score"]
            })
        
        return {
            "outcomes": outcome_tiers,
            "total_outcomes": len(outcome_tiers),
            "measurement_horizon_months": 12,
            "payment_structure": "tiered_outcome_based"
        }
    
    def _calculate_risk_adjusted_pricing(self,
                                       base_fee: float,
                                       outcome_structure: Dict,
                                       risk_profile: Dict) -> Dict:
        """Calculate risk-adjusted pricing structure"""
        
        # Calculate total potential outcome value
        total_potential_value = sum([
            outcome["maximum_potential_value"] 
            for outcome in outcome_structure["outcomes"]
        ])
        
        # Risk-adjusted value (expected value)
        expected_value = total_potential_value * 0.7  # Assume 70% probability of achieving targets
        risk_premium = total_potential_value * 0.1  # 10% risk premium
        expected_net_value = expected_value - risk_premium
        
        # Design payment structure
        payment_structure = {
            "base_fee": base_fee,
            "minimum_outcome_fee": expected_net_value * 0.6,  # 60% of expected value minimum
            "maximum_outcome_fee": total_potential_value,     # Up to full potential value
            "target_outcome_fee": expected_net_value,         # Target outcome fee
            "total_minimum_compensation": base_fee + (expected_net_value * 0.6),
            "total_maximum_compensation": base_fee + total_potential_value,
            "risk_adjustment": risk_premium,
            "payment_splits": {
                "base_component": base_fee / (base_fee + expected_net_value),
                "outcome_component": expected_net_value / (base_fee + expected_net_value)
            }
        }
        
        # Risk sharing terms
        risk_sharing = {
            "downside_protection": 0.8,  # 80% downside protection for vendor
            "upside_participation": 0.7,  # 70% upside sharing with customer
            "vendor_share": 0.6,  # Vendor bears 60% of risk
            "customer_share": 0.4  # Customer gets 40% of upside
        }
        
        return {
            "payment_structure": payment_structure,
            "risk_sharing": risk_sharing,
            "value_breakdown": {
                "base_platform": base_fee,
                "expected_outcomes": expected_net_value,
                "risk_premium": risk_premium,
                "total_expected_value": base_fee + expected_net_value
            }
        }
    
    def _generate_contract_terms(self,
                               customer_profile: Dict,
                               outcome_structure: Dict,
                               pricing_structure: Dict,
                               contract_duration_months: int) -> Dict:
        """Generate contract terms for value-based arrangement"""
        
        return {
            "contract_duration": {
                "initial_term_months": contract_duration_months,
                "renewal_options": "mutual_agreement",
                "early_termination": {
                    "notice_period_days": 90,
                    "termination_fee": "pro_rated_base_fee"
                }
            },
            "measurement_and_settlement": {
                "measurement_periods": "quarterly",
                "settlement_timing": "30_days_after_measurement",
                "data_verification": "third_party_audit_available",
                "dispute_resolution": "binding_arbitration"
            },
            "performance_standards": {
                "minimum_acceptance_threshold": "baseline_performance",
                "improvement_verification": "statistical_significance_required",
                "data_source_requirements": "verified_clinical_data"
            },
            "risk_management": {
                "force_majeure_clauses": "industry_standard",
                "regulatory_change_protection": "adjustment_mechanism",
                "technology_performance_standards": "uptime_and_accuracy_requirements"
            },
            "governance": {
                "joint_oversight_committee": "quarterly_reviews",
                "performance_reporting": "monthly_dashboard",
                "escalation_procedures": "defined_dispute_resolution"
            }
        }
    
    def _calculate_expected_value_and_risk(self,
                                         outcome_structure: Dict,
                                         pricing_structure: Dict,
                                         risk_profile: Dict) -> Dict:
        """Calculate expected value and risk analysis"""
        
        # Calculate expected outcomes by probability
        expected_outcomes = []
        total_expected_value = 0
        
        for outcome in outcome_structure["outcomes"]:
            # Assume probability distributions for outcomes
            min_probability = 0.6    # 60% chance of minimum tier
            target_probability = 0.4 # 40% chance of target tier
            excellence_probability = 0.2  # 20% chance of excellence tier
            
            # Risk-adjusted probabilities
            risk_factor = risk_profile["composite_risk_score"]
            min_prob_adj = min_probability * (2 - risk_factor)  # Adjust for risk
            target_prob_adj = target_probability * (2 - risk_factor)
            excellence_prob_adj = excellence_probability * (2 - risk_factor)
            
            # Calculate expected value
            min_value = outcome["tiers"][0]["improvement_threshold"] * outcome["value_per_unit"] * outcome["tiers"][0]["payment_rate"]
            target_value = outcome["tiers"][1]["improvement_threshold"] * outcome["value_per_unit"] * outcome["tiers"][1]["payment_rate"]
            excellence_value = outcome["tiers"][2]["improvement_threshold"] * outcome["value_per_unit"] * outcome["tiers"][2]["payment_rate"]
            
            expected_value = (min_value * min_prob_adj + 
                            target_value * target_prob_adj + 
                            excellence_value * excellence_prob_adj)
            
            total_expected_value += expected_value
            
            expected_outcomes.append({
                "outcome": outcome["outcome_metric"],
                "expected_value": expected_value,
                "probability_breakdown": {
                    "minimum": min_prob_adj,
                    "target": target_prob_adj,
                    "excellence": excellence_prob_adj
                }
            })
        
        # Risk analysis
        risk_analysis = {
            "value_at_risk_5_percent": total_expected_value * 0.3,  # 5th percentile risk
            "value_at_risk_95_percent": total_expected_value * 1.5,  # 95th percentile upside
            "expected_shortfall": total_expected_value * 0.15,  # Expected below-target performance
            "expected_excess": total_expected_value * 0.25,    # Expected above-target performance
            "risk_concentration": self._assess_risk_concentration(outcome_structure)
        }
        
        return {
            "expected_outcomes": expected_outcomes,
            "total_expected_outcome_value": total_expected_value,
            "base_platform_value": pricing_structure["value_breakdown"]["base_platform"],
            "total_expected_contract_value": (pricing_structure["value_breakdown"]["base_platform"] + 
                                            total_expected_value),
            "risk_analysis": risk_analysis,
            "confidence_intervals": {
                "conservative_estimate": total_expected_value * 0.7,
                "optimistic_estimate": total_expected_value * 1.3
            }
        }
    
    def _assess_risk_concentration(self, outcome_structure: Dict) -> str:
        """Assess concentration risk in outcome portfolio"""
        
        outcome_categories = [outcome["category"] for outcome in outcome_structure["outcomes"]]
        category_counts = {}
        for category in outcome_categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        max_concentration = max(category_counts.values()) / len(outcome_categories)
        
        if max_concentration > 0.5:
            return "high_concentration"
        elif max_concentration > 0.3:
            return "medium_concentration"
        else:
            return "low_concentration"
    
    def _create_implementation_plan(self, outcome_structure: Dict) -> Dict:
        """Create implementation plan for outcome-based contract"""
        
        phases = []
        
        # Phase 1: Setup and Baseline
        phases.append({
            "phase": "Baseline Establishment",
            "duration_weeks": 4,
            "objectives": [
                "Establish baseline measurements",
                "Implement data collection systems",
                "Validate measurement methodology",
                "Set up monitoring dashboards"
            ],
            "deliverables": [
                "Baseline measurement report",
                "Data collection validation",
                "Dashboard implementation",
                "Measurement protocol documentation"
            ]
        })
        
        # Phase 2: Initial Monitoring
        phases.append({
            "phase": "Initial Performance Monitoring",
            "duration_weeks": 12,
            "objectives": [
                "Begin continuous monitoring",
                "Identify early improvement trends",
                "Adjust implementation as needed",
                "Establish performance patterns"
            ],
            "deliverables": [
                "Monthly performance reports",
                "Trend analysis",
                "Implementation adjustments",
                "Risk mitigation actions"
            ]
        })
        
        # Phase 3: Full Optimization
        phases.append({
            "phase": "Full Performance Optimization",
            "duration_weeks": 24,
            "objectives": [
                "Achieve target performance levels",
                "Maximize outcome improvements",
                "Optimize resource allocation",
                "Prepare for settlement cycles"
            ],
            "deliverables": [
                "Quarterly outcome assessments",
                "Performance optimization report",
                "Settlement calculations",
                "Continuous improvement plan"
            ]
        })
        
        return {
            "implementation_phases": phases,
            "total_implementation_time_weeks": sum([phase["duration_weeks"] for phase in phases]),
            "critical_success_factors": [
                "Data quality and integrity",
                "Stakeholder engagement",
                "Technology performance",
                "Process optimization"
            ]
        }
    
    def _create_measurement_framework(self, outcome_structure: Dict) -> Dict:
        """Create measurement framework for outcomes"""
        
        measurement_framework = {
            "data_sources": {
                "clinical_data": "Electronic Health Records",
                "administrative_data": "Billing and claims systems",
                "patient_feedback": "Survey systems and HCAHPS",
                "operational_data": "Workflow and efficiency metrics"
            },
            "measurement_standards": {
                "data_validation": "Automated quality checks",
                "statistical_methods": "Industry-standard statistical analysis",
                "benchmark_comparisons": "Peer group comparisons",
                "trend_analysis": "Longitudinal performance tracking"
            },
            "reporting_schedule": {
                "monthly": ["length_of_stay_reduction", "cost_per_case_reduction", "diagnostic_accuracy_improvement"],
                "quarterly": ["mortality_reduction_cardiac", "readmission_reduction_hf", "patient_satisfaction_score"],
                "annually": ["quality_score_improvement"]
            },
            "quality_assurance": {
                "data_audit": "Monthly data quality reviews",
                "methodology_review": "Quarterly methodology assessment",
                "external_validation": "Annual third-party audit",
                "continuous_improvement": "Ongoing measurement refinement"
            }
        }
        
        return measurement_framework
    
    def _design_risk_mitigation(self, risk_profile: Dict) -> List[Dict]:
        """Design risk mitigation strategies"""
        
        mitigation_strategies = []
        
        # Risk-specific mitigations
        if risk_profile["risk_tier"] == "high_risk":
            mitigation_strategies.append({
                "risk_type": "high_overall_risk",
                "mitigation": "Implement phased performance targets",
                "description": "Start with lower targets and increase gradually"
            })
        
        if risk_profile["baseline_performance"] == "low":
            mitigation_strategies.append({
                "risk_type": "low_baseline_performance",
                "mitigation": "Enhanced implementation support",
                "description": "Provide additional training and change management resources"
            })
        
        if risk_profile["organization_readiness"] == "low":
            mitigation_strategies.append({
                "risk_type": "low_readiness",
                "mitigation": "Extended implementation timeline",
                "description": "Allow additional time for organizational readiness development"
            })
        
        # Standard mitigations
        mitigation_strategies.extend([
            {
                "risk_type": "measurement_uncertainty",
                "mitigation": "Robust measurement framework",
                "description": "Implement comprehensive data validation and quality assurance"
            },
            {
                "risk_type": "regulatory_changes",
                "mitigation": "Regulatory change clause",
                "description": "Include provisions for adjusting targets due to regulatory changes"
            },
            {
                "risk_type": "technology_performance",
                "mitigation": "Performance guarantees",
                "description": "Include technology uptime and accuracy requirements"
            }
        ])
        
        return mitigation_strategies
    
    def calculate_outcome_settlement(self,
                                   contract_id: str,
                                   measurement_data: Dict,
                                   measurement_date: datetime) -> Dict:
        """Calculate outcome-based payment settlement"""
        
        # This would process actual measurement data and calculate payments
        # For demo purposes, we'll simulate the settlement calculation
        
        # Retrieve contract structure (simplified)
        base_fee = 300000
        outcome_metrics = measurement_data.get("outcome_metrics", {})
        
        settlement = {
            "contract_id": contract_id,
            "measurement_date": measurement_date.isoformat(),
            "measurement_period": "Q1_2024",
            "settlement_amount": 0,
            "outcome_payments": [],
            "total_outcome_value": 0,
            "measurement_status": "preliminary"
        }
        
        # Calculate payments for each outcome (simplified)
        total_payment = 0
        
        # Example calculation for one outcome
        if "readmission_reduction_hf" in outcome_metrics:
            improvement = outcome_metrics["readmission_reduction_hf"]
            value_per_unit = 30000
            tier_payment = improvement * value_per_unit
            total_payment += tier_payment
            
            settlement["outcome_payments"].append({
                "outcome": "Heart Failure Readmission Reduction",
                "improvement": improvement,
                "payment": tier_payment,
                "tier": "target"
            })
        
        settlement["settlement_amount"] = total_payment
        settlement["total_outcome_value"] = total_payment
        settlement["base_platform_payment"] = base_fee / 4  # Quarterly payment
        settlement["total_payment"] = settlement["base_platform_payment"] + total_payment
        
        return settlement

class SettlementEngine:
    """Outcome measurement and settlement processing engine"""
    
    def __init__(self):
        self.measurement_standards = {}
        self.data_validators = {}
    
    def process_measurement(self, measurement_data: Dict) -> Dict:
        """Process and validate outcome measurement data"""
        
        validation_results = self._validate_measurement_data(measurement_data)
        
        if not validation_results["is_valid"]:
            return {
                "status": "rejected",
                "validation_errors": validation_results["errors"],
                "rejection_reason": "Data validation failed"
            }
        
        # Calculate improvement scores
        improvement_scores = self._calculate_improvement_scores(measurement_data)
        
        # Determine payment tier
        payment_tiers = self._determine_payment_tiers(improvement_scores)
        
        return {
            "status": "validated",
            "measurement_id": measurement_data.get("measurement_id"),
            "improvement_scores": improvement_scores,
            "payment_tiers": payment_tiers,
            "validation_passed": True
        }
    
    def _validate_measurement_data(self, data: Dict) -> Dict:
        """Validate measurement data quality and completeness"""
        
        errors = []
        
        # Check required fields
        required_fields = ["baseline_value", "current_value", "measurement_date", "data_source"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate data ranges
        if "baseline_value" in data and "current_value" in data:
            baseline = data["baseline_value"]
            current = data["current_value"]
            
            if baseline <= 0:
                errors.append("Baseline value must be positive")
            
            if current < 0:
                errors.append("Current value must be non-negative")
            
            if current > baseline * 2:  # Unreasonable improvement
                errors.append("Current value indicates unrealistic improvement")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors
        }
    
    def _calculate_improvement_scores(self, measurement_data: Dict) -> Dict:
        """Calculate improvement scores from measurement data"""
        
        baseline = measurement_data["baseline_value"]
        current = measurement_data["current_value"]
        
        if baseline == 0:
            return {"improvement_score": 0, "improvement_percentage": 0}
        
        # Calculate absolute and percentage improvement
        absolute_improvement = baseline - current  # Assuming lower is better
        improvement_percentage = (absolute_improvement / baseline) * 100
        
        # Calculate risk-adjusted score
        risk_adjustment = measurement_data.get("risk_adjustment", 0.1)
        adjusted_score = absolute_improvement * (1 - risk_adjustment)
        
        return {
            "baseline_value": baseline,
            "current_value": current,
            "absolute_improvement": absolute_improvement,
            "improvement_percentage": improvement_percentage,
            "risk_adjusted_score": adjusted_score,
            "improvement_direction": "positive" if absolute_improvement > 0 else "negative"
        }
    
    def _determine_payment_tiers(self, improvement_scores: Dict) -> Dict:
        """Determine payment tier based on improvement scores"""
        
        improvement_percentage = improvement_scores["improvement_percentage"]
        
        if improvement_percentage >= 15:  # Excellence tier
            tier = "excellence"
            payment_rate = 1.2
        elif improvement_percentage >= 8:  # Target tier
            tier = "target"
            payment_rate = 1.0
        elif improvement_percentage >= 3:  # Minimum tier
            tier = "minimum"
            payment_rate = 0.5
        else:  # Below minimum
            tier = "below_threshold"
            payment_rate = 0.0
        
        return {
            "tier": tier,
            "payment_rate": payment_rate,
            "qualifies_for_payment": payment_rate > 0,
            "tier_description": self._get_tier_description(tier)
        }
    
    def _get_tier_description(self, tier: str) -> str:
        """Get description for payment tier"""
        
        descriptions = {
            "excellence": "Exceptional performance exceeding all targets",
            "target": "Target performance achieved",
            "minimum": "Minimum acceptable performance",
            "below_threshold": "Performance below minimum threshold"
        }
        
        return descriptions.get(tier, "Unknown tier")

# Example usage and testing
if __name__ == "__main__":
    # Create sample customer profile
    customer_profile = {
        "organization_type": "academic_medical_center",
        "baseline_performance": "average",
        "technology_readiness": 0.85,
        "outcome_focus": 0.90,
        "data_quality_score": 0.88,
        "patient_volume": 25000,
        "current_mortality_rate": 2.8,
        "current_readmission_rate": 24.5,
        "current_length_of_stay": 4.5
    }
    
    # Design value-based contract
    pricing_engine = ValueBasedPricingEngine()
    
    target_outcomes = [
        "mortality_reduction_cardiac",
        "readmission_reduction_hf",
        "length_of_stay_reduction"
    ]
    
    contract_design = pricing_engine.design_value_based_contract(
        customer_profile=customer_profile,
        target_outcomes=target_outcomes,
        contract_duration_months=36,
        base_fee=350000
    )
    
    print("Value-Based Pricing Engine Demo")
    print("=" * 50)
    print(f"Customer Risk Tier: {contract_design['contract_design']['risk_profile']['risk_tier']}")
    print(f"Base Platform Fee: ${contract_design['contract_design']['pricing_structure']['value_breakdown']['base_platform']:,.0f}")
    print(f"Expected Outcome Value: ${contract_design['contract_design']['pricing_structure']['value_breakdown']['expected_outcomes']:,.0f}")
    print(f"Total Expected Contract Value: ${contract_design['financial_projections']['total_expected_contract_value']:,.0f}")
    print(f"Number of Outcome Metrics: {contract_design['contract_design']['outcome_structure']['total_outcomes']}")
    print(f"Implementation Timeline: {contract_design['implementation_plan']['total_implementation_time_weeks']} weeks")
