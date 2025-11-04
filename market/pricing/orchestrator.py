"""
Healthcare AI Pricing Framework - Main Integration Module
Unified interface for all pricing and revenue optimization components
"""

import os
import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Import all framework components
from frameworks.main_pricing_framework import HealthcarePricingFramework, RevenueOperationsEngine
from calculators.healthcare_roi_calculator import HealthcareROICalculator, HospitalProfile, ROIParameters
from analysis.customer_value_analyzer import CustomerValueAnalyzer, CustomerProfile
from attribution.revenue_attribution_system import RevenueAttributionEngine
from models.value_based_pricing import ValueBasedPricingEngine
from models.subscription_licensing import SubscriptionPricingEngine, EnterpriseLicenseEngine
from dashboard.healthcare_pricing_dashboard import HealthcarePricingDashboard

class HealthcarePricingOrchestrator:
    """
    Healthcare AI Pricing Framework Orchestrator
    Main entry point for all pricing and revenue optimization functionality
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the pricing framework orchestrator
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_configuration(config_path)
        self.framework_initialized = False
        
        # Initialize all components
        self._initialize_framework()
        
    def _load_configuration(self, config_path: str = None) -> Dict:
        """Load framework configuration"""
        
        default_config = {
            "framework_settings": {
                "default_currency": "USD",
                "default_discount_rate": 0.08,
                "default_measurement_period_months": 36,
                "default_implementation_risk_factor": 0.9
            },
            "pricing_models": {
                "subscription": {
                    "default_tier": "professional",
                    "annual_discount": 0.08,
                    "minimum_contract_months": 12
                },
                "enterprise": {
                    "minimum_facilities": 3,
                    "default_volume_discount": 0.20,
                    "support_level": "dedicated"
                },
                "value_based": {
                    "minimum_outcome_threshold": 0.15,
                    "default_risk_sharing": [0.6, 0.4],
                    "measurement_frequency": "quarterly"
                }
            },
            "market_segments": {
                "large_hospital": {
                    "typical_bed_count": 500,
                    "typical_patient_volume": 20000,
                    "decision_timeline_months": 6
                },
                "academic_medical_center": {
                    "typical_bed_count": 700,
                    "typical_patient_volume": 30000,
                    "decision_timeline_months": 8
                },
                "community_clinic": {
                    "typical_bed_count": 50,
                    "typical_patient_volume": 5000,
                    "decision_timeline_months": 3
                }
            },
            "roi_benchmarks": {
                "target_roi_ratio": 2.5,
                "acceptable_roi_ratio": 1.5,
                "target_payback_months": 24,
                "maximum_payback_months": 36
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    user_config = yaml.safe_load(f)
                else:
                    user_config = json.load(f)
            
            # Merge with default config
            default_config = self._deep_merge(default_config, user_config)
        
        return default_config
    
    def _deep_merge(self, default: Dict, user: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _initialize_framework(self):
        """Initialize all framework components"""
        
        if not self.framework_initialized:
            # Core framework components
            self.pricing_framework = HealthcarePricingFramework()
            self.roi_calculator = HealthcareROICalculator()
            self.value_analyzer = CustomerValueAnalyzer()
            self.attribution_engine = RevenueAttributionEngine()
            self.value_based_engine = ValueBasedPricingEngine()
            self.subscription_engine = SubscriptionPricingEngine()
            self.enterprise_engine = EnterpriseLicenseEngine()
            self.revenue_operations = RevenueOperationsEngine()
            self.dashboard = HealthcarePricingDashboard()
            
            self.framework_initialized = True
    
    def analyze_customer_pricing(self, 
                                customer_data: Dict,
                                analysis_options: Dict = None) -> Dict:
        """
        Comprehensive customer pricing analysis
        
        Args:
            customer_data: Complete customer information
            analysis_options: Additional analysis options
        
        Returns:
            Comprehensive pricing analysis results
        """
        
        if not self.framework_initialized:
            self._initialize_framework()
        
        # Validate customer data
        validation_result = self._validate_customer_data(customer_data)
        if not validation_result["is_valid"]:
            raise ValueError(f"Invalid customer data: {validation_result['errors']}")
        
        # Set default analysis options
        if analysis_options is None:
            analysis_options = {
                "include_segment_analysis": True,
                "include_roi_analysis": True,
                "include_value_analysis": True,
                "include_competitiveness": True,
                "include_revenue_forecast": True,
                "include_marketing_attribution": True,
                "generate_visualizations": True,
                "analysis_depth": "comprehensive"  # basic, standard, comprehensive
            }
        
        # Initialize results container
        analysis_results = {
            "customer_profile": customer_data,
            "analysis_metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "framework_version": "1.0",
                "analysis_options": analysis_options,
                "config_applied": self.config
            },
            "analysis_results": {}
        }
        
        # Perform component analyses
        if analysis_options.get("include_segment_analysis", True):
            print("Performing segment pricing analysis...")
            segment_results = self._analyze_pricing_segments(customer_data)
            analysis_results["analysis_results"]["segment_analysis"] = segment_results
        
        if analysis_options.get("include_roi_analysis", True):
            print("Calculating ROI projections...")
            roi_results = self._analyze_roi_projections(customer_data)
            analysis_results["analysis_results"]["roi_analysis"] = roi_results
        
        if analysis_options.get("include_value_analysis", True):
            print("Analyzing customer value...")
            value_results = self._analyze_customer_value(customer_data)
            analysis_results["analysis_results"]["value_analysis"] = value_results
        
        if analysis_options.get("include_competitiveness", True):
            print("Assessing competitive positioning...")
            competitive_results = self._assess_competitiveness(customer_data)
            analysis_results["analysis_results"]["competitive_analysis"] = competitive_results
        
        if analysis_options.get("include_revenue_forecast", True):
            print("Generating revenue forecasts...")
            forecast_results = self._generate_revenue_forecasts(customer_data)
            analysis_results["analysis_results"]["revenue_forecast"] = forecast_results
        
        # Generate pricing recommendations
        pricing_recommendations = self._generate_pricing_recommendations(
            analysis_results["analysis_results"]
        )
        analysis_results["pricing_recommendations"] = pricing_recommendations
        
        # Generate implementation strategy
        implementation_strategy = self._generate_implementation_strategy(
            pricing_recommendations, customer_data
        )
        analysis_results["implementation_strategy"] = implementation_strategy
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis_results)
        analysis_results["executive_summary"] = executive_summary
        
        return analysis_results
    
    def _validate_customer_data(self, customer_data: Dict) -> Dict:
        """Validate customer data for analysis"""
        
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["organization_name", "organization_type", "annual_revenue"]
        for field in required_fields:
            if field not in customer_data:
                errors.append(f"Missing required field: {field}")
        
        # Data type validation
        numeric_fields = ["annual_revenue", "annual_patients", "bed_count"]
        for field in numeric_fields:
            if field in customer_data and not isinstance(customer_data[field], (int, float)):
                errors.append(f"Field {field} must be numeric")
        
        # Value range validation
        if "annual_revenue" in customer_data:
            if customer_data["annual_revenue"] <= 0:
                errors.append("Annual revenue must be positive")
            elif customer_data["annual_revenue"] < 1000000:
                warnings.append("Annual revenue seems low for enterprise healthcare organization")
        
        if "annual_patients" in customer_data:
            if customer_data["annual_patients"] <= 0:
                errors.append("Annual patient volume must be positive")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _analyze_pricing_segments(self, customer_data: Dict) -> Dict:
        """Analyze pricing by market segments"""
        
        # Determine market segment
        segment_key = self._determine_market_segment(customer_data)
        
        # Generate pricing proposal
        pricing_proposal = self.pricing_framework.generate_pricing_proposal(
            segment_key=segment_key,
            organization_size=customer_data.get("organization_size", "medium"),
            geographic_region=customer_data.get("geographic_region", "US"),
            specific_needs=customer_data.get("specific_needs", [])
        )
        
        return {
            "segment_identified": segment_key,
            "pricing_proposal": pricing_proposal,
            "segment_characteristics": self.config["market_segments"].get(segment_key, {}),
            "recommended_pricing_model": pricing_proposal["recommended_approach"]["optimal_pricing_model"]
        }
    
    def _determine_market_segment(self, customer_data: Dict) -> str:
        """Determine customer's market segment"""
        
        bed_count = customer_data.get("bed_count", 0)
        patient_volume = customer_data.get("annual_patients", 0)
        org_type = customer_data.get("organization_type", "").lower()
        
        # Academic Medical Centers
        if "academic" in org_type or "university" in org_type:
            return "academic_medical_center"
        
        # Integrated Delivery Networks
        if "idn" in org_type or "network" in org_type or bed_count > 1000:
            return "integrated_delivery_network"
        
        # Large hospitals
        if bed_count > 400 or patient_volume > 20000:
            return "large_hospital"
        
        # Critical access hospitals
        if 25 <= bed_count <= 50:
            return "critical_access_hospital"
        
        # Community clinics and small hospitals
        return "community_clinic"
    
    def _analyze_roi_projections(self, customer_data: Dict) -> Dict:
        """Analyze ROI projections"""
        
        # Create hospital profile
        hospital_profile = HospitalProfile(
            name=customer_data["organization_name"],
            bed_count=customer_data.get("bed_count", 100),
            annual_patients=customer_data.get("annual_patients", 5000),
            avg_length_of_stay=customer_data.get("avg_length_of_stay", 4.0),
            avg_cost_per_patient=customer_data.get("avg_cost_per_patient", 8000),
            annual_revenue=customer_data["annual_revenue"],
            readmission_rate=customer_data.get("readmission_rate", 0.15),
            mortality_rate=customer_data.get("mortality_rate", 0.05),
            patient_satisfaction=customer_data.get("patient_satisfaction", 80),
            geographic_region=customer_data.get("geographic_region", "US")
        )
        
        # Create ROI parameters
        roi_parameters = ROIParameters(
            implementation_cost=customer_data.get("implementation_cost", 100000),
            annual_subscription_cost=customer_data.get("annual_subscription_cost", 300000),
            expected_improvements=customer_data.get("expected_improvements", {
                "mortality_reduction": 0.20,
                "readmission_reduction": 0.25,
                "length_of_stay_reduction": 0.15
            }),
            measurement_period_months=self.config["framework_settings"]["default_measurement_period_months"],
            discount_rate=self.config["framework_settings"]["default_discount_rate"],
            risk_adjustment_factor=customer_data.get("risk_adjustment_factor", 0.9)
        )
        
        # Calculate ROI
        roi_analysis = self.roi_calculator.calculate_comprehensive_roi(
            hospital=hospital_profile,
            ai_solution_type=customer_data.get("ai_solution_type", "comprehensive_ai_platform"),
            parameters=roi_parameters
        )
        
        return roi_analysis
    
    def _analyze_customer_value(self, customer_data: Dict) -> Dict:
        """Analyze customer value"""
        
        # Create customer profile
        customer_profile = CustomerProfile(
            organization_id=customer_data.get("organization_id", "ORG001"),
            organization_type=customer_data["organization_type"],
            revenue=customer_data["annual_revenue"],
            patient_volume=customer_data.get("annual_patients", 5000),
            technology_readiness=customer_data.get("technology_readiness", 0.7),
            outcome_focus=customer_data.get("outcome_focus", 0.7),
            cost_sensitivity=customer_data.get("cost_sensitivity", 0.6),
            decision_speed=customer_data.get("decision_speed", "medium"),
            competitive_position=customer_data.get("competitive_position", "average"),
            geographic_market=customer_data.get("geographic_region", "US"),
            current_solutions=customer_data.get("current_solutions", []),
            pain_points=customer_data.get("pain_points", [])
        )
        
        # Analyze value
        value_analysis = self.value_analyzer.analyze_customer_value(customer_profile)
        
        return value_analysis
    
    def _assess_competitiveness(self, customer_data: Dict) -> Dict:
        """Assess competitive positioning"""
        
        competitiveness = self.pricing_framework.analyze_pricing_competitiveness(
            organization_type=customer_data["organization_type"],
            geographic_region=customer_data.get("geographic_region", "US"),
            competitive_set=customer_data.get("competitors", [])
        )
        
        return competitiveness
    
    def _generate_revenue_forecasts(self, customer_data: Dict) -> Dict:
        """Generate revenue forecasts"""
        
        segment_key = self._determine_market_segment(customer_data)
        
        forecast = self.revenue_operations.create_revenue_forecast(
            forecast_horizon_months=24,
            target_segments=[segment_key],
            growth_scenarios={"conservative": 1.0, "base_case": 1.2, "optimistic": 1.5}
        )
        
        return forecast
    
    def _generate_pricing_recommendations(self, analysis_results: Dict) -> Dict:
        """Generate comprehensive pricing recommendations"""
        
        recommendations = {
            "primary_recommendation": {},
            "alternative_options": [],
            "pricing_strategy": {},
            "implementation_priority": "high"
        }
        
        # Extract key insights from analyses
        roi_analysis = analysis_results.get("roi_analysis", {})
        value_analysis = analysis_results.get("value_analysis", {})
        segment_analysis = analysis_results.get("segment_analysis", {})
        
        # ROI-based recommendation logic
        roi_ratio = roi_analysis.get("roi_metrics", {}).get("roi_metrics", {}).get("roi_ratio", 0)
        payback_months = roi_analysis.get("roi_metrics", {}).get("roi_metrics", {}).get("payback_period_months", float('inf'))
        
        # Value-based recommendation logic
        wtp_percentage = value_analysis.get("financial_analysis", {}).get("willingness_to_pay", {}).get("willingness_to_pay_percentage", 0)
        customer_ltv = value_analysis.get("financial_analysis", {}).get("customer_lifetime_value", {}).get("total_clv", 0)
        
        # Determine primary recommendation
        if roi_ratio > 3.0 and wtp_percentage > 0.35:
            recommendations["primary_recommendation"] = {
                "model": "value_based_pricing",
                "rationale": "Strong ROI and high willingness to pay justify outcome-based pricing",
                "confidence": "high"
            }
        elif roi_ratio > 2.0 and customer_ltv > 1000000:
            recommendations["primary_recommendation"] = {
                "model": "enterprise_licensing",
                "rationale": "Strong ROI and high LTV support enterprise investment",
                "confidence": "high"
            }
        else:
            recommendations["primary_recommendation"] = {
                "model": "subscription_pricing",
                "rationale": "Balanced approach with lower risk and predictable costs",
                "confidence": "medium"
            }
        
        # Generate alternative options
        models = ["value_based_pricing", "enterprise_licensing", "subscription_pricing"]
        primary_model = recommendations["primary_recommendation"]["model"]
        
        for model in models:
            if model != primary_model:
                recommendations["alternative_options"].append({
                    "model": model,
                    "description": f"Alternative {model.replace('_', ' ').title()} approach",
                    "when_to_consider": self._get_model_consideration_criteria(model, analysis_results)
                })
        
        return recommendations
    
    def _get_model_consideration_criteria(self, model: str, analysis_results: Dict) -> str:
        """Get criteria for considering alternative pricing models"""
        
        criteria = {
            "value_based_pricing": "When customer has strong outcome focus and reliable measurement capability",
            "enterprise_licensing": "When organization size and complexity justify comprehensive solution",
            "subscription_pricing": "When quick deployment and predictable costs are priorities"
        }
        
        return criteria.get(model, "General alternative pricing approach")
    
    def _generate_implementation_strategy(self, recommendations: Dict, customer_data: Dict) -> Dict:
        """Generate implementation strategy"""
        
        return {
            "implementation_phases": [
                {
                    "phase": "Contract Negotiation",
                    "duration_weeks": 2,
                    "activities": ["Pricing finalization", "Terms negotiation", "Contract signing"]
                },
                {
                    "phase": "Technical Planning",
                    "duration_weeks": 4,
                    "activities": ["Requirements finalization", "Architecture planning", "Resource allocation"]
                },
                {
                    "phase": "Implementation",
                    "duration_weeks": 8,
                    "activities": ["System deployment", "Integration", "Testing"]
                },
                {
                    "phase": "Go-Live",
                    "duration_weeks": 4,
                    "activities": ["Training", "Pilot deployment", "Full rollout"]
                }
            ],
            "total_timeline_weeks": 18,
            "critical_success_factors": [
                "Executive sponsorship",
                "Clear success metrics",
                "Comprehensive change management",
                "Continuous monitoring and optimization"
            ],
            "risk_mitigation": [
                "Phased rollout approach",
                "Performance guarantees",
                "Flexible payment terms",
                "24/7 support during transition"
            ]
        }
    
    def _generate_executive_summary(self, analysis_results: Dict) -> str:
        """Generate executive summary"""
        
        customer_data = analysis_results["customer_profile"]
        roi_metrics = analysis_results["analysis_results"].get("roi_analysis", {}).get("roi_metrics", {})
        value_metrics = analysis_results["analysis_results"].get("value_analysis", {}).get("financial_analysis", {})
        
        summary = f"""
EXECUTIVE SUMMARY
================

Organization: {customer_data['organization_name']}
Segment: {customer_data['organization_type']}
Annual Revenue: ${customer_data['annual_revenue']:,.0f}
Patient Volume: {customer_data.get('annual_patients', 'N/A'):,} annually

RECOMMENDED PRICING STRATEGY
--------------------------
Model: {analysis_results['pricing_recommendations']['primary_recommendation']['model'].replace('_', ' ').title()}
Confidence: {analysis_results['pricing_recommendations']['primary_recommendation']['confidence'].upper()}
Rationale: {analysis_results['pricing_recommendations']['primary_recommendation']['rationale']}

FINANCIAL PROJECTIONS
--------------------
ROI Ratio: {roi_metrics.get('roi_metrics', {}).get('roi_ratio', 0):.2f}x
Payback Period: {roi_metrics.get('roi_metrics', {}).get('payback_period_months', 0):.1f} months
Customer LTV: ${value_metrics.get('customer_lifetime_value', {}).get('total_clv', 0):,.0f}

IMPLEMENTATION
-------------
Total Timeline: {analysis_results['implementation_strategy']['total_timeline_weeks']} weeks
Key Success Factors: {len(analysis_results['implementation_strategy']['critical_success_factors'])} critical factors identified

Generated: {analysis_results['analysis_metadata']['analysis_timestamp']}
        """
        
        return summary.strip()
    
    def export_analysis(self, analysis_results: Dict, output_path: str, formats: List[str] = None) -> Dict:
        """
        Export analysis results in multiple formats
        
        Args:
            analysis_results: Analysis results to export
            output_path: Base output path
            formats: List of formats (json, csv, xlsx, pdf)
        
        Returns:
            Dictionary with export results
        """
        
        if formats is None:
            formats = ["json", "xlsx"]
        
        export_results = {}
        
        # JSON export
        if "json" in formats:
            json_path = f"{output_path}.json"
            with open(json_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            export_results["json"] = json_path
        
        # CSV export (for key metrics)
        if "csv" in formats:
            csv_path = f"{output_path}_summary.csv"
            self._export_summary_csv(analysis_results, csv_path)
            export_results["csv"] = csv_path
        
        # Excel export
        if "xlsx" in formats:
            xlsx_path = f"{output_path}.xlsx"
            self._export_comprehensive_excel(analysis_results, xlsx_path)
            export_results["xlsx"] = xlsx_path
        
        return export_results
    
    def _export_summary_csv(self, analysis_results: Dict, output_path: str):
        """Export summary metrics to CSV"""
        
        customer_data = analysis_results["customer_profile"]
        roi_metrics = analysis_results["analysis_results"].get("roi_analysis", {}).get("roi_metrics", {})
        value_metrics = analysis_results["analysis_results"].get("value_analysis", {}).get("financial_analysis", {})
        
        summary_data = {
            "Organization": [customer_data["organization_name"]],
            "Segment": [customer_data["organization_type"]],
            "Annual_Revenue": [customer_data["annual_revenue"]],
            "Patient_Volume": [customer_data.get("annual_patients", 0)],
            "ROI_Ratio": [roi_metrics.get("roi_metrics", {}).get("roi_ratio", 0)],
            "Payback_Months": [roi_metrics.get("roi_metrics", {}).get("payback_period_months", 0)],
            "NPV": [roi_metrics.get("roi_metrics", {}).get("net_present_value", 0)],
            "Customer_LTV": [value_metrics.get("customer_lifetime_value", {}).get("total_clv", 0)],
            "Recommended_Model": [analysis_results["pricing_recommendations"]["primary_recommendation"]["model"]],
            "Analysis_Date": [analysis_results["analysis_metadata"]["analysis_timestamp"]]
        }
        
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)
    
    def _export_comprehensive_excel(self, analysis_results: Dict, output_path: str):
        """Export comprehensive analysis to Excel"""
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Customer Summary
            customer_data = analysis_results["customer_profile"]
            customer_df = pd.DataFrame([customer_data])
            customer_df.to_excel(writer, sheet_name='Customer_Profile', index=False)
            
            # ROI Metrics
            roi_data = analysis_results["analysis_results"].get("roi_analysis", {}).get("roi_metrics", {})
            roi_df = pd.DataFrame([roi_data])
            roi_df.to_excel(writer, sheet_name='ROI_Analysis', index=False)
            
            # Value Analysis
            value_data = analysis_results["analysis_results"].get("value_analysis", {})
            value_df = pd.DataFrame([value_data])
            value_df.to_excel(writer, sheet_name='Value_Analysis', index=False)
            
            # Pricing Recommendations
            recommendations = analysis_results["pricing_recommendations"]
            rec_df = pd.DataFrame([recommendations])
            rec_df.to_excel(writer, sheet_name='Recommendations', index=False)

def create_sample_analysis():
    """Create sample pricing analysis"""
    
    orchestrator = HealthcarePricingOrchestrator()
    
    # Sample customer data
    sample_customer = {
        "organization_name": "Regional Medical Center",
        "organization_type": "hospital",
        "annual_revenue": 300000000,
        "annual_patients": 25000,
        "bed_count": 450,
        "avg_length_of_stay": 4.0,
        "avg_cost_per_patient": 8500,
        "readmission_rate": 0.14,
        "mortality_rate": 0.045,
        "patient_satisfaction": 85,
        "technology_readiness": 0.8,
        "outcome_focus": 0.85,
        "cost_sensitivity": 0.5,
        "geographic_region": "Midwest",
        "specific_needs": ["clinical_outcomes", "operational_efficiency"],
        "implementation_cost": 150000,
        "annual_subscription_cost": 400000,
        "expected_improvements": {
            "mortality_reduction": 0.20,
            "readmission_reduction": 0.25,
            "length_of_stay_reduction": 0.15
        }
    }
    
    # Perform analysis
    print("Running comprehensive pricing analysis...")
    analysis = orchestrator.analyze_customer_pricing(sample_customer)
    
    # Export results
    print("Exporting analysis results...")
    export_results = orchestrator.export_analysis(
        analysis, 
        "/workspace/market/pricing/sample_analysis",
        ["json", "csv", "xlsx"]
    )
    
    # Print summary
    print(analysis["executive_summary"])
    
    return analysis, export_results

if __name__ == "__main__":
    # Create sample analysis
    analysis, exports = create_sample_analysis()
    
    print(f"\nAnalysis complete! Export files created:")
    for format_type, file_path in exports.items():
        print(f"- {format_type.upper()}: {file_path}")
