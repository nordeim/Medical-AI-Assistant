"""
Configuration Management for Revenue Optimization Framework

This module provides comprehensive configuration management for the healthcare AI
pricing framework, including market segment configurations, pricing parameters,
and operational settings.
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import os

@dataclass
class MarketSegmentConfig:
    """Configuration for market segments"""
    segment_name: str
    base_price: float
    win_rate: float
    sales_cycle_days: int
    average_deal_size: float
    decision_maker_count: int
    price_sensitivity: float
    competitive_intensity: float
    typical_budget_range: Dict[str, float]
    value_drivers: List[str]
    success_factors: List[str]

@dataclass
class PricingModelConfig:
    """Configuration for pricing models"""
    model_name: str
    base_price: float
    price_factors: Dict[str, float]
    discount_tiers: Dict[str, float]
    usage_tiers: Dict[str, float]
    contractual_terms: Dict[str, str]

@dataclass
class ROIConfig:
    """Configuration for ROI calculations"""
    discount_rate: float
    time_horizon_months: int
    confidence_threshold: float
    risk_multipliers: Dict[str, float]
    benefit_categories: Dict[str, Dict]

@dataclass
class RevenueOpsConfig:
    """Configuration for revenue operations"""
    forecast_horizon_months: int
    attribution_models: Dict[str, float]
    pipeline_stages: List[Dict]
    customer_health_metrics: List[str]
    churn_indicators: List[str]

class PricingConfigManager:
    """Configuration manager for pricing framework"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        self.market_segments: Dict[str, MarketSegmentConfig] = {}
        self.pricing_models: Dict[str, PricingModelConfig] = {}
        self.roi_config: Optional[ROIConfig] = None
        self.revenue_ops_config: Optional[RevenueOpsConfig] = None
        
        # Load all configurations
        self._load_configurations()
        
    def _load_configurations(self) -> None:
        """Load all configuration files"""
        self._load_market_segments()
        self._load_pricing_models()
        self._load_roi_config()
        self._load_revenue_ops_config()
        
    def _load_market_segments(self) -> None:
        """Load market segment configurations"""
        config_file = self.config_dir / "market_segments.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                
            for segment_name, config_data in data.items():
                self.market_segments[segment_name] = MarketSegmentConfig(**config_data)
        else:
            # Create default configurations
            self._create_default_market_segments()
            
    def _load_pricing_models(self) -> None:
        """Load pricing model configurations"""
        config_file = self.config_dir / "pricing_models.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
                
            for model_name, config_data in data.items():
                self.pricing_models[model_name] = PricingModelConfig(**config_data)
        else:
            self._create_default_pricing_models()
            
    def _load_roi_config(self) -> None:
        """Load ROI calculation configurations"""
        config_file = self.config_dir / "roi_config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            self.roi_config = ROIConfig(**data)
        else:
            self._create_default_roi_config()
            
    def _load_revenue_ops_config(self) -> None:
        """Load revenue operations configurations"""
        config_file = self.config_dir / "revenue_ops.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            self.revenue_ops_config = RevenueOpsConfig(**data)
        else:
            self._create_default_revenue_ops_config()
            
    def _create_default_market_segments(self) -> None:
        """Create default market segment configurations"""
        segments = {
            "hospital_system": MarketSegmentConfig(
                segment_name="Hospital System",
                base_price=500000,
                win_rate=0.25,
                sales_cycle_days=180,
                average_deal_size=450000,
                decision_maker_count=8,
                price_sensitivity=0.6,
                competitive_intensity=0.8,
                typical_budget_range={"min": 300000, "max": 800000},
                value_drivers=["clinical_outcomes", "operational_efficiency", "quality_metrics"],
                success_factors=["strong_clinical_champion", "clear_roi_demonstration", "integration_capabilities"]
            ),
            "academic_medical_center": MarketSegmentConfig(
                segment_name="Academic Medical Center",
                base_price=300000,
                win_rate=0.30,
                sales_cycle_days=240,
                average_deal_size=320000,
                decision_maker_count=6,
                price_sensitivity=0.5,
                competitive_intensity=0.7,
                typical_budget_range={"min": 200000, "max": 500000},
                value_drivers=["research_capabilities", "educational_value", "innovation"],
                success_factors=["research_partnership", "faculty_engagement", "educational_integration"]
            ),
            "clinic": MarketSegmentConfig(
                segment_name="Clinic",
                base_price=120000,
                win_rate=0.35,
                sales_cycle_days=90,
                average_deal_size=85000,
                decision_maker_count=3,
                price_sensitivity=0.8,
                competitive_intensity=0.6,
                typical_budget_range={"min": 80000, "max": 150000},
                value_drivers=["efficiency_gains", "patient_satisfaction", "competitive_advantage"],
                success_factors=["quick_implementation", "clear_value_proposition", "affordable_pricing"]
            ),
            "integrated_delivery_network": MarketSegmentConfig(
                segment_name="Integrated Delivery Network",
                base_price=650000,
                win_rate=0.22,
                sales_cycle_days=210,
                average_deal_size=650000,
                decision_maker_count=10,
                price_sensitivity=0.5,
                competitive_intensity=0.9,
                typical_budget_range={"min": 400000, "max": 1200000},
                value_drivers=["network_efficiency", "population_health", "standardization"],
                success_factors=["enterprise_scale", "cross_facility_coordination", "network_wide_roi"]
            ),
            "regional_hospital": MarketSegmentConfig(
                segment_name="Regional Hospital",
                base_price=180000,
                win_rate=0.28,
                sales_cycle_days=120,
                average_deal_size=180000,
                decision_maker_count=5,
                price_sensitivity=0.7,
                competitive_intensity=0.6,
                typical_budget_range={"min": 120000, "max": 350000},
                value_drivers=["cost_effectiveness", "quality_improvement", "operational_efficiency"],
                success_factors=["regional_focus", "support_localization", "cost_competitiveness"]
            ),
            "specialty_clinic": MarketSegmentConfig(
                segment_name="Specialty Clinic",
                base_price=120000,
                win_rate=0.32,
                sales_cycle_days=100,
                average_deal_size=120000,
                decision_maker_count=4,
                price_sensitivity=0.7,
                competitive_intensity=0.5,
                typical_budget_range={"min": 80000, "max": 200000},
                value_drivers=["specialty_workflows", "efficiency", "patient_outcomes"],
                success_factors=["specialty_expertise", "workflow_optimization", "roi_focus"]
            ),
            "rural_hospital": MarketSegmentConfig(
                segment_name="Rural Hospital",
                base_price=65000,
                win_rate=0.40,
                sales_cycle_days=80,
                average_deal_size=65000,
                decision_maker_count=2,
                price_sensitivity=0.9,
                competitive_intensity=0.4,
                typical_budget_range={"min": 40000, "max": 120000},
                value_drivers=["affordability", "simplicity", "support"],
                success_factors=["affordable_pricing", "easy_implementation", "strong_support"]
            )
        }
        
        self.market_segments = segments
        self._save_market_segments()
        
    def _create_default_pricing_models(self) -> None:
        """Create default pricing model configurations"""
        models = {
            "subscription": PricingModelConfig(
                model_name="Subscription",
                base_price=120000,
                price_factors={
                    "market_segment": 1.0,
                    "customer_tier": 1.0,
                    "volume_discount": 0.95,
                    "competitive_pressure": 1.0
                },
                discount_tiers={
                    "platinum": 1.0,
                    "gold": 0.9,
                    "silver": 0.8,
                    "bronze": 0.7
                },
                usage_tiers={
                    "basic": 1.0,
                    "standard": 1.2,
                    "premium": 1.5,
                    "enterprise": 2.0
                },
                contractual_terms={
                    "term_length": "3 years",
                    "payment_terms": "Annual",
                    "renewal_discount": "5%"
                }
            ),
            "enterprise_license": PricingModelConfig(
                model_name="Enterprise License",
                base_price=500000,
                price_factors={
                    "market_segment": 1.2,
                    "customer_tier": 1.0,
                    "implementation_complexity": 1.1,
                    "custom_requirements": 1.3
                },
                discount_tiers={
                    "platinum": 1.0,
                    "gold": 0.95,
                    "silver": 0.9,
                    "bronze": 0.85
                },
                usage_tiers={
                    "single_facility": 1.0,
                    "multi_facility": 1.5,
                    "network": 2.0,
                    "enterprise": 3.0
                },
                contractual_terms={
                    "setup_fee": "20% of annual fee",
                    "maintenance_fee": "15% annually",
                    "support_included": "Business hours"
                }
            ),
            "value_based": PricingModelConfig(
                model_name="Value Based",
                base_price=200000,
                price_factors={
                    "clinical_outcomes": 1.5,
                    "operational_efficiency": 1.3,
                    "implementation_success": 1.2,
                    "value_capture_rate": 0.25
                },
                discount_tiers={
                    "outcome_guaranteed": 1.0,
                    "high_confidence": 0.9,
                    "medium_confidence": 0.8,
                    "pilot_program": 0.7
                },
                usage_tiers={
                    "pilot": 0.5,
                    "single_department": 1.0,
                    "multi_department": 1.3,
                    "enterprise": 1.8
                },
                contractual_terms={
                    "payment_schedule": "Performance based",
                    "measurement_period": "12 months",
                    "value_adjustment": "Semi-annual"
                }
            ),
            "per_patient": PricingModelConfig(
                model_name="Per Patient",
                base_price=0.50,
                price_factors={
                    "patient_volume": 1.0,
                    "clinical_complexity": 1.4,
                    "service_level": 1.0,
                    "integration_depth": 1.2
                },
                discount_tiers={
                    "high_volume": 0.8,
                    "medium_volume": 0.9,
                    "low_volume": 1.0,
                    "pilot": 1.2
                },
                usage_tiers={
                    "minimum_100k": 0.8,
                    "minimum_50k": 0.9,
                    "minimum_25k": 1.0,
                    "minimum_10k": 1.2
                },
                contractual_terms={
                    "billing_cycle": "Monthly",
                    "volume_commitment": "Annual minimum",
                    "overage_rates": "Standard rates"
                }
            )
        }
        
        self.pricing_models = models
        self._save_pricing_models()
        
    def _create_default_roi_config(self) -> None:
        """Create default ROI configuration"""
        self.roi_config = ROIConfig(
            discount_rate=0.08,
            time_horizon_months=36,
            confidence_threshold=0.8,
            risk_multipliers={
                "implementation_risk": 0.9,
                "adoption_risk": 0.85,
                "clinical_risk": 0.8,
                "competitive_risk": 0.9,
                "regulatory_risk": 0.95
            },
            benefit_categories={
                "clinical_quality": {
                    "weight": 0.4,
                    "measurement_frequency": "monthly",
                    "confidence_factors": {
                        "peer_reviewed_studies": 0.9,
                        "case_studies": 0.8,
                        "pilot_results": 0.7,
                        "vendor_claims": 0.5
                    }
                },
                "operational_efficiency": {
                    "weight": 0.3,
                    "measurement_frequency": "monthly",
                    "confidence_factors": {
                        "documented_processes": 0.9,
                        "historical_data": 0.8,
                        "industry_benchmarks": 0.7,
                        "vendor_estimates": 0.6
                    }
                },
                "financial_impact": {
                    "weight": 0.3,
                    "measurement_frequency": "quarterly",
                    "confidence_factors": {
                        "actual_savings": 0.95,
                        "validated_projections": 0.85,
                        "market_benchmarks": 0.75,
                        "estimated_benefits": 0.6
                    }
                }
            }
        )
        self._save_roi_config()
        
    def _create_default_revenue_ops_config(self) -> None:
        """Create default revenue operations configuration"""
        self.revenue_ops_config = RevenueOpsConfig(
            forecast_horizon_months=12,
            attribution_models={
                "first_touch": 0.3,
                "last_touch": 0.4,
                "linear": 0.2,
                "time_decay": 0.1
            },
            pipeline_stages=[
                {"stage": "lead", "probability": 0.1, "avg_days": 14},
                {"stage": "qualified", "probability": 0.25, "avg_days": 30},
                {"stage": "proposal", "probability": 0.5, "avg_days": 45},
                {"stage": "negotiation", "probability": 0.75, "avg_days": 30},
                {"stage": "closed_won", "probability": 1.0, "avg_days": 0},
                {"stage": "closed_lost", "probability": 0.0, "avg_days": 0}
            ],
            customer_health_metrics=[
                "product_usage",
                "support_ticket_rate",
                "satisfaction_score",
                "renewal_probability",
                "expansion_indicators"
            ],
            churn_indicators=[
                "low_usage_frequency",
                "high_support_tickets",
                "low_satisfaction_scores",
                "missed_renewal_dates",
                "competitive_solicitation"
            ]
        )
        self._save_revenue_ops_config()
        
    def _save_market_segments(self) -> None:
        """Save market segment configurations to file"""
        config_file = self.config_dir / "market_segments.yaml"
        data = {
            name: asdict(config) for name, config in self.market_segments.items()
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
            
    def _save_pricing_models(self) -> None:
        """Save pricing model configurations to file"""
        config_file = self.config_dir / "pricing_models.yaml"
        data = {
            name: asdict(config) for name, config in self.pricing_models.items()
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
            
    def _save_roi_config(self) -> None:
        """Save ROI configuration to file"""
        if self.roi_config:
            config_file = self.config_dir / "roi_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(asdict(self.roi_config), f, default_flow_style=False, indent=2)
                
    def _save_revenue_ops_config(self) -> None:
        """Save revenue operations configuration to file"""
        if self.revenue_ops_config:
            config_file = self.config_dir / "revenue_ops.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(asdict(self.revenue_ops_config), f, default_flow_style=False, indent=2)
                
    def get_market_segment_config(self, segment_name: str) -> Optional[MarketSegmentConfig]:
        """Get configuration for specific market segment"""
        return self.market_segments.get(segment_name)
        
    def get_pricing_model_config(self, model_name: str) -> Optional[PricingModelConfig]:
        """Get configuration for specific pricing model"""
        return self.pricing_models.get(model_name)
        
    def update_market_segment_config(self, segment_name: str, 
                                   config: MarketSegmentConfig) -> None:
        """Update market segment configuration"""
        self.market_segments[segment_name] = config
        self._save_market_segments()
        
    def update_pricing_model_config(self, model_name: str, 
                                  config: PricingModelConfig) -> None:
        """Update pricing model configuration"""
        self.pricing_models[model_name] = config
        self._save_pricing_models()
        
    def export_config_to_json(self, output_file: str) -> None:
        """Export all configurations to JSON file"""
        config_data = {
            "market_segments": {
                name: asdict(config) for name, config in self.market_segments.items()
            },
            "pricing_models": {
                name: asdict(config) for name, config in self.pricing_models.items()
            },
            "roi_config": asdict(self.roi_config) if self.roi_config else None,
            "revenue_ops_config": asdict(self.revenue_ops_config) if self.revenue_ops_config else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
            
    def import_config_from_json(self, input_file: str) -> None:
        """Import configurations from JSON file"""
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Import market segments
        if "market_segments" in data:
            self.market_segments = {
                name: MarketSegmentConfig(**config_data)
                for name, config_data in data["market_segments"].items()
            }
            
        # Import pricing models
        if "pricing_models" in data:
            self.pricing_models = {
                name: PricingModelConfig(**config_data)
                for name, config_data in data["pricing_models"].items()
            }
            
        # Import ROI config
        if "roi_config" in data and data["roi_config"]:
            self.roi_config = ROIConfig(**data["roi_config"])
            
        # Import revenue ops config
        if "revenue_ops_config" in data and data["revenue_ops_config"]:
            self.revenue_ops_config = RevenueOpsConfig(**data["revenue_ops_config"])
            
    def validate_configurations(self) -> Dict[str, List[str]]:
        """Validate all configurations for completeness and correctness"""
        validation_results = {}
        
        # Validate market segments
        segment_errors = []
        for name, config in self.market_segments.items():
            if config.base_price <= 0:
                segment_errors.append(f"{name}: Base price must be positive")
            if not 0 <= config.win_rate <= 1:
                segment_errors.append(f"{name}: Win rate must be between 0 and 1")
            if config.sales_cycle_days <= 0:
                segment_errors.append(f"{name}: Sales cycle must be positive")
                
        validation_results["market_segments"] = segment_errors
        
        # Validate pricing models
        model_errors = []
        for name, config in self.pricing_models.items():
            if config.base_price <= 0:
                model_errors.append(f"{name}: Base price must be positive")
            if not all(0 <= discount <= 1 for discount in config.discount_tiers.values()):
                model_errors.append(f"{name}: Discounts must be between 0 and 1")
                
        validation_results["pricing_models"] = model_errors
        
        # Validate ROI config
        roi_errors = []
        if self.roi_config:
            if not 0 <= self.roi_config.discount_rate <= 1:
                roi_errors.append("Discount rate must be between 0 and 1")
            if self.roi_config.time_horizon_months <= 0:
                roi_errors.append("Time horizon must be positive")
        else:
            roi_errors.append("ROI configuration is missing")
            
        validation_results["roi_config"] = roi_errors
        
        # Validate revenue ops config
        rev_ops_errors = []
        if self.revenue_ops_config:
            if self.revenue_ops_config.forecast_horizon_months <= 0:
                rev_ops_errors.append("Forecast horizon must be positive")
        else:
            rev_ops_errors.append("Revenue operations configuration is missing")
            
        validation_results["revenue_ops"] = rev_ops_errors
        
        return validation_results
        
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        return {
            "market_segments": {
                "count": len(self.market_segments),
                "segments": list(self.market_segments.keys())
            },
            "pricing_models": {
                "count": len(self.pricing_models),
                "models": list(self.pricing_models.keys())
            },
            "roi_config": {
                "configured": self.roi_config is not None,
                "discount_rate": self.roi_config.discount_rate if self.roi_config else None,
                "time_horizon": self.roi_config.time_horizon_months if self.roi_config else None
            },
            "revenue_ops_config": {
                "configured": self.revenue_ops_config is not None,
                "forecast_horizon": self.revenue_ops_config.forecast_horizon_months if self.revenue_ops_config else None,
                "pipeline_stages": len(self.revenue_ops_config.pipeline_stages) if self.revenue_ops_config else 0
            }
        }


if __name__ == "__main__":
    # Example usage
    config_manager = PricingConfigManager()
    
    # Get configuration summary
    summary = config_manager.get_config_summary()
    print("Configuration Summary:")
    print(json.dumps(summary, indent=2, default=str))
    
    # Validate configurations
    validation = config_manager.validate_configurations()
    print("\nValidation Results:")
    print(json.dumps(validation, indent=2))
    
    # Export configurations
    config_manager.export_config_to_json("pricing_config_export.json")
    print("\nConfigurations exported to pricing_config_export.json")
    
    # Get specific configuration
    hospital_config = config_manager.get_market_segment_config("hospital_system")
    if hospital_config:
        print(f"\nHospital System Config:")
        print(f"Base Price: ${hospital_config.base_price:,}")
        print(f"Win Rate: {hospital_config.win_rate:.1%}")
        print(f"Sales Cycle: {hospital_config.sales_cycle_days} days")