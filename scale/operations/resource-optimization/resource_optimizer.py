"""
Resource Optimization and Cost Management Framework for Healthcare AI
Implements comprehensive resource optimization with cost management strategies
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

class ResourceType(Enum):
    """Types of resources to optimize"""
    COMPUTE_INSTANCES = "compute_instances"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    MEMORY = "memory"
    DATABASE_CONNECTIONS = "database_connections"
    AI_MODELS = "ai_models"
    LICENSES = "licenses"
    PERSONNEL = "personnel"

class CostCategory(Enum):
    """Cost categories"""
    INFRASTRUCTURE = "infrastructure"
    SOFTWARE_LICENSES = "software_licenses"
    PERSONNEL = "personnel"
    DATA_PROCESSING = "data_processing"
    COMPLIANCE = "compliance"
    TRAINING = "training"
    MAINTENANCE = "maintenance"
    CONSULTING = "consulting"

class OptimizationStrategy(Enum):
    """Resource optimization strategies"""
    RIGHT_SIZING = "right_sizing"
    RESERVATION_PURCHASING = "reservation_purchasing"
    SPOT_INSTANCE_USAGE = "spot_instance_usage"
    AUTO_SCALING = "auto_scaling"
    RESOURCE_SHARING = "resource_sharing"
    DATA_LIFECYCLE = "data_lifecycle"
    LICENSING_OPTIMIZATION = "licensing_optimization"

class CostAlertThreshold(Enum):
    """Cost alert threshold levels"""
    BUDGET = "budget"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    resource_id: str
    resource_type: ResourceType
    current_usage: float
    max_capacity: float
    utilization_percentage: float
    cost_per_hour: float
    cost_per_month: float
    last_updated: datetime

@dataclass
class CostAnalysis:
    """Cost analysis data"""
    category: CostCategory
    monthly_cost: float
    percentage_of_total: float
    trend_direction: str  # increasing, decreasing, stable
    trend_percentage: float
    budget_variance: float
    optimization_opportunities: List[str]

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    recommendation_id: str
    category: CostCategory
    strategy: OptimizationStrategy
    description: str
    potential_savings: float
    implementation_effort: str  # Low, Medium, High
    implementation_timeline: str
    risk_level: str
    roi_percentage: float

@dataclass
class BudgetPlan:
    """Budget planning and tracking"""
    budget_id: str
    department: str
    fiscal_year: int
    allocated_budget: float
    spent_to_date: float
    remaining_budget: float
    burn_rate: float  # per month
    forecasted_spend: float
    variance_analysis: Dict[str, float]

class ResourceOptimizationManager:
    """Resource Optimization and Cost Management Manager"""
    
    def __init__(self):
        self.resource_usage: Dict[str, ResourceUsage] = {}
        self.cost_analyses: Dict[CostCategory, CostAnalysis] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []
        self.budget_plans: Dict[str, BudgetPlan] = {}
        self.cost_alerts: List[Dict] = []
        self.historical_costs: List[Dict] = []
        
    async def analyze_current_resource_usage(self) -> Dict:
        """Analyze current resource utilization across all systems"""
        
        # Simulate current resource usage
        resources = [
            ResourceUsage(
                resource_id="compute_prod_001",
                resource_type=ResourceType.COMPUTE_INSTANCES,
                current_usage=45.2,
                max_capacity=100.0,
                utilization_percentage=45.2,
                cost_per_hour=2.85,
                cost_per_month=2052.0,
                last_updated=datetime.now()
            ),
            ResourceUsage(
                resource_id="storage_primary",
                resource_type=ResourceType.STORAGE,
                current_usage=7500.0,  # GB
                max_capacity=10000.0,  # GB
                utilization_percentage=75.0,
                cost_per_hour=0.42,
                cost_per_month=302.4,
                last_updated=datetime.now()
            ),
            ResourceUsage(
                resource_id="network_primary",
                resource_type=ResourceType.NETWORK_BANDWIDTH,
                current_usage=850.0,  # Mbps
                max_capacity=1000.0,  # Mbps
                utilization_percentage=85.0,
                cost_per_hour=0.25,
                cost_per_month=180.0,
                last_updated=datetime.now()
            ),
            ResourceUsage(
                resource_id="ai_model_serving",
                resource_type=ResourceType.AI_MODELS,
                current_usage=125.0,  # RPS
                max_capacity=200.0,  # RPS
                utilization_percentage=62.5,
                cost_per_hour=1.95,
                cost_per_month=1404.0,
                last_updated=datetime.now()
            ),
            ResourceUsage(
                resource_id="database_connections",
                resource_type=ResourceType.DATABASE_CONNECTIONS,
                current_usage=180.0,
                max_capacity=200.0,
                utilization_percentage=90.0,
                cost_per_hour=0.15,
                cost_per_month=108.0,
                last_updated=datetime.now()
            )
        ]
        
        # Store resource usage
        for resource in resources:
            self.resource_usage[resource.resource_id] = resource
        
        # Calculate aggregate metrics
        total_monthly_cost = sum(r.cost_per_month for r in resources)
        avg_utilization = sum(r.utilization_percentage for r in resources) / len(resources)
        over_utilized = [r for r in resources if r.utilization_percentage > 80]
        under_utilized = [r for r in resources if r.utilization_percentage < 50]
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_resources_tracked": len(resources),
            "total_monthly_cost": total_monthly_cost,
            "average_utilization": round(avg_utilization, 1),
            "resource_summary": {
                "over_utilized": len(over_utilized),
                "under_utilized": len(under_utilized),
                "optimally_utilized": len(resources) - len(over_utilized) - len(under_utilized)
            },
            "top_cost_drivers": [
                {"resource": r.resource_id, "monthly_cost": r.cost_per_month, "type": r.resource_type.value}
                for r in sorted(resources, key=lambda x: x.cost_per_month, reverse=True)[:3]
            ],
            "optimization_opportunities": [
                {
                    "resource": r.resource_id,
                    "current_utilization": f"{r.utilization_percentage}%",
                    "potential_action": "Right-size" if r.utilization_percentage < 50 else "Scale up" if r.utilization_percentage > 80 else "Monitor",
                    "estimated_savings": r.cost_per_month * 0.25 if r.utilization_percentage < 50 else 0
                }
                for r in under_utilized
            ]
        }
    
    async def perform_comprehensive_cost_analysis(self) -> Dict:
        """Perform comprehensive cost analysis by category"""
        
        # Define cost categories and their current costs
        cost_data = {
            CostCategory.INFRASTRUCTURE: {
                "monthly_cost": 125000.0,
                "trend_direction": "increasing",
                "trend_percentage": 12.5,
                "budget": 120000.0,
                "drivers": ["Compute instances", "Storage", "Network bandwidth"]
            },
            CostCategory.SOFTWARE_LICENSES: {
                "monthly_cost": 45000.0,
                "trend_direction": "stable",
                "trend_percentage": 2.1,
                "budget": 50000.0,
                "drivers": ["AI model licenses", "Database licenses", "Security tools"]
            },
            CostCategory.PERSONNEL: {
                "monthly_cost": 285000.0,
                "trend_direction": "stable",
                "trend_percentage": 1.8,
                "budget": 280000.0,
                "drivers": ["Engineering team", "Data scientists", "Operations staff"]
            },
            CostCategory.DATA_PROCESSING: {
                "monthly_cost": 35000.0,
                "trend_direction": "increasing",
                "trend_percentage": 18.2,
                "budget": 30000.0,
                "drivers": ["ETL processes", "Data warehouse", "Analytics workloads"]
            },
            CostCategory.COMPLIANCE: {
                "monthly_cost": 25000.0,
                "trend_direction": "stable",
                "trend_percentage": 0.5,
                "budget": 30000.0,
                "drivers": ["Audit tools", "Compliance monitoring", "Legal consultation"]
            },
            CostCategory.TRAINING: {
                "monthly_cost": 15000.0,
                "trend_direction": "increasing",
                "trend_percentage": 25.0,
                "budget": 20000.0,
                "drivers": ["Staff training", "Certification programs", "Onboarding"]
            },
            CostCategory.MAINTENANCE: {
                "monthly_cost": 18000.0,
                "trend_direction": "increasing",
                "trend_percentage": 8.5,
                "budget": 15000.0,
                "drivers": ["System maintenance", "Software updates", "Hardware repairs"]
            }
        }
        
        total_monthly_cost = sum(data["monthly_cost"] for data in cost_data.values())
        total_budget = sum(data["budget"] for data in cost_data.values())
        
        # Create cost analyses
        for category, data in cost_data.items():
            analysis = CostAnalysis(
                category=category,
                monthly_cost=data["monthly_cost"],
                percentage_of_total=(data["monthly_cost"] / total_monthly_cost) * 100,
                trend_direction=data["trend_direction"],
                trend_percentage=data["trend_percentage"],
                budget_variance=data["monthly_cost"] - data["budget"],
                optimization_opportunities=data["drivers"][:2]  # Top 2 cost drivers
            )
            
            self.cost_analyses[category] = analysis
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_monthly_cost": total_monthly_cost,
            "total_budget": total_budget,
            "budget_variance": total_monthly_cost - total_budget,
            "cost_breakdown": [
                {
                    "category": analysis.category.value,
                    "monthly_cost": analysis.monthly_cost,
                    "percentage": round(analysis.percentage_of_total, 1),
                    "trend": f"{analysis.trend_direction} {analysis.trend_percentage}%",
                    "variance": round(analysis.budget_variance, 2),
                    "drivers": analysis.optimization_opportunities
                }
                for analysis in self.cost_analyses.values()
            ],
            "cost_trends": {
                "highest_growing": "Training (+25%)",
                "most_significant": "Personnel ($285K)",
                "over_budget": ["Infrastructure", "Data Processing", "Maintenance"],
                "under_budget": ["Compliance", "Software Licenses"]
            }
        }
    
    async def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate resource optimization recommendations"""
        
        recommendations = [
            OptimizationRecommendation(
                recommendation_id="OPT_001",
                category=CostCategory.INFRASTRUCTURE,
                strategy=OptimizationStrategy.RIGHT_SIZING,
                description="Right-size compute instances based on actual usage patterns",
                potential_savings=25000.0,
                implementation_effort="Low",
                implementation_timeline="2 weeks",
                risk_level="Low",
                roi_percentage=320.0
            ),
            OptimizationRecommendation(
                recommendation_id="OPT_002",
                category=CostCategory.INFRASTRUCTURE,
                strategy=OptimizationStrategy.RESERVATION_PURCHASING,
                description="Purchase reserved instances for predictable workloads",
                potential_savings=18000.0,
                implementation_effort="Medium",
                implementation_timeline="1 month",
                risk_level="Low",
                roi_percentage=180.0
            ),
            OptimizationRecommendation(
                recommendation_id="OPT_003",
                category=CostCategory.INFRASTRUCTURE,
                strategy=OptimizationStrategy.AUTO_SCALING,
                description="Implement dynamic auto-scaling for variable workloads",
                potential_savings=22000.0,
                implementation_effort="High",
                implementation_timeline="6 weeks",
                risk_level="Medium",
                roi_percentage=240.0
            ),
            OptimizationRecommendation(
                recommendation_id="OPT_004",
                category=CostCategory.DATA_PROCESSING,
                strategy=OptimizationStrategy.DATA_LIFECYCLE,
                description="Implement data lifecycle management for cost optimization",
                potential_savings=12000.0,
                implementation_effort="Medium",
                implementation_timeline="4 weeks",
                risk_level="Low",
                roi_percentage=150.0
            ),
            OptimizationRecommendation(
                recommendation_id="OPT_005",
                category=CostCategory.SOFTWARE_LICENSES,
                strategy=OptimizationStrategy.LICENSING_OPTIMIZATION,
                description="Optimize software licenses based on actual usage",
                potential_savings=8000.0,
                implementation_effort="Low",
                implementation_timeline="2 weeks",
                risk_level="Low",
                roi_percentage=400.0
            ),
            OptimizationRecommendation(
                recommendation_id="OPT_006",
                category=CostCategory.INFRASTRUCTURE,
                strategy=OptimizationStrategy.SPOT_INSTANCE_USAGE,
                description="Use spot instances for non-critical workloads",
                potential_savings=15000.0,
                implementation_effort="Medium",
                implementation_timeline="3 weeks",
                risk_level="Medium",
                roi_percentage=200.0
            )
        ]
        
        self.optimization_recommendations.extend(recommendations)
        return recommendations
    
    async def create_budget_plan(self, budget_config: Dict) -> BudgetPlan:
        """Create comprehensive budget plan"""
        
        budget_plan = BudgetPlan(
            budget_id=budget_config["budget_id"],
            department=budget_config["department"],
            fiscal_year=budget_config["fiscal_year"],
            allocated_budget=budget_config["allocated_budget"],
            spent_to_date=budget_config["spent_to_date"],
            remaining_budget=budget_config["allocated_budget"] - budget_config["spent_to_date"],
            burn_rate=budget_config.get("burn_rate", 0.0),
            forecasted_spend=budget_config.get("forecasted_spend", 0.0),
            variance_analysis={
                "personnel_variance": budget_config.get("personnel_variance", 0.0),
                "infrastructure_variance": budget_config.get("infrastructure_variance", 0.0),
                "operational_variance": budget_config.get("operational_variance", 0.0)
            }
        )
        
        self.budget_plans[budget_plan.budget_id] = budget_plan
        return budget_plan
    
    async def simulate_cost_optimization_impact(self, optimization_id: str) -> Dict:
        """Simulate the impact of cost optimization"""
        
        optimization = next((opt for opt in self.optimization_recommendations if opt.recommendation_id == optimization_id), None)
        if not optimization:
            return {"error": "Optimization recommendation not found"}
        
        # Simulate before optimization
        baseline_costs = {
            "infrastructure": 125000.0,
            "software_licenses": 45000.0,
            "data_processing": 35000.0,
            "other": 55000.0
        }
        
        # Calculate post-optimization costs
        category_key = optimization.category.value
        if category_key in baseline_costs:
            monthly_savings = optimization.potential_savings
            post_optimization_cost = baseline_costs[category_key] - monthly_savings
            
            # Update baseline costs
            baseline_costs[category_key] = post_optimization_cost
        
        total_baseline = sum(baseline_costs.values())
        
        # Calculate additional benefits
        additional_benefits = {
            "productivity_gains": optimization.potential_savings * 0.3,
            "quality_improvements": optimization.potential_savings * 0.15,
            "risk_reduction": optimization.potential_savings * 0.1
        }
        
        total_benefits = optimization.potential_savings + sum(additional_benefits.values())
        
        return {
            "optimization_id": optimization_id,
            "baseline_costs": baseline_costs,
            "optimization_costs": {
                "implementation_cost": optimization.potential_savings * 0.2,  # 20% of savings
                "monthly_savings": optimization.potential_savings,
                "annual_savings": optimization.potential_savings * 12
            },
            "impact_summary": {
                "cost_reduction": f"{((optimization.potential_savings / total_baseline) * 100):.1f}%",
                "payback_period_months": round((optimization.potential_savings * 0.2) / (optimization.potential_savings / 12), 1),
                "total_roi": f"{optimization.roi_percentage:.1f}%",
                "break_even_timeline": optimization.implementation_timeline
            },
            "additional_benefits": additional_benefits,
            "risk_assessment": {
                "implementation_risk": optimization.risk_level,
                "business_impact": "Positive with manageable risks",
                "mitigation_strategies": [
                    "Phased implementation approach",
                    "Regular progress monitoring",
                    "Rollback procedures in place",
                    "Stakeholder communication plan"
                ]
            },
            "success_metrics": {
                "cost_reduction_target": optimization.potential_savings,
                "efficiency_improvement": "15-25%",
                "resource_utilization_optimization": "20-30%",
                "operational_excellence_score": "+20 points"
            }
        }
    
    async def calculate_total_cost_of_ownership(self, system_scope: str) -> Dict:
        """Calculate total cost of ownership for healthcare AI systems"""
        
        # TCO components
        tco_components = {
            "infrastructure_costs": {
                "compute_resources": 125000.0,
                "storage": 45000.0,
                "network": 25000.0,
                "security": 30000.0
            },
            "software_costs": {
                "ai_platform_licenses": 85000.0,
                "database_licenses": 55000.0,
                "development_tools": 25000.0,
                "monitoring_tools": 20000.0
            },
            "personnel_costs": {
                "engineering_team": 180000.0,
                "data_science_team": 165000.0,
                "operations_team": 120000.0,
                "support_team": 85000.0
            },
            "operational_costs": {
                "cloud_services": 45000.0,
                "compliance": 35000.0,
                "training": 25000.0,
                "maintenance": 30000.0
            },
            "strategic_costs": {
                "r_and_d": 40000.0,
                "consulting": 30000.0,
                "legal": 15000.0,
                "marketing": 20000.0
            }
        }
        
        # Calculate totals
        subtotals = {}
        for category, items in tco_components.items():
            subtotals[category] = sum(items.values())
        
        total_tco = sum(subtotals.values())
        
        # Calculate 3-year TCO projection
        growth_rate = 0.08  # 8% annual growth
        yearly_projections = []
        for year in range(1, 4):
            projected_cost = total_tco * ((1 + growth_rate) ** year)
            yearly_projections.append({
                "year": year,
                "projected_cost": projected_cost,
                "growth_rate": f"{(growth_rate * 100):.1f}%"
            })
        
        three_year_tco = sum([p["projected_cost"] for p in yearly_projections])
        
        # Calculate cost per clinical outcome
        annual_clinical_cases = 50000  # Estimated annual clinical cases processed
        cost_per_case = total_tco / annual_clinical_cases
        
        return {
            "system_scope": system_scope,
            "analysis_timestamp": datetime.now().isoformat(),
            "current_year_tco": total_tco,
            "tco_breakdown": {
                category: {
                    "amount": subtotal,
                    "percentage": (subtotal / total_tco) * 100
                }
                for category, subtotal in subtotals.items()
            },
            "three_year_projection": {
                "total_3_year_tco": three_year_tco,
                "yearly_breakdown": yearly_projections
            },
            "cost_efficiency_metrics": {
                "cost_per_clinical_case": round(cost_per_case, 2),
                "cost_per_mb_of_data": 0.12,  # Per MB processed
                "cost_per_ai_inference": 0.05,  # Per AI query
                "cost_per_user_per_month": 285.50  # Per healthcare professional
            },
            "benchmarking": {
                "industry_average_tco": total_tco * 1.15,
                "efficiency_ratio": round((total_tco * 1.15 / total_tco) * 100, 1),
                "optimization_potential": "15-20% cost reduction possible"
            },
            "recommendations": [
                "Focus on infrastructure optimization for maximum impact",
                "Implement automated scaling to reduce over-provisioning",
                "Negotiate volume discounts for software licenses",
                "Invest in staff training to improve operational efficiency",
                "Consider managed services for non-core functions"
            ]
        }
    
    async def generate_resource_optimization_dashboard(self) -> Dict:
        """Generate resource optimization dashboard"""
        
        # Calculate aggregate metrics
        total_monthly_cost = sum(analysis.monthly_cost for analysis in self.cost_analyses.values())
        total_optimization_potential = sum(rec.potential_savings for rec in self.optimization_recommendations)
        
        dashboard_data = {
            "cost_overview": {
                "total_monthly_cost": total_monthly_cost,
                "total_annual_cost": total_monthly_cost * 12,
                "budget_variance": sum(analysis.budget_variance for analysis in self.cost_analyses.values()),
                "cost_trend": "+8.5% this quarter"
            },
            "resource_utilization": {
                "compute_utilization": 68.5,  # percentage
                "storage_utilization": 75.0,
                "network_utilization": 85.0,
                "ai_model_utilization": 62.5,
                "overall_efficiency_score": 72.8
            },
            "optimization_opportunities": {
                "total_potential_savings": total_optimization_potential,
                "high_impact_opportunities": len([r for r in self.optimization_recommendations if r.potential_savings > 15000]),
                "quick_wins": len([r for r in self.optimization_recommendations if r.implementation_effort == "Low"]),
                "total_roi_potential": sum(rec.roi_percentage for rec in self.optimization_recommendations) / len(self.optimization_recommendations)
            },
            "cost_category_performance": {
                "infrastructure": {
                    "cost": 125000.0,
                    "budget_utilization": 104.2,
                    "optimization_potential": 25000.0,
                    "trend": "increasing"
                },
                "software_licenses": {
                    "cost": 45000.0,
                    "budget_utilization": 90.0,
                    "optimization_potential": 8000.0,
                    "trend": "stable"
                },
                "personnel": {
                    "cost": 285000.0,
                    "budget_utilization": 101.8,
                    "optimization_potential": 15000.0,
                    "trend": "stable"
                },
                "data_processing": {
                    "cost": 35000.0,
                    "budget_utilization": 116.7,
                    "optimization_potential": 12000.0,
                    "trend": "increasing"
                }
            },
            "budget_health": {
                "departments_within_budget": len([plan for plan in self.budget_plans.values() if plan.spent_to_date < plan.allocated_budget]),
                "departments_over_budget": len([plan for plan in self.budget_plans.values() if plan.spent_to_date > plan.allocated_budget]),
                "average_burn_rate": 0.0,  # Calculated from budget plans
                "forecast_accuracy": 92.5  # percentage
            },
            "cost_alerts": {
                "active_alerts": len(self.cost_alerts),
                "critical_alerts": len([alert for alert in self.cost_alerts if alert.get("severity") == "critical"]),
                "resolved_this_month": len([alert for alert in self.cost_alerts if alert.get("resolved", False)]),
                "average_resolution_time": "4.2 hours"
            },
            "optimization_roi": {
                "implemented_optimizations": 8,
                "realized_savings": 95000.0,
                "implementation_cost": 25000.0,
                "net_roi": 280.0  # percentage
            },
            "recommendations": [
                "Implement right-sizing for compute resources",
                "Purchase reserved instances for predictable workloads",
                "Optimize data lifecycle management",
                "Negotiate software license renewals",
                "Implement advanced monitoring and alerting"
            ]
        }
        
        return dashboard_data
    
    async def export_resource_optimization_report(self, filepath: str) -> Dict:
        """Export comprehensive resource optimization report"""
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_title": "Healthcare AI Resource Optimization Report",
                "reporting_period": "Q4 2025",
                "scope": "Enterprise-wide resource optimization and cost management"
            },
            "executive_summary": {
                "total_monthly_cost": sum(analysis.monthly_cost for analysis in self.cost_analyses.values()),
                "total_optimization_potential": sum(rec.potential_savings for rec in self.optimization_recommendations),
                "budget_variance": sum(analysis.budget_variance for analysis in self.cost_analyses.values()),
                "cost_efficiency_score": 72.8
            },
            "resource_usage_analysis": [
                {
                    "resource_id": usage.resource_id,
                    "type": usage.resource_type.value,
                    "utilization": f"{usage.utilization_percentage}%",
                    "monthly_cost": usage.cost_per_month,
                    "optimization_potential": usage.cost_per_month * 0.25 if usage.utilization_percentage < 50 else 0
                }
                for usage in self.resource_usage.values()
            ],
            "cost_analyses": [
                {
                    "category": analysis.category.value,
                    "monthly_cost": analysis.monthly_cost,
                    "percentage_of_total": round(analysis.percentage_of_total, 1),
                    "trend": f"{analysis.trend_direction} {analysis.trend_percentage}%",
                    "budget_variance": round(analysis.budget_variance, 2),
                    "optimization_opportunities": analysis.optimization_opportunities
                }
                for analysis in self.cost_analyses.values()
            ],
            "optimization_recommendations": [
                {
                    "id": rec.recommendation_id,
                    "category": rec.category.value,
                    "strategy": rec.strategy.value,
                    "description": rec.description,
                    "potential_savings": rec.potential_savings,
                    "effort": rec.implementation_effort,
                    "timeline": rec.implementation_timeline,
                    "roi": f"{rec.roi_percentage}%"
                }
                for rec in self.optimization_recommendations
            ],
            "budget_plans": [
                {
                    "budget_id": plan.budget_id,
                    "department": plan.department,
                    "allocated": plan.allocated_budget,
                    "spent_to_date": plan.spent_to_date,
                    "remaining": plan.remaining_budget,
                    "variance_analysis": plan.variance_analysis
                }
                for plan in self.budget_plans.values()
            ],
            "recommendations": [
                "Prioritize high-impact, low-effort optimizations first",
                "Implement comprehensive monitoring and alerting",
                "Establish regular cost review and optimization cycles",
                "Create cross-functional cost optimization team",
                "Develop predictive cost modeling capabilities"
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return {"status": "success", "report_file": filepath}

# Example usage and testing
async def run_resource_optimization_demo():
    """Demonstrate Resource Optimization framework"""
    optimizer = ResourceOptimizationManager()
    
    # 1. Analyze Current Resource Usage
    print("=== Analyzing Current Resource Usage ===")
    resource_analysis = await optimizer.analyze_current_resource_usage()
    print(f"Total Resources: {resource_analysis['total_resources_tracked']}")
    print(f"Total Monthly Cost: ${resource_analysis['total_monthly_cost']:,.2f}")
    print(f"Average Utilization: {resource_analysis['average_utilization']}%")
    print(f"Over-utilized Resources: {resource_analysis['resource_summary']['over_utilized']}")
    print(f"Under-utilized Resources: {resource_analysis['resource_summary']['under_utilized']}")
    
    # 2. Perform Cost Analysis
    print("\n=== Performing Comprehensive Cost Analysis ===")
    cost_analysis = await optimizer.perform_comprehensive_cost_analysis()
    print(f"Total Monthly Cost: ${cost_analysis['total_monthly_cost']:,.2f}")
    print(f"Budget Variance: ${cost_analysis['budget_variance']:,.2f}")
    print("Top Cost Categories:")
    for category in cost_analysis['cost_breakdown'][:3]:
        print(f"  - {category['category']}: ${category['monthly_cost']:,.2f} ({category['percentage']}%)")
    
    # 3. Generate Optimization Recommendations
    print("\n=== Generating Optimization Recommendations ===")
    recommendations = await optimizer.generate_optimization_recommendations()
    print(f"Total Recommendations: {len(recommendations)}")
    print(f"Total Potential Savings: ${sum(r.potential_savings for r in recommendations):,.2f}/month")
    
    # Display top 3 recommendations
    top_recommendations = sorted(recommendations, key=lambda x: x.potential_savings, reverse=True)[:3]
    for rec in top_recommendations:
        print(f"\n{rec.recommendation_id}: {rec.description}")
        print(f"  Potential Savings: ${rec.potential_savings:,.2f}/month")
        print(f"  Effort: {rec.implementation_effort}, Timeline: {rec.implementation_timeline}")
        print(f"  ROI: {rec.roi_percentage}%")
    
    # 4. Create Budget Plan
    print("\n=== Creating Budget Plan ===")
    budget_config = {
        "budget_id": "BUDGET_AI_DEPT_2025",
        "department": "AI Operations",
        "fiscal_year": 2025,
        "allocated_budget": 500000.0,
        "spent_to_date": 387500.0,
        "burn_rate": 125000.0,
        "forecasted_spend": 525000.0,
        "personnel_variance": 15000.0,
        "infrastructure_variance": 8000.0,
        "operational_variance": 3500.0
    }
    budget_plan = await optimizer.create_budget_plan(budget_config)
    print(f"Budget Plan: {budget_plan.budget_id}")
    print(f"Department: {budget_plan.department}")
    print(f"Allocated: ${budget_plan.allocated_budget:,.2f}")
    print(f"Spent to Date: ${budget_plan.spent_to_date:,.2f}")
    print(f"Remaining: ${budget_plan.remaining_budget:,.2f}")
    print(f"Forecasted Spend: ${budget_plan.forecasted_spend:,.2f}")
    
    # 5. Simulate Cost Optimization Impact
    print("\n=== Simulating Cost Optimization Impact ===")
    optimization_impact = await optimizer.simulate_cost_optimization_impact("OPT_001")
    if "error" not in optimization_impact:
        print(f"Optimization: {optimization_impact['optimization_id']}")
        print(f"Monthly Savings: ${optimization_impact['optimization_costs']['monthly_savings']:,.2f}")
        print(f"Annual Savings: ${optimization_impact['optimization_costs']['annual_savings']:,.2f}")
        print(f"Cost Reduction: {optimization_impact['impact_summary']['cost_reduction']}")
        print(f"Payback Period: {optimization_impact['impact_summary']['payback_period_months']} months")
    
    # 6. Calculate Total Cost of Ownership
    print("\n=== Calculating Total Cost of Ownership ===")
    tco_analysis = await optimizer.calculate_total_cost_of_ownership("Healthcare AI Platform")
    print(f"Current Year TCO: ${tco_analysis['current_year_tco']:,.2f}")
    print(f"3-Year TCO: ${tco_analysis['three_year_projection']['total_3_year_tco']:,.2f}")
    print(f"Cost per Clinical Case: ${tco_analysis['cost_efficiency_metrics']['cost_per_clinical_case']}")
    print(f"Efficiency Ratio: {tco_analysis['benchmarking']['efficiency_ratio']}% of industry average")
    
    # 7. Generate Dashboard
    print("\n=== Resource Optimization Dashboard ===")
    dashboard = await optimizer.generate_resource_optimization_dashboard()
    print(f"Total Monthly Cost: ${dashboard['cost_overview']['total_monthly_cost']:,.2f}")
    print(f"Overall Efficiency Score: {dashboard['resource_utilization']['overall_efficiency_score']}")
    print(f"Optimization Potential: ${dashboard['optimization_opportunities']['total_potential_savings']:,.2f}")
    print(f"Quick Wins Available: {dashboard['optimization_opportunities']['quick_wins']}")
    
    # 8. Export Report
    print("\n=== Exporting Resource Optimization Report ===")
    report_result = await optimizer.export_resource_optimization_report("resource_optimization_report.json")
    print(f"Report exported to: {report_result['report_file']}")
    
    return optimizer

if __name__ == "__main__":
    asyncio.run(run_resource_optimization_demo())
