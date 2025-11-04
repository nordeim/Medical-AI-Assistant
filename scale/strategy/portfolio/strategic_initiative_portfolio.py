"""
Strategic Initiative Portfolio Management and Optimization System
Comprehensive portfolio governance, resource allocation, and performance optimization
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class InitiativeStatus(Enum):
    PLANNING = "planning"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"

class InitiativePriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class PortfolioStrategy(Enum):
    AGGRESSIVE_GROWTH = "aggressive_growth"
    BALANCED_GROWTH = "balanced_growth"
    CONSERVATIVE = "conservative"
    TRANSFORMATION = "transformation"
    INNOVATION_FOCUSED = "innovation_focused"
    EFFICIENCY_DRIVEN = "efficiency_driven"

class ResourceType(Enum):
    HUMAN = "human"
    FINANCIAL = "financial"
    TECHNOLOGICAL = "technological"
    INFRASTRUCTURE = "infrastructure"
    MARKET = "market"

@dataclass
class StrategicInitiative:
    """Strategic Initiative Definition"""
    initiative_id: str
    name: str
    description: str
    strategic_objectives: List[str]
    status: InitiativeStatus
    priority: InitiativePriority
    portfolio_category: str  # Growth, Efficiency, Innovation, etc.
    
    # Resource Requirements
    resource_requirements: Dict[str, Dict[str, float]]  # {resource_type: {amount, cost, availability}}
    
    # Timeline and Milestones
    start_date: datetime
    planned_end_date: datetime
    actual_end_date: Optional[datetime]
    critical_milestones: List[Dict[str, Any]]  # {milestone, date, status, dependencies}
    
    # Financial Metrics
    estimated_investment: float
    expected_roi: float
    payback_period: int  # months
    npv: float
    irr: float
    risk_score: float
    
    # Performance Metrics
    success_criteria: List[str]
    kpis: Dict[str, float]  # {kpi_name: target_value}
    current_performance: Dict[str, float]  # {kpi_name: current_value}
    progress_percentage: float
    
    # Dependencies and Relationships
    dependencies: List[str]  # Initiative IDs
    complementary_initiatives: List[str]
    conflicting_initiatives: List[str]
    
    # Stakeholder Impact
    key_stakeholders: List[str]
    impact_assessment: Dict[str, float]  # {stakeholder_group: impact_level}
    
    # Risk and Contingency
    risk_factors: List[str]
    mitigation_strategies: List[str]
    contingency_plans: List[str]

@dataclass
class PortfolioMetrics:
    """Portfolio Performance Metrics"""
    total_investment: float
    expected_total_return: float
    portfolio_roi: float
    average_payback_period: float
    risk_adjusted_return: float
    resource_utilization: Dict[str, float]
    initiative_success_rate: float
    strategic_alignment_score: float

class StrategicInitiativePortfolioManager:
    """
    Strategic Initiative Portfolio Management and Optimization System
    """
    
    def __init__(self, organization_name: str):
        self.organization_name = organization_name
        self.initiatives: List[StrategicInitiative] = []
        self.portfolio_metrics: Optional[PortfolioMetrics] = None
        self.current_strategy: Optional[PortfolioStrategy] = None
        self.resource_constraints: Dict[str, float] = {}
        self.portfolio_optimization_results: Dict[str, Any] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
    def add_initiative(self,
                     initiative: StrategicInitiative) -> StrategicInitiative:
        """Add strategic initiative to portfolio"""
        
        self.initiatives.append(initiative)
        return initiative
    
    def create_portfolio(self,
                        portfolio_strategy: PortfolioStrategy,
                        strategic_objectives: List[str],
                        resource_constraints: Dict[str, float],
                        investment_budget: float) -> Dict[str, Any]:
        """Create strategic portfolio based on strategy and constraints"""
        
        self.current_strategy = portfolio_strategy
        self.resource_constraints = resource_constraints
        
        # Filter initiatives based on strategy
        portfolio_initiatives = self._select_portfolio_initiatives(portfolio_strategy, strategic_objectives)
        
        # Optimize portfolio composition
        optimized_portfolio = self._optimize_portfolio_composition(portfolio_initiatives, investment_budget)
        
        # Create implementation roadmap
        implementation_roadmap = self._create_implementation_roadmap(optimized_portfolio)
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(optimized_portfolio)
        
        portfolio = {
            "portfolio_strategy": portfolio_strategy.value,
            "strategic_objectives": strategic_objectives,
            "resource_constraints": resource_constraints,
            "investment_budget": investment_budget,
            "selected_initiatives": [asdict(initiative) for initiative in optimized_portfolio],
            "implementation_roadmap": implementation_roadmap,
            "portfolio_metrics": asdict(portfolio_metrics),
            "optimization_results": self._generate_optimization_insights(optimized_portfolio),
            "risk_assessment": self._assess_portfolio_risks(optimized_portfolio),
            "governance_framework": self._define_portfolio_governance()
        }
        
        self.portfolio_metrics = portfolio_metrics
        self.portfolio_optimization_results = portfolio
        return portfolio
    
    def _select_portfolio_initiatives(self,
                                    strategy: PortfolioStrategy,
                                    objectives: List[str]) -> List[StrategicInitiative]:
        """Select initiatives based on portfolio strategy"""
        
        if not self.initiatives:
            # Generate sample initiatives if none exist
            self._generate_sample_initiatives()
        
        selected_initiatives = []
        
        # Strategy-based selection criteria
        strategy_criteria = {
            PortfolioStrategy.AGGRESSIVE_GROWTH: {
                "priority_filter": [InitiativePriority.CRITICAL, InitiativePriority.HIGH],
                "category_preference": ["growth", "innovation", "market_expansion"],
                "roi_threshold": 2.0,
                "risk_tolerance": 0.7
            },
            PortfolioStrategy.BALANCED_GROWTH: {
                "priority_filter": [InitiativePriority.CRITICAL, InitiativePriority.HIGH, InitiativePriority.MEDIUM],
                "category_preference": ["growth", "efficiency", "innovation"],
                "roi_threshold": 1.5,
                "risk_tolerance": 0.5
            },
            PortfolioStrategy.CONSERVATIVE: {
                "priority_filter": [InitiativePriority.CRITICAL, InitiativePriority.HIGH],
                "category_preference": ["efficiency", "optimization"],
                "roi_threshold": 1.2,
                "risk_tolerance": 0.3
            },
            PortfolioStrategy.TRANSFORMATION: {
                "priority_filter": [InitiativePriority.CRITICAL],
                "category_preference": ["transformation", "innovation", "digital"],
                "roi_threshold": 1.8,
                "risk_tolerance": 0.8
            }
        }
        
        criteria = strategy_criteria.get(strategy, strategy_criteria[PortfolioStrategy.BALANCED_GROWTH])
        
        for initiative in self.initiatives:
            # Priority filter
            if initiative.priority not in criteria["priority_filter"]:
                continue
            
            # ROI threshold
            if initiative.expected_roi < criteria["roi_threshold"]:
                continue
            
            # Risk assessment
            if initiative.risk_score > criteria["risk_tolerance"]:
                continue
            
            # Strategic alignment
            if self._assess_strategic_alignment(initiative, objectives) < 0.6:
                continue
            
            selected_initiatives.append(initiative)
        
        return selected_initiatives
    
    def _generate_sample_initiatives(self):
        """Generate sample strategic initiatives for demonstration"""
        
        sample_initiatives = [
            StrategicInitiative(
                initiative_id="INIT_001",
                name="Digital Transformation Program",
                description="Comprehensive digital transformation of core business processes",
                strategic_objectives=["digitalization", "efficiency", "customer_experience"],
                status=InitiativeStatus.APPROVED,
                priority=InitiativePriority.CRITICAL,
                portfolio_category="transformation",
                resource_requirements={
                    ResourceType.HUMAN.value: {"amount": 50, "cost": 5.0, "availability": 0.8},
                    ResourceType.TECHNOLOGICAL.value: {"amount": 1, "cost": 10.0, "availability": 0.9},
                    ResourceType.FINANCIAL.value: {"amount": 1, "cost": 20.0, "availability": 0.7}
                },
                start_date=datetime.now(),
                planned_end_date=datetime.now() + timedelta(days=730),  # 24 months
                actual_end_date=None,
                critical_milestones=[
                    {"milestone": "Phase 1 Planning Complete", "date": datetime.now() + timedelta(days=90), "status": "on_track"},
                    {"milestone": "Technology Implementation", "date": datetime.now() + timedelta(days=365), "status": "planned"},
                    {"milestone": "Full Deployment", "date": datetime.now() + timedelta(days=730), "status": "planned"}
                ],
                estimated_investment=35.0,
                expected_roi=2.5,
                payback_period=24,
                npv=50.0,
                irr=0.25,
                risk_score=0.6,
                success_criteria=["Technology adoption >90%", "Cost reduction >20%", "Customer satisfaction >4.5"],
                kpis={"digital_adoption": 0.9, "cost_reduction": 0.2, "customer_satisfaction": 4.5},
                current_performance={"digital_adoption": 0.4, "cost_reduction": 0.1, "customer_satisfaction": 4.2},
                progress_percentage=35.0,
                dependencies=[],
                complementary_initiatives=["INIT_002", "INIT_003"],
                conflicting_initiatives=[],
                key_stakeholders=["Executive Team", "IT Department", "Operations"],
                impact_assessment={"employees": 0.8, "customers": 0.7, "shareholders": 0.9}
            ),
            StrategicInitiative(
                initiative_id="INIT_002",
                name="Market Expansion Initiative",
                description="Expansion into new geographic markets and customer segments",
                strategic_objectives=["growth", "market_expansion", "revenue"],
                status=InitiativeStatus.IN_PROGRESS,
                priority=InitiativePriority.HIGH,
                portfolio_category="growth",
                resource_requirements={
                    ResourceType.HUMAN.value: {"amount": 30, "cost": 3.0, "availability": 0.7},
                    ResourceType.MARKET.value: {"amount": 3, "cost": 5.0, "availability": 0.6},
                    ResourceType.FINANCIAL.value: {"amount": 1, "cost": 15.0, "availability": 0.8}
                },
                start_date=datetime.now() - timedelta(days=180),
                planned_end_date=datetime.now() + timedelta(days=365),
                actual_end_date=None,
                critical_milestones=[
                    {"milestone": "Market Research Complete", "date": datetime.now() - timedelta(days=90), "status": "completed"},
                    {"milestone": "Market Entry", "date": datetime.now() + timedelta(days=90), "status": "in_progress"},
                    {"milestone": "Revenue Targets Met", "date": datetime.now() + timedelta(days=365), "status": "planned"}
                ],
                estimated_investment=25.0,
                expected_roi=3.0,
                payback_period=18,
                npv=40.0,
                irr=0.30,
                risk_score=0.5,
                success_criteria=["Market share >15%", "Revenue growth >25%", "Profitability >20%"],
                kpis={"market_share": 0.15, "revenue_growth": 0.25, "profitability": 0.2},
                current_performance={"market_share": 0.08, "revenue_growth": 0.18, "profitability": 0.12},
                progress_percentage=60.0,
                dependencies=["INIT_001"],
                complementary_initiatives=["INIT_003"],
                conflicting_initiatives=[],
                key_stakeholders=["Sales Team", "Marketing", "Regional Management"],
                impact_assessment={"employees": 0.6, "customers": 0.5, "shareholders": 0.8}
            ),
            StrategicInitiative(
                initiative_id="INIT_003",
                name="Innovation Lab Program",
                description="Establish innovation capabilities and R&D excellence",
                strategic_objectives=["innovation", "competitive_advantage", "future_growth"],
                status=InitiativeStatus.PLANNING,
                priority=InitiativePriority.HIGH,
                portfolio_category="innovation",
                resource_requirements={
                    ResourceType.HUMAN.value: {"amount": 25, "cost": 4.0, "availability": 0.8},
                    ResourceType.TECHNOLOGICAL.value: {"amount": 1, "cost": 8.0, "availability": 0.9},
                    ResourceType.FINANCIAL.value: {"amount": 1, "cost": 12.0, "availability": 0.7}
                },
                start_date=datetime.now() + timedelta(days=60),
                planned_end_date=datetime.now() + timedelta(days=1095),  # 36 months
                actual_end_date=None,
                critical_milestones=[
                    {"milestone": "Lab Setup", "date": datetime.now() + timedelta(days=180), "status": "planned"},
                    {"milestone": "First Innovation Projects", "date": datetime.now() + timedelta(days=365), "status": "planned"},
                    {"milestone": "Commercialization", "date": datetime.now() + timedelta(days=730), "status": "planned"}
                ],
                estimated_investment=30.0,
                expected_roi=4.0,
                payback_period=30,
                npv=60.0,
                irr=0.35,
                risk_score=0.7,
                success_criteria=["3+ patents filed", "2+ products launched", "Innovation pipeline >10 projects"],
                kpis={"patents_filed": 3, "products_launched": 2, "pipeline_projects": 10},
                current_performance={"patents_filed": 0, "products_launched": 0, "pipeline_projects": 5},
                progress_percentage=5.0,
                dependencies=[],
                complementary_initiatives=["INIT_001", "INIT_002"],
                conflicting_initiatives=[],
                key_stakeholders=["R&D Team", "Product Management", "Innovation Office"],
                impact_assessment={"employees": 0.7, "customers": 0.4, "shareholders": 0.8}
            )
        ]
        
        self.initiatives = sample_initiatives
    
    def _assess_strategic_alignment(self, initiative: StrategicInitiative, objectives: List[str]) -> float:
        """Assess strategic alignment score for initiative"""
        
        # Count matching objectives
        matching_objectives = len(set(initiative.strategic_objectives) & set(objectives))
        alignment_score = matching_objectives / max(len(objectives), 1)
        
        # Adjust for priority level
        priority_weights = {
            InitiativePriority.CRITICAL: 1.0,
            InitiativePriority.HIGH: 0.8,
            InitiativePriority.MEDIUM: 0.6,
            InitiativePriority.LOW: 0.4
        }
        
        alignment_score *= priority_weights.get(initiative.priority, 0.6)
        
        return min(1.0, alignment_score)
    
    def _optimize_portfolio_composition(self,
                                      candidate_initiatives: List[StrategicInitiative],
                                      budget: float) -> List[StrategicInitiative]:
        """Optimize portfolio composition based on multiple criteria"""
        
        if not candidate_initiatives:
            return []
        
        # Calculate optimization scores for each initiative
        optimization_scores = {}
        
        for initiative in candidate_initiatives:
            score = self._calculate_optimization_score(initiative, budget)
            optimization_scores[initiative.initiative_id] = score
        
        # Sort initiatives by optimization score
        sorted_initiatives = sorted(candidate_initiatives, 
                                  key=lambda x: optimization_scores[x.initiative_id], 
                                  reverse=True)
        
        # Select initiatives within budget constraints
        selected_initiatives = []
        total_investment = 0
        
        for initiative in sorted_initiatives:
            if total_investment + initiative.estimated_investment <= budget:
                selected_initiatives.append(initiative)
                total_investment += initiative.estimated_investment
            elif len(selected_initiatives) < 3:  # Ensure minimum portfolio size
                # Replace lowest scoring initiative if new one has significantly better score
                if selected_initiatives and optimization_scores[initiative.initiative_id] > min(
                    optimization_scores[inv.initiative_id] for inv in selected_initiatives
                ):
                    # Remove lowest scoring initiative
                    lowest_scoring = min(selected_initiatives, 
                                       key=lambda x: optimization_scores[x.initiative_id])
                    selected_initiatives.remove(lowest_scoring)
                    total_investment -= lowest_scoring.estimated_investment
                    
                    # Add new initiative
                    selected_initiatives.append(initiative)
                    total_investment += initiative.estimated_investment
        
        return selected_initiatives
    
    def _calculate_optimization_score(self, initiative: StrategicInitiative, total_budget: float) -> float:
        """Calculate optimization score for initiative selection"""
        
        # Financial metrics (40% weight)
        roi_score = min(initiative.expected_roi / 3.0, 1.0)  # Normalize to 1.0
        payback_score = max(0, (36 - initiative.payback_period) / 36)  # 3-year payback baseline
        npv_score = min(initiative.npv / 100.0, 1.0)  # Normalize to $100M baseline
        
        financial_score = (roi_score + payback_score + npv_score) / 3
        
        # Strategic alignment (25% weight)
        # Use current performance and strategic objectives for assessment
        alignment_score = initiative.progress_percentage / 100.0 if initiative.progress_percentage > 0 else 0.5
        
        # Risk-adjusted return (20% weight)
        risk_score = 1.0 - initiative.risk_score
        risk_adjusted_return = initiative.expected_roi * risk_score
        risk_adjusted_score = min(risk_adjusted_return / 3.0, 1.0)
        
        # Resource efficiency (15% weight)
        resource_efficiency = 1.0 - (initiative.estimated_investment / total_budget)
        
        # Combine scores
        optimization_score = (
            0.40 * financial_score +
            0.25 * alignment_score +
            0.20 * risk_adjusted_score +
            0.15 * resource_efficiency
        )
        
        return optimization_score
    
    def _create_implementation_roadmap(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Create implementation roadmap for portfolio"""
        
        if not initiatives:
            return {}
        
        # Sort initiatives by start date
        sorted_initiatives = sorted(initiatives, key=lambda x: x.start_date)
        
        # Create phases
        phases = []
        phase_duration = 6  # months per phase
        
        current_phase_start = min(init.start_date for init in initiatives)
        phase_number = 1
        
        while current_phase_start < max(init.planned_end_date for init in initiatives):
            phase_end = current_phase_start + timedelta(days=phase_duration * 30)
            
            # Find initiatives active in this phase
            active_initiatives = [
                init for init in initiatives
                if init.start_date <= phase_end and init.planned_end_date >= current_phase_start
            ]
            
            phase = {
                "phase_number": phase_number,
                "start_date": current_phase_start.isoformat(),
                "end_date": phase_end.isoformat(),
                "duration_months": phase_duration,
                "active_initiatives": [init.initiative_id for init in active_initiatives],
                "phase_focus": self._get_phase_focus(active_initiatives, phase_number),
                "critical_milestones": self._extract_phase_milestones(active_initiatives, current_phase_start, phase_end),
                "resource_allocation": self._calculate_phase_resources(active_initiatives),
                "success_criteria": self._define_phase_success_criteria(active_initiatives)
            }
            
            phases.append(phase)
            current_phase_start = phase_end
            phase_number += 1
        
        return {
            "total_phases": len(phases),
            "total_duration_months": len(phases) * phase_duration,
            "phases": phases,
            "critical_path": self._identify_critical_path(initiatives),
            "resource_optimization": self._optimize_resource_allocation(phases),
            "risk_mitigation": self._plan_phase_risk_mitigation(phases)
        }
    
    def _get_phase_focus(self, initiatives: List[StrategicInitiative], phase_number: int) -> str:
        """Determine phase focus based on active initiatives"""
        
        if not initiatives:
            return "Foundation and preparation"
        
        categories = [init.portfolio_category for init in initiatives]
        
        category_focus = {
            "transformation": "Digital transformation and capability building",
            "growth": "Market expansion and revenue growth",
            "innovation": "Innovation development and R&D",
            "efficiency": "Process optimization and cost reduction"
        }
        
        # Determine primary focus
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        primary_category = max(category_counts.items(), key=lambda x: x[1])[0]
        return category_focus.get(primary_category, "Strategic development")
    
    def _extract_phase_milestones(self,
                                initiatives: List[StrategicInitiative],
                                phase_start: datetime,
                                phase_end: datetime) -> List[Dict[str, Any]]:
        """Extract critical milestones for the phase"""
        
        milestones = []
        
        for initiative in initiatives:
            for milestone in initiative.critical_milestones:
                milestone_date = milestone["date"]
                if phase_start <= milestone_date <= phase_end:
                    milestones.append({
                        "initiative_id": initiative.initiative_id,
                        "initiative_name": initiative.name,
                        "milestone": milestone["milestone"],
                        "date": milestone_date.isoformat(),
                        "status": milestone["status"]
                    })
        
        return sorted(milestones, key=lambda x: x["date"])
    
    def _calculate_phase_resources(self, initiatives: List[StrategicInitiative]) -> Dict[str, float]:
        """Calculate resource allocation for phase"""
        
        resource_allocation = {}
        
        for initiative in initiatives:
            for resource_type, requirements in initiative.resource_requirements.items():
                if resource_type not in resource_allocation:
                    resource_allocation[resource_type] = 0
                resource_allocation[resource_type] += requirements["cost"]
        
        return resource_allocation
    
    def _define_phase_success_criteria(self, initiatives: List[StrategicInitiative]) -> List[str]:
        """Define success criteria for phase"""
        
        criteria = []
        
        for initiative in initiatives:
            # Add initiative-specific success criteria
            for kpi, target in initiative.kpis.items():
                criteria.append(f"{initiative.name}: {kpi} target {target}")
        
        # Add portfolio-level criteria
        total_progress = sum(init.progress_percentage for init in initiatives) / len(initiatives)
        criteria.append(f"Portfolio progress: {total_progress:.1f}% average")
        
        return criteria
    
    def _identify_critical_path(self, initiatives: List[StrategicInitiative]) -> List[str]:
        """Identify critical path for initiative execution"""
        
        # Create dependency graph
        initiative_map = {init.initiative_id: init for init in initiatives}
        critical_path = []
        
        # Find initiative with no dependencies (start of critical path)
        start_initiatives = [init for init in initiatives if not init.dependencies]
        
        # Build critical path by following dependencies
        current_initiatives = start_initiatives
        
        while current_initiatives:
            # Add current initiatives to critical path
            for initiative in current_initiatives:
                if initiative.initiative_id not in critical_path:
                    critical_path.append(initiative.initiative_id)
            
            # Find next level initiatives
            next_initiatives = []
            for initiative in initiatives:
                if (initiative.initiative_id not in critical_path and 
                    any(dep in critical_path for dep in initiative.dependencies)):
                    next_initiatives.append(initiative)
            
            current_initiatives = next_initiatives
        
        return critical_path
    
    def _optimize_resource_allocation(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize resource allocation across phases"""
        
        total_resources = {}
        phase_resource_peaks = {}
        
        # Calculate total resources needed
        for phase in phases:
            for resource_type, amount in phase["resource_allocation"].items():
                total_resources[resource_type] = total_resources.get(resource_type, 0) + amount
        
        # Identify resource peaks by phase
        max_resources = {}
        for phase in phases:
            for resource_type, amount in phase["resource_allocation"].items():
                max_resources[resource_type] = max(max_resources.get(resource_type, 0), amount)
        
        return {
            "total_resource_requirements": total_resources,
            "peak_resource_demand": max_resources,
            "optimization_recommendations": [
                "Stagger resource-intensive initiatives across phases",
                "Build resource buffers for critical activities",
                "Establish resource sharing agreements between initiatives",
                "Implement resource tracking and reallocation mechanisms"
            ]
        }
    
    def _plan_phase_risk_mitigation(self, phases: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Plan risk mitigation for each phase"""
        
        risk_mitigation = {}
        
        for i, phase in enumerate(phases):
            phase_risks = []
            
            # High-resource phases
            total_resources = sum(phase["resource_allocation"].values())
            if total_resources > 50:  # Threshold for high resource usage
                phase_risks.append("Resource availability and allocation risks")
            
            # High number of active initiatives
            if len(phase["active_initiatives"]) > 3:
                phase_risks.append("Coordination and dependency management risks")
            
            # Many critical milestones
            if len(phase["critical_milestones"]) > 5:
                phase_risks.append("Milestone achievement and timing risks")
            
            # Long phase duration
            if phase["duration_months"] > 6:
                phase_risks.append("Scope creep and priority drift risks")
            
            phase_risk_mitigation = {
                "risks": phase_risks,
                "mitigation_strategies": [
                    "Implement weekly resource allocation reviews",
                    "Establish initiative coordination meetings",
                    "Create milestone tracking dashboards",
                    "Define phase exit criteria and go/no-go decisions"
                ]
            }
            
            risk_mitigation[f"phase_{i+1}"] = phase_risk_mitigation
        
        return risk_mitigation
    
    def _calculate_portfolio_metrics(self, initiatives: List[StrategicInitiative]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        if not initiatives:
            return PortfolioMetrics(0, 0, 0, 0, 0, {}, 0, 0)
        
        # Financial metrics
        total_investment = sum(init.estimated_investment for init in initiatives)
        total_expected_return = sum(init.estimated_investment * init.expected_roi for init in initiatives)
        portfolio_roi = total_expected_return / total_investment if total_investment > 0 else 0
        
        # Weighted average payback period
        weighted_payback = sum(init.payback_period * init.estimated_investment 
                             for init in initiatives) / total_investment if total_investment > 0 else 0
        
        # Risk-adjusted return
        risk_scores = [init.risk_score for init in initiatives]
        avg_risk = np.mean(risk_scores)
        risk_adjusted_return = portfolio_roi * (1 - avg_risk)
        
        # Resource utilization
        resource_utilization = {}
        for resource_type in ResourceType:
            total_resource_cost = sum(
                init.resource_requirements.get(resource_type.value, {}).get("cost", 0)
                for init in initiatives
            )
            resource_utilization[resource_type.value] = total_resource_cost
        
        # Success rate
        active_initiatives = [init for init in initiatives if init.status != InitiativeStatus.CANCELLED]
        completed_initiatives = [init for init in initiatives if init.status == InitiativeStatus.COMPLETED]
        success_rate = len(completed_initiatives) / len(active_initiatives) if active_initiatives else 0
        
        # Strategic alignment score
        alignment_scores = []
        for init in initiatives:
            # Calculate alignment based on progress and priority
            priority_weight = {"critical": 1.0, "high": 0.8, "medium": 0.6, "low": 0.4}[init.priority.value]
            alignment_score = (init.progress_percentage / 100.0) * priority_weight
            alignment_scores.append(alignment_score)
        
        strategic_alignment_score = np.mean(alignment_scores)
        
        return PortfolioMetrics(
            total_investment=total_investment,
            expected_total_return=total_expected_return,
            portfolio_roi=portfolio_roi,
            average_payback_period=weighted_payback,
            risk_adjusted_return=risk_adjusted_return,
            resource_utilization=resource_utilization,
            initiative_success_rate=success_rate,
            strategic_alignment_score=strategic_alignment_score
        )
    
    def _generate_optimization_insights(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Generate insights from portfolio optimization"""
        
        if not initiatives:
            return {}
        
        insights = {
            "portfolio_diversification": self._analyze_portfolio_diversification(initiatives),
            "resource_optimization": self._analyze_resource_optimization(initiatives),
            "risk_distribution": self._analyze_risk_distribution(initiatives),
            "timeline_optimization": self._analyze_timeline_optimization(initiatives),
            "strategic_alignment": self._analyze_strategic_alignment(initiatives)
        }
        
        return insights
    
    def _analyze_portfolio_diversification(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze portfolio diversification across categories and priorities"""
        
        categories = [init.portfolio_category for init in initiatives]
        priorities = [init.priority.value for init in initiatives]
        
        category_distribution = {}
        for category in categories:
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        priority_distribution = {}
        for priority in priorities:
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        # Calculate diversification score
        category_diversity = len(set(categories)) / len(set([cat.value for cat in InitiativePriority])) if categories else 0
        priority_diversity = len(set(priorities)) / 4 if priorities else 0
        diversification_score = (category_diversity + priority_diversity) / 2
        
        return {
            "category_distribution": category_distribution,
            "priority_distribution": priority_distribution,
            "diversification_score": diversification_score,
            "diversification_assessment": "Well diversified" if diversification_score > 0.6 else "Needs more diversity"
        }
    
    def _analyze_resource_optimization(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze resource optimization across initiatives"""
        
        # Resource allocation analysis
        total_resource_costs = {}
        resource_efficiency = {}
        
        for resource_type in ResourceType:
            total_cost = sum(
                init.resource_requirements.get(resource_type.value, {}).get("cost", 0)
                for init in initiatives
            )
            total_resource_costs[resource_type.value] = total_cost
            
            # Calculate efficiency (ROI per resource unit)
            if total_cost > 0:
                total_roi = sum(init.expected_roi * init.estimated_investment for init in initiatives)
                resource_efficiency[resource_type.value] = total_roi / total_cost
            else:
                resource_efficiency[resource_type.value] = 0
        
        return {
            "resource_allocation": total_resource_costs,
            "resource_efficiency": resource_efficiency,
            "optimization_opportunities": self._identify_resource_optimization_opportunities(initiatives)
        }
    
    def _identify_resource_optimization_opportunities(self, initiatives: List[StrategicInitiative]) -> List[str]:
        """Identify resource optimization opportunities"""
        
        opportunities = []
        
        # High-cost, low-ROI initiatives
        low_efficiency_initiatives = [
            init for init in initiatives
            if init.expected_roi < 1.5 and init.estimated_investment > 20
        ]
        if low_efficiency_initiatives:
            opportunities.append(f"Review {len(low_efficiency_initiatives)} initiatives with low ROI and high investment")
        
        # Resource conflicts
        resource_conflicts = self._identify_resource_conflicts(initiatives)
        if resource_conflicts:
            opportunities.append("Resolve resource conflicts between initiatives")
        
        # Overlapping initiatives
        similar_initiatives = self._identify_similar_initiatives(initiatives)
        if similar_initiatives:
            opportunities.append(f"Consider combining {len(similar_initiatives)} similar initiatives")
        
        return opportunities
    
    def _identify_resource_conflicts(self, initiatives: List[StrategicInitiative]) -> List[Dict[str, Any]]:
        """Identify resource conflicts between initiatives"""
        
        conflicts = []
        timeline_overlaps = []
        
        for i, init1 in enumerate(initiatives):
            for init2 in initiatives[i+1:]:
                # Check timeline overlap
                if (init1.start_date <= init2.planned_end_date and 
                    init2.start_date <= init1.planned_end_date):
                    timeline_overlaps.append((init1.initiative_id, init2.initiative_id))
                    
                    # Check resource conflicts
                    for resource_type in ResourceType:
                        cost1 = init1.resource_requirements.get(resource_type.value, {}).get("cost", 0)
                        cost2 = init2.resource_requirements.get(resource_type.value, {}).get("cost", 0)
                        
                        if cost1 > 0 and cost2 > 0:
                            conflicts.append({
                                "initiative_1": init1.initiative_id,
                                "initiative_2": init2.initiative_id,
                                "resource_type": resource_type.value,
                                "conflict_severity": min(cost1, cost2) / max(cost1, cost2)
                            })
        
        return conflicts
    
    def _identify_similar_initiatives(self, initiatives: List[StrategicInitiative]) -> List[List[str]]:
        """Identify similar or overlapping initiatives"""
        
        similar_groups = []
        processed = set()
        
        for i, init1 in enumerate(initiatives):
            if init1.initiative_id in processed:
                continue
            
            similar_group = [init1.initiative_id]
            
            for init2 in initiatives[i+1:]:
                if init2.initiative_id in processed:
                    continue
                
                # Check for overlapping objectives
                overlap = set(init1.strategic_objectives) & set(init2.strategic_objectives)
                if len(overlap) >= 2:  # Significant overlap
                    similar_group.append(init2.initiative_id)
                    processed.add(init2.initiative_id)
            
            processed.add(init1.initiative_id)
            if len(similar_group) > 1:
                similar_groups.append(similar_group)
        
        return similar_groups
    
    def _analyze_risk_distribution(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze risk distribution across portfolio"""
        
        risk_scores = [init.risk_score for init in initiatives]
        risk_categories = {
            "low_risk": [init for init in initiatives if init.risk_score < 0.3],
            "medium_risk": [init for init in initiatives if 0.3 <= init.risk_score <= 0.7],
            "high_risk": [init for init in initiatives if init.risk_score > 0.7]
        }
        
        return {
            "average_risk": np.mean(risk_scores),
            "risk_distribution": {category: len(inits) for category, init in risk_categories.items() for init in (init,)},  # Fix for dictionary comprehension
            "high_risk_initiatives": [init.initiative_id for init in risk_categories["high_risk"]],
            "risk_rebalancing_recommendations": self._generate_risk_rebalancing_recommendations(risk_categories)
        }
    
    def _generate_risk_rebalancing_recommendations(self, risk_categories: Dict[str, List[StrategicInitiative]]) -> List[str]:
        """Generate risk rebalancing recommendations"""
        
        recommendations = []
        total_initiatives = sum(len(initiatives) for initiatives in risk_categories.values())
        
        # Check balance
        high_risk_percentage = len(risk_categories["high_risk"]) / total_initiatives
        low_risk_percentage = len(risk_categories["low_risk"]) / total_initiatives
        
        if high_risk_percentage > 0.3:
            recommendations.append("Consider reducing high-risk initiatives or implementing stronger risk mitigation")
        
        if low_risk_percentage < 0.2:
            recommendations.append("Consider adding more low-risk initiatives to balance portfolio risk")
        
        return recommendations
    
    def _analyze_timeline_optimization(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze timeline optimization across portfolio"""
        
        start_dates = [init.start_date for init in initiatives]
        end_dates = [init.planned_end_date for init in initiatives]
        
        timeline_analysis = {
            "earliest_start": min(start_dates).isoformat() if start_dates else None,
            "latest_end": max(end_dates).isoformat() if end_dates else None,
            "total_duration_days": (max(end_dates) - min(start_dates)).days if start_dates and end_dates else 0,
            "parallel_execution_potential": self._assess_parallel_execution_potential(initiatives),
            "bottleneck_identification": self._identify_timeline_bottlenecks(initiatives)
        }
        
        return timeline_analysis
    
    def _assess_parallel_execution_potential(self, initiatives: List[StrategicInitiative]) -> float:
        """Assess potential for parallel execution"""
        
        if len(initiatives) <= 1:
            return 0.0
        
        # Count initiatives that can run in parallel
        parallel_count = 0
        for i, init1 in enumerate(initiatives):
            can_run_parallel = True
            for init2 in initiatives:
                if init1 != init2:
                    # Check for dependencies
                    if (init2.initiative_id in init1.dependencies or 
                        init1.initiative_id in init2.dependencies):
                        can_run_parallel = False
                        break
            
            if can_run_parallel:
                parallel_count += 1
        
        return parallel_count / len(initiatives)
    
    def _identify_timeline_bottlenecks(self, initiatives: List[StrategicInitiative]) -> List[Dict[str, Any]]:
        """Identify timeline bottlenecks in portfolio"""
        
        bottlenecks = []
        
        # Find initiatives with longest duration
        durations = [(init.initiative_id, (init.planned_end_date - init.start_date).days) 
                    for init in initiatives]
        durations.sort(key=lambda x: x[1], reverse=True)
        
        # Identify top bottlenecks
        for initiative_id, duration in durations[:3]:  # Top 3 longest
            initiative = next(init for init in initiatives if init.initiative_id == initiative_id)
            bottlenecks.append({
                "initiative_id": initiative_id,
                "initiative_name": initiative.name,
                "duration_days": duration,
                "bottleneck_type": "Long duration" if duration > 730 else "Medium duration",
                "mitigation_suggestion": self._suggest_bottleneck_mitigation(initiative)
            })
        
        return bottlenecks
    
    def _suggest_bottleneck_mitigation(self, initiative: StrategicInitiative) -> str:
        """Suggest mitigation for timeline bottleneck"""
        
        if initiative.estimated_investment > 30:
            return "Consider phase-based implementation or resource augmentation"
        elif len(initiative.dependencies) > 2:
            return "Review and optimize dependency chain"
        elif initiative.portfolio_category == "transformation":
            return "Implement agile methodology and incremental delivery"
        else:
            return "Assess for parallel workstreams or process optimization"
    
    def _analyze_strategic_alignment(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze strategic alignment across portfolio"""
        
        # Aggregate strategic objectives
        all_objectives = []
        for init in initiatives:
            all_objectives.extend(init.strategic_objectives)
        
        objective_frequency = {}
        for objective in all_objectives:
            objective_frequency[objective] = objective_frequency.get(objective, 0) + 1
        
        # Calculate alignment score
        weighted_alignment = sum(
            init.progress_percentage * {"critical": 4, "high": 3, "medium": 2, "low": 1}[init.priority.value]
            for init in initiatives
        ) / sum({"critical": 4, "high": 3, "medium": 2, "low": 1}[init.priority.value] for init in initiatives)
        
        return {
            "top_strategic_objectives": sorted(objective_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
            "overall_alignment_score": weighted_alignment / 100.0,  # Normalize to 0-1
            "alignment_assessment": "Well aligned" if weighted_alignment > 70 else "Needs improvement"
        }
    
    def _assess_portfolio_risks(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Assess overall portfolio risks"""
        
        portfolio_risks = {
            "concentration_risks": self._identify_concentration_risks(initiatives),
            "execution_risks": self._assess_execution_risks(initiatives),
            "resource_risks": self._assess_resource_risks(initiatives),
            "timeline_risks": self._assess_timeline_risks(initiatives),
            "overall_portfolio_risk": self._calculate_overall_portfolio_risk(initiatives)
        }
        
        return portfolio_risks
    
    def _identify_concentration_risks(self, initiatives: List[StrategicInitiative]) -> List[str]:
        """Identify concentration risks in portfolio"""
        
        risks = []
        
        # Category concentration
        categories = [init.portfolio_category for init in initiatives]
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        
        max_category_count = max(category_counts.values())
        if max_category_count > len(initiatives) * 0.5:
            dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]
            risks.append(f"High concentration in {dominant_category} initiatives")
        
        # Investment concentration
        investments = [init.estimated_investment for init in initiatives]
        total_investment = sum(investments)
        max_investment = max(investments)
        
        if max_investment > total_investment * 0.4:
            risky_initiative = next(init for init in initiatives if init.estimated_investment == max_investment)
            risks.append(f"High investment concentration in {risky_initiative.name}")
        
        return risks
    
    def _assess_execution_risks(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Assess execution risks across portfolio"""
        
        # High-risk initiatives
        high_risk_initiatives = [init for init in initiatives if init.risk_score > 0.7]
        
        # Complex initiatives (high dependencies)
        complex_initiatives = [init for init in initiatives if len(init.dependencies) > 2]
        
        # Long-duration initiatives
        long_duration_initiatives = [
            init for init in initiatives
            if (init.planned_end_date - init.start_date).days > 730  # 2 years
        ]
        
        return {
            "high_risk_count": len(high_risk_initiatives),
            "complex_initiative_count": len(complex_initiatives),
            "long_duration_count": len(long_duration_initiatives),
            "execution_risk_score": len(high_risk_initiatives) / len(initiatives) if initiatives else 0
        }
    
    def _assess_resource_risks(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Assess resource-related risks"""
        
        # Total resource demand
        resource_demands = {}
        for resource_type in ResourceType:
            total_demand = sum(
                init.resource_requirements.get(resource_type.value, {}).get("cost", 0)
                for init in initiatives
            )
            resource_demands[resource_type.value] = total_demand
        
        # Check against constraints
        resource_risks = []
        for resource_type, demand in resource_demands.items():
            constraint = self.resource_constraints.get(resource_type, float('inf'))
            if demand > constraint:
                resource_risks.append(f"{resource_type} demand exceeds capacity")
        
        return {
            "resource_demands": resource_demands,
            "resource_risks": resource_risks,
            "resource_risk_score": len(resource_risks) / len(resource_demands) if resource_demands else 0
        }
    
    def _assess_timeline_risks(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Assess timeline-related risks"""
        
        # Overlapping initiatives
        overlapping_pairs = 0
        for i, init1 in enumerate(initiatives):
            for init2 in initiatives[i+1:]:
                if (init1.start_date <= init2.planned_end_date and 
                    init2.start_date <= init1.planned_end_date):
                    overlapping_pairs += 1
        
        # Initiatives behind schedule
        behind_schedule = [
            init for init in initiatives
            if init.progress_percentage < ((datetime.now() - init.start_date).days / 
                                          (init.planned_end_date - init.start_date).days * 100)
        ]
        
        return {
            "overlapping_initiative_pairs": overlapping_pairs,
            "behind_schedule_count": len(behind_schedule),
            "timeline_risk_score": len(behind_schedule) / len(initiatives) if initiatives else 0
        }
    
    def _calculate_overall_portfolio_risk(self, initiatives: List[StrategicInitiative]) -> float:
        """Calculate overall portfolio risk score"""
        
        if not initiatives:
            return 0.0
        
        # Weighted average of individual risks
        total_weighted_risk = sum(init.risk_score * init.estimated_investment for init in initiatives)
        total_investment = sum(init.estimated_investment for init in initiatives)
        
        base_risk = total_weighted_risk / total_investment if total_investment > 0 else 0
        
        # Add concentration and execution risk penalties
        concentration_penalty = 0.1 if len(set(init.portfolio_category for init in initiatives)) < 2 else 0
        execution_penalty = 0.1 if len([init for init in initiatives if init.risk_score > 0.7]) > len(initiatives) * 0.3 else 0
        
        overall_risk = min(1.0, base_risk + concentration_penalty + execution_penalty)
        
        return overall_risk
    
    def _define_portfolio_governance(self) -> Dict[str, Any]:
        """Define portfolio governance framework"""
        
        return {
            "governance_structure": {
                "portfolio_steering_committee": {
                    "composition": ["CEO", "CFO", "COO", "Chief Strategy Officer"],
                    "responsibilities": ["Strategic alignment", "Resource allocation", "Risk oversight"],
                    "meeting_frequency": "Monthly"
                },
                "initiative_review_board": {
                    "composition": ["Senior leaders from each function"],
                    "responsibilities": ["Initiative approval", "Performance review", "Issue resolution"],
                    "meeting_frequency": "Bi-weekly"
                },
                "project_management_office": {
                    "composition": ["PMO Director", "Project managers", "Business analysts"],
                    "responsibilities": ["Execution monitoring", "Status reporting", "Risk management"],
                    "meeting_frequency": "Weekly"
                }
            },
            "decision_making_process": {
                "initiative_approval": "Steering committee approval required for initiatives >$10M",
                "resource_reallocation": "Review board approval for resource shifts >20%",
                "scope_changes": "Change control process for significant scope modifications",
                "termination_decisions": "Steering committee decision with business case review"
            },
            "performance_management": {
                "kpi_dashboard": "Real-time performance tracking dashboard",
                "monthly_reviews": "Detailed performance reviews with corrective actions",
                "quarterly_assessments": "Comprehensive portfolio health assessments",
                "annual_strategic_review": "Strategic alignment and portfolio optimization review"
            },
            "risk_management": {
                "risk_assessment": "Monthly risk assessments with mitigation plans",
                "escalation_procedures": "Clear escalation paths for critical issues",
                "contingency_planning": "Pre-defined contingency plans for major risks",
                "scenario_planning": "Regular scenario planning for portfolio adaptation"
            }
        }
    
    def monitor_portfolio_performance(self,
                                    initiative_ids: Optional[List[str]] = None,
                                    performance_period: str = "monthly") -> Dict[str, Any]:
        """Monitor portfolio performance across initiatives"""
        
        if initiative_ids is None:
            monitored_initiatives = self.initiatives
        else:
            monitored_initiatives = [init for init in self.initiatives if init.initiative_id in initiative_ids]
        
        if not monitored_initiatives:
            return {"error": "No initiatives to monitor"}
        
        # Performance metrics
        performance_data = self._collect_performance_data(monitored_initiatives, performance_period)
        
        # Variance analysis
        variance_analysis = self._analyze_performance_variance(monitored_initiatives)
        
        # Trend analysis
        trend_analysis = self._analyze_performance_trends(monitored_initiatives)
        
        # Alert system
        alerts = self._generate_performance_alerts(monitored_initiatives)
        
        # Recommendations
        recommendations = self._generate_performance_recommendations(monitored_initiatives, alerts)
        
        performance_report = {
            "monitoring_period": performance_period,
            "monitoring_date": datetime.now().isoformat(),
            "initiatives_monitored": len(monitored_initiatives),
            "performance_data": performance_data,
            "variance_analysis": variance_analysis,
            "trend_analysis": trend_analysis,
            "alerts": alerts,
            "recommendations": recommendations,
            "portfolio_health_score": self._calculate_portfolio_health_score(monitored_initiatives)
        }
        
        return performance_report
    
    def _collect_performance_data(self, initiatives: List[StrategicInitiative], period: str) -> Dict[str, Any]:
        """Collect performance data for monitored initiatives"""
        
        performance_data = {}
        
        for initiative in initiatives:
            performance_data[initiative.initiative_id] = {
                "initiative_name": initiative.name,
                "status": initiative.status.value,
                "progress_percentage": initiative.progress_percentage,
                "planned_vs_actual": self._calculate_planned_vs_actual(initiative),
                "kpi_performance": self._assess_kpi_performance(initiative),
                "resource_utilization": self._calculate_resource_utilization(initiative),
                "milestone_status": self._assess_milestone_status(initiative),
                "risk_level": initiative.risk_score,
                "budget_variance": self._calculate_budget_variance(initiative)
            }
        
        return performance_data
    
    def _calculate_planned_vs_actual(self, initiative: StrategicInitiative) -> Dict[str, float]:
        """Calculate planned vs actual progress"""
        
        elapsed_time = (datetime.now() - initiative.start_date).days
        total_planned_time = (initiative.planned_end_date - initiative.start_date).days
        
        planned_progress = min(100, (elapsed_time / total_planned_time * 100)) if total_planned_time > 0 else 0
        actual_progress = initiative.progress_percentage
        
        return {
            "planned_progress": planned_progress,
            "actual_progress": actual_progress,
            "variance": actual_progress - planned_progress
        }
    
    def _assess_kpi_performance(self, initiative: StrategicInitiative) -> Dict[str, float]:
        """Assess KPI performance against targets"""
        
        kpi_performance = {}
        
        for kpi_name, target_value in initiative.kpis.items():
            current_value = initiative.current_performance.get(kpi_name, 0)
            performance_ratio = current_value / target_value if target_value > 0 else 0
            kpi_performance[kpi_name] = {
                "target": target_value,
                "current": current_value,
                "achievement_ratio": performance_ratio,
                "status": "on_track" if performance_ratio >= 0.9 else "at_risk" if performance_ratio >= 0.7 else "behind"
            }
        
        return kpi_performance
    
    def _calculate_resource_utilization(self, initiative: StrategicInitiative) -> Dict[str, float]:
        """Calculate resource utilization for initiative"""
        
        utilization = {}
        
        for resource_type, requirements in initiative.resource_requirements.items():
            # Simplified utilization calculation
            utilization[resource_type] = {
                "planned": requirements["cost"],
                "estimated_actual": requirements["cost"] * 1.1,  # 10% overrun assumption
                "utilization_rate": 1.1  # 110% utilization
            }
        
        return utilization
    
    def _assess_milestone_status(self, initiative: StrategicInitiative) -> Dict[str, Any]:
        """Assess milestone completion status"""
        
        milestone_status = {
            "total_milestones": len(initiative.critical_milestones),
            "completed_milestones": len([m for m in initiative.critical_milestones if m["status"] == "completed"]),
            "in_progress_milestones": len([m for m in initiative.critical_milestones if m["status"] == "in_progress"]),
            "overdue_milestones": len([m for m in initiative.critical_milestones 
                                     if m["status"] != "completed" and m["date"] < datetime.now()]),
            "upcoming_milestones": len([m for m in initiative.critical_milestones 
                                      if m["date"] >= datetime.now() and m["date"] <= datetime.now() + timedelta(days=30)])
        }
        
        return milestone_status
    
    def _calculate_budget_variance(self, initiative: StrategicInitiative) -> Dict[str, float]:
        """Calculate budget variance for initiative"""
        
        # Simplified budget calculation
        estimated_cost = initiative.estimated_investment
        actual_cost = estimated_cost * 1.15  # 15% overrun assumption
        budget_variance = actual_cost - estimated_cost
        variance_percentage = (budget_variance / estimated_cost * 100) if estimated_cost > 0 else 0
        
        return {
            "budgeted_amount": estimated_cost,
            "estimated_actual": actual_cost,
            "variance": budget_variance,
            "variance_percentage": variance_percentage
        }
    
    def _analyze_performance_variance(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze performance variance across initiatives"""
        
        variance_summary = {
            "progress_variances": [],
            "budget_variances": [],
            "timeline_variances": []
        }
        
        for initiative in initiatives:
            # Progress variance
            elapsed_time = (datetime.now() - initiative.start_date).days
            total_planned_time = (initiative.planned_end_date - initiative.start_date).days
            planned_progress = min(100, (elapsed_time / total_planned_time * 100)) if total_planned_time > 0 else 0
            progress_variance = initiative.progress_percentage - planned_progress
            
            variance_summary["progress_variances"].append({
                "initiative_id": initiative.initiative_id,
                "variance": progress_variance,
                "status": "ahead" if progress_variance > 5 else "on_track" if progress_variance >= -5 else "behind"
            })
            
            # Budget variance
            budget_variance = (actual_cost - initiative.estimated_investment) / initiative.estimated_investment * 100
            variance_summary["budget_variances"].append({
                "initiative_id": initiative.initiative_id,
                "variance": budget_variance,
                "status": "over_budget" if budget_variance > 10 else "on_budget" if budget_variance >= -10 else "under_budget"
            })
        
        return variance_summary
    
    def _analyze_performance_trends(self, initiatives: List[StrategicInitiative]) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        
        trends = {
            "overall_portfolio_trend": "improving",  # Simplified trend analysis
            "initiative_trends": {},
            "trend_insights": []
        }
        
        for initiative in initiatives:
            # Simplified trend calculation
            progress_velocity = initiative.progress_percentage / max(1, (datetime.now() - initiative.start_date).days / 30)
            
            if progress_velocity > 5:  # 5% per month
                trend = "improving"
            elif progress_velocity > 3:
                trend = "stable"
            else:
                trend = "declining"
            
            trends["initiative_trends"][initiative.initiative_id] = {
                "trend": trend,
                "velocity": progress_velocity,
                "confidence": 0.7  # Simplified confidence score
            }
        
        return trends
    
    def _generate_performance_alerts(self, initiatives: List[StrategicInitiative]) -> List[Dict[str, Any]]:
        """Generate performance alerts for initiatives"""
        
        alerts = []
        
        for initiative in initiatives:
            # Progress alerts
            elapsed_time = (datetime.now() - initiative.start_date).days
            total_planned_time = (initiative.planned_end_date - initiative.start_date).days
            planned_progress = min(100, (elapsed_time / total_planned_time * 100)) if total_planned_time > 0 else 0
            progress_variance = initiative.progress_percentage - planned_progress
            
            if progress_variance < -10:
                alerts.append({
                    "initiative_id": initiative.initiative_id,
                    "alert_type": "progress_delay",
                    "severity": "high",
                    "message": f"{initiative.name} is significantly behind schedule",
                    "recommended_action": "Review timeline and resource allocation"
                })
            
            # Budget alerts
            if initiative.estimated_investment > 20 and initiative.risk_score > 0.6:
                alerts.append({
                    "initiative_id": initiative.initiative_id,
                    "alert_type": "high_risk_budget",
                    "severity": "medium",
                    "message": f"{initiative.name} has high risk and significant budget",
                    "recommended_action": "Implement enhanced monitoring and risk mitigation"
                })
            
            # Milestone alerts
            overdue_milestones = [m for m in initiative.critical_milestones 
                                if m["status"] != "completed" and m["date"] < datetime.now()]
            if overdue_milestones:
                alerts.append({
                    "initiative_id": initiative.initiative_id,
                    "alert_type": "overdue_milestones",
                    "severity": "high" if len(overdue_milestones) > 1 else "medium",
                    "message": f"{initiative.name} has {len(overdue_milestones)} overdue milestones",
                    "recommended_action": "Review milestone dependencies and resource allocation"
                })
        
        return alerts
    
    def _generate_performance_recommendations(self, initiatives: List[StrategicInitiative], alerts: List[Dict[str, Any]]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Alert-based recommendations
        high_severity_alerts = [alert for alert in alerts if alert["severity"] == "high"]
        if high_severity_alerts:
            recommendations.append(f"Address {len(high_severity_alerts)} high-severity performance issues immediately")
        
        # Portfolio-level recommendations
        behind_schedule_count = len([
            init for init in initiatives
            if init.progress_percentage < ((datetime.now() - init.start_date).days / 
                                          (init.planned_end_date - init.start_date).days * 100)
        ])
        if behind_schedule_count > len(initiatives) * 0.3:
            recommendations.append("Consider portfolio-level resource reallocation to support delayed initiatives")
        
        # Risk-based recommendations
        high_risk_initiatives = [init for init in initiatives if init.risk_score > 0.7]
        if len(high_risk_initiatives) > len(initiatives) * 0.25:
            recommendations.append("Implement enhanced risk monitoring for high-risk initiatives")
        
        # Progress-based recommendations
        avg_progress = np.mean([init.progress_percentage for init in initiatives])
        if avg_progress < 50:
            recommendations.append("Portfolio performance below expectations - review execution strategies")
        
        return recommendations
    
    def _calculate_portfolio_health_score(self, initiatives: List[StrategicInitiative]) -> float:
        """Calculate overall portfolio health score"""
        
        if not initiatives:
            return 0.0
        
        # Health score components
        progress_score = np.mean([init.progress_percentage for init in initiatives]) / 100.0
        
        # Risk penalty
        avg_risk = np.mean([init.risk_score for init in initiatives])
        risk_penalty = avg_risk * 0.2
        
        # Status penalty
        on_track_count = len([init for init in initiatives if init.status in [InitiativeStatus.IN_PROGRESS, InitiativeStatus.APPROVED]])
        status_score = on_track_count / len(initiatives)
        
        # Calculate health score
        health_score = (progress_score * 0.4 + status_score * 0.4 + (1 - risk_penalty) * 0.2)
        
        return min(1.0, max(0.0, health_score))
    
    def optimize_portfolio_performance(self,
                                     optimization_objectives: Dict[str, float],
                                     constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio performance based on objectives and constraints"""
        
        if not self.initiatives:
            return {"error": "No initiatives in portfolio to optimize"}
        
        # Current state analysis
        current_state = self._assess_current_portfolio_state()
        
        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities()
        
        # Optimization scenarios
        optimization_scenarios = self._generate_optimization_scenarios(optimization_objectives, constraints)
        
        # Implementation roadmap
        optimization_roadmap = self._create_optimization_roadmap(optimization_scenarios)
        
        optimization_plan = {
            "optimization_objectives": optimization_objectives,
            "constraints": constraints,
            "current_state": current_state,
            "optimization_opportunities": optimization_opportunities,
            "optimization_scenarios": optimization_scenarios,
            "implementation_roadmap": optimization_roadmap,
            "expected_improvements": self._project_optimization_improvements(optimization_scenarios),
            "success_metrics": self._define_optimization_success_metrics()
        }
        
        return optimization_plan
    
    def _assess_current_portfolio_state(self) -> Dict[str, Any]:
        """Assess current portfolio state"""
        
        if not self.initiatives:
            return {}
        
        # Calculate current metrics
        total_investment = sum(init.estimated_investment for init in self.initiatives)
        weighted_roi = sum(init.expected_roi * init.estimated_investment for init in self.initiatives) / total_investment
        avg_progress = np.mean([init.progress_percentage for init in self.initiatives])
        avg_risk = np.mean([init.risk_score for init in self.initiatives])
        
        # Portfolio health components
        financial_health = min(1.0, weighted_roi / 3.0)  # Normalize to 3.0 ROI baseline
        execution_health = avg_progress / 100.0
        risk_health = 1.0 - avg_risk
        
        overall_health = (financial_health + execution_health + risk_health) / 3
        
        return {
            "financial_metrics": {
                "total_investment": total_investment,
                "weighted_average_roi": weighted_roi,
                "portfolio_value_at_risk": sum(init.estimated_investment * init.risk_score for init in self.initiatives)
            },
            "execution_metrics": {
                "average_progress": avg_progress,
                "on_track_initiatives": len([init for init in self.initiatives if init.status == InitiativeStatus.IN_PROGRESS]),
                "completed_initiatives": len([init for init in self.initiatives if init.status == InitiativeStatus.COMPLETED])
            },
            "risk_metrics": {
                "average_risk_score": avg_risk,
                "high_risk_initiatives": len([init for init in self.initiatives if init.risk_score > 0.7]),
                "risk_distribution": self._calculate_risk_distribution()
            },
            "portfolio_health": {
                "overall_health_score": overall_health,
                "financial_health": financial_health,
                "execution_health": execution_health,
                "risk_health": risk_health
            }
        }
    
    def _calculate_risk_distribution(self) -> Dict[str, float]:
        """Calculate risk distribution across portfolio"""
        
        if not self.initiatives:
            return {}
        
        low_risk = len([init for init in self.initiatives if init.risk_score < 0.3]) / len(self.initiatives)
        medium_risk = len([init for init in self.initiatives if 0.3 <= init.risk_score <= 0.7]) / len(self.initiatives)
        high_risk = len([init for init in self.initiatives if init.risk_score > 0.7]) / len(self.initiatives)
        
        return {
            "low_risk_percentage": low_risk,
            "medium_risk_percentage": medium_risk,
            "high_risk_percentage": high_risk
        }
    
    def _identify_optimization_opportunities(self) -> Dict[str, List[str]]:
        """Identify portfolio optimization opportunities"""
        
        opportunities = {
            "financial_optimization": [],
            "execution_optimization": [],
            "risk_optimization": [],
            "resource_optimization": [],
            "strategic_optimization": []
        }
        
        # Financial optimization
        low_roi_initiatives = [init for init in self.initiatives if init.expected_roi < 1.5]
        if low_roi_initiatives:
            opportunities["financial_optimization"].append(f"Consider reallocating {len(low_roi_initiatives)} low-ROI initiatives")
        
        # Execution optimization
        behind_schedule = [init for init in self.initiatives if init.progress_percentage < 50]
        if behind_schedule:
            opportunities["execution_optimization"].append(f"Accelerate {len(behind_schedule)} behind-schedule initiatives")
        
        # Risk optimization
        high_risk_initiatives = [init for init in self.initiatives if init.risk_score > 0.8]
        if high_risk_initiatives:
            opportunities["risk_optimization"].append(f"Implement enhanced risk mitigation for {len(high_risk_initiatives)} high-risk initiatives")
        
        # Resource optimization
        resource_conflicts = self._identify_resource_conflicts(self.initiatives)
        if resource_conflicts:
            opportunities["resource_optimization"].append("Resolve resource conflicts through better scheduling")
        
        # Strategic optimization
        similar_initiatives = self._identify_similar_initiatives(self.initiatives)
        if similar_initiatives:
            opportunities["strategic_optimization"].append(f"Combine {len(similar_initiatives)} similar initiatives for efficiency")
        
        return opportunities
    
    def _generate_optimization_scenarios(self,
                                       objectives: Dict[str, float],
                                       constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization scenarios based on objectives"""
        
        scenarios = []
        
        # Scenario 1: Conservative optimization (minimal changes)
        conservative_scenario = {
            "scenario_name": "Conservative Optimization",
            "description": "Minimal changes with risk reduction focus",
            "expected_improvements": {
                "risk_reduction": 0.1,
                "roi_improvement": 0.05,
                "execution_improvement": 0.08
            },
            "required_changes": [
                "Implement enhanced risk monitoring",
                "Improve milestone tracking",
                "Optimize resource allocation"
            ],
            "implementation_effort": "Low",
            "time_to_impact": "3 months"
        }
        scenarios.append(conservative_scenario)
        
        # Scenario 2: Balanced optimization
        balanced_scenario = {
            "scenario_name": "Balanced Optimization",
            "description": "Balanced approach addressing multiple objectives",
            "expected_improvements": {
                "risk_reduction": 0.15,
                "roi_improvement": 0.12,
                "execution_improvement": 0.15
            },
            "required_changes": [
                "Reallocate resources from low-ROI initiatives",
                "Accelerate high-potential initiatives",
                "Implement advanced monitoring systems",
                "Enhance stakeholder engagement"
            ],
            "implementation_effort": "Medium",
            "time_to_impact": "6 months"
        }
        scenarios.append(balanced_scenario)
        
        # Scenario 3: Aggressive optimization
        aggressive_scenario = {
            "scenario_name": "Aggressive Optimization",
            "description": "Comprehensive optimization with significant changes",
            "expected_improvements": {
                "risk_reduction": 0.25,
                "roi_improvement": 0.20,
                "execution_improvement": 0.25
            },
            "required_changes": [
                "Portfolio restructuring and initiative reprioritization",
                "Significant resource reallocation",
                "Implementation of advanced project management tools",
                "Strategic partnership development",
                "Change management program"
            ],
            "implementation_effort": "High",
            "time_to_impact": "12 months"
        }
        scenarios.append(aggressive_scenario)
        
        return scenarios
    
    def _create_optimization_roadmap(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimization implementation roadmap"""
        
        # Use balanced scenario as default implementation plan
        selected_scenario = next((s for s in scenarios if s["scenario_name"] == "Balanced Optimization"), scenarios[0])
        
        roadmap = {
            "implementation_phases": [
                {
                    "phase": 1,
                    "name": "Foundation and Assessment",
                    "duration_months": 2,
                    "activities": [
                        "Detailed portfolio assessment",
                        "Stakeholder alignment",
                        "Resource availability confirmation",
                        "Tool and process preparation"
                    ],
                    "deliverables": ["Optimization plan", "Resource allocation schedule", "Success metrics definition"]
                },
                {
                    "phase": 2,
                    "name": "Quick Wins and Stabilization",
                    "duration_months": 3,
                    "activities": [
                        "Implement enhanced monitoring",
                        "Address immediate issues",
                        "Resource reallocation",
                        "Communication and training"
                    ],
                    "deliverables": ["Monitoring dashboard", "Resource reallocation plan", "Training completion"]
                },
                {
                    "phase": 3,
                    "name": "Structural Optimization",
                    "duration_months": 6,
                    "activities": [
                        "Portfolio restructuring",
                        "Advanced tool implementation",
                        "Process optimization",
                        "Performance improvement"
                    ],
                    "deliverables": ["Optimized portfolio", "Enhanced processes", "Performance improvements"]
                },
                {
                    "phase": 4,
                    "name": "Continuous Improvement",
                    "duration_months": 12,
                    "activities": [
                        "Performance monitoring",
                        "Continuous optimization",
                        "Best practice sharing",
                        "Portfolio evolution"
                    ],
                    "deliverables": ["Sustained performance", "Best practices", "Evolution roadmap"]
                }
            ],
            "success_milestones": [
                "Month 2: Optimization plan approved",
                "Month 5: Monitoring system operational",
                "Month 8: Portfolio restructuring complete",
                "Month 12: Performance targets achieved"
            ],
            "resource_requirements": {
                "dedicated_team": 5,
                "external_consultants": 2,
                "tool_licensing": 3,
                "training_budget": 0.5
            }
        }
        
        return roadmap
    
    def _project_optimization_improvements(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Project expected improvements from optimization"""
        
        # Aggregate improvements across scenarios
        improvements = {
            "conservative": {"roi": 0.05, "risk": -0.1, "execution": 0.08},
            "balanced": {"roi": 0.12, "risk": -0.15, "execution": 0.15},
            "aggressive": {"roi": 0.20, "risk": -0.25, "execution": 0.25}
        }
        
        return {
            "expected_improvements": improvements,
            "optimization_roi": 2.5,  # Expected ROI of optimization investment
            "payback_period": 8,  # months
            "confidence_level": 0.8
        }
    
    def _define_optimization_success_metrics(self) -> Dict[str, List[str]]:
        """Define success metrics for optimization"""
        
        return {
            "financial_metrics": [
                "Portfolio ROI improvement",
                "Value at risk reduction",
                "Investment efficiency enhancement",
                "Cost optimization achievement"
            ],
            "execution_metrics": [
                "On-time completion rate",
                "Budget variance reduction",
                "Milestone achievement rate",
                "Resource utilization efficiency"
            ],
            "risk_metrics": [
                "Portfolio risk score reduction",
                "High-risk initiative mitigation",
                "Risk incident frequency",
                "Contingency activation rate"
            ],
            "strategic_metrics": [
                "Strategic alignment score",
                "Stakeholder satisfaction",
                "Portfolio diversification",
                "Innovation pipeline strength"
            ]
        }
    
    def generate_portfolio_report(self) -> Dict[str, Any]:
        """Generate comprehensive portfolio management report"""
        
        return {
            "executive_summary": {
                "organization": self.organization_name,
                "report_date": datetime.now().isoformat(),
                "total_initiatives": len(self.initiatives),
                "portfolio_strategy": self.current_strategy.value if self.current_strategy else "Not defined",
                "total_portfolio_investment": self.portfolio_metrics.total_investment if self.portfolio_metrics else 0,
                "expected_portfolio_roi": self.portfolio_metrics.portfolio_roi if self.portfolio_metrics else 0,
                "portfolio_health_score": self._calculate_portfolio_health_score(self.initiatives)
            },
            "portfolio_composition": {
                "initiatives_by_category": self._get_initiatives_by_category(),
                "initiatives_by_priority": self._get_initiatives_by_priority(),
                "initiatives_by_status": self._get_initiatives_by_status(),
                "investment_distribution": self._get_investment_distribution(),
                "risk_distribution": self._get_risk_distribution()
            },
            "performance_analysis": {
                "current_performance": self.monitor_portfolio_performance(),
                "optimization_opportunities": self._identify_optimization_opportunities(),
                "performance_trends": "Trend analysis available",
                "risk_assessment": self._assess_portfolio_risks(self.initiatives)
            },
            "strategic_alignment": {
                "alignment_with_objectives": "Strategic alignment assessment",
                "category_diversification": "Portfolio diversification analysis",
                "resource_optimization": "Resource allocation optimization",
                "timeline_optimization": "Implementation timeline analysis"
            },
            "recommendations": {
                "immediate_actions": self._generate_immediate_actions(),
                "medium_term_optimizations": self._generate_medium_term_optimizations(),
                "long_term_strategic_changes": self._generate_long_term_strategic_changes(),
                "governance_improvements": self._generate_governance_improvements()
            }
        }
    
    def _get_initiatives_by_category(self) -> Dict[str, int]:
        """Get initiative count by category"""
        return {category: len([init for init in self.initiatives if init.portfolio_category == category])
                for category in set(init.portfolio_category for init in self.initiatives)}
    
    def _get_initiatives_by_priority(self) -> Dict[str, int]:
        """Get initiative count by priority"""
        return {priority.value: len([init for init in self.initiatives if init.priority == priority])
                for priority in InitiativePriority}
    
    def _get_initiatives_by_status(self) -> Dict[str, int]:
        """Get initiative count by status"""
        return {status.value: len([init for init in self.initiatives if init.status == status])
                for status in InitiativeStatus}
    
    def _get_investment_distribution(self) -> Dict[str, float]:
        """Get investment distribution by category"""
        distribution = {}
        for init in self.initiatives:
            category = init.portfolio_category
            if category not in distribution:
                distribution[category] = 0
            distribution[category] += init.estimated_investment
        return distribution
    
    def _get_risk_distribution(self) -> Dict[str, float]:
        """Get risk distribution by category"""
        distribution = {}
        for init in self.initiatives:
            category = init.portfolio_category
            if category not in distribution:
                distribution[category] = []
            distribution[category].append(init.risk_score)
        
        # Calculate average risk per category
        return {category: np.mean(risks) for category, risks in distribution.items()}
    
    def _generate_immediate_actions(self) -> List[str]:
        """Generate immediate action recommendations"""
        
        actions = []
        
        # High-risk initiatives
        high_risk_count = len([init for init in self.initiatives if init.risk_score > 0.7])
        if high_risk_count > 0:
            actions.append(f"Implement enhanced monitoring for {high_risk_count} high-risk initiatives")
        
        # Behind schedule initiatives
        behind_schedule_count = len([
            init for init in self.initiatives
            if init.progress_percentage < ((datetime.now() - init.start_date).days / 
                                          (init.planned_end_date - init.start_date).days * 100)
        ])
        if behind_schedule_count > 0:
            actions.append(f"Review timeline and resources for {behind_schedule_count} behind-schedule initiatives")
        
        # Resource conflicts
        resource_conflicts = self._identify_resource_conflicts(self.initiatives)
        if resource_conflicts:
            actions.append("Resolve resource conflicts through better scheduling and allocation")
        
        return actions
    
    def _generate_medium_term_optimizations(self) -> List[str]:
        """Generate medium-term optimization recommendations"""
        
        return [
            "Implement advanced portfolio management tools and dashboards",
            "Establish cross-initiative synergy and collaboration",
            "Develop contingency planning and risk mitigation strategies",
            "Create performance incentive alignment across initiatives"
        ]
    
    def _generate_long_term_strategic_changes(self) -> List[str]:
        """Generate long-term strategic change recommendations"""
        
        return [
            "Realign portfolio strategy with evolving market conditions",
            "Build organizational capability for strategic initiative management",
            "Establish innovation and experimentation capabilities",
            "Develop ecosystem partnerships for strategic initiatives"
        ]
    
    def _generate_governance_improvements(self) -> List[str]:
        """Generate governance improvement recommendations"""
        
        return [
            "Enhance steering committee effectiveness with clear decision criteria",
            "Implement regular portfolio health assessments",
            "Establish escalation procedures for critical issues",
            "Create transparent performance reporting and accountability"
        ]
    
    def export_portfolio_analysis(self, output_path: str) -> bool:
        """Export portfolio analysis to JSON file"""
        
        try:
            analysis_data = self.generate_portfolio_report()
            
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting portfolio analysis: {str(e)}")
            return False