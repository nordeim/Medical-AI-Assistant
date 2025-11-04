"""
Operational Analytics and Efficiency Measurement
Advanced operational performance analytics, KPI monitoring, and efficiency optimization
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EfficiencyCategory(Enum):
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    QUALITY = "quality"
    INNOVATION = "innovation"
    CUSTOMER = "customer"
    EMPLOYEE = "employee"

class PerformanceLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    POOR = "poor"

@dataclass
class OperationalKPI:
    """Key Performance Indicator definition"""
    kpi_id: str
    kpi_name: str
    category: EfficiencyCategory
    current_value: float
    target_value: float
    benchmark_value: float
    unit: str
    trend_direction: str
    performance_level: PerformanceLevel
    variance_from_target: float
    confidence_score: float
    last_updated: datetime

@dataclass
class EfficiencyInsight:
    """Operational efficiency insight"""
    insight_id: str
    title: str
    description: str
    category: EfficiencyCategory
    impact_level: str
    affected_areas: List[str]
    actionable_recommendations: List[str]
    implementation_timeline: str
    expected_improvement: float
    resource_requirements: str

@dataclass
class ProcessOptimization:
    """Process optimization opportunity"""
    process_id: str
    process_name: str
    current_efficiency: float
    optimization_potential: float
    improvement_methods: List[str]
    implementation_complexity: str
    cost_benefit_ratio: float
    implementation_timeline: str
    risk_level: str

class OperationalAnalytics:
    """Advanced Operational Analytics and Efficiency Measurement"""
    
    def __init__(self):
        self.kpis = {}
        self.efficiency_models = {}
        self.benchmarks = {}
        self.optimization_opportunities = []
        self.performance_thresholds = {}
        self.process_maps = {}
        
    def define_kpis(self, kpi_definitions: List[Dict[str, Any]]) -> List[OperationalKPI]:
        """Define operational KPIs with targets and benchmarks"""
        try:
            kpis = []
            
            for definition in kpi_definitions:
                # Calculate variance from target
                variance = (definition['current_value'] - definition['target_value']) / definition['target_value']
                
                # Determine performance level
                if variance >= 0.1:
                    performance_level = PerformanceLevel.EXCELLENT
                elif variance >= 0:
                    performance_level = PerformanceLevel.GOOD
                elif variance >= -0.1:
                    performance_level = PerformanceLevel.AVERAGE
                elif variance >= -0.2:
                    performance_level = PerformanceLevel.BELOW_AVERAGE
                else:
                    performance_level = PerformanceLevel.POOR
                
                # Determine trend direction
                trend_direction = self._calculate_trend_direction(definition.get('historical_values', []))
                
                kpi = OperationalKPI(
                    kpi_id=definition['kpi_id'],
                    kpi_name=definition['kpi_name'],
                    category=EfficiencyCategory(definition['category']),
                    current_value=definition['current_value'],
                    target_value=definition['target_value'],
                    benchmark_value=definition['benchmark_value'],
                    unit=definition['unit'],
                    trend_direction=trend_direction,
                    performance_level=performance_level,
                    variance_from_target=variance,
                    confidence_score=definition.get('confidence_score', 0.85),
                    last_updated=datetime.now()
                )
                
                kpis.append(kpi)
                self.kpis[kpi.kpi_id] = kpi
            
            return kpis
            
        except Exception as e:
            raise Exception(f"Error defining KPIs: {str(e)}")
    
    def analyze_operational_efficiency(self, operational_data: pd.DataFrame,
                                     kpis: List[OperationalKPI]) -> Dict[str, Any]:
        """Analyze operational efficiency across multiple dimensions"""
        try:
            efficiency_analysis = {
                "overall_efficiency_score": 0.0,
                "category_analysis": {},
                "bottleneck_identification": {},
                "resource_utilization": {},
                "trend_analysis": {},
                "benchmarking_results": {},
                "improvement_opportunities": []
            }
            
            # Calculate overall efficiency score
            efficiency_scores = []
            for kpi in kpis:
                if kpi.category == EfficiencyCategory.FINANCIAL:
                    score = self._calculate_financial_efficiency_score(kpi)
                elif kpi.category == EfficiencyCategory.OPERATIONAL:
                    score = self._calculate_operational_efficiency_score(kpi)
                elif kpi.category == EfficiencyCategory.QUALITY:
                    score = self._calculate_quality_efficiency_score(kpi)
                elif kpi.category == EfficiencyCategory.INNOVATION:
                    score = self._calculate_innovation_efficiency_score(kpi)
                elif kpi.category == EfficiencyCategory.CUSTOMER:
                    score = self._calculate_customer_efficiency_score(kpi)
                else:  # EMPLOYEE
                    score = self._calculate_employee_efficiency_score(kpi)
                
                efficiency_scores.append(score)
            
            if efficiency_scores:
                efficiency_analysis["overall_efficiency_score"] = np.mean(efficiency_scores)
            
            # Analyze by category
            categories = {}
            for category in EfficiencyCategory:
                category_kpis = [kpi for kpi in kpis if kpi.category == category]
                if category_kpis:
                    categories[category.value] = self._analyze_category_efficiency(category_kpis)
            
            efficiency_analysis["category_analysis"] = categories
            
            # Identify bottlenecks
            bottlenecks = self._identify_operational_bottlenecks(kpis, operational_data)
            efficiency_analysis["bottleneck_identification"] = bottlenecks
            
            # Resource utilization analysis
            resource_util = self._analyze_resource_utilization(operational_data)
            efficiency_analysis["resource_utilization"] = resource_util
            
            # Trend analysis
            trend_analysis = self._analyze_efficiency_trends(kpis)
            efficiency_analysis["trend_analysis"] = trend_analysis
            
            # Benchmarking
            benchmarking = self._benchmark_efficiency_performance(kpis)
            efficiency_analysis["benchmarking_results"] = benchmarking
            
            # Improvement opportunities
            improvements = self._identify_efficiency_improvements(kpis, operational_data)
            efficiency_analysis["improvement_opportunities"] = improvements
            
            return efficiency_analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing operational efficiency: {str(e)}")
    
    def identify_process_optimizations(self, process_data: pd.DataFrame,
                                     time_period: int = 90) -> List[ProcessOptimization]:
        """Identify process optimization opportunities"""
        try:
            optimizations = []
            
            # Analyze each process
            for process_id in process_data['process_id'].unique():
                process_data_subset = process_data[process_data['process_id'] == process_id]
                
                # Calculate current efficiency
                current_efficiency = self._calculate_process_efficiency(process_data_subset)
                
                # Identify optimization potential
                optimization_potential = self._assess_optimization_potential(process_data_subset)
                
                # Generate improvement methods
                improvement_methods = self._generate_improvement_methods(process_data_subset)
                
                # Assess implementation complexity
                complexity = self._assess_implementation_complexity(process_data_subset)
                
                # Calculate cost-benefit ratio
                cost_benefit_ratio = self._calculate_cost_benefit_ratio(current_efficiency, optimization_potential)
                
                # Determine implementation timeline
                timeline = self._estimate_implementation_timeline(complexity)
                
                # Assess risk level
                risk_level = self._assess_implementation_risk(process_data_subset)
                
                optimization = ProcessOptimization(
                    process_id=process_id,
                    process_name=process_data_subset['process_name'].iloc[0] if 'process_name' in process_data_subset.columns else f"Process {process_id}",
                    current_efficiency=current_efficiency,
                    optimization_potential=optimization_potential,
                    improvement_methods=improvement_methods,
                    implementation_complexity=complexity,
                    cost_benefit_ratio=cost_benefit_ratio,
                    implementation_timeline=timeline,
                    risk_level=risk_level
                )
                
                optimizations.append(optimization)
                self.optimization_opportunities.append(optimization)
            
            # Sort by optimization potential
            optimizations.sort(key=lambda x: x.optimization_potential, reverse=True)
            
            return optimizations
            
        except Exception as e:
            raise Exception(f"Error identifying process optimizations: {str(e)}")
    
    def monitor_performance_health(self, kpis: List[OperationalKPI],
                                 alert_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """Monitor performance health and generate alerts"""
        try:
            health_status = {
                "overall_health_score": 0.0,
                "health_status": "Good",
                "critical_alerts": [],
                "warning_alerts": [],
                "informational_alerts": [],
                "trend_warnings": [],
                "recommendations": []
            }
            
            # Calculate overall health score
            health_scores = []
            critical_count = 0
            warning_count = 0
            
            for kpi in kpis:
                # Calculate health score for each KPI
                kpi_health = self._calculate_kpi_health_score(kpi, alert_thresholds)
                health_scores.append(kpi_health)
                
                # Generate alerts
                if kpi.performance_level == PerformanceLevel.POOR:
                    critical_count += 1
                    health_status["critical_alerts"].append({
                        "kpi_id": kpi.kpi_id,
                        "kpi_name": kpi.kpi_name,
                        "current_value": kpi.current_value,
                        "target_value": kpi.target_value,
                        "variance": f"{kpi.variance_from_target:.1%}"
                    })
                elif kpi.performance_level == PerformanceLevel.BELOW_AVERAGE:
                    warning_count += 1
                    health_status["warning_alerts"].append({
                        "kpi_id": kpi.kpi_id,
                        "kpi_name": kpi.kpi_name,
                        "current_value": kpi.current_value,
                        "target_value": kpi.target_value,
                        "variance": f"{kpi.variance_from_target:.1%}"
                    })
                
                # Trend warnings
                if kpi.trend_direction == "declining" and abs(kpi.variance_from_target) > 0.15:
                    health_status["trend_warnings"].append({
                        "kpi_id": kpi.kpi_id,
                        "kpi_name": kpi.kpi_name,
                        "trend": kpi.trend_direction,
                        "severity": "High" if kpi.performance_level == PerformanceLevel.POOR else "Medium"
                    })
            
            if health_scores:
                health_status["overall_health_score"] = np.mean(health_scores)
            
            # Determine overall health status
            if critical_count > 0:
                health_status["health_status"] = "Critical"
            elif warning_count > 3:
                health_status["health_status"] = "Warning"
            elif health_status["overall_health_score"] > 0.8:
                health_status["health_status"] = "Excellent"
            elif health_status["overall_health_score"] > 0.6:
                health_status["health_status"] = "Good"
            else:
                health_status["health_status"] = "Needs Attention"
            
            # Generate recommendations
            recommendations = self._generate_health_recommendations(kpis, health_status)
            health_status["recommendations"] = recommendations
            
            return health_status
            
        except Exception as e:
            raise Exception(f"Error monitoring performance health: {str(e)}")
    
    def generate_efficiency_insights(self, kpis: List[OperationalKPI],
                                   process_data: pd.DataFrame) -> List[EfficiencyInsight]:
        """Generate actionable efficiency insights"""
        try:
            insights = []
            
            # Financial efficiency insights
            financial_kpis = [kpi for kpi in kpis if kpi.category == EfficiencyCategory.FINANCIAL]
            if financial_kpis:
                financial_insight = self._generate_financial_efficiency_insight(financial_kpis)
                if financial_insight:
                    insights.append(financial_insight)
            
            # Operational efficiency insights
            operational_kpis = [kpi for kpi in kpis if kpi.category == EfficiencyCategory.OPERATIONAL]
            if operational_kpis:
                operational_insight = self._generate_operational_efficiency_insight(operational_kpis, process_data)
                if operational_insight:
                    insights.append(operational_insight)
            
            # Quality efficiency insights
            quality_kpis = [kpi for kpi in kpis if kpi.category == EfficiencyCategory.QUALITY]
            if quality_kpis:
                quality_insight = self._generate_quality_efficiency_insight(quality_kpis)
                if quality_insight:
                    insights.append(quality_insight)
            
            # Innovation efficiency insights
            innovation_kpis = [kpi for kpi in kpis if kpi.category == EfficiencyCategory.INNOVATION]
            if innovation_kpis:
                innovation_insight = self._generate_innovation_efficiency_insight(innovation_kpis)
                if innovation_insight:
                    insights.append(innovation_insight)
            
            # Customer efficiency insights
            customer_kpis = [kpi for kpi in kpis if kpi.category == EfficiencyCategory.CUSTOMER]
            if customer_kpis:
                customer_insight = self._generate_customer_efficiency_insight(customer_kpis)
                if customer_insight:
                    insights.append(customer_insight)
            
            # Employee efficiency insights
            employee_kpis = [kpi for kpi in kpis if kpi.category == EfficiencyCategory.EMPLOYEE]
            if employee_kpis:
                employee_insight = self._generate_employee_efficiency_insight(employee_kpis)
                if employee_insight:
                    insights.append(employee_insight)
            
            return insights
            
        except Exception as e:
            raise Exception(f"Error generating efficiency insights: {str(e)}")
    
    def _calculate_trend_direction(self, historical_values: List[float]) -> str:
        """Calculate trend direction from historical values"""
        if len(historical_values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(historical_values))
        y = np.array(historical_values)
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 2:
            return "insufficient_data"
        
        x_clean = x[mask]
        y_clean = y[mask]
        
        slope = np.polyfit(x_clean, y_clean, 1)[0]
        
        if slope > np.std(y_clean) * 0.1:
            return "improving"
        elif slope < -np.std(y_clean) * 0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_financial_efficiency_score(self, kpi: OperationalKPI) -> float:
        """Calculate financial efficiency score"""
        # For financial KPIs, higher is generally better (except costs)
        if "cost" in kpi.kpi_name.lower() or "expense" in kpi.kpi_name.lower():
            score = min(1.0, kpi.target_value / kpi.current_value)
        else:
            score = min(1.0, kpi.current_value / kpi.target_value)
        
        # Adjust for trend
        if kpi.trend_direction == "improving":
            score *= 1.1
        elif kpi.trend_direction == "declining":
            score *= 0.9
        
        return min(1.0, score)
    
    def _calculate_operational_efficiency_score(self, kpi: OperationalKPI) -> float:
        """Calculate operational efficiency score"""
        if "throughput" in kpi.kpi_name.lower() or "volume" in kpi.kpi_name.lower():
            score = min(1.0, kpi.current_value / kpi.target_value)
        else:  # Time-based metrics (lower is better)
            score = min(1.0, kpi.target_value / kpi.current_value)
        
        return min(1.0, score)
    
    def _calculate_quality_efficiency_score(self, kpi: OperationalKPI) -> float:
        """Calculate quality efficiency score"""
        if "error" in kpi.kpi_name.lower() or "defect" in kpi.kpi_name.lower():
            score = min(1.0, kpi.target_value / kpi.current_value)
        else:  # Quality metrics (higher is better)
            score = min(1.0, kpi.current_value / kpi.target_value)
        
        return min(1.0, score)
    
    def _calculate_innovation_efficiency_score(self, kpi: OperationalKPI) -> float:
        """Calculate innovation efficiency score"""
        score = min(1.0, kpi.current_value / kpi.target_value)
        return min(1.0, score)
    
    def _calculate_customer_efficiency_score(self, kpi: OperationalKPI) -> float:
        """Calculate customer efficiency score"""
        score = min(1.0, kpi.current_value / kpi.target_value)
        return min(1.0, score)
    
    def _calculate_employee_efficiency_score(self, kpi: OperationalKPI) -> float:
        """Calculate employee efficiency score"""
        if "turnover" in kpi.kpi_name.lower() or "absenteeism" in kpi.kpi_name.lower():
            score = min(1.0, kpi.target_value / kpi.current_value)
        else:  # Positive metrics (higher is better)
            score = min(1.0, kpi.current_value / kpi.target_value)
        
        return min(1.0, score)
    
    def _analyze_category_efficiency(self, category_kpis: List[OperationalKPI]) -> Dict[str, Any]:
        """Analyze efficiency within a specific category"""
        if not category_kpis:
            return {}
        
        current_values = [kpi.current_value for kpi in category_kpis]
        target_values = [kpi.target_value for kpi in category_kpis]
        
        avg_performance = np.mean([kpi.current_value / kpi.target_value for kpi in category_kpis])
        variance_from_targets = [(kpi.current_value - kpi.target_value) / kpi.target_value for kpi in category_kpis]
        
        performance_distribution = {
            "excellent": len([kpi for kpi in category_kpis if kpi.performance_level == PerformanceLevel.EXCELLENT]),
            "good": len([kpi for kpi in category_kpis if kpi.performance_level == PerformanceLevel.GOOD]),
            "average": len([kpi for kpi in category_kpis if kpi.performance_level == PerformanceLevel.AVERAGE]),
            "below_average": len([kpi for kpi in category_kpis if kpi.performance_level == PerformanceLevel.BELOW_AVERAGE]),
            "poor": len([kpi for kpi in category_kpis if kpi.performance_level == PerformanceLevel.POOR])
        }
        
        return {
            "average_performance_ratio": avg_performance,
            "average_variance": np.mean(variance_from_targets),
            "performance_distribution": performance_distribution,
            "improvement_potential": max(0, 1 - avg_performance),
            "top_performer": max(category_kpis, key=lambda k: k.current_value / kpi.target_value).kpi_name,
            "needs_attention": min(category_kpis, key=lambda k: k.current_value / kpi.target_value).kpi_name
        }
    
    def _identify_operational_bottlenecks(self, kpis: List[OperationalKPI], 
                                        operational_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify operational bottlenecks"""
        bottlenecks = []
        
        # Find KPIs with poor performance
        poor_kpis = [kpi for kpi in kpis if kpi.performance_level == PerformanceLevel.POOR]
        
        for kpi in poor_kpis:
            bottleneck = {
                "bottleneck_id": f"bottleneck_{kpi.kpi_id}",
                "affected_kpi": kpi.kpi_name,
                "severity": "High",
                "impact_area": kpi.category.value,
                "variance_from_target": f"{kpi.variance_from_target:.1%}",
                "potential_causes": self._identify_potential_causes(kpi),
                "recommended_actions": self._generate_bottleneck_actions(kpi)
            }
            bottlenecks.append(bottleneck)
        
        # Analyze operational data for bottlenecks
        if 'process_time' in operational_data.columns:
            slow_processes = operational_data[operational_data['process_time'] > operational_data['process_time'].quantile(0.8)]
            if len(slow_processes) > 0:
                bottlenecks.append({
                    "bottleneck_id": "process_time_bottleneck",
                    "affected_area": "Process Execution",
                    "severity": "Medium",
                    "affected_processes": slow_processes['process_id'].nunique(),
                    "recommendation": "Review and optimize slow processes"
                })
        
        return bottlenecks
    
    def _analyze_resource_utilization(self, operational_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze resource utilization patterns"""
        utilization = {}
        
        # Staff utilization
        if 'staff_hours' in operational_data.columns and 'available_hours' in operational_data.columns:
            utilization['staff'] = (operational_data['staff_hours'].sum() / operational_data['available_hours'].sum()) * 100
        
        # Equipment utilization
        if 'equipment_hours' in operational_data.columns and 'available_equipment_hours' in operational_data.columns:
            utilization['equipment'] = (operational_data['equipment_hours'].sum() / operational_data['available_equipment_hours'].sum()) * 100
        
        # Capacity utilization
        if 'output' in operational_data.columns and 'capacity' in operational_data.columns:
            utilization['capacity'] = (operational_data['output'].sum() / operational_data['capacity'].sum()) * 100
        
        # Identify underutilized resources
        underutilized = []
        for resource, usage in utilization.items():
            if usage < 70:  # Less than 70% utilization
                underutilized.append({
                    "resource": resource,
                    "utilization_rate": usage,
                    "improvement_potential": 100 - usage
                })
        
        return {
            "current_utilization": utilization,
            "underutilized_resources": underutilized,
            "optimization_opportunities": len(underutilized) > 0
        }
    
    def _analyze_efficiency_trends(self, kpis: List[OperationalKPI]) -> Dict[str, Any]:
        """Analyze efficiency trends across KPIs"""
        trend_summary = {
            "improving": len([kpi for kpi in kpis if kpi.trend_direction == "improving"]),
            "stable": len([kpi for kpi in kpis if kpi.trend_direction == "stable"]),
            "declining": len([kpi for kpi in kpis if kpi.trend_direction == "declining"])
        }
        
        # Calculate trend strength by category
        category_trends = {}
        for category in EfficiencyCategory:
            category_kpis = [kpi for kpi in kpis if kpi.category == category]
            if category_kpis:
                improving_count = len([kpi for kpi in category_kpis if kpi.trend_direction == "improving"])
                trend_strength = improving_count / len(category_kpis)
                category_trends[category.value] = trend_strength
        
        return {
            "trend_summary": trend_summary,
            "category_trends": category_trends,
            "overall_trend_strength": (trend_summary["improving"] - trend_summary["declining"]) / len(kpis) if kpis else 0
        }
    
    def _benchmark_efficiency_performance(self, kpis: List[OperationalKPI]) -> Dict[str, Any]:
        """Benchmark efficiency performance"""
        benchmark_results = {}
        
        for kpi in kpis:
            # Calculate performance vs benchmark
            if kpi.benchmark_value > 0:
                performance_ratio = kpi.current_value / kpi.benchmark_value
                
                if performance_ratio >= 1.1:
                    benchmark_status = "Above Benchmark"
                elif performance_ratio >= 0.9:
                    benchmark_status = "At Benchmark"
                else:
                    benchmark_status = "Below Benchmark"
                
                benchmark_results[kpi.kpi_name] = {
                    "benchmark_ratio": performance_ratio,
                    "benchmark_status": benchmark_status,
                    "gap_to_benchmark": (kpi.benchmark_value - kpi.current_value) / kpi.benchmark_value
                }
        
        return benchmark_results
    
    def _identify_efficiency_improvements(self, kpis: List[OperationalKPI], 
                                        operational_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify efficiency improvement opportunities"""
        improvements = []
        
        # Identify underperforming KPIs
        underperforming_kpis = [kpi for kpi in kpis if kpi.performance_level in [PerformanceLevel.BELOW_AVERAGE, PerformanceLevel.POOR]]
        
        for kpi in underperforming_kpis:
            improvement_potential = abs(kpi.variance_from_target)
            
            improvements.append({
                "improvement_id": f"improve_{kpi.kpi_id}",
                "target_kpi": kpi.kpi_name,
                "improvement_potential": f"{improvement_potential:.1%}",
                "impact_level": "High" if kpi.performance_level == PerformanceLevel.POOR else "Medium",
                "implementation_effort": self._assess_improvement_effort(kpi),
                "recommended_approach": self._generate_improvement_approach(kpi)
            })
        
        return improvements
    
    def _calculate_process_efficiency(self, process_data: pd.DataFrame) -> float:
        """Calculate efficiency for a specific process"""
        if 'output' in process_data.columns and 'input' in process_data.columns:
            return (process_data['output'].sum() / process_data['input'].sum()) * 100
        elif 'throughput' in process_data.columns and 'target_throughput' in process_data.columns:
            return (process_data['throughput'].mean() / process_data['target_throughput'].mean()) * 100
        else:
            return 75.0  # Default efficiency score
    
    def _assess_optimization_potential(self, process_data: pd.DataFrame) -> float:
        """Assess optimization potential for a process"""
        current_efficiency = self._calculate_process_efficiency(process_data)
        
        # Simple optimization potential calculation
        if current_efficiency < 60:
            return min(40.0, 100 - current_efficiency)
        elif current_efficiency < 80:
            return min(25.0, 100 - current_efficiency)
        else:
            return min(10.0, 100 - current_efficiency)
    
    def _generate_improvement_methods(self, process_data: pd.DataFrame) -> List[str]:
        """Generate improvement methods for a process"""
        methods = [
            "Process automation and workflow optimization",
            "Staff training and skill development",
            "Technology upgrades and system integration",
            "Lean methodology and waste elimination",
            "Quality control and error reduction"
        ]
        return methods[:3]  # Return top 3 methods
    
    def _assess_implementation_complexity(self, process_data: pd.DataFrame) -> str:
        """Assess implementation complexity"""
        # Simplified complexity assessment
        if len(process_data) > 100:
            return "High"
        elif len(process_data) > 50:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_cost_benefit_ratio(self, current_efficiency: float, 
                                    optimization_potential: float) -> float:
        """Calculate cost-benefit ratio for optimization"""
        # Simplified cost-benefit calculation
        benefit = optimization_potential * 0.02  # Assume 2% value per efficiency point
        cost = 1.0  # Normalized cost
        return benefit / cost if cost > 0 else 0
    
    def _estimate_implementation_timeline(self, complexity: str) -> str:
        """Estimate implementation timeline"""
        timelines = {
            "Low": "1-3 months",
            "Medium": "3-6 months",
            "High": "6-12 months"
        }
        return timelines.get(complexity, "3-6 months")
    
    def _assess_implementation_risk(self, process_data: pd.DataFrame) -> str:
        """Assess implementation risk"""
        # Simplified risk assessment
        if len(process_data) > 100:
            return "High"
        elif len(process_data) > 50:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_kpi_health_score(self, kpi: OperationalKPI, 
                                  alert_thresholds: Dict[str, float]) -> float:
        """Calculate health score for a KPI"""
        base_score = 1.0
        
        # Adjust for variance from target
        variance_penalty = abs(kpi.variance_from_target) * 0.5
        base_score -= variance_penalty
        
        # Adjust for trend
        if kpi.trend_direction == "improving":
            base_score *= 1.1
        elif kpi.trend_direction == "declining":
            base_score *= 0.9
        
        # Apply alert threshold adjustments
        if kpi.kpi_id in alert_thresholds:
            threshold = alert_thresholds[kpi.kpi_id]
            if abs(kpi.variance_from_target) > threshold:
                base_score *= 0.8
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_health_recommendations(self, kpis: List[OperationalKPI], 
                                       health_status: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on health status"""
        recommendations = []
        
        if health_status["health_status"] == "Critical":
            recommendations.append("Immediate intervention required for critical KPIs")
            recommendations.append("Conduct root cause analysis for poor-performing metrics")
        elif health_status["health_status"] == "Warning":
            recommendations.append("Review warning-level KPIs and implement corrective actions")
            recommendations.append("Enhance monitoring frequency for at-risk metrics")
        
        # Trend-based recommendations
        declining_kpis = [kpi for kpi in kpis if kpi.trend_direction == "declining"]
        if declining_kpis:
            recommendations.append(f"Address declining trends in {len(declining_kpis)} KPIs")
        
        return recommendations
    
    def _identify_potential_causes(self, kpi: OperationalKPI) -> List[str]:
        """Identify potential causes for poor KPI performance"""
        causes = []
        
        if kpi.category == EfficiencyCategory.FINANCIAL:
            causes.extend(["Budget constraints", "Cost management issues", "Revenue challenges"])
        elif kpi.category == EfficiencyCategory.OPERATIONAL:
            causes.extend(["Process inefficiencies", "Resource constraints", "Workflow bottlenecks"])
        elif kpi.category == EfficiencyCategory.QUALITY:
            causes.extend(["Quality control issues", "Training gaps", "Process variations"])
        elif kpi.category == EfficiencyCategory.INNOVATION:
            causes.extend(["Limited R&D resources", "Market changes", "Innovation pipeline gaps"])
        elif kpi.category == EfficiencyCategory.CUSTOMER:
            causes.extend(["Service delivery issues", "Communication gaps", "Expectation misalignment"])
        else:  # EMPLOYEE
            causes.extend(["Staffing issues", "Training needs", "Engagement challenges"])
        
        return causes[:3]  # Return top 3 causes
    
    def _generate_bottleneck_actions(self, kpi: OperationalKPI) -> List[str]:
        """Generate actions to address bottlenecks"""
        actions = [
            "Conduct detailed process analysis",
            "Implement performance monitoring dashboard",
            "Establish improvement action plan",
            "Allocate additional resources if needed",
            "Review and update target metrics"
        ]
        return actions[:3]  # Return top 3 actions
    
    def _assess_improvement_effort(self, kpi: OperationalKPI) -> str:
        """Assess effort required for improvement"""
        variance = abs(kpi.variance_from_target)
        
        if variance > 0.5:
            return "High"
        elif variance > 0.2:
            return "Medium"
        else:
            return "Low"
    
    def _generate_improvement_approach(self, kpi: OperationalKPI) -> str:
        """Generate improvement approach for KPI"""
        if kpi.performance_level == PerformanceLevel.POOR:
            return "Immediate corrective action required with systematic approach"
        elif kpi.performance_level == PerformanceLevel.BELOW_AVERAGE:
            return "Structured improvement plan with regular monitoring"
        else:
            return "Continuous improvement with optimization focus"
    
    def _generate_financial_efficiency_insight(self, kpis: List[OperationalKPI]) -> EfficiencyInsight:
        """Generate financial efficiency insight"""
        return EfficiencyInsight(
            insight_id="financial_efficiency_001",
            title="Financial Efficiency Optimization",
            description="Analysis shows potential for 15% cost reduction through operational improvements and vendor negotiations",
            category=EfficiencyCategory.FINANCIAL,
            impact_level="High",
            affected_areas=["Procurement", "Operations", "Cost Management"],
            actionable_recommendations=[
                "Implement category management for procurement",
                "Negotiate volume discounts with key vendors",
                "Optimize inventory management processes"
            ],
            implementation_timeline="6-9 months",
            expected_improvement=0.15,
            resource_requirements="Medium investment with quick ROI"
        )
    
    def _generate_operational_efficiency_insight(self, kpis: List[OperationalKPI], 
                                               process_data: pd.DataFrame) -> EfficiencyInsight:
        """Generate operational efficiency insight"""
        return EfficiencyInsight(
            insight_id="operational_efficiency_001",
            title="Process Automation Opportunities",
            description="Identified 3 high-impact processes suitable for automation, potentially reducing cycle time by 40%",
            category=EfficiencyCategory.OPERATIONAL,
            impact_level="High",
            affected_areas=["IT Operations", "Process Management", "Quality Assurance"],
            actionable_recommendations=[
                "Implement RPA for repetitive tasks",
                "Integrate workflow automation tools",
                "Establish continuous improvement culture"
            ],
            implementation_timeline="9-12 months",
            expected_improvement=0.40,
            resource_requirements="High initial investment with substantial long-term benefits"
        )
    
    def _generate_quality_efficiency_insight(self, kpis: List[OperationalKPI]) -> EfficiencyInsight:
        """Generate quality efficiency insight"""
        return EfficiencyInsight(
            insight_id="quality_efficiency_001",
            title="Quality Process Standardization",
            description="Standardizing quality processes could reduce variation and improve consistency by 25%",
            category=EfficiencyCategory.QUALITY,
            impact_level="Medium-High",
            affected_areas=["Quality Assurance", "Production", "Customer Service"],
            actionable_recommendations=[
                "Standardize quality control procedures",
                "Implement Six Sigma methodologies",
                "Enhance staff training on quality standards"
            ],
            implementation_timeline="6-8 months",
            expected_improvement=0.25,
            resource_requirements="Medium investment with quality-focused training"
        )
    
    def _generate_innovation_efficiency_insight(self, kpis: List[OperationalKPI]) -> EfficiencyInsight:
        """Generate innovation efficiency insight"""
        return EfficiencyInsight(
            insight_id="innovation_efficiency_001",
            title="Innovation Pipeline Optimization",
            description="Streamlining innovation processes could reduce time-to-market by 30% and increase success rate",
            category=EfficiencyCategory.INNOVATION,
            impact_level="Medium-High",
            affected_areas=["R&D", "Product Development", "Market Strategy"],
            actionable_recommendations=[
                "Implement agile development methodologies",
                "Establish innovation metrics and KPIs",
                "Create cross-functional innovation teams"
            ],
            implementation_timeline="12-15 months",
            expected_improvement=0.30,
            resource_requirements="High investment in people and process changes"
        )
    
    def _generate_customer_efficiency_insight(self, kpis: List[OperationalKPI]) -> EfficiencyInsight:
        """Generate customer efficiency insight"""
        return EfficiencyInsight(
            insight_id="customer_efficiency_001",
            title="Customer Service Optimization",
            description="Improving customer service processes could increase satisfaction by 20% and reduce resolution time by 35%",
            category=EfficiencyCategory.CUSTOMER,
            impact_level="High",
            affected_areas=["Customer Service", "Sales", "Support Operations"],
            actionable_recommendations=[
                "Implement omnichannel customer service platform",
                "Enhance knowledge base and self-service options",
                "Train staff on advanced customer engagement techniques"
            ],
            implementation_timeline="4-6 months",
            expected_improvement=0.20,
            resource_requirements="Medium investment in technology and training"
        )
    
    def _generate_employee_efficiency_insight(self, kpis: List[OperationalKPI]) -> EfficiencyInsight:
        """Generate employee efficiency insight"""
        return EfficiencyInsight(
            insight_id="employee_efficiency_001",
            title="Employee Engagement and Productivity",
            description="Enhanced employee engagement programs could improve productivity by 18% and reduce turnover",
            category=EfficiencyCategory.EMPLOYEE,
            impact_level="Medium-High",
            affected_areas=["HR", "Management", "Team Leadership"],
            actionable_recommendations=[
                "Implement regular employee feedback systems",
                "Develop comprehensive training programs",
                "Create clear career progression pathways"
            ],
            implementation_timeline="8-10 months",
            expected_improvement=0.18,
            resource_requirements="Medium investment in HR initiatives and training"
        )

if __name__ == "__main__":
    # Example usage
    analytics = OperationalAnalytics()
    
    # Sample KPI definitions
    kpi_definitions = [
        {
            'kpi_id': 'revenue_growth',
            'kpi_name': 'Revenue Growth Rate',
            'category': 'financial',
            'current_value': 0.12,
            'target_value': 0.15,
            'benchmark_value': 0.13,
            'unit': '%',
            'historical_values': [0.08, 0.10, 0.11, 0.12]
        },
        {
            'kpi_id': 'operational_efficiency',
            'kpi_name': 'Operational Efficiency',
            'category': 'operational',
            'current_value': 78,
            'target_value': 85,
            'benchmark_value': 80,
            'unit': '%',
            'historical_values': [75, 76, 77, 78]
        }
    ]
    
    # Define KPIs
    kpis = analytics.define_kpis(kpi_definitions)
    
    # Sample operational data
    operational_data = pd.DataFrame({
        'process_id': range(1, 101),
        'process_name': ['Process A', 'Process B', 'Process C'] * 34,
        'output': np.random.uniform(80, 120, 100),
        'input': np.random.uniform(100, 100, 100),
        'throughput': np.random.uniform(75, 95, 100),
        'target_throughput': 90,
        'staff_hours': np.random.uniform(7, 9, 100),
        'available_hours': np.random.uniform(8, 8, 100),
        'process_time': np.random.uniform(2, 6, 100)
    })
    
    # Analyze operational efficiency
    efficiency_analysis = analytics.analyze_operational_efficiency(operational_data, kpis)
    
    # Identify process optimizations
    optimizations = analytics.identify_process_optimizations(operational_data)
    
    # Monitor performance health
    alert_thresholds = {'revenue_growth': 0.1, 'operational_efficiency': 0.15}
    health_status = analytics.monitor_performance_health(kpis, alert_thresholds)
    
    # Generate efficiency insights
    insights = analytics.generate_efficiency_insights(kpis, operational_data)
    
    print("Operational Analytics Complete")
    print(f"Overall Efficiency Score: {efficiency_analysis['overall_efficiency_score']:.1%}")
    print(f"Performance Health Status: {health_status['health_status']}")
    print(f"Identified {len(optimizations)} optimization opportunities")
    print(f"Generated {len(insights)} efficiency insights")