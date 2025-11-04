"""
Production Healthcare Analytics and Business Intelligence System
Implements comprehensive analytics dashboard with healthcare KPIs and metrics
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from collections import defaultdict, OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class MetricCategory(Enum):
    """Healthcare metric categories"""
    SAFETY = "safety"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    COST = "cost"
    OUTCOMES = "outcomes"
    PATIENT_SATISFACTION = "patient_satisfaction"
    OPERATIONAL = "operational"

class DashboardType(Enum):
    """Types of analytics dashboards"""
    EXECUTIVE = "executive"
    CLINICAL = "clinical"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    QUALITY = "quality"

@dataclass
class KPI:
    """Healthcare Key Performance Indicator"""
    kpi_id: str
    name: str
    category: MetricCategory
    current_value: float
    target_value: float
    unit: str
    trend: str  # "improving", "declining", "stable"
    last_updated: datetime
    data_source: str
    calculation_method: str
    threshold_values: Dict[str, float] = field(default_factory=dict)
    benchmark_values: Dict[str, float] = field(default_factory=dict)

@dataclass
class AnalyticsDashboard:
    """Complete analytics dashboard configuration"""
    dashboard_id: str
    dashboard_type: DashboardType
    title: str
    description: str
    kpis: List[KPI]
    visualizations: List[Dict[str, Any]]
    filters: List[str]
    refresh_frequency: str
    alert_config: Dict[str, Any]
    export_options: List[str]

class HealthcareAnalyticsEngine:
    """Production analytics engine for healthcare data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.engines = {}
        self.current_kpis = {}
        self.dashboards = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup analytics logging"""
        logger = logging.getLogger("healthcare_analytics")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def initialize_analytics(self) -> None:
        """Initialize analytics engine connections and calculations"""
        try:
            # Initialize database connections
            await self._initialize_connections()
            
            # Initialize healthcare KPIs
            await self._initialize_healthcare_kpis()
            
            # Initialize dashboards
            await self._initialize_dashboards()
            
            self.logger.info("Analytics engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Analytics initialization failed: {str(e)}")
            raise
    
    async def _initialize_connections(self) -> None:
        """Initialize database connections for analytics"""
        # Initialize analytics warehouse
        warehouse_connection = self.config.get("analytics_warehouse_connection")
        if warehouse_connection:
            self.engines["analytics_warehouse"] = create_engine(warehouse_connection)
        
        # Initialize data mart connections
        for mart_name, connection_string in self.config.get("data_marts", {}).items():
            self.engines[mart_name] = create_engine(connection_string)
    
    async def _initialize_healthcare_kpis(self) -> None:
        """Initialize healthcare-specific KPIs"""
        self.current_kpis = {
            # Patient Safety Metrics
            "patient_safety_score": KPI(
                kpi_id="safety_score_001",
                name="Patient Safety Score",
                category=MetricCategory.SAFETY,
                current_value=0.0,
                target_value=95.0,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(),
                data_source="patient_safety_events",
                calculation_method="weighted_average"
            ),
            
            # Clinical Quality Metrics
            "clinical_quality_index": KPI(
                kpi_id="quality_index_001",
                name="Clinical Quality Index",
                category=MetricCategory.QUALITY,
                current_value=0.0,
                target_value=90.0,
                unit="score",
                trend="stable",
                last_updated=datetime.now(),
                data_source="quality_metrics",
                calculation_method="composite_index"
            ),
            
            # Operational Efficiency
            "operational_efficiency": KPI(
                kpi_id="efficiency_001",
                name="Operational Efficiency",
                category=MetricCategory.EFFICIENCY,
                current_value=0.0,
                target_value=85.0,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(),
                data_source="operational_metrics",
                calculation_method="resource_utilization"
            ),
            
            # Cost Management
            "cost_per_encounter": KPI(
                kpi_id="cost_001",
                name="Cost Per Encounter",
                category=MetricCategory.COST,
                current_value=0.0,
                target_value=1500.0,
                unit="currency",
                trend="stable",
                last_updated=datetime.now(),
                data_source="financial_data",
                calculation_method="total_cost_div_encounters"
            ),
            
            # Patient Outcomes
            "readmission_rate": KPI(
                kpi_id="outcome_001",
                name="30-Day Readmission Rate",
                category=MetricCategory.OUTCOMES,
                current_value=0.0,
                target_value=0.12,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(),
                data_source="patient_outcomes",
                calculation_method="readmissions_div_discharges"
            ),
            
            "mortality_rate": KPI(
                kpi_id="outcome_002",
                name="In-Hospital Mortality Rate",
                category=MetricCategory.OUTCOMES,
                current_value=0.0,
                target_value=0.02,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(),
                data_source="patient_outcomes",
                calculation_method="deaths_div_admissions"
            ),
            
            # Patient Satisfaction
            "patient_satisfaction_score": KPI(
                kpi_id="satisfaction_001",
                name="Patient Satisfaction Score",
                category=MetricCategory.PATIENT_SATISFACTION,
                current_value=0.0,
                target_value=4.5,
                unit="rating",
                trend="stable",
                last_updated=datetime.now(),
                data_source="patient_surveys",
                calculation_method="average_rating"
            ),
            
            # Wait Time Metrics
            "average_wait_time": KPI(
                kpi_id="operational_001",
                name="Average Wait Time",
                category=MetricCategory.OPERATIONAL,
                current_value=0.0,
                target_value=15.0,
                unit="minutes",
                trend="stable",
                last_updated=datetime.now(),
                data_source="appointment_data",
                calculation_method="average_wait_duration"
            ),
            
            # Bed Occupancy
            "bed_occupancy_rate": KPI(
                kpi_id="operational_002",
                name="Bed Occupancy Rate",
                category=MetricCategory.OPERATIONAL,
                current_value=0.0,
                target_value=0.85,
                unit="percentage",
                trend="stable",
                last_updated=datetime.now(),
                data_source="bed_management",
                calculation_method="occupied_beds_div_total_beds"
            )
        }
    
    async def _initialize_dashboards(self) -> None:
        """Initialize pre-configured dashboards"""
        self.dashboards = {
            "executive_dashboard": AnalyticsDashboard(
                dashboard_id="exec_001",
                dashboard_type=DashboardType.EXECUTIVE,
                title="Executive Healthcare Dashboard",
                description="High-level KPIs for executive decision making",
                kpis=[
                    self.current_kpis["patient_safety_score"],
                    self.current_kpis["clinical_quality_index"],
                    self.current_kpis["cost_per_encounter"],
                    self.current_kpis["patient_satisfaction_score"]
                ],
                visualizations=[
                    {"type": "line_chart", "metrics": ["safety_score", "quality_index"], "timeframe": "12_months"},
                    {"type": "gauge_chart", "metrics": ["patient_satisfaction"], "current_values_only": True},
                    {"type": "heatmap", "metrics": ["department_performance"]}
                ],
                filters=["time_period", "department", "service_line"],
                refresh_frequency="hourly",
                alert_config={
                    "critical_thresholds": {"safety_score": 85, "quality_index": 80},
                    "notification_channels": ["email", "dashboard"]
                },
                export_options=["pdf", "pptx", "excel"]
            ),
            
            "clinical_dashboard": AnalyticsDashboard(
                dashboard_id="clinical_001",
                dashboard_type=DashboardType.CLINICAL,
                title="Clinical Quality Dashboard",
                description="Clinical metrics for healthcare professionals",
                kpis=[
                    self.current_kpis["readmission_rate"],
                    self.current_kpis["mortality_rate"],
                    self.current_kpis["clinical_quality_index"]
                ],
                visualizations=[
                    {"type": "trend_chart", "metrics": ["readmission_rate", "mortality_rate"], "timeframe": "6_months"},
                    {"type": "comparative_bar", "metrics": ["quality_benchmarks"]},
                    {"type": "clinical_pathway_analysis", "metrics": ["pathway_performance"]}
                ],
                filters=["clinical_service", "time_period", "patient_population"],
                refresh_frequency="real_time",
                alert_config={
                    "critical_thresholds": {"readmission_rate": 0.15, "mortality_rate": 0.03},
                    "notification_channels": ["dashboard", "mobile_app"]
                },
                export_options=["pdf", "excel", "csv"]
            ),
            
            "operational_dashboard": AnalyticsDashboard(
                dashboard_id="operational_001",
                dashboard_type=DashboardType.OPERATIONAL,
                title="Operational Efficiency Dashboard",
                description="Operational metrics for healthcare administration",
                kpis=[
                    self.current_kpis["operational_efficiency"],
                    self.current_kpis["average_wait_time"],
                    self.current_kpis["bed_occupancy_rate"]
                ],
                visualizations=[
                    {"type": "real_time_metrics", "metrics": ["wait_times", "occupancy_rates"]},
                    {"type": "efficiency_trends", "metrics": ["operational_efficiency"], "timeframe": "daily"},
                    {"type": "resource_utilization", "metrics": ["staff_utilization", "equipment_utilization"]}
                ],
                filters=["department", "shift", "day_of_week"],
                refresh_frequency="real_time",
                alert_config={
                    "critical_thresholds": {"average_wait_time": 30, "bed_occupancy_rate": 0.95},
                    "notification_channels": ["dashboard", "sms"]
                },
                export_options=["excel", "csv", "json"]
            )
        }
    
    async def calculate_all_kpis(self, date_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, KPI]:
        """Calculate all healthcare KPIs"""
        if date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            date_range = (start_date, end_date)
        
        calculations = []
        
        # Calculate each KPI
        for kpi_id, kpi in self.current_kpis.items():
            try:
                if kpi.data_source == "patient_safety_events":
                    result = await self._calculate_patient_safety_score(date_range)
                elif kpi.data_source == "quality_metrics":
                    result = await self._calculate_clinical_quality_index(date_range)
                elif kpi.data_source == "operational_metrics":
                    result = await self._calculate_operational_efficiency(date_range)
                elif kpi.data_source == "financial_data":
                    result = await self._calculate_cost_per_encounter(date_range)
                elif kpi.data_source == "patient_outcomes":
                    if kpi_id == "readmission_rate":
                        result = await self._calculate_readmission_rate(date_range)
                    else:
                        result = await self._calculate_mortality_rate(date_range)
                elif kpi.data_source == "patient_surveys":
                    result = await self._calculate_patient_satisfaction_score(date_range)
                elif kpi.data_source == "appointment_data":
                    result = await self._calculate_average_wait_time(date_range)
                elif kpi.data_source == "bed_management":
                    result = await self._calculate_bed_occupancy_rate(date_range)
                else:
                    result = 0.0
                
                # Update KPI with calculated value
                kpi.current_value = result
                kpi.last_updated = datetime.now()
                kpi.trend = await self._calculate_trend(kpi_id, result)
                
                calculations.append((kpi_id, result))
                
            except Exception as e:
                self.logger.error(f"Failed to calculate KPI {kpi_id}: {str(e)}")
                # Keep previous value or set to 0
                kpi.current_value = 0.0
                calculations.append((kpi_id, 0.0))
        
        self.logger.info(f"Calculated {len(calculations)} KPIs successfully")
        return self.current_kpis
    
    async def _calculate_patient_safety_score(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate patient safety score"""
        # Simulate safety score calculation
        # In production, this would query actual safety event data
        
        start_date, end_date = date_range
        
        # Sample calculation based on safety event rates
        safety_events = await self._get_safety_events(start_date, end_date)
        
        if not safety_events:
            return 95.0
        
        # Calculate weighted safety score
        weights = {
            "medication_errors": 0.3,
            "falls": 0.25,
            "infections": 0.25,
            "other_events": 0.2
        }
        
        score = 100.0
        for event_type, weight in weights.items():
            event_count = safety_events.get(event_type, 0)
            # Deduct points based on event frequency
            score -= min(event_count * weight * 5, 20)  # Max 20 point deduction
        
        return max(score, 0.0)
    
    async def _calculate_clinical_quality_index(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate clinical quality index"""
        # Composite quality score based on multiple quality indicators
        
        quality_components = [
            await self._get_prevention_quality_indicator(date_range),
            await self._get_treatment_effectiveness_score(date_range),
            await self._get_care_coordination_score(date_range),
            await self._get_patient_safety_score(date_range)
        ]
        
        # Weighted average of quality components
        weights = [0.3, 0.3, 0.25, 0.15]
        index_score = sum(score * weight for score, weight in zip(quality_components, weights))
        
        return min(index_score, 100.0)
    
    async def _calculate_operational_efficiency(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate operational efficiency score"""
        # Based on resource utilization and throughput
        
        utilization_metrics = await self._get_resource_utilization_metrics(date_range)
        throughput_metrics = await self._get_throughput_metrics(date_range)
        
        # Efficiency components
        staff_efficiency = utilization_metrics.get("staff_utilization", 0.8)
        equipment_efficiency = utilization_metrics.get("equipment_utilization", 0.85)
        throughput_efficiency = throughput_metrics.get("patient_throughput", 0.9)
        
        # Combined efficiency score
        efficiency_score = (staff_efficiency * 0.4 + equipment_efficiency * 0.3 + throughput_efficiency * 0.3) * 100
        
        return min(efficiency_score, 100.0)
    
    async def _calculate_cost_per_encounter(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate average cost per patient encounter"""
        financial_data = await self._get_financial_data(date_range)
        
        total_cost = financial_data.get("total_costs", 0)
        total_encounters = financial_data.get("total_encounters", 1)
        
        return total_cost / total_encounters if total_encounters > 0 else 0.0
    
    async def _calculate_readmission_rate(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate 30-day readmission rate"""
        outcome_data = await self._get_outcome_data(date_range)
        
        readmissions = outcome_data.get("readmissions_30_day", 0)
        discharges = outcome_data.get("total_discharges", 1)
        
        return readmissions / discharges if discharges > 0 else 0.0
    
    async def _calculate_mortality_rate(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate in-hospital mortality rate"""
        outcome_data = await self._get_outcome_data(date_range)
        
        deaths = outcome_data.get("in_hospital_deaths", 0)
        admissions = outcome_data.get("total_admissions", 1)
        
        return deaths / admissions if admissions > 0 else 0.0
    
    async def _calculate_patient_satisfaction_score(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate patient satisfaction score"""
        survey_data = await self._get_patient_survey_data(date_range)
        
        if not survey_data:
            return 4.0
        
        # Calculate weighted satisfaction score
        satisfaction_scores = []
        weights = {
            "overall_satisfaction": 0.3,
            "care_quality": 0.25,
            "communication": 0.2,
            "facilities": 0.15,
            "billing": 0.1
        }
        
        total_weighted_score = 0.0
        total_weights = 0.0
        
        for category, weight in weights.items():
            category_score = survey_data.get(category, 4.0)
            total_weighted_score += category_score * weight
            total_weights += weight
        
        return total_weighted_score / total_weights if total_weights > 0 else 4.0
    
    async def _calculate_average_wait_time(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate average patient wait time"""
        appointment_data = await self._get_appointment_data(date_range)
        
        wait_times = appointment_data.get("wait_times", [])
        if not wait_times:
            return 15.0  # Default value
        
        return statistics.mean(wait_times)
    
    async def _calculate_bed_occupancy_rate(self, date_range: Tuple[datetime, datetime]) -> float:
        """Calculate bed occupancy rate"""
        occupancy_data = await self._get_bed_occupancy_data(date_range)
        
        occupied_beds = occupancy_data.get("currently_occupied", 0)
        total_beds = occupancy_data.get("total_beds", 100)
        
        return occupied_beds / total_beds if total_beds > 0 else 0.0
    
    # Helper methods for data retrieval (simulated)
    async def _get_safety_events(self, start_date: datetime, end_date: datetime) -> Dict[str, int]:
        """Get patient safety events (simulated data)"""
        return {
            "medication_errors": 5,
            "falls": 3,
            "infections": 2,
            "other_events": 4
        }
    
    async def _get_prevention_quality_indicator(self, date_range: Tuple[datetime, datetime]) -> float:
        """Get prevention quality indicator score"""
        return 88.5
    
    async def _get_treatment_effectiveness_score(self, date_range: Tuple[datetime, datetime]) -> float:
        """Get treatment effectiveness score"""
        return 91.2
    
    async def _get_care_coordination_score(self, date_range: Tuple[datetime, datetime]) -> float:
        """Get care coordination score"""
        return 87.8
    
    async def _get_patient_safety_score(self, date_range: Tuple[datetime, datetime]) -> float:
        """Get patient safety score"""
        return 92.3
    
    async def _get_resource_utilization_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Get resource utilization metrics"""
        return {
            "staff_utilization": 0.82,
            "equipment_utilization": 0.88,
            "room_utilization": 0.85
        }
    
    async def _get_throughput_metrics(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Get throughput metrics"""
        return {
            "patient_throughput": 0.87,
            "discharge_efficiency": 0.91,
            "admission_processing": 0.85
        }
    
    async def _get_financial_data(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Get financial data"""
        return {
            "total_costs": 150000.0,
            "total_encounters": 100
        }
    
    async def _get_outcome_data(self, date_range: Tuple[datetime, datetime]) -> Dict[str, int]:
        """Get patient outcome data"""
        return {
            "readmissions_30_day": 12,
            "total_discharges": 100,
            "in_hospital_deaths": 2,
            "total_admissions": 100
        }
    
    async def _get_patient_survey_data(self, date_range: Tuple[datetime, datetime]) -> Dict[str, float]:
        """Get patient satisfaction survey data"""
        return {
            "overall_satisfaction": 4.3,
            "care_quality": 4.5,
            "communication": 4.2,
            "facilities": 4.0,
            "billing": 3.8
        }
    
    async def _get_appointment_data(self, date_range: Tuple[datetime, datetime]) -> Dict[str, List[float]]:
        """Get appointment and wait time data"""
        return {
            "wait_times": [12, 18, 8, 25, 15, 20, 10, 22, 16, 14]
        }
    
    async def _get_bed_occupancy_data(self, date_range: Tuple[datetime, datetime]) -> Dict[str, int]:
        """Get bed occupancy data"""
        return {
            "currently_occupied": 85,
            "total_beds": 100
        }
    
    async def _calculate_trend(self, kpi_id: str, current_value: float) -> str:
        """Calculate trend direction for KPI"""
        # In production, this would compare with historical data
        # For now, simulate trend based on target comparison
        
        target = self.current_kpis[kpi_id].target_value
        
        if current_value >= target * 1.05:
            return "improving"
        elif current_value <= target * 0.95:
            return "declining"
        else:
            return "stable"
    
    async def generate_dashboard_report(self, dashboard_id: str) -> Dict[str, Any]:
        """Generate comprehensive dashboard report"""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"Dashboard {dashboard_id} not found")
        
        dashboard = self.dashboards[dashboard_id]
        
        # Calculate current KPI values
        await self.calculate_all_kpis()
        
        # Generate report
        report = {
            "dashboard_info": {
                "dashboard_id": dashboard.dashboard_id,
                "title": dashboard.title,
                "generated_at": datetime.now().isoformat(),
                "type": dashboard.dashboard_type.value
            },
            "kpis": {},
            "alerts": [],
            "recommendations": []
        }
        
        for kpi in dashboard.kpis:
            kpi_data = self.current_kpis.get(kpi.kpi_id)
            if kpi_data:
                report["kpis"][kpi.kpi_id] = {
                    "name": kpi_data.name,
                    "current_value": kpi_data.current_value,
                    "target_value": kpi_data.target_value,
                    "unit": kpi_data.unit,
                    "trend": kpi_data.trend,
                    "status": self._get_kpi_status(kpi_data)
                }
        
        # Generate alerts
        report["alerts"] = await self._generate_dashboard_alerts(dashboard)
        
        # Generate recommendations
        report["recommendations"] = await self._generate_recommendations(dashboard)
        
        return report
    
    def _get_kpi_status(self, kpi: KPI) -> str:
        """Determine KPI status based on performance"""
        performance_ratio = kpi.current_value / kpi.target_value if kpi.target_value > 0 else 0
        
        if kpi.category in [MetricCategory.COST, MetricCategory.WAIT_TIME]:
            # For cost and time metrics, lower is better
            if performance_ratio <= 0.9:
                return "excellent"
            elif performance_ratio <= 1.0:
                return "good"
            elif performance_ratio <= 1.2:
                return "fair"
            else:
                return "poor"
        else:
            # For quality metrics, higher is better
            if performance_ratio >= 0.95:
                return "excellent"
            elif performance_ratio >= 0.85:
                return "good"
            elif performance_ratio >= 0.75:
                return "fair"
            else:
                return "poor"
    
    async def _generate_dashboard_alerts(self, dashboard: AnalyticsDashboard) -> List[Dict[str, Any]]:
        """Generate alerts for dashboard KPIs"""
        alerts = []
        
        for kpi in dashboard.kpis:
            kpi_data = self.current_kpis.get(kpi.kpi_id)
            if not kpi_data:
                continue
            
            # Check against threshold values
            critical_threshold = dashboard.alert_config.get("critical_thresholds", {}).get(kpi.kpi_id.split('_')[0])
            if critical_threshold:
                if self._check_critical_threshold(kpi_data, critical_threshold):
                    alerts.append({
                        "type": "critical",
                        "kpi_id": kpi.kpi_id,
                        "message": f"Critical threshold reached for {kpi_data.name}",
                        "current_value": kpi_data.current_value,
                        "threshold": critical_threshold,
                        "timestamp": datetime.now().isoformat()
                    })
        
        return alerts
    
    def _check_critical_threshold(self, kpi: KPI, threshold: float) -> bool:
        """Check if KPI value exceeds critical threshold"""
        if kpi.category in [MetricCategory.COST, MetricCategory.WAIT_TIME]:
            return kpi.current_value > threshold
        else:
            return kpi.current_value < threshold
    
    async def _generate_recommendations(self, dashboard: AnalyticsDashboard) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on KPI performance"""
        recommendations = []
        
        for kpi in dashboard.kpis:
            kpi_data = self.current_kpis.get(kpi.kpi_id)
            if not kpi_data:
                continue
            
            # Generate recommendations based on performance
            performance_ratio = kpi_data.current_value / kpi_data.target_value if kpi_data.target_value > 0 else 0
            
            if kpi_data.category == MetricCategory.SAFETY and performance_ratio < 0.9:
                recommendations.append({
                    "kpi_id": kpi.kpi_id,
                    "priority": "high",
                    "category": "safety_improvement",
                    "action": "Review and enhance safety protocols",
                    "impact": "Improve patient safety scores",
                    "timeline": "2-4 weeks"
                })
            
            elif kpi_data.category == MetricCategory.EFFICIENCY and performance_ratio < 0.85:
                recommendations.append({
                    "kpi_id": kpi.kpi_id,
                    "priority": "medium",
                    "category": "operational_optimization",
                    "action": "Optimize resource allocation and workflows",
                    "impact": "Improve operational efficiency",
                    "timeline": "1-3 months"
                })
            
            elif kpi_data.category == MetricCategory.COST and performance_ratio > 1.1:
                recommendations.append({
                    "kpi_id": kpi.kpi_id,
                    "priority": "high",
                    "category": "cost_reduction",
                    "action": "Implement cost reduction strategies",
                    "impact": "Reduce cost per encounter",
                    "timeline": "1-2 months"
                })
        
        return recommendations
    
    async def export_dashboard_data(self, dashboard_id: str, format: str = "json") -> Dict[str, Any]:
        """Export dashboard data in specified format"""
        dashboard_report = await self.generate_dashboard_report(dashboard_id)
        
        if format.lower() == "json":
            return dashboard_report
        elif format.lower() == "csv":
            # Convert to CSV-friendly format
            csv_data = {
                "kpis": [],
                "timestamp": datetime.now().isoformat()
            }
            
            for kpi_id, kpi_data in dashboard_report["kpis"].items():
                csv_data["kpis"].append(kpi_data)
            
            return csv_data
        elif format.lower() == "excel":
            # Return data structure suitable for Excel export
            return {
                "excel_data": dashboard_report,
                "sheets": {
                    "kpis": "Key Performance Indicators",
                    "alerts": "Current Alerts",
                    "recommendations": "Recommendations"
                }
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")

def create_analytics_engine(config: Dict[str, Any] = None) -> HealthcareAnalyticsEngine:
    """Factory function to create analytics engine"""
    if config is None:
        config = {
            "analytics_warehouse_connection": "sqlite:///analytics_warehouse.db",
            "data_marts": {}
        }
    
    return HealthcareAnalyticsEngine(config)

# Example usage
if __name__ == "__main__":
    async def main():
        engine = create_analytics_engine()
        
        # Initialize analytics
        await engine.initialize_analytics()
        
        # Calculate all KPIs
        kpis = await engine.calculate_all_kpis()
        
        print("Healthcare Analytics KPIs:")
        print("=" * 50)
        for kpi_id, kpi in kpis.items():
            print(f"{kpi.name}: {kpi.current_value:.2f} {kpi.unit} (Target: {kpi.target_value} {kpi.unit})")
            print(f"  Trend: {kpi.trend}")
            print(f"  Status: {engine._get_kpi_status(kpi)}")
            print()
        
        # Generate dashboard report
        report = await engine.generate_dashboard_report("executive_dashboard")
        
        print(f"Dashboard Report: {report['dashboard_info']['title']}")
        print(f"Alerts: {len(report['alerts'])}")
        print(f"Recommendations: {len(report['recommendations'])}")
    
    asyncio.run(main())
