"""
Operational Dashboards and KPI Monitoring System for Healthcare AI
Implements real-time KPI monitoring with comprehensive operational dashboards
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque

class DashboardType(Enum):
    """Types of operational dashboards"""
    EXECUTIVE = "executive"
    CLINICAL_PERFORMANCE = "clinical_performance"
    SYSTEM_HEALTH = "system_health"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    OPERATIONS = "operations"
    CUSTOMER_SUCCESS = "customer_success"

class KPILevel(Enum):
    """KPI performance levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics"""
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    TIMESERIES = "timeseries"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"

class AlertThreshold(Enum):
    """Alert threshold types"""
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class KPI:
    """Key Performance Indicator definition"""
    kpi_id: str
    name: str
    description: str
    metric_type: MetricType
    target_value: float
    current_value: float
    unit: str
    category: str
    weight: float  # Relative importance (0-1)
    threshold_warning: float
    threshold_critical: float
    trend_direction: str  # up, down, stable
    last_updated: datetime
    historical_data: deque = field(default_factory=lambda: deque(maxlen=100))

@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    dashboard_type: DashboardType
    description: str
    kpis: List[str]  # KPI IDs
    layout: Dict[str, Any]
    refresh_interval: int  # seconds
    target_audience: List[str]
    permissions: Dict[str, List[str]]

@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    kpi_id: str
    severity: AlertThreshold
    message: str
    threshold_value: float
    current_value: float
    triggered_at: datetime
    status: str  # active, acknowledged, resolved
    assigned_to: str

@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    kpi_id: str
    excellent_threshold: float
    good_threshold: float
    acceptable_threshold: float
    poor_threshold: float
    auto_escalation: bool
    escalation_delays: Dict[str, int]  # minutes

class OperationalDashboardManager:
    """Operational Dashboard and KPI Monitoring Manager"""
    
    def __init__(self):
        self.kpis: Dict[str, KPI] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self.alerts: List[Alert] = []
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.dashboard_data: Dict[str, Dict] = {}
        
    async def create_executive_dashboard(self, config: Dict) -> Dashboard:
        """Create executive dashboard with strategic KPIs"""
        
        # Executive KPIs
        kpi_configs = [
            {
                "kpi_id": "EXEC_REVENUE",
                "name": "Monthly Revenue",
                "description": "Total monthly revenue from healthcare AI services",
                "metric_type": MetricType.CURRENCY,
                "target_value": 500000.0,
                "current_value": 485000.0,
                "unit": "USD",
                "category": "financial",
                "weight": 0.25,
                "threshold_warning": 450000.0,
                "threshold_critical": 400000.0
            },
            {
                "kpi_id": "EXEC_PATIENT_SATISFACTION",
                "name": "Patient Satisfaction Score",
                "description": "Overall patient satisfaction with AI-assisted care",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 95.0,
                "current_value": 93.2,
                "unit": "%",
                "category": "customer_success",
                "weight": 0.20,
                "threshold_warning": 90.0,
                "threshold_critical": 85.0
            },
            {
                "kpi_id": "EXEC_CLINICAL_ACCURACY",
                "name": "AI Clinical Accuracy",
                "description": "AI model accuracy in clinical decision support",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 99.0,
                "current_value": 97.8,
                "unit": "%",
                "category": "clinical_performance",
                "weight": 0.25,
                "threshold_warning": 96.0,
                "threshold_critical": 94.0
            },
            {
                "kpi_id": "EXEC_SYSTEM_UPTIME",
                "name": "System Uptime",
                "description": "Overall system availability percentage",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 99.9,
                "current_value": 99.7,
                "unit": "%",
                "category": "system_health",
                "weight": 0.15,
                "threshold_warning": 99.5,
                "threshold_critical": 99.0
            },
            {
                "kpi_id": "EXEC_COST_EFFICIENCY",
                "name": "Cost per Clinical Case",
                "description": "Average cost per clinical case processed",
                "metric_type": MetricType.CURRENCY,
                "target_value": 25.0,
                "current_value": 28.5,
                "unit": "USD",
                "category": "financial",
                "weight": 0.15,
                "threshold_warning": 30.0,
                "threshold_critical": 35.0
            }
        ]
        
        # Create KPIs
        for kpi_config in kpi_configs:
            kpi = KPI(
                kpi_id=kpi_config["kpi_id"],
                name=kpi_config["name"],
                description=kpi_config["description"],
                metric_type=MetricType(kpi_config["metric_type"]),
                target_value=kpi_config["target_value"],
                current_value=kpi_config["current_value"],
                unit=kpi_config["unit"],
                category=kpi_config["category"],
                weight=kpi_config["weight"],
                threshold_warning=kpi_config["threshold_warning"],
                threshold_critical=kpi_config["threshold_critical"],
                trend_direction="stable",
                last_updated=datetime.now()
            )
            self.kpis[kpi.kpi_id] = kpi
        
        # Create dashboard
        dashboard = Dashboard(
            dashboard_id=config["dashboard_id"],
            name=config["dashboard_name"],
            dashboard_type=DashboardType.EXECUTIVE,
            description="Executive dashboard for strategic healthcare AI KPIs",
            kpis=[kpi["kpi_id"] for kpi in kpi_configs],
            layout={
                "sections": ["financial", "clinical_performance", "system_health", "customer_success"],
                "chart_types": ["gauge", "line", "bar", "pie"],
                "widgets": 5
            },
            refresh_interval=300,  # 5 minutes
            target_audience=["CEO", "CMO", "CTO", "CFO"],
            permissions={"view": ["executive_team"], "admin": ["CIO"]}
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def create_clinical_performance_dashboard(self, config: Dict) -> Dashboard:
        """Create clinical performance dashboard"""
        
        # Clinical performance KPIs
        kpi_configs = [
            {
                "kpi_id": "CLIN_AI_ACCURACY",
                "name": "AI Diagnostic Accuracy",
                "description": "AI model accuracy in diagnostic recommendations",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 98.5,
                "current_value": 97.2,
                "unit": "%",
                "category": "accuracy",
                "weight": 0.30,
                "threshold_warning": 96.0,
                "threshold_critical": 94.0
            },
            {
                "kpi_id": "CLIN_RESPONSE_TIME",
                "name": "Clinical Response Time",
                "description": "Average time for AI response to clinical queries",
                "metric_type": MetricType.GAUGE,
                "target_value": 2.0,
                "current_value": 2.8,
                "unit": "seconds",
                "category": "performance",
                "weight": 0.25,
                "threshold_warning": 3.5,
                "threshold_critical": 5.0
            },
            {
                "kpi_id": "CLIN_SAFETY_SCORE",
                "name": "Patient Safety Score",
                "description": "Composite score of patient safety indicators",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 99.5,
                "current_value": 99.1,
                "unit": "%",
                "category": "safety",
                "weight": 0.30,
                "threshold_warning": 98.5,
                "threshold_critical": 97.0
            },
            {
                "kpi_id": "CLIN_ADOPTION_RATE",
                "name": "Clinical Adoption Rate",
                "description": "Percentage of clinicians actively using AI tools",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 85.0,
                "current_value": 78.5,
                "unit": "%",
                "category": "adoption",
                "weight": 0.15,
                "threshold_warning": 75.0,
                "threshold_critical": 70.0
            }
        ]
        
        # Create KPIs
        for kpi_config in kpi_configs:
            kpi = KPI(
                kpi_id=kpi_config["kpi_id"],
                name=kpi_config["name"],
                description=kpi_config["description"],
                metric_type=MetricType(kpi_config["metric_type"]),
                target_value=kpi_config["target_value"],
                current_value=kpi_config["current_value"],
                unit=kpi_config["unit"],
                category=kpi_config["category"],
                weight=kpi_config["weight"],
                threshold_warning=kpi_config["threshold_warning"],
                threshold_critical=kpi_config["threshold_critical"],
                trend_direction="stable",
                last_updated=datetime.now()
            )
            self.kpis[kpi.kpi_id] = kpi
        
        dashboard = Dashboard(
            dashboard_id=config["dashboard_id"],
            name=config["dashboard_name"],
            dashboard_type=DashboardType.CLINICAL_PERFORMANCE,
            description="Clinical performance monitoring dashboard",
            kpis=[kpi["kpi_id"] for kpi in kpi_configs],
            layout={
                "sections": ["accuracy", "performance", "safety", "adoption"],
                "chart_types": ["gauge", "timeseries", "heatmap"],
                "widgets": 4
            },
            refresh_interval=60,  # 1 minute
            target_audience=["Chief Medical Officer", "Clinical Directors", "Quality Assurance"],
            permissions={"view": ["clinical_team", "quality_team"], "admin": ["CMO"]}
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def create_system_health_dashboard(self, config: Dict) -> Dashboard:
        """Create system health monitoring dashboard"""
        
        # System health KPIs
        kpi_configs = [
            {
                "kpi_id": "SYS_UPTIME",
                "name": "System Uptime",
                "description": "Overall system availability",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 99.95,
                "current_value": 99.87,
                "unit": "%",
                "category": "availability",
                "weight": 0.25,
                "threshold_warning": 99.9,
                "threshold_critical": 99.5
            },
            {
                "kpi_id": "SYS_RESPONSE_TIME",
                "name": "Average Response Time",
                "description": "System average response time",
                "metric_type": MetricType.GAUGE,
                "target_value": 200.0,
                "current_value": 285.0,
                "unit": "ms",
                "category": "performance",
                "weight": 0.20,
                "threshold_warning": 350.0,
                "threshold_critical": 500.0
            },
            {
                "kpi_id": "SYS_CPU_UTILIZATION",
                "name": "CPU Utilization",
                "description": "Average CPU utilization across all servers",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 65.0,
                "current_value": 72.5,
                "unit": "%",
                "category": "resources",
                "weight": 0.15,
                "threshold_warning": 80.0,
                "threshold_critical": 90.0
            },
            {
                "kpi_id": "SYS_MEMORY_UTILIZATION",
                "name": "Memory Utilization",
                "description": "Average memory utilization",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 70.0,
                "current_value": 68.2,
                "unit": "%",
                "category": "resources",
                "weight": 0.15,
                "threshold_warning": 85.0,
                "threshold_critical": 95.0
            },
            {
                "kpi_id": "SYS_ERROR_RATE",
                "name": "Error Rate",
                "description": "System error rate percentage",
                "metric_type": MetricType.PERCENTAGE,
                "target_value": 0.1,
                "current_value": 0.15,
                "unit": "%",
                "category": "quality",
                "weight": 0.25,
                "threshold_warning": 0.3,
                "threshold_critical": 0.5
            }
        ]
        
        # Create KPIs
        for kpi_config in kpi_configs:
            kpi = KPI(
                kpi_id=kpi_config["kpi_id"],
                name=kpi_config["name"],
                description=kpi_config["description"],
                metric_type=MetricType(kpi_config["metric_type"]),
                target_value=kpi_config["target_value"],
                current_value=kpi_config["current_value"],
                unit=kpi_config["unit"],
                category=kpi_config["category"],
                weight=kpi_config["weight"],
                threshold_warning=kpi_config["threshold_warning"],
                threshold_critical=kpi_config["threshold_critical"],
                trend_direction="stable",
                last_updated=datetime.now()
            )
            self.kpis[kpi.kpi_id] = kpi
        
        dashboard = Dashboard(
            dashboard_id=config["dashboard_id"],
            name=config["dashboard_name"],
            dashboard_type=DashboardType.SYSTEM_HEALTH,
            description="Real-time system health monitoring dashboard",
            kpis=[kpi["kpi_id"] for kpi in kpi_configs],
            layout={
                "sections": ["availability", "performance", "resources", "quality"],
                "chart_types": ["gauge", "timeseries", "status_indicators"],
                "widgets": 5
            },
            refresh_interval=30,  # 30 seconds
            target_audience=["DevOps Team", "System Administrators", "IT Operations"],
            permissions={"view": ["operations_team"], "admin": ["CTO"]}
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def update_kpi_values(self) -> Dict:
        """Update all KPI values with current data"""
        
        updates = {}
        
        # Simulate KPI updates
        kpi_updates = {
            "EXEC_REVENUE": {"value": 485000.0 + (hash("revenue") % 10000 - 5000), "trend": "stable"},
            "EXEC_PATIENT_SATISFACTION": {"value": 93.2 + (hash("satisfaction") % 20 - 10) / 10, "trend": "up"},
            "EXEC_CLINICAL_ACCURACY": {"value": 97.8 + (hash("accuracy") % 10 - 5) / 10, "trend": "up"},
            "EXEC_SYSTEM_UPTIME": {"value": 99.7 + (hash("uptime") % 5 - 2.5) / 10, "trend": "stable"},
            "EXEC_COST_EFFICIENCY": {"value": 28.5 + (hash("cost") % 20 - 10), "trend": "down"},
            "CLIN_AI_ACCURACY": {"value": 97.2 + (hash("clin_accuracy") % 10 - 5) / 10, "trend": "up"},
            "CLIN_RESPONSE_TIME": {"value": 2.8 + (hash("response") % 20 - 10) / 10, "trend": "down"},
            "CLIN_SAFETY_SCORE": {"value": 99.1 + (hash("safety") % 5 - 2.5) / 10, "trend": "stable"},
            "CLIN_ADOPTION_RATE": {"value": 78.5 + (hash("adoption") % 30 - 15) / 10, "trend": "up"},
            "SYS_UPTIME": {"value": 99.87 + (hash("sys_uptime") % 3 - 1.5) / 10, "trend": "stable"},
            "SYS_RESPONSE_TIME": {"value": 285.0 + (hash("sys_response") % 100 - 50), "trend": "down"},
            "SYS_CPU_UTILIZATION": {"value": 72.5 + (hash("cpu") % 40 - 20) / 10, "trend": "stable"},
            "SYS_MEMORY_UTILIZATION": {"value": 68.2 + (hash("memory") % 30 - 15) / 10, "trend": "up"},
            "SYS_ERROR_RATE": {"value": 0.15 + (hash("error_rate") % 10 - 5) / 100, "trend": "stable"}
        }
        
        for kpi_id, update in kpi_updates.items():
            if kpi_id in self.kpis:
                kpi = self.kpis[kpi_id]
                # Update historical data
                kpi.historical_data.append({
                    "timestamp": datetime.now().isoformat(),
                    "value": kpi.current_value
                })
                
                # Update current value
                kpi.current_value = update["value"]
                kpi.trend_direction = update["trend"]
                kpi.last_updated = datetime.now()
                
                updates[kpi_id] = {
                    "previous_value": kpi.historical_data[-1]["value"],
                    "current_value": kpi.current_value,
                    "change": kpi.current_value - kpi.historical_data[-1]["value"],
                    "trend": kpi.trend_direction
                }
        
        return updates
    
    async def generate_alerts(self) -> List[Alert]:
        """Generate alerts based on KPI thresholds"""
        
        alerts = []
        
        for kpi_id, kpi in self.kpis.items():
            # Check threshold violations
            if kpi.metric_type == MetricType.PERCENTAGE:
                # Higher is better
                if kpi.current_value <= kpi.threshold_critical:
                    severity = AlertThreshold.CRITICAL
                elif kpi.current_value <= kpi.threshold_warning:
                    severity = AlertThreshold.WARNING
                else:
                    severity = None
            else:
                # Lower is better for response times, error rates, etc.
                if kpi.current_value >= kpi.threshold_critical:
                    severity = AlertThreshold.CRITICAL
                elif kpi.current_value >= kpi.threshold_warning:
                    severity = AlertThreshold.WARNING
                else:
                    severity = None
            
            # Generate alert if threshold breached
            if severity:
                alert = Alert(
                    alert_id=f"alert_{kpi_id}_{int(time.time())}",
                    kpi_id=kpi_id,
                    severity=severity,
                    message=f"{kpi.name} is {severity.value}: {kpi.current_value} {kpi.unit} (target: {kpi.target_value} {kpi.unit})",
                    threshold_value=kpi.threshold_warning if severity == AlertThreshold.WARNING else kpi.threshold_critical,
                    current_value=kpi.current_value,
                    triggered_at=datetime.now(),
                    status="active",
                    assigned_to="Operations Team"
                )
                alerts.append(alert)
                self.alerts.append(alert)
        
        return alerts
    
    async def calculate_dashboard_score(self, dashboard_id: str) -> Dict:
        """Calculate overall dashboard score based on KPI performance"""
        
        dashboard = self.dashboards[dashboard_id]
        dashboard_kpis = [self.kpis[kpi_id] for kpi_id in dashboard.kpis if kpi_id in self.kpis]
        
        if not dashboard_kpis:
            return {"error": "No KPIs found for dashboard"}
        
        # Calculate weighted scores
        total_weight = sum(kpi.weight for kpi in dashboard_kpis)
        weighted_score = 0.0
        
        kpi_scores = {}
        for kpi in dashboard_kpis:
            # Calculate individual KPI score (0-100)
            if kpi.metric_type == MetricType.PERCENTAGE or "accuracy" in kpi.name.lower() or "uptime" in kpi.name.lower():
                # Higher is better
                score = (kpi.current_value / kpi.target_value) * 100
            else:
                # Lower is better
                score = min(100, (kpi.target_value / max(kpi.current_value, 0.01)) * 100)
            
            # Normalize score to 0-100 range
            score = min(100, max(0, score))
            
            kpi_scores[kpi.kpi_id] = {
                "score": round(score, 1),
                "current_value": kpi.current_value,
                "target_value": kpi.target_value,
                "unit": kpi.unit,
                "weight": kpi.weight,
                "contribution": score * kpi.weight
            }
            
            weighted_score += score * kpi.weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # Determine performance level
        if overall_score >= 95:
            performance_level = KPILevel.EXCELLENT
        elif overall_score >= 85:
            performance_level = KPILevel.GOOD
        elif overall_score >= 75:
            performance_level = KPILevel.ACCEPTABLE
        elif overall_score >= 60:
            performance_level = KPILevel.POOR
        else:
            performance_level = KPILevel.CRITICAL
        
        return {
            "dashboard_id": dashboard_id,
            "overall_score": round(overall_score, 1),
            "performance_level": performance_level.value,
            "kpi_count": len(dashboard_kpis),
            "kpi_scores": kpi_scores,
            "top_performers": sorted(kpi_scores.items(), key=lambda x: x[1]["score"], reverse=True)[:3],
            "bottom_performers": sorted(kpi_scores.items(), key=lambda x: x[1]["score"])[:3],
            "trend_analysis": {
                "improving_kpis": len([kpi for kpi in dashboard_kpis if kpi.trend_direction == "up"]),
                "stable_kpis": len([kpi for kpi in dashboard_kpis if kpi.trend_direction == "stable"]),
                "declining_kpis": len([kpi for kpi in dashboard_kpis if kpi.trend_direction == "down"])
            }
        }
    
    async def generate_real_time_dashboard_data(self, dashboard_id: str) -> Dict:
        """Generate real-time dashboard data"""
        
        dashboard = self.dashboards[dashboard_id]
        dashboard_kpis = [self.kpis[kpi_id] for kpi_id in dashboard.kpis if kpi_id in self.kpis]
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.alerts if 
                        alert.triggered_at > datetime.now() - timedelta(hours=24)]
        
        # Calculate dashboard score
        score_analysis = await self.calculate_dashboard_score(dashboard_id)
        
        # Generate timeseries data
        timeseries_data = {}
        for kpi in dashboard_kpis:
            if len(kpi.historical_data) > 0:
                timeseries_data[kpi.kpi_id] = {
                    "name": kpi.name,
                    "current_value": kpi.current_value,
                    "target": kpi.target_value,
                    "unit": kpi.unit,
                    "trend": kpi.trend_direction,
                    "performance": kpi_scores.get(kpi.kpi_id, {}).get("score", 0) if 'kpi_scores' in locals() else 0,
                    "data_points": list(kpi.historical_data)[-10:]  # Last 10 data points
                }
        
        return {
            "dashboard_info": {
                "dashboard_id": dashboard.dashboard_id,
                "name": dashboard.name,
                "type": dashboard.dashboard_type.value,
                "last_updated": datetime.now().isoformat(),
                "refresh_interval": dashboard.refresh_interval
            },
            "overall_performance": {
                "score": score_analysis.get("overall_score", 0),
                "level": score_analysis.get("performance_level", "unknown"),
                "trend": "stable"
            },
            "kpi_summary": {
                "total_kpis": len(dashboard_kpis),
                "excellent": len([kpi for kpi in dashboard_kpis if kpi.current_value >= kpi.target_value * 0.95]),
                "good": len([kpi for kpi in dashboard_kpis if kpi.target_value * 0.85 <= kpi.current_value < kpi.target_value * 0.95]),
                "needs_attention": len([kpi for kpi in dashboard_kpis if kpi.current_value < kpi.target_value * 0.85])
            },
            "kpi_details": timeseries_data,
            "alerts": {
                "total_active": len([a for a in recent_alerts if a.status == "active"]),
                "critical": len([a for a in recent_alerts if a.severity == AlertThreshold.CRITICAL]),
                "warning": len([a for a in recent_alerts if a.severity == AlertThreshold.WARNING]),
                "recent_alerts": [
                    {
                        "id": alert.alert_id,
                        "kpi": self.kpis.get(alert.kpi_id, {}).name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "time": alert.triggered_at.strftime("%H:%M:%S")
                    }
                    for alert in recent_alerts[:5]  # Last 5 alerts
                ]
            },
            "insights": [
                "Overall system performance is stable",
                "Clinical accuracy trending positively",
                "Cost efficiency requires attention",
                "System uptime meets SLA targets"
            ]
        }
    
    async def create_custom_dashboard(self, dashboard_config: Dict) -> Dashboard:
        """Create custom dashboard with specific KPIs"""
        
        # Use provided KPI IDs or create default ones
        kpi_ids = dashboard_config.get("kpi_ids", [])
        
        dashboard = Dashboard(
            dashboard_id=dashboard_config["dashboard_id"],
            name=dashboard_config["dashboard_name"],
            dashboard_type=DashboardType(dashboard_config.get("dashboard_type", "operations")),
            description=dashboard_config.get("description", "Custom operational dashboard"),
            kpis=kpi_ids,
            layout=dashboard_config.get("layout", {"sections": [], "chart_types": [], "widgets": 0}),
            refresh_interval=dashboard_config.get("refresh_interval", 300),
            target_audience=dashboard_config.get("target_audience", []),
            permissions=dashboard_config.get("permissions", {"view": [], "admin": []})
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard
    
    async def export_dashboard_configuration(self, filepath: str) -> Dict:
        """Export dashboard configurations"""
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "dashboards": [
                {
                    "dashboard_id": d.dashboard_id,
                    "name": d.name,
                    "type": d.dashboard_type.value,
                    "description": d.description,
                    "kpis": d.kpis,
                    "layout": d.layout,
                    "refresh_interval": d.refresh_interval,
                    "target_audience": d.target_audience,
                    "permissions": d.permissions
                }
                for d in self.dashboards.values()
            ],
            "kpis": [
                {
                    "kpi_id": kpi.kpi_id,
                    "name": kpi.name,
                    "description": kpi.description,
                    "metric_type": kpi.metric_type.value,
                    "target_value": kpi.target_value,
                    "current_value": kpi.current_value,
                    "unit": kpi.unit,
                    "category": kpi.category,
                    "weight": kpi.weight,
                    "threshold_warning": kpi.threshold_warning,
                    "threshold_critical": kpi.threshold_critical,
                    "trend_direction": kpi.trend_direction,
                    "last_updated": kpi.last_updated.isoformat()
                }
                for kpi in self.kpis.values()
            ],
            "alerts": [
                {
                    "alert_id": alert.alert_id,
                    "kpi_id": alert.kpi_id,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "status": alert.status,
                    "triggered_at": alert.triggered_at.isoformat()
                }
                for alert in self.alerts[-50:]  # Last 50 alerts
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return {"status": "success", "export_file": filepath}
    
    async def generate_operational_dashboard_summary(self) -> Dict:
        """Generate comprehensive operational dashboard summary"""
        
        summary = {
            "summary_timestamp": datetime.now().isoformat(),
            "dashboard_overview": {
                "total_dashboards": len(self.dashboards),
                "total_kpis": len(self.kpis),
                "total_alerts": len(self.alerts),
                "active_alerts": len([a for a in self.alerts if a.status == "active"])
            },
            "dashboard_types": {
                dashboard_type.value: len([d for d in self.dashboards.values() if d.dashboard_type == dashboard_type])
                for dashboard_type in DashboardType
            },
            "kpi_categories": {
                category: len([kpi for kpi in self.kpis.values() if kpi.category == category])
                for category in set(kpi.category for kpi in self.kpis.values())
            },
            "performance_summary": {
                "excellent_performance": len([kpi for kpi in self.kpis.values() if kpi.current_value >= kpi.target_value * 0.95]),
                "good_performance": len([kpi for kpi in self.kpis.values() if kpi.target_value * 0.85 <= kpi.current_value < kpi.target_value * 0.95]),
                "needs_improvement": len([kpi for kpi in self.kpis.values() if kpi.current_value < kpi.target_value * 0.85])
            },
            "alert_summary": {
                "critical_alerts": len([a for a in self.alerts if a.severity == AlertThreshold.CRITICAL]),
                "warning_alerts": len([a for a in self.alerts if a.severity == AlertThreshold.WARNING]),
                "resolved_alerts": len([a for a in self.alerts if a.status == "resolved"])
            },
            "recommendations": [
                "Maintain high standards for clinical accuracy KPIs",
                "Address system performance bottlenecks",
                "Implement proactive alerting for critical thresholds",
                "Regular dashboard review and optimization",
                "Expand KPI coverage for emerging business needs"
            ]
        }
        
        return summary

# Example usage and testing
async def run_dashboard_demo():
    """Demonstrate Operational Dashboard framework"""
    dashboard_manager = OperationalDashboardManager()
    
    # 1. Create Executive Dashboard
    print("=== Creating Executive Dashboard ===")
    exec_config = {
        "dashboard_id": "EXECUTIVE_DASHBOARD_001",
        "dashboard_name": "Executive Healthcare AI Dashboard"
    }
    exec_dashboard = await dashboard_manager.create_executive_dashboard(exec_config)
    print(f"Dashboard: {exec_dashboard.name}")
    print(f"Type: {exec_dashboard.dashboard_type.value}")
    print(f"Target Audience: {', '.join(exec_dashboard.target_audience)}")
    print(f"KPIs: {len(exec_dashboard.kpis)}")
    
    # 2. Create Clinical Performance Dashboard
    print("\n=== Creating Clinical Performance Dashboard ===")
    clinical_config = {
        "dashboard_id": "CLINICAL_DASHBOARD_001",
        "dashboard_name": "Clinical Performance Dashboard"
    }
    clinical_dashboard = await dashboard_manager.create_clinical_performance_dashboard(clinical_config)
    print(f"Dashboard: {clinical_dashboard.name}")
    print(f"KPIs: {len(clinical_dashboard.kpis)}")
    print(f"Refresh Interval: {clinical_dashboard.refresh_interval} seconds")
    
    # 3. Create System Health Dashboard
    print("\n=== Creating System Health Dashboard ===")
    system_config = {
        "dashboard_id": "SYSTEM_DASHBOARD_001",
        "dashboard_name": "System Health Dashboard"
    }
    system_dashboard = await dashboard_manager.create_system_health_dashboard(system_config)
    print(f"Dashboard: {system_dashboard.name}")
    print(f"KPIs: {len(system_dashboard.kpis)}")
    print(f"Target Audience: {', '.join(system_dashboard.target_audience)}")
    
    # 4. Update KPI Values
    print("\n=== Updating KPI Values ===")
    updates = await dashboard_manager.update_kpi_values()
    print(f"KPIs Updated: {len(updates)}")
    
    # Show a few KPI updates
    for kpi_id, update_data in list(updates.items())[:3]:
        kpi = dashboard_manager.kpis[kpi_id]
        print(f"{kpi.name}: {update_data['current_value']:.1f} {kpi.unit} (trend: {update_data['trend']})")
    
    # 5. Generate Alerts
    print("\n=== Generating Alerts ===")
    alerts = await dashboard_manager.generate_alerts()
    print(f"Alerts Generated: {len(alerts)}")
    
    # Show critical alerts
    critical_alerts = [a for a in alerts if a.severity == AlertThreshold.CRITICAL]
    warning_alerts = [a for a in alerts if a.severity == AlertThreshold.WARNING]
    
    if critical_alerts:
        print(f"Critical Alerts: {len(critical_alerts)}")
        for alert in critical_alerts[:2]:
            print(f"  - {dashboard_manager.kpis[alert.kpi_id].name}: {alert.message}")
    
    if warning_alerts:
        print(f"Warning Alerts: {len(warning_alerts)}")
    
    # 6. Calculate Dashboard Scores
    print("\n=== Dashboard Performance Scores ===")
    for dashboard_id in [exec_dashboard.dashboard_id, clinical_dashboard.dashboard_id, system_dashboard.dashboard_id]:
        score_analysis = await dashboard_manager.calculate_dashboard_score(dashboard_id)
        dashboard = dashboard_manager.dashboards[dashboard_id]
        print(f"{dashboard.name}:")
        print(f"  Overall Score: {score_analysis['overall_score']}% ({score_analysis['performance_level']})")
        print(f"  KPIs: {score_analysis['kpi_count']}")
        print(f"  Top Performers: {len(score_analysis['top_performers'])}")
        print(f"  Needs Attention: {len(score_analysis['bottom_performers'])}")
    
    # 7. Generate Real-time Dashboard Data
    print("\n=== Real-time Dashboard Data ===")
    for dashboard_id in [exec_dashboard.dashboard_id]:
        real_time_data = await dashboard_manager.generate_real_time_dashboard_data(dashboard_id)
        dashboard = dashboard_manager.dashboards[dashboard_id]
        print(f"\n{dashboard.name} Real-time Data:")
        print(f"  Overall Score: {real_time_data['overall_performance']['score']}%")
        print(f"  Performance Level: {real_time_data['overall_performance']['level']}")
        print(f"  KPI Summary: {real_time_data['kpi_summary']}")
        print(f"  Active Alerts: {real_time_data['alerts']['total_active']}")
    
    # 8. Create Custom Dashboard
    print("\n=== Creating Custom Dashboard ===")
    custom_config = {
        "dashboard_id": "CUSTOM_DASHBOARD_001",
        "dashboard_name": "Custom Operations Dashboard",
        "dashboard_type": "operations",
        "description": "Custom dashboard for operations team",
        "kpi_ids": ["SYS_UPTIME", "SYS_RESPONSE_TIME", "SYS_CPU_UTILIZATION"],
        "refresh_interval": 60,
        "target_audience": ["Operations Team", "DevOps"],
        "layout": {"sections": ["performance", "resources"], "chart_types": ["gauge", "timeseries"]}
    }
    custom_dashboard = await dashboard_manager.create_custom_dashboard(custom_config)
    print(f"Custom Dashboard: {custom_dashboard.name}")
    print(f"KPIs: {len(custom_dashboard.kpis)}")
    print(f"Target Audience: {', '.join(custom_dashboard.target_audience)}")
    
    # 9. Export Dashboard Configuration
    print("\n=== Exporting Dashboard Configuration ===")
    export_result = await dashboard_manager.export_dashboard_configuration("dashboard_configuration.json")
    print(f"Configuration exported to: {export_result['export_file']}")
    
    # 10. Generate Summary
    print("\n=== Dashboard Summary ===")
    summary = await dashboard_manager.generate_operational_dashboard_summary()
    print(f"Total Dashboards: {summary['dashboard_overview']['total_dashboards']}")
    print(f"Total KPIs: {summary['dashboard_overview']['total_kpis']}")
    print(f"Active Alerts: {summary['dashboard_overview']['active_alerts']}")
    print(f"Excellent Performance: {summary['performance_summary']['excellent_performance']} KPIs")
    print(f"Needs Improvement: {summary['performance_summary']['needs_improvement']} KPIs")
    
    return dashboard_manager

if __name__ == "__main__":
    asyncio.run(run_dashboard_demo())
