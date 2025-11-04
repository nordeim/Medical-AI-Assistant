"""
KPI Models for Executive Dashboard and Performance Tracking
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal
from enum import Enum

class KPIType(Enum):
    """Types of KPIs"""
    REVENUE = "revenue"
    CUSTOMER = "customer"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    MARKETING = "marketing"
    SALES = "sales"
    PRODUCT = "product"
    EXECUTIVE = "executive"

class KPIStatus(Enum):
    """KPI performance status"""
    EXCELLENT = "excellent"      # Green
    GOOD = "good"                # Light Green
    TARGET = "target"            # Yellow
    BELOW_TARGET = "below_target" # Orange
    CRITICAL = "critical"        # Red

@dataclass
class KPIMetrics:
    """Key Performance Indicator metrics"""
    kpi_id: str
    kpi_name: str
    kpi_type: KPIType
    description: str
    unit_of_measure: str
    target_value: Decimal
    current_value: Decimal
    previous_value: Optional[Decimal]
    
    # Performance tracking
    period_start: date
    period_end: date
    performance_status: KPIStatus
    variance_from_target: float  # % variance
    variance_from_previous: float  # % change
    
    # Thresholds
    excellent_threshold: Decimal
    good_threshold: Decimal
    target_threshold: Decimal
    below_target_threshold: Decimal
    
    # Trend analysis
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_consistency: float  # 0-1 scale
    months_of_data: int
    
    # Benchmarking
    industry_benchmark: Optional[Decimal]
    benchmark_percentile: Optional[float]
    
    # Alerting
    alert_enabled: bool
    alert_thresholds: Dict[str, Decimal]
    last_alert_date: Optional[date]
    
    def get_status_color(self) -> str:
        """Get status color for dashboard"""
        status_colors = {
            KPIStatus.EXCELLENT: "#10B981",  # Green
            KPIStatus.GOOD: "#34D399",       # Light Green
            KPIStatus.TARGET: "#F59E0B",     # Yellow
            KPIStatus.BELOW_TARGET: "#FB923C", # Orange
            KPIStatus.CRITICAL: "#EF4444"    # Red
        }
        return status_colors.get(self.performance_status, "#6B7280")
    
    def calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        # Base score on target achievement
        if self.target_value > 0:
            target_achievement = float(self.current_value / self.target_value)
        else:
            target_achievement = 1.0
        
        # Adjust for trend (10% weight)
        trend_adjustment = 0
        if self.trend_direction == 'improving':
            trend_adjustment = 10
        elif self.trend_direction == 'declining':
            trend_adjustment = -10
        
        # Base score (90% weight)
        base_score = min(target_achievement * 90, 90)
        
        # Apply trend adjustment
        final_score = base_score + trend_adjustment
        return max(0, min(final_score, 100))
    
    def get_action_required(self) -> List[Dict[str, Any]]:
        """Get required actions based on performance"""
        actions = []
        
        if self.performance_status == KPIStatus.CRITICAL:
            actions.append({
                'priority': 'Critical',
                'action': 'Immediate attention required',
                'description': f'{self.kpi_name} is significantly below target',
                'owner': 'Executive Team'
            })
        elif self.performance_status == KPIStatus.BELOW_TARGET:
            actions.append({
                'priority': 'High',
                'action': 'Performance improvement needed',
                'description': f'{self.kpi_name} is below target threshold',
                'owner': 'Department Head'
            })
        elif self.trend_direction == 'declining':
            actions.append({
                'priority': 'Medium',
                'action': 'Trend monitoring',
                'description': f'{self.kpi_name} showing declining trend',
                'owner': 'KPI Owner'
            })
        
        return actions

@dataclass
class PerformanceDashboard:
    """Performance dashboard configuration"""
    dashboard_id: str
    dashboard_name: str
    dashboard_type: str  # 'executive', 'operational', 'financial'
    target_audience: str
    refresh_frequency: str  # 'real_time', 'hourly', 'daily', 'weekly'
    
    # Dashboard layout
    layout_config: Dict[str, Any]
    widgets: List[Dict[str, Any]]
    
    # KPI assignments
    primary_kpis: List[str]
    secondary_kpis: List[str]
    supporting_metrics: List[str]
    
    # Visualizations
    charts: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    gauges: List[Dict[str, Any]]
    
    # Filters and controls
    available_filters: List[str]
    default_filters: Dict[str, Any]
    
    # Sharing and access
    shared_with: List[str]
    access_level: str  # 'public', 'restricted', 'confidential'
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get dashboard summary metrics"""
        summary = {
            'total_kpis': len(self.primary_kpis) + len(self.secondary_kpis),
            'widgets_count': len(self.widgets),
            'charts_count': len(self.charts),
            'last_updated': datetime.now().isoformat()
        }
        
        return summary

@dataclass
class ExecutiveMetrics:
    """Executive-level summary metrics"""
    report_date: date
    report_period: str  # 'monthly', 'quarterly', 'annual'
    
    # Financial metrics
    total_revenue: Decimal
    revenue_growth: float
    gross_margin: float
    operating_margin: float
    net_margin: float
    cash_flow: Decimal
    
    # Customer metrics
    total_customers: int
    new_customers: int
    churn_rate: float
    net_revenue_retention: float
    customer_lifetime_value: Decimal
    average_revenue_per_user: Decimal
    
    # Sales metrics
    sales_pipeline_value: Decimal
    win_rate: float
    average_deal_size: Decimal
    sales_cycle_length: int
    quota_attainment: float
    
    # Market metrics
    market_share: float
    competitive_position: str
    brand_recognition: float
    market_growth_rate: float
    
    # Operational metrics
    employee_count: int
    revenue_per_employee: Decimal
    customer_satisfaction: float
    employee_satisfaction: float
    operational_efficiency: float
    
    # Strategic metrics
    product_adoption_rate: float
    innovation_index: float
    digital_transformation_score: float
    sustainability_score: float
    
    def get_executive_summary_score(self) -> Dict[str, float]:
        """Calculate executive summary performance scores"""
        
        # Financial health score (30% weight)
        financial_score = 0
        if self.gross_margin > 0.7:
            financial_score += 20
        elif self.gross_margin > 0.5:
            financial_score += 15
        
        if self.revenue_growth > 0.2:
            financial_score += 10
        elif self.revenue_growth > 0:
            financial_score += 5
        
        # Customer health score (25% weight)
        customer_score = 0
        if self.churn_rate < 0.05:
            customer_score += 15
        elif self.churn_rate < 0.10:
            customer_score += 10
        
        if self.net_revenue_retention > 1.1:
            customer_score += 10
        elif self.net_revenue_retention > 1.0:
            customer_score += 5
        
        # Sales effectiveness score (20% weight)
        sales_score = 0
        if self.win_rate > 0.25:
            sales_score += 10
        elif self.win_rate > 0.15:
            sales_score += 7
        
        if self.quota_attainment > 0.9:
            sales_score += 10
        elif self.quota_attainment > 0.75:
            sales_score += 5
        
        # Market position score (15% weight)
        market_score = 0
        if self.market_share > 0.2:
            market_score += 15
        elif self.market_share > 0.1:
            market_score += 10
        
        # Operational excellence score (10% weight)
        operational_score = 0
        if self.customer_satisfaction > 8.5:
            operational_score += 10
        elif self.customer_satisfaction > 7.5:
            operational_score += 7
        
        total_score = financial_score + customer_score + sales_score + market_score + operational_score
        
        return {
            'financial_health': financial_score,
            'customer_health': customer_score,
            'sales_effectiveness': sales_score,
            'market_position': market_score,
            'operational_excellence': operational_score,
            'overall_score': min(total_score, 100)
        }
    
    def get_risk_assessment(self) -> Dict[str, Any]:
        """Assess business risks based on metrics"""
        risks = []
        risk_level = "Low"
        
        # Financial risks
        if self.revenue_growth < 0:
            risks.append({
                'category': 'Financial',
                'risk': 'Negative revenue growth',
                'severity': 'High',
                'mitigation': 'Increase marketing investment, review pricing strategy'
            })
            risk_level = "High"
        
        if self.churn_rate > 0.15:
            risks.append({
                'category': 'Customer',
                'risk': 'High customer churn rate',
                'severity': 'High',
                'mitigation': 'Improve customer success programs, enhance product value'
            })
            risk_level = "High"
        
        if self.win_rate < 0.1:
            risks.append({
                'category': 'Sales',
                'risk': 'Low sales win rate',
                'severity': 'Medium',
                'mitigation': 'Improve sales training, review product positioning'
            })
            if risk_level == "Low":
                risk_level = "Medium"
        
        # Market risks
        if self.market_share < 0.05:
            risks.append({
                'category': 'Market',
                'risk': 'Low market share',
                'severity': 'Medium',
                'mitigation': 'Increase market presence, improve competitive differentiation'
            })
            if risk_level == "Low":
                risk_level = "Medium"
        
        return {
            'overall_risk_level': risk_level,
            'risks_identified': len(risks),
            'risk_details': risks
        }