"""
Executive Dashboard and Reporting System
Provides comprehensive executive-level reporting and dashboards
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal
import logging
from dataclasses import asdict

# Import necessary components
from ..data_models.kpi_models import KPIMetrics, KPIType, ExecutiveMetrics
from ..kpi_tracking.kpi_monitor import KPIMonitor
from ..cohort_analysis.cohort_analyzer import CohortAnalyzer
from ..revenue_forecasting.revenue_predictor import RevenuePredictor
from ..pipeline_management.pipeline_analyzer import PipelineAnalyzer

class ExecutiveDashboard:
    """Executive-level dashboard and reporting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dashboard configuration
        self.dashboard_refresh_interval = config.get('dashboard_refresh_minutes', 60)
        self.report_generation_time = config.get('report_generation_time', '08:00')
        self.executive_distribution_list = config.get('executive_distribution_list', [])
        
        # Initialize supporting components
        self.kpi_monitor = KPIMonitor(config.get('kpi_config', {}))
        self.cohort_analyzer = CohortAnalyzer(config.get('cohort_config', {}))
        self.revenue_predictor = RevenuePredictor(config.get('revenue_config', {}))
        self.pipeline_analyzer = PipelineAnalyzer(config.get('pipeline_config', {}))
        
        # Dashboard templates
        self.dashboard_templates = self._load_dashboard_templates()
        
    def create_dashboard(self, kpi_data: List[KPIMetrics], 
                        analysis_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create comprehensive executive dashboard"""
        self.logger.info("Creating executive dashboard")
        
        # Generate dashboard sections
        dashboard_overview = self._create_dashboard_overview(kpi_data)
        financial_section = self._create_financial_section(kpi_data)
        customer_section = self._create_customer_section(kpi_data, analysis_data)
        sales_section = self._create_sales_section(kpi_data, analysis_data)
        market_section = self._create_market_section(kpi_data)
        operational_section = self._create_operational_section(kpi_data)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            dashboard_overview, financial_section, customer_section, 
            sales_section, market_section, operational_section
        )
        
        # Calculate dashboard score
        dashboard_score = self._calculate_dashboard_score(dashboard_overview)
        
        # Identify critical items
        critical_items = self._identify_critical_items(dashboard_overview)
        
        dashboard_data = {
            'dashboard_metadata': {
                'dashboard_type': 'executive',
                'generation_time': datetime.now().isoformat(),
                'refresh_interval': self.dashboard_refresh_interval,
                'dashboard_score': dashboard_score
            },
            'executive_summary': executive_summary,
            'dashboard_overview': dashboard_overview,
            'financial_metrics': financial_section,
            'customer_metrics': customer_section,
            'sales_metrics': sales_section,
            'market_metrics': market_section,
            'operational_metrics': operational_section,
            'critical_items': critical_items,
            'recommendations': self._generate_dashboard_recommendations(dashboard_overview),
            'data_quality_indicators': self._assess_data_quality(kpi_data)
        }
        
        return dashboard_data
    
    def _create_dashboard_overview(self, kpi_data: List[KPIMetrics]) -> Dict[str, Any]:
        """Create high-level dashboard overview"""
        # Calculate overall metrics
        total_kpis = len(kpi_data)
        excellent_kpis = len([kpi for kpi in kpi_data if kpi.performance_status.value == 'excellent'])
        good_kpis = len([kpi for kpi in kpi_data if kpi.performance_status.value == 'good'])
        critical_kpis = len([kpi for kpi in kpi_data if kpi.performance_status.value == 'critical'])
        
        # Calculate overall health score
        health_score = self._calculate_overall_health_score(kpi_data)
        
        # Identify top performers and concerns
        top_performers = self._identify_top_performers(kpi_data)
        major_concerns = self._identify_major_concerns(kpi_data)
        
        return {
            'overall_health_score': health_score,
            'total_kpis_monitored': total_kpis,
            'kpi_distribution': {
                'excellent': excellent_kpis,
                'good': good_kpis,
                'critical': critical_kpis,
                'health_percentage': (excellent_kpis + good_kpis) / total_kpis * 100 if total_kpis > 0 else 0
            },
            'top_performers': top_performers,
            'major_concerns': major_concerns,
            'trend_summary': self._summarize_trends(kpi_data),
            'key_highlights': self._generate_key_highlights(kpi_data)
        }
    
    def _create_financial_section(self, kpi_data: List[KPIMetrics]) -> Dict[str, Any]:
        """Create financial metrics section"""
        # Filter financial KPIs
        financial_kpis = [kpi for kpi in kpi_data if kpi.kpi_type == KPIType.FINANCIAL]
        
        # Extract key financial metrics
        revenue_kpi = next((kpi for kpi in financial_kpis if kpi.kpi_id == 'revenue'), None)
        margin_kpi = next((kpi for kpi in financial_kpis if kpi.kpi_id == 'gross_margin'), None)
        
        # Calculate financial health indicators
        revenue_trend = revenue_kpi.trend_direction if revenue_kpi else 'stable'
        margin_health = margin_kpi.performance_status.value if margin_kpi else 'unknown'
        
        # Generate insights
        insights = self._generate_financial_insights(financial_kpis)
        
        return {
            'key_metrics': {
                'revenue': {
                    'current_value': float(revenue_kpi.current_value) if revenue_kpi else 0,
                    'target_value': float(revenue_kpi.target_value) if revenue_kpi else 0,
                    'variance': revenue_kpi.variance_from_target if revenue_kpi else 0,
                    'trend': revenue_trend
                },
                'gross_margin': {
                    'current_value': float(margin_kpi.current_value) if margin_kpi else 0,
                    'target_value': float(margin_kpi.target_value) if margin_kpi else 0,
                    'status': margin_health
                }
            },
            'financial_health_score': self._calculate_financial_health_score(financial_kpis),
            'insights': insights,
            'revenue_forecast': self._generate_revenue_insights(),
            'cost_management': self._assess_cost_management(financial_kpis)
        }
    
    def _create_customer_section(self, kpi_data: List[KPIMetrics], 
                               analysis_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create customer metrics section"""
        # Filter customer KPIs
        customer_kpis = [kpi for kpi in kpi_data if kpi.kpi_type == KPIType.CUSTOMER]
        
        # Extract key customer metrics
        acquisition_kpi = next((kpi for kpi in customer_kpis if kpi.kpi_id == 'customer_acquisition'), None)
        churn_kpi = next((kpi for kpi in customer_kpis if kpi.kpi_id == 'churn_rate'), None)
        nrr_kpi = next((kpi for kpi in customer_kpis if kpi.kpi_id == 'net_revenue_retention'), None)
        
        # Get cohort analysis if available
        cohort_insights = []
        if analysis_data and 'customer_analysis' in analysis_data:
            cohort_insights = analysis_data['customer_analysis'].get('insights', [])
        
        return {
            'key_metrics': {
                'customer_acquisition': {
                    'current_value': int(acquisition_kpi.current_value) if acquisition_kpi else 0,
                    'target_value': int(acquisition_kpi.target_value) if acquisition_kpi else 0,
                    'variance': acquisition_kpi.variance_from_target if acquisition_kpi else 0,
                    'trend': acquisition_kpi.trend_direction if acquisition_kpi else 'stable'
                },
                'churn_rate': {
                    'current_value': float(churn_kpi.current_value) if churn_kpi else 0,
                    'target_value': float(churn_kpi.target_value) if churn_kpi else 0,
                    'status': churn_kpi.performance_status.value if churn_kpi else 'unknown'
                },
                'net_revenue_retention': {
                    'current_value': float(nrr_kpi.current_value) if nrr_kpi else 0,
                    'target_value': float(nrr_kpi.target_value) if nrr_kpi else 0,
                    'trend': nrr_kpi.trend_direction if nrr_kpi else 'stable'
                }
            },
            'customer_health_score': self._calculate_customer_health_score(customer_kpis),
            'cohort_analysis_insights': cohort_insights,
            'retention_metrics': self._assess_retention_health(customer_kpis),
            'expansion_opportunities': self._identify_expansion_opportunities(customer_kpis)
        }
    
    def _create_sales_section(self, kpi_data: List[KPIMetrics], 
                            analysis_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create sales metrics section"""
        # Filter sales KPIs
        sales_kpis = [kpi for kpi in kpi_data if kpi.kpi_type == KPIType.SALES]
        
        # Extract key sales metrics
        win_rate_kpi = next((kpi for kpi in sales_kpis if kpi.kpi_id == 'win_rate'), None)
        pipeline_kpi = next((kpi for kpi in sales_kpis if kpi.kpi_id == 'pipeline_coverage'), None)
        deal_size_kpi = next((kpi for kpi in sales_kpis if kpi.kpi_id == 'average_deal_size'), None)
        
        # Get pipeline insights if available
        pipeline_insights = []
        if analysis_data and 'sales_analysis' in analysis_data:
            pipeline_insights = analysis_data['sales_analysis'].get('insights', [])
        
        return {
            'key_metrics': {
                'win_rate': {
                    'current_value': float(win_rate_kpi.current_value) if win_rate_kpi else 0,
                    'target_value': float(win_rate_kpi.target_value) if win_rate_kpi else 0,
                    'variance': win_rate_kpi.variance_from_target if win_rate_kpi else 0,
                    'trend': win_rate_kpi.trend_direction if win_rate_kpi else 'stable'
                },
                'pipeline_coverage': {
                    'current_value': float(pipeline_kpi.current_value) if pipeline_kpi else 0,
                    'target_value': float(pipeline_kpi.target_value) if pipeline_kpi else 0,
                    'status': pipeline_kpi.performance_status.value if pipeline_kpi else 'unknown'
                },
                'average_deal_size': {
                    'current_value': float(deal_size_kpi.current_value) if deal_size_kpi else 0,
                    'trend': deal_size_kpi.trend_direction if deal_size_kpi else 'stable'
                }
            },
            'sales_health_score': self._calculate_sales_health_score(sales_kpis),
            'pipeline_insights': pipeline_insights,
            'sales_forecast': self._generate_sales_forecast_insights(),
            'performance_by_rep': self._assess_rep_performance(sales_kpis)
        }
    
    def _create_market_section(self, kpi_data: List[KPIMetrics]) -> Dict[str, Any]:
        """Create market metrics section"""
        # Filter market-related KPIs
        market_kpis = [kpi for kpi in kpi_data if kpi.kpi_type in [KPIType.MARKETING, KPIType.EXECUTIVE]]
        
        # Extract key market metrics
        market_share_kpi = next((kpi for kpi in market_kpis if kpi.kpi_id == 'market_share'), None)
        cac_kpi = next((kpi for kpi in market_kpis if kpi.kpi_id == 'cost_per_acquisition'), None)
        conversion_kpi = next((kpi for kpi in market_kpis if kpi.kpi_id == 'conversion_rate'), None)
        
        return {
            'key_metrics': {
                'market_share': {
                    'current_value': float(market_share_kpi.current_value) if market_share_kpi else 0,
                    'trend': market_share_kpi.trend_direction if market_share_kpi else 'stable'
                },
                'cost_per_acquisition': {
                    'current_value': float(cac_kpi.current_value) if cac_kpi else 0,
                    'target_value': float(cac_kpi.target_value) if cac_kpi else 0,
                    'status': cac_kpi.performance_status.value if cac_kpi else 'unknown'
                },
                'conversion_rate': {
                    'current_value': float(conversion_kpi.current_value) if conversion_kpi else 0,
                    'trend': conversion_kpi.trend_direction if conversion_kpi else 'stable'
                }
            },
            'market_position': self._assess_market_position(market_kpis),
            'competitive_landscape': self._assess_competitive_position(),
            'growth_opportunities': self._identify_growth_opportunities(market_kpis)
        }
    
    def _create_operational_section(self, kpi_data: List[KPIMetrics]) -> Dict[str, Any]:
        """Create operational metrics section"""
        # Filter operational KPIs
        operational_kpis = [kpi for kpi in kpi_data if kpi.kpi_type == KPIType.OPERATIONAL]
        
        # Extract key operational metrics
        satisfaction_kpi = next((kpi for kpi in operational_kpis if kpi.kpi_id == 'customer_satisfaction'), None)
        efficiency_kpi = next((kpi for kpi in operational_kpis if kpi.kpi_id == 'operational_efficiency'), None)
        
        return {
            'key_metrics': {
                'customer_satisfaction': {
                    'current_value': float(satisfaction_kpi.current_value) if satisfaction_kpi else 0,
                    'target_value': float(satisfaction_kpi.target_value) if satisfaction_kpi else 0,
                    'trend': satisfaction_kpi.trend_direction if satisfaction_kpi else 'stable'
                },
                'operational_efficiency': {
                    'current_value': float(efficiency_kpi.current_value) if efficiency_kpi else 0,
                    'status': efficiency_kpi.performance_status.value if efficiency_kpi else 'unknown'
                }
            },
            'operational_health_score': self._calculate_operational_health_score(operational_kpis),
            'efficiency_trends': self._analyze_efficiency_trends(operational_kpis),
            'improvement_initiatives': self._recommend_improvements(operational_kpis)
        }
    
    def _generate_executive_summary(self, *sections) -> Dict[str, Any]:
        """Generate executive summary across all sections"""
        overview = sections[0] if sections else {}
        
        # Calculate overall business health
        overall_score = overview.get('overall_health_score', 0)
        
        # Identify critical actions
        critical_actions = []
        if overview.get('major_concerns'):
            critical_actions.append({
                'priority': 'Critical',
                'area': 'Performance',
                'action': 'Address identified concerns',
                'timeline': 'Within 48 hours'
            })
        
        # Generate highlights
        highlights = overview.get('key_highlights', [])
        
        return {
            'business_health_score': overall_score,
            'overall_status': self._determine_overall_status(overall_score),
            'critical_actions': critical_actions,
            'key_highlights': highlights,
            'strategic_recommendations': self._generate_strategic_recommendations(sections),
            'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
        }
    
    def _load_dashboard_templates(self) -> Dict[str, Any]:
        """Load dashboard configuration templates"""
        return {
            'executive': {
                'layout': 'grid',
                'refresh_interval': 60,
                'sections': ['overview', 'financial', 'customer', 'sales', 'market', 'operational'],
                'color_scheme': 'corporate',
                'show_trends': True,
                'show_targets': True
            },
            'operational': {
                'layout': 'detailed',
                'refresh_interval': 30,
                'sections': ['operations', 'performance', 'alerts'],
                'color_scheme': 'operational',
                'show_trends': True,
                'show_targets': False
            }
        }
    
    # Helper methods for score calculations
    
    def _calculate_overall_health_score(self, kpi_data: List[KPIMetrics]) -> float:
        """Calculate overall business health score"""
        if not kpi_data:
            return 0
        
        scores = [kpi.calculate_performance_score() for kpi in kpi_data]
        return sum(scores) / len(scores)
    
    def _calculate_financial_health_score(self, financial_kpis: List[KPIMetrics]) -> float:
        """Calculate financial health score"""
        if not financial_kpis:
            return 0
        
        # Weight financial KPIs differently
        weights = {
            'revenue': 0.5,
            'gross_margin': 0.3,
            'operating_margin': 0.2
        }
        
        weighted_score = 0
        total_weight = 0
        
        for kpi in financial_kpis:
            weight = weights.get(kpi.kpi_id, 0.1)
            score = kpi.calculate_performance_score()
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0
    
    def _calculate_customer_health_score(self, customer_kpis: List[KPIMetrics]) -> float:
        """Calculate customer health score"""
        if not customer_kpis:
            return 0
        
        scores = [kpi.calculate_performance_score() for kpi in customer_kpis]
        return sum(scores) / len(scores)
    
    def _calculate_sales_health_score(self, sales_kpis: List[KPIMetrics]) -> float:
        """Calculate sales health score"""
        if not sales_kpis:
            return 0
        
        scores = [kpi.calculate_performance_score() for kpi in sales_kpis]
        return sum(scores) / len(scores)
    
    def _calculate_operational_health_score(self, operational_kpis: List[KPIMetrics]) -> float:
        """Calculate operational health score"""
        if not operational_kpis:
            return 0
        
        scores = [kpi.calculate_performance_score() for kpi in operational_kpis]
        return sum(scores) / len(scores)
    
    def _calculate_dashboard_score(self, overview: Dict[str, Any]) -> float:
        """Calculate overall dashboard score"""
        health_percentage = overview.get('kpi_distribution', {}).get('health_percentage', 0)
        return health_percentage
    
    # Helper methods for insights and analysis
    
    def _identify_top_performers(self, kpi_data: List[KPIMetrics]) -> List[Dict[str, Any]]:
        """Identify top performing KPIs"""
        excellent_kpis = [kpi for kpi in kpi_data if kpi.performance_status.value == 'excellent']
        
        top_performers = []
        for kpi in excellent_kpis[:5]:  # Top 5
            top_performers.append({
                'kpi_name': kpi.kpi_name,
                'current_value': float(kpi.current_value),
                'target_value': float(kpi.target_value),
                'variance': kpi.variance_from_target,
                'status': kpi.performance_status.value
            })
        
        return top_performers
    
    def _identify_major_concerns(self, kpi_data: List[KPIMetrics]) -> List[Dict[str, Any]]:
        """Identify major concerns"""
        critical_kpis = [kpi for kpi in kpi_data if kpi.performance_status.value == 'critical']
        below_target_kpis = [kpi for kpi in kpi_data if kpi.performance_status.value == 'below_target']
        
        concerns = []
        
        for kpi in critical_kpis:
            concerns.append({
                'kpi_name': kpi.kpi_name,
                'current_value': float(kpi.current_value),
                'target_value': float(kpi.target_value),
                'severity': 'critical',
                'recommended_action': 'Immediate attention required'
            })
        
        for kpi in below_target_kpis[:3]:  # Top 3 concerns
            concerns.append({
                'kpi_name': kpi.kpi_name,
                'current_value': float(kpi.current_value),
                'target_value': float(kpi.target_value),
                'severity': 'warning',
                'recommended_action': 'Performance improvement needed'
            })
        
        return concerns
    
    def _summarize_trends(self, kpi_data: List[KPIMetrics]) -> Dict[str, int]:
        """Summarize KPI trends"""
        trends = {'improving': 0, 'declining': 0, 'stable': 0}
        
        for kpi in kpi_data:
            trend = kpi.trend_direction
            if trend in trends:
                trends[trend] += 1
        
        return trends
    
    def _generate_key_highlights(self, kpi_data: List[KPIMetrics]) -> List[str]:
        """Generate key highlights"""
        highlights = []
        
        # Revenue highlight
        revenue_kpi = next((kpi for kpi in kpi_data if kpi.kpi_id == 'revenue'), None)
        if revenue_kpi and revenue_kpi.variance_from_target > 0.05:
            highlights.append(f"Revenue exceeded target by {revenue_kpi.variance_from_target:.1%}")
        
        # Customer satisfaction highlight
        satisfaction_kpi = next((kpi for kpi in kpi_data if kpi.kpi_id == 'customer_satisfaction'), None)
        if satisfaction_kpi and satisfaction_kpi.performance_status.value in ['excellent', 'good']:
            highlights.append(f"Customer satisfaction at {satisfaction_kpi.current_value:.1f}/10")
        
        # Growth highlight
        nrr_kpi = next((kpi for kpi in kpi_data if kpi.kpi_id == 'net_revenue_retention'), None)
        if nrr_kpi and float(nrr_kpi.current_value) > 1.1:
            highlights.append(f"Strong net revenue retention at {float(nrr_kpi.current_value):.1%}")
        
        return highlights
    
    def _determine_overall_status(self, health_score: float) -> str:
        """Determine overall business status"""
        if health_score >= 90:
            return 'Excellent'
        elif health_score >= 75:
            return 'Good'
        elif health_score >= 60:
            return 'Fair'
        elif health_score >= 40:
            return 'Needs Attention'
        else:
            return 'Critical'
    
    def _generate_dashboard_recommendations(self, overview: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dashboard recommendations"""
        recommendations = []
        
        health_score = overview.get('overall_health_score', 0)
        
        if health_score < 60:
            recommendations.append({
                'category': 'Performance',
                'recommendation': 'Focus on critical KPIs below target',
                'priority': 'High',
                'timeline': 'Immediate'
            })
        
        # Add more specific recommendations based on data
        return recommendations
    
    def _assess_data_quality(self, kpi_data: List[KPIMetrics]) -> Dict[str, Any]:
        """Assess data quality indicators"""
        total_kpis = len(kpi_data)
        complete_kpis = len([kpi for kpi in kpi_data if kpi.current_value > 0])
        
        return {
            'data_completeness': complete_kpis / total_kpis if total_kpis > 0 else 0,
            'last_updated': datetime.now().isoformat(),
            'data_freshness': 'Good'  # Simplified
        }
    
    # Additional helper methods (simplified implementations)
    def _generate_financial_insights(self, financial_kpis: List[KPIMetrics]) -> List[str]:
        return ["Revenue growth trending positively", "Margin performance within target range"]
    
    def _generate_revenue_insights(self) -> Dict[str, Any]:
        return {"forecast_confidence": "High", "growth_projections": "Positive"}
    
    def _assess_cost_management(self, financial_kpis: List[KPIMetrics]) -> Dict[str, Any]:
        return {"cost_control": "Good", "efficiency_score": 85}
    
    def _assess_retention_health(self, customer_kpis: List[KPIMetrics]) -> Dict[str, Any]:
        return {"retention_trend": "Stable", "churn_risk": "Low"}
    
    def _identify_expansion_opportunities(self, customer_kpis: List[KPIMetrics]) -> List[str]:
        return ["Enterprise upselling", "Cross-sell existing customer base"]
    
    def _generate_sales_forecast_insights(self) -> Dict[str, Any]:
        return {"pipeline_health": "Good", "forecast_accuracy": "High"}
    
    def _assess_rep_performance(self, sales_kpis: List[KPIMetrics]) -> Dict[str, Any]:
        return {"top_performers": "2 reps", "improvement_needed": "1 rep"}
    
    def _assess_market_position(self, market_kpis: List[KPIMetrics]) -> Dict[str, Any]:
        return {"market_position": "Strong", "competitive_advantage": "Product differentiation"}
    
    def _assess_competitive_position(self) -> Dict[str, Any]:
        return {"competitive_momentum": "Positive", "market_share_trend": "Growing"}
    
    def _identify_growth_opportunities(self, market_kpis: List[KPIMetrics]) -> List[str]:
        return ["Geographic expansion", "New customer segments"]
    
    def _analyze_efficiency_trends(self, operational_kpis: List[KPIMetrics]) -> Dict[str, Any]:
        return {"efficiency_trend": "Improving", "productivity_gains": "Positive"}
    
    def _recommend_improvements(self, operational_kpis: List[KPIMetrics]) -> List[str]:
        return ["Process automation", "Training programs"]
    
    def _generate_strategic_recommendations(self, sections) -> List[str]:
        return ["Focus on customer retention initiatives", "Optimize sales process efficiency"]