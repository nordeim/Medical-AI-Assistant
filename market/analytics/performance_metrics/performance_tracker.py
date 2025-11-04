"""
Performance Tracking and Benchmarking System
Tracks business performance against targets and benchmarks
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import logging
from dataclasses import asdict

class PerformanceTracker:
    """Performance tracking and monitoring system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking parameters
        self.tracking_periods = config.get('tracking_periods', ['daily', 'weekly', 'monthly', 'quarterly'])
        self.benchmark_sources = config.get('benchmark_sources', {})
        self.performance_thresholds = config.get('performance_thresholds', {
            'excellent': 0.9,
            'good': 0.8,
            'satisfactory': 0.7,
            'needs_improvement': 0.6,
            'critical': 0.5
        })
        
        # Performance history
        self.performance_history = {}
        self.benchmark_data = {}
        
    def track_performance_metrics(self, metrics_data: Dict[str, Any], 
                                period: str = 'monthly') -> Dict[str, Any]:
        """Track performance against targets"""
        self.logger.info(f"Tracking performance metrics for period: {period}")
        
        performance_results = {
            'tracking_period': period,
            'tracking_date': datetime.now().isoformat(),
            'metrics_tracked': [],
            'performance_summary': {},
            'benchmark_comparison': {},
            'performance_alerts': [],
            'improvement_opportunities': []
        }
        
        # Track financial performance
        financial_performance = self._track_financial_performance(metrics_data)
        performance_results['metrics_tracked'].append(financial_performance)
        
        # Track customer performance
        customer_performance = self._track_customer_performance(metrics_data)
        performance_results['metrics_tracked'].append(customer_performance)
        
        # Track sales performance
        sales_performance = self._track_sales_performance(metrics_data)
        performance_results['metrics_tracked'].append(sales_performance)
        
        # Track operational performance
        operational_performance = self._track_operational_performance(metrics_data)
        performance_results['metrics_tracked'].append(operational_performance)
        
        # Generate performance summary
        performance_results['performance_summary'] = self._generate_performance_summary(
            performance_results['metrics_tracked']
        )
        
        # Compare against benchmarks
        performance_results['benchmark_comparison'] = self._compare_against_benchmarks(
            performance_results['metrics_tracked']
        )
        
        # Generate alerts
        performance_results['performance_alerts'] = self._generate_performance_alerts(
            performance_results['metrics_tracked']
        )
        
        # Identify opportunities
        performance_results['improvement_opportunities'] = self._identify_improvement_opportunities(
            performance_results['metrics_tracked']
        )
        
        return performance_results
    
    def _track_financial_performance(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track financial performance metrics"""
        financial_metrics = metrics_data.get('financial', {})
        
        # Revenue performance
        revenue = Decimal(str(financial_metrics.get('revenue', '1000000')))
        revenue_target = Decimal(str(financial_metrics.get('revenue_target', '1000000')))
        revenue_performance = float(revenue / revenue_target) if revenue_target > 0 else 0
        
        # Gross margin performance
        gross_margin = Decimal(str(financial_metrics.get('gross_margin', '0.75')))
        margin_target = Decimal(str(financial_metrics.get('margin_target', '0.75')))
        margin_performance = float(gross_margin / margin_target) if margin_target > 0 else 0
        
        # Cost efficiency
        cost_efficiency = Decimal(str(financial_metrics.get('cost_efficiency', '0.85')))
        efficiency_target = Decimal(str(financial_metrics.get('efficiency_target', '0.85')))
        efficiency_performance = float(cost_efficiency / efficiency_target) if efficiency_target > 0 else 0
        
        return {
            'category': 'Financial',
            'metrics': {
                'revenue_performance': {
                    'actual': float(revenue),
                    'target': float(revenue_target),
                    'performance_ratio': revenue_performance,
                    'status': self._get_performance_status(revenue_performance)
                },
                'gross_margin_performance': {
                    'actual': float(gross_margin),
                    'target': float(margin_target),
                    'performance_ratio': margin_performance,
                    'status': self._get_performance_status(margin_performance)
                },
                'cost_efficiency_performance': {
                    'actual': float(cost_efficiency),
                    'target': float(efficiency_target),
                    'performance_ratio': efficiency_performance,
                    'status': self._get_performance_status(efficiency_performance)
                }
            },
            'overall_score': (revenue_performance + margin_performance + efficiency_performance) / 3,
            'trend': self._calculate_financial_trend(financial_metrics)
        }
    
    def _track_customer_performance(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track customer performance metrics"""
        customer_metrics = metrics_data.get('customer', {})
        
        # Customer acquisition
        new_customers = Decimal(str(customer_metrics.get('new_customers', '50')))
        acquisition_target = Decimal(str(customer_metrics.get('acquisition_target', '50')))
        acquisition_performance = float(new_customers / acquisition_target) if acquisition_target > 0 else 0
        
        # Churn rate (lower is better)
        churn_rate = Decimal(str(customer_metrics.get('churn_rate', '0.05')))
        churn_target = Decimal(str(customer_metrics.get('churn_target', '0.05')))
        # For churn, performance is inverse
        churn_performance = float(churn_target / churn_rate) if churn_rate > 0 and churn_target > 0 else 0
        
        # Net revenue retention
        nrr = Decimal(str(customer_metrics.get('net_revenue_retention', '1.10')))
        nrr_target = Decimal(str(customer_metrics.get('nrr_target', '1.10')))
        nrr_performance = float(nrr / nrr_target) if nrr_target > 0 else 0
        
        # Customer satisfaction
        satisfaction = Decimal(str(customer_metrics.get('satisfaction_score', '8.5')))
        satisfaction_target = Decimal(str(customer_metrics.get('satisfaction_target', '8.5')))
        satisfaction_performance = float(satisfaction / satisfaction_target) if satisfaction_target > 0 else 0
        
        return {
            'category': 'Customer',
            'metrics': {
                'customer_acquisition_performance': {
                    'actual': float(new_customers),
                    'target': float(acquisition_target),
                    'performance_ratio': acquisition_performance,
                    'status': self._get_performance_status(acquisition_performance)
                },
                'churn_rate_performance': {
                    'actual': float(churn_rate),
                    'target': float(churn_target),
                    'performance_ratio': churn_performance,
                    'status': self._get_performance_status(churn_performance)
                },
                'net_revenue_retention_performance': {
                    'actual': float(nrr),
                    'target': float(nrr_target),
                    'performance_ratio': nrr_performance,
                    'status': self._get_performance_status(nrr_performance)
                },
                'customer_satisfaction_performance': {
                    'actual': float(satisfaction),
                    'target': float(satisfaction_target),
                    'performance_ratio': satisfaction_performance,
                    'status': self._get_performance_status(satisfaction_performance)
                }
            },
            'overall_score': (acquisition_performance + churn_performance + nrr_performance + satisfaction_performance) / 4,
            'trend': self._calculate_customer_trend(customer_metrics)
        }
    
    def _track_sales_performance(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track sales performance metrics"""
        sales_metrics = metrics_data.get('sales', {})
        
        # Win rate
        win_rate = Decimal(str(sales_metrics.get('win_rate', '0.25')))
        win_rate_target = Decimal(str(sales_metrics.get('win_rate_target', '0.25')))
        win_rate_performance = float(win_rate / win_rate_target) if win_rate_target > 0 else 0
        
        # Pipeline coverage
        pipeline_coverage = Decimal(str(sales_metrics.get('pipeline_coverage', '3.0')))
        coverage_target = Decimal(str(sales_metrics.get('coverage_target', '3.0')))
        coverage_performance = float(pipeline_coverage / coverage_target) if coverage_target > 0 else 0
        
        # Average deal size
        deal_size = Decimal(str(sales_metrics.get('average_deal_size', '25000')))
        deal_size_target = Decimal(str(sales_metrics.get('deal_size_target', '25000')))
        deal_size_performance = float(deal_size / deal_size_target) if deal_size_target > 0 else 0
        
        # Sales cycle length
        sales_cycle = Decimal(str(sales_metrics.get('sales_cycle_days', '45')))
        cycle_target = Decimal(str(sales_metrics.get('cycle_target', '45')))
        # For sales cycle, shorter is better
        cycle_performance = float(cycle_target / sales_cycle) if sales_cycle > 0 and cycle_target > 0 else 0
        
        return {
            'category': 'Sales',
            'metrics': {
                'win_rate_performance': {
                    'actual': float(win_rate),
                    'target': float(win_rate_target),
                    'performance_ratio': win_rate_performance,
                    'status': self._get_performance_status(win_rate_performance)
                },
                'pipeline_coverage_performance': {
                    'actual': float(pipeline_coverage),
                    'target': float(coverage_target),
                    'performance_ratio': coverage_performance,
                    'status': self._get_performance_status(coverage_performance)
                },
                'average_deal_size_performance': {
                    'actual': float(deal_size),
                    'target': float(deal_size_target),
                    'performance_ratio': deal_size_performance,
                    'status': self._get_performance_status(deal_size_performance)
                },
                'sales_cycle_performance': {
                    'actual': float(sales_cycle),
                    'target': float(cycle_target),
                    'performance_ratio': cycle_performance,
                    'status': self._get_performance_status(cycle_performance)
                }
            },
            'overall_score': (win_rate_performance + coverage_performance + deal_size_performance + cycle_performance) / 4,
            'trend': self._calculate_sales_trend(sales_metrics)
        }
    
    def _track_operational_performance(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track operational performance metrics"""
        operational_metrics = metrics_data.get('operational', {})
        
        # Operational efficiency
        efficiency = Decimal(str(operational_metrics.get('operational_efficiency', '0.85')))
        efficiency_target = Decimal(str(operational_metrics.get('efficiency_target', '0.85')))
        efficiency_performance = float(efficiency / efficiency_target) if efficiency_target > 0 else 0
        
        # Employee satisfaction
        employee_satisfaction = Decimal(str(operational_metrics.get('employee_satisfaction', '8.0')))
        emp_sat_target = Decimal(str(operational_metrics.get('emp_sat_target', '8.0')))
        emp_sat_performance = float(employee_satisfaction / emp_sat_target) if emp_sat_target > 0 else 0
        
        # Process automation level
        automation = Decimal(str(operational_metrics.get('automation_level', '0.60')))
        automation_target = Decimal(str(operational_metrics.get('automation_target', '0.60')))
        automation_performance = float(automation / automation_target) if automation_target > 0 else 0
        
        # Quality metrics
        quality = Decimal(str(operational_metrics.get('quality_score', '0.90')))
        quality_target = Decimal(str(operational_metrics.get('quality_target', '0.90')))
        quality_performance = float(quality / quality_target) if quality_target > 0 else 0
        
        return {
            'category': 'Operational',
            'metrics': {
                'operational_efficiency_performance': {
                    'actual': float(efficiency),
                    'target': float(efficiency_target),
                    'performance_ratio': efficiency_performance,
                    'status': self._get_performance_status(efficiency_performance)
                },
                'employee_satisfaction_performance': {
                    'actual': float(employee_satisfaction),
                    'target': float(emp_sat_target),
                    'performance_ratio': emp_sat_performance,
                    'status': self._get_performance_status(emp_sat_performance)
                },
                'automation_performance': {
                    'actual': float(automation),
                    'target': float(automation_target),
                    'performance_ratio': automation_performance,
                    'status': self._get_performance_status(automation_performance)
                },
                'quality_performance': {
                    'actual': float(quality),
                    'target': float(quality_target),
                    'performance_ratio': quality_performance,
                    'status': self._get_performance_status(quality_performance)
                }
            },
            'overall_score': (efficiency_performance + emp_sat_performance + automation_performance + quality_performance) / 4,
            'trend': self._calculate_operational_trend(operational_metrics)
        }
    
    def _get_performance_status(self, performance_ratio: float) -> str:
        """Get performance status based on ratio"""
        if performance_ratio >= self.performance_thresholds['excellent']:
            return 'excellent'
        elif performance_ratio >= self.performance_thresholds['good']:
            return 'good'
        elif performance_ratio >= self.performance_thresholds['satisfactory']:
            return 'satisfactory'
        elif performance_ratio >= self.performance_thresholds['needs_improvement']:
            return 'needs_improvement'
        else:
            return 'critical'
    
    def _generate_performance_summary(self, tracked_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall performance summary"""
        total_score = 0
        category_scores = {}
        metric_count = 0
        
        for category_data in tracked_metrics:
            category = category_data['category']
            category_score = category_data['overall_score']
            category_scores[category] = category_score
            total_score += category_score
            metric_count += 1
        
        overall_score = total_score / metric_count if metric_count > 0 else 0
        
        return {
            'overall_performance_score': overall_score,
            'overall_status': self._get_performance_status(overall_score),
            'category_scores': category_scores,
            'best_performing_category': max(category_scores.keys(), key=lambda k: category_scores[k]) if category_scores else None,
            'worst_performing_category': min(category_scores.keys(), key=lambda k: category_scores[k]) if category_scores else None,
            'performance_distribution': self._calculate_performance_distribution(tracked_metrics)
        }
    
    def _compare_against_benchmarks(self, tracked_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare performance against industry benchmarks"""
        benchmark_comparison = {
            'industry_benchmarks': {},
            'competitive_position': {},
            'benchmark_score': 0,
            'improvement_areas': []
        }
        
        # Mock industry benchmarks
        industry_benchmarks = {
            'revenue_growth': 0.15,
            'customer_satisfaction': 8.2,
            'churn_rate': 0.06,
            'win_rate': 0.22,
            'gross_margin': 0.72,
            'operational_efficiency': 0.82
        }
        
        benchmark_scores = []
        
        for category_data in tracked_metrics:
            category = category_data['category']
            for metric_name, metric_data in category_data['metrics'].items():
                actual_value = metric_data['actual']
                
                # Find corresponding benchmark
                benchmark_key = metric_name.replace('_performance', '').replace('_', '_')
                benchmark_value = industry_benchmarks.get(benchmark_key, actual_value)
                
                if benchmark_value > 0:
                    benchmark_ratio = actual_value / benchmark_value
                    benchmark_scores.append(benchmark_ratio)
                    
                    benchmark_comparison['industry_benchmarks'][metric_name] = {
                        'our_performance': actual_value,
                        'industry_benchmark': benchmark_value,
                        'performance_vs_benchmark': benchmark_ratio,
                        'status': 'above_benchmark' if benchmark_ratio > 1 else 'below_benchmark'
                    }
        
        if benchmark_scores:
            benchmark_comparison['benchmark_score'] = sum(benchmark_scores) / len(benchmark_scores)
        
        return benchmark_comparison
    
    def _generate_performance_alerts(self, tracked_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate performance alerts"""
        alerts = []
        
        for category_data in tracked_metrics:
            category = category_data['category']
            
            for metric_name, metric_data in category_data['metrics'].items():
                status = metric_data['status']
                performance_ratio = metric_data['performance_ratio']
                
                if status == 'critical':
                    alerts.append({
                        'category': category,
                        'metric': metric_name,
                        'alert_type': 'critical',
                        'message': f"Critical performance issue: {metric_name} at {performance_ratio:.1%} of target",
                        'recommended_action': 'Immediate attention required',
                        'priority': 'high'
                    })
                elif status == 'needs_improvement':
                    alerts.append({
                        'category': category,
                        'metric': metric_name,
                        'alert_type': 'warning',
                        'message': f"Performance below target: {metric_name} at {performance_ratio:.1%} of target",
                        'recommended_action': 'Performance improvement plan needed',
                        'priority': 'medium'
                    })
        
        return alerts
    
    def _identify_improvement_opportunities(self, tracked_metrics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities"""
        opportunities = []
        
        # Analyze each category for improvement opportunities
        for category_data in tracked_metrics:
            category = category_data['category']
            metrics = category_data['metrics']
            
            # Find metrics with lowest performance ratios
            sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['performance_ratio'])
            
            for metric_name, metric_data in sorted_metrics[:2]:  # Top 2 improvement opportunities per category
                performance_ratio = metric_data['performance_ratio']
                gap = 1 - performance_ratio  # Gap to target
                
                if gap > 0.1:  # Only suggest improvements for significant gaps
                    opportunities.append({
                        'category': category,
                        'metric': metric_name,
                        'current_performance': performance_ratio,
                        'improvement_potential': gap,
                        'recommended_actions': self._generate_improvement_actions(category, metric_name),
                        'estimated_impact': self._estimate_improvement_impact(category, metric_name, gap)
                    })
        
        # Sort by improvement potential
        opportunities.sort(key=lambda x: x['improvement_potential'], reverse=True)
        return opportunities[:10]  # Top 10 opportunities
    
    def _generate_improvement_actions(self, category: str, metric_name: str) -> List[str]:
        """Generate specific improvement actions"""
        action_map = {
            'Financial': {
                'revenue_performance': ['Increase sales efforts', 'Expand market reach', 'Improve pricing strategy'],
                'gross_margin_performance': ['Optimize costs', 'Improve operational efficiency', 'Review pricing'],
                'cost_efficiency_performance': ['Process optimization', 'Automation initiatives', 'Resource optimization']
            },
            'Customer': {
                'customer_acquisition_performance': ['Increase marketing spend', 'Improve lead quality', 'Expand channels'],
                'churn_rate_performance': ['Customer success programs', 'Product improvements', 'Proactive support'],
                'net_revenue_retention_performance': ['Upselling programs', 'Cross-selling initiatives', 'Customer expansion'],
                'customer_satisfaction_performance': ['Improve support', 'Product enhancements', 'Experience optimization']
            },
            'Sales': {
                'win_rate_performance': ['Sales training', 'Better qualification', 'Competitive positioning'],
                'pipeline_coverage_performance': ['Increase prospecting', 'Improve lead generation', 'Expand pipeline'],
                'average_deal_size_performance': ['Value-based selling', 'Solution selling', 'Enterprise focus'],
                'sales_cycle_performance': ['Process optimization', 'Reduce friction', 'Decision maker access']
            },
            'Operational': {
                'operational_efficiency_performance': ['Process automation', 'Workflow optimization', 'Technology upgrades'],
                'employee_satisfaction_performance': ['Training programs', 'Work environment', 'Compensation review'],
                'automation_performance': ['Technology investments', 'Process digitization', 'AI implementation'],
                'quality_performance': ['Quality processes', 'Training', 'Continuous improvement']
            }
        }
        
        return action_map.get(category, {}).get(metric_name, ['Review and optimize process', 'Set improvement targets'])
    
    def _estimate_improvement_impact(self, category: str, metric_name: str, gap: float) -> Dict[str, Any]:
        """Estimate impact of improvement"""
        # Simplified impact estimation
        return {
            'revenue_impact': gap * 0.1 if category == 'Financial' else 0,
            'efficiency_gain': gap * 0.05 if category == 'Operational' else 0,
            'customer_impact': gap * 0.08 if category == 'Customer' else 0,
            'implementation_effort': 'Medium' if gap > 0.2 else 'Low',
            'timeframe': '3-6 months' if gap > 0.15 else '1-3 months'
        }
    
    def _calculate_performance_distribution(self, tracked_metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate performance distribution across metrics"""
        distribution = {
            'excellent': 0,
            'good': 0,
            'satisfactory': 0,
            'needs_improvement': 0,
            'critical': 0
        }
        
        for category_data in tracked_metrics:
            for metric_data in category_data['metrics'].values():
                status = metric_data['status']
                if status in distribution:
                    distribution[status] += 1
        
        return distribution
    
    # Trend calculation methods
    def _calculate_financial_trend(self, financial_metrics: Dict[str, Any]) -> str:
        """Calculate financial trend"""
        # Simplified trend calculation
        revenue = financial_metrics.get('revenue', '1000000')
        prev_revenue = financial_metrics.get('previous_revenue', '950000')
        
        revenue_change = (Decimal(str(revenue)) - Decimal(str(prev_revenue))) / Decimal(str(prev_revenue))
        
        if revenue_change > 0.05:
            return 'improving'
        elif revenue_change < -0.05:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_customer_trend(self, customer_metrics: Dict[str, Any]) -> str:
        """Calculate customer trend"""
        # Simplified - focus on satisfaction trend
        satisfaction = customer_metrics.get('satisfaction_score', '8.5')
        prev_satisfaction = customer_metrics.get('previous_satisfaction', '8.3')
        
        if Decimal(str(satisfaction)) > Decimal(str(prev_satisfaction)):
            return 'improving'
        elif Decimal(str(satisfaction)) < Decimal(str(prev_satisfaction)):
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_sales_trend(self, sales_metrics: Dict[str, Any]) -> str:
        """Calculate sales trend"""
        # Simplified - focus on win rate trend
        win_rate = sales_metrics.get('win_rate', '0.25')
        prev_win_rate = sales_metrics.get('previous_win_rate', '0.23')
        
        if Decimal(str(win_rate)) > Decimal(str(prev_win_rate)):
            return 'improving'
        elif Decimal(str(win_rate)) < Decimal(str(prev_win_rate)):
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_operational_trend(self, operational_metrics: Dict[str, Any]) -> str:
        """Calculate operational trend"""
        # Simplified - focus on efficiency trend
        efficiency = operational_metrics.get('operational_efficiency', '0.85')
        prev_efficiency = operational_metrics.get('previous_efficiency', '0.83')
        
        if Decimal(str(efficiency)) > Decimal(str(prev_efficiency)):
            return 'improving'
        elif Decimal(str(efficiency)) < Decimal(str(prev_efficiency)):
            return 'declining'
        else:
            return 'stable'
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """Get performance data formatted for dashboard"""
        # Generate sample performance data
        sample_metrics = {
            'financial': {
                'revenue': '1100000',
                'revenue_target': '1000000',
                'gross_margin': '0.77',
                'margin_target': '0.75',
                'cost_efficiency': '0.87',
                'efficiency_target': '0.85'
            },
            'customer': {
                'new_customers': '62',
                'acquisition_target': '50',
                'churn_rate': '0.04',
                'churn_target': '0.05',
                'net_revenue_retention': '1.15',
                'nrr_target': '1.10',
                'satisfaction_score': '8.7',
                'satisfaction_target': '8.5'
            },
            'sales': {
                'win_rate': '0.28',
                'win_rate_target': '0.25',
                'pipeline_coverage': '3.2',
                'coverage_target': '3.0',
                'average_deal_size': '27000',
                'deal_size_target': '25000',
                'sales_cycle_days': '42',
                'cycle_target': '45'
            },
            'operational': {
                'operational_efficiency': '0.88',
                'efficiency_target': '0.85',
                'employee_satisfaction': '7.9',
                'emp_sat_target': '8.0',
                'automation_level': '0.65',
                'automation_target': '0.60',
                'quality_score': '0.92',
                'quality_target': '0.90'
            }
        }
        
        return self.track_performance_metrics(sample_metrics)