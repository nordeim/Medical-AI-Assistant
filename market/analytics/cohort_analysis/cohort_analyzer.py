"""
Cohort Analysis for Customer Lifetime Value and Retention
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from decimal import Decimal
import logging

class CohortAnalyzer:
    """Customer cohort analysis and retention tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cohort analysis parameters
        self.analysis_period_months = config.get('analysis_period_months', 12)
        self.min_cohort_size = config.get('min_cohort_size', 10)
        self.retention_thresholds = config.get('retention_thresholds', {
            'excellent': 0.8,
            'good': 0.65,
            'average': 0.5,
            'poor': 0.35
        })
    
    def analyze_cohorts(self, cohorts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive cohort analysis"""
        self.logger.info(f"Starting cohort analysis for {len(cohorts_data)} cohorts")
        
        # Calculate retention rates
        retention_analysis = self._calculate_retention_rates(cohorts_data)
        
        # Calculate revenue retention
        revenue_analysis = self._calculate_revenue_retention(cohorts_data)
        
        # Calculate customer lifetime value by cohort
        ltv_analysis = self._calculate_cohort_ltv(cohorts_data)
        
        # Segment cohorts by performance
        cohort_performance = self._segment_cohorts(cohorts_data, retention_analysis)
        
        # Identify trends and patterns
        trend_analysis = self._analyze_cohort_trends(retention_analysis, revenue_analysis)
        
        # Generate insights and recommendations
        insights = self._generate_cohort_insights(
            retention_analysis, revenue_analysis, cohort_performance
        )
        
        return {
            'retention_analysis': retention_analysis,
            'revenue_analysis': revenue_analysis,
            'ltv_analysis': ltv_analysis,
            'cohort_performance': cohort_performance,
            'trend_analysis': trend_analysis,
            'insights': insights,
            'summary': self._generate_cohort_summary(retention_analysis, revenue_analysis)
        }
    
    def _calculate_retention_rates(self, cohorts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate customer retention rates by cohort"""
        retention_data = {}
        
        for cohort in cohorts_data:
            cohort_month = cohort.get('cohort_month')
            customers = cohort.get('customers', [])
            cohort_size = len(customers)
            
            if cohort_size < self.min_cohort_size:
                continue
            
            retention_rates = {}
            
            # Calculate retention for each month
            for month_offset in range(self.analysis_period_months + 1):
                active_customers = 0
                total_customers_in_cohort = 0
                
                for customer in customers:
                    # Check if customer is still active at month_offset
                    acquisition_date = customer.get('acquisition_date')
                    if isinstance(acquisition_date, str):
                        acquisition_date = datetime.strptime(acquisition_date, '%Y-%m-%d').date()
                    
                    customer_age = self._calculate_months_since_acquisition(acquisition_date, date.today())
                    
                    if customer_age >= month_offset and customer.get('status') == 'Active':
                        active_customers += 1
                    
                    total_customers_in_cohort += 1
                
                retention_rate = active_customers / max(total_customers_in_cohort, 1)
                retention_rates[month_offset] = retention_rate
            
            retention_data[cohort_month] = {
                'cohort_size': cohort_size,
                'retention_rates': retention_rates,
                'avg_retention_12m': retention_rates.get(12, 0),
                'retention_health': self._assess_retention_health(retention_rates.get(12, 0))
            }
        
        return {
            'cohorts': retention_data,
            'overall_metrics': self._calculate_overall_retention_metrics(retention_data)
        }
    
    def _calculate_revenue_retention(self, cohorts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate revenue retention by cohort"""
        revenue_data = {}
        
        for cohort in cohorts_data:
            cohort_month = cohort.get('cohort_month')
            customers = cohort.get('customers', [])
            cohort_size = len(customers)
            
            if cohort_size < self.min_cohort_size:
                continue
            
            monthly_revenue = {}
            
            # Calculate revenue for each month
            for month_offset in range(self.analysis_period_months + 1):
                cohort_revenue = Decimal('0')
                active_customers = 0
                
                for customer in customers:
                    acquisition_date = customer.get('acquisition_date')
                    if isinstance(acquisition_date, str):
                        acquisition_date = datetime.strptime(acquisition_date, '%Y-%m-%d').date()
                    
                    customer_age = self._calculate_months_since_acquisition(acquisition_date, date.today())
                    
                    if customer_age >= month_offset and customer.get('status') == 'Active':
                        monthly_mrr = Decimal(str(customer.get('monthly_recurring_revenue', '0')))
                        cohort_revenue += monthly_mrr
                        active_customers += 1
                
                monthly_revenue[month_offset] = {
                    'total_revenue': cohort_revenue,
                    'active_customers': active_customers,
                    'avg_revenue_per_customer': cohort_revenue / max(active_customers, 1)
                }
            
            # Calculate revenue retention (relative to month 0)
            base_revenue = monthly_revenue.get(0, {}).get('total_revenue', Decimal('1'))
            revenue_retention = {}
            
            for month, data in monthly_revenue.items():
                if base_revenue > 0:
                    retention_rate = float(data['total_revenue'] / base_revenue)
                else:
                    retention_rate = 0.0
                revenue_retention[month] = retention_rate
            
            revenue_data[cohort_month] = {
                'monthly_revenue': monthly_revenue,
                'revenue_retention': revenue_retention,
                'revenue_retention_12m': revenue_retention.get(12, 0)
            }
        
        return {
            'cohorts': revenue_data,
            'overall_metrics': self._calculate_overall_revenue_metrics(revenue_data)
        }
    
    def _calculate_cohort_ltv(self, cohorts_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate LTV metrics by cohort"""
        ltv_data = {}
        
        for cohort in cohorts_data:
            cohort_month = cohort.get('cohort_month')
            customers = cohort.get('customers', [])
            cohort_size = len(customers)
            
            if cohort_size < self.min_cohort_size:
                continue
            
            # Calculate individual customer LTVs
            customer_ltvs = []
            total_acquisition_costs = Decimal('0')
            total_monthly_revenue = Decimal('0')
            
            for customer in customers:
                # Calculate customer LTV
                monthly_revenue = Decimal(str(customer.get('monthly_recurring_revenue', '0')))
                acquisition_cost = Decimal(str(customer.get('acquisition_cost', '0')))
                
                # Simplified LTV calculation: MRR * 36 months * 75% margin - CAC
                gross_margin = Decimal('0.75')
                avg_lifespan_months = Decimal('36')
                ltv = (monthly_revenue * gross_margin * avg_lifespan_months) - acquisition_cost
                ltv = max(ltv, Decimal('0'))
                
                customer_ltvs.append(ltv)
                total_acquisition_costs += acquisition_cost
                total_monthly_revenue += monthly_revenue
            
            # Cohort metrics
            avg_ltv = sum(customer_ltvs) / max(len(customer_ltvs), 1)
            median_ltv = self._calculate_median(customer_ltvs)
            total_cohort_ltv = sum(customer_ltvs)
            avg_cac = total_acquisition_costs / max(cohort_size, 1)
            ltv_cac_ratio = float(avg_ltv / max(avg_cac, 1))
            
            ltv_data[cohort_month] = {
                'cohort_size': cohort_size,
                'avg_ltv': avg_ltv,
                'median_ltv': median_ltv,
                'total_cohort_ltv': total_cohort_ltv,
                'avg_cac': avg_cac,
                'ltv_cac_ratio': ltv_cac_ratio,
                'ltv_health': self._assess_ltv_health(ltv_cac_ratio)
            }
        
        return {
            'cohorts': ltv_data,
            'overall_metrics': self._calculate_overall_ltv_metrics(ltv_data)
        }
    
    def _segment_cohorts(self, cohorts_data: List[Dict[str, Any]], retention_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Segment cohorts by performance"""
        segments = {
            'high_performers': [],
            'good_performers': [],
            'average_performers': [],
            'poor_performers': []
        }
        
        for cohort_month, retention_info in retention_analysis.get('cohorts', {}).items():
            retention_rate_12m = retention_info.get('avg_retention_12m', 0)
            
            if retention_rate_12m >= self.retention_thresholds['excellent']:
                segments['high_performers'].append(cohort_month)
            elif retention_rate_12m >= self.retention_thresholds['good']:
                segments['good_performers'].append(cohort_month)
            elif retention_rate_12m >= self.retention_thresholds['average']:
                segments['average_performers'].append(cohort_month)
            else:
                segments['poor_performers'].append(cohort_month)
        
        return segments
    
    def _analyze_cohort_trends(self, retention_analysis: Dict[str, Any], revenue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends across cohorts"""
        # Get retention rates for trend analysis
        cohort_months = sorted(retention_analysis.get('cohorts', {}).keys())
        
        if len(cohort_months) < 3:
            return {'trend': 'insufficient_data', 'confidence': 0}
        
        # Calculate trend in 12-month retention
        retention_trend = []
        for month in cohort_months:
            retention_data = retention_analysis['cohorts'][month]
            retention_trend.append(retention_data.get('avg_retention_12m', 0))
        
        # Calculate trend slope
        if len(retention_trend) >= 3:
            recent_avg = sum(retention_trend[-3:]) / 3
            earlier_avg = sum(retention_trend[:-3]) / max(len(retention_trend) - 3, 1)
            trend_change = recent_avg - earlier_avg
            
            if trend_change > 0.05:
                trend_direction = 'improving'
                trend_confidence = 0.8
            elif trend_change < -0.05:
                trend_direction = 'declining'
                trend_confidence = 0.8
            else:
                trend_direction = 'stable'
                trend_confidence = 0.6
        else:
            trend_direction = 'stable'
            trend_confidence = 0.3
        
        return {
            'direction': trend_direction,
            'confidence': trend_confidence,
            'change_rate': trend_change if 'trend_change' in locals() else 0,
            'cohort_count': len(cohort_months)
        }
    
    def _generate_cohort_insights(self, retention_analysis: Dict[str, Any], 
                                revenue_analysis: Dict[str, Any], 
                                cohort_performance: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Generate actionable insights from cohort analysis"""
        insights = []
        
        # Overall retention insights
        overall_retention = retention_analysis.get('overall_metrics', {})
        avg_retention_12m = overall_retention.get('avg_retention_12m', 0)
        
        if avg_retention_12m >= self.retention_thresholds['excellent']:
            insights.append({
                'type': 'positive',
                'category': 'Customer Retention',
                'insight': f'Excellent customer retention at {avg_retention_12m:.1%}',
                'recommendation': 'Maintain current customer success strategies',
                'priority': 'low'
            })
        elif avg_retention_12m < self.retention_thresholds['poor']:
            insights.append({
                'type': 'warning',
                'category': 'Customer Retention',
                'insight': f'Poor customer retention at {avg_retention_12m:.1%}',
                'recommendation': 'Implement immediate customer retention initiatives',
                'priority': 'high'
            })
        
        # Cohort performance insights
        poor_cohorts = cohort_performance.get('poor_performers', [])
        if poor_cohorts:
            insights.append({
                'type': 'warning',
                'category': 'Cohort Performance',
                'insight': f'{len(poor_cohorts)} cohorts showing poor retention',
                'recommendation': 'Analyze acquisition channels and customer segments for poor-performing cohorts',
                'priority': 'high'
            })
        
        # Revenue retention insights
        revenue_metrics = revenue_analysis.get('overall_metrics', {})
        avg_revenue_retention = revenue_metrics.get('avg_revenue_retention_12m', 0)
        
        if avg_revenue_retention < 0.8:
            insights.append({
                'type': 'warning',
                'category': 'Revenue Retention',
                'insight': f'Revenue retention below target at {avg_revenue_retention:.1%}',
                'recommendation': 'Focus on customer expansion and upselling programs',
                'priority': 'medium'
            })
        
        return insights
    
    def _generate_cohort_summary(self, retention_analysis: Dict[str, Any], 
                               revenue_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of cohort analysis"""
        retention_metrics = retention_analysis.get('overall_metrics', {})
        revenue_metrics = revenue_analysis.get('overall_metrics', {})
        
        return {
            'total_cohorts_analyzed': len(retention_analysis.get('cohorts', {})),
            'average_retention_12m': retention_metrics.get('avg_retention_12m', 0),
            'average_revenue_retention_12m': revenue_metrics.get('avg_revenue_retention_12m', 0),
            'overall_health_score': self._calculate_overall_health_score(retention_metrics, revenue_metrics),
            'key_metrics': {
                'retention_trend': retention_metrics.get('trend_direction', 'stable'),
                'revenue_trend': revenue_metrics.get('trend_direction', 'stable'),
                'top_performing_cohort': self._get_top_performing_cohort(retention_analysis),
                'improvement_areas': self._identify_improvement_areas(retention_analysis)
            }
        }
    
    # Helper methods
    def _calculate_months_since_acquisition(self, acquisition_date: date, reference_date: date) -> int:
        """Calculate months since acquisition"""
        months = (reference_date.year - acquisition_date.year) * 12 + (reference_date.month - acquisition_date.month)
        return max(0, months)
    
    def _assess_retention_health(self, retention_rate_12m: float) -> str:
        """Assess retention health based on 12-month retention"""
        if retention_rate_12m >= self.retention_thresholds['excellent']:
            return 'excellent'
        elif retention_rate_12m >= self.retention_thresholds['good']:
            return 'good'
        elif retention_rate_12m >= self.retention_thresholds['average']:
            return 'average'
        else:
            return 'poor'
    
    def _assess_ltv_health(self, ltv_cac_ratio: float) -> str:
        """Assess LTV health based on LTV:CAC ratio"""
        if ltv_cac_ratio >= 3.0:
            return 'excellent'
        elif ltv_cac_ratio >= 2.0:
            return 'good'
        elif ltv_cac_ratio >= 1.5:
            return 'average'
        else:
            return 'poor'
    
    def _calculate_median(self, values: List[Decimal]) -> Decimal:
        """Calculate median value"""
        if not values:
            return Decimal('0')
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    def _calculate_overall_retention_metrics(self, retention_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall retention metrics"""
        if not retention_data:
            return {}
        
        retention_rates_12m = [cohort_data.get('avg_retention_12m', 0) for cohort_data in retention_data.values()]
        
        if retention_rates_12m:
            avg_retention = sum(retention_rates_12m) / len(retention_rates_12m)
            return {
                'avg_retention_12m': avg_retention,
                'best_cohort_retention': max(retention_rates_12m),
                'worst_cohort_retention': min(retention_rates_12m),
                'retention_variance': max(retention_rates_12m) - min(retention_rates_12m)
            }
        
        return {}
    
    def _calculate_overall_revenue_metrics(self, revenue_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall revenue metrics"""
        if not revenue_data:
            return {}
        
        revenue_retention_rates = [cohort_data.get('revenue_retention_12m', 0) for cohort_data in revenue_data.values()]
        
        if revenue_retention_rates:
            avg_revenue_retention = sum(revenue_retention_rates) / len(revenue_retention_rates)
            return {
                'avg_revenue_retention_12m': avg_revenue_retention,
                'best_cohort_revenue_retention': max(revenue_retention_rates),
                'worst_cohort_revenue_retention': min(revenue_retention_rates)
            }
        
        return {}
    
    def _calculate_overall_ltv_metrics(self, ltv_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall LTV metrics"""
        if not ltv_data:
            return {}
        
        ltv_ratios = [cohort_data.get('ltv_cac_ratio', 0) for cohort_data in ltv_data.values()]
        
        if ltv_ratios:
            avg_ltv_ratio = sum(ltv_ratios) / len(ltv_ratios)
            return {
                'avg_ltv_cac_ratio': avg_ltv_ratio,
                'best_cohort_ltv_ratio': max(ltv_ratios),
                'worst_cohort_ltv_ratio': min(ltv_ratios)
            }
        
        return {}
    
    def _calculate_overall_health_score(self, retention_metrics: Dict[str, Any], revenue_metrics: Dict[str, Any]) -> float:
        """Calculate overall cohort health score"""
        score = 0
        
        # Retention score (50% weight)
        avg_retention = retention_metrics.get('avg_retention_12m', 0)
        score += avg_retention * 50
        
        # Revenue retention score (30% weight)
        avg_revenue_retention = revenue_metrics.get('avg_revenue_retention_12m', 0)
        score += avg_revenue_retention * 30
        
        # Consistency score (20% weight) - lower variance is better
        retention_variance = retention_metrics.get('retention_variance', 1)
        consistency_score = max(0, 1 - retention_variance) * 20
        score += consistency_score
        
        return min(score, 100)
    
    def _get_top_performing_cohort(self, retention_analysis: Dict[str, Any]) -> Optional[str]:
        """Get top performing cohort by retention"""
        cohorts = retention_analysis.get('cohorts', {})
        if not cohorts:
            return None
        
        best_cohort = max(cohorts.keys(), key=lambda k: cohorts[k].get('avg_retention_12m', 0))
        return best_cohort
    
    def _identify_improvement_areas(self, retention_analysis: Dict[str, Any]) -> List[str]:
        """Identify key improvement areas"""
        improvement_areas = []
        
        cohorts = retention_analysis.get('cohorts', {})
        if not cohorts:
            return improvement_areas
        
        retention_rates = [cohort_data.get('avg_retention_12m', 0) for cohort_data in cohorts.values()]
        avg_retention = sum(retention_rates) / len(retention_rates)
        
        if avg_retention < self.retention_thresholds['good']:
            improvement_areas.append('Customer Retention Programs')
        
        # Check for high variance
        if max(retention_rates) - min(retention_rates) > 0.3:
            improvement_areas.append('Cohort Consistency')
        
        return improvement_areas