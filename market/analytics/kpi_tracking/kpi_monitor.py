"""
KPI Monitoring and Alert System
Real-time KPI tracking and alerting for business intelligence
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Callable
from decimal import Decimal
import logging
from dataclasses import asdict

# Import KPI models
from ..data_models.kpi_models import KPIMetrics, KPIType, KPIStatus

class KPIMonitor:
    """KPI monitoring and tracking system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # KPI definitions and thresholds
        self.kpi_definitions = self._load_kpi_definitions()
        self.kpi_cache = {}
        self.alert_handlers = []
        
        # Monitoring configuration
        self.refresh_interval = config.get('refresh_interval_minutes', 60)
        self.alert_cooldown_minutes = config.get('alert_cooldown_minutes', 60)
        self.last_alert_times = {}
    
    def _load_kpi_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load KPI definitions with thresholds"""
        return {
            # Financial KPIs
            'revenue': {
                'name': 'Total Revenue',
                'type': KPIType.FINANCIAL,
                'unit': 'USD',
                'target_value': Decimal('1000000'),
                'excellent_threshold': Decimal('1200000'),
                'good_threshold': Decimal('1000000'),
                'target_threshold': Decimal('900000'),
                'below_target_threshold': Decimal('700000'),
                'alert_enabled': True,
                'calculation_method': 'sum'
            },
            'gross_margin': {
                'name': 'Gross Margin',
                'type': KPIType.FINANCIAL,
                'unit': 'percentage',
                'target_value': Decimal('0.75'),
                'excellent_threshold': Decimal('0.80'),
                'good_threshold': Decimal('0.75'),
                'target_threshold': Decimal('0.70'),
                'below_target_threshold': Decimal('0.60'),
                'alert_enabled': True,
                'calculation_method': 'ratio'
            },
            
            # Customer KPIs
            'customer_acquisition': {
                'name': 'New Customers',
                'type': KPIType.CUSTOMER,
                'unit': 'count',
                'target_value': Decimal('50'),
                'excellent_threshold': Decimal('75'),
                'good_threshold': Decimal('50'),
                'target_threshold': Decimal('40'),
                'below_target_threshold': Decimal('25'),
                'alert_enabled': True,
                'calculation_method': 'count'
            },
            'churn_rate': {
                'name': 'Customer Churn Rate',
                'type': KPIType.CUSTOMER,
                'unit': 'percentage',
                'target_value': Decimal('0.05'),
                'excellent_threshold': Decimal('0.03'),
                'good_threshold': Decimal('0.05'),
                'target_threshold': Decimal('0.08'),
                'below_target_threshold': Decimal('0.12'),
                'alert_enabled': True,
                'calculation_method': 'rate',
                'reverse_indicator': True  # Lower is better
            },
            'net_revenue_retention': {
                'name': 'Net Revenue Retention',
                'type': KPIType.CUSTOMER,
                'unit': 'percentage',
                'target_value': Decimal('1.10'),
                'excellent_threshold': Decimal('1.20'),
                'good_threshold': Decimal('1.10'),
                'target_threshold': Decimal('1.05'),
                'below_target_threshold': Decimal('0.95'),
                'alert_enabled': True,
                'calculation_method': 'ratio'
            },
            
            # Sales KPIs
            'win_rate': {
                'name': 'Sales Win Rate',
                'type': KPIType.SALES,
                'unit': 'percentage',
                'target_value': Decimal('0.25'),
                'excellent_threshold': Decimal('0.35'),
                'good_threshold': Decimal('0.25'),
                'target_threshold': Decimal('0.20'),
                'below_target_threshold': Decimal('0.15'),
                'alert_enabled': True,
                'calculation_method': 'rate'
            },
            'pipeline_coverage': {
                'name': 'Pipeline Coverage Ratio',
                'type': KPIType.SALES,
                'unit': 'ratio',
                'target_value': Decimal('3.0'),
                'excellent_threshold': Decimal('4.0'),
                'good_threshold': Decimal('3.0'),
                'target_threshold': Decimal('2.5'),
                'below_target_threshold': Decimal('1.5'),
                'alert_enabled': True,
                'calculation_method': 'ratio'
            },
            'average_deal_size': {
                'name': 'Average Deal Size',
                'type': KPIType.SALES,
                'unit': 'USD',
                'target_value': Decimal('25000'),
                'excellent_threshold': Decimal('35000'),
                'good_threshold': Decimal('25000'),
                'target_threshold': Decimal('20000'),
                'below_target_threshold': Decimal('15000'),
                'alert_enabled': False,  # Less critical
                'calculation_method': 'average'
            },
            
            # Marketing KPIs
            'cost_per_acquisition': {
                'name': 'Cost Per Acquisition',
                'type': KPIType.MARKETING,
                'unit': 'USD',
                'target_value': Decimal('500'),
                'excellent_threshold': Decimal('400'),
                'good_threshold': Decimal('500'),
                'target_threshold': Decimal('600'),
                'below_target_threshold': Decimal('800'),
                'alert_enabled': True,
                'calculation_method': 'cost_ratio',
                'reverse_indicator': True  # Lower is better
            },
            'conversion_rate': {
                'name': 'Lead Conversion Rate',
                'type': KPIType.MARKETING,
                'unit': 'percentage',
                'target_value': Decimal('0.05'),
                'excellent_threshold': Decimal('0.08'),
                'good_threshold': Decimal('0.05'),
                'target_threshold': Decimal('0.03'),
                'below_target_threshold': Decimal('0.02'),
                'alert_enabled': True,
                'calculation_method': 'rate'
            },
            
            # Operational KPIs
            'customer_satisfaction': {
                'name': 'Customer Satisfaction Score',
                'type': KPIType.OPERATIONAL,
                'unit': 'score',
                'target_value': Decimal('8.5'),
                'excellent_threshold': Decimal('9.0'),
                'good_threshold': Decimal('8.5'),
                'target_threshold': Decimal('8.0'),
                'below_target_threshold': Decimal('7.5'),
                'alert_enabled': True,
                'calculation_method': 'average'
            },
            'employee_satisfaction': {
                'name': 'Employee Satisfaction Score',
                'type': KPIType.OPERATIONAL,
                'unit': 'score',
                'target_value': Decimal('8.0'),
                'excellent_threshold': Decimal('8.5'),
                'good_threshold': Decimal('8.0'),
                'target_threshold': Decimal('7.5'),
                'below_target_threshold': Decimal('7.0'),
                'alert_enabled': False,  # Less frequent monitoring
                'calculation_method': 'average'
            },
            'operational_efficiency': {
                'name': 'Operational Efficiency',
                'type': KPIType.OPERATIONAL,
                'unit': 'percentage',
                'target_value': Decimal('0.85'),
                'excellent_threshold': Decimal('0.90'),
                'good_threshold': Decimal('0.85'),
                'target_threshold': Decimal('0.80'),
                'below_target_threshold': Decimal('0.75'),
                'alert_enabled': True,
                'calculation_method': 'ratio'
            },
            
            # Market KPIs
            'market_share': {
                'name': 'Market Share',
                'type': KPIType.EXECUTIVE,
                'unit': 'percentage',
                'target_value': Decimal('0.15'),
                'excellent_threshold': Decimal('0.25'),
                'good_threshold': Decimal('0.15'),
                'target_threshold': Decimal('0.10'),
                'below_target_threshold': Decimal('0.05'),
                'alert_enabled': False,  # Quarterly review
                'calculation_method': 'percentage'
            },
            
            # Executive KPIs
            'revenue_per_employee': {
                'name': 'Revenue Per Employee',
                'type': KPIType.EXECUTIVE,
                'unit': 'USD',
                'target_value': Decimal('100000'),
                'excellent_threshold': Decimal('150000'),
                'good_threshold': Decimal('100000'),
                'target_threshold': Decimal('80000'),
                'below_target_threshold': Decimal('60000'),
                'alert_enabled': False,  # Monthly review
                'calculation_method': 'ratio'
            }
        }
    
    def get_current_kpis(self, date_range: Optional[tuple[date, date]] = None) -> List[KPIMetrics]:
        """Get current KPI values with calculations"""
        self.logger.info("Calculating current KPI values")
        
        if date_range is None:
            end_date = date.today()
            start_date = end_date - timedelta(days=30)  # Last 30 days
            date_range = (start_date, end_date)
        
        kpi_metrics = []
        
        for kpi_id, definition in self.kpi_definitions.items():
            try:
                # Calculate KPI value
                current_value = self._calculate_kpi_value(kpi_id, definition, date_range)
                
                # Get previous value for comparison
                prev_date_range = (date_range[0] - timedelta(days=30), date_range[0])
                previous_value = self._calculate_kpi_value(kpi_id, definition, prev_date_range)
                
                # Determine status
                status = self._determine_kpi_status(current_value, definition)
                variance_from_target = self._calculate_variance(current_value, definition['target_value'])
                variance_from_previous = self._calculate_variance(current_value, previous_value) if previous_value > 0 else 0
                
                # Determine trend
                trend_direction = self._determine_trend_direction(current_value, previous_value, definition)
                
                # Create KPI metric
                kpi_metric = KPIMetrics(
                    kpi_id=kpi_id,
                    kpi_name=definition['name'],
                    kpi_type=definition['type'],
                    description=f"Business KPI for {definition['name']}",
                    unit_of_measure=definition['unit'],
                    target_value=definition['target_value'],
                    current_value=current_value,
                    previous_value=previous_value,
                    period_start=date_range[0],
                    period_end=date_range[1],
                    performance_status=status,
                    variance_from_target=variance_from_target,
                    variance_from_previous=variance_from_previous,
                    excellent_threshold=definition['excellent_threshold'],
                    good_threshold=definition['good_threshold'],
                    target_threshold=definition['target_threshold'],
                    below_target_threshold=definition['below_target_threshold'],
                    trend_direction=trend_direction,
                    trend_consistency=0.8,  # Simplified
                    months_of_data=3,  # Simplified
                    industry_benchmark=None,
                    benchmark_percentile=None,
                    alert_enabled=definition['alert_enabled'],
                    alert_thresholds=self._create_alert_thresholds(definition),
                    last_alert_date=self.last_alert_times.get(kpi_id),
                    action_required=self._get_action_required(status, variance_from_target)
                )
                
                kpi_metrics.append(kpi_metric)
                
            except Exception as e:
                self.logger.error(f"Error calculating KPI {kpi_id}: {e}")
                continue
        
        # Cache the results
        cache_key = f"kpis_{date_range[0]}_{date_range[1]}"
        self.kpi_cache[cache_key] = kpi_metrics
        
        self.logger.info(f"Calculated {len(kpi_metrics)} KPIs")
        return kpi_metrics
    
    def _calculate_kpi_value(self, kpi_id: str, definition: Dict[str, Any], 
                           date_range: tuple[date, date]) -> Decimal:
        """Calculate specific KPI value"""
        method = definition.get('calculation_method', 'sum')
        
        # Mock data sources - in reality, these would query actual data systems
        if method == 'sum':
            return self._mock_sum_calculation(kpi_id, date_range)
        elif method == 'average':
            return self._mock_average_calculation(kpi_id, date_range)
        elif method == 'count':
            return self._mock_count_calculation(kpi_id, date_range)
        elif method == 'ratio':
            return self._mock_ratio_calculation(kpi_id, date_range)
        elif method == 'rate':
            return self._mock_rate_calculation(kpi_id, date_range)
        elif method == 'cost_ratio':
            return self._mock_cost_ratio_calculation(kpi_id, date_range)
        elif method == 'percentage':
            return self._mock_percentage_calculation(kpi_id, date_range)
        else:
            return Decimal('0')
    
    def _determine_kpi_status(self, current_value: Decimal, definition: Dict[str, Any]) -> KPIStatus:
        """Determine KPI status based on thresholds"""
        reverse_indicator = definition.get('reverse_indicator', False)
        
        if reverse_indicator:
            # For metrics where lower is better (e.g., churn rate, CAC)
            if current_value <= definition['excellent_threshold']:
                return KPIStatus.EXCELLENT
            elif current_value <= definition['good_threshold']:
                return KPIStatus.GOOD
            elif current_value <= definition['target_threshold']:
                return KPIStatus.TARGET
            elif current_value <= definition['below_target_threshold']:
                return KPIStatus.BELOW_TARGET
            else:
                return KPIStatus.CRITICAL
        else:
            # For metrics where higher is better
            if current_value >= definition['excellent_threshold']:
                return KPIStatus.EXCELLENT
            elif current_value >= definition['good_threshold']:
                return KPIStatus.GOOD
            elif current_value >= definition['target_threshold']:
                return KPIStatus.TARGET
            elif current_value >= definition['below_target_threshold']:
                return KPIStatus.BELOW_TARGET
            else:
                return KPIStatus.CRITICAL
    
    def _calculate_variance(self, current_value: Decimal, target_value: Decimal) -> float:
        """Calculate percentage variance from target"""
        if target_value <= 0:
            return 0.0
        return float((current_value - target_value) / target_value)
    
    def _determine_trend_direction(self, current_value: Decimal, previous_value: Decimal, 
                                 definition: Dict[str, Any]) -> str:
        """Determine trend direction"""
        if previous_value <= 0:
            return 'stable'
        
        variance = self._calculate_variance(current_value, previous_value)
        threshold = 0.02  # 2% threshold for significance
        
        if variance > threshold:
            return 'improving'
        elif variance < -threshold:
            return 'declining'
        else:
            return 'stable'
    
    def _create_alert_thresholds(self, definition: Dict[str, Any]) -> Dict[str, Decimal]:
        """Create alert thresholds for KPI"""
        return {
            'critical': definition['below_target_threshold'],
            'warning': definition['target_threshold'],
            'info': definition['good_threshold']
        }
    
    def _get_action_required(self, status: KPIStatus, variance: float) -> List[Dict[str, Any]]:
        """Get required actions based on KPI status"""
        actions = []
        
        if status == KPIStatus.CRITICAL:
            actions.append({
                'priority': 'Critical',
                'action': 'Immediate attention required',
                'timeline': 'Within 24 hours'
            })
        elif status == KPIStatus.BELOW_TARGET:
            actions.append({
                'priority': 'High',
                'action': 'Performance improvement needed',
                'timeline': 'Within 1 week'
            })
        elif status == KPIStatus.TARGET:
            actions.append({
                'priority': 'Medium',
                'action': 'Monitor closely',
                'timeline': 'Within 2 weeks'
            })
        
        return actions
    
    # Mock calculation methods for demo
    def _mock_sum_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock sum calculation"""
        if kpi_id == 'revenue':
            return Decimal('1050000')  # Slightly above target
        return Decimal('0')
    
    def _mock_average_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock average calculation"""
        if kpi_id == 'customer_satisfaction':
            return Decimal('8.7')
        elif kpi_id == 'employee_satisfaction':
            return Decimal('7.9')
        return Decimal('0')
    
    def _mock_count_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock count calculation"""
        if kpi_id == 'customer_acquisition':
            return Decimal('62')  # Above target
        return Decimal('0')
    
    def _mock_ratio_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock ratio calculation"""
        if kpi_id == 'gross_margin':
            return Decimal('0.77')  # 77% margin
        elif kpi_id == 'net_revenue_retention':
            return Decimal('1.15')  # 115% NRR
        elif kpi_id == 'pipeline_coverage':
            return Decimal('3.2')  # 3.2x coverage
        return Decimal('1')
    
    def _mock_rate_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock rate calculation"""
        if kpi_id == 'churn_rate':
            return Decimal('0.04')  # 4% churn
        elif kpi_id == 'win_rate':
            return Decimal('0.28')  # 28% win rate
        elif kpi_id == 'conversion_rate':
            return Decimal('0.06')  # 6% conversion
        return Decimal('0.05')
    
    def _mock_cost_ratio_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock cost ratio calculation"""
        if kpi_id == 'cost_per_acquisition':
            return Decimal('480')  # $480 CAC
        return Decimal('500')
    
    def _mock_percentage_calculation(self, kpi_id: str, date_range: tuple[date, date]) -> Decimal:
        """Mock percentage calculation"""
        if kpi_id == 'market_share':
            return Decimal('0.16')  # 16% market share
        elif kpi_id == 'operational_efficiency':
            return Decimal('0.87')  # 87% efficiency
        return Decimal('0.15')
    
    def get_kpi_dashboard_data(self, kpi_types: Optional[List[KPIType]] = None) -> Dict[str, Any]:
        """Get data formatted for KPI dashboard"""
        current_kpis = self.get_current_kpis()
        
        # Filter by types if specified
        if kpi_types:
            current_kpis = [kpi for kpi in current_kpis if kpi.kpi_type in kpi_types]
        
        # Group KPIs by type
        kpis_by_type = {}
        for kpi in current_kpis:
            kpi_type = kpi.kpi_type.value
            if kpi_type not in kpis_by_type:
                kpis_by_type[kpi_type] = []
            kpis_by_type[kpi_type].append(kpi)
        
        # Calculate summary metrics
        summary = self._calculate_dashboard_summary(current_kpis)
        
        # Get status distribution
        status_distribution = self._calculate_status_distribution(current_kpis)
        
        # Get trending KPIs
        trending_kpis = self._get_trending_kpis(current_kpis)
        
        return {
            'kpis_by_type': {k: [asdict(kpi) for kpi in v] for k, v in kpis_by_type.items()},
            'summary': summary,
            'status_distribution': status_distribution,
            'trending_kpis': trending_kpis,
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_dashboard_summary(self, kpis: List[KPIMetrics]) -> Dict[str, Any]:
        """Calculate dashboard summary metrics"""
        total_kpis = len(kpis)
        excellent_count = len([kpi for kpi in kpis if kpi.performance_status == KPIStatus.EXCELLENT])
        good_count = len([kpi for kpi in kpis if kpi.performance_status == KPIStatus.GOOD])
        critical_count = len([kpi for kpi in kpis if kpi.performance_status == KPIStatus.CRITICAL])
        
        avg_performance_score = sum(kpi.calculate_performance_score() for kpi in kpis) / total_kpis if total_kpis > 0 else 0
        
        return {
            'total_kpis': total_kpis,
            'excellent_kpis': excellent_count,
            'good_kpis': good_count,
            'critical_kpis': critical_count,
            'average_performance_score': avg_performance_score,
            'health_percentage': (excellent_count + good_count) / total_kpis * 100 if total_kpis > 0 else 0
        }
    
    def _calculate_status_distribution(self, kpis: List[KPIMetrics]) -> Dict[str, int]:
        """Calculate KPI status distribution"""
        distribution = {}
        
        for status in KPIStatus:
            distribution[status.value] = len([kpi for kpi in kpis if kpi.performance_status == status])
        
        return distribution
    
    def _get_trending_kpis(self, kpis: List[KPIMetrics]) -> List[Dict[str, Any]]:
        """Get KPIs with notable trends"""
        trending = []
        
        for kpi in kpis:
            if kpi.trend_direction == 'improving':
                trending.append({
                    'kpi_id': kpi.kpi_id,
                    'kpi_name': kpi.kpi_name,
                    'trend': kpi.trend_direction,
                    'variance': kpi.variance_from_target,
                    'status': kpi.performance_status.value
                })
        
        # Sort by performance score
        trending.sort(key=lambda x: x['variance'], reverse=True)
        return trending[:5]  # Top 5 trending
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check for KPI alerts"""
        alerts = []
        current_time = datetime.now()
        
        for kpi in self.get_current_kpis():
            if not kpi.alert_enabled:
                continue
            
            # Check cooldown period
            if kpi.kpi_id in self.last_alert_times:
                time_since_last_alert = (current_time - self.last_alert_times[kpi.kpi_id]).total_seconds() / 60
                if time_since_last_alert < self.alert_cooldown_minutes:
                    continue
            
            # Generate alert based on status
            if kpi.performance_status == KPIStatus.CRITICAL:
                alerts.append({
                    'kpi_id': kpi.kpi_id,
                    'kpi_name': kpi.kpi_name,
                    'alert_type': 'critical',
                    'message': f'Critical: {kpi.kpi_name} is at {kpi.current_value} vs target {kpi.target_value}',
                    'severity': 'critical',
                    'timestamp': current_time.isoformat(),
                    'actions': kpi.action_required
                })
                self.last_alert_times[kpi.kpi_id] = current_time
            
            elif kpi.performance_status == KPIStatus.BELOW_TARGET:
                alerts.append({
                    'kpi_id': kpi.kpi_id,
                    'kpi_name': kpi.kpi_name,
                    'alert_type': 'warning',
                    'message': f'Warning: {kpi.kpi_name} is below target at {kpi.current_value}',
                    'severity': 'warning',
                    'timestamp': current_time.isoformat(),
                    'actions': kpi.action_required
                })
                self.last_alert_times[kpi.kpi_id] = current_time
        
        return alerts
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
    
    def process_alerts(self) -> None:
        """Process alerts through handlers"""
        alerts = self.check_alerts()
        
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Error in alert handler: {e}")
    
    def get_kpi_history(self, kpi_id: str, periods: int = 12) -> List[Dict[str, Any]]:
        """Get KPI historical data"""
        history = []
        current_date = date.today()
        
        for i in range(periods):
            period_end = current_date - timedelta(days=30 * i)
            period_start = period_end - timedelta(days=30)
            
            try:
                kpis = self.get_current_kpis((period_start, period_end))
                target_kpi = next((kpi for kpi in kpis if kpi.kpi_id == kpi_id), None)
                
                if target_kpi:
                    history.append({
                        'period_start': period_start.isoformat(),
                        'period_end': period_end.isoformat(),
                        'value': float(target_kpi.current_value),
                        'target': float(target_kpi.target_value),
                        'status': target_kpi.performance_status.value
                    })
            except Exception as e:
                self.logger.error(f"Error getting history for period {i}: {e}")
                continue
        
        return sorted(history, key=lambda x: x['period_start'])
    
    def export_kpi_data(self, format_type: str = 'json') -> str:
        """Export KPI data"""
        kpis = self.get_current_kpis()
        data = {
            'export_date': datetime.now().isoformat(),
            'kpis': [asdict(kpi) for kpi in kpis],
            'summary': self._calculate_dashboard_summary(kpis)
        }
        
        if format_type == 'json':
            import json
            return json.dumps(data, indent=2, default=str)
        else:
            # Could add CSV export, etc.
            return str(data)