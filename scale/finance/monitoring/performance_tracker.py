#!/usr/bin/env python3
"""
Financial Performance Monitoring Component
Real-time monitoring, analysis, and alerting for financial performance
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from collections import deque
import json
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_id: str
    metric_name: str
    alert_type: str  # warning, critical, info
    current_value: float
    threshold: float
    message: str
    timestamp: datetime
    is_active: bool = True
    resolved_at: Optional[datetime] = None

@dataclass
class PerformanceMetric:
    """Financial performance metric"""
    name: str
    current_value: float
    previous_value: float
    target_value: float
    unit: str
    frequency: str
    trend: str  # up, down, stable
    percent_change: float
    last_updated: datetime

@dataclass
class KPIResult:
    """Key Performance Indicator result"""
    kpi_name: str
    current_value: float
    target_value: float
    achievement_ratio: float
    trend_direction: str
    performance_rating: str  # excellent, good, fair, poor

class FinancialPerformanceMonitor:
    """
    Financial Performance Monitoring and Analysis Engine
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('performance_monitoring')
        
        # Monitoring components
        self.metrics = {}
        self.kpis = {}
        self.alerts = {}
        self.alert_handlers = {}
        self.performance_history = deque(maxlen=1000)
        self.real_time_data = {}
        
        # Alert thresholds
        self.thresholds = config.alert_thresholds
        
        # Monitoring frequency
        self.monitoring_intervals = {
            'real_time': 60,  # seconds
            'hourly': 3600,
            'daily': 86400,
            'weekly': 604800
        }
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the performance monitoring component"""
        try:
            # Initialize default metrics
            await self._initialize_default_metrics()
            
            # Initialize KPIs
            await self._initialize_default_kpis()
            
            # Setup alert handlers
            await self._setup_alert_handlers()
            
            # Start monitoring tasks
            monitoring_tasks = await self._start_monitoring_tasks()
            
            self.logger.info("Performance monitoring component initialized")
            return {
                'status': 'success',
                'metrics_count': len(self.metrics),
                'kpis_count': len(self.kpis),
                'alert_handlers': len(self.alert_handlers),
                'monitoring_tasks': monitoring_tasks
            }
            
        except Exception as e:
            self.logger.error(f"Performance monitoring initialization failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _initialize_default_metrics(self):
        """Initialize default performance metrics"""
        default_metrics = [
            'revenue', 'costs', 'profit_margin', 'cash_flow', 'roi', 'roe', 
            'roa', 'debt_to_equity', 'current_ratio', 'working_capital',
            'gross_margin', 'operating_margin', 'asset_turnover', 'inventory_turnover'
        ]
        
        for metric_name in default_metrics:
            self.metrics[metric_name] = {
                'current_value': 0.0,
                'previous_value': 0.0,
                'target_value': self._get_default_target(metric_name),
                'unit': self._get_metric_unit(metric_name),
                'frequency': 'daily',
                'trend': 'stable',
                'percent_change': 0.0,
                'last_updated': datetime.now(),
                'data_history': deque(maxlen=100)
            }
    
    async def _initialize_default_kpis(self):
        """Initialize default KPIs"""
        default_kpis = [
            {
                'name': 'Revenue Growth',
                'target': 0.15,
                'weight': 0.20,
                'category': 'Growth'
            },
            {
                'name': 'Profit Margin',
                'target': 0.12,
                'weight': 0.25,
                'category': 'Profitability'
            },
            {
                'name': 'Return on Equity',
                'target': 0.18,
                'weight': 0.20,
                'category': 'Returns'
            },
            {
                'name': 'Cash Flow Efficiency',
                'target': 0.85,
                'weight': 0.15,
                'category': 'Efficiency'
            },
            {
                'name': 'Debt Management',
                'target': 1.0,
                'weight': 0.20,
                'category': 'Financial Health'
            }
        ]
        
        for kpi_config in default_kpis:
            self.kpis[kpi_config['name']] = {
                'target': kpi_config['target'],
                'weight': kpi_config['weight'],
                'category': kpi_config['category'],
                'current_value': 0.0,
                'achievement_ratio': 0.0,
                'trend_direction': 'stable',
                'performance_rating': 'poor',
                'last_calculated': datetime.now()
            }
    
    async def _setup_alert_handlers(self):
        """Setup alert handlers for different alert types"""
        self.alert_handlers = {
            'warning': self._handle_warning_alert,
            'critical': self._handle_critical_alert,
            'info': self._handle_info_alert
        }
    
    async def _start_monitoring_tasks(self) -> Dict[str, Any]:
        """Start background monitoring tasks"""
        # In a real implementation, this would start async monitoring tasks
        # For now, we'll simulate the task setup
        
        monitoring_tasks = {
            'real_time_monitoring': 'started',
            'alert_processing': 'started',
            'kpi_calculation': 'started',
            'trend_analysis': 'started'
        }
        
        return monitoring_tasks
    
    def _get_default_target(self, metric_name: str) -> float:
        """Get default target value for a metric"""
        targets = {
            'revenue': 1000000.0,
            'costs': 700000.0,
            'profit_margin': 0.30,
            'cash_flow': 200000.0,
            'roi': 0.15,
            'roe': 0.18,
            'roa': 0.08,
            'debt_to_equity': 1.0,
            'current_ratio': 2.0,
            'working_capital': 300000.0,
            'gross_margin': 0.40,
            'operating_margin': 0.15,
            'asset_turnover': 1.5,
            'inventory_turnover': 12.0
        }
        return targets.get(metric_name, 1.0)
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get unit for a metric"""
        units = {
            'revenue': 'USD',
            'costs': 'USD',
            'profit_margin': '%',
            'cash_flow': 'USD',
            'roi': '%',
            'roe': '%',
            'roa': '%',
            'debt_to_equity': 'ratio',
            'current_ratio': 'ratio',
            'working_capital': 'USD',
            'gross_margin': '%',
            'operating_margin': '%',
            'asset_turnover': 'ratio',
            'inventory_turnover': 'times'
        }
        return units.get(metric_name, 'ratio')
    
    async def setup_monitoring(self, 
                             financial_data: Dict[str, Any], 
                             capital_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup monitoring for financial data and capital allocation
        
        Args:
            financial_data: Current financial data
            capital_allocation: Capital allocation results
            
        Returns:
            Dict containing monitoring setup results
        """
        self.logger.info("Setting up financial performance monitoring...")
        
        try:
            # Update metrics with current data
            metrics_update = await self._update_metrics_with_data(financial_data)
            
            # Setup capital allocation monitoring
            allocation_monitoring = await self._setup_allocation_monitoring(capital_allocation)
            
            # Initialize real-time dashboard data
            dashboard_data = await self._initialize_dashboard_data()
            
            # Setup automated alerts
            alert_setup = await self._setup_automated_alerts(financial_data)
            
            return {
                'status': 'success',
                'metrics_updated': metrics_update,
                'allocation_monitoring': allocation_monitoring,
                'dashboard_data': dashboard_data,
                'alert_setup': alert_setup,
                'monitoring_configuration': {
                    'real_time_enabled': True,
                    'alert_threshold_updated': True,
                    'kpi_tracking_enabled': True,
                    'trend_analysis_enabled': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _update_metrics_with_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update metrics with new financial data"""
        updated_metrics = []
        
        # Map financial data to metrics
        data_mapping = {
            'revenue': financial_data.get('revenue', [1000000])[-1] if isinstance(financial_data.get('revenue'), list) else financial_data.get('revenue', 1000000),
            'costs': financial_data.get('costs', [700000])[-1] if isinstance(financial_data.get('costs'), list) else financial_data.get('costs', 700000),
            'profit_margin': (financial_data.get('revenue', 1000000) - financial_data.get('costs', 700000)) / financial_data.get('revenue', 1000000) if financial_data.get('revenue', 0) > 0 else 0.0,
            'roi': self._calculate_roi(financial_data),
            'roe': self._calculate_roe(financial_data),
            'roa': self._calculate_roa(financial_data),
            'debt_to_equity': self._calculate_debt_to_equity(financial_data),
            'current_ratio': self._calculate_current_ratio(financial_data),
            'working_capital': self._calculate_working_capital(financial_data)
        }
        
        for metric_name, value in data_mapping.items():
            if metric_name in self.metrics:
                # Store previous value
                previous_value = self.metrics[metric_name]['current_value']
                
                # Update current value
                self.metrics[metric_name]['current_value'] = value
                self.metrics[metric_name]['previous_value'] = previous_value
                self.metrics[metric_name]['last_updated'] = datetime.now()
                
                # Calculate percent change
                if previous_value != 0:
                    percent_change = ((value - previous_value) / abs(previous_value)) * 100
                else:
                    percent_change = 0.0
                
                self.metrics[metric_name]['percent_change'] = percent_change
                
                # Determine trend
                if percent_change > 1.0:
                    self.metrics[metric_name]['trend'] = 'up'
                elif percent_change < -1.0:
                    self.metrics[metric_name]['trend'] = 'down'
                else:
                    self.metrics[metric_name]['trend'] = 'stable'
                
                # Add to data history
                self.metrics[metric_name]['data_history'].append({
                    'timestamp': datetime.now(),
                    'value': value,
                    'trend': self.metrics[metric_name]['trend']
                })
                
                updated_metrics.append(metric_name)
        
        return {'updated_metrics': updated_metrics, 'update_count': len(updated_metrics)}
    
    def _calculate_roi(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Return on Investment"""
        profit = financial_data.get('profit', 300000)
        investment = financial_data.get('assets', 2000000)
        return profit / investment if investment > 0 else 0.0
    
    def _calculate_roe(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Return on Equity"""
        profit = financial_data.get('profit', 300000)
        equity = financial_data.get('equity', 1500000)
        return profit / equity if equity > 0 else 0.0
    
    def _calculate_roa(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Return on Assets"""
        profit = financial_data.get('profit', 300000)
        assets = financial_data.get('assets', 2000000)
        return profit / assets if assets > 0 else 0.0
    
    def _calculate_debt_to_equity(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Debt-to-Equity ratio"""
        debt = financial_data.get('liabilities', 500000)
        equity = financial_data.get('equity', 1500000)
        return debt / equity if equity > 0 else 0.0
    
    def _calculate_current_ratio(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Current ratio"""
        current_assets = financial_data.get('current_assets', 800000)
        current_liabilities = financial_data.get('current_liabilities', 400000)
        return current_assets / current_liabilities if current_liabilities > 0 else 0.0
    
    def _calculate_working_capital(self, financial_data: Dict[str, Any]) -> float:
        """Calculate Working Capital"""
        current_assets = financial_data.get('current_assets', 800000)
        current_liabilities = financial_data.get('current_liabilities', 400000)
        return current_assets - current_liabilities
    
    async def _setup_allocation_monitoring(self, capital_allocation: Dict[str, Any]) -> Dict[str, Any]:
        """Setup monitoring for capital allocation performance"""
        allocation_monitoring = {
            'allocation_efficiency': 'monitoring',
            'rebalancing_alerts': 'enabled',
            'performance_attribution': 'active',
            'risk_monitoring': 'enabled'
        }
        
        # Store allocation data for monitoring
        if 'allocation' in capital_allocation:
            self.real_time_data['current_allocation'] = capital_allocation['allocation']
        
        return allocation_monitoring
    
    async def _initialize_dashboard_data(self) -> Dict[str, Any]:
        """Initialize real-time dashboard data"""
        dashboard_data = {
            'key_metrics': self._get_current_metrics_summary(),
            'alert_summary': await self._get_alert_summary(),
            'kpi_overview': await self._get_kpi_overview(),
            'trend_analysis': await self._get_trend_analysis(),
            'real_time_updates': True
        }
        
        return dashboard_data
    
    async def _setup_automated_alerts(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Setup automated alerting based on thresholds"""
        alert_setup = {
            'threshold_monitoring': 'enabled',
            'alert_rules_updated': True,
            'notification_channels': ['email', 'dashboard', 'api']
        }
        
        # Check for alert conditions immediately
        await self._evaluate_alert_conditions()
        
        return alert_setup
    
    def _get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        summary = {}
        
        for metric_name, metric_data in self.metrics.items():
            summary[metric_name] = {
                'current_value': metric_data['current_value'],
                'target_value': metric_data['target_value'],
                'percent_change': metric_data['percent_change'],
                'trend': metric_data['trend'],
                'unit': metric_data['unit'],
                'last_updated': metric_data['last_updated'].isoformat()
            }
        
        return summary
    
    async def _get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alerts"""
        active_alerts = [alert for alert in self.alerts.values() if alert.is_active]
        
        summary = {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.alert_type == 'critical']),
            'warning_alerts': len([a for a in active_alerts if a.alert_type == 'warning']),
            'recent_alerts': list(self.alerts.keys())[-5:]  # Last 5 alerts
        }
        
        return summary
    
    async def _get_kpi_overview(self) -> Dict[str, Any]:
        """Get KPI performance overview"""
        overview = {}
        
        for kpi_name, kpi_data in self.kpis.items():
            overview[kpi_name] = {
                'current_value': kpi_data['current_value'],
                'target_value': kpi_data['target'],
                'achievement_ratio': kpi_data['achievement_ratio'],
                'performance_rating': kpi_data['performance_rating'],
                'trend_direction': kpi_data['trend_direction']
            }
        
        return overview
    
    async def _get_trend_analysis(self) -> Dict[str, Any]:
        """Get trend analysis for key metrics"""
        trends = {}
        
        for metric_name, metric_data in self.metrics.items():
            if len(metric_data['data_history']) >= 2:
                # Simple trend analysis based on recent data points
                recent_values = [entry['value'] for entry in list(metric_data['data_history'])[-5:]]
                
                if len(recent_values) >= 3:
                    # Calculate trend direction
                    first_half_avg = np.mean(recent_values[:len(recent_values)//2])
                    second_half_avg = np.mean(recent_values[len(recent_values)//2:])
                    
                    if second_half_avg > first_half_avg * 1.02:
                        trend_direction = 'improving'
                    elif second_half_avg < first_half_avg * 0.98:
                        trend_direction = 'declining'
                    else:
                        trend_direction = 'stable'
                    
                    trends[metric_name] = {
                        'trend_direction': trend_direction,
                        'momentum': second_half_avg / first_half_avg - 1,
                        'data_points': len(recent_values)
                    }
        
        return trends
    
    async def _evaluate_alert_conditions(self):
        """Evaluate current data against alert thresholds"""
        for metric_name, metric_data in self.metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                current_value = metric_data['current_value']
                
                # Determine alert type based on deviation from threshold
                if self._is_critical_threshold_breach(metric_name, current_value, threshold):
                    await self._create_alert(metric_name, current_value, threshold, 'critical')
                elif self._is_warning_threshold_breach(metric_name, current_value, threshold):
                    await self._create_alert(metric_name, current_value, threshold, 'warning')
    
    def _is_critical_threshold_breach(self, metric_name: str, current_value: float, threshold: float) -> bool:
        """Check if metric breaches critical threshold"""
        # Different logic for different metric types
        if metric_name in ['debt_to_equity']:
            return current_value > threshold * 1.5
        elif metric_name in ['current_ratio']:
            return current_value < threshold * 0.5
        else:
            return abs(current_value - threshold) / threshold > 0.3
    
    def _is_warning_threshold_breach(self, metric_name: str, current_value: float, threshold: float) -> bool:
        """Check if metric breaches warning threshold"""
        if metric_name in ['debt_to_equity']:
            return current_value > threshold * 1.2
        elif metric_name in ['current_ratio']:
            return current_value < threshold * 0.75
        else:
            return abs(current_value - threshold) / threshold > 0.15
    
    async def _create_alert(self, metric_name: str, current_value: float, threshold: float, alert_type: str):
        """Create a new alert"""
        alert_id = f"{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        message = f"{metric_name} is {current_value:.3f} (threshold: {threshold:.3f})"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            metric_name=metric_name,
            alert_type=alert_type,
            current_value=current_value,
            threshold=threshold,
            message=message,
            timestamp=datetime.now(),
            is_active=True
        )
        
        self.alerts[alert_id] = alert
        
        # Handle alert based on type
        if alert_type in self.alert_handlers:
            await self.alert_handlers[alert_type](alert)
    
    async def _handle_warning_alert(self, alert: PerformanceAlert):
        """Handle warning alert"""
        self.logger.warning(f"Warning Alert: {alert.message}")
        # In real implementation, send notification
    
    async def _handle_critical_alert(self, alert: PerformanceAlert):
        """Handle critical alert"""
        self.logger.critical(f"Critical Alert: {alert.message}")
        # In real implementation, send immediate notification
    
    async def _handle_info_alert(self, alert: PerformanceAlert):
        """Handle info alert"""
        self.logger.info(f"Info Alert: {alert.message}")
        # In real implementation, log for dashboard
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'metrics': {name: {
                'current_value': data['current_value'],
                'target_value': data['target_value'],
                'trend': data['trend'],
                'percent_change': data['percent_change']
            } for name, data in self.metrics.items()},
            'active_alerts': len([a for a in self.alerts.values() if a.is_active]),
            'last_update': datetime.now().isoformat()
        }
    
    async def update_real_time_data(self, new_data: Dict[str, Any]):
        """Update real-time monitoring data"""
        self.real_time_data.update(new_data)
        
        # Update relevant metrics
        if 'financial_data' in new_data:
            await self._update_metrics_with_data(new_data['financial_data'])
        
        # Re-evaluate alert conditions
        await self._evaluate_alert_conditions()
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            'is_initialized': len(self.metrics) > 0,
            'metrics_tracked': len(self.metrics),
            'kpis_monitored': len(self.kpis),
            'active_alerts': len([a for a in self.alerts.values() if a.is_active]),
            'monitoring_active': True,
            'data_history_size': sum(len(m['data_history']) for m in self.metrics.values())
        }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown the component"""
        try:
            # Clear data structures
            self.metrics.clear()
            self.kpis.clear()
            self.alerts.clear()
            self.real_time_data.clear()
            self.performance_history.clear()
            
            self.logger.info("Performance monitoring component shutdown completed")
            return {'status': 'success'}
        except Exception as e:
            self.logger.error(f"Shutdown failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}