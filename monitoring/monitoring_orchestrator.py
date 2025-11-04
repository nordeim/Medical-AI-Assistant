#!/usr/bin/env python3
"""
Medical AI Monitoring Orchestrator
Central orchestrator that coordinates all monitoring, analytics, and alerting activities
for the Medical AI system with enterprise-grade observability.
"""

import asyncio
import threading
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import schedule
import yaml
from concurrent.futures import ThreadPoolExecutor

# Import monitoring components
try:
    from drift_detection.ai_accuracy_monitor import ModelMonitoringOrchestrator
    from predictive.predictive_analytics import PredictiveOrchestrator
    from alerting.alert_manager import AlertManager
    from audit.compliance_system import AuditLogger, ComplianceFramework
    from health_checks.health_monitoring_system import HealthMonitoringOrchestrator
except ImportError as e:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    
    from drift_detection.ai_accuracy_monitor import ModelMonitoringOrchestrator
    from predictive.predictive_analytics import PredictiveOrchestrator
    from alerting.alert_manager import AlertManager
    from audit.compliance_system import AuditLogger, ComplianceFramework
    from health_checks.health_monitoring_system import HealthMonitoringOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfiguration:
    """Configuration for the monitoring orchestrator"""
    monitoring_interval_seconds: int = 30
    health_check_interval_seconds: int = 60
    compliance_report_interval_hours: int = 24
    predictive_analytics_interval_hours: int = 6
    alert_escalation_check_interval_seconds: int = 60
    data_retention_days: int = 90
    enable_predictive_analytics: bool = True
    enable_compliance_reporting: bool = True
    enable_health_monitoring: bool = True
    alert_notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.alert_notification_channels is None:
            self.alert_notification_channels = ['email', 'slack']

@dataclass
class MonitoringMetrics:
    """Metrics collected by the monitoring orchestrator"""
    timestamp: datetime
    system_health_status: str
    active_alerts_count: int
    model_health_score: float
    compliance_score: float
    predictive_insights: List[str]
    recommendations: List[str]
    system_load: Dict[str, float]
    data_quality_score: float

class MonitoringOrchestrator:
    """Main orchestrator for Medical AI monitoring system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize monitoring orchestrator
        
        Args:
            config_path: Path to monitoring configuration file
        """
        self.config = self._load_configuration(config_path)
        
        # Initialize monitoring components
        self.health_monitor: Optional[HealthMonitoringOrchestrator] = None
        self.alert_manager: Optional[AlertManager] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.model_monitor: Optional[ModelMonitoringOrchestrator] = None
        self.predictive_orchestrator: Optional[PredictiveOrchestrator] = None
        
        # Initialize components based on configuration
        self._initialize_components()
        
        # Threading and scheduling
        self.running = False
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Metrics storage
        self.metrics_history: List[MonitoringMetrics] = []
        self.last_health_check: Optional[datetime] = None
        self.last_compliance_report: Optional[datetime] = None
        self.last_predictive_analysis: Optional[datetime] = None
        
        # Statistics
        self.stats = {
            'monitoring_cycles': 0,
            'health_checks_performed': 0,
            'alerts_generated': 0,
            'compliance_reports_generated': 0,
            'predictive_analyses_performed': 0,
            'uptime_start': datetime.now(),
            'last_error': None
        }
        
        # Callbacks for external integration
        self.external_callbacks: Dict[str, Callable] = {}
        
        logger.info("Medical AI Monitoring Orchestrator initialized")
    
    def _load_configuration(self, config_path: Optional[str]) -> MonitoringConfiguration:
        """Load monitoring configuration"""
        default_config = MonitoringConfiguration()
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Update default config with loaded values
                for key, value in config_data.items():
                    if hasattr(default_config, key):
                        setattr(default_config, key, value)
                
                logger.info(f"Loaded monitoring configuration from {config_path}")
                
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _initialize_components(self) -> None:
        """Initialize monitoring components"""
        try:
            # Health monitoring
            if self.config.enable_health_monitoring:
                self.health_monitor = HealthMonitoringOrchestrator()
                logger.info("Initialized health monitoring component")
            
            # Alert management
            self.alert_manager = AlertManager()
            self.alert_manager.add_medical_alert_rules()
            logger.info("Initialized alert management component")
            
            # Audit logging
            self.audit_logger = AuditLogger()
            logger.info("Initialized audit logging component")
            
            # AI model monitoring
            self.model_monitor = ModelMonitoringOrchestrator(
                model_name="medical_ai_system",
                protected_attributes=["age_group", "gender", "ethnicity"]
            )
            logger.info("Initialized AI model monitoring component")
            
            # Predictive analytics
            if self.config.enable_predictive_analytics:
                self.predictive_orchestrator = PredictiveOrchestrator()
                logger.info("Initialized predictive analytics component")
            
            logger.info("All monitoring components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing monitoring components: {str(e)}")
            raise
    
    def start_monitoring(self) -> None:
        """Start the complete monitoring system"""
        if self.running:
            logger.warning("Monitoring is already running")
            return
        
        self.running = True
        logger.info("Starting Medical AI monitoring system...")
        
        try:
            # Start individual components
            if self.health_monitor:
                self.health_monitor.start_continuous_monitoring(self.config.health_check_interval_seconds)
            
            if self.alert_manager:
                self.alert_manager.start_monitoring()
            
            # Start monitoring threads
            self._start_monitoring_threads()
            
            # Schedule periodic tasks
            self._schedule_periodic_tasks()
            
            # Log startup
            self._log_audit_event("monitoring_system_started", "Monitoring system started successfully")
            
            logger.info("Medical AI monitoring system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting monitoring system: {str(e)}")
            self.stats['last_error'] = str(e)
            self.stop_monitoring()
            raise
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        if not self.running:
            logger.warning("Monitoring is not running")
            return
        
        self.running = False
        logger.info("Stopping Medical AI monitoring system...")
        
        try:
            # Stop monitoring threads
            for thread_name, thread in self.monitoring_threads.items():
                logger.info(f"Stopping monitoring thread: {thread_name}")
                # Note: In a production system, you would implement proper thread stopping
                # For this demo, threads will naturally exit when self.running = False
            
            # Stop components
            if self.alert_manager:
                self.alert_manager.stop_monitoring()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Log shutdown
            self._log_audit_event("monitoring_system_stopped", "Monitoring system stopped")
            
            logger.info("Medical AI monitoring system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring system: {str(e)}")
    
    def _start_monitoring_threads(self) -> None:
        """Start monitoring threads"""
        # Main monitoring cycle
        self.monitoring_threads['main_monitoring'] = threading.Thread(
            target=self._main_monitoring_loop,
            daemon=True
        )
        self.monitoring_threads['main_monitoring'].start()
        
        # Health monitoring thread
        if self.health_monitor:
            self.monitoring_threads['health_monitoring'] = threading.Thread(
                target=self._health_monitoring_loop,
                daemon=True
            )
            self.monitoring_threads['health_monitoring'].start()
        
        # Compliance reporting thread
        if self.config.enable_compliance_reporting:
            self.monitoring_threads['compliance_reporting'] = threading.Thread(
                target=self._compliance_reporting_loop,
                daemon=True
            )
            self.monitoring_threads['compliance_reporting'].start()
        
        # Predictive analytics thread
        if self.config.enable_predictive_analytics:
            self.monitoring_threads['predictive_analytics'] = threading.Thread(
                target=self._predictive_analytics_loop,
                daemon=True
            )
            self.monitoring_threads['predictive_analytics'].start()
        
        logger.info(f"Started {len(self.monitoring_threads)} monitoring threads")
    
    def _main_monitoring_loop(self) -> None:
        """Main monitoring loop that coordinates all monitoring activities"""
        while self.running:
            try:
                start_time = time.time()
                
                # Perform comprehensive monitoring cycle
                self._perform_monitoring_cycle()
                
                # Calculate sleep time to maintain interval
                cycle_time = time.time() - start_time
                sleep_time = max(0, self.config.monitoring_interval_seconds - cycle_time)
                
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main monitoring loop: {str(e)}")
                self.stats['last_error'] = str(e)
                time.sleep(self.config.monitoring_interval_seconds)
    
    def _perform_monitoring_cycle(self) -> None:
        """Perform one complete monitoring cycle"""
        cycle_start = time.time()
        self.stats['monitoring_cycles'] += 1
        
        try:
            logger.debug(f"Starting monitoring cycle {self.stats['monitoring_cycles']}")
            
            # Collect system metrics
            system_metrics = self._collect_system_metrics()
            
            # Run health checks
            health_status = self._run_health_checks()
            
            # Process alerts
            alert_results = self._process_alerts(system_metrics)
            
            # AI model monitoring
            model_results = self._monitor_ai_models(system_metrics)
            
            # Generate insights and recommendations
            insights = self._generate_insights(system_metrics, health_status, alert_results, model_results)
            
            # Store metrics
            metrics = MonitoringMetrics(
                timestamp=datetime.now(),
                system_health_status=health_status.get('overall_status', 'unknown'),
                active_alerts_count=alert_results.get('active_alerts_count', 0),
                model_health_score=model_results.get('health_score', 0.0),
                compliance_score=0.8,  # Simplified - would calculate from audit data
                predictive_insights=insights.get('predictive', []),
                recommendations=insights.get('recommendations', []),
                system_load=system_metrics.get('system_load', {}),
                data_quality_score=insights.get('data_quality', 0.0)
            )
            
            self.metrics_history.append(metrics)
            
            # Trim metrics history if too long
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Log cycle completion
            cycle_time = time.time() - cycle_start
            logger.debug(f"Monitoring cycle {self.stats['monitoring_cycles']} completed in {cycle_time:.2f}s")
            
            # Run external callbacks
            self._run_external_callbacks('monitoring_cycle', {
                'cycle_number': self.stats['monitoring_cycles'],
                'metrics': asdict(metrics),
                'processing_time': cycle_time
            })
            
        except Exception as e:
            logger.error(f"Error in monitoring cycle: {str(e)}")
            self.stats['last_error'] = str(e)
            self._log_audit_event("monitoring_cycle_error", f"Error in monitoring cycle: {str(e)}")
    
    def _health_monitoring_loop(self) -> None:
        """Dedicated health monitoring loop"""
        while self.running:
            try:
                if self.health_monitor:
                    # Run health checks
                    health_results = self.health_monitor.run_all_checks()
                    system_health = self.health_monitor.get_system_health()
                    
                    # Log significant health changes
                    if self.last_health_check:
                        if system_health.overall_status.value != self.last_health_check.overall_status.value:
                            self._log_audit_event(
                                "health_status_change",
                                f"Health status changed from {self.last_health_check.overall_status.value} to {system_health.overall_status.value}"
                            )
                    
                    self.last_health_check = system_health
                    self.stats['health_checks_performed'] += 1
                
                time.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {str(e)}")
                time.sleep(self.config.health_check_interval_seconds)
    
    def _compliance_reporting_loop(self) -> None:
        """Dedicated compliance reporting loop"""
        while self.running:
            try:
                if self.config.enable_compliance_reporting and self.audit_logger:
                    # Check if it's time for compliance reporting
                    now = datetime.now()
                    if (self.last_compliance_report is None or 
                        (now - self.last_compliance_report).total_seconds() >= self.config.compliance_report_interval_hours * 3600):
                        
                        # Generate compliance reports
                        end_date = now
                        start_date = now - timedelta(days=30)
                        
                        for framework in [ComplianceFramework.HIPAA, ComplianceFramework.FDA_21CFR_PART11]:
                            try:
                                report = self.audit_logger.generate_compliance_report(
                                    framework=framework,
                                    start_date=start_date,
                                    end_date=end_date,
                                    report_format='json'
                                )
                                
                                logger.info(f"Generated {framework.value} compliance report: {report.report_id}")
                                self.stats['compliance_reports_generated'] += 1
                                
                                # Log report generation
                                self._log_audit_event(
                                    "compliance_report_generated",
                                    f"Generated {framework.value} compliance report",
                                    details={
                                        'report_id': report.report_id,
                                        'framework': framework.value,
                                        'total_events': report.total_events
                                    }
                                )
                                
                            except Exception as e:
                                logger.error(f"Error generating {framework.value} compliance report: {str(e)}")
                        
                        self.last_compliance_report = now
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in compliance reporting loop: {str(e)}")
                time.sleep(3600)
    
    def _predictive_analytics_loop(self) -> None:
        """Dedicated predictive analytics loop"""
        while self.running:
            try:
                if self.config.enable_predictive_analytics and self.predictive_orchestrator:
                    # Check if it's time for predictive analysis
                    now = datetime.now()
                    if (self.last_predictive_analysis is None or 
                        (now - self.last_predictive_analysis).total_seconds() >= self.config.predictive_analytics_interval_hours * 3600):
                        
                        # Collect system metrics for prediction
                        system_metrics = self._collect_system_metrics()
                        timestamps = [datetime.now() - timedelta(hours=i) for i in range(168, 0, -1)]
                        
                        # Run predictive analysis
                        prediction_results = self.predictive_orchestrator.run_comprehensive_prediction(
                            system_metrics=system_metrics,
                            timestamps=timestamps
                        )
                        
                        logger.info("Completed predictive analytics analysis")
                        self.stats['predictive_analyses_performed'] += 1
                        
                        # Log significant predictions
                        recommendations = prediction_results.get('recommendations', [])
                        if recommendations:
                            self._log_audit_event(
                                "predictive_insights_generated",
                                f"Generated {len(recommendations)} predictive insights",
                                details={'recommendations': recommendations}
                            )
                        
                        self.last_predictive_analysis = now
                
                time.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Error in predictive analytics loop: {str(e)}")
                time.sleep(1800)
    
    def _schedule_periodic_tasks(self) -> None:
        """Schedule periodic tasks using the schedule library"""
        # Daily tasks
        schedule.every().day.at("02:00").do(self._daily_maintenance)
        schedule.every().day.at("08:00").do(self._generate_daily_summary)
        
        # Weekly tasks
        schedule.every().monday.at("09:00").do(self._weekly_compliance_review)
        schedule.every().sunday.at("10:00").do(self._generate_weekly_report)
        
        # Start scheduler in separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Periodic tasks scheduled")
    
    def _run_scheduler(self) -> None:
        """Run the scheduler"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            import psutil
            
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Medical AI specific metrics (simulated)
            ai_metrics = {
                'clinical_decision_accuracy': 0.85 + (hash(str(datetime.now().hour)) % 100) / 1000,
                'model_bias_score': 0.05 + (hash(str(datetime.now().minute)) % 50) / 1000,
                'phi_exposure_count': 0,  # Should be monitored for actual exposure
                'inference_time_ms': 150 + (hash(str(datetime.now().second)) % 100),
                'active_sessions': 25 + (hash(str(datetime.now().hour)) % 50),
                'patient_data_access_count': 100 + (hash(str(datetime.now().hour)) % 200)
            }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'system_load': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': (disk.used / disk.total) * 100,
                    'disk_free_gb': disk.free / (1024**3),
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv,
                    'process_count': process_count
                },
                'ai_metrics': ai_metrics,
                'application_metrics': {
                    'requests_per_minute': 150 + (hash(str(datetime.now().minute)) % 100),
                    'response_time_ms': 200 + (hash(str(datetime.now().second)) % 100),
                    'error_rate_percent': 1.5 + (hash(str(datetime.now().hour)) % 50) / 10,
                    'active_connections': 25 + (hash(str(datetime.now().minute)) % 50)
                }
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _run_health_checks(self) -> Dict[str, Any]:
        """Run health checks and return results"""
        if not self.health_monitor:
            return {'error': 'Health monitor not available'}
        
        try:
            # Get system health status
            system_health = self.health_monitor.get_system_health()
            
            return {
                'overall_status': system_health.overall_status.value,
                'alert_level': system_health.alert_level,
                'checks_summary': {
                    'total': system_health.checks_total,
                    'healthy': system_health.checks_healthy,
                    'warning': system_health.checks_warning,
                    'critical': system_health.checks_critical
                },
                'system_metrics': system_health.system_metrics,
                'recommendations': system_health.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error running health checks: {str(e)}")
            return {'error': str(e)}
    
    def _process_alerts(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process alerts based on system metrics"""
        if not self.alert_manager:
            return {'error': 'Alert manager not available'}
        
        try:
            # Prepare metrics for alert processing
            alert_metrics = {
                'metrics': system_metrics.get('ai_metrics', {}),
                'system': system_metrics.get('system_load', {}),
                'application': system_metrics.get('application_metrics', {})
            }
            
            # Process metrics through alert manager
            alert_results = self.alert_manager.process_metrics(alert_metrics)
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            return {
                'active_alerts_count': len(active_alerts),
                'alerts_generated': len(alert_results.get('alerts_generated', [])),
                'alerts_resolved': len(alert_results.get('alerts_resolved', [])),
                'active_alerts': active_alerts,
                'processing_results': alert_results
            }
            
        except Exception as e:
            logger.error(f"Error processing alerts: {str(e)}")
            return {'error': str(e)}
    
    def _monitor_ai_models(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor AI models for drift, bias, and performance"""
        if not self.model_monitor:
            return {'error': 'Model monitor not available'}
        
        try:
            # Generate sample data for monitoring (in production, use actual data)
            import numpy as np
            
            np.random.seed(42)
            n_samples = 100
            
            # Generate sample predictions and labels
            predictions = np.random.choice([0, 1], size=n_samples)
            labels = np.random.choice([0, 1], size=n_samples)
            
            # Generate sample protected attributes
            protected_attributes = {
                'age_group': np.random.choice(['young', 'middle', 'elderly'], size=n_samples),
                'gender': np.random.choice(['male', 'female'], size=n_samples),
                'ethnicity': np.random.choice(['group_a', 'group_b', 'group_c'], size=n_samples)
            }
            
            # Run comprehensive model monitoring
            monitoring_results = self.model_monitor.run_comprehensive_monitoring(
                predictions=predictions,
                labels=labels,
                protected_attrs=protected_attributes
            )
            
            return {
                'health_score': monitoring_results.get('overall_health_score', 0.0),
                'drift_detected': monitoring_results.get('drift_detection', {}).get('drift_detected', False),
                'bias_detected': len(monitoring_results.get('bias_analysis', {})) > 0,
                'alerts_generated': len(monitoring_results.get('alerts', [])),
                'monitoring_results': monitoring_results
            }
            
        except Exception as e:
            logger.error(f"Error monitoring AI models: {str(e)}")
            return {'error': str(e)}
    
    def _generate_insights(self, 
                          system_metrics: Dict[str, Any],
                          health_status: Dict[str, Any],
                          alert_results: Dict[str, Any],
                          model_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate insights and recommendations"""
        insights = {
            'predictive': [],
            'recommendations': [],
            'data_quality': 0.0
        }
        
        try:
            # System performance insights
            system_load = system_metrics.get('system_load', {})
            
            if system_load.get('cpu_percent', 0) > 80:
                insights['recommendations'].append("High CPU usage detected - consider scaling resources")
                insights['predictive'].append("CPU utilization trending upward")
            
            if system_load.get('memory_percent', 0) > 85:
                insights['recommendations'].append("High memory usage - investigate memory leaks")
                insights['predictive'].append("Memory usage approaching capacity")
            
            # Health status insights
            if health_status.get('overall_status') == 'critical':
                insights['recommendations'].append("Critical system health issues - immediate attention required")
                insights['predictive'].append("System health critical - downtime likely")
            
            # Alert insights
            active_alerts = alert_results.get('active_alerts_count', 0)
            if active_alerts > 5:
                insights['recommendations'].append("High number of active alerts - review alert thresholds")
            
            # AI model insights
            model_health = model_results.get('health_score', 0.0)
            if model_health < 0.7:
                insights['recommendations'].append("AI model performance below threshold - consider retraining")
                insights['predictive'].append("Model drift likely - schedule performance review")
            
            if model_results.get('bias_detected'):
                insights['recommendations'].append("AI bias detected - review training data and fairness constraints")
            
            # Data quality assessment
            data_quality_indicators = [
                1.0 if system_metrics.get('ai_metrics', {}).get('phi_exposure_count', 1) == 0 else 0.5,
                1.0 if system_load.get('cpu_percent', 100) < 90 else 0.7,
                1.0 if alert_results.get('active_alerts_count', 100) < 10 else 0.8
            ]
            
            insights['data_quality'] = sum(data_quality_indicators) / len(data_quality_indicators)
            
            # Default recommendations if none generated
            if not insights['recommendations']:
                insights['recommendations'].append("System operating within normal parameters")
            
            if not insights['predictive']:
                insights['predictive'].append("No significant trends detected - continue monitoring")
                
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            insights['recommendations'].append(f"Unable to generate specific recommendations: {str(e)}")
        
        return insights
    
    def _daily_maintenance(self) -> None:
        """Perform daily maintenance tasks"""
        try:
            logger.info("Performing daily maintenance...")
            
            # Clean up old metrics
            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
            self.metrics_history = [
                m for m in self.metrics_history 
                if m.timestamp > cutoff_date
            ]
            
            # Log maintenance activity
            self._log_audit_event("daily_maintenance", "Daily maintenance completed")
            
            logger.info("Daily maintenance completed")
            
        except Exception as e:
            logger.error(f"Error in daily maintenance: {str(e)}")
    
    def _generate_daily_summary(self) -> None:
        """Generate daily monitoring summary"""
        try:
            logger.info("Generating daily summary...")
            
            # Calculate daily metrics
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_metrics = [m for m in self.metrics_history if m.timestamp >= today]
            
            if today_metrics:
                summary = {
                    'date': today.date().isoformat(),
                    'monitoring_cycles': len(today_metrics),
                    'average_system_health': sum(1 for m in today_metrics if m.system_health_status == 'healthy') / len(today_metrics),
                    'total_alerts': sum(m.active_alerts_count for m in today_metrics),
                    'average_model_health': sum(m.model_health_score for m in today_metrics) / len(today_metrics),
                    'data_quality_score': sum(m.data_quality_score for m in today_metrics) / len(today_metrics)
                }
                
                # Log summary
                self._log_audit_event("daily_summary_generated", "Daily monitoring summary generated", summary)
                
                logger.info(f"Daily summary generated: {summary}")
            else:
                logger.info("No monitoring data available for today")
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {str(e)}")
    
    def _weekly_compliance_review(self) -> None:
        """Perform weekly compliance review"""
        try:
            logger.info("Performing weekly compliance review...")
            
            # Generate comprehensive compliance report
            if self.audit_logger:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                
                for framework in [ComplianceFramework.HIPAA, ComplianceFramework.FDA_21CFR_PART11]:
                    try:
                        report = self.audit_logger.generate_compliance_report(
                            framework=framework,
                            start_date=start_date,
                            end_date=end_date,
                            report_format='json'
                        )
                        
                        logger.info(f"Weekly {framework.value} compliance report generated: {report.report_id}")
                        
                    except Exception as e:
                        logger.error(f"Error generating weekly {framework.value} report: {str(e)}")
            
            self._log_audit_event("weekly_compliance_review", "Weekly compliance review completed")
            
        except Exception as e:
            logger.error(f"Error in weekly compliance review: {str(e)}")
    
    def _generate_weekly_report(self) -> None:
        """Generate comprehensive weekly monitoring report"""
        try:
            logger.info("Generating weekly report...")
            
            # Calculate weekly metrics
            week_ago = datetime.now() - timedelta(days=7)
            week_metrics = [m for m in self.metrics_history if m.timestamp >= week_ago]
            
            if week_metrics:
                report = {
                    'report_period': {
                        'start': week_ago.isoformat(),
                        'end': datetime.now().isoformat()
                    },
                    'summary_statistics': {
                        'total_monitoring_cycles': len(week_metrics),
                        'system_uptime_percent': (sum(1 for m in week_metrics if m.system_health_status == 'healthy') / len(week_metrics)) * 100,
                        'average_alerts_per_cycle': sum(m.active_alerts_count for m in week_metrics) / len(week_metrics),
                        'average_model_health': sum(m.model_health_score for m in week_metrics) / len(week_metrics),
                        'average_compliance_score': sum(m.compliance_score for m in week_metrics) / len(week_metrics)
                    },
                    'key_insights': [],
                    'recommendations': [],
                    'trends': []
                }
                
                # Add insights from the week
                all_recommendations = []
                all_predictive = []
                
                for metrics in week_metrics:
                    all_recommendations.extend(metrics.recommendations)
                    all_predictive.extend(metrics.predictive_insights)
                
                # Get most common recommendations
                from collections import Counter
                recommendation_counts = Counter(all_recommendations)
                report['key_insights'] = [rec for rec, count in recommendation_counts.most_common(5)]
                
                # Log weekly report
                self._log_audit_event("weekly_report_generated", "Weekly monitoring report generated", report)
                
                logger.info(f"Weekly report generated with {len(week_metrics)} data points")
            else:
                logger.info("No monitoring data available for the past week")
            
        except Exception as e:
            logger.error(f"Error generating weekly report: {str(e)}")
    
    def _log_audit_event(self, action: str, description: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log audit event"""
        if self.audit_logger:
            try:
                self.audit_logger.log_event(
                    event_type="system_configuration",  # Would be more specific in production
                    action_performed=action,
                    outcome="success",
                    details=details or {}
                )
            except Exception as e:
                logger.error(f"Error logging audit event: {str(e)}")
    
    def _run_external_callbacks(self, event_type: str, data: Dict[str, Any]) -> None:
        """Run external callbacks for the given event type"""
        if event_type in self.external_callbacks:
            try:
                for callback_name, callback in self.external_callbacks[event_type].items():
                    callback(data)
            except Exception as e:
                logger.error(f"Error running external callback {event_type}: {str(e)}")
    
    def register_callback(self, event_type: str, callback_name: str, callback: Callable) -> None:
        """Register external callback for events"""
        if event_type not in self.external_callbacks:
            self.external_callbacks[event_type] = {}
        
        self.external_callbacks[event_type][callback_name] = callback
        logger.info(f"Registered callback {callback_name} for {event_type}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            # Get current health status
            health_status = {}
            if self.health_monitor:
                system_health = self.health_monitor.get_system_health()
                health_status = {
                    'overall_status': system_health.overall_status.value,
                    'alert_level': system_health.alert_level,
                    'checks_summary': {
                        'total': system_health.checks_total,
                        'healthy': system_health.checks_healthy,
                        'warning': system_health.checks_warning,
                        'critical': system_health.checks_critical
                    }
                }
            
            # Get alert status
            alert_status = {}
            if self.alert_manager:
                active_alerts = self.alert_manager.get_active_alerts()
                alert_status = {
                    'active_alerts_count': len(active_alerts),
                    'active_alerts': active_alerts[:10],  # Limit to first 10
                    'total_alerts': self.alert_manager.get_statistics().get('total_alerts', 0)
                }
            
            # Get recent metrics
            recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
            
            return {
                'timestamp': datetime.now().isoformat(),
                'running': self.running,
                'uptime_seconds': (datetime.now() - self.stats['uptime_start']).total_seconds(),
                'monitoring_cycles': self.stats['monitoring_cycles'],
                'health_status': health_status,
                'alert_status': alert_status,
                'recent_metrics': [asdict(m) for m in recent_metrics],
                'statistics': self.stats,
                'last_error': self.stats.get('last_error')
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'running': self.running
            }
    
    def export_monitoring_data(self, output_path: str, format: str = 'json') -> None:
        """Export monitoring data to file"""
        try:
            data = {
                'export_timestamp': datetime.now().isoformat(),
                'configuration': asdict(self.config),
                'statistics': self.stats,
                'metrics_history': [asdict(m) for m in self.metrics_history],
                'system_status': self.get_system_status()
            }
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == 'yaml':
                import yaml
                with open(output_path, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Monitoring data exported to {output_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting monitoring data: {str(e)}")
            raise

# Main execution
if __name__ == "__main__":
    import sys
    
    # Initialize orchestrator
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    orchestrator = MonitoringOrchestrator(config_path)
    
    # Register example callback
    def example_callback(data):
        print(f"Callback received: {data['cycle_number']}")
    
    orchestrator.register_callback('monitoring_cycle', 'example', example_callback)
    
    try:
        # Start monitoring
        print("Starting Medical AI Monitoring Orchestrator...")
        orchestrator.start_monitoring()
        
        # Keep running
        print("Monitoring system is running. Press Ctrl+C to stop.")
        while orchestrator.running:
            time.sleep(30)
            
            # Print status every 5 minutes
            if orchestrator.stats['monitoring_cycles'] % 10 == 0:
                status = orchestrator.get_system_status()
                print(f"\n=== System Status ===")
                print(f"Running: {status['running']}")
                print(f"Uptime: {status['uptime_seconds']/3600:.1f} hours")
                print(f"Monitoring Cycles: {status['monitoring_cycles']}")
                if status['health_status']:
                    print(f"Health: {status['health_status']['overall_status']}")
                if status['alert_status']:
                    print(f"Active Alerts: {status['alert_status']['active_alerts_count']}")
                print("===================\n")
        
    except KeyboardInterrupt:
        print("\nShutting down monitoring system...")
        orchestrator.stop_monitoring()
        print("Monitoring system stopped.")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        orchestrator.stop_monitoring()