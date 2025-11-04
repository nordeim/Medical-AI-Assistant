"""
Comprehensive Performance Monitoring and Metrics System for Medical AI Platform
Phase 6 - Advanced Observability and Clinical Outcome Tracking

This module provides:
- Real-time inference metrics collection with Prometheus integration
- Latency and throughput monitoring with SLA tracking
- Memory usage tracking with GPU/CPU monitoring
- Model accuracy monitoring with drift detection
- Clinical outcome tracking with medical effectiveness measurement
- Grafana dashboard integration with real-time charts
- Advanced alerting system with configurable thresholds
"""

from .metrics.collector import MetricsCollector, InferenceMetrics, SystemMetrics, ModelMetrics
from .metrics.prometheus import PrometheusMetricsCollector
from .tracking.drift_detector import ModelDriftDetector, AccuracyMonitor
from .tracking.clinical_outcomes import ClinicalOutcomeTracker
from .alerts.alert_manager import AlertManager, ThresholdManager
from .alerts.notification_system import NotificationSystem
from .dashboards.grafana_integration import GrafanaDashboardManager

__all__ = [
    'MetricsCollector',
    'InferenceMetrics', 
    'SystemMetrics',
    'ModelMetrics',
    'PrometheusMetricsCollector',
    'ModelDriftDetector',
    'AccuracyMonitor',
    'ClinicalOutcomeTracker',
    'AlertManager',
    'ThresholdManager',
    'NotificationSystem',
    'GrafanaDashboardManager'
]

__version__ = "6.0.0"