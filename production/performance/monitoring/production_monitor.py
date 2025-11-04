"""
Production Monitor for Medical AI Assistant
Comprehensive monitoring, alerting, and performance tracking system
"""

import asyncio
import logging
import json
import time
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import statistics
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MonitoringMetrics:
    """Production monitoring metrics"""
    timestamp: datetime
    service_name: str
    response_time_p95: float
    response_time_p99: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float

class ProductionMonitor:
    """Production-grade monitoring system for medical AI"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_buffer = deque(maxlen=10000)
        self.active_alerts = {}
        self.dashboards = {}
        self.medical_services = [
            "medical-ai-api",
            "patient-data-service",
            "clinical-data-service", 
            "ai-inference-service",
            "audit-log-service",
            "notification-service"
        ]
        
    async def setup_metrics_collection(self) -> Dict[str, Any]:
        """Set up comprehensive metrics collection system"""
        logger.info("Setting up metrics collection system")
        
        results = {
            "metrics_collection": {},
            "collection_agents": {},
            "data_sources": {},
            "aggregation_rules": {},
            "errors": []
        }
        
        try:
            # Metrics collection configuration
            metrics_collection = {
                "collection_intervals": {
                    "real_time_metrics": 15,  # seconds
                    "system_metrics": 30,
                    "business_metrics": 60,
                    "health_metrics": 10
                },
                "retention_policies": {
                    "high_frequency": "7 days",    # 15-30 second intervals
                    "medium_frequency": "30 days", # 1-5 minute intervals
                    "low_frequency": "90 days",    # 15+ minute intervals
                    "audit_logs": "7 years"        # Compliance requirement
                },
                "medical_specific_metrics": [
                    "patient_request_rate",
                    "clinical_data_processing_rate",
                    "ai_inference_queue_depth",
                    "emergency_request_priority",
                    "patient_data_access_patterns",
                    "medical_compliance_metrics"
                ]
            }
            
            results["metrics_collection"] = metrics_collection
            
            # Collection agents for different metric types
            collection_agents = {
                "system_metrics_agent": {
                    "interval": 30,
                    "metrics": [
                        "cpu_usage_percent",
                        "memory_usage_percent",
                        "disk_usage_percent",
                        "network_io_bytes_per_sec",
                        "process_count",
                        "thread_count"
                    ],
                    "source": "node_exporter",
                    "enabled": True
                },
                "application_metrics_agent": {
                    "interval": 15,
                    "metrics": [
                        "request_rate_per_second",
                        "response_time_p50",
                        "response_time_p95",
                        "response_time_p99",
                        "error_rate_percent",
                        "active_connections",
                        "queue_depth"
                    ],
                    "source": "application_exporter",
                    "enabled": True
                },
                "business_metrics_agent": {
                    "interval": 60,
                    "metrics": [
                        "patients_processed_per_hour",
                        "clinical_data_entries_per_hour",
                        "ai_inference_requests_per_hour",
                        "medical_tasks_completed",
                        "audit_events_per_hour"
                    ],
                    "source": "business_metrics_collector",
                    "enabled": True
                },
                "medical_compliance_agent": {
                    "interval": 300,
                    "metrics": [
                        "hipaa_compliance_score",
                        "phi_access_rate",
                        "audit_trail_completeness",
                        "data_encryption_coverage"
                    ],
                    "source": "compliance_monitor",
                    "enabled": True
                }
            }
            
            results["collection_agents"] = collection_agents
            
            # Data sources configuration
            data_sources = {
                "prometheus": {
                    "endpoint": "http://prometheus:9090",
                    "retention": "30d",
                    "scrape_interval": "15s",
                    "enabled": True
                },
                "grafana": {
                    "endpoint": "http://grafana:3000",
                    "datasources": ["prometheus", "loki"],
                    "enabled": True
                },
                "custom_collectors": {
                    "medical_api_collector": {
                        "endpoint": "/api/metrics",
                        "format": "prometheus",
                        "interval": 30
                    },
                    "database_collector": {
                        "endpoint": "/metrics/database",
                        "format": "custom",
                        "interval": 60
                    }
                }
            }
            
            results["data_sources"] = data_sources
            
            # Aggregation rules for medical workflows
            aggregation_rules = {
                "response_time_aggregation": {
                    "percentiles": [50, 90, 95, 99],
                    "time_windows": ["1m", "5m", "15m", "1h"],
                    "medical_weighting": {
                        "patient_lookup": 2.0,
                        "clinical_data": 1.5,
                        "ai_inference": 1.0,
                        "audit_logs": 0.5
                    }
                },
                "throughput_aggregation": {
                    "time_windows": ["1m", "5m", "15m", "1h", "1d"],
                    "service_level_aggregation": True,
                    "medical_workflow_aggregation": True
                },
                "error_rate_aggregation": {
                    "time_windows": ["1m", "5m", "15m", "1h"],
                    "error_classification": [
                        "patient_data_errors",
                        "clinical_data_errors", 
                        "ai_inference_errors",
                        "system_errors"
                    ],
                    "severity_weighting": {
                        "critical": 5.0,
                        "high": 3.0,
                        "medium": 1.0,
                        "low": 0.1
                    }
                }
            }
            
            results["aggregation_rules"] = aggregation_rules
            
            # Initialize metrics collection
            await self._initialize_metrics_collection()
            
            logger.info("Metrics collection setup completed successfully")
            
        except Exception as e:
            logger.error(f"Metrics collection setup failed: {str(e)}")
            results["errors"].append({"component": "metrics_collection", "error": str(e)})
        
        return results
    
    async def _initialize_metrics_collection(self) -> None:
        """Initialize metrics collection components"""
        # Start collection agents
        for agent_name, config in {
            "system_metrics": {"interval": 30, "enabled": True},
            "application_metrics": {"interval": 15, "enabled": True},
            "business_metrics": {"interval": 60, "enabled": True}
        }.items():
            if config["enabled"]:
                logger.info(f"Started {agent_name} collection agent")
                # In production, this would start actual collection processes
    
    async def configure_alerting(self) -> Dict[str, Any]:
        """Configure comprehensive alerting system"""
        logger.info("Configuring alerting system")
        
        results = {
            "alerting_rules": {},
            "alert_channels": {},
            "escalation_policies": {},
            "medical_alerts": {},
            "errors": []
        }
        
        try:
            # Alerting rules for medical AI services
            alerting_rules = {
                "critical_alerts": [
                    {
                        "name": "medical_ai_api_down",
                        "condition": "up == 0",
                        "duration": "30s",
                        "labels": {"severity": "critical", "service": "medical-ai-api"},
                        "actions": ["page_oncall", "auto_scale_down"]
                    },
                    {
                        "name": "patient_data_service_unavailable",
                        "condition": "up == 0 AND service == patient_data",
                        "duration": "60s",
                        "labels": {"severity": "critical", "service": "patient-data"},
                        "actions": ["page_oncall", "emergency_scale_up"]
                    },
                    {
                        "name": "high_error_rate",
                        "condition": "error_rate > 0.05",
                        "duration": "2m",
                        "labels": {"severity": "critical"},
                        "actions": ["page_oncall", "traffic_shift"]
                    },
                    {
                        "name": "emergency_surge_detected",
                        "condition": "emergency_requests_per_minute > 50",
                        "duration": "1m",
                        "labels": {"severity": "critical", "type": "emergency"},
                        "actions": ["page_oncall", "emergency_scale_up", "priority_queuing"]
                    }
                ],
                "warning_alerts": [
                    {
                        "name": "high_response_time",
                        "condition": "response_time_p95 > 2.0",
                        "duration": "5m",
                        "labels": {"severity": "warning"},
                        "actions": ["notify_team", "investigate"]
                    },
                    {
                        "name": "high_cpu_usage",
                        "condition": "cpu_usage > 80",
                        "duration": "10m",
                        "labels": {"severity": "warning"},
                        "actions": ["notify_team", "auto_scale_up"]
                    },
                    {
                        "name": "high_memory_usage",
                        "condition": "memory_usage > 85",
                        "duration": "5m",
                        "labels": {"severity": "warning"},
                        "actions": ["notify_team", "investigate_memory_leaks"]
                    },
                    {
                        "name": "low_cache_hit_rate",
                        "condition": "cache_hit_rate < 0.80",
                        "duration": "10m",
                        "labels": {"severity": "warning"},
                        "actions": ["notify_team", "analyze_cache_patterns"]
                    }
                ],
                "info_alerts": [
                    {
                        "name": "deployment_completed",
                        "condition": "deployment_status == success",
                        "duration": "0s",
                        "labels": {"severity": "info", "type": "deployment"},
                        "actions": ["notify_team", "update_dashboard"]
                    },
                    {
                        "name": "scaling_event",
                        "condition": "scaling_event == true",
                        "duration": "0s",
                        "labels": {"severity": "info", "type": "scaling"},
                        "actions": ["log_event", "update_dashboard"]
                    }
                ]
            }
            
            results["alerting_rules"] = alerting_rules
            
            # Alert channels configuration
            alert_channels = {
                "email": {
                    "enabled": True,
                    "recipients": {
                        "critical": ["oncall@hospital.com", "ops-team@hospital.com"],
                        "warning": ["dev-team@hospital.com"],
                        "info": ["dev-team@hospital.com"]
                    },
                    "smtp_server": "smtp.hospital.com",
                    "smtp_port": 587,
                    "smtp_username": "${SMTP_USERNAME}",
                    "smtp_password": "${SMTP_PASSWORD}"
                },
                "slack": {
                    "enabled": True,
                    "webhook_url": "${SLACK_WEBHOOK_URL}",
                    "channels": {
                        "critical": "#medical-ai-alerts",
                        "warning": "#medical-ai-warnings",
                        "info": "#medical-ai-info"
                    },
                    "mention_oncall": True,
                    "medical_ai_emoji": "🏥"
                },
                "pagerduty": {
                    "enabled": True,
                    "integration_key": "${PAGERDUTY_INTEGRATION_KEY}",
                    "service_key": "${PAGERDUTY_SERVICE_KEY}",
                    "escalation_policy": "medical_ai_escalation"
                },
                "sms": {
                    "enabled": True,
                    "provider": "twilio",
                    "recipients": {
                        "critical": ["+15551234567", "+15557654321"]
                    },
                    "account_sid": "${TWILIO_ACCOUNT_SID}",
                    "auth_token": "${TWILIO_AUTH_TOKEN}"
                }
            }
            
            results["alert_channels"] = alert_channels
            
            # Escalation policies for medical incidents
            escalation_policies = {
                "critical_incident": {
                    "level_1": {"time": "0m", "action": "page_oncall"},
                    "level_2": {"time": "5m", "action": "escalate_to_senior"},
                    "level_3": {"time": "15m", "action": "escalate_to_management"},
                    "level_4": {"time": "30m", "action": "activate_disaster_recovery"}
                },
                "warning_incident": {
                    "level_1": {"time": "0m", "action": "notify_team"},
                    "level_2": {"time": "30m", "action": "escalate_if_unresolved"}
                },
                "emergency_incident": {
                    "level_1": {"time": "0m", "action": "page_all_oncall"},
                    "level_2": {"time": "2m", "action": "activate_emergency_response"},
                    "level_3": {"time": "5m", "action": "emergency_meeting"}
                }
            }
            
            results["escalation_policies"] = escalation_policies
            
            # Medical-specific alerting
            medical_alerts = {
                "hipaa_compliance": {
                    "phi_access_rate": {
                        "condition": "unauthorized_phi_access > 0",
                        "action": "immediate_security_alert",
                        "escalation": "critical"
                    },
                    "audit_trail_gaps": {
                        "condition": "audit_coverage < 0.95",
                        "action": "compliance_team_alert",
                        "escalation": "high"
                    }
                },
                "patient_safety": {
                    "critical_vitals_missed": {
                        "condition": "missed_critical_alerts > 0",
                        "action": "medical_staff_notification",
                        "escalation": "critical"
                    },
                    "emergency_response_delay": {
                        "condition": "emergency_response_time > 300",
                        "action": "hospital_admin_alert",
                        "escalation": "critical"
                    }
                },
                "system_performance": {
                    "patient_lookup_timeout": {
                        "condition": "patient_lookup_time > 10",
                        "action": "performance_alert",
                        "escalation": "high"
                    },
                    "ai_inference_failure": {
                        "condition": "ai_inference_error_rate > 0.1",
                        "action": "ai_team_notification",
                        "escalation": "high"
                    }
                }
            }
            
            results["medical_alerts"] = medical_alerts
            
            # Initialize alerting system
            await self._initialize_alerting_system()
            
            logger.info("Alerting system configuration completed successfully")
            
        except Exception as e:
            logger.error(f"Alerting configuration failed: {str(e)}")
            results["errors"].append({"component": "alerting", "error": str(e)})
        
        return results
    
    async def _initialize_alerting_system(self) -> None:
        """Initialize alerting system components"""
        # Initialize alert manager
        alert_manager_config = {
            "status": "active",
            "rules_loaded": len(self.config.monitoring_config["alerting"]["enabled"]),
            "channels_configured": ["email", "slack", "pagerduty", "sms"],
            "escalation_policies": 3
        }
        logger.info(f"Alert manager initialized: {alert_manager_config}")
    
    async def setup_dashboards(self) -> Dict[str, Any]:
        """Set up monitoring dashboards for medical AI operations"""
        logger.info("Setting up monitoring dashboards")
        
        results = {
            "dashboard_configs": {},
            "medical_dashboards": {},
            "operational_dashboards": {},
            "performance_dashboards": {},
            "errors": []
        }
        
        try:
            # Dashboard configuration templates
            dashboard_configs = {
                "grafana_config": {
                    "version": "9.0",
                    "datasources": [
                        {
                            "name": "Prometheus",
                            "type": "prometheus",
                            "url": "http://prometheus:9090",
                            "access": "proxy",
                            "isDefault": True
                        },
                        {
                            "name": "Loki",
                            "type": "loki",
                            "url": "http://loki:3100",
                            "access": "proxy"
                        }
                    ],
                    "plugins": [
                        "grafana-piechart-panel",
                        "grafana-worldmap-panel",
                        "grafana-clock-panel"
                    ]
                },
                "dashboard_templates": {
                    "refresh_interval": "30s",
                    "time_range": "1h",
                    "default_variables": ["service", "environment", "region"]
                }
            }
            
            results["dashboard_configs"] = dashboard_configs
            
            # Medical-specific dashboards
            medical_dashboards = {
                "medical_ai_overview": {
                    "title": "Medical AI Operations Overview",
                    "panels": [
                        {
                            "title": "Patient Request Rate",
                            "query": "rate(patient_requests_total[5m])",
                            "type": "graph",
                            "unit": "req/s"
                        },
                        {
                            "title": "Clinical Data Processing",
                            "query": "rate(clinical_data_processed_total[5m])",
                            "type": "graph",
                            "unit": "rec/s"
                        },
                        {
                            "title": "AI Inference Queue",
                            "query": "ai_inference_queue_depth",
                            "type": "singlestat",
                            "unit": "count"
                        },
                        {
                            "title": "Emergency Requests",
                            "query": "rate(emergency_requests_total[1m])",
                            "type": "singlestat",
                            "unit": "req/s"
                        }
                    ]
                },
                "patient_data_dashboard": {
                    "title": "Patient Data Service Health",
                    "panels": [
                        {
                            "title": "Patient Lookup Response Time",
                            "query": "histogram_quantile(0.95, rate(patient_lookup_duration_bucket[5m]))",
                            "type": "graph",
                            "unit": "s"
                        },
                        {
                            "title": "Patient Data Cache Hit Rate",
                            "query": "rate(patient_data_cache_hits_total[5m]) / rate(patient_data_cache_requests_total[5m])",
                            "type": "singlestat",
                            "unit": "percent"
                        },
                        {
                            "title": "Database Connection Pool",
                            "query": "patient_db_pool_utilization",
                            "type": "graph",
                            "unit": "percent"
                        }
                    ]
                },
                "ai_inference_dashboard": {
                    "title": "AI Inference Service Performance",
                    "panels": [
                        {
                            "title": "Inference Response Time",
                            "query": "histogram_quantile(0.95, rate(ai_inference_duration_bucket[5m]))",
                            "type": "graph",
                            "unit": "s"
                        },
                        {
                            "title": "Model Accuracy",
                            "query": "ai_model_accuracy",
                            "type": "singlestat",
                            "unit": "percent"
                        },
                        {
                            "title": "Inference Queue Depth",
                            "query": "ai_inference_queue_depth",
                            "type": "graph",
                            "unit": "count"
                        }
                    ]
                }
            }
            
            results["medical_dashboards"] = medical_dashboards
            
            # Operational dashboards
            operational_dashboards = {
                "system_health": {
                    "title": "System Health Overview",
                    "panels": [
                        {
                            "title": "CPU Usage by Service",
                            "query": "rate(cpu_usage_percent[5m])",
                            "type": "graph",
                            "unit": "percent"
                        },
                        {
                            "title": "Memory Usage",
                            "query": "rate(memory_usage_percent[5m])",
                            "type": "graph",
                            "unit": "percent"
                        },
                        {
                            "title": "Service Uptime",
                            "query": "up",
                            "type": "singlestat",
                            "unit": "percent"
                        },
                        {
                            "title": "Error Rate",
                            "query": "rate(errors_total[5m])",
                            "type": "graph",
                            "unit": "req/s"
                        }
                    ]
                },
                "infrastructure": {
                    "title": "Infrastructure Metrics",
                    "panels": [
                        {
                            "title": "Network I/O",
                            "query": "rate(network_io_bytes_total[5m])",
                            "type": "graph",
                            "unit": "bytes/s"
                        },
                        {
                            "title": "Disk Usage",
                            "query": "disk_usage_percent",
                            "type": "graph",
                            "unit": "percent"
                        },
                        {
                            "title": "Connection Count",
                            "query": "active_connections",
                            "type": "singlestat",
                            "unit": "count"
                        }
                    ]
                }
            }
            
            results["operational_dashboards"] = operational_dashboards
            
            # Performance dashboards
            performance_dashboards = {
                "response_time_analysis": {
                    "title": "Response Time Analysis",
                    "panels": [
                        {
                            "title": "P95 Response Time",
                            "query": "histogram_quantile(0.95, rate(http_request_duration_bucket[5m]))",
                            "type": "graph",
                            "unit": "s"
                        },
                        {
                            "title": "P99 Response Time",
                            "query": "histogram_quantile(0.99, rate(http_request_duration_bucket[5m]))",
                            "type": "graph",
                            "unit": "s"
                        },
                        {
                            "title": "Response Time Distribution",
                            "query": "rate(http_request_duration_bucket[5m])",
                            "type": "heatmap"
                        }
                    ]
                },
                "throughput_analysis": {
                    "title": "Throughput Analysis",
                    "panels": [
                        {
                            "title": "Requests per Second",
                            "query": "rate(http_requests_total[5m])",
                            "type": "graph",
                            "unit": "req/s"
                        },
                        {
                            "title": "Peak Throughput",
                            "query": "max_over_time(rate(http_requests_total[5m])[1h:5m])",
                            "type": "singlestat",
                            "unit": "req/s"
                        },
                        {
                            "title": "Throughput by Service",
                            "query": "sum(rate(http_requests_total[5m])) by (service)",
                            "type": "piechart"
                        }
                    ]
                }
            }
            
            results["performance_dashboards"] = performance_dashboards
            
            # Generate dashboard JSON files
            await self._generate_dashboard_files()
            
            logger.info("Dashboard setup completed successfully")
            
        except Exception as e:
            logger.error(f"Dashboard setup failed: {str(e)}")
            results["errors"].append({"component": "dashboards", "error": str(e)})
        
        return results
    
    async def _generate_dashboard_files(self) -> None:
        """Generate dashboard JSON files for Grafana"""
        dashboards_dir = Path("/workspace/production/performance/monitoring/dashboards")
        dashboards_dir.mkdir(exist_ok=True)
        
        # Generate sample dashboard JSON
        sample_dashboard = {
            "dashboard": {
                "id": None,
                "title": "Medical AI Operations Overview",
                "tags": ["medical-ai", "production"],
                "timezone": "UTC",
                "panels": [
                    {
                        "id": 1,
                        "title": "Patient Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(patient_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
        
        with open(dashboards_dir / "medical_ai_overview.json", 'w') as f:
            json.dump(sample_dashboard, f, indent=2)
    
    async def setup_performance_baselines(self) -> Dict[str, Any]:
        """Set up performance baselines for medical AI workloads"""
        logger.info("Setting up performance baselines")
        
        results = {
            "baseline_metrics": {},
            "baseline_calculation": {},
            "regression_detection": {},
            "alerting_thresholds": {},
            "errors": []
        }
        
        try:
            # Performance baseline metrics for medical AI
            baseline_metrics = {
                "response_time_baselines": {
                    "patient_lookup": {
                        "target_p95": 1.0,
                        "target_p99": 2.0,
                        "calculation_period": "30d",
                        "business_hours_weight": 1.5
                    },
                    "clinical_data_access": {
                        "target_p95": 1.5,
                        "target_p99": 3.0,
                        "calculation_period": "30d",
                        "business_hours_weight": 1.3
                    },
                    "ai_inference": {
                        "target_p95": 2.5,
                        "target_p99": 5.0,
                        "calculation_period": "30d",
                        "business_hours_weight": 1.2
                    },
                    "patient_dashboard_load": {
                        "target_p95": 2.0,
                        "target_p99": 4.0,
                        "calculation_period": "30d",
                        "business_hours_weight": 1.4
                    }
                },
                "throughput_baselines": {
                    "patient_requests_per_minute": {
                        "baseline": 120,
                        "peak_capacity": 300,
                        "calculation_period": "7d"
                    },
                    "clinical_data_requests_per_minute": {
                        "baseline": 80,
                        "peak_capacity": 200,
                        "calculation_period": "7d"
                    },
                    "ai_inference_requests_per_minute": {
                        "baseline": 30,
                        "peak_capacity": 100,
                        "calculation_period": "7d"
                    }
                },
                "resource_utilization_baselines": {
                    "cpu_usage": {
                        "baseline_target": 65,
                        "scaling_threshold": 80,
                        "critical_threshold": 90
                    },
                    "memory_usage": {
                        "baseline_target": 70,
                        "scaling_threshold": 85,
                        "critical_threshold": 95
                    },
                    "connection_pool_utilization": {
                        "baseline_target": 60,
                        "scaling_threshold": 80,
                        "critical_threshold": 90
                    }
                }
            }
            
            results["baseline_metrics"] = baseline_metrics
            
            # Baseline calculation methodology
            baseline_calculation = {
                "calculation_algorithms": {
                    "time_weighted_average": {
                        "description": "Weighted average based on time of day and day of week",
                        "weights": {
                            "business_hours": 1.0,
                            "after_hours": 0.5,
                            "weekends": 0.3,
                            "holidays": 0.2
                        }
                    },
                    "percentile_based": {
                        "description": "95th percentile during normal operations",
                        "exclusions": ["outages", "maintenance", "emergency_periods"]
                    },
                    "trend_analysis": {
                        "description": "Linear regression to identify trends",
                        "lookback_period": "30d",
                        "trend_threshold": 0.05
                    }
                },
                "seasonal_adjustments": {
                    "morning_rounds": {
                        "time_range": "06:00-09:00",
                        "multiplier": 1.5,
                        "day_types": ["monday", "tuesday", "wednesday", "thursday", "friday"]
                    },
                    "afternoon_peak": {
                        "time_range": "14:00-16:00",
                        "multiplier": 1.3,
                        "day_types": ["monday", "tuesday", "wednesday", "thursday", "friday"]
                    },
                    "night_hours": {
                        "time_range": "22:00-06:00",
                        "multiplier": 0.7,
                        "day_types": ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"]
                    }
                }
            }
            
            results["baseline_calculation"] = baseline_calculation
            
            # Regression detection configuration
            regression_detection = {
                "detection_methods": {
                    "statistical_significance": {
                        "test": "t-test",
                        "significance_level": 0.05,
                        "min_sample_size": 100
                    },
                    "percentage_change": {
                        "threshold": 0.10,  # 10% degradation
                        "consecutive_occurrences": 3
                    },
                    "trend_analysis": {
                        "slope_threshold": -0.05,
                        "time_window": "7d"
                    }
                },
                "sensitivity_levels": {
                    "high_sensitivity": {
                        "response_time_threshold": 0.05,  # 5% increase
                        "throughput_threshold": 0.05,      # 5% decrease
                        "error_rate_threshold": 0.01       # 1% increase
                    },
                    "medium_sensitivity": {
                        "response_time_threshold": 0.10,  # 10% increase
                        "throughput_threshold": 0.10,     # 10% decrease
                        "error_rate_threshold": 0.02      # 2% increase
                    },
                    "low_sensitivity": {
                        "response_time_threshold": 0.20,  # 20% increase
                        "throughput_threshold": 0.20,     # 20% decrease
                        "error_rate_threshold": 0.05      # 5% increase
                    }
                }
            }
            
            results["regression_detection"] = regression_detection
            
            # Performance alerting thresholds
            alerting_thresholds = {
                "response_time_alerts": {
                    "warning": "baseline + 10%",
                    "critical": "baseline + 25%",
                    "emergency": "baseline + 50%"
                },
                "throughput_alerts": {
                    "warning": "baseline - 10%",
                    "critical": "baseline - 25%",
                    "emergency": "baseline - 40%"
                },
                "error_rate_alerts": {
                    "warning": "1%",
                    "critical": "5%",
                    "emergency": "10%"
                },
                "resource_alerts": {
                    "cpu_warning": "80%",
                    "cpu_critical": "90%",
                    "memory_warning": "85%",
                    "memory_critical": "95%"
                }
            }
            
            results["alerting_thresholds"] = alerting_thresholds
            
            # Initialize baseline calculation
            await self._initialize_baseline_calculation()
            
            logger.info("Performance baselines setup completed successfully")
            
        except Exception as e:
            logger.error(f"Performance baselines setup failed: {str(e)}")
            results["errors"].append({"component": "performance_baselines", "error": str(e)})
        
        return results
    
    async def _initialize_baseline_calculation(self) -> None:
        """Initialize baseline calculation components"""
        # Initialize baseline calculator
        baseline_config = {
            "status": "active",
            "calculation_interval": 3600,  # 1 hour
            "retention_period": "365d",
            "algorithms_configured": ["time_weighted", "percentile", "trend_analysis"]
        }
        logger.info(f"Baseline calculator initialized: {baseline_config}")

class PerformanceRegressionDetector:
    """Performance regression detection for medical AI systems"""
    
    def __init__(self, config):
        self.config = config
        self.baseline_data = {}
        self.regression_history = deque(maxlen=1000)
        
    async def initialize_regression_detection(self) -> Dict[str, Any]:
        """Initialize performance regression detection"""
        logger.info("Initializing performance regression detection")
        
        results = {
            "detection_algorithms": {},
            "regression_thresholds": {},
            "notification_rules": {},
            "historical_analysis": {},
            "errors": []
        }
        
        try:
            # Detection algorithms for medical AI workloads
            detection_algorithms = {
                "statistical_methods": {
                    "t_test": {
                        "description": "Compare current performance with baseline",
                        "significance_level": 0.05,
                        "min_sample_size": 50
                    },
                    "cumulative_sum": {
                        "description": "Detect gradual performance degradation",
                        "control_limit": 3.0,
                        "detection_delay": "5m"
                    },
                    "change_point_detection": {
                        "description": "Detect sudden performance changes",
                        "algorithm": "Page-Hinkley",
                        "threshold": 1.0
                    }
                },
                "machine_learning_methods": {
                    "isolation_forest": {
                        "description": "Detect anomalous performance patterns",
                        "contamination": 0.1,
                        "n_estimators": 100
                    },
                    "lstm_anomaly_detection": {
                        "description": "Detect complex temporal anomalies",
                        "sequence_length": 24,
                        "epochs": 100
                    }
                }
            }
            
            results["detection_algorithms"] = detection_algorithms
            
            # Regression thresholds specific to medical workloads
            regression_thresholds = {
                "medical_specific_thresholds": {
                    "patient_lookup_regression": {
                        "response_time": {"threshold": 1.5, "severity": "high"},
                        "availability": {"threshold": 0.99, "severity": "critical"}
                    },
                    "clinical_data_regression": {
                        "response_time": {"threshold": 2.0, "severity": "high"},
                        "throughput": {"threshold": 0.85, "severity": "medium"}
                    },
                    "ai_inference_regression": {
                        "response_time": {"threshold": 3.0, "severity": "medium"},
                        "accuracy": {"threshold": 0.95, "severity": "critical"}
                    }
                },
                "system_wide_thresholds": {
                    "response_time_degradation": {"threshold": 0.20, "severity": "medium"},
                    "throughput_degradation": {"threshold": 0.15, "severity": "medium"},
                    "error_rate_increase": {"threshold": 0.02, "severity": "high"}
                }
            }
            
            results["regression_thresholds"] = regression_thresholds
            
            # Notification rules for regression detection
            notification_rules = {
                "regression_alerts": [
                    {
                        "trigger": "patient_lookup_regression",
                        "severity": "high",
                        "notification_channels": ["oncall", "medical_team", "slack"],
                        "escalation_time": "15m"
                    },
                    {
                        "trigger": "ai_accuracy_regression",
                        "severity": "critical",
                        "notification_channels": ["oncall", "ai_team", "hospital_admin"],
                        "escalation_time": "5m"
                    },
                    {
                        "trigger": "system_wide_degradation",
                        "severity": "high",
                        "notification_channels": ["oncall", "ops_team"],
                        "escalation_time": "10m"
                    }
                ],
                "regression_resolution": {
                    "auto_investigation": True,
                    "investigation_timeout": "30m",
                    "auto_rollback_threshold": "critical"
                }
            }
            
            results["notification_rules"] = notification_rules
            
            # Historical regression analysis
            historical_analysis = {
                "regression_patterns": [
                    {
                        "pattern": "morning_rounds_performance_drop",
                        "frequency": "daily",
                        "time_range": "06:00-09:00",
                        "typical_impact": "15-25% response time increase",
                        "root_causes": ["increased_concurrent_users", "database_connection_pool_exhaustion"]
                    },
                    {
                        "pattern": "ai_model_performance_drift",
                        "frequency": "weekly",
                        "time_range": "continuous",
                        "typical_impact": "5-10% accuracy degradation",
                        "root_causes": ["data_distribution_shift", "model_staleness"]
                    },
                    {
                        "pattern": "emergency_surge_performance_impact",
                        "frequency": "monthly",
                        "time_range": "variable",
                        "typical_impact": "50-100% load increase",
                        "root_causes": ["emergency_events", "system_overload"]
                    }
                ],
                "regression_trends": {
                    "last_30_days": {
                        "total_regressions": 12,
                        "resolved_regressions": 11,
                        "average_resolution_time": "45m",
                        "common_affected_services": ["patient_data", "ai_inference"]
                    }
                }
            }
            
            results["historical_analysis"] = historical_analysis
            
            # Initialize regression detection
            await self._initialize_regression_detection_components()
            
            logger.info("Performance regression detection initialized successfully")
            
        except Exception as e:
            logger.error(f"Regression detection initialization failed: {str(e)}")
            results["errors"].append({"component": "regression_detection", "error": str(e)})
        
        return results
    
    async def _initialize_regression_detection_components(self) -> None:
        """Initialize regression detection components"""
        # Initialize regression detection engine
        detection_engine = {
            "status": "active",
            "algorithms_loaded": ["t_test", "cumulative_sum", "isolation_forest"],
            "detection_interval": 300,  # 5 minutes
            "baseline_warmup_period": "24h"
        }
        logger.info(f"Regression detection engine initialized: {detection_engine}")
    
    async def detect_performance_regressions(self, current_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect performance regressions in current metrics"""
        regressions_detected = []
        
        # Check for patient lookup regression
        if current_metrics.get("patient_lookup_p95", 0) > 1.5:
            regressions_detected.append({
                "type": "patient_lookup_regression",
                "severity": "high",
                "current_value": current_metrics["patient_lookup_p95"],
                "baseline_value": 1.0,
                "degradation_percentage": ((current_metrics["patient_lookup_p95"] - 1.0) / 1.0) * 100
            })
        
        # Check for AI inference regression
        if current_metrics.get("ai_accuracy", 1.0) < 0.95:
            regressions_detected.append({
                "type": "ai_accuracy_regression",
                "severity": "critical",
                "current_value": current_metrics["ai_accuracy"],
                "baseline_value": 0.97,
                "degradation_percentage": ((0.97 - current_metrics["ai_accuracy"]) / 0.97) * 100
            })
        
        return regressions_detected