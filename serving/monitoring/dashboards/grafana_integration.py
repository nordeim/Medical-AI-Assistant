"""
Grafana Dashboard Integration

Provides comprehensive dashboard integration with Grafana for real-time monitoring,
alert visualization, and medical AI specific metrics display.
"""

import asyncio
import base64
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import aiohttp
import structlog
import yaml

from ...config.logging_config import get_logger

logger = structlog.get_logger("grafana_integration")


class GrafanaDashboardManager:
    """Manages Grafana dashboards for medical AI monitoring."""
    
    def __init__(self, 
                 grafana_config: Dict[str, Any],
                 prometheus_config: Dict[str, Any] = None):
        
        self.grafana_config = grafana_config
        self.prometheus_config = prometheus_config or {}
        
        # Grafana API settings
        self.base_url = grafana_config.get('url', 'http://localhost:3000')
        self.api_key = grafana_config.get('api_key')
        self.username = grafana_config.get('username')
        self.password = grafana_config.get('password')
        self.org_id = grafana_config.get('org_id', 1)
        
        # Dashboard settings
        self.dashboard_folder = grafana_config.get('dashboard_folder', 'Medical AI')
        self.refresh_interval = grafana_config.get('refresh_interval', '30s')
        
        # Dashboard storage
        self.dashboard_templates: Dict[str, Dict[str, Any]] = {}
        self.deployed_dashboards: Dict[str, str] = {}  # dashboard_name -> dashboard_uid
        
        # Headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            self.headers['Authorization'] = f'Bearer {self.api_key}'
        
        self.logger = structlog.get_logger("grafana_manager")
        
        # Load dashboard templates
        self._load_dashboard_templates()
        
        self.logger.info("GrafanaDashboardManager initialized")
    
    def _load_dashboard_templates(self):
        """Load predefined dashboard templates."""
        
        # System Overview Dashboard
        self.dashboard_templates['system_overview'] = self._create_system_overview_dashboard()
        
        # Model Performance Dashboard  
        self.dashboard_templates['model_performance'] = self._create_model_performance_dashboard()
        
        # Clinical Outcomes Dashboard
        self.dashboard_templates['clinical_outcomes'] = self._create_clinical_outcomes_dashboard()
        
        # Alert Management Dashboard
        self.dashboard_templates['alert_management'] = self._create_alert_management_dashboard()
        
        # Regulatory Compliance Dashboard
        self.dashboard_templates['regulatory_compliance'] = self._create_regulatory_compliance_dashboard()
        
        # Medical AI Executive Dashboard
        self.dashboard_templates['executive_dashboard'] = self._create_executive_dashboard()
        
        logger.info("Dashboard templates loaded", 
                   count=len(self.dashboard_templates))
    
    async def create_or_update_dashboard(self, 
                                       dashboard_name: str,
                                       custom_config: Dict[str, Any] = None) -> bool:
        """Create or update a Grafana dashboard."""
        
        try:
            if dashboard_name not in self.dashboard_templates:
                logger.error("Dashboard template not found", 
                           dashboard_name=dashboard_name)
                return False
            
            # Get template
            dashboard_config = self.dashboard_templates[dashboard_name].copy()
            
            # Apply custom configuration
            if custom_config:
                dashboard_config = self._merge_config(dashboard_config, custom_config)
            
            # Update dashboard metadata
            dashboard_config['title'] = custom_config.get('title', dashboard_config['title'])
            dashboard_config['uid'] = self.deployed_dashboards.get(dashboard_name)
            
            if not dashboard_config['uid']:
                dashboard_config['uid'] = self._generate_uid()
            
            # Create or update dashboard
            success = await self._deploy_dashboard(dashboard_config)
            
            if success:
                self.deployed_dashboards[dashboard_name] = dashboard_config['uid']
                logger.info("Dashboard deployed successfully",
                          dashboard_name=dashboard_name,
                          uid=dashboard_config['uid'])
            
            return success
            
        except Exception as e:
            logger.error("Dashboard deployment failed",
                        dashboard_name=dashboard_name,
                        error=str(e))
            return False
    
    async def _deploy_dashboard(self, dashboard_config: Dict[str, Any]) -> bool:
        """Deploy dashboard to Grafana."""
        
        try:
            # Ensure folder exists
            await self._ensure_folder_exists(self.dashboard_folder)
            
            # Prepare dashboard payload
            payload = {
                'dashboard': dashboard_config,
                'overwrite': True,
                'folderId': await self._get_folder_id(self.dashboard_folder),
                'message': f'Updated Medical AI dashboard: {dashboard_config["title"]}'
            }
            
            # API endpoint for creating/updating dashboards
            url = f"{self.base_url}/api/dashboards/db"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        dashboard_config['uid'] = result.get('uid')
                        return True
                    else:
                        error_text = await response.text()
                        logger.error("Grafana API error",
                                   status=response.status,
                                   response=error_text)
                        return False
                        
        except Exception as e:
            logger.error("Dashboard deployment exception", error=str(e))
            return False
    
    async def _ensure_folder_exists(self, folder_name: str):
        """Ensure Grafana folder exists."""
        
        try:
            # Check if folder exists
            url = f"{self.base_url}/api/folders"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None
                ) as response:
                    if response.status == 200:
                        folders = await response.json()
                        for folder in folders:
                            if folder.get('title') == folder_name:
                                return folder.get('id')
            
            # Create folder if it doesn't exist
            folder_payload = {
                'title': folder_name,
                'uid': self._generate_uid()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/folders",
                    json=folder_payload,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('id')
                    else:
                        logger.warning("Failed to create folder", 
                                     status=response.status)
                        return 0  # Default to root folder
                        
        except Exception as e:
            logger.error("Folder management error", error=str(e))
            return 0
    
    async def _get_folder_id(self, folder_name: str) -> int:
        """Get folder ID by name."""
        
        try:
            url = f"{self.base_url}/api/folders"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None
                ) as response:
                    if response.status == 200:
                        folders = await response.json()
                        for folder in folders:
                            if folder.get('title') == folder_name:
                                return folder.get('id')
            
            # Return root folder ID if not found
            return 0
            
        except Exception as e:
            logger.error("Failed to get folder ID", error=str(e))
            return 0
    
    def _get_auth(self) -> Optional[aiohttp.BasicAuth]:
        """Get HTTP basic auth if configured."""
        if self.username and self.password:
            return aiohttp.BasicAuth(self.username, self.password)
        return None
    
    def _generate_uid(self) -> str:
        """Generate a unique dashboard UID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _merge_config(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge custom configuration with base template."""
        
        merged = base_config.copy()
        
        # Merge top-level properties
        for key, value in custom_config.items():
            if key in ['title', 'description', 'tags', 'timezone']:
                merged[key] = value
        
        # Merge panels if provided
        if 'panels' in custom_config:
            merged['panels'] = custom_config['panels']
        
        # Merge time range
        if 'time' in custom_config:
            merged['time'] = custom_config['time']
        
        # Merge refresh intervals
        if 'refresh' in custom_config:
            merged['refresh'] = custom_config['refresh']
        
        return merged
    
    # Dashboard template creation methods
    
    def _create_system_overview_dashboard(self) -> Dict[str, Any]:
        """Create system overview dashboard."""
        
        return {
            'title': 'Medical AI - System Overview',
            'description': 'System resource monitoring and health overview',
            'tags': ['medical-ai', 'system', 'overview'],
            'timezone': 'browser',
            'refresh': self.refresh_interval,
            'time': {
                'from': 'now-1h',
                'to': 'now'
            },
            'panels': [
                # CPU Usage Panel
                {
                    'id': 1,
                    'title': 'CPU Usage',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_system_cpu_usage_percent',
                            'legendFormat': 'CPU Usage',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 0, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percent',
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 70},
                                    {'color': 'red', 'value': 90}
                                ]
                            }
                        }
                    }
                },
                
                # Memory Usage Panel
                {
                    'id': 2,
                    'title': 'Memory Usage',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_system_memory_usage_percent',
                            'legendFormat': 'Memory Usage',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 6, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percent',
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 75},
                                    {'color': 'red', 'value': 90}
                                ]
                            }
                        }
                    }
                },
                
                # GPU Memory Panel
                {
                    'id': 3,
                    'title': 'GPU Memory',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_system_gpu_memory_usage_percent',
                            'legendFormat': 'GPU Memory',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 12, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percent',
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 80},
                                    {'color': 'red', 'value': 95}
                                ]
                            }
                        }
                    }
                },
                
                # Active Alerts Panel
                {
                    'id': 4,
                    'title': 'Active Alerts',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_sla_violations_total',
                            'legendFormat': 'Active Alerts',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 18, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 1},
                                    {'color': 'red', 'value': 5}
                                ]
                            }
                        }
                    }
                },
                
                # System Metrics Graph
                {
                    'id': 5,
                    'title': 'System Metrics Over Time',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'medical_ai_system_cpu_usage_percent',
                            'legendFormat': 'CPU %',
                            'refId': 'A'
                        },
                        {
                            'expr': 'medical_ai_system_memory_usage_percent',
                            'legendFormat': 'Memory %',
                            'refId': 'B'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 8}
                }
            ]
        }
    
    def _create_model_performance_dashboard(self) -> Dict[str, Any]:
        """Create model performance dashboard."""
        
        return {
            'title': 'Medical AI - Model Performance',
            'description': 'Model accuracy, latency, and drift monitoring',
            'tags': ['medical-ai', 'model', 'performance'],
            'timezone': 'browser',
            'refresh': self.refresh_interval,
            'time': {
                'from': 'now-6h',
                'to': 'now'
            },
            'panels': [
                # Model Accuracy Panel
                {
                    'id': 1,
                    'title': 'Model Accuracy',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_model_accuracy_score',
                            'legendFormat': '{{model_id}} - Accuracy',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 0, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.8},
                                    {'color': 'green', 'value': 0.95}
                                ]
                            }
                        }
                    }
                },
                
                # Average Latency Panel
                {
                    'id': 2,
                    'title': 'Average Latency',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_inference_duration_seconds_sum[5m]) / rate(medical_ai_inference_duration_seconds_count[5m]) * 1000',
                            'legendFormat': '{{model_id}} - Latency (ms)',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 8, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'ms',
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 1000},
                                    {'color': 'red', 'value': 2000}
                                ]
                            }
                        }
                    }
                },
                
                # Requests Per Second Panel
                {
                    'id': 3,
                    'title': 'Requests Per Second',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_inference_requests_total[5m])',
                            'legendFormat': '{{model_id}} - RPS',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 16, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'ops',
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 1},
                                    {'color': 'green', 'value': 10}
                                ]
                            }
                        }
                    }
                },
                
                # Accuracy Over Time
                {
                    'id': 4,
                    'title': 'Model Accuracy Trends',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'medical_ai_model_accuracy_score',
                            'legendFormat': '{{model_id}} - Accuracy',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
                },
                
                # Latency Over Time
                {
                    'id': 5,
                    'title': 'Latency Trends',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_inference_duration_seconds_sum[5m]) / rate(medical_ai_inference_duration_seconds_count[5m]) * 1000',
                            'legendFormat': '{{model_id}} - Latency (ms)',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
                },
                
                # Error Rate Panel
                {
                    'id': 6,
                    'title': 'Error Rate',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_inference_errors_total[5m]) / rate(medical_ai_inference_requests_total[5m]) * 100',
                            'legendFormat': '{{model_id}} - Error Rate %',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 16}
                },
                
                # Drift Detection Panel
                {
                    'id': 7,
                    'title': 'Model Drift Detection',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'medical_ai_model_drift_score{drift_type="data"}',
                            'legendFormat': '{{model_id}} - Data Drift',
                            'refId': 'A'
                        },
                        {
                            'expr': 'medical_ai_model_drift_score{drift_type="concept"}',
                            'legendFormat': '{{model_id}} - Concept Drift',
                            'refId': 'B'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 16}
                }
            ]
        }
    
    def _create_clinical_outcomes_dashboard(self) -> Dict[str, Any]:
        """Create clinical outcomes dashboard."""
        
        return {
            'title': 'Medical AI - Clinical Outcomes',
            'description': 'Clinical effectiveness and outcome tracking',
            'tags': ['medical-ai', 'clinical', 'outcomes'],
            'timezone': 'browser',
            'refresh': self.refresh_interval,
            'time': {
                'from': 'now-24h',
                'to': 'now'
            },
            'panels': [
                # Clinical Effectiveness Panel
                {
                    'id': 1,
                    'title': 'Clinical Effectiveness',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_clinical_effectiveness_score',
                            'legendFormat': '{{model_id}} - {{specialty}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 0, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.7},
                                    {'color': 'green', 'value': 0.9}
                                ]
                            }
                        }
                    }
                },
                
                # Medical Relevance Panel
                {
                    'id': 2,
                    'title': 'Medical Relevance',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_medical_relevance_score',
                            'legendFormat': '{{model_id}} - {{context_type}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 8, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.6},
                                    {'color': 'green', 'value': 0.8}
                                ]
                            }
                        }
                    }
                },
                
                # Safety Score Panel
                {
                    'id': 3,
                    'title': 'Safety Score',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_safety_score',
                            'legendFormat': '{{model_id}} - {{risk_level}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 16, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.7},
                                    {'color': 'green', 'value': 0.95}
                                ]
                            }
                        }
                    }
                },
                
                # Clinical Outcomes Over Time
                {
                    'id': 4,
                    'title': 'Clinical Effectiveness Trends',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'medical_ai_clinical_effectiveness_score',
                            'legendFormat': '{{model_id}} - {{specialty}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
                },
                
                # Outcome Distribution
                {
                    'id': 5,
                    'title': 'Clinical Outcome Distribution',
                    'type': 'piechart',
                    'targets': [
                        {
                            'expr': 'medical_ai_clinical_outcomes_total',
                            'legendFormat': '{{model_id}} - {{outcome_type}} - {{result}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
                },
                
                # Bias Detection Panel
                {
                    'id': 6,
                    'title': 'Bias Detection',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'medical_ai_bias_score',
                            'legendFormat': '{{model_id}} - {{demographic_type}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 16}
                },
                
                # Physician Feedback Panel
                {
                    'id': 7,
                    'title': 'Physician Feedback',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_physician_feedback_score',
                            'legendFormat': '{{model_id}} - {{feedback_type}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 16}
                }
            ]
        }
    
    def _create_alert_management_dashboard(self) -> Dict[str, Any]:
        """Create alert management dashboard."""
        
        return {
            'title': 'Medical AI - Alert Management',
            'description': 'Alert tracking and management interface',
            'tags': ['medical-ai', 'alerts', 'management'],
            'timezone': 'browser',
            'refresh': self.refresh_interval,
            'time': {
                'from': 'now-1h',
                'to': 'now'
            },
            'panels': [
                # Active Alerts Count
                {
                    'id': 1,
                    'title': 'Active Alerts',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_sla_violations_total',
                            'legendFormat': 'Total Active Alerts',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 0, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 1},
                                    {'color': 'red', 'value': 5}
                                ]
                            }
                        }
                    }
                },
                
                # Critical Alerts
                {
                    'id': 2,
                    'title': 'Critical Alerts',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_sla_violations_total{sla_type="critical"}',
                            'legendFormat': 'Critical Alerts',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 6, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 1},
                                    {'color': 'red', 'value': 3}
                                ]
                            }
                        }
                    }
                },
                
                # SLA Compliance
                {
                    'id': 3,
                    'title': 'SLA Compliance',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_sla_compliance_percent',
                            'legendFormat': '{{sla_type}} - {{time_window}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 12, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.95},
                                    {'color': 'green', 'value': 0.99}
                                ]
                            }
                        }
                    }
                },
                
                # Service Availability
                {
                    'id': 4,
                    'title': 'Service Availability',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_service_availability_percent',
                            'legendFormat': '{{service_name}} - {{time_window}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 18, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.99},
                                    {'color': 'green', 'value': 0.999}
                                ]
                            }
                        }
                    }
                },
                
                # Alert Timeline
                {
                    'id': 5,
                    'title': 'Alert Timeline',
                    'type': 'table',
                    'targets': [
                        {
                            'expr': 'medical_ai_sla_violations_total',
                            'legendFormat': '{{sla_type}} - {{severity}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 8},
                    'transformations': [
                        {
                            'id': 'organize',
                            'options': {
                                'excludeByName': {},
                                'indexByName': {},
                                'renameByName': {}
                            }
                        }
                    ]
                },
                
                # SLA Violations Over Time
                {
                    'id': 6,
                    'title': 'SLA Violations Over Time',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_sla_violations_total[5m])',
                            'legendFormat': '{{sla_type}} - {{severity}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 16}
                },
                
                # Compliance Trends
                {
                    'id': 7,
                    'title': 'SLA Compliance Trends',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'medical_ai_sla_compliance_percent',
                            'legendFormat': '{{sla_type}} - {{time_window}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 16}
                }
            ]
        }
    
    def _create_regulatory_compliance_dashboard(self) -> Dict[str, Any]:
        """Create regulatory compliance dashboard."""
        
        return {
            'title': 'Medical AI - Regulatory Compliance',
            'description': 'HIPAA, FDA, and other regulatory compliance monitoring',
            'tags': ['medical-ai', 'compliance', 'regulatory'],
            'timezone': 'browser',
            'refresh': self.refresh_interval,
            'time': {
                'from': 'now-24h',
                'to': 'now'
            },
            'panels': [
                # HIPAA Compliance
                {
                    'id': 1,
                    'title': 'HIPAA Compliance',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_hipaa_compliance_checks_total{result="pass"} / (medical_ai_hipaa_compliance_checks_total{result="pass"} + medical_ai_hipaa_compliance_checks_total{result="fail"})',
                            'legendFormat': 'HIPAA Compliance Rate',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 0, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.95},
                                    {'color': 'green', 'value': 0.99}
                                ]
                            }
                        }
                    }
                },
                
                # FDA Compliance
                {
                    'id': 2,
                    'title': 'FDA SaMD Compliance',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_regulatory_validations_total{regulation_type="fda", validation_result="pass"} / medical_ai_regulatory_validations_total{regulation_type="fda"}',
                            'legendFormat': 'FDA Compliance Rate',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 8, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.9},
                                    {'color': 'green', 'value': 0.99}
                                ]
                            }
                        }
                    }
                },
                
                # Data Privacy Violations
                {
                    'id': 3,
                    'title': 'Data Privacy Violations',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_data_privacy_violations_total',
                            'legendFormat': 'Privacy Violations',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 16, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'thresholds': {
                                'steps': [
                                    {'color': 'green', 'value': 0},
                                    {'color': 'yellow', 'value': 1},
                                    {'color': 'red', 'value': 5}
                                ]
                            }
                        }
                    }
                },
                
                # Compliance Checks Over Time
                {
                    'id': 4,
                    'title': 'Compliance Checks Over Time',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_hipaa_compliance_checks_total[5m])',
                            'legendFormat': 'HIPAA Checks',
                            'refId': 'A'
                        },
                        {
                            'expr': 'rate(medical_ai_regulatory_validations_total[5m])',
                            'legendFormat': 'Regulatory Validations',
                            'refId': 'B'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 0, 'y': 8}
                },
                
                # Violation Types
                {
                    'id': 5,
                    'title': 'Privacy Violation Types',
                    'type': 'piechart',
                    'targets': [
                        {
                            'expr': 'medical_ai_data_privacy_violations_total',
                            'legendFormat': '{{violation_type}} - {{severity}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 12, 'x': 12, 'y': 8}
                },
                
                # Audit Log Entries
                {
                    'id': 6,
                    'title': 'Audit Log Activity',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'rate(medical_ai_audit_logs_total[5m])',
                            'legendFormat': '{{action_type}} - {{resource_type}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 16}
                }
            ]
        }
    
    def _create_executive_dashboard(self) -> Dict[str, Any]:
        """Create executive summary dashboard."""
        
        return {
            'title': 'Medical AI - Executive Summary',
            'description': 'High-level overview for executives and stakeholders',
            'tags': ['medical-ai', 'executive', 'summary'],
            'timezone': 'browser',
            'refresh': self.refresh_interval,
            'time': {
                'from': 'now-7d',
                'to': 'now'
            },
            'panels': [
                # Key Performance Indicators
                {
                    'id': 1,
                    'title': 'System Health Score',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': '(medical_ai_service_availability_percent + medical_ai_sla_compliance_percent + (1 - medical_ai_sla_violations_total/100)) / 3',
                            'legendFormat': 'Overall Health Score',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 0, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.8},
                                    {'color': 'green', 'value': 0.95}
                                ]
                            }
                        }
                    }
                },
                
                # Clinical Impact Score
                {
                    'id': 2,
                    'title': 'Clinical Impact Score',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'medical_ai_clinical_effectiveness_score',
                            'legendFormat': 'Clinical Effectiveness',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 6, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.7},
                                    {'color': 'green', 'value': 0.9}
                                ]
                            }
                        }
                    }
                },
                
                # Model Performance Summary
                {
                    'id': 3,
                    'title': 'Average Model Accuracy',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'avg(medical_ai_model_accuracy_score)',
                            'legendFormat': 'Average Accuracy',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 12, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.8},
                                    {'color': 'green', 'value': 0.95}
                                ]
                            }
                        }
                    }
                },
                
                # Compliance Status
                {
                    'id': 4,
                    'title': 'Regulatory Compliance',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': '(medical_ai_hipaa_compliance_checks_total{result="pass"}/medical_ai_hipaa_compliance_checks_total + medical_ai_regulatory_validations_total{validation_result="pass"}/medical_ai_regulatory_validations_total)/2',
                            'legendFormat': 'Compliance Rate',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 6, 'x': 18, 'y': 0},
                    'fieldConfig': {
                        'defaults': {
                            'unit': 'percentunit',
                            'min': 0,
                            'max': 1,
                            'thresholds': {
                                'steps': [
                                    {'color': 'red', 'value': 0},
                                    {'color': 'yellow', 'value': 0.9},
                                    {'color': 'green', 'value': 0.99}
                                ]
                            }
                        }
                    }
                },
                
                # Weekly Trends
                {
                    'id': 5,
                    'title': '7-Day Performance Trends',
                    'type': 'timeseries',
                    'targets': [
                        {
                            'expr': 'avg_over_time(medical_ai_clinical_effectiveness_score[7d])',
                            'legendFormat': 'Clinical Effectiveness',
                            'refId': 'A'
                        },
                        {
                            'expr': 'avg_over_time(medical_ai_model_accuracy_score[7d])',
                            'legendFormat': 'Model Accuracy',
                            'refId': 'B'
                        },
                        {
                            'expr': 'avg_over_time(medical_ai_service_availability_percent[7d])',
                            'legendFormat': 'Service Availability',
                            'refId': 'C'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 16, 'x': 0, 'y': 8}
                },
                
                # Alert Summary
                {
                    'id': 6,
                    'title': 'Alert Summary (Last 7 Days)',
                    'type': 'table',
                    'targets': [
                        {
                            'expr': 'increase(medical_ai_sla_violations_total[7d])',
                            'legendFormat': '{{sla_type}} - {{severity}}',
                            'refId': 'A'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 8, 'x': 16, 'y': 8}
                },
                
                # Key Metrics Summary
                {
                    'id': 7,
                    'title': 'Key Metrics Summary',
                    'type': 'stat',
                    'targets': [
                        {
                            'expr': 'sum(medical_ai_inference_requests_total[7d])',
                            'legendFormat': 'Total Requests (7d)',
                            'refId': 'A'
                        },
                        {
                            'expr': 'sum(medical_ai_clinical_outcomes_total[7d])',
                            'legendFormat': 'Clinical Outcomes (7d)',
                            'refId': 'B'
                        }
                    ],
                    'gridPos': {'h': 8, 'w': 24, 'x': 0, 'y': 16}
                }
            ]
        }
    
    async def get_dashboard_info(self, dashboard_name: str) -> Optional[Dict[str, Any]]:
        """Get dashboard information."""
        
        if dashboard_name not in self.deployed_dashboards:
            return None
        
        try:
            uid = self.deployed_dashboards[dashboard_name]
            url = f"{self.base_url}/api/dashboards/uid/{uid}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning("Failed to get dashboard info",
                                     status=response.status)
                        return None
                        
        except Exception as e:
            logger.error("Dashboard info retrieval failed", error=str(e))
            return None
    
    async def export_dashboard_json(self, dashboard_name: str) -> Optional[str]:
        """Export dashboard as JSON string."""
        
        dashboard_info = await self.get_dashboard_info(dashboard_name)
        if dashboard_info:
            return json.dumps(dashboard_info, indent=2)
        return None
    
    async def delete_dashboard(self, dashboard_name: str) -> bool:
        """Delete a Grafana dashboard."""
        
        if dashboard_name not in self.deployed_dashboards:
            return False
        
        try:
            uid = self.deployed_dashboards[dashboard_name]
            url = f"{self.base_url}/api/dashboards/uid/{uid}"
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    url,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None
                ) as response:
                    if response.status == 200:
                        del self.deployed_dashboards[dashboard_name]
                        logger.info("Dashboard deleted successfully",
                                  dashboard_name=dashboard_name)
                        return True
                    else:
                        logger.error("Failed to delete dashboard",
                                   status=response.status)
                        return False
                        
        except Exception as e:
            logger.error("Dashboard deletion failed", error=str(e))
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Grafana API health."""
        
        try:
            url = f"{self.base_url}/api/health"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self.headers,
                    auth=self._get_auth() if not self.api_key else None,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        health_info = await response.json()
                        return {
                            'status': 'healthy',
                            'version': health_info.get('version'),
                            'commit': health_info.get('commit'),
                            'database': health_info.get('database'),
                            'apis': health_info.get('apis', {})
                        }
                    else:
                        return {
                            'status': 'unhealthy',
                            'status_code': response.status,
                            'error': f'HTTP {response.status}'
                        }
                        
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }