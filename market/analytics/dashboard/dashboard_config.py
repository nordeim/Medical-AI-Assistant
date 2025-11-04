"""
Dashboard Configuration and Setup for Business Intelligence
"""

from datetime import datetime, date
from typing import Dict, Any, List
import json

class BIConfig:
    """Configuration management for Business Intelligence system"""
    
    def __init__(self):
        self.config = self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            'business_intelligence': {
                'version': '1.0.0',
                'last_updated': datetime.now().isoformat(),
                'refresh_intervals': {
                    'real_time_metrics': 60,  # seconds
                    'kpi_updates': 300,       # 5 minutes
                    'dashboard_refresh': 900,  # 15 minutes
                    'report_generation': 3600  # 1 hour
                }
            },
            'data_sources': {
                'database': {
                    'connection_string': 'postgresql://user:pass@localhost:5432/bi_db',
                    'query_timeout': 30,
                    'connection_pool_size': 10
                },
                'api': {
                    'base_url': 'https://api.company.com',
                    'api_key': 'your-api-key',
                    'rate_limit_per_minute': 100,
                    'timeout': 30
                },
                'files': [
                    {
                        'path': '/data/exports/sales_data.csv',
                        'format': 'csv',
                        'refresh_frequency': 'daily'
                    },
                    {
                        'path': '/data/exports/customer_data.json',
                        'format': 'json',
                        'refresh_frequency': 'hourly'
                    }
                ]
            },
            'kpi_config': {
                'refresh_interval_minutes': 60,
                'alert_cooldown_minutes': 60,
                'alert_channels': ['email', 'slack', 'webhook'],
                'notification_levels': ['critical', 'warning', 'info'],
                'kpi_categories': {
                    'financial': ['revenue', 'gross_margin', 'operating_margin', 'cash_flow'],
                    'customer': ['customer_acquisition', 'churn_rate', 'nrr', 'satisfaction'],
                    'sales': ['win_rate', 'pipeline_coverage', 'deal_size', 'sales_cycle'],
                    'marketing': ['cac', 'conversion_rate', 'lead_quality', 'campaign_roi'],
                    'operational': ['efficiency', 'quality', 'employee_satisfaction', 'automation']
                }
            },
            'cohort_config': {
                'analysis_period_months': 12,
                'min_cohort_size': 10,
                'retention_thresholds': {
                    'excellent': 0.8,
                    'good': 0.65,
                    'average': 0.5,
                    'poor': 0.35
                },
                'ltv_assumptions': {
                    'gross_margin': 0.75,
                    'average_lifespan_months': 36,
                    'discount_rate': 0.1
                }
            },
            'forecasting_config': {
                'confidence_intervals': [0.1, 0.25, 0.5, 0.75, 0.9],
                'min_data_points': 6,
                'forecast_horizon_months': 12,
                'model_weights': {
                    'linear': 0.3,
                    'exponential': 0.25,
                    'seasonal': 0.25,
                    'moving_average': 0.2
                }
            },
            'pipeline_config': {
                'pipeline_stages': [
                    'lead', 'qualified', 'proposal', 'negotiation', 'closed_won', 'closed_lost'
                ],
                'stage_probabilities': {
                    'lead': 0.10,
                    'qualified': 0.25,
                    'proposal': 0.50,
                    'negotiation': 0.75,
                    'closed_won': 1.0,
                    'closed_lost': 0.0
                },
                'avg_sales_cycle_days': 60,
                'healthy_conversion_rates': {
                    'lead_to_qualified': 0.25,
                    'qualified_to_proposal': 0.60,
                    'proposal_to_negotiation': 0.70,
                    'negotiation_to_close': 0.60
                }
            },
            'performance_config': {
                'tracking_periods': ['daily', 'weekly', 'monthly', 'quarterly'],
                'benchmark_sources': {
                    'industry_reports': 'quarterly',
                    'competitive_data': 'monthly',
                    'internal_benchmarks': 'weekly'
                },
                'performance_thresholds': {
                    'excellent': 0.9,
                    'good': 0.8,
                    'satisfactory': 0.7,
                    'needs_improvement': 0.6,
                    'critical': 0.5
                }
            },
            'dashboard_config': {
                'dashboard_refresh_minutes': 60,
                'report_generation_time': '08:00',
                'executive_distribution_list': [
                    'ceo@company.com',
                    'cto@company.com',
                    'cfo@company.com'
                ],
                'dashboard_templates': {
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
            },
            'business_rules': {
                'ltv_cac_ratio_targets': {
                    'excellent': 4.0,
                    'good': 3.0,
                    'acceptable': 2.0,
                    'poor': 1.0
                },
                'payback_period_targets': {
                    'excellent': 12,  # months
                    'good': 18,
                    'acceptable': 24,
                    'poor': 36
                },
                'churn_rate_thresholds': {
                    'excellent': 0.03,
                    'good': 0.05,
                    'acceptable': 0.08,
                    'poor': 0.12
                },
                'nrr_thresholds': {
                    'excellent': 1.20,
                    'good': 1.10,
                    'acceptable': 1.05,
                    'poor': 0.95
                }
            },
            'alerting': {
                'channels': {
                    'email': {
                        'smtp_server': 'smtp.company.com',
                        'smtp_port': 587,
                        'username': 'bi-alerts@company.com',
                        'password': 'your-password'
                    },
                    'slack': {
                        'webhook_url': 'https://hooks.slack.com/services/...',
                        'channel': '#bi-alerts'
                    },
                    'webhook': {
                        'url': 'https://api.company.com/webhooks/bi-alerts',
                        'authentication': 'Bearer token'
                    }
                },
                'escalation_rules': {
                    'critical': {
                        'immediate': True,
                        'repeat_interval': 3600,  # 1 hour
                        'escalate_after': 7200   # 2 hours
                    },
                    'warning': {
                        'immediate': False,
                        'repeat_interval': 14400,  # 4 hours
                        'escalate_after': 28800   # 8 hours
                    }
                }
            },
            'security': {
                'data_encryption': True,
                'access_control': {
                    'role_based': True,
                    'minimum_permissions': True,
                    'session_timeout': 3600
                },
                'audit_logging': {
                    'enabled': True,
                    'retention_days': 90,
                    'log_level': 'INFO'
                }
            },
            'data_governance': {
                'data_quality': {
                    'validation_rules': True,
                    'completeness_checks': True,
                    'consistency_checks': True
                },
                'privacy': {
                    'gdpr_compliance': True,
                    'data_anonymization': True,
                    'consent_management': True
                },
                'retention': {
                    'operational_data': 730,  # 2 years
                    'aggregated_reports': 2555,  # 7 years
                    'audit_logs': 2555
                }
            }
        }
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """Get configuration section"""
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update configuration section"""
        if section in self.config:
            self.config[section].update(updates)
        else:
            self.config[section] = updates
    
    def save_config(self, file_path: str = None) -> None:
        """Save configuration to file"""
        if file_path is None:
            file_path = '/workspace/market/analytics/config/bi_config.json'
        
        with open(file_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_config(self, file_path: str) -> None:
        """Load configuration from file"""
        with open(file_path, 'r') as f:
            self.config = json.load(f)
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration"""
        errors = []
        warnings = []
        
        # Validate required sections
        required_sections = [
            'business_intelligence', 'data_sources', 'kpi_config',
            'cohort_config', 'forecasting_config', 'pipeline_config',
            'performance_config', 'dashboard_config'
        ]
        
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate data sources
        if 'data_sources' in self.config:
            if not self.config['data_sources'].get('database', {}).get('connection_string'):
                warnings.append("Database connection string not configured")
            
            if not self.config['data_sources'].get('api', {}).get('base_url'):
                warnings.append("API base URL not configured")
        
        # Validate KPI thresholds
        if 'kpi_config' in self.config:
            if not self.config['kpi_config'].get('alert_channels'):
                warnings.append("No alert channels configured")
        
        return {
            'errors': errors,
            'warnings': warnings
        }

class DashboardTheme:
    """Dashboard theme configuration"""
    
    def __init__(self, theme_name: str = 'corporate'):
        self.theme_name = theme_name
        self.themes = self._load_themes()
    
    def _load_themes(self) -> Dict[str, Dict[str, Any]]:
        """Load available themes"""
        return {
            'corporate': {
                'colors': {
                    'primary': '#1f2937',
                    'secondary': '#3b82f6',
                    'success': '#10b981',
                    'warning': '#f59e0b',
                    'danger': '#ef4444',
                    'info': '#3b82f6',
                    'background': '#f9fafb',
                    'surface': '#ffffff',
                    'text': '#1f2937',
                    'text_secondary': '#6b7280'
                },
                'fonts': {
                    'primary': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                    'heading': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
                    'mono': 'JetBrains Mono, monospace'
                },
                'spacing': {
                    'xs': '0.25rem',
                    'sm': '0.5rem',
                    'md': '1rem',
                    'lg': '1.5rem',
                    'xl': '2rem',
                    '2xl': '3rem'
                },
                'border_radius': {
                    'sm': '0.25rem',
                    'md': '0.5rem',
                    'lg': '0.75rem',
                    'xl': '1rem'
                },
                'shadows': {
                    'sm': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    'lg': '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
                }
            },
            'minimal': {
                'colors': {
                    'primary': '#000000',
                    'secondary': '#6b7280',
                    'success': '#059669',
                    'warning': '#d97706',
                    'danger': '#dc2626',
                    'info': '#2563eb',
                    'background': '#ffffff',
                    'surface': '#f8fafc',
                    'text': '#000000',
                    'text_secondary': '#6b7280'
                },
                'fonts': {
                    'primary': 'SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif',
                    'heading': 'SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif',
                    'mono': 'SF Mono, monospace'
                },
                'spacing': {
                    'xs': '0.25rem',
                    'sm': '0.5rem',
                    'md': '1rem',
                    'lg': '1.25rem',
                    'xl': '1.5rem',
                    '2xl': '2rem'
                },
                'border_radius': {
                    'sm': '0.125rem',
                    'md': '0.25rem',
                    'lg': '0.375rem',
                    'xl': '0.5rem'
                },
                'shadows': {
                    'sm': '0 1px 3px 0 rgba(0, 0, 0, 0.1)',
                    'md': '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
                    'lg': '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
                }
            }
        }
    
    def get_theme(self) -> Dict[str, Any]:
        """Get current theme configuration"""
        return self.themes.get(self.theme_name, self.themes['corporate'])

class AlertConfiguration:
    """Alert configuration management"""
    
    def __init__(self):
        self.alert_rules = self._create_default_rules()
    
    def _create_default_rules(self) -> List[Dict[str, Any]]:
        """Create default alert rules"""
        return [
            {
                'name': 'Revenue Below Target',
                'metric': 'revenue',
                'condition': 'below',
                'threshold': 0.9,
                'severity': 'critical',
                'cooldown_minutes': 60,
                'channels': ['email', 'slack'],
                'template': 'revenue_alert'
            },
            {
                'name': 'High Churn Rate',
                'metric': 'churn_rate',
                'condition': 'above',
                'threshold': 0.08,
                'severity': 'warning',
                'cooldown_minutes': 120,
                'channels': ['email'],
                'template': 'churn_alert'
            },
            {
                'name': 'Pipeline Coverage Low',
                'metric': 'pipeline_coverage',
                'condition': 'below',
                'threshold': 2.0,
                'severity': 'warning',
                'cooldown_minutes': 180,
                'channels': ['slack'],
                'template': 'pipeline_alert'
            },
            {
                'name': 'Customer Satisfaction Drop',
                'metric': 'customer_satisfaction',
                'condition': 'below',
                'threshold': 8.0,
                'severity': 'warning',
                'cooldown_minutes': 240,
                'channels': ['email', 'slack'],
                'template': 'satisfaction_alert'
            },
            {
                'name': 'Critical System Alert',
                'metric': 'system_health',
                'condition': 'below',
                'threshold': 0.8,
                'severity': 'critical',
                'cooldown_minutes': 30,
                'channels': ['email', 'slack', 'webhook'],
                'template': 'system_alert'
            }
        ]
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """Get all alert rules"""
        return self.alert_rules
    
    def add_alert_rule(self, rule: Dict[str, Any]) -> None:
        """Add new alert rule"""
        self.alert_rules.append(rule)
    
    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove alert rule by name"""
        for i, rule in enumerate(self.alert_rules):
            if rule['name'] == rule_name:
                del self.alert_rules[i]
                return True
        return False

# Global configuration instances
bi_config = BIConfig()
default_theme = DashboardTheme('corporate')
alert_config = AlertConfiguration()