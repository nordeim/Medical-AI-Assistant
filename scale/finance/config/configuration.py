#!/usr/bin/env python3
"""
Financial Optimization Configuration Module
Configuration management for the financial optimization framework
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ModelingConfig:
    """Financial modeling configuration"""
    model_complexity: str = "advanced"
    optimization_horizon: int = 365
    confidence_level: float = 0.95
    regression_model: str = "random_forest"
    time_series_model: str = "arima"
    validation_split: float = 0.2
    cross_validation_folds: int = 5

@dataclass 
class CapitalConfig:
    """Capital allocation configuration"""
    max_allocation_variance: float = 0.05
    risk_free_rate: float = 0.03
    market_risk_premium: float = 0.08
    rebalance_frequency: str = "quarterly"
    portfolio_optimization: str = "sharpe_ratio"
    asset_universe_size: int = 50
    min_position_size: float = 0.01
    max_position_size: float = 0.40

@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    monitoring_frequency: str = "daily"
    alert_thresholds: Dict[str, float] = None
    kpi_targets: Dict[str, float] = None
    dashboard_update_interval: int = 3600  # seconds
    data_retention_days: int = 365
    notification_channels: List[str] = None
    real_time_monitoring: bool = True
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'roe': 0.15,
                'roa': 0.08,
                'debt_to_equity': 1.5,
                'current_ratio': 1.5,
                'profit_margin': 0.10,
                'working_capital_ratio': 1.2
            }
        
        if self.kpi_targets is None:
            self.kpi_targets = {
                'revenue_growth': 0.15,
                'profit_margin': 0.12,
                'return_on_equity': 0.18,
                'asset_turnover': 1.5,
                'cash_flow_ratio': 0.85
            }
        
        if self.notification_channels is None:
            self.notification_channels = ['email', 'dashboard', 'api']

@dataclass
class OptimizationConfig:
    """Cost optimization configuration"""
    cost_reduction_target: float = 0.15
    efficiency_threshold: float = 0.85
    optimization_scope: List[str] = None
    improvement_timeline: int = 90
    budget_constraints: Dict[str, float] = None
    
    def __post_init__(self):
        if self.optimization_scope is None:
            self.optimization_scope = [
                'personnel_costs', 'technology_infrastructure', 
                'marketing_advertising', 'operational_supplies'
            ]
        
        if self.budget_constraints is None:
            self.budget_constraints = {
                'implementation_cost_limit': 500000,
                'annual_optimization_budget': 1000000,
                'quick_win_threshold': 50000
            }

@dataclass
class RiskConfig:
    """Risk management configuration"""
    var_limit: float = 0.02
    max_drawdown: float = 0.10
    concentration_limit: float = 0.30
    leverage_limit: float = 2.0
    liquidity_minimum: float = 0.10
    stress_test_scenarios: int = 5
    risk_monitoring_frequency: str = "daily"
    escalation_rules: Dict[str, str] = None
    
    def __post_init__(self):
        if self.escalation_rules is None:
            self.escalation_rules = {
                'critical': 'immediate',
                'high': '1_hour',
                'medium': '4_hours',
                'low': '24_hours'
            }

@dataclass
class InvestorConfig:
    """Investor relations configuration"""
    disclosure_frequency: str = "quarterly"
    communication_channels: List[str] = None
    reporting_templates: List[str] = None
    investor_presentation_cycle: int = 90
    funding_optimization_enabled: bool = True
    
    def __post_init__(self):
        if self.communication_channels is None:
            self.communication_channels = [
                'email', 'webinar', 'conference_call', 'annual_report'
            ]
        
        if self.reporting_templates is None:
            self.reporting_templates = [
                'quarterly_presentation', 'annual_report', 
                'investor_briefing', 'press_release'
            ]

@dataclass
class IntelligenceConfig:
    """Financial intelligence configuration"""
    prediction_horizon: int = 90
    sensitivity_level: float = 0.05
    model_update_frequency: str = "monthly"
    scenario_analysis_depth: str = "comprehensive"
    market_data_sources: List[str] = None
    competitive_intelligence: bool = True
    
    def __post_init__(self):
        if self.market_data_sources is None:
            self.market_data_sources = [
                'bloomberg', 'reuters', 'factset', 
                'internal_databases', 'third_party_apis'
            ]

@dataclass
class FrameworkConfig:
    """Main framework configuration"""
    # Component configurations
    modeling: ModelingConfig = None
    capital: CapitalConfig = None
    monitoring: MonitoringConfig = None
    optimization: OptimizationConfig = None
    risk: RiskConfig = None
    investor: InvestorConfig = None
    intelligence: IntelligenceConfig = None
    
    # Framework-level settings
    environment: str = "development"
    debug_mode: bool = False
    log_level: str = "INFO"
    data_directory: str = "./data"
    reports_directory: str = "./reports"
    export_format: str = "json"
    parallel_processing: bool = True
    max_workers: int = 4
    
    # API settings
    api_enabled: bool = True
    api_port: int = 8000
    api_host: str = "localhost"
    
    # Database settings
    database_url: str = "sqlite:///finance_framework.db"
    connection_pool_size: int = 10
    
    def __post_init__(self):
        # Initialize default configurations for all components
        if self.modeling is None:
            self.modeling = ModelingConfig()
        if self.capital is None:
            self.capital = CapitalConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.optimization is None:
            self.optimization = OptimizationConfig()
        if self.risk is None:
            self.risk = RiskConfig()
        if self.investor is None:
            self.investor = InvestorConfig()
        if self.intelligence is None:
            self.intelligence = IntelligenceConfig()
        
        # Create directories if they don't exist
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)
        Path(self.reports_directory).mkdir(parents=True, exist_ok=True)

class ConfigManager:
    """Configuration management class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = None
        
    def load_config(self, config_path: Optional[str] = None) -> FrameworkConfig:
        """Load configuration from file or use defaults"""
        if config_path:
            self.config_path = config_path
        
        if self.config_path and Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create nested configuration objects
            config = FrameworkConfig(**config_data)
            
            # Load component configs
            if 'modeling' in config_data:
                config.modeling = ModelingConfig(**config_data['modeling'])
            if 'capital' in config_data:
                config.capital = CapitalConfig(**config_data['capital'])
            if 'monitoring' in config_data:
                config.monitoring = MonitoringConfig(**config_data['monitoring'])
            if 'optimization' in config_data:
                config.optimization = OptimizationConfig(**config_data['optimization'])
            if 'risk' in config_data:
                config.risk = RiskConfig(**config_data['risk'])
            if 'investor' in config_data:
                config.investor = InvestorConfig(**config_data['investor'])
            if 'intelligence' in config_data:
                config.intelligence = IntelligenceConfig(**config_data['intelligence'])
            
            self.config = config
        else:
            # Use default configuration
            self.config = FrameworkConfig()
        
        return self.config
    
    def save_config(self, config: FrameworkConfig, config_path: Optional[str] = None):
        """Save configuration to file"""
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            raise ValueError("No configuration path specified")
        
        # Convert configuration to dictionary
        config_dict = asdict(config)
        
        # Save to file
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_config(self) -> FrameworkConfig:
        """Get current configuration"""
        if self.config is None:
            self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        if self.config is None:
            self.load_config()
        
        # Update top-level attributes
        for key, value in updates.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Update nested configuration
                    if hasattr(self.config, key):
                        nested_config = getattr(self.config, key)
                        if hasattr(nested_config, '__dict__'):
                            for nested_key, nested_value in value.items():
                                if hasattr(nested_config, nested_key):
                                    setattr(nested_config, nested_key, nested_value)
                else:
                    setattr(self.config, key, value)
    
    def validate_config(self, config: FrameworkConfig) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Validate modeling settings
        if config.modeling.optimization_horizon < 30:
            validation_results['warnings'].append("Short optimization horizon may limit analysis")
        
        if not 0.5 <= config.modeling.confidence_level <= 0.999:
            validation_results['errors'].append("Confidence level must be between 0.5 and 0.999")
        
        # Validate capital settings
        if config.capital.max_allocation_variance > 0.5:
            validation_results['warnings'].append("High allocation variance may increase portfolio risk")
        
        if config.capital.risk_free_rate < 0 or config.capital.risk_free_rate > 0.2:
            validation_results['warnings'].append("Unusual risk-free rate value")
        
        # Validate monitoring settings
        if config.monitoring.real_time_monitoring and config.monitoring.dashboard_update_interval < 300:
            validation_results['warnings'].append("Very frequent dashboard updates may impact performance")
        
        # Validate optimization settings
        if config.optimization.cost_reduction_target > 0.5:
            validation_results['warnings'].append("Very high cost reduction target may be unrealistic")
        
        # Validate risk settings
        if config.risk.var_limit > 0.1:
            validation_results['warnings'].append("High VaR limit may indicate high risk tolerance")
        
        if config.risk.max_drawdown > 0.5:
            validation_results['warnings'].append("Very high maximum drawdown tolerance")
        
        # Validate investor settings
        valid_frequencies = ['monthly', 'quarterly', 'annually']
        if config.investor.disclosure_frequency not in valid_frequencies:
            validation_results['errors'].append(f"Disclosure frequency must be one of: {valid_frequencies}")
        
        # Validate intelligence settings
        if config.intelligence.prediction_horizon > 365:
            validation_results['warnings'].append("Very long prediction horizon may reduce accuracy")
        
        # Check for errors
        if validation_results['errors']:
            validation_results['is_valid'] = False
        
        return validation_results

def create_default_config() -> FrameworkConfig:
    """Create a default configuration"""
    return FrameworkConfig()

def load_config_from_env() -> FrameworkConfig:
    """Load configuration from environment variables"""
    config = FrameworkConfig()
    
    # Load from environment variables if available
    config.environment = os.getenv('FINANCE_ENV', 'development')
    config.debug_mode = os.getenv('FINANCE_DEBUG', 'false').lower() == 'true'
    config.log_level = os.getenv('FINANCE_LOG_LEVEL', 'INFO')
    config.data_directory = os.getenv('FINANCE_DATA_DIR', './data')
    config.reports_directory = os.getenv('FINANCE_REPORTS_DIR', './reports')
    
    return config

# Predefined configuration presets
PRESET_CONFIGS = {
    'development': FrameworkConfig(
        environment='development',
        debug_mode=True,
        log_level='DEBUG',
        parallel_processing=False,
        api_enabled=False
    ),
    
    'testing': FrameworkConfig(
        environment='testing',
        debug_mode=True,
        log_level='DEBUG',
        monitoring_frequency='hourly',
        data_retention_days=30
    ),
    
    'staging': FrameworkConfig(
        environment='staging',
        debug_mode=False,
        log_level='INFO',
        monitoring_frequency='daily',
        api_enabled=True,
        api_port=8080
    ),
    
    'production': FrameworkConfig(
        environment='production',
        debug_mode=False,
        log_level='WARNING',
        monitoring_frequency='real_time',
        parallel_processing=True,
        max_workers=8,
        data_retention_days=1095,  # 3 years
        api_enabled=True,
        api_port=8000
    )
}

def get_preset_config(preset_name: str) -> FrameworkConfig:
    """Get a preset configuration"""
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available presets: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[preset_name]