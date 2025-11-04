"""
Financial Optimization and Capital Management Framework
"""

from .financial_optimization_orchestrator import (
    FinancialOptimizationOrchestrator,
    FinancialConfig,
    create_default_config,
    load_config,
    save_config
)

from .modeling.financial_models import (
    AdvancedFinancialModeling,
    FinancialMetrics,
    ForecastResults
)

from .capital.capital_allocation import (
    CapitalAllocationOptimizer,
    InvestmentOption,
    AllocationResult,
    PortfolioMetrics
)

from .monitoring.performance_tracker import (
    FinancialPerformanceMonitor,
    PerformanceAlert,
    PerformanceMetric,
    KPIResult
)

from .optimization.cost_optimizer import (
    CostOptimizationEngine,
    CostCategory,
    OptimizationStrategy,
    ProfitabilityEnhancement
)

from .risk.risk_manager import (
    FinancialRiskManager,
    RiskFactor,
    RiskMetric,
    RiskMitigationStrategy
)

from .investor.investor_relations import (
    InvestorRelationsManager,
    InvestorProfile,
    FundingStrategy,
    CommunicationPlan
)

from .intelligence.financial_intelligence import (
    FinancialIntelligenceEngine,
    IntelligenceInsight,
    StrategicScenario,
    PredictiveModel
)

from .config.configuration import (
    FrameworkConfig,
    ModelingConfig,
    CapitalConfig,
    MonitoringConfig,
    OptimizationConfig,
    RiskConfig,
    InvestorConfig,
    IntelligenceConfig,
    ConfigManager,
    create_default_config as create_config,
    get_preset_config
)

__version__ = "1.0.0"
__author__ = "Financial Optimization Team"
__description__ = "Comprehensive Financial Optimization and Capital Management Framework"

__all__ = [
    # Main orchestrator
    "FinancialOptimizationOrchestrator",
    "FinancialConfig",
    
    # Modeling components
    "AdvancedFinancialModeling",
    "FinancialMetrics", 
    "ForecastResults",
    
    # Capital allocation
    "CapitalAllocationOptimizer",
    "InvestmentOption",
    "AllocationResult",
    "PortfolioMetrics",
    
    # Performance monitoring
    "FinancialPerformanceMonitor",
    "PerformanceAlert",
    "PerformanceMetric",
    "KPIResult",
    
    # Cost optimization
    "CostOptimizationEngine",
    "CostCategory",
    "OptimizationStrategy",
    "ProfitabilityEnhancement",
    
    # Risk management
    "FinancialRiskManager",
    "RiskFactor",
    "RiskMetric",
    "RiskMitigationStrategy",
    
    # Investor relations
    "InvestorRelationsManager",
    "InvestorProfile",
    "FundingStrategy",
    "CommunicationPlan",
    
    # Intelligence engine
    "FinancialIntelligenceEngine",
    "IntelligenceInsight",
    "StrategicScenario",
    "PredictiveModel",
    
    # Configuration
    "FrameworkConfig",
    "ModelingConfig",
    "CapitalConfig", 
    "MonitoringConfig",
    "OptimizationConfig",
    "RiskConfig",
    "InvestorConfig",
    "IntelligenceConfig",
    "ConfigManager",
    "create_config",
    "get_preset_config"
]