#!/usr/bin/env python3
"""
Capital Allocation and Investment Optimization Component
Advanced capital allocation, investment optimization, and portfolio management
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

@dataclass
class InvestmentOption:
    """Investment option definition"""
    name: str
    expected_return: float
    risk_level: float
    minimum_allocation: float = 0.0
    maximum_allocation: float = 1.0
    correlation_matrix: List[List[float]] = None
    sector: str = "General"
    liquidity: str = "High"  # High, Medium, Low
    time_horizon: int = 365  # days

@dataclass
class AllocationResult:
    """Capital allocation optimization result"""
    optimal_allocation: Dict[str, float]
    expected_portfolio_return: float
    portfolio_risk: float
    sharpe_ratio: float
    risk_adjusted_return: float
    rebalancing_recommendations: List[Dict[str, Any]]

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    total_return: float
    volatility: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    information_ratio: float

class CapitalAllocationOptimizer:
    """
    Advanced Capital Allocation and Investment Optimization Engine
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('capital_allocation')
        self.investment_options = {}
        self.allocation_history = []
        self.portfolio_performance = {}
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the capital allocation component"""
        try:
            # Initialize default investment options
            await self._initialize_default_options()
            
            # Load optimization parameters
            self.optimization_params = {
                'objective_function': 'maximize_sharpe_ratio',
                'risk_tolerance': 'moderate',
                'constraints': ['min_allocation', 'max_allocation', 'sum_to_one'],
                'rebalance_frequency': 'quarterly'
            }
            
            self.logger.info("Capital allocation component initialized")
            return {
                'status': 'success',
                'investment_options': len(self.investment_options),
                'optimization_params': self.optimization_params
            }
            
        except Exception as e:
            self.logger.error(f"Capital allocation initialization failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _initialize_default_options(self):
        """Initialize default investment options"""
        default_options = [
            InvestmentOption(
                name="US_EQUITIES", 
                expected_return=0.08, 
                risk_level=0.15,
                minimum_allocation=0.0,
                maximum_allocation=0.40,
                sector="Equities",
                liquidity="High"
            ),
            InvestmentOption(
                name="INTERNATIONAL_EQUITIES", 
                expected_return=0.09, 
                risk_level=0.18,
                minimum_allocation=0.0,
                maximum_allocation=0.30,
                sector="Equities",
                liquidity="High"
            ),
            InvestmentOption(
                name="CORPORATE_BONDS", 
                expected_return=0.04, 
                risk_level=0.05,
                minimum_allocation=0.10,
                maximum_allocation=0.50,
                sector="Fixed Income",
                liquidity="Medium"
            ),
            InvestmentOption(
                name="GOVERNMENT_BONDS", 
                expected_return=0.025, 
                risk_level=0.03,
                minimum_allocation=0.05,
                maximum_allocation=0.40,
                sector="Fixed Income",
                liquidity="High"
            ),
            InvestmentOption(
                name="REAL_ESTATE", 
                expected_return=0.07, 
                risk_level=0.12,
                minimum_allocation=0.0,
                maximum_allocation=0.20,
                sector="Alternatives",
                liquidity="Low"
            ),
            InvestmentOption(
                name="COMMODITIES", 
                expected_return=0.06, 
                risk_level=0.20,
                minimum_allocation=0.0,
                maximum_allocation=0.15,
                sector="Alternatives",
                liquidity="Medium"
            ),
            InvestmentOption(
                name="CASH_EQUIVALENTS", 
                expected_return=0.015, 
                risk_level=0.01,
                minimum_allocation=0.0,
                maximum_allocation=0.25,
                sector="Cash",
                liquidity="High"
            )
        ]
        
        for option in default_options:
            self.investment_options[option.name] = option
        
        # Initialize correlation matrix
        await self._initialize_correlation_matrix()
    
    async def _initialize_correlation_matrix(self):
        """Initialize correlation matrix for investments"""
        asset_names = list(self.investment_options.keys())
        n_assets = len(asset_names)
        
        # Default correlation matrix (can be enhanced with real data)
        correlation_base = {
            ('US_EQUITIES', 'INTERNATIONAL_EQUITIES'): 0.85,
            ('US_EQUITIES', 'CORPORATE_BONDS'): -0.1,
            ('US_EQUITIES', 'GOVERNMENT_BONDS'): -0.2,
            ('US_EQUITIES', 'REAL_ESTATE'): 0.3,
            ('US_EQUITIES', 'COMMODITIES'): 0.1,
            ('US_EQUITIES', 'CASH_EQUIVALENTS'): -0.05,
            ('INTERNATIONAL_EQUITIES', 'CORPORATE_BONDS'): -0.05,
            ('INTERNATIONAL_EQUITIES', 'GOVERNMENT_BONDS'): -0.15,
            ('INTERNATIONAL_EQUITIES', 'REAL_ESTATE'): 0.25,
            ('INTERNATIONAL_EQUITIES', 'COMMODITIES'): 0.15,
            ('INTERNATIONAL_EQUITIES', 'CASH_EQUIVALENTS'): -0.03,
            ('CORPORATE_BONDS', 'GOVERNMENT_BONDS'): 0.6,
            ('CORPORATE_BONDS', 'REAL_ESTATE'): 0.0,
            ('CORPORATE_BONDS', 'COMMODITIES'): -0.05,
            ('CORPORATE_BONDS', 'CASH_EQUIVALENTS'): 0.1,
            ('GOVERNMENT_BONDS', 'REAL_ESTATE'): -0.1,
            ('GOVERNMENT_BONDS', 'COMMODITIES'): -0.1,
            ('GOVERNMENT_BONDS', 'CASH_EQUIVALENTS'): 0.2,
            ('REAL_ESTATE', 'COMMODITIES'): 0.1,
            ('REAL_ESTATE', 'CASH_EQUIVALENTS'): -0.02,
            ('COMMODITIES', 'CASH_EQUIVALENTS'): -0.01
        }
        
        # Create full correlation matrix
        correlation_matrix = np.eye(n_assets)
        
        for i, asset1 in enumerate(asset_names):
            for j, asset2 in enumerate(asset_names):
                if i != j:
                    correlation = correlation_base.get((asset1, asset2), 
                                                     correlation_base.get((asset2, asset1), 0.0))
                    correlation_matrix[i, j] = correlation
        
        # Store correlation matrix in each option
        for i, option in enumerate(self.investment_options.values()):
            option.correlation_matrix = correlation_matrix.tolist()
    
    async def optimize_allocation(self, 
                                financial_data: Dict[str, Any], 
                                modeling_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize capital allocation based on financial data and modeling results
        
        Args:
            financial_data: Current financial data
            modeling_results: Results from financial modeling component
            
        Returns:
            Dict containing allocation optimization results
        """
        self.logger.info("Starting capital allocation optimization...")
        
        try:
            # Extract relevant data
            total_capital = await self._extract_total_capital(financial_data)
            
            if total_capital <= 0:
                return {
                    'status': 'failed',
                    'error': 'Invalid total capital value',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Get current market conditions
            market_conditions = await self._assess_market_conditions(financial_data)
            
            # Adjust investment options based on current conditions
            adjusted_options = await self._adjust_options_for_market(market_conditions)
            
            # Perform optimization
            optimization_result = await self._perform_optimization(
                adjusted_options, total_capital, market_conditions
            )
            
            # Generate rebalancing recommendations
            rebalancing_recs = await self._generate_rebalancing_recommendations(optimization_result)
            
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics(optimization_result)
            
            # Store results
            allocation_result = AllocationResult(
                optimal_allocation=optimization_result['allocation'],
                expected_portfolio_return=optimization_result['expected_return'],
                portfolio_risk=optimization_result['portfolio_risk'],
                sharpe_ratio=optimization_result['sharpe_ratio'],
                risk_adjusted_return=optimization_result['risk_adjusted_return'],
                rebalancing_recommendations=rebalancing_recs
            )
            
            self.allocation_history.append({
                'timestamp': datetime.now().isoformat(),
                'total_capital': total_capital,
                'result': allocation_result
            })
            
            return {
                'status': 'success',
                'allocation': asdict(allocation_result),
                'portfolio_metrics': asdict(portfolio_metrics),
                'market_conditions': market_conditions,
                'optimization_details': optimization_result['details'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Capital allocation optimization failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _extract_total_capital(self, financial_data: Dict[str, Any]) -> float:
        """Extract total available capital from financial data"""
        # Try multiple sources for capital calculation
        capital_sources = []
        
        if 'total_capital' in financial_data:
            capital_sources.append(financial_data['total_capital'])
        
        if 'cash_and_equivalents' in financial_data:
            capital_sources.append(financial_data['cash_and_equivalents'])
        
        if 'available_capital' in financial_data:
            capital_sources.append(financial_data['available_capital'])
        
        if 'working_capital' in financial_data:
            # If only working capital is available, assume 70% is available for investment
            capital_sources.append(financial_data['working_capital'] * 0.7)
        
        # Calculate from balance sheet if individual components available
        if 'assets' in financial_data and 'liabilities' in financial_data:
            equity_capital = financial_data['assets'] - financial_data['liabilities']
            capital_sources.append(equity_capital * 0.3)  # Assume 30% equity available for investment
        
        return max(capital_sources) if capital_sources else 1000000.0  # Default to $1M
    
    async def _assess_market_conditions(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current market conditions"""
        # Default market conditions (can be enhanced with real market data)
        conditions = {
            'volatility_regime': 'moderate',  # low, moderate, high
            'interest_rate_environment': 'stable',  # rising, stable, falling
            'credit_spreads': 'normal',  # tight, normal, wide
            'economic_cycle': 'expansion',  # recession, recovery, expansion, late_cycle
            'market_sentiment': 'neutral',  # bearish, neutral, bullish
            'risk_appetite': 'moderate'  # low, moderate, high
        }
        
        # Adjust conditions based on financial data if available
        if 'volatility_index' in financial_data:
            vix = financial_data['volatility_index']
            if vix > 30:
                conditions['volatility_regime'] = 'high'
            elif vix < 15:
                conditions['volatility_regime'] = 'low'
            else:
                conditions['volatility_regime'] = 'moderate'
        
        return conditions
    
    async def _adjust_options_for_market(self, market_conditions: Dict[str, Any]) -> List[InvestmentOption]:
        """Adjust investment options based on market conditions"""
        adjusted_options = []
        
        for option in self.investment_options.values():
            # Create adjusted version of the option
            adjusted_option = InvestmentOption(
                name=option.name,
                expected_return=option.expected_return,
                risk_level=option.risk_level,
                minimum_allocation=option.minimum_allocation,
                maximum_allocation=option.maximum_allocation,
                correlation_matrix=option.correlation_matrix,
                sector=option.sector,
                liquidity=option.liquidity,
                time_horizon=option.time_horizon
            )
            
            # Adjust expected returns based on market conditions
            if market_conditions.get('market_sentiment') == 'bearish':
                adjusted_option.expected_return *= 0.9
            elif market_conditions.get('market_sentiment') == 'bullish':
                adjusted_option.expected_return *= 1.1
            
            # Adjust risk levels based on volatility
            if market_conditions.get('volatility_regime') == 'high':
                adjusted_option.risk_level *= 1.2
            elif market_conditions.get('volatility_regime') == 'low':
                adjusted_option.risk_level *= 0.9
            
            # Adjust maximum allocations based on risk appetite
            risk_appetite = market_conditions.get('risk_appetite', 'moderate')
            if risk_appetite == 'low':
                adjusted_option.maximum_allocation *= 0.7
            elif risk_appetite == 'high':
                adjusted_option.maximum_allocation *= 1.3
            
            adjusted_options.append(adjusted_option)
        
        return adjusted_options
    
    async def _perform_optimization(self, 
                                  options: List[InvestmentOption], 
                                  total_capital: float,
                                  market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Perform portfolio optimization"""
        n_assets = len(options)
        
        # Extract parameters
        expected_returns = np.array([opt.expected_return for opt in options])
        risk_levels = np.array([opt.risk_level for opt in options])
        
        # Build correlation matrix
        correlation_matrix = np.array(options[0].correlation_matrix) if options[0].correlation_matrix else np.eye(n_assets)
        
        # Calculate covariance matrix
        cov_matrix = np.outer(risk_levels, risk_levels) * correlation_matrix
        
        # Define objective function (maximize Sharpe ratio)
        def negative_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            risk_free_rate = self.config.risk_free_rate
            
            if portfolio_volatility == 0:
                return -np.inf
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio  # Minimize negative Sharpe ratio
        
        # Define constraints
        constraints = []
        
        # Sum of weights = 1
        constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Minimum allocation constraints
        for i, opt in enumerate(options):
            if opt.minimum_allocation > 0:
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - opt.minimum_allocation})
        
        # Maximum allocation constraints
        for i, opt in enumerate(options):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: opt.maximum_allocation - x[i]})
        
        # Bounds for each weight (0 to max_allocation)
        bounds = [(0, opt.maximum_allocation) for opt in options]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        x0 = np.minimum(x0, [opt.maximum_allocation for opt in options])
        x0 = x0 / np.sum(x0)  # Normalize to sum to 1
        
        # Perform optimization
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return = np.sum(optimal_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            
            # Calculate Sharpe ratio
            risk_free_rate = self.config.risk_free_rate
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calculate risk-adjusted return
            risk_adjusted_return = portfolio_return - 0.5 * (portfolio_volatility ** 2)
            
            # Create allocation dictionary
            allocation = {options[i].name: optimal_weights[i] for i in range(n_assets)}
            
            return {
                'allocation': allocation,
                'expected_return': portfolio_return,
                'portfolio_risk': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_adjusted_return': risk_adjusted_return,
                'details': {
                    'optimization_method': 'SLSQP',
                    'convergence_achieved': result.success,
                    'optimization_iterations': result.nit,
                    'constraint_violations': result.ineqlin.get('residual', 0) if hasattr(result.ineqlin, 'get') else 0
                }
            }
        else:
            # Return equal weights if optimization fails
            equal_weights = np.ones(n_assets) / n_assets
            equal_allocation = {options[i].name: equal_weights[i] for i in range(n_assets)}
            
            portfolio_return = np.sum(equal_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                'allocation': equal_allocation,
                'expected_return': portfolio_return,
                'portfolio_risk': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_adjusted_return': portfolio_return - 0.5 * (portfolio_volatility ** 2),
                'details': {
                    'optimization_method': 'equal_weight_fallback',
                    'warning': 'Optimization failed, using equal weights'
                }
            }
    
    async def _generate_rebalancing_recommendations(self, optimization_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate rebalancing recommendations"""
        recommendations = []
        allocation = optimization_result['allocation']
        
        # Get current allocation (if available from history)
        current_allocation = None
        if len(self.allocation_history) > 0:
            current_allocation = self.allocation_history[-1]['result'].optimal_allocation
        
        for asset_name, target_weight in allocation.items():
            # Calculate recommended action
            if current_allocation and asset_name in current_allocation:
                current_weight = current_allocation[asset_name]
                weight_diff = target_weight - current_weight
                
                if abs(weight_diff) > 0.02:  # 2% threshold for rebalancing
                    action = "increase" if weight_diff > 0 else "decrease"
                    recommendations.append({
                        'asset': asset_name,
                        'action': action,
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'weight_difference': weight_diff,
                        'priority': 'high' if abs(weight_diff) > 0.05 else 'medium'
                    })
            else:
                # New position recommendation
                if target_weight > 0.01:  # Only recommend if weight > 1%
                    recommendations.append({
                        'asset': asset_name,
                        'action': 'establish',
                        'current_weight': 0,
                        'target_weight': target_weight,
                        'weight_difference': target_weight,
                        'priority': 'medium'
                    })
        
        return recommendations
    
    async def _calculate_portfolio_metrics(self, optimization_result: Dict[str, Any]) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        allocation = optimization_result['allocation']
        expected_return = optimization_result['expected_return']
        portfolio_risk = optimization_result['portfolio_risk']
        sharpe_ratio = optimization_result['sharpe_ratio']
        
        # Calculate additional metrics
        total_return = expected_return * 1.0  # Simplified - assume 1 year horizon
        max_drawdown = portfolio_risk * 2.5  # Rough estimate
        sortino_ratio = sharpe_ratio * 1.2  # Simplified adjustment
        beta = 1.0  # Simplified - assume market beta of 1
        alpha = expected_return - (0.03 + 1.0 * (0.08 - 0.03))  # CAPM
        information_ratio = sharpe_ratio * 0.8  # Simplified
        
        return PortfolioMetrics(
            total_return=total_return,
            volatility=portfolio_risk,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            'is_initialized': len(self.investment_options) > 0,
            'investment_options': list(self.investment_options.keys()),
            'allocation_history_size': len(self.allocation_history),
            'optimization_params': self.optimization_params
        }
    
    async def shutdown(self) -> Dict[str, Any]:
        """Shutdown the component"""
        try:
            self.investment_options.clear()
            self.allocation_history.clear()
            self.portfolio_performance.clear()
            self.logger.info("Capital allocation component shutdown completed")
            return {'status': 'success'}
        except Exception as e:
            self.logger.error(f"Shutdown failed: {str(e)}")
            return {'status': 'failed', 'error': str(e)}

# Utility functions for capital allocation
def calculate_portfolio_return(weights: np.ndarray, returns: np.ndarray) -> float:
    """Calculate portfolio expected return"""
    return np.dot(weights, returns)

def calculate_portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def calculate_sharpe_ratio(portfolio_return: float, portfolio_volatility: float, risk_free_rate: float = 0.03) -> float:
    """Calculate Sharpe ratio"""
    if portfolio_volatility == 0:
        return 0
    return (portfolio_return - risk_free_rate) / portfolio_volatility

def calculate_information_ratio(portfolio_return: float, benchmark_return: float, tracking_error: float) -> float:
    """Calculate Information ratio"""
    if tracking_error == 0:
        return 0
    return (portfolio_return - benchmark_return) / tracking_error