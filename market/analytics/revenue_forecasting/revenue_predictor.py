"""
Revenue Prediction and Forecasting Engine
"""

from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from decimal import Decimal
import numpy as np
import logging
from dataclasses import asdict

class RevenuePredictor:
    """Revenue forecasting using multiple models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Forecasting parameters
        self.confidence_intervals = config.get('confidence_intervals', [0.1, 0.25, 0.5, 0.75, 0.9])
        self.min_data_points = config.get('min_data_points', 6)
        self.forecast_horizon_months = config.get('forecast_horizon_months', 12)
        
        # Model weights for ensemble
        self.model_weights = config.get('model_weights', {
            'linear': 0.3,
            'exponential': 0.25,
            'seasonal': 0.25,
            'moving_average': 0.2
        })
    
    def predict_revenue(self, period_months: int, model_type: str = 'ensemble', 
                       data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Generate revenue forecast"""
        self.logger.info(f"Generating {period_months}-month revenue forecast using {model_type} model")
        
        if not data:
            data = self._get_sample_revenue_data()
        
        if len(data) < self.min_data_points:
            raise ValueError(f"Insufficient data points: {len(data)} < {self.min_data_points}")
        
        # Run specific model or ensemble
        if model_type == 'ensemble':
            forecast = self._ensemble_forecast(data, period_months)
        else:
            forecast = self._single_model_forecast(data, period_months, model_type)
        
        # Calculate accuracy metrics
        accuracy_metrics = self._calculate_forecast_accuracy(data)
        
        # Add confidence intervals
        forecast['confidence_intervals'] = self._calculate_confidence_intervals(
            forecast['forecast_values'], accuracy_metrics['MAPE']
        )
        
        return {
            'forecast_period': period_months,
            'forecast_values': forecast['forecast_values'],
            'forecast_dates': forecast['forecast_dates'],
            'model_type': model_type,
            'confidence_intervals': forecast['confidence_intervals'],
            'accuracy_metrics': accuracy_metrics,
            'forecast_quality': self._assess_forecast_quality(accuracy_metrics),
            'generated_at': datetime.now().isoformat()
        }
    
    def _ensemble_forecast(self, data: List[Dict[str, Any]], period_months: int) -> Dict[str, Any]:
        """Generate ensemble forecast from multiple models"""
        forecasts = {}
        
        # Generate forecasts using different models
        if 'linear' in self.model_weights:
            forecasts['linear'] = self._linear_forecast(data, period_months)
        
        if 'exponential' in self.model_weights:
            forecasts['exponential'] = self._exponential_forecast(data, period_months)
        
        if 'seasonal' in self.model_weights:
            forecasts['seasonal'] = self._seasonal_forecast(data, period_months)
        
        if 'moving_average' in self.model_weights:
            forecasts['moving_average'] = self._moving_average_forecast(data, period_months)
        
        # Combine forecasts using weighted average
        ensemble_forecast = []
        total_weight = sum(self.model_weights.values())
        
        for i in range(period_months):
            weighted_sum = 0
            for model_name, weight in self.model_weights.items():
                if model_name in forecasts:
                    weighted_sum += forecasts[model_name][i] * (weight / total_weight)
            ensemble_forecast.append(weighted_sum)
        
        # Generate forecast dates
        last_date = data[-1]['date'] if data else date.today()
        forecast_dates = self._generate_forecast_dates(last_date, period_months)
        
        return {
            'forecast_values': ensemble_forecast,
            'forecast_dates': forecast_dates,
            'individual_forecasts': forecasts,
            'model_contributions': self.model_weights
        }
    
    def _single_model_forecast(self, data: List[Dict[str, Any]], period_months: int, model_type: str) -> Dict[str, Any]:
        """Generate forecast using single model"""
        if model_type == 'linear':
            forecast_values = self._linear_forecast(data, period_months)
        elif model_type == 'exponential':
            forecast_values = self._exponential_forecast(data, period_months)
        elif model_type == 'seasonal':
            forecast_values = self._seasonal_forecast(data, period_months)
        elif model_type == 'moving_average':
            forecast_values = self._moving_average_forecast(data, period_months)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Generate forecast dates
        last_date = data[-1]['date'] if data else date.today()
        forecast_dates = self._generate_forecast_dates(last_date, period_months)
        
        return {
            'forecast_values': forecast_values,
            'forecast_dates': forecast_dates,
            'model_type': model_type
        }
    
    def _linear_forecast(self, data: List[Dict[str, Any]], period_months: int) -> List[float]:
        """Generate linear trend forecast"""
        revenues = [float(d['revenue']) for d in data]
        n = len(revenues)
        
        if n < 2:
            return [revenues[-1]] * period_months
        
        # Calculate linear trend
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(revenues) / n
        
        numerator = sum((x[i] - x_mean) * (revenues[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Generate forecast
        forecast = []
        for i in range(period_months):
            future_x = n + i
            predicted_y = intercept + slope * future_x
            forecast.append(max(predicted_y, 0))  # Ensure non-negative
        
        return forecast
    
    def _exponential_forecast(self, data: List[Dict[str, Any]], period_months: int) -> List[float]:
        """Generate exponential growth forecast"""
        revenues = [float(d['revenue']) for d in data]
        
        if len(revenues) < 2:
            return [revenues[-1]] * period_months
        
        # Calculate growth rate
        growth_rates = []
        for i in range(1, len(revenues)):
            if revenues[i-1] > 0:
                growth_rate = revenues[i] / revenues[i-1] - 1
                growth_rates.append(growth_rate)
        
        if not growth_rates:
            return [revenues[-1]] * period_months
        
        # Use average growth rate
        avg_growth_rate = sum(growth_rates) / len(growth_rates)
        
        # Apply slight dampening to prevent unrealistic growth
        dampened_rate = avg_growth_rate * 0.8
        
        # Generate forecast
        forecast = []
        last_revenue = revenues[-1]
        
        for i in range(period_months):
            predicted_revenue = last_revenue * ((1 + dampened_rate) ** (i + 1))
            forecast.append(max(predicted_revenue, 0))
        
        return forecast
    
    def _seasonal_forecast(self, data: List[Dict[str, Any]], period_months: int) -> List[float]:
        """Generate seasonal forecast"""
        revenues = [float(d['revenue']) for d in data]
        
        if len(revenues) < 12:  # Need at least 1 year for seasonality
            return self._linear_forecast(data, period_months)
        
        # Calculate seasonal factors by month
        seasonal_factors = {}
        month_revenues = {}
        
        # Group revenues by month
        for i, revenue in enumerate(revenues):
            month = (i % 12) + 1
            if month not in month_revenues:
                month_revenues[month] = []
            month_revenues[month].append(revenue)
        
        # Calculate average revenue per month
        overall_avg = sum(revenues) / len(revenues)
        
        for month in range(1, 13):
            if month in month_revenues:
                month_avg = sum(month_revenues[month]) / len(month_revenues[month])
                seasonal_factors[month] = month_avg / overall_avg if overall_avg > 0 else 1
            else:
                seasonal_factors[month] = 1
        
        # Calculate base trend (linear)
        base_forecast = self._linear_forecast(data, period_months)
        
        # Apply seasonal adjustments
        last_date = data[-1]['date'] if data else date.today()
        seasonal_forecast = []
        
        for i, base_value in enumerate(base_forecast):
            # Calculate future date
            future_date = last_date + timedelta(days=30 * (i + 1))
            future_month = future_date.month
            
            # Apply seasonal factor
            seasonal_factor = seasonal_factors.get(future_month, 1)
            adjusted_value = base_value * seasonal_factor
            seasonal_forecast.append(max(adjusted_value, 0))
        
        return seasonal_forecast
    
    def _moving_average_forecast(self, data: List[Dict[str, Any]], period_months: int, window: int = 6) -> List[float]:
        """Generate moving average forecast"""
        revenues = [float(d['revenue']) for d in data]
        
        if len(revenues) < window:
            return [revenues[-1]] * period_months
        
        # Calculate moving average
        recent_revenues = revenues[-window:]
        avg_revenue = sum(recent_revenues) / len(recent_revenues)
        
        # Apply slight trend adjustment
        if len(revenues) >= 2:
            recent_trend = (recent_revenues[-1] - recent_revenues[0]) / window
            trend_adjustment = recent_trend * 0.5  # Dampen trend
        else:
            trend_adjustment = 0
        
        # Generate forecast
        forecast = []
        for i in range(period_months):
            predicted_revenue = avg_revenue + (trend_adjustment * (i + 1))
            forecast.append(max(predicted_revenue, 0))
        
        return forecast
    
    def _generate_forecast_dates(self, last_date: date, period_months: int) -> List[date]:
        """Generate forecast date range"""
        forecast_dates = []
        current_date = last_date
        
        for i in range(period_months):
            # Add one month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
            forecast_dates.append(current_date)
        
        return forecast_dates
    
    def _calculate_confidence_intervals(self, forecast_values: List[float], mape: float) -> Dict[str, List[float]]:
        """Calculate confidence intervals for forecast"""
        # Use MAPE as proxy for forecast error
        confidence_multipliers = {
            'p10': 1.645,  # 90% confidence
            'p25': 1.281,  # 80% confidence
            'p75': 1.281,  # 20% confidence (lower bound)
            'p90': 1.645   # 10% confidence (lower bound)
        }
        
        intervals = {}
        
        for interval_name, multiplier in confidence_multipliers.items():
            interval_values = []
            for value in forecast_values:
                error_margin = value * mape * multiplier
                if interval_name in ['p10', 'p25']:  # Upper bounds
                    interval_values.append(value + error_margin)
                else:  # Lower bounds
                    interval_values.append(max(value - error_margin, 0))
            intervals[interval_name] = interval_values
        
        return intervals
    
    def _calculate_forecast_accuracy(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        if len(data) < 3:
            return {'MAPE': 0.1, 'MAE': 0.0, 'RMSE': 0.0}
        
        # Generate one-step-ahead forecasts for validation
        actual_values = []
        predicted_values = []
        
        for i in range(1, len(data)):
            train_data = data[:i]
            actual_value = float(data[i]['revenue'])
            actual_values.append(actual_value)
            
            # Generate one-step forecast
            try:
                forecast = self._linear_forecast(train_data, 1)
                predicted_values.append(forecast[0])
            except:
                predicted_values.append(actual_value)  # Fallback
        
        # Calculate metrics
        if not predicted_values:
            return {'MAPE': 0.1, 'MAE': 0.0, 'RMSE': 0.0}
        
        # Mean Absolute Percentage Error
        mape_values = []
        for i in range(len(actual_values)):
            if actual_values[i] != 0:
                mape_values.append(abs((actual_values[i] - predicted_values[i]) / actual_values[i]))
        
        mape = sum(mape_values) / len(mape_values) if mape_values else 0.1
        
        # Mean Absolute Error
        mae = sum(abs(actual_values[i] - predicted_values[i]) for i in range(len(actual_values))) / len(actual_values)
        
        # Root Mean Square Error
        mse = sum((actual_values[i] - predicted_values[i]) ** 2 for i in range(len(actual_values))) / len(actual_values)
        rmse = mse ** 0.5
        
        return {
            'MAPE': mape,
            'MAE': mae,
            'RMSE': rmse,
            'sample_size': len(actual_values)
        }
    
    def _assess_forecast_quality(self, accuracy_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall forecast quality"""
        mape = accuracy_metrics.get('MAPE', 1.0)
        sample_size = accuracy_metrics.get('sample_size', 0)
        
        # Quality assessment based on MAPE
        if mape <= 0.05:
            quality = 'Excellent'
            quality_score = 100
        elif mape <= 0.1:
            quality = 'Good'
            quality_score = 85
        elif mape <= 0.2:
            quality = 'Fair'
            quality_score = 70
        else:
            quality = 'Poor'
            quality_score = 50
        
        # Adjust for sample size
        if sample_size < 6:
            quality_score *= 0.8
            quality = f"Low confidence ({quality})"
        
        return {
            'overall_quality': quality,
            'quality_score': quality_score,
            'confidence_level': max(0, min(1, 1 - mape))
        }
    
    def analyze_trends(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze revenue trends and patterns"""
        self.logger.info("Analyzing revenue trends")
        
        if not data:
            return {'error': 'No data provided'}
        
        revenues = [float(d['revenue']) for d in data]
        
        # Trend analysis
        trend_analysis = self._calculate_trend_metrics(revenues)
        
        # Growth analysis
        growth_analysis = self._calculate_growth_metrics(revenues)
        
        # Volatility analysis
        volatility_analysis = self._calculate_volatility_metrics(revenues)
        
        # Pattern detection
        patterns = self._detect_patterns(data)
        
        return {
            'trend_analysis': trend_analysis,
            'growth_analysis': growth_analysis,
            'volatility_analysis': volatility_analysis,
            'patterns_detected': patterns,
            'data_points': len(data),
            'analysis_date': datetime.now().isoformat()
        }
    
    def _calculate_trend_metrics(self, revenues: List[float]) -> Dict[str, Any]:
        """Calculate trend metrics"""
        if len(revenues) < 2:
            return {'trend': 'insufficient_data', 'strength': 0}
        
        # Linear trend
        n = len(revenues)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(revenues) / n
        
        numerator = sum((x[i] - x_mean) * (revenues[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Trend direction and strength
        if abs(slope) < y_mean * 0.01:  # Less than 1% change per period
            trend = 'stable'
            strength = 0
        elif slope > 0:
            trend = 'increasing'
            strength = min(slope / y_mean, 1) if y_mean > 0 else 0
        else:
            trend = 'decreasing'
            strength = min(abs(slope) / y_mean, 1) if y_mean > 0 else 0
        
        return {
            'trend': trend,
            'strength': strength,
            'slope': slope,
            'r_squared': self._calculate_r_squared(revenues)
        }
    
    def _calculate_growth_metrics(self, revenues: List[float]) -> Dict[str, Any]:
        """Calculate growth metrics"""
        if len(revenues) < 2:
            return {'cagr': 0, 'period_growth': {}}
        
        # Compound Annual Growth Rate (CAGR)
        periods = len(revenues) - 1
        start_value = revenues[0]
        end_value = revenues[-1]
        
        if start_value > 0:
            cagr = ((end_value / start_value) ** (1 / periods)) - 1
        else:
            cagr = 0
        
        # Period-over-period growth rates
        growth_rates = []
        for i in range(1, len(revenues)):
            if revenues[i-1] > 0:
                growth_rate = (revenues[i] / revenues[i-1]) - 1
                growth_rates.append(growth_rate)
        
        # Recent vs historical growth
        if len(growth_rates) >= 4:
            recent_growth = sum(growth_rates[-2:]) / 2  # Last 2 periods
            historical_growth = sum(growth_rates[:-2]) / max(len(growth_rates) - 2, 1)
            growth_acceleration = recent_growth - historical_growth
        else:
            recent_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
            historical_growth = recent_growth
            growth_acceleration = 0
        
        return {
            'cagr': cagr,
            'average_growth_rate': sum(growth_rates) / len(growth_rates) if growth_rates else 0,
            'recent_growth_rate': recent_growth,
            'historical_growth_rate': historical_growth,
            'growth_acceleration': growth_acceleration,
            'growth_consistency': self._calculate_growth_consistency(growth_rates)
        }
    
    def _calculate_volatility_metrics(self, revenues: List[float]) -> Dict[str, Any]:
        """Calculate volatility and stability metrics"""
        if len(revenues) < 2:
            return {'coefficient_of_variation': 0, 'volatility': 'low'}
        
        # Coefficient of Variation
        mean_revenue = sum(revenues) / len(revenues)
        variance = sum((r - mean_revenue) ** 2 for r in revenues) / len(revenues)
        std_dev = variance ** 0.5
        
        cv = std_dev / mean_revenue if mean_revenue > 0 else 0
        
        # Volatility classification
        if cv < 0.1:
            volatility = 'very_low'
        elif cv < 0.2:
            volatility = 'low'
        elif cv < 0.3:
            volatility = 'moderate'
        elif cv < 0.5:
            volatility = 'high'
        else:
            volatility = 'very_high'
        
        return {
            'coefficient_of_variation': cv,
            'volatility': volatility,
            'standard_deviation': std_dev,
            'mean_revenue': mean_revenue
        }
    
    def _detect_patterns(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns in revenue data"""
        patterns = []
        revenues = [float(d['revenue']) for d in data]
        
        # Seasonal pattern detection
        if len(data) >= 12:
            seasonal_pattern = self._detect_seasonal_pattern(data)
            if seasonal_pattern['detected']:
                patterns.append({
                    'type': 'seasonal',
                    'description': f"Strong seasonal pattern detected with {seasonal_pattern['strength']:.1%} consistency",
                    'confidence': seasonal_pattern['confidence']
                })
        
        # Trend change detection
        if len(revenues) >= 6:
            trend_change = self._detect_trend_change(revenues)
            if trend_change['detected']:
                patterns.append({
                    'type': 'trend_change',
                    'description': trend_change['description'],
                    'confidence': trend_change['confidence']
                })
        
        # Growth phase detection
        if len(revenues) >= 4:
            growth_phase = self._detect_growth_phase(revenues)
            if growth_phase['phase'] != 'stable':
                patterns.append({
                    'type': 'growth_phase',
                    'description': f"Detected {growth_phase['phase']} growth phase",
                    'confidence': growth_phase['confidence']
                })
        
        return patterns
    
    def _detect_seasonal_pattern(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect seasonal patterns in data"""
        if len(data) < 12:
            return {'detected': False, 'strength': 0, 'confidence': 0}
        
        # Calculate seasonal correlation
        seasonal_correlations = []
        for lag in range(1, 13):
            correlation = self._calculate_lag_correlation(data, lag)
            seasonal_correlations.append(abs(correlation))
        
        max_correlation = max(seasonal_correlations)
        
        return {
            'detected': max_correlation > 0.3,
            'strength': max_correlation,
            'confidence': max_correlation,
            'optimal_lag': seasonal_correlations.index(max_correlation) + 1
        }
    
    def _detect_trend_change(self, revenues: List[float]) -> Dict[str, Any]:
        """Detect significant trend changes"""
        if len(revenues) < 6:
            return {'detected': False}
        
        # Split data into two halves
        mid_point = len(revenues) // 2
        first_half = revenues[:mid_point]
        second_half = revenues[mid_point:]
        
        # Calculate trends for each half
        first_trend = self._calculate_simple_trend(first_half)
        second_trend = self._calculate_simple_trend(second_half)
        
        # Detect significant change
        trend_change = abs(second_trend - first_trend)
        
        if trend_change > 0.1:  # 10% change threshold
            change_direction = 'accelerating' if second_trend > first_trend else 'decelerating'
            return {
                'detected': True,
                'description': f"Revenue trend {change_direction} detected",
                'confidence': min(trend_change, 1.0)
            }
        
        return {'detected': False}
    
    def _detect_growth_phase(self, revenues: List[float]) -> Dict[str, Any]:
        """Detect current growth phase"""
        if len(revenues) < 4:
            return {'phase': 'stable', 'confidence': 0}
        
        # Calculate recent growth rates
        recent_growth = []
        for i in range(1, len(revenues)):
            if revenues[i-1] > 0:
                growth = (revenues[i] / revenues[i-1]) - 1
                recent_growth.append(growth)
        
        if not recent_growth:
            return {'phase': 'stable', 'confidence': 0}
        
        avg_growth = sum(recent_growth) / len(recent_growth)
        
        if avg_growth > 0.15:
            phase = 'rapid_growth'
            confidence = min(avg_growth, 1.0)
        elif avg_growth > 0.05:
            phase = 'steady_growth'
            confidence = min(avg_growth * 2, 1.0)
        elif avg_growth < -0.05:
            phase = 'decline'
            confidence = min(abs(avg_growth), 1.0)
        else:
            phase = 'stable'
            confidence = 0.5
        
        return {'phase': phase, 'confidence': confidence}
    
    def _calculate_r_squared(self, revenues: List[float]) -> float:
        """Calculate R-squared for linear trend"""
        if len(revenues) < 2:
            return 0
        
        n = len(revenues)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(revenues) / n
        
        # Calculate regression line
        numerator = sum((x[i] - x_mean) * (revenues[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        predicted_values = [intercept + slope * x_i for x_i in x]
        ss_res = sum((revenues[i] - predicted_values[i]) ** 2 for i in range(n))
        ss_tot = sum((revenues[i] - y_mean) ** 2 for i in range(n))
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)
    
    def _calculate_lag_correlation(self, data: List[Dict[str, Any]], lag: int) -> float:
        """Calculate correlation with lag"""
        if len(data) <= lag:
            return 0
        
        revenues = [float(d['revenue']) for d in data]
        x = revenues[lag:]
        y = revenues[:-lag]
        
        if len(x) < 2 or len(y) < 2:
            return 0
        
        # Calculate correlation
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator_x = sum((x[i] - x_mean) ** 2 for i in range(len(x))) ** 0.5
        denominator_y = sum((y[i] - y_mean) ** 2 for i in range(len(y))) ** 0.5
        
        if denominator_x == 0 or denominator_y == 0:
            return 0
        
        return numerator / (denominator_x * denominator_y)
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple trend as percentage change"""
        if len(values) < 2:
            return 0
        
        start_value = values[0]
        end_value = values[-1]
        
        if start_value <= 0:
            return 0
        
        return (end_value / start_value) - 1
    
    def _calculate_growth_consistency(self, growth_rates: List[float]) -> float:
        """Calculate consistency of growth rates"""
        if len(growth_rates) < 2:
            return 0
        
        # Use coefficient of variation as inverse consistency measure
        mean_growth = sum(growth_rates) / len(growth_rates)
        
        if mean_growth == 0:
            return 0
        
        variance = sum((g - mean_growth) ** 2 for g in growth_rates) / len(growth_rates)
        std_dev = variance ** 0.5
        
        cv = std_dev / abs(mean_growth)
        
        # Return consistency as 1 - CV, capped at 0
        return max(0, 1 - cv)
    
    def _get_sample_revenue_data(self) -> List[Dict[str, Any]]:
        """Generate sample revenue data for demo"""
        sample_data = []
        base_revenue = 500000
        current_date = date(2023, 1, 1)
        
        for i in range(24):  # 2 years of monthly data
            # Add some growth and seasonality
            growth_factor = 1 + (i * 0.02)  # 2% monthly growth
            seasonal_factor = 1 + (0.1 if current_date.month in [11, 12] else 0)  # Holiday boost
            
            revenue = base_revenue * growth_factor * seasonal_factor
            
            sample_data.append({
                'date': current_date,
                'revenue': Decimal(str(int(revenue)))
            })
            
            # Move to next month
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        
        return sample_data