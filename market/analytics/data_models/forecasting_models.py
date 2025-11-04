"""
Revenue Forecasting and Pipeline Prediction Models
"""

from dataclasses import dataclass
from datetime import datetime, date
from typing import List, Optional, Dict, Any
from decimal import Decimal
import json

@dataclass
class RevenueForecast:
    """Revenue forecasting model"""
    forecast_id: str
    forecast_date: date
    forecast_period_start: date
    forecast_period_end: date
    forecast_model: str  # 'linear', 'exponential', 'seasonal', 'ml_model'
    confidence_level: float  # 0-1 scale
    
    # Base revenue
    base_revenue: Decimal
    forecast_revenue: Decimal
    
    # Forecast components
    new_business_forecast: Decimal
    expansion_forecast: Decimal
    contraction_forecast: Decimal
    churn_forecast: Decimal
    
    # Confidence intervals
    optimistic_forecast: Decimal
    pessimistic_forecast: Decimal
    
    # Scenario analysis
    best_case_scenario: Decimal
    worst_case_scenario: Decimal
    most_likely_scenario: Decimal
    
    # Forecast accuracy tracking
    previous_forecasts: List[Dict[str, Any]]
    accuracy_metrics: Dict[str, float]
    
    def get_forecast_range(self) -> Dict[str, Decimal]:
        """Get forecast range (optimistic to pessimistic)"""
        return {
            'pessimistic': self.pessimistic_forecast,
            'most_likely': self.forecast_revenue,
            'optimistic': self.optimistic_forecast
        }
    
    def calculate_forecast_quality(self) -> float:
        """Calculate forecast quality score"""
        if not self.previous_forecasts:
            return 50.0  # Neutral if no history
        
        # Calculate average accuracy from historical forecasts
        total_accuracy = 0
        valid_forecasts = 0
        
        for prev_forecast in self.previous_forecasts[-6:]:  # Last 6 forecasts
            if 'actual_revenue' in prev_forecast and 'predicted_revenue' in prev_forecast:
                predicted = Decimal(str(prev_forecast['predicted_revenue']))
                actual = Decimal(str(prev_forecast['actual_revenue']))
                
                if predicted > 0:
                    accuracy = 1 - abs(predicted - actual) / predicted
                    total_accuracy += max(accuracy, 0)
                    valid_forecasts += 1
        
        if valid_forecasts == 0:
            return 50.0
        
        avg_accuracy = total_accuracy / valid_forecasts
        return avg_accuracy * 100
    
    def get_scenario_probabilities(self) -> Dict[str, float]:
        """Get probabilities for different scenarios"""
        base_confidence = self.confidence_level
        
        # Simplify probability calculation
        total_range = self.optimistic_forecast - self.pessimistic_forecast
        if total_range <= 0:
            return {'most_likely': 1.0, 'best_case': 0.0, 'worst_case': 0.0}
        
        # Distance from most likely to extremes
        dist_to_best = self.optimistic_forecast - self.forecast_revenue
        dist_to_worst = self.forecast_revenue - self.pessimistic_forecast
        
        # Probability inversely related to distance
        best_prob = base_confidence * (1 - dist_to_best / total_range)
        worst_prob = base_confidence * (1 - dist_to_worst / total_range)
        most_likely_prob = base_confidence
        
        # Normalize to sum to 1
        total_prob = best_prob + worst_prob + most_likely_prob
        
        if total_prob > 0:
            return {
                'best_case': best_prob / total_prob,
                'worst_case': worst_prob / total_prob,
                'most_likely': most_likely_prob / total_prob
            }
        
        return {'most_likely': 1.0, 'best_case': 0.0, 'worst_case': 0.0}

@dataclass
class PipelineForecast:
    """Pipeline-based revenue forecasting"""
    forecast_date: date
    forecast_months_ahead: int
    
    # Pipeline composition
    total_pipeline_value: Decimal
    weighted_pipeline_value: Decimal
    
    # By stage
    qualified_pipeline: Decimal
    proposal_pipeline: Decimal
    negotiation_pipeline: Decimal
    
    # Forecast metrics
    projected_bookings: Decimal
    win_rate_forecast: float
    average_deal_size_forecast: Decimal
    
    # Risk analysis
    high_risk_pipeline: Decimal
    medium_risk_pipeline: Decimal
    low_risk_pipeline: Decimal
    deals_at_risk: List[str]
    
    # Seasonal adjustments
    seasonal_factor: float
    seasonal_confidence: float
    
    def get_forecast_confidence(self) -> float:
        """Calculate forecast confidence based on pipeline quality"""
        if self.weighted_pipeline_value <= 0:
            return 0.0
        
        # Base confidence on win rate and deal quality
        base_confidence = self.win_rate_forecast
        
        # Adjust for deal risk distribution
        risk_adjustment = 0
        total_value = self.total_pipeline_value
        
        if total_value > 0:
            high_risk_pct = float(self.high_risk_pipeline / total_value)
            medium_risk_pct = float(self.medium_risk_pipeline / total_value)
            low_risk_pct = float(self.low_risk_pipeline / total_value)
            
            risk_adjustment = (low_risk_pct * 0.1) - (high_risk_pct * 0.15)
        
        confidence = max(0, min(1, base_confidence + risk_adjustment))
        return confidence
    
    def get_deal_coverage_ratio(self) -> float:
        """Get quota coverage ratio"""
        # Assuming quota is based on historical performance
        # This would typically come from sales targets
        
        # Simplified calculation - would need actual quota data
        estimated_quota = self.projected_bookings * 1.2  # Assume 20% buffer
        if estimated_quota <= 0:
            return 0.0
        
        return float(self.weighted_pipeline_value / estimated_quota)

@dataclass
class TrendAnalysis:
    """Revenue trend analysis and pattern recognition"""
    metric_name: str
    analysis_date: date
    time_period: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    
    # Trend data
    historical_data: List[Dict[str, Any]]  # date -> value
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength: float  # 0-1 scale
    trend_rate: float  # % change per period
    
    # Seasonality
    has_seasonality: bool
    seasonal_pattern: List[float]  # period -> seasonal factor
    seasonal_confidence: float
    
    # Growth analysis
    compound_growth_rate: float
    linear_growth_rate: float
    acceleration_deceleration: float
    
    # Forecast intervals
    next_period_forecast: Decimal
    next_3_periods_forecast: List[Decimal]
    forecast_confidence: float
    
    # Pattern recognition
    patterns_detected: List[str]
    anomalies: List[Dict[str, Any]]
    
    def get_growth_quality_score(self) -> float:
        """Assess quality of growth pattern"""
        score = 0
        
        # Trend consistency (30 points)
        if abs(self.trend_rate) < 2:  # Stable growth
            score += 30
        elif self.trend_direction == 'increasing':
            score += 20
        elif self.trend_direction == 'stable':
            score += 10
        
        # Growth sustainability (40 points)
        if self.trend_strength > 0.7:  # Strong consistent trend
            score += 40
        elif self.trend_strength > 0.4:
            score += 25
        elif self.trend_strength > 0.2:
            score += 15
        
        # Predictability (30 points)
        if self.forecast_confidence > 0.8:
            score += 30
        elif self.forecast_confidence > 0.6:
            score += 20
        elif self.forecast_confidence > 0.4:
            score += 10
        
        return min(score, 100)
    
    def get_forecast_recommendation(self) -> Dict[str, Any]:
        """Get recommendations based on trend analysis"""
        recommendations = []
        
        if self.trend_direction == 'decreasing' and self.trend_rate < -5:
            recommendations.append({
                'type': 'Risk Alert',
                'message': 'Negative trend detected - investigate causes',
                'priority': 'High'
            })
        
        if self.trend_strength < 0.3:
            recommendations.append({
                'type': 'Forecast Caution',
                'message': 'Low trend strength reduces forecast reliability',
                'priority': 'Medium'
            })
        
        if self.has_seasonality and self.seasonal_confidence > 0.7:
            recommendations.append({
                'type': 'Seasonal Planning',
                'message': 'Plan for seasonal variations in forecast',
                'priority': 'Medium'
            })
        
        if self.acceleration_deceleration > 10:
            recommendations.append({
                'type': 'Growth Acceleration',
                'message': 'Growth is accelerating - may need capacity planning',
                'priority': 'Medium'
            })
        
        if len(self.anomalies) > 2:
            recommendations.append({
                'type': 'Data Quality',
                'message': 'Multiple anomalies detected - review data sources',
                'priority': 'Low'
            })
        
        return {
            'recommendations': recommendations,
            'forecast_quality': self.get_growth_quality_score(),
            'confidence_level': self.forecast_confidence
        }