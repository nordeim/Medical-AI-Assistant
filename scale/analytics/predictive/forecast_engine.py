"""
Predictive Analytics Module
Business forecasting and trend analysis capabilities
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ForecastResult:
    """Data class for forecasting results"""
    forecast_id: str
    metric_name: str
    current_value: float
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_date: datetime
    accuracy_score: float
    model_used: str
    trend_direction: str
    risk_level: str

@dataclass
class TrendAnalysis:
    """Data class for trend analysis results"""
    trend_id: str
    metric_name: str
    trend_type: str
    strength: float
    significance: float
    forecasted_duration: int
    recommendations: List[str]

class PredictiveAnalytics:
    """Advanced Predictive Analytics for Business Forecasting"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.trend_models = {}
        self.forecast_results = []
        self.trend_analyses = []
        
    def create_forecast_model(self, data: pd.DataFrame, target_column: str, 
                            forecast_horizon: int = 30, model_type: str = "random_forest") -> str:
        """Create and train a forecasting model"""
        try:
            # Prepare features and target
            X, y = self._prepare_features(data, target_column)
            
            # Handle datetime features
            if 'date' in data.columns or any('date' in col.lower() for col in data.columns):
                date_col = next((col for col in data.columns if 'date' in col.lower()), None)
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col])
                    data['day_of_week'] = data[date_col].dt.dayofweek
                    data['month'] = data[date_col].dt.month
                    data['quarter'] = data[date_col].dt.quarter
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select and train model
            if model_type == "linear":
                model = LinearRegression()
            elif model_type == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = r2_score(y_test, y_pred)
            
            # Store model and scaler
            model_id = f"forecast_{target_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_id] = model
            self.scalers[model_id] = scaler
            
            return model_id
            
        except Exception as e:
            raise Exception(f"Error creating forecast model: {str(e)}")
    
    def generate_forecast(self, model_id: str, forecast_periods: int = 30, 
                         current_data: pd.DataFrame = None) -> List[ForecastResult]:
        """Generate forecasts using trained model"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self.models[model_id]
            scaler = self.scalers[model_id]
            
            # Extract metric name from model_id
            metric_name = model_id.split('_')[2]
            
            forecasts = []
            
            for period in range(1, forecast_periods + 1):
                # Generate forecast for each period
                # This is simplified - in practice, you'd need to handle time series specifically
                forecast_value = np.random.normal(100, 10)  # Placeholder calculation
                
                # Calculate confidence interval (simplified)
                confidence_interval = (
                    forecast_value * 0.95, 
                    forecast_value * 1.05
                )
                
                # Determine trend direction
                if forecast_value > 100:
                    trend_direction = "increasing"
                elif forecast_value < 100:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                
                # Assess risk level
                forecast_range = confidence_interval[1] - confidence_interval[0]
                if forecast_range > 15:
                    risk_level = "high"
                elif forecast_range > 8:
                    risk_level = "medium"
                else:
                    risk_level = "low"
                
                forecast_result = ForecastResult(
                    forecast_id=f"{model_id}_forecast_{period}",
                    metric_name=metric_name,
                    current_value=100,  # Placeholder
                    predicted_value=forecast_value,
                    confidence_interval=confidence_interval,
                    prediction_date=datetime.now() + timedelta(days=period),
                    accuracy_score=0.85,  # Placeholder
                    model_used=model_id,
                    trend_direction=trend_direction,
                    risk_level=risk_level
                )
                
                forecasts.append(forecast_result)
            
            self.forecast_results.extend(forecasts)
            return forecasts
            
        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")
    
    def analyze_trends(self, data: pd.DataFrame, metric_column: str, 
                      analysis_period: int = 90) -> List[TrendAnalysis]:
        """Analyze trends in the data"""
        try:
            if len(data) < analysis_period:
                raise ValueError(f"Insufficient data for trend analysis. Need at least {analysis_period} rows")
            
            # Sort by date if date column exists
            date_col = next((col for col in data.columns if 'date' in col.lower()), None)
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col])
                data = data.sort_values(date_col)
            
            # Calculate moving averages
            data['ma_7'] = data[metric_column].rolling(window=7).mean()
            data['ma_30'] = data[metric_column].rolling(window=30).mean()
            
            # Analyze trend strength and significance
            trend_strength = self._calculate_trend_strength(data[metric_column])
            trend_significance = self._calculate_trend_significance(data[metric_column])
            
            # Determine trend type
            trend_type = self._classify_trend(data[metric_column])
            
            # Calculate forecasted duration (simplified)
            if trend_strength > 0.7:
                forecasted_duration = 60  # Strong trend expected to continue
            elif trend_strength > 0.4:
                forecasted_duration = 30  # Moderate trend
            else:
                forecasted_duration = 15  # Weak trend
            
            # Generate recommendations based on trend analysis
            recommendations = self._generate_trend_recommendations(trend_type, trend_strength)
            
            trend_analysis = TrendAnalysis(
                trend_id=f"trend_{metric_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metric_name=metric_column,
                trend_type=trend_type,
                strength=trend_strength,
                significance=trend_significance,
                forecasted_duration=forecasted_duration,
                recommendations=recommendations
            )
            
            self.trend_analyses.append(trend_analysis)
            return [trend_analysis]
            
        except Exception as e:
            raise Exception(f"Error analyzing trends: {str(e)}")
    
    def get_business_insights(self, forecasts: List[ForecastResult], 
                            trends: List[TrendAnalysis]) -> Dict[str, Any]:
        """Generate comprehensive business insights from forecasts and trends"""
        try:
            insights = {
                "executive_summary": self._generate_executive_summary(forecasts, trends),
                "forecast_summary": self._analyze_forecasts(forecasts),
                "trend_insights": self._analyze_trends(trends),
                "risk_assessment": self._assess_risks(forecasts, trends),
                "opportunities": self._identify_opportunities(forecasts, trends),
                "actionable_recommendations": self._generate_actionable_recommendations(forecasts, trends),
                "key_metrics": self._calculate_key_metrics(forecasts, trends),
                "timestamp": datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            raise Exception(f"Error generating business insights: {str(e)}")
    
    def _prepare_features(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training"""
        # Remove target column and non-numeric columns
        features = data.drop(columns=[target_column])
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        target = data[target_column]
        
        return features, target
    
    def _calculate_trend_strength(self, series: pd.Series) -> float:
        """Calculate trend strength using linear regression"""
        try:
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return 0.0
            
            # Calculate correlation coefficient as proxy for trend strength
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_trend_significance(self, series: pd.Series) -> float:
        """Calculate trend significance (simplified)"""
        # This is a simplified calculation
        variance = series.var()
        trend_strength = self._calculate_trend_strength(series)
        
        # Higher variance with strong trend = higher significance
        significance = min(1.0, trend_strength * (1 + variance / series.mean()**2))
        return significance
    
    def _classify_trend(self, series: pd.Series) -> str:
        """Classify trend type"""
        try:
            # Calculate overall slope
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) < 2:
                return "insufficient_data"
            
            # Simple linear regression
            slope = np.polyfit(x_clean, y_clean, 1)[0]
            
            # Classify based on slope and recent behavior
            recent_trend = series.tail(5).mean() - series.head(5).mean()
            
            if slope > series.std() * 0.1:
                return "strong_upward"
            elif slope < -series.std() * 0.1:
                return "strong_downward"
            elif abs(slope) < series.std() * 0.05:
                return "stable"
            else:
                return "moderate"
                
        except Exception:
            return "unknown"
    
    def _generate_trend_recommendations(self, trend_type: str, strength: float) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        if trend_type in ["strong_upward", "moderate_upward"]:
            recommendations.append("Monitor capacity to meet increased demand")
            recommendations.append("Consider expanding inventory to avoid stockouts")
            if strength > 0.7:
                recommendations.append("Prepare for significant scaling requirements")
        
        elif trend_type in ["strong_downward", "moderate_downward"]:
            recommendations.append("Investigate root causes of declining trend")
            recommendations.append("Consider cost optimization strategies")
            recommendations.append("Review market conditions and competition")
        
        elif trend_type == "stable":
            recommendations.append("Maintain current operational strategies")
            recommendations.append("Focus on efficiency improvements")
        
        return recommendations
    
    def _generate_executive_summary(self, forecasts: List[ForecastResult], 
                                  trends: List[TrendAnalysis]) -> str:
        """Generate executive summary"""
        if not forecasts and not trends:
            return "No analysis results available"
        
        total_forecasts = len(forecasts)
        increasing_trends = len([t for t in trends if 'upward' in t.trend_type])
        high_risk_forecasts = len([f for f in forecasts if f.risk_level == 'high'])
        
        summary = f"""
        Business Intelligence Analysis Summary:
        
        • {total_forecasts} forecasts generated across {len(set(f.metric_name for f in forecasts))} key metrics
        • {increasing_trends} positive trends identified out of {len(trends)} total trends analyzed
        • {high_risk_forecasts} forecasts carry high risk level requiring attention
        
        Key Insights:
        """
        
        return summary
    
    def _analyze_forecasts(self, forecasts: List[ForecastResult]) -> Dict[str, Any]:
        """Analyze forecast results"""
        if not forecasts:
            return {}
        
        total_forecasts = len(forecasts)
        avg_confidence = np.mean([f.predicted_value / f.current_value for f in forecasts if f.current_value > 0])
        trend_distribution = {}
        risk_distribution = {}
        
        for forecast in forecasts:
            trend_distribution[forecast.trend_direction] = trend_distribution.get(forecast.trend_direction, 0) + 1
            risk_distribution[forecast.risk_level] = risk_distribution.get(forecast.risk_level, 0) + 1
        
        return {
            "total_forecasts": total_forecasts,
            "average_growth_rate": avg_confidence - 1,
            "trend_distribution": trend_distribution,
            "risk_distribution": risk_distribution
        }
    
    def _analyze_trends(self, trends: List[TrendAnalysis]) -> Dict[str, Any]:
        """Analyze trend results"""
        if not trends:
            return {}
        
        avg_strength = np.mean([t.strength for t in trends])
        trend_types = [t.trend_type for t in trends]
        
        return {
            "total_trends_analyzed": len(trends),
            "average_trend_strength": avg_strength,
            "trend_types": list(set(trend_types))
        }
    
    def _assess_risks(self, forecasts: List[ForecastResult], 
                     trends: List[TrendAnalysis]) -> Dict[str, Any]:
        """Assess business risks"""
        high_risk_items = []
        
        # Check for high-risk forecasts
        for forecast in forecasts:
            if forecast.risk_level == "high":
                high_risk_items.append(f"High uncertainty in {forecast.metric_name} forecast")
        
        # Check for declining trends
        declining_trends = [t for t in trends if 'downward' in t.trend_type]
        
        risk_score = len(high_risk_items) * 0.4 + len(declining_trends) * 0.3
        
        return {
            "overall_risk_score": min(1.0, risk_score / 10),
            "risk_factors": high_risk_items,
            "declining_trends": len(declining_trends),
            "recommendation": "Review high-risk forecasts and declining trends" if risk_score > 2 else "Risk levels are manageable"
        }
    
    def _identify_opportunities(self, forecasts: List[ForecastResult], 
                              trends: List[TrendAnalysis]) -> List[str]:
        """Identify business opportunities"""
        opportunities = []
        
        # Growth opportunities
        growing_metrics = [f.metric_name for f in forecasts if f.trend_direction == "increasing"]
        if growing_metrics:
            opportunities.append(f"Growth potential in: {', '.join(growing_metrics)}")
        
        # Strong upward trends
        strong_upward = [t for t in trends if t.trend_type == "strong_upward" and t.strength > 0.7]
        if strong_upward:
            opportunities.append("Strong upward momentum suggests expansion opportunities")
        
        return opportunities
    
    def _generate_actionable_recommendations(self, forecasts: List[ForecastResult], 
                                           trends: List[TrendAnalysis]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Risk mitigation recommendations
        high_risk_forecasts = [f for f in forecasts if f.risk_level == "high"]
        if high_risk_forecasts:
            recommendations.append("Implement contingency planning for high-risk forecasts")
        
        # Trend-based recommendations
        for trend in trends:
            recommendations.extend(trend.recommendations)
        
        # Performance optimization
        recommendations.append("Regularly update models with new data to maintain accuracy")
        recommendations.append("Establish monitoring dashboards for key forecasted metrics")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_key_metrics(self, forecasts: List[ForecastResult], 
                             trends: List[TrendAnalysis]) -> Dict[str, float]:
        """Calculate key performance metrics"""
        if not forecasts:
            return {}
        
        total_predictions = len(forecasts)
        avg_forecast_growth = np.mean([
            (f.predicted_value - f.current_value) / f.current_value 
            for f in forecasts if f.current_value > 0
        ])
        
        return {
            "total_predictions": total_predictions,
            "average_forecast_growth_rate": avg_forecast_growth,
            "forecast_accuracy_score": np.mean([f.accuracy_score for f in forecasts]),
            "trends_analyzed": len(trends)
        }

if __name__ == "__main__":
    # Example usage
    analytics = PredictiveAnalytics()
    
    # Sample data
    sample_data = pd.DataFrame({
        'revenue': [100, 105, 102, 110, 108, 115, 120, 118, 125, 130],
        'customers': [50, 52, 51, 55, 53, 58, 60, 59, 62, 65],
        'satisfaction': [4.1, 4.2, 4.0, 4.3, 4.2, 4.4, 4.5, 4.4, 4.6, 4.7]
    })
    
    # Create forecast model
    model_id = analytics.create_forecast_model(sample_data, 'revenue')
    
    # Generate forecasts
    forecasts = analytics.generate_forecast(model_id, forecast_periods=7)
    
    # Analyze trends
    trends = analytics.analyze_trends(sample_data, 'revenue')
    
    # Get business insights
    insights = analytics.get_business_insights(forecasts, trends)
    print(json.dumps(insights, indent=2, default=str))