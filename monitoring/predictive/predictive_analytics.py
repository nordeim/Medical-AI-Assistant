"""
Predictive Analytics for System Performance and Health Monitoring
Provides capacity planning, anomaly prediction, and proactive issue detection
for medical AI systems with advanced forecasting and trend analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PerformancePrediction:
    """Container for performance predictions"""
    timestamp: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon_hours: int
    trend_direction: str
    anomaly_probability: float

@dataclass
class CapacityPlanningForecast:
    """Container for capacity planning forecasts"""
    resource_type: str
    current_utilization: float
    predicted_peak_utilization: float
    capacity_threshold_date: datetime
    recommended_capacity_increase: float
    confidence_level: float

class TimeSeriesAnomalyDetector:
    """Advanced anomaly detection for time series data with multiple algorithms"""
    
    def __init__(self,
                 window_size: int = 24,
                 seasonality_period: int = 24,
                 anomaly_threshold: float = 3.0):
        """
        Initialize anomaly detector
        
        Args:
            window_size: Sliding window size for anomaly detection
            seasonality_period: Period for seasonal decomposition
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.window_size = window_size
        self.seasonality_period = seasonality_period
        self.anomaly_threshold = anomaly_threshold
        self.baseline_stats = {}
        
    def detect_statistical_anomalies(self, 
                                   time_series: np.ndarray,
                                   timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods
        
        Args:
            time_series: Time series data
            timestamps: Corresponding timestamps
            
        Returns:
            Dict containing anomaly detection results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'statistical',
            'anomalies_detected': False,
            'anomaly_indices': [],
            'anomaly_timestamps': [],
            'z_scores': [],
            'anomaly_scores': []
        }
        
        try:
            if len(time_series) < self.window_size:
                results['error'] = "Insufficient data for anomaly detection"
                return results
            
            # Calculate rolling statistics
            rolling_mean = pd.Series(time_series).rolling(window=self.window_size).mean()
            rolling_std = pd.Series(time_series).rolling(window=self.window_size).std()
            
            # Calculate z-scores
            z_scores = np.abs((time_series - rolling_mean) / rolling_std)
            results['z_scores'] = z_scores.fillna(0).tolist()
            
            # Detect anomalies
            anomaly_mask = z_scores > self.anomaly_threshold
            anomaly_indices = np.where(anomaly_mask)[0]
            
            results['anomaly_indices'] = anomaly_indices.tolist()
            results['anomaly_timestamps'] = [timestamps[i] for i in anomaly_indices if i < len(timestamps)]
            results['anomalies_detected'] = len(anomaly_indices) > 0
            results['anomaly_count'] = len(anomaly_indices)
            
            # Calculate anomaly scores
            if len(anomaly_indices) > 0:
                results['anomaly_scores'] = z_scores.iloc[anomaly_indices].tolist()
                results['max_anomaly_score'] = float(z_scores.iloc[anomaly_indices].max())
            
        except Exception as e:
            logging.error(f"Error in statistical anomaly detection: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def detect_isolation_forest_anomalies(self,
                                        time_series: np.ndarray,
                                        contamination: float = 0.1) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest algorithm
        
        Args:
            time_series: Time series data
            contamination: Expected proportion of anomalies
            
        Returns:
            Dict containing Isolation Forest results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'isolation_forest',
            'anomalies_detected': False,
            'anomaly_indices': [],
            'anomaly_scores': [],
            'decision_scores': []
        }
        
        try:
            # Prepare features for Isolation Forest
            if len(time_series) < 10:
                results['error'] = "Insufficient data for Isolation Forest"
                return results
            
            # Create feature matrix with lag features
            features = self._create_lag_features(time_series, max_lags=5)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)
            decision_scores = iso_forest.decision_function(features)
            
            # Extract anomalies (labeled as -1)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            results['anomaly_indices'] = anomaly_indices.tolist()
            results['anomalies_detected'] = len(anomaly_indices) > 0
            results['anomaly_count'] = len(anomaly_indices)
            results['decision_scores'] = decision_scores.tolist()
            results['anomaly_scores'] = decision_scores[anomaly_indices].tolist()
            
        except Exception as e:
            logging.error(f"Error in Isolation Forest anomaly detection: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def detect_seasonal_anomalies(self,
                                time_series: np.ndarray,
                                timestamps: List[datetime]) -> Dict[str, Any]:
        """
        Detect seasonal anomalies using time series decomposition
        
        Args:
            time_series: Time series data
            timestamps: Corresponding timestamps
            
        Returns:
            Dict containing seasonal anomaly results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'seasonal',
            'anomalies_detected': False,
            'seasonal_anomalies': [],
            'trend_anomalies': []
        }
        
        try:
            if len(time_series) < 2 * self.seasonality_period:
                results['error'] = "Insufficient data for seasonal decomposition"
                return results
            
            # Simple seasonal decomposition (manual implementation)
            seasonal_component = self._extract_seasonal_component(time_series)
            trend_component = self._extract_trend_component(time_series)
            residual_component = time_series - seasonal_component - trend_component
            
            # Detect seasonal anomalies
            seasonal_threshold = 2 * np.std(seasonal_component)
            seasonal_anomalies = np.where(np.abs(seasonal_component) > seasonal_threshold)[0]
            
            # Detect trend anomalies
            trend_threshold = 2 * np.std(np.diff(trend_component))
            trend_changes = np.abs(np.diff(trend_component))
            trend_anomalies = np.where(trend_changes > trend_threshold)[0]
            
            results['seasonal_anomalies'] = seasonal_anomalies.tolist()
            results['trend_anomalies'] = trend_anomalies.tolist()
            results['anomalies_detected'] = len(seasonal_anomalies) > 0 or len(trend_anomalies) > 0
            
        except Exception as e:
            logging.error(f"Error in seasonal anomaly detection: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def _create_lag_features(self, time_series: np.ndarray, max_lags: int = 5) -> np.ndarray:
        """Create lag features for anomaly detection"""
        features = []
        
        for i in range(max_lags, len(time_series)):
            lag_features = time_series[i-max_lags:i]
            features.append(lag_features)
        
        return np.array(features)
    
    def _extract_seasonal_component(self, time_series: np.ndarray) -> np.ndarray:
        """Extract seasonal component from time series"""
        seasonal = np.zeros_like(time_series)
        
        # Simple seasonal extraction using moving averages
        for i in range(len(time_series)):
            if i < self.seasonality_period:
                seasonal[i] = np.mean(time_series[:i+1])
            else:
                seasonal[i] = np.mean(time_series[i-self.seasonality_period:i])
        
        return seasonal - np.mean(seasonal)
    
    def _extract_trend_component(self, time_series: np.ndarray) -> np.ndarray:
        """Extract trend component from time series"""
        trend = np.zeros_like(time_series)
        
        # Simple trend extraction using polynomial fitting
        x = np.arange(len(time_series))
        
        try:
            # Fit polynomial of degree 2
            coeffs = np.polyfit(x, time_series, 2)
            trend = np.polyval(coeffs, x)
        except Exception:
            # Fallback to linear trend
            coeffs = np.polyfit(x, time_series, 1)
            trend = np.polyval(coeffs, x)
        
        return trend

class PerformancePredictor:
    """Predictive analytics for system performance forecasting"""
    
    def __init__(self,
                 prediction_horizon_hours: int = 24,
                 training_window_days: int = 7):
        """
        Initialize performance predictor
        
        Args:
            prediction_horizon_hours: Hours to predict ahead
            training_window_days: Days of historical data for training
        """
        self.prediction_horizon_hours = prediction_horizon_hours
        self.training_window_days = training_window_days
        self.models = {}
        self.scalers = {}
        
    def predict_cpu_utilization(self,
                              historical_data: np.ndarray,
                              timestamps: List[datetime],
                              external_factors: Optional[Dict[str, np.ndarray]] = None) -> PerformancePrediction:
        """
        Predict CPU utilization
        
        Args:
            historical_data: Historical CPU utilization data
            timestamps: Corresponding timestamps
            external_factors: External factors like scheduled maintenance, user activity
            
        Returns:
            PerformancePrediction with forecast
        """
        try:
            if len(historical_data) < 24:
                raise ValueError("Insufficient historical data for prediction")
            
            # Prepare features
            features = self._prepare_time_series_features(historical_data, timestamps)
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            scaler = StandardScaler()
            
            X_scaled = scaler.fit_transform(features)
            model.fit(X_scaled, historical_data)
            
            # Make predictions
            future_timestamps = self._generate_future_timestamps(timestamps, self.prediction_horizon_hours)
            future_features = self._prepare_forecast_features(historical_data, timestamps, future_timestamps)
            X_future_scaled = scaler.transform(future_features)
            
            predictions = model.predict(X_future_scaled)
            
            # Calculate confidence intervals using quantile regression
            confidence_intervals = self._calculate_confidence_intervals(predictions, historical_data)
            
            # Determine trend direction
            recent_trend = np.mean(np.diff(predictions[-6:]))  # Last 6 hours
            trend_direction = "increasing" if recent_trend > 0 else "decreasing" if recent_trend < 0 else "stable"
            
            # Calculate anomaly probability
            anomaly_probability = self._calculate_anomaly_probability(predictions, historical_data)
            
            return PerformancePrediction(
                timestamp=future_timestamps[-1],
                predicted_value=float(np.mean(predictions)),
                confidence_interval=confidence_intervals,
                prediction_horizon_hours=self.prediction_horizon_hours,
                trend_direction=trend_direction,
                anomaly_probability=anomaly_probability
            )
            
        except Exception as e:
            logging.error(f"Error predicting CPU utilization: {str(e)}")
            raise
    
    def predict_memory_usage(self,
                           historical_data: np.ndarray,
                           timestamps: List[datetime],
                           growth_rate: float = 0.05) -> PerformancePrediction:
        """
        Predict memory usage with growth trend
        
        Args:
            historical_data: Historical memory usage data
            timestamps: Corresponding timestamps
            growth_rate: Expected daily growth rate
            
        Returns:
            PerformancePrediction with forecast
        """
        try:
            # Fit trend and seasonal components
            time_index = np.arange(len(historical_data))
            
            # Polynomial trend fitting
            trend_coeffs = np.polyfit(time_index, historical_data, 2)
            trend = np.polyval(trend_coeffs, time_index)
            
            # Remove trend to analyze seasonality
            detrended = historical_data - trend
            
            # Seasonal pattern (simple sine wave approximation)
            seasonal_period = 24  # Daily pattern
            seasonal_amplitude = np.std(detrended)
            seasonal_phase = np.arctan2(np.sum(detrended * np.sin(2 * np.pi * time_index / seasonal_period)),
                                      np.sum(detrended * np.cos(2 * np.pi * time_index / seasonal_period)))
            
            # Generate future predictions
            future_time_index = np.arange(len(historical_data), len(historical_data) + self.prediction_horizon_hours)
            
            # Predict trend with growth
            growth_factor = np.exp(growth_rate * future_time_index / 24)  # Daily growth
            future_trend = np.polyval(trend_coeffs, future_time_index) * growth_factor
            
            # Predict seasonal component
            future_seasonal = seasonal_amplitude * np.sin(2 * np.pi * future_time_index / seasonal_period + seasonal_phase)
            
            # Combine predictions
            predictions = future_trend + future_seasonal
            
            # Ensure predictions are positive
            predictions = np.maximum(predictions, 0)
            
            # Calculate confidence intervals
            residual_std = np.std(historical_data - trend)
            confidence_intervals = (np.mean(predictions) - 1.96 * residual_std,
                                 np.mean(predictions) + 1.96 * residual_std)
            
            # Determine trend direction
            trend_slope = np.polyval(np.polyder(trend_coeffs), future_time_index[-1])
            trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
            
            # Calculate anomaly probability
            anomaly_probability = self._calculate_anomaly_probability(predictions, historical_data)
            
            return PerformancePrediction(
                timestamp=timestamps[-1] + timedelta(hours=self.prediction_horizon_hours),
                predicted_value=float(np.mean(predictions)),
                confidence_interval=confidence_intervals,
                prediction_horizon_hours=self.prediction_horizon_hours,
                trend_direction=trend_direction,
                anomaly_probability=anomaly_probability
            )
            
        except Exception as e:
            logging.error(f"Error predicting memory usage: {str(e)}")
            raise
    
    def predict_database_performance(self,
                                   latency_data: np.ndarray,
                                   throughput_data: np.ndarray,
                                   timestamps: List[datetime]) -> Dict[str, PerformancePrediction]:
        """
        Predict database performance metrics
        
        Args:
            latency_data: Historical database latency data
            throughput_data: Historical database throughput data
            timestamps: Corresponding timestamps
            
        Returns:
            Dict with latency and throughput predictions
        """
        predictions = {}
        
        try:
            # Predict latency
            latency_prediction = self._predict_timeseries_with_trend(
                latency_data, timestamps, "database_latency")
            predictions['latency'] = latency_prediction
            
            # Predict throughput
            throughput_prediction = self._predict_timeseries_with_trend(
                throughput_data, timestamps, "database_throughput")
            predictions['throughput'] = throughput_prediction
            
            # Calculate correlation for joint analysis
            correlation = np.corrcoef(latency_data[-min(len(latency_data), 100):],
                                    throughput_data[-min(len(throughput_data), 100):])[0, 1]
            
            predictions['correlation'] = correlation
            
        except Exception as e:
            logging.error(f"Error predicting database performance: {str(e)}")
            
        return predictions
    
    def _predict_timeseries_with_trend(self,
                                     data: np.ndarray,
                                     timestamps: List[datetime],
                                     metric_name: str) -> PerformancePrediction:
        """Helper method for time series prediction with trend analysis"""
        try:
            # Simple linear trend prediction
            time_index = np.arange(len(data))
            coeffs = np.polyfit(time_index, data, 1)
            
            # Extrapolate trend
            future_time_index = np.arange(len(data), len(data) + self.prediction_horizon_hours)
            trend_predictions = np.polyval(coeffs, future_time_index)
            
            # Add seasonal component if sufficient data
            if len(data) > 48:  # At least 2 days of hourly data
                # Extract daily pattern
                daily_pattern = self._extract_daily_pattern(data)
                seasonal_predictions = [daily_pattern[hour % 24] for hour in future_time_index]
                
                # Combine trend and seasonal
                predictions = trend_predictions + seasonal_predictions - np.mean(seasonal_predictions)
            else:
                predictions = trend_predictions
            
            # Calculate confidence intervals
            residual_std = np.std(data - np.polyval(coeffs, time_index))
            confidence_intervals = (np.mean(predictions) - 1.96 * residual_std,
                                 np.mean(predictions) + 1.96 * residual_std)
            
            # Determine trend direction
            trend_direction = "increasing" if coeffs[0] > 0 else "decreasing" if coeffs[0] < 0 else "stable"
            
            # Calculate anomaly probability
            anomaly_probability = self._calculate_anomaly_probability(predictions, data)
            
            return PerformancePrediction(
                timestamp=timestamps[-1] + timedelta(hours=self.prediction_horizon_hours),
                predicted_value=float(np.mean(predictions)),
                confidence_interval=confidence_intervals,
                prediction_horizon_hours=self.prediction_horizon_hours,
                trend_direction=trend_direction,
                anomaly_probability=anomaly_probability
            )
            
        except Exception as e:
            logging.error(f"Error in time series prediction: {str(e)}")
            raise
    
    def _prepare_time_series_features(self,
                                    data: np.ndarray,
                                    timestamps: List[datetime],
                                    feature_lags: int = 5) -> np.ndarray:
        """Prepare features for machine learning models"""
        features = []
        
        for i in range(feature_lags, len(data)):
            # Lag features
            lag_features = data[i-feature_lags:i]
            
            # Time-based features
            current_time = timestamps[i]
            hour_feature = current_time.hour / 24.0
            day_of_week_feature = current_time.weekday() / 7.0
            
            # Statistical features
            rolling_mean = np.mean(data[max(0, i-24):i])
            rolling_std = np.std(data[max(0, i-24):i])
            
            # Combine features
            feature_vector = np.concatenate([
                lag_features,
                [hour_feature, day_of_week_feature, rolling_mean, rolling_std]
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _prepare_forecast_features(self,
                                 historical_data: np.ndarray,
                                 historical_timestamps: List[datetime],
                                 future_timestamps: List[datetime]) -> np.ndarray:
        """Prepare features for forecasting"""
        features = []
        
        for i, future_time in enumerate(future_timestamps):
            # Use recent data as base
            recent_data = historical_data[-5:] if len(historical_data) >= 5 else historical_data
            
            # Create lag features (using recent data)
            lag_features = recent_data
            
            # Time-based features
            hour_feature = future_time.hour / 24.0
            day_of_week_feature = future_time.weekday() / 7.0
            
            # Statistical features
            rolling_mean = np.mean(historical_data[-24:]) if len(historical_data) >= 24 else np.mean(historical_data)
            rolling_std = np.std(historical_data[-24:]) if len(historical_data) >= 24 else np.std(historical_data)
            
            # Combine features
            feature_vector = np.concatenate([
                lag_features,
                [hour_feature, day_of_week_feature, rolling_mean, rolling_std]
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _generate_future_timestamps(self,
                                  historical_timestamps: List[datetime],
                                  hours_ahead: int) -> List[datetime]:
        """Generate future timestamps for prediction"""
        if not historical_timestamps:
            return []
        
        last_timestamp = historical_timestamps[-1]
        return [last_timestamp + timedelta(hours=i+1) for i in range(hours_ahead)]
    
    def _calculate_confidence_intervals(self,
                                      predictions: np.ndarray,
                                      historical_data: np.ndarray,
                                      confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for predictions"""
        # Calculate prediction error from historical fit
        time_index = np.arange(len(historical_data))
        
        try:
            # Fit simple linear model for error estimation
            coeffs = np.polyfit(time_index, historical_data, 1)
            predictions_historical = np.polyval(coeffs, time_index)
            residuals = historical_data - predictions_historical
            
            # Calculate standard error
            residual_std = np.std(residuals)
            
            # Calculate confidence interval
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            mean_prediction = np.mean(predictions)
            
            margin_of_error = z_score * residual_std / np.sqrt(len(predictions))
            
            return (mean_prediction - margin_of_error, mean_prediction + margin_of_error)
            
        except Exception:
            # Fallback to simple std-based confidence interval
            residual_std = np.std(historical_data)
            mean_prediction = np.mean(predictions)
            return (mean_prediction - 2 * residual_std, mean_prediction + 2 * residual_std)
    
    def _calculate_anomaly_probability(self,
                                     predictions: np.ndarray,
                                     historical_data: np.ndarray) -> float:
        """Calculate probability of anomalies in predictions"""
        try:
            # Use historical data to establish normal range
            historical_std = np.std(historical_data)
            historical_mean = np.mean(historical_data)
            
            # Calculate how far predictions deviate from normal range
            prediction_std = np.std(predictions)
            mean_deviation = abs(np.mean(predictions) - historical_mean)
            
            # Simple probability calculation
            if mean_deviation > 2 * historical_std:
                return min(0.9, mean_deviation / (3 * historical_std))
            elif mean_deviation > historical_std:
                return min(0.7, mean_deviation / (2 * historical_std))
            else:
                return min(0.3, mean_deviation / historical_std)
                
        except Exception:
            return 0.1  # Default low anomaly probability
    
    def _extract_daily_pattern(self, data: np.ndarray) -> List[float]:
        """Extract daily pattern from time series data"""
        if len(data) < 24:
            return [np.mean(data)] * 24
        
        # Create daily buckets
        daily_buckets = [[] for _ in range(24)]
        
        for i, value in enumerate(data):
            hour = i % 24
            daily_buckets[hour].append(value)
        
        # Calculate average for each hour
        daily_pattern = [np.mean(bucket) if bucket else 0 for bucket in daily_buckets]
        
        # Normalize pattern (remove mean)
        pattern_mean = np.mean(daily_pattern)
        return [x - pattern_mean for x in daily_pattern]

class CapacityPlanningSystem:
    """Advanced capacity planning for medical AI infrastructure"""
    
    def __init__(self,
                 capacity_threshold_warning: float = 0.8,
                 capacity_threshold_critical: float = 0.9):
        """
        Initialize capacity planning system
        
        Args:
            capacity_threshold_warning: Warning threshold (80%)
            capacity_threshold_critical: Critical threshold (90%)
        """
        self.warning_threshold = capacity_threshold_warning
        self.critical_threshold = capacity_threshold_critical
        
    def analyze_cpu_capacity(self,
                           historical_utilization: np.ndarray,
                           timestamps: List[datetime],
                           growth_rate: float = 0.02) -> CapacityPlanningForecast:
        """
        Analyze CPU capacity and predict capacity needs
        
        Args:
            historical_utilization: CPU utilization history
            timestamps: Corresponding timestamps
            growth_rate: Expected growth rate per day
            
        Returns:
            CapacityPlanningForecast with recommendations
        """
        try:
            current_utilization = float(historical_utilization[-1])
            
            # Fit growth trend
            time_index = np.arange(len(historical_utilization))
            growth_coeffs = np.polyfit(time_index, historical_utilization, 1)
            
            # Project utilization over next 90 days
            projection_days = 90
            future_time_index = np.arange(len(historical_utilization), 
                                        len(historical_utilization) + projection_days * 24)
            
            # Apply growth rate
            growth_factor = np.exp(growth_rate * future_time_index / 24)
            projected_utilization = np.polyval(growth_coeffs, future_time_index) * growth_factor
            
            # Find when utilization exceeds thresholds
            warning_date = self._find_threshold_date(projected_utilization, future_time_index, 
                                                   timestamps, self.warning_threshold)
            critical_date = self._find_threshold_date(projected_utilization, future_time_index,
                                                     timestamps, self.critical_threshold)
            
            # Calculate recommended capacity increase
            predicted_peak = float(np.max(projected_utilization))
            current_capacity = 100.0  # Assuming 100% is full capacity
            
            if predicted_peak > self.warning_threshold * 100:
                recommended_increase = (predicted_peak - self.warning_threshold * 100) / 100
            else:
                recommended_increase = 0.0
            
            # Calculate confidence level based on model fit quality
            historical_fit = np.polyval(growth_coeffs, time_index)
            r_squared = 1 - np.sum((historical_utilization - historical_fit) ** 2) / np.sum((historical_utilization - np.mean(historical_utilization)) ** 2)
            confidence_level = max(0.5, min(0.95, r_squared))
            
            return CapacityPlanningForecast(
                resource_type="cpu",
                current_utilization=current_utilization,
                predicted_peak_utilization=predicted_peak,
                capacity_threshold_date=warning_date or critical_date or (timestamps[-1] + timedelta(days=30)),
                recommended_capacity_increase=recommended_increase,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logging.error(f"Error in CPU capacity analysis: {str(e)}")
            raise
    
    def analyze_storage_capacity(self,
                               historical_usage: np.ndarray,
                               timestamps: List[datetime],
                               data_retention_days: int = 2555) -> CapacityPlanningForecast:
        """
        Analyze storage capacity needs
        
        Args:
            historical_usage: Storage usage history in GB
            timestamps: Corresponding timestamps
            data_retention_days: Data retention policy in days
            
        Returns:
            CapacityPlanningForecast with storage recommendations
        """
        try:
            current_usage = float(historical_usage[-1])
            
            # Analyze growth pattern
            if len(historical_usage) >= 30:  # At least 30 days of data
                # Fit exponential growth model
                time_index = np.arange(len(historical_usage))
                log_usage = np.log(np.maximum(historical_usage, 1))  # Avoid log(0)
                
                growth_coeffs = np.polyfit(time_index, log_usage, 1)
                growth_rate = growth_coeffs[0]
                
                # Project storage needs for retention period
                projection_days = min(data_retention_days, 365)  # Cap at 1 year
                future_time_index = np.arange(len(historical_usage), 
                                            len(historical_usage) + projection_days)
                
                projected_usage = np.exp(np.polyval(growth_coeffs, future_time_index))
                predicted_peak = float(np.max(projected_usage))
                
                # Find threshold dates
                warning_threshold = current_usage * 1.2  # 20% above current
                critical_threshold = current_usage * 1.5  # 50% above current
                
                warning_date = self._find_storage_threshold_date(projected_usage, future_time_index,
                                                               timestamps, warning_threshold)
                critical_date = self._find_storage_threshold_date(projected_usage, future_time_index,
                                                                 timestamps, critical_threshold)
                
            else:
                # Simple linear projection for insufficient data
                daily_growth = np.mean(np.diff(historical_usage[-7:]))  # Average daily growth
                predicted_peak = current_usage + daily_growth * 365  # 1 year projection
                warning_date = timestamps[-1] + timedelta(days=90)
                critical_date = timestamps[-1] + timedelta(days=180)
            
            # Calculate recommended capacity increase
            buffer_factor = 1.3  # 30% buffer
            recommended_capacity = predicted_peak * buffer_factor
            recommended_increase = (recommended_capacity - current_usage) / current_usage
            
            # Confidence level based on data availability
            confidence_level = min(0.9, len(historical_usage) / 365)
            
            return CapacityPlanningForecast(
                resource_type="storage",
                current_utilization=current_usage,
                predicted_peak_utilization=predicted_peak,
                capacity_threshold_date=warning_date or critical_date or (timestamps[-1] + timedelta(days=180)),
                recommended_capacity_increase=max(0, recommended_increase),
                confidence_level=confidence_level
            )
            
        except Exception as e:
            logging.error(f"Error in storage capacity analysis: {str(e)}")
            raise
    
    def analyze_network_capacity(self,
                               bandwidth_usage: np.ndarray,
                               latency_measurements: np.ndarray,
                               timestamps: List[datetime]) -> Dict[str, CapacityPlanningForecast]:
        """
        Analyze network capacity requirements
        
        Args:
            bandwidth_usage: Network bandwidth usage history
            latency_measurements: Network latency history
            timestamps: Corresponding timestamps
            
        Returns:
            Dict with bandwidth and latency forecasts
        """
        forecasts = {}
        
        try:
            # Bandwidth analysis
            current_bandwidth_usage = float(bandwidth_usage[-1])
            bandwidth_growth = np.mean(np.diff(bandwidth_usage[-30:])) if len(bandwidth_usage) >= 30 else 0
            
            projected_bandwidth = current_bandwidth_usage + bandwidth_growth * 90  # 90 days projection
            recommended_bandwidth_increase = max(0, (projected_bandwidth - current_bandwidth_usage) / current_bandwidth_usage)
            
            forecasts['bandwidth'] = CapacityPlanningForecast(
                resource_type="network_bandwidth",
                current_utilization=current_bandwidth_usage,
                predicted_peak_utilization=projected_bandwidth,
                capacity_threshold_date=timestamps[-1] + timedelta(days=60),
                recommended_capacity_increase=recommended_bandwidth_increase,
                confidence_level=0.7
            )
            
            # Latency analysis (capacity in terms of acceptable latency)
            current_latency = float(np.median(latency_measurements[-24:]))  # Median of last 24 hours
            latency_growth = np.mean(np.diff(latency_measurements[-30:])) if len(latency_measurements) >= 30 else 0
            
            projected_latency = current_latency + latency_growth * 90
            latency_threshold = 100  # ms - acceptable threshold
            
            forecasts['latency'] = CapacityPlanningForecast(
                resource_type="network_latency",
                current_utilization=current_latency,
                predicted_peak_utilization=projected_latency,
                capacity_threshold_date=timestamps[-1] + timedelta(days=30) if projected_latency > latency_threshold else None,
                recommended_capacity_increase=0.2 if projected_latency > latency_threshold else 0.0,
                confidence_level=0.6
            )
            
        except Exception as e:
            logging.error(f"Error in network capacity analysis: {str(e)}")
            
        return forecasts
    
    def _find_threshold_date(self,
                           projected_values: np.ndarray,
                           future_time_index: np.ndarray,
                           base_timestamps: List[datetime],
                           threshold: float) -> Optional[datetime]:
        """Find when projected values exceed threshold"""
        try:
            threshold_exceeded = projected_values > threshold
            if np.any(threshold_exceeded):
                first_exceed = np.where(threshold_exceeded)[0][0]
                hours_ahead = future_time_index[first_exceed]
                return base_timestamps[-1] + timedelta(hours=hours_ahead)
        except Exception:
            pass
        return None
    
    def _find_storage_threshold_date(self,
                                   projected_usage: np.ndarray,
                                   future_time_index: np.ndarray,
                                   base_timestamps: List[datetime],
                                   threshold: float) -> Optional[datetime]:
        """Find when storage usage exceeds threshold"""
        try:
            threshold_exceeded = projected_usage > threshold
            if np.any(threshold_exceeded):
                first_exceed = np.where(threshold_exceeded)[0][0]
                days_ahead = future_time_index[first_exceed]
                return base_timestamps[-1] + timedelta(days=days_ahead)
        except Exception:
            pass
        return None

class PredictiveOrchestrator:
    """Orchestrates all predictive analytics activities"""
    
    def __init__(self,
                 prediction_horizon_hours: int = 24,
                 capacity_planning_days: int = 90):
        """
        Initialize predictive orchestrator
        
        Args:
            prediction_horizon_hours: Hours for performance predictions
            capacity_planning_days: Days for capacity planning
        """
        self.prediction_horizon_hours = prediction_horizon_hours
        self.capacity_planning_days = capacity_planning_days
        
        self.anomaly_detector = TimeSeriesAnomalyDetector()
        self.performance_predictor = PerformancePredictor(prediction_horizon_hours)
        self.capacity_planner = CapacityPlanningSystem()
        
        self.prediction_history = []
        self.capacity_forecasts = []
        
    def run_comprehensive_prediction(self,
                                   system_metrics: Dict[str, np.ndarray],
                                   timestamps: List[datetime],
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run comprehensive predictive analytics
        
        Args:
            system_metrics: Dict mapping metric names to time series data
            timestamps: Corresponding timestamps
            metadata: Additional metadata about the system
            
        Returns:
            Dict containing comprehensive predictive analysis
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'prediction_horizon_hours': self.prediction_horizon_hours,
            'anomaly_detection': {},
            'performance_predictions': {},
            'capacity_planning': {},
            'recommendations': [],
            'overall_health_trend': 'stable'
        }
        
        try:
            # Anomaly detection for each metric
            for metric_name, data in system_metrics.items():
                if len(data) >= 24:  # Minimum data requirement
                    # Statistical anomalies
                    stats_anomalies = self.anomaly_detector.detect_statistical_anomalies(data, timestamps)
                    
                    # Isolation Forest anomalies
                    iso_anomalies = self.anomaly_detector.detect_isolation_forest_anomalies(data)
                    
                    # Seasonal anomalies
                    seasonal_anomalies = self.anomaly_detector.detect_seasonal_anomalies(data, timestamps)
                    
                    results['anomaly_detection'][metric_name] = {
                        'statistical': stats_anomalies,
                        'isolation_forest': iso_anomalies,
                        'seasonal': seasonal_anomalies,
                        'anomaly_summary': self._summarize_anomalies(stats_anomalies, iso_anomalies, seasonal_anomalies)
                    }
            
            # Performance predictions
            if 'cpu_utilization' in system_metrics:
                results['performance_predictions']['cpu'] = self.performance_predictor.predict_cpu_utilization(
                    system_metrics['cpu_utilization'], timestamps, metadata)
            
            if 'memory_usage' in system_metrics:
                results['performance_predictions']['memory'] = self.performance_predictor.predict_memory_usage(
                    system_metrics['memory_usage'], timestamps)
            
            if 'database_latency' in system_metrics and 'database_throughput' in system_metrics:
                results['performance_predictions']['database'] = self.performance_predictor.predict_database_performance(
                    system_metrics['database_latency'], system_metrics['database_throughput'], timestamps)
            
            # Capacity planning
            if 'cpu_utilization' in system_metrics:
                cpu_capacity = self.capacity_planner.analyze_cpu_capacity(
                    system_metrics['cpu_utilization'], timestamps)
                results['capacity_planning']['cpu'] = cpu_capacity
                self.capacity_forecasts.append(cpu_capacity)
            
            if 'storage_usage' in system_metrics:
                storage_capacity = self.capacity_planner.analyze_storage_capacity(
                    system_metrics['storage_usage'], timestamps)
                results['capacity_planning']['storage'] = storage_capacity
                self.capacity_forecasts.append(storage_capacity)
            
            if 'network_bandwidth' in system_metrics and 'network_latency' in system_metrics:
                network_capacity = self.capacity_planner.analyze_network_capacity(
                    system_metrics['network_bandwidth'], system_metrics['network_latency'], timestamps)
                results['capacity_planning']['network'] = network_capacity
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # Overall health trend
            results['overall_health_trend'] = self._calculate_health_trend(results)
            
            # Store prediction history
            self.prediction_history.append(results)
            
        except Exception as e:
            logging.error(f"Error in comprehensive prediction: {str(e)}")
            results['error'] = str(e)
            
        return results
    
    def _summarize_anomalies(self,
                           stats_anomalies: Dict[str, Any],
                           iso_anomalies: Dict[str, Any],
                           seasonal_anomalies: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize anomaly detection results from multiple methods"""
        summary = {
            'total_anomalies': 0,
            'high_confidence_anomalies': 0,
            'anomaly_types': []
        }
        
        try:
            # Count anomalies from each method
            stats_count = stats_anomalies.get('anomaly_count', 0)
            iso_count = iso_anomalies.get('anomaly_count', 0)
            seasonal_count = len(seasonal_anomalies.get('seasonal_anomalies', [])) + len(seasonal_anomalies.get('trend_anomalies', []))
            
            summary['total_anomalies'] = stats_count + iso_count + seasonal_count
            
            # High confidence anomalies (detected by multiple methods)
            summary['high_confidence_anomalies'] = max(stats_count, iso_count, seasonal_count // 2)
            
            # Anomaly types
            if stats_count > 0:
                summary['anomaly_types'].append('statistical_outlier')
            if iso_count > 0:
                summary['anomaly_types'].append('isolation_anomaly')
            if seasonal_count > 0:
                summary['anomaly_types'].append('seasonal_deviation')
                
        except Exception as e:
            logging.error(f"Error summarizing anomalies: {str(e)}")
            
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on predictions"""
        recommendations = []
        
        try:
            # Performance-based recommendations
            perf_predictions = results.get('performance_predictions', {})
            
            for resource, prediction in perf_predictions.items():
                if hasattr(prediction, 'anomaly_probability') and prediction.anomaly_probability > 0.7:
                    recommendations.append(f"High anomaly probability predicted for {resource} - monitor closely")
                
                if hasattr(prediction, 'trend_direction') and prediction.trend_direction == 'increasing':
                    recommendations.append(f"Increasing trend detected in {resource} - prepare for scaling")
            
            # Capacity-based recommendations
            capacity_planning = results.get('capacity_planning', {})
            
            for resource, forecast in capacity_planning.items():
                if isinstance(forecast, CapacityPlanningForecast):
                    if forecast.capacity_threshold_date:
                        days_until_threshold = (forecast.capacity_threshold_date - datetime.now()).days
                        if days_until_threshold < 30:
                            recommendations.append(f"Critical: {resource} capacity threshold in {days_until_threshold} days")
                        elif days_until_threshold < 90:
                            recommendations.append(f"Warning: {resource} capacity threshold in {days_until_threshold} days")
                    
                    if forecast.recommended_capacity_increase > 0.2:
                        recommendations.append(f"Consider {resource} capacity increase of {forecast.recommended_capacity_increase:.1%}")
            
            # Anomaly-based recommendations
            anomaly_detection = results.get('anomaly_detection', {})
            
            for metric, detection_results in anomaly_detection.items():
                summary = detection_results.get('anomaly_summary', {})
                if summary.get('total_anomalies', 0) > 10:
                    recommendations.append(f"Frequent anomalies detected in {metric} - investigate root cause")
            
            # Default recommendation if none generated
            if not recommendations:
                recommendations.append("System performance within normal parameters - continue monitoring")
                
        except Exception as e:
            logging.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate specific recommendations")
            
        return recommendations
    
    def _calculate_health_trend(self, results: Dict[str, Any]) -> str:
        """Calculate overall system health trend"""
        try:
            trend_scores = []
            
            # Anomaly trend
            anomaly_detection = results.get('anomaly_detection', {})
            total_anomalies = sum(
                detection.get('anomaly_summary', {}).get('total_anomalies', 0)
                for detection in anomaly_detection.values()
            )
            
            if total_anomalies > 20:
                trend_scores.append(-1)  # Declining
            elif total_anomalies > 5:
                trend_scores.append(0)   # Stable
            else:
                trend_scores.append(1)   # Improving
            
            # Performance trend
            perf_predictions = results.get('performance_predictions', {})
            for resource, prediction in perf_predictions.items():
                if hasattr(prediction, 'trend_direction'):
                    if prediction.trend_direction == 'increasing':
                        trend_scores.append(-0.5)  # Potentially concerning
                    elif prediction.trend_direction == 'decreasing':
                        trend_scores.append(0.5)   # Improving
                    else:
                        trend_scores.append(0)     # Stable
            
            # Capacity trend
            capacity_planning = results.get('capacity_planning', {})
            critical_capacity_issues = sum(
                1 for forecast in capacity_planning.values()
                if isinstance(forecast, CapacityPlanningForecast) 
                and forecast.capacity_threshold_date 
                and (forecast.capacity_threshold_date - datetime.now()).days < 30
            )
            
            if critical_capacity_issues > 0:
                trend_scores.append(-1)  # Declining
            elif critical_capacity_issues > 2:
                trend_scores.append(-0.5)  # Warning
            else:
                trend_scores.append(1)   # Good
            
            # Calculate overall trend
            if trend_scores:
                avg_score = np.mean(trend_scores)
                if avg_score > 0.5:
                    return 'improving'
                elif avg_score < -0.5:
                    return 'declining'
                else:
                    return 'stable'
            else:
                return 'stable'
                
        except Exception as e:
            logging.error(f"Error calculating health trend: {str(e)}")
            return 'unknown'

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize predictive orchestrator
    orchestrator = PredictiveOrchestrator(
        prediction_horizon_hours=24,
        capacity_planning_days=90
    )
    
    # Generate sample system metrics
    np.random.seed(42)
    n_hours = 168  # 1 week of hourly data
    
    # Create realistic system metrics
    base_cpu = 50
    cpu_trend = np.linspace(0, 10, n_hours)  # Gradual increase
    cpu_noise = np.random.normal(0, 10, n_hours)
    cpu_utilization = np.maximum(0, np.minimum(100, base_cpu + cpu_trend + cpu_noise))
    
    # Memory usage with daily patterns
    hours = np.arange(n_hours)
    daily_pattern = 10 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
    memory_base = 70
    memory_usage = np.maximum(0, memory_base + daily_pattern + np.random.normal(0, 5, n_hours))
    
    # Database metrics
    database_latency = 50 + np.random.normal(0, 20, n_hours)
    database_latency = np.maximum(10, database_latency)
    database_throughput = 1000 + np.random.normal(0, 200, n_hours)
    
    # Storage growth
    storage_start = 1000
    storage_growth = np.linspace(0, 100, n_hours)  # GB growth
    storage_usage = storage_start + storage_growth + np.random.normal(0, 20, n_hours)
    
    # Network metrics
    bandwidth_usage = 500 + np.random.normal(0, 100, n_hours)
    network_latency = 20 + np.random.normal(0, 5, n_hours)
    
    # Generate timestamps
    base_time = datetime.now() - timedelta(hours=n_hours)
    timestamps = [base_time + timedelta(hours=i) for i in range(n_hours)]
    
    # System metrics dictionary
    system_metrics = {
        'cpu_utilization': cpu_utilization,
        'memory_usage': memory_usage,
        'database_latency': database_latency,
        'database_throughput': database_throughput,
        'storage_usage': storage_usage,
        'network_bandwidth': bandwidth_usage,
        'network_latency': network_latency
    }
    
    # Run comprehensive prediction
    results = orchestrator.run_comprehensive_prediction(
        system_metrics=system_metrics,
        timestamps=timestamps,
        metadata={'environment': 'production', 'region': 'us-east-1'}
    )
    
    # Print results
    print("=== Predictive Analytics Results ===")
    print(f"Prediction Horizon: {results['prediction_horizon_hours']} hours")
    print(f"Overall Health Trend: {results['overall_health_trend']}")
    print(f"Number of Recommendations: {len(results['recommendations'])}")
    
    print("\n=== Performance Predictions ===")
    for resource, prediction in results.get('performance_predictions', {}).items():
        print(f"{resource}:")
        print(f"  Predicted Value: {prediction.predicted_value:.2f}")
        print(f"  Trend: {prediction.trend_direction}")
        print(f"  Anomaly Probability: {prediction.anomaly_probability:.3f}")
    
    print("\n=== Capacity Planning ===")
    for resource, forecast in results.get('capacity_planning', {}).items():
        if isinstance(forecast, CapacityPlanningForecast):
            print(f"{resource}:")
            print(f"  Current Utilization: {forecast.current_utilization:.2f}")
            print(f"  Predicted Peak: {forecast.predicted_peak_utilization:.2f}")
            print(f"  Recommended Increase: {forecast.recommended_capacity_increase:.1%}")
            if forecast.capacity_threshold_date:
                print(f"  Threshold Date: {forecast.capacity_threshold_date.strftime('%Y-%m-%d')}")
    
    print("\n=== Recommendations ===")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"{i}. {rec}")
    
    print("\n=== Anomaly Detection Summary ===")
    for metric, detection in results.get('anomaly_detection', {}).items():
        summary = detection.get('anomaly_summary', {})
        print(f"{metric}: {summary.get('total_anomalies', 0)} anomalies detected")