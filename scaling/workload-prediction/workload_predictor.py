"""
Workload Prediction System for Medical AI Auto-scaling
Predicts healthcare workloads using historical patterns and machine learning
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import deque
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sqlite3
import pickle

logger = logging.getLogger(__name__)

@dataclass
class WorkloadSample:
    """Individual workload measurement"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    request_rate: float
    active_users: int
    response_time: float
    endpoint_category: str
    day_of_week: int
    hour_of_day: int
    is_business_hours: bool
    is_emergency_period: bool

@dataclass
class PredictionResult:
    """Workload prediction result"""
    timestamp: datetime
    predicted_cpu: float
    predicted_memory: float
    predicted_request_rate: float
    predicted_response_time: float
    confidence_score: float
    recommended_replicas: int
    reasoning: str

class HealthcareWorkloadPredictor:
    """
    Advanced workload prediction system for medical AI workloads
    Uses historical patterns, time series analysis, and ML models
    """
    
    def __init__(self, 
                 prediction_horizon: int = 24,  # hours
                 model_retrain_interval: int = 24,  # hours
                 database_path: str = "workload_data.db"):
        self.prediction_horizon = prediction_horizon
        self.model_retrain_interval = model_retrain_interval
        self.database_path = database_path
        
        # ML models
        self.cpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.memory_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.request_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.response_time_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Model components
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Data storage
        self.workload_history = deque(maxlen=10000)  # Store last 10k samples
        self.predictions_cache = {}
        
        # Healthcare-specific patterns
        self.healthcare_patterns = {
            'peak_hours': [(6, 9), (14, 17)],  # Morning and afternoon rounds
            'low_hours': [(22, 24), (0, 6)],   # Night hours
            'peak_days': [0, 1, 2, 3, 4],      # Weekdays
            'emergency_indicators': {
                'time_patterns': [(0, 2), (8, 10), (16, 18)],  # Common emergency times
                'load_surge_threshold': 0.8
            }
        }
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for workload data"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workload_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                cpu_usage REAL,
                memory_usage REAL,
                request_rate REAL,
                active_users INTEGER,
                response_time REAL,
                endpoint_category TEXT,
                day_of_week INTEGER,
                hour_of_day INTEGER,
                is_business_hours BOOLEAN,
                is_emergency_period BOOLEAN
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_time DATETIME,
                target_time DATETIME,
                predicted_cpu REAL,
                predicted_memory REAL,
                predicted_request_rate REAL,
                predicted_response_time REAL,
                confidence_score REAL,
                recommended_replicas INTEGER,
                reasoning TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_workload_sample(self, sample: WorkloadSample):
        """Add workload measurement sample"""
        self.workload_history.append(sample)
        
        # Store in database
        self._store_sample_in_db(sample)
        
        # Update predictions cache
        if not self.models_trained and len(self.workload_history) > 100:
            self._train_models()
    
    def _store_sample_in_db(self, sample: WorkloadSample):
        """Store sample in SQLite database"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO workload_samples 
            (timestamp, cpu_usage, memory_usage, request_rate, active_users, 
             response_time, endpoint_category, day_of_week, hour_of_day, 
             is_business_hours, is_emergency_period)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample.timestamp, sample.cpu_usage, sample.memory_usage,
            sample.request_rate, sample.active_users, sample.response_time,
            sample.endpoint_category, sample.day_of_week, sample.hour_of_day,
            sample.is_business_hours, sample.is_emergency_period
        ))
        
        conn.commit()
        conn.close()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from historical samples"""
        if len(self.workload_history) < 50:
            logger.warning("Not enough data for training")
            return None, None, None, None
        
        # Convert samples to DataFrame
        data = []
        for sample in self.workload_history:
            data.append(asdict(sample))
        
        df = pd.DataFrame(data)
        
        # Features for prediction
        feature_columns = [
            'day_of_week', 'hour_of_day', 'is_business_hours', 'is_emergency_period',
            'endpoint_category'  # We'll encode this numerically
        ]
        
        # Encode endpoint category
        df['endpoint_category_encoded'] = pd.Categorical(df['endpoint_category']).codes
        
        # Features matrix
        X = df[feature_columns + ['endpoint_category_encoded']].values
        
        # Target variables
        y_cpu = df['cpu_usage'].values
        y_memory = df['memory_usage'].values
        y_request_rate = df['request_rate'].values
        y_response_time = df['response_time'].values
        
        return X, y_cpu, y_memory, y_request_rate, y_response_time
    
    def _train_models(self):
        """Train ML models for workload prediction"""
        try:
            X, y_cpu, y_memory, y_request_rate, y_response_time = self._prepare_training_data()
            
            if X is None:
                return
            
            # Split data
            X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(X, y_cpu, test_size=0.2)
            _, _, y_mem_train, y_mem_test = train_test_split(X, y_memory, test_size=0.2)
            _, _, y_req_train, y_req_test = train_test_split(X, y_request_rate, test_size=0.2)
            _, _, y_resp_train, y_resp_test = train_test_split(X, y_response_time, test_size=0.2)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.cpu_model.fit(X_train_scaled, y_cpu_train)
            self.memory_model.fit(X_train_scaled, y_mem_train)
            self.request_model.fit(X_train_scaled, y_req_train)
            self.response_time_model.fit(X_train_scaled, y_resp_train)
            
            # Calculate training scores
            cpu_score = self.cpu_model.score(X_test_scaled, y_cpu_test)
            mem_score = self.memory_model.score(X_test_scaled, y_mem_test)
            req_score = self.request_model.score(X_test_scaled, y_req_test)
            resp_score = self.response_time_model.score(X_test_scaled, y_resp_test)
            
            logger.info(f"Model training completed - CPU: {cpu_score:.3f}, Memory: {mem_score:.3f}, "
                       f"Request Rate: {req_score:.3f}, Response Time: {resp_score:.3f}")
            
            self.models_trained = True
            
            # Save models
            self._save_models()
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            model_data = {
                'cpu_model': self.cpu_model,
                'memory_model': self.memory_model,
                'request_model': self.request_model,
                'response_time_model': self.response_time_model,
                'scaler': self.scaler
            }
            
            with open('workload_models.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            with open('workload_models.pkl', 'rb') as f:
                model_data = pickle.load(f)
            
            self.cpu_model = model_data['cpu_model']
            self.memory_model = model_data['memory_model']
            self.request_model = model_data['request_model']
            self.response_time_model = model_data['response_time_model']
            self.scaler = model_data['scaler']
            
            self.models_trained = True
            logger.info("Models loaded successfully")
            
        except FileNotFoundError:
            logger.warning("No saved models found, will train from scratch")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def predict_workload(self, target_time: datetime) -> PredictionResult:
        """Predict workload for a specific time"""
        
        # Check cache first
        cache_key = target_time.strftime('%Y-%m-%d_%H')
        if cache_key in self.predictions_cache:
            return self.predictions_cache[cache_key]
        
        # If models not trained, use pattern-based prediction
        if not self.models_trained:
            return self._pattern_based_prediction(target_time)
        
        # Prepare features
        features = self._prepare_features(target_time)
        
        # Make predictions
        try:
            features_scaled = self.scaler.transform([features])
            
            predicted_cpu = self.cpu_model.predict(features_scaled)[0]
            predicted_memory = self.memory_model.predict(features_scaled)[0]
            predicted_request_rate = self.request_model.predict(features_scaled)[0]
            predicted_response_time = self.response_time_model.predict(features_scaled)[0]
            
            # Calculate confidence score
            confidence = self._calculate_confidence(target_time, features)
            
            # Calculate recommended replicas
            recommended_replicas = self._calculate_recommended_replicas(
                predicted_cpu, predicted_memory, predicted_request_rate
            )
            
            result = PredictionResult(
                timestamp=datetime.now(),
                predicted_cpu=max(0, min(100, predicted_cpu)),
                predicted_memory=max(0, min(100, predicted_memory)),
                predicted_request_rate=max(0, predicted_request_rate),
                predicted_response_time=max(0.1, predicted_response_time),
                confidence_score=confidence,
                recommended_replicas=recommended_replicas,
                reasoning=self._generate_reasoning(target_time, predicted_cpu, predicted_memory)
            )
            
            # Cache result
            self.predictions_cache[cache_key] = result
            
            # Store in database
            self._store_prediction_in_db(target_time, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._pattern_based_prediction(target_time)
    
    def _prepare_features(self, target_time: datetime) -> List[float]:
        """Prepare feature vector for prediction"""
        day_of_week = target_time.weekday()
        hour_of_day = target_time.hour
        
        # Check business hours (7 AM - 7 PM)
        is_business_hours = 7 <= hour_of_day <= 19
        
        # Check for emergency periods
        is_emergency_period = self._is_emergency_period(target_time)
        
        # Encode endpoint category (default to most common)
        endpoint_category = "general_api"
        
        # Encode category
        category_encoding = {"patient_api": 0, "clinical_api": 1, "ai_inference": 2, "general_api": 3}
        endpoint_encoded = category_encoding.get(endpoint_category, 3)
        
        return [
            day_of_week,
            hour_of_day,
            1 if is_business_hours else 0,
            1 if is_emergency_period else 0,
            endpoint_encoded
        ]
    
    def _is_emergency_period(self, target_time: datetime) -> bool:
        """Check if time falls in typical emergency periods"""
        hour = target_time.hour
        
        # Check common emergency times
        emergency_ranges = self.healthcare_patterns['emergency_indicators']['time_patterns']
        
        for start_hour, end_hour in emergency_ranges:
            if start_hour <= hour <= end_hour:
                return True
        
        return False
    
    def _calculate_confidence(self, target_time: datetime, features: List[float]) -> float:
        """Calculate confidence score for prediction"""
        base_confidence = 0.7
        
        # Boost confidence during predictable hours
        if 8 <= target_time.hour <= 17:  # Business hours
            base_confidence += 0.1
        
        # Reduce confidence for unusual patterns
        if target_time.weekday() in [5, 6]:  # Weekends
            base_confidence -= 0.1
        
        # Recent data availability bonus
        recent_data_bonus = min(0.1, len(self.workload_history) / 1000 * 0.1)
        base_confidence += recent_data_bonus
        
        return max(0.3, min(0.95, base_confidence))
    
    def _calculate_recommended_replicas(self, cpu: float, memory: float, request_rate: float) -> int:
        """Calculate recommended number of replicas based on predicted load"""
        
        # Base calculation considering multiple factors
        cpu_factor = cpu / 70  # Target 70% CPU utilization
        memory_factor = memory / 80  # Target 80% memory utilization
        request_factor = request_rate / 100  # Target 100 requests per second
        
        # Take maximum to handle different types of bottlenecks
        max_factor = max(cpu_factor, memory_factor, request_factor)
        
        # Base replica count
        base_replicas = 2
        
        # Calculate recommended replicas
        recommended = int(base_replicas * max_factor)
        
        # Ensure reasonable bounds
        return max(1, min(20, recommended))
    
    def _generate_reasoning(self, target_time: datetime, cpu: float, memory: float) -> str:
        """Generate human-readable reasoning for prediction"""
        hour = target_time.hour
        day = target_time.strftime("%A")
        
        if 8 <= hour <= 11:
            return f"Morning rounds typically see increased activity. {day} pattern shows higher usage."
        elif 14 <= hour <= 16:
            return f"Afternoon clinical activities typically require more resources."
        elif hour <= 6 or hour >= 22:
            return f"Night hours ({day}) typically have reduced but steady load."
        else:
            return f"Standard {day} activity pattern with moderate resource usage."
    
    def _pattern_based_prediction(self, target_time: datetime) -> PredictionResult:
        """Fallback prediction using healthcare patterns when ML models unavailable"""
        
        hour = target_time.hour
        day_of_week = target_time.weekday()
        
        # Healthcare workload patterns
        if hour in range(6, 10):  # Morning rounds
            cpu_usage = 65 + np.random.normal(0, 10)
            memory_usage = 60 + np.random.normal(0, 8)
            request_rate = 120 + np.random.normal(0, 20)
        elif hour in range(14, 17):  # Afternoon rounds
            cpu_usage = 55 + np.random.normal(0, 10)
            memory_usage = 55 + np.random.normal(0, 8)
            request_rate = 90 + np.random.normal(0, 15)
        elif hour in range(0, 6) or hour in range(22, 24):  # Night hours
            cpu_usage = 25 + np.random.normal(0, 5)
            memory_usage = 35 + np.random.normal(0, 5)
            request_rate = 20 + np.random.normal(0, 5)
        else:  # Regular hours
            cpu_usage = 40 + np.random.normal(0, 8)
            memory_usage = 45 + np.random.normal(0, 7)
            request_rate = 60 + np.random.normal(0, 10)
        
        # Weekend adjustment
        if day_of_week in [5, 6]:  # Weekend
            cpu_usage *= 0.7
            memory_usage *= 0.7
            request_rate *= 0.6
        
        response_time = 1.0 + (cpu_usage / 100) * 2  # Simple response time model
        confidence = 0.6
        recommended_replicas = self._calculate_recommended_replicas(cpu_usage, memory_usage, request_rate)
        
        return PredictionResult(
            timestamp=datetime.now(),
            predicted_cpu=max(0, min(100, cpu_usage)),
            predicted_memory=max(0, min(100, memory_usage)),
            predicted_request_rate=max(0, request_rate),
            predicted_response_time=max(0.1, response_time),
            confidence_score=confidence,
            recommended_replicas=recommended_replicas,
            reasoning=f"Pattern-based prediction for {day_of_week} at {hour}:00"
        )
    
    def _store_prediction_in_db(self, target_time: datetime, result: PredictionResult):
        """Store prediction in database for future analysis"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (prediction_time, target_time, predicted_cpu, predicted_memory, 
             predicted_request_rate, predicted_response_time, confidence_score, 
             recommended_replicas, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.timestamp, target_time, result.predicted_cpu, result.predicted_memory,
            result.predicted_request_rate, result.predicted_response_time,
            result.confidence_score, result.recommended_replicas, result.reasoning
        ))
        
        conn.commit()
        conn.close()
    
    async def predict_workload_range(self, 
                                   start_time: datetime,
                                   hours_ahead: int = 24) -> List[PredictionResult]:
        """Predict workload for a range of future times"""
        predictions = []
        
        for i in range(hours_ahead):
            target_time = start_time + timedelta(hours=i)
            prediction = self.predict_workload(target_time)
            predictions.append(prediction)
        
        return predictions
    
    def analyze_workload_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze recent workload trends and patterns"""
        if len(self.workload_history) < 24:  # Need at least 24 hours of data
            return {"error": "Insufficient data for trend analysis"}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([asdict(sample) for sample in list(self.workload_history)[-24*days:]])
        
        trends = {
            'avg_cpu_usage': df['cpu_usage'].mean(),
            'avg_memory_usage': df['memory_usage'].mean(),
            'avg_request_rate': df['request_rate'].mean(),
            'peak_hours': df.groupby('hour_of_day')['cpu_usage'].mean().idxmax(),
            'low_hours': df.groupby('hour_of_day')['cpu_usage'].mean().idxmin(),
            'busiest_day': df.groupby('day_of_week')['request_rate'].mean().idxmax(),
            'trend_direction': self._calculate_trend(df['cpu_usage'].values),
            'anomaly_count': self._detect_anomalies(df)
        }
        
        return trends
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """Calculate trend direction for time series data"""
        if len(values) < 10:
            return "insufficient_data"
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 1:
            return "increasing"
        elif slope < -1:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> int:
        """Detect anomalous workload patterns"""
        cpu_mean = df['cpu_usage'].mean()
        cpu_std = df['cpu_usage'].std()
        
        anomalies = df[
            (df['cpu_usage'] > cpu_mean + threshold * cpu_std) |
            (df['cpu_usage'] < cpu_mean - threshold * cpu_std)
        ]
        
        return len(anomalies)


class WorkloadPredictionService:
    """
    Service interface for workload prediction system
    """
    
    def __init__(self):
        self.predictor = HealthcareWorkloadPredictor()
        self.predictor.load_models()  # Try to load existing models
    
    async def start_monitoring(self, monitoring_interval: int = 60):
        """Start continuous workload monitoring and prediction"""
        logger.info("Starting workload prediction service")
        
        while True:
            try:
                # Collect current workload metrics (placeholder implementation)
                current_metrics = await self._collect_current_metrics()
                
                # Add to training data
                self.predictor.add_workload_sample(current_metrics)
                
                # Generate predictions for next 24 hours
                predictions = await self.predictor.predict_workload_range(
                    datetime.now(), hours_ahead=24
                )
                
                # Update auto-scaling recommendations
                await self._update_scaling_recommendations(predictions)
                
                # Wait for next interval
                await asyncio.sleep(monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(monitoring_interval)
    
    async def _collect_current_metrics(self) -> WorkloadSample:
        """Collect current system metrics (placeholder)"""
        # In real implementation, this would collect from monitoring system
        now = datetime.now()
        
        # Simulated metrics based on time patterns
        hour = now.hour
        if 8 <= hour <= 11:
            cpu = 60 + np.random.normal(0, 10)
            memory = 55 + np.random.normal(0, 8)
            request_rate = 100 + np.random.normal(0, 15)
        elif hour <= 6 or hour >= 22:
            cpu = 30 + np.random.normal(0, 5)
            memory = 40 + np.random.normal(0, 5)
            request_rate = 25 + np.random.normal(0, 5)
        else:
            cpu = 45 + np.random.normal(0, 8)
            memory = 50 + np.random.normal(0, 7)
            request_rate = 70 + np.random.normal(0, 12)
        
        return WorkloadSample(
            timestamp=now,
            cpu_usage=max(0, min(100, cpu)),
            memory_usage=max(0, min(100, memory)),
            request_rate=max(0, request_rate),
            active_users=int(request_rate / 10),
            response_time=1.0 + (cpu / 100),
            endpoint_category="patient_api",
            day_of_week=now.weekday(),
            hour_of_day=hour,
            is_business_hours=7 <= hour <= 19,
            is_emergency_period=self.predictor._is_emergency_period(now)
        )
    
    async def _update_scaling_recommendations(self, predictions: List[PredictionResult]):
        """Update auto-scaling recommendations based on predictions"""
        # Implementation would integrate with Kubernetes HPA/VPA
        # For now, just log recommendations
        for i, prediction in enumerate(predictions[:3]):  # Log next 3 hours
            logger.info(f"Hour {i+1}: CPU {prediction.predicted_cpu:.1f}%, "
                       f"Memory {prediction.predicted_memory:.1f}%, "
                       f"Recommended replicas: {prediction.recommended_replicas}")


async def main():
    """Example usage of workload prediction system"""
    
    # Initialize prediction service
    service = WorkloadPredictionService()
    
    # Add some sample data
    for i in range(100):
        sample_time = datetime.now() - timedelta(hours=i)
        sample = WorkloadSample(
            timestamp=sample_time,
            cpu_usage=50 + np.random.normal(0, 15),
            memory_usage=45 + np.random.normal(0, 12),
            request_rate=80 + np.random.normal(0, 20),
            active_users=8 + np.random.normal(0, 2),
            response_time=1.2 + np.random.normal(0, 0.3),
            endpoint_category="patient_api",
            day_of_week=sample_time.weekday(),
            hour_of_day=sample_time.hour,
            is_business_hours=7 <= sample_time.hour <= 19,
            is_emergency_period=False
        )
        service.predictor.add_workload_sample(sample)
    
    # Predict future workload
    predictions = await service.predictor.predict_workload_range(
        datetime.now() + timedelta(hours=1), hours_ahead=8
    )
    
    print("Workload Predictions for next 8 hours:")
    for i, pred in enumerate(predictions):
        print(f"Hour {i+1}: CPU {pred.predicted_cpu:.1f}%, "
              f"Memory {pred.predicted_memory:.1f}%, "
              f"Replicas: {pred.recommended_replicas}, "
              f"Confidence: {pred.confidence_score:.2f}")
    
    # Analyze trends
    trends = service.predictor.analyze_workload_trends()
    print(f"\nWorkload Trends: {trends}")


if __name__ == "__main__":
    asyncio.run(main())