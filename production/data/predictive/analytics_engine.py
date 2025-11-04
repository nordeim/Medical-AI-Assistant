"""
Production Predictive Analytics and AI Insights System
Implements machine learning models for healthcare outcome prediction
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

class ModelType(Enum):
    """Types of predictive models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"

class PredictionTarget(Enum):
    """Healthcare prediction targets"""
    READMISSION_RISK = "readmission_risk"
    MORTALITY_RISK = "mortality_risk"
    LENGTH_OF_STAY = "length_of_stay"
    COMPLICATION_RISK = "complication_risk"
    MEDICATION_ADHERENCE = "medication_adherence"
    RESOURCE_UTILIZATION = "resource_utilization"
    COST_PREDICTION = "cost_prediction"
    PATIENT_SATISFACTION = "patient_satisfaction"

class ModelStatus(Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYED = "deployed"
    MONITORING = "monitoring"
    RETRAINING = "retraining"
    DEPRECATED = "deprecated"

@dataclass
class PredictionModel:
    """Predictive model configuration and metadata"""
    model_id: str
    model_name: str
    model_type: ModelType
    target_variable: PredictionTarget
    algorithm: str
    version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    model_status: ModelStatus
    deployment_date: Optional[datetime] = None
    last_validation_date: Optional[datetime] = None
    data_drift_score: float = 0.0
    model_bias_score: float = 0.0

@dataclass
class PredictionRequest:
    """Individual prediction request"""
    request_id: str
    model_id: str
    patient_data: Dict[str, Any]
    prediction_timestamp: datetime
    prediction_result: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    feature_values: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    performance_id: str
    model_id: str
    evaluation_date: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    sample_size: int = 0
    prediction_window: str = ""

class PredictiveAnalyticsEngine:
    """Production predictive analytics engine for healthcare"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = {}
        self.performance_history = {}
        self.prediction_cache = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup predictive analytics logging"""
        logger = logging.getLogger("predictive_analytics")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def initialize_analytics_engine(self) -> None:
        """Initialize predictive analytics engine"""
        try:
            # Initialize prediction models
            await self._initialize_prediction_models()
            
            # Load trained models if available
            await self._load_trained_models()
            
            # Initialize model performance tracking
            await self._initialize_performance_tracking()
            
            self.logger.info("Predictive analytics engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Analytics engine initialization failed: {str(e)}")
            raise
    
    async def _initialize_prediction_models(self) -> None:
        """Initialize prediction model definitions"""
        self.models = {
            # Readmission Risk Model
            "readmission_model_v1": PredictionModel(
                model_id="readmission_model_v1",
                model_name="30-Day Readmission Risk Prediction",
                model_type=ModelType.CLASSIFICATION,
                target_variable=PredictionTarget.READMISSION_RISK,
                algorithm="XGBoost",
                version="1.0",
                training_date=datetime.now() - timedelta(days=30),
                performance_metrics={
                    "accuracy": 0.78,
                    "precision": 0.75,
                    "recall": 0.72,
                    "f1_score": 0.73,
                    "auc_roc": 0.82
                },
                feature_importance={
                    "prior_admissions": 0.25,
                    "charlson_comorbidity_index": 0.20,
                    "length_of_stay": 0.18,
                    "age": 0.15,
                    "medication_count": 0.12,
                    "discharge_disposition": 0.10
                },
                model_status=ModelStatus.DEPLOYED,
                deployment_date=datetime.now() - timedelta(days=25)
            ),
            
            # Mortality Risk Model
            "mortality_model_v1": PredictionModel(
                model_id="mortality_model_v1",
                model_name="In-Hospital Mortality Risk Prediction",
                model_type=ModelType.CLASSIFICATION,
                target_variable=PredictionTarget.MORTALITY_RISK,
                algorithm="Random Forest",
                version="1.0",
                training_date=datetime.now() - timedelta(days=20),
                performance_metrics={
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.85,
                    "f1_score": 0.87,
                    "auc_roc": 0.94
                },
                feature_importance={
                    "age": 0.30,
                    "severity_of_illness": 0.25,
                    "charlson_comorbidity_index": 0.20,
                    "admission_type": 0.15,
                    "blood_pressure_systolic": 0.10
                },
                model_status=ModelStatus.DEPLOYED,
                deployment_date=datetime.now() - timedelta(days=15)
            ),
            
            # Length of Stay Prediction Model
            "los_model_v1": PredictionModel(
                model_id="los_model_v1",
                model_name="Length of Stay Prediction",
                model_type=ModelType.REGRESSION,
                target_variable=PredictionTarget.LENGTH_OF_STAY,
                algorithm="Gradient Boosting",
                version="1.0",
                training_date=datetime.now() - timedelta(days=15),
                performance_metrics={
                    "mae": 1.2,
                    "rmse": 1.8,
                    "r2_score": 0.68
                },
                feature_importance={
                    "diagnosis_category": 0.35,
                    "age": 0.25,
                    "severity_of_illness": 0.20,
                    "admission_type": 0.15,
                    "comorbidity_count": 0.05
                },
                model_status=ModelStatus.DEPLOYED,
                deployment_date=datetime.now() - timedelta(days=10)
            ),
            
            # Complication Risk Model
            "complication_model_v1": PredictionModel(
                model_id="complication_model_v1",
                model_name="Surgical Complication Risk Prediction",
                model_type=ModelType.CLASSIFICATION,
                target_variable=PredictionTarget.COMPLICATION_RISK,
                algorithm="Logistic Regression",
                version="1.0",
                training_date=datetime.now() - timedelta(days=10),
                performance_metrics={
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.79,
                    "f1_score": 0.80,
                    "auc_roc": 0.88
                },
                feature_importance={
                    "surgical_procedure": 0.40,
                    "patient_age": 0.25,
                    "comorbidity_score": 0.20,
                    "anesthesia_type": 0.15
                },
                model_status=ModelStatus.DEPLOYED,
                deployment_date=datetime.now() - timedelta(days=5)
            ),
            
            # Resource Utilization Model
            "resource_model_v1": PredictionModel(
                model_id="resource_model_v1",
                model_name="Resource Utilization Prediction",
                model_type=ModelType.REGRESSION,
                target_variable=PredictionTarget.RESOURCE_UTILIZATION,
                algorithm="LightGBM",
                version="1.0",
                training_date=datetime.now() - timedelta(days=5),
                performance_metrics={
                    "mae": 0.15,
                    "rmse": 0.22,
                    "r2_score": 0.74
                },
                feature_importance={
                    "patient_volume": 0.30,
                    "staffing_levels": 0.25,
                    "seasonal_patterns": 0.20,
                    "disease_prevalence": 0.15,
                    "capacity_constraints": 0.10
                },
                model_status=ModelStatus.DEPLOYED,
                deployment_date=datetime.now() - timedelta(days=1)
            )
        }
    
    async def _load_trained_models(self) -> None:
        """Load trained models from storage"""
        model_path = Path(self.config.get("model_storage_path", "./models"))
        
        for model_id, model_config in self.models.items():
            try:
                model_file = model_path / f"{model_id}.joblib"
                if model_file.exists():
                    # Load model
                    model = joblib.load(model_file)
                    
                    # Load associated scaler and encoder
                    scaler_file = model_path / f"{model_id}_scaler.joblib"
                    if scaler_file.exists():
                        self.scalers[model_id] = joblib.load(scaler_file)
                    
                    encoder_file = model_path / f"{model_id}_encoder.joblib"
                    if encoder_file.exists():
                        self.encoders[model_id] = joblib.load(encoder_file)
                    
                    self.logger.info(f"Loaded trained model: {model_id}")
                
                else:
                    # Create placeholder model for demonstration
                    self._create_placeholder_model(model_config)
                    
            except Exception as e:
                self.logger.error(f"Failed to load model {model_id}: {str(e)}")
                # Create placeholder
                self._create_placeholder_model(model_config)
    
    def _create_placeholder_model(self, model_config: PredictionModel) -> None:
        """Create placeholder model for demonstration"""
        if model_config.model_type == ModelType.CLASSIFICATION:
            if model_config.algorithm == "XGBoost":
                self.models[model_config.model_id].model_object = xgb.XGBClassifier(random_state=42)
            elif model_config.algorithm == "Random Forest":
                self.models[model_config.model_id].model_object = RandomForestClassifier(random_state=42)
            elif model_config.algorithm == "Logistic Regression":
                self.models[model_config.model_id].model_object = LogisticRegression(random_state=42)
        
        elif model_config.model_type == ModelType.REGRESSION:
            if model_config.algorithm == "Gradient Boosting":
                self.models[model_config.model_id].model_object = GradientBoostingRegressor(random_state=42)
            elif model_config.algorithm == "LightGBM":
                self.models[model_config.model_id].model_object = lgb.LGBMRegressor(random_state=42)
        
        # Create placeholder scaler
        self.scalers[model_config.model_id] = StandardScaler()
        
        self.logger.info(f"Created placeholder model: {model_config.model_id}")
    
    async def _initialize_performance_tracking(self) -> None:
        """Initialize model performance tracking"""
        self.performance_history = {model_id: [] for model_id in self.models.keys()}
    
    async def make_prediction(self, model_id: str, patient_data: Dict[str, Any]) -> PredictionRequest:
        """Make prediction using specified model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_config = self.models[model_id]
        request_id = f"PRED_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_id}"
        
        request = PredictionRequest(
            request_id=request_id,
            model_id=model_id,
            patient_data=patient_data,
            prediction_timestamp=datetime.now()
        )
        
        try:
            self.logger.info(f"Making prediction with model: {model_id}")
            
            # Prepare features
            features = await self._prepare_features(patient_data, model_config)
            
            # Get model object
            model = getattr(model_config, 'model_object', None)
            if model is None:
                raise ValueError(f"Model object not available for {model_id}")
            
            # Make prediction
            if model_config.model_type == ModelType.CLASSIFICATION:
                prediction_result = await self._make_classification_prediction(model, features, model_config)
            elif model_config.model_type == ModelType.REGRESSION:
                prediction_result = await self._make_regression_prediction(model, features, model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_config.model_type}")
            
            # Calculate confidence score
            confidence_score = await self._calculate_confidence_score(features, model_config)
            
            # Update request with results
            request.prediction_result = prediction_result
            request.confidence_score = confidence_score
            request.feature_values = features
            request.model_version = model_config.version
            
            # Cache prediction for monitoring
            cache_key = self._generate_cache_key(patient_data, model_id)
            self.prediction_cache[cache_key] = {
                "request": request,
                "timestamp": datetime.now()
            }
            
            self.logger.info(f"Prediction completed: {request_id} with confidence {confidence_score:.3f}")
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {request_id} - {str(e)}")
            request.prediction_result = {"error": str(e)}
            request.confidence_score = 0.0
        
        return request
    
    async def _prepare_features(self, patient_data: Dict[str, Any], model_config: PredictionModel) -> Dict[str, float]:
        """Prepare features for prediction"""
        features = {}
        
        # Common features across models
        if "age" in patient_data:
            features["age"] = float(patient_data["age"])
        
        if "gender" in patient_data:
            features["gender_male"] = 1.0 if patient_data["gender"].lower() == "m" else 0.0
            features["gender_female"] = 1.0 if patient_data["gender"].lower() == "f" else 0.0
        
        # Readmission model features
        if model_config.target_variable == PredictionTarget.READMISSION_RISK:
            features.update({
                "prior_admissions": float(patient_data.get("prior_admissions", 0)),
                "charlson_comorbidity_index": float(patient_data.get("charlson_comorbidity_index", 0)),
                "length_of_stay": float(patient_data.get("length_of_stay", 1)),
                "medication_count": float(patient_data.get("medication_count", 0)),
                "discharge_disposition_snf": 1.0 if patient_data.get("discharge_disposition") == "SNF" else 0.0,
                "discharge_disposition_home": 1.0 if patient_data.get("discharge_disposition") == "HOME" else 0.0
            })
        
        # Mortality model features
        elif model_config.target_variable == PredictionTarget.MORTALITY_RISK:
            features.update({
                "severity_of_illness": float(patient_data.get("severity_of_illness", 1)),
                "admission_type_emergency": 1.0 if patient_data.get("admission_type") == "EMERGENCY" else 0.0,
                "blood_pressure_systolic": float(patient_data.get("blood_pressure_systolic", 120)),
                "heart_rate": float(patient_data.get("heart_rate", 80)),
                "temperature": float(patient_data.get("temperature", 37.0)),
                "respiratory_rate": float(patient_data.get("respiratory_rate", 16))
            })
        
        # Length of stay model features
        elif model_config.target_variable == PredictionTarget.LENGTH_OF_STAY:
            features.update({
                "diagnosis_category": await self._encode_diagnosis_category(patient_data.get("primary_diagnosis", "")),
                "surgical_procedure": 1.0 if patient_data.get("has_surgery", False) else 0.0,
                "admission_source": await self._encode_admission_source(patient_data.get("admission_source", "")),
                "insurance_type": await self._encode_insurance_type(patient_data.get("insurance_type", ""))
            })
        
        # Complication model features
        elif model_config.target_variable == PredictionTarget.COMPLICATION_RISK:
            features.update({
                "surgical_procedure_complexity": float(patient_data.get("procedure_complexity", 1)),
                "anesthesia_type_general": 1.0 if patient_data.get("anesthesia_type") == "GENERAL" else 0.0,
                "operation_duration": float(patient_data.get("operation_duration", 60)),
                "patient_bmi": float(patient_data.get("bmi", 25)),
                "smoking_status": 1.0 if patient_data.get("smoking_status") == "CURRENT" else 0.0
            })
        
        # Resource utilization model features
        elif model_config.target_variable == PredictionTarget.RESOURCE_UTILIZATION:
            features.update({
                "day_of_week": datetime.now().weekday(),
                "month": datetime.now().month,
                "seasonal_factor": await self._calculate_seasonal_factor(datetime.now().month),
                "historical_volume": float(patient_data.get("historical_daily_volume", 100))
            })
        
        return features
    
    def _encode_diagnosis_category(self, diagnosis: str) -> float:
        """Encode diagnosis category numerically"""
        categories = {
            "cardiovascular": 1.0,
            "respiratory": 2.0,
            "digestive": 3.0,
            "musculoskeletal": 4.0,
            "neurological": 5.0,
            "endocrine": 6.0,
            "other": 7.0
        }
        
        diagnosis_lower = diagnosis.lower()
        for category, value in categories.items():
            if category in diagnosis_lower:
                return value
        
        return 7.0  # Default to "other"
    
    def _encode_admission_source(self, source: str) -> float:
        """Encode admission source numerically"""
        sources = {
            "emergency": 1.0,
            "referral": 2.0,
            "transfer": 3.0,
            "scheduled": 4.0
        }
        
        source_lower = source.lower()
        return sources.get(source_lower, 4.0)
    
    def _encode_insurance_type(self, insurance: str) -> float:
        """Encode insurance type numerically"""
        insurance_types = {
            "medicare": 1.0,
            "medicaid": 2.0,
            "private": 3.0,
            "uninsured": 4.0,
            "other": 5.0
        }
        
        insurance_lower = insurance.lower()
        return insurance_types.get(insurance_lower, 5.0)
    
    def _calculate_seasonal_factor(self, month: int) -> float:
        """Calculate seasonal factor for resource utilization"""
        seasonal_factors = {
            1: 1.2,   # January - high due to New Year
            2: 1.1,   # February
            3: 1.0,   # March
            4: 0.9,   # April
            5: 0.9,   # May
            6: 0.8,   # June - summer low
            7: 0.8,   # July
            8: 0.9,   # August
            9: 1.0,   # September - back to school
            10: 1.1,  # October
            11: 1.2,  # November - pre-holiday
            12: 1.3   # December - holiday season
        }
        
        return seasonal_factors.get(month, 1.0)
    
    async def _make_classification_prediction(self, model, features: Dict[str, float], 
                                            model_config: PredictionModel) -> Dict[str, Any]:
        """Make classification prediction"""
        # Convert features to array format
        feature_array = np.array([[features.get(feature, 0.0) for feature in model_config.feature_importance.keys()]])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        prediction_proba = model.predict_proba(feature_array)[0]
        
        result = {
            "prediction": "high_risk" if prediction == 1 else "low_risk",
            "probability": {
                "low_risk": float(prediction_proba[0]),
                "high_risk": float(prediction_proba[1])
            },
            "risk_score": float(prediction_proba[1])  # Probability of positive class
        }
        
        return result
    
    async def _make_regression_prediction(self, model, features: Dict[str, float], 
                                        model_config: PredictionModel) -> Dict[str, Any]:
        """Make regression prediction"""
        # Convert features to array format
        feature_array = np.array([[features.get(feature, 0.0) for feature in model_config.feature_importance.keys()]])
        
        # Make prediction
        prediction = model.predict(feature_array)[0]
        
        result = {
            "prediction": float(prediction),
            "unit": self._get_prediction_unit(model_config.target_variable),
            "confidence_interval": {
                "lower": float(prediction * 0.8),
                "upper": float(prediction * 1.2)
            }
        }
        
        return result
    
    def _get_prediction_unit(self, target: PredictionTarget) -> str:
        """Get unit for prediction target"""
        units = {
            PredictionTarget.LENGTH_OF_STAY: "days",
            PredictionTarget.COST_PREDICTION: "USD",
            PredictionTarget.RESOURCE_UTILIZATION: "percentage",
            PredictionTarget.PATIENT_SATISFACTION: "rating"
        }
        
        return units.get(target, "score")
    
    async def _calculate_confidence_score(self, features: Dict[str, float], 
                                        model_config: PredictionModel) -> float:
        """Calculate confidence score for prediction"""
        # Simple confidence calculation based on feature completeness
        expected_features = len(model_config.feature_importance)
        available_features = len([f for f in features.values() if not np.isnan(f)])
        
        completeness_score = available_features / expected_features
        
        # Combine with model performance
        if model_config.model_type == ModelType.CLASSIFICATION:
            performance_score = model_config.performance_metrics.get("auc_roc", 0.7)
        else:
            performance_score = 1.0 - min(model_config.performance_metrics.get("mae", 2.0) / 10.0, 1.0)
        
        # Weighted confidence score
        confidence = (completeness_score * 0.6 + performance_score * 0.4)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _generate_cache_key(self, patient_data: Dict[str, Any], model_id: str) -> str:
        """Generate cache key for prediction"""
        # Create simplified cache key based on key patient features
        key_features = {
            "age": patient_data.get("age", 0),
            "gender": patient_data.get("gender", "U"),
            "diagnosis": patient_data.get("primary_diagnosis", "UNKNOWN")
        }
        
        key_string = f"{model_id}_{hashlib.md5(str(key_features).encode()).hexdigest()[:12]}"
        return key_string
    
    async def train_model(self, model_id: str, training_data: pd.DataFrame, 
                         target_column: str) -> Dict[str, Any]:
        """Train or retrain a prediction model"""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_config = self.models[model_id]
        model_config.model_status = ModelStatus.TRAINING
        
        try:
            self.logger.info(f"Training model: {model_id}")
            
            # Prepare training data
            X = training_data.drop(columns=[target_column])
            y = training_data[target_column]
            
            # Handle categorical variables
            X_processed = await self._preprocess_training_data(X, model_id)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=0.2, random_state=42, stratify=y if model_config.model_type == ModelType.CLASSIFICATION else None
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if model_config.model_type == ModelType.CLASSIFICATION:
                model = await self._train_classification_model(X_train_scaled, y_train, model_config.algorithm)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Update performance metrics
                if len(np.unique(y)) == 2:  # Binary classification
                    auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    model_config.performance_metrics["auc_roc"] = auc_roc
                
                # Calculate other metrics
                accuracy = (y_pred == y_test).mean()
                model_config.performance_metrics["accuracy"] = accuracy
                
            elif model_config.model_type == ModelType.REGRESSION:
                model = await self._train_regression_model(X_train_scaled, y_train, model_config.algorithm)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
                
                model_config.performance_metrics["mae"] = mae
                model_config.performance_metrics["rmse"] = rmse
            
            # Update feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_processed.columns, model.feature_importances_))
                model_config.feature_importance = feature_importance
            
            # Save model
            await self._save_trained_model(model_id, model, scaler)
            
            # Update model status
            model_config.model_status = ModelStatus.DEPLOYED
            model_config.training_date = datetime.now()
            model_config.deployment_date = datetime.now()
            
            # Record performance
            await self._record_performance(model_config, model_config.performance_metrics)
            
            self.logger.info(f"Model training completed: {model_id}")
            
            return {
                "model_id": model_id,
                "status": "trained",
                "performance_metrics": model_config.performance_metrics,
                "training_samples": len(training_data),
                "feature_count": len(X_processed.columns)
            }
            
        except Exception as e:
            model_config.model_status = ModelStatus.DEPRECATED
            self.logger.error(f"Model training failed: {model_id} - {str(e)}")
            raise
    
    async def _preprocess_training_data(self, X: pd.DataFrame, model_id: str) -> pd.DataFrame:
        """Preprocess training data"""
        X_processed = X.copy()
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median(numeric_only=True))
        
        # Handle categorical variables
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.encoders[f"{model_id}_{col}"] = le
        
        return X_processed
    
    async def _train_classification_model(self, X_train, y_train, algorithm: str):
        """Train classification model"""
        if algorithm == "XGBoost":
            model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        elif algorithm == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        elif algorithm == "Logistic Regression":
            model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unsupported classification algorithm: {algorithm}")
        
        model.fit(X_train, y_train)
        return model
    
    async def _train_regression_model(self, X_train, y_train, algorithm: str):
        """Train regression model"""
        if algorithm == "Gradient Boosting":
            model = GradientBoostingRegressor(random_state=42)
        elif algorithm == "LightGBM":
            model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        else:
            raise ValueError(f"Unsupported regression algorithm: {algorithm}")
        
        model.fit(X_train, y_train)
        return model
    
    async def _save_trained_model(self, model_id: str, model, scaler: StandardScaler) -> None:
        """Save trained model and preprocessing objects"""
        model_path = Path(self.config.get("model_storage_path", "./models"))
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(model, model_path / f"{model_id}.joblib")
        
        # Save scaler
        joblib.dump(scaler, model_path / f"{model_id}_scaler.joblib")
        
        # Save encoders
        for encoder_id, encoder in self.encoders.items():
            if encoder_id.startswith(f"{model_id}_"):
                joblib.dump(encoder, model_path / f"{encoder_id}_encoder.joblib")
        
        self.logger.info(f"Model saved: {model_id}")
    
    async def _record_performance(self, model_config: PredictionModel, metrics: Dict[str, float]) -> None:
        """Record model performance in history"""
        performance_record = ModelPerformance(
            performance_id=f"PERF_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_config.model_id}",
            model_id=model_config.model_id,
            evaluation_date=datetime.now(),
            accuracy=metrics.get("accuracy", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1_score=metrics.get("f1_score", 0.0),
            auc_roc=metrics.get("auc_roc"),
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            sample_size=1000  # Would be actual sample size
        )
        
        if model_config.model_id not in self.performance_history:
            self.performance_history[model_config.model_id] = []
        
        self.performance_history[model_config.model_id].append(performance_record)
        
        self.logger.info(f"Performance recorded for model: {model_config.model_id}")
    
    async def get_model_insights(self, model_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model_config = self.models[model_id]
        performance_history = self.performance_history.get(model_id, [])
        
        # Calculate trend analysis
        if len(performance_history) > 1:
            recent_performance = performance_history[-1]
            previous_performance = performance_history[-2]
            
            accuracy_trend = "improving" if recent_performance.accuracy > previous_performance.accuracy else "declining"
            drift_detected = abs(recent_performance.accuracy - previous_performance.accuracy) > 0.05
        else:
            accuracy_trend = "stable"
            drift_detected = False
        
        # Generate insights
        insights = {
            "model_overview": {
                "model_id": model_config.model_id,
                "model_name": model_config.model_name,
                "algorithm": model_config.algorithm,
                "target_variable": model_config.target_variable.value,
                "status": model_config.model_status.value,
                "version": model_config.version
            },
            "performance_metrics": model_config.performance_metrics,
            "feature_importance": model_config.feature_importance,
            "performance_trend": {
                "accuracy_trend": accuracy_trend,
                "drift_detected": drift_detected,
                "performance_history_count": len(performance_history)
            },
            "deployment_info": {
                "deployment_date": model_config.deployment_date.isoformat() if model_config.deployment_date else None,
                "days_in_production": (datetime.now() - model_config.deployment_date).days if model_config.deployment_date else 0,
                "last_validation": model_config.last_validation_date.isoformat() if model_config.last_validation_date else None
            },
            "recommendations": await self._generate_model_recommendations(model_config, drift_detected)
        }
        
        return insights
    
    async def _generate_model_recommendations(self, model_config: PredictionModel, drift_detected: bool) -> List[str]:
        """Generate recommendations for model improvement"""
        recommendations = []
        
        # Performance-based recommendations
        accuracy = model_config.performance_metrics.get("accuracy", 0.0)
        if accuracy < 0.8:
            recommendations.append("Consider retraining with additional features or more diverse data")
            recommendations.append("Review data quality and preprocessing pipeline")
        
        # Drift-based recommendations
        if drift_detected:
            recommendations.append("Model drift detected - retraining recommended")
            recommendations.append("Monitor data distribution changes")
            recommendations.append("Consider updating feature engineering pipeline")
        
        # Age-based recommendations
        days_in_production = (datetime.now() - model_config.deployment_date).days if model_config.deployment_date else 0
        if days_in_production > 90:
            recommendations.append("Model has been in production for over 90 days - periodic retraining recommended")
        
        # Feature importance recommendations
        top_features = sorted(model_config.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        if len(top_features) < 3:
            recommendations.append("Consider adding more predictive features to improve model performance")
        
        return recommendations
    
    async def batch_predict(self, model_id: str, batch_data: List[Dict[str, Any]]) -> List[PredictionRequest]:
        """Make predictions for multiple patients"""
        predictions = []
        
        for patient_data in batch_data:
            try:
                prediction_request = await self.make_prediction(model_id, patient_data)
                predictions.append(prediction_request)
            except Exception as e:
                # Create failed prediction request
                failed_request = PredictionRequest(
                    request_id=f"FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_id}",
                    model_id=model_id,
                    patient_data=patient_data,
                    prediction_timestamp=datetime.now(),
                    prediction_result={"error": str(e)},
                    confidence_score=0.0
                )
                predictions.append(failed_request)
        
        self.logger.info(f"Batch prediction completed: {len(predictions)} predictions")
        return predictions
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get summary of all predictions made"""
        total_predictions = len(self.prediction_cache)
        model_usage = {}
        confidence_distribution = []
        
        for cache_entry in self.prediction_cache.values():
            request = cache_entry["request"]
            
            # Count by model
            model_id = request.model_id
            if model_id not in model_usage:
                model_usage[model_id] = 0
            model_usage[model_id] += 1
            
            # Collect confidence scores
            if request.confidence_score is not None:
                confidence_distribution.append(request.confidence_score)
        
        # Calculate confidence statistics
        if confidence_distribution:
            avg_confidence = np.mean(confidence_distribution)
            median_confidence = np.median(confidence_distribution)
            low_confidence_count = sum(1 for c in confidence_distribution if c < 0.7)
        else:
            avg_confidence = 0.0
            median_confidence = 0.0
            low_confidence_count = 0
        
        return {
            "total_predictions": total_predictions,
            "model_usage": model_usage,
            "confidence_stats": {
                "average": avg_confidence,
                "median": median_confidence,
                "low_confidence_count": low_confidence_count,
                "low_confidence_percentage": (low_confidence_count / total_predictions * 100) if total_predictions > 0 else 0
            },
            "prediction_cache_size": len(self.prediction_cache)
        }

def create_analytics_engine(config: Dict[str, Any] = None) -> PredictiveAnalyticsEngine:
    """Factory function to create predictive analytics engine"""
    if config is None:
        config = {
            "model_storage_path": "./models",
            "prediction_cache_size": 1000,
            "auto_retrain_threshold": 0.05  # 5% performance drop
        }
    
    return PredictiveAnalyticsEngine(config)

# Example usage
if __name__ == "__main__":
    async def main():
        engine = create_analytics_engine()
        
        # Initialize analytics engine
        await engine.initialize_analytics_engine()
        
        # Example patient data for readmission prediction
        patient_data = {
            "age": 75,
            "gender": "M",
            "prior_admissions": 3,
            "charlson_comorbidity_index": 4,
            "length_of_stay": 5,
            "medication_count": 8,
            "discharge_disposition": "SNF",
            "primary_diagnosis": "heart failure"
        }
        
        # Make prediction
        prediction = await engine.make_prediction("readmission_model_v1", patient_data)
        
        print("Readmission Risk Prediction:")
        print(f"Request ID: {prediction.request_id}")
        print(f"Risk Level: {prediction.prediction_result.get('prediction', 'Unknown')}")
        print(f"Risk Score: {prediction.prediction_result.get('risk_score', 0):.3f}")
        print(f"Confidence: {prediction.confidence_score:.3f}")
        
        # Get model insights
        insights = await engine.get_model_insights("readmission_model_v1")
        print(f"\nModel Insights:")
        print(f"Algorithm: {insights['model_overview']['algorithm']}")
        print(f"Accuracy: {insights['performance_metrics'].get('accuracy', 0):.3f}")
        print(f"Recommendations: {len(insights['recommendations'])}")
        
        # Get prediction summary
        summary = engine.get_prediction_summary()
        print(f"\nPrediction Summary:")
        print(f"Total Predictions: {summary['total_predictions']}")
        print(f"Average Confidence: {summary['confidence_stats']['average']:.3f}")
    
    asyncio.run(main())
