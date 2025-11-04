"""
Churn Prediction Engine
Predicts customer churn using advanced ML models and analytics
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import sqlite3
from pathlib import Path


class ChurnPredictionEngine:
    """
    AI-powered churn prediction engine with real-time scoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # ML Models
        self.churn_model = None
        self.feature_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Data storage
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.customer_data_cache = {}
        
        # Prediction configuration
        self.prediction_config = self.load_prediction_config()
        self.feature_columns = self.define_feature_columns()
        
        # Model performance metrics
        self.model_metrics = {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "auc_score": 0,
            "last_trained": None
        }
        
        self.initialize_models()
        self.logger.info("Churn Prediction Engine initialized")
    
    def load_prediction_config(self) -> Dict[str, Any]:
        """Load churn prediction configuration"""
        return {
            "model_settings": {
                "algorithm": "ensemble",  # ensemble, random_forest, gradient_boosting, logistic
                "ensemble_weights": {
                    "random_forest": 0.4,
                    "gradient_boosting": 0.3,
                    "logistic_regression": 0.3
                },
                "confidence_threshold": 0.7,
                "early_warning_threshold": 0.5,
                "critical_threshold": 0.8
            },
            "feature_engineering": {
                "lookback_period_days": 90,
                "rolling_windows": [7, 14, 30],
                "lag_features": [1, 3, 7, 14, 30],
                "seasonal_features": True,
                "interaction_features": True
            },
            "prediction_intervals": {
                "daily_update": True,
                "real_time_scoring": True,
                "batch_prediction_frequency": "hourly",
                "individual_prediction_frequency": "realtime"
            },
            "risk_categories": {
                "high_risk": {"min_score": 0.8, "action": "immediate_intervention"},
                "medium_risk": {"min_score": 0.5, "max_score": 0.8, "action": "proactive_outreach"},
                "low_risk": {"min_score": 0.2, "max_score": 0.5, "action": "monitoring"},
                "minimal_risk": {"min_score": 0.0, "max_score": 0.2, "action": "standard_engagement"}
            }
        }
    
    def define_feature_columns(self) -> List[str]:
        """Define feature columns for churn prediction"""
        return [
            # Usage patterns
            "total_login_days",
            "avg_session_duration", 
            "feature_usage_count",
            "feature_adoption_rate",
            "last_login_days",
            "session_frequency",
            
            # Engagement metrics
            "engagement_score",
            "email_open_rate",
            "email_click_rate",
            "support_tickets_30d",
            "help_docs_viewed",
            "community_participation",
            
            # Financial metrics
            "monthly_value",
            "payment_delays",
            "contract_renewal_date",
            "discount_received",
            "upgrade_frequency",
            
            # Customer lifecycle
            "tenure_months",
            "onboarding_completion",
            "first_value_time",
            "milestone_achievements",
            
            # Behavioral indicators
            "competitor_research",
            "feature_requests",
            "complaint_frequency",
            "satisfaction_score",
            
            # Industry/market factors
            "industry_churn_rate",
            "seasonal_trends",
            "economic_indicators",
            
            # Derived features
            "usage_trend_slope",
            "engagement_momentum",
            "value_stability",
            "health_score",
            "support_ratio"
        ]
    
    def initialize_models(self):
        """Initialize machine learning models for churn prediction"""
        try:
            # Create ensemble model
            self.churn_model = {
                "random_forest": RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                "gradient_boosting": GradientBoostingClassifier(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=8,
                    random_state=42
                ),
                "logistic_regression": LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                )
            }
            
            # Initialize label encoders for categorical features
            categorical_features = ['industry', 'subscription_tier', 'company_size']
            for feature in categorical_features:
                self.label_encoders[feature] = LabelEncoder()
            
            self.logger.info("Churn prediction models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def prepare_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """Prepare customer data for prediction"""
        try:
            # In real implementation, this would fetch from database
            # For demo, generate synthetic data
            
            customer_data = self.generate_synthetic_customer_data(customer_id)
            
            # Extract features for prediction
            features = self.extract_prediction_features(customer_data)
            
            return {
                "customer_id": customer_id,
                "features": features,
                "raw_data": customer_data,
                "prediction_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing customer data: {str(e)}")
            return {}
    
    def generate_synthetic_customer_data(self, customer_id: str) -> Dict[str, Any]:
        """Generate synthetic customer data for demonstration"""
        import random
        
        # Set seed based on customer_id for consistent data
        random.seed(hash(customer_id) % 2**32)
        
        # Generate realistic customer data
        return {
            "customer_id": customer_id,
            "industry": random.choice(["technology", "healthcare", "finance", "retail", "manufacturing"]),
            "company_size": random.choice(["small", "medium", "large", "enterprise"]),
            "subscription_tier": random.choice(["basic", "professional", "enterprise"]),
            "tenure_months": random.randint(1, 36),
            "monthly_value": random.randint(100, 5000),
            
            # Usage metrics
            "total_login_days": random.randint(10, 200),
            "avg_session_duration": random.randint(10, 180),
            "feature_usage_count": random.randint(5, 100),
            "feature_adoption_rate": random.uniform(0.1, 0.9),
            "last_login_days": random.randint(1, 30),
            "session_frequency": random.uniform(0.5, 3.0),
            
            # Engagement metrics
            "engagement_score": random.randint(30, 95),
            "email_open_rate": random.uniform(0.2, 0.8),
            "email_click_rate": random.uniform(0.05, 0.4),
            "support_tickets_30d": random.randint(0, 10),
            "help_docs_viewed": random.randint(0, 50),
            "community_participation": random.randint(0, 20),
            
            # Health and satisfaction
            "health_score": random.randint(40, 95),
            "satisfaction_score": random.uniform(3.0, 5.0),
            "nps_score": random.randint(0, 10),
            
            # Support and issues
            "payment_delays": random.randint(0, 3),
            "complaint_frequency": random.randint(0, 5),
            "feature_requests": random.randint(0, 10),
            
            # Contract and renewal
            "contract_renewal_date": (datetime.now() + timedelta(days=random.randint(30, 365))).isoformat(),
            "upgrade_frequency": random.randint(0, 5),
            "discount_received": random.choice([True, False])
        }
    
    def extract_prediction_features(self, customer_data: Dict[str, Any]) -> np.ndarray:
        """Extract and engineer features for churn prediction"""
        try:
            features = []
            
            # Usage pattern features
            features.extend([
                customer_data.get("total_login_days", 0),
                customer_data.get("avg_session_duration", 0),
                customer_data.get("feature_usage_count", 0),
                customer_data.get("feature_adoption_rate", 0),
                customer_data.get("last_login_days", 30),
                customer_data.get("session_frequency", 1.0)
            ])
            
            # Engagement metrics
            features.extend([
                customer_data.get("engagement_score", 50),
                customer_data.get("email_open_rate", 0.5),
                customer_data.get("email_click_rate", 0.1),
                customer_data.get("support_tickets_30d", 0),
                customer_data.get("help_docs_viewed", 0),
                customer_data.get("community_participation", 0)
            ])
            
            # Financial metrics
            features.extend([
                customer_data.get("monthly_value", 0),
                customer_data.get("payment_delays", 0),
                self.days_until_renewal(customer_data.get("contract_renewal_date")),
                1 if customer_data.get("discount_received") else 0,
                customer_data.get("upgrade_frequency", 0)
            ])
            
            # Lifecycle features
            features.extend([
                customer_data.get("tenure_months", 0),
                1 if self.assess_onboarding_completion(customer_data) else 0,
                self.calculate_first_value_time(customer_data),
                customer_data.get("milestone_achievements", 0)
            ])
            
            # Behavioral indicators (inverse sentiment indicators)
            features.extend([
                customer_data.get("feature_requests", 0) * -1,  # Negative correlation
                customer_data.get("complaint_frequency", 0) * -1,  # Negative correlation
                customer_data.get("satisfaction_score", 3.0)
            ])
            
            # Derived features
            usage_trend = self.calculate_usage_trend(customer_data)
            engagement_momentum = self.calculate_engagement_momentum(customer_data)
            value_stability = self.calculate_value_stability(customer_data)
            
            features.extend([
                usage_trend,
                engagement_momentum,
                value_stability,
                customer_data.get("health_score", 50),
                self.calculate_support_ratio(customer_data)
            ])
            
            # Industry-specific features (simplified)
            industry_churn_rates = {
                "technology": 0.15, "healthcare": 0.10, "finance": 0.12,
                "retail": 0.18, "manufacturing": 0.14, "other": 0.15
            }
            industry = customer_data.get("industry", "other")
            features.append(industry_churn_rates.get(industry, 0.15))
            
            # Convert to numpy array and handle any missing values
            features = np.array(features, dtype=float)
            features = np.nan_to_num(features, nan=0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return np.zeros(len(self.feature_columns))
    
    def days_until_renewal(self, renewal_date_str: str) -> float:
        """Calculate days until contract renewal"""
        try:
            renewal_date = datetime.fromisoformat(renewal_date_str.replace('Z', '+00:00'))
            days_until = (renewal_date - datetime.now()).days
            return max(0, days_until)
        except:
            return 365  # Default to 1 year
    
    def assess_onboarding_completion(self, customer_data: Dict[str, Any]) -> bool:
        """Assess if onboarding was completed successfully"""
        # Simplified assessment based on usage patterns
        return (customer_data.get("total_login_days", 0) > 10 and 
                customer_data.get("feature_adoption_rate", 0) > 0.3)
    
    def calculate_first_value_time(self, customer_data: Dict[str, Any]) -> float:
        """Calculate time to first value (days) - simplified"""
        # In real implementation, would track actual first value achievement
        tenure = customer_data.get("tenure_months", 0)
        return min(30, tenure * 7)  # Simplified calculation
    
    def calculate_usage_trend(self, customer_data: Dict[str, Any]) -> float:
        """Calculate usage trend over time"""
        last_login = customer_data.get("last_login_days", 30)
        total_days = customer_data.get("total_login_days", 1)
        
        # Calculate usage frequency
        usage_frequency = total_days / max(1, last_login)
        
        # Calculate trend (simplified)
        if usage_frequency > 2.0:
            return 1.0  # Strong upward trend
        elif usage_frequency > 1.0:
            return 0.5  # Stable
        elif usage_frequency > 0.5:
            return 0.0  # Slight decline
        else:
            return -1.0  # Declining
    
    def calculate_engagement_momentum(self, customer_data: Dict[str, Any]) -> float:
        """Calculate engagement momentum"""
        engagement_score = customer_data.get("engagement_score", 50)
        email_engagement = (customer_data.get("email_open_rate", 0.5) + 
                           customer_data.get("email_click_rate", 0.1)) / 2
        
        return (engagement_score / 100 + email_engagement) / 2
    
    def calculate_value_stability(self, customer_data: Dict[str, Any]) -> float:
        """Calculate value stability score"""
        tenure = customer_data.get("tenure_months", 1)
        payment_delays = customer_data.get("payment_delays", 0)
        
        # Base stability on tenure and payment history
        tenure_score = min(1.0, tenure / 12)  # More tenure = more stability
        payment_score = max(0, 1.0 - (payment_delays / 3))  # Fewer delays = more stability
        
        return (tenure_score + payment_score) / 2
    
    def calculate_support_ratio(self, customer_data: Dict[str, Any]) -> float:
        """Calculate support interaction ratio"""
        support_tickets = customer_data.get("support_tickets_30d", 0)
        total_login_days = customer_data.get("total_login_days", 1)
        
        return min(1.0, support_tickets / max(1, total_login_days * 0.1))
    
    def predict_churn_risk(self, customer_id: str) -> Dict[str, Any]:
        """Predict churn risk for a specific customer"""
        try:
            # Prepare customer data
            customer_data = self.prepare_customer_data(customer_id)
            if not customer_data:
                return {"error": "Failed to prepare customer data"}
            
            features = customer_data["features"]
            
            # Scale features
            features_scaled = self.feature_scaler.fit_transform([features])
            
            # Get predictions from all models
            model_predictions = {}
            for model_name, model in self.churn_model.items():
                try:
                    prediction = model.predict_proba(features_scaled)[0]
                    model_predictions[model_name] = {
                        "churn_probability": prediction[1],
                        "no_churn_probability": prediction[0]
                    }
                except Exception as e:
                    self.logger.warning(f"Error with {model_name}: {e}")
                    model_predictions[model_name] = {"churn_probability": 0.5}
            
            # Calculate ensemble prediction
            weights = self.prediction_config["model_settings"]["ensemble_weights"]
            ensemble_probability = sum(
                predictions["churn_probability"] * weights[model_name]
                for model_name, predictions in model_predictions.items()
            )
            
            # Determine risk category
            risk_category = self.classify_risk_category(ensemble_probability)
            
            # Generate insights
            insights = self.generate_churn_insights(customer_data["raw_data"], ensemble_probability)
            
            # Generate recommendations
            recommendations = self.generate_churn_recommendations(risk_category, customer_data["raw_data"])
            
            return {
                "customer_id": customer_id,
                "churn_risk_score": ensemble_probability,
                "risk_category": risk_category,
                "model_predictions": model_predictions,
                "confidence": self.calculate_confidence(model_predictions),
                "insights": insights,
                "recommendations": recommendations,
                "prediction_timestamp": datetime.now().isoformat(),
                "factors": self.identify_key_factors(features)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting churn risk: {str(e)}")
            return {"error": str(e)}
    
    def classify_risk_category(self, risk_score: float) -> str:
        """Classify churn risk into categories"""
        risk_categories = self.prediction_config["risk_categories"]
        
        for category, config in risk_categories.items():
            if config["min_score"] <= risk_score < (risk_categories.get(f"{category}s", {}).get("max_score", 1.0) if category != "minimal_risk" else 1.0):
                return category
        
        return "minimal_risk"  # Default fallback
    
    def generate_churn_insights(self, customer_data: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate insights about churn risk"""
        insights = []
        
        # Usage-based insights
        last_login = customer_data.get("last_login_days", 30)
        if last_login > 14:
            insights.append("Low recent activity - customer hasn't logged in recently")
        
        feature_adoption = customer_data.get("feature_adoption_rate", 0)
        if feature_adoption < 0.3:
            insights.append("Low feature adoption - customer may not be fully utilizing the platform")
        
        # Engagement insights
        engagement_score = customer_data.get("engagement_score", 50)
        if engagement_score < 60:
            insights.append("Low engagement score suggests potential satisfaction issues")
        
        # Support insights
        support_tickets = customer_data.get("support_tickets_30d", 0)
        if support_tickets > 3:
            insights.append("High number of support tickets indicates potential issues")
        
        # Financial insights
        monthly_value = customer_data.get("monthly_value", 0)
        payment_delays = customer_data.get("payment_delays", 0)
        if payment_delays > 1:
            insights.append("Payment delays may indicate financial stress")
        
        # Contract insights
        days_to_renewal = self.days_until_renewal(customer_data.get("contract_renewal_date"))
        if days_to_renewal < 60 and risk_score > 0.6:
            insights.append("Renewal period approaching with high churn risk")
        
        # Risk-specific insights
        if risk_score > 0.8:
            insights.append("Critical churn risk detected - immediate intervention recommended")
        elif risk_score > 0.6:
            insights.append("High churn risk - proactive outreach needed")
        
        return insights[:5]  # Return top 5 insights
    
    def generate_churn_recommendations(self, risk_category: str, customer_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk category and customer data"""
        recommendations = []
        
        # Risk category specific recommendations
        if risk_category == "high_risk":
            recommendations.extend([
                "Schedule immediate executive call within 24 hours",
                "Assign dedicated success manager for daily check-ins",
                "Offer personalized solution or service adjustment",
                "Provide special pricing or incentive to retain",
                "Conduct comprehensive needs assessment"
            ])
        elif risk_category == "medium_risk":
            recommendations.extend([
                "Schedule proactive outreach within 1 week",
                "Provide additional training or resources",
                "Share relevant case studies and success stories",
                "Offer feature optimization consultation",
                "Implement regular check-in schedule"
            ])
        elif risk_category == "low_risk":
            recommendations.extend([
                "Send personalized engagement content",
                "Invite to community events or webinars",
                "Provide industry-specific insights",
                "Offer advanced feature training",
                "Maintain regular communication schedule"
            ])
        
        # Data-driven recommendations
        if customer_data.get("support_tickets_30d", 0) > 3:
            recommendations.append("Address support ticket backlog and proactively resolve issues")
        
        if customer_data.get("feature_adoption_rate", 0) < 0.3:
            recommendations.append("Provide feature adoption training and support")
        
        if customer_data.get("email_open_rate", 0.5) < 0.3:
            recommendations.append("Improve email content relevance and targeting")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def calculate_confidence(self, model_predictions: Dict[str, Any]) -> float:
        """Calculate confidence in prediction based on model agreement"""
        probabilities = [pred["churn_probability"] for pred in model_predictions.values()]
        
        # Calculate standard deviation of predictions (lower = higher confidence)
        std_dev = np.std(probabilities)
        
        # Convert to confidence score (0-1, higher = more confident)
        confidence = max(0, 1 - (std_dev * 2))
        
        return confidence
    
    def identify_key_factors(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Identify key factors contributing to churn risk"""
        try:
            # In real implementation, would use feature importance from models
            # For demonstration, use feature values and business logic
            
            factor_importance = {}
            
            # Map feature indices to meaningful names
            feature_names = [
                "Total Login Days", "Avg Session Duration", "Feature Usage Count",
                "Feature Adoption Rate", "Days Since Last Login", "Session Frequency",
                "Engagement Score", "Email Open Rate", "Email Click Rate",
                "Support Tickets (30d)", "Help Docs Viewed", "Community Participation",
                "Monthly Value", "Payment Delays", "Days to Renewal",
                "Discount Received", "Upgrade Frequency", "Tenure (months)",
                "Onboarding Complete", "Days to First Value", "Milestone Achievements",
                "Feature Requests", "Complaint Frequency", "Satisfaction Score",
                "Usage Trend", "Engagement Momentum", "Value Stability",
                "Health Score", "Support Ratio", "Industry Churn Rate"
            ]
            
            for i, feature_name in enumerate(feature_names):
                if i < len(features):
                    factor_value = features[i]
                    
                    # Determine if factor is concerning (high risk indicator)
                    if feature_name in ["Days Since Last Login", "Support Tickets (30d)", "Payment Delays", "Feature Requests", "Complaint Frequency"]:
                        factor_impact = "high" if factor_value > np.mean(features) * 1.5 else "medium" if factor_value > np.mean(features) else "low"
                    elif feature_name in ["Engagement Score", "Feature Adoption Rate", "Health Score", "Satisfaction Score"]:
                        factor_impact = "high" if factor_value < 40 else "medium" if factor_value < 60 else "low"
                    else:
                        factor_impact = "medium"
                    
                    factor_importance[feature_name] = {
                        "value": factor_value,
                        "impact": factor_impact,
                        "normalized_value": min(1.0, factor_value / (np.mean(features) * 2) if np.mean(features) > 0 else 0)
                    }
            
            # Sort by impact and return top factors
            sorted_factors = sorted(
                factor_importance.items(), 
                key=lambda x: x[1]["normalized_value"], 
                reverse=True
            )
            
            return [
                {"factor": factor, "value": data["value"], "impact": data["impact"]}
                for factor, data in sorted_factors[:8]  # Top 8 factors
            ]
            
        except Exception as e:
            self.logger.error(f"Error identifying key factors: {str(e)}")
            return []
    
    def batch_predict_churn_risk(self, customer_ids: List[str]) -> List[Dict[str, Any]]:
        """Batch predict churn risk for multiple customers"""
        predictions = []
        
        for customer_id in customer_ids:
            try:
                prediction = self.predict_churn_risk(customer_id)
                predictions.append(prediction)
            except Exception as e:
                self.logger.error(f"Error predicting for customer {customer_id}: {str(e)}")
                predictions.append({
                    "customer_id": customer_id,
                    "error": str(e),
                    "prediction_timestamp": datetime.now().isoformat()
                })
        
        return predictions
    
    def update_predictions(self) -> Dict[str, Any]:
        """Update predictions for all customers (scheduled task)"""
        try:
            # In real implementation, would fetch all active customer IDs
            # For demo, generate some sample customer IDs
            
            customer_ids = [f"customer_{i:04d}" for i in range(1, 101)]  # 100 sample customers
            
            # Batch predict
            predictions = self.batch_predict_churn_risk(customer_ids)
            
            # Calculate summary statistics
            risk_distribution = self.calculate_risk_distribution(predictions)
            
            # Identify customers needing immediate attention
            high_risk_customers = [
                pred["customer_id"] for pred in predictions 
                if pred.get("risk_category") == "high_risk"
            ]
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "total_customers": len(predictions),
                "predictions_updated": len([p for p in predictions if "error" not in p]),
                "risk_distribution": risk_distribution,
                "high_risk_count": len(high_risk_customers),
                "high_risk_customers": high_risk_customers,
                "average_churn_risk": np.mean([p.get("churn_risk_score", 0) for p in predictions if "error" not in p])
            }
            
            self.logger.info(f"Updated predictions for {len(predictions)} customers")
            return results
            
        except Exception as e:
            self.logger.error(f"Error updating predictions: {str(e)}")
            return {"error": str(e)}
    
    def calculate_risk_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of risk categories"""
        distribution = {
            "high_risk": 0,
            "medium_risk": 0,
            "low_risk": 0,
            "minimal_risk": 0,
            "unknown": 0
        }
        
        for prediction in predictions:
            risk_category = prediction.get("risk_category", "unknown")
            if risk_category in distribution:
                distribution[risk_category] += 1
            else:
                distribution["unknown"] += 1
        
        return distribution
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        # In real implementation, would calculate from validation set
        # For demo, return simulated metrics
        
        return {
            "accuracy": 0.87,
            "precision": 0.82,
            "recall": 0.79,
            "f1_score": 0.80,
            "auc_score": 0.89,
            "model_type": "ensemble",
            "feature_count": len(self.feature_columns),
            "last_trained": "2024-11-01T10:00:00Z",
            "training_samples": 10000,
            "validation_samples": 2000
        }
    
    def get_risk_distribution(self) -> Dict[str, Any]:
        """Get current risk distribution across all customers"""
        # This would query the database for current customer risk scores
        # For demo, return simulated data
        
        return {
            "total_customers": 1000,
            "risk_distribution": {
                "high_risk": 50,  # 5%
                "medium_risk": 150,  # 15%
                "low_risk": 300,  # 30%
                "minimal_risk": 500  # 50%
            },
            "average_risk_score": 0.28,
            "high_risk_percentage": 5.0,
            "trend": "improving",  # vs last month
            "last_updated": datetime.now().isoformat()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get churn prediction engine status"""
        return {
            "status": "operational",
            "model_performance": self.get_model_performance(),
            "prediction_frequency": "real-time",
            "supported_customers": 10000,
            "risk_categories": list(self.prediction_config["risk_categories"].keys()),
            "feature_count": len(self.feature_columns),
            "last_update": datetime.now().isoformat()
        }
