"""
Customer Behavior Analytics Engine
Analyzes customer behavior patterns and provides insights for optimization
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sqlite3
from pathlib import Path


class BehaviorAnalytics:
    """
    Customer behavior analytics and pattern recognition engine
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analytics configuration
        self.behavior_metrics = self.define_behavior_metrics()
        self.behavior_patterns = self.load_behavior_patterns()
        self.analytics_rules = self.load_analytics_rules()
        
        # Machine learning models
        self.behavior_clustering_model = None
        self.pattern_recognition_model = None
        self.feature_scaler = StandardScaler()
        
        # Data storage
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Analytics tracking
        self.behavior_insights = {}
        self.pattern_analysis_cache = {}
        
        self.initialize_models()
        
        self.logger.info("Behavior Analytics Engine initialized")
    
    def define_behavior_metrics(self) -> Dict[str, Any]:
        """Define comprehensive behavior metrics"""
        return {
            "usage_metrics": {
                "session_metrics": {
                    "total_sessions": "Total number of login sessions",
                    "avg_session_duration": "Average session length in minutes",
                    "session_frequency": "Sessions per week/month",
                    "peak_usage_hours": "Most active usage hours",
                    "weekend_usage": "Weekend vs weekday usage patterns"
                },
                "feature_metrics": {
                    "features_used": "Number of distinct features used",
                    "feature_frequency": "How often each feature is used",
                    "feature_adoption_rate": "Rate of adopting new features",
                    "feature_dependencies": "Which features are used together",
                    "advanced_feature_usage": "Usage of advanced/premium features"
                },
                "content_metrics": {
                    "content_consumed": "Pages/articles/videos viewed",
                    "content_types": "Types of content consumed",
                    "content_engagement": "Time spent on content",
                    "content_sharing": "Content sharing behavior",
                    "content_completion": "Completion rate of content"
                }
            },
            "engagement_metrics": {
                "interaction_metrics": {
                    "email_open_rate": "Percentage of emails opened",
                    "email_click_rate": "Percentage of email links clicked",
                    "notification_response": "Response to in-app notifications",
                    "support_interactions": "Support ticket frequency and resolution",
                    "community_participation": "Forum/chat community activity"
                },
                "participation_metrics": {
                    "event_attendance": "Webinar and event attendance",
                    "training_completion": "Training course completion",
                    "survey_responses": "Survey and feedback participation",
                    "feedback_quality": "Quality and helpfulness of feedback",
                    "advocacy_actions": "Referral and testimonial activities"
                },
                "progression_metrics": {
                    "onboarding_completion": "Completion rate of onboarding",
                    "milestone_achievement": "Achievement of key milestones",
                    "goal_completion": "Completion of stated goals",
                    "success_indicators": "Key success metric progression",
                    "value_realization": "Time to first value achievement"
                }
            },
            "satisfaction_metrics": {
                "explicit_feedback": {
                    "nps_score": "Net Promoter Score",
                    "csat_score": "Customer Satisfaction Score",
                    "ces_score": "Customer Effort Score",
                    "feature_ratings": "Ratings for specific features",
                    "overall_rating": "Overall product rating"
                },
                "behavioral_feedback": {
                    "usage_consistency": "Consistency of platform usage",
                    "renewal_behavior": "Contract renewal patterns",
                    "support_ticket_urgency": "Urgency level of support requests",
                    "feature_requests": "Frequency and type of feature requests",
                    "complaint_patterns": "Patterns in complaints and issues"
                },
                "loyalty_indicators": {
                    "referral_activity": "Customer referral generation",
                    "testimonial_willingness": "Willingness to provide testimonials",
                    "case_study_participation": "Participation in case studies",
                    "community_leadership": "Leadership in customer community",
                    "brand_advocacy": "Public advocacy and promotion"
                }
            },
            "business_metrics": {
                "value_metrics": {
                    "monthly_value": "Monthly contract value",
                    "lifetime_value": "Total customer lifetime value",
                    "expansion_revenue": "Revenue from upsells/cross-sells",
                    "cost_to_serve": "Cost to serve the customer",
                    "roi_realization": "Customer's return on investment"
                },
                "retention_metrics": {
                    "retention_rate": "Customer retention percentage",
                    "churn_risk": "Probability of customer churn",
                    "renewal_likelihood": "Likelihood of contract renewal",
                    "engagement_retention": "Engagement level impact on retention",
                    "satisfaction_retention": "Satisfaction level impact on retention"
                },
                "growth_metrics": {
                    "user_growth": "Number of users in customer organization",
                    "feature_adoption_growth": "Growth in feature adoption",
                    "usage_intensity_growth": "Growth in usage intensity",
                    "value_expansion": "Expansion in customer value",
                    "strategic_importance": "Customer's strategic importance"
                }
            }
        }
    
    def load_behavior_patterns(self) -> Dict[str, Any]:
        """Load predefined behavior patterns and classifications"""
        return {
            "usage_patterns": {
                "power_user": {
                    "characteristics": {
                        "daily_logins": ">3",
                        "advanced_features_used": ">5",
                        "session_duration": ">60 minutes",
                        "feature_adoption_rate": ">80%",
                        "training_completion": "100%"
                    },
                    "behavioral_indicators": [
                        "Explores new features immediately",
                        "Uses automation capabilities",
                        "Integrates with other systems",
                        "Provides feature feedback",
                        "Shares best practices"
                    ],
                    "success_profile": "High value, high engagement, potential advocate"
                },
                "regular_user": {
                    "characteristics": {
                        "weekly_logins": "3-7",
                        "core_features_used": "2-4",
                        "session_duration": "15-45 minutes",
                        "feature_adoption_rate": "40-70%",
                        "training_completion": "60-80%"
                    },
                    "behavioral_indicators": [
                        "Uses platform consistently",
                        "Sticks to familiar features",
                        "Occasional new feature adoption",
                        "Moderate support interaction",
                        "Steady workflow integration"
                    ],
                    "success_profile": "Stable value, consistent usage, expansion potential"
                },
                "occasional_user": {
                    "characteristics": {
                        "weekly_logins": "1-2",
                        "basic_features_used": "1-3",
                        "session_duration": "5-20 minutes",
                        "feature_adoption_rate": "20-40%",
                        "training_completion": "30-50%"
                    },
                    "behavioral_indicators": [
                        "Uses platform sporadically",
                        "Limited feature exploration",
                        "Task-focused usage",
                        "Occasional support needs",
                        "Basic workflow support"
                    ],
                    "success_profile": "Variable value, limited engagement, at-risk"
                },
                "dormant_user": {
                    "characteristics": {
                        "monthly_logins": "<4",
                        "features_used": "<2",
                        "session_duration": "<10 minutes",
                        "feature_adoption_rate": "<20%",
                        "training_completion": "<30%"
                    },
                    "behavioral_indicators": [
                        "Rarely logs in",
                        "Minimal feature usage",
                        "No new feature adoption",
                        "Frequent support tickets",
                        "Limited workflow integration"
                    ],
                    "success_profile": "Low value, high churn risk, intervention needed"
                }
            },
            "engagement_patterns": {
                "highly_engaged": {
                    "email_metrics": {"open_rate": ">70%", "click_rate": ">20%"},
                    "interaction_frequency": "daily",
                    "content_consumption": "high",
                    "event_participation": "frequent",
                    "feedback_quality": "detailed"
                },
                "moderately_engaged": {
                    "email_metrics": {"open_rate": "40-70%", "click_rate": "10-20%"},
                    "interaction_frequency": "weekly",
                    "content_consumption": "moderate",
                    "event_participation": "occasional",
                    "feedback_quality": "standard"
                },
                "low_engaged": {
                    "email_metrics": {"open_rate": "<40%", "click_rate": "<10%"},
                    "interaction_frequency": "monthly",
                    "content_consumption": "low",
                    "event_participation": "rare",
                    "feedback_quality": "minimal"
                }
            },
            "satisfaction_patterns": {
                "promoters": {
                    "nps_score": "9-10",
                    "behavioral_indicators": [
                        "Renewals without hesitation",
                        "Positive feedback sharing",
                        "Referral generation",
                        "Case study participation",
                        "Community leadership"
                    ]
                },
                "passives": {
                    "nps_score": "7-8",
                    "behavioral_indicators": [
                        "Renewal with questions",
                        "Mixed feedback",
                        "Occasional referrals",
                        "Limited participation",
                        "Standard community member"
                    ]
                },
                "detractors": {
                    "nps_score": "0-6",
                    "behavioral_indicators": [
                        "Renewal hesitation",
                        "Critical feedback",
                        "Complaint behavior",
                        "Churn consideration",
                        "Negative sentiment"
                    ]
                }
            },
            "business_patterns": {
                "growth_customer": {
                    "characteristics": [
                        "User count increasing",
                        "Feature adoption growing",
                        "Usage intensity rising",
                        "Value expansion occurring",
                        "Strategic importance increasing"
                    ]
                },
                "stable_customer": {
                    "characteristics": [
                        "User count steady",
                        "Feature adoption plateau",
                        "Usage intensity stable",
                        "Value consistent",
                        "Strategic importance maintained"
                    ]
                },
                "declining_customer": {
                    "characteristics": [
                        "User count decreasing",
                        "Feature adoption declining",
                        "Usage intensity dropping",
                        "Value erosion",
                        "Strategic importance declining"
                    ]
                }
            }
        }
    
    def load_analytics_rules(self) -> Dict[str, Any]:
        """Load analytics rules and algorithms"""
        return {
            "pattern_recognition_rules": {
                "anomaly_detection": {
                    "usage_anomalies": "Detect unusual usage patterns",
                    "engagement_anomalies": "Detect sudden engagement changes",
                    "satisfaction_anomalies": "Detect satisfaction score changes",
                    "behavioral_anomalies": "Detect unusual behavioral shifts"
                },
                "trend_analysis": {
                    "usage_trends": "Analyze usage pattern trends over time",
                    "engagement_trends": "Analyze engagement evolution",
                    "satisfaction_trends": "Analyze satisfaction trajectory",
                    "value_trends": "Analyze customer value evolution"
                },
                "correlation_analysis": {
                    "usage_satisfaction": "Correlation between usage and satisfaction",
                    "engagement_retention": "Correlation between engagement and retention",
                    "feature_value": "Correlation between feature usage and value",
                    "interaction_outcomes": "Correlation between interactions and outcomes"
                }
            },
            "segmentation_rules": {
                "behavioral_segmentation": {
                    "usage_based": "Segment by usage patterns and intensity",
                    "engagement_based": "Segment by engagement levels and types",
                    "satisfaction_based": "Segment by satisfaction levels and drivers",
                    "value_based": "Segment by customer value and growth potential"
                },
                "dynamic_segmentation": {
                    "real_time_segmentation": "Update segments based on real-time behavior",
                    "lifecycle_segmentation": "Segment based on customer lifecycle stage",
                    "event_based_segmentation": "Segment based on specific events or triggers",
                    "predictive_segmentation": "Predict future segment membership"
                }
            },
            "predictive_analytics": {
                "behavior_prediction": {
                    "usage_prediction": "Predict future usage patterns",
                    "engagement_prediction": "Predict engagement level changes",
                    "satisfaction_prediction": "Predict satisfaction score changes",
                    "churn_prediction": "Predict churn likelihood based on behavior"
                },
                "outcome_prediction": {
                    "renewal_prediction": "Predict renewal likelihood",
                    "expansion_prediction": "Predict expansion opportunities",
                    "advocacy_prediction": "Predict advocacy potential",
                    "retention_prediction": "Predict retention probability"
                }
            },
            "optimization_insights": {
                "behavioral_optimization": {
                    "usage_optimization": "Recommendations to improve usage patterns",
                    "engagement_optimization": "Recommendations to boost engagement",
                    "satisfaction_optimization": "Recommendations to improve satisfaction",
                    "value_optimization": "Recommendations to maximize customer value"
                },
                "intervention_insights": {
                    "intervention_triggers": "Behavioral triggers for intervention",
                    "intervention_effectiveness": "Effectiveness of different intervention types",
                    "intervention_timing": "Optimal timing for interventions",
                    "intervention_messaging": "Behavioral insights for intervention messaging"
                }
            }
        }
    
    def initialize_models(self):
        """Initialize machine learning models for behavior analytics"""
        try:
            # Behavior clustering model
            self.behavior_clustering_model = KMeans(
                n_clusters=6, random_state=42, max_iter=300
            )
            
            # Pattern recognition model
            self.pattern_recognition_model = RandomForestClassifier(
                n_estimators=100, random_state=42, max_depth=10
            )
            
            # Feature scaler
            self.feature_scaler = StandardScaler()
            
            self.logger.info("Behavior analytics models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing behavior models: {str(e)}")
    
    def analyze_customer_behavior(self, customer_id: str, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive behavior analysis for a customer"""
        try:
            # Extract behavior features
            behavior_features = self.extract_behavior_features(behavior_data)
            
            # Identify behavior patterns
            pattern_identification = self.identify_behavior_patterns(behavior_features)
            
            # Perform segmentation
            customer_segment = self.segment_customer(behavior_features)
            
            # Calculate behavior metrics
            behavior_metrics = self.calculate_behavior_metrics(behavior_data)
            
            # Analyze trends
            trend_analysis = self.analyze_behavior_trends(behavior_data)
            
            # Detect anomalies
            anomaly_detection = self.detect_behavior_anomalies(behavior_data)
            
            # Generate insights
            behavior_insights = self.generate_behavior_insights(
                pattern_identification, customer_segment, behavior_metrics
            )
            
            # Generate recommendations
            recommendations = self.generate_behavior_recommendations(
                customer_segment, behavior_insights, behavior_data
            )
            
            return {
                "customer_id": customer_id,
                "behavior_features": behavior_features,
                "pattern_identification": pattern_identification,
                "customer_segment": customer_segment,
                "behavior_metrics": behavior_metrics,
                "trend_analysis": trend_analysis,
                "anomaly_detection": anomaly_detection,
                "behavior_insights": behavior_insights,
                "recommendations": recommendations,
                "analysis_timestamp": datetime.now().isoformat(),
                "confidence_score": self.calculate_analysis_confidence(behavior_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing customer behavior: {str(e)}")
            return {"error": str(e)}
    
    def extract_behavior_features(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for behavior analysis"""
        try:
            features = {
                "usage_features": {
                    "login_frequency": behavior_data.get("login_frequency", 0),
                    "session_duration_avg": behavior_data.get("session_duration_avg", 0),
                    "session_duration_total": behavior_data.get("session_duration_total", 0),
                    "features_used_count": behavior_data.get("features_used_count", 0),
                    "feature_adoption_rate": behavior_data.get("feature_adoption_rate", 0),
                    "advanced_features_usage": behavior_data.get("advanced_features_usage", 0),
                    "content_consumption": behavior_data.get("content_consumption", 0),
                    "content_completion_rate": behavior_data.get("content_completion_rate", 0)
                },
                "engagement_features": {
                    "email_open_rate": behavior_data.get("email_open_rate", 0),
                    "email_click_rate": behavior_data.get("email_click_rate", 0),
                    "notification_response_rate": behavior_data.get("notification_response_rate", 0),
                    "support_ticket_frequency": behavior_data.get("support_ticket_frequency", 0),
                    "community_participation": behavior_data.get("community_participation", 0),
                    "event_attendance": behavior_data.get("event_attendance", 0),
                    "training_completion": behavior_data.get("training_completion", 0),
                    "feedback_provided": behavior_data.get("feedback_provided", 0)
                },
                "satisfaction_features": {
                    "nps_score": behavior_data.get("nps_score", 7),
                    "csat_score": behavior_data.get("csat_score", 3.5),
                    "ces_score": behavior_data.get("ces_score", 3.0),
                    "renewal_behavior": behavior_data.get("renewal_behavior", "neutral"),
                    "complaint_frequency": behavior_data.get("complaint_frequency", 0),
                    "referral_activity": behavior_data.get("referral_activity", 0),
                    "testimonial_willingness": behavior_data.get("testimonial_willingness", 0),
                    "advocacy_actions": behavior_data.get("advocacy_actions", 0)
                },
                "business_features": {
                    "monthly_value": behavior_data.get("monthly_value", 100),
                    "user_count": behavior_data.get("user_count", 1),
                    "usage_intensity": behavior_data.get("usage_intensity", 0),
                    "expansion_activity": behavior_data.get("expansion_activity", 0),
                    "contract_duration": behavior_data.get("contract_duration", 12),
                    "payment_consistency": behavior_data.get("payment_consistency", "regular"),
                    "strategic_importance": behavior_data.get("strategic_importance", "medium")
                },
                "temporal_features": {
                    "tenure_months": behavior_data.get("tenure_months", 0),
                    "days_since_last_login": behavior_data.get("days_since_last_login", 30),
                    "activity_recency": behavior_data.get("activity_recency", 7),
                    "engagement_momentum": behavior_data.get("engagement_momentum", 0)
                }
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting behavior features: {str(e)}")
            return {}
    
    def identify_behavior_patterns(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Identify behavior patterns from features"""
        try:
            usage_pattern = self.classify_usage_pattern(behavior_features)
            engagement_pattern = self.classify_engagement_pattern(behavior_features)
            satisfaction_pattern = self.classify_satisfaction_pattern(behavior_features)
            business_pattern = self.classify_business_pattern(behavior_features)
            
            return {
                "usage_pattern": usage_pattern,
                "engagement_pattern": engagement_pattern,
                "satisfaction_pattern": satisfaction_pattern,
                "business_pattern": business_pattern,
                "overall_pattern": self.combine_patterns(usage_pattern, engagement_pattern, satisfaction_pattern),
                "pattern_confidence": self.calculate_pattern_confidence(behavior_features)
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying behavior patterns: {str(e)}")
            return {}
    
    def classify_usage_pattern(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify customer usage pattern"""
        try:
            usage_features = behavior_features.get("usage_features", {})
            
            login_frequency = usage_features.get("login_frequency", 0)
            session_duration_avg = usage_features.get("session_duration_avg", 0)
            feature_adoption_rate = usage_features.get("feature_adoption_rate", 0)
            advanced_features_usage = usage_features.get("advanced_features_usage", 0)
            
            # Classify based on usage patterns
            if (login_frequency > 21 and session_duration_avg > 60 and 
                feature_adoption_rate > 0.8 and advanced_features_usage > 5):
                pattern = "power_user"
                confidence = 0.9
            elif (login_frequency > 7 and session_duration_avg > 30 and 
                  feature_adoption_rate > 0.5):
                pattern = "regular_user"
                confidence = 0.8
            elif (login_frequency > 1 and session_duration_avg > 10 and 
                  feature_adoption_rate > 0.2):
                pattern = "occasional_user"
                confidence = 0.7
            else:
                pattern = "dormant_user"
                confidence = 0.8
            
            # Get pattern characteristics
            pattern_characteristics = self.behavior_patterns["usage_patterns"].get(pattern, {})
            
            return {
                "pattern": pattern,
                "confidence": confidence,
                "characteristics": pattern_characteristics.get("characteristics", {}),
                "behavioral_indicators": pattern_characteristics.get("behavioral_indicators", []),
                "success_profile": pattern_characteristics.get("success_profile", ""),
                "key_metrics": {
                    "login_frequency": login_frequency,
                    "session_duration_avg": session_duration_avg,
                    "feature_adoption_rate": feature_adoption_rate,
                    "advanced_features_usage": advanced_features_usage
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying usage pattern: {str(e)}")
            return {"pattern": "unknown", "confidence": 0}
    
    def classify_engagement_pattern(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify customer engagement pattern"""
        try:
            engagement_features = behavior_features.get("engagement_features", {})
            
            email_open_rate = engagement_features.get("email_open_rate", 0.5)
            email_click_rate = engagement_features.get("email_click_rate", 0.1)
            community_participation = engagement_features.get("community_participation", 0)
            event_attendance = engagement_features.get("event_attendance", 0)
            
            # Classify based on engagement levels
            if email_open_rate > 0.7 and email_click_rate > 0.2 and community_participation > 0.5:
                pattern = "highly_engaged"
                confidence = 0.85
            elif email_open_rate > 0.4 and email_click_rate > 0.1:
                pattern = "moderately_engaged"
                confidence = 0.75
            else:
                pattern = "low_engaged"
                confidence = 0.8
            
            # Get pattern characteristics
            pattern_characteristics = self.behavior_patterns["engagement_patterns"].get(pattern, {})
            
            return {
                "pattern": pattern,
                "confidence": confidence,
                "characteristics": pattern_characteristics.get("characteristics", {}),
                "key_metrics": {
                    "email_open_rate": email_open_rate,
                    "email_click_rate": email_click_rate,
                    "community_participation": community_participation,
                    "event_attendance": event_attendance
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying engagement pattern: {str(e)}")
            return {"pattern": "unknown", "confidence": 0}
    
    def classify_satisfaction_pattern(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify customer satisfaction pattern"""
        try:
            satisfaction_features = behavior_features.get("satisfaction_features", {})
            
            nps_score = satisfaction_features.get("nps_score", 7)
            referral_activity = satisfaction_features.get("referral_activity", 0)
            testimonial_willingness = satisfaction_features.get("testimonial_willingness", 0)
            advocacy_actions = satisfaction_features.get("advocacy_actions", 0)
            
            # Classify based on satisfaction indicators
            if nps_score >= 9:
                pattern = "promoters"
                confidence = 0.9
            elif nps_score >= 7:
                pattern = "passives"
                confidence = 0.8
            else:
                pattern = "detractors"
                confidence = 0.85
            
            # Get pattern characteristics
            pattern_characteristics = self.behavior_patterns["satisfaction_patterns"].get(pattern, {})
            
            return {
                "pattern": pattern,
                "confidence": confidence,
                "nps_score": nps_score,
                "behavioral_indicators": pattern_characteristics.get("behavioral_indicators", []),
                "key_metrics": {
                    "nps_score": nps_score,
                    "referral_activity": referral_activity,
                    "testimonial_willingness": testimonial_willingness,
                    "advocacy_actions": advocacy_actions
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying satisfaction pattern: {str(e)}")
            return {"pattern": "unknown", "confidence": 0}
    
    def classify_business_pattern(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify customer business pattern"""
        try:
            business_features = behavior_features.get("business_features", {})
            temporal_features = behavior_features.get("temporal_features", {})
            
            monthly_value = business_features.get("monthly_value", 100)
            user_count = business_features.get("user_count", 1)
            usage_intensity = business_features.get("usage_intensity", 0)
            expansion_activity = business_features.get("expansion_activity", 0)
            tenure_months = temporal_features.get("tenure_months", 0)
            
            # Classify based on business patterns
            if expansion_activity > 0.3 and usage_intensity > 0.7 and user_count > 10:
                pattern = "growth_customer"
                confidence = 0.8
            elif usage_intensity > 0.4 and expansion_activity > 0.1:
                pattern = "stable_customer"
                confidence = 0.75
            else:
                pattern = "declining_customer"
                confidence = 0.8
            
            # Get pattern characteristics
            pattern_characteristics = self.behavior_patterns["business_patterns"].get(pattern, {})
            
            return {
                "pattern": pattern,
                "confidence": confidence,
                "characteristics": pattern_characteristics.get("characteristics", []),
                "key_metrics": {
                    "monthly_value": monthly_value,
                    "user_count": user_count,
                    "usage_intensity": usage_intensity,
                    "expansion_activity": expansion_activity,
                    "tenure_months": tenure_months
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying business pattern: {str(e)}")
            return {"pattern": "unknown", "confidence": 0}
    
    def combine_patterns(self, usage_pattern: Dict[str, Any], 
                        engagement_pattern: Dict[str, Any], 
                        satisfaction_pattern: Dict[str, Any]) -> str:
        """Combine individual patterns into overall behavior pattern"""
        try:
            usage_type = usage_pattern.get("pattern", "unknown")
            engagement_type = engagement_pattern.get("pattern", "unknown")
            satisfaction_type = satisfaction_pattern.get("pattern", "unknown")
            
            # Simple combination logic
            if usage_type in ["power_user", "regular_user"] and engagement_type in ["highly_engaged", "moderately_engaged"]:
                if satisfaction_type in ["promoters"]:
                    return "advocate"
                elif satisfaction_type in ["passives"]:
                    return "champion"
                else:
                    return "engaged_user"
            elif usage_type == "regular_user" and engagement_type == "moderately_engaged":
                return "steady_customer"
            elif usage_type == "occasional_user" or engagement_type == "low_engaged":
                return "at_risk"
            else:
                return "underperforming"
                
        except Exception as e:
            self.logger.error(f"Error combining patterns: {str(e)}")
            return "unknown"
    
    def calculate_pattern_confidence(self, behavior_features: Dict[str, Any]) -> float:
        """Calculate confidence in pattern identification"""
        try:
            confidence_factors = []
            
            # Data completeness factor
            required_features = [
                "usage_features.login_frequency",
                "engagement_features.email_open_rate",
                "satisfaction_features.nps_score",
                "business_features.monthly_value"
            ]
            
            completeness = 0
            for feature in required_features:
                if self.get_nested_value(behavior_features, feature) is not None:
                    completeness += 1
            
            confidence_factors.append(completeness / len(required_features))
            
            # Feature consistency factor
            usage_features = behavior_features.get("usage_features", {})
            if usage_features.get("login_frequency", 0) > 0 and usage_features.get("session_duration_avg", 0) > 0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.5)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern confidence: {str(e)}")
            return 0.5
    
    def get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def segment_customer(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Segment customer based on behavior features"""
        try:
            # Extract features for clustering
            clustering_features = []
            feature_names = []
            
            # Usage features
            usage_features = behavior_features.get("usage_features", {})
            clustering_features.extend([
                usage_features.get("login_frequency", 0),
                usage_features.get("session_duration_avg", 0),
                usage_features.get("feature_adoption_rate", 0),
                usage_features.get("advanced_features_usage", 0)
            ])
            feature_names.extend(["login_frequency", "session_duration", "feature_adoption", "advanced_usage"])
            
            # Engagement features
            engagement_features = behavior_features.get("engagement_features", {})
            clustering_features.extend([
                engagement_features.get("email_open_rate", 0),
                engagement_features.get("email_click_rate", 0),
                engagement_features.get("community_participation", 0)
            ])
            feature_names.extend(["email_open_rate", "email_click_rate", "community_participation"])
            
            # Satisfaction features
            satisfaction_features = behavior_features.get("satisfaction_features", {})
            clustering_features.extend([
                satisfaction_features.get("nps_score", 7),
                satisfaction_features.get("referral_activity", 0),
                satisfaction_features.get("advocacy_actions", 0)
            ])
            feature_names.extend(["nps_score", "referral_activity", "advocacy_actions"])
            
            # Business features
            business_features = behavior_features.get("business_features", {})
            clustering_features.extend([
                business_features.get("monthly_value", 100),
                business_features.get("usage_intensity", 0),
                business_features.get("expansion_activity", 0)
            ])
            feature_names.extend(["monthly_value", "usage_intensity", "expansion_activity"])
            
            # Perform clustering
            features_array = np.array(clustering_features).reshape(1, -1)
            features_scaled = self.feature_scaler.fit_transform(features_array)
            
            # In real implementation, would use trained model
            # For demo, use simple segmentation logic
            segment = self.determine_simple_segment(behavior_features)
            
            return {
                "segment": segment["segment"],
                "segment_characteristics": segment["characteristics"],
                "clustering_features": dict(zip(feature_names, clustering_features)),
                "segmentation_confidence": segment["confidence"],
                "recommended_actions": segment["actions"]
            }
            
        except Exception as e:
            self.logger.error(f"Error segmenting customer: {str(e)}")
            return {"segment": "unknown", "characteristics": []}
    
    def determine_simple_segment(self, behavior_features: Dict[str, Any]) -> Dict[str, Any]:
        """Determine customer segment using simple rules"""
        try:
            usage_features = behavior_features.get("usage_features", {})
            engagement_features = behavior_features.get("engagement_features", {})
            satisfaction_features = behavior_features.get("satisfaction_features", {})
            business_features = behavior_features.get("business_features", {})
            
            nps_score = satisfaction_features.get("nps_score", 7)
            monthly_value = business_features.get("monthly_value", 100)
            login_frequency = usage_features.get("login_frequency", 0)
            feature_adoption_rate = usage_features.get("feature_adoption_rate", 0)
            
            # Segment determination logic
            if nps_score >= 9 and monthly_value > 1000 and login_frequency > 14 and feature_adoption_rate > 0.7:
                segment = "champions"
                characteristics = ["High satisfaction", "High value", "High usage", "Strong advocacy"]
                actions = ["Expansion opportunities", "Case study requests", "Referral programs"]
                confidence = 0.9
            elif nps_score >= 7 and monthly_value > 500:
                segment = "loyal_customers"
                characteristics = ["Good satisfaction", "Stable value", "Consistent usage"]
                actions = ["Retention focus", "Value enhancement", "Feature adoption"]
                confidence = 0.8
            elif login_frequency < 4 or feature_adoption_rate < 0.3:
                segment = "at_risk"
                characteristics = ["Low usage", "Low engagement", "Potential churn risk"]
                actions = ["Engagement campaigns", "Usage optimization", "Support enhancement"]
                confidence = 0.85
            elif monthly_value < 200:
                segment = "small_value"
                characteristics = ["Low monetary value", "Basic usage", "Growth potential"]
                actions = ["Usage optimization", "Feature expansion", "Value demonstration"]
                confidence = 0.75
            else:
                segment = "standard"
                characteristics = ["Average metrics", "Standard usage", "Stable engagement"]
                actions = ["Engagement improvement", "Feature education", "Value optimization"]
                confidence = 0.7
            
            return {
                "segment": segment,
                "characteristics": characteristics,
                "actions": actions,
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error determining simple segment: {str(e)}")
            return {"segment": "unknown", "characteristics": [], "actions": [], "confidence": 0.5}
    
    def calculate_behavior_metrics(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive behavior metrics"""
        try:
            metrics = {
                "usage_metrics": self.calculate_usage_metrics(behavior_data),
                "engagement_metrics": self.calculate_engagement_metrics(behavior_data),
                "satisfaction_metrics": self.calculate_satisfaction_metrics(behavior_data),
                "business_metrics": self.calculate_business_metrics(behavior_data),
                "composite_scores": self.calculate_composite_scores(behavior_data)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating behavior metrics: {str(e)}")
            return {}
    
    def calculate_usage_metrics(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate usage-specific metrics"""
        try:
            return {
                "usage_frequency_score": self.calculate_frequency_score(behavior_data),
                "usage_intensity_score": self.calculate_intensity_score(behavior_data),
                "feature_adoption_score": self.calculate_adoption_score(behavior_data),
                "usage_efficiency_score": self.calculate_efficiency_score(behavior_data),
                "overall_usage_score": self.calculate_overall_usage_score(behavior_data)
            }
        except Exception as e:
            self.logger.error(f"Error calculating usage metrics: {str(e)}")
            return {}
    
    def calculate_engagement_metrics(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate engagement-specific metrics"""
        try:
            return {
                "communication_engagement": self.calculate_communication_engagement(behavior_data),
                "participation_engagement": self.calculate_participation_engagement(behavior_data),
                "interaction_quality": self.calculate_interaction_quality(behavior_data),
                "overall_engagement_score": self.calculate_overall_engagement_score(behavior_data)
            }
        except Exception as e:
            self.logger.error(f"Error calculating engagement metrics: {str(e)}")
            return {}
    
    def calculate_satisfaction_metrics(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate satisfaction-specific metrics"""
        try:
            return {
                "explicit_satisfaction": self.calculate_explicit_satisfaction(behavior_data),
                "behavioral_satisfaction": self.calculate_behavioral_satisfaction(behavior_data),
                "loyalty_indicators": self.calculate_loyalty_indicators(behavior_data),
                "overall_satisfaction_score": self.calculate_overall_satisfaction_score(behavior_data)
            }
        except Exception as e:
            self.logger.error(f"Error calculating satisfaction metrics: {str(e)}")
            return {}
    
    def calculate_business_metrics(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business-specific metrics"""
        try:
            return {
                "value_realization": self.calculate_value_realization(behavior_data),
                "retention_indicators": self.calculate_retention_indicators(behavior_data),
                "growth_potential": self.calculate_growth_potential(behavior_data),
                "overall_business_score": self.calculate_overall_business_score(behavior_data)
            }
        except Exception as e:
            self.logger.error(f"Error calculating business metrics: {str(e)}")
            return {}
    
    def calculate_composite_scores(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate composite behavior scores"""
        try:
            # Individual component scores
            usage_score = self.calculate_overall_usage_score(behavior_data)
            engagement_score = self.calculate_overall_engagement_score(behavior_data)
            satisfaction_score = self.calculate_overall_satisfaction_score(behavior_data)
            business_score = self.calculate_overall_business_score(behavior_data)
            
            # Weighted composite scores
            overall_behavior_score = (
                usage_score * 0.3 + 
                engagement_score * 0.25 + 
                satisfaction_score * 0.25 + 
                business_score * 0.2
            )
            
            health_score = (
                usage_score * 0.4 + 
                satisfaction_score * 0.6
            )
            
            value_score = (
                business_score * 0.7 + 
                engagement_score * 0.3
            )
            
            return {
                "overall_behavior_score": overall_behavior_score,
                "customer_health_score": health_score,
                "customer_value_score": value_score,
                "engagement_retention_score": engagement_score,
                "satisfaction_loyalty_score": satisfaction_score
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating composite scores: {str(e)}")
            return {}
    
    # Metric calculation helper methods
    def calculate_frequency_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate usage frequency score"""
        login_frequency = behavior_data.get("login_frequency", 0)
        return min(100, login_frequency * 5)  # Scale to 0-100
    
    def calculate_intensity_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate usage intensity score"""
        session_duration = behavior_data.get("session_duration_avg", 0)
        features_used = behavior_data.get("features_used_count", 0)
        return min(100, (session_duration / 60 * 40) + (features_used * 10))
    
    def calculate_adoption_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate feature adoption score"""
        adoption_rate = behavior_data.get("feature_adoption_rate", 0)
        return adoption_rate * 100
    
    def calculate_efficiency_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate usage efficiency score"""
        content_completion = behavior_data.get("content_completion_rate", 0)
        return content_completion * 100
    
    def calculate_overall_usage_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate overall usage score"""
        frequency = self.calculate_frequency_score(behavior_data)
        intensity = self.calculate_intensity_score(behavior_data)
        adoption = self.calculate_adoption_score(behavior_data)
        efficiency = self.calculate_efficiency_score(behavior_data)
        
        return (frequency + intensity + adoption + efficiency) / 4
    
    def calculate_communication_engagement(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate communication engagement score"""
        open_rate = behavior_data.get("email_open_rate", 0)
        click_rate = behavior_data.get("email_click_rate", 0)
        return (open_rate * 0.6 + click_rate * 0.4) * 100
    
    def calculate_participation_engagement(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate participation engagement score"""
        community_participation = behavior_data.get("community_participation", 0)
        event_attendance = behavior_data.get("event_attendance", 0)
        return (community_participation * 0.6 + event_attendance * 0.4) * 100
    
    def calculate_interaction_quality(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate interaction quality score"""
        support_ticket_frequency = behavior_data.get("support_ticket_frequency", 0)
        feedback_quality = behavior_data.get("feedback_provided", 0)
        
        # Lower support frequency is better, higher feedback quality is better
        support_score = max(0, 100 - (support_ticket_frequency * 20))
        feedback_score = feedback_quality * 100
        
        return (support_score + feedback_score) / 2
    
    def calculate_overall_engagement_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate overall engagement score"""
        communication = self.calculate_communication_engagement(behavior_data)
        participation = self.calculate_participation_engagement(behavior_data)
        interaction = self.calculate_interaction_quality(behavior_data)
        
        return (communication + participation + interaction) / 3
    
    def calculate_explicit_satisfaction(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate explicit satisfaction score"""
        nps_score = behavior_data.get("nps_score", 7)
        return (nps_score / 10) * 100
    
    def calculate_behavioral_satisfaction(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate behavioral satisfaction indicators"""
        renewal_behavior = behavior_data.get("renewal_behavior", "neutral")
        complaint_frequency = behavior_data.get("complaint_frequency", 0)
        
        # Map renewal behavior to score
        renewal_scores = {"early": 100, "on_time": 80, "late": 40, "reluctant": 20, "neutral": 60}
        renewal_score = renewal_scores.get(renewal_behavior, 60)
        
        # Lower complaint frequency is better
        complaint_score = max(0, 100 - (complaint_frequency * 25))
        
        return (renewal_score + complaint_score) / 2
    
    def calculate_loyalty_indicators(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate loyalty indicators score"""
        referral_activity = behavior_data.get("referral_activity", 0)
        testimonial_willingness = behavior_data.get("testimonial_willingness", 0)
        advocacy_actions = behavior_data.get("advocacy_actions", 0)
        
        return ((referral_activity + testimonial_willingness + advocacy_actions) / 3) * 100
    
    def calculate_overall_satisfaction_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate overall satisfaction score"""
        explicit = self.calculate_explicit_satisfaction(behavior_data)
        behavioral = self.calculate_behavioral_satisfaction(behavior_data)
        loyalty = self.calculate_loyalty_indicators(behavior_data)
        
        return (explicit * 0.5 + behavioral * 0.3 + loyalty * 0.2)
    
    def calculate_value_realization(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate value realization score"""
        monthly_value = behavior_data.get("monthly_value", 100)
        usage_intensity = behavior_data.get("usage_intensity", 0)
        
        return min(100, (monthly_value / 1000 * 60) + (usage_intensity * 40))
    
    def calculate_retention_indicators(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate retention indicators score"""
        contract_duration = behavior_data.get("contract_duration", 12)
        payment_consistency = behavior_data.get("payment_consistency", "regular")
        tenure_months = behavior_data.get("tenure_months", 0)
        
        # Longer contracts and consistent payments indicate retention
        duration_score = min(100, (contract_duration / 24) * 100)
        payment_scores = {"early": 100, "regular": 80, "late": 40, "irregular": 20}
        payment_score = payment_scores.get(payment_consistency, 60)
        tenure_score = min(100, (tenure_months / 36) * 100)
        
        return (duration_score * 0.4 + payment_score * 0.4 + tenure_score * 0.2)
    
    def calculate_growth_potential(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate growth potential score"""
        expansion_activity = behavior_data.get("expansion_activity", 0)
        user_count = behavior_data.get("user_count", 1)
        strategic_importance = behavior_data.get("strategic_importance", "medium")
        
        # Map strategic importance to score
        importance_scores = {"high": 100, "medium": 70, "low": 40}
        importance_score = importance_scores.get(strategic_importance, 70)
        
        return (expansion_activity * 40 + (user_count / 10 * 30) + importance_score * 0.3)
    
    def calculate_overall_business_score(self, behavior_data: Dict[str, Any]) -> float:
        """Calculate overall business score"""
        value_realization = self.calculate_value_realization(behavior_data)
        retention_indicators = self.calculate_retention_indicators(behavior_data)
        growth_potential = self.calculate_growth_potential(behavior_data)
        
        return (value_realization * 0.4 + retention_indicators * 0.4 + growth_potential * 0.2)
    
    def analyze_behavior_trends(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavior trends over time"""
        try:
            trends = {
                "usage_trends": self.analyze_usage_trends(behavior_data),
                "engagement_trends": self.analyze_engagement_trends(behavior_data),
                "satisfaction_trends": self.analyze_satisfaction_trends(behavior_data),
                "business_trends": self.analyze_business_trends(behavior_data)
            }
            
            # Overall trend assessment
            overall_trend = self.calculate_overall_trend(trends)
            trends["overall_assessment"] = overall_trend
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing behavior trends: {str(e)}")
            return {}
    
    def analyze_usage_trends(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze usage trends"""
        # In real implementation, would analyze historical data
        # For demo, return simulated trends
        return {
            "trend_direction": "stable",
            "trend_strength": "moderate",
            "key_changes": ["Increased feature adoption", "Consistent session duration"],
            "trend_confidence": 0.75
        }
    
    def analyze_engagement_trends(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze engagement trends"""
        return {
            "trend_direction": "improving",
            "trend_strength": "strong",
            "key_changes": ["Higher email engagement", "Increased community participation"],
            "trend_confidence": 0.8
        }
    
    def analyze_satisfaction_trends(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze satisfaction trends"""
        return {
            "trend_direction": "stable",
            "trend_strength": "weak",
            "key_changes": ["Consistent NPS scores", "Reduced complaints"],
            "trend_confidence": 0.7
        }
    
    def analyze_business_trends(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze business trends"""
        return {
            "trend_direction": "growing",
            "trend_strength": "moderate",
            "key_changes": ["Steady value growth", "Expansion opportunities"],
            "trend_confidence": 0.85
        }
    
    def calculate_overall_trend(self, trends: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall trend assessment"""
        # Simple trend combination logic
        trend_directions = {
            "usage_trends": trends.get("usage_trends", {}).get("trend_direction", "stable"),
            "engagement_trends": trends.get("engagement_trends", {}).get("trend_direction", "stable"),
            "satisfaction_trends": trends.get("satisfaction_trends", {}).get("trend_direction", "stable"),
            "business_trends": trends.get("business_trends", {}).get("trend_direction", "stable")
        }
        
        # Determine overall direction
        improving_count = sum(1 for direction in trend_directions.values() if direction == "improving")
        declining_count = sum(1 for direction in trend_directions.values() if direction == "declining")
        
        if improving_count > declining_count:
            overall_direction = "improving"
        elif declining_count > improving_count:
            overall_direction = "declining"
        else:
            overall_direction = "stable"
        
        return {
            "overall_direction": overall_direction,
            "component_trends": trend_directions,
            "trend_health": "good" if overall_direction == "improving" else "concerning" if overall_direction == "declining" else "stable"
        }
    
    def detect_behavior_anomalies(self, behavior_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect behavior anomalies"""
        try:
            anomalies = {
                "usage_anomalies": self.detect_usage_anomalies(behavior_data),
                "engagement_anomalies": self.detect_engagement_anomalies(behavior_data),
                "satisfaction_anomalies": self.detect_satisfaction_anomalies(behavior_data),
                "business_anomalies": self.detect_business_anomalies(behavior_data)
            }
            
            # Overall anomaly assessment
            total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
            overall_anomaly_score = min(100, total_anomalies * 20)
            
            anomalies["overall_assessment"] = {
                "anomaly_score": overall_anomaly_score,
                "risk_level": "high" if overall_anomaly_score > 60 else "medium" if overall_anomaly_score > 30 else "low",
                "anomaly_count": total_anomalies
            }
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error detecting behavior anomalies: {str(e)}")
            return {}
    
    def detect_usage_anomalies(self, behavior_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect usage-related anomalies"""
        anomalies = []
        
        login_frequency = behavior_data.get("login_frequency", 0)
        session_duration = behavior_data.get("session_duration_avg", 0)
        days_since_last_login = behavior_data.get("days_since_last_login", 0)
        
        # Detect unusual usage patterns
        if days_since_last_login > 30:
            anomalies.append({
                "type": "inactivity",
                "severity": "high",
                "description": "Customer hasn't logged in for over 30 days",
                "recommendation": "Immediate outreach required"
            })
        
        if login_frequency > 50:  # Unusually high usage
            anomalies.append({
                "type": "unusual_activity",
                "severity": "medium",
                "description": "Unusually high login frequency",
                "recommendation": "Review for potential data quality issues"
            })
        
        if session_duration > 480:  # 8 hours - unusually long sessions
            anomalies.append({
                "type": "unusual_session_duration",
                "severity": "medium",
                "description": "Unusually long average session duration",
                "recommendation": "Check for automated usage or session issues"
            })
        
        return anomalies
    
    def detect_engagement_anomalies(self, behavior_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect engagement-related anomalies"""
        anomalies = []
        
        email_open_rate = behavior_data.get("email_open_rate", 0.5)
        email_click_rate = behavior_data.get("email_click_rate", 0.1)
        support_ticket_frequency = behavior_data.get("support_ticket_frequency", 0)
        
        # Detect engagement anomalies
        if email_open_rate > 0.95:  # Suspiciously high open rate
            anomalies.append({
                "type": "suspicious_email_engagement",
                "severity": "medium",
                "description": "Unusually high email open rate",
                "recommendation": "Verify email engagement metrics"
            })
        
        if email_click_rate == 0 and email_open_rate > 0.7:
            anomalies.append({
                "type": "engagement_disconnect",
                "severity": "high",
                "description": "High open rate but zero clicks",
                "recommendation": "Review email content and targeting"
            })
        
        if support_ticket_frequency > 10:
            anomalies.append({
                "type": "high_support_volume",
                "severity": "high",
                "description": "Excessive support ticket volume",
                "recommendation": "Investigate underlying issues"
            })
        
        return anomalies
    
    def detect_satisfaction_anomalies(self, behavior_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect satisfaction-related anomalies"""
        anomalies = []
        
        nps_score = behavior_data.get("nps_score", 7)
        complaint_frequency = behavior_data.get("complaint_frequency", 0)
        renewal_behavior = behavior_data.get("renewal_behavior", "neutral")
        
        # Detect satisfaction anomalies
        if nps_score < 3:  # Very low NPS
            anomalies.append({
                "type": "critical_satisfaction_issue",
                "severity": "critical",
                "description": "Extremely low satisfaction score",
                "recommendation": "Immediate executive intervention required"
            })
        
        if complaint_frequency > 5:
            anomalies.append({
                "type": "high_complaint_volume",
                "severity": "high",
                "description": "High frequency of complaints",
                "recommendation": "Comprehensive issue resolution"
            })
        
        if renewal_behavior == "reluctant":
            anomalies.append({
                "type": "renewal_resistance",
                "severity": "high",
                "description": "Customer showing renewal resistance",
                "recommendation": "Retention intervention needed"
            })
        
        return anomalies
    
    def detect_business_anomalies(self, behavior_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect business-related anomalies"""
        anomalies = []
        
        monthly_value = behavior_data.get("monthly_value", 100)
        payment_consistency = behavior_data.get("payment_consistency", "regular")
        user_count = behavior_data.get("user_count", 1)
        
        # Detect business anomalies
        if payment_consistency == "irregular":
            anomalies.append({
                "type": "payment_issues",
                "severity": "high",
                "description": "Irregular payment patterns",
                "recommendation": "Financial health assessment"
            })
        
        if monthly_value < 50:  # Very low value
            anomalies.append({
                "type": "low_value_customer",
                "severity": "medium",
                "description": "Significantly below average customer value",
                "recommendation": "Value optimization analysis"
            })
        
        if user_count == 0:
            anomalies.append({
                "type": "no_active_users",
                "severity": "critical",
                "description": "No active users in customer organization",
                "recommendation": "Immediate account review"
            })
        
        return anomalies
    
    def generate_behavior_insights(self, pattern_identification: Dict[str, Any],
                                 customer_segment: Dict[str, Any],
                                 behavior_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable behavior insights"""
        try:
            insights = []
            
            # Pattern-based insights
            overall_pattern = pattern_identification.get("overall_pattern", "unknown")
            if overall_pattern == "advocate":
                insights.append({
                    "category": "advocacy_opportunity",
                    "insight": "Customer shows strong advocacy potential",
                    "description": "High satisfaction, engagement, and usage patterns indicate strong advocacy potential",
                    "actionability": "high",
                    "recommended_action": "Engage in referral and testimonial programs"
                })
            elif overall_pattern == "at_risk":
                insights.append({
                    "category": "retention_risk",
                    "insight": "Customer shows signs of potential churn",
                    "description": "Low engagement and usage patterns suggest churn risk",
                    "actionability": "high",
                    "recommended_action": "Implement retention intervention strategy"
                })
            
            # Segment-based insights
            segment = customer_segment.get("segment", "unknown")
            if segment == "champions":
                insights.append({
                    "category": "expansion_opportunity",
                    "insight": "High-value champion customer with expansion potential",
                    "description": "Champion segment customers often have expansion opportunities",
                    "actionability": "high",
                    "recommended_action": "Present expansion and upsell opportunities"
                })
            
            # Metrics-based insights
            composite_scores = behavior_metrics.get("composite_scores", {})
            health_score = composite_scores.get("customer_health_score", 50)
            
            if health_score < 40:
                insights.append({
                    "category": "health_improvement",
                    "insight": "Customer health score needs improvement",
                    "description": f"Customer health score of {health_score:.1f} indicates intervention needed",
                    "actionability": "high",
                    "recommended_action": "Develop health improvement plan"
                })
            
            # Trend-based insights
            if pattern_identification.get("pattern_confidence", 0) > 0.8:
                insights.append({
                    "category": "confidence_high",
                    "insight": "High confidence in behavioral assessment",
                    "description": "Strong data quality enables confident behavioral insights",
                    "actionability": "medium",
                    "recommended_action": "Use insights for targeted optimization"
                })
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            self.logger.error(f"Error generating behavior insights: {str(e)}")
            return []
    
    def generate_behavior_recommendations(self, customer_segment: Dict[str, Any],
                                        behavior_insights: List[Dict[str, Any]],
                                        behavior_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate behavior-based recommendations"""
        try:
            recommendations = []
            
            # Segment-specific recommendations
            segment = customer_segment.get("segment", "unknown")
            segment_actions = customer_segment.get("recommended_actions", [])
            
            for action in segment_actions:
                recommendations.append({
                    "type": "segment_specific",
                    "recommendation": action,
                    "priority": "high" if segment in ["champions", "at_risk"] else "medium",
                    "rationale": f"Based on {segment} segment characteristics"
                })
            
            # Insight-based recommendations
            for insight in behavior_insights:
                if insight.get("actionability") == "high":
                    recommendations.append({
                        "type": "insight_driven",
                        "recommendation": insight.get("recommended_action", ""),
                        "priority": "high",
                        "rationale": f"Driven by {insight.get('category')} insight"
                    })
            
            # Usage optimization recommendations
            login_frequency = behavior_data.get("login_frequency", 0)
            feature_adoption_rate = behavior_data.get("feature_adoption_rate", 0)
            
            if login_frequency < 7:
                recommendations.append({
                    "type": "usage_optimization",
                    "recommendation": "Increase usage frequency through engagement campaigns",
                    "priority": "medium",
                    "rationale": "Low login frequency suggests usage optimization opportunity"
                })
            
            if feature_adoption_rate < 0.5:
                recommendations.append({
                    "type": "feature_adoption",
                    "recommendation": "Promote underutilized features through targeted campaigns",
                    "priority": "medium",
                    "rationale": "Feature adoption rate below optimal threshold"
                })
            
            return recommendations[:8]  # Return top 8 recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating behavior recommendations: {str(e)}")
            return []
    
    def calculate_analysis_confidence(self, behavior_features: Dict[str, Any]) -> float:
        """Calculate confidence in behavior analysis"""
        try:
            confidence_factors = []
            
            # Data completeness
            feature_categories = ["usage_features", "engagement_features", "satisfaction_features", "business_features"]
            completeness_scores = []
            
            for category in feature_categories:
                category_features = behavior_features.get(category, {})
                category_completeness = len(category_features) / 8  # Assuming 8 features per category
                completeness_scores.append(min(category_completeness, 1.0))
            
            confidence_factors.append(sum(completeness_scores) / len(completeness_scores))
            
            # Feature consistency
            usage_features = behavior_features.get("usage_features", {})
            if (usage_features.get("login_frequency", 0) > 0 and 
                usage_features.get("session_duration_avg", 0) > 0):
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.5)
            
            return sum(confidence_factors) / len(confidence_factors)
            
        except Exception as e:
            self.logger.error(f"Error calculating analysis confidence: {str(e)}")
            return 0.5
    
    def get_status(self) -> Dict[str, Any]:
        """Get behavior analytics engine status"""
        return {
            "status": "operational",
            "behavior_metrics_categories": len(self.behavior_metrics),
            "behavior_patterns_loaded": len(self.behavior_patterns),
            "analytics_rules_active": len(self.analytics_rules.get("pattern_recognition_rules", {})),
            "ai_models_loaded": {
                "behavior_clustering": self.behavior_clustering_model is not None,
                "pattern_recognition": self.pattern_recognition_model is not None,
                "feature_scaler": self.feature_scaler is not None
            },
            "supported_patterns": {
                "usage_patterns": len(self.behavior_patterns.get("usage_patterns", {})),
                "engagement_patterns": len(self.behavior_patterns.get("engagement_patterns", {})),
                "satisfaction_patterns": len(self.behavior_patterns.get("satisfaction_patterns", {})),
                "business_patterns": len(self.behavior_patterns.get("business_patterns", {}))
            },
            "last_update": datetime.now().isoformat()
        }
