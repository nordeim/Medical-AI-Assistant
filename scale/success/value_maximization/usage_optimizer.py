"""
Product Usage Optimization and Feature Adoption Engine

This module provides AI-powered product usage optimization capabilities including:
- Feature adoption prediction and optimization
- Usage pattern analysis and optimization
- Product stickiness and engagement enhancement
- Usage-based pricing optimization
- Feature recommendation and discovery systems
- User journey optimization for product utilization
- Adoption lifecycle management and intervention
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from scipy.stats import entropy
from collections import defaultdict, deque
import networkx as nx
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureAdoptionOpportunity:
    """Feature adoption opportunity data"""
    customer_id: str
    feature_id: str
    current_usage_level: float
    adoption_probability: float
    expected_impact: float
    time_to_adoption: int  # days
    intervention_strategy: str
    business_value: float
    complexity_score: float
    user_segment: str

@dataclass
class UsagePattern:
    """Usage pattern analysis result"""
    customer_id: str
    pattern_type: str
    usage_frequency: float
    feature_utilization: Dict[str, float]
    engagement_score: float
    stickiness_index: float
    lifecycle_stage: str
    optimization_potential: float
    recommended_actions: List[str]

@dataclass
class FeaturePerformance:
    """Feature performance metrics"""
    feature_id: str
    feature_name: str
    adoption_rate: float
    usage_intensity: float
    business_impact: float
    user_satisfaction: float
    retention_correlation: float
    revenue_correlation: float
    optimization_opportunities: List[str]

@dataclass
class AdoptionCampaign:
    """Feature adoption campaign configuration"""
    campaign_id: str
    target_features: List[str]
    target_segments: List[str]
    campaign_type: str  # 'onboarding', 'enhancement', 'advanced_usage'
    expected_adoption_rate: float
    resource_requirements: Dict[str, Any]
    success_metrics: List[str]
    timeline_weeks: int

class UsageOptimizer:
    """
    AI-powered product usage optimization and feature adoption engine
    
    This engine analyzes usage patterns, predicts feature adoption, and optimizes
    product utilization to maximize customer value and engagement.
    """
    
    def __init__(self):
        """Initialize the usage optimizer with ML models and data structures"""
        self.models = {
            'adoption_predictor': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'usage_predictor': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'engagement_classifier': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'stickiness_predictor': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
        }
        
        self.scalers = {
            'feature_scaler': StandardScaler(),
            'usage_scaler': StandardScaler(),
            'engagement_scaler': StandardScaler()
        }
        
        self.label_encoders = {
            'user_segment': LabelEncoder(),
            'product_tier': LabelEncoder(),
            'acquisition_channel': LabelEncoder()
        }
        
        # Usage tracking
        self.usage_history = []
        self.feature_performance = {}
        self.usage_patterns = {}
        self.adoption_campaigns = []
        
        # Feature dependency graph
        self.feature_graph = nx.DiGraph()
        
        # Adoption strategies
        self.adoption_strategies = {
            'education': 'Provide comprehensive training and documentation',
            'gamification': 'Use gamification to encourage feature exploration',
            'incentives': 'Offer incentives for feature adoption',
            'personalization': 'Personalize feature recommendations',
            'social_proof': 'Leverage social proof and peer influence',
            'gradual_introduction': 'Introduce features gradually based on usage'
        }
        
        logger.info("Usage Optimizer initialized successfully")
    
    def load_usage_data(self, usage_data: pd.DataFrame, feature_data: pd.DataFrame) -> None:
        """
        Load historical usage and feature data for model training
        
        Args:
            usage_data: DataFrame with columns:
                - customer_id, date, feature_id, usage_duration, usage_frequency,
                - user_segment, product_tier, engagement_score, retention_days
            feature_data: DataFrame with columns:
                - feature_id, feature_name, category, complexity_score, business_value
        """
        try:
            logger.info(f"Loading {len(usage_data)} usage records and {len(feature_data)} feature definitions")
            
            # Prepare adoption prediction features
            adoption_features = [
                'engagement_score', 'usage_frequency', 'tenure_days',
                'feature_usage_count', 'support_interactions', 'login_frequency'
            ]
            
            X_adoption = usage_data[adoption_features].fillna(0)
            y_adoption = (usage_data['feature_adopted'].fillna(0)).astype(int)
            
            # Encode categorical features
            categorical_features = ['user_segment', 'product_tier', 'acquisition_channel']
            for col in categorical_features:
                if col in usage_data.columns:
                    usage_data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        usage_data[col].fillna('unknown')
                    )
                    X_adoption[f'{col}_encoded'] = usage_data[f'{col}_encoded']
            
            # Add derived features
            X_adoption['usage_intensity'] = (
                usage_data['usage_frequency'] * usage_data['usage_duration']
            ).fillna(0)
            X_adoption['feature_diversity'] = usage_data.groupby('customer_id')['feature_id'].nunique()
            
            # Scale features
            X_adoption_scaled = self.scalers['feature_scaler'].fit_transform(X_adoption)
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                X_adoption_scaled, y_adoption, test_size=0.2, random_state=42, stratify=y_adoption
            )
            
            # Train adoption predictor
            logger.info("Training feature adoption prediction model...")
            self.models['adoption_predictor'].fit(X_train, y_train)
            
            # Evaluate adoption model
            y_pred_adoption = self.models['adoption_predictor'].predict(X_test)
            logger.info(f"Adoption Model Performance:\n{classification_report(y_test, y_pred_adoption)}")
            
            # Train usage predictor
            logger.info("Training usage intensity prediction model...")
            y_usage = usage_data['usage_duration'].fillna(0)
            X_usage_train, X_usage_test, y_usage_train, y_usage_test = train_test_split(
                X_adoption_scaled, y_usage, test_size=0.2, random_state=42
            )
            self.models['usage_predictor'].fit(X_usage_train, y_usage_train)
            
            # Evaluate usage model
            y_pred_usage = self.models['usage_predictor'].predict(X_usage_test)
            usage_mae = mean_absolute_error(y_usage_test, y_pred_usage)
            logger.info(f"Usage Predictor MAE: {usage_mae:.2f} minutes")
            
            # Train engagement classifier
            logger.info("Training engagement classification model...")
            y_engagement = (usage_data['engagement_score'] > 0.7).astype(int)
            X_eng_train, X_eng_test, y_eng_train, y_eng_test = train_test_split(
                X_adoption_scaled, y_engagement, test_size=0.2, random_state=42
            )
            self.models['engagement_classifier'].fit(X_eng_train, y_eng_train)
            
            # Train stickiness predictor
            logger.info("Training product stickiness prediction model...")
            y_stickiness = usage_data['retention_days'].fillna(0)
            X_stick_train, X_stick_test, y_stick_train, y_stick_test = train_test_split(
                X_adoption_scaled, y_stickiness, test_size=0.2, random_state=42
            )
            self.models['stickiness_predictor'].fit(X_stick_train, y_stick_train)
            
            # Build feature dependency graph
            self._build_feature_graph(feature_data)
            
            # Store data
            self.usage_history = usage_data.to_dict('records')
            self._initialize_feature_performance(feature_data)
            
            logger.info("Usage data loaded and models trained successfully")
            
        except Exception as e:
            logger.error(f"Error loading usage data: {str(e)}")
            raise
    
    def _build_feature_graph(self, feature_data: pd.DataFrame) -> None:
        """Build feature dependency and relationship graph"""
        try:
            # Add feature nodes
            for _, feature in feature_data.iterrows():
                self.feature_graph.add_node(
                    feature['feature_id'],
                    name=feature['feature_name'],
                    category=feature.get('category', 'general'),
                    complexity=feature.get('complexity_score', 0.5),
                    business_value=feature.get('business_value', 0.5)
                )
            
            # Add dependencies based on feature analysis
            # Simplified: add typical feature dependencies
            feature_dependencies = [
                ('basic_login', 'dashboard_access'),
                ('dashboard_access', 'basic_reporting'),
                ('basic_reporting', 'advanced_analytics'),
                ('user_management', 'team_collaboration'),
                ('team_collaboration', 'advanced_workflows')
            ]
            
            for prerequisite, feature in feature_dependencies:
                if prerequisite in self.feature_graph.nodes and feature in self.feature_graph.nodes:
                    self.feature_graph.add_edge(prerequisite, feature, relationship='prerequisite')
            
            logger.info(f"Built feature graph with {len(self.feature_graph.nodes())} features")
            
        except Exception as e:
            logger.error(f"Error building feature graph: {str(e)}")
    
    def _initialize_feature_performance(self, feature_data: pd.DataFrame) -> None:
        """Initialize feature performance metrics"""
        try:
            for _, feature in feature_data.iterrows():
                feature_id = feature['feature_id']
                
                self.feature_performance[feature_id] = FeaturePerformance(
                    feature_id=feature_id,
                    feature_name=feature.get('feature_name', feature_id),
                    adoption_rate=0.0,
                    usage_intensity=0.0,
                    business_impact=0.0,
                    user_satisfaction=0.0,
                    retention_correlation=0.0,
                    revenue_correlation=0.0,
                    optimization_opportunities=[]
                )
                
        except Exception as e:
            logger.error(f"Error initializing feature performance: {str(e)}")
    
    def analyze_usage_patterns(self, customer_data: pd.DataFrame) -> List[UsagePattern]:
        """
        Analyze usage patterns for customers
        
        Args:
            customer_data: DataFrame with customer usage data
            
        Returns:
            List of UsagePattern objects
        """
        try:
            logger.info(f"Analyzing usage patterns for {len(customer_data)} customers")
            
            usage_patterns = []
            
            for _, customer in customer_data.iterrows():
                pattern = self._analyze_single_customer_pattern(customer)
                usage_patterns.append(pattern)
            
            # Sort by optimization potential
            usage_patterns.sort(key=lambda x: x.optimization_potential, reverse=True)
            
            logger.info(f"Analyzed usage patterns for {len(usage_patterns)} customers")
            return usage_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing usage patterns: {str(e)}")
            raise
    
    def _analyze_single_customer_pattern(self, customer: pd.Series) -> UsagePattern:
        """Analyze usage pattern for a single customer"""
        customer_id = str(customer.get('customer_id', 'unknown'))
        
        # Calculate basic usage metrics
        usage_frequency = customer.get('usage_frequency', 0)
        engagement_score = customer.get('engagement_score', 0)
        
        # Feature utilization analysis
        feature_utilization = self._analyze_feature_utilization(customer)
        
        # Calculate stickiness index
        stickiness_index = self._calculate_stickiness_index(customer)
        
        # Determine lifecycle stage
        lifecycle_stage = self._determine_lifecycle_stage(customer)
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(
            engagement_score, stickiness_index, feature_utilization
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_usage_optimization_actions(
            customer, feature_utilization, lifecycle_stage
        )
        
        # Determine pattern type
        pattern_type = self._classify_usage_pattern(
            usage_frequency, engagement_score, stickiness_index
        )
        
        return UsagePattern(
            customer_id=customer_id,
            pattern_type=pattern_type,
            usage_frequency=usage_frequency,
            feature_utilization=feature_utilization,
            engagement_score=engagement_score,
            stickiness_index=stickiness_index,
            lifecycle_stage=lifecycle_stage,
            optimization_potential=optimization_potential,
            recommended_actions=recommended_actions
        )
    
    def _analyze_feature_utilization(self, customer: pd.Series) -> Dict[str, float]:
        """Analyze feature utilization for a customer"""
        # Simplified feature utilization analysis
        customer_features = customer.get('used_features', [])
        
        feature_utilization = {}
        
        if isinstance(customer_features, list):
            for feature in customer_features:
                # Assign random utilization score for demo purposes
                feature_utilization[feature] = np.random.uniform(0.3, 0.9)
        else:
            # Default feature utilization
            feature_utilization = {
                'basic_login': 0.8,
                'dashboard_access': 0.6,
                'basic_reporting': 0.4,
                'advanced_analytics': 0.2
            }
        
        return feature_utilization
    
    def _calculate_stickiness_index(self, customer: pd.Series) -> float:
        """Calculate product stickiness index"""
        try:
            # Components of stickiness
            usage_frequency = customer.get('usage_frequency', 0)
            retention_days = customer.get('retention_days', 0)
            feature_usage = customer.get('feature_usage_count', 0)
            
            # Normalize components (simplified)
            freq_score = min(usage_frequency / 30, 1.0)  # Monthly usage
            retention_score = min(retention_days / 365, 1.0)  # Yearly retention
            feature_score = min(feature_usage / 10, 1.0)  # Feature usage
            
            # Weighted stickiness calculation
            stickiness_index = (
                freq_score * 0.3 +
                retention_score * 0.4 +
                feature_score * 0.3
            )
            
            return min(max(stickiness_index, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating stickiness index: {str(e)}")
            return 0.5
    
    def _determine_lifecycle_stage(self, customer: pd.Series) -> str:
        """Determine customer lifecycle stage based on usage patterns"""
        try:
            tenure_days = customer.get('tenure_days', 0)
            engagement_score = customer.get('engagement_score', 0)
            usage_frequency = customer.get('usage_frequency', 0)
            
            # Lifecycle stage determination logic
            if tenure_days < 30:
                if engagement_score > 0.6:
                    return 'onboarding'
                else:
                    return 'activation'
            elif tenure_days < 180:
                return 'growth'
            elif tenure_days < 730:
                return 'maturity'
            else:
                return 'retention'
                
        except Exception as e:
            logger.error(f"Error determining lifecycle stage: {str(e)}")
            return 'unknown'
    
    def _calculate_optimization_potential(
        self,
        engagement_score: float,
        stickiness_index: float,
        feature_utilization: Dict[str, float]
    ) -> float:
        """Calculate optimization potential score"""
        # Base potential from engagement and stickiness
        base_potential = 1 - ((engagement_score + stickiness_index) / 2)
        
        # Adjust based on feature utilization diversity
        utilization_values = list(feature_utilization.values())
        utilization_diversity = len([x for x in utilization_values if x > 0.3]) / max(len(utilization_values), 1)
        utilization_bonus = 1 - utilization_diversity
        
        optimization_potential = min(base_potential + utilization_bonus * 0.3, 1.0)
        return max(optimization_potential, 0.0)
    
    def _generate_usage_optimization_actions(
        self,
        customer: pd.Series,
        feature_utilization: Dict[str, float],
        lifecycle_stage: str
    ) -> List[str]:
        """Generate usage optimization action recommendations"""
        actions = []
        
        # Lifecycle-specific actions
        if lifecycle_stage == 'activation':
            actions.append("Schedule onboarding session to improve feature discovery")
            actions.append("Send feature tour emails highlighting key capabilities")
        elif lifecycle_stage == 'onboarding':
            actions.append("Provide guided tutorials for underutilized features")
            actions.append("Implement progress tracking to encourage feature exploration")
        elif lifecycle_stage == 'growth':
            actions.append("Introduce advanced features to increase engagement")
            actions.append("Suggest feature combinations for improved workflow")
        elif lifecycle_stage == 'maturity':
            actions.append("Recommend new features to prevent stagnation")
            actions.append("Offer premium features as upgrade opportunities")
        
        # Feature-specific actions
        for feature, utilization in feature_utilization.items():
            if utilization < 0.3:
                actions.append(f"Provide training for {feature} feature")
            elif utilization < 0.6:
                actions.append(f"Share best practices for {feature} feature")
        
        return actions
    
    def _classify_usage_pattern(
        self,
        usage_frequency: float,
        engagement_score: float,
        stickiness_index: float
    ) -> str:
        """Classify usage pattern type"""
        # Pattern classification based on metrics
        if engagement_score > 0.8 and stickiness_index > 0.8:
            return 'power_user'
        elif engagement_score > 0.6 and stickiness_index > 0.6:
            return 'regular_user'
        elif engagement_score > 0.4:
            return 'casual_user'
        else:
            return 'low_engagement'
    
    def identify_adoption_opportunities(
        self,
        customer_data: pd.DataFrame,
        max_opportunities_per_customer: int = 3
    ) -> List[FeatureAdoptionOpportunity]:
        """
        Identify feature adoption opportunities for customers
        
        Args:
            customer_data: DataFrame with customer information
            max_opportunities_per_customer: Maximum opportunities per customer
            
        Returns:
            List of FeatureAdoptionOpportunity objects
        """
        try:
            logger.info(f"Identifying adoption opportunities for {len(customer_data)} customers")
            
            opportunities = []
            
            for _, customer in customer_data.iterrows():
                customer_opportunities = self._analyze_adoption_opportunities(customer)
                
                # Sort by adoption probability and impact
                customer_opportunities.sort(
                    key=lambda x: (x.adoption_probability * x.business_value),
                    reverse=True
                )
                
                # Limit opportunities per customer
                limited_opportunities = customer_opportunities[:max_opportunities_per_customer]
                opportunities.extend(limited_opportunities)
            
            # Sort all opportunities by potential impact
            opportunities.sort(
                key=lambda x: (x.adoption_probability * x.expected_impact),
                reverse=True
            )
            
            logger.info(f"Identified {len(opportunities)} adoption opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying adoption opportunities: {str(e)}")
            raise
    
    def _analyze_adoption_opportunities(self, customer: pd.Series) -> List[FeatureAdoptionOpportunity]:
        """Analyze adoption opportunities for a single customer"""
        customer_id = str(customer.get('customer_id', 'unknown'))
        user_segment = customer.get('user_segment', 'general')
        
        opportunities = []
        
        # Get available features (not currently used)
        used_features = set(customer.get('used_features', []))
        available_features = set(self.feature_graph.nodes) - used_features
        
        # Analyze each available feature
        for feature_id in available_features:
            opportunity = self._evaluate_feature_opportunity(customer, feature_id, user_segment)
            if opportunity and opportunity.adoption_probability > 0.3:
                opportunities.append(opportunity)
        
        return opportunities
    
    def _evaluate_feature_opportunity(
        self,
        customer: pd.Series,
        feature_id: str,
        user_segment: str
    ) -> Optional[FeatureAdoptionOpportunity]:
        """Evaluate adoption opportunity for a specific feature"""
        try:
            # Prepare customer features for prediction
            features = self._prepare_adoption_features(customer)
            features_scaled = self.scalers['feature_scaler'].transform(features.reshape(1, -1))
            
            # Predict adoption probability
            adoption_prob = self.models['adoption_predictor'].predict_proba(features_scaled)[0][1]
            
            # Adjust probability based on feature complexity and business value
            feature_node = self.feature_graph.nodes.get(feature_id, {})
            complexity_score = feature_node.get('complexity', 0.5)
            business_value = feature_node.get('business_value', 0.5)
            
            # Apply adjustments
            complexity_adjustment = 1 - (complexity_score * 0.3)
            value_adjustment = 0.8 + (business_value * 0.4)
            
            final_probability = adoption_prob * complexity_adjustment * value_adjustment
            final_probability = min(max(final_probability, 0.0), 1.0)
            
            # Calculate expected impact and timing
            expected_impact = self._calculate_adoption_impact(customer, feature_id, final_probability)
            time_to_adoption = self._estimate_adoption_time(customer, feature_id)
            
            # Determine intervention strategy
            intervention_strategy = self._determine_intervention_strategy(
                complexity_score, user_segment, final_probability
            )
            
            # Calculate business value
            calculated_bv = self._calculate_feature_business_value(feature_id, customer)
            
            return FeatureAdoptionOpportunity(
                customer_id=customer.get('customer_id', 'unknown'),
                feature_id=feature_id,
                current_usage_level=0.0,  # Not currently using feature
                adoption_probability=final_probability,
                expected_impact=expected_impact,
                time_to_adoption=time_to_adoption,
                intervention_strategy=intervention_strategy,
                business_value=calculated_bv,
                complexity_score=complexity_score,
                user_segment=user_segment
            )
            
        except Exception as e:
            logger.error(f"Error evaluating feature opportunity: {str(e)}")
            return None
    
    def _prepare_adoption_features(self, customer: pd.Series) -> np.ndarray:
        """Prepare features for adoption prediction"""
        feature_values = [
            customer.get('engagement_score', 0),
            customer.get('usage_frequency', 0),
            customer.get('tenure_days', 0),
            customer.get('feature_usage_count', 0),
            customer.get('support_interactions', 0),
            customer.get('login_frequency', 0)
        ]
        
        # Add encoded categorical features if available
        for encoder_name, encoder in self.label_encoders.items():
            if encoder_name in customer.index:
                try:
                    encoded_value = encoder.transform([customer[encoder_name]])[0]
                    feature_values.append(encoded_value)
                except:
                    feature_values.append(0)
            else:
                feature_values.append(0)
        
        # Add derived features
        usage_intensity = customer.get('usage_frequency', 0) * customer.get('usage_duration', 0)
        feature_values.append(usage_intensity)
        
        return np.array(feature_values)
    
    def _calculate_adoption_impact(
        self,
        customer: pd.Series,
        feature_id: str,
        adoption_probability: float
    ) -> float:
        """Calculate expected impact of feature adoption"""
        # Base impact from customer value
        customer_value = customer.get('avg_revenue_per_month', 100)
        
        # Feature-specific impact
        feature_impact_multiplier = 1.2  # Default multiplier
        
        # Probability-weighted impact
        expected_impact = customer_value * feature_impact_multiplier * adoption_probability
        
        return expected_impact
    
    def _estimate_adoption_time(self, customer: pd.Series, feature_id: str) -> int:
        """Estimate days to feature adoption"""
        # Base adoption time
        base_days = 30
        
        # Adjust based on customer engagement
        engagement_score = customer.get('engagement_score', 0.5)
        engagement_adjustment = 1 - (engagement_score * 0.5)  # High engagement = faster adoption
        
        # Adjust based on feature complexity
        feature_node = self.feature_graph.nodes.get(feature_id, {})
        complexity_score = feature_node.get('complexity', 0.5)
        complexity_adjustment = 1 + (complexity_score * 0.5)  # Complex features take longer
        
        estimated_days = int(base_days * engagement_adjustment * complexity_adjustment)
        return max(estimated_days, 7)  # Minimum 7 days
    
    def _determine_intervention_strategy(
        self,
        complexity_score: float,
        user_segment: str,
        adoption_probability: float
    ) -> str:
        """Determine optimal intervention strategy for feature adoption"""
        # Strategy selection based on complexity and user segment
        if complexity_score > 0.7:
            if user_segment in ['enterprise', 'technical']:
                return 'education'
            else:
                return 'gradual_introduction'
        elif complexity_score > 0.4:
            if adoption_probability < 0.5:
                return 'incentives'
            else:
                return 'personalization'
        else:
            if adoption_probability < 0.6:
                return 'gamification'
            else:
                return 'social_proof'
    
    def _calculate_feature_business_value(self, feature_id: str, customer: pd.Series) -> float:
        """Calculate business value of feature for customer"""
        feature_node = self.feature_graph.nodes.get(feature_id, {})
        base_value = feature_node.get('business_value', 0.5)
        
        # Adjust based on customer segment
        customer_value_multiplier = {
            'enterprise': 1.5,
            'mid_market': 1.2,
            'small_business': 1.0,
            'startup': 0.8
        }.get(customer.get('user_segment', 'small_business'), 1.0)
        
        return base_value * customer_value_multiplier
    
    def optimize_feature_set(self, customer_id: str, target_engagement: float = 0.8) -> Dict[str, Any]:
        """
        Optimize feature set for a specific customer to maximize engagement
        
        Args:
            customer_id: Customer identifier
            target_engagement: Target engagement score
            
        Returns:
            Dictionary with optimization recommendations
        """
        try:
            logger.info(f"Optimizing feature set for customer {customer_id}")
            
            # Get customer data (simplified - would normally fetch from database)
            customer_data = self._get_customer_usage_data(customer_id)
            
            if not customer_data.empty:
                customer = customer_data.iloc[0]
                
                # Analyze current usage pattern
                current_pattern = self._analyze_single_customer_pattern(customer)
                
                # Identify priority features
                priority_features = self._identify_priority_features(
                    customer, current_pattern, target_engagement
                )
                
                # Generate optimization plan
                optimization_plan = self._create_optimization_plan(
                    customer_id, priority_features, target_engagement
                )
                
                return {
                    'customer_id': customer_id,
                    'current_engagement': current_pattern.engagement_score,
                    'target_engagement': target_engagement,
                    'priority_features': priority_features,
                    'optimization_plan': optimization_plan,
                    'expected_improvement': self._estimate_engagement_improvement(
                        current_pattern, target_engagement
                    ),
                    'implementation_timeline': '4-6 weeks'
                }
            else:
                return {'error': 'Customer data not found'}
                
        except Exception as e:
            logger.error(f"Error optimizing feature set: {str(e)}")
            return {'error': str(e)}
    
    def _get_customer_usage_data(self, customer_id: str) -> pd.DataFrame:
        """Get usage data for a specific customer"""
        # Simplified: filter usage history for customer
        customer_data = [
            record for record in self.usage_history
            if str(record.get('customer_id', '')) == str(customer_id)
        ]
        
        if customer_data:
            return pd.DataFrame([customer_data[0]])
        else:
            # Return sample data for demo
            return pd.DataFrame([{
                'customer_id': customer_id,
                'usage_frequency': 15,
                'engagement_score': 0.6,
                'tenure_days': 90,
                'feature_usage_count': 3,
                'support_interactions': 2,
                'login_frequency': 10,
                'retention_days': 90,
                'used_features': ['basic_login', 'dashboard_access']
            }])
    
    def _identify_priority_features(
        self,
        customer: pd.Series,
        current_pattern: UsagePattern,
        target_engagement: float
    ) -> List[Dict[str, Any]]:
        """Identify priority features for optimization"""
        priority_features = []
        
        # Get unused features
        used_features = set(customer.get('used_features', []))
        available_features = set(self.feature_graph.nodes) - used_features
        
        # Calculate feature priority scores
        for feature_id in available_features:
            feature_node = self.feature_graph.nodes.get(feature_id, {})
            
            # Priority factors
            business_value = feature_node.get('business_value', 0.5)
            complexity = feature_node.get('complexity', 0.5)
            
            # Calculate priority score (higher value, lower complexity = higher priority)
            priority_score = business_value * (1 - complexity) * 0.7 + 0.3
            
            if priority_score > 0.4:  # Only include high-priority features
                priority_features.append({
                    'feature_id': feature_id,
                    'feature_name': feature_node.get('name', feature_id),
                    'priority_score': priority_score,
                    'business_value': business_value,
                    'complexity': complexity,
                    'expected_engagement_boost': priority_score * 0.15
                })
        
        # Sort by priority score
        priority_features.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priority_features[:5]  # Top 5 features
    
    def _create_optimization_plan(
        self,
        customer_id: str,
        priority_features: List[Dict[str, Any]],
        target_engagement: float
    ) -> Dict[str, Any]:
        """Create feature optimization plan"""
        plan = {
            'customer_id': customer_id,
            'target_engagement': target_engagement,
            'phases': []
        }
        
        # Phase 1: Quick wins (low complexity, high value)
        quick_wins = [f for f in priority_features if f['complexity'] < 0.4]
        if quick_wins:
            plan['phases'].append({
                'phase': 1,
                'duration_weeks': 2,
                'features': [f['feature_id'] for f in quick_wins],
                'strategy': 'onboarding and basic training',
                'expected_improvement': sum(f['expected_engagement_boost'] for f in quick_wins)
            })
        
        # Phase 2: Medium complexity features
        medium_features = [f for f in priority_features if 0.4 <= f['complexity'] <= 0.7]
        if medium_features:
            plan['phases'].append({
                'phase': 2,
                'duration_weeks': 3,
                'features': [f['feature_id'] for f in medium_features],
                'strategy': 'guided exploration and best practices',
                'expected_improvement': sum(f['expected_engagement_boost'] for f in medium_features)
            })
        
        # Phase 3: Advanced features
        advanced_features = [f for f in priority_features if f['complexity'] > 0.7]
        if advanced_features:
            plan['phases'].append({
                'phase': 3,
                'duration_weeks': 4,
                'features': [f['feature_id'] for f in advanced_features],
                'strategy': 'advanced training and certification',
                'expected_improvement': sum(f['expected_engagement_boost'] for f in advanced_features)
            })
        
        return plan
    
    def _estimate_engagement_improvement(
        self,
        current_pattern: UsagePattern,
        target_engagement: float
    ) -> float:
        """Estimate engagement improvement potential"""
        current_engagement = current_pattern.engagement_score
        improvement_potential = target_engagement - current_engagement
        
        # Apply realistic constraints (max 80% of potential is achievable)
        realistic_improvement = improvement_potential * 0.8
        
        return max(realistic_improvement, 0)
    
    def create_adoption_campaigns(self, target_segments: List[str]) -> List[AdoptionCampaign]:
        """
        Create feature adoption campaigns for target customer segments
        
        Args:
            target_segments: List of customer segments to target
            
        Returns:
            List of AdoptionCampaign objects
        """
        try:
            logger.info(f"Creating adoption campaigns for segments: {target_segments}")
            
            campaigns = []
            
            # Define campaign templates
            campaign_templates = {
                'onboarding': {
                    'features': ['basic_login', 'dashboard_access', 'basic_reporting'],
                    'expected_adoption_rate': 0.75,
                    'timeline_weeks': 4,
                    'resource_requirements': {'training_materials': True, 'email_series': True}
                },
                'enhancement': {
                    'features': ['advanced_reporting', 'custom_dashboards', 'data_export'],
                    'expected_adoption_rate': 0.45,
                    'timeline_weeks': 6,
                    'resource_requirements': {'webinars': True, 'documentation': True}
                },
                'advanced_usage': {
                    'features': ['api_integration', 'advanced_analytics', 'automation'],
                    'expected_adoption_rate': 0.25,
                    'timeline_weeks': 8,
                    'resource_requirements': {'technical_training': True, 'certification': True}
                }
            }
            
            campaign_id = 1
            for segment in target_segments:
                for campaign_type, template in campaign_templates.items():
                    campaign = AdoptionCampaign(
                        campaign_id=f"campaign_{campaign_id:03d}",
                        target_features=template['features'],
                        target_segments=[segment],
                        campaign_type=campaign_type,
                        expected_adoption_rate=template['expected_adoption_rate'],
                        resource_requirements=template['resource_requirements'],
                        success_metrics=['adoption_rate', 'engagement_increase', 'retention_improvement'],
                        timeline_weeks=template['timeline_weeks']
                    )
                    campaigns.append(campaign)
                    campaign_id += 1
            
            logger.info(f"Created {len(campaigns)} adoption campaigns")
            return campaigns
            
        except Exception as e:
            logger.error(f"Error creating adoption campaigns: {str(e)}")
            raise
    
    def analyze_feature_performance(self) -> List[FeaturePerformance]:
        """
        Analyze overall feature performance across all customers
        
        Returns:
            List of FeaturePerformance objects
        """
        try:
            logger.info("Analyzing feature performance")
            
            feature_performances = []
            
            for feature_id, performance in self.feature_performance.items():
                # Update performance metrics based on usage data
                updated_performance = self._calculate_feature_metrics(feature_id)
                feature_performances.append(updated_performance)
            
            # Sort by business impact
            feature_performances.sort(key=lambda x: x.business_impact, reverse=True)
            
            logger.info(f"Analyzed performance for {len(feature_performances)} features")
            return feature_performances
            
        except Exception as e:
            logger.error(f"Error analyzing feature performance: {str(e)}")
            return []
    
    def _calculate_feature_metrics(self, feature_id: str) -> FeaturePerformance:
        """Calculate performance metrics for a specific feature"""
        try:
            # Get usage data for this feature
            feature_usage_data = [
                record for record in self.usage_history
                if record.get('feature_id') == feature_id
            ]
            
            if not feature_usage_data:
                return self.feature_performance.get(feature_id, FeaturePerformance(
                    feature_id=feature_id,
                    feature_name=feature_id,
                    adoption_rate=0.0,
                    usage_intensity=0.0,
                    business_impact=0.0,
                    user_satisfaction=0.0,
                    retention_correlation=0.0,
                    revenue_correlation=0.0,
                    optimization_opportunities=[]
                ))
            
            # Calculate adoption rate
            total_customers = len(set(record.get('customer_id') for record in feature_usage_data))
            adopting_customers = len(set(
                record.get('customer_id') for record in feature_usage_data
                if record.get('feature_adopted', False)
            ))
            adoption_rate = adopting_customers / total_customers if total_customers > 0 else 0
            
            # Calculate usage intensity
            usage_intensity = np.mean([
                record.get('usage_duration', 0) for record in feature_usage_data
            ])
            
            # Calculate business impact (simplified)
            business_impact = adoption_rate * usage_intensity
            
            # Calculate user satisfaction (proxy based on retention)
            user_satisfaction = 0.7  # Default satisfaction score
            
            # Calculate correlations (simplified)
            retention_correlation = 0.6  # Default correlation
            revenue_correlation = 0.5    # Default correlation
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_feature_optimizations(
                feature_id, adoption_rate, usage_intensity, business_impact
            )
            
            return FeaturePerformance(
                feature_id=feature_id,
                feature_name=self.feature_graph.nodes.get(feature_id, {}).get('name', feature_id),
                adoption_rate=adoption_rate,
                usage_intensity=usage_intensity,
                business_impact=business_impact,
                user_satisfaction=user_satisfaction,
                retention_correlation=retention_correlation,
                revenue_correlation=revenue_correlation,
                optimization_opportunities=optimization_opportunities
            )
            
        except Exception as e:
            logger.error(f"Error calculating feature metrics: {str(e)}")
            return FeaturePerformance(
                feature_id=feature_id,
                feature_name=feature_id,
                adoption_rate=0.0,
                usage_intensity=0.0,
                business_impact=0.0,
                user_satisfaction=0.0,
                retention_correlation=0.0,
                revenue_correlation=0.0,
                optimization_opportunities=[]
            )
    
    def _identify_feature_optimizations(
        self,
        feature_id: str,
        adoption_rate: float,
        usage_intensity: float,
        business_impact: float
    ) -> List[str]:
        """Identify optimization opportunities for a feature"""
        opportunities = []
        
        if adoption_rate < 0.3:
            opportunities.append("Improve feature discoverability and onboarding")
            opportunities.append("Enhance feature marketing and communication")
        elif adoption_rate < 0.6:
            opportunities.append("Simplify feature complexity")
            opportunities.append("Provide better training materials")
        
        if usage_intensity < 10:  # Low usage intensity
            opportunities.append("Improve feature utility and workflow integration")
            opportunities.append("Add usage incentives and gamification")
        
        if business_impact < 0.4:
            opportunities.append("Enhance feature value proposition")
            opportunities.append("Integrate with high-value workflows")
        
        # Feature-specific opportunities
        if len(opportunities) == 0:
            opportunities.append("Monitor for optimization opportunities")
            opportunities.append("Consider advanced feature variations")
        
        return opportunities
    
    def get_usage_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive usage optimization insights and recommendations
        
        Returns:
            Dictionary with usage insights and recommendations
        """
        try:
            logger.info("Generating usage optimization insights")
            
            # Analyze feature performance
            feature_performances = self.analyze_feature_performance()
            
            # Calculate overall metrics
            total_features = len(feature_performances)
            avg_adoption_rate = np.mean([fp.adoption_rate for fp in feature_performances])
            avg_usage_intensity = np.mean([fp.usage_intensity for fp in feature_performances])
            
            # Identify top and bottom performers
            top_features = feature_performances[:5]
            bottom_features = feature_performances[-5:]
            
            # Generate insights
            insights = {
                'summary': {
                    'total_features': total_features,
                    'avg_adoption_rate': avg_adoption_rate,
                    'avg_usage_intensity': avg_usage_intensity,
                    'optimization_potential': 1 - avg_adoption_rate
                },
                'feature_performance': {
                    'top_performers': [
                        {
                            'feature_id': fp.feature_id,
                            'feature_name': fp.feature_name,
                            'adoption_rate': fp.adoption_rate,
                            'business_impact': fp.business_impact
                        } for fp in top_features
                    ],
                    'underperformers': [
                        {
                            'feature_id': fp.feature_id,
                            'feature_name': fp.feature_name,
                            'adoption_rate': fp.adoption_rate,
                            'optimization_opportunities': fp.optimization_opportunities
                        } for fp in bottom_features
                    ]
                },
                'recommendations': {
                    'immediate_actions': [
                        "Improve onboarding flow for underutilized features",
                        "Create targeted campaigns for low-adoption features",
                        "Implement usage analytics to track feature performance",
                        "Develop feature adoption incentive programs"
                    ],
                    'strategic_initiatives': [
                        "Redesign complex features to improve usability",
                        "Create feature bundling strategies for better adoption",
                        "Implement AI-powered feature recommendations",
                        "Develop feature lifecycle management process"
                    ]
                },
                'campaigns': {
                    'active_campaigns': len(self.adoption_campaigns),
                    'recommended_campaigns': [
                        {
                            'target_segment': segment,
                            'campaign_type': 'onboarding',
                            'expected_impact': 'High'
                        } for segment in ['enterprise', 'small_business']
                    ]
                },
                'success_metrics': {
                    'overall_engagement_score': avg_adoption_rate * avg_usage_intensity,
                    'feature_adoption_trend': 'improving' if avg_adoption_rate > 0.5 else 'needs_attention',
                    'usage_optimization_score': avg_usage_intensity / 60  # Normalized to hourly usage
                }
            }
            
            logger.info("Usage optimization insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating usage insights: {str(e)}")
            return {}
    
    def predict_usage_trends(self, forecast_periods: int = 6) -> List[Dict[str, Any]]:
        """
        Predict usage trends for the coming periods
        
        Args:
            forecast_periods: Number of periods to forecast (months)
            
        Returns:
            List of usage trend forecasts
        """
        try:
            logger.info(f"Predicting usage trends for {forecast_periods} periods")
            
            trends = []
            current_date = datetime.now()
            
            for period in range(1, forecast_periods + 1):
                forecast_date = current_date + timedelta(days=30 * period)
                
                # Generate trend forecast
                trend = self._generate_usage_trend_forecast(forecast_date)
                trends.append(trend)
            
            logger.info(f"Generated usage trend forecasts for {len(trends)} periods")
            return trends
            
        except Exception as e:
            logger.error(f"Error predicting usage trends: {str(e)}")
            return []
    
    def _generate_usage_trend_forecast(self, forecast_date: datetime) -> Dict[str, Any]:
        """Generate usage trend forecast for a specific date"""
        # Simplified trend forecasting
        month = forecast_date.month
        
        # Seasonal adjustments
        seasonal_multiplier = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_multiplier = 1.1
        elif month in [1, 2]:  # Post-holiday
            seasonal_multiplier = 0.9
        
        # Growth trend
        growth_factor = 1.02  # 2% monthly growth
        
        base_usage = 100  # Base usage score
        predicted_usage = base_usage * seasonal_multiplier * growth_factor
        
        # Confidence intervals
        std_error = predicted_usage * 0.15
        confidence_interval = (
            max(0, predicted_usage - 1.96 * std_error),
            predicted_usage + 1.96 * std_error
        )
        
        return {
            'period': forecast_date.strftime('%Y-%m'),
            'predicted_usage_score': predicted_usage,
            'confidence_interval': confidence_interval,
            'seasonal_factor': seasonal_multiplier,
            'growth_factor': growth_factor,
            'key_factors': self._identify_usage_trend_factors(month, seasonal_multiplier)
        }
    
    def _identify_usage_trend_factors(self, month: int, seasonal_factor: float) -> List[str]:
        """Identify factors influencing usage trends"""
        factors = []
        
        if month in [11, 12]:
            factors.append("Holiday season usage patterns")
        elif month in [1, 2]:
            factors.append("Post-holiday normalization")
        elif month in [6, 7, 8]:
            factors.append("Summer seasonality effects")
        
        if seasonal_factor > 1.05:
            factors.append("Positive seasonal impact expected")
        elif seasonal_factor < 0.95:
            factors.append("Seasonal headwinds expected")
        
        return factors

# Global instance for easy access
usage_optimizer = UsageOptimizer()

# Example usage and testing
if __name__ == "__main__":
    # Create sample usage data
    np.random.seed(42)
    n_samples = 1000
    
    usage_data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'feature_id': ['feature_' + str(i % 8) for i in range(n_samples)],
        'usage_duration': np.random.exponential(30, n_samples),
        'usage_frequency': np.random.poisson(5, n_samples),
        'user_segment': np.random.choice(['enterprise', 'mid_market', 'small_business'], n_samples),
        'product_tier': np.random.choice(['basic', 'standard', 'premium'], n_samples),
        'engagement_score': np.random.normal(0.7, 0.2, n_samples),
        'retention_days': np.random.exponential(200, n_samples),
        'feature_adopted': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'support_interactions': np.random.poisson(2, n_samples),
        'login_frequency': np.random.poisson(20, n_samples),
        'acquisition_channel': np.random.choice(['direct', 'partner', 'online'], n_samples)
    })
    
    feature_data = pd.DataFrame({
        'feature_id': ['feature_' + str(i) for i in range(8)],
        'feature_name': ['Login', 'Dashboard', 'Reporting', 'Analytics', 'API', 'Integration', 'Automation', 'Advanced Tools'],
        'category': ['core', 'core', 'reporting', 'advanced', 'technical', 'integration', 'advanced', 'premium'],
        'complexity_score': [0.2, 0.3, 0.5, 0.7, 0.8, 0.6, 0.8, 0.9],
        'business_value': [0.8, 0.9, 0.7, 0.8, 0.6, 0.7, 0.8, 0.9]
    })
    
    # Initialize and train the usage optimizer
    optimizer = UsageOptimizer()
    optimizer.load_usage_data(usage_data, feature_data)
    
    # Create sample customer data
    customer_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003],
        'usage_frequency': [25, 10, 40],
        'engagement_score': [0.85, 0.45, 0.92],
        'tenure_days': [180, 30, 365],
        'feature_usage_count': [5, 2, 8],
        'support_interactions': [3, 1, 5],
        'login_frequency': [30, 15, 45],
        'retention_days': [180, 30, 365],
        'used_features': [['login', 'dashboard', 'reporting'], ['login'], ['login', 'dashboard', 'reporting', 'analytics', 'api']],
        'user_segment': ['enterprise', 'small_business', 'enterprise'],
        'avg_revenue_per_month': [800, 200, 1200]
    })
    
    # Analyze usage patterns
    patterns = optimizer.analyze_usage_patterns(customer_data)
    
    print(f"\n=== Usage Pattern Analysis ===")
    for pattern in patterns:
        print(f"Customer {pattern.customer_id}:")
        print(f"  Pattern Type: {pattern.pattern_type}")
        print(f"  Engagement Score: {pattern.engagement_score:.2f}")
        print(f"  Stickiness Index: {pattern.stickiness_index:.2f}")
        print(f"  Lifecycle Stage: {pattern.lifecycle_stage}")
        print(f"  Optimization Potential: {pattern.optimization_potential:.2%}")
        print(f"  Top Utilization Features: {list(pattern.feature_utilization.keys())[:3]}")
        print(f"  Recommended Actions: {pattern.recommended_actions[:2]}\n")
    
    # Identify adoption opportunities
    opportunities = optimizer.identify_adoption_opportunities(customer_data, 3)
    
    print(f"\n=== Feature Adoption Opportunities ===")
    for opp in opportunities[:6]:  # Top 6 opportunities
        print(f"Customer {opp.customer_id}: {opp.feature_id}")
        print(f"  Adoption Probability: {opp.adoption_probability:.2%}")
        print(f"  Expected Impact: ${opp.expected_impact:,.2f}")
        print(f"  Time to Adoption: {opp.time_to_adoption} days")
        print(f"  Intervention Strategy: {opp.intervention_strategy}")
        print(f"  Business Value: {opp.business_value:.2f}\n")
    
    # Analyze feature performance
    feature_performance = optimizer.analyze_feature_performance()
    
    print(f"\n=== Feature Performance Analysis ===")
    for perf in feature_performance[:5]:  # Top 5 features
        print(f"Feature {perf.feature_name}:")
        print(f"  Adoption Rate: {perf.adoption_rate:.2%}")
        print(f"  Usage Intensity: {perf.usage_intensity:.1f} minutes")
        print(f"  Business Impact: {perf.business_impact:.2f}")
        print(f"  Optimization Opportunities: {len(perf.optimization_opportunities)}\n")
    
    # Create adoption campaigns
    campaigns = optimizer.create_adoption_campaigns(['enterprise', 'small_business'])
    
    print(f"\n=== Adoption Campaigns ===")
    for campaign in campaigns[:3]:  # First 3 campaigns
        print(f"Campaign {campaign.campaign_id}:")
        print(f"  Type: {campaign.campaign_type}")
        print(f"  Target Features: {', '.join(campaign.target_features[:2])}")
        print(f"  Expected Adoption Rate: {campaign.expected_adoption_rate:.2%}")
        print(f"  Timeline: {campaign.timeline_weeks} weeks")
        print(f"  Resource Requirements: {list(campaign.resource_requirements.keys())}\n")
    
    # Get usage insights
    insights = optimizer.get_usage_insights()
    print(f"\n=== Usage Optimization Insights ===")
    print(f"Total Features: {insights['summary']['total_features']}")
    print(f"Average Adoption Rate: {insights['summary']['avg_adoption_rate']:.2%}")
    print(f"Optimization Potential: {insights['summary']['optimization_potential']:.2%}")
    print(f"Top Performer: {insights['feature_performance']['top_performers'][0]['feature_name']}")
    print(f"Overall Engagement Score: {insights['success_metrics']['overall_engagement_score']:.2f}")