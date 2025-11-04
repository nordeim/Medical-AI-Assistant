"""
Revenue Maximization Strategies and Lifetime Value Optimization Engine

This module provides AI-powered revenue optimization capabilities including:
- Customer Lifetime Value (CLV) prediction and optimization
- Pricing strategy optimization
- Revenue forecasting and trend analysis
- Monetization opportunity identification
- Revenue stream diversification strategies
- Dynamic pricing and package optimization
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from collections import defaultdict, deque
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CLVPrediction:
    """Customer Lifetime Value prediction result"""
    customer_id: str
    predicted_clv: float
    prediction_horizon_months: int
    confidence_interval: Tuple[float, float]
    key_factors: List[str]
    optimization_opportunities: List[str]
    confidence_score: float
    predicted_at: datetime

@dataclass
class PricingStrategy:
    """Pricing strategy configuration"""
    product_id: str
    current_price: float
    optimized_price: float
    price_change_percentage: float
    expected_revenue_impact: float
    confidence_score: float
    strategy_rationale: str
    competitor_analysis: Dict[str, float]
    recommended_implementation: str

@dataclass
class RevenueStream:
    """Revenue stream analysis"""
    stream_id: str
    stream_name: str
    current_revenue: float
    projected_revenue: float
    growth_rate: float
    sustainability_score: float
    optimization_potential: float
    risk_factors: List[str]

@dataclass
class RevenueForecast:
    """Revenue forecast data"""
    period: str
    predicted_revenue: float
    confidence_interval: Tuple[float, float]
    growth_rate: float
    contributing_factors: List[str]
    risk_assessment: str

class RevenueOptimizer:
    """
    AI-powered revenue maximization and Customer Lifetime Value optimization engine
    
    This engine predicts CLV, optimizes pricing strategies, forecasts revenue,
    and identifies monetization opportunities.
    """
    
    def __init__(self):
        """Initialize the revenue optimizer with ML models and data structures"""
        self.models = {
            'clv_predictor': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            ),
            'pricing_optimizer': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'revenue_forecaster': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=10,
                random_state=42
            ),
            'churn_predictor': RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
        }
        
        self.scalers = {
            'feature_scaler': StandardScaler(),
            'time_series_scaler': StandardScaler()
        }
        
        self.label_encoders = {
            'customer_segment': LabelEncoder(),
            'product_category': LabelEncoder(),
            'region': LabelEncoder()
        }
        
        self.revenue_history = []
        self.clv_cache = {}
        self.pricing_history = {}
        self.revenue_trends = {}
        
        # Revenue optimization strategies
        self.pricing_strategies = {
            'value_based': 'optimize based on perceived value',
            'competitive': 'optimize based on competitor pricing',
            'dynamic': 'optimize based on demand and time',
            'premium': 'optimize for premium positioning',
            'penetration': 'optimize for market penetration'
        }
        
        logger.info("Revenue Optimizer initialized successfully")
    
    def load_revenue_data(self, revenue_data: pd.DataFrame) -> None:
        """
        Load historical revenue data for model training
        
        Args:
            revenue_data: DataFrame with columns:
                - customer_id, revenue, period_start, period_end, product_id,
                - customer_segment, region, acquisition_channel, engagement_score,
                - tenure_months, avg_order_value, purchase_frequency, churn_flag
        """
        try:
            logger.info(f"Loading {len(revenue_data)} revenue records for training")
            
            # Prepare features for CLV prediction
            feature_columns = [
                'tenure_months', 'engagement_score', 'avg_order_value', 'purchase_frequency',
                'customer_tenure', 'feature_usage', 'support_interactions',
                'upgrade_frequency', 'referral_activity'
            ]
            
            # Handle missing values
            X = revenue_data[feature_columns].fillna(0)
            y_clv = revenue_data['revenue']
            
            # Encode categorical features
            categorical_features = ['customer_segment', 'region', 'acquisition_channel']
            for col in categorical_features:
                if col in revenue_data.columns:
                    revenue_data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        revenue_data[col].fillna('unknown')
                    )
                    X[f'{col}_encoded'] = revenue_data[f'{col}_encoded']
            
            # Add derived features
            X['revenue_per_month'] = revenue_data['revenue'] / (revenue_data['tenure_months'] + 1)
            X['revenue_per_order'] = revenue_data['revenue'] / (revenue_data['purchase_frequency'] + 1)
            
            # Scale features
            X_scaled = self.scalers['feature_scaler'].fit_transform(X)
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_clv, test_size=0.2, random_state=42
            )
            
            # Train CLV predictor
            logger.info("Training CLV prediction model...")
            self.models['clv_predictor'].fit(X_train, y_train)
            
            # Evaluate CLV model
            y_pred = self.models['clv_predictor'].predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"CLV Model Performance - MAE: ${mae:,.2f}, RMSE: ${rmse:,.2f}, R²: {r2:.3f}")
            
            # Train churn predictor for revenue optimization
            logger.info("Training churn prediction model...")
            if 'churn_flag' in revenue_data.columns:
                y_churn = revenue_data['churn_flag'].fillna(0).astype(int)
                _, X_test_churn, y_train_churn, y_test_churn = train_test_split(
                    X_scaled, y_churn, test_size=0.2, random_state=42, stratify=y_churn
                )
                self.models['churn_predictor'].fit(X_train, y_train_churn)
            
            # Store revenue history
            self.revenue_history = revenue_data.to_dict('records')
            
            # Analyze revenue trends
            self._analyze_revenue_trends(revenue_data)
            
            logger.info("Revenue data loaded and models trained successfully")
            
        except Exception as e:
            logger.error(f"Error loading revenue data: {str(e)}")
            raise
    
    def predict_customer_lifetime_value(
        self,
        customer_data: pd.DataFrame,
        prediction_horizon_months: int = 12
    ) -> List[CLVPrediction]:
        """
        Predict Customer Lifetime Value for customers
        
        Args:
            customer_data: DataFrame with customer information
            prediction_horizon_months: Number of months to predict CLV for
            
        Returns:
            List of CLVPrediction objects
        """
        try:
            logger.info(f"Predicting CLV for {len(customer_data)} customers over {prediction_horizon_months} months")
            
            clv_predictions = []
            
            for _, customer in customer_data.iterrows():
                prediction = self._predict_single_customer_clv(
                    customer, prediction_horizon_months
                )
                clv_predictions.append(prediction)
            
            # Sort by predicted CLV
            clv_predictions.sort(key=lambda x: x.predicted_clv, reverse=True)
            
            logger.info(f"Generated CLV predictions for {len(clv_predictions)} customers")
            return clv_predictions
            
        except Exception as e:
            logger.error(f"Error predicting customer lifetime value: {str(e)}")
            raise
    
    def _predict_single_customer_clv(
        self,
        customer: pd.Series,
        prediction_horizon_months: int
    ) -> CLVPrediction:
        """Predict CLV for a single customer"""
        customer_id = str(customer.get('customer_id', 'unknown'))
        
        # Prepare features
        features = self._prepare_clv_features(customer, prediction_horizon_months)
        features_scaled = self.scalers['feature_scaler'].transform(features.reshape(1, -1))
        
        # Predict base CLV
        predicted_clv = self.models['clv_predictor'].predict(features_scaled)[0]
        
        # Calculate confidence interval (simplified)
        standard_error = predicted_clv * 0.15  # 15% standard error assumption
        confidence_interval = (
            max(0, predicted_clv - 1.96 * standard_error),
            predicted_clv + 1.96 * standard_error
        )
        
        # Determine key factors affecting CLV
        key_factors = self._identify_clv_factors(customer, features)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_clv_opportunities(customer, predicted_clv)
        
        # Calculate confidence score
        confidence_score = self._calculate_clv_confidence(customer, features)
        
        return CLVPrediction(
            customer_id=customer_id,
            predicted_clv=max(predicted_clv, 0),
            prediction_horizon_months=prediction_horizon_months,
            confidence_interval=confidence_interval,
            key_factors=key_factors,
            optimization_opportunities=optimization_opportunities,
            confidence_score=confidence_score,
            predicted_at=datetime.now()
        )
    
    def _prepare_clv_features(self, customer: pd.Series, horizon_months: int) -> np.ndarray:
        """Prepare features for CLV prediction"""
        feature_values = [
            customer.get('tenure_months', 0),
            customer.get('engagement_score', 0),
            customer.get('avg_order_value', 0),
            customer.get('purchase_frequency', 0),
            customer.get('customer_tenure', 0),
            customer.get('feature_usage', 0),
            customer.get('support_interactions', 0),
            customer.get('upgrade_frequency', 0),
            customer.get('referral_activity', 0),
            horizon_months,  # Prediction horizon as feature
            customer.get('avg_order_value', 0) / max(customer.get('tenure_months', 1), 1),  # Revenue velocity
            max(customer.get('purchase_frequency', 0), 0.1) / max(horizon_months, 1)  # Purchase rate
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
        
        return np.array(feature_values)
    
    def _identify_clv_factors(self, customer: pd.Series, features: np.ndarray) -> List[str]:
        """Identify key factors affecting customer CLV"""
        factors = []
        
        if customer.get('engagement_score', 0) > 0.8:
            factors.append("High engagement level")
        elif customer.get('engagement_score', 0) < 0.4:
            factors.append("Low engagement level")
        
        if customer.get('tenure_months', 0) > 24:
            factors.append("Long customer tenure")
        elif customer.get('tenure_months', 0) < 6:
            factors.append("Short customer tenure")
        
        if customer.get('avg_order_value', 0) > 500:
            factors.append("High average order value")
        
        if customer.get('purchase_frequency', 0) > 10:
            factors.append("High purchase frequency")
        
        customer_segment = customer.get('customer_segment', '')
        if customer_segment:
            factors.append(f"Customer segment: {customer_segment}")
        
        return factors
    
    def _identify_clv_opportunities(self, customer: pd.Series, predicted_clv: float) -> List[str]:
        """Identify opportunities to increase customer CLV"""
        opportunities = []
        
        engagement_score = customer.get('engagement_score', 0)
        if engagement_score < 0.6:
            opportunities.append("Increase product engagement through training and onboarding")
        
        feature_usage = customer.get('feature_usage', 0)
        if feature_usage < 0.5:
            opportunities.append("Encourage feature adoption to increase usage value")
        
        purchase_frequency = customer.get('purchase_frequency', 0)
        if purchase_frequency < 5:
            opportunities.append("Increase purchase frequency through targeted campaigns")
        
        avg_order_value = customer.get('avg_order_value', 0)
        if avg_order_value < 300:
            opportunities.append("Increase average order value through upselling")
        
        if predicted_clv < 1000:
            opportunities.append("Implement retention strategies for low-CLV customers")
        
        return opportunities
    
    def _calculate_clv_confidence(self, customer: pd.Series, features: np.ndarray) -> float:
        """Calculate confidence score for CLV prediction"""
        # Base confidence on feature completeness
        total_features = len(features)
        available_features = np.count_nonzero(features)
        feature_completeness = available_features / total_features
        
        # Adjust based on customer tenure (more historical data = higher confidence)
        tenure_factor = min(customer.get('tenure_months', 0) / 24, 1.0)
        
        # Model-specific confidence (simplified)
        model_confidence = 0.75
        
        confidence = (feature_completeness * 0.4 + tenure_factor * 0.3 + model_confidence * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def optimize_pricing_strategy(
        self,
        product_data: pd.DataFrame,
        strategy_type: str = 'value_based',
        competitor_data: Optional[pd.DataFrame] = None
    ) -> List[PricingStrategy]:
        """
        Optimize pricing strategies for products
        
        Args:
            product_data: DataFrame with product information and current pricing
            strategy_type: Type of pricing strategy to apply
            competitor_data: Optional competitor pricing data
            
        Returns:
            List of PricingStrategy objects
        """
        try:
            logger.info(f"Optimizing pricing strategies using {strategy_type} approach")
            
            pricing_strategies = []
            
            for _, product in product_data.iterrows():
                strategy = self._calculate_optimal_pricing(
                    product, strategy_type, competitor_data
                )
                pricing_strategies.append(strategy)
            
            # Sort by expected revenue impact
            pricing_strategies.sort(key=lambda x: x.expected_revenue_impact, reverse=True)
            
            logger.info(f"Generated pricing strategies for {len(pricing_strategies)} products")
            return pricing_strategies
            
        except Exception as e:
            logger.error(f"Error optimizing pricing strategy: {str(e)}")
            raise
    
    def _calculate_optimal_pricing(
        self,
        product: pd.Series,
        strategy_type: str,
        competitor_data: Optional[pd.DataFrame]
    ) -> PricingStrategy:
        """Calculate optimal pricing for a product"""
        product_id = str(product.get('product_id', 'unknown'))
        current_price = product.get('current_price', 100)
        
        # Get competitor pricing if available
        competitor_analysis = {}
        if competitor_data is not None:
            competitor_analysis = self._analyze_competitor_pricing(product_id, competitor_data)
        
        # Calculate optimal price based on strategy
        if strategy_type == 'value_based':
            optimized_price = self._calculate_value_based_price(product)
        elif strategy_type == 'competitive':
            optimized_price = self._calculate_competitive_price(product, competitor_analysis)
        elif strategy_type == 'dynamic':
            optimized_price = self._calculate_dynamic_price(product)
        elif strategy_type == 'premium':
            optimized_price = self._calculate_premium_price(product)
        elif strategy_type == 'penetration':
            optimized_price = self._calculate_penetration_price(product)
        else:
            optimized_price = current_price
        
        # Calculate expected revenue impact
        expected_revenue_impact = self._estimate_pricing_impact(
            product, current_price, optimized_price
        )
        
        # Generate strategy rationale
        rationale = self._generate_pricing_rationale(
            strategy_type, current_price, optimized_price, competitor_analysis
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_pricing_confidence(
            product, competitor_analysis, strategy_type
        )
        
        price_change_percentage = ((optimized_price - current_price) / current_price) * 100
        
        return PricingStrategy(
            product_id=product_id,
            current_price=current_price,
            optimized_price=optimized_price,
            price_change_percentage=price_change_percentage,
            expected_revenue_impact=expected_revenue_impact,
            confidence_score=confidence_score,
            strategy_rationale=rationale,
            competitor_analysis=competitor_analysis,
            recommended_implementation=self._recommend_implementation_timeline(
                price_change_percentage, confidence_score
            )
        )
    
    def _calculate_value_based_price(self, product: pd.Series) -> float:
        """Calculate value-based pricing"""
        base_price = product.get('current_price', 100)
        perceived_value = product.get('perceived_value', 0.7)  # 0-1 scale
        customer_satisfaction = product.get('customer_satisfaction', 0.8)
        
        # Value-based multiplier: higher perceived value and satisfaction = higher price
        value_multiplier = (perceived_value * 0.6 + customer_satisfaction * 0.4) * 0.5 + 0.75
        
        optimized_price = base_price * value_multiplier
        return max(optimized_price, base_price * 0.8)  # Don't drop below 80% of current
    
    def _calculate_competitive_price(
        self,
        product: pd.Series,
        competitor_analysis: Dict[str, float]
    ) -> float:
        """Calculate competitive pricing"""
        base_price = product.get('current_price', 100)
        
        if 'avg_competitor_price' in competitor_analysis:
            competitor_avg = competitor_analysis['avg_competitor_price']
            # Price at 95% of competitor average to be competitive
            optimized_price = competitor_avg * 0.95
        else:
            # Default competitive adjustment
            optimized_price = base_price * 0.95
        
        return optimized_price
    
    def _calculate_dynamic_price(self, product: pd.Series) -> float:
        """Calculate dynamic pricing based on demand and time"""
        base_price = product.get('current_price', 100)
        
        # Time-based factors
        day_of_week = datetime.now().weekday()
        month = datetime.now().month
        
        # Demand factors (simplified)
        demand_multiplier = 1.0
        if day_of_week in [5, 6]:  # Weekend boost
            demand_multiplier += 0.1
        if month in [11, 12]:  # Holiday season boost
            demand_multiplier += 0.15
        
        # Feature complexity factor
        feature_complexity = product.get('feature_complexity', 0.5)
        complexity_multiplier = 1 + (feature_complexity * 0.2)
        
        optimized_price = base_price * demand_multiplier * complexity_multiplier
        return optimized_price
    
    def _calculate_premium_price(self, product: pd.Series) -> float:
        """Calculate premium pricing"""
        base_price = product.get('current_price', 100)
        
        # Premium positioning factor
        quality_score = product.get('quality_score', 0.8)
        innovation_score = product.get('innovation_score', 0.7)
        
        premium_multiplier = (quality_score * 0.5 + innovation_score * 0.5) * 0.3 + 1.2
        
        optimized_price = base_price * premium_multiplier
        return optimized_price
    
    def _calculate_penetration_price(self, product: pd.Series) -> float:
        """Calculate penetration pricing"""
        base_price = product.get('current_price', 100)
        
        # Market penetration factor
        market_share = product.get('market_share', 0.1)
        
        penetration_multiplier = 1.0 - (market_share * 0.2)  # Lower price for lower market share
        penetration_multiplier = max(penetration_multiplier, 0.7)  # Don't go below 70%
        
        optimized_price = base_price * penetration_multiplier
        return optimized_price
    
    def _analyze_competitor_pricing(
        self,
        product_id: str,
        competitor_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Analyze competitor pricing for a product"""
        try:
            product_competitors = competitor_data[
                competitor_data['product_id'] == product_id
            ]
            
            if product_competitors.empty:
                return {}
            
            prices = product_competitors['competitor_price'].values
            
            analysis = {
                'avg_competitor_price': np.mean(prices),
                'min_competitor_price': np.min(prices),
                'max_competitor_price': np.max(prices),
                'competitor_price_std': np.std(prices),
                'competitor_count': len(prices)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing competitor pricing: {str(e)}")
            return {}
    
    def _estimate_pricing_impact(
        self,
        product: pd.Series,
        current_price: float,
        optimized_price: float
    ) -> float:
        """Estimate revenue impact of pricing change"""
        # Simplified price elasticity model
        price_change_ratio = (optimized_price - current_price) / current_price
        
        # Estimate demand elasticity (simplified)
        demand_elasticity = -1.5  # Typical elasticity for digital products
        demand_change = demand_elasticity * price_change_ratio
        
        # Estimate volume change
        current_volume = product.get('monthly_sales_volume', 1000)
        new_volume = current_volume * (1 + demand_change)
        
        # Calculate revenue impact
        current_revenue = current_price * current_volume
        new_revenue = optimized_price * new_volume
        revenue_impact = new_revenue - current_revenue
        
        return revenue_impact
    
    def _generate_pricing_rationale(
        self,
        strategy_type: str,
        current_price: float,
        optimized_price: float,
        competitor_analysis: Dict[str, float]
    ) -> str:
        """Generate rationale for pricing strategy"""
        rationale_parts = []
        
        rationale_parts.append(f"Strategy: {self.pricing_strategies[strategy_type]}")
        
        price_change = ((optimized_price - current_price) / current_price) * 100
        if price_change > 5:
            rationale_parts.append(f"Price increase of {price_change:.1f}% recommended")
        elif price_change < -5:
            rationale_parts.append(f"Price decrease of {abs(price_change):.1f}% recommended")
        else:
            rationale_parts.append("Minor price adjustment recommended")
        
        if competitor_analysis:
            rationale_parts.append("Based on competitive market analysis")
        
        return ". ".join(rationale_parts)
    
    def _calculate_pricing_confidence(
        self,
        product: pd.Series,
        competitor_analysis: Dict[str, float],
        strategy_type: str
    ) -> float:
        """Calculate confidence score for pricing strategy"""
        # Base confidence
        base_confidence = 0.7
        
        # Adjust based on data availability
        if competitor_analysis:
            base_confidence += 0.1
        
        # Strategy-specific adjustments
        strategy_confidence = {
            'value_based': 0.8,
            'competitive': 0.9,
            'dynamic': 0.6,
            'premium': 0.7,
            'penetration': 0.8
        }
        
        confidence = strategy_confidence.get(strategy_type, base_confidence)
        return min(max(confidence, 0.0), 1.0)
    
    def _recommend_implementation_timeline(
        self,
        price_change_percentage: float,
        confidence_score: float
    ) -> str:
        """Recommend implementation timeline for pricing changes"""
        if abs(price_change_percentage) > 20:
            return "Gradual implementation over 3-6 months with monitoring"
        elif abs(price_change_percentage) > 10:
            return "Moderate implementation over 1-2 months"
        else:
            return "Quick implementation within 2-4 weeks"
    
    def forecast_revenue(
        self,
        forecast_period_months: int = 6,
        include_seasonality: bool = True
    ) -> List[RevenueForecast]:
        """
        Forecast revenue for the specified period
        
        Args:
            forecast_period_months: Number of months to forecast
            include_seasonality: Whether to include seasonal factors
            
        Returns:
            List of RevenueForecast objects
        """
        try:
            logger.info(f"Forecasting revenue for {forecast_period_months} months")
            
            forecasts = []
            
            # Get current date and prepare forecast periods
            current_date = datetime.now()
            for month_offset in range(1, forecast_period_months + 1):
                forecast_date = current_date + timedelta(days=30 * month_offset)
                
                # Forecast revenue for this period
                forecast = self._forecast_period_revenue(
                    forecast_date, include_seasonality
                )
                forecasts.append(forecast)
            
            logger.info(f"Generated revenue forecasts for {len(forecasts)} periods")
            return forecasts
            
        except Exception as e:
            logger.error(f"Error forecasting revenue: {str(e)}")
            raise
    
    def _forecast_period_revenue(
        self,
        forecast_date: datetime,
        include_seasonality: bool
    ) -> RevenueForecast:
        """Forecast revenue for a specific period"""
        try:
            # Prepare time-based features
            features = self._prepare_forecast_features(forecast_date, include_seasonality)
            
            # Predict revenue
            predicted_revenue = self.models['revenue_forecaster'].predict([features])[0]
            
            # Calculate confidence interval
            standard_error = predicted_revenue * 0.12  # 12% standard error
            confidence_interval = (
                max(0, predicted_revenue - 1.96 * standard_error),
                predicted_revenue + 1.96 * standard_error
            )
            
            # Calculate growth rate
            previous_revenue = self._get_previous_period_revenue(forecast_date)
            growth_rate = ((predicted_revenue - previous_revenue) / previous_revenue * 100) if previous_revenue > 0 else 0
            
            # Identify contributing factors
            contributing_factors = self._identify_revenue_factors(forecast_date, features)
            
            # Assess risk
            risk_assessment = self._assess_revenue_risk(forecast_date, predicted_revenue)
            
            return RevenueForecast(
                period=forecast_date.strftime('%Y-%m'),
                predicted_revenue=max(predicted_revenue, 0),
                confidence_interval=confidence_interval,
                growth_rate=growth_rate,
                contributing_factors=contributing_factors,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error forecasting period revenue: {str(e)}")
            # Return default forecast
            return RevenueForecast(
                period=forecast_date.strftime('%Y-%m'),
                predicted_revenue=100000,
                confidence_interval=(85000, 115000),
                growth_rate=5.0,
                contributing_factors=["Base forecast"],
                risk_assessment="Low risk"
            )
    
    def _prepare_forecast_features(
        self,
        forecast_date: datetime,
        include_seasonality: bool
    ) -> np.ndarray:
        """Prepare features for revenue forecasting"""
        # Time-based features
        month = forecast_date.month
        quarter = (month - 1) // 3 + 1
        day_of_week = forecast_date.weekday()
        
        # Seasonal factors
        seasonal_multiplier = 1.0
        if include_seasonality:
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier = 1.2
            elif month in [1, 2]:  # Post-holiday dip
                seasonal_multiplier = 0.85
            elif month in [6, 7, 8]:  # Summer
                seasonal_multiplier = 0.95
        
        # Trend features (simplified)
        trend_factor = 1.05  # 5% growth assumption
        
        features = np.array([
            month, quarter, day_of_week, seasonal_multiplier, trend_factor,
            1, 1, 1, 1, 1  # Placeholder features for model compatibility
        ])
        
        return features
    
    def _get_previous_period_revenue(self, forecast_date: datetime) -> float:
        """Get revenue from previous period for growth calculation"""
        # Simplified: use average from revenue history
        if self.revenue_history:
            recent_revenues = [record.get('revenue', 0) for record in self.revenue_history[-12:]]
            return np.mean(recent_revenues) if recent_revenues else 100000
        return 100000
    
    def _identify_revenue_factors(self, forecast_date: datetime, features: np.ndarray) -> List[str]:
        """Identify factors contributing to revenue forecast"""
        factors = []
        
        month = int(features[0])
        
        if month in [11, 12]:
            factors.append("Holiday season demand boost")
        elif month in [1, 2]:
            factors.append("Post-holiday adjustment period")
        elif month in [6, 7, 8]:
            factors.append("Summer seasonal effect")
        
        seasonal_multiplier = features[3]
        if seasonal_multiplier > 1.1:
            factors.append("Strong seasonal performance expected")
        elif seasonal_multiplier < 0.9:
            factors.append("Seasonal headwinds expected")
        
        trend_factor = features[4]
        if trend_factor > 1.03:
            factors.append("Positive growth trend")
        
        return factors
    
    def _assess_revenue_risk(self, forecast_date: datetime, predicted_revenue: float) -> str:
        """Assess risk factors for revenue forecast"""
        month = forecast_date.month
        
        # High-risk periods
        if month in [1, 2]:
            return "High risk - Post-holiday slowdown and budget constraints"
        elif month == 9:
            return "Medium risk - Potential end-of-summer adjustments"
        
        # Medium-risk periods
        if month in [3, 4, 5]:
            return "Medium risk - Spring volatility"
        elif month in [10]:
            return "Medium risk - Pre-holiday planning adjustments"
        
        # Low-risk periods
        return "Low risk - Stable period expected"
    
    def identify_revenue_opportunities(
        self,
        customer_data: pd.DataFrame,
        product_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Identify revenue optimization opportunities
        
        Args:
            customer_data: Customer data for opportunity analysis
            product_data: Product data for opportunity analysis
            
        Returns:
            List of revenue opportunity dictionaries
        """
        try:
            logger.info("Identifying revenue optimization opportunities")
            
            opportunities = []
            
            # Analyze pricing optimization opportunities
            pricing_opportunities = self._identify_pricing_opportunities(product_data)
            opportunities.extend(pricing_opportunities)
            
            # Analyze customer expansion opportunities
            expansion_opportunities = self._identify_expansion_opportunities(customer_data)
            opportunities.extend(expansion_opportunities)
            
            # Analyze upselling opportunities
            upsell_opportunities = self._identify_upsell_opportunities(customer_data)
            opportunities.extend(upsell_opportunities)
            
            # Sort by expected revenue impact
            opportunities.sort(key=lambda x: x.get('expected_impact', 0), reverse=True)
            
            logger.info(f"Identified {len(opportunities)} revenue opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying revenue opportunities: {str(e)}")
            return []
    
    def _identify_pricing_opportunities(self, product_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify pricing optimization opportunities"""
        opportunities = []
        
        for _, product in product_data.iterrows():
            current_price = product.get('current_price', 100)
            demand_sensitivity = product.get('demand_sensitivity', 0.5)
            
            # Price optimization opportunity
            if demand_sensitivity < 0.6:
                # Low demand sensitivity = can increase price
                opportunity = {
                    'type': 'pricing_optimization',
                    'product_id': product.get('product_id', 'unknown'),
                    'opportunity': 'Increase price due to low demand sensitivity',
                    'current_price': current_price,
                    'recommended_action': 'Increase price by 10-15%',
                    'expected_impact': current_price * 0.125,  # 12.5% revenue increase
                    'confidence': 0.8
                }
                opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_expansion_opportunities(self, customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify customer expansion opportunities"""
        opportunities = []
        
        high_value_customers = customer_data[
            (customer_data.get('avg_order_value', 0) > 500) |
            (customer_data.get('engagement_score', 0) > 0.8)
        ]
        
        for _, customer in high_value_customers.iterrows():
            opportunity = {
                'type': 'customer_expansion',
                'customer_id': customer.get('customer_id', 'unknown'),
                'opportunity': 'Expand product portfolio for high-value customer',
                'recommended_action': 'Offer premium product bundle',
                'expected_impact': customer.get('avg_order_value', 300) * 0.5,
                'confidence': 0.7
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def _identify_upsell_opportunities(self, customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify upselling opportunities"""
        opportunities = []
        
        engaged_customers = customer_data[
            (customer_data.get('engagement_score', 0) > 0.7) &
            (customer_data.get('tenure_months', 0) > 6)
        ]
        
        for _, customer in engaged_customers.iterrows():
            opportunity = {
                'type': 'upselling',
                'customer_id': customer.get('customer_id', 'unknown'),
                'opportunity': 'Upsell to higher tier based on high engagement',
                'recommended_action': 'Offer premium tier upgrade',
                'expected_impact': customer.get('avg_order_value', 200) * 0.8,
                'confidence': 0.75
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def analyze_revenue_streams(self) -> List[RevenueStream]:
        """
        Analyze current revenue streams and optimization potential
        
        Returns:
            List of RevenueStream objects
        """
        try:
            logger.info("Analyzing revenue streams")
            
            # Define revenue stream categories
            stream_definitions = {
                'subscription_revenue': {'name': 'Subscription Revenue', 'growth_factor': 1.1},
                'usage_based_revenue': {'name': 'Usage-Based Revenue', 'growth_factor': 1.15},
                'professional_services': {'name': 'Professional Services', 'growth_factor': 1.05},
                'partner_commissions': {'name': 'Partner Commissions', 'growth_factor': 1.08},
                'add_on_products': {'name': 'Add-on Products', 'growth_factor': 1.2}
            }
            
            revenue_streams = []
            
            for stream_id, definition in stream_definitions.items():
                # Calculate current revenue (simplified)
                current_revenue = self._calculate_stream_revenue(stream_id)
                
                # Project future revenue
                growth_factor = definition['growth_factor']
                projected_revenue = current_revenue * growth_factor
                
                # Calculate growth rate
                growth_rate = (growth_factor - 1) * 100
                
                # Assess sustainability
                sustainability_score = self._assess_stream_sustainability(stream_id)
                
                # Calculate optimization potential
                optimization_potential = self._calculate_optimization_potential(stream_id)
                
                # Identify risk factors
                risk_factors = self._identify_stream_risks(stream_id)
                
                stream = RevenueStream(
                    stream_id=stream_id,
                    stream_name=definition['name'],
                    current_revenue=current_revenue,
                    projected_revenue=projected_revenue,
                    growth_rate=growth_rate,
                    sustainability_score=sustainability_score,
                    optimization_potential=optimization_potential,
                    risk_factors=risk_factors
                )
                
                revenue_streams.append(stream)
            
            # Sort by projected revenue
            revenue_streams.sort(key=lambda x: x.projected_revenue, reverse=True)
            
            logger.info(f"Analyzed {len(revenue_streams)} revenue streams")
            return revenue_streams
            
        except Exception as e:
            logger.error(f"Error analyzing revenue streams: {str(e)}")
            return []
    
    def _calculate_stream_revenue(self, stream_id: str) -> float:
        """Calculate current revenue for a stream"""
        # Simplified revenue calculation
        base_revenues = {
            'subscription_revenue': 250000,
            'usage_based_revenue': 150000,
            'professional_services': 75000,
            'partner_commissions': 50000,
            'add_on_products': 100000
        }
        return base_revenues.get(stream_id, 50000)
    
    def _assess_stream_sustainability(self, stream_id: str) -> float:
        """Assess sustainability score for revenue stream"""
        sustainability_scores = {
            'subscription_revenue': 0.9,  # Very sustainable
            'usage_based_revenue': 0.8,   # Moderately sustainable
            'professional_services': 0.7, # Less sustainable
            'partner_commissions': 0.6,   # Dependent on partners
            'add_on_products': 0.85       # Good sustainability
        }
        return sustainability_scores.get(stream_id, 0.5)
    
    def _calculate_optimization_potential(self, stream_id: str) -> float:
        """Calculate optimization potential for revenue stream"""
        optimization_potentials = {
            'subscription_revenue': 0.15,  # 15% optimization potential
            'usage_based_revenue': 0.25,   # 25% optimization potential
            'professional_services': 0.20, # 20% optimization potential
            'partner_commissions': 0.10,   # 10% optimization potential
            'add_on_products': 0.30        # 30% optimization potential
        }
        return optimization_potentials.get(stream_id, 0.15)
    
    def _identify_stream_risks(self, stream_id: str) -> List[str]:
        """Identify risk factors for revenue stream"""
        risk_factors = {
            'subscription_revenue': ['Customer churn', 'Competitive pricing pressure'],
            'usage_based_revenue': ['Economic downturn impact', 'Usage pattern changes'],
            'professional_services': ['Resource availability', 'Project scope creep'],
            'partner_commissions': ['Partner performance', 'Market saturation'],
            'add_on_products': ['Market acceptance', 'Feature cannibalization']
        }
        return risk_factors.get(stream_id, ['General market risks'])
    
    def get_revenue_insights(self) -> Dict[str, Any]:
        """
        Get comprehensive revenue insights and recommendations
        
        Returns:
            Dictionary with revenue insights and recommendations
        """
        try:
            logger.info("Generating revenue insights")
            
            insights = {
                'summary': {
                    'total_revenue_streams': len(self.analyze_revenue_streams()),
                    'key_performance_indicators': self._calculate_revenue_kpis(),
                    'revenue_trends': self._analyze_trend_insights()
                },
                'opportunities': {
                    'high_impact_opportunities': self.identify_revenue_opportunities(pd.DataFrame(), pd.DataFrame())[:5],
                    'optimization_recommendations': self._generate_optimization_recommendations()
                },
                'risks': {
                    'revenue_risks': self._identify_revenue_risks(),
                    'mitigation_strategies': self._generate_risk_mitigation_strategies()
                },
                'forecasts': {
                    'short_term_forecast': self.forecast_revenue(3),
                    'medium_term_forecast': self.forecast_revenue(6),
                    'long_term_forecast': self.forecast_revenue(12)
                },
                'recommendations': {
                    'immediate_actions': self._generate_immediate_recommendations(),
                    'strategic_initiatives': self._generate_strategic_initiatives()
                }
            }
            
            logger.info("Revenue insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating revenue insights: {str(e)}")
            return {}
    
    def _analyze_revenue_trends(self, revenue_data: pd.DataFrame) -> None:
        """Analyze revenue trends from historical data"""
        try:
            # Group by time periods for trend analysis
            if 'period_start' in revenue_data.columns:
                revenue_data['period'] = pd.to_datetime(revenue_data['period_start'])
                monthly_revenue = revenue_data.groupby(
                    revenue_data['period'].dt.to_period('M')
                )['revenue'].sum()
                
                self.revenue_trends = {
                    'monthly_revenue': monthly_revenue.to_dict(),
                    'growth_rates': self._calculate_growth_rates(monthly_revenue),
                    'seasonality': self._detect_seasonality(monthly_revenue)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing revenue trends: {str(e)}")
    
    def _calculate_growth_rates(self, revenue_series: pd.Series) -> Dict[str, float]:
        """Calculate growth rates from revenue series"""
        if len(revenue_series) < 2:
            return {}
        
        current_revenue = revenue_series.iloc[-1]
        previous_revenue = revenue_series.iloc[-2]
        
        growth_rate = ((current_revenue - previous_revenue) / previous_revenue) * 100
        
        return {
            'month_over_month_growth': growth_rate,
            'average_monthly_growth': growth_rate  # Simplified
        }
    
    def _detect_seasonality(self, revenue_series: pd.Series) -> Dict[str, float]:
        """Detect seasonal patterns in revenue"""
        if len(revenue_series) < 12:
            return {}
        
        try:
            # Convert to monthly data and calculate seasonal indices
            seasonal_indices = {}
            
            for period, revenue in revenue_series.items():
                month = period.month
                if month not in seasonal_indices:
                    seasonal_indices[month] = []
                seasonal_indices[month].append(revenue)
            
            # Calculate average for each month
            monthly_averages = {}
            overall_average = np.mean(list(revenue_series.values))
            
            for month, revenues in seasonal_indices.items():
                monthly_averages[month] = np.mean(revenues) / overall_average
            
            return monthly_averages
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {str(e)}")
            return {}
    
    def _calculate_revenue_kpis(self) -> Dict[str, float]:
        """Calculate key performance indicators for revenue"""
        if not self.revenue_history:
            return {}
        
        # Calculate basic KPIs
        total_revenue = sum(record.get('revenue', 0) for record in self.revenue_history)
        avg_revenue_per_customer = total_revenue / len(self.revenue_history) if self.revenue_history else 0
        
        # Customer metrics
        customers = set(record.get('customer_id') for record in self.revenue_history if record.get('customer_id'))
        avg_revenue_per_customer_month = total_revenue / len(customers) if customers else 0
        
        return {
            'total_revenue': total_revenue,
            'avg_revenue_per_customer': avg_revenue_per_customer,
            'avg_revenue_per_customer_month': avg_revenue_per_customer_month,
            'total_customers': len(customers)
        }
    
    def _analyze_trend_insights(self) -> Dict[str, Any]:
        """Analyze trend insights"""
        return {
            'overall_trend': 'positive' if self.revenue_trends.get('growth_rates', {}).get('month_over_month_growth', 0) > 0 else 'stable',
            'seasonal_patterns_detected': len(self.revenue_trends.get('seasonality', {})) > 0,
            'trend_confidence': 0.75  # Simplified confidence score
        }
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        return [
            "Focus on high-CLV customer segments for expansion opportunities",
            "Implement dynamic pricing for usage-based products",
            "Strengthen retention strategies for subscription revenue",
            "Develop professional services packages for enterprise customers",
            "Create add-on product bundles to increase average order value"
        ]
    
    def _identify_revenue_risks(self) -> List[str]:
        """Identify revenue risks"""
        return [
            "Customer concentration risk in top accounts",
            "Seasonal revenue volatility",
            "Competitive pricing pressure in core markets",
            "Economic downturn impact on discretionary spending",
            "Technology disruption in traditional revenue models"
        ]
    
    def _generate_risk_mitigation_strategies(self) -> List[str]:
        """Generate risk mitigation strategies"""
        return [
            "Diversify customer base to reduce concentration risk",
            "Implement counter-seasonal offerings",
            "Develop competitive differentiation beyond price",
            "Build recession-resistant product features",
            "Invest in emerging technology and market opportunities"
        ]
    
    def _generate_immediate_recommendations(self) -> List[str]:
        """Generate immediate action recommendations"""
        return [
            "Implement pricing optimization for top 20% of products",
            "Launch targeted expansion campaigns for high-engagement customers",
            "Increase professional services capacity for Q4 demand",
            "Deploy predictive churn models to prevent revenue loss",
            "Optimize product packaging and bundles for Q4 sales"
        ]
    
    def _generate_strategic_initiatives(self) -> List[str]:
        """Generate strategic initiative recommendations"""
        return [
            "Develop AI-powered pricing optimization system",
            "Create customer success program to increase CLV",
            "Build partnership ecosystem for new revenue streams",
            "Invest in product innovation for premium positioning",
            "Establish market expansion strategy for new geographies"
        ]

# Global instance for easy access
revenue_optimizer = RevenueOptimizer()

# Example usage and testing
if __name__ == "__main__":
    # Create sample revenue data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'revenue': np.random.normal(500, 200, n_samples),
        'period_start': pd.date_range('2022-01-01', periods=n_samples, freq='D'),
        'product_id': ['product_' + str(i % 10) for i in range(n_samples)],
        'customer_segment': np.random.choice(['enterprise', 'mid_market', 'small_business'], n_samples),
        'region': np.random.choice(['north_america', 'europe', 'asia_pacific'], n_samples),
        'acquisition_channel': np.random.choice(['direct', 'partner', 'online'], n_samples),
        'engagement_score': np.random.normal(0.7, 0.2, n_samples),
        'tenure_months': np.random.normal(18, 12, n_samples),
        'avg_order_value': np.random.normal(300, 150, n_samples),
        'purchase_frequency': np.random.poisson(5, n_samples),
        'churn_flag': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Initialize and train the revenue optimizer
    optimizer = RevenueOptimizer()
    optimizer.load_revenue_data(sample_data)
    
    # Predict CLV for sample customers
    customer_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003],
        'tenure_months': [24, 6, 36],
        'engagement_score': [0.85, 0.45, 0.92],
        'avg_order_value': [800, 200, 1200],
        'purchase_frequency': [12, 3, 20],
        'customer_segment': ['enterprise', 'small_business', 'enterprise'],
        'region': ['north_america', 'europe', 'asia_pacific'],
        'acquisition_channel': ['direct', 'partner', 'online']
    })
    
    clv_predictions = optimizer.predict_customer_lifetime_value(customer_data)
    
    print(f"\n=== CLV Predictions ===")
    for prediction in clv_predictions:
        print(f"Customer {prediction.customer_id}:")
        print(f"  Predicted CLV: ${prediction.predicted_clv:,.2f}")
        print(f"  Confidence Interval: ${prediction.confidence_interval[0]:,.2f} - ${prediction.confidence_interval[1]:,.2f}")
        print(f"  Confidence Score: {prediction.confidence_score:.2%}")
        print(f"  Key Factors: {', '.join(prediction.key_factors)}")
        print(f"  Optimization Opportunities: {', '.join(prediction.optimization_opportunities[:2])}\n")
    
    # Optimize pricing strategy
    product_data = pd.DataFrame({
        'product_id': ['product_1', 'product_2', 'product_3'],
        'current_price': [100, 250, 500],
        'perceived_value': [0.8, 0.7, 0.9],
        'customer_satisfaction': [0.85, 0.75, 0.95],
        'monthly_sales_volume': [1000, 500, 200]
    })
    
    pricing_strategies = optimizer.optimize_pricing_strategy(product_data, 'value_based')
    
    print(f"\n=== Pricing Strategies ===")
    for strategy in pricing_strategies:
        print(f"Product {strategy.product_id}:")
        print(f"  Current Price: ${strategy.current_price:,.2f}")
        print(f"  Optimized Price: ${strategy.optimized_price:,.2f}")
        print(f"  Change: {strategy.price_change_percentage:+.1f}%")
        print(f"  Expected Revenue Impact: ${strategy.expected_revenue_impact:+,.2f}")
        print(f"  Strategy: {strategy.strategy_rationale}\n")
    
    # Forecast revenue
    forecasts = optimizer.forecast_revenue(6)
    
    print(f"\n=== Revenue Forecasts ===")
    for forecast in forecasts:
        print(f"{forecast.period}: ${forecast.predicted_revenue:,.2f} "
              f"(Growth: {forecast.growth_rate:+.1f}%)")
        print(f"  Confidence: ${forecast.confidence_interval[0]:,.2f} - ${forecast.confidence_interval[1]:,.2f}")
        print(f"  Key Factors: {', '.join(forecast.contributing_factors)}")
        print(f"  Risk: {forecast.risk_assessment}\n")
    
    # Get comprehensive insights
    insights = optimizer.get_revenue_insights()
    print(f"\n=== Revenue Insights Summary ===")
    print(f"Total Revenue Streams: {insights['summary']['total_revenue_streams']}")
    print(f"Key KPIs: {insights['summary']['key_performance_indicators']}")
    print(f"Revenue Trend: {insights['summary']['revenue_trends']['overall_trend']}")