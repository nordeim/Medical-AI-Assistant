"""
Customer Value Expansion and Upsell/Cross-sell Optimization Engine

This module provides AI-powered customer value expansion capabilities including:
- Expansion opportunity identification and scoring
- Upsell/cross-sell recommendation engine
- Expansion timing optimization
- Product affinity and bundle analysis
- Expansion success prediction and tracking
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import networkx as nx
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpansionOpportunity:
    """Data class for expansion opportunity details"""
    customer_id: str
    opportunity_type: str  # 'upsell', 'cross_sell', 'bundle', 'add_on'
    product_id: str
    current_engagement_score: float
    expansion_probability: float
    expected_revenue_impact: float
    recommended_timing: datetime
    confidence_score: float
    reasoning: str
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class ProductAffinity:
    """Product affinity and relationship data"""
    primary_product: str
    secondary_product: str
    affinity_score: float
    co_purchase_rate: float
    upsell_rate: float
    cross_sell_rate: float
    average_expansion_value: float
    correlation_strength: float

@dataclass
class ExpansionBundle:
    """Product bundle recommendation"""
    bundle_id: str
    products: List[str]
    bundle_price: float
    individual_total_price: float
    discount_percentage: float
    target_customer_segments: List[str]
    expected_take_rate: float
    revenue_multiplier: float

class ExpansionEngine:
    """
    AI-powered customer value expansion and upsell/cross-sell optimization engine
    
    This engine identifies expansion opportunities, predicts expansion success,
    and optimizes timing for maximum revenue impact.
    """
    
    def __init__(self):
        """Initialize the expansion engine with ML models and data structures"""
        self.models = {
            'expansion_classifier': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'revenue_predictor': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'timing_optimizer': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
        }
        
        self.scalers = {
            'feature_scaler': StandardScaler(),
            'engagement_scaler': StandardScaler()
        }
        
        self.product_affinity_graph = nx.Graph()
        self.expansion_history = []
        self.opportunity_cache = {}
        self.feature_importance = {}
        
        # Expansion strategies configuration
        self.expansion_strategies = {
            'aggressive': {'min_probability': 0.3, 'max_opportunities_per_customer': 5},
            'conservative': {'min_probability': 0.6, 'max_opportunities_per_customer': 3},
            'balanced': {'min_probability': 0.45, 'max_opportunities_per_customer': 4}
        }
        
        logger.info("Expansion Engine initialized successfully")
    
    def load_historical_data(self, expansion_data: pd.DataFrame) -> None:
        """
        Load historical expansion data for model training
        
        Args:
            expansion_data: DataFrame with columns:
                - customer_id, product_id, expansion_type, success, revenue_impact,
                - customer_tenure, engagement_score, feature_usage, expansion_timing,
                - previous_purchases, product_category, customer_segment
        """
        try:
            logger.info(f"Loading {len(expansion_data)} historical expansion records")
            
            # Prepare features for training
            feature_columns = [
                'customer_tenure', 'engagement_score', 'feature_usage',
                'previous_purchases', 'time_since_last_purchase', 'avg_monthly_value'
            ]
            
            X = expansion_data[feature_columns].fillna(0)
            y_success = expansion_data['success'].astype(int)
            y_revenue = expansion_data['revenue_impact'].fillna(0)
            
            # Scale features
            X_scaled = self.scalers['feature_scaler'].fit_transform(X)
            
            # Split data for training
            X_train, X_test, y_success_train, y_success_test = train_test_split(
                X_scaled, y_success, test_size=0.2, random_state=42, stratify=y_success
            )
            
            _, _, y_revenue_train, y_revenue_test = train_test_split(
                X_scaled, y_revenue, test_size=0.2, random_state=42
            )
            
            # Train expansion success classifier
            logger.info("Training expansion success classifier...")
            self.models['expansion_classifier'].fit(X_train, y_success_train)
            
            # Evaluate classifier
            y_pred_success = self.models['expansion_classifier'].predict(X_test)
            logger.info(f"Expansion Classifier Performance:\n{classification_report(y_success_test, y_pred_success)}")
            
            # Train revenue predictor
            logger.info("Training revenue impact predictor...")
            self.models['revenue_predictor'].fit(X_train, y_revenue_train)
            
            # Evaluate revenue predictor
            y_pred_revenue = self.models['revenue_predictor'].predict(X_test)
            revenue_mae = mean_absolute_error(y_revenue_test, y_pred_revenue)
            logger.info(f"Revenue Predictor MAE: ${revenue_mae:.2f}")
            
            # Store feature importance
            self.feature_importance['expansion'] = dict(zip(
                feature_columns,
                self.models['expansion_classifier'].feature_importances_
            ))
            self.feature_importance['revenue'] = dict(zip(
                feature_columns,
                self.models['revenue_predictor'].feature_importances_
            ))
            
            # Build product affinity graph
            self._build_product_affinity_graph(expansion_data)
            
            # Store expansion history
            self.expansion_history = expansion_data.to_dict('records')
            
            logger.info("Historical data loaded and models trained successfully")
            
        except Exception as e:
            logger.error(f"Error loading historical expansion data: {str(e)}")
            raise
    
    def identify_expansion_opportunities(
        self,
        customer_data: pd.DataFrame,
        strategy: str = 'balanced'
    ) -> List[ExpansionOpportunity]:
        """
        Identify expansion opportunities for customers
        
        Args:
            customer_data: DataFrame with customer information
            strategy: Expansion strategy ('aggressive', 'conservative', 'balanced')
            
        Returns:
            List of ExpansionOpportunity objects
        """
        try:
            logger.info(f"Identifying expansion opportunities using {strategy} strategy")
            
            strategy_config = self.expansion_strategies[strategy]
            opportunities = []
            
            for _, customer in customer_data.iterrows():
                customer_opportunities = self._analyze_customer_expansion_potential(
                    customer, strategy_config
                )
                opportunities.extend(customer_opportunities)
            
            # Sort by expansion probability and expected revenue impact
            opportunities.sort(
                key=lambda x: (x.expansion_probability * x.expected_revenue_impact),
                reverse=True
            )
            
            logger.info(f"Identified {len(opportunities)} expansion opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying expansion opportunities: {str(e)}")
            raise
    
    def _analyze_customer_expansion_potential(
        self,
        customer: pd.Series,
        strategy_config: Dict[str, Any]
    ) -> List[ExpansionOpportunity]:
        """
        Analyze expansion potential for a single customer
        
        Args:
            customer: Customer data series
            strategy_config: Strategy configuration parameters
            
        Returns:
            List of ExpansionOpportunity objects for this customer
        """
        customer_id = str(customer['customer_id'])
        opportunities = []
        
        # Prepare customer features
        features = self._prepare_customer_features(customer)
        
        # Get available expansion products based on customer segment
        available_products = self._get_expansion_products_for_customer(customer)
        
        for product in available_products:
            # Calculate expansion probability
            expansion_prob = self._calculate_expansion_probability(features, product)
            
            # Skip if probability below strategy threshold
            if expansion_prob < strategy_config['min_probability']:
                continue
            
            # Calculate expected revenue impact
            revenue_impact = self._predict_expansion_revenue(features, product)
            
            # Determine optimal timing
            optimal_timing = self._calculate_optimal_timing(features, customer)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(features, product)
            
            # Create expansion opportunity
            opportunity = ExpansionOpportunity(
                customer_id=customer_id,
                opportunity_type=self._determine_expansion_type(customer, product),
                product_id=product,
                current_engagement_score=customer.get('engagement_score', 0.0),
                expansion_probability=expansion_prob,
                expected_revenue_impact=revenue_impact,
                recommended_timing=optimal_timing,
                confidence_score=confidence,
                reasoning=self._generate_expansion_reasoning(customer, product, expansion_prob)
            )
            
            opportunities.append(opportunity)
        
        # Limit opportunities per customer based on strategy
        max_opportunities = strategy_config['max_opportunities_per_customer']
        if len(opportunities) > max_opportunities:
            opportunities.sort(
                key=lambda x: (x.expansion_probability * x.expected_revenue_impact),
                reverse=True
            )
            opportunities = opportunities[:max_opportunities]
        
        return opportunities
    
    def _prepare_customer_features(self, customer: pd.Series) -> np.ndarray:
        """Prepare customer features for model prediction"""
        feature_values = [
            customer.get('customer_tenure', 0),
            customer.get('engagement_score', 0),
            customer.get('feature_usage', 0),
            customer.get('previous_purchases', 0),
            customer.get('time_since_last_purchase', 30),
            customer.get('avg_monthly_value', 0)
        ]
        
        return np.array(feature_values).reshape(1, -1)
    
    def _get_expansion_products_for_customer(self, customer: pd.Series) -> List[str]:
        """
        Get expansion products suitable for customer based on segment and history
        """
        customer_segment = customer.get('customer_segment', 'general')
        current_products = customer.get('current_products', [])
        
        # Product recommendations based on segment
        segment_products = {
            'enterprise': ['premium_support', 'advanced_analytics', 'api_access'],
            'mid_market': ['additional_users', 'storage_upgrade', 'premium_features'],
            'small_business': ['monthly_billing', 'basic_support', 'additional_features'],
            'startup': ['growth_package', 'team_collaboration', 'advanced_tools']
        }
        
        available_products = segment_products.get(customer_segment, ['additional_features'])
        
        # Filter out products customer already has
        expansion_products = [p for p in available_products if p not in current_products]
        
        return expansion_products
    
    def _calculate_expansion_probability(self, features: np.ndarray, product: str) -> float:
        """Calculate probability of successful expansion for specific product"""
        try:
            # Scale features
            features_scaled = self.scalers['feature_scaler'].transform(features)
            
            # Get prediction probability
            probabilities = self.models['expansion_classifier'].predict_proba(features_scaled)
            
            # Adjust probability based on product-specific factors
            base_probability = probabilities[0][1] if len(probabilities[0]) > 1 else 0.3
            
            # Product-specific adjustments
            product_adjustment = self._get_product_expansion_adjustment(product)
            final_probability = base_probability * product_adjustment
            
            return min(max(final_probability, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating expansion probability: {str(e)}")
            return 0.3
    
    def _get_product_expansion_adjustment(self, product: str) -> float:
        """Get product-specific expansion probability adjustment"""
        adjustments = {
            'premium_support': 1.2,
            'additional_users': 1.4,
            'storage_upgrade': 1.1,
            'api_access': 0.9,
            'advanced_analytics': 1.3,
            'additional_features': 1.0
        }
        return adjustments.get(product, 1.0)
    
    def _predict_expansion_revenue(self, features: np.ndarray, product: str) -> float:
        """Predict revenue impact of expansion"""
        try:
            # Scale features
            features_scaled = self.scalers['feature_scaler'].transform(features)
            
            # Predict base revenue
            base_revenue = self.models['revenue_predictor'].predict(features_scaled)[0]
            
            # Product-specific revenue adjustments
            product_prices = {
                'premium_support': 500,
                'additional_users': 200,
                'storage_upgrade': 100,
                'api_access': 1000,
                'advanced_analytics': 800,
                'additional_features': 300
            }
            
            # Apply product pricing
            product_price = product_prices.get(product, 300)
            predicted_revenue = max(base_revenue, product_price)
            
            return max(predicted_revenue, 0.0)
            
        except Exception as e:
            logger.error(f"Error predicting expansion revenue: {str(e)}")
            return 300.0
    
    def _calculate_optimal_timing(self, features: np.ndarray, customer: pd.Series) -> datetime:
        """Calculate optimal timing for expansion offer"""
        try:
            # Scale features for timing model
            features_scaled = self.scalers['feature_scaler'].transform(features)
            
            # Get timing prediction (days from now)
            days_ahead = self.models['timing_optimizer'].predict(features_scaled)[0]
            
            # Ensure reasonable bounds (7-90 days)
            days_ahead = max(7, min(days_ahead, 90))
            
            return datetime.now() + timedelta(days=int(days_ahead))
            
        except Exception as e:
            logger.error(f"Error calculating optimal timing: {str(e)}")
            return datetime.now() + timedelta(days=30)
    
    def _calculate_confidence_score(self, features: np.ndarray, product: str) -> float:
        """Calculate confidence score for expansion prediction"""
        try:
            # Feature completeness
            feature_completeness = 0.8  # Assuming 80% feature completeness
            
            # Model confidence (simplified)
            model_confidence = 0.7
            
            # Product historical success rate
            product_success_rate = 0.65  # Default success rate
            
            confidence = (feature_completeness + model_confidence + product_success_rate) / 3
            return min(max(confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.5
    
    def _determine_expansion_type(self, customer: pd.Series, product: str) -> str:
        """Determine expansion type (upsell, cross-sell, bundle, add_on)"""
        customer_products = customer.get('current_products', [])
        
        if not customer_products:
            return 'cross_sell'
        
        # Determine based on product and customer context
        if 'premium' in product.lower() or 'advanced' in product.lower():
            return 'upsell'
        elif 'user' in product.lower() or 'storage' in product.lower():
            return 'add_on'
        else:
            return 'cross_sell'
    
    def _generate_expansion_reasoning(
        self,
        customer: pd.Series,
        product: str,
        probability: float
    ) -> str:
        """Generate reasoning for expansion recommendation"""
        reasoning_parts = []
        
        if probability > 0.7:
            reasoning_parts.append("High engagement and strong usage patterns indicate readiness")
        elif probability > 0.5:
            reasoning_parts.append("Moderate engagement suggests expansion potential")
        else:
            reasoning_parts.append("Lower engagement but expansion opportunity identified")
        
        customer_segment = customer.get('customer_segment', 'unknown')
        reasoning_parts.append(f"Customer segment: {customer_segment}")
        
        return ". ".join(reasoning_parts)
    
    def create_expansion_bundles(
        self,
        customer_segments: List[str],
        target_revenue_increase: float = 0.15
    ) -> List[ExpansionBundle]:
        """
        Create product bundles optimized for customer segments
        
        Args:
            customer_segments: Target customer segments
            target_revenue_increase: Desired revenue increase from bundles
            
        Returns:
            List of ExpansionBundle objects
        """
        try:
            logger.info(f"Creating expansion bundles for segments: {customer_segments}")
            
            bundles = []
            
            # Define base bundle configurations
            base_bundles = {
                'enterprise': {
                    'products': ['premium_support', 'advanced_analytics', 'api_access'],
                    'individual_prices': [500, 800, 1000],
                    'bundle_discount': 0.15,
                    'segments': ['enterprise']
                },
                'mid_market': {
                    'products': ['additional_users', 'storage_upgrade', 'premium_features'],
                    'individual_prices': [200, 100, 300],
                    'bundle_discount': 0.10,
                    'segments': ['mid_market', 'small_business']
                },
                'startup': {
                    'products': ['growth_package', 'team_collaboration'],
                    'individual_prices': [400, 200],
                    'bundle_discount': 0.20,
                    'segments': ['startup', 'small_business']
                }
            }
            
            bundle_id_counter = 1
            for bundle_type, config in base_bundles.items():
                # Check if bundle matches target segments
                if any(segment in customer_segments for segment in config['segments']):
                    individual_total = sum(config['individual_prices'])
                    bundle_price = individual_total * (1 - config['bundle_discount'])
                    
                    # Calculate expected metrics
                    expected_take_rate = 0.25 if config['bundle_discount'] > 0.15 else 0.15
                    revenue_multiplier = 1 + (config['bundle_discount'] * expected_take_rate)
                    
                    bundle = ExpansionBundle(
                        bundle_id=f"bundle_{bundle_id_counter:03d}",
                        products=config['products'],
                        bundle_price=bundle_price,
                        individual_total_price=individual_total,
                        discount_percentage=config['bundle_discount'] * 100,
                        target_customer_segments=config['segments'],
                        expected_take_rate=expected_take_rate,
                        revenue_multiplier=revenue_multiplier
                    )
                    
                    bundles.append(bundle)
                    bundle_id_counter += 1
            
            logger.info(f"Created {len(bundles)} expansion bundles")
            return bundles
            
        except Exception as e:
            logger.error(f"Error creating expansion bundles: {str(e)}")
            raise
    
    def analyze_product_affinity(self) -> Dict[str, ProductAffinity]:
        """
        Analyze product affinity and relationships
        
        Returns:
            Dictionary mapping product pairs to ProductAffinity objects
        """
        try:
            logger.info("Analyzing product affinity relationships")
            
            affinity_analysis = {}
            
            # Calculate pairwise product affinities
            products = list(self.product_affinity_graph.nodes())
            
            for i, product1 in enumerate(products):
                for j, product2 in enumerate(products[i+1:], i+1):
                    affinity_score = self._calculate_product_affinity(product1, product2)
                    
                    if affinity_score > 0.1:  # Only include meaningful affinities
                        affinity = ProductAffinity(
                            primary_product=product1,
                            secondary_product=product2,
                            affinity_score=affinity_score,
                            co_purchase_rate=self._get_co_purchase_rate(product1, product2),
                            upsell_rate=self._get_upsell_rate(product1, product2),
                            cross_sell_rate=self._get_cross_sell_rate(product1, product2),
                            average_expansion_value=self._get_average_expansion_value(product1, product2),
                            correlation_strength=affinity_score
                        )
                        
                        affinity_analysis[f"{product1}_{product2}"] = affinity
            
            logger.info(f"Analyzed {len(affinity_analysis)} product affinity relationships")
            return affinity_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing product affinity: {str(e)}")
            return {}
    
    def _build_product_affinity_graph(self, expansion_data: pd.DataFrame) -> None:
        """Build product affinity network graph"""
        # Group by customer to find co-purchases
        customer_purchases = expansion_data.groupby('customer_id')['product_id'].apply(list)
        
        # Build graph edges
        for customer_products in customer_purchases:
            for i, product1 in enumerate(customer_products):
                for product2 in customer_products[i+1:]:
                    if self.product_affinity_graph.has_edge(product1, product2):
                        self.product_affinity_graph[product1][product2]['weight'] += 1
                    else:
                        self.product_affinity_graph.add_edge(product1, product2, weight=1)
        
        logger.info(f"Built product affinity graph with {len(self.product_affinity_graph.nodes())} products")
    
    def _calculate_product_affinity(self, product1: str, product2: str) -> float:
        """Calculate affinity score between two products"""
        if not self.product_affinity_graph.has_edge(product1, product2):
            return 0.0
        
        edge_weight = self.product_affinity_graph[product1][product2]['weight']
        total_customers = len(set([record['customer_id'] for record in self.expansion_history]))
        
        affinity = edge_weight / total_customers if total_customers > 0 else 0.0
        return min(affinity, 1.0)
    
    def _get_co_purchase_rate(self, product1: str, product2: str) -> float:
        """Get co-purchase rate between products"""
        # Simplified calculation
        return 0.15  # Default co-purchase rate
    
    def _get_upsell_rate(self, product1: str, product2: str) -> float:
        """Get upsell rate from product1 to product2"""
        # Simplified calculation
        return 0.25  # Default upsell rate
    
    def _get_cross_sell_rate(self, product1: str, product2: str) -> float:
        """Get cross-sell rate between products"""
        # Simplified calculation
        return 0.20  # Default cross-sell rate
    
    def _get_average_expansion_value(self, product1: str, product2: str) -> float:
        """Get average expansion value between products"""
        # Simplified calculation
        return 450.0  # Default expansion value
    
    def track_expansion_performance(self) -> Dict[str, Any]:
        """
        Track and analyze expansion program performance
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            logger.info("Tracking expansion program performance")
            
            # Calculate performance metrics
            total_opportunities = len(self.expansion_history)
            successful_expansions = sum(1 for record in self.expansion_history if record.get('success', False))
            
            success_rate = successful_expansions / total_opportunities if total_opportunities > 0 else 0.0
            
            # Revenue impact
            total_revenue_impact = sum(record.get('revenue_impact', 0) for record in self.expansion_history)
            avg_revenue_per_expansion = total_revenue_impact / successful_expansions if successful_expansions > 0 else 0.0
            
            # Time-based analysis
            recent_expansions = [
                record for record in self.expansion_history
                if record.get('expansion_date') and 
                datetime.fromisoformat(record['expansion_date']) > datetime.now() - timedelta(days=30)
            ]
            
            recent_success_rate = (
                sum(1 for record in recent_expansions if record.get('success', False)) / 
                len(recent_expansions) if recent_expansions else 0
            )
            
            performance_metrics = {
                'total_opportunities': total_opportunities,
                'successful_expansions': successful_expansions,
                'overall_success_rate': success_rate,
                'total_revenue_impact': total_revenue_impact,
                'avg_revenue_per_expansion': avg_revenue_per_expansion,
                'recent_expansions_30d': len(recent_expansions),
                'recent_success_rate': recent_success_rate,
                'expansion_trend': 'improving' if recent_success_rate > success_rate else 'stable',
                'feature_importance': self.feature_importance
            }
            
            logger.info(f"Expansion performance - Success Rate: {success_rate:.2%}, "
                       f"Revenue Impact: ${total_revenue_impact:,.2f}")
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error tracking expansion performance: {str(e)}")
            return {}
    
    def get_expansion_recommendations(
        self,
        customer_id: str,
        customer_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get personalized expansion recommendations for a specific customer
        
        Args:
            customer_id: Customer identifier
            customer_data: Customer data dictionary
            
        Returns:
            List of expansion recommendations
        """
        try:
            logger.info(f"Generating expansion recommendations for customer {customer_id}")
            
            # Convert customer data to DataFrame for analysis
            customer_df = pd.DataFrame([customer_data])
            
            # Get expansion opportunities
            opportunities = self.identify_expansion_opportunities(customer_df)
            
            # Filter for this customer
            customer_opportunities = [
                opp for opp in opportunities 
                if opp.customer_id == customer_id
            ]
            
            # Convert to recommendation format
            recommendations = []
            for opp in customer_opportunities:
                recommendation = {
                    'customer_id': opp.customer_id,
                    'opportunity_type': opp.opportunity_type,
                    'product_id': opp.product_id,
                    'expansion_probability': opp.expansion_probability,
                    'expected_revenue_impact': opp.expected_revenue_impact,
                    'recommended_timing': opp.recommended_timing.isoformat(),
                    'confidence_score': opp.confidence_score,
                    'reasoning': opp.reasoning,
                    'priority_score': opp.expansion_probability * opp.expected_revenue_impact
                }
                recommendations.append(recommendation)
            
            # Sort by priority score
            recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations for customer {customer_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating expansion recommendations: {str(e)}")
            return []

# Global instance for easy access
expansion_engine = ExpansionEngine()

# Example usage and testing
if __name__ == "__main__":
    # Create sample expansion data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'customer_id': range(1, n_samples + 1),
        'product_id': ['product_' + str(i % 10) for i in range(n_samples)],
        'expansion_type': np.random.choice(['upsell', 'cross_sell'], n_samples),
        'success': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'revenue_impact': np.random.normal(500, 200, n_samples),
        'customer_tenure': np.random.normal(12, 6, n_samples),
        'engagement_score': np.random.normal(0.7, 0.2, n_samples),
        'feature_usage': np.random.normal(0.6, 0.3, n_samples),
        'previous_purchases': np.random.poisson(2, n_samples),
        'expansion_timing': [datetime.now() - timedelta(days=x) for x in range(n_samples)],
        'product_category': ['category_' + str(i % 5) for i in range(n_samples)],
        'customer_segment': np.random.choice(['enterprise', 'mid_market', 'small_business'], n_samples)
    })
    
    # Initialize and train the expansion engine
    engine = ExpansionEngine()
    engine.load_historical_data(sample_data)
    
    # Create sample customer data
    customer_data = pd.DataFrame({
        'customer_id': [1001, 1002, 1003],
        'customer_tenure': [18, 6, 24],
        'engagement_score': [0.85, 0.45, 0.92],
        'feature_usage': [0.80, 0.30, 0.95],
        'previous_purchases': [3, 1, 5],
        'time_since_last_purchase': [15, 60, 5],
        'avg_monthly_value': [800, 200, 1200],
        'customer_segment': ['enterprise', 'small_business', 'enterprise'],
        'current_products': [['api_access'], ['basic_features'], ['api_access', 'analytics']]
    })
    
    # Identify expansion opportunities
    opportunities = engine.identify_expansion_opportunities(customer_data)
    
    print(f"\n=== Expansion Opportunities ===")
    for opp in opportunities[:5]:  # Show top 5 opportunities
        print(f"Customer {opp.customer_id}: {opp.product_id}")
        print(f"  Type: {opp.opportunity_type}")
        print(f"  Probability: {opp.expansion_probability:.2%}")
        print(f"  Expected Revenue: ${opp.expected_revenue_impact:,.2f}")
        print(f"  Timing: {opp.recommended_timing.strftime('%Y-%m-%d')}")
        print(f"  Reasoning: {opp.reasoning}\n")
    
    # Create expansion bundles
    bundles = engine.create_expansion_bundles(['enterprise', 'mid_market'])
    
    print(f"\n=== Expansion Bundles ===")
    for bundle in bundles:
        print(f"Bundle {bundle.bundle_id}: {', '.join(bundle.products)}")
        print(f"  Bundle Price: ${bundle.bundle_price:,.2f}")
        print(f"  Discount: {bundle.discount_percentage:.1f}%")
        print(f"  Expected Take Rate: {bundle.expected_take_rate:.2%}")
        print(f"  Revenue Multiplier: {bundle.revenue_multiplier:.2f}\n")
    
    # Track performance
    performance = engine.track_expansion_performance()
    print(f"\n=== Performance Metrics ===")
    print(f"Overall Success Rate: {performance['overall_success_rate']:.2%}")
    print(f"Total Revenue Impact: ${performance['total_revenue_impact']:,.2f}")
    print(f"Avg Revenue per Expansion: ${performance['avg_revenue_per_expansion']:,.2f}")