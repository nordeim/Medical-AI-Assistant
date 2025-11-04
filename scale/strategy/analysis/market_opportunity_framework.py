"""
Market Opportunity Identification and Prioritization Framework
Comprehensive market analysis and opportunity assessment system
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class OpportunityType(Enum):
    MARKET_EXPANSION = "market_expansion"
    PRODUCT_INNOVATION = "product_innovation"
    CUSTOMER_SEGMENT = "customer_segment"
    PARTNERSHIP = "partnership"
    DIGITAL_TRANSFORMATION = "digital_transformation"
    ACQUISITION = "acquisition"
    NEW_GEOGRAPHY = "new_geography"
    STRATEGIC_ALLIANCE = "strategic_alliance"

class OpportunityStage(Enum):
    IDENTIFIED = "identified"
    QUALIFIED = "qualified"
    ASSESSED = "assessed"
    PRIORITIZED = "prioritized"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class MarketSegment(Enum):
    ENTERPRISE = "enterprise"
    SMB = "smb"
    CONSUMER = "consumer"
    GOVERNMENT = "government"
    NON_PROFIT = "non_profit"

@dataclass
class MarketOpportunity:
    """Market Opportunity Definition"""
    opportunity_id: str
    name: str
    description: str
    opportunity_type: OpportunityType
    market_segment: MarketSegment
    estimated_market_size: float  # in millions
    growth_rate: float  # annual growth percentage
    time_to_market: int  # months
    investment_required: float  # in millions
    expected_roi: float
    strategic_fit: float  # 0-1 scale
    competitive_advantage: float  # 0-1 scale
    risk_level: float  # 0-1 scale
    implementation_complexity: str  # Low, Medium, High
    key_stakeholders: List[str]
    success_factors: List[str]
    assumptions: List[str]
    dependencies: List[str]

@dataclass
class CompetitorAnalysis:
    """Competitor Analysis for Opportunity Assessment"""
    competitor_id: str
    name: str
    market_share: float
    competitive_strengths: List[str]
    competitive_weaknesses: List[str]
    strategy: str
    market_position: str
    innovation_capability: float
    financial_strength: float

@dataclass
class CustomerInsight:
    """Customer Insight for Market Understanding"""
    insight_id: str
    customer_segment: str
    pain_points: List[str]
    unmet_needs: List[str]
    buying_behavior: str
    price_sensitivity: float
    decision_factors: List[str]
    satisfaction_level: float

class MarketOpportunityFramework:
    """
    Market Opportunity Identification and Prioritization Framework
    """
    
    def __init__(self, organization_name: str):
        self.organization_name = organization_name
        self.market_opportunities: List[MarketOpportunity] = []
        self.competitor_analysis: List[CompetitorAnalysis] = []
        self.customer_insights: List[CustomerInsight] = []
        self.market_trends: Dict[str, Any] = {}
        self.prioritization_scores: Dict[str, float] = {}
        self.opportunity_portfolio: Dict[str, Any] = {}
        
    def add_market_opportunity(self,
                             opportunity_id: str,
                             name: str,
                             description: str,
                             opportunity_type: OpportunityType,
                             market_segment: MarketSegment,
                             estimated_market_size: float,
                             growth_rate: float,
                             time_to_market: int,
                             investment_required: float,
                             expected_roi: float,
                             strategic_fit: float,
                             competitive_advantage: float,
                             risk_level: float,
                             implementation_complexity: str,
                             key_stakeholders: List[str],
                             success_factors: List[str],
                             assumptions: List[str],
                             dependencies: List[str]) -> MarketOpportunity:
        """Add market opportunity to analysis"""
        
        opportunity = MarketOpportunity(
            opportunity_id=opportunity_id,
            name=name,
            description=description,
            opportunity_type=opportunity_type,
            market_segment=market_segment,
            estimated_market_size=estimated_market_size,
            growth_rate=growth_rate,
            time_to_market=time_to_market,
            investment_required=investment_required,
            expected_roi=expected_roi,
            strategic_fit=strategic_fit,
            competitive_advantage=competitive_advantage,
            risk_level=risk_level,
            implementation_complexity=implementation_complexity,
            key_stakeholders=key_stakeholders,
            success_factors=success_factors,
            assumptions=assumptions,
            dependencies=dependencies
        )
        
        self.market_opportunities.append(opportunity)
        return opportunity
    
    def add_competitor_analysis(self,
                               competitor_id: str,
                               name: str,
                               market_share: float,
                               competitive_strengths: List[str],
                               competitive_weaknesses: List[str],
                               strategy: str,
                               market_position: str,
                               innovation_capability: float,
                               financial_strength: float) -> CompetitorAnalysis:
        """Add competitor analysis"""
        
        competitor = CompetitorAnalysis(
            competitor_id=competitor_id,
            name=name,
            market_share=market_share,
            competitive_strengths=competitive_strengths,
            competitive_weaknesses=competitive_weaknesses,
            strategy=strategy,
            market_position=market_position,
            innovation_capability=innovation_capability,
            financial_strength=financial_strength
        )
        
        self.competitor_analysis.append(competitor)
        return competitor
    
    def add_customer_insight(self,
                           insight_id: str,
                           customer_segment: str,
                           pain_points: List[str],
                           unmet_needs: List[str],
                           buying_behavior: str,
                           price_sensitivity: float,
                           decision_factors: List[str],
                           satisfaction_level: float) -> CustomerInsight:
        """Add customer insight"""
        
        insight = CustomerInsight(
            insight_id=insight_id,
            customer_segment=customer_segment,
            pain_points=pain_points,
            unmet_needs=unmet_needs,
            buying_behavior=buying_behavior,
            price_sensitivity=price_sensitivity,
            decision_factors=decision_factors,
            satisfaction_level=satisfaction_level
        )
        
        self.customer_insights.append(insight)
        return insight
    
    def identify_market_opportunities(self,
                                    market_data: Dict[str, Any],
                                    trend_analysis: Dict[str, Any]) -> List[MarketOpportunity]:
        """Identify market opportunities based on market data and trends"""
        
        # Analyze market trends for opportunity signals
        opportunity_signals = self._analyze_market_signals(market_data, trend_analysis)
        
        # Generate opportunity hypotheses
        opportunity_hypotheses = self._generate_opportunity_hypotheses(opportunity_signals)
        
        # Validate and refine opportunities
        validated_opportunities = self._validate_opportunities(opportunity_hypotheses)
        
        return validated_opportunities
    
    def _analyze_market_signals(self, market_data: Dict[str, Any], trend_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze market signals to identify opportunity patterns"""
        
        signals = []
        
        # Growth opportunities
        if "growth_indicators" in market_data:
            for indicator, value in market_data["growth_indicators"].items():
                if value > 0.15:  # 15% growth threshold
                    signals.append({
                        "signal_type": "growth",
                        "indicator": indicator,
                        "value": value,
                        "opportunity_implication": f"High growth in {indicator}",
                        "opportunity_type": OpportunityType.MARKET_EXPANSION
                    })
        
        # Technology disruption opportunities
        if "technological_trends" in trend_analysis:
            for trend in trend_analysis["technological_trends"]:
                if trend.get("adoption_rate", 0) > 0.6:
                    signals.append({
                        "signal_type": "technology",
                        "indicator": trend["name"],
                        "value": trend["adoption_rate"],
                        "opportunity_implication": f"Technology disruption in {trend['name']}",
                        "opportunity_type": OpportunityType.DIGITAL_TRANSFORMATION
                    })
        
        # Customer unmet needs opportunities
        if "customer_feedback" in market_data:
            for feedback in market_data["customer_feedback"]:
                if feedback.get("satisfaction_gap", 0) > 0.3:
                    signals.append({
                        "signal_type": "customer_need",
                        "indicator": feedback["area"],
                        "value": feedback["satisfaction_gap"],
                        "opportunity_implication": f"Unmet customer need in {feedback['area']}",
                        "opportunity_type": OpportunityType.PRODUCT_INNOVATION
                    })
        
        return signals
    
    def _generate_opportunity_hypotheses(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate opportunity hypotheses from market signals"""
        
        hypotheses = []
        
        for signal in signals:
            hypothesis = {
                "signal_source": signal,
                "hypothesis_statement": f"Opportunity exists to {signal['opportunity_implication']}",
                "market_size_estimate": self._estimate_market_size(signal),
                "investment_required": self._estimate_investment(signal),
                "time_to_market": self._estimate_time_to_market(signal),
                "expected_roi": self._estimate_roi(signal),
                "validation_required": True
            }
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _estimate_market_size(self, signal: Dict[str, Any]) -> float:
        """Estimate market size based on signal"""
        
        # Base estimates by opportunity type
        size_estimates = {
            OpportunityType.MARKET_EXPANSION: 50.0,  # $50M
            OpportunityType.PRODUCT_INNOVATION: 25.0,  # $25M
            OpportunityType.DIGITAL_TRANSFORMATION: 100.0,  # $100M
            OpportunityType.PARTNERSHIP: 15.0,  # $15M
            OpportunityType.CUSTOMER_SEGMENT: 30.0  # $30M
        }
        
        return size_estimates.get(signal.get("opportunity_type", OpportunityType.MARKET_EXPANSION), 25.0)
    
    def _estimate_investment(self, signal: Dict[str, Any]) -> float:
        """Estimate investment required based on signal"""
        
        # Base investment estimates by opportunity type
        investment_estimates = {
            OpportunityType.MARKET_EXPANSION: 5.0,  # $5M
            OpportunityType.PRODUCT_INNOVATION: 3.0,  # $3M
            OpportunityType.DIGITAL_TRANSFORMATION: 10.0,  # $10M
            OpportunityType.PARTNERSHIP: 1.0,  # $1M
            OpportunityType.CUSTOMER_SEGMENT: 2.0  # $2M
        }
        
        return investment_estimates.get(signal.get("opportunity_type", OpportunityType.MARKET_EXPANSION), 3.0)
    
    def _estimate_time_to_market(self, signal: Dict[str, Any]) -> int:
        """Estimate time to market based on signal"""
        
        # Time estimates by opportunity type (months)
        time_estimates = {
            OpportunityType.MARKET_EXPANSION: 18,
            OpportunityType.PRODUCT_INNOVATION: 12,
            OpportunityType.DIGITAL_TRANSFORMATION: 24,
            OpportunityType.PARTNERSHIP: 6,
            OpportunityType.CUSTOMER_SEGMENT: 15
        }
        
        return time_estimates.get(signal.get("opportunity_type", OpportunityType.MARKET_EXPANSION), 15)
    
    def _estimate_roi(self, signal: Dict[str, Any]) -> float:
        """Estimate expected ROI based on signal"""
        
        # ROI estimates by opportunity type
        roi_estimates = {
            OpportunityType.MARKET_EXPANSION: 2.5,  # 250%
            OpportunityType.PRODUCT_INNOVATION: 3.0,  # 300%
            OpportunityType.DIGITAL_TRANSFORMATION: 2.0,  # 200%
            OpportunityType.PARTNERSHIP: 1.5,  # 150%
            OpportunityType.CUSTOMER_SEGMENT: 2.2  # 220%
        }
        
        return roi_estimates.get(signal.get("opportunity_type", OpportunityType.MARKET_EXPANSION), 2.5)
    
    def _validate_opportunities(self, hypotheses: List[Dict[str, Any]]) -> List[MarketOpportunity]:
        """Validate and convert hypotheses to validated opportunities"""
        
        validated_opportunities = []
        
        for i, hypothesis in enumerate(hypotheses):
            signal = hypothesis["signal_source"]
            
            # Create validated opportunity
            opportunity = MarketOpportunity(
                opportunity_id=f"opp_{i+1:03d}",
                name=hypothesis["hypothesis_statement"],
                description=f"Validated opportunity based on {signal['signal_type']} signal",
                opportunity_type=signal.get("opportunity_type", OpportunityType.MARKET_EXPANSION),
                market_segment=MarketSegment.ENTERPRISE,  # Default segment
                estimated_market_size=hypothesis["market_size_estimate"],
                growth_rate=signal.get("value", 0.1) * 100,  # Convert to percentage
                time_to_market=hypothesis["time_to_market"],
                investment_required=hypothesis["investment_required"],
                expected_roi=hypothesis["expected_roi"],
                strategic_fit=0.7,  # Default strategic fit
                competitive_advantage=0.6,  # Default competitive advantage
                risk_level=0.4,  # Default risk level
                implementation_complexity="Medium",
                key_stakeholders=["Market team", "Product team"],
                success_factors=["Market demand", "Competitive positioning"],
                assumptions=["Market growth continues", "Technology adoption accelerates"],
                dependencies=["Resource availability", "Partnership agreements"]
            )
            
            validated_opportunities.append(opportunity)
            self.market_opportunities.append(opportunity)
        
        return validated_opportunities
    
    def prioritize_opportunities(self,
                               opportunity_ids: Optional[List[str]] = None,
                               criteria_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Prioritize opportunities using multi-criteria analysis"""
        
        if criteria_weights is None:
            criteria_weights = {
                "market_size": 0.25,
                "growth_rate": 0.20,
                "roi": 0.20,
                "strategic_fit": 0.15,
                "competitive_advantage": 0.10,
                "risk_adjusted_score": 0.10
            }
        
        if opportunity_ids is None:
            opportunities = self.market_opportunities
        else:
            opportunities = [opp for opp in self.market_opportunities if opp.opportunity_id in opportunity_ids]
        
        priority_scores = {}
        
        for opportunity in opportunities:
            # Calculate weighted score
            score = (
                criteria_weights["market_size"] * (opportunity.estimated_market_size / 100.0) +  # Normalize to 0-1
                criteria_weights["growth_rate"] * (opportunity.growth_rate / 100.0) +  # Normalize to 0-1
                criteria_weights["roi"] * min(opportunity.expected_roi / 5.0, 1.0) +  # Cap at 500%
                criteria_weights["strategic_fit"] * opportunity.strategic_fit +
                criteria_weights["competitive_advantage"] * opportunity.competitive_advantage +
                criteria_weights["risk_adjusted_score"] * (1.0 - opportunity.risk_level)
            )
            
            priority_scores[opportunity.opportunity_id] = score
        
        # Sort by priority score
        sorted_priorities = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.prioritization_scores = dict(sorted_priorities)
        return self.prioritization_scores
    
    def conduct_portfolio_analysis(self) -> Dict[str, Any]:
        """Conduct comprehensive portfolio analysis"""
        
        if not self.market_opportunities:
            return {"error": "No opportunities in portfolio"}
        
        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics()
        
        # Analyze opportunity types
        type_analysis = self._analyze_by_opportunity_type()
        
        # Analyze market segments
        segment_analysis = self._analyze_by_market_segment()
        
        # Risk-return analysis
        risk_return_analysis = self._conduct_risk_return_analysis()
        
        # Portfolio optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations()
        
        portfolio_analysis = {
            "portfolio_overview": {
                "total_opportunities": len(self.market_opportunities),
                "total_market_size": sum(opp.estimated_market_size for opp in self.market_opportunities),
                "total_investment_required": sum(opp.investment_required for opp in self.market_opportunities),
                "weighted_average_roi": np.mean([opp.expected_roi for opp in self.market_opportunities]),
                "portfolio_diversification_score": self._calculate_diversification_score()
            },
            "portfolio_metrics": portfolio_metrics,
            "type_analysis": type_analysis,
            "segment_analysis": segment_analysis,
            "risk_return_analysis": risk_return_analysis,
            "optimization_recommendations": optimization_recommendations
        }
        
        self.opportunity_portfolio = portfolio_analysis
        return portfolio_analysis
    
    def _calculate_portfolio_metrics(self) -> Dict[str, float]:
        """Calculate key portfolio metrics"""
        
        if not self.market_opportunities:
            return {}
        
        metrics = {
            "average_market_size": np.mean([opp.estimated_market_size for opp in self.market_opportunities]),
            "average_growth_rate": np.mean([opp.growth_rate for opp in self.market_opportunities]),
            "average_time_to_market": np.mean([opp.time_to_market for opp in self.market_opportunities]),
            "average_investment": np.mean([opp.investment_required for opp in self.market_opportunities]),
            "average_roi": np.mean([opp.expected_roi for opp in self.market_opportunities]),
            "portfolio_risk_score": np.mean([opp.risk_level for opp in self.market_opportunities]),
            "strategic_alignment_score": np.mean([opp.strategic_fit for opp in self.market_opportunities])
        }
        
        return metrics
    
    def _analyze_by_opportunity_type(self) -> Dict[str, Dict[str, Any]]:
        """Analyze opportunities by type"""
        
        type_analysis = {}
        
        for opportunity_type in OpportunityType:
            type_opportunities = [opp for opp in self.market_opportunities if opp.opportunity_type == opportunity_type]
            
            if type_opportunities:
                analysis = {
                    "count": len(type_opportunities),
                    "total_market_size": sum(opp.estimated_market_size for opp in type_opportunities),
                    "total_investment": sum(opp.investment_required for opp in type_opportunities),
                    "average_roi": np.mean([opp.expected_roi for opp in type_opportunities]),
                    "average_risk": np.mean([opp.risk_level for opp in type_opportunities]),
                    "opportunity_ids": [opp.opportunity_id for opp in type_opportunities]
                }
                type_analysis[opportunity_type.value] = analysis
        
        return type_analysis
    
    def _analyze_by_market_segment(self) -> Dict[str, Dict[str, Any]]:
        """Analyze opportunities by market segment"""
        
        segment_analysis = {}
        
        for market_segment in MarketSegment:
            segment_opportunities = [opp for opp in self.market_opportunities if opp.market_segment == market_segment]
            
            if segment_opportunities:
                analysis = {
                    "count": len(segment_opportunities),
                    "total_market_size": sum(opp.estimated_market_size for opp in segment_opportunities),
                    "total_investment": sum(opp.investment_required for opp in segment_opportunities),
                    "average_roi": np.mean([opp.expected_roi for opp in segment_opportunities]),
                    "segment_growth_rate": np.mean([opp.growth_rate for opp in segment_opportunities]),
                    "opportunity_ids": [opp.opportunity_id for opp in segment_opportunities]
                }
                segment_analysis[market_segment.value] = analysis
        
        return segment_analysis
    
    def _conduct_risk_return_analysis(self) -> Dict[str, Any]:
        """Conduct risk-return analysis of opportunity portfolio"""
        
        if not self.market_opportunities:
            return {}
        
        # Calculate risk-return metrics
        risk_levels = [opp.risk_level for opp in self.market_opportunities]
        roi_values = [opp.expected_roi for opp in self.market_opportunities]
        
        # Correlation analysis
        if len(self.market_opportunities) > 1:
            correlation_matrix = np.corrcoef([risk_levels, roi_values])
            risk_return_correlation = correlation_matrix[0, 1]
        else:
            risk_return_correlation = 0
        
        # Risk categories
        high_risk_opportunities = [opp for opp in self.market_opportunities if opp.risk_level > 0.7]
        medium_risk_opportunities = [opp for opp in self.market_opportunities if 0.3 <= opp.risk_level <= 0.7]
        low_risk_opportunities = [opp for opp in self.market_opportunities if opp.risk_level < 0.3]
        
        # Return categories
        high_return_opportunities = [opp for opp in self.market_opportunities if opp.expected_roi > 3.0]
        medium_return_opportunities = [opp for opp in self.market_opportunities if 1.5 <= opp.expected_roi <= 3.0]
        low_return_opportunities = [opp for opp in self.market_opportunities if opp.expected_roi < 1.5]
        
        return {
            "portfolio_metrics": {
                "average_risk": np.mean(risk_levels),
                "average_return": np.mean(roi_values),
                "risk_return_correlation": risk_return_correlation,
                "risk_adjusted_return": np.mean(roi_values) / (np.mean(risk_levels) + 0.1)  # Avoid division by zero
            },
            "risk_distribution": {
                "high_risk_count": len(high_risk_opportunities),
                "medium_risk_count": len(medium_risk_opportunities),
                "low_risk_count": len(low_risk_opportunities),
                "high_risk_opportunities": [opp.opportunity_id for opp in high_risk_opportunities],
                "medium_risk_opportunities": [opp.opportunity_id for opp in medium_risk_opportunities],
                "low_risk_opportunities": [opp.opportunity_id for opp in low_risk_opportunities]
            },
            "return_distribution": {
                "high_return_count": len(high_return_opportunities),
                "medium_return_count": len(medium_return_opportunities),
                "low_return_count": len(low_return_opportunities),
                "high_return_opportunities": [opp.opportunity_id for opp in high_return_opportunities],
                "medium_return_opportunities": [opp.opportunity_id for opp in medium_return_opportunities],
                "low_return_opportunities": [opp.opportunity_id for opp in low_return_opportunities]
            }
        }
    
    def _generate_optimization_recommendations(self) -> Dict[str, Any]:
        """Generate portfolio optimization recommendations"""
        
        if not self.market_opportunities:
            return {}
        
        recommendations = {
            "portfolio_balance": [],
            "risk_management": [],
            "resource_allocation": [],
            "timing_optimization": []
        }
        
        # Portfolio balance recommendations
        if len(self.market_opportunities) < 5:
            recommendations["portfolio_balance"].append("Increase opportunity diversity - consider more strategic initiatives")
        
        if len(self.market_opportunities) > 20:
            recommendations["portfolio_balance"].append("Consider focusing on highest-priority opportunities")
        
        # Risk management recommendations
        high_risk_count = len([opp for opp in self.market_opportunities if opp.risk_level > 0.7])
        if high_risk_count > len(self.market_opportunities) * 0.3:
            recommendations["risk_management"].append("Consider reducing high-risk opportunities or implementing risk mitigation")
        
        # Resource allocation recommendations
        total_investment = sum(opp.investment_required for opp in self.market_opportunities)
        avg_investment = total_investment / len(self.market_opportunities)
        
        large_investments = [opp for opp in self.market_opportunities if opp.investment_required > avg_investment * 2]
        if large_investments:
            recommendations["resource_allocation"].append("Consider staged investment approach for large opportunities")
        
        # Timing optimization recommendations
        long_ttm_opportunities = [opp for opp in self.market_opportunities if opp.time_to_market > 24]
        if long_ttm_opportunities:
            recommendations["timing_optimization"].append("Prioritize shorter time-to-market opportunities for quick wins")
        
        return recommendations
    
    def _calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score"""
        
        if not self.market_opportunities:
            return 0.0
        
        # Diversity dimensions
        type_diversity = len(set(opp.opportunity_type for opp in self.market_opportunities))
        segment_diversity = len(set(opp.market_segment for opp in self.market_opportunities))
        risk_diversity = len(set("High" if opp.risk_level > 0.7 else "Medium" if opp.risk_level > 0.3 else "Low" 
                               for opp in self.market_opportunities))
        
        # Calculate diversification score (0-1 scale)
        max_types = len(OpportunityType)
        max_segments = len(MarketSegment)
        max_risk_categories = 3
        
        diversification_score = (type_diversity / max_types + 
                               segment_diversity / max_segments + 
                               risk_diversity / max_risk_categories) / 3
        
        return diversification_score
    
    def conduct_competitive_landscape_analysis(self) -> Dict[str, Any]:
        """Conduct competitive landscape analysis"""
        
        if not self.competitor_analysis:
            return {"error": "No competitor analysis data available"}
        
        # Market share analysis
        total_market_share = sum(comp.market_share for comp in self.competitor_analysis)
        
        # Competitive positioning analysis
        competitive_positioning = self._analyze_competitive_positioning()
        
        # Competitive threats assessment
        competitive_threats = self._assess_competitive_threats()
        
        # Strategic gaps identification
        strategic_gaps = self._identify_strategic_gaps()
        
        # Competitive response planning
        response_plans = self._create_competitive_response_plans()
        
        competitive_analysis = {
            "market_overview": {
                "total_market_share_analyzed": total_market_share,
                "competitor_count": len(self.competitor_analysis),
                "market_concentration": self._calculate_market_concentration(),
                "competitive_intensity": self._assess_competitive_intensity()
            },
            "competitive_positioning": competitive_positioning,
            "competitive_threats": competitive_threats,
            "strategic_gaps": strategic_gaps,
            "response_plans": response_plans,
            "opportunity_alignment": self._align_opportunities_with_competitive_analysis()
        }
        
        return competitive_analysis
    
    def _analyze_competitive_positioning(self) -> Dict[str, str]:
        """Analyze competitive positioning of key players"""
        
        positioning = {}
        
        for competitor in self.competitor_analysis:
            # Determine competitive position based on market share and strengths
            if competitor.market_share > 0.25:
                position = "Market Leader"
            elif competitor.market_share > 0.15:
                position = "Strong Challenger"
            elif competitor.market_share > 0.05:
                position = "Established Player"
            else:
                position = "Niche Player"
            
            # Consider innovation and financial strength
            if competitor.innovation_capability > 0.8 and competitor.financial_strength > 0.7:
                position += " - Innovation Leader"
            elif competitor.innovation_capability > 0.6:
                position += " - Innovation Focused"
            
            positioning[competitor.name] = position
        
        return positioning
    
    def _assess_competitive_threats(self) -> Dict[str, List[str]]:
        """Assess competitive threats based on competitor analysis"""
        
        threats = {
            "direct_threats": [],
            "indirect_threats": [],
            "emerging_threats": [],
            "disruptive_threats": []
        }
        
        for competitor in self.competitor_analysis:
            # Direct threats (high market share, strong position)
            if competitor.market_share > 0.20 and competitor.financial_strength > 0.7:
                threats["direct_threats"].append(f"{competitor.name}: Strong market position with financial resources")
            
            # Innovation threats
            if competitor.innovation_capability > 0.8:
                threats["disruptive_threats"].append(f"{competitor.name}: High innovation capability could disrupt market")
            
            # Emerging threats (growing market share or capabilities)
            if competitor.market_share > 0.10 and competitor.innovation_capability > 0.6:
                threats["emerging_threats"].append(f"{competitor.name}: Growing presence with innovation capabilities")
        
        return threats
    
    def _identify_strategic_gaps(self) -> List[str]:
        """Identify strategic gaps in competitive positioning"""
        
        gaps = []
        
        # Innovation gaps
        avg_innovation = np.mean([comp.innovation_capability for comp in self.competitor_analysis])
        if avg_innovation > 0.7:
            gaps.append("Industry innovation levels are high - need to strengthen innovation capabilities")
        
        # Market coverage gaps
        market_segments_covered = set()
        for comp in self.competitor_analysis:
            # Assume different competitors cover different segments
            if "enterprise" in comp.strategy.lower():
                market_segments_covered.add("enterprise")
            if "smb" in comp.strategy.lower():
                market_segments_covered.add("smb")
            if "consumer" in comp.strategy.lower():
                market_segments_covered.add("consumer")
        
        if len(market_segments_covered) > 2:
            gaps.append("Multiple market segments are well-covered by competitors - need differentiation")
        
        # Financial capability gaps
        if any(comp.financial_strength < 0.5 for comp in self.competitor_analysis):
            gaps.append("Some competitors have limited financial resources - opportunity for resource advantage")
        
        return gaps
    
    def _create_competitive_response_plans(self) -> Dict[str, List[str]]:
        """Create competitive response plans"""
        
        response_plans = {
            "immediate_responses": [],
            "medium_term_strategies": [],
            "long_term_positioning": [],
            "defensive_moves": [],
            "offensive_moves": []
        }
        
        # Immediate responses
        response_plans["immediate_responses"] = [
            "Monitor competitive moves and market developments",
            "Accelerate customer relationship building",
            "Strengthen value proposition differentiation"
        ]
        
        # Medium-term strategies
        response_plans["medium_term_strategies"] = [
            "Invest in innovation capabilities",
            "Build strategic partnerships",
            "Expand market coverage strategically"
        ]
        
        # Long-term positioning
        response_plans["long_term_positioning"] = [
            "Establish market leadership in key segments",
            "Create ecosystem and platform advantages",
            "Build sustainable competitive moats"
        ]
        
        return response_plans
    
    def _align_opportunities_with_competitive_analysis(self) -> Dict[str, List[str]]:
        """Align market opportunities with competitive analysis insights"""
        
        alignment = {
            "competitive_mitigation_opportunities": [],
            "competitive_attack_opportunities": [],
            "market_gap_opportunities": [],
            "differentiation_opportunities": []
        }
        
        for opportunity in self.market_opportunities:
            # High competitive advantage opportunities
            if opportunity.competitive_advantage > 0.8:
                alignment["competitive_attack_opportunities"].append(
                    f"{opportunity.name}: Strong competitive advantage for market attack"
                )
            
            # Strategic fit opportunities
            if opportunity.strategic_fit > 0.7:
                alignment["differentiation_opportunities"].append(
                    f"{opportunity.name}: High strategic fit for differentiation"
                )
            
            # Market size opportunities in underserved segments
            if opportunity.estimated_market_size > 50.0:
                alignment["market_gap_opportunities"].append(
                    f"{opportunity.name}: Large market opportunity"
                )
        
        return alignment
    
    def _calculate_market_concentration(self) -> str:
        """Calculate market concentration level"""
        
        if not self.competitor_analysis:
            return "Unknown"
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = sum(comp.market_share**2 for comp in self.competitor_analysis)
        
        if hhi > 0.25:
            return "Highly Concentrated"
        elif hhi > 0.15:
            return "Moderately Concentrated"
        else:
            return "Fragmented"
    
    def _assess_competitive_intensity(self) -> str:
        """Assess competitive intensity level"""
        
        if not self.competitor_analysis:
            return "Unknown"
        
        # Average capabilities as proxy for intensity
        avg_innovation = np.mean([comp.innovation_capability for comp in self.competitor_analysis])
        avg_financial = np.mean([comp.financial_strength for comp in self.competitor_analysis])
        
        intensity_score = (avg_innovation + avg_financial) / 2
        
        if intensity_score > 0.8:
            return "Very High"
        elif intensity_score > 0.6:
            return "High"
        elif intensity_score > 0.4:
            return "Medium"
        else:
            return "Low"
    
    def generate_market_opportunity_report(self) -> Dict[str, Any]:
        """Generate comprehensive market opportunity analysis report"""
        
        return {
            "executive_summary": {
                "organization": self.organization_name,
                "analysis_date": datetime.now().isoformat(),
                "total_opportunities_identified": len(self.market_opportunities),
                "competitors_analyzed": len(self.competitor_analysis),
                "customer_insights_collected": len(self.customer_insights),
                "portfolio_status": "Ready for prioritization"
            },
            "opportunity_analysis": {
                "market_opportunities": [asdict(opp) for opp in self.market_opportunities],
                "prioritization_scores": self.prioritize_opportunities(),
                "portfolio_analysis": self.conduct_portfolio_analysis(),
                "competitive_analysis": self.conduct_competitive_landscape_analysis()
            },
            "strategic_insights": {
                "key_opportunities": self._identify_key_opportunities(),
                "market_trends": self._synthesize_market_trends(),
                "competitive_landscape": self._synthesize_competitive_landscape(),
                "customer_insights": self._synthesize_customer_insights()
            },
            "recommendations": {
                "immediate_priorities": self._generate_immediate_priorities(),
                "portfolio_optimization": self._generate_portfolio_recommendations(),
                "competitive_strategy": self._generate_competitive_strategy(),
                "investment_recommendations": self._generate_investment_recommendations()
            }
        }
    
    def _identify_key_opportunities(self) -> List[Dict[str, Any]]:
        """Identify key strategic opportunities"""
        
        if not self.prioritization_scores:
            self.prioritize_opportunities()
        
        # Get top 5 opportunities
        top_opportunities = sorted(self.prioritization_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        key_opportunities = []
        for opp_id, score in top_opportunities:
            opportunity = next((opp for opp in self.market_opportunities if opp.opportunity_id == opp_id), None)
            if opportunity:
                key_opportunities.append({
                    "opportunity": asdict(opportunity),
                    "priority_score": score,
                    "strategic_importance": "High" if score > 0.7 else "Medium" if score > 0.5 else "Low"
                })
        
        return key_opportunities
    
    def _synthesize_market_trends(self) -> List[str]:
        """Synthesize key market trends"""
        
        trends = []
        
        # Analyze growth trends
        high_growth_opportunities = [opp for opp in self.market_opportunities if opp.growth_rate > 20]
        if high_growth_opportunities:
            trends.append(f"High growth market opportunities: {len(high_growth_opportunities)} opportunities with >20% growth")
        
        # Analyze market size trends
        large_markets = [opp for opp in self.market_opportunities if opp.estimated_market_size > 50]
        if large_markets:
            trends.append(f"Large market opportunities: {len(large_markets)} opportunities with >$50M market size")
        
        # Analyze time to market trends
        quick_wins = [opp for opp in self.market_opportunities if opp.time_to_market < 12]
        if quick_wins:
            trends.append(f"Quick win opportunities: {len(quick_wins)} opportunities with <12 months time to market")
        
        return trends
    
    def _synthesize_competitive_landscape(self) -> Dict[str, Any]:
        """Synthesize competitive landscape insights"""
        
        if not self.competitor_analysis:
            return {}
        
        # Market leader analysis
        market_leaders = [comp for comp in self.competitor_analysis if comp.market_share > 0.20]
        
        # Innovation leaders
        innovation_leaders = [comp for comp in self.competitor_analysis if comp.innovation_capability > 0.8]
        
        return {
            "market_leaders": [comp.name for comp in market_leaders],
            "innovation_leaders": [comp.name for comp in innovation_leaders],
            "market_concentration": self._calculate_market_concentration(),
            "competitive_intensity": self._assess_competitive_intensity()
        }
    
    def _synthesize_customer_insights(self) -> List[str]:
        """Synthesize customer insights"""
        
        insights = []
        
        for insight in self.customer_insights:
            # High satisfaction gaps indicate opportunities
            if insight.satisfaction_level < 0.6:
                insights.append(f"Customer segment '{insight.customer_segment}' has low satisfaction ({insight.satisfaction_level:.2f})")
            
            # High unmet needs indicate innovation opportunities
            if len(insight.unmet_needs) > 3:
                insights.append(f"Customer segment '{insight.customer_segment}' has multiple unmet needs")
        
        return insights
    
    def _generate_immediate_priorities(self) -> List[str]:
        """Generate immediate priority recommendations"""
        
        priorities = []
        
        # Prioritize high-scoring opportunities
        if self.prioritization_scores:
            top_3_opportunities = sorted(self.prioritization_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for opp_id, score in top_3_opportunities:
                opportunity = next((opp for opp in self.market_opportunities if opp.opportunity_id == opp_id), None)
                if opportunity and score > 0.7:
                    priorities.append(f"Prioritize {opportunity.name} (priority score: {score:.2f})")
        
        # Address competitive threats
        if self.competitor_analysis:
            high_threat_competitors = [comp for comp in self.competitor_analysis 
                                     if comp.market_share > 0.15 and comp.innovation_capability > 0.7]
            if high_threat_competitors:
                priorities.append("Monitor and respond to competitive threats from innovation leaders")
        
        return priorities
    
    def _generate_portfolio_recommendations(self) -> List[str]:
        """Generate portfolio optimization recommendations"""
        
        if not self.market_opportunities:
            return []
        
        recommendations = []
        
        # Diversification recommendations
        if len(set(opp.opportunity_type for opp in self.market_opportunities)) < 3:
            recommendations.append("Increase opportunity type diversification")
        
        # Investment balance recommendations
        total_investment = sum(opp.investment_required for opp in self.market_opportunities)
        if total_investment > 100:  # Assuming $100M budget limit
            recommendations.append("Consider prioritizing opportunities based on investment efficiency")
        
        # Risk balance recommendations
        high_risk_count = len([opp for opp in self.market_opportunities if opp.risk_level > 0.7])
        if high_risk_count > len(self.market_opportunities) * 0.4:
            recommendations.append("Balance portfolio risk by including more medium/low-risk opportunities")
        
        return recommendations
    
    def _generate_competitive_strategy(self) -> List[str]:
        """Generate competitive strategy recommendations"""
        
        strategies = []
        
        if self.competitor_analysis:
            # Address market leaders
            market_leaders = [comp for comp in self.competitor_analysis if comp.market_share > 0.20]
            if market_leaders:
                strategies.append("Differentiate from market leaders through unique value proposition")
            
            # Leverage innovation gaps
            innovation_leaders = [comp for comp in self.competitor_analysis if comp.innovation_capability > 0.8]
            if len(innovation_leaders) < len(self.competitor_analysis) * 0.5:
                strategies.append("Build innovation capabilities to compete with innovation leaders")
        
        return strategies
    
    def _generate_investment_recommendations(self) -> List[str]:
        """Generate investment recommendations"""
        
        recommendations = []
        
        if not self.market_opportunities:
            return recommendations
        
        # ROI-based recommendations
        high_roi_opportunities = [opp for opp in self.market_opportunities if opp.expected_roi > 3.0]
        if high_roi_opportunities:
            recommendations.append(f"Prioritize {len(high_roi_opportunities)} high-ROI opportunities (>300% ROI)")
        
        # Market size-based recommendations
        large_market_opportunities = [opp for opp in self.market_opportunities if opp.estimated_market_size > 50]
        if large_market_opportunities:
            recommendations.append(f"Focus on {len(large_market_opportunities)} large market opportunities (>$50M)")
        
        # Time to market recommendations
        quick_wins = [opp for opp in self.market_opportunities if opp.time_to_market < 12]
        if quick_wins:
            recommendations.append(f"Start with {len(quick_wins)} quick win opportunities (<12 months)")
        
        return recommendations
    
    def export_opportunity_analysis(self, output_path: str) -> bool:
        """Export market opportunity analysis to JSON file"""
        
        try:
            analysis_data = self.generate_market_opportunity_report()
            
            with open(output_path, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting opportunity analysis: {str(e)}")
            return False