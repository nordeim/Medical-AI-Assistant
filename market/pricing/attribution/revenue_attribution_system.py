"""
Revenue Attribution and Marketing ROI Tracking System
Comprehensive marketing attribution and ROI analytics for healthcare AI
"""

import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class MarketingActivity:
    """Marketing activity tracking"""
    activity_id: str
    activity_type: str  # conference, webinar, content, paid_ad, email, event
    channel: str  # digital, events, content, partnerships
    campaign_name: str
    start_date: datetime
    end_date: Optional[datetime]
    cost: float
    target_audience: str
    geography: str
    stage: str  # awareness, consideration, decision
    content_theme: str
    cta_type: str  # demo_request, download, contact_sales, trial_signup

@dataclass
class LeadActivity:
    """Lead activity and attribution"""
    lead_id: str
    marketing_activities: List[str]  # activity_ids
    conversion_path: List[str]  # sequence of touchpoints
    lead_source: str
    lead_grade: str  # cold, warm, hot, qualified
    created_date: datetime
    first_response_date: Optional[datetime]
    converted_date: Optional[datetime]
    final_status: str  # qualified, unqualified, converted, lost
    deal_value: Optional[float]
    sales_cycle_days: Optional[int]

@dataclass
class CustomerJourney:
    """Customer journey mapping"""
    customer_id: str
    organization_type: str
    journey_stages: List[Dict]
    total_touchpoints: int
    attribution_model: str
    time_to_conversion_days: int
    marketing_contribution: float
    sales_contribution: float
    channel_performance: Dict

class RevenueAttributionEngine:
    """
    Revenue Attribution and Marketing ROI Tracking Engine
    Comprehensive attribution modeling for healthcare AI marketing
    """
    
    def __init__(self):
        self.attribution_models = self._initialize_attribution_models()
        self.marketing_activities = {}
        self.leads = {}
        self.attribution_calculator = AttributionCalculator()
        self.roi_tracker = ROITracker()
        
    def _initialize_attribution_models(self) -> Dict:
        """Initialize attribution models"""
        return {
            "first_touch": {
                "name": "First Touch Attribution",
                "description": "100% credit to first marketing touchpoint",
                "weight_first": 1.0,
                "weight_intermediate": 0.0,
                "weight_last": 0.0
            },
            "last_touch": {
                "name": "Last Touch Attribution",
                "description": "100% credit to last marketing touchpoint",
                "weight_first": 0.0,
                "weight_intermediate": 0.0,
                "weight_last": 1.0
            },
            "linear": {
                "name": "Linear Attribution",
                "description": "Equal credit to all touchpoints",
                "weight_first": 0.33,
                "weight_intermediate": 0.34,
                "weight_last": 0.33
            },
            "time_decay": {
                "name": "Time Decay Attribution",
                "description": "More credit to recent touchpoints",
                "weight_first": 0.20,
                "weight_intermediate": 0.30,
                "weight_last": 0.50
            },
            "position_based": {
                "name": "Position Based Attribution",
                "description": "40% to first, 40% to last, 20% distributed",
                "weight_first": 0.40,
                "weight_intermediate": 0.20,
                "weight_last": 0.40
            },
            "machine_learning": {
                "name": "ML-Based Attribution",
                "description": "AI-driven attribution using Random Forest",
                "dynamic_weights": True
            }
        }
    
    def track_marketing_activity(self, activity: MarketingActivity) -> Dict:
        """Track marketing activity and calculate initial metrics"""
        
        # Store activity
        self.marketing_activities[activity.activity_id] = activity
        
        # Calculate initial metrics
        metrics = {
            "activity_id": activity.activity_id,
            "activity_type": activity.activity_type,
            "channel": activity.channel,
            "cost": activity.cost,
            "duration_days": (activity.end_date - activity.start_date).days if activity.end_date else 1,
            "cost_per_day": activity.cost / ((activity.end_date - activity.start_date).days + 1) if activity.end_date else activity.cost,
            "target_audience": activity.target_audience,
            "geography": activity.geography,
            "campaign_stage": activity.stage,
            "tracking_status": "active",
            "created_date": datetime.now().isoformat()
        }
        
        # Calculate expected metrics (to be updated with actual performance)
        expected_metrics = self._calculate_expected_metrics(activity)
        metrics.update(expected_metrics)
        
        return metrics
    
    def _calculate_expected_metrics(self, activity: MarketingActivity) -> Dict:
        """Calculate expected performance metrics"""
        
        # Benchmark conversion rates by activity type
        conversion_benchmarks = {
            "conference": {"lead_rate": 0.15, "demo_rate": 0.05, "qualified_rate": 0.02},
            "webinar": {"lead_rate": 0.25, "demo_rate": 0.08, "qualified_rate": 0.03},
            "content": {"lead_rate": 0.05, "demo_rate": 0.015, "qualified_rate": 0.005},
            "paid_ad": {"lead_rate": 0.02, "demo_rate": 0.005, "qualified_rate": 0.002},
            "email": {"lead_rate": 0.08, "demo_rate": 0.02, "qualified_rate": 0.008},
            "event": {"lead_rate": 0.20, "demo_rate": 0.06, "qualified_rate": 0.025}
        }
        
        benchmarks = conversion_benchmarks.get(activity.activity_type, conversion_benchmarks["content"])
        
        # Cost-based reach estimation
        estimated_reach = activity.cost / 50  # Assume $50 cost per reach
        
        expected_metrics = {
            "estimated_reach": estimated_reach,
            "expected_leads": estimated_reach * benchmarks["lead_rate"],
            "expected_demos": estimated_reach * benchmarks["demo_rate"],
            "expected_qualified": estimated_reach * benchmarks["qualified_rate"],
            "expected_cost_per_lead": activity.cost / (estimated_reach * benchmarks["lead_rate"]) if estimated_reach * benchmarks["lead_rate"] > 0 else 0,
            "expected_cost_per_qualified": activity.cost / (estimated_reach * benchmarks["qualified_rate"]) if estimated_reach * benchmarks["qualified_rate"] > 0 else 0
        }
        
        return expected_metrics
    
    def track_lead_activity(self, lead: LeadActivity) -> Dict:
        """Track lead activity and calculate attribution"""
        
        # Store lead
        self.leads[lead.lead_id] = lead
        
        # Calculate conversion metrics
        conversion_metrics = self._calculate_conversion_metrics(lead)
        
        # Calculate attribution
        attribution_results = self._calculate_attribution(lead)
        
        # Calculate journey metrics
        journey_metrics = self._calculate_journey_metrics(lead)
        
        return {
            "lead_id": lead.lead_id,
            "conversion_metrics": conversion_metrics,
            "attribution_results": attribution_results,
            "journey_metrics": journey_metrics,
            "lead_grade": lead.lead_grade,
            "final_status": lead.final_status,
            "revenue_attributed": lead.deal_value if lead.deal_value else 0,
            "tracking_date": datetime.now().isoformat()
        }
    
    def _calculate_conversion_metrics(self, lead: LeadActivity) -> Dict:
        """Calculate lead conversion metrics"""
        
        metrics = {
            "lead_id": lead.lead_id,
            "total_touchpoints": len(lead.conversion_path),
            "marketing_touchpoints": len(lead.marketing_activities),
            "conversion_time_days": 0,
            "lead_to_opportunity_rate": 0,
            "opportunity_to_close_rate": 0
        }
        
        # Calculate conversion time
        if lead.converted_date:
            metrics["conversion_time_days"] = (lead.converted_date - lead.created_date).days
        
        # Calculate conversion rates
        if lead.final_status == "converted":
            metrics["lead_to_opportunity_rate"] = 1.0
            if lead.deal_value and lead.deal_value > 0:
                metrics["opportunity_to_close_rate"] = 1.0
        elif lead.final_status == "qualified":
            metrics["lead_to_opportunity_rate"] = 1.0
            metrics["opportunity_to_close_rate"] = 0.0
        else:
            metrics["lead_to_opportunity_rate"] = 0.0
            metrics["opportunity_to_close_rate"] = 0.0
        
        return metrics
    
    def _calculate_attribution(self, lead: LeadActivity) -> Dict:
        """Calculate attribution using multiple models"""
        
        attribution_results = {}
        
        for model_name, model_config in self.attribution_models.items():
            if model_name == "machine_learning":
                # ML-based attribution (simplified for demo)
                attribution_weights = self._calculate_ml_attribution(lead)
            else:
                # Rule-based attribution
                attribution_weights = self._calculate_rule_based_attribution(
                    lead, model_config
                )
            
            # Calculate revenue attribution
            revenue_attribution = self._calculate_revenue_attribution(
                lead, attribution_weights
            )
            
            attribution_results[model_name] = {
                "model_name": model_config["name"],
                "weights": attribution_weights,
                "revenue_attribution": revenue_attribution,
                "touchpoint_count": len(attribution_weights)
            }
        
        return attribution_results
    
    def _calculate_rule_based_attribution(self, 
                                        lead: LeadActivity, 
                                        model_config: Dict) -> Dict:
        """Calculate rule-based attribution weights"""
        
        touchpoints = lead.conversion_path
        weights = {}
        
        if len(touchpoints) == 1:
            # Single touchpoint gets all credit
            weights[touchpoints[0]] = 1.0
        elif len(touchpoints) == 2:
            # Two touchpoints
            weights[touchpoints[0]] = model_config["weight_first"]
            weights[touchpoints[1]] = model_config["weight_last"]
        else:
            # Multiple touchpoints
            weights[touchpoints[0]] = model_config["weight_first"]
            weights[touchpoints[-1]] = model_config["weight_last"]
            
            # Distribute intermediate weight
            intermediate_count = len(touchpoints) - 2
            if intermediate_count > 0:
                intermediate_weight = model_config["weight_intermediate"] / intermediate_count
                for touchpoint in touchpoints[1:-1]:
                    weights[touchpoint] = intermediate_weight
        
        return weights
    
    def _calculate_ml_attribution(self, lead: LeadActivity) -> Dict:
        """Calculate ML-based attribution weights"""
        
        # This would normally use a trained ML model
        # For demo purposes, we'll simulate ML attribution
        
        touchpoints = lead.conversion_path
        weights = {}
        
        # Simulate ML model considering:
        # - Touchpoint order (recency bias)
        # - Activity type importance
        # - Time between touchpoints
        
        total_weight = 0
        for i, touchpoint in enumerate(touchpoints):
            # More weight for recent touchpoints
            recency_weight = 1.0 + (i * 0.1)
            
            # Get activity type if available
            activity = self.marketing_activities.get(touchpoint)
            if activity:
                activity_weight = self._get_activity_importance_weight(activity)
            else:
                activity_weight = 1.0
            
            weight = recency_weight * activity_weight
            weights[touchpoint] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for touchpoint in weights:
                weights[touchpoint] /= total_weight
        
        return weights
    
    def _get_activity_importance_weight(self, activity: MarketingActivity) -> float:
        """Get importance weight for activity type"""
        
        importance_weights = {
            "conference": 1.5,
            "webinar": 1.3,
            "content": 1.0,
            "paid_ad": 0.8,
            "email": 1.1,
            "event": 1.2
        }
        
        return importance_weights.get(activity.activity_type, 1.0)
    
    def _calculate_revenue_attribution(self, lead: LeadActivity, weights: Dict) -> Dict:
        """Calculate revenue attribution"""
        
        if not lead.deal_value or lead.deal_value <= 0:
            return {"total_attributed": 0, "attribution_by_touchpoint": {}}
        
        attribution_by_touchpoint = {}
        for touchpoint, weight in weights.items():
            attributed_value = lead.deal_value * weight
            attribution_by_touchpoint[touchpoint] = {
                "weight": weight,
                "attributed_revenue": attributed_value
            }
        
        return {
            "total_attributed": lead.deal_value,
            "attribution_by_touchpoint": attribution_by_touchpoint
        }
    
    def _calculate_journey_metrics(self, lead: LeadActivity) -> Dict:
        """Calculate customer journey metrics"""
        
        metrics = {
            "journey_length": len(lead.conversion_path),
            "touchpoint_velocity": 0,
            "channel_diversity": 0,
            "stage_progression": [],
            "time_to_first_response": 0,
            "response_quality_score": 0
        }
        
        # Calculate touchpoint velocity (touchpoints per week)
        if lead.converted_date:
            journey_duration = (lead.converted_date - lead.created_date).days
            metrics["touchpoint_velocity"] = len(lead.conversion_path) / max(1, journey_duration / 7)
        
        # Calculate channel diversity
        channels = set()
        for activity_id in lead.marketing_activities:
            activity = self.marketing_activities.get(activity_id)
            if activity:
                channels.add(activity.channel)
        metrics["channel_diversity"] = len(channels)
        
        # Calculate time to first response
        if lead.first_response_date:
            metrics["time_to_first_response"] = (lead.first_response_date - lead.created_date).days
        
        # Stage progression analysis
        stage_progression = []
        for activity_id in lead.marketing_activities:
            activity = self.marketing_activities.get(activity_id)
            if activity:
                stage_progression.append(activity.stage)
        metrics["stage_progression"] = stage_progression
        
        return metrics
    
    def generate_attribution_report(self, 
                                  start_date: datetime, 
                                  end_date: datetime,
                                  model_selection: List[str] = None) -> Dict:
        """Generate comprehensive attribution report"""
        
        if model_selection is None:
            model_selection = ["linear", "time_decay", "machine_learning"]
        
        # Filter data by date range
        filtered_activities = self._filter_activities_by_date(start_date, end_date)
        filtered_leads = self._filter_leads_by_date(start_date, end_date)
        
        # Generate attribution analysis
        attribution_summary = self._generate_attribution_summary(filtered_leads, model_selection)
        channel_performance = self._analyze_channel_performance(filtered_leads)
        campaign_roi = self._calculate_campaign_roi(filtered_activities, filtered_leads)
        customer_journey_insights = self._analyze_customer_journeys(filtered_leads)
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days_included": (end_date - start_date).days
            },
            "executive_summary": {
                "total_marketing_spend": sum([a.cost for a in filtered_activities]),
                "total_leads_generated": len(filtered_leads),
                "total_revenue_attributed": sum([l.deal_value for l in filtered_leads if l.deal_value]),
                "overall_marketing_roi": 0,  # Will be calculated
                "best_performing_channel": max(channel_performance.keys(), 
                                             key=lambda x: channel_performance[x]["revenue"]) if channel_performance else "N/A"
            },
            "attribution_analysis": attribution_summary,
            "channel_performance": channel_performance,
            "campaign_roi": campaign_roi,
            "customer_journey_insights": customer_journey_insights,
            "recommendations": self._generate_attribution_recommendations(
                attribution_summary, channel_performance, campaign_roi
            )
        }
    
    def _filter_activities_by_date(self, start_date: datetime, end_date: datetime) -> List[MarketingActivity]:
        """Filter marketing activities by date range"""
        
        filtered = []
        for activity in self.marketing_activities.values():
            if activity.start_date >= start_date and activity.start_date <= end_date:
                filtered.append(activity)
        
        return filtered
    
    def _filter_leads_by_date(self, start_date: datetime, end_date: datetime) -> List[LeadActivity]:
        """Filter leads by date range"""
        
        filtered = []
        for lead in self.leads.values():
            if lead.created_date >= start_date and lead.created_date <= end_date:
                filtered.append(lead)
        
        return filtered
    
    def _generate_attribution_summary(self, leads: List[LeadActivity], model_selection: List[str]) -> Dict:
        """Generate attribution summary using selected models"""
        
        summary = {}
        total_revenue = sum([l.deal_value for l in leads if l.deal_value])
        
        for model in model_selection:
            channel_attribution = {}
            activity_attribution = {}
            
            for lead in leads:
                if lead.deal_value and model in self.attribution_models:
                    # Get attribution weights for this lead and model
                    lead_attribution = self._calculate_single_lead_attribution(lead, model)
                    
                    # Aggregate by channel and activity
                    for activity_id, weight in lead_attribution.items():
                        activity = self.marketing_activities.get(activity_id)
                        if activity:
                            # Channel attribution
                            if activity.channel not in channel_attribution:
                                channel_attribution[activity.channel] = 0
                            channel_attribution[activity.channel] += lead.deal_value * weight
                            
                            # Activity attribution
                            if activity_id not in activity_attribution:
                                activity_attribution[activity_id] = {
                                    "activity_name": activity.campaign_name,
                                    "revenue_attributed": 0,
                                    "lead_count": 0
                                }
                            activity_attribution[activity_id]["revenue_attributed"] += lead.deal_value * weight
                            activity_attribution[activity_id]["lead_count"] += 1
            
            summary[model] = {
                "channel_attribution": channel_attribution,
                "activity_attribution": activity_attribution,
                "total_attributed": sum(channel_attribution.values()),
                "model_efficiency": sum(channel_attribution.values()) / total_revenue if total_revenue > 0 else 0
            }
        
        return summary
    
    def _calculate_single_lead_attribution(self, lead: LeadActivity, model: str) -> Dict:
        """Calculate attribution for a single lead using specified model"""
        
        if model not in self.attribution_models:
            return {}
        
        model_config = self.attribution_models[model]
        
        if model == "machine_learning":
            return self._calculate_ml_attribution(lead)
        else:
            return self._calculate_rule_based_attribution(lead, model_config)
    
    def _analyze_channel_performance(self, leads: List[LeadActivity]) -> Dict:
        """Analyze performance by marketing channel"""
        
        channel_metrics = {}
        
        for lead in leads:
            for activity_id in lead.marketing_activities:
                activity = self.marketing_activities.get(activity_id)
                if activity:
                    channel = activity.channel
                    
                    if channel not in channel_metrics:
                        channel_metrics[channel] = {
                            "activities": 0,
                            "total_cost": 0,
                            "leads": 0,
                            "qualified_leads": 0,
                            "converted_leads": 0,
                            "revenue": 0,
                            "conversion_rates": {}
                        }
                    
                    channel_metrics[channel]["activities"] += 1
                    channel_metrics[channel]["total_cost"] += activity.cost
                    channel_metrics[channel]["leads"] += 1
                    
                    if lead.lead_grade in ["hot", "qualified"]:
                        channel_metrics[channel]["qualified_leads"] += 1
                    
                    if lead.final_status == "converted":
                        channel_metrics[channel]["converted_leads"] += 1
                        if lead.deal_value:
                            channel_metrics[channel]["revenue"] += lead.deal_value
            
            # Calculate conversion rates
            for channel in channel_metrics:
                metrics = channel_metrics[channel]
                metrics["lead_to_qualified_rate"] = (metrics["qualified_leads"] / 
                                                   max(1, metrics["leads"]))
                metrics["qualified_to_converted_rate"] = (metrics["converted_leads"] / 
                                                        max(1, metrics["qualified_leads"]))
                metrics["lead_to_converted_rate"] = (metrics["converted_leads"] / 
                                                   max(1, metrics["leads"]))
                metrics["cost_per_lead"] = (metrics["total_cost"] / 
                                          max(1, metrics["leads"]))
                metrics["revenue_per_lead"] = (metrics["revenue"] / 
                                             max(1, metrics["leads"]))
                metrics["roi"] = ((metrics["revenue"] - metrics["total_cost"]) / 
                                max(1, metrics["total_cost"]))
        
        return channel_metrics
    
    def _calculate_campaign_roi(self, activities: List[MarketingActivity], leads: List[LeadActivity]) -> Dict:
        """Calculate ROI by campaign"""
        
        campaign_metrics = {}
        
        # Group activities by campaign
        campaigns = {}
        for activity in activities:
            if activity.campaign_name not in campaigns:
                campaigns[activity.campaign_name] = []
            campaigns[activity.campaign_name].append(activity)
        
        # Calculate ROI for each campaign
        for campaign_name, campaign_activities in campaigns.items():
            campaign_cost = sum([a.cost for a in campaign_activities])
            campaign_leads = self._get_leads_for_activities([a.activity_id for a in campaign_activities], leads)
            campaign_revenue = sum([l.deal_value for l in campaign_leads if l.deal_value])
            
            campaign_metrics[campaign_name] = {
                "total_cost": campaign_cost,
                "total_activities": len(campaign_activities),
                "leads_generated": len(campaign_leads),
                "revenue_attributed": campaign_revenue,
                "roi": ((campaign_revenue - campaign_cost) / max(1, campaign_cost)),
                "cost_per_lead": campaign_cost / max(1, len(campaign_leads)),
                "revenue_per_activity": campaign_revenue / max(1, len(campaign_activities)),
                "conversion_rate": len([l for l in campaign_leads if l.final_status == "converted"]) / max(1, len(campaign_leads))
            }
        
        return campaign_metrics
    
    def _get_leads_for_activities(self, activity_ids: List[str], all_leads: List[LeadActivity]) -> List[LeadActivity]:
        """Get leads associated with specific activities"""
        
        return [lead for lead in all_leads 
                if any(activity_id in lead.marketing_activities for activity_id in activity_ids)]
    
    def _analyze_customer_journeys(self, leads: List[LeadActivity]) -> Dict:
        """Analyze customer journey patterns"""
        
        journey_analysis = {
            "average_journey_length": 0,
            "common_pathways": {},
            "stage_conversion_rates": {},
            "time_to_conversion": {},
            "channel_sequence_patterns": {},
            "optimization_opportunities": []
        }
        
        # Calculate average journey length
        journey_lengths = [len(lead.conversion_path) for lead in leads]
        journey_analysis["average_journey_length"] = np.mean(journey_lengths) if journey_lengths else 0
        
        # Analyze conversion time
        conversion_times = [(lead.converted_date - lead.created_date).days 
                          for lead in leads if lead.converted_date]
        if conversion_times:
            journey_analysis["time_to_conversion"] = {
                "average_days": np.mean(conversion_times),
                "median_days": np.median(conversion_times),
                "fastest_conversion": min(conversion_times),
                "slowest_conversion": max(conversion_times)
            }
        
        # Analyze common pathways
        pathways = {}
        for lead in leads:
            pathway = " -> ".join(lead.conversion_path)
            pathways[pathway] = pathways.get(pathway, 0) + 1
        
        journey_analysis["common_pathways"] = dict(sorted(pathways.items(), 
                                                        key=lambda x: x[1], 
                                                        reverse=True)[:10])
        
        return journey_analysis
    
    def _generate_attribution_recommendations(self, 
                                            attribution_summary: Dict,
                                            channel_performance: Dict,
                                            campaign_roi: Dict) -> List[str]:
        """Generate recommendations based on attribution analysis"""
        
        recommendations = []
        
        # Channel recommendations
        if channel_performance:
            best_channel = max(channel_performance.keys(), 
                             key=lambda x: channel_performance[x]["roi"])
            recommendations.append(f"Increase investment in {best_channel} - highest ROI channel")
            
            worst_channel = min(channel_performance.keys(), 
                              key=lambda x: channel_performance[x]["roi"])
            recommendations.append(f"Review strategy for {worst_channel} - lowest ROI channel")
        
        # Campaign recommendations
        if campaign_roi:
            best_campaign = max(campaign_roi.keys(), 
                              key=lambda x: campaign_roi[x]["roi"])
            recommendations.append(f"Expand successful {best_campaign} campaign model")
        
        # Journey optimization
        recommendations.append("Focus on shortening customer journeys through better lead nurturing")
        recommendations.append("Implement multi-touch attribution for more accurate ROI measurement")
        recommendations.append("Develop channel-specific content strategies based on attribution insights")
        
        return recommendations

class AttributionCalculator:
    """Advanced attribution calculation engine"""
    
    def __init__(self):
        self.model_weights = {}
        self.calibration_data = []
    
    def calculate_attribution_credibility_score(self, attribution_data: Dict) -> float:
        """Calculate credibility score for attribution model"""
        
        # Factors affecting credibility:
        # 1. Data completeness
        # 2. Model consistency across different segments
        # 3. Business logic alignment
        # 4. Historical accuracy
        
        completeness_score = self._assess_data_completeness(attribution_data)
        consistency_score = self._assess_model_consistency(attribution_data)
        logic_score = self._assess_business_logic_alignment(attribution_data)
        
        credibility_score = (completeness_score * 0.4 + 
                           consistency_score * 0.3 + 
                           logic_score * 0.3)
        
        return credibility_score
    
    def _assess_data_completeness(self, data: Dict) -> float:
        """Assess completeness of attribution data"""
        
        # Check for required data points
        required_fields = ["channel_attribution", "activity_attribution", "total_attributed"]
        present_fields = sum([1 for field in required_fields if field in data])
        
        return present_fields / len(required_fields)
    
    def _assess_model_consistency(self, data: Dict) -> float:
        """Assess consistency across attribution models"""
        
        if len(data) < 2:
            return 1.0  # Cannot assess consistency with single model
        
        # Check if top channels are consistent across models
        top_channels_by_model = []
        for model_data in data.values():
            if "channel_attribution" in model_data:
                sorted_channels = sorted(model_data["channel_attribution"].items(), 
                                       key=lambda x: x[1], reverse=True)
                top_channels_by_model.append([ch[0] for ch in sorted_channels[:3]])
        
        if len(top_channels_by_model) < 2:
            return 1.0
        
        # Calculate consistency score
        consistency_scores = []
        for i in range(len(top_channels_by_model) - 1):
            common_channels = set(top_channels_by_model[i]) & set(top_channels_by_model[i+1])
            consistency_scores.append(len(common_channels) / 3)  # 3 is max common channels
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _assess_business_logic_alignment(self, data: Dict) -> float:
        """Assess alignment with business logic"""
        
        # Business logic checks:
        # 1. High-cost activities should have higher attribution (generally)
        # 2. Last touch should have meaningful weight
        # 3. Multiple touchpoints should be better than single touchpoints
        
        alignment_score = 0.5  # Default neutral score
        
        # This would implement specific business logic checks
        # For demo purposes, returning neutral score
        
        return alignment_score

class ROITracker:
    """Marketing ROI tracking and optimization"""
    
    def __init__(self):
        self.roi_benchmarks = self._load_roi_benchmarks()
        self.optimization_engine = ROIOptimizationEngine()
    
    def _load_roi_benchmarks(self) -> Dict:
        """Load ROI benchmarks by channel and activity type"""
        
        return {
            "channel_benchmarks": {
                "digital": {"target_roi": 4.0, "acceptable_roi": 2.0},
                "events": {"target_roi": 6.0, "acceptable_roi": 3.0},
                "content": {"target_roi": 8.0, "acceptable_roi": 4.0},
                "partnerships": {"target_roi": 5.0, "acceptable_roi": 2.5}
            },
            "activity_benchmarks": {
                "conference": {"target_roi": 5.0, "acceptable_roi": 2.5},
                "webinar": {"target_roi": 7.0, "acceptable_roi": 3.5},
                "content": {"target_roi": 10.0, "acceptable_roi": 5.0},
                "paid_ad": {"target_roi": 3.0, "acceptable_roi": 1.5},
                "email": {"target_roi": 8.0, "acceptable_roi": 4.0}
            }
        }
    
    def calculate_comprehensive_roi(self, 
                                  marketing_data: Dict,
                                  time_horizon_months: int = 12) -> Dict:
        """Calculate comprehensive marketing ROI"""
        
        # Basic ROI calculations
        basic_roi = self._calculate_basic_roi(marketing_data)
        
        # Long-term ROI (with retention and expansion)
        long_term_roi = self._calculate_long_term_roi(marketing_data, time_horizon_months)
        
        # Customer lifetime value impact
        clv_impact = self._calculate_clv_impact(marketing_data)
        
        # Channel efficiency analysis
        channel_efficiency = self._analyze_channel_efficiency(marketing_data)
        
        # Optimization recommendations
        optimization_opportunities = self.optimization_engine.identify_opportunities(
            marketing_data, basic_roi, channel_efficiency
        )
        
        return {
            "basic_roi_metrics": basic_roi,
            "long_term_roi": long_term_roi,
            "clv_impact": clv_impact,
            "channel_efficiency": channel_efficiency,
            "optimization_opportunities": optimization_opportunities,
            "roi_forecast": self._generate_roi_forecast(marketing_data),
            "benchmark_comparison": self._compare_to_benchmarks(basic_roi, channel_efficiency)
        }
    
    def _calculate_basic_roi(self, marketing_data: Dict) -> Dict:
        """Calculate basic ROI metrics"""
        
        total_cost = marketing_data.get("total_marketing_cost", 0)
        total_revenue = marketing_data.get("attributed_revenue", 0)
        
        roi_ratio = (total_revenue - total_cost) / max(1, total_cost)
        roi_percentage = roi_ratio * 100
        
        # Calculate additional metrics
        cost_per_acquisition = total_cost / max(1, marketing_data.get("conversions", 1))
        revenue_per_dollar_spent = total_revenue / max(1, total_cost)
        
        return {
            "total_investment": total_cost,
            "total_revenue": total_revenue,
            "net_revenue": total_revenue - total_cost,
            "roi_ratio": roi_ratio,
            "roi_percentage": roi_percentage,
            "cost_per_acquisition": cost_per_acquisition,
            "revenue_per_dollar": revenue_per_dollar_spent,
            "payback_period_months": self._calculate_payback_period(marketing_data)
        }
    
    def _calculate_long_term_roi(self, marketing_data: Dict, time_horizon: int) -> Dict:
        """Calculate long-term ROI with retention and expansion"""
        
        # This would incorporate customer retention rates and expansion revenue
        # For demo purposes, simplified calculation
        
        base_revenue = marketing_data.get("attributed_revenue", 0)
        retention_rate = marketing_data.get("customer_retention_rate", 0.85)
        expansion_rate = marketing_data.get("expansion_rate", 0.15)
        
        # Calculate multi-year revenue
        multi_year_revenue = 0
        for year in range(1, time_horizon // 12 + 1):
            yearly_revenue = base_revenue * (retention_rate ** year) * (1 + expansion_rate) ** year
            multi_year_revenue += yearly_revenue
        
        total_cost = marketing_data.get("total_marketing_cost", 0)
        long_term_roi = (multi_year_revenue - total_cost) / max(1, total_cost)
        
        return {
            "multi_year_revenue": multi_year_revenue,
            "long_term_roi_ratio": long_term_roi,
            "retention_adjusted_value": base_revenue * retention_rate,
            "expansion_adjusted_value": base_revenue * (1 + expansion_rate),
            "time_horizon_years": time_horizon / 12
        }
    
    def _calculate_clv_impact(self, marketing_data: Dict) -> Dict:
        """Calculate customer lifetime value impact"""
        
        # This would calculate the impact on customer lifetime value
        # For demo purposes, simplified
        
        customers_acquired = marketing_data.get("customers_acquired", 0)
        average_clv = marketing_data.get("average_customer_lifetime_value", 500000)
        total_clv_impact = customers_acquired * average_clv
        
        marketing_cost_per_customer = marketing_data.get("total_marketing_cost", 0) / max(1, customers_acquired)
        clv_to_cac_ratio = average_clv / max(1, marketing_cost_per_customer)
        
        return {
            "customers_acquired": customers_acquired,
            "average_clv": average_clv,
            "total_clv_impact": total_clv_impact,
            "marketing_cost_per_customer": marketing_cost_per_customer,
            "clv_to_cac_ratio": clv_to_cac_ratio,
            "clv_roi_ratio": total_clv_impact / max(1, marketing_data.get("total_marketing_cost", 1))
        }
    
    def _analyze_channel_efficiency(self, marketing_data: Dict) -> Dict:
        """Analyze efficiency by marketing channel"""
        
        channel_data = marketing_data.get("channel_performance", {})
        efficiency_analysis = {}
        
        for channel, data in channel_data.items():
            efficiency = {
                "roi": data.get("roi", 0),
                "cost_per_lead": data.get("cost_per_lead", 0),
                "conversion_rate": data.get("conversion_rate", 0),
                "efficiency_score": self._calculate_efficiency_score(data),
                "benchmark_comparison": self._compare_channel_to_benchmark(channel, data)
            }
            efficiency_analysis[channel] = efficiency
        
        return efficiency_analysis
    
    def _calculate_efficiency_score(self, channel_data: Dict) -> float:
        """Calculate overall efficiency score for channel"""
        
        # Weighted scoring based on multiple factors
        roi_score = min(1.0, channel_data.get("roi", 0) / 5.0)  # Normalize ROI to 0-1
        conversion_score = min(1.0, channel_data.get("conversion_rate", 0) * 10)  # Normalize conversion rate
        
        efficiency_score = (roi_score * 0.6 + conversion_score * 0.4)
        
        return efficiency_score
    
    def _compare_channel_to_benchmark(self, channel: str, data: Dict) -> Dict:
        """Compare channel performance to benchmarks"""
        
        benchmarks = self.roi_benchmarks["channel_benchmarks"].get(channel, 
                                                                {"target_roi": 4.0, "acceptable_roi": 2.0})
        
        actual_roi = data.get("roi", 0)
        
        if actual_roi >= benchmarks["target_roi"]:
            status = "exceeds_target"
        elif actual_roi >= benchmarks["acceptable_roi"]:
            status = "meets_acceptable"
        else:
            status = "below_acceptable"
        
        return {
            "benchmark_status": status,
            "target_roi": benchmarks["target_roi"],
            "acceptable_roi": benchmarks["acceptable_roi"],
            "actual_roi": actual_roi,
            "performance_gap": actual_roi - benchmarks["acceptable_roi"]
        }
    
    def _generate_roi_forecast(self, marketing_data: Dict) -> Dict:
        """Generate ROI forecast based on trends"""
        
        # This would use historical data and trends to forecast ROI
        # For demo purposes, simplified forecast
        
        current_roi = marketing_data.get("current_roi", 0)
        growth_trend = marketing_data.get("roi_growth_trend", 0.05)  # 5% monthly growth
        
        forecast_months = 6
        forecast_data = []
        
        for month in range(1, forecast_months + 1):
            forecasted_roi = current_roi * (1 + growth_trend) ** month
            forecast_data.append({
                "month": month,
                "projected_roi": forecasted_roi,
                "confidence_interval": {
                    "low": forecasted_roi * 0.8,
                    "high": forecasted_roi * 1.2
                }
            })
        
        return {
            "forecast_period_months": forecast_months,
            "forecast_data": forecast_data,
            "trend_analysis": {
                "growth_rate": growth_trend,
                "trend_direction": "improving" if growth_trend > 0 else "declining",
                "confidence_level": 0.75
            }
        }
    
    def _compare_to_benchmarks(self, basic_roi: Dict, channel_efficiency: Dict) -> Dict:
        """Compare performance to industry benchmarks"""
        
        overall_roi = basic_roi.get("roi_ratio", 0)
        
        # Industry benchmark for healthcare marketing ROI
        industry_benchmark = 4.5
        
        performance_vs_benchmark = {
            "our_roi": overall_roi,
            "industry_benchmark": industry_benchmark,
            "performance_ratio": overall_roi / industry_benchmark,
            "performance_rating": self._rate_performance(overall_roi, industry_benchmark)
        }
        
        # Channel-level comparisons
        channel_comparisons = {}
        for channel, efficiency in channel_efficiency.items():
            channel_comparisons[channel] = self._compare_channel_to_benchmark(
                channel, {"roi": efficiency.get("roi", 0)}
            )
        
        return {
            "overall_performance": performance_vs_benchmark,
            "channel_performance": channel_comparisons,
            "improvement_opportunities": self._identify_improvement_opportunities(
                performance_vs_benchmark, channel_comparisons
            )
        }
    
    def _rate_performance(self, our_value: float, benchmark: float) -> str:
        """Rate performance against benchmark"""
        
        ratio = our_value / benchmark if benchmark > 0 else 0
        
        if ratio >= 1.2:
            return "excellent"
        elif ratio >= 1.0:
            return "good"
        elif ratio >= 0.8:
            return "fair"
        else:
            return "poor"
    
    def _identify_improvement_opportunities(self, overall_performance: Dict, channel_comparisons: Dict) -> List[str]:
        """Identify specific improvement opportunities"""
        
        opportunities = []
        
        # Overall performance opportunities
        if overall_performance.get("performance_ratio", 0) < 1.0:
            opportunities.append("Overall ROI below industry benchmark - review marketing mix strategy")
        
        # Channel-specific opportunities
        for channel, comparison in channel_comparisons.items():
            if comparison.get("benchmark_status") == "below_acceptable":
                opportunities.append(f"Improve {channel} channel performance - currently below acceptable ROI")
            elif comparison.get("performance_gap", 0) > 0:
                opportunities.append(f"Scale {channel} channel - exceeding acceptable performance")
        
        return opportunities
    
    def _calculate_payback_period(self, marketing_data: Dict) -> float:
        """Calculate payback period in months"""
        
        total_cost = marketing_data.get("total_marketing_cost", 0)
        monthly_revenue = marketing_data.get("monthly_attributed_revenue", 0)
        
        if monthly_revenue <= 0:
            return float('inf')  # No payback if no revenue
        
        payback_months = total_cost / monthly_revenue
        
        return payback_months

class ROIOptimizationEngine:
    """ROI optimization and recommendation engine"""
    
    def identify_opportunities(self, 
                             marketing_data: Dict,
                             current_roi: Dict,
                             channel_efficiency: Dict) -> List[Dict]:
        """Identify specific optimization opportunities"""
        
        opportunities = []
        
        # Budget reallocation opportunities
        budget_opportunities = self._identify_budget_opportunities(channel_efficiency)
        opportunities.extend(budget_opportunities)
        
        # Performance improvement opportunities
        performance_opportunities = self._identify_performance_opportunities(channel_efficiency)
        opportunities.extend(performance_opportunities)
        
        # Strategy optimization opportunities
        strategy_opportunities = self._identify_strategy_opportunities(marketing_data)
        opportunities.extend(strategy_opportunities)
        
        return opportunities
    
    def _identify_budget_opportunities(self, channel_efficiency: Dict) -> List[Dict]:
        """Identify budget reallocation opportunities"""
        
        opportunities = []
        
        # Find best and worst performing channels
        sorted_channels = sorted(channel_efficiency.items(), 
                               key=lambda x: x[1].get("roi", 0), 
                               reverse=True)
        
        if len(sorted_channels) >= 2:
            best_channel = sorted_channels[0]
            worst_channel = sorted_channels[-1]
            
            if best_channel[1].get("roi", 0) > 2 * worst_channel[1].get("roi", 0):
                opportunities.append({
                    "type": "budget_reallocation",
                    "description": f"Reallocate budget from {worst_channel[0]} to {best_channel[0]}",
                    "potential_impact": "High ROI improvement",
                    "implementation_effort": "Medium",
                    "recommendation": "Gradual reallocation over 3 months"
                })
        
        return opportunities
    
    def _identify_performance_opportunities(self, channel_efficiency: Dict) -> List[Dict]:
        """Identify performance improvement opportunities"""
        
        opportunities = []
        
        for channel, efficiency in channel_efficiency.items():
            roi = efficiency.get("roi", 0)
            conversion_rate = efficiency.get("conversion_rate", 0)
            
            if roi > 0 and conversion_rate < 0.1:  # Low conversion rate
                opportunities.append({
                    "type": "conversion_optimization",
                    "description": f"Improve conversion rate for {channel} channel",
                    "current_conversion_rate": conversion_rate,
                    "potential_impact": "Significant ROI improvement",
                    "implementation_effort": "High",
                    "recommendation": "A/B test landing pages and CTAs"
                })
        
        return opportunities
    
    def _identify_strategy_opportunities(self, marketing_data: Dict) -> List[Dict]:
        """Identify strategic optimization opportunities"""
        
        opportunities = []
        
        # Check for multi-channel coordination
        channels_used = len(marketing_data.get("channel_performance", {}))
        if channels_used < 3:
            opportunities.append({
                "type": "channel_diversification",
                "description": "Expand to additional marketing channels",
                "potential_impact": "Reduced risk and increased reach",
                "implementation_effort": "High",
                "recommendation": "Pilot test 2-3 additional channels"
            })
        
        return opportunities

# Example usage and testing
if __name__ == "__main__":
    # Create sample marketing activities
    activities = [
        MarketingActivity(
            activity_id="ACT001",
            activity_type="webinar",
            channel="digital",
            campaign_name="AI in Healthcare Q1",
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 1, 15),
            cost=15000,
            target_audience="hospital_cmos",
            geography="US",
            stage="awareness",
            content_theme="clinical_outcomes",
            cta_type="demo_request"
        ),
        MarketingActivity(
            activity_id="ACT002",
            activity_type="conference",
            channel="events",
            campaign_name="HIMSS24 Exhibition",
            start_date=datetime(2024, 3, 1),
            end_date=datetime(2024, 3, 4),
            cost=50000,
            target_audience="healthcare_executives",
            geography="US",
            stage="decision",
            content_theme="comprehensive_solution",
            cta_type="trial_signup"
        )
    ]
    
    # Create sample leads
    leads = [
        LeadActivity(
            lead_id="LEAD001",
            marketing_activities=["ACT001", "ACT002"],
            conversion_path=["ACT001", "ACT002", "SALES_MEETING"],
            lead_source="webinar",
            lead_grade="qualified",
            created_date=datetime(2024, 1, 15),
            first_response_date=datetime(2024, 1, 16),
            converted_date=datetime(2024, 3, 15),
            final_status="converted",
            deal_value=750000,
            sales_cycle_days=60
        )
    ]
    
    # Initialize attribution engine
    engine = RevenueAttributionEngine()
    
    # Track activities and leads
    for activity in activities:
        engine.track_marketing_activity(activity)
    
    for lead in leads:
        engine.track_lead_activity(lead)
    
    # Generate attribution report
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    report = engine.generate_attribution_report(start_date, end_date)
    
    print("Revenue Attribution and Marketing ROI Tracking Demo")
    print("=" * 60)
    print(f"Report Period: {report['report_period']['start_date']} to {report['report_period']['end_date']}")
    print(f"Total Marketing Spend: ${report['executive_summary']['total_marketing_spend']:,.0f}")
    print(f"Total Revenue Attributed: ${report['executive_summary']['total_revenue_attributed']:,.0f}")
    print(f"Best Performing Channel: {report['executive_summary']['best_performing_channel']}")
