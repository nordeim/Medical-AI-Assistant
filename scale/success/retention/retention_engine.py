"""
Customer Retention Engine
Manages retention strategies and optimizes customer lifecycle
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path


class RetentionEngine:
    """
    Customer retention optimization engine
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Retention strategies
        self.retention_strategies = self.load_retention_strategies()
        self.retention_tactics = self.load_retention_tactics()
        
        # Customer segments
        self.customer_segments = {}
        
        # Performance tracking
        self.retention_metrics = {
            "retention_rate": 0,
            "churn_rate": 0,
            "save_rate": 0,
            "intervention_success_rate": 0,
            "average_save_value": 0
        }
        
        # Initialize segment models
        self.initialize_segment_models()
        
        self.logger.info("Retention Engine initialized")
    
    def load_retention_strategies(self) -> Dict[str, Any]:
        """Load retention strategies and frameworks"""
        return {
            "retention_framework": {
                "identify": {
                    "customer_health_scoring": True,
                    "churn_risk_prediction": True,
                    "behavior_pattern_analysis": True,
                    "satisfaction_tracking": True
                },
                "engage": {
                    "proactive_outreach": True,
                    "value_demonstration": True,
                    "education_and_training": True,
                    "community_building": True
                },
                "resolve": {
                    "issue_identification": True,
                    "rapid_response": True,
                    "solution_implementation": True,
                    "follow_up_tracking": True
                },
                "expand": {
                    "value_identification": True,
                    "expansion_opportunities": True,
                    "cross_selling": True,
                    "upselling": True
                }
            },
            "retention_strategies": {
                "proactive_retention": {
                    "description": "Prevent churn before it happens",
                    "triggers": ["low_engagement", "support_issues", "payment_delays"],
                    "tactics": ["personalized_engagement", "value_demonstration", "training_programs"],
                    "timeline": "immediate",
                    "success_metrics": ["engagement_score", "feature_adoption", "satisfaction"]
                },
                "reactive_retention": {
                    "description": "Win back customers at risk",
                    "triggers": ["high_churn_risk", "contract_renewal", "cancellation_intent"],
                    "tactics": ["executive_outreach", "special_offers", "solution_customization"],
                    "timeline": "urgent",
                    "success_metrics": ["save_rate", "renewal_rate", "satisfaction"]
                },
                "expansion_retention": {
                    "description": "Increase value to prevent churn",
                    "triggers": ["high_engagement", "success_achievement", "expansion_readiness"],
                    "tactics": ["feature_upgrade", "additional_services", "partnership_opportunities"],
                    "timeline": "planned",
                    "success_metrics": ["revenue_growth", "feature_adoption", "stakeholder_satisfaction"]
                },
                "community_retention": {
                    "description": "Build community to increase stickiness",
                    "triggers": ["active_users", "advocacy_potential", "community_engagement"],
                    "tactics": ["peer_networking", "knowledge_sharing", "community_events"],
                    "timeline": "ongoing",
                    "success_metrics": ["community_participation", "peer_support", "advocacy_actions"]
                }
            },
            "segment_specific_strategies": {
                "enterprise": {
                    "key_contacts": ["executive", "technical", "business"],
                    "decision_makers": ["c_level", "vp", "director"],
                    "success_factors": ["roi_demonstration", "stakeholder_alignment", "risk_mitigation"],
                    "retention_tactics": ["quarterly_business_reviews", "executive_sponsorship", "custom_solutions"]
                },
                "mid_market": {
                    "key_contacts": ["business_owner", "technical_lead", "end_users"],
                    "decision_makers": ["owner", "manager", "team_lead"],
                    "success_factors": ["value_delivery", "ease_of_use", "support_quality"],
                    "retention_tactics": ["regular_check_ins", "training_programs", "peer_networking"]
                },
                "small_business": {
                    "key_contacts": ["owner", "primary_user"],
                    "decision_makers": ["owner", "founder"],
                    "success_factors": ["simple_setup", "immediate_value", "cost_effectiveness"],
                    "retention_tactics": ["onboarding_support", "self_service_resources", "community_support"]
                }
            }
        }
    
    def load_retention_tactics(self) -> Dict[str, Any]:
        """Load specific retention tactics and playbooks"""
        return {
            "engagement_tactics": {
                "value_demonstration": {
                    "description": "Show clear value and ROI",
                    "actions": [
                        "generate_personalized_roi_report",
                        "create_success_metrics_dashboard",
                        "share_relevant_case_studies",
                        "schedule_value_review_meeting"
                    ],
                    "resources": ["roi_calculator", "case_study_library", "success_metrics"],
                    "timeline": "1-2 weeks",
                    "effectiveness_score": 0.85
                },
                "education_and_training": {
                    "description": "Improve product knowledge and adoption",
                    "actions": [
                        "assess_current_knowledge_level",
                        "create_personalized_training_plan",
                        "provide_on_demand_training_resources",
                        "schedule_live_training_sessions"
                    ],
                    "resources": ["training_platform", "course_library", "certification_programs"],
                    "timeline": "2-4 weeks",
                    "effectiveness_score": 0.78
                },
                "feature_optimization": {
                    "description": "Optimize feature usage for better outcomes",
                    "actions": [
                        "analyze_current_feature_usage",
                        "identify_underutilized_features",
                        "create_optimization_recommendations",
                        "provide_feature_adoption_support"
                    ],
                    "resources": ["feature_usage_analytics", "optimization_playbooks", "adoption_tools"],
                    "timeline": "1-3 weeks",
                    "effectiveness_score": 0.82
                },
                "community_engagement": {
                    "description": "Connect with peer customers",
                    "actions": [
                        "invite_to_customer_community",
                        "facilitate_peer_connections",
                        "organize_networking_events",
                        "create_knowledge_sharing_sessions"
                    ],
                    "resources": ["community_platform", "networking_tools", "event_platform"],
                    "timeline": "ongoing",
                    "effectiveness_score": 0.75
                }
            },
            "intervention_tactics": {
                "immediate_outreach": {
                    "description": "Direct contact with at-risk customers",
                    "actions": [
                        "schedule_executive_call",
                        "assign_dedicated_success_manager",
                        "conduct_comprehensive_health_check",
                        "create_personalized_action_plan"
                    ],
                    "resources": ["success_managers", "health_assessment_tools", "action_plan_templates"],
                    "timeline": "24-48 hours",
                    "effectiveness_score": 0.92
                },
                "solution_customization": {
                    "description": "Customize solution to meet specific needs",
                    "actions": [
                        "conduct_needs_assessment",
                        "design_custom_solution",
                        "implement_configuration_changes",
                        "monitor_solution_effectiveness"
                    ],
                    "resources": ["solution_architects", "configuration_tools", "monitoring_systems"],
                    "timeline": "1-2 weeks",
                    "effectiveness_score": 0.88
                },
                "financial_incentives": {
                    "description": "Offer financial incentives to retain",
                    "actions": [
                        "offer_discount_or_credit",
                        "provide_extended_payment_terms",
                        "create_loyalty_program_benefits",
                        "offer_free_upgrades"
                    ],
                    "resources": ["pricing_tools", "billing_systems", "loyalty_programs"],
                    "timeline": "immediate",
                    "effectiveness_score": 0.70
                },
                "executive_escalation": {
                    "description": "Escalate to executive level",
                    "actions": [
                        "notify_executive_team",
                        "schedule_c_level_meeting",
                        "present_customer_value_proposition",
                        "negotiate_retention_terms"
                    ],
                    "resources": ["executive_team", "value_proposition_templates", "negotiation_playbooks"],
                    "timeline": "1-3 days",
                    "effectiveness_score": 0.95
                }
            },
            "expansion_tactics": {
                "value_expansion": {
                    "description": "Expand value to increase stickiness",
                    "actions": [
                        "identify_expansion_opportunities",
                        "create_expansion_proposal",
                        "present_value_addition_plan",
                        "implement_expanded_solution"
                    ],
                    "resources": ["expansion_analytics", "proposal_templates", "implementation_team"],
                    "timeline": "2-6 weeks",
                    "effectiveness_score": 0.80
                },
                "additional_services": {
                    "description": "Offer complementary services",
                    "actions": [
                        "assess_service_needs",
                        "create_service_bundle_proposal",
                        "offer_pilot_programs",
                        "implement_selected_services"
                    ],
                    "resources": ["service_catalog", "pilot_programs", "service_teams"],
                    "timeline": "1-4 weeks",
                    "effectiveness_score": 0.75
                },
                "strategic_partnership": {
                    "description": "Develop strategic partnership",
                    "actions": [
                        "identify_partnership_opportunities",
                        "negotiate_partnership_terms",
                        "create_partnership_agreement",
                        "establish_partnership_operations"
                    ],
                    "resources": ["partnership_team", "legal_templates", "operations_team"],
                    "timeline": "1-3 months",
                    "effectiveness_score": 0.85
                }
            }
        }
    
    def initialize_segment_models(self):
        """Initialize customer segmentation models"""
        try:
            # Define customer segments
            self.customer_segments = {
                "champions": {
                    "description": "High-value, high-engagement customers",
                    "characteristics": {
                        "health_score": (80, 100),
                        "engagement_score": (75, 100),
                        "monthly_value": (1000, 10000),
                        "tenure_months": (12, 100)
                    },
                    "retention_strategy": "expansion",
                    "success_metrics": ["expansion_rate", "advocacy_score", "reference_willingness"]
                },
                "loyal_customers": {
                    "description": "Long-term, stable customers",
                    "characteristics": {
                        "health_score": (70, 100),
                        "engagement_score": (60, 90),
                        "monthly_value": (500, 3000),
                        "tenure_months": (24, 100)
                    },
                    "retention_strategy": "maintain_and_enhance",
                    "success_metrics": ["retention_rate", "satisfaction_score", "referral_rate"]
                },
                "at_risk": {
                    "description": "Customers showing signs of potential churn",
                    "characteristics": {
                        "health_score": (40, 70),
                        "engagement_score": (30, 70),
                        "monthly_value": (200, 2000),
                        "tenure_months": (3, 50)
                    },
                    "retention_strategy": "intervention",
                    "success_metrics": ["save_rate", "health_score_improvement", "engagement_recovery"]
                },
                "new_customers": {
                    "description": "Recently onboarded customers",
                    "characteristics": {
                        "health_score": (50, 90),
                        "engagement_score": (40, 100),
                        "monthly_value": (100, 5000),
                        "tenure_months": (0, 6)
                    },
                    "retention_strategy": "onboarding_and_engagement",
                    "success_metrics": ["onboarding_completion", "time_to_value", "early_adoption"]
                }
            }
            
            self.logger.info("Customer segment models initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing segment models: {str(e)}")
    
    def segment_customer(self, customer_data: Dict[str, Any]) -> str:
        """Segment customer based on characteristics"""
        try:
            health_score = customer_data.get("health_score", 50)
            engagement_score = customer_data.get("engagement_score", 50)
            monthly_value = customer_data.get("monthly_value", 100)
            tenure_months = customer_data.get("tenure_months", 0)
            
            # Evaluate against segment criteria
            for segment_name, segment_config in self.customer_segments.items():
                char = segment_config["characteristics"]
                
                health_match = char["health_score"][0] <= health_score <= char["health_score"][1]
                engagement_match = char["engagement_score"][0] <= engagement_score <= char["engagement_score"][1]
                value_match = char["monthly_value"][0] <= monthly_value <= char["monthly_value"][1]
                tenure_match = char["tenure_months"][0] <= tenure_months <= char["tenure_months"][1]
                
                if health_match and engagement_match and value_match and tenure_match:
                    return segment_name
            
            # Default to at_risk if no perfect match
            return "at_risk"
            
        except Exception as e:
            self.logger.error(f"Error segmenting customer: {str(e)}")
            return "unknown"
    
    def develop_retention_strategy(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop personalized retention strategy for customer"""
        try:
            # Segment customer
            segment = self.segment_customer(customer_data)
            
            # Get segment-specific strategy
            segment_strategy = self.customer_segments.get(segment, {})
            base_strategy = segment_strategy.get("retention_strategy", "standard")
            
            # Customize strategy based on customer risk level
            churn_risk = customer_data.get("churn_risk", 0)
            health_score = customer_data.get("health_score", 50)
            
            if churn_risk > 0.8 or health_score < 40:
                strategy_type = "critical_intervention"
            elif churn_risk > 0.6 or health_score < 60:
                strategy_type = "proactive_intervention"
            elif base_strategy == "expansion":
                strategy_type = "expansion_focused"
            else:
                strategy_type = "engagement_maintenance"
            
            # Generate strategic recommendations
            strategic_recommendations = self.generate_strategic_recommendations(
                customer_data, segment, strategy_type
            )
            
            # Select appropriate tactics
            selected_tactics = self.select_retention_tactics(
                customer_data, segment, strategy_type
            )
            
            # Create implementation timeline
            timeline = self.create_implementation_timeline(
                customer_data, strategy_type, selected_tactics
            )
            
            # Define success metrics
            success_metrics = self.define_success_metrics(
                customer_data, segment, strategy_type
            )
            
            return {
                "customer_id": customer_data.get("customer_id"),
                "segment": segment,
                "strategy_type": strategy_type,
                "base_strategy": base_strategy,
                "strategic_recommendations": strategic_recommendations,
                "selected_tactics": selected_tactics,
                "implementation_timeline": timeline,
                "success_metrics": success_metrics,
                "estimated_success_probability": self.calculate_success_probability(
                    customer_data, strategy_type, selected_tactics
                ),
                "resource_requirements": self.assess_resource_requirements(
                    customer_data, strategy_type
                ),
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error developing retention strategy: {str(e)}")
            return {}
    
    def generate_strategic_recommendations(self, customer_data: Dict[str, Any], 
                                         segment: str, strategy_type: str) -> List[Dict[str, Any]]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Segment-specific recommendations
        segment_recommendations = {
            "champions": [
                {
                    "category": "expansion",
                    "title": "Explore Advanced Features",
                    "description": "Customer shows readiness for advanced capabilities",
                    "priority": "high",
                    "impact": "high",
                    "effort": "medium"
                },
                {
                    "category": "advocacy",
                    "title": "Develop Advocacy Program",
                    "description": "Customer has high advocacy potential",
                    "priority": "medium",
                    "impact": "high",
                    "effort": "low"
                }
            ],
            "loyal_customers": [
                {
                    "category": "retention",
                    "title": "Maintain Engagement",
                    "description": "Continue current engagement level",
                    "priority": "medium",
                    "impact": "medium",
                    "effort": "low"
                },
                {
                    "category": "enhancement",
                    "title": "Optimize Current Usage",
                    "description": "Maximize value from existing features",
                    "priority": "medium",
                    "impact": "medium",
                    "effort": "medium"
                }
            ],
            "at_risk": [
                {
                    "category": "intervention",
                    "title": "Immediate Risk Assessment",
                    "description": "Conduct comprehensive risk analysis",
                    "priority": "critical",
                    "impact": "high",
                    "effort": "high"
                },
                {
                    "category": "engagement",
                    "title": "Re-engagement Campaign",
                    "description": "Implement targeted re-engagement efforts",
                    "priority": "high",
                    "impact": "high",
                    "effort": "high"
                }
            ],
            "new_customers": [
                {
                    "category": "onboarding",
                    "title": "Accelerate Onboarding",
                    "description": "Ensure quick time to value",
                    "priority": "high",
                    "impact": "high",
                    "effort": "medium"
                },
                {
                    "category": "education",
                    "title": "Provide Training Resources",
                    "description": "Ensure proper product knowledge",
                    "priority": "high",
                    "impact": "medium",
                    "effort": "medium"
                }
            ]
        }
        
        # Add segment-specific recommendations
        recommendations.extend(segment_recommendations.get(segment, []))
        
        # Add strategy-type specific recommendations
        if strategy_type == "critical_intervention":
            recommendations.append({
                "category": "escalation",
                "title": "Executive Escalation",
                "description": "Escalate to executive level for immediate attention",
                "priority": "critical",
                "impact": "high",
                "effort": "high"
            })
        elif strategy_type == "expansion_focused":
            recommendations.append({
                "category": "expansion",
                "title": "Expansion Opportunity Assessment",
                "description": "Identify and pursue expansion opportunities",
                "priority": "high",
                "impact": "high",
                "effort": "medium"
            })
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def select_retention_tactics(self, customer_data: Dict[str, Any], 
                               segment: str, strategy_type: str) -> List[Dict[str, Any]]:
        """Select appropriate retention tactics"""
        selected_tactics = []
        
        # Base tactics by strategy type
        strategy_tactics = {
            "critical_intervention": ["immediate_outreach", "executive_escalation", "solution_customization"],
            "proactive_intervention": ["immediate_outreach", "value_demonstration", "education_and_training"],
            "expansion_focused": ["value_expansion", "additional_services", "strategic_partnership"],
            "engagement_maintenance": ["community_engagement", "education_and_training", "feature_optimization"]
        }
        
        base_tactics = strategy_tactics.get(strategy_type, [])
        
        # Select tactics with effectiveness scores
        for tactic_key in base_tactics:
            if tactic_key in self.retention_tactics["engagement_tactics"]:
                tactic = self.retention_tactics["engagement_tactics"][tactic_key].copy()
                tactic["category"] = "engagement"
                tactic["key"] = tactic_key
                selected_tactics.append(tactic)
            elif tactic_key in self.retention_tactics["intervention_tactics"]:
                tactic = self.retention_tactics["intervention_tactics"][tactic_key].copy()
                tactic["category"] = "intervention"
                tactic["key"] = tactic_key
                selected_tactics.append(tactic)
            elif tactic_key in self.retention_tactics["expansion_tactics"]:
                tactic = self.retention_tactics["expansion_tactics"][tactic_key].copy()
                tactic["category"] = "expansion"
                tactic["key"] = tactic_key
                selected_tactics.append(tactic)
        
        # Sort by effectiveness score and select top tactics
        selected_tactics.sort(key=lambda x: x.get("effectiveness_score", 0.5), reverse=True)
        
        return selected_tactics[:3]  # Return top 3 tactics
    
    def create_implementation_timeline(self, customer_data: Dict[str, Any],
                                     strategy_type: str, tactics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create implementation timeline for retention strategy"""
        try:
            timeline = {
                "immediate_actions": [],
                "short_term_actions": [],
                "medium_term_actions": [],
                "ongoing_actions": []
            }
            
            current_date = datetime.now()
            
            for tactic in tactics:
                tactic_timeline = tactic.get("timeline", "1-2 weeks")
                
                # Parse timeline and add to appropriate category
                if "immediate" in tactic_timeline or "24-48" in tactic_timeline:
                    timeline["immediate_actions"].append({
                        "tactic": tactic["key"],
                        "description": tactic["description"],
                        "due_date": (current_date + timedelta(days=2)).isoformat(),
                        "priority": "high"
                    })
                elif "1-2 weeks" in tactic_timeline or "1-3 weeks" in tactic_timeline:
                    timeline["short_term_actions"].append({
                        "tactic": tactic["key"],
                        "description": tactic["description"],
                        "due_date": (current_date + timedelta(weeks=2)).isoformat(),
                        "priority": "medium"
                    })
                elif "2-6 weeks" in tactic_timeline or "1-4 weeks" in tactic_timeline:
                    timeline["medium_term_actions"].append({
                        "tactic": tactic["key"],
                        "description": tactic["description"],
                        "due_date": (current_date + timedelta(weeks=6)).isoformat(),
                        "priority": "medium"
                    })
                else:
                    timeline["ongoing_actions"].append({
                        "tactic": tactic["key"],
                        "description": tactic["description"],
                        "due_date": (current_date + timedelta(weeks=12)).isoformat(),
                        "priority": "low"
                    })
            
            # Add strategic milestones
            milestones = []
            
            if strategy_type in ["critical_intervention", "proactive_intervention"]:
                milestones.append({
                    "milestone": "Risk Assessment Complete",
                    "due_date": (current_date + timedelta(days=3)).isoformat(),
                    "description": "Complete comprehensive risk assessment"
                })
                milestones.append({
                    "milestone": "Intervention Success",
                    "due_date": (current_date + timedelta(weeks=4)).isoformat(),
                    "description": "Achieve measurable improvement in customer health"
                })
            elif strategy_type == "expansion_focused":
                milestones.append({
                    "milestone": "Expansion Opportunity Identified",
                    "due_date": (current_date + timedelta(weeks=2)).isoformat(),
                    "description": "Identify specific expansion opportunities"
                })
                milestones.append({
                    "milestone": "Expansion Implemented",
                    "due_date": (current_date + timedelta(weeks=8)).isoformat(),
                    "description": "Implement expansion and measure impact"
                })
            
            timeline["milestones"] = milestones
            timeline["strategy_duration"] = "12 weeks"  # Default strategy duration
            
            return timeline
            
        except Exception as e:
            self.logger.error(f"Error creating implementation timeline: {str(e)}")
            return {"error": str(e)}
    
    def define_success_metrics(self, customer_data: Dict[str, Any], 
                             segment: str, strategy_type: str) -> List[Dict[str, Any]]:
        """Define success metrics for retention strategy"""
        metrics = []
        
        # Base metrics by segment
        base_metrics = self.customer_segments.get(segment, {}).get("success_metrics", ["retention_rate"])
        
        # Strategy-specific metrics
        if strategy_type in ["critical_intervention", "proactive_intervention"]:
            metrics.extend([
                {"metric": "customer_health_score", "target": 75, "measurement": "score", "frequency": "weekly"},
                {"metric": "churn_risk_score", "target": 0.3, "measurement": "probability", "frequency": "weekly"},
                {"metric": "engagement_score", "target": 70, "measurement": "score", "frequency": "bi-weekly"},
                {"metric": "save_rate", "target": 0.8, "measurement": "percentage", "frequency": "monthly"}
            ])
        elif strategy_type == "expansion_focused":
            metrics.extend([
                {"metric": "revenue_growth", "target": 0.25, "measurement": "percentage", "frequency": "monthly"},
                {"metric": "feature_adoption", "target": 0.8, "measurement": "percentage", "frequency": "monthly"},
                {"metric": "stakeholder_satisfaction", "target": 4.5, "measurement": "rating", "frequency": "quarterly"}
            ])
        else:
            metrics.extend([
                {"metric": "retention_rate", "target": 0.95, "measurement": "percentage", "frequency": "monthly"},
                {"metric": "satisfaction_score", "target": 4.0, "measurement": "rating", "frequency": "quarterly"},
                {"metric": "engagement_score", "target": 75, "measurement": "score", "frequency": "monthly"}
            ])
        
        # Add base metrics if not already included
        for base_metric in base_metrics:
            if not any(m["metric"] == base_metric for m in metrics):
                if base_metric == "retention_rate":
                    metrics.append({"metric": "retention_rate", "target": 0.92, "measurement": "percentage", "frequency": "monthly"})
                elif base_metric == "satisfaction_score":
                    metrics.append({"metric": "satisfaction_score", "target": 4.0, "measurement": "rating", "frequency": "quarterly"})
        
        return metrics[:5]  # Return top 5 metrics
    
    def calculate_success_probability(self, customer_data: Dict[str, Any],
                                    strategy_type: str, tactics: List[Dict[str, Any]]) -> float:
        """Calculate probability of strategy success"""
        try:
            # Base success rates by strategy type
            base_success_rates = {
                "critical_intervention": 0.65,
                "proactive_intervention": 0.75,
                "expansion_focused": 0.80,
                "engagement_maintenance": 0.85
            }
            
            base_probability = base_success_rates.get(strategy_type, 0.70)
            
            # Adjust based on customer characteristics
            health_score = customer_data.get("health_score", 50)
            engagement_score = customer_data.get("engagement_score", 50)
            tenure_months = customer_data.get("tenure_months", 0)
            
            # Health score adjustment
            if health_score > 80:
                base_probability *= 1.1
            elif health_score < 40:
                base_probability *= 0.8
            
            # Engagement adjustment
            if engagement_score > 70:
                base_probability *= 1.05
            elif engagement_score < 40:
                base_probability *= 0.9
            
            # Tenure adjustment (longer tenure = more stable = higher success probability)
            if tenure_months > 24:
                base_probability *= 1.05
            elif tenure_months < 6:
                base_probability *= 0.95
            
            # Tactics effectiveness adjustment
            if tactics:
                avg_effectiveness = np.mean([tactic.get("effectiveness_score", 0.5) for tactic in tactics])
                base_probability *= (0.8 + avg_effectiveness * 0.2)
            
            return min(0.95, max(0.15, base_probability))
            
        except Exception as e:
            self.logger.error(f"Error calculating success probability: {str(e)}")
            return 0.70
    
    def assess_resource_requirements(self, customer_data: Dict[str, Any],
                                   strategy_type: str) -> Dict[str, Any]:
        """Assess resource requirements for retention strategy"""
        try:
            resource_requirements = {
                "human_resources": [],
                "technical_resources": [],
                "financial_resources": {},
                "time_requirements": {}
            }
            
            # Human resources based on strategy type
            if strategy_type in ["critical_intervention", "proactive_intervention"]:
                resource_requirements["human_resources"] = [
                    {"role": "Customer Success Manager", "allocation": "high", "duration": "4-8 weeks"},
                    {"role": "Executive Sponsor", "allocation": "medium", "duration": "2-4 weeks"},
                    {"role": "Solution Architect", "allocation": "medium", "duration": "1-2 weeks"}
                ]
            elif strategy_type == "expansion_focused":
                resource_requirements["human_resources"] = [
                    {"role": "Account Executive", "allocation": "high", "duration": "6-12 weeks"},
                    {"role": "Solution Consultant", "allocation": "medium", "duration": "4-8 weeks"},
                    {"role": "Product Specialist", "allocation": "low", "duration": "2-4 weeks"}
                ]
            else:
                resource_requirements["human_resources"] = [
                    {"role": "Customer Success Manager", "allocation": "medium", "duration": "ongoing"},
                    {"role": "Support Specialist", "allocation": "low", "duration": "as needed"}
                ]
            
            # Technical resources
            resource_requirements["technical_resources"] = [
                {"resource": "Health Monitoring Dashboard", "availability": "immediate"},
                {"resource": "Customer Data Analytics", "availability": "immediate"},
                {"resource": "Communication Platform", "availability": "immediate"},
                {"resource": "Training Resources", "availability": "immediate"}
            ]
            
            # Financial resources
            company_size = customer_data.get("company_size", "medium")
            if strategy_type == "critical_intervention":
                resource_requirements["financial_resources"] = {
                    "estimated_cost": {"small": 5000, "medium": 10000, "large": 20000, "enterprise": 50000}.get(company_size, 10000),
                    "cost_categories": ["personnel_time", "incentives", "solution_customization"]
                }
            elif strategy_type == "expansion_focused":
                resource_requirements["financial_resources"] = {
                    "estimated_cost": {"small": 2000, "medium": 5000, "large": 10000, "enterprise": 25000}.get(company_size, 5000),
                    "cost_categories": ["sales_time", "proposal_development", "implementation"]
                }
            else:
                resource_requirements["financial_resources"] = {
                    "estimated_cost": {"small": 1000, "medium": 2500, "large": 5000, "enterprise": 10000}.get(company_size, 2500),
                    "cost_categories": ["support_time", "training_resources", "communication"]
                }
            
            return resource_requirements
            
        except Exception as e:
            self.logger.error(f"Error assessing resource requirements: {str(e)}")
            return {"error": str(e)}
    
    def execute_retention_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retention strategy"""
        try:
            execution_result = {
                "strategy_id": strategy.get("customer_id"),
                "execution_status": "initiated",
                "actions_taken": [],
                "timeline_progress": {},
                "success_metrics_progress": {},
                "created_at": datetime.now().isoformat()
            }
            
            # Execute immediate actions
            immediate_actions = strategy.get("implementation_timeline", {}).get("immediate_actions", [])
            for action in immediate_actions:
                execution_result["actions_taken"].append({
                    "action": action["tactic"],
                    "status": "initiated",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Update timeline progress
            execution_result["timeline_progress"] = {
                "immediate_actions": len(immediate_actions),
                "scheduled_actions": 0,
                "completed_actions": 0
            }
            
            # In a real implementation, this would trigger actual actions
            # For demo purposes, we'll simulate execution
            
            execution_result["execution_status"] = "in_progress"
            execution_result["next_milestone"] = strategy.get("implementation_timeline", {}).get("milestones", [{}])[0].get("milestone", "Risk Assessment")
            
            self.logger.info(f"Executed retention strategy for customer {strategy.get('customer_id')}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Error executing retention strategy: {str(e)}")
            return {"error": str(e)}
    
    def monitor_retention_performance(self) -> Dict[str, Any]:
        """Monitor overall retention performance"""
        try:
            # In real implementation, would calculate from actual data
            # For demo, return simulated metrics
            
            performance_metrics = {
                "overall_retention_rate": 0.92,
                "monthly_churn_rate": 0.08,
                "save_rate": 0.73,
                "intervention_success_rate": 0.81,
                "average_save_value": 15750,
                "customer_health_distribution": {
                    "excellent": 35,  # 85-100
                    "good": 40,       # 70-84
                    "fair": 18,       # 50-69
                    "poor": 7         # 0-49
                },
                "retention_by_segment": {
                    "champions": 0.98,
                    "loyal_customers": 0.95,
                    "at_risk": 0.67,
                    "new_customers": 0.84
                },
                "monthly_trends": {
                    "retention_rate_trend": "improving",
                    "churn_rate_trend": "stable",
                    "save_rate_trend": "improving"
                },
                "intervention_effectiveness": {
                    "critical_intervention": 0.78,
                    "proactive_intervention": 0.83,
                    "expansion_focused": 0.87,
                    "engagement_maintenance": 0.91
                }
            }
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring retention performance: {str(e)}")
            return {"error": str(e)}
    
    def optimize_retention_strategies(self) -> Dict[str, Any]:
        """Optimize retention strategies based on performance data"""
        try:
            optimization_result = {
                "optimization_timestamp": datetime.now().isoformat(),
                "strategies_analyzed": 0,
                "optimizations_identified": [],
                "performance_improvements": {},
                "recommendations": []
            }
            
            # Analyze current performance
            current_performance = self.monitor_retention_performance()
            
            # Identify optimization opportunities
            if current_performance.get("overall_retention_rate", 0) < 0.90:
                optimization_result["optimizations_identified"].append({
                    "area": "Overall Retention",
                    "opportunity": "Improve retention rate through enhanced intervention strategies",
                    "potential_impact": "3-5% improvement",
                    "recommended_actions": [
                        "Implement more proactive monitoring",
                        "Enhance early intervention tactics",
                        "Improve customer segmentation"
                    ]
                })
            
            if current_performance.get("save_rate", 0) < 0.75:
                optimization_result["optimizations_identified"].append({
                    "area": "Save Rate",
                    "opportunity": "Increase save rate for at-risk customers",
                    "potential_impact": "10-15% improvement in saves",
                    "recommended_actions": [
                        "Refine churn prediction models",
                        "Improve intervention timing",
                        "Enhance solution customization"
                    ]
                })
            
            # Analyze segment-specific optimizations
            retention_by_segment = current_performance.get("retention_by_segment", {})
            for segment, rate in retention_by_segment.items():
                if rate < 0.90:
                    optimization_result["optimizations_identified"].append({
                        "area": f"{segment.title()} Segment",
                        "opportunity": f"Improve retention for {segment} customers",
                        "potential_impact": f"{(0.90 - rate) * 100:.1f}% improvement",
                        "recommended_actions": [
                            f"Enhance {segment} specific strategies",
                            "Improve segment identification accuracy",
                            "Customize tactics for segment characteristics"
                        ]
                    })
            
            # Generate optimization recommendations
            optimization_result["recommendations"] = [
                {
                    "priority": "high",
                    "action": "Implement predictive retention scoring",
                    "description": "Deploy advanced ML models for better churn prediction",
                    "timeline": "4-6 weeks",
                    "resource_requirement": "medium"
                },
                {
                    "priority": "medium", 
                    "action": "Enhance customer segmentation",
                    "description": "Improve customer segmentation with behavioral clustering",
                    "timeline": "6-8 weeks",
                    "resource_requirement": "high"
                },
                {
                    "priority": "medium",
                    "action": "Optimize intervention timing",
                    "description": "Fine-tune intervention triggers based on historical data",
                    "timeline": "2-3 weeks",
                    "resource_requirement": "low"
                }
            ]
            
            self.logger.info("Retention strategy optimization completed")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error optimizing retention strategies: {str(e)}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get retention engine status"""
        return {
            "status": "operational",
            "segments_supported": list(self.customer_segments.keys()),
            "strategies_available": len(self.retention_strategies.get("retention_strategies", {})),
            "tactics_available": sum(
                len(category.get("engagement_tactics", [])) + 
                len(category.get("intervention_tactics", [])) + 
                len(category.get("expansion_tactics", []))
                for category in [self.retention_tactics]
            ),
            "current_performance": self.monitor_retention_performance(),
            "last_optimization": datetime.now().isoformat(),
            "active_customers": 1000  # Demo value
        }
