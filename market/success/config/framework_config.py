"""
Customer Success Framework Configuration
Enterprise healthcare customer success and retention management system
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum
import datetime

class CustomerTier(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    STRATEGIC = "strategic"

class HealthScoreStatus(Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    CRITICAL = "critical"

class InterventionLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthcareKPI:
    """Healthcare-specific Key Performance Indicators"""
    clinical_outcome_improvement: float  # % improvement in patient outcomes
    clinical_efficiency_gain: float     # % improvement in clinical workflows
    cost_reduction: float              # % reduction in operational costs
    compliance_score: float           # Regulatory compliance score (0-100)
    staff_satisfaction: float         # Healthcare staff satisfaction (0-100)
    patient_satisfaction: float       # Patient satisfaction score (0-100)
    roi_percentage: float             # Return on investment percentage
    implementation_success_rate: float # Successful deployment rate

@dataclass
class CustomerSegment:
    """Customer segmentation for healthcare organizations"""
    segment_name: str
    organization_type: str  # Hospital, Clinic, Health System, etc.
    size_category: str      # Small, Medium, Large, Enterprise
    clinical_specialty: str # Cardiology, Oncology, etc.
    geographic_region: str
    maturity_level: str     # Early, Growing, Mature
    tech_adoption: str      # Conservative, Moderate, Progressive, Innovative

@dataclass
class SuccessMetrics:
    """Comprehensive success metrics tracking"""
    customer_health_score: float
    nps_score: float
    churn_risk_score: float
    expansion_potential: float
    engagement_level: float
    clinical_impact_score: float
    support_ticket_volume: int
    feature_adoption_rate: float
    roi_delivery_score: float

class HealthcareCSConfig:
    """Configuration for healthcare customer success framework"""
    
    # Health Score Thresholds
    HEALTH_SCORE_THRESHOLDS = {
        HealthScoreStatus.GREEN: (80, 100),
        HealthScoreStatus.YELLOW: (60, 79),
        HealthScoreStatus.RED: (40, 59),
        HealthScoreStatus.CRITICAL: (0, 39)
    }
    
    # Intervention Triggers
    INTERVENTION_TRIGGERS = {
        InterventionLevel.INFO: {
            "health_score_drop": 15,
            "nps_score": 8,
            "support_tickets": 2,
            "engagement_decline": 0.2
        },
        InterventionLevel.WARNING: {
            "health_score_drop": 25,
            "nps_score": 6,
            "support_tickets": 5,
            "engagement_decline": 0.35
        },
        InterventionLevel.CRITICAL: {
            "health_score_drop": 35,
            "nps_score": 4,
            "support_tickets": 8,
            "engagement_decline": 0.5
        },
        InterventionLevel.EMERGENCY: {
            "health_score_drop": 50,
            "nps_score": 2,
            "support_tickets": 12,
            "engagement_decline": 0.65
        }
    }
    
    # Customer Success Manager Workload
    CSM_WORKLOAD_LIMITS = {
        CustomerTier.ENTERPRISE: 15,   # Max customers per CSM
        CustomerTier.PREMIUM: 25,      # Max customers per CSM
        CustomerTier.STANDARD: 50,     # Max customers per CSM
        CustomerTier.BASIC: 100        # Max customers per CSM
    }
    
    # Review Schedules
    REVIEW_SCHEDULES = {
        CustomerTier.STRATEGIC: "monthly",     # Monthly strategic reviews
        CustomerTier.ENTERPRISE: "quarterly",  # Quarterly business reviews
        CustomerTier.PREMIUM: "quarterly",     # Quarterly business reviews
        CustomerTier.STANDARD: "semiannual",   # Semi-annual reviews
        CustomerTier.BASIC: "annual"           # Annual check-ins
    }
    
    # Health Score Components (weighted percentages)
    HEALTH_SCORE_WEIGHTS = {
        "product_usage": 0.25,
        "clinical_outcomes": 0.25,
        "financial_health": 0.20,
        "customer_satisfaction": 0.15,
        "support_health": 0.10,
        "engagement_level": 0.05
    }
    
    # Risk Factors for Churn Prediction
    CHURN_RISK_FACTORS = {
        "low_engagement": 0.2,
        "negative_outcomes": 0.25,
        "support_issues": 0.15,
        "competing_solution": 0.20,
        "financial_stress": 0.10,
        "organizational_changes": 0.10
    }
    
    # Expansion Revenue Opportunities
    EXPANSION_OPPORTUNITIES = {
        "user_licenses": {
            "threshold": 80,  # License utilization %
            "opportunity_score": 0.8
        },
        "feature_adoption": {
            "threshold": 60,  # Feature usage %
            "opportunity_score": 0.7
        },
        "workflow_optimization": {
            "threshold": 70,  # Workflow adoption %
            "opportunity_score": 0.9
        },
        "integrations": {
            "threshold": 50,  # Integration utilization %
            "opportunity_score": 0.6
        }
    }
    
    @classmethod
    def get_health_score_status(cls, score: float) -> HealthScoreStatus:
        """Determine health score status based on score value"""
        for status, (min_score, max_score) in cls.HEALTH_SCORE_THRESHOLDS.items():
            if min_score <= score <= max_score:
                return status
        return HealthScoreStatus.CRITICAL
    
    @classmethod
    def get_intervention_level(cls, metrics: Dict) -> InterventionLevel:
        """Determine intervention level based on customer metrics"""
        scores = {}
        
        # Calculate health score drop
        if "current_health_score" in metrics and "previous_health_score" in metrics:
            health_drop = max(0, metrics["previous_health_score"] - metrics["current_health_score"])
            scores["health_score_drop"] = health_drop
        
        # NPS score
        if "nps_score" in metrics:
            scores["nps_score"] = metrics["nps_score"]
        
        # Support tickets
        if "support_tickets" in metrics:
            scores["support_tickets"] = metrics["support_tickets"]
        
        # Engagement decline
        if "current_engagement" in metrics and "previous_engagement" in metrics:
            engagement_decline = max(0, metrics["previous_engagement"] - metrics["current_engagement"])
            scores["engagement_decline"] = engagement_decline
        
        # Determine highest intervention level triggered
        max_level = InterventionLevel.INFO
        for level, triggers in cls.INTERVENTION_TRIGGERS.items():
            triggered = False
            for metric, threshold in triggers.items():
                if metric in scores and scores[metric] >= threshold:
                    triggered = True
                    break
            
            if triggered:
                level_priority = {
                    InterventionLevel.INFO: 0,
                    InterventionLevel.WARNING: 1,
                    InterventionLevel.CRITICAL: 2,
                    InterventionLevel.EMERGENCY: 3
                }
                if level_priority[level] > level_priority[max_level]:
                    max_level = level
        
        return max_level
    
    @classmethod
    def calculate_expansion_score(cls, customer_data: Dict) -> float:
        """Calculate expansion revenue potential score"""
        expansion_scores = []
        
        for opportunity_type, criteria in cls.EXPANSION_OPPORTUNITIES.items():
            if opportunity_type in customer_data:
                current_usage = customer_data[opportunity_type]
                threshold = criteria["threshold"]
                opportunity_score = criteria["opportunity_score"]
                
                # Calculate potential based on gap from threshold
                if current_usage >= threshold:
                    potential = opportunity_score * (current_usage / 100)
                    expansion_scores.append(potential)
        
        return sum(expansion_scores) / len(expansion_scores) if expansion_scores else 0.0