"""
Customer Feedback Loops and Product Improvement Process
Systematic feedback collection, analysis, and product enhancement for healthcare AI
"""

import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

class FeedbackType(Enum):
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    USABILITY_FEEDBACK = "usability_feedback"
    CLINICAL_FEEDBACK = "clinical_feedback"
    INTEGRATION_ISSUE = "integration_issue"
    PERFORMANCE_FEEDBACK = "performance_feedback"
    TRAINING_REQUEST = "training_request"
    GENERAL_FEEDBACK = "general_feedback"

class FeedbackPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FeedbackStatus(Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    UNDER_REVIEW = "under_review"
    PLANNED = "planned"
    IN_DEVELOPMENT = "in_development"
    IMPLEMENTED = "implemented"
    CLOSED = "closed"

class ProductImpact(Enum):
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CustomerFeedback:
    """Customer feedback entry"""
    feedback_id: str
    customer_id: str
    feedback_type: FeedbackType
    title: str
    description: str
    priority: FeedbackPriority
    status: FeedbackStatus
    
    # Metadata
    submitted_by: str
    submitted_date: datetime.datetime
    assigned_to: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Analysis
    business_impact: ProductImpact = ProductImpact.LOW
    clinical_impact: ProductImpact = ProductImpact.LOW
    user_impact_score: float = 5.0  # 1-10 scale
    
    # Tracking
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)
    customer_satisfaction_impact: float = 0.0  # -10 to +10 scale
    related_feedback: List[str] = field(default_factory=list)
    customer_updates: List[str] = field(default_factory=list)
    resolution_notes: List[str] = field(default_factory=list)

@dataclass
class ProductEnhancementRequest:
    """Product enhancement request from customer feedback"""
    request_id: str
    feedback_ids: List[str]  # Multiple feedback items that led to this request
    title: str
    description: str
    
    # Business Case
    business_value: str
    customer_impact: str
    competitive_advantage: str
    
    # Technical Details
    development_effort: str  # small, medium, large, very_large
    technical_complexity: str  # low, medium, high, very_high
    dependencies: List[str] = field(default_factory=list)
    
    # Planning
    target_release: str = ""
    estimated_completion: Optional[datetime.date] = None
    assigned_team: str = ""
    status: str = "proposed"
    
    # Tracking
    created_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    approved_date: Optional[datetime.datetime] = None
    implemented_date: Optional[datetime.datetime] = None

@dataclass
class FeedbackCampaign:
    """Targeted feedback collection campaign"""
    campaign_id: str
    name: str
    description: str
    target_customer_segments: List[str]
    feedback_types: List[FeedbackType]
    
    # Campaign Details
    start_date: datetime.date
    end_date: datetime.date
    status: str = "active"
    
    # Goals
    target_response_count: int = 0
    current_response_count: int = 0
    response_rate: float = 0.0
    
    # Messaging
    campaign_message: str = ""
    follow_up_message: str = ""
    
    # Incentives
    incentives: List[str] = field(default_factory=list)

@dataclass
class FeedbackInsight:
    """Analyzed feedback insight"""
    insight_id: str
    insight_type: str  # trend, pattern, issue, opportunity
    title: str
    description: str
    
    # Analysis
    supporting_feedback_count: int
    confidence_level: float
    business_impact: ProductImpact
    recommended_actions: List[str]
    
    # Tracking
    created_date: datetime.datetime = field(default_factory=datetime.datetime.now)
    valid_until: Optional[datetime.date] = None
    status: str = "active"

class FeedbackLoopManager:
    """Customer feedback loop and product improvement management system"""
    
    def __init__(self):
        self.feedback: Dict[str, CustomerFeedback] = {}
        self.enhancement_requests: Dict[str, ProductEnhancementRequest] = {}
        self.campaigns: Dict[str, FeedbackCampaign] = {}
        self.insights: Dict[str, FeedbackInsight] = {}
        self.feedback_trends: Dict[str, List] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize feedback categories
        self.feedback_categories = self._initialize_feedback_categories()
        
        # Initialize product roadmap alignment
        self.roadmap_priorities = self._initialize_roadmap_priorities()
    
    def _initialize_feedback_categories(self) -> Dict[FeedbackType, Dict]:
        """Initialize feedback categories with handling guidelines"""
        return {
            FeedbackType.FEATURE_REQUEST: {
                "description": "Request for new features or functionality",
                "default_priority": FeedbackPriority.MEDIUM,
                "handling_team": "product_team",
                "response_time_hours": 48
            },
            FeedbackType.BUG_REPORT: {
                "description": "Report of software bugs or errors",
                "default_priority": FeedbackPriority.HIGH,
                "handling_team": "engineering_team",
                "response_time_hours": 24
            },
            FeedbackType.CLINICAL_FEEDBACK: {
                "description": "Feedback related to clinical workflows and outcomes",
                "default_priority": FeedbackPriority.HIGH,
                "handling_team": "clinical_team",
                "response_time_hours": 24
            },
            FeedbackType.INTEGRATION_ISSUE: {
                "description": "Issues with system integrations",
                "default_priority": FeedbackPriority.HIGH,
                "handling_team": "integration_team",
                "response_time_hours": 12
            },
            FeedbackType.USABILITY_FEEDBACK: {
                "description": "User interface and experience feedback",
                "default_priority": FeedbackPriority.MEDIUM,
                "handling_team": "ux_team",
                "response_time_hours": 48
            },
            FeedbackType.TRAINING_REQUEST: {
                "description": "Requests for training or education",
                "default_priority": FeedbackPriority.MEDIUM,
                "handling_team": "customer_success",
                "response_time_hours": 24
            }
        }
    
    def _initialize_roadmap_priorities(self) -> Dict[str, Any]:
        """Initialize product roadmap priorities"""
        return {
            "healthcare_workflows": {
                "priority_level": 1,
                "description": "Enhancements to clinical workflows",
                "value_multiplier": 1.5
            },
            "clinical_outcomes": {
                "priority_level": 1,
                "description": "Features that improve patient outcomes",
                "value_multiplier": 2.0
            },
            "integration_capabilities": {
                "priority_level": 2,
                "description": "System integration improvements",
                "value_multiplier": 1.3
            },
            "user_experience": {
                "priority_level": 3,
                "description": "UI/UX improvements",
                "value_multiplier": 1.1
            },
            "performance_optimization": {
                "priority_level": 2,
                "description": "System performance improvements",
                "value_multiplier": 1.2
            }
        }
    
    def submit_customer_feedback(self, customer_id: str, feedback_type: FeedbackType,
                               title: str, description: str, submitted_by: str,
                               priority: Optional[FeedbackPriority] = None) -> CustomerFeedback:
        """Submit customer feedback"""
        
        # Determine priority if not specified
        if priority is None:
            category_info = self.feedback_categories.get(feedback_type, {})
            priority = category_info.get("default_priority", FeedbackPriority.MEDIUM)
        
        feedback = CustomerFeedback(
            feedback_id=f"fb_{customer_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=customer_id,
            feedback_type=feedback_type,
            title=title,
            description=description,
            priority=priority,
            status=FeedbackStatus.NEW,
            submitted_by=submitted_by,
            submitted_date=datetime.datetime.now(),
            category=feedback_type.value,
            tags=self._generate_feedback_tags(feedback_type, description)
        )
        
        self.feedback[feedback.feedback_id] = feedback
        
        # Auto-assign based on feedback type
        category_info = self.feedback_categories.get(feedback_type, {})
        feedback.assigned_to = category_info.get("handling_team", "general")
        
        # Check for related feedback
        related_feedback = self._find_related_feedback(feedback)
        feedback.related_feedback = related_feedback
        
        # Trigger immediate actions for critical feedback
        if priority == FeedbackPriority.CRITICAL:
            self._trigger_critical_feedback_response(feedback)
        
        self.logger.info(f"Submitted feedback {feedback.feedback_id} from customer {customer_id}")
        return feedback
    
    def _generate_feedback_tags(self, feedback_type: FeedbackType, description: str) -> List[str]:
        """Generate relevant tags for feedback"""
        tags = [feedback_type.value]
        
        # Add clinical workflow tags
        if "workflow" in description.lower():
            tags.append("workflow")
        if "integration" in description.lower():
            tags.append("integration")
        if "clinical" in description.lower():
            tags.append("clinical")
        if "performance" in description.lower():
            tags.append("performance")
        if "training" in description.lower():
            tags.append("training")
        if "usability" in description.lower() or "ui" in description.lower() or "ux" in description.lower():
            tags.append("usability")
        
        return tags
    
    def _find_related_feedback(self, new_feedback: CustomerFeedback) -> List[str]:
        """Find related feedback based on keywords and type"""
        related = []
        
        # Find feedback with same type
        same_type_feedback = [
            fb for fb in self.feedback.values()
            if fb.feedback_type == new_feedback.feedback_type and fb.feedback_id != new_feedback.feedback_id
        ]
        
        # Check for keyword similarity
        for feedback in same_type_feedback:
            similarity_score = self._calculate_text_similarity(
                new_feedback.description, feedback.description
            )
            if similarity_score > 0.6:  # 60% similarity threshold
                related.append(feedback.feedback_id)
        
        return related[:5]  # Limit to top 5 related items
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        # Simple word-based similarity (would use more sophisticated NLP in production)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _trigger_critical_feedback_response(self, feedback: CustomerFeedback):
        """Trigger immediate response for critical feedback"""
        # In a real implementation, this would:
        # 1. Send immediate notifications to relevant teams
        # 2. Create emergency support ticket
        # 3. Assign senior engineer/manager
        # 4. Schedule immediate customer call
        
        self.logger.critical(f"Critical feedback received: {feedback.feedback_id}")
    
    def analyze_feedback_trends(self, timeframe_days: int = 30) -> Dict:
        """Analyze feedback trends over specified timeframe"""
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=timeframe_days)
        
        recent_feedback = [
            fb for fb in self.feedback.values()
            if fb.submitted_date >= cutoff_date
        ]
        
        if not recent_feedback:
            return {"message": "No feedback in specified timeframe"}
        
        # Analyze by type
        type_analysis = {}
        for fb_type in FeedbackType:
            type_count = len([fb for fb in recent_feedback if fb.feedback_type == fb_type])
            if type_count > 0:
                type_analysis[fb_type.value] = {
                    "count": type_count,
                    "percentage": (type_count / len(recent_feedback)) * 100,
                    "avg_priority": self._calculate_avg_priority([fb for fb in recent_feedback if fb.feedback_type == fb_type])
                }
        
        # Analyze priority distribution
        priority_analysis = {}
        for priority in FeedbackPriority:
            count = len([fb for fb in recent_feedback if fb.priority == priority])
            if count > 0:
                priority_analysis[priority.value] = {
                    "count": count,
                    "percentage": (count / len(recent_feedback)) * 100
                }
        
        # Analyze customer satisfaction impact
        satisfaction_scores = [fb.customer_satisfaction_impact for fb in recent_feedback if fb.customer_satisfaction_impact != 0]
        avg_satisfaction_impact = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0
        
        # Identify top themes
        top_themes = self._identify_top_feedback_themes(recent_feedback)
        
        # Analyze resolution times
        resolved_feedback = [fb for fb in recent_feedback if fb.status in [FeedbackStatus.CLOSED, FeedbackStatus.IMPLEMENTED]]
        avg_resolution_time = self._calculate_avg_resolution_time(resolved_feedback)
        
        return {
            "timeframe_days": timeframe_days,
            "total_feedback": len(recent_feedback),
            "feedback_by_type": type_analysis,
            "priority_distribution": priority_analysis,
            "customer_satisfaction_impact": {
                "average_impact": avg_satisfaction_impact,
                "positive_feedback": len([s for s in satisfaction_scores if s > 0]),
                "negative_feedback": len([s for s in satisfaction_scores if s < 0])
            },
            "top_themes": top_themes,
            "resolution_metrics": {
                "total_resolved": len(resolved_feedback),
                "average_resolution_time_hours": avg_resolution_time,
                "resolution_rate": (len(resolved_feedback) / len(recent_feedback)) * 100 if recent_feedback else 0
            }
        }
    
    def _calculate_avg_priority(self, feedback_list: List[CustomerFeedback]) -> str:
        """Calculate average priority for a list of feedback"""
        priority_values = {
            FeedbackPriority.LOW: 1,
            FeedbackPriority.MEDIUM: 2,
            FeedbackPriority.HIGH: 3,
            FeedbackPriority.CRITICAL: 4
        }
        
        total_priority = sum(priority_values.get(fb.priority, 0) for fb in feedback_list)
        avg_priority_value = total_priority / len(feedback_list) if feedback_list else 0
        
        # Convert back to priority enum
        for priority, value in priority_values.items():
            if abs(avg_priority_value - value) < 0.5:
                return priority.value
        
        return FeedbackPriority.MEDIUM.value
    
    def _identify_top_feedback_themes(self, feedback_list: List[CustomerFeedback]) -> List[Dict]:
        """Identify top themes from feedback"""
        theme_count = {}
        
        for feedback in feedback_list:
            for tag in feedback.tags:
                if tag not in theme_count:
                    theme_count[tag] = {"count": 0, "feedback_ids": []}
                theme_count[tag]["count"] += 1
                theme_count[tag]["feedback_ids"].append(feedback.feedback_id)
        
        # Sort by count and return top 10
        sorted_themes = sorted(theme_count.items(), key=lambda x: x[1]["count"], reverse=True)
        
        return [
            {
                "theme": theme,
                "count": data["count"],
                "percentage": (data["count"] / len(feedback_list)) * 100
            }
            for theme, data in sorted_themes[:10]
        ]
    
    def _calculate_avg_resolution_time(self, resolved_feedback: List[CustomerFeedback]) -> float:
        """Calculate average resolution time in hours"""
        if not resolved_feedback:
            return 0.0
        
        total_time = 0
        resolved_count = 0
        
        for feedback in resolved_feedback:
            # Find the most recent update that indicates resolution
            for update in feedback.customer_updates:
                if "resolved" in update.lower() or "implemented" in update.lower():
                    # Would extract timestamps in real implementation
                    # For now, use a simplified calculation
                    resolution_time = (datetime.datetime.now() - feedback.submitted_date).total_seconds() / 3600
                    total_time += resolution_time
                    resolved_count += 1
                    break
        
        return total_time / resolved_count if resolved_count > 0 else 0.0
    
    def create_enhancement_request(self, feedback_ids: List[str], title: str, description: str,
                                 business_value: str) -> ProductEnhancementRequest:
        """Create product enhancement request from feedback"""
        
        # Validate feedback exists
        existing_feedback = [fb for fb in feedback_ids if fb in self.feedback]
        if not existing_feedback:
            raise ValueError("No valid feedback IDs provided")
        
        # Analyze feedback to determine effort and complexity
        development_effort, technical_complexity = self._analyze_enhancement_complexity(
            existing_feedback, description
        )
        
        # Determine business and clinical impact
        business_impact = self._assess_business_impact(existing_feedback)
        clinical_impact = self._assess_clinical_impact(existing_feedback)
        
        request = ProductEnhancementRequest(
            request_id=f"enh_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            feedback_ids=feedback_ids,
            title=title,
            description=description,
            business_value=business_value,
            customer_impact=self._summarize_customer_impact(existing_feedback),
            competitive_advantage=self._assess_competitive_advantage(existing_feedback),
            development_effort=development_effort,
            technical_complexity=technical_complexity
        )
        
        self.enhancement_requests[request.request_id] = request
        
        # Update feedback status
        for feedback_id in existing_feedback:
            feedback = self.feedback[feedback_id]
            feedback.status = FeedbackStatus.PLANNED
            feedback.last_updated = datetime.datetime.now()
        
        self.logger.info(f"Created enhancement request {request.request_id} from {len(existing_feedback)} feedback items")
        return request
    
    def _analyze_enhancement_complexity(self, feedback: List[CustomerFeedback], description: str) -> Tuple[str, str]:
        """Analyze development effort and technical complexity"""
        
        # Keywords indicating complexity
        integration_keywords = ["integration", "api", "system", "connect", "import", "export"]
        workflow_keywords = ["workflow", "process", "automate", "step", "stage"]
        clinical_keywords = ["clinical", "patient", "medical", "diagnosis", "treatment"]
        ui_keywords = ["interface", "ui", "ux", "display", "screen", "visual"]
        
        complexity_score = 0
        integration_count = sum(1 for keyword in integration_keywords if keyword in description.lower())
        workflow_count = sum(1 for keyword in workflow_keywords if keyword in description.lower())
        clinical_count = sum(1 for keyword in clinical_keywords if keyword in description.lower())
        ui_count = sum(1 for keyword in ui_keywords if keyword in description.lower())
        
        complexity_score += integration_count * 3  # Integrations are complex
        complexity_score += workflow_count * 2    # Workflows are medium complexity
        complexity_score += clinical_count * 2    # Clinical features are medium complexity
        complexity_score += ui_count * 1          # UI changes are lower complexity
        
        # Determine effort level
        if complexity_score <= 2:
            development_effort = "small"
            technical_complexity = "low"
        elif complexity_score <= 5:
            development_effort = "medium"
            technical_complexity = "medium"
        elif complexity_score <= 8:
            development_effort = "large"
            technical_complexity = "high"
        else:
            development_effort = "very_large"
            technical_complexity = "very_high"
        
        return development_effort, technical_complexity
    
    def _assess_business_impact(self, feedback: List[CustomerFeedback]) -> str:
        """Assess business impact of enhancement"""
        # Count high-value feedback
        high_value_feedback = [fb for fb in feedback if fb.business_impact in [ProductImpact.HIGH, ProductImpact.CRITICAL]]
        
        if len(high_value_feedback) >= 3:
            return "High business impact - multiple enterprise customers requesting"
        elif len(high_value_feedback) >= 1:
            return "Medium business impact - strategic customer request"
        else:
            return "Low business impact - incremental improvement"
    
    def _assess_clinical_impact(self, feedback: List[CustomerFeedback]) -> str:
        """Assess clinical impact of enhancement"""
        clinical_feedback = [fb for fb in feedback if fb.feedback_type == FeedbackType.CLINICAL_FEEDBACK]
        high_impact_clinical = [fb for fb in clinical_feedback if fb.clinical_impact in [ProductImpact.HIGH, ProductImpact.CRITICAL]]
        
        if len(high_impact_clinical) >= 2:
            return "High clinical impact - affects patient outcomes"
        elif len(clinical_feedback) >= 1:
            return "Medium clinical impact - improves clinical workflows"
        else:
            return "Low clinical impact - convenience improvement"
    
    def _summarize_customer_impact(self, feedback: List[CustomerFeedback]) -> str:
        """Summarize customer impact from feedback"""
        customer_ids = set(fb.customer_id for fb in feedback)
        avg_satisfaction_impact = sum(fb.customer_satisfaction_impact for fb in feedback) / len(feedback)
        
        return f"Impacts {len(customer_ids)} customers with avg satisfaction impact of {avg_satisfaction_impact:.1f}"
    
    def _assess_competitive_advantage(self, feedback: List[CustomerFeedback]) -> str:
        """Assess competitive advantage of enhancement"""
        # Check if feedback mentions competitive features
        competitive_mentions = 0
        for fb in feedback:
            if any(keyword in fb.description.lower() for keyword in ["competitor", "alternative", "instead of", "like"]):
                competitive_mentions += 1
        
        if competitive_mentions >= 2:
            return "High competitive advantage - directly addresses competitor gaps"
        elif competitive_mentions >= 1:
            return "Medium competitive advantage - improves competitive position"
        else:
            return "Low competitive advantage - internal improvement"
    
    def launch_feedback_campaign(self, name: str, description: str, target_segments: List[str],
                               feedback_types: List[FeedbackType], duration_days: int = 30) -> FeedbackCampaign:
        """Launch targeted feedback collection campaign"""
        
        campaign = FeedbackCampaign(
            campaign_id=f"fb_campaign_{datetime.datetime.now().strftime('%Y%m%d')}",
            name=name,
            description=description,
            target_customer_segments=target_segments,
            feedback_types=feedback_types,
            start_date=datetime.date.today(),
            end_date=datetime.date.today() + datetime.timedelta(days=duration_days)
        )
        
        self.campaigns[campaign.campaign_id] = campaign
        
        self.logger.info(f"Launched feedback campaign: {name}")
        return campaign
    
    def generate_feedback_insight(self, insight_type: str, title: str, description: str,
                                supporting_feedback: List[str]) -> FeedbackInsight:
        """Generate actionable insight from feedback analysis"""
        
        confidence_level = min(len(supporting_feedback) * 0.15, 1.0)  # More feedback = higher confidence
        business_impact = self._determine_insight_business_impact(supporting_feedback)
        
        insight = FeedbackInsight(
            insight_id=f"insight_{insight_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            insight_type=insight_type,
            title=title,
            description=description,
            supporting_feedback_count=len(supporting_feedback),
            confidence_level=confidence_level,
            business_impact=business_impact,
            recommended_actions=self._generate_recommended_actions(insight_type, supporting_feedback),
            valid_until=datetime.date.today() + datetime.timedelta(days=90)
        )
        
        self.insights[insight.insight_id] = insight
        
        self.logger.info(f"Generated insight: {title}")
        return insight
    
    def _determine_insight_business_impact(self, feedback_ids: List[str]) -> ProductImpact:
        """Determine business impact level for an insight"""
        feedback_items = [self.feedback[fb_id] for fb_id in feedback_ids if fb_id in self.feedback]
        
        high_impact_count = sum(1 for fb in feedback_items if fb.business_impact in [ProductImpact.HIGH, ProductImpact.CRITICAL])
        
        if high_impact_count >= 3:
            return ProductImpact.CRITICAL
        elif high_impact_count >= 2:
            return ProductImpact.HIGH
        elif high_impact_count >= 1:
            return ProductImpact.MEDIUM
        else:
            return ProductImpact.LOW
    
    def _generate_recommended_actions(self, insight_type: str, feedback_ids: List[str]) -> List[str]:
        """Generate recommended actions for an insight"""
        actions = []
        
        if insight_type == "trend":
            actions.extend([
                "Monitor trend closely for next 30 days",
                "Prepare应对策略",
                "Consider proactive customer outreach"
            ])
        elif insight_type == "issue":
            actions.extend([
                "Immediate investigation required",
                "Create bug fix roadmap",
                "Communicate with affected customers"
            ])
        elif insight_type == "opportunity":
            actions.extend([
                "Evaluate market opportunity size",
                "Assess competitive landscape",
                "Plan product development roadmap"
            ])
        
        return actions
    
    def get_product_improvement_dashboard(self) -> Dict:
        """Generate comprehensive product improvement dashboard"""
        
        # Calculate overall metrics
        total_feedback = len(self.feedback)
        open_feedback = len([fb for fb in self.feedback.values() if fb.status not in [FeedbackStatus.CLOSED, FeedbackStatus.IMPLEMENTED]])
        enhancement_requests = len(self.enhancement_requests)
        
        # Feedback trends (last 30 days)
        trend_analysis = self.analyze_feedback_trends(30)
        
        # Top enhancement requests by customer impact
        top_requests = sorted(
            self.enhancement_requests.values(),
            key=lambda x: len(x.feedback_ids),
            reverse=True
        )[:10]
        
        # Recent insights
        recent_insights = sorted(
            self.insights.values(),
            key=lambda x: x.created_date,
            reverse=True
        )[:5]
        
        return {
            "overview": {
                "total_feedback": total_feedback,
                "open_feedback": open_feedback,
                "enhancement_requests": enhancement_requests,
                "active_campaigns": len([c for c in self.campaigns.values() if c.status == "active"])
            },
            "feedback_trends": trend_analysis,
            "top_enhancement_requests": [
                {
                    "request_id": req.request_id,
                    "title": req.title,
                    "feedback_count": len(req.feedback_ids),
                    "development_effort": req.development_effort,
                    "status": req.status
                }
                for req in top_requests
            ],
            "recent_insights": [
                {
                    "insight_id": ins.insight_id,
                    "title": ins.title,
                    "insight_type": ins.insight_type,
                    "confidence_level": ins.confidence_level,
                    "business_impact": ins.business_impact.value
                }
                for ins in recent_insights
            ],
            "campaign_performance": self._get_campaign_performance_summary(),
            "improvement_metrics": {
                "feedback_to_enhancement_ratio": enhancement_requests / total_feedback if total_feedback > 0 else 0,
                "avg_feedback_resolution_time": self._calculate_avg_resolution_time(
                    [fb for fb in self.feedback.values() if fb.status == FeedbackStatus.CLOSED]
                ),
                "customer_satisfaction_trend": self._calculate_satisfaction_trend()
            }
        }
    
    def _get_campaign_performance_summary(self) -> Dict:
        """Get campaign performance summary"""
        total_campaigns = len(self.campaigns)
        active_campaigns = len([c for c in self.campaigns.values() if c.status == "active"])
        
        if total_campaigns == 0:
            return {"message": "No campaigns launched"}
        
        total_responses = sum(c.current_response_count for c in self.campaigns.values())
        avg_response_rate = sum(c.response_rate for c in self.campaigns.values()) / total_campaigns
        
        return {
            "total_campaigns": total_campaigns,
            "active_campaigns": active_campaigns,
            "total_responses": total_responses,
            "average_response_rate": avg_response_rate
        }
    
    def _calculate_satisfaction_trend(self) -> float:
        """Calculate customer satisfaction trend from feedback"""
        recent_feedback = [
            fb for fb in self.feedback.values()
            if fb.submitted_date >= datetime.datetime.now() - datetime.timedelta(days=90)
        ]
        
        if not recent_feedback:
            return 0.0
        
        return sum(fb.customer_satisfaction_impact for fb in recent_feedback) / len(recent_feedback)