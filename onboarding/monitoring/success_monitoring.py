"""
Customer Success Monitoring and Health Scoring System
Real-time monitoring with predictive success indicators
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

class CustomerHealthStatus(Enum):
    """Customer health status levels"""
    CRITICAL = "critical"
    AT_RISK = "at_risk"
    NEEDS_ATTENTION = "needs_attention"
    HEALTHY = "healthy"
    THRIVING = "thriving"

class MetricCategory(Enum):
    """Categories of success metrics"""
    ADOPTION = "adoption"
    ENGAGEMENT = "engagement"
    PERFORMANCE = "performance"
    SUPPORT = "support"
    BUSINESS_VALUE = "business_value"
    CLINICAL_OUTCOMES = "clinical_outcomes"

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class HealthMetric:
    """Individual health metric"""
    metric_id: str
    category: MetricCategory
    name: str
    current_value: float
    target_value: float
    threshold_warning: float
    threshold_critical: float
    weight: float  # Importance weight in overall health score
    measurement_frequency: str
    trend_direction: str  # "improving", "stable", "declining"
    last_updated: str

@dataclass
class CustomerSuccessProfile:
    """Customer success profile"""
    organization_id: str
    organization_name: str
    customer_health_status: CustomerHealthStatus
    health_score: float
    metrics: List[HealthMetric]
    risk_factors: List[str]
    success_indicators: List[str]
    last_health_check: str
    next_review_date: str
    account_manager: str
    success_team_assignments: List[str]

class CustomerSuccessMonitor:
    """Customer success monitoring and health scoring system"""
    
    def __init__(self):
        self.metric_templates = self._initialize_metric_templates()
        self.health_scoring_weights = self._initialize_health_scoring_weights()
        self.alert_rules = self._initialize_alert_rules()
        self.success_benchmarks = self._initialize_success_benchmarks()
    
    def _initialize_metric_templates(self) -> Dict[str, HealthMetric]:
        """Initialize health metric templates for healthcare organizations"""
        return {
            # Adoption Metrics
            "DAILY_ACTIVE_USERS": HealthMetric(
                metric_id="DAILY_ACTIVE_USERS",
                category=MetricCategory.ADOPTION,
                name="Daily Active Users",
                current_value=0.0,
                target_value=80.0,  # 80% of licensed users
                threshold_warning=60.0,
                threshold_critical=40.0,
                weight=0.15,
                measurement_frequency="daily",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "WEEKLY_ACTIVE_USERS": HealthMetric(
                metric_id="WEEKLY_ACTIVE_USERS",
                category=MetricCategory.ADOPTION,
                name="Weekly Active Users",
                current_value=0.0,
                target_value=90.0,  # 90% of licensed users
                threshold_warning=75.0,
                threshold_critical=50.0,
                weight=0.15,
                measurement_frequency="weekly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "FEATURE_ADOPTION_RATE": HealthMetric(
                metric_id="FEATURE_ADOPTION_RATE",
                category=MetricCategory.ADOPTION,
                name="Feature Adoption Rate",
                current_value=0.0,
                target_value=70.0,  # 70% using core features
                threshold_warning=50.0,
                threshold_critical=30.0,
                weight=0.10,
                measurement_frequency="weekly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            
            # Engagement Metrics
            "SESSION_DURATION": HealthMetric(
                metric_id="SESSION_DURATION",
                category=MetricCategory.ENGAGEMENT,
                name="Average Session Duration",
                current_value=0.0,
                target_value=15.0,  # 15 minutes average
                threshold_warning=10.0,
                threshold_critical=5.0,
                weight=0.08,
                measurement_frequency="daily",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "CLINICAL_DECISIONS_SUPPORTED": HealthMetric(
                metric_id="CLINICAL_DECISIONS_SUPPORTED",
                category=MetricCategory.ENGAGEMENT,
                name="Clinical Decisions Supported",
                current_value=0.0,
                target_value=50.0,  # 50 decisions per day per 100 users
                threshold_warning=30.0,
                threshold_critical=20.0,
                weight=0.12,
                measurement_frequency="daily",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "USER_SATISFACTION_SCORE": HealthMetric(
                metric_id="USER_SATISFACTION_SCORE",
                category=MetricCategory.ENGAGEMENT,
                name="User Satisfaction Score",
                current_value=0.0,
                target_value=4.5,  # Out of 5.0
                threshold_warning=3.5,
                threshold_critical=2.5,
                weight=0.15,
                measurement_frequency="monthly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            
            # Performance Metrics
            "SYSTEM_UPTIME": HealthMetric(
                metric_id="SYSTEM_UPTIME",
                category=MetricCategory.PERFORMANCE,
                name="System Uptime",
                current_value=0.0,
                target_value=99.5,  # 99.5% uptime
                threshold_warning=99.0,
                threshold_critical=98.0,
                weight=0.10,
                measurement_frequency="daily",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "RESPONSE_TIME": HealthMetric(
                metric_id="RESPONSE_TIME",
                category=MetricCategory.PERFORMANCE,
                name="Average Response Time",
                current_value=0.0,
                target_value=2.0,  # 2 seconds max
                threshold_warning=3.0,
                threshold_critical=5.0,
                weight=0.08,
                measurement_frequency="hourly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "CLINICAL_ACCURACY": HealthMetric(
                metric_id="CLINICAL_ACCURACY",
                category=MetricCategory.PERFORMANCE,
                name="Clinical Accuracy Rate",
                current_value=0.0,
                target_value=95.0,  # 95% accuracy
                threshold_warning=90.0,
                threshold_critical=85.0,
                weight=0.15,
                measurement_frequency="weekly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            
            # Support Metrics
            "SUPPORT_TICKET_VOLUME": HealthMetric(
                metric_id="SUPPORT_TICKET_VOLUME",
                category=MetricCategory.SUPPORT,
                name="Support Tickets per User per Month",
                current_value=0.0,
                target_value=2.0,  # Less than 2 tickets per user
                threshold_warning=5.0,
                threshold_critical=10.0,
                weight=-0.05,  # Negative weight (higher is worse)
                measurement_frequency="monthly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "FIRST_RESPONSE_TIME": HealthMetric(
                metric_id="FIRST_RESPONSE_TIME",
                category=MetricCategory.SUPPORT,
                name="First Response Time (Hours)",
                current_value=0.0,
                target_value=2.0,  # 2 hours max
                threshold_warning=4.0,
                threshold_critical=8.0,
                weight=0.06,
                measurement_frequency="daily",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "RESOLUTION_TIME": HealthMetric(
                metric_id="RESOLUTION_TIME",
                category=MetricCategory.SUPPORT,
                name="Ticket Resolution Time (Hours)",
                current_value=0.0,
                target_value=24.0,  # 24 hours max
                threshold_warning=48.0,
                threshold_critical=72.0,
                weight=0.06,
                measurement_frequency="daily",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            
            # Business Value Metrics
            "ROI_ACHIEVEMENT": HealthMetric(
                metric_id="ROI_ACHIEVEMENT",
                category=MetricCategory.BUSINESS_VALUE,
                name="ROI Achievement",
                current_value=0.0,
                target_value=150.0,  # 150% ROI target
                threshold_warning=100.0,
                threshold_critical=75.0,
                weight=0.20,
                measurement_frequency="quarterly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "TIME_SAVINGS": HealthMetric(
                metric_id="TIME_SAVINGS",
                category=MetricCategory.BUSINESS_VALUE,
                name="Clinical Time Savings (Hours/Week)",
                current_value=0.0,
                target_value=10.0,  # 10 hours saved per clinician per week
                threshold_warning=5.0,
                threshold_critical=2.0,
                weight=0.12,
                measurement_frequency="monthly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            
            # Clinical Outcomes Metrics
            "CLINICAL_EFFICIENCY": HealthMetric(
                metric_id="CLINICAL_EFFICIENCY",
                category=MetricCategory.CLINICAL_OUTCOMES,
                name="Clinical Efficiency Improvement",
                current_value=0.0,
                target_value=25.0,  # 25% improvement
                threshold_warning=15.0,
                threshold_critical=5.0,
                weight=0.18,
                measurement_frequency="monthly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            ),
            "PATIENT_OUTCOMES": HealthMetric(
                metric_id="PATIENT_OUTCOMES",
                category=MetricCategory.CLINICAL_OUTCOMES,
                name="Patient Outcome Improvement",
                current_value=0.0,
                target_value=15.0,  # 15% improvement in outcomes
                threshold_warning=10.0,
                threshold_critical=5.0,
                weight=0.20,
                measurement_frequency="quarterly",
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            )
        }
    
    def _initialize_health_scoring_weights(self) -> Dict[MetricCategory, float]:
        """Initialize health scoring weights by category"""
        return {
            MetricCategory.ADOPTION: 0.30,      # 30% - High importance for adoption
            MetricCategory.ENGAGEMENT: 0.25,    # 25% - Critical for long-term success
            MetricCategory.PERFORMANCE: 0.20,   # 20% - Essential for trust
            MetricCategory.SUPPORT: 0.10,       # 10% - Important for satisfaction
            MetricCategory.BUSINESS_VALUE: 0.10, # 10% - Key for renewal
            MetricCategory.CLINICAL_OUTCOMES: 0.05 # 5% - Important but harder to measure
        }
    
    def _initialize_alert_rules(self) -> List[Dict[str, Any]]:
        """Initialize alert rules for customer health monitoring"""
        return [
            {
                "rule_id": "ALERT_001",
                "name": "Low User Adoption",
                "condition": "DAILY_ACTIVE_USERS < 50.0",
                "severity": AlertSeverity.HIGH,
                "action": "increase_training",
                "description": "User adoption is below acceptable threshold"
            },
            {
                "rule_id": "ALERT_002",
                "name": "Clinical Accuracy Concern",
                "condition": "CLINICAL_ACCURACY < 90.0",
                "severity": AlertSeverity.CRITICAL,
                "action": "immediate_review",
                "description": "Clinical accuracy has dropped below safety threshold"
            },
            {
                "rule_id": "ALERT_003",
                "name": "High Support Volume",
                "condition": "SUPPORT_TICKET_VOLUME > 5.0",
                "severity": AlertSeverity.MEDIUM,
                "action": "proactive_support",
                "description": "Support ticket volume indicates potential issues"
            },
            {
                "rule_id": "ALERT_004",
                "name": "System Performance Issues",
                "condition": "SYSTEM_UPTIME < 99.0",
                "severity": AlertSeverity.HIGH,
                "action": "technical_review",
                "description": "System uptime below acceptable level"
            },
            {
                "rule_id": "ALERT_005",
                "name": "Declining User Satisfaction",
                "condition": "USER_SATISFACTION_SCORE < 3.5",
                "severity": AlertSeverity.MEDIUM,
                "action": "customer_success_outreach",
                "description": "User satisfaction score indicates potential churn risk"
            }
        ]
    
    def _initialize_success_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize success benchmarks for different organization types"""
        return {
            "hospital": {
                "adoption_benchmarks": {
                    "daily_active_users": {"target": 75, "top_quartile": 85},
                    "feature_adoption": {"target": 60, "top_quartile": 80}
                },
                "clinical_benchmarks": {
                    "accuracy": {"target": 95, "top_quartile": 98},
                    "efficiency": {"target": 20, "top_quartile": 35}
                },
                "support_benchmarks": {
                    "ticket_volume": {"target": 3, "top_quartile": 1},
                    "resolution_time": {"target": 24, "top_quartile": 12}
                }
            },
            "clinic": {
                "adoption_benchmarks": {
                    "daily_active_users": {"target": 80, "top_quartile": 90},
                    "feature_adoption": {"target": 70, "top_quartile": 85}
                },
                "clinical_benchmarks": {
                    "accuracy": {"target": 92, "top_quartile": 96},
                    "efficiency": {"target": 25, "top_quartile": 40}
                },
                "support_benchmarks": {
                    "ticket_volume": {"target": 2, "top_quartile": 0.5},
                    "resolution_time": {"target": 18, "top_quartile": 8}
                }
            },
            "health_system": {
                "adoption_benchmarks": {
                    "daily_active_users": {"target": 70, "top_quartile": 80},
                    "feature_adoption": {"target": 65, "top_quartile": 80}
                },
                "clinical_benchmarks": {
                    "accuracy": {"target": 94, "top_quartile": 97},
                    "efficiency": {"target": 22, "top_quartile": 32}
                },
                "support_benchmarks": {
                    "ticket_volume": {"target": 4, "top_quartile": 2},
                    "resolution_time": {"target": 30, "top_quartile": 18}
                }
            }
        }
    
    def create_customer_profile(self, organization_id: str, organization_name: str, 
                              provider_type: str, account_manager: str) -> CustomerSuccessProfile:
        """Create customer success profile"""
        # Initialize metrics with default values
        metrics = []
        for metric_template in self.metric_templates.values():
            metric = HealthMetric(
                metric_id=metric_template.metric_id,
                category=metric_template.category,
                name=metric_template.name,
                current_value=0.0,  # Will be populated from actual data
                target_value=metric_template.target_value,
                threshold_warning=metric_template.threshold_warning,
                threshold_critical=metric_template.threshold_critical,
                weight=metric_template.weight,
                measurement_frequency=metric_template.measurement_frequency,
                trend_direction="stable",
                last_updated=datetime.now().isoformat()
            )
            metrics.append(metric)
        
        profile = CustomerSuccessProfile(
            organization_id=organization_id,
            organization_name=organization_name,
            customer_health_status=CustomerHealthStatus.HEALTHY,  # Start neutral
            health_score=0.0,
            metrics=metrics,
            risk_factors=[],
            success_indicators=[],
            last_health_check=datetime.now().isoformat(),
            next_review_date=(datetime.now() + timedelta(days=30)).isoformat(),
            account_manager=account_manager,
            success_team_assignments=[account_manager]
        )
        
        return profile
    
    def update_metric_values(self, profile: CustomerSuccessProfile, 
                           metric_data: Dict[str, float]) -> None:
        """Update metric values for customer profile"""
        for metric in profile.metrics:
            if metric.metric_id in metric_data:
                old_value = metric.current_value
                metric.current_value = metric_data[metric.metric_id]
                metric.last_updated = datetime.now().isoformat()
                
                # Update trend direction
                if metric.current_value > old_value:
                    metric.trend_direction = "improving"
                elif metric.current_value < old_value:
                    metric.trend_direction = "declining"
                else:
                    metric.trend_direction = "stable"
        
        # Recalculate health score
        self.calculate_health_score(profile)
    
    def calculate_health_score(self, profile: CustomerSuccessProfile) -> float:
        """Calculate overall health score for customer"""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Group metrics by category for weighted scoring
        category_scores = {}
        category_weights = {}
        
        for metric in profile.metrics:
            category = metric.category
            
            # Calculate normalized score (0-100 scale)
            if metric.threshold_critical > 0:
                # For metrics where higher is better
                if metric.weight > 0:
                    if metric.current_value >= metric.target_value:
                        normalized_score = 100.0
                    elif metric.current_value >= metric.threshold_warning:
                        # Linear interpolation between warning and target
                        normalized_score = 50 + (metric.current_value - metric.threshold_warning) / \
                                         (metric.target_value - metric.threshold_warning) * 50
                    else:
                        # Below warning threshold
                        normalized_score = max(0, (metric.current_value / metric.threshold_warning) * 50)
                else:
                    # For negative weight metrics (like support volume)
                    if metric.current_value <= metric.threshold_warning:
                        normalized_score = 100.0
                    elif metric.current_value <= metric.threshold_critical:
                        normalized_score = 50 + (metric.threshold_critical - metric.current_value) / \
                                         (metric.threshold_critical - metric.threshold_warning) * 50
                    else:
                        normalized_score = max(0, (metric.threshold_critical / metric.current_value) * 50)
            else:
                normalized_score = 50.0  # Default for undefined thresholds
            
            # Add to category totals
            if category not in category_scores:
                category_scores[category] = 0.0
                category_weights[category] = 0.0
            
            category_scores[category] += normalized_score * abs(metric.weight)
            category_weights[category] += abs(metric.weight)
        
        # Calculate weighted category scores
        for category in MetricCategory:
            if category in category_scores and category_weights[category] > 0:
                category_score = category_scores[category] / category_weights[category]
                category_weight = self.health_scoring_weights.get(category, 0.0)
                
                total_weighted_score += category_score * category_weight
                total_weight += category_weight
        
        # Final health score
        if total_weight > 0:
            health_score = total_weighted_score / total_weight
        else:
            health_score = 50.0  # Default neutral score
        
        profile.health_score = round(health_score, 2)
        
        # Update health status based on score
        if health_score >= 85:
            profile.customer_health_status = CustomerHealthStatus.THRIVING
        elif health_score >= 70:
            profile.customer_health_status = CustomerHealthStatus.HEALTHY
        elif health_score >= 55:
            profile.customer_health_status = CustomerHealthStatus.NEEDS_ATTENTION
        elif health_score >= 40:
            profile.customer_health_status = CustomerHealthStatus.AT_RISK
        else:
            profile.customer_health_status = CustomerHealthStatus.CRITICAL
        
        return health_score
    
    def generate_health_insights(self, profile: CustomerSuccessProfile) -> Dict[str, Any]:
        """Generate health insights and recommendations"""
        insights = {
            "health_score": profile.health_score,
            "health_status": profile.customer_health_status.value,
            "key_strengths": [],
            "improvement_areas": [],
            "risk_factors": [],
            "success_indicators": [],
            "recommendations": [],
            "alert_triggers": []
        }
        
        # Analyze metrics by category
        category_analysis = {}
        for metric in profile.metrics:
            category = metric.category
            if category not in category_analysis:
                category_analysis[category] = {"metrics": [], "score": 0}
            
            category_analysis[category]["metrics"].append(metric)
            
            # Determine if this metric is a strength or improvement area
            if metric.current_value >= metric.target_value:
                insights["key_strengths"].append(metric.name)
            elif metric.current_value < metric.threshold_warning:
                insights["improvement_areas"].append(metric.name)
                insights["risk_factors"].append(metric.name)
            elif metric.current_value < metric.target_value:
                insights["improvement_areas"].append(metric.name)
        
        # Generate specific recommendations
        insights["recommendations"] = self._generate_recommendations(profile, insights)
        
        # Generate alert triggers
        insights["alert_triggers"] = self._generate_alert_triggers(profile)
        
        return insights
    
    def _generate_recommendations(self, profile: CustomerSuccessProfile, 
                                insights: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations based on health analysis"""
        recommendations = []
        
        # Adoption recommendations
        adoption_metrics = [m for m in profile.metrics if m.category == MetricCategory.ADOPTION]
        avg_adoption = statistics.mean([m.current_value for m in adoption_metrics]) if adoption_metrics else 0
        
        if avg_adoption < 60:
            recommendations.append("Increase user training and adoption initiatives")
            recommendations.append("Implement gamification for user engagement")
        elif avg_adoption < 80:
            recommendations.append("Enhance advanced feature training")
        
        # Performance recommendations
        performance_metrics = [m for m in profile.metrics if m.category == MetricCategory.PERFORMANCE]
        avg_performance = statistics.mean([m.current_value for m in performance_metrics]) if performance_metrics else 0
        
        if avg_performance < 90:
            recommendations.append("Review system performance and optimize")
            recommendations.append("Conduct technical infrastructure assessment")
        
        # Support recommendations
        support_metrics = [m for m in profile.metrics if m.category == MetricCategory.SUPPORT]
        high_volume_metric = next((m for m in support_metrics if m.metric_id == "SUPPORT_TICKET_VOLUME"), None)
        
        if high_volume_metric and high_volume_metric.current_value > 5:
            recommendations.append("Increase proactive support and documentation")
            recommendations.append("Provide additional user training on common issues")
        
        # Clinical outcome recommendations
        clinical_metrics = [m for m in profile.metrics if m.category == MetricCategory.CLINICAL_OUTCOMES]
        avg_clinical = statistics.mean([m.current_value for m in clinical_metrics]) if clinical_metrics else 0
        
        if avg_clinical < 15:
            recommendations.append("Focus on clinical workflow optimization")
            recommendations.append("Provide additional clinical training and support")
        
        return recommendations
    
    def _generate_alert_triggers(self, profile: CustomerSuccessProfile) -> List[Dict[str, Any]]:
        """Generate alert triggers based on current metrics"""
        alerts = []
        
        for rule in self.alert_rules:
            # Check if rule condition is met (simplified evaluation)
            condition_met = self._evaluate_alert_condition(rule, profile)
            
            if condition_met:
                alerts.append({
                    "alert_id": rule["rule_id"],
                    "severity": rule["severity"].value,
                    "title": rule["name"],
                    "description": rule["description"],
                    "triggered_metric": self._get_triggered_metric(rule["condition"], profile),
                    "recommendation": self._get_recommendation_for_action(rule["action"]),
                    "created_at": datetime.now().isoformat()
                })
        
        return alerts
    
    def _evaluate_alert_condition(self, rule: Dict[str, Any], profile: CustomerSuccessProfile) -> bool:
        """Evaluate if alert rule condition is met"""
        # Simplified condition evaluation - in production, this would be more robust
        condition = rule["condition"]
        
        for metric in profile.metrics:
            if metric.metric_id in condition:
                if "<" in condition:
                    threshold = float(condition.split("<")[1])
                    return metric.current_value < threshold
                elif ">" in condition:
                    threshold = float(condition.split(">")[1])
                    return metric.current_value > threshold
        
        return False
    
    def _get_triggered_metric(self, condition: str, profile: CustomerSuccessProfile) -> str:
        """Get the metric that triggered an alert"""
        for metric in profile.metrics:
            if metric.metric_id in condition:
                return metric.name
        return "Unknown Metric"
    
    def _get_recommendation_for_action(self, action: str) -> str:
        """Get recommendation based on action type"""
        action_recommendations = {
            "increase_training": "Schedule additional training sessions for users",
            "immediate_review": "Escalate to clinical team for immediate review",
            "proactive_support": "Increase proactive support outreach to customers",
            "technical_review": "Conduct technical review with engineering team",
            "customer_success_outreach": "Schedule customer success check-in call"
        }
        
        return action_recommendations.get(action, "Follow standard customer success protocols")
    
    def generate_success_dashboard(self, profiles: List[CustomerSuccessProfile]) -> Dict[str, Any]:
        """Generate success monitoring dashboard"""
        dashboard = {
            "overall_health_summary": {},
            "health_distribution": {},
            "top_performers": [],
            "at_risk_customers": [],
            "trend_analysis": {},
            "category_performance": {},
            "alert_summary": {},
            "recommendations_summary": []
        }
        
        if not profiles:
            return dashboard
        
        # Overall health summary
        health_scores = [p.health_score for p in profiles]
        dashboard["overall_health_summary"] = {
            "average_health_score": round(statistics.mean(health_scores), 2),
            "median_health_score": round(statistics.median(health_scores), 2),
            "total_customers": len(profiles),
            "score_range": {
                "min": min(health_scores),
                "max": max(health_scores)
            }
        }
        
        # Health distribution
        status_counts = {}
        for status in CustomerHealthStatus:
            status_counts[status.value] = 0
        
        for profile in profiles:
            status_counts[profile.customer_health_status.value] += 1
        
        dashboard["health_distribution"] = status_counts
        
        # Top performers and at-risk customers
        sorted_profiles = sorted(profiles, key=lambda p: p.health_score, reverse=True)
        dashboard["top_performers"] = [
            {
                "organization_name": p.organization_name,
                "health_score": p.health_score,
                "health_status": p.customer_health_status.value
            }
            for p in sorted_profiles[:5]
        ]
        
        dashboard["at_risk_customers"] = [
            {
                "organization_name": p.organization_name,
                "health_score": p.health_score,
                "health_status": p.customer_health_status.value
            }
            for p in sorted_profiles if p.customer_health_status in [CustomerHealthStatus.AT_RISK, CustomerHealthStatus.CRITICAL]
        ]
        
        return dashboard
    
    def export_health_report(self, profile: CustomerSuccessProfile, output_path: str) -> None:
        """Export customer health report"""
        health_insights = self.generate_health_insights(profile)
        
        report = {
            "customer_profile": {
                "organization_id": profile.organization_id,
                "organization_name": profile.organization_name,
                "health_status": profile.customer_health_status.value,
                "health_score": profile.health_score
            },
            "metrics": [asdict(metric) for metric in profile.metrics],
            "health_insights": health_insights,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)