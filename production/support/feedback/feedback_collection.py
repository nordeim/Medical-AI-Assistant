"""
User Feedback Collection and Analysis System
Healthcare-focused feedback with sentiment analysis and medical context awareness
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, Counter

from config.support_config import SupportConfig

class FeedbackType(Enum):
    POST_INTERACTION = "post_interaction"
    PERIODIC_SATISFACTION = "periodic_satisfaction"
    MEDICAL_OUTCOME = "medical_outcome"
    CLINICAL_WORKFLOW = "clinical_workflow"
    SYSTEM_PERFORMANCE = "system_performance"
    TRAINING_FEEDBACK = "training_feedback"
    INCIDENT_REVIEW = "incident_review"
    GENERAL_FEEDBACK = "general_feedback"

class SentimentLabel(Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"

class MedicalContextType(Enum):
    PATIENT_CARE = "patient_care"
    CLINICAL_DECISION = "clinical_decision"
    WORKFLOW_EFFICIENCY = "workflow_efficiency"
    SYSTEM_INTEGRATION = "system_integration"
    SAFETY_PROTOCOL = "safety_protocol"
    REGULATORY_COMPLIANCE = "regulatory_compliance"
    EMERGENCY_SITUATION = "emergency_situation"

@dataclass
class FeedbackResponse:
    """Individual feedback response"""
    id: str
    feedback_type: FeedbackType
    user_id: str
    user_name: str
    user_facility: str
    user_role: str
    content: str
    rating: Optional[int]  # 1-5 scale
    submitted_at: datetime
    medical_context: Optional[str]
    patient_safety_mentioned: bool
    emergency_situation: bool
    system_performance_issues: bool
    clinical_outcome_impact: str
    tags: List[str]
    metadata: Dict[str, Any]

@dataclass
class SentimentAnalysisResult:
    """Results of sentiment analysis"""
    overall_sentiment: SentimentLabel
    confidence_score: float
    medical_sentiment_context: str
    urgency_level: str
    patient_safety_concern: bool
    clinical_impact_assessment: str
    emotional_indicators: Dict[str, float]
    medical_keyword_mentions: List[str]
    action_required: bool

@dataclass
class FeedbackTrend:
    """Trend analysis for feedback"""
    period_start: datetime
    period_end: datetime
    total_responses: int
    average_rating: float
    sentiment_distribution: Dict[SentimentLabel, int]
    common_themes: List[str]
    urgent_issues: List[str]
    medical_safety_alerts: List[str]
    improvement_recommendations: List[str]

class HealthcareSentimentAnalyzer:
    """Medical-aware sentiment analysis for healthcare feedback"""
    
    def __init__(self):
        # Medical-specific positive keywords
        self.medical_positive_keywords = [
            "improved patient care", "enhanced workflow", "better outcomes",
            "faster diagnosis", "accurate results", "streamlined process",
            "intuitive interface", "reliable system", "time-saving",
            "user-friendly", "helpful guidance", "supportive features"
        ]
        
        # Medical-specific negative keywords
        self.medical_negative_keywords = [
            "patient safety risk", "delayed treatment", "system failure",
            "data inconsistency", "workflow disruption", "critical error",
            "safety concern", "medical error", "clinical risk",
            "patient harm", "emergency situation", "urgent issue"
        ]
        
        # Emergency indicators
        self.emergency_indicators = [
            "emergency", "urgent", "critical", "immediate",
            "life-threatening", "cardiac", "respiratory", "sepsis",
            "stroke", "trauma", "code blue", "rapid response"
        ]
        
        # Patient safety keywords
        self.patient_safety_keywords = [
            "patient safety", "medical error", "clinical decision",
            "treatment delay", "diagnosis accuracy", "medication error",
            "adverse event", "near miss", "safety protocol", "risk assessment"
        ]
        
        # Emotion indicators in medical context
        self.emotion_indicators = {
            "frustration": ["frustrated", "annoying", "difficult", "complicated"],
            "urgency": ["urgent", "immediate", "asap", "critical", "emergency"],
            "concern": ["concerned", "worried", "afraid", "anxious"],
            "satisfaction": ["satisfied", "pleased", "happy", "good", "excellent"],
            "dissatisfaction": ["dissatisfied", "disappointed", "poor", "bad", "terrible"]
        }
    
    def analyze_sentiment(self, feedback: FeedbackResponse) -> SentimentAnalysisResult:
        """Analyze sentiment with medical context awareness"""
        
        content_lower = feedback.content.lower()
        words = content_lower.split()
        
        # Basic sentiment analysis
        positive_score = sum(1 for word in words if word in self.medical_positive_keywords)
        negative_score = sum(1 for word in words if word in self.medical_negative_keywords)
        
        # Medical context analysis
        medical_context = self._identify_medical_context(content_lower)
        urgency_level = self._assess_urgency_level(content_lower)
        patient_safety_concern = self._check_patient_safety_concern(content_lower)
        
        # Emergency detection
        emergency_detected = any(indicator in content_lower for indicator in self.emergency_indicators)
        
        # Emotional indicators
        emotional_indicators = self._analyze_emotions(content_lower)
        
        # Medical keyword mentions
        medical_keywords = self._extract_medical_keywords(content_lower)
        
        # Overall sentiment calculation
        sentiment_score = positive_score - negative_score
        
        if sentiment_score >= 3:
            overall_sentiment = SentimentLabel.VERY_POSITIVE
        elif sentiment_score >= 1:
            overall_sentiment = SentimentLabel.POSITIVE
        elif sentiment_score <= -3:
            overall_sentiment = SentimentLabel.VERY_NEGATIVE
        elif sentiment_score <= -1:
            overall_sentiment = SentimentLabel.NEGATIVE
        else:
            overall_sentiment = SentimentLabel.NEUTRAL
        
        # Confidence score based on keyword density
        total_keywords = positive_score + negative_score
        confidence_score = min(total_keywords / len(words), 1.0) if words else 0.0
        
        # Clinical impact assessment
        clinical_impact = self._assess_clinical_impact(content_lower, patient_safety_concern, emergency_detected)
        
        # Action required determination
        action_required = (patient_safety_concern or 
                          emergency_detected or 
                          urgency_level in ["high", "critical"] or
                          overall_sentiment in [SentimentLabel.VERY_NEGATIVE])
        
        return SentimentAnalysisResult(
            overall_sentiment=overall_sentiment,
            confidence_score=confidence_score,
            medical_sentiment_context=medical_context,
            urgency_level=urgency_level,
            patient_safety_concern=patient_safety_concern,
            clinical_impact_assessment=clinical_impact,
            emotional_indicators=emotional_indicators,
            medical_keyword_mentions=medical_keywords,
            action_required=action_required
        )
    
    def _identify_medical_context(self, content: str) -> str:
        """Identify the medical context of the feedback"""
        contexts = {
            "patient care": ["patient", "care", "treatment", "diagnosis", "therapy"],
            "workflow efficiency": ["workflow", "process", "efficiency", "time", "speed"],
            "system usability": ["interface", "user", "easy", "difficult", "navigation"],
            "clinical integration": ["integration", "ehr", "system", "data", "chart"],
            "safety protocols": ["safety", "protocol", "guidelines", "compliance"]
        }
        
        for context, keywords in contexts.items():
            if any(keyword in content for keyword in keywords):
                return context
        
        return "general"
    
    def _assess_urgency_level(self, content: str) -> str:
        """Assess urgency level based on content"""
        urgency_score = 0
        
        # High urgency indicators
        high_urgency = ["emergency", "urgent", "immediate", "critical", "life-threatening"]
        if any(word in content for word in high_urgency):
            urgency_score += 3
        
        # Medium urgency indicators
        medium_urgency = ["asap", "important", "soon", "priority", "concerning"]
        if any(word in content for word in medium_urgency):
            urgency_score += 2
        
        # Low urgency indicators
        low_urgency = ["when possible", "convenient", "sometime", "eventually"]
        if any(word in content for word in low_urgency):
            urgency_score -= 1
        
        if urgency_score >= 3:
            return "critical"
        elif urgency_score >= 2:
            return "high"
        elif urgency_score >= 1:
            return "medium"
        else:
            return "low"
    
    def _check_patient_safety_concern(self, content: str) -> bool:
        """Check if feedback mentions patient safety concerns"""
        return any(keyword in content for keyword in self.patient_safety_keywords)
    
    def _analyze_emotions(self, content: str) -> Dict[str, float]:
        """Analyze emotional indicators in the content"""
        emotions = {}
        words = content.split()
        
        for emotion, keywords in self.emotion_indicators.items():
            count = sum(1 for word in words if word in keywords)
            emotions[emotion] = min(count / len(words) * 10, 1.0) if words else 0.0
        
        return emotions
    
    def _extract_medical_keywords(self, content: str) -> List[str]:
        """Extract relevant medical keywords from content"""
        medical_terms = [
            "patient", "diagnosis", "treatment", "therapy", "medication",
            "symptom", "condition", "disease", "procedure", "surgery",
            "emergency", "critical", "stable", "monitoring", "assessment"
        ]
        
        return [word for word in medical_terms if word in content]
    
    def _assess_clinical_impact(self, content: str, patient_safety: bool, emergency: bool) -> str:
        """Assess potential clinical impact"""
        if patient_safety or emergency:
            return "high_impact"
        
        impact_keywords = {
            "high": ["significant", "major", "substantial", "important"],
            "medium": ["moderate", "some", "noticeable", "apparent"],
            "low": ["minor", "slight", "minimal", "small"]
        }
        
        for impact, keywords in impact_keywords.items():
            if any(keyword in content for keyword in keywords):
                return f"{impact}_impact"
        
        return "unknown_impact"

class FeedbackCollectionSystem:
    """Main feedback collection and analysis system"""
    
    def __init__(self):
        self.feedback_responses: Dict[str, FeedbackResponse] = {}
        self.sentiment_analyzer = HealthcareSentimentAnalyzer()
        self.feedback_counter = 0
        self.auto_collection_enabled = True
        self.medical_alert_threshold = 0.8
    
    async def collect_feedback(
        self,
        feedback_type: FeedbackType,
        user_id: str,
        user_name: str,
        user_facility: str,
        user_role: str,
        content: str,
        rating: Optional[int] = None,
        medical_context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeedbackResponse:
        """Collect new feedback response"""
        
        self.feedback_counter += 1
        feedback_id = f"FB-{datetime.now().strftime('%Y%m%d')}-{self.feedback_counter:04d}"
        
        # Analyze content for medical indicators
        patient_safety_mentioned = any(keyword in content.lower() for keyword in 
                                     self.sentiment_analyzer.patient_safety_keywords)
        emergency_situation = any(indicator in content.lower() for indicator in
                                self.sentiment_analyzer.emergency_indicators)
        system_performance_issues = any(issue in content.lower() for issue in
                                      ["slow", "error", "crash", "bug", "down", "unavailable"])
        
        feedback = FeedbackResponse(
            id=feedback_id,
            feedback_type=feedback_type,
            user_id=user_id,
            user_name=user_name,
            user_facility=user_facility,
            user_role=user_role,
            content=content,
            rating=rating,
            submitted_at=datetime.now(),
            medical_context=medical_context,
            patient_safety_mentioned=patient_safety_mentioned,
            emergency_situation=emergency_situation,
            system_performance_issues=system_performance_issues,
            clinical_outcome_impact=self._assess_clinical_outcome_impact(content),
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self.feedback_responses[feedback_id] = feedback
        
        # Trigger real-time analysis
        sentiment_result = await self.analyze_feedback_sentiment(feedback)
        
        # Send alerts for critical feedback
        if sentiment_result.action_required:
            await self._trigger_critical_feedback_alert(feedback, sentiment_result)
        
        logger.info(f"Collected feedback {feedback_id} with sentiment: {sentiment_result.overall_sentiment.value}")
        return feedback
    
    async def analyze_feedback_sentiment(self, feedback: FeedbackResponse) -> SentimentAnalysisResult:
        """Perform real-time sentiment analysis on feedback"""
        return self.sentiment_analyzer.analyze_sentiment(feedback)
    
    async def get_feedback_trends(
        self,
        start_date: datetime,
        end_date: datetime,
        facility_filter: Optional[str] = None,
        feedback_type_filter: Optional[FeedbackType] = None
    ) -> FeedbackTrend:
        """Generate comprehensive feedback trend analysis"""
        
        # Filter feedback responses
        period_responses = [
            response for response in self.feedback_responses.values()
            if start_date <= response.submitted_at <= end_date
        ]
        
        if facility_filter:
            period_responses = [r for r in period_responses if r.user_facility == facility_filter]
        
        if feedback_type_filter:
            period_responses = [r for r in period_responses if r.feedback_type == feedback_type_filter]
        
        # Calculate metrics
        total_responses = len(period_responses)
        ratings = [r.rating for r in period_responses if r.rating is not None]
        average_rating = statistics.mean(ratings) if ratings else 0
        
        # Sentiment distribution
        sentiment_distribution = defaultdict(int)
        for response in period_responses:
            # Perform sentiment analysis for trend
            sentiment = self.sentiment_analyzer.analyze_sentiment(response)
            sentiment_distribution[sentiment.overall_sentiment] += 1
        
        # Common themes analysis
        common_themes = self._identify_common_themes(period_responses)
        
        # Urgent issues identification
        urgent_issues = self._identify_urgent_issues(period_responses)
        
        # Medical safety alerts
        medical_safety_alerts = self._identify_medical_safety_alerts(period_responses)
        
        # Improvement recommendations
        improvement_recommendations = self._generate_improvement_recommendations(period_responses)
        
        return FeedbackTrend(
            period_start=start_date,
            period_end=end_date,
            total_responses=total_responses,
            average_rating=average_rating,
            sentiment_distribution=dict(sentiment_distribution),
            common_themes=common_themes,
            urgent_issues=urgent_issues,
            medical_safety_alerts=medical_safety_alerts,
            improvement_recommendations=improvement_recommendations
        )
    
    async def get_feedback_by_facility(self, facility: str, days: int = 30) -> List[FeedbackResponse]:
        """Get feedback responses for a specific facility"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            response for response in self.feedback_responses.values()
            if response.user_facility == facility and response.submitted_at >= cutoff_date
        ]
    
    async def get_critical_feedback_alerts(self, hours: int = 24) -> List[FeedbackResponse]:
        """Get critical feedback requiring immediate attention"""
        cutoff_date = datetime.now() - timedelta(hours=hours)
        
        critical_feedback = []
        for response in self.feedback_responses.values():
            if response.submitted_at >= cutoff_date:
                sentiment = self.sentiment_analyzer.analyze_sentiment(response)
                if sentiment.action_required:
                    critical_feedback.append(response)
        
        return sorted(critical_feedback, key=lambda x: x.submitted_at, reverse=True)
    
    async def generate_feedback_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive feedback analytics report"""
        
        # Get trend data
        trend = await self.get_feedback_trends(start_date, end_date)
        
        # Additional analytics
        feedback_types = Counter(r.feedback_type for r in self.feedback_responses.values()
                               if start_date <= r.submitted_at <= end_date)
        
        user_roles = Counter(r.user_role for r in self.feedback_responses.values()
                           if start_date <= r.submitted_at <= end_date)
        
        # Medical safety statistics
        safety_mentions = sum(1 for r in self.feedback_responses.values()
                            if r.patient_safety_mentioned and start_date <= r.submitted_at <= end_date)
        
        emergency_mentions = sum(1 for r in self.feedback_responses.values()
                               if r.emergency_situation and start_date <= r.submitted_at <= end_date)
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_responses": trend.total_responses,
                "average_rating": trend.average_rating,
                "safety_concerns": safety_mentions,
                "emergency_situations": emergency_mentions
            },
            "sentiment_analysis": {
                "distribution": trend.sentiment_distribution,
                "most_common_themes": trend.common_themes[:10]
            },
            "feedback_breakdown": {
                "by_type": dict(feedback_types),
                "by_user_role": dict(user_roles)
            },
            "urgent_items": {
                "critical_feedback": trend.urgent_issues,
                "safety_alerts": trend.medical_safety_alerts
            },
            "recommendations": trend.improvement_recommendations,
            "trends": {
                "period_comparison": await self._compare_periods(start_date, end_date),
                "facility_comparison": await self._compare_facilities(start_date, end_date)
            }
        }
    
    def _assess_clinical_outcome_impact(self, content: str) -> str:
        """Assess impact on clinical outcomes"""
        high_impact_keywords = ["improved", "better", "enhanced", "faster", "accurate"]
        negative_impact_keywords = ["delayed", "worse", "complicated", "difficult", "error"]
        
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in high_impact_keywords):
            return "positive_impact"
        elif any(keyword in content_lower for keyword in negative_impact_keywords):
            return "negative_impact"
        else:
            return "neutral_impact"
    
    def _identify_common_themes(self, responses: List[FeedbackResponse]) -> List[str]:
        """Identify common themes from feedback responses"""
        all_words = []
        for response in responses:
            # Extract meaningful words (exclude common stop words)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', response.content.lower())
            meaningful_words = [w for w in words if w not in self._get_stop_words()]
            all_words.extend(meaningful_words)
        
        # Get most common words
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(20)]
    
    def _identify_urgent_issues(self, responses: List[FeedbackResponse]) -> List[str]:
        """Identify urgent issues from feedback"""
        urgent_issues = []
        
        for response in responses:
            sentiment = self.sentiment_analyzer.analyze_sentiment(response)
            if sentiment.urgency_level in ["high", "critical"] or sentiment.action_required:
                urgent_issues.append({
                    "feedback_id": response.id,
                    "user": response.user_name,
                    "facility": response.user_facility,
                    "content": response.content[:200] + "..." if len(response.content) > 200 else response.content,
                    "urgency": sentiment.urgency_level,
                    "timestamp": response.submitted_at.isoformat()
                })
        
        return urgent_issues
    
    def _identify_medical_safety_alerts(self, responses: List[FeedbackResponse]) -> List[str]:
        """Identify medical safety alerts from feedback"""
        safety_alerts = []
        
        for response in responses:
            if response.patient_safety_mentioned:
                sentiment = self.sentiment_analyzer.analyze_sentiment(response)
                safety_alerts.append({
                    "feedback_id": response.id,
                    "user": response.user_name,
                    "facility": response.user_facility,
                    "concern_level": "high" if sentiment.patient_safety_concern else "medium",
                    "content": response.content,
                    "timestamp": response.submitted_at.isoformat()
                })
        
        return safety_alerts
    
    def _generate_improvement_recommendations(self, responses: List[FeedbackResponse]) -> List[str]:
        """Generate improvement recommendations based on feedback analysis"""
        recommendations = []
        
        # Analyze common complaints
        negative_themes = []
        for response in responses:
            sentiment = self.sentiment_analyzer.analyze_sentiment(response)
            if sentiment.overall_sentiment in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]:
                negative_themes.extend(sentiment.medical_keyword_mentions)
        
        # Generate recommendations based on themes
        theme_counts = Counter(negative_themes)
        for theme, count in theme_counts.most_common(5):
            if count >= 3:  # Only include themes mentioned multiple times
                recommendations.append(f"Address recurring issues with {theme} based on {count} feedback mentions")
        
        # Add generic recommendations based on sentiment distribution
        total_negative = sum(1 for r in responses if 
                           self.sentiment_analyzer.analyze_sentiment(r).overall_sentiment in 
                           [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE])
        
        if total_negative > len(responses) * 0.3:  # More than 30% negative
            recommendations.append("Consider comprehensive user training and system optimization")
        
        if total_negative > len(responses) * 0.5:  # More than 50% negative
            recommendations.append("Urgent review of system performance and user experience needed")
        
        return recommendations
    
    async def _trigger_critical_feedback_alert(self, feedback: FeedbackResponse, sentiment: SentimentAnalysisResult) -> None:
        """Trigger alerts for critical feedback"""
        alert_data = {
            "feedback_id": feedback.id,
            "user": feedback.user_name,
            "facility": feedback.user_facility,
            "urgency": sentiment.urgency_level,
            "patient_safety_concern": sentiment.patient_safety_concern,
            "content": feedback.content,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.critical(f"CRITICAL FEEDBACK ALERT: {json.dumps(alert_data)}")
        
        # In production, this would send alerts to:
        # - Email alerts to medical directors
        # - SMS to emergency contacts
        # - Slack notifications to support teams
        # - Integration with incident management system
    
    def _get_stop_words(self) -> set:
        """Get common stop words for text analysis"""
        return {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "been", "be", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his",
            "her", "its", "our", "their"
        }
    
    async def _compare_periods(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Compare feedback metrics between periods"""
        # Implementation would compare current period with previous period
        return {"improvement_rate": 0, "degradation_areas": [], "improvement_areas": []}
    
    async def _compare_facilities(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Compare feedback metrics across facilities"""
        facilities = set(r.user_facility for r in self.feedback_responses.values()
                        if start_date <= r.submitted_at <= end_date)
        
        facility_comparison = {}
        for facility in facilities:
            facility_feedback = [r for r in self.feedback_responses.values()
                               if r.user_facility == facility and start_date <= r.submitted_at <= end_date]
            
            if facility_feedback:
                avg_rating = statistics.mean([r.rating for r in facility_feedback if r.rating])
                sentiment_scores = [self.sentiment_analyzer.analyze_sentiment(r).overall_sentiment.value 
                                  for r in facility_feedback]
                facility_comparison[facility] = {
                    "average_rating": avg_rating,
                    "sentiment_breakdown": Counter(sentiment_scores),
                    "total_feedback": len(facility_feedback)
                }
        
        return facility_comparison

import logging
logger = logging.getLogger(__name__)

# Global feedback system instance
feedback_system = FeedbackCollectionSystem()

# Example usage and testing functions
async def create_sample_feedback():
    """Create sample feedback responses for testing"""
    
    # Positive feedback
    positive_feedback = await feedback_system.collect_feedback(
        feedback_type=FeedbackType.POST_INTERACTION,
        user_id="dr_johnson_001",
        user_name="Dr. Michael Johnson",
        user_facility="City Medical Center",
        user_role="Cardiologist",
        content="The new cardiac monitoring integration has significantly improved our workflow efficiency. The real-time alerts are very helpful.",
        rating=5,
        medical_context="cardiology_workflow"
    )
    
    # Critical feedback with patient safety concern
    critical_feedback = await feedback_system.collect_feedback(
        feedback_type=FeedbackType.INCIDENT_REVIEW,
        user_id="nurse_smith_002",
        user_name="Nurse Sarah Smith",
        user_facility="General Hospital",
        user_role="Registered Nurse",
        content="System delay caused a 15-minute delay in critical medication administration. This poses patient safety risks.",
        rating=1,
        medical_context="medication_safety"
    )
    
    # Emergency situation feedback
    emergency_feedback = await feedback_system.collect_feedback(
        feedback_type=FeedbackType.SYSTEM_PERFORMANCE,
        user_id="dr_wilson_003",
        user_name="Dr. Lisa Wilson",
        user_facility="Emergency Department",
        user_role="Emergency Physician",
        content="Emergency! System crashed during cardiac arrest case. Need immediate technical support.",
        medical_context="emergency_situation"
    )
    
    print(f"Created feedback responses: {positive_feedback.id}, {critical_feedback.id}, {emergency_feedback.id}")
    
    # Analyze sentiment
    for feedback in [positive_feedback, critical_feedback, emergency_feedback]:
        sentiment = await feedback_system.analyze_feedback_sentiment(feedback)
        print(f"Feedback {feedback.id}: {sentiment.overall_sentiment.value} (confidence: {sentiment.confidence_score:.2f})")
        if sentiment.action_required:
            print(f"  ⚠️  ACTION REQUIRED - {sentiment.urgency_level} urgency")

if __name__ == "__main__":
    asyncio.run(create_sample_feedback())