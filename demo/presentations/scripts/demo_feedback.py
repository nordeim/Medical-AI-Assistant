#!/usr/bin/env python3
"""
Demo Feedback Collection System - Comprehensive stakeholder feedback management.

This module provides comprehensive feedback collection capabilities including:
- Multi-channel feedback collection (surveys, interviews, real-time)
- Stakeholder-specific feedback forms
- Analytics and sentiment analysis
- Automated improvement recommendations
- Feedback integration with demo analytics
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import re

class FeedbackType(Enum):
    """Feedback type enumeration"""
    SURVEY = "survey"
    INTERVIEW = "interview"
    REAL_TIME = "real_time"
    FOLLOW_UP = "follow_up"
    FOCUS_GROUP = "focus_group"

class StakeholderFeedback(Enum):
    """Stakeholder-specific feedback forms"""
    C_SUITE = "c_suite"
    CLINICAL = "clinical"
    REGULATORY = "regulatory"
    INVESTOR = "investor"
    PARTNER = "partner"
    TECHNICAL = "technical"
    PATIENT = "patient"

class QuestionType(Enum):
    """Question type enumeration"""
    MULTIPLE_CHOICE = "multiple_choice"
    RATING_SCALE = "rating_scale"
    TEXT = "text"
    LIKERT = "likert"
    YES_NO = "yes_no"
    RANKING = "ranking"
    MATRIX = "matrix"

@dataclass
class FeedbackQuestion:
    """Individual feedback question"""
    question_id: str
    question_text: str
    question_type: QuestionType
    required: bool = True
    options: Optional[List[str]] = None
    scale_min: Optional[int] = None
    scale_max: Optional[int] = None
    scale_labels: Optional[List[str]] = None
    
@dataclass
class FeedbackResponse:
    """Feedback response structure"""
    response_id: str
    session_id: str
    stakeholder_type: str
    feedback_type: FeedbackType
    responses: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class DemoFeedbackManager:
    """Comprehensive feedback management system"""
    
    def __init__(self, db_path: str = "feedback_analytics.db"):
        self.db_path = db_path
        self._init_database()
        self._load_question_templates()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for feedback manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('demo_feedback.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback templates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_templates (
                template_id TEXT PRIMARY KEY,
                stakeholder_type TEXT NOT NULL,
                template_name TEXT NOT NULL,
                questions TEXT NOT NULL,
                created_at TEXT
            )
        ''')
        
        # Feedback responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_responses (
                response_id TEXT PRIMARY KEY,
                session_id TEXT,
                stakeholder_type TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                responses TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT,
                overall_rating INTEGER,
                nps_score INTEGER,
                recommendation_score REAL
            )
        ''')
        
        # Sentiment analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id TEXT,
                overall_sentiment TEXT,
                confidence_score REAL,
                key_themes TEXT,
                improvement_areas TEXT,
                positive_points TEXT,
                created_at TEXT,
                FOREIGN KEY (response_id) REFERENCES feedback_responses (response_id)
            )
        ''')
        
        # Demo analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_feedback_analytics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                stakeholder_type TEXT,
                overall_satisfaction REAL,
                likelihood_to_recommend REAL,
                demo_effectiveness REAL,
                feature_interest_scores TEXT,
                improvement_priorities TEXT,
                competitive_positioning REAL,
                created_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_question_templates(self):
        """Load stakeholder-specific question templates"""
        self.question_templates = {
            StakeholderFeedback.C_SUITE.value: [
                FeedbackQuestion(
                    question_id="business_impact",
                    question_text="How would you rate the demo's ability to demonstrate business impact?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5,
                    scale_labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
                ),
                FeedbackQuestion(
                    question_id="roi_clarity",
                    question_text="How clear was the return on investment demonstration?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="strategic_value",
                    question_text="Does this solution align with your strategic objectives?",
                    question_type=QuestionType.YES_NO
                ),
                FeedbackQuestion(
                    question_id="cost_efficiency",
                    question_text="How would you assess the cost-efficiency benefits demonstrated?",
                    question_type=QuestionType.LIKERT,
                    scale_labels=["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
                ),
                FeedbackQuestion(
                    question_id="implementation_complexity",
                    question_text="What concerns do you have about implementation complexity?",
                    question_type=QuestionType.TEXT
                ),
                FeedbackQuestion(
                    question_id="competitive_advantage",
                    question_text="How would you rate our competitive advantage compared to alternatives?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="budget_approval",
                    question_text="Based on this demo, how likely are you to approve budget for this solution?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=10,
                    scale_labels=["Not Likely", "Somewhat Likely", "Likely", "Very Likely", "Definitely"]
                )
            ],
            
            StakeholderFeedback.CLINICAL.value: [
                FeedbackQuestion(
                    question_id="clinical_relevance",
                    question_text="How clinically relevant were the scenarios demonstrated?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="workflow_integration",
                    question_text="How well would this integrate into current clinical workflows?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="decision_support",
                    question_text="How valuable was the AI-powered clinical decision support?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="patient_outcomes",
                    question_text="Do you believe this would improve patient outcomes?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="evidence_base",
                    question_text="How would you rate the evidence base for the recommendations?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="user_interface",
                    question_text="How user-friendly is the interface for clinical use?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="safety_concerns",
                    question_text="Do you have any safety concerns about using this system?",
                    question_type=QuestionType.TEXT
                ),
                FeedbackQuestion(
                    question_id="training_needs",
                    question_text="What additional training would be needed for clinical staff?",
                    question_type=QuestionType.TEXT
                )
            ],
            
            StakeholderFeedback.REGULATORY.value: [
                FeedbackQuestion(
                    question_id="compliance_clarity",
                    question_text="How clearly were compliance and regulatory requirements addressed?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="hipaa_adequacy",
                    question_text="How adequate were the HIPAA compliance demonstrations?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="risk_assessment",
                    question_text="How comprehensive was the risk assessment for patient safety?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="documentation_quality",
                    question_text="How would you rate the quality of documentation provided?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="audit_capabilities",
                    question_text="How suitable are the audit trails for regulatory review?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="validation_evidence",
                    question_text="How sufficient is the validation evidence presented?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="regulatory_roadmap",
                    question_text="What regulatory approvals or clearances are needed?",
                    question_type=QuestionType.TEXT
                ),
                FeedbackQuestion(
                    question_id="monitoring_requirements",
                    question_text="What ongoing monitoring and reporting would be required?",
                    question_type=QuestionType.TEXT
                )
            ],
            
            StakeholderFeedback.INVESTOR.value: [
                FeedbackQuestion(
                    question_id="market_opportunity",
                    question_text="How compelling is the market opportunity demonstrated?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="scalability_potential",
                    question_text="How scalable does the solution appear?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="competitive_position",
                    question_text="How strong is the competitive positioning?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="revenue_model",
                    question_text="How clear is the revenue model and monetization strategy?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="technical_innovation",
                    question_text="How innovative is the technical approach?",
                    question_type=QuestionType.RATING_SCALE,
                    scale_min=1,
                    scale_max=5
                ),
                FeedbackQuestion(
                    question_id="team_execution",
                    question_text="How well did the team demonstrate execution capability?",
                    question_type=QuestionType.LIKERT
                ),
                FeedbackQuestion(
                    question_id="investment_rationale",
                    question_text="What is your overall investment rationale based on this demo?",
                    question_type=QuestionType.TEXT
                ),
                FeedbackQuestion(
                    question_id="next_steps",
                    question_text="What are the key next steps for moving forward?",
                    question_type=QuestionType.TEXT
                )
            ]
        }
    
    def create_feedback_form(
        self,
        stakeholder_type: StakeholderFeedback,
        session_id: str,
        custom_questions: Optional[List[FeedbackQuestion]] = None
    ) -> List[FeedbackQuestion]:
        """Create feedback form for specific stakeholder"""
        # Get base questions for stakeholder
        base_questions = self.question_templates.get(stakeholder_type.value, [])
        
        # Add custom questions if provided
        if custom_questions:
            base_questions.extend(custom_questions)
        
        # Add standard evaluation questions
        standard_questions = [
            FeedbackQuestion(
                question_id="overall_demo_rating",
                question_text="Overall, how would you rate this demonstration?",
                question_type=QuestionType.RATING_SCALE,
                scale_min=1,
                scale_max=10,
                scale_labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]
            ),
            FeedbackQuestion(
                question_id="likelihood_to_recommend",
                question_text="How likely are you to recommend this solution to colleagues?",
                question_type=QuestionType.RATING_SCALE,
                scale_min=0,
                scale_max=10,
                scale_labels=["Not at all likely", "Very likely"]
            ),
            FeedbackQuestion(
                question_id="key_improvements",
                question_text="What are the top 3 areas that need improvement?",
                question_type=QuestionType.TEXT
            ),
            FeedbackQuestion(
                question_id="most_impressive",
                question_text="What was the most impressive feature demonstrated?",
                question_type=QuestionType.TEXT
            ),
            FeedbackQuestion(
                question_id="additional_feedback",
                question_text="Any additional comments or feedback?",
                question_type=QuestionType.TEXT,
                required=False
            )
        ]
        
        all_questions = base_questions + standard_questions
        self.logger.info(f"Created feedback form for {stakeholder_type.value} with {len(all_questions)} questions")
        
        return all_questions
    
    def collect_feedback(
        self,
        session_id: str,
        stakeholder_type: StakeholderFeedback,
        responses: Dict[str, Any],
        feedback_type: FeedbackType = FeedbackType.SURVEY,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Collect and store feedback response"""
        response_id = f"fb_{int(time.time())}_{stakeholder_type.value}_{len(responses)}"
        
        # Calculate overall metrics
        overall_rating = self._calculate_overall_rating(responses)
        nps_score = self._calculate_nps_score(responses)
        recommendation_score = self._calculate_recommendation_score(responses)
        
        feedback_response = FeedbackResponse(
            response_id=response_id,
            session_id=session_id,
            stakeholder_type=stakeholder_type.value,
            feedback_type=feedback_type,
            responses=responses,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store in database
        self._save_feedback_response(feedback_response, overall_rating, nps_score, recommendation_score)
        
        # Perform sentiment analysis
        self._analyze_sentiment(response_id, responses)
        
        self.logger.info(f"Collected feedback: {response_id}")
        return response_id
    
    def _calculate_overall_rating(self, responses: Dict[str, Any]) -> float:
        """Calculate overall rating from responses"""
        rating_questions = [
            "overall_demo_rating", "business_impact", "clinical_relevance", 
            "compliance_clarity", "market_opportunity", "roi_clarity",
            "decision_support", "user_interface", "documentation_quality"
        ]
        
        ratings = []
        for question_id in rating_questions:
            if question_id in responses:
                try:
                    rating = float(responses[question_id])
                    ratings.append(rating)
                except (ValueError, TypeError):
                    continue
        
        return sum(ratings) / len(ratings) if ratings else 0.0
    
    def _calculate_nps_score(self, responses: Dict[str, Any]) -> int:
        """Calculate Net Promoter Score"""
        nps_question = "likelihood_to_recommend"
        if nps_question in responses:
            try:
                score = int(responses[nps_question])
                # NPS calculation: % Promoters (9-10) - % Detractors (0-6)
                if score >= 9:
                    return 100  # Promoter
                elif score >= 7:
                    return 0    # Passive
                else:
                    return -100 # Detractor
            except (ValueError, TypeError):
                return 0
        return 0
    
    def _calculate_recommendation_score(self, responses: Dict[str, Any]) -> float:
        """Calculate general recommendation likelihood"""
        # Combine multiple recommendation indicators
        factors = []
        
        # Overall rating (normalized to 0-1)
        if "overall_demo_rating" in responses:
            try:
                rating = float(responses["overall_demo_rating"])
                factors.append(rating / 10.0)  # Normalize to 0-1
            except (ValueError, TypeError):
                pass
        
        # Specific recommendation questions
        rec_questions = [
            "budget_approval", "strategic_value", "clinical_relevance", 
            "compliance_adequacy", "market_opportunity"
        ]
        
        for question in rec_questions:
            if question in responses:
                try:
                    score = float(responses[question])
                    factors.append(score / 5.0)  # Normalize to 0-1
                except (ValueError, TypeError):
                    pass
        
        return sum(factors) / len(factors) if factors else 0.0
    
    def _save_feedback_response(
        self,
        response: FeedbackResponse,
        overall_rating: float,
        nps_score: int,
        recommendation_score: float
    ):
        """Save feedback response to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback_responses
            (response_id, session_id, stakeholder_type, feedback_type, 
             responses, timestamp, metadata, overall_rating, nps_score, recommendation_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            response.response_id,
            response.session_id,
            response.stakeholder_type,
            response.feedback_type.value,
            json.dumps(response.responses),
            response.timestamp.isoformat(),
            json.dumps(response.metadata),
            overall_rating,
            nps_score,
            recommendation_score
        ))
        
        conn.commit()
        conn.close()
    
    def _analyze_sentiment(self, response_id: str, responses: Dict[str, Any]):
        """Perform sentiment analysis on feedback responses"""
        # Extract text responses for analysis
        text_responses = []
        for key, value in responses.items():
            if isinstance(value, str) and len(value) > 10:  # Meaningful text
                text_responses.append(value)
        
        if not text_responses:
            return
        
        # Simple sentiment analysis (in production, use NLP libraries like spaCy, NLTK)
        sentiment_results = self._simple_sentiment_analysis(text_responses)
        
        # Store sentiment analysis
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_analysis
            (response_id, overall_sentiment, confidence_score, key_themes, 
             improvement_areas, positive_points, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            response_id,
            sentiment_results['overall_sentiment'],
            sentiment_results['confidence_score'],
            json.dumps(sentiment_results['key_themes']),
            json.dumps(sentiment_results['improvement_areas']),
            json.dumps(sentiment_results['positive_points']),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def _simple_sentiment_analysis(self, text_responses: List[str]) -> Dict[str, Any]:
        """Simple sentiment analysis using keyword detection"""
        positive_words = [
            "excellent", "great", "impressive", "innovative", "clear", "useful",
            "helpful", "valuable", "effective", "efficient", "intuitive",
            "professional", "comprehensive", "well-designed", "useful"
        ]
        
        negative_words = [
            "confusing", "difficult", "unclear", "complex", "problematic",
            "concern", "worry", "issue", "limitation", "gap", "weak",
            "inadequate", "insufficient", "complicated"
        ]
        
        all_text = " ".join(text_responses).lower()
        
        # Count positive and negative words
        positive_count = sum(all_text.count(word) for word in positive_words)
        negative_count = sum(all_text.count(word) for word in negative_words)
        
        total_words = len(all_text.split())
        positive_ratio = positive_count / total_words if total_words > 0 else 0
        negative_ratio = negative_count / total_words if total_words > 0 else 0
        
        # Determine overall sentiment
        if positive_ratio > negative_ratio * 1.5:
            overall_sentiment = "positive"
            confidence = min(0.9, positive_ratio / (positive_ratio + negative_ratio + 0.1))
        elif negative_ratio > positive_ratio * 1.5:
            overall_sentiment = "negative"
            confidence = min(0.9, negative_ratio / (positive_ratio + negative_ratio + 0.1))
        else:
            overall_sentiment = "neutral"
            confidence = 0.7
        
        return {
            "overall_sentiment": overall_sentiment,
            "confidence_score": confidence,
            "key_themes": self._extract_themes(text_responses),
            "improvement_areas": self._extract_improvement_areas(text_responses),
            "positive_points": self._extract_positive_points(text_responses)
        }
    
    def _extract_themes(self, text_responses: List[str]) -> List[str]:
        """Extract key themes from text responses"""
        themes = []
        theme_keywords = {
            "User Experience": ["interface", "user-friendly", "easy", "intuitive"],
            "Clinical Value": ["patient outcome", "clinical decision", "workflow"],
            "Technical Performance": ["speed", "reliability", "accuracy"],
            "Compliance": ["hipaa", "security", "audit", "regulation"],
            "Business Value": ["roi", "cost", "efficiency", "savings"]
        }
        
        combined_text = " ".join(text_responses).lower()
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _extract_improvement_areas(self, text_responses: List[str]) -> List[str]:
        """Extract improvement areas from feedback"""
        improvement_indicators = [
            "improve", "better", "enhance", "upgrade", "fix", "address",
            "concern", "issue", "problem", "limitation", "gap"
        ]
        
        improvements = []
        for response in text_responses:
            response_lower = response.lower()
            if any(indicator in response_lower for indicator in improvement_indicators):
                # Simple extraction - in production use NLP
                sentences = response.split('.')
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in improvement_indicators):
                        improvements.append(sentence.strip())
        
        return improvements[:5]  # Limit to top 5
    
    def _extract_positive_points(self, text_responses: List[str]) -> List[str]:
        """Extract positive highlights from feedback"""
        positive_indicators = [
            "great", "excellent", "impressive", "helpful", "valuable",
            "useful", "effective", "innovative", "professional"
        ]
        
        positives = []
        for response in text_responses:
            response_lower = response.lower()
            if any(indicator in response_lower for indicator in positive_indicators):
                # Simple extraction - in production use NLP
                sentences = response.split('.')
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in positive_indicators):
                        positives.append(sentence.strip())
        
        return positives[:5]  # Limit to top 5
    
    def get_feedback_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive feedback analytics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Base query
        if session_id:
            cursor.execute('''
                SELECT * FROM feedback_responses WHERE session_id = ?
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT * FROM feedback_responses
            ''')
        
        responses = cursor.fetchall()
        
        if not responses:
            return {"error": "No feedback responses found"}
        
        # Calculate overall metrics
        overall_ratings = [r[7] for r in responses if r[7] is not None]  # overall_rating column
        nps_scores = [r[8] for r in responses if r[8] is not None]  # nps_score column
        recommendation_scores = [r[9] for r in responses if r[9] is not None]  # recommendation_score column
        
        # Stakeholder breakdown
        stakeholder_breakdown = {}
        for response in responses:
            stakeholder = response[2]  # stakeholder_type column
            if stakeholder not in stakeholder_breakdown:
                stakeholder_breakdown[stakeholder] = 0
            stakeholder_breakdown[stakeholder] += 1
        
        # Sentiment analysis
        cursor.execute('''
            SELECT overall_sentiment, COUNT(*) as count
            FROM sentiment_analysis
            GROUP BY overall_sentiment
        ''')
        sentiment_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
        
        conn.close()
        
        analytics = {
            "summary": {
                "total_responses": len(responses),
                "average_rating": round(sum(overall_ratings) / len(overall_ratings), 2) if overall_ratings else 0,
                "nps_score": self._calculate_overall_nps(nps_scores),
                "recommendation_likelihood": round(sum(recommendation_scores) / len(recommendation_scores), 2) if recommendation_scores else 0
            },
            "stakeholder_breakdown": stakeholder_breakdown,
            "sentiment_analysis": sentiment_breakdown,
            "top_improvement_areas": self._get_top_improvement_areas(),
            "key_positive_themes": self._get_positive_themes(),
            "recommendations": self._generate_improvement_recommendations(responses)
        }
        
        return analytics
    
    def _calculate_overall_nps(self, nps_scores: List[int]) -> int:
        """Calculate overall NPS from individual scores"""
        if not nps_scores:
            return 0
        
        # Group by sentiment
        promoters = len([score for score in nps_scores if score == 100])
        detractors = len([score for score in nps_scores if score == -100])
        total = len(nps_scores)
        
        if total == 0:
            return 0
        
        return round(((promoters / total) - (detractors / total)) * 100)
    
    def _get_top_improvement_areas(self) -> List[str]:
        """Get top improvement areas from all feedback"""
        # In production, analyze all improvement_areas from sentiment_analysis
        return [
            "User interface complexity",
            "Workflow integration challenges",
            "Training and adoption support",
            "Regulatory documentation clarity",
            "Performance optimization needs"
        ]
    
    def _get_positive_themes(self) -> List[str]:
        """Get top positive themes from all feedback"""
        # In production, analyze all positive_points from sentiment_analysis
        return [
            "AI accuracy and reliability",
            "Clinical decision support value",
            "User-friendly interface design",
            "Comprehensive documentation",
            "Professional demonstration quality"
        ]
    
    def _generate_improvement_recommendations(self, responses: List) -> List[str]:
        """Generate improvement recommendations based on feedback"""
        recommendations = []
        
        # Analyze response patterns
        overall_ratings = [r[7] for r in responses if r[7] is not None]
        
        if overall_ratings and sum(overall_ratings) / len(overall_ratings) < 7:
            recommendations.append("Focus on improving overall demo quality and impact")
        
        # Analyze common improvement areas from responses
        common_improvements = self._get_top_improvement_areas()
        recommendations.extend([
            f"Address: {improvement}" for improvement in common_improvements[:3]
        ])
        
        return recommendations
    
    def generate_feedback_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive feedback report for session"""
        analytics = self.get_feedback_analytics(session_id)
        
        # Get session-specific responses
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM feedback_responses WHERE session_id = ?
        ''', (session_id,))
        
        session_responses = cursor.fetchall()
        
        # Get sentiment analysis for session
        cursor.execute('''
            SELECT * FROM sentiment_analysis WHERE response_id IN
            (SELECT response_id FROM feedback_responses WHERE session_id = ?)
        ''', (session_id,))
        
        session_sentiments = cursor.fetchall()
        
        conn.close()
        
        report = {
            "session_report": {
                "session_id": session_id,
                "total_responses": len(session_responses),
                "feedback_types": list(set(r[3] for r in session_responses)),
                "stakeholder_types": list(set(r[2] for r in session_responses)),
                "collection_date_range": {
                    "start": min(r[5] for r in session_responses) if session_responses else None,
                    "end": max(r[5] for r in session_responses) if session_responses else None
                }
            },
            "feedback_analytics": analytics,
            "sentiment_insights": self._generate_sentiment_insights(session_sentiments),
            "stakeholder_feedback": self._generate_stakeholder_summary(session_responses),
            "action_items": self._generate_action_items(session_responses, session_sentiments),
            "next_steps": [
                "Review and implement priority improvements",
                "Schedule follow-up sessions with key stakeholders",
                "Update demo materials based on feedback",
                "Track improvement implementation",
                "Conduct follow-up feedback collection"
            ]
        }
        
        return report
    
    def _generate_sentiment_insights(self, sentiments: List) -> Dict[str, Any]:
        """Generate sentiment insights from analysis"""
        if not sentiments:
            return {"message": "No sentiment analysis available"}
        
        sentiment_counts = {}
        for sentiment in sentiments:
            overall_sentiment = sentiment[2]  # overall_sentiment column
            sentiment_counts[overall_sentiment] = sentiment_counts.get(overall_sentiment, 0) + 1
        
        return {
            "overall_sentiment_distribution": sentiment_counts,
            "dominant_sentiment": max(sentiment_counts, key=sentiment_counts.get),
            "confidence_levels": [s[3] for s in sentiments if s[3]]  # confidence_score column
        }
    
    def _generate_stakeholder_summary(self, responses: List) -> Dict[str, Any]:
        """Generate stakeholder-specific summary"""
        stakeholder_summaries = {}
        
        for response in responses:
            stakeholder = response[2]  # stakeholder_type column
            if stakeholder not in stakeholder_summaries:
                stakeholder_summaries[stakeholder] = {
                    "response_count": 0,
                    "average_rating": [],
                    "key_concerns": [],
                    "positive_feedback": []
                }
            
            stakeholder_summaries[stakeholder]["response_count"] += 1
            
            # Parse responses
            try:
                response_data = json.loads(response[4])  # responses column
                
                # Collect ratings
                if "overall_demo_rating" in response_data:
                    try:
                        rating = float(response["overall_demo_rating"])
                        stakeholder_summaries[stakeholder]["average_rating"].append(rating)
                    except (ValueError, TypeError):
                        pass
                
                # Collect key concerns and positive feedback
                if "additional_feedback" in response_data:
                    feedback_text = response_data["additional_feedback"]
                    if len(feedback_text) > 20:
                        stakeholder_summaries[stakeholder]["positive_feedback"].append(feedback_text[:200])
                
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Calculate averages
        for stakeholder in stakeholder_summaries:
            ratings = stakeholder_summaries[stakeholder]["average_rating"]
            if ratings:
                stakeholder_summaries[stakeholder]["average_rating"] = sum(ratings) / len(ratings)
            else:
                stakeholder_summaries[stakeholder]["average_rating"] = 0
        
        return stakeholder_summaries
    
    def _generate_action_items(self, responses: List, sentiments: List) -> List[str]:
        """Generate specific action items from feedback"""
        action_items = []
        
        # Analyze improvement priorities
        improvement_keywords = {
            "User Interface": ["interface", "ui", "user experience", "ease of use"],
            "Performance": ["speed", "performance", "lag", "response time"],
            "Documentation": ["documentation", "help", "instructions", "guide"],
            "Integration": ["integration", "workflow", "compatibility", "system"],
            "Training": ["training", "education", "learning", "support"]
        }
        
        # Check for common themes in responses
        all_feedback_text = []
        for response in responses:
            try:
                response_data = json.loads(response[4])
                for key, value in response_data.items():
                    if isinstance(value, str) and len(value) > 20:
                        all_feedback_text.append(value.lower())
            except (json.JSONDecodeError, TypeError):
                continue
        
        combined_text = " ".join(all_feedback_text)
        
        for category, keywords in improvement_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                action_items.append(f"Improve {category.lower()} based on stakeholder feedback")
        
        # Add general action items
        if len(responses) < 5:
            action_items.append("Increase stakeholder engagement and response rates")
        
        if not action_items:
            action_items = ["Continue monitoring feedback trends", "Maintain high demo quality standards"]
        
        return action_items

def main():
    """Main function for feedback manager CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Demo Feedback Manager")
    parser.add_argument("--create-form", type=str, help="Create feedback form for stakeholder type")
    parser.add_argument("--session-id", type=str, help="Session ID for feedback")
    parser.add_argument("--collect", action="store_true", help="Collect feedback")
    parser.add_argument("--analytics", action="store_true", help="Show feedback analytics")
    parser.add_argument("--report", action="store_true", help="Generate feedback report")
    parser.add_argument("--stakeholder", type=str, help="Stakeholder type")
    
    args = parser.parse_args()
    
    manager = DemoFeedbackManager()
    
    if args.create_form:
        try:
            stakeholder_type = StakeholderFeedback(args.stakeholder)
            questions = manager.create_feedback_form(stakeholder_type, args.session_id or "demo_session")
            print(f"Created feedback form for {args.stakeholder}:")
            for i, question in enumerate(questions, 1):
                print(f"{i}. {question.question_text}")
        except Exception as e:
            print(f"Error creating form: {e}")
    
    elif args.analytics:
        analytics = manager.get_feedback_analytics()
        print(json.dumps(analytics, indent=2))
    
    elif args.report:
        if args.session_id:
            report = manager.generate_feedback_report(args.session_id)
            print(json.dumps(report, indent=2))
        else:
            print("Session ID required for report generation")

if __name__ == "__main__":
    main()
