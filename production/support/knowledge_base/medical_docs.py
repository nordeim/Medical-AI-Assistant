"""
Knowledge Base and Self-Service Support System
Medical documentation and self-service support for healthcare users
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import re
import logging

from config.support_config import SupportConfig

logger = logging.getLogger(__name__)

class ContentType(Enum):
    MEDICAL_DOCUMENTATION = "medical_documentation"
    VIDEO_TUTORIAL = "video_tutorial"
    INTERACTIVE_GUIDE = "interactive_guide"
    FAQ = "faq"
    CASE_STUDY = "case_study"
    CLINICAL_PROTOCOL = "clinical_protocol"
    TROUBLESHOOTING_GUIDE = "troubleshooting_guide"
    SYSTEM_ADMIN_GUIDE = "system_admin_guide"

class MedicalSpecialty(Enum):
    GENERAL_MEDICINE = "general_medicine"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    ONCOLOGY = "oncology"
    PEDIATRICS = "pediatrics"
    PSYCHIATRY = "psychiatry"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    FAMILY_MEDICINE = "family_medicine"
    SURGERY = "surgery"
    INTERNAL_MEDICINE = "internal_medicine"

class UserRole(Enum):
    PHYSICIAN = "physician"
    NURSE = "nurse"
    TECHNICIAN = "technician"
    ADMINISTRATOR = "administrator"
    IT_SUPPORT = "it_support"
    MEDICAL_STUDENT = "medical_student"

@dataclass
class MedicalKeyword:
    """Medical keyword for enhanced search"""
    term: str
    medical_code: Optional[str]  # ICD-10, CPT, etc.
    specialty: MedicalSpecialty
    context: str
    frequency: int

@dataclass
class KnowledgeContent:
    """Knowledge base content"""
    id: str
    title: str
    content: str
    content_type: ContentType
    medical_specialty: Optional[MedicalSpecialty]
    target_roles: List[UserRole]
    keywords: List[str]
    medical_keywords: List[MedicalKeyword]
    difficulty_level: int  # 1-5, 1 = basic, 5 = advanced
    last_updated: datetime
    author_id: str
    author_name: str
    version: str
    approved: bool
    tags: List[str]
    view_count: int
    helpful_count: int
    not_helpful_count: int
    related_content: List[str]
    compliance_tags: List[str]
    regulatory_references: List[str]

@dataclass
class SearchResult:
    """Knowledge base search result"""
    content_id: str
    title: str
    content_type: ContentType
    relevance_score: float
    medical_specialty: Optional[MedicalSpecialty]
    target_roles: List[UserRole]
    snippet: str
    tags: List[str]
    last_updated: datetime
    view_count: int

@dataclass
class UserFeedback:
    """Feedback on knowledge base content"""
    id: str
    content_id: str
    user_id: str
    user_name: str
    user_facility: str
    user_role: UserRole
    rating: int  # 1-5
    helpful: bool
    comment: Optional[str]
    submitted_at: datetime

class MedicalSearchEngine:
    """Medical-aware search engine for knowledge base"""
    
    def __init__(self):
        self.medical_dictionary = {
            # Basic medical terms
            "chest pain": {"specialty": MedicalSpecialty.CARDIOLOGY, "code": "R07.9"},
            "myocardial infarction": {"specialty": MedicalSpecialty.CARDIOLOGY, "code": "I21.9"},
            "stroke": {"specialty": MedicalSpecialty.NEUROLOGY, "code": "I63.9"},
            "diabetes": {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "code": "E11.9"},
            "hypertension": {"specialty": MedicalSpecialty.CARDIOLOGY, "code": "I10"},
            "arrhythmia": {"specialty": MedicalSpecialty.CARDIOLOGY, "code": "I47.9"},
            "sepsis": {"specialty": MedicalSpecialty.EMERGENCY_MEDICINE, "code": "A41.9"},
            "anemia": {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "code": "D64.9"},
            
            # System-related terms
            "ehr integration": {"specialty": MedicalSpecialty.GENERAL_MEDICINE, "code": None},
            "patient portal": {"specialty": MedicalSpecialty.GENERAL_MEDICINE, "code": None},
            "medical imaging": {"specialty": MedicalSpecialty.RADIOLOGY, "code": None},
            "laboratory results": {"specialty": MedicalSpecialty.PATHOLOGY, "code": None},
            "medication management": {"specialty": MedicalSpecialty.INTERNAL_MEDICINE, "code": None}
        }
        
        self.role_based_keywords = {
            UserRole.PHYSICIAN: ["diagnosis", "treatment", "clinical decision", "prescription", "procedure"],
            UserRole.NURSE: ["patient care", "medication administration", "vital signs", "nursing assessment"],
            UserRole.ADMINISTRATOR: ["scheduling", "billing", "compliance", "reporting", "workflow"],
            UserRole.IT_SUPPORT: ["configuration", "troubleshooting", "integration", "maintenance", "security"]
        }
    
    def search_knowledge_base(
        self,
        query: str,
        user_role: Optional[UserRole] = None,
        medical_specialty: Optional[MedicalSpecialty] = None,
        content_types: Optional[List[ContentType]] = None,
        max_results: int = 20
    ) -> List[SearchResult]:
        """Search knowledge base with medical context awareness"""
        
        # Parse query for medical terms
        medical_terms = self._extract_medical_terms(query.lower())
        
        # Get relevant content
        candidates = self._get_candidate_content(medical_specialty, content_types)
        
        # Score and rank results
        scored_results = []
        for content in candidates:
            score = self._calculate_relevance_score(
                content, query, medical_terms, user_role, medical_specialty
            )
            
            if score > 0:  # Only include relevant results
                snippet = self._generate_snippet(content.content, query, 200)
                
                scored_results.append(SearchResult(
                    content_id=content.id,
                    title=content.title,
                    content_type=content.content_type,
                    relevance_score=score,
                    medical_specialty=content.medical_specialty,
                    target_roles=content.target_roles,
                    snippet=snippet,
                    tags=content.tags,
                    last_updated=content.last_updated,
                    view_count=content.view_count
                ))
        
        # Sort by relevance score and return top results
        return sorted(scored_results, key=lambda x: x.relevance_score, reverse=True)[:max_results]
    
    def suggest_related_content(self, content_id: str, existing_knowledge_base: Dict[str, KnowledgeContent]) -> List[SearchResult]:
        """Suggest related content based on medical keywords and tags"""
        
        if content_id not in existing_knowledge_base:
            return []
        
        content = existing_knowledge_base[content_id]
        related_results = []
        
        # Find content with overlapping keywords
        for other_id, other_content in existing_knowledge_base.items():
            if other_id == content_id:
                continue
            
            # Calculate overlap score
            keyword_overlap = len(set(content.keywords) & set(other_content.keywords))
            tag_overlap = len(set(content.tags) & set(other_content.tags))
            specialty_match = 1 if content.medical_specialty == other_content.medical_specialty else 0
            
            if keyword_overlap > 0 or tag_overlap > 0 or specialty_match:
                relevance_score = (keyword_overlap * 2 + tag_overlap + specialty_match) / 10
                
                related_results.append(SearchResult(
                    content_id=other_id,
                    title=other_content.title,
                    content_type=other_content.content_type,
                    relevance_score=relevance_score,
                    medical_specialty=other_content.medical_specialty,
                    target_roles=other_content.target_roles,
                    snippet=other_content.content[:200] + "...",
                    tags=other_content.tags,
                    last_updated=other_content.last_updated,
                    view_count=other_content.view_count
                ))
        
        return sorted(related_results, key=lambda x: x.relevance_score, reverse=True)[:10]
    
    def _extract_medical_terms(self, query: str) -> List[str]:
        """Extract medical terms from search query"""
        found_terms = []
        
        for term, metadata in self.medical_dictionary.items():
            if term in query:
                found_terms.append(term)
        
        return found_terms
    
    def _get_candidate_content(
        self,
        medical_specialty: Optional[MedicalSpecialty],
        content_types: Optional[List[ContentType]]
    ) -> List[KnowledgeContent]:
        """Get candidate content based on filters"""
        # This would access the actual knowledge base
        # For now, return empty list as this is a placeholder
        return []
    
    def _calculate_relevance_score(
        self,
        content: KnowledgeContent,
        query: str,
        medical_terms: List[str],
        user_role: Optional[UserRole],
        medical_specialty: Optional[MedicalSpecialty]
    ) -> float:
        """Calculate relevance score for content"""
        
        score = 0.0
        query_lower = query.lower()
        content_lower = content.content.lower()
        title_lower = content.title.lower()
        
        # Title match (highest weight)
        if query_lower in title_lower:
            score += 10.0
        
        # Content match
        if query_lower in content_lower:
            score += 5.0
        
        # Medical term matches
        for term in medical_terms:
            if term in content_lower or term in content.keywords:
                score += 3.0
        
        # Specialty match
        if medical_specialty and content.medical_specialty == medical_specialty:
            score += 2.0
        
        # Role-based relevance
        if user_role and user_role in content.target_roles:
            score += 1.5
        
        # Keyword matches
        query_words = set(query_lower.split())
        content_words = set(content_lower.split()) | set(content.keywords)
        keyword_matches = len(query_words & content_words)
        score += keyword_matches * 0.5
        
        return score
    
    def _generate_snippet(self, content: str, query: str, max_length: int) -> str:
        """Generate search result snippet"""
        
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Find first occurrence of query in content
        index = content_lower.find(query_lower)
        
        if index == -1:
            # Fallback to beginning of content
            return content[:max_length] + "..." if len(content) > max_length else content
        
        # Extract snippet around the match
        start = max(0, index - max_length // 4)
        end = min(len(content), start + max_length)
        
        snippet = content[start:end]
        
        # Clean up snippet
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet

class KnowledgeBaseSystem:
    """Main knowledge base system for healthcare support"""
    
    def __init__(self):
        self.content_library: Dict[str, KnowledgeContent] = {}
        self.user_feedback: Dict[str, List[UserFeedback]] = defaultdict(list)
        self.content_counter = 0
        self.search_engine = MedicalSearchEngine()
        self.categories = {
            "clinical_workflows": ["patient_admission", "discharge_process", "care_planning"],
            "patient_care": ["medication_management", "vital_signs", "pain_management"],
            "system_administration": ["user_management", "configuration", "troubleshooting"],
            "compliance": ["hipaa", "regulatory", "audit_trail", "data_protection"],
            "emergency_procedures": ["code_blue", "rapid_response", "emergency_contact"],
            "integration": ["ehr_connection", "lab_systems", "imaging_systems"],
            "training": ["user_training", "certification", "best_practices"]
        }
        
        # Initialize with sample content
        self._initialize_sample_content()
    
    async def create_content(
        self,
        title: str,
        content: str,
        content_type: ContentType,
        author_id: str,
        author_name: str,
        medical_specialty: Optional[MedicalSpecialty] = None,
        target_roles: Optional[List[UserRole]] = None,
        tags: Optional[List[str]] = None,
        compliance_tags: Optional[List[str]] = None
    ) -> KnowledgeContent:
        """Create new knowledge base content"""
        
        self.content_counter += 1
        content_id = f"KB-{self.content_counter:04d}"
        
        # Extract medical keywords
        medical_keywords = self._extract_medical_keywords(content, title)
        
        # Set default target roles if not specified
        if target_roles is None:
            target_roles = list(UserRole)
        
        # Set default tags if not specified
        if tags is None:
            tags = []
        
        if compliance_tags is None:
            compliance_tags = []
        
        knowledge_content = KnowledgeContent(
            id=content_id,
            title=title,
            content=content,
            content_type=content_type,
            medical_specialty=medical_specialty,
            target_roles=target_roles,
            keywords=self._extract_keywords(content, title),
            medical_keywords=medical_keywords,
            difficulty_level=self._assess_difficulty_level(content),
            last_updated=datetime.now(),
            author_id=author_id,
            author_name=author_name,
            version="1.0",
            approved=False,  # Requires approval in production
            tags=tags,
            view_count=0,
            helpful_count=0,
            not_helpful_count=0,
            related_content=[],
            compliance_tags=compliance_tags,
            regulatory_references=self._extract_regulatory_references(content)
        )
        
        self.content_library[content_id] = knowledge_content
        
        logger.info(f"Created knowledge base content: {content_id} - {title}")
        return knowledge_content
    
    async def search_content(
        self,
        query: str,
        user_role: Optional[UserRole] = None,
        medical_specialty: Optional[MedicalSpecialty] = None,
        content_types: Optional[List[ContentType]] = None,
        max_results: int = 20
    ) -> List[SearchResult]:
        """Search knowledge base content"""
        
        results = self.search_engine.search_knowledge_base(
            query, user_role, medical_specialty, content_types, max_results
        )
        
        # Update view counts
        for result in results:
            if result.content_id in self.content_library:
                self.content_library[result.content_id].view_count += 1
        
        logger.info(f"Search performed: '{query}' - {len(results)} results")
        return results
    
    async def get_content_by_id(self, content_id: str) -> Optional[KnowledgeContent]:
        """Get specific content by ID"""
        return self.content_library.get(content_id)
    
    async def get_content_by_category(self, category: str, limit: int = 50) -> List[KnowledgeContent]:
        """Get content by category"""
        
        if category not in self.categories:
            return []
        
        category_content = [
            content for content in self.content_library.values()
            if any(tag in content.tags for tag in self.categories[category])
        ]
        
        return sorted(category_content, key=lambda x: x.last_updated, reverse=True)[:limit]
    
    async def get_content_by_specialty(self, specialty: MedicalSpecialty, limit: int = 50) -> List[KnowledgeContent]:
        """Get content by medical specialty"""
        
        specialty_content = [
            content for content in self.content_library.values()
            if content.medical_specialty == specialty
        ]
        
        return sorted(specialty_content, key=lambda x: x.last_updated, reverse=True)[:limit]
    
    async def get_popular_content(self, days: int = 30, limit: int = 20) -> List[KnowledgeContent]:
        """Get most popular content based on view count and recency"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_content = [
            content for content in self.content_library.values()
            if content.last_updated >= cutoff_date
        ]
        
        return sorted(recent_content, key=lambda x: x.view_count, reverse=True)[:limit]
    
    async def submit_feedback(
        self,
        content_id: str,
        user_id: str,
        user_name: str,
        user_facility: str,
        user_role: UserRole,
        rating: int,
        helpful: bool,
        comment: Optional[str] = None
    ) -> UserFeedback:
        """Submit feedback on knowledge base content"""
        
        if content_id not in self.content_library:
            raise ValueError(f"Content {content_id} not found")
        
        feedback_id = f"FB-{content_id}-{len(self.user_feedback[content_id]) + 1}"
        
        feedback = UserFeedback(
            id=feedback_id,
            content_id=content_id,
            user_id=user_id,
            user_name=user_name,
            user_facility=user_facility,
            user_role=user_role,
            rating=rating,
            helpful=helpful,
            comment=comment,
            submitted_at=datetime.now()
        )
        
        self.user_feedback[content_id].append(feedback)
        
        # Update content helpfulness counts
        content = self.content_library[content_id]
        if helpful:
            content.helpful_count += 1
        else:
            content.not_helpful_count += 1
        
        logger.info(f"Feedback submitted for {content_id}: {rating} stars, helpful={helpful}")
        return feedback
    
    async def get_related_content(self, content_id: str) -> List[SearchResult]:
        """Get related content suggestions"""
        return self.search_engine.suggest_related_content(content_id, self.content_library)
    
    async def get_content_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get knowledge base analytics"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        period_content = [
            content for content in self.content_library.values()
            if content.last_updated >= cutoff_date
        ]
        
        total_content = len(self.content_library)
        total_views = sum(content.view_count for content in self.content_library.values())
        total_feedback = sum(len(feedback_list) for feedback_list in self.user_feedback.values())
        
        # Content type distribution
        type_distribution = defaultdict(int)
        for content in self.content_library.values():
            type_distribution[content.content_type.value] += 1
        
        # Medical specialty distribution
        specialty_distribution = defaultdict(int)
        for content in self.content_library.values():
            if content.medical_specialty:
                specialty_distribution[content.medical_specialty.value] += 1
        
        # Top performing content
        top_content = sorted(
            self.content_library.values(),
            key=lambda x: x.view_count,
            reverse=True
        )[:10]
        
        return {
            "period_days": days,
            "content_summary": {
                "total_content": total_content,
                "period_content": len(period_content),
                "total_views": total_views,
                "total_feedback": total_feedback
            },
            "content_distribution": {
                "by_type": dict(type_distribution),
                "by_specialty": dict(specialty_distribution)
            },
            "top_performing_content": [
                {
                    "id": content.id,
                    "title": content.title,
                    "views": content.view_count,
                    "helpful_ratio": content.helpful_count / (content.helpful_count + content.not_helpful_count) if (content.helpful_count + content.not_helpful_count) > 0 else 0
                }
                for content in top_content
            ],
            "user_engagement": {
                "average_rating": self._calculate_average_rating(),
                "feedback_rate": self._calculate_feedback_rate(),
                "search_queries": self._get_popular_search_terms()
            }
        }
    
    async def suggest_content_for_user(
        self,
        user_role: UserRole,
        medical_specialty: Optional[MedicalSpecialty] = None,
        recent_searches: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Suggest relevant content based on user profile"""
        
        # Get content relevant to user role
        role_relevant = [
            content for content in self.content_library.values()
            if user_role in content.target_roles
        ]
        
        # Filter by medical specialty if provided
        if medical_specialty:
            role_relevant = [
                content for content in role_relevant
                if content.medical_specialty == medical_specialty
            ]
        
        # Score and rank
        suggestions = []
        for content in role_relevant[:20]:  # Limit to top 20
            score = content.view_count * 0.5 + content.helpful_count * 2
            
            suggestions.append(SearchResult(
                content_id=content.id,
                title=content.title,
                content_type=content.content_type,
                relevance_score=score / 100,  # Normalize score
                medical_specialty=content.medical_specialty,
                target_roles=content.target_roles,
                snippet=content.content[:200] + "...",
                tags=content.tags,
                last_updated=content.last_updated,
                view_count=content.view_count
            ))
        
        return sorted(suggestions, key=lambda x: x.relevance_score, reverse=True)[:10]
    
    def _initialize_sample_content(self) -> None:
        """Initialize knowledge base with sample medical content"""
        
        sample_content = [
            {
                "title": "Emergency Cardiac Arrest Protocol",
                "content": "Immediate response protocol for cardiac arrest situations. Start CPR within 10 seconds, use defibrillator within 2 minutes, call code team immediately.",
                "type": ContentType.CLINICAL_PROTOCOL,
                "specialty": MedicalSpecialty.EMERGENCY_MEDICINE,
                "roles": [UserRole.PHYSICIAN, UserRole.NURSE],
                "tags": ["emergency", "cardiac", "protocol", "cpr"]
            },
            {
                "title": "EHR System User Guide",
                "content": "Complete guide to using the Electronic Health Record system for patient documentation, order entry, and results review.",
                "type": ContentType.SYSTEM_ADMIN_GUIDE,
                "specialty": MedicalSpecialty.GENERAL_MEDICINE,
                "roles": [UserRole.PHYSICIAN, UserRole.NURSE, UserRole.ADMINISTRATOR],
                "tags": ["ehr", "documentation", "system", "guide"]
            },
            {
                "title": "Patient Data Security Best Practices",
                "content": "Essential security practices for protecting patient data in compliance with HIPAA regulations and hospital policies.",
                "type": ContentType.MEDICAL_DOCUMENTATION,
                "specialty": MedicalSpecialty.GENERAL_MEDICINE,
                "roles": [UserRole.ADMINISTRATOR, UserRole.IT_SUPPORT],
                "tags": ["security", "hipaa", "compliance", "data_protection"]
            },
            {
                "title": "Troubleshooting Medical Device Connections",
                "content": "Common issues and solutions for medical device connectivity problems. Includes network configuration and driver updates.",
                "type": ContentType.TROUBLESHOOTING_GUIDE,
                "specialty": MedicalSpecialty.GENERAL_MEDICINE,
                "roles": [UserRole.IT_SUPPORT, UserRole.TECHNICIAN],
                "tags": ["troubleshooting", "medical_device", "connectivity", "network"]
            }
        ]
        
        # Create sample content
        for item in sample_content:
            asyncio.create_task(
                self.create_content(
                    title=item["title"],
                    content=item["content"],
                    content_type=item["type"],
                    author_id="system",
                    author_name="System Administrator",
                    medical_specialty=item["specialty"],
                    target_roles=item["roles"],
                    tags=item["tags"],
                    compliance_tags=["hipaa"] if "security" in item["tags"] else []
                )
            )
    
    def _extract_medical_keywords(self, content: str, title: str) -> List[MedicalKeyword]:
        """Extract medical keywords from content"""
        
        keywords = []
        full_text = f"{title} {content}".lower()
        
        for term, metadata in self.search_engine.medical_dictionary.items():
            if term in full_text:
                keywords.append(MedicalKeyword(
                    term=term,
                    medical_code=metadata["code"],
                    specialty=metadata["specialty"],
                    context=self._determine_keyword_context(term, full_text),
                    frequency=full_text.count(term)
                ))
        
        return keywords
    
    def _determine_keyword_context(self, term: str, text: str) -> str:
        """Determine the context in which a medical term appears"""
        
        sentences = text.split('.')
        for sentence in sentences:
            if term in sentence:
                if any(word in sentence for word in ["diagnosis", "treatment", "management"]):
                    return "clinical"
                elif any(word in sentence for word in ["emergency", "urgent", "critical"]):
                    return "emergency"
                elif any(word in sentence for word in ["monitoring", "observation"]):
                    return "monitoring"
                else:
                    return "general"
        
        return "general"
    
    def _extract_keywords(self, content: str, title: str) -> List[str]:
        """Extract relevant keywords from content"""
        
        # Simple keyword extraction (in production, would use more sophisticated NLP)
        text = f"{title} {content}".lower()
        
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        stop_words = {"this", "that", "with", "have", "will", "from", "they", "know", "want", "been", "good", "much", "some", "time"}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        return list(set(keywords))[:20]  # Limit to 20 unique keywords
    
    def _assess_difficulty_level(self, content: str) -> int:
        """Assess difficulty level of content (1-5 scale)"""
        
        advanced_indicators = [
            "comprehensive", "advanced", "complex", "sophisticated", "intricate",
            "pathophysiology", "pharmacokinetics", "contraindications"
        ]
        
        basic_indicators = [
            "basic", "simple", "introduction", "overview", "beginner", "getting started"
        ]
        
        content_lower = content.lower()
        
        advanced_count = sum(1 for indicator in advanced_indicators if indicator in content_lower)
        basic_count = sum(1 for indicator in basic_indicators if indicator in content_lower)
        
        if advanced_count >= 3:
            return 5
        elif advanced_count >= 1:
            return 4
        elif basic_count >= 2:
            return 1
        else:
            return 3  # Default medium difficulty
    
    def _extract_regulatory_references(self, content: str) -> List[str]:
        """Extract regulatory references from content"""
        
        references = []
        
        # Common healthcare regulations
        regulations = {
            "HIPAA": r'\bHIPAA\b',
            "FDA": r'\bFDA\b',
            "CMS": r'\bCMS\b',
            "Joint Commission": r'\bJoint Commission\b',
            "OSHA": r'\bOSHA\b',
            "CLIA": r'\bCLIA\b'
        }
        
        for regulation, pattern in regulations.items():
            if re.search(pattern, content, re.IGNORECASE):
                references.append(regulation)
        
        return references
    
    def _calculate_average_rating(self) -> float:
        """Calculate average rating across all feedback"""
        
        all_ratings = []
        for feedback_list in self.user_feedback.values():
            all_ratings.extend([f.rating for f in feedback_list])
        
        return sum(all_ratings) / len(all_ratings) if all_ratings else 0.0
    
    def _calculate_feedback_rate(self) -> float:
        """Calculate feedback rate (feedback/views ratio)"""
        
        total_views = sum(content.view_count for content in self.content_library.values())
        total_feedback = sum(len(feedback_list) for feedback_list in self.user_feedback.values())
        
        return (total_feedback / total_views * 100) if total_views > 0 else 0.0
    
    def _get_popular_search_terms(self) -> List[str]:
        """Get most popular search terms (placeholder)"""
        # In production, this would track actual search queries
        return [
            "patient data security",
            "ehr troubleshooting", 
            "medical device connection",
            "cardiac arrest protocol",
            "hipaa compliance"
        ]

# Global knowledge base system instance
knowledge_base = KnowledgeBaseSystem()

# Example usage and testing functions
async def setup_sample_knowledge_base():
    """Set up and test the knowledge base system"""
    
    # Search for emergency content
    emergency_results = await knowledge_base.search_content(
        query="cardiac arrest emergency",
        user_role=UserRole.PHYSICIAN,
        medical_specialty=MedicalSpecialty.EMERGENCY_MEDICINE
    )
    
    print(f"Found {len(emergency_results)} emergency-related results")
    
    # Get content by category
    clinical_content = await knowledge_base.get_content_by_category("emergency_procedures")
    print(f"Found {len(clinical_content)} clinical procedure documents")
    
    # Submit feedback
    if emergency_results:
        content_id = emergency_results[0].content_id
        await knowledge_base.submit_feedback(
            content_id=content_id,
            user_id="dr_smith_001",
            user_name="Dr. Sarah Smith",
            user_facility="General Hospital",
            user_role=UserRole.PHYSICIAN,
            rating=5,
            helpful=True,
            comment="Very helpful protocol guide"
        )
    
    # Get analytics
    analytics = await knowledge_base.get_content_analytics(30)
    print(f"Knowledge base analytics: {analytics['content_summary']}")

if __name__ == "__main__":
    asyncio.run(setup_sample_knowledge_base())