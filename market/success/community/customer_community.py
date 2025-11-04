"""
Customer Community and Peer Networking for Medical Professionals
Community platform for healthcare AI customers to share knowledge, best practices, and collaborate
"""

import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json

class CommunityContentType(Enum):
    DISCUSSION = "discussion"
    BEST_PRACTICE = "best_practice"
    CASE_STUDY = "case_study"
    CLINICAL_INSIGHT = "clinical_insight"
    SUCCESS_STORY = "success_story"
    QUESTION = "question"
    ANNOUNCEMENT = "announcement"
    RESEARCH_FINDING = "research_finding"

class ContentVisibility(Enum):
    PUBLIC = "public"
    COMMUNITY_ONLY = "community_only"
    TIER_BASED = "tier_based"  # Only certain customer tiers
    PRIVATE = "private"  # Only author and tagged users

class NetworkingEventType(Enum):
    WEBINAR = "webinar"
    VIRTUAL_MEETUP = "virtual_meetup"
    CONFERENCE_SESSION = "conference_session"
    ROUNDTABLE = "roundtable"
    TRAINING_SESSION = "training_session"
    CLINICAL_GRAND_ROUNDS = "clinical_grand_rounds"

class MemberRole(Enum):
    ADMIN = "admin"
    MODERATOR = "moderator"
    EXPERT = "expert"
    MEMBER = "member"
    OBSERVER = "observer"

@dataclass
class CommunityMember:
    """Community member profile"""
    member_id: str
    customer_id: str
    name: str
    title: str
    organization: str
    specialty: str
    role: MemberRole
    join_date: datetime.date
    reputation_score: float = 0.0
    contribution_count: int = 0
    expertise_areas: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    is_active: bool = True
    last_active: Optional[datetime.datetime] = None

@dataclass
class CommunityContent:
    """Community content post"""
    content_id: str
    author_id: str
    title: str
    content_type: CommunityContentType
    description: str
    content_body: str
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    visibility: ContentVisibility = ContentVisibility.COMMUNITY_ONLY
    target_tiers: List[str] = field(default_factory=list)  # For tier-based content
    
    # Engagement
    view_count: int = 0
    like_count: int = 0
    comment_count: int = 0
    share_count: int = 0
    
    # Clinical Context
    clinical_specialty: str = ""
    patient_population: str = ""
    relevant_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    is_featured: bool = False
    is_verified: bool = False
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_updated: datetime.datetime = field(default_factory=datetime.datetime.now)

@dataclass
class NetworkingEvent:
    """Networking event for healthcare professionals"""
    event_id: str
    title: str
    description: str
    event_type: NetworkingEventType
    
    # Event Details
    start_datetime: datetime.datetime
    end_datetime: datetime.datetime
    timezone: str
    location: str = "Virtual"
    
    # Registration
    max_attendees: int = 100
    current_registrations: int = 0
    registration_required: bool = True
    registration_deadline: datetime.datetime
    
    # Content
    agenda: List[str] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    
    # Clinical Focus
    clinical_topics: List[str] = field(default_factory=list)
    target_specialties: List[str] = field(default_factory=list)
    
    # Engagement
    event_url: str = ""
    recording_url: str = ""
    resources: List[str] = field(default_factory=list)

@dataclass
class MentorshipPair:
    """Mentorship pairing in community"""
    pair_id: str
    mentor_id: str
    mentee_id: str
    specialty_focus: str
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    
    # Meeting Details
    meeting_frequency: str = "monthly"  # weekly, biweekly, monthly
    meeting_format: str = "virtual"  # virtual, in-person, hybrid
    
    # Goals and Progress
    mentorship_goals: List[str] = field(default_factory=list)
    sessions_completed: int = 0
    progress_notes: List[str] = field(default_factory=list)
    status: str = "active"  # active, completed, paused, cancelled

@dataclass
class PeerLearningGroup:
    """Peer learning group for specific clinical areas"""
    group_id: str
    name: str
    description: str
    specialty_focus: str
    
    # Group Details
    max_members: int = 20
    current_members: int = 0
    is_private: bool = False
    meeting_schedule: str = ""  # "First Tuesday of each month"
    
    # Learning Objectives
    learning_goals: List[str] = field(default_factory=list)
    current_topics: List[str] = field(default_factory=list)
    resources_shared: List[str] = field(default_factory=list)
    
    # Activity Tracking
    meetings_held: int = 0
    members_engaged: int = 0
    topics_discussed: int = 0

class CommunityManager:
    """Customer community and peer networking management system"""
    
    def __init__(self):
        self.members: Dict[str, CommunityMember] = {}
        self.content: Dict[str, CommunityContent] = {}
        self.events: Dict[str, NetworkingEvent] = {}
        self.mentorships: Dict[str, MentorshipPair] = {}
        self.learning_groups: Dict[str, PeerLearningGroup] = {}
        self.member_engagement: Dict[str, Dict] = {}
        
        # Initialize default learning groups
        self._initialize_learning_groups()
        
        # Community guidelines and best practices
        self.community_guidelines = self._initialize_community_guidelines()
    
    def _initialize_learning_groups(self):
        """Initialize default peer learning groups"""
        
        # Cardiology Learning Group
        cardiology_group = PeerLearningGroup(
            group_id="group_cardiology",
            name="AI in Cardiology Excellence",
            description="Peer learning group focused on AI applications in cardiovascular care",
            specialty_focus="cardiology",
            meeting_schedule="Second Wednesday of each month",
            learning_goals=[
                "Share AI-driven diagnostic approaches in cardiology",
                "Discuss treatment optimization using predictive analytics",
                "Review cardiovascular outcome improvements",
                "Explore emerging AI technologies in heart care"
            ]
        )
        self.learning_groups[cardiology_group.group_id] = cardiology_group
        
        # Emergency Medicine Group
        emergency_group = PeerLearningGroup(
            group_id="group_emergency",
            name="Emergency Department AI Innovation",
            description="Collaborative learning for emergency medicine AI implementations",
            specialty_focus="emergency_medicine",
            meeting_schedule="First Friday of each month",
            learning_goals=[
                "Optimize triage processes with AI assistance",
                "Improve patient flow and resource allocation",
                "Enhance diagnostic accuracy in high-pressure environments",
                "Share emergency department efficiency strategies"
            ]
        )
        self.learning_groups[emergency_group.group_id] = emergency_group
        
        # Oncology Group
        oncology_group = PeerLearningGroup(
            group_id="group_oncology",
            name="AI-Driven Cancer Care",
            description="Oncology-focused AI implementation and best practices",
            specialty_focus="oncology",
            meeting_schedule="Third Thursday of each month",
            learning_goals=[
                "Enhance treatment planning with AI insights",
                "Improve patient monitoring and follow-up",
                "Optimize care coordination across multidisciplinary teams",
                "Explore precision medicine applications"
            ]
        )
        self.learning_groups[oncology_group.group_id] = oncology_group
        
        # Quality & Safety Group
        quality_group = PeerLearningGroup(
            group_id="group_quality",
            name="Healthcare Quality & Safety AI",
            description="Quality improvement and patient safety through AI",
            specialty_focus="quality_safety",
            meeting_schedule="Second Tuesday of each month",
            learning_goals=[
                "Reduce medical errors through AI assistance",
                "Improve patient safety metrics",
                "Enhance compliance and regulatory adherence",
                "Share quality improvement success stories"
            ]
        )
        self.learning_groups[quality_group.group_id] = quality_group
    
    def _initialize_community_guidelines(self) -> Dict[str, List[str]]:
        """Initialize community guidelines and best practices"""
        return {
            "posting_guidelines": [
                "Share specific, actionable insights from your experience",
                "Include relevant clinical context and patient population details",
                "Use appropriate medical terminology while maintaining clarity",
                "Cite sources and provide supporting data when possible",
                "Respect patient privacy and HIPAA compliance",
                "Be constructive and supportive in all interactions"
            ],
            "networking_etiquette": [
                "Introduce yourself when joining new conversations",
                "Ask thoughtful questions to encourage discussion",
                "Offer to connect members with similar interests",
                "Share relevant resources and knowledge generously",
                "Respect different perspectives and clinical approaches",
                "Maintain professional communication standards"
            ],
            "content_quality": [
                "Provide detailed case studies with measurable outcomes",
                "Include before/after comparisons when relevant",
                "Share lessons learned and implementation challenges",
                "Use data to support your recommendations",
                "Highlight innovative approaches and novel solutions",
                "Focus on patient-centered improvements"
            ]
        }
    
    def join_community(self, customer_id: str, name: str, title: str, organization: str,
                      specialty: str, expertise_areas: List[str]) -> CommunityMember:
        """Add a new member to the community"""
        
        member = CommunityMember(
            member_id=f"member_{customer_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            customer_id=customer_id,
            name=name,
            title=title,
            organization=organization,
            specialty=specialty,
            role=MemberRole.MEMBER,
            join_date=datetime.date.today(),
            expertise_areas=expertise_areas,
            last_active=datetime.datetime.now()
        )
        
        self.members[member.member_id] = member
        
        # Initialize engagement tracking
        self.member_engagement[member.member_id] = {
            "posts_created": 0,
            "comments_made": 0,
            "events_attended": 0,
            "connections_made": 0,
            "help_provided": 0,
            "total_engagement_score": 0.0
        }
        
        return member
    
    def create_content_post(self, author_id: str, title: str, content_type: CommunityContentType,
                          description: str, content_body: str, tags: List[str],
                          clinical_specialty: str = "", visibility: ContentVisibility = ContentVisibility.COMMUNITY_ONLY) -> CommunityContent:
        """Create new community content post"""
        
        content = CommunityContent(
            content_id=f"content_{author_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            author_id=author_id,
            title=title,
            content_type=content_type,
            description=description,
            content_body=content_body,
            tags=tags,
            visibility=visibility,
            clinical_specialty=clinical_specialty
        )
        
        self.content[content.content_id] = content
        
        # Update author engagement
        if author_id in self.member_engagement:
            self.member_engagement[author_id]["posts_created"] += 1
            self._update_reputation_score(author_id)
        
        return content
    
    def schedule_networking_event(self, title: str, description: str, event_type: NetworkingEventType,
                                start_datetime: datetime.datetime, end_datetime: datetime.datetime,
                                speakers: List[str], clinical_topics: List[str]) -> NetworkingEvent:
        """Schedule a new networking event"""
        
        event = NetworkingEvent(
            event_id=f"event_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=title,
            description=description,
            event_type=event_type,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            registration_deadline=start_datetime - datetime.timedelta(days=7),
            speakers=speakers,
            clinical_topics=clinical_topics,
            target_specialties=self._extract_specialties_from_topics(clinical_topics)
        )
        
        self.events[event.event_id] = event
        
        return event
    
    def _extract_specialties_from_topics(self, topics: List[str]) -> List[str]:
        """Extract relevant medical specialties from event topics"""
        specialty_keywords = {
            "cardiology": ["heart", "cardiac", "cardiovascular", "coronary"],
            "oncology": ["cancer", "tumor", "oncology", "chemotherapy"],
            "emergency_medicine": ["emergency", "trauma", "urgent", "critical"],
            "radiology": ["imaging", "radiology", "scan", "diagnostic"],
            "pathology": ["pathology", "laboratory", "diagnostic"],
            "surgery": ["surgical", "surgery", "operative"],
            "internal_medicine": ["internal", "general", "primary"],
            "pediatrics": ["pediatric", "children", "child"],
            "psychiatry": ["mental", "psychiatric", "behavioral"],
            "neurology": ["neurological", "brain", "spinal"]
        }
        
        relevant_specialties = []
        topics_text = " ".join(topics).lower()
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in topics_text for keyword in keywords):
                relevant_specialties.append(specialty)
        
        return relevant_specialties
    
    def create_mentorship_pair(self, mentor_id: str, mentee_id: str, specialty_focus: str,
                             goals: List[str], frequency: str = "monthly") -> MentorshipPair:
        """Create a new mentorship pairing"""
        
        mentorship = MentorshipPair(
            pair_id=f"mentorship_{mentor_id}_{mentee_id}_{datetime.datetime.now().strftime('%Y%m%d')}",
            mentor_id=mentor_id,
            mentee_id=mentee_id,
            specialty_focus=specialty_focus,
            start_date=datetime.date.today(),
            meeting_frequency=frequency,
            mentorship_goals=goals
        )
        
        self.mentorships[mentorship.pair_id] = mentorship
        
        return mentorship
    
    def join_learning_group(self, member_id: str, group_id: str) -> bool:
        """Add member to a learning group"""
        
        if group_id not in self.learning_groups or member_id not in self.members:
            return False
        
        group = self.learning_groups[group_id]
        
        # Check if group has space
        if group.current_members >= group.max_members:
            return False
        
        # Check if member's specialty matches group focus
        member = self.members[member_id]
        if group.specialty_focus not in member.specialty.lower() and group.specialty_focus not in member.expertise_areas:
            # Still allow joining but with note
            pass
        
        group.current_members += 1
        group.members_engaged += 1
        
        # Update member engagement
        if member_id in self.member_engagement:
            self.member_engagement[member_id]["connections_made"] += 1
        
        return True
    
    def get_recommended_connections(self, member_id: str, limit: int = 10) -> List[CommunityMember]:
        """Get recommended connections based on specialty, expertise, and activity"""
        
        if member_id not in self.members:
            return []
        
        current_member = self.members[member_id]
        recommendations = []
        
        # Find members with similar specialties or expertise
        for member in self.members.values():
            if member.member_id == member_id or not member.is_active:
                continue
            
            similarity_score = 0
            
            # Specialty match
            if member.specialty.lower() in current_member.specialty.lower() or \
               current_member.specialty.lower() in member.specialty.lower():
                similarity_score += 30
            
            # Expertise overlap
            expertise_overlap = len(set(member.expertise_areas) & set(current_member.expertise_areas))
            similarity_score += expertise_overlap * 10
            
            # Reputation bonus
            if member.reputation_score > 50:
                similarity_score += 20
            
            # Activity bonus
            if member.last_active and (datetime.datetime.now() - member.last_active).days < 30:
                similarity_score += 10
            
            recommendations.append((member, similarity_score))
        
        # Sort by similarity score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [member for member, score in recommendations[:limit]]
    
    def get_content_feed(self, member_id: str, content_types: List[CommunityContentType] = None,
                        specialty_filter: str = "", limit: int = 20) -> List[CommunityContent]:
        """Get personalized content feed for a member"""
        
        if member_id not in self.members:
            return []
        
        member = self.members[member_id]
        feed_content = []
        
        for content in self.content.values():
            # Filter by content type
            if content_types and content.content_type not in content_types:
                continue
            
            # Filter by specialty
            if specialty_filter and content.clinical_specialty != specialty_filter:
                continue
            
            # Check visibility permissions
            if not self._can_member_view_content(member, content):
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_content_relevance(member, content)
            feed_content.append((content, relevance_score))
        
        # Sort by relevance and recency
        feed_content.sort(key=lambda x: (x[1], x[0].created_at), reverse=True)
        
        return [content for content, score in feed_content[:limit]]
    
    def _can_member_view_content(self, member: CommunityMember, content: CommunityContent) -> bool:
        """Check if member can view specific content"""
        
        # Public content is visible to all
        if content.visibility == ContentVisibility.PUBLIC:
            return True
        
        # Community-only content
        if content.visibility == ContentVisibility.COMMUNITY_ONLY:
            return member.role != MemberRole.OBSERVER
        
        # Tier-based content
        if content.visibility == ContentVisibility.TIER_BASED:
            # Would check member's customer tier against target_tiers
            # For now, assume all members can view tier-based content
            return True
        
        # Private content (author only)
        if content.visibility == ContentVisibility.PRIVATE:
            return content.author_id == member.member_id
        
        return True
    
    def _calculate_content_relevance(self, member: CommunityMember, content: CommunityContent) -> float:
        """Calculate relevance score of content for a member"""
        score = 0.0
        
        # Specialty match
        if content.clinical_specialty and member.specialty.lower() in content.clinical_specialty.lower():
            score += 20
        
        # Expertise area match
        for expertise in member.expertise_areas:
            if expertise.lower() in content.content_body.lower():
                score += 10
        
        # Tag match
        for tag in content.tags:
            if tag.lower() in [e.lower() for e in member.expertise_areas]:
                score += 5
        
        # Author expertise similarity
        if content.author_id in self.members:
            author = self.members[content.author_id]
            if author.specialty == member.specialty:
                score += 15
            elif set(author.expertise_areas) & set(member.expertise_areas):
                score += 8
        
        # Content popularity bonus
        total_engagement = content.like_count + content.comment_count + content.share_count
        score += min(total_engagement * 0.5, 10)  # Max 10 points from engagement
        
        # Featured content bonus
        if content.is_featured:
            score += 20
        
        return score
    
    def _update_reputation_score(self, member_id: str):
        """Update member's reputation score based on contributions"""
        if member_id not in self.member_engagement:
            return
        
        engagement = self.member_engagement[member_id]
        
        # Calculate reputation based on various factors
        base_score = 0
        base_score += engagement["posts_created"] * 10  # 10 points per post
        base_score += engagement["comments_made"] * 2   # 2 points per comment
        base_score += engagement["events_attended"] * 5  # 5 points per event
        base_score += engagement["help_provided"] * 15   # 15 points per help
        base_score += engagement["connections_made"] * 3 # 3 points per connection
        
        # Bonus for high-quality contributions (would be calculated from content analysis)
        if member_id in self.members:
            member = self.members[member_id]
            # Add member reputation
            member.reputation_score = base_score
            
            # Update member status based on reputation
            if member.reputation_score >= 100:
                member.role = MemberRole.EXPERT
                if "Community Expert" not in member.achievements:
                    member.achievements.append("Community Expert")
            elif member.reputation_score >= 50:
                if MemberRole.EXPERT not in [MemberRole.EXPERT]:
                    member.role = MemberRole.MODERATOR
        
        # Update engagement total
        self.member_engagement[member_id]["total_engagement_score"] = base_score
    
    def record_event_attendance(self, member_id: str, event_id: str) -> bool:
        """Record that a member attended an event"""
        
        if member_id not in self.members or event_id not in self.events:
            return False
        
        # Update event registration count
        if self.events[event_id].current_registrations < self.events[event_id].max_attendees:
            self.events[event_id].current_registrations += 1
        
        # Update member engagement
        if member_id in self.member_engagement:
            self.member_engagement[member_id]["events_attended"] += 1
            self._update_reputation_score(member_id)
        
        # Update member last active
        self.members[member_id].last_active = datetime.datetime.now()
        
        return True
    
    def generate_community_dashboard(self) -> Dict:
        """Generate comprehensive community dashboard"""
        
        total_members = len(self.members)
        active_members = len([m for m in self.members.values() if m.is_active])
        
        # Content statistics
        content_by_type = {}
        for content in self.content.values():
            content_type = content.content_type.value
            if content_type not in content_by_type:
                content_by_type[content_type] = 0
            content_by_type[content_type] += 1
        
        # Upcoming events
        upcoming_events = [
            event for event in self.events.values()
            if event.start_datetime > datetime.datetime.now()
        ]
        
        # Top contributors
        top_contributors = sorted(
            self.members.values(),
            key=lambda m: m.reputation_score,
            reverse=True
        )[:10]
        
        # Learning group activity
        learning_group_stats = []
        for group in self.learning_groups.values():
            learning_group_stats.append({
                "group_name": group.name,
                "specialty": group.specialty_focus,
                "current_members": group.current_members,
                "meetings_held": group.meetings_held,
                "engagement_level": group.members_engaged / group.current_members if group.current_members > 0 else 0
            })
        
        # Mentorship program stats
        active_mentorships = len([m for m in self.mentorships.values() if m.status == "active"])
        completed_mentorships = len([m for m in self.mentorships.values() if m.status == "completed"])
        
        return {
            "overview": {
                "total_members": total_members,
                "active_members": active_members,
                "total_posts": len(self.content),
                "upcoming_events": len(upcoming_events),
                "active_learning_groups": len(self.learning_groups),
                "active_mentorships": active_mentorships
            },
            "member_demographics": {
                "specialty_breakdown": self._get_member_specialty_breakdown(),
                "role_distribution": self._get_role_distribution(),
                "geographic_distribution": "Would track based on organization locations"
            },
            "content_engagement": {
                "content_by_type": content_by_type,
                "total_engagement": sum(
                    c.like_count + c.comment_count + c.share_count 
                    for c in self.content.values()
                ),
                "most_engaging_content": self._get_most_engaging_content()
            },
            "upcoming_events": [
                {
                    "title": event.title,
                    "event_type": event.event_type.value,
                    "start_date": event.start_datetime,
                    "current_registrations": event.current_registrations,
                    "max_attendees": event.max_attendees
                }
                for event in sorted(upcoming_events, key=lambda x: x.start_datetime)[:5]
            ],
            "top_contributors": [
                {
                    "name": member.name,
                    "title": member.title,
                    "organization": member.organization,
                    "specialty": member.specialty,
                    "reputation_score": member.reputation_score,
                    "achievements": member.achievements
                }
                for member in top_contributors
            ],
            "learning_groups": learning_group_stats,
            "mentorship_program": {
                "active_pairs": active_mentorships,
                "completed_pairs": completed_mentorships,
                "average_duration_months": self._calculate_average_mentorship_duration()
            },
            "community_health": {
                "engagement_rate": active_members / total_members if total_members > 0 else 0,
                "content_creation_rate": len(self.content) / total_members if total_members > 0 else 0,
                "peer_help_score": sum(
                    self.member_engagement.get(m.member_id, {}).get("help_provided", 0)
                    for m in self.members.values()
                ) / total_members if total_members > 0 else 0
            }
        }
    
    def _get_member_specialty_breakdown(self) -> Dict[str, int]:
        """Get breakdown of members by medical specialty"""
        specialty_count = {}
        for member in self.members.values():
            specialty = member.specialty
            if specialty not in specialty_count:
                specialty_count[specialty] = 0
            specialty_count[specialty] += 1
        return specialty_count
    
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of member roles"""
        role_count = {}
        for member in self.members.values():
            role = member.role.value
            if role not in role_count:
                role_count[role] = 0
            role_count[role] += 1
        return role_count
    
    def _get_most_engaging_content(self) -> List[Dict]:
        """Get most engaging content posts"""
        content_with_engagement = []
        for content in self.content.values():
            total_engagement = content.like_count + content.comment_count + content.share_count
            content_with_engagement.append((content, total_engagement))
        
        # Sort by engagement and return top 5
        content_with_engagement.sort(key=lambda x: x[1], reverse=True)
        
        return [
            {
                "title": content.title,
                "author": self.members[content.author_id].name if content.author_id in self.members else "Unknown",
                "content_type": content.content_type.value,
                "engagement_score": engagement,
                "created_date": content.created_at
            }
            for content, engagement in content_with_engagement[:5]
        ]
    
    def _calculate_average_mentorship_duration(self) -> float:
        """Calculate average mentorship duration in months"""
        completed_mentorships = [m for m in self.mentorships.values() if m.status == "completed"]
        
        if not completed_mentorships:
            return 0.0
        
        total_duration = 0
        for mentorship in completed_mentorships:
            if mentorship.end_date:
                duration = (mentorship.end_date - mentorship.start_date).days / 30.44  # Average days per month
                total_duration += duration
        
        return total_duration / len(completed_mentorships)