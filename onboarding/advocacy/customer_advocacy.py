"""
Customer Advocacy and Reference Programs
Healthcare user testimonials, case studies, and reference programs
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class AdvocacyType(Enum):
    """Types of customer advocacy activities"""
    TESTIMONIAL = "testimonial"
    CASE_STUDY = "case_study"
    REFERENCE_CUSTOMER = "reference_customer"
    SPEAKING_ENGAGEMENT = "speaking_engagement"
    INDUSTRY_AWARD = "industry_award"
    PEER_REVIEW = "peer_review"
    USER_GROUP_LEADERSHIP = "user_group_leadership"
    PRODUCT_FEEDBACK = "product_feedback"

class CaseStudyStatus(Enum):
    """Case study development status"""
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    LEGAL_REVIEW = "legal_review"
    APPROVED = "approved"
    PUBLISHED = "published"

class ReferenceTier(Enum):
    """Customer reference tiers"""
    STRATEGIC = "strategic"      # Large deployments, high visibility
    DEVELOPMENTAL = "developmental"  # Early adopters, innovation focus
    STANDARD = "standard"        # Successful implementations
    BEGINNER = "beginner"        # New customers, learning phase

@dataclass
class CustomerAdvocate:
    """Customer advocate profile"""
    advocate_id: str
    name: str
    title: str
    organization_id: str
    organization_name: str
    healthcare_sector: str
    expertise_areas: List[str]
    communication_skills: List[str]
    availability_status: str
    advocacy_history: List[Dict[str, Any]]
    reference_rating: float  # 0-5 scale
    speaking_experience: bool
    case_study_participation: bool
    preferred_activities: List[AdvocacyType]

@dataclass
class SuccessStory:
    """Customer success story"""
    story_id: str
    organization_id: str
    organization_name: str
    implementation_scope: str
    challenges_addressed: List[str]
    solutions_implemented: List[str]
    measurable_outcomes: Dict[str, Any]
    clinical_impact: Dict[str, Any]
    financial_impact: Dict[str, Any]
    timeline_to_value: str
    key_stakeholders: List[str]
    story_narrative: str
    media_assets: List[str]
    approval_status: CaseStudyStatus
    publish_readiness: bool
    target_audiences: List[str]

class CustomerAdvocacyManager:
    """Customer advocacy and reference management system"""
    
    def __init__(self):
        self.advocacy_templates = self._initialize_advocacy_templates()
        self.case_study_framework = self._initialize_case_study_framework()
        self.reference_program_structure = self._initialize_reference_program_structure()
        self.success_metrics = self._initialize_success_metrics()
        self.media_production_workflow = self._initialize_media_production_workflow()
    
    def _initialize_advocacy_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize advocacy content templates"""
        return {
            "clinical_efficiency_case_study": {
                "template_name": "Clinical Efficiency Transformation Case Study",
                "sections": [
                    "Executive Summary",
                    "Organization Profile",
                    "Initial Challenge Description",
                    "Solution Implementation",
                    "Clinical Workflow Transformation",
                    "Quantified Outcomes",
                    "Clinical Impact Assessment",
                    "ROI Analysis",
                    "Lessons Learned",
                    "Future Roadmap"
                ],
                "key_metrics": [
                    "Clinical time savings per day",
                    "Documentation efficiency improvement",
                    "Diagnostic accuracy improvement",
                    "Clinical decision speed increase",
                    "Patient satisfaction scores",
                    "Provider satisfaction scores"
                ],
                "narrative_structure": "Problem-Solution-Impact-Story",
                "target_length": "8-12 pages",
                "media_requirements": [
                    "Organization logo and branding",
                    "Leadership photos",
                    "Clinical workflow screenshots",
                    "Performance dashboard images",
                    "Patient testimonial video (optional)"
                ]
            },
            
            "implementation_success_story": {
                "template_name": "Implementation Success Story",
                "sections": [
                    "Quick Win Overview",
                    "Organization Profile",
                    "Implementation Timeline",
                    "Training and Adoption",
                    "Early Results",
                    "Best Practices Learned",
                    "Success Factors"
                ],
                "key_metrics": [
                    "Time to first value",
                    "User adoption rate",
                    "Training completion rate",
                    "Implementation satisfaction"
                ],
                "narrative_structure": "Timeline-Implementation-Results",
                "target_length": "4-6 pages",
                "media_requirements": [
                    "Implementation timeline graphic",
                    "Training completion charts",
                    "User testimonial quotes"
                ]
            },
            
            "clinical_innovation_story": {
                "template_name": "Clinical Innovation and AI Adoption Story",
                "sections": [
                    "Innovation Overview",
                    "Clinical Problem Statement",
                    "AI Solution Approach",
                    "Innovation Process",
                    "Clinical Validation",
                    "Broader Impact",
                    "Future Innovation Vision"
                ],
                "key_metrics": [
                    "Innovation adoption rate",
                    "Clinical outcome improvements",
                    "Research publications",
                    "Industry recognition"
                ],
                "narrative_structure": "Innovation-Challenge-Solution-Impact",
                "target_length": "6-10 pages",
                "media_requirements": [
                    "Innovation process diagrams",
                    "Clinical validation data",
                    "Innovation team photos",
                    "Research presentation slides"
                ]
            },
            
            "patient_experience_story": {
                "template_name": "Patient Experience Enhancement Story",
                "sections": [
                    "Patient Experience Overview",
                    "Patient Journey Analysis",
                    "AI-Enhanced Experience Design",
                    "Implementation Approach",
                    "Patient Feedback Results",
                    "Clinical Staff Impact",
                    "Experience Transformation Metrics"
                ],
                "key_metrics": [
                    "Patient satisfaction improvement",
                    "Wait time reduction",
                    "Communication quality scores",
                    "Patient safety indicators"
                ],
                "narrative_structure": "Experience-Transformation-Results",
                "target_length": "5-8 pages",
                "media_requirements": [
                    "Patient journey maps",
                    "Satisfaction survey results",
                    "Patient feedback videos (with consent)",
                    "Clinical staff testimonials"
                ]
            }
        }
    
    def _initialize_case_study_framework(self) -> Dict[str, Any]:
        """Initialize comprehensive case study framework"""
        return {
            "development_process": {
                "phase_1": {
                    "name": "Initial Assessment and Planning",
                    "activities": [
                        "Identify potential success story candidates",
                        "Conduct initial customer interviews",
                        "Assess story potential and impact",
                        "Develop story outline and approach",
                        "Secure stakeholder buy-in"
                    ],
                    "deliverables": [
                        "Story concept document",
                        "Interview schedule",
                        "Success metrics identification",
                        "Stakeholder approval"
                    ],
                    "timeline": "1-2 weeks"
                },
                
                "phase_2": {
                    "name": "Data Collection and Content Development",
                    "activities": [
                        "Conduct detailed customer interviews",
                        "Collect quantitative performance data",
                        "Gather clinical outcome metrics",
                        "Document implementation journey",
                        "Collect media assets"
                    ],
                    "deliverables": [
                        "Interview transcripts",
                        "Performance data compilation",
                        "Implementation timeline",
                        "Initial content draft"
                    ],
                    "timeline": "2-3 weeks"
                },
                
                "phase_3": {
                    "name": "Content Creation and Review",
                    "activities": [
                        "Create comprehensive case study document",
                        "Develop supporting visual content",
                        "Internal review and feedback",
                        "Customer review and approval",
                        "Legal and compliance review"
                    ],
                    "deliverables": [
                        "Complete case study document",
                        "Visual content package",
                        "Review feedback compilation",
                        "Legal approval documentation"
                    ],
                    "timeline": "2-3 weeks"
                },
                
                "phase_4": {
                    "name": "Publication and Distribution",
                    "activities": [
                        "Final formatting and design",
                        "Multi-format content creation",
                        "Website publication",
                        "Marketing material integration",
                        "Sales team distribution"
                    ],
                    "deliverables": [
                        "Published case study",
                        "Marketing materials",
                        "Sales enablement content",
                        "Distribution plan"
                    ],
                    "timeline": "1-2 weeks"
                }
            },
            
            "quality_standards": {
                "narrative_quality": [
                    "Clear problem statement",
                    "Compelling solution narrative",
                    "Quantifiable results",
                    "Authentic customer voice",
                    "Relevant clinical context"
                ],
                
                "data_validation": [
                    "All metrics verified by customer",
                    "Clinical outcomes validated",
                    "Financial data accuracy confirmed",
                    "Timeline accuracy verified",
                    "Compliance approval obtained"
                ],
                
                "legal_compliance": [
                    "HIPAA compliance verified",
                    "Customer approval documented",
                    "Legal review completed",
                    "Brand guidelines followed",
                    "Attribution permissions obtained"
                ]
            }
        }
    
    def _initialize_reference_program_structure(self) -> Dict[str, Dict[str, Any]]:
        """Initialize customer reference program structure"""
        return {
            ReferenceTier.STRATEGIC: {
                "criteria": [
                    "Large-scale implementation (1000+ users)",
                    "Multi-facility deployment",
                    "High visibility in industry",
                    "Strong clinical outcomes",
                    "Innovative use cases"
                ],
                "program_benefits": [
                    "Exclusive customer advisory board participation",
                    "Co-marketing opportunities",
                    "Early access to new features",
                    "Dedicated account management",
                    "Speaking opportunities at events"
                ],
                "commitment_requirements": [
                    "Quarterly reference calls",
                    "Annual case study development",
                    "Speaking engagement participation",
                    "Advisory board membership",
                    "Product feedback sessions"
                ],
                "recognition_program": [
                    "Customer of the Year awards",
                    "Industry recognition opportunities",
                    "Executive networking events",
                    "Strategic partnership discussions"
                ]
            },
            
            ReferenceTier.DEVELOPMENTAL: {
                "criteria": [
                    "Early adopter of new features",
                    "Innovation-focused implementation",
                    "Collaborative approach to development",
                    "Feedback-driven improvements",
                    "Pilot program participation"
                ],
                "program_benefits": [
                    "Beta feature access",
                    "Product roadmap influence",
                    "Direct development team access",
                    "Innovation workshop participation",
                    "Recognition in product releases"
                ],
                "commitment_requirements": [
                    "Regular feedback sessions",
                    "Beta testing participation",
                    "Innovation case studies",
                    "Product advisory input"
                ],
                "recognition_program": [
                    "Innovation Partner recognition",
                    "Beta testing certificates",
                    "Development contribution acknowledgment"
                ]
            },
            
            ReferenceTier.STANDARD: {
                "criteria": [
                    "Successful implementation",
                    "Good adoption rates",
                    "Positive customer satisfaction",
                    "Willingness to provide references",
                    "Stable production environment"
                ],
                "program_benefits": [
                    "Reference call participation",
                    "Case study opportunity",
                    "Customer success resources",
                    "Community forum participation",
                    "Training and certification programs"
                ],
                "commitment_requirements": [
                    "Reference call availability",
                    "Annual satisfaction survey",
                    "Best practice sharing",
                    "Success story participation"
                ],
                "recognition_program": [
                    "Customer success recognition",
                    "Community contributor acknowledgment"
                ]
            },
            
            ReferenceTier.BEGINNER: {
                "criteria": [
                    "Recent implementation (within 6 months)",
                    "Basic feature adoption",
                    "Initial training completed",
                    "Seeking optimization support",
                    "Building success foundation"
                ],
                "program_benefits": [
                    "Enhanced onboarding support",
                    "Best practice training",
                    "Success planning assistance",
                    "Community mentorship",
                    "Early success documentation"
                ],
                "commitment_requirements": [
                    "Success metric tracking",
                    "Regular check-in calls",
                    "Feedback on onboarding experience",
                    "Future reference willingness"
                ],
                "recognition_program": [
                    "New customer success certificates",
                    "Onboarding milestone recognition"
                ]
            }
        }
    
    def _initialize_success_metrics(self) -> Dict[str, Any]:
        """Initialize success metrics for advocacy programs"""
        return {
            "reference_program_metrics": {
                "reference_availability": {
                    "metric": "Percentage of customers available for references",
                    "target": "85%",
                    "measurement_frequency": "monthly",
                    "reporting_format": "percentage"
                },
                "reference_quality": {
                    "metric": "Average rating of reference experiences",
                    "target": "4.5/5.0",
                    "measurement_frequency": "quarterly",
                    "reporting_format": "rating_scale"
                },
                "reference_conversion": {
                    "metric": "Percentage of prospects requesting references who become customers",
                    "target": "75%",
                    "measurement_frequency": "monthly",
                    "reporting_format": "percentage"
                },
                "reference_program_growth": {
                    "metric": "Net new reference customers per quarter",
                    "target": "10 new customers",
                    "measurement_frequency": "quarterly",
                    "reporting_format": "count"
                }
            },
            
            "case_study_metrics": {
                "case_study_engagement": {
                    "metric": "Average time spent on case study content",
                    "target": "5+ minutes",
                    "measurement_frequency": "monthly",
                    "reporting_format": "time_duration"
                },
                "case_study_conversion": {
                    "metric": "Prospect conversion rate after viewing case studies",
                    "target": "15%",
                    "measurement_frequency": "monthly",
                    "reporting_format": "percentage"
                },
                "case_study_quality_score": {
                    "metric": "Internal and customer rating of case study quality",
                    "target": "4.0/5.0",
                    "measurement_frequency": "per_case_study",
                    "reporting_format": "rating_scale"
                },
                "case_study_roi": {
                    "metric": "Revenue attributed to case study content",
                    "target": "ROI > 300%",
                    "measurement_frequency": "quarterly",
                    "reporting_format": "percentage"
                }
            },
            
            "advocacy_engagement_metrics": {
                "customer_participation_rate": {
                    "metric": "Percentage of customers participating in advocacy activities",
                    "target": "25%",
                    "measurement_frequency": "quarterly",
                    "reporting_format": "percentage"
                },
                "advocacy_activity_diversity": {
                    "metric": "Number of different advocacy activities per customer",
                    "target": "2+ activities per customer",
                    "measurement_frequency": "quarterly",
                    "reporting_format": "average_count"
                },
                "customer_lifetime_value_impact": {
                    "metric": "CLV increase for participating customers",
                    "target": "30% increase",
                    "measurement_frequency": "annually",
                    "reporting_format": "percentage"
                },
                "advocacy_network_growth": {
                    "metric": "Growth in customer advocacy network",
                    "target": "20% growth annually",
                    "measurement_frequency": "annually",
                    "reporting_format": "percentage"
                }
            },
            
            "business_impact_metrics": {
                "sales_impact": {
                    "metric": "Revenue influenced by customer advocacy",
                    "target": "40% of new revenue",
                    "measurement_frequency": "monthly",
                    "reporting_format": "revenue_amount"
                },
                "marketing_effectiveness": {
                    "metric": "Lead quality improvement from advocacy content",
                    "target": "25% higher conversion rate",
                    "measurement_frequency": "monthly",
                    "reporting_format": "percentage"
                },
                "customer_retention": {
                    "metric": "Retention rate of advocacy program participants",
                    "target": "95%",
                    "measurement_frequency": "annually",
                    "reporting_format": "percentage"
                },
                "brand_reputation": {
                    "metric": "Brand reputation score improvement",
                    "target": "10% improvement",
                    "measurement_frequency": "annually",
                    "reporting_format": "percentage"
                }
            }
        }
    
    def _initialize_media_production_workflow(self) -> Dict[str, Any]:
        """Initialize media production workflow for advocacy content"""
        return {
            "content_production": {
                "written_content": {
                    "case_studies": {
                        "drafting_timeline": "5-7 business days",
                        "review_cycles": 3,
                        "approval_stakeholders": ["customer", "legal", "marketing", "executive"],
                        "format_options": ["PDF", "Web page", "Print-ready"],
                        "quality_checkpoints": ["Fact verification", "Brand compliance", "Legal review"]
                    },
                    
                    "testimonials": {
                        "collection_timeline": "2-3 business days",
                        "interview_format": "Structured questionnaire",
                        "approval_process": "Customer review and approval",
                        "format_options": ["Quote graphics", "Written testimonials", "Video snippets"],
                        "quality_checkpoints": ["Authenticity verification", "Compliance check"]
                    }
                },
                
                "video_production": {
                    "customer_testimonials": {
                        "pre_production": "3-5 business days",
                        "production_timeline": "1-2 days",
                        "post_production": "5-7 business days",
                        "deliverables": ["Master files", "Web-optimized versions", "Social media clips"],
                        "approval_stakeholders": ["customer", "legal", "marketing"]
                    },
                    
                    "case_study_videos": {
                        "pre_production": "5-7 business days",
                        "production_timeline": "2-3 days",
                        "post_production": "10-14 business days",
                        "deliverables": ["Full video", "Chapter segments", "Social media teasers"],
                        "approval_stakeholders": ["customer", "legal", "marketing", "executive"]
                    }
                },
                
                "visual_content": {
                    "infographics": {
                        "design_timeline": "3-5 business days",
                        "design_cycles": 2,
                        "deliverables": ["High-res images", "Print-ready files", "Social media variants"],
                        "style_requirements": ["Brand guidelines", "Healthcare-appropriate design", "Accessibility compliance"]
                    },
                    
                    "data_visualizations": {
                        "creation_timeline": "2-4 business days",
                        "data_verification": "Customer approval required",
                        "deliverables": ["Interactive charts", "Static images", "Animated GIFs"],
                        "accessibility": ["Alt text for images", "Screen reader compatibility"]
                    }
                }
            },
            
            "distribution_strategy": {
                "owned_channels": [
                    "Company website case study library",
                    "Customer portal success stories",
                    "Email newsletter features",
                    "Social media campaigns",
                    "Sales enablement materials"
                ],
                
                "earned_channels": [
                    "Industry publication features",
                    "Conference presentations",
                    "Webinar content",
                    "Press release distribution",
                    "Industry analyst briefings"
                ],
                
                "paid_channels": [
                    "LinkedIn advertising campaigns",
                    "Industry publication sponsorships",
                    "Conference speaking sponsorships",
                    "Digital advertising retargeting",
                    "Content syndication"
                ]
            }
        }
    
    def identify_advocacy_candidates(self, customer_profiles: List[Dict[str, Any]]) -> List[CustomerAdvocate]:
        """Identify potential customer advocates"""
        advocates = []
        
        for profile in customer_profiles:
            # Assess advocacy potential based on customer profile
            advocacy_score = self._calculate_advocacy_score(profile)
            
            if advocacy_score >= 70:  # Threshold for advocacy candidate
                advocate = self._create_advocate_profile(profile, advocacy_score)
                advocates.append(advocate)
        
        return advocates
    
    def _calculate_advocacy_score(self, customer_profile: Dict[str, Any]) -> float:
        """Calculate advocacy potential score for customer"""
        score = 0.0
        
        # Implementation success factors
        implementation_score = customer_profile.get("implementation_success_score", 0)
        score += implementation_score * 0.25
        
        # Customer satisfaction
        satisfaction_score = customer_profile.get("satisfaction_score", 0)
        score += satisfaction_score * 0.20
        
        # Organization size and visibility
        org_size = customer_profile.get("organization_size", "medium")
        size_scores = {"small": 60, "medium": 75, "large": 85, "enterprise": 95}
        score += size_scores.get(org_size, 75) * 0.15
        
        # Healthcare sector
        sector = customer_profile.get("healthcare_sector", "general")
        sector_scores = {
            "academic_medical_center": 90,
            "large_hospital_system": 85,
            "community_hospital": 75,
            "specialty_clinic": 70,
            "research_institution": 80
        }
        score += sector_scores.get(sector, 70) * 0.15
        
        # Innovation adoption
        innovation_score = customer_profile.get("innovation_adoption_score", 0)
        score += innovation_score * 0.15
        
        # Communication skills and willingness
        communication_score = customer_profile.get("communication_score", 0)
        score += communication_score * 0.10
        
        return min(score, 100.0)
    
    def _create_advocate_profile(self, customer_profile: Dict[str, Any], 
                               advocacy_score: float) -> CustomerAdvocate:
        """Create advocate profile from customer data"""
        # Determine reference tier based on advocacy score and profile
        if advocacy_score >= 90:
            tier = ReferenceTier.STRATEGIC
        elif advocacy_score >= 80:
            tier = ReferenceTier.DEVELOPMENTAL
        elif advocacy_score >= 70:
            tier = ReferenceTier.STANDARD
        else:
            tier = ReferenceTier.BEGINNER
        
        # Generate advocate ID
        org_id = customer_profile.get("organization_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d")
        advocate_id = f"ADV_{org_id}_{timestamp}"
        
        # Create advocate profile
        advocate = CustomerAdvocate(
            advocate_id=advocate_id,
            name=customer_profile.get("primary_contact_name", ""),
            title=customer_profile.get("primary_contact_title", ""),
            organization_id=customer_profile.get("organization_id", ""),
            organization_name=customer_profile.get("organization_name", ""),
            healthcare_sector=customer_profile.get("healthcare_sector", "general"),
            expertise_areas=customer_profile.get("clinical_specialties", []),
            communication_skills=self._assess_communication_skills(customer_profile),
            availability_status="available",
            advocacy_history=[],
            reference_rating=customer_profile.get("satisfaction_score", 4.0) / 20.0,  # Convert to 0-5 scale
            speaking_experience=customer_profile.get("speaking_experience", False),
            case_study_participation=customer_profile.get("case_study_willingness", False),
            preferred_activities=self._determine_preferred_activities(tier, customer_profile)
        )
        
        return advocate
    
    def _assess_communication_skills(self, customer_profile: Dict[str, Any]) -> List[str]:
        """Assess communication skills based on customer profile"""
        skills = []
        
        if customer_profile.get("executive_communication", False):
            skills.append("Executive Presentation")
        
        if customer_profile.get("technical_communication", False):
            skills.append("Technical Communication")
        
        if customer_profile.get("clinical_communication", False):
            skills.append("Clinical Communication")
        
        if customer_profile.get("public_speaking_experience", False):
            skills.append("Public Speaking")
        
        if customer_profile.get("written_communication", False):
            skills.append("Written Communication")
        
        return skills
    
    def _determine_preferred_activities(self, tier: ReferenceTier, 
                                      customer_profile: Dict[str, Any]) -> List[AdvocacyType]:
        """Determine preferred advocacy activities based on tier and profile"""
        activities = []
        
        # Base activities by tier
        if tier == ReferenceTier.STRATEGIC:
            activities = [
                AdvocacyType.SPEAKING_ENGAGEMENT,
                AdvocacyType.CASE_STUDY,
                AdvocacyType.INDUSTRY_AWARD,
                AdvocacyType.REFERENCE_CUSTOMER,
                AdvocacyType.USER_GROUP_LEADERSHIP
            ]
        elif tier == ReferenceTier.DEVELOPMENTAL:
            activities = [
                AdvocacyType.PRODUCT_FEEDBACK,
                AdvocacyType.CASE_STUDY,
                AdvocacyType.PEER_REVIEW,
                AdvocacyType.REFERENCE_CUSTOMER
            ]
        elif tier == ReferenceTier.STANDARD:
            activities = [
                AdvocacyType.TESTIMONIAL,
                AdvocacyType.REFERENCE_CUSTOMER,
                AdvocacyType.CASE_STUDY
            ]
        else:
            activities = [
                AdvocacyType.TESTIMONIAL,
                AdvocacyType.PRODUCT_FEEDBACK
            ]
        
        # Customize based on customer preferences
        if customer_profile.get("prefers_written_content", False):
            activities.insert(0, AdvocacyType.TESTIMONIAL)
        
        if customer_profile.get("prefers_speaking", False):
            activities.insert(0, AdvocacyType.SPEAKING_ENGAGEMENT)
        
        return activities
    
    def develop_success_story(self, customer_data: Dict[str, Any], 
                            success_type: str) -> SuccessStory:
        """Develop customer success story"""
        story_id = f"STORY_{customer_data.get('organization_id', 'unknown')}_{datetime.now().strftime('%Y%m%d')}"
        
        # Generate success story based on type and customer data
        story = SuccessStory(
            story_id=story_id,
            organization_id=customer_data.get("organization_id", ""),
            organization_name=customer_data.get("organization_name", ""),
            implementation_scope=self._define_implementation_scope(customer_data),
            challenges_addressed=self._identify_challenges(customer_data, success_type),
            solutions_implemented=self._define_solutions(customer_data, success_type),
            measurable_outcomes=self._extract_outcomes(customer_data),
            clinical_impact=self._assess_clinical_impact(customer_data),
            financial_impact=self._calculate_financial_impact(customer_data),
            timeline_to_value=self._determine_timeline_to_value(customer_data),
            key_stakeholders=self._identify_stakeholders(customer_data),
            story_narrative=self._create_story_narrative(customer_data, success_type),
            media_assets=self._identify_media_assets(customer_data),
            approval_status=CaseStudyStatus.DRAFT,
            publish_readiness=False,
            target_audiences=self._define_target_audiences(success_type)
        )
        
        return story
    
    def _define_implementation_scope(self, customer_data: Dict[str, Any]) -> str:
        """Define implementation scope for story"""
        scope_elements = []
        
        org_size = customer_data.get("organization_size", "medium")
        if org_size == "enterprise":
            scope_elements.append("Enterprise-wide deployment")
        elif org_size == "large":
            scope_elements.append("Multi-department implementation")
        else:
            scope_elements.append("Focused departmental deployment")
        
        user_count = customer_data.get("user_count", 0)
        scope_elements.append(f"{user_count} healthcare professionals")
        
        clinical_areas = customer_data.get("clinical_specialties", [])
        if clinical_areas:
            scope_elements.append(f"Specialties: {', '.join(clinical_areas[:3])}")
        
        return ", ".join(scope_elements)
    
    def _identify_challenges(self, customer_data: Dict[str, Any], success_type: str) -> List[str]:
        """Identify key challenges addressed"""
        challenges = []
        
        if success_type == "clinical_efficiency":
            challenges = [
                "Manual clinical documentation taking excessive time",
                "Inconsistent clinical decision-making processes",
                "Limited clinical workflow integration",
                "Difficulty tracking clinical quality metrics"
            ]
        elif success_type == "implementation_success":
            challenges = [
                "Complex healthcare system integration requirements",
                "Staff training and adoption across multiple departments",
                "Regulatory compliance and validation processes",
                "Change management across clinical teams"
            ]
        elif success_type == "patient_experience":
            challenges = [
                "Patient communication gaps and delays",
                "Inconsistent patient education delivery",
                "Limited patient engagement tools",
                "Difficulty measuring patient satisfaction"
            ]
        else:
            challenges = [
                "Healthcare workflow optimization needs",
                "Clinical decision support requirements",
                "Regulatory compliance challenges",
                "Cost and efficiency pressures"
            ]
        
        return challenges
    
    def _define_solutions(self, customer_data: Dict[str, Any], success_type: str) -> List[str]:
        """Define solutions implemented"""
        base_solutions = [
            "AI-powered clinical decision support system",
            "Integrated healthcare workflow automation",
            "Real-time clinical performance monitoring",
            "Comprehensive staff training and certification program"
        ]
        
        if success_type == "clinical_efficiency":
            base_solutions.extend([
                "Automated clinical documentation assistance",
                "Intelligent diagnostic workflow support",
                "Clinical quality improvement tracking"
            ])
        elif success_type == "implementation_success":
            base_solutions.extend([
                "Phased implementation approach",
                "Dedicated change management support",
                "Continuous optimization and refinement"
            ])
        elif success_type == "patient_experience":
            base_solutions.extend([
                "Patient communication enhancement tools",
                "Personalized patient education delivery",
                "Patient satisfaction monitoring and improvement"
            ])
        
        return base_solutions
    
    def _extract_outcomes(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract measurable outcomes from customer data"""
        return {
            "clinical_efficiency": {
                "documentation_time_reduction": customer_data.get("doc_time_reduction", "35%"),
                "clinical_decision_speed": customer_data.get("decision_speed_improvement", "40%"),
                "workflow_optimization": customer_data.get("workflow_improvement", "30%")
            },
            
            "financial_impact": {
                "cost_savings_annual": customer_data.get("annual_cost_savings", "$500,000"),
                "roi_percentage": customer_data.get("roi_percentage", "200%"),
                "time_to_roi": customer_data.get("time_to_roi", "12 months")
            },
            
            "quality_metrics": {
                "clinical_accuracy_improvement": customer_data.get("accuracy_improvement", "25%"),
                "patient_satisfaction_increase": customer_data.get("patient_satisfaction_increase", "20%"),
                "provider_satisfaction_increase": customer_data.get("provider_satisfaction_increase", "35%")
            },
            
            "adoption_metrics": {
                "user_adoption_rate": customer_data.get("adoption_rate", "85%"),
                "training_completion_rate": customer_data.get("training_completion", "98%"),
                "system_utilization": customer_data.get("utilization_rate", "90%")
            }
        }
    
    def _assess_clinical_impact(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess clinical impact of implementation"""
        return {
            "patient_care_quality": {
                "improvement_description": "Enhanced clinical decision-making leading to better patient outcomes",
                "measurable_metrics": [
                    "Reduced diagnostic errors",
                    "Faster clinical decision-making",
                    "Improved care coordination"
                ]
            },
            
            "clinical_workflow": {
                "improvement_description": "Streamlined clinical workflows reducing administrative burden",
                "measurable_metrics": [
                    "Reduced documentation time",
                    "Improved clinical efficiency",
                    "Enhanced workflow integration"
                ]
            },
            
            "clinical_outcomes": {
                "improvement_description": "Measurable improvements in clinical outcomes and patient satisfaction",
                "measurable_metrics": [
                    "Patient outcome improvements",
                    "Reduced length of stay",
                    "Improved patient satisfaction scores"
                ]
            }
        }
    
    def _calculate_financial_impact(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate financial impact of implementation"""
        return {
            "cost_savings": {
                "annual_savings": customer_data.get("annual_cost_savings", "$500,000"),
                "breakdown": {
                    "labor_cost_reduction": "40%",
                    "efficiency_gains": "35%",
                    "error_reduction": "25%"
                }
            },
            
            "revenue_impact": {
                "increased_capacity": customer_data.get("capacity_increase", "15%"),
                "improved_billing_accuracy": customer_data.get("billing_improvement", "20%"),
                "reduced_denials": customer_data.get("denial_reduction", "30%")
            },
            
            "roi_analysis": {
                "total_investment": customer_data.get("implementation_cost", "$250,000"),
                "annual_benefit": customer_data.get("annual_benefit", "$750,000"),
                "payback_period": customer_data.get("payback_period", "8 months"),
                "three_year_roi": customer_data.get("three_year_roi", "400%")
            }
        }
    
    def _determine_timeline_to_value(self, customer_data: Dict[str, Any]) -> str:
        """Determine timeline to value realization"""
        return customer_data.get("time_to_value", "12 weeks from implementation start")
    
    def _identify_stakeholders(self, customer_data: Dict[str, Any]) -> List[str]:
        """Identify key stakeholders for success story"""
        stakeholders = [
            customer_data.get("chief_medical_officer", ""),
            customer_data.get("chief_information_officer", ""),
            customer_data.get("clinical_department_head", ""),
            customer_data.get("implementation_lead", ""),
            customer_data.get("nursing_director", "")
        ]
        
        return [s for s in stakeholders if s]  # Remove empty entries
    
    def _create_story_narrative(self, customer_data: Dict[str, Any], success_type: str) -> str:
        """Create narrative structure for success story"""
        organization_name = customer_data.get("organization_name", "Healthcare Organization")
        clinical_specialty = customer_data.get("clinical_specialties", ["general medicine"])[0]
        
        narrative = f"""
        {organization_name}, a leading {clinical_specialty} healthcare provider, faced significant challenges in clinical workflow efficiency and decision-making processes. With a growing patient population and increasing regulatory requirements, the organization needed a solution that could enhance clinical decision-making while maintaining the highest standards of patient care.
        
        Through implementation of our AI-powered clinical decision support system, {organization_name} achieved remarkable transformations across multiple dimensions of healthcare delivery. The solution integrated seamlessly with existing clinical workflows, providing real-time assistance to healthcare providers while maintaining clinical autonomy and accountability.
        
        The results speak for themselves: {customer_data.get('doc_time_reduction', '35%')} reduction in clinical documentation time, {customer_data.get('accuracy_improvement', '25%')} improvement in clinical decision accuracy, and {customer_data.get('patient_satisfaction_increase', '20%')} increase in patient satisfaction scores. These improvements translated into significant cost savings of {customer_data.get('annual_cost_savings', '$500,000')} annually, with a payback period of just {customer_data.get('payback_period', '8 months')}.
        
        Today, {organization_name} continues to leverage the system's advanced capabilities to drive continuous improvement in clinical outcomes, patient experience, and operational efficiency, establishing itself as a leader in AI-enhanced healthcare delivery.
        """
        
        return narrative.strip()
    
    def _identify_media_assets(self, customer_data: Dict[str, Any]) -> List[str]:
        """Identify media assets needed for success story"""
        assets = []
        
        # Organization assets
        if customer_data.get("organization_logo"):
            assets.append("Organization Logo")
        
        # Leadership photos
        if customer_data.get("executive_willing_for_photo"):
            assets.append("Executive Leadership Photos")
        
        # Clinical workflow screenshots
        if customer_data.get("workflow_documentation"):
            assets.append("Clinical Workflow Screenshots")
        
        # Performance dashboards
        if customer_data.get("dashboard_sharing"):
            assets.append("Performance Dashboard Images")
        
        # Testimonial videos
        if customer_data.get("video_testimonial_willingness"):
            assets.append("Customer Testimonial Video")
        
        return assets
    
    def _define_target_audiences(self, success_type: str) -> List[str]:
        """Define target audiences for success story"""
        if success_type == "clinical_efficiency":
            return [
                "Clinical Directors",
                "Chief Medical Officers",
                "Healthcare Quality Leaders",
                "Clinical Department Heads"
            ]
        elif success_type == "implementation_success":
            return [
                "IT Directors",
                "Implementation Managers",
                "Healthcare Executives",
                "Project Management Offices"
            ]
        elif success_type == "patient_experience":
            return [
                "Patient Experience Officers",
                "Healthcare Marketing Directors",
                "Hospital Administrators",
                "Quality Improvement Leaders"
            ]
        else:
            return [
                "Healthcare Executives",
                "Clinical Leaders",
                "IT Decision Makers",
                "Implementation Teams"
            ]
    
    def manage_reference_requests(self, prospect_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Manage customer reference requests"""
        suitable_references = []
        
        # Find customers matching requirements
        for tier in ReferenceTier:
            tier_customers = self._get_customers_by_tier(tier)
            matching_customers = self._filter_customers_by_requirements(
                tier_customers, prospect_requirements
            )
            suitable_references.extend(matching_customers)
        
        # Sort by suitability score
        suitable_references.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)
        
        # Prepare reference responses
        reference_responses = []
        for ref in suitable_references[:5]:  # Top 5 candidates
            response = {
                "reference_customer": ref,
                "reference_type": self._determine_reference_type(ref, prospect_requirements),
                "talking_points": self._generate_talking_points(ref, prospect_requirements),
                "availability_confirmation": self._check_availability(ref),
                "contact_information": ref.get("primary_contact"),
                "coordination_notes": self._prepare_coordination_notes(ref, prospect_requirements)
            }
            reference_responses.append(response)
        
        return reference_responses
    
    def _get_customers_by_tier(self, tier: ReferenceTier) -> List[Dict[str, Any]]:
        """Get customers by reference tier (simplified for demo)"""
        # This would query actual customer database
        return [
            {
                "organization_id": "ORG001",
                "organization_name": "Metro General Hospital",
                "tier": tier,
                "primary_contact": {"name": "Dr. Sarah Johnson", "title": "CMO", "email": "sarah.johnson@metrogeneral.org"},
                "suitability_score": 85,
                "specialties": ["emergency_medicine", "internal_medicine"],
                "recent_projects": ["clinical_decision_support", "workflow_optimization"]
            }
        ]
    
    def _filter_customers_by_requirements(self, customers: List[Dict[str, Any]], 
                                        requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter customers based on prospect requirements"""
        filtered = []
        
        for customer in customers:
            score = 0
            
            # Match healthcare sector
            required_sector = requirements.get("healthcare_sector")
            if required_sector and required_sector in customer.get("specialties", []):
                score += 20
            
            # Match implementation size
            required_size = requirements.get("organization_size")
            customer_size = customer.get("organization_size", "medium")
            if required_size and required_size == customer_size:
                score += 15
            
            # Match special use case
            required_use_case = requirements.get("primary_use_case")
            if required_use_case and required_use_case in customer.get("recent_projects", []):
                score += 25
            
            # Availability
            if customer.get("reference_availability") == "available":
                score += 20
            
            # Quality rating
            score += customer.get("reference_rating", 3.0) * 10
            
            customer["suitability_score"] = score
            if score >= 50:  # Minimum threshold
                filtered.append(customer)
        
        return filtered
    
    def _determine_reference_type(self, customer: Dict[str, Any], 
                                requirements: Dict[str, Any]) -> str:
        """Determine appropriate reference type"""
        if requirements.get("decision_stage") == "early":
            return "Implementation Experience Reference"
        elif requirements.get("decision_stage") == "evaluation":
            return "Technical Capability Reference"
        elif requirements.get("decision_stage") == "decision":
            return "ROI and Business Value Reference"
        else:
            return "General Reference"
    
    def _generate_talking_points(self, customer: Dict[str, Any], 
                               requirements: Dict[str, Any]) -> List[str]:
        """Generate talking points for reference conversation"""
        talking_points = [
            "Overview of implementation approach and timeline",
            "Key challenges encountered and how they were overcome",
            "Measurable outcomes and ROI achieved",
            "Clinical impact and provider acceptance",
            "Lessons learned and best practices"
        ]
        
        # Customize based on prospect requirements
        if requirements.get("focus_area") == "clinical_outcomes":
            talking_points.extend([
                "Specific clinical quality improvements",
                "Patient safety enhancements",
                "Provider workflow benefits"
            ])
        elif requirements.get("focus_area") == "implementation":
            talking_points.extend([
                "Project management approach",
                "Change management strategies",
                "Training and adoption success factors"
            ])
        
        return talking_points
    
    def _check_availability(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """Check customer availability for reference calls"""
        # Simplified availability check
        return {
            "availability": "available",
            "preferred_contact_methods": ["email", "phone"],
            "timezone": customer.get("timezone", "EST"),
            "best_contact_times": ["business_hours"],
            "notice_required": "24 hours"
        }
    
    def _prepare_coordination_notes(self, customer: Dict[str, Any], 
                                  requirements: Dict[str, Any]) -> str:
        """Prepare coordination notes for reference call"""
        notes = f"""
        Reference coordination notes for {customer.get('organization_name')}:
        
        Key talking points: Focus on {requirements.get('focus_area', 'general implementation')}
        Decision stage: {requirements.get('decision_stage', 'general')}
        Prospect profile: {requirements.get('prospect_organization_type', 'unknown')}
        
        Customer context: {customer.get('organization_name')} implemented clinical decision support
        successfully with focus on {', '.join(customer.get('specialties', ['general medicine'])[:2])}.
        
        Preferred approach: Structured conversation with specific questions about 
        {requirements.get('primary_concerns', ['implementation experience'])}.
        """
        
        return notes.strip()
    
    def generate_advocacy_dashboard(self, advocates: List[CustomerAdvocate], 
                                  stories: List[SuccessStory]) -> Dict[str, Any]:
        """Generate advocacy program dashboard"""
        dashboard = {
            "program_overview": {
                "total_advocates": len(advocates),
                "active_references": len([a for a in advocates if a.availability_status == "available"]),
                "case_studies_in_development": len([s for s in stories if s.approval_status in [CaseStudyStatus.DRAFT, CaseStudyStatus.IN_REVIEW]]),
                "published_success_stories": len([s for s in stories if s.approval_status == CaseStudyStatus.PUBLISHED])
            },
            
            "advocate_breakdown": {
                "by_tier": {},
                "by_specialty": {},
                "by_engagement_level": {}
            },
            
            "content_pipeline": {
                "case_studies_by_status": {},
                "content_types_distribution": {},
                "approval_pipeline": []
            },
            
            "program_impact": {
                "reference_conversion_rate": 0.0,
                "case_study_engagement_metrics": {},
                "customer_lifetime_value_impact": {},
                "brand_mention_volume": 0
            },
            
            "recommendations": []
        }
        
        # Advocate breakdown by tier
        for tier in ReferenceTier:
            count = len([a for a in advocates if tier in a.preferred_activities])
            dashboard["advocate_breakdown"]["by_tier"][tier.value] = count
        
        # Content pipeline analysis
        for status in CaseStudyStatus:
            count = len([s for s in stories if s.approval_status == status])
            dashboard["content_pipeline"]["case_studies_by_status"][status.value] = count
        
        # Generate recommendations
        dashboard["recommendations"] = self._generate_advocacy_recommendations(dashboard, advocates, stories)
        
        return dashboard
    
    def _generate_advocacy_recommendations(self, dashboard: Dict[str, Any], 
                                         advocates: List[CustomerAdvocate],
                                         stories: List[SuccessStory]) -> List[str]:
        """Generate advocacy program recommendations"""
        recommendations = []
        
        # Advocate recruitment
        if dashboard["program_overview"]["active_references"] < 10:
            recommendations.append("Recruit additional reference customers to meet demand")
        
        # Case study development
        if dashboard["content_pipeline"]["case_studies_by_status"].get("draft", 0) > 5:
            recommendations.append("Accelerate case study review and approval process")
        
        # Tier distribution
        tier_distribution = dashboard["advocate_breakdown"]["by_tier"]
        if tier_distribution.get("strategic", 0) < 2:
            recommendations.append("Develop more strategic reference relationships")
        
        return recommendations
    
    def export_advocacy_program_data(self, advocates: List[CustomerAdvocate],
                                   stories: List[SuccessStory],
                                   output_path: str) -> None:
        """Export advocacy program data"""
        export_data = {
            "customer_advocates": [asdict(advocate) for advocate in advocates],
            "success_stories": [asdict(story) for story in stories],
            "advocacy_dashboard": self.generate_advocacy_dashboard(advocates, stories),
            "export_timestamp": datetime.now().isoformat(),
            "program_summary": {
                "total_participants": len(advocates),
                "total_stories": len(stories),
                "active_references": len([a for a in advocates if a.availability_status == "available"])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)