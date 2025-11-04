"""
Customer Support Ticketing System
Healthcare-focused ticketing system with medical case prioritization
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import logging

from config.support_config import PriorityLevel, SupportTier, SupportConfig

logger = logging.getLogger(__name__)

class TicketStatus(Enum):
    NEW = "new"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    PENDING_MEDICAL = "pending_medical"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class TicketCategory(Enum):
    MEDICAL_EMERGENCY = "medical_emergency"
    CLINICAL_ISSUE = "clinical_issue"
    TECHNICAL_ISSUE = "technical_issue"
    INTEGRATION_ISSUE = "integration_issue"
    COMPLIANCE_ISSUE = "compliance_issue"
    TRAINING_REQUEST = "training_request"
    ADMINISTRATIVE = "administrative"
    SYSTEM_OUTAGE = "system_outage"

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

@dataclass
class MedicalContext:
    """Medical context for healthcare support tickets"""
    patient_age: Optional[int] = None
    medical_specialty: Optional[MedicalSpecialty] = None
    clinical_priority: Optional[str] = None
    facility_type: Optional[str] = None
    department: Optional[str] = None
    attending_physician: Optional[str] = None
    urgency_level: Optional[str] = None
    patient_safety_impact: Optional[str] = None

@dataclass
class TicketAttachment:
    """Attachment for support tickets"""
    id: str
    filename: str
    file_type: str
    size_bytes: int
    upload_timestamp: datetime
    is_medical_record: bool
    compliance_classification: Optional[str] = None

@dataclass
class TicketComment:
    """Comment on support ticket"""
    id: str
    ticket_id: str
    author_id: str
    author_name: str
    author_role: str
    content: str
    timestamp: datetime
    is_internal: bool
    is_medical_update: bool
    attachments: List[TicketAttachment]

@dataclass
class SupportTicket:
    """Main support ticket class"""
    id: str
    title: str
    description: str
    category: TicketCategory
    priority: PriorityLevel
    status: TicketStatus
    reporter_id: str
    reporter_name: str
    reporter_facility: str
    reporter_role: str
    assigned_to: Optional[str]
    assigned_team: Optional[str]
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    sla_due_at: Optional[datetime]
    medical_context: MedicalContext
    tags: List[str]
    attachments: List[TicketAttachment]
    comments: List[TicketComment]
    escalation_history: List[Dict]
    correlation_id: Optional[str]  # For linking related tickets
    affected_users: List[str]
    business_impact: Optional[str]
    resolution_summary: Optional[str]

class MedicalPrioritizationEngine:
    """AI-powered medical prioritization engine"""
    
    def __init__(self):
        self.priority_keywords = {
            PriorityLevel.EMERGENCY: [
                "emergency", "urgent", "immediate", "critical", "life-threatening",
                "cardiac arrest", "respiratory failure", "severe bleeding",
                "loss of consciousness", "acute pain", "surgery needed"
            ],
            PriorityLevel.CRITICAL_MEDICAL: [
                "severe", "acute", "worsening", "unstable", "deteriorating",
                "high fever", "severe pain", "breathing difficulty",
                "chest pain", "stroke symptoms", "sepsis"
            ],
            PriorityLevel.HIGH_MEDICAL: [
                "concerning", "abnormal", "unexpected", "persistent",
                "moderate pain", "chronic condition", "medication issue",
                "test results", "appointment scheduling"
            ],
            PriorityLevel.STANDARD_MEDICAL: [
                "general inquiry", "normal", "routine", "scheduling",
                "account access", "feature request", "documentation"
            ]
        }
        
        self.medical_specialty_keywords = {
            MedicalSpecialty.CARDIOLOGY: ["heart", "cardiac", "chest pain", "blood pressure"],
            MedicalSpecialty.NEUROLOGY: ["brain", "stroke", "seizure", "neurological"],
            MedicalSpecialty.EMERGENCY_MEDICINE: ["emergency", "trauma", "critical", "urgent"],
            MedicalSpecialty.PEDIATRICS: ["child", "pediatric", "infant", "baby"],
            MedicalSpecialty.PSYCHIATRY: ["mental health", "psychiatric", "depression", "anxiety"]
        }
    
    def classify_ticket_priority(self, title: str, description: str, context: MedicalContext) -> Tuple[PriorityLevel, float]:
        """Classify ticket priority based on content and context"""
        combined_text = f"{title} {description}".lower()
        
        # Base priority from keywords
        max_score = 0
        best_priority = PriorityLevel.STANDARD_MEDICAL
        
        for priority, keywords in self.priority_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > max_score:
                max_score = score
                best_priority = priority
        
        # Boost priority based on medical context
        if context.urgency_level == "immediate":
            best_priority = PriorityLevel.EMERGENCY
        elif context.urgency_level == "urgent" and best_priority != PriorityLevel.EMERGENCY:
            best_priority = PriorityLevel.CRITICAL_MEDICAL
        elif context.patient_safety_impact in ["high", "critical"]:
            best_priority = max(best_priority, PriorityLevel.HIGH_MEDICAL)
        
        # Boost for emergency medicine specialty
        if context.medical_specialty == MedicalSpecialty.EMERGENCY_MEDICINE:
            best_priority = max(best_priority, PriorityLevel.CRITICAL_MEDICAL)
        
        confidence = min(max_score / 5.0, 1.0)  # Normalize confidence
        return best_priority, confidence
    
    def identify_medical_specialty(self, content: str, context: MedicalContext) -> Optional[MedicalSpecialty]:
        """Identify relevant medical specialty for the ticket"""
        combined_text = content.lower()
        
        for specialty, keywords in self.medical_specialty_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return specialty
        
        return context.medical_specialty

class TicketManagementSystem:
    """Main ticket management system for healthcare support"""
    
    def __init__(self):
        self.tickets: Dict[str, SupportTicket] = {}
        self.prioritization_engine = MedicalPrioritizationEngine()
        self.ticket_counter = 0
        self.escalation_manager = EscalationManager()
    
    async def create_ticket(
        self,
        title: str,
        description: str,
        category: TicketCategory,
        reporter_id: str,
        reporter_name: str,
        reporter_facility: str,
        reporter_role: str,
        medical_context: Optional[MedicalContext] = None,
        priority: Optional[PriorityLevel] = None,
        tags: Optional[List[str]] = None,
        affected_users: Optional[List[str]] = None
    ) -> SupportTicket:
        """Create a new support ticket with medical prioritization"""
        
        # Auto-prioritize if not specified
        if priority is None and medical_context:
            priority, confidence = self.prioritization_engine.classify_ticket_priority(
                title, description, medical_context
            )
            logger.info(f"Auto-prioritized ticket with confidence {confidence:.2f}: {priority}")
        
        if priority is None:
            priority = PriorityLevel.STANDARD_MEDICAL
        
        # Generate unique ticket ID
        self.ticket_counter += 1
        ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{self.ticket_counter:04d}"
        
        # Calculate SLA due time
        sla_config = SupportConfig.get_sla_for_priority(priority)
        sla_due_at = datetime.now() + timedelta(hours=sla_config.resolution_time_hours)
        
        # Create ticket
        ticket = SupportTicket(
            id=ticket_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            status=TicketStatus.NEW,
            reporter_id=reporter_id,
            reporter_name=reporter_name,
            reporter_facility=reporter_facility,
            reporter_role=reporter_role,
            assigned_to=None,
            assigned_team=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            closed_at=None,
            sla_due_at=sla_due_at,
            medical_context=medical_context or MedicalContext(),
            tags=tags or [],
            attachments=[],
            comments=[],
            escalation_history=[],
            correlation_id=None,
            affected_users=affected_users or [],
            business_impact=None,
            resolution_summary=None
        )
        
        # Store ticket
        self.tickets[ticket_id] = ticket
        
        # Auto-assign based on priority and medical context
        await self._auto_assign_ticket(ticket)
        
        # Trigger escalation if emergency
        if SupportConfig.is_medical_emergency(priority):
            await self.escalation_manager.trigger_emergency_escalation(ticket)
        
        logger.info(f"Created ticket {ticket_id} with priority {priority}")
        return ticket
    
    async def _auto_assign_ticket(self, ticket: SupportTicket) -> None:
        """Auto-assign ticket based on priority and medical context"""
        
        # Determine appropriate team
        if ticket.priority == PriorityLevel.EMERGENCY:
            ticket.assigned_team = "emergency_medical_team"
        elif SupportConfig.requires_medical_specialist(ticket.priority):
            # Assign to appropriate medical specialty team
            specialty = ticket.medical_context.medical_specialty
            if specialty:
                ticket.assigned_team = f"medical_specialists_{specialty.value}"
            else:
                ticket.assigned_team = "medical_specialists_general"
        elif ticket.category == TicketCategory.TECHNICAL_ISSUE:
            ticket.assigned_team = "technical_support"
        elif ticket.category == TicketCategory.COMPLIANCE_ISSUE:
            ticket.assigned_team = "compliance_team"
        elif ticket.category == TicketCategory.TRAINING_REQUEST:
            ticket.assigned_team = "training_team"
        else:
            ticket.assigned_team = "healthcare_support"
        
        # Update status
        ticket.status = TicketStatus.ASSIGNED
        ticket.updated_at = datetime.now()
    
    async def update_ticket_status(
        self,
        ticket_id: str,
        new_status: TicketStatus,
        updater_id: str,
        comment: Optional[str] = None
    ) -> SupportTicket:
        """Update ticket status and add comment if provided"""
        
        if ticket_id not in self.tickets:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        ticket = self.tickets[ticket_id]
        old_status = ticket.status
        ticket.status = new_status
        ticket.updated_at = datetime.now()
        
        # Set resolved/closed timestamps
        if new_status == TicketStatus.RESOLVED:
            ticket.resolved_at = datetime.now()
        elif new_status == TicketStatus.CLOSED:
            ticket.closed_at = datetime.now()
        
        # Add status change comment
        if comment:
            await self.add_ticket_comment(
                ticket_id,
                comment,
                updater_id,
                is_internal=False
            )
        
        # Trigger escalation if status indicates worsening
        if old_status != TicketStatus.ESCALATED and new_status == TicketStatus.ESCALATED:
            await self.escalation_manager.escalate_ticket(ticket, "Manual escalation by support agent")
        
        logger.info(f"Updated ticket {ticket_id} status from {old_status} to {new_status}")
        return ticket
    
    async def add_ticket_comment(
        self,
        ticket_id: str,
        content: str,
        author_id: str,
        author_name: str = None,
        author_role: str = None,
        is_internal: bool = False,
        is_medical_update: bool = False,
        attachments: Optional[List[TicketAttachment]] = None
    ) -> TicketComment:
        """Add comment to ticket"""
        
        if ticket_id not in self.tickets:
            raise ValueError(f"Ticket {ticket_id} not found")
        
        comment = TicketComment(
            id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            author_id=author_id,
            author_name=author_name or author_id,
            author_role=author_role or "unknown",
            content=content,
            timestamp=datetime.now(),
            is_internal=is_internal,
            is_medical_update=is_medical_update,
            attachments=attachments or []
        )
        
        self.tickets[ticket_id].comments.append(comment)
        self.tickets[ticket_id].updated_at = datetime.now()
        
        return comment
    
    async def get_tickets_by_priority(self, priority: PriorityLevel) -> List[SupportTicket]:
        """Get all tickets by priority level"""
        return [ticket for ticket in self.tickets.values() if ticket.priority == priority]
    
    async def get_overdue_tickets(self) -> List[SupportTicket]:
        """Get all tickets that are overdue for SLA"""
        now = datetime.now()
        return [
            ticket for ticket in self.tickets.values()
            if ticket.sla_due_at and ticket.sla_due_at < now 
            and ticket.status not in [TicketStatus.RESOLVED, TicketStatus.CLOSED]
        ]
    
    async def get_tickets_by_facility(self, facility: str) -> List[SupportTicket]:
        """Get all tickets for a specific healthcare facility"""
        return [ticket for ticket in self.tickets.values() if ticket.reporter_facility == facility]
    
    async def link_tickets(self, primary_ticket_id: str, secondary_ticket_id: str, relationship_type: str) -> None:
        """Link related tickets together"""
        if primary_ticket_id in self.tickets and secondary_ticket_id in self.tickets:
            # Set correlation ID
            self.tickets[secondary_ticket_id].correlation_id = primary_ticket_id
            logger.info(f"Linked tickets {primary_ticket_id} and {secondary_ticket_id} as {relationship_type}")
    
    async def generate_ticket_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive ticket analytics report"""
        
        # Filter tickets by date range
        period_tickets = [
            ticket for ticket in self.tickets.values()
            if start_date <= ticket.created_at <= end_date
        ]
        
        # Calculate metrics
        total_tickets = len(period_tickets)
        resolved_tickets = len([t for t in period_tickets if t.status == TicketStatus.RESOLVED])
        escalated_tickets = len([t for t in period_tickets if t.status == TicketStatus.ESCALATED])
        
        # Priority distribution
        priority_distribution = defaultdict(int)
        for ticket in period_tickets:
            priority_distribution[ticket.priority.value] += 1
        
        # Average resolution time by priority
        avg_resolution_time = {}
        for priority in PriorityLevel:
            priority_tickets = [t for t in period_tickets if t.priority == priority and t.resolved_at]
            if priority_tickets:
                total_time = sum([
                    (t.resolved_at - t.created_at).total_seconds() / 3600  # Convert to hours
                    for t in priority_tickets
                ])
                avg_resolution_time[priority.value] = total_time / len(priority_tickets)
        
        # SLA compliance
        sla_compliance = {}
        for priority in PriorityLevel:
            priority_tickets = [t for t in period_tickets if t.priority == priority]
            if priority_tickets:
                compliant = len([t for t in priority_tickets if t.sla_due_at and t.resolved_at and t.resolved_at <= t.sla_due_at])
                sla_compliance[priority.value] = (compliant / len(priority_tickets)) * 100
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_tickets": total_tickets,
                "resolved_tickets": resolved_tickets,
                "escalated_tickets": escalated_tickets,
                "resolution_rate": (resolved_tickets / total_tickets * 100) if total_tickets > 0 else 0
            },
            "priority_distribution": dict(priority_distribution),
            "average_resolution_time_hours": avg_resolution_time,
            "sla_compliance_percentage": sla_compliance,
            "top_categories": self._get_top_categories(period_tickets),
            "facility_breakdown": self._get_facility_breakdown(period_tickets),
            "medical_context_analysis": self._analyze_medical_context(period_tickets)
        }
    
    def _get_top_categories(self, tickets: List[SupportTicket]) -> Dict[str, int]:
        """Get top ticket categories"""
        category_counts = defaultdict(int)
        for ticket in tickets:
            category_counts[ticket.category.value] += 1
        return dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _get_facility_breakdown(self, tickets: List[SupportTicket]) -> Dict[str, int]:
        """Get ticket breakdown by healthcare facility"""
        facility_counts = defaultdict(int)
        for ticket in tickets:
            facility_counts[ticket.reporter_facility] += 1
        return dict(sorted(facility_counts.items(), key=lambda x: x[1], reverse=True))
    
    def _analyze_medical_context(self, tickets: List[SupportTicket]) -> Dict[str, Any]:
        """Analyze medical context patterns"""
        specialty_breakdown = defaultdict(int)
        urgency_breakdown = defaultdict(int)
        facility_types = defaultdict(int)
        
        for ticket in tickets:
            if ticket.medical_context.medical_specialty:
                specialty_breakdown[ticket.medical_context.medical_specialty.value] += 1
            if ticket.medical_context.urgency_level:
                urgency_breakdown[ticket.medical_context.urgency_level] += 1
            if ticket.medical_context.facility_type:
                facility_types[ticket.medical_context.facility_type] += 1
        
        return {
            "specialty_breakdown": dict(specialty_breakdown),
            "urgency_breakdown": dict(urgency_breakdown),
            "facility_type_breakdown": dict(facility_types)
        }

class EscalationManager:
    """Handles ticket escalation for medical emergencies"""
    
    def __init__(self):
        self.escalation_contacts = {
            "emergency_medical_team": {
                "email": "emergency@medicalai.com",
                "phone": "+1-800-MEDICAL",
                "slack": "#emergency-medical"
            },
            "medical_specialists": {
                "email": "specialists@medicalai.com", 
                "phone": "+1-800-SPECIAL",
                "slack": "#medical-specialists"
            }
        }
    
    async def trigger_emergency_escalation(self, ticket: SupportTicket) -> None:
        """Trigger emergency escalation for critical medical tickets"""
        
        escalation_data = {
            "ticket_id": ticket.id,
            "priority": ticket.priority.value,
            "title": ticket.title,
            "reporter": f"{ticket.reporter_name} ({ticket.reporter_facility})",
            "medical_context": asdict(ticket.medical_context),
            "timestamp": datetime.now().isoformat()
        }
        
        # Log escalation
        logger.critical(f"EMERGENCY ESCALATION: {json.dumps(escalation_data)}")
        
        # Send immediate notifications (implementation would depend on notification system)
        await self._send_emergency_notifications(ticket, escalation_data)
        
        # Record in escalation history
        ticket.escalation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "emergency_escalation",
            "triggered_by": "automatic_priority_detection",
            "details": escalation_data
        })
    
    async def escalate_ticket(self, ticket: SupportTicket, reason: str) -> None:
        """Manual escalation of ticket"""
        
        ticket.status = TicketStatus.ESCALATED
        ticket.updated_at = datetime.now()
        
        escalation_data = {
            "ticket_id": ticket.id,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning(f"TICKET ESCALATED: {json.dumps(escalation_data)}")
        
        # Add escalation to history
        ticket.escalation_history.append({
            "timestamp": datetime.now().isoformat(),
            "type": "manual_escalation",
            "reason": reason,
            "details": escalation_data
        })
    
    async def _send_emergency_notifications(self, ticket: SupportTicket, escalation_data: Dict) -> None:
        """Send emergency notifications (placeholder implementation)"""
        # In production, this would integrate with:
        # - Email systems
        # - SMS services
        # - Slack/Microsoft Teams
        # - PagerDuty/OnCall systems
        # - Hospital communication systems
        
        logger.info(f"Emergency notifications sent for ticket {ticket.id}")

# Global ticket management system instance
ticket_system = TicketManagementSystem()

# Example usage and testing functions
async def create_sample_tickets():
    """Create sample tickets for testing"""
    
    # Emergency medical ticket
    emergency_context = MedicalContext(
        medical_specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
        urgency_level="immediate",
        patient_safety_impact="critical",
        department="Emergency Department"
    )
    
    emergency_ticket = await ticket_system.create_ticket(
        title="System outage during cardiac arrest situation",
        description="Medical AI system is down during emergency cardiac case. Need immediate assistance.",
        category=TicketCategory.SYSTEM_OUTAGE,
        reporter_id="dr_smith_001",
        reporter_name="Dr. Sarah Smith",
        reporter_facility="General Hospital",
        reporter_role="Emergency Physician",
        medical_context=emergency_context
    )
    
    # Standard technical ticket
    standard_ticket = await ticket_system.create_ticket(
        title="Cannot access patient records in cardiology module",
        description="Getting error when trying to view cardiology patient data",
        category=TicketCategory.TECHNICAL_ISSUE,
        reporter_id="nurse_jones_002",
        reporter_name="Nurse Jones",
        reporter_facility="Heart Center",
        reporter_role="Registered Nurse"
    )
    
    print(f"Created tickets: {emergency_ticket.id}, {standard_ticket.id}")

if __name__ == "__main__":
    asyncio.run(create_sample_tickets())