"""
Incident Management and Escalation System
Healthcare-focused incident management with medical emergency procedures
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging

from config.support_config import PriorityLevel, SupportConfig

logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    SEV1_CRITICAL = "sev1_critical"  # Complete system outage, patient safety risk
    SEV2_HIGH = "sev2_high"          # Major functionality loss, clinical impact
    SEV3_MEDIUM = "sev3_medium"      # Minor functionality loss, workaround available
    SEV4_LOW = "sev4_low"           # Cosmetic issues, no functional impact

class IncidentStatus(Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    IDENTIFIED = "identified"
    MONITORING = "monitoring"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"

class IncidentType(Enum):
    SYSTEM_OUTAGE = "system_outage"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_INTEGRITY = "data_integrity"
    SECURITY_INCIDENT = "security_incident"
    MEDICAL_DEVICE_FAILURE = "medical_device_failure"
    EHR_INTEGRATION_FAILURE = "ehr_integration_failure"
    CLINICAL_WORKFLOW_DISRUPTION = "clinical_workflow_disruption"
    PATIENT_SAFETY_RISK = "patient_safety_risk"
    COMPLIANCE_VIOLATION = "compliance_violation"
    TRAINING_ISSUE = "training_issue"

class EscalationLevel(Enum):
    L1_SUPPORT = "l1_support"
    L2_SUPPORT = "l2_support"
    L3_ENGINEERING = "l3_engineering"
    MEDICAL_SPECIALIST = "medical_specialist"
    EMERGENCY_MEDICAL_TEAM = "emergency_medical_team"
    EXECUTIVE_TEAM = "executive_team"
    REGULATORY_COMPLIANCE = "regulatory_compliance"

@dataclass
class MedicalContext:
    """Medical context for incident management"""
    affected_patients: Optional[int] = None
    clinical_area: Optional[str] = None
    medical_specialty: Optional[str] = None
    patient_safety_impact: Optional[str] = None
    regulatory_reporting_required: bool = False
    clinical_workflow_disruption: bool = False
    emergency_situation: bool = False

@dataclass
class IncidentUpdate:
    """Update to an incident"""
    id: str
    incident_id: str
    author_id: str
    author_name: str
    author_role: str
    content: str
    timestamp: datetime
    status_change: Optional[IncidentStatus] = None
    severity_change: Optional[IncidentSeverity] = None
    estimated_resolution_time: Optional[datetime] = None
    internal_notes: Optional[str] = None

@dataclass
class EscalationRecord:
    """Record of incident escalation"""
    id: str
    incident_id: str
    from_level: EscalationLevel
    to_level: EscalationLevel
    reason: str
    escalated_by: str
    escalated_at: datetime
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

@dataclass
class IncidentImpact:
    """Impact assessment for incident"""
    user_count: int
    facility_count: int
    patient_count: Optional[int]
    clinical_areas_affected: List[str]
    estimated_downtime: timedelta
    business_impact: str
    regulatory_impact: str
    patient_safety_risk: str

@dataclass
class SupportIncident:
    """Main incident class"""
    id: str
    title: str
    description: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime]
    closed_at: Optional[datetime]
    reporter_id: str
    reporter_name: str
    reporter_facility: str
    assigned_to: Optional[str]
    assigned_team: Optional[str]
    impact: IncidentImpact
    medical_context: MedicalContext
    root_cause: Optional[str]
    resolution_summary: Optional[str]
    post_incident_review_required: bool = False
    regulatory_notification_sent: bool = False
    customer_communication_required: bool = False

class EscalationManager:
    """Handles incident escalation based on medical context and severity"""
    
    def __init__(self):
        self.escalation_rules = {
            # Patient Safety Escalations
            IncidentSeverity.SEV1_CRITICAL: EscalationLevel.EMERGENCY_MEDICAL_TEAM,
            IncidentSeverity.SEV2_HIGH: EscalationLevel.MEDICAL_SPECIALIST,
            
            # Time-based escalation
            "patient_safety_risk": EscalationLevel.EMERGENCY_MEDICAL_TEAM,
            "medical_device_failure": EscalationLevel.MEDICAL_SPECIALIST,
            "clinical_workflow_disruption": EscalationLevel.MEDICAL_SPECIALIST,
            
            # Time-based automatic escalation
            "time_threshold_15min": EscalationLevel.L2_SUPPORT,
            "time_threshold_30min": EscalationLevel.L3_ENGINEERING,
            "time_threshold_60min": EscalationLevel.EXECUTIVE_TEAM
        }
        
        self.escalation_contacts = {
            EscalationLevel.L1_SUPPORT: {
                "team": "Primary Support Team",
                "availability": "24/7",
                "response_time": "15 minutes"
            },
            EscalationLevel.L2_SUPPORT: {
                "team": "Senior Support Engineers", 
                "availability": "24/7",
                "response_time": "30 minutes"
            },
            EscalationLevel.L3_ENGINEERING: {
                "team": "Development Team",
                "availability": "On-call 24/7",
                "response_time": "1 hour"
            },
            EscalationLevel.MEDICAL_SPECIALIST: {
                "team": "Medical Technology Specialists",
                "availability": "24/7",
                "response_time": "30 minutes"
            },
            EscalationLevel.EMERGENCY_MEDICAL_TEAM: {
                "team": "Emergency Medical Response Team",
                "availability": "Immediate",
                "response_time": "5 minutes"
            }
        }
    
    def determine_initial_escalation(self, incident: SupportIncident) -> EscalationLevel:
        """Determine initial escalation level based on incident characteristics"""
        
        # Patient safety incidents
        if incident.medical_context.patient_safety_impact in ["high", "critical"]:
            return EscalationLevel.EMERGENCY_MEDICAL_TEAM
        
        # Medical device failures
        if incident.incident_type == IncidentType.MEDICAL_DEVICE_FAILURE:
            return EscalationLevel.MEDICAL_SPECIALIST
        
        # Severity-based escalation
        if incident.severity == IncidentSeverity.SEV1_CRITICAL:
            return EscalationLevel.EMERGENCY_MEDICAL_TEAM
        elif incident.severity == IncidentSeverity.SEV2_HIGH:
            return EscalationLevel.MEDICAL_SPECIALIST
        elif incident.severity == IncidentSeverity.SEV3_MEDIUM:
            return EscalationLevel.L2_SUPPORT
        else:
            return EscalationLevel.L1_SUPPORT
        
        # Emergency situation override
        if incident.medical_context.emergency_situation:
            return EscalationLevel.EMERGENCY_MEDICAL_TEAM
    
    def should_auto_escalate(self, incident: SupportIncident, current_time: datetime) -> Optional[EscalationLevel]:
        """Determine if automatic escalation is needed based on time thresholds"""
        
        incident_age = current_time - incident.created_at
        
        # Critical incidents - escalate immediately
        if incident.severity == IncidentSeverity.SEV1_CRITICAL:
            if incident_age > timedelta(minutes=5):
                return EscalationLevel.EMERGENCY_MEDICAL_TEAM
        elif incident.severity == IncidentSeverity.SEV2_HIGH:
            if incident_age > timedelta(minutes=30):
                return EscalationLevel.MEDICAL_SPECIALIST
            elif incident_age > timedelta(minutes=15):
                return EscalationLevel.L2_SUPPORT
        
        # Time-based escalation for all incidents
        if incident_age > timedelta(hours=1):
            return EscalationLevel.EXECUTIVE_TEAM
        elif incident_age > timedelta(minutes=60):
            return EscalationLevel.L3_ENGINEERING
        elif incident_age > timedelta(minutes=30):
            return EscalationLevel.L2_SUPPORT
        
        return None
    
    def get_escalation_contacts(self, level: EscalationLevel) -> Dict[str, Any]:
        """Get contact information for escalation level"""
        return self.escalation_contacts.get(level, {})

class IncidentManagementSystem:
    """Main incident management system for healthcare support"""
    
    def __init__(self):
        self.incidents: Dict[str, SupportIncident] = {}
        self.incident_updates: Dict[str, List[IncidentUpdate]] = defaultdict(list)
        self.escalations: Dict[str, List[EscalationRecord]] = defaultdict(list)
        self.incident_counter = 0
        self.escalation_manager = EscalationManager()
        self.active_incident_tasks = {}
        
        # Start background monitoring
        self._start_incident_monitoring()
    
    async def create_incident(
        self,
        title: str,
        description: str,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        reporter_id: str,
        reporter_name: str,
        reporter_facility: str,
        medical_context: Optional[MedicalContext] = None,
        estimated_impact: Optional[IncidentImpact] = None
    ) -> SupportIncident:
        """Create a new incident with automatic escalation assessment"""
        
        self.incident_counter += 1
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{self.incident_counter:04d}"
        
        # Auto-assess medical context if not provided
        if medical_context is None:
            medical_context = await self._auto_assess_medical_context(title, description, incident_type)
        
        # Auto-assess impact if not provided
        if estimated_impact is None:
            estimated_impact = await self._auto_assess_impact(title, description, incident_type)
        
        # Create incident
        incident = SupportIncident(
            id=incident_id,
            title=title,
            description=description,
            incident_type=incident_type,
            severity=severity,
            status=IncidentStatus.OPEN,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            resolved_at=None,
            closed_at=None,
            reporter_id=reporter_id,
            reporter_name=reporter_name,
            reporter_facility=reporter_facility,
            assigned_to=None,
            assigned_team=None,
            impact=estimated_impact,
            medical_context=medical_context,
            root_cause=None,
            resolution_summary=None,
            post_incident_review_required=severity in [IncidentSeverity.SEV1_CRITICAL, IncidentSeverity.SEV2_HIGH],
            regulatory_notification_sent=medical_context.regulatory_reporting_required,
            customer_communication_required=True
        )
        
        self.incidents[incident_id] = incident
        
        # Determine initial escalation
        initial_escalation = self.escalation_manager.determine_initial_escalation(incident)
        await self._assign_incident(incident_id, initial_escalation)
        
        # Auto-escalate if emergency
        if medical_context.emergency_situation or severity == IncidentSeverity.SEV1_CRITICAL:
            await self.escalate_incident(incident_id, EscalationLevel.EMERGENCY_MEDICAL_TEAM, "Emergency situation - immediate escalation")
        
        # Send initial notifications
        await self._send_initial_notifications(incident)
        
        logger.critical(f"INCIDENT CREATED: {incident_id} - {title} (Severity: {severity.value})")
        return incident
    
    async def update_incident(
        self,
        incident_id: str,
        updater_id: str,
        updater_name: str,
        updater_role: str,
        content: str,
        status_change: Optional[IncidentStatus] = None,
        severity_change: Optional[IncidentSeverity] = None,
        estimated_resolution_time: Optional[datetime] = None,
        internal_notes: Optional[str] = None
    ) -> IncidentUpdate:
        """Update incident with new information"""
        
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        
        # Create update record
        update = IncidentUpdate(
            id=str(len(self.incident_updates[incident_id]) + 1),
            incident_id=incident_id,
            author_id=updater_id,
            author_name=updater_name,
            author_role=updater_role,
            content=content,
            timestamp=datetime.now(),
            status_change=status_change,
            severity_change=severity_change,
            estimated_resolution_time=estimated_resolution_time,
            internal_notes=internal_notes
        )
        
        self.incident_updates[incident_id].append(update)
        
        # Update incident status
        if status_change:
            incident.status = status_change
            incident.updated_at = datetime.now()
            
            # Set resolution time
            if status_change == IncidentStatus.RESOLVED:
                incident.resolved_at = datetime.now()
            elif status_change == IncidentStatus.CLOSED:
                incident.closed_at = datetime.now()
        
        # Update severity if changed
        if severity_change:
            incident.severity = severity_change
            incident.updated_at = datetime.now()
        
        # Check for auto-escalation
        current_escalation = await self._get_current_escalation_level(incident_id)
        auto_escalation = self.escalation_manager.should_auto_escalate(incident, datetime.now())
        
        if auto_escalation and auto_escalation != current_escalation:
            await self.escalate_incident(incident_id, auto_escalation, "Automatic escalation based on time thresholds")
        
        logger.info(f"Incident {incident_id} updated by {updater_name}: {content[:100]}...")
        return update
    
    async def escalate_incident(
        self,
        incident_id: str,
        new_escalation_level: EscalationLevel,
        reason: str,
        escalated_by: Optional[str] = None
    ) -> EscalationRecord:
        """Escalate incident to higher level"""
        
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        current_level = await self._get_current_escalation_level(incident_id)
        
        # Create escalation record
        escalation = EscalationRecord(
            id=f"ESC-{incident_id}-{len(self.escalations[incident_id]) + 1}",
            incident_id=incident_id,
            from_level=current_level,
            to_level=new_escalation_level,
            reason=reason,
            escalated_by=escalated_by or "system",
            escalated_at=datetime.now()
        )
        
        self.escalations[incident_id].append(escalation)
        
        # Update incident status
        incident.status = IncidentStatus.ESCALATED
        incident.assigned_team = self._get_team_for_escalation_level(new_escalation_level)
        incident.updated_at = datetime.now()
        
        # Send escalation notifications
        await self._send_escalation_notifications(incident, escalation)
        
        logger.warning(f"INCIDENT ESCALATED: {incident_id} -> {new_escalation_level.value} (Reason: {reason})")
        return escalation
    
    async def acknowledge_escalation(self, escalation_id: str, acknowledged_by: str) -> None:
        """Acknowledge escalation"""
        for incident_id, escalations in self.escalations.items():
            for escalation in escalations:
                if escalation.id == escalation_id:
                    escalation.acknowledged_by = acknowledged_by
                    escalation.acknowledged_at = datetime.now()
                    
                    # Update incident assignment
                    if escalation.incident_id in self.incidents:
                        incident = self.incidents[escalation.incident_id]
                        incident.assigned_to = acknowledged_by
                        incident.status = IncidentStatus.INVESTIGATING
                        incident.updated_at = datetime.now()
                    
                    logger.info(f"Escalation {escalation_id} acknowledged by {acknowledged_by}")
                    return
    
    async def resolve_incident(
        self,
        incident_id: str,
        resolved_by: str,
        resolution_summary: str,
        root_cause: Optional[str] = None
    ) -> None:
        """Resolve incident"""
        
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        
        # Update incident
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now()
        incident.updated_at = datetime.now()
        incident.resolution_summary = resolution_summary
        incident.root_cause = root_cause
        
        # Add resolution update
        await self.update_incident(
            incident_id,
            resolved_by,
            resolved_by,
            "Incident Resolution",
            content=f"Incident resolved: {resolution_summary}",
            status_change=IncidentStatus.RESOLVED
        )
        
        # Send resolution notifications
        await self._send_resolution_notifications(incident)
        
        logger.info(f"INCIDENT RESOLVED: {incident_id} - {resolution_summary[:100]}...")
    
    async def get_incident_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get incident summary for the last N hours"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_incidents = [
            incident for incident in self.incidents.values()
            if incident.created_at >= cutoff_time
        ]
        
        # Calculate metrics
        total_incidents = len(recent_incidents)
        open_incidents = len([i for i in recent_incidents if i.status in [IncidentStatus.OPEN, IncidentStatus.INVESTIGATING]])
        resolved_incidents = len([i for i in recent_incidents if i.status == IncidentStatus.RESOLVED])
        
        # Severity distribution
        severity_dist = defaultdict(int)
        for incident in recent_incidents:
            severity_dist[incident.severity.value] += 1
        
        # Incident type distribution
        type_dist = defaultdict(int)
        for incident in recent_incidents:
            type_dist[incident.incident_type.value] += 1
        
        # Average resolution time
        resolved_with_time = [i for i in recent_incidents if i.resolved_at]
        avg_resolution_time = 0
        if resolved_with_time:
            total_resolution_time = sum([
                (i.resolved_at - i.created_at).total_seconds() / 3600  # Hours
                for i in resolved_with_time
            ])
            avg_resolution_time = total_resolution_time / len(resolved_with_time)
        
        # Escalation statistics
        total_escalations = sum(len(escalations) for escalations in self.escalations.values())
        
        return {
            "period_hours": hours,
            "total_incidents": total_incidents,
            "open_incidents": open_incidents,
            "resolved_incidents": resolved_incidents,
            "resolution_rate": (resolved_incidents / total_incidents * 100) if total_incidents > 0 else 0,
            "average_resolution_time_hours": avg_resolution_time,
            "severity_distribution": dict(severity_dist),
            "type_distribution": dict(type_dist),
            "total_escalations": total_escalations,
            "patient_safety_incidents": len([
                i for i in recent_incidents 
                if i.medical_context.patient_safety_impact in ["high", "critical"]
            ]),
            "emergency_situations": len([
                i for i in recent_incidents 
                if i.medical_context.emergency_situation
            ])
        }
    
    async def generate_incident_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive incident report"""
        
        # Filter incidents by date range
        period_incidents = [
            incident for incident in self.incidents.values()
            if start_date <= incident.created_at <= end_date
        ]
        
        # Get summary metrics
        summary = await self.get_incident_summary(24)  # This would be calculated for the full period
        
        # Detailed incident analysis
        incident_details = []
        for incident in period_incidents:
            incident_details.append({
                "id": incident.id,
                "title": incident.title,
                "severity": incident.severity.value,
                "status": incident.status.value,
                "type": incident.incident_type.value,
                "created_at": incident.created_at.isoformat(),
                "resolved_at": incident.resolved_at.isoformat() if incident.resolved_at else None,
                "resolution_time_hours": (
                    (incident.resolved_at - incident.created_at).total_seconds() / 3600
                    if incident.resolved_at else None
                ),
                "facility": incident.reporter_facility,
                "escalated": len(self.escalations.get(incident.id, [])),
                "medical_context": asdict(incident.medical_context),
                "impact": asdict(incident.impact)
            })
        
        # Top issues analysis
        top_incident_types = self._get_top_incident_types(period_incidents)
        top_facilities = self._get_top_incident_facilities(period_incidents)
        
        # Escalation analysis
        escalation_analysis = self._analyze_escalations(period_incidents)
        
        # Medical safety analysis
        medical_safety_analysis = self._analyze_medical_safety_incidents(period_incidents)
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": summary,
            "incident_details": incident_details,
            "trends": {
                "top_incident_types": top_incident_types,
                "top_facilities": top_facilities,
                "escalation_patterns": escalation_analysis,
                "resolution_time_trends": self._analyze_resolution_trends(period_incidents)
            },
            "medical_analysis": medical_safety_analysis,
            "recommendations": self._generate_incident_recommendations(period_incidents, summary)
        }
    
    def _start_incident_monitoring(self) -> None:
        """Start background incident monitoring tasks"""
        # This would start tasks for:
        # - Automatic escalation checking
        # - SLA monitoring
        # - Notification cleanup
        # - Report generation
        logger.info("Started incident management system")
    
    async def _auto_assess_medical_context(self, title: str, description: str, incident_type: IncidentType) -> MedicalContext:
        """Auto-assess medical context from incident description"""
        
        content_lower = f"{title} {description}".lower()
        
        # Check for patient safety keywords
        patient_safety_risk = any(keyword in content_lower for keyword in [
            "patient safety", "medical error", "clinical risk", "adverse event"
        ])
        
        # Check for emergency situations
        emergency_situation = any(keyword in content_lower for keyword in [
            "emergency", "urgent", "critical", "immediate", "life-threatening"
        ])
        
        # Check for regulatory reporting requirements
        regulatory_required = any(keyword in content_lower for keyword in [
            "medical device", "fda", "regulatory", "compliance", "safety report"
        ])
        
        # Determine clinical workflow disruption
        workflow_disruption = incident_type in [
            IncidentType.CLINICAL_WORKFLOW_DISRUPTION,
            IncidentType.EHR_INTEGRATION_FAILURE
        ]
        
        return MedicalContext(
            affected_patients=None,  # Would be extracted from actual data
            clinical_area=None,  # Would be determined from facility/department
            medical_specialty=None,  # Would be determined from context
            patient_safety_impact="high" if patient_safety_risk else "low",
            regulatory_reporting_required=regulatory_required,
            clinical_workflow_disruption=workflow_disruption,
            emergency_situation=emergency_situation
        )
    
    async def _auto_assess_impact(self, title: str, description: str, incident_type: IncidentType) -> IncidentImpact:
        """Auto-assess incident impact"""
        
        content_lower = f"{title} {description}".lower()
        
        # Estimate user count impact
        if "hospital" in content_lower or "facility" in content_lower:
            user_count = 100  # Estimated hospital users
            facility_count = 1
        elif "system" in content_lower and "down" in content_lower:
            user_count = 50  # Estimated system users
            facility_count = 1
        else:
            user_count = 10  # Small group
            facility_count = 1
        
        # Estimate downtime
        if incident_type == IncidentType.SYSTEM_OUTAGE:
            estimated_downtime = timedelta(hours=2)
        elif incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            estimated_downtime = timedelta(hours=4)
        else:
            estimated_downtime = timedelta(minutes=30)
        
        # Determine business impact
        if incident_type in [IncidentType.PATIENT_SAFETY_RISK, IncidentType.MEDICAL_DEVICE_FAILURE]:
            business_impact = "high"
        elif incident_type in [IncidentType.SYSTEM_OUTAGE, IncidentType.CLINICAL_WORKFLOW_DISRUPTION]:
            business_impact = "medium"
        else:
            business_impact = "low"
        
        return IncidentImpact(
            user_count=user_count,
            facility_count=facility_count,
            patient_count=None,  # Would be estimated based on context
            clinical_areas_affected=[],  # Would be determined from facility data
            estimated_downtime=estimated_downtime,
            business_impact=business_impact,
            regulatory_impact="none",  # Would be assessed based on incident type
            patient_safety_risk="medium" if incident_type == IncidentType.PATIENT_SAFETY_RISK else "low"
        )
    
    async def _assign_incident(self, incident_id: str, escalation_level: EscalationLevel) -> None:
        """Assign incident to appropriate team"""
        
        if incident_id in self.incidents:
            incident = self.incidents[incident_id]
            incident.assigned_team = self._get_team_for_escalation_level(escalation_level)
            incident.status = IncidentStatus.INVESTIGATING
            incident.updated_at = datetime.now()
    
    def _get_team_for_escalation_level(self, level: EscalationLevel) -> str:
        """Get team name for escalation level"""
        team_mapping = {
            EscalationLevel.L1_SUPPORT: "primary_support_team",
            EscalationLevel.L2_SUPPORT: "senior_support_team", 
            EscalationLevel.L3_ENGINEERING: "engineering_team",
            EscalationLevel.MEDICAL_SPECIALIST: "medical_specialists",
            EscalationLevel.EMERGENCY_MEDICAL_TEAM: "emergency_medical_team",
            EscalationLevel.EXECUTIVE_TEAM: "executive_team"
        }
        return team_mapping.get(level, "unknown_team")
    
    async def _get_current_escalation_level(self, incident_id: str) -> EscalationLevel:
        """Get current escalation level for incident"""
        if incident_id not in self.incidents:
            return EscalationLevel.L1_SUPPORT
        
        incident = self.incidents[incident_id]
        
        # Map team back to escalation level
        if incident.assigned_team == "emergency_medical_team":
            return EscalationLevel.EMERGENCY_MEDICAL_TEAM
        elif incident.assigned_team == "medical_specialists":
            return EscalationLevel.MEDICAL_SPECIALIST
        elif incident.assigned_team == "engineering_team":
            return EscalationLevel.L3_ENGINEERING
        elif incident.assigned_team == "senior_support_team":
            return EscalationLevel.L2_SUPPORT
        else:
            return EscalationLevel.L1_SUPPORT
    
    async def _send_initial_notifications(self, incident: SupportIncident) -> None:
        """Send initial incident notifications"""
        # In production, this would send notifications via:
        # - Email to relevant teams
        # - SMS to on-call personnel
        # - Slack/Teams notifications
        # - PagerDuty alerts
        # - Medical device integration for critical alerts
        
        logger.info(f"Initial notifications sent for incident {incident.id}")
    
    async def _send_escalation_notifications(self, incident: SupportIncident, escalation: EscalationRecord) -> None:
        """Send escalation notifications"""
        contacts = self.escalation_manager.get_escalation_contacts(escalation.to_level)
        
        logger.warning(f"Escalation notifications sent for incident {incident.id} -> {escalation.to_level.value}")
        logger.info(f"Escalation contacts: {contacts}")
    
    async def _send_resolution_notifications(self, incident: SupportIncident) -> None:
        """Send incident resolution notifications"""
        logger.info(f"Resolution notifications sent for incident {incident.id}")
    
    def _get_top_incident_types(self, incidents: List[SupportIncident]) -> List[Dict[str, Any]]:
        """Get top incident types by frequency"""
        type_counts = defaultdict(int)
        for incident in incidents:
            type_counts[incident.incident_type.value] += 1
        
        return [
            {"type": incident_type, "count": count}
            for incident_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _get_top_incident_facilities(self, incidents: List[SupportIncident]) -> List[Dict[str, Any]]:
        """Get top facilities by incident count"""
        facility_counts = defaultdict(int)
        for incident in incidents:
            facility_counts[incident.reporter_facility] += 1
        
        return [
            {"facility": facility, "count": count}
            for facility, count in sorted(facility_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    
    def _analyze_escalations(self, incidents: List[SupportIncident]) -> Dict[str, Any]:
        """Analyze escalation patterns"""
        total_incidents = len(incidents)
        incidents_with_escalations = len([
            i for i in incidents if len(self.escalations.get(i.id, [])) > 0
        ])
        
        escalation_rate = (incidents_with_escalations / total_incidents * 100) if total_incidents > 0 else 0
        
        # Escalation levels distribution
        escalation_levels = defaultdict(int)
        for incident in incidents:
            for escalation in self.escalations.get(incident.id, []):
                escalation_levels[escalation.to_level.value] += 1
        
        return {
            "escalation_rate": escalation_rate,
            "escalation_levels": dict(escalation_levels),
            "average_escalations_per_incident": (
                sum(len(self.escalations.get(i.id, [])) for i in incidents) / total_incidents
                if total_incidents > 0 else 0
            )
        }
    
    def _analyze_medical_safety_incidents(self, incidents: List[SupportIncident]) -> Dict[str, Any]:
        """Analyze medical safety-related incidents"""
        safety_incidents = [
            i for i in incidents
            if i.medical_context.patient_safety_impact in ["high", "critical"]
        ]
        
        emergency_incidents = [i for i in incidents if i.medical_context.emergency_situation]
        
        return {
            "total_safety_incidents": len(safety_incidents),
            "emergency_incidents": len(emergency_incidents),
            "safety_incident_rate": (len(safety_incidents) / len(incidents) * 100) if incidents else 0,
            "top_safety_concerns": self._identify_safety_concerns(safety_incidents)
        }
    
    def _identify_safety_concerns(self, safety_incidents: List[SupportIncident]) -> List[str]:
        """Identify common safety concerns"""
        concerns = defaultdict(int)
        
        for incident in safety_incidents:
            if incident.incident_type == IncidentType.PATIENT_SAFETY_RISK:
                concerns["patient_safety_risk"] += 1
            elif incident.incident_type == IncidentType.MEDICAL_DEVICE_FAILURE:
                concerns["medical_device_failure"] += 1
            elif incident.incident_type == IncidentType.CLINICAL_WORKFLOW_DISRUPTION:
                concerns["workflow_disruption"] += 1
        
        return [
            f"{concern}: {count} incidents"
            for concern, count in sorted(concerns.items(), key=lambda x: x[1], reverse=True)
        ]
    
    def _analyze_resolution_trends(self, incidents: List[SupportIncident]) -> Dict[str, Any]:
        """Analyze resolution time trends"""
        resolved_incidents = [i for i in incidents if i.resolved_at]
        
        if not resolved_incidents:
            return {"message": "No resolved incidents in period"}
        
        resolution_times = [
            (i.resolved_at - i.created_at).total_seconds() / 3600  # Hours
            for i in resolved_incidents
        ]
        
        return {
            "average_resolution_time_hours": statistics.mean(resolution_times),
            "median_resolution_time_hours": statistics.median(resolution_times),
            "fastest_resolution_hours": min(resolution_times),
            "slowest_resolution_hours": max(resolution_times),
            "resolution_time_by_severity": self._analyze_resolution_by_severity(resolved_incidents)
        }
    
    def _analyze_resolution_by_severity(self, incidents: List[SupportIncident]) -> Dict[str, float]:
        """Analyze resolution times by severity"""
        severity_times = defaultdict(list)
        
        for incident in incidents:
            resolution_hours = (incident.resolved_at - incident.created_at).total_seconds() / 3600
            severity_times[incident.severity.value].append(resolution_hours)
        
        return {
            severity: statistics.mean(times)
            for severity, times in severity_times.items()
        }
    
    def _generate_incident_recommendations(self, incidents: List[SupportIncident], summary: Dict[str, Any]) -> List[str]:
        """Generate incident management recommendations"""
        recommendations = []
        
        # High escalation rate
        if summary.get("total_escalations", 0) > summary.get("total_incidents", 1) * 0.5:
            recommendations.append(
                "High escalation rate detected. Consider improving first-level support capabilities."
            )
        
        # Patient safety incidents
        if summary.get("patient_safety_incidents", 0) > 0:
            recommendations.append(
                "Patient safety incidents require immediate attention. Review safety protocols and preventive measures."
            )
        
        # Long resolution times
        if summary.get("average_resolution_time_hours", 0) > 24:
            recommendations.append(
                "Long average resolution times. Consider improving incident response procedures and tools."
            )
        
        # Emergency situations
        if summary.get("emergency_situations", 0) > 0:
            recommendations.append(
                "Emergency situations detected. Review emergency response procedures and escalation paths."
            )
        
        # Common incident types
        top_types = self._get_top_incident_types(incidents)
        if top_types:
            top_type = top_types[0]
            recommendations.append(
                f"Most common incident type: {top_type['type']} ({top_type['count']} incidents). "
                "Consider preventive measures for this issue type."
            )
        
        return recommendations

# Global incident management system instance
incident_system = IncidentManagementSystem()

# Example usage and testing functions
async def create_sample_incidents():
    """Create sample incidents for testing"""
    
    # Critical medical device failure
    medical_context = MedicalContext(
        affected_patients=5,
        clinical_area="Cardiology",
        medical_specialty="Cardiology",
        patient_safety_impact="high",
        regulatory_reporting_required=True,
        clinical_workflow_disruption=True,
        emergency_situation=True
    )
    
    critical_incident = await incident_system.create_incident(
        title="Cardiac monitoring device failure in ICU",
        description="Multiple cardiac monitoring devices showing error states. Patient safety risk identified.",
        incident_type=IncidentType.MEDICAL_DEVICE_FAILURE,
        severity=IncidentSeverity.SEV1_CRITICAL,
        reporter_id="dr_wilson_001",
        reporter_name="Dr. Lisa Wilson",
        reporter_facility="General Hospital ICU",
        medical_context=medical_context
    )
    
    # Standard system outage
    standard_incident = await incident_system.create_incident(
        title="Database connection timeout affecting patient records",
        description="Intermittent database connection issues preventing access to patient records.",
        incident_type=IncidentType.SYSTEM_OUTAGE,
        severity=IncidentSeverity.SEV2_HIGH,
        reporter_id="nurse_johnson_002",
        reporter_name="Nurse Johnson",
        reporter_facility="City Medical Center"
    )
    
    print(f"Created incidents: {critical_incident.id}, {standard_incident.id}")
    
    # Update critical incident
    await incident_system.update_incident(
        critical_incident.id,
        "tech_lead_001",
        "Technical Lead",
        "Senior Engineer",
        "Identified root cause: Network configuration issue. Implementing fix now.",
        estimated_resolution_time=datetime.now() + timedelta(minutes=30)
    )
    
    # Escalate if needed
    summary = await incident_system.get_incident_summary(1)
    print(f"Incident summary: {summary}")

if __name__ == "__main__":
    asyncio.run(create_sample_incidents())