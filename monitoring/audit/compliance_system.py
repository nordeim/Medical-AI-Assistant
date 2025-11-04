"""
Comprehensive Audit Trail and Compliance Reporting System
Provides HIPAA, FDA, and regulatory compliance tracking with comprehensive
audit logging, evidence collection, and regulatory reporting capabilities.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from pathlib import Path
import sqlite3
import csv
import yaml
from cryptography.fernet import Fernet
import base64

class AuditEventType(Enum):
    """Types of audit events"""
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    USER_AUTHENTICATION = "user_authentication"
    SYSTEM_ACCESS = "system_access"
    CLINICAL_DECISION = "clinical_decision"
    AI_RECOMMENDATION = "ai_recommendation"
    PATIENT_INTERACTION = "patient_interaction"
    SYSTEM_CONFIGURATION = "system_configuration"
    BACKUP_OPERATION = "backup_operation"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"

class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    HIPAA = "hipaa"
    FDA_21CFR_PART11 = "fda_21cfr_part11"
    ISO_27001 = "iso_27001"
    SOC2 = "soc2"
    GDPR = "gdpr"

@dataclass
class AuditEvent:
    """Audit event record"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    user_role: Optional[str]
    patient_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    resource_accessed: Optional[str]
    action_performed: str
    outcome: str  # success, failure, warning
    details: Dict[str, Any]
    compliance_tags: List[ComplianceFramework]
    retention_until: datetime
    cryptographic_hash: str
    digital_signature: Optional[str] = None
    
    def __post_init__(self):
        # Generate event ID if not provided
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        # Calculate retention period based on compliance requirements
        if not self.retention_until:
            self.retention_until = self._calculate_retention_period()
        
        # Generate cryptographic hash
        if not self.cryptographic_hash:
            self.cryptographic_hash = self._generate_hash()
    
    def _calculate_retention_period(self) -> datetime:
        """Calculate retention period based on compliance requirements"""
        # HIPAA: 6 years minimum
        if ComplianceFramework.HIPAA in self.compliance_tags:
            return self.timestamp + timedelta(days=2190)  # 6 years
        
        # FDA 21 CFR Part 11: Data integrity and records
        if ComplianceFramework.FDA_21CFR_PART11 in self.compliance_tags:
            return self.timestamp + timedelta(days=5475)  # 15 years
        
        # Default: 7 years
        return self.timestamp + timedelta(days=2555)  # 7 years
    
    def _generate_hash(self) -> str:
        """Generate cryptographic hash of event data"""
        event_data = json.dumps({
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'action_performed': self.action_performed,
            'outcome': self.outcome,
            'details': self.details
        }, sort_keys=True)
        
        return hashlib.sha256(event_data.encode()).hexdigest()

@dataclass
class ComplianceReport:
    """Compliance report structure"""
    report_id: str
    framework: ComplianceFramework
    generated_at: datetime
    report_period_start: datetime
    report_period_end: datetime
    total_events: int
    events_by_category: Dict[str, int]
    compliance_violations: List[Dict[str, Any]]
    security_incidents: List[Dict[str, Any]]
    data_access_summary: Dict[str, Any]
    system_integrity_checks: Dict[str, Any]
    recommendations: List[str]
    evidence_locations: List[str]

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self, 
                 database_path: str = "audit_logs.db",
                 encryption_key: Optional[bytes] = None,
                 auto_backup: bool = True):
        """
        Initialize audit logger
        
        Args:
            database_path: Path to SQLite database for audit logs
            encryption_key: Key for encrypting sensitive audit data
            auto_backup: Whether to automatically backup audit logs
        """
        self.database_path = database_path
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.auto_backup = auto_backup
        
        # Initialize database
        self._initialize_database()
        
        # Compliance configurations
        self.compliance_configs = self._load_compliance_configs()
        
        # Statistics
        self.stats = {
            'total_events_logged': 0,
            'events_by_type': {event_type.value: 0 for event_type in AuditEventType},
            'events_by_framework': {framework.value: 0 for framework in ComplianceFramework},
            'compliance_violations': 0,
            'security_incidents': 0
        }
    
    def _initialize_database(self) -> None:
        """Initialize SQLite database for audit logs"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create audit_events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    user_role TEXT,
                    patient_id TEXT,
                    session_id TEXT,
                    ip_address TEXT,
                    resource_accessed TEXT,
                    action_performed TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    details TEXT,
                    compliance_tags TEXT,
                    retention_until TEXT,
                    cryptographic_hash TEXT,
                    digital_signature TEXT,
                    encrypted_data TEXT
                )
            ''')
            
            # Create compliance_reports table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_reports (
                    report_id TEXT PRIMARY KEY,
                    framework TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    report_period_start TEXT NOT NULL,
                    report_period_end TEXT NOT NULL,
                    report_data TEXT NOT NULL
                )
            ''')
            
            # Create indices for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON audit_events(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_patient_id ON audit_events(patient_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_retention ON audit_events(retention_until)')
            
            conn.commit()
            conn.close()
            
            logging.info("Audit database initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing audit database: {str(e)}")
            raise
    
    def _load_compliance_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance framework configurations"""
        return {
            'hipaa': {
                'required_retention_days': 2190,  # 6 years
                'required_logging_events': [
                    'data_access', 'data_modification', 'user_authentication',
                    'system_access', 'security_event'
                ],
                'privacy_protection': True,
                'access_control_monitoring': True,
                'audit_trail_requirements': True
            },
            'fda_21cfr_part11': {
                'required_retention_days': 5475,  # 15 years
                'required_logging_events': [
                    'data_access', 'data_modification', 'system_access',
                    'clinical_decision', 'system_configuration'
                ],
                'electronic_signatures': True,
                'data_integrity': True,
                'audit_trail_requirements': True
            },
            'iso_27001': {
                'required_retention_days': 2555,  # 7 years
                'required_logging_events': [
                    'system_access', 'security_event', 'user_authentication',
                    'system_configuration'
                ],
                'access_control_monitoring': True,
                'security_incident_tracking': True,
                'audit_trail_requirements': True
            }
        }
    
    def log_event(self,
                  event_type: AuditEventType,
                  action_performed: str,
                  outcome: str,
                  user_id: Optional[str] = None,
                  user_role: Optional[str] = None,
                  patient_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  resource_accessed: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None,
                  compliance_frameworks: Optional[List[ComplianceFramework]] = None) -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of audit event
            action_performed: Description of action performed
            outcome: Outcome of the action (success, failure, warning)
            user_id: ID of user performing action
            user_role: Role of user performing action
            patient_id: ID of patient (if applicable)
            session_id: Session identifier
            ip_address: IP address of user
            resource_accessed: Resource that was accessed
            details: Additional event details
            compliance_frameworks: Applicable compliance frameworks
            
        Returns:
            Event ID of the logged event
        """
        try:
            # Prepare details
            if details is None:
                details = {}
            
            # Add metadata
            details['system_info'] = {
                'hostname': 'medical-ai-system',
                'version': '1.0.0',
                'timestamp': datetime.now().isoformat()
            }
            
            # Create audit event
            event = AuditEvent(
                event_id="",
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                user_role=user_role,
                patient_id=patient_id,
                session_id=session_id,
                ip_address=ip_address,
                resource_accessed=resource_accessed,
                action_performed=action_performed,
                outcome=outcome,
                details=details,
                compliance_tags=compliance_frameworks or []
            )
            
            # Store event in database
            self._store_event(event)
            
            # Update statistics
            self._update_statistics(event)
            
            # Check for compliance violations
            self._check_compliance_violations(event)
            
            # Auto-backup if enabled
            if self.auto_backup:
                self._auto_backup_if_needed()
            
            logging.info(f"Audit event logged: {event.event_id} - {event_type.value}")
            return event.event_id
            
        except Exception as e:
            logging.error(f"Error logging audit event: {str(e)}")
            raise
    
    def _store_event(self, event: AuditEvent) -> None:
        """Store event in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Encrypt sensitive data if needed
            encrypted_data = None
            if event.patient_id or event.details:
                sensitive_data = {
                    'patient_id': event.patient_id,
                    'details': event.details
                }
                encrypted_data = self.cipher_suite.encrypt(json.dumps(sensitive_data).encode()).decode()
            
            # Store event
            cursor.execute('''
                INSERT OR REPLACE INTO audit_events 
                (event_id, event_type, timestamp, user_id, user_role, patient_id, session_id,
                 ip_address, resource_accessed, action_performed, outcome, details, compliance_tags,
                 retention_until, cryptographic_hash, digital_signature, encrypted_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.user_id,
                event.user_role,
                None,  # Store patient_id separately for encryption
                event.session_id,
                event.ip_address,
                event.resource_accessed,
                event.action_performed,
                event.outcome,
                json.dumps(event.details),
                ','.join([tag.value for tag in event.compliance_tags]),
                event.retention_until.isoformat(),
                event.cryptographic_hash,
                event.digital_signature,
                encrypted_data
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error storing audit event: {str(e)}")
            raise
    
    def _update_statistics(self, event: AuditEvent) -> None:
        """Update audit logging statistics"""
        self.stats['total_events_logged'] += 1
        
        # Update event type statistics
        event_type_str = event.event_type.value
        if event_type_str in self.stats['events_by_type']:
            self.stats['events_by_type'][event_type_str] += 1
        
        # Update compliance framework statistics
        for framework in event.compliance_tags:
            framework_str = framework.value
            if framework_str in self.stats['events_by_framework']:
                self.stats['events_by_framework'][framework_str] += 1
    
    def _check_compliance_violations(self, event: AuditEvent) -> None:
        """Check for compliance violations based on the event"""
        try:
            violations = []
            
            # Check HIPAA compliance
            if ComplianceFramework.HIPAA in event.compliance_tags:
                violations.extend(self._check_hipaa_compliance(event))
            
            # Check FDA compliance
            if ComplianceFramework.FDA_21CFR_PART11 in event.compliance_tags:
                violations.extend(self._check_fda_compliance(event))
            
            # Log violations
            for violation in violations:
                self.log_event(
                    event_type=AuditEventType.COMPLIANCE_CHECK,
                    action_performed=f"Compliance violation detected: {violation['type']}",
                    outcome='warning',
                    details=violation,
                    compliance_frameworks=[ComplianceFramework.HIPAA, ComplianceFramework.FDA_21CFR_PART11]
                )
                
                self.stats['compliance_violations'] += 1
                
        except Exception as e:
            logging.error(f"Error checking compliance violations: {str(e)}")
    
    def _check_hipaa_compliance(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Check HIPAA compliance requirements"""
        violations = []
        
        # Check if PHI access is properly logged
        if event.event_type in [AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION]:
            if event.patient_id and not event.user_id:
                violations.append({
                    'type': 'missing_user_identification',
                    'event_id': event.event_id,
                    'description': 'PHI access without user identification',
                    'severity': 'high',
                    'framework': 'hipaa'
                })
        
        # Check for unauthorized access attempts
        if event.outcome == 'failure' and event.event_type == AuditEventType.USER_AUTHENTICATION:
            # Count failures in short time window (simplified)
            failures_count = self._count_recent_failures(event.ip_address, minutes=15)
            if failures_count > 5:
                violations.append({
                    'type': 'multiple_authentication_failures',
                    'event_id': event.event_id,
                    'ip_address': event.ip_address,
                    'failure_count': failures_count,
                    'description': 'Multiple authentication failures detected',
                    'severity': 'critical',
                    'framework': 'hipaa'
                })
                self.stats['security_incidents'] += 1
        
        return violations
    
    def _check_fda_compliance(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Check FDA 21 CFR Part 11 compliance requirements"""
        violations = []
        
        # Check for electronic signatures
        if event.event_type == AuditEventType.CLINICAL_DECISION:
            if not event.digital_signature:
                violations.append({
                    'type': 'missing_electronic_signature',
                    'event_id': event.event_id,
                    'description': 'Clinical decision without electronic signature',
                    'severity': 'critical',
                    'framework': 'fda_21cfr_part11'
                })
        
        # Check data integrity
        if event.event_type == AuditEventType.DATA_MODIFICATION:
            if not event.cryptographic_hash:
                violations.append({
                    'type': 'missing_data_integrity_check',
                    'event_id': event.event_id,
                    'description': 'Data modification without integrity verification',
                    'severity': 'high',
                    'framework': 'fda_21cfr_part11'
                })
        
        return violations
    
    def _count_recent_failures(self, ip_address: Optional[str], minutes: int) -> int:
        """Count recent authentication failures from an IP"""
        if not ip_address:
            return 0
        
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            
            cursor.execute('''
                SELECT COUNT(*) FROM audit_events
                WHERE event_type = ? AND outcome = 'failure' 
                AND ip_address = ? AND timestamp > ?
            ''', (AuditEventType.USER_AUTHENTICATION.value, ip_address, cutoff_time.isoformat()))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception:
            return 0
    
    def _auto_backup_if_needed(self) -> None:
        """Automatically backup audit logs if needed"""
        try:
            # Check if backup is needed (e.g., every 1000 events or daily)
            if self.stats['total_events_logged'] % 1000 == 0:
                backup_path = f"audit_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                self.create_backup(backup_path)
                
        except Exception as e:
            logging.error(f"Error in auto-backup: {str(e)}")
    
    def create_backup(self, backup_path: str) -> None:
        """Create backup of audit logs"""
        try:
            # Simple file copy backup (in production, use proper backup strategies)
            import shutil
            shutil.copy2(self.database_path, backup_path)
            
            logging.info(f"Audit log backup created: {backup_path}")
            
        except Exception as e:
            logging.error(f"Error creating backup: {str(e)}")
            raise
    
    def generate_compliance_report(self,
                                 framework: ComplianceFramework,
                                 start_date: datetime,
                                 end_date: datetime,
                                 report_format: str = 'json') -> ComplianceReport:
        """
        Generate compliance report for specified framework and time period
        
        Args:
            framework: Compliance framework to report on
            start_date: Start of reporting period
            end_date: End of reporting period
            report_format: Format of report ('json', 'csv', 'pdf')
            
        Returns:
            ComplianceReport object
        """
        try:
            # Query events for the period
            events = self._query_events_by_date_range(start_date, end_date)
            
            # Filter events by framework
            framework_events = [
                event for event in events
                if framework.value in event['compliance_tags']
            ]
            
            # Analyze events
            analysis = self._analyze_events_for_compliance(framework_events, framework)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(analysis, framework)
            
            # Create report
            report = ComplianceReport(
                report_id=str(uuid.uuid4()),
                framework=framework,
                generated_at=datetime.now(),
                report_period_start=start_date,
                report_period_end=end_date,
                total_events=len(framework_events),
                events_by_category=analysis['events_by_category'],
                compliance_violations=analysis['violations'],
                security_incidents=analysis['security_incidents'],
                data_access_summary=analysis['data_access_summary'],
                system_integrity_checks=analysis['integrity_checks'],
                recommendations=recommendations,
                evidence_locations=[self.database_path]
            )
            
            # Store report
            self._store_compliance_report(report)
            
            # Export report in requested format
            if report_format.lower() != 'json':
                self._export_report(report, report_format)
            
            logging.info(f"Generated {framework.value} compliance report: {report.report_id}")
            return report
            
        except Exception as e:
            logging.error(f"Error generating compliance report: {str(e)}")
            raise
    
    def _query_events_by_date_range(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Query audit events within date range"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM audit_events
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_date.isoformat(), end_date.isoformat()))
            
            columns = [description[0] for description in cursor.description]
            events = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return events
            
        except Exception as e:
            logging.error(f"Error querying events by date range: {str(e)}")
            return []
    
    def _analyze_events_for_compliance(self, 
                                     events: List[Dict[str, Any]], 
                                     framework: ComplianceFramework) -> Dict[str, Any]:
        """Analyze events for compliance requirements"""
        analysis = {
            'events_by_category': {},
            'violations': [],
            'security_incidents': [],
            'data_access_summary': {},
            'integrity_checks': {}
        }
        
        try:
            # Categorize events
            for event in events:
                event_type = event['event_type']
                if event_type not in analysis['events_by_category']:
                    analysis['events_by_category'][event_type] = 0
                analysis['events_by_category'][event_type] += 1
            
            # Analyze data access patterns
            data_access_events = [
                event for event in events
                if event['event_type'] in ['data_access', 'data_modification']
            ]
            
            analysis['data_access_summary'] = {
                'total_data_events': len(data_access_events),
                'unique_users': len(set(event['user_id'] for event in data_access_events if event['user_id'])),
                'unique_patients': len(set(event['patient_id'] for event in data_access_events if event['patient_id'])),
                'failed_access_attempts': len([e for e in data_access_events if e['outcome'] == 'failure'])
            }
            
            # Check for violations (simplified analysis)
            for event in events:
                if event['outcome'] == 'failure':
                    if 'authentication' in event['action_performed'].lower():
                        analysis['security_incidents'].append({
                            'event_id': event['event_id'],
                            'type': 'authentication_failure',
                            'timestamp': event['timestamp'],
                            'user_id': event['user_id'],
                            'ip_address': event['ip_address']
                        })
            
            # System integrity checks
            analysis['integrity_checks'] = {
                'events_with_hashes': len([e for e in events if e['cryptographic_hash']]),
                'events_with_signatures': len([e for e in events if e['digital_signature']]),
                'total_events': len(events)
            }
            
        except Exception as e:
            logging.error(f"Error analyzing events for compliance: {str(e)}")
        
        return analysis
    
    def _generate_compliance_recommendations(self,
                                           analysis: Dict[str, Any],
                                           framework: ComplianceFramework) -> List[str]:
        """Generate compliance recommendations based on analysis"""
        recommendations = []
        
        try:
            # General recommendations based on framework
            if framework == ComplianceFramework.HIPAA:
                # HIPAA-specific recommendations
                if analysis['data_access_summary']['failed_access_attempts'] > 10:
                    recommendations.append("Review and strengthen access control mechanisms due to high failure rate")
                
                if analysis['data_access_summary']['unique_patients'] > 0 and analysis['data_access_summary']['unique_users'] < 3:
                    recommendations.append("Increase number of authorized users for patient data access")
                
                recommendations.extend([
                    "Ensure all PHI access is properly logged and monitored",
                    "Implement regular access reviews and user access certifications",
                    "Maintain comprehensive audit trail for 6 years minimum"
                ])
            
            elif framework == ComplianceFramework.FDA_21CFR_PART11:
                # FDA-specific recommendations
                integrity_ratio = analysis['integrity_checks']['events_with_signatures'] / max(analysis['integrity_checks']['total_events'], 1)
                if integrity_ratio < 0.9:
                    recommendations.append("Increase electronic signature coverage for critical decisions")
                
                recommendations.extend([
                    "Implement robust data integrity controls",
                    "Ensure all clinical decisions have proper electronic signatures",
                    "Maintain 15-year record retention for regulated data"
                ])
            
            # General recommendations
            total_violations = len(analysis['violations']) + len(analysis['security_incidents'])
            if total_violations > 0:
                recommendations.append(f"Address {total_violations} compliance violations and security incidents identified")
            
            if not recommendations:
                recommendations.append("No specific recommendations - compliance appears adequate")
                
        except Exception as e:
            logging.error(f"Error generating compliance recommendations: {str(e)}")
            recommendations.append("Unable to generate specific recommendations due to analysis error")
        
        return recommendations
    
    def _store_compliance_report(self, report: ComplianceReport) -> None:
        """Store compliance report in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            report_data = asdict(report)
            report_data['events_by_category'] = json.dumps(report.events_by_category)
            report_data['compliance_violations'] = json.dumps(report.compliance_violations)
            report_data['security_incidents'] = json.dumps(report.security_incidents)
            report_data['data_access_summary'] = json.dumps(report.data_access_summary)
            report_data['system_integrity_checks'] = json.dumps(report.system_integrity_checks)
            report_data['recommendations'] = json.dumps(report.recommendations)
            report_data['evidence_locations'] = json.dumps(report.evidence_locations)
            
            cursor.execute('''
                INSERT OR REPLACE INTO compliance_reports
                (report_id, framework, generated_at, report_period_start, 
                 report_period_end, report_data)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                report.report_id,
                report.framework.value,
                report.generated_at.isoformat(),
                report.report_period_start.isoformat(),
                report.report_period_end.isoformat(),
                json.dumps(report_data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error storing compliance report: {str(e)}")
            raise
    
    def _export_report(self, report: ComplianceReport, format_type: str) -> str:
        """Export compliance report in specified format"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"compliance_report_{report.framework.value}_{timestamp}"
            
            if format_type.lower() == 'csv':
                filename = f"{base_filename}.csv"
                self._export_as_csv(report, filename)
            elif format_type.lower() == 'yaml':
                filename = f"{base_filename}.yaml"
                self._export_as_yaml(report, filename)
            else:
                filename = f"{base_filename}.json"
                self._export_as_json(report, filename)
            
            return filename
            
        except Exception as e:
            logging.error(f"Error exporting report: {str(e)}")
            raise
    
    def _export_as_json(self, report: ComplianceReport, filename: str) -> None:
        """Export report as JSON"""
        report_dict = asdict(report)
        
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
    
    def _export_as_csv(self, report: ComplianceReport, filename: str) -> None:
        """Export report as CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Compliance Report', report.framework.value])
            writer.writerow(['Generated At', report.generated_at])
            writer.writerow(['Report Period', f"{report.report_period_start} to {report.report_period_end}"])
            writer.writerow([])
            
            # Write summary
            writer.writerow(['Total Events', report.total_events])
            writer.writerow(['Compliance Violations', len(report.compliance_violations)])
            writer.writerow(['Security Incidents', len(report.security_incidents)])
            writer.writerow([])
            
            # Write events by category
            writer.writerow(['Events by Category'])
            for category, count in report.events_by_category.items():
                writer.writerow([category, count])
            writer.writerow([])
            
            # Write recommendations
            writer.writerow(['Recommendations'])
            for i, rec in enumerate(report.recommendations, 1):
                writer.writerow([f"{i}.", rec])
    
    def _export_as_yaml(self, report: ComplianceReport, filename: str) -> None:
        """Export report as YAML"""
        report_dict = asdict(report)
        
        with open(filename, 'w') as f:
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)
    
    def query_audit_logs(self,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        event_types: Optional[List[AuditEventType]] = None,
                        user_id: Optional[str] = None,
                        patient_id: Optional[str] = None,
                        limit: Optional[int] = 1000) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters
        
        Args:
            start_date: Start date for query
            end_date: End date for query
            event_types: Filter by event types
            user_id: Filter by user ID
            patient_id: Filter by patient ID
            limit: Maximum number of results
            
        Returns:
            List of audit events matching criteria
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM audit_events WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if event_types:
                placeholders = ','.join(['?' for _ in event_types])
                query += f" AND event_type IN ({placeholders})"
                params.extend([event_type.value for event_type in event_types])
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if patient_id:
                query += " AND patient_id = ?"
                params.append(patient_id)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            columns = [description[0] for description in cursor.description]
            events = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            
            logging.info(f"Retrieved {len(events)} audit events")
            return events
            
        except Exception as e:
            logging.error(f"Error querying audit logs: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        return {
            'database_size_mb': self._get_database_size(),
            'total_events': self.stats['total_events_logged'],
            'events_by_type': self.stats['events_by_type'],
            'events_by_framework': self.stats['events_by_framework'],
            'compliance_violations': self.stats['compliance_violations'],
            'security_incidents': self.stats['security_incidents']
        }
    
    def _get_database_size(self) -> float:
        """Get database file size in MB"""
        try:
            db_path = Path(self.database_path)
            if db_path.exists():
                return db_path.stat().st_size / (1024 * 1024)  # Convert to MB
            return 0.0
        except Exception:
            return 0.0

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize audit logger
    audit_logger = AuditLogger(database_path="test_audit.db")
    
    # Add medical AI alert rules
    audit_logger.add_medical_alert_rules()
    
    # Example audit events
    events_to_log = [
        {
            'event_type': AuditEventType.USER_AUTHENTICATION,
            'action_performed': 'User login attempt',
            'outcome': 'success',
            'user_id': 'dr_smith',
            'user_role': 'physician',
            'ip_address': '192.168.1.100',
            'compliance_frameworks': [ComplianceFramework.HIPAA]
        },
        {
            'event_type': AuditEventType.DATA_ACCESS,
            'action_performed': 'Patient chart access',
            'outcome': 'success',
            'user_id': 'dr_smith',
            'user_role': 'physician',
            'patient_id': 'PATIENT_123',
            'ip_address': '192.168.1.100',
            'details': {'chart_type': 'full', 'access_reason': 'treatment'},
            'compliance_frameworks': [ComplianceFramework.HIPAA, ComplianceFramework.FDA_21CFR_PART11]
        },
        {
            'event_type': AuditEventType.CLINICAL_DECISION,
            'action_performed': 'AI diagnosis recommendation',
            'outcome': 'success',
            'user_id': 'dr_smith',
            'user_role': 'physician',
            'patient_id': 'PATIENT_123',
            'ip_address': '192.168.1.100',
            'details': {'diagnosis': 'Diabetes Type 2', 'confidence': 0.94, 'ai_model': 'medical_diagnosis_v1'},
            'compliance_frameworks': [ComplianceFramework.HIPAA, ComplianceFramework.FDA_21CFR_PART11]
        }
    ]
    
    print("=== Logging Audit Events ===")
    event_ids = []
    
    for event_data in events_to_log:
        event_id = audit_logger.log_event(**event_data)
        event_ids.append(event_id)
        print(f"Logged event: {event_id} - {event_data['event_type'].value}")
    
    # Generate compliance report
    print("\n=== Generating Compliance Reports ===")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    for framework in [ComplianceFramework.HIPAA, ComplianceFramework.FDA_21CFR_PART11]:
        try:
            report = audit_logger.generate_compliance_report(
                framework=framework,
                start_date=start_date,
                end_date=end_date,
                report_format='json'
            )
            
            print(f"\n{framework.value.upper()} Compliance Report:")
            print(f"  Report ID: {report.report_id}")
            print(f"  Total Events: {report.total_events}")
            print(f"  Violations: {len(report.compliance_violations)}")
            print(f"  Security Incidents: {len(report.security_incidents)}")
            print(f"  Recommendations: {len(report.recommendations)}")
            
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"    {i}. {rec}")
                
        except Exception as e:
            print(f"Error generating {framework.value} report: {str(e)}")
    
    # Query audit logs
    print("\n=== Querying Audit Logs ===")
    
    recent_events = audit_logger.query_audit_logs(
        start_date=start_date,
        end_date=end_date,
        limit=10
    )
    
    print(f"Retrieved {len(recent_events)} events from last 30 days")
    
    # Show statistics
    print("\n=== Audit System Statistics ===")
    stats = audit_logger.get_statistics()
    for key, value in stats.items():
        if key != 'events_by_type':
            print(f"{key}: {value}")
    
    print("\n=== Events by Type ===")
    for event_type, count in stats['events_by_type'].items():
        if count > 0:
            print(f"{event_type}: {count}")
    
    print("\nAudit logging system demonstration completed.")