"""
Clinical Outcome Tracking System

Provides comprehensive tracking of clinical outcomes, medical effectiveness measurement,
and regulatory compliance monitoring for medical AI systems.
"""

import asyncio
import json
import numpy as np
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import structlog
from enum import Enum

from ...config.logging_config import get_logger

logger = structlog.get_logger("clinical_tracking")


class OutcomeType(Enum):
    """Types of clinical outcomes."""
    DIAGNOSIS = "diagnosis"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    PROGNOSIS = "prognosis"
    TRIAGE = "triage"
    MEDICATION_ADJUSTMENT = "medication_adjustment"
    SURGICAL_DECISION = "surgical_decision"
    MONITORING_ALERT = "monitoring_alert"
    PREVENTIVE_CARE = "preventive_care"


class OutcomeResult(Enum):
    """Clinical outcome results."""
    TRUE_POSITIVE = "true_positive"      # Correct positive prediction
    TRUE_NEGATIVE = "true_negative"      # Correct negative prediction
    FALSE_POSITIVE = "false_positive"    # Incorrect positive prediction
    FALSE_NEGATIVE = "false_negative"    # Incorrect negative prediction
    PARTIAL_CORRECT = "partial_correct"  # Partially correct prediction
    INCONCLUSIVE = "inconclusive"        # Unable to determine
    PENDING = "pending"                  # Outcome pending


class SeverityLevel(Enum):
    """Medical condition severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    PREVENTIVE = "preventive"


@dataclass
class ClinicalOutcome:
    """Clinical outcome tracking record."""
    # Identification
    outcome_id: str
    model_id: str
    patient_id: str  # Hash or pseudonymized
    prediction_id: str
    
    # Outcome information
    outcome_type: OutcomeType
    severity_level: SeverityLevel
    medical_specialty: str
    clinical_context: Dict[str, Any]
    
    # Prediction details
    predicted_outcome: Any
    confidence_score: float
    model_version: str
    prediction_timestamp: float
    
    # Follow-up and validation
    follow_up_duration_days: int
    validation_timestamp: Optional[float] = None
    validator_id: Optional[str] = None
    
    # Outcome assessment
    outcome_result: Optional[OutcomeResult] = None
    clinical_effectiveness_score: Optional[float] = None
    medical_relevance_score: Optional[float] = None
    safety_score: Optional[float] = None
    bias_score: Optional[float] = None
    
    # Impact metrics
    patient_impact: Optional[str] = None  # positive, negative, neutral, unknown
    cost_impact: Optional[float] = None   # Healthcare cost change
    time_impact: Optional[float] = None   # Time savings or delays
    
    # Feedback
    physician_feedback: Optional[str] = None
    patient_feedback: Optional[str] = None
    quality_score: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_timestamp: float = field(default_factory=time.time)


@dataclass
class MedicalEffectivenessMetrics:
    """Medical effectiveness measurement metrics."""
    model_id: str
    calculation_timestamp: float = field(default_factory=time.time)
    
    # Overall effectiveness
    overall_effectiveness_score: float = 0.0
    clinical_accuracy: float = 0.0
    medical_relevance: float = 0.0
    safety_score: float = 0.0
    bias_score: float = 0.0
    
    # Outcome-specific metrics
    diagnosis_accuracy: float = 0.0
    treatment_recommendation_accuracy: float = 0.0
    prognosis_accuracy: float = 0.0
    triage_accuracy: float = 0.0
    
    # Specialty-specific performance
    specialty_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Patient outcome improvements
    patient_benefit_rate: float = 0.0
    adverse_event_rate: float = 0.0
    cost_effectiveness: float = 0.0
    time_effectiveness: float = 0.0
    
    # Regulatory compliance
    regulatory_compliance_score: float = 0.0
    audit_readiness: float = 0.0
    documentation_completeness: float = 0.0
    
    # Sample information
    total_outcomes_tracked: int = 0
    validated_outcomes: int = 0
    pending_outcomes: int = 0
    time_window_days: int = 0
    
    # Confidence intervals
    effectiveness_ci_lower: float = 0.0
    effectiveness_ci_upper: float = 0.0
    safety_ci_lower: float = 0.0
    safety_ci_upper: float = 0.0


class RegulatoryComplianceMonitor:
    """Monitor regulatory compliance for medical AI systems."""
    
    def __init__(self):
        self.logger = structlog.get_logger("regulatory_monitor")
        
        # Compliance frameworks
        self.frameworks = {
            'hipaa': {
                'name': 'Health Insurance Portability and Accountability Act',
                'checks': [
                    'phi_protection',
                    'data_encryption',
                    'audit_logging',
                    'access_control',
                    'data_retention'
                ]
            },
            'fda': {
                'name': 'Food and Drug Administration (SaMD)',
                'checks': [
                    'clinical_validation',
                    'risk_assessment',
                    'software_lifecycle',
                    'cybersecurity',
                    'post_market_surveillance'
                ]
            },
            'gdpr': {
                'name': 'General Data Protection Regulation',
                'checks': [
                    'consent_management',
                    'data_minimization',
                    'right_to_deletion',
                    'data_portability',
                    'privacy_by_design'
                ]
            },
            'iso_27001': {
                'name': 'ISO 27001 Information Security',
                'checks': [
                    'information_security_policy',
                    'risk_management',
                    'access_control',
                    'cryptographic_controls',
                    'incident_management'
                ]
            },
            'iec_62304': {
                'name': 'IEC 62304 Medical Device Software',
                'checks': [
                    'software_classification',
                    'development_process',
                    'risk_management_file',
                    'maintenance_process',
                    'problem_resolution'
                ]
            }
        }
        
        # Compliance tracking
        self.compliance_status: Dict[str, Dict[str, bool]] = defaultdict(lambda: defaultdict(bool))
        self.violations_log: List[Dict[str, Any]] = []
        self.audit_readiness_score = 0.0
        
        self.logger.info("RegulatoryComplianceMonitor initialized")
    
    def check_compliance(self, framework: str, check_type: str, context: Dict[str, Any] = None) -> bool:
        """Check compliance for a specific framework and check type."""
        try:
            if framework not in self.frameworks:
                self.logger.warning(f"Unknown compliance framework: {framework}")
                return False
            
            if check_type not in self.frameworks[framework]['checks']:
                self.logger.warning(f"Unknown check type {check_type} for framework {framework}")
                return False
            
            # Perform the actual compliance check
            compliance_result = self._perform_compliance_check(framework, check_type, context)
            
            # Update status
            self.compliance_status[framework][check_type] = compliance_result
            
            # Log violations
            if not compliance_result:
                self._log_violation(framework, check_type, context)
            
            self.logger.debug("Compliance check performed",
                            framework=framework,
                            check_type=check_type,
                            compliant=compliance_result)
            
            return compliance_result
            
        except Exception as e:
            self.logger.error("Compliance check failed", error=str(e))
            return False
    
    def _perform_compliance_check(self, framework: str, check_type: str, context: Dict[str, Any] = None) -> bool:
        """Perform specific compliance check."""
        context = context or {}
        
        if framework == 'hipaa':
            return self._check_hipaa_compliance(check_type, context)
        elif framework == 'fda':
            return self._check_fda_compliance(check_type, context)
        elif framework == 'gdpr':
            return self._check_gdpr_compliance(check_type, context)
        elif framework == 'iso_27001':
            return self._check_iso27001_compliance(check_type, context)
        elif framework == 'iec_62304':
            return self._check_iec62304_compliance(check_type, context)
        
        return False
    
    def _check_hipaa_compliance(self, check_type: str, context: Dict[str, Any]) -> bool:
        """Check HIPAA compliance requirements."""
        # This is a simplified implementation - real implementation would be more comprehensive
        
        if check_type == 'phi_protection':
            # Check if PHI is properly protected
            phi_detected = context.get('phi_detected', False)
            phi_redacted = context.get('phi_redacted', False)
            return not phi_detected or phi_redacted
        
        elif check_type == 'data_encryption':
            # Check if data is encrypted
            encryption_enabled = context.get('encryption_enabled', True)
            return encryption_enabled
        
        elif check_type == 'audit_logging':
            # Check if audit logging is enabled
            audit_enabled = context.get('audit_logging_enabled', True)
            return audit_enabled
        
        elif check_type == 'access_control':
            # Check access control implementation
            access_control = context.get('access_control', 'rbac')
            return access_control in ['rbac', 'abac']
        
        elif check_type == 'data_retention':
            # Check data retention policies
            retention_days = context.get('data_retention_days', 30)
            return retention_days <= 2555  # 7 years max for medical records
        
        return True
    
    def _check_fda_compliance(self, check_type: str, context: Dict[str, Any]) -> bool:
        """Check FDA compliance for Software as Medical Device (SaMD)."""
        
        if check_type == 'clinical_validation':
            # Check if clinical validation is performed
            validation_performed = context.get('clinical_validation_performed', False)
            return validation_performed
        
        elif check_type == 'risk_assessment':
            # Check risk assessment documentation
            risk_assessment = context.get('risk_assessment_complete', True)
            return risk_assessment
        
        elif check_type == 'software_lifecycle':
            # Check software lifecycle documentation
            lifecycle_doc = context.get('software_lifecycle_documented', True)
            return lifecycle_doc
        
        elif check_type == 'cybersecurity':
            # Check cybersecurity measures
            cybersecurity_measures = context.get('cybersecurity_measures', ['encryption', 'access_control'])
            return len(cybersecurity_measures) >= 2
        
        elif check_type == 'post_market_surveillance':
            # Check post-market surveillance plan
            surveillance_plan = context.get('post_market_surveillance_plan', True)
            return surveillance_plan
        
        return True
    
    def _check_gdpr_compliance(self, check_type: str, context: Dict[str, Any]) -> bool:
        """Check GDPR compliance."""
        
        if check_type == 'consent_management':
            # Check consent management
            consent_obtained = context.get('consent_obtained', True)
            return consent_obtained
        
        elif check_type == 'data_minimization':
            # Check data minimization
            data_minimized = context.get('data_minimized', True)
            return data_minimized
        
        elif check_type == 'right_to_deletion':
            # Check right to deletion capability
            deletion_capability = context.get('deletion_capability', True)
            return deletion_capability
        
        elif check_type == 'data_portability':
            # Check data portability
            portability_enabled = context.get('data_portability', True)
            return portability_enabled
        
        elif check_type == 'privacy_by_design':
            # Check privacy by design implementation
            privacy_design = context.get('privacy_by_design', True)
            return privacy_design
        
        return True
    
    def _check_iso27001_compliance(self, check_type: str, context: Dict[str, Any]) -> bool:
        """Check ISO 27001 compliance."""
        
        if check_type == 'information_security_policy':
            # Check information security policy
            policy_exists = context.get('security_policy_exists', True)
            return policy_exists
        
        elif check_type == 'risk_management':
            # Check risk management process
            risk_management = context.get('risk_management_process', True)
            return risk_management
        
        elif check_type == 'access_control':
            # Check access control measures
            access_control = context.get('access_control_measures', ['authentication', 'authorization'])
            return len(access_control) >= 2
        
        elif check_type == 'cryptographic_controls':
            # Check cryptographic controls
            crypto_controls = context.get('cryptographic_controls', ['encryption', 'hashing'])
            return len(crypto_controls) >= 1
        
        elif check_type == 'incident_management':
            # Check incident management process
            incident_management = context.get('incident_management_process', True)
            return incident_management
        
        return True
    
    def _check_iec62304_compliance(self, check_type: str, context: Dict[str, Any]) -> bool:
        """Check IEC 62304 compliance for medical device software."""
        
        if check_type == 'software_classification':
            # Check software classification
            classification = context.get('software_class', 'class_a')  # A, B, or C
            return classification in ['class_a', 'class_b', 'class_c']
        
        elif check_type == 'development_process':
            # Check software development process
            dev_process = context.get('development_process_documented', True)
            return dev_process
        
        elif check_type == 'risk_management_file':
            # Check risk management file
            rmf_complete = context.get('risk_management_file_complete', True)
            return rmf_complete
        
        elif check_type == 'maintenance_process':
            # Check maintenance process
            maintenance_process = context.get('maintenance_process_documented', True)
            return maintenance_process
        
        elif check_type == 'problem_resolution':
            # Check problem resolution process
            problem_resolution = context.get('problem_resolution_process', True)
            return problem_resolution
        
        return True
    
    def _log_violation(self, framework: str, check_type: str, context: Dict[str, Any]):
        """Log compliance violation."""
        violation = {
            'framework': framework,
            'check_type': check_type,
            'timestamp': time.time(),
            'context': context,
            'severity': self._assess_violation_severity(framework, check_type)
        }
        
        self.violations_log.append(violation)
        
        self.logger.warning("Compliance violation detected",
                          framework=framework,
                          check_type=check_type,
                          severity=violation['severity'])
    
    def _assess_violation_severity(self, framework: str, check_type: str) -> str:
        """Assess the severity of a compliance violation."""
        # Define high-risk violations
        high_risk_violations = {
            'hipaa': ['phi_protection', 'data_encryption'],
            'fda': ['clinical_validation', 'risk_assessment'],
            'gdpr': ['consent_management', 'data_minimization'],
            'iso_27001': ['information_security_policy', 'access_control'],
            'iec_62304': ['software_classification', 'development_process']
        }
        
        if framework in high_risk_violations and check_type in high_risk_violations[framework]:
            return 'high'
        else:
            return 'medium'
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary for all frameworks."""
        summary = {
            'frameworks': {},
            'overall_compliance_score': 0.0,
            'total_violations': len(self.violations_log),
            'critical_violations': len([v for v in self.violations_log if v.get('severity') == 'high']),
            'audit_readiness': 0.0
        }
        
        total_checks = 0
        total_compliant = 0
        
        for framework, checks in self.compliance_status.items():
            framework_info = {
                'name': self.frameworks.get(framework, {}).get('name', framework),
                'total_checks': len(checks),
                'compliant_checks': sum(checks.values()),
                'compliance_rate': 0.0,
                'violations': []
            }
            
            total_checks += len(checks)
            total_compliant += sum(checks.values())
            
            # Calculate compliance rate
            if len(checks) > 0:
                framework_info['compliance_rate'] = sum(checks.values()) / len(checks)
            
            # Get violations for this framework
            framework_violations = [v for v in self.violations_log if v['framework'] == framework]
            framework_info['violations'] = len(framework_violations)
            
            summary['frameworks'][framework] = framework_info
        
        # Calculate overall compliance score
        if total_checks > 0:
            summary['overall_compliance_score'] = total_compliant / total_checks
        
        # Calculate audit readiness (higher score = more ready for audit)
        audit_factors = {
            'compliance_score': summary['overall_compliance_score'],
            'violation_penalty': max(0, 1 - (summary['critical_violations'] * 0.1)),
            'documentation_score': 0.9  # Assume good documentation
        }
        
        summary['audit_readiness'] = np.mean(list(audit_factors.values()))
        
        return summary


class ClinicalOutcomeTracker:
    """Comprehensive clinical outcome tracking and effectiveness measurement."""
    
    def __init__(self, 
                 tracking_window_days: int = 90,
                 validation_delay_days: int = 30,
                 min_outcomes_for_analysis: int = 50):
        
        self.tracking_window_days = tracking_window_days
        self.validation_delay_days = validation_delay_days
        self.min_outcomes_for_analysis = min_outcomes_for_analysis
        
        self.logger = structlog.get_logger("clinical_tracker")
        
        # Data storage
        self.outcomes: deque = deque(maxlen=10000)
        self.outcomes_by_model: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.outcomes_by_patient: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Effectiveness calculators
        self.regulatory_monitor = RegulatoryComplianceMonitor()
        
        # Statistical aggregators
        self.effectiveness_history: deque = deque(maxlen=1000)
        
        self.logger.info("ClinicalOutcomeTracker initialized")
    
    def track_prediction_outcome(self, 
                               prediction_id: str,
                               model_id: str,
                               patient_id: str,
                               outcome_type: OutcomeType,
                               predicted_outcome: Any,
                               confidence_score: float,
                               clinical_context: Dict[str, Any],
                               model_version: str) -> str:
        """Track a new prediction outcome."""
        
        # Create outcome ID
        outcome_id = f"{prediction_id}_{int(time.time())}"
        
        # Determine severity and specialty
        severity_level = self._determine_severity_level(clinical_context)
        medical_specialty = clinical_context.get('specialty', 'general')
        
        # Calculate follow-up duration
        follow_up_duration = self._calculate_follow_up_duration(outcome_type, severity_level)
        
        outcome = ClinicalOutcome(
            outcome_id=outcome_id,
            model_id=model_id,
            patient_id=patient_id,
            prediction_id=prediction_id,
            outcome_type=outcome_type,
            severity_level=severity_level,
            medical_specialty=medical_specialty,
            clinical_context=clinical_context,
            predicted_outcome=predicted_outcome,
            confidence_score=confidence_score,
            model_version=model_version,
            prediction_timestamp=time.time(),
            follow_up_duration_days=follow_up_duration
        )
        
        # Store outcome
        self.outcomes.append(outcome)
        self.outcomes_by_model[model_id].append(outcome)
        self.outcomes_by_patient[patient_id].append(outcome)
        
        # Check regulatory compliance
        self._check_prediction_compliance(outcome)
        
        self.logger.info("Prediction outcome tracked",
                       outcome_id=outcome_id,
                       model_id=model_id,
                       outcome_type=outcome_type.value,
                       severity_level=severity_level.value)
        
        return outcome_id
    
    def validate_outcome(self,
                        outcome_id: str,
                        outcome_result: OutcomeResult,
                        clinical_effectiveness_score: float,
                        safety_score: float,
                        validator_id: str,
                        feedback: Dict[str, Any] = None) -> bool:
        """Validate a tracked outcome."""
        
        try:
            # Find the outcome
            outcome = None
            for o in self.outcomes:
                if o.outcome_id == outcome_id:
                    outcome = o
                    break
            
            if not outcome:
                self.logger.error("Outcome not found for validation", outcome_id=outcome_id)
                return False
            
            # Update outcome with validation results
            outcome.outcome_result = outcome_result
            outcome.clinical_effectiveness_score = clinical_effectiveness_score
            outcome.safety_score = safety_score
            outcome.validator_id = validator_id
            outcome.validation_timestamp = time.time()
            
            if feedback:
                outcome.physician_feedback = feedback.get('physician_feedback')
                outcome.patient_feedback = feedback.get('patient_feedback')
                outcome.quality_score = feedback.get('quality_score')
            
            # Calculate additional metrics
            outcome.medical_relevance_score = self._calculate_medical_relevance(outcome)
            outcome.bias_score = self._calculate_bias_score(outcome)
            outcome.patient_impact = feedback.get('patient_impact') if feedback else None
            outcome.cost_impact = feedback.get('cost_impact') if feedback else None
            outcome.time_impact = feedback.get('time_impact') if feedback else None
            
            self.logger.info("Outcome validated",
                           outcome_id=outcome_id,
                           result=outcome_result.value,
                           effectiveness_score=clinical_effectiveness_score)
            
            return True
            
        except Exception as e:
            self.logger.error("Outcome validation failed", error=str(e))
            return False
    
    def calculate_effectiveness_metrics(self, model_id: str, time_window_days: int = None) -> MedicalEffectivenessMetrics:
        """Calculate medical effectiveness metrics for a model."""
        
        if time_window_days is None:
            time_window_days = self.tracking_window_days
        
        # Get outcomes for the model and time window
        cutoff_time = time.time() - (time_window_days * 24 * 3600)
        relevant_outcomes = [
            o for o in self.outcomes_by_model[model_id]
            if o.prediction_timestamp >= cutoff_time
        ]
        
        if len(relevant_outcomes) < self.min_outcomes_for_analysis:
            self.logger.warning("Insufficient outcomes for effectiveness calculation",
                              model_id=model_id,
                              outcomes_count=len(relevant_outcomes),
                              required=self.min_outcomes_for_analysis)
        
        # Calculate metrics
        metrics = MedicalEffectivenessMetrics(
            model_id=model_id,
            time_window_days=time_window_days,
            total_outcomes_tracked=len(relevant_outcomes)
        )
        
        if not relevant_outcomes:
            return metrics
        
        # Filter validated outcomes
        validated_outcomes = [o for o in relevant_outcomes if o.outcome_result is not None]
        metrics.validated_outcomes = len(validated_outcomes)
        metrics.pending_outcomes = len(relevant_outcomes) - len(validated_outcomes)
        
        if not validated_outcomes:
            return metrics
        
        try:
            # Overall effectiveness
            effectiveness_scores = [o.clinical_effectiveness_score for o in validated_outcomes 
                                  if o.clinical_effectiveness_score is not None]
            
            if effectiveness_scores:
                metrics.overall_effectiveness_score = np.mean(effectiveness_scores)
                metrics.effectiveness_ci_lower = np.percentile(effectiveness_scores, 5)
                metrics.effectiveness_ci_upper = np.percentile(effectiveness_scores, 95)
            
            # Safety scores
            safety_scores = [o.safety_score for o in validated_outcomes if o.safety_score is not None]
            if safety_scores:
                metrics.safety_score = np.mean(safety_scores)
                metrics.safety_ci_lower = np.percentile(safety_scores, 5)
                metrics.safety_ci_upper = np.percentile(safety_scores, 95)
            
            # Clinical accuracy (based on outcome results)
            true_positives = len([o for o in validated_outcomes if o.outcome_result == OutcomeResult.TRUE_POSITIVE])
            true_negatives = len([o for o in validated_outcomes if o.outcome_result == OutcomeResult.TRUE_NEGATIVE])
            total_validated = len(validated_outcomes)
            
            if total_validated > 0:
                metrics.clinical_accuracy = (true_positives + true_negatives) / total_validated
            
            # Medical relevance
            relevance_scores = [o.medical_relevance_score for o in validated_outcomes 
                              if o.medical_relevance_score is not None]
            if relevance_scores:
                metrics.medical_relevance = np.mean(relevance_scores)
            
            # Bias scores
            bias_scores = [o.bias_score for o in validated_outcomes if o.bias_score is not None]
            if bias_scores:
                metrics.bias_score = np.mean(bias_scores)
            
            # Outcome-type specific accuracy
            metrics.diagnosis_accuracy = self._calculate_outcome_type_accuracy(
                validated_outcomes, OutcomeType.DIAGNOSIS
            )
            metrics.treatment_recommendation_accuracy = self._calculate_outcome_type_accuracy(
                validated_outcomes, OutcomeType.TREATMENT_RECOMMENDATION
            )
            metrics.prognosis_accuracy = self._calculate_outcome_type_accuracy(
                validated_outcomes, OutcomeType.PROGNOSIS
            )
            metrics.triage_accuracy = self._calculate_outcome_type_accuracy(
                validated_outcomes, OutcomeType.TRIAGE
            )
            
            # Specialty performance
            metrics.specialty_performance = self._calculate_specialty_performance(validated_outcomes)
            
            # Patient outcome improvements
            metrics.patient_benefit_rate = self._calculate_patient_benefit_rate(validated_outcomes)
            
            # Adverse event rate
            adverse_events = len([o for o in validated_outcomes 
                                if o.outcome_result in [OutcomeResult.FALSE_POSITIVE, OutcomeResult.FALSE_NEGATIVE]
                                and o.severity_level in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]])
            
            if total_validated > 0:
                metrics.adverse_event_rate = adverse_events / total_validated
            
            # Regulatory compliance
            compliance_summary = self.regulatory_monitor.get_compliance_summary()
            metrics.regulatory_compliance_score = compliance_summary['overall_compliance_score']
            metrics.audit_readiness = compliance_summary['audit_readiness']
            metrics.documentation_completeness = 0.9  # Placeholder
            
            # Store in history
            self.effectiveness_history.append(metrics)
            
            self.logger.info("Effectiveness metrics calculated",
                           model_id=model_id,
                           overall_effectiveness=metrics.overall_effectiveness_score,
                           clinical_accuracy=metrics.clinical_accuracy,
                           validated_outcomes=metrics.validated_outcomes)
            
            return metrics
            
        except Exception as e:
            self.logger.error("Effectiveness metrics calculation failed", error=str(e))
            return metrics
    
    def _determine_severity_level(self, clinical_context: Dict[str, Any]) -> SeverityLevel:
        """Determine severity level from clinical context."""
        specialty = clinical_context.get('specialty', 'general')
        condition_type = clinical_context.get('condition_type', 'routine')
        urgency = clinical_context.get('urgency', 'routine')
        
        # Critical conditions
        if urgency in ['emergency', 'critical'] or specialty in ['emergency', 'cardiology', 'oncology']:
            return SeverityLevel.CRITICAL
        
        # High severity conditions
        if condition_type in ['serious', 'chronic'] or specialty in ['neurology', 'surgery']:
            return SeverityLevel.HIGH
        
        # Medium severity
        if condition_type in ['moderate', 'acute']:
            return SeverityLevel.MEDIUM
        
        # Low severity / preventive
        if condition_type in ['routine', 'preventive', 'screening']:
            return SeverityLevel.PREVENTIVE
        
        # Default
        return SeverityLevel.MEDIUM
    
    def _calculate_follow_up_duration(self, outcome_type: OutcomeType, severity_level: SeverityLevel) -> int:
        """Calculate appropriate follow-up duration in days."""
        base_durations = {
            OutcomeType.DIAGNOSIS: 30,
            OutcomeType.TREATMENT_RECOMMENDATION: 60,
            OutcomeType.PROGNOSIS: 90,
            OutcomeType.TRIAGE: 7,
            OutcomeType.MEDICATION_ADJUSTMENT: 14,
            OutcomeType.SURGICAL_DECISION: 30,
            OutcomeType.MONITORING_ALERT: 3,
            OutcomeType.PREVENTIVE_CARE: 365
        }
        
        base_duration = base_durations.get(outcome_type, 30)
        
        # Adjust for severity
        severity_multipliers = {
            SeverityLevel.CRITICAL: 0.5,  # Shorter follow-up for critical cases
            SeverityLevel.HIGH: 0.75,
            SeverityLevel.MEDIUM: 1.0,
            SeverityLevel.LOW: 1.5,
            SeverityLevel.PREVENTIVE: 2.0
        }
        
        multiplier = severity_multipliers.get(severity_level, 1.0)
        return max(1, int(base_duration * multiplier))
    
    def _check_prediction_compliance(self, outcome: ClinicalOutcome):
        """Check regulatory compliance for a prediction."""
        context = {
            'clinical_context': outcome.clinical_context,
            'medical_specialty': outcome.medical_specialty,
            'severity_level': outcome.severity_level.value,
            'timestamp': outcome.prediction_timestamp
        }
        
        # Check HIPAA compliance
        self.regulatory_monitor.check_compliance('hipaa', 'phi_protection', context)
        self.regulatory_monitor.check_compliance('hipaa', 'audit_logging', context)
        
        # Check FDA compliance (if applicable)
        if outcome.severity_level in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
            self.regulatory_monitor.check_compliance('fda', 'clinical_validation', context)
            self.regulatory_monitor.check_compliance('fda', 'post_market_surveillance', context)
    
    def _calculate_medical_relevance(self, outcome: ClinicalOutcome) -> float:
        """Calculate medical relevance score for an outcome."""
        # Base relevance (medical AI should always be relevant)
        base_relevance = 0.8
        
        # Adjust for specialty
        specialty_relevance = {
            'cardiology': 0.95,
            'oncology': 0.95,
            'emergency': 0.95,
            'radiology': 0.9,
            'neurology': 0.9,
            'surgery': 0.9,
            'general': 0.8,
            'preventive': 0.7
        }.get(outcome.medical_specialty, 0.8)
        
        # Adjust for severity
        severity_relevance = {
            SeverityLevel.CRITICAL: 1.0,
            SeverityLevel.HIGH: 0.95,
            SeverityLevel.MEDIUM: 0.9,
            SeverityLevel.LOW: 0.8,
            SeverityLevel.PREVENTIVE: 0.7
        }.get(outcome.severity_level, 0.8)
        
        # Adjust for confidence
        confidence_factor = outcome.confidence_score
        
        relevance_score = base_relevance * specialty_relevance * severity_relevance * confidence_factor
        return min(1.0, relevance_score)
    
    def _calculate_bias_score(self, outcome: ClinicalOutcome) -> float:
        """Calculate bias score for an outcome (lower is better)."""
        # This is a simplified bias calculation
        # In practice, this would be much more sophisticated
        
        context = outcome.clinical_context
        demographic_info = context.get('demographics', {})
        
        # Check for demographic representation
        age_group = demographic_info.get('age_group')
        gender = demographic_info.get('gender')
        ethnicity = demographic_info.get('ethnicity')
        
        bias_indicators = []
        
        # Age bias check
        if age_group and age_group in ['elderly', 'pediatric']:
            bias_indicators.append(0.1)  # Potential age bias
        
        # Gender bias check
        if gender and outcome.severity_level == SeverityLevel.CRITICAL:
            bias_indicators.append(0.05)  # Potential gender bias in critical care
        
        # Ethnicity bias check (simplified)
        if ethnicity and ethnicity not in ['caucasian', 'mixed']:
            bias_indicators.append(0.05)  # Potential ethnic bias
        
        # Base bias score
        base_bias = 0.02
        bias_penalty = sum(bias_indicators)
        
        bias_score = base_bias + bias_penalty
        return min(0.5, bias_score)  # Cap at 0.5 (50% bias)
    
    def _calculate_outcome_type_accuracy(self, outcomes: List[ClinicalOutcome], outcome_type: OutcomeType) -> float:
        """Calculate accuracy for specific outcome type."""
        type_outcomes = [o for o in outcomes if o.outcome_type == outcome_type]
        
        if not type_outcomes:
            return 0.0
        
        correct_outcomes = [
            o for o in type_outcomes
            if o.outcome_result in [OutcomeResult.TRUE_POSITIVE, OutcomeResult.TRUE_NEGATIVE]
        ]
        
        return len(correct_outcomes) / len(type_outcomes)
    
    def _calculate_specialty_performance(self, outcomes: List[ClinicalOutcome]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by medical specialty."""
        specialty_performance = defaultdict(lambda: {'accuracy': 0.0, 'count': 0, 'effectiveness': 0.0})
        
        for outcome in outcomes:
            specialty = outcome.medical_specialty
            
            specialty_performance[specialty]['count'] += 1
            
            # Accuracy
            if outcome.outcome_result in [OutcomeResult.TRUE_POSITIVE, OutcomeResult.TRUE_NEGATIVE]:
                specialty_performance[specialty]['accuracy'] += 1
            
            # Effectiveness
            if outcome.clinical_effectiveness_score is not None:
                specialty_performance[specialty]['effectiveness'] += outcome.clinical_effectiveness_score
        
        # Calculate averages
        for specialty, metrics in specialty_performance.items():
            count = metrics['count']
            if count > 0:
                metrics['accuracy'] = metrics['accuracy'] / count
                if metrics['effectiveness'] > 0:
                    metrics['effectiveness'] = metrics['effectiveness'] / count
        
        return dict(specialty_performance)
    
    def _calculate_patient_benefit_rate(self, outcomes: List[ClinicalOutcome]) -> float:
        """Calculate rate of positive patient outcomes."""
        positive_outcomes = [
            o for o in outcomes
            if o.outcome_result in [OutcomeResult.TRUE_POSITIVE, OutcomeResult.TRUE_NEGATIVE]
            and o.patient_impact in ['positive', 'improved']
        ]
        
        return len(positive_outcomes) / len(outcomes) if outcomes else 0.0
    
    def get_outcome_summary(self, model_id: str = None, time_window_days: int = None) -> Dict[str, Any]:
        """Get summary of tracked outcomes."""
        
        if time_window_days is None:
            time_window_days = self.tracking_window_days
        
        cutoff_time = time.time() - (time_window_days * 24 * 3600)
        
        if model_id:
            relevant_outcomes = [o for o in self.outcomes_by_model[model_id] 
                               if o.prediction_timestamp >= cutoff_time]
        else:
            relevant_outcomes = [o for o in self.outcomes 
                               if o.prediction_timestamp >= cutoff_time]
        
        # Calculate summary statistics
        summary = {
            'time_window_days': time_window_days,
            'total_outcomes': len(relevant_outcomes),
            'outcomes_by_type': defaultdict(int),
            'outcomes_by_severity': defaultdict(int),
            'outcomes_by_specialty': defaultdict(int),
            'validation_rate': 0.0,
            'average_effectiveness': 0.0,
            'average_safety_score': 0.0
        }
        
        if not relevant_outcomes:
            return dict(summary)
        
        # Count by categories
        for outcome in relevant_outcomes:
            summary['outcomes_by_type'][outcome.outcome_type.value] += 1
            summary['outcomes_by_severity'][outcome.severity_level.value] += 1
            summary['outcomes_by_specialty'][outcome.medical_specialty] += 1
        
        # Validation statistics
        validated_outcomes = [o for o in relevant_outcomes if o.outcome_result is not None]
        summary['validation_rate'] = len(validated_outcomes) / len(relevant_outcomes)
        
        # Effectiveness statistics
        effectiveness_scores = [o.clinical_effectiveness_score for o in validated_outcomes 
                              if o.clinical_effectiveness_score is not None]
        if effectiveness_scores:
            summary['average_effectiveness'] = np.mean(effectiveness_scores)
        
        # Safety statistics
        safety_scores = [o.safety_score for o in validated_outcomes if o.safety_score is not None]
        if safety_scores:
            summary['average_safety_score'] = np.mean(safety_scores)
        
        return dict(summary)
    
    def export_outcomes_data(self, model_id: str = None, format: str = 'json') -> str:
        """Export outcomes data for analysis."""
        
        if model_id:
            relevant_outcomes = list(self.outcomes_by_model[model_id])
        else:
            relevant_outcomes = list(self.outcomes)
        
        if format.lower() == 'json':
            # Export as JSON
            outcomes_data = [asdict(outcome) for outcome in relevant_outcomes]
            return json.dumps(outcomes_data, indent=2, default=str)
        
        elif format.lower() == 'csv':
            # Export as CSV (simplified)
            import csv
            import io
            
            output = io.StringIO()
            if relevant_outcomes:
                fieldnames = asdict(relevant_outcomes[0]).keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                
                for outcome in relevant_outcomes:
                    writer.writerow(asdict(outcome))
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")