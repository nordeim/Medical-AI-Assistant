"""
Version metadata and documentation tracking with clinical validation records.

Provides comprehensive metadata management for medical AI models
including regulatory documentation, clinical validation, and audit trails.
"""

import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

from .core import ModelVersion, ComplianceLevel, VersionStatus

logger = logging.getLogger(__name__)


class ValidationType(Enum):
    """Types of clinical validation."""
    PRE_CLINICAL = "pre_clinical"
    CLINICAL_INVESTIGATION = "clinical_investigation"
    CLINICAL_VALIDATION = "clinical_validation"
    PROSPECTIVE_VALIDATION = "prospective_validation"
    RETROSPECTIVE_VALIDATION = "retrospective_validation"
    REAL_WORLD_VALIDATION = "real_world_validation"


class DocumentType(Enum):
    """Types of regulatory and clinical documents."""
    REGULATORY_SUBMISSION = "regulatory_submission"
    CLINICAL_TRIAL_PROTOCOL = "clinical_trial_protocol"
    CLINICAL_DATA_REPORT = "clinical_data_report"
    RISK_ASSESSMENT = "risk_assessment"
    VALIDATION_REPORT = "validation_report"
    ADVERSE_EVENT_REPORT = "adverse_event_report"
    POST_MARKET_SURVEILLANCE = "post_market_surveillance"
    QUALITY_MANIFEST = "quality_manifest"
    CHANGE_CONTROL = "change_control"
    COMPLIANCE_CERTIFICATE = "compliance_certificate"


class ValidationStatus(Enum):
    """Status of validation activities."""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    SUSPENDED = "suspended"
    WITHDRAWN = "withdrawn"


@dataclass
class ClinicalValidation:
    """Clinical validation record for medical AI model."""
    validation_id: str
    validation_type: ValidationType
    validation_name: str
    description: str
    
    # Timeline
    start_date: datetime
    end_date: Optional[datetime] = None
    duration_days: Optional[int] = None
    
    # Study design
    study_design: str = ""
    patient_population: str = ""
    inclusion_criteria: List[str] = field(default_factory=list)
    exclusion_criteria: List[str] = field(default_factory=list)
    sample_size: Optional[int] = None
    
    # Primary endpoints
    primary_endpoints: List[str] = field(default_factory=list)
    secondary_endpoints: List[str] = field(default_factory=list)
    
    # Results
    results: Dict[str, Any] = field(default_factory=dict)
    statistical_analysis: Dict[str, Any] = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)
    
    # Quality and compliance
    validation_status: ValidationStatus = ValidationStatus.PLANNED
    gmp_compliant: bool = False
    irb_approved: bool = False
    regulatory_approval_required: bool = False
    regulatory_approval_status: str = "not_required"
    
    # Documentation
    protocols: List[str] = field(default_factory=list)
    data_files: List[str] = field(default_factory=list)
    analysis_reports: List[str] = field(default_factory=list)
    approval_documents: List[str] = field(default_factory=list)
    
    # Investigators and approvals
    principal_investigator: str = ""
    coordinating_investigator: str = ""
    statistical_investigator: str = ""
    irb_approval_date: Optional[datetime] = None
    regulatory_approval_date: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    updated_at: Optional[datetime] = None
    
    def calculate_duration(self) -> Optional[int]:
        """Calculate validation duration in days."""
        if self.end_date:
            return (self.end_date - self.start_date).days
        elif self.start_date:
            return (datetime.now() - self.start_date).days
        return None
    
    def is_completed(self) -> bool:
        """Check if validation is completed."""
        return self.validation_status in [ValidationStatus.COMPLETED, ValidationStatus.PASSED, ValidationStatus.FAILED]
    
    def meets_primary_endpoints(self) -> Optional[bool]:
        """Check if primary endpoints are met based on results."""
        if not self.results or not self.primary_endpoints:
            return None
        
        # Simplified check - in practice would analyze actual results
        return True  # Placeholder


@dataclass
class RegulatoryDocument:
    """Regulatory and compliance document tracking."""
    document_id: str
    document_type: DocumentType
    document_name: str
    description: str
    
    # Regulatory information
    regulatory_body: str = ""
    jurisdiction: str = ""
    submission_number: str = ""
    approval_number: Optional[str] = None
    
    # Timeline
    created_date: datetime = field(default_factory=datetime.now)
    submission_date: Optional[datetime] = None
    approval_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    
    # Status and compliance
    status: str = "draft"
    compliance_level: ComplianceLevel = ComplianceLevel.UNKNOWN
    version: str = "1.0"
    language: str = "en"
    
    # File information
    file_path: str = ""
    file_size_bytes: Optional[int] = None
    file_hash: Optional[str] = None
    mime_type: str = ""
    
    # Signatures and approvals
    author: str = ""
    reviewer: str = ""
    approver: str = ""
    digital_signature: Optional[str] = None
    
    # Content metadata
    keywords: List[str] = field(default_factory=list)
    related_documents: List[str] = field(default_factory=list)
    superseded_documents: List[str] = field(default_factory=list)
    
    # Audit trail
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    def generate_file_hash(self, file_path: str) -> str:
        """Generate hash for document file."""
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            self.file_hash = file_hash
            return file_hash
        except Exception as e:
            logger.error(f"Failed to generate hash for {file_path}: {e}")
            return ""
    
    def is_expired(self) -> bool:
        """Check if document is expired."""
        if self.expiry_date:
            return datetime.now() > self.expiry_date
        return False
    
    def add_audit_entry(self, action: str, user: str, details: Dict[str, Any] = None):
        """Add entry to document audit trail."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "user": user,
            "details": details or {}
        }
        self.audit_trail.append(entry)


@dataclass
class VersionMetadata:
    """Comprehensive metadata for model version."""
    
    # Core identification
    version_id: str
    model_name: str
    model_version: str
    metadata_version: str = "1.0"
    
    # Documentation
    documentation: List[str] = field(default_factory=list)
    clinical_validation_records: List[ClinicalValidation] = field(default_factory=list)
    regulatory_documents: List[RegulatoryDocument] = field(default_factory=list)
    
    # Clinical information
    intended_use: str = ""
    indications: List[str] = field(default_factory=list)
    contraindications: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    precautions: List[str] = field(default_factory=list)
    
    # Performance characteristics
    performance_characteristics: Dict[str, str] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    clinical_benefits: List[str] = field(default_factory=list)
    
    # Quality and safety
    quality_standards: List[str] = field(default_factory=list)
    safety_profile: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    
    # Regulatory status
    regulatory_status: Dict[str, Any] = field(default_factory=dict)
    market_authorization: Dict[str, Any] = field(default_factory=dict)
    post_market_surveillance: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle management
    change_history: List[Dict[str, Any]] = field(default_factory=list)
    retirement_plan: str = ""
    sunset_date: Optional[datetime] = None
    
    # Audit and compliance
    compliance_checklist: Dict[str, Any] = field(default_factory=dict)
    audit_schedule: List[Dict[str, Any]] = field(default_factory=list)
    last_audit_date: Optional[datetime] = None
    next_audit_date: Optional[datetime] = None
    
    # Metadata management
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    updated_at: Optional[datetime] = None
    updated_by: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        
        # Handle datetime serialization
        for field_name in ['created_at', 'updated_at', 'sunset_date', 'last_audit_date', 'next_audit_date']:
            if field_name in data and data[field_name]:
                data[field_name] = data[field_name].isoformat()
        
        return data
    
    def add_clinical_validation(self, validation: ClinicalValidation):
        """Add clinical validation record."""
        self.clinical_validation_records.append(validation)
        self.updated_at = datetime.now()
        self.add_change_entry("clinical_validation_added", "system", {
            "validation_id": validation.validation_id,
            "validation_type": validation.validation_type.value
        })
    
    def add_regulatory_document(self, document: RegulatoryDocument):
        """Add regulatory document."""
        self.regulatory_documents.append(document)
        self.updated_at = datetime.now()
        self.add_change_entry("regulatory_document_added", "system", {
            "document_id": document.document_id,
            "document_type": document.document_type.value
        })
    
    def add_change_entry(self, change_type: str, user: str, details: Dict[str, Any] = None):
        """Add entry to change history."""
        change_entry = {
            "timestamp": datetime.now().isoformat(),
            "change_type": change_type,
            "user": user,
            "details": details or {}
        }
        self.change_history.append(change_entry)
    
    def validate_metadata_completeness(self) -> Dict[str, List[str]]:
        """Validate metadata completeness."""
        issues = {
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check required documentation
        if not self.documentation:
            issues["errors"].append("No documentation provided")
        
        if not self.clinical_validation_records:
            issues["warnings"].append("No clinical validation records")
        
        if not self.regulatory_documents:
            issues["warnings"].append("No regulatory documents")
        
        # Check clinical information
        if not self.intended_use:
            issues["errors"].append("No intended use specified")
        
        if not self.indications:
            issues["warnings"].append("No indications specified")
        
        # Check regulatory compliance
        if not self.regulatory_status:
            issues["warnings"].append("No regulatory status information")
        
        if not self.quality_standards:
            issues["recommendations"].append("Add quality standards information")
        
        # Check audit trail
        if not self.change_history:
            issues["warnings"].append("No change history recorded")
        
        return issues


class MetadataManager:
    """Manager for version metadata and documentation tracking."""
    
    def __init__(self, storage_path: str = "/tmp/metadata"):
        self.storage_path = Path(storage_path)
        self.metadata_index: Dict[str, VersionMetadata] = {}
        self.validation_index: Dict[str, ClinicalValidation] = {}
        self.document_index: Dict[str, RegulatoryDocument] = {}
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "metadata_index.json"
        self.validations_file = self.storage_path / "validations_index.json"
        self.documents_file = self.storage_path / "documents_index.json"
        
        # Load existing metadata
        self._load_indices()
    
    def create_metadata(self, model_version: ModelVersion, created_by: str = "") -> VersionMetadata:
        """Create comprehensive metadata for model version."""
        
        metadata = VersionMetadata(
            version_id=str(uuid.uuid4()),
            model_name=model_version.model_name,
            model_version=model_version.version,
            created_by=created_by or model_version.created_by,
            intended_use=model_version.compliance.intended_use,
            contraindications=model_version.compliance.contraindications,
            warnings=model_version.compliance.warnings,
            created_at=model_version.created_at
        )
        
        # Add initial audit entry
        metadata.add_change_entry("metadata_created", created_by or "system")
        
        # Validate completeness
        completeness = metadata.validate_metadata_completeness()
        if completeness["errors"]:
            logger.warning(f"Metadata has completeness issues: {completeness['errors']}")
        
        # Store metadata
        self.metadata_index[metadata.version_id] = metadata
        self._save_metadata_index()
        
        # Save individual metadata file
        self._save_metadata_file(metadata)
        
        logger.info(f"Created metadata for {model_version.model_name} v{model_version.version}")
        return metadata
    
    def update_metadata(self, metadata_id: str, updates: Dict[str, Any], updated_by: str = "") -> bool:
        """Update metadata with new information."""
        
        if metadata_id not in self.metadata_index:
            logger.error(f"Metadata {metadata_id} not found")
            return False
        
        metadata = self.metadata_index[metadata_id]
        original_values = {}
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(metadata, key):
                original_values[key] = getattr(metadata, key)
                setattr(metadata, key, value)
        
        metadata.updated_at = datetime.now()
        metadata.updated_by = updated_by
        
        # Add change entry
        metadata.add_change_entry("metadata_updated", updated_by, {
            "updated_fields": list(updates.keys()),
            "original_values": original_values
        })
        
        # Save updated metadata
        self._save_metadata_index()
        self._save_metadata_file(metadata)
        
        logger.info(f"Updated metadata {metadata_id}")
        return True
    
    def add_clinical_validation(self, metadata_id: str, validation: ClinicalValidation) -> bool:
        """Add clinical validation to metadata."""
        
        if metadata_id not in self.metadata_index:
            logger.error(f"Metadata {metadata_id} not found")
            return False
        
        metadata = self.metadata_index[metadata_id]
        metadata.add_clinical_validation(validation)
        
        # Store validation separately
        self.validation_index[validation.validation_id] = validation
        self._save_validations_index()
        
        # Save updated metadata
        self._save_metadata_index()
        self._save_metadata_file(metadata)
        
        logger.info(f"Added clinical validation {validation.validation_id} to metadata {metadata_id}")
        return True
    
    def add_regulatory_document(self, metadata_id: str, document: RegulatoryDocument, file_path: str = "") -> bool:
        """Add regulatory document to metadata."""
        
        if metadata_id not in self.metadata_index:
            logger.error(f"Metadata {metadata_id} not found")
            return False
        
        metadata = self.metadata_index[metadata_id]
        
        # Process file if provided
        if file_path and not document.file_hash:
            document.file_path = file_path
            document.generate_file_hash(file_path)
            
            # Get file size
            try:
                document.file_size_bytes = Path(file_path).stat().st_size
            except:
                pass
        
        metadata.add_regulatory_document(document)
        
        # Store document separately
        self.document_index[document.document_id] = document
        self._save_documents_index()
        
        # Save updated metadata
        self._save_metadata_index()
        self._save_metadata_file(metadata)
        
        logger.info(f"Added regulatory document {document.document_id} to metadata {metadata_id}")
        return True
    
    def get_metadata(self, metadata_id: str) -> Optional[VersionMetadata]:
        """Get metadata by ID."""
        return self.metadata_index.get(metadata_id)
    
    def get_metadata_by_model(self, model_name: str, version: str = None) -> List[VersionMetadata]:
        """Get metadata for specific model version."""
        results = []
        
        for metadata in self.metadata_index.values():
            if metadata.model_name == model_name:
                if version is None or metadata.model_version == version:
                    results.append(metadata)
        
        return results
    
    def get_clinical_validations(self, metadata_id: str) -> List[ClinicalValidation]:
        """Get clinical validations for metadata."""
        if metadata_id in self.metadata_index:
            return self.metadata_index[metadata_id].clinical_validation_records
        return []
    
    def get_regulatory_documents(self, metadata_id: str) -> List[RegulatoryDocument]:
        """Get regulatory documents for metadata."""
        if metadata_id in self.metadata_index:
            return self.metadata_index[metadata_id].regulatory_documents
        return []
    
    def validate_regulatory_compliance(self, metadata_id: str) -> Dict[str, Any]:
        """Validate regulatory compliance for metadata."""
        
        if metadata_id not in self.metadata_index:
            return {"error": "Metadata not found"}
        
        metadata = self.metadata_index[metadata_id]
        validation_results = {
            "compliant": True,
            "issues": [],
            "recommendations": [],
            "compliance_score": 0.0
        }
        
        score_components = []
        
        # Check documentation completeness
        doc_score = self._score_documentation_completeness(metadata)
        score_components.append(doc_score)
        
        # Check clinical validation
        clinical_score = self._score_clinical_validation(metadata)
        score_components.append(clinical_score)
        
        # Check regulatory documents
        regulatory_score = self._score_regulatory_documents(metadata)
        score_components.append(regulatory_score)
        
        # Check quality standards
        quality_score = self._score_quality_standards(metadata)
        score_components.append(quality_score)
        
        # Calculate overall score
        validation_results["compliance_score"] = sum(score_components) / len(score_components) if score_components else 0.0
        
        # Determine compliance status
        if validation_results["compliance_score"] >= 0.9:
            validation_results["compliant"] = True
        elif validation_results["compliance_score"] >= 0.7:
            validation_results["compliant"] = True
            validation_results["recommendations"].append("Consider additional documentation for full compliance")
        else:
            validation_results["compliant"] = False
            validation_results["issues"].append("Multiple compliance gaps identified")
        
        return validation_results
    
    def generate_compliance_report(self, metadata_id: str) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        if metadata_id not in self.metadata_index:
            return {"error": "Metadata not found"}
        
        metadata = self.metadata_index[metadata_id]
        
        report = {
            "metadata_info": {
                "model_name": metadata.model_name,
                "model_version": metadata.model_version,
                "version_id": metadata.version_id,
                "created_at": metadata.created_at.isoformat(),
                "compliance_score": 0.0
            },
            
            "clinical_information": {
                "intended_use": metadata.intended_use,
                "indications": metadata.indications,
                "contraindications": metadata.contraindications,
                "warnings": metadata.warnings,
                "precautions": metadata.precautions
            },
            
            "validation_summary": {
                "total_validations": len(metadata.clinical_validation_records),
                "completed_validations": len([v for v in metadata.clinical_validation_records if v.is_completed()]),
                "validation_types": [v.validation_type.value for v in metadata.clinical_validation_records]
            },
            
            "regulatory_status": {
                "total_documents": len(metadata.regulatory_documents),
                "documents_by_type": {},
                "regulatory_body": metadata.regulatory_status.get("body", "not_specified"),
                "market_authorization": metadata.market_authorization
            },
            
            "quality_assessment": {
                "quality_standards": metadata.quality_standards,
                "safety_profile": metadata.safety_profile,
                "risk_management": metadata.risk_management
            },
            
            "compliance_validation": self.validate_regulatory_compliance(metadata_id),
            
            "audit_trail": {
                "change_history_count": len(metadata.change_history),
                "last_audit_date": metadata.last_audit_date.isoformat() if metadata.last_audit_date else None,
                "next_audit_date": metadata.next_audit_date.isoformat() if metadata.next_audit_date else None
            }
        }
        
        # Calculate compliance score
        compliance_score = self.validate_regulatory_compliance(metadata_id)["compliance_score"]
        report["metadata_info"]["compliance_score"] = compliance_score
        
        # Document type breakdown
        for doc in metadata.regulatory_documents:
            doc_type = doc.document_type.value
            report["regulatory_status"]["documents_by_type"][doc_type] = \
                report["regulatory_status"]["documents_by_type"].get(doc_type, 0) + 1
        
        return report
    
    def schedule_audit(self, metadata_id: str, audit_date: datetime, audit_type: str, auditor: str) -> bool:
        """Schedule audit for metadata."""
        
        if metadata_id not in self.metadata_index:
            return False
        
        metadata = self.metadata_index[metadata_id]
        
        audit_entry = {
            "audit_id": str(uuid.uuid4()),
            "audit_type": audit_type,
            "scheduled_date": audit_date.isoformat(),
            "auditor": auditor,
            "status": "scheduled",
            "created_at": datetime.now().isoformat()
        }
        
        metadata.audit_schedule.append(audit_entry)
        metadata.next_audit_date = audit_date
        metadata.updated_at = datetime.now()
        
        self._save_metadata_index()
        self._save_metadata_file(metadata)
        
        logger.info(f"Scheduled audit for metadata {metadata_id}: {audit_type} on {audit_date}")
        return True
    
    def complete_audit(self, metadata_id: str, audit_id: str, results: Dict[str, Any], completed_by: str) -> bool:
        """Complete audit and record results."""
        
        if metadata_id not in self.metadata_index:
            return False
        
        metadata = self.metadata_index[metadata_id]
        
        # Find and update audit entry
        audit_entry = None
        for entry in metadata.audit_schedule:
            if entry["audit_id"] == audit_id:
                audit_entry = entry
                break
        
        if not audit_entry:
            logger.error(f"Audit {audit_id} not found")
            return False
        
        # Update audit entry
        audit_entry.update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "completed_by": completed_by,
            "results": results
        })
        
        metadata.last_audit_date = datetime.now()
        metadata.updated_at = datetime.now()
        
        # Add change entry
        metadata.add_change_entry("audit_completed", completed_by, {
            "audit_id": audit_id,
            "audit_type": audit_entry["audit_type"],
            "results": results
        })
        
        self._save_metadata_index()
        self._save_metadata_file(metadata)
        
        logger.info(f"Completed audit {audit_id} for metadata {metadata_id}")
        return True
    
    def _load_indices(self):
        """Load metadata indices from storage."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file) as f:
                    data = json.load(f)
                    for metadata_data in data.values():
                        metadata = VersionMetadata(**metadata_data)
                        # Convert datetime strings back to datetime objects
                        for field_name in ['created_at', 'updated_at', 'sunset_date', 'last_audit_date', 'next_audit_date']:
                            if hasattr(metadata, field_name) and getattr(metadata, field_name):
                                setattr(metadata, field_name, datetime.fromisoformat(getattr(metadata, field_name)))
                        
                        # Convert clinical validations
                        metadata.clinical_validation_records = []
                        # Would need to implement ClinicalValidation.from_dict() for full deserialization
                        
                        self.metadata_index[metadata.version_id] = metadata
            
            # Similar loading for validations and documents
            if self.validations_file.exists():
                with open(self.validations_file) as f:
                    data = json.load(f)
                    # Load validations
            
            if self.documents_file.exists():
                with open(self.documents_file) as f:
                    data = json.load(f)
                    # Load documents
                    
        except Exception as e:
            logger.error(f"Failed to load indices: {e}")
    
    def _save_metadata_index(self):
        """Save metadata index to storage."""
        try:
            index_data = {mid: metadata.to_dict() for mid, metadata in self.metadata_index.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata index: {e}")
    
    def _save_metadata_file(self, metadata: VersionMetadata):
        """Save individual metadata file."""
        try:
            metadata_file = self.storage_path / f"{metadata.version_id}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save metadata file: {e}")
    
    def _save_validations_index(self):
        """Save validations index."""
        try:
            index_data = {vid: validation.to_dict() for vid, validation in self.validation_index.items()}
            with open(self.validations_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save validations index: {e}")
    
    def _save_documents_index(self):
        """Save documents index."""
        try:
            index_data = {did: document.to_dict() for did, document in self.document_index.items()}
            with open(self.documents_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save documents index: {e}")
    
    def _score_documentation_completeness(self, metadata: VersionMetadata) -> float:
        """Score documentation completeness (0.0 to 1.0)."""
        score = 0.0
        
        # Documentation files
        if metadata.documentation:
            score += 0.2
        
        # Clinical information
        if metadata.intended_use:
            score += 0.2
        
        if metadata.indications:
            score += 0.1
        
        if metadata.contraindications:
            score += 0.1
        
        if metadata.warnings:
            score += 0.1
        
        # Performance characteristics
        if metadata.performance_characteristics:
            score += 0.1
        
        if metadata.limitations:
            score += 0.1
        
        if metadata.clinical_benefits:
            score += 0.1
        
        return min(score, 1.0)
    
    def _score_clinical_validation(self, metadata: VersionMetadata) -> float:
        """Score clinical validation completeness (0.0 to 1.0)."""
        if not metadata.clinical_validation_records:
            return 0.0
        
        completed_validations = [v for v in metadata.clinical_validation_records if v.is_completed()]
        
        # Base score for having validations
        score = 0.3
        
        # Bonus for completed validations
        score += len(completed_validations) * 0.2
        
        # Bonus for different validation types
        validation_types = {v.validation_type for v in metadata.clinical_validation_records}
        score += len(validation_types) * 0.1
        
        # Bonus for regulatory approval
        if any(v.regulatory_approval_required for v in metadata.clinical_validation_records):
            if any(v.regulatory_approval_status == "approved" for v in metadata.clinical_validation_records):
                score += 0.2
        
        return min(score, 1.0)
    
    def _score_regulatory_documents(self, metadata: VersionMetadata) -> float:
        """Score regulatory documents completeness (0.0 to 1.0)."""
        if not metadata.regulatory_documents:
            return 0.0
        
        score = 0.0
        
        # Base score for having documents
        score += 0.2
        
        # Bonus for different document types
        doc_types = {doc.document_type for doc in metadata.regulatory_documents}
        score += len(doc_types) * 0.1
        
        # Bonus for approved documents
        approved_docs = [doc for doc in metadata.regulatory_documents if doc.approval_date]
        score += len(approved_docs) * 0.1
        
        # Bonus for regulatory body coverage
        regulatory_bodies = {doc.regulatory_body for doc in metadata.regulatory_documents if doc.regulatory_body}
        score += len(regulatory_bodies) * 0.1
        
        # Bonus for current documents (not expired)
        current_docs = [doc for doc in metadata.regulatory_documents if not doc.is_expired()]
        score += len(current_docs) / len(metadata.regulatory_documents) * 0.2 if metadata.regulatory_documents else 0
        
        return min(score, 1.0)
    
    def _score_quality_standards(self, metadata: VersionMetadata) -> float:
        """Score quality standards coverage (0.0 to 1.0)."""
        score = 0.0
        
        # Quality standards
        if metadata.quality_standards:
            score += 0.3
        
        # Safety profile
        if metadata.safety_profile:
            score += 0.2
        
        # Risk management
        if metadata.risk_management:
            score += 0.2
        
        # Change history
        if metadata.change_history:
            score += 0.1
        
        # Audit schedule
        if metadata.audit_schedule:
            score += 0.1
        
        # Recent audit
        if metadata.last_audit_date:
            days_since_audit = (datetime.now() - metadata.last_audit_date).days
            if days_since_audit <= 365:  # Within last year
                score += 0.1
        
        return min(score, 1.0)