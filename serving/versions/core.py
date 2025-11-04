"""
Core model versioning components for medical AI assistant.

This module provides the foundation for semantic versioning with
medical compliance tracking and version management.
"""

import json
import uuid
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VersionType(Enum):
    """Types of model versions in medical compliance context."""
    MAJOR = "major"
    MINOR = "minor" 
    PATCH = "patch"
    MEDICAL_CRITICAL = "medical_critical"
    CLINICAL_RELEASE = "clinical_release"
    RESEARCH = "research"
    EMERGENCY = "emergency"


class ComplianceLevel(Enum):
    """Medical compliance levels for version tracking."""
    UNKNOWN = "unknown"
    PRE_CLINICAL = "pre_clinical"
    CLINICAL_INVESTIGATION = "clinical_investigation"
    CLINICAL_VALIDATION = "clinical_validation"
    PRODUCTION = "production"
    WITHDRAWN = "withdrawn"
    DEPRECATED = "deprecated"


class VersionStatus(Enum):
    """Status of model version in lifecycle."""
    DRAFT = "draft"
    DEVELOPMENT = "development"
    TESTING = "testing"
    VALIDATION = "validation"
    PRODUCTION = "production"
    MAINTENANCE = "maintenance"
    RETIRED = "retired"
    ARCHIVED = "archived"


@dataclass
class ComplianceMetadata:
    """Medical compliance metadata for version tracking."""
    compliance_level: ComplianceLevel = ComplianceLevel.UNKNOWN
    regulatory_status: str = ""
    clinical_approval_date: Optional[datetime] = None
    approval_authority: str = ""
    medical_device_class: str = ""
    intended_use: str = ""
    contraindications: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    validation_protocol: str = ""
    clinical_data_file: Optional[str] = None
    adverse_events: List[Dict[str, Any]] = field(default_factory=list)

    def add_audit_entry(self, action: str, user: str, details: Dict[str, Any] = None):
        """Add entry to compliance audit trail."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "user": user,
            "details": details or {}
        }
        self.audit_trail.append(entry)


@dataclass
class ModelVersion:
    """
    Semantic versioning with medical compliance tracking.
    
    Implements semantic versioning (major.minor.patch) with medical
    compliance tracking and regulatory metadata.
    """
    
    # Core version information
    version: str
    model_name: str
    model_type: str
    description: str = ""
    changelog: List[str] = field(default_factory=list)
    
    # Version metadata
    version_type: VersionType = VersionType.MINOR
    status: VersionStatus = VersionStatus.DEVELOPMENT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = ""
    updated_at: Optional[datetime] = None
    
    # Dependencies and compatibility
    parent_version: Optional[str] = None
    child_versions: List[str] = field(default_factory=list)
    deprecated_versions: List[str] = field(default_factory=list)
    breaking_changes: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    # Medical compliance
    compliance: ComplianceMetadata = field(default_factory=ComplianceMetadata)
    
    # Performance and metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    validation_scores: Dict[str, float] = field(default_factory=dict)
    
    # Technical details
    model_path: str = ""
    config_path: str = ""
    artifacts: List[str] = field(default_factory=list)
    framework_version: str = ""
    training_data_version: str = ""
    
    # Deployment and distribution
    deployment_status: str = "not_deployed"
    deployment_targets: List[str] = field(default_factory=list)
    rollout_percentage: float = 0.0
    canary_percentage: float = 0.0
    
    # Documentation and validation
    documentation_links: List[str] = field(default_factory=list)
    clinical_validation_files: List[str] = field(default_factory=list)
    approval_documents: List[str] = field(default_factory=list)
    
    # Internal tracking
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _hash: str = ""
    
    def __post_init__(self):
        """Validate and initialize version after creation."""
        self._validate_version_format()
        self._generate_hash()
        
        # Set updated timestamp
        if not self.updated_at:
            self.updated_at = self.created_at
    
    def _validate_version_format(self):
        """Validate semantic version format (major.minor.patch)."""
        version_pattern = r'^\d+\.\d+\.\d+$'
        if not re.match(version_pattern, self.version):
            raise ValueError(f"Invalid semantic version format: {self.version}. "
                           "Expected format: major.minor.patch (e.g., 1.2.3)")
    
    def _generate_hash(self):
        """Generate unique hash for version identification."""
        version_data = {
            "model_name": self.model_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "model_path": self.model_path
        }
        # Simple hash for identification (in production, use proper hashing)
        self._hash = str(hash(json.dumps(version_data, sort_keys=True)))
    
    def increment_version(self, version_type: VersionType) -> str:
        """Increment version based on type."""
        major, minor, patch = map(int, self.version.split('.'))
        
        if version_type == VersionType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif version_type == VersionType.MINOR:
            minor += 1
            patch = 0
        elif version_type == VersionType.PATCH:
            patch += 1
        else:
            raise ValueError(f"Cannot increment version for type: {version_type}")
        
        return f"{major}.{minor}.{patch}"
    
    def is_compatible_with(self, other_version: 'ModelVersion') -> bool:
        """Check version compatibility."""
        # Major version incompatibility
        self_major = int(self.version.split('.')[0])
        other_major = int(other_version.version.split('.')[0])
        
        if self_major != other_major:
            return False
        
        # Check for breaking changes
        if other_version.version in self.breaking_changes:
            return False
        
        return True
    
    def can_rollback_to(self, target_version: str) -> bool:
        """Check if can rollback to target version."""
        # Cannot rollback to itself
        if self.version == target_version:
            return False
        
        # Check if target version is in child versions
        if target_version not in self.child_versions:
            return False
        
        # Check compliance requirements
        target = self.registry.get_version(self.model_name, target_version) if self.registry else None
        if target and target.compliance.compliance_level == ComplianceLevel.PRODUCTION:
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        # Handle datetime serialization
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        if self.compliance.clinical_approval_date:
            data['compliance']['clinical_approval_date'] = \
                self.compliance.clinical_approval_date.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create ModelVersion from dictionary."""
        # Handle datetime deserialization
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        if 'compliance' in data and 'clinical_approval_date' in data['compliance']:
            if isinstance(data['compliance']['clinical_approval_date'], str):
                data['compliance']['clinical_approval_date'] = \
                    datetime.fromisoformat(data['compliance']['clinical_approval_date'])
        
        return cls(**data)
    
    def validate_compliance(self) -> Dict[str, List[str]]:
        """Validate compliance requirements."""
        errors = []
        warnings = []
        
        # Check compliance level requirements
        if self.compliance.compliance_level == ComplianceLevel.PRODUCTION:
            if not self.compliance.clinical_approval_date:
                errors.append("Production models require clinical approval date")
            
            if not self.compliance.approval_authority:
                errors.append("Production models require approval authority")
            
            if not self.compliance.intended_use:
                errors.append("Production models require intended use statement")
            
            if not self.compliance.risk_assessment:
                errors.append("Production models require risk assessment")
        
        # Check documentation requirements
        if not self.documentation_links:
            warnings.append("No documentation links provided")
        
        if self.compliance.compliance_level == ComplianceLevel.CLINICAL_VALIDATION:
            if not self.clinical_validation_files:
                errors.append("Clinical validation requires validation files")
        
        # Check performance metrics
        if not self.performance_metrics:
            warnings.append("No performance metrics recorded")
        
        return {"errors": errors, "warnings": warnings}
    
    def save_to_file(self, filepath: Union[str, Path]):
        """Save version metadata to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"ModelVersion saved to {filepath}")


@dataclass 
class VersionRegistry:
    """Registry for managing model versions."""
    
    registry_path: Path
    max_versions_per_model: int = 10
    
    def __post_init__(self):
        """Initialize registry."""
        self.registry_path = Path(self.registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._index_file = self.registry_path / "index.json"
        self._index = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load registry index."""
        if self._index_file.exists():
            with open(self._index_file) as f:
                return json.load(f)
        return {"models": {}, "latest_versions": {}, "metadata": {}}
    
    def _save_index(self):
        """Save registry index."""
        with open(self._index_file, 'w') as f:
            json.dump(self._index, f, indent=2)
    
    def register_version(self, version: ModelVersion) -> bool:
        """Register new model version."""
        try:
            # Check if version already exists
            if version.model_name in self._index["models"]:
                existing_versions = self._index["models"][version.model_name]
                if version.version in [v.get("version") for v in existing_versions]:
                    logger.warning(f"Version {version.version} already exists for {version.model_name}")
                    return False
            
            # Add to index
            if version.model_name not in self._index["models"]:
                self._index["models"][version.model_name] = []
            
            version_data = version.to_dict()
            self._index["models"][version.model_name].append(version_data)
            
            # Update latest version
            self._index["latest_versions"][version.model_name] = version.version
            
            # Maintain version limit
            versions = self._index["models"][version.model_name]
            if len(versions) > self.max_versions_per_model:
                # Keep latest versions, archive older ones
                versions.sort(key=lambda x: x["created_at"], reverse=True)
                self._index["models"][version.model_name] = versions[:self.max_versions_per_model]
            
            # Save index and version file
            self._save_index()
            
            version_path = self.registry_path / version.model_name / f"{version.version}.json"
            version_path.parent.mkdir(parents=True, exist_ok=True)
            version.save_to_file(version_path)
            
            logger.info(f"Registered version {version.version} for {version.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register version: {e}")
            return False
    
    def get_version(self, model_name: str, version: str = None) -> Optional[ModelVersion]:
        """Get specific version or latest version."""
        if model_name not in self._index["models"]:
            return None
        
        versions = self._index["models"][model_name]
        
        if version is None:
            # Return latest version
            latest = max(versions, key=lambda x: x["created_at"])
        else:
            # Return specific version
            versions = [v for v in versions if v["version"] == version]
            if not versions:
                return None
            latest = versions[0]
        
        return ModelVersion.from_dict(latest)
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model."""
        if model_name not in self._index["models"]:
            return []
        
        versions = self._index["models"][model_name]
        return sorted([v["version"] for v in versions])
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get latest version for a model."""
        return self.get_version(model_name)
    
    def deprecate_version(self, model_name: str, version: str, reason: str) -> bool:
        """Mark version as deprecated."""
        version_obj = self.get_version(model_name, version)
        if not version_obj:
            return False
        
        version_obj.status = VersionStatus.RETIRED
        version_obj.compliance.compliance_level = ComplianceLevel.DEPRECATED
        version_obj.compliance.add_audit_entry("deprecated", "system", {"reason": reason})
        
        return self.register_version(version_obj)
    
    def get_compliant_versions(self, model_name: str, 
                             min_compliance: ComplianceLevel) -> List[ModelVersion]:
        """Get versions that meet minimum compliance requirements."""
        versions = []
        level_order = {level: idx for idx, level in enumerate(ComplianceLevel)}
        
        for version_str in self.list_versions(model_name):
            version_obj = self.get_version(model_name, version_str)
            if version_obj:
                current_level = version_obj.compliance.compliance_level
                if level_order.get(current_level, -1) >= level_order.get(min_compliance, -1):
                    versions.append(version_obj)
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
    
    def find_compatible_versions(self, model_name: str, 
                               reference_version: str) -> List[ModelVersion]:
        """Find versions compatible with reference version."""
        ref = self.get_version(model_name, reference_version)
        if not ref:
            return []
        
        compatible = []
        for version_str in self.list_versions(model_name):
            candidate = self.get_version(model_name, version_str)
            if candidate and ref.is_compatible_with(candidate):
                compatible.append(candidate)
        
        return compatible
    
    def generate_compliance_report(self, model_name: str) -> Dict[str, Any]:
        """Generate compliance report for model."""
        versions = []
        for version_str in self.list_versions(model_name):
            version_obj = self.get_version(model_name, version_str)
            if version_obj:
                versions.append(version_obj)
        
        report = {
            "model_name": model_name,
            "total_versions": len(versions),
            "compliance_levels": {},
            "production_ready": [],
            "deprecated": [],
            "clinical_validated": [],
            "audit_trail": []
        }
        
        for version in versions:
            level = version.compliance.compliance_level.value
            report["compliance_levels"][level] = \
                report["compliance_levels"].get(level, 0) + 1
            
            if level == ComplianceLevel.PRODUCTION.value:
                report["production_ready"].append(version.version)
            elif level == ComplianceLevel.DEPRECATED.value:
                report["deprecated"].append(version.version)
            elif level == ComplianceLevel.CLINICAL_VALIDATION.value:
                report["clinical_validated"].append(version.version)
            
            report["audit_trail"].extend(version.compliance.audit_trail)
        
        return report


@dataclass
class VersionManager:
    """Manager for model version operations."""
    
    registry: VersionRegistry
    
    def create_new_version(self, model_name: str, version_type: VersionType,
                          created_by: str, description: str = "",
                          parent_version: str = None) -> Optional[ModelVersion]:
        """Create new version with proper increments."""
        # Get parent version
        parent = None
        if parent_version:
            parent = self.registry.get_version(model_name, parent_version)
        
        if not parent:
            parent = self.registry.get_latest_version(model_name)
        
        if not parent:
            # First version
            new_version_str = "1.0.0"
        else:
            new_version_str = parent.increment_version(version_type)
        
        # Create new version
        new_version = ModelVersion(
            version=new_version_str,
            model_name=model_name,
            model_type=parent.model_type if parent else "unknown",
            description=description,
            created_by=created_by,
            parent_version=parent.version if parent else None,
            version_type=version_type,
            status=VersionStatus.DEVELOPMENT
        )
        
        # Register in registry
        if self.registry.register_version(new_version):
            logger.info(f"Created new version {new_version_str} for {model_name}")
            return new_version
        
        return None
    
    def deploy_version(self, model_name: str, version: str, 
                      deployment_target: str, rollout_percentage: float) -> bool:
        """Deploy version to target."""
        version_obj = self.registry.get_version(model_name, version)
        if not version_obj:
            return False
        
        # Update deployment status
        version_obj.deployment_status = "deploying"
        version_obj.deployment_targets.append(deployment_target)
        version_obj.rollout_percentage = rollout_percentage
        
        # Add compliance audit entry
        version_obj.compliance.add_audit_entry(
            "deployment_started", "system",
            {"target": deployment_target, "rollout_percentage": rollout_percentage}
        )
        
        # Register updated version
        updated = self.registry.register_version(version_obj)
        
        if updated:
            logger.info(f"Deployed version {version} of {model_name} to {deployment_target}")
        
        return updated
    
    def rollback_version(self, model_name: str, current_version: str,
                        target_version: str, reason: str) -> bool:
        """Rollback from current to target version."""
        current = self.registry.get_version(model_name, current_version)
        target = self.registry.get_version(model_name, target_version)
        
        if not current or not target:
            return False
        
        if not current.can_rollback_to(target_version):
            return False
        
        # Add rollback audit entry
        target.compliance.add_audit_entry(
            "rollback", "system",
            {"from": current_version, "to": target_version, "reason": reason}
        )
        
        # Update target version status
        target.status = VersionStatus.PRODUCTION
        
        # Update current version status  
        current.status = VersionStatus.MAINTENANCE
        current.compliance.add_audit_entry(
            "rolled_back", "system", {"to": target_version}
        )
        
        # Register changes
        self.registry.register_version(target)
        self.registry.register_version(current)
        
        logger.info(f"Rolled back {model_name} from {current_version} to {target_version}")
        return True
    
    def validate_compliance(self, model_name: str, version: str) -> Dict[str, List[str]]:
        """Validate compliance for version."""
        version_obj = self.registry.get_version(model_name, version)
        if not version_obj:
            return {"errors": ["Version not found"], "warnings": []}
        
        validation_result = version_obj.validate_compliance()
        
        # Add validation to audit trail
        version_obj.compliance.add_audit_entry(
            "compliance_validation", "system", validation_result
        )
        
        # Register if validation passed (even if there are warnings)
        if not validation_result["errors"]:
            self.registry.register_version(version_obj)
        
        return validation_result
    
    def get_deployment_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get deployment history for model."""
        history = []
        
        for version_str in self.registry.list_versions(model_name):
            version_obj = self.registry.get_version(model_name, version_str)
            if version_obj and version_obj.deployment_targets:
                history.append({
                    "version": version_str,
                    "deployment_targets": version_obj.deployment_targets,
                    "rollout_percentage": version_obj.rollout_percentage,
                    "deployment_status": version_obj.deployment_status,
                    "timestamp": version_obj.created_at
                })
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)