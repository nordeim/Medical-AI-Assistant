"""
Version compatibility checking with backward/forward compatibility validation.

Provides comprehensive compatibility checking for model versions including
API compatibility, data format compatibility, and dependency validation.
"""

import json
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .core import ModelVersion, VersionType, ComplianceLevel

logger = logging.getLogger(__name__)


class CompatibilityType(Enum):
    """Types of compatibility checks."""
    BACKWARD = "backward"
    FORWARD = "forward"
    FULL = "full"
    API = "api"
    DATA_FORMAT = "data_format"
    DEPENDENCY = "dependency"
    MODEL_ARCHITECTURE = "model_architecture"


class CompatibilityLevel(Enum):
    """Compatibility level classification."""
    INCOMPATIBLE = "incompatible"
    MINOR_BREAKING = "minor_breaking"
    BACKWARD_COMPATIBLE = "backward_compatible"
    FORWARD_COMPATIBLE = "forward_compatible"
    FULLY_COMPATIBLE = "fully_compatible"


@dataclass
class CompatibilityRule:
    """Rule for checking compatibility."""
    name: str
    description: str
    check_function: str
    severity: str = "error"  # error, warning, info
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompatibilityCheck:
    """Result of a compatibility check."""
    rule_name: str
    check_type: CompatibilityType
    result: CompatibilityLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class VersionCompatibility:
    """Compatibility analysis between two model versions."""
    
    source_version: str
    target_version: str
    overall_compatibility: CompatibilityLevel
    api_compatibility: CompatibilityLevel
    data_format_compatibility: CompatibilityLevel
    dependency_compatibility: CompatibilityLevel
    model_architecture_compatibility: CompatibilityLevel
    checks_performed: List[CompatibilityCheck] = field(default_factory=list)
    migration_notes: List[str] = field(default_factory=list)
    rollback_plan: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_version": self.source_version,
            "target_version": self.target_version,
            "overall_compatibility": self.overall_compatibility.value,
            "api_compatibility": self.api_compatibility.value,
            "data_format_compatibility": self.data_format_compatibility.value,
            "dependency_compatibility": self.dependency_compatibility.value,
            "model_architecture_compatibility": self.model_architecture_compatibility.value,
            "checks_performed": [
                {
                    "rule_name": check.rule_name,
                    "check_type": check.check_type.value,
                    "result": check.result.value,
                    "message": check.message,
                    "details": check.details,
                    "timestamp": check.timestamp.isoformat(),
                    "recommendations": check.recommendations
                }
                for check in self.checks_performed
            ],
            "migration_notes": self.migration_notes,
            "rollback_plan": self.rollback_plan
        }


class CompatibilityChecker:
    """Advanced compatibility checker for model versions."""
    
    def __init__(self):
        self.rules = self._load_default_rules()
        self._initialize_compatibility_patterns()
    
    def _initialize_compatibility_patterns(self):
        """Initialize patterns for compatibility checking."""
        self.api_patterns = {
            "endpoint": r"^/api/v(\d+)/",
            "method": r"^(GET|POST|PUT|DELETE|PATCH)$",
            "parameter": r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        }
        
        self.data_format_patterns = {
            "json_schema": r"^{.*}$",
            "csv_columns": r"^[a-zA-Z_][a-zA-Z0-9_]*(,\s*[a-zA-Z_][a-zA-Z0-9_]*)*$",
            "feature_name": r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        }
    
    def _load_default_rules(self) -> List[CompatibilityRule]:
        """Load default compatibility rules."""
        return [
            CompatibilityRule(
                name="major_version_check",
                description="Check if major versions match",
                check_function="check_major_version_compatibility"
            ),
            CompatibilityRule(
                name="dependency_version_check",
                description="Check dependency versions",
                check_function="check_dependency_compatibility"
            ),
            CompatibilityRule(
                name="api_signature_check",
                description="Check API signature compatibility",
                check_function="check_api_signature_compatibility"
            ),
            CompatibilityRule(
                name="data_format_check",
                description="Check data format compatibility",
                check_function="check_data_format_compatibility"
            ),
            CompatibilityRule(
                name="model_architecture_check",
                description="Check model architecture compatibility",
                check_function="check_model_architecture_compatibility"
            ),
            CompatibilityRule(
                name="compliance_check",
                description="Check compliance level requirements",
                check_function="check_compliance_compatibility"
            ),
            CompatibilityRule(
                name="performance_threshold_check",
                description="Check performance thresholds",
                check_function="check_performance_compatibility"
            )
        ]
    
    def check_compatibility(self, 
                          source_version: ModelVersion,
                          target_version: ModelVersion,
                          check_types: List[CompatibilityType] = None) -> VersionCompatibility:
        """Perform comprehensive compatibility check."""
        if check_types is None:
            check_types = [CompatibilityType.FULL]
        
        # Initialize compatibility analysis
        analysis = VersionCompatibility(
            source_version=source_version.version,
            target_version=target_version.version,
            overall_compatibility=CompatibilityLevel.INCOMPATIBLE,
            api_compatibility=CompatibilityLevel.INCOMPATIBLE,
            data_format_compatibility=CompatibilityLevel.INCOMPATIBLE,
            dependency_compatibility=CompatibilityLevel.INCOMPATIBLE,
            model_architecture_compatibility=CompatibilityLevel.INCOMPATIBLE
        )
        
        # Perform individual checks
        api_checks = []
        data_checks = []
        dependency_checks = []
        architecture_checks = []
        
        for check_type in check_types:
            if check_type in [CompatibilityType.API, CompatibilityType.FULL]:
                api_checks = self._perform_api_checks(source_version, target_version)
                analysis.checks_performed.extend(api_checks)
            
            if check_type in [CompatibilityType.DATA_FORMAT, CompatibilityType.FULL]:
                data_checks = self._perform_data_format_checks(source_version, target_version)
                analysis.checks_performed.extend(data_checks)
            
            if check_type in [CompatibilityType.DEPENDENCY, CompatibilityType.FULL]:
                dependency_checks = self._perform_dependency_checks(source_version, target_version)
                analysis.checks_performed.extend(dependency_checks)
            
            if check_type in [CompatibilityType.MODEL_ARCHITECTURE, CompatibilityType.FULL]:
                architecture_checks = self._perform_architecture_checks(source_version, target_version)
                analysis.checks_performed.extend(architecture_checks)
        
        # Evaluate compatibility levels
        analysis.api_compatibility = self._evaluate_compatibility_level(api_checks)
        analysis.data_format_compatibility = self._evaluate_compatibility_level(data_checks)
        analysis.dependency_compatibility = self._evaluate_compatibility_level(dependency_checks)
        analysis.model_architecture_compatibility = self._evaluate_compatibility_level(architecture_checks)
        
        # Determine overall compatibility
        analysis.overall_compatibility = self._determine_overall_compatibility([
            analysis.api_compatibility,
            analysis.data_format_compatibility,
            analysis.dependency_compatibility,
            analysis.model_architecture_compatibility
        ])
        
        # Generate migration notes and rollback plan
        analysis.migration_notes = self._generate_migration_notes(analysis)
        analysis.rollback_plan = self._generate_rollback_plan(analysis)
        
        return analysis
    
    def _perform_api_checks(self, source: ModelVersion, target: ModelVersion) -> List[CompatibilityCheck]:
        """Perform API compatibility checks."""
        checks = []
        
        # Check major version compatibility
        source_major = int(source.version.split('.')[0])
        target_major = int(target.version.split('.')[0])
        
        if source_major == target_major:
            checks.append(CompatibilityCheck(
                rule_name="major_version_match",
                check_type=CompatibilityType.API,
                result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                message="Major versions match",
                details={"source_major": source_major, "target_major": target_major}
            ))
        else:
            checks.append(CompatibilityCheck(
                rule_name="major_version_mismatch",
                check_type=CompatibilityType.API,
                result=CompatibilityLevel.INCOMPATIBLE,
                message="Major version mismatch indicates breaking changes",
                details={"source_major": source_major, "target_major": target_major},
                recommendations=[
                    f"Migration required from v{source_major}.x to v{target_major}.x",
                    "Review breaking changes documentation",
                    "Update all API consumers"
                ]
            ))
        
        # Check breaking changes
        if target.version in source.breaking_changes:
            checks.append(CompatibilityCheck(
                rule_name="breaking_change_check",
                check_type=CompatibilityType.API,
                result=CompatibilityLevel.INCOMPATIBLE,
                message="Target version contains breaking changes",
                details={"breaking_version": target.version},
                recommendations=[
                    "Review breaking changes list",
                    "Update client code",
                    "Test all API endpoints"
                ]
            ))
        
        # Check endpoint signatures (mock check - would need actual API defs)
        source_api = getattr(source, 'api_signature', {})
        target_api = getattr(target, 'api_signature', {})
        
        if source_api == target_api:
            checks.append(CompatibilityCheck(
                rule_name="api_signature_match",
                check_type=CompatibilityType.API,
                result=CompatibilityLevel.FULLY_COMPATIBLE,
                message="API signatures match",
                details={"api_count": len(source_api)}
            ))
        elif self._are_api_signatures_backward_compatible(source_api, target_api):
            checks.append(CompatibilityCheck(
                rule_name="api_signature_backward_compatible",
                check_type=CompatibilityType.API,
                result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                message="API signatures are backward compatible",
                details={"source_api_count": len(source_api), "target_api_count": len(target_api)},
                recommendations=["Update client libraries to support new endpoints"]
            ))
        else:
            checks.append(CompatibilityCheck(
                rule_name="api_signature_incompatible",
                check_type=CompatibilityType.API,
                result=CompatibilityLevel.INCOMPATIBLE,
                message="API signatures are not backward compatible",
                details={"source_api_count": len(source_api), "target_api_count": len(target_api)},
                recommendations=["Update all API calls", "Review API documentation"]
            ))
        
        return checks
    
    def _perform_data_format_checks(self, source: ModelVersion, target: ModelVersion) -> List[CompatibilityCheck]:
        """Perform data format compatibility checks."""
        checks = []
        
        # Check input format compatibility
        source_input_format = getattr(source, 'input_format', {})
        target_input_format = getattr(target, 'input_format', {})
        
        if source_input_format == target_input_format:
            checks.append(CompatibilityCheck(
                rule_name="input_format_match",
                check_type=CompatibilityType.DATA_FORMAT,
                result=CompatibilityLevel.FULLY_COMPATIBLE,
                message="Input data formats match",
                details={"input_format": target_input_format}
            ))
        elif self._are_data_formats_compatible(source_input_format, target_input_format):
            checks.append(CompatibilityCheck(
                rule_name="input_format_backward_compatible",
                check_type=CompatibilityType.DATA_FORMAT,
                result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                message="Input data formats are backward compatible",
                details={"source_format": source_input_format, "target_format": target_input_format},
                recommendations=["Update data preprocessing if needed"]
            ))
        else:
            checks.append(CompatibilityCheck(
                rule_name="input_format_incompatible",
                check_type=CompatibilityType.DATA_FORMAT,
                result=CompatibilityLevel.INCOMPATIBLE,
                message="Input data formats are not compatible",
                details={"source_format": source_input_format, "target_format": target_input_format},
                recommendations=["Update data pipeline", "Convert existing data"]
            ))
        
        # Check output format compatibility
        source_output_format = getattr(source, 'output_format', {})
        target_output_format = getattr(target, 'output_format', {})
        
        if source_output_format == target_output_format:
            checks.append(CompatibilityCheck(
                rule_name="output_format_match",
                check_type=CompatibilityType.DATA_FORMAT,
                result=CompatibilityLevel.FULLY_COMPATIBLE,
                message="Output data formats match",
                details={"output_format": target_output_format}
            ))
        else:
            checks.append(CompatibilityCheck(
                rule_name="output_format_changed",
                check_type=CompatibilityType.DATA_FORMAT,
                result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                message="Output data formats changed",
                details={"source_format": source_output_format, "target_format": target_output_format},
                recommendations=["Update code consuming model outputs"]
            ))
        
        return checks
    
    def _perform_dependency_checks(self, source: ModelVersion, target: ModelVersion) -> List[CompatibilityCheck]:
        """Perform dependency compatibility checks."""
        checks = []
        
        # Check framework version compatibility
        source_framework = source.framework_version
        target_framework = target.framework_version
        
        if source_framework == target_framework:
            checks.append(CompatibilityCheck(
                rule_name="framework_version_match",
                check_type=CompatibilityType.DEPENDENCY,
                result=CompatibilityLevel.FULLY_COMPATIBLE,
                message="Framework versions match",
                details={"framework_version": target_framework}
            ))
        elif self._is_framework_compatible(source_framework, target_framework):
            checks.append(CompatibilityCheck(
                rule_name="framework_version_compatible",
                check_type=CompatibilityType.DEPENDENCY,
                result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                message="Framework versions are compatible",
                details={"source_framework": source_framework, "target_framework": target_framework},
                recommendations=["Test thoroughly with new framework version"]
            ))
        else:
            checks.append(CompatibilityCheck(
                rule_name="framework_version_incompatible",
                check_type=CompatibilityType.DEPENDENCY,
                result=CompatibilityLevel.INCOMPATIBLE,
                message="Framework versions are incompatible",
                details={"source_framework": source_framework, "target_framework": target_framework},
                recommendations=["Update dependency constraints", "Test migration"]
            ))
        
        # Check training data version compatibility
        if source.training_data_version and target.training_data_version:
            if source.training_data_version == target.training_data_version:
                checks.append(CompatibilityCheck(
                    rule_name="training_data_version_match",
                    check_type=CompatibilityType.DEPENDENCY,
                    result=CompatibilityLevel.FULLY_COMPATIBLE,
                    message="Training data versions match",
                    details={"training_data_version": target.training_data_version}
                ))
            else:
                checks.append(CompatibilityCheck(
                    rule_name="training_data_version_changed",
                    check_type=CompatibilityType.DEPENDENCY,
                    result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                    message="Training data version changed",
                    details={"source_data_version": source.training_data_version, 
                           "target_data_version": target.training_data_version},
                    recommendations=["Verify data compatibility", "Retrain if necessary"]
                ))
        
        # Check explicit dependencies
        source_deps = source.dependencies
        target_deps = target.dependencies
        
        dep_compatibility = self._check_explicit_dependencies(source_deps, target_deps)
        
        checks.append(CompatibilityCheck(
            rule_name="explicit_dependencies_check",
            check_type=CompatibilityType.DEPENDENCY,
            result=dep_compatibility["level"],
            message=dep_compatibility["message"],
            details=dep_compatibility["details"],
            recommendations=dep_compatibility["recommendations"]
        ))
        
        return checks
    
    def _perform_architecture_checks(self, source: ModelVersion, target: ModelVersion) -> List[CompatibilityCheck]:
        """Perform model architecture compatibility checks."""
        checks = []
        
        # Check model type compatibility
        if source.model_type == target.model_type:
            checks.append(CompatibilityCheck(
                rule_name="model_type_match",
                check_type=CompatibilityType.MODEL_ARCHITECTURE,
                result=CompatibilityLevel.FULLY_COMPATIBLE,
                message="Model types match",
                details={"model_type": target.model_type}
            ))
        else:
            # Different model types might still be compatible if they have same interface
            checks.append(CompatibilityCheck(
                rule_name="model_type_changed",
                check_type=CompatibilityType.MODEL_ARCHITECTURE,
                result=CompatibilityLevel.BACKWARD_COMPATIBLE,
                message="Model type changed",
                details={"source_type": source.model_type, "target_type": target.model_type},
                recommendations=["Verify interface compatibility", "Update model loading code"]
            ))
        
        # Check compliance requirements for architecture changes
        if source.compliance.compliance_level != target.compliance.compliance_level:
            level_compatibility = self._check_compliance_compatibility(
                source.compliance.compliance_level,
                target.compliance.compliance_level
            )
            
            checks.append(CompatibilityCheck(
                rule_name="compliance_level_change",
                check_type=CompatibilityType.MODEL_ARCHITECTURE,
                result=level_compatibility["level"],
                message=level_compatibility["message"],
                details={"source_level": source.compliance.compliance_level.value,
                        "target_level": target.compliance.compliance_level.value},
                recommendations=level_compatibility["recommendations"]
            ))
        
        # Check model artifacts compatibility
        if self._are_model_artifacts_compatible(source, target):
            checks.append(CompatibilityCheck(
                rule_name="model_artifacts_compatible",
                check_type=CompatibilityType.MODEL_ARCHITECTURE,
                result=CompatibilityLevel.FULLY_COMPATIBLE,
                message="Model artifacts are compatible",
                details={"artifacts_count": len(target.artifacts)}
            ))
        else:
            checks.append(CompatibilityCheck(
                rule_name="model_artifacts_incompatible",
                check_type=CompatibilityType.MODEL_ARCHITECTURE,
                result=CompatibilityLevel.INCOMPATIBLE,
                message="Model artifacts are not compatible",
                details={"source_artifacts": source.artifacts, "target_artifacts": target.artifacts},
                recommendations=["Update model loading code", "Verify artifact formats"]
            ))
        
        return checks
    
    def _evaluate_compatibility_level(self, checks: List[CompatibilityCheck]) -> CompatibilityLevel:
        """Evaluate overall compatibility level from individual checks."""
        if not checks:
            return CompatibilityLevel.INCOMPATIBLE
        
        # Count results
        results = [check.result for check in checks]
        
        if CompatibilityLevel.INCOMPATIBLE in results:
            return CompatibilityLevel.INCOMPATIBLE
        elif CompatibilityLevel.MINOR_BREAKING in results:
            return CompatibilityLevel.MINOR_BREAKING
        elif all(result == CompatibilityLevel.BACKWARD_COMPATIBLE for result in results):
            return CompatibilityLevel.BACKWARD_COMPATIBLE
        elif all(result == CompatibilityLevel.FORWARD_COMPATIBLE for result in results):
            return CompatibilityLevel.FORWARD_COMPATIBLE
        elif all(result == CompatibilityLevel.FULLY_COMPATIBLE for result in results):
            return CompatibilityLevel.FULLY_COMPATIBLE
        
        return CompatibilityLevel.MINOR_BREAKING
    
    def _determine_overall_compatibility(self, levels: List[CompatibilityLevel]) -> CompatibilityLevel:
        """Determine overall compatibility from individual levels."""
        if CompatibilityLevel.INCOMPATIBLE in levels:
            return CompatibilityLevel.INCOMPATIBLE
        
        incompatible_count = levels.count(CompatibilityLevel.INCOMPATIBLE)
        minor_breaking_count = levels.count(CompatibilityLevel.MINOR_BREAKING)
        backward_compatible_count = levels.count(CompatibilityLevel.BACKWARD_COMPATIBLE)
        forward_compatible_count = levels.count(CompatibilityLevel.FORWARD_COMPATIBLE)
        fully_compatible_count = levels.count(CompatibilityLevel.FULLY_COMPATIBLE)
        
        # If most checks are fully compatible, overall is fully compatible
        if fully_compatible_count >= len(levels) * 0.8:
            return CompatibilityLevel.FULLY_COMPATIBLE
        
        # If most checks are backward compatible, overall is backward compatible
        if backward_compatible_count + fully_compatible_count >= len(levels) * 0.7:
            return CompatibilityLevel.BACKWARD_COMPATIBLE
        
        # If there are some minor breaking changes
        if minor_breaking_count > 0:
            return CompatibilityLevel.MINOR_BREAKING
        
        return CompatibilityLevel.BACKWARD_COMPATIBLE
    
    def _generate_migration_notes(self, analysis: VersionCompatibility) -> List[str]:
        """Generate migration notes based on compatibility analysis."""
        notes = []
        
        # API compatibility notes
        if analysis.api_compatibility == CompatibilityLevel.INCOMPATIBLE:
            notes.append("Critical API breaking changes detected - immediate migration required")
        elif analysis.api_compatibility == CompatibilityLevel.MINOR_BREAKING:
            notes.append("Minor API breaking changes - review and update API calls")
        elif analysis.api_compatibility == CompatibilityLevel.BACKWARD_COMPATIBLE:
            notes.append("API is backward compatible - existing clients will continue to work")
        
        # Data format notes
        if analysis.data_format_compatibility == CompatibilityLevel.INCOMPATIBLE:
            notes.append("Data format changes are not backward compatible - data migration required")
        elif analysis.data_format_compatibility == CompatibilityLevel.BACKWARD_COMPATIBLE:
            notes.append("Data format changes are backward compatible - existing data can be processed")
        
        # Dependency notes
        if analysis.dependency_compatibility == CompatibilityLevel.INCOMPATIBLE:
            notes.append("Dependency changes are incompatible - update dependency constraints")
        
        # Architecture notes
        if analysis.model_architecture_compatibility == CompatibilityLevel.INCOMPATIBLE:
            notes.append("Model architecture changes are incompatible - code refactoring required")
        
        return notes
    
    def _generate_rollback_plan(self, analysis: VersionCompatibility) -> str:
        """Generate rollback plan based on compatibility analysis."""
        if analysis.overall_compatibility == CompatibilityLevel.FULLY_COMPATIBLE:
            return "No rollback needed - changes are fully compatible"
        
        if analysis.overall_compatibility == CompatibilityLevel.BACKWARD_COMPATIBLE:
            return "Simple rollback possible - revert to previous version if issues arise"
        
        if analysis.overall_compatibility == CompatibilityLevel.MINOR_BREAKING:
            return "Moderate rollback complexity - test thoroughly before full deployment"
        
        # Incompatible - complex rollback required
        return """Complex rollback procedure required:
1. Revert to previous version
2. Update dependencies if necessary
3. Restore previous data formats
4. Verify all API endpoints
5. Run full regression testing"""
    
    # Helper methods for compatibility checks
    def _are_api_signatures_backward_compatible(self, source_api: Dict, target_api: Dict) -> bool:
        """Check if API signatures are backward compatible."""
        # Simplified check - in practice would need more sophisticated analysis
        source_endpoints = set(source_api.keys())
        target_endpoints = set(target_api.keys())
        
        # All source endpoints should be present in target (backward compatibility)
        return source_endpoints.issubset(target_endpoints)
    
    def _are_data_formats_compatible(self, source_format: Dict, target_format: Dict) -> bool:
        """Check if data formats are backward compatible."""
        # Simplified check - would need schema validation in practice
        return True  # Placeholder
    
    def _is_framework_compatible(self, source_framework: str, target_framework: str) -> bool:
        """Check if framework versions are compatible."""
        # Simplified semantic versioning check
        source_parts = source_framework.split('.')
        target_parts = target_framework.split('.')
        
        # Same major version usually means compatibility
        if len(source_parts) > 0 and len(target_parts) > 0:
            return source_parts[0] == target_parts[0]
        
        return False
    
    def _check_explicit_dependencies(self, source_deps: Dict, target_deps: Dict) -> Dict[str, Any]:
        """Check explicit dependency compatibility."""
        added_deps = set(target_deps.keys()) - set(source_deps.keys())
        removed_deps = set(source_deps.keys()) - set(target_deps.keys())
        version_changes = {}
        
        for dep in set(source_deps.keys()) & set(target_deps.keys()):
            if source_deps[dep] != target_deps[dep]:
                version_changes[dep] = {
                    "from": source_deps[dep],
                    "to": target_deps[dep]
                }
        
        if removed_deps:
            return {
                "level": CompatibilityLevel.INCOMPATIBLE,
                "message": f"Removed dependencies: {removed_deps}",
                "details": {"removed": list(removed_deps), "added": list(added_deps), "version_changes": version_changes},
                "recommendations": ["Review dependency removal impact", "Update dependency constraints"]
            }
        elif version_changes:
            return {
                "level": CompatibilityLevel.MINOR_BREAKING,
                "message": f"Dependency version changes: {list(version_changes.keys())}",
                "details": {"version_changes": version_changes},
                "recommendations": ["Test with new dependency versions", "Update version constraints"]
            }
        else:
            return {
                "level": CompatibilityLevel.FULLY_COMPATIBLE,
                "message": "Dependencies are identical",
                "details": {},
                "recommendations": []
            }
    
    def _are_model_artifacts_compatible(self, source: ModelVersion, target: ModelVersion) -> bool:
        """Check if model artifacts are compatible."""
        # Simplified check - would need actual file format validation
        return len(target.artifacts) >= len(source.artifacts) * 0.8  # At least 80% artifacts preserved
    
    def _check_compliance_compatibility(self, source_level: ComplianceLevel, target_level: ComplianceLevel) -> Dict[str, Any]:
        """Check compliance level compatibility."""
        level_order = {
            ComplianceLevel.UNKNOWN: 0,
            ComplianceLevel.PRE_CLINICAL: 1,
            ComplianceLevel.CLINICAL_INVESTIGATION: 2,
            ComplianceLevel.CLINICAL_VALIDATION: 3,
            ComplianceLevel.PRODUCTION: 4,
            ComplianceLevel.DEPRECATED: -1,
            ComplianceLevel.WITHDRAWN: -2
        }
        
        source_order = level_order.get(source_level, 0)
        target_order = level_order.get(target_level, 0)
        
        if source_order == target_order:
            return {
                "level": CompatibilityLevel.FULLY_COMPATIBLE,
                "message": "Same compliance level",
                "details": {"level": target_level.value},
                "recommendations": []
            }
        elif target_order > source_order:
            return {
                "level": CompatibilityLevel.FULLY_COMPATIBLE,
                "message": "Compliance level increased (better)",
                "details": {"from": source_level.value, "to": target_level.value},
                "recommendations": ["Verify new compliance requirements are met"]
            }
        else:
            return {
                "level": CompatibilityLevel.BACKWARD_COMPATIBLE,
                "message": "Compliance level decreased (caution required)",
                "details": {"from": source_level.value, "to": target_level.value},
                "recommendations": ["Review compliance implications", "Ensure regulatory compliance"]
            }
    
    def validate_rollback_safety(self, source_version: ModelVersion, target_version: str) -> Dict[str, Any]:
        """Validate if rollback to target version is safe."""
        target = source_version.registry.get_version(source_version.model_name, target_version) if source_version.registry else None
        
        if not target:
            return {"safe": False, "reason": "Target version not found"}
        
        # Check compatibility in reverse direction (target to source)
        compatibility = self.check_compatibility(target, source_version)
        
        # Check if rollback introduces breaking changes
        if compatibility.overall_compatibility == CompatibilityLevel.INCOMPATIBLE:
            return {
                "safe": False,
                "reason": "Rollback introduces breaking changes",
                "compatibility": compatibility.to_dict()
            }
        
        # Check compliance requirements
        if target.compliance.compliance_level == ComplianceLevel.WITHDRAWN:
            return {
                "safe": False,
                "reason": "Target version is withdrawn",
                "compatibility": compatibility.to_dict()
            }
        
        # Check performance regression
        if self._check_performance_regression(source_version, target):
            return {
                "safe": True,
                "safe_with_notifications": True,
                "reason": "Performance regression detected but rollback is safe",
                "compatibility": compatibility.to_dict()
            }
        
        return {
            "safe": True,
            "safe_with_notifications": False,
            "reason": "Rollback is safe",
            "compatibility": compatibility.to_dict()
        }
    
    def _check_performance_regression(self, source: ModelVersion, target: ModelVersion) -> bool:
        """Check if rollback introduces performance regression."""
        # Simplified performance check
        source_metrics = source.performance_metrics
        target_metrics = target.performance_metrics
        
        for metric_name in source_metrics:
            if metric_name in target_metrics:
                # If source performance is significantly better than target
                source_value = source_metrics[metric_name]
                target_value = target_metrics[metric_name]
                
                # Simple threshold check (in practice would be more sophisticated)
                if source_value > target_value * 1.1:  # 10% degradation threshold
                    return True
        
        return False