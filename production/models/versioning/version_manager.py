"""
Model Versioning and Compatibility Management
Semantic versioning, dependency tracking, and backward compatibility for medical AI models.
"""

import os
import sys
import logging
import json
import hashlib
import semver
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import semantic_version

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'registry'))
from model_registry import ModelRegistry

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    """Model version information"""
    name: str
    version: str  # Semantic version
    created_at: datetime
    created_by: str
    description: str
    parent_version: Optional[str] = None
    breaking_changes: List[str] = None
    new_features: List[str] = None
    bug_fixes: List[str] = None
    dependencies: Dict[str, str] = None  # model_name -> version
    compatibility_matrix: Dict[str, List[str]] = None  # version -> [compatible versions]
    metadata: Dict[str, Any] = None

@dataclass
class CompatibilityCheck:
    """Compatibility check result"""
    model_a: str
    version_a: str
    model_b: str
    version_b: str
    is_compatible: bool
    compatibility_level: str  # full, partial, breaking
    issues: List[str]
    recommendations: List[str]

@dataclass
class VersionCompatibilityMatrix:
    """Version compatibility matrix"""
    model_name: str
    versions: List[str]
    compatibility_matrix: Dict[str, Dict[str, str]]  # version_a -> version_b -> compatibility_level
    last_updated: datetime

class ModelVersionManager:
    """Production model versioning and compatibility management"""
    
    def __init__(self, config_path: str = "config/versioning_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Version registry
        self.version_registry: Dict[str, List[ModelVersion]] = {}
        self.compatibility_matrices: Dict[str, VersionCompatibilityMatrix] = {}
        
        # Semantic versioning rules
        self.versioning_rules = self.config.get("versioning_rules", {})
        
        # Model registry integration
        self.model_registry = ModelRegistry()
        
        # Version storage
        self.version_storage_path = self.config.get("version_storage_path", "/tmp/model_versions")
        
        # Compatibility rules
        self.compatibility_rules = self.config.get("compatibility_rules", {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load versioning configuration"""
        default_config = {
            "version_storage_path": "/tmp/model_versions",
            "versioning_rules": {
                "major_version_for": ["breaking_api_changes", "model_architecture_changes"],
                "minor_version_for": ["new_features", "accuracy_improvements"],
                "patch_version_for": ["bug_fixes", "performance_improvements"],
                "pre_release_tags": ["alpha", "beta", "rc"]
            },
            "compatibility_rules": {
                "auto_detect_compatibility": True,
                "require_compatibility_testing": True,
                "backward_compatibility_threshold": 0.95,
                "cross_model_compatibility": True
            },
            "semantic_versioning": {
                "enabled": True,
                "validate_versions": True,
                "auto_increment_patch": True
            }
        }
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
                return default_config
        except FileNotFoundError:
            logger.warning(f"Versioning config {config_path} not found, using defaults")
            return default_config
    
    def create_model_version(self, model_name: str, version_type: str = "patch",
                           description: str = "", created_by: str = "system",
                           parent_version: Optional[str] = None,
                           changes: Optional[Dict[str, List[str]]] = None) -> ModelVersion:
        """Create a new model version with semantic versioning"""
        try:
            # Get existing versions
            existing_versions = self.version_registry.get(model_name, [])
            
            # Determine next version
            next_version = self._determine_next_version(model_name, version_type, parent_version)
            
            # Parse changes
            change_data = changes or {}
            
            # Create version
            model_version = ModelVersion(
                name=model_name,
                version=next_version,
                created_at=datetime.utcnow(),
                created_by=created_by,
                description=description,
                parent_version=parent_version,
                breaking_changes=change_data.get("breaking_changes", []),
                new_features=change_data.get("new_features", []),
                bug_fixes=change_data.get("bug_fixes", []),
                dependencies=self._extract_dependencies(model_name, next_version),
                compatibility_matrix={},
                metadata={
                    "version_type": version_type,
                    "auto_generated": False,
                    "status": "development"
                }
            )
            
            # Store version
            if model_name not in self.version_registry:
                self.version_registry[model_name] = []
            
            self.version_registry[model_name].append(model_version)
            
            # Update compatibility matrix
            self._update_compatibility_matrix(model_name)
            
            # Save version to storage
            self._save_version_to_storage(model_version)
            
            logger.info(f"Created model version: {model_name} v{next_version}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Model version creation failed: {str(e)}")
            raise
    
    def _determine_next_version(self, model_name: str, version_type: str, 
                              parent_version: Optional[str]) -> str:
        """Determine the next semantic version"""
        existing_versions = self.version_registry.get(model_name, [])
        
        if not existing_versions and not parent_version:
            # First version
            return "1.0.0"
        
        # Use parent version or latest version
        base_version = parent_version
        if not base_version:
            # Get latest version
            latest_version = self._get_latest_version(model_name)
            if latest_version:
                base_version = latest_version.version
            else:
                base_version = "1.0.0"
        
        # Parse base version
        try:
            semver_version = semantic_version.Version(base_version)
        except Exception:
            # Fallback to basic versioning
            return f"{int(base_version.split('.')[0]) + (1 if version_type == 'major' else 0)}.{int(base_version.split('.')[1]) + (1 if version_type == 'minor' else 0)}.{int(base_version.split('.')[2]) + 1}"
        
        # Increment based on version type
        if version_type == "major":
            semver_version.major += 1
            semver_version.minor = 0
            semver_version.patch = 0
        elif version_type == "minor":
            semver_version.minor += 1
            semver_version.patch = 0
        else:  # patch
            semver_version.patch += 1
        
        return str(semver_version)
    
    def _get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        versions = self.version_registry.get(model_name, [])
        if not versions:
            return None
        
        # Sort by semantic version
        versions.sort(key=lambda v: semantic_version.Version(v.version), reverse=True)
        return versions[0]
    
    def _extract_dependencies(self, model_name: str, version: str) -> Dict[str, str]:
        """Extract dependencies for a model version"""
        # In production, this would analyze model requirements, data schemas, etc.
        # For demo, return mock dependencies
        
        dependencies = {}
        
        # Add common medical AI dependencies
        common_deps = {
            "scikit-learn": ">=1.0.0",
            "numpy": ">=1.19.0",
            "pandas": ">=1.3.0"
        }
        
        # Add model-specific dependencies based on version
        if "pytorch" in version.lower():
            dependencies["torch"] = ">=1.9.0"
        elif "tensorflow" in version.lower():
            dependencies["tensorflow"] = ">=2.6.0"
        
        return dependencies
    
    def _update_compatibility_matrix(self, model_name: str):
        """Update compatibility matrix for a model"""
        try:
            versions = self.version_registry.get(model_name, [])
            if len(versions) < 2:
                return
            
            # Create compatibility matrix
            matrix = {}
            
            for version_a in versions:
                matrix[version_a.version] = {}
                
                for version_b in versions:
                    if version_a.version == version_b.version:
                        compatibility = "identical"
                    else:
                        compatibility = self._check_version_compatibility(version_a, version_b)
                    
                    matrix[version_a.version][version_b.version] = compatibility
            
            # Create or update matrix
            compatibility_matrix = VersionCompatibilityMatrix(
                model_name=model_name,
                versions=[v.version for v in versions],
                compatibility_matrix=matrix,
                last_updated=datetime.utcnow()
            )
            
            self.compatibility_matrices[model_name] = compatibility_matrix
            
            # Update individual version compatibility matrices
            for version in versions:
                version.compatibility_matrix = matrix[version.version]
            
            logger.info(f"Updated compatibility matrix for {model_name}: {len(versions)} versions")
            
        except Exception as e:
            logger.error(f"Compatibility matrix update failed: {str(e)}")
    
    def _check_version_compatibility(self, version_a: ModelVersion, 
                                   version_b: ModelVersion) -> str:
        """Check compatibility between two model versions"""
        try:
            # Check for breaking changes
            if version_a.breaking_changes and version_b.breaking_changes:
                # If either has breaking changes, check if they're compatible
                if version_a.breaking_changes and version_b.breaking_changes:
                    # Check if breaking changes are related
                    shared_breaking = set(version_a.breaking_changes) & set(version_b.breaking_changes)
                    if shared_breaking:
                        return "breaking"
            
            # Check semantic version compatibility
            semver_a = semantic_version.Version(version_a.version)
            semver_b = semantic_version.Version(version_b.version)
            
            # Same major version = backward compatible
            if semver_a.major == semver_b.major:
                if semver_a.minor > semver_b.minor:
                    return "partial"  # Older minor version, newer features not available
                else:
                    return "full"
            elif semver_a.major == semver_b.major + 1:
                # New major version, check if old version is still supported
                if version_a.new_features:
                    # Check if new features can be disabled
                    return "partial"
                else:
                    return "partial"
            else:
                return "breaking"
            
        except Exception as e:
            logger.warning(f"Version compatibility check failed: {str(e)}")
            return "partial"
    
    def check_compatibility(self, model_a: str, version_a: str, 
                          model_b: str, version_b: str) -> CompatibilityCheck:
        """Check compatibility between two model versions"""
        try:
            # Get version objects
            version_obj_a = self._get_version_object(model_a, version_a)
            version_obj_b = self._get_version_object(model_b, version_b)
            
            if not version_obj_a or not version_obj_b:
                return CompatibilityCheck(
                    model_a=model_a, version_a=version_a,
                    model_b=model_b, version_b=version_b,
                    is_compatible=False,
                    compatibility_level="unknown",
                    issues=["Version objects not found"],
                    recommendations=["Verify versions exist"]
                )
            
            # Check model name compatibility
            same_model = model_a == model_b
            
            if same_model:
                # Same model - check internal compatibility
                if model_a in self.compatibility_matrices:
                    matrix = self.compatibility_matrices[model_a]
                    if version_a in matrix.compatibility_matrix and version_b in matrix.compatibility_matrix[version_a]:
                        compatibility_level = matrix.compatibility_matrix[version_a][version_b]
                    else:
                        compatibility_level = "partial"
                else:
                    compatibility_level = "partial"
                
                issues, recommendations = self._analyze_internal_compatibility(
                    version_obj_a, version_obj_b, compatibility_level
                )
            else:
                # Different models - check cross-model compatibility
                compatibility_level, issues, recommendations = self._analyze_cross_model_compatibility(
                    version_obj_a, version_obj_b
                )
            
            is_compatible = compatibility_level in ["full", "partial"]
            
            return CompatibilityCheck(
                model_a=model_a, version_a=version_a,
                model_b=model_b, version_b=version_b,
                is_compatible=is_compatible,
                compatibility_level=compatibility_level,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Compatibility check failed: {str(e)}")
            return CompatibilityCheck(
                model_a=model_a, version_a=version_a,
                model_b=model_b, version_b=version_b,
                is_compatible=False,
                compatibility_level="unknown",
                issues=[str(e)],
                recommendations=["Check error logs"]
            )
    
    def _get_version_object(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get version object by model name and version"""
        versions = self.version_registry.get(model_name, [])
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def _analyze_internal_compatibility(self, version_a: ModelVersion, 
                                      version_b: ModelVersion, 
                                      compatibility_level: str) -> Tuple[List[str], List[str]]:
        """Analyze internal model compatibility"""
        issues = []
        recommendations = []
        
        if compatibility_level == "breaking":
            issues.extend([
                f"Breaking changes detected between {version_a.version} and {version_b.version}",
                "API compatibility not guaranteed"
            ])
            recommendations.extend([
                "Migrate to latest version incrementally",
                "Test all existing integrations",
                "Update documentation and examples"
            ])
        elif compatibility_level == "partial":
            issues.append(f"Partial compatibility between {version_a.version} and {version_b.version}")
            recommendations.extend([
                "Some features may not be available",
                "Review feature matrix for available capabilities",
                "Consider gradual migration"
            ])
        
        # Check dependencies
        deps_a = version_a.dependencies or {}
        deps_b = version_b.dependencies or {}
        
        common_deps = set(deps_a.keys()) & set(deps_b.keys())
        for dep in common_deps:
            if deps_a[dep] != deps_b[dep]:
                issues.append(f"Dependency version mismatch for {dep}: {deps_a[dep]} vs {deps_b[dep]}")
                recommendations.append(f"Align dependency versions for {dep}")
        
        return issues, recommendations
    
    def _analyze_cross_model_compatibility(self, version_a: ModelVersion, 
                                         version_b: ModelVersion) -> Tuple[str, List[str], List[str]]:
        """Analyze cross-model compatibility"""
        issues = []
        recommendations = []
        
        # Check data format compatibility
        metadata_a = version_a.metadata or {}
        metadata_b = version_b.metadata or {}
        
        # In production, would check:
        # - Input/output schemas
        # - Data types
        # - Feature definitions
        # - API contracts
        
        # For demo, assume good compatibility
        compatibility_level = "full"
        
        recommendations.extend([
            "Verify data format compatibility",
            "Test model integration in staging",
            "Monitor cross-model performance"
        ])
        
        return compatibility_level, issues, recommendations
    
    def deprecate_version(self, model_name: str, version: str, 
                         deprecation_date: datetime, replacement_version: Optional[str] = None) -> bool:
        """Mark a model version as deprecated"""
        try:
            version_obj = self._get_version_object(model_name, version)
            if not version_obj:
                logger.error(f"Version {version} not found for {model_name}")
                return False
            
            # Update version metadata
            version_obj.metadata = version_obj.metadata or {}
            version_obj.metadata.update({
                "status": "deprecated",
                "deprecated_at": deprecation_date.isoformat(),
                "replacement_version": replacement_version,
                "deprecation_reason": "Automatically deprecated by versioning policy"
            })
            
            # Save updated version
            self._save_version_to_storage(version_obj)
            
            logger.info(f"Deprecated version {model_name} v{version} (replacement: {replacement_version})")
            
            return True
            
        except Exception as e:
            logger.error(f"Version deprecation failed: {str(e)}")
            return False
    
    def get_compatible_versions(self, model_name: str, version: str, 
                              compatibility_level: str = "partial") -> List[str]:
        """Get versions compatible with a given version"""
        try:
            if model_name not in self.compatibility_matrices:
                return []
            
            matrix = self.compatibility_matrices[model_name]
            if version not in matrix.compatibility_matrix:
                return []
            
            compatible_versions = []
            for compat_version, level in matrix.compatibility_matrix[version].items():
                if compat_version != version and level in [compatibility_level, "full"]:
                    compatible_versions.append(compat_version)
            
            return compatible_versions
            
        except Exception as e:
            logger.error(f"Compatible version lookup failed: {str(e)}")
            return []
    
    def create_version_branch(self, model_name: str, base_version: str, 
                            branch_name: str, branch_version: str) -> ModelVersion:
        """Create a branch version from a base version"""
        try:
            base_version_obj = self._get_version_object(model_name, base_version)
            if not base_version_obj:
                raise ValueError(f"Base version {base_version} not found")
            
            # Create branch version
            branch_version_obj = ModelVersion(
                name=model_name,
                version=branch_version,
                created_at=datetime.utcnow(),
                created_by="system",
                description=f"Branch '{branch_name}' from {base_version}",
                parent_version=base_version,
                breaking_changes=[],
                new_features=[f"Branch: {branch_name}"],
                bug_fixes=[],
                dependencies=base_version_obj.dependencies.copy(),
                compatibility_matrix={},
                metadata={
                    "branch_name": branch_name,
                    "branch_type": "feature_branch",
                    "base_version": base_version,
                    "status": "development"
                }
            )
            
            # Store branch version
            if model_name not in self.version_registry:
                self.version_registry[model_name] = []
            
            self.version_registry[model_name].append(branch_version_obj)
            
            # Update compatibility matrix
            self._update_compatibility_matrix(model_name)
            
            # Save to storage
            self._save_version_to_storage(branch_version_obj)
            
            logger.info(f"Created branch version: {model_name} v{branch_version} (from {base_version})")
            
            return branch_version_obj
            
        except Exception as e:
            logger.error(f"Branch creation failed: {str(e)}")
            raise
    
    def _save_version_to_storage(self, version: ModelVersion):
        """Save version to persistent storage"""
        try:
            version_dir = os.path.join(self.version_storage_path, version.name)
            os.makedirs(version_dir, exist_ok=True)
            
            version_file = os.path.join(version_dir, f"v{version.version}.json")
            
            with open(version_file, 'w') as f:
                version_dict = asdict(version)
                version_dict['created_at'] = version.created_at.isoformat()
                if version.parent_version:
                    version_dict['parent_version'] = version.parent_version
                json.dump(version_dict, f, indent=2)
            
            logger.debug(f"Saved version {version.name} v{version.version} to storage")
            
        except Exception as e:
            logger.error(f"Version storage failed: {str(e)}")
    
    def load_version_from_storage(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Load version from persistent storage"""
        try:
            version_file = os.path.join(self.version_storage_path, model_name, f"v{version}.json")
            
            if not os.path.exists(version_file):
                return None
            
            with open(version_file, 'r') as f:
                version_dict = json.load(f)
            
            # Convert back to ModelVersion
            version_dict['created_at'] = datetime.fromisoformat(version_dict['created_at'])
            
            version = ModelVersion(**version_dict)
            
            # Store in registry
            if model_name not in self.version_registry:
                self.version_registry[model_name] = []
            
            # Avoid duplicates
            existing_versions = [v.version for v in self.version_registry[model_name]]
            if version.version not in existing_versions:
                self.version_registry[model_name].append(version)
            
            return version
            
        except Exception as e:
            logger.error(f"Version loading failed: {str(e)}")
            return None
    
    def get_version_history(self, model_name: str) -> List[ModelVersion]:
        """Get complete version history for a model"""
        versions = self.version_registry.get(model_name, [])
        
        # Sort by semantic version
        versions.sort(key=lambda v: semantic_version.Version(v.version), reverse=True)
        
        return versions
    
    def compare_versions(self, model_name: str, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare two versions and highlight differences"""
        try:
            version_obj_a = self._get_version_object(model_name, version_a)
            version_obj_b = self._get_version_object(model_name, version_b)
            
            if not version_obj_a or not version_obj_b:
                return {"error": "Version objects not found"}
            
            comparison = {
                "model_name": model_name,
                "version_a": version_a,
                "version_b": version_b,
                "differences": {
                    "new_features": list(set(version_obj_b.new_features) - set(version_obj_a.new_features)),
                    "removed_features": list(set(version_obj_a.new_features) - set(version_obj_b.new_features)),
                    "new_breaking_changes": list(set(version_obj_b.breaking_changes) - set(version_obj_a.breaking_changes)),
                    "new_bug_fixes": list(set(version_obj_b.bug_fixes) - set(version_obj_a.bug_fixes))
                },
                "dependencies": {
                    "new_dependencies": [dep for dep in (version_obj_b.dependencies or {}) if dep not in (version_obj_a.dependencies or {})],
                    "removed_dependencies": [dep for dep in (version_obj_a.dependencies or {}) if dep not in (version_obj_b.dependencies or {})],
                    "version_changes": [f"{dep}: {version_obj_a.dependencies[dep]} -> {version_obj_b.dependencies[dep]}" 
                                      for dep in set(version_obj_a.dependencies or {}) & set(version_obj_b.dependencies or {})
                                      if version_obj_a.dependencies[dep] != version_obj_b.dependencies[dep]]
                },
                "migration_recommendations": self._generate_migration_recommendations(version_obj_a, version_obj_b)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Version comparison failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_migration_recommendations(self, version_a: ModelVersion, 
                                          version_b: ModelVersion) -> List[str]:
        """Generate migration recommendations between versions"""
        recommendations = []
        
        # Check for breaking changes
        if version_b.breaking_changes:
            recommendations.append("⚠️  Breaking changes detected - review API documentation")
            recommendations.append("Test all integrations in staging environment")
        
        # Check for new features
        if version_b.new_features:
            recommendations.append(f"New features available: {', '.join(version_b.new_features)}")
        
        # Check dependencies
        deps_a = version_a.dependencies or {}
        deps_b = version_b.dependencies or {}
        
        if deps_a != deps_b:
            recommendations.append("Dependencies have changed - update requirements")
        
        # Version type recommendations
        version_type = version_b.metadata.get("version_type", "patch") if version_b.metadata else "patch"
        
        if version_type == "major":
            recommendations.append("Major version update - expect significant changes")
        elif version_type == "minor":
            recommendations.append("Minor version update - new features, backward compatible")
        else:
            recommendations.append("Patch update - bug fixes, fully backward compatible")
        
        return recommendations
    
    def export_version_graph(self, model_name: str) -> Dict[str, Any]:
        """Export version dependency graph"""
        try:
            versions = self.version_registry.get(model_name, [])
            
            graph = {
                "model_name": model_name,
                "nodes": [],
                "edges": []
            }
            
            for version in versions:
                node = {
                    "id": version.version,
                    "label": f"v{version.version}",
                    "version": version.version,
                    "created_at": version.created_at.isoformat(),
                    "status": version.metadata.get("status", "active") if version.metadata else "active",
                    "has_breaking_changes": bool(version.breaking_changes),
                    "is_deprecated": version.metadata.get("status") == "deprecated" if version.metadata else False
                }
                graph["nodes"].append(node)
                
                # Add edge to parent
                if version.parent_version:
                    edge = {
                        "from": version.parent_version,
                        "to": version.version,
                        "type": "parent_child"
                    }
                    graph["edges"].append(edge)
            
            return graph
            
        except Exception as e:
            logger.error(f"Version graph export failed: {str(e)}")
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Create initial version
    v1 = version_manager.create_model_version(
        model_name="medical-diagnosis",
        version_type="major",
        description="Initial medical diagnosis model",
        changes={
            "new_features": ["diagnosis prediction", "confidence scoring"],
            "breaking_changes": ["new API format"],
            "bug_fixes": []
        }
    )
    
    print(f"Created version: {v1.name} v{v1.version}")
    
    # Create patch version
    v1_1 = version_manager.create_model_version(
        model_name="medical-diagnosis",
        version_type="patch",
        description="Bug fixes and performance improvements",
        parent_version=v1.version,
        changes={
            "bug_fixes": ["fixed edge case in confidence calculation"],
            "performance_improvements": ["optimized inference speed"]
        }
    )
    
    print(f"Created version: {v1_1.name} v{v1_1.version}")
    
    # Check compatibility
    compat_check = version_manager.check_compatibility(
        "medical-diagnosis", v1.version,
        "medical-diagnosis", v1_1.version
    )
    
    print(f"Compatibility: {compat_check.compatibility_level}")
    print(f"Compatible: {compat_check.is_compatible}")
    
    # Get version history
    history = version_manager.get_version_history("medical-diagnosis")
    print(f"Version history: {[v.version for v in history]}")
    
    # Compare versions
    comparison = version_manager.compare_versions("medical-diagnosis", v1.version, v1_1.version)
    print(f"Comparison: {comparison['migration_recommendations']}")