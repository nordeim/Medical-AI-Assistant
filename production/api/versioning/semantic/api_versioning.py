# API Versioning System with Semantic Versioning and Compatibility Management
# Production-grade API versioning with backward compatibility support

import re
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from packaging import version
import aiofiles
import asyncio
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionType(Enum):
    """Types of version increments"""
    MAJOR = "major"      # Breaking changes
    MINOR = "minor"      # Backward compatible features
    PATCH = "patch"      # Bug fixes, no API changes

class CompatibilityLevel(Enum):
    """API compatibility levels"""
    FULLY_COMPATIBLE = "fully_compatible"
    BACKWARD_COMPATIBLE = "backward_compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"

class DeprecationStatus(Enum):
    """Deprecation lifecycle status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"
    REMOVED = "removed"

class ChangeType(Enum):
    """Types of API changes"""
    # Additive changes (backward compatible)
    NEW_ENDPOINT = "new_endpoint"
    NEW_PARAMETER = "new_parameter"
    NEW_PROPERTY = "new_property"
    NEW_RESPONSE_FIELD = "new_response_field"
    
    # Breaking changes
    REMOVED_ENDPOINT = "removed_endpoint"
    REMOVED_PARAMETER = "removed_parameter"
    REMOVED_PROPERTY = "removed_property"
    REMOVED_RESPONSE_FIELD = "removed_response_field"
    CHANGED_PARAMETER_TYPE = "changed_parameter_type"
    CHANGED_RESPONSE_FORMAT = "changed_response_format"
    CHANGED_AUTH_REQUIREMENTS = "changed_auth_requirements"
    
    # Semantic changes
    CHANGED_ENDPOINT_PATH = "changed_endpoint_path"
    CHANGED_PARAMETER_REQUIRED = "changed_parameter_required"
    CHANGED_PROPERTY_TYPE = "changed_property_type"
    CHANGED_ERROR_CODES = "changed_error_codes"

@dataclass
class APIVersion:
    """API version information"""
    version: str  # Semantic version (e.g., "1.2.3")
    release_date: datetime
    status: DeprecationStatus
    description: str
    changes: List[Dict[str, Any]]
    supported_until: Optional[datetime] = None
    sunset_date: Optional[datetime] = None
    migration_guide: Optional[str] = None
    compatibility_level: CompatibilityLevel = CompatibilityLevel.FULLY_COMPATIBLE
    
    def __post_init__(self):
        if not self.supported_until:
            self.supported_until = self.release_date.replace(year=self.release_date.year + 2)
        if not self.sunset_date:
            self.sunset_date = self.release_date.replace(year=self.release_date.year + 3)

@dataclass
class APIChange:
    """Individual API change"""
    change_type: ChangeType
    endpoint: Optional[str] = None
    parameter: Optional[str] = None
    property: Optional[str] = None
    description: str = ""
    impact: str = ""
    migration_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class SemanticVersionManager:
    """Semantic versioning manager for healthcare API"""
    
    def __init__(self, base_version: str = "1.0.0"):
        self.base_version = base_version
        self.versions: Dict[str, APIVersion] = {}
        self.compatibility_matrix: Dict[str, Dict[str, CompatibilityLevel]] = {}
        self._load_initial_versions()
    
    def _load_initial_versions(self):
        """Load initial version information"""
        # Version 1.0.0 - Initial release
        self.versions["1.0.0"] = APIVersion(
            version="1.0.0",
            release_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            status=DeprecationStatus.DEPRECATED,
            description="Initial API release with basic patient and observation management",
            changes=[
                {
                    "change_type": ChangeType.NEW_ENDPOINT.value,
                    "endpoint": "/api/v1/patients",
                    "description": "Patient management endpoints"
                },
                {
                    "change_type": ChangeType.NEW_ENDPOINT.value,
                    "endpoint": "/api/v1/observations", 
                    "description": "Observation data endpoints"
                }
            ]
        )
        
        # Version 2.0.0 - Major update with FHIR integration
        self.versions["2.0.0"] = APIVersion(
            version="2.0.0",
            release_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
            status=DeprecationStatus.DEPRECATED,
            description="Major release with FHIR R4 compliance and webhook support",
            changes=[
                {
                    "change_type": ChangeType.NEW_ENDPOINT.value,
                    "endpoint": "/fhir/{resourceType}",
                    "description": "FHIR-compliant resource endpoints"
                },
                {
                    "change_type": ChangeType.NEW_ENDPOINT.value,
                    "endpoint": "/webhooks",
                    "description": "Webhook management endpoints"
                },
                {
                    "change_type": ChangeType.CHANGED_AUTH_REQUIREMENTS.value,
                    "endpoint": "all",
                    "description": "Updated to OAuth 2.0 authentication",
                    "migration_required": True
                }
            ]
        )
        
        # Version 3.0.0 - Current production version
        self.versions["3.0.0"] = APIVersion(
            version="3.0.0",
            release_date=datetime(2024, 11, 4, tzinfo=timezone.utc),
            status=DeprecationStatus.ACTIVE,
            description="Current production API with enhanced analytics and real-time features",
            changes=[
                {
                    "change_type": ChangeType.NEW_ENDPOINT.value,
                    "endpoint": "/api/v1/analytics",
                    "description": "Advanced analytics endpoints"
                },
                {
                    "change_type": ChangeType.NEW_ENDPOINT.value,
                    "endpoint": "/api/v1/websocket",
                    "description": "Real-time WebSocket connections"
                },
                {
                    "change_type": ChangeType.NEW_PROPERTY.value,
                    "endpoint": "/api/v1/patients",
                    "property": "preferences",
                    "description": "Patient preferences and settings"
                }
            ]
        )
    
    def create_new_version(
        self,
        current_version: str,
        version_type: VersionType,
        changes: List[APIChange],
        description: str
    ) -> APIVersion:
        """Create new version based on semantic versioning rules"""
        
        current_ver = version.parse(current_version)
        
        if version_type == VersionType.PATCH:
            new_version = f"{current_ver.major}.{current_ver.minor}.{current_ver.micro + 1}"
        elif version_type == VersionType.MINOR:
            new_version = f"{current_ver.major}.{current_ver.minor + 1}.0"
        else:  # MAJOR
            new_version = f"{current_ver.major + 1}.0.0"
        
        # Determine compatibility level
        compatibility_level = self._calculate_compatibility_level(changes)
        
        # Create new version
        new_api_version = APIVersion(
            version=new_version,
            release_date=datetime.now(timezone.utc),
            status=DeprecationStatus.ACTIVE,
            description=description,
            changes=[change.to_dict() for change in changes],
            compatibility_level=compatibility_level
        )
        
        self.versions[new_version] = new_api_version
        self._update_compatibility_matrix(current_version, new_version, changes)
        
        logger.info(f"Created new API version: {new_version}")
        return new_api_version
    
    def _calculate_compatibility_level(self, changes: List[APIChange]) -> CompatibilityLevel:
        """Calculate compatibility level based on changes"""
        
        breaking_changes = [
            ChangeType.REMOVED_ENDPOINT,
            ChangeType.REMOVED_PARAMETER,
            ChangeType.REMOVED_PROPERTY,
            ChangeType.CHANGED_PARAMETER_TYPE,
            ChangeType.CHANGED_RESPONSE_FORMAT,
            ChangeType.CHANGED_AUTH_REQUIREMENTS,
            ChangeType.CHANGED_ENDPOINT_PATH,
            ChangeType.CHANGED_PROPERTY_TYPE
        ]
        
        has_breaking_changes = any(change.change_type in breaking_changes for change in changes)
        has_semantic_changes = any(
            change.change_type in [
                ChangeType.NEW_PARAMETER,
                ChangeType.NEW_PROPERTY,
                ChangeType.NEW_RESPONSE_FIELD,
                ChangeType.CHANGED_PARAMETER_REQUIRED
            ] 
            for change in changes
        )
        
        if has_breaking_changes:
            return CompatibilityLevel.INCOMPATIBLE
        elif has_semantic_changes:
            return CompatibilityLevel.PARTIALLY_COMPATIBLE
        else:
            return CompatibilityLevel.FULLY_COMPATIBLE
    
    def _update_compatibility_matrix(
        self,
        from_version: str,
        to_version: str,
        changes: List[APIChange]
    ):
        """Update compatibility matrix between versions"""
        
        compatibility = self._calculate_compatibility_level(changes)
        
        if from_version not in self.compatibility_matrix:
            self.compatibility_matrix[from_version] = {}
        
        self.compatibility_matrix[from_version][to_version] = compatibility
    
    def get_compatibility(
        self,
        from_version: str,
        to_version: str
    ) -> Optional[CompatibilityLevel]:
        """Get compatibility level between two versions"""
        
        matrix = self.compatibility_matrix.get(from_version, {})
        return matrix.get(to_version)
    
    def get_latest_version(self) -> Optional[APIVersion]:
        """Get the latest active version"""
        active_versions = [
            v for v in self.versions.values() 
            if v.status == DeprecationStatus.ACTIVE
        ]
        
        if not active_versions:
            return None
        
        return max(active_versions, key=lambda v: version.parse(v.version))
    
    def get_compatible_versions(
        self,
        version_str: str,
        max_level: CompatibilityLevel = CompatibilityLevel.BACKWARD_COMPATIBLE
    ) -> List[APIVersion]:
        """Get versions compatible with the given version"""
        
        compatible = []
        base_version = version.parse(version_str)
        
        for api_version in self.versions.values():
            if api_version.status == DeprecationStatus.REMOVED:
                continue
                
            ver = version.parse(api_version.version)
            
            # Check if versions are compatible
            compatibility = self.get_compatibility(version_str, api_version.version)
            
            if compatibility and self._is_compatible_level(compatibility, max_level):
                compatible.append(api_version)
        
        return sorted(compatible, key=lambda v: version.parse(v.version), reverse=True)
    
    def _is_compatible_level(
        self,
        level: CompatibilityLevel,
        max_level: CompatibilityLevel
    ) -> bool:
        """Check if compatibility level is within acceptable range"""
        
        level_order = {
            CompatibilityLevel.FULLY_COMPATIBLE: 0,
            CompatibilityLevel.BACKWARD_COMPATIBLE: 1,
            CompatibilityLevel.PARTIALLY_COMPATIBLE: 2,
            CompatibilityLevel.INCOMPATIBLE: 3
        }
        
        return level_order[level] <= level_order[max_level]

class VersionRouter:
    """Routes requests to appropriate API version"""
    
    def __init__(self, version_manager: SemanticVersionManager):
        self.version_manager = version_manager
        self.version_handlers: Dict[str, Any] = {}
        self.default_version = "3.0.0"
    
    def register_version_handler(self, version: str, handler: Any):
        """Register handler for specific API version"""
        self.version_handlers[version] = handler
    
    async def route_request(
        self,
        request_path: str,
        request_method: str,
        version_hint: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Route request to appropriate version handler"""
        
        # Determine target version
        target_version = self._determine_target_version(version_hint, headers)
        
        if target_version not in self.version_handlers:
            # Try compatible version or default
            compatible_versions = self.version_manager.get_compatible_versions(target_version)
            if compatible_versions:
                target_version = compatible_versions[0].version
            else:
                target_version = self.default_version
        
        # Route to handler
        handler = self.version_handlers.get(target_version)
        
        if not handler:
            raise ValueError(f"No handler available for version {target_version}")
        
        # Execute request with version context
        return await self._execute_with_version_context(
            handler, request_path, request_method, target_version
        )
    
    def _determine_target_version(
        self,
        version_hint: Optional[str],
        headers: Optional[Dict[str, str]]
    ) -> str:
        """Determine target API version from hints"""
        
        # Priority order:
        # 1. Explicit version hint
        # 2. Accept header with version
        # 3. API version header
        # 4. Default version
        
        if version_hint and version_hint in self.version_manager.versions:
            return version_hint
        
        if headers:
            accept_header = headers.get("accept", "")
            api_version_header = headers.get("x-api-version", "")
            
            # Check Accept header for version
            if "application/vnd.healthcare.v" in accept_header:
                match = re.search(r"v(\d+\.\d+\.\d+)", accept_header)
                if match:
                    version_str = match.group(1)
                    if version_str in self.version_manager.versions:
                        return version_str
            
            # Check API version header
            if api_version_header in self.version_manager.versions:
                return api_version_header
        
        return self.default_version
    
    async def _execute_with_version_context(
        self,
        handler: Any,
        path: str,
        method: str,
        version: str
    ) -> Dict[str, Any]:
        """Execute request with version context"""
        
        context = {
            "version": version,
            "version_info": self.version_manager.versions.get(version),
            "compatibility_warnings": self._get_compatibility_warnings(version)
        }
        
        # Execute handler (implementation depends on handler type)
        try:
            result = await handler(path, method, context)
            result["version_context"] = context
            return result
        except Exception as e:
            return {
                "error": str(e),
                "version_context": context,
                "status": 500
            }
    
    def _get_compatibility_warnings(self, version: str) -> List[str]:
        """Get compatibility warnings for the version"""
        warnings = []
        
        api_version = self.version_manager.versions.get(version)
        if not api_version:
            return warnings
        
        # Check if version is deprecated
        if api_version.status == DeprecationStatus.DEPRECATED:
            warnings.append(f"API version {version} is deprecated")
        
        if api_version.status == DeprecationStatus.SUNSET:
            warnings.append(f"API version {version} will be removed soon")
        
        # Check compatibility with latest version
        latest = self.version_manager.get_latest_version()
        if latest and latest.version != version:
            compatibility = self.version_manager.get_compatibility(version, latest.version)
            if compatibility == CompatibilityLevel.INCOMPATIBLE:
                warnings.append(f"Version {version} is not compatible with latest version")
        
        return warnings

class MigrationGuide:
    """API migration guide generator"""
    
    def __init__(self, version_manager: SemanticVersionManager):
        self.version_manager = version_manager
    
    def generate_migration_guide(
        self,
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """Generate migration guide between versions"""
        
        from_ver = self.version_manager.versions.get(from_version)
        to_ver = self.version_manager.versions.get(to_version)
        
        if not from_ver or not to_ver:
            raise ValueError("One or both versions not found")
        
        # Find migration path
        migration_path = self._find_migration_path(from_version, to_version)
        
        guide = {
            "from_version": from_version,
            "to_version": to_version,
            "migration_path": migration_path,
            "breaking_changes": [],
            "deprecations": [],
            "new_features": [],
            "code_examples": {},
            "timeline": self._generate_migration_timeline(migration_path)
        }
        
        # Analyze changes across the migration path
        for version_pair in migration_path:
            from_ver, to_ver = version_pair
            changes = self._get_changes_between_versions(from_ver, to_ver)
            
            # Categorize changes
            guide["breaking_changes"].extend(self._categorize_changes(changes, "breaking"))
            guide["deprecations"].extend(self._categorize_changes(changes, "deprecation"))
            guide["new_features"].extend(self._categorize_changes(changes, "addition"))
        
        # Generate code examples
        guide["code_examples"] = self._generate_code_examples(guide["breaking_changes"])
        
        return guide
    
    def _find_migration_path(self, from_version: str, to_version: str) -> List[Tuple[str, str]]:
        """Find migration path between versions"""
        
        # Simplified path finding - in production would use more sophisticated algorithms
        path = []
        current = from_version
        
        while current != to_version:
            current_ver = version.parse(current)
            target_ver = version.parse(to_version)
            
            if current_ver < target_ver:
                # Move forward
                next_version = f"{current_ver.major}.{current_ver.minor + 1}.0"
                path.append((current, next_version))
                current = next_version
            else:
                # Move backward  
                path.append((current, f"{current_ver.major}.{current_ver.minor - 1}.0"))
                current = path[-1][1]
        
        return path
    
    def _get_changes_between_versions(
        self,
        from_version: str,
        to_version: str
    ) -> List[APIChange]:
        """Get all changes between two versions"""
        
        changes = []
        
        from_ver = self.version_manager.versions.get(from_version)
        to_ver = self.version_manager.versions.get(to_version)
        
        if from_ver and to_ver:
            # Collect changes from both versions
            for change_data in from_ver.changes + to_ver.changes:
                change = APIChange(
                    change_type=ChangeType(change_data.get("change_type")),
                    endpoint=change_data.get("endpoint"),
                    parameter=change_data.get("parameter"),
                    property=change_data.get("property"),
                    description=change_data.get("description", ""),
                    impact=change_data.get("impact", ""),
                    migration_required=change_data.get("migration_required", False)
                )
                changes.append(change)
        
        return changes
    
    def _categorize_changes(
        self,
        changes: List[APIChange],
        category: str
    ) -> List[Dict[str, Any]]:
        """Categorize changes by type"""
        
        breaking_types = [
            ChangeType.REMOVED_ENDPOINT,
            ChangeType.REMOVED_PARAMETER,
            ChangeType.REMOVED_PROPERTY,
            ChangeType.CHANGED_PARAMETER_TYPE,
            ChangeType.CHANGED_RESPONSE_FORMAT,
            ChangeType.CHANGED_AUTH_REQUIREMENTS,
            ChangeType.CHANGED_ENDPOINT_PATH,
            ChangeType.CHANGED_PROPERTY_TYPE
        ]
        
        deprecation_types = [
            ChangeType.REMOVED_ENDPOINT,
            ChangeType.REMOVED_PARAMETER,
            ChangeType.REMOVED_PROPERTY
        ]
        
        addition_types = [
            ChangeType.NEW_ENDPOINT,
            ChangeType.NEW_PARAMETER,
            ChangeType.NEW_PROPERTY,
            ChangeType.NEW_RESPONSE_FIELD
        ]
        
        if category == "breaking":
            return [change.to_dict() for change in changes if change.change_type in breaking_types]
        elif category == "deprecation":
            return [change.to_dict() for change in changes if change.change_type in deprecation_types]
        elif category == "addition":
            return [change.to_dict() for change in changes if change.change_type in addition_types]
        
        return []
    
    def _generate_migration_timeline(self, migration_path: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """Generate migration timeline"""
        timeline = []
        
        for from_ver, to_ver in migration_path:
            timeline.append({
                "step": len(timeline) + 1,
                "from_version": from_ver,
                "to_version": to_ver,
                "estimated_effort": self._estimate_migration_effort(from_ver, to_ver),
                "test_phase": "Compatibility testing required"
            })
        
        return timeline
    
    def _estimate_migration_effort(self, from_ver: str, to_ver: str) -> str:
        """Estimate migration effort level"""
        
        from_version_parsed = version.parse(from_ver)
        to_version_parsed = version.parse(to_ver)
        
        if from_version_parsed.major != to_version_parsed.major:
            return "High - Major version upgrade"
        elif from_version_parsed.minor != to_version_parsed.minor:
            return "Medium - Minor version upgrade"
        else:
            return "Low - Patch update"
    
    def _generate_code_examples(self, breaking_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate code examples for migration"""
        
        examples = {}
        
        # Generate authentication update examples
        auth_changes = [c for c in breaking_changes if "auth" in c.get("description", "").lower()]
        if auth_changes:
            examples["authentication"] = {
                "old": {
                    "curl": "curl -H 'X-API-Key: old-api-key' https://api.healthcare.org/api/v1/patients",
                    "python": "headers = {'X-API-Key': 'old-api-key'}"
                },
                "new": {
                    "curl": "curl -H 'Authorization: Bearer new-oauth-token' https://api.healthcare.org/api/v1/patients",
                    "python": """headers = {'Authorization': 'Bearer new-oauth-token'}
# Get token first
token_response = requests.post('https://api.healthcare.org/oauth/token', data={
    'grant_type': 'client_credentials',
    'client_id': 'your-client-id',
    'client_secret': 'your-client-secret'
})"""
                }
            }
        
        return examples

class VersionCompatibilityChecker:
    """Checks API compatibility between versions"""
    
    def __init__(self, version_manager: SemanticVersionManager):
        self.version_manager = version_manager
    
    def check_client_compatibility(
        self,
        client_version: str,
        server_versions: List[str]
    ) -> Dict[str, Any]:
        """Check if client is compatible with server versions"""
        
        compatibility_results = []
        
        for server_version in server_versions:
            compatibility = self.version_manager.get_compatibility(client_version, server_version)
            
            if compatibility:
                compatibility_results.append({
                    "client_version": client_version,
                    "server_version": server_version,
                    "compatibility_level": compatibility.value,
                    "compatible": compatibility in [
                        CompatibilityLevel.FULLY_COMPATIBLE,
                        CompatibilityLevel.BACKWARD_COMPATIBLE
                    ]
                })
        
        return {
            "client_version": client_version,
            "server_versions": compatibility_results,
            "overall_compatible": all(result["compatible"] for result in compatibility_results)
        }

# Example usage
if __name__ == "__main__":
    # Initialize versioning system
    version_manager = SemanticVersionManager("3.0.0")
    
    # Create migration guide
    migration_guide = MigrationGuide(version_manager)
    guide = migration_guide.generate_migration_guide("1.0.0", "3.0.0")
    
    print(f"Migration guide from {guide['from_version']} to {guide['to_version']}:")
    print(f"Breaking changes: {len(guide['breaking_changes'])}")
    print(f"New features: {len(guide['new_features'])}")
    print(f"Migration steps: {len(guide['migration_path'])}")
    
    # Get latest version
    latest = version_manager.get_latest_version()
    print(f"Latest API version: {latest.version}")
    
    # Check compatibility
    checker = VersionCompatibilityChecker(version_manager)
    compatibility = checker.check_client_compatibility("2.0.0", ["3.0.0"])
    print(f"Client compatibility: {compatibility['overall_compatible']}")