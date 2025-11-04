"""
Versioning Utilities

Provides comprehensive versioning functionality including:
- Semantic versioning utilities
- Git integration
- Model comparison tools
- Change tracking
"""

import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime
import logging


class SemanticVersion:
    """
    Semantic Versioning (SemVer) implementation
    
    Format: MAJOR.MINOR.PATCH-PRERELEASE+BUILD
    Example: 1.2.3-alpha.1+build.456
    """
    
    def __init__(self, version: str):
        """Initialize semantic version"""
        self.original = version
        self._parse_version(version)
    
    def _parse_version(self, version: str):
        """Parse semantic version string"""
        pattern = r'^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        match = re.match(pattern, version)
        
        if not match:
            raise ValueError(f"Invalid semantic version: {version}")
        
        groups = match.groupdict()
        self.major = int(groups['major'])
        self.minor = int(groups['minor'])
        self.patch = int(groups['patch'])
        self.prerelease = groups['prerelease']
        self.build = groups['build']
    
    def __str__(self) -> str:
        """String representation"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        
        if self.prerelease:
            version += f"-{self.prerelease}"
        
        if self.build:
            version += f"+{self.build}"
        
        return version
    
    def __lt__(self, other) -> bool:
        """Less than comparison"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        # Compare major.minor.patch
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
        # No prerelease vs prerelease
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and not other.prerelease:
            return True
        
        # Both have prerelease
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        
        # Both stable releases
        return False
    
    def __le__(self, other) -> bool:
        """Less than or equal comparison"""
        return self < other or self == other
    
    def __gt__(self, other) -> bool:
        """Greater than comparison"""
        return not self <= other
    
    def __ge__(self, other) -> bool:
        """Greater than or equal comparison"""
        return not self < other
    
    def __eq__(self, other) -> bool:
        """Equality comparison"""
        if not isinstance(other, SemanticVersion):
            return NotImplemented
        
        return (self.major == other.major and
                self.minor == other.minor and
                self.patch == other.patch and
                self.prerelease == other.prerelease and
                self.build == other.build)
    
    def __ne__(self, other) -> bool:
        """Inequality comparison"""
        return not self == other
    
    def bump_major(self) -> 'SemanticVersion':
        """Bump major version"""
        return SemanticVersion(f"{self.major + 1}.0.0")
    
    def bump_minor(self) -> 'SemanticVersion':
        """Bump minor version"""
        return SemanticVersion(f"{self.major}.{self.minor + 1}.0")
    
    def bump_patch(self) -> 'SemanticVersion':
        """Bump patch version"""
        return SemanticVersion(f"{self.major}.{self.minor}.{self.patch + 1}")
    
    def add_prerelease(self, prerelease: str) -> 'SemanticVersion':
        """Add prerelease identifier"""
        return SemanticVersion(f"{self.major}.{self.minor}.{self.patch}-{prerelease}")
    
    def add_build(self, build: str) -> 'SemanticVersion':
        """Add build metadata"""
        return SemanticVersion(f"{self.major}.{self.minor}.{self.patch}+{build}")


class GitIntegration:
    """Git integration for version tracking"""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize git integration"""
        self.repo_path = Path(repo_path or '.')
        self.logger = logging.getLogger(f"GitIntegration.{id(self)}")
    
    def get_current_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Git not available or not in a git repository")
            return None
    
    def get_current_branch(self) -> Optional[str]:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("Git not available")
            return None
    
    def get_latest_tag(self) -> Optional[str]:
        """Get latest git tag"""
        try:
            result = subprocess.run(
                ['git', 'describe', '--tags', '--abbrev=0'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def get_commit_message(self) -> Optional[str]:
        """Get current commit message"""
        try:
            result = subprocess.run(
                ['git', 'log', '-1', '--pretty=%B'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def create_tag(self, tag: str, message: Optional[str] = None) -> bool:
        """Create a git tag"""
        try:
            cmd = ['git', 'tag', '-a', tag]
            if message:
                cmd.extend(['-m', message])
            
            subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error(f"Failed to create git tag: {tag}")
            return False
    
    def push_tags(self) -> bool:
        """Push tags to remote"""
        try:
            subprocess.run(
                ['git', 'push', '--tags'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Failed to push tags")
            return False
    
    def get_file_hash(self, file_path: str) -> Optional[str]:
        """Get git hash of a file"""
        try:
            result = subprocess.run(
                ['git', 'hash-object', str(Path(self.repo_path) / file_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def get_changes_since_tag(self, tag: str) -> List[str]:
        """Get files changed since a specific tag"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', f'{tag}..HEAD'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip().split('\n') if result.stdout.strip() else []
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []


class VersionTracker:
    """Version tracking and management"""
    
    def __init__(self, repo_path: Optional[str] = None):
        """Initialize version tracker"""
        self.git = GitIntegration(repo_path)
        self.logger = logging.getLogger(f"VersionTracker.{id(self)}")
    
    def parse_version(self, version: str) -> Optional[SemanticVersion]:
        """Parse version string"""
        try:
            return SemanticVersion(version)
        except ValueError as e:
            self.logger.warning(f"Failed to parse version {version}: {e}")
            return None
    
    def format_version(
        self,
        major: int = 0,
        minor: int = 0,
        patch: int = 0,
        prerelease: Optional[str] = None,
        build: Optional[str] = None
    ) -> str:
        """Format semantic version"""
        version = f"{major}.{minor}.{patch}"
        
        if prerelease:
            version += f"-{prerelease}"
        
        if build:
            version += f"+{build}"
        
        return version
    
    def increment_version(
        self,
        current_version: Optional[str],
        increment_type: str = 'patch'
    ) -> str:
        """Increment version based on type"""
        if not current_version:
            # Default to 0.1.0 for new projects
            return "0.1.0"
        
        version = self.parse_version(current_version)
        if not version:
            return current_version
        
        if increment_type == 'major':
            return str(version.bump_major())
        elif increment_type == 'minor':
            return str(version.bump_minor())
        elif increment_type == 'patch':
            return str(version.bump_patch())
        else:
            self.logger.warning(f"Unknown increment type: {increment_type}")
            return current_version
    
    def generate_version_from_git(
        self,
        increment_type: str = 'patch',
        include_commit: bool = True
    ) -> str:
        """Generate version based on git state"""
        # Get current version from latest tag
        latest_tag = self.git.get_latest_tag()
        current_version = latest_tag if latest_tag else None
        
        # Increment version
        new_version = self.increment_version(current_version, increment_type)
        
        # Add prerelease if on non-main branch
        branch = self.git.get_current_branch()
        if branch and branch not in ['main', 'master', 'develop']:
            new_version = f"{new_version.split('+')[0]}-{branch.replace('/', '.')}"
        
        # Add commit hash as build metadata
        if include_commit:
            commit = self.git.get_current_commit()
            if commit:
                # Use short hash
                short_commit = commit[:8]
                if '+' in new_version:
                    new_version += f".{short_commit}"
                else:
                    new_version += f"+{short_commit}"
        
        return new_version
    
    def create_version_tag(
        self,
        version: str,
        push: bool = False,
        message: Optional[str] = None
    ) -> bool:
        """Create git tag for version"""
        success = self.git.create_tag(version, message)
        
        if success and push:
            self.git.push_tags()
        
        return success
    
    def get_version_from_git(
        self,
        include_build: bool = True,
        fallback_version: str = "0.0.0"
    ) -> str:
        """Get version information from git"""
        commit = self.git.get_current_commit()
        branch = self.git.get_current_branch()
        
        if not commit:
            return fallback_version
        
        version_info = {
            'commit': commit[:8],
            'branch': branch or 'unknown'
        }
        
        if include_build:
            return f"{fallback_version}+{commit[:8]}.{version_info['branch'].replace('/', '.')}"
        else:
            return fallback_version
    
    def compare_versions(self, version_a: str, version_b: str) -> int:
        """
        Compare two versions
        
        Returns:
            -1 if version_a < version_b
             0 if version_a == version_b
             1 if version_a > version_b
        """
        sem_a = self.parse_version(version_a)
        sem_b = self.parse_version(version_b)
        
        if not sem_a or not sem_b:
            return 0
        
        if sem_a < sem_b:
            return -1
        elif sem_a > sem_b:
            return 1
        else:
            return 0
    
    def sort_versions(self, versions: List[str], reverse: bool = False) -> List[str]:
        """Sort versions semantically"""
        def sort_key(version_str):
            sem = self.parse_version(version_str)
            return (sem.major, sem.minor, sem.patch, sem.prerelease or ''.zfill(10))
        
        return sorted(versions, key=sort_key, reverse=reverse)
    
    def is_newer_version(self, candidate: str, reference: str) -> bool:
        """Check if candidate version is newer than reference"""
        return self.compare_versions(candidate, reference) > 0
    
    def get_version_from_config(
        self,
        config_path: str,
        config_key: str = "version"
    ) -> Optional[str]:
        """Get version from configuration file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            return None
        
        try:
            if config_file.suffix == '.json':
                with open(config_file) as f:
                    config = json.load(f)
                    return config.get(config_key)
            elif config_file.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                    return config.get(config_key)
        except Exception as e:
            self.logger.warning(f"Failed to read config file {config_path}: {e}")
        
        return None
    
    def update_version_in_config(
        self,
        config_path: str,
        new_version: str,
        config_key: str = "version"
    ) -> bool:
        """Update version in configuration file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            return False
        
        try:
            if config_file.suffix == '.json':
                with open(config_file) as f:
                    config = json.load(f)
                
                config[config_key] = new_version
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                return True
            elif config_file.suffix in ['.yaml', '.yml']:
                import yaml
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                
                config[config_key] = new_version
                
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                    
                return True
        except Exception as e:
            self.logger.error(f"Failed to update config file {config_path}: {e}")
        
        return False


class ModelComparator:
    """Compare model versions and changes"""
    
    def __init__(self):
        """Initialize model comparator"""
        self.logger = logging.getLogger(f"ModelComparator.{id(self)}")
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """Compare two model versions"""
        sem_a = SemanticVersion(version_a)
        sem_b = SemanticVersion(version_b)
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'relationship': self._get_version_relationship(sem_a, sem_b),
            'changes': self._get_version_changes(sem_a, sem_b)
        }
        
        return comparison
    
    def _get_version_relationship(
        self,
        sem_a: SemanticVersion,
        sem_b: SemanticVersion
    ) -> str:
        """Determine version relationship"""
        if sem_a == sem_b:
            return 'identical'
        elif sem_a < sem_b:
            return 'older'
        else:
            return 'newer'
    
    def _get_version_changes(
        self,
        sem_a: SemanticVersion,
        sem_b: SemanticVersion
    ) -> Dict[str, Any]:
        """Get detailed version changes"""
        changes = {
            'major_changed': sem_a.major != sem_b.major,
            'minor_changed': sem_a.minor != sem_b.minor,
            'patch_changed': sem_a.patch != sem_b.patch,
            'prerelease_changed': sem_a.prerelease != sem_b.prerelease,
            'build_changed': sem_a.build != sem_b.build
        }
        
        # Determine change type
        if changes['major_changed']:
            changes['change_type'] = 'major'
        elif changes['minor_changed']:
            changes['change_type'] = 'minor'
        elif changes['patch_changed']:
            changes['change_type'] = 'patch'
        else:
            changes['change_type'] = 'metadata'
        
        return changes
    
    def is_compatible_change(
        self,
        from_version: str,
        to_version: str,
        compatibility_type: str = 'backward'
    ) -> bool:
        """
        Check if version change is compatible
        
        Args:
            from_version: Source version
            to_version: Target version
            compatibility_type: 'backward', 'forward', 'both'
        """
        comparison = self.compare_versions(from_version, to_version)
        changes = comparison['changes']
        
        if compatibility_type == 'backward':
            # New version is compatible with old version
            return not changes['major_changed']
        elif compatibility_type == 'forward':
            # Old version is compatible with new version
            return not changes['major_changed']
        elif compatibility_type == 'both':
            # Fully compatible
            return not (changes['major_changed'] or changes['minor_changed'])
        else:
            return False
    
    def suggest_version_bump(
        self,
        changes: List[str],
        current_version: str
    ) -> Tuple[str, str]:
        """
        Suggest version bump based on changes
        
        Args:
            changes: List of change descriptions
            current_version: Current version string
            
        Returns:
            Tuple of (suggested_increment_type, suggested_version)
        """
        version = SemanticVersion(current_version)
        
        # Determine if changes are breaking
        breaking_keywords = ['breaking', 'major', 'remove', 'delete', 'incompatible']
        feature_keywords = ['feature', 'add', 'enhance', 'improve']
        fix_keywords = ['fix', 'bug', 'patch', 'hotfix']
        
        has_breaking = any(
            any(keyword in change.lower() for keyword in breaking_keywords)
            for change in changes
        )
        
        has_features = any(
            any(keyword in change.lower() for keyword in feature_keywords)
            for change in changes
        )
        
        has_fixes = any(
            any(keyword in change.lower() for keyword in fix_keywords)
            for change in changes
        )
        
        if has_breaking:
            increment_type = 'major'
            new_version = str(version.bump_major())
        elif has_features:
            increment_type = 'minor'
            new_version = str(version.bump_minor())
        elif has_fixes:
            increment_type = 'patch'
            new_version = str(version.bump_patch())
        else:
            increment_type = 'patch'
            new_version = str(version.bump_patch())
        
        return increment_type, new_version


class ChangeTracker:
    """Track and manage changes"""
    
    def __init__(self, tracker_path: Optional[str] = None):
        """Initialize change tracker"""
        self.tracker_path = Path(tracker_path or './.version_tracker')
        self.tracker_path.mkdir(parents=True, exist_ok=True)
        self.changes_file = self.tracker_path / 'changes.json'
        self.logger = logging.getLogger(f"ChangeTracker.{id(self)}")
        
        self._init_changes_file()
    
    def _init_changes_file(self):
        """Initialize changes file"""
        if not self.changes_file.exists():
            with open(self.changes_file, 'w') as f:
                json.dump({
                    'changes': [],
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
    
    def record_change(
        self,
        change_type: str,
        description: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a change"""
        with open(self.changes_file) as f:
            data = json.load(f)
        
        change = {
            'id': len(data['changes']) + 1,
            'type': change_type,
            'description': description,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        data['changes'].append(change)
        data['last_updated'] = datetime.now().isoformat()
        
        with open(self.changes_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_changes(self, version: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get changes, optionally filtered by version"""
        with open(self.changes_file) as f:
            data = json.load(f)
        
        changes = data['changes']
        
        if version:
            changes = [c for c in changes if c['version'] == version]
        
        return changes
    
    def get_change_summary(self) -> Dict[str, Any]:
        """Get change summary"""
        with open(self.changes_file) as f:
            data = json.load(f)
        
        changes = data['changes']
        
        # Group by version
        by_version = {}
        by_type = {}
        
        for change in changes:
            # By version
            version = change['version']
            if version not in by_version:
                by_version[version] = 0
            by_version[version] += 1
            
            # By type
            change_type = change['type']
            if change_type not in by_type:
                by_type[change_type] = 0
            by_type[change_type] += 1
        
        return {
            'total_changes': len(changes),
            'by_version': by_version,
            'by_type': by_type,
            'last_updated': data['last_updated']
        }


def create_version_tracker(repo_path: Optional[str] = None) -> VersionTracker:
    """
    Factory function to create VersionTracker instance
    
    Args:
        repo_path: Optional repository path
        
    Returns:
        VersionTracker instance
    """
    return VersionTracker(repo_path)


def get_next_version(
    current_version: Optional[str] = None,
    increment_type: str = 'patch',
    repo_path: Optional[str] = None
) -> str:
    """
    Convenience function to get next version
    
    Args:
        current_version: Current version string
        increment_type: Type of increment (major, minor, patch)
        repo_path: Repository path
        
    Returns:
        Next version string
    """
    tracker = VersionTracker(repo_path)
    return tracker.increment_version(current_version, increment_type)


def generate_version_from_git(
    increment_type: str = 'patch',
    repo_path: Optional[str] = None
) -> str:
    """
    Convenience function to generate version from git state
    
    Args:
        increment_type: Type of increment
        repo_path: Repository path
        
    Returns:
        Generated version string
    """
    tracker = VersionTracker(repo_path)
    return tracker.generate_version_from_git(increment_type)