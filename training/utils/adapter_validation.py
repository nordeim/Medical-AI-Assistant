"""
Adapter Compatibility Validation Utilities

This module provides comprehensive validation for adapter compatibility,
model architecture checks, and deployment readiness verification.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel
from peft import PeftConfig, PeftModel

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Adapter compatibility levels."""
    FULL = "full"           # 100% compatible
    PARTIAL = "partial"     # Requires adaptation
    INCOMPATIBLE = "incompatible"  # Cannot be used
    UNKNOWN = "unknown"     # Cannot determine


@dataclass
class CompatibilityIssue:
    """Represents a compatibility issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "architecture", "config", "model", "tokenizer"
    message: str
    suggestion: Optional[str] = None
    can_fix: bool = False


@dataclass
class ValidationResult:
    """Result of compatibility validation."""
    adapter_id: str
    compatibility_level: CompatibilityLevel
    overall_score: float  # 0.0 to 1.0
    issues: List[CompatibilityIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return any(issue.severity == "error" for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return any(issue.severity == "warning" for issue in self.issues)


class ModelArchitectureValidator:
    """Validates model architecture compatibility."""
    
    def __init__(self):
        self.supported_architectures = {
            "llama": {
                "model_types": ["llama"],
                "supported_peft_types": ["lora", "adalora", "ia3"],
                "required_config_fields": ["hidden_size", "num_attention_heads"]
            },
            "mistral": {
                "model_types": ["mistral"],
                "supported_peft_types": ["lora", "adalora", "ia3"],
                "required_config_fields": ["hidden_size", "num_attention_heads"]
            },
            "gpt2": {
                "model_types": ["gpt2", "gpt_neo", "gptj"],
                "supported_peft_types": ["lora", "adalora"],
                "required_config_fields": ["hidden_size", "num_attention_heads"]
            },
            "t5": {
                "model_types": ["t5", "by5"],
                "supported_peft_types": ["lora", "prefix_tuning", "p_tuning"],
                "required_config_fields": ["d_model", "num_heads"]
            },
            "bart": {
                "model_types": ["bart", "mbart"],
                "supported_peft_types": ["lora", "prefix_tuning"],
                "required_config_fields": ["d_model", "encoder_attention_heads"]
            }
        }
        
        self.unsupported_architectures = {
            "bert": "Use BERT-specific adapters",
            "roberta": "Use RoBERTa-specific adapters", 
            "distilbert": "Use DistilBERT-specific adapters"
        }
    
    def validate_model_architecture(self, 
                                   model_config: Dict[str, Any], 
                                   adapter_config: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Validate model architecture compatibility."""
        issues = []
        
        # Get model architecture
        model_type = model_config.get("model_type", "").lower()
        
        # Check if architecture is supported
        if model_type in self.unsupported_architectures:
            issues.append(CompatibilityIssue(
                severity="error",
                category="architecture",
                message=f"Model architecture '{model_type}' is not directly supported",
                suggestion=self.unsupported_architectures[model_type],
                can_fix=False
            ))
            return issues
        
        # Find compatible architecture family
        compatible_arch = None
        for arch, config in self.supported_architectures.items():
            if model_type in config["model_types"]:
                compatible_arch = arch
                break
        
        if not compatible_arch:
            issues.append(CompatibilityIssue(
                severity="warning",
                category="architecture", 
                message=f"Model architecture '{model_type}' may require adaptation",
                suggestion="Test adapter compatibility with sample inputs",
                can_fix=True
            ))
        else:
            # Check PEFT type compatibility
            peft_type = adapter_config.get("peft_type", "").lower()
            supported_peft_types = self.supported_architectures[compatible_arch]["supported_peft_types"]
            
            if peft_type not in supported_peft_types:
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="architecture",
                    message=f"PEFT type '{peft_type}' not supported for {model_type} architecture",
                    suggestion=f"Supported PEFT types: {supported_peft_types}",
                    can_fix=False
                ))
            else:
                logger.info(f"Architecture compatibility confirmed: {model_type} + {peft_type}")
        
        # Validate required configuration fields
        required_fields = self.supported_architectures.get(compatible_arch, {}).get("required_config_fields", [])
        for field in required_fields:
            if field not in model_config:
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="config",
                    message=f"Required config field '{field}' missing from model config",
                    suggestion="Check if base model config is complete",
                    can_fix=False
                ))
        
        return issues


class PEFTConfigValidator:
    """Validates PEFT adapter configuration."""
    
    def __init__(self):
        self.peft_config_schemas = {
            "lora": {
                "required_fields": ["peft_type", "r", "lora_alpha", "lora_dropout"],
                "optional_fields": ["target_modules", "fan_in_fan_out", "bias"],
                "value_ranges": {
                    "r": (1, 1024),
                    "lora_alpha": (1, 1024),
                    "lora_dropout": (0.0, 1.0)
                }
            },
            "adalora": {
                "required_fields": ["peft_type", "target_r", "r", "lora_alpha"],
                "optional_fields": ["target_modules", "dropout_p"],
                "value_ranges": {
                    "target_r": (1, 128),
                    "r": (1, 64),
                    "lora_alpha": (1, 1024)
                }
            },
            "ia3": {
                "required_fields": ["peft_type", "target_modules"],
                "optional_fields": [],
                "value_ranges": {}
            },
            "prefix_tuning": {
                "required_fields": ["peft_type", "num_virtual_tokens"],
                "optional_fields": ["encoder_hidden_size"],
                "value_ranges": {
                    "num_virtual_tokens": (1, 100)
                }
            },
            "p_tuning": {
                "required_fields": ["peft_type", "encoder_hidden_size"],
                "optional_fields": [],
                "value_ranges": {
                    "encoder_hidden_size": (64, 8192)
                }
            }
        }
    
    def validate_peft_config(self, adapter_config: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Validate PEFT adapter configuration."""
        issues = []
        
        peft_type = adapter_config.get("peft_type", "").lower()
        
        if peft_type not in self.peft_config_schemas:
            issues.append(CompatibilityIssue(
                severity="error",
                category="config",
                message=f"Unknown PEFT type: {peft_type}",
                suggestion=f"Supported types: {list(self.peft_config_schemas.keys())}",
                can_fix=False
            ))
            return issues
        
        schema = self.peft_config_schemas[peft_type]
        
        # Check required fields
        for field in schema["required_fields"]:
            if field not in adapter_config:
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="config",
                    message=f"Required field '{field}' missing from adapter config",
                    suggestion=f"Add {field} to adapter configuration",
                    can_fix=True
                ))
        
        # Check field value ranges
        value_ranges = schema.get("value_ranges", {})
        for field, (min_val, max_val) in value_ranges.items():
            if field in adapter_config:
                value = adapter_config[field]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    issues.append(CompatibilityIssue(
                        severity="warning",
                        category="config",
                        message=f"Field '{field}' value {value} outside recommended range [{min_val}, {max_val}]",
                        suggestion=f"Consider using value in range [{min_val}, {max_val}]",
                        can_fix=True
                    ))
        
        # Validate target_modules for supported PEFT types
        if peft_type in ["lora", "adalora", "ia3"]:
            target_modules = adapter_config.get("target_modules", [])
            if not target_modules:
                issues.append(CompatibilityIssue(
                    severity="warning",
                    category="config",
                    message="No target_modules specified",
                    suggestion="Specify target modules for better performance",
                    can_fix=True
                ))
            else:
                # Validate target module names
                valid_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                invalid_modules = [m for m in target_modules if m not in valid_modules]
                if invalid_modules:
                    issues.append(CompatibilityIssue(
                        severity="warning",
                        category="config",
                        message=f"Potentially invalid target modules: {invalid_modules}",
                        suggestion=f"Valid modules: {valid_modules}",
                        can_fix=True
                    ))
        
        return issues


class TokenizerCompatibilityValidator:
    """Validates tokenizer compatibility."""
    
    def __init__(self):
        self.tokenizer_special_tokens = {
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "bos_token": "<s>",
            "unk_token": "<unk>"
        }
        
        self.model_specific_requirements = {
            "llama": {
                "required_special_tokens": ["pad_token", "eos_token"],
                "special_token_ids": {
                    "eos_token_id": 2,
                    "bos_token_id": 1
                }
            },
            "mistral": {
                "required_special_tokens": ["pad_token", "eos_token"],
                "special_token_ids": {
                    "eos_token_id": 2,
                    "bos_token_id": 1
                }
            }
        }
    
    def validate_tokenizer_compatibility(self, 
                                        tokenizer_config: Dict[str, Any],
                                        model_config: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Validate tokenizer compatibility with model."""
        issues = []
        
        model_type = model_config.get("model_type", "").lower()
        
        # Check if model has specific requirements
        if model_type not in self.model_specific_requirements:
            # Generic validation
            issues.extend(self._validate_generic_tokenizer(tokenizer_config))
        else:
            # Model-specific validation
            reqs = self.model_specific_requirements[model_type]
            
            # Check required special tokens
            required_tokens = reqs["required_special_tokens"]
            for token in required_tokens:
                if token not in tokenizer_config:
                    issues.append(CompatibilityIssue(
                        severity="warning",
                        category="tokenizer",
                        message=f"Required special token '{token}' not found in tokenizer",
                        suggestion="Ensure tokenizer has all required special tokens",
                        can_fix=True
                    ))
            
            # Check special token IDs
            special_token_ids = reqs.get("special_token_ids", {})
            for token_name, expected_id in special_token_ids.items():
                if token_name in tokenizer_config:
                    actual_id = tokenizer_config[token_name]
                    if actual_id != expected_id:
                        issues.append(CompatibilityIssue(
                            severity="warning",
                            category="tokenizer",
                            message=f"Special token ID mismatch for '{token_name}': expected {expected_id}, got {actual_id}",
                            suggestion="Verify tokenizer configuration matches model requirements",
                            can_fix=True
                        ))
        
        # Check tokenizer vocab size compatibility
        vocab_size = tokenizer_config.get("vocab_size")
        model_vocab_size = model_config.get("vocab_size")
        
        if vocab_size and model_vocab_size:
            if vocab_size != model_vocab_size:
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="tokenizer",
                    message=f"Vocab size mismatch: tokenizer={vocab_size}, model={model_vocab_size}",
                    suggestion="Ensure tokenizer vocab size matches model vocab size",
                    can_fix=False
                ))
        
        return issues
    
    def _validate_generic_tokenizer(self, tokenizer_config: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Generic tokenizer validation."""
        issues = []
        
        # Check for basic special tokens
        for token_name, token_value in self.tokenizer_special_tokens.items():
            if token_name not in tokenizer_config:
                issues.append(CompatibilityIssue(
                    severity="info",
                    category="tokenizer",
                    message=f"Special token '{token_name}' not found",
                    suggestion="Consider adding standard special tokens for better compatibility",
                    can_fix=True
                ))
        
        return issues


class PerformanceValidator:
    """Validates adapter performance characteristics."""
    
    def __init__(self):
        self.performance_thresholds = {
            "model_size_mb": 1000,  # Maximum reasonable model size
            "load_time_seconds": 300,  # Maximum load time
            "memory_usage_gb": 16,  # Maximum memory usage
            "inference_latency_ms": 100,  # Maximum inference latency
            "max_context_length": 8192  # Maximum context length
        }
    
    def validate_performance(self, 
                           adapter_path: str,
                           metadata: Dict[str, Any]) -> List[CompatibilityIssue]:
        """Validate adapter performance characteristics."""
        issues = []
        
        # Check model file size
        file_size_mb = metadata.get("file_size", 0) / (1024 * 1024)
        max_size = self.performance_thresholds["model_size_mb"]
        
        if file_size_mb > max_size:
            issues.append(CompatibilityIssue(
                severity="warning",
                category="performance",
                message=f"Adapter size {file_size_mb:.1f}MB exceeds recommended maximum {max_size}MB",
                suggestion="Consider model compression or adapter pruning",
                can_fix=True
            ))
        
        # Check predicted load time
        predicted_load_time = self._estimate_load_time(file_size_mb)
        max_load_time = self.performance_thresholds["load_time_seconds"]
        
        if predicted_load_time > max_load_time:
            issues.append(CompatibilityIssue(
                severity="info",
                category="performance",
                message=f"Estimated load time {predicted_load_time:.1f}s may be slow",
                suggestion="Consider using smaller adapters or caching strategies",
                can_fix=True
            ))
        
        # Check context length compatibility
        context_length = metadata.get("max_position_embeddings", 0)
        max_context = self.performance_thresholds["max_context_length"]
        
        if context_length > max_context:
            issues.append(CompatibilityIssue(
                severity="warning",
                category="performance",
                message=f"Context length {context_length} exceeds recommended maximum {max_context}",
                suggestion="May impact performance and memory usage",
                can_fix=False
            ))
        
        return issues
    
    def _estimate_load_time(self, file_size_mb: float) -> float:
        """Estimate adapter load time based on file size."""
        # Simple estimation - real implementation would be more sophisticated
        base_time = 5.0  # Base load time
        size_factor = file_size_mb / 100.0  # Additional time based on size
        return base_time + size_factor


class DeploymentValidator:
    """Validates deployment readiness."""
    
    def __init__(self):
        self.deployment_requirements = {
            "required_files": ["adapter_config.json", "adapter_model.safetensors"],
            "recommended_files": ["tokenizer.json", "special_tokens_map.json"],
            "validation_commands": [
                self._validate_adapter_loading,
                self._validate_model_format
            ]
        }
    
    def validate_deployment_readiness(self, adapter_path: str) -> List[CompatibilityIssue]:
        """Validate adapter deployment readiness."""
        issues = []
        
        adapter_dir = Path(adapter_path)
        
        # Check required files
        for required_file in self.deployment_requirements["required_files"]:
            file_path = adapter_dir / required_file
            if not file_path.exists():
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="deployment",
                    message=f"Required file '{required_file}' missing",
                    suggestion="Ensure all adapter files are present",
                    can_fix=False
                ))
        
        # Check recommended files
        for recommended_file in self.deployment_requirements["recommended_files"]:
            file_path = adapter_dir / recommended_file
            if not file_path.exists():
                issues.append(CompatibilityIssue(
                    severity="warning",
                    category="deployment",
                    message=f"Recommended file '{recommended_file}' missing",
                    suggestion="Consider adding tokenizer files for better compatibility",
                    can_fix=True
                ))
        
        # Validate adapter loading
        try:
            loading_issues = self._validate_adapter_loading(adapter_path)
            issues.extend(loading_issues)
        except Exception as e:
            issues.append(CompatibilityIssue(
                severity="error",
                category="deployment",
                message=f"Adapter loading validation failed: {str(e)}",
                suggestion="Check adapter files for corruption",
                can_fix=False
            ))
        
        return issues
    
    def _validate_adapter_loading(self, adapter_path: str) -> List[CompatibilityIssue]:
        """Validate adapter can be loaded successfully."""
        issues = []
        
        try:
            # Try to load adapter config
            config_path = Path(adapter_path) / "adapter_config.json"
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Basic config validation
            if "peft_type" not in config:
                issues.append(CompatibilityIssue(
                    severity="error",
                    category="deployment",
                    message="Adapter config missing peft_type",
                    suggestion="Fix adapter configuration",
                    can_fix=False
                ))
            
            # Try to load PEFT model (this would require a base model)
            # Note: This is a simplified check - real implementation would be more thorough
            
        except Exception as e:
            issues.append(CompatibilityIssue(
                severity="error",
                category="deployment",
                message=f"Failed to load adapter config: {str(e)}",
                suggestion="Check adapter config file format",
                can_fix=False
            ))
        
        return issues
    
    def _validate_model_format(self, adapter_path: str) -> List[CompatibilityIssue]:
        """Validate adapter model file format."""
        issues = []
        
        # Check for common model file formats
        model_files = list(Path(adapter_path).glob("*.safetensors")) + \
                     list(Path(adapter_path).glob("*.bin")) + \
                     list(Path(adapter_path).glob("*.ckpt"))
        
        if not model_files:
            issues.append(CompatibilityIssue(
                severity="error",
                category="deployment",
                message="No model files found in adapter directory",
                suggestion="Ensure adapter model files are present",
                can_fix=False
            ))
        
        return issues


class AdapterCompatibilityValidator:
    """Main adapter compatibility validation class."""
    
    def __init__(self):
        self.model_validator = ModelArchitectureValidator()
        self.peft_validator = PEFTConfigValidator()
        self.tokenizer_validator = TokenizerCompatibilityValidator()
        self.performance_validator = PerformanceValidator()
        self.deployment_validator = DeploymentValidator()
    
    def validate_adapter(self, 
                        adapter_path: str,
                        base_model_id: str,
                        adapter_id: Optional[str] = None) -> ValidationResult:
        """Comprehensive adapter compatibility validation."""
        
        adapter_id = adapter_id or Path(adapter_path).name
        issues = []
        warnings = []
        recommendations = []
        
        try:
            # Load adapter config
            adapter_config = self._load_adapter_config(adapter_path)
            if not adapter_config:
                return ValidationResult(
                    adapter_id=adapter_id,
                    compatibility_level=CompatibilityLevel.INCOMPATIBLE,
                    overall_score=0.0,
                    issues=[CompatibilityIssue(
                        severity="error",
                        category="config",
                        message="Failed to load adapter configuration",
                        can_fix=False
                    )]
                )
            
            # Load base model config
            base_model_config = self._load_model_config(base_model_id)
            if not base_model_config:
                return ValidationResult(
                    adapter_id=adapter_id,
                    compatibility_level=CompatibilityLevel.UNKNOWN,
                    overall_score=0.0,
                    issues=[CompatibilityIssue(
                        severity="error",
                        category="config",
                        message="Failed to load base model configuration",
                        can_fix=False
                    )]
                )
            
            # Load tokenizer config
            tokenizer_config = self._load_tokenizer_config(base_model_id)
            
            # Run validation checks
            issues.extend(self.model_validator.validate_model_architecture(
                base_model_config, adapter_config
            ))
            
            issues.extend(self.peft_validator.validate_peft_config(adapter_config))
            
            if tokenizer_config:
                issues.extend(self.tokenizer_validator.validate_tokenizer_compatibility(
                    tokenizer_config, base_model_config
                ))
            
            # Load adapter metadata
            metadata = self._get_adapter_metadata(adapter_path)
            issues.extend(self.performance_validator.validate_performance(
                adapter_path, metadata
            ))
            
            issues.extend(self.deployment_validator.validate_deployment_readiness(adapter_path))
            
            # Calculate compatibility score
            compatibility_score = self._calculate_compatibility_score(issues)
            compatibility_level = self._determine_compatibility_level(issues)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(issues)
            
            return ValidationResult(
                adapter_id=adapter_id,
                compatibility_level=compatibility_level,
                overall_score=compatibility_score,
                issues=issues,
                warnings=[issue.message for issue in issues if issue.severity == "warning"],
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Validation failed for adapter {adapter_id}: {str(e)}")
            return ValidationResult(
                adapter_id=adapter_id,
                compatibility_level=CompatibilityLevel.UNKNOWN,
                overall_score=0.0,
                issues=[CompatibilityIssue(
                    severity="error",
                    category="system",
                    message=f"Validation system error: {str(e)}",
                    can_fix=False
                )]
            )
    
    def _load_adapter_config(self, adapter_path: str) -> Optional[Dict[str, Any]]:
        """Load adapter configuration."""
        try:
            config_path = Path(adapter_path) / "adapter_config.json"
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load adapter config: {str(e)}")
            return None
    
    def _load_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load base model configuration."""
        try:
            config = AutoConfig.from_pretrained(model_id)
            return config.to_dict()
        except Exception as e:
            logger.error(f"Failed to load model config: {str(e)}")
            return None
    
    def _load_tokenizer_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load tokenizer configuration."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            return tokenizer.save_vocabulary(".")[0] if hasattr(tokenizer, 'save_vocabulary') else {}
        except Exception as e:
            logger.warning(f"Failed to load tokenizer config: {str(e)}")
            return {}
    
    def _get_adapter_metadata(self, adapter_path: str) -> Dict[str, Any]:
        """Get adapter metadata."""
        adapter_dir = Path(adapter_path)
        
        # Calculate file size
        total_size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())
        
        # Try to get additional metadata from config
        metadata = {"file_size": total_size}
        
        try:
            adapter_config = self._load_adapter_config(adapter_path)
            if adapter_config:
                # Extract relevant metadata
                if "max_position_embeddings" in adapter_config:
                    metadata["max_position_embeddings"] = adapter_config["max_position_embeddings"]
                
                if "vocab_size" in adapter_config:
                    metadata["vocab_size"] = adapter_config["vocab_size"]
                    
        except:
            pass
        
        return metadata
    
    def _calculate_compatibility_score(self, issues: List[CompatibilityIssue]) -> float:
        """Calculate compatibility score based on issues."""
        if not issues:
            return 1.0
        
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        
        # Start with perfect score and deduct for issues
        score = 1.0
        score -= error_count * 0.3  # Each error reduces score significantly
        score -= warning_count * 0.1  # Each warning reduces score slightly
        
        return max(0.0, score)
    
    def _determine_compatibility_level(self, issues: List[CompatibilityIssue]) -> CompatibilityLevel:
        """Determine compatibility level based on issues."""
        has_errors = any(issue.severity == "error" for issue in issues)
        has_warnings = any(issue.severity == "warning" for issue in issues)
        
        if has_errors:
            return CompatibilityLevel.INCOMPATIBLE
        elif has_warnings:
            return CompatibilityLevel.PARTIAL
        else:
            return CompatibilityLevel.FULL
    
    def _generate_recommendations(self, issues: List[CompatibilityIssue]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        for issue in issues:
            if issue.suggestion:
                recommendations.append(f"{issue.category}: {issue.suggestion}")
        
        # Add general recommendations based on issue patterns
        config_issues = [i for i in issues if i.category == "config" and i.severity == "warning"]
        if config_issues:
            recommendations.append("Consider reviewing adapter configuration for optimization")
        
        performance_issues = [i for i in issues if i.category == "performance" and i.severity == "warning"]
        if performance_issues:
            recommendations.append("Monitor adapter performance in production")
        
        return list(set(recommendations))  # Remove duplicates


# Utility functions
def validate_adapter_compatibility(adapter_path: str, 
                                 base_model_id: str,
                                 adapter_id: Optional[str] = None) -> ValidationResult:
    """Quick validation function."""
    validator = AdapterCompatibilityValidator()
    return validator.validate_adapter(adapter_path, base_model_id, adapter_id)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create validator
        validator = AdapterCompatibilityValidator()
        
        # Validate an adapter
        result = validator.validate_adapter(
            adapter_path="./sample_adapter",
            base_model_id="microsoft/DialoGPT-medium"
        )
        
        print(f"Compatibility Level: {result.compatibility_level.value}")
        print(f"Overall Score: {result.overall_score:.2f}")
        
        if result.issues:
            print("\nIssues:")
            for issue in result.issues:
                print(f"  [{issue.severity.upper()}] {issue.category}: {issue.message}")
        
        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
    
    # Run example
    # asyncio.run(main())