"""
Adapter Validator with Medical Model Compatibility

Provides comprehensive validation for adapter compatibility,
medical compliance, and production readiness.
"""

import json
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import yaml
from transformers import AutoConfig, AutoTokenizer, AutoModel
from peft import PeftConfig

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Adapter compatibility levels."""
    FULLY_COMPATIBLE = "fully_compatible"
    MOSTLY_COMPATIBLE = "mostly_compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class ValidationSeverity(Enum):
    """Validation issue severity."""
    CRITICAL = "critical"      # Blocks deployment
    ERROR = "error"            # Deployment not recommended
    WARNING = "warning"        # Deployment possible with caution
    INFO = "info"              # Informational only


class MedicalComplianceLevel(Enum):
    """Medical AI compliance levels."""
    HIPAA_COMPLIANT = "hipaa_compliant"
    FDA_COMPLIANT = "fda_compliant"
    CLINICALLY_VALIDATED = "clinically_validated"
    RESEARCH_ONLY = "research_only"
    NON_MEDICAL = "non_medical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    suggestion: Optional[str] = None
    code: Optional[str] = None
    can_fix: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "suggestion": self.suggestion,
            "code": self.code,
            "can_fix": self.can_fix
        }


@dataclass
class ValidationResult:
    """Result of adapter validation."""
    adapter_id: str
    is_compatible: bool
    compatibility_level: CompatibilityLevel
    compatibility_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    medical_compliance: Optional[MedicalComplianceLevel] = None
    validation_time_ms: float = 0.0
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)
    
    @property
    def has_errors(self) -> bool:
        """Check if there are errors."""
        return any(issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR] 
                  for issue in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "is_compatible": self.is_compatible,
            "compatibility_level": self.compatibility_level.value,
            "compatibility_score": self.compatibility_score,
            "issues": [issue.to_dict() for issue in self.issues],
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "medical_compliance": self.medical_compliance.value if self.medical_compliance else None,
            "validation_time_ms": self.validation_time_ms
        }


class ModelArchitectureValidator:
    """Validates model architecture compatibility."""
    
    def __init__(self):
        self.supported_architectures = {
            "llama": {
                "model_types": ["llama"],
                "supported_peft_types": ["lora", "adalora", "ia3"],
                "medical_compatible": True
            },
            "mistral": {
                "model_types": ["mistral"],
                "supported_peft_types": ["lora", "adalora", "ia3"],
                "medical_compatible": True
            },
            "phi": {
                "model_types": ["phi"],
                "supported_peft_types": ["lora", "adalora"],
                "medical_compatible": True
            },
            "gpt2": {
                "model_types": ["gpt2", "gpt_neo", "gptj"],
                "supported_peft_types": ["lora", "adalora"],
                "medical_compatible": False
            },
            "t5": {
                "model_types": ["t5", "by5"],
                "supported_peft_types": ["lora", "prefix_tuning", "p_tuning"],
                "medical_compatible": True
            },
            "bart": {
                "model_types": ["bart", "mbart"],
                "supported_peft_types": ["lora", "prefix_tuning"],
                "medical_compatible": False
            }
        }
        
        self.medical_model_requirements = {
            "required_fields": ["vocab_size", "hidden_size", "num_attention_heads"],
            "recommended_fields": ["max_position_embeddings", "intermediate_size"],
            "parameter_ranges": {
                "hidden_size": (256, 8192),
                "num_attention_heads": (1, 128),
                "num_layers": (1, 100),
                "max_position_embeddings": (512, 32768)
            }
        }
    
    def validate_architecture(self, 
                            model_config: Dict[str, Any],
                            adapter_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate model architecture compatibility."""
        issues = []
        
        model_type = model_config.get("model_type", "").lower()
        
        # Check if architecture is supported
        if model_type not in self.supported_architectures:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="architecture",
                message=f"Model architecture '{model_type}' is not explicitly supported",
                suggestion="Test adapter compatibility thoroughly before deployment",
                code="ARCH_NOT_SUPPORTED"
            ))
            return issues
        
        architecture_info = self.supported_architectures[model_type]
        
        # Check PEFT type compatibility
        peft_type = adapter_config.get("peft_type", "").lower()
        supported_peft_types = architecture_info["supported_peft_types"]
        
        if peft_type not in supported_peft_types:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="architecture",
                message=f"PEFT type '{peft_type}' not supported for {model_type}",
                suggestion=f"Supported PEFT types: {', '.join(supported_peft_types)}",
                code="PEFT_NOT_SUPPORTED",
                can_fix=True
            ))
        else:
            logger.info(f"Architecture compatibility confirmed: {model_type} + {peft_type}")
        
        # Validate medical model requirements
        if architecture_info.get("medical_compatible"):
            issues.extend(self._validate_medical_requirements(model_config))
        
        # Validate parameter ranges
        issues.extend(self._validate_parameter_ranges(model_config))
        
        return issues
    
    def _validate_medical_requirements(self, model_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate medical model requirements."""
        issues = []
        
        # Check required fields
        for field in self.medical_model_requirements["required_fields"]:
            if field not in model_config:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="medical",
                    message=f"Required field '{field}' missing for medical model",
                    suggestion="Ensure model config includes all required fields",
                    code="MISSING_MEDICAL_FIELD"
                ))
        
        # Check recommended fields
        for field in self.medical_model_requirements["recommended_fields"]:
            if field not in model_config:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="medical",
                    message=f"Recommended field '{field}' missing for medical model",
                    suggestion="Consider adding this field for optimal medical AI performance",
                    code="MISSING_RECOMMENDED_FIELD"
                ))
        
        return issues
    
    def _validate_parameter_ranges(self, model_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate parameter value ranges."""
        issues = []
        
        ranges = self.medical_model_requirements["parameter_ranges"]
        
        for param_name, (min_val, max_val) in ranges.items():
            if param_name in model_config:
                value = model_config[param_name]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        message=f"Parameter '{param_name}' value {value} outside recommended range [{min_val}, {max_val}]",
                        suggestion="Verify this parameter value is appropriate for your use case",
                        code="PARAMETER_OUT_OF_RANGE",
                        can_fix=True
                    ))
        
        return issues


class PEFTConfigValidator:
    """Validates PEFT adapter configuration."""
    
    def __init__(self):
        self.peft_schemas = {
            "lora": {
                "required_fields": ["peft_type", "r", "lora_alpha", "lora_dropout"],
                "optional_fields": ["target_modules", "fan_in_fan_out", "bias"],
                "medical_optimized": {
                    "r": (16, 128),  # Balanced for medical tasks
                    "lora_alpha": (32, 256),
                    "lora_dropout": (0.0, 0.2)
                }
            },
            "adalora": {
                "required_fields": ["peft_type", "target_r", "r", "lora_alpha"],
                "optional_fields": ["target_modules", "dropout_p"],
                "medical_optimized": {
                    "target_r": (8, 64),
                    "r": (4, 32),
                    "lora_alpha": (16, 128)
                }
            },
            "ia3": {
                "required_fields": ["peft_type", "target_modules"],
                "optional_fields": [],
                "medical_optimized": {
                    # IA3 specific medical optimizations
                }
            }
        }
    
    def validate_config(self, adapter_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate PEFT adapter configuration."""
        issues = []
        
        peft_type = adapter_config.get("peft_type", "").lower()
        
        if peft_type not in self.peft_schemas:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="configuration",
                message=f"Unknown or unsupported PEFT type: {peft_type}",
                suggestion=f"Supported types: {', '.join(self.peft_schemas.keys())}",
                code="UNKNOWN_PEFT_TYPE"
            ))
            return issues
        
        schema = self.peft_schemas[peft_type]
        
        # Check required fields
        for field in schema["required_fields"]:
            if field not in adapter_config:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="configuration",
                    message=f"Required field '{field}' missing from adapter config",
                    suggestion=f"Add {field} to adapter configuration",
                    code="MISSING_REQUIRED_FIELD",
                    can_fix=True
                ))
        
        # Validate values for medical optimization
        medical_ranges = schema.get("medical_optimized", {})
        for field, (min_val, max_val) in medical_ranges.items():
            if field in adapter_config:
                value = adapter_config[field]
                if isinstance(value, (int, float)) and not (min_val <= value <= max_val):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="medical_optimization",
                        message=f"Field '{field}' value {value} not optimized for medical AI (recommended range: [{min_val}, {max_val}])",
                        suggestion="Consider using medical-optimized values for better performance",
                        code="NOT_MEDICAL_OPTIMIZED",
                        can_fix=True
                    ))
        
        # Validate target modules for medical models
        if peft_type in ["lora", "adalora", "ia3"]:
            issues.extend(self._validate_target_modules(adapter_config))
        
        return issues
    
    def _validate_target_modules(self, adapter_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate target modules for medical models."""
        issues = []
        
        target_modules = adapter_config.get("target_modules", [])
        
        if not target_modules:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="configuration",
                message="No target_modules specified for LoRA adapter",
                suggestion="Specify target modules for better control and performance",
                code="NO_TARGET_MODULES",
                can_fix=True
            ))
        else:
            # Validate module names
            valid_modules = {
                "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "mlp": ["gate_proj", "up_proj", "down_proj"],
                "all": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            }
            
            all_valid = any(
                module in valid_modules["all"] for module in target_modules
            )
            
            if not all_valid:
                invalid_modules = [m for m in target_modules if m not in valid_modules["all"]]
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="configuration",
                    message=f"Potentially invalid target modules: {invalid_modules}",
                    suggestion=f"Valid modules: {', '.join(valid_modules['all'])}",
                    code="INVALID_TARGET_MODULES",
                    can_fix=True
                ))
        
        return issues


class TokenizerCompatibilityValidator:
    """Validates tokenizer compatibility."""
    
    def __init__(self):
        self.medical_special_tokens = {
            "clinical_terms": ["<CLINICAL_TERM>", "<MEDICAL_CONDITION>", "<DIAGNOSIS>", "<TREATMENT>"],
            "safety_tokens": ["<SAFE_RESPONSE>", "<UNSAFE_CONTENT>", "<PRIVACY_PROTECTED>"],
            "compliance_tokens": ["<HIPAA_COMPLIANT>", "<PHI_PROTECTED>", "<CONSENT_REQUIRED>"]
        }
        
        self.model_specific_requirements = {
            "llama": {
                "required_special_tokens": ["pad_token", "eos_token", "bos_token"],
                "tokenizer_class": "LlamaTokenizer"
            },
            "mistral": {
                "required_special_tokens": ["pad_token", "eos_token", "bos_token"],
                "tokenizer_class": "MistralTokenizer"
            },
            "phi": {
                "required_special_tokens": ["pad_token", "eos_token"],
                "tokenizer_class": "PhiTokenizer"
            }
        }
    
    def validate_compatibility(self,
                              tokenizer_config: Dict[str, Any],
                              model_config: Dict[str, Any],
                              adapter_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate tokenizer compatibility."""
        issues = []
        
        model_type = model_config.get("model_type", "").lower()
        
        # Check vocab size compatibility
        vocab_size = tokenizer_config.get("vocab_size")
        model_vocab_size = model_config.get("vocab_size")
        
        if vocab_size and model_vocab_size and vocab_size != model_vocab_size:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="tokenizer",
                message=f"Vocab size mismatch: tokenizer={vocab_size}, model={model_vocab_size}",
                suggestion="Ensure tokenizer vocab size matches model vocab size",
                code="VOCAB_SIZE_MISMATCH"
            ))
        
        # Model-specific validation
        if model_type in self.model_specific_requirements:
            issues.extend(
                self._validate_model_specific_tokenizer(
                    tokenizer_config, model_type
                )
            )
        
        # Medical token validation
        issues.extend(self._validate_medical_tokens(tokenizer_config))
        
        # Check for proper tokenizer configuration
        issues.extend(self._validate_tokenizer_structure(tokenizer_config))
        
        return issues
    
    def _validate_model_specific_tokenizer(self, 
                                         tokenizer_config: Dict[str, Any],
                                         model_type: str) -> List[ValidationIssue]:
        """Validate model-specific tokenizer requirements."""
        issues = []
        
        reqs = self.model_specific_requirements[model_type]
        
        # Check required special tokens
        for token in reqs["required_special_tokens"]:
            if token not in tokenizer_config:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="tokenizer",
                    message=f"Required special token '{token}' not found",
                    suggestion=f"Ensure tokenizer has proper {model_type} special tokens",
                    code="MISSING_SPECIAL_TOKEN",
                    can_fix=True
                ))
        
        return issues
    
    def _validate_medical_tokens(self, tokenizer_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate medical AI specific tokens."""
        issues = []
        
        # Check if medical special tokens are present
        vocab = tokenizer_config.get("vocab", {})
        if isinstance(vocab, dict):
            medical_tokens = []
            for category, tokens in self.medical_special_tokens.items():
                for token in tokens:
                    if token in vocab:
                        medical_tokens.append(token)
            
            if not medical_tokens:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    category="medical",
                    message="No medical-specific special tokens found",
                    suggestion="Consider adding medical AI special tokens for better domain adaptation",
                    code="NO_MEDICAL_TOKENS"
                ))
        
        return issues
    
    def _validate_tokenizer_structure(self, tokenizer_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate tokenizer structure and configuration."""
        issues = []
        
        # Check for essential tokenizer components
        required_components = ["vocab", "tokenizer_class"]
        for component in required_components:
            if component not in tokenizer_config:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="tokenizer",
                    message=f"Missing tokenizer component: {component}",
                    suggestion="Verify tokenizer configuration is complete",
                    code="MISSING_TOKENIZER_COMPONENT"
                ))
        
        return issues


class MedicalComplianceValidator:
    """Validates medical AI compliance requirements."""
    
    def __init__(self):
        self.compliance_checklists = {
            MedicalComplianceLevel.HIPAA_COMPLIANT: {
                "required_files": ["phi_redaction_config.json", "audit_log_config.yaml"],
                "required_settings": ["enable_phi_protection", "audit_logging"],
                "security_checks": ["data_encryption", "access_control", "data_minimization"]
            },
            MedicalComplianceLevel.CLINICALLY_VALIDATED: {
                "required_files": ["clinical_validation_report.json", "bias_assessment.yaml"],
                "required_settings": ["fairness_validation", "clinical_trial_data"],
                "validation_checks": ["demographic_parity", "equalized_odds", "clinical_accuracy"]
            }
        }
        
        self.safety_requirements = {
            "max_context_length": 8192,
            "response_filtering": True,
            "content_moderation": True,
            "privacy_protection": True
        }
    
    def validate_compliance(self,
                           adapter_path: str,
                           adapter_config: Dict[str, Any],
                           metadata: Dict[str, Any]) -> Tuple[MedicalComplianceLevel, List[ValidationIssue]]:
        """Validate medical AI compliance."""
        issues = []
        
        # Determine expected compliance level
        expected_level = self._determine_compliance_level(adapter_config, metadata)
        
        # Run compliance checks
        if expected_level in self.compliance_checklists:
            checklist = self.compliance_checklists[expected_level]
            issues.extend(self._run_compliance_checklist(adapter_path, checklist))
        
        # Safety requirements
        issues.extend(self._validate_safety_requirements(adapter_config, metadata))
        
        # Privacy protection
        issues.extend(self._validate_privacy_protection(adapter_path))
        
        return expected_level, issues
    
    def _determine_compliance_level(self, 
                                  adapter_config: Dict[str, Any],
                                  metadata: Dict[str, Any]) -> MedicalComplianceLevel:
        """Determine expected compliance level."""
        # Check metadata for compliance indicators
        compliance_flags = metadata.get("compliance_flags", [])
        
        if "fda_compliant" in compliance_flags:
            return MedicalComplianceLevel.FDA_COMPLIANT
        elif "hipaa_compliant" in compliance_flags:
            return MedicalComplianceLevel.HIPAA_COMPLIANT
        elif "clinically_validated" in compliance_flags:
            return MedicalComplianceLevel.CLINICALLY_VALIDATED
        else:
            # Default to non-medical for safety
            return MedicalComplianceLevel.NON_MEDICAL
    
    def _run_compliance_checklist(self, 
                                adapter_path: str,
                                checklist: Dict[str, List[str]]) -> List[ValidationIssue]:
        """Run compliance checklist."""
        issues = []
        
        adapter_dir = Path(adapter_path)
        
        # Check required files
        for required_file in checklist.get("required_files", []):
            file_path = adapter_dir / required_file
            if not file_path.exists():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category="compliance",
                    message=f"Required compliance file missing: {required_file}",
                    suggestion="Add required compliance documentation",
                    code="MISSING_COMPLIANCE_FILE",
                    can_fix=True
                ))
        
        # Check required settings (would need adapter config validation)
        for required_setting in checklist.get("required_settings", []):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="compliance",
                message=f"Compliance setting should be enabled: {required_setting}",
                suggestion="Review and enable required compliance settings",
                code="MISSING_COMPLIANCE_SETTING",
                can_fix=True
            ))
        
        return issues
    
    def _validate_safety_requirements(self,
                                    adapter_config: Dict[str, Any],
                                    metadata: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate safety requirements."""
        issues = []
        
        # Check context length
        max_length = metadata.get("max_position_embeddings", 0)
        if max_length > self.safety_requirements["max_context_length"]:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="safety",
                message=f"Context length {max_length} exceeds safe maximum {self.safety_requirements['max_context_length']}",
                suggestion="Consider using shorter context lengths for safety",
                code="CONTEXT_LENGTH_HIGH"
            ))
        
        # Check for response filtering
        if not adapter_config.get("enable_response_filtering", False):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="safety",
                message="Response filtering not enabled",
                suggestion="Enable response filtering for medical AI safety",
                code="NO_RESPONSE_FILTERING",
                can_fix=True
            ))
        
        return issues
    
    def _validate_privacy_protection(self, adapter_path: str) -> List[ValidationIssue]:
        """Validate privacy protection mechanisms."""
        issues = []
        
        adapter_dir = Path(adapter_path)
        
        # Check for PHI protection configuration
        phi_config_path = adapter_dir / "phi_protection_config.json"
        if not phi_config_path.exists():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="privacy",
                message="PHI protection configuration not found",
                suggestion="Add PHI protection configuration for medical data",
                code="NO_PHI_CONFIG",
                can_fix=True
            ))
        
        # Check for audit logging
        audit_config_path = adapter_dir / "audit_config.yaml"
        if not audit_config_path.exists():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.INFO,
                category="privacy",
                message="Audit logging configuration not found",
                suggestion="Consider adding audit logging for compliance",
                code="NO_AUDIT_CONFIG",
                can_fix=True
            ))
        
        return issues


class AdapterValidator:
    """Main adapter validation system."""
    
    def __init__(self):
        self.model_validator = ModelArchitectureValidator()
        self.peft_validator = PEFTConfigValidator()
        self.tokenizer_validator = TokenizerCompatibilityValidator()
        self.compliance_validator = MedicalComplianceValidator()
    
    def validate_adapter(self,
                        adapter_path: str,
                        base_model_id: str,
                        adapter_id: str) -> ValidationResult:
        """
        Comprehensive adapter validation.
        
        Args:
            adapter_path: Path to adapter files
            base_model_id: Base model identifier
            adapter_id: Adapter identifier
            
        Returns:
            ValidationResult with comprehensive validation results
        """
        start_time = time.time()
        
        try:
            # Load configurations
            adapter_config = self._load_adapter_config(adapter_path)
            if not adapter_config:
                return self._create_error_result(
                    adapter_id, "Failed to load adapter configuration"
                )
            
            base_model_config = self._load_model_config(base_model_id)
            if not base_model_config:
                return self._create_error_result(
                    adapter_id, "Failed to load base model configuration"
                )
            
            tokenizer_config = self._load_tokenizer_config(base_model_id)
            
            # Run validation checks
            all_issues = []
            
            # Architecture validation
            all_issues.extend(
                self.model_validator.validate_architecture(
                    base_model_config, adapter_config
                )
            )
            
            # PEFT configuration validation
            all_issues.extend(self.peft_validator.validate_config(adapter_config))
            
            # Tokenizer compatibility
            if tokenizer_config:
                all_issues.extend(
                    self.tokenizer_validator.validate_compatibility(
                        tokenizer_config, base_model_config, adapter_config
                    )
                )
            
            # Medical compliance
            metadata = self._get_adapter_metadata(adapter_path)
            compliance_level, compliance_issues = self.compliance_validator.validate_compliance(
                adapter_path, adapter_config, metadata
            )
            all_issues.extend(compliance_issues)
            
            # Calculate compatibility metrics
            compatibility_result = self._calculate_compatibility(all_issues)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(all_issues)
            
            # Create validation result
            validation_time_ms = (time.time() - start_time) * 1000
            
            result = ValidationResult(
                adapter_id=adapter_id,
                is_compatible=not any(
                    issue.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR]
                    for issue in all_issues
                ),
                compatibility_level=compatibility_result["level"],
                compatibility_score=compatibility_result["score"],
                issues=all_issues,
                medical_compliance=compliance_level,
                recommendations=recommendations,
                validation_time_ms=validation_time_ms
            )
            
            logger.info(f"Validation completed for {adapter_id}: "
                       f"compatible={result.is_compatible}, "
                       f"score={result.compatibility_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {adapter_id}: {e}")
            return self._create_error_result(
                adapter_id, f"Validation system error: {str(e)}"
            )
    
    def _load_adapter_config(self, adapter_path: str) -> Optional[Dict[str, Any]]:
        """Load adapter configuration."""
        try:
            config_path = Path(adapter_path) / "adapter_config.json"
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load adapter config: {e}")
            return None
    
    def _load_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load base model configuration."""
        try:
            config = AutoConfig.from_pretrained(model_id)
            return config.to_dict()
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            return None
    
    def _load_tokenizer_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load tokenizer configuration."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            # Convert to dict for validation
            config_dict = {
                "vocab": dict(tokenizer.get_vocab()),
                "vocab_size": tokenizer.vocab_size,
                "tokenizer_class": tokenizer.__class__.__name__
            }
            
            # Add special tokens
            for attr in ["pad_token", "eos_token", "bos_token", "unk_token"]:
                if hasattr(tokenizer, attr):
                    value = getattr(tokenizer, attr)
                    if value is not None:
                        config_dict[attr] = value
            
            return config_dict
            
        except Exception as e:
            logger.warning(f"Failed to load tokenizer config: {e}")
            return {}
    
    def _get_adapter_metadata(self, adapter_path: str) -> Dict[str, Any]:
        """Get adapter metadata."""
        adapter_dir = Path(adapter_path)
        
        metadata = {
            "file_size": sum(
                f.stat().st_size for f in adapter_dir.rglob("*") 
                if f.is_file()
            )
        }
        
        try:
            adapter_config = self._load_adapter_config(adapter_path)
            if adapter_config:
                # Extract relevant metadata
                for key in ["max_position_embeddings", "vocab_size", "compliance_flags"]:
                    if key in adapter_config:
                        metadata[key] = adapter_config[key]
                        
        except:
            pass
        
        return metadata
    
    def _calculate_compatibility(self, 
                               issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Calculate compatibility metrics."""
        if not issues:
            return {
                "level": CompatibilityLevel.FULLY_COMPATIBLE,
                "score": 1.0
            }
        
        # Count issues by severity
        critical_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
        
        # Calculate score
        score = 1.0
        score -= critical_count * 0.5  # Critical issues are severe
        score -= error_count * 0.2
        score -= warning_count * 0.05
        
        score = max(0.0, score)
        
        # Determine level
        if critical_count > 0:
            level = CompatibilityLevel.INCOMPATIBLE
        elif error_count > 0:
            level = CompatibilityLevel.PARTIALLY_COMPATIBLE
        elif warning_count > 0:
            level = CompatibilityLevel.MOSTLY_COMPATIBLE
        else:
            level = CompatibilityLevel.FULLY_COMPATIBLE
        
        return {"level": level, "score": score}
    
    def _generate_recommendations(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on issues."""
        recommendations = []
        
        # Collect suggestions from fixable issues
        for issue in issues:
            if issue.suggestion and issue.can_fix:
                recommendations.append(f"{issue.category}: {issue.suggestion}")
        
        # Add general recommendations based on issue patterns
        config_issues = [i for i in issues if i.category == "configuration" and i.severity == ValidationSeverity.WARNING]
        if config_issues:
            recommendations.append("Review adapter configuration for optimization opportunities")
        
        medical_issues = [i for i in issues if i.category == "medical"]
        if medical_issues:
            recommendations.append("Ensure medical AI compliance requirements are met")
        
        performance_issues = [i for i in issues if i.category == "performance"]
        if performance_issues:
            recommendations.append("Monitor adapter performance in production environment")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _create_error_result(self, adapter_id: str, error_message: str) -> ValidationResult:
        """Create error result for validation failures."""
        return ValidationResult(
            adapter_id=adapter_id,
            is_compatible=False,
            compatibility_level=CompatibilityLevel.UNKNOWN,
            compatibility_score=0.0,
            issues=[ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="system",
                message=error_message,
                code="VALIDATION_ERROR"
            )],
            validation_time_ms=0.0
        )


# Utility functions
def validate_adapter_compatibility(adapter_path: str,
                                 base_model_id: str,
                                 adapter_id: str) -> ValidationResult:
    """Quick validation function."""
    validator = AdapterValidator()
    return validator.validate_adapter(adapter_path, base_model_id, adapter_id)


if __name__ == "__main__":
    # Example usage
    validator = AdapterValidator()
    
    # Validate adapter
    result = validator.validate_adapter(
        adapter_path="./sample_medical_adapter",
        base_model_id="microsoft/DialoGPT-medium",
        adapter_id="medical_diagnosis_v1"
    )
    
    print(f"Compatibility: {result.compatibility_level.value}")
    print(f"Score: {result.compatibility_score:.2f}")
    print(f"Medical Compliance: {result.medical_compliance.value if result.medical_compliance else 'None'}")
    
    if result.issues:
        print("\nIssues:")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.category}: {issue.message}")
    
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  • {rec}")