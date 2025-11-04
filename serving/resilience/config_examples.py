"""
Medical AI Resilience System - Configuration Examples
Example configurations for different medical AI deployment scenarios.
"""

from datetime import timedelta
from resilience import (
    ResilienceConfig, RetryStrategy, ValidationLevel,
    PHIProtectionLevel, DegradationLevel
)


def get_critical_care_config() -> ResilienceConfig:
    """
    Configuration for critical care medical AI systems.
    
    Features:
    - Maximum safety and reliability
    - Quick failure detection
    - Strict patient safety protocols
    - Comprehensive monitoring
    """
    return ResilienceConfig(
        # Error Handling
        enable_critical_error_alerts=True,
        critical_error_timeout=60,  # 1 minute for critical care
        
        # Circuit Breaker
        circuit_breaker_enabled=True,
        default_failure_threshold=2,      # Very sensitive for critical care
        default_recovery_timeout=30.0,    # Quick recovery
        
        # Retry Configuration
        default_max_retries=1,            # Minimal retries for critical operations
        default_timeout=30.0,             # Quick timeout
        enable_medical_safe_retry=True,
        
        # Fallback Strategy
        fallback_enabled=True,
        performance_threshold=0.95,       # High performance threshold
        degradation_levels=[
            DegradationLevel.FULL_CAPABILITY,
            DegradationLevel.REDUCED_PERFORMANCE,
            DegradationLevel.BASIC_FUNCTIONALITY,
            DegradationLevel.EMERGENCY_MODE,
            DegradationLevel.FAILSAFE_MODE
        ],
        
        # Validation
        validation_enabled=True,
        default_validation_level=ValidationLevel.CRITICAL,  # Maximum validation
        phi_protection_enabled=True,
        
        # Health Checks
        health_check_enabled=True,
        health_check_interval=15,         # Frequent checks (15 seconds)
        critical_health_threshold=1,      # Immediate response to failures
        
        # Logging and Audit
        logging_enabled=True,
        audit_enabled=True,
        hipaa_compliance_enabled=True,
        audit_retention_days=2555,        # 7 years for HIPAA
        
        # System Isolation
        medical_isolation_enabled=True,
        patient_safety_mode=True          # Maximum safety mode
    )


def get_routine_clinic_config() -> ResilienceConfig:
    """
    Configuration for routine clinic medical AI systems.
    
    Features:
    - Balanced performance and reliability
    - Standard medical operation protocols
    - Normal monitoring frequency
    """
    return ResilienceConfig(
        # Error Handling
        enable_critical_error_alerts=True,
        critical_error_timeout=300,  # 5 minutes for routine operations
        
        # Circuit Breaker
        circuit_breaker_enabled=True,
        default_failure_threshold=5,      # Standard threshold
        default_recovery_timeout=60.0,    # Standard recovery time
        
        # Retry Configuration
        default_max_retries=3,            # Standard retry attempts
        default_timeout=120.0,            # Standard timeout
        enable_medical_safe_retry=True,
        
        # Fallback Strategy
        fallback_enabled=True,
        performance_threshold=0.80,       # Standard performance threshold
        degradation_levels=[
            DegradationLevel.FULL_CAPABILITY,
            DegradationLevel.REDUCED_PERFORMANCE,
            DegradationLevel.BASIC_FUNCTIONALITY,
            DegradationLevel.EMERGENCY_MODE
        ],
        
        # Validation
        validation_enabled=True,
        default_validation_level=ValidationLevel.STANDARD,  # Standard validation
        phi_protection_enabled=True,
        
        # Health Checks
        health_check_enabled=True,
        health_check_interval=60,         # Standard checks (1 minute)
        critical_health_threshold=3,      # Standard threshold
        
        # Logging and Audit
        logging_enabled=True,
        audit_enabled=True,
        hipaa_compliance_enabled=True,
        audit_retention_days=2555,        # 7 years for HIPAA
        
        # System Isolation
        medical_isolation_enabled=True,
        patient_safety_mode=False         # Standard safety mode
    )


def get_research_system_config() -> ResilienceConfig:
    """
    Configuration for medical research AI systems.
    
    Features:
    - High throughput focus
    - Extended timeouts for complex operations
    - Complete PHI protection
    """
    return ResilienceConfig(
        # Error Handling
        enable_critical_error_alerts=False,  # Research systems may handle more errors
        critical_error_timeout=600,  # 10 minutes for research
        
        # Circuit Breaker
        circuit_breaker_enabled=True,
        default_failure_threshold=10,     # More tolerant for research
        default_recovery_timeout=120.0,   # Longer recovery time
        
        # Retry Configuration
        default_max_retries=5,            # More retries for research
        default_timeout=600.0,            # Extended timeout for research
        enable_medical_safe_retry=True,
        
        # Fallback Strategy
        fallback_enabled=True,
        performance_threshold=0.70,       # Lower threshold for research
        degradation_levels=[
            DegradationLevel.FULL_CAPABILITY,
            DegradationLevel.REDUCED_PERFORMANCE,
            DegradationLevel.BASIC_FUNCTIONALITY
        ],
        
        # Validation
        validation_enabled=True,
        default_validation_level=ValidationLevel.STRICT,  # Strict for research data
        phi_protection_enabled=True,
        
        # Health Checks
        health_check_enabled=True,
        health_check_interval=300,        # Less frequent checks (5 minutes)
        critical_health_threshold=5,      # More tolerant threshold
        
        # Logging and Audit
        logging_enabled=True,
        audit_enabled=True,
        hipaa_compliance_enabled=True,
        audit_retention_days=2555,        # 7 years for HIPAA
        
        # System Isolation
        medical_isolation_enabled=True,
        patient_safety_mode=False         # Research mode
    )


def get_emergency_department_config() -> ResilienceConfig:
    """
    Configuration for emergency department medical AI systems.
    
    Features:
    - Ultra-fast response times
    - Maximum fault tolerance
    - Emergency-specific protocols
    """
    return ResilienceConfig(
        # Error Handling
        enable_critical_error_alerts=True,
        critical_error_timeout=15,  # 15 seconds for emergency
        
        # Circuit Breaker
        circuit_breaker_enabled=True,
        default_failure_threshold=1,      # Fail immediately on errors
        default_recovery_timeout=10.0,    # Very quick recovery
        
        # Retry Configuration
        default_max_retries=0,            # No retries for emergencies
        default_timeout=10.0,             # Ultra-fast timeout
        enable_medical_safe_retry=False,  # No retry for emergencies
        
        # Fallback Strategy
        fallback_enabled=True,
        performance_threshold=0.99,       # Very high threshold
        degradation_levels=[
            DegradationLevel.FULL_CAPABILITY,
            DegradationLevel.EMERGENCY_MODE,
            DegradationLevel.FAILSAFE_MODE
        ],
        
        # Validation
        validation_enabled=True,
        default_validation_level=ValidationLevel.CRITICAL,  # Maximum validation
        phi_protection_enabled=True,
        
        # Health Checks
        health_check_enabled=True,
        health_check_interval=5,          # Very frequent checks (5 seconds)
        critical_health_threshold=1,      # Immediate response
        
        # Logging and Audit
        logging_enabled=True,
        audit_enabled=True,
        hipaa_compliance_enabled=True,
        audit_retention_days=2555,        # 7 years for HIPAA
        
        # System Isolation
        medical_isolation_enabled=True,
        patient_safety_mode=True          # Emergency safety mode
    )


def get_telemetry_monitoring_config() -> ResilienceConfig:
    """
    Configuration for medical telemetry monitoring systems.
    
    Features:
    - High availability focus
    - Continuous monitoring
    - Minimal latency
    """
    return ResilienceConfig(
        # Error Handling
        enable_critical_error_alerts=True,
        critical_error_timeout=30,  # 30 seconds for telemetry
        
        # Circuit Breaker
        circuit_breaker_enabled=True,
        default_failure_threshold=3,      # Moderate sensitivity
        default_recovery_timeout=45.0,    # Quick recovery
        
        # Retry Configuration
        default_max_retries=2,            # Limited retries
        default_timeout=45.0,             # Short timeout
        enable_medical_safe_retry=True,
        
        # Fallback Strategy
        fallback_enabled=True,
        performance_threshold=0.90,       # High threshold for telemetry
        degradation_levels=[
            DegradationLevel.FULL_CAPABILITY,
            DegradationLevel.REDUCED_PERFORMANCE,
            DegradationLevel.EMERGENCY_MODE
        ],
        
        # Validation
        validation_enabled=True,
        default_validation_level=ValidationLevel.STANDARD,  # Standard validation
        phi_protection_enabled=True,
        
        # Health Checks
        health_check_enabled=True,
        health_check_interval=30,         # Frequent checks (30 seconds)
        critical_health_threshold=2,      # Low threshold
        
        # Logging and Audit
        logging_enabled=True,
        audit_enabled=True,
        hipaa_compliance_enabled=True,
        audit_retention_days=2555,        # 7 years for HIPAA
        
        # System Isolation
        medical_isolation_enabled=True,
        patient_safety_mode=True          # Safety mode for telemetry
    )


def get_custom_config(
    environment: str = "development",
    compliance_level: str = "standard",
    performance_priority: str = "balanced"
) -> ResilienceConfig:
    """
    Generate custom configuration based on parameters.
    
    Args:
        environment: development, staging, production
        compliance_level: basic, standard, strict, critical
        performance_priority: speed, balanced, reliability
    
    Returns:
        Configured ResilienceConfig
    """
    # Base configuration
    config = ResilienceConfig()
    
    # Environment-specific adjustments
    if environment == "development":
        config.health_check_interval = 300  # Less frequent in dev
        config.default_timeout = 300.0      # Longer timeouts in dev
    elif environment == "production":
        config.enable_critical_error_alerts = True
        config.health_check_interval = 30   # More frequent in prod
    
    # Compliance level adjustments
    if compliance_level == "critical":
        config.default_validation_level = ValidationLevel.CRITICAL
        config.phi_protection_enabled = True
        config.audit_retention_days = 2555  # Maximum retention
    elif compliance_level == "strict":
        config.default_validation_level = ValidationLevel.STRICT
        config.audit_retention_days = 1826  # 5 years
    elif compliance_level == "basic":
        config.default_validation_level = ValidationLevel.BASIC
        config.audit_retention_days = 365   # 1 year
    
    # Performance priority adjustments
    if performance_priority == "speed":
        config.default_timeout = 30.0
        config.health_check_interval = 60
        config.default_max_retries = 1
    elif performance_priority == "reliability":
        config.default_timeout = 300.0
        config.health_check_interval = 15
        config.default_max_retries = 5
        config.default_failure_threshold = 10
    
    return config


# Example usage functions
async def setup_for_critical_care():
    """Setup resilience system for critical care."""
    from resilience.orchestrator import ResilienceOrchestrator
    
    config = get_critical_care_config()
    orchestrator = ResilienceOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator


async def setup_for_clinic():
    """Setup resilience system for routine clinic."""
    from resilience.orchestrator import ResilienceOrchestrator
    
    config = get_routine_clinic_config()
    orchestrator = ResilienceOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator


async def setup_for_research():
    """Setup resilience system for research."""
    from resilience.orchestrator import ResilienceOrchestrator
    
    config = get_research_system_config()
    orchestrator = ResilienceOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator


async def setup_for_emergency():
    """Setup resilience system for emergency department."""
    from resilience.orchestrator import ResilienceOrchestrator
    
    config = get_emergency_department_config()
    orchestrator = ResilienceOrchestrator(config)
    await orchestrator.initialize()
    return orchestrator


# Configuration templates for common scenarios
SCENARIO_CONFIGS = {
    "critical_care": get_critical_care_config,
    "routine_clinic": get_routine_clinic_config,
    "research_system": get_research_system_config,
    "emergency_department": get_emergency_department_config,
    "telemetry_monitoring": get_telemetry_monitoring_config,
    "custom": get_custom_config
}


def get_config_for_scenario(scenario: str, **kwargs) -> ResilienceConfig:
    """
    Get configuration for a specific scenario.
    
    Args:
        scenario: One of 'critical_care', 'routine_clinic', 'research_system',
                 'emergency_department', 'telemetry_monitoring', or 'custom'
        **kwargs: Additional parameters for custom configuration
    
    Returns:
        ResilienceConfig instance
    """
    if scenario in SCENARIO_CONFIGS:
        if scenario == "custom":
            return get_custom_config(**kwargs)
        else:
            return SCENARIO_CONFIGS[scenario]()
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# Environment variable configuration
def load_config_from_env() -> ResilienceConfig:
    """
    Load configuration from environment variables.
    
    Environment variables:
    - MEDICAL_AI_FAILURE_THRESHOLD: int
    - MEDICAL_AI_RECOVERY_TIMEOUT: float
    - MEDICAL_AI_MAX_RETRIES: int
    - MEDICAL_AI_TIMEOUT: float
    - MEDICAL_AI_HEALTH_CHECK_INTERVAL: int
    - MEDICAL_AI_PATIENT_SAFETY_MODE: bool
    - MEDICAL_AI_PHI_PROTECTION_LEVEL: str
    - MEDICAL_AI_VALIDATION_LEVEL: str
    - MEDICAL_AI_COMPLIANCE_ENABLED: bool
    """
    import os
    
    config = ResilienceConfig()
    
    # Map environment variables to config
    env_mappings = {
        "MEDICAL_AI_FAILURE_THRESHOLD": ("default_failure_threshold", int),
        "MEDICAL_AI_RECOVERY_TIMEOUT": ("default_recovery_timeout", float),
        "MEDICAL_AI_MAX_RETRIES": ("default_max_retries", int),
        "MEDICAL_AI_TIMEOUT": ("default_timeout", float),
        "MEDICAL_AI_HEALTH_CHECK_INTERVAL": ("health_check_interval", int),
    }
    
    bool_mappings = {
        "MEDICAL_AI_PATIENT_SAFETY_MODE": "patient_safety_mode",
        "MEDICAL_AI_COMPLIANCE_ENABLED": "hipaa_compliance_enabled",
    }
    
    # Load numeric values
    for env_var, (config_attr, converter) in env_mappings.items():
        value = os.getenv(env_var)
        if value:
            try:
                setattr(config, config_attr, converter(value))
            except ValueError:
                print(f"Warning: Invalid value for {env_var}: {value}")
    
    # Load boolean values
    for env_var, config_attr in bool_mappings.items():
        value = os.getenv(env_var)
        if value:
            setattr(config, config_attr, value.lower() in ["true", "1", "yes"])
    
    # Load string enum values
    phi_level = os.getenv("MEDICAL_AI_PHI_PROTECTION_LEVEL")
    if phi_level:
        try:
            config.phi_protection_level = PHIProtectionLevel(phi_level)
        except ValueError:
            print(f"Warning: Invalid PHI protection level: {phi_level}")
    
    validation_level = os.getenv("MEDICAL_AI_VALIDATION_LEVEL")
    if validation_level:
        try:
            config.default_validation_level = ValidationLevel(validation_level)
        except ValueError:
            print(f"Warning: Invalid validation level: {validation_level}")
    
    return config


# Validation of configurations
def validate_config(config: ResilienceConfig) -> List[str]:
    """
    Validate configuration and return list of issues.
    
    Args:
        config: Configuration to validate
    
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Validate thresholds
    if config.default_failure_threshold < 1:
        issues.append("Failure threshold must be at least 1")
    
    if config.default_failure_threshold > 100:
        issues.append("Failure threshold should not exceed 100")
    
    # Validate timeouts
    if config.default_timeout < 1.0:
        issues.append("Default timeout must be at least 1 second")
    
    if config.default_timeout > 3600:
        issues.append("Default timeout should not exceed 1 hour")
    
    # Validate retry counts
    if config.default_max_retries < 0:
        issues.append("Max retries cannot be negative")
    
    if config.default_max_retries > 20:
        issues.append("Max retries should not exceed 20")
    
    # Validate health check interval
    if config.health_check_interval < 5:
        issues.append("Health check interval should be at least 5 seconds")
    
    if config.health_check_interval > 3600:
        issues.append("Health check interval should not exceed 1 hour")
    
    # Validate retention periods
    if config.audit_retention_days < 1:
        issues.append("Audit retention must be at least 1 day")
    
    if config.audit_retention_days > 3653:  # 10 years
        issues.append("Audit retention should not exceed 10 years")
    
    # Validate thresholds consistency
    if config.patient_safety_mode:
        if config.default_failure_threshold > 3:
            issues.append("Patient safety mode should have lower failure thresholds")
        
        if config.default_max_retries > 2:
            issues.append("Patient safety mode should have fewer retries")
    
    return issues


# Configuration profiles for different organizations
def get_hospital_config(
    size: str = "medium",  # small, medium, large
    specialty: str = "general"  # general, emergency, research, academic
) -> ResilienceConfig:
    """
    Get configuration optimized for hospital environments.
    """
    base_config = get_routine_clinic_config()
    
    # Adjust based on hospital size
    if size == "small":
        base_config.default_failure_threshold = 3
        base_config.default_max_retries = 2
        base_config.health_check_interval = 120
    elif size == "large":
        base_config.default_failure_threshold = 7
        base_config.default_max_retries = 4
        base_config.health_check_interval = 30
    
    # Adjust based on specialty
    if specialty == "emergency":
        return get_emergency_department_config()
    elif specialty == "research":
        return get_research_system_config()
    elif specialty == "academic":
        base_config.audit_retention_days = 3653  # 10 years for academic records
    
    return base_config


def get_clinic_config(
    practice_type: str = "family_medicine",  # family_medicine, specialist, urgent_care
    patient_volume: str = "moderate"  # low, moderate, high
) -> ResilienceConfig:
    """
    Get configuration optimized for clinic environments.
    """
    base_config = get_routine_clinic_config()
    
    # Adjust based on practice type
    if practice_type == "urgent_care":
        base_config.default_timeout = 60.0
        base_config.health_check_interval = 45
    elif practice_type == "specialist":
        base_config.default_max_retries = 4
        base_config.validation_enabled = True
    
    # Adjust based on patient volume
    if patient_volume == "high":
        base_config.default_failure_threshold = 8
        base_config.performance_threshold = 0.85
    elif patient_volume == "low":
        base_config.default_failure_threshold = 3
        base_config.performance_threshold = 0.75
    
    return base_config


# Example configuration usage
if __name__ == "__main__":
    import asyncio
    
    async def demonstrate_configurations():
        """Demonstrate different configurations."""
        print("Medical AI Resilience Configurations")
        print("=" * 50)
        
        # Critical care configuration
        critical_config = get_critical_care_config()
        print(f"Critical Care Config - Patient Safety Mode: {critical_config.patient_safety_mode}")
        print(f"  Failure Threshold: {critical_config.default_failure_threshold}")
        print(f"  Max Retries: {critical_config.default_max_retries}")
        print(f"  Health Check Interval: {critical_config.health_check_interval}s")
        
        # Validate configuration
        issues = validate_config(critical_config)
        if issues:
            print(f"  Validation Issues: {issues}")
        else:
            print("  ✓ Configuration Valid")
        
        print()
        
        # Emergency department configuration
        emergency_config = get_emergency_department_config()
        print(f"Emergency Department Config - Patient Safety Mode: {emergency_config.patient_safety_mode}")
        print(f"  Failure Threshold: {emergency_config.default_failure_threshold}")
        print(f"  Max Retries: {emergency_config.default_max_retries}")
        print(f"  Health Check Interval: {emergency_config.health_check_interval}s")
        
        print()
        
        # Research system configuration
        research_config = get_research_system_config()
        print(f"Research System Config - Patient Safety Mode: {research_config.patient_safety_mode}")
        print(f"  Failure Threshold: {research_config.default_failure_threshold}")
        print(f"  Max Retries: {research_config.default_max_retries}")
        print(f"  Health Check Interval: {research_config.health_check_interval}s")
        
        print()
        
        # Custom configuration
        custom_config = get_custom_config(
            environment="production",
            compliance_level="critical",
            performance_priority="reliability"
        )
        print(f"Custom Config - Patient Safety Mode: {custom_config.patient_safety_mode}")
        print(f"  Environment: production")
        print(f"  Compliance: critical")
        print(f"  Priority: reliability")
    
    # Run demonstration
    asyncio.run(demonstrate_configurations())