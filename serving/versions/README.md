# Model Version Tracking System

Enterprise-grade model lifecycle management with medical compliance, audit trails, and production safety mechanisms for Phase 6 of the Medical AI Assistant.

## Overview

This comprehensive model version tracking system provides end-to-end management of AI model versions with specialized focus on medical compliance, regulatory requirements, and production safety. It integrates semantic versioning, external registry support, compatibility checking, A/B testing, deployment management, performance comparison, and comprehensive documentation tracking.

## 🏗️ Architecture

```
serving/versions/
├── __init__.py                 # Main package initialization
├── core.py                     # Core versioning classes and registry
├── registry.py                 # MLflow/W&B integration adapters
├── compatibility.py            # Version compatibility checking
├── testing.py                  # A/B testing infrastructure
├── deployment.py               # Rollout and rollback mechanisms
├── comparison.py               # Performance comparison utilities
├── metadata.py                 # Metadata and documentation tracking
└── example_usage.py            # Comprehensive demonstration
```

## 🚀 Key Features

### 1. Semantic Versioning with Medical Compliance

- **Semantic Versioning**: Standard major.minor.patch versioning
- **Medical Device Classification**: Class I, II, III device tracking
- **Clinical Approval**: Regulatory approval date and authority tracking
- **Compliance Levels**: Unknown → Pre-clinical → Clinical Investigation → Clinical Validation → Production
- **Audit Trails**: Comprehensive change tracking with user attribution

### 2. Model Registry Integration

- **MLflow Integration**: Full MLflow tracking server support
- **Weights & Biases Integration**: W&B experiment and artifact management
- **Multi-Registry Support**: Simultaneous sync across multiple registries
- **Metadata Synchronization**: Automatic metadata propagation
- **Experiment Management**: A/B testing experiment creation and tracking

### 3. Version Compatibility Checking

- **Backward Compatibility**: Ensures existing clients continue working
- **Forward Compatibility**: Validates upgrade paths
- **Breaking Change Detection**: Identifies API and data format changes
- **Dependency Analysis**: Framework and library version compatibility
- **Migration Planning**: Automated migration notes and rollback plans

### 4. A/B Testing Infrastructure

- **Experiment Configuration**: Flexible experiment setup with statistical parameters
- **Traffic Allocation**: Configurable traffic splitting (50/50, 80/20, etc.)
- **Statistical Testing**: T-test, Chi-square, Fisher exact, Bayesian analysis
- **Safety Monitoring**: Real-time health monitoring with automated stopping
- **Clinical Validation**: Medical-specific A/B testing for clinical outcomes

### 5. Deployment Management

- **Blue-Green Deployment**: Zero-downtime deployment strategy
- **Canary Deployment**: Gradual rollout with safety monitoring
- **Rolling Deployment**: Incremental deployment across infrastructure
- **Health Checks**: Automated latency, accuracy, and error rate monitoring
- **Automated Rollback**: Safety-triggered rollback based on health metrics

### 6. Performance Comparison

- **Medical Accuracy Metrics**: Sensitivity, specificity, PPV, NPV, clinical accuracy
- **Statistical Significance**: Confidence intervals, p-values, effect sizes
- **Clinical Outcome Analysis**: Medical-specific performance evaluation
- **Regulatory Compliance**: FDA/EMA requirement validation
- **Performance Thresholds**: Clinical threshold compliance checking

### 7. Documentation & Compliance

- **Clinical Validation Records**: Prospective/retrospective validation tracking
- **Regulatory Documents**: FDA submissions, approval documents
- **Compliance Reporting**: Automated compliance score calculation
- **Audit Scheduling**: Regular compliance audit management
- **Change Control**: Version change approval workflows

## 📊 Medical Compliance Features

### Clinical Validation Tracking
- Pre-clinical, clinical investigation, clinical validation phases
- IRB approval tracking
- Principal investigator assignment
- Sample size and endpoint management
- Statistical analysis results storage

### Regulatory Compliance
- Medical device classification (Class I, II, III)
- FDA/EMA submission tracking
- Regulatory approval status monitoring
- Adverse event reporting integration
- Post-market surveillance data

### Risk Management
- Risk assessment documentation
- Safety profile tracking
- Clinical benefit documentation
- Limitation and warning management
- Lifecycle risk evaluation

## 🔧 Installation & Setup

```python
# Install dependencies (add to requirements.txt)
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
mlflow>=1.20.0  # Optional
wandb>=0.12.0   # Optional
requests>=2.25.0
```

```python
# Basic setup
from serving.versions import (
    VersionManager, VersionRegistry, CompatibilityChecker,
    ABTestingManager, RolloutManager, RollbackManager,
    PerformanceComparator, MetadataManager
)

# Initialize core components
registry = VersionRegistry("/path/to/registry")
version_manager = VersionManager(registry)
compatibility_checker = CompatibilityChecker()
ab_testing = ABTestingManager()
rollout_manager = RolloutManager(version_manager, ab_testing)
metadata_manager = MetadataManager("/path/to/metadata")
```

## 🎯 Core Usage Examples

### 1. Creating and Managing Versions

```python
from serving.versions.core import ModelVersion, VersionType, ComplianceLevel

# Create medical AI model version
version = ModelVersion(
    version="1.0.0",
    model_name="diagnosis_model",
    model_type="neural_network",
    description="Initial FDA-approved diagnosis model",
    created_by="dr.smith",
    version_type=VersionType.MAJOR,
    framework_version="pytorch 1.12.0"
)

# Set medical compliance
version.compliance.compliance_level = ComplianceLevel.PRODUCTION
version.compliance.clinical_approval_date = datetime.now() - timedelta(days=30)
version.compliance.approval_authority = "FDA"
version.compliance.medical_device_class = "Class II"
version.compliance.intended_use = "Assist in medical diagnosis"

# Register version
registry.register_version(version)
```

### 2. Compatibility Checking

```python
from serving.versions.compatibility import CompatibilityType

# Check version compatibility
source_version = registry.get_version("diagnosis_model", "1.0.0")
target_version = registry.get_version("diagnosis_model", "1.1.0")

compatibility = compatibility_checker.check_compatibility(
    source_version, target_version,
    check_types=[CompatibilityType.FULL]
)

print(f"Compatibility: {compatibility.overall_compatibility.value}")
print(f"Migration notes: {compatibility.migration_notes}")
print(f"Rollback plan: {compatibility.rollback_plan}")
```

### 3. A/B Testing Setup

```python
from serving.versions.testing import ExperimentConfig, TestType, StatisticalTest

# Create A/B test experiment
experiment_config = ExperimentConfig(
    name="Accuracy Improvement Test",
    description="Test v1.1.0 accuracy improvements",
    model_name="diagnosis_model",
    control_version="1.0.0", 
    treatment_version="1.1.0",
    test_type=TestType.MODEL_PERFORMANCE,
    statistical_test=StatisticalTest.T_TEST,
    minimum_sample_size=1000,
    significance_level=0.05,
    requires_clinical_approval=True
)

# Create and start experiment
experiment_id = ab_testing.create_experiment(experiment_config)
ab_testing.start_experiment(experiment_id)

# Assign users to groups
assignment = ab_testing.assign_user_to_group(experiment_id, "user_123")
```

### 4. Deployment Management

```python
from serving.versions.deployment import DeploymentTarget, DeploymentType

# Setup deployment targets
staging_target = DeploymentTarget(
    name="staging_server",
    url="https://staging.medical-ai.example.com",
    environment="staging"
)

production_target = DeploymentTarget(
    name="prod_server", 
    url="https://api.medical-ai.example.com",
    environment="production"
)

# Deploy with canary strategy
deployment_id = rollout_manager.deploy_model(
    model_name="diagnosis_model",
    version="1.1.0", 
    targets=[staging_target, production_target],
    deployment_type=DeploymentType.CANARY,
    rollout_percentage=10.0,
    initiated_by="dr.smith"
)
```

### 5. Performance Comparison

```python
from serving.versions.comparison import MedicalMetrics, PerformanceComparator

# Compare model performance
medical_metrics = MedicalMetrics()
comparator = PerformanceComparator(medical_metrics)

comparison_result = comparator.compare_models(
    control_version=v1_0_0,
    treatment_version=v1_1_0,
    test_data=test_data,
    statistical_test="t_test"
)

print(f"Overall improvement: {comparison_result.overall_improvement:.1%}")
print(f"Recommendation: {comparison_result.recommendation}")
```

### 6. Metadata & Documentation

```python
from serving.versions.metadata import ClinicalValidation, ValidationType

# Create clinical validation
validation = ClinicalValidation(
    validation_id="VAL001",
    validation_type=ValidationType.CLINICAL_VALIDATION,
    validation_name="Accuracy Validation Study",
    start_date=datetime.now() - timedelta(days=60),
    end_date=datetime.now() - timedelta(days=30),
    sample_size=500,
    validation_status=ValidationStatus.PASSED
)

# Create metadata
metadata = metadata_manager.create_metadata(version, "dr.smith")
metadata_manager.add_clinical_validation(metadata.version_id, validation)

# Validate compliance
compliance_result = metadata_manager.validate_regulatory_compliance(metadata.version_id)
```

## 🏥 Medical-Specific Features

### Clinical Validation Workflow

1. **Pre-clinical Testing**: Initial model validation
2. **Clinical Investigation**: IRB-approved clinical study
3. **Clinical Validation**: Prospective validation study
4. **Regulatory Submission**: FDA/EMA submission preparation
5. **Production Approval**: Regulatory approval and monitoring

### Performance Thresholds

| Metric | Minimum Threshold | Clinical Significance |
|--------|------------------|----------------------|
| Medical Accuracy | 85% | Critical for safety |
| Diagnostic Accuracy | 90% | High diagnostic confidence |
| Clinical Sensitivity | 95% | Disease detection critical |
| Clinical Specificity | 90% | Reduce false positives |
| AUC-ROC | 85% | Good discriminative ability |
| Latency | 1000ms | Acceptable response time |
| Error Rate | 1% | Low error rate for safety |

### Compliance Validation

```python
# Automated compliance checking
validation_result = medical_metrics.evaluate_against_clinical_thresholds(metrics)

for threshold in clinical_thresholds:
    if not threshold.is_met(actual_value):
        if threshold.regulatory_requirement:
            compliance_issues.append(f"Regulatory violation: {threshold.metric_name}")
        else:
            recommendations.append(f"Consider improvement: {threshold.metric_name}")
```

## 🚦 Production Safety Mechanisms

### Health Check Monitoring

```python
# Configure health checks
health_checks = [
    HealthCheckConfig("latency", "latency", 1000.0, failure_threshold=3),
    HealthCheckConfig("error_rate", "error_rate", 0.05, failure_threshold=2),
    HealthCheckConfig("accuracy", "accuracy", 0.85, failure_threshold=3)
]

# Automated rollback on health degradation
if health_score < rollback_threshold:
    rollback_manager.rollback_deployment(
        deployment_id=deployment_id,
        reason=f"Health score {health_score:.2f} below threshold"
    )
```

### Statistical Safety

- **Minimum Sample Size**: Ensures statistical power
- **Significance Testing**: Validates performance improvements
- **Confidence Intervals**: Quantifies uncertainty
- **Effect Size Analysis**: Clinical significance assessment

### Audit Trail

Every operation is logged with:
- User attribution
- Timestamp
- Change details
- Compliance implications
- Regulatory impact

## 📈 Metrics & Monitoring

### Performance Metrics

- **Clinical Accuracy**: Overall diagnostic accuracy
- **Sensitivity/Specificity**: True positive/negative rates
- **PPV/NPV**: Predictive values
- **AUC-ROC**: Discriminative performance
- **Latency**: Response time monitoring
- **Throughput**: Query processing capacity
- **Error Rate**: Failure rate tracking

### Compliance Metrics

- **Regulatory Compliance Score**: 0.0-1.0 scale
- **Documentation Completeness**: Required document tracking
- **Clinical Validation Status**: Validation phase tracking
- **Audit Trail Coverage**: Change documentation
- **Risk Assessment Score**: Risk evaluation metrics

## 🔒 Security & Compliance

### Data Protection

- PHI (Protected Health Information) handling
- Audit trail encryption
- Secure storage of clinical data
- Access control and authentication

### Regulatory Compliance

- FDA 21 CFR Part 820 (Quality System Regulation)
- ISO 13485 (Medical Device Quality Management)
- IEC 62304 (Medical Device Software)
- HIPAA compliance for health data

### Audit & Documentation

- Comprehensive change logs
- Regulatory submission tracking
- Clinical validation documentation
- Risk assessment records
- Quality management records

## 🛠️ Configuration

### Environment Variables

```bash
# MLflow configuration
MLFLOW_TRACKING_URI=https://mlflow.company.com
MLFLOW_REGISTRY_URI=https://mlflow.company.com

# W&B configuration  
WANDB_API_KEY=your_api_key
WANDB_PROJECT=medical-ai-models

# Database configuration
DATABASE_URL=postgresql://user:pass@localhost/medical_ai

# Security configuration
AUDIT_LOG_ENCRYPTION_KEY=your_encryption_key
```

### Registry Configuration

```python
# MLflow Registry Setup
mlflow_registry = MLflowRegistry(tracking_uri="https://mlflow.company.com")

# W&B Registry Setup  
wandb_registry = WandbRegistry(project="medical-ai-models")

# Register with manager
registry_manager = RegistryManager()
registry_manager.register_adapter("mlflow", mlflow_registry)
registry_manager.register_adapter("wandb", wandb_registry)

# Connect to all registries
credentials = {
    "mlflow": {"tracking_uri": "https://mlflow.company.com"},
    "wandb": {"project": "medical-ai-models"}
}
registry_manager.connect_all(credentials)
```

## 🧪 Testing & Validation

### Unit Testing

```python
import pytest
from serving.versions import VersionManager, ModelVersion

def test_version_creation():
    version = ModelVersion(
        version="1.0.0",
        model_name="test_model",
        model_type="neural_network"
    )
    assert version.version == "1.0.0"
    assert version.model_name == "test_model"

def test_compatibility_checking():
    # Test compatibility validation
    compatibility = checker.check_compatibility(v1_0_0, v1_1_0)
    assert compatibility.overall_compatibility != CompatibilityLevel.INCOMPATIBLE
```

### Integration Testing

```python
def test_deployment_pipeline():
    # Test full deployment pipeline
    deployment_id = rollout_manager.deploy_model(
        "test_model", "1.0.0", [target], DeploymentType.CANARY
    )
    
    # Wait for deployment completion
    time.sleep(10)
    
    deployment = rollout_manager.get_deployment_status(deployment_id)
    assert deployment.status == DeploymentStatus.SUCCESS
```

## 📚 API Reference

### Core Classes

- **ModelVersion**: Core version representation with medical compliance
- **VersionRegistry**: Internal registry for version management
- **VersionManager**: Operations for version lifecycle management
- **CompatibilityChecker**: Version compatibility analysis
- **ABTestingManager**: A/B testing experiment management
- **RolloutManager**: Model deployment and rollout management
- **RollbackManager**: Automated rollback operations
- **PerformanceComparator**: Performance comparison utilities
- **MetadataManager**: Documentation and compliance management

### Key Methods

```python
# Version Management
registry.register_version(version)
registry.get_version(model_name, version)
registry.list_versions(model_name)

# Compatibility
compatibility = checker.check_compatibility(source, target)
rollback_safety = checker.validate_rollback_safety(current, target)

# A/B Testing
experiment_id = ab_testing.create_experiment(config)
ab_testing.start_experiment(experiment_id)
assignment = ab_testing.assign_user_to_group(experiment_id, user_id)

# Deployment
deployment_id = rollout_manager.deploy_model(model, version, targets)
status = rollout_manager.get_deployment_status(deployment_id)
rollback_id = rollback_manager.rollback_deployment(deployment_id)

# Performance Comparison
comparison = comparator.compare_models(control, treatment, test_data)
report = comparator.generate_performance_report(comparison)

# Metadata
metadata = metadata_manager.create_metadata(version)
metadata_manager.add_clinical_validation(metadata_id, validation)
compliance = metadata_manager.validate_regulatory_compliance(metadata_id)
```

## 🤝 Contributing

1. Follow medical device software development standards
2. Ensure comprehensive testing for all new features
3. Maintain backward compatibility where possible
4. Document all regulatory compliance changes
5. Update audit trails for all modifications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For technical support or questions about medical compliance:
- Create an issue in the repository
- Contact the Medical AI compliance team
- Refer to the documentation wiki

---

**⚠️ Medical Device Disclaimer**: This software is designed for use in medical device applications. Users are responsible for ensuring compliance with all applicable regulatory requirements including FDA, EMA, and other regulatory bodies. Always validate performance in your specific medical use case before clinical deployment.