# Model Registry System with Versioning

A comprehensive machine learning model registry with version control, performance tracking, and integration with popular ML platforms like MLflow and Weights & Biases.

## Features

### 🏗️ Core Registry Management
- **Model Metadata Storage**: Complete model metadata with version control
- **Artifact Management**: Secure storage and retrieval of model files
- **Performance Metrics**: Track accuracy, F1-score, precision, recall, and custom metrics
- **Model Lineage**: Track parent models and model evolution

### 🔢 Advanced Versioning
- **Semantic Versioning**: Standard major.minor.patch versioning
- **Git Integration**: Automatic versioning based on git state
- **Version Comparison**: Compare versions and assess compatibility
- **Change Tracking**: Track all model changes with metadata

### 🚀 Model Lifecycle Management
- **Multi-Stage Promotion**: Development → Staging → Production
- **Automated Validation**: Requirements-based model promotion
- **A/B Testing Support**: Built-in A/B testing workflow
- **Rollback Capabilities**: Easy rollback to previous versions

### 🔗 Platform Integration
- **MLflow Integration**: Automatic logging to MLflow
- **Weights & Biases**: Optional wandb integration
- **Multiple Storage Backends**: Local, S3, GCS, Azure support

### 🛠️ Developer Tools
- **CLI Interface**: Comprehensive command-line management
- **Python API**: Easy integration in Python code
- **REST API**: Programmatic access (optional)
- **Monitoring**: Real-time registry monitoring

## Installation

### Prerequisites

```bash
# Python 3.8+
python --version

# Required packages
pip install numpy scikit-learn pandas joblib pyyaml
```

### Optional Dependencies

```bash
# For MLflow integration
pip install mlflow

# For Weights & Biases integration
pip install wandb

# For advanced features
pip install sqlite3-postgres  # For PostgreSQL support
pip install boto3            # For S3 storage
pip install google-cloud-storage  # For GCS storage
```

### Setup

1. **Clone or copy the training directory**:
   ```bash
   cp -r training/ /path/to/your/project/
   cd /path/to/your/project/training
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the registry**:
   ```bash
   cp configs/registry_config.yaml configs/my_registry_config.yaml
   # Edit my_registry_config.yaml with your settings
   ```

## Quick Start

### 1. Basic Model Registration

```python
from utils.model_registry import ModelRegistry, register_sklearn_model

# Initialize registry
registry = ModelRegistry("./model_registry")

# Create and train a model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=10, random_state=42)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Register model
model_id = registry.register_model(
    model=model,
    name="my_model",
    version="1.0.0",
    description="My first registered model",
    performance_metrics={"accuracy": 0.85, "f1_score": 0.82}
)

print(f"Model registered with ID: {model_id}")
```

### 2. Using Convenience Functions

```python
# Register sklearn model
model_id = register_sklearn_model(
    model=model,
    name="sklearn_model",
    registry_path="./model_registry",
    description="Sklearn model example",
    performance_metrics={"accuracy": 0.90}
)

# Get best performing model
best_model_id = get_best_model(
    name="sklearn_model",
    metric="accuracy",
    registry_path="./model_registry"
)
```

### 3. Model Lifecycle Management

```python
from utils.model_registry import ModelStage

# Promote model to staging
registry.promote_model(
    model_id,
    ModelStage.STAGING,
    requirements={"min_accuracy": 0.80}
)

# Promote to production
registry.promote_model(
    model_id,
    ModelStage.PRODUCTION,
    requirements={"min_accuracy": 0.80, "min_f1": 0.75}
)

# Load and use the model
loaded_model = registry.load_model(model_id)
predictions = loaded_model.predict(X_new)
```

### 4. Model Comparison

```python
# Register two models
id1 = registry.register_model(model1, name="model_a")
id2 = registry.register_model(model2, name="model_b")

# Compare models
comparison = registry.compare_models(id1, id2)
print(f"Winner: {comparison.winner}")
print(f"Confidence: {comparison.confidence_score:.3f}")

# Compare specific metrics
comparison = registry.compare_models(
    id1, id2, 
    metrics_to_compare=["accuracy", "f1_score"]
)
```

## CLI Usage

The model registry includes a comprehensive CLI interface:

### Register a Model

```bash
python scripts/manage_models.py register \
    --model-path ./model.joblib \
    --name fraud_detector \
    --version 1.0.0 \
    --stage development \
    --description "Model for detecting fraudulent transactions" \
    --metrics '{"accuracy": 0.92, "f1_score": 0.89}'
```

### List and Search Models

```bash
# List all models
python scripts/manage_models.py list

# List production models
python scripts/manage_models.py list --stage production

# Search models
python scripts/manage_models.py search "fraud"

# Export registry
python scripts/manage_models.py export ./registry_export.json
```

### Model Management

```bash
# Promote model
python scripts/manage_models.py promote model-id-123 \
    --target-stage production \
    --requirements ./requirements.json

# Compare models
python scripts/manage_models.py compare model-a-id model-b-id

# Rollback model
python scripts/manage_models.py rollback model-id-123

# Bulk register models
python scripts/manage_models.py bulk-register ./models_config.json
```

### Monitoring

```bash
# Monitor registry
python scripts/manage_models.py monitor --interval 30

# Get statistics
python scripts/manage_models.py stats
```

## Configuration

The registry uses a YAML configuration file (`configs/registry_config.yaml`):

### Basic Configuration

```yaml
# Registry settings
registry:
  path: "./model_registry"
  default_stage: "development"

# Storage settings
storage:
  backend: "local"
  local:
    compression: true

# Versioning settings
versioning:
  strategy: "semantic"
  git:
    enabled: true

# MLflow integration
mlflow:
  enabled: true
  tracking_uri: "sqlite:///mlflow.db"

# Wandb integration
wandb:
  enabled: true
  project: "my-ml-registry"
```

### Advanced Configuration

See `configs/registry_config.yaml` for complete configuration options including:
- Database settings (SQLite, PostgreSQL, MySQL)
- Cloud storage (S3, GCS, Azure)
- Security and access control
- Notification settings
- Monitoring and alerting

## Versioning System

### Semantic Versioning

The system uses semantic versioning (MAJOR.MINOR.PATCH):

```python
from utils.versioning import VersionTracker

tracker = VersionTracker()

# Parse versions
v1 = tracker.parse_version("1.2.3")
v2 = tracker.parse_version("1.2.4")

# Compare versions
assert v1 < v2  # True

# Increment versions
next_patch = tracker.increment_version("1.2.3", "patch")  # "1.2.4"
next_minor = tracker.increment_version("1.2.3", "minor")  # "1.3.0"
next_major = tracker.increment_version("1.2.3", "major")  # "2.0.0"
```

### Git Integration

```python
# Generate version from git state
version = tracker.generate_version_from_git("patch")
# Creates: "1.2.4+abc123.dev"

# Create git tag
tracker.create_version_tag("model-v1.2.3", push=True)
```

## Model Promotion Workflow

### Development → Staging

```python
# Register model
model_id = registry.register_model(
    model, name="my_model", version="1.0.0"
)

# Promote to staging with requirements
registry.promote_model(
    model_id,
    ModelStage.STAGING,
    requirements={
        "min_accuracy": 0.80,
        "validation_required": True
    }
)
```

### Staging → Production

```python
# Promote to production
registry.promote_model(
    model_id,
    ModelStage.PRODUCTION,
    requirements={
        "min_accuracy": 0.85,
        "min_f1": 0.80,
        "tests_passed": True,
        "documentation_complete": True
    }
)
```

### A/B Testing

```python
# Two models in production
model_a_id = registry.get_latest_version("model_a")
model_b_id = registry.get_latest_version("model_b")

# Set up A/B test (80% traffic to A, 20% to B)
# After test period, compare performance
comparison = registry.compare_models(model_a_id, model_b_id)

# Promote winner
if comparison.winner == model_b_id:
    registry.promote_model(model_b_id, ModelStage.PRODUCTION)
```

## Performance Tracking

### Custom Metrics

```python
registry.register_model(
    model,
    name="my_model",
    performance_metrics={
        "accuracy": 0.92,
        "precision": 0.89,
        "recall": 0.94,
        "f1_score": 0.91,
        "custom_metric": 0.87
    }
)
```

### Update Metrics

```python
# Update metrics after retraining
new_metrics = {
    "accuracy": 0.94,
    "f1_score": 0.93
}
registry.update_model_metrics(model_id, new_metrics)
```

### Track Metrics Over Time

```python
# Get model lineage
lineage = registry.get_model_lineage(model_id)

# Compare versions
for prev_model in lineage[:-1]:
    comparison = registry.compare_models(prev_model.model_id, model_id)
    print(f"Improvement: {comparison.confidence_score:.3f}")
```

## Integration Examples

### MLflow Integration

```python
# Automatically logs to MLflow when enabled
model_id = registry.register_model(
    model, name="mlflow_model", version="1.0.0"
)

# Access MLflow experiment
import mlflow
experiment = mlflow.get_experiment_by_name("model_registry")
```

### Weights & Biases Integration

```python
# Automatically logs to wandb when enabled
model_id = registry.register_model(
    model, name="wandb_model", version="1.0.0"
)

# View in wandb dashboard
# Models appear as artifacts in your project
```

## Testing

### Run All Tests

```bash
python scripts/run_model_registry_tests.py
```

### Run Specific Test Types

```bash
# Unit tests only
python scripts/run_model_registry_tests.py --test-type unit

# Functional tests only
python scripts/run_model_registry_tests.py --test-type functional

# Performance tests
python scripts/run_model_registry_tests.py --test-type performance
```

### Example and Validation

```bash
# Run complete example
python examples/model_registry_example.py

# With custom registry path
python examples/model_registry_example.py --registry-path ./my_registry

# Clean registry before running
python examples/model_registry_example.py --clean
```

### Output Test Results

```bash
# Save results to JSON
python scripts/run_model_registry_tests.py --output test_results.json
```

## Best Practices

### 1. Model Naming

```python
# Good naming conventions
model_id = registry.register_model(
    model, 
    name="fraud_detection_v1",  # Descriptive + version
    description="RFC model for credit card fraud detection"
)

# Avoid generic names
# model_id = registry.register_model(model, name="my_model")  # Bad
```

### 2. Version Management

```python
# Use semantic versioning
model_id = registry.register_model(
    model, name="model", version="1.0.0"
)

# Or let system auto-generate
model_id = registry.register_model(
    model, name="model"  # Gets 0.1.0
)
```

### 3. Metadata Documentation

```python
# Include comprehensive metadata
model_id = registry.register_model(
    model,
    name="medical_diagnosis",
    description="AI model for medical diagnosis",
    tags={
        "domain": "healthcare",
        "data_type": "medical_images",
        "confidentiality": "phi_compliant"
    },
    hyperparams={
        "learning_rate": 0.001,
        "batch_size": 32
    },
    data_lineage={
        "dataset": "medical_images_v2",
        "training_date": "2024-01-15",
        "preprocessing": "normalized_augmented"
    }
)
```

### 4. Performance Thresholds

```python
# Set appropriate thresholds for promotion
registry.promote_model(
    model_id,
    ModelStage.PRODUCTION,
    requirements={
        "min_accuracy": 0.90,    # High bar for production
        "min_f1": 0.85,
        "min_precision": 0.88,   # Important for medical AI
        "min_recall": 0.87
    }
)
```

### 5. Regular Monitoring

```bash
# Monitor registry health
python scripts/manage_models.py monitor --interval 60

# Check statistics
python scripts/manage_models.py stats

# Export registry regularly
python scripts/manage_models.py export backup_$(date +%Y%m%d).json
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Install required dependencies
   pip install scikit-learn numpy pandas joblib pyyaml
   ```

2. **Permission Errors**:
   ```bash
   # Check registry path permissions
   chmod 755 ./model_registry
   ```

3. **Database Locked**:
   ```python
   # Ensure proper connection handling
   with registry._get_db_connection() as conn:
       # Your database operations
   ```

4. **Model Loading Fails**:
   ```python
   # Check artifact path exists
   metadata = registry.get_model(model_id)
   if metadata.artifact_path:
       print(f"Artifact: {metadata.artifact_path}")
   ```

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check registry state
stats = registry.get_registry_stats()
print(f"Total models: {stats['total_models']}")

# List all models with details
models = registry.list_models(limit=10)
for model in models:
    print(f"{model.name} v{model.version} - {model.stage.value}")
```

## API Reference

### ModelRegistry Class

#### Core Methods

- `register_model(model, **kwargs)`: Register a new model
- `get_model(model_id)`: Retrieve model metadata
- `load_model(model_id)`: Load model from registry
- `list_models(**filters)`: List models with filters
- `update_model_stage(model_id, stage)`: Update model stage
- `update_model_metrics(model_id, metrics)`: Update performance metrics

#### Lifecycle Methods

- `promote_model(model_id, target_stage, requirements)`: Promote model
- `rollback_model(model_id, target_version)`: Rollback to previous version
- `archive_model(model_id)`: Archive model
- `delete_model(model_id, force)`: Delete model

#### Analysis Methods

- `compare_models(model_a_id, model_b_id, metrics)`: Compare models
- `search_models(query)`: Search for models
- `get_model_lineage(model_id)`: Get model lineage
- `get_registry_stats()`: Get statistics
- `export_registry(output_path)`: Export registry data

### VersionTracker Class

- `parse_version(version)`: Parse semantic version
- `increment_version(current, increment_type)`: Generate next version
- `generate_version_from_git(increment_type)`: Generate from git state
- `compare_versions(v1, v2)`: Compare versions
- `sort_versions(versions)`: Sort versions

## Contributing

1. **Add Tests**: All new features must include tests
2. **Update Documentation**: Update README and docstrings
3. **Follow Standards**: Use semantic versioning and proper naming
4. **Test Integration**: Ensure MLflow/wandb integration works

### Adding New Features

1. **Add to ModelRegistry**: Extend the main class
2. **Update CLI**: Add new commands to manage_models.py
3. **Update Configuration**: Add options to registry_config.yaml
4. **Add Tests**: Comprehensive test coverage
5. **Update Documentation**: Update this README

## License

This model registry system is part of the Medical AI Assistant project.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the examples in `examples/`
3. Run the test suite to identify issues
4. Check the logs for detailed error messages

## Changelog

### Version 1.0.0
- Initial release
- Core registry functionality
- Semantic versioning
- MLflow and wandb integration
- CLI interface
- Comprehensive testing
- Documentation and examples