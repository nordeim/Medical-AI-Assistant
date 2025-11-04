# Medical AI Monitoring System

A comprehensive performance monitoring and metrics system for Medical AI applications, providing real-time inference monitoring, clinical outcome tracking, model drift detection, and regulatory compliance monitoring.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Dashboard Integration](#dashboard-integration)
- [Alerting System](#alerting-system)
- [Clinical Outcomes Tracking](#clinical-outcomes-tracking)
- [Model Drift Detection](#model-drift-detection)
- [Security and Compliance](#security-and-compliance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

The Medical AI Monitoring System is designed to provide comprehensive observability for healthcare AI applications. It integrates with existing serving infrastructure to monitor model performance, track clinical outcomes, detect model drift, and ensure regulatory compliance in medical environments.

### Key Benefits

- **Real-time Monitoring**: Track inference metrics, latency, and throughput in real-time
- **Clinical Effectiveness**: Monitor diagnostic accuracy and treatment success rates
- **Model Drift Detection**: Early detection of model performance degradation
- **Regulatory Compliance**: HIPAA-compliant audit logging and PHI protection
- **Automated Alerting**: Multi-channel notifications with configurable thresholds
- **Grafana Integration**: Professional dashboards for visualization and analysis

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Medical AI Monitoring System                 │
├─────────────────────────────────────────────────────────────────┤
│  Metrics Collection Layer                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ Inference       │ │ Memory          │ │ Model           │    │
│  │ Metrics         │ │ Monitoring      │ │ Accuracy        │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Tracking Layer                                                 │
│  ┌─────────────────┐ ┌─────────────────┐                       │
│  │ Drift           │ │ Clinical        │                       │
│  │ Detection       │ │ Outcomes        │                       │
│  └─────────────────┘ └─────────────────┘                       │
├─────────────────────────────────────────────────────────────────┤
│  Alerting Layer                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ Alert Manager   │ │ Notification    │ │ Escalation      │    │
│  │                 │ │ System          │ │ Manager         │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│  Visualization Layer                                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │ Grafana         │ │ Prometheus      │ │ Custom          │    │
│  │ Dashboards      │ │ Metrics         │ │ APIs            │    │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Components

1. **Metrics Collector** (`metrics/collector.py`)
   - Real-time inference metrics collection
   - Latency and throughput monitoring
   - Memory usage tracking (CPU/GPU)
   - System resource monitoring

2. **Prometheus Integration** (`metrics/prometheus.py`)
   - Prometheus metrics export
   - Custom medical AI metrics
   - Time-series data management

3. **Drift Detector** (`tracking/drift_detector.py`)
   - Statistical drift detection (KS test, PSI, Chi-square)
   - Model performance monitoring
   - Quality metrics tracking

4. **Clinical Outcomes Tracker** (`tracking/clinical_outcomes.py`)
   - Patient outcome tracking
   - Diagnostic accuracy monitoring
   - Treatment effectiveness measurement
   - Adverse event detection

5. **Alert Manager** (`alerts/alert_manager.py`)
   - Configurable alert rules
   - Severity-based escalation
   - Alert lifecycle management

6. **Notification System** (`alerts/notification_system.py`)
   - Multi-channel notifications (Email, Slack, PagerDuty, Webhooks)
   - Delivery tracking and retry logic
   - Rate limiting and throttling

7. **Grafana Integration** (`dashboards/grafana_integration.py`)
   - Automated dashboard provisioning
   - Medical AI-specific visualizations
   - Real-time monitoring panels

## Features

### 🔍 Real-time Inference Monitoring
- Request count and throughput tracking
- Latency percentiles (P50, P95, P99)
- Error rate monitoring
- Model response time tracking

### 📊 Performance Metrics
- CPU and GPU memory utilization
- System resource consumption
- Model loading and inference times
- Queue depth and processing rates

### 🏥 Clinical Outcomes Tracking
- Diagnostic accuracy measurement
- Treatment effectiveness monitoring
- Patient outcome analysis
- Adverse event detection and reporting

### 📈 Model Drift Detection
- Statistical drift detection algorithms
- Performance degradation alerts
- Model quality monitoring
- Reference data comparison

### 🚨 Advanced Alerting
- Configurable thresholds and conditions
- Multi-channel notification delivery
- Escalation policies
- Alert suppression and grouping

### 📱 Grafana Dashboards
- Pre-configured medical AI dashboards
- Real-time visualization
- Clinical metrics panels
- SLA tracking displays

### 🔒 Security & Compliance
- HIPAA-compliant audit logging
- PHI protection and anonymization
- Data encryption and access control
- Regulatory compliance monitoring

## Installation

### Prerequisites

- Python 3.8 or higher
- PostgreSQL 12+ (for clinical outcomes tracking)
- Prometheus (for metrics collection)
- Grafana (for visualization)

### Install Dependencies

```bash
cd serving/monitoring
pip install -r requirements.txt
```

### System Dependencies

For GPU monitoring, install NVIDIA drivers and CUDA toolkit:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-utils-470  # Adjust version as needed

# CentOS/RHEL
sudo yum install nvidia-utils
```

## Configuration

### 1. Environment Variables

Create a `.env` file with required environment variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/medical_ai_monitoring

# Notification Channels
EMAIL_PASSWORD=your_email_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PAGERDUTY_INTEGRATION_KEY=your_pagerduty_key

# Grafana Configuration
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=admin_password
GRAFANA_API_KEY=your_api_key

# Security
JWT_SECRET_KEY=your_jwt_secret
ENCRYPTION_KEY=your_encryption_key
```

### 2. Configuration File

Copy and customize the example configuration:

```bash
cp monitoring_config_example.yaml monitoring_config.yaml
```

Edit `monitoring_config.yaml` for your environment:

```yaml
# Basic Configuration
environment: production
database:
  url: "postgresql://user:password@localhost:5432/medical_ai_monitoring"

# SLA Configuration
sla:
  latency_p95: 2000  # milliseconds
  availability: 99.9  # percentage
  error_rate: 0.1    # percentage

# Alert Rules
alerts:
  rules:
    - name: "High Latency"
      condition: "latency_p95 > 2000"
      severity: "warning"
```

## Quick Start

### 1. Basic Setup

```python
from serving.monitoring import (
    MetricsCollector, 
    PrometheusExporter,
    ClinicalOutcomeTracker,
    AlertManager
)

# Initialize monitoring components
config = MonitoringConfig.load("monitoring_config.yaml")
metrics_collector = MetricsCollector(config)
prometheus_exporter = PrometheusExporter(config)
clinical_tracker = ClinicalOutcomeTracker(config)
alert_manager = AlertManager(config)

# Start metrics collection
metrics_collector.start()
prometheus_exporter.start()
clinical_tracker.start()
alert_manager.start()
```

### 2. Monitor Inference Requests

```python
from serving.monitoring.metrics import InferenceMetrics

# Track a single inference request
with InferenceMetrics.track_inference("diagnostic_model_v1"):
    # Your model inference code here
    result = model.predict(patient_data)
    
    # Track accuracy if ground truth is available
    if ground_truth:
        InferenceMetrics.record_accuracy("diagnostic_model_v1", result, ground_truth)
```

### 3. Monitor Clinical Outcomes

```python
from serving.monitoring.tracking import PatientOutcome

# Record patient outcome
outcome = PatientOutcome(
    patient_id="anonymized_id",
    diagnosis_predicted=predicted_diagnosis,
    diagnosis_actual=actual_diagnosis,
    treatment_recommended=treatment_plan,
    treatment_outcome=outcome_status
)

clinical_tracker.record_outcome(outcome)
```

### 4. Configure Alerts

```python
from serving.monitoring.alerts import AlertRule

# Create custom alert rule
alert_rule = AlertRule(
    name="High Error Rate",
    condition="error_rate > 0.05",
    severity="warning",
    duration=300,  # 5 minutes
    channels=["email", "slack"]
)

alert_manager.add_rule(alert_rule)
```

## API Reference

### Metrics Collector

```python
class MetricsCollector:
    def start(self) -> None:
        """Start metrics collection"""
        
    def stop(self) -> None:
        """Stop metrics collection"""
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        
    def record_inference(self, model_name: str, latency: float, success: bool) -> None:
        """Record inference metrics"""
        
    def record_memory_usage(self, cpu_percent: float, gpu_memory: Dict) -> None:
        """Record memory usage"""
```

### Clinical Outcome Tracker

```python
class ClinicalOutcomeTracker:
    def record_outcome(self, outcome: PatientOutcome) -> None:
        """Record patient outcome"""
        
    def get_diagnostic_accuracy(self, time_range: Tuple[str, str]) -> float:
        """Get diagnostic accuracy for time range"""
        
    def get_treatment_effectiveness(self, treatment_type: str) -> float:
        """Get treatment effectiveness"""
        
    def get_adverse_events(self, time_range: Tuple[str, str]) -> List[AdverseEvent]:
        """Get adverse events"""
```

### Drift Detector

```python
class DriftDetector:
    def detect_drift(self, current_data: np.ndarray, reference_data: np.ndarray) -> DriftResult:
        """Detect model drift using statistical tests"""
        
    def is_drift_detected(self) -> bool:
        """Check if drift is currently detected"""
        
    def get_drift_metrics(self) -> Dict[str, float]:
        """Get current drift metrics"""
```

### Alert Manager

```python
class AlertManager:
    def add_rule(self, rule: AlertRule) -> None:
        """Add alert rule"""
        
    def remove_rule(self, rule_name: str) -> None:
        """Remove alert rule"""
        
    def test_rule(self, rule_name: str) -> bool:
        """Test if alert rule condition is met"""
        
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
```

## Dashboard Integration

### Grafana Setup

1. **Install Grafana**: Follow [official installation guide](https://grafana.com/docs/grafana/latest/setup/)

2. **Configure Data Source**:
   ```python
   from serving.monitoring.dashboards import GrafanaIntegration
   
   grafana = GrafanaIntegration(config)
   grafana.configure_data_source("Prometheus", "http://localhost:9090")
   ```

3. **Provision Dashboards**:
   ```python
   # Create medical AI dashboards
   grafana.create_medical_ai_dashboards()
   
   # Create clinical outcomes dashboard
   grafana.create_clinical_outcomes_dashboard()
   
   # Create system performance dashboard
   grafana.create_system_performance_dashboard()
   ```

### Dashboard Components

#### Medical AI Metrics Dashboard
- Real-time inference rates
- Latency percentiles
- Error rates and trends
- Model performance metrics

#### Clinical Outcomes Dashboard
- Diagnostic accuracy over time
- Treatment effectiveness metrics
- Patient outcome distributions
- Adverse events monitoring

#### System Performance Dashboard
- CPU and GPU utilization
- Memory usage trends
- System resource consumption
- SLA compliance tracking

## Alerting System

### Alert Rules Configuration

```yaml
alerts:
  rules:
    # Latency Alerts
    - name: "High Latency P95"
      condition: "latency_p95 > 2000"
      severity: "warning"
      duration: 300  # 5 minutes
      channels: ["slack"]
      
    # Error Rate Alerts  
    - name: "High Error Rate"
      condition: "error_rate > 0.01"
      severity: "critical"
      duration: 60
      channels: ["email", "slack", "pagerduty"]
      
    # Clinical Accuracy Alerts
    - name: "Low Diagnostic Accuracy"
      condition: "diagnostic_accuracy < 0.85"
      severity: "critical"
      duration: 600
      channels: ["email"]
```

### Notification Channels

#### Email Configuration
```python
from serving.monitoring.alerts import EmailNotifier

email_notifier = EmailNotifier(
    smtp_server="smtp.company.com",
    smtp_port=587,
    username="alerts@company.com",
    password_env_var="EMAIL_PASSWORD"
)
```

#### Slack Configuration
```python
from serving.monitoring.alerts import SlackNotifier

slack_notifier = SlackNotifier(
    webhook_url_env_var="SLACK_WEBHOOK_URL",
    channel="#medical-ai-alerts"
)
```

#### PagerDuty Configuration
```python
from serving.monitoring.alerts import PagerDutyNotifier

pagerduty_notifier = PagerDutyNotifier(
    integration_key_env_var="PAGERDUTY_INTEGRATION_KEY"
)
```

### Escalation Policies

```yaml
escalation:
  escalation_time: 15      # Minutes before escalation
  max_level: 3             # Maximum escalation level
  escalation_channels:
    level_1: ["email"]           # First level: Email only
    level_2: ["slack", "email"]  # Second level: Slack + Email
    level_3: ["pagerduty"]       # Third level: PagerDuty
```

## Clinical Outcomes Tracking

### Patient Outcome Recording

```python
from serving.monitoring.tracking import PatientOutcome, OutcomeType

# Record diagnostic outcome
diagnostic_outcome = PatientOutcome(
    patient_id="hashed_patient_id",  # Anonymized for HIPAA compliance
    outcome_type=OutcomeType.DIAGNOSTIC,
    predicted_value="pneumonia",
    actual_value="pneumonia",
    confidence_score=0.92,
    model_name="chest_xray_model_v2"
)

clinical_tracker.record_outcome(diagnostic_outcome)

# Record treatment outcome
treatment_outcome = PatientOutcome(
    patient_id="hashed_patient_id",
    outcome_type=OutcomeType.TREATMENT,
    predicted_value="antibiotic_therapy",
    actual_value="improved",
    treatment_effectiveness=0.88,
    model_name="treatment_recommendation_model_v1"
)

clinical_tracker.record_outcome(treatment_outcome)
```

### Effectiveness Metrics

```python
# Get diagnostic accuracy over time
accuracy = clinical_tracker.get_diagnostic_accuracy(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Get treatment effectiveness by type
effectiveness = clinical_tracker.get_treatment_effectiveness("antibiotic_therapy")

# Get adverse events
adverse_events = clinical_tracker.get_adverse_events(
    start_date="2024-01-01", 
    end_date="2024-12-31"
)
```

### Privacy Protection

All patient data is automatically anonymized and protected:

- **PHI Redaction**: Personal health information is automatically redacted
- **Hashing**: Patient IDs are hashed for anonymity
- **Audit Logging**: All access is logged for compliance
- **Encryption**: Data is encrypted at rest and in transit

## Model Drift Detection

### Statistical Tests

The system supports multiple drift detection methods:

1. **Kolmogorov-Smirnov Test**: Detects distribution changes
2. **Population Stability Index (PSI)**: Measures population stability
3. **Chi-Square Test**: Tests categorical data drift

### Drift Detection Setup

```python
from serving.monitoring.tracking import DriftDetector, StatisticalMethod

# Initialize drift detector
drift_detector = DriftDetector(
    methods=[StatisticalMethod.KS_TEST, StatisticalMethod.PSI],
    alpha=0.05,
    reference_data=reference_dataset
)

# Check for drift
drift_result = drift_detector.detect_drift(current_data, reference_data)

if drift_result.drift_detected:
    print(f"Drift detected: {drift_result.test_results}")
    
    # Trigger alerts
    alert_manager.trigger_alert("model_drift_detected", drift_result)
```

### Drift Thresholds

```yaml
drift_detection:
  methods:
    - ks_test
    - psi
    - chi_square
  alpha: 0.05
  psi_warning: 0.2
  psi_critical: 0.25
  interval: 3600  # Check every hour
```

## Security and Compliance

### HIPAA Compliance

The monitoring system is designed with HIPAA compliance in mind:

- **Audit Logging**: All access to patient data is logged
- **Data Anonymization**: Patient identifiers are automatically hashed
- **Access Control**: Role-based access to monitoring dashboards
- **Encryption**: Data encrypted at rest and in transit
- **Retention Policies**: Configurable data retention for compliance

### Privacy Protection

```python
from serving.monitoring.tracking import PHIProtector

# Configure PHI protection
phi_protector = PHIProtector(
    hash_salt="medical_ai_salt_2024",
    anonymization_method="hashing",
    redact_fields=["patient_name", "ssn", "dob"]
)

# Anonymize patient data
anonymized_data = phi_protector.anonymize(patient_data)
```

### Security Best Practices

1. **Environment Variables**: Store sensitive data in environment variables
2. **API Keys**: Use API keys for external service authentication
3. **Network Security**: Restrict access to monitoring endpoints
4. **Regular Updates**: Keep dependencies updated for security patches

## Troubleshooting

### Common Issues

#### High Memory Usage
```python
# Check memory utilization
from serving.monitoring.metrics import MemoryMetrics

memory_info = MemoryMetrics.get_current_usage()
print(f"CPU Memory: {memory_info.cpu_percent}%")
print(f"GPU Memory: {memory_info.gpu_memory}%")
```

#### Missing Metrics
```python
# Verify Prometheus exporter is running
import requests

response = requests.get("http://localhost:8000/metrics")
if response.status_code != 200:
    print("Prometheus exporter not responding")
```

#### Alert Delivery Issues
```python
# Test notification channels
from serving.monitoring.alerts import NotificationSystem

notifier = NotificationSystem(config)

# Test email delivery
result = notifier.test_channel("email")
print(f"Email test result: {result}")

# Test Slack delivery  
result = notifier.test_channel("slack")
print(f"Slack test result: {result}")
```

### Performance Optimization

1. **Adjust Collection Intervals**: Increase intervals for less critical metrics
2. **Batch Database Writes**: Configure batch sizes for clinical outcomes
3. **Prometheus Retention**: Set appropriate retention policies
4. **Alert Suppression**: Implement alert deduplication

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger("serving.monitoring").setLevel(logging.DEBUG)
```

### Health Checks

```python
from serving.monitoring.utils import MonitoringHealthCheck

health_check = MonitoringHealthCheck(config)

# Check all components
status = health_check.check_all()
print(f"System health: {status}")

# Check individual components
metrics_status = health_check.check_metrics_collection()
alerts_status = health_check.check_alert_system()
```

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from serving.monitoring import MetricsCollector

app = FastAPI()
metrics_collector = MetricsCollector(config)

@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    latency = (time.time() - start_time) * 1000
    model_name = "fastapi_inference_model"
    
    metrics_collector.record_inference(
        model_name=model_name,
        latency=latency,
        success=response.status_code < 400
    )
    
    return response

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Your prediction logic here
    result = model.predict(request.data)
    
    # Record clinical outcome if available
    if hasattr(request, 'ground_truth'):
        outcome = PatientOutcome(
            patient_id=request.patient_id,
            predicted_value=result,
            actual_value=request.ground_truth
        )
        clinical_tracker.record_outcome(outcome)
    
    return {"prediction": result}
```

### Celery Integration

```python
from celery import Celery
from serving.monitoring import MetricsCollector

app = Celery('medical_ai_tasks')

@app.task
def medical_inference_task(patient_data, model_name):
    with InferenceMetrics.track_inference(model_name):
        result = model.predict(patient_data)
        return result

@app.task  
def batch_clinical_tracking(batch_data):
    for data in batch_data:
        outcome = PatientOutcome(**data)
        clinical_tracker.record_outcome(outcome)
```

## Monitoring Best Practices

### 1. Define Clear SLAs
- Set realistic performance thresholds
- Monitor SLA compliance
- Alert on SLA violations

### 2. Use Appropriate Alert Severity
- **Critical**: Immediate attention required
- **Warning**: Requires investigation
- **Info**: For awareness and trending

### 3. Monitor Clinical Outcomes
- Track diagnostic accuracy trends
- Monitor treatment effectiveness
- Alert on unusual patterns

### 4. Regular Drift Detection
- Run drift detection regularly
- Monitor drift thresholds
- Plan model retraining schedules

### 5. Audit and Compliance
- Maintain audit logs
- Review access patterns
- Ensure HIPAA compliance

## Contributing

### Development Setup

1. Clone the repository
2. Install development dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest`
4. Run linting: `flake8 serving/monitoring/`

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Add tests for new features

### Testing

```bash
# Run all tests
pytest serving/monitoring/tests/

# Run specific test categories
pytest serving/monitoring/tests/test_metrics.py
pytest serving/monitoring/tests/test_clinical_tracking.py
pytest serving/monitoring/tests/test_alerts.py

# Run with coverage
pytest --cov=serving.monitoring serving/monitoring/tests/
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Contact the medical AI team
- Check the troubleshooting section

---

**Note**: This monitoring system is designed for medical AI applications and includes features specific to healthcare environments including HIPAA compliance, clinical outcome tracking, and regulatory compliance monitoring.