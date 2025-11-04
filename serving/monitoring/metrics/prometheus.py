"""
Prometheus Metrics Integration

Provides comprehensive Prometheus metrics collection and export for medical AI serving platform.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from prometheus_client import (
    CollectorRegistry, 
    generate_latest, 
    CONTENT_TYPE_LATEST,
    Counter, 
    Histogram, 
    Gauge, 
    Summary,
    Info,
    multiprocess
)
import structlog

logger = structlog.get_logger("prometheus_metrics")


class PrometheusMetricsCollector:
    """
    Prometheus metrics collector for medical AI serving platform.
    
    This class provides comprehensive metrics collection specifically designed
    for medical AI systems with clinical outcome tracking and compliance monitoring.
    """
    
    def __init__(self, 
                 registry: Optional[CollectorRegistry] = None,
                 enable_clinical_metrics: bool = True,
                 enable_compliance_metrics: bool = True):
        
        self.registry = registry or CollectorRegistry()
        self.enable_clinical_metrics = enable_clinical_metrics
        self.enable_compliance_metrics = enable_compliance_metrics
        
        self.logger = structlog.get_logger("prometheus_collector")
        
        # Initialize all metric collectors
        self._init_inference_metrics()
        self._init_system_metrics()
        self._init_model_performance_metrics()
        self._init_clinical_metrics()
        self._init_compliance_metrics()
        self._init_sla_metrics()
        
        self.logger.info("Prometheus metrics collector initialized")
    
    def _init_inference_metrics(self):
        """Initialize inference-specific metrics."""
        
        # Request counters
        self.inference_requests_total = Counter(
            'medical_ai_inference_requests_total',
            'Total number of inference requests processed',
            ['model_id', 'model_type', 'status_code', 'user_type'],
            registry=self.registry
        )
        
        self.inference_errors_total = Counter(
            'medical_ai_inference_errors_total',
            'Total number of inference errors',
            ['model_id', 'error_type', 'severity'],
            registry=self.registry
        )
        
        # Latency histograms
        self.inference_duration_seconds = Histogram(
            'medical_ai_inference_duration_seconds',
            'Time spent processing inference requests',
            ['model_id', 'model_type', 'operation_type'],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        self.preprocessing_duration_seconds = Histogram(
            'medical_ai_preprocessing_duration_seconds',
            'Time spent preprocessing input data',
            ['model_id', 'data_type'],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            registry=self.registry
        )
        
        self.postprocessing_duration_seconds = Histogram(
            'medical_ai_postprocessing_duration_seconds',
            'Time spent postprocessing model output',
            ['model_id', 'output_type'],
            buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
            registry=self.registry
        )
        
        # Throughput metrics
        self.inference_tokens_total = Counter(
            'medical_ai_inference_tokens_total',
            'Total number of tokens processed (input and output)',
            ['model_id', 'token_type'],  # token_type: input, output
            registry=self.registry
        )
        
        self.inference_characters_total = Counter(
            'medical_ai_inference_characters_total',
            'Total number of characters processed',
            ['model_id', 'data_type'],  # data_type: input, output
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_lookups_total = Counter(
            'medical_ai_cache_lookups_total',
            'Total number of cache lookups',
            ['model_id', 'cache_type', 'result'],  # result: hit, miss
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'medical_ai_cache_hit_ratio',
            'Cache hit ratio',
            ['model_id', 'cache_type'],
            registry=self.registry
        )
        
        # Request size metrics
        self.request_size_bytes = Histogram(
            'medical_ai_request_size_bytes',
            'Size of inference requests in bytes',
            ['model_id', 'data_type'],
            buckets=[1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
            registry=self.registry
        )
        
        self.response_size_bytes = Histogram(
            'medical_ai_response_size_bytes',
            'Size of inference responses in bytes',
            ['model_id', 'data_type'],
            buckets=[1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system resource metrics."""
        
        # CPU metrics
        self.cpu_usage_percent = Gauge(
            'medical_ai_system_cpu_usage_percent',
            'System CPU usage percentage',
            ['core'],  # Specific core or 'total'
            registry=self.registry
        )
        
        self.cpu_frequency_mhz = Gauge(
            'medical_ai_system_cpu_frequency_mhz',
            'CPU frequency in MHz',
            registry=self.registry
        )
        
        self.cpu_temperature_celsius = Gauge(
            'medical_ai_system_cpu_temperature_celsius',
            'CPU temperature in Celsius',
            registry=self.registry
        )
        
        # Memory metrics
        self.memory_usage_bytes = Gauge(
            'medical_ai_system_memory_usage_bytes',
            'System memory usage in bytes',
            ['type'],  # type: total, available, used, cached, buffers
            registry=self.registry
        )
        
        self.memory_usage_percent = Gauge(
            'medical_ai_system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.swap_usage_bytes = Gauge(
            'medical_ai_system_swap_usage_bytes',
            'System swap usage in bytes',
            ['type'],  # type: total, used, free
            registry=self.registry
        )
        
        # GPU metrics
        self.gpu_available = Gauge(
            'medical_ai_system_gpu_available',
            'Number of available GPUs',
            registry=self.registry
        )
        
        self.gpu_memory_usage_bytes = Gauge(
            'medical_ai_system_gpu_memory_usage_bytes',
            'GPU memory usage in bytes',
            ['gpu_id', 'type'],  # type: total, used, free, cached
            registry=self.registry
        )
        
        self.gpu_utilization_percent = Gauge(
            'medical_ai_system_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_temperature_celsius = Gauge(
            'medical_ai_system_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu_id'],
            registry=self.registry
        )
        
        self.gpu_power_watts = Gauge(
            'medical_ai_system_gpu_power_watts',
            'GPU power consumption in watts',
            ['gpu_id'],
            registry=self.registry
        )
        
        # Disk metrics
        self.disk_usage_bytes = Gauge(
            'medical_ai_system_disk_usage_bytes',
            'Disk usage in bytes',
            ['device', 'mountpoint', 'type'],  # type: total, used, free
            registry=self.registry
        )
        
        self.disk_io_bytes_total = Counter(
            'medical_ai_system_disk_io_bytes_total',
            'Total disk I/O in bytes',
            ['device', 'operation'],  # operation: read, write
            registry=self.registry
        )
        
        # Network metrics
        self.network_io_bytes_total = Counter(
            'medical_ai_system_network_io_bytes_total',
            'Total network I/O in bytes',
            ['interface', 'operation'],  # operation: sent, received
            registry=self.registry
        )
        
        self.network_connections = Gauge(
            'medical_ai_system_network_connections',
            'Number of network connections',
            ['state'],  # state: established, listening, time_wait, etc.
            registry=self.registry
        )
        
        # Process metrics
        self.process_cpu_percent = Gauge(
            'medical_ai_process_cpu_percent',
            'Process CPU usage percentage',
            registry=self.registry
        )
        
        self.process_memory_bytes = Gauge(
            'medical_ai_process_memory_bytes',
            'Process memory usage in bytes',
            ['type'],  # type: rss, vms, data, stack, etc.
            registry=self.registry
        )
        
        self.process_threads = Gauge(
            'medical_ai_process_threads',
            'Number of process threads',
            registry=self.registry
        )
    
    def _init_model_performance_metrics(self):
        """Initialize model performance metrics."""
        
        # Model information
        self.model_info = Info(
            'medical_ai_model_info',
            'Model information and metadata',
            ['model_id', 'model_type', 'version', 'framework', 'quantization_type'],
            registry=self.registry
        )
        
        # Accuracy metrics
        self.model_accuracy_score = Gauge(
            'medical_ai_model_accuracy_score',
            'Model accuracy score',
            ['model_id', 'metric_type'],  # metric_type: accuracy, precision, recall, f1
            registry=self.registry
        )
        
        # Performance drift metrics
        self.model_drift_score = Gauge(
            'medical_ai_model_drift_score',
            'Model drift detection score',
            ['model_id', 'drift_type'],  # drift_type: data, concept, performance
            registry=self.registry
        )
        
        # Model size and complexity
        self.model_size_mb = Gauge(
            'medical_ai_model_size_mb',
            'Model size in megabytes',
            ['model_id'],
            registry=self.registry
        )
        
        self.model_parameters = Gauge(
            'medical_ai_model_parameters',
            'Number of model parameters',
            ['model_id'],
            registry=self.registry
        )
        
        # Inference statistics
        self.model_inference_stats = Summary(
            'medical_ai_model_inference_stats',
            'Model inference statistics (25/50/75/90/95/99th percentiles)',
            ['model_id', 'stat_type'],  # stat_type: latency_ms, tokens_per_sec, confidence
            registry=self.registry
        )
    
    def _init_clinical_metrics(self):
        """Initialize clinical effectiveness metrics."""
        if not self.enable_clinical_metrics:
            return
        
        # Clinical effectiveness
        self.clinical_effectiveness_score = Gauge(
            'medical_ai_clinical_effectiveness_score',
            'Clinical effectiveness score for the model',
            ['model_id', 'specialty'],  # specialty: cardiology, oncology, etc.
            registry=self.registry
        )
        
        # Medical relevance
        self.medical_relevance_score = Gauge(
            'medical_ai_medical_relevance_score',
            'Medical relevance score for model predictions',
            ['model_id', 'context_type'],  # context_type: diagnosis, treatment, prognosis
            registry=self.registry
        )
        
        # Safety metrics
        self.safety_score = Gauge(
            'medical_ai_safety_score',
            'Safety score indicating risk level',
            ['model_id', 'risk_level'],  # risk_level: low, medium, high
            registry=self.registry
        )
        
        # Bias detection
        self.bias_score = Gauge(
            'medical_ai_bias_score',
            'Bias detection score across different demographics',
            ['model_id', 'demographic_type'],  # demographic_type: age, gender, ethnicity
            registry=self.registry
        )
        
        # Clinical outcome tracking
        self.clinical_outcomes_total = Counter(
            'medical_ai_clinical_outcomes_total',
            'Total clinical outcomes tracked',
            ['model_id', 'outcome_type', 'result'],  # outcome_type: diagnosis, treatment_recommendation, etc.
            registry=self.registry
        )
        
        # Physician feedback
        self.physician_feedback_score = Gauge(
            'medical_ai_physician_feedback_score',
            'Physician feedback score on model predictions',
            ['model_id', 'feedback_type'],  # feedback_type: helpful, accurate, safe
            registry=self.registry
        )
    
    def _init_compliance_metrics(self):
        """Initialize regulatory compliance metrics."""
        if not self.enable_compliance_metrics:
            return
        
        # HIPAA compliance
        self.hipaa_compliance_checks_total = Counter(
            'medical_ai_hipaa_compliance_checks_total',
            'Total HIPAA compliance checks performed',
            ['check_type', 'result'],  # check_type: phi_detection, data_encryption, audit_log
            registry=self.registry
        )
        
        # Data privacy
        self.data_privacy_violations_total = Counter(
            'medical_ai_data_privacy_violations_total',
            'Total data privacy violations detected',
            ['violation_type', 'severity'],
            registry=self.registry
        )
        
        # Audit logging
        self.audit_logs_total = Counter(
            'medical_ai_audit_logs_total',
            'Total audit log entries created',
            ['action_type', 'resource_type', 'user_type'],
            registry=self.registry
        )
        
        # Regulatory validation
        self.regulatory_validations_total = Counter(
            'medical_ai_regulatory_validations_total',
            'Total regulatory validation checks',
            ['regulation_type', 'validation_result'],  # regulation_type: fda, ce, hipaa
            registry=self.registry
        )
    
    def _init_sla_metrics(self):
        """Initialize SLA tracking metrics."""
        
        # SLA violations
        self.sla_violations_total = Counter(
            'medical_ai_sla_violations_total',
            'Total SLA violations',
            ['sla_type', 'severity', 'model_id'],
            registry=self.registry
        )
        
        # SLA compliance percentage
        self.sla_compliance_percent = Gauge(
            'medical_ai_sla_compliance_percent',
            'SLA compliance percentage',
            ['sla_type', 'time_window'],  # time_window: 1h, 24h, 7d, 30d
            registry=self.registry
        )
        
        # Service availability
        self.service_availability_percent = Gauge(
            'medical_ai_service_availability_percent',
            'Service availability percentage',
            ['service_name', 'time_window'],
            registry=self.registry
        )
    
    # Metric update methods
    def record_inference_request(self, 
                                model_id: str,
                                model_type: str,
                                status_code: int,
                                user_type: str,
                                duration_seconds: float,
                                preprocessing_seconds: float = 0,
                                postprocessing_seconds: float = 0,
                                input_tokens: int = 0,
                                output_tokens: int = 0,
                                request_size_bytes: int = 0,
                                response_size_bytes: int = 0):
        """Record inference request metrics."""
        try:
            self.inference_requests_total.labels(
                model_id=model_id,
                model_type=model_type,
                status_code=status_code,
                user_type=user_type
            ).inc()
            
            if status_code >= 400:
                self.inference_errors_total.labels(
                    model_id=model_id,
                    error_type=f"http_{status_code}",
                    severity="error" if status_code >= 500 else "warning"
                ).inc()
            
            self.inference_duration_seconds.labels(
                model_id=model_id,
                model_type=model_type,
                operation_type="total"
            ).observe(duration_seconds)
            
            if preprocessing_seconds > 0:
                self.preprocessing_duration_seconds.labels(
                    model_id=model_id,
                    data_type="medical_text"
                ).observe(preprocessing_seconds)
            
            if postprocessing_seconds > 0:
                self.postprocessing_duration_seconds.labels(
                    model_id=model_id,
                    output_type="medical_response"
                ).observe(postprocessing_seconds)
            
            if input_tokens > 0:
                self.inference_tokens_total.labels(
                    model_id=model_id,
                    token_type="input"
                ).inc(input_tokens)
            
            if output_tokens > 0:
                self.inference_tokens_total.labels(
                    model_id=model_id,
                    token_type="output"
                ).inc(output_tokens)
            
            if request_size_bytes > 0:
                self.request_size_bytes.labels(
                    model_id=model_id,
                    data_type="input"
                ).observe(request_size_bytes)
            
            if response_size_bytes > 0:
                self.response_size_bytes.labels(
                    model_id=model_id,
                    data_type="output"
                ).observe(response_size_bytes)
                
        except Exception as e:
            self.logger.error("Failed to record inference metrics", error=str(e))
    
    def record_cache_operation(self, 
                              model_id: str,
                              cache_type: str,
                              hit: bool):
        """Record cache operation metrics."""
        try:
            result = "hit" if hit else "miss"
            self.cache_lookups_total.labels(
                model_id=model_id,
                cache_type=cache_type,
                result=result
            ).inc()
            
        except Exception as e:
            self.logger.error("Failed to record cache metrics", error=str(e))
    
    def update_system_metrics(self, system_metrics: Dict[str, Any]):
        """Update system resource metrics."""
        try:
            # CPU metrics
            if 'cpu_percent' in system_metrics:
                self.cpu_usage_percent.labels(core='total').set(system_metrics['cpu_percent'])
            
            if 'cpu_frequency_mhz' in system_metrics:
                self.cpu_frequency_mhz.set(system_metrics['cpu_frequency_mhz'])
            
            if 'cpu_temperature_c' in system_metrics:
                self.cpu_temperature_celsius.set(system_metrics['cpu_temperature_c'])
            
            # Memory metrics
            if 'memory_usage_percent' in system_metrics:
                self.memory_usage_percent.set(system_metrics['memory_usage_percent'])
            
            if 'memory_total_gb' in system_metrics:
                self.memory_usage_bytes.labels(type='total').set(
                    system_metrics['memory_total_gb'] * 1024**3
                )
            
            if 'memory_used_gb' in system_metrics:
                self.memory_usage_bytes.labels(type='used').set(
                    system_metrics['memory_used_gb'] * 1024**3
                )
            
            if 'memory_available_gb' in system_metrics:
                self.memory_usage_bytes.labels(type='available').set(
                    system_metrics['memory_available_gb'] * 1024**3
                )
            
            # GPU metrics
            if system_metrics.get('gpu_available'):
                self.gpu_available.set(system_metrics.get('gpu_count', 0))
                
                if 'gpu_memory_usage_percent' in system_metrics:
                    self.gpu_memory_usage_bytes.labels(
                        gpu_id='0', type='used'
                    ).set(
                        system_metrics['gpu_memory_used_gb'] * 1024**3
                    )
                
                if 'gpu_utilization_percent' in system_metrics:
                    self.gpu_utilization_percent.labels(gpu_id='0').set(
                        system_metrics['gpu_utilization_percent']
                    )
                
                if 'gpu_temperature_c' in system_metrics:
                    self.gpu_temperature_celsius.labels(gpu_id='0').set(
                        system_metrics['gpu_temperature_c']
                    )
                
                if 'gpu_power_watts' in system_metrics:
                    self.gpu_power_watts.labels(gpu_id='0').set(
                        system_metrics['gpu_power_watts']
                    )
            
        except Exception as e:
            self.logger.error("Failed to update system metrics", error=str(e))
    
    def update_model_info(self, model_id: str, model_info: Dict[str, Any]):
        """Update model information metrics."""
        try:
            self.model_info.labels(
                model_id=model_id,
                model_type=model_info.get('type', 'unknown'),
                version=model_info.get('version', 'unknown'),
                framework=model_info.get('framework', 'pytorch'),
                quantization_type=model_info.get('quantization', 'none')
            ).info(model_info)
            
        except Exception as e:
            self.logger.error("Failed to update model info", error=str(e))
    
    def update_model_performance(self, 
                                model_id: str,
                                accuracy_metrics: Dict[str, float],
                                drift_scores: Dict[str, float]):
        """Update model performance metrics."""
        try:
            # Accuracy metrics
            for metric_type, value in accuracy_metrics.items():
                self.model_accuracy_score.labels(
                    model_id=model_id,
                    metric_type=metric_type
                ).set(value)
            
            # Drift scores
            for drift_type, score in drift_scores.items():
                self.model_drift_score.labels(
                    model_id=model_id,
                    drift_type=drift_type
                ).set(score)
                
        except Exception as e:
            self.logger.error("Failed to update model performance metrics", error=str(e))
    
    def record_clinical_outcome(self,
                               model_id: str,
                               outcome_type: str,
                               result: str,
                               effectiveness_score: float = 0.0):
        """Record clinical outcome metrics."""
        if not self.enable_clinical_metrics:
            return
        
        try:
            self.clinical_outcomes_total.labels(
                model_id=model_id,
                outcome_type=outcome_type,
                result=result
            ).inc()
            
            if effectiveness_score > 0:
                self.clinical_effectiveness_score.labels(
                    model_id=model_id,
                    specialty="general"  # Could be enhanced with specialty detection
                ).set(effectiveness_score)
                
        except Exception as e:
            self.logger.error("Failed to record clinical outcome", error=str(e))
    
    def record_sla_violation(self,
                           sla_type: str,
                           severity: str,
                           model_id: str = "system"):
        """Record SLA violation."""
        try:
            self.sla_violations_total.labels(
                sla_type=sla_type,
                severity=severity,
                model_id=model_id
            ).inc()
            
        except Exception as e:
            self.logger.error("Failed to record SLA violation", error=str(e))
    
    def update_sla_compliance(self, 
                            sla_type: str,
                            time_window: str,
                            compliance_percent: float):
        """Update SLA compliance metrics."""
        try:
            self.sla_compliance_percent.labels(
                sla_type=sla_type,
                time_window=time_window
            ).set(compliance_percent)
            
        except Exception as e:
            self.logger.error("Failed to update SLA compliance", error=str(e))
    
    def generate_metrics_output(self) -> bytes:
        """Generate Prometheus metrics output."""
        try:
            return generate_latest(self.registry)
        except Exception as e:
            self.logger.error("Failed to generate metrics output", error=str(e))
            return b"# No metrics available"
    
    def get_metrics_content_type(self) -> str:
        """Get the content type for metrics output."""
        return CONTENT_TYPE_LATEST


# Global instance for easy access
prometheus_collector = None

def get_prometheus_collector() -> Optional[PrometheusMetricsCollector]:
    """Get the global Prometheus collector instance."""
    return prometheus_collector

def init_prometheus_collector(registry: Optional[CollectorRegistry] = None,
                            enable_clinical_metrics: bool = True,
                            enable_compliance_metrics: bool = True) -> PrometheusMetricsCollector:
    """Initialize the global Prometheus collector instance."""
    global prometheus_collector
    prometheus_collector = PrometheusMetricsCollector(
        registry=registry,
        enable_clinical_metrics=enable_clinical_metrics,
        enable_compliance_metrics=enable_compliance_metrics
    )
    return prometheus_collector