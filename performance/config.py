"""
Performance Optimization Configuration
Enterprise-grade configuration for medical AI workloads
"""

# Database Configuration
DATABASE_CONFIG = {
    'connection_string': 'postgresql://user:pass@localhost:5432/medical_ai',
    'min_connections': 10,
    'max_connections': 50,
    'command_timeout': 60,
    'index_optimization': True,
    'query_optimization': True
}

# Redis Caching Configuration
REDIS_CONFIG = {
    'url': 'redis://localhost:6379',
    'max_connections': 20,
    'retry_on_timeout': True,
    'socket_keepalive': True,
    'cache_ttl': {
        'patient_data': 1800,      # 30 minutes
        'clinical_data': 900,      # 15 minutes
        'ai_inference': 3600,      # 1 hour
        'audit_logs': 7200,        # 2 hours
        'vital_signs': 600,        # 10 minutes
        'medications': 1200        # 20 minutes
    }
}

# Model Optimization Configuration
MODEL_CONFIG = {
    'quantization_levels': ['4bit', '8bit', 'fp16'],
    'default_quantization': '4bit',
    'batch_processing': {
        'enabled': True,
        'min_batch_size': 1,
        'max_batch_size': 8,
        'optimal_size': 4
    },
    'performance_targets': {
        'max_inference_time': 2.0,  # seconds
        'min_throughput': 10.0,     # tokens/second
        'max_memory_usage': 8.0     # GB
    }
}

# Auto-scaling Configuration
SCALING_CONFIG = {
    'hpa': {
        'min_replicas': 2,
        'max_replicas': 50,
        'cpu_target': 70.0,
        'memory_target': 80.0,
        'scale_up_stabilization': 60,   # seconds
        'scale_down_stabilization': 300  # seconds
    },
    'vpa': {
        'enabled': True,
        'update_mode': 'Auto',
        'resource_policy': {
            'min_cpu': '500m',
            'max_cpu': '4000m',
            'min_memory': '512Mi',
            'max_memory': '8Gi'
        }
    },
    'predictive_scaling': {
        'enabled': True,
        'prediction_horizon': 24,  # hours
        'confidence_threshold': 0.6
    }
}

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    'strategy': 'adaptive',
    'endpoints': {
        '/api/patient-data': {'requests': 100, 'window': 60},
        '/api/ai-inference': {'requests': 50, 'window': 60},
        '/api/clinical-data': {'requests': 200, 'window': 60},
        '/api/audit-logs': {'requests': 30, 'window': 60},
        '/api/vital-signs': {'requests': 150, 'window': 60},
        '/api/medications': {'requests': 120, 'window': 60}
    },
    'adaptive_scaling': {
        'high_load_multiplier': 0.5,    # Reduce limits by 50% under high load
        'low_load_multiplier': 1.2      # Increase limits by 20% under low load
    }
}

# Performance Targets
PERFORMANCE_TARGETS = {
    'response_time': {
        'max_p95': 2.0,    # seconds
        'max_p99': 3.0,    # seconds
        'target_avg': 1.0  # seconds
    },
    'throughput': {
        'min_requests_per_second': 100,
        'target_per_service': 50
    },
    'cache': {
        'min_hit_rate': 0.8,      # 80%
        'max_memory_usage': 0.85,  # 85% of available
        'eviction_threshold': 0.9  # 90% of max size
    },
    'resource_utilization': {
        'max_cpu': 70.0,      # %
        'max_memory': 80.0,   # %
        'max_disk': 90.0      # %
    }
}

# Frontend Performance Configuration
FRONTEND_CONFIG = {
    'bundle_optimization': {
        'code_splitting': True,
        'lazy_loading': True,
        'tree_shaking': True,
        'compression': 'gzip'
    },
    'performance_targets': {
        'first_paint': 1.5,    # seconds
        'interactive': 3.0,    # seconds
        'bundle_size': 500,    # KB
        'largest_contentful_paint': 2.5  # seconds
    },
    'caching_strategy': {
        'static_assets': 31536000,  # 1 year
        'api_responses': 300,       # 5 minutes
        'user_specific': 1800       # 30 minutes
    }
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'metrics_collection': {
        'interval': 30,        # seconds
        'retention_period': 30, # days
        'alert_thresholds': {
            'response_time': 2.0,
            'error_rate': 0.05,      # 5%
            'cpu_utilization': 80.0,
            'memory_utilization': 85.0
        }
    },
    'performance_benchmarking': {
        'load_test': {
            'concurrent_users': 10,
            'duration': 300,     # seconds
            'ramp_up_time': 60   # seconds
        },
        'stress_test': {
            'max_concurrent_users': 50,
            'duration': 600      # seconds
        },
        'spike_test': {
            'normal_load': 5,
            'spike_load': 50,
            'spike_duration': 60 # seconds
        },
        'endurance_test': {
            'concurrent_users': 5,
            'duration_hours': 4
        }
    }
}

# Workload Prediction Configuration
WORKLOAD_PREDICTION_CONFIG = {
    'model_parameters': {
        'prediction_horizon': 24,       # hours
        'retrain_interval': 24,         # hours
        'min_training_samples': 100,
        'feature_importance': {
            'hour_of_day': 0.3,
            'day_of_week': 0.2,
            'is_business_hours': 0.2,
            'is_emergency_period': 0.15,
            'endpoint_category': 0.15
        }
    },
    'healthcare_patterns': {
        'peak_hours': [(6, 9), (14, 17)],  # Morning and afternoon
        'low_hours': [(22, 24), (0, 6)],   # Night
        'peak_days': [0, 1, 2, 3, 4],      # Weekdays
        'emergency_periods': [(0, 2), (8, 10), (16, 18)]
    },
    'scaling_recommendations': {
        'cpu_weight': 0.4,
        'memory_weight': 0.3,
        'request_rate_weight': 0.3,
        'base_replicas': 2,
        'max_replicas': 50
    }
}