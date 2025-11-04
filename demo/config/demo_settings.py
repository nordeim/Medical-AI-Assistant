# Demo Environment Configuration
# Optimized for presentations and demonstrations

# Database Configuration
DATABASE_URL = "sqlite:///demo.db"
DATABASE_CONFIG = {
    "echo": False,
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "connect_args": {
        "check_same_thread": False,
        "timeout": 30
    }
}

# Demo Mode Settings
DEMO_MODE = {
    "enabled": True,
    "version": "1.0",
    "features": [
        "synthetic_data",
        "demo_scenarios", 
        "analytics",
        "role_based_access",
        "optimized_performance"
    ],
    "reset_after_demo": False,
    "auto_backup": True
}

# Performance Optimization
PERFORMANCE_CONFIG = {
    "fast_responses": True,
    "preload_models": True,
    "cache_enabled": True,
    "cache_ttl": 300,  # 5 minutes
    "model_warmup": True,
    "query_optimization": True,
    "response_timeout": 2.0,  # 2 seconds max for demo
    "batch_size": 50,
    "concurrent_requests": 10
}

# API Configuration for Demo
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 1,  # Single worker for demo consistency
    "timeout": 30,
    "keep_alive_timeout": 5,
    "max_request_size": 10485760,  # 10MB
    "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
    "cors_allow_credentials": True,
    "cors_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "cors_headers": ["*"]
}

# Authentication Configuration
AUTH_CONFIG = {
    "secret_key": "demo-secret-key-for-presentations-only",
    "algorithm": "HS256",
    "access_token_expire_minutes": 480,  # 8 hours for demo
    "password_min_length": 8,
    "session_timeout_minutes": 60,
    "max_login_attempts": 3,
    "lockout_duration_minutes": 15
}

# Demo Users (Pre-configured for easy access)
DEMO_USERS = {
    "admin": {
        "email": "admin@demo.medai.com",
        "password": "DemoAdmin123!",
        "role": "admin",
        "permissions": ["all"]
    },
    "nurse_jones": {
        "email": "nurse.jones@demo.medai.com", 
        "password": "DemoNurse456!",
        "role": "nurse",
        "permissions": ["read_patients", "write_assessments", "read_vitals"]
    },
    "patient_smith": {
        "email": "patient.smith@demo.medai.com",
        "password": "DemoPatient789!", 
        "role": "patient",
        "permissions": ["read_own_data"]
    }
}

# Demo Scenarios Configuration
DEMO_SCENARIOS = {
    "diabetes_management": {
        "name": "Diabetes Management Demo",
        "description": "Real-time glucose monitoring and insulin recommendations",
        "patient_id": 1,
        "estimated_duration": 15,
        "difficulty": "intermediate",
        "features": [
            "glucose_trend_analysis",
            "insulin_dose_calculation",
            "dietary_recommendations",
            "alert_system"
        ]
    },
    "hypertension_monitoring": {
        "name": "Hypertension Monitoring Demo", 
        "description": "Blood pressure tracking and cardiovascular risk assessment",
        "patient_id": 2,
        "estimated_duration": 12,
        "difficulty": "beginner",
        "features": [
            "bp_trend_analysis", 
            "medication_adherence",
            "risk_stratification",
            "lifestyle_recommendations"
        ]
    },
    "chest_pain_assessment": {
        "name": "Chest Pain Assessment Demo",
        "description": "Emergency triage and cardiovascular risk evaluation",
        "patient_id": 3, 
        "estimated_duration": 10,
        "difficulty": "advanced",
        "features": [
            "symptom_evaluation",
            "risk_stratification",
            "emergency_protocols",
            "specialist_referral"
        ]
    }
}

# Analytics and Tracking
ANALYTICS_CONFIG = {
    "enabled": True,
    "track_user_interactions": True,
    "track_system_performance": True,
    "track_demo_completion": True,
    "session_timeout": 3600,  # 1 hour
    "data_retention_days": 7,
    "privacy_mode": "synthetic_only"
}

# Caching Configuration
CACHE_CONFIG = {
    "type": "redis",  # or "memory" for simple demo
    "default_timeout": 300,
    "max_entries": 1000,
    "cull_frequency": 3,
    "urls": {
        "redis": "redis://localhost:6379/0"
    }
}

# Logging Configuration for Demo
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "INFO"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "demo.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,
            "formatter": "detailed",
            "level": "DEBUG"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

# Health Check Configuration
HEALTH_CONFIG = {
    "enabled": True,
    "check_database": True,
    "check_models": True,
    "check_external_services": False,  # Disabled for demo
    "response_timeout": 5.0,
    "metrics_enabled": True
}

# Demo Backup Configuration
BACKUP_CONFIG = {
    "enabled": True,
    "auto_backup": True,
    "backup_interval_hours": 24,
    "backup_location": "./backups/",
    "retention_days": 7,
    "compression": True,
    "encryption": False,  # Disabled for demo simplicity
    "notify_on_failure": False
}

# UI/UX Demo Settings
UI_CONFIG = {
    "demo_mode_banner": True,
    "simplified_workflows": True,
    "show_tooltips": True,
    "auto_save": True,
    "confirmation_dialogs": False,  # Disabled for smoother demo
    "loading_animations": True,
    "success_feedback": True,
    "error_handling": "user_friendly"
}

# Security Configuration (Relaxed for Demo)
SECURITY_CONFIG = {
    "demo_mode": True,
    "phi_protection": "synthetic_only",
    "audit_logging": True,
    "session_security": "relaxed",
    "cors_enabled": True,
    "rate_limiting": False,  # Disabled for demo
    "ssl_verify": False,     # Disabled for localhost demo
    "debug_mode": True
}

# Demo Reset Configuration
RESET_CONFIG = {
    "enabled": True,
    "reset_on_startup": False,
    "reset_endpoints": {
        "database": "/api/demo/reset-database",
        "analytics": "/api/demo/reset-analytics", 
        "cache": "/api/demo/reset-cache"
    },
    "backup_before_reset": True
}