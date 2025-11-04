# Training Configuration for Synthetic Data Generation

# Default generation parameters
DEFAULT_GENERATION_CONFIG = {
    "num_scenarios": 1000,
    "age_range": [18, 80],
    "seed": 42,
    "specialty_distribution": {
        "general": 0.30,
        "cardiology": 0.15,
        "respiratory": 0.15,
        "gastrointestinal": 0.10,
        "neurology": 0.10,
        "pediatrics": 0.10,
        "geriatrics": 0.05,
        "orthopedics": 0.025,
        "dermatology": 0.025,
        "mental_health": 0.025
    },
    "triage_distribution": {
        "NON_URGENT": 0.40,
        "LESS_URGENT": 0.30,
        "URGENT": 0.20,
        "EMERGENT": 0.08,
        "IMMEDIATE": 0.02
    }
}

# Augmentation parameters
DEFAULT_AUGMENTATION_CONFIG = {
    "synonym_probability": 0.30,
    "paraphrase_probability": 0.40,
    "adversarial_probability": 0.20,
    "diversity_threshold": 0.80,
    "max_augmentations": 5,
    "preserve_medical_terms": True,
    "context_aware": True
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_dataset_size": 50,
    "min_diversity_score": 0.30,
    "max_semantic_similarity": 0.90,
    "min_medical_accuracy": 0.80,
    "min_conversation_length": 4,
    "max_conversation_length": 50
}

# Batch generation settings
BATCH_CONFIG = {
    "default_batch_size": 200,
    "max_concurrent_batches": 5,
    "memory_limit_mb": 2048,
    "output_compression": True
}