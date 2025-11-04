# Configuration Reference

Complete reference for all configuration options in the Medical AI Training Pipeline.

## Table of Contents

1. [Configuration File Formats](#configuration-file-formats)
2. [DeepSpeed Configuration](#deepspeed-configuration)
3. [LoRA Configuration](#lora-configuration)
4. [Training Configuration](#training-configuration)
5. [Data Configuration](#data-configuration)
6. [PHI Protection Configuration](#phi-protection-configuration)
7. [Evaluation Configuration](#evaluation-configuration)
8. [Serving Configuration](#serving-configuration)
9. [Monitoring Configuration](#monitoring-configuration)

## Configuration File Formats

### JSON Configuration (Recommended)
```json
{
    "deepspeed_config": {
        "zero_optimization": { ... },
        "bfloat16": { ... }
    },
    "training": {
        "learning_rate": 5e-5,
        "batch_size": 8
    }
}
```

### YAML Configuration (LoRA)
```yaml
model_name: "microsoft/DialoGPT-medium"
output_dir: "./checkpoints"

lora_r: 16
lora_alpha: 32
lora_dropout: 0.1

training:
  learning_rate: 5e-5
  per_device_train_batch_size: 4
```

### Python Configuration (Advanced)
```python
config = {
    "deepspeed_config": {
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": True,
            "cpu_offload_params": True
        },
        "bfloat16": {"enabled": True}
    },
    "training": {
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_steps": 1000
    }
}
```

## DeepSpeed Configuration

### Zero Optimization Stages

#### Stage 1: Optimizer State Partitioning
```json
{
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": false,
        "offload_param": false,
        "gather_16bit_weights_on_model_save": false,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8
    }
}
```

**Description**: Partitions optimizer states across processes. Suitable for small to medium models with minimal memory savings but good performance.

**Memory Savings**: ~25-30%
**Performance Impact**: Low
**Use Cases**: Models < 1B parameters

#### Stage 2: Optimizer + Gradient Partitioning
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": false,
        "gather_16bit_weights_on_model_save": false,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "sub_group_size": 1e9,
        "prefetch_bucket_size": 5e7,
        "param_persistence_threshold": 4e5
    }
}
```

**Description**: Partitions optimizer states and gradients. Reduces memory by ~50% and includes CPU optimizer offloading.

**Memory Savings**: ~50-60%
**Performance Impact**: Medium
**Use Cases**: Models 1-10B parameters

#### Stage 3: Complete Model Parallelism
```json
{
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "gather_16bit_weights_on_model_save": true,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "sub_group_size": 1e9,
        "prefetch_bucket_size": 5e7,
        "param_persistence_threshold": 4e5,
        "stage3_prefetch_stream_reserve_memory": 5e7,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    }
}
```

**Description**: Partitions optimizer states, gradients, and model parameters. Enables training of extremely large models with CPU offloading.

**Memory Savings**: ~75-80%
**Performance Impact**: High
**Use Cases**: Models > 10B parameters

### DeepSpeed Configuration Parameters

#### Zero Optimization Settings

```json
{
    "zero_optimization": {
        "stage": 2,                          // ZeRO optimization stage (0, 1, 2, 3)
        
        "offload_optimizer": {
            "device": "cpu",                 // Offload device ("cpu", "nvme")
            "pin_memory": true,              // Pin memory for faster CPU transfers
            "nvme_path": "/tmp/nvme_offload" // NVMe path for offloading
        },
        
        "offload_param": {
            "device": "cpu",                 // Offload device for parameters
            "pin_memory": true
        },
        
        "gather_16bit_weights_on_model_save": false, // Gather 16-bit weights when saving
        
        "allgather_partitions": true,        // All-gather model weights in partitions
        "allgather_bucket_size": 5e8,       // Bucket size for all-gather (bytes)
        
        "reduce_scatter": true,              // Use reduce-scatter for gradients
        "reduce_bucket_size": 5e8,          // Bucket size for reduce-scatter (bytes)
        
        "sub_group_size": 1e9,              // Size of subgroups for reducing optimizer states
        "prefetch_bucket_size": 5e7,        // Bucket size for prefetching parameters
        
        "param_persistence_threshold": 4e5, // Parameters larger than this will be persisted
        "stage3_prefetch_stream_reserve_memory": 5e7, // Reserve memory for prefetching
        "stage3_max_live_parameters": 1e9,   // Maximum number of live parameters
        "stage3_max_reuse_distance": 1e9,    // Maximum distance for reusing parameters
        "gather_bucket_size": 5e8,          // Bucket size for gathering parameters
        "gather_fragment_size": 5e8,        // Fragment size for gathering
        "profile": false                     // Enable profiling
    }
}
```

#### Communication Optimization

```json
{
    "communication": {
        "backend": "nccl",                   // Communication backend ("nccl", "gloo", "mpi")
        "timeout": 1800,                    // Communication timeout (seconds)
        "max_train_batch_size": 32,         // Maximum training batch size
        "scatter_gather_tensors_to_gpu": true // Scatter/gather tensors to GPU
    }
}
```

#### Mixed Precision Configuration

```json
{
    "bfloat16": {
        "enabled": true                     // Enable BF16 mixed precision
    },
    
    "fp16": {
        "enabled": false,                   // Enable FP16 mixed precision (use BF16 instead)
        "initial_scale_power": 16,         // Initial scale factor power
        "loss_scale_window": 1000,         // Loss scale window
        "loss_scale": 1.0,                 // Initial loss scale
        "hysteresis": 2,                   // Hysteresis factor
        "min_loss_scale": 1.0              // Minimum loss scale
    }
}
```

#### Training Configuration

```json
{
    "train_batch_size": 8,                 // Effective training batch size
    "train_micro_batch_size_per_gpu": 1,   // Micro batch size per GPU
    "gradient_accumulation_steps": 8,      // Number of gradient accumulation steps
    "gradient_clipping": 1.0,             // Gradient clipping value
    
    "zero_allow_untested_optimizer": true, // Allow untested optimizers
    
    "steps_per_print": 10,                 // Steps between prints
    "wall_clock_breakdown": true,          // Enable wall clock breakdown
    "disable_allgather": false             // Disable all-gather operations
}
```

#### Checkpointing Configuration

```json
{
    "checkpoint": {
        "save_interval": 500,              // Save checkpoint every N steps
        "save_total_limit": 10,            // Maximum number of checkpoints to keep
        "load_universal": false,           // Load checkpoints universally
        "use_pin_memory": true,            // Use pinned memory for loading
        "async_save": false,               // Enable async checkpoint saving
        "tag": "latest",                   // Checkpoint tag
        "load_lr_scheduler_states": true,  // Load learning rate scheduler states
        "load_optimizer_states": true      // Load optimizer states
    }
}
```

#### Logging Configuration

```json
{
    "logging": {
        "tensorboard": {
            "enabled": true,
            "output_path": "./logs/tensorboard/",
            "job_name": "medical_ai_training"
        },
        "wandb": {
            "enabled": false,
            "project": "medical_ai_training",
            "name": "experiment_1"
        },
        "level": "INFO"                    // Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    }
}
```

## LoRA Configuration

### Basic LoRA Parameters

```yaml
# Basic LoRA Configuration
model_name: "microsoft/DialoGPT-medium"
base_model_name_or_path: "microsoft/DialoGPT-medium"
output_dir: "./lora_checkpoints"

# LoRA Parameters
lora_r: 16                    # Rank of the low-rank matrices
lora_alpha: 32                # Scaling parameter for LoRA
lora_dropout: 0.1             # Dropout probability for LoRA layers
lora_target_modules: [
    "q_proj",
    "v_proj", 
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# Training Parameters
learning_rate: 0.0002         # Learning rate for LoRA training
weight_decay: 0.001           # Weight decay
per_device_train_batch_size: 4 # Batch size per device
gradient_accumulation_steps: 8 # Gradient accumulation steps
num_train_epochs: 3           # Number of training epochs
max_steps: -1                # Maximum training steps (-1 for epoch-based)
logging_steps: 10            # Logging frequency
save_steps: 500              # Save frequency
eval_steps: 500              # Evaluation frequency
dataloader_num_workers: 4    # Number of data loader workers
warmup_steps: 100            # Number of warmup steps
lr_scheduler_type: "cosine"  # Learning rate scheduler type
```

### Advanced LoRA Parameters

```yaml
# Advanced LoRA Configuration
model_name: "facebook/opt-6.7b"
output_dir: "./advanced_lora_checkpoints"

# LoRA Parameters for Large Models
lora_r: 64                    # Higher rank for larger models
lora_alpha: 128               # Scaling parameter
lora_dropout: 0.05            # Lower dropout for large models
lora_target_modules: [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "mlp.down_proj", "mlp.up_proj"
]

# LoRA Advanced Settings
bias: "none"                  # Bias type ("none", "all", "lora_only")
task_type: "CAUSAL_LM"       # Task type
inference_mode: false        # Set to True for inference only

# LoRA Parameters for Specific Architectures
fan_in_fan_out: false        # Set to True if conv1d layer stores weight as (fan_in, fan_out)
modules_to_save: []          # Additional modules to save and load

# Quantization Settings
load_in_4bit: false          # Load model in 4-bit quantized format
load_in_8bit: true           # Load model in 8-bit quantized format
llm_int8_threshold: 6.0      # Int8 threshold for outlier detection
llm_int8_skip_modules: null  # Modules to skip 8-bit quantization
llm_int8_enable_fp32_cpu_offload: true # Enable FP32 CPU offload
bnb_4bit_quant_type: "nf4"  # 4-bit quantization type ("fp4", "nf4")
bnb_4bit_use_double_quant: true # Use double quantization
bnb_4bit_compute_dtype: "bfloat16" # Compute dtype for 4-bit quantization
```

### LoRA+ Configuration

```yaml
# LoRA+ Configuration
model_name: "microsoft/DialoGPT-medium"

# LoRA+ Parameters
lora_r: 8                    # LoRA rank (LoRA+ uses lower rank)
lora_alpha: 16               # LoRA scaling
lora_dropout: 0.1            # LoRA dropout
lora_target_modules: ["q_proj", "v_proj"]

# LoRA+ Specific Settings
lora_lr_ratio: 10.0          # LoRA-specific learning rate ratio
lora_weight_decay: 0.01      # LoRA-specific weight decay
lora_layer_replication: true # Enable layer replication
```

### AdaLoRA Configuration

```yaml
# AdaLoRA Configuration
model_name: "microsoft/DialoGPT-medium"

# AdaLoRA Parameters
lora_r: 8                    # Initial rank
lora_alpha: 32               # Scaling parameter
lora_dropout: 0.1            # Dropout probability
lora_target_modules: ["q_proj", "v_proj"]

# AdaLoRA Specific Settings
target_rank: 8               # Target rank for adaptation
init_r: 12                   # Initial rank
adaptation_pruning: true     # Enable rank adaptation
pruning_ratio: 0.1          # Pruning ratio
max_rank_per_module: 64     # Maximum rank per module
min_rank_per_module: 8      # Minimum rank per module
```

## Training Configuration

### Basic Training Parameters

```json
{
    "training": {
        "output_dir": "./checkpoints",
        "overwrite_output_dir": false,
        
        "num_train_epochs": 3,
        "max_steps": -1,
        "warmup_steps": 100,
        
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        
        "lr_scheduler_type": "cosine",
        "scheduler_kwargs": {},
        "warmup_ratio": 0.1,
        "decay_type": "cosine",
        "warmup_steps": 100,
        
        "logging_steps": 10,
        "logging_strategy": "steps",
        "save_steps": 500,
        "save_strategy": "steps",
        "save_total_limit": 3,
        
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "eval_delay": 0,
        "eval_accumulation_steps": null,
        
        "metric_for_best_model": "eval_loss",
        "load_best_model_at_end": true,
        "greater_is_better": false,
        
        "seed": 42,
        "data_seed": 42,
        "jit_mode_eval": false,
        "use_ipex": false,
        
        "torch_compile": false,
        "torch_compile_backend": "inductor",
        "torch_compile_mode": "default",
        
        "dispatch_batches": null,
        "split_batches": false,
        "include_inputs_for_metrics": false,
        
        "fp16": false,
        "bf16": true,
        "fp16_opt_level": "O1",
        "fp16_full_eval": false,
        "tf32": null,
        
        "gradient_checkpointing": true,
        "label_smoothing_factor": 0.0,
        
        "dropout_rate": 0.1,
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": true,
        "remove_unused_columns": true,
        
        "load_best_model_at_end": true,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": false,
        "ignore_data_skip": false
    }
}
```

### Advanced Training Parameters

```json
{
    "advanced_training": {
        "use_ipex": false,
        "torch_compile": false,
        "torch_compile_backend": "inductor",
        "torch_compile_mode": "default",
        
        "ddp_backend": "nccl",
        "ddp_broadcast_buffers": null,
        "ddp_find_unused_parameters": false,
        "ddp_bucket_cap_mb": null,
        
        "group_by_length": false,
        "length_column_name": "length",
        "report_to": ["tensorboard"],
        "run_name": "medical_ai_training",
        "disable_tqdm": false,
        
        "remove_unused_columns": true,
        "no_cuda": false,
        "use_mps_device": false,
        "use_cpu": false,
        
        "dataloader_pin_memory": true,
        "dataloader_num_workers": 4,
        "dataloader_prefetch_factor": 2,
        
        "past_index": -1,
        "run_name": null,
        "disable_tqdm": false,
        "remove_unused_columns": true,
        
        "ignore_data_skip": false,
        "sharded_ddp": false,
        "fsdp": false,
        "fsdp_min_num_params": 0,
        "fsdp_config": {},
        "deepspeed": null,
        
        "label_smoothing_factor": 0.0,
        "debug": "",
        "optim": "adamw_hf",
        "optim_args": null,
        
        "group_by_length": false,
        "length_column_name": "length",
        "auto_find_batch_size": false,
        "full_determinism": false,
        "torchdynamo": null,
        "ray_scope": "last"
    }
}
```

## Data Configuration

### Data Loading Parameters

```json
{
    "data": {
        "train_file": "./data/train.json",
        "validation_file": "./data/validation.json",
        "test_file": "./data/test.json",
        
        "max_length": 512,
        "padding_side": "right",  // "left" or "right"
        "truncation": true,
        
        "stride": 0,              // Overlap for long sequences
        "return_overflowing_tokens": false,
        "return_special_tokens_mask": true,
        "return_offsets_mapping": false,
        
        "load_from_cache_file": true,
        "cache_dir": "./data_cache",
        "overwrite_cache": false,
        
        "preprocessing_num_workers": 4,
        "preprocessing_processes": 4,
        
        "keep_in_memory": false,
        
        "column_names": ["text", "label"],
        "text_column_name": "text",
        "label_column_name": "label",
        "remove_columns": null,
        
        "label_names": null,
        "num_labels": null,
        "problem_type": null,
        
        "is_regression": false,
        "is_pair": false,
        "text_pair_second_column": false,
        
        "max_examples": null,
        "filter_by_length": true,
        "min_length": 10,
        "max_length": 512,
        
        "shuffle_seed": 42,
        "shuffle_buffer_size": 1000,
        "shuffle": true,
        
        "batch_size": 8,
        "drop_last": false,
        "num_workers": 4,
        "worker_init_fn": null,
        "collate_fn": null,
        "pin_memory": true,
        "persistent_workers": false
    }
}
```

### Medical Data Configuration

```json
{
    "medical_data": {
        "phi_protection": {
            "enabled": true,
            "method": "replacement",    // "replacement", "encryption", "anonymization"
            "preserve_structure": true,
            "phi_patterns": [
                "patient_name",
                "patient_id",
                "dob",
                "ssn",
                "phone",
                "email",
                "address"
            ],
            "redaction_patterns": [
                "PHI_[TYPE]_[ID]",
                "[REDACTED]"
            ],
            "custom_patterns": []
        },
        
        "medical_validation": {
            "validate_format": true,
            "validate_phi": true,
            "validate_medical_terms": true,
            "strict_mode": false,
            
            "required_fields": [
                "text",
                "label"
            ],
            
            "optional_fields": [
                "patient_id",
                "age",
                "gender",
                "diagnosis_date",
                "severity"
            ],
            
            "value_constraints": {
                "age": {"min": 0, "max": 150},
                "severity": {"values": [1, 2, 3, 4, 5]},
                "gender": {"values": ["M", "F", "O", "U"]}
            },
            
            "custom_validators": []
        },
        
        "data_augmentation": {
            "enabled": false,
            "methods": [
                "back_translation",
                "synonym_replacement",
                "random_insertion",
                "random_deletion"
            ],
            
            "back_translation": {
                "enabled": false,
                "source_lang": "en",
                "target_lang": "fr",
                "translator": "google"
            },
            
            "synonym_replacement": {
                "enabled": false,
                "replacement_ratio": 0.1,
                "medical_dictionary": "./data/medical_synonyms.json"
            }
        }
    }
}
```

## PHI Protection Configuration

### PHI Redaction Settings

```json
{
    "phi_protection": {
        "enabled": true,
        "strict_mode": true,
        "audit_logging": true,
        
        "redaction_method": "replacement",  // "replacement", "encryption", "anonymization"
        "preserve_structure": true,
        "preserve_sentiment": false,
        "preserve_context": false,
        
        "phi_patterns": {
            "patient_names": {
                "enabled": true,
                "method": "replace_with_token",
                "token_format": "PATIENT_[ID]",
                "case_sensitive": false
            },
            
            "patient_ids": {
                "enabled": true,
                "patterns": ["P[0-9]{5}", "PATIENT[0-9]{6}"],
                "method": "hash",
                "salt": "medical_ai_salt_2023"
            },
            
            "dates_of_birth": {
                "enabled": true,
                "method": "age_replacement",
                "preserve_year": false,
                "preserve_month": false
            },
            
            "ssn": {
                "enabled": true,
                "patterns": ["[0-9]{3}-[0-9]{2}-[0-9]{4}"],
                "method": "mask",
                "mask_format": "XXX-XX-####"
            },
            
            "phone_numbers": {
                "enabled": true,
                "patterns": [
                    "[0-9]{3}-[0-9]{3}-[0-9]{4}",
                    "\\([0-9]{3}\\) [0-9]{3}-[0-9]{4}"
                ],
                "method": "mask",
                "mask_format": "(XXX) XXX-XXXX"
            },
            
            "email_addresses": {
                "enabled": true,
                "method": "domain_replacement",
                "default_domain": "redacted.email"
            },
            
            "addresses": {
                "enabled": true,
                "method": "generalize",
                "keep_zip": true,
                "generalization_level": "city"
            }
        },
        
        "custom_patterns": [
            {
                "name": "medical_record_number",
                "patterns": ["MRN[0-9]{8}", "[0-9]{8}"],
                "method": "replace",
                "replacement": "MRN_########"
            }
        ],
        
        "validation": {
            "enable_validation": true,
            "validation_level": "strict",  // "strict", "moderate", "lenient"
            "false_positive_tolerance": 0.01,
            "false_negative_tolerance": 0.01
        },
        
        "audit": {
            "log_phi_detection": true,
            "log_phi_removal": true,
            "log_validation_results": true,
            "audit_file": "./logs/phi_audit.log",
            "audit_format": "json"
        }
    }
}
```

### PHI Validation Settings

```json
{
    "phi_validation": {
        "enabled": true,
        "strict_mode": false,
        "comprehensive_scan": true,
        
        "validation_categories": {
            "direct_identifiers": {
                "enabled": true,
                "check_patient_names": true,
                "check_patient_ids": true,
                "check_ssn": true,
                "check_email": true,
                "check_phone": true
            },
            
            "quasi_identifiers": {
                "enabled": true,
                "check_dob": true,
                "check_address": true,
                "check_zip_code": true,
                "check_age": true,
                "check_gender": true
            },
            
            "sensitive_attributes": {
                "enabled": true,
                "check_medical_conditions": true,
                "check_medications": true,
                "check_procedures": true,
                "check_lab_results": true
            }
        },
        
        "confidence_thresholds": {
            "high_confidence": 0.9,
            "medium_confidence": 0.7,
            "low_confidence": 0.5
        },
        
        "false_positive_handling": {
            "auto_approve_threshold": 0.1,
            "review_threshold": 0.05,
            "auto_reject_threshold": 0.02
        },
        
        "reporting": {
            "generate_reports": true,
            "report_format": "json",
            "report_file": "./reports/phi_validation_report.json",
            "include_statistics": true,
            "include_examples": true
        }
    }
}
```

## Evaluation Configuration

### Medical Evaluation Settings

```json
{
    "medical_evaluation": {
        "enabled": true,
        "comprehensive_evaluation": true,
        
        "clinical_metrics": {
            "precision": {
                "enabled": true,
                "average": "weighted",
                "pos_label": 1
            },
            "recall": {
                "enabled": true,
                "average": "weighted",
                "pos_label": 1
            },
            "f1_score": {
                "enabled": true,
                "average": "weighted",
                "pos_label": 1
            },
            "roc_auc": {
                "enabled": true,
                "multi_class": "ovr",
                "average": "weighted"
            },
            "confusion_matrix": {
                "enabled": true,
                "normalize": true,
                "save_plot": true
            },
            "classification_report": {
                "enabled": true,
                "output_dict": true,
                "save_to_file": true
            }
        },
        
        "medical_specific_metrics": {
            "clinical_accuracy": {
                "enabled": true,
                "accuracy_threshold": 0.85,
                "minimum_confidence": 0.7
            },
            "medical_sensitivity": {
                "enabled": true,
                "sensitivity_threshold": 0.9,
                "specificity_threshold": 0.8
            },
            "drug_interaction_detection": {
                "enabled": true,
                "interaction_threshold": 0.8
            },
            "diagnosis_consistency": {
                "enabled": true,
                "consistency_threshold": 0.9
            }
        },
        
        "bias_evaluation": {
            "enabled": true,
            "protected_attributes": [
                "age",
                "gender", 
                "ethnicity",
                "socioeconomic_status"
            ],
            "fairness_metrics": [
                "demographic_parity",
                "equalized_odds",
                "calibration"
            ],
            "bias_thresholds": {
                "demographic_parity_ratio": 0.8,
                "equalized_odds_ratio": 0.8,
                "calibration_ratio": 0.9
            }
        },
        
        "validation_approaches": {
            "cross_validation": {
                "enabled": true,
                "n_folds": 5,
                "stratified": true,
                "random_state": 42
            },
            "bootstrap_validation": {
                "enabled": false,
                "n_bootstrap": 1000,
                "confidence_interval": 0.95
            },
            "temporal_validation": {
                "enabled": false,
                "temporal_split": "monthly",
                "test_size": 0.2
            }
        }
    }
}
```

### Performance Evaluation

```json
{
    "performance_evaluation": {
        "timing_metrics": {
            "enabled": true,
            "measure_inference_time": true,
            "measure_training_time": true,
            "measure_memory_usage": true
        },
        
        "resource_metrics": {
            "gpu_memory": {
                "enabled": true,
                "peak_usage": true,
                "average_usage": true
            },
            "cpu_memory": {
                "enabled": true,
                "peak_usage": true,
                "average_usage": true
            },
            "disk_io": {
                "enabled": true,
                "read_bytes": true,
                "write_bytes": true
            }
        },
        
        "scalability_metrics": {
            "enabled": true,
            "batch_size_scaling": true,
            "sequence_length_scaling": true,
            "model_size_scaling": true
        },
        
        "comparison_baselines": {
            "enabled": true,
            "baseline_models": [
                "bert-base-uncased",
                "roberta-base",
                "distilbert-base"
            ],
            "comparison_metrics": [
                "accuracy",
                "f1_score",
                "inference_time",
                "memory_usage"
            ]
        }
    }
}
```

## Serving Configuration

### FastAPI Model Serving

```json
{
    "model_serving": {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1,
            "max_connections": 1000,
            "timeout_keep_alive": 5,
            "max_request_size": 10485760,  // 10MB
            "max_request_time": 30,        // seconds
            "max_response_time": 30,       // seconds
            "compression": "gzip",
            "cors_enabled": true,
            "cors_origins": ["*"],
            "cors_methods": ["GET", "POST", "OPTIONS"],
            "cors_headers": ["*"]
        },
        
        "model": {
            "model_path": "./models/medical_ai_model",
            "model_type": "causal_lm",
            "load_in_8bit": false,
            "load_in_4bit": false,
            "device": "auto",
            "torch_dtype": "auto",
            "trust_remote_code": false,
            "revision": "main",
            "use_auth_token": false,
            "cache_dir": "./cache",
            
            "max_length": 512,
            "max_new_tokens": 100,
            "min_length": 1,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "length_penalty": 1.0,
            "num_beams": 1,
            "do_sample": true,
            "early_stopping": false,
            "pad_token_id": null,
            "eos_token_id": null,
            
            "stop_sequences": ["\n\n", "Human:", "Assistant:"],
            "bad_words": ["[BAD_WORD]"],
            
            "gradient_checkpointing": false,
            "low_cpu_mem_usage": true,
            "torch_compile": false
        },
        
        "security": {
            "enable_phi_protection": true,
            "phi_validation_required": true,
            "rate_limiting": {
                "enabled": true,
                "requests_per_minute": 60,
                "burst_requests": 10
            },
            "authentication": {
                "enabled": false,
                "method": "api_key",  // "api_key", "jwt", "oauth"
                "api_key_header": "X-API-Key",
                "jwt_secret_key": "your-secret-key"
            },
            "input_validation": {
                "max_text_length": 10000,
                "allowed_content_types": ["text/plain"],
                "block_suspicious_input": true
            },
            "output_filtering": {
                "enable_phi_filtering": true,
                "medical_terminology_check": true,
                "inappropriate_content_filter": true
            }
        },
        
        "monitoring": {
            "logging": {
                "enabled": true,
                "level": "INFO",
                "format": "json",
                "file": "./logs/serving.log",
                "max_size": "100MB",
                "backup_count": 5
            },
            "metrics": {
                "enabled": true,
                "prometheus_endpoint": "/metrics",
                "metrics_interval": 60,
                "track_requests": true,
                "track_latency": true,
                "track_errors": true
            },
            "health_checks": {
                "enabled": true,
                "health_endpoint": "/health",
                "readiness_endpoint": "/ready",
                "liveness_endpoint": "/live"
            }
        },
        
        "caching": {
            "enabled": true,
            "cache_size": 1000,
            "cache_ttl": 3600,  // seconds
            "cache_type": "memory",  // "memory", "redis", "disk"
            "redis_url": "redis://localhost:6379"
        }
    }
}
```

### Model Loading Configuration

```json
{
    "model_loading": {
        "base_model": {
            "model_name_or_path": "bert-base-uncased",
            "revision": "main",
            "use_auth_token": false,
            "cache_dir": "./cache",
            "trust_remote_code": false
        },
        
        "peft_config": {
            "peft_type": "LORA",
            "base_model_name_or_path": "bert-base-uncased",
            "task_type": "CAUSAL_LM",
            
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj"],
            
            "bias": "none",
            "fan_in_fan_out": false,
            "modules_to_save": null,
            
            "layers_to_transform": null,
            "layers_pattern": null,
            "rank_pattern": {},
            "alpha_pattern": {}
        },
        
        "quantization": {
            "load_in_8bit": false,
            "load_in_4bit": false,
            "llm_int8_threshold": 6.0,
            "llm_int8_skip_modules": null,
            "llm_int8_enable_fp32_cpu_offload": true,
            
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": true,
            "bnb_4bit_compute_dtype": "bfloat16"
        },
        
        "device_map": {
            "enabled": true,
            "strategy": "auto",  // "auto", "sequential", "balanced", "balanced_low_0"
            "max_memory": null,  // {"cuda:0": "4GB", "cpu": "8GB"}
            "no_split_module_classes": ["BertLayer"],
            "offload_folder": "./offload",
            "offload_index": null
        },
        
        "torch_config": {
            "torch_dtype": "auto",
            "torch_dtype_mapping": {
                "fp32": "torch.float32",
                "fp16": "torch.float16",
                "bf16": "torch.bfloat16"
            },
            "trust_remote_code": false,
            "use_safetensors": false,
            "use_flash_attention": false
        }
    }
}
```

## Monitoring Configuration

### TensorBoard Configuration

```json
{
    "tensorboard": {
        "enabled": true,
        "output_dir": "./logs/tensorboard",
        "log_level": "INFO",
        
        "default_writer": "tensorboard",
        "flush_secs": 30,
        "max_queue": 10,
        
        "write_to_disk": true,
        "overwrite": false,
        
        "config_experiments": false,
        
        "track_log_dir": "./logs",
        "job_name": "medical_ai_training",
        
        "metrics": {
            "log_training_loss": true,
            "log_validation_loss": true,
            "log_learning_rate": true,
            "log_gradient_norm": true,
            "log_model_parameters": true,
            "log_optimization_steps": true,
            "log_memory_usage": true,
            "log_communication_overhead": true,
            "log_step_time": true,
            "log_throughput": true
        },
        
        "histograms": {
            "enabled": true,
            "track_weight_histograms": true,
            "track_activation_histograms": true,
            "histogram_freq": 100,
            "max_bins": 100
        },
        
        "images": {
            "enabled": false,
            "log_model_graph": false,
            "log_attention_maps": false,
            "log_confusion_matrix": false
        }
    }
}
```

### Weights & Biases Configuration

```json
{
    "wandb": {
        "enabled": false,
        "project": "medical_ai_training",
        "entity": null,
        "name": null,
        "notes": null,
        "tags": ["medical", "ai", "training"],
        
        "save_code": true,
        "save_code_programmatically": true,
        "save_init_kwargs": true,
        "save_unverified_code": false,
        
        "monitor_gym": true,
        "resume": "allow",
        "force": false,
        "reinit": false,
        "update": false,
        "gradio": false,
        
        "config": {
            "model_architecture": "bert-base",
            "training_strategy": "deepspeed_zero_stage2",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "epochs": 3,
            "dataset": "medical_qa",
            "phi_protection": true
        },
        
        "settings": {
            "start_method": "thread",
            "sweep_id": null,
            "anonymous": "never",
            "run_name": null,
            "group": null,
            "job_type": null
        }
    }
}
```

### Custom Monitoring Configuration

```json
{
    "custom_monitoring": {
        "memory_profiling": {
            "enabled": true,
            "profile_memory": true,
            "memory_interval": 100,  // steps
            "save_memory_plots": true,
            "memory_log_file": "./logs/memory_profile.log"
        },
        
        "performance_monitoring": {
            "enabled": true,
            "track_step_time": true,
            "track_communication_time": true,
            "track_data_loading_time": true,
            "performance_log_interval": 50
        },
        
        "model_validation": {
            "enabled": true,
            "validate_model_compatibility": true,
            "validate_phi_protection": true,
            "validate_medical_terms": true,
            "validation_interval": 500
        },
        
        "custom_metrics": {
            "enabled": true,
            "medical_accuracy": {
                "enabled": true,
                "calculation_frequency": "validation",
                "threshold": 0.85
            },
            "clinical_bias": {
                "enabled": true,
                "protected_attributes": ["age", "gender", "ethnicity"],
                "bias_threshold": 0.1
            },
            "phi_leakage": {
                "enabled": true,
                "detection_sensitivity": "high",
                "tolerance_rate": 0.001
            }
        },
        
        "alerting": {
            "enabled": false,
            "alert_conditions": {
                "high_memory_usage": {
                    "enabled": true,
                    "threshold": 0.9,
                    "duration": 300
                },
                "low_accuracy": {
                    "enabled": true,
                    "threshold": 0.7,
                    "duration": 1800
                },
                "phi_leakage_detected": {
                    "enabled": true,
                    "immediate": true
                }
            },
            "notification_channels": {
                "email": {
                    "enabled": false,
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "recipients": ["admin@example.com"]
                },
                "slack": {
                    "enabled": false,
                    "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
                    "channel": "#ai-training"
                }
            }
        }
    }
}
```

This completes the comprehensive Configuration Reference. Each section provides detailed documentation for all configuration options, their parameters, usage guidelines, and optimization recommendations for the Medical AI Training Pipeline.
