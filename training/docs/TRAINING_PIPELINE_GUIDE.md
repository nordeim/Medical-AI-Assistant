# Training Pipeline Guide

Complete step-by-step guide for training medical AI models with the Medical AI Training Pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Model Configuration](#model-configuration)
5. [Training Execution](#training-execution)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)
7. [Model Saving and Deployment](#model-saving-and-deployment)
8. [Advanced Training Scenarios](#advanced-training-scenarios)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **Operating System**: Ubuntu 18.04+ or equivalent
- **Python**: 3.8+ (recommended 3.10)
- **RAM**: 16GB system memory
- **GPU**: NVIDIA GPU with 8GB VRAM (RTX 3070 or better)
- **Storage**: 100GB available space

#### Recommended Requirements
- **Operating System**: Ubuntu 20.04+
- **RAM**: 64GB+ system memory
- **GPU**: NVIDIA A100 or RTX 4090 (24GB+ VRAM)
- **Storage**: 500GB+ NVMe SSD

#### Distributed Training Requirements
- **Network**: High-bandwidth interconnects (InfiniBand recommended)
- **Storage**: Shared filesystem (NFS, Lustre, or equivalent)
- **Configuration**: Identical environment across all nodes

### Software Dependencies

```bash
# Core PyTorch and CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# DeepSpeed and distributed training
pip install deepspeed

# Transformers and datasets
pip install transformers datasets accelerate

# LoRA and PEFT
pip install peft bitsandbytes

# Medical AI specific
pip install fastapi uvicorn
pip install scikit-learn pandas numpy matplotlib seaborn

# Development tools
pip install pytest black flake8 mypy
```

### Hardware Verification

```bash
# Check GPU availability
nvidia-smi

# Verify CUDA version
nvcc --version

# Test PyTorch with CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test DeepSpeed installation
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
```

## Environment Setup

### 1. Virtual Environment Creation

```bash
# Using Python venv (recommended)
python -m venv medical_ai_env
source medical_ai_env/bin/activate  # Linux/Mac
# medical_ai_env\Scripts\activate  # Windows

# Verify activation
python --version
which python
```

### 2. Repository Setup

```bash
# Clone repository (if using Git)
git clone https://github.com/your-org/medical-ai-training.git
cd medical-ai-training

# Or download and extract
# wget https://releases/medical-ai-training-v1.0.zip
# unzip medical-ai-training-v1.0.zip
```

### 3. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install LoRA-specific requirements
pip install -r lora_requirements.txt

# Install development requirements (optional)
pip install -r requirements-dev.txt
```

### 4. Environment Validation

```bash
# Run setup validation script
python test_setup.py

# Expected output should show all checks passing
```

### 5. Environment Variables

Create `.env` file in the project root:

```bash
# DeepSpeed Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=4

# Model and Data Paths
export MODEL_CACHE_DIR=./models
export DATA_CACHE_DIR=./data_cache
export OUTPUT_DIR=./outputs

# Distributed Training (multi-node)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0

# PHI Protection
export PHI_PROTECTION_LEVEL=strict
export PHI_AUDIT_LOGGING=true

# Monitoring
export TENSORBOARD_LOG_DIR=./logs/tensorboard
export WANDB_PROJECT=medical_ai_training
```

## Data Preparation

### 1. Data Format Standards

The training pipeline supports multiple data formats:

#### JSON Format (Recommended)
```json
{
  "data": [
    {
      "id": "sample_001",
      "text": "Patient presents with symptoms of...",
      "label": "diagnosis",
      "metadata": {
        "patient_id": "P12345",
        "age": 45,
        "gender": "M"
      }
    }
  ]
}
```

#### CSV Format
```csv
id,text,label,patient_id,age,gender
sample_001,Patient presents with symptoms...,diagnosis,P12345,45,M
```

#### HuggingFace Dataset
```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "text": ["sample text 1", "sample text 2"],
    "label": ["label1", "label2"],
    "patient_id": ["P123", "P456"]
})
```

### 2. Data Preprocessing Steps

#### Basic Preprocessing
```python
from utils.data_utils import DataProcessor

processor = DataProcessor(
    max_length=512,
    tokenizer_name="bert-base-uncased",
    enable_phi_protection=True
)

# Load and preprocess data
train_dataset, val_dataset = processor.load_and_preprocess(
    train_file="data/train.json",
    validation_file="data/val.json"
)
```

#### PHI Protection Setup
```python
from utils.phi_redactor import PHIRedactor
from utils.phi_validator import PHIValidator

# Initialize PHI protection
phi_redactor = PHIRedactor(
    redaction_method="replacement",
    preserve_structure=True
)

# Validate PHI protection
phi_validator = PHIValidator(strict_mode=True)

# Process data with PHI protection
def preprocess_with_phi_protection(text):
    # Validate PHI presence
    if phi_validator.contains_phi(text):
        # Redact PHI
        cleaned_text = phi_redactor.redact_phi(text)
        return cleaned_text
    return text

# Apply to dataset
protected_dataset = dataset.map(
    lambda x: {"text": preprocess_with_phi_protection(x["text"])}
)
```

### 3. Data Validation

#### Medical Data Validation
```python
from utils.compliance_checker import ComplianceChecker

compliance_checker = ComplianceChecker()

# Validate data compliance
validation_results = compliance_checker.validate_dataset(
    dataset=train_dataset,
    check_phi=True,
    check_format=True,
    check_medical_terms=True
)

print(f"Validation passed: {validation_results['passed']}")
if validation_results['violations']:
    print(f"Violations: {validation_results['violations']}")
```

#### Data Splitting
```python
from sklearn.model_selection import train_test_split

# Stratified split for medical data
train_data, test_data = train_test_split(
    dataset,
    test_size=0.1,
    stratify=dataset['label'],
    random_state=42
)

# Additional validation split
train_data, val_data = train_test_split(
    train_data,
    test_size=0.1,
    stratify=train_data['label'],
    random_state=42
)
```

### 4. Data Loading Pipeline

#### Basic Data Loading
```python
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define data processing function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Setup data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)
```

## Model Configuration

### 1. Base Model Selection

#### Recommended Models for Medical AI
```python
MODELS = {
    "small": "microsoft/DialoGPT-medium",
    "medium": "bert-base-uncased", 
    "large": "bert-large-uncased",
    "xlarge": "facebook/opt-2.7b",
    "medical": "Clinical-AI-Apollo/Medical-Llama3-8B"
}

# Load base model
from transformers import AutoModelForCausalLM

model_name = "bert-base-uncased"
model = AutoModelForCausalLM.from_pretrained(model_name)
```

### 2. LoRA Configuration

#### Basic LoRA Setup
```yaml
# configs/lora_config.yaml
model_name: "microsoft/DialoGPT-medium"
output_dir: "./lora_checkpoints"

# LoRA Parameters
lora_r: 16                    # Rank of the low-rank matrices
lora_alpha: 32                # Scaling parameter
lora_dropout: 0.1             # Dropout probability
lora_target_modules: [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training Parameters
learning_rate: 0.0002
weight_decay: 0.001
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
num_train_epochs: 3
logging_steps: 10
save_steps: 500
eval_steps: 500
dataloader_num_workers: 4
warmup_steps: 100
lr_scheduler_type: "cosine"

# Data Parameters
max_length: 512
train_file: "./data/train.json"
validation_file: "./data/validation.json"

# PHI Protection
enable_phi_protection: true
phi_redaction_method: "replacement"
phi_validation_strict: true

# Performance Optimizations
fp16: true
gradient_checkpointing: true
deepspeed: "configs/deepspeed_config.json"
```

#### Advanced LoRA Configuration
```yaml
# configs/advanced_lora_config.yaml
model_name: "facebook/opt-2.7b"

# LoRA Parameters for Large Models
lora_r: 64                    # Higher rank for large models
lora_alpha: 128               # Scaling parameter
lora_dropout: 0.05            # Lower dropout
lora_target_modules: [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "mlp.down_proj", "mlp.up_proj"
]

# Training Parameters
learning_rate: 0.0001         # Lower learning rate
weight_decay: 0.01
per_device_train_batch_size: 2
gradient_accumulation_steps: 16  # Higher accumulation
num_train_epochs: 5
logging_steps: 5
save_steps: 100
eval_steps: 100

# Optimization Settings
deepspeed: "configs/large_model_stage3_config.json"
gradient_checkpointing: true
max_grad_norm: 1.0

# Memory Optimization
use_8bit: true               # 8-bit quantization
cpu_offloading: true          # CPU offloading for parameters
```

### 3. DeepSpeed Configuration

#### Single GPU Setup
```json
{
    "zero_optimization": {
        "stage": 1,
        "offload_optimizer": false,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    },
    "bfloat16": {
        "enabled": true
    },
    "zero_allow_untested_optimizer": true,
    "train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": true,
    "logging": {
        "tensorboard": {
            "enabled": true,
            "output_path": "./logs/tensorboard/"
        }
    }
}
```

#### Multi-GPU Setup
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "reduce_scatter": true,
        "allgather_bucket_size": 2e8,
        "reduce_bucket_size": 2e8,
        "sub_group_size": 1e9,
        "prefetch_bucket_size": 5e7,
        "param_persistence_threshold": 4e5,
        "stage3_prefetch_stream_reserve_memory": 5e7
    },
    "bfloat16": {
        "enabled": true
    },
    "train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "save_interval": 100,
    "logging": {
        "tensorboard": {
            "enabled": true,
            "output_path": "./logs/tensorboard/"
        }
    }
}
```

## Training Execution

### 1. Basic Training

#### Simple Training Script
```python
from scripts.train_distributed import main
import argparse

def run_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    run_training()
```

#### Training Command
```bash
# Single GPU training
python scripts/train_distributed.py \
    --config configs/single_node_config.json \
    --model_name bert-base-uncased \
    --dataset_path ./data/train.json \
    --output_dir ./checkpoints \
    --epochs 3 \
    --learning_rate 5e-5

# Multi-GPU training
torchrun \
    --nproc_per_node=4 \
    scripts/train_distributed.py \
    --config configs/multi_node_config.json \
    --model_name bert-large-uncased \
    --dataset_path ./data/large_dataset.json
```

### 2. LoRA Training

#### LoRA Training Script
```python
from scripts.train_lora import LoRATrainer

def train_with_lora():
    trainer = LoRATrainer(
        model_name="microsoft/DialoGPT-medium",
        config_path="configs/lora_config.yaml",
        output_dir="./lora_checkpoints"
    )
    
    # Setup model and tokenizer
    trainer.setup_model()
    
    # Load data
    train_dataset, val_dataset = trainer.load_data()
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model()

if __name__ == "__main__":
    train_with_lora()
```

#### LoRA Training Command
```bash
# Basic LoRA training
python scripts/train_lora.py \
    --config configs/lora_config.yaml

# Advanced LoRA training
python scripts/train_lora.py \
    --config configs/advanced_lora_config.yaml \
    --resume_from_checkpoint ./lora_checkpoints/checkpoint-1000
```

### 3. Distributed Training

#### Multi-Node Setup
```bash
# Master node (node 0)
export MASTER_ADDR=192.168.1.100
export WORLD_SIZE=8
export RANK=0

torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/multi_node_config.json \
    --model_name bert-large-uncased \
    --dataset_path ./data/distributed_train.json

# Worker node (node 1)
export MASTER_ADDR=192.168.1.100
export WORLD_SIZE=8
export RANK=4

torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/multi_node_config.json \
    --model_name bert-large-uncased \
    --dataset_path ./data/distributed_train.json
```

#### SLURM Cluster Setup
```bash
#!/bin/bash
# sbatch script: launch_slurm.sbatch

#SBATCH --job-name=medical_ai_training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Load modules
module load cuda/11.8
module load pytorch/2.0.0

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO

# Launch training
srun torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_PROCID \
    --master_addr=$(scontrol show job $SLURM_JOBID | grep BatchHost | awk '{print $3}' | sed 's/BatchHost=//') \
    --master_port=29500 \
    scripts/train_distributed.py \
    --config configs/multi_node_config.json \
    --model_name bert-large-uncased \
    --dataset_path $DATASET_PATH
```

### 4. Resume Training

#### From Checkpoint
```bash
# Resume from checkpoint
python scripts/train_distributed.py \
    --config configs/single_node_config.json \
    --model_name bert-base-uncased \
    --dataset_path ./data/train.json \
    --resume_from_checkpoint ./checkpoints/checkpoint-1000 \
    --output_dir ./checkpoints
```

#### From Best Checkpoint
```python
from utils.deepspeed_utils import CheckpointManager

checkpoint_manager = CheckpointManager(save_dir="./checkpoints")

# Find best checkpoint based on validation loss
best_checkpoint = checkpoint_manager.find_best_checkpoint("eval_loss", ascending=True)

# Resume training
import subprocess
subprocess.run([
    "python", "scripts/train_distributed.py",
    "--config", "configs/single_node_config.json",
    "--model_name", "bert-base-uncased",
    "--dataset_path", "./data/train.json",
    "--resume_from_checkpoint", best_checkpoint["path"]
])
```

## Monitoring and Evaluation

### 1. Training Monitoring

#### TensorBoard Setup
```bash
# Install TensorBoard
pip install tensorboard

# Start TensorBoard
tensorboard --logdir=./logs/tensorboard --port=6006

# Access at http://localhost:6006
```

#### Real-time Monitoring Script
```python
from utils.performance_benchmark import PerformanceMonitor

class TrainingMonitor:
    def __init__(self):
        self.perf_monitor = PerformanceMonitor()
        self.memory_profiler = MemoryProfiler()
        
    def start_monitoring(self):
        self.perf_monitor.start_monitoring()
        self.memory_profiler.start_monitoring()
        
    def log_training_step(self, global_step, loss, learning_rate):
        self.perf_monitor.log_step(global_step, batch_idx=1, batch_size=8)
        self.perf_monitor.log_metric("loss", loss, global_step)
        self.perf_monitor.log_metric("learning_rate", learning_rate, global_step)
        
    def get_summary(self):
        return {
            "performance": self.perf_monitor.get_summary(),
            "memory": self.memory_profiler.get_summary()
        }

# Usage in training loop
monitor = TrainingMonitor()
monitor.start_monitoring()

for step in range(total_steps):
    # Training step...
    monitor.log_training_step(step, loss, learning_rate)
```

### 2. Evaluation During Training

#### Medical-Specific Evaluation
```python
from utils.clinical_evaluation import ClinicalEvaluator

class MedicalTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = ClinicalEvaluator(model, tokenizer)
        
    def evaluate_step(self, eval_dataset, step):
        # Run medical-specific evaluation
        results = self.evaluator.evaluate_all_metrics(eval_dataset)
        
        # Log results
        for metric, value in results.items():
            print(f"Step {step}: {metric} = {value:.4f}")
            
        return results
    
    def evaluate_clinical_accuracy(self, test_dataset):
        # Calculate clinical accuracy metrics
        clinical_results = self.evaluator.calculate_clinical_accuracy(test_dataset)
        
        # Print clinical-specific metrics
        print("\nClinical Evaluation Results:")
        print(f"Precision: {clinical_results['precision']:.4f}")
        print(f"Recall: {clinical_results['recall']:.4f}")
        print(f"F1-Score: {clinical_results['f1']:.4f}")
        print(f"AUC-ROC: {clinical_results['auc_roc']:.4f}")
        
        return clinical_results
```

### 3. Validation Strategies

#### Cross-Validation for Medical Data
```python
from sklearn.model_selection import StratifiedKFold
from utils.clinical_evaluation import ClinicalEvaluator

def cross_validate_model(model, dataset, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, dataset['label'])):
        print(f"\nFold {fold + 1}/{n_folds}")
        
        # Create fold datasets
        fold_train = dataset.select(train_idx)
        fold_val = dataset.select(val_idx)
        
        # Train on fold
        trainer = MedicalTrainer(model, tokenizer, config)
        trainer.train_on_fold(fold_train)
        
        # Evaluate on fold
        evaluator = ClinicalEvaluator(model, tokenizer)
        fold_results.append(evaluator.evaluate_all_metrics(fold_val))
    
    # Calculate average results
    avg_results = {}
    for metric in fold_results[0].keys():
        avg_results[metric] = np.mean([result[metric] for result in fold_results])
    
    return avg_results, fold_results
```

## Model Saving and Deployment

### 1. Model Checkpointing

#### Automatic Checkpointing
```json
{
    "checkpoint": {
        "save_interval": 500,
        "save_total_limit": 10,
        "async_save": false,
        "load_universal": false,
        "use_pin_memory": true
    },
    "logging": {
        "tensorboard": {
            "enabled": true,
            "output_path": "./logs/tensorboard/"
        }
    }
}
```

#### Manual Checkpointing
```python
from utils.deepspeed_utils import CheckpointManager

checkpoint_manager = CheckpointManager(save_dir="./checkpoints")

# Save checkpoint manually
checkpoint_manager.save_checkpoint(
    engine=engine,
    epoch=epoch,
    step=step,
    metrics={"loss": loss, "accuracy": accuracy},
    save_path=f"./checkpoints/checkpoint-{step}"
)

# Save model for deployment
def save_model_for_deployment(model, tokenizer, output_dir):
    # Save base model + LoRA adapters
    model.save_pretrained(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    
    # Save additional metadata
    metadata = {
        "model_type": "medical_ai_lora",
        "base_model": "bert-base-uncased",
        "training_config": config,
        "phi_protection_enabled": True
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
```

### 2. Model Export

#### Export for Inference
```python
def export_model_for_inference(model_path, output_path):
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load LoRA configuration
    config = PeftConfig.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    # Save for inference
    model.save_pretrained(f"{output_path}/final_model")
    tokenizer.save_pretrained(f"{output_path}/final_model")
    
    # Create serving configuration
    serving_config = {
        "model_path": f"{output_path}/final_model",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "phi_protection_enabled": True
    }
    
    with open(f"{output_path}/serving_config.json", "w") as f:
        json.dump(serving_config, f, indent=2)
```

### 3. Model Serving Setup

#### FastAPI Serving
```python
from utils.model_serving import ModelServer
from fastapi import FastAPI

def setup_model_serving(model_path, port=8000):
    # Initialize model server
    server = ModelServer(
        model_path=f"{model_path}/final_model",
        config_path=f"{model_path}/serving_config.json"
    )
    
    # Load model
    server.load_model()
    
    # Create FastAPI app
    app = FastAPI(
        title="Medical AI Assistant",
        description="HIPAA-compliant medical AI model serving",
        version="1.0.0"
    )
    
    # Add health check endpoint
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": server.is_model_loaded()}
    
    # Add inference endpoint
    @app.post("/predict")
    async def predict(request: dict):
        text = request.get("text", "")
        
        # Validate input
        if not text:
            return {"error": "No text provided"}
        
        # Run inference
        result = server.generate(
            prompt=text,
            max_length=512,
            do_sample=True,
            temperature=0.7
        )
        
        return {
            "input": text,
            "output": result,
            "phi_protected": True
        }
    
    return app

# Start server
if __name__ == "__main__":
    import uvicorn
    
    app = setup_model_serving("./outputs/medical_model")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Advanced Training Scenarios

### 1. Multi-Task Learning

#### Multi-Task Configuration
```yaml
# configs/multitask_config.yaml
model_name: "bert-base-uncased"
output_dir: "./multitask_checkpoints"

tasks:
  - name: "diagnosis_classification"
    dataset: "./data/diagnosis.json"
    loss_weight: 1.0
    num_labels: 50
  
  - name: "severity_prediction"
    dataset: "./data/severity.json"
    loss_weight: 0.5
    num_labels: 4
  
  - name: "treatment_recommendation"
    dataset: "./data/treatment.json"
    loss_weight: 0.8
    max_length: 256

# Training configuration
learning_rate: 0.0001
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
num_train_epochs: 5
warmup_steps: 1000
weight_decay: 0.01

# Multi-task specific settings
task_balancing_strategy: "dynamic"
loss_aggregation: "weighted_sum"
```

### 2. Continual Learning

#### Continual Learning Setup
```python
from utils.model_utils import ContinualLearningManager

class ContinualTrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.continual_manager = ContinualLearningManager(model)
        
    def train_new_task(self, task_name, task_data, previous_tasks=None):
        # Load previous task models
        if previous_tasks:
            self.continual_manager.load_previous_tasks(previous_tasks)
        
        # Setup task-specific adapters
        task_adapter = self.continual_manager.create_task_adapter(task_name)
        
        # Train on new task
        trainer = LoRATrainer(
            model=self.model,
            config={
                "lora_r": 16,
                "learning_rate": 0.0002,
                "per_device_train_batch_size": 4
            }
        )
        
        trainer.train_on_dataset(task_data)
        
        # Save task-specific knowledge
        self.continual_manager.save_task_knowledge(task_name, task_adapter)
        
        return task_adapter
```

### 3. Federated Learning

#### Federated Learning Setup
```python
from utils.model_utils import FederatedTrainer

class FederatedMedicalTrainer:
    def __init__(self, central_model, config):
        self.central_model = central_model
        self.federated_trainer = FederatedTrainer(central_model, config)
        
    def run_federated_round(self, client_data_splits):
        # Send model to clients
        self.federated_trainer.distribute_model_to_clients()
        
        client_updates = []
        
        # Train on each client
        for client_id, client_data in enumerate(client_data_splits):
            # Client training
            client_update = self.federated_trainer.client_train(
                client_id=client_id,
                local_epochs=3,
                client_data=client_data
            )
            
            client_updates.append(client_update)
        
        # Aggregate updates
        self.federated_trainer.aggregate_updates(client_updates)
        
        return self.central_model
```

This completes the comprehensive Training Pipeline Guide. The guide covers all aspects from initial setup through advanced training scenarios, providing detailed instructions and examples for training medical AI models safely and efficiently.
