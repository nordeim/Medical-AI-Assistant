#!/bin/bash

# Medical AI Assistant Training Environment Setup Script
# Author: Medical AI Assistant Team
# Date: 2025-11-04
# Description: Sets up the complete training environment for medical AI models

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} Medical AI Training Setup${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

# Check system requirements
check_system_requirements() {
    print_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_success "Operating System: $OS"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        REQUIRED_VERSION="3.8"
        if [[ $(echo "$PYTHON_VERSION >= $REQUIRED_VERSION" | bc -l) == 1 ]]; then
            print_success "Python version: $PYTHON_VERSION"
        else
            print_error "Python 3.8 or higher required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        PIP_VERSION=$(pip3 --version | cut -d' ' -f2)
        print_success "pip version: $PIP_VERSION"
    else
        print_error "pip3 not found. Please install pip."
        exit 1
    fi
    
    # Check available disk space
    DISK_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    MIN_SPACE=50
    if [[ $DISK_SPACE -gt $MIN_SPACE ]]; then
        print_success "Available disk space: ${DISK_SPACE}GB"
    else
        print_warning "Low disk space: ${DISK_SPACE}GB. At least ${MIN_SPACE}GB recommended."
    fi
    
    # Check for NVIDIA GPU (optional)
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "NVIDIA GPU detected: $GPU_INFO"
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
        print_info "CUDA driver version: $CUDA_VERSION"
    else
        print_warning "NVIDIA GPU not detected. Training will use CPU (much slower)."
    fi
}

# Setup Python virtual environment
setup_virtual_environment() {
    print_info "Setting up Python virtual environment..."
    
    VENV_NAME="medical_ai_env"
    
    # Check if virtual environment already exists
    if [[ -d "$VENV_NAME" ]]; then
        print_warning "Virtual environment '$VENV_NAME' already exists."
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$VENV_NAME"
        else
            print_info "Using existing virtual environment."
            source "$VENV_NAME/bin/activate"
            return
        fi
    fi
    
    # Create virtual environment
    python3 -m venv "$VENV_NAME"
    print_success "Virtual environment created: $VENV_NAME"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    print_success "Virtual environment activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip
    print_success "pip upgraded to: $(pip --version | cut -d' ' -f2)"
}

# Install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found in current directory"
        exit 1
    fi
    
    # Install dependencies
    pip install -r requirements.txt
    print_success "All dependencies installed successfully"
    
    # Verify critical packages
    print_info "Verifying critical packages..."
    python3 -c "
import torch
import transformers
import datasets
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')
print('✓ All critical packages verified')
"
    print_success "Package verification completed"
}

# Setup directories
setup_directories() {
    print_info "Creating training directory structure..."
    
    DIRECTORIES=(
        "configs"
        "scripts"
        "models"
        "data"
        "utils"
        "evaluation"
        "outputs"
        "checkpoints"
        "logs"
        "logs/tensorboard"
        "logs/wandb"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    done
}

# Setup environment file
setup_environment_file() {
    print_info "Setting up environment configuration..."
    
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            print_success "Created .env file from .env.example"
            print_warning "Please update .env file with your specific configuration"
            print_info "Key settings to update:"
            echo "  - WANDB_PROJECT and WANDB_API_KEY (for experiment tracking)"
            echo "  - HUGGINGFACE_HUB_TOKEN (for model access)"
            echo "  - CUDA_VISIBLE_DEVICES (GPU selection)"
            echo "  - MODEL_NAME (base model selection)"
        else
            print_error ".env.example not found"
        fi
    else
        print_info ".env file already exists"
    fi
}

# Download base model
download_base_model() {
    print_info "Setting up base model..."
    
    # Source environment variables if .env exists
    if [[ -f ".env" ]]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    # Model name from environment or default
    MODEL_NAME=${MODEL_NAME:-"microsoft/DialoGPT-medium"}
    
    print_info "Downloading base model: $MODEL_NAME"
    
    # Create Python script to download model
    cat > /tmp/download_model.py << 'EOF'
import os
import sys
from transformers import AutoTokenizer, AutoModel

model_name = sys.argv[1] if len(sys.argv) > 1 else "microsoft/DialoGPT-medium"

try:
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer downloaded successfully")
    
    print(f"Downloading model for {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    print("✓ Model downloaded successfully")
    
    print("Base model setup completed!")
    
except Exception as e:
    print(f"Error downloading model: {e}")
    print("This is normal for large models - they will be downloaded during training")
EOF
    
    # Try to download model (may fail for large models)
    python3 /tmp/download_model.py "$MODEL_NAME" || print_warning "Model download failed - will download during training"
    rm -f /tmp/download_model.py
}

# Create sample configurations
create_sample_configs() {
    print_info "Creating sample configuration files..."
    
    # Create DeepSpeed configuration
    cat > configs/deepspeed_config.json << 'EOF'
{
    "bf16": {
        "enabled": false
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter_bucket_size": 5e8,
        "sub_group_size": 1e9,
        "gather_16bit_weights_on_fp32_gpu": true
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "train_batch_size": 4,
    "train_micro_batch_size_per_gpu": 4,
    "wall_clock_breakdown": false
}
EOF
    print_success "Created DeepSpeed configuration"
    
    # Create training configuration template
    cat > configs/training_config.yaml << 'EOF'
# Training Configuration
model:
  name: "microsoft/DialoGPT-medium"
  type: "causal_lm"
  max_seq_length: 512

training:
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  num_epochs: 3
  warmup_steps: 100
  max_steps: 1000
  
  # Logging
  logging_steps: 100
  evaluation_steps: 500
  save_steps: 1000
  
  # Optimization
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1e-8
  
  # Mixed precision
  fp16: true
  gradient_checkpointing: true
  
# Data
data:
  train_file: "data/train.jsonl"
  validation_file: "data/validation.jsonl"
  test_file: "data/test.jsonl"
  
# LoRA Configuration
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["attn", "mlp"]

# Evaluation
evaluation:
  save_best_model: true
  metric_for_best_model: "loss"
  load_best_model_at_end: true
EOF
    print_success "Created training configuration template"
}

# Setup Jupyter notebook
setup_jupyter() {
    print_info "Setting up Jupyter notebook kernel..."
    
    # Install ipykernel
    pip install ipykernel
    
    # Create kernel
    python3 -m ipykernel install --user --name=medical_ai_env --display-name="Medical AI Environment"
    
    # Create notebook configuration
    mkdir -p ~/.jupyter
    cat > ~/.jupyter/jupyter_notebook_config.py << 'EOF'
# Jupyter notebook configuration for Medical AI training
c.NotebookApp.iopub_data_rate_limit = 10000000
c.NotebookApp.iopub_msg_rate_limit = 3000
c.NotebookApp.rate_limit_window = 3.0
EOF
    
    print_success "Jupyter notebook kernel configured"
    print_info "Start Jupyter with: jupyter notebook --ip=0.0.0.0 --port=8888"
}

# Final setup verification
verify_setup() {
    print_info "Performing final setup verification..."
    
    # Check if virtual environment is activated
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_success "Virtual environment is active: $VIRTUAL_ENV"
    else
        print_warning "Virtual environment not detected. Run: source medical_ai_env/bin/activate"
    fi
    
    # Test imports
    python3 -c "
try:
    import torch
    import transformers
    import datasets
    import numpy
    import pandas
    import sklearn
    print('✓ All core libraries importable')
except ImportError as e:
    print(f'✗ Import error: {e}')
"
    
    # Check directory structure
    REQUIRED_DIRS=("configs", "scripts", "models", "data", "utils", "evaluation")
    for dir in "${REQUIREDIRS[@]}"; do
        if [[ -d "$dir" ]]; then
            print_success "Directory exists: $dir"
        else
            print_error "Missing directory: $dir"
        fi
    done
}

# Main execution
main() {
    print_header
    
    # Get script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$SCRIPT_DIR"
    
    print_info "Working directory: $SCRIPT_DIR"
    
    # Run setup steps
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    setup_directories
    setup_environment_file
    download_base_model
    create_sample_configs
    setup_jupyter
    verify_setup
    
    print_header
    print_success "Medical AI Training Environment Setup Completed!"
    echo ""
    print_info "Next steps:"
    echo "  1. Activate the environment: source medical_ai_env/bin/activate"
    echo "  2. Update .env file with your configuration"
    echo "  3. Prepare training data in data/ directory"
    echo "  4. Run training: bash scripts/train_model.sh"
    echo "  5. Start Jupyter: jupyter notebook"
    echo ""
    print_info "For detailed documentation, see: training/README.md"
    echo ""
    print_success "Happy Training! 🚀"
}

# Run main function
main "$@"