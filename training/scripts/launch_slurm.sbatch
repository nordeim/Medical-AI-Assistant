#!/bin/bash
# SLURM script for distributed training on HPC clusters

#SBATCH --job-name=deepspeed_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=your_account
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# SLURM environment variables
export SLURM_NNODES=${SLURM_NNODES:-1}
export SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-4}
export SLURM_LOCALID=${SLURM_LOCALID:-0}
export SLURM_PROCID=${SLURM_PROCID:-0}
export SLURM_GTIDS=${SLURM_GTIDS:-0}
export SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-localhost}

# Training parameters (override via sbatch options or environment variables)
export MODEL_NAME=${MODEL_NAME:-"bert-base-uncased"}
export DATASET_PATH=${DATASET_PATH:-""}
export EPOCHS=${EPOCHS:-10}
export OUTPUT_DIR=${OUTPUT_DIR:-$SLURM_JOB_ID/checkpoints}
export CONFIG_PATH=${CONFIG_PATH:-"configs/multi_node_config.json"}

# Parse command line arguments if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            export MODEL_NAME="$2"
            shift 2
            ;;
        --dataset_path)
            export DATASET_PATH="$2"
            shift 2
            ;;
        --epochs)
            export EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            export OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            export CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$DATASET_PATH" ]; then
    echo "Error: DATASET_PATH environment variable or --dataset_path argument is required"
    echo "Usage: export DATASET_PATH=/path/to/dataset"
    exit 1
fi

echo "==========================================="
echo "DeepSpeed Distributed Training on SLURM"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $((SLURM_NNODES * SLURM_NTASKS_PER_NODE))"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Master node: $(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)"
echo ""

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# Load required modules (adjust for your cluster)
module load cuda/11.8
module load pytorch/2.0.1

# Set up distributed environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export RANK=$((SLURM_NNODES * SLURM_LOCALID + SLURM_PROCID))
export LOCAL_RANK=$SLURM_LOCALID

echo "Distributed training configuration:"
echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create script for the main training process
cat > slurm_train.py << 'EOF'
import os
import sys
import logging

# Set up environment for this process
os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
os.environ["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
os.environ["RANK"] = os.environ.get("RANK", "0")

# Set CUDA device
import torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# Import and run training
sys.path.append("/path/to/training/scripts")
from train_distributed import main

if __name__ == "__main__":
    # Parse arguments for this process
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.environ.get("CONFIG_PATH", "configs/multi_node_config.json"))
    parser.add_argument("--local_rank", type=int, default=local_rank)
    parser.add_argument("--model_name", default=os.environ.get("MODEL_NAME", "bert-base-uncased"))
    parser.add_argument("--dataset_path", default=os.environ.get("DATASET_PATH"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "10")))
    parser.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", "./checkpoints"))
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not args.dataset_path:
        print("Error: dataset_path is required")
        sys.exit(1)
    
    print(f"Starting training on rank {local_rank}")
    
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")
        raise
EOF

echo "Launching distributed training..."

# Launch training using torchrun
torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --node_rank=$SLURM_LOCALID \
    --master_addr="$MASTER_ADDR" \
    --master_port=$MASTER_PORT \
    --rdzv_backend=c10d \
    slurm_train.py

# Save final checkpoint info
if [ $RANK -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Final checkpoint saved to: $OUTPUT_DIR"
    echo "Training time: $(date -d @$(( $(date +%s) - $SLURM_JOB_START )))"
    
    # Clean up temporary files
    rm -f slurm_train.py
fi

echo "SLURM job $SLURM_JOB_ID completed"