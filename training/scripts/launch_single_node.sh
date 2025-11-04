#!/bin/bash
# Single-node multi-GPU training launch script

# Usage: ./launch_single_node.sh --model_name MODEL --dataset_path DATASET --epochs EPOCHS

# Default parameters
MODEL_NAME="bert-base-uncased"
DATASET_PATH=""
EPOCHS=10
OUTPUT_DIR="./checkpoints"
CONFIG_PATH="configs/single_node_config.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if dataset path is provided
if [ -z "$DATASET_PATH" ]; then
    echo "Error: --dataset_path is required"
    echo "Usage: $0 --model_name MODEL --dataset_path DATASET [--epochs EPOCHS] [--output_dir OUTPUT_DIR] [--config CONFIG_PATH]"
    exit 1
fi

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# Get number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch distributed training
echo "Starting single-node distributed training..."
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Epochs: $EPOCHS"
echo "GPUs: $NUM_GPUS"
echo "Output: $OUTPUT_DIR"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_port=29500 \
    --master_addr=localhost \
    scripts/train_distributed.py \
    --config "$CONFIG_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR"

echo "Training completed!"