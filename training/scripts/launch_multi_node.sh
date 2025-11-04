#!/bin/bash
# Multi-node distributed training launch script

# Usage: ./launch_multi_node.sh --master_addr MASTER_ADDR --node_rank NODE_RANK --nnodes TOTAL_NODES --model_name MODEL --dataset_path DATASET

# Default parameters
MASTER_ADDR=""
NODE_RANK=0
NNODES=1
MODEL_NAME="bert-base-uncased"
DATASET_PATH=""
EPOCHS=10
OUTPUT_DIR="./checkpoints"
CONFIG_PATH="configs/multi_node_config.json"
GPUS_PER_NODE=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
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
        --gpus_per_node)
            GPUS_PER_NODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$MASTER_ADDR" ]; then
    echo "Error: --master_addr is required"
    echo "Usage: $0 --master_addr MASTER_ADDR [--node_rank NODE_RANK] [--nnodes TOTAL_NODES] --model_name MODEL --dataset_path DATASET"
    exit 1
fi

if [ -z "$DATASET_PATH" ]; then
    echo "Error: --dataset_path is required"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_MIN_NRINGS=4
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=4
export NCCL_IB_HCA=mlx5
export NCCL_IB_SL=5
export NCCL_IB_TC=136

# Get number of GPUs for this node
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Node $NODE_RANK: Detected $NUM_GPUS GPUs"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Calculate total world size
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "Multi-node distributed training configuration:"
echo "Master address: $MASTER_ADDR"
echo "Node rank: $NODE_RANK"
echo "Total nodes: $NNODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total world size: $WORLD_SIZE"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_PATH"
echo "Epochs: $EPOCHS"
echo "Output: $OUTPUT_DIR"

# Launch distributed training
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr="$MASTER_ADDR" \
    --master_port=29500 \
    scripts/train_distributed.py \
    --config "$CONFIG_PATH" \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --epochs "$EPOCHS" \
    --output_dir "$OUTPUT_DIR"

echo "Node $NODE_RANK: Training completed!"