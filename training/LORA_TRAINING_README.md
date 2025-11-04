# LoRA/PEFT Training System

A comprehensive training system for fine-tuning large language models using LoRA (Low-Rank Adaptation) and other PEFT (Parameter-Efficient Fine-Tuning) techniques.

## Features

### 🎯 Core Training Features
- **Multi-Model Support**: LLaMA, Mistral, Qwen, GPT-2, GPT-Neo, OPT
- **LoRA Configuration**: Rank, alpha, dropout, target modules
- **Quantization**: 4-bit and 8-bit quantization with BitsAndBytes
- **Memory Optimization**: Gradient checkpointing, batch size optimization
- **Mixed Precision**: FP16/BF16 training support

### 🚀 Advanced Features
- **Distributed Training**: Multi-GPU training with DeepSpeed
- **Early Stopping**: Automatic early stopping with patience
- **Checkpointing**: Flexible save/load with metadata
- **Data Handling**: Dynamic padding, ChatML format support
- **Monitoring**: WandB integration and memory monitoring

### 📊 Data Processing
- **Multiple Formats**: JSONL, JSON, CSV, TXT, HuggingFace Datasets
- **ChatML Support**: Built-in conversation formatting
- **Data Validation**: Quality filtering and deduplication
- **Statistics**: Comprehensive dataset analysis

## Installation

```bash
# Install required dependencies
pip install torch transformers datasets peft bitsandbytes accelerate
pip install wandb  # Optional for monitoring
pip install deepspeed  # Optional for distributed training
pip install pandas pyyaml  # For data processing
```

## Quick Start

### 1. Basic Training

```python
from training.scripts.train_lora import LoRATrainer, ModelConfig, LoRAConfig, TrainingConfig

# Configure model
model_config = ModelConfig(
    model_name="microsoft/DialoGPT-medium",
    model_type="gpt2",
    max_length=512
)

# Configure LoRA
lora_config = LoRAConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)

# Configure training
training_config = TrainingConfig(
    output_dir="./output",
    num_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    report_to=["wandb"]
)

# Create trainer and train
trainer = LoRATrainer(
    model_config=model_config,
    lora_config=lora_config,
    training_config=training_config,
    # ... other configs
)

trainer.train()
```

### 2. Using Configuration Files

```yaml
# configs/lora_config.yaml
model:
  model_name: "microsoft/DialoGPT-medium"
  model_type: "gpt2"
  max_length: 512

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1

training:
  output_dir: "./output"
  num_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 4
  report_to: ["wandb"]
  run_name: "my-lora-training"

optimization:
  gradient_checkpointing: true
  group_by_length: true

data:
  data_path: "data/train.jsonl"
  data_type: "jsonl"
  text_column: "text"
  train_split_ratio: 0.9
  val_split_ratio: 0.05
```

```bash
python training/scripts/train_lora.py --config configs/lora_config.yaml
```

### 3. 4-bit Quantized Training

```python
from training.scripts.train_lora import QuantizationConfig

quantization_config = QuantizationConfig(
    use_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

trainer = LoRATrainer(
    model_config=model_config,
    lora_config=lora_config,
    quantization_config=quantization_config,
    training_config=training_config,
    # ... other configs
)
```

## Configuration Reference

### Model Configuration

```python
ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",  # Base model name
    model_type="llama",                     # Model type
    trust_remote_code=False,                # Trust remote code
    max_length=2048,                        # Maximum sequence length
    device_map="auto"                       # Device mapping
)
```

### LoRA Configuration

```python
LoRAConfig(
    r=16,                           # LoRA rank
    lora_alpha=32,                  # Scaling parameter
    lora_dropout=0.1,               # Dropout rate
    target_modules=["q_proj", "v_proj"],  # Target modules
    bias="none",                    # Bias type
    use_rslora=False,              # Use RSLoRA
    use_dora=False                 # Use DoRA
)
```

### Training Configuration

```python
TrainingConfig(
    output_dir="./output",              # Output directory
    num_epochs=3,                       # Training epochs
    per_device_train_batch_size=4,      # Batch size per GPU
    gradient_accumulation_steps=1,      # Gradient accumulation
    learning_rate=2e-4,                 # Learning rate
    warmup_ratio=0.1,                   # Warmup ratio
    lr_scheduler_type="linear",         # Scheduler type
    fp16=False,                         # FP16 mixed precision
    bf16=False,                         # BF16 mixed precision
    evaluation_strategy="steps",        # Evaluation strategy
    eval_steps=500,                     # Evaluate every N steps
    save_steps=500,                     # Save every N steps
    early_stopping_patience=3,          # Early stopping patience
    report_to=["wandb"],                # Logging services
    run_name="lora-training"            # Run name
)
```

### Data Configuration

```python
DataConfig(
    data_path="data/train.jsonl",       # Data file path
    data_type="jsonl",                  # Data format
    text_column="text",                 # Text column name
    prompt_column="prompt",             # Prompt column (optional)
    response_column="response",         # Response column (optional)
    train_split_ratio=0.9,              # Training split
    val_split_ratio=0.05,               # Validation split
    max_samples=10000,                  # Max samples (optional)
    stream=False                        # Stream data loading
)
```

## Data Formats

### JSONL Format
```json
{"text": "This is a training sample."}
{"prompt": "What is AI?", "response": "AI is artificial intelligence."}
{"text": "Another training sample."}
```

### ChatML Format
The system supports ChatML conversation format:
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is machine learning?<|im_end|>
<|im_start|>assistant
Machine learning is a subset of AI...<|im_end|>
```

## Model Utilities

### Model Management

```python
from training.utils.model_utils import ModelManager

# Initialize model manager
model_manager = ModelManager(cache_dir="./cache")

# Setup model with LoRA and quantization
model, tokenizer, peft_model = model_manager.setup_model(
    model_name="microsoft/DialoGPT-medium",
    lora_config=lora_config,
    quantization_config=quantization_config
)

# Get model information
model_info = model_manager.model_info_collector.get_model_info(model)
model_manager.model_info_collector.print_model_summary(model)

# Save model
model_manager.model_saver.save_model(
    model=model,
    tokenizer=tokenizer,
    save_directory="./saved_model"
)

# Convert to 8-bit/4-bit
quantized_model = model_manager.model_converter.convert_to_8bit(model)

# Merge LoRA adapters
merged_model = model_manager.model_converter.merge_and_unload(peft_model)
```

## Data Processing

### Data Preprocessing

```python
from training.utils.data_utils import DataPreprocessor, DataPreprocessingConfig

# Configure preprocessing
preprocessing_config = DataPreprocessingConfig(
    max_length=512,
    min_length=10,
    remove_empty=True,
    remove_duplicates=True,
    normalize_whitespace=True,
    conversation_format="chatml",
    add_system_prompt=True,
    system_prompt="You are a helpful assistant."
)

# Initialize preprocessor
preprocessor = DataPreprocessor(preprocessing_config)

# Process dataset
dataset = preprocessor.prepare_dataset(
    data_path="data/train.jsonl",
    data_type="jsonl",
    save_path="./processed_data"
)

# Get dataset statistics
from training.utils.data_utils import DataStatistics
DataStatistics.print_dataset_summary(dataset)
```

### ChatML Processing

```python
from training.utils.data_utils import ChatMLProcessor

# Create conversation
conversation = ChatMLProcessor.create_conversation(
    prompt="What is the capital of France?",
    response="The capital of France is Paris.",
    system="You are a helpful assistant."
)

# Parse conversation
conversations = ChatMLProcessor.parse_conversations(conversation)
prompt, response = ChatMLProcessor.extract_prompt_response(conversation)
```

## Advanced Features

### Distributed Training

```python
optimization_config = OptimizationConfig(
    use_distributed=True,
    local_rank=0,
    deepspeed_config="configs/deepspeed_zero3.json"
)
```

### Memory Optimization

```python
optimization_config = OptimizationConfig(
    gradient_checkpointing=True,
    group_by_length=True,
    dataloader_pin_memory=False,
    skip_memory_metrics=True
)
```

### Early Stopping

```python
training_config = TrainingConfig(
    evaluation_strategy="steps",
    eval_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    early_stopping_patience=3
)
```

## Preset Configurations

The system includes several preset configurations for different scenarios:

### Llama-2-7B Full Precision
```python
llama7b_full_precision = {
    'model': {
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'model_type': 'llama'
    },
    'lora': {
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
    },
    'training': {
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'bf16': True
    }
}
```

### Memory-Constrained Training
```python
memory_optimized = {
    'model': {
        'max_length': 1024
    },
    'lora': {
        'r': 8,
        'lora_alpha': 16
    },
    'training': {
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 32,
        'fp16': True
    },
    'optimization': {
        'gradient_checkpointing': True,
        'group_by_length': True
    }
}
```

See `training/configs/lora_config.yaml` for all available presets.

## Examples

Run the basic usage example:

```bash
python training/examples/basic_usage.py
```

This creates sample data and demonstrates various configuration options.

## Testing

Run the comprehensive test suite:

```bash
cd training/tests
python test_lora_training.py
```

The tests cover:
- Configuration validation
- Model loading and saving
- Data preprocessing
- Training pipeline integration
- Error handling

## Monitoring

### WandB Integration

```python
training_config = TrainingConfig(
    report_to=["wandb"],
    run_name="my-lora-experiment"
)

# Set WANDB_API_KEY environment variable
# export WANDB_API_KEY=your_api_key
```

### Memory Monitoring

The system includes automatic memory monitoring:

```python
class MemoryMonitoringCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Logs GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {cached:.2f}GB")
```

## Error Handling

The system includes comprehensive error handling:

- **ModelLoadingError**: Issues loading base models
- **ModelSavingError**: Problems saving trained models
- **QuantizationError**: Quantization-related issues

```python
try:
    trainer.train()
except ModelLoadingError as e:
    print(f"Failed to load model: {e}")
except QuantizationError as e:
    print(f"Quantization issue: {e}")
```

## Best Practices

### Memory Efficiency
1. Use 4-bit quantization for large models
2. Enable gradient checkpointing
3. Use gradient accumulation for larger effective batch sizes
4. Set appropriate `max_length` to avoid excessive memory usage

### Training Stability
1. Start with smaller LoRA ranks (r=8, r=16)
2. Use warmup ratio of 0.1-0.2
3. Monitor validation loss for overfitting
4. Use early stopping with patience

### Data Quality
1. Clean and preprocess data thoroughly
2. Remove duplicates and low-quality samples
3. Use appropriate conversation formatting
4. Balance dataset if needed

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use quantization (4-bit/8-bit)
   - Reduce `max_length`

2. **Slow Training**
   - Increase batch size if memory allows
   - Use mixed precision (FP16/BF16)
   - Enable gradient checkpointing
   - Consider distributed training

3. **Model Not Learning**
   - Check learning rate (try smaller values)
   - Verify data format and preprocessing
   - Ensure proper LoRA configuration
   - Check for data quality issues

4. **Distributed Training Issues**
   - Verify CUDA installation
   - Check DeepSpeed configuration
   - Ensure proper environment setup

## Directory Structure

```
training/
├── scripts/
│   └── train_lora.py              # Main training script
├── configs/
│   └── lora_config.yaml           # Configuration file with presets
├── utils/
│   ├── model_utils.py             # Model utilities
│   ├── data_utils.py              # Data processing utilities
│   └── __init__.py                # Package initialization
├── tests/
│   └── test_lora_training.py      # Comprehensive test suite
├── examples/
│   └── basic_usage.py             # Usage examples
├── __init__.py                    # Main package initialization
└── LORA_TRAINING_README.md        # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PEFT library for LoRA implementation
- Transformers library for model support
- BitsAndBytes for quantization support
- DeepSpeed for distributed training