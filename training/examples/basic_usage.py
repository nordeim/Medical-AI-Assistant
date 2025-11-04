#!/usr/bin/env python3
"""
Example usage of the LoRA Training System
Demonstrates how to use the comprehensive training pipeline.
"""

import os
import sys
from pathlib import Path

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.scripts.train_lora import (
    ModelConfig,
    LoRAConfig,
    QuantizationConfig,
    TrainingConfig,
    OptimizationConfig,
    DataConfig,
    LoRATrainer,
    load_config_from_yaml
)

from training.utils.model_utils import ModelManager
from training.utils.data_utils import DataPreprocessor, DataPreprocessingConfig

def example_basic_training():
    """Example of basic LoRA training setup."""
    print("=" * 60)
    print("BASIC LORA TRAINING EXAMPLE")
    print("=" * 60)
    
    # Configure model
    model_config = ModelConfig(
        model_name="gpt2",
        model_type="gpt2",
        max_length=512
    )
    
    # Configure LoRA
    lora_config = LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn"]
    )
    
    # Configure quantization
    quantization_config = QuantizationConfig()
    
    # Configure training
    training_config = TrainingConfig(
        output_dir="./output/basic_training",
        num_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.1,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        report_to=[]  # Disable wandb for example
    )
    
    # Configure optimization
    optimization_config = OptimizationConfig(
        gradient_checkpointing=True,
        group_by_length=False
    )
    
    # Configure data
    data_config = DataConfig(
        data_path="data/sample_data.jsonl",
        data_type="jsonl",
        text_column="text",
        train_split_ratio=0.9,
        val_split_ratio=0.05,
        max_samples=1000
    )
    
    # Create trainer
    trainer = LoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        quantization_config=quantization_config,
        training_config=training_config,
        optimization_config=optimization_config,
        data_config=data_config
    )
    
    print("✓ Trainer created successfully")
    print(f"Model: {model_config.model_name}")
    print(f"LoRA rank: {lora_config.r}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Batch size: {training_config.per_device_train_batch_size}")
    
    return trainer

def example_4bit_quantized_training():
    """Example of 4-bit quantized LoRA training."""
    print("\n" + "=" * 60)
    print("4-BIT QUANTIZED TRAINING EXAMPLE")
    print("=" * 60)
    
    # Configure model for 4-bit training
    model_config = ModelConfig(
        model_name="microsoft/DialoGPT-medium",
        model_type="gpt2",
        max_length=1024
    )
    
    # Configure LoRA for quantized model
    lora_config = LoRAConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["c_attn"]
    )
    
    # Configure 4-bit quantization
    quantization_config = QuantizationConfig(
        use_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Configure training for quantized model
    training_config = TrainingConfig(
        output_dir="./output/4bit_training",
        num_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        bf16=True,
        save_steps=200,
        eval_steps=200,
        evaluation_strategy="steps",
        report_to=[]
    )
    
    # Configure optimization
    optimization_config = OptimizationConfig(
        gradient_checkpointing=True,
        dataloader_pin_memory=False  # Save memory for quantized training
    )
    
    # Configure data
    data_config = DataConfig(
        data_path="data/conversation_data.jsonl",
        data_type="jsonl",
        prompt_column="prompt",
        response_column="response",
        train_split_ratio=0.85,
        val_split_ratio=0.1,
        max_samples=5000
    )
    
    # Create trainer
    trainer = LoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        quantization_config=quantization_config,
        training_config=training_config,
        optimization_config=optimization_config,
        data_config=data_config
    )
    
    print("✓ 4-bit quantized trainer created successfully")
    print(f"Model: {model_config.model_name}")
    print(f"Quantization: 4-bit ({quantization_config.bnb_4bit_quant_type})")
    print(f"LoRA rank: {lora_config.r}")
    print(f"Batch size: {training_config.per_device_train_batch_size}")
    print(f"Mixed precision: BF16")
    
    return trainer

def example_config_file_training():
    """Example of training using YAML configuration file."""
    print("\n" + "=" * 60)
    print("CONFIG FILE TRAINING EXAMPLE")
    print("=" * 60)
    
    # Create example config file
    config_path = "./configs/example_config.yaml"
    
    example_config = {
        'model': {
            'model_name': 'microsoft/DialoGPT-small',
            'model_type': 'gpt2',
            'max_length': 512
        },
        'lora': {
            'r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.1,
            'target_modules': ['c_attn']
        },
        'training': {
            'output_dir': './output/config_training',
            'num_epochs': 1,
            'per_device_train_batch_size': 4,
            'gradient_accumulation_steps': 2,
            'learning_rate': 3e-4,
            'save_steps': 50,
            'eval_steps': 50,
            'evaluation_strategy': 'steps',
            'report_to': []
        },
        'optimization': {
            'gradient_checkpointing': True,
            'group_by_length': True
        },
        'data': {
            'data_path': 'data/simple_text.txt',
            'data_type': 'txt',
            'train_split_ratio': 0.9,
            'val_split_ratio': 0.05,
            'max_samples': 200
        },
        'quantization': {}
    }
    
    # Save config to file
    import yaml
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(example_config, f, default_flow_style=False)
    
    print(f"✓ Example config created: {config_path}")
    
    # Load configuration
    config_dict = load_config_from_yaml(config_path)
    
    # Create configuration objects
    model_config = ModelConfig(**config_dict['model'])
    lora_config = LoRAConfig(**config_dict['lora'])
    quantization_config = QuantizationConfig(**config_dict.get('quantization', {}))
    training_config = TrainingConfig(**config_dict['training'])
    optimization_config = OptimizationConfig(**config_dict['optimization'])
    data_config = DataConfig(**config_dict['data'])
    
    # Create trainer
    trainer = LoRATrainer(
        model_config=model_config,
        lora_config=lora_config,
        quantization_config=quantization_config,
        training_config=training_config,
        optimization_config=optimization_config,
        data_config=data_config
    )
    
    print("✓ Trainer created from config file")
    print(f"Configuration loaded from: {config_path}")
    
    return trainer, config_path

def example_model_utils():
    """Example of using model utilities."""
    print("\n" + "=" * 60)
    print("MODEL UTILITIES EXAMPLE")
    print("=" * 60)
    
    # Initialize model manager
    model_manager = ModelManager(cache_dir="./cache")
    
    print("✓ ModelManager initialized")
    
    # Example configuration
    lora_config = {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_modules': ['c_attn']
    }
    
    quantization_config = {
        'use_4bit': False,
        'use_8bit': False
    }
    
    print("✓ Example configurations prepared")
    print(f"LoRA config: {lora_config}")
    print(f"Quantization config: {quantization_config}")
    
    # Note: Actual model setup would require internet connection and model downloads
    print("\nNote: Actual model loading requires internet connection")
    print("Example setup_model() call:")
    print("model, tokenizer, peft_model = model_manager.setup_model(")
    print("    model_name='gpt2',")
    print("    lora_config=lora_config,")
    print("    quantization_config=quantization_config")
    print(")")
    
    return model_manager

def example_data_utils():
    """Example of using data utilities."""
    print("\n" + "=" * 60)
    print("DATA UTILITIES EXAMPLE")
    print("=" * 60)
    
    # Configure data preprocessing
    preprocessing_config = DataPreprocessingConfig(
        max_length=512,
        min_length=10,
        remove_empty=True,
        remove_duplicates=True,
        filter_by_length=True,
        normalize_whitespace=True,
        conversation_format="chatml",
        add_system_prompt=True,
        system_prompt="You are a helpful assistant."
    )
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(preprocessing_config)
    
    print("✓ DataPreprocessor initialized")
    print(f"Max length: {preprocessing_config.max_length}")
    print(f"Conversation format: {preprocessing_config.conversation_format}")
    print(f"System prompt: {preprocessing_config.add_system_prompt}")
    
    # ChatML processor example
    from training.utils.data_utils import ChatMLProcessor
    
    # Example conversation
    conversation = ChatMLProcessor.create_conversation(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        system="You are a helpful assistant."
    )
    
    print(f"\n✓ Example ChatML conversation created:")
    print(conversation)
    
    # Parse conversation back
    parsed = ChatMLProcessor.parse_conversations(conversation)
    prompt, response = ChatMLProcessor.extract_prompt_response(conversation)
    
    print(f"\n✓ Extracted prompt: '{prompt}'")
    print(f"✓ Extracted response: '{response}'")
    
    return preprocessor

def create_sample_data():
    """Create sample data for demonstration."""
    print("\n" + "=" * 60)
    print("CREATING SAMPLE DATA")
    print("=" * 60)
    
    # Create sample data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample conversation data
    conversations = [
        {"prompt": "What is machine learning?", "response": "Machine learning is a subset of AI that allows systems to learn and improve from experience without being explicitly programmed."},
        {"prompt": "Explain deep learning", "response": "Deep learning is a type of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."},
        {"prompt": "What is Python used for?", "response": "Python is a versatile programming language used for web development, data analysis, artificial intelligence, scientific computing, and automation."},
        {"prompt": "How does neural network work?", "response": "Neural networks work by processing input data through interconnected nodes (neurons) with weighted connections, learning patterns through training."},
        {"prompt": "What is natural language processing?", "response": "Natural Language Processing (NLP) is a field of AI that helps computers understand, interpret, and generate human language."}
    ]
    
    # Save as JSONL
    jsonl_path = data_dir / "sample_conversations.jsonl"
    with open(jsonl_path, 'w') as f:
        for conv in conversations:
            import json
            json.dump(conv, f)
            f.write('\n')
    
    # Save as simple text
    text_path = data_dir / "sample_text.txt"
    with open(text_path, 'w') as f:
        f.write("Machine learning is transforming many industries.\n")
        f.write("Neural networks are inspired by biological brains.\n")
        f.write("Python is one of the most popular programming languages.\n")
        f.write("Deep learning has achieved remarkable results in image recognition.\n")
        f.write("Natural language processing enables computers to understand text.\n")
        f.write("Artificial intelligence is changing how we work and live.\n")
        f.write("Data science combines statistics and programming skills.\n")
        f.write("Computer vision allows machines to interpret visual information.\n")
        f.write("Reinforcement learning teaches agents through trial and error.\n")
        f.write("Large language models like GPT have shown impressive capabilities.\n")
    
    print(f"✓ Sample data created:")
    print(f"  - JSONL conversations: {jsonl_path}")
    print(f"  - Simple text: {text_path}")
    
    return str(jsonl_path), str(text_path)

def main():
    """Main example function."""
    print("LoRA Training System - Examples")
    print("This script demonstrates various features of the training system.\n")
    
    try:
        # Create sample data
        jsonl_path, text_path = create_sample_data()
        
        # Basic training example
        trainer1 = example_basic_training()
        
        # 4-bit quantized training example
        trainer2 = example_4bit_quantized_training()
        
        # Config file training example
        trainer3, config_path = example_config_file_training()
        
        # Model utilities example
        model_manager = example_model_utils()
        
        # Data utilities example
        preprocessor = example_data_utils()
        
        print("\n" + "=" * 60)
        print("EXAMPLE SUMMARY")
        print("=" * 60)
        print("✓ Basic LoRA training configuration created")
        print("✓ 4-bit quantized training configuration created") 
        print("✓ Configuration file training setup completed")
        print("✓ Model utilities demonstrated")
        print("✓ Data utilities demonstrated")
        print("✓ Sample data files created")
        
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Update model paths to actual models you want to train")
        print("2. Update data paths to your training data")
        print("3. Configure hyperparameters as needed")
        print("4. Run training with: python training/scripts/train_lora.py --config configs/example_config.yaml")
        print("5. Monitor training with WandB (add your token to config)")
        
        print(f"\nConfiguration file location: {config_path}")
        print(f"Sample data location: ./data/")
        
    except Exception as e:
        print(f"\nError in examples: {e}")
        print("This is normal if dependencies are not installed.")
        print("The code structure is complete and ready to use with proper dependencies.")

if __name__ == "__main__":
    main()