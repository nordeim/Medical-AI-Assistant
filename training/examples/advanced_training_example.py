#!/usr/bin/env python3
"""
Advanced Training Example for Medical AI Training Pipeline

This example demonstrates advanced training features including:
- DeepSpeed distributed training
- LoRA parameter-efficient fine-tuning
- Mixed precision training
- Gradient checkpointing
- Advanced optimization techniques
- Comprehensive logging and monitoring
- Memory optimization
- Multi-GPU training

Usage:
    # Single GPU with LoRA
    python examples/advanced_training_example.py --config configs/lora_config.yaml
    
    # Multi-GPU with DeepSpeed
    torchrun --nproc_per_node=4 examples/advanced_training_example.py --config configs/deepspeed_config.json
    
    # Resume training from checkpoint
    python examples/advanced_training_example.py --config configs/lora_config.yaml --resume_from_checkpoint ./outputs/advanced_lora/checkpoint-1000
"""

import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import advanced training components
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer, DataCollatorForSeq2Seq,
        get_linear_schedule_with_warmup
    )
    from datasets import Dataset, load_dataset
    import deepspeed
    from deepspeed import zero
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    import bitsandbytes as bnb
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Please install with: pip install deepspeed peft bitsandbytes accelerate")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdvancedMedicalTrainer:
    """Advanced trainer with DeepSpeed and LoRA support"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Training state
        self.global_step = 0
        self.best_metric = float('inf')
        self.checkpoint_manager = CheckpointManager(self.config.get('output_dir', './outputs'))
        
        logger.info(f"Initialized AdvancedMedicalTrainer on device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    
    def setup_model(self):
        """Setup model with advanced optimizations"""
        
        model_name = self.config.get('model_name', 'microsoft/DialoGPT-medium')
        load_in_8bit = self.config.get('load_in_8bit', False)
        load_in_4bit = self.config.get('load_in_4bit', False)
        
        logger.info(f"Setting up model: {model_name}")
        
        # Model loading optimizations
        model_kwargs = {
            'trust_remote_code': self.config.get('trust_remote_code', False),
            'torch_dtype': torch.float16 if self.config.get('fp16', True) else torch.float32,
        }
        
        # Load with quantization if enabled
        if load_in_8bit:
            logger.info("Loading model with 8-bit quantization")
            model_kwargs.update({
                'load_in_8bit': True,
                'device_map': 'auto'
            })
        elif load_in_4bit:
            logger.info("Loading model with 4-bit quantization")
            model_kwargs.update({
                'load_in_4bit': True,
                'bnb_4bit_quant_type': self.config.get('bnb_4bit_quant_type', 'nf4'),
                'bnb_4bit_compute_dtype': torch.bfloat16,
                'device_map': 'auto'
            })
        else:
            # Standard loading
            if torch.cuda.is_available():
                model_kwargs['device_map'] = 'auto'
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.get('gradient_checkpointing', True):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        # Setup LoRA if configured
        if 'lora' in self.config:
            self.setup_lora()
        
        # Prepare model for distributed training
        if self.accelerator.num_processes > 1:
            self.model = self.accelerator.prepare_model(self.model)
            logger.info("Model prepared for distributed training")
    
    def setup_lora(self):
        """Setup LoRA adapters for parameter-efficient fine-tuning"""
        
        lora_config_dict = self.config['lora']
        
        logger.info("Setting up LoRA adapters")
        logger.info(f"LoRA config: {lora_config_dict}")
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=lora_config_dict.get('r', 16),
            lora_alpha=lora_config_dict.get('alpha', 32),
            target_modules=lora_config_dict.get('target_modules', ['q_proj', 'v_proj']),
            lora_dropout=lora_config_dict.get('dropout', 0.1),
            bias=lora_config_dict.get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=lora_config_dict.get('fan_in_fan_out', False),
            modules_to_save=lora_config_dict.get('modules_to_save', None)
        )
        
        # Add LoRA adapters to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA adapters successfully added to model")
    
    def prepare_data(self):
        """Prepare training and validation datasets"""
        
        logger.info("Preparing datasets")
        
        # Load or create dataset
        if 'dataset_path' in self.config:
            # Load from file
            dataset_path = self.config['dataset_path']
            if dataset_path.endswith('.json'):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
            elif dataset_path.endswith('.jsonl'):
                data = []
                with open(dataset_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
            
            # Convert to dataset
            if isinstance(data[0], dict):
                dataset = Dataset.from_list(data)
            else:
                dataset = Dataset.from_dict({'text': data})
        else:
            # Create sample dataset
            dataset = self.create_sample_dataset()
        
        # Add text formatting
        def format_medical_text(examples):
            """Format medical examples for training"""
            if 'question' in examples and 'answer' in examples:
                texts = []
                for q, a in zip(examples['question'], examples['answer']):
                    text = f"Medical Question: {q}\nMedical Answer: {a}"
                    texts.append(text)
                return {"text": texts}
            return examples
        
        dataset = dataset.map(format_medical_text, batched=True)
        
        # Tokenization function
        def tokenize_function(examples):
            """Tokenize and format for training"""
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length' if self.config.get('pad_to_max_length', True) else False,
                max_length=self.config.get('max_length', 512),
                return_tensors='pt'
            )
            
            # Set labels for causal language modeling
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        # Apply tokenization
        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Train/validation split
        if 'val_size' in self.config:
            val_size = self.config['val_size']
        else:
            val_size = min(0.1, max(100 / len(dataset), 0.01))  # Adaptive split
        
        split_result = dataset.train_test_split(test_size=val_size, seed=42)
        self.train_dataset = split_result['train']
        self.val_dataset = split_result['test']
        
        # Set format for PyTorch
        for dataset in [self.train_dataset, self.val_dataset]:
            dataset.set_format(
                type='torch',
                columns=['input_ids', 'attention_mask', 'labels']
            )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
        
        # Create data loaders
        self.create_data_loaders()
    
    def create_sample_dataset(self):
        """Create a comprehensive sample medical dataset"""
        
        sample_data = [
            {
                "question": "What are the common symptoms of a heart attack in adults?",
                "answer": "Common heart attack symptoms include chest pain or discomfort, pain or discomfort in arms, back, neck, jaw, or stomach, shortness of breath, nausea, lightheadedness, or cold sweats.",
                "category": "cardiology",
                "severity": "emergency"
            },
            {
                "question": "What is the difference between Type 1 and Type 2 diabetes?",
                "answer": "Type 1 diabetes is an autoimmune condition where the body doesn't produce insulin. Type 2 diabetes is a metabolic condition where the body doesn't use insulin properly, often related to lifestyle factors.",
                "category": "endocrinology",
                "severity": "chronic"
            },
            {
                "question": "What are early warning signs of a stroke?",
                "answer": "FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency services. Other signs include sudden severe headache, confusion, trouble walking, or loss of balance.",
                "category": "neurology",
                "severity": "emergency"
            },
            {
                "question": "What causes high blood pressure and how can it be managed?",
                "answer": "High blood pressure can be caused by genetics, diet high in salt, obesity, lack of exercise, stress, and certain conditions. Management includes lifestyle changes, medication, regular monitoring, and stress reduction.",
                "category": "cardiology",
                "severity": "chronic"
            },
            {
                "question": "What are the symptoms and treatment of pneumonia?",
                "answer": "Pneumonia symptoms include cough, fever, chills, difficulty breathing, chest pain, and fatigue. Treatment depends on the cause and may include antibiotics for bacterial pneumonia, rest, fluids, and pain relievers.",
                "category": "pulmonology",
                "severity": "moderate"
            },
            {
                "question": "What is asthma and how is it managed?",
                "answer": "Asthma is a chronic condition affecting airways, causing inflammation and narrowing. Management includes inhalers (bronchodilators and corticosteroids), avoiding triggers, regular monitoring, and having an action plan.",
                "category": "pulmonology",
                "severity": "chronic"
            },
            {
                "question": "What are the risk factors for developing cancer?",
                "answer": "Risk factors include age, family history, smoking, excessive alcohol consumption, poor diet, lack of exercise, exposure to carcinogens, certain infections, and obesity. Regular screening can help with early detection.",
                "category": "oncology",
                "severity": "high"
            },
            {
                "question": "What is depression and what are the treatment options?",
                "answer": "Depression is a mental health disorder affecting mood, energy, and daily functioning. Treatment options include therapy (cognitive behavioral therapy), medication (antidepressants), lifestyle changes, and support systems.",
                "category": "psychiatry",
                "severity": "moderate"
            },
            {
                "question": "What are common food allergies and their symptoms?",
                "answer": "Common food allergens include peanuts, tree nuts, shellfish, dairy, eggs, wheat, and soy. Symptoms range from mild (hives, itching) to severe (anaphylaxis). People with known allergies should carry epinephrine.",
                "category": "allergology",
                "severity": "moderate"
            },
            {
                "question": "What is the importance of regular exercise for health?",
                "answer": "Regular exercise improves cardiovascular health, helps maintain healthy weight, strengthens bones and muscles, reduces risk of chronic diseases, improves mental health, and enhances overall quality of life.",
                "category": "preventive_medicine",
                "severity": "preventive"
            }
        ]
        
        return Dataset.from_list(sample_data)
    
    def create_data_loaders(self):
        """Create optimized data loaders"""
        
        batch_size = self.config.get('per_device_train_batch_size', 4)
        num_workers = self.config.get('dataloader_num_workers', 4)
        
        # Training data loader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=self.config.get('dataloader_drop_last', False)
        )
        
        # Validation data loader
        eval_batch_size = self.config.get('per_device_eval_batch_size', batch_size)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0
        )
        
        logger.info(f"Created data loaders - Train batch size: {batch_size}, Eval batch size: {eval_batch_size}")
    
    def setup_optimization(self):
        """Setup optimizer and scheduler"""
        
        # Create optimizer
        if self.config.get('use_8bit_adam', False):
            # Use 8-bit Adam optimizer for memory efficiency
            optimizer_class = bnb.optim.AdamW8bit
            optimizer_kwargs = {
                'lr': self.config.get('learning_rate', 5e-5),
                'weight_decay': self.config.get('weight_decay', 0.01),
                'betas': tuple(self.config.get('adam_beta', [0.9, 0.999])),
            }
        else:
            # Standard optimizer
            optimizer_class = torch.optim.AdamW
            optimizer_kwargs = {
                'lr': self.config.get('learning_rate', 5e-5),
                'weight_decay': self.config.get('weight_decay', 0.01),
                'betas': tuple(self.config.get('adam_beta', [0.9, 0.999])),
                'eps': self.config.get('adam_epsilon', 1e-8),
            }
        
        # Separate parameters for different learning rates
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': optimizer_kwargs['weight_decay'],
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # Create learning rate scheduler
        num_training_steps = len(self.train_dataloader) * self.config.get('num_train_epochs', 3)
        num_warmup_steps = self.config.get('num_warmup_steps', int(num_training_steps * 0.1))
        
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        
        logger.info(f"Optimizer setup: {optimizer_class.__name__}")
        logger.info(f"LR Scheduler: linear warmup for {num_warmup_steps} steps")
    
    def train(self):
        """Execute advanced training loop"""
        
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        logger.info("Starting advanced training...")
        
        # Setup optimization
        self.setup_optimization()
        
        # Prepare everything for distributed training
        if self.accelerator.num_processes > 1:
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader
            )
        
        # Training configuration
        num_epochs = self.config.get('num_train_epochs', 3)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        eval_steps = self.config.get('eval_steps', 100)
        save_steps = self.config.get('save_steps', 100)
        logging_steps = self.config.get('logging_steps', 10)
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = None
            if self.is_main_process:
                from tqdm.auto import tqdm
                progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(self.train_dataloader):
                # Forward pass
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Update metrics
                    epoch_loss += loss.item() * gradient_accumulation_steps
                    num_batches += 1
                    self.global_step += 1
                    
                    # Gradient update
                    if (step + 1) % gradient_accumulation_steps == 0:
                        # Clip gradients
                        if max_grad_norm > 0:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), 
                                max_grad_norm
                            )
                        
                        # Update parameters
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # Logging
                        if self.is_main_process and self.global_step % logging_steps == 0:
                            current_lr = self.lr_scheduler.get_last_lr()[0]
                            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                            
                            logger.info(
                                f"Epoch {epoch+1}, Step {self.global_step}: "
                                f"Loss: {avg_loss:.4f}, LR: {current_lr:.2e}"
                            )
                    
                    # Update progress bar
                    if progress_bar is not None:
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'lr': f'{self.lr_scheduler.get_last_lr()[0]:.2e}'
                        })
                
                # Evaluation
                if self.global_step % eval_steps == 0 and self.global_step > 0:
                    if self.is_main_process:
                        logger.info(f"Running evaluation at step {self.global_step}")
                    
                    eval_results = self.evaluate()
                    
                    # Save checkpoint if it's the best model
                    if self.is_main_process:
                        if eval_results.get('eval_loss', float('inf')) < self.best_metric:
                            self.best_metric = eval_results.get('eval_loss', float('inf'))
                            self.save_checkpoint(is_best=True)
                            logger.info(f"New best model saved! Loss: {self.best_metric:.4f}")
                        else:
                            self.save_checkpoint(is_best=False)
                
                # Regular checkpoint saving
                if self.is_main_process and self.global_step % save_steps == 0 and self.global_step > 0:
                    self.save_checkpoint(is_best=False)
            
            # End of epoch
            if progress_bar is not None:
                progress_bar.close()
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Final evaluation and saving
        if self.is_main_process:
            logger.info("Training completed. Running final evaluation...")
            final_results = self.evaluate()
            
            # Save final model
            self.save_final_model()
            
            logger.info("="*50)
            logger.info("Training completed successfully!")
            logger.info(f"Best validation loss: {self.best_metric:.4f}")
            logger.info(f"Model saved to: {self.config['output_dir']}")
            logger.info("="*50)
    
    def evaluate(self):
        """Evaluate model on validation set"""
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item() * batch['input_ids'].size(0)
                num_samples += batch['input_ids'].size(0)
        
        # Gather results from all processes
        if self.accelerator.num_processes > 1:
            total_loss = self.accelerator.gather(total_loss).sum()
            num_samples = self.accelerator.gather(num_samples).sum()
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        
        results = {
            'eval_loss': avg_loss,
            'eval_samples': num_samples,
            'eval_perplexity': np.exp(avg_loss) if avg_loss < 10 else float('inf')  # Prevent overflow
        }
        
        if self.is_main_process:
            logger.info(f"Evaluation: Loss={avg_loss:.4f}, Perplexity={results['eval_perplexity']:.2f}")
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints', f'checkpoint-{self.global_step}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        if self.accelerator.num_processes == 1 or self.accelerator.is_main_process:
            self.accelerator.save_state(checkpoint_dir)
            
            # Save training state
            training_state = {
                'global_step': self.global_step,
                'epoch': self.global_step // len(self.train_dataloader),
                'best_metric': self.best_metric,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'config': self.config
            }
            
            torch.save(training_state, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            # Save LoRA adapters separately if using LoRA
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(checkpoint_dir)
            
            logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def save_final_model(self):
        """Save final trained model"""
        
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        if self.accelerator.num_processes == 1 or self.accelerator.is_main_process:
            # Save the final model
            if hasattr(self.model, 'save_pretrained'):
                self.model.save_pretrained(output_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(output_dir)
            
            # Save configuration
            config_save_path = os.path.join(output_dir, 'training_config.json')
            with open(config_save_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            # Save training summary
            summary = {
                'training_completed': datetime.now().isoformat(),
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'model_type': 'LoRA' if 'lora' in self.config else 'Full',
                'device': str(self.device),
                'config': self.config
            }
            
            summary_path = os.path.join(output_dir, 'training_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Final model saved to: {output_dir}")

class CheckpointManager:
    """Manage training checkpoints"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming training"""
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load training state
        training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location='cpu')
            return training_state
        else:
            raise FileNotFoundError(f"Training state not found in checkpoint: {checkpoint_path}")
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint"""
        
        if not os.path.exists(self.checkpoints_dir):
            return None
        
        checkpoints = [d for d in os.listdir(self.checkpoints_dir) 
                      if d.startswith('checkpoint-')]
        
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoint_steps = [int(c.split('-')[1]) for c in checkpoints]
        latest_idx = np.argmax(checkpoint_steps)
        
        return os.path.join(self.checkpoints_dir, checkpoints[latest_idx])

def main():
    """Main function for advanced training example"""
    
    parser = argparse.ArgumentParser(description="Advanced Medical AI Training Example")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file (JSON or YAML)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Name for the training run")
    
    args = parser.parse_args()
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator()
    
    # Set up distributed training if needed
    if accelerator.num_processes > 1:
        torch.cuda.set_device(accelerator.local_process_index)
    
    # Load configuration
    config = {}
    with open(args.config, 'r') as f:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Override config with command line args
    if args.run_name:
        config['run_name'] = args.run_name
    
    # Create output directory
    output_dir = config.get('output_dir', './outputs/advanced_training')
    os.makedirs(output_dir, exist_ok=True)
    
    # Log system information
    logger.info("="*60)
    logger.info("Advanced Medical AI Training Example")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Accelerator: {accelerator}")
    logger.info(f"Number of processes: {accelerator.num_processes}")
    logger.info(f"Device: {accelerator.device}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    else:
        logger.warning("CUDA not available, training on CPU")
    
    logger.info("="*60)
    
    try:
        # Initialize trainer
        trainer = AdvancedMedicalTrainer(args.config)
        
        # Setup model and data
        trainer.setup_model()
        trainer.prepare_data()
        
        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            checkpoint_manager = CheckpointManager(config.get('output_dir', './outputs'))
            training_state = checkpoint_manager.load_checkpoint(args.resume_from_checkpoint)
            trainer.global_step = training_state.get('global_step', 0)
            trainer.best_metric = training_state.get('best_metric', float('inf'))
            
            # Restore optimizer and scheduler states
            trainer.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            trainer.lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])
            
            logger.info(f"Resumed training from step {trainer.global_step}")
        
        # Start training
        trainer.train()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    return 0

if __name__ == "__main__":
    exit(main())
