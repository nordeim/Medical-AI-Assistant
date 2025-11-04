#!/usr/bin/env python3
"""
Distributed Training Example for Medical AI Training Pipeline

This example demonstrates distributed training across multiple GPUs and nodes:
- DeepSpeed ZeRO optimization
- Multi-node communication setup
- Gradient accumulation strategies
- Memory optimization across devices
- Fault tolerance and checkpointing
- Performance monitoring
- Elastic training

Usage:
    # Single node, multi-GPU
    torchrun --nproc_per_node=4 examples/distributed_training_example.py --config configs/distributed_config.json
    
    # Multi-node training
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 examples/distributed_training_example.py --config configs/distributed_config.json
    
    # SLURM cluster
    sbatch --nodes=2 --ntasks-per-node=4 examples/distributed_training_example.py --config configs/distributed_config.json
"""

import os
import sys
import json
import argparse
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import time
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import distributed training components
try:
    import deepspeed
    from deepspeed import get_accelerator
    from deepspeed.ops.adam import FusedAdam
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    from deepspeed.profiling.flops_profiler import get_model_profile
    from deepspeed.accelerator import get_accelerator
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    import torch.nn.functional as F
except ImportError as e:
    print(f"Warning: Some distributed training dependencies not available: {e}")
    print("Please install with: pip install deepspeed accelerate transformers datasets")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DistributedMedicalTrainer:
    """Distributed trainer for medical AI models using DeepSpeed"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.global_step = 0
        self.local_step = 0
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision=self.config.get('mixed_precision', 'bf16'),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 1),
            split_batches=self.config.get('split_batches', False),
            dispatch_batches=self.config.get('dispatch_batches', None)
        )
        
        # Get distributed info
        self.rank = self.accelerator.process_index
        self.local_rank = self.accelerator.local_process_index
        self.world_size = self.accelerator.num_processes
        self.device = self.accelerator.device
        
        # Model and tokenizer
        self.model = None
        self.tokenizer = None
        self.engine = None
        self.optimizer = None
        self.scheduler = None
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        self.best_metric = float('inf')
        
        logger.info(f"Initialized DistributedMedicalTrainer - Rank: {self.rank}/{self.world_size}")
        logger.info(f"Local rank: {self.local_rank}, Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(self.local_rank).total_memory / 1024**3:.1f}GB")
    
    def setup_distributed_environment(self):
        """Setup distributed training environment"""
        
        # Set environment variables for DeepSpeed
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
        
        # Configure NCCL settings for optimal performance
        if 'nccl_config' in self.config:
            for key, value in self.config['nccl_config'].items():
                os.environ[key] = str(value)
        
        # Initialize process group if not already done
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
        
        # Set device for current process
        torch.cuda.set_device(self.local_rank)
        
        logger.info(f"Distributed environment setup complete - Rank {self.rank}")
    
    def setup_model_and_tokenizer(self):
        """Setup model with distributed optimizations"""
        
        model_name = self.config.get('model_name', 'microsoft/DialoGPT-medium')
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left" if model_name == "gpt2" else "right"
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading configuration
        model_kwargs = {
            'trust_remote_code': self.config.get('trust_remote_code', False),
            'torch_dtype': torch.float16 if self.config.get('fp16', False) else torch.float32,
        }
        
        # Enable memory efficient loading
        if self.world_size > 1:
            model_kwargs['device_map'] = None  # Let DeepSpeed handle device placement
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.get('gradient_checkpointing', True):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
        
        # Add LoRA adapters if configured
        if 'lora' in self.config:
            self.setup_lora()
        
        # Move model to device
        if self.world_size == 1:
            self.model = self.model.to(self.device)
        
        logger.info("Model and tokenizer setup complete")
    
    def setup_lora(self):
        """Setup LoRA adapters for parameter-efficient training"""
        
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config_dict = self.config['lora']
            logger.info("Setting up LoRA adapters")
            
            lora_config = LoraConfig(
                r=lora_config_dict.get('r', 16),
                lora_alpha=lora_config_dict.get('alpha', 32),
                target_modules=lora_config_dict.get('target_modules', ['q_proj', 'v_proj']),
                lora_dropout=lora_config_dict.get('dropout', 0.1),
                bias=lora_config_dict.get('bias', 'none'),
                task_type='CAUSAL_LM'
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # Count trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            
        except ImportError:
            logger.warning("PEFT not available, skipping LoRA setup")
    
    def prepare_dataset(self):
        """Prepare distributed dataset"""
        
        logger.info("Preparing distributed dataset")
        
        # Create or load dataset
        if 'dataset_path' in self.config:
            dataset = self.load_dataset_from_file(self.config['dataset_path'])
        else:
            dataset = self.create_medical_dataset()
        
        # Format for training
        dataset = dataset.map(
            self.format_medical_example,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Tokenize dataset
        dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            desc="Tokenizing dataset"
        )
        
        # Set format for PyTorch
        dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels']
        )
        
        # Split dataset
        if 'val_size' in self.config:
            val_size = self.config['val_size']
        else:
            val_size = min(0.1, max(100 / len(dataset), 0.01))
        
        split_dataset = dataset.train_test_split(test_size=val_size, seed=42)
        
        # Create distributed samplers
        from torch.utils.data.distributed import DistributedSampler
        
        train_sampler = DistributedSampler(
            split_dataset['train'],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=self.config.get('dataloader_drop_last', False)
        )
        
        val_sampler = DistributedSampler(
            split_dataset['test'],
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )
        
        # Create data loaders
        batch_size = self.config.get('per_device_train_batch_size', 4)
        eval_batch_size = self.config.get('per_device_eval_batch_size', batch_size)
        num_workers = self.config.get('dataloader_num_workers', 4)
        
        self.train_loader = self.accelerator.prepare(
            torch.utils.data.DataLoader(
                split_dataset['train'],
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=train_sampler.drop_last
            )
        )
        
        self.val_loader = self.accelerator.prepare(
            torch.utils.data.DataLoader(
                split_dataset['test'],
                batch_size=eval_batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
        )
        
        logger.info(f"Dataset prepared - Train: {len(split_dataset['train'])}, Val: {len(split_dataset['test'])}")
    
    def load_dataset_from_file(self, file_path: str) -> Dataset:
        """Load dataset from file"""
        
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
            elif file_path.endswith('.jsonl'):
                data = []
                for line in f:
                    data.append(json.loads(line.strip()))
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        if isinstance(data[0], dict):
            return Dataset.from_list(data)
        else:
            return Dataset.from_dict({'text': data})
    
    def create_medical_dataset(self) -> Dataset:
        """Create a comprehensive medical training dataset"""
        
        # Extended medical dataset for distributed training
        medical_data = []
        
        # Cardiology examples
        for i in range(100):  # More data for distributed training
            medical_data.append({
                "text": f"Medical Question {i}: What are the symptoms of heart disease?",
                "answer": f"Heart disease symptoms include chest pain, shortness of breath, irregular heartbeat, and fatigue. Medical Answer {i}",
                "specialty": "cardiology",
                "complexity": "moderate"
            })
        
        # Neurology examples
        for i in range(100):
            medical_data.append({
                "text": f"Medical Question {i}: What causes headaches?",
                "answer": f"Headaches can be caused by tension, stress, dehydration, or medical conditions. Medical Answer {i}",
                "specialty": "neurology",
                "complexity": "simple"
            })
        
        # Emergency medicine
        for i in range(50):
            medical_data.append({
                "text": f"Medical Question {i}: When should I call 911?",
                "answer": f"Call 911 for chest pain, difficulty breathing, stroke symptoms, or severe injury. Medical Answer {i}",
                "specialty": "emergency_medicine",
                "complexity": "critical"
            })
        
        # General medicine
        for i in range(100):
            medical_data.append({
                "text": f"Medical Question {i}: What causes fever?",
                "answer": f"Fever can be caused by infections, inflammation, or other medical conditions. Medical Answer {i}",
                "specialty": "general_medicine",
                "complexity": "simple"
            })
        
        return Dataset.from_list(medical_data)
    
    def format_medical_example(self, examples):
        """Format medical examples for training"""
        
        formatted_texts = []
        for text, answer in zip(examples['text'], examples['answer']):
            formatted = f"Medical Question: {text.replace('Medical Question ', '')}\nMedical Answer: {answer}"
            formatted_texts.append(formatted)
        
        return {"text": formatted_texts}
    
    def tokenize_function(self, examples):
        """Tokenize examples for training"""
        
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length' if self.config.get('pad_to_max_length', True) else False,
            max_length=self.config.get('max_length', 512),
            return_tensors='pt'
        )
    
    def setup_deepspeed_engine(self):
        """Setup DeepSpeed engine for distributed training"""
        
        logger.info("Initializing DeepSpeed engine")
        
        # Load DeepSpeed configuration
        if 'deepspeed_config' in self.config:
            ds_config = self.config['deepspeed_config']
        else:
            ds_config = self.get_default_deepspeed_config()
        
        # Add training configuration to DeepSpeed config
        training_config = {
            'train_batch_size': self.config.get('per_device_train_batch_size', 4) * self.world_size,
            'train_micro_batch_size_per_gpu': self.config.get('per_device_train_batch_size', 4),
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 1),
            'steps_per_print': self.config.get('steps_per_print', 10),
            'save_interval': self.config.get('save_interval', 500),
            'logging': {
                'tensorboard': {
                    'enabled': self.config.get('tensorboard_logging', True),
                    'output_path': f'./logs/tensorboard/rank_{self.rank}'
                }
            }
        }
        
        # Merge configurations
        if 'bfloat16' not in ds_config and self.config.get('use_bf16', False):
            ds_config['bfloat16'] = {'enabled': True}
        
        if 'gradient_clipping' not in ds_config:
            ds_config['gradient_clipping'] = self.config.get('gradient_clipping', 1.0)
        
        # Initialize DeepSpeed engine
        self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config=ds_config,
            args=None,
            optimizer=self.config.get('optimizer'),
            lr_scheduler=self.config.get('lr_scheduler')
        )
        
        # Replace optimizer if specified
        if 'optimizer' in self.config:
            if self.config['optimizer'] == 'adam8bit':
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    self.model.parameters(),
                    lr=self.config.get('learning_rate', 5e-5),
                    weight_decay=self.config.get('weight_decay', 0.01)
                )
        
        logger.info(f"DeepSpeed engine initialized with config: {ds_config}")
    
    def get_default_deepspeed_config(self) -> Dict[str, Any]:
        """Get default DeepSpeed configuration based on model size and hardware"""
        
        # Auto-detect optimal configuration
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(self.local_rank).total_memory / 1024**3
            num_gpus = self.world_size
        else:
            gpu_memory_gb = 8  # Assume some memory if no CUDA
            num_gpus = 1
        
        # Model size estimation
        if 'DialoGPT' in self.config.get('model_name', ''):
            if 'medium' in self.config['model_name']:
                model_size = 'medium'
            elif 'large' in self.config['model_name']:
                model_size = 'large'
            else:
                model_size = 'small'
        else:
            model_size = 'medium'
        
        # Determine ZeRO stage
        if gpu_memory_gb < 8 or num_gpus < 2:
            zero_stage = 1  # Small models or single GPU
        elif gpu_memory_gb < 16 or num_gpus < 4:
            zero_stage = 2  # Medium models or limited GPUs
        else:
            zero_stage = 3  # Large models or many GPUs
        
        config = {
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": zero_stage >= 2,
                "offload_param": zero_stage == 3,
                "gather_16bit_weights_on_model_save": zero_stage == 3,
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": 2e8 if zero_stage >= 2 else 5e8,
                "reduce_bucket_size": 2e8 if zero_stage >= 2 else 5e8,
            },
            "bfloat16": {
                "enabled": self.config.get('use_bf16', True) and gpu_memory_gb >= 16
            },
            "zero_allow_untested_optimizer": True
        }
        
        logger.info(f"Auto-configured DeepSpeed: Stage {zero_stage}, GPUs: {num_gpus}, Memory: {gpu_memory_gb}GB")
        return config
    
    def train(self):
        """Execute distributed training"""
        
        logger.info("Starting distributed training")
        
        # Setup everything
        self.setup_distributed_environment()
        self.setup_model_and_tokenizer()
        self.prepare_dataset()
        self.setup_deepspeed_engine()
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Training configuration
        num_epochs = self.config.get('num_train_epochs', 3)
        eval_steps = self.config.get('eval_steps', 200)
        save_steps = self.config.get('save_steps', 500)
        logging_steps = self.config.get('logging_steps', 10)
        
        # Resume from checkpoint if specified
        resume_from_checkpoint = self.config.get('resume_from_checkpoint')
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.engine.load_checkpoint(resume_from_checkpoint)
        
        # Main training loop
        for epoch in range(num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            epoch_start_time = time.time()
            self.engine.train()
            
            for step, batch in enumerate(self.train_loader):
                # Forward pass
                outputs = self.engine(**batch)
                loss = outputs.loss
                
                # Backward pass and optimizer step
                self.engine.backward(loss)
                self.engine.step()
                
                # Update global step
                self.global_step += 1
                self.local_step += 1
                
                # Logging
                if self.rank == 0 and self.global_step % logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.get('learning_rate', 5e-5)
                    
                    logger.info(
                        f"Epoch {epoch+1}, Step {self.global_step}: "
                        f"Loss={loss.item():.4f}, LR={current_lr:.2e}, "
                        f"Time={time.time() - epoch_start_time:.2f}s"
                    )
                    
                    # Log performance metrics
                    self.performance_monitor.log_training_step(
                        self.global_step, loss.item(), current_lr
                    )
                
                # Evaluation
                if self.global_step % eval_steps == 0 and self.global_step > 0:
                    if self.rank == 0:
                        logger.info(f"Running evaluation at step {self.global_step}")
                    
                    eval_results = self.evaluate()
                    
                    # Save checkpoint if best
                    if eval_results.get('eval_loss', float('inf')) < self.best_metric:
                        self.best_metric = eval_results.get('eval_loss', float('inf'))
                        self.save_checkpoint(is_best=True)
                        if self.rank == 0:
                            logger.info(f"New best model saved! Loss: {self.best_metric:.4f}")
                
                # Regular checkpoint saving
                if self.global_step % save_steps == 0 and self.global_step > 0:
                    self.save_checkpoint(is_best=False)
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            if self.rank == 0:
                logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Final evaluation and saving
        if self.rank == 0:
            logger.info("Training completed. Running final evaluation...")
            
        final_results = self.evaluate()
        
        # Save final model
        self.save_final_model()
        
        if self.rank == 0:
            logger.info("="*60)
            logger.info("Distributed training completed successfully!")
            logger.info(f"Best validation loss: {self.best_metric:.4f}")
            logger.info(f"Total steps: {self.global_step}")
            logger.info("="*60)
        
        # Clean up
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def evaluate(self):
        """Evaluate model on validation set"""
        
        self.engine.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.engine(**batch)
                loss = outputs.loss
                
                # Gather loss across all processes
                gathered_loss = self.accelerator.gather(loss)
                total_loss += gathered_loss.sum().item()
                num_samples += gathered_loss.numel()
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        results = {
            'eval_loss': avg_loss,
            'eval_perplexity': perplexity,
            'eval_samples': num_samples,
            'step': self.global_step
        }
        
        if self.rank == 0:
            logger.info(f"Evaluation: Loss={avg_loss:.4f}, Perplexity={perplexity:.2f}")
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        
        if self.rank != 0:
            return
        
        checkpoint_dir = os.path.join(
            self.config.get('output_dir', './outputs/distributed'),
            'checkpoints',
            f'checkpoint-{self.global_step}'
        )
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save DeepSpeed checkpoint
        self.engine.save_checkpoint(checkpoint_dir)
        
        # Save additional metadata
        metadata = {
            'global_step': self.global_step,
            'epoch': self.global_step // len(self.train_loader),
            'best_metric': self.best_metric,
            'config': self.config,
            'performance_summary': self.performance_monitor.get_summary()
        }
        
        metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        if is_best:
            # Save as best checkpoint
            best_dir = os.path.join(
                self.config.get('output_dir', './outputs/distributed'),
                'checkpoints',
                'best'
            )
            os.makedirs(best_dir, exist_ok=True)
            
            # Copy checkpoint to best directory
            import shutil
            shutil.copytree(checkpoint_dir, best_dir, dirs_exist_ok=True)
    
    def save_final_model(self):
        """Save final trained model"""
        
        if self.rank != 0:
            return
        
        output_dir = self.config.get('output_dir', './outputs/distributed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model using DeepSpeed
        self.engine.save_16bit_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config_path = os.path.join(output_dir, 'distributed_training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save training summary
        summary = {
            'training_completed': datetime.now().isoformat(),
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'world_size': self.world_size,
            'model_type': 'LoRA' if 'lora' in self.config else 'Full',
            'deepspeed_stage': self.config.get('deepspeed_config', {}).get('zero_optimization', {}).get('stage', 'unknown'),
            'performance_summary': self.performance_monitor.get_summary()
        }
        
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

class PerformanceMonitor:
    """Monitor training performance metrics"""
    
    def __init__(self):
        self.step_times = []
        self.memory_usage = []
        self.throughput = []
        self.training_start_time = None
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.training_start_time = time.time()
    
    def log_training_step(self, step: int, loss: float, learning_rate: float):
        """Log performance metrics for a training step"""
        
        current_time = time.time()
        
        if self.step_times:
            step_time = current_time - self.step_times[-1]['timestamp']
            self.step_times.append({
                'step': step,
                'loss': loss,
                'learning_rate': learning_rate,
                'step_time': step_time,
                'timestamp': current_time
            })
            
            # Calculate throughput (examples per second)
            if len(self.step_times) > 1:
                time_diff = step_time
                throughput = 1 / time_diff  # Simplified throughput calculation
                self.throughput.append(throughput)
        
        else:
            self.step_times.append({
                'step': step,
                'loss': loss,
                'learning_rate': learning_rate,
                'step_time': 0,
                'timestamp': current_time
            })
        
        # Log memory usage if CUDA available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            self.memory_usage.append({
                'step': step,
                'allocated_gb': memory_allocated,
                'reserved_gb': memory_reserved,
                'timestamp': current_time
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        if not self.step_times:
            return {}
        
        step_times = [entry['step_time'] for entry in self.step_times[1:]]  # Skip first entry
        avg_step_time = np.mean(step_times) if step_times else 0
        std_step_time = np.std(step_times) if step_times else 0
        
        avg_throughput = np.mean(self.throughput) if self.throughput else 0
        
        peak_memory = 0
        if self.memory_usage:
            peak_memory = max(entry['reserved_gb'] for entry in self.memory_usage)
        
        total_training_time = (
            time.time() - self.training_start_time 
            if self.training_start_time else 0
        )
        
        return {
            'average_step_time': avg_step_time,
            'std_step_time': std_step_time,
            'average_throughput': avg_throughput,
            'peak_memory_gb': peak_memory,
            'total_training_time': total_training_time,
            'total_steps': len(self.step_times)
        }

def setup_slurm_environment():
    """Setup environment for SLURM cluster"""
    
    # Get SLURM environment variables
    if 'SLURM_JOB_ID' in os.environ:
        logger.info("Detected SLURM environment")
        
        # Set distributed training environment
        os.environ['MASTER_ADDR'] = os.environ.get('SLURM_LAUNCH_NODE_IPADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('SLURM_LAUNCH_NODE_PORT', '29500')
        
        # Set NCCL settings for cluster
        os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'INFO')
        os.environ['NCCL_IB_DISABLE'] = os.environ.get('NCCL_IB_DISABLE', '1')
        
        logger.info(f"SLURM Job ID: {os.environ.get('SLURM_JOB_ID')}")
        logger.info(f"SLURM Job Name: {os.environ.get('SLURM_JOB_NAME')}")
        logger.info(f"Number of tasks: {os.environ.get('SLURM_NNODES')}")
        logger.info(f"Master address: {os.environ.get('MASTER_ADDR')}")

def create_sample_config(output_path: str):
    """Create a sample distributed training configuration"""
    
    config = {
        "model_name": "microsoft/DialoGPT-medium",
        "output_dir": "./outputs/distributed_training",
        
        # Distributed training settings
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        
        # Memory and performance optimizations
        "gradient_checkpointing": True,
        "use_bf16": True,
        "mixed_precision": "bf16",
        "load_in_8bit": False,
        "load_in_4bit": False,
        
        # DeepSpeed configuration
        "deepspeed_config": {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "allgather_partitions": True,
                "reduce_scatter": True,
                "allgather_bucket_size": 2e8,
                "reduce_bucket_size": 2e8
            },
            "bfloat16": {
                "enabled": True
            },
            "steps_per_print": 10,
            "save_interval": 500
        },
        
        # Data settings
        "max_length": 512,
        "pad_to_max_length": True,
        "dataloader_num_workers": 4,
        "dataloader_drop_last": False,
        "val_size": 0.1,
        
        # Logging and monitoring
        "tensorboard_logging": True,
        "logging_steps": 10,
        "eval_steps": 200,
        "save_steps": 500,
        
        # Advanced settings
        "optimizer": "adam8bit",
        "gradient_clipping": 1.0,
        "warmup_steps": 1000,
        "lr_scheduler_type": "cosine"
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Sample configuration created: {output_path}")

def main():
    """Main function for distributed training example"""
    
    parser = argparse.ArgumentParser(description="Distributed Medical AI Training Example")
    parser.add_argument("--config", type=str, default="configs/distributed_config.json",
                       help="Path to training configuration file")
    parser.add_argument("--create_sample_config", action="store_true",
                       help="Create sample configuration file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Create sample configuration if requested
    if args.create_sample_config:
        create_sample_config(args.config)
        return 0
    
    # Check if configuration file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    
    # Setup environment for SLURM if detected
    setup_slurm_environment()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Add resume path to config
    if args.resume_from_checkpoint:
        config['resume_from_checkpoint'] = args.resume_from_checkpoint
    
    # Create output directory
    output_dir = config.get('output_dir', './outputs/distributed_training')
    os.makedirs(output_dir, exist_ok=True)
    
    # Log system information
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    else:
        logger.warning("CUDA not available, training on CPU")
    
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize and run distributed trainer
        trainer = DistributedMedicalTrainer(config)
        trainer.train()
        
    except Exception as e:
        logger.error(f"Distributed training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
