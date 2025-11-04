#!/usr/bin/env python3
"""
DeepSpeed Distributed Training Script
Supports multi-GPU and multi-node training with comprehensive monitoring and error handling.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from deepspeed import get_accelerator

# Custom imports
from utils.deepspeed_utils import (
    DeepSpeedUtils,
    MemoryProfiler,
    PerformanceMonitor,
    CheckpointManager
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class DistributedTrainer:
    """Main class for distributed training with DeepSpeed."""
    
    def __init__(self, config_path: str, local_rank: int = 0):
        """
        Initialize distributed trainer.
        
        Args:
            config_path: Path to DeepSpeed configuration file
            local_rank: Local GPU rank
        """
        self.config_path = config_path
        self.local_rank = local_rank
        self.config = self._load_config()
        self.model = None
        self.engine = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.logger = self._setup_logging()
        self.memory_profiler = MemoryProfiler()
        self.performance_monitor = PerformanceMonitor()
        self.checkpoint_manager = CheckpointManager()
        
        # Distributed training settings
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        self.logger.info(f"Initialized trainer - Rank: {self.rank}/{self.world_size}, Local rank: {self.local_rank}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load DeepSpeed configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                f'%(asctime)s - Rank {self.rank} - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler (only on main process)
            if self.rank == 0:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                file_handler = logging.FileHandler(log_dir / "distributed_training.log")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(console_formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        try:
            # Initialize process group
            deepspeed.init_distributed(dist_backend='nccl')
            
            # Set device
            torch.cuda.set_device(self.local_rank)
            torch.cuda.empty_cache()
            
            self.logger.info(f"Distributed initialized - World size: {dist.get_world_size()}")
            self.logger.info(f"Process group initialized for rank {dist.get_rank()}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed: {e}")
            raise
    
    def load_model(self, model_class, *model_args, **model_kwargs):
        """Load and prepare model for distributed training."""
        try:
            self.logger.info("Loading model for distributed training...")
            
            # Create model
            self.model = model_class(*model_args, **model_kwargs)
            
            # Wrap with DistributedDataParallel if needed
            if self.config.get('zero_optimization', {}).get('stage', 0) < 3:
                if hasattr(self.model, 'module'):
                    self.model = self.model.module
                
                # Use DeepSpeed engine instead of DDP
                self.logger.info("Using DeepSpeed for model management")
            else:
                self.logger.info("Using ZeRO Stage 3 - no explicit DDP needed")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_training(self, model, optimizer=None, scheduler=None, *dataset_args, **dataset_kwargs):
        """Setup training environment including optimizer, scheduler, and dataloaders."""
        try:
            # Setup training arguments for DeepSpeed
            training_args = {
                'model': model,
                'model_parameters': self._get_model_parameters(model),
                'config_params': self.config
            }
            
            # Add optimizer if provided
            if optimizer is not None:
                training_args['optimizer'] = optimizer
            
            # Add scheduler if provided
            if scheduler is not None:
                training_args['lr_scheduler'] = scheduler
            
            # Initialize DeepSpeed engine
            self.engine, self.optimizer, self.scheduler, _ = deepspeed.initialize(**training_args)
            
            # Setup dataloaders
            self.train_dataloader = self._setup_dataloader(*dataset_args, training=True, **dataset_kwargs)
            if 'validation_dataset' in dataset_kwargs:
                self.valid_dataloader = self._setup_dataloader(
                    dataset_kwargs['validation_dataset'], training=False
                )
            
            self.logger.info("Training environment setup completed")
            
            # Memory profiling
            if self.rank == 0:
                self.memory_profiler.profile_memory()
            
        except Exception as e:
            self.logger.error(f"Failed to setup training: {e}")
            raise
    
    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Get model parameters configuration."""
        return {
            'params': model.parameters(),
            'weight_decay': 1e-2,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
    
    def _setup_dataloader(self, dataset, training=True, **kwargs) -> DataLoader:
        """Setup dataloader with distributed sampling."""
        sampler = None
        batch_size = self.config.get('train_micro_batch_size_per_gpu', 'auto')
        
        if isinstance(batch_size, str):
            batch_size = 8  # Default batch size
        
        if training:
            sampler = DistributedSampler(dataset, shuffle=True)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=4,
            pin_memory=True,
            drop_last=True if training else False,
            **kwargs
        )
        
        self.logger.info(f"Setup dataloader - {'Training' if training else 'Validation'} mode")
        return dataloader
    
    def train_epoch(self, train_fn) -> Dict[str, float]:
        """Train for one epoch."""
        if not self.train_dataloader:
            raise ValueError("Training dataloader not set up")
        
        self.engine.train()
        epoch_metrics = {'loss': 0.0, 'samples': 0}
        
        # Initialize progress tracking
        total_batches = len(self.train_dataloader)
        progress_bar = None
        if self.rank == 0:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_batches, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass and loss calculation
                loss, metrics = train_fn(self.engine, batch)
                
                # Backward pass and optimization
                self.engine.backward(loss)
                self.engine.step()
                
                # Update metrics
                batch_size = self._get_batch_size(batch)
                epoch_metrics['loss'] += loss.item() * batch_size
                epoch_metrics['samples'] += batch_size
                
                # Update progress bar
                if progress_bar:
                    progress_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'LR': f'{self.get_current_lr():.6f}'
                    })
                    progress_bar.update(1)
                
                # Memory monitoring
                if batch_idx % 100 == 0 and self.rank == 0:
                    self.memory_profiler.log_memory_usage(f"Step {batch_idx}")
                
                # Performance monitoring
                self.performance_monitor.log_step(self.global_step, batch_idx)
                
                self.global_step += 1
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                if self.rank == 0:
                    self.memory_profiler.log_memory_usage(f"Error at batch {batch_idx}")
                raise
        
        # Finalize epoch
        if progress_bar:
            progress_bar.close()
        
        # Calculate epoch averages
        epoch_metrics['loss'] /= max(epoch_metrics['samples'], 1)
        
        return epoch_metrics
    
    def validate(self, val_fn) -> Dict[str, float]:
        """Run validation."""
        if not self.valid_dataloader:
            self.logger.warning("No validation dataloader available")
            return {}
        
        self.engine.eval()
        val_metrics = {'val_loss': 0.0, 'val_samples': 0}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_dataloader):
                try:
                    batch = self._move_batch_to_device(batch)
                    
                    # Validation forward pass
                    val_loss, metrics = val_fn(self.engine, batch)
                    
                    # Update metrics
                    batch_size = self._get_batch_size(batch)
                    val_metrics['val_loss'] += val_loss.item() * batch_size
                    val_metrics['val_samples'] += batch_size
                    
                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    raise
        
        # Calculate validation averages
        val_metrics['val_loss'] /= max(val_metrics['val_samples'], 1)
        
        return val_metrics
    
    def _move_batch_to_device(self, batch):
        """Move batch to appropriate device."""
        if isinstance(batch, dict):
            return {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                   for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [item.cuda() if isinstance(item, torch.Tensor) else item 
                   for item in batch]
        else:
            return batch.cuda() if isinstance(batch, torch.Tensor) else batch
    
    def _get_batch_size(self, batch) -> int:
        """Get batch size from batch data."""
        if isinstance(batch, dict):
            return len(next(v for v in batch.values() if isinstance(v, torch.Tensor)))
        elif isinstance(batch, (tuple, list)):
            return len(next(v for v in batch if isinstance(v, torch.Tensor)))
        else:
            return 1
    
    def get_current_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler:
            return self.scheduler.get_last_lr()[0]
        elif self.optimizer:
            return self.optimizer.param_groups[0]['lr']
        return 0.0
    
    def save_checkpoint(self, checkpoint_dir: str, save_name: str = None, epoch: int = None):
        """Save training checkpoint."""
        try:
            if save_name is None:
                save_name = f"checkpoint-{self.global_step}"
            if epoch is None:
                epoch = self.epoch
            
            checkpoint_path = Path(checkpoint_dir) / save_name
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Use DeepSpeed checkpoint saving
            self.engine.save_checkpoint(checkpoint_path)
            
            # Additional checkpoint information
            checkpoint_info = {
                'epoch': epoch,
                'global_step': self.global_step,
                'best_metric': self.best_metric,
                'config': self.config
            }
            
            with open(checkpoint_path / 'trainer_info.json', 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_dir: str, load_optimizer: bool = True, load_scheduler: bool = True):
        """Load training checkpoint."""
        try:
            # Load DeepSpeed checkpoint
            self.engine.load_checkpoint(checkpoint_dir)
            
            # Load additional checkpoint information
            trainer_info_path = Path(checkpoint_dir) / 'trainer_info.json'
            if trainer_info_path.exists():
                with open(trainer_info_path, 'r') as f:
                    checkpoint_info = json.load(f)
                
                self.epoch = checkpoint_info.get('epoch', 0)
                self.global_step = checkpoint_info.get('global_step', 0)
                self.best_metric = checkpoint_info.get('best_metric', float('inf'))
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def cleanup(self):
        """Cleanup distributed training."""
        try:
            # Save final checkpoint
            if self.rank == 0:
                self.save_checkpoint("checkpoints/final")
            
            # Performance report
            if self.rank == 0:
                self.performance_monitor.print_summary()
            
            # Memory report
            if self.rank == 0:
                self.memory_profiler.print_summary()
            
            # Cleanup distributed process group
            if dist.is_initialized():
                dist.destroy_process_group()
            
            self.logger.info("Distributed training cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def default_train_fn(engine, batch):
    """Default training function - customize as needed."""
    # This is a placeholder - implement your specific training logic here
    
    if isinstance(batch, dict):
        # Assuming batch contains 'input' and 'target' keys
        input_ids = batch.get('input_ids')
        targets = batch.get('labels')
    else:
        # Assuming batch is a tuple (inputs, targets)
        input_ids, targets = batch
    
    # Forward pass
    outputs = engine(input_ids)
    loss = outputs.loss
    
    # Calculate additional metrics if needed
    metrics = {'loss': loss.item()}
    
    return loss, metrics


def default_val_fn(engine, batch):
    """Default validation function - customize as needed."""
    if isinstance(batch, dict):
        input_ids = batch.get('input_ids')
        targets = batch.get('labels')
    else:
        input_ids, targets = batch
    
    # Forward pass
    outputs = engine(input_ids)
    loss = outputs.loss
    
    metrics = {'val_loss': loss.item()}
    
    return loss, metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="DeepSpeed Distributed Training")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to DeepSpeed configuration file")
    parser.add_argument("--local_rank", type=int, default=0, 
                       help="Local GPU rank")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Model name for loading")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to training dataset")
    parser.add_argument("--validation_path", type=str, default=None,
                       help="Path to validation dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--max_steps", type=int, default=None,
                       help="Maximum training steps")
    
    args = parser.parse_args()
    
    # Parse arguments
    torch.cuda.set_device(args.local_rank)
    
    try:
        # Initialize trainer
        trainer = DistributedTrainer(args.config, args.local_rank)
        
        # Setup distributed training
        trainer._setup_distributed()
        
        # Load model (example - customize based on your model)
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(args.model_name)
        model = trainer.load_model(lambda: model)
        
        # Setup training environment
        trainer.setup_training(model)
        
        # Resume from checkpoint if specified
        if args.resume_from:
            trainer.load_checkpoint(args.resume_from)
        
        # Training loop
        for epoch in range(args.epochs, trainer.epoch + args.epochs):
            trainer.epoch = epoch
            
            # Training
            train_metrics = trainer.train_epoch(default_train_fn)
            
            # Validation
            val_metrics = trainer.validate(default_val_fn) if trainer.valid_dataloader else {}
            
            # Print metrics
            if trainer.rank == 0:
                metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in {**train_metrics, **val_metrics}.items()])
                trainer.logger.info(f"Epoch {epoch}: {metrics_str}")
            
            # Save checkpoint
            if trainer.rank == 0 and epoch % 1 == 0:
                trainer.save_checkpoint(args.output_dir)
        
        # Final cleanup
        trainer.cleanup()
        
        if trainer.rank == 0:
            trainer.logger.info("Training completed successfully!")
    
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()