#!/usr/bin/env python3
"""
Comprehensive Example: Model Checkpointing and Resume Functionality

This example demonstrates how to integrate the advanced checkpointing system
into a real training workflow. It shows:

1. Setting up all managers (checkpoint, training state, backup, analytics)
2. Integrating checkpointing into a training loop
3. Resuming training from checkpoints
4. Monitoring and analytics
5. Backup and recovery operations

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any

# Add training utils to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import training utilities
from utils import (
    create_checkpoint_manager,
    create_training_state,
    create_backup_manager,
    integrate_with_existing_training_loop
)
from utils.checkpoint_manager import CheckpointManager, CheckpointConfig
from utils.training_state import TrainingMetrics
from utils.analytics import CheckpointAnalytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    
    def __init__(self, input_size=10, hidden_size=50, num_classes=2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_data(num_samples=1000, input_size=10):
    """Create dummy data for demonstration"""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, 2, (num_samples,))
    return TensorDataset(X, y)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    checkpoint_manager,
    training_state,
    backup_manager,
    num_epochs=10,
    experiment_name="demo_experiment"
):
    """
    Training loop with integrated checkpointing and monitoring
    """
    logger.info(f"Starting training for {num_epochs} epochs")
    
    # Training metrics tracking
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Integration helpers
    integration = integrate_with_existing_training_loop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        training_state=training_state,
        save_frequency=500  # Save every 500 steps
    )
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update metrics and potentially save checkpoint
            metrics = {
                "loss": loss.item(),
                "accuracy": 100 * train_correct / train_total if train_total > 0 else 0,
                "learning_rate": scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            }
            
            integration["update_metrics"](epoch, metrics)
            integration["save_checkpoint_if_needed"](epoch, metrics)
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # Update training state with epoch-level metrics
        epoch_metrics = TrainingMetrics(
            epoch=epoch,
            step=integration["get_step"](),
            phase="validation",
            loss=avg_train_loss,
            val_loss=avg_val_loss,
            accuracy=train_accuracy,
            val_accuracy=val_accuracy,
            learning_rate=scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        )
        
        training_state.update_metrics(epoch_metrics)
        
        # Save checkpoint for this epoch (if not already saved)
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=integration["get_step"](),
            metrics={
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "accuracy": train_accuracy,
                "val_accuracy": val_accuracy
            },
            training_config={
                "model_type": "SimpleModel",
                "input_size": 10,
                "hidden_size": 50,
                "num_classes": 2,
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
        )
        
        logger.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                   f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save as best model
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=integration["get_step"](),
                metrics={
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "accuracy": train_accuracy,
                    "val_accuracy": val_accuracy
                },
                checkpoint_id=f"{experiment_name}_best_model",
                save_best=True
            )
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break
        
        # Create backup every 5 epochs
        if (epoch + 1) % 5 == 0:
            backup_manager.create_backup(
                backup_name=f"epoch_{epoch}",
                backup_type="incremental",
                validate=True
            )
    
    logger.info("Training completed!")
    return model


def resume_training_from_checkpoint(
    checkpoint_id: str,
    model,
    optimizer,
    scheduler,
    checkpoint_manager,
    training_state,
    experiment_name: str = "demo_experiment"
):
    """
    Resume training from a specific checkpoint
    """
    logger.info(f"Resuming training from checkpoint: {checkpoint_id}")
    
    try:
        # Load checkpoint
        loaded_data = checkpoint_manager.load_checkpoint(
            checkpoint_id=checkpoint_id,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        
        # Get loaded state
        loaded_epoch = loaded_data['loaded_epoch']
        loaded_step = loaded_data['loaded_step']
        loaded_metrics = loaded_data['loaded_metrics']
        
        logger.info(f"Resumed from epoch {loaded_epoch}, step {loaded_step}")
        logger.info(f"Loaded metrics: {loaded_metrics}")
        
        # Continue training from the loaded epoch
        return loaded_epoch + 1, loaded_step
        
    except Exception as e:
        logger.error(f"Failed to resume from checkpoint: {e}")
        return 0, 0


def demonstrate_analytics(checkpoint_manager, training_state, output_dir):
    """Demonstrate analytics capabilities"""
    logger.info("Generating analytics and reports...")
    
    # Create analytics instance
    analytics = CheckpointAnalytics(
        checkpoint_manager=checkpoint_manager,
        training_state=training_state,
        output_dir=output_dir
    )
    
    # Generate comprehensive analytics
    analytics_data = analytics.generate_comprehensive_analytics()
    
    # Save analytics data
    analytics_file = Path(output_dir) / "analytics_data.json"
    import json
    with open(analytics_file, 'w') as f:
        json.dump(analytics_data, f, indent=2, default=str)
    
    logger.info(f"Analytics saved to: {analytics_file}")
    return analytics_data


def demonstrate_backup_operations(backup_manager):
    """Demonstrate backup operations"""
    logger.info("Demonstrating backup operations...")
    
    # Create a comprehensive backup
    backup_id = backup_manager.create_backup(
        backup_name="comprehensive_demo_backup",
        backup_type="full",
        include_checkpoints=True,
        include_logs=True,
        include_config=True,
        validate=True
    )
    
    if backup_id:
        logger.info(f"Backup created: {backup_id}")
        
        # Test backup restoration
        test_result = backup_manager.test_backup_recovery(backup_id)
        logger.info(f"Backup test result: {test_result}")
        
        # List all backups
        backups = backup_manager.list_backups()
        logger.info(f"Total backups: {len(backups)}")
        
        for backup in backups[:3]:  # Show first 3
            logger.info(f"  - {backup['backup_id']}: {backup['size_mb']:.2f} MB")
    
    return backup_id


def main():
    """Main demonstration function"""
    parser = argparse.ArgumentParser(description="Model checkpointing demonstration")
    parser.add_argument("--resume-from", type=str, help="Checkpoint ID to resume from")
    parser.add_argument("--experiment-name", type=str, default="demo_experiment", 
                       help="Experiment name")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument("--skip-analytics", action="store_true", help="Skip analytics generation")
    parser.add_argument("--skip-backup", action="store_true", help="Skip backup operations")
    
    args = parser.parse_args()
    
    # Setup experiment directories
    base_dir = Path("./demo_experiments") / args.experiment_name
    checkpoint_dir = base_dir / "checkpoints"
    state_dir = base_dir / "state"
    backup_dir = base_dir / "backups"
    analytics_dir = base_dir / "analytics"
    
    # Create directories
    for dir_path in [checkpoint_dir, state_dir, backup_dir, analytics_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Experiment directory: {base_dir}")
    
    # Initialize managers
    logger.info("Initializing managers...")
    
    checkpoint_manager = create_checkpoint_manager(
        save_dir=checkpoint_dir,
        experiment_name=args.experiment_name,
        config={
            "save_every_n_epochs": 2,
            "save_every_n_steps": 200,
            "compress_checkpoints": True,
            "use_cloud_backup": False,
            "validate_checkpoints": True,
            "analytics_enabled": True
        }
    )
    
    training_state = create_training_state(
        state_dir=state_dir,
        experiment_name=args.experiment_name,
        max_history=2000
    )
    
    backup_manager = create_backup_manager(
        backup_dir=backup_dir,
        experiment_name=args.experiment_name,
        config={
            "auto_backup": False,
            "validate_after_backup": True,
            "compress_backups": True,
            "include_checkpoints": True,
            "include_logs": True
        }
    )
    
    # Create model and data
    logger.info("Setting up model and data...")
    model = SimpleModel(input_size=10, hidden_size=50, num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Create datasets
    train_dataset = create_dummy_data(num_samples=800, input_size=10)
    val_dataset = create_dummy_data(num_samples=200, input_size=10)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    
    if args.resume_from and not args.no_resume:
        start_epoch, start_step = resume_training_from_checkpoint(
            args.resume_from, model, optimizer, scheduler,
            checkpoint_manager, training_state, args.experiment_name
        )
    
    logger.info(f"Starting training from epoch {start_epoch}")
    
    # Train model
    try:
        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_manager=checkpoint_manager,
            training_state=training_state,
            backup_manager=backup_manager,
            num_epochs=args.num_epochs,
            experiment_name=args.experiment_name
        )
        
        # Demonstrate backup operations
        if not args.skip_backup:
            demonstrate_backup_operations(backup_manager)
        
        # Generate analytics and reports
        if not args.skip_analytics:
            analytics_data = demonstrate_analytics(
                checkpoint_manager, training_state, analytics_dir
            )
            
            # Print key analytics summary
            if 'summary' in analytics_data:
                summary = analytics_data['summary']
                logger.info("\n=== Analytics Summary ===")
                logger.info(f"Total checkpoints: {summary.get('total_checkpoints', 0)}")
                logger.info(f"Total storage: {summary.get('total_storage_mb', 0):.2f} MB")
                logger.info(f"Training duration: {summary.get('training_duration_hours', 0):.2f} hours")
                logger.info(f"Checkpoint frequency: {summary.get('checkpoint_frequency', 0):.2f} per hour")
        
        # List checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        logger.info(f"\n=== Checkpoint Summary ===")
        logger.info(f"Total checkpoints created: {len(checkpoints)}")
        
        for checkpoint in checkpoints[:5]:  # Show first 5
            logger.info(f"  - {checkpoint['checkpoint_id']}: "
                       f"Epoch {checkpoint['epoch']}, Size {checkpoint['file_size_mb']:.2f} MB")
        
        # Show training state
        current_state = training_state.get_current_state()
        logger.info(f"\n=== Final Training State ===")
        logger.info(f"Final epoch: {current_state['metrics']['epoch']}")
        logger.info(f"Final step: {current_state['metrics']['step']}")
        logger.info(f"Final loss: {current_state['metrics']['loss']:.4f}")
        logger.info(f"Final accuracy: {current_state['metrics']['accuracy']:.2f}%")
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        checkpoint_manager.close()
        training_state.close()
        backup_manager.close()
    
    return 0


if __name__ == "__main__":
    exit(main())