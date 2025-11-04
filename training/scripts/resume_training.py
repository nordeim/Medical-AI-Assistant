#!/usr/bin/env python3
"""
Resume Training CLI Script

This script provides a comprehensive command-line interface for resuming training
with checkpoint management and training state recovery.

Features:
- Checkpoint selection and validation
- Training configuration updates
- Progress monitoring
- Cloud integration
- Incremental training support

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import argparse
import json
import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add training utils to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.checkpoint_manager import CheckpointManager, CheckpointConfig
from utils.training_state import TrainingState, TrainingMetrics, TrainingConfiguration
from utils.backup_manager import BackupManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResumeTrainingCLI:
    """Command-line interface for resume training functionality"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.args = None
        self.checkpoint_manager = None
        self.training_state = None
        self.backup_manager = None
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create command line argument parser"""
        parser = argparse.ArgumentParser(
            description="Resume Training CLI - Advanced checkpoint and training state management",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Resume from latest checkpoint
  python resume_training.py resume --experiment medical-ai
  
  # Resume from specific checkpoint with updated config
  python resume_training.py resume --checkpoint-id medical-ai_epoch_10_step_5000 --learning-rate 1e-4
  
  # List available checkpoints
  python resume_training.py list --experiment medical-ai
  
  # Create backup before resuming
  python resume_training.py resume --backup-before-resume --experiment medical-ai
  
  # Monitor training progress
  python resume_training.py monitor --experiment medical-ai --duration 3600
            """
        )
        
        # Global arguments
        parser.add_argument(
            '--experiment', '-e',
            type=str,
            default='default',
            help='Experiment name for training state management'
        )
        
        parser.add_argument(
            '--experiment-dir', '-d',
            type=str,
            default='./models/checkpoints',
            help='Base directory for experiment data'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        parser.add_argument(
            '--config-file', '-c',
            type=str,
            help='Configuration file for training parameters'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Resume command
        resume_parser = subparsers.add_parser('resume', help='Resume training from checkpoint')
        self._add_resume_arguments(resume_parser)
        
        # List command
        list_parser = subparsers.add_parser('list', help='List checkpoints and training state')
        self._add_list_arguments(list_parser)
        
        # Monitor command
        monitor_parser = subparsers.add_parser('monitor', help='Monitor training progress')
        self._add_monitor_arguments(monitor_parser)
        
        # Backup command
        backup_parser = subparsers.add_parser('backup', help='Backup training state')
        self._add_backup_arguments(backup_parser)
        
        # Validate command
        validate_parser = subparsers.add_parser('validate', help='Validate checkpoints')
        self._add_validate_arguments(validate_parser)
        
        # Show command
        show_parser = subparsers.add_parser('show', help='Show training state details')
        self._add_show_arguments(show_parser)
        
        return parser
    
    def _add_resume_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for resume command"""
        parser.add_argument(
            '--checkpoint-id', '-cp',
            type=str,
            help='Specific checkpoint ID to resume from'
        )
        
        parser.add_argument(
            '--best-metric', '-bm',
            type=str,
            default='accuracy',
            help='Metric to use for best checkpoint selection'
        )
        
        parser.add_argument(
            '--best-mode', '-bmd',
            type=str,
            default='max',
            choices=['max', 'min'],
            help='Mode for best checkpoint selection (maximize/minimize)'
        )
        
        parser.add_argument(
            '--start-epoch', '-se',
            type=int,
            help='Override starting epoch'
        )
        
        parser.add_argument(
            '--start-step', '-ss',
            type=int,
            help='Override starting step'
        )
        
        parser.add_argument(
            '--learning-rate', '-lr',
            type=float,
            help='Override learning rate'
        )
        
        parser.add_argument(
            '--batch-size', '-bs',
            type=int,
            help='Override batch size'
        )
        
        parser.add_argument(
            '--num-epochs', '-ne',
            type=int,
            help='Override number of epochs'
        )
        
        parser.add_argument(
            '--save-every-n-steps', '-sns',
            type=int,
            help='Override checkpoint save frequency'
        )
        
        parser.add_argument(
            '--save-every-n-epochs', '-sne',
            type=int,
            help='Override checkpoint save frequency'
        )
        
        parser.add_argument(
            '--backup-before-resume', '-bbr',
            action='store_true',
            help='Create backup before resuming training'
        )
        
        parser.add_argument(
            '--dry-run', '-dr',
            action='store_true',
            help='Show what would be done without executing'
        )
        
        parser.add_argument(
            '--force-resume', '-fr',
            action='store_true',
            help='Force resume even if validation fails'
        )
        
        parser.add_argument(
            '--incremental-training', '-it',
            action='store_true',
            help='Enable incremental training mode'
        )
        
        parser.add_argument(
            '--training-script', '-ts',
            type=str,
            help='Path to training script to execute'
        )
        
        parser.add_argument(
            '--additional-args', '-aa',
            nargs='*',
            help='Additional arguments to pass to training script'
        )
    
    def _add_list_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for list command"""
        parser.add_argument(
            '--checkpoint-id', '-cp',
            type=str,
            help='Show details for specific checkpoint'
        )
        
        parser.add_argument(
            '--sort-by', '-sb',
            type=str,
            default='timestamp',
            choices=['timestamp', 'epoch', 'step', 'size'],
            help='Sort checkpoints by specified criterion'
        )
        
        parser.add_argument(
            '--reverse', '-r',
            action='store_true',
            help='Reverse sort order'
        )
        
        parser.add_argument(
            '--format', '-f',
            type=str,
            default='table',
            choices=['table', 'json', 'yaml'],
            help='Output format'
        )
        
        parser.add_argument(
            '--include-metadata', '-im',
            action='store_true',
            help='Include checkpoint metadata'
        )
        
        parser.add_argument(
            '--filter-metrics', '-fm',
            type=str,
            help='Filter checkpoints by metric condition (e.g., "accuracy>0.9")'
        )
    
    def _add_monitor_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for monitor command"""
        parser.add_argument(
            '--duration', '-du',
            type=int,
            default=3600,
            help='Monitoring duration in seconds (0 = infinite)'
        )
        
        parser.add_argument(
            '--interval', '-i',
            type=int,
            default=30,
            help='Update interval in seconds'
        )
        
        parser.add_argument(
            '--log-file', '-lf',
            type=str,
            help='Path to training log file to monitor'
        )
        
        parser.add_argument(
            '--metrics-only', '-mo',
            action='store_true',
            help='Show only metrics progress'
        )
        
        parser.add_argument(
            '--export-metrics', '-em',
            type=str,
            help='Export metrics to file'
        )
    
    def _add_backup_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for backup command"""
        parser.add_argument(
            '--backup-name', '-bn',
            type=str,
            help='Name for backup (auto-generated if not specified)'
        )
        
        parser.add_argument(
            '--include-checkpoints', '-ic',
            action='store_true',
            help='Include model checkpoints in backup'
        )
        
        parser.add_argument(
            '--include-logs', '-il',
            action='store_true',
            help='Include training logs in backup'
        )
        
        parser.add_argument(
            '--compression', '-co',
            action='store_true',
            help='Compress backup files'
        )
        
        parser.add_argument(
            '--cloud-upload', '-cu',
            action='store_true',
            help='Upload backup to cloud storage'
        )
        
        parser.add_argument(
            '--validate-backup', '-vb',
            action='store_true',
            help='Validate backup after creation'
        )
    
    def _add_validate_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for validate command"""
        parser.add_argument(
            '--checkpoint-id', '-cp',
            type=str,
            help='Specific checkpoint to validate'
        )
        
        parser.add_argument(
            '--all-checkpoints', '-ac',
            action='store_true',
            help='Validate all checkpoints'
        )
        
        parser.add_argument(
            '--repair', '-r',
            action='store_true',
            help='Attempt to repair corrupted checkpoints'
        )
        
        parser.add_argument(
            '--report-file', '-rf',
            type=str,
            help='Save validation report to file'
        )
    
    def _add_show_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add arguments for show command"""
        parser.add_argument(
            '--show-config', '-sc',
            action='store_true',
            help='Show training configuration'
        )
        
        parser.add_argument(
            '--show-metrics', '-sm',
            action='store_true',
            help='Show current training metrics'
        )
        
        parser.add_argument(
            '--show-snapshots', '-ss',
            action='store_true',
            help='Show available snapshots'
        )
        
        parser.add_argument(
            '--show-analytics', '-sa',
            action='store_true',
            help='Show checkpoint analytics'
        )
        
        parser.add_argument(
            '--format', '-f',
            type=str,
            default='yaml',
            choices=['yaml', 'json'],
            help='Output format'
        )
    
    def parse_args(self) -> None:
        """Parse command line arguments"""
        self.args = self.parser.parse_args()
        
        if self.args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def _initialize_managers(self) -> None:
        """Initialize checkpoint and training state managers"""
        experiment_dir = Path(self.args.experiment_dir) / self.args.experiment
        
        # Checkpoint manager configuration
        checkpoint_config = CheckpointConfig()
        if self.args.config_file:
            with open(self.args.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
                for key, value in config_data.items():
                    if hasattr(checkpoint_config, key):
                        setattr(checkpoint_config, key, value)
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=experiment_dir / "checkpoints",
            config=checkpoint_config,
            experiment_name=self.args.experiment
        )
        
        # Training state manager
        self.training_state = TrainingState(
            state_dir=experiment_dir / "state",
            experiment_name=self.args.experiment
        )
        
        # Backup manager
        self.backup_manager = BackupManager(
            backup_dir=experiment_dir / "backups",
            experiment_name=self.args.experiment
        )
    
    def cmd_resume(self) -> int:
        """Execute resume training command"""
        try:
            # Initialize managers
            self._initialize_managers()
            
            # Determine checkpoint to resume from
            checkpoint_id = self._get_resume_checkpoint()
            if not checkpoint_id:
                logger.error("No suitable checkpoint found for resume")
                return 1
            
            logger.info(f"Resuming from checkpoint: {checkpoint_id}")
            
            # Create backup if requested
            if self.args.backup_before_resume:
                self._create_backup()
            
            # Update configuration if requested
            config_updates = self._get_config_updates()
            if config_updates:
                logger.info("Updating training configuration")
                self.training_state.update_config(config_updates)
            
            # Override start epoch/step if specified
            if self.args.start_epoch is not None or self.args.start_step is not None:
                self._override_start_state(checkpoint_id)
            
            # Validate checkpoint if not forced
            if not self.args.force_resume:
                if not self._validate_checkpoint(checkpoint_id):
                    logger.error("Checkpoint validation failed")
                    return 1
            
            # Show resume plan
            if self.args.dry_run:
                self._show_resume_plan(checkpoint_id, config_updates)
                return 0
            
            # Execute training script or return resume command
            return self._execute_resume(checkpoint_id)
            
        except Exception as e:
            logger.error(f"Resume training failed: {e}")
            return 1
    
    def cmd_list(self) -> int:
        """Execute list command"""
        try:
            self._initialize_managers()
            
            if self.args.checkpoint_id:
                self._show_checkpoint_details(self.args.checkpoint_id)
            else:
                self._list_checkpoints()
            
            return 0
            
        except Exception as e:
            logger.error(f"List command failed: {e}")
            return 1
    
    def cmd_monitor(self) -> int:
        """Execute monitor command"""
        try:
            self._initialize_managers()
            self._monitor_progress()
            return 0
            
        except Exception as e:
            logger.error(f"Monitor command failed: {e}")
            return 1
    
    def cmd_backup(self) -> int:
        """Execute backup command"""
        try:
            self._initialize_managers()
            self._create_backup()
            return 0
            
        except Exception as e:
            logger.error(f"Backup command failed: {e}")
            return 1
    
    def cmd_validate(self) -> int:
        """Execute validate command"""
        try:
            self._initialize_managers()
            self._validate_checkpoints()
            return 0
            
        except Exception as e:
            logger.error(f"Validate command failed: {e}")
            return 1
    
    def cmd_show(self) -> int:
        """Execute show command"""
        try:
            self._initialize_managers()
            self._show_state_details()
            return 0
            
        except Exception as e:
            logger.error(f"Show command failed: {e}")
            return 1
    
    def _get_resume_checkpoint(self) -> Optional[str]:
        """Determine checkpoint to resume from"""
        if self.args.checkpoint_id:
            if self.args.checkpoint_id in self.checkpoint_manager.metadata:
                return self.args.checkpoint_id
            else:
                logger.error(f"Checkpoint {self.args.checkpoint_id} not found")
                return None
        
        # Try to find best checkpoint
        if self.args.best_metric:
            best_checkpoint = self.checkpoint_manager.get_best_checkpoint_id(
                metric=self.args.best_metric,
                mode=self.args.best_mode
            )
            if best_checkpoint:
                return best_checkpoint
        
        # Use latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint_id()
        if latest_checkpoint:
            return latest_checkpoint
        
        return None
    
    def _get_config_updates(self) -> Dict[str, Any]:
        """Get configuration updates from command line arguments"""
        updates = {}
        
        if self.args.learning_rate is not None:
            updates['learning_rate'] = self.args.learning_rate
        
        if self.args.batch_size is not None:
            updates['batch_size'] = self.args.batch_size
        
        if self.args.num_epochs is not None:
            updates['num_epochs'] = self.args.num_epochs
        
        if self.args.save_every_n_steps is not None:
            updates['save_every_n_steps'] = self.args.save_every_n_steps
        
        if self.args.save_every_n_epochs is not None:
            updates['save_every_n_epochs'] = self.args.save_every_n_epochs
        
        # Enable incremental training if requested
        if self.args.incremental_training:
            updates['incremental_training'] = True
        
        return updates
    
    def _override_start_state(self, checkpoint_id: str) -> None:
        """Override starting epoch/step for training"""
        metadata = self.checkpoint_manager.metadata[checkpoint_id]
        
        if self.args.start_epoch is not None:
            metadata.epoch = self.args.start_epoch
        
        if self.args.start_step is not None:
            metadata.step = self.args.start_step
        
        # Update in metadata
        self.checkpoint_manager.metadata[checkpoint_id] = metadata
        self.checkpoint_manager._save_metadata()
    
    def _validate_checkpoint(self, checkpoint_id: str) -> bool:
        """Validate checkpoint for resume"""
        metadata = self.checkpoint_manager.metadata[checkpoint_id]
        checkpoint_file = Path(metadata.storage_path)
        
        logger.info(f"Validating checkpoint {checkpoint_id}")
        
        # Check if file exists
        if not checkpoint_file.exists():
            if metadata.cloud_backup:
                logger.info("Downloading checkpoint from cloud storage")
                self.checkpoint_manager._download_from_cloud(checkpoint_file, checkpoint_id)
            else:
                logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return False
        
        # Run validation
        if self.checkpoint_manager.config.validate_checkpoints:
            return self.checkpoint_manager._validate_checkpoint(checkpoint_file, metadata)
        
        return True
    
    def _show_resume_plan(self, checkpoint_id: str, config_updates: Dict[str, Any]) -> None:
        """Show planned actions for resume"""
        metadata = self.checkpoint_manager.metadata[checkpoint_id]
        
        print("\n" + "="*60)
        print("RESUME TRAINING PLAN")
        print("="*60)
        
        print(f"Checkpoint ID: {checkpoint_id}")
        print(f"Checkpoint epoch: {metadata.epoch}")
        print(f"Checkpoint step: {metadata.step}")
        print(f"Checkpoint metrics: {metadata.metrics}")
        print(f"Checkpoint timestamp: {metadata.timestamp}")
        
        if config_updates:
            print("\nConfiguration Updates:")
            for key, value in config_updates.items():
                print(f"  {key}: {value}")
        
        if self.args.training_script:
            print(f"\nTraining Script: {self.args.training_script}")
        
        additional_args = ' '.join(self.args.additional_args) if self.args.additional_args else "None"
        print(f"Additional Arguments: {additional_args}")
        
        print("\n" + "="*60)
    
    def _execute_resume(self, checkpoint_id: str) -> int:
        """Execute the actual resume training"""
        metadata = self.checkpoint_manager.metadata[checkpoint_id]
        
        if self.args.training_script:
            # Execute training script
            import subprocess
            
            cmd = [
                'python', self.args.training_script,
                '--experiment', self.args.experiment,
                '--checkpoint-id', checkpoint_id,
                '--resume-from-epoch', str(metadata.epoch),
                '--resume-from-step', str(metadata.step)
            ]
            
            # Add configuration updates
            for key, value in self._get_config_updates().items():
                cmd.extend([f'--{key}', str(value)])
            
            # Add additional arguments
            if self.args.additional_args:
                cmd.extend(self.args.additional_args)
            
            logger.info(f"Executing training script: {' '.join(cmd)}")
            
            try:
                result = subprocess.run(cmd, check=True)
                return result.returncode
            except subprocess.CalledProcessError as e:
                logger.error(f"Training script failed: {e}")
                return e.returncode
        else:
            # Return resume command for manual execution
            resume_command = self._generate_resume_command(checkpoint_id)
            print(f"\nResume Training Command:")
            print(f"{resume_command}")
            print(f"\nExecute this command to resume training from checkpoint {checkpoint_id}")
            return 0
    
    def _generate_resume_command(self, checkpoint_id: str) -> str:
        """Generate command for resuming training"""
        metadata = self.checkpoint_manager.metadata[checkpoint_id]
        
        cmd_parts = [
            "python your_training_script.py",
            f"--experiment {self.args.experiment}",
            f"--checkpoint-id {checkpoint_id}",
            f"--resume-from-epoch {metadata.epoch}",
            f"--resume-from-step {metadata.step}"
        ]
        
        # Add configuration updates
        config_updates = self._get_config_updates()
        for key, value in config_updates.items():
            cmd_parts.append(f"--{key.replace('_', '-')} {value}")
        
        return ' '.join(cmd_parts)
    
    def _create_backup(self) -> bool:
        """Create backup of training state"""
        backup_name = self.args.backup_name or f"resume_backup_{int(time.time())}"
        
        logger.info(f"Creating backup: {backup_name}")
        
        success = self.backup_manager.create_backup(
            backup_name=backup_name,
            include_checkpoints=self.args.include_checkpoints or self.args.backup_before_resume,
            include_logs=self.args.include_logs,
            compress=self.args.compression,
            upload_to_cloud=self.args.cloud_upload,
            validate=self.args.validate_backup
        )
        
        if success:
            logger.info(f"Backup created successfully: {backup_name}")
        else:
            logger.error(f"Backup creation failed: {backup_name}")
        
        return success
    
    def _list_checkpoints(self) -> None:
        """List all checkpoints"""
        checkpoints = self.checkpoint_manager.list_checkpoints(
            include_metadata=self.args.include_metadata
        )
        
        # Filter if requested
        if self.args.filter_metrics:
            checkpoints = self._filter_checkpoints_by_metrics(checkpoints, self.args.filter_metrics)
        
        # Sort checkpoints
        checkpoints.sort(
            key=lambda x: x.get(self.args.sort_by, 0),
            reverse=self.args.reverse
        )
        
        # Format output
        if self.args.format == 'json':
            print(json.dumps(checkpoints, indent=2))
        elif self.args.format == 'yaml':
            print(yaml.dump(checkpoints, default_flow_style=False))
        else:
            self._print_checkpoints_table(checkpoints)
    
    def _filter_checkpoints_by_metrics(self, checkpoints: List[Dict[str, Any]], filter_condition: str) -> List[Dict[str, Any]]:
        """Filter checkpoints by metric conditions"""
        # Parse condition (e.g., "accuracy>0.9")
        try:
            if '>' in filter_condition:
                metric, threshold = filter_condition.split('>')
                threshold = float(threshold)
                return [cp for cp in checkpoints if cp.get('metrics', {}).get(metric, 0) > threshold]
            elif '<' in filter_condition:
                metric, threshold = filter_condition.split('<')
                threshold = float(threshold)
                return [cp for cp in checkpoints if cp.get('metrics', {}).get(metric, 0) < threshold]
            elif '>=' in filter_condition:
                metric, threshold = filter_condition.split('>=')
                threshold = float(threshold)
                return [cp for cp in checkpoints if cp.get('metrics', {}).get(metric, 0) >= threshold]
            elif '<=' in filter_condition:
                metric, threshold = filter_condition.split('<=')
                threshold = float(threshold)
                return [cp for cp in checkpoints if cp.get('metrics', {}).get(metric, 0) <= threshold]
        except Exception as e:
            logger.warning(f"Failed to parse filter condition '{filter_condition}': {e}")
        
        return checkpoints
    
    def _print_checkpoints_table(self, checkpoints: List[Dict[str, Any]]) -> None:
        """Print checkpoints in table format"""
        if not checkpoints:
            print("No checkpoints found")
            return
        
        # Determine column widths
        max_id_length = max(len(cp['checkpoint_id']) for cp in checkpoints)
        max_size_length = max(len(f"{cp.get('file_size_mb', 0):.2f}") for cp in checkpoints)
        
        # Print header
        print(f"{'ID':<{max_id_length}} {'Epoch':<6} {'Step':<8} {'Size(MB)':<{max_size_length}} {'Metrics':<30}")
        print("-" * (max_id_length + 6 + 8 + max_size_length + 30))
        
        # Print checkpoints
        for cp in checkpoints:
            metrics_str = str(cp.get('metrics', {}))
            if len(metrics_str) > 27:
                metrics_str = metrics_str[:24] + "..."
            
            print(f"{cp['checkpoint_id']:<{max_id_length}} "
                  f"{cp.get('epoch', 0):<6} "
                  f"{cp.get('step', 0):<8} "
                  f"{cp.get('file_size_mb', 0):.2f:<{max_size_length}} "
                  f"{metrics_str:<30}")
    
    def _show_checkpoint_details(self, checkpoint_id: str) -> None:
        """Show detailed information about a specific checkpoint"""
        if checkpoint_id not in self.checkpoint_manager.metadata:
            logger.error(f"Checkpoint {checkpoint_id} not found")
            return
        
        metadata = self.checkpoint_manager.metadata[checkpoint_id]
        
        print(f"\nCheckpoint Details: {checkpoint_id}")
        print("="*60)
        print(f"Timestamp: {metadata.timestamp}")
        print(f"Epoch: {metadata.epoch}")
        print(f"Step: {metadata.step}")
        print(f"Model: {metadata.model_name}")
        print(f"Size: {metadata.file_size / 1024 / 1024:.2f} MB")
        print(f"Compression: {metadata.compression}")
        print(f"Cloud Backup: {metadata.cloud_backup}")
        print(f"Metrics: {metadata.metrics}")
        print(f"Training Config: {metadata.training_config}")
        print(f"State Sizes: {metadata.state_size}")
    
    def _monitor_progress(self) -> None:
        """Monitor training progress"""
        start_time = time.time()
        duration = self.args.duration
        interval = self.args.interval
        
        logger.info(f"Starting monitoring for {duration} seconds (interval: {interval}s)")
        
        try:
            while True:
                if duration > 0 and (time.time() - start_time) > duration:
                    break
                
                # Get current metrics
                current_metrics = self.training_state.current_metrics
                
                if self.args.metrics_only:
                    print(f"Epoch: {current_metrics.epoch}, Step: {current_metrics.step}, "
                          f"Loss: {current_metrics.loss:.4f}, Accuracy: {current_metrics.accuracy:.4f}")
                else:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Training Progress")
                    print(f"Epoch: {current_metrics.epoch}")
                    print(f"Step: {current_metrics.step}")
                    print(f"Phase: {current_metrics.phase}")
                    print(f"Loss: {current_metrics.loss:.4f}")
                    print(f"Validation Loss: {current_metrics.val_loss:.4f}")
                    print(f"Accuracy: {current_metrics.accuracy:.4f}")
                    print(f"Validation Accuracy: {current_metrics.val_accuracy:.4f}")
                    print(f"Learning Rate: {current_metrics.learning_rate:.2e}")
                    print("-" * 40)
                
                # Export metrics if requested
                if self.args.export_metrics:
                    self._export_metrics_progress(current_metrics)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    
    def _export_metrics_progress(self, metrics: TrainingMetrics) -> None:
        """Export current metrics to file"""
        try:
            export_file = Path(self.args.export_metrics)
            export_data = {
                'timestamp': time.time(),
                'metrics': metrics.to_dict()
            }
            
            # Append to file
            with open(export_file, 'a') as f:
                json.dump(export_data, f)
                f.write('\n')
                
        except Exception as e:
            logger.warning(f"Failed to export metrics: {e}")
    
    def _validate_checkpoints(self) -> None:
        """Validate checkpoints"""
        if self.args.all_checkpoints:
            checkpoints_to_validate = list(self.checkpoint_manager.metadata.keys())
        elif self.args.checkpoint_id:
            checkpoints_to_validate = [self.args.checkpoint_id]
        else:
            logger.error("Must specify checkpoint ID or use --all-checkpoints")
            return
        
        validation_results = []
        
        for checkpoint_id in checkpoints_to_validate:
            logger.info(f"Validating checkpoint: {checkpoint_id}")
            
            metadata = self.checkpoint_manager.metadata[checkpoint_id]
            checkpoint_file = Path(metadata.storage_path)
            
            is_valid = True
            error_message = None
            
            try:
                if not checkpoint_file.exists():
                    if metadata.cloud_backup:
                        # Try to download from cloud
                        if not self.checkpoint_manager._download_from_cloud(checkpoint_file, checkpoint_id):
                            is_valid = False
                            error_message = "Checkpoint not found locally and cloud download failed"
                    else:
                        is_valid = False
                        error_message = "Checkpoint file not found"
                else:
                    # Run validation
                    is_valid = self.checkpoint_manager._validate_checkpoint(checkpoint_file, metadata)
                    if not is_valid:
                        error_message = "Validation failed"
                        
            except Exception as e:
                is_valid = False
                error_message = str(e)
            
            result = {
                'checkpoint_id': checkpoint_id,
                'valid': is_valid,
                'error': error_message,
                'epoch': metadata.epoch,
                'step': metadata.step,
                'size_mb': metadata.file_size / 1024 / 1024
            }
            
            validation_results.append(result)
            
            if is_valid:
                logger.info(f"✓ {checkpoint_id} is valid")
            else:
                logger.error(f"✗ {checkpoint_id} is invalid: {error_message}")
        
        # Generate report
        if self.args.report_file:
            self._save_validation_report(validation_results, self.args.report_file)
        
        # Summary
        valid_count = sum(1 for r in validation_results if r['valid'])
        print(f"\nValidation Summary: {valid_count}/{len(validation_results)} checkpoints valid")
    
    def _save_validation_report(self, results: List[Dict[str, Any]], report_file: str) -> None:
        """Save validation report to file"""
        report_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiment': self.args.experiment,
            'total_checkpoints': len(results),
            'valid_checkpoints': sum(1 for r in results if r['valid']),
            'results': results
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Validation report saved to: {report_file}")
    
    def _show_state_details(self) -> None:
        """Show detailed training state information"""
        state_data = self.training_state.get_current_state()
        
        if self.args.show_config:
            print("\nTraining Configuration:")
            print("="*40)
            if self.args.format == 'json':
                print(json.dumps(state_data['config'], indent=2))
            else:
                print(yaml.dump(state_data['config'], default_flow_style=False))
        
        if self.args.show_metrics:
            print("\nCurrent Metrics:")
            print("="*40)
            if self.args.format == 'json':
                print(json.dumps(state_data['metrics'], indent=2))
            else:
                print(yaml.dump(state_data['metrics'], default_flow_style=False))
        
        if self.args.show_snapshots:
            print("\nSnapshots:")
            print("="*40)
            snapshots = self.training_state.list_snapshots()
            if self.args.format == 'json':
                print(json.dumps(snapshots, indent=2))
            else:
                print(yaml.dump(snapshots, default_flow_style=False))
        
        if self.args.show_analytics:
            print("\nCheckpoint Analytics:")
            print("="*40)
            analytics = self.checkpoint_manager.get_analytics()
            if self.args.format == 'json':
                print(json.dumps(analytics, indent=2))
            else:
                print(yaml.dump(analytics, default_flow_style=False))
        
        # General state info
        if not any([self.args.show_config, self.args.show_metrics, self.args.show_snapshots, self.args.show_analytics]):
            print("\nTraining State Overview:")
            print("="*40)
            print(f"Experiment: {state_data['experiment_name']}")
            print(f"Config Version: {state_data['config']['version']}")
            print(f"Current Epoch: {state_data['metrics']['epoch']}")
            print(f"Current Step: {state_data['metrics']['step']}")
            print(f"Metrics History: {state_data['metrics_history_length']} entries")
            print(f"Has Snapshots: {state_data['has_snapshots']}")
    
    def run(self) -> int:
        """Main execution method"""
        try:
            self.parse_args()
            
            if not self.args.command:
                self.parser.print_help()
                return 1
            
            # Initialize managers for most commands
            if self.args.command != 'help':
                self._initialize_managers()
            
            # Execute command
            if self.args.command == 'resume':
                return self.cmd_resume()
            elif self.args.command == 'list':
                return self.cmd_list()
            elif self.args.command == 'monitor':
                return self.cmd_monitor()
            elif self.args.command == 'backup':
                return self.cmd_backup()
            elif self.args.command == 'validate':
                return self.cmd_validate()
            elif self.args.command == 'show':
                return self.cmd_show()
            else:
                self.parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            logger.info("Operation cancelled by user")
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        finally:
            # Cleanup
            if self.checkpoint_manager:
                self.checkpoint_manager.close()
            if self.training_state:
                self.training_state.close()
            if self.backup_manager:
                self.backup_manager.close()


def main():
    """Main entry point"""
    cli = ResumeTrainingCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()