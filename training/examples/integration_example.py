"""
Integration Example: Using Performance Monitoring with Existing Training Scripts

This example demonstrates how to integrate the performance monitoring system
into existing PyTorch training code.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

# Import monitoring components
import sys
sys.path.append('../utils')

from performance_monitor import TrainingMetrics
from monitor_training import TrainingMonitor


class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def create_training_monitor(config_path=None):
    """Create and configure training monitor."""
    
    # Default configuration
    config = {
        'save_dir': './monitoring_logs',
        'tensorboard_dir': './tensorboard_logs',
        'visualization_dir': './visualizations',
        'enable_dashboard': True,
        'dashboard_host': '127.0.0.1',
        'dashboard_port': 8080,
        'enable_performance_monitor': True,
        'enable_metrics_collector': True,
        'enable_alerts': True,
        'enable_visualizations': True,
        'monitor_system': True,
        'track_gradients': True,
        'buffer_size': 1000,
        'flush_interval': 5.0,
        'alert_thresholds': {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0,
            'temperature': 80.0,
            'grad_norm_exploded': 10.0
        }
    }
    
    # Load custom config if provided
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
    
    # Create training monitor
    monitor = TrainingMonitor(config)
    return monitor


def integrate_with_training_loop(model, optimizer, train_loader, val_loader=None, 
                                num_epochs=10, device='cpu', config_path=None):
    """
    Integrate performance monitoring with training loop.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        num_epochs: Number of training epochs
        device: Device to run training on
        config_path: Path to monitoring configuration file
    """
    
    # Create and start training monitor
    monitor = create_training_monitor(config_path)
    monitor.start_monitoring()
    
    # Model and criterion
    criterion = nn.CrossEntropyLoss()
    
    # Track training metrics
    best_val_acc = 0.0
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Custom progress tracking
            epoch_start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_start_time = time.time()
                
                # Move data to device
                data, target = data.to(device), target.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # Update training loss
                train_loss += loss.item()
                
                # Calculate batch metrics
                batch_time = time.time() - batch_start_time
                batch_accuracy = (predicted == target).float().mean().item()
                learning_rate = optimizer.param_groups[0]['lr']
                
                # Log step metrics to monitoring system
                monitor.log_training_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss.item(),
                    accuracy=batch_accuracy,
                    learning_rate=learning_rate,
                    custom_metrics={
                        'batch_idx': batch_idx,
                        'batch_size': data.size(0),
                        'data_load_time': batch_time * 0.3,  # Estimated
                        'forward_pass_time': batch_time * 0.4,  # Estimated
                        'backward_pass_time': batch_time * 0.3,  # Estimated
                    }
                )
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f'  Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {batch_accuracy:.4f}')
            
            # Calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = train_correct / train_total
            
            print(f"  Training - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Validation phase (if provided)
            if val_loader:
                val_loss, val_acc = validate_model(model, val_loader, criterion, device)
                
                # Log validation metrics
                monitor.log_validation_step(val_loss, val_acc)
                
                print(f"  Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), 'best_model.pth')
                    print(f"  Saved best model with accuracy: {val_acc:.4f}")
                
                # Update current epoch in monitor
                monitor.current_epoch = epoch
            else:
                # Update current epoch in monitor
                monitor.current_epoch = epoch
                
                # If no validation, we can still benchmark inference
                if epoch % 5 == 4:  # Every 5 epochs
                    try:
                        # Create dummy input for benchmarking
                        dummy_input = torch.randn(1, 1, 28, 28).to(device)
                        monitor.log_model_inference_benchmark(model, device, dummy_input)
                    except Exception as e:
                        print(f"  Warning: Could not run inference benchmark: {e}")
            
            # Check monitoring status periodically
            if epoch % 2 == 1:
                status = monitor.get_monitoring_status()
                print(f"  Monitoring Status - Steps: {status['current_step']}")
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise
    finally:
        # Always stop monitoring gracefully
        monitor.stop_monitoring()
        print("\nMonitoring stopped")


def validate_model(model, val_loader, criterion, device='cpu'):
    """Validate model on validation dataset."""
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            val_total += target.size(0)
            val_correct += (predicted == target).sum().item()
    
    model.train()  # Set back to training mode
    
    return val_loss / len(val_loader), val_correct / val_total


def create_sample_data(num_samples=1000, input_size=784, num_classes=10):
    """Create sample training and validation data."""
    
    # Generate random data
    train_size = int(0.8 * num_samples)
    val_size = num_samples - train_size
    
    # Create training data
    train_data = torch.randn(train_size, 1, 28, 28)
    train_targets = torch.randint(0, num_classes, (train_size,))
    
    # Create validation data
    val_data = torch.randn(val_size, 1, 28, 28)
    val_targets = torch.randint(0, num_classes, (val_size,))
    
    # Create datasets
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def main():
    """Main function demonstrating integration."""
    
    # Setup
    print("Performance Monitoring Integration Example")
    print("=" * 50)
    
    # Configuration
    config = {
        'num_epochs': 5,
        'batch_size': 32,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'monitoring_config': {
            'enable_dashboard': True,
            'dashboard_port': 8080,
            'buffer_size': 500,
            'flush_interval': 2.0,
            'alert_thresholds': {
                'cpu_usage': 85.0,
                'memory_usage': 80.0,
                'grad_norm_exploded': 5.0
            }
        }
    }
    
    device = config['device']
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleNeuralNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create sample data
    print("Creating sample data...")
    train_loader, val_loader = create_sample_data(num_samples=2000)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Start training with monitoring
    print("\nStarting training with performance monitoring...")
    print("Dashboard will be available at: http://127.0.0.1:8080")
    print("Press Ctrl+C to stop training early\n")
    
    # Run training
    integrate_with_training_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        device=device
    )
    
    print("\nIntegration example completed!")
    print("\nGenerated files:")
    print("- monitoring_logs/ (contains all metrics)")
    print("- tensorboard_logs/ (TensorBoard data)")
    print("- visualizations/ (performance plots)")


if __name__ == "__main__":
    main()