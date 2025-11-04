#!/usr/bin/env python3
"""
Setup Script for Performance Monitoring System

This script helps set up the performance monitoring system by:
1. Installing required dependencies
2. Creating configuration files
3. Setting up directory structure
4. Running basic tests
5. Providing usage examples
"""

import argparse
import os
import subprocess
import sys
import yaml
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-monitoring.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def setup_directory_structure():
    """Create necessary directories."""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "monitoring_logs",
        "tensorboard_logs", 
        "visualizations",
        "configs",
        "reports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def create_sample_config():
    """Create sample configuration file."""
    print("\n⚙️ Creating sample configuration...")
    
    config_path = Path("configs/monitoring_config.yaml")
    
    # Check if config already exists
    if config_path.exists():
        print(f"⚠️ Configuration file already exists: {config_path}")
        overwrite = input("Overwrite? (y/N): ").lower() == 'y'
        if not overwrite:
            print("✅ Keeping existing configuration")
            return True
    
    # Create sample configuration
    config = {
        # Basic settings
        'save_dir': './monitoring_logs',
        'tensorboard_dir': './tensorboard_logs',
        'visualization_dir': './visualizations',
        
        # Dashboard settings
        'dashboard': {
            'enabled': True,
            'host': '127.0.0.1',
            'port': 8080
        },
        
        # Monitoring settings
        'performance_monitor': {
            'enabled': True,
            'monitor_system': True,
            'track_gradients': True
        },
        
        # Metrics collection
        'metrics_collection': {
            'enabled': True,
            'buffer_size': 1000,
            'flush_interval': 10.0
        },
        
        # Alert system
        'alert_system': {
            'enabled': True,
            'thresholds': {
                'cpu_usage_percent': 90.0,
                'memory_usage_percent': 85.0,
                'gpu_memory_usage_percent': 90.0,
                'grad_norm_exploded': 10.0
            }
        },
        
        # Visualization
        'visualization': {
            'enabled': True,
            'chart_types': [
                'training_progress',
                'system_monitoring',
                'interactive_dashboard'
            ]
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created configuration file: {config_path}")
    return True


def create_sample_script():
    """Create sample monitoring script."""
    print("\n📝 Creating sample script...")
    
    script_path = Path("examples/simple_monitoring_example.py")
    
    script_content = '''#!/usr/bin/env python3
"""
Simple Performance Monitoring Example

This script demonstrates basic usage of the performance monitoring system.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../utils')

from monitor_training import TrainingMonitor
from performance_monitor import TrainingMetrics, SystemMetrics

def create_dummy_model():
    """Create a simple dummy model."""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

def simulate_training():
    """Simulate training with monitoring."""
    
    # Configuration
    config = {
        'save_dir': './monitoring_logs',
        'enable_dashboard': True,
        'dashboard_port': 8080,
        'enable_alerts': True,
        'buffer_size': 100,
        'flush_interval': 5.0
    }
    
    # Create and start monitor
    print("🚀 Starting performance monitoring...")
    monitor = TrainingMonitor(config)
    monitor.start_monitoring()
    
    # Create dummy model and optimizer
    model = create_dummy_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        print("📊 Training simulation started...")
        print("📱 Dashboard available at: http://127.0.0.1:8080")
        print("⏹️ Press Ctrl+C to stop")
        print()
        
        for epoch in range(5):
            print(f"Epoch {epoch + 1}/5")
            
            for step in range(20):
                # Simulate training metrics
                base_loss = 2.0 * (0.9 ** (epoch * 20 + step))
                loss = base_loss + np.random.normal(0, 0.1)
                accuracy = min(0.95, 0.1 + (epoch * 20 + step) * 0.01) + np.random.normal(0, 0.05)
                learning_rate = 0.001 * (0.95 ** (epoch * 5))
                
                # Log training step
                monitor.log_training_step(
                    model=model,
                    optimizer=optimizer,
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=learning_rate,
                    custom_metrics={
                        'batch_size': 32,
                        'throughput': 32 / (0.5 + step * 0.01)
                    }
                )
                
                # Print progress
                if step % 5 == 0:
                    print(f"  Step {step + 1}/20 - Loss: {loss:.4f}, Acc: {accuracy:.3f}, LR: {learning_rate:.2e}")
                
                time.sleep(0.2)  # Simulate training time
            
            # Update epoch in monitor
            monitor.current_epoch = epoch + 1
            print()
        
        print("✅ Training simulation completed!")
        
    except KeyboardInterrupt:
        print("\\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
    finally:
        monitor.stop_monitoring()
        print("📊 Monitoring stopped")

if __name__ == "__main__":
    simulate_training()
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"✅ Created sample script: {script_path}")
    return True


def run_basic_test():
    """Run basic functionality test."""
    print("\n🧪 Running basic functionality test...")
    
    try:
        # Test imports
        sys.path.append('../utils')
        
        from performance_monitor import PerformanceMonitor, TrainingMetrics
        from metrics_collector import RealTimeMetricsCollector
        from performance_recommendations import PerformanceAnalyzer
        
        print("✅ All modules imported successfully")
        
        # Test creating a simple monitor
        config = {
            'save_dir': './test_monitor',
            'enable_performance_monitor': True,
            'enable_metrics_collector': True
        }
        
        monitor = PerformanceMonitor(**config)
        print("✅ Performance monitor created successfully")
        
        # Test metrics collector
        collector = RealTimeMetricsCollector('./test_collector.db')
        print("✅ Metrics collector created successfully")
        
        # Test performance analyzer
        analyzer = PerformanceAnalyzer()
        print("✅ Performance analyzer created successfully")
        
        # Cleanup test files
        import shutil
        shutil.rmtree('./test_monitor', ignore_errors=True)
        Path('./test_collector.db').unlink(missing_ok=True)
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def print_usage_examples():
    """Print usage examples."""
    print("\n📚 Usage Examples:")
    print("=" * 50)
    
    print("\n1. Basic Usage with Configuration File:")
    print("   python scripts/monitor_training.py --config configs/monitoring_config.yaml")
    
    print("\n2. Simulation Mode (No Real Training):")
    print("   python scripts/monitor_training.py --simulate --simulate_steps 100")
    
    print("\n3. Integration with Training Code:")
    print("   # In your training loop:")
    print("   from monitor_training import TrainingMonitor")
    print("   ")
    print("   monitor = TrainingMonitor('configs/monitoring_config.yaml')")
    print("   monitor.start_monitoring()")
    print("   ")
    print("   # In training loop:")
    print("   monitor.log_training_step(model, optimizer, loss, accuracy)")
    
    print("\n4. Using with Custom Configuration:")
    print("   python scripts/monitor_training.py \\")
    print("     --dashboard_port 8080 \\")
    print("     --enable_alerts \\")
    print("     --save_dir ./custom_logs")
    
    print("\n5. Accessing Dashboard:")
    print("   🌐 Open http://127.0.0.1:8080 in your browser")
    
    print("\n6. Running Sample Example:")
    print("   python examples/simple_monitoring_example.py")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Performance Monitoring System")
    parser.add_argument('--skip-deps', action='store_true', help='Skip dependency installation')
    parser.add_argument('--skip-test', action='store_true', help='Skip basic tests')
    parser.add_argument('--example-only', action='store_true', help='Only create example files')
    
    args = parser.parse_args()
    
    print("🚀 Setting up Performance Monitoring System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    success = True
    
    # Install dependencies unless skipped
    if not args.skip_deps and not args.example_only:
        if not install_dependencies():
            success = False
    
    # Setup directory structure
    if not setup_directory_structure():
        success = False
    
    # Create configuration files
    if not create_sample_config():
        success = False
    
    # Create sample script
    if not create_sample_script():
        success = False
    
    # Run basic tests unless skipped
    if not args.skip_test and not args.example_only:
        if not run_basic_test():
            success = False
    
    # Print usage examples
    print_usage_examples()
    
    # Final status
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Review configuration in configs/monitoring_config.yaml")
        print("2. Run the sample script: python examples/simple_monitoring_example.py")
        print("3. Integrate with your training code")
        print("4. Check the dashboard at http://127.0.0.1:8080")
    else:
        print("❌ Setup completed with errors")
        print("\n🔧 Troubleshooting:")
        print("1. Check Python version (3.8+ required)")
        print("2. Ensure pip is available")
        print("3. Check network connectivity for package installation")
        print("4. Review error messages above")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())