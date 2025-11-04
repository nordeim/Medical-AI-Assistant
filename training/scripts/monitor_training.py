#!/usr/bin/env python3
"""
ML Training Performance Monitoring Script

This script provides comprehensive real-time monitoring for machine learning training,
including a web-based dashboard, automated reporting, and performance alerts.

Usage:
    python monitor_training.py --config config.yaml
    python monitor_training.py --model model.pth --data data_loader --port 8080
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml

# Add training utils to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.performance_monitor import (
        PerformanceMonitor, TrainingMetrics, SystemMetrics, ModelPerformanceMetrics,
        SystemMonitor, GradientMonitor, ModelProfiler, create_performance_monitor,
        monitor_training_step
    )
    from utils.metrics_collector import (
        RealTimeMetricsCollector, MetricsDatabase, DataVisualizer, AlertSystem,
        MonitoringDashboard, create_metrics_collector, create_dashboard
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install numpy pandas matplotlib seaborn plotly flask flask-socketio")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_monitor.log')
    ]
)

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Main training monitoring class that orchestrates all monitoring components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stop_event = threading.Event()
        
        # Initialize components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.metrics_collector: Optional[RealTimeMetricsCollector] = None
        self.alert_system: Optional[AlertSystem] = None
        self.dashboard: Optional[MonitoringDashboard] = None
        self.visualizer: Optional[DataVisualizer] = None
        
        # State tracking
        self.is_monitoring = False
        self.training_start_time: Optional[float] = None
        self.current_epoch = 0
        self.current_step = 0
        
        # Setup components based on configuration
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup all monitoring components based on configuration."""
        
        # Create directories
        dirs_to_create = [
            self.config.get('save_dir', './monitoring_logs'),
            self.config.get('tensorboard_dir', './tensorboard_logs'),
            self.config.get('visualization_dir', './visualizations'),
            self.config.get('dashboard_host', '127.0.0.1')
        ]
        
        for dir_path in dirs_to_create:
            if isinstance(dir_path, str):
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize performance monitor
        if self.config.get('enable_performance_monitor', True):
            self.performance_monitor = create_performance_monitor(
                save_dir=self.config['save_dir'],
                tensorboard_dir=self.config.get('tensorboard_dir'),
                monitor_system=self.config.get('monitor_system', True),
                track_gradients=self.config.get('track_gradients', True)
            )
            logger.info("Performance monitor initialized")
        
        # Initialize metrics collector
        if self.config.get('enable_metrics_collector', True):
            db_path = Path(self.config['save_dir']) / "metrics.db"
            self.metrics_collector = create_metrics_collector(
                db_path=db_path,
                buffer_size=self.config.get('buffer_size', 1000),
                flush_interval=self.config.get('flush_interval', 10.0)
            )
            
            # Add callbacks for real-time processing
            self.metrics_collector.add_callback('training', self._on_training_metrics)
            self.metrics_collector.add_callback('system', self._on_system_metrics)
            self.metrics_collector.add_callback('alert', self._on_alert)
            
            logger.info("Metrics collector initialized")
        
        # Initialize alert system
        if self.config.get('enable_alerts', True) and self.metrics_collector:
            self.alert_system = AlertSystem(self.metrics_collector)
            
            # Add custom alert rules from config
            custom_rules = self.config.get('custom_alert_rules', [])
            for rule in custom_rules:
                self.alert_system.add_alert_rule(**rule)
            
            logger.info("Alert system initialized")
        
        # Initialize visualizer
        if self.config.get('enable_visualizations', True):
            self.visualizer = DataVisualizer(
                output_dir=self.config.get('visualization_dir', './visualizations')
            )
            logger.info("Data visualizer initialized")
        
        # Initialize dashboard
        if self.config.get('enable_dashboard', True) and self.metrics_collector:
            self.dashboard = create_dashboard(
                metrics_collector=self.metrics_collector,
                host=self.config.get('dashboard_host', '127.0.0.1'),
                port=self.config.get('dashboard_port', 8080)
            )
            logger.info("Web dashboard initialized")
    
    def start_monitoring(self):
        """Start all monitoring components."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.training_start_time = time.time()
        
        # Start performance monitor
        if self.performance_monitor:
            self.performance_monitor.start_monitoring()
        
        # Start metrics collection
        if self.metrics_collector:
            self.metrics_collector.start_collection()
        
        # Start web dashboard
        if self.dashboard:
            self.dashboard.start_dashboard()
        
        logger.info("Training monitoring started")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Stop performance monitor
        if self.performance_monitor:
            self.performance_monitor.stop_monitoring()
        
        # Stop metrics collection
        if self.metrics_collector:
            self.metrics_collector.stop_collection()
        
        # Save final metrics and generate reports
        if self.performance_monitor:
            self.performance_monitor.save_metrics()
        
        if self.visualizer and self.metrics_collector:
            self._generate_final_reports()
        
        logger.info("Training monitoring stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop_event.set()
        self.stop_monitoring()
        sys.exit(0)
    
    def log_training_step(self, 
                         model,
                         optimizer,
                         loss: float,
                         accuracy: Optional[float] = None,
                         learning_rate: Optional[float] = None,
                         custom_metrics: Optional[Dict[str, float]] = None,
                         batch_data: Optional[Any] = None):
        """Log a training step with all monitoring components."""
        
        if not self.is_monitoring:
            return
        
        # Create training metrics
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            global_step=self.current_step,  # Simplified for demo
            phase="train",
            loss=loss,
            accuracy=accuracy or 0.0,
            learning_rate=learning_rate or (optimizer.param_groups[0]['lr'] if optimizer else 0.0)
        )
        
        # Monitor gradients if performance monitor is available
        if self.performance_monitor and hasattr(model, 'parameters'):
            try:
                grad_stats = self.performance_monitor.gradient_monitor.monitor_step(model) if self.performance_monitor.gradient_monitor else {}
                metrics.grad_norm = grad_stats.get('grad_norm', 0.0)
            except Exception as e:
                logger.warning(f"Failed to monitor gradients: {e}")
        
        # Collect metrics in real-time
        if self.metrics_collector:
            self.metrics_collector.collect_training_metrics(metrics)
        
        # Log with performance monitor
        if self.performance_monitor:
            try:
                result = self.performance_monitor.log_training_step(metrics, custom_metrics)
                
                # Trigger alerts if any
                if self.alert_system and result.get('alerts'):
                    for alert_msg in result['alerts']:
                        self.metrics_collector.add_alert(
                            alert_type="Performance Alert",
                            severity="warning",
                            message=alert_msg
                        )
            except Exception as e:
                logger.error(f"Failed to log training step: {e}")
        
        self.current_step += 1
    
    def log_validation_step(self, loss: float, accuracy: float, custom_metrics: Optional[Dict[str, float]] = None):
        """Log validation step."""
        
        if not self.is_monitoring:
            return
        
        metrics = TrainingMetrics(
            epoch=self.current_epoch,
            step=self.current_step,
            global_step=self.current_step,
            phase="val",
            loss=loss,
            accuracy=accuracy
        )
        
        if self.metrics_collector:
            self.metrics_collector.collect_training_metrics(metrics)
    
    def log_model_inference_benchmark(self, model, device, input_data):
        """Log model inference performance benchmark."""
        
        if not self.is_monitoring or not self.performance_monitor:
            return
        
        try:
            self.performance_monitor.set_model(model, device)
            
            # Run benchmark
            benchmark_results = self.performance_monitor.model_profiler.benchmark_inference(input_data)
            
            # Collect metrics
            if self.metrics_collector:
                self.metrics_collector.collect_model_performance_metrics(benchmark_results)
            
            logger.info(f"Inference benchmark completed: {benchmark_results.samples_per_second:.1f} samples/sec")
            
        except Exception as e:
            logger.error(f"Failed to run inference benchmark: {e}")
    
    def _on_training_metrics(self, metrics: TrainingMetrics):
        """Callback for training metrics."""
        # Custom processing can be added here
        pass
    
    def _on_system_metrics(self, metrics: SystemMetrics):
        """Callback for system metrics."""
        # Evaluate alert rules
        if self.alert_system:
            self.alert_system.evaluate_rules(system_metrics=metrics)
    
    def _on_alert(self, alert: Dict[str, Any]):
        """Callback for alerts."""
        logger.warning(f"Alert: {alert['severity'].upper()} - {alert['message']}")
    
    def _generate_final_reports(self):
        """Generate final performance reports and visualizations."""
        
        try:
            # Get historical data
            end_time = time.time()
            start_time = end_time - 3600  # Last hour
            
            training_metrics = self.metrics_collector.db.get_training_metrics(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            system_metrics = self.metrics_collector.db.get_system_metrics(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            # Generate visualizations
            if training_metrics:
                training_plots = self.visualizer.create_training_progress_plots(training_metrics)
                logger.info(f"Generated training plots: {list(training_plots.keys())}")
            
            if system_metrics:
                system_plots = self.visualizer.create_system_monitoring_plots(system_metrics)
                logger.info(f"Generated system plots: {list(system_plots.keys())}")
            
            # Create interactive dashboard
            dashboard_file = self.visualizer.create_interactive_dashboard(
                training_metrics, system_metrics
            )
            logger.info(f"Interactive dashboard created: {dashboard_file}")
            
            # Performance comparison (if benchmark data available)
            # This could be extended to compare different training runs
            
            # Export data
            export_path = Path(self.config.get('save_dir', './monitoring_logs')) / "final_export"
            self.metrics_collector.export_data(
                export_path,
                metric_type="all",
                format="json"
            )
            
            # Generate summary report
            self._generate_summary_report(training_metrics, system_metrics)
            
        except Exception as e:
            logger.error(f"Failed to generate final reports: {e}")
    
    def _generate_summary_report(self, training_metrics: List[TrainingMetrics], system_metrics: List[SystemMetrics]):
        """Generate a summary performance report."""
        
        report = {
            'monitoring_session': {
                'start_time': datetime.fromtimestamp(self.training_start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_hours': (time.time() - self.training_start_time) / 3600,
                'total_steps': self.current_step
            },
            'training_summary': {},
            'system_summary': {},
            'recommendations': []
        }
        
        # Training summary
        if training_metrics:
            training_data = [m for m in training_metrics if m.phase == 'train']
            if training_data:
                final_loss = training_data[-1].loss
                initial_loss = training_data[0].loss
                improvement = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
                
                report['training_summary'] = {
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'loss_improvement_percent': improvement,
                    'final_accuracy': training_data[-1].accuracy,
                    'total_training_steps': len(training_data),
                    'average_gradient_norm': sum(m.grad_norm for m in training_data) / len(training_data) if training_data else 0
                }
        
        # System summary
        if system_metrics:
            avg_cpu = sum(m.cpu_usage_percent for m in system_metrics) / len(system_metrics)
            avg_memory = sum(m.memory_usage_percent for m in system_metrics) / len(system_metrics)
            max_gpu_util = max(m.gpu_utilization for m in system_metrics if m.gpu_utilization > 0) if system_metrics else 0
            
            report['system_summary'] = {
                'average_cpu_usage': avg_cpu,
                'average_memory_usage': avg_memory,
                'peak_gpu_utilization': max_gpu_util,
                'total_system_samples': len(system_metrics)
            }
        
        # Generate recommendations
        recommendations = []
        
        if training_metrics and len(training_metrics) > 1:
            # Check for gradient issues
            avg_grad_norm = sum(m.grad_norm for m in training_metrics) / len(training_metrics)
            if avg_grad_norm > 5.0:
                recommendations.append("Consider gradient clipping or reducing learning rate due to high gradient norms.")
            elif avg_grad_norm < 0.001:
                recommendations.append("Gradient norms are very low. Consider increasing learning rate.")
        
        if system_metrics:
            avg_memory = sum(m.memory_usage_percent for m in system_metrics) / len(system_metrics)
            if avg_memory > 80:
                recommendations.append("High memory usage detected. Consider reducing batch size or using gradient accumulation.")
            
            avg_cpu = sum(m.cpu_usage_percent for m in system_metrics) / len(system_metrics)
            if avg_cpu < 50:
                recommendations.append("CPU utilization is low. Consider increasing batch size or data loading workers.")
        
        report['recommendations'] = recommendations
        
        # Save report
        report_path = Path(self.config.get('save_dir', './monitoring_logs')) / f"performance_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("TRAINING PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Session Duration: {report['monitoring_session']['duration_hours']:.2f} hours")
        print(f"Total Steps: {report['monitoring_session']['total_steps']}")
        
        if report['training_summary']:
            ts = report['training_summary']
            print(f"Loss Improvement: {ts['loss_improvement_percent']:.1f}%")
            print(f"Final Accuracy: {ts['final_accuracy']:.3f}")
        
        if report['system_summary']:
            ss = report['system_summary']
            print(f"Average CPU Usage: {ss['average_cpu_usage']:.1f}%")
            print(f"Average Memory Usage: {ss['average_memory_usage']:.1f}%")
            print(f"Peak GPU Utilization: {ss['peak_gpu_utilization']:.1f}%")
        
        if recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        print("="*60)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        status = {
            'is_monitoring': self.is_monitoring,
            'start_time': self.training_start_time,
            'current_epoch': self.current_epoch,
            'current_step': self.current_step,
            'components': {}
        }
        
        if self.metrics_collector:
            status['components']['metrics_collector'] = self.metrics_collector.get_statistics()
        
        if self.performance_monitor:
            status['components']['performance_monitor'] = self.performance_monitor.get_performance_summary()
        
        if self.alert_system:
            status['components']['alert_system'] = self.alert_system.get_alert_statistics()
        
        return status


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration."""
    return {
        # Directories
        'save_dir': './monitoring_logs',
        'tensorboard_dir': './tensorboard_logs',
        'visualization_dir': './visualizations',
        
        # Dashboard settings
        'enable_dashboard': True,
        'dashboard_host': '127.0.0.1',
        'dashboard_port': 8080,
        
        # Monitoring settings
        'enable_performance_monitor': True,
        'enable_metrics_collector': True,
        'enable_alerts': True,
        'enable_visualizations': True,
        'monitor_system': True,
        'track_gradients': True,
        
        # Collection settings
        'buffer_size': 1000,
        'flush_interval': 10.0,
        
        # Alert thresholds
        'alert_thresholds': {
            'cpu_usage': 90.0,
            'memory_usage': 85.0,
            'gpu_memory_usage': 90.0,
            'temperature': 80.0,
            'grad_norm_exploded': 10.0
        },
        
        # Custom alert rules (can be added via config)
        'custom_alert_rules': []
    }


def main():
    """Main entry point for the monitoring script."""
    parser = argparse.ArgumentParser(description="ML Training Performance Monitor")
    
    # Configuration options
    parser.add_argument('--config', type=str, help='Configuration file path (YAML)')
    parser.add_argument('--save_dir', type=str, default='./monitoring_logs', help='Save directory')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboard_logs', help='TensorBoard log directory')
    parser.add_argument('--visualization_dir', type=str, default='./visualizations', help='Visualization output directory')
    
    # Dashboard options
    parser.add_argument('--enable_dashboard', action='store_true', help='Enable web dashboard')
    parser.add_argument('--disable_dashboard', dest='enable_dashboard', action='store_false', help='Disable web dashboard')
    parser.set_defaults(enable_dashboard=True)
    parser.add_argument('--dashboard_host', type=str, default='127.0.0.1', help='Dashboard host')
    parser.add_argument('--dashboard_port', type=int, default=8080, help='Dashboard port')
    
    # Monitoring options
    parser.add_argument('--enable_performance_monitor', action='store_true', help='Enable performance monitor')
    parser.add_argument('--disable_performance_monitor', dest='enable_performance_monitor', action='store_false', help='Disable performance monitor')
    parser.set_defaults(enable_performance_monitor=True)
    
    parser.add_argument('--monitor_system', action='store_true', help='Monitor system resources')
    parser.add_argument('--no_monitor_system', dest='monitor_system', action='store_false', help='Disable system monitoring')
    parser.set_defaults(monitor_system=True)
    
    parser.add_argument('--track_gradients', action='store_true', help='Track gradient norms')
    parser.add_argument('--no_track_gradients', dest='track_gradients', action='store_false', help='Disable gradient tracking')
    parser.set_defaults(track_gradients=True)
    
    # Alert options
    parser.add_argument('--enable_alerts', action='store_true', help='Enable alerts')
    parser.add_argument('--disable_alerts', dest='enable_alerts', action='store_false', help='Disable alerts')
    parser.set_defaults(enable_alerts=True)
    
    # Simulation options
    parser.add_argument('--simulate', action='store_true', help='Run simulation instead of real monitoring')
    parser.add_argument('--simulate_steps', type=int, default=100, help='Number of simulation steps')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if key.startswith('enable_') or key in ['save_dir', 'tensorboard_dir', 'visualization_dir', 
                                              'dashboard_host', 'dashboard_port', 'monitor_system', 'track_gradients']:
            if value is not None:
                config[key] = value
    
    # Create monitor
    monitor = TrainingMonitor(config)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        if args.simulate:
            logger.info("Running simulation...")
            _run_simulation(monitor, args.simulate_steps)
        else:
            logger.info("Monitoring started. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            try:
                while not monitor.stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
        
    except Exception as e:
        logger.error(f"Error in monitoring: {e}")
        raise
    finally:
        monitor.stop_monitoring()
        logger.info("Monitoring script completed")


def _run_simulation(monitor: TrainingMonitor, num_steps: int):
    """Run a simulation of training with monitoring."""
    
    logger.info(f"Starting simulation for {num_steps} steps...")
    
    # Simulate model and optimizer (dummy objects)
    class DummyModel:
        def parameters(self):
            # Return some dummy parameters
            return [torch.randn(10, 10, requires_grad=True) for _ in range(5)]
    
    class DummyOptimizer:
        def __init__(self):
            self.param_groups = [{'lr': 1e-4}]
    
    model = DummyModel()
    optimizer = DummyOptimizer()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate training steps
    for step in range(num_steps):
        # Simulate training metrics
        epoch = step // 20
        initial_loss = 2.0
        loss = initial_loss * (0.95 ** step) + 0.1 * (1 - 0.95 ** step) + 0.05 * (step % 10) * 0.01
        accuracy = min(0.95, 0.1 + step * 0.01)
        learning_rate = 1e-4 * (0.95 ** (step // 10))
        
        custom_metrics = {
            'batch_size': 32,
            'tokens_processed': step * 32,
            'throughput': 32 / (0.1 + step * 0.001)
        }
        
        # Log training step
        monitor.log_training_step(
            model=model,
            optimizer=optimizer,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate,
            custom_metrics=custom_metrics
        )
        
        # Simulate validation every 20 steps
        if step % 20 == 19:
            val_loss = loss * 1.1  # Slightly higher validation loss
            val_accuracy = accuracy * 0.95  # Slightly lower validation accuracy
            monitor.log_validation_step(val_loss, val_accuracy)
        
        # Print progress
        if step % 10 == 0:
            print(f"Simulated step {step}/{num_steps}: Loss={loss:.4f}, Accuracy={accuracy:.3f}")
        
        time.sleep(0.1)  # Simulate processing time
    
    # Final status
    status = monitor.get_monitoring_status()
    print(f"\nSimulation completed!")
    print(f"Final step: {status['current_step']}")
    print(f"Training metrics collected: {status['components'].get('metrics_collector', {}).get('collected_training', 0)}")
    
    monitor.stop_monitoring()


if __name__ == "__main__":
    # Import torch for simulation
    try:
        import torch
    except ImportError:
        print("PyTorch not available. Install with: pip install torch")
        sys.exit(1)
    
    main()