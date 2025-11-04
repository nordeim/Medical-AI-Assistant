"""
Checkpoint Analytics and Reporting System

This module provides comprehensive analytics and reporting for checkpoint management:
- Performance analytics
- Storage utilization analysis
- Training progress insights
- Cloud storage metrics
- Automated reporting

Author: Medical AI Assistant Team
Date: 2025-11-04
"""

import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import seaborn as sns
from collections import defaultdict

from .checkpoint_manager import CheckpointManager, CheckpointMetadata
from .training_state import TrainingState, TrainingMetrics

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsConfig:
    """Configuration for analytics"""
    # Analysis settings
    auto_generate_reports: bool = True
    report_interval_hours: int = 24
    
    # Visualization settings
    plot_style: str = "seaborn-v0_8"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    # Data retention
    analytics_retention_days: int = 90
    
    # Report formats
    report_formats: List[str] = None
    output_directory: str = "analytics"
    
    def __post_init__(self):
        if self.report_formats is None:
            self.report_formats = ["html", "pdf"]


class CheckpointAnalytics:
    """Advanced checkpoint analytics and reporting"""
    
    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        training_state: TrainingState,
        output_dir: Union[str, Path],
        config: Optional[AnalyticsConfig] = None
    ):
        self.checkpoint_manager = checkpoint_manager
        self.training_state = training_state
        self.output_dir = Path(output_dir)
        self.config = config or AnalyticsConfig()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # Set style
        plt.style.use(self.config.plot_style)
        
        logger.info("CheckpointAnalytics initialized")
    
    def generate_comprehensive_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive analytics report"""
        logger.info("Generating comprehensive analytics report...")
        
        try:
            analytics_data = {
                "generated_at": datetime.now().isoformat(),
                "experiment": self.training_state.experiment_name,
                "summary": self._generate_summary_stats(),
                "training_progress": self._analyze_training_progress(),
                "checkpoint_analysis": self._analyze_checkpoints(),
                "storage_analytics": self._analyze_storage_usage(),
                "performance_trends": self._analyze_performance_trends(),
                "recommendations": self._generate_recommendations()
            }
            
            # Generate visualizations
            self._create_visualizations(analytics_data)
            
            # Generate reports
            self._generate_reports(analytics_data)
            
            logger.info("Comprehensive analytics report generated successfully")
            return analytics_data
            
        except Exception as e:
            logger.error(f"Failed to generate analytics: {e}")
            return {}
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return {"message": "No checkpoint data available"}
            
            # Basic statistics
            total_checkpoints = len(metadata_list)
            total_size_mb = sum(m.file_size for m in metadata_list) / 1024 / 1024
            
            # Time-based statistics
            timestamps = [datetime.fromisoformat(m.timestamp) for m in metadata_list]
            start_time = min(timestamps)
            end_time = max(timestamps)
            total_duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # Storage statistics
            compressed_count = sum(1 for m in metadata_list if m.compression == "gzip")
            cloud_backups = sum(1 for m in metadata_list if m.cloud_backup)
            
            # Performance statistics
            metrics_data = defaultdict(list)
            for m in metadata_list:
                for metric_name, value in m.metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_data[metric_name].append(value)
            
            metric_stats = {}
            for metric_name, values in metrics_data.items():
                if values:
                    metric_stats[metric_name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "trend": "improving" if self._calculate_trend(values) > 0 else "declining"
                    }
            
            return {
                "total_checkpoints": total_checkpoints,
                "total_storage_mb": total_size_mb,
                "training_duration_hours": total_duration_hours,
                "checkpoint_frequency": total_checkpoints / max(total_duration_hours, 1),
                "compression_ratio": compressed_count / total_checkpoints if total_checkpoints > 0 else 0,
                "cloud_backup_ratio": cloud_backups / total_checkpoints if total_checkpoints > 0 else 0,
                "metric_statistics": metric_stats,
                "latest_checkpoint": {
                    "id": metadata_list[-1].checkpoint_id if metadata_list else None,
                    "epoch": metadata_list[-1].epoch if metadata_list else 0,
                    "timestamp": metadata_list[-1].timestamp if metadata_list else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summary stats: {e}")
            return {}
    
    def _analyze_training_progress(self) -> Dict[str, Any]:
        """Analyze training progress"""
        try:
            metrics_history = self.training_state.get_metrics_history()
            
            if not metrics_history:
                return {"message": "No training metrics available"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame([m.to_dict() for m in metrics_history])
            
            # Progress analysis
            progress_analysis = {
                "total_epochs": df['epoch'].max() if not df.empty else 0,
                "current_epoch": df['epoch'].iloc[-1] if not df.empty else 0,
                "total_steps": df['step'].max() if not df.empty else 0,
                "current_step": df['step'].iloc[-1] if not df.empty else 0,
                "training_phases": df['phase'].unique().tolist(),
                "last_update": df['timestamp'].iloc[-1] if not df.empty else None
            }
            
            # Loss progression
            if 'loss' in df.columns:
                loss_progression = {
                    "initial_loss": df['loss'].iloc[0] if not df.empty else 0,
                    "current_loss": df['loss'].iloc[-1] if not df.empty else 0,
                    "loss_improvement": df['loss'].iloc[0] - df['loss'].iloc[-1] if not df.empty else 0,
                    "improvement_percentage": ((df['loss'].iloc[0] - df['loss'].iloc[-1]) / df['loss'].iloc[0] * 100) if not df.empty and df['loss'].iloc[0] > 0 else 0
                }
                progress_analysis["loss_progression"] = loss_progression
            
            # Accuracy progression
            if 'accuracy' in df.columns:
                accuracy_progression = {
                    "initial_accuracy": df['accuracy'].iloc[0] if not df.empty else 0,
                    "current_accuracy": df['accuracy'].iloc[-1] if not df.empty else 0,
                    "accuracy_improvement": df['accuracy'].iloc[-1] - df['accuracy'].iloc[0] if not df.empty else 0
                }
                progress_analysis["accuracy_progression"] = accuracy_progression
            
            return progress_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze training progress: {e}")
            return {}
    
    def _analyze_checkpoints(self) -> Dict[str, Any]:
        """Analyze checkpoint patterns and usage"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return {"message": "No checkpoint data available"}
            
            # Pattern analysis
            checkpoint_analysis = {
                "distribution": {
                    "by_epoch": self._analyze_distribution_by_epoch(metadata_list),
                    "by_compression": self._analyze_distribution_by_compression(metadata_list),
                    "by_cloud_backup": self._analyze_distribution_by_cloud(metadata_list)
                },
                "size_analysis": self._analyze_checkpoint_sizes(metadata_list),
                "frequency_analysis": self._analyze_checkpoint_frequency(metadata_list),
                "validation_results": self._analyze_validation_results()
            }
            
            return checkpoint_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze checkpoints: {e}")
            return {}
    
    def _analyze_storage_usage(self) -> Dict[str, Any]:
        """Analyze storage usage patterns"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return {"message": "No storage data available"}
            
            storage_analysis = {
                "total_usage_mb": sum(m.file_size for m in metadata_list) / 1024 / 1024,
                "average_checkpoint_size_mb": np.mean([m.file_size for m in metadata_list]) / 1024 / 1024,
                "storage_trends": self._analyze_storage_trends(metadata_list),
                "compression_effectiveness": self._analyze_compression_effectiveness(metadata_list),
                "cloud_storage_analysis": self._analyze_cloud_storage(metadata_list)
            }
            
            return storage_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze storage: {e}")
            return {}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across checkpoints"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return {"message": "No performance data available"}
            
            # Collect performance metrics
            performance_data = defaultdict(list)
            epochs = []
            
            for m in metadata_list:
                epochs.append(m.epoch)
                for metric_name, value in m.metrics.items():
                    if isinstance(value, (int, float)):
                        performance_data[metric_name].append((m.epoch, value))
            
            trends_analysis = {}
            for metric_name, data_points in performance_data.items():
                if data_points:
                    values = [point[1] for point in data_points]
                    trend_slope = self._calculate_trend(values)
                    
                    trends_analysis[metric_name] = {
                        "trend_slope": trend_slope,
                        "trend_direction": "improving" if trend_slope > 0 else "declining",
                        "correlation_with_epoch": self._calculate_correlation([p[0] for p in data_points], [p[1] for p in data_points]),
                        "best_value": max(values),
                        "best_epoch": data_points[values.index(max(values))][0] if values else 0,
                        "latest_value": values[-1] if values else 0
                    }
            
            return trends_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze performance trends: {e}")
            return {}
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return [{"type": "info", "message": "No data available for analysis"}]
            
            # Storage recommendations
            total_size_mb = sum(m.file_size for m in metadata_list) / 1024 / 1024
            
            if total_size_mb > 10000:  # 10GB
                recommendations.append({
                    "type": "warning",
                    "category": "storage",
                    "priority": "high",
                    "message": f"High storage usage detected ({total_size_mb:.1f} MB). Consider enabling compression or cleanup old checkpoints.",
                    "action": "Enable compression and implement cleanup policies"
                })
            
            # Compression recommendations
            uncompressed_count = sum(1 for m in metadata_list if m.compression == "none")
            if uncompressed_count > 0:
                compression_ratio = uncompressed_count / len(metadata_list)
                recommendations.append({
                    "type": "suggestion",
                    "category": "optimization",
                    "priority": "medium",
                    "message": f"{uncompressed_count} checkpoints ({compression_ratio:.1%}) are not compressed.",
                    "action": "Enable checkpoint compression to save storage space"
                })
            
            # Frequency recommendations
            if len(metadata_list) > 1:
                time_diffs = []
                sorted_metadata = sorted(metadata_list, key=lambda x: x.timestamp)
                for i in range(1, len(sorted_metadata)):
                    t1 = datetime.fromisoformat(sorted_metadata[i-1].timestamp)
                    t2 = datetime.fromisoformat(sorted_metadata[i].timestamp)
                    time_diffs.append((t2 - t1).total_seconds())
                
                avg_interval_hours = np.mean(time_diffs) / 3600 if time_diffs else 0
                
                if avg_interval_hours > 48:  # Less than daily
                    recommendations.append({
                        "type": "suggestion",
                        "category": "frequency",
                        "priority": "low",
                        "message": f"Checkpoints are being saved infrequently (every {avg_interval_hours:.1f} hours).",
                        "action": "Consider increasing checkpoint frequency for better recovery options"
                    })
            
            # Cloud backup recommendations
            cloud_backup_ratio = sum(1 for m in metadata_list if m.cloud_backup) / len(metadata_list)
            if cloud_backup_ratio < 0.5:
                recommendations.append({
                    "type": "suggestion",
                    "category": "reliability",
                    "priority": "medium",
                    "message": f"Only {cloud_backup_ratio:.1%} of checkpoints have cloud backups.",
                    "action": "Enable cloud backup for better disaster recovery protection"
                })
            
            # Validation recommendations
            validation_failures = len([m for m in metadata_list if m.file_hash])
            if validation_failures == 0:
                recommendations.append({
                    "type": "warning",
                    "category": "integrity",
                    "priority": "high",
                    "message": "No checkpoint validation data found.",
                    "action": "Enable checkpoint validation to ensure data integrity"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            return [{"type": "error", "message": f"Failed to generate recommendations: {e}"}]
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Simple linear regression
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0
        
        return np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
    
    def _analyze_distribution_by_epoch(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze distribution of checkpoints by epoch"""
        epoch_counts = defaultdict(int)
        for m in metadata_list:
            epoch_counts[m.epoch] += 1
        
        return {
            "total_epochs": len(epoch_counts),
            "checkpoints_per_epoch": dict(epoch_counts),
            "most_active_epoch": max(epoch_counts.items(), key=lambda x: x[1])[0] if epoch_counts else 0
        }
    
    def _analyze_distribution_by_compression(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze distribution by compression status"""
        compression_counts = defaultdict(int)
        for m in metadata_list:
            compression_counts[m.compression] += 1
        
        return dict(compression_counts)
    
    def _analyze_distribution_by_cloud(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze distribution by cloud backup status"""
        cloud_backups = sum(1 for m in metadata_list if m.cloud_backup)
        local_only = len(metadata_list) - cloud_backups
        
        return {
            "cloud_backups": cloud_backups,
            "local_only": local_only,
            "cloud_ratio": cloud_backups / len(metadata_list) if metadata_list else 0
        }
    
    def _analyze_checkpoint_sizes(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze checkpoint sizes"""
        sizes = [m.file_size for m in metadata_list]
        
        return {
            "average_size_mb": np.mean(sizes) / 1024 / 1024 if sizes else 0,
            "median_size_mb": np.median(sizes) / 1024 / 1024 if sizes else 0,
            "size_variance": np.var(sizes),
            "largest_checkpoint_mb": max(sizes) / 1024 / 1024 if sizes else 0,
            "smallest_checkpoint_mb": min(sizes) / 1024 / 1024 if sizes else 0
        }
    
    def _analyze_checkpoint_frequency(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze checkpoint frequency patterns"""
        if len(metadata_list) < 2:
            return {"message": "Insufficient data for frequency analysis"}
        
        # Time intervals between checkpoints
        timestamps = [datetime.fromisoformat(m.timestamp) for m in metadata_list]
        timestamps.sort()
        
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        return {
            "average_interval_hours": np.mean(intervals) / 3600 if intervals else 0,
            "interval_variance": np.var(intervals),
            "most_frequent_interval_hours": np.median(intervals) / 3600 if intervals else 0,
            "frequency_trend": "increasing" if self._calculate_trend([1/i for i in intervals if i > 0]) > 0 else "decreasing"
        }
    
    def _analyze_validation_results(self) -> Dict[str, Any]:
        """Analyze checkpoint validation results"""
        analytics = self.checkpoint_manager.get_analytics()
        
        return {
            "validation_failures": analytics.get("validation_failures", 0),
            "success_rate": (analytics.get("total_checkpoints", 0) - analytics.get("validation_failures", 0)) / max(analytics.get("total_checkpoints", 1), 1)
        }
    
    def _analyze_storage_trends(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze storage usage trends"""
        if not metadata_list:
            return {}
        
        # Sort by timestamp
        sorted_metadata = sorted(metadata_list, key=lambda x: x.timestamp)
        
        # Calculate cumulative storage
        cumulative_sizes = []
        total_size = 0
        for m in sorted_metadata:
            total_size += m.file_size
            cumulative_sizes.append(total_size / 1024 / 1024)  # Convert to MB
        
        return {
            "growth_rate_mb_per_day": self._calculate_storage_growth_rate(cumulative_sizes),
            "current_total_mb": cumulative_sizes[-1] if cumulative_sizes else 0,
            "projected_7_days_mb": self._project_storage_usage(cumulative_sizes, 7) if cumulative_sizes else 0,
            "projected_30_days_mb": self._project_storage_usage(cumulative_sizes, 30) if cumulative_sizes else 0
        }
    
    def _analyze_compression_effectiveness(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze compression effectiveness"""
        compressed_sizes = []
        uncompressed_sizes = []
        
        for m in metadata_list:
            if m.compression == "gzip":
                compressed_sizes.append(m.file_size)
            else:
                uncompressed_sizes.append(m.file_size)
        
        return {
            "compressed_count": len(compressed_sizes),
            "uncompressed_count": len(uncompressed_sizes),
            "average_compressed_size_mb": np.mean(compressed_sizes) / 1024 / 1024 if compressed_sizes else 0,
            "average_uncompressed_size_mb": np.mean(uncompressed_sizes) / 1024 / 1024 if uncompressed_sizes else 0,
            "compression_ratio": np.mean(compressed_sizes) / np.mean(uncompressed_sizes) if compressed_sizes and uncompressed_sizes else 0
        }
    
    def _analyze_cloud_storage(self, metadata_list: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Analyze cloud storage usage"""
        cloud_backups = [m for m in metadata_list if m.cloud_backup]
        
        return {
            "cloud_backup_count": len(cloud_backups),
            "cloud_storage_used_mb": sum(m.file_size for m in cloud_backups) / 1024 / 1024,
            "cost_estimation": self._estimate_cloud_costs(cloud_backups),
            "upload_success_rate": len(cloud_backups) / len(metadata_list) if metadata_list else 0
        }
    
    def _calculate_storage_growth_rate(self, cumulative_sizes: List[float]) -> float:
        """Calculate storage growth rate"""
        if len(cumulative_sizes) < 2:
            return 0
        
        # Use linear regression on the last 10 data points
        recent_sizes = cumulative_sizes[-10:]
        x = np.arange(len(recent_sizes))
        
        slope, _ = np.polyfit(x, recent_sizes, 1)
        return slope
    
    def _project_storage_usage(self, cumulative_sizes: List[float], days: int) -> float:
        """Project storage usage for specified days"""
        if not cumulative_sizes:
            return 0
        
        # Simple linear projection based on recent trend
        recent_sizes = cumulative_sizes[-10:] if len(cumulative_sizes) >= 10 else cumulative_sizes
        x = np.arange(len(recent_sizes))
        
        slope, intercept = np.polyfit(x, recent_sizes, 1)
        projected_size = slope * (days * 24) + intercept  # Convert days to hours
        
        return max(projected_size, cumulative_sizes[-1])  # Don't project below current usage
    
    def _estimate_cloud_costs(self, cloud_backups: List[CheckpointMetadata]) -> Dict[str, Any]:
        """Estimate cloud storage costs (rough approximation)"""
        total_size_mb = sum(m.file_size for m in cloud_backups) / 1024 / 1024
        
        # Very rough cost estimates (USD per GB per month)
        cost_per_gb_month = {
            "aws_s3": 0.023,
            "azure_blob": 0.0184,
            "gcp_storage": 0.020
        }
        
        return {
            "total_size_gb": total_size_mb / 1024,
            "estimated_monthly_cost_aws": (total_size_mb / 1024) * cost_per_gb_month["aws_s3"],
            "estimated_monthly_cost_azure": (total_size_mb / 1024) * cost_per_gb_month["azure_blob"],
            "estimated_monthly_cost_gcp": (total_size_mb / 1024) * cost_per_gb_month["gcp_storage"]
        }
    
    def _create_visualizations(self, analytics_data: Dict[str, Any]) -> None:
        """Create visualization plots"""
        try:
            plots_dir = self.output_dir / "plots"
            
            # Training progress plot
            self._plot_training_progress(plots_dir)
            
            # Checkpoint frequency plot
            self._plot_checkpoint_frequency(plots_dir)
            
            # Storage usage plot
            self._plot_storage_usage(plots_dir)
            
            # Performance trends plot
            self._plot_performance_trends(plots_dir)
            
            # Cloud storage distribution plot
            self._plot_cloud_distribution(plots_dir)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
    
    def _plot_training_progress(self, plots_dir: Path) -> None:
        """Plot training progress over time"""
        try:
            metrics_history = self.training_state.get_metrics_history()
            
            if not metrics_history:
                return
            
            df = pd.DataFrame([m.to_dict() for m in metrics_history])
            
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
            fig.suptitle('Training Progress Analytics', fontsize=16)
            
            # Loss progression
            if 'loss' in df.columns:
                axes[0, 0].plot(df['epoch'], df['loss'], label='Training Loss')
                if 'val_loss' in df.columns:
                    axes[0, 0].plot(df['epoch'], df['val_loss'], label='Validation Loss')
                axes[0, 0].set_title('Loss Progression')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Accuracy progression
            if 'accuracy' in df.columns:
                axes[0, 1].plot(df['epoch'], df['accuracy'], label='Training Accuracy')
                if 'val_accuracy' in df.columns:
                    axes[0, 1].plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
                axes[0, 1].set_title('Accuracy Progression')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Learning rate
            if 'learning_rate' in df.columns:
                axes[1, 0].plot(df['epoch'], df['learning_rate'])
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True)
            
            # Step progression
            axes[1, 1].plot(df['epoch'], df['step'])
            axes[1, 1].set_title('Training Steps')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Step')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "training_progress.png", dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot training progress: {e}")
    
    def _plot_checkpoint_frequency(self, plots_dir: Path) -> None:
        """Plot checkpoint frequency analysis"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
            fig.suptitle('Checkpoint Analytics', fontsize=16)
            
            # Epoch distribution
            epochs = [m.epoch for m in metadata_list]
            axes[0, 0].hist(epochs, bins=20, alpha=0.7)
            axes[0, 0].set_title('Checkpoint Distribution by Epoch')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].grid(True)
            
            # File size distribution
            sizes_mb = [m.file_size / 1024 / 1024 for m in metadata_list]
            axes[0, 1].hist(sizes_mb, bins=20, alpha=0.7)
            axes[0, 1].set_title('Checkpoint Size Distribution')
            axes[0, 1].set_xlabel('Size (MB)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].grid(True)
            
            # Compression status
            compression_counts = {}
            for m in metadata_list:
                compression_counts[m.compression] = compression_counts.get(m.compression, 0) + 1
            
            axes[1, 0].pie(compression_counts.values(), labels=compression_counts.keys(), autopct='%1.1f%%')
            axes[1, 0].set_title('Compression Status')
            
            # Cloud backup status
            cloud_counts = {'Cloud Backup': 0, 'Local Only': 0}
            for m in metadata_list:
                if m.cloud_backup:
                    cloud_counts['Cloud Backup'] += 1
                else:
                    cloud_counts['Local Only'] += 1
            
            axes[1, 1].pie(cloud_counts.values(), labels=cloud_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Backup Location')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "checkpoint_analytics.png", dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot checkpoint frequency: {e}")
    
    def _plot_storage_usage(self, plots_dir: Path) -> None:
        """Plot storage usage trends"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return
            
            # Sort by timestamp
            sorted_metadata = sorted(metadata_list, key=lambda x: x.timestamp)
            
            fig, axes = plt.subplots(2, 1, figsize=(self.config.figure_size[0], self.config.figure_size[1] * 0.75), dpi=self.config.dpi)
            fig.suptitle('Storage Usage Analytics', fontsize=16)
            
            # Cumulative storage usage
            cumulative_sizes = []
            total_size = 0
            timestamps = []
            
            for m in sorted_metadata:
                total_size += m.file_size
                cumulative_sizes.append(total_size / 1024 / 1024)  # Convert to MB
                timestamps.append(datetime.fromisoformat(m.timestamp))
            
            axes[0].plot(timestamps, cumulative_sizes, marker='o')
            axes[0].set_title('Cumulative Storage Usage')
            axes[0].set_xlabel('Time')
            axes[0].set_ylabel('Storage Used (MB)')
            axes[0].grid(True)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Individual checkpoint sizes over time
            sizes_mb = [m.file_size / 1024 / 1024 for m in sorted_metadata]
            axes[1].scatter(timestamps, sizes_mb, alpha=0.6)
            axes[1].set_title('Individual Checkpoint Sizes')
            axes[1].set_xlabel('Time')
            axes[1].set_ylabel('Checkpoint Size (MB)')
            axes[1].grid(True)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "storage_usage.png", dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot storage usage: {e}")
    
    def _plot_performance_trends(self, plots_dir: Path) -> None:
        """Plot performance trend analysis"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return
            
            # Collect performance metrics
            performance_data = defaultdict(list)
            for m in metadata_list:
                for metric_name, value in m.metrics.items():
                    if isinstance(value, (int, float)):
                        performance_data[metric_name].append((m.epoch, value))
            
            if not performance_data:
                return
            
            num_metrics = len(performance_data)
            cols = min(2, num_metrics)
            rows = (num_metrics + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(self.config.figure_size[0], self.config.figure_size[1] * rows / 2), dpi=self.config.dpi)
            if num_metrics == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes] if cols == 1 else axes
            else:
                axes = axes.flatten()
            
            fig.suptitle('Performance Trends', fontsize=16)
            
            for i, (metric_name, data_points) in enumerate(performance_data.items()):
                if i >= len(axes):
                    break
                
                epochs = [point[0] for point in data_points]
                values = [point[1] for point in data_points]
                
                axes[i].plot(epochs, values, marker='o', linewidth=2)
                axes[i].set_title(f'{metric_name.title()} Trend')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name.title())
                axes[i].grid(True)
                
                # Add trend line
                if len(epochs) > 1:
                    z = np.polyfit(epochs, values, 1)
                    p = np.poly1d(z)
                    axes[i].plot(epochs, p(epochs), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.4f})')
                    axes[i].legend()
            
            # Hide unused subplots
            for i in range(num_metrics, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(plots_dir / "performance_trends.png", dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot performance trends: {e}")
    
    def _plot_cloud_distribution(self, plots_dir: Path) -> None:
        """Plot cloud storage distribution"""
        try:
            metadata_list = list(self.checkpoint_manager.metadata.values())
            
            if not metadata_list:
                return
            
            cloud_backups = [m for m in metadata_list if m.cloud_backup]
            local_only = [m for m in metadata_list if not m.cloud_backup]
            
            fig, axes = plt.subplots(1, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
            fig.suptitle('Cloud Storage Distribution', fontsize=16)
            
            # Backup location pie chart
            location_counts = {
                'Cloud Backup': len(cloud_backups),
                'Local Only': len(local_only)
            }
            
            axes[0].pie(location_counts.values(), labels=location_counts.keys(), autopct='%1.1f%%')
            axes[0].set_title('Backup Location Distribution')
            
            # Storage by location
            cloud_size = sum(m.file_size for m in cloud_backups) / 1024 / 1024
            local_size = sum(m.file_size for m in local_only) / 1024 / 1024
            
            storage_data = {
                'Cloud Backup': cloud_size,
                'Local Only': local_size
            }
            
            axes[1].bar(storage_data.keys(), storage_data.values())
            axes[1].set_title('Storage Usage by Location')
            axes[1].set_ylabel('Storage (MB)')
            
            plt.tight_layout()
            plt.savefig(plots_dir / "cloud_distribution.png", dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to plot cloud distribution: {e}")
    
    def _generate_reports(self, analytics_data: Dict[str, Any]) -> None:
        """Generate analytics reports in specified formats"""
        try:
            reports_dir = self.output_dir / "reports"
            
            if "html" in self.config.report_formats:
                self._generate_html_report(analytics_data, reports_dir / "analytics_report.html")
            
            if "pdf" in self.config.report_formats:
                self._generate_pdf_report(analytics_data, reports_dir / "analytics_report.pdf")
            
            if "json" in self.config.report_formats:
                self._generate_json_report(analytics_data, reports_dir / "analytics_data.json")
            
            logger.info("Analytics reports generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
    
    def _generate_html_report(self, analytics_data: Dict[str, Any], output_file: Path) -> None:
        """Generate HTML analytics report"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Checkpoint Analytics Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #333; }}
                    .section {{ margin: 30px 0; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .recommendation {{ background: #f0f8ff; padding: 10px; margin: 5px 0; border-left: 4px solid #007bff; }}
                    .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
                    .suggestion {{ background: #d4edda; border-left: 4px solid #28a745; }}
                    .plot {{ text-align: center; margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Checkpoint Analytics Report</h1>
                    <p>Generated: {analytics_data.get('generated_at', 'Unknown')}</p>
                    <p>Experiment: {analytics_data.get('experiment', 'Unknown')}</p>
                </div>
                
                <div class="section">
                    <h2>Summary Statistics</h2>
                    {self._html_summary_stats(analytics_data.get('summary', {}))}
                </div>
                
                <div class="section">
                    <h2>Training Progress</h2>
                    {self._html_training_progress(analytics_data.get('training_progress', {}))}
                </div>
                
                <div class="section">
                    <h2>Checkpoint Analysis</h2>
                    {self._html_checkpoint_analysis(analytics_data.get('checkpoint_analysis', {}))}
                </div>
                
                <div class="section">
                    <h2>Storage Analytics</h2>
                    {self._html_storage_analytics(analytics_data.get('storage_analytics', {}))}
                </div>
                
                <div class="section">
                    <h2>Performance Trends</h2>
                    {self._html_performance_trends(analytics_data.get('performance_trends', {}))}
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    {self._html_recommendations(analytics_data.get('recommendations', []))}
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <div class="plot">
                        <img src="../plots/training_progress.png" alt="Training Progress" style="max-width: 100%;">
                    </div>
                    <div class="plot">
                        <img src="../plots/checkpoint_analytics.png" alt="Checkpoint Analytics" style="max-width: 100%;">
                    </div>
                    <div class="plot">
                        <img src="../plots/storage_usage.png" alt="Storage Usage" style="max-width: 100%;">
                    </div>
                    <div class="plot">
                        <img src="../plots/performance_trends.png" alt="Performance Trends" style="max-width: 100%;">
                    </div>
                    <div class="plot">
                        <img src="../plots/cloud_distribution.png" alt="Cloud Distribution" style="max-width: 100%;">
                    </div>
                </div>
            </body>
            </html>
            """
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
    
    def _html_summary_stats(self, summary: Dict[str, Any]) -> str:
        """Generate HTML for summary statistics"""
        if not summary:
            return "<p>No summary data available</p>"
        
        html = "<div>"
        for key, value in summary.items():
            if isinstance(value, dict):
                html += f"<h4>{key.replace('_', ' ').title()}</h4>"
                html += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for sub_key, sub_value in value.items():
                    html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td><td>{sub_value}</td></tr>"
                html += "</table>"
            else:
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        html += "</div>"
        
        return html
    
    def _html_training_progress(self, progress: Dict[str, Any]) -> str:
        """Generate HTML for training progress"""
        if not progress:
            return "<p>No training progress data available</p>"
        
        html = "<div>"
        for key, value in progress.items():
            if isinstance(value, dict):
                html += f"<h4>{key.replace('_', ' ').title()}</h4>"
                html += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for sub_key, sub_value in value.items():
                    html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td><td>{sub_value}</td></tr>"
                html += "</table>"
            else:
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        html += "</div>"
        
        return html
    
    def _html_checkpoint_analysis(self, analysis: Dict[str, Any]) -> str:
        """Generate HTML for checkpoint analysis"""
        if not analysis:
            return "<p>No checkpoint analysis data available</p>"
        
        html = "<div>"
        for key, value in analysis.items():
            if isinstance(value, dict):
                html += f"<h4>{key.replace('_', ' ').title()}</h4>"
                html += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for sub_key, sub_value in value.items():
                    html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td><td>{sub_value}</td></tr>"
                html += "</table>"
            else:
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        html += "</div>"
        
        return html
    
    def _html_storage_analytics(self, analytics: Dict[str, Any]) -> str:
        """Generate HTML for storage analytics"""
        if not analytics:
            return "<p>No storage analytics data available</p>"
        
        html = "<div>"
        for key, value in analytics.items():
            if isinstance(value, dict):
                html += f"<h4>{key.replace('_', ' ').title()}</h4>"
                html += "<table><tr><th>Metric</th><th>Value</th></tr>"
                for sub_key, sub_value in value.items():
                    html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td><td>{sub_value}</td></tr>"
                html += "</table>"
            else:
                html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
        html += "</div>"
        
        return html
    
    def _html_performance_trends(self, trends: Dict[str, Any]) -> str:
        """Generate HTML for performance trends"""
        if not trends:
            return "<p>No performance trend data available</p>"
        
        html = "<div>"
        for key, value in trends.items():
            html += f"<h4>{key.replace('_', ' ').title()}</h4>"
            html += "<table><tr><th>Metric</th><th>Value</th></tr>"
            for sub_key, sub_value in value.items():
                html += f"<tr><td>{sub_key.replace('_', ' ').title()}</td><td>{sub_value}</td></tr>"
            html += "</table>"
        html += "</div>"
        
        return html
    
    def _html_recommendations(self, recommendations: List[Dict[str, Any]]) -> str:
        """Generate HTML for recommendations"""
        if not recommendations:
            return "<p>No recommendations available</p>"
        
        html = "<div>"
        for rec in recommendations:
            css_class = rec.get('type', 'info')
            html += f'<div class="recommendation {css_class}">'
            html += f"<strong>{rec.get('category', 'General').upper()}:</strong> {rec.get('message', '')}<br>"
            if 'action' in rec:
                html += f"<em>Action: {rec['action']}</em>"
            html += "</div>"
        html += "</div>"
        
        return html
    
    def _generate_pdf_report(self, analytics_data: Dict[str, Any], output_file: Path) -> None:
        """Generate PDF analytics report"""
        try:
            # For now, create a simplified report
            # In a full implementation, you'd use a library like ReportLab or WeasyPrint
            logger.info("PDF report generation not fully implemented")
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
    
    def _generate_json_report(self, analytics_data: Dict[str, Any], output_file: Path) -> None:
        """Generate JSON analytics report"""
        try:
            with open(output_file, 'w') as f:
                json.dump(analytics_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")