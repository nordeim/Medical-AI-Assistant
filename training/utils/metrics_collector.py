"""
Metrics Collection and Storage System

This module provides real-time metrics collection, historical data storage,
data visualization utilities, and alert/notification systems for performance monitoring.
"""

import asyncio
import json
import sqlite3
import threading
import time
import zlib
from collections import deque, defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import queue
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, render_template_string, request, send_file
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

from .performance_monitor import (
    TrainingMetrics, SystemMetrics, ModelPerformanceMetrics,
    PerformanceMonitor, SystemMonitor
)

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """SQLite database for storing and querying metrics."""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
    def _init_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Training metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    epoch INTEGER,
                    step INTEGER,
                    global_step INTEGER,
                    phase TEXT,
                    loss REAL,
                    accuracy REAL,
                    learning_rate REAL,
                    momentum REAL,
                    grad_norm REAL,
                    grad_norm_clipped REAL,
                    param_norm REAL,
                    batch_time REAL,
                    data_load_time REAL,
                    forward_pass_time REAL,
                    backward_pass_time REAL,
                    optimization_time REAL,
                    cpu_memory_mb REAL,
                    gpu_memory_mb REAL,
                    gpu_memory_utilization REAL,
                    custom_metrics TEXT,  -- JSON string
                    data_compressed INTEGER DEFAULT 0,
                    UNIQUE(timestamp, global_step)
                )
            ''')
            
            # System metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    cpu_usage_percent REAL,
                    cpu_temperature REAL,
                    cpu_freq REAL,
                    memory_usage_percent REAL,
                    memory_available_gb REAL,
                    swap_usage_percent REAL,
                    gpu_utilization REAL,
                    gpu_memory_utilization REAL,
                    gpu_temperature REAL,
                    gpu_power_watts REAL,
                    gpu_memory_mb REAL,
                    disk_read_mb_s REAL,
                    disk_write_mb_s REAL,
                    disk_usage_percent REAL,
                    network_sent_mb_s REAL,
                    network_recv_mb_s REAL,
                    process_cpu_percent REAL,
                    process_memory_mb REAL,
                    data_compressed INTEGER DEFAULT 0
                )
            ''')
            
            # Model performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    avg_latency_ms REAL,
                    median_latency_ms REAL,
                    p95_latency_ms REAL,
                    p99_latency_ms REAL,
                    min_latency_ms REAL,
                    max_latency_ms REAL,
                    latency_std_ms REAL,
                    samples_per_second REAL,
                    tokens_per_second REAL,
                    model_memory_mb REAL,
                    peak_memory_mb REAL,
                    model_size_mb REAL,
                    quantization_ratio REAL,
                    compression_ratio REAL,
                    inference_quality_score REAL,
                    consistency_score REAL,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    data_compressed INTEGER DEFAULT 0
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,  -- JSON string
                    acknowledged INTEGER DEFAULT 0
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_timestamp ON training_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_global_step ON training_metrics(global_step)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_timestamp ON model_performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)')
            
            conn.commit()
            
    def insert_training_metrics(self, metrics: TrainingMetrics) -> int:
        """Insert training metrics into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if record already exists
            cursor.execute(
                'SELECT id FROM training_metrics WHERE timestamp = ? AND global_step = ?',
                (metrics.timestamp, metrics.global_step)
            )
            if cursor.fetchone():
                return cursor.lastrowid  # Already exists
            
            custom_metrics_json = json.dumps(metrics.custom_metrics) if metrics.custom_metrics else None
            
            cursor.execute('''
                INSERT INTO training_metrics (
                    timestamp, epoch, step, global_step, phase, loss, accuracy,
                    learning_rate, momentum, grad_norm, grad_norm_clipped, param_norm,
                    batch_time, data_load_time, forward_pass_time, backward_pass_time,
                    optimization_time, cpu_memory_mb, gpu_memory_mb, gpu_memory_utilization,
                    custom_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.epoch, metrics.step, metrics.global_step,
                metrics.phase, metrics.loss, metrics.accuracy, metrics.learning_rate,
                metrics.momentum, metrics.grad_norm, metrics.grad_norm_clipped, metrics.param_norm,
                metrics.batch_time, metrics.data_load_time, metrics.forward_pass_time,
                metrics.backward_pass_time, metrics.optimization_time, metrics.cpu_memory_mb,
                metrics.gpu_memory_mb, metrics.gpu_memory_utilization, custom_metrics_json
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def insert_system_metrics(self, metrics: SystemMetrics) -> int:
        """Insert system metrics into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_metrics (
                    timestamp, cpu_usage_percent, cpu_temperature, cpu_freq,
                    memory_usage_percent, memory_available_gb, swap_usage_percent,
                    gpu_utilization, gpu_memory_utilization, gpu_temperature,
                    gpu_power_watts, gpu_memory_mb, disk_read_mb_s, disk_write_mb_s,
                    disk_usage_percent, network_sent_mb_s, network_recv_mb_s,
                    process_cpu_percent, process_memory_mb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.cpu_usage_percent, metrics.cpu_temperature,
                metrics.cpu_freq, metrics.memory_usage_percent, metrics.memory_available_gb,
                metrics.swap_usage_percent, metrics.gpu_utilization, metrics.gpu_memory_utilization,
                metrics.gpu_temperature, metrics.gpu_power_watts, metrics.gpu_memory_mb,
                metrics.disk_read_mb_s, metrics.disk_write_mb_s, metrics.disk_usage_percent,
                metrics.network_sent_mb_s, metrics.network_recv_mb_s, metrics.process_cpu_percent,
                metrics.process_memory_mb
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def insert_model_performance_metrics(self, metrics: ModelPerformanceMetrics) -> int:
        """Insert model performance metrics into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance_metrics (
                    timestamp, avg_latency_ms, median_latency_ms, p95_latency_ms,
                    p99_latency_ms, min_latency_ms, max_latency_ms, latency_std_ms,
                    samples_per_second, tokens_per_second, model_memory_mb,
                    peak_memory_mb, model_size_mb, quantization_ratio, compression_ratio,
                    inference_quality_score, consistency_score, batch_size, sequence_length
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.timestamp, metrics.avg_latency_ms, metrics.median_latency_ms,
                metrics.p95_latency_ms, metrics.p99_latency_ms, metrics.min_latency_ms,
                metrics.max_latency_ms, metrics.latency_std_ms, metrics.samples_per_second,
                metrics.tokens_per_second, metrics.model_memory_mb, metrics.peak_memory_mb,
                metrics.model_size_mb, metrics.quantization_ratio, metrics.compression_ratio,
                metrics.inference_quality_score, metrics.consistency_score, metrics.batch_size,
                metrics.sequence_length
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def insert_alert(self, alert_type: str, severity: str, message: str, metadata: Optional[Dict] = None) -> int:
        """Insert alert into database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, message, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (time.time(), alert_type, severity, message, metadata_json))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_training_metrics(self, 
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           global_step_range: Optional[Tuple[int, int]] = None,
                           limit: Optional[int] = 1000) -> List[TrainingMetrics]:
        """Get training metrics with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM training_metrics WHERE 1=1'
            params = []
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            if global_step_range:
                query += ' AND global_step BETWEEN ? AND ?'
                params.extend(global_step_range)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                custom_metrics = json.loads(row['custom_metrics']) if row['custom_metrics'] else {}
                
                metrics.append(TrainingMetrics(
                    timestamp=row['timestamp'],
                    epoch=row['epoch'],
                    step=row['step'],
                    global_step=row['global_step'],
                    phase=row['phase'],
                    loss=row['loss'],
                    accuracy=row['accuracy'],
                    learning_rate=row['learning_rate'],
                    momentum=row['momentum'],
                    grad_norm=row['grad_norm'],
                    grad_norm_clipped=row['grad_norm_clipped'],
                    param_norm=row['param_norm'],
                    batch_time=row['batch_time'],
                    data_load_time=row['data_load_time'],
                    forward_pass_time=row['forward_pass_time'],
                    backward_pass_time=row['backward_pass_time'],
                    optimization_time=row['optimization_time'],
                    cpu_memory_mb=row['cpu_memory_mb'],
                    gpu_memory_mb=row['gpu_memory_mb'],
                    gpu_memory_utilization=row['gpu_memory_utilization'],
                    custom_metrics=custom_metrics
                ))
            
            return metrics
    
    def get_system_metrics(self,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         limit: Optional[int] = 1000) -> List[SystemMetrics]:
        """Get system metrics with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM system_metrics WHERE 1=1'
            params = []
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                metrics.append(SystemMetrics(
                    timestamp=row['timestamp'],
                    cpu_usage_percent=row['cpu_usage_percent'],
                    cpu_temperature=row['cpu_temperature'],
                    cpu_freq=row['cpu_freq'],
                    memory_usage_percent=row['memory_usage_percent'],
                    memory_available_gb=row['memory_available_gb'],
                    swap_usage_percent=row['swap_usage_percent'],
                    gpu_utilization=row['gpu_utilization'],
                    gpu_memory_utilization=row['gpu_memory_utilization'],
                    gpu_temperature=row['gpu_temperature'],
                    gpu_power_watts=row['gpu_power_watts'],
                    gpu_memory_mb=row['gpu_memory_mb'],
                    disk_read_mb_s=row['disk_read_mb_s'],
                    disk_write_mb_s=row['disk_write_mb_s'],
                    disk_usage_percent=row['disk_usage_percent'],
                    network_sent_mb_s=row['network_sent_mb_s'],
                    network_recv_mb_s=row['network_recv_mb_s'],
                    process_cpu_percent=row['process_cpu_percent'],
                    process_memory_mb=row['process_memory_mb']
                ))
            
            return metrics
    
    def get_alerts(self,
                  severity: Optional[str] = None,
                  acknowledged: Optional[bool] = None,
                  start_time: Optional[float] = None,
                  limit: Optional[int] = 100) -> List[Dict]:
        """Get alerts with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = 'SELECT * FROM alerts WHERE 1=1'
            params = []
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity)
            
            if acknowledged is not None:
                query += ' AND acknowledged = ?'
                params.append(1 if acknowledged else 0)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            alerts = []
            for row in rows:
                metadata = json.loads(row['metadata']) if row['metadata'] else None
                
                alerts.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'alert_type': row['alert_type'],
                    'severity': row['severity'],
                    'message': row['message'],
                    'metadata': metadata,
                    'acknowledged': bool(row['acknowledged'])
                })
            
            return alerts
    
    def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('UPDATE alerts SET acknowledged = 1 WHERE id = ?', (alert_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Count records in each table
            for table in ['training_metrics', 'system_metrics', 'model_performance_metrics', 'alerts']:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                stats[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get date ranges
            for table in ['training_metrics', 'system_metrics', 'model_performance_metrics', 'alerts']:
                cursor.execute(f'SELECT MIN(timestamp), MAX(timestamp) FROM {table}')
                min_time, max_time = cursor.fetchone()
                if min_time and max_time:
                    stats[f'{table}_date_range'] = {
                        'start': min_time,
                        'end': max_time,
                        'duration_hours': (max_time - min_time) / 3600
                    }
            
            return stats


class RealTimeMetricsCollector:
    """Real-time metrics collection and buffering system."""
    
    def __init__(self, 
                 db_path: Union[str, Path],
                 buffer_size: int = 1000,
                 flush_interval: float = 10.0,
                 compression_enabled: bool = True):
        
        self.db = MetricsDatabase(db_path)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.compression_enabled = compression_enabled
        
        # Buffers for different metric types
        self.training_buffer: deque = deque(maxlen=buffer_size)
        self.system_buffer: deque = deque(maxlen=buffer_size)
        self.model_performance_buffer: deque = deque(maxlen=buffer_size)
        
        # Real-time data for WebSocket streaming
        self.realtime_data = {
            'training_metrics': deque(maxlen=100),
            'system_metrics': deque(maxlen=100),
            'alerts': deque(maxlen=50)
        }
        
        # Callbacks for real-time processing
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Threading
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'collected_training': 0,
            'collected_system': 0,
            'collected_model_performance': 0,
            'flushed_batches': 0,
            'compression_ratio': 0.0
        }
        
        logger.info(f"RealTimeMetricsCollector initialized with buffer size: {buffer_size}")
    
    def add_callback(self, metric_type: str, callback: Callable):
        """Add callback for real-time metric processing."""
        self.callbacks[metric_type].append(callback)
    
    def collect_training_metrics(self, metrics: TrainingMetrics):
        """Collect training metrics in real-time."""
        with self._lock:
            self.training_buffer.append(metrics)
            self.realtime_data['training_metrics'].append(metrics)
            self.stats['collected_training'] += 1
            
            # Execute callbacks
            for callback in self.callbacks.get('training', []):
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in training metrics callback: {e}")
    
    def collect_system_metrics(self, metrics: SystemMetrics):
        """Collect system metrics in real-time."""
        with self._lock:
            self.system_buffer.append(metrics)
            self.realtime_data['system_metrics'].append(metrics)
            self.stats['collected_system'] += 1
            
            # Execute callbacks
            for callback in self.callbacks.get('system', []):
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in system metrics callback: {e}")
    
    def collect_model_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """Collect model performance metrics in real-time."""
        with self._lock:
            self.model_performance_buffer.append(metrics)
            self.stats['collected_model_performance'] += 1
            
            # Execute callbacks
            for callback in self.callbacks.get('model_performance', []):
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in model performance callback: {e}")
    
    def add_alert(self, alert_type: str, severity: str, message: str, metadata: Optional[Dict] = None):
        """Add alert to the system."""
        alert_id = self.db.insert_alert(alert_type, severity, message, metadata)
        alert = {
            'id': alert_id,
            'timestamp': time.time(),
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'metadata': metadata or {},
            'acknowledged': False
        }
        
        self.realtime_data['alerts'].append(alert)
        
        # Execute alert callbacks
        for callback in self.callbacks.get('alert', []):
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Alert [{severity.upper()}] {alert_type}: {message}")
    
    def start_collection(self):
        """Start real-time metrics collection."""
        if self._flush_thread and self._flush_thread.is_alive():
            logger.warning("Collection already running")
            return
        
        self._stop_event.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        logger.info("Real-time metrics collection started")
    
    def stop_collection(self):
        """Stop real-time metrics collection."""
        if self._flush_thread:
            self._stop_event.set()
            self._flush_thread.join(timeout=30)
            logger.info("Real-time metrics collection stopped")
        
        # Flush remaining data
        self._flush_all()
    
    def _flush_loop(self):
        """Background thread for flushing buffers."""
        while not self._stop_event.wait(self.flush_interval):
            try:
                self._flush_all()
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
    
    def _flush_all(self):
        """Flush all buffers to database."""
        with self._lock:
            # Flush training metrics
            if self.training_buffer:
                for metrics in self.training_buffer:
                    self.db.insert_training_metrics(metrics)
                self.training_buffer.clear()
            
            # Flush system metrics
            if self.system_buffer:
                for metrics in self.system_buffer:
                    self.db.insert_system_metrics(metrics)
                self.system_buffer.clear()
            
            # Flush model performance metrics
            if self.model_performance_buffer:
                for metrics in self.model_performance_buffer:
                    self.db.insert_model_performance_metrics(metrics)
                self.model_performance_buffer.clear()
            
            self.stats['flushed_batches'] += 1
    
    def get_realtime_data(self, metric_type: str, limit: int = 50) -> List[Any]:
        """Get recent real-time data."""
        if metric_type not in self.realtime_data:
            return []
        
        data = list(self.realtime_data[metric_type])
        return data[-limit:] if len(data) > limit else data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        with self._lock:
            return self.stats.copy()
    
    def export_data(self, 
                   output_path: Union[str, Path],
                   metric_type: str = "all",
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   format: str = "json") -> bool:
        """Export metrics data to file."""
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if metric_type == "training" or metric_type == "all":
                training_data = self.db.get_training_metrics(start_time, end_time)
                if format == "json":
                    with open(output_path / "training_metrics.json", 'w') as f:
                        json.dump([asdict(m) for m in training_data], f, indent=2)
                elif format == "csv":
                    df = pd.DataFrame([asdict(m) for m in training_data])
                    df.to_csv(output_path / "training_metrics.csv", index=False)
            
            if metric_type == "system" or metric_type == "all":
                system_data = self.db.get_system_metrics(start_time, end_time)
                if format == "json":
                    with open(output_path / "system_metrics.json", 'w') as f:
                        json.dump([asdict(m) for m in system_data], f, indent=2)
                elif format == "csv":
                    df = pd.DataFrame([asdict(m) for m in system_data])
                    df.to_csv(output_path / "system_metrics.csv", index=False)
            
            if metric_type == "alerts" or metric_type == "all":
                alerts_data = self.db.get_alerts()
                if format == "json":
                    with open(output_path / "alerts.json", 'w') as f:
                        json.dump(alerts_data, f, indent=2)
            
            logger.info(f"Data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False


class DataVisualizer:
    """Create visualizations for metrics data."""
    
    def __init__(self, output_dir: Union[str, Path] = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def create_training_progress_plots(self, training_metrics: List[TrainingMetrics]) -> Dict[str, Path]:
        """Create training progress plots."""
        if not training_metrics:
            return {}
        
        # Convert to DataFrame
        data = [asdict(m) for m in training_metrics]
        df = pd.DataFrame(data)
        
        plots = {}
        
        # Loss curve
        if 'loss' in df.columns and df['loss'].notna().any():
            plt.figure(figsize=(12, 6))
            plt.plot(df['global_step'], df['loss'], linewidth=2, alpha=0.8)
            plt.title('Training Loss Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, alpha=0.3)
            loss_plot = self.output_dir / "training_loss.png"
            plt.savefig(loss_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['loss'] = loss_plot
        
        # Learning rate schedule
        if 'learning_rate' in df.columns and df['learning_rate'].notna().any():
            plt.figure(figsize=(12, 6))
            plt.plot(df['global_step'], df['learning_rate'], linewidth=2, color='green')
            plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            lr_plot = self.output_dir / "learning_rate.png"
            plt.savefig(lr_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['learning_rate'] = lr_plot
        
        # Gradient norms
        if 'grad_norm' in df.columns and df['grad_norm'].notna().any():
            plt.figure(figsize=(12, 6))
            plt.plot(df['global_step'], df['grad_norm'], linewidth=2, alpha=0.8, label='Gradient Norm')
            if 'grad_norm_clipped' in df.columns and df['grad_norm_clipped'].notna().any():
                plt.plot(df['global_step'], df['grad_norm_clipped'], linewidth=2, alpha=0.8, label='Clipped Gradient Norm')
            plt.title('Gradient Norms Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Gradient Norm', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            grad_plot = self.output_dir / "gradient_norms.png"
            plt.savefig(grad_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['gradients'] = grad_plot
        
        # Memory usage
        memory_cols = [col for col in df.columns if 'memory' in col]
        if memory_cols:
            plt.figure(figsize=(12, 8))
            for col in memory_cols:
                if df[col].notna().any():
                    plt.plot(df['global_step'], df[col], linewidth=2, label=col.replace('_', ' ').title())
            plt.title('Memory Usage Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Global Step', fontsize=12)
            plt.ylabel('Memory (MB)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            memory_plot = self.output_dir / "memory_usage.png"
            plt.savefig(memory_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['memory'] = memory_plot
        
        return plots
    
    def create_system_monitoring_plots(self, system_metrics: List[SystemMetrics]) -> Dict[str, Path]:
        """Create system monitoring plots."""
        if not system_metrics:
            return {}
        
        data = [asdict(m) for m in system_metrics]
        df = pd.DataFrame(data)
        
        plots = {}
        
        # CPU and Memory usage
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        if 'cpu_usage_percent' in df.columns:
            plt.plot(df.index, df['cpu_usage_percent'], linewidth=2)
            plt.title('CPU Usage', fontsize=12, fontweight='bold')
            plt.ylabel('CPU Usage (%)', fontsize=10)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        if 'memory_usage_percent' in df.columns:
            plt.plot(df.index, df['memory_usage_percent'], linewidth=2, color='orange')
            plt.title('Memory Usage', fontsize=12, fontweight='bold')
            plt.ylabel('Memory Usage (%)', fontsize=10)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        if 'gpu_utilization' in df.columns:
            plt.plot(df.index, df['gpu_utilization'], linewidth=2, color='green')
            plt.title('GPU Utilization', fontsize=12, fontweight='bold')
            plt.ylabel('GPU Utilization (%)', fontsize=10)
            plt.xlabel('Sample Index', fontsize=10)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        if 'gpu_memory_utilization' in df.columns:
            plt.plot(df.index, df['gpu_memory_utilization'], linewidth=2, color='red')
            plt.title('GPU Memory Usage', fontsize=12, fontweight='bold')
            plt.ylabel('GPU Memory (%)', fontsize=10)
            plt.xlabel('Sample Index', fontsize=10)
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        system_plot = self.output_dir / "system_monitoring.png"
        plt.savefig(system_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['system'] = system_plot
        
        # Temperature monitoring
        temp_cols = [col for col in df.columns if 'temperature' in col]
        if temp_cols:
            plt.figure(figsize=(12, 6))
            for col in temp_cols:
                if df[col].notna().any():
                    plt.plot(df.index, df[col], linewidth=2, label=col.replace('_', ' ').title())
            plt.title('Temperature Monitoring', fontsize=16, fontweight='bold')
            plt.xlabel('Sample Index', fontsize=12)
            plt.ylabel('Temperature (°C)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            temp_plot = self.output_dir / "temperature_monitoring.png"
            plt.savefig(temp_plot, dpi=300, bbox_inches='tight')
            plt.close()
            plots['temperature'] = temp_plot
        
        return plots
    
    def create_interactive_dashboard(self, 
                                   training_metrics: List[TrainingMetrics],
                                   system_metrics: List[SystemMetrics],
                                   output_file: str = "interactive_dashboard.html") -> Path:
        """Create interactive Plotly dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Training Loss', 'Learning Rate', 'System CPU', 
                           'Memory Usage', 'GPU Metrics', 'Training Timeline'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training metrics
        if training_metrics:
            train_data = [asdict(m) for m in training_metrics]
            train_df = pd.DataFrame(train_data)
            
            # Loss
            if 'loss' in train_df.columns:
                fig.add_trace(
                    go.Scatter(x=train_df['global_step'], y=train_df['loss'],
                              mode='lines', name='Loss', line=dict(color='blue')),
                    row=1, col=1
                )
            
            # Learning rate
            if 'learning_rate' in train_df.columns:
                fig.add_trace(
                    go.Scatter(x=train_df['global_step'], y=train_df['learning_rate'],
                              mode='lines', name='Learning Rate', line=dict(color='green')),
                    row=1, col=2
                )
            
            # Gradient norms
            if 'grad_norm' in train_df.columns:
                fig.add_trace(
                    go.Scatter(x=train_df['global_step'], y=train_df['grad_norm'],
                              mode='lines', name='Gradient Norm', line=dict(color='orange')),
                    row=3, col=2
                )
        
        # System metrics
        if system_metrics:
            sys_data = [asdict(m) for m in system_metrics]
            sys_df = pd.DataFrame(sys_data)
            
            # CPU
            if 'cpu_usage_percent' in sys_df.columns:
                fig.add_trace(
                    go.Scatter(x=sys_df.index, y=sys_df['cpu_usage_percent'],
                              mode='lines', name='CPU Usage (%)', line=dict(color='red')),
                    row=2, col=1
                )
            
            # Memory
            if 'memory_usage_percent' in sys_df.columns:
                fig.add_trace(
                    go.Scatter(x=sys_df.index, y=sys_df['memory_usage_percent'],
                              mode='lines', name='Memory Usage (%)', line=dict(color='purple')),
                    row=2, col=2
                )
            
            # GPU metrics
            if 'gpu_utilization' in sys_df.columns and 'gpu_memory_utilization' in sys_df.columns:
                fig.add_trace(
                    go.Scatter(x=sys_df.index, y=sys_df['gpu_utilization'],
                              mode='lines', name='GPU Utilization (%)', line=dict(color='cyan')),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=sys_df.index, y=sys_df['gpu_memory_utilization'],
                              mode='lines', name='GPU Memory (%)', line=dict(color='pink')),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="ML Training Performance Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save dashboard
        output_path = self.output_dir / output_file
        pyo.plot(fig, filename=str(output_path), auto_open=False)
        
        return output_path
    
    def create_performance_comparison_plots(self, 
                                          baseline_metrics: List[TrainingMetrics],
                                          optimized_metrics: List[TrainingMetrics],
                                          labels: Tuple[str, str] = ("Baseline", "Optimized")) -> Dict[str, Path]:
        """Create performance comparison plots."""
        
        plots = {}
        
        # Loss comparison
        plt.figure(figsize=(15, 5))
        
        baseline_data = [asdict(m) for m in baseline_metrics]
        optimized_data = [asdict(m) for m in optimized_metrics]
        
        if baseline_data:
            baseline_df = pd.DataFrame(baseline_data)
            plt.subplot(1, 2, 1)
            plt.plot(baseline_df['global_step'], baseline_df['loss'], 
                    linewidth=2, label=labels[0], alpha=0.8)
            if optimized_data:
                optimized_df = pd.DataFrame(optimized_data)
                plt.plot(optimized_df['global_step'], optimized_df['loss'],
                        linewidth=2, label=labels[1], alpha=0.8)
            plt.title('Training Loss Comparison', fontsize=12, fontweight='bold')
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if optimized_data and baseline_data:
            plt.subplot(1, 2, 2)
            # Calculate convergence rate
            baseline_losses = np.array([m.loss for m in baseline_metrics[-50:]])
            optimized_losses = np.array([m.loss for m in optimized_metrics[-50:]])
            
            convergence_baseline = np.mean(baseline_losses)
            convergence_optimized = np.mean(optimized_losses)
            
            plt.bar([labels[0], labels[1]], [convergence_baseline, convergence_optimized],
                   color=['blue', 'green'], alpha=0.7)
            plt.title('Final Loss Comparison', fontsize=12, fontweight='bold')
            plt.ylabel('Final Loss')
            
            improvement = (convergence_baseline - convergence_optimized) / convergence_baseline * 100
            plt.text(0.5, max(convergence_baseline, convergence_optimized) * 0.8,
                    f'Improvement: {improvement:.1f}%', ha='center', fontsize=10, fontweight='bold')
        
        comparison_plot = self.output_dir / "performance_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        plt.close()
        plots['comparison'] = comparison_plot
        
        return plots


class AlertSystem:
    """Alert and notification system for performance monitoring."""
    
    def __init__(self, metrics_collector: RealTimeMetricsCollector):
        self.collector = metrics_collector
        self.alert_rules = []
        self.notification_channels = []
        
        # Default alert rules
        self._setup_default_rules()
        
        # Statistics
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'false_positives': 0
        }
        
        logger.info("AlertSystem initialized")
    
    def _setup_default_rules(self):
        """Setup default alert rules."""
        
        # High CPU usage
        self.add_alert_rule(
            name="High CPU Usage",
            condition=lambda metrics: metrics.cpu_usage_percent > 90,
            severity="warning",
            message="High CPU usage detected: {cpu_usage:.1f}%",
            cooldown_seconds=300
        )
        
        # High memory usage
        self.add_alert_rule(
            name="High Memory Usage",
            condition=lambda metrics: metrics.memory_usage_percent > 85,
            severity="warning",
            message="High memory usage detected: {memory_usage:.1f}%",
            cooldown_seconds=300
        )
        
        # Gradient explosion
        self.add_alert_rule(
            name="Gradient Explosion",
            condition=lambda metrics: hasattr(metrics, 'grad_norm') and metrics.grad_norm > 10.0,
            severity="critical",
            message="Gradient explosion detected: {grad_norm:.2f}",
            cooldown_seconds=60
        )
        
        # Training loss spike
        self.add_alert_rule(
            name="Training Loss Spike",
            condition=lambda metrics: hasattr(metrics, 'loss') and metrics.loss > 100.0,
            severity="critical",
            message="Training loss spike detected: {loss:.2f}",
            cooldown_seconds=180
        )
        
        # GPU temperature
        self.add_alert_rule(
            name="High GPU Temperature",
            condition=lambda metrics: hasattr(metrics, 'gpu_temperature') and metrics.gpu_temperature > 80,
            severity="warning",
            message="High GPU temperature detected: {gpu_temperature:.1f}°C",
            cooldown_seconds=300
        )
    
    def add_alert_rule(self, 
                      name: str,
                      condition: Callable,
                      severity: str,
                      message: str,
                      cooldown_seconds: int = 300,
                      metadata: Optional[Dict] = None):
        """Add custom alert rule."""
        rule = {
            'name': name,
            'condition': condition,
            'severity': severity,
            'message': message,
            'cooldown_seconds': cooldown_seconds,
            'last_triggered': 0,
            'metadata': metadata or {}
        }
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, Slack, webhook, etc.)."""
        channel = {
            'type': channel_type,
            'config': config,
            'enabled': True
        }
        self.notification_channels.append(channel)
        logger.info(f"Added notification channel: {channel_type}")
    
    def evaluate_rules(self, training_metrics: Optional[TrainingMetrics] = None, 
                      system_metrics: Optional[SystemMetrics] = None):
        """Evaluate alert rules against current metrics."""
        
        current_time = time.time()
        
        for rule in self.alert_rules:
            try:
                # Check if in cooldown period
                if current_time - rule['last_triggered'] < rule['cooldown_seconds']:
                    continue
                
                # Evaluate condition
                should_alert = False
                evaluated_metrics = None
                
                if training_metrics and hasattr(rule['condition'], '__call__'):
                    evaluated_metrics = training_metrics
                    should_alert = rule['condition'](training_metrics)
                elif system_metrics and hasattr(rule['condition'], '__call__'):
                    evaluated_metrics = system_metrics
                    should_alert = rule['condition'](system_metrics)
                
                if should_alert:
                    # Format message
                    message = rule['message']
                    if evaluated_metrics:
                        # Extract values from metrics
                        metric_dict = asdict(evaluated_metrics)
                        try:
                            message = message.format(**metric_dict)
                        except (KeyError, ValueError):
                            pass  # Keep original message if formatting fails
                    
                    # Add alert
                    self.collector.add_alert(
                        alert_type=rule['name'],
                        severity=rule['severity'],
                        message=message,
                        metadata=rule['metadata']
                    )
                    
                    # Send notifications
                    self._send_notifications(rule, message)
                    
                    # Update statistics
                    self.alert_stats['total_alerts'] += 1
                    self.alert_stats['alerts_by_severity'][rule['severity']] += 1
                    self.alert_stats['alerts_by_type'][rule['name']] += 1
                    
                    # Update cooldown
                    rule['last_triggered'] = current_time
                    
                    logger.info(f"Alert triggered: {rule['name']}")
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
    
    def _send_notifications(self, rule: Dict, message: str):
        """Send notifications through configured channels."""
        
        for channel in self.notification_channels:
            if not channel['enabled']:
                continue
            
            try:
                if channel['type'] == 'webhook':
                    self._send_webhook_notification(channel, rule, message)
                elif channel['type'] == 'slack':
                    self._send_slack_notification(channel, rule, message)
                elif channel['type'] == 'email':
                    self._send_email_notification(channel, rule, message)
                    
            except Exception as e:
                logger.error(f"Failed to send notification via {channel['type']}: {e}")
    
    def _send_webhook_notification(self, channel: Dict, rule: Dict, message: str):
        """Send webhook notification."""
        import requests
        
        webhook_url = channel['config'].get('url')
        if not webhook_url:
            return
        
        payload = {
            'alert_name': rule['name'],
            'severity': rule['severity'],
            'message': message,
            'timestamp': time.time(),
            'metadata': rule['metadata']
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_slack_notification(self, channel: Dict, rule: Dict, message: str):
        """Send Slack notification."""
        import requests
        
        webhook_url = channel['config'].get('webhook_url')
        if not webhook_url:
            return
        
        color = {
            'critical': 'danger',
            'warning': 'warning',
            'info': 'good'
        }.get(rule['severity'], 'warning')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f"ML Training Alert: {rule['name']}",
                'text': message,
                'fields': [
                    {'title': 'Severity', 'value': rule['severity'], 'short': True},
                    {'title': 'Timestamp', 'value': datetime.fromtimestamp(time.time()).isoformat(), 'short': True}
                ]
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    
    def _send_email_notification(self, channel: Dict, rule: Dict, message: str):
        """Send email notification (placeholder)."""
        # This would require email configuration
        # For now, just log the message
        logger.info(f"Email notification: {rule['name']} - {message}")
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics."""
        return dict(self.alert_stats)


# Web Dashboard for Real-time Monitoring
class MonitoringDashboard:
    """Web-based dashboard for real-time monitoring."""
    
    def __init__(self, 
                 metrics_collector: RealTimeMetricsCollector,
                 host: str = "127.0.0.1",
                 port: int = 8080):
        
        self.collector = metrics_collector
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        self.host = host
        self.port = port
        
        self._setup_routes()
        self._setup_socketio()
        
        logger.info(f"Monitoring dashboard initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/training_metrics')
        def get_training_metrics():
            limit = request.args.get('limit', 100, type=int)
            data = self.collector.get_realtime_data('training_metrics', limit)
            return jsonify([asdict(m) for m in data])
        
        @self.app.route('/api/system_metrics')
        def get_system_metrics():
            limit = request.args.get('limit', 100, type=int)
            data = self.collector.get_realtime_data('system_metrics', limit)
            return jsonify([asdict(m) for m in data])
        
        @self.app.route('/api/alerts')
        def get_alerts():
            data = self.collector.get_realtime_data('alerts', 50)
            return jsonify(data)
        
        @self.app.route('/api/statistics')
        def get_statistics():
            db_stats = self.collector.db.get_database_stats()
            collection_stats = self.collector.get_statistics()
            return jsonify({
                'database': db_stats,
                'collection': collection_stats
            })
        
        @self.app.route('/export')
        def export_data():
            metric_type = request.args.get('type', 'all')
            format_type = request.args.get('format', 'json')
            export_path = f"/tmp/metrics_export_{int(time.time())}"
            
            success = self.collector.export_data(export_path, metric_type, format=format_type)
            
            if success:
                return send_file(f"{export_path}/{metric_type}_metrics.{format_type}")
            else:
                return jsonify({'error': 'Export failed'}), 500
    
    def _setup_socketio(self):
        """Setup SocketIO for real-time updates."""
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info('Dashboard client connected')
            emit('connected', {'data': 'Connected to monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info('Dashboard client disconnected')
        
        @self.socketio.on('subscribe_alerts')
        def handle_subscribe_alerts():
            # Start sending alert updates
            emit('alert_subscription', {'status': 'subscribed'})
    
    def start_dashboard(self):
        """Start the web dashboard."""
        
        def start_server():
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)
        
        # Start in background thread
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        logger.info(f"Monitoring dashboard started at http://{self.host}:{self.port}")
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for dashboard."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML Training Performance Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .header {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    text-align: center;
                }
                .grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .card {
                    background: white;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .metric {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin: 10px 0;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 5px;
                }
                .metric-value {
                    font-weight: bold;
                    font-size: 1.2em;
                }
                .alert {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 5px;
                    border-left: 4px solid;
                }
                .alert.critical { background: #fee; border-color: #dc3545; }
                .alert.warning { background: #fff3cd; border-color: #ffc107; }
                .alert.info { background: #d1ecf1; border-color: #17a2b8; }
                .status-indicator {
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    display: inline-block;
                    margin-right: 8px;
                }
                .status-online { background-color: #28a745; }
                .status-offline { background-color: #dc3545; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 ML Training Performance Dashboard</h1>
                <p>Real-time monitoring and metrics visualization</p>
            </div>
            
            <div class="grid">
                <div class="card">
                    <h3>📊 System Status</h3>
                    <div class="metric">
                        <span>CPU Usage</span>
                        <span class="metric-value" id="cpu-usage">--%</span>
                    </div>
                    <div class="metric">
                        <span>Memory Usage</span>
                        <span class="metric-value" id="memory-usage">--%</span>
                    </div>
                    <div class="metric">
                        <span>GPU Utilization</span>
                        <span class="metric-value" id="gpu-util">--%</span>
                    </div>
                    <div class="metric">
                        <span>GPU Memory</span>
                        <span class="metric-value" id="gpu-memory">--%</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🎯 Training Status</h3>
                    <div class="metric">
                        <span>Current Step</span>
                        <span class="metric-value" id="current-step">--</span>
                    </div>
                    <div class="metric">
                        <span>Current Loss</span>
                        <span class="metric-value" id="current-loss">--</span>
                    </div>
                    <div class="metric">
                        <span>Learning Rate</span>
                        <span class="metric-value" id="learning-rate">--</span>
                    </div>
                    <div class="metric">
                        <span>Gradient Norm</span>
                        <span class="metric-value" id="grad-norm">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>🚨 Recent Alerts</h3>
                    <div id="alerts-container">
                        <p>No alerts</p>
                    </div>
                </div>
                
                <div class="card">
                    <h3>📈 Performance Charts</h3>
                    <div id="training-charts"></div>
                </div>
            </div>
            
            <script>
                const socket = io();
                
                // Connection status
                socket.on('connected', function(data) {
                    console.log('Connected:', data);
                });
                
                // Update dashboard data
                function updateDashboard() {
                    fetch('/api/statistics')
                        .then(response => response.json())
                        .then(stats => {
                            console.log('Statistics:', stats);
                        });
                    
                    fetch('/api/system_metrics?limit=1')
                        .then(response => response.json())
                        .then(metrics => {
                            if (metrics.length > 0) {
                                const m = metrics[0];
                                document.getElementById('cpu-usage').textContent = m.cpu_usage_percent.toFixed(1) + '%';
                                document.getElementById('memory-usage').textContent = m.memory_usage_percent.toFixed(1) + '%';
                                document.getElementById('gpu-util').textContent = (m.gpu_utilization || 0).toFixed(1) + '%';
                                document.getElementById('gpu-memory').textContent = (m.gpu_memory_utilization || 0).toFixed(1) + '%';
                            }
                        });
                    
                    fetch('/api/training_metrics?limit=1')
                        .then(response => response.json())
                        .then(metrics => {
                            if (metrics.length > 0) {
                                const m = metrics[0];
                                document.getElementById('current-step').textContent = m.global_step || '--';
                                document.getElementById('current-loss').textContent = (m.loss || 0).toFixed(4);
                                document.getElementById('learning-rate').textContent = (m.learning_rate || 0).toExponential(2);
                                document.getElementById('grad-norm').textContent = (m.grad_norm || 0).toFixed(3);
                            }
                        });
                    
                    fetch('/api/alerts')
                        .then(response => response.json())
                        .then(alerts => {
                            const container = document.getElementById('alerts-container');
                            if (alerts.length === 0) {
                                container.innerHTML = '<p>No alerts</p>';
                            } else {
                                container.innerHTML = alerts.map(alert => 
                                    `<div class="alert ${alert.severity}">
                                        <strong>${alert.alert_type}</strong>: ${alert.message}
                                        <br><small>${new Date(alert.timestamp * 1000).toLocaleString()}</small>
                                    </div>`
                                ).join('');
                            }
                        });
                }
                
                // Update charts
                function updateCharts() {
                    fetch('/api/training_metrics?limit=50')
                        .then(response => response.json())
                        .then(metrics => {
                            if (metrics.length === 0) return;
                            
                            const steps = metrics.map(m => m.global_step).reverse();
                            const losses = metrics.map(m => m.loss).reverse();
                            const lrs = metrics.map(m => m.learning_rate).reverse();
                            
                            const trainingData = [{
                                x: steps,
                                y: losses,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Loss',
                                xaxis: 'x',
                                yaxis: 'y'
                            }, {
                                x: steps,
                                y: lrs,
                                type: 'scatter',
                                mode: 'lines',
                                name: 'Learning Rate',
                                xaxis: 'x2',
                                yaxis: 'y2'
                            }];
                            
                            const layout = {
                                grid: {rows: 2, columns: 1, pattern: 'independent'},
                                xaxis: {title: 'Global Step'},
                                yaxis: {title: 'Loss'},
                                xaxis2: {title: 'Global Step'},
                                yaxis2: {title: 'Learning Rate'},
                                height: 400
                            };
                            
                            Plotly.newPlot('training-charts', trainingData, layout);
                        });
                }
                
                // Initialize
                updateDashboard();
                updateCharts();
                
                // Set up intervals
                setInterval(updateDashboard, 5000);
                setInterval(updateCharts, 10000);
            </script>
        </body>
        </html>
        """


# Utility functions
def create_metrics_collector(db_path: Union[str, Path],
                           buffer_size: int = 1000,
                           flush_interval: float = 10.0) -> RealTimeMetricsCollector:
    """Create and initialize a metrics collector."""
    return RealTimeMetricsCollector(db_path, buffer_size, flush_interval)


def create_dashboard(metrics_collector: RealTimeMetricsCollector,
                    host: str = "127.0.0.1",
                    port: int = 8080) -> MonitoringDashboard:
    """Create and start a monitoring dashboard."""
    dashboard = MonitoringDashboard(metrics_collector, host, port)
    dashboard.start_dashboard()
    return dashboard


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Metrics Collector Demo")
    parser.add_argument("--db_path", type=str, default="./monitoring.db", help="Database path")
    parser.add_argument("--buffer_size", type=int, default=1000, help="Buffer size")
    parser.add_argument("--flush_interval", type=float, default=10.0, help="Flush interval (seconds)")
    parser.add_argument("--dashboard_host", type=str, default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--dashboard_port", type=int, default=8080, help="Dashboard port")
    
    args = parser.parse_args()
    
    # Create collector
    collector = create_metrics_collector(args.db_path, args.buffer_size, args.flush_interval)
    
    # Create dashboard
    dashboard = create_dashboard(collector, args.dashboard_host, args.dashboard_port)
    
    try:
        collector.start_collection()
        
        # Simulate real-time data collection
        for i in range(100):
            # Simulate training metrics
            from .performance_monitor import TrainingMetrics, SystemMetrics
            import random
            
            training_metrics = TrainingMetrics(
                epoch=1,
                step=i,
                global_step=i,
                loss=2.0 - i * 0.02 + random.normalvariate(0, 0.1),
                accuracy=0.1 + i * 0.008 + random.normalvariate(0, 0.05),
                learning_rate=1e-4 * (0.95 ** (i // 10)),
                grad_norm=random.uniform(0.1, 5.0),
                phase="train"
            )
            collector.collect_training_metrics(training_metrics)
            
            # Simulate system metrics
            system_metrics = SystemMetrics(
                cpu_usage_percent=random.uniform(20, 80),
                memory_usage_percent=random.uniform(30, 70),
                gpu_utilization=random.uniform(40, 90),
                gpu_memory_utilization=random.uniform(50, 85)
            )
            collector.collect_system_metrics(system_metrics)
            
            # Simulate alerts
            if i == 50:
                collector.add_alert("test_alert", "info", "Test alert triggered")
            
            time.sleep(1)
        
        print(f"Collected {collector.get_statistics()}")
        
        # Wait for dashboard
        print(f"Dashboard available at http://{args.dashboard_host}:{args.dashboard_port}")
        input("Press Enter to stop...")
        
    finally:
        collector.stop_collection()