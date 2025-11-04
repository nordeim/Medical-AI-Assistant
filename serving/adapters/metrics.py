"""
Adapter Metrics and Usage Statistics System

Provides comprehensive metrics collection, usage tracking,
and analytics for medical AI adapter management.
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"          # Cumulative count
    GAUGE = "gauge"             # Current value
    HISTOGRAM = "histogram"     # Distribution of values
    RATE = "rate"              # Rate of change


class OperationType(Enum):
    """Types of operations."""
    LOAD = "load"
    UNLOAD = "unload"
    VALIDATE = "validate"
    INFERENCE = "inference"
    HOT_SWAP = "hot_swap"
    ROLLBACK = "rollback"
    HEALTH_CHECK = "health_check"
    ERROR = "error"


@dataclass
class MetricData:
    """Represents a single metric data point."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    operation_id: Optional[str] = None
    adapter_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "operation_id": self.operation_id,
            "adapter_id": self.adapter_id
        }


@dataclass
class OperationEvent:
    """Represents an operation event."""
    operation_id: str
    operation_type: OperationType
    adapter_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self, success: bool, error_msg: Optional[str] = None):
        """Mark operation as completed."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_message = error_msg
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "adapter_id": self.adapter_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for adapters."""
    adapter_id: str
    
    # Load metrics
    avg_load_time_ms: float = 0.0
    min_load_time_ms: float = 0.0
    max_load_time_ms: float = 0.0
    total_loads: int = 0
    
    # Memory metrics
    avg_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    memory_efficiency_score: float = 0.0
    
    # Inference metrics
    avg_inference_latency_ms: float = 0.0
    p95_inference_latency_ms: float = 0.0
    p99_inference_latency_ms: float = 0.0
    total_inferences: int = 0
    throughput_tokens_per_sec: float = 0.0
    
    # Reliability metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    availability_score: float = 0.0
    
    # Medical-specific metrics
    safety_score: float = 0.0
    compliance_score: float = 0.0
    clinical_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UsageStatistics:
    """Usage statistics for adapters."""
    adapter_id: str
    period_start: float
    period_end: float
    
    # Usage counts
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    
    # Operation breakdown
    operation_counts: Dict[str, int] = field(default_factory=dict)
    
    # User statistics
    unique_users: int = 0
    peak_concurrent_usage: int = 0
    avg_session_duration_minutes: float = 0.0
    
    # Performance summary
    total_duration_ms: float = 0.0
    avg_operation_duration_ms: float = 0.0
    
    # Resource utilization
    total_memory_used_mb: float = 0.0
    total_compute_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["success_rate"] = (self.successful_operations / self.total_operations) if self.total_operations > 0 else 0.0
        return data


class MetricsCollector:
    """Collects and manages adapter metrics."""
    
    def __init__(self, max_memory_points: int = 10000):
        self.max_memory_points = max_memory_points
        self._metrics_buffer: deque = deque(maxlen=max_memory_points)
        self._operation_buffer: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        
        # Real-time metrics
        self.realtime_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Aggregated metrics
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("MetricsCollector initialized")
    
    def record_metric(self, metric: MetricData):
        """Record a metric data point."""
        with self._lock:
            self._metrics_buffer.append(metric)
            
            # Update real-time metrics
            key = f"{metric.metric_name}:{metric.adapter_id or 'system'}"
            self.realtime_metrics[key].append(metric.value)
            
            # Update aggregated metrics
            self._update_aggregated_metrics(metric)
    
    def start_operation(self, operation_type: OperationType, adapter_id: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation."""
        operation_id = f"{operation_type.value}_{adapter_id}_{int(time.time() * 1000)}"
        
        operation = OperationEvent(
            operation_id=operation_id,
            operation_type=operation_type,
            adapter_id=adapter_id,
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._operation_buffer.append(operation)
        
        logger.debug(f"Started operation tracking: {operation_id}")
        return operation_id
    
    def complete_operation(self, operation_id: str, success: bool, 
                          error_message: Optional[str] = None):
        """Complete an operation."""
        with self._lock:
            # Find operation in buffer
            operation = None
            for op in reversed(self._operation_buffer):
                if op.operation_id == operation_id:
                    operation = op
                    break
            
            if operation:
                operation.complete(success, error_message)
                
                # Record completion metrics
                if operation.duration_ms:
                    self.record_metric(MetricData(
                        metric_name=f"operation_duration_ms",
                        metric_type=MetricType.HISTOGRAM,
                        value=operation.duration_ms,
                        timestamp=operation.end_time or time.time(),
                        tags={"operation_type": operation.operation_type.value},
                        adapter_id=operation.adapter_id
                    ))
                
                # Record success/failure metrics
                self.record_metric(MetricData(
                    metric_name=f"operation_{'success' if success else 'failure'}",
                    metric_type=MetricType.COUNTER,
                    value=1,
                    timestamp=time.time(),
                    tags={"operation_type": operation.operation_type.value},
                    adapter_id=operation.adapter_id
                ))
                
                logger.debug(f"Completed operation: {operation_id} ({'success' if success else 'failed'})")
            else:
                logger.warning(f"Operation not found for completion: {operation_id}")
    
    def get_realtime_metrics(self, metric_name: str, adapter_id: Optional[str] = None) -> List[float]:
        """Get recent real-time metric values."""
        key = f"{metric_name}:{adapter_id or 'system'}"
        return list(self.realtime_metrics.get(key, []))
    
    def get_aggregated_metrics(self, metric_name: str, adapter_id: Optional[str] = None) -> Dict[str, float]:
        """Get aggregated metric values."""
        key = f"{metric_name}:{adapter_id or 'system'}"
        return self.aggregated_metrics.get(key, {})
    
    def _update_aggregated_metrics(self, metric: MetricData):
        """Update aggregated metrics."""
        key = f"{metric.metric_name}:{metric.adapter_id or 'system'}"
        
        if key not in self.aggregated_metrics:
            self.aggregated_metrics[key] = {}
        
        metrics = self.aggregated_metrics[key]
        
        # Update based on metric type
        if metric.metric_type == MetricType.COUNTER:
            # For counters, we might want to track rate or total
            current_total = metrics.get("total", 0.0)
            metrics["total"] = current_total + metric.value
            
        elif metric.metric_type == MetricType.GAUGE:
            # For gauges, track current and historical stats
            metrics["current"] = metric.value
            
            # Calculate rolling statistics if we have enough data
            recent_values = list(self.realtime_metrics.get(key, []))
            if len(recent_values) >= 10:
                metrics.update({
                    "avg": np.mean(recent_values),
                    "min": np.min(recent_values),
                    "max": np.max(recent_values),
                    "std": np.std(recent_values)
                })
        
        elif metric.metric_type == MetricType.HISTOGRAM:
            # For histograms, track distribution statistics
            recent_values = list(self.realtime_metrics.get(key, []))
            if recent_values:
                metrics.update({
                    "avg": np.mean(recent_values),
                    "p50": np.percentile(recent_values, 50),
                    "p95": np.percentile(recent_values, 95),
                    "p99": np.percentile(recent_values, 99)
                })
    
    def get_performance_metrics(self, adapter_id: str) -> PerformanceMetrics:
        """Calculate performance metrics for an adapter."""
        try:
            # Get load operation metrics
            load_metrics = self.get_aggregated_metrics("operation_duration_ms", adapter_id)
            load_duration = load_metrics.get("avg", 0.0)
            load_count = len(self.get_realtime_metrics("operation_success", adapter_id))
            
            # Get memory usage metrics
            memory_metrics = self.get_aggregated_metrics("memory_usage_mb", adapter_id)
            avg_memory = memory_metrics.get("avg", 0.0)
            peak_memory = memory_metrics.get("max", 0.0)
            
            # Get inference metrics
            inference_metrics = self.get_aggregated_metrics("inference_latency_ms", adapter_id)
            avg_inference = inference_metrics.get("avg", 0.0)
            p95_inference = inference_metrics.get("p95", 0.0)
            p99_inference = inference_metrics.get("p99", 0.0)
            
            # Calculate success rate
            success_count = len([v for v in self.get_realtime_metrics("operation_success", adapter_id) if v > 0])
            failure_count = len([v for v in self.get_realtime_metrics("operation_failure", adapter_id) if v > 0])
            total_ops = success_count + failure_count
            success_rate = success_count / total_ops if total_ops > 0 else 0.0
            
            return PerformanceMetrics(
                adapter_id=adapter_id,
                avg_load_time_ms=load_duration,
                min_load_time_ms=load_metrics.get("min", 0.0),
                max_load_time_ms=load_metrics.get("max", 0.0),
                total_loads=load_count,
                avg_memory_usage_mb=avg_memory,
                peak_memory_usage_mb=peak_memory,
                avg_inference_latency_ms=avg_inference,
                p95_inference_latency_ms=p95_inference,
                p99_inference_latency_ms=p99_inference,
                success_rate=success_rate,
                error_rate=1.0 - success_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics for {adapter_id}: {e}")
            return PerformanceMetrics(adapter_id=adapter_id)
    
    def get_usage_statistics(self, adapter_id: str, hours: int = 24) -> UsageStatistics:
        """Get usage statistics for an adapter."""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            # Filter operations for the time period
            relevant_operations = [
                op for op in self._operation_buffer
                if op.adapter_id == adapter_id and op.start_time >= cutoff_time
            ]
            
            # Calculate statistics
            total_ops = len(relevant_operations)
            successful_ops = sum(1 for op in relevant_operations if op.success)
            failed_ops = total_ops - successful_ops
            
            # Operation breakdown
            op_counts = defaultdict(int)
            total_duration = 0.0
            for op in relevant_operations:
                op_counts[op.operation_type.value] += 1
                if op.duration_ms:
                    total_duration += op.duration_ms
            
            avg_duration = total_duration / total_ops if total_ops > 0 else 0.0
            
            return UsageStatistics(
                adapter_id=adapter_id,
                period_start=cutoff_time,
                period_end=time.time(),
                total_operations=total_ops,
                successful_operations=successful_ops,
                failed_operations=failed_ops,
                operation_counts=dict(op_counts),
                total_duration_ms=total_duration,
                avg_operation_duration_ms=avg_duration
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate usage statistics for {adapter_id}: {e}")
            return UsageStatistics(
                adapter_id=adapter_id,
                period_start=time.time() - (hours * 3600),
                period_end=time.time()
            )


class MetricsStorage:
    """Storage layer for metrics data using SQLite."""
    
    def __init__(self, storage_path: str = "./adapter_metrics.db"):
        self.storage_path = Path(storage_path)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database schema for metrics."""
        with sqlite3.connect(self.storage_path) as conn:
            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    operation_id TEXT,
                    adapter_id TEXT
                )
            """)
            
            # Operations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT UNIQUE NOT NULL,
                    operation_type TEXT NOT NULL,
                    adapter_id TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    metadata TEXT
                )
            """)
            
            # Performance snapshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    adapter_id TEXT NOT NULL,
                    snapshot_time REAL NOT NULL,
                    performance_data TEXT NOT NULL
                )
            """)
            
            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics (metric_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_adapter ON metrics (adapter_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_operations_adapter ON operations (adapter_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_operations_timestamp ON operations (start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_performance_adapter ON performance_snapshots (adapter_id)")
            
            conn.commit()
            
        logger.info(f"Metrics storage initialized: {self.storage_path}")
    
    def store_metric(self, metric: MetricData):
        """Store metric in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO metrics (
                        metric_name, metric_type, value, timestamp, tags,
                        operation_id, adapter_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_name,
                    metric.metric_type.value,
                    metric.value,
                    metric.timestamp,
                    json.dumps(metric.tags),
                    metric.operation_id,
                    metric.adapter_id
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def store_operation(self, operation: OperationEvent):
        """Store operation in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO operations (
                        operation_id, operation_type, adapter_id, start_time,
                        end_time, duration_ms, success, error_message, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    operation.operation_id,
                    operation.operation_type.value,
                    operation.adapter_id,
                    operation.start_time,
                    operation.end_time,
                    operation.duration_ms,
                    operation.success,
                    operation.error_message,
                    json.dumps(operation.metadata)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store operation: {e}")
    
    def store_performance_snapshot(self, adapter_id: str, performance_data: Dict[str, Any]):
        """Store performance snapshot in database."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO performance_snapshots (
                        adapter_id, snapshot_time, performance_data
                    ) VALUES (?, ?, ?)
                """, (
                    adapter_id,
                    time.time(),
                    json.dumps(performance_data)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store performance snapshot: {e}")
    
    def get_metrics_range(self, 
                         metric_name: str,
                         adapter_id: Optional[str] = None,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> List[MetricData]:
        """Get metrics within time range."""
        try:
            query = """
                SELECT metric_name, metric_type, value, timestamp, tags,
                       operation_id, adapter_id
                FROM metrics WHERE metric_name = ?
            """
            params = [metric_name]
            
            if adapter_id:
                query += " AND adapter_id = ?"
                params.append(adapter_id)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp"
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(query, params)
                
                metrics = []
                for row in cursor.fetchall():
                    metric = MetricData(
                        metric_name=row[0],
                        metric_type=MetricType(row[1]),
                        value=row[2],
                        timestamp=row[3],
                        tags=json.loads(row[4]) if row[4] else {},
                        operation_id=row[5],
                        adapter_id=row[6]
                    )
                    metrics.append(metric)
                
                return metrics
                
        except Exception as e:
            logger.error(f"Failed to get metrics range: {e}")
            return []
    
    def get_operations_range(self,
                           adapter_id: Optional[str] = None,
                           operation_type: Optional[OperationType] = None,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> List[OperationEvent]:
        """Get operations within time range."""
        try:
            query = "SELECT * FROM operations WHERE 1=1"
            params = []
            
            if adapter_id:
                query += " AND adapter_id = ?"
                params.append(adapter_id)
            
            if operation_type:
                query += " AND operation_type = ?"
                params.append(operation_type.value)
            
            if start_time:
                query += " AND start_time >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND start_time <= ?"
                params.append(end_time)
            
            query += " ORDER BY start_time"
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(query, params)
                
                operations = []
                for row in cursor.fetchall():
                    operation = OperationEvent(
                        operation_id=row[1],
                        operation_type=OperationType(row[2]),
                        adapter_id=row[3],
                        start_time=row[4],
                        end_time=row[5],
                        duration_ms=row[6],
                        success=row[7],
                        error_message=row[8],
                        metadata=json.loads(row[9]) if row[9] else {}
                    )
                    operations.append(operation)
                
                return operations
                
        except Exception as e:
            logger.error(f"Failed to get operations range: {e}")
            return []
    
    def get_performance_history(self, adapter_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get performance history for adapter."""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    SELECT snapshot_time, performance_data
                    FROM performance_snapshots
                    WHERE adapter_id = ? AND snapshot_time >= ?
                    ORDER BY snapshot_time
                """, (adapter_id, cutoff_time))
                
                history = []
                for row in cursor.fetchall():
                    data = json.loads(row[1])
                    data["snapshot_time"] = row[0]
                    history.append(data)
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get performance history for {adapter_id}: {e}")
            return []


class AdapterMetrics:
    """
    Main adapter metrics and analytics system.
    
    Features:
    - Real-time metrics collection
    - Historical data storage
    - Performance analytics
    - Usage statistics
    - Medical compliance tracking
    - Alerting and monitoring
    """
    
    def __init__(self, storage_path: str = "./adapter_metrics.db"):
        self.storage = MetricsStorage(storage_path)
        self.collector = MetricsCollector()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._monitoring_active = False
        
        # Alerting thresholds
        self.alert_thresholds = {
            "error_rate": 0.1,           # 10% error rate
            "latency_ms": 2000,          # 2 second latency
            "memory_mb": 16384,          # 16GB memory
            "load_time_ms": 30000        # 30 second load time
        }
        
        logger.info("AdapterMetrics initialized")
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring and metrics collection."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Start periodic metrics snapshot task
        snapshot_task = asyncio.create_task(
            self._periodic_snapshot_task(interval_seconds)
        )
        self._background_tasks.append(snapshot_task)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(
            self._periodic_cleanup_task(3600)  # Every hour
        )
        self._background_tasks.append(cleanup_task)
        
        logger.info("Adapter metrics monitoring started")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._background_tasks.clear()
        logger.info("Adapter metrics monitoring stopped")
    
    def record_metric(self, metric_name: str, value: float,
                     adapter_id: Optional[str] = None,
                     metric_type: MetricType = MetricType.GAUGE,
                     tags: Optional[Dict[str, str]] = None,
                     operation_id: Optional[str] = None):
        """Record a metric."""
        metric = MetricData(
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            operation_id=operation_id,
            adapter_id=adapter_id
        )
        
        # Store in memory
        self.collector.record_metric(metric)
        
        # Store in database
        self.storage.store_metric(metric)
    
    def start_operation_tracking(self, operation_type: OperationType,
                                adapter_id: str,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """Start tracking an operation."""
        return self.collector.start_operation(operation_type, adapter_id, metadata)
    
    def complete_operation_tracking(self, operation_id: str,
                                   success: bool,
                                   error_message: Optional[str] = None):
        """Complete operation tracking."""
        self.collector.complete_operation(operation_id, success, error_message)
        
        # Find and store operation
        # This is simplified - in practice, you'd want to track operations better
        pass
    
    def get_adapter_performance(self, adapter_id: str) -> Dict[str, Any]:
        """Get comprehensive performance data for adapter."""
        try:
            # Get current performance metrics
            performance = self.collector.get_performance_metrics(adapter_id)
            
            # Get usage statistics
            usage_stats = self.collector.get_usage_statistics(adapter_id)
            
            # Get historical performance
            history = self.storage.get_performance_history(adapter_id)
            
            # Get recent metrics
            recent_metrics = {}
            for metric_name in ["operation_duration_ms", "memory_usage_mb", "inference_latency_ms"]:
                metrics = self.storage.get_metrics_range(
                    metric_name, adapter_id, 
                    start_time=time.time() - 3600  # Last hour
                )
                recent_metrics[metric_name] = [m.value for m in metrics]
            
            return {
                "adapter_id": adapter_id,
                "performance_metrics": performance.to_dict(),
                "usage_statistics": usage_stats.to_dict(),
                "performance_history": history,
                "recent_metrics": recent_metrics,
                "alerts": self._check_alerts(adapter_id)
            }
            
        except Exception as e:
            logger.error(f"Failed to get adapter performance for {adapter_id}: {e}")
            return {"adapter_id": adapter_id, "error": str(e)}
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system-wide metrics overview."""
        try:
            # Get overall statistics
            active_adapters = len(set(
                m.adapter_id for m in self.collector._metrics_buffer 
                if m.adapter_id and time.time() - m.timestamp < 3600
            ))
            
            # Calculate overall success rate
            success_metrics = self.storage.get_metrics_range("operation_success")
            failure_metrics = self.storage.get_metrics_range("operation_failure")
            
            total_ops = len(success_metrics) + len(failure_metrics)
            success_rate = len(success_metrics) / total_ops if total_ops > 0 else 0.0
            
            # Get system health indicators
            recent_errors = len([m for m in self.collector._metrics_buffer 
                               if m.metric_name == "operation_failure" 
                               and time.time() - m.timestamp < 300])  # Last 5 minutes
            
            return {
                "timestamp": time.time(),
                "active_adapters": active_adapters,
                "total_operations": total_ops,
                "overall_success_rate": success_rate,
                "recent_errors_5min": recent_errors,
                "system_health": "healthy" if recent_errors < 10 else "degraded",
                "metrics_buffer_size": len(self.collector._metrics_buffer),
                "operations_buffer_size": len(self.collector._operation_buffer)
            }
            
        except Exception as e:
            logger.error(f"Failed to get system overview: {e}")
            return {"error": str(e)}
    
    def export_metrics(self, adapter_id: Optional[str] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      format: str = "json") -> Union[str, Dict[str, Any]]:
        """Export metrics data."""
        try:
            if format == "json":
                # Get metrics and operations
                metrics = self.storage.get_metrics_range(
                    "", adapter_id, start_time, end_time
                ) if adapter_id else self.storage.get_metrics_range(
                    "", None, start_time, end_time
                )
                
                operations = self.storage.get_operations_range(
                    adapter_id, None, start_time, end_time
                )
                
                # Convert to serializable format
                export_data = {
                    "export_timestamp": time.time(),
                    "adapter_id": adapter_id,
                    "time_range": {"start": start_time, "end": end_time},
                    "metrics": [m.to_dict() for m in metrics],
                    "operations": [op.to_dict() for op in operations]
                }
                
                return json.dumps(export_data, indent=2)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return json.dumps({"error": str(e)})
    
    def _check_alerts(self, adapter_id: str) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        alerts = []
        
        try:
            # Get recent performance data
            performance = self.collector.get_performance_metrics(adapter_id)
            
            # Check thresholds
            if performance.error_rate > self.alert_thresholds["error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "severity": "warning",
                    "message": f"Error rate {performance.error_rate:.1%} exceeds threshold",
                    "value": performance.error_rate,
                    "threshold": self.alert_thresholds["error_rate"]
                })
            
            if performance.avg_inference_latency_ms > self.alert_thresholds["latency_ms"]:
                alerts.append({
                    "type": "high_latency",
                    "severity": "warning",
                    "message": f"Average latency {performance.avg_inference_latency_ms:.0f}ms exceeds threshold",
                    "value": performance.avg_inference_latency_ms,
                    "threshold": self.alert_thresholds["latency_ms"]
                })
            
            if performance.peak_memory_usage_mb > self.alert_thresholds["memory_mb"]:
                alerts.append({
                    "type": "high_memory_usage",
                    "severity": "info",
                    "message": f"Peak memory usage {performance.peak_memory_usage_mb:.0f}MB",
                    "value": performance.peak_memory_usage_mb,
                    "threshold": self.alert_thresholds["memory_mb"]
                })
                
        except Exception as e:
            logger.error(f"Failed to check alerts for {adapter_id}: {e}")
        
        return alerts
    
    async def _periodic_snapshot_task(self, interval_seconds: int):
        """Periodic task to create performance snapshots."""
        try:
            while self._monitoring_active:
                # Get active adapters from recent metrics
                recent_adapters = set()
                cutoff_time = time.time() - 300  # Last 5 minutes
                
                for metric in self.collector._metrics_buffer:
                    if metric.adapter_id and metric.timestamp > cutoff_time:
                        recent_adapters.add(metric.adapter_id)
                
                # Create snapshots for active adapters
                for adapter_id in recent_adapters:
                    try:
                        performance = self.collector.get_performance_metrics(adapter_id)
                        self.storage.store_performance_snapshot(adapter_id, performance.to_dict())
                    except Exception as e:
                        logger.error(f"Failed to create snapshot for {adapter_id}: {e}")
                
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Periodic snapshot task cancelled")
        except Exception as e:
            logger.error(f"Periodic snapshot task failed: {e}")
    
    async def _periodic_cleanup_task(self, interval_seconds: int):
        """Periodic task to clean up old data."""
        try:
            while self._monitoring_active:
                # Clean up in-memory buffers
                current_time = time.time()
                cutoff_time = current_time - 86400  # 24 hours
                
                # Remove old metrics
                while (self.collector._metrics_buffer and 
                       self.collector._metrics_buffer[0].timestamp < cutoff_time):
                    self.collector._metrics_buffer.popleft()
                
                # Remove old operations
                while (self.collector._operation_buffer and 
                       self.collector._operation_buffer[0].start_time < cutoff_time):
                    self.collector._operation_buffer.popleft()
                
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
        except Exception as e:
            logger.error(f"Periodic cleanup task failed: {e}")


# Utility functions
def create_metrics_system(storage_path: str = "./adapter_metrics.db") -> AdapterMetrics:
    """Factory function to create metrics system."""
    return AdapterMetrics(storage_path)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Create metrics system
        metrics = create_metrics_system("./test_metrics.db")
        
        # Start monitoring
        await metrics.start_monitoring()
        
        try:
            # Record some test metrics
            metrics.record_metric("system_load", 0.75, metric_type=MetricType.GAUGE)
            metrics.record_metric("memory_usage_mb", 2048.0, "adapter_v1", MetricType.GAUGE)
            
            # Track an operation
            op_id = metrics.start_operation_tracking(OperationType.LOAD, "adapter_v1")
            
            # Simulate operation
            await asyncio.sleep(1.0)
            
            # Complete operation
            metrics.complete_operation_tracking(op_id, success=True)
            
            # Get performance data
            performance = metrics.get_adapter_performance("adapter_v1")
            print(f"Adapter performance: {performance}")
            
            # Get system overview
            overview = metrics.get_system_overview()
            print(f"System overview: {overview}")
            
        finally:
            # Stop monitoring
            await metrics.stop_monitoring()
    
    # asyncio.run(main())