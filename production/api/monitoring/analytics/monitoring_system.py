# Production API Monitoring and Analytics System
# Real-time monitoring with metrics, alerts, and usage analytics

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
import statistics
from collections import defaultdict, deque
import aiohttp
import aiofiles
import redis.asyncio as redis
import psutil
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"

class MetricCategory(Enum):
    """Categories of metrics"""
    API_PERFORMANCE = "api_performance"
    API_USAGE = "api_usage"
    ERROR_RATE = "error_rate"
    SECURITY = "security"
    BUSINESS = "business"
    SYSTEM = "system"

@dataclass
class Metric:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    category: MetricCategory
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # e.g., ">", "<", ">=", "<=", "=="
    threshold: float
    severity: AlertSeverity
    time_window: int  # minutes
    description: str
    enabled: bool = True
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class APICall:
    """Individual API call record"""
    timestamp: datetime
    method: str
    path: str
    status_code: int
    response_time: float
    user_id: Optional[str] = None
    api_key: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    error_message: Optional[str] = None
    
    def to_metric(self) -> Metric:
        """Convert API call to metrics"""
        return [
            Metric(
                name="api_request_duration_seconds",
                value=self.response_time,
                metric_type=MetricType.TIMER,
                category=MetricCategory.API_PERFORMANCE,
                timestamp=self.timestamp,
                labels={
                    "method": self.method,
                    "endpoint": self.path,
                    "status_code": str(self.status_code)
                },
                unit="seconds"
            ),
            Metric(
                name="api_requests_total",
                value=1,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.API_USAGE,
                timestamp=self.timestamp,
                labels={
                    "method": self.method,
                    "endpoint": self.path,
                    "status_code": str(self.status_code),
                    "user_id": self.user_id or "anonymous"
                }
            ),
            Metric(
                name="api_error_rate",
                value=1 if self.status_code >= 400 else 0,
                metric_type=MetricType.COUNTER,
                category=MetricCategory.ERROR_RATE,
                timestamp=self.timestamp,
                labels={
                    "method": self.method,
                    "endpoint": self.path,
                    "status_code": str(self.status_code)
                }
            )
        ]

class MetricsCollector:
    """Collects and stores API metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: deque = deque(maxlen=100000)  # Keep last 100k metrics
        self.redis_client = None
        self.influx_client = None
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize external monitoring clients"""
        try:
            # Redis for real-time metrics
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            # self.redis_client = redis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client: {e}")
        
        try:
            # InfluxDB for time-series data
            influx_url = self.config.get("influx_url")
            if influx_url:
                # self.influx_client = InfluxDBClient(url=influx_url, ...)
                pass
        except Exception as e:
            logger.warning(f"Failed to initialize InfluxDB client: {e}")
    
    async def record_api_call(self, api_call: APICall):
        """Record an API call"""
        metrics = api_call.to_metric()
        for metric in metrics:
            await self.record_metric(metric)
    
    async def record_metric(self, metric: Metric):
        """Record a single metric"""
        self.metrics.append(metric)
        
        # Store in Redis for real-time queries
        if self.redis_client:
            try:
                await self._store_in_redis(metric)
            except Exception as e:
                logger.error(f"Failed to store metric in Redis: {e}")
        
        # Store in InfluxDB for long-term storage
        if self.influx_client:
            try:
                await self._store_in_influxdb(metric)
            except Exception as e:
                logger.error(f"Failed to store metric in InfluxDB: {e}")
    
    async def _store_in_redis(self, metric: Metric):
        """Store metric in Redis for real-time access"""
        key = f"metrics:{metric.name}:{int(metric.timestamp.timestamp())}"
        value = json.dumps({
            "name": metric.name,
            "value": metric.value,
            "type": metric.metric_type.value,
            "category": metric.category.value,
            "timestamp": metric.timestamp.isoformat(),
            "labels": metric.labels,
            "unit": metric.unit
        })
        
        await self.redis_client.setex(key, 3600, value)  # Expire after 1 hour
        
        # Also store in sorted set for time-series queries
        score = metric.timestamp.timestamp()
        await self.redis_client.zadd(f"timeseries:{metric.name}", {key: score})
    
    async def _store_in_influxdb(self, metric: Metric):
        """Store metric in InfluxDB for long-term storage"""
        point = {
            "measurement": metric.name,
            "tags": metric.labels,
            "fields": {"value": float(metric.value)},
            "time": metric.timestamp.isoformat() + "Z"
        }
        
        # self.influx_client.write_points([point])
    
    def get_recent_metrics(
        self,
        metric_name: str,
        minutes: int = 60,
        labels: Dict[str, str] = None
    ) -> List[Metric]:
        """Get recent metrics by name and optional labels"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        recent_metrics = [
            m for m in self.metrics 
            if m.name == metric_name 
            and m.timestamp > cutoff_time
            and self._matches_labels(m.labels, labels or {})
        ]
        
        return sorted(recent_metrics, key=lambda m: m.timestamp)
    
    def _matches_labels(self, metric_labels: Dict[str, str], filter_labels: Dict[str, str]) -> bool:
        """Check if metric labels match filter labels"""
        for key, value in filter_labels.items():
            if metric_labels.get(key) != value:
                return False
        return True
    
    def aggregate_metrics(
        self,
        metrics: List[Metric],
        aggregation: str = "avg"
    ) -> Dict[str, Any]:
        """Aggregate metrics over time"""
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        aggregations = {
            "avg": statistics.mean,
            "median": statistics.median,
            "min": min,
            "max": max,
            "count": len,
            "sum": sum,
            "std": statistics.stdev
        }
        
        func = aggregations.get(aggregation, statistics.mean)
        
        return {
            "metric_name": metrics[0].name,
            "aggregation": aggregation,
            "value": func(values),
            "sample_count": len(values),
            "time_range": {
                "start": min(m.timestamp for m in metrics).isoformat(),
                "end": max(m.timestamp for m in metrics).isoformat()
            }
        }

class AlertManager:
    """Manages alerts based on metric thresholds"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    async def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alert rules and trigger alerts if needed"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            alert_triggered = await self._check_rule(rule)
            
            if alert_triggered:
                alert_info = await self._trigger_alert(rule)
                triggered_alerts.append(alert_info)
        
        return triggered_alerts
    
    async def _check_rule(self, rule: AlertRule) -> bool:
        """Check if alert rule conditions are met"""
        # Get metrics for the time window
        metrics = self.metrics_collector.get_recent_metrics(
            rule.metric_name,
            minutes=rule.time_window
        )
        
        if not metrics:
            return False
        
        # Aggregate metrics
        aggregated = self.metrics_collector.aggregate_metrics(metrics, "avg")
        current_value = aggregated["value"]
        
        # Check threshold condition
        threshold_met = self._check_threshold(current_value, rule.condition, rule.threshold)
        
        return threshold_met
    
    def _check_threshold(self, value: float, condition: str, threshold: float) -> bool:
        """Check if value meets threshold condition"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001  # Floating point comparison
        else:
            return False
    
    async def _trigger_alert(self, rule: AlertRule) -> Dict[str, Any]:
        """Trigger an alert"""
        alert_id = f"{rule.name}:{int(time.time())}"
        
        alert_info = {
            "alert_id": alert_id,
            "rule_name": rule.name,
            "severity": rule.severity.value,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "description": rule.description,
            "labels": rule.labels
        }
        
        # Store active alert
        self.active_alerts[alert_id] = alert_info
        
        # Log alert
        logger.warning(f"ALERT TRIGGERED: {rule.name} - {rule.description}")
        
        # Send notification (email, Slack, PagerDuty, etc.)
        await self._send_alert_notification(alert_info)
        
        # Store in history
        self.alert_history.append(alert_info)
        
        return alert_info
    
    async def _send_alert_notification(self, alert_info: Dict[str, Any]):
        """Send alert notification to external systems"""
        # In production, this would send to:
        # - Email
        # - Slack
        # - PagerDuty
        # - SMS
        # - PagerDuty
        
        logger.info(f"Sending alert notification: {alert_info['alert_id']}")

class AnalyticsEngine:
    """Analyzes API usage patterns and generates insights"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    async def generate_usage_report(
        self,
        start_time: datetime,
        end_time: datetime,
        group_by: str = "hour"
    ) -> Dict[str, Any]:
        """Generate comprehensive usage report"""
        
        # Get all API calls in time range
        metrics = self.metrics_collector.get_recent_metrics(
            "api_requests_total",
            minutes=int((end_time - start_time).total_seconds() / 60)
        )
        
        # Filter by time range
        time_filtered_metrics = [
            m for m in metrics 
            if start_time <= m.timestamp <= end_time
        ]
        
        # Generate analytics
        report = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "grouping": group_by
            },
            "summary": await self._generate_usage_summary(time_filtered_metrics),
            "endpoint_breakdown": await self._analyze_endpoint_usage(time_filtered_metrics),
            "user_analysis": await self._analyze_user_patterns(time_filtered_metrics),
            "performance_metrics": await self._analyze_performance(time_filtered_metrics),
            "error_analysis": await self._analyze_errors(time_filtered_metrics),
            "security_insights": await self._analyze_security_patterns(time_filtered_metrics)
        }
        
        return report
    
    async def _generate_usage_summary(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Generate usage summary statistics"""
        if not metrics:
            return {}
        
        total_requests = len(metrics)
        total_users = len(set(m.labels.get("user_id", "anonymous") for m in metrics))
        
        # Group by time periods
        time_groups = defaultdict(int)
        for metric in metrics:
            key = metric.timestamp.strftime("%Y-%m-%d %H:00")
            time_groups[key] += 1
        
        peak_hour = max(time_groups, key=time_groups.get) if time_groups else None
        peak_requests = time_groups[peak_hour] if peak_hour else 0
        
        return {
            "total_requests": total_requests,
            "unique_users": total_users,
            "requests_per_user": total_requests / max(total_users, 1),
            "peak_hour": peak_hour,
            "peak_hour_requests": peak_requests,
            "time_distribution": dict(time_groups)
        }
    
    async def _analyze_endpoint_usage(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Analyze endpoint usage patterns"""
        endpoint_stats = defaultdict(lambda: {"count": 0, "methods": set(), "users": set()})
        
        for metric in metrics:
            endpoint = metric.labels.get("endpoint", "unknown")
            method = metric.labels.get("method", "unknown")
            user = metric.labels.get("user_id", "anonymous")
            
            endpoint_stats[endpoint]["count"] += 1
            endpoint_stats[endpoint]["methods"].add(method)
            endpoint_stats[endpoint]["users"].add(user)
        
        # Convert to serializable format
        analysis = {}
        for endpoint, stats in endpoint_stats.items():
            analysis[endpoint] = {
                "total_requests": stats["count"],
                "unique_methods": len(stats["methods"]),
                "unique_users": len(stats["users"]),
                "methods": list(stats["methods"])
            }
        
        # Sort by usage
        sorted_endpoints = sorted(
            analysis.items(),
            key=lambda x: x[1]["total_requests"],
            reverse=True
        )
        
        return {
            "total_endpoints": len(analysis),
            "top_endpoints": dict(sorted_endpoints[:10]),
            "endpoint_details": analysis
        }
    
    async def _analyze_user_patterns(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        user_stats = defaultdict(lambda: {
            "request_count": 0,
            "endpoints": set(),
            "first_request": None,
            "last_request": None
        })
        
        for metric in metrics:
            user_id = metric.labels.get("user_id", "anonymous")
            endpoint = metric.labels.get("endpoint", "unknown")
            
            user_stats[user_id]["request_count"] += 1
            user_stats[user_id]["endpoints"].add(endpoint)
            
            if not user_stats[user_id]["first_request"]:
                user_stats[user_id]["first_request"] = metric.timestamp
            
            user_stats[user_id]["last_request"] = metric.timestamp
        
        # Convert to serializable format
        user_analysis = {}
        for user_id, stats in user_stats.items():
            session_duration = None
            if stats["first_request"] and stats["last_request"]:
                session_duration = (stats["last_request"] - stats["first_request"]).total_seconds()
            
            user_analysis[user_id] = {
                "total_requests": stats["request_count"],
                "unique_endpoints": len(stats["endpoints"]),
                "endpoints_used": list(stats["endpoints"]),
                "session_duration_seconds": session_duration,
                "activity_period": {
                    "start": stats["first_request"].isoformat() if stats["first_request"] else None,
                    "end": stats["last_request"].isoformat() if stats["last_request"] else None
                }
            }
        
        return {
            "total_users": len(user_analysis),
            "user_details": user_analysis,
            "power_users": sorted(
                [(uid, stats) for uid, stats in user_analysis.items()],
                key=lambda x: x[1]["total_requests"],
                reverse=True
            )[:10]
        }
    
    async def _analyze_performance(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Analyze API performance metrics"""
        
        # Get response time metrics
        response_time_metrics = self.metrics_collector.get_recent_metrics(
            "api_request_duration_seconds",
            minutes=1440  # Last 24 hours
        )
        
        if not response_time_metrics:
            return {}
        
        values = [m.value for m in response_time_metrics]
        
        performance = {
            "average_response_time": statistics.mean(values),
            "median_response_time": statistics.median(values),
            "p95_response_time": np.percentile(values, 95),
            "p99_response_time": np.percentile(values, 99),
            "min_response_time": min(values),
            "max_response_time": max(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0,
            "sample_size": len(values)
        }
        
        # Analyze by endpoint
        endpoint_performance = defaultdict(list)
        for metric in response_time_metrics:
            endpoint = metric.labels.get("endpoint", "unknown")
            endpoint_performance[endpoint].append(metric.value)
        
        # Calculate per-endpoint stats
        endpoint_stats = {}
        for endpoint, times in endpoint_performance.items():
            endpoint_stats[endpoint] = {
                "avg_response_time": statistics.mean(times),
                "p95_response_time": np.percentile(times, 95),
                "sample_count": len(times)
            }
        
        performance["endpoint_breakdown"] = endpoint_stats
        
        return performance
    
    async def _analyze_errors(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Analyze error patterns"""
        
        error_metrics = [m for m in metrics if m.labels.get("status_code", "200").startswith(("4", "5"))]
        
        if not error_metrics:
            return {"total_errors": 0, "error_rate": 0.0}
        
        # Group errors by status code
        error_by_code = defaultdict(int)
        error_by_endpoint = defaultdict(int)
        
        for metric in error_metrics:
            status_code = metric.labels.get("status_code", "unknown")
            endpoint = metric.labels.get("endpoint", "unknown")
            
            error_by_code[status_code] += 1
            error_by_endpoint[endpoint] += 1
        
        total_requests = len(metrics)
        total_errors = len(error_metrics)
        error_rate = total_errors / max(total_requests, 1)
        
        return {
            "total_errors": total_errors,
            "error_rate": error_rate,
            "errors_by_status_code": dict(error_by_code),
            "errors_by_endpoint": dict(error_by_endpoint),
            "common_errors": sorted(
                error_by_code.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    async def _analyze_security_patterns(self, metrics: List[Metric]) -> Dict[str, Any]:
        """Analyze security-related patterns"""
        
        # Analyze request patterns for potential security issues
        ip_requests = defaultdict(int)
        user_agent_requests = defaultdict(int)
        
        # Note: In a real implementation, you'd store IP and UA info
        # For now, using labels as proxy
        for metric in metrics:
            # Simulated security analysis
            if metric.labels.get("user_id") == "anonymous":
                ip_requests["anonymous"] = ip_requests["anonymous"] + 1
        
        return {
            "anonymous_requests": ip_requests.get("anonymous", 0),
            "potential_issues": [
                "Monitor for excessive anonymous requests",
                "Review authentication failure patterns",
                "Check for unusual traffic patterns"
            ]
        }

class MonitoringDashboard:
    """Real-time monitoring dashboard"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        
        now = datetime.now(timezone.utc)
        last_5_min = now - timedelta(minutes=5)
        last_hour = now - timedelta(hours=1)
        last_24_hours = now - timedelta(hours=24)
        
        # Get recent metrics
        recent_requests = self.metrics_collector.get_recent_metrics("api_requests_total", minutes=5)
        recent_errors = [m for m in recent_requests if m.labels.get("status_code", "").startswith(("4", "5"))]
        
        # Calculate rates
        requests_per_minute = len(recent_requests) / 5 if recent_requests else 0
        error_rate = len(recent_errors) / max(len(recent_requests), 1)
        
        # Get response time metrics
        response_times = self.metrics_collector.get_recent_metrics(
            "api_request_duration_seconds",
            minutes=60
        )
        
        avg_response_time = 0
        if response_times:
            values = [m.value for m in response_times]
            avg_response_time = statistics.mean(values)
        
        return {
            "timestamp": now.isoformat(),
            "real_time_metrics": {
                "requests_per_minute": round(requests_per_minute, 2),
                "error_rate_percent": round(error_rate * 100, 2),
                "average_response_time_ms": round(avg_response_time * 1000, 2)
            },
            "hourly_summary": await self._get_hourly_summary(last_hour),
            "top_endpoints": await self._get_top_endpoints(last_hour),
            "alert_status": await self._get_alert_status(),
            "system_health": await self._get_system_health()
        }
    
    async def _get_hourly_summary(self, start_time: datetime) -> Dict[str, Any]:
        """Get hourly usage summary"""
        
        metrics = self.metrics_collector.get_recent_metrics(
            "api_requests_total",
            minutes=int((datetime.now(timezone.utc) - start_time).total_seconds() / 60)
        )
        
        if not metrics:
            return {"total_requests": 0, "error_count": 0}
        
        total_requests = len(metrics)
        error_requests = len([m for m in metrics if m.labels.get("status_code", "").startswith(("4", "5"))])
        
        return {
            "total_requests": total_requests,
            "error_count": error_requests,
            "error_rate_percent": round(error_requests / max(total_requests, 1) * 100, 2)
        }
    
    async def _get_top_endpoints(self, start_time: datetime) -> List[Dict[str, Any]]:
        """Get top used endpoints"""
        
        metrics = self.metrics_collector.get_recent_metrics(
            "api_requests_total",
            minutes=int((datetime.now(timezone.utc) - start_time).total_seconds() / 60)
        )
        
        endpoint_counts = defaultdict(int)
        for metric in metrics:
            endpoint = metric.labels.get("endpoint", "unknown")
            endpoint_counts[endpoint] += 1
        
        return [
            {"endpoint": endpoint, "request_count": count}
            for endpoint, count in sorted(endpoint_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
    
    async def _get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status"""
        # This would integrate with the AlertManager
        return {
            "active_alerts": 0,
            "critical_alerts": 0,
            "warning_alerts": 0,
            "last_alert_time": None
        }
    
    async def _get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        
        return {
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "network_io": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            }
        }

# Default alert rules for healthcare API
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_error_rate",
        metric_name="api_error_rate",
        condition=">",
        threshold=0.05,  # 5% error rate
        severity=AlertSeverity.CRITICAL,
        time_window=15,
        description="API error rate is too high (5-minute average > 5%)",
        labels={"category": "reliability"}
    ),
    AlertRule(
        name="high_response_time",
        metric_name="api_request_duration_seconds",
        condition=">",
        threshold=2.0,  # 2 seconds
        severity=AlertSeverity.WARNING,
        time_window=30,
        description="API response time is high (30-minute average > 2s)",
        labels={"category": "performance"}
    ),
    AlertRule(
        name="low_throughput",
        metric_name="api_requests_total",
        condition="<",
        threshold=10.0,  # 10 requests per minute
        severity=AlertSeverity.WARNING,
        time_window=60,
        description="API throughput is low (60-minute average < 10 RPM)",
        labels={"category": "availability"}
    )
]

# Example usage
if __name__ == "__main__":
    # Initialize monitoring system
    config = {
        "redis_url": "redis://localhost:6379",
        "influx_url": "http://localhost:8086"
    }
    
    metrics_collector = MetricsCollector(config)
    
    # Set up alert rules
    alert_manager = AlertManager(metrics_collector)
    for rule in DEFAULT_ALERT_RULES:
        alert_manager.add_alert_rule(rule)
    
    # Initialize analytics
    analytics = AnalyticsEngine(metrics_collector)
    
    # Create dashboard
    dashboard = MonitoringDashboard(metrics_collector)
    
    print("Healthcare API Monitoring System initialized")
    print(f"Configured {len(DEFAULT_ALERT_RULES)} alert rules")
    
    # Simulate some API calls
    import random
    
    for i in range(100):
        api_call = APICall(
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 60)),
            method=random.choice(["GET", "POST", "PUT", "DELETE"]),
            path=random.choice([
                "/api/v1/patients",
                "/api/v1/observations", 
                "/fhir/Patient",
                "/analytics/metrics"
            ]),
            status_code=random.choice([200, 200, 200, 200, 400, 500]),  # Mostly 200s
            response_time=random.uniform(0.1, 2.0),
            user_id=f"user_{random.randint(1, 10)}"
        )
        
        asyncio.run(metrics_collector.record_api_call(api_call))
    
    # Generate analytics report
    report = asyncio.run(analytics.generate_usage_report(
        datetime.now(timezone.utc) - timedelta(hours=1),
        datetime.now(timezone.utc)
    ))
    
    print(f"Generated usage report with {report['summary']['total_requests']} requests")
    print(f"Top endpoint: {list(report['endpoint_breakdown']['top_endpoints'].keys())[0] if report['endpoint_breakdown']['top_endpoints'] else 'None'}")