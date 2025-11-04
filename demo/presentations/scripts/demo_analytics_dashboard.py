#!/usr/bin/env python3
"""
Demo Analytics Dashboard - Real-time demo performance tracking and insights.

This module provides comprehensive analytics capabilities including:
- Real-time demo monitoring and performance tracking
- Interactive dashboards with key metrics
- Automated reporting and insights generation
- Stakeholder-specific analytics and benchmarking
- Predictive analytics for demo optimization
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class MetricType(Enum):
    """Metric type enumeration"""
    DEMO_COMPLETION = "demo_completion"
    ENGAGEMENT_SCORE = "engagement_score"
    CONVERSION_RATE = "conversion_rate"
    STAKEHOLDER_SATISFACTION = "stakeholder_satisfaction"
    DEMO_DURATION = "demo_duration"
    SCENARIO_COMPLETION = "scenario_completion"
    FEEDBACK_QUALITY = "feedback_quality"
    RECORDING_QUALITY = "recording_quality"

class DashboardView(Enum):
    """Dashboard view options"""
    EXECUTIVE = "executive"
    CLINICAL = "clinical"
    SALES = "sales"
    OPERATIONAL = "operational"
    TECHNICAL = "technical"

@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    metric_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    context: Dict[str, Any]
    target_value: Optional[float] = None
    status: str = "normal"  # normal, warning, critical

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    title: str
    widget_type: str  # chart, gauge, table, text
    metric_types: List[MetricType]
    refresh_interval: int  # seconds
    position: Tuple[int, int]  # x, y
    size: Tuple[int, int]  # width, height

class DemoAnalyticsDashboard:
    """Real-time analytics dashboard for demo management"""
    
    def __init__(self, db_path: str = "analytics.db", dashboard_config: str = "dashboard_config.json"):
        self.db_path = db_path
        self.config_file = Path(dashboard_config)
        self.current_metrics: Dict[str, MetricValue] = {}
        self.dashboard_widgets: List[DashboardWidget] = []
        self._init_database()
        self._load_dashboard_config()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for analytics dashboard"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('demo_analytics.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Real-time metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_time_metrics (
                metric_id TEXT PRIMARY KEY,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                context TEXT,
                target_value REAL,
                status TEXT
            )
        ''')
        
        # Demo performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_performance (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                demo_type TEXT,
                stakeholder_type TEXT,
                completion_rate REAL,
                engagement_score REAL,
                conversion_probability REAL,
                duration_minutes REAL,
                feedback_rating REAL,
                recording_quality_score REAL,
                timestamp TEXT
            )
        ''')
        
        # Benchmark metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS benchmark_metrics (
                benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT,
                stakeholder_type TEXT,
                benchmark_value REAL,
                industry_average REAL,
                best_in_class REAL,
                target_value REAL,
                measurement_date TEXT
            )
        ''')
        
        # Conversion tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversion_tracking (
                conversion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                stakeholder_type TEXT,
                demo_type TEXT,
                initial_interest_level REAL,
                final_interest_level REAL,
                conversion_status TEXT,
                next_action TEXT,
                follow_up_date TEXT,
                conversion_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_dashboard_config(self):
        """Load dashboard configuration"""
        default_config = {
            "dashboard_views": {
                DashboardView.EXECUTIVE.value: {
                    "title": "Executive Overview",
                    "refresh_interval": 30,
                    "key_metrics": ["demo_completion", "conversion_rate", "stakeholder_satisfaction", "revenue_impact"],
                    "widgets": [
                        {
                            "type": "gauge",
                            "metric": "overall_demo_completion",
                            "title": "Demo Completion Rate",
                            "position": {"x": 0, "y": 0},
                            "target": 95
                        },
                        {
                            "type": "chart",
                            "metric": "conversion_rate_trend",
                            "title": "Conversion Rate Trend",
                            "position": {"x": 1, "y": 0}
                        },
                        {
                            "type": "table",
                            "metric": "stakeholder_satisfaction",
                            "title": "Stakeholder Satisfaction",
                            "position": {"x": 0, "y": 1}
                        }
                    ]
                },
                DashboardView.CLINICAL.value: {
                    "title": "Clinical Impact Dashboard",
                    "refresh_interval": 15,
                    "key_metrics": ["clinical_accuracy", "patient_outcome_improvement", "workflow_efficiency"],
                    "widgets": [
                        {
                            "type": "chart",
                            "metric": "clinical_accuracy_trend",
                            "title": "Clinical Accuracy Over Time",
                            "position": {"x": 0, "y": 0}
                        },
                        {
                            "type": "gauge",
                            "metric": "patient_outcome_score",
                            "title": "Patient Outcome Impact",
                            "position": {"x": 1, "y": 0}
                        }
                    ]
                },
                DashboardView.SALES.value: {
                    "title": "Sales Performance Dashboard",
                    "refresh_interval": 60,
                    "key_metrics": ["demo_to_meeting_conversion", "pipeline_value", "sales_cycle_length"],
                    "widgets": [
                        {
                            "type": "chart",
                            "metric": "pipeline_trend",
                            "title": "Sales Pipeline",
                            "position": {"x": 0, "y": 0}
                        },
                        {
                            "type": "gauge",
                            "metric": "conversion_rate",
                            "title": "Demo Conversion Rate",
                            "position": {"x": 1, "y": 0}
                        }
                    ]
                }
            },
            "alert_thresholds": {
                "demo_completion_rate": {"warning": 85, "critical": 70},
                "engagement_score": {"warning": 7.0, "critical": 5.0},
                "conversion_rate": {"warning": 0.4, "critical": 0.25},
                "satisfaction_score": {"warning": 7.5, "critical": 6.0}
            },
            "refresh_intervals": {
                "real_time": 5,
                "normal": 30,
                "slow": 300
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in self.config:
                        self.config[key] = value
            except Exception as e:
                self.logger.error(f"Error loading dashboard config: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_config()
    
    def save_config(self):
        """Save dashboard configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Dashboard configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving dashboard config: {e}")
    
    def update_metric(
        self,
        metric_type: MetricType,
        value: float,
        context: Optional[Dict[str, Any]] = None,
        target_value: Optional[float] = None
    ) -> str:
        """Update real-time metric"""
        metric_id = f"{metric_type.value}_{int(time.time())}"
        
        # Determine status based on thresholds
        status = self._calculate_metric_status(metric_type, value)
        
        metric = MetricValue(
            metric_id=metric_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            context=context or {},
            target_value=target_value,
            status=status
        )
        
        self.current_metrics[metric_id] = metric
        self._save_metric_to_db(metric)
        
        self.logger.info(f"Updated metric {metric_type.value}: {value}")
        return metric_id
    
    def _calculate_metric_status(self, metric_type: MetricType, value: float) -> str:
        """Calculate metric status based on thresholds"""
        thresholds = self.config.get("alert_thresholds", {})
        
        if metric_type.value in thresholds:
            threshold_config = thresholds[metric_type.value]
            critical_threshold = threshold_config.get("critical", 0)
            warning_threshold = threshold_config.get("warning", 0)
            
            if value < critical_threshold:
                return "critical"
            elif value < warning_threshold:
                return "warning"
        
        return "normal"
    
    def _save_metric_to_db(self, metric: MetricValue):
        """Save metric to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO real_time_metrics
            (metric_id, metric_type, value, timestamp, context, target_value, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            metric.metric_id,
            metric.metric_type.value,
            metric.value,
            metric.timestamp.isoformat(),
            json.dumps(metric.context),
            metric.target_value,
            metric.status
        ))
        
        conn.commit()
        conn.close()
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        current_time = datetime.now()
        recent_metrics = {}
        
        for metric_id, metric in self.current_metrics.items():
            # Include metrics from last 5 minutes
            if (current_time - metric.timestamp).total_seconds() < 300:
                recent_metrics[metric_id] = asdict(metric)
        
        return {
            "timestamp": current_time.isoformat(),
            "metrics_count": len(recent_metrics),
            "metrics": recent_metrics,
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health"""
        recent_metrics = [
            metric for metric in self.current_metrics.values()
            if (datetime.now() - metric.timestamp).total_seconds() < 300
        ]
        
        if not recent_metrics:
            return {"status": "no_data", "score": 0}
        
        # Count metrics by status
        status_counts = {}
        for metric in recent_metrics:
            status = metric.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        total_metrics = len(recent_metrics)
        
        # Calculate health score
        normal_ratio = status_counts.get("normal", 0) / total_metrics
        warning_ratio = status_counts.get("warning", 0) / total_metrics
        critical_ratio = status_counts.get("critical", 0) / total_metrics
        
        health_score = (normal_ratio * 1.0 + warning_ratio * 0.6 + critical_ratio * 0.2) * 100
        
        if health_score >= 90:
            overall_status = "excellent"
        elif health_score >= 75:
            overall_status = "good"
        elif health_score >= 60:
            overall_status = "warning"
        else:
            overall_status = "critical"
        
        return {
            "status": overall_status,
            "score": round(health_score, 1),
            "status_breakdown": status_counts,
            "total_metrics": total_metrics
        }
    
    def record_demo_performance(
        self,
        session_id: str,
        demo_type: str,
        stakeholder_type: str,
        completion_rate: float,
        engagement_score: float,
        duration_minutes: float,
        feedback_rating: Optional[float] = None,
        conversion_probability: Optional[float] = None
    ) -> str:
        """Record comprehensive demo performance"""
        performance_id = f"perf_{int(time.time())}_{session_id}"
        
        # Calculate recording quality score (simulated)
        recording_quality_score = self._simulate_recording_quality(duration_minutes)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO demo_performance
            (session_id, demo_type, stakeholder_type, completion_rate, engagement_score,
             conversion_probability, duration_minutes, feedback_rating, recording_quality_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            demo_type,
            stakeholder_type,
            completion_rate,
            engagement_score,
            conversion_probability,
            duration_minutes,
            feedback_rating,
            recording_quality_score,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Recorded demo performance: {performance_id}")
        return performance_id
    
    def _simulate_recording_quality(self, duration_minutes: float) -> float:
        """Simulate recording quality based on demo characteristics"""
        # In production, this would analyze actual recording metrics
        base_quality = 8.5
        duration_factor = min(1.0, duration_minutes / 30.0)  # Optimal around 30 minutes
        quality_score = base_quality + (duration_factor - 0.5) * 2
        return max(5.0, min(10.0, quality_score))
    
    def get_dashboard_data(self, dashboard_view: DashboardView) -> Dict[str, Any]:
        """Get dashboard data for specific view"""
        view_config = self.config.get("dashboard_views", {}).get(dashboard_view.value, {})
        
        # Get relevant metrics
        key_metrics = view_config.get("key_metrics", [])
        dashboard_data = {
            "view_name": view_config.get("title", dashboard_view.value),
            "last_updated": datetime.now().isoformat(),
            "refresh_interval": view_config.get("refresh_interval", 30),
            "metrics": {},
            "alerts": self._get_active_alerts(),
            "benchmarks": self._get_benchmark_comparison(key_metrics),
            "trends": self._get_metric_trends(key_metrics)
        }
        
        # Populate metrics
        for metric_type_str in key_metrics:
            try:
                metric_type = MetricType(metric_type_str)
                metric_data = self._get_latest_metric_value(metric_type)
                dashboard_data["metrics"][metric_type_str] = metric_data
            except ValueError:
                continue
        
        return dashboard_data
    
    def _get_latest_metric_value(self, metric_type: MetricType) -> Optional[Dict[str, Any]]:
        """Get latest value for specific metric type"""
        latest_metric = None
        latest_time = None
        
        for metric in self.current_metrics.values():
            if metric.metric_type == metric_type:
                if latest_time is None or metric.timestamp > latest_time:
                    latest_metric = metric
                    latest_time = metric.timestamp
        
        if latest_metric:
            return asdict(latest_metric)
        return None
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts"""
        alerts = []
        
        for metric in self.current_metrics.values():
            if metric.status in ["warning", "critical"]:
                alerts.append({
                    "alert_id": metric.metric_id,
                    "metric_type": metric.metric_type.value,
                    "severity": metric.status,
                    "current_value": metric.value,
                    "target_value": metric.target_value,
                    "timestamp": metric.timestamp.isoformat(),
                    "message": self._generate_alert_message(metric)
                })
        
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    def _generate_alert_message(self, metric: MetricValue) -> str:
        """Generate human-readable alert message"""
        status_messages = {
            "demo_completion": {
                "warning": "Demo completion rate is below target",
                "critical": "Demo completion rate is critically low"
            },
            "engagement_score": {
                "warning": "Audience engagement is lower than expected",
                "critical": "Audience engagement is critically low"
            },
            "conversion_rate": {
                "warning": "Demo conversion rate needs improvement",
                "critical": "Demo conversion rate is critically low"
            }
        }
        
        metric_messages = status_messages.get(metric.metric_type.value, {})
        return metric_messages.get(metric.status, f"{metric.metric_type.value} is {metric.status}")
    
    def _get_benchmark_comparison(self, metric_types: List[str]) -> Dict[str, float]:
        """Get benchmark comparisons for metrics"""
        # In production, this would query actual benchmark data
        benchmarks = {}
        
        benchmark_values = {
            "demo_completion": 95.0,
            "engagement_score": 8.5,
            "conversion_rate": 0.65,
            "stakeholder_satisfaction": 8.7,
            "clinical_accuracy": 99.7
        }
        
        for metric_type in metric_types:
            if metric_type in benchmark_values:
                benchmarks[metric_type] = benchmark_values[metric_type]
        
        return benchmarks
    
    def _get_metric_trends(self, metric_types: List[str]) -> Dict[str, List[float]]:
        """Get trend data for metrics"""
        # In production, this would analyze historical data
        trends = {}
        
        for metric_type in metric_types:
            # Generate sample trend data
            base_value = {
                "demo_completion": 92.0,
                "engagement_score": 8.2,
                "conversion_rate": 0.60,
                "stakeholder_satisfaction": 8.5
            }.get(metric_type, 7.0)
            
            # Create 30-day trend
            trend = []
            current_value = base_value
            for i in range(30):
                # Add some variation
                variation = (hash(f"{metric_type}_{i}") % 20 - 10) / 100  # -0.1 to +0.1
                current_value += variation
                trend.append(current_value)
            
            trends[metric_type] = trend
        
        return trends
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary dashboard data"""
        # Get performance data for last 30 days
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT demo_type, stakeholder_type, AVG(completion_rate) as avg_completion,
                   AVG(engagement_score) as avg_engagement, COUNT(*) as demo_count
            FROM demo_performance
            WHERE timestamp > ?
            GROUP BY demo_type, stakeholder_type
        ''', ((datetime.now() - timedelta(days=30)).isoformat(),))
        
        performance_data = cursor.fetchall()
        
        cursor.execute('''
            SELECT COUNT(*) as total_conversions, SUM(conversion_value) as total_value
            FROM conversion_tracking
            WHERE follow_up_date > ?
        ''', ((datetime.now() - timedelta(days=30)).isoformat(),))
        
        conversion_data = cursor.fetchone()
        
        conn.close()
        
        return {
            "executive_summary": {
                "reporting_period": "Last 30 Days",
                "total_demos": len(performance_data),
                "average_completion_rate": round(sum(p[2] for p in performance_data) / len(performance_data), 1) if performance_data else 0,
                "average_engagement_score": round(sum(p[3] for p in performance_data) / len(performance_data), 1) if performance_data else 0,
                "total_conversions": conversion_data[0] if conversion_data and conversion_data[0] else 0,
                "total_conversion_value": conversion_data[1] if conversion_data and conversion_data[1] else 0,
                "performance_by_stakeholder": self._analyze_stakeholder_performance(performance_data),
                "key_insights": self._generate_key_insights(performance_data),
                "recommendations": self._generate_executive_recommendations(performance_data)
            }
        }
    
    def _analyze_stakeholder_performance(self, performance_data: List) -> Dict[str, Any]:
        """Analyze performance by stakeholder type"""
        stakeholder_stats = {}
        
        for demo_type, stakeholder_type, completion_rate, engagement_score, demo_count in performance_data:
            if stakeholder_type not in stakeholder_stats:
                stakeholder_stats[stakeholder_type] = {
                    "demo_count": 0,
                    "total_completion": 0,
                    "total_engagement": 0
                }
            
            stats = stakeholder_stats[stakeholder_type]
            stats["demo_count"] += demo_count
            stats["total_completion"] += completion_rate * demo_count
            stats["total_engagement"] += engagement_score * demo_count
        
        # Calculate averages
        for stakeholder_type in stakeholder_stats:
            stats = stakeholder_stats[stakeholder_type]
            stats["average_completion_rate"] = round(stats["total_completion"] / stats["demo_count"], 1)
            stats["average_engagement_score"] = round(stats["total_engagement"] / stats["demo_count"], 1)
            del stats["total_completion"]
            del stats["total_engagement"]
        
        return stakeholder_stats
    
    def _generate_key_insights(self, performance_data: List) -> List[str]:
        """Generate key insights from performance data"""
        insights = []
        
        if not performance_data:
            return ["No recent demo data available for analysis"]
        
        # Analyze completion rates
        completion_rates = [p[2] for p in performance_data]
        avg_completion = sum(completion_rates) / len(completion_rates)
        
        if avg_completion >= 95:
            insights.append("Excellent demo completion rates across all stakeholder types")
        elif avg_completion >= 85:
            insights.append("Good demo completion rates with room for improvement")
        else:
            insights.append("Demo completion rates need significant improvement")
        
        # Analyze engagement scores
        engagement_scores = [p[3] for p in performance_data]
        avg_engagement = sum(engagement_scores) / len(engagement_scores)
        
        if avg_engagement >= 8.5:
            insights.append("High stakeholder engagement indicates strong demo impact")
        elif avg_engagement >= 7.5:
            insights.append("Good stakeholder engagement with optimization opportunities")
        else:
            insights.append("Stakeholder engagement needs improvement")
        
        return insights
    
    def _generate_executive_recommendations(self, performance_data: List) -> List[str]:
        """Generate executive recommendations"""
        recommendations = []
        
        if not performance_data:
            return ["Increase demo frequency to gather performance data"]
        
        # Analyze by demo type
        demo_performance = {}
        for demo_type, stakeholder_type, completion_rate, engagement_score, demo_count in performance_data:
            if demo_type not in demo_performance:
                demo_performance[demo_type] = []
            demo_performance[demo_type].append((completion_rate, engagement_score))
        
        # Find best and worst performing demo types
        demo_averages = {}
        for demo_type, scores in demo_performance.items():
            avg_completion = sum(s[0] for s in scores) / len(scores)
            avg_engagement = sum(s[1] for s in scores) / len(scores)
            demo_averages[demo_type] = (avg_completion, avg_engagement)
        
        if demo_averages:
            best_demo = max(demo_averages, key=lambda x: demo_averages[x][0] + demo_averages[x][1])
            worst_demo = min(demo_averages, key=lambda x: demo_averages[x][0] + demo_averages[x][1])
            
            recommendations.append(f"Scale {best_demo} demos - they show strongest performance")
            recommendations.append(f"Optimize {worst_demo} demos to improve completion rates")
        
        recommendations.extend([
            "Implement automated demo feedback collection",
            "Increase focus on high-converting stakeholder segments",
            "Develop stakeholder-specific optimization strategies"
        ])
        
        return recommendations
    
    def track_conversion(
        self,
        session_id: str,
        stakeholder_type: str,
        demo_type: str,
        initial_interest: float,
        final_interest: float,
        next_action: str,
        follow_up_date: Optional[datetime] = None,
        conversion_value: Optional[float] = None
    ) -> str:
        """Track conversion progress"""
        conversion_id = f"conv_{int(time.time())}_{session_id}"
        
        # Determine conversion status
        if final_interest >= 8.0 and next_action:
            conversion_status = "high_potential"
        elif final_interest >= 6.0:
            conversion_status = "medium_potential"
        else:
            conversion_status = "low_potential"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversion_tracking
            (session_id, stakeholder_type, demo_type, initial_interest_level, final_interest_level,
             conversion_status, next_action, follow_up_date, conversion_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            stakeholder_type,
            demo_type,
            initial_interest,
            final_interest,
            conversion_status,
            next_action,
            follow_up_date.isoformat() if follow_up_date else None,
            conversion_value
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Tracked conversion: {conversion_id}")
        return conversion_id

def main():
    """Main function for analytics dashboard CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Demo Analytics Dashboard")
    parser.add_argument("--metrics", action="store_true", help="Show real-time metrics")
    parser.add_argument("--dashboard", type=str, choices=[v.value for v in DashboardView], 
                       help="Show dashboard for specific view")
    parser.add_argument("--summary", action="store_true", help="Show executive summary")
    parser.add_argument("--metric-type", type=str, help="Update specific metric")
    parser.add_argument("--value", type=float, help="Metric value")
    parser.add_argument("--alerts", action="store_true", help="Show active alerts")
    
    args = parser.parse_args()
    
    dashboard = DemoAnalyticsDashboard()
    
    if args.metrics:
        metrics = dashboard.get_real_time_metrics()
        print(json.dumps(metrics, indent=2, default=str))
    
    elif args.dashboard:
        try:
            dashboard_view = DashboardView(args.dashboard)
            data = dashboard.get_dashboard_data(dashboard_view)
            print(json.dumps(data, indent=2, default=str))
        except ValueError:
            print(f"Invalid dashboard view: {args.dashboard}")
    
    elif args.summary:
        summary = dashboard.generate_executive_summary()
        print(json.dumps(summary, indent=2, default=str))
    
    elif args.alerts:
        alerts = dashboard._get_active_alerts()
        print(json.dumps(alerts, indent=2, default=str))
    
    elif args.metric_type and args.value is not None:
        try:
            metric_type = MetricType(args.metric_type)
            metric_id = dashboard.update_metric(metric_type, args.value)
            print(f"Updated metric {args.metric_type}: {args.value} (ID: {metric_id})")
        except ValueError:
            print(f"Invalid metric type: {args.metric_type}")

if __name__ == "__main__":
    main()