"""
Demo Analytics and Usage Tracking System
Tracks user behavior, system performance, and demo completion metrics
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import statistics

@dataclass
class UserAction:
    """Single user action tracking"""
    user_id: int
    session_id: str
    action_type: str
    component: str
    timestamp: datetime
    duration_ms: Optional[int] = None
    success: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class SystemMetric:
    """System performance metric"""
    timestamp: datetime
    metric_type: str  # 'response_time', 'memory_usage', 'cpu_usage', etc.
    value: float
    unit: str
    component: str  # 'api', 'database', 'ai_model', etc.

@dataclass
class DemoSession:
    """Complete demo session tracking"""
    session_id: str
    user_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    scenario_id: Optional[str] = None
    total_actions: int = 0
    completion_percentage: float = 0.0
    feedback_score: Optional[int] = None
    user_agent: str = ""
    ip_address: str = ""

@dataclass
class DemoCompletion:
    """Demo scenario completion metrics"""
    scenario_id: str
    scenario_name: str
    user_id: int
    start_time: datetime
    end_time: datetime
    completion_status: str  # 'completed', 'abandoned', 'failed'
    steps_completed: int
    total_steps: int
    total_duration_minutes: float
    user_satisfaction: Optional[int]

class DemoAnalyticsManager:
    """Manages demo analytics and usage tracking"""
    
    def __init__(self, db_path: str = "demo_analytics.db"):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # User actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                component TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_ms INTEGER,
                success BOOLEAN DEFAULT 1,
                metadata TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                component TEXT NOT NULL
            )
        """)
        
        # Demo sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS demo_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                scenario_id TEXT,
                total_actions INTEGER DEFAULT 0,
                completion_percentage REAL DEFAULT 0.0,
                feedback_score INTEGER,
                user_agent TEXT,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Demo completion tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS demo_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id TEXT NOT NULL,
                scenario_name TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                completion_status TEXT NOT NULL,
                steps_completed INTEGER NOT NULL,
                total_steps INTEGER NOT NULL,
                total_duration_minutes REAL NOT NULL,
                user_satisfaction INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Performance benchmarks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                component TEXT NOT NULL,
                operation TEXT NOT NULL,
                average_response_time REAL NOT NULL,
                p95_response_time REAL NOT NULL,
                success_rate REAL NOT NULL,
                sample_size INTEGER NOT NULL
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON user_actions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_user ON user_actions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_session ON user_actions(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON demo_sessions(user_id)")
        
        conn.commit()
        conn.close()
        
    def track_user_action(self, action: UserAction):
        """Track a single user action"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO user_actions 
            (user_id, session_id, action_type, component, timestamp, duration_ms, success, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            action.user_id,
            action.session_id,
            action.action_type,
            action.component,
            action.timestamp,
            action.duration_ms,
            action.success,
            json.dumps(action.metadata) if action.metadata else None
        ))
        
        conn.commit()
        conn.close()
        
    def track_system_metric(self, metric: SystemMetric):
        """Track system performance metric"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO system_metrics (timestamp, metric_type, value, unit, component)
            VALUES (?, ?, ?, ?, ?)
        """, (
            metric.timestamp,
            metric.metric_type,
            metric.value,
            metric.unit,
            metric.component
        ))
        
        conn.commit()
        conn.close()
        
    def start_demo_session(self, session_id: str, user_id: int, 
                          user_agent: str = "", ip_address: str = "") -> bool:
        """Start tracking a demo session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO demo_sessions 
                (session_id, user_id, user_agent, ip_address)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_id, user_agent, ip_address))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error starting demo session: {e}")
            return False
        finally:
            conn.close()
            
    def end_demo_session(self, session_id: str, feedback_score: Optional[int] = None) -> bool:
        """End tracking a demo session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE demo_sessions 
                SET end_time = ?, feedback_score = ?
                WHERE session_id = ?
            """, (datetime.now(), feedback_score, session_id))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error ending demo session: {e}")
            return False
        finally:
            conn.close()
            
    def track_demo_completion(self, completion: DemoCompletion):
        """Track demo scenario completion"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO demo_completions 
            (scenario_id, scenario_name, user_id, start_time, end_time, 
             completion_status, steps_completed, total_steps, 
             total_duration_minutes, user_satisfaction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            completion.scenario_id,
            completion.scenario_name,
            completion.user_id,
            completion.start_time,
            completion.end_time,
            completion.completion_status,
            completion.steps_completed,
            completion.total_steps,
            completion.total_duration_minutes,
            completion.user_satisfaction
        ))
        
        conn.commit()
        conn.close()
        
    def get_user_analytics(self, user_id: Optional[int] = None, 
                          days: int = 7) -> Dict[str, Any]:
        """Get user analytics for specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Time range filter
        start_date = datetime.now() - timedelta(days=days)
        
        # User actions analysis
        if user_id:
            cursor.execute("""
                SELECT action_type, component, success, duration_ms, timestamp
                FROM user_actions 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (user_id, start_date))
        else:
            cursor.execute("""
                SELECT action_type, component, success, duration_ms, timestamp
                FROM user_actions 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (start_date,))
            
        actions = cursor.fetchall()
        
        # Process analytics
        total_actions = len(actions)
        successful_actions = sum(1 for a in actions if a[2])
        avg_response_time = statistics.mean([a[3] for a in actions if a[3]]) if any(a[3] for a in actions) else 0
        
        action_types = Counter(a[0] for a in actions)
        components = Counter(a[1] for a in actions)
        daily_activity = defaultdict(int)
        
        for action in actions:
            day = action[4].split(' ')[0]
            daily_activity[day] += 1
            
        # Session analytics
        if user_id:
            cursor.execute("""
                SELECT session_id, start_time, end_time, total_actions, completion_percentage, feedback_score
                FROM demo_sessions 
                WHERE user_id = ? AND start_time >= ?
            """, (user_id, start_date))
        else:
            cursor.execute("""
                SELECT session_id, start_time, end_time, total_actions, completion_percentage, feedback_score
                FROM demo_sessions 
                WHERE start_time >= ?
            """, (start_date,))
            
        sessions = cursor.fetchall()
        
        conn.close()
        
        return {
            "period_days": days,
            "total_actions": total_actions,
            "successful_actions": successful_actions,
            "success_rate": (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            "average_response_time_ms": round(avg_response_time, 2),
            "most_common_actions": dict(action_types.most_common(10)),
            "most_used_components": dict(components.most_common(10)),
            "daily_activity": dict(daily_activity),
            "total_sessions": len(sessions),
            "average_session_duration": self._calculate_avg_session_duration(sessions),
            "completion_rate": self._calculate_completion_rate(sessions)
        }
        
    def get_system_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get system performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute("""
            SELECT metric_type, value, unit, component, timestamp
            FROM system_metrics 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (start_time,))
        
        metrics = cursor.fetchall()
        
        # Group by metric type
        metrics_by_type = defaultdict(list)
        for metric in metrics:
            metrics_by_type[metric[0]].append({
                'value': metric[1],
                'unit': metric[2],
                'component': metric[3],
                'timestamp': metric[4]
            })
            
        # Calculate statistics
        performance_summary = {}
        for metric_type, values in metrics_by_type.items():
            metric_values = [m['value'] for m in values]
            performance_summary[metric_type] = {
                'average': statistics.mean(metric_values),
                'median': statistics.median(metric_values),
                'min': min(metric_values),
                'max': max(metric_values),
                'sample_count': len(metric_values),
                'unit': values[0]['unit'] if values else '',
                'latest': values[0]['value'] if values else 0
            }
            
        conn.close()
        
        return {
            "period_hours": hours,
            "total_metrics": len(metrics),
            "performance_summary": performance_summary,
            "components_monitored": list(set(m[3] for m in metrics))
        }
        
    def get_demo_scenarios_analytics(self) -> Dict[str, Any]:
        """Get analytics for demo scenarios"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT scenario_id, scenario_name, completion_status, 
                   steps_completed, total_steps, total_duration_minutes, 
                   user_satisfaction, start_time
            FROM demo_completions 
            ORDER BY start_time DESC
        """)
        
        completions = cursor.fetchall()
        
        # Group by scenario
        scenarios = defaultdict(list)
        for completion in completions:
            scenarios[completion[1]].append({
                'status': completion[2],
                'completion_rate': completion[3] / completion[4],
                'duration': completion[5],
                'satisfaction': completion[6],
                'date': completion[7]
            })
            
        scenario_analytics = {}
        for scenario_name, data in scenarios.items():
            completions_only = [d for d in data if d['status'] == 'completed']
            if completions_only:
                scenario_analytics[scenario_name] = {
                    'total_attempts': len(data),
                    'completion_rate': len(completions_only) / len(data) * 100,
                    'average_duration_minutes': statistics.mean([d['duration'] for d in completions_only]),
                    'average_satisfaction': statistics.mean([d['satisfaction'] for d in completions_only if d['satisfaction']]),
                    'recent_trend': 'improving' if len(completions_only) >= 3 else 'stable'
                }
                
        conn.close()
        
        return {
            "total_scenarios": len(scenarios),
            "total_completions": len(completions),
            "scenario_analytics": scenario_analytics
        }
        
    def get_demo_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive demo dashboard data"""
        return {
            "user_analytics": self.get_user_analytics(days=1),  # Last 24 hours
            "system_performance": self.get_system_performance(hours=1),  # Last hour
            "scenario_analytics": self.get_demo_scenarios_analytics(),
            "active_sessions": self._get_active_sessions(),
            "real_time_metrics": self._get_real_time_metrics()
        }
        
    def _calculate_avg_session_duration(self, sessions: List) -> float:
        """Calculate average session duration"""
        completed_sessions = [s for s in sessions if s[2]]  # Has end_time
        if not completed_sessions:
            return 0
            
        durations = []
        for session in completed_sessions:
            start = datetime.fromisoformat(session[1])
            end = datetime.fromisoformat(session[2])
            durations.append((end - start).total_seconds() / 60)  # Minutes
            
        return round(statistics.mean(durations), 2)
        
    def _calculate_completion_rate(self, sessions: List) -> float:
        """Calculate demo completion rate"""
        total_sessions = len(sessions)
        if total_sessions == 0:
            return 0
            
        completed_sessions = [s for s in sessions if s[4] >= 80]  # 80%+ completion
        return round(len(completed_sessions) / total_sessions * 100, 2)
        
    def _get_active_sessions(self) -> List[Dict]:
        """Get currently active demo sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions without end time in last 4 hours
        cursor.execute("""
            SELECT s.session_id, u.first_name, u.last_name, s.start_time, s.total_actions
            FROM demo_sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.end_time IS NULL AND s.start_time >= datetime('now', '-4 hours')
            ORDER BY s.start_time DESC
        """)
        
        sessions = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': session[0],
                'user_name': f"{session[1]} {session[2]}",
                'start_time': session[3],
                'actions_count': session[4],
                'duration_minutes': round((datetime.now() - datetime.fromisoformat(session[3])).total_seconds() / 60, 1)
            }
            for session in sessions
        ]
        
    def _get_real_time_metrics(self) -> Dict[str, float]:
        """Get current real-time system metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Latest metrics by type
        metrics = {}
        metric_types = ['response_time', 'memory_usage', 'cpu_usage', 'request_rate']
        
        for metric_type in metric_types:
            cursor.execute("""
                SELECT value FROM system_metrics 
                WHERE metric_type = ? 
                ORDER BY timestamp DESC LIMIT 1
            """, (metric_type,))
            
            result = cursor.fetchone()
            metrics[metric_type] = result[0] if result else 0
            
        conn.close()
        return metrics

class DemoTracker:
    """Convenience class for tracking demo events"""
    
    def __init__(self, analytics_manager: DemoAnalyticsManager):
        self.analytics = analytics_manager
        
    def track_page_view(self, user_id: int, session_id: str, page: str, 
                       duration_ms: Optional[int] = None):
        """Track page view"""
        action = UserAction(
            user_id=user_id,
            session_id=session_id,
            action_type="page_view",
            component=page,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            metadata={"page": page}
        )
        self.analytics.track_user_action(action)
        
    def track_api_call(self, user_id: int, session_id: str, endpoint: str, 
                      duration_ms: int, success: bool = True, metadata: Dict = None):
        """Track API call"""
        action = UserAction(
            user_id=user_id,
            session_id=session_id,
            action_type="api_call",
            component=endpoint,
            timestamp=datetime.now(),
            duration_ms=duration_ms,
            success=success,
            metadata=metadata
        )
        self.analytics.track_user_action(action)
        
    def track_scenario_step(self, user_id: int, session_id: str, 
                           step: str, success: bool = True):
        """Track scenario step completion"""
        action = UserAction(
            user_id=user_id,
            session_id=session_id,
            action_type="scenario_step",
            component="demo_scenario",
            timestamp=datetime.now(),
            success=success,
            metadata={"step": step}
        )
        self.analytics.track_user_action(action)
        
    def track_demo_completion(self, user_id: int, session_id: str, 
                            scenario_id: str, scenario_name: str,
                            steps_completed: int, total_steps: int,
                            duration_minutes: float, satisfaction: Optional[int] = None):
        """Track demo scenario completion"""
        completion = DemoCompletion(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            user_id=user_id,
            start_time=datetime.now() - timedelta(minutes=duration_minutes),
            end_time=datetime.now(),
            completion_status="completed",
            steps_completed=steps_completed,
            total_steps=total_steps,
            total_duration_minutes=duration_minutes,
            user_satisfaction=satisfaction
        )
        self.analytics.track_demo_completion(completion)

if __name__ == "__main__":
    # Test analytics system
    analytics = DemoAnalyticsManager("test_analytics.db")
    tracker = DemoTracker(analytics)
    
    print("Demo Analytics System Test")
    print("=" * 30)
    
    # Track some sample actions
    tracker.track_page_view(1, "session_123", "patient_dashboard", 500)
    tracker.track_api_call(1, "session_123", "/api/vitals", 200, True)
    tracker.track_scenario_step(1, "session_123", "Review glucose data", True)
    
    # Track system metrics
    analytics.track_system_metric(SystemMetric(
        timestamp=datetime.now(),
        metric_type="response_time",
        value=250.5,
        unit="ms",
        component="api"
    ))
    
    # Get analytics
    user_data = analytics.get_user_analytics(user_id=1, days=1)
    system_data = analytics.get_system_performance(hours=1)
    
    print(f"User Actions: {user_data['total_actions']}")
    print(f"Avg Response Time: {user_data['average_response_time_ms']}ms")
    print(f"Success Rate: {user_data['success_rate']:.1f}%")