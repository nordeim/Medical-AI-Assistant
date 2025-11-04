"""
Health Check and Uptime Monitoring System
Production-grade monitoring with SLA tracking for healthcare applications
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque
import logging

from config.support_config import SupportConfig, PriorityLevel

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    MAINTENANCE = "maintenance"

class ComponentType(Enum):
    API_ENDPOINT = "api_endpoint"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    FILE_STORAGE = "file_storage"
    EXTERNAL_INTEGRATION = "external_integration"
    MEDICAL_DEVICE = "medical_device"
    EHR_SYSTEM = "ehr_system"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    component_id: str
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class ComponentHealth:
    """Overall health status of a system component"""
    component_id: str
    component_name: str
    component_type: ComponentType
    current_status: HealthStatus
    last_check: datetime
    uptime_percentage: float
    avg_response_time: float
    sla_target: float
    sla_compliance: bool
    incident_count_24h: int
    maintenance_mode: bool
    health_history: deque

@dataclass
class UptimeMetrics:
    """Uptime metrics for a time period"""
    component_id: str
    period_start: datetime
    period_end: datetime
    total_checks: int
    successful_checks: int
    uptime_percentage: float
    downtime_duration_seconds: int
    incident_count: int
    mttr_minutes: float  # Mean Time To Recovery
    availability_score: float

@dataclass
class HealthAlert:
    """Health monitoring alert"""
    id: str
    component_id: str
    severity: AlertSeverity
    title: str
    description: str
    triggered_at: datetime
    acknowledged: bool
    resolved_at: Optional[datetime]
    escalation_level: int
    notifications_sent: List[str]

class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self):
        self.components: Dict[str, ComponentHealth] = {}
        self.health_checks: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        self.alerts: Dict[str, HealthAlert] = {}
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.performance_baselines = {}
        self.circuit_breaker_states = {}
        
        # Load configuration
        self.config = SupportConfig.MONITORING_CONFIG
        
        # Start background monitoring tasks
        self._start_monitoring_tasks()
    
    def register_component(
        self,
        component_id: str,
        component_name: str,
        component_type: ComponentType,
        health_check_function,
        sla_target: float = 99.9,
        check_interval: int = 60
    ) -> None:
        """Register a component for health monitoring"""
        
        component = ComponentHealth(
            component_id=component_id,
            component_name=component_name,
            component_type=component_type,
            current_status=HealthStatus.MAINTENANCE,
            last_check=datetime.now(),
            uptime_percentage=100.0,
            avg_response_time=0.0,
            sla_target=sla_target,
            sla_compliance=True,
            incident_count_24h=0,
            maintenance_mode=False,
            health_history=deque(maxlen=1000)
        )
        
        self.components[component_id] = component
        self.performance_baselines[component_id] = {
            "avg_response_time": 0.0,
            "max_response_time": 1000.0,
            "error_rate": 0.0
        }
        
        # Start monitoring task for this component
        if component_id not in self.monitoring_tasks:
            task = asyncio.create_task(
                self._monitor_component(component_id, health_check_function, check_interval)
            )
            self.monitoring_tasks[component_id] = task
        
        logger.info(f"Registered component {component_name} ({component_id}) for monitoring")
    
    async def perform_health_check(self, component_id: str) -> HealthCheckResult:
        """Perform a health check on a specific component"""
        
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not registered")
        
        component = self.components[component_id]
        start_time = time.time()
        
        try:
            # Perform the actual health check
            # This would call the registered health check function
            status, details = await self._execute_health_check(component_id)
            
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Apply circuit breaker logic
            if self._should_trip_circuit_breaker(component_id, status):
                self._trip_circuit_breaker(component_id)
                status = HealthStatus.CRITICAL
            
            # Update performance baseline
            self._update_performance_baseline(component_id, response_time, status == HealthStatus.HEALTHY)
            
            result = HealthCheckResult(
                component_id=component_id,
                component_name=component.component_name,
                component_type=component.component_type,
                status=status,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                component_id=component_id,
                component_name=component.component_name,
                component_type=component.component_type,
                status=HealthStatus.CRITICAL,
                response_time_ms=response_time,
                timestamp=datetime.now(),
                error_message=str(e)
            )
        
        # Update component status
        await self._update_component_status(component_id, result)
        
        # Store result
        self.health_checks[component_id].append(result)
        
        # Check for alerts
        await self._check_alert_conditions(component_id, result)
        
        return result
    
    async def get_component_health(self, component_id: str) -> ComponentHealth:
        """Get current health status of a component"""
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not registered")
        
        return self.components[component_id]
    
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system health overview"""
        
        total_components = len(self.components)
        if total_components == 0:
            return {"status": "no_components", "message": "No components registered for monitoring"}
        
        healthy_count = sum(1 for c in self.components.values() if c.current_status == HealthStatus.HEALTHY)
        warning_count = sum(1 for c in self.components.values() if c.current_status == HealthStatus.WARNING)
        critical_count = sum(1 for c in self.components.values() if c.current_status == HealthStatus.CRITICAL)
        down_count = sum(1 for c in self.components.values() if c.current_status == HealthStatus.DOWN)
        
        overall_health_score = (healthy_count / total_components) * 100
        
        # Calculate overall SLA compliance
        sla_compliant_components = sum(1 for c in self.components.values() if c.sla_compliance)
        overall_sla_compliance = (sla_compliant_components / total_components) * 100
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self._get_overall_status(healthy_count, warning_count, critical_count, down_count),
            "health_score": overall_health_score,
            "sla_compliance": overall_sla_compliance,
            "component_breakdown": {
                "total": total_components,
                "healthy": healthy_count,
                "warning": warning_count,
                "critical": critical_count,
                "down": down_count
            },
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved_at]),
            "system_uptime": self._calculate_system_uptime()
        }
    
    async def get_uptime_metrics(
        self,
        component_id: str,
        hours: int = 24
    ) -> UptimeMetrics:
        """Get uptime metrics for a component"""
        
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not registered")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get health checks in the time period
        checks = [
            check for check in self.health_checks[component_id]
            if start_time <= check.timestamp <= end_time
        ]
        
        total_checks = len(checks)
        successful_checks = sum(1 for check in checks if check.status in [HealthStatus.HEALTHY, HealthStatus.WARNING])
        
        uptime_percentage = (successful_checks / total_checks * 100) if total_checks > 0 else 100
        
        # Calculate downtime
        downtime_seconds = 0
        incident_count = 0
        
        if checks:
            # Calculate total downtime
            current_period_start = None
            for check in sorted(checks, key=lambda x: x.timestamp):
                if check.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                    if current_period_start is None:
                        current_period_start = check.timestamp
                elif current_period_start is not None:
                    downtime_seconds += (check.timestamp - current_period_start).total_seconds()
                    incident_count += 1
                    current_period_start = None
            
            # Handle ongoing outage
            if current_period_start is not None:
                downtime_seconds += (end_time - current_period_start).total_seconds()
                incident_count += 1
        
        # Calculate MTTR (Mean Time To Recovery)
        mttr_minutes = (downtime_seconds / incident_count / 60) if incident_count > 0 else 0
        
        # Calculate availability score
        availability_score = self._calculate_availability_score(uptime_percentage, incident_count, mttr_minutes)
        
        return UptimeMetrics(
            component_id=component_id,
            period_start=start_time,
            period_end=end_time,
            total_checks=total_checks,
            successful_checks=successful_checks,
            uptime_percentage=uptime_percentage,
            downtime_duration_seconds=int(downtime_seconds),
            incident_count=incident_count,
            mttr_minutes=mttr_minutes,
            availability_score=availability_score
        )
    
    async def create_custom_alert(
        self,
        component_id: str,
        severity: AlertSeverity,
        title: str,
        description: str
    ) -> HealthAlert:
        """Create a custom health alert"""
        
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{component_id}"
        
        alert = HealthAlert(
            id=alert_id,
            component_id=component_id,
            severity=severity,
            title=title,
            description=description,
            triggered_at=datetime.now(),
            acknowledged=False,
            resolved_at=None,
            escalation_level=0,
            notifications_sent=[]
        )
        
        self.alerts[alert_id] = alert
        
        # Send notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(f"Custom alert created: {alert_id} - {title}")
        return alert
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> None:
        """Acknowledge an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
    
    async def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> None:
        """Resolve an alert"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved_at = datetime.now()
            
            logger.info(f"Alert {alert_id} resolved by {resolved_by}: {resolution_notes}")
    
    async def generate_health_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive health monitoring report"""
        
        report_time = datetime.now()
        start_time = report_time - timedelta(hours=hours)
        
        # System overview
        system_overview = await self.get_system_overview()
        
        # Component details
        component_reports = {}
        for component_id, component in self.components.items():
            uptime_metrics = await self.get_uptime_metrics(component_id, hours)
            recent_checks = [
                check for check in self.health_checks[component_id]
                if start_time <= check.timestamp <= report_time
            ]
            
            component_reports[component_id] = {
                "component_info": {
                    "name": component.component_name,
                    "type": component.component_type.value,
                    "sla_target": component.sla_target
                },
                "uptime_metrics": asdict(uptime_metrics),
                "recent_performance": {
                    "avg_response_time": statistics.mean([c.response_time_ms for c in recent_checks]) if recent_checks else 0,
                    "max_response_time": max([c.response_time_ms for c in recent_checks]) if recent_checks else 0,
                    "error_rate": sum(1 for c in recent_checks if c.status == HealthStatus.CRITICAL) / len(recent_checks) * 100 if recent_checks else 0
                },
                "status_history": [asdict(check) for check in recent_checks[-20:]]  # Last 20 checks
            }
        
        # Alert summary
        recent_alerts = [
            alert for alert in self.alerts.values()
            if start_time <= alert.triggered_at <= report_time
        ]
        
        alert_summary = {
            "total_alerts": len(recent_alerts),
            "by_severity": {
                severity.value: len([a for a in recent_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            "resolved_alerts": len([a for a in recent_alerts if a.resolved_at]),
            "pending_alerts": len([a for a in recent_alerts if not a.resolved_at])
        }
        
        return {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": report_time.isoformat(),
                "duration_hours": hours
            },
            "system_overview": system_overview,
            "component_reports": component_reports,
            "alert_summary": alert_summary,
            "recommendations": self._generate_health_recommendations(component_reports, alert_summary)
        }
    
    def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks"""
        # This would start periodic health checks, cleanup tasks, etc.
        logger.info("Started health monitoring system")
    
    async def _monitor_component(self, component_id: str, health_check_func, interval: int) -> None:
        """Background monitoring task for a component"""
        while True:
            try:
                await self.perform_health_check(component_id)
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring task for {component_id}: {e}")
                await asyncio.sleep(interval)
    
    async def _execute_health_check(self, component_id: str) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Execute the actual health check for a component"""
        # This is a placeholder - in real implementation, this would call
        # the registered health check function for the component
        
        component = self.components[component_id]
        
        if component.component_type == ComponentType.API_ENDPOINT:
            return await self._check_api_endpoint(component_id)
        elif component.component_type == ComponentType.DATABASE:
            return await self._check_database(component_id)
        elif component.component_type == ComponentType.EXTERNAL_INTEGRATION:
            return await self._check_external_integration(component_id)
        else:
            return HealthStatus.HEALTHY, {"message": "Component health check not implemented"}
    
    async def _check_api_endpoint(self, component_id: str) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check health of API endpoint"""
        # Placeholder implementation
        # In real implementation, this would make HTTP requests to the API
        import random
        
        if random.random() < 0.05:  # 5% chance of failure for demo
            return HealthStatus.CRITICAL, {"error": "Connection timeout"}
        
        response_time = random.uniform(50, 200)  # 50-200ms response time
        return HealthStatus.HEALTHY, {"response_time_ms": response_time}
    
    async def _check_database(self, component_id: str) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check health of database connection"""
        # Placeholder implementation
        # In real implementation, this would test database connectivity and query performance
        import random
        
        if random.random() < 0.02:  # 2% chance of failure for demo
            return HealthStatus.CRITICAL, {"error": "Database connection failed"}
        
        query_time = random.uniform(5, 50)  # 5-50ms query time
        return HealthStatus.HEALTHY, {"query_time_ms": query_time}
    
    async def _check_external_integration(self, component_id: str) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Check health of external integration"""
        # Placeholder implementation for EHR systems, etc.
        import random
        
        if random.random() < 0.03:  # 3% chance of failure for demo
            return HealthStatus.WARNING, {"warning": "High latency detected"}
        
        return HealthStatus.HEALTHY, {"status": "integration healthy"}
    
    def _should_trip_circuit_breaker(self, component_id: str, status: HealthStatus) -> bool:
        """Determine if circuit breaker should trip"""
        recent_checks = self.health_checks[component_id][-5:]  # Last 5 checks
        
        if len(recent_checks) < 5:
            return False
        
        failed_checks = sum(1 for check in recent_checks 
                          if check.status in [HealthStatus.CRITICAL, HealthStatus.DOWN])
        
        return failed_checks >= 3  # Trip if 3 out of last 5 checks failed
    
    def _trip_circuit_breaker(self, component_id: str) -> None:
        """Trip the circuit breaker for a component"""
        self.circuit_breaker_states[component_id] = {
            "state": "open",
            "opened_at": datetime.now(),
            "failure_count": 0
        }
        logger.warning(f"Circuit breaker tripped for component {component_id}")
    
    def _update_performance_baseline(self, component_id: str, response_time: float, is_healthy: bool) -> None:
        """Update performance baseline for a component"""
        baseline = self.performance_baselines[component_id]
        
        # Update moving averages (simplified)
        if baseline["avg_response_time"] == 0:
            baseline["avg_response_time"] = response_time
        else:
            baseline["avg_response_time"] = (baseline["avg_response_time"] + response_time) / 2
        
        baseline["max_response_time"] = max(baseline["max_response_time"], response_time)
        
        if not is_healthy:
            baseline["error_rate"] += 1
    
    async def _update_component_status(self, component_id: str, check_result: HealthCheckResult) -> None:
        """Update component status based on health check result"""
        component = self.components[component_id]
        component.last_check = check_result.timestamp
        component.current_status = check_result.status
        
        # Update uptime percentage
        component.uptime_percentage = self._calculate_uptime_percentage(component_id)
        
        # Update average response time
        recent_checks = self.health_checks[component_id][-10:]  # Last 10 checks
        component.avg_response_time = statistics.mean([c.response_time_ms for c in recent_checks])
        
        # Check SLA compliance
        component.sla_compliance = component.uptime_percentage >= component.sla_target
        
        # Add to history
        component.health_history.append(check_result)
    
    def _calculate_uptime_percentage(self, component_id: str) -> float:
        """Calculate uptime percentage for component"""
        checks = self.health_checks[component_id][-100:]  # Last 100 checks
        
        if not checks:
            return 100.0
        
        healthy_checks = sum(1 for check in checks 
                           if check.status in [HealthStatus.HEALTHY, HealthStatus.WARNING])
        
        return (healthy_checks / len(checks)) * 100
    
    async def _check_alert_conditions(self, component_id: str, check_result: HealthCheckResult) -> None:
        """Check if alert conditions are met"""
        component = self.components[component_id]
        
        # Check for critical status
        if check_result.status == HealthStatus.CRITICAL:
            await self.create_custom_alert(
                component_id,
                AlertSeverity.CRITICAL,
                f"Critical status for {component.component_name}",
                f"Component is in critical state. Response time: {check_result.response_time_ms}ms"
            )
        
        # Check for high response time
        if check_result.response_time_ms > component.sla_target * 10:  # Arbitrary threshold
            await self.create_custom_alert(
                component_id,
                AlertSeverity.WARNING,
                f"High response time for {component.component_name}",
                f"Response time exceeded threshold: {check_result.response_time_ms}ms"
            )
    
    async def _send_alert_notifications(self, alert: HealthAlert) -> None:
        """Send alert notifications (placeholder implementation)"""
        # In production, this would send notifications via:
        # - Email to operations team
        # - SMS to on-call engineers
        # - Slack/Teams notifications
        # - PagerDuty integration
        # - Medical device integration for critical alerts
        
        alert.notifications_sent.append("email")
        alert.notifications_sent.append("slack")
        
        if alert.severity == AlertSeverity.EMERGENCY:
            alert.notifications_sent.append("sms")
            alert.notifications_sent.append("phone_call")
    
    def _get_overall_status(self, healthy: int, warning: int, critical: int, down: int) -> str:
        """Determine overall system status"""
        if down > 0:
            return "critical"
        elif critical > 0:
            return "critical"
        elif warning > 0:
            return "warning"
        elif healthy > 0:
            return "healthy"
        else:
            return "unknown"
    
    def _calculate_system_uptime(self) -> float:
        """Calculate overall system uptime"""
        if not self.components:
            return 100.0
        
        total_uptime = sum(component.uptime_percentage for component in self.components.values())
        return total_uptime / len(self.components)
    
    def _calculate_availability_score(self, uptime_percentage: float, incident_count: int, mttr_minutes: float) -> float:
        """Calculate overall availability score"""
        # Weighted score based on uptime, incidents, and MTTR
        uptime_score = uptime_percentage * 0.6  # 60% weight
        incident_penalty = max(0, 100 - (incident_count * 5))  # Penalize incidents
        mttr_bonus = max(0, 100 - (mttr_minutes * 2))  # Bonus for quick recovery
        
        return (uptime_score + incident_penalty * 0.3 + mttr_bonus * 0.1) / 100 * 100
    
    def _generate_health_recommendations(self, component_reports: Dict, alert_summary: Dict) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []
        
        # Check for components with low uptime
        for component_id, report in component_reports.items():
            if report["uptime_metrics"]["uptime_percentage"] < 99.0:
                recommendations.append(
                    f"Component {report['component_info']['name']} has low uptime "
                    f"({report['uptime_metrics']['uptime_percentage']:.1f}%). "
                    f"Consider investigating reliability issues."
                )
        
        # Check for high incident count
        if alert_summary["total_alerts"] > 50:
            recommendations.append(
                f"High number of alerts ({alert_summary['total_alerts']}). "
                f"Review system stability and consider preventive maintenance."
            )
        
        # Check for slow response times
        slow_components = [
            component_id for component_id, report in component_reports.items()
            if report["recent_performance"]["avg_response_time"] > 1000  # 1 second
        ]
        
        if slow_components:
            recommendations.append(
                f"Components with slow response times detected: {', '.join(slow_components)}. "
                f"Consider performance optimization."
            )
        
        return recommendations

# Global health monitor instance
health_monitor = HealthMonitor()

# Example usage and testing functions
async def setup_sample_monitoring():
    """Set up sample components for monitoring"""
    
    # Register API endpoint
    health_monitor.register_component(
        "api_core",
        "Core Medical AI API",
        ComponentType.API_ENDPOINT,
        health_check_function=None,  # Would be actual health check function
        sla_target=99.9,
        check_interval=30
    )
    
    # Register database
    health_monitor.register_component(
        "db_medical",
        "Medical Records Database",
        ComponentType.DATABASE,
        health_check_function=None,
        sla_target=99.99,  # Higher SLA for medical data
        check_interval=60
    )
    
    # Register EHR integration
    health_monitor.register_component(
        "ehr_epic",
        "Epic EHR Integration",
        ComponentType.EXTERNAL_INTEGRATION,
        health_check_function=None,
        sla_target=99.95,
        check_interval=120
    )
    
    print("Sample monitoring components registered")
    
    # Perform some health checks
    for component_id in ["api_core", "db_medical", "ehr_epic"]:
        result = await health_monitor.perform_health_check(component_id)
        print(f"Health check for {component_id}: {result.status.value} ({result.response_time_ms:.1f}ms)")

if __name__ == "__main__":
    asyncio.run(setup_sample_monitoring())