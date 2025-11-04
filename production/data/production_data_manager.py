"""
Production Data Management Orchestrator
Coordinates all data management and analytics systems for healthcare applications
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import all data management components
from production.data.config.data_config import (
    PRODUCTION_CONFIG, DataSourceConfig, ETLConfig, AnalyticsConfig, RetentionConfig
)
from production.data.etl.medical_etl_pipeline import MedicalETLPipeline, create_etl_pipeline
from production.data.quality.quality_monitor import MedicalDataQualityMonitor, create_quality_monitor
from production.data.analytics.healthcare_analytics import HealthcareAnalyticsEngine, create_analytics_engine
from production.data.clinical.outcome_tracker import ClinicalOutcomeTracker, create_outcome_tracker
from production.data.retention.retention_manager import DataRetentionManager, create_retention_manager
from production.data.predictive.analytics_engine import PredictiveAnalyticsEngine, create_analytics_engine as create_predictive_engine
from production.data.export.export_manager import DataExportManager, create_export_manager

class SystemStatus(Enum):
    """Overall system status"""
    INITIALIZING = "initializing"
    OPERATIONAL = "operational"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    DEGRADED = "degraded"

class DataPipelineStatus(Enum):
    """Data pipeline status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"

@dataclass
class PipelineOrchestration:
    """Pipeline orchestration configuration"""
    pipeline_id: str
    pipeline_name: str
    component_order: List[str]  # etl -> quality -> analytics -> outcomes -> retention
    schedule: str  # cron-like schedule
    enabled: bool = True
    error_handling: str = "continue"  # continue, stop, retry
    notification_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health metrics"""
    component_status: Dict[str, SystemStatus]
    last_update: datetime
    data_flow_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    alert_count: int = 0
    uptime_percentage: float = 100.0

class ProductionDataManager:
    """Production Data Management Orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize all subsystems
        self.etl_pipeline = None
        self.quality_monitor = None
        self.analytics_engine = None
        self.outcome_tracker = None
        self.retention_manager = None
        self.predictive_engine = None
        self.export_manager = None
        
        # System state
        self.system_status = SystemStatus.INITIALIZING
        self.system_health = SystemHealth(
            component_status={},
            last_update=datetime.now(),
            data_flow_metrics={},
            performance_metrics={}
        )
        
        # Pipeline orchestration
        self.pipelines = {}
        self.active_executions = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup main system logging"""
        logger = logging.getLogger("production_data_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def initialize_system(self) -> None:
        """Initialize all data management components"""
        try:
            self.logger.info("Initializing Production Data Management System")
            
            # Initialize ETL pipeline
            self.etl_pipeline = create_etl_pipeline(PRODUCTION_CONFIG)
            await self.etl_pipeline.initialize_connections()
            self.system_health.component_status["etl"] = SystemStatus.OPERATIONAL
            
            # Initialize quality monitor
            self.quality_monitor = create_quality_monitor({
                "alert_thresholds": {
                    "critical_score": 75,
                    "warning_score": 85
                }
            })
            self.system_health.component_status["quality"] = SystemStatus.OPERATIONAL
            
            # Initialize analytics engine
            self.analytics_engine = create_analytics_engine({
                "analytics_warehouse_connection": "sqlite:///analytics_warehouse.db"
            })
            await self.analytics_engine.initialize_analytics()
            self.system_health.component_status["analytics"] = SystemStatus.OPERATIONAL
            
            # Initialize outcome tracker
            self.outcome_tracker = create_outcome_tracker({
                "risk_adjustment_enabled": True,
                "benchmark_comparison_enabled": True
            })
            await self.outcome_tracker.initialize_outcome_system()
            self.system_health.component_status["outcomes"] = SystemStatus.OPERATIONAL
            
            # Initialize retention manager
            self.retention_manager = create_retention_manager({
                "archive_base_path": "./archives",
                "metadata_db_path": "retention_metadata.db",
                "encryption_required": True
            })
            await self.retention_manager.initialize_retention_system()
            self.system_health.component_status["retention"] = SystemStatus.OPERATIONAL
            
            # Initialize predictive engine
            self.predictive_engine = create_predictive_engine({
                "model_storage_path": "./models"
            })
            await self.predictive_engine.initialize_analytics_engine()
            self.system_health.component_status["predictive"] = SystemStatus.OPERATIONAL
            
            # Initialize export manager
            self.export_manager = create_export_manager({
                "export_directory": "./exports",
                "report_directory": "./reports",
                "encryption_required": True
            })
            await self.export_manager.initialize_export_system()
            self.system_health.component_status["export"] = SystemStatus.OPERATIONAL
            
            # Initialize pipeline orchestrations
            await self._initialize_pipeline_orchestrations()
            
            # Update system status
            self.system_status = SystemStatus.OPERATIONAL
            self.system_health.last_update = datetime.now()
            
            self.logger.info("Production Data Management System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            self.system_status = SystemStatus.ERROR
            raise
    
    async def _initialize_pipeline_orchestrations(self) -> None:
        """Initialize pipeline orchestrations"""
        
        self.pipelines = {
            # Patient Data Pipeline
            "patient_data_pipeline": PipelineOrchestration(
                pipeline_id="pipeline_001",
                pipeline_name="Patient Data Processing Pipeline",
                component_order=["etl", "quality", "analytics"],
                schedule="0 */6 * * *",  # Every 6 hours
                notification_settings={
                    "email": ["data-team@hospital.org"],
                    "slack": "#data-alerts",
                    "failure_threshold": 3
                }
            ),
            
            # Clinical Outcomes Pipeline
            "clinical_outcomes_pipeline": PipelineOrchestration(
                pipeline_id="pipeline_002",
                pipeline_name="Clinical Outcomes Analytics Pipeline",
                component_order=["etl", "quality", "outcomes", "analytics"],
                schedule="0 2 * * *",  # Daily at 2 AM
                notification_settings={
                    "email": ["clinical-team@hospital.org"],
                    "dashboard": True,
                    "failure_threshold": 2
                }
            ),
            
            # Predictive Analytics Pipeline
            "predictive_analytics_pipeline": PipelineOrchestration(
                pipeline_id="pipeline_003",
                pipeline_name="Predictive Analytics Pipeline",
                component_order=["etl", "quality", "predictive", "analytics"],
                schedule="0 1 * * *",  # Daily at 1 AM
                notification_settings={
                    "email": ["ai-team@hospital.org"],
                    "ml_ops_channel": True,
                    "failure_threshold": 1
                }
            ),
            
            # Data Retention Pipeline
            "data_retention_pipeline": PipelineOrchestration(
                pipeline_id="pipeline_004",
                pipeline_name="Data Retention and Archive Pipeline",
                component_order=["etl", "retention"],
                schedule="0 3 1 * *",  # Monthly on 1st at 3 AM
                notification_settings={
                    "email": ["compliance@hospital.org"],
                    "legal_team": True,
                    "failure_threshold": 1
                }
            ),
            
            # Real-time Monitoring Pipeline
            "monitoring_pipeline": PipelineOrchestration(
                pipeline_id="pipeline_005",
                pipeline_name="Real-time Monitoring Pipeline",
                component_order=["quality", "analytics"],
                schedule="*/15 * * * *",  # Every 15 minutes
                notification_settings={
                    "alerts": True,
                    "dashboard": True,
                    "failure_threshold": 5
                }
            )
        }
    
    async def execute_pipeline(self, pipeline_id: str, 
                             execution_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a complete data pipeline"""
        
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_id} not found")
        
        pipeline_config = self.pipelines[pipeline_id]
        execution_id = f"EXEC_{pipeline_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"Starting pipeline execution: {pipeline_id}")
            
            execution_result = {
                "execution_id": execution_id,
                "pipeline_id": pipeline_id,
                "started_at": datetime.now(),
                "component_results": {},
                "status": DataPipelineStatus.RUNNING,
                "errors": []
            }
            
            # Execute components in order
            for component_name in pipeline_config.component_order:
                try:
                    self.logger.info(f"Executing component: {component_name}")
                    
                    if component_name == "etl":
                        result = await self._execute_etl_component(execution_config)
                    elif component_name == "quality":
                        result = await self._execute_quality_component(execution_config)
                    elif component_name == "analytics":
                        result = await self._execute_analytics_component(execution_config)
                    elif component_name == "outcomes":
                        result = await self._execute_outcomes_component(execution_config)
                    elif component_name == "retention":
                        result = await self._execute_retention_component(execution_config)
                    elif component_name == "predictive":
                        result = await self._execute_predictive_component(execution_config)
                    else:
                        raise ValueError(f"Unknown component: {component_name}")
                    
                    execution_result["component_results"][component_name] = result
                    
                    # Check if component failed and pipeline should stop
                    if result.get("status") == "failed" and pipeline_config.error_handling == "stop":
                        execution_result["status"] = DataPipelineStatus.FAILED
                        execution_result["errors"].append(f"Component {component_name} failed, pipeline stopped")
                        break
                
                except Exception as e:
                    error_msg = f"Component {component_name} execution failed: {str(e)}"
                    execution_result["errors"].append(error_msg)
                    self.logger.error(error_msg)
                    
                    if pipeline_config.error_handling == "stop":
                        execution_result["status"] = DataPipelineStatus.FAILED
                        break
                    
                    # Continue with next component
            
            # Determine final status
            if not execution_result["errors"]:
                execution_result["status"] = DataPipelineStatus.COMPLETED
            elif execution_result["component_results"]:
                execution_result["status"] = DataPipelineStatus.COMPLETED
            else:
                execution_result["status"] = DataPipelineStatus.FAILED
            
            execution_result["completed_at"] = datetime.now()
            execution_result["duration_seconds"] = (
                execution_result["completed_at"] - execution_result["started_at"]
            ).total_seconds()
            
            # Store execution result
            self.active_executions[execution_id] = execution_result
            
            # Update system health metrics
            await self._update_health_metrics(execution_result)
            
            # Send notifications if configured
            await self._send_pipeline_notifications(pipeline_config, execution_result)
            
            self.logger.info(f"Pipeline execution completed: {pipeline_id}")
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {pipeline_id} - {str(e)}")
            raise
    
    async def _execute_etl_component(self, execution_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute ETL component"""
        try:
            # Run incremental ETL
            etl_config = execution_config.get("etl", {}) if execution_config else {}
            
            etl_config = {
                "sources": [
                    {
                        "name": "Epic EHR",
                        "table": "patients",
                        "target_table": "analytics_patients",
                        "transformations": [
                            {"type": "anonymization", "fields": ["name", "ssn"]},
                            {"type": "standardization", "fields": ["diagnosis_codes"]}
                        ]
                    }
                ]
            }
            
            jobs = await self.etl_pipeline.run_incremental_etl(etl_config)
            
            return {
                "status": "completed",
                "jobs_executed": len(jobs),
                "total_records_processed": sum(job.records_processed for job in jobs),
                "success_rate": sum(1 for job in jobs if job.status.value == "completed") / len(jobs)
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_quality_component(self, execution_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute quality monitoring component"""
        try:
            # Run quality checks on key tables
            quality_results = []
            
            tables_to_check = ["patients", "encounters", "medications", "lab_results"]
            
            for table_name in tables_to_check:
                # Generate sample data for quality check
                sample_data = pd.DataFrame({
                    "patient_id": [f"PAT_{i:04d}" for i in range(1, 101)],
                    "birth_date": pd.date_range("1980-01-01", "2000-01-01", periods=100),
                    "age": np.random.randint(18, 90, 100),
                    "gender": np.random.choice(["M", "F"], 100),
                    "blood_pressure_systolic": np.random.randint(90, 180, 100)
                })
                
                report = await self.quality_monitor.perform_quality_check(table_name, sample_data)
                quality_results.append({
                    "table": table_name,
                    "overall_score": report.overall_score,
                    "quality_level": report.quality_level.value,
                    "critical_issues": len(report.critical_issues)
                })
            
            return {
                "status": "completed",
                "tables_checked": len(quality_results),
                "average_quality_score": np.mean([r["overall_score"] for r in quality_results]),
                "tables_below_threshold": len([r for r in quality_results if r["overall_score"] < 80])
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_analytics_component(self, execution_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute analytics component"""
        try:
            # Calculate all KPIs
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            kpis = await self.analytics_engine.calculate_all_kpis((start_date, end_date))
            
            # Generate dashboard report
            report = await self.analytics_engine.generate_dashboard_report("executive_dashboard")
            
            return {
                "status": "completed",
                "kpis_calculated": len(kpis),
                "dashboard_reports_generated": 1,
                "alerts_generated": len(report.get("alerts", [])),
                "recommendations_generated": len(report.get("recommendations", []))
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_outcomes_component(self, execution_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute clinical outcomes component"""
        try:
            # Calculate clinical outcomes
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            outcomes = await self.outcome_tracker.calculate_clinical_outcomes(start_date, end_date)
            
            # Generate outcome report
            report = await self.outcome_tracker.generate_outcome_report(start_date, end_date)
            
            return {
                "status": "completed",
                "outcomes_calculated": len(outcomes),
                "reports_generated": 1,
                "recommendations": len(report.quality_improvement_recommendations),
                "statistical_analysis": len(report.statistical_analysis)
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_retention_component(self, execution_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute data retention component"""
        try:
            # Execute retention cleanup
            cleanup_results = await self.retention_manager.execute_retention_cleanup()
            
            return {
                "status": "completed",
                "cleanup_executed": True,
                "records_archived": cleanup_results.get("records_archived", 0),
                "records_deleted": cleanup_results.get("records_deleted", 0),
                "policies_processed": len(cleanup_results.get("policies_processed", []))
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _execute_predictive_component(self, execution_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute predictive analytics component"""
        try:
            # Make sample predictions
            patient_data = {
                "age": 75,
                "gender": "M",
                "prior_admissions": 3,
                "charlson_comorbidity_index": 4,
                "length_of_stay": 5,
                "medication_count": 8,
                "discharge_disposition": "SNF",
                "primary_diagnosis": "heart failure"
            }
            
            # Make readmission prediction
            prediction = await self.predictive_engine.make_prediction("readmission_model_v1", patient_data)
            
            # Get model insights
            insights = await self.predictive_engine.get_model_insights("readmission_model_v1")
            
            return {
                "status": "completed",
                "predictions_made": 1,
                "models_used": 1,
                "insights_generated": len(insights.get("recommendations", [])),
                "prediction_confidence": prediction.confidence_score
            }
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    
    async def _update_health_metrics(self, execution_result: Dict[str, Any]) -> None:
        """Update system health metrics"""
        try:
            # Update data flow metrics
            self.system_health.data_flow_metrics.update({
                "last_pipeline_execution": execution_result["started_at"].isoformat(),
                "last_pipeline_status": execution_result["status"].value,
                "last_execution_duration": execution_result.get("duration_seconds", 0)
            })
            
            # Calculate success rate
            successful_components = sum(
                1 for result in execution_result["component_results"].values()
                if result.get("status") == "completed"
            )
            total_components = len(execution_result["component_results"])
            
            success_rate = successful_components / total_components if total_components > 0 else 0
            
            # Update performance metrics
            self.system_health.performance_metrics.update({
                "component_success_rate": success_rate,
                "total_errors": len(execution_result.get("errors", [])),
                "pipeline_efficiency": 1.0 - (len(execution_result.get("errors", [])) / max(total_components, 1))
            })
            
            # Update component statuses
            for component_name, result in execution_result["component_results"].items():
                if result.get("status") == "completed":
                    self.system_health.component_status[component_name] = SystemStatus.OPERATIONAL
                else:
                    self.system_health.component_status[component_name] = SystemStatus.ERROR
            
            # Count alerts
            self.system_health.alert_count = len(execution_result.get("errors", []))
            
            self.system_health.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update health metrics: {str(e)}")
    
    async def _send_pipeline_notifications(self, pipeline_config: PipelineOrchestration, 
                                         execution_result: Dict[str, Any]) -> None:
        """Send pipeline execution notifications"""
        try:
            notifications = pipeline_config.notification_settings
            
            # Check if notification is needed
            if execution_result["status"] == DataPipelineStatus.FAILED:
                # Send failure notification
                if "email" in notifications:
                    for email in notifications["email"]:
                        await self._send_email_notification(email, execution_result)
                
                if "slack" in notifications:
                    await self._send_slack_notification(notifications["slack"], execution_result)
            
            elif execution_result["status"] == DataPipelineStatus.COMPLETED and notifications.get("dashboard"):
                # Send success notification to dashboard
                await self._send_dashboard_notification(execution_result)
                
        except Exception as e:
            self.logger.error(f"Failed to send notifications: {str(e)}")
    
    async def _send_email_notification(self, email: str, execution_result: Dict[str, Any]) -> None:
        """Send email notification (placeholder)"""
        self.logger.info(f"Email notification sent to {email} for execution {execution_result['execution_id']}")
    
    async def _send_slack_notification(self, channel: str, execution_result: Dict[str, Any]) -> None:
        """Send Slack notification (placeholder)"""
        self.logger.info(f"Slack notification sent to {channel} for execution {execution_result['execution_id']}")
    
    async def _send_dashboard_notification(self, execution_result: Dict[str, Any]) -> None:
        """Send dashboard notification (placeholder)"""
        self.logger.info(f"Dashboard notification sent for execution {execution_result['execution_id']}")
    
    async def run_continuous_monitoring(self) -> None:
        """Run continuous system monitoring"""
        self.logger.info("Starting continuous system monitoring")
        
        while True:
            try:
                # Check system health
                await self._check_system_health()
                
                # Update dashboard metrics
                await self._update_dashboard_metrics()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                # Wait for next monitoring cycle (5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Continuous monitoring error: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _check_system_health(self) -> None:
        """Check overall system health"""
        # Check if all components are operational
        operational_count = sum(
            1 for status in self.system_health.component_status.values()
            if status == SystemStatus.OPERATIONAL
        )
        total_components = len(self.system_health.component_status)
        
        if operational_count == total_components:
            self.system_status = SystemStatus.OPERATIONAL
        elif operational_count > total_components * 0.7:
            self.system_status = SystemStatus.DEGRADED
        else:
            self.system_status = SystemStatus.ERROR
        
        self.system_health.last_update = datetime.now()
    
    async def _update_dashboard_metrics(self) -> None:
        """Update dashboard metrics"""
        # Update real-time metrics for dashboards
        
        # Get export status
        export_status = self.export_manager.get_export_status()
        
        # Get retention status
        retention_status = self.retention_manager.get_retention_status()
        
        # Update system health with current metrics
        self.system_health.data_flow_metrics.update({
            "export_operations": export_status,
            "retention_operations": retention_status,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _check_alert_conditions(self) -> None:
        """Check for alert conditions"""
        # Check quality scores
        if hasattr(self, 'quality_monitor') and self.quality_monitor.quality_history:
            latest_report = self.quality_monitor.quality_history[-1]
            if latest_report.overall_score < 75:
                self.logger.warning(f"Alert: Quality score below threshold: {latest_report.overall_score}")
        
        # Check pipeline success rates
        recent_executions = [
            exec_result for exec_result in self.active_executions.values()
            if (datetime.now() - exec_result["started_at"]).days <= 1
        ]
        
        if recent_executions:
            failure_rate = sum(
                1 for exec_result in recent_executions
                if exec_result["status"] == DataPipelineStatus.FAILED
            ) / len(recent_executions)
            
            if failure_rate > 0.2:  # More than 20% failure rate
                self.logger.warning(f"Alert: High pipeline failure rate: {failure_rate:.1%}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_status": self.system_status.value,
            "last_update": self.system_health.last_update.isoformat(),
            "component_status": {
                component: status.value 
                for component, status in self.system_health.component_status.items()
            },
            "data_flow_metrics": self.system_health.data_flow_metrics,
            "performance_metrics": self.system_health.performance_metrics,
            "active_pipelines": len(self.pipelines),
            "recent_executions": len([
                exec_result for exec_result in self.active_executions.values()
                if (datetime.now() - exec_result["started_at"]).days <= 7
            ]),
            "alert_count": self.system_health.alert_count
        }
    
    async def shutdown_system(self) -> None:
        """Gracefully shutdown the system"""
        try:
            self.logger.info("Shutting down Production Data Management System")
            
            # Close connections
            if self.etl_pipeline:
                await self.etl_pipeline.close_connections()
            
            # Clean up resources
            self.active_executions.clear()
            
            self.system_status = SystemStatus.MAINTENANCE
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {str(e)}")

def create_production_data_manager(config: Dict[str, Any] = None) -> ProductionDataManager:
    """Factory function to create production data manager"""
    if config is None:
        config = {
            "environment": "production",
            "log_level": "INFO",
            "monitoring_enabled": True,
            "notification_enabled": True
        }
    
    return ProductionDataManager(config)

# Main execution function
async def main():
    """Main execution function for production data management"""
    try:
        # Create and initialize system
        data_manager = create_production_data_manager()
        await data_manager.initialize_system()
        
        print("Production Data Management System Initialized")
        print("=" * 50)
        
        # Execute sample pipeline
        execution_result = await data_manager.execute_pipeline("patient_data_pipeline")
        
        print(f"Pipeline Execution Results:")
        print(f"Execution ID: {execution_result['execution_id']}")
        print(f"Status: {execution_result['status'].value}")
        print(f"Duration: {execution_result['duration_seconds']:.2f} seconds")
        print(f"Components executed: {len(execution_result['component_results'])}")
        
        # Get system status
        status = data_manager.get_system_status()
        print(f"\nSystem Status: {status['system_status']}")
        print(f"Components operational: {sum(1 for s in status['component_status'].values() if s == 'operational')}/{len(status['component_status'])}")
        
        # Note: In production, you would run continuous monitoring
        # await data_manager.run_continuous_monitoring()
        
    except Exception as e:
        print(f"System execution failed: {str(e)}")
        raise
    finally:
        if 'data_manager' in locals():
            await data_manager.shutdown_system()

if __name__ == "__main__":
    asyncio.run(main())
