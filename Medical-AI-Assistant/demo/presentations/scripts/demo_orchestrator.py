#!/usr/bin/env python3
"""
Demo Automation Orchestrator - Central orchestration system for demo presentations.

This module provides comprehensive demo automation capabilities including:
- Automated demo sequence control
- Multi-stakeholder demo orchestration
- Real-time demo state management
- Integration with all demo subsystems
- Professional demo delivery automation
"""

import asyncio
import json
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Import demo subsystems
from .demo_manager import DemoManager, DemoType, StakeholderType, DemoStatus
from .demo_scenarios import DemoScenarioManager, MedicalScenario
from .demo_recorder import DemoRecorder, RecordingType, RecordingQuality
from .demo_feedback import DemoFeedbackManager, FeedbackType, StakeholderFeedback

class DemoState(Enum):
    """Demo orchestration states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    RECORDING = "recording"
    COLLECTING_FEEDBACK = "collecting_feedback"
    COMPLETING = "completing"
    CLEANUP = "cleanup"

class AutomationLevel(Enum):
    """Automation level options"""
    MANUAL = "manual"
    SEMI_AUTOMATED = "semi_automated"
    FULLY_AUTOMATED = "fully_automated"
    AI_POWERED = "ai_powered"

@dataclass
class DemoConfiguration:
    """Demo configuration structure"""
    demo_id: str
    demo_type: DemoType
    stakeholder_type: StakeholderType
    scenario_sequence: List[str]
    automation_level: AutomationLevel
    recording_enabled: bool
    feedback_collection: bool
    custom_parameters: Dict[str, Any]
    
@dataclass
class DemoExecution:
    """Demo execution tracking"""
    execution_id: str
    configuration: DemoConfiguration
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_state: DemoState = DemoState.IDLE
    current_scenario: Optional[str] = None
    completed_scenarios: List[str] = None
    automation_progress: float = 0.0
    interaction_count: int = 0
    recording_session_id: Optional[str] = None
    feedback_session_id: Optional[str] = None

class DemoOrchestrator:
    """Central demo orchestration system"""
    
    def __init__(
        self,
        demo_manager: Optional[DemoManager] = None,
        scenario_manager: Optional[DemoScenarioManager] = None,
        recorder: Optional[DemoRecorder] = None,
        feedback_manager: Optional[DemoFeedbackManager] = None,
        config_file: str = "orchestrator_config.json"
    ):
        # Initialize subsystems
        self.demo_manager = demo_manager or DemoManager()
        self.scenario_manager = scenario_manager or DemoScenarioManager()
        self.recorder = recorder or DemoRecorder()
        self.feedback_manager = feedback_manager or DemoFeedbackManager()
        
        self.config_file = Path(config_file)
        self.current_execution: Optional[DemoExecution] = None
        self.automation_callbacks: Dict[str, Callable] = {}
        self.demo_states_history: List[Dict] = []
        
        self._load_configuration()
        self.setup_logging()
        self._init_database()
        
    def setup_logging(self):
        """Setup logging for orchestrator"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('demo_orchestrator.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_configuration(self):
        """Load orchestrator configuration"""
        default_config = {
            "automation_settings": {
                "default_automation_level": "semi_automated",
                "auto_record": False,
                "auto_feedback": True,
                "auto_recovery": True,
                "timeout_minutes": 45
            },
            "demo_sequences": {
                "c_suite_quick": ["stemi_pci", "heart_failure", "diabetes_management"],
                "clinical_comprehensive": ["stemi_pci", "stroke_tpa", "breast_cancer", "diabetes_management"],
                "investor_demo": ["stemi_pci", "heart_failure", "breast_cancer", "chest_pain"],
                "regulatory_review": ["stemi_pci", "stroke_tpa", "breast_cancer"]
            },
            "scenario_settings": {
                "default_duration_minutes": 5,
                "max_interaction_time": 3,
                "auto_advance_threshold": 0.8,
                "pause_triggers": ["question_detected", "low_engagement"]
            },
            "integration_settings": {
                "crm_sync": True,
                "slack_notifications": True,
                "calendar_integration": True,
                "analytics_sync": True
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
                self.logger.error(f"Error loading config: {e}")
                self.config = default_config
        else:
            self.config = default_config
            self.save_configuration()
    
    def save_configuration(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def _init_database(self):
        """Initialize orchestration database"""
        self.db_path = "demo_orchestration.db"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Demo executions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_executions (
                execution_id TEXT PRIMARY KEY,
                demo_id TEXT NOT NULL,
                demo_type TEXT NOT NULL,
                stakeholder_type TEXT NOT NULL,
                scenario_sequence TEXT NOT NULL,
                automation_level TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                current_state TEXT NOT NULL,
                current_scenario TEXT,
                completed_scenarios TEXT,
                automation_progress REAL DEFAULT 0.0,
                interaction_count INTEGER DEFAULT 0,
                recording_session_id TEXT,
                feedback_session_id TEXT,
                custom_parameters TEXT,
                execution_summary TEXT
            )
        ''')
        
        # Demo automation events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS automation_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                timestamp TEXT NOT NULL,
                duration_seconds REAL,
                FOREIGN KEY (execution_id) REFERENCES demo_executions (execution_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_demo_configuration(
        self,
        demo_type: DemoType,
        stakeholder_type: StakeholderType,
        scenario_sequence: Optional[List[str]] = None,
        automation_level: AutomationLevel = AutomationLevel.SEMI_AUTOMATED,
        recording_enabled: bool = False,
        feedback_collection: bool = True,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create demo configuration"""
        
        # Use default scenario sequence if not provided
        if not scenario_sequence:
            sequence_key = f"{stakeholder_type.value}_{demo_type.value}"
            scenario_sequence = self.config["demo_sequences"].get(sequence_key, 
                ["stemi_pci", "heart_failure", "diabetes_management"])
        
        # Generate demo ID
        demo_id = f"demo_{int(time.time())}_{demo_type.value}_{stakeholder_type.value}"
        
        configuration = DemoConfiguration(
            demo_id=demo_id,
            demo_type=demo_type,
            stakeholder_type=stakeholder_type,
            scenario_sequence=scenario_sequence,
            automation_level=automation_level,
            recording_enabled=recording_enabled,
            feedback_collection=feedback_collection,
            custom_parameters=custom_parameters or {}
        )
        
        self.logger.info(f"Created demo configuration: {demo_id}")
        return demo_id
    
    def start_automated_demo(self, demo_id: str, presenter_override: Optional[str] = None) -> str:
        """Start automated demo execution"""
        
        # Create execution record
        execution_id = f"exec_{int(time.time())}_{demo_id}"
        
        # For demo purposes, create a simple configuration
        if demo_id == "quick_cardiology":
            config = DemoConfiguration(
                demo_id=demo_id,
                demo_type=DemoType.CARDIOLOGY,
                stakeholder_type=StakeholderType.C_SUITE,
                scenario_sequence=["stemi_pci"],
                automation_level=AutomationLevel.SEMI_AUTOMATED,
                recording_enabled=True,
                feedback_collection=True,
                custom_parameters={}
            )
        else:
            # Default configuration
            config = DemoConfiguration(
                demo_id=demo_id,
                demo_type=DemoType.CARDIOLOGY,
                stakeholder_type=StakeholderType.C_SUITE,
                scenario_sequence=["stemi_pci"],
                automation_level=AutomationLevel.SEMI_AUTOMATED,
                recording_enabled=True,
                feedback_collection=True,
                custom_parameters={}
            )
        
        execution = DemoExecution(
            execution_id=execution_id,
            configuration=config,
            start_time=datetime.now(),
            current_state=DemoState.INITIALIZING,
            completed_scenarios=[]
        )
        
        self.current_execution = execution
        
        # Initialize demo subsystems
        self._initialize_demo_subsystems(execution)
        
        # Start automation based on level
        if config.automation_level == AutomationLevel.FULLY_AUTOMATED:
            asyncio.create_task(self._run_fully_automated_demo(execution))
        elif config.automation_level == AutomationLevel.AI_POWERED:
            asyncio.create_task(self._run_ai_powered_demo(execution))
        else:
            # Manual/semi-automated
            self._start_semi_automated_demo(execution)
        
        self.logger.info(f"Started demo execution: {execution_id}")
        return execution_id
    
    def _initialize_demo_subsystems(self, execution: DemoExecution):
        """Initialize all demo subsystems for execution"""
        config = execution.configuration
        
        # Initialize demo manager
        if config.recording_enabled:
            self.demo_manager.start_demo_session(
                demo_type=config.demo_type,
                stakeholder_type=config.stakeholder_type,
                recording_required=True
            )
        
        # Initialize recorder if needed
        if config.recording_enabled:
            self.recorder.start_recording(
                recording_type=RecordingType.INTERACTIVE,
                quality=RecordingQuality.HIGH
            )
            execution.recording_session_id = self.recorder.current_session.session_id
        
        # Initialize feedback system
        if config.feedback_collection:
            self.feedback_manager.create_feedback_form(
                stakeholder_type=StakeholderFeedback(config.stakeholder_type.value),
                session_id=execution.execution_id
            )
        
        self._record_automation_event(execution.execution_id, "initialization_complete", {})
    
    async def _run_fully_automated_demo(self, execution: DemoExecution):
        """Run fully automated demo"""
        config = execution.configuration
        execution.current_state = DemoState.RUNNING
        
        try:
            for scenario_id in config.scenario_sequence:
                execution.current_scenario = scenario_id
                self.logger.info(f"Running scenario: {scenario_id}")
                
                # Run scenario automation
                await self._run_automated_scenario(execution, scenario_id)
                
                execution.completed_scenarios.append(scenario_id)
                execution.automation_progress = len(execution.completed_scenarios) / len(config.scenario_sequence)
                
                # Add scenario transition delay
                await asyncio.sleep(2)
            
            # Complete demo
            await self._complete_automated_demo(execution)
            
        except Exception as e:
            self.logger.error(f"Error in fully automated demo: {e}")
            await self._handle_demo_error(execution, e)
    
    async def _run_ai_powered_demo(self, execution: DemoExecution):
        """Run AI-powered adaptive demo"""
        execution.current_state = DemoState.RUNNING
        
        config = execution.configuration
        
        for scenario_id in config.scenario_sequence:
            execution.current_scenario = scenario_id
            
            # AI-powered scenario adaptation
            scenario = self.scenario_manager.get_scenario(scenario_id)
            if scenario:
                # Adapt scenario based on AI analysis
                adapted_scenario = await self._adapt_scenario_with_ai(scenario, execution)
                
                # Run adapted scenario
                await self._run_adapted_scenario(execution, adapted_scenario)
            
            execution.completed_scenarios.append(scenario_id)
            execution.automation_progress = len(execution.completed_scenarios) / len(config.scenario_sequence)
        
        await self._complete_automated_demo(execution)
    
    async def _run_automated_scenario(self, execution: DemoExecution, scenario_id: str):
        """Run individual automated scenario"""
        scenario = self.scenario_manager.get_scenario(scenario_id)
        if not scenario:
            return
        
        # Get scenario steps
        steps = scenario.scenario_steps
        
        for i, step in enumerate(steps):
            self.logger.info(f"Running step {i+1}/{len(steps)}: {step.title}")
            
            # Simulate scenario execution
            await self._execute_scenario_step(execution, step)
            
            # Add step timing
            await asyncio.sleep(1)  # Simulate step duration
        
        self._record_automation_event(execution.execution_id, "scenario_completed", {
            "scenario_id": scenario_id,
            "steps_completed": len(steps)
        })
    
    async def _execute_scenario_step(self, execution: DemoExecution, step):
        """Execute individual scenario step"""
        # Simulate step execution
        self.logger.info(f"Executing: {step.title} - {step.description}")
        
        # Add annotations if recording
        if execution.recording_session_id:
            self.recorder.add_annotation(
                annotation_type="timestamp",
                timestamp=time.time(),
                text=step.title
            )
        
        # Simulate interaction
        execution.interaction_count += 1
        
        # Add realistic timing
        await asyncio.sleep(0.1)  # Short delay for simulation
    
    async def _adapt_scenario_with_ai(self, scenario: MedicalScenario, execution: DemoExecution) -> MedicalScenario:
        """AI-powered scenario adaptation"""
        # In production, this would use AI to analyze stakeholder responses
        # and adapt the scenario in real-time
        
        adapted_scenario = scenario
        
        # Simple adaptation based on stakeholder type
        if execution.configuration.stakeholder_type == StakeholderType.C_SUITE:
            # Focus on business value
            for step in adapted_scenario.scenario_steps:
                if "business" not in step.key_learning_points:
                    step.key_learning_points.append("Business value demonstration")
        
        elif execution.configuration.stakeholder_type == StakeholderType.CLINICAL:
            # Focus on clinical evidence
            for step in adapted_scenario.scenario_steps:
                if "evidence" not in step.key_learning_points:
                    step.key_learning_points.append("Evidence-based practice")
        
        return adapted_scenario
    
    async def _run_adapted_scenario(self, execution: DemoExecution, adapted_scenario: MedicalScenario):
        """Run AI-adapted scenario"""
        await self._run_automated_scenario(execution, adapted_scenario.scenario_id)
    
    def _start_semi_automated_demo(self, execution: DemoExecution):
        """Start semi-automated demo (with presenter control)"""
        execution.current_state = DemoState.PREPARING
        
        # Provide presenter interface
        self._setup_presenter_interface(execution)
        
        # Start with first scenario
        if execution.configuration.scenario_sequence:
            first_scenario = execution.configuration.scenario_sequence[0]
            self.start_scenario(execution.execution_id, first_scenario)
    
    def _setup_presenter_interface(self, execution: DemoExecution):
        """Setup presenter control interface"""
        # In production, this would set up a real-time control interface
        presenter_controls = {
            "pause_demo": self.pause_demo,
            "resume_demo": self.resume_demo,
            "advance_scenario": self.advance_scenario,
            "add_annotation": self.add_demo_annotation,
            "collect_feedback": self.collect_real_time_feedback
        }
        
        self._record_automation_event(execution.execution_id, "presenter_interface_ready", {
            "available_controls": list(presenter_controls.keys())
        })
    
    async def _complete_automated_demo(self, execution: DemoExecution):
        """Complete automated demo execution"""
        execution.current_state = DemoState.COMPLETING
        execution.end_time = datetime.now()
        
        # Stop recording if active
        if execution.recording_session_id:
            self.recorder.stop_recording()
        
        # Collect feedback if enabled
        if execution.configuration.feedback_collection:
            execution.current_state = DemoState.COLLECTING_FEEDBACK
            await self._collect_automated_feedback(execution)
        
        # Generate final report
        await self._generate_demo_report(execution)
        
        # Cleanup
        execution.current_state = DemoState.CLEANUP
        await self._cleanup_demo_execution(execution)
        
        execution.current_state = DemoState.COMPLETING
        
        self.logger.info(f"Demo execution completed: {execution.execution_id}")
    
    async def _collect_automated_feedback(self, execution: DemoExecution):
        """Collect automated feedback"""
        # Simulate automated feedback collection
        automated_feedback = {
            "overall_demo_rating": "8.5",
            "engagement_level": "High",
            "stakeholder_satisfaction": "Very Positive",
            "conversion_likelihood": "High",
            "key_interests": ["ROI", "Clinical evidence", "Implementation timeline"],
            "improvement_areas": ["More interactive elements", "Additional use cases"]
        }
        
        self.feedback_manager.collect_feedback(
            session_id=execution.execution_id,
            stakeholder_type=StakeholderFeedback(execution.configuration.stakeholder_type.value),
            responses=automated_feedback,
            feedback_type=FeedbackType.FOLLOW_UP
        )
        
        self._record_automation_event(execution.execution_id, "feedback_collected", automated_feedback)
    
    async def _generate_demo_report(self, execution: DemoExecution):
        """Generate comprehensive demo report"""
        report = {
            "execution_summary": {
                "execution_id": execution.execution_id,
                "demo_id": execution.configuration.demo_id,
                "demo_type": execution.configuration.demo_type.value,
                "stakeholder_type": execution.configuration.stakeholder_type.value,
                "duration_minutes": (
                    (execution.end_time - execution.start_time).total_seconds() / 60
                    if execution.start_time and execution.end_time else 0
                ),
                "scenarios_completed": len(execution.completed_scenarios),
                "scenarios_total": len(execution.configuration.scenario_sequence),
                "automation_progress": execution.automation_progress,
                "interaction_count": execution.interaction_count
            },
            "performance_metrics": {
                "completion_rate": len(execution.completed_scenarios) / len(execution.configuration.scenario_sequence),
                "automation_effectiveness": self._calculate_automation_effectiveness(execution),
                "stakeholder_engagement": self._assess_stakeholder_engagement(execution),
                "conversion_probability": self._calculate_conversion_probability(execution)
            },
            "recommendations": self._generate_demo_recommendations(execution),
            "next_actions": self._generate_next_actions(execution)
        }
        
        # Save report
        report_file = Path(f"demo_reports/{execution.execution_id}_report.json")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._record_automation_event(execution.execution_id, "report_generated", {
            "report_file": str(report_file),
            "report_summary": report["execution_summary"]
        })
    
    def _calculate_automation_effectiveness(self, execution: DemoExecution) -> float:
        """Calculate automation effectiveness score"""
        if not execution.completed_scenarios:
            return 0.0
        
        # Base score on completion rate
        completion_score = len(execution.completed_scenarios) / len(execution.configuration.scenario_sequence)
        
        # Adjust for interaction quality
        interaction_score = min(1.0, execution.interaction_count / 10.0)  # Normalize to 10 interactions
        
        return (completion_score * 0.7) + (interaction_score * 0.3)
    
    def _assess_stakeholder_engagement(self, execution: DemoExecution) -> str:
        """Assess stakeholder engagement level"""
        if execution.interaction_count > 15:
            return "Very High"
        elif execution.interaction_count > 10:
            return "High"
        elif execution.interaction_count > 5:
            return "Moderate"
        else:
            return "Low"
    
    def _calculate_conversion_probability(self, execution: DemoExecution) -> float:
        """Calculate conversion probability based on demo metrics"""
        engagement_score = min(1.0, execution.interaction_count / 10.0)
        completion_score = len(execution.completed_scenarios) / len(execution.configuration.scenario_sequence)
        
        # Stakeholder type modifiers
        stakeholder_modifiers = {
            StakeholderType.C_SUITE: 0.8,  # C-suite has higher conversion rates
            StakeholderType.INVESTOR: 0.9,
            StakeholderType.CLINICAL: 0.6,
            StakeholderType.REGULATORY: 0.4
        }
        
        modifier = stakeholder_modifiers.get(execution.configuration.stakeholder_type, 0.5)
        base_probability = (engagement_score * 0.4) + (completion_score * 0.6)
        
        return min(1.0, base_probability * modifier)
    
    def _generate_demo_recommendations(self, execution: DemoExecution) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if len(execution.completed_scenarios) < len(execution.configuration.scenario_sequence):
            recommendations.append("Consider extending demo time or reducing scenario complexity")
        
        if execution.interaction_count < 5:
            recommendations.append("Increase interactive elements to boost engagement")
        
        if execution.configuration.automation_level == AutomationLevel.SEMI_AUTOMATED:
            recommendations.append("Consider upgrading to fully automated for consistent delivery")
        
        return recommendations
    
    def _generate_next_actions(self, execution: DemoExecution) -> List[str]:
        """Generate next action items"""
        actions = [
            "Follow up with stakeholder within 24 hours",
            "Send demo recording and materials",
            "Schedule technical deep-dive if requested",
            "Prepare pilot program proposal"
        ]
        
        conversion_prob = self._calculate_conversion_probability(execution)
        if conversion_prob > 0.7:
            actions.append("Priority follow-up for immediate conversion")
        elif conversion_prob > 0.4:
            actions.append("Standard nurture campaign for gradual conversion")
        
        return actions
    
    async def _cleanup_demo_execution(self, execution: DemoExecution):
        """Cleanup demo execution resources"""
        # Save execution data
        self._save_execution_to_db(execution)
        
        # Clean up temporary resources
        self._record_automation_event(execution.execution_id, "cleanup_complete", {})
        
        # Reset current execution
        self.current_execution = None
    
    def _save_execution_to_db(self, execution: DemoExecution):
        """Save execution to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO demo_executions
            (execution_id, demo_id, demo_type, stakeholder_type, scenario_sequence,
             automation_level, start_time, end_time, current_state, current_scenario,
             completed_scenarios, automation_progress, interaction_count, recording_session_id,
             feedback_session_id, custom_parameters, execution_summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            execution.execution_id,
            execution.configuration.demo_id,
            execution.configuration.demo_type.value,
            execution.configuration.stakeholder_type.value,
            json.dumps(execution.configuration.scenario_sequence),
            execution.configuration.automation_level.value,
            execution.start_time.isoformat() if execution.start_time else None,
            execution.end_time.isoformat() if execution.end_time else None,
            execution.current_state.value,
            execution.current_scenario,
            json.dumps(execution.completed_scenarios),
            execution.automation_progress,
            execution.interaction_count,
            execution.recording_session_id,
            execution.feedback_session_id,
            json.dumps(execution.configuration.custom_parameters),
            "Automated execution completed"  # Summary would be more detailed in production
        ))
        
        conn.commit()
        conn.close()
    
    def _record_automation_event(self, execution_id: str, event_type: str, event_data: Dict[str, Any]):
        """Record automation event"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO automation_events
            (execution_id, event_type, event_data, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (
            execution_id,
            event_type,
            json.dumps(event_data),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Automation event: {event_type} for {execution_id}")
    
    # Presenter control methods
    def pause_demo(self, execution_id: str):
        """Pause demo execution"""
        if self.current_execution and self.current_execution.execution_id == execution_id:
            self.current_execution.current_state = DemoState.PAUSED
            self.demo_manager.pause_demo()
    
    def resume_demo(self, execution_id: str):
        """Resume paused demo"""
        if self.current_execution and self.current_execution.execution_id == execution_id:
            self.current_execution.current_state = DemoState.RUNNING
            self.demo_manager.resume_demo()
    
    def start_scenario(self, execution_id: str, scenario_id: str) -> bool:
        """Manually start specific scenario"""
        if not (self.current_execution and self.current_execution.execution_id == execution_id):
            return False
        
        scenario_started = self.demo_manager.start_scenario(scenario_id)
        if scenario_started:
            self.current_execution.current_scenario = scenario_id
        
        return scenario_started
    
    def advance_scenario(self, execution_id: str) -> bool:
        """Advance to next scenario in sequence"""
        if not (self.current_execution and self.current_execution.execution_id == execution_id):
            return False
        
        config = self.current_execution.configuration
        current_index = config.scenario_sequence.index(self.current_execution.current_scenario) \
            if self.current_execution.current_scenario in config.scenario_sequence else -1
        
        if current_index < len(config.scenario_sequence) - 1:
            next_scenario = config.scenario_sequence[current_index + 1]
            return self.start_scenario(execution_id, next_scenario)
        
        return False
    
    def add_demo_annotation(self, execution_id: str, text: str, timestamp: Optional[float] = None):
        """Add annotation to current demo"""
        if execution_id == self.current_execution.execution_id and self.recorder.is_recording:
            self.recorder.add_annotation(
                annotation_type="text",
                timestamp=timestamp or time.time(),
                text=text
            )
    
    def collect_real_time_feedback(self, execution_id: str) -> Dict[str, Any]:
        """Collect real-time feedback during demo"""
        # In production, this would collect live feedback
        return {
            "attention_level": "High",
            "engagement_signals": ["Questions asked", "Calculator used", "Active discussion"],
            "conversion_indicators": ["Positive body language", "Asking about timeline", "Financial questions"]
        }
    
    async def _handle_demo_error(self, execution: DemoExecution, error: Exception):
        """Handle demo execution errors"""
        execution.current_state = DemoState.CLEANUP
        self.logger.error(f"Demo error in {execution.execution_id}: {error}")
        
        # Attempt recovery if configured
        if self.config["automation_settings"]["auto_recovery"]:
            await self._attempt_error_recovery(execution, error)
        
        await self._cleanup_demo_execution(execution)
    
    async def _attempt_error_recovery(self, execution: DemoExecution, error: Exception):
        """Attempt to recover from demo errors"""
        self.logger.info(f"Attempting error recovery for {execution.execution_id}")
        
        # Simple recovery strategies
        if "recording" in str(error).lower():
            # Restart recording
            if execution.configuration.recording_enabled:
                self.recorder.start_recording(RecordingType.INTERACTIVE)
        
        elif "scenario" in str(error).lower():
            # Skip to next scenario
            if execution.current_scenario:
                self.advance_scenario(execution.execution_id)
        
        self._record_automation_event(execution.execution_id, "error_recovery_attempted", {
            "error": str(error),
            "recovery_action": "attempted"
        })

def main():
    """Main function for demo orchestrator CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Demo Orchestrator")
    parser.add_argument("--create-demo", action="store_true", help="Create new demo configuration")
    parser.add_argument("--demo-type", type=str, choices=[t.value for t in DemoType], help="Demo type")
    parser.add_argument("--stakeholder", type=str, choices=[s.value for s in StakeholderType], help="Stakeholder type")
    parser.add_argument("--start-demo", type=str, help="Start demo execution with demo ID")
    parser.add_argument("--automation-level", type=str, choices=[a.value for a in AutomationLevel], 
                       default="semi_automated", help="Automation level")
    parser.add_argument("--record", action="store_true", help="Enable recording")
    parser.add_argument("--feedback", action="store_true", help="Enable feedback collection")
    
    args = parser.parse_args()
    
    orchestrator = DemoOrchestrator()
    
    if args.create_demo:
        demo_id = orchestrator.create_demo_configuration(
            demo_type=DemoType(args.demo_type),
            stakeholder_type=StakeholderType(args.stakeholder),
            automation_level=AutomationLevel(args.automation_level),
            recording_enabled=args.record,
            feedback_collection=args.feedback
        )
        print(f"Created demo configuration: {demo_id}")
    
    elif args.start_demo:
        execution_id = orchestrator.start_automated_demo(args.start_demo)
        print(f"Started demo execution: {execution_id}")
        print("Demo is running... Use Ctrl+C to stop")
        
        try:
            # Keep the script running for demo purposes
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDemo execution stopped")

if __name__ == "__main__":
    main()
