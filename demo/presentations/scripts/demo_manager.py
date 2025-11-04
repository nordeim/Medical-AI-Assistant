#!/usr/bin/env python3
"""
Demo Manager - Centralized demo control system for Medical AI Assistant presentations.

This module provides comprehensive demo management capabilities including:
- Real-time scenario selection and control
- Stakeholder-specific presentation flows
- Demo analytics and tracking
- Recording and documentation features
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class DemoType(Enum):
    """Demo type enumeration"""
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    EMERGENCY_MEDICINE = "emergency_medicine"
    CHRONIC_DISEASE = "chronic_disease"
    MULTI_SPECIALTY = "multi_specialty"

class StakeholderType(Enum):
    """Stakeholder type enumeration"""
    C_SUITE = "c_suite"
    CLINICAL = "clinical"
    REGULATORY = "regulatory"
    INVESTOR = "investor"
    PARTNER = "partner"
    TECHNICAL = "technical"

class DemoStatus(Enum):
    """Demo status enumeration"""
    IDLE = "idle"
    PREPARING = "preparing"
    RUNNING = "running"
    PAUSED = "paused"
    RECORDING = "recording"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class DemoSession:
    """Demo session data structure"""
    session_id: str
    demo_type: DemoType
    stakeholder_type: StakeholderType
    status: DemoStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_scenario: Optional[str] = None
    scenario_progress: float = 0.0
    total_audience_size: int = 1
    recording_active: bool = False
    feedback_collected: bool = False
    notes: Optional[str] = None

@dataclass
class ScenarioStep:
    """Individual scenario step"""
    step_id: str
    title: str
    description: str
    duration_seconds: int
    key_points: List[str]
    expected_responses: List[str]
    visual_cues: List[str]
    timing_notes: Optional[str] = None

class DemoManager:
    """Central demo management system"""
    
    def __init__(self, db_path: str = "demo_analytics.db"):
        self.db_path = db_path
        self.current_session: Optional[DemoSession] = None
        self.setup_logging()  # Initialize logger first
        self.scenarios_db = self._load_scenarios_db()
        self._init_database()
        
    def setup_logging(self):
        """Setup logging for demo manager"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('demo_manager.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _init_database(self):
        """Initialize demo analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Demo sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS demo_sessions (
                session_id TEXT PRIMARY KEY,
                demo_type TEXT NOT NULL,
                stakeholder_type TEXT NOT NULL,
                status TEXT NOT NULL,
                start_time TEXT,
                end_time TEXT,
                current_scenario TEXT,
                scenario_progress REAL DEFAULT 0.0,
                total_audience_size INTEGER DEFAULT 1,
                recording_active BOOLEAN DEFAULT FALSE,
                feedback_collected BOOLEAN DEFAULT FALSE,
                notes TEXT
            )
        ''')
        
        # Scenario interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenario_interactions (
                interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                step_id TEXT,
                timestamp TEXT,
                interaction_type TEXT,
                duration_seconds REAL,
                FOREIGN KEY (session_id) REFERENCES demo_sessions (session_id)
            )
        ''')
        
        # Feedback responses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_responses (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                stakeholder_type TEXT,
                question_id TEXT,
                response_value TEXT,
                rating INTEGER,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES demo_sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_scenarios_db(self) -> Dict[str, Dict]:
        """Load scenario database"""
        scenarios_path = Path("../medical-scenarios")
        
        if not scenarios_path.exists():
            self.logger.warning(f"Medical scenarios directory not found: {scenarios_path}")
            return {}
        
        scenarios_db = {}
        
        # Load cardiology scenarios
        cardiology_path = scenarios_path / "specialties" / "cardiology-scenarios.md"
        if cardiology_path.exists():
            scenarios_db["cardiology"] = self._parse_scenarios_file(cardiology_path)
        
        # Load oncology scenarios
        oncology_path = scenarios_path / "specialties" / "oncology-scenarios.md"
        if oncology_path.exists():
            scenarios_db["oncology"] = self._parse_scenarios_file(oncology_path)
        
        # Load emergency scenarios
        emergency_path = scenarios_path / "emergency" / "emergency-scenarios.md"
        if emergency_path.exists():
            scenarios_db["emergency_medicine"] = self._parse_scenarios_file(emergency_path)
        
        # Load chronic disease scenarios
        chronic_path = scenarios_path / "chronic-disease" / "chronic-disease-scenarios.md"
        if chronic_path.exists():
            scenarios_db["chronic_disease"] = self._parse_scenarios_file(chronic_path)
        
        return scenarios_db
    
    def _parse_scenarios_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse scenarios from markdown file"""
        # This is a simplified parser - in production, you'd use a proper markdown parser
        scenarios = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract scenario titles and descriptions
            lines = content.split('\n')
            current_scenario = None
            
            for line in lines:
                if line.startswith('### ') and not line.startswith('####'):
                    current_scenario = line.strip('# ')
                    scenarios[current_scenario] = {
                        'title': current_scenario,
                        'description': '',
                        'key_points': [],
                        'expected_outcomes': []
                    }
                elif current_scenario and line.strip():
                    scenarios[current_scenario]['description'] += line + ' '
            
        except Exception as e:
            self.logger.error(f"Error parsing scenarios file {file_path}: {e}")
        
        return scenarios
    
    def start_demo_session(
        self,
        demo_type: DemoType,
        stakeholder_type: StakeholderType,
        audience_size: int = 1,
        recording_required: bool = False,
        notes: Optional[str] = None
    ) -> str:
        """Start a new demo session"""
        session_id = f"demo_{int(time.time())}_{demo_type.value}_{stakeholder_type.value}"
        
        self.current_session = DemoSession(
            session_id=session_id,
            demo_type=demo_type,
            stakeholder_type=stakeholder_type,
            status=DemoStatus.PREPARING,
            start_time=datetime.now(),
            total_audience_size=audience_size,
            recording_active=recording_required,
            notes=notes
        )
        
        self._save_session_to_db()
        self.logger.info(f"Started demo session: {session_id}")
        return session_id
    
    def get_scenarios(self, demo_type: DemoType) -> Dict[str, Dict]:
        """Get available scenarios for demo type"""
        if demo_type.value in self.scenarios_db:
            return self.scenarios_db[demo_type.value]
        return {}
    
    def start_scenario(self, scenario_name: str) -> bool:
        """Start a specific scenario"""
        if not self.current_session:
            self.logger.error("No active demo session")
            return False
        
        # Check if scenario exists
        scenarios = self.get_scenarios(self.current_session.demo_type)
        if scenario_name not in scenarios:
            self.logger.error(f"Scenario not found: {scenario_name}")
            return False
        
        self.current_session.current_scenario = scenario_name
        self.current_session.scenario_progress = 0.0
        self.current_session.status = DemoStatus.RUNNING
        
        self._save_session_to_db()
        self._record_interaction("scenario_start", scenario_name, 0)
        
        self.logger.info(f"Started scenario: {scenario_name}")
        return True
    
    def update_scenario_progress(self, progress: float, step_completed: Optional[str] = None):
        """Update scenario progress"""
        if not self.current_session:
            return
        
        self.current_session.scenario_progress = min(100.0, max(0.0, progress))
        
        if step_completed:
            self._record_interaction("step_complete", step_completed, progress)
        
        self._save_session_to_db()
    
    def pause_demo(self):
        """Pause the current demo"""
        if self.current_session and self.current_session.status == DemoStatus.RUNNING:
            self.current_session.status = DemoStatus.PAUSED
            self._save_session_to_db()
            self.logger.info("Demo paused")
    
    def resume_demo(self):
        """Resume the paused demo"""
        if self.current_session and self.current_session.status == DemoStatus.PAUSED:
            self.current_session.status = DemoStatus.RUNNING
            self._save_session_to_db()
            self.logger.info("Demo resumed")
    
    def complete_demo(self):
        """Complete the current demo"""
        if not self.current_session:
            return
        
        self.current_session.status = DemoStatus.COMPLETED
        self.current_session.end_time = datetime.now()
        self.current_session.scenario_progress = 100.0
        
        self._save_session_to_db()
        
        session_duration = (
            self.current_session.end_time - self.current_session.start_time
        ).total_seconds() if self.current_session.start_time else 0
        
        self.logger.info(f"Demo completed. Duration: {session_duration:.1f}s")
        
        # Generate demo summary
        return self.generate_demo_summary()
    
    def _record_interaction(self, interaction_type: str, detail: str, progress: float):
        """Record demo interaction"""
        if not self.current_session:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO scenario_interactions 
            (session_id, step_id, timestamp, interaction_type, duration_seconds)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.current_session.session_id,
            detail,
            datetime.now().isoformat(),
            interaction_type,
            progress
        ))
        
        conn.commit()
        conn.close()
    
    def _save_session_to_db(self):
        """Save current session to database"""
        if not self.current_session:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO demo_sessions
            (session_id, demo_type, stakeholder_type, status, start_time, end_time,
             current_scenario, scenario_progress, total_audience_size,
             recording_active, feedback_collected, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.current_session.session_id,
            self.current_session.demo_type.value,
            self.current_session.stakeholder_type.value,
            self.current_session.status.value,
            self.current_session.start_time.isoformat() if self.current_session.start_time else None,
            self.current_session.end_time.isoformat() if self.current_session.end_time else None,
            self.current_session.current_scenario,
            self.current_session.scenario_progress,
            self.current_session.total_audience_size,
            self.current_session.recording_active,
            self.current_session.feedback_collected,
            self.current_session.notes
        ))
        
        conn.commit()
        conn.close()
    
    def get_demo_analytics(self) -> Dict[str, Any]:
        """Get demo analytics for current session"""
        if not self.current_session:
            return {"error": "No active demo session"}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get interactions for current session
        cursor.execute('''
            SELECT interaction_type, COUNT(*) as count, AVG(duration_seconds) as avg_duration
            FROM scenario_interactions
            WHERE session_id = ?
            GROUP BY interaction_type
        ''', (self.current_session.session_id,))
        
        interactions = cursor.fetchall()
        
        conn.close()
        
        return {
            "session_id": self.current_session.session_id,
            "demo_type": self.current_session.demo_type.value,
            "stakeholder_type": self.current_session.stakeholder_type.value,
            "status": self.current_session.status.value,
            "current_scenario": self.current_session.current_scenario,
            "progress": self.current_session.scenario_progress,
            "interactions": [
                {
                    "type": interaction[0],
                    "count": interaction[1],
                    "avg_duration": interaction[2]
                }
                for interaction in interactions
            ],
            "start_time": self.current_session.start_time.isoformat() if self.current_session.start_time else None,
            "recording_active": self.current_session.recording_active
        }
    
    def generate_demo_summary(self) -> Dict[str, Any]:
        """Generate comprehensive demo summary"""
        if not self.current_session:
            return {"error": "No demo session to summarize"}
        
        analytics = self.get_demo_analytics()
        total_duration = (
            self.current_session.end_time - self.current_session.start_time
        ).total_seconds() if self.current_session.end_time and self.current_session.start_time else 0
        
        return {
            "session_summary": {
                "session_id": self.current_session.session_id,
                "demo_type": self.current_session.demo_type.value,
                "stakeholder_type": self.current_session.stakeholder_type.value,
                "duration_minutes": round(total_duration / 60, 2),
                "audience_size": self.current_session.total_audience_size,
                "completed_scenarios": [self.current_session.current_scenario] if self.current_session.current_scenario else [],
                "overall_progress": self.current_session.scenario_progress,
                "recording_generated": self.current_session.recording_active,
                "feedback_required": self.current_session.feedback_collected
            },
            "analytics": analytics,
            "recommendations": self._generate_recommendations(analytics),
            "next_actions": self._generate_next_actions()
        }
    
    def _generate_recommendations(self, analytics: Dict[str, Any]) -> List[str]:
        """Generate demo improvement recommendations"""
        recommendations = []
        
        # Check scenario completion rate
        if analytics.get("progress", 0) < 80:
            recommendations.append("Consider extending demo time for better scenario completion")
        
        # Check interaction levels
        interactions = analytics.get("interactions", [])
        total_interactions = sum(i.get("count", 0) for i in interactions)
        
        if total_interactions < 10:
            recommendations.append("Increase audience interaction to improve engagement")
        
        # Check demo type specific recommendations
        if analytics.get("demo_type") == "cardiology":
            recommendations.append("Consider highlighting emergency medicine scenarios for dramatic effect")
        elif analytics.get("demo_type") == "oncology":
            recommendations.append("Emphasize multidisciplinary care coordination for comprehensive value")
        
        return recommendations
    
    def _generate_next_actions(self) -> List[str]:
        """Generate next actions for demo improvement"""
        actions = []
        
        if not self.current_session.feedback_collected:
            actions.append("Collect stakeholder feedback via survey")
        
        if self.current_session.recording_active:
            actions.append("Review and edit demo recording for future use")
        
        actions.append("Update scenario scripts based on live feedback")
        actions.append("Schedule follow-up demonstration if required")
        
        return actions
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        if not self.current_session:
            return None
        
        return {
            "session_id": self.current_session.session_id,
            "demo_type": self.current_session.demo_type.value,
            "stakeholder_type": self.current_session.stakeholder_type.value,
            "status": self.current_session.status.value,
            "current_scenario": self.current_session.current_scenario,
            "progress": self.current_session.scenario_progress,
            "start_time": self.current_session.start_time.isoformat() if self.current_session.start_time else None,
            "duration_minutes": (
                (datetime.now() - self.current_session.start_time).total_seconds() / 60
                if self.current_session.start_time else 0
            ),
            "recording_active": self.current_session.recording_active
        }

def main():
    """Main function for demo manager CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI Assistant Demo Manager")
    parser.add_argument("--demo-type", type=str, choices=[t.value for t in DemoType], required=True)
    parser.add_argument("--stakeholder", type=str, choices=[s.value for s in StakeholderType], required=True)
    parser.add_argument("--audience-size", type=int, default=1, help="Number of people in audience")
    parser.add_argument("--recording", action="store_true", help="Enable recording")
    parser.add_argument("--notes", type=str, help="Demo notes")
    
    args = parser.parse_args()
    
    # Initialize demo manager
    manager = DemoManager()
    
    # Start demo session
    session_id = manager.start_demo_session(
        demo_type=DemoType(args.demo_type),
        stakeholder_type=StakeholderType(args.stakeholder),
        audience_size=args.audience_size,
        recording_required=args.recording,
        notes=args.notes
    )
    
    print(f"Demo session started: {session_id}")
    print(f"Demo type: {args.demo_type}")
    print(f"Stakeholder: {args.stakeholder}")
    print(f"Audience size: {args.audience_size}")
    print(f"Recording: {'Enabled' if args.recording else 'Disabled'}")
    
    # Display available scenarios
    scenarios = manager.get_scenarios(DemoType(args.demo_type))
    print(f"\nAvailable scenarios for {args.demo_type}:")
    for i, scenario_name in enumerate(scenarios.keys(), 1):
        print(f"{i}. {scenario_name}")
    
    return session_id

if __name__ == "__main__":
    main()
