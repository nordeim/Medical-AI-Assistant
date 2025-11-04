"""
Agile Operational Excellence Framework for Healthcare AI
Implements Agile methodologies for iterative improvements and adaptive planning
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

class AgileMethodology(Enum):
    """Agile methodologies for healthcare AI operations"""
    SCRUM = "scrum"
    KANBAN = "kanban"
    LEAN_AGGILE = "lean_agile"
    SAFe = "safe"  # Scaled Agile Framework
    DAD = "dad"  # Disciplined Agile Delivery

class SprintStatus(Enum):
    """Sprint status tracking"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class WorkItemType(Enum):
    """Types of work items in agile processes"""
    EPIC = "epic"
    FEATURE = "feature"
    STORY = "story"
    TASK = "task"
    BUG = "bug"
    IMPROVEMENT = "improvement"

class WorkItemStatus(Enum):
    """Work item status tracking"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    TESTING = "testing"
    DONE = "done"
    BLOCKED = "blocked"

@dataclass
class WorkItem:
    """Agile work item"""
    item_id: str
    title: str
    description: str
    work_item_type: WorkItemType
    status: WorkItemStatus
    priority: str  # Critical, High, Medium, Low
    story_points: int
    estimated_hours: float
    actual_hours: float
    assignee: str
    epic_id: Optional[str] = None
    sprint_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)
    completed_date: Optional[datetime] = None

@dataclass
class Sprint:
    """Agile sprint"""
    sprint_id: str
    name: str
    start_date: datetime
    end_date: datetime
    sprint_goal: str
    status: SprintStatus
    planned_story_points: int
    completed_story_points: int
    velocity: float
    burndown_data: List[Dict] = field(default_factory=list)
    team_members: List[str] = field(default_factory=list)
    retrospective_notes: str = ""

@dataclass
class AgileMetrics:
    """Agile performance metrics"""
    velocity: float
    burndown_rate: float
    cycle_time: float
    lead_time: float
    throughput: int
    quality_metrics: Dict[str, float]
    customer_satisfaction: float
    team_productivity: float

@dataclass
class ImprovementInitiative:
    """Continuous improvement initiative"""
    initiative_id: str
    title: str
    description: str
    methodology: AgileMethodology
    current_sprint: int
    total_sprints: int
    success_metrics: Dict[str, float]
    stakeholder_impact: str
    implementation_status: str

class AgileHealthcareAIManager:
    """Agile Operations Manager for Healthcare AI"""
    
    def __init__(self):
        self.work_items: Dict[str, WorkItem] = {}
        self.sprints: Dict[str, Sprint] = {}
        self.epics: Dict[str, List[WorkItem]] = {}
        self.teams: Dict[str, List[str]] = {}
        self.sprint_velocity_history: List[float] = []
        self.improvement_initiatives: List[ImprovementInitiative] = []
        
        # Initialize default teams
        self.teams = {
            "ml_operations": ["ML Engineer", "Data Scientist", "DevOps Engineer", "QA Engineer"],
            "clinical_integration": ["Clinical Specialist", "Product Manager", "UX Designer", "Backend Engineer"],
            "infrastructure": ["System Architect", "Security Engineer", "Cloud Engineer", "Database Admin"],
            "quality_assurance": ["QA Lead", "Test Automation Engineer", "Clinical Auditor", "Compliance Officer"]
        }
    
    async def create_sprint(self, sprint_data: Dict) -> Sprint:
        """Create new agile sprint"""
        
        sprint = Sprint(
            sprint_id=sprint_data["sprint_id"],
            name=sprint_data["name"],
            start_date=datetime.strptime(sprint_data["start_date"], "%Y-%m-%d"),
            end_date=datetime.strptime(sprint_data["end_date"], "%Y-%m-%d"),
            sprint_goal=sprint_data["sprint_goal"],
            status=SprintStatus.PLANNED,
            planned_story_points=sprint_data["planned_story_points"],
            completed_story_points=0,
            velocity=0,
            team_members=sprint_data["team_members"]
        )
        
        # Generate burndown data
        days_in_sprint = (sprint.end_date - sprint.start_date).days
        for day in range(days_in_sprint):
            remaining_points = sprint.planned_story_points - (day * (sprint.planned_story_points / days_in_sprint))
            sprint.burndown_data.append({
                "day": day + 1,
                "remaining_points": max(0, remaining_points),
                "date": (sprint.start_date + timedelta(days=day)).isoformat()
            })
        
        self.sprints[sprint.sprint_id] = sprint
        return sprint
    
    async def create_work_item(self, item_data: Dict) -> WorkItem:
        """Create new work item"""
        
        work_item = WorkItem(
            item_id=item_data["item_id"],
            title=item_data["title"],
            description=item_data["description"],
            work_item_type=WorkItemType(item_data["work_item_type"]),
            status=WorkItemStatus.TODO,
            priority=item_data["priority"],
            story_points=item_data["story_points"],
            estimated_hours=item_data["estimated_hours"],
            actual_hours=0,
            assignee=item_data["assignee"],
            epic_id=item_data.get("epic_id"),
            sprint_id=item_data.get("sprint_id"),
            dependencies=item_data.get("dependencies", []),
            acceptance_criteria=item_data.get("acceptance_criteria", [])
        )
        
        self.work_items[work_item.item_id] = work_item
        
        # Add to epic if specified
        if work_item.epic_id:
            if work_item.epic_id not in self.epics:
                self.epics[work_item.epic_id] = []
            self.epics[work_item.epic_id].append(work_item)
        
        return work_item
    
    async def plan_sprint(self, sprint_id: str, work_item_ids: List[str]) -> Dict:
        """Plan sprint with work items"""
        
        sprint = self.sprints[sprint_id]
        selected_items = [self.work_items[item_id] for item_id in work_item_ids]
        
        # Calculate total planned story points
        total_story_points = sum(item.story_points for item in selected_items)
        
        # Validate sprint capacity
        team_capacity = len(sprint.team_members) * 40  # 40 hours per team member
        total_estimated_hours = sum(item.estimated_hours for item in selected_items)
        
        capacity_utilization = (total_estimated_hours / team_capacity * 100) if team_capacity > 0 else 0
        
        sprint.planned_story_points = total_story_points
        
        # Assign items to sprint
        for item in selected_items:
            item.sprint_id = sprint_id
            item.status = WorkItemStatus.TODO
        
        planning_result = {
            "sprint_id": sprint_id,
            "planning_status": "completed",
            "total_story_points": total_story_points,
            "total_estimated_hours": total_estimated_hours,
            "team_capacity_hours": team_capacity,
            "capacity_utilization_percentage": round(capacity_utilization, 1),
            "items_planned": len(selected_items),
            "capacity_risk": "high" if capacity_utilization > 90 else "medium" if capacity_utilization > 75 else "low",
            "recommendations": [
                "Capacity looks good" if capacity_utilization <= 75 else "Consider reducing scope",
                "Monitor blocker resolution closely" if len([item for item in selected_items if item.dependencies]) > 0 else "No major dependencies",
                "Ensure proper testing coverage" if any(item.work_item_type == WorkItemType.FEATURE for item in selected_items) else "Mostly bug fixes and improvements"
            ]
        }
        
        return planning_result
    
    async def execute_sprint(self, sprint_id: str, execution_data: Dict) -> Dict:
        """Execute sprint with daily progress tracking"""
        
        sprint = self.sprints[sprint_id]
        sprint.status = SprintStatus.IN_PROGRESS
        
        # Update work item progress
        work_item_progress = execution_data.get("work_item_progress", [])
        total_completed_points = 0
        
        for progress in work_item_progress:
            item_id = progress["item_id"]
            if item_id in self.work_items:
                item = self.work_items[item_id]
                item.status = WorkItemStatus(progress["status"])
                item.actual_hours = progress.get("actual_hours", item.actual_hours)
                
                if item.status == WorkItemStatus.DONE:
                    total_completed_points += item.story_points
                    item.completed_date = datetime.now()
        
        sprint.completed_story_points = total_completed_points
        sprint.velocity = total_completed_points / ((datetime.now() - sprint.start_date).days + 1)
        
        # Calculate sprint metrics
        completion_percentage = (sprint.completed_story_points / sprint.planned_story_points * 100) if sprint.planned_story_points > 0 else 0
        
        # Update burndown data
        current_day = (datetime.now() - sprint.start_date).days + 1
        for data_point in sprint.burndown_data:
            if data_point["day"] == current_day:
                data_point["remaining_points"] = max(0, sprint.planned_story_points - sprint.completed_story_points)
        
        execution_result = {
            "sprint_id": sprint_id,
            "execution_status": "in_progress",
            "current_day": current_day,
            "total_sprint_days": (sprint.end_date - sprint.start_date).days,
            "completion_percentage": round(completion_percentage, 1),
            "velocity": round(sprint.velocity, 2),
            "completed_story_points": sprint.completed_story_points,
            "remaining_story_points": sprint.planned_story_points - sprint.completed_story_points,
            "burndown_status": "on_track" if completion_percentage >= (current_day / (sprint.end_date - sprint.start_date).days * 100) else "at_risk",
            "daily_metrics": {
                "stories_completed_today": len([p for p in work_item_progress if p["status"] == "done"]),
                "average_completion_time": 6.5,  # hours
                "bug_rate": 2.3,  # percentage
                "code_quality_score": 94.2
            }
        }
        
        return execution_result
    
    async def conduct_sprint_review(self, sprint_id: str) -> Dict:
        """Conduct sprint review meeting"""
        
        sprint = self.sprints[sprint_id]
        
        # Gather completed items
        completed_items = [item for item in self.work_items.values() 
                          if item.sprint_id == sprint_id and item.status == WorkItemStatus.DONE]
        
        # Calculate metrics
        completed_points = sum(item.story_points for item in completed_items)
        completion_rate = (completed_points / sprint.planned_story_points * 100) if sprint.planned_story_points > 0 else 0
        
        # Simulate stakeholder feedback
        stakeholder_feedback = {
            "clinical_team": {
                "satisfaction_score": 4.2,  # out of 5
                "feedback": "AI accuracy improvements are working well",
                "suggestions": ["Faster response times", "More clinical context"]
            },
            "it_team": {
                "satisfaction_score": 4.0,
                "feedback": "System stability has improved significantly",
                "suggestions": ["Better monitoring", "Easier deployment process"]
            },
            "quality_team": {
                "satisfaction_score": 4.3,
                "feedback": "Quality metrics are trending positively",
                "suggestions": ["Enhanced testing automation", "Continuous compliance monitoring"]
            }
        }
        
        # Review outcomes
        review_outcomes = {
            "delivered_features": [
                {
                    "feature": "Enhanced AI Model Accuracy",
                    "business_value": "High - Improves clinical decision support",
                    "acceptance_criteria_met": True,
                    "stakeholder_feedback": "Excellent - Exceeds expectations"
                },
                {
                    "feature": "Automated Data Validation",
                    "business_value": "Medium - Reduces manual errors",
                    "acceptance_criteria_met": True,
                    "stakeholder_feedback": "Good - As requested"
                },
                {
                    "feature": "Performance Monitoring Dashboard",
                    "business_value": "Medium - Improves operational visibility",
                    "acceptance_criteria_met": True,
                    "stakeholder_feedback": "Very good - Much needed"
                }
            ],
            "unfinished_work": [
                {
                    "item": "Advanced Clinical Reporting",
                    "reason": "Scope creep during sprint",
                    "reassignment": "Next sprint priority"
                }
            ],
            "improvements_identified": [
                "Increase automation in testing pipeline",
                "Better integration between clinical and technical teams",
                "Earlier stakeholder involvement in requirements"
            ]
        }
        
        review_result = {
            "sprint_id": sprint_id,
            "review_status": "completed",
            "completion_rate": round(completion_rate, 1),
            "velocity_achieved": sprint.velocity,
            "stakeholder_feedback": stakeholder_feedback,
            "review_outcomes": review_outcomes,
            "recommendations": [
                "Continue focus on clinical accuracy improvements",
                "Address automation gaps in testing",
                "Strengthen cross-functional collaboration",
                "Implement earlier stakeholder feedback loops"
            ]
        }
        
        return review_result
    
    async def conduct_sprint_retrospective(self, sprint_id: str) -> Dict:
        """Conduct sprint retrospective"""
        
        sprint = self.sprints[sprint_id]
        
        # Simulate retrospective feedback
        retrospective_data = {
            "what_went_well": [
                "Strong collaboration between ML and clinical teams",
                "Automated testing caught issues early",
                "Daily standups were effective and focused",
                "Customer feedback integration was smooth"
            ],
            "what_could_be_improved": [
                "More time needed for testing complex clinical scenarios",
                "Better coordination with external system teams",
                "Earlier identification of technical dependencies",
                "More frequent stakeholder demos"
            ],
            "action_items": [
                {
                    "action": "Implement comprehensive testing framework",
                    "owner": "QA Lead",
                    "due_date": "Next sprint",
                    "priority": "High"
                },
                {
                    "action": "Schedule regular stakeholder sync meetings",
                    "owner": "Product Manager",
                    "due_date": "Immediate",
                    "priority": "Medium"
                },
                {
                    "action": "Create dependency mapping for complex features",
                    "owner": "Tech Lead",
                    "due_date": "Next sprint",
                    "priority": "Medium"
                }
            ],
            "team_metrics": {
                "team_satisfaction": 4.1,  # out of 5
                "process_effectiveness": 4.0,
                "collaboration_score": 4.3,
                "technical_debt_impact": "Low"
            }
        }
        
        # Update sprint with retrospective notes
        sprint.retrospective_notes = json.dumps(retrospective_data)
        
        retrospective_result = {
            "sprint_id": sprint_id,
            "retrospective_status": "completed",
            "team_satisfaction": retrospective_data["team_metrics"]["team_satisfaction"],
            "improvement_focus": "Testing automation and stakeholder collaboration",
            "action_items_count": len(retrospective_data["action_items"]),
            "high_priority_actions": len([item for item in retrospective_data["action_items"] if item["priority"] == "High"]),
            "process_improvements": [
                "Enhanced testing protocols",
                "Improved stakeholder communication",
                "Better technical planning"
            ]
        }
        
        return retrospective_result
    
    async def calculate_agile_metrics(self) -> AgileMetrics:
        """Calculate comprehensive agile metrics"""
        
        # Velocity calculation (average over last 5 sprints)
        completed_sprints = [s for s in self.sprints.values() if s.status == SprintStatus.COMPLETED]
        if completed_sprints:
            velocities = [s.velocity for s in completed_sprints[-5:]]
            avg_velocity = sum(velocities) / len(velocities)
        else:
            avg_velocity = 25.0  # Default
        
        # Burndown rate
        active_sprints = [s for s in self.sprints.values() if s.status == SprintStatus.IN_PROGRESS]
        if active_sprints:
            current_sprint = active_sprints[0]
            days_passed = (datetime.now() - current_sprint.start_date).days
            total_days = (current_sprint.end_date - current_sprint.start_date).days
            planned_burndown = (total_days - days_passed) / total_days
            actual_completion = current_sprint.completed_story_points / current_sprint.planned_story_points if current_sprint.planned_story_points > 0 else 0
            burndown_rate = actual_completion / planned_burndown if planned_burndown > 0 else 1.0
        else:
            burndown_rate = 1.0
        
        # Cycle time (average time from start to completion)
        completed_items = [item for item in self.work_items.values() if item.completed_date]
        if completed_items:
            cycle_times = [(item.completed_date - item.created_date).days for item in completed_items]
            avg_cycle_time = sum(cycle_times) / len(cycle_times)
        else:
            avg_cycle_time = 5.0  # Default
        
        # Lead time (similar to cycle time)
        lead_time = avg_cycle_time
        
        # Throughput (items completed per sprint)
        throughput = len(completed_items) / len(completed_sprints) if completed_sprints else 8.0
        
        # Quality metrics
        bug_items = [item for item in self.work_items.values() if item.work_item_type == WorkItemType.BUG]
        completed_bugs = [item for item in bug_items if item.status == WorkItemStatus.DONE]
        quality_metrics = {
            "bug_reopening_rate": 5.2,  # percentage
            "test_coverage": 94.5,  # percentage
            "defect_density": 0.8,  # bugs per KLOC
            "customer_defect_rate": 1.2  # percentage
        }
        
        # Customer satisfaction
        customer_satisfaction = 4.2  # out of 5
        
        # Team productivity
        team_productivity = 92.3  # percentage
        
        return AgileMetrics(
            velocity=avg_velocity,
            burndown_rate=burndown_rate,
            cycle_time=avg_cycle_time,
            lead_time=lead_time,
            throughput=throughput,
            quality_metrics=quality_metrics,
            customer_satisfaction=customer_satisfaction,
            team_productivity=team_productivity
        )
    
    async def implement_continuous_improvement(self, initiative: ImprovementInitiative) -> Dict:
        """Implement continuous improvement using agile practices"""
        
        # Improvement phases
        improvement_phases = {
            "scoping": "Define improvement scope and success criteria",
            "analysis": "Analyze current state and identify root causes",
            "design": "Design future state and solution approach",
            "pilot": "Implement pilot solution with limited scope",
            "scale": "Scale successful pilot to full implementation",
            "sustain": "Establish monitoring and continuous improvement cycle"
        }
        
        # Simulate improvement implementation
        implementation_results = {
            "initiative_id": initiative.initiative_id,
            "current_phase": "pilot",
            "progress_percentage": 65,
            "milestones_achieved": [
                "Completed current state analysis",
                "Designed improvement solution",
                "Implemented pilot in clinical decision support"
            ],
            "success_metrics": {
                "process_efficiency": "+28%",
                "quality_improvement": "+15%",
                "cost_reduction": "$45K annually",
                "user_satisfaction": "+0.8 points"
            },
            "lessons_learned": [
                "Early stakeholder engagement critical for success",
                "Pilot testing reveals unexpected implementation challenges",
                "Incremental approach reduces risk and improves adoption"
            ],
            "next_steps": [
                "Scale pilot to additional departments",
                "Implement automated monitoring",
                "Establish regular review cycles"
            ]
        }
        
        self.improvement_initiatives.append(initiative)
        return implementation_results
    
    async def generate_agile_dashboard(self) -> Dict:
        """Generate agile operations dashboard"""
        
        metrics = await self.calculate_agile_metrics()
        
        dashboard_data = {
            "team_velocity": {
                "current_velocity": round(metrics.velocity, 1),
                "velocity_trend": "+12% this sprint",
                "commitment_reliability": 87.5,  # percentage
                "throughput": round(metrics.throughput, 1)
            },
            "sprint_health": {
                "active_sprints": len([s for s in self.sprints.values() if s.status == SprintStatus.IN_PROGRESS]),
                "completion_rate": 92.3,  # percentage
                "burndown_health": "on_track" if metrics.burndown_rate >= 1.0 else "at_risk",
                "quality_score": 94.2
            },
            "work_item_status": {
                "total_items": len(self.work_items),
                "completed_items": len([item for item in self.work_items.values() if item.status == WorkItemStatus.DONE]),
                "in_progress": len([item for item in self.work_items.values() if item.status == WorkItemStatus.IN_PROGRESS]),
                "blocked_items": len([item for item in self.work_items.values() if item.status == WorkItemStatus.BLOCKED])
            },
            "cycle_time_analysis": {
                "average_cycle_time": round(metrics.cycle_time, 1),
                "cycle_time_trend": "-15% this month",
                "target_cycle_time": 5.0,
                "improvement_rate": "23%"
            },
            "quality_metrics": metrics.quality_metrics,
            "customer_satisfaction": {
                "overall_score": metrics.customer_satisfaction,
                "trend": "+0.3 points this quarter",
                "stakeholder_feedback": "Very positive"
            },
            "team_productivity": {
                "productivity_score": round(metrics.team_productivity, 1),
                "trend": "+8% this quarter",
                "collaboration_score": 4.3,
                "process_effectiveness": 4.0
            },
            "continuous_improvement": {
                "active_initiatives": len(self.improvement_initiatives),
                "improvements_completed": 12,
                "roi_achieved": 285.5,  # percentage
                "innovation_rate": 15.2  # percentage of work focused on improvements
            }
        }
        
        return dashboard_data
    
    async def export_agile_report(self, filepath: str) -> Dict:
        """Export comprehensive agile operations report"""
        
        metrics = await self.calculate_agile_metrics()
        
        report_data = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_title": "Healthcare AI Agile Operations Report",
                "reporting_period": "Q4 2025",
                "methodology": "Scrum with Kanban elements"
            },
            "executive_summary": {
                "team_velocity": round(metrics.velocity, 1),
                "completion_rate": 92.3,
                "customer_satisfaction": metrics.customer_satisfaction,
                "quality_score": 94.2,
                "continuous_improvements": len(self.improvement_initiatives)
            },
            "sprint_details": [
                {
                    "sprint_id": s.sprint_id,
                    "name": s.name,
                    "goal": s.sprint_goal,
                    "planned_points": s.planned_story_points,
                    "completed_points": s.completed_story_points,
                    "velocity": round(s.velocity, 1),
                    "status": s.status.value
                }
                for s in self.sprints.values()
            ],
            "work_item_analysis": {
                "by_type": {wt.value: len([item for item in self.work_items.values() if item.work_item_type == wt]) 
                           for wt in WorkItemType},
                "by_priority": {
                    "Critical": len([item for item in self.work_items.values() if item.priority == "Critical"]),
                    "High": len([item for item in self.work_items.values() if item.priority == "High"]),
                    "Medium": len([item for item in self.work_items.values() if item.priority == "Medium"]),
                    "Low": len([item for item in self.work_items.values() if item.priority == "Low"])
                }
            },
            "recommendations": [
                "Maintain focus on clinical accuracy improvements",
                "Increase automation in testing and deployment",
                "Strengthen stakeholder collaboration",
                "Implement advanced analytics for predictive planning",
                "Establish continuous feedback loops with clinical users"
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return {"status": "success", "report_file": filepath}

# Example usage and testing
async def run_agile_demo():
    """Demonstrate Agile Framework implementation"""
    agile_manager = AgileHealthcareAIManager()
    
    # 1. Create Sprint
    print("=== Creating Sprint ===")
    sprint_data = {
        "sprint_id": "SPRINT_2025_11_001",
        "name": "Clinical AI Enhancement Sprint",
        "start_date": "2025-11-04",
        "end_date": "2025-11-18",
        "sprint_goal": "Improve AI model accuracy and response time for clinical decisions",
        "planned_story_points": 35,
        "team_members": ["ML Engineer", "Clinical Specialist", "DevOps", "QA Engineer"]
    }
    
    sprint = await agile_manager.create_sprint(sprint_data)
    print(f"Sprint: {sprint.name}")
    print(f"Goal: {sprint.sprint_goal}")
    print(f"Duration: {sprint.start_date.strftime('%Y-%m-%d')} to {sprint.end_date.strftime('%Y-%m-%d')}")
    
    # 2. Create Work Items
    print("\n=== Creating Work Items ===")
    work_items_data = [
        {
            "item_id": "WI_001",
            "title": "Enhance Model Training Pipeline",
            "description": "Improve AI model training with better data validation",
            "work_item_type": "feature",
            "priority": "High",
            "story_points": 8,
            "estimated_hours": 20.0,
            "assignee": "ML Engineer",
            "sprint_id": sprint.sprint_id,
            "acceptance_criteria": ["Automated data validation", "Performance monitoring"]
        },
        {
            "item_id": "WI_002", 
            "title": "Fix Clinical Decision Bug",
            "description": "Resolve accuracy issue in emergency triage",
            "work_item_type": "bug",
            "priority": "Critical",
            "story_points": 5,
            "estimated_hours": 12.0,
            "assignee": "Clinical Specialist",
            "sprint_id": sprint.sprint_id
        },
        {
            "item_id": "WI_003",
            "title": "Performance Monitoring Dashboard",
            "description": "Create real-time performance monitoring for clinical team",
            "work_item_type": "feature",
            "priority": "Medium",
            "story_points": 13,
            "estimated_hours": 32.0,
            "assignee": "DevOps",
            "sprint_id": sprint.sprint_id
        }
    ]
    
    work_items = []
    for item_data in work_items_data:
        item = await agile_manager.create_work_item(item_data)
        work_items.append(item)
        print(f"Created: {item.title} ({item.story_points} points)")
    
    # 3. Plan Sprint
    print("\n=== Planning Sprint ===")
    planning_result = await agile_manager.plan_sprint(
        sprint.sprint_id,
        [item.item_id for item in work_items]
    )
    print(f"Total Story Points: {planning_result['total_story_points']}")
    print(f"Capacity Utilization: {planning_result['capacity_utilization_percentage']}%")
    print(f"Risk Level: {planning_result['capacity_risk']}")
    
    # 4. Execute Sprint
    print("\n=== Executing Sprint ===")
    execution_data = {
        "work_item_progress": [
            {"item_id": "WI_001", "status": "done", "actual_hours": 18.5},
            {"item_id": "WI_002", "status": "done", "actual_hours": 10.0},
            {"item_id": "WI_003", "status": "in_progress", "actual_hours": 24.0}
        ]
    }
    
    execution_result = await agile_manager.execute_sprint(sprint.sprint_id, execution_data)
    print(f"Completion: {execution_result['completion_percentage']}%")
    print(f"Velocity: {execution_result['velocity']} points/day")
    print(f"Burndown Status: {execution_result['burndown_status']}")
    
    # 5. Sprint Review
    print("\n=== Sprint Review ===")
    review_result = await agile_manager.conduct_sprint_review(sprint.sprint_id)
    print(f"Completion Rate: {review_result['completion_rate']}%")
    print(f"Clinical Team Satisfaction: {review_result['stakeholder_feedback']['clinical_team']['satisfaction_score']}/5")
    
    # 6. Sprint Retrospective
    print("\n=== Sprint Retrospective ===")
    retrospective_result = await agile_manager.conduct_sprint_retrospective(sprint.sprint_id)
    print(f"Team Satisfaction: {retrospective_result['team_satisfaction']}/5")
    print(f"Action Items: {retrospective_result['action_items_count']}")
    
    # 7. Calculate Metrics
    print("\n=== Agile Metrics ===")
    metrics = await agile_manager.calculate_agile_metrics()
    print(f"Velocity: {metrics.velocity:.1f} points/sprint")
    print(f"Cycle Time: {metrics.cycle_time:.1f} days")
    print(f"Customer Satisfaction: {metrics.customer_satisfaction}/5")
    print(f"Quality Score: {metrics.quality_metrics['test_coverage']}%")
    
    # 8. Continuous Improvement
    print("\n=== Continuous Improvement ===")
    initiative = ImprovementInitiative(
        initiative_id="CI_001",
        title="Automated Testing Enhancement",
        description="Implement comprehensive automated testing for clinical AI",
        methodology=AgileMethodology.SCRUM,
        current_sprint=3,
        total_sprints=6,
        success_metrics={"test_coverage": 95.0, "defect_rate": 2.0},
        stakeholder_impact="High - Improves clinical safety and confidence",
        implementation_status="pilot"
    )
    
    improvement_result = await agile_manager.implement_continuous_improvement(initiative)
    print(f"Initiative: {initiative.title}")
    print(f"Progress: {improvement_result['progress_percentage']}%")
    print(f"Efficiency Improvement: {improvement_result['success_metrics']['process_efficiency']}")
    
    # 9. Dashboard
    print("\n=== Agile Dashboard ===")
    dashboard = await agile_manager.generate_agile_dashboard()
    print(f"Team Velocity: {dashboard['team_velocity']['current_velocity']} points")
    print(f"Completion Rate: {dashboard['sprint_health']['completion_rate']}%")
    print(f"Customer Satisfaction: {dashboard['customer_satisfaction']['overall_score']}/5")
    
    # 10. Export Report
    print("\n=== Exporting Agile Report ===")
    report_result = await agile_manager.export_agile_report("agile_operations_report.json")
    print(f"Report exported to: {report_result['report_file']}")
    
    return agile_manager

if __name__ == "__main__":
    asyncio.run(run_agile_demo())
