"""
Demo Environment API Endpoints
Provides REST API endpoints for demo functionality and testing
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Dict, List, Optional, Any
import json
import uuid
from datetime import datetime
import sys
import os

# Add demo modules to path
sys.path.append('demo')

from auth.demo_auth import DemoAuthManager, UserRole, Permission
from analytics.demo_analytics import DemoAnalyticsManager, DemoTracker
from backup.demo_backup import DemoBackupManager
from scenarios.medical_scenarios import ScenarioManager

app = FastAPI(
    title="Medical AI Assistant Demo API",
    description="Demo API for Medical AI Assistant system with synthetic data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
auth_manager = DemoAuthManager()
analytics_manager = DemoAnalyticsManager()
backup_manager = DemoBackupManager()
scenario_manager = ScenarioManager()
security = HTTPBearer()

# Pydantic models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    first_name: str
    last_name: str
    role: str
    permissions: List[str]

class DemoSessionRequest(BaseModel):
    scenario_id: str
    user_id: int

class FeedbackRequest(BaseModel):
    session_id: str
    feedback_score: int
    comments: Optional[str] = None

# Dependency to get current user
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    user = auth_manager.get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for demo environment"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "demo_mode": True,
        "components": {
            "database": "available",
            "analytics": "available", 
            "backup": "available",
            "auth": "available"
        }
    }

# Authentication endpoints
@app.post("/api/auth/demo/login")
async def demo_login(request: LoginRequest):
    """Demo login endpoint"""
    user = auth_manager.authenticate_user(
        request.email, 
        request.password,
        user_agent="demo-client/1.0"
    )
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Track login event
    tracker = DemoTracker(analytics_manager)
    tracker.track_page_view(user.id, user.session_token, "login_page")
    
    return {
        "access_token": user.session_token,
        "token_type": "bearer",
        "expires_in": 28800,  # 8 hours
        "user": {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions]
        }
    }

@app.post("/api/auth/demo/logout")
async def demo_logout(current_user: dict = Depends(get_current_user)):
    """Demo logout endpoint"""
    auth_manager.logout_user(current_user.session_token)
    return {"message": "Successfully logged out"}

@app.get("/api/auth/demo/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "role": current_user.role.value,
        "permissions": [p.value for p in current_user.permissions],
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }

# Demo scenarios endpoints
@app.get("/api/demo/scenarios")
async def get_demo_scenarios(current_user: dict = Depends(get_current_user)):
    """Get available demo scenarios"""
    scenarios = [
        {
            "id": "diabetes",
            "name": "Diabetes Management",
            "description": "Real-time glucose monitoring and insulin recommendations",
            "duration_minutes": 15,
            "difficulty": "intermediate",
            "patient_name": "John Smith",
            "features": [
                "glucose_trend_analysis",
                "insulin_dose_calculation", 
                "dietary_recommendations",
                "alert_system"
            ]
        },
        {
            "id": "hypertension",
            "name": "Hypertension Monitoring",
            "description": "Blood pressure tracking and cardiovascular risk assessment",
            "duration_minutes": 12,
            "difficulty": "beginner",
            "patient_name": "Emily Davis",
            "features": [
                "bp_trend_analysis",
                "medication_adherence",
                "risk_stratification", 
                "lifestyle_recommendations"
            ]
        },
        {
            "id": "chest_pain",
            "name": "Chest Pain Assessment",
            "description": "Emergency triage and cardiovascular risk evaluation",
            "duration_minutes": 10,
            "difficulty": "advanced",
            "patient_name": "Michael Johnson",
            "features": [
                "symptom_evaluation",
                "risk_stratification",
                "emergency_protocols",
                "specialist_referral"
            ]
        }
    ]
    
    return {"scenarios": scenarios}

@app.post("/api/demo/scenarios/{scenario_id}/start")
async def start_demo_scenario(
    scenario_id: str, 
    request: DemoSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Start a demo scenario"""
    
    # Create session tracking
    session_id = str(uuid.uuid4())
    
    # Initialize analytics session
    analytics_manager.start_demo_session(
        session_id=session_id,
        user_id=request.user_id,
        user_agent="demo-client/1.0"
    )
    
    # Create scenario instance
    scenario = scenario_manager.create_scenario(scenario_id, request.user_id)
    
    return {
        "session_id": session_id,
        "scenario_id": scenario_id,
        "status": "started",
        "timestamp": datetime.now().isoformat(),
        "estimated_duration": 15  # minutes
    }

@app.get("/api/demo/scenarios/{scenario_id}/data")
async def get_scenario_data(
    scenario_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get demo scenario data"""
    
    try:
        scenario_data = scenario_manager.get_scenario_data(scenario_id)
        return {
            "scenario_id": scenario_id,
            "data": scenario_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scenario data: {str(e)}")

@app.post("/api/demo/scenarios/{scenario_id}/complete")
async def complete_demo_scenario(
    scenario_id: str,
    session_id: str,
    steps_completed: int,
    total_steps: int,
    duration_minutes: float,
    current_user: dict = Depends(get_current_user)
):
    """Complete a demo scenario"""
    
    # End analytics session
    analytics_manager.end_demo_session(session_id)
    
    # Track completion
    tracker = DemoTracker(analytics_manager)
    scenario_names = {
        "diabetes": "Diabetes Management",
        "hypertension": "Hypertension Monitoring", 
        "chest_pain": "Chest Pain Assessment"
    }
    
    tracker.track_demo_completion(
        user_id=current_user.id,
        session_id=session_id,
        scenario_id=scenario_id,
        scenario_name=scenario_names.get(scenario_id, scenario_id),
        steps_completed=steps_completed,
        total_steps=total_steps,
        duration_minutes=duration_minutes
    )
    
    return {
        "status": "completed",
        "scenario_id": scenario_id,
        "session_id": session_id,
        "completion_time": datetime.now().isoformat(),
        "message": "Demo scenario completed successfully"
    }

# Analytics endpoints
@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard(current_user: dict = Depends(get_current_user)):
    """Get demo analytics dashboard data"""
    
    # Check permission
    if not auth_manager.check_permission(current_user, Permission.VIEW_ANALYTICS):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    dashboard_data = analytics_manager.get_demo_dashboard_data()
    return dashboard_data

@app.get("/api/analytics/user/{user_id}")
async def get_user_analytics(
    user_id: int,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics for specific user"""
    
    # Check permissions
    if current_user.role == UserRole.PATIENT and current_user.id != user_id:
        raise HTTPException(status_code=403, detail="Cannot view other user's analytics")
    
    user_analytics = analytics_manager.get_user_analytics(user_id=user_id)
    return user_analytics

@app.post("/api/analytics/track")
async def track_demo_event(
    event_type: str,
    component: str,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
    current_user: dict = Depends(get_current_user)
):
    """Track custom demo event"""
    
    from analytics.demo_analytics import UserAction
    
    action = UserAction(
        user_id=current_user.id,
        session_id=session_id or str(uuid.uuid4()),
        action_type=event_type,
        component=component,
        timestamp=datetime.now(),
        metadata=metadata
    )
    
    analytics_manager.track_user_action(action)
    return {"status": "tracked", "timestamp": datetime.now().isoformat()}

# Backup and recovery endpoints
@app.post("/api/demo/backup/create")
async def create_demo_backup(
    description: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Create a demo environment backup"""
    
    # Check permissions
    if not auth_manager.check_permission(current_user, Permission.DEMO_ADMIN):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    backup = backup_manager.create_full_backup(description or "Manual demo backup")
    
    return {
        "status": "backup_created",
        "backup_id": backup.backup_id,
        "timestamp": backup.timestamp.isoformat(),
        "size_bytes": backup.size_bytes,
        "components": backup.components
    }

@app.post("/api/demo/backup/restore/{backup_id}")
async def restore_demo_backup(
    backup_id: str,
    components: Optional[List[str]] = None,
    current_user: dict = Depends(get_current_user)
):
    """Restore demo environment from backup"""
    
    # Check permissions
    if not auth_manager.check_permission(current_user, Permission.DEMO_ADMIN):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    success = backup_manager.restore_backup(backup_id, components)
    
    if success:
        return {
            "status": "restore_completed",
            "backup_id": backup_id,
            "timestamp": datetime.now().isoformat(),
            "components_restored": components or ["database", "config", "analytics", "models"]
        }
    else:
        raise HTTPException(status_code=500, detail="Backup restore failed")

@app.get("/api/demo/backup/status")
async def get_backup_status(current_user: dict = Depends(get_current_user)):
    """Get backup system status"""
    
    if not auth_manager.check_permission(current_user, Permission.DEMO_ADMIN):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    readiness_report = backup_manager.get_demo_readiness_report()
    return readiness_report

@app.post("/api/demo/reset")
async def reset_demo_environment(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Reset demo environment to clean state"""
    
    # Check permissions
    if not auth_manager.check_permission(current_user, Permission.DEMO_ADMIN):
        raise HTTPException(status_code=403, detail="Admin permissions required")
    
    # Create backup before reset
    pre_reset_backup = backup_manager.create_demo_reset_backup()
    
    # Schedule reset tasks
    background_tasks.add_task(reset_demo_databases)
    
    return {
        "status": "reset_initiated",
        "pre_reset_backup_id": pre_reset_backup.backup_id,
        "timestamp": datetime.now().isoformat(),
        "message": "Demo environment reset in progress"
    }

async def reset_demo_databases():
    """Reset demo databases (background task)"""
    try:
        # Reset main demo database
        if os.path.exists("demo/demo.db"):
            os.remove("demo/demo.db")
            
        # Reset analytics database
        if os.path.exists("demo_analytics.db"):
            os.remove("demo_analytics.db")
            
        # Recreate databases
        from database.populate_demo_data import DemoDatabasePopulator
        populator = DemoDatabasePopulator("demo/demo.db")
        populator.run_population()
        
        # Reinitialize demo users
        auth_manager.initialize_demo_users()
        
        print("Demo environment reset completed")
        
    except Exception as e:
        print(f"Demo reset failed: {e}")

# Demo configuration endpoints
@app.get("/api/demo/config")
async def get_demo_config(current_user: dict = Depends(get_current_user)):
    """Get demo configuration"""
    
    config = {
        "demo_mode": True,
        "performance_mode": "fast",
        "analytics_enabled": True,
        "backup_enabled": True,
        "features": {
            "synthetic_data": True,
            "demo_scenarios": True,
            "analytics": True,
            "backup_recovery": True
        },
        "demo_users": {
            "admin": "admin@demo.medai.com",
            "nurse": "nurse.jones@demo.medai.com", 
            "patient": "patient.smith@demo.medai.com"
        }
    }
    
    return config

@app.post("/api/demo/feedback")
async def submit_demo_feedback(
    feedback: FeedbackRequest,
    current_user: dict = Depends(get_current_user)
):
    """Submit demo feedback"""
    
    # End session with feedback
    analytics_manager.end_demo_session(feedback.session_id, feedback.feedback_score)
    
    return {
        "status": "feedback_submitted",
        "timestamp": datetime.now().isoformat(),
        "feedback_score": feedback.feedback_score
    }

# Demo management endpoints
@app.get("/api/demo/status")
async def get_demo_status(current_user: dict = Depends(get_current_user)):
    """Get overall demo environment status"""
    
    # Verify demo state
    verification = backup_manager.verify_demo_state()
    
    # Get active sessions
    active_sessions = analytics_manager._get_active_sessions()
    
    # Get latest backup info
    latest_backup = backup_manager._get_latest_backup()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "demo_ready": verification["all_tests_passed"],
        "verification_results": verification,
        "active_sessions": len(active_sessions),
        "latest_backup": {
            "backup_id": latest_backup.backup_id if latest_backup else None,
            "timestamp": latest_backup.timestamp.isoformat() if latest_backup else None
        },
        "system_health": {
            "database": verification["database_accessible"],
            "analytics": verification["analytics_working"],
            "backup_system": verification["models_loaded"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")