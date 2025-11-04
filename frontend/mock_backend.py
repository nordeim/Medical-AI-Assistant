"""
Mock Backend Server for Medical AI Nurse Dashboard Testing

This script provides a simple FastAPI server that simulates the backend API
for testing the nurse dashboard frontend integration.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import asyncio
import json

app = FastAPI(title="Medical AI Mock Backend")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data store
users_db = {}
pars_db = {}
sessions_db = {}

# WebSocket connections
active_connections: List[WebSocket] = []

# Models
class LoginRequest(BaseModel):
    email: str
    password: str

class User(BaseModel):
    id: str
    email: str
    role: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    created_at: str
    updated_at: str

class PARQueueItem(BaseModel):
    id: str
    session_id: str
    patient_id: str
    patient_name: str
    chief_complaint: str
    risk_level: str
    urgency: str
    has_red_flags: bool
    created_at: str
    wait_time_minutes: int
    status: str = "pending"

# Initialize test nurse user
test_nurse = User(
    id=str(uuid.uuid4()),
    email="nurse@test.com",
    role="nurse",
    first_name="Test",
    last_name="Nurse",
    is_active=True,
    created_at=datetime.now().isoformat(),
    updated_at=datetime.now().isoformat()
)
users_db["nurse@test.com"] = test_nurse

# Generate sample PARs
def generate_sample_pars():
    complaints = [
        ("Severe chest pain and shortness of breath", "high", "immediate", True),
        ("Persistent headache and dizziness", "medium", "urgent", False),
        ("Mild fever and cough", "low", "routine", False),
        ("Abdominal pain and nausea", "medium", "urgent", False),
        ("Difficulty breathing", "critical", "immediate", True),
    ]
    
    for i, (complaint, risk, urgency, red_flags) in enumerate(complaints):
        par_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        patient_id = str(uuid.uuid4())
        
        created_at = datetime.now() - timedelta(minutes=30 - i * 5)
        
        par = {
            "id": par_id,
            "session_id": session_id,
            "patient_id": patient_id,
            "patient_name": f"Patient {chr(65 + i)}",
            "chief_complaint": complaint,
            "risk_level": risk,
            "urgency": urgency,
            "has_red_flags": red_flags,
            "created_at": created_at.isoformat(),
            "wait_time_minutes": 30 - i * 5,
            "status": "pending",
            "symptoms": [
                complaint.split(" and ")[0],
                complaint.split(" and ")[1] if " and " in complaint else "Other symptoms"
            ],
            "red_flags": ["Critical vitals", "Immediate attention required"] if red_flags else [],
            "recommendations": [
                f"Evaluate {complaint.lower()}",
                "Monitor vital signs",
                "Consider immediate intervention" if urgency == "immediate" else "Schedule follow-up"
            ],
            "differential_diagnosis": [
                "Condition A",
                "Condition B",
                "Condition C"
            ],
            "rag_sources": [
                {
                    "id": str(uuid.uuid4()),
                    "source_type": "guideline",
                    "title": "Clinical Practice Guideline",
                    "content": "Relevant medical guideline content...",
                    "relevance_score": 0.89,
                    "reference_url": "https://example.com/guideline"
                }
            ],
            "guideline_references": [
                {
                    "guideline_id": str(uuid.uuid4()),
                    "title": "Emergency Medicine Protocol",
                    "section": "Chest Pain Assessment",
                    "recommendation": "Immediate ECG and troponin",
                    "evidence_level": "Level A"
                }
            ],
            "confidence_scores": {
                "urgency": 0.92,
                "risk_level": 0.88,
                "diagnosis": 0.75
            }
        }
        
        pars_db[par_id] = par
        sessions_db[session_id] = {
            "id": session_id,
            "patient_id": patient_id,
            "status": "completed",
            "created_at": created_at.isoformat()
        }

generate_sample_pars()

# Endpoints

@app.get("/health")
async def health_check():
    return {
        "success": True,
        "data": {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    }

@app.post("/auth/login")
async def login(credentials: LoginRequest):
    if credentials.email == "nurse@test.com" and credentials.password == "test123":
        return {
            "success": True,
            "data": {
                "user": test_nurse.dict(),
                "access_token": "mock_token_" + str(uuid.uuid4())
            }
        }
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/auth/me")
async def get_current_user():
    return {
        "success": True,
        "data": test_nurse.dict()
    }

@app.get("/pars/queue")
async def get_par_queue(
    limit: int = 100,
    offset: int = 0,
    urgency: Optional[str] = None,
    risk_level: Optional[str] = None,
    has_red_flags: Optional[bool] = None
):
    queue = list(pars_db.values())
    
    # Apply filters
    if urgency:
        queue = [p for p in queue if p["urgency"] == urgency]
    if risk_level:
        queue = [p for p in queue if p["risk_level"] == risk_level]
    if has_red_flags is not None:
        queue = [p for p in queue if p["has_red_flags"] == has_red_flags]
    
    # Count stats
    urgent_count = len([p for p in pars_db.values() if p["urgency"] == "urgent"])
    immediate_count = len([p for p in pars_db.values() if p["urgency"] == "immediate"])
    red_flag_count = len([p for p in pars_db.values() if p["has_red_flags"]])
    
    return {
        "success": True,
        "data": {
            "queue": queue[offset:offset + limit],
            "total": len(queue),
            "urgent_count": urgent_count,
            "immediate_count": immediate_count,
            "red_flag_count": red_flag_count
        }
    }

@app.get("/pars/search")
async def search_pars(
    q: str = "",
    urgency: Optional[List[str]] = None,
    risk_level: Optional[List[str]] = None,
    has_red_flags: Optional[bool] = None
):
    results = list(pars_db.values())
    
    # Text search
    if q:
        results = [
            p for p in results
            if q.lower() in p["chief_complaint"].lower() or
               q.lower() in p.get("patient_name", "").lower()
        ]
    
    # Filters
    if urgency:
        results = [p for p in results if p["urgency"] in urgency]
    if risk_level:
        results = [p for p in results if p["risk_level"] in risk_level]
    if has_red_flags is not None:
        results = [p for p in results if p["has_red_flags"] == has_red_flags]
    
    return {
        "success": True,
        "data": results
    }

@app.get("/pars/{par_id}")
async def get_par(par_id: str):
    if par_id not in pars_db:
        raise HTTPException(status_code=404, detail="PAR not found")
    
    return {
        "success": True,
        "data": pars_db[par_id]
    }

@app.post("/pars/{par_id}/review")
async def review_par(par_id: str, action: Dict[str, Any]):
    if par_id not in pars_db:
        raise HTTPException(status_code=404, detail="PAR not found")
    
    par = pars_db[par_id]
    par["status"] = "reviewed"
    par["review_status"] = action.get("action")
    par["nurse_notes"] = action.get("notes")
    par["reviewed_by"] = test_nurse.id
    
    # Broadcast update via WebSocket
    await broadcast_message({
        "type": "par_update",
        "payload": {"par_id": par_id, "status": "reviewed"},
        "timestamp": datetime.now().isoformat()
    })
    
    return {
        "success": True,
        "data": par
    }

@app.get("/nurse/analytics")
async def get_analytics():
    total_pars = len(pars_db)
    approved_count = len([p for p in pars_db.values() if p.get("review_status") == "approved"])
    override_count = len([p for p in pars_db.values() if p.get("review_status") == "override"])
    
    # Generate hourly distribution
    pars_by_hour = []
    for hour in range(24):
        count = max(0, 15 - abs(hour - 10) * 2)  # Peak at 10 AM
        pars_by_hour.append({"hour": hour, "count": count})
    
    return {
        "success": True,
        "data": {
            "total_pars_today": total_pars,
            "total_pars_week": total_pars * 5,
            "avg_wait_time_minutes": 15,
            "avg_review_time_minutes": 8,
            "approval_rate": 0.75 if total_pars > 0 else 0,
            "override_rate": 0.25 if total_pars > 0 else 0,
            "red_flag_rate": len([p for p in pars_db.values() if p["has_red_flags"]]) / max(total_pars, 1),
            "high_risk_rate": len([p for p in pars_db.values() if p["risk_level"] in ["high", "critical"]]) / max(total_pars, 1),
            "pars_by_urgency": {
                "routine": len([p for p in pars_db.values() if p["urgency"] == "routine"]),
                "urgent": len([p for p in pars_db.values() if p["urgency"] == "urgent"]),
                "immediate": len([p for p in pars_db.values() if p["urgency"] == "immediate"])
            },
            "pars_by_hour": pars_by_hour,
            "top_complaints": [
                {"complaint": "Chest pain", "count": 2},
                {"complaint": "Headache", "count": 1},
                {"complaint": "Abdominal pain", "count": 1},
                {"complaint": "Respiratory distress", "count": 1}
            ]
        }
    }

# WebSocket
async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps(message))
        except:
            pass

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    # Send connection confirmation
    await websocket.send_text(json.dumps({
        "type": "connection",
        "payload": {"status": "connected"},
        "timestamp": datetime.now().isoformat()
    }))
    
    try:
        # Simulate periodic queue updates
        async def send_updates():
            while True:
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({
                    "type": "queue_update",
                    "payload": {"pending_count": len(pars_db)},
                    "timestamp": datetime.now().isoformat()
                }))
        
        update_task = asyncio.create_task(send_updates())
        
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle session join
            if message.get("type") == "session_join":
                await websocket.send_text(json.dumps({
                    "type": "session_update",
                    "payload": {"status": "joined"},
                    "timestamp": datetime.now().isoformat()
                }))
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        update_task.cancel()
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Mock Medical AI Backend Server")
    print("="*60)
    print("\nTest Credentials:")
    print("  Email: nurse@test.com")
    print("  Password: test123")
    print("\nServer starting on http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
