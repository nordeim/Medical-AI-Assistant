# Backend Development Plan - Medical AI Assistant

**Document Version:** 1.0  
**Created:** November 4, 2025  
**Component:** FastAPI Backend, AI Agents, RAG System  
**Phase Coverage:** Phases 1-2 Core Implementation  

---

## 🎯 Backend Architecture Overview

The Medical AI Assistant backend is built on FastAPI 0.109.0 with Python 3.9+, implementing a microservices architecture optimized for healthcare applications. The backend serves as the core intelligence layer, orchestrating AI agents, managing patient data, and ensuring HIPAA compliance.

### 🏗️ Core Technology Stack
```yaml
Framework: FastAPI 0.109.0
Language: Python 3.9+
Database: PostgreSQL + SQLAlchemy ORM
Cache: Redis (sessions & performance)
AI/ML: PyTorch, Transformers, LangChain v1.0
Vector DB: Chroma (medical knowledge RAG)
Message Queue: Celery with Redis broker
Monitoring: Prometheus metrics integration
```

---

## 📂 Backend Directory Structure

### Core Application Structure
```
backend/
├── app/                           # Main Application Core
│   ├── main.py                   # 🎯 FastAPI Application Entry Point
│   ├── config.py                 # 🎯 Configuration Management
│   ├── database.py               # 🎯 Database Connection & Session Management
│   └── dependencies.py           # 🎯 FastAPI Dependency Injection
│
├── agent/                         # AI Agent Orchestration System
│   ├── orchestrator.py           # 🎯 Main AI Agent Coordinator
│   ├── par_generator.py          # 🎯 Patient Assessment Report Generator
│   ├── config.py                 # Agent-specific configuration
│   ├── callbacks/                # Agent callback handlers
│   │   ├── audit_callback.py     # Audit trail logging
│   │   ├── safety_callback.py    # Safety monitoring
│   │   └── streaming_callback.py # Real-time streaming
│   ├── safety/                   # Medical Safety Systems
│   │   ├── content_filter.py     # Content safety filtering
│   │   └── red_flag_rules.py     # Emergency detection rules
│   └── tools/                    # Clinical Reasoning Tools
│       ├── ehr_connector.py      # EHR system integration
│       ├── rag_retrieval.py      # Medical knowledge retrieval
│       └── red_flag_detector.py  # Emergency symptom detection
│
├── api/                          # API Route Definitions
│   ├── auth.py                   # 🎯 Authentication endpoints
│   ├── health.py                 # 🎯 Health check endpoints
│   ├── messages.py               # 🎯 Chat message endpoints
│   ├── pars.py                   # 🎯 Patient assessment endpoints
│   └── sessions.py               # 🎯 Patient session endpoints
│
├── models/                       # Data Models & Schemas
│   ├── audit_log.py              # Audit trail models
│   ├── message.py                # Chat message models
│   ├── par.py                    # Patient assessment models
│   ├── session.py                # Patient session models
│   └── user.py                   # User management models
│
├── rag/                          # Retrieval-Augmented Generation
│   ├── config.py                 # RAG system configuration
│   ├── document_processor.py     # Medical document processing
│   ├── embeddings.py             # Medical text embeddings
│   ├── retriever.py              # Knowledge retrieval system
│   └── vector_store.py           # Chroma vector database
│
├── services/                     # Business Logic Services
│   ├── audit_service.py          # Audit logging service
│   ├── message_service.py        # Message processing service
│   ├── par_service.py            # Assessment generation service
│   └── session_service.py        # Session management service
│
└── websocket/                    # Real-time Communication
    ├── handlers.py               # WebSocket event handlers
    └── manager.py                # Connection management
```

---

## 🚀 Implementation Roadmap

### Phase 1: Core Backend Foundation (Weeks 1-4)

#### Week 1: FastAPI Application Setup
**Objectives:**
- Initialize FastAPI application structure
- Configure development environment
- Implement basic routing and middleware

**Key Tasks:**
```python
# app/main.py - FastAPI Application Entry Point
from fastapi import FastAPI, middleware
from fastapi.middleware.cors import CORSMiddleware
from app.api import auth, health, messages, sessions, pars
from app.database import engine, Base
from app.config import settings

app = FastAPI(
    title="Medical AI Assistant API",
    description="HIPAA-compliant healthcare AI platform",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])
app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["sessions"])
app.include_router(messages.router, prefix="/api/v1/messages", tags=["messages"])
app.include_router(pars.router, prefix="/api/v1/pars", tags=["assessments"])
```

**Configuration Management:**
```python
# app/config.py - Environment Configuration
from pydantic import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str
    REDIS_URL: str
    
    # Security Configuration
    SECRET_KEY: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI Services Configuration
    OPENAI_API_KEY: str
    LANGCHAIN_API_KEY: str
    VECTOR_DB_PATH: str = "./data/chroma_db"
    
    # Application Configuration
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"

settings = Settings()
```

#### Week 2: Database Integration & Models
**Objectives:**
- Set up PostgreSQL database connection
- Implement SQLAlchemy models
- Create database migration system

**Database Setup:**
```python
# app/database.py - Database Connection Management
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=20,
    max_overflow=30
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Core Models Implementation:**
```python
# models/user.py - User Management Model
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from app.database import Base
import enum

class UserRole(str, enum.Enum):
    PATIENT = "patient"
    NURSE = "nurse"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
```

#### Week 3: Authentication & Security
**Objectives:**
- Implement JWT-based authentication
- Set up role-based access control (RBAC)
- Configure password hashing and validation

**Authentication Implementation:**
```python
# auth/jwt.py - JWT Token Management
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt
```

#### Week 4: API Routes & Basic Services
**Objectives:**
- Implement core API endpoints
- Set up basic service layer
- Configure request/response validation

**API Route Implementation:**
```python
# api/sessions.py - Patient Session Endpoints
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.session import PatientSession
from app.schemas.session import SessionCreate, SessionResponse
from app.services.session_service import SessionService

router = APIRouter()

@router.post("/", response_model=SessionResponse)
async def create_session(
    session_data: SessionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new patient consultation session."""
    session_service = SessionService(db)
    return await session_service.create_session(session_data, current_user.id)

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve a specific patient session."""
    session_service = SessionService(db)
    session = await session_service.get_session(session_id, current_user.id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
```

---

### Phase 2: AI Agent Development (Weeks 5-8)

#### Week 5: LangChain Agent Foundation
**Objectives:**
- Set up LangChain agent framework
- Implement basic agent orchestration
- Configure medical reasoning tools

**Agent Orchestrator Setup:**
```python
# agent/orchestrator.py - Main AI Agent Coordinator
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from app.agent.tools.rag_retrieval import MedicalRAGTool
from app.agent.tools.red_flag_detector import RedFlagDetector
from app.agent.safety.content_filter import ContentFilter

class MedicalAIOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,  # Low temperature for medical accuracy
            api_key=settings.OPENAI_API_KEY
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.tools = self._setup_tools()
        self.agent = self._create_agent()
        self.safety_filter = ContentFilter()
        
    def _setup_tools(self) -> List[Tool]:
        return [
            Tool(
                name="medical_knowledge",
                description="Query medical knowledge base for clinical information",
                func=MedicalRAGTool().query
            ),
            Tool(
                name="red_flag_detector",
                description="Detect emergency symptoms and red flags",
                func=RedFlagDetector().detect
            ),
            Tool(
                name="assessment_generator",
                description="Generate patient assessment reports",
                func=self.generate_assessment
            )
        ]
    
    async def process_message(self, message: str, session_id: str) -> str:
        """Process patient message and generate AI response."""
        # Safety pre-filtering
        if not self.safety_filter.is_safe_input(message):
            return "I'm sorry, I can't respond to that message. Please rephrase your question."
        
        # Red flag detection
        red_flags = await self.tools[1].func(message)
        if red_flags:
            return await self._handle_emergency(red_flags, session_id)
        
        # Regular AI processing
        response = await self.agent.arun(
            input=message,
            session_id=session_id
        )
        
        # Safety post-filtering
        return self.safety_filter.filter_response(response)
```

#### Week 6: RAG System Implementation
**Objectives:**
- Implement medical knowledge vector database
- Set up document processing pipeline
- Configure knowledge retrieval system

**RAG System Setup:**
```python
# rag/vector_store.py - Chroma Vector Database Integration
import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from app.rag.document_processor import MedicalDocumentProcessor

class MedicalVectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.client = chromadb.PersistentClient(
            path=settings.VECTOR_DB_PATH,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self.client.get_or_create_collection(
            name="medical_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.vectorstore = Chroma(
            client=self.client,
            collection_name="medical_knowledge",
            embedding_function=self.embeddings
        )
    
    async def add_documents(self, documents: List[str], metadata: List[dict]):
        """Add medical documents to the vector store."""
        processed_docs = MedicalDocumentProcessor().process_documents(documents)
        await self.vectorstore.aadd_documents(processed_docs, metadatas=metadata)
    
    async def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant medical information."""
        results = await self.vectorstore.asimilarity_search(
            query=query,
            k=k,
            filter={"source": "medical_guidelines"}
        )
        return [doc.page_content for doc in results]
```

#### Week 7: Safety & Red Flag Detection
**Objectives:**
- Implement medical safety systems
- Set up emergency detection rules
- Configure content filtering

**Red Flag Detection System:**
```python
# agent/tools/red_flag_detector.py - Emergency Symptom Detection
import re
from typing import List, Dict
from app.models.safety_filter_log import SafetyFilterLog

class RedFlagDetector:
    def __init__(self):
        self.emergency_patterns = self._load_emergency_patterns()
        self.severity_levels = {
            "CRITICAL": ["chest pain", "difficulty breathing", "severe bleeding"],
            "HIGH": ["severe pain", "high fever", "vision loss"],
            "MEDIUM": ["persistent pain", "moderate fever", "nausea"]
        }
    
    async def detect(self, message: str) -> Dict[str, any]:
        """Detect red flags in patient messages."""
        red_flags = []
        severity = "LOW"
        
        message_lower = message.lower()
        
        for level, patterns in self.severity_levels.items():
            for pattern in patterns:
                if pattern in message_lower:
                    red_flags.append({
                        "pattern": pattern,
                        "severity": level,
                        "detected_text": self._extract_context(message, pattern)
                    })
                    if level == "CRITICAL":
                        severity = "CRITICAL"
                    elif level == "HIGH" and severity != "CRITICAL":
                        severity = "HIGH"
        
        if red_flags:
            await self._log_red_flag_detection(message, red_flags, severity)
        
        return {
            "has_red_flags": bool(red_flags),
            "red_flags": red_flags,
            "severity": severity,
            "requires_escalation": severity in ["CRITICAL", "HIGH"]
        }
```

#### Week 8: Patient Assessment Generation
**Objectives:**
- Implement PAR (Patient Assessment Report) generation
- Set up clinical reasoning framework
- Configure assessment validation

**PAR Generation System:**
```python
# agent/par_generator.py - Patient Assessment Report Generator
from langchain.prompts import PromptTemplate
from app.models.par import PatientAssessmentReport
from app.rag.retriever import MedicalKnowledgeRetriever

class PARGenerator:
    def __init__(self):
        self.knowledge_retriever = MedicalKnowledgeRetriever()
        self.assessment_prompt = self._load_assessment_prompt()
    
    async def generate_assessment(self, session_data: Dict) -> PatientAssessmentReport:
        """Generate comprehensive patient assessment report."""
        
        # Retrieve relevant medical knowledge
        symptoms = session_data.get("symptoms", [])
        relevant_knowledge = await self.knowledge_retriever.retrieve_guidelines(symptoms)
        
        # Extract key information
        assessment_data = {
            "symptoms": symptoms,
            "medical_history": session_data.get("medical_history", []),
            "current_medications": session_data.get("medications", []),
            "relevant_guidelines": relevant_knowledge
        }
        
        # Generate assessment using LLM
        assessment_text = await self._generate_assessment_text(assessment_data)
        
        # Create structured PAR
        par = PatientAssessmentReport(
            session_id=session_data["session_id"],
            patient_id=session_data["patient_id"],
            symptoms_summary=self._summarize_symptoms(symptoms),
            risk_assessment=self._assess_risk_level(assessment_data),
            recommendations=self._generate_recommendations(assessment_data),
            triage_priority=self._determine_triage_priority(assessment_data),
            follow_up_required=self._requires_follow_up(assessment_data),
            generated_text=assessment_text
        )
        
        return par
```

---

### Phase 3: WebSocket & Real-time Communication (Weeks 9-10)

#### Week 9: WebSocket Infrastructure
**Objectives:**
- Implement WebSocket connection management
- Set up real-time message handling
- Configure streaming AI responses

**WebSocket Manager:**
```python
# websocket/manager.py - Connection Management
from fastapi import WebSocket
from typing import Dict, List
import json
import asyncio

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_connections: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        await websocket.accept()
        connection_id = f"{user_id}_{session_id}"
        self.active_connections[connection_id] = websocket
        
        if session_id not in self.session_connections:
            self.session_connections[session_id] = []
        self.session_connections[session_id].append(connection_id)
    
    def disconnect(self, session_id: str, user_id: str):
        connection_id = f"{user_id}_{session_id}"
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        if session_id in self.session_connections:
            self.session_connections[session_id].remove(connection_id)
    
    async def send_personal_message(self, message: str, session_id: str, user_id: str):
        connection_id = f"{user_id}_{session_id}"
        if connection_id in self.active_connections:
            await self.active_connections[connection_id].send_text(message)
    
    async def broadcast_to_session(self, message: str, session_id: str):
        if session_id in self.session_connections:
            for connection_id in self.session_connections[session_id]:
                if connection_id in self.active_connections:
                    await self.active_connections[connection_id].send_text(message)
```

#### Week 10: Streaming AI Responses
**Objectives:**
- Implement token-by-token streaming
- Set up real-time safety monitoring
- Configure response optimization

**Streaming Handler:**
```python
# websocket/handlers.py - WebSocket Event Handlers
from app.agent.orchestrator import MedicalAIOrchestrator
from app.websocket.manager import ConnectionManager
import asyncio

class ChatHandler:
    def __init__(self):
        self.ai_orchestrator = MedicalAIOrchestrator()
        self.connection_manager = ConnectionManager()
    
    async def handle_chat_message(self, websocket: WebSocket, data: dict):
        """Handle incoming chat messages with streaming responses."""
        session_id = data.get("session_id")
        message = data.get("message")
        user_id = data.get("user_id")
        
        try:
            # Process message with AI agent
            async for chunk in self.ai_orchestrator.stream_response(message, session_id):
                response_data = {
                    "type": "ai_response_chunk",
                    "session_id": session_id,
                    "chunk": chunk,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.connection_manager.send_personal_message(
                    json.dumps(response_data),
                    session_id,
                    user_id
                )
                
                # Small delay for smooth streaming
                await asyncio.sleep(0.01)
            
            # Send completion signal
            completion_data = {
                "type": "ai_response_complete",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.connection_manager.send_personal_message(
                json.dumps(completion_data),
                session_id,
                user_id
            )
            
        except Exception as e:
            error_data = {
                "type": "error",
                "message": "An error occurred processing your message",
                "session_id": session_id
            }
            
            await self.connection_manager.send_personal_message(
                json.dumps(error_data),
                session_id,
                user_id
            )
```

---

## 🔧 Development Guidelines & Best Practices

### Code Quality Standards
```python
# Code formatting and linting
black backend/                    # Code formatting
isort backend/                    # Import sorting
flake8 backend/                   # Linting
mypy backend/                     # Type checking

# Testing requirements
pytest backend/tests/ --cov=80%   # Minimum 80% coverage
pytest backend/tests/security/    # Security testing
pytest backend/tests/compliance/  # HIPAA compliance testing
```

### Performance Optimization
```python
# Database optimization
- Connection pooling (20 connections, 30 overflow)
- Query optimization with SQLAlchemy
- Redis caching for session data
- Async database operations

# AI Performance optimization
- Model caching and reuse
- Batch processing for multiple requests
- Streaming responses for better UX
- Timeout handling (30 seconds max)
```

### Security Implementation
```python
# HIPAA compliance measures
- PHI field-level encryption
- Audit logging for all operations
- Secure session management
- Input validation and sanitization
- Output filtering for medical safety
```

---

## 🧪 Testing Strategy

### Unit Testing (80%+ Coverage)
```python
# test_orchestrator.py - AI Agent Testing
import pytest
from app.agent.orchestrator import MedicalAIOrchestrator

@pytest.mark.asyncio
async def test_process_safe_message():
    orchestrator = MedicalAIOrchestrator()
    response = await orchestrator.process_message(
        "I have a mild headache",
        "test_session_123"
    )
    assert response is not None
    assert "emergency" not in response.lower()

@pytest.mark.asyncio
async def test_red_flag_detection():
    orchestrator = MedicalAIOrchestrator()
    response = await orchestrator.process_message(
        "I'm having severe chest pain",
        "test_session_456"
    )
    assert "seek immediate medical attention" in response.lower()
```

### Integration Testing
```python
# test_api_integration.py - API Integration Testing
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_session():
    response = client.post(
        "/api/v1/sessions/",
        json={"patient_id": "test_patient_123"},
        headers={"Authorization": "Bearer test_token"}
    )
    assert response.status_code == 200
    assert "session_id" in response.json()
```

---

## 📊 Monitoring & Observability

### Health Checks
```python
# api/health.py - Comprehensive Health Checks
from fastapi import APIRouter, status, Depends
from app.services.health_service import HealthService

router = APIRouter()

@router.get("/")
async def basic_health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/detailed")
async def detailed_health_check(health_service: HealthService = Depends()):
    return {
        "database": await health_service.check_database(),
        "redis": await health_service.check_redis(),
        "ai_models": await health_service.check_ai_models(),
        "vector_store": await health_service.check_vector_store()
    }
```

### Performance Metrics
```python
# Prometheus metrics integration
from prometheus_client import Counter, Histogram, generate_latest

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'API request latency')
AI_PROCESSING_TIME = Histogram('ai_processing_duration_seconds', 'AI processing time')
```

---

## 🚀 Deployment Configuration

### Docker Configuration
```dockerfile
# Dockerfile for Backend
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration
```yaml
# docker-compose.yml - Development Setup
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/medical_ai
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:13
    environment:
      POSTGRES_DB: medical_ai
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6-alpine
```

---

## 📚 Documentation Requirements

### API Documentation
- OpenAPI/Swagger specifications
- Endpoint documentation with examples
- Authentication and authorization guides
- Error handling documentation

### Code Documentation
- Comprehensive docstrings for all functions
- Type hints for better IDE support
- Architecture decision records (ADRs)
- Development setup guides

---

**Document Status:** Implementation Ready  
**Last Updated:** November 4, 2025  
**Review Schedule:** Weekly during development phases  
**Classification:** Technical Implementation Guide