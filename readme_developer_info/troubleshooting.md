# Medical AI Assistant - Developer Troubleshooting Guide

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Docker & Container Issues](#docker--container-issues)
3. [Database & Cache Problems](#database--cache-problems)
4. [Backend API Issues](#backend-api-issues)
5. [Frontend Development Issues](#frontend-development-issues)
6. [Model & AI Service Issues](#model--ai-service-issues)
7. [WebSocket & Real-time Communication](#websocket--real-time-communication)
8. [Performance Issues](#performance-issues)
9. [Security & Compliance Issues](#security--compliance-issues)
10. [Log Analysis & Monitoring](#log-analysis--monitoring)
11. [Common Error Patterns](#common-error-patterns)
12. [Recovery Procedures](#recovery-procedures)
13. [Getting Help](#getting-help)

---

## Development Environment Setup

### Python Environment Issues

#### Problem: Poetry/Virtual Environment Conflicts

```bash
# Error: Poetry lock file conflicts
poetry lock --no-update
```

**Solutions:**
```bash
# Clean Poetry environment
poetry env remove python
poetry install --no-cache

# Or use venv directly
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Environment Validation:**
```python
# backend/core/config.py - Environment checker
import os
from typing import List

def validate_environment():
    required_vars = [
        'DATABASE_URL',
        'SECRET_KEY', 
        'MODEL_PATH',
        'VECTOR_STORE_PATH'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {missing}")
    
    print("✅ All required environment variables are set")
    return True
```

#### Problem: Node.js Version Conflicts

**Check Node.js version:**
```bash
node --version  # Should be 18+ for React
npm --version
```

**Version management with nvm:**
```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Install and use correct version
nvm install 18
nvm use 18
nvm alias default 18
```

**Package installation issues:**
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Use specific npm registry if needed
npm config set registry https://registry.npmjs.org/
```

---

## Docker & Container Issues

### Container Build Failures

#### Problem: Python Dependencies Won't Install

```dockerfile
# Common issue in Dockerfile.backend
# Error: pip install fails with compiler errors

# Solution: Install build dependencies
FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

#### Problem: Frontend Build Failures

```dockerfile
# Dockerfile.frontend optimization
FROM node:22-alpine as builder

WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

### Container Communication Issues

#### Problem: Services Can't Connect

**Check Docker network:**
```bash
docker network ls
docker network inspect medical-ai-network

# Test connectivity between containers
docker exec -it medai-backend sh
# Inside container:
wget http://medai-db:5432
ping medai-db
```

**Environment variables in Docker Compose:**
```yaml
services:
  backend:
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
      - REDIS_URL=redis://user:pass@redis:6379/0
      - MODEL_PATH=/app/models/medical-llm
```

#### Problem: Health Checks Failing

```yaml
services:
  backend:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
```

**Manual health check:**
```bash
# Check container health
docker inspect medai-backend --format='{{.State.Health.Status}}'

# Test endpoints manually
curl -f http://localhost:8000/api/v1/health
curl -f http://localhost:3000
```

### Database Connection Issues

#### Problem: PostgreSQL Connection Refused

```bash
# Check if PostgreSQL is running
docker logs medai-db

# Connect to database
docker exec -it medai-db psql -U meduser -d meddb

# Test connection from backend container
docker exec -it medai-backend python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://meduser:medpass@medai-db:5432/meddb')
    result = await conn.fetchval('SELECT version()')
    print(result)
asyncio.run(test())
"
```

**PostgreSQL Configuration:**
```yaml
services:
  db:
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-meduser}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-medpass}
      POSTGRES_DB: ${POSTGRES_DB:-meddb}
      POSTGRES_INITDB_ARGS: "-E UTF8 --locale=C"
    command: postgres -c shared_preload_libraries=pg_stat_statements
```

#### Problem: Redis Connection Issues

```bash
# Test Redis connection
docker exec -it medai-redis redis-cli -a ${REDIS_PASSWORD} ping

# Check Redis logs
docker logs medai-redis

# Test from Python
docker exec -it medai-backend python -c "
import redis
r = redis.Redis(host='medai-redis', port=6379, password='${REDIS_PASSWORD}', db=0)
print(r.ping())
"
```

---

## Backend API Issues

### FastAPI Startup Problems

#### Problem: Application Won't Start

```python
# backend/main.py - Basic startup check
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical AI Assistant")

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Medical AI Assistant")
    # Add startup checks here

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Shutting down Medical AI Assistant")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

**Common startup issues:**

1. **Port already in use:**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   
   # Kill process
   kill -9 <PID>
   ```

2. **Environment variables missing:**
   ```python
   from pydantic import BaseSettings
   
   class Settings(BaseSettings):
       database_url: str
       secret_key: str
       
       class Config:
           env_file = ".env"
   
   settings = Settings()
   ```

### API Endpoint Issues

#### Problem: 500 Internal Server Error

**Add detailed error handling:**
```python
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

@app.get("/api/v1/health")
async def health_check():
    try:
        # Check database
        await db.fetch_one("SELECT 1")
        
        # Check Redis
        await redis.ping()
        
        return {"status": "healthy", "components": {"db": "ok", "redis": "ok"}}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))
```

#### Problem: WebSocket Connection Issues

```python
from fastapi import WebSocket, WebSocketDisconnect
import json

@app.websocket("/ws/chat/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message
            response = await process_chat_message(message)
            
            # Send response back
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()
```

---

## Frontend Development Issues

### React Build Issues

#### Problem: TypeScript Errors

```typescript
// frontend/src/types/api.ts - API types
export interface SessionResponse {
  session_id: string;
  token: string;
  expires_at: string;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: string;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'unhealthy';
  components: {
    db: string;
    redis: string;
    ai_model: string;
  };
}
```

**TypeScript configuration:**
```json
// frontend/tsconfig.json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "esModuleInterop": true,
    "moduleResolution": "bundler",
    "resolveJsonModule": true
  }
}
```

#### Problem: Vite Development Server Issues

```typescript
// frontend/vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, './src'),
    },
  },
})
```

### API Integration Issues

#### Problem: CORS Errors

```typescript
// frontend/src/services/api.ts
class ApiService {
  private baseURL: string;
  
  constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  }
  
  async createSession(): Promise<SessionResponse> {
    const response = await fetch(`${this.baseURL}/api/v1/session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }
}
```

#### Problem: WebSocket Connection in React

```typescript
// frontend/src/hooks/useWebSocket.ts
import { useEffect, useState, useRef } from 'react';

interface UseWebSocketProps {
  url: string;
  onMessage?: (data: any) => void;
  onError?: (error: Event) => void;
}

export function useWebSocket({ url, onMessage, onError }: UseWebSocketProps) {
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        setIsConnected(true);
        console.log('WebSocket connected');
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage?.(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };

      ws.current.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected');
        // Attempt to reconnect after 3 seconds
        setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      ws.current?.close();
    };
  }, [url, onMessage, onError]);

  const sendMessage = (message: any) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket is not connected');
    }
  };

  return { isConnected, sendMessage };
}
```

---

## Model & AI Service Issues

### LangChain Integration Problems

#### Problem: Model Loading Failures

```python
# backend/services/llm_service.py
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device
        
    async def initialize_model(self):
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                load_in_8bit=True if self.device == "cuda" else False
            )
            
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=2048,
                temperature=0.1,
                do_sample=True,
                device=0 if torch.cuda.is_available() and self.device == "cuda" else -1
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

llm_service = LLMService(os.getenv("MODEL_PATH"))
```

#### Problem: RAG Vector Store Issues

```python
# backend/services/rag_service.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, vector_store_path: str):
        self.vector_store_path = vector_store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    async def initialize_vector_store(self):
        try:
            self.vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    async def add_documents(self, documents):
        try:
            docs = self.text_splitter.split_documents(documents)
            self.vectorstore.add_documents(docs)
            self.vectorstore.persist()
            logger.info(f"Added {len(docs)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

rag_service = RAGService(os.getenv("VECTOR_STORE_PATH"))
```

### Memory & Performance Issues

#### Problem: GPU Memory Out of Memory

```python
# backend/services/model_optimization.py
import torch
import gc
import psutil
import logging

logger = logging.getLogger(__name__)

def check_memory():
    """Check available system memory"""
    memory = psutil.virtual_memory()
    logger.info(f"System memory: {memory.percent}% used")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_allocated = torch.cuda.memory_allocated(0)
        gpu_free = gpu_memory - gpu_allocated
        logger.info(f"GPU memory: {gpu_free / 1024**3:.2f}GB free")
        return gpu_free > 1024**3 * 2  # Need at least 2GB free
    return True

def clear_cache():
    """Clear memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Memory cache cleared")

class OptimizedLLMService:
    def __init__(self):
        if not check_memory():
            raise RuntimeError("Insufficient GPU memory")
    
    async def generate_with_error_handling(self, prompt):
        try:
            response = self.pipeline(prompt, max_new_tokens=500)
            return response[0]['generated_text']
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            clear_cache()
            # Retry with shorter response
            return self.pipeline(prompt, max_new_tokens=200)
```

---

## WebSocket & Real-time Communication

### Connection Issues

#### Problem: WebSocket Frequent Disconnects

```python
# backend/core/websocket_manager.py
from fastapi import WebSocket
import asyncio
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.heartbeat_interval = 30  # seconds

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")
        
        # Start heartbeat
        asyncio.create_task(self.heartbeat(session_id))

    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: dict):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(
                    json.dumps(message)
                )
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                await self.disconnect(session_id)

    async def heartbeat(self, session_id: str):
        """Send periodic heartbeats to keep connection alive"""
        while session_id in self.active_connections:
            try:
                await self.send_message(session_id, {"type": "heartbeat"})
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat failed for {session_id}: {e}")
                break

manager = ConnectionManager()
```

#### Problem: Message Queue Overflow

```python
# backend/services/message_queue.py
import asyncio
from queue import Queue, Empty
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MessageQueue:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}
        
    async def enqueue(self, session_id: str, message: Any):
        if session_id not in self.queues:
            self.queues[session_id] = asyncio.Queue(maxsize=100)
        
        try:
            await self.queues[session_id].put(message)
        except asyncio.QueueFull:
            logger.warning(f"Message queue full for {session_id}, dropping oldest")
            try:
                await self.queues[session_id].get_nowait()
                await self.queues[session_id].put(message)
            except:
                pass

    async def dequeue(self, session_id: str, timeout: float = 10.0):
        if session_id not in self.queues:
            return None
        
        try:
            return await asyncio.wait_for(
                self.queues[session_id].get(), 
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
```

---

## Performance Issues

### Database Performance

#### Problem: Slow Database Queries

```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY tablename;

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

#### Problem: N+1 Query Problem

```python
# backend/models/patient.py - Using lazy loading properly
from sqlalchemy.orm import relationship, selectinload

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    
    # Use selectinload for relationships
    sessions = relationship("Session", back_populates="patient", lazy="selectinload")

# Instead of this (N+1 problem):
# patients = session.query(Patient).all()
# for patient in patients:
#     sessions = patient.sessions  # This causes N+1

# Use this:
patients = session.query(Patient).options(selectinload(Patient.sessions)).all()
```

### API Performance

#### Problem: Slow Response Times

```python
# Add request timing middleware
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    if process_time > 1.0:  # Log slow requests
        logger.warning(f"Slow request: {request.url.path} took {process_time:.2f}s")
    
    return response

# Use caching for expensive operations
from functools import lru_cache
from redis import asyncio as aioredis

@app.lru_cache(maxsize=1000)
def get_medical_guidelines(guideline_type: str):
    # Expensive operation
    return fetch_guidelines_from_db(guideline_type)

# Redis caching example
async def get_cached_response(key: str):
    redis = aioredis.from_url(settings.redis_url)
    cached = await redis.get(key)
    return json.loads(cached) if cached else None

async def set_cached_response(key: str, value: dict, expire: int = 3600):
    redis = aioredis.from_url(settings.redis_url)
    await redis.setex(key, expire, json.dumps(value))
```

---

## Security & Compliance Issues

### Authentication & Authorization

#### Problem: JWT Token Issues

```python
# backend/core/auth.py
from jose import JWTError, jwt
from passlib.context import CryptContext
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, 
            os.getenv("SECRET_KEY"), 
            algorithm=os.getenv("ALGORITHM", "HS256")
        )
        return encoded_jwt

    def verify_token(self, token: str):
        try:
            payload = jwt.decode(
                token, 
                os.getenv("SECRET_KEY"), 
                algorithms=[os.getenv("ALGORITHM", "HS256")]
            )
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
```

#### Problem: Role-Based Access Control

```python
# backend/core/rbac.py
from enum import Enum
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

class Role(Enum):
    PATIENT = "patient"
    NURSE = "nurse"
    ADMIN = "admin"

security = HTTPBearer()

def require_role(required_role: Role):
    def role_checker(credentials: HTTPAuthorizationCredentials = Depends(security)):
        payload = auth_service.verify_token(credentials.credentials)
        user_role = Role(payload.get("role"))
        
        if user_role.value != required_role.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {required_role.value} required"
            )
        
        return payload
    
    return role_checker

# Usage
@app.get("/admin/users", dependencies=[Depends(require_role(Role.ADMIN))])
async def get_users():
    return {"users": []}
```

### HIPAA Compliance Issues

#### Problem: PHI Logging

```python
# backend/core/security.py - PHI-safe logging
import logging
import hashlib
from typing import Any

class PHI_SafeLogger:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def log_with_phi_hash(self, message: str, phi_data: dict):
        """Log with hashed PHI for audit trails"""
        phi_hash = hashlib.sha256(
            str(sorted(phi_data.items())).encode()
        ).hexdigest()[:8]
        
        self.logger.info(f"[PHI-{phi_hash}] {message}")
        
    def log_access(self, user_id: str, resource: str, action: str):
        """Log data access for audit compliance"""
        self.logger.info(
            f"AUDIT: User {user_id} {action} resource {resource} "
            f"at {datetime.utcnow().isoformat()}"
        )

# Usage
phi_logger = PHI_SafeLogger()
phi_logger.log_access(
    user_id="user_123",
    resource="patient_records",
    action="read"
)
```

---

## Log Analysis & Monitoring

### Structured Logging

```python
# backend/core/logging_config.py
import structlog
import sys
from pythonjsonlogger import jsonlogger

def setup_logging():
    logHandler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logHandler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(logHandler)
    logger.setLevel(logging.INFO)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Usage in services
import structlog

logger = structlog.get_logger()

# Add context to logs
logger.bind(session_id="session_123", user_id="user_456")
logger.info("Processing chat message", message_type="chat", word_count=150)

# Log errors with context
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", 
                error=str(e), 
                error_type=type(e).__name__,
                session_id=session_id)
```

### Performance Monitoring

```python
# backend/core/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import functools

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_WEBSOCKETS = Gauge('websocket_connections_active', 'Active WebSocket connections')
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')

def monitor_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(
                method='GET', 
                endpoint=func.__name__, 
                status='200'
            ).inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(
                method='GET', 
                endpoint=func.__name__, 
                status='500'
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
    return wrapper

# Health check endpoint for monitoring
@app.get("/metrics")
async def metrics():
    return generate_latest()

@app.get("/api/v1/health/detailed")
async def detailed_health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": await check_database(),
            "redis": await check_redis(),
            "model": await check_model(),
            "vector_store": await check_vector_store()
        }
    }
```

---

## Common Error Patterns

### Import and Dependency Issues

```python
# backend/core/exceptions.py - Custom exception handling
class MedicalAIException(Exception):
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ModelNotFoundError(MedicalAIException):
    def __init__(self, model_path: str):
        super().__init__(
            f"Model not found at {model_path}", 
            "MODEL_NOT_FOUND"
        )

class DatabaseConnectionError(MedicalAIException):
    def __init__(self, reason: str):
        super().__init__(
            f"Database connection failed: {reason}", 
            "DB_CONNECTION_ERROR"
        )

# Global exception handler
@app.exception_handler(MedicalAIException)
async def medical_ai_exception_handler(request: Request, exc: MedicalAIException):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": exc.message,
                "code": exc.error_code,
                "type": "MedicalAIException"
            }
        }
    )
```

### Configuration Issues

```python
# backend/core/config.py - Comprehensive configuration
from pydantic import BaseSettings, validator
from typing import Optional, List
import os

class Settings(BaseSettings):
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str
    redis_password: Optional[str] = None
    
    # Model
    model_path: str
    model_device: str = "auto"  # "cpu", "cuda", "auto"
    model_max_tokens: int = 2048
    
    # Vector Store
    vector_store_path: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Logging
    log_level: str = "INFO"
    enable_structured_logging: bool = True
    
    # Performance
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith('postgresql://'):
            raise ValueError('Database URL must be PostgreSQL')
        return v
        
    @validator('model_path')
    def validate_model_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f'Model path does not exist: {v}')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
```

---

## Recovery Procedures

### Database Recovery

```bash
#!/bin/bash
# scripts/emergency_recovery.sh

echo "🚨 Starting emergency database recovery..."

# Check database status
docker exec medai-db pg_isready -U meduser

# Create backup before recovery
docker exec medai-db pg_dump -U meduser meddb > backup_$(date +%Y%m%d_%H%M%S).sql

# If corruption suspected, try to fix
echo "Checking for database corruption..."
docker exec medai-db psql -U meduser -d meddb -c "SELECT pg_database(datname) FROM pg_database WHERE datname = 'meddb';"

# Reset connection pool if needed
echo "Resetting connection pools..."
docker restart medai-backend

# Verify recovery
echo "Verifying database recovery..."
sleep 10
curl -f http://localhost:8000/api/v1/health || echo "❌ Health check failed"

echo "✅ Emergency recovery completed"
```

### Model Recovery

```bash
#!/bin/bash
# scripts/model_recovery.sh

echo "🤖 Starting model recovery..."

# Check if model files exist
if [ ! -d "/app/models/medical-llm" ]; then
    echo "❌ Model directory not found, downloading..."
    
    # Download model from backup
    gsutil cp gs://medical-ai-backups/models/latest.tar.gz /tmp/model.tar.gz
    cd /app/models
    tar -xzf /tmp/model.tar.gz
    mv medical-llm-backup medical-llm
    
    echo "✅ Model restored from backup"
fi

# Verify model integrity
docker exec medai-backend python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('/app/models/medical-llm')
print('✅ Model loaded successfully')
"

echo "✅ Model recovery completed"
```

---

## Getting Help

### Debug Mode

```python
# Add debug endpoints for development
@app.get("/api/v1/debug/config")
async def debug_config():
    """Debug endpoint to check configuration (only in development)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=403, detail="Forbidden")
    
    return {
        "database_url_configured": bool(settings.database_url),
        "model_path_exists": os.path.exists(settings.model_path),
        "vector_store_path_exists": os.path.exists(settings.vector_store_path),
        "redis_url_configured": bool(settings.redis_url),
        "python_version": sys.version,
        "environment": os.getenv("ENVIRONMENT", "unknown")
    }

@app.get("/api/v1/debug/health")
async def debug_health():
    """Detailed health check for debugging"""
    health_status = {}
    
    try:
        # Test database
        result = await db.fetch_one("SELECT 1")
        health_status["database"] = "ok" if result else "failed"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
    
    try:
        # Test Redis
        await redis.ping()
        health_status["redis"] = "ok"
    except Exception as e:
        health_status["redis"] = f"error: {str(e)}"
    
    try:
        # Test model loading
        import transformers
        health_status["transformers"] = transformers.__version__
    except Exception as e:
        health_status["transformers"] = f"error: {str(e)}"
    
    return health_status
```

### Diagnostic Commands

```bash
#!/bin/bash
# scripts/diagnostic.sh

echo "🔍 Running Medical AI Assistant Diagnostics..."

# Check Docker containers
echo "Docker containers status:"
docker ps -a --filter name=medai

# Check network
echo "Docker networks:"
docker network ls
docker network inspect medical-ai-network

# Check volumes
echo "Docker volumes:"
docker volume ls

# Check logs for errors
echo "Backend logs (last 50 lines):"
docker logs medai-backend --tail 50

echo "Database logs (last 50 lines):"
docker logs medai-db --tail 50

# Test connectivity
echo "Testing connectivity:"
docker exec medai-backend curl -f http://medai-db:5432 && echo "✅ DB reachable" || echo "❌ DB unreachable"
docker exec medai-backend redis-cli -h medai-redis -a ${REDIS_PASSWORD} ping && echo "✅ Redis reachable" || echo "❌ Redis unreachable"

# Check resource usage
echo "Resource usage:"
docker stats --no-stream

echo "✅ Diagnostic complete"
```

### Support Information

**When reporting issues, please include:**

1. **System Information:**
   - OS version and architecture
   - Docker version
   - Python/Node.js versions
   - GPU/CPU information

2. **Error Details:**
   - Complete error messages
   - Stack traces
   - Log files (sanitized)
   - Steps to reproduce

3. **Environment:**
   - Configuration files (.env, docker-compose.yml)
   - Model versions and paths
   - Database schema version

4. **Context:**
   - What was being performed
   - Expected vs actual behavior
   - Recent changes or deployments

**Emergency Contacts:**
- **Technical Lead:** tech-lead@medical-ai.example
- **Security Team:** security@medical-ai.example
- **DevOps:** devops@medical-ai.example

---

## Additional Resources

- [API Documentation](./api/README.md)
- [Architecture Guide](../docs/architecture/README.md)
- [Security Guidelines](../docs/security/README.md)
- [Performance Tuning](../docs/performance/README.md)
- [Deployment Guide](../docs/deployment/README.md)

---

**Last Updated:** November 2025  
**Version:** 1.0  
**Maintainer:** Medical AI Development Team