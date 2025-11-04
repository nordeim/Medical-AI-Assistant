# Phase 2: Core Backend - Detailed Sub-Plan

**Scope**: Steps 3-5 of the master implementation plan
**Focus**: FastAPI Backend, LangChain v1.0 Agent Runtime, and RAG Layer

---

## Step 3: FastAPI Backend Foundation

### 3.1 Core Application Setup (Priority: CRITICAL)

**Files to Create:**

1. **`backend/app/__init__.py`** (~30 lines)
   - Package initialization
   - Version information
   - Export main application

2. **`backend/app/main.py`** (~150 lines)
   - FastAPI application factory
   - CORS middleware configuration
   - Router registration (sessions, messages, pars, auth, health)
   - WebSocket endpoint mounting
   - Startup/shutdown event handlers
   - Error handlers (500, 404, 422)
   - Logging configuration

3. **`backend/app/config.py`** (~120 lines)
   - Pydantic Settings for environment variables
   - Database configuration
   - Model configuration (paths, device)
   - Vector store configuration
   - API settings (rate limits, timeouts)
   - Security settings (JWT secret, expiry)
   - Logging configuration
   - Validation with defaults

4. **`backend/app/dependencies.py`** (~80 lines)
   - Database session dependency
   - Current user dependency (JWT validation)
   - Rate limiting dependency
   - WebSocket connection manager dependency
   - Configuration dependency

### 3.2 Database Layer (Priority: CRITICAL)

**Files to Create:**

5. **`backend/app/database.py`** (~90 lines)
   - SQLAlchemy engine creation
   - SessionLocal factory
   - Database connection management
   - Connection pooling configuration
   - Health check function

6. **`backend/app/models/__init__.py`** (~20 lines)
   - Export all SQLAlchemy models

7. **`backend/app/models/session.py`** (~70 lines)
   - Session SQLAlchemy model
   - Relationships to messages and PARs
   - Status enum (active, completed, escalated)
   - Timestamps and metadata

8. **`backend/app/models/message.py`** (~80 lines)
   - Message SQLAlchemy model
   - Role enum (user, assistant, system)
   - Relationship to session
   - Content with JSON storage
   - Timestamps

9. **`backend/app/models/par.py`** (~90 lines)
   - PAR (Preliminary Assessment Report) SQLAlchemy model
   - JSON fields for chief_complaint, symptoms, assessment, urgency
   - Relationship to session
   - Nurse review fields (reviewed_by, reviewed_at, override_reason)

10. **`backend/app/models/user.py`** (~100 lines)
    - User SQLAlchemy model
    - Role enum (patient, nurse, admin)
    - Password hashing methods
    - Relationships to sessions (as nurse)
    - Authentication fields

11. **`backend/app/models/audit_log.py`** (~70 lines)
    - AuditLog SQLAlchemy model
    - Action enum
    - User tracking
    - Metadata JSON field

12. **`backend/app/models/safety_filter_log.py`** (~80 lines)
    - SafetyFilterLog SQLAlchemy model
    - Filter type enum
    - Triggered filters tracking
    - Message content snapshot

### 3.3 Schema Layer (Pydantic) (Priority: HIGH)

**Files to Create:**

13. **`backend/app/schemas/__init__.py`** (~15 lines)
    - Export all Pydantic schemas

14. **`backend/app/schemas/session.py`** (~100 lines)
    - SessionCreate, SessionUpdate, SessionResponse
    - SessionStatus enum
    - Nested message and PAR schemas

15. **`backend/app/schemas/message.py`** (~80 lines)
    - MessageCreate, MessageResponse
    - MessageRole enum
    - Content validation

16. **`backend/app/schemas/par.py`** (~120 lines)
    - PARResponse, PARReview
    - Urgency enum (routine, urgent, immediate)
    - Validation for medical fields
    - RAG source tracking

17. **`backend/app/schemas/user.py`** (~90 lines)
    - UserCreate, UserLogin, UserResponse
    - Token response schema
    - Role enum

18. **`backend/app/schemas/websocket.py`** (~70 lines)
    - WebSocketMessage schema
    - MessageType enum (chat, status, error, par_ready)
    - Streaming message format

### 3.4 API Routes (Priority: HIGH)

**Files to Create:**

19. **`backend/app/api/__init__.py`** (~10 lines)
    - Package initialization

20. **`backend/app/api/sessions.py`** (~180 lines)
    - POST `/api/sessions` - Create new session
    - GET `/api/sessions/{session_id}` - Get session details
    - GET `/api/sessions` - List sessions (with nurse filtering)
    - PATCH `/api/sessions/{session_id}/status` - Update status
    - Dependencies: auth, rate limiting

21. **`backend/app/api/messages.py`** (~120 lines)
    - GET `/api/sessions/{session_id}/messages` - Get message history
    - POST `/api/sessions/{session_id}/messages` - Create message (triggers agent)
    - Dependencies: auth, session validation

22. **`backend/app/api/pars.py`** (~150 lines)
    - GET `/api/pars/{par_id}` - Get PAR details
    - GET `/api/sessions/{session_id}/par` - Get PAR for session
    - POST `/api/pars/{par_id}/review` - Nurse review (accept/override)
    - GET `/api/pars/queue` - Get nurse queue (unreviewed PARs)
    - Dependencies: auth, role validation (nurse only)

23. **`backend/app/api/auth.py`** (~140 lines)
    - POST `/api/auth/register` - Patient registration
    - POST `/api/auth/login` - Login (all roles)
    - POST `/api/auth/refresh` - Refresh JWT token
    - GET `/api/auth/me` - Get current user
    - Dependencies: JWT handling

24. **`backend/app/api/health.py`** (~80 lines)
    - GET `/api/health` - Basic health check
    - GET `/api/health/detailed` - Detailed health (DB, model, vector store)
    - Dependencies: none (public endpoints)

### 3.5 WebSocket Manager (Priority: CRITICAL)

**Files to Create:**

25. **`backend/app/websocket/manager.py`** (~200 lines)
    - WebSocketManager class (singleton)
    - Connection management (connect, disconnect, get_connection)
    - Message broadcasting to specific session
    - Streaming message handling
    - Error handling and reconnection logic
    - Active connections tracking

26. **`backend/app/websocket/handlers.py`** (~180 lines)
    - WebSocket endpoint handler
    - Message routing (chat, ping, disconnect)
    - Authentication via query params (JWT)
    - Session validation
    - Integration with agent orchestrator
    - Streaming response handling

### 3.6 Authentication & Authorization (Priority: HIGH)

**Files to Create:**

27. **`backend/app/auth/jwt.py`** (~120 lines)
    - JWT token creation (access, refresh)
    - Token verification
    - Payload extraction
    - Expiry handling

28. **`backend/app/auth/password.py`** (~60 lines)
    - Password hashing (bcrypt)
    - Password verification
    - Password strength validation

29. **`backend/app/auth/permissions.py`** (~100 lines)
    - Role-based permission checks
    - Decorators for route protection
    - Resource ownership validation

### 3.7 Service Layer (Priority: HIGH)

**Files to Create:**

30. **`backend/app/services/__init__.py`** (~10 lines)
    - Package initialization

31. **`backend/app/services/session_service.py`** (~150 lines)
    - Session CRUD operations
    - Session status management
    - Session retrieval with filters
    - Integration with audit logging

32. **`backend/app/services/message_service.py`** (~120 lines)
    - Message CRUD operations
    - Message history retrieval
    - Safety filter integration

33. **`backend/app/services/par_service.py`** (~180 lines)
    - PAR generation from agent output
    - PAR retrieval and queuing logic
    - Nurse review processing
    - Override reason validation

34. **`backend/app/services/audit_service.py`** (~100 lines)
    - Audit log creation
    - Structured logging
    - Event tracking (session start, PAR review, escalation)

---

## Step 4: LangChain v1.0 Agent Runtime

### 4.1 Agent Orchestrator (Priority: CRITICAL)

**Files to Create:**

35. **`backend/app/agent/__init__.py`** (~15 lines)
    - Package initialization

36. **`backend/app/agent/orchestrator.py`** (~300 lines)
    - MedicalAgentOrchestrator class
    - LangChain v1.0 agent initialization
    - Tool registration (RAG retrieval, red flag detection)
    - Streaming callback handling
    - PAR generation orchestration
    - Error handling and fallback logic
    - Session state management

37. **`backend/app/agent/config.py`** (~100 lines)
    - Agent configuration dataclass
    - System prompt template loading
    - Model parameters (temperature, max_tokens)
    - Tool configuration
    - Streaming settings

### 4.2 System Prompts (Priority: CRITICAL)

**Files to Create:**

38. **`backend/app/prompts/system_prompt.txt`** (~150 lines)
    - Medical triage system prompt
    - Safety guidelines (no diagnosis, no prescription)
    - Conversation flow instructions
    - Information gathering strategy
    - PAR generation guidelines
    - Red flag escalation rules

39. **`backend/app/prompts/par_generation_prompt.txt`** (~100 lines)
    - Structured PAR extraction prompt
    - JSON format specification
    - Urgency level determination
    - Summary guidelines

40. **`backend/app/prompts/safety_prompt.txt`** (~80 lines)
    - Safety check instructions
    - Prohibited language examples
    - Correction templates

### 4.3 LangChain Tools (Priority: HIGH)

**Files to Create:**

41. **`backend/app/agent/tools/__init__.py`** (~10 lines)
    - Package initialization

42. **`backend/app/agent/tools/rag_retrieval.py`** (~150 lines)
    - RAGRetrievalTool class (LangChain Tool)
    - Query embedding
    - Vector store search
    - Source formatting for LLM
    - Relevance scoring

43. **`backend/app/agent/tools/red_flag_detector.py`** (~120 lines)
    - RedFlagDetectorTool class
    - Keyword matching for emergent symptoms
    - Emergency escalation logic
    - Warning message generation

44. **`backend/app/agent/tools/ehr_connector.py`** (~100 lines)
    - EHRConnectorTool class (placeholder)
    - Mock patient history retrieval
    - Integration interface for future EHR systems

### 4.4 Callbacks & Streaming (Priority: HIGH)

**Files to Create:**

45. **`backend/app/agent/callbacks/streaming_callback.py`** (~150 lines)
    - StreamingCallbackHandler class (LangChain callback)
    - Token-by-token streaming via WebSocket
    - Integration with WebSocketManager
    - Error handling during streaming

46. **`backend/app/agent/callbacks/safety_callback.py`** (~180 lines)
    - SafetyCallbackHandler class
    - Real-time content filtering
    - Prohibited pattern detection (diagnosis, prescription)
    - Message blocking and correction
    - Safety filter logging

47. **`backend/app/agent/callbacks/audit_callback.py`** (~120 lines)
    - AuditCallbackHandler class
    - Tool usage logging
    - Token usage tracking
    - Latency monitoring

### 4.5 PAR Generation (Priority: HIGH)

**Files to Create:**

48. **`backend/app/agent/par_generator.py`** (~200 lines)
    - PARGenerator class
    - Conversation summarization
    - Structured extraction (chief complaint, symptoms, urgency)
    - JSON validation
    - RAG source citation
    - Urgency level determination logic

### 4.6 Safety Filters (Priority: CRITICAL)

**Files to Create:**

49. **`backend/app/agent/safety/__init__.py`** (~10 lines)
    - Package initialization

50. **`backend/app/agent/safety/content_filter.py`** (~200 lines)
    - ContentFilter class
    - Pattern-based filtering (regex for diagnosis, prescription terms)
    - Severity classification
    - Correction suggestions
    - Allowlist management

51. **`backend/app/agent/safety/red_flag_rules.py`** (~150 lines)
    - RedFlagRules class
    - Symptom-based emergency detection
    - Rule definitions (chest pain + radiation, difficulty breathing, etc.)
    - Immediate escalation triggers

---

## Step 5: RAG Layer & Vector Database

### 5.1 Embeddings Service (Priority: CRITICAL)

**Files to Create:**

52. **`backend/app/rag/__init__.py`** (~10 lines)
    - Package initialization

53. **`backend/app/rag/embeddings.py`** (~150 lines)
    - EmbeddingsService class
    - Model initialization (sentence-transformers)
    - Batch embedding generation
    - Caching layer (Redis integration)
    - Device management (CPU/GPU)

### 5.2 Vector Store Manager (Priority: CRITICAL)

**Files to Create:**

54. **`backend/app/rag/vector_store.py`** (~200 lines)
    - VectorStoreManager class
    - Chroma client initialization
    - Collection management
    - Document ingestion with metadata
    - Similarity search with filters
    - Index persistence and loading

55. **`backend/app/rag/config.py`** (~80 lines)
    - RAG configuration dataclass
    - Embedding model settings
    - Vector store paths
    - Retrieval parameters (top_k, similarity threshold)

### 5.3 Document Processing (Priority: HIGH)

**Files to Create:**

56. **`backend/app/rag/document_processor.py`** (~180 lines)
    - DocumentProcessor class
    - Text chunking (semantic, fixed-size)
    - Metadata extraction
    - Document cleaning and normalization
    - Format handling (txt, md, pdf)

57. **`backend/app/rag/ingestion.py`** (~150 lines)
    - IngestionPipeline class
    - Batch document ingestion
    - Duplicate detection
    - Progress tracking
    - Error handling and retry logic

### 5.4 Retrieval Logic (Priority: HIGH)

**Files to Create:**

58. **`backend/app/rag/retriever.py`** (~200 lines)
    - Retriever class
    - Query preprocessing
    - Multi-stage retrieval (keyword + semantic)
    - Re-ranking with cross-encoder
    - Source deduplication
    - Context window management

59. **`backend/app/rag/reranker.py`** (~120 lines)
    - Reranker class
    - Cross-encoder model loading
    - Score calculation
    - Result sorting

### 5.5 Medical Guidelines (Priority: HIGH)

**Files to Create:**

60. **`backend/data/guidelines/general_triage.md`** (~200 lines)
    - General triage guidelines
    - Chief complaint classification
    - Common symptom inquiries
    - When to escalate

61. **`backend/data/guidelines/red_flags.md`** (~150 lines)
    - Emergency red flag symptoms
    - Immediate escalation criteria
    - Critical symptom combinations

62. **`backend/data/guidelines/consent_privacy.md`** (~100 lines)
    - Consent requirements
    - Privacy notices
    - Data handling guidelines

### 5.6 Ingestion Scripts (Priority: MEDIUM)

**Files to Create:**

63. **`backend/scripts/ingest_guidelines.py`** (~150 lines)
    - CLI script for ingesting medical guidelines
    - Directory scanning
    - Batch processing
    - Verification and statistics

64. **`backend/scripts/test_retrieval.py`** (~120 lines)
    - CLI script for testing RAG retrieval
    - Query examples
    - Result visualization
    - Performance metrics

---

## Execution Sequence

### Phase A: Backend Core (Files 1-34)
**Time Estimate**: 60-90 minutes
**Dependencies**: None (foundational)

1. Core application setup (files 1-4)
2. Database layer (files 5-12)
3. Schema layer (files 13-18)
4. API routes (files 19-24)
5. WebSocket manager (files 25-26)
6. Authentication (files 27-29)
7. Service layer (files 30-34)

**Validation**: 
- `curl http://localhost:8000/api/health` returns 200
- All endpoints documented in OpenAPI spec
- Database models alembic migration ready

### Phase B: Agent Runtime (Files 35-51)
**Time Estimate**: 90-120 minutes
**Dependencies**: Phase A complete

1. Agent orchestrator (files 35-37)
2. System prompts (files 38-40)
3. LangChain tools (files 41-44)
4. Callbacks and streaming (files 45-47)
5. PAR generation (file 48)
6. Safety filters (files 49-51)

**Validation**:
- Mock conversation returns PAR JSON
- Safety filters block prohibited content
- Streaming callback works end-to-end

### Phase C: RAG Layer (Files 52-64)
**Time Estimate**: 60-90 minutes
**Dependencies**: Phase B complete

1. Embeddings service (files 52-53)
2. Vector store manager (files 54-55)
3. Document processing (files 56-57)
4. Retrieval logic (files 58-59)
5. Medical guidelines (files 60-62)
6. Ingestion scripts (files 63-64)

**Validation**:
- Guidelines ingested successfully
- Test queries return relevant results
- RAG tool integrated with agent

---

## Testing Strategy

### Unit Tests (Per Phase)
- **Phase A**: Database models, schemas, auth functions
- **Phase B**: Safety filters, PAR generator, tools
- **Phase C**: Embeddings, retrieval, document processing

### Integration Tests (End of Phase 2)
- Full conversation flow (patient message → agent → PAR)
- WebSocket streaming
- Nurse dashboard API workflows
- Safety filter triggering

### Manual Testing Checklist
- [ ] Health endpoint responds
- [ ] Patient registration and login
- [ ] WebSocket connection established
- [ ] Agent responds with streaming
- [ ] Safety filters block bad content
- [ ] PAR generated after conversation
- [ ] Nurse can review PAR
- [ ] RAG retrieval returns relevant guidelines
- [ ] Audit logs persisted

---

## File Count Summary

- **Total Files**: 64 files
- **Estimated Lines of Code**: ~8,500 lines
- **Configuration Files**: 3 (orchestrator config, RAG config, agent config)
- **Documentation Files**: 3 (prompts, guidelines)
- **Script Files**: 2 (ingestion, testing)

---

## Dependencies to Install (Already in requirements.txt)

Core dependencies already specified in Phase 1:
- fastapi[all]==0.109.0
- langchain==0.1.0
- langchain-community==0.0.13
- sqlalchemy==2.0.25
- alembic==1.13.1
- chromadb==0.4.22
- sentence-transformers==2.2.2
- python-jose[cryptography]==3.3.0
- passlib[bcrypt]==1.7.4

---

## Success Criteria

**Phase 2 is complete when:**

1. ✅ FastAPI server starts without errors
2. ✅ All 24 API endpoints respond correctly
3. ✅ WebSocket connection accepts patient messages
4. ✅ LangChain agent responds with streaming
5. ✅ Safety filters block prohibited content
6. ✅ PAR generated and stored in database
7. ✅ Nurse queue endpoint returns unreviewed PARs
8. ✅ RAG retrieval returns relevant medical guidelines
9. ✅ Audit logs persisted for all actions
10. ✅ Health endpoint shows all services healthy

---

## Risk Mitigation

### Known Challenges:
1. **LangChain v1.0 Stability**: Use exact version pinning and tested patterns
2. **WebSocket Concurrency**: Implement proper connection pooling and cleanup
3. **Safety Filter Accuracy**: Start with conservative patterns, iterate based on testing
4. **RAG Relevance**: Tune embedding model and retrieval parameters

### Fallback Plans:
- If LangChain streaming fails → Implement manual SSE streaming
- If Chroma has issues → Provide simple in-memory vector store fallback
- If embeddings are slow → Implement aggressive caching

---

**Ready for execution upon user confirmation.**
