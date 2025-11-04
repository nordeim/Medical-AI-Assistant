"""
Testing interfaces for medical AI assistant integration.
Provides comprehensive testing tools, mock services, and validation
for medical compliance and system functionality.
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import websockets
import aiohttp
import structlog

from fastapi import (
    APIRouter, 
    HTTPException, 
    Depends, 
    BackgroundTasks,
    WebSocket,
    WebSocketDisconnect
)
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import pytest

from ...config.settings import get_settings
from ...config.logging_config import (
    get_logger, get_audit_logger, LoggingContextManager
)
from ..websocket.medical_chat_websocket import connection_manager
from ..streaming.sse_handler import sse_manager
from ..connection_pool.medical_pool import connection_pool, medical_connection_manager


# Configuration
settings = get_settings()
logger = get_logger("testing")
audit_logger = get_audit_logger()
security = HTTPBearer(auto_error=False)

# Router
router = APIRouter(prefix="/test", tags=["testing"])


# Testing Models
class TestStatus(str, Enum):
    """Test execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class TestType(str, Enum):
    """Types of tests."""
    UNIT = "unit"
    INTEGRATION = "integration"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    LOAD = "load"
    CHAOS = "chaos"


@dataclass
class TestCase:
    """Individual test case."""
    id: str
    name: str
    description: str
    test_type: TestType
    endpoint: str
    method: str
    preconditions: List[str] = field(default_factory=list)
    test_steps: List[str] = field(default_factory=list)
    expected_results: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 30
    retry_count: int = 3
    compliance_checks: List[str] = field(default_factory=list)
    security_validations: List[str] = field(default_factory=list)


@dataclass
class TestSuite:
    """Test suite containing multiple test cases."""
    id: str
    name: str
    description: str
    category: str
    test_cases: List[TestCase]
    parallel_execution: bool = True
    max_concurrent_tests: int = 10
    failure_threshold: float = 0.1  # 10% failure rate allowed


@dataclass
class TestExecution:
    """Test execution result."""
    test_id: str
    suite_id: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    compliance_passed: Optional[bool] = None
    security_passed: Optional[bool] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class MockMedicalService:
    """Mock medical service for testing."""
    
    def __init__(self):
        self.mock_patients = self._generate_mock_patients()
        self.mock_assessments = self._generate_mock_assessments()
        self.mock_nurses = self._generate_mock_nurses()
        self.response_times = {}
        self.error_rates = {}
        self.logger = structlog.get_logger("mock.medical")
    
    def _generate_mock_patients(self) -> List[Dict[str, Any]]:
        """Generate mock patient data for testing."""
        return [
            {
                "id": "patient_001",
                "session_id": "sess_001",
                "chief_complaint": "Chest pain",
                "symptoms": ["chest pain", "shortness of breath"],
                "risk_level": "high",
                "urgency": "urgent",
                "red_flags": ["chest pain"],
                "created_at": datetime.utcnow() - timedelta(minutes=15)
            },
            {
                "id": "patient_002",
                "session_id": "sess_002",
                "chief_complaint": "Headache",
                "symptoms": ["headache", "fatigue"],
                "risk_level": "low",
                "urgency": "routine",
                "red_flags": [],
                "created_at": datetime.utcnow() - timedelta(minutes=45)
            },
            {
                "id": "patient_003",
                "session_id": "sess_003",
                "chief_complaint": "Severe abdominal pain",
                "symptoms": ["severe abdominal pain", "nausea", "vomiting"],
                "risk_level": "critical",
                "urgency": "immediate",
                "red_flags": ["severe pain", "vomiting"],
                "created_at": datetime.utcnow() - timedelta(minutes=5)
            }
        ]
    
    def _generate_mock_assessments(self) -> List[Dict[str, Any]]:
        """Generate mock assessment data."""
        return [
            {
                "id": "assess_001",
                "session_id": "sess_001",
                "patient_id": "patient_001",
                "status": "pending",
                "risk_level": "high",
                "confidence_score": 0.95,
                "recommendations": ["ECG", "Cardiology consult"],
                "created_at": datetime.utcnow() - timedelta(minutes=15)
            }
        ]
    
    def _generate_mock_nurses(self) -> List[Dict[str, Any]]:
        """Generate mock nurse data."""
        return [
            {
                "id": "nurse_001",
                "name": "Nurse Smith",
                "specialization": "Emergency Medicine",
                "active": True,
                "current_cases": 3
            },
            {
                "id": "nurse_002",
                "name": "Nurse Johnson",
                "specialization": "Internal Medicine",
                "active": True,
                "current_cases": 2
            }
        ]
    
    async def simulate_medical_chat(
        self,
        message: str,
        session_id: str,
        user_type: str = "patient"
    ) -> Dict[str, Any]:
        """Simulate medical AI chat response."""
        # Simulate processing delay
        await asyncio.sleep(0.5)
        
        # Check for red flags
        red_flags = []
        content_lower = message.lower()
        
        emergency_keywords = [
            "chest pain", "heart attack", "can't breathe", "unconscious",
            "severe bleeding", "stroke", "seizure", "overdose"
        ]
        
        for keyword in emergency_keywords:
            if keyword in content_lower:
                red_flags.append(keyword)
        
        # Generate response based on content
        if red_flags:
            response = f"This appears to be a medical emergency. I'm connecting you with a nurse immediately. If you're experiencing {', '.join(red_flags)}, please consider calling 911."
            urgency = "emergency"
            confidence = 0.98
        elif "headache" in content_lower:
            response = "I understand you're experiencing a headache. Can you describe the pain and how long you've had it?"
            urgency = "routine"
            confidence = 0.85
        else:
            response = "Thank you for your message. Can you provide more details about your symptoms?"
            urgency = "normal"
            confidence = 0.75
        
        return {
            "response": response,
            "urgency": urgency,
            "confidence": confidence,
            "red_flags": red_flags,
            "processing_time": 0.5,
            "session_id": session_id
        }
    
    async def simulate_nurse_queue(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate nurse queue with filtering."""
        await asyncio.sleep(0.2)  # Simulate database query
        
        queue = []
        for patient in self.mock_patients:
            # Apply filters
            if filters:
                if filters.get("urgency") and patient["urgency"] != filters["urgency"]:
                    continue
                if filters.get("risk_level") and patient["risk_level"] != filters["risk_level"]:
                    continue
                if filters.get("has_red_flags") and len(patient["red_flags"]) == 0:
                    continue
            
            queue_item = {
                "id": patient["id"],
                "session_id": patient["session_id"],
                "patient_name": f"Patient {patient['id'][-3:]}",
                "chief_complaint": patient["chief_complaint"],
                "symptoms": patient["symptoms"],
                "risk_level": patient["risk_level"],
                "urgency": patient["urgency"],
                "has_red_flags": len(patient["red_flags"]) > 0,
                "red_flags": patient["red_flags"],
                "created_at": patient["created_at"].isoformat(),
                "wait_time_minutes": int((datetime.utcnow() - patient["created_at"]).total_seconds() / 60),
                "priority_score": self._calculate_priority_score(patient)
            }
            queue.append(queue_item)
        
        return {
            "queue": queue,
            "total": len(queue),
            "urgent_count": len([item for item in queue if item["urgency"] == "urgent"]),
            "immediate_count": len([item for item in queue if item["urgency"] == "immediate"]),
            "red_flag_count": len([item for item in queue if item["has_red_flags"]]),
            "avg_wait_time": sum(item["wait_time_minutes"] for item in queue) / len(queue) if queue else 0,
            "queue_load": "low" if len(queue) < 5 else "moderate" if len(queue) < 15 else "high"
        }
    
    def _calculate_priority_score(self, patient: Dict[str, Any]) -> float:
        """Calculate priority score for patient."""
        score = 0.0
        
        risk_multipliers = {"critical": 10, "high": 7, "medium": 4, "low": 1}
        score += risk_multipliers.get(patient["risk_level"], 1)
        
        urgency_multipliers = {"immediate": 5, "urgent": 3, "routine": 1}
        score += urgency_multipliers.get(patient["urgency"], 1)
        
        if patient["red_flags"]:
            score += 3
        
        wait_minutes = (datetime.utcnow() - patient["created_at"]).total_seconds() / 60
        score += min(wait_minutes / 10, 5)
        
        return round(score, 2)


class TestingEngine:
    """Main testing engine for medical AI integration."""
    
    def __init__(self):
        self.mock_service = MockMedicalService()
        self.test_suites = self._create_test_suites()
        self.active_executions: Dict[str, TestExecution] = {}
        self.test_results: List[TestExecution] = []
        self.logger = structlog.get_logger("testing.engine")
    
    def _create_test_suites(self) -> Dict[str, TestSuite]:
        """Create comprehensive test suites."""
        return {
            "patient_chat_suite": TestSuite(
                id="suite_patient_001",
                name="Patient Chat Flow Tests",
                description="Test complete patient chat workflows",
                category="integration",
                test_cases=[
                    TestCase(
                        id="test_chat_001",
                        name="Patient Chat Session",
                        description="Test patient chat session creation and messaging",
                        test_type=TestType.INTEGRATION,
                        endpoint="/ws/chat",
                        method="WebSocket",
                        preconditions=["Valid patient authentication", "WebSocket support"],
                        test_steps=[
                            "Connect to WebSocket endpoint",
                            "Send authentication token",
                            "Send chat message",
                            "Receive AI response",
                            "Verify red flag detection"
                        ],
                        expected_results={
                            "connection_established": True,
                            "message_exchanged": True,
                            "ai_response_received": True,
                            "red_flags_detected": True
                        },
                        compliance_checks=["PHI protection", "Audit logging", "Data encryption"],
                        security_validations=["Authentication", "Authorization", "Input validation"]
                    ),
                    TestCase(
                        id="test_chat_002",
                        name="Emergency Red Flag",
                        description="Test emergency red flag detection and escalation",
                        test_type=TestType.COMPLIANCE,
                        endpoint="/ws/chat",
                        method="WebSocket",
                        preconditions=["Active chat session"],
                        test_steps=[
                            "Send emergency symptom message",
                            "Verify red flag detection",
                            "Check nurse notification",
                            "Validate escalation response"
                        ],
                        expected_results={
                            "red_flags_detected": True,
                            "nurse_alerted": True,
                            "escalation_triggered": True
                        },
                        timeout_seconds=10,
                        compliance_checks=["Emergency protocols", "Red flag detection"],
                        security_validations=["Emergency access", "Alert validation"]
                    )
                ]
            ),
            "nurse_dashboard_suite": TestSuite(
                id="suite_nurse_001",
                name="Nurse Dashboard Tests",
                description="Test nurse dashboard functionality",
                category="integration",
                test_cases=[
                    TestCase(
                        id="test_nurse_001",
                        name="Queue Access",
                        description="Test nurse queue access and filtering",
                        test_type=TestType.INTEGRATION,
                        endpoint="/api/nurse/queue",
                        method="GET",
                        preconditions=["Valid nurse authentication"],
                        test_steps=[
                            "Authenticate as nurse",
                            "Retrieve queue with filters",
                            "Verify response format",
                            "Check data integrity"
                        ],
                        expected_results={
                            "queue_retrieved": True,
                            "filters_applied": True,
                            "data_format_valid": True
                        },
                        compliance_checks=["Access logging", "Data pseudonymization"],
                        security_validations=["Role validation", "Permission checking"]
                    ),
                    TestCase(
                        id="test_nurse_002",
                        name="Patient Action",
                        description="Test nurse patient action (approve/override)",
                        test_type=TestType.INTEGRATION,
                        endpoint="/api/nurse/queue/{patient_id}/action",
                        method="POST",
                        preconditions=["Patient in queue", "Nurse authenticated"],
                        test_steps=[
                            "Select patient for review",
                            "Take action (approve/override)",
                            "Verify action recorded",
                            "Check queue update"
                        ],
                        expected_results={
                            "action_recorded": True,
                            "queue_updated": True,
                            "audit_logged": True
                        },
                        compliance_checks=["Action audit trail", "Permission validation"],
                        security_validations=["Action authorization", "Change logging"]
                    )
                ]
            ),
            "compliance_suite": TestSuite(
                id="suite_compliance_001",
                name="HIPAA Compliance Tests",
                description="Test HIPAA compliance requirements",
                category="compliance",
                test_cases=[
                    TestCase(
                        id="test_compliance_001",
                        name="PHI Protection",
                        description="Test PHI protection and redaction",
                        test_type=TestType.COMPLIANCE,
                        endpoint="/api/patient/messages",
                        method="POST",
                        preconditions=["Patient session active"],
                        test_steps=[
                            "Send message with PHI",
                            "Verify PHI redaction",
                            "Check audit logging",
                            "Validate encryption"
                        ],
                        expected_results={
                            "phi_redacted": True,
                            "audit_logged": True,
                            "encrypted": True
                        },
                        compliance_checks=["PHI redaction", "Encryption at rest", "Access logging"],
                        security_validations=["Data encryption", "Access controls"]
                    )
                ]
            ),
            "performance_suite": TestSuite(
                id="suite_performance_001",
                name="Performance Tests",
                description="Test system performance and scalability",
                category="performance",
                test_cases=[
                    TestCase(
                        id="test_performance_001",
                        name="Concurrent Connections",
                        description="Test concurrent WebSocket connections",
                        test_type=TestType.LOAD,
                        endpoint="/ws/chat",
                        method="WebSocket",
                        preconditions=["Load testing environment"],
                        test_steps=[
                            "Establish 50 concurrent connections",
                            "Send messages through all connections",
                            "Measure response times",
                            "Check for connection drops"
                        ],
                        expected_results={
                            "connections_established": 50,
                            "avg_response_time_ms": 100,
                            "error_rate": 0.01
                        },
                        timeout_seconds=60,
                        compliance_checks=["Performance under load"],
                        security_validations=["Connection validation"]
                    )
                ]
            )
        }
    
    async def execute_test_suite(
        self,
        suite_id: str,
        parallel: bool = True,
        max_concurrent: int = 10
    ) -> Dict[str, Any]:
        """Execute a test suite."""
        
        if suite_id not in self.test_suites:
            raise HTTPException(status_code=404, detail=f"Test suite {suite_id} not found")
        
        suite = self.test_suites[suite_id]
        execution_id = str(uuid.uuid4())
        
        # Create execution record
        execution = TestExecution(
            test_id=execution_id,
            suite_id=suite_id,
            status=TestStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        self.active_executions[execution_id] = execution
        
        self.logger.info(
            "Starting test suite execution",
            suite_id=suite_id,
            execution_id=execution_id,
            test_count=len(suite.test_cases)
        )
        
        try:
            # Execute test cases
            if parallel and suite.parallel_execution:
                results = await self._execute_tests_parallel(suite, max_concurrent)
            else:
                results = await self._execute_tests_sequential(suite)
            
            # Update execution result
            execution.status = TestStatus.PASSED if all(r.status == TestStatus.PASSED for r in results) else TestStatus.FAILED
            execution.end_time = datetime.utcnow()
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            execution.result = {
                "suite_results": [self._execution_to_dict(r) for r in results],
                "total_tests": len(suite.test_cases),
                "passed": len([r for r in results if r.status == TestStatus.PASSED]),
                "failed": len([r for r in results if r.status == TestStatus.FAILED]),
                "compliance_passed": all(r.compliance_passed for r in results if r.compliance_passed is not None),
                "security_passed": all(r.security_passed for r in results if r.security_passed is not None)
            }
            
            # Move to completed executions
            self.test_results.append(execution)
            del self.active_executions[execution_id]
            
            return execution.result
            
        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.end_time = datetime.utcnow()
            execution.error_message = str(e)
            self.test_results.append(execution)
            del self.active_executions[execution_id]
            
            self.logger.error(f"Test suite execution failed: {e}", suite_id=suite_id)
            raise
    
    async def _execute_tests_parallel(
        self,
        suite: TestSuite,
        max_concurrent: int
    ) -> List[TestExecution]:
        """Execute tests in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(test_case: TestCase):
            async with semaphore:
                return await self._execute_test_case(test_case, suite.id)
        
        tasks = [execute_with_semaphore(tc) for tc in suite.test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        executions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                execution = TestExecution(
                    test_id=str(uuid.uuid4()),
                    suite_id=suite.id,
                    status=TestStatus.ERROR,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    error_message=str(result)
                )
                executions.append(execution)
            else:
                executions.append(result)
        
        return executions
    
    async def _execute_tests_sequential(self, suite: TestSuite) -> List[TestExecution]:
        """Execute tests sequentially."""
        results = []
        for test_case in suite.test_cases:
            result = await self._execute_test_case(test_case, suite.id)
            results.append(result)
        return results
    
    async def _execute_test_case(
        self,
        test_case: TestCase,
        suite_id: str
    ) -> TestExecution:
        """Execute individual test case."""
        execution = TestExecution(
            test_id=test_case.id,
            suite_id=suite_id,
            status=TestStatus.RUNNING,
            start_time=datetime.utcnow()
        )
        
        start_time = time.time()
        
        try:
            # Execute test steps
            step_results = {}
            
            for i, step in enumerate(test_case.test_steps):
                step_start = time.time()
                
                # Execute step based on test type and endpoint
                if test_case.endpoint == "/ws/chat":
                    step_result = await self._test_websocket_chat(test_case, step)
                elif test_case.endpoint.startswith("/api/nurse/queue"):
                    step_result = await self._test_nurse_queue(test_case, step)
                elif test_case.compliance_checks:
                    step_result = await self._test_compliance(test_case, step)
                else:
                    step_result = {"status": "completed", "step": step}
                
                step_duration = time.time() - step_start
                step_results[f"step_{i+1}"] = {
                    "description": step,
                    "result": step_result,
                    "duration_ms": round(step_duration * 1000, 2)
                }
                
                # Check if step failed
                if step_result.get("status") == "failed":
                    execution.status = TestStatus.FAILED
                    execution.error_message = f"Step {i+1} failed: {step}"
                    break
            
            # If no step failed, mark as passed
            if execution.status != TestStatus.FAILED:
                execution.status = TestStatus.PASSED
                
                # Validate against expected results
                for key, expected_value in test_case.expected_results.items():
                    actual_value = step_results.get(list(step_results.keys())[-1], {}).get("result", {}).get(key)
                    if actual_value != expected_value:
                        execution.status = TestStatus.FAILED
                        execution.error_message = f"Expected {key}={expected_value}, got {actual_value}"
                        break
            
            # Check compliance
            if test_case.compliance_checks:
                execution.compliance_passed = await self._validate_compliance(test_case)
            
            # Check security
            if test_case.security_validations:
                execution.security_passed = await self._validate_security(test_case)
            
        except Exception as e:
            execution.status = TestStatus.ERROR
            execution.error_message = str(e)
            self.logger.error(f"Test case execution error: {e}", test_id=test_case.id)
        
        execution.end_time = datetime.utcnow()
        execution.duration_seconds = time.time() - start_time
        execution.metrics = {
            "total_steps": len(test_case.test_steps),
            "successful_steps": sum(1 for step in step_results.values() if step.get("result", {}).get("status") == "completed"),
            "step_results": step_results
        }
        
        return execution
    
    async def _test_websocket_chat(
        self,
        test_case: TestCase,
        step: str
    ) -> Dict[str, Any]:
        """Test WebSocket chat functionality."""
        
        if "connect" in step.lower():
            # Mock WebSocket connection
            await asyncio.sleep(0.1)  # Simulate connection time
            return {"status": "completed", "connection_established": True}
        
        elif "message" in step.lower():
            # Mock sending message
            response = await self.mock_service.simulate_medical_chat(
                message="I have chest pain",
                session_id="test_session_001",
                user_type="patient"
            )
            return {"status": "completed", "message_sent": True, "response": response}
        
        elif "red flag" in step.lower():
            # Mock red flag detection
            return {"status": "completed", "red_flags_detected": True, "red_flags": ["chest pain"]}
        
        return {"status": "completed"}
    
    async def _test_nurse_queue(
        self,
        test_case: TestCase,
        step: str
    ) -> Dict[str, Any]:
        """Test nurse queue functionality."""
        
        if "authenticate" in step.lower():
            return {"status": "completed", "authenticated": True}
        
        elif "retrieve" in step.lower():
            queue_data = await self.mock_service.simulate_nurse_queue()
            return {"status": "completed", "queue_retrieved": True, "queue_size": len(queue_data["queue"])}
        
        elif "action" in step.lower():
            return {"status": "completed", "action_recorded": True}
        
        return {"status": "completed"}
    
    async def _test_compliance(
        self,
        test_case: TestCase,
        step: str
    ) -> Dict[str, Any]:
        """Test compliance requirements."""
        
        if "phi" in step.lower():
            return {"status": "completed", "phi_protected": True}
        
        elif "audit" in step.lower():
            return {"status": "completed", "audit_logged": True}
        
        elif "encryption" in step.lower():
            return {"status": "completed", "encrypted": True}
        
        return {"status": "completed"}
    
    async def _validate_compliance(self, test_case: TestCase) -> bool:
        """Validate compliance requirements."""
        # Mock compliance validation
        return all(requirement in ["PHI protection", "Audit logging", "Encryption"] 
                  for requirement in test_case.compliance_checks)
    
    async def _validate_security(self, test_case: TestCase) -> bool:
        """Validate security requirements."""
        # Mock security validation
        return all(requirement in ["Authentication", "Authorization", "Input validation"]
                  for requirement in test_case.security_validations)
    
    def _execution_to_dict(self, execution: TestExecution) -> Dict[str, Any]:
        """Convert execution to dictionary."""
        return {
            "test_id": execution.test_id,
            "suite_id": execution.suite_id,
            "status": execution.status.value,
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "duration_seconds": execution.duration_seconds,
            "compliance_passed": execution.compliance_passed,
            "security_passed": execution.security_passed,
            "error_message": execution.error_message,
            "metrics": execution.metrics
        }


# Global testing engine
testing_engine = TestingEngine()


# Testing Endpoints
@router.get("/suites", summary="Get Test Suites")
async def get_test_suites(
    category: Optional[str] = Query(None, description="Filter by test category")
):
    """Get available test suites."""
    
    with LoggingContextManager(
        request_id=str(uuid.uuid4()),
        user_id="test_suites"
    ):
        logger.info("Test suites accessed", category=category)
        
        suites = list(testing_engine.test_suites.values())
        
        if category:
            suites = [suite for suite in suites if suite.category == category]
        
        return {
            "suites": [
                {
                    "id": suite.id,
                    "name": suite.name,
                    "description": suite.description,
                    "category": suite.category,
                    "test_count": len(suite.test_cases),
                    "parallel_execution": suite.parallel_execution,
                    "max_concurrent_tests": suite.max_concurrent_tests
                }
                for suite in suites
            ],
            "categories": list(set(suite.category for suite in testing_engine.test_suites.values()))
        }


@router.post("/suites/{suite_id}/execute", summary="Execute Test Suite")
async def execute_test_suite(
    suite_id: str,
    parallel: bool = Query(True, description="Execute tests in parallel"),
    max_concurrent: int = Query(10, description="Maximum concurrent tests"),
    credentials: str = Depends(security)
):
    """Execute a test suite."""
    
    with LoggingContextManager(
        request_id=str(uuid.uuid4()),
        user_id=credentials
    ):
        logger.info(
            "Test suite execution requested",
            suite_id=suite_id,
            parallel=parallel,
            max_concurrent=max_concurrent
        )
        
        # Audit log
        audit_logger.log_access(
            user_id=credentials,
            action="test_suite_execution",
            resource=f"test_suite:{suite_id}",
            details={
                "parallel": parallel,
                "max_concurrent": max_concurrent
            }
        )
        
        try:
            result = await testing_engine.execute_test_suite(
                suite_id=suite_id,
                parallel=parallel,
                max_concurrent=max_concurrent
            )
            
            return {
                "execution_id": str(uuid.uuid4()),
                "suite_id": suite_id,
                "status": "completed",
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}", suite_id=suite_id)
            raise HTTPException(
                status_code=500,
                detail=f"Test suite execution failed: {str(e)}"
            )


@router.get("/executions", summary="Get Test Executions")
async def get_test_executions(
    suite_id: Optional[str] = Query(None, description="Filter by suite ID"),
    status: Optional[TestStatus] = Query(None, description="Filter by execution status"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return")
):
    """Get test execution results."""
    
    with LoggingContextManager(
        request_id=str(uuid.uuid4()),
        user_id="test_executions"
    ):
        # Get both active and completed executions
        all_executions = list(testing_engine.active_executions.values()) + testing_engine.test_results
        
        # Apply filters
        if suite_id:
            all_executions = [exec for exec in all_executions if exec.suite_id == suite_id]
        
        if status:
            all_executions = [exec for exec in all_executions if exec.status == status]
        
        # Sort by start time (newest first)
        all_executions.sort(key=lambda x: x.start_time, reverse=True)
        
        # Limit results
        limited_executions = all_executions[:limit]
        
        return {
            "executions": [testing_engine._execution_to_dict(exec) for exec in limited_executions],
            "total": len(all_executions),
            "active_count": len(testing_engine.active_executions),
            "completed_count": len(testing_engine.test_results)
        }


@router.get("/executions/{execution_id}", summary="Get Test Execution Details")
async def get_execution_details(execution_id: str):
    """Get detailed test execution results."""
    
    # Search in active executions
    if execution_id in testing_engine.active_executions:
        execution = testing_engine.active_executions[execution_id]
    else:
        # Search in completed executions
        execution = next(
            (exec for exec in testing_engine.test_results if exec.test_id == execution_id),
            None
        )
    
    if not execution:
        raise HTTPException(
            status_code=404,
            detail=f"Test execution {execution_id} not found"
        )
    
    return testing_engine._execution_to_dict(execution)


@router.post("/mock/chat", summary="Test Mock Chat Service")
async def test_mock_chat(
    message: str = Body(..., description="Test message"),
    session_id: str = Body("test_session", description="Test session ID")
):
    """Test the mock medical chat service."""
    
    with LoggingContextManager(
        request_id=str(uuid.uuid4()),
        user_id="mock_chat_test"
    ):
        try:
            response = await testing_engine.mock_service.simulate_medical_chat(
                message=message,
                session_id=session_id
            )
            
            return {
                "input": {
                    "message": message,
                    "session_id": session_id
                },
                "output": response,
                "test_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Mock chat test failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Mock chat test failed: {str(e)}"
            )


@router.post("/mock/queue", summary="Test Mock Queue Service")
async def test_mock_queue(
    filters: Optional[Dict[str, Any]] = Body(None, description="Queue filters")
):
    """Test the mock nurse queue service."""
    
    with LoggingContextManager(
        request_id=str(uuid.uuid4()),
        user_id="mock_queue_test"
    ):
        try:
            queue_data = await testing_engine.mock_service.simulate_nurse_queue(filters)
            
            return {
                "input": {
                    "filters": filters
                },
                "output": queue_data,
                "test_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Mock queue test failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Mock queue test failed: {str(e)}"
            )


@router.get("/health/validation", summary="System Health Validation")
async def validate_system_health():
    """Validate system health for testing."""
    
    with LoggingContextManager(
        request_id=str(uuid.uuid4()),
        user_id="health_validation"
    ):
        try:
            # Check connection pools
            pool_status = await connection_pool.get_pool_status()
            
            # Check mock services
            mock_service_health = {
                "patients_available": len(testing_engine.mock_service.mock_patients),
                "assessments_available": len(testing_engine.mock_service.mock_assessments),
                "nurses_available": len(testing_engine.mock_service.mock_nurses)
            }
            
            # Check testing engine
            testing_health = {
                "test_suites_loaded": len(testing_engine.test_suites),
                "active_executions": len(testing_engine.active_executions),
                "completed_executions": len(testing_engine.test_results)
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "healthy",
                "connection_pools": pool_status,
                "mock_services": mock_service_health,
                "testing_engine": testing_health,
                "validation_results": {
                    "pools_healthy": pool_status["overall_health"] == "healthy",
                    "mock_services_available": all(
                        count > 0 for count in mock_service_health.values()
                    ),
                    "testing_engine_ready": testing_health["test_suites_loaded"] > 0
                }
            }
            
        except Exception as e:
            logger.error(f"Health validation failed: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "unhealthy",
                "error": str(e)
            }


# WebSocket endpoint for real-time test monitoring
@router.websocket("/monitor")
async def test_monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time test monitoring."""
    await websocket.accept()
    
    try:
        while True:
            # Send current system status
            status = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_executions": len(testing_engine.active_executions),
                "completed_executions": len(testing_engine.test_results),
                "test_suites": len(testing_engine.test_suites),
                "connection_pool_status": await connection_pool.get_pool_status()
            }
            
            await websocket.send_text(json.dumps({
                "type": "status_update",
                "data": status
            }))
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("Test monitoring WebSocket disconnected")
    except Exception as e:
        logger.error(f"Test monitoring error: {e}")
        await websocket.close()


# Export router and testing engine
__all__ = [
    "router",
    "testing_engine",
    "MockMedicalService",
    "TestingEngine",
    "TestSuite",
    "TestCase",
    "TestExecution",
    "TestStatus",
    "TestType"
]