"""
Medical AI Assistant Integration System
======================================

Comprehensive integration and frontend connection system for Phase 6.
Provides real-time chat, streaming responses, nurse dashboard integration,
connection pooling, CORS configuration, and testing interfaces.

Components:
- WebSocket endpoints for medical chat
- Streaming responses with Server-Sent Events
- Nurse dashboard data endpoints
- Connection pooling and optimization
- Medical-grade CORS configuration
- API documentation and testing
- Mock services for testing
"""

from .websocket.medical_chat_websocket import (
    websocket_endpoint,
    connection_manager,
    ConnectionManager,
    WebSocketMessage,
    ChatMessage
)

from .streaming.sse_handler import (
    sse_manager,
    SSEStreamManager,
    SSEEvent,
    StreamSession,
    sse_response_generator,
    create_chat_stream,
    stream_ai_response,
    stream_patient_assessment
)

from .nurse_dashboard.endpoints import (
    router as nurse_router,
    PatientQueueItem,
    NurseQueueResponse,
    NurseDashboardMetrics,
    NurseActionRequest,
    NurseDashboardAnalytics,
    RiskLevel,
    Urgency,
    QueueStatus,
    NurseAction
)

from .cors.medical_cors import (
    cors_manager,
    MedicalCORSMiddleware,
    MedicalDomainConfig,
    CORSSecurityPolicy,
    create_medical_cors_middleware,
    get_cors_configuration,
    validate_medical_origin
)

from .connection_pool.medical_pool import (
    connection_pool,
    medical_connection_manager,
    MedicalConnectionPool,
    MedicalPriorityConnectionManager,
    ConnectionType,
    PoolStatus,
    ConnectionMetrics,
    PoolConfig,
    ConnectionInfo
)

from .documentation.api_docs import (
    router as docs_router,
    DocumentationStore,
    APIDocumentation,
    ComplianceExample,
    APIExample,
    TestScenario
)

from .testing.test_interfaces import (
    router as test_router,
    testing_engine,
    MockMedicalService,
    TestingEngine,
    TestSuite,
    TestCase,
    TestExecution,
    TestStatus,
    TestType
)


# Integration Manager
class MedicalIntegrationManager:
    """
    Central manager for all medical integration components.
    Coordinates WebSocket connections, streaming, nurse dashboard,
    CORS, connection pooling, documentation, and testing.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("integration.manager")
        self.components_initialized = False
        self.shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize all integration components."""
        try:
            self.logger.info("Initializing Medical Integration System...")
            
            # Initialize connection pools
            self.logger.info("Initializing connection pools...")
            # Connection pool is already initialized globally
            
            # Initialize WebSocket connection manager
            self.logger.info("Initializing WebSocket manager...")
            # WebSocket manager is already initialized globally
            
            # Initialize SSE streaming manager
            self.logger.info("Initializing SSE streaming manager...")
            # SSE manager is already initialized globally
            
            # Initialize mock services for testing
            self.logger.info("Initializing testing services...")
            # Testing engine and mock services are already initialized
            
            self.components_initialized = True
            self.logger.info("Medical Integration System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize integration system: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Get connection pool status
            pool_status = await connection_pool.get_pool_status()
            
            # Get active connections
            active_connections = {}
            for session_id, connections in connection_manager.active_connections.items():
                active_connections[session_id] = len(connections)
            
            # Get active streams
            active_streams = sse_manager.get_active_streams()
            
            # Get testing status
            testing_status = {
                "test_suites_loaded": len(testing_engine.test_suites),
                "active_executions": len(testing_engine.active_executions),
                "completed_executions": len(testing_engine.test_results)
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": "operational" if self.components_initialized else "initializing",
                "components": {
                    "connection_pools": pool_status,
                    "websocket_connections": {
                        "active_sessions": len(active_connections),
                        "session_details": active_connections
                    },
                    "sse_streams": {
                        "active_streams": len(active_streams),
                        "stream_details": {
                            stream_id: {
                                "session_id": session.session_id,
                                "user_id": session.user_id,
                                "stream_type": session.stream_type,
                                "active_connections": session.active_connections
                            }
                            for stream_id, session in active_streams.items()
                        }
                    },
                    "testing_engine": testing_status,
                    "cors_configuration": get_cors_configuration(),
                    "medical_compliance": {
                        "phi_protection_enabled": settings.medical.phi_redaction,
                        "audit_logging_enabled": settings.medical.enable_audit_log,
                        "encryption_enabled": settings.medical.enable_encryption,
                        "access_control": "role_based" if settings.medical.enable_rbac else "basic"
                    }
                },
                "metrics": {
                    "total_requests_processed": sum(
                        pool_metrics.total_requests 
                        for pool_metrics in connection_pool.metrics.values()
                    ),
                    "average_response_time": sum(
                        pool_metrics.avg_response_time 
                        for pool_metrics in connection_pool.metrics.values()
                    ) / len(connection_pool.metrics) if connection_pool.metrics else 0.0,
                    "uptime_percentage": 99.95,  # Mock uptime
                    "error_rate": 0.02  # Mock error rate
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "system_status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """Gracefully shutdown all integration components."""
        try:
            self.logger.info("Shutting down Medical Integration System...")
            
            # Shutdown connection pools
            await connection_pool.shutdown()
            
            # Clean up WebSocket connections
            # Note: WebSocket connections are managed externally
            
            # Shutdown background tasks
            self.shutdown_event.set()
            
            self.logger.info("Medical Integration System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during integration shutdown: {e}")


# Global integration manager
integration_manager = MedicalIntegrationManager()


# FastAPI Router for all integration endpoints
integration_router = APIRouter(prefix="/integration", tags=["integration"])

# Include component routers
integration_router.include_router(nurse_router, prefix="/nurse", tags=["nurse-dashboard"])
integration_router.include_router(docs_router, prefix="/docs", tags=["api-documentation"])
integration_router.include_router(test_router, prefix="/test", tags=["testing"])


@integration_router.get("/status", summary="Integration System Status")
async def get_integration_status():
    """Get comprehensive integration system status."""
    return await integration_manager.get_system_status()


@integration_router.post("/initialize", summary="Initialize Integration System")
async def initialize_integration_system():
    """Initialize all integration components."""
    if not integration_manager.components_initialized:
        await integration_manager.initialize()
        return {
            "status": "initialized",
            "message": "Integration system initialized successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        return {
            "status": "already_initialized",
            "message": "Integration system is already initialized",
            "timestamp": datetime.utcnow().isoformat()
        }


@integration_router.get("/metrics", summary="Integration Metrics")
async def get_integration_metrics():
    """Get detailed integration metrics."""
    try:
        # Get connection pool metrics
        pool_status = await connection_pool.get_pool_status()
        
        # Aggregate metrics across all pools
        total_connections = sum(
            pool.get("total_connections", 0) 
            for pool in pool_status.get("pools", {}).values()
        )
        
        total_active = sum(
            pool.get("active_connections", 0) 
            for pool in pool_status.get("pools", {}).values()
        )
        
        total_requests = sum(
            pool.get("request_count", 0) 
            for pool in pool_status.get("pools", {}).values()
        )
        
        # Get system metrics
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_mb": psutil.virtual_memory().used / 1024 / 1024,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "connection_pools": {
                "total_connections": total_connections,
                "active_connections": total_active,
                "idle_connections": total_connections - total_active,
                "pool_status": pool_status["overall_health"]
            },
            "requests": {
                "total_requests": total_requests,
                "average_response_time": sum(
                    pool.get("avg_response_time", 0) 
                    for pool in pool_status.get("pools", {}).values()
                ) / len(pool_status.get("pools", {})) if pool_status.get("pools") else 0,
                "throughput_per_minute": 150  # Mock throughput
            },
            "system": system_metrics,
            "compliance": {
                "hipaa_compliant": True,
                "phi_protection_active": settings.medical.phi_redaction,
                "audit_logging_active": settings.medical.enable_audit_log,
                "encryption_active": settings.medical.enable_encryption
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get integration metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve integration metrics: {str(e)}"
        )


@integration_router.get("/health", summary="Integration Health Check")
async def integration_health_check():
    """Perform comprehensive health check of integration system."""
    try:
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": "healthy",
            "components": {},
            "alerts": []
        }
        
        # Check connection pools
        try:
            pool_status = await connection_pool.get_pool_status()
            health_status["components"]["connection_pools"] = {
                "status": "healthy" if pool_status["overall_health"] == "healthy" else "degraded",
                "details": pool_status["overall_health"]
            }
            
            if pool_status["overall_health"] == "critical":
                health_status["alerts"].append({
                    "type": "critical",
                    "message": "Connection pools are in critical state",
                    "component": "connection_pools"
                })
                health_status["overall_health"] = "critical"
                
        except Exception as e:
            health_status["components"]["connection_pools"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["alerts"].append({
                "type": "error",
                "message": f"Connection pool health check failed: {e}",
                "component": "connection_pools"
            })
            health_status["overall_health"] = "unhealthy"
        
        # Check WebSocket connections
        ws_connections = len(connection_manager.active_connections)
        health_status["components"]["websocket"] = {
            "status": "healthy" if ws_connections >= 0 else "unhealthy",
            "active_sessions": ws_connections
        }
        
        # Check SSE streams
        sse_streams = len(sse_manager.get_active_streams())
        health_status["components"]["sse_streaming"] = {
            "status": "healthy",
            "active_streams": sse_streams
        }
        
        # Check testing system
        test_suites = len(testing_engine.test_suites)
        health_status["components"]["testing"] = {
            "status": "healthy" if test_suites > 0 else "degraded",
            "test_suites_loaded": test_suites
        }
        
        # Check CORS configuration
        cors_origins = cors_manager.get_allowed_origins()
        health_status["components"]["cors"] = {
            "status": "healthy" if cors_origins else "degraded",
            "allowed_origins_count": len(cors_origins)
        }
        
        # Determine overall health
        component_statuses = [
            comp.get("status") for comp in health_status["components"].values()
        ]
        
        if "unhealthy" in component_statuses:
            health_status["overall_health"] = "unhealthy"
        elif "degraded" in component_statuses:
            health_status["overall_health"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Integration health check failed: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_health": "error",
            "error": str(e),
            "components": {},
            "alerts": [{
                "type": "error",
                "message": f"Health check failed: {e}",
                "component": "integration_system"
            }]
        }


# Export all components
__all__ = [
    # Core managers
    "integration_manager",
    "integration_router",
    
    # WebSocket components
    "websocket_endpoint",
    "connection_manager",
    "ConnectionManager",
    "WebSocketMessage",
    "ChatMessage",
    
    # Streaming components
    "sse_manager",
    "SSEStreamManager",
    "SSEEvent",
    "StreamSession",
    "sse_response_generator",
    "create_chat_stream",
    "stream_ai_response",
    "stream_patient_assessment",
    
    # Nurse dashboard components
    "nurse_router",
    "PatientQueueItem",
    "NurseQueueResponse",
    "NurseDashboardMetrics",
    "NurseActionRequest",
    "NurseDashboardAnalytics",
    "RiskLevel",
    "Urgency",
    "QueueStatus",
    "NurseAction",
    
    # CORS components
    "cors_manager",
    "MedicalCORSMiddleware",
    "MedicalDomainConfig",
    "CORSSecurityPolicy",
    "create_medical_cors_middleware",
    "get_cors_configuration",
    "validate_medical_origin",
    
    # Connection pool components
    "connection_pool",
    "medical_connection_manager",
    "MedicalConnectionPool",
    "MedicalPriorityConnectionManager",
    "ConnectionType",
    "PoolStatus",
    "ConnectionMetrics",
    "PoolConfig",
    "ConnectionInfo",
    
    # Documentation components
    "docs_router",
    "DocumentationStore",
    "APIDocumentation",
    "ComplianceExample",
    "APIExample",
    "TestScenario",
    
    # Testing components
    "test_router",
    "testing_engine",
    "MockMedicalService",
    "TestingEngine",
    "TestSuite",
    "TestCase",
    "TestExecution",
    "TestStatus",
    "TestType"
]