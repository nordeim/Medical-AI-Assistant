# Nurse dashboard integration components
from .endpoints import (
    router,
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

__all__ = [
    "router",
    "PatientQueueItem",
    "NurseQueueResponse", 
    "NurseDashboardMetrics",
    "NurseActionRequest",
    "NurseDashboardAnalytics",
    "RiskLevel",
    "Urgency",
    "QueueStatus",
    "NurseAction"
]