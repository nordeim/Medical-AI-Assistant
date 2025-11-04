"""
Batch Processing Endpoints
Efficient processing for multiple patients and queries
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Literal
from datetime import datetime, timezone
from enum import Enum
import queue
import threading
from dataclasses import dataclass, asdict

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks, status
from pydantic import BaseModel, Field, validator
import structlog

from ..utils.exceptions import (
    BatchProcessingError, 
    ValidationError, 
    RateLimitExceededError,
    ModelUnavailableError
)
from ..utils.security import SecurityValidator, rate_limiter
from ..utils.logger import get_logger
from ..config import get_settings

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter()

# Batch processing status enum
class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Batch job data structure"""
    job_id: str
    status: BatchStatus
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    estimated_completion: Optional[str]
    error_details: List[str]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class BatchProcessor:
    """Batch processor with medical data handling"""
    
    def __init__(self):
        self.active_jobs: Dict[str, BatchJob] = {}
        self.processing_queue = queue.Queue()
        self.max_concurrent_batches = 5
        self.max_items_per_batch = 100
        self.processing_threads = []
        self.running = True
        
        # Start processing threads
        for i in range(self.max_concurrent_batches):
            thread = threading.Thread(
                target=self._batch_processing_worker,
                args=(f"worker-{i}",),
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
    
    def create_batch_job(
        self, 
        requests: List[Dict[str, Any]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create new batch processing job"""
        
        job_id = str(uuid.uuid4())
        
        # Validate batch size
        if len(requests) > self.max_items_per_batch:
            raise ValidationError(f"Batch size exceeds maximum of {self.max_items_per_batch}")
        
        if len(requests) == 0:
            raise ValidationError("Batch cannot be empty")
        
        # Create job
        job = BatchJob(
            job_id=job_id,
            status=BatchStatus.PENDING,
            total_items=len(requests),
            processed_items=0,
            successful_items=0,
            failed_items=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            started_at=None,
            completed_at=None,
            estimated_completion=None,
            error_details=[],
            results=[],
            metadata=metadata or {}
        )
        
        self.active_jobs[job_id] = job
        self.processing_queue.put((job_id, requests))
        
        logger.info(
            "Batch job created",
            job_id=job_id,
            item_count=len(requests),
            metadata=metadata
        )
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get batch job status"""
        return self.active_jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel batch job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            if job.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
                job.status = BatchStatus.CANCELLED
                logger.info("Batch job cancelled", job_id=job_id)
                return True
        return False
    
    def _batch_processing_worker(self, worker_name: str):
        """Background worker for batch processing"""
        
        while self.running:
            try:
                job_id, requests = self.processing_queue.get(timeout=1)
                
                if job_id not in self.active_jobs:
                    continue
                
                job = self.active_jobs[job_id]
                if job.status == BatchStatus.CANCELLED:
                    continue
                
                # Start processing
                job.status = BatchStatus.PROCESSING
                job.started_at = datetime.now(timezone.utc).isoformat()
                
                # Calculate estimated completion
                estimated_duration = len(requests) * 0.1  # 0.1 seconds per item estimate
                job.estimated_completion = (
                    datetime.now(timezone.utc).timestamp() + estimated_duration
                )
                
                logger.info(
                    f"Starting batch processing {worker_name}",
                    job_id=job_id,
                    item_count=len(requests)
                )
                
                # Process items
                asyncio.run(self._process_batch_items(job, requests))
                
                # Complete job
                job.completed_at = datetime.now(timezone.utc).isoformat()
                if job.failed_items == 0:
                    job.status = BatchStatus.COMPLETED
                else:
                    job.status = BatchStatus.FAILED
                
                logger.info(
                    f"Batch processing completed {worker_name}",
                    job_id=job_id,
                    successful=job.successful_items,
                    failed=job.failed_items
                )
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Batch worker error {worker_name}: {e}")
                time.sleep(1)
    
    async def _process_batch_items(self, job: BatchJob, requests: List[Dict[str, Any]]):
        """Process items in batch"""
        
        semaphore = asyncio.Semaphore(10)  # Limit concurrent processing
        
        async def process_single_item(item_data: Dict[str, Any], index: int):
            async with semaphore:
                try:
                    # Process individual item
                    result = await self._process_single_item(item_data)
                    
                    # Update job status
                    job.processed_items += 1
                    job.successful_items += 1
                    job.results.append({
                        "index": index,
                        "success": True,
                        "result": result,
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    })
                    
                    logger.debug(
                        "Batch item processed",
                        job_id=job.job_id,
                        index=index,
                        success=True
                    )
                    
                except Exception as e:
                    # Handle item failure
                    job.processed_items += 1
                    job.failed_items += 1
                    error_msg = f"Item {index} failed: {str(e)}"
                    job.error_details.append(error_msg)
                    
                    job.results.append({
                        "index": index,
                        "success": False,
                        "error": str(e),
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    })
                    
                    logger.warning(
                        "Batch item processing failed",
                        job_id=job.job_id,
                        index=index,
                        error=str(e)
                    )
        
        # Process all items concurrently
        tasks = [
            process_single_item(item_data, i) 
            for i, item_data in enumerate(requests)
        ]
        
        await asyncio.gather(*tasks)
    
    async def _process_single_item(self, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single batch item"""
        
        # Simulate processing (in production, this would call actual model)
        await asyncio.sleep(0.1)
        
        query = item_data.get("query", "")
        medical_domain = item_data.get("medical_domain", "general")
        
        # Generate mock response
        response = f"Processed query: {query[:100]}... for {medical_domain} domain"
        confidence = 0.85
        
        return {
            "response": response,
            "confidence": confidence,
            "medical_context": {
                "domain": medical_domain,
                "query_length": len(query),
                "processing_notes": "Batch processed successfully"
            },
            "tokens_used": len(query.split()) * 1.3
        }
    
    def stop(self):
        """Stop batch processor"""
        self.running = False
        for thread in self.processing_threads:
            thread.join(timeout=1)


# Global batch processor
batch_processor = BatchProcessor()

# Pydantic models
class BatchRequest(BaseModel):
    """Batch processing request"""
    
    items: List[Dict[str, Any]] = Field(..., min_items=1, max_items=100, description="Items to process")
    priority: Literal["low", "normal", "high", "urgent"] = Field("normal", description="Processing priority")
    batch_name: Optional[str] = Field(None, max_length=100, description="Optional batch name")
    max_concurrent: Optional[int] = Field(10, ge=1, le=50, description="Maximum concurrent items")
    timeout_seconds: Optional[int] = Field(3600, ge=60, le=7200, description="Processing timeout")
    
    @validator('items')
    def validate_items(cls, v):
        if len(v) > 100:
            raise ValueError("Maximum 100 items per batch")
        return v
    
    def validate_item_format(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate individual item format"""
        
        required_fields = ["query"]
        for field in required_fields:
            if field not in item:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate query
        query = item["query"]
        if not isinstance(query, str) or len(query.strip()) == 0:
            raise ValidationError("Query must be a non-empty string")
        
        if len(query) > 5000:
            raise ValidationError("Query exceeds maximum length of 5000 characters")
        
        # Validate optional fields
        if "medical_domain" in item:
            valid_domains = [
                "general", "cardiology", "oncology", "neurology", 
                "emergency", "pediatrics", "psychiatry", "dermatology"
            ]
            if item["medical_domain"] not in valid_domains:
                raise ValidationError(f"Invalid medical domain: {item['medical_domain']}")
        
        if "urgency_level" in item:
            valid_levels = ["low", "medium", "high", "critical"]
            if item["urgency_level"] not in valid_levels:
                raise ValidationError(f"Invalid urgency level: {item['urgency_level']}")
        
        return item


class BatchStatusResponse(BaseModel):
    """Batch job status response"""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Job status")
    total_items: int = Field(..., description="Total items to process")
    processed_items: int = Field(..., description="Items processed so far")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    progress_percentage: float = Field(..., description="Progress percentage")
    created_at: str = Field(..., description="Job creation timestamp")
    started_at: Optional[str] = Field(None, description="Processing start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    estimated_duration_seconds: Optional[float] = Field(None, description="Estimated duration")
    error_details: List[str] = Field([], description="Error details")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "batch-job-uuid",
                "status": "processing",
                "total_items": 50,
                "processed_items": 25,
                "successful_items": 23,
                "failed_items": 2,
                "progress_percentage": 50.0,
                "created_at": "2024-01-15T10:00:00Z",
                "started_at": "2024-01-15T10:00:01Z",
                "completed_at": None,
                "estimated_completion": "2024-01-15T10:05:00Z",
                "estimated_duration_seconds": 300.0,
                "error_details": [
                    "Item 5: Medical validation failed",
                    "Item 12: Query too long"
                ]
            }
        }


class BatchResultsResponse(BaseModel):
    """Batch results response"""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Final status")
    total_items: int = Field(..., description="Total items processed")
    successful_items: int = Field(..., description="Successfully processed items")
    failed_items: int = Field(..., description="Failed items")
    total_processing_time: float = Field(..., description="Total processing time")
    average_processing_time: float = Field(..., description="Average time per item")
    results: List[Dict[str, Any]] = Field(..., description="Processing results")
    summary: Dict[str, Any] = Field(..., description="Processing summary")
    completed_at: str = Field(..., description="Completion timestamp")


# Endpoint implementations
@router.post("/create")
async def create_batch(
    request: BatchRequest,
    http_request: Request,
    background_tasks: BackgroundTasks
):
    """
    Create new batch processing job.
    
    Supports processing up to 100 medical queries simultaneously with:
    - Parallel processing for efficiency
    - Individual validation for each item
    - Progress tracking
    - Error handling and recovery
    - Medical domain specialization
    """
    
    # Rate limiting check
    client_id = http_request.client.host if http_request.client else "unknown"
    
    if rate_limiter.is_rate_limited(
        identifier=f"{client_id}:batch",
        limit=5,  # 5 batches per hour
        window=3600
    ):
        raise RateLimitExceededError("Batch creation rate limit exceeded")
    
    # Validate all items
    validated_items = []
    for i, item in enumerate(request.items):
        try:
            validated_item = request.validate_item_format(item)
            validated_item["batch_index"] = i
            validated_items.append(validated_item)
        except ValidationError as e:
            raise ValidationError(f"Item {i}: {e.detail}")
    
    # Create batch job
    batch_id = batch_processor.create_batch_job(
        requests=validated_items,
        metadata={
            "batch_name": request.batch_name,
            "priority": request.priority,
            "client_ip": http_request.client.host if http_request.client else None,
            "user_agent": http_request.headers.get("user-agent"),
            "max_concurrent": request.max_concurrent,
            "timeout_seconds": request.timeout_seconds
        }
    )
    
    logger.info(
        "Batch job created successfully",
        batch_id=batch_id,
        item_count=len(validated_items),
        priority=request.priority,
        client_ip=client_id
    )
    
    return {
        "batch_id": batch_id,
        "status": "created",
        "estimated_processing_time": len(validated_items) * 0.1,  # 0.1 sec per item
        "message": "Batch job created successfully and queued for processing"
    }


@router.get("/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(batch_id: str):
    """
    Get batch job status and progress information.
    
    Returns real-time status including:
    - Processing progress
    - Success/failure counts
    - Estimated completion time
    - Error details
    """
    
    job = batch_processor.get_job_status(batch_id)
    
    if not job:
        raise ValidationError(f"Batch job not found: {batch_id}")
    
    # Calculate progress
    progress_percentage = (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0
    
    # Calculate estimated duration
    estimated_duration = None
    if job.started_at and progress_percentage > 0:
        elapsed = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(job.started_at.replace('Z', '+00:00')).timestamp()
        estimated_duration = elapsed / (progress_percentage / 100)
    
    return BatchStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        total_items=job.total_items,
        processed_items=job.processed_items,
        successful_items=job.successful_items,
        failed_items=job.failed_items,
        progress_percentage=progress_percentage,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        estimated_completion=job.estimated_completion,
        estimated_duration_seconds=estimated_duration,
        error_details=job.error_details
    )


@router.get("/{batch_id}/results", response_model=BatchResultsResponse)
async def get_batch_results(batch_id: str):
    """
    Get complete batch results.
    
    Returns all processing results including:
    - Individual item results
    - Processing statistics
    - Error details
    - Medical validation results
    """
    
    job = batch_processor.get_job_status(batch_id)
    
    if not job:
        raise ValidationError(f"Batch job not found: {batch_id}")
    
    if job.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
        raise ValidationError(f"Batch job not completed yet: {batch_id}")
    
    # Calculate processing times
    total_processing_time = 0
    if job.started_at and job.completed_at:
        start_ts = datetime.fromisoformat(job.started_at.replace('Z', '+00:00')).timestamp()
        end_ts = datetime.fromisoformat(job.completed_at.replace('Z', '+00:00')).timestamp()
        total_processing_time = end_ts - start_ts
    
    average_processing_time = total_processing_time / job.total_items if job.total_items > 0 else 0
    
    # Generate summary
    summary = {
        "success_rate": (job.successful_items / job.total_items * 100) if job.total_items > 0 else 0,
        "failure_rate": (job.failed_items / job.total_items * 100) if job.total_items > 0 else 0,
        "average_confidence": _calculate_average_confidence(job.results),
        "medical_domains_processed": _extract_medical_domains(job.results),
        "processing_efficiency": {
            "items_per_second": job.total_items / total_processing_time if total_processing_time > 0 else 0,
            "throughput_rating": _calculate_throughput_rating(job.successful_items, total_processing_time)
        }
    }
    
    return BatchResultsResponse(
        job_id=job.job_id,
        status=job.status.value,
        total_items=job.total_items,
        successful_items=job.successful_items,
        failed_items=job.failed_items,
        total_processing_time=total_processing_time,
        average_processing_time=average_processing_time,
        results=job.results,
        summary=summary,
        completed_at=job.completed_at or datetime.now(timezone.utc).isoformat()
    )


@router.delete("/{batch_id}")
async def cancel_batch(batch_id: str):
    """
    Cancel batch processing job.
    
    Allows cancellation of pending or currently processing jobs.
    Returns partial results for completed items.
    """
    
    success = batch_processor.cancel_job(batch_id)
    
    if not success:
        raise ValidationError(f"Cannot cancel batch job: {batch_id}. Job may already be completed or not found.")
    
    logger.info("Batch job cancelled", batch_id=batch_id)
    
    return {
        "batch_id": batch_id,
        "status": "cancelled",
        "message": "Batch job cancelled successfully. Partial results may be available."
    }


@router.get("/active")
async def get_active_batches():
    """
    Get list of active batch jobs.
    
    Returns current status of all processing batch jobs.
    """
    
    active_jobs = []
    for job_id, job in batch_processor.active_jobs.items():
        if job.status in [BatchStatus.PENDING, BatchStatus.PROCESSING]:
            active_jobs.append({
                "job_id": job.job_id,
                "status": job.status.value,
                "total_items": job.total_items,
                "processed_items": job.processed_items,
                "progress_percentage": (job.processed_items / job.total_items * 100) if job.total_items > 0 else 0,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "estimated_completion": job.estimated_completion
            })
    
    return {
        "active_batches": active_jobs,
        "total_active": len(active_jobs),
        "queue_length": batch_processor.processing_queue.qsize()
    }


# Helper functions
def _calculate_average_confidence(results: List[Dict[str, Any]]) -> float:
    """Calculate average confidence score from results"""
    
    confidence_scores = []
    for result in results:
        if result.get("success") and "result" in result:
            # Extract confidence from result
            # This would depend on the actual result structure
            confidence_scores.append(0.85)  # Mock confidence
    
    return sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0


def _extract_medical_domains(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Extract medical domains processed"""
    
    domains = {}
    for result in results:
        if result.get("success") and "result" in result:
            domain = result["result"].get("medical_context", {}).get("domain", "general")
            domains[domain] = domains.get(domain, 0) + 1
    
    return domains


def _calculate_throughput_rating(successful_items: int, processing_time: float) -> str:
    """Calculate throughput rating"""
    
    if processing_time == 0:
        return "excellent"
    
    items_per_second = successful_items / processing_time
    
    if items_per_second >= 10:
        return "excellent"
    elif items_per_second >= 5:
        return "good"
    elif items_per_second >= 1:
        return "average"
    else:
        return "slow"