# Production API Billing and Quota Management System
# Usage tracking, billing automation, and quota enforcement

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time
import statistics
from collections import defaultdict, deque
import aiohttp
import aiofiles
import redis.asyncio as redis
import uuid
from decimal import Decimal, ROUND_HALF_UP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlanType(Enum):
    """API usage plan types"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"

class ResourceType(Enum):
    """Types of API resources"""
    API_REQUEST = "api_request"
    DATA_STORAGE = "data_storage"
    BANDWIDTH = "bandwidth"
    ANALYTICS_COMPUTE = "analytics_compute"
    FHIR_TRANSACTIONS = "fhir_transactions"
    WEBHOOK_DELIVERIES = "webhook_deliveries"
    REAL_TIME_CONNECTIONS = "real_time_connections"

class BillingPeriod(Enum):
    """Billing periods"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class UsageStatus(Enum):
    """Usage status"""
    ACTIVE = "active"
    EXCEEDED = "exceeded"
    THRESHOLD_WARNING = "threshold_warning"
    SUSPENDED = "suspended"

@dataclass
class QuotaLimit:
    """Quota limit configuration"""
    resource_type: ResourceType
    limit: int  # -1 for unlimited
    unit: str  # "requests", "GB", "hours", etc.
    period: BillingPeriod
    cost_per_unit: Decimal = Decimal('0.00')
    overage_rate: Decimal = Decimal('0.00')  # Higher rate for overages
    
@dataclass
class UsageRecord:
    """Individual usage record"""
    user_id: str
    api_key: str
    resource_type: ResourceType
    quantity: float
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class BillingSubscription:
    """Customer billing subscription"""
    customer_id: str
    plan_type: PlanType
    quotas: List[QuotaLimit]
    billing_period: BillingPeriod
    start_date: datetime
    end_date: Optional[datetime] = None
    status: str = "active"
    billing_email: str = ""
    company_name: str = ""
    
@dataclass
class UsageSummary:
    """Usage summary for a period"""
    user_id: str
    resource_type: ResourceType
    period_start: datetime
    period_end: datetime
    total_usage: float
    quota_limit: Optional[float]
    usage_percentage: float
    overage_usage: float
    estimated_cost: Decimal
    status: UsageStatus

class QuotaManager:
    """Manages API quotas and usage limits"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or redis.from_url("redis://localhost:6379")
        self.subscriptions: Dict[str, BillingSubscription] = {}
        self.usage_records: deque = deque(maxlen=1000000)  # Keep 1M records
        
    async def register_subscription(self, subscription: BillingSubscription):
        """Register a new billing subscription"""
        self.subscriptions[subscription.customer_id] = subscription
        await self._store_subscription(subscription)
        logger.info(f"Registered subscription for customer {subscription.customer_id} - Plan: {subscription.plan_type.value}")
    
    async def check_quota(
        self,
        user_id: str,
        resource_type: ResourceType,
        requested_quantity: float = 1.0
    ) -> Dict[str, Any]:
        """Check if user has quota available for resource"""
        
        subscription = await self._get_user_subscription(user_id)
        if not subscription:
            return {"allowed": False, "reason": "No active subscription"}
        
        # Find quota for resource type
        quota = self._find_quota(subscription.quotas, resource_type)
        if not quota:
            return {"allowed": False, "reason": f"No quota configured for {resource_type.value}"}
        
        # Check current usage
        current_usage = await self._get_current_usage(user_id, resource_type, quota.period)
        
        # Calculate remaining quota
        if quota.limit == -1:  # Unlimited
            remaining = float('inf')
        else:
            remaining = quota.limit - current_usage
        
        # Check if request would exceed quota
        allowed = requested_quantity <= remaining
        
        # Calculate overage if exceeding
        overage = 0.0
        if not allowed and quota.limit != -1:
            overage = requested_quantity - remaining
        
        # Determine status
        status = UsageStatus.ACTIVE
        if allowed and quota.limit != -1:
            usage_percentage = current_usage / quota.limit * 100
            if usage_percentage >= 90:
                status = UsageStatus.THRESHOLD_WARNING
            elif usage_percentage >= 100:
                status = UsageStatus.EXCEEDED
                allowed = False
        elif not allowed:
            status = UsageStatus.EXCEEDED
        
        return {
            "allowed": allowed,
            "remaining": remaining if remaining != float('inf') else -1,
            "total_limit": quota.limit,
            "current_usage": current_usage,
            "usage_percentage": current_usage / max(quota.limit, 1) * 100 if quota.limit != -1 else 0,
            "requested_quantity": requested_quantity,
            "overage_quantity": overage,
            "status": status.value,
            "estimated_overage_cost": float(overage) * float(quota.overage_rate) if overage > 0 else 0.0
        }
    
    async def record_usage(self, usage_record: UsageRecord):
        """Record usage for billing"""
        
        self.usage_records.append(usage_record)
        await self._store_usage_record(usage_record)
        
        # Update real-time usage in Redis
        await self._update_realtime_usage(usage_record)
        
        logger.info(f"Recorded usage: {usage_record.user_id} - {usage_record.resource_type.value} - {usage_record.quantity}")
    
    async def _get_user_subscription(self, user_id: str) -> Optional[BillingSubscription]:
        """Get user's active subscription"""
        # In production, this would query a database
        return self.subscriptions.get(user_id)
    
    def _find_quota(self, quotas: List[QuotaLimit], resource_type: ResourceType) -> Optional[QuotaLimit]:
        """Find quota configuration for resource type"""
        for quota in quotas:
            if quota.resource_type == resource_type:
                return quota
        return None
    
    async def _get_current_usage(
        self,
        user_id: str,
        resource_type: ResourceType,
        period: BillingPeriod
    ) -> float:
        """Get current usage for resource in billing period"""
        
        # Calculate period bounds
        now = datetime.now(timezone.utc)
        if period == BillingPeriod.MONTHLY:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif period == BillingPeriod.QUARTERLY:
            quarter = (now.month - 1) // 3
            period_start = datetime(now.year, quarter * 3 + 1, 1, tzinfo=timezone.utc)
        else:  # ANNUALLY
            period_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        
        # Sum usage records for the period
        total_usage = 0.0
        for record in self.usage_records:
            if (record.user_id == user_id and 
                record.resource_type == resource_type and 
                record.timestamp >= period_start):
                total_usage += record.quantity
        
        return total_usage
    
    async def _store_subscription(self, subscription: BillingSubscription):
        """Store subscription in persistent storage"""
        subscription_data = asdict(subscription)
        subscription_data['start_date'] = subscription.start_date.isoformat()
        if subscription.end_date:
            subscription_data['end_date'] = subscription.end_date.isoformat()
        
        await self.redis_client.hset(
            f"subscription:{subscription.customer_id}",
            mapping={
                k: json.dumps(v) if not isinstance(v, str) else v 
                for k, v in subscription_data.items()
            }
        )
    
    async def _store_usage_record(self, record: UsageRecord):
        """Store usage record in persistent storage"""
        record_data = {
            "user_id": record.user_id,
            "api_key": record.api_key,
            "resource_type": record.resource_type.value,
            "quantity": record.quantity,
            "timestamp": record.timestamp.isoformat(),
            "metadata": record.metadata
        }
        
        await self.redis_client.lpush(
            f"usage:{record.user_id}",
            json.dumps(record_data)
        )
        
        # Also store in time-series structure
        timestamp_score = record.timestamp.timestamp()
        await self.redis_client.zadd(
            f"usage_timeseries:{record.resource_type.value}",
            {f"{record.user_id}:{record.timestamp.isoformat()}": timestamp_score}
        )
    
    async def _update_realtime_usage(self, record: UsageRecord):
        """Update real-time usage counters in Redis"""
        # Update current hour usage
        hour_key = f"realtime_usage:{record.user_id}:{record.resource_type.value}:hour:{record.timestamp.strftime('%Y%m%d%H')}"
        await self.redis_client.incrbyfloat(hour_key, record.quantity)
        await self.redis_client.expire(hour_key, 86400)  # Expire after 24 hours
        
        # Update current day usage
        day_key = f"realtime_usage:{record.user_id}:{record.resource_type.value}:day:{record.timestamp.strftime('%Y%m%d')}"
        await self.redis_client.incrbyfloat(day_key, record.quantity)
        await self.redis_client.expire(day_key, 86400 * 7)  # Expire after 7 days

class BillingEngine:
    """Handles billing calculations and invoicing"""
    
    def __init__(self, quota_manager: QuotaManager):
        self.quota_manager = quota_manager
        
    async def calculate_monthly_bill(
        self,
        user_id: str,
        billing_period: BillingPeriod = BillingPeriod.MONTHLY
    ) -> Dict[str, Any]:
        """Calculate monthly bill for user"""
        
        subscription = await self.quota_manager._get_user_subscription(user_id)
        if not subscription:
            raise ValueError(f"No subscription found for user {user_id}")
        
        # Calculate period bounds
        now = datetime.now(timezone.utc)
        if billing_period == BillingPeriod.MONTHLY:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = now
            period_name = "Monthly"
        elif billing_period == BillingPeriod.QUARTERLY:
            quarter = (now.month - 1) // 3
            period_start = datetime(now.year, quarter * 3 + 1, 1, tzinfo=timezone.utc)
            period_end = now
            period_name = "Quarterly"
        else:
            period_start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            period_end = now
            period_name = "Annual"
        
        # Calculate usage for each resource type
        resource_usage = []
        total_cost = Decimal('0.00')
        
        for quota in subscription.quotas:
            usage = await self._calculate_resource_usage(
                user_id, quota, period_start, period_end
            )
            
            cost = await self._calculate_resource_cost(usage, quota)
            total_cost += cost
            
            resource_usage.append({
                "resource_type": quota.resource_type.value,
                "usage": usage["total_usage"],
                "limit": quota.limit,
                "usage_percentage": usage["usage_percentage"],
                "overage": usage["overage"],
                "base_cost": float(quota.cost_per_unit) * usage["total_usage"],
                "overage_cost": float(cost) - float(quota.cost_per_unit) * usage["total_usage"],
                "total_cost": float(cost),
                "unit": quota.unit
            })
        
        return {
            "customer_id": user_id,
            "billing_period": period_name,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "plan_type": subscription.plan_type.value,
            "subscription_status": subscription.status,
            "resource_breakdown": resource_usage,
            "subtotal": float(total_cost),
            "tax_rate": 0.08,  # 8% tax
            "tax_amount": float(total_cost * Decimal('0.08')),
            "total_amount": float(total_cost * Decimal('1.08')),
            "currency": "USD",
            "generated_at": now.isoformat()
        }
    
    async def _calculate_resource_usage(
        self,
        user_id: str,
        quota: QuotaLimit,
        period_start: datetime,
        period_end: datetime
    ) -> Dict[str, float]:
        """Calculate usage for a specific resource in the billing period"""
        
        # Get usage records for the period
        total_usage = 0.0
        for record in self.quota_manager.usage_records:
            if (record.user_id == user_id and 
                record.resource_type == quota.resource_type and 
                period_start <= record.timestamp <= period_end):
                total_usage += record.quantity
        
        # Calculate overage
        overage = 0.0
        if quota.limit != -1 and total_usage > quota.limit:
            overage = total_usage - quota.limit
        
        usage_percentage = (total_usage / quota.limit * 100) if quota.limit != -1 else 0
        
        return {
            "total_usage": total_usage,
            "overage": overage,
            "usage_percentage": usage_percentage
        }
    
    async def _calculate_resource_cost(self, usage: Dict[str, float], quota: QuotaLimit) -> Decimal:
        """Calculate cost for resource usage"""
        
        base_cost = Decimal(str(usage["total_usage"])) * quota.cost_per_unit
        overage_cost = Decimal(str(usage["overage"])) * quota.overage_rate
        
        return base_cost + overage_cost
    
    async def generate_invoice(
        self,
        user_id: str,
        billing_period: BillingPeriod = BillingPeriod.MONTHLY
    ) -> Dict[str, Any]:
        """Generate detailed invoice for user"""
        
        bill = await self.calculate_monthly_bill(user_id, billing_period)
        
        # Generate invoice number
        invoice_number = f"INV-{datetime.now().strftime('%Y%m%d')}-{user_id[:8].upper()}"
        
        invoice = {
            "invoice_number": invoice_number,
            "customer_id": user_id,
            "customer_email": await self._get_customer_email(user_id),
            "customer_name": await self._get_customer_name(user_id),
            "billing_address": await self._get_customer_address(user_id),
            "invoice_date": datetime.now(timezone.utc).isoformat(),
            "due_date": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
            "line_items": [
                {
                    "description": f"{item['resource_type'].replace('_', ' ').title()} Usage",
                    "quantity": item["usage"],
                    "unit": item["unit"],
                    "unit_price": item["base_cost"] / item["usage"] if item["usage"] > 0 else 0,
                    "amount": item["total_cost"]
                }
                for item in bill["resource_breakdown"]
            ],
            "subtotal": bill["subtotal"],
            "tax_rate": bill["tax_rate"],
            "tax_amount": bill["tax_amount"],
            "total_amount": bill["total_amount"],
            "currency": bill["currency"],
            "payment_terms": "Net 30 days",
            "notes": "Thank you for your business!"
        }
        
        return invoice
    
    async def _get_customer_email(self, user_id: str) -> str:
        """Get customer email address"""
        # In production, this would query a customer database
        subscription = await self.quota_manager._get_user_subscription(user_id)
        return subscription.billing_email if subscription else "customer@example.com"
    
    async def _get_customer_name(self, user_id: str) -> str:
        """Get customer company name"""
        subscription = await self.quota_manager._get_user_subscription(user_id)
        return subscription.company_name if subscription else "Customer"
    
    async def _get_customer_address(self, user_id: str) -> Dict[str, str]:
        """Get customer billing address"""
        # In production, this would query a customer database
        return {
            "line1": "123 Healthcare Ave",
            "line2": "Suite 456",
            "city": "Medical City",
            "state": "HC",
            "postal_code": "12345",
            "country": "USA"
        }

class UsageAnalytics:
    """Analytics for API usage patterns"""
    
    def __init__(self, quota_manager: QuotaManager):
        self.quota_manager = quota_manager
    
    async def generate_usage_analytics(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Generate usage analytics for user"""
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Get usage for the period
        usage_data = await self._get_usage_by_day(user_id, start_date, datetime.now(timezone.utc))
        
        # Calculate analytics
        total_requests = sum(day["total_requests"] for day in usage_data)
        avg_daily_requests = total_requests / max(days, 1)
        
        # Peak usage analysis
        peak_day = max(usage_data, key=lambda x: x["total_requests"]) if usage_data else None
        peak_usage = peak_day["total_requests"] if peak_day else 0
        
        # Resource distribution
        resource_distribution = defaultdict(float)
        for record in self.quota_manager.usage_records:
            if (record.user_id == user_id and record.timestamp >= start_date):
                resource_distribution[record.resource_type.value] += record.quantity
        
        # Usage trends
        usage_trend = self._calculate_trend(usage_data)
        
        return {
            "user_id": user_id,
            "period_days": days,
            "period_start": start_date.isoformat(),
            "period_end": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_requests": total_requests,
                "average_daily_requests": round(avg_daily_requests, 2),
                "peak_day_requests": peak_usage,
                "peak_day": peak_day["date"] if peak_day else None
            },
            "resource_breakdown": dict(resource_distribution),
            "daily_usage": usage_data,
            "usage_trend": usage_trend,
            "quota_utilization": await self._analyze_quota_utilization(user_id),
            "cost_forecast": await self._forecast_monthly_cost(user_id)
        }
    
    async def _get_usage_by_day(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get usage data grouped by day"""
        
        daily_usage = defaultdict(lambda: {
            "total_requests": 0,
            "error_requests": 0,
            "unique_endpoints": set()
        })
        
        for record in self.quota_manager.usage_records:
            if (record.user_id == user_id and 
                start_date <= record.timestamp <= end_date and
                record.resource_type == ResourceType.API_REQUEST):
                
                day_key = record.timestamp.strftime("%Y-%m-%d")
                daily_usage[day_key]["total_requests"] += record.quantity
        
        # Convert to serializable format and sort by date
        usage_data = []
        for date_str, data in sorted(daily_usage.items()):
            usage_data.append({
                "date": date_str,
                "total_requests": data["total_requests"],
                "unique_endpoints": len(data["unique_endpoints"])
            })
        
        return usage_data
    
    def _calculate_trend(self, usage_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate usage trend"""
        
        if len(usage_data) < 2:
            return {"trend": "insufficient_data", "change_percent": 0}
        
        # Simple trend calculation
        first_half = usage_data[:len(usage_data)//2]
        second_half = usage_data[len(usage_data)//2:]
        
        first_avg = sum(day["total_requests"] for day in first_half) / max(len(first_half), 1)
        second_avg = sum(day["total_requests"] for day in second_half) / max(len(second_half), 1)
        
        if second_avg > first_avg * 1.1:
            trend = "increasing"
        elif second_avg < first_avg * 0.9:
            trend = "decreasing"
        else:
            trend = "stable"
        
        change_percent = ((second_avg - first_avg) / max(first_avg, 1)) * 100
        
        return {
            "trend": trend,
            "change_percent": round(change_percent, 2),
            "first_period_avg": round(first_avg, 2),
            "second_period_avg": round(second_avg, 2)
        }
    
    async def _analyze_quota_utilization(self, user_id: str) -> Dict[str, Any]:
        """Analyze quota utilization for user"""
        
        subscription = await self.quota_manager._get_user_subscription(user_id)
        if not subscription:
            return {}
        
        utilization = {}
        for quota in subscription.quotas:
            current_usage = await self.quota_manager._get_current_usage(
                user_id, quota.resource_type, quota.period
            )
            
            utilization[quota.resource_type.value] = {
                "current_usage": current_usage,
                "quota_limit": quota.limit,
                "utilization_percent": (current_usage / max(quota.limit, 1)) * 100 if quota.limit != -1 else 0,
                "days_remaining": await self._get_days_remaining(quota.period),
                "projected_monthly_usage": current_usage * 30 / 30  # Simplified projection
            }
        
        return utilization
    
    async def _get_days_remaining(self, period: BillingPeriod) -> int:
        """Get days remaining in billing period"""
        
        now = datetime.now(timezone.utc)
        if period == BillingPeriod.MONTHLY:
            next_month = now.replace(day=28) + timedelta(days=4)
            period_end = next_month - timedelta(days=next_month.day)
            return (period_end - now).days
        elif period == BillingPeriod.QUARTERLY:
            quarter_end_months = [3, 6, 9, 12]
            current_quarter_end = next(m for m in quarter_end_months if m >= now.month)
            period_end = datetime(now.year, current_quarter_end, 
                                [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][current_quarter_end-1], 
                                tzinfo=timezone.utc)
            return (period_end - now).days
        else:  # ANNUAL
            period_end = datetime(now.year, 12, 31, tzinfo=timezone.utc)
            return (period_end - now).days
    
    async def _forecast_monthly_cost(self, user_id: str) -> Dict[str, Any]:
        """Forecast monthly cost based on current usage"""
        
        subscription = await self.quota_manager._get_user_subscription(user_id)
        if not subscription:
            return {}
        
        # Calculate current rate
        current_usage = 0
        current_cost = Decimal('0.00')
        
        for quota in subscription.quotas:
            usage = await self.quota_manager._get_current_usage(
                user_id, quota.resource_type, quota.period
            )
            
            # Scale to monthly
            monthly_usage = usage * 30 / 30  # Simplified - assume current period is monthly
            monthly_cost = Decimal(str(monthly_usage)) * quota.cost_per_unit
            
            current_usage += monthly_usage
            current_cost += monthly_cost
        
        # Forecast based on trend (simplified)
        forecast_cost = current_cost * Decimal('1.1')  # Assume 10% growth
        
        return {
            "current_monthly_cost": float(current_cost),
            "forecasted_monthly_cost": float(forecast_cost),
            "growth_assumption": "10% monthly growth",
            "confidence_level": "medium"
        }

# Default usage plans for healthcare API
DEFAULT_PLANS = {
    PlanType.FREE: {
        "name": "Free Tier",
        "monthly_cost": 0.00,
        "quotas": [
            QuotaLimit(ResourceType.API_REQUEST, 1000, "requests", BillingPeriod.MONTHLY),
            QuotaLimit(ResourceType.FHIR_TRANSACTIONS, 100, "transactions", BillingPeriod.MONTHLY),
            QuotaLimit(ResourceType.WEBHOOK_DELIVERIES, 50, "deliveries", BillingPeriod.MONTHLY)
        ]
    },
    PlanType.BASIC: {
        "name": "Basic Plan",
        "monthly_cost": 99.00,
        "quotas": [
            QuotaLimit(ResourceType.API_REQUEST, 10000, "requests", BillingPeriod.MONTHLY, Decimal('0.01')),
            QuotaLimit(ResourceType.FHIR_TRANSACTIONS, 1000, "transactions", BillingPeriod.MONTHLY, Decimal('0.10')),
            QuotaLimit(ResourceType.WEBHOOK_DELIVERIES, 500, "deliveries", BillingPeriod.MONTHLY, Decimal('0.05')),
            QuotaLimit(ResourceType.DATA_STORAGE, 10, "GB", BillingPeriod.MONTHLY, Decimal('1.00')),
            QuotaLimit(ResourceType.ANALYTICS_COMPUTE, 100, "hours", BillingPeriod.MONTHLY, Decimal('2.00'))
        ]
    },
    PlanType.PROFESSIONAL: {
        "name": "Professional Plan",
        "monthly_cost": 299.00,
        "quotas": [
            QuotaLimit(ResourceType.API_REQUEST, 50000, "requests", BillingPeriod.MONTHLY, Decimal('0.008')),
            QuotaLimit(ResourceType.FHIR_TRANSACTIONS, 5000, "transactions", BillingPeriod.MONTHLY, Decimal('0.08')),
            QuotaLimit(ResourceType.WEBHOOK_DELIVERIES, 2000, "deliveries", BillingPeriod.MONTHLY, Decimal('0.03')),
            QuotaLimit(ResourceType.DATA_STORAGE, 100, "GB", BillingPeriod.MONTHLY, Decimal('0.80')),
            QuotaLimit(ResourceType.ANALYTICS_COMPUTE, 500, "hours", BillingPeriod.MONTHLY, Decimal('1.50')),
            QuotaLimit(ResourceType.REAL_TIME_CONNECTIONS, 100, "connections", BillingPeriod.MONTHLY, Decimal('1.00'))
        ]
    },
    PlanType.ENTERPRISE: {
        "name": "Enterprise Plan",
        "monthly_cost": 999.00,
        "quotas": [
            QuotaLimit(ResourceType.API_REQUEST, 200000, "requests", BillingPeriod.MONTHLY, Decimal('0.005')),
            QuotaLimit(ResourceType.FHIR_TRANSACTIONS, 20000, "transactions", BillingPeriod.MONTHLY, Decimal('0.05')),
            QuotaLimit(ResourceType.WEBHOOK_DELIVERIES, 10000, "deliveries", BillingPeriod.MONTHLY, Decimal('0.02')),
            QuotaLimit(ResourceType.DATA_STORAGE, 1000, "GB", BillingPeriod.MONTHLY, Decimal('0.50')),
            QuotaLimit(ResourceType.ANALYTICS_COMPUTE, 2000, "hours", BillingPeriod.MONTHLY, Decimal('1.00')),
            QuotaLimit(ResourceType.REAL_TIME_CONNECTIONS, 500, "connections", BillingPeriod.MONTHLY, Decimal('0.50'))
        ]
    }
}

# Example usage
if __name__ == "__main__":
    async def main():
        # Initialize billing system
        quota_manager = QuotaManager()
        billing_engine = BillingEngine(quota_manager)
        analytics = UsageAnalytics(quota_manager)
        
        # Register sample subscription
        subscription = BillingSubscription(
            customer_id="hospital_001",
            plan_type=PlanType.PROFESSIONAL,
            quotas=DEFAULT_PLANS[PlanType.PROFESSIONAL]["quotas"],
            billing_period=BillingPeriod.MONTHLY,
            start_date=datetime.now(timezone.utc),
            billing_email="billing@hospital.org",
            company_name="Healthcare Hospital System"
        )
        
        await quota_manager.register_subscription(subscription)
        
        # Check quota
        quota_check = await quota_manager.check_quota(
            "hospital_001",
            ResourceType.API_REQUEST,
            50
        )
        print(f"Quota check: {quota_check}")
        
        # Record usage
        usage_record = UsageRecord(
            user_id="hospital_001",
            api_key="hospital_api_key_123",
            resource_type=ResourceType.API_REQUEST,
            quantity=1.0,
            timestamp=datetime.now(timezone.utc),
            metadata={"endpoint": "/api/v1/patients", "method": "GET"}
        )
        
        await quota_manager.record_usage(usage_record)
        
        # Calculate bill
        bill = await billing_engine.calculate_monthly_bill("hospital_001")
        print(f"Monthly bill: ${bill['total_amount']:.2f}")
        
        # Generate analytics
        analytics_report = await analytics.generate_usage_analytics("hospital_001", days=7)
        print(f"Total requests in last 7 days: {analytics_report['summary']['total_requests']}")
    
    asyncio.run(main())