"""
Customer Success Tracking and Reporting System
Healthcare-focused success metrics and KPI tracking
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
import logging

from config.support_config import SupportConfig

logger = logging.getLogger(__name__)

class SuccessCategory(Enum):
    USER_ADOPTION = "user_adoption"
    SYSTEM_USAGE = "system_usage"
    CLINICAL_OUTCOMES = "clinical_outcomes"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    CUSTOMER_SATISFACTION = "customer_satisfaction"
    COMPLIANCE_ADHERENCE = "compliance_adherence"
    ROI_MEASUREMENT = "roi_measurement"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    AT_RISK = "at_risk"
    CRITICAL = "critical"
    EXCELLENT = "excellent"

class MilestoneType(Enum):
    ONBOARDING_COMPLETE = "onboarding_complete"
    FIRST_CLINICAL_USE = "first_clinical_use"
    USAGE_MILESTONE = "usage_milestone"
    TRAINING_CERTIFICATION = "training_certification"
    INTEGRATION_COMPLETE = "integration_complete"
    ROI_ACHIEVED = "roi_achieved"

@dataclass
class HealthcareKPIs:
    """Healthcare-specific key performance indicators"""
    patient_safety_score: float
    clinical_workflow_efficiency: float
    regulatory_compliance_rate: float
    user_adoption_rate: float
    system_uptime_percentage: float
    average_response_time: float
    incident_resolution_time: float
    training_completion_rate: float

@dataclass
class CustomerHealthMetrics:
    """Customer health assessment metrics"""
    facility_id: str
    facility_name: str
    facility_type: str
    number_of_users: int
    last_active_date: datetime
    health_score: float
    health_status: HealthStatus
    adoption_milestones: List[Dict[str, Any]]
    usage_metrics: Dict[str, float]
    satisfaction_metrics: Dict[str, float]
    support_ticket_count: int
    incident_count: int
    training_completion_rate: float

@dataclass
class ROIMetrics:
    """Return on Investment metrics"""
    facility_id: str
    investment_amount: float
    estimated_savings: float
    efficiency_gains: float
    time_savings_hours: float
    error_reduction_percentage: float
    patient_outcome_improvements: float
    roi_percentage: float
    payback_period_months: float
    measurement_date: datetime

@dataclass
class SuccessMilestone:
    """Customer success milestone"""
    id: str
    customer_id: str
    milestone_type: MilestoneType
    title: str
    description: str
    achieved_date: datetime
    impact_score: float
    celebration_required: bool

class HealthcareKPIAnalyzer:
    """Healthcare-specific KPI analysis engine"""
    
    def __init__(self):
        self.benchmark_data = {
            "patient_safety_score": {
                "excellent": 95,
                "healthy": 85,
                "at_risk": 70,
                "critical": 50
            },
            "clinical_workflow_efficiency": {
                "excellent": 90,
                "healthy": 80,
                "at_risk": 60,
                "critical": 40
            },
            "regulatory_compliance_rate": {
                "excellent": 99,
                "healthy": 95,
                "at_risk": 85,
                "critical": 70
            },
            "user_adoption_rate": {
                "excellent": 90,
                "healthy": 75,
                "at_risk": 50,
                "critical": 25
            }
        }
    
    def calculate_health_score(self, kpis: HealthcareKPIs) -> Tuple[float, HealthStatus]:
        """Calculate overall customer health score"""
        
        # Weighted scores for different KPIs
        weights = {
            "patient_safety_score": 0.25,
            "clinical_workflow_efficiency": 0.20,
            "regulatory_compliance_rate": 0.20,
            "user_adoption_rate": 0.15,
            "system_uptime_percentage": 0.10,
            "average_response_time": 0.05,
            "incident_resolution_time": 0.03,
            "training_completion_rate": 0.02
        }
        
        # Normalize response time and resolution time (lower is better)
        normalized_response_time = max(0, 100 - (kpis.average_response_time / 1000 * 100))  # Assume 1000ms baseline
        normalized_resolution_time = max(0, 100 - (kpis.incident_resolution_time / 24 * 100))  # Assume 24h baseline
        
        # Calculate weighted health score
        health_score = (
            kpis.patient_safety_score * weights["patient_safety_score"] +
            kpis.clinical_workflow_efficiency * weights["clinical_workflow_efficiency"] +
            kpis.regulatory_compliance_rate * weights["regulatory_compliance_rate"] +
            kpis.user_adoption_rate * weights["user_adoption_rate"] +
            kpis.system_uptime_percentage * weights["system_uptime_percentage"] +
            normalized_response_time * weights["average_response_time"] +
            normalized_resolution_time * weights["incident_resolution_time"] +
            kpis.training_completion_rate * weights["training_completion_rate"]
        )
        
        # Determine health status
        if health_score >= 90:
            status = HealthStatus.EXCELLENT
        elif health_score >= 75:
            status = HealthStatus.HEALTHY
        elif health_score >= 50:
            status = HealthStatus.AT_RISK
        else:
            status = HealthStatus.CRITICAL
        
        return health_score, status
    
    def analyze_trends(self, current_kpis: HealthcareKPIs, previous_kpis: HealthcareKPIs) -> Dict[str, Any]:
        """Analyze KPI trends"""
        
        trends = {}
        
        # Calculate percentage changes
        kpi_fields = [
            "patient_safety_score", "clinical_workflow_efficiency", "regulatory_compliance_rate",
            "user_adoption_rate", "system_uptime_percentage", "average_response_time",
            "incident_resolution_time", "training_completion_rate"
        ]
        
        for field in kpi_fields:
            current_value = getattr(current_kpis, field)
            previous_value = getattr(previous_kpis, field)
            
            if previous_value != 0:
                change_percentage = ((current_value - previous_value) / previous_value) * 100
            else:
                change_percentage = 0 if current_value == 0 else 100
            
            trends[field] = {
                "current_value": current_value,
                "previous_value": previous_value,
                "change_percentage": change_percentage,
                "trend_direction": "improving" if change_percentage > 0 else "declining" if change_percentage < 0 else "stable"
            }
        
        return trends

class CustomerSuccessSystem:
    """Main customer success tracking system"""
    
    def __init__(self):
        self.customers: Dict[str, CustomerHealthMetrics] = {}
        self.kpi_history: Dict[str, List[HealthcareKPIs]] = defaultdict(list)
        self.roi_metrics: Dict[str, ROIMetrics] = {}
        self.milestones: Dict[str, List[SuccessMilestone]] = defaultdict(list)
        self.success_counter = 0
        self.kpi_analyzer = HealthcareKPIAnalyzer()
        
        # Success thresholds and alerts
        self.alert_thresholds = {
            "health_score_critical": 50,
            "health_score_at_risk": 75,
            "inactive_days_threshold": 7,
            "support_ticket_threshold": 10,
            "incident_threshold": 5
        }
    
    async def register_customer(
        self,
        facility_id: str,
        facility_name: str,
        facility_type: str,
        number_of_users: int,
        initial_kpis: Optional[HealthcareKPIs] = None
    ) -> CustomerHealthMetrics:
        """Register a new customer for success tracking"""
        
        if initial_kpis is None:
            # Default KPIs for new customers
            initial_kpis = HealthcareKPIs(
                patient_safety_score=75.0,
                clinical_workflow_efficiency=70.0,
                regulatory_compliance_rate=90.0,
                user_adoption_rate=30.0,  # Low initially
                system_uptime_percentage=99.0,
                average_response_time=2000.0,  # 2 seconds
                incident_resolution_time=48.0,  # 48 hours
                training_completion_rate=0.0  # No training completed yet
            )
        
        # Calculate initial health score
        health_score, health_status = self.kpi_analyzer.calculate_health_score(initial_kpis)
        
        customer = CustomerHealthMetrics(
            facility_id=facility_id,
            facility_name=facility_name,
            facility_type=facility_type,
            number_of_users=number_of_users,
            last_active_date=datetime.now(),
            health_score=health_score,
            health_status=health_status,
            adoption_milestones=[],
            usage_metrics={},
            satisfaction_metrics={},
            support_ticket_count=0,
            incident_count=0,
            training_completion_rate=0.0
        )
        
        self.customers[facility_id] = customer
        self.kpi_history[facility_id].append(initial_kpis)
        
        # Create initial milestone
        await self._create_milestone(
            facility_id,
            MilestoneType.ONBOARDING_COMPLETE,
            "Customer Onboarded",
            "Successfully onboarded to medical AI system"
        )
        
        logger.info(f"Registered customer {facility_name} ({facility_id}) with health score: {health_score:.1f}")
        return customer
    
    async def update_customer_kpis(
        self,
        facility_id: str,
        new_kpis: HealthcareKPIs
    ) -> CustomerHealthMetrics:
        """Update customer KPIs and calculate health status"""
        
        if facility_id not in self.customers:
            raise ValueError(f"Customer {facility_id} not found")
        
        customer = self.customers[facility_id]
        
        # Store KPI history
        self.kpi_history[facility_id].append(new_kpis)
        
        # Calculate new health score
        health_score, health_status = self.kpi_analyzer.calculate_health_score(new_kpis)
        
        # Update customer metrics
        customer.health_score = health_score
        customer.health_status = health_status
        customer.last_active_date = datetime.now()
        
        # Check for milestone achievements
        await self._check_milestone_achievements(facility_id, new_kpis)
        
        # Check for health alerts
        await self._check_health_alerts(customer)
        
        logger.info(f"Updated KPIs for {facility_id}. Health score: {health_score:.1f} ({health_status.value})")
        return customer
    
    async def record_roi_metrics(
        self,
        facility_id: str,
        investment_amount: float,
        savings_calculated: Dict[str, float],
        measurement_date: Optional[datetime] = None
    ) -> ROIMetrics:
        """Record ROI metrics for customer"""
        
        if facility_id not in self.customers:
            raise ValueError(f"Customer {facility_id} not found")
        
        measurement_date = measurement_date or datetime.now()
        
        # Calculate total savings
        total_savings = sum(savings_calculated.values())
        
        # Calculate ROI percentage
        roi_percentage = ((total_savings - investment_amount) / investment_amount * 100) if investment_amount > 0 else 0
        
        # Calculate payback period
        monthly_investment = investment_amount / 12  # Assuming annual investment
        monthly_savings = total_savings / 12  # Assuming annual savings
        payback_period_months = (monthly_investment / monthly_savings) if monthly_savings > 0 else 0
        
        roi_metrics = ROIMetrics(
            facility_id=facility_id,
            investment_amount=investment_amount,
            estimated_savings=total_savings,
            efficiency_gains=savings_calculated.get("efficiency_gains", 0),
            time_savings_hours=savings_calculated.get("time_savings_hours", 0),
            error_reduction_percentage=savings_calculated.get("error_reduction_percentage", 0),
            patient_outcome_improvements=savings_calculated.get("patient_outcome_improvements", 0),
            roi_percentage=roi_percentage,
            payback_period_months=payback_period_months,
            measurement_date=measurement_date
        )
        
        self.roi_metrics[facility_id] = roi_metrics
        
        # Check for ROI achievement milestone
        if roi_percentage > 100:  # More than 100% ROI
            await self._create_milestone(
                facility_id,
                MilestoneType.ROI_ACHIEVED,
                "ROI Target Achieved",
                f"Achieved {roi_percentage:.1f}% return on investment"
            )
        
        logger.info(f"Recorded ROI metrics for {facility_id}: {roi_percentage:.1f}% ROI")
        return roi_metrics
    
    async def get_customer_health_dashboard(self, facility_id: str) -> Dict[str, Any]:
        """Get comprehensive customer health dashboard"""
        
        if facility_id not in self.customers:
            raise ValueError(f"Customer {facility_id} not found")
        
        customer = self.customers[facility_id]
        kpi_history = self.kpi_history[facility_id]
        
        # Get latest KPIs
        current_kpis = kpi_history[-1] if kpi_history else None
        previous_kpis = kpi_history[-2] if len(kpi_history) > 1 else None
        
        # Calculate trends
        trends = {}
        if current_kpis and previous_kpis:
            trends = self.kpi_analyzer.analyze_trends(current_kpis, previous_kpis)
        
        # Get ROI data
        roi_data = self.roi_metrics.get(facility_id)
        
        # Get recent milestones
        recent_milestones = [
            milestone for milestone in self.milestones.get(facility_id, [])
            if milestone.achieved_date >= datetime.now() - timedelta(days=90)
        ]
        
        # Calculate engagement metrics
        days_since_active = (datetime.now() - customer.last_active_date).days
        engagement_status = "active" if days_since_active <= 7 else "inactive"
        
        return {
            "customer_info": {
                "facility_id": facility_id,
                "facility_name": customer.facility_name,
                "facility_type": customer.facility_type,
                "number_of_users": customer.number_of_users
            },
            "health_status": {
                "overall_score": customer.health_score,
                "status": customer.health_status.value,
                "days_since_active": days_since_active,
                "engagement_status": engagement_status
            },
            "current_kpis": asdict(current_kpis) if current_kpis else None,
            "kpi_trends": trends,
            "roi_metrics": asdict(roi_data) if roi_data else None,
            "milestones": [
                {
                    "type": milestone.milestone_type.value,
                    "title": milestone.title,
                    "description": milestone.description,
                    "achieved_date": milestone.achieved_date.isoformat(),
                    "impact_score": milestone.impact_score
                }
                for milestone in recent_milestones
            ],
            "support_metrics": {
                "support_tickets": customer.support_ticket_count,
                "incidents": customer.incident_count,
                "training_completion": customer.training_completion_rate
            },
            "recommendations": await self._generate_customer_recommendations(customer, trends)
        }
    
    async def generate_health_report(self, facility_id: str, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive health report for customer"""
        
        if facility_id not in self.customers:
            raise ValueError(f"Customer {facility_id} not found")
        
        customer = self.customers[facility_id]
        kpi_history = self.kpi_history[facility_id]
        
        # Filter KPI history for the report period
        start_date = datetime.now() - timedelta(days=days)
        period_kpis = [
            kpi for kpi in kpi_history
            if len(self.kpi_history[facility_id]) == 0 or 
            (len(self.kpi_history[facility_id]) > 0 and 
             max(self.kpi_history[facility_id][-len(self.kpi_history[facility_id]):]) >= start_date)
        ]
        
        if len(period_kpis) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Calculate period metrics
        start_kpis = period_kpis[0]
        end_kpis = period_kpis[-1]
        
        trends = self.kpi_analyzer.analyze_trends(end_kpis, start_kpis)
        
        # Calculate improvement areas
        improving_areas = [kpi for kpi, trend in trends.items() if trend["trend_direction"] == "improving"]
        declining_areas = [kpi for kpi, trend in trends.items() if trend["trend_direction"] == "declining"]
        
        # Get milestones achieved in period
        period_milestones = [
            milestone for milestone in self.milestones.get(facility_id, [])
            if milestone.achieved_date >= start_date
        ]
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": datetime.now().isoformat(),
                "duration_days": days
            },
            "customer_summary": {
                "facility_name": customer.facility_name,
                "facility_type": customer.facility_type,
                "current_health_score": customer.health_score,
                "health_status": customer.health_status.value
            },
            "performance_analysis": {
                "start_health_score": self.kpi_analyzer.calculate_health_score(start_kpis)[0],
                "end_health_score": self.kpi_analyzer.calculate_health_score(end_kpis)[0],
                "overall_trend": "improving" if customer.health_score > self.kpi_analyzer.calculate_health_score(start_kpis)[0] else "declining",
                "improving_areas": improving_areas,
                "declining_areas": declining_areas
            },
            "detailed_trends": trends,
            "milestones_achieved": len(period_milestones),
            "roi_analysis": asdict(self.roi_metrics.get(facility_id)) if facility_id in self.roi_metrics else None,
            "recommendations": await self._generate_customer_recommendations(customer, trends)
        }
    
    async def get_at_risk_customers(self) -> List[CustomerHealthMetrics]:
        """Get list of customers at risk"""
        
        at_risk_customers = []
        
        for customer in self.customers.values():
            # Check multiple risk indicators
            risk_indicators = []
            
            if customer.health_status in [HealthStatus.AT_RISK, HealthStatus.CRITICAL]:
                risk_indicators.append("low_health_score")
            
            days_since_active = (datetime.now() - customer.last_active_date).days
            if days_since_active > self.alert_thresholds["inactive_days_threshold"]:
                risk_indicators.append("inactive_usage")
            
            if customer.support_ticket_count > self.alert_thresholds["support_ticket_threshold"]:
                risk_indicators.append("high_support_volume")
            
            if customer.incident_count > self.alert_thresholds["incident_threshold"]:
                risk_indicators.append("high_incident_count")
            
            if customer.training_completion_rate < 50:
                risk_indicators.append("low_training_completion")
            
            # Add risk indicators to customer object
            if risk_indicators:
                customer_dict = asdict(customer)
                customer_dict["risk_indicators"] = risk_indicators
                at_risk_customers.append(customer)
        
        return sorted(at_risk_customers, key=lambda x: x.health_score)
    
    async def create_success_celebration(self, facility_id: str, milestone: SuccessMilestone) -> Dict[str, Any]:
        """Create celebration for customer milestone achievement"""
        
        celebration_data = {
            "customer_id": facility_id,
            "milestone": {
                "type": milestone.milestone_type.value,
                "title": milestone.title,
                "description": milestone.description,
                "achieved_date": milestone.achieved_date.isoformat(),
                "impact_score": milestone.impact_score
            },
            "celebration_type": self._determine_celebration_type(milestone),
            "actions": self._get_celebration_actions(milestone)
        }
        
        logger.info(f"Created celebration for {facility_id}: {milestone.title}")
        return celebration_data
    
    async def _create_milestone(
        self,
        facility_id: str,
        milestone_type: MilestoneType,
        title: str,
        description: str,
        impact_score: float = 1.0
    ) -> SuccessMilestone:
        """Create a new milestone"""
        
        milestone = SuccessMilestone(
            id=f"MIL-{facility_id}-{len(self.milestones.get(facility_id, [])) + 1}",
            customer_id=facility_id,
            milestone_type=milestone_type,
            title=title,
            description=description,
            achieved_date=datetime.now(),
            impact_score=impact_score,
            celebration_required=impact_score >= 1.0
        )
        
        if facility_id not in self.milestones:
            self.milestones[facility_id] = []
        
        self.milestones[facility_id].append(milestone)
        
        # Update customer adoption metrics
        if facility_id in self.customers:
            customer = self.customers[facility_id]
            customer.adoption_milestones.append({
                "type": milestone_type.value,
                "title": title,
                "achieved_date": milestone.achieved_date.isoformat(),
                "impact_score": impact_score
            })
        
        return milestone
    
    async def _check_milestone_achievements(self, facility_id: str, current_kpis: HealthcareKPIs) -> None:
        """Check for milestone achievements based on KPIs"""
        
        customer = self.customers[facility_id]
        kpi_history = self.kpi_history[facility_id]
        
        # Check for first clinical use milestone
        if (len(kpi_history) > 1 and 
            current_kpis.user_adoption_rate > 25 and 
            not any(milestone.milestone_type == MilestoneType.FIRST_CLINICAL_USE 
                   for milestone in self.milestones.get(facility_id, []))):
            
            await self._create_milestone(
                facility_id,
                MilestoneType.FIRST_CLINICAL_USE,
                "First Clinical Use Achieved",
                "System successfully used in clinical setting",
                impact_score=2.0
            )
        
        # Check for usage milestone (50% adoption)
        if (current_kpis.user_adoption_rate >= 50 and 
            not any(m.milestone_type == MilestoneType.USAGE_MILESTONE and m.impact_score == 1.0
                   for m in self.milestones.get(facility_id, []))):
            
            await self._create_milestone(
                facility_id,
                MilestoneType.USAGE_MILESTONE,
                "50% User Adoption Reached",
                "Half of potential users are actively using the system",
                impact_score=3.0
            )
        
        # Check for training certification milestone
        if (current_kpis.training_completion_rate >= 80 and 
            not any(m.milestone_type == MilestoneType.TRAINING_CERTIFICATION
                   for m in self.milestones.get(facility_id, []))):
            
            await self._create_milestone(
                facility_id,
                MilestoneType.TRAINING_CERTIFICATION,
                "Training Certification Complete",
                "80% of users have completed training certification",
                impact_score=2.5
            )
    
    async def _check_health_alerts(self, customer: CustomerHealthMetrics) -> None:
        """Check for health alerts and send notifications if needed"""
        
        if customer.health_status == HealthStatus.CRITICAL:
            logger.critical(f"CRITICAL HEALTH ALERT: {customer.facility_name} ({customer.facility_id}) - Health score: {customer.health_score}")
            # In production, this would send alerts to customer success team
        
        elif customer.health_status == HealthStatus.AT_RISK:
            logger.warning(f"HEALTH RISK ALERT: {customer.facility_name} ({customer.facility_id}) - Health score: {customer.health_score}")
            # In production, this would send warnings to customer success team
    
    async def _generate_customer_recommendations(self, customer: CustomerHealthMetrics, trends: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations for customer"""
        
        recommendations = []
        
        # Health score recommendations
        if customer.health_score < 75:
            recommendations.append("Schedule a customer success review to address health score concerns")
        
        # Adoption recommendations
        if "user_adoption_rate" in trends and trends["user_adoption_rate"]["change_percentage"] < 0:
            recommendations.append("Consider additional user training and adoption programs")
        
        # Training recommendations
        if customer.training_completion_rate < 60:
            recommendations.append("Increase training completion rates through additional support and incentives")
        
        # System performance recommendations
        if customer.health_status in [HealthStatus.AT_RISK, HealthStatus.CRITICAL]:
            if "system_uptime_percentage" in trends and trends["system_uptime_percentage"]["current_value"] < 99:
                recommendations.append("Address system uptime and performance issues")
            
            if "average_response_time" in trends and trends["average_response_time"]["current_value"] > 2000:
                recommendations.append("Optimize system response times for better user experience")
        
        # Safety recommendations
        if "patient_safety_score" in trends and trends["patient_safety_score"]["current_value"] < 80:
            recommendations.append("Focus on patient safety score improvement through system optimization")
        
        return recommendations
    
    def _determine_celebration_type(self, milestone: SuccessMilestone) -> str:
        """Determine appropriate celebration type for milestone"""
        
        if milestone.impact_score >= 3.0:
            return "major_celebration"
        elif milestone.impact_score >= 2.0:
            return "recognition"
        else:
            return "acknowledgment"
    
    def _get_celebration_actions(self, milestone: SuccessMilestone) -> List[str]:
        """Get celebration actions based on milestone type"""
        
        actions_map = {
            MilestoneType.ONBOARDING_COMPLETE: ["Send welcome email", "Assign success manager"],
            MilestoneType.FIRST_CLINICAL_USE: ["Send congratulations", "Share case study"],
            MilestoneType.USAGE_MILESTONE: ["Executive recognition", "Success story documentation"],
            MilestoneType.TRAINING_CERTIFICATION: ["Certificate issuance", "Training completion badge"],
            MilestoneType.INTEGRATION_COMPLETE: ["Integration completion celebration", "Technical team recognition"],
            MilestoneType.ROI_ACHIEVED: ["ROI achievement announcement", "Executive briefing"]
        }
        
        return actions_map.get(milestone.milestone_type, ["Standard acknowledgment"])

# Global customer success system instance
success_system = CustomerSuccessSystem()

# Example usage and testing functions
async def setup_sample_customer_success():
    """Set up sample customer success tracking"""
    
    # Register sample customers
    customer1 = await success_system.register_customer(
        facility_id="HOSP001",
        facility_name="General Hospital",
        facility_type="Hospital",
        number_of_users=150
    )
    
    customer2 = await success_system.register_customer(
        facility_id="CARD001", 
        facility_name="Heart Center",
        facility_type="Specialty Clinic",
        number_of_users=75
    )
    
    # Update KPIs with realistic values
    hospital_kpis = HealthcareKPIs(
        patient_safety_score=88.5,
        clinical_workflow_efficiency=82.3,
        regulatory_compliance_rate=96.8,
        user_adoption_rate=67.2,
        system_uptime_percentage=99.7,
        average_response_time=850.0,
        incident_resolution_time=18.5,
        training_completion_rate=78.9
    )
    
    await success_system.update_customer_kpis("HOSP001", hospital_kpis)
    
    # Record ROI metrics
    roi_data = {
        "efficiency_gains": 45000.0,
        "time_savings_hours": 2400.0,
        "error_reduction_percentage": 35.0,
        "patient_outcome_improvements": 15.0
    }
    
    await success_system.record_roi_metrics("HOSP001", 75000.0, roi_data)
    
    # Get health dashboard
    dashboard = await success_system.get_customer_health_dashboard("HOSP001")
    print(f"Customer Health Dashboard for HOSP001:")
    print(f"Health Score: {dashboard['health_status']['overall_score']:.1f}")
    print(f"Status: {dashboard['health_status']['status']}")
    
    # Get at-risk customers
    at_risk = await success_system.get_at_risk_customers()
    print(f"At-risk customers: {len(at_risk)}")

if __name__ == "__main__":
    asyncio.run(setup_sample_customer_success())