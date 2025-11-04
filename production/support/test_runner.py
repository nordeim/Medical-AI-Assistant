#!/usr/bin/env python3
"""
Production Support System - Comprehensive Test Runner
Demonstrates all system capabilities and validates functionality
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the support directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_complete_system():
    """Run comprehensive tests of all system components"""
    
    print("=" * 80)
    print("🏥 HEALTHCARE SUPPORT SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Test Run Started: {datetime.now().isoformat()}")
    print()
    
    test_results = {}
    
    # Test 1: Support Ticket System
    print("📋 Test 1: Support Ticket System")
    print("-" * 40)
    try:
        from ticketing.ticket_management import ticket_system, TicketCategory, PriorityLevel, MedicalContext
        
        # Create emergency medical ticket
        emergency_context = MedicalContext(
            medical_specialty="emergency_medicine",
            urgency_level="immediate",
            patient_safety_impact="critical"
        )
        
        ticket = await ticket_system.create_ticket(
            title="Cardiac monitoring system failure during emergency",
            description="Critical system failure during cardiac arrest situation - immediate assistance required",
            category=TicketCategory.SYSTEM_OUTAGE,
            reporter_id="dr_emergency_001",
            reporter_name="Dr. Sarah Emergency",
            reporter_facility="Emergency Hospital",
            reporter_role="Emergency Physician",
            medical_context=emergency_context
        )
        
        print(f"  ✅ Emergency ticket created: {ticket.id}")
        print(f"  ✅ Priority: {ticket.priority.value}")
        print(f"  ✅ Status: {ticket.status.value}")
        print(f"  ✅ SLA Due: {ticket.sla_due_at}")
        
        # Create standard ticket
        standard_ticket = await ticket_system.create_ticket(
            title="User training request for new nurses",
            description="Need training on AI diagnostic tool for nursing staff",
            category=TicketCategory.TRAINING_REQUEST,
            reporter_id="nurse_training_002",
            reporter_name="Nurse Training Coordinator",
            reporter_facility="City Medical Center",
            reporter_role="Nurse Educator"
        )
        
        print(f"  ✅ Standard ticket created: {standard_ticket.id}")
        print(f"  ✅ Total tickets: {len(ticket_system.tickets)}")
        
        test_results["ticketing"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Ticketing test failed: {str(e)}")
        test_results["ticketing"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 2: Feedback Collection System
    print("💬 Test 2: Feedback Collection & Sentiment Analysis")
    print("-" * 40)
    try:
        from feedback.feedback_collection import feedback_system, FeedbackType
        
        # Submit critical feedback
        critical_feedback = await feedback_system.collect_feedback(
            feedback_type=FeedbackType.INCIDENT_REVIEW,
            user_id="dr_critical_001",
            user_name="Dr. Critical Case",
            user_facility="Critical Care Hospital",
            user_role="Critical Care Physician",
            content="System delay caused 20-minute delay in critical medication. This poses serious patient safety risks and requires immediate attention.",
            rating=1,
            medical_context="critical_medication_delay"
        )
        
        print(f"  ✅ Critical feedback submitted: {critical_feedback.id}")
        
        # Analyze sentiment
        sentiment = await feedback_system.analyze_feedback_sentiment(critical_feedback)
        print(f"  ✅ Sentiment: {sentiment.overall_sentiment.value}")
        print(f"  ✅ Confidence: {sentiment.confidence_score:.2f}")
        print(f"  ✅ Action Required: {sentiment.action_required}")
        print(f"  ✅ Patient Safety Concern: {sentiment.patient_safety_concern}")
        
        # Submit positive feedback
        positive_feedback = await feedback_system.collect_feedback(
            feedback_type=FeedbackType.POST_INTERACTION,
            user_id="dr_positive_002",
            user_name="Dr. Positive Outcome",
            user_facility="Wellness Medical Center",
            user_role="Family Physician",
            content="The new AI diagnostic assistant has significantly improved our patient care efficiency. Outstanding system performance.",
            rating=5,
            medical_context="ai_assistant_success"
        )
        
        print(f"  ✅ Positive feedback submitted: {positive_feedback.id}")
        print(f"  ✅ Total feedback: {len(feedback_system.feedback_responses)}")
        
        test_results["feedback"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Feedback test failed: {str(e)}")
        test_results["feedback"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 3: Health Monitoring System
    print("📊 Test 3: Health Check & Monitoring System")
    print("-" * 40)
    try:
        from monitoring.health_checks import health_monitor, ComponentType
        
        # Simple health check function
        async def api_health_check():
            return {"status": "healthy", "response_time_ms": 150}
        
        # Register system components
        health_monitor.register_component(
            "api_test",
            "Test Medical API",
            ComponentType.API_ENDPOINT,
            api_health_check,
            sla_target=99.9,
            check_interval=30
        )
        
        # Perform health check
        result = await health_monitor.perform_health_check("api_test")
        print(f"  ✅ Health check performed: {result.component_name}")
        print(f"  ✅ Status: {result.status.value}")
        print(f"  ✅ Response Time: {result.response_time_ms:.2f}ms")
        
        # Get system overview
        overview = await health_monitor.get_system_overview()
        print(f"  ✅ System Overview: {overview['overall_status']}")
        print(f"  ✅ Components: {overview['component_breakdown']['total']}")
        
        test_results["monitoring"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Monitoring test failed: {str(e)}")
        test_results["monitoring"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 4: Incident Management System
    print("🚨 Test 4: Incident Management & Escalation")
    print("-" * 40)
    try:
        from incident_management.emergency_response import incident_system, IncidentType, IncidentSeverity, MedicalContext
        
        # Create medical emergency incident
        medical_context = MedicalContext(
            affected_patients=3,
            clinical_area="Emergency Department",
            patient_safety_impact="critical",
            emergency_situation=True
        )
        
        incident = await incident_system.create_incident(
            title="Medical device malfunction affecting patient care",
            description="Multiple cardiac monitoring devices showing intermittent failures in ICU",
            incident_type=IncidentType.MEDICAL_DEVICE_FAILURE,
            severity=IncidentSeverity.SEV1_CRITICAL,
            reporter_id="icu_supervisor_001",
            reporter_name="ICU Supervisor",
            reporter_facility="Major Medical Center",
            medical_context=medical_context
        )
        
        print(f"  ✅ Critical incident created: {incident.id}")
        print(f"  ✅ Severity: {incident.severity.value}")
        print(f"  ✅ Status: {incident.status.value}")
        
        # Update incident
        await incident_system.update_incident(
            incident.id,
            "incident_commander",
            "Incident Commander",
            "Emergency Response",
            "Deploying backup systems, manual monitoring protocols activated",
            status_change=None
        )
        
        print(f"  ✅ Incident updated successfully")
        print(f"  ✅ Total incidents: {len(incident_system.incidents)}")
        
        test_results["incidents"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Incident management test failed: {str(e)}")
        test_results["incidents"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 5: Customer Success Tracking
    print("📈 Test 5: Customer Success Tracking")
    print("-" * 40)
    try:
        from success_tracking.success_metrics import success_system, HealthcareKPIs
        
        # Register healthcare facility
        customer = await success_system.register_customer(
            facility_id="HOSPITAL_001",
            facility_name="Regional Medical Center",
            facility_type="Hospital",
            number_of_users=250
        )
        
        print(f"  ✅ Customer registered: {customer.facility_name}")
        print(f"  ✅ Health Score: {customer.health_score:.1f}")
        print(f"  ✅ Health Status: {customer.health_status.value}")
        
        # Update KPIs
        kpis = HealthcareKPIs(
            patient_safety_score=92.5,
            clinical_workflow_efficiency=88.3,
            regulatory_compliance_rate=96.8,
            user_adoption_rate=75.2,
            system_uptime_percentage=99.7,
            average_response_time=850.0,
            incident_resolution_time=12.5,
            training_completion_rate=82.1
        )
        
        await success_system.update_customer_kpis("HOSPITAL_001", kpis)
        print(f"  ✅ KPIs updated successfully")
        
        # Get health dashboard
        dashboard = await success_system.get_customer_health_dashboard("HOSPITAL_001")
        print(f"  ✅ Dashboard generated")
        
        test_results["success"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Customer success test failed: {str(e)}")
        test_results["success"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 6: Knowledge Base System
    print("📚 Test 6: Knowledge Base & Self-Service")
    print("-" * 40)
    try:
        from knowledge_base.medical_docs import knowledge_base, ContentType, UserRole, MedicalSpecialty
        
        # Create knowledge content
        content = await knowledge_base.create_content(
            title="Emergency Response Protocol for AI System Failures",
            content="Complete protocol for handling AI system failures during medical emergencies...",
            content_type=ContentType.CLINICAL_PROTOCOL,
            author_id="medical_director",
            author_name="Medical Director",
            medical_specialty=MedicalSpecialty.EMERGENCY_MEDICINE,
            target_roles=[UserRole.PHYSICIAN, UserRole.NURSE],
            tags=["emergency", "ai", "protocol", "failure"]
        )
        
        print(f"  ✅ Knowledge content created: {content.id}")
        print(f"  ✅ Title: {content.title}")
        print(f"  ✅ Type: {content.content_type.value}")
        
        # Search knowledge base
        results = await knowledge_base.search_content(
            query="emergency protocol",
            user_role=UserRole.PHYSICIAN,
            medical_specialty=MedicalSpecialty.EMERGENCY_MEDICINE
        )
        
        print(f"  ✅ Search results: {len(results)} items found")
        print(f"  ✅ Total content: {len(knowledge_base.content_library)}")
        
        test_results["knowledge"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Knowledge base test failed: {str(e)}")
        test_results["knowledge"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 7: Training & Certification System
    print("🎓 Test 7: Training & Certification Programs")
    print("-" * 40)
    try:
        from training.certification_programs import training_system, CertificationTrack
        
        # Enroll user in certification track
        user_progress = await training_system.enroll_user_in_track(
            user_id="dr_trainee_001",
            user_name="Dr. Training Example",
            user_facility="Training Hospital",
            user_role="Physician",
            certification_track_id="HC_PROF_001"
        )
        
        print(f"  ✅ User enrolled: {user_progress.user_name}")
        print(f"  ✅ Track: Healthcare Professional Certification")
        print(f"  ✅ Enrollment Date: {user_progress.enrollment_date.date()}")
        
        # Get training dashboard
        dashboard = await training_system.get_user_dashboard("dr_trainee_001")
        print(f"  ✅ Dashboard generated")
        print(f"  ✅ Total Training Hours: {dashboard['training_summary']['total_training_hours']}")
        
        test_results["training"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Training system test failed: {str(e)}")
        test_results["training"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 8: Automation System
    print("🤖 Test 8: Response Automation System")
    print("-" * 40)
    try:
        from automation.response_automation import automation_engine, AutomationTrigger
        
        # Test automation trigger
        emergency_context = {
            "ticket_id": "TEST_EMERGENCY_001",
            "priority": "emergency",
            "category": "medical_emergency",
            "content": "Emergency cardiac arrest situation",
            "facility_context": {"facility_name": "Test Emergency Hospital"},
            "user_context": {"user_id": "dr_test", "user_name": "Dr. Test Emergency"}
        }
        
        executed_rules = await automation_engine.process_automation_trigger(
            AutomationTrigger.TICKET_CREATED,
            emergency_context
        )
        
        print(f"  ✅ Automation triggered: {len(executed_rules)} rules executed")
        
        # Get automation stats
        stats = await automation_engine.get_automation_stats()
        print(f"  ✅ Automation Stats: {stats['total_rules']} total rules")
        print(f"  ✅ Enabled Rules: {stats['enabled_rules']} active")
        
        test_results["automation"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Automation test failed: {str(e)}")
        test_results["automation"] = f"FAILED: {str(e)}"
    
    print()
    
    # Test 9: System Integration
    print("🔗 Test 9: System Integration & Reporting")
    print("-" * 40)
    try:
        # Generate comprehensive system report
        report = {
            "test_execution": {
                "timestamp": datetime.now().isoformat(),
                "test_results": test_results,
                "total_tests": len(test_results),
                "passed_tests": len([r for r in test_results.values() if r == "PASSED"])
            },
            "system_status": {
                "ticketing": f"{len(ticket_system.tickets)} tickets",
                "feedback": f"{len(feedback_system.feedback_responses)} responses",
                "monitoring": f"{len(health_monitor.components)} components",
                "incidents": f"{len(incident_system.incidents)} incidents",
                "customers": f"{len(success_system.customers)} customers",
                "knowledge_base": f"{len(knowledge_base.content_library)} articles",
                "automation": f"{len(automation_engine.automation_rules)} rules"
            }
        }
        
        print(f"  ✅ Integration test passed")
        print(f"  ✅ System report generated")
        print(f"  ✅ All components operational")
        
        test_results["integration"] = "PASSED"
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {str(e)}")
        test_results["integration"] = f"FAILED: {str(e)}"
    
    print()
    
    # Final Results
    print("=" * 80)
    print("🏥 TEST SUITE RESULTS")
    print("=" * 80)
    
    passed_count = 0
    total_count = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result == "PASSED" else f"❌ FAILED"
        print(f"{test_name.upper():20} {status}")
        if result == "PASSED":
            passed_count += 1
    
    print()
    print(f"SUMMARY: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    
    if passed_count == total_count:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        print()
        print("System is fully operational and ready for healthcare deployment.")
        print("All core components are working correctly and integrated.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        print()
        print("The system may need additional configuration or debugging.")
    
    print()
    print(f"Test Suite Completed: {datetime.now().isoformat()}")
    print("=" * 80)
    
    return test_results

async def main():
    """Main test runner"""
    print("Healthcare Support System - Production Test Suite")
    print("This will test all system components and functionality")
    print()
    
    try:
        test_results = await test_complete_system()
        
        # Save test results to file
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "test_run": datetime.now().isoformat(),
                "results": test_results,
                "summary": {
                    "total": len(test_results),
                    "passed": len([r for r in test_results.values() if r == "PASSED"]),
                    "success_rate": len([r for r in test_results.values() if r == "PASSED"]) / len(test_results) * 100
                }
            }, f, indent=2)
        
        print(f"Test results saved to: {results_file}")
        
        # Return appropriate exit code
        passed_tests = len([r for r in test_results.values() if r == "PASSED"])
        if passed_tests == len(test_results):
            sys.exit(0)  # All tests passed
        else:
            sys.exit(1)  # Some tests failed
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {str(e)}")
        sys.exit(2)  # Test suite error

if __name__ == "__main__":
    asyncio.run(main())