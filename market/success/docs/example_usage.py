"""
Customer Success Framework Example
Demonstration of how to use the Enterprise Customer Success Framework for Healthcare AI
"""

import datetime
from framework_coordinator import EnterpriseCustomerSuccessFramework
from config.framework_config import CustomerTier, CustomerSegment
from reviews.annual_business_reviews import ReviewType
from feedback.feedback_loops import FeedbackType
from retention.retention_strategies import RetentionStrategyType

def demonstrate_framework_usage():
    """Demonstrate comprehensive usage of the customer success framework"""
    
    print("🏥 Healthcare AI Customer Success Framework Demo")
    print("=" * 60)
    
    # Initialize the framework
    framework = EnterpriseCustomerSuccessFramework()
    
    # Step 1: Customer Onboarding
    print("\n1. CUSTOMER ONBOARDING")
    print("-" * 30)
    
    customer_data = {
        "customer_id": "METRO_HEALTH_001",
        "organization_name": "Metro Health System",
        "tier": "enterprise",
        "segment": {
            "segment_name": "Large Health System",
            "organization_type": "health_system",
            "size_category": "enterprise",
            "clinical_specialty": "multi_specialty",
            "geographic_region": "midwest",
            "maturity_level": "mature",
            "tech_adoption": "progressive"
        },
        "primary_contact": "Dr. Sarah Johnson",
        "email": "sarah.johnson@metrohealth.org",
        "csm_assigned": "mike.smith",
        "contract_start": "2024-01-15",
        "contract_value": 750000,
        "team_members": [
            {
                "name": "Dr. Sarah Johnson",
                "title": "Chief Medical Officer",
                "specialty": "internal_medicine",
                "expertise_areas": ["clinical_analytics", "quality_improvement"]
            },
            {
                "name": "Mark Thompson",
                "title": "VP of Operations",
                "specialty": "operations",
                "expertise_areas": ["workflow_optimization", "process_improvement"]
            }
        ]
    }
    
    framework.initialize_customer_onboarding(customer_data)
    print(f"✅ Onboarded customer: {customer_data['organization_name']}")
    
    # Step 2: Health Score Monitoring
    print("\n2. HEALTH SCORE MONITORING")
    print("-" * 30)
    
    # Update health metrics
    health_metrics = {
        "usage_percentage": 87.5,
        "clinical_outcome_improvement": 22.3,
        "user_adoption_rate": 0.84,
        "support_ticket_volume": 2,
        "nps_score": 8.7,
        "engagement_level": 0.81
    }
    
    health_results = framework.process_customer_health_update("METRO_HEALTH_001", health_metrics)
    print(f"✅ Health Score: {health_results.get('current_health_score', 'N/A'):.1f}/100")
    print(f"✅ Alerts Triggered: {health_results.get('alerts_triggered', 0)}")
    
    # Step 3: Retention Risk Assessment
    print("\n3. RETENTION RISK ASSESSMENT")
    print("-" * 30)
    
    retention_data = {
        "clinical_outcome_trend": "improving",
        "roi_concerns": False,
        "workflow_adoption": 85.0,
        "competitive_pressure": 0.25,
        "staff_satisfaction": 87.0,
        "error_rate": 1.2,
        "quality_metrics": 94.0
    }
    
    retention_prediction = framework.retention_manager.assess_retention_risk(
        "METRO_HEALTH_001", retention_data
    )
    
    print(f"✅ Churn Risk: {retention_prediction.churn_probability:.1%}")
    print(f"✅ Risk Level: {'Low' if retention_prediction.churn_probability < 0.3 else 'Medium' if retention_prediction.churn_probability < 0.6 else 'High'}")
    print(f"✅ Top Risk Factors: {len(retention_prediction.risk_factors)} identified")
    
    # Step 4: Expansion Opportunity Identification
    print("\n4. EXPANSION OPPORTUNITIES")
    print("-" * 30)
    
    expansion_data = framework.identify_expansion_opportunities("METRO_HEALTH_001")
    
    print(f"✅ Opportunities Identified: {expansion_data.get('opportunities_identified', 0)}")
    print(f"✅ Pipeline Value: ${expansion_data.get('total_pipeline_value', 0):,.0f}")
    print(f"✅ Weighted Value: ${expansion_data.get('weighted_pipeline_value', 0):,.0f}")
    
    # Step 5: Customer Feedback Processing
    print("\n5. CUSTOMER FEEDBACK PROCESSING")
    print("-" * 30)
    
    feedback_data = {
        "type": "feature_request",
        "title": "Enhanced Clinical Decision Support",
        "description": "Would like to see more predictive analytics for treatment recommendations",
        "submitted_by": "Dr. Sarah Johnson",
        "priority": "medium"
    }
    
    feedback_results = framework.process_customer_feedback("METRO_HEALTH_001", feedback_data)
    print(f"✅ Feedback Submitted: {feedback_results.get('feedback_submitted', False)}")
    print(f"✅ Enhancement Request Created: {feedback_results.get('enhancement_request_created', False)}")
    
    # Step 6: Community Engagement
    print("\n6. COMMUNITY ENGAGEMENT")
    print("-" * 30)
    
    # Add team members to community
    community_member = framework.community_manager.join_community(
        customer_id="METRO_HEALTH_001",
        name="Dr. Sarah Johnson",
        title="Chief Medical Officer",
        organization="Metro Health System",
        specialty="internal_medicine",
        expertise_areas=["clinical_analytics", "quality_improvement"]
    )
    
    print(f"✅ Community Member Added: {community_member.name}")
    print(f"✅ Specialty: {community_member.specialty}")
    
    # Create best practice content
    best_practice = framework.community_manager.create_content_post(
        author_id=community_member.member_id,
        title="AI-Driven Quality Improvement Results",
        content_type="best_practice",
        description="How we improved quality metrics by 22% using AI insights",
        content_body="Our organization implemented AI-driven quality improvement processes...",
        tags=["quality", "ai", "improvement"],
        clinical_specialty="quality_safety"
    )
    
    print(f"✅ Best Practice Shared: {best_practice.title}")
    
    # Step 7: Annual Business Review
    print("\n7. ANNUAL BUSINESS REVIEW")
    print("-" * 30)
    
    # Prepare comprehensive business review
    review_period_data = {
        "contract_value": 750000,
        "actual_spend": 680000,
        "projected_savings": 185000,
        "clinical_outcome_improvement": 22.3,
        "efficiency_improvement": 28.5,
        "user_adoption_rate": 0.84,
        "support_ticket_volume": 2,
        "time_saved_hours": 240,
        "errors_prevented": 18,
        "patient_volume": 12500,
        "compliance_score_improvement": 7.5,
        "period_description": "2024 Annual Period"
    }
    
    review = framework.review_manager.prepare_business_review_report(
        "METRO_HEALTH_001",
        ReviewType.ANNUAL_BUSINESS_REVIEW,
        review_period_data
    )
    
    print(f"✅ Business Review Prepared")
    print(f"✅ Clinical ROI Demonstrated: ${sum(analysis.financial_impact for analysis in review.clinical_roi_analysis):,.0f}")
    print(f"✅ Strategic Objectives: {len(review.strategic_objectives)}")
    
    # Step 8: Comprehensive Health Report
    print("\n8. COMPREHENSIVE HEALTH REPORT")
    print("-" * 30)
    
    health_report = framework.generate_customer_health_report("METRO_HEALTH_001")
    
    print(f"✅ Executive Summary Generated")
    print(f"✅ Current Health Score: {health_report['health_score']['current_score']:.1f}/100")
    print(f"✅ Retention Risk: {health_report['retention_analysis']['churn_risk']:.1%}")
    print(f"✅ Expansion Potential: {health_report['expansion_potential'].get('total_pipeline_value', 0):,.0f}")
    
    # Step 9: Framework Dashboard
    print("\n9. FRAMEWORK DASHBOARD")
    print("-" * 30)
    
    dashboard = framework.generate_framework_dashboard()
    
    print(f"✅ Framework Overview:")
    print(f"   - Total Customers: {dashboard['framework_overview']['total_customers']}")
    print(f"   - Healthy Customers: {dashboard['framework_overview']['healthy_customers']}")
    print(f"   - At-Risk Customers: {dashboard['framework_overview']['at_risk_customers']}")
    print(f"✅ Expansion Pipeline: ${dashboard['expansion_pipeline']['total_pipeline_value']:,.0f}")
    print(f"✅ Community Members: {dashboard['community_engagement']['active_members']}")
    print(f"✅ Active Workflows: {dashboard['automated_workflows']['active_workflows']}")
    
    # Step 10: Data Export
    print("\n10. DATA EXPORT")
    print("-" * 30)
    
    export_data = framework.export_customer_data("METRO_HEALTH_001")
    
    print(f"✅ Customer Data Exported")
    print(f"✅ Components Included: {len(export_data['components'])}")
    print(f"✅ Export Date: {export_data['export_date'].strftime('%Y-%m-%d %H:%M')}")
    
    print("\n" + "=" * 60)
    print("🎉 Framework Demo Complete!")
    print("All customer success components are operational and integrated.")
    
    return framework

def run_automated_health_checks():
    """Run automated health checks across all customers"""
    print("\n🔄 RUNNING AUTOMATED HEALTH CHECKS")
    print("-" * 40)
    
    framework = demonstrate_framework_usage()
    
    # Simulate health check run
    health_check_results = framework.run_automated_health_checks()
    
    print(f"✅ Customers Checked: {health_check_results['customers_checked']}")
    print(f"✅ Alerts Generated: {health_check_results['alerts_generated']}")
    print(f"✅ Interventions Triggered: {health_check_results['interventions_triggered']}")
    print(f"✅ Expansion Opportunities: {health_check_results['expansion_opportunities_identified']}")
    print(f"✅ Customers Needing Attention: {len(health_check_results['customers_needing_attention'])}")
    
    return health_check_results

def demonstrate_retention_campaign():
    """Demonstrate retention campaign creation and execution"""
    print("\n🎯 RETENTION CAMPAIGN DEMO")
    print("-" * 30)
    
    framework = EnterpriseCustomerSuccessFramework()
    
    # First, onboard a customer at risk
    at_risk_customer_data = {
        "customer_id": "COMMUNITY_HOSP_002",
        "organization_name": "Community Hospital",
        "tier": "premium",
        "segment": {
            "segment_name": "Mid-size Hospital",
            "organization_type": "hospital",
            "size_category": "medium",
            "clinical_specialty": "general_medicine",
            "geographic_region": "south",
            "maturity_level": "growing",
            "tech_adoption": "moderate"
        },
        "primary_contact": "Dr. Robert Davis",
        "email": "robert.davis@communityhosp.org",
        "csm_assigned": "lisa.wilson",
        "contract_start": "2023-06-01",
        "contract_value": 250000,
        "team_members": [
            {
                "name": "Dr. Robert Davis",
                "title": "Medical Director",
                "specialty": "emergency_medicine",
                "expertise_areas": ["emergency_care", "patient_flow"]
            }
        ]
    }
    
    framework.initialize_customer_onboarding(at_risk_customer_data)
    
    # Assess high risk scenario
    high_risk_data = {
        "clinical_outcome_trend": "declining",
        "roi_concerns": True,
        "workflow_adoption": 45.0,
        "competitive_pressure": 0.8,
        "staff_satisfaction": 65.0,
        "error_rate": 4.8,
        "quality_metrics": 72.0
    }
    
    retention_prediction = framework.retention_manager.assess_retention_risk(
        "COMMUNITY_HOSP_002", high_risk_data
    )
    
    print(f"🚨 High Risk Customer Identified")
    print(f"   Churn Risk: {retention_prediction.churn_probability:.1%}")
    print(f"   Recommended Actions: {len(retention_prediction.recommended_actions)}")
    
    # Create retention strategy
    retention_strategy = framework.retention_manager.create_retention_strategy(
        customer_id="COMMUNITY_HOSP_002",
        strategy_type=RetentionStrategyType.CLINICAL_OUTCOMES,
        risk_level="high"
    )
    
    print(f"✅ Retention Strategy Created: {retention_strategy.strategy_id}")
    print(f"   Type: {retention_strategy.strategy_type.value}")
    print(f"   Expected Impact: {retention_strategy.expected_impact:.1%}")
    
    # Launch retention campaign
    campaign = framework.retention_manager.launch_retention_campaign(
        customer_ids=["COMMUNITY_HOSP_002"],
        campaign_type="clinical_outcomes_recovery",
        objectives=[
            "Improve clinical outcome metrics",
            "Increase staff satisfaction",
            "Demonstrate ROI through data",
            "Restore confidence in solution"
        ]
    )
    
    print(f"✅ Retention Campaign Launched: {campaign.campaign_id}")
    print(f"   Budget: ${campaign.budget:,}")
    print(f"   Duration: {campaign.end_date - campaign.start_date} days")
    
    return framework

def demonstrate_expansion_strategy():
    """Demonstrate expansion strategy for high-performing customer"""
    print("\n📈 EXPANSION STRATEGY DEMO")
    print("-" * 30)
    
    framework = EnterpriseCustomerSuccessFramework()
    
    # High-performing customer ready for expansion
    expansion_ready_customer = {
        "customer_id": "UNIVERSITY_MEDICAL_003",
        "organization_name": "University Medical Center",
        "tier": "enterprise",
        "segment": {
            "segment_name": "Academic Medical Center",
            "organization_type": "academic_medical_center",
            "size_category": "enterprise",
            "clinical_specialty": "multi_specialty_research",
            "geographic_region": "northeast",
            "maturity_level": "innovative",
            "tech_adoption": "innovative"
        },
        "primary_contact": "Dr. Amanda Foster",
        "email": "amanda.foster@universitymed.org",
        "csm_assigned": "david.chang",
        "contract_start": "2023-03-01",
        "contract_value": 1200000,
        "team_members": [
            {
                "name": "Dr. Amanda Foster",
                "title": "Chief Innovation Officer",
                "specialty": "research_innovation",
                "expertise_areas": ["ai_research", "clinical_trials", "innovation"]
            }
        ]
    }
    
    framework.initialize_customer_onboarding(expansion_ready_customer)
    
    # High usage scenario indicating expansion readiness
    high_usage_data = {
        "annual_contract_value": 1200000,
        "usage_metrics": {
            "overall_usage": 92.5,
            "feature_adoption": 88.0,
            "clinical_outcome_improvement": 31.2,
            "workflow_optimization": 85.0
        },
        "organization_size": "enterprise",
        "customer_satisfaction": 9.2,
        "ai_performance_score": 91.0,
        "ai_maturity_level": 5,
        "has_analytics_basic": True,
        "analytics_usage": 89.0,
        "workflow_count": 8,
        "automation_usage": 87.0,
        "current_integrations": 12,
        "current_ai_modules": ["diagnosis_assistance", "treatment_optimization"]
    }
    
    # Identify expansion opportunities
    opportunities = framework.expansion_manager.identify_expansion_opportunities(
        "UNIVERSITY_MEDICAL_003", high_usage_data
    )
    
    print(f"🎯 Expansion Opportunities Identified: {len(opportunities)}")
    
    for i, opp in enumerate(opportunities[:3], 1):  # Show top 3
        print(f"   {i}. {opp.product_or_service}")
        print(f"      Type: {opp.expansion_type.value}")
        print(f"      Value: ${opp.potential_additional_value:,.0f}")
        print(f"      Probability: {opp.probability:.1%}")
        print(f"      Timeline: {opp.implementation_timeline}")
    
    # Calculate pipeline value
    pipeline_value = framework.expansion_manager.calculate_expansion_pipeline_value("UNIVERSITY_MEDICAL_003")
    
    print(f"\n📊 Expansion Pipeline Summary:")
    print(f"   Total Pipeline: ${pipeline_value['total_pipeline_value']:,.0f}")
    print(f"   Weighted Pipeline: ${pipeline_value['weighted_pipeline_value']:,.0f}")
    print(f"   Average Deal Size: ${pipeline_value['average_deal_size']:,.0f}")
    
    # Launch expansion campaign
    expansion_campaign = framework.expansion_manager.launch_expansion_campaign({
        "name": "Advanced AI Module Expansion",
        "target_segment": "enterprise_innovative",
        "expansion_types": ["module_expansion", "add_on"],
        "triggers": ["high_usage", "outcome_improvement"],
        "target_customers": ["UNIVERSITY_MEDICAL_003"],
        "goals": {
            "revenue_target": 1000000,
            "win_rate_target": 0.70
        },
        "budget": 75000
    })
    
    print(f"✅ Expansion Campaign Launched: {expansion_campaign.name}")
    
    return framework

if __name__ == "__main__":
    print("Healthcare AI Customer Success Framework")
    print("Comprehensive Demo and Examples")
    print("=" * 60)
    
    # Run main demonstration
    framework = demonstrate_framework_usage()
    
    # Run specialized demos
    print("\n" + "=" * 60)
    run_automated_health_checks()
    
    print("\n" + "=" * 60)
    demonstrate_retention_campaign()
    
    print("\n" + "=" * 60)
    demonstrate_expansion_strategy()
    
    print("\n" + "=" * 60)
    print("🏆 ALL DEMONSTRATIONS COMPLETE")
    print("Enterprise Customer Success Framework is fully operational!")