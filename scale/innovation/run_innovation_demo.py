#!/usr/bin/env python3
"""
Enterprise Innovation Framework Demo
Complete demonstration of continuous innovation and product development system
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os

# Add the innovation directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from innovation_framework_orchestrator import InnovationFrameworkOrchestrator, InnovationType, InnovationStatus
from ai_powered_feature_development import AIFeatureDevelopment
from customer_feedback_system import CustomerFeedbackSystem, FeedbackType
from rapid_prototyping_engine import RapidPrototypingEngine
from competitive_analysis_engine import CompetitiveAnalysisEngine
from roadmap_optimizer import RoadmapOptimizer, Priority, Status
from innovation_labs import InnovationLabs, ExperimentType, ResearchArea

class InnovationFrameworkDemo:
    def __init__(self):
        self.logger = self._setup_logging()
        self.orchestrator = None
        self.demo_running = False
    
    def _setup_logging(self) -> logging.Logger:
        """Setup demo logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/innovation_demo.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    async def run_complete_demo(self):
        """Run the complete innovation framework demonstration"""
        self.logger.info("🚀 Starting Enterprise Innovation Framework Demo")
        
        try:
            # Initialize the framework
            await self._initialize_framework()
            
            # Demo Phase 1: Innovation Idea Creation and AI Analysis
            await self._demo_phase_1_innovation_creation()
            
            # Demo Phase 2: AI-Powered Feature Development
            await self._demo_phase_2_ai_feature_development()
            
            # Demo Phase 3: Customer Feedback System
            await self._demo_phase_3_customer_feedback()
            
            # Demo Phase 4: Rapid Prototyping
            await self._demo_phase_4_rapid_prototyping()
            
            # Demo Phase 5: Competitive Analysis
            await self._demo_phase_5_competitive_analysis()
            
            # Demo Phase 6: Roadmap Optimization
            await self._demo_phase_6_roadmap_optimization()
            
            # Demo Phase 7: Innovation Labs
            await self._demo_phase_7_innovation_labs()
            
            # Demo Phase 8: Comprehensive Dashboard
            await self._demo_phase_8_dashboard_integration()
            
            # Demo Phase 9: Continuous Innovation Cycle
            await self._demo_phase_9_continuous_innovation()
            
            self.logger.info("✅ Innovation Framework Demo Completed Successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {str(e)}")
            raise
    
    async def _initialize_framework(self):
        """Initialize all framework components"""
        self.logger.info("🔧 Initializing Innovation Framework Components...")
        
        # Initialize main orchestrator
        self.orchestrator = InnovationFrameworkOrchestrator()
        
        self.logger.info("✅ Framework initialized successfully")
    
    async def _demo_phase_1_innovation_creation(self):
        """Demonstrate innovation idea creation and analysis"""
        self.logger.info("\n📋 PHASE 1: Innovation Idea Creation and Analysis")
        
        # Create multiple innovation ideas
        innovation_ideas = [
            {
                'title': 'AI-Powered Code Review Assistant',
                'description': 'An AI assistant that automatically reviews code for bugs, security issues, and best practices, providing actionable feedback to developers in real-time.',
                'type': 'breakthrough',
                'team_members': ['Alice Smith', 'Bob Johnson', 'Carol Davis'],
                'resource_requirements': {
                    'ai_engineers': 3,
                    'devops': 1,
                    'ui_designer': 1
                }
            },
            {
                'title': 'Real-time Collaboration Platform',
                'description': 'A next-generation real-time collaboration platform with AI-powered features like automated meeting summaries, sentiment analysis, and smart task extraction.',
                'type': 'disruptive',
                'team_members': ['David Wilson', 'Emma Brown'],
                'resource_requirements': {
                    'full_stack_developers': 4,
                    'ai_specialist': 2,
                    'product_manager': 1
                }
            },
            {
                'title': 'Predictive Customer Support System',
                'description': 'An AI system that predicts customer issues before they occur and automatically initiates proactive support, reducing support tickets by 40%.',
                'type': 'incremental',
                'team_members': ['Frank Miller', 'Grace Lee'],
                'resource_requirements': {
                    'data_scientists': 2,
                    'backend_developers': 2,
                    'frontend_developer': 1
                }
            }
        ]
        
        # Create innovation ideas
        idea_ids = []
        for idea_data in innovation_ideas:
            idea_id = await self.orchestrator.create_innovation_idea(idea_data)
            idea_ids.append(idea_id)
            self.logger.info(f"✅ Created innovation idea: {idea_data['title']} ({idea_id[:8]}...)")
        
        # Advance some ideas through status pipeline
        await self.orchestrator.advance_idea_status(idea_ids[0], InnovationStatus.RESEARCH)
        await self.orchestrator.advance_idea_status(idea_ids[1], InnovationStatus.DEVELOPMENT)
        
        self.logger.info(f"📊 Created {len(idea_ids)} innovation ideas and advanced 2 through pipeline")
    
    async def _demo_phase_2_ai_feature_development(self):
        """Demonstrate AI-powered feature development"""
        self.logger.info("\n🤖 PHASE 2: AI-Powered Feature Development")
        
        # Create AI feature development instance
        ai_feature_dev = AIFeatureDevelopment()
        
        # Analyze one of the ideas for AI development opportunities
        analysis_results = await ai_feature_dev.analyze_idea("demo_idea_1")
        
        self.logger.info("🔍 AI Analysis Results:")
        self.logger.info(f"  - Technical Feasibility: {analysis_results['technical_feasibility']['overall_score']:.2f}")
        self.logger.info(f"  - Complexity: {analysis_results['technical_feasibility']['technical_complexity']}/10")
        self.logger.info(f"  - Development Effort: {analysis_results['development_estimate']['total_hours']} hours")
        self.logger.info(f"  - AI Acceleration Potential: {analysis_results['development_estimate']['ai_acceleration_potential']:.1%}")
        
        # Generate AI-powered code suggestions
        suggestions = await ai_feature_dev.generate_suggestions("demo_idea_1")
        self.logger.info(f"🔧 Generated {len(suggestions)} AI-powered code suggestions")
        
        for suggestion in suggestions[:3]:  # Show first 3 suggestions
            self.logger.info(f"  - {suggestion.function_name}: {suggestion.confidence_score:.1%} confidence")
        
        # Setup development environment
        dev_env = await ai_feature_dev.setup_development_environment("demo_idea_1")
        self.logger.info("⚙️ Setup AI-powered development environment")
    
    async def _demo_phase_3_customer_feedback(self):
        """Demonstrate customer feedback system"""
        self.logger.info("\n💬 PHASE 3: Customer Feedback System")
        
        # Initialize feedback system
        feedback_system = CustomerFeedbackSystem()
        
        # Simulate collecting feedback from various sources
        feedback_samples = [
            {
                'source': 'surveys',
                'data': {
                    'content': 'The new AI assistant is amazing! It caught 3 bugs I would have missed.',
                    'type': 'compliment',
                    'user_id': 'user_123',
                    'satisfaction_score': 0.95,
                    'tags': ['ai_assistant', 'code_review', 'productivity']
                }
            },
            {
                'source': 'support_tickets',
                'data': {
                    'content': 'The collaboration platform is slow when handling large video calls.',
                    'type': 'performance_issue',
                    'user_id': 'user_456',
                    'satisfaction_score': 0.4,
                    'tags': ['performance', 'video_calls', 'scalability']
                }
            },
            {
                'source': 'app_reviews',
                'data': {
                    'content': 'Great app but needs better mobile interface design.',
                    'type': 'usability_issue',
                    'platform': 'ios',
                    'rating': 4,
                    'satisfaction_score': 0.75
                }
            }
        ]
        
        # Collect feedback
        feedback_ids = []
        for sample in feedback_samples:
            feedback_id = await feedback_system.collect_feedback(sample['source'], sample['data'])
            feedback_ids.append(feedback_id)
        
        self.logger.info(f"📝 Collected {len(feedback_ids)} feedback items from multiple sources")
        
        # Generate insights
        insights = await feedback_system.generate_insights()
        self.logger.info("📊 Feedback Insights:")
        self.logger.info(f"  - Total Feedback: {insights['overview']['total_feedback']}")
        self.logger.info(f"  - Average Impact Score: {insights['overview']['average_impact_score']:.2f}")
        self.logger.info(f"  - Resolution Rate: {insights['overview']['resolution_rate']:.1%}")
        
        # Start monitoring
        await feedback_system.start_monitoring()
        self.logger.info("🔄 Started real-time feedback monitoring")
    
    async def _demo_phase_4_rapid_prototyping(self):
        """Demonstrate rapid prototyping engine"""
        self.logger.info("\n⚡ PHASE 4: Rapid Prototyping Engine")
        
        # Initialize prototyping engine
        proto_engine = RapidPrototypingEngine()
        
        # Create prototype for AI Code Review Assistant
        prototype_id = await proto_engine.create_prototype("demo_idea_1")
        self.logger.info(f"🛠️ Created prototype: {prototype_id[:8]}...")
        
        # Create iterations
        iteration_changes = [
            "add: code_analysis_engine",
            "add: real_time_suggestions",
            "fix: performance_optimization"
        ]
        
        iteration_id = await proto_engine.iterate_prototype(prototype_id, iteration_changes)
        self.logger.info(f"🔄 Created iteration: {iteration_id[:8]}...")
        
        # Deploy to testing
        testing_deployment = await proto_engine.deploy_to_testing(prototype_id)
        self.logger.info(f"🧪 Deployed to testing: {testing_deployment['deployment_id'][:8]}...")
        
        # Deploy to production
        production_deployment = await proto_engine.deploy_to_production(prototype_id)
        self.logger.info(f"🚀 Deployed to production: {production_deployment['deployment_id'][:8]}...")
        
        # Setup A/B testing
        ab_test = await proto_engine.setup_ab_testing(prototype_id)
        self.logger.info(f"📊 Setup A/B testing: {ab_test.name}")
    
    async def _demo_phase_5_competitive_analysis(self):
        """Demonstrate competitive analysis engine"""
        self.logger.info("\n🔍 PHASE 5: Competitive Analysis")
        
        # Initialize competitive analysis engine
        comp_engine = CompetitiveAnalysisEngine()
        
        # Add competitors
        competitors = [
            {
                'name': 'CodeClimate',
                'domain': 'codeclimate.com',
                'type': 'direct',
                'market_share': 0.25,
                'strengths': ['Established brand', 'Large customer base', 'Comprehensive features'],
                'weaknesses': ['Complex setup', 'High cost', 'Limited AI features'],
                'key_features': ['code_analysis', 'maintainability_index', 'test_coverage'],
                'pricing_model': 'tiered_saas',
                'target_market': 'Enterprise development teams'
            },
            {
                'name': 'SonarQube',
                'domain': 'sonarqube.org',
                'type': 'direct',
                'market_share': 0.30,
                'strengths': ['Open source', 'Large community', 'Extensive language support'],
                'weaknesses': ['Technical setup complexity', 'Resource intensive'],
                'key_features': ['static_analysis', 'security_hotspots', 'technical_debt'],
                'pricing_model': 'enterprise_license',
                'target_market': 'Large enterprises and open source projects'
            },
            {
                'name': 'GitHub Copilot',
                'domain': 'github.com/copilot',
                'type': 'aspirational',
                'market_share': 0.15,
                'strengths': ['AI-powered', 'GitHub integration', 'Developer adoption'],
                'weaknesses': ['Limited to GitHub', 'Code completion only', 'Privacy concerns'],
                'key_features': ['ai_code_completion', 'context_aware', 'multiple_languages'],
                'pricing_model': 'subscription',
                'target_market': 'Individual developers and small teams'
            }
        ]
        
        # Add competitors
        competitor_ids = []
        for comp_data in competitors:
            comp_id = await comp_engine.add_competitor(comp_data)
            competitor_ids.append(comp_id)
        
        self.logger.info(f"🏢 Added {len(competitor_ids)} competitors for analysis")
        
        # Perform comprehensive analysis
        analysis_results = await comp_engine.comprehensive_analysis("demo_idea_1")
        
        self.logger.info("📈 Competitive Analysis Results:")
        self.logger.info(f"  - Competitors Analyzed: {analysis_results['competitor_landscape']['total_competitors_analyzed']}")
        self.logger.info(f"  - Market Share Covered: {analysis_results['competitor_landscape']['total_market_share_covered']:.1%}")
        self.logger.info(f"  - High Threat Competitors: {len(analysis_results['threat_assessment']['high_threat_competitors'])}")
        self.logger.info(f"  - Priority Opportunities: {len(analysis_results['opportunity_matrix'])}")
        
        # Get competitive dashboard
        dashboard = await comp_engine.get_competitive_dashboard()
        self.logger.info("📊 Generated competitive intelligence dashboard")
    
    async def _demo_phase_6_roadmap_optimization(self):
        """Demonstrate roadmap optimization"""
        self.logger.info("\n🗺️ PHASE 6: Roadmap Optimization")
        
        # Initialize roadmap optimizer
        roadmap_optimizer = RoadmapOptimizer()
        
        # Create roadmap items
        roadmap_items = [
            {
                'title': 'AI Code Review Assistant - MVP',
                'description': 'Launch minimum viable product with basic code analysis',
                'priority': 'high',
                'start_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d'),
                'estimated_effort': 40,
                'assigned_team': ['Alice Smith', 'Bob Johnson'],
                'business_value': 0.9,
                'technical_risk': 0.3,
                'market_timing': 0.8,
                'customer_impact': 0.85
            },
            {
                'title': 'Real-time Collaboration Platform',
                'description': 'Develop core collaboration features with AI integration',
                'priority': 'high',
                'start_date': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d'),
                'estimated_effort': 60,
                'assigned_team': ['David Wilson', 'Emma Brown'],
                'business_value': 0.8,
                'technical_risk': 0.5,
                'market_timing': 0.7,
                'customer_impact': 0.9
            },
            {
                'title': 'Mobile App Development',
                'description': 'Create native mobile applications for iOS and Android',
                'priority': 'medium',
                'start_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
                'target_date': (datetime.now() + timedelta(days=120)).strftime('%Y-%m-%d'),
                'estimated_effort': 50,
                'assigned_team': ['Frank Miller', 'Grace Lee'],
                'business_value': 0.7,
                'technical_risk': 0.4,
                'market_timing': 0.6,
                'customer_impact': 0.8
            }
        ]
        
        # Create roadmap items
        item_ids = []
        for item_data in roadmap_items:
            item_id = await roadmap_optimizer.create_roadmap_item(item_data)
            item_ids.append(item_id)
        
        self.logger.info(f"📅 Created {len(item_ids)} roadmap items")
        
        # Optimize roadmap
        optimization_results = await roadmap_optimizer.optimize_roadmap()
        self.logger.info("🎯 Optimized roadmap using AI algorithms")
        
        # Generate dashboard
        dashboard = await roadmap_optimizer.generate_roadmap_dashboard()
        self.logger.info("📊 Generated roadmap dashboard:")
        self.logger.info(f"  - Total Items: {dashboard['overview']['total_items']}")
        self.logger.info(f"  - Overall Progress: {dashboard['overview']['overall_progress']:.1f}%")
        self.logger.info(f"  - Average Alignment Score: {dashboard['overview']['average_alignment_score']:.2f}")
    
    async def _demo_phase_7_innovation_labs(self):
        """Demonstrate innovation labs"""
        self.logger.info("\n🧪 PHASE 7: Innovation Labs")
        
        # Initialize innovation labs
        labs = InnovationLabs()
        
        # Create experiments
        experiments = [
            {
                'title': 'Advanced Code Analysis Using Transformer Models',
                'description': 'Research and implement state-of-the-art transformer models for more accurate code analysis and bug detection.',
                'type': 'ai_research',
                'research_area': 'artificial_intelligence',
                'priority': 'research',
                'hypothesis': 'Transformer models can improve code analysis accuracy by 25% compared to traditional methods',
                'success_criteria': ['25% improvement in accuracy', 'Real-time processing', 'Cross-language support'],
                'start_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
                'estimated_duration': 60,
                'budget_allocated': 75000,
                'team_lead': 'Dr. Sarah Chen',
                'team_members': ['Alex Kumar', 'Maria Rodriguez'],
                'resources_required': {'gpu_hours': 1000, 'data_scientists': 3},
                'external_collaborations': ['Stanford AI Lab', 'MIT CSAIL']
            },
            {
                'title': 'Voice-Controlled Development Environment',
                'description': 'Experiment with voice-controlled programming interfaces for hands-free development.',
                'type': 'technology_exploration',
                'research_area': 'mobile_technology',
                'priority': 'exploration',
                'hypothesis': 'Voice control can increase developer productivity by 15% during coding sessions',
                'success_criteria': ['Voice accuracy > 95%', 'Natural language processing', 'Multi-language support'],
                'start_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                'estimated_duration': 45,
                'budget_allocated': 40000,
                'team_lead': 'James Thompson',
                'team_members': ['Lisa Wang'],
                'resources_required': {'audio_engineers': 2, 'ml_engineers': 1},
                'external_collaborations': ['Google Voice AI Team']
            }
        ]
        
        # Create experiments
        experiment_ids = []
        for exp_data in experiments:
            exp_id = await labs.create_experiment(exp_data)
            experiment_ids.append(exp_id)
            self.logger.info(f"🧪 Created experiment: {exp_data['title']}")
        
        # Start first experiment
        start_result = await labs.start_experiment(experiment_ids[0])
        self.logger.info(f"🚀 Started experiment: {experiment_ids[0][:8]}...")
        
        # Create research project
        project_data = {
            'title': 'Next-Generation AI Development Tools',
            'description': 'Comprehensive research into AI-powered development tools and their impact on software engineering productivity.',
            'research_area': 'artificial_intelligence',
            'objective': 'Develop and validate AI-powered tools that increase developer productivity by 30%',
            'methodology': 'Mixed methods: quantitative metrics, user studies, A/B testing',
            'timeline_months': 18,
            'budget': 250000,
            'team_size': 8,
            'deliverables': ['Research papers', 'Prototypes', 'Industry reports', 'Patent applications'],
            'industry_partnerships': ['Microsoft Research', 'Google AI', 'Amazon Lab126']
        }
        
        project_id = await labs.create_research_project(project_data)
        self.logger.info(f"🔬 Created research project: {project_data['title']}")
        
        # Generate labs dashboard
        dashboard = await labs.generate_lab_dashboard()
        self.logger.info("📊 Innovation Labs Dashboard:")
        self.logger.info(f"  - Total Experiments: {dashboard['overview']['total_experiments']}")
        self.logger.info(f"  - Active Experiments: {dashboard['overview']['active_experiments']}")
        self.logger.info(f"  - Success Rate: {dashboard['overview']['success_rate']:.1%}")
        self.logger.info(f"  - Average Innovation Score: {dashboard['innovation_metrics']['average_innovation_score']:.2f}")
    
    async def _demo_phase_8_dashboard_integration(self):
        """Demonstrate integrated dashboard"""
        self.logger.info("\n📊 PHASE 8: Integrated Dashboard")
        
        # Generate comprehensive innovation dashboard
        dashboard = await self.orchestrator.get_innovation_dashboard()
        
        self.logger.info("🎛️ Innovation Framework Dashboard Overview:")
        self.logger.info(f"  - Active Ideas: {dashboard['overview']['total_active_ideas']}")
        self.logger.info(f"  - Performance Metrics: {len(dashboard['performance_metrics'])} tracked")
        self.logger.info(f"  - AI Insights: {len(dashboard['ai_insights'])} generated")
        self.logger.info(f"  - Recommendations: {len(dashboard['recommendations'])} actionable items")
        
        # Display key metrics
        if dashboard['performance_metrics']:
            self.logger.info("📈 Key Performance Metrics:")
            for metric, value in dashboard['performance_metrics'].items():
                self.logger.info(f"  - {metric.replace('_', ' ').title()}: {value:.2f}")
        
        # Display recommendations
        if dashboard['recommendations']:
            self.logger.info("💡 Top Recommendations:")
            for i, rec in enumerate(dashboard['recommendations'][:3], 1):
                self.logger.info(f"  {i}. {rec['description']} (Priority: {rec['priority']})")
        
        # Show visualizations available
        if 'visualizations' in dashboard:
            self.logger.info("📊 Generated Visualizations:")
            for viz_name in dashboard['visualizations'].keys():
                self.logger.info(f"  - {viz_name.replace('_', ' ').title()}")
    
    async def _demo_phase_9_continuous_innovation(self):
        """Demonstrate continuous innovation cycle"""
        self.logger.info("\n🔄 PHASE 9: Continuous Innovation Cycle")
        
        self.logger.info("🔍 Running Innovation Pipeline Analysis...")
        
        # Analyze innovation pipeline
        pipeline_analysis = await self.orchestrator._analyze_innovation_pipeline()
        self.logger.info(f"📊 Pipeline Analysis:")
        self.logger.info(f"  - Total Ideas: {pipeline_analysis['total_ideas']}")
        self.logger.info(f"  - Bottlenecks Identified: {len(pipeline_analysis['bottlenecks'])}")
        
        # Generate trend-based ideas
        new_idea_ids = await self.orchestrator._generate_trend_based_ideas(pipeline_analysis)
        self.logger.info(f"🎯 Generated {len(new_idea_ids)} trend-based innovation ideas")
        
        # Optimize existing ideas
        await self.orchestrator._optimize_existing_ideas()
        self.logger.info("⚡ Optimized existing ideas using AI")
        
        # Update AI models
        await self.orchestrator._update_ai_models()
        self.logger.info("🧠 Updated AI models with new data")
        
        self.logger.info("✅ Continuous innovation cycle completed")
    
    async def run_simple_demo(self):
        """Run a simplified demonstration of the framework"""
        self.logger.info("🚀 Starting Simplified Innovation Framework Demo")
        
        try:
            await self._initialize_framework()
            
            # Create one innovation idea
            idea_data = {
                'title': 'AI-Powered Developer Productivity Suite',
                'description': 'An integrated suite of AI tools to enhance developer productivity through intelligent code assistance, automated testing, and predictive debugging.',
                'type': 'breakthrough',
                'team_members': ['Tech Lead', 'AI Engineer', 'Product Manager']
            }
            
            idea_id = await self.orchestrator.create_innovation_idea(idea_data)
            self.logger.info(f"✅ Created innovation idea: {idea_data['title']}")
            
            # Advance through pipeline
            await self.orchestrator.advance_idea_status(idea_id, InnovationStatus.RESEARCH)
            await self.orchestrator.advance_idea_status(idea_id, InnovationStatus.DEVELOPMENT)
            
            # Generate dashboard
            dashboard = await self.orchestrator.get_innovation_dashboard()
            self.logger.info("📊 Innovation Framework Active")
            self.logger.info(f"  - Ideas Created: {dashboard['overview']['total_active_ideas']}")
            
            self.logger.info("✅ Simplified demo completed successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Demo failed: {str(e)}")
            raise

def print_banner():
    """Print demo banner"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    🚀 ENTERPRISE INNOVATION FRAMEWORK DEMO 🚀                ║
║                                                               ║
║    Continuous Innovation & Product Development System        ║
║                                                               ║
║    ✨ AI-Powered Feature Development                         ║
║    💬 Customer Feedback Systems                              ║
║    ⚡ Rapid Prototyping Engine                               ║
║    🔍 Competitive Analysis Engine                            ║
║    🗺️ Roadmap Optimization                                   ║
║    🧪 Innovation Labs                                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)

async def main():
    """Main demo function"""
    print_banner()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Initialize demo
    demo = InnovationFrameworkDemo()
    
    try:
        # Check if full demo or simple demo
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--simple':
            await demo.run_simple_demo()
        else:
            await demo.run_complete_demo()
            
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        return 1
    
    print("\n🎉 Demo completed! Check the logs for detailed output.")
    print("📋 Next steps:")
    print("   1. Review the generated dashboards")
    print("   2. Examine the configuration files")
    print("   3. Explore the code architecture")
    print("   4. Customize for your specific needs")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)