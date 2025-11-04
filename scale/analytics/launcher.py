"""
Advanced Analytics and Business Intelligence Platform
Main entry point and launcher

This is a comprehensive analytics platform that provides:
- AI-powered insights and analytics
- Predictive analytics for business forecasting
- Customer behavior analytics and segmentation
- Market intelligence and competitive analysis
- Clinical outcome analytics for healthcare
- Operational analytics and efficiency measurement
- Executive decision support and strategic intelligence
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add the analytics directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.orchestrator import AnalyticsOrchestrator
from config.configuration import ConfigManager, get_production_config, get_development_config

class AnalyticsPlatformLauncher:
    """Main launcher for the Advanced Analytics Platform"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.orchestrator = None
        self.config_manager = None
        
    def initialize_platform(self) -> None:
        """Initialize the analytics platform"""
        print(f"Initializing Advanced Analytics Platform (v1.0.0)")
        print(f"Environment: {self.environment}")
        print("=" * 60)
        
        try:
            # Load configuration
            if self.environment == "production":
                config = get_production_config()
            else:
                config = get_development_config()
            
            self.config_manager = ConfigManager()
            self.config_manager.config = config
            self.config_manager.save_config(config)
            
            # Initialize orchestrator
            self.orchestrator = AnalyticsOrchestrator()
            
            print("✓ Platform initialization completed successfully")
            
        except Exception as e:
            print(f"✗ Platform initialization failed: {e}")
            raise
    
    def run_demo(self) -> None:
        """Run a demonstration of the platform capabilities"""
        print("\n🚀 Running Analytics Platform Demo")
        print("-" * 40)
        
        # Generate demo data
        demo_data = self._generate_demo_data()
        
        # Run comprehensive analysis
        reports = self.orchestrator.execute_comprehensive_analysis(demo_data)
        
        # Display results
        self._display_demo_results(reports)
        
        print("\n✅ Demo completed successfully!")
    
    def generate_sample_report(self) -> str:
        """Generate a sample analytics report"""
        demo_data = self._generate_demo_data()
        
        # Run a single pipeline for demonstration
        report = self.orchestrator.execute_pipeline("executive_dashboard", demo_data)
        
        # Convert report to JSON
        report_data = {
            "report_id": report.report_id,
            "title": report.title,
            "executive_summary": report.executive_summary,
            "key_insights": report.key_insights,
            "recommendations": report.recommendations,
            "generated_at": report.generated_at.isoformat(),
            "confidence_score": report.confidence_score
        }
        
        # Save report
        report_path = f"sample_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get comprehensive platform information"""
        if not self.orchestrator:
            return {"error": "Platform not initialized"}
        
        status = self.orchestrator.get_platform_status()
        insights = self.orchestrator.get_insights_summary()
        
        return {
            "platform_status": status,
            "insights_summary": insights,
            "capabilities": {
                "ai_powered_insights": "Advanced machine learning algorithms for automated insights",
                "predictive_analytics": "Business forecasting and trend analysis",
                "customer_analytics": "Customer segmentation and behavior analysis",
                "market_intelligence": "Competitive analysis and market insights",
                "clinical_analytics": "Healthcare outcome analytics and quality metrics",
                "operational_analytics": "Efficiency measurement and process optimization",
                "executive_intelligence": "Strategic decision support and board-level reporting"
            }
        }
    
    def _generate_demo_data(self) -> Dict[str, Any]:
        """Generate sample data for demonstration"""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)  # For reproducible results
        
        # Main business data
        main_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'revenue': np.random.uniform(50000, 150000, 100),
            'customers': np.random.randint(100, 500, 100),
            'satisfaction': np.random.uniform(3.5, 5.0, 100),
            'costs': np.random.uniform(30000, 80000, 100),
            'efficiency': np.random.uniform(70, 95, 100)
        })
        
        # Customer data
        customer_data = pd.DataFrame({
            'customer_id': range(1, 201),
            'age': np.random.randint(18, 80, 200),
            'gender': np.random.choice(['M', 'F'], 200),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], 200),
            'income': np.random.randint(30000, 150000, 200),
            'tenure_months': np.random.randint(1, 60, 200)
        })
        
        # Transaction data
        transaction_data = pd.DataFrame({
            'customer_id': np.random.choice(range(1, 201), 1000),
            'date': pd.date_range('2023-01-01', periods=1000, freq='H'),
            'amount': np.random.uniform(10, 1000, 1000),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food'], 1000),
            'quantity': np.random.randint(1, 5, 1000)
        })
        
        # Market data
        market_data = pd.DataFrame({
            'market_name': ['Technology Sector'] * 50,
            'revenue': np.random.uniform(10000000, 100000000, 50),
            'growth_rate': np.random.uniform(0.05, 0.25, 50),
            'market_share': np.random.uniform(0.1, 0.4, 50),
            'competitive_intensity': np.random.uniform(0.3, 0.8, 50)
        })
        
        # Competitor data
        competitor_data = pd.DataFrame({
            'id': range(1, 11),
            'name': [f'Competitor {i}' for i in range(1, 11)],
            'revenue': np.random.uniform(1000000, 50000000, 10),
            'market_share': np.random.uniform(0.05, 0.30, 10),
            'growth_rate': np.random.uniform(0.02, 0.20, 10),
            'profit_margin': np.random.uniform(0.05, 0.25, 10)
        })
        
        # Operational data
        operational_data = pd.DataFrame({
            'process_id': range(1, 101),
            'process_name': np.random.choice([
                'Manufacturing', 'Quality Control', 'Shipping', 'Customer Service', 'IT Support'
            ], 100),
            'efficiency': np.random.uniform(60, 98, 100),
            'cost': np.random.uniform(1000, 50000, 100),
            'time_hours': np.random.uniform(1, 48, 100),
            'quality_score': np.random.uniform(3.0, 5.0, 100)
        })
        
        # Clinical data (healthcare analytics)
        clinical_data = pd.DataFrame({
            'patient_id': range(1, 501),
            'age': np.random.randint(18, 90, 500),
            'readmission_rate': np.random.uniform(0.05, 0.20, 500),
            'mortality_rate': np.random.uniform(0.005, 0.03, 500),
            'length_of_stay': np.random.uniform(2, 10, 500),
            'patient_satisfaction': np.random.uniform(3.0, 5.0, 500),
            'complication_rate': np.random.uniform(0.02, 0.08, 500)
        })
        
        return {
            "main_data": main_data,
            "customer_data": customer_data,
            "transaction_data": transaction_data,
            "market_data": market_data,
            "competitor_data": competitor_data,
            "operational_data": operational_data,
            "clinical_data": clinical_data
        }
    
    def _display_demo_results(self, reports: Dict[str, Any]) -> None:
        """Display demonstration results"""
        print(f"\n📊 Analytics Reports Generated: {len(reports)}")
        
        for pipeline_id, report in reports.items():
            print(f"\n--- {report.title} ---")
            print(f"Report ID: {report.report_id}")
            print(f"Confidence Score: {report.confidence_score:.1%}")
            print(f"Key Insights: {len(report.key_insights)}")
            print(f"Recommendations: {len(report.recommendations)}")
            
            # Show top insights
            if report.key_insights:
                print("\nTop Insights:")
                for i, insight in enumerate(report.key_insights[:5], 1):
                    print(f"  {i}. {insight}")
    
    def interactive_mode(self) -> None:
        """Run the platform in interactive mode"""
        print("\n🎯 Entering Interactive Mode")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command = input("\nanalytics> ").strip().lower()
                
                if command == "exit" or command == "quit":
                    print("Goodbye!")
                    break
                elif command == "help":
                    self._show_interactive_help()
                elif command == "status":
                    info = self.get_platform_info()
                    self._display_platform_info(info)
                elif command == "demo":
                    self.run_demo()
                elif command == "report":
                    report_path = self.generate_sample_report()
                    print(f"Sample report generated: {report_path}")
                elif command == "pipelines":
                    self._list_pipelines()
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_interactive_help(self) -> None:
        """Show interactive mode help"""
        print("\n📖 Available Commands:")
        print("  status  - Show platform status and capabilities")
        print("  demo    - Run platform demonstration")
        print("  report  - Generate sample analytics report")
        print("  pipelines - List available analytics pipelines")
        print("  help    - Show this help message")
        print("  exit    - Exit the platform")
    
    def _display_platform_info(self, info: Dict[str, Any]) -> None:
        """Display platform information"""
        print("\n🏢 Platform Information:")
        status = info["platform_status"]["platform_info"]
        print(f"  Name: {status['name']}")
        print(f"  Version: {status['version']}")
        print(f"  Environment: {status['environment']}")
        print(f"  Uptime: {status['uptime']}")
        
        print("\n📈 Capabilities:")
        for capability, description in info["capabilities"].items():
            print(f"  • {capability.replace('_', ' ').title()}: {description}")
    
    def _list_pipelines(self) -> None:
        """List available analytics pipelines"""
        print("\n🔄 Available Analytics Pipelines:")
        for pipeline_id, pipeline in self.orchestrator.pipelines.items():
            status = "✓ Enabled" if pipeline.enabled else "✗ Disabled"
            last_exec = pipeline.last_execution.strftime("%Y-%m-%d %H:%M") if pipeline.last_execution else "Never"
            print(f"  • {pipeline.name} ({pipeline_id})")
            print(f"    Status: {status} | Last Run: {last_exec}")
            print(f"    Schedule: {pipeline.schedule}")
            print(f"    Description: {pipeline.description}")
            print()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Advanced Analytics and Business Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py demo                    # Run platform demonstration
  python launcher.py interactive             # Start interactive mode
  python launcher.py report                  # Generate sample report
  python launcher.py status                  # Show platform status
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive", "report", "status"],
        default="demo",
        help="Execution mode (default: demo)"
    )
    
    parser.add_argument(
        "--env",
        choices=["development", "production"],
        default="development",
        help="Platform environment (default: development)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize platform
    launcher = AnalyticsPlatformLauncher(args.env)
    
    try:
        if args.config:
            # Load custom configuration
            launcher.config_manager = ConfigManager(args.config)
        
        launcher.initialize_platform()
        
        # Execute requested mode
        if args.mode == "demo":
            launcher.run_demo()
        elif args.mode == "interactive":
            launcher.interactive_mode()
        elif args.mode == "report":
            report_path = launcher.generate_sample_report()
            print(f"📊 Sample report generated: {report_path}")
        elif args.mode == "status":
            info = launcher.get_platform_info()
            launcher._display_platform_info(info)
            
    except Exception as e:
        print(f"❌ Platform error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()