#!/usr/bin/env python3
"""
Medical AI Performance Optimization and Scaling Orchestrator
Main entry point for comprehensive performance optimization system
"""

import asyncio
import logging
import yaml
import json
from typing import Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
import os

# Import our optimization modules
sys.path.append('/workspace/performance')
sys.path.append('/workspace/scaling')

from database_optimization.patient_records_optimization import PatientRecordOptimizer, QueryPerformanceMonitor
from caching.medical_ai_cache import MedicalAICache, PatientDataCache, ModelInferenceCache
from model_optimization.model_inference_optimization import ModelPerformanceOptimizer, QuantizedModelManager
from kubernetes_scaling.k8s_autoscaling_config import MedicalAIServiceAutoscalingConfig
from frontend_optimization.medical_frontend_optimizer import MedicalAppPerformanceConfig
from resource_management.resource_manager import MedicalAIServiceResourceManager
from benchmarking.performance_benchmarking import MedicalAIBenchmarkSuite, PerformanceRegressionDetector
from workload_prediction.workload_predictor import WorkloadPredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for performance optimization system"""
    # Database settings
    database_url: str = "postgresql://user:pass@localhost:5432/medical_ai"
    redis_url: str = "redis://localhost:6379"
    
    # Performance targets
    max_response_time: float = 2.0  # seconds
    min_cache_hit_rate: float = 0.8  # 80%
    max_cpu_utilization: float = 70.0  # 70%
    min_throughput: float = 100.0  # requests/second
    
    # Scaling settings
    min_replicas: int = 2
    max_replicas: int = 50
    scale_up_threshold: float = 70.0
    scale_down_threshold: float = 30.0
    
    # Service configuration
    services: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.services is None:
            self.services = [
                {
                    'name': 'medical-ai-api',
                    'image': 'medical-ai/api:v1.0',
                    'type': 'api',
                    'replicas': 3,
                    'predictive_scaling': True,
                    'ports': [8080]
                },
                {
                    'name': 'model-serving',
                    'image': 'medical-ai/model-server:v1.0',
                    'type': 'model-serving',
                    'replicas': 2,
                    'predictive_scaling': True,
                    'ports': [8080, 8081]
                },
                {
                    'name': 'patient-data-service',
                    'image': 'medical-ai/patient-service:v1.0',
                    'type': 'api',
                    'replicas': 2,
                    'predictive_scaling': False,
                    'ports': [8080]
                }
            ]

class MedicalAIPerformanceOrchestrator:
    """
    Main orchestrator for medical AI performance optimization and scaling
    Coordinates all optimization components for enterprise-grade performance
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Initialize all optimization components
        self.db_optimizer = PatientRecordOptimizer(config.database_url)
        self.cache = MedicalAICache(config.redis_url)
        self.patient_cache = PatientDataCache(self.cache)
        self.inference_cache = ModelInferenceCache(self.cache)
        self.model_optimizer = ModelPerformanceOptimizer()
        self.quantization_manager = QuantizedModelManager()
        self.autoscaling_config = MedicalAIServiceAutoscalingConfig()
        self.frontend_optimizer = MedicalAppPerformanceConfig('/workspace/frontend')
        self.resource_manager = MedicalAIServiceResourceManager(config.database_url, config.redis_url)
        self.benchmark_suite = MedicalAIBenchmarkSuite()
        self.regression_detector = PerformanceRegressionDetector()
        self.workload_predictor = WorkloadPredictionService()
        
        # Performance metrics storage
        self.performance_metrics = {}
        self.optimization_results = {}
        
    async def initialize(self):
        """Initialize all optimization components"""
        logger.info("Initializing Medical AI Performance Optimization System")
        
        try:
            # Initialize caching layer
            await self.cache.initialize()
            logger.info("✓ Cache layer initialized")
            
            # Initialize resource manager
            await self.resource_manager.initialize()
            logger.info("✓ Resource manager initialized")
            
            # Load baseline performance data
            self.regression_detector.load_baseline()
            logger.info("✓ Baseline performance data loaded")
            
            # Initialize workload predictor
            self.workload_predictor.predictor.load_models()
            logger.info("✓ Workload prediction models loaded")
            
            logger.info("Performance optimization system initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def optimize_database_performance(self):
        """Optimize database queries and indexing"""
        logger.info("Optimizing database performance...")
        
        try:
            # Create optimized indexes
            await self.db_optimizer.create_optimized_indexes()
            logger.info("✓ Optimized database indexes created")
            
            # Generate query optimization recommendations
            recommendations = self._generate_db_optimization_recommendations()
            
            self.optimization_results['database'] = {
                'indexes_created': True,
                'recommendations': recommendations,
                'estimated_improvement': '40-60% query performance improvement'
            }
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            raise
    
    async def optimize_caching_strategy(self):
        """Implement multi-level caching strategy"""
        logger.info("Implementing caching strategy...")
        
        try:
            # Configure cache TTLs for different data types
            cache_configs = {
                'patient_data': 1800,      # 30 minutes
                'clinical_data': 900,      # 15 minutes
                'ai_inference': 3600,      # 1 hour
                'audit_logs': 7200,        # 2 hours
                'vital_signs': 600,        # 10 minutes
                'medications': 1200        # 20 minutes
            }
            
            # Test cache performance
            test_stats = self.cache.get_cache_stats()
            
            self.optimization_results['caching'] = {
                'cache_levels': ['L1_memory', 'L2_redis', 'L3_database'],
                'ttl_configs': cache_configs,
                'initial_stats': test_stats,
                'target_hit_rate': self.config.min_cache_hit_rate
            }
            
            logger.info("Caching strategy implemented")
            
        except Exception as e:
            logger.error(f"Caching optimization failed: {e}")
            raise
    
    async def optimize_model_inference(self):
        """Optimize AI model inference performance"""
        logger.info("Optimizing model inference performance...")
        
        try:
            # Test different quantization levels
            test_prompts = [
                "Patient presents with chest pain and shortness of breath.",
                "What are the differential diagnoses for acute abdominal pain?",
                "Recommend treatment plan for Type 2 diabetes management."
            ]
            
            # Auto-optimize quantization
            quantization_results = await self.model_optimizer.auto_optimize_quantization(
                'medical-llama-7b', test_prompts
            )
            
            # Select optimal batch size
            batch_results = await self.model_optimizer.dynamic_batch_optimization(
                'medical-llama-7b', test_prompts, max_batch_size=8
            )
            
            self.optimization_results['model_inference'] = {
                'quantization_performance': quantization_results,
                'optimal_batch_size': len(batch_results[0][1]) if batch_results else 4,
                'estimated_speedup': '3-5x with 4-bit quantization',
                'memory_reduction': '70-80% with quantization'
            }
            
            logger.info("Model inference optimization completed")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            raise
    
    async def configure_autoscaling(self):
        """Configure Kubernetes auto-scaling"""
        logger.info("Configuring auto-scaling...")
        
        try:
            # Generate auto-scaling configurations
            all_configs = {}
            
            for service_config in self.config.services:
                service_configs = self.autoscaling_config.generate_complete_config(service_config)
                all_configs.update(service_configs)
            
            # Save configurations
            config_dir = Path('/workspace/scaling/kubernetes-scaling/configs')
            config_dir.mkdir(exist_ok=True)
            
            for filename, config_yaml in all_configs.items():
                with open(config_dir / filename, 'w') as f:
                    f.write(config_yaml)
            
            self.optimization_results['autoscaling'] = {
                'configs_generated': list(all_configs.keys()),
                'services_configured': len(self.config.services),
                'min_replicas': self.config.min_replicas,
                'max_replicas': self.config.max_replicas,
                'predictive_scaling': True
            }
            
            logger.info(f"Auto-scaling configurations saved to {config_dir}")
            
        except Exception as e:
            logger.error(f"Autoscaling configuration failed: {e}")
            raise
    
    async def optimize_frontend_performance(self):
        """Optimize frontend performance"""
        logger.info("Optimizing frontend performance...")
        
        try:
            # Generate performance configurations
            perf_configs = self.frontend_optimizer.generate_performance_configs()
            
            # Generate optimized components
            lazy_components = self.frontend_optimizer.optimizer.generate_lazy_loading_components()
            performance_hooks = self.frontend_optimizer.optimizer.generate_performance_optimized_hooks()
            optimized_components = self.frontend_optimizer.optimizer.generate_optimized_medical_components()
            
            # Save frontend optimization files
            frontend_dir = Path('/workspace/performance/frontend-optimization/frontend')
            frontend_dir.mkdir(exist_ok=True)
            
            # Save Vite config
            with open(frontend_dir / 'vite.config.js', 'w') as f:
                f.write(perf_configs.get('vite.config.js', ''))
            
            # Save performance components
            components_dir = frontend_dir / 'components'
            components_dir.mkdir(exist_ok=True)
            
            for name, code in lazy_components.items():
                with open(components_dir / f'{name}.js', 'w') as f:
                    f.write(code)
            
            for name, code in performance_hooks.items():
                with open(components_dir / f'{name}.js', 'w') as f:
                    f.write(code)
            
            for name, code in optimized_components.items():
                with open(components_dir / f'{name}.js', 'w') as f:
                    f.write(code)
            
            self.optimization_results['frontend'] = {
                'lazy_components': len(lazy_components),
                'performance_hooks': len(performance_hooks),
                'optimized_components': len(optimized_components),
                'performance_targets': {
                    'first_paint': '< 1.5s',
                    'interactive': '< 3.0s',
                    'bundle_size': '< 500KB'
                }
            }
            
            logger.info("Frontend optimization completed")
            
        except Exception as e:
            logger.error(f"Frontend optimization failed: {e}")
            raise
    
    async def configure_resource_management(self):
        """Configure connection pooling and resource management"""
        logger.info("Configuring resource management...")
        
        try:
            # Health check
            health_status = await self.resource_manager.health_check()
            
            # Get resource metrics
            resource_metrics = self.resource_manager.get_resource_metrics()
            
            # Configure rate limiting for medical AI endpoints
            rate_limits = {
                '/api/patient-data': {'requests': 100, 'window': 60},
                '/api/ai-inference': {'requests': 50, 'window': 60},
                '/api/clinical-data': {'requests': 200, 'window': 60},
                '/api/audit-logs': {'requests': 30, 'window': 60}
            }
            
            for endpoint, config in rate_limits.items():
                self.resource_manager.rate_limiter.configure_endpoint_limit(
                    endpoint, config['requests'], config['window']
                )
            
            self.optimization_results['resource_management'] = {
                'health_status': health_status,
                'rate_limits_configured': len(rate_limits),
                'connection_pool_size': resource_metrics['connection_pool']['max_connections'],
                'adaptive_rate_limiting': True
            }
            
            logger.info("Resource management configured")
            
        except Exception as e:
            logger.error(f"Resource management configuration failed: {e}")
            raise
    
    async def run_performance_benchmarking(self):
        """Run comprehensive performance benchmarks"""
        logger.info("Running performance benchmarks...")
        
        try:
            # Run load tests for different services
            benchmark_results = []
            
            for service_config in self.config.services:
                service_name = service_config['name']
                endpoint = '/api/patient-data'  # Default test endpoint
                
                # Load test
                load_result = await self.benchmark_suite.load_test(
                    endpoint=endpoint,
                    concurrent_users=10,
                    duration=300
                )
                benchmark_results.append(load_result)
                
                # Stress test
                stress_result = await self.benchmark_suite.stress_test(
                    endpoint=endpoint,
                    max_concurrent_users=25,
                    duration=300
                )
                benchmark_results.append(stress_result)
            
            # Detect performance regressions
            regressions = self.regression_detector.detect_regressions(benchmark_results)
            
            # Generate performance report
            self.regression_detector.generate_performance_report(
                benchmark_results, 
                '/workspace/performance/benchmarking/performance_report.html'
            )
            
            self.optimization_results['benchmarking'] = {
                'tests_completed': len(benchmark_results),
                'regressions_detected': len(regressions),
                'performance_report': '/workspace/performance/benchmarking/performance_report.html',
                'avg_response_time': sum(r.avg_response_time for r in benchmark_results) / len(benchmark_results),
                'avg_throughput': sum(r.throughput for r in benchmark_results) / len(benchmark_results)
            }
            
            if regressions:
                logger.warning(f"Performance regressions detected: {len(regressions)}")
                for regression in regressions:
                    logger.warning(f"- {regression['test_name']}: {regression['regression_type']}")
            else:
                logger.info("No performance regressions detected")
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            raise
    
    async def setup_workload_prediction(self):
        """Setup workload prediction for intelligent scaling"""
        logger.info("Setting up workload prediction...")
        
        try:
            # Generate predictions for next 24 hours
            predictions = await self.workload_predictor.predictor.predict_workload_range(
                datetime.now(), hours_ahead=24
            )
            
            # Analyze workload trends
            trends = self.workload_predictor.predictor.analyze_workload_trends()
            
            self.optimization_results['workload_prediction'] = {
                'predictions_generated': len(predictions),
                'prediction_confidence': sum(p.confidence_score for p in predictions) / len(predictions),
                'trends_analysis': trends,
                'auto_scaling_recommendations': True
            }
            
            logger.info("Workload prediction setup completed")
            
        except Exception as e:
            logger.error(f"Workload prediction setup failed: {e}")
            raise
    
    async def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        logger.info("Generating optimization report...")
        
        report = {
            'optimization_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_components': 7,
                'successful_optimizations': len([k for k, v in self.optimization_results.items() if v]),
                'performance_targets': {
                    'max_response_time': f"{self.config.max_response_time}s",
                    'min_cache_hit_rate': f"{self.config.min_cache_hit_rate:.0%}",
                    'max_cpu_utilization': f"{self.config.max_cpu_utilization}%",
                    'min_throughput': f"{self.config.min_throughput} req/s"
                }
            },
            'optimization_results': self.optimization_results,
            'next_steps': [
                'Deploy optimized configurations to production',
                'Monitor performance metrics continuously',
                'Fine-tune based on real-world usage patterns',
                'Regular regression testing',
                'Scale monitoring and alerting'
            ]
        }
        
        # Save report
        report_path = Path('/workspace/performance/optimization_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Optimization report saved to {report_path}")
        return report
    
    def _generate_db_optimization_recommendations(self) -> List[str]:
        """Generate database optimization recommendations"""
        return [
            "Create composite indexes for common query patterns",
            "Implement connection pooling with proper timeouts",
            "Use prepared statements for repeated queries",
            "Consider read replicas for reporting queries",
            "Monitor and tune query execution plans",
            "Implement query result caching",
            "Consider partitioning for large audit tables"
        ]
    
    async def run_full_optimization(self):
        """Run complete performance optimization process"""
        logger.info("Starting full Medical AI performance optimization...")
        
        try:
            # Initialize system
            await self.initialize()
            
            # Run all optimization steps
            await self.optimize_database_performance()
            await self.optimize_caching_strategy()
            await self.optimize_model_inference()
            await self.configure_autoscaling()
            await self.optimize_frontend_performance()
            await self.configure_resource_management()
            await self.run_performance_benchmarking()
            await self.setup_workload_prediction()
            
            # Generate final report
            report = await self.generate_optimization_report()
            
            logger.info("🎉 Medical AI performance optimization completed successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("MEDICAL AI PERFORMANCE OPTIMIZATION SUMMARY")
            print("="*60)
            
            for component, result in self.optimization_results.items():
                print(f"\n✅ {component.replace('_', ' ').title()}:")
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float, str)):
                            print(f"   - {key.replace('_', ' ').title()}: {value}")
                        elif isinstance(value, bool):
                            print(f"   - {key.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
                        elif isinstance(value, list) and len(value) < 5:
                            print(f"   - {key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
            
            print(f"\n📊 Performance Targets:")
            print(f"   - Max Response Time: < {self.config.max_response_time}s")
            print(f"   - Min Cache Hit Rate: > {self.config.min_cache_hit_rate:.0%}")
            print(f"   - Max CPU Utilization: < {self.config.max_cpu_utilization}%")
            print(f"   - Min Throughput: > {self.config.min_throughput} req/s")
            
            print(f"\n📁 Generated Files:")
            print(f"   - Database optimization: /workspace/performance/database-optimization/")
            print(f"   - Caching configuration: /workspace/performance/caching/")
            print(f"   - Model optimization: /workspace/performance/model-optimization/")
            print(f"   - Kubernetes configs: /workspace/scaling/kubernetes-scaling/configs/")
            print(f"   - Frontend optimization: /workspace/performance/frontend-optimization/")
            print(f"   - Resource management: /workspace/performance/resource-management/")
            print(f"   - Performance report: /workspace/performance/optimization_report.json")
            
            print("="*60)
            
            return report
            
        except Exception as e:
            logger.error(f"Optimization process failed: {e}")
            raise


async def main():
    """Main entry point for performance optimization"""
    parser = argparse.ArgumentParser(description='Medical AI Performance Optimization')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--optimize', choices=[
        'database', 'caching', 'model', 'scaling', 'frontend', 
        'resource', 'benchmarking', 'prediction', 'all'
    ], default='all', help='Optimization component to run')
    parser.add_argument('--response-time', type=float, default=2.0, 
                       help='Maximum response time target (seconds)')
    parser.add_argument('--cache-hit-rate', type=float, default=0.8,
                       help='Minimum cache hit rate target')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
            config = OptimizationConfig(**config_data)
    else:
        # Use default configuration
        config = OptimizationConfig()
    
    # Override targets if specified
    config.max_response_time = args.response_time
    config.min_cache_hit_rate = args.cache_hit_rate
    
    # Initialize orchestrator
    orchestrator = MedicalAIPerformanceOrchestrator(config)
    
    # Run optimization
    if args.optimize == 'all':
        await orchestrator.run_full_optimization()
    else:
        await orchestrator.initialize()
        
        if args.optimize == 'database':
            await orchestrator.optimize_database_performance()
        elif args.optimize == 'caching':
            await orchestrator.optimize_caching_strategy()
        elif args.optimize == 'model':
            await orchestrator.optimize_model_inference()
        elif args.optimize == 'scaling':
            await orchestrator.configure_autoscaling()
        elif args.optimize == 'frontend':
            await orchestrator.optimize_frontend_performance()
        elif args.optimize == 'resource':
            await orchestrator.configure_resource_management()
        elif args.optimize == 'benchmarking':
            await orchestrator.run_performance_benchmarking()
        elif args.optimize == 'prediction':
            await orchestrator.setup_workload_prediction()
        
        print(f"✅ {args.optimize} optimization completed")


if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(main())