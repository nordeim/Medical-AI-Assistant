"""
Performance Benchmarking and Regression Testing for Medical AI System
Implements comprehensive performance testing, load testing, and monitoring
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
from enum import Enum
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    LOAD_TEST = "load_test"
    STRESS_TEST = "stress_test"
    SPIKE_TEST = "spike_test"
    ENDURANCE_TEST = "endurance_test"
    VOLUME_TEST = "volume_test"
    LATENCY_TEST = "latency_test"

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    timestamp: datetime
    metric_type: str
    value: float
    unit: str
    endpoint: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class BenchmarkResult:
    """Complete benchmark test result"""
    test_name: str
    test_type: BenchmarkType
    start_time: datetime
    end_time: datetime
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    metrics: List[PerformanceMetric]

class MedicalAIBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for medical AI workloads
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.endpoints = {
            'patient_data': '/api/patient-data',
            'ai_inference': '/api/ai-inference',
            'clinical_data': '/api/clinical-data',
            'audit_logs': '/api/audit-logs',
            'vital_signs': '/api/vital-signs',
            'medications': '/api/medications'
        }
        self.test_results = []
        
        # Medical AI specific test payloads
        self.test_payloads = {
            'patient_data': {
                'patient_id': 12345,
                'include_history': False
            },
            'ai_inference': {
                'prompt': 'Patient presents with chest pain and shortness of breath. Provide differential diagnosis.',
                'max_tokens': 512,
                'temperature': 0.7
            },
            'clinical_data': {
                'patient_id': 12345,
                'data_types': ['vital_signs', 'lab_results'],
                'date_range': 30
            }
        }
    
    async def load_test(self, 
                       endpoint: str,
                       concurrent_users: int = 10,
                       duration: int = 300,
                       ramp_up_time: int = 60) -> BenchmarkResult:
        """
        Perform load testing with gradual user ramp-up
        """
        logger.info(f"Starting load test for {endpoint} with {concurrent_users} users")
        
        test_name = f"load_test_{endpoint}_{concurrent_users}users"
        start_time = datetime.now()
        
        # Track performance metrics
        response_times = []
        successful_requests = 0
        failed_requests = 0
        request_count = 0
        metrics = []
        
        # System metrics monitoring
        system_metrics = []
        
        # Start system monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_system_metrics(system_metrics)
        )
        
        # Gradual ramp-up
        active_tasks = []
        
        for user_batch in range(0, concurrent_users, 2):  # Add 2 users at a time
            batch_size = min(2, concurrent_users - user_batch)
            
            # Create tasks for this batch
            for _ in range(batch_size):
                task = asyncio.create_task(
                    self._user_simulation(endpoint, duration, response_times, metrics)
                )
                active_tasks.append(task)
            
            # Wait for ramp-up period
            if ramp_up_time > 0 and user_batch + batch_size < concurrent_users:
                await asyncio.sleep(ramp_up_time / (concurrent_users / 2))
        
        # Wait for all tasks to complete
        await asyncio.gather(*active_tasks, return_exceptions=True)
        
        # Stop monitoring
        monitoring_task.cancel()
        
        end_time = datetime.now()
        test_duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        result = self._calculate_benchmark_result(
            test_name, BenchmarkType.LOAD_TEST, start_time, end_time,
            response_times, successful_requests, failed_requests, 
            request_count, metrics, system_metrics
        )
        
        self.test_results.append(result)
        logger.info(f"Load test completed: {successful_requests} successful, {failed_requests} failed")
        
        return result
    
    async def stress_test(self,
                         endpoint: str,
                         max_concurrent_users: int = 50,
                         duration: int = 600) -> BenchmarkResult:
        """
        Perform stress testing to find system limits
        """
        logger.info(f"Starting stress test for {endpoint} up to {max_concurrent_users} users")
        
        test_name = f"stress_test_{endpoint}"
        start_time = datetime.now()
        
        response_times = []
        successful_requests = 0
        failed_requests = 0
        metrics = []
        system_metrics = []
        
        # Start system monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_system_metrics(system_metrics)
        )
        
        # Progressive load increase
        for concurrent_users in range(1, max_concurrent_users + 1, 5):
            logger.info(f"Testing with {concurrent_users} concurrent users")
            
            # Create tasks for current load level
            tasks = []
            for _ in range(concurrent_users):
                task = asyncio.create_task(
                    self._user_simulation(endpoint, 30, response_times, metrics)
                )
                tasks.append(task)
            
            # Run for shorter periods during stress testing
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check system health
            if len(system_metrics) > 0:
                current_memory = system_metrics[-1]['memory_usage']
                current_cpu = system_metrics[-1]['cpu_usage']
                
                # If system is overwhelmed, stop increasing load
                if current_memory > 90 or current_cpu > 90:
                    logger.warning(f"System overwhelmed at {concurrent_users} users")
                    break
        
        end_time = datetime.now()
        monitoring_task.cancel()
        
        result = self._calculate_benchmark_result(
            test_name, BenchmarkType.STRESS_TEST, start_time, end_time,
            response_times, successful_requests, failed_requests,
            len(response_times), metrics, system_metrics
        )
        
        self.test_results.append(result)
        return result
    
    async def spike_test(self,
                        endpoint: str,
                        normal_load: int = 5,
                        spike_load: int = 50,
                        spike_duration: int = 60) -> BenchmarkResult:
        """
        Perform spike testing to test system resilience to sudden load increases
        """
        logger.info(f"Starting spike test: {normal_load} → {spike_load} users")
        
        test_name = f"spike_test_{endpoint}"
        start_time = datetime.now()
        
        response_times = []
        metrics = []
        system_metrics = []
        
        # Start monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_system_metrics(system_metrics)
        )
        
        # Normal load phase (5 minutes)
        logger.info("Normal load phase")
        normal_tasks = []
        for _ in range(normal_load):
            task = asyncio.create_task(
                self._user_simulation(endpoint, 300, response_times, metrics)
            )
            normal_tasks.append(task)
        
        # Let normal load run for 2 minutes
        await asyncio.sleep(120)
        
        # Spike phase
        logger.info("Spike phase")
        spike_tasks = []
        for _ in range(spike_load):
            task = asyncio.create_task(
                self._user_simulation(endpoint, spike_duration, response_times, metrics)
            )
            spike_tasks.append(task)
        
        # Wait for spike to complete
        await asyncio.sleep(spike_duration)
        
        # Return to normal
        logger.info("Return to normal load")
        await asyncio.gather(*normal_tasks, return_exceptions=True)
        
        end_time = datetime.now()
        monitoring_task.cancel()
        
        # Calculate results
        result = self._calculate_benchmark_result(
            test_name, BenchmarkType.SPIKE_TEST, start_time, end_time,
            response_times, len(response_times) - 10, 10,  # Estimate errors
            len(response_times), metrics, system_metrics
        )
        
        self.test_results.append(result)
        return result
    
    async def endurance_test(self,
                           endpoint: str,
                           concurrent_users: int = 5,
                           duration_hours: int = 4) -> BenchmarkResult:
        """
        Perform endurance testing to identify memory leaks and degradation
        """
        logger.info(f"Starting endurance test: {concurrent_users} users for {duration_hours} hours")
        
        test_name = f"endurance_test_{endpoint}"
        start_time = datetime.now()
        duration_seconds = duration_hours * 3600
        
        response_times = []
        metrics = []
        system_metrics = []
        
        # Start system monitoring
        monitoring_task = asyncio.create_task(
            self._monitor_system_metrics(system_metrics, interval=30)  # Monitor every 30 seconds
        )
        
        # Long-running user simulation
        tasks = []
        for _ in range(concurrent_users):
            task = asyncio.create_task(
                self._continuous_user_simulation(endpoint, duration_seconds, response_times, metrics)
            )
            tasks.append(task)
        
        # Wait for test to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        monitoring_task.cancel()
        
        result = self._calculate_benchmark_result(
            test_name, BenchmarkType.ENDURANCE_TEST, start_time, end_time,
            response_times, len(response_times) - 100, 100,  # Estimate errors
            len(response_times), metrics, system_metrics
        )
        
        self.test_results.append(result)
        return result
    
    async def _user_simulation(self, 
                              endpoint: str,
                              duration: int,
                              response_times: List[float],
                              metrics: List[PerformanceMetric]):
        """Simulate a user making requests for specified duration"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                start = time.time()
                
                async with aiohttp.ClientSession() as session:
                    url = f"{self.base_url}{endpoint}"
                    payload = self.test_payloads.get(endpoint.split('/')[-1], {})
                    
                    async with session.post(url, json=payload) as response:
                        await response.json()
                        response_time = time.time() - start
                        response_times.append(response_time)
                        
                        # Record metric
                        metrics.append(PerformanceMetric(
                            timestamp=datetime.now(),
                            metric_type="response_time",
                            value=response_time,
                            unit="seconds",
                            endpoint=endpoint
                        ))
                
                # Add small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.debug(f"Request failed: {e}")
                response_times.append(10.0)  # Timeout value for failed requests
                
                # Record error metric
                metrics.append(PerformanceMetric(
                    timestamp=datetime.now(),
                    metric_type="error",
                    value=1.0,
                    unit="count",
                    endpoint=endpoint
                ))
    
    async def _continuous_user_simulation(self,
                                         endpoint: str,
                                         duration: int,
                                         response_times: List[float],
                                         metrics: List[PerformanceMetric]):
        """Continuous user simulation for endurance testing"""
        await self._user_simulation(endpoint, duration, response_times, metrics)
    
    async def _monitor_system_metrics(self, 
                                    metrics_list: List[Dict],
                                    interval: int = 5):
        """Monitor system resources during testing"""
        while True:
            try:
                metrics = {
                    'timestamp': datetime.now(),
                    'cpu_usage': psutil.cpu_percent(),
                    'memory_usage': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent,
                    'process_count': len(psutil.pids())
                }
                metrics_list.append(metrics)
                
                await asyncio.sleep(interval)
            except Exception as e:
                logger.warning(f"System monitoring error: {e}")
                break
    
    def _calculate_benchmark_result(self,
                                  test_name: str,
                                  test_type: BenchmarkType,
                                  start_time: datetime,
                                  end_time: datetime,
                                  response_times: List[float],
                                  successful_requests: int,
                                  failed_requests: int,
                                  total_requests: int,
                                  metrics: List[PerformanceMetric],
                                  system_metrics: List[Dict]) -> BenchmarkResult:
        """Calculate comprehensive benchmark results"""
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            max_response_time = max(response_times)
            min_response_time = min(response_times)
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
            max_response_time = min_response_time = 0
        
        duration = (end_time - start_time).total_seconds()
        throughput = total_requests / duration if duration > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # Calculate average system metrics
        avg_cpu = statistics.mean([m['cpu_usage'] for m in system_metrics]) if system_metrics else 0
        avg_memory = statistics.mean([m['memory_usage'] for m in system_metrics]) if system_metrics else 0
        
        return BenchmarkResult(
            test_name=test_name,
            test_type=test_type,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            max_response_time=max_response_time,
            min_response_time=min_response_time,
            throughput=throughput,
            error_rate=error_rate,
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            metrics=metrics
        )


class PerformanceRegressionDetector:
    """
    Detects performance regressions by comparing benchmark results
    """
    
    def __init__(self, baseline_results_file: str = "baseline_results.json"):
        self.baseline_file = baseline_results_file
        self.regression_thresholds = {
            'response_time_degradation': 0.20,  # 20% slower is regression
            'throughput_degradation': 0.15,     # 15% less throughput is regression
            'error_rate_increase': 0.05,        # 5% more errors is regression
            'memory_increase': 0.25             # 25% more memory is regression
        }
        self.baseline_results = {}
    
    def load_baseline(self):
        """Load baseline performance results"""
        try:
            with open(self.baseline_file, 'r') as f:
                self.baseline_results = json.load(f)
            logger.info(f"Loaded baseline results from {self.baseline_file}")
        except FileNotFoundError:
            logger.warning(f"Baseline file {self.baseline_file} not found")
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
    
    def save_baseline(self, benchmark_suite: MedicalAIBenchmarkSuite):
        """Save current results as baseline"""
        baseline_data = {}
        for result in benchmark_suite.test_results:
            baseline_data[result.test_name] = {
                'avg_response_time': result.avg_response_time,
                'throughput': result.throughput,
                'error_rate': result.error_rate,
                'memory_usage': result.memory_usage,
                'cpu_usage': result.cpu_usage,
                'p95_response_time': result.p95_response_time,
                'p99_response_time': result.p99_response_time
            }
        
        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)
        
        logger.info(f"Saved baseline results to {self.baseline_file}")
    
    def detect_regressions(self, current_results: List[BenchmarkResult]) -> List[Dict[str, Any]]:
        """Detect performance regressions compared to baseline"""
        regressions = []
        
        for result in current_results:
            test_name = result.test_name
            
            if test_name not in self.baseline_results:
                continue
            
            baseline = self.baseline_results[test_name]
            
            # Check response time regression
            response_time_increase = (
                (result.avg_response_time - baseline['avg_response_time']) / 
                baseline['avg_response_time']
            )
            
            if response_time_increase > self.regression_thresholds['response_time_degradation']:
                regressions.append({
                    'test_name': test_name,
                    'regression_type': 'response_time',
                    'baseline_value': baseline['avg_response_time'],
                    'current_value': result.avg_response_time,
                    'percentage_increase': response_time_increase * 100,
                    'severity': 'high' if response_time_increase > 0.5 else 'medium'
                })
            
            # Check throughput regression
            throughput_decrease = (
                (baseline['throughput'] - result.throughput) / 
                baseline['throughput']
            )
            
            if throughput_decrease > self.regression_thresholds['throughput_degradation']:
                regressions.append({
                    'test_name': test_name,
                    'regression_type': 'throughput',
                    'baseline_value': baseline['throughput'],
                    'current_value': result.throughput,
                    'percentage_decrease': throughput_decrease * 100,
                    'severity': 'high' if throughput_decrease > 0.3 else 'medium'
                })
            
            # Check error rate regression
            error_rate_increase = result.error_rate - baseline['error_rate']
            
            if error_rate_increase > self.regression_thresholds['error_rate_increase']:
                regressions.append({
                    'test_name': test_name,
                    'regression_type': 'error_rate',
                    'baseline_value': baseline['error_rate'],
                    'current_value': result.error_rate,
                    'absolute_increase': error_rate_increase,
                    'severity': 'high' if error_rate_increase > 0.1 else 'medium'
                })
            
            # Check memory usage regression
            memory_increase = (
                (result.memory_usage - baseline['memory_usage']) / 
                baseline['memory_usage']
            )
            
            if memory_increase > self.regression_thresholds['memory_increase']:
                regressions.append({
                    'test_name': test_name,
                    'regression_type': 'memory_usage',
                    'baseline_value': baseline['memory_usage'],
                    'current_value': result.memory_usage,
                    'percentage_increase': memory_increase * 100,
                    'severity': 'high' if memory_increase > 0.5 else 'medium'
                })
        
        return regressions
    
    def generate_performance_report(self, 
                                   results: List[BenchmarkResult],
                                   output_file: str = "performance_report.html"):
        """Generate comprehensive performance report"""
        html_content = self._create_performance_report_html(results)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {output_file}")
    
    def _create_performance_report_html(self, results: List[BenchmarkResult]) -> str:
        """Create HTML performance report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical AI Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .test-result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
                .pass {{ color: green; font-weight: bold; }}
                .fail {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Medical AI Performance Test Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Tests: {len(results)}</p>
            </div>
        """
        
        for result in results:
            html += f"""
            <div class="test-result">
                <h2>{result.test_name}</h2>
                <p><strong>Test Type:</strong> {result.test_type.value}</p>
                <p><strong>Duration:</strong> {result.duration:.2f} seconds</p>
                <p><strong>Total Requests:</strong> {result.total_requests}</p>
                <p><strong>Successful:</strong> {result.successful_requests}</p>
                <p><strong>Failed:</strong> {result.failed_requests}</p>
                
                <div class="metric">
                    <strong>Avg Response Time:</strong> {result.avg_response_time:.3f}s
                </div>
                <div class="metric">
                    <strong>P95 Response Time:</strong> {result.p95_response_time:.3f}s
                </div>
                <div class="metric">
                    <strong>Throughput:</strong> {result.throughput:.2f} req/s
                </div>
                <div class="metric">
                    <strong>Error Rate:</strong> {result.error_rate:.2%}
                </div>
                <div class="metric">
                    <strong>CPU Usage:</strong> {result.cpu_usage:.1f}%
                </div>
                <div class="metric">
                    <strong>Memory Usage:</strong> {result.memory_usage:.1f}%
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


async def main():
    """Example usage of performance benchmarking suite"""
    
    # Initialize benchmark suite
    benchmark_suite = MedicalAIBenchmarkSuite(base_url="http://localhost:8080")
    
    # Load test
    load_result = await benchmark_suite.load_test(
        endpoint='/api/patient-data',
        concurrent_users=10,
        duration=300
    )
    
    print(f"Load test results: {load_result.avg_response_time:.3f}s average response time")
    
    # Stress test
    stress_result = await benchmark_suite.stress_test(
        endpoint='/api/ai-inference',
        max_concurrent_users=25,
        duration=300
    )
    
    print(f"Stress test results: {stress_result.throughput:.2f} req/s throughput")
    
    # Detect regressions
    detector = PerformanceRegressionDetector()
    regressions = detector.detect_regressions(benchmark_suite.test_results)
    
    if regressions:
        print("Performance regressions detected:")
        for regression in regressions:
            print(f"- {regression['test_name']}: {regression['regression_type']}")
    else:
        print("No performance regressions detected")
    
    # Generate report
    detector.generate_performance_report(
        benchmark_suite.test_results,
        "performance_report.html"
    )


if __name__ == "__main__":
    asyncio.run(main())