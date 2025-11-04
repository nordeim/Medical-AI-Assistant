#!/usr/bin/env python3
"""
Automated Test Execution Script
===============================

Comprehensive automated testing and quality assurance pipeline for Medical AI Training System.

Features:
- Automated test execution with parallel processing
- Comprehensive test result reporting
- CI/CD integration support
- Quality gate enforcement
- Performance monitoring
- Coverage analysis
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import shutil
import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

class TestRunner:
    """Main test execution orchestrator."""
    
    def __init__(self, config_path: str = "test_config.yaml"):
        self.config = self._load_config(config_path)
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize test tracking
        self.test_results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'test_suite': 'Medical AI Training System',
            'version': self._get_version(),
            'test_categories': {},
            'overall_metrics': {},
            'quality_gates': {},
            'recommendations': []
        }
        
        # Initialize quality gates
        self.quality_gates = {
            'min_test_coverage': 80.0,
            'max_failure_rate': 5.0,
            'min_performance_score': 70.0,
            'max_memory_usage_mb': 2048,
            'max_execution_time_minutes': 30
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load test configuration."""
        default_config = {
            'test_categories': {
                'unit_tests': {
                    'enabled': True,
                    'timeout_minutes': 10,
                    'parallel': True,
                    'files': ['test_lora_training.py']
                },
                'integration_tests': {
                    'enabled': True,
                    'timeout_minutes': 15,
                    'parallel': False,
                    'files': ['comprehensive_test_suite.py::TestIntegrationWorkflows']
                },
                'performance_tests': {
                    'enabled': True,
                    'timeout_minutes': 20,
                    'parallel': True,
                    'files': ['performance_benchmarks.py']
                },
                'data_quality_tests': {
                    'enabled': True,
                    'timeout_minutes': 5,
                    'parallel': True,
                    'files': ['test_data_quality.py']
                },
                'stress_tests': {
                    'enabled': True,
                    'timeout_minutes': 15,
                    'parallel': False,
                    'files': ['comprehensive_test_suite.py::TestStressConditions']
                }
            },
            'reporting': {
                'format': ['json', 'html', 'xml'],
                'output_dir': 'test_results',
                'include_logs': True,
                'include_performance_metrics': True
            },
            'quality_gates': {
                'min_test_coverage': 80.0,
                'max_failure_rate': 5.0,
                'min_performance_score': 70.0,
                'max_memory_usage_mb': 2048
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if key in default_config and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}. Using defaults.")
        
        return default_config
    
    def _get_version(self) -> str:
        """Get version information."""
        try:
            # Try to get from git
            result = subprocess.run(
                ['git', 'describe', '--tags', '--always'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        # Fallback
        return "unknown"
    
    def run_test_category(self, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test category."""
        logger.info(f"🚀 Running {category} tests...")
        
        category_result = {
            'category': category,
            'status': 'pending',
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'duration_seconds': 0,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'tests_skipped': 0,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        start_time = time.time()
        
        try:
            # Create pytest command
            cmd = ['python', '-m', 'pytest', '-v', '--tb=short']
            
            # Add timeout
            timeout_seconds = config.get('timeout_minutes', 10) * 60
            cmd.extend(['--timeout', str(timeout_seconds)])
            
            # Add coverage if enabled
            if self.config.get('enable_coverage', True):
                cmd.extend(['--cov=scripts', '--cov=utils', '--cov-report=term-missing'])
            
            # Add parallel execution if supported
            if config.get('parallel', False) and category != 'integration_tests':
                cpu_count = min(os.cpu_count() or 1, 4)  # Limit to 4 cores
                cmd.extend(['-n', str(cpu_count)])
            
            # Add test files
            for test_file in config.get('files', []):
                if '::' in test_file:
                    # Specific test class
                    cmd.append(test_file)
                else:
                    # Test file
                    test_path = Path(__file__).parent / test_file
                    cmd.append(str(test_path))
            
            # Add output format
            results_file = self.results_dir / f"{category}_results.xml"
            cmd.extend(['--junitxml', str(results_file)])
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run tests
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 60  # Extra buffer
            )
            
            # Calculate duration
            duration = time.time() - start_time
            category_result['duration_seconds'] = duration
            category_result['end_time'] = datetime.now(timezone.utc).isoformat()
            
            # Parse results
            if process.returncode == 0:
                category_result['status'] = 'passed'
            else:
                category_result['status'] = 'failed'
                category_result['errors'].append(process.stderr)
            
            # Extract test counts from pytest output
            output_lines = process.stdout.split('\n')
            for line in output_lines:
                if 'passed' in line and 'failed' in line:
                    # Parse line like: "5 passed, 2 failed in 10.5s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed,':
                            category_result['tests_passed'] = int(parts[i-1])
                        elif part == 'failed' and 'in' in parts:
                            category_result['tests_failed'] = int(parts[i-1])
                elif 'skipped' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'skipped':
                            category_result['tests_skipped'] = int(parts[i-1])
            
            category_result['tests_run'] = (
                category_result['tests_passed'] + 
                category_result['tests_failed'] + 
                category_result['tests_skipped']
            )
            
            # Parse XML results if available
            if results_file.exists():
                try:
                    tree = ET.parse(results_file)
                    root = tree.getroot()
                    
                    # Get test counts from XML
                    testsuite = root.get('tests')
                    failures = root.get('failures')
                    errors = root.get('errors')
                    skipped = root.get('skipped')
                    
                    if testsuite:
                        category_result['tests_run'] = int(testsuite)
                    if failures:
                        category_result['tests_failed'] = int(failures)
                    if errors:
                        category_result['errors'].append(f"{errors} errors in test execution")
                    if skipped:
                        category_result['tests_skipped'] = int(skipped)
                    
                    # Calculate passed tests
                    category_result['tests_passed'] = (
                        category_result['tests_run'] - 
                        category_result['tests_failed'] - 
                        category_result['tests_skipped']
                    )
                    
                except Exception as e:
                    logger.warning(f"Failed to parse XML results: {e}")
            
            # Add performance metrics
            if 'performance' in category.lower():
                category_result['metrics'] = self._extract_performance_metrics(process.stdout)
            
        except subprocess.TimeoutExpired:
            category_result['status'] = 'timeout'
            category_result['errors'].append(f"Tests timed out after {timeout_seconds} seconds")
            category_result['duration_seconds'] = timeout_seconds
        
        except Exception as e:
            category_result['status'] = 'error'
            category_result['errors'].append(str(e))
            category_result['duration_seconds'] = time.time() - start_time
        
        # Log results
        logger.info(f"✅ {category} completed: {category_result['status']}")
        logger.info(f"   Tests run: {category_result['tests_run']}")
        logger.info(f"   Passed: {category_result['tests_passed']}")
        logger.info(f"   Failed: {category_result['tests_failed']}")
        logger.info(f"   Duration: {category_result['duration_seconds']:.1f}s")
        
        if category_result['errors']:
            logger.warning(f"   Errors: {len(category_result['errors'])}")
        
        return category_result
    
    def _extract_performance_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract performance metrics from test output."""
        metrics = {}
        
        # Look for performance-related lines
        lines = stdout.split('\n')
        for line in lines:
            if '📊' in line:
                # Parse performance metrics
                parts = line.split('📊')[1].strip()
                if ':' in parts:
                    key, value = parts.split(':', 1)
                    metrics[key.strip()] = value.strip()
        
        return metrics
    
    def run_all_tests(self, categories: Optional[List[str]] = None) -> bool:
        """Run all configured test categories."""
        logger.info("🧪 Starting Automated Test Execution")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Determine which categories to run
        if categories is None:
            categories = list(self.config['test_categories'].keys())
        
        # Run tests
        total_passed = 0
        total_failed = 0
        total_tests = 0
        
        for category in categories:
            if category not in self.config['test_categories']:
                logger.warning(f"Unknown test category: {category}")
                continue
            
            category_config = self.config['test_categories'][category]
            if not category_config.get('enabled', True):
                logger.info(f"Skipping disabled category: {category}")
                continue
            
            # Run tests for this category
            result = self.run_test_category(category, category_config)
            self.test_results['test_categories'][category] = result
            
            # Update totals
            total_passed += result['tests_passed']
            total_failed += result['tests_failed']
            total_tests += result['tests_run']
        
        # Calculate overall metrics
        total_duration = time.time() - start_time
        self.test_results['overall_metrics'] = {
            'total_duration_seconds': total_duration,
            'total_tests_run': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_skipped': sum(r['tests_skipped'] for r in self.test_results['test_categories'].values()),
            'success_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'test_coverage': self._calculate_coverage(),
            'performance_score': self._calculate_performance_score()
        }
        
        # Check quality gates
        self.test_results['quality_gates'] = self._check_quality_gates()
        
        # Generate recommendations
        self.test_results['recommendations'] = self._generate_recommendations()
        
        # Final status
        all_passed = all(
            result['status'] == 'passed' for result in self.test_results['test_categories'].values()
        )
        
        self.test_results['final_status'] = 'passed' if all_passed else 'failed'
        
        # Generate reports
        self._generate_reports()
        
        # Print summary
        self._print_summary()
        
        return all_passed
    
    def _calculate_coverage(self) -> float:
        """Calculate test coverage percentage."""
        # Simple heuristic: based on number of test files executed
        total_possible_categories = len(self.config['test_categories'])
        executed_categories = len(self.test_results['test_categories'])
        return (executed_categories / total_possible_categories * 100) if total_possible_categories > 0 else 0
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""
        scores = []
        
        for category, result in self.test_results['test_categories'].items():
            if category.lower().includes('performance'):
                # Extract performance metrics
                metrics = result.get('metrics', {})
                if metrics:
                    # Simple scoring based on metric quality
                    score = 80.0  # Base score
                    if 'memory_mb' in str(metrics):
                        score -= 10  # Penalty for high memory usage
                    scores.append(score)
        
        return statistics.mean(scores) if scores else 100.0
    
    def _check_quality_gates(self) -> Dict[str, Any]:
        """Check if quality gates are met."""
        gates_status = {}
        metrics = self.test_results['overall_metrics']
        
        # Check test coverage
        coverage = metrics.get('test_coverage', 0)
        gates_status['test_coverage'] = {
            'value': coverage,
            'threshold': self.quality_gates['min_test_coverage'],
            'passed': coverage >= self.quality_gates['min_test_coverage']
        }
        
        # Check failure rate
        if metrics['total_tests_run'] > 0:
            failure_rate = (metrics['total_failed'] / metrics['total_tests_run']) * 100
        else:
            failure_rate = 100
        gates_status['failure_rate'] = {
            'value': failure_rate,
            'threshold': self.quality_gates['max_failure_rate'],
            'passed': failure_rate <= self.quality_gates['max_failure_rate']
        }
        
        # Check performance score
        performance_score = metrics.get('performance_score', 0)
        gates_status['performance_score'] = {
            'value': performance_score,
            'threshold': self.quality_gates['min_performance_score'],
            'passed': performance_score >= self.quality_gates['min_performance_score']
        }
        
        # Overall gate status
        all_gates_passed = all(gate['passed'] for gate in gates_status.values())
        gates_status['overall'] = all_gates_passed
        
        return gates_status
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check each quality gate
        gates = self.test_results['quality_gates']
        
        if not gates.get('test_coverage', {}).get('passed', True):
            recommendations.append(
                f"Test coverage ({gates['test_coverage']['value']:.1f}%) is below threshold "
                f"({gates['test_coverage']['threshold']:.1f}%). Consider adding more tests."
            )
        
        if not gates.get('failure_rate', {}).get('passed', True):
            recommendations.append(
                f"Failure rate ({gates['failure_rate']['value']:.1f}%) is above threshold "
                f"({gates['failure_rate']['threshold']:.1f}%). Review and fix failing tests."
            )
        
        if not gates.get('performance_score', {}).get('passed', True):
            recommendations.append(
                f"Performance score ({gates['performance_score']['value']:.1f}) is below threshold "
                f"({gates['performance_score']['threshold']:.1f}). Optimize performance."
            )
        
        # Check for common issues
        failed_categories = [
            cat for cat, result in self.test_results['test_categories'].items()
            if result['status'] != 'passed'
        ]
        
        if failed_categories:
            recommendations.append(
                f"Failed test categories: {', '.join(failed_categories)}. Review and fix these tests."
            )
        
        # Check execution time
        total_duration = self.test_results['overall_metrics']['total_duration_seconds']
        max_duration = self.quality_gates['max_execution_time_minutes'] * 60
        
        if total_duration > max_duration:
            recommendations.append(
                f"Test execution time ({total_duration/60:.1f} minutes) exceeds "
                f"threshold ({self.quality_gates['max_execution_time_minutes']} minutes). "
                "Consider parallelizing tests or optimizing test performance."
            )
        
        return recommendations
    
    def _generate_reports(self):
        """Generate test reports in various formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON report
        json_path = self.results_dir / f"test_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # XML report for CI/CD
        xml_path = self.results_dir / f"test_report_{timestamp}.xml"
        self._generate_xml_report(xml_path)
        
        # HTML report
        html_path = self.results_dir / f"test_report_{timestamp}.html"
        self._generate_html_report(html_path)
        
        logger.info(f"📊 Reports generated:")
        logger.info(f"   JSON: {json_path}")
        logger.info(f"   XML: {xml_path}")
        logger.info(f"   HTML: {html_path}")
    
    def _generate_xml_report(self, output_path: Path):
        """Generate XML report for CI/CD integration."""
        root = ET.Element('testsuite')
        root.set('name', 'Medical AI Training System Tests')
        
        total_tests = self.test_results['overall_metrics']['total_tests_run']
        total_failures = self.test_results['overall_metrics']['total_failed']
        total_errors = 0  # We treat errors as failures
        total_skipped = self.test_results['overall_metrics']['total_skipped']
        total_time = self.test_results['overall_metrics']['total_duration_seconds']
        
        root.set('tests', str(total_tests))
        root.set('failures', str(total_failures))
        root.set('errors', str(total_errors))
        root.set('skipped', str(total_skipped))
        root.set('time', str(total_time))
        
        # Add test cases
        for category, result in self.test_results['test_categories'].items():
            testcase = ET.SubElement(root, 'testcase')
            testcase.set('classname', category)
            testcase.set('name', f"{category}_test")
            testcase.set('time', str(result['duration_seconds']))
            
            if result['status'] != 'passed':
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', f"Test category {category} failed")
                failure.text = '\n'.join(result.get('errors', []))
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _generate_html_report(self, output_path: Path):
        """Generate HTML report for human readability."""
        html_content = self._create_html_template()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _create_html_template(self) -> str:
        """Create HTML report template."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Generate CSS for the report
        css_styles = """
        <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { display: flex; justify-content: space-around; margin: 20px 0; }
        .metric { text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; }
        .metric-label { font-size: 12px; color: #666; }
        .category { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .status-passed { border-left: 5px solid #4CAF50; }
        .status-failed { border-left: 5px solid #f44336; }
        .status-error { border-left: 5px solid #ff9800; }
        .recommendations { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        </style>
        """
        
        # Generate HTML content
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Medical AI Training System - Test Report</title>
            {css_styles}
        </head>
        <body>
            <div class="header">
                <h1>Medical AI Training System - Test Report</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Version:</strong> {self.test_results.get('version', 'unknown')}</p>
                <p><strong>Overall Status:</strong> 
                   <span style="color: {'green' if self.test_results['final_status'] == 'passed' else 'red'}">
                       {self.test_results['final_status'].upper()}
                   </span>
                </p>
            </div>
            
            <div class="summary">
                <div class="metric">
                    <div class="metric-value">{self.test_results['overall_metrics']['total_tests_run']}</div>
                    <div class="metric-label">Total Tests</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.test_results['overall_metrics']['total_passed']}</div>
                    <div class="metric-label">Passed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.test_results['overall_metrics']['total_failed']}</div>
                    <div class="metric-label">Failed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.test_results['overall_metrics']['success_rate']:.1f}%</div>
                    <div class="metric-label">Success Rate</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{self.test_results['overall_metrics']['total_duration_seconds']/60:.1f}m</div>
                    <div class="metric-label">Duration</div>
                </div>
            </div>
        """
        
        # Add quality gates section
        html += "<h2>Quality Gates</h2><table><tr><th>Gate</th><th>Value</th><th>Threshold</th><th>Status</th></tr>"
        for gate, data in self.test_results['quality_gates'].items():
            if gate != 'overall':
                status_color = 'green' if data['passed'] else 'red'
                status_text = 'PASS' if data['passed'] else 'FAIL'
                html += f"""
                <tr>
                    <td>{gate.replace('_', ' ').title()}</td>
                    <td>{data['value']:.1f}</td>
                    <td>{data['threshold']:.1f}</td>
                    <td style="color: {status_color}; font-weight: bold;">{status_text}</td>
                </tr>
                """
        html += "</table>"
        
        # Add test categories section
        html += "<h2>Test Categories</h2>"
        for category, result in self.test_results['test_categories'].items():
            status_class = f"status-{result['status']}"
            status_color = 'green' if result['status'] == 'passed' else 'red'
            
            html += f"""
            <div class="category {status_class}">
                <h3 style="color: {status_color};">{category.replace('_', ' ').title()}</h3>
                <p><strong>Status:</strong> {result['status'].upper()}</p>
                <p><strong>Tests Run:</strong> {result['tests_run']} | 
                   <strong>Passed:</strong> {result['tests_passed']} | 
                   <strong>Failed:</strong> {result['tests_failed']} | 
                   <strong>Duration:</strong> {result['duration_seconds']:.1f}s</p>
            """
            
            if result['errors']:
                html += "<p><strong>Errors:</strong></p><ul>"
                for error in result['errors']:
                    html += f"<li>{error}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        # Add recommendations section
        if self.test_results['recommendations']:
            html += "<div class='recommendations'><h2>Recommendations</h2><ul>"
            for recommendation in self.test_results['recommendations']:
                html += f"<li>{recommendation}</li>"
            html += "</ul></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def _print_summary(self):
        """Print test execution summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TEST EXECUTION SUMMARY")
        logger.info("=" * 60)
        
        metrics = self.test_results['overall_metrics']
        logger.info(f"Total Tests Run: {metrics['total_tests_run']}")
        logger.info(f"Tests Passed: {metrics['total_passed']}")
        logger.info(f"Tests Failed: {metrics['total_failed']}")
        logger.info(f"Tests Skipped: {metrics['total_skipped']}")
        logger.info(f"Success Rate: {metrics['success_rate']:.1f}%")
        logger.info(f"Test Coverage: {metrics['test_coverage']:.1f}%")
        logger.info(f"Performance Score: {metrics['performance_score']:.1f}")
        logger.info(f"Total Duration: {metrics['total_duration_seconds']/60:.1f} minutes")
        
        logger.info("\nQuality Gates:")
        gates = self.test_results['quality_gates']
        for gate, data in gates.items():
            if gate != 'overall':
                status = "✅ PASS" if data['passed'] else "❌ FAIL"
                logger.info(f"  {gate.replace('_', ' ').title()}: {status} "
                          f"({data['value']:.1f} vs {data['threshold']:.1f})")
        
        logger.info(f"\nOverall Quality Gates: {'✅ PASSED' if gates.get('overall', False) else '❌ FAILED'}")
        
        if self.test_results['recommendations']:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(self.test_results['recommendations'], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("\n" + "=" * 60)
        
        if self.test_results['final_status'] == 'passed':
            logger.info("🎉 ALL TESTS PASSED!")
        else:
            logger.error("❌ SOME TESTS FAILED!")
        
        logger.info("=" * 60)

# ==================== CI/CD INTEGRATION ====================

def create_ci_config():
    """Create CI/CD configuration files."""
    
    # GitHub Actions workflow
    github_workflow = """name: Automated Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install pytest pytest-cov pytest-html
        pip install psutil pyyaml
    
    - name: Run automated tests
      run: |
        python scripts/run_all_tests.py --categories unit_tests data_quality_tests
        python scripts/run_all_tests.py --categories integration_tests performance_tests
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test_results/
    
    - name: Publish test results
      uses: dorny/test-reporter@v1
      if: success() || failure()
      with:
        name: Test Results
        path: test_results/*_results.xml
        reporter: java-junit
"""
    
    with open('.github/workflows/tests.yml', 'w') as f:
        f.write(github_workflow)
    
    logger.info("✅ Created GitHub Actions workflow configuration")
    
    # Jenkinsfile
    jenkinsfile = """
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                script {
                    sh 'python -m pip install --upgrade pip'
                    sh 'pip install -r requirements.txt'
                    sh 'pip install -r requirements-dev.txt'
                    sh 'pip install pytest pytest-cov pytest-html psutil pyyaml'
                }
            }
        }
        
        stage('Unit Tests') {
            steps {
                sh 'python scripts/run_all_tests.py --categories unit_tests data_quality_tests'
            }
        }
        
        stage('Integration Tests') {
            steps {
                sh 'python scripts/run_all_tests.py --categories integration_tests performance_tests'
            }
        }
        
        stage('Performance Tests') {
            steps {
                sh 'python scripts/run_all_tests.py --categories stress_tests'
            }
        }
    }
    
    post {
        always {
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'test_results',
                reportFiles: '*.html',
                reportName: 'Test Report'
            ])
        }
        
        failure {
            emailext (
                subject: "Test Build Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "The test build failed. Please check the console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
"""
    
    with open('Jenkinsfile', 'w') as f:
        f.write(jenkinsfile)
    
    logger.info("✅ Created Jenkins pipeline configuration")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run automated tests for Medical AI Training System')
    parser.add_argument('--categories', nargs='+', help='Specific test categories to run')
    parser.add_argument('--config', default='test_config.yaml', help='Test configuration file')
    parser.add_argument('--output-dir', default='test_results', help='Output directory for results')
    parser.add_argument('--ci-setup', action='store_true', help='Setup CI/CD configuration files')
    parser.add_argument('--quality-gates-only', action='store_true', help='Check quality gates only')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel where possible')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.ci_setup:
        create_ci_config()
        return
    
    # Create test runner
    runner = TestRunner(args.config)
    
    # Set parallel execution
    runner.config['parallel'] = args.parallel
    
    # Run tests
    try:
        success = runner.run_all_tests(args.categories)
        
        if args.quality_gates_only:
            gates_passed = runner.test_results['quality_gates'].get('overall', False)
            logger.info(f"Quality Gates Status: {'PASSED' if gates_passed else 'FAILED'}")
            sys.exit(0 if gates_passed else 1)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()