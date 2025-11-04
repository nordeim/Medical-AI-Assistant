"""
Main test runner and configuration for comprehensive medical AI testing.

This module provides the central configuration and runner for all test suites
including unit tests, integration tests, load tests, security tests, and compliance tests.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import json
from datetime import datetime
import subprocess


class MedicalAITestRunner:
    """Main test runner for medical AI system testing."""
    
    def __init__(self):
        self.test_directories = {
            "unit": "tests/unit",
            "integration": "tests/integration", 
            "load": "tests/load",
            "e2e": "tests/e2e",
            "security": "tests/security",
            "compliance": "tests/compliance"
        }
        
        self.test_categories = {
            "fast": ["unit"],
            "medium": ["integration", "security"],
            "slow": ["load", "e2e", "compliance"],
            "critical": ["security", "compliance"]
        }
        
        self.default_pytest_args = [
            "--tb=short",
            "--strict-markers",
            "--strict-config",
            "--disable-warnings",
            "-ra"
        ]
    
    def run_all_tests(self, test_level: str = "all", parallel: bool = False) -> Dict[str, Any]:
        """Run all tests or specific test categories."""
        
        test_results = {
            "start_time": datetime.now().isoformat(),
            "test_level": test_level,
            "results": {},
            "summary": {}
        }
        
        if test_level == "all":
            categories = list(self.test_directories.keys())
        else:
            categories = [test_level]
        
        for category in categories:
            print(f"\n{'='*60}")
            print(f"Running {category.upper()} tests...")
            print(f"{'='*60}")
            
            try:
                result = self._run_test_category(category, parallel)
                test_results["results"][category] = result
                
                # Print category summary
                self._print_category_summary(category, result)
                
            except Exception as e:
                print(f"ERROR running {category} tests: {str(e)}")
                test_results["results"][category] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Calculate overall summary
        test_results["summary"] = self._calculate_summary(test_results["results"])
        test_results["end_time"] = datetime.now().isoformat()
        
        self._print_final_summary(test_results["summary"])
        
        return test_results
    
    def _run_test_category(self, category: str, parallel: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category."""
        
        test_dir = self.test_directories[category]
        test_path = Path(test_dir)
        
        if not test_path.exists():
            return {
                "status": "skipped",
                "reason": f"Test directory {test_dir} not found"
            }
        
        # Build pytest arguments
        pytest_args = self.default_pytest_args.copy()
        pytest_args.extend([
            str(test_path),
            f"--junitxml=reports/{category}_results.xml",
            f"--html=reports/{category}_report.html",
            "--self-contained-html"
        ])
        
        # Add category-specific markers
        if category == "unit":
            pytest_args.extend(["-m", "unit or not slow"])
        elif category == "integration":
            pytest_args.extend(["-m", "integration"])
        elif category == "load":
            pytest_args.extend(["-m", "load"])
        elif category == "e2e":
            pytest_args.extend(["-m", "e2e"])
        elif category == "security":
            pytest_args.extend(["-m", "security"])
        elif category == "compliance":
            pytest_args.extend(["-m", "compliance"])
        
        # Add parallel execution if requested
        if parallel:
            pytest_args.extend(["-n", "auto"])
        
        # Run pytest
        try:
            result = pytest.main(pytest_args)
            
            return {
                "status": "completed",
                "exit_code": result,
                "passed": result == 0,
                "test_path": str(test_path)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_quick_tests(self) -> Dict[str, Any]:
        """Run only fast tests for quick validation."""
        
        return self.run_all_tests(test_level="fast")
    
    def run_critical_tests(self) -> Dict[str, Any]:
        """Run only critical tests (security and compliance)."""
        
        return self.run_all_tests(test_level="critical")
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and load tests."""
        
        return self.run_all_tests(test_level="load")
    
    def _print_category_summary(self, category: str, result: Dict[str, Any]):
        """Print summary for a test category."""
        
        if result["status"] == "completed":
            status_icon = "✅" if result["passed"] else "❌"
            print(f"{status_icon} {category.upper()} tests: {'PASSED' if result['passed'] else 'FAILED'}")
        elif result["status"] == "skipped":
            print(f"⏭️ {category.upper()} tests: SKIPPED")
        else:
            print(f"❌ {category.upper()} tests: ERROR")
    
    def _calculate_summary(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall test summary."""
        
        total_categories = len(results)
        passed_categories = sum(1 for r in results.values() if r.get("passed", False))
        failed_categories = sum(1 for r in results.values() if r.get("passed", False) is False)
        error_categories = sum(1 for r in results.values() if r.get("status") == "error")
        skipped_categories = sum(1 for r in results.values() if r.get("status") == "skipped")
        
        overall_success = failed_categories == 0 and error_categories == 0
        
        return {
            "total_categories": total_categories,
            "passed_categories": passed_categories,
            "failed_categories": failed_categories,
            "error_categories": error_categories,
            "skipped_categories": skipped_categories,
            "overall_success": overall_success,
            "success_rate": (passed_categories / max(total_categories - skipped_categories, 1)) * 100
        }
    
    def _print_final_summary(self, summary: Dict[str, Any]):
        """Print final test summary."""
        
        print(f"\n{'='*60}")
        print("FINAL TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total test categories: {summary['total_categories']}")
        print(f"Passed: {summary['passed_categories']}")
        print(f"Failed: {summary['failed_categories']}")
        print(f"Errors: {summary['error_categories']}")
        print(f"Skipped: {summary['skipped_categories']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        
        if summary['overall_success']:
            print("\n🎉 ALL TESTS PASSED - System ready for deployment!")
        else:
            print("\n⚠️  SOME TESTS FAILED - Review results before deployment")
        
        print(f"{'='*60}\n")
    
    def generate_test_report(self, results: Dict[str, Any], output_path: str = "reports/test_report.json"):
        """Generate comprehensive test report."""
        
        # Ensure reports directory exists
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Test report generated: {output_path}")


def create_reports_directory():
    """Create reports directory if it doesn't exist."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for different report types
    (reports_dir / "unit").mkdir(exist_ok=True)
    (reports_dir / "integration").mkdir(exist_ok=True)
    (reports_dir / "load").mkdir(exist_ok=True)
    (reports_dir / "e2e").mkdir(exist_ok=True)
    (reports_dir / "security").mkdir(exist_ok=True)
    (reports_dir / "compliance").mkdir(exist_ok=True)


def setup_test_environment():
    """Setup test environment and dependencies."""
    
    print("Setting up test environment...")
    
    # Create reports directory
    create_reports_directory()
    
    # Check if required packages are installed
    required_packages = [
        "pytest", "pytest-asyncio", "pytest-html", "pytest-xdist",
        "httpx", "psutil", "numpy", "cryptography"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"WARNING: Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
    else:
        print("✅ All required test packages are installed")
    
    print("Test environment setup complete!\n")


def main():
    """Main function to run medical AI tests."""
    
    print("="*60)
    print("MEDICAL AI TESTING FRAMEWORK")
    print("="*60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run Medical AI Tests")
    parser.add_argument(
        "--level", 
        choices=["all", "fast", "critical", "performance", "unit", "integration", "load", "e2e", "security", "compliance"],
        default="fast",
        help="Test level to run"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup test environment, don't run tests"
    )
    parser.add_argument(
        "--report-path",
        default="reports/test_report.json",
        help="Path for test report output"
    )
    
    args = parser.parse_args()
    
    # Setup test environment
    setup_test_environment()
    
    if args.setup_only:
        print("Test environment setup complete. Exiting.")
        return
    
    # Initialize test runner
    runner = MedicalAITestRunner()
    
    # Run tests based on level
    print(f"Running {args.level.upper()} tests...")
    start_time = datetime.now()
    
    try:
        if args.level == "fast":
            results = runner.run_quick_tests()
        elif args.level == "critical":
            results = runner.run_critical_tests()
        elif args.level == "performance":
            results = runner.run_performance_tests()
        else:
            results = runner.run_all_tests(test_level=args.level, parallel=args.parallel)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nTotal execution time: {duration:.2f} seconds")
        
        # Generate report
        runner.generate_test_report(results, args.report_path)
        
        # Exit with appropriate code
        if results["summary"]["overall_success"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error running tests: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()