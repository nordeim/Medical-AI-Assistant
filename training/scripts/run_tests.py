#!/usr/bin/env python3
"""Test runner for data validation utilities."""

import os
import sys
import unittest
import argparse
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import test modules
from training.tests.test_data_validation import (
    TestValidationConfig,
    TestDataValidator,
    TestMedicalDataValidator,
    TestValidationReporter,
    TestBatchValidationReporter,
    TestIntegration,
    TestEdgeCases
)


def run_tests(test_pattern=None, verbosity=2, generate_coverage=False):
    """Run the test suite."""
    
    # Discover and load tests
    loader = unittest.TestLoader()
    
    if test_pattern:
        # Run specific test class or method
        if ':' in test_pattern:
            class_name, method_name = test_pattern.split(':', 1)
            test_class = globals().get(class_name)
            if test_class:
                suite = loader.loadTestsFromName(test_pattern, test_class)
            else:
                print(f"Test class '{class_name}' not found")
                return False
        else:
            # Run specific test class
            suite = loader.loadTestsFromName(test_pattern, globals())
    else:
        # Load all test classes
        test_classes = [
            TestValidationConfig,
            TestDataValidator,
            TestMedicalDataValidator,
            TestValidationReporter,
            TestBatchValidationReporter,
            TestIntegration,
            TestEdgeCases
        ]
        
        suite = unittest.TestSuite()
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    start_time = datetime.now()
    result = runner.run(suite)
    end_time = datetime.now()
    
    # Print summary
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        
        # Show failure details
        if result.failures:
            print(f"\n{len(result.failures)} FAILURES:")
            for i, (test, traceback) in enumerate(result.failures, 1):
                print(f"\n{i}. {test}:")
                print(traceback)
        
        if result.errors:
            print(f"\n{len(result.errors)} ERRORS:")
            for i, (test, traceback) in enumerate(result.errors, 1):
                print(f"\n{i}. {test}:")
                print(traceback)
        
        return False


def run_coverage_analysis():
    """Run tests with coverage analysis (if coverage is available)."""
    try:
        import coverage
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Run tests
        success = run_tests(verbosity=1)
        
        # Stop coverage and generate report
        cov.stop()
        cov.save()
        
        print(f"\n{'='*70}")
        print("COVERAGE ANALYSIS")
        print(f"{'='*70}")
        cov.report(show_missing=True)
        
        # Generate HTML coverage report
        cov.html_report(directory='coverage_html_report')
        print(f"\nHTML coverage report generated in 'coverage_html_report' directory")
        
        return success
        
    except ImportError:
        print("Coverage package not available. Install with: pip install coverage")
        return run_tests()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for data validation utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py
  
  # Run specific test class
  python run_tests.py --pattern TestDataValidator
  
  # Run specific test method
  python run_tests.py --pattern TestDataValidator:test_validate_dataset_valid
  
  # Run with coverage analysis
  python run_tests.py --coverage
  
  # Quiet mode (less verbose output)
  python run_tests.py --quiet
        """
    )
    
    parser.add_argument('--pattern', '-p',
                       help='Test pattern (class name or class:method)')
    parser.add_argument('--coverage', '-c',
                       action='store_true',
                       help='Run with coverage analysis')
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Reduce verbosity')
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Increase verbosity')
    
    args = parser.parse_args()
    
    # Determine verbosity level
    if args.quiet:
        verbosity = 1
    elif args.verbose:
        verbosity = 3
    else:
        verbosity = 2
    
    # Run tests
    try:
        if args.coverage:
            success = run_coverage_analysis()
        else:
            success = run_tests(test_pattern=args.pattern, verbosity=verbosity)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error during test execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()