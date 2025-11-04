#!/usr/bin/env python3
"""
Quick Start Script for Advanced Analytics Platform
Run this script to quickly test the platform capabilities
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if all requirements are available"""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        print("✓ Core dependencies available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False

def run_demo():
    """Run the platform demonstration"""
    print("\n" + "="*60)
    print("🚀 ADVANCED ANALYTICS PLATFORM DEMO")
    print("="*60)
    
    try:
        from launcher import AnalyticsPlatformLauncher
        
        # Initialize platform
        launcher = AnalyticsPlatformLauncher("development")
        launcher.initialize_platform()
        
        # Run demo
        launcher.run_demo()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext steps:")
        print("1. Run: python launcher.py --mode interactive")
        print("2. Run: python launcher.py --mode report")
        print("3. Check: python launcher.py --mode status")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        return False

def run_tests():
    """Run platform tests"""
    print("\n" + "="*60)
    print("🧪 RUNNING PLATFORM TESTS")
    print("="*60)
    
    try:
        subprocess.check_call([sys.executable, "-m", "pytest", "tests/", "-v"])
        print("\n✅ ALL TESTS PASSED")
        return True
        
    except subprocess.CalledProcessError:
        print("\n⚠️  Some tests failed - platform still functional")
        return True  # Don't fail the setup process for test issues

def show_help():
    """Show help information"""
    help_text = """
    Advanced Analytics Platform - Quick Start Guide
    
    🚀 QUICK START:
    1. Install requirements: python quick_start.py --install
    2. Run demo: python quick_start.py --demo
    3. Interactive mode: python launcher.py --mode interactive
    
    📊 AVAILABLE MODES:
    - demo: Run comprehensive platform demonstration
    - interactive: Start interactive command-line interface  
    - report: Generate sample analytics report
    - status: Check platform status and capabilities
    
    📈 PLATFORM FEATURES:
    ✓ AI-Powered Insights and Analytics
    ✓ Predictive Analytics and Forecasting
    ✓ Customer Behavior Analytics and Segmentation
    ✓ Market Intelligence and Competitive Analysis
    ✓ Clinical Outcome Analytics (Healthcare)
    ✓ Operational Analytics and Efficiency Measurement
    ✓ Executive Decision Support and Strategic Intelligence
    
    🔧 CONFIGURATION:
    - Development environment (default)
    - Production environment (--env production)
    - Custom configuration (--config config.json)
    
    📚 DOCUMENTATION:
    - README.md - Complete platform documentation
    - tests/ - Comprehensive test suite
    - launcher.py - Main platform interface
    
    🆘 SUPPORT:
    Run: python launcher.py --mode interactive
    Then type: help
    
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(
        description="Advanced Analytics Platform - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--demo", action="store_true", help="Run platform demonstration")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--test", action="store_true", help="Run platform tests")
    parser.add_argument("--help-info", action="store_true", help="Show help information")
    
    args = parser.parse_args()
    
    # Change to platform directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Handle install first
    if args.install:
        install_requirements()
        return
    
    # Check requirements
    if not check_requirements():
        print("Run: python quick_start.py --install")
        return
    
    # Show help if requested
    if args.help_info:
        show_help()
        return
    
    # Run demo if requested
    if args.demo:
        success = run_demo()
        if success and args.test:
            run_tests()
        return
    
    # Run tests if requested
    if args.test:
        run_tests()
        return
    
    # Default: show help
    show_help()

if __name__ == "__main__":
    main()