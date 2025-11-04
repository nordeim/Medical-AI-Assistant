#!/usr/bin/env python3
"""
Financial Optimization Framework - Quick Test Script
Simple test to verify framework functionality
"""

import asyncio
import sys
from pathlib import Path

# Add the framework to the path
sys.path.append(str(Path(__file__).parent))

from financial_optimization_orchestrator import (
    FinancialOptimizationOrchestrator,
    FinancialConfig,
    create_default_config
)

async def quick_test():
    """Quick functionality test"""
    print("🚀 Financial Optimization Framework - Quick Test")
    print("=" * 60)
    
    # Test 1: Configuration
    print("\n1️⃣ Testing Configuration...")
    config = create_default_config()
    print(f"   ✅ Default config created: {config.model_complexity}")
    
    # Test 2: Framework Initialization
    print("\n2️⃣ Testing Framework Initialization...")
    orchestrator = FinancialOptimizationOrchestrator(config)
    init_result = await orchestrator.initialize_framework()
    
    if init_result['status'] == 'success':
        print("   ✅ Framework initialized successfully")
        print(f"   - Components loaded: {len(init_result['components'])}")
    else:
        print(f"   ❌ Initialization failed: {init_result.get('error')}")
        return False
    
    # Test 3: Sample Data Processing
    print("\n3️⃣ Testing Sample Data Processing...")
    sample_data = {
        'revenue': 1000000,
        'costs': 700000,
        'assets': 2000000,
        'liabilities': 800000,
        'equity': 1200000,
        'cash_flow': 200000
    }
    
    # Test financial modeling
    modeling_result = await orchestrator.financial_models.analyze_financial_data(
        sample_data, ['optimize_costs']
    )
    
    if modeling_result['status'] == 'success':
        print("   ✅ Financial modeling completed")
        metrics = modeling_result.get('metrics', {})
        print(f"   - Revenue: ${metrics.get('revenue', 0):,.0f}")
        print(f"   - Profit Margin: {metrics.get('profit_margin', 0):.1%}")
    else:
        print(f"   ❌ Financial modeling failed: {modeling_result.get('error')}")
        return False
    
    # Test 4: Framework Status
    print("\n4️⃣ Testing Framework Status...")
    status = await orchestrator.get_framework_status()
    if status.get('is_running', False):
        print("   ✅ Framework is running and operational")
        print(f"   - Component count: {len(status.get('component_status', {}))}")
    else:
        print("   ❌ Framework status check failed")
        return False
    
    # Test 5: Shutdown
    print("\n5️⃣ Testing Framework Shutdown...")
    shutdown_result = await orchestrator.shutdown_framework()
    
    if shutdown_result['status'] == 'success':
        print("   ✅ Framework shutdown completed successfully")
    else:
        print(f"   ❌ Shutdown failed: {shutdown_result.get('error')}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED - FRAMEWORK IS READY!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(quick_test())
        if success:
            print("\n✅ Framework is ready for use!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        sys.exit(1)