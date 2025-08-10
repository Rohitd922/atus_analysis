#!/usr/bin/env python
"""
Test script for run_single_experiments.py

This script performs basic checks to ensure the run_single_experiments.py
script can run safely without causing computational hazards.
"""

import sys
import subprocess
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available."""
    print("Testing dependencies...")
    
    try:
        import psutil
        print("✓ psutil is available")
        print(f"  Current CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
        print(f"  Current memory usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"  Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    except ImportError:
        print("✗ psutil is not available - please install it")
        return False
    
    try:
        import argparse, json, logging, subprocess, time
        from datetime import datetime
        from pathlib import Path
        from typing import Dict, List, Optional, Tuple
        print("✓ All standard library imports available")
    except ImportError as e:
        print(f"✗ Missing standard library import: {e}")
        return False
    
    return True

def test_script_syntax():
    """Test if the script has valid Python syntax."""
    print("\nTesting script syntax...")
    
    script_path = Path(__file__).parent / "run_single_experiments.py"
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "py_compile", str(script_path)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Script has valid Python syntax")
            return True
        else:
            print(f"✗ Syntax error in script: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error testing syntax: {e}")
        return False

def test_help_output():
    """Test if the script can display help without errors."""
    print("\nTesting help output...")
    
    script_path = Path(__file__).parent / "run_single_experiments.py"
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ Script help displays correctly")
            return True
        else:
            print(f"✗ Help output failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ Script help timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing help: {e}")
        return False

def test_resource_monitoring():
    """Test the resource monitoring functionality."""
    print("\nTesting resource monitoring...")
    
    try:
        # Import the ResourceMonitor class
        sys.path.insert(0, str(Path(__file__).parent))
        from run_single_experiments import ResourceMonitor
        
        monitor = ResourceMonitor(max_memory_percent=75.0, max_cpu_percent=85.0)
        status = monitor.get_resource_status()
        
        expected_keys = ['timestamp', 'memory_percent', 'memory_available_gb', 
                        'memory_used_gb', 'cpu_percent', 'memory_ok', 'cpu_ok', 'system_ok']
        
        if all(key in status for key in expected_keys):
            print("✓ Resource monitoring works correctly")
            print(f"  System status: {'OK' if status['system_ok'] else 'HIGH USAGE'}")
            return True
        else:
            print("✗ Resource monitoring missing expected keys")
            return False
    except Exception as e:
        print(f"✗ Resource monitoring test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("TESTING run_single_experiments.py")
    print("="*60)
    
    tests = [
        test_dependencies,
        test_script_syntax,
        test_help_output,
        test_resource_monitoring
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ run_single_experiments.py is ready to use safely")
        print("\nUsage examples:")
        print("  # Interactive mode (safest):")
        print("  python atus_analysis/scripts/run_single_experiments.py")
        print("\n  # Automated routing ladder:")
        print("  python atus_analysis/scripts/run_single_experiments.py --mode routing --delay 120")
        print("\n  # Single rung test:")
        print("  python atus_analysis/scripts/run_single_experiments.py --mode single --rung R1")
    else:
        print("✗ Some tests failed - please fix issues before running experiments")
        return 1
    
    print("="*60)
    return 0

if __name__ == "__main__":
    exit(main())
