#!/usr/bin/env python3
"""
Simple script to run the CLI topics test suite.

This script demonstrates how to run the comprehensive test suite
that covers all the examples from the CLI epilog.
"""

import os
import sys

# Add the parent directory to the path so we can import the test module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import test functions
from test.test import run_all_tests

if __name__ == "__main__":
    print("Starting LLM Topic Modeller Test Suite...")
    print("This will test:")
    print("- CLI examples from the epilog")
    print("- GUI application functionality")
    print("Using a mock inference-server to avoid API costs.")
    print("=" * 60)

    success = run_all_tests()

    if success:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)
