"""
Test Runner - Run all unit tests
"""

import unittest
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

if __name__ == '__main__':
    # Discover and run all tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())
