#!/usr/bin/env python3
"""Main test runner script."""
import sys
import pytest
from pathlib import Path

if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(str(Path(__file__).parent))
    
    # Run pytest with our default configuration
    sys.exit(pytest.main()) 