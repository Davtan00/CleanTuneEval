"""Main test runner script."""
import sys
import pytest
from pathlib import Path

# TODO: Take time to actually create a proper up to date test suite, 
# All tests were done with LLM, so they are not reliable and only function as a temporary solution
if __name__ == "__main__":
    # Add project root to Python path
    sys.path.append(str(Path(__file__).parent))
    
    # Run pytest with our default configuration
    sys.exit(pytest.main()) 