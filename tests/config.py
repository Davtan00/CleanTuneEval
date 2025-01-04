"""Test configuration and utilities."""
import shutil
from pathlib import Path

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_STORAGE_DIR = TEST_DATA_DIR / "storage"
TEST_DATASETS_DIR = TEST_DATA_DIR / "datasets"

def setup_test_environment():
    """Create test directories if they don't exist."""
    TEST_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_test_environment():
    """Remove all test data after tests complete."""
    if TEST_DATA_DIR.exists():
        shutil.rmtree(TEST_DATA_DIR)
