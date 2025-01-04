"""Utility functions for data management."""
from datetime import datetime
from typing import Optional

def generate_dataset_id(
    domain: str,
    data_size: int,
    custom_tag: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """Generate a unique dataset ID.
    
    Args:
        domain: Domain of the dataset (e.g., 'technology', 'movie')
        data_size: Number of items in the dataset
        custom_tag: Optional custom identifier
        timestamp: Optional timestamp (if not provided, current time will be used)
    
    Returns:
        A unique dataset ID in the format: domain_size_timestamp[_tag]
        Example: technology_10k_20250104_144548_v1
    """
    # Format the size (e.g., 1000 -> 1k, 1000000 -> 1m)
    if data_size >= 1_000_000:
        size_str = f"{data_size//1_000_000}m"
    elif data_size >= 1_000:
        size_str = f"{data_size//1_000}k"
    else:
        size_str = str(data_size)
    
    # Use provided timestamp or generate new one
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build the ID
    dataset_id = f"{domain}_{size_str}_{ts}"
    if custom_tag:
        dataset_id = f"{dataset_id}_{custom_tag}"
    
    return dataset_id
