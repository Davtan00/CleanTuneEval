"""Utility functions for data management."""
from datetime import datetime
from typing import Optional

def generate_dataset_id(
    domain: str,
    data_size: int,
    custom_tag: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Generates a unique dataset ID string, e.g. "movie_10k_20250104_144548_synthetic"
      - domain: domain name (e.g. "movie", "technology")
      - data_size: final number of reviews after filtering
      - custom_tag: optional extra identifier
      - timestamp: specify a custom timestamp if needed, else we use current time
    Size gets represented as:
      >= 1_000_000 -> "Xm"
      >= 1_000 -> "Xk"
      < 1_000 -> actual integer
    """
    # Convert data_size to a string format
    if data_size >= 1_000_000:
        size_str = f"{data_size // 1_000_000}m"
    elif data_size >= 1_000:
        size_str = f"{data_size // 1_000}k"
    else:
        size_str = str(data_size)

    # If no timestamp is provided, use the current date/time
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct final ID
    dataset_id = f"{domain}_{size_str}_{ts}"
    if custom_tag:
        dataset_id = f"{dataset_id}_{custom_tag}"

    return dataset_id
