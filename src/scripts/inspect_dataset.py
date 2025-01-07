import sys
from pathlib import Path
from datasets import load_from_disk

def inspect_dataset(dataset_path: str):
    """
    Prints out available splits and column names from a Hugging Face dataset stored on disk.
    Optionally prints sample data for each split to see the structure of the fields.
    """
    # Convert string path to Path object
    dataset_dir = Path(dataset_path).resolve()
    
    if not dataset_dir.exists():
        print(f"Error: The dataset path '{dataset_dir}' does not exist.")
        return

    print(f"Loading dataset from: {dataset_dir}")
    dataset = load_from_disk(str(dataset_dir))

    # List the splits and column names
    print("\n=== Dataset Splits and Columns ===")
    for split_name, split_data in dataset.items():
        print(f"\nSplit Name: {split_name}")
        print("Column Names:", split_data.column_names)
        
        # Uncomment if you also want to see a sample record
        # (Make sure each split has at least one record):
        print("Sample record:", split_data[0])
    
    print("\nInspection complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_dataset.py <dataset_path>")
        sys.exit(1)
    
    ds_path = sys.argv[1]
    inspect_dataset(ds_path)

#python3 src/scripts/inspect_dataset.py src/data/datasets/technology_18k_20250106_095813