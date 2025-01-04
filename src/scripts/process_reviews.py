import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.pipeline import DataPipeline
from src.config.logging_config import setup_logging
from src.config.environment import HardwareConfig

logger = setup_logging()

def format_filtering_stats(stats: Dict[str, Any]) -> str:
    """Format filtering statistics with percentages."""
    total = stats.get('total_processed', 1)  # Avoid division by zero
    return (
        f"Length filtered: {stats['length_filtered']} ({stats['length_filtered']/total*100:.1f}%)\n"
        f"Duplicates removed: {stats['duplicates_removed']} ({stats['duplicates_removed']/total*100:.1f}%)\n"
        f"  - Exact duplicates: {stats.get('exact_duplicates', 0)} ({stats.get('exact_duplicates', 0)/total*100:.1f}%)\n"
        f"  - Near duplicates: {stats.get('near_duplicates', 0)} ({stats.get('near_duplicates', 0)/total*100:.1f}%)\n"
        f"Total removed: {stats['total_removed']} ({stats['total_removed']/total*100:.1f}%)"
    )

def main():
    parser = argparse.ArgumentParser(description='Process review data through the cleaning pipeline.')
    parser.add_argument('input_file', help='Path to input JSON file containing reviews')
    parser.add_argument('--tag', help='Optional custom tag for the dataset', default=None)
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Batch size for processing (default: 1000)')
    parser.add_argument('--use-mps', action='store_true', 
                       help='Use MPS acceleration on Apple Silicon')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        # Configure hardware settings
        hw_config = HardwareConfig(
            device='mps' if args.use_mps else 'cpu',
            n_cores=14,  # M4 Pro core count
            memory_limit=48,  # 48GB RAM
            use_mps=args.use_mps
        )
        
        # Load input data with validation
        with open(args.input_file) as f:
            data = json.load(f)
            
        # Validate input structure
        required_keys = {'domain', 'generated_data', 'summary'}
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            logger.error(f"Missing required keys in input data: {missing}")
            sys.exit(1)
            
        if not data['generated_data']:
            logger.error("Input data contains no reviews")
            sys.exit(1)
            
        # Initialize pipeline with validated data
        pipeline = DataPipeline(hw_config)
        result = pipeline.process_synthetic_data(
            data, 
            custom_tag=args.tag,
            batch_size=args.batch_size
        )
        
        # Validate result structure
        if result['status'] != 'success' or 'data' not in result:
            logger.error(f"Processing failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
        
        # Log enhanced summary statistics
        summary = result['data']['summary']
        filtering = summary['filtering_summary']
        sentiment = summary['sentiment_distribution']
        
        logger.info("\n=== Processing Summary ===")
        logger.info("\nSentiment Distribution:")
        total_reviews = sum(sentiment.values())
        for sent, count in sentiment.items():
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            logger.info(f"  {sent.capitalize()}: {count} ({percentage:.1f}%)")
        
        logger.info("\nFiltering Summary:")
        logger.info(format_filtering_stats(filtering))
        
        logger.info("\nPerformance Metrics:")
        logger.info(f"Processing time: {result['performance']['processing_time']:.2f}s")
        logger.info(f"Average time per review: {result['performance'].get('avg_time_per_review', 0):.3f}s")
        
        logger.info("\nOutput Locations:")
        logger.info(f"Dataset ID: {result['dataset_info']['id']}")
        logger.info(f"Input data: {args.input_file}")
        logger.info(f"Processed data: {result['storage']['processed_path']}")
        logger.info(f"Metrics: {result['storage']['metrics_path']}")
        logger.info(f"Dataset path: {result['dataset_info']['path']}")
        logger.info("Dataset splits:", result['dataset_info']['splits'])
        
    except Exception as e:
        logger.error(f"Error processing reviews: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
