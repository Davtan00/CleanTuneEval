

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Allow imports from the parent directory
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.pipeline import DataPipeline
from src.config.logging_config import setup_logging
from src.config.environment import HardwareConfig

logger = setup_logging()

def format_filtering_stats(stats: Dict[str, Any]) -> str:
    """
    Format filtering statistics with percentages.
    'stats' is typically the 'filtering_summary' dict from the pipeline's final output.
    """
    total = stats.get('total_processed', 1)  # Avoid division by zero
    return (
        f"Length filtered: {stats['length_filtered']} ({stats['length_filtered']/total*100:.1f}%)\n"
        f"Duplicates removed: {stats['duplicates_removed']} ({stats['duplicates_removed']/total*100:.1f}%)\n"
        f"  - Exact duplicates: {stats.get('exact_duplicates', 0)} "
        f"({stats.get('exact_duplicates', 0)/total*100:.1f}%)\n"
        f"  - Near duplicates: {stats.get('near_duplicates', 0)} "
        f"({stats.get('near_duplicates', 0)/total*100:.1f}%)\n"
        f"Total removed: {stats['total_removed']} ({stats['total_removed']/total*100:.1f}%)"
    )

def main():
    parser = argparse.ArgumentParser(description='Process review data through the cleaning pipeline.')
    parser.add_argument('input_file', help='Path to input JSON file containing reviews')
    parser.add_argument('--tag', help='Optional custom tag for the dataset', default=None)
    parser.add_argument('--batch-size', type=int, default=1500, 
                       help='Batch size for processing (default: 1500)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='Force CPU usage even if accelerators are available')
    parser.add_argument('--preserve-distribution', action='store_true',
                       help='Preserve original data distribution (for evaluation datasets)')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    try:
        # Configure hardware settings with possible distribution preservation and CPU forcing
        hw_config = HardwareConfig(
            preserve_distribution=args.preserve_distribution,
            force_cpu=args.force_cpu
        )
        
        # Load input data
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Validate essential fields
        required_keys = {'domain', 'generated_data'}
        if not all(key in data for key in required_keys):
            missing = required_keys - set(data.keys())
            logger.error(f"Missing required keys in input data: {missing}")
            sys.exit(1)
            
        # Ensure 'generated_data' is non-empty
        if not data['generated_data']:
            logger.error("Input data contains no reviews in 'generated_data'")
            sys.exit(1)
            
        # Initialize pipeline with hardware configuration
        pipeline = DataPipeline(hw_config)
        
        # Run data through the pipeline
        result = pipeline.process_synthetic_data(
            data, 
            custom_tag=args.tag,
            batch_size=args.batch_size
        )
        
        # Check the pipeline result
        if result.get('status') != 'success' or 'data' not in result:
            logger.error(f"Processing failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
        
        # Unpack summary details
        summary = result['data']['summary']
        filtering = summary['filtering_summary']
        sentiment = summary['sentiment_distribution']
        
        logger.info("\n=== Processing Summary ===")
        
        # Sentiment Distribution
        logger.info("\nSentiment Distribution:")
        total_reviews = sum(sentiment.values())
        for sent, count in sentiment.items():
            percentage = (count / total_reviews * 100) if total_reviews > 0 else 0
            logger.info(f"  {sent.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Filtering Summary
        logger.info("\nFiltering Summary:")
        logger.info(format_filtering_stats(filtering))
        
        # Performance
        logger.info("\nPerformance Metrics:")
        performance = result.get('performance', {})
        logger.info(f"Processing time: {performance.get('processing_time', 0.0):.2f}s")
        avg_time = performance.get('avg_time_per_review', 0.0)
        logger.info(f"Average time per review: {avg_time:.5f}s")
        
        # Output Paths
        logger.info("\nOutput Locations:")
        logger.info(f"Dataset ID: {result['dataset_info']['id']}")
        logger.info(f"Input data: {args.input_file}")
        logger.info(f"Processed data: {result['storage']['data_path']}")
        logger.info(f"Metrics: {result['storage']['metrics_path']}")
        logger.info(f"Dataset path: {result['dataset_info']['path']}")
        logger.info(f"Dataset splits: {result['dataset_info']['splits']}")
        
        # Optional: Verify filtering counts quickly
        processed_data = result['data']['generated_data']
        accepted_count = len(processed_data)
        removed_count = filtering.get('total_removed', 0)
        logger.info(f"\nFinal Verification:")
        logger.info(f"Reviews in final data: {accepted_count}")
        logger.info(f"Reviews removed: {removed_count}")
        logger.info(f"Initial count: {filtering.get('total_processed', 0)}")
        
    except Exception as e:
        logger.error(f"Error processing reviews: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
