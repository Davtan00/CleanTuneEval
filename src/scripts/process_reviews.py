#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.pipeline import DataPipeline
from src.config.logging_config import setup_logging

logger = setup_logging()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process review data through the cleaning pipeline.')
    parser.add_argument('input_file', help='Path to input JSON file containing reviews')
    parser.add_argument('--tag', help='Optional custom tag for the dataset', default=None)
    
    args = parser.parse_args()
    input_file = args.input_file
    
    if not Path(input_file).exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    try:
        # Load input data
        with open(input_file) as f:
            data = json.load(f)
        
        # Initialize and run pipeline
        pipeline = DataPipeline()
        result = pipeline.process_synthetic_data(data, custom_tag=args.tag)
        
        if result['status'] != 'success':
            logger.error(f"Processing failed: {result['message']}")
            sys.exit(1)
        
        # Log summary statistics
        summary = result['data']['summary']
        filtering = summary['filtering_summary']
        sentiment = summary['sentiment_distribution']
        
        logger.info("\nProcessing Summary:")
        logger.info("Sentiment Distribution:")
        logger.info(f"  Positive: {sentiment['positive']}")
        logger.info(f"  Neutral: {sentiment['neutral']}")
        logger.info(f"  Negative: {sentiment['negative']}")
        
        logger.info("\nFiltering Summary:")
        logger.info(f"Length filtered: {filtering['length_filtered']}")
        logger.info(f"Duplicates removed: {filtering['duplicates_removed']}")
        logger.info(f"Total removed: {filtering['total_removed']}")
        
        logger.info("\nOutput Locations:")
        logger.info(f"Dataset ID: {result['dataset_info']['id']}")
        logger.info(f"Input data: {input_file}")
        logger.info(f"Processed data: {result['storage']['processed_path']}")
        logger.info(f"Metrics: {result['storage']['metrics_path']}")
        logger.info(f"Dataset path: {result['dataset_info']['path']}")
        logger.info("Dataset splits:", result['dataset_info']['splits'])
        
    except Exception as e:
        logger.error(f"Error processing reviews: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
