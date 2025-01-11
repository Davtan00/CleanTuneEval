import argparse
import logging
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_leakage(dataset_path: str):
    """Check for exact text overlap between train/validation/test and show label distribution."""
    dataset = load_from_disk(dataset_path)
    
    # Gather the text fields in each split into sets
    train_texts = set(example['text'] for example in dataset['train'])
    valid_texts = set(example['text'] for example in dataset['validation'])
    test_texts = set(example['text'] for example in dataset['test'])
    
    # Compute intersections to detect any overlap
    train_valid_overlap = train_texts.intersection(valid_texts)
    valid_test_overlap = valid_texts.intersection(test_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    
    # Log the actual overlapping texts
    if train_valid_overlap:
        logger.info("\nTrain-Validation overlapping texts:")
        for text in train_valid_overlap:
            train_examples = [ex for ex in dataset['train'] if ex['text'] == text]
            valid_examples = [ex for ex in dataset['validation'] if ex['text'] == text]
            logger.info(f"\nText: {text[:100]}...")
            logger.info(f"Train label: {train_examples[0]['labels']}")
            logger.info(f"Validation label: {valid_examples[0]['labels']}")
    
    if valid_test_overlap:
        logger.info("\nValidation-Test overlapping texts:")
        for text in valid_test_overlap:
            valid_examples = [ex for ex in dataset['validation'] if ex['text'] == text]
            test_examples = [ex for ex in dataset['test'] if ex['text'] == text]
            logger.info(f"\nText: {text[:100]}...")
            logger.info(f"Validation label: {valid_examples[0]['labels']}")
            logger.info(f"Test label: {test_examples[0]['labels']}")
    
    if train_test_overlap:
        logger.info("\nTrain-Test overlapping texts:")
        for text in train_test_overlap:
            train_examples = [ex for ex in dataset['train'] if ex['text'] == text]
            test_examples = [ex for ex in dataset['test'] if ex['text'] == text]
            logger.info(f"\nText: {text[:100]}...")
            logger.info(f"Train label: {train_examples[0]['labels']}")
            logger.info(f"Test label: {test_examples[0]['labels']}")
    
    logger.info(f"Number of train-validation overlapping texts: {len(train_valid_overlap)}")
    logger.info(f"Number of validation-test overlapping texts: {len(valid_test_overlap)}")
    logger.info(f"Number of train-test overlapping texts: {len(train_test_overlap)}")
    
    # Label distribution checks
    def get_label_distribution(split_name):
        dist_dict = {}
        for example in dataset[split_name]:
            label = example['labels']  # numeric label if already mapped
            dist_dict[label] = dist_dict.get(label, 0) + 1
        return dist_dict

    train_dist = get_label_distribution('train')
    valid_dist = get_label_distribution('validation')
    test_dist = get_label_distribution('test')
    
    logger.info(f"Train label distribution: {train_dist}")
    logger.info(f"Validation label distribution: {valid_dist}")
    logger.info(f"Test label distribution: {test_dist}")
    
    # Return these stats for any further analysis if needed
    return {
        'train_valid_overlap': len(train_valid_overlap),
        'valid_test_overlap': len(valid_test_overlap),
        'train_test_overlap': len(train_test_overlap),
        'train_label_dist': train_dist,
        'valid_label_dist': valid_dist,
        'test_label_dist': test_dist
    }

def main():
    parser = argparse.ArgumentParser(description="Check dataset for possible leakage and label distribution.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the dataset (same as used in your training script).")
    args = parser.parse_args()

    results = check_data_leakage(args.dataset_path)
    
    # Simple detection message
    if (results['train_valid_overlap'] > 0 or 
        results['valid_test_overlap'] > 0 or 
        results['train_test_overlap'] > 0):
        logger.info("Data leakage detected: Some texts appear in multiple splits.")
    else:
        logger.info("No data leakage based on exact text matching.")
        logger.info("If accuracy is still extremely high, the dataset may simply be easy.")

if __name__ == "__main__":
    main()
