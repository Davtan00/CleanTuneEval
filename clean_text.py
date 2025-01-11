import json
import re

def clean_text(text):
    """
    Clean text by removing problematic symbols and standardizing formatting.
    """
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)
    
    # Standardize quotes
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    # Standardize ellipsis
    text = re.sub(r'\.{2,}', '...', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def process_json_file(input_file, output_file):
    """
    Process JSON file by cleaning text in reviews.
    """
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Clean text in each review
    for review in data['generated_data']:
        review['text'] = clean_text(review['text'])
    
    # Write cleaned data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_file = "improvhcsyn.json"
    output_file = "cleaned_reviews.json"
    process_json_file(input_file, output_file)
    print(f"Cleaned data saved to {output_file}") 