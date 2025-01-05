# Duplicate Detection Analysis and Challenges

## Test Cases and Sentences

### 1. Exact Duplicate Test
```python
texts = [
    "After updating the software on my tablet, its been running slower than molasses. I regret hitting that update button it feels like ive lost functionality.",
    "After updating the software on my tablet, its been running slower than molasses. I regret hitting that update button it feels like ive lost functionality.",  # Exact duplicate
    "The tablet i purchased has a nice screen but is incredibly slow. Apps take forever to open, and it struggles with basic tasks. I wouldnt recommend it to anyone looking for efficiency.",
    "AFTER UPDATING THE SOFTWARE ON MY TABLET, ITS BEEN RUNNING SLOWER THAN MOLASSES. I REGRET HITTING THAT UPDATE BUTTON IT FEELS LIKE IVE LOST FUNCTIONALITY."  # Case different
]
```

### 2. Similar Text Test (Word Rearrangement)
```python
texts = [
    "The laptop freezes when running multiple apps and customer support hasnt helped resolve these issues.",
    "Customer support hasnt helped resolve these issues when the laptop freezes running multiple apps.",  # Word rearrangement
    "The smart home device has simplified my life with responsive voice controls."  # Different review
]
```

### 3. Semantic Similarity Test
```python
texts = [
    "This device is extremely sluggish and unresponsive, making it frustrating to use.",
    "The performance is terribly slow and the system barely responds to input, which is very annoying.",  # Different words, same meaning
    "The software update added new features but had some minor issues."  # Different meaning
]
```

## Current Implementation and Failed Attempts

### Libraries Used
1. **N-gram Similarity**
   - Library: NLTK
   - Method: Character n-grams with n=3
   - Metric: Jaccard similarity coefficient

2. **Semantic Similarity**
   - Library: Sentence-Transformers
   - Model: 'all-MiniLM-L6-v2'
   - Metric: Cosine similarity between sentence embeddings

### Attempted Configurations

1. **First Attempt**: Combined weighted similarity
```python
'ngram_weight': 0.5,
'semantic_weight': 0.5,
'threshold': 0.5
```
Result: Failed both test cases - missed semantic duplicates

2. **Second Attempt**: Lower threshold
```python
'ngram_weight': 0.5,
'semantic_weight': 0.5,
'threshold': 0.45
```
Result: Too many false positives, still missed semantic duplicates

3. **Third Attempt**: Separate thresholds
```python
'ngram_threshold': 0.4,
'semantic_threshold': 0.6
```
Result: False positives on different reviews

4. **Fourth Attempt**: Higher thresholds
```python
'ngram_threshold': 0.6,
'semantic_threshold': 0.75
```
Result: Still failing both test cases

## Current Problems

1. **Word Rearrangement Test**
   - Can't find proper threshold for n-gram similarity
   - Too low: marks different reviews as duplicates
   - Too high: misses rearranged sentences

2. **Semantic Similarity Test**
   - Current model/thresholds can't reliably detect paraphrased content
   - Semantic similarity scores are inconsistent

## Help Needed

1. **Threshold Determination**
   - What are appropriate thresholds for our use case?
   - How to balance precision vs recall?

2. **Alternative Approaches**
   - Are there better similarity metrics?
   - Should we use different models?
   - How to handle word rearrangement better?

3. **Model Selection**
   - Is all-MiniLM-L6-v2 appropriate for review similarity?
   - Would a domain-specific model work better?

We need expert guidance on these questions as our current approach is not working despite multiple attempts at tuning the parameters.
