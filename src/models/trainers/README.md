# Training Pipeline Enhancement Guide

## Phase 1: Basic Training (Current)
### Success Criteria
- Accuracy > 0.70
- No single class dominates predictions (< 80% of total)

### Triggers for Phase 2
Move to Phase 2 (Data Balancing) if after 5 epochs:
1. Accuracy < 0.70 OR
2. All predictions fall into single class OR
3. Two or more classes have 0% predictions

### Metrics Priority
1. Accuracy (primary metric for Phase 1)
2. Prediction Distribution (secondary check)
3. F1 Scores (monitored but not primary)

## Future Phases Implementation Guide TBD
... 