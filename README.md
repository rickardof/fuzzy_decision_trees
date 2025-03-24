# Fuzzy Decision Tree

A fuzzy decision tree classifier for machine learning tasks.

## Installation

```bash
pip install fuzzy-tree
```

## Usage

```python
from fuzzy_tree import FuzzyDecisionTree

# Create and train the model
model = FuzzyDecisionTree(num_fuzzy_sets=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Extract fuzzy rules
rules = model.extract_rules()
model.print_rules()
```