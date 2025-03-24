from .fuzzy_decision_tree import FuzzyDecisionTree
from .fuzzy_node import FuzzyNode
from .fuzzy_sets import FuzzyDiscretizer, create_triangular_fuzzy_sets

__version__ = "1.0.0"

# Export main classes for easier imports
__all__ = [
    "FuzzyDecisionTree", 
    "FuzzyNode", 
    "FuzzyDiscretizer", 
    "create_triangular_fuzzy_sets"
]