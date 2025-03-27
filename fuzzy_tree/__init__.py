# Import delle classi principali
from .fuzzy_decision_tree import FuzzyDecisionTree
from .fuzzy_node import FuzzyNode
from .fuzzy_sets import FuzzyDiscretizer, create_triangular_fuzzy_sets

# Versione del pacchetto
__version__ = "1.0.4"

# Esporta le classi principali per semplificare gli import
__all__ = [
    "FuzzyDecisionTree", 
    "FuzzyNode", 
    "FuzzyDiscretizer", 
    "create_triangular_fuzzy_sets"
]