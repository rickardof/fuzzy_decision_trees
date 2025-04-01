from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from simpful import FuzzySet

class FuzzyNode:
    """
    Fuzzy decision tree node.
    
    Each node represents either a fuzzy set on a feature (internal node)
    or a probability distribution of classes (leaf node).
    """
    
    __LAST_ID = 0
    
    @staticmethod
    def reset_node_id():
        """Resets the node ID counter""" #for when creating multiple trees or retraining the model
        FuzzyNode.__LAST_ID = 0
    
    def __init__(self, 
                 feature: int = -1, 
                 feature_name: str = None,
                 fuzzy_set = None, 
                 depth: int = 0, 
                 parent = None):
        """
        Initializes a fuzzy decision tree node
        
        Args:
            feature: Index of the feature tested by the node (-1 for root or leaf node)
            feature_name: Name of the feature (for interpretability)
            fuzzy_set: Fuzzy set used to test the feature
            depth: Depth of the node in the tree
            parent: Parent node
        """

        FuzzyNode.__LAST_ID += 1
        self.id = FuzzyNode.__LAST_ID
        self.feature = feature
        self.feature_name = feature_name
        self.fuzzy_set = fuzzy_set
        self.depth = depth
        self.parent = parent
        
        # tree's relationships
        self.children = []
        self.is_leaf = False
        
        # for leaf nodes 
        self.class_distribution = None
        
        # stats for tree construction
        self.information_gain = None
        self.samples_count = 0
        self.samples_above_threshold = 0
        self.fuzzy_entropy = None
        self.mean_activation_force = None
    
    def add_child(self, node):
        """Add a child node to the tree
            node: 'FuzzyNode'
        """
        self.children.append(node)
    
    def mark_as_leaf(self, class_distribution=None):
        """
        Marks the node as a leaf and sets the class distribution
        
        Args:
            class_distribution: [np.darray], Probability distribution of classes [p(c1), p(c2), ...]
    """
        self.is_leaf = True
        self.class_distribution = class_distribution
    
    def get_rule_path(self) -> List[Tuple[str, str]]:
        """
        Generates a textual representation of the path from the root to this node
        
        Returns:
            List of tuples (feature_name, term) representing the path
    """
        if self.parent is None:
            return []
            
        path = self.parent.get_rule_path()
        if self.feature >= 0 and self.fuzzy_set:
            path.append((self.feature_name, self.fuzzy_set.term))
        return path
    
    def membership_degree(self, x: np.ndarray) -> float:
        """
    Calculates the membership degree of an example to the node's fuzzy set
    
    Args:
        x: Feature vector
        
    Returns:
        Membership degree [0,1]
    """
        if self.fuzzy_set is None:
            return 1.0  # root or node without fuzzy set
        
        try:
            return self.fuzzy_set.get_value(x[self.feature])
        except Exception as e:
            print(f"Error while computing membership dregree: {e}")
            return 0.0
    

    
    def __str__(self):
        """Textual representation of the node"""
        if self.is_leaf:
            return f"Leaf(id={self.id}, depth={self.depth}, distr={self.class_distribution})"
        else:
            fuzzy_term = self.fuzzy_set.term if self.fuzzy_set else "ROOT"
            return f"Node(id={self.id}, depth={self.depth}, feature={self.feature_name}, term={fuzzy_term})"