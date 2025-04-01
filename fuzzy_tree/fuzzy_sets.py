
from typing import List, Tuple, Dict
import numpy as np
from simpful import FuzzySet, Triangular_MF

class FuzzyDiscretizer:
    """Creates fuzzy partitions for continuous features"""
    
    def __init__(self, num_fuzzy_sets: int, method: str = "uniform"):
        """
        Initializes the fuzzy discretizer
        
        Args:
            num_fuzzy_sets: Number of fuzzy sets per feature
            method: Discretization method ('uniform' or 'quantile')
    """
        self.num_fuzzy_sets = num_fuzzy_sets
        self.method = method
        
    def run(self, X: np.ndarray, features_to_discretize: List[bool]) -> List[List[float]]:
        """
        Calculates the division points for the selected features
        
        Args:
            X: Input data (numpy array)
            features_to_discretize: List of booleans indicating which features to discretize
            
        Returns:
            List of lists, each list contains division points for a feature
    """
            # Validate inputs
        if X is None or X.size == 0:
            raise ValueError("Input data X cannot be empty")

        splits = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            if features_to_discretize[i]:
                feature_values = X[:, i]
                min_val = np.min(feature_values)
                max_val = np.max(feature_values)
                
                if self.method == "uniform":
                    # uniform division of the interval
                    points = np.linspace(min_val, max_val, self.num_fuzzy_sets + 1)
                elif self.method == "quantile":
                    # quantile based
                    points = np.percentile(feature_values, 
                                          np.linspace(0, 100, self.num_fuzzy_sets + 1))
                else:
                    raise ValueError(f"Method not supported: {self.method}")
                
                splits.append(points.tolist())
            else:
                splits.append([])
                
        return splits

def create_triangular_fuzzy_sets(points: List[float]) -> List[FuzzySet]:
    """
        Creates triangular fuzzy sets from a list of points with strong partition
        
        Args:
            points: List of points defining the partition
            
        Returns:
            List of fuzzy sets (Simpful FuzzySet objects)
    """

    if len(points) < 2:
        raise ValueError("Points list must contain at least 2 points to create fuzzy sets")
    
    # Check for non-numeric values
    try:
        [float(p) for p in points]
    except (TypeError, ValueError):
        raise ValueError("All points must be numeric values")

    fuzzy_sets = []
    n_sets = len(points) - 1
    
    # linguistic names for the terms, chosen arbitrarily for 2
    linguistic_terms_7 = ["EXTREMELY LOW", "VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH", "EXTREMELY HIGH"]
    linguistic_terms_5 = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    linguistic_terms_3 = ["LOW","MEDIUM","HIGH"]
    linguistic_terms_2 = ["LOW","HIGH"]

    
    if n_sets == 7:
        linguistic_terms = linguistic_terms_7
    elif n_sets == 5:
        linguistic_terms = linguistic_terms_5
    elif n_sets == 3:
        linguistic_terms = linguistic_terms_3
    elif n_sets == 2:
        linguistic_terms = linguistic_terms_2
    else:
        linguistic_terms = [f"FS_{i}" for i in range(n_sets)]
    
    for i in range(n_sets):
        # first set (left border)
        if i == 0:
            a = points[i]
            b = points[i]
            c = points[i + 1]
        # last set (right border)
        elif i == n_sets - 1:
            a = points[i - 1]  
            b = points[i]
            c = points[i]
        # internal sets
        else:
            a = points[i - 1]  
            b = points[i]      
            c = points[i + 1]  
            
        term = linguistic_terms[i] if i < len(linguistic_terms) else f"FS_{i}"
        fs = FuzzySet(function=Triangular_MF(a=a, b=b, c=c), term=term)
        fuzzy_sets.append(fs)
    
    return fuzzy_sets