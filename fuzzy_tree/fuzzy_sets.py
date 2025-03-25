# src/utils/fuzzy_sets.py
from typing import List, Tuple, Dict
import numpy as np
from simpful import FuzzySet, Triangular_MF

class FuzzyDiscretizer:
    """Crea partizioni fuzzy per feature continue"""
    
    def __init__(self, num_fuzzy_sets: int, method: str = "uniform"):
        """
        Inizializza il discretizzatore fuzzy
        
        Args:
            num_fuzzy_sets: Numero di insiemi fuzzy per feature
            method: Metodo di discretizzazione ('uniform' o 'quantile')
        """
        self.num_fuzzy_sets = num_fuzzy_sets
        self.method = method
        
    def run(self, X: np.ndarray, features_to_discretize: List[bool]) -> List[List[float]]:
        """
        Calcola i punti di divisione per le feature selezionate
        
        Args:
            X: Dati di input (numpy array)
            features_to_discretize: Lista di booleani che indica quali feature discretizzare
            
        Returns:
            Lista di liste, ogni lista contiene i punti di divisione per una feature
        """
        splits = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            if features_to_discretize[i]:
                feature_values = X[:, i]
                min_val = np.min(feature_values)
                max_val = np.max(feature_values)
                
                if self.method == "uniform":
                    # Divisione uniforme dell'intervallo
                    points = np.linspace(min_val, max_val, self.num_fuzzy_sets + 1)
                elif self.method == "quantile":
                    # Divisione basata sui quantili
                    points = np.percentile(feature_values, 
                                          np.linspace(0, 100, self.num_fuzzy_sets + 1))
                else:
                    raise ValueError(f"Metodo non supportato: {self.method}")
                
                splits.append(points.tolist())
            else:
                splits.append([])
                
        return splits

def create_triangular_fuzzy_sets(points: List[float]) -> List[FuzzySet]:
    """
    Crea insiemi fuzzy triangolari da una lista di punti
    
    Args:
        points: Lista di punti che definiscono la partizione
        
    Returns:
        Lista di insiemi fuzzy (oggetti FuzzySet di Simpful)
    """
    fuzzy_sets = []
    n_sets = len(points) - 1
    
    # Nomi linguistici degli insiemi fuzzy
    linguistic_terms_5 = ["VERY_LOW", "LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    linguistic_terms_3 = ["LOW","MEDIUM","HIGH"]
    linguistic_terms_2 = ["LOW","HIGH"]

    # Seleziona i termini linguistici appropriati in base al numero di insiemi
    if n_sets == 5:
        linguistic_terms = linguistic_terms_5
    elif n_sets == 3:
        linguistic_terms = linguistic_terms_3
    elif n_sets == 2:
        linguistic_terms = linguistic_terms_2
    else:
        # Per altri valori di n_sets, generiamo nomi numerici
        linguistic_terms = [f"FS_{i}" for i in range(n_sets)]
    
    for i in range(n_sets):
        # Per il primo insieme (bordo sinistro)
        if i == 0:
            a = points[i]
            b = points[i]
            c = points[i + 1]
        # Per l'ultimo insieme (bordo destro)
        elif i == n_sets - 1:
            a = points[i]
            b = points[i + 1]
            c = points[i + 1]
        # Per gli insiemi interni
        else:
            a = points[i]
            b = (points[i] + points[i + 1]) / 2
            c = points[i + 1]
            
        term = linguistic_terms[i] if i < len(linguistic_terms) else f"FS_{i}"
        fs = FuzzySet(function=Triangular_MF(a=a, b=b, c=c), term=term)
        fuzzy_sets.append(fs)
    
    return fuzzy_sets