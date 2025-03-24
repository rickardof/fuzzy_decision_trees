from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from simpful import FuzzySet

class FuzzyNode:
    """
    Nodo dell'albero decisionale fuzzy per classificazione.
    
    Ogni nodo rappresenta o un test fuzzy su una feature (nodo interno) 
    o una distribuzione di probabilità delle classi (nodo foglia).
    """
    
    __LAST_ID = 0
    
    @staticmethod
    def reset_node_id():
        """Resetta il contatore degli ID dei nodi"""
        FuzzyNode.__LAST_ID = 0
    
    def __init__(self, 
                 feature: int = -1, 
                 feature_name: str = None,
                 fuzzy_set = None, 
                 depth: int = 0, 
                 parent = None):
        """
        Inizializza un nodo dell'albero decisionale fuzzy
        
        Args:
            feature: Indice della feature testata dal nodo (-1 per nodo radice o foglia)
            feature_name: Nome della feature (per interpretabilità)
            fuzzy_set: Insieme fuzzy usato per testare la feature
            depth: Profondità del nodo nell'albero
            parent: Nodo genitore
        """
        FuzzyNode.__LAST_ID += 1
        self.id = FuzzyNode.__LAST_ID
        self.feature = feature
        self.feature_name = feature_name
        self.fuzzy_set = fuzzy_set
        self.depth = depth
        self.parent = parent
        
        # Relazioni dell'albero
        self.children = []
        self.is_leaf = False
        
        # Per i nodi foglia (classificazione)
        self.class_distribution = None
        
        # Statistiche per la costruzione dell'albero
        self.information_gain = None
        self.samples_count = 0
        self.samples_above_threshold = 0
        self.fuzzy_entropy = None
        self.mean_activation_force = None
    
    def add_child(self, node):
        """Aggiunge un nodo figlio"""
        self.children.append(node)
    
    def mark_as_leaf(self, class_distribution=None):
        """
        Marca il nodo come foglia e imposta la distribuzione di classe
        
        Args:
            class_distribution: Distribuzione di probabilità delle classi [p(c1), p(c2), ...]
        """
        self.is_leaf = True
        self.class_distribution = class_distribution
    
    def get_rule_path(self) -> List[Tuple[str, str]]:
        """
        Genera una rappresentazione testuale del percorso dalla radice a questo nodo
        
        Returns:
            Lista di tuple (feature_name, term) che rappresentano il percorso
        """
        if self.parent is None:
            return []
            
        path = self.parent.get_rule_path()
        if self.feature >= 0 and self.fuzzy_set:
            path.append((self.feature_name, self.fuzzy_set.term))
        return path
    
    def membership_degree(self, x: np.ndarray) -> float:
        """
        Calcola il grado di appartenenza di un esempio all'insieme fuzzy del nodo
        
        Args:
            x: Vettore delle feature
            
        Returns:
            Grado di appartenenza [0,1]
        """
        if self.fuzzy_set is None:
            return 1.0  # Nodo radice o senza insieme fuzzy
        
        try:
            return self.fuzzy_set.get_value(x[self.feature])
        except Exception as e:
            print(f"Errore nel calcolo del grado di appartenenza: {e}")
            return 0.0
    

    
    def __str__(self):
        """Rappresentazione testuale del nodo"""
        if self.is_leaf:
            return f"Foglia(id={self.id}, prof={self.depth}, distr={self.class_distribution})"
        else:
            fuzzy_term = self.fuzzy_set.term if self.fuzzy_set else "ROOT"
            return f"Nodo(id={self.id}, prof={self.depth}, feature={self.feature_name}, term={fuzzy_term})"