
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from fuzzy_tree.fuzzy_node import FuzzyNode
from fuzzy_tree.fuzzy_sets import FuzzyDiscretizer, create_triangular_fuzzy_sets

class FuzzyDecisionTree:
    """
    Implementazione di un albero decisionale fuzzy per classificazione.
    """
    
    def __init__(self, 
             num_fuzzy_sets: int = 5, 
             max_depth: Optional[int] = None, 
             min_samples_split: int = 2,
             min_samples_leaf: int = 1,
             gain_threshold: float = 0.001,
             membership_threshold: float = 0.5,
             prediction_policy: str = "max_matching"):
        """
        Initializes the fuzzy decision tree classifier
        
        Args:
            num_fuzzy_sets: Number of fuzzy sets per feature
            max_depth: Maximum depth of the tree (None = unlimited)
            min_samples_split: Minimum number of samples to perform a split
            min_samples_leaf: Minimum number of samples for a leaf node
            gain_threshold: Minimum information gain to perform a split
            membership_threshold: Minimum membership threshold to consider an example
            prediction_policy: Prediction policy ('max_matching' or 'weighted')
        """
        # Parameter validation
        if num_fuzzy_sets <= 0:
            raise ValueError("num_fuzzy_sets must be positive")
        if membership_threshold < 0 or membership_threshold > 1:
            raise ValueError("membership_threshold must be in the range [0,1]")
        
        # Verify that the prediction policy is valid
        valid_policies = ["max_matching", "weighted"]
        if prediction_policy not in valid_policies:
            raise ValueError(f"prediction_policy must be one of {valid_policies}")
        
        self.num_fuzzy_sets = num_fuzzy_sets
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.membership_threshold = membership_threshold
        self.prediction_policy = prediction_policy
        
        # Initialize the tree
        self.root = None
        self.fuzzy_sets = None
        self.feature_names = None
        self.n_features = None
        self.n_classes = None
        self.class_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None, class_names=None):
        """
        Trains the fuzzy decision tree
        
        Args:
            X: Feature matrix
            y: Label vector
            feature_names: Names of features (optional)
            class_names: Names of classes (optional)
        """
        # Reset node IDs
        FuzzyNode.reset_node_id()

        # Validation checks
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contains NaN or infinite values")
        
        self.y_train = y.copy()
        
        # Save data information
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        if feature_names is None:
            self.feature_names = [f"Feature {i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names
            
        if class_names is None:
            self.class_names = [f"Class {i}" for i in range(self.n_classes)]
        else:
            self.class_names = class_names
        
        # Create fuzzy partitions for each feature
        self._create_fuzzy_partitions(X)
        
        # Create the root node
        self.root = FuzzyNode(feature=-1, feature_name="Root")
        
        # Recursive call to build the tree
        self._build_tree(self.root, X, y, depth=0)
        
        return self
    
    def _create_fuzzy_partitions(self, X: np.ndarray):
        """
        Creates fuzzy partitions for each feature
        
        Args:
            X: Feature matrix
        """
        # Use the FuzzyDiscretizer to get division points
        discretizer = FuzzyDiscretizer(self.num_fuzzy_sets, method="uniform")
        features_to_discretize = [True] * self.n_features
        splits = discretizer.run(X, features_to_discretize)
        
        # Create fuzzy sets for each feature
        self.fuzzy_sets = {}  # dict
        for i in range(self.n_features):
            if len(splits[i]) > 0:  # if there are splits in the feature
                self.fuzzy_sets[i] = create_triangular_fuzzy_sets(splits[i])
            else:
                self.fuzzy_sets[i] = []
    
    def _build_tree(self, node: FuzzyNode, X: np.ndarray, y: np.ndarray, depth: int):
        """
        Recursively builds the fuzzy decision tree
        
        Args:
            node: Current node
            X: Feature matrix
            y: Label vector
            depth: Current depth
        """
        # CHECK FOR RECURSION
        if depth > 100:  # arbitrarily chosen
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return

        n_samples = X.shape[0]   #how many examples reach this node
        node.samples_count = n_samples
        
        # check for stop criteria specified at the start (maximum depth, not enough samples or all samples belonging to the same class)
        if (self.max_depth is not None and depth >= self.max_depth) or \
        n_samples < self.min_samples_split or \
        len(np.unique(y)) == 1:
            # create a leaf
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return
            
        # find the best feature to split the node
        best_feature, best_gain = self._find_best_split(X, y)


        #---------- as of now no check there is to guarantee al features are unique for every node ------------


        
        # if we don't find a valid split, create a leaf
        if best_feature is None or best_gain <= self.gain_threshold:
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return
        
        #set chosen feature for this node
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature] if self.feature_names is not None else f"Feature_{best_feature}"
        
        # Check: if feature values are too similar, create a leaf
        feature_values = X[:, best_feature]
        if np.max(feature_values) - np.min(feature_values) < 1e-6:
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return
        
        # create a finger print of samples to detect loops (when the dame data subset appears multiple times)
        data_hash = hash(str(X.shape) + str(np.sum(X)) + str(np.sum(y)))
        if hasattr(self, '_data_hashes'):
            if data_hash in self._data_hashes:
                # we have found a potential loops 
                class_distribution = self._compute_class_distribution(y)
                node.mark_as_leaf(class_distribution)
                return
            self._data_hashes.add(data_hash)
        else:
            self._data_hashes = {data_hash}
        
        # now for every fuzzy set, create a node
        for fs_idx, fuzzy_set in enumerate(self.fuzzy_sets[best_feature]):
            # compute samples membership degree to the fuzzy set
            membership_degrees = np.array([
                fuzzy_set.get_value(x[best_feature]) for x in X
            ])
            
            # filtering operation based on membership treashold
            valid_samples = membership_degrees >= self.membership_threshold
            X_child = X[valid_samples]
            y_child = y[valid_samples]
            
            # if there are no valid samples, skip this child
            if len(X_child) == 0:
                continue
            
            # create new node
            child_node = FuzzyNode(
                feature=-1,  # will be chosen during recursive call
                fuzzy_set=fuzzy_set,
                depth=depth + 1,
                parent=node
            )
            
            # add child to this node
            node.add_child(child_node)
            
            # build recursively the sub tree
            self._build_tree(child_node, X_child, y_child, depth + 1)
        
        # if we don't have children, mark it as leaf
        if len(node.children) == 0:
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Finds the feature with the best information gain
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Index of the best feature and its information gain
        """
        best_feature = None
        best_gain = -float('inf')
        
        # Initial entropy (before split)
        parent_entropy = self._entropy(y)
        
        for feature in range(self.n_features):
            # Calculate information gain for this feature
            gain = self._information_gain(X, y, feature, parent_entropy)
            
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
                
        return best_feature, best_gain
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculates Shannon entropy
        
        Args:
            y: Label vector
            
        Returns:
            Entropy
        """
        n_samples = len(y)
        if n_samples == 0:
            return 0
        
        counts = np.bincount(y.astype(int))
        proportions = counts[counts > 0] / n_samples
        
        return -np.sum(proportions * np.log2(proportions))
    
    def _fuzzy_entropy(self, y: np.ndarray, membership_degrees: np.ndarray) -> float:
        """
        Calculates fuzzy entropy
        
        Args:
            y: Label vector
            membership_degrees: Membership degrees of examples
            
        Returns:
            Fuzzy entropy
        """
        classes = np.unique(y)
        total_membership = np.sum(membership_degrees)
        
        if total_membership == 0:
            return 0
        
        entropy = 0  # initialization

        for c in classes:
            class_membership = np.sum(membership_degrees[y == c])
            if class_membership > 0:
                p = class_membership / total_membership
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray, feature: int, 
                         parent_entropy: Optional[float] = None) -> float:
        """
        Calculates the fuzzy information gain for a feature
        
        Args:
            X: Feature matrix
            y: Label vector
            feature: Index of the feature to evaluate
            parent_entropy: Entropy of the parent node (if known)
            
        Returns:
            Fuzzy information gain
    """
        if parent_entropy is None:
            parent_entropy = self._entropy(y)
            
        # Entropy computation after split
        weighted_entropy = 0
        total_samples = len(y)
        
        for fuzzy_set in self.fuzzy_sets[feature]:  #for every fuzzy set associated with the feature to be analyzed...

            # membership degrees for this fuzzy set
            membership_degrees = np.array([
                fuzzy_set.get_value(x[feature]) for x in X
            ])
            
            # fuzzy entropy for this fuzzy set
            fuzzy_entropy = self._fuzzy_entropy(y, membership_degrees)
            
            # weight based on the sum of membership degrees
            weight = np.sum(membership_degrees) / total_samples
            
            weighted_entropy += weight * fuzzy_entropy
            
        # Information gain
        return parent_entropy - weighted_entropy
    
    def _compute_class_distribution(self, y: np.ndarray) -> np.ndarray:
        """
        Calculates the probability distribution of classes
        
        Args:
            y: Label vector
            
        Returns:
            Probability array [p(class_0), p(class_1), ...]
        """
        if len(y) == 0:
            # If there are no examples, use uniform distribution
            return np.ones(self.n_classes) / self.n_classes
            
        # Count classes
        counts = np.bincount(y.astype(int), minlength=self.n_classes)
        
        # Normalize to get probabilities
        return counts / len(y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classifies new examples using maximum matching
        
        Args:
            X: Feature matrix or single example
            
        Returns:
            Predicted labels
        """
        if self.root is None:
            raise ValueError("The tree has not been trained")
        
        # If the input is a single example, reformat it as a matrix with a single row
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        # Process predictions in batches for very large datasets
        batch_size = 10000  # Arbitrarily chosen
        if len(X) > batch_size:
            predictions = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_preds = np.zeros(len(batch), dtype=int)
                batch_probas = self.predict_proba(batch)
                batch_preds = np.argmax(batch_probas, axis=1)
                
                predictions.append(batch_preds)
            return np.concatenate(predictions)
        
        # Normal predictions
        return np.array([np.argmax(self.predict_proba(sample.reshape(1, -1))[0]) for sample in X])
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns class probabilities using the prediction_policy
        
        Args:
            X: Feature matrix or single example
            
        Returns:
            Array of class probabilities
        """
        if self.root is None:
            raise ValueError("The tree has not been trained")
                
        if len(X.shape) == 1:
            # Single example
            X = X.reshape(1, -1)
                
        # Verify dimensions
        if X.shape[1] != self.n_features:
            raise ValueError(f"Incorrect number of features: expected {self.n_features}, received {X.shape[1]}")
                
        probas = []
        for sample in X:
            # _predict_single_example finds the path with maximum activation and returns the resulting leaf distribution
            _, class_distribution = self._predict_single_example(sample, self.root)
            probas.append(class_distribution)
                
        return np.array(probas)
    
    def _predict_single_example(self, x: np.ndarray, node: FuzzyNode, current_activation: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Dispatcher for the appropriate prediction method based on policy
        
        Args:
            x: Feature vector
            node: Current node
            current_activation: Accumulated activation up to this node
            
        Returns:
            A tuple (max_activation, best_leaf_distribution)
        """
        if self.prediction_policy == "max_matching":
            return self._predict_max_matching(x, node, current_activation)
        elif self.prediction_policy == "weighted":
            return self._predict_weighted(x, node, current_activation)
        else:
            # By default use max_matching
            return self._predict_max_matching(x, node, current_activation)
    
    def _predict_max_matching(self, x: np.ndarray, node: FuzzyNode, current_activation: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Finds the path with maximum activation for a single example (maximum matching policy)
        
        Args:
            x: Feature vector
            node: Current node
            current_activation: Accumulated activation up to this node
            
        Returns:
            A tuple (max_activation, best_leaf_distribution)
            
        Notes:
        - Uses multiplicative t-norm (AND) for combining membership degrees along the path
        - Returns the leaf with highest path activation (product of all membership degrees)
        - Falls back to prior distribution if no valid path is found
    
    """
        

        # base case: leaf node
        if node.is_leaf:
            return current_activation, node.class_distribution
        
        # if the node doesn't have childs, treat it as a leaf with a uniform distribution (rare case)
        if len(node.children) == 0:
            return current_activation, np.ones(self.n_classes) / self.n_classes
            
        #For each child, compute activation and find best route
        max_activation = 0.0
        best_leaf_distribution = None  # None for when we don't find routes
        
        for child in node.children:
            # Calcola il grado di appartenenza a questo insieme fuzzy
            membership = child.fuzzy_set.get_value(x[node.feature])

            # Prevent numerical issues with very small values
            membership = np.clip(membership, 1e-10, 1.0)
            
            #Compute activation of this branch (using multiplication like t-norm for AND)
            branch_activation = current_activation * membership

            
            # IF the activation is significant, explore the branch
            if branch_activation > 0:
                # recurisve call
                child_activation, child_distribution = self._predict_max_matching(
                    x, child, branch_activation
                )
                
                # if this route has higher activation, update stats (maximum values)
                if child_activation > max_activation:
                    max_activation = child_activation
                    best_leaf_distribution = child_distribution
        
        # case where we don't find any valid path
        if best_leaf_distribution is None:
            # use prior class distribution if available, otherwise a uniform distribution
            if hasattr(self, 'y_train'):
                prior_distribution = self._compute_class_distribution(self.y_train)
                return 0.0, prior_distribution
            else:
                return 0.0, np.ones(self.n_classes) / self.n_classes
        
        return max_activation, best_leaf_distribution
    
    def _predict_weighted(self, x: np.ndarray, node: FuzzyNode, current_activation: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Combines predictions from all paths proportionally to their activation
        
        Args:
            x: Feature vector
            node: Current node
            current_activation: Accumulated activation up to this node
            
        Returns:
            A tuple (total_activation, weighted_distribution)
    """
        

        # Base case
        if node.is_leaf:
            return current_activation, node.class_distribution
            
        # case where node doesn't have children, treat it as leaf with uniform distribution (rare case)
        if len(node.children) == 0:
            return current_activation, np.ones(self.n_classes) / self.n_classes
            
        # accumulators for weighted mean
        total_activation = 0.0
        combined_distribution = np.zeros(self.n_classes)
        
        # explor all children with activation >0
        for child in node.children:
            #compute membership degree for this fuzzy sets

            membership = child.fuzzy_set.get_value(x[node.feature])
            
            membership = np.clip(membership, 1e-10, 1.0)

            # compute activation for this branch
            branch_activation = current_activation * membership
            
            # if activation is greater than 0 explore the branch
            if branch_activation > 0:
                child_activation, child_distribution = self._predict_weighted(
                    x, child, branch_activation
                )
                
                # add this weighted distribution
                combined_distribution += child_activation * child_distribution
                total_activation += child_activation
        
        # normalize combined distribution
        if total_activation > 0:
            combined_distribution /= total_activation
        else:
            # if no path has activation
            if hasattr(self, 'y_train'):
                combined_distribution = self._compute_class_distribution(self.y_train)
            else:
                combined_distribution = np.ones(self.n_classes) / self.n_classes
            
        return total_activation, combined_distribution
    
    def extract_rules(self) -> List[Dict]:
        """
        Extracts fuzzy rules from the tree
        
        Returns:
            List of fuzzy rules in dictionary format
        """
        rules = []
        self._extract_rules_recursive(self.root, [], rules)
        return rules
    
    def _extract_rules_recursive(self, node: FuzzyNode, path: List, rules: List):
        """
        Recursively extracts fuzzy rules from the tree
        
        Args:
            node: Current node
            path: Current path (list of tuples (feature, fuzzy_set))
            rules: List of rules (modified in-place)
        """
        # Base case
        if node.is_leaf:
            # Create a rule for this path
            predicted_class = np.argmax(node.class_distribution)
            confidence = node.class_distribution[predicted_class]
            
            rule = {
                'antecedent': path.copy(),
                'consequent': predicted_class,
                'class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"Class_{predicted_class}",
                'confidence': confidence,
                'class_distribution': node.class_distribution.copy()
            }
            rules.append(rule)
            return
            
        for child in node.children:
            # Add this condition to the path
            condition = (node.feature, node.feature_name, child.fuzzy_set)
            path.append(condition)
            
            # Recursive call
            self._extract_rules_recursive(child, path, rules)
            
            # Remove the condition from the path
            path.pop()
    
    def print_rules(self):
        """Prints fuzzy rules in readable format"""
        rules = self.extract_rules()

        print(f"Fuzzy decision tree rules ({len(rules)} rules):")
        print("=" * 80)
        
        for i, rule in enumerate(rules):
            print(f"Rule {i + 1}:")
            
            # Antecedent
            if len(rule['antecedent']) > 0:
                print("  IF ", end="")
                for j, (feature_idx, feature_name, fuzzy_set) in enumerate(rule['antecedent']):
                    if j > 0:
                        print(" AND ", end="")
                    
                    fuzzy_term = FuzzyDecisionTree.extract_term(fuzzy_set)
                    print(f"{feature_name} IS {fuzzy_term}", end="")
            else:
                print("  IF (root)", end="")
                
            # Consequent
            class_name = rule['class_name']
            confidence = rule['confidence']
            print(f" THEN class = {class_name} (confidence: {confidence:.2f})")
            
            print()


    @staticmethod
    def extract_term(fuzzy_set):
        """Extract linguistic term from a fuzzy set"""
        if hasattr(fuzzy_set, 'term'):
            return fuzzy_set.term
        str_rep = str(fuzzy_set)
        if "term='" in str_rep:
            start = str_rep.find("term='") + 6
            end = str_rep.find("'", start)
            if end > start:
                return str_rep[start:end]
        

        return str_rep
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on a test set
        
        Args:
            X: Feature vectors
            y: Label vector
            
        Returns:
            Dict with evaluation metrics
        """
        y_pred = self.predict(X)
        
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        
        return {
            'accuracy': acc,
            'f1_score': f1
        }
    

                         #### ----------------- VISUALIZZAZIONE ALBERO, WORK IN PROGRESS ------------------------- ####


                         
    
    # def visualize(self, max_depth=3):
    #     """
    #     Visualizza l'albero fino a una certa profondità
        
    #     Args:
    #         max_depth: Profondità massima da visualizzare
    #     """
    #     if self.root is None:
    #         print("L'albero non è stato addestrato")
    #         return
            
    #     print(f"Visualizzazione dell'albero (profondità max: {max_depth}):")
    #     self._visualize_recursive(self.root, "", True, 0, max_depth)
    
    # def _visualize_recursive(self, node: FuzzyNode, prefix: str, is_last: bool, depth: int, max_depth: int):
    #     """
    #     Visualizza ricorsivamente l'albero
        
    #     Args:
    #         node: Nodo corrente
    #         prefix: Prefisso per l'indentazione
    #         is_last: Se il nodo è l'ultimo figlio del genitore
    #         depth: Profondità corrente
    #         max_depth: Profondità massima da visualizzare
    #     """
    #     # Simboli per la visualizzazione dell'albero
    #     branch = "└── " if is_last else "├── "
        
    #     # Stampa il nodo corrente
    #     if node.is_leaf:
    #         predicted_class = np.argmax(node.class_distribution)
    #         confidence = node.class_distribution[predicted_class]
    #         print(f"{prefix}{branch}Foglia: classe={self.class_names[predicted_class]} (conf={confidence:.2f})")
    #     else:
    #         if node.feature >= 0:  # Non è il nodo radice
    #             print(f"{prefix}{branch}{self.feature_names[node.feature]}")
    #         else:
    #             print(f"{prefix}{branch}Radice")
        
    #     # Se abbiamo raggiunto la profondità massima o non ci sono figli, ci fermiamo
    #     if depth >= max_depth or len(node.children) == 0:
    #         return
            
    #     # Simbolo per l'indentazione dei figli
    #     extension = "    " if is_last else "│   "
        
    #     # Visualizza i figli
    #     for i, child in enumerate(node.children):
    #         is_last_child = (i == len(node.children) - 1)
    #         if child.fuzzy_set:
    #             fuzzy_term = FuzzyDecisionTree.extract_term(child.fuzzy_set)
    #             print(f"{prefix}{extension}└── È {fuzzy_term}")
    #             print(f"{prefix}{extension}│")
    #         self._visualize_recursive(child, prefix + extension, is_last_child, depth + 1, max_depth)



    # def plot_tree(self, figsize=(15, 10), dpi=100, fontsize=9,show=True):
    #     """
    #     Visualizza graficamente l'albero decisionale fuzzy utilizzando matplotlib
        
    #     Args:
    #         figsize: Dimensione della figura (larghezza, altezza) in pollici
    #         dpi: Risoluzione in dots per inch
    #         fontsize: Dimensione del font per le etichette
    #     """
    #     if self.root is None:
    #         print("L'albero non è stato addestrato")
    #         return
            
    #     # Adattiamo la dimensione della figura in base alla complessità dell'albero
    #     max_depth = self._get_max_depth(self.root)
    #     total_leaves = self._count_leaves(self.root)

            
    #     # Calcolo dinamico della dimensione della figura in base alla complessità dell'albero
    #     width_per_leaf = 2.5  # Unità di larghezza per foglia
    #     height_per_level = 2.0  # Unità di altezza per livello
        
    #     # Aggiungiamo un fattore di scala per alberi molto larghi o profondi
    #     scale_factor = 1.0
    #     if total_leaves > 20:
    #         scale_factor = 1.2
    #     if total_leaves > 40:
    #         scale_factor = 1.5
    #     if max_depth > 5:
    #         scale_factor *= 1.2
        
    #         # Calcoliamo le dimensioni finali
    #     adjusted_width = max(15, width_per_leaf * total_leaves * scale_factor)
    #     adjusted_height = max(10, height_per_level * max_depth * scale_factor)
        
    #     # Limitiamo le dimensioni massime per praticità
    #     adjusted_width = min(50, adjusted_width)
    #     adjusted_height = min(35, adjusted_height)
        
    #     # Creiamo la figura con dimensioni adattate
    #     plt.figure(figsize=(adjusted_width, adjusted_height), dpi=dpi)
        
    #     # Creiamo margini ampi nella figura per evitare sovrapposizioni
    #     plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Aumentato il margine superiore
        
    #     # Aggiungiamo titolo e legenda già qui, prima di disegnare l'albero
    #     plt.title("Fuzzy Decision Tree", fontsize=fontsize+6, pad=30)  # Incrementato il padding del titolo
    #     plt.axis('off')  # Nascondiamo gli assi

    #     # Dizionario per memorizzare la posizione di ogni nodo
    #     positions = {}
        
    #     # Calcoliamo le posizioni dei nodi
    #     self._compute_node_positions(self.root, positions, max_depth, total_leaves)

    #     margin = 0.05
    #     plt.xlim(-margin, 1 + margin)
    #     plt.ylim(-margin, 1 + margin)
        
    #     # Disegniamo l'albero
    #     self._draw_tree(self.root, positions, fontsize)
        
        
    #     # Creiamo la legenda per le classi
    #     handles = []
    #     for i, class_name in enumerate(self.class_names):
    #         handle = plt.Line2D([0], [0], color=plt.cm.tab10(i), lw=0, marker='o', 
    #                             markersize=10, label=class_name)
    #         handles.append(handle)
        
    #     plt.legend(handles=handles, loc='upper right', fontsize=fontsize-1)
        
    #     # Mostriamo il grafico con più spazio tra i contenuti
    #     plt.tight_layout(pad=4.0)
    #     # Mostriamo il grafico solo se richiesto
    #     if show:
    #         plt.show()
        
    #     # Ritorniamo la figura per poterla manipolare (es. salvare) in seguito
    #     return plt.gcf()

    # def _compute_node_positions(self, node, positions, max_depth, total_leaves, x_min=0, x_max=1, y=0):
    #     """
    #     Calcola le posizioni (x, y) di ogni nodo nell'albero con più spazio
    #     """
    #     if node is None:
    #         return
            
    #     # Il nodo corrente ha questa posizione
    #     positions[node.id] = ((x_min + x_max) / 2, 1-y)
        
    #     # Se il nodo è una foglia o non ha figli, ci fermiamo
    #     if node.is_leaf or len(node.children) == 0:
    #         return
            
    #     # Aumentiamo drasticamente lo spazio tra i livelli per alberi più complessi
    #     y_factor = max(1.5, 3.0 - (max_depth * 0.2))  # Riduzione progressiva per alberi più profondi
    #     level_height = 1.0 / (max_depth * y_factor + 1)
        
    #     # Aumentiamo significativamente il margine orizzontale tra i nodi figli
    #     margin = min(0.2, 0.05 * len(node.children))  # Margine adattivo basato sul numero di figli
        
    #     # Per alberi con molte foglie, usiamo una strategia di spaziatura variabile
    #     if node.depth > 1 and len(node.children) > 4:
    #         # Distribuzione non lineare per alberi ampi
    #         child_positions = []
    #         for i in range(len(node.children)):
    #             # Usiamo una distribuzione che dà più spazio ai nodi laterali
    #             pos = (i / (len(node.children) - 1)) ** 0.8  # Esponente < 1 per espandere i nodi esterni
    #             child_positions.append(x_min + (x_max - x_min) * pos)
    #     else:
    #         # Spaziatura uniforme per alberi piccoli
    #         effective_width = (x_max - x_min) * (1 - margin * (len(node.children) - 1))
    #         x_step = effective_width / len(node.children) if len(node.children) > 0 else 0
    #         child_positions = [x_min + i * (x_step + margin * (x_max - x_min) / len(node.children)) 
    #                         for i in range(len(node.children))]
        
    #     # Chiamata ricorsiva per ogni figlio
    #     for i, child in enumerate(node.children):
    #         child_x_min = child_positions[i]
    #         child_x_max = child_positions[i] + (x_step if len(node.children) <= 4 else 
    #                                         (child_positions[i+1] - child_positions[i]) if i < len(node.children) - 1 
    #                                         else (x_max - child_positions[i]))
            
    #         self._compute_node_positions(
    #             child, positions, max_depth, total_leaves,
    #             child_x_min, child_x_max, y + level_height
    #         )
        
    # def _get_max_depth(self, node):
    #     """Calcola la profondità massima dell'albero"""
    #     if node is None or len(node.children) == 0:
    #         return 0
    #     return 1 + max(self._get_max_depth(child) for child in node.children)

    # def _count_leaves(self, node):
    #     """Conta il numero di foglie nell'albero"""
    #     if node is None:
    #         return 0
    #     if node.is_leaf:
    #         return 1
    #     return sum(self._count_leaves(child) for child in node.children)

    # def _compute_node_positions(self, node, positions, max_depth, total_leaves, x_min=0, x_max=1, y=0):
    #     """
    #     Calcola le posizioni (x, y) di ogni nodo nell'albero
        
    #     Args:
    #         node: Nodo corrente
    #         positions: Dizionario che mapperà nodo -> (x, y)
    #         max_depth: Profondità massima dell'albero
    #         total_leaves: Numero totale di foglie
    #         x_min, x_max: Limiti dell'asse x per questo nodo
    #         y: Posizione verticale del nodo
    #     """
    #     if node is None:
    #         return
            
    #     # Il nodo corrente ha questa posizione
    #     positions[node.id] = ((x_min + x_max) / 2, 1-y)
        
    #     # Se il nodo è una foglia o non ha figli, ci fermiamo
    #     if node.is_leaf or len(node.children) == 0:
    #         return
            
    #     # Altezza di ogni livello
    #     level_height = 1.0 / (max_depth + 1)
        
    #     # Per ogni figlio, calcoliamo la sua sezione dell'asse x
    #     x_step = (x_max - x_min) / len(node.children)
        
    #     for i, child in enumerate(node.children):
    #         # Calcoliamo i limiti dell'asse x per questo figlio
    #         child_x_min = x_min + i * x_step
    #         child_x_max = child_x_min + x_step
            
    #         # Chiamata ricorsiva per il figlio
    #         self._compute_node_positions(
    #             child, positions, max_depth, total_leaves,
    #             child_x_min, child_x_max, y + level_height
    #         )

    # def _draw_tree(self, node, positions, fontsize):
    #     """
    #     Disegna l'albero usando le posizioni calcolate
        
    #     Args:
    #         node: Nodo corrente
    #         positions: Dizionario con le posizioni dei nodi
    #         fontsize: Dimensione del font per le etichette
    #     """
    #     if node is None or node.id not in positions:
    #         return
            
    #     # Posizione del nodo corrente
    #     x, y = positions[node.id]
        
    #     # Calcola il numero totale di foglie per adattare la visualizzazione
    #     total_leaves = self._count_leaves(self.root)
        
    #     # Aspetto del nodo dipende se è una foglia o un nodo interno
    #     if node.is_leaf:
    #         # Determiniamo il colore in base alla classe più probabile
    #         predicted_class = np.argmax(node.class_distribution)
    #         confidence = node.class_distribution[predicted_class]
            
    #         # Il colore rappresenta la classe predetta, l'opacità la confidenza
    #         node_color = plt.cm.tab10(predicted_class)
            
    #         # Disegniamo il nodo foglia con bordo più spesso
    #         node_size = 300 - max(0, (node.depth - 2) * 30)  # Dimensione più piccola per nodi più profondi
    #         plt.scatter(x, y, s=node_size, alpha=min(1.0, confidence + 0.3), color=node_color,
    #                 edgecolors='black', linewidths=1.5, zorder=10)
            
    #         # Etichetta con la classe predetta e distribuzione completa
    #         class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"Class_{predicted_class}"
            
    #         # Aggiungiamo informazioni sul numero di campioni
    #         samples_info = ""
    #         if hasattr(node, 'samples_count') and node.samples_count is not None:
    #             samples_info = f"Campioni: {node.samples_count}"
            
    #         # Adattiamo la quantità di informazioni in base alla dimensione dell'albero
    #         if total_leaves > 15:
    #             # Versione compatta per alberi grandi
    #             label = f"{class_name}\n({confidence:.2f})"
    #             if samples_info:
    #                 label += f"\n{samples_info}"
    #         else:
    #             # Versione dettagliata per alberi piccoli
    #             # Creiamo una rappresentazione visiva della distribuzione delle classi
    #             dist_str = ""
    #             for i, prob in enumerate(node.class_distribution):
    #                 if prob > 0.05:  # Mostriamo solo classi con probabilità significativa
    #                     class_i = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
    #                     dist_str += f"{class_i}: {prob:.2f}\n"
                
    #             label = f"{class_name}\n({confidence:.2f})\n{samples_info}"
    #             if dist_str:
    #                 label += f"\n{dist_str}"
            
    #         # Adattiamo dimensione e posizione dell'etichetta in base alla profondità
    #         vertical_offset = max(15, 30 - node.depth * 2)
            
    #         # Dimensione del font ridotta per alberi grandi
    #         font_scale = 1.0
    #         if total_leaves > 20:
    #             font_scale = 0.9
    #         if total_leaves > 35:
    #             font_scale = 0.8
            
    #         # Aggiungiamo l'etichetta
    #         plt.annotate(label, (x, y), xytext=(0, vertical_offset), textcoords='offset points',
    #                     ha='center', va='bottom', fontsize=fontsize * font_scale,
    #                     bbox=dict(boxstyle='round,pad=0.7', alpha=0.2, fc='white'))
    #     else:
    #         # Nodo interno 
    #         if node.feature >= 0:  # Non è il nodo radice
    #             # Disegniamo il nodo interno con colore basato sulla profondità
    #             depth_color = max(0.6, 0.9 - (node.depth * 0.1))
    #             node_color = (0.7, 0.9, depth_color)  # Azzurro che varia con la profondità
                
    #             # Dimensione del nodo adattiva
    #             node_size = 250 - max(0, (node.depth - 1) * 20)
                
    #             plt.scatter(x, y, s=node_size, color=node_color, edgecolors='blue', 
    #                     linewidths=1.5, zorder=10)
                
    #             # Etichetta più informativa con nome feature e statistiche
    #             feature_name = self.feature_names[node.feature]
                
    #             # Tronchiamo i nomi delle feature troppo lunghi
    #             if len(feature_name) > 20:
    #                 short_name = feature_name[:17] + "..."
    #             else:
    #                 short_name = feature_name
                
    #             # Informazioni aggiuntive
    #             samples_info = ""
    #             if hasattr(node, 'samples_count') and node.samples_count is not None:
    #                 samples_info = f"Campioni: {node.samples_count}"
                
    #             gain_info = ""
    #             if hasattr(node, 'information_gain') and node.information_gain is not None:
    #                 gain_info = f"Gain: {node.information_gain:.4f}"
                
    #             # Adattiamo l'etichetta in base alla dimensione dell'albero
    #             if total_leaves > 20:
    #                 node_label = short_name
    #                 if gain_info:
    #                     node_label += f"\n{gain_info}"
    #             else:
    #                 node_label = f"{short_name}\n{samples_info}"
    #                 if gain_info:
    #                     node_label += f"\n{gain_info}"
                
    #             # Offset verticale adattivo
    #             vertical_offset = max(15, 25 - node.depth * 2)
                
    #             plt.annotate(node_label, (x, y), xytext=(0, vertical_offset), textcoords='offset points',
    #                         ha='center', va='bottom', fontsize=fontsize * font_scale if 'font_scale' in locals() else fontsize,
    #                         bbox=dict(boxstyle='round,pad=0.7', alpha=0.2, fc='white'))
    #         else:
    #             # Nodo radice con più informazioni
    #             plt.scatter(x, y, s=300, color='lightgreen', edgecolors='darkgreen', 
    #                     linewidths=2.0, zorder=10)
                
    #             # Informazioni sul dataset
    #             samples_info = ""
    #             if hasattr(node, 'samples_count') and node.samples_count is not None:
    #                 samples_info = f"Campioni totali: {node.samples_count}"
                
    #             classes_info = f"Classi: {self.n_classes}"
                
    #             root_label = f"Root\n{samples_info}\n{classes_info}"
    #             plt.annotate(root_label, (x, y), xytext=(0, 25), textcoords='offset points',
    #                         ha='center', va='bottom', fontsize=fontsize,
    #                         bbox=dict(boxstyle='round,pad=0.7', alpha=0.2, fc='white'))
        
    #     # Disegniamo i collegamenti ai figli e processiamo ricorsivamente
    #     for child in node.children:
    #         if child.id in positions:
    #             child_x, child_y = positions[child.id]
                
    #             # Disegniamo la linea tra il nodo corrente e il figlio
    #             # Utilizziamo un gradiente di colore basato sulla profondità
    #             edge_color = 'black'
    #             edge_alpha = max(0.3, 0.7 - (node.depth * 0.1))
                
    #             plt.plot([x, child_x], [y, child_y], '-', color=edge_color, 
    #                     alpha=edge_alpha, linewidth=1.2, zorder=5)
                
    #             # Se c'è un insieme fuzzy, aggiungiamo un'etichetta sulla linea
    #             if child.fuzzy_set:
    #                 # Punto più vicino al nodo genitore (40% verso il figlio)
    #                 mid_x = x + (child_x - x) * 0.4
    #                 mid_y = y + (child_y - y) * 0.4
                    
    #                 # Per evitare sovrapposizioni calcoliamo anche un offset perpendicolare
    #                 dx = child_x - x
    #                 dy = child_y - y
    #                 length = np.sqrt(dx*dx + dy*dy)
                    
    #                 # Normalizziamo e ruotiamo di 90 gradi per ottenere un vettore perpendicolare
    #                 if length > 0:
    #                     nx, ny = dy/length, -dx/length
                        
    #                     # Aggiungiamo un offset perpendicolare più grande per alberi densi
    #                     offset = 0.01 * (1 + (len(node.children) / 10))
    #                     mid_x += nx * offset
    #                     mid_y += ny * offset
                    
    #                 # Estrai il termine linguistico
    #                 fuzzy_term = FuzzyDecisionTree.extract_term(child.fuzzy_set)
                    
    #                 # Angolo della linea per allineare il testo
    #                 angle = np.degrees(np.arctan2(child_y - y, child_x - x))
    #                 if angle > 90:
    #                     angle -= 180
    #                 if angle < -90:
    #                     angle += 180
                    
    #                 # Adattiamo le etichette in base alla dimensione dell'albero
    #                 term_fontsize = fontsize - 1
    #                 if total_leaves > 20:
    #                     term_fontsize -= 1
                    
    #                 # Aumentiamo l'opacità dello sfondo per maggiore leggibilità
    #                 bg_alpha = 0.7 if total_leaves > 15 else 0.5
                    
    #                 # Aggiungiamo l'etichetta ruotata con sfondo più visibile
    #                 plt.annotate(fuzzy_term, (mid_x, mid_y), 
    #                             ha='center', va='center', fontsize=term_fontsize,
    #                             bbox=dict(boxstyle='round,pad=0.4', alpha=bg_alpha, fc='white', ec='lightgray'),
    #                             rotation=angle)
                
    #             # Chiamata ricorsiva per i figli
    #             self._draw_tree(child, positions, fontsize)