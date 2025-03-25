import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
                 prediction_policy: str = "max_matching"):  # Nuovo parametro
        """
        Inizializza il classificatore fuzzy decision tree
        
        Args:
            num_fuzzy_sets: Numero di insiemi fuzzy per ogni feature
            max_depth: Profondità massima dell'albero (None = illimitata)
            min_samples_split: Numero minimo di campioni per effettuare uno split
            min_samples_leaf: Numero minimo di campioni per un nodo foglia
            gain_threshold: Guadagno informativo minimo per effettuare uno split
            membership_threshold: Soglia minima di appartenenza per considerare un esempio
            prediction_policy: Politica di predizione ('max_matching' o 'weighted')
        """
        # Validazione dei parametri
        if num_fuzzy_sets <= 0:
            raise ValueError("num_fuzzy_sets deve essere positivo")
        if membership_threshold < 0 or membership_threshold > 1:
            raise ValueError("membership_threshold deve essere nell'intervallo [0,1]")
        
        # Verificare che la policy di predizione sia valida
        valid_policies = ["max_matching", "weighted"]
        if prediction_policy not in valid_policies:
            raise ValueError(f"prediction_policy deve essere uno di {valid_policies}")
        
        self.num_fuzzy_sets = num_fuzzy_sets
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.gain_threshold = gain_threshold
        self.membership_threshold = membership_threshold
        self.prediction_policy = prediction_policy  # Nuova proprietà
        
        # Inizializziamo l'albero
        self.root = None
        self.fuzzy_sets = None
        self.feature_names = None
        self.n_features = None
        self.n_classes = None
        self.class_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None, class_names=None):
        """
        Addestra l'albero decisionale fuzzy
        
        Args:
            X: Matrice delle feature
            y: Vettore delle etichette
            feature_names: Nomi delle feature (opzionale)
            class_names: Nomi delle classi (opzionale)
        """
        # Reset dell'ID dei nodi
        FuzzyNode.reset_node_id()

            # controlli di validazione
        if X is None or y is None:
            raise ValueError("X e y non possono essere None")
        if len(X) != len(y):
            raise ValueError("X e y devono avere lo stesso numero di campioni")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X contiene valori NaN o infiniti")
        
        self.y_train= y.copy()
        
        # Salvataggio delle informazioni sui dati
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
        
        # Creazione degli insiemi fuzzy per ogni feature
        self._create_fuzzy_partitions(X)
        
        # Creazione del nodo radice
        self.root = FuzzyNode(feature=-1, feature_name="Root")
        
        # Chiamata ricorsiva per costruire l'albero
        self._build_tree(self.root, X, y, depth=0)
        
        return self
    
    def _create_fuzzy_partitions(self, X: np.ndarray):
        """
        Crea le partizioni fuzzy per ogni feature
        
        Args:
            X: Matrice delle feature
        """
        # Utilizziamo il FuzzyDiscretizer per ottenere i punti di divisione
        discretizer = FuzzyDiscretizer(self.num_fuzzy_sets, method="uniform")
        features_to_discretize = [True] * self.n_features
        splits = discretizer.run(X, features_to_discretize)
        
        # Creiamo gli insiemi fuzzy per ogni feature
        self.fuzzy_sets = {}  #dict
        for i in range(self.n_features):
            if len(splits[i]) > 0:  #if there are splits in the feature
                self.fuzzy_sets[i] = create_triangular_fuzzy_sets(splits[i])
            else:
                self.fuzzy_sets[i] = []
    
    def _build_tree(self, node: FuzzyNode, X: np.ndarray, y: np.ndarray, depth: int):
        """
        Costruisce ricorsivamente l'albero decisionale fuzzy
        
        Args:
            node: Nodo corrente
            X: Matrice delle feature
            y: Vettore delle etichette
            depth: Profondità corrente
        """
        # Controllo di sicurezza per la ricorsione
        if depth > 100:  # Limite fissato a 100 per sicurezza
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return

        n_samples = X.shape[0]   #quanti esempi raggiungono questo nodo
        node.samples_count = n_samples
        
        # Controllo dei criteri di arresto specificati all'inizio
        if (self.max_depth is not None and depth >= self.max_depth) or \
        n_samples < self.min_samples_split or \
        len(np.unique(y)) == 1:
            # Creiamo un nodo foglia
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return
            
        # Troviamo il migliore split
        best_feature, best_gain = self._find_best_split(X, y)


        #-----------attualmente non c'è un controllo per impedire la stessa feature in nodi diversi ------------


        
        # Se non troviamo uno split valido, creiamo una foglia
        if best_feature is None or best_gain <= self.gain_threshold:
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return
        
        # Impostiamo la feature scelta per questo nodo
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature] if self.feature_names is not None else f"Feature_{best_feature}"
        
        # Controllo: se i valori della feature sono troppo simili, creiamo una foglia
        feature_values = X[:, best_feature]
        if np.max(feature_values) - np.min(feature_values) < 1e-6:
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
            return
        
        # Creiamo un fingerprint dei dati per rilevare cicli
        data_hash = hash(str(X.shape) + str(np.sum(X)) + str(np.sum(y)))
        if hasattr(self, '_data_hashes'):
            if data_hash in self._data_hashes:
                # Abbiamo rilevato un ciclo potenziale
                class_distribution = self._compute_class_distribution(y)
                node.mark_as_leaf(class_distribution)
                return
            self._data_hashes.add(data_hash)
        else:
            self._data_hashes = {data_hash}
        
        # Per ogni insieme fuzzy, creiamo un nodo figlio
        for fs_idx, fuzzy_set in enumerate(self.fuzzy_sets[best_feature]):
            # Calcoliamo i gradi di appartenenza degli esempi all'insieme fuzzy
            membership_degrees = np.array([
                fuzzy_set.get_value(x[best_feature]) for x in X
            ])
            
            # Filtriamo gli esempi con appartenenza sopra la soglia (default è 0.5)
            valid_samples = membership_degrees >= self.membership_threshold
            X_child = X[valid_samples]
            y_child = y[valid_samples]
            
            # Se non ci sono campioni validi, saltiamo questo figlio
            if len(X_child) == 0:
                continue
            
            # Creiamo un nuovo nodo figlio
            child_node = FuzzyNode(
                feature=-1,  # Sarà deciso durante la costruzione ricorsiva
                fuzzy_set=fuzzy_set,
                depth=depth + 1,
                parent=node
            )
            
            # Aggiungiamo il figlio al nodo corrente
            node.add_child(child_node)
            
            # Costruiamo ricorsivamente il sottoalbero
            self._build_tree(child_node, X_child, y_child, depth + 1)
        
        # Se dopo tutto non abbiamo figli, rendiamo questo nodo una foglia
        if len(node.children) == 0:
            class_distribution = self._compute_class_distribution(y)
            node.mark_as_leaf(class_distribution)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Trova la feature con il miglior guadagno informativo
        
        Args:
            X: Matrice delle feature
            y: Vettore delle etichette
            
        Returns:
            Indice della feature migliore e il guadagno informativo
        """
        best_feature = None
        best_gain = -float('inf')
        
        # Entropia iniziale (prima dello split)
        parent_entropy = self._entropy(y)
        
        for feature in range(self.n_features):
            # Calcolo del guadagno informativo per questa feature
            gain = self._information_gain(X, y, feature, parent_entropy)
            
            if gain > best_gain:
                best_feature = feature
                best_gain = gain
                
        return best_feature, best_gain
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calcola l'entropia di Shannon
        
        Args:
            y: Vettore delle etichette
            
        Returns:
            Entropia
        """
        n_samples = len(y)
        if n_samples == 0:
            return 0
        
        counts = np.bincount(y.astype(int))
        proportions = counts[counts > 0] / n_samples
        
        return -np.sum(proportions * np.log2(proportions))
    
    def _fuzzy_entropy(self, y: np.ndarray, membership_degrees: np.ndarray) -> float:
        """
        Calcola l'entropia fuzzy
        
        Args:
            y: Vettore delle etichette
            membership_degrees: Gradi di appartenenza degli esempi
            
        Returns:
            Entropia fuzzy
        """
        classes = np.unique(y)
        total_membership = np.sum(membership_degrees)
        
        if total_membership == 0:
            return 0
        
        entropy = 0 #inizializzazione

        for c in classes:
            class_membership = np.sum(membership_degrees[y == c])
            if class_membership > 0:
                p = class_membership / total_membership
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _information_gain(self, X: np.ndarray, y: np.ndarray, feature: int, 
                         parent_entropy: Optional[float] = None) -> float:
        """
        Calcola il guadagno informativo fuzzy per una feature
        
        Args:
            X: Matrice delle feature
            y: Vettore delle etichette
            feature: Indice della feature da valutare
            parent_entropy: Entropia del nodo genitore (se nota)
            
        Returns:
            Guadagno informativo fuzzy
        """
        if parent_entropy is None:
            parent_entropy = self._entropy(y)
            
        # Calcolo dell'entropia pesata dopo lo split
        weighted_entropy = 0
        total_samples = len(y)
        
        for fuzzy_set in self.fuzzy_sets[feature]:  #per ogni insieme fuzzy associato alla feature da analizzare...

            # Calcolo dei gradi di appartenenza per questo insieme fuzzy
            membership_degrees = np.array([
                fuzzy_set.get_value(x[feature]) for x in X
            ])
            
            # Calcolo dell'entropia fuzzy per questo insieme
            fuzzy_entropy = self._fuzzy_entropy(y, membership_degrees)
            
            # Peso basato sulla somma dei gradi di appartenenza
            weight = np.sum(membership_degrees) / total_samples
            
            weighted_entropy += weight * fuzzy_entropy
            
        # Guadagno informativo
        return parent_entropy - weighted_entropy
    
    def _compute_class_distribution(self, y: np.ndarray) -> np.ndarray:
        """
        Calcola la distribuzione di probabilità delle classi
        
        Args:
            y: Vettore delle etichette
            
        Returns:
            Array di probabilità [p(class_0), p(class_1), ...]
        """
        if len(y) == 0:
            # Se non ci sono esempi, distribuzione uniforme
            return np.ones(self.n_classes) / self.n_classes
            
        # Conteggio delle classi
        counts = np.bincount(y.astype(int), minlength=self.n_classes)
        
        # Normalizzazione per ottenere probabilità
        return counts / len(y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classifica nuovi esempi usando maximum matching
        
        Args:
            X: Matrice delle feature o singolo esempio
            
        Returns:
            Etichette predette
        """
        if self.root is None:
            raise ValueError("L'albero non è stato addestrato")
        
        #Se l'input ha un solo esempio, lo riformattiamo come una matrice con una sola riga
        if len(X.shape) == 1:
            X = X.reshape(1,-1)
            
        # Frammentaiamo le predizioni per batch con dataset molto grandi
        batch_size = 10000  # Scelto arbitrariamente
        if len(X) > batch_size:
            predictions = []
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_preds = np.zeros(len(batch), dtype=int)
                batch_probas = self.predict_proba(batch)
                batch_preds = np.argmax(batch_probas, axis=1)
                
                predictions.append(batch_preds)
            return np.concatenate(predictions)
        
        # Predizioni normali
        return np.array([np.argmax(self.predict_proba(sample.reshape(1, -1))[0]) for sample in X])
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Restituisce le probabilità delle classi usando maximum matching
        
        Args:
            X: Matrice delle feature o singolo esempio
            
        Returns:
            Array di probabilità delle classi
        """
        if self.root is None:
            raise ValueError("L'albero non è stato addestrato")
            
        if len(X.shape) == 1:
            # Singolo esempio
            X = X.reshape(1, -1)

             
        #Verifichiamo le dimensioni
        if X.shape[1] != self.n_features:
            raise ValueError(f"Numero di feature errato: atteso {self.n_features}, ricevuto {X.shape[1]}")
            
        probas = []
        for sample in X:
            # predict_single_example trova il percorso con attivazione massima e restituisce la distribuzione della foglia risultante
            _, class_distribution = self._predict_single_example(sample, self.root)
            probas.append(class_distribution)
            
        return np.array(probas)
    
    def _predict_single_example(self, x: np.ndarray, node: FuzzyNode, current_activation: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Dispatcher per il metodo di predizione appropriato in base alla policy
        
        Args:
            x: Vettore delle feature
            node: Nodo corrente
            current_activation: Attivazione accumulata fino a questo nodo
            
        Returns:
            Una tupla (max_activation, best_leaf_distribution)
        """
        if self.prediction_policy == "max_matching":
            return self._predict_max_matching(x, node, current_activation)
        elif self.prediction_policy == "weighted":
            return self._predict_weighted(x, node, current_activation)
        else:
            # Di default usiamo max_matching
            return self._predict_max_matching(x, node, current_activation)
    
    def _predict_max_matching(self, x: np.ndarray, node: FuzzyNode, current_activation: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Trova il percorso con attivazione massima per un singolo esempio (maximum matching policy)
        
        Args:
            x: Vettore delle feature
            node: Nodo corrente
            current_activation: Attivazione accumulata fino a questo nodo
            
        Returns:
            Una tupla (max_activation, best_leaf_distribution)
        """
        

        # Caso base: nodo foglia
        if node.is_leaf:
            return current_activation, node.class_distribution
            
        # Se il nodo non ha figli, trattiamolo come una foglia con distribuzione uniforme (caso raro)
        if len(node.children) == 0:
            return current_activation, np.ones(self.n_classes) / self.n_classes
            
        # Per ogni figlio, calcola l'attivazione e trova il percorso migliore
        max_activation = 0.0
        best_leaf_distribution = None  # Inizializzato a None per rilevare se non troviamo percorsi
        
        for child in node.children:
            # Calcola il grado di appartenenza a questo insieme fuzzy
            membership = child.fuzzy_set.get_value(x[node.feature])

            membership = np.clip(membership, 1e-10, 1.0)
            
            # Calcola l'attivazione di questo ramo (usando il prodotto come t-norm per AND)
            branch_activation = current_activation * membership

            
            # Se l'attivazione è significativa, esplora questo ramo
            if branch_activation > 0:
                # Chiamata ricorsiva
                child_activation, child_distribution = self._predict_max_matching(
                    x, child, branch_activation
                )
                
                # Se questo percorso ha attivazione maggiore, aggiorna i valori massimi
                if child_activation > max_activation:
                    max_activation = child_activation
                    best_leaf_distribution = child_distribution
        
        # Se non abbiamo trovato alcun percorso valido
        if best_leaf_distribution is None:
            # Usa la distribuzione di classe a priori se disponibile, altrimenti distribuzione uniforme
            if hasattr(self, 'y_train'):
                prior_distribution = self._compute_class_distribution(self.y_train)
                return 0.0, prior_distribution
            else:
                return 0.0, np.ones(self.n_classes) / self.n_classes
        
        return max_activation, best_leaf_distribution
    
    def _predict_weighted(self, x: np.ndarray, node: FuzzyNode, current_activation: float = 1.0) -> Tuple[float, np.ndarray]:
        """
        Combina le predizioni di tutti i percorsi proporzionalmente alla loro attivazione
        
        Args:
            x: Vettore delle feature
            node: Nodo corrente
            current_activation: Attivazione accumulata fino a questo nodo
            
        Returns:
            Una tupla (total_activation, weighted_distribution)
        """
        

        # Caso base: nodo foglia
        if node.is_leaf:
            return current_activation, node.class_distribution
            
        # Se il nodo non ha figli, trattiamolo come una foglia con distribuzione uniforme
        if len(node.children) == 0:
            return current_activation, np.ones(self.n_classes) / self.n_classes
            
        # Accumulatori per la combinazione pesata
        total_activation = 0.0
        combined_distribution = np.zeros(self.n_classes)
        
        # Esplora tutti i figli con attivazione > 0
        for child in node.children:
            # Calcola il grado di appartenenza a questo insieme fuzzy
            membership = child.fuzzy_set.get_value(x[node.feature])
            
            membership = np.clip(membership, 1e-10, 1.0)

            # Calcola l'attivazione di questo ramo
            branch_activation = current_activation * membership
            
            # Se l'attivazione è significativa, esplora questo ramo
            if branch_activation > 0:
                child_activation, child_distribution = self._predict_weighted(
                    x, child, branch_activation
                )
                
                # Aggiungi questa distribuzione pesata per l'attivazione
                combined_distribution += child_activation * child_distribution
                total_activation += child_activation
        
        # Normalizza la distribuzione combinata
        if total_activation > 0:
            combined_distribution /= total_activation
        else:
            # Se nessun percorso ha attivazione > 0, usa la distribuzione a priori
            if hasattr(self, 'y_train'):
                combined_distribution = self._compute_class_distribution(self.y_train)
            else:
                combined_distribution = np.ones(self.n_classes) / self.n_classes
            
        return total_activation, combined_distribution
    
    def extract_rules(self) -> List[Dict]:
        """
        Estrae le regole fuzzy dall'albero
        
        Returns:
            Lista di regole fuzzy in formato dizionario
        """
        rules = []
        self._extract_rules_recursive(self.root, [], rules)
        return rules
    
    def _extract_rules_recursive(self, node: FuzzyNode, path: List, rules: List):
        """
        Estrae ricorsivamente le regole fuzzy dall'albero
        
        Args:
            node: Nodo corrente
            path: Percorso corrente (lista di tuple (feature, fuzzy_set))
            rules: Lista di regole (modificata in-place)
        """
        #caso limite

        if node.is_leaf:
            # Creiamo una regola per questo percorso
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
            # Aggiungiamo questa condizione al percorso
            condition = (node.feature, node.feature_name, child.fuzzy_set)
            path.append(condition)
            
            # Chiamata ricorsiva
            self._extract_rules_recursive(child, path, rules)
            
            # Rimuoviamo la condizione dal percorso
            path.pop()
    
    def print_rules(self):
        """Stampa le regole fuzzy in formato leggibile"""
        rules = self.extract_rules()
    
        print(f"Regole dell'albero decisionale fuzzy ({len(rules)} regole):")
        print("=" * 80)
        
        for i, rule in enumerate(rules):
            print(f"Regola {i + 1}:")
            
            # Antecedente
            if len(rule['antecedent']) > 0:
                print("  SE ", end="")
                for j, (feature_idx, feature_name, fuzzy_set) in enumerate(rule['antecedent']):
                    if j > 0:
                        print(" E ", end="")
                    
                    fuzzy_term = FuzzyDecisionTree.extract_term(fuzzy_set)
                    print(f"{feature_name} È {fuzzy_term}", end="")
            else:
                print("  SE (radice)", end="")
                
            # Conseguente
            class_name = rule['class_name']
            confidence = rule['confidence']
            print(f" ALLORA classe = {class_name} (confidenza: {confidence:.2f})")
            
            print()


    @staticmethod
    def extract_term(fuzzy_set):
        """Estrae il termine linguistico da un insieme fuzzy"""
        # Prima prova ad accedere all'attributo term direttamente
        if hasattr(fuzzy_set, 'term'):
            return fuzzy_set.term
        
        # Se fallisce, prova a estrarlo dalla rappresentazione in stringa
        str_rep = str(fuzzy_set)
        if "term='" in str_rep:
            # Estrai il valore tra term=' e il successivo '
            start = str_rep.find("term='") + 6
            end = str_rep.find("'", start)
            if end > start:
                return str_rep[start:end]
        
        # In caso di fallimento, restituisci la rappresentazione in stringa
        return str_rep
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Valuta il modello su un set di test
        
        Args:
            X: Matrice delle feature
            y: Vettore delle etichette vere
            
        Returns:
            Dizionario con le metriche di valutazione
        """
        y_pred = self.predict(X)
        
        # Calcolo delle metriche
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average='macro')
        
        return {
            'accuracy': acc,
            'f1_score': f1
        }
    
    def visualize(self, max_depth=3):
        """
        Visualizza l'albero fino a una certa profondità
        
        Args:
            max_depth: Profondità massima da visualizzare
        """
        if self.root is None:
            print("L'albero non è stato addestrato")
            return
            
        print(f"Visualizzazione dell'albero (profondità max: {max_depth}):")
        self._visualize_recursive(self.root, "", True, 0, max_depth)
    
    def _visualize_recursive(self, node: FuzzyNode, prefix: str, is_last: bool, depth: int, max_depth: int):
        """
        Visualizza ricorsivamente l'albero
        
        Args:
            node: Nodo corrente
            prefix: Prefisso per l'indentazione
            is_last: Se il nodo è l'ultimo figlio del genitore
            depth: Profondità corrente
            max_depth: Profondità massima da visualizzare
        """
        # Simboli per la visualizzazione dell'albero
        branch = "└── " if is_last else "├── "
        
        # Stampa il nodo corrente
        if node.is_leaf:
            predicted_class = np.argmax(node.class_distribution)
            confidence = node.class_distribution[predicted_class]
            print(f"{prefix}{branch}Foglia: classe={self.class_names[predicted_class]} (conf={confidence:.2f})")
        else:
            if node.feature >= 0:  # Non è il nodo radice
                print(f"{prefix}{branch}{self.feature_names[node.feature]}")
            else:
                print(f"{prefix}{branch}Radice")
        
        # Se abbiamo raggiunto la profondità massima o non ci sono figli, ci fermiamo
        if depth >= max_depth or len(node.children) == 0:
            return
            
        # Simbolo per l'indentazione dei figli
        extension = "    " if is_last else "│   "
        
        # Visualizza i figli
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            if child.fuzzy_set:
                fuzzy_term = FuzzyDecisionTree.extract_term(child.fuzzy_set)
                print(f"{prefix}{extension}└── È {fuzzy_term}")
                print(f"{prefix}{extension}│")
            self._visualize_recursive(child, prefix + extension, is_last_child, depth + 1, max_depth)