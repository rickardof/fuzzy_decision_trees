# Fuzzy Decision Tree

Una libreria per alberi di decisione fuzzy per task di machine learning, che offre modelli di classificazione interpretabili basati sulla logica fuzzy.



## Utilizzo base

```python
from fuzzy_tree import FuzzyDecisionTree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Carica un dataset di esempio
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crea e addestra il modello
model = FuzzyDecisionTree(
    num_fuzzy_sets=3,               # Numero di insiemi fuzzy per feature
    max_depth=4,                    # Profondità massima dell'albero
    prediction_policy="max_matching"  # Metodo per le predizioni
)
model.fit(X_train, y_train, feature_names=data.feature_names)

# Esegui predizioni
y_pred = model.predict(X_test)

# Valuta il modello
metrics = model.evaluate(X_test, y_test)
print(f"Accuratezza: {metrics['accuracy']:.4f}, F1-score: {metrics['f1_score']:.4f}")

# Estrai e visualizza regole fuzzy
rules = model.extract_rules()
model.print_rules()
```

## Parametri principali

La classe `FuzzyDecisionTree` accetta i seguenti parametri:

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `num_fuzzy_sets` | Numero di insiemi fuzzy per ogni feature | 5 |
| `max_depth` | Profondità massima dell'albero (None = illimitata) | None |
| `min_samples_split` | Numero minimo di campioni per effettuare uno split | 2 |
| `min_samples_leaf` | Numero minimo di campioni per un nodo foglia | 1 |
| `gain_threshold` | Guadagno minimo di informazione per effettuare uno split | 0.001 |
| `membership_threshold` | Soglia minima di appartenenza per considerare un esempio | 0.5 |
| `prediction_policy` | Politica di predizione ('max_matching' o 'weighted') | 'max_matching' |

## Metodi principali

### fit(X, y, feature_names=None, class_names=None)

Addestra il modello sui dati forniti.

- `X`: Matrice delle feature
- `y`: Vettore delle etichette
- `feature_names`: Nomi delle feature (opzionale)
- `class_names`: Nomi delle classi (opzionale)

### predict(X)

Esegue la classificazione di nuovi esempi, restituendo le etichette predette.

### predict_proba(X)

Restituisce le probabilità di appartenenza a ciascuna classe.

### extract_rules()

Estrae regole fuzzy dall'albero di decisione, restituendo una lista di dizionari che rappresentano le regole.

### print_rules()

Stampa le regole fuzzy in un formato leggibile.

### evaluate(X, y)

Valuta il modello su un set di test, restituendo metriche di accuratezza e F1-score.

## Politiche di previsione

La libreria supporta due diverse politiche di previsione:

- **max_matching**: Sceglie il percorso con la massima attivazione nell'albero
- **weighted**: Combina le previsioni di tutti i percorsi proporzionalmente alla loro attivazione

## Componenti interni

La libreria è composta da tre moduli principali:

- `FuzzyDecisionTree`: La classe principale che implementa l'albero di decisione fuzzy
- `FuzzyNode`: Rappresenta un nodo dell'albero di decisione
- `FuzzyDiscretizer`: Crea partizioni fuzzy per feature continue

## Esempio: Personalizzazione del modello

```python
from fuzzy_tree import FuzzyDecisionTree
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Carica il dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crea un modello con configurazione personalizzata
model = FuzzyDecisionTree(
    num_fuzzy_sets=5,             # 5 insiemi fuzzy per feature
    max_depth=4,                  # Limita profondità a 4 livelli
    min_samples_split=10,         # Richiede almeno 10 campioni per effettuare uno split
    gain_threshold=0.01,          # Guadagno di informazione minimo più alto
    prediction_policy="weighted"  # Usa predizione ponderata
)

# Addestra il modello
model.fit(X_train, y_train, 
          feature_names=data.feature_names,
          class_names=['benigno', 'maligno'])

# Valuta e stampa le regole
print(model.evaluate(X_test, y_test))
model.print_rules()
```

## Requisiti

- Python 3.6+
- numpy
- simpful (per la rappresentazione degli insiemi fuzzy)
- scikit-learn
- matplotlib 