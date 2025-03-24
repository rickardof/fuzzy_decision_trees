# src/config.py
from typing import Final

# Configurazione generale
RANDOM_STATE: Final[int] = 19
SHUFFLE: Final[bool] = True
NUMBER_OF_ITERATIONS: Final[int] = 5

# Parametri dell'albero fuzzy
NUM_FUZZY_SETS: Final[int] = 5  # Numero di insiemi fuzzy per feature
MAX_DEPTH: Final[int] = None    # Profondità massima (None = illimitata)
MIN_SAMPLES_SPLIT_RATIO: Final[float] = 0.1  # Minimo campioni per split

# Parametri federazione
MAX_NUMBER_ROUNDS: Final[int] = 100  # Numero massimo di round di federazione

# Soglie
GAIN_THRESHOLD: Final[float] = 0.0001  # Soglia minima per information gain
MEMBERSHIP_THRESHOLD: Final[float] = 0.5  # Soglia per l'appartenenza fuzzy