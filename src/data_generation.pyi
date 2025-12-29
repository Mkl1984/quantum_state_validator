"""
Fichier Stub de Type (PEP 484) pour data_generation.py
========================================================

Ce fichier fournit les annotations de types et la documentation complète
pour toutes les fonctions du module data_generation.

Auteur: Mkl Zenin
Date: 2024-11-12
"""

from typing import Tuple, Optional, Dict, Any, Literal
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.random import Generator

# ============================================================================
# TYPE ALIASES
# ============================================================================

# Alias pour les stratégies de génération d'états valides
ValidStrategy = Literal["random", "dirichlet", "basis"]

# Alias pour les stratégies de génération d'états invalides
InvalidStrategy = Literal["scaling", "noise", "direct", "mixed"]

# Alias pour un tableau d'états quantiques (array 2D de nombres complexes)
QuantumStates = np.ndarray  # shape: (n_samples, dim), dtype: complex128

# Alias pour les normes au carré
NormsSquared = np.ndarray  # shape: (n_samples,), dtype: float64

# Alias pour le DataFrame de dataset
QuantumDataFrame = pd.DataFrame

# ============================================================================
# GÉNÉRATION D'ÉTATS VALIDES (NORMALISÉS)
# ============================================================================

def generate_valid_states(
    n_samples: int,
    dim: int,
    strategy: ValidStrategy = "random",
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> QuantumStates:
    """
    Génère des états quantiques valides (normalisés: ||ψ||² = 1).

    Paramètres
    ----------
    n_samples : int
        Nombre d'états à générer (doit être > 0).

    dim : int
        Dimension de l'espace de Hilbert (nombre de coefficients par état).
        Correspond au nombre de niveaux d'énergie du système quantique.
        Exemple: dim=2 pour un qubit, dim=4 pour 2 qubits.

    strategy : ValidStrategy, default="random"
        Stratégie de génération. Options disponibles:

        - "random": Génération gaussienne + normalisation
            * Génère parties réelles et imaginaires indépendantes N(0,1)
            * Normalise pour obtenir ||ψ||² = 1
            * Explore uniformément l'espace des états quantiques
            * **Recommandé pour usage général**

        - "dirichlet": Distribution de Dirichlet pour les probabilités
            * Génère les probabilités |c_i|² via Dirichlet(alpha)
            * Phases uniformes dans [0, 2π]
            * Paramètre alpha contrôle la dispersion:
                - alpha = 1: distribution uniforme
                - alpha > 1: probabilités équilibrées
                - alpha < 1: pics (probabilités concentrées)
            * Utile pour tester différentes distributions de probabilités

        - "basis": États purs de la base canonique
            * Génère des états du type |0⟩, |1⟩, ..., |dim-1⟩
            * Chaque état a un seul coefficient = 1, les autres = 0
            * Utile pour avoir des cas triviaux dans le dataset

    alpha : float, default=1.0
        Paramètre de concentration pour la stratégie "dirichlet".
        Ignoré pour les autres stratégies.
        Valeurs typiques: [0.1, 10.0]

    seed : int, optional
        Graine pour le générateur aléatoire (reproductibilité).
        Si None, utilise une graine aléatoire.

    Retourne
    --------
    states : QuantumStates
        Tableau NumPy de shape (n_samples, dim) et dtype complex128.
        Chaque ligne est un état quantique normalisé: Σ|c_i|² = 1.

        Format:
            states[i, j] = c_j du i-ème état

        Exemple pour dim=3:
            states[0] = [c0, c1, c2]  où |c0|² + |c1|² + |c2|² = 1

    Lève
    ----
    ValueError
        Si n_samples <= 0
        Si dim <= 0
        Si strategy n'est pas dans ["random", "dirichlet", "basis"]

    Exemples
    --------
    >>> # Génère 100 états de dimension 4 avec stratégie random
    >>> states = generate_valid_states(n_samples=100, dim=4, strategy="random", seed=42)
    >>> states.shape
    (100, 4)
    >>> np.allclose(np.sum(np.abs(states)**2, axis=1), 1.0)
    True

    >>> # États de la base canonique
    >>> states_basis = generate_valid_states(n_samples=5, dim=3, strategy="basis", seed=42)
    >>> # Peut produire: |0⟩, |1⟩, |2⟩, |0⟩, |1⟩

    Notes
    -----
    - Les états générés sont des vecteurs de l'espace de Hilbert ℂ^dim
    - La normalisation ||ψ||² = 1 est garantie numériquement (tolérance ~1e-15)
    - Pour des qubits: utiliser dim = 2^n où n est le nombre de qubits

    Voir aussi
    ---------
    verify_normalization : Vérifie la normalisation des états
    get_strategy_info : Obtient la description des stratégies
    generate_invalid_states : Génère des états non normalisés
    """
    ...

def verify_normalization(
    states: QuantumStates, tolerance: float = 1e-6
) -> Tuple[bool, NormsSquared]:
    """
    Vérifie que tous les états sont normalisés (||ψ||² ≈ 1).

    Paramètres
    ----------
    states : QuantumStates
        Tableau d'états quantiques de shape (n_samples, dim).

    tolerance : float, default=1e-6
        Tolérance absolue pour la comparaison avec 1.0.
        Un état est considéré normalisé si: |||ψ||² - 1| < tolerance

    Retourne
    --------
    all_valid : bool
        True si TOUS les états sont normalisés, False sinon.

    norms_squared : NormsSquared
        Tableau 1D contenant ||ψ||² pour chaque état.
        Shape: (n_samples,), dtype: float64

    Exemples
    --------
    >>> states = generate_valid_states(100, 4, seed=42)
    >>> is_valid, norms = verify_normalization(states)
    >>> is_valid
    True
    >>> norms
    array([1.0, 1.0, 1.0, ...])  # Tous proches de 1.0

    >>> # État invalide
    >>> invalid_state = np.array([[1+0j, 2+0j, 3+0j]])
    >>> is_valid, norms = verify_normalization(invalid_state)
    >>> is_valid
    False
    >>> norms[0]
    14.0  # 1² + 2² + 3² = 14

    Notes
    -----
    - Utilise np.allclose() pour la comparaison numérique
    - La tolérance par défaut (1e-6) est adaptée pour la plupart des usages
    """
    ...

# ============================================================================
# INFORMATIONS SUR LES STRATÉGIES VALIDES
# ============================================================================

def get_strategy_info() -> Dict[str, str]:
    """
    Retourne un dictionnaire décrivant les stratégies de génération valides.

    Retourne
    --------
    info : dict[str, str]
        Dictionnaire {nom_stratégie: description}.
        Clés: "random", "dirichlet", "basis"

    Exemples
    --------
    >>> info = get_strategy_info()
    >>> print(info["random"])
    Génération gaussienne + normalisation. Explore uniformément...
    """
    ...

def print_strategy_info() -> None:
    """
    Affiche dans la console les informations sur les stratégies valides.

    Exemples
    --------
    >>> print_strategy_info()
    ======================================================================
    STRATÉGIES DE GÉNÉRATION D'ÉTATS VALIDES
    ======================================================================

     RANDOM
       Génération gaussienne + normalisation...
    ...
    """
    ...

# ============================================================================
# GÉNÉRATION D'ÉTATS INVALIDES (NON NORMALISÉS)
# ============================================================================

def generate_invalid_states(
    n_samples: int,
    dim: int,
    strategy: InvalidStrategy = "scaling",
    scale_range: Tuple[float, float] = (0.1, 2.0),
    noise_level: float = 0.3,
    extreme_prob: float = 0.1,
    seed: Optional[int] = None,
) -> QuantumStates:
    """
    Génère des états quantiques invalides (NON normalisés: ||ψ||² ≠ 1).

    Paramètres
    ----------
    n_samples : int
        Nombre d'états à générer (doit être > 0).

    dim : int
        Dimension de l'espace de Hilbert.

    strategy : InvalidStrategy, default="scaling"
        Stratégie de génération. Options disponibles:

        - "scaling": Multiplie des états valides par un facteur k ≠ 1
            * Génère d'abord un état normalisé
            * Applique un facteur k ∈ [k_min, k_max]
            * Évite automatiquement [0.95, 1.05] pour éviter ambiguïté
            * Produit: ||ψ||² = k²
            * **Recommandé pour usage général**

        - "noise": Ajoute du bruit à des états valides sans renormaliser
            * Génère un état normalisé
            * Ajoute du bruit complexe N(0, noise_level)
            * Ne renormalise PAS après
            * Produit: états "presque valides" (utile pour robustesse)

        - "direct": Génère directement sans normalisation
            * Génère coefficients N(0, 1) sans normaliser
            * Applique un scaling aléatoire pour varier ||ψ||²
            * Produit: large distribution de normes

        - "mixed": Combine scaling, noise, direct + cas extrêmes
            * Mélange les 3 stratégies précédentes
            * Ajoute des cas extrêmes (états quasi-nuls, énormes, etc.)
            * Paramètre extreme_prob contrôle % de cas extrêmes
            * **Recommandé pour dataset final (diversité maximale)**

    scale_range : tuple[float, float], default=(0.1, 2.0)
        Intervalle [k_min, k_max] pour les facteurs de scaling.
        Utilisé par les stratégies "scaling" et "mixed".
        L'intervalle [0.95, 1.05] est automatiquement exclu.

    noise_level : float, default=0.3
        Écart-type du bruit gaussien ajouté.
        Utilisé par les stratégies "noise" et "mixed".
        Valeurs typiques: [0.1, 0.5]

    extreme_prob : float, default=0.1
        Probabilité de générer un cas extrême dans la stratégie "mixed".
        Exemple: 0.1 = 10% de cas extrêmes.
        Valeurs typiques: [0.05, 0.2]

    seed : int, optional
        Graine pour le générateur aléatoire.

    Retourne
    --------
    states : QuantumStates
        Tableau NumPy de shape (n_samples, dim) et dtype complex128.
        AUCUN état n'est normalisé: ||ψ||² ≠ 1 pour tous les états.

    Lève
    ----
    ValueError
        Si n_samples <= 0
        Si dim <= 0
        Si strategy n'est pas dans ["scaling", "noise", "direct", "mixed"]

    Exemples
    --------
    >>> # États invalides par scaling
    >>> states = generate_invalid_states(100, 4, strategy="scaling",
    ...                                   scale_range=(0.5, 1.5), seed=42)
    >>> norms = np.sum(np.abs(states)**2, axis=1)
    >>> np.any(np.isclose(norms, 1.0))
    False  # Aucun état normalisé

    >>> # Stratégie mixed (diversité maximale)
    >>> states_mixed = generate_invalid_states(1000, 4, strategy="mixed",
    ...                                         extreme_prob=0.2, seed=42)
    >>> norms = np.sum(np.abs(states_mixed)**2, axis=1)
    >>> norms.min(), norms.max()
    (0.0001, 10000.0)  # Très large plage

    Notes
    -----
    - La fonction garantit qu'aucun état n'est accidentellement normalisé
    - Si un état tombe trop proche de ||ψ||²=1, il est automatiquement rescalé
    - La stratégie "mixed" est recommandée pour créer un dataset robuste

    Voir aussi
    ---------
    generate_valid_states : Génère des états normalisés
    get_invalid_strategy_info : Obtient la description des stratégies
    """
    ...

def _generate_extreme_states(n_samples: int, dim: int, rng: Generator) -> QuantumStates:
    """
    Génère des cas extrêmes pour tester la robustesse des modèles.

    Fonction privée utilisée par generate_invalid_states(strategy="mixed").

    Types de cas extrêmes:
    - "null": États quasi-nuls (norme très petite)
    - "huge": États très grands (coefficients >> 1)
    - "unbalanced": Une composante énorme, les autres petites

    Paramètres
    ----------
    n_samples : int
        Nombre d'états extrêmes à générer.

    dim : int
        Dimension des états.

    rng : numpy.random.Generator
        Générateur aléatoire NumPy.

    Retourne
    --------
    states : QuantumStates
        Tableau d'états extrêmes de shape (n_samples, dim).
    """
    ...

def get_invalid_strategy_info() -> Dict[str, str]:
    """
    Retourne un dictionnaire décrivant les stratégies de génération invalides.

    Retourne
    --------
    info : dict[str, str]
        Dictionnaire {nom_stratégie: description}.
        Clés: "scaling", "noise", "direct", "mixed"

    Exemples
    --------
    >>> info = get_invalid_strategy_info()
    >>> print(info["mixed"])
    Combine scaling, noise, direct + cas extrêmes...
    """
    ...

def print_invalid_strategy_info() -> None:
    """
    Affiche dans la console les informations sur les stratégies invalides.

    Exemples
    --------
    >>> print_invalid_strategy_info()
    ======================================================================
    STRATÉGIES DE GÉNÉRATION D'ÉTATS INVALIDES
    ======================================================================

     SCALING
       Multiplie des états valides par un facteur k ≠ 1...
    ...
    """
    ...

# ============================================================================
# CRÉATION DU DATASET COMPLET
# ============================================================================

def create_dataset(
    n_valid: int,
    n_invalid: int,
    dim: int,
    valid_strategy: ValidStrategy = "random",
    invalid_strategy: InvalidStrategy = "mixed",
    valid_kwargs: Optional[Dict[str, Any]] = None,
    invalid_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> QuantumDataFrame:
    """
    Crée un dataset complet pour l'entraînement de modèles ML.

    Cette fonction combine états valides et invalides dans un DataFrame
    avec toutes les features nécessaires pour la classification binaire.

    Paramètres
    ----------
    n_valid : int
        Nombre d'états valides à générer (label = 1).

    n_invalid : int
        Nombre d'états invalides à générer (label = 0).

    dim : int
        Dimension de l'espace de Hilbert.
        Détermine le nombre de features: 2*dim (parties réelles + imaginaires).

    valid_strategy : ValidStrategy, default="random"
        Stratégie pour générer les états valides.
        Options: "random", "dirichlet", "basis"

    invalid_strategy : InvalidStrategy, default="mixed"
        Stratégie pour générer les états invalides.
        Options: "scaling", "noise", "direct", "mixed"

    valid_kwargs : dict, optional
        Arguments supplémentaires pour generate_valid_states().
        Exemple: {"alpha": 2.0} pour stratégie "dirichlet"

    invalid_kwargs : dict, optional
        Arguments supplémentaires pour generate_invalid_states().
        Exemple: {"extreme_prob": 0.2, "scale_range": (0.1, 3.0)}

    seed : int, optional
        Graine pour la reproductibilité.
        Si fournie, le dataset sera identique à chaque exécution.

    shuffle : bool, default=True
        Si True, mélange aléatoirement les lignes du DataFrame.
        Recommandé pour éviter un biais d'ordre (tous valides puis invalides).

    Retourne
    --------
    df : pd.DataFrame
        DataFrame avec les colonnes suivantes:

        - state_id : int
            Identifiant unique de l'état (0, 1, 2, ...)

        - c{i}_real : float
            Partie réelle du coefficient i (pour i=0..dim-1)

        - c{i}_imag : float
            Partie imaginaire du coefficient i (pour i=0..dim-1)

        - norm_squared : float
            Norme au carré ||ψ||² = Σ|c_i|²
            ≈ 1.0 pour états valides
            ≠ 1.0 pour états invalides

        - is_valid : int
            Label de classification (variable cible)
            1 = état valide (normalisé)
            0 = état invalide (non normalisé)

        Shape: (n_valid + n_invalid, 2*dim + 3)

    Exemples
    --------
    >>> # Dataset équilibré de 10k échantillons, dimension 4
    >>> df = create_dataset(
    ...     n_valid=5000,
    ...     n_invalid=5000,
    ...     dim=4,
    ...     valid_strategy="random",
    ...     invalid_strategy="mixed",
    ...     invalid_kwargs={"extreme_prob": 0.1, "scale_range": (0.1, 2.5)},
    ...     seed=42,
    ...     shuffle=True
    ... )
    >>> df.shape
    (10000, 11)  # 11 colonnes: state_id + 8 coefs + norm_squared + is_valid

    >>> # Colonnes du DataFrame
    >>> df.columns.tolist()
    ['state_id', 'c0_real', 'c0_imag', 'c1_real', 'c1_imag',
     'c2_real', 'c2_imag', 'c3_real', 'c3_imag', 'norm_squared', 'is_valid']

    >>> # Vérifier l'équilibre
    >>> df['is_valid'].value_counts()
    0    5000
    1    5000
    Name: is_valid, dtype: int64

    Notes
    -----
    - Le dataset est prêt pour sklearn.model_selection.train_test_split()
    - Les features sont les colonnes c{i}_real, c{i}_imag, norm_squared
    - La target est la colonne is_valid
    - Si shuffle=True, les state_id sont réassignés après mélange

    Voir aussi
    ---------
    save_dataset : Sauvegarde le dataset en CSV
    load_dataset : Charge un dataset depuis CSV
    print_dataset_info : Affiche les statistiques du dataset
    """
    ...

# ============================================================================
# SAUVEGARDE ET CHARGEMENT DU DATASET
# ============================================================================

def save_dataset(
    df: QuantumDataFrame,
    filename: str = "quantum_states_dataset.csv",
    data_dir: str = "data/processed",
) -> Path:
    """
    Sauvegarde le dataset dans un fichier CSV.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame à sauvegarder (créé par create_dataset()).

    filename : str, default="quantum_states_dataset.csv"
        Nom du fichier CSV.

    data_dir : str, default="data/processed"
        Dossier de destination (créé automatiquement si inexistant).

    Retourne
    --------
    filepath : Path
        Chemin complet du fichier sauvegardé.

    Exemples
    --------
    >>> df = create_dataset(5000, 5000, 4, seed=42)
    >>> path = save_dataset(df, filename="train_dataset.csv")
    Dataset sauvegardé:
       Chemin: data/processed/train_dataset.csv
       Taille: 1234.56 KB
       Lignes: 10000
       Colonnes: 11
    >>> str(path)
    'data/processed/train_dataset.csv'

    Notes
    -----
    - Utilise df.to_csv() avec index=False
    - Crée le dossier data_dir si nécessaire (mkdir parents=True)
    - Affiche la taille du fichier en KB
    """
    ...

def load_dataset(
    filename: str = "quantum_states_dataset.csv", data_dir: str = "data/processed"
) -> QuantumDataFrame:
    """
    Charge un dataset depuis un fichier CSV.

    Paramètres
    ----------
    filename : str, default="quantum_states_dataset.csv"
        Nom du fichier CSV à charger.

    data_dir : str, default="data/processed"
        Dossier contenant le fichier.

    Retourne
    --------
    df : pd.DataFrame
        DataFrame chargé avec toutes les colonnes.

    Lève
    ----
    FileNotFoundError
        Si le fichier n'existe pas.

    Exemples
    --------
    >>> df = load_dataset("train_dataset.csv")
    Dataset chargé depuis data/processed/train_dataset.csv
    Shape: (10000, 11)
    >>> df.head()
       state_id  c0_real  c0_imag  ...  norm_squared  is_valid
    0         0   0.1234   0.5678  ...      1.000000         1
    ...

    Notes
    -----
    - Utilise pd.read_csv()
    - Affiche le chemin et la shape du DataFrame chargé
    """
    ...

# ============================================================================
# INFORMATIONS ET STATISTIQUES DU DATASET
# ============================================================================

def get_dataset_info(df: QuantumDataFrame) -> Dict[str, Any]:
    """
    Extrait les informations structurées du dataset.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame du dataset.

    Retourne
    --------
    info : dict
        Dictionnaire contenant:

        - n_samples : int
            Nombre total d'échantillons

        - n_features : int
            Nombre de features (2*dim + 1)

        - dim : int
            Dimension déduite du nombre de colonnes c{i}_*

        - n_valid : int
            Nombre d'états valides (is_valid=1)

        - n_invalid : int
            Nombre d'états invalides (is_valid=0)

        - balance_ratio : float
            Ratio n_valid / n_invalid
            Idéal: ≈ 1.0 pour dataset équilibré

        - norm_stats : dict
            Statistiques de norm_squared (count, mean, std, min, max, ...)

    Exemples
    --------
    >>> df = create_dataset(5000, 5000, 4, seed=42)
    >>> info = get_dataset_info(df)
    >>> info['n_samples']
    10000
    >>> info['dim']
    4
    >>> info['balance_ratio']
    1.0
    >>> info['norm_stats']['mean']
    150.5  # Moyenne de norm_squared
    """
    ...

def print_dataset_info(df: QuantumDataFrame) -> None:
    """
    Affiche dans la console les informations détaillées du dataset.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame du dataset.

    Exemples
    --------
    >>> df = create_dataset(5000, 5000, 4, seed=42)
    >>> print_dataset_info(df)
    ======================================================================
    INFORMATIONS SUR LE DATASET
    ======================================================================

     Taille:
       Échantillons: 10000
       Features: 9
       Dimension: 4

     Distribution des classes:
       Valides (1): 5000 (50.0%)
       Invalides (0): 5000 (50.0%)
       Ratio: 1.000
        ✓ Dataset bien équilibré

     Statistiques de norm_squared:
       count   : 10000.000000
       mean    : 150.123456
       ...
    ======================================================================

    Notes
    -----
    - Utilise get_dataset_info() en interne
    - Affiche un warning si le dataset est déséquilibré (ratio loin de 1.0)
    """
    ...

# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

"""
EXEMPLES COMPLETS
=================

1. Workflow de base
-------------------
>>> import numpy as np
>>> import pandas as pd
>>> from data_generation import (
...     create_dataset, save_dataset, load_dataset, print_dataset_info
... )

>>> # Créer un dataset de 10k échantillons
>>> df = create_dataset(
...     n_valid=5000,
...     n_invalid=5000,
...     dim=4,
...     seed=42
... )

>>> # Sauvegarder
>>> save_dataset(df, filename="my_dataset.csv")

>>> # Charger
>>> df_loaded = load_dataset("my_dataset.csv")

>>> # Informations
>>> print_dataset_info(df_loaded)


2. Personnalisation avancée
----------------------------
>>> # Dataset avec stratégies spécifiques
>>> df_custom = create_dataset(
...     n_valid=3000,
...     n_invalid=7000,  # Dataset déséquilibré
...     dim=8,  # 3 qubits
...     valid_strategy="dirichlet",
...     valid_kwargs={"alpha": 0.5},  # Distribution concentrée
...     invalid_strategy="mixed",
...     invalid_kwargs={
...         "extreme_prob": 0.15,  # 15% de cas extrêmes
...         "scale_range": (0.05, 5.0),  # Large plage
...         "noise_level": 0.4  # Bruit élevé
...     },
...     seed=123,
...     shuffle=True
... )


3. Préparation pour Machine Learning
-------------------------------------
>>> from sklearn.model_selection import train_test_split

>>> # Séparer features et target
>>> X = df[['c0_real', 'c0_imag', 'c1_real', 'c1_imag',
...          'c2_real', 'c2_imag', 'c3_real', 'c3_imag', 'norm_squared']]
>>> y = df['is_valid']

>>> # Split train/test
>>> X_train, X_test, y_train, y_test = train_test_split(
...     X, y, test_size=0.2, random_state=42, stratify=y
... )


4. Tests et vérifications
--------------------------
>>> from data_generation import verify_normalization, generate_valid_states

>>> # Générer et vérifier
>>> states = generate_valid_states(100, 4, seed=42)
>>> is_valid, norms = verify_normalization(states)
>>> assert is_valid, "Tous les états doivent être normalisés!"

>>> # Vérifier le dataset
>>> df_valid = df[df['is_valid'] == 1]
>>> norms_valid = df_valid['norm_squared'].values
>>> assert np.allclose(norms_valid, 1.0), "États valides mal normalisés!"
"""
