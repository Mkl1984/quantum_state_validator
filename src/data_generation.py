"""
Module: data_generation.py
Objectif: Générer des états quantiques valides et invalides
Auteur: [Ton nom]
Date: 2024-11-12

Ce module contient les fonctions pour créer des datasets d'états quantiques
avec différentes stratégies de génération.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


def generate_valid_states(
    n_samples: int,
    dim: int,
    strategy: str = "random",
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Génère des états quantiques valides (normalisés).

    Paramètres
    ----------
    n_samples : int
        Nombre d'états à générer.

    dim : int
        Dimension de l'espace de Hilbert (nombre de composantes).
        Exemples: dim=2 (qubit), dim=3 (qutrit), dim=4, etc.

    strategy : str, optional
        Stratégie de génération:
        - "random" : génération gaussienne + normalisation (défaut)
        - "dirichlet" : distribution de Dirichlet pour les probabilités
        - "basis" : états purs de la base canonique

    alpha : float, optional
        Paramètre de concentration pour stratégie "dirichlet".
        - alpha = 1.0 : uniforme (défaut)
        - alpha > 1.0 : favorise probabilités équilibrées
        - alpha < 1.0 : favorise probabilités déséquilibrées

    seed : int, optional
        Graine aléatoire pour reproductibilité.
        Si None, utilise l'état aléatoire actuel de NumPy.

    Retourne
    --------
    states : np.ndarray
        Tableau de shape (n_samples, dim) contenant les états générés.
        Chaque ligne est un état quantique normalisé (dtype=complex128).

    Raises
    ------
    ValueError
        Si strategy n'est pas reconnue.
        Si n_samples <= 0 ou dim <= 0.

    Exemples
    --------
    >>> # Générer 100 qubits avec stratégie random
    >>> states = generate_valid_states(100, dim=2, strategy="random", seed=42)
    >>> states.shape
    (100, 2)
    >>> np.allclose(np.sum(np.abs(states)**2, axis=1), 1.0)
    True

    Notes
    -----
    Toutes les stratégies garantissent que ||ψ||² = 1 pour chaque état.
    """

    # === Validation des paramètres ===
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être > 0, reçu: {n_samples}")

    if dim <= 0:
        raise ValueError(f"dim doit être > 0, reçu: {dim}")

    if strategy not in ["random", "dirichlet", "basis"]:
        raise ValueError(
            f"strategy '{strategy}' inconnue. "
            f"Choix possibles: 'random', 'dirichlet', 'basis'"
        )

    # === Initialisation du générateur aléatoire ===
    rng = np.random.default_rng(seed)  # Générateur moderne NumPy

    # === Génération selon la stratégie choisie ===

    if strategy == "random":
        # Stratégie 1 : Génération gaussienne + normalisation

        # Génère parties réelles et imaginaires indépendantes
        # Distribution normale centrée réduite N(0, 1)
        real_parts = rng.normal(loc=0.0, scale=1.0, size=(n_samples, dim))
        imag_parts = rng.normal(loc=0.0, scale=1.0, size=(n_samples, dim))

        # Construit les coefficients complexes
        states = real_parts + 1j * imag_parts

        # Normalise chaque état
        # norms: shape (n_samples,) contenant ||ψ||² pour chaque état
        norms = np.sqrt(np.sum(np.abs(states) ** 2, axis=1, keepdims=True))

        # Évite division par zéro (cas extrêmement rare)
        norms = np.where(norms == 0, 1.0, norms)

        states = states / norms

    elif strategy == "dirichlet":
        # Stratégie 2 : Distribution de Dirichlet pour les probabilités

        # Génère les probabilités via Dirichlet
        # alpha_vec: vecteur de paramètres de concentration
        alpha_vec = np.full(dim, alpha)

        # probabilities: shape (n_samples, dim)
        # Chaque ligne somme à 1.0
        probabilities = rng.dirichlet(alpha_vec, size=n_samples)

        # Génère des phases aléatoires uniformes dans [0, 2π]
        phases = rng.uniform(0, 2 * np.pi, size=(n_samples, dim))

        # Construit les coefficients complexes
        # c_i = √p_i · e^(iφ_i) = √p_i · (cos(φ_i) + i·sin(φ_i))
        amplitudes = np.sqrt(probabilities)
        states = amplitudes * np.exp(1j * phases)

        # Vérification (devrait déjà être normalisé par construction)
        # Mais on normalise quand même pour éviter erreurs numériques
        norms = np.sqrt(np.sum(np.abs(states) ** 2, axis=1, keepdims=True))
        states = states / norms

    elif strategy == "basis":
        # Stratégie 3 : États purs de la base canonique

        # Si n_samples > dim, on génère plusieurs copies de chaque état
        states = np.zeros((n_samples, dim), dtype=complex)

        for i in range(n_samples):
            # Sélectionne un indice de base aléatoirement
            basis_index = rng.integers(0, dim)

            # Crée l'état pur |basis_index⟩
            states[i, basis_index] = 1.0 + 0j

    # === Vérification finale (optionnelle, pour debug) ===
    # Décommente pour vérifier que tous les états sont bien normalisés
    # norms_check = np.sum(np.abs(states)**2, axis=1)
    # assert np.allclose(norms_check, 1.0), "Certains états ne sont pas normalisés!"

    return states


def verify_normalization(
    states: np.ndarray, tolerance: float = 1e-6
) -> Tuple[bool, np.ndarray]:
    """
    Vérifie que tous les états d'un batch sont normalisés.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de shape (n_samples, dim) contenant les états.

    tolerance : float, optional
        Tolérance numérique pour la vérification.

    Retourne
    --------
    all_valid : bool
        True si TOUS les états sont normalisés, False sinon.

    norms_squared : np.ndarray
        Tableau de shape (n_samples,) contenant ||ψ||² pour chaque état.

    Exemples
    --------
    >>> states = generate_valid_states(10, dim=3, seed=42)
    >>> all_valid, norms = verify_normalization(states)
    >>> all_valid
    True
    >>> np.allclose(norms, 1.0)
    True
    """

    # Calcule ||ψ||² pour chaque état
    norms_squared = np.sum(np.abs(states) ** 2, axis=1)

    # Vérifie si tous sont proches de 1.0
    all_valid = np.allclose(norms_squared, 1.0, atol=tolerance)

    return all_valid, norms_squared


# === Fonctions utilitaires ===


def get_strategy_info() -> dict:
    """
    Retourne un dictionnaire décrivant les stratégies disponibles.

    Retourne
    --------
    info : dict
        Dictionnaire {strategy_name: description}.
    """

    info = {
        "random": (
            "Génération gaussienne + normalisation. "
            "Explore uniformément l'espace des états quantiques. "
            "Recommandé pour usage général."
        ),
        "dirichlet": (
            "Distribution de Dirichlet pour les probabilités. "
            "Paramètre alpha contrôle la dispersion: "
            "alpha=1 (uniforme), alpha>1 (équilibré), alpha<1 (pics). "
            "Utile pour tester différentes distributions de probabilités."
        ),
        "basis": (
            "États purs de la base canonique. "
            "Génère des états du type |0⟩, |1⟩, ..., |n-1⟩. "
            "Utile pour avoir des cas triviaux dans le dataset."
        ),
    }

    return info


def print_strategy_info():
    """
    Affiche les informations sur les stratégies de génération.
    """
    info = get_strategy_info()

    print("=" * 70)
    print("STRATÉGIES DE GÉNÉRATION D'ÉTATS VALIDES")
    print("=" * 70)

    for strategy, description in info.items():
        print(f"\n📌 {strategy.upper()}")
        print(f"   {description}")

    print("\n" + "=" * 70)


# === Exemple d'utilisation (si le script est exécuté directement) ===

# ============================================================================
# GÉNÉRATION D'ÉTATS INVALIDES (NON NORMALISÉS)
# ============================================================================


def generate_invalid_states(
    n_samples: int,
    dim: int,
    strategy: str = "scaling",
    scale_range: Tuple[float, float] = (0.1, 2.0),
    noise_level: float = 0.3,
    extreme_prob: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Génère des états quantiques invalides (non normalisés).

    Paramètres
    ----------
    n_samples : int
        Nombre d'états invalides à générer.

    dim : int
        Dimension de l'espace de Hilbert.

    strategy : str, optional
        Stratégie de génération:
        - "scaling" : multiplie des états valides par un facteur k ≠ 1 (défaut)
        - "noise" : ajoute du bruit à des états valides sans renormaliser
        - "direct" : génère directement sans normalisation
        - "mixed" : mélange des 3 stratégies + cas extrêmes

    scale_range : tuple of float, optional
        Pour stratégie "scaling": intervalle [k_min, k_max] pour le facteur k.
        Par défaut: (0.1, 2.0) en évitant [0.95, 1.05] pour éviter ambiguïté.

    noise_level : float, optional
        Pour stratégie "noise": intensité du bruit (epsilon).
        Par défaut: 0.3

    extreme_prob : float, optional
        Pour stratégie "mixed": probabilité de générer un cas extrême.
        Par défaut: 0.1 (10% de cas extrêmes)

    seed : int, optional
        Graine aléatoire pour reproductibilité.

    Retourne
    --------
    states : np.ndarray
        Tableau de shape (n_samples, dim) contenant les états invalides.
        dtype=complex128.
        Garantie: AUCUN état n'est normalisé (||ψ||² ≠ 1).

    Raises
    ------
    ValueError
        Si strategy n'est pas reconnue.

    Exemples
    --------
    >>> states = generate_invalid_states(100, dim=3, strategy="scaling", seed=42)
    >>> all_valid, norms = verify_normalization(states)
    >>> all_valid
    False
    >>> (norms != 1.0).all()
    True

    Notes
    -----
    Pour stratégie "scaling", on évite k ∈ [0.95, 1.05] pour créer une
    séparation claire entre états valides et invalides.
    """

    # Validation
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être > 0, reçu: {n_samples}")

    if dim <= 0:
        raise ValueError(f"dim doit être > 0, reçu: {dim}")

    valid_strategies = ["scaling", "noise", "direct", "mixed"]
    if strategy not in valid_strategies:
        raise ValueError(
            f"strategy '{strategy}' inconnue. " f"Choix possibles: {valid_strategies}"
        )

    rng = np.random.default_rng(seed)

    # === STRATÉGIE SCALING ===
    if strategy == "scaling":
        # Génère d'abord des états valides
        states_valid = generate_valid_states(
            n_samples=n_samples,
            dim=dim,
            strategy="random",
            seed=rng.integers(0, 1e9),  # Seed aléatoire différent
        )

        # Génère des facteurs de scaling k
        # On évite l'intervalle [0.95, 1.05] pour éviter ambiguïté
        k_min, k_max = scale_range

        # Génère k uniformément dans [k_min, k_max]
        factors = rng.uniform(k_min, k_max, size=n_samples)

        # Exclut l'intervalle [0.95, 1.05]
        # Si k tombe dans cet intervalle, on le repousse
        mask_ambiguous = (factors >= 0.95) & (factors <= 1.05)
        n_ambiguous = mask_ambiguous.sum()

        if n_ambiguous > 0:
            # Remplace les valeurs ambiguës par des valeurs claires
            # 50% en dessous de 0.95, 50% au-dessus de 1.05
            new_factors = np.where(
                rng.random(n_ambiguous) < 0.5,
                rng.uniform(k_min, 0.95, size=n_ambiguous),
                rng.uniform(1.05, k_max, size=n_ambiguous),
            )
            factors[mask_ambiguous] = new_factors

        # Applique le scaling
        # Broadcasting: (n_samples,) × (n_samples, dim)
        states = states_valid * factors[:, np.newaxis]

    # === STRATÉGIE NOISE ===
    elif strategy == "noise":
        # Génère des états valides
        states_valid = generate_valid_states(
            n_samples=n_samples, dim=dim, strategy="random", seed=rng.integers(0, 1e9)
        )

        # Génère du bruit complexe
        noise_real = rng.normal(0, noise_level, size=(n_samples, dim))
        noise_imag = rng.normal(0, noise_level, size=(n_samples, dim))
        noise = noise_real + 1j * noise_imag

        # Ajoute le bruit (sans renormaliser !)
        states = states_valid + noise

    # === STRATÉGIE DIRECT ===
    elif strategy == "direct":
        # Génère directement sans normaliser
        real_parts = rng.normal(0, 1, size=(n_samples, dim))
        imag_parts = rng.normal(0, 1, size=(n_samples, dim))
        states = real_parts + 1j * imag_parts

        # Applique un scaling aléatoire pour varier ||ψ||²
        scale_factors = rng.uniform(0.1, 3.0, size=n_samples)
        states = states * scale_factors[:, np.newaxis]

    # === STRATÉGIE MIXED ===
    elif strategy == "mixed":
        states = np.zeros((n_samples, dim), dtype=complex)

        # Répartition des sous-stratégies
        n_extreme = int(n_samples * extreme_prob)
        n_remaining = n_samples - n_extreme

        # Distribution du reste entre scaling, noise, direct
        n_scaling = n_remaining // 3
        n_noise = n_remaining // 3
        n_direct = n_remaining - n_scaling - n_noise

        idx = 0

        # 1. Cas extrêmes
        if n_extreme > 0:
            states_extreme = _generate_extreme_states(n_extreme, dim, rng)
            states[idx : idx + n_extreme] = states_extreme
            idx += n_extreme

        # 2. Scaling
        if n_scaling > 0:
            states_scaling = generate_invalid_states(
                n_scaling,
                dim,
                strategy="scaling",
                scale_range=scale_range,
                seed=rng.integers(0, 1e9),
            )
            states[idx : idx + n_scaling] = states_scaling
            idx += n_scaling

        # 3. Noise
        if n_noise > 0:
            states_noise = generate_invalid_states(
                n_noise,
                dim,
                strategy="noise",
                noise_level=noise_level,
                seed=rng.integers(0, 1e9),
            )
            states[idx : idx + n_noise] = states_noise
            idx += n_noise

        # 4. Direct
        if n_direct > 0:
            states_direct = generate_invalid_states(
                n_direct, dim, strategy="direct", seed=rng.integers(0, 1e9)
            )
            states[idx : idx + n_direct] = states_direct

        # Mélange aléatoirement
        rng.shuffle(states)

    # === VÉRIFICATION FINALE ===
    # S'assure qu'aucun état n'est accidentellement normalisé
    norms_squared = np.sum(np.abs(states) ** 2, axis=1)
    accidentally_normalized = np.isclose(norms_squared, 1.0, atol=1e-4)

    if accidentally_normalized.any():
        # Rescale légèrement ces états
        indices = np.where(accidentally_normalized)[0]
        for idx in indices:
            # Multiplie par un facteur aléatoire loin de 1
            factor = rng.choice([0.7, 0.8, 1.2, 1.3])
            states[idx] *= factor

    return states


def _generate_extreme_states(n_samples: int, dim: int, rng) -> np.ndarray:
    """
    Génère des cas extrêmes (outliers) pour tester la robustesse.

    Cas générés:
    - États nuls ou quasi-nuls (||ψ||² ≈ 0)
    - États très grands (||ψ||² >> 1)
    - États avec une composante dominante énorme

    Fonction interne, pas destinée à être utilisée directement.
    """
    states = np.zeros((n_samples, dim), dtype=complex)

    for i in range(n_samples):
        case_type = rng.choice(["null", "huge", "unbalanced"])

        if case_type == "null":
            # État quasi-nul
            states[i] = rng.normal(0, 0.01, dim) + 1j * rng.normal(0, 0.01, dim)

        elif case_type == "huge":
            # État très grand
            states[i] = rng.normal(10, 5, dim) + 1j * rng.normal(10, 5, dim)

        elif case_type == "unbalanced":
            # Une composante énorme, les autres petites
            dominant_idx = rng.integers(0, dim)
            states[i] = rng.normal(0, 0.1, dim) + 1j * rng.normal(0, 0.1, dim)
            states[i, dominant_idx] = rng.uniform(50, 100) + 1j * rng.uniform(50, 100)

    return states


def get_invalid_strategy_info() -> dict:
    """
    Retourne les descriptions des stratégies de génération d'états invalides.
    """
    info = {
        "scaling": (
            "Multiplie des états valides par un facteur k ≠ 1. "
            "Contrôle: k ∈ [k_min, k_max] en évitant [0.95, 1.05]. "
            "Produit: ||ψ||² = k². "
            "Recommandé pour usage général."
        ),
        "noise": (
            "Ajoute du bruit à des états valides sans renormaliser. "
            "Paramètre noise_level contrôle l'intensité. "
            "Produit: états 'presque valides' (utile pour robustesse)."
        ),
        "direct": (
            "Génère directement des coefficients sans normalisation. "
            "Produit: large distribution de ||ψ||². "
            "Bonne diversité."
        ),
        "mixed": (
            "Combine scaling, noise, direct + cas extrêmes. "
            "Paramètre extreme_prob contrôle % de outliers. "
            "Produit: dataset très diversifié. "
            "Recommandé pour dataset final."
        ),
    }
    return info


def print_invalid_strategy_info():
    """
    Affiche les informations sur les stratégies d'états invalides.
    """
    info = get_invalid_strategy_info()

    print("=" * 70)
    print("STRATÉGIES DE GÉNÉRATION D'ÉTATS INVALIDES")
    print("=" * 70)

    for strategy, description in info.items():
        print(f"\n📌 {strategy.upper()}")
        print(f"   {description}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Ce bloc s'exécute uniquement si on lance: python src/data_generation.py

    print("Test du module data_generation.py\n")

    # Affiche les stratégies disponibles
    print_strategy_info()

    # Test des 3 stratégies
    print("\n" + "=" * 70)
    print("TESTS DE GÉNÉRATION")
    print("=" * 70)

    dim = 3
    n_samples = 5

    for strategy in ["random", "dirichlet", "basis"]:
        print(f"\n--- Stratégie: {strategy} ---")

        states = generate_valid_states(
            n_samples=n_samples, dim=dim, strategy=strategy, seed=42
        )

        print(f"Shape: {states.shape}")
        print(f"Dtype: {states.dtype}")

        # Vérification
        all_valid, norms = verify_normalization(states)
        print(f"Tous normalisés? {all_valid}")
        print(f"Normes²: {norms}")

        # Affiche les 2 premiers états
        print(f"\n2 premiers états:")
        for i in range(min(2, n_samples)):
            print(f"  État {i}: {states[i]}")
            print(f"    ||ψ||² = {norms[i]:.10f}")
