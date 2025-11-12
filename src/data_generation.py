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
