"""
Module: data_generation.py
Objectif: Générer des états quantiques valides et invalides
Auteur: Mkl Zenin
Date: 2024-11-12

Ce module contient les fonctions pour créer des datasets d'états quantiques avec différentes stratégies de génération.
"""

import numpy as np
from typing import Tuple, Optional
import warnings
from pathlib import Path


def generate_valid_states(
    n_samples: int,
    dim: int,
    strategy: str = "random",
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:

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
        print(f"\n {strategy.upper()}")
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
            seed=int(
                rng.integers(0, int(1e9))
            ),  # Seed aléatoire différent (cast to Python int)
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
            n_samples=n_samples,
            dim=dim,
            strategy="random",
            seed=int(rng.integers(0, int(1e9))),
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
                seed=int(rng.integers(0, int(1e9))),
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
                seed=int(rng.integers(0, int(1e9))),
            )
            states[idx : idx + n_noise] = states_noise
            idx += n_noise

        # 4. Direct
        if n_direct > 0:
            states_direct = generate_invalid_states(
                n_direct, dim, strategy="direct", seed=int(rng.integers(0, int(1e9)))
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

    info = get_invalid_strategy_info()

    print("=" * 70)
    print("STRATÉGIES DE GÉNÉRATION D'ÉTATS INVALIDES")
    print("=" * 70)

    for strategy, description in info.items():
        print(f"\n {strategy.upper()}")
        print(f"   {description}")

    print("\n" + "=" * 70)


# ============================================================================
# CRÉATION DU DATASET COMPLET
# ============================================================================

import pandas as pd


def create_dataset(
    n_valid: int,
    n_invalid: int,
    dim: int,
    valid_strategy: str = "random",
    invalid_strategy: str = "mixed",
    valid_kwargs: Optional[dict] = None,
    invalid_kwargs: Optional[dict] = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> pd.DataFrame:

    # Initialisation du générateur aléatoire
    rng = np.random.default_rng(seed)

    # Graines pour les sous-générateurs (pour reproductibilité)
    # Utilise des bornes entières et convertit le résultat en int Python
    seed_valid = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
    seed_invalid = int(rng.integers(0, 1_000_000_000)) if seed is not None else None

    # === GÉNÉRATION DES ÉTATS VALIDES ===
    print(f"Génération de {n_valid} états valides (stratégie: {valid_strategy})...")

    valid_kwargs = valid_kwargs or {}
    states_valid = generate_valid_states(
        n_samples=n_valid,
        dim=dim,
        strategy=valid_strategy,
        seed=seed_valid,
        **valid_kwargs,
    )

    # === GÉNÉRATION DES ÉTATS INVALIDES ===
    print(
        f"Génération de {n_invalid} états invalides (stratégie: {invalid_strategy})..."
    )

    invalid_kwargs = invalid_kwargs or {}
    states_invalid = generate_invalid_states(
        n_samples=n_invalid,
        dim=dim,
        strategy=invalid_strategy,
        seed=seed_invalid,
        **invalid_kwargs,
    )

    # === CONSTRUCTION DU DATAFRAME ===
    print("Construction du DataFrame...")

    # Combine les deux ensembles
    all_states = np.vstack([states_valid, states_invalid])
    n_total = n_valid + n_invalid

    # Crée les labels
    labels = np.concatenate(
        [
            np.ones(n_valid, dtype=int),  # 1 pour valides
            np.zeros(n_invalid, dtype=int),  # 0 pour invalides
        ]
    )

    # Calcule les normes²
    norms_squared = np.sum(np.abs(all_states) ** 2, axis=1)

    # Construit le dictionnaire de données
    data_dict = {}

    # Colonne state_id
    data_dict["state_id"] = np.arange(n_total)

    # Assure que all_states est un ndarray de dtype complex pour des opérations sûres
    all_states = np.asarray(all_states, dtype=complex)

    # Colonnes c{i}_real et c{i}_imag (utilise les fonctions numpy pour éviter les problèmes de typage)
    for i in range(dim):
        data_dict[f"c{i}_real"] = np.real(all_states[:, i])
        data_dict[f"c{i}_imag"] = np.imag(all_states[:, i])

    # Colonne norm_squared
    data_dict["norm_squared"] = norms_squared

    # Colonne is_valid (target)
    data_dict["is_valid"] = labels

    # Crée le DataFrame
    df = pd.DataFrame(data_dict)

    # === MÉLANGE (SHUFFLE) ===
    if shuffle:
        print("Mélange du dataset...")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Réattribue les state_id après shuffle
        df["state_id"] = np.arange(len(df))

    # === STATISTIQUES ===
    print("\n" + "=" * 70)
    print("DATASET CRÉÉ")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Dimension: {dim}")
    print(f"États valides: {n_valid} ({n_valid/n_total*100:.1f}%)")
    print(f"États invalides: {n_invalid} ({n_invalid/n_total*100:.1f}%)")
    print(f"\n Distribution de is_valid:")
    print(df["is_valid"].value_counts().sort_index())

    print(f"\n Statistiques de norm_squared:")
    print(df["norm_squared"].describe())

    print("=" * 70)

    return df


def save_dataset(
    df: pd.DataFrame,
    filename: str = "quantum_states_dataset.csv",
    data_dir: str = "data/processed",
) -> Path:

    # Crée le dossier si nécessaire
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Chemin complet
    filepath = data_path / filename

    # Sauvegarde
    df.to_csv(filepath, index=False)

    # Taille du fichier
    file_size = filepath.stat().st_size / 1024  # en KB

    print(f"\n Dataset sauvegardé:")
    print(f"   Chemin: {filepath}")
    print(f"   Taille: {file_size:.2f} KB")
    print(f"   Lignes: {len(df)}")
    print(f"   Colonnes: {len(df.columns)}")

    return filepath


def load_dataset(
    filename: str = "quantum_states_dataset.csv", data_dir: str = "data/processed"
) -> pd.DataFrame:

    filepath = Path(data_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Le fichier {filepath} n'existe pas. "
            f"Utilisez create_dataset() puis save_dataset() d'abord."
        )

    df = pd.read_csv(filepath)

    print(f"Dataset chargé depuis {filepath}")
    print(f"Shape: {df.shape}")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:

    # Déduit la dimension
    # Colonnes: state_id, c0_real, c0_imag, ..., c{d-1}_real, c{d-1}_imag, norm_squared, is_valid
    # Nombre de colonnes c{i}_* = 2*dim
    n_c_columns = len([col for col in df.columns if col.startswith("c")])
    dim = n_c_columns // 2

    info = {
        "n_samples": len(df),
        "n_features": n_c_columns + 1,  # +1 pour norm_squared
        "dim": dim,
        "n_valid": (df["is_valid"] == 1).sum(),
        "n_invalid": (df["is_valid"] == 0).sum(),
        "balance_ratio": (df["is_valid"] == 1).sum() / (df["is_valid"] == 0).sum(),
        "norm_stats": df["norm_squared"].describe().to_dict(),
    }

    return info


def print_dataset_info(df: pd.DataFrame):

    info = get_dataset_info(df)

    print("=" * 70)
    print("INFORMATIONS SUR LE DATASET")
    print("=" * 70)
    print(f"\n Taille:")
    print(f"   Échantillons: {info['n_samples']}")
    print(f"   Features: {info['n_features']}")
    print(f"   Dimension: {info['dim']}")

    print(f"\n Distribution des classes:")
    print(
        f"   Valides (1): {info['n_valid']} ({info['n_valid']/info['n_samples']*100:.1f}%)"
    )
    print(
        f"   Invalides (0): {info['n_invalid']} ({info['n_invalid']/info['n_samples']*100:.1f}%)"
    )
    print(f"   Ratio: {info['balance_ratio']:.3f}")

    if abs(info["balance_ratio"] - 1.0) < 0.05:
        print(f"    Dataset bien équilibré")
    else:
        print(f"     Dataset déséquilibré (idéal: ratio ≈ 1.0)")

    print(f"\n Statistiques de norm_squared:")
    for key, value in info["norm_stats"].items():
        print(f"   {key:8s}: {value:.6f}")

    print("=" * 70)


if __name__ == "__main__":
    # Ce bloc s'exécute uniquement si on lance : python src/data_generation.py

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
        print(f"\n--- Stratégie : {strategy} ---")

        states = generate_valid_states(
            n_samples=n_samples, dim=dim, strategy=strategy, seed=42
        )

        print(f"Forme du tableau : {states.shape}")
        print(f"Type de données : {states.dtype}")

        # Vérification
        all_valid, norms = verify_normalization(states)
        print(f"Tous les états sont normalisés ? {all_valid}")
        print(f"Normes au carré : {norms}")

        # Affiche les 2 premiers états
        print(f"\n2 premiers états :")
        for i in range(min(2, n_samples)):
            print(f"  État {i} : {states[i]}")
            print(f"    ||ψ||² = {norms[i]:.10f}")
