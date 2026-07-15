"""
Module: data_generation.py
Objectif: Générer des états quantiques valides et invalides
Auteur: Mkl Zenin
Date: 2024-11-12

Ce module contient les fonctions pour créer des datasets d'états quantiques avec différentes stratégies de génération.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate_valid_states(
    n_samples: int,
    dim: int,
    strategy: str = "random",
    alpha: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Génère des états quantiques VALIDES (‖ψ‖² = 1 à la précision machine).

    Stratégies
    ----------
    - "random"    : cᵢ = xᵢ + i·yᵢ avec x, y ~ N(0, 1), puis division par ‖ψ‖.
      Lecture : « c i égale x i plus i fois y i, l'état est ensuite divisé
      par sa norme ». Propriété clé : la mesure gaussienne isotrope induit,
      après normalisation, la mesure UNIFORME sur la sphère unité complexe
      (mesure de Haar sur les états purs) — l'échantillonnage le plus
      « neutre » possible de l'espace des états.
    - "dirichlet" : probabilités pᵢ ~ Dirichlet(α, …, α) puis cᵢ = √pᵢ·e^{iφᵢ},
      φᵢ ~ U[0, 2π[. Contrôle de la concentration : α = 1 uniforme sur le
      simplex, α > 1 états équilibrés, α < 1 états piqués.
    - "basis"     : états purs de la base canonique |k⟩ (cas triviaux utiles
      pour borner les features : H = 0, pureté = 1).

    Paramètres
    ----------
    n_samples : nombre d'états (> 0).
    dim       : dimension de l'espace de Hilbert (> 0).
    strategy  : "random" | "dirichlet" | "basis".
    alpha     : paramètre de concentration Dirichlet (stratégie dirichlet).
    seed      : graine de reproductibilité.

    Retourne
    --------
    states : ndarray complexe (n_samples, dim), chaque ligne de norme 1.
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
        # Stratégie 3 : États purs de la base canonique — vectorisé (Q7).
        # Un indice de base aléatoire par état, puis indexation avancée :
        # states[n, indices[n]] = 1 pour chaque ligne n en une seule opération.
        states = np.zeros((n_samples, dim), dtype=complex)
        basis_indices = rng.integers(0, dim, size=n_samples)
        states[np.arange(n_samples), basis_indices] = 1.0 + 0j

    # === Vérification finale (optionnelle, pour debug) ===
    # Décommente pour vérifier que tous les états sont bien normalisés
    # norms_check = np.sum(np.abs(states)**2, axis=1)
    # assert np.allclose(norms_check, 1.0), "Certains états ne sont pas normalisés!"

    return states


def verify_normalization(
    states: np.ndarray, tolerance: float = 1e-6
) -> Tuple[bool, np.ndarray]:
    """
    Vérifie la normalisation : |‖ψ‖² − 1| ≤ tolerance pour chaque état.

    Lecture : « la valeur absolue de la norme au carré moins un est
    inférieure ou égale à la tolérance ».

    Sémantique de tolérance (correction Q6 de l'audit)
    --------------------------------------------------
    Le critère est STRICTEMENT absolu. L'ancienne implémentation utilisait
    ``np.allclose(..., atol=tolerance)`` dont le ``rtol=1e-5`` par défaut
    restait actif et s'ADDITIONNAIT à atol (critère effectif :
    |x − 1| ≤ atol + rtol·|1|). Pour un validateur, la définition du seuil
    doit être exacte et sans terme caché.

    Paramètres
    ----------
    states    : ndarray complexe (n_samples, dim).
    tolerance : borne absolue sur |‖ψ‖² − 1| (défaut 1e-6, largement
                au-dessus des erreurs d'arrondi float64 ~ 1e-15 pour d ≤ 100).

    Retourne
    --------
    (all_valid, norms_squared) : booléen global + normes² individuelles.
    """
    norms_squared = np.sum(np.abs(states) ** 2, axis=1)
    all_valid = bool(np.all(np.abs(norms_squared - 1.0) <= tolerance))
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

    logger.info("=" * 70)
    logger.info("STRATÉGIES DE GÉNÉRATION D'ÉTATS VALIDES")
    logger.info("=" * 70)

    for strategy, description in info.items():
        logger.info(f"\n {strategy.upper()}")
        logger.info(f"   {description}")

    logger.info("\n" + "=" * 70)


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
    norm_margin: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Génère des états invalides (non normalisés) selon la stratégie choisie.

    Frontière de classe (correction F2 de l'audit du 2026-07-07)
    ------------------------------------------------------------
    Tout état retourné vérifie la GARANTIE : |‖ψ‖² − 1| ≥ norm_margin.

    Lecture : « la valeur absolue de la norme au carré moins un est
    supérieure ou égale à la marge ».

    Avant cette correction, chaque stratégie avait sa propre notion de
    « proche de 1 » (scaling excluait k ∈ [0.95, 1.05], noise/direct ne
    garantissaient que |‖ψ‖² − 1| > 10⁻⁴) : la frontière entre classes
    n'était pas définie de façon unique. Désormais la définition est
    centralisée ici : les états valides ont ‖ψ‖² = 1 (à la précision
    machine), les invalides ont ‖ψ‖² hors de [1 − marge, 1 + marge].
    La bande interdite matérialise l'ambiguïté physique : un état à
    ‖ψ‖² = 1.001 est indistinguable d'une erreur d'arrondi.

    Paramètres
    ----------
    n_samples   : nombre d'états à générer (> 0).
    dim         : dimension de l'espace de Hilbert (> 0).
    strategy    : "scaling" | "noise" | "direct" | "mixed".
    scale_range : (k_min, k_max) pour la stratégie scaling.
    noise_level : écart-type du bruit pour la stratégie noise.
    extreme_prob: proportion de cas extrêmes pour la stratégie mixed.
    norm_margin : demi-largeur de la bande interdite autour de ‖ψ‖² = 1
                  (0 < norm_margin < 1). Les états générés dans la bande
                  sont repoussés à l'extérieur.
    seed        : graine du générateur pour reproductibilité.

    Retourne
    --------
    states : ndarray complexe (n_samples, dim), tous hors de la bande.
    """

    # Validation
    if n_samples <= 0:
        raise ValueError(f"n_samples doit être > 0, reçu: {n_samples}")

    if dim <= 0:
        raise ValueError(f"dim doit être > 0, reçu: {dim}")

    if not (0.0 < norm_margin < 1.0):
        raise ValueError(f"norm_margin doit être dans ]0, 1[, reçu: {norm_margin}")

    valid_strategies = ["scaling", "noise", "direct", "extreme", "mixed"]
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

    # === STRATÉGIE EXTREME ===
    elif strategy == "extreme":
        # Cas pathologiques : quasi-nuls, énormes, déséquilibrés.
        # Exposée comme stratégie de première classe pour le diagnostic
        # multiclasse de cause d'invalidité (jalon 4).
        states = _generate_extreme_states(n_samples, dim, rng)

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
                norm_margin=norm_margin,
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
                norm_margin=norm_margin,
                seed=int(rng.integers(0, int(1e9))),
            )
            states[idx : idx + n_noise] = states_noise
            idx += n_noise

        # 4. Direct
        if n_direct > 0:
            states_direct = generate_invalid_states(
                n_direct,
                dim,
                strategy="direct",
                norm_margin=norm_margin,
                seed=int(rng.integers(0, int(1e9))),
            )
            states[idx : idx + n_direct] = states_direct

        # Mélange aléatoirement
        rng.shuffle(states)

    # === GARANTIE DE LA FRONTIÈRE DE CLASSE (F2) ===
    # Repousse hors de la bande interdite [1 − marge, 1 + marge] tout état
    # dont la norme² serait tombée dedans (possible pour noise/direct).
    # Vectorisé : pour chaque état fautif on choisit une cible
    # ‖ψ‖²_cible = 1 ± marge·u avec u ~ U(1, 2) (côté aléatoire), puis on
    # multiplie l'état par √(cible / ‖ψ‖²) — le scaling d'un vecteur par f
    # multiplie sa norme² par f².
    norms_squared = np.sum(np.abs(states) ** 2, axis=1)
    ambiguous = np.abs(norms_squared - 1.0) < norm_margin

    if ambiguous.any():
        idx = np.where(ambiguous)[0]
        u = rng.uniform(1.0, 2.0, size=idx.size)
        side = rng.choice([-1.0, 1.0], size=idx.size)
        target = np.clip(1.0 + side * norm_margin * u, 0.01, None)
        factors = np.sqrt(target / norms_squared[idx])
        states[idx] *= factors[:, np.newaxis]

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

    logger.info("=" * 70)
    logger.info("STRATÉGIES DE GÉNÉRATION D'ÉTATS INVALIDES")
    logger.info("=" * 70)

    for strategy, description in info.items():
        logger.info(f"\n {strategy.upper()}")
        logger.info(f"   {description}")

    logger.info("\n" + "=" * 70)


# ============================================================================
# CRÉATION DU DATASET COMPLET
# ============================================================================


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
    logger.info(
        f"Génération de {n_valid} états valides (stratégie: {valid_strategy})..."
    )

    valid_kwargs = valid_kwargs or {}
    states_valid = generate_valid_states(
        n_samples=n_valid,
        dim=dim,
        strategy=valid_strategy,
        seed=seed_valid,
        **valid_kwargs,
    )

    # === GÉNÉRATION DES ÉTATS INVALIDES ===
    logger.info(
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
    logger.info("Construction du DataFrame...")

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
        logger.info("Mélange du dataset...")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Réattribue les state_id après shuffle
        df["state_id"] = np.arange(len(df))

    # === STATISTIQUES ===
    logger.info("\n" + "=" * 70)
    logger.info("DATASET CRÉÉ")
    logger.info("=" * 70)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Dimension: {dim}")
    logger.info(f"États valides: {n_valid} ({n_valid/n_total*100:.1f}%)")
    logger.info(f"États invalides: {n_invalid} ({n_invalid/n_total*100:.1f}%)")
    logger.info(f"\n Distribution de is_valid:")
    logger.info(df["is_valid"].value_counts().sort_index())

    logger.info(f"\n Statistiques de norm_squared:")
    logger.info(df["norm_squared"].describe())

    logger.info("=" * 70)

    return df


def create_multiclass_dataset(
    n_valid: int,
    n_per_cause: int,
    dim: int,
    norm_margin: float = 0.05,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> pd.DataFrame:
    """
    Dataset multiclasse : chaque état invalide porte le label de la CAUSE
    de son invalidité (jalon 4 — diagnostic, pas seulement détection).

    Pourquoi c'est un vrai problème d'apprentissage
    -----------------------------------------------
    Contrairement à la validité (fonction déterministe de la norme), la
    cause n'est PAS calculable depuis les amplitudes : un état de norme
    1.3 peut provenir d'un scaling (k ≈ 1.14), d'un bruit additif ou d'une
    génération directe. Les distributions des causes se chevauchent — le
    classifieur doit exploiter des signatures statistiques fines (profil
    des composantes, corrélations résiduelles avec un état normalisé).
    C'est le régime où le ML gagne sa place face au test à seuil.

    Analogie ingénierie : un système FDIR (Fault Detection, Isolation and
    Recovery) de satellite ne se contente pas de détecter qu'un capteur
    dévie — il doit isoler la cause (biais de calibration ? bruit anormal ?
    défaillance franche ?) pour choisir la récupération adaptée.

    Paramètres
    ----------
    n_valid     : nombre d'états valides (cause "valid").
    n_per_cause : nombre d'états PAR cause invalide
                  ("scaling", "noise", "direct", "extreme").
    dim         : dimension de l'espace de Hilbert.
    norm_margin : garantie de frontière F2 (voir generate_invalid_states).
    seed        : graine de reproductibilité.
    shuffle     : mélange final du DataFrame.

    Retourne
    --------
    DataFrame au schéma standard + colonnes ``cause`` (str) et ``is_valid``.
    """
    causes = ["scaling", "noise", "direct", "extreme"]
    rng = np.random.default_rng(seed)

    blocks = []
    labels = []

    states_valid = generate_valid_states(
        n_samples=n_valid,
        dim=dim,
        strategy="random",
        seed=int(rng.integers(0, 1_000_000_000)) if seed is not None else None,
    )
    blocks.append(states_valid)
    labels.extend(["valid"] * n_valid)

    for cause in causes:
        states_c = generate_invalid_states(
            n_samples=n_per_cause,
            dim=dim,
            strategy=cause,
            norm_margin=norm_margin,
            seed=int(rng.integers(0, 1_000_000_000)) if seed is not None else None,
        )
        blocks.append(states_c)
        labels.extend([cause] * n_per_cause)

    all_states = np.vstack(blocks).astype(complex)
    n_total = len(labels)
    norms_squared = np.sum(np.abs(all_states) ** 2, axis=1)

    data_dict = {"state_id": np.arange(n_total)}
    for i in range(dim):
        data_dict[f"c{i}_real"] = np.real(all_states[:, i])
        data_dict[f"c{i}_imag"] = np.imag(all_states[:, i])
    data_dict["norm_squared"] = norms_squared
    data_dict["cause"] = labels
    data_dict["is_valid"] = (np.array(labels) == "valid").astype(int)

    df = pd.DataFrame(data_dict)

    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df["state_id"] = np.arange(len(df))

    logger.info(
        "Dataset multiclasse créé : %d états (%d valides, %d par cause invalide)",
        n_total,
        n_valid,
        n_per_cause,
    )
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

    logger.info(f"\n Dataset sauvegardé:")
    logger.info(f"   Chemin: {filepath}")
    logger.info(f"   Taille: {file_size:.2f} KB")
    logger.info(f"   Lignes: {len(df)}")
    logger.info(f"   Colonnes: {len(df.columns)}")

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

    logger.info(f"Dataset chargé depuis {filepath}")
    logger.info(f"Shape: {df.shape}")

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

    logger.info("=" * 70)
    logger.info("INFORMATIONS SUR LE DATASET")
    logger.info("=" * 70)
    logger.info(f"\n Taille:")
    logger.info(f"   Échantillons: {info['n_samples']}")
    logger.info(f"   Features: {info['n_features']}")
    logger.info(f"   Dimension: {info['dim']}")

    logger.info(f"\n Distribution des classes:")
    logger.info(
        f"   Valides (1): {info['n_valid']} ({info['n_valid']/info['n_samples']*100:.1f}%)"
    )
    logger.info(
        f"   Invalides (0): {info['n_invalid']} ({info['n_invalid']/info['n_samples']*100:.1f}%)"
    )
    logger.info(f"   Ratio: {info['balance_ratio']:.3f}")

    if abs(info["balance_ratio"] - 1.0) < 0.05:
        logger.info(f"    Dataset bien équilibré")
    else:
        logger.info(f"     Dataset déséquilibré (idéal: ratio ≈ 1.0)")

    logger.info(f"\n Statistiques de norm_squared:")
    for key, value in info["norm_stats"].items():
        logger.info(f"   {key:8s}: {value:.6f}")

    logger.info("=" * 70)


if __name__ == "__main__":
    # Ce bloc s'exécute uniquement si on lance : python src/data_generation.py
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Test du module data_generation.py\n")

    # Affiche les stratégies disponibles
    print_strategy_info()

    # Test des 3 stratégies
    logger.info("\n" + "=" * 70)
    logger.info("TESTS DE GÉNÉRATION")
    logger.info("=" * 70)

    dim = 3
    n_samples = 5

    for strategy in ["random", "dirichlet", "basis"]:
        logger.info(f"\n--- Stratégie : {strategy} ---")

        states = generate_valid_states(
            n_samples=n_samples, dim=dim, strategy=strategy, seed=42
        )

        logger.info(f"Forme du tableau : {states.shape}")
        logger.info(f"Type de données : {states.dtype}")

        # Vérification
        all_valid, norms = verify_normalization(states)
        logger.info(f"Tous les états sont normalisés ? {all_valid}")
        logger.info(f"Normes au carré : {norms}")

        # Affiche les 2 premiers états
        logger.info(f"\n2 premiers états :")
        for i in range(min(2, n_samples)):
            logger.info(f"  État {i} : {states[i]}")
            logger.info(f"    ||ψ||² = {norms[i]:.10f}")
