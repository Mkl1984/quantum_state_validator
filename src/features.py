"""
Module: features.py
Objectif: Features quantiques avec séparation explicite entre features
          sensibles à l'échelle (qui « voient » la norme) et features
          invariantes d'échelle (qui ne la voient pas).

Pourquoi ce module existe (audit §4 — target leakage du notebook 07)
====================================================================

La validité d'un état |ψ⟩ = (c₁, …, c_d) est définie par ‖ψ‖² = Σᵢ |cᵢ|² = 1.
C'est une fonction **déterministe** des features brutes. Conséquence :

- Toute feature *sensible à l'échelle* (norm_squared, norm_deviation, mais
  aussi entropy/purity calculées sur les |cᵢ|² bruts) contient la réponse :
  le « modèle » ne fait que réapprendre la définition du label → 100 %
  d'accuracy, zéro science.
- Toute feature *invariante d'échelle* (calculée sur les probabilités
  renormalisées p̃ᵢ) ne contient par construction AUCUNE information sur la
  validité : c → k·c laisse p̃ᵢ inchangé, donc la feature ne peut pas
  distinguer un état de son multiple invalide.

Il n'y a pas de milieu : sur données exactes, le problème est soit trivial,
soit impossible. Le problème ne devient un vrai problème d'apprentissage
statistique QUE si les amplitudes sont observées avec du bruit — voir
``add_measurement_noise`` et le notebook 08.

Rappel physique fondamental
---------------------------
La norme d'un état quantique n'est PAS une observable : les statistiques de
mesure de |ψ⟩ et de k|ψ⟩ sont identiques (règle de Born renormalisée,
P(i) = |cᵢ|²/‖ψ‖²). Valider ‖ψ‖² = 1 est donc un contrôle de *données*
(sorties de tomographie, de simulateurs, de pipelines numériques), pas une
mesure physique. C'est exactement le rôle d'un module de qualification de
données dans un système embarqué : horloges atomiques GNSS, gyromètres à
interférométrie atomique ou liens QKD par satellite doivent qualifier leurs
états reconstruits à partir de statistiques finies, jamais des amplitudes
exactes.

Conventions
-----------
Le DataFrame d'entrée suit le schéma du dataset principal :
colonnes ``c{i}_real`` / ``c{i}_imag`` pour i = 0 … d−1.
Aucune fonction de ce module n'imprime : elles retournent, point.
"""

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "extract_amplitudes",
    "raw_probabilities",
    "norm_squared",
    "scale_sensitive_features",
    "scale_invariant_features",
    "compute_features",
    "add_measurement_noise",
    "sigma_from_shots",
]

# Garde numérique pour log(0) dans l'entropie.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_amplitudes(df: pd.DataFrame) -> np.ndarray:
    """
    Reconstruit la matrice complexe des amplitudes depuis les colonnes plates.

    Lecture de la formule : states[n, i] = c{i}_real[n] + i·c{i}_imag[n],
    soit « l'amplitude i de l'état n est la partie réelle plus i fois la
    partie imaginaire ».

    Paramètres
    ----------
    df : DataFrame contenant les colonnes ``c{i}_real`` / ``c{i}_imag``.

    Retourne
    --------
    states : ndarray complexe de forme (n_samples, dim).

    Lève
    ----
    ValueError si les colonnes d'amplitudes sont absentes ou dépareillées.
    """
    real_cols = sorted(
        (c for c in df.columns if c.startswith("c") and c.endswith("_real")),
        key=lambda c: int(c[1:-5]),
    )
    imag_cols = sorted(
        (c for c in df.columns if c.startswith("c") and c.endswith("_imag")),
        key=lambda c: int(c[1:-5]),
    )
    if not real_cols or len(real_cols) != len(imag_cols):
        raise ValueError(
            "Colonnes d'amplitudes introuvables ou dépareillées "
            f"(real: {len(real_cols)}, imag: {len(imag_cols)})."
        )
    return df[real_cols].to_numpy() + 1j * df[imag_cols].to_numpy()


def raw_probabilities(states: np.ndarray) -> np.ndarray:
    """
    Poids bruts pᵢ = |cᵢ|² (NON renormalisés — ce ne sont des probabilités
    au sens strict que si l'état est valide).

    Lecture : « p i égale module de c i au carré ».
    """
    return np.abs(states) ** 2


def norm_squared(states: np.ndarray) -> np.ndarray:
    """
    Norme au carré ‖ψ‖² = Σᵢ |cᵢ|² de chaque état.

    Lecture : « somme sur i des modules carrés des amplitudes ».
    C'est LA quantité qui définit la validité (= 1 pour un état physique).
    """
    return raw_probabilities(states).sum(axis=1)


# ---------------------------------------------------------------------------
# Features SENSIBLES à l'échelle — elles contiennent la réponse
# ---------------------------------------------------------------------------


def scale_sensitive_features(states: np.ndarray) -> pd.DataFrame:
    """
    Features qui changent sous c → k·c : elles encodent la norme, donc le
    label. À n'utiliser QUE (a) sur données bruitées (notebook 08), où elles
    deviennent des *estimateurs* légitimes, ou (b) pour illustrer le leakage.

    Colonnes retournées
    -------------------
    norm_squared   : ‖ψ‖² = Σ|cᵢ|².                      Se transforme en k²·‖ψ‖².
    norm_deviation : |‖ψ‖² − 1|. C'est littéralement la définition du label —
                     l'utiliser comme feature EST le leakage du notebook 07.
    entropy_raw    : −Σ pᵢ ln pᵢ sur les pᵢ BRUTS. Pour un état invalide,
                     les pᵢ ne somment pas à 1 : cette « entropie » n'en est
                     pas une (elle peut être négative, énorme…) et elle fuit
                     la norme.
    purity_raw     : Σ pᵢ² sur les pᵢ bruts. Se transforme en k⁴·purity :
                     fuite de norme encore plus violente (10⁸ observé sur les
                     états extrêmes du dataset).
    """
    p = raw_probabilities(states)
    ns = p.sum(axis=1)
    return pd.DataFrame(
        {
            "norm_squared": ns,
            "norm_deviation": np.abs(ns - 1.0),
            "entropy_raw": -(p * np.log(p + _EPS)).sum(axis=1),
            "purity_raw": (p**2).sum(axis=1),
        }
    )


# ---------------------------------------------------------------------------
# Features INVARIANTES d'échelle — aveugles à la norme par construction
# ---------------------------------------------------------------------------


def scale_invariant_features(states: np.ndarray) -> pd.DataFrame:
    """
    Features calculées sur les probabilités renormalisées p̃ᵢ = pᵢ / Σⱼ pⱼ.

    Invariance : sous c → k·c, pᵢ → k²·pᵢ et Σⱼ pⱼ → k²·Σⱼ pⱼ, donc p̃ᵢ est
    inchangé. Ces features ne peuvent mathématiquement PAS distinguer un état
    valide de son multiple invalide : un classifieur qui n'utilise qu'elles
    plafonne à ~50 % sur ce dataset. C'est un résultat, pas un échec — il
    prouve que toute performance au-dessus du hasard vient de la norme.

    Colonnes retournées
    -------------------
    entropy_shannon     : H(p̃) = −Σ p̃ᵢ ln p̃ᵢ ∈ [0, ln d].
                          Lecture : « moins la somme des p̃ᵢ log p̃ᵢ ».
                          Mesure l'étalement de l'état sur la base : 0 pour un
                          état de base pur, ln d pour la superposition uniforme.
    purity_normalized   : Σ p̃ᵢ² ∈ [1/d, 1]. Concentration de la distribution
                          (1 = état de base, 1/d = uniforme).
    participation_ratio : 1 / Σ p̃ᵢ² ∈ [1, d]. « Nombre effectif de composantes
                          peuplées » — l'inverse de la pureté, plus lisible
                          physiquement (≈ combien d'états de base participent).
    max_prob            : max p̃ᵢ. Poids de la composante dominante.
    """
    p = raw_probabilities(states)
    total = p.sum(axis=1, keepdims=True)
    # États pathologiques de norme quasi nulle : renormalisation impossible,
    # on retombe sur une distribution uniforme (choix documenté, testé).
    dim = p.shape[1]
    safe_total = np.where(total < _EPS, 1.0, total)
    p_tilde = np.where(total < _EPS, 1.0 / dim, p / safe_total)

    purity = (p_tilde**2).sum(axis=1)
    return pd.DataFrame(
        {
            "entropy_shannon": -(p_tilde * np.log(p_tilde + _EPS)).sum(axis=1),
            "purity_normalized": purity,
            "participation_ratio": 1.0 / purity,
            "max_prob": p_tilde.max(axis=1),
        }
    )


def compute_features(df: pd.DataFrame, kind: str = "invariant") -> pd.DataFrame:
    """
    Point d'entrée : calcule un bloc de features depuis le DataFrame du dataset.

    Paramètres
    ----------
    kind : "invariant" (sans fuite de norme), "sensitive" (avec fuite —
           assumée et documentée), ou "all" (les deux, pour comparaison).

    Retourne
    --------
    DataFrame de features, même index que ``df``.
    """
    if kind not in ("invariant", "sensitive", "all"):
        raise ValueError(f"kind '{kind}' inconnu (invariant | sensitive | all).")
    states = extract_amplitudes(df)
    blocks = []
    if kind in ("invariant", "all"):
        blocks.append(scale_invariant_features(states))
    if kind in ("sensitive", "all"):
        blocks.append(scale_sensitive_features(states))
    out = pd.concat(blocks, axis=1)
    out.index = df.index
    return out


# ---------------------------------------------------------------------------
# Bruit de mesure — ce qui rend le problème statistique (notebook 08)
# ---------------------------------------------------------------------------


def sigma_from_shots(n_shots: int) -> float:
    """
    Écart-type du bruit d'estimation d'amplitude pour un budget de N mesures.

    Modèle : σ = 1 / (2·√N).

    Lecture : « sigma égale un sur deux racine de N ».

    Justification (modèle simplifié, assumé comme tel) : estimer une
    amplitude à partir de N répétitions donne une erreur statistique en
    1/√N — la loi universelle du bruit de grenaille (shot noise), la même
    qui gouverne la précision d'une horloge atomique GNSS ou d'un
    accéléromètre à atomes froids en fonction du temps d'intégration.
    Le facteur 1/2 vient de p = |c|² : δp ≈ 2|c|·δc, et δp ~ √(p(1−p)/N)
    donne δc ~ 1/(2√N) au voisinage des amplitudes typiques. L'ordre de
    grandeur est ce qui compte : N contrôle la difficulté du problème.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots doit être > 0, reçu: {n_shots}")
    return 1.0 / (2.0 * np.sqrt(n_shots))


def add_measurement_noise(
    df: pd.DataFrame,
    n_shots: int = 1000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simule une reconstruction tomographique à budget de mesures fini :
    chaque composante réelle/imaginaire est observée avec un bruit gaussien
    additif ĉᵢ = cᵢ + ε, ε ~ N(0, σ²), σ = 1/(2√N).

    Lecture : « c i estimé égale c i plus epsilon, epsilon suivant une
    normale centrée d'écart-type sigma ».

    Pourquoi c'est LE bon cadre pour ce projet : sur amplitudes exactes,
    valider ‖ψ‖² = 1 est un calcul, pas un apprentissage. Sur amplitudes
    estimées, ‖ψ̂‖² fluctue autour de ‖ψ‖² avec une dispersion en 1/√N :
    près de la frontière, les classes se chevauchent réellement, et décider
    devient un problème statistique légitime (compromis faux positifs /
    faux négatifs, seuil dépendant de N — voir notebook 08).

    Paramètres
    ----------
    df      : dataset au schéma standard (c{i}_real / c{i}_imag).
    n_shots : budget de mesures N (grand N = peu de bruit = problème facile).
    seed    : graine du générateur pour reproductibilité.

    Retourne
    --------
    Copie de ``df`` dont les colonnes d'amplitudes sont bruitées et dont la
    colonne ``norm_squared`` (si présente) est recalculée sur les amplitudes
    bruitées. La colonne ``is_valid`` reste le label VRAI (celui de l'état
    sous-jacent) : c'est précisément ce qu'on demande au classifieur de
    retrouver à travers le bruit.
    """
    sigma = sigma_from_shots(n_shots)
    rng = np.random.default_rng(seed)
    out = df.copy()

    amp_cols = [
        c
        for c in df.columns
        if c.startswith("c") and (c.endswith("_real") or c.endswith("_imag"))
    ]
    if not amp_cols:
        raise ValueError("Aucune colonne d'amplitude c{i}_real / c{i}_imag trouvée.")

    noise = rng.normal(0.0, sigma, size=(len(df), len(amp_cols)))
    out[amp_cols] = df[amp_cols].to_numpy() + noise

    if "norm_squared" in out.columns:
        out["norm_squared"] = norm_squared(extract_amplitudes(out))
    return out
