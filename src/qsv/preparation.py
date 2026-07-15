"""
Module: preparation.py
Purpose: State-preparation quality control against KNOWN target states.

In the anonymous-state validation task (notebooks 08-10), only the norm
carries label information, and notebook 09 proved a Bayes limit on cause
diagnosis by an isotropy argument: without a reference direction, a scaled
state and a noisy state are indistinguishable beyond their norms.

Here the setting changes: the validator knows WHICH state the hardware was
asked to prepare (a register initialisation, an interferometer input state,
a QKD symbol). Direction now carries information, the isotropy argument
collapses, and fidelity to the target becomes a legitimate statistic.

Error model per prepared sample:
- "ok":      the exact target state
- "rotated": cos(theta)|target> + sin(theta)|chi> with |chi> orthogonal to
             the target -- a unitary pointing error. Still normalised, so
             any norm-based test is blind to it. Fidelity = cos^2(theta).
- "scaled":  k * |target> with |k^2 - 1| >= norm_margin -- a gain error.
             Direction is preserved, so any fidelity-based test is blind
             to it.

All samples are then read through shot noise (sigma = 1 / (2 sqrt(N))),
consistent with the measurement model of notebooks 08-10.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from qsv.features import norm_squared, sigma_from_shots

logger = logging.getLogger(__name__)

__all__ = ["make_targets", "fidelity_to_target", "create_preparation_dataset"]


def make_targets(n_targets: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    """Draw Haar-random reference states the validator will know exactly."""
    if n_targets <= 0 or dim <= 0:
        raise ValueError("n_targets and dim must be > 0")
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(n_targets, dim)) + 1j * rng.normal(size=(n_targets, dim))
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def fidelity_to_target(states: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Estimated fidelity F = |<target|state>|^2 / ||state||^2, row by row.

    The ||state||^2 division makes F scale-invariant: it measures direction
    only, which is exactly why it is blind to gain errors ("scaled" class)
    and why it must be paired with a norm statistic.
    """
    overlap = np.abs(np.sum(np.conj(targets) * states, axis=1)) ** 2
    ns = norm_squared(states)
    ns = np.where(ns < 1e-12, 1.0, ns)
    return overlap / ns


def create_preparation_dataset(
    n_per_class: int,
    dim: int = 4,
    n_targets: int = 4,
    n_shots: int = 500,
    rotation_fidelity: Tuple[float, float] = (0.5, 0.9),
    norm_margin: float = 0.05,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Build a preparation-QA dataset: each row is a state that was SUPPOSED
    to be targets[target_id], read through shot noise.

    Returns (df, targets). Columns: c{i}_real/imag (noisy amplitudes),
    norm_squared (recomputed on noisy amplitudes), target_id, error_type
    in {"ok", "rotated", "scaled"}, prep_ok (1 for "ok").
    """
    if n_per_class <= 0:
        raise ValueError("n_per_class must be > 0")
    f_lo, f_hi = rotation_fidelity
    if not (0.0 < f_lo < f_hi < 1.0):
        raise ValueError("rotation_fidelity must satisfy 0 < lo < hi < 1")

    rng = np.random.default_rng(seed)
    targets = make_targets(n_targets, dim, seed=int(rng.integers(1e9)))

    n_total = 3 * n_per_class
    target_ids = rng.integers(0, n_targets, size=n_total)
    phi = targets[target_ids]

    states = np.empty((n_total, dim), dtype=complex)
    labels = np.array(
        ["ok"] * n_per_class + ["rotated"] * n_per_class + ["scaled"] * n_per_class
    )

    # "ok": exact targets
    sl = slice(0, n_per_class)
    states[sl] = phi[sl]

    # "rotated": mix with a random direction orthogonal to the target
    sl = slice(n_per_class, 2 * n_per_class)
    chi = rng.normal(size=(n_per_class, dim)) + 1j * rng.normal(size=(n_per_class, dim))
    # remove the component along the target, then normalise
    proj = np.sum(np.conj(phi[sl]) * chi, axis=1, keepdims=True)
    chi = chi - proj * phi[sl]
    chi = chi / np.linalg.norm(chi, axis=1, keepdims=True)
    f_true = rng.uniform(f_lo, f_hi, size=(n_per_class, 1))
    states[sl] = np.sqrt(f_true) * phi[sl] + np.sqrt(1.0 - f_true) * chi

    # "scaled": gain error outside the validity band
    sl = slice(2 * n_per_class, n_total)
    side = rng.choice([-1.0, 1.0], size=n_per_class)
    k2 = 1.0 + side * norm_margin * rng.uniform(1.0, 6.0, size=n_per_class)
    k2 = np.clip(k2, 0.05, None)
    states[sl] = np.sqrt(k2)[:, np.newaxis] * phi[sl]

    # shot-noise readout, same model as notebooks 08-10
    sigma = sigma_from_shots(n_shots)
    states = (
        states
        + rng.normal(0, sigma, states.shape)
        + 1j * rng.normal(0, sigma, states.shape)
    )

    data = {"state_id": np.arange(n_total)}
    for i in range(dim):
        data[f"c{i}_real"] = states[:, i].real
        data[f"c{i}_imag"] = states[:, i].imag
    data["norm_squared"] = norm_squared(states)
    data["target_id"] = target_ids
    data["error_type"] = labels
    data["prep_ok"] = (labels == "ok").astype(int)

    df = pd.DataFrame(data)
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df["state_id"] = np.arange(len(df))

    logger.info(
        "Preparation dataset: %d samples, %d targets, N=%d shots",
        n_total,
        n_targets,
        n_shots,
    )
    return df, targets
