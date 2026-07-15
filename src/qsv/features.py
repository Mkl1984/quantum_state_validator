"""
Module: features.py
Purpose: quantum-state features with an explicit split between
         scale-SENSITIVE features (which "see" the norm) and
         scale-INVARIANT features (which cannot).

Why this module exists (audit section 4 - the notebook 07 target leakage)
==========================================================================

The validity of a state |psi> = (c_1, ..., c_d) is defined by
||psi||^2 = sum_i |c_i|^2 = 1. That is a DETERMINISTIC function of the raw
features. Consequences:

- Any scale-sensitive feature (norm_squared, norm_deviation, but also
  entropy/purity computed on the raw |c_i|^2) contains the answer: a
  "model" trained on it merely re-learns the label definition - 100%
  accuracy, zero science.
- Any scale-invariant feature (computed on the renormalized probabilities
  p_tilde_i) carries, by construction, NO information about validity:
  c -> k*c leaves p_tilde_i unchanged, so the feature cannot distinguish a
  state from its invalid multiple.

There is no middle ground: on exact data the problem is either trivial or
impossible. It becomes a genuine statistical learning problem ONLY when the
amplitudes are observed through noise - see ``add_measurement_noise`` and
notebook 08.

Fundamental physics reminder
----------------------------
The norm of a quantum state is NOT an observable: the measurement
statistics of |psi> and k|psi> are identical (renormalized Born rule,
P(i) = |c_i|^2 / ||psi||^2). Checking ||psi||^2 = 1 is therefore a *data*
quality check (tomography outputs, simulators, numerical pipelines), not a
physical measurement. This is exactly the role of a data-qualification
module in an embedded system: GNSS atomic clocks, atom-interferometry
gyroscopes and satellite QKD links must qualify their reconstructed states
from finite statistics, never from exact amplitudes.

Conventions
-----------
Input DataFrames follow the main dataset schema: columns ``c{i}_real`` /
``c{i}_imag`` for i = 0 ... d-1. No function in this module prints:
they return, period.
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
    "add_correlated_noise",
    "add_calibration_drift",
    "sigma_from_shots",
]

# Numerical guard for log(0) in the entropy.
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_amplitudes(df: pd.DataFrame) -> np.ndarray:
    """
    Rebuild the complex amplitude matrix from the flat columns.

    Reading the formula: states[n, i] = c{i}_real[n] + i * c{i}_imag[n],
    i.e. "amplitude i of state n is the real part plus i times the
    imaginary part".

    Parameters
    ----------
    df : DataFrame with ``c{i}_real`` / ``c{i}_imag`` columns.

    Returns
    -------
    states : complex ndarray of shape (n_samples, dim).

    Raises
    ------
    ValueError if the amplitude columns are missing or mismatched.
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
            "Amplitude columns missing or mismatched "
            f"(real: {len(real_cols)}, imag: {len(imag_cols)})."
        )
    return df[real_cols].to_numpy() + 1j * df[imag_cols].to_numpy()


def raw_probabilities(states: np.ndarray) -> np.ndarray:
    """
    Raw weights p_i = |c_i|^2 (NOT renormalized - they are probabilities in
    the strict sense only when the state is valid).

    Reading: "p i equals the squared modulus of c i".
    """
    return np.abs(states) ** 2


def norm_squared(states: np.ndarray) -> np.ndarray:
    """
    Squared norm ||psi||^2 = sum_i |c_i|^2 of each state.

    Reading: "sum over i of the squared amplitude moduli".
    This is THE quantity that defines validity (= 1 for a physical state).
    """
    return raw_probabilities(states).sum(axis=1)


# ---------------------------------------------------------------------------
# Scale-SENSITIVE features - they contain the answer
# ---------------------------------------------------------------------------


def scale_sensitive_features(states: np.ndarray) -> pd.DataFrame:
    """
    Features that change under c -> k*c: they encode the norm, hence the
    label. To be used ONLY (a) on noisy data (notebook 08), where they
    become legitimate *estimators*, or (b) to illustrate leakage.

    Returned columns
    ----------------
    norm_squared   : ||psi||^2 = sum |c_i|^2.       Transforms as k^2 * ||psi||^2.
    norm_deviation : |  ||psi||^2 - 1 |. This is literally the label
                     definition - using it as a feature IS the notebook 07
                     leakage.
    entropy_raw    : -sum p_i ln p_i on the RAW p_i. For an invalid state
                     the p_i do not sum to 1: this "entropy" is not one (it
                     can be negative, huge...) and it leaks the norm.
    purity_raw     : sum p_i^2 on the raw p_i. Transforms as k^4 * purity:
                     an even more violent norm leak (1e8 observed on the
                     dataset's extreme states).
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
# Scale-INVARIANT features - norm-blind by construction
# ---------------------------------------------------------------------------


def scale_invariant_features(states: np.ndarray) -> pd.DataFrame:
    """
    Features computed on the renormalized probabilities
    p_tilde_i = p_i / sum_j p_j.

    Invariance: under c -> k*c, p_i -> k^2 * p_i and sum_j p_j -> k^2 *
    sum_j p_j, so p_tilde_i is unchanged. These features mathematically
    CANNOT distinguish a valid state from its invalid multiple: a
    classifier using only them caps at ~50% on this dataset. That is a
    result, not a failure - it proves that any above-chance performance
    comes from the norm.

    Returned columns
    ----------------
    entropy_shannon     : H(p_tilde) = -sum p_tilde_i ln p_tilde_i in [0, ln d].
                          Reading: "minus the sum of p_tilde_i log p_tilde_i".
                          Measures the spread of the state over the basis: 0
                          for a pure basis state, ln d for the uniform
                          superposition.
    purity_normalized   : sum p_tilde_i^2 in [1/d, 1]. Concentration of the
                          distribution (1 = basis state, 1/d = uniform).
    participation_ratio : 1 / sum p_tilde_i^2 in [1, d]. "Effective number
                          of populated components" - the inverse purity,
                          physically more readable (roughly how many basis
                          states participate).
    max_prob            : max p_tilde_i. Weight of the dominant component.
    """
    p = raw_probabilities(states)
    total = p.sum(axis=1, keepdims=True)
    # Pathological near-zero-norm states: renormalization is impossible, we
    # fall back to a uniform distribution (documented choice, tested).
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
    Entry point: compute a feature block from the dataset DataFrame.

    Parameters
    ----------
    kind : "invariant" (no norm leak), "sensitive" (leaking - owned and
           documented), or "all" (both, for comparisons).

    Returns
    -------
    Feature DataFrame, same index as ``df``.
    """
    if kind not in ("invariant", "sensitive", "all"):
        raise ValueError(f"Unknown kind '{kind}' (invariant | sensitive | all).")
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
# Measurement noise - what turns the problem into statistics (notebook 08)
# ---------------------------------------------------------------------------


def sigma_from_shots(n_shots: int) -> float:
    """
    Standard deviation of the amplitude-estimation noise for a budget of N
    measurements.

    Model: sigma = 1 / (2 * sqrt(N)).

    Reading: "sigma equals one over two square root of N".

    Justification (simplified model, owned as such): estimating an
    amplitude from N repetitions carries a statistical error scaling as
    1/sqrt(N) - the universal shot-noise law, the same one that governs the
    precision of a GNSS atomic clock or a cold-atom accelerometer as a
    function of integration time. The 1/2 factor comes from p = |c|^2:
    delta_p ~ 2|c| * delta_c, and delta_p ~ sqrt(p(1-p)/N) gives
    delta_c ~ 1/(2 sqrt(N)) near typical amplitudes. The order of magnitude
    is what matters: N controls the difficulty of the problem.
    """
    if n_shots <= 0:
        raise ValueError(f"n_shots must be > 0, got: {n_shots}")
    return 1.0 / (2.0 * np.sqrt(n_shots))


def add_measurement_noise(
    df: pd.DataFrame,
    n_shots: int = 1000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate a finite-shot tomographic reconstruction: each real/imaginary
    component is observed with additive Gaussian noise
    c_hat_i = c_i + eps, eps ~ N(0, sigma^2), sigma = 1/(2 sqrt(N)).

    Reading: "estimated c i equals c i plus epsilon, epsilon following a
    centered normal of standard deviation sigma".

    Why this is THE right frame for this project: on exact amplitudes,
    checking ||psi||^2 = 1 is a computation, not learning. On estimated
    amplitudes, ||psi_hat||^2 fluctuates around ||psi||^2 with a 1/sqrt(N)
    spread: near the boundary the classes genuinely overlap, and deciding
    becomes a legitimate statistical problem (false-positive/false-negative
    trade-off, N-dependent threshold - see notebook 08).

    Parameters
    ----------
    df      : dataset in the standard schema (c{i}_real / c{i}_imag).
    n_shots : measurement budget N (large N = low noise = easy problem).
    seed    : RNG seed for reproducibility.

    Returns
    -------
    Copy of ``df`` with noisy amplitude columns and, if present, the
    ``norm_squared`` column recomputed on the noisy amplitudes. The
    ``is_valid`` column stays the TRUE label (of the underlying state):
    that is precisely what the classifier is asked to recover through the
    noise.
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
        raise ValueError("No amplitude columns c{i}_real / c{i}_imag found.")

    noise = rng.normal(0.0, sigma, size=(len(df), len(amp_cols)))
    out[amp_cols] = df[amp_cols].to_numpy() + noise

    if "norm_squared" in out.columns:
        out["norm_squared"] = norm_squared(extract_amplitudes(out))
    return out


def add_correlated_noise(
    df: pd.DataFrame,
    n_shots: int = 1000,
    rho: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    EQUICORRELATED measurement noise across components: one common mode per
    state (milestone 4 - the regime where the "estimated norm" statistic
    stops being complete, and where learning may earn its place).

    Model
    -----
    For each state, each part (real and imaginary separately):

        eps_i = sigma * (sqrt(rho) * z + sqrt(1 - rho) * w_i),
        z ~ N(0, 1) shared,  w_i ~ N(0, 1) iid.

    Reading: "epsilon i equals sigma times square root of rho times z, plus
    square root of one minus rho times w i" - z is the common mode shared
    by all components of the state, w_i the component-specific noise.

    Properties (locked by tests):
    - Var(eps_i) = sigma^2 whatever rho: at fixed budget N, the
      per-component noise is identical to the iid case - only the STRUCTURE
      changes;
    - Corr(eps_i, eps_j) = rho for i != j.

    Physics of the common mode
    --------------------------
    In a real instrument, part of the noise is shared by all measurement
    channels: carrier vibration on an atom interferometer, intensity
    fluctuation of the readout laser, thermal drift of the acquisition
    chain. The iid noise of notebook 08 is the idealization; the common
    mode is the embedded reality. Statistical consequence: the noise on
    ||psi_hat||^2 becomes heteroscedastic - its variance depends on the
    state through the (sum_i c_i)^2 signal/common-mode coupling, which a
    global threshold cannot exploit but a model holding the coordinates
    can, in principle, learn. (Measured verdict in notebook 09: it barely
    matters - the 2d-component sum filters the common mode.)

    Parameters
    ----------
    df      : dataset in the standard schema.
    n_shots : measurement budget N (sigma = 1/(2 sqrt(N))).
    rho     : cross-component correlation, 0 <= rho < 1 (0 = iid,
              equivalent to add_measurement_noise).
    seed    : RNG seed for reproducibility.

    Returns
    -------
    Noisy copy of ``df`` (same conventions as add_measurement_noise:
    label untouched, norm_squared recomputed).
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1), got: {rho}")
    sigma = sigma_from_shots(n_shots)
    rng = np.random.default_rng(seed)
    out = df.copy()

    real_cols = sorted(
        (c for c in df.columns if c.startswith("c") and c.endswith("_real")),
        key=lambda c: int(c[1:-5]),
    )
    imag_cols = sorted(
        (c for c in df.columns if c.startswith("c") and c.endswith("_imag")),
        key=lambda c: int(c[1:-5]),
    )
    if not real_cols:
        raise ValueError("No amplitude columns c{i}_real / c{i}_imag found.")

    n, d = len(df), len(real_cols)
    for cols in (real_cols, imag_cols):
        z = rng.normal(0.0, 1.0, size=(n, 1))  # per-state common mode
        w = rng.normal(0.0, 1.0, size=(n, d))  # component-specific noise
        eps = sigma * (np.sqrt(rho) * z + np.sqrt(1.0 - rho) * w)
        out[cols] = df[cols].to_numpy() + eps

    if "norm_squared" in out.columns:
        out["norm_squared"] = norm_squared(extract_amplitudes(out))
    return out


def add_calibration_drift(
    df: pd.DataFrame,
    n_shots: int = 1000,
    drift_amplitude: float = 0.08,
    drift_period: float = 2000.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Slowly varying calibration drift on top of shot noise (milestone 4b -
    the NON-STATIONARY regime, designed to defeat any fixed threshold).

    Model
    -----
    Each state carries an acquisition time t (its index in the DataFrame).
    The measurement chain has a slowly drifting gain:

        g(t) = A * sin(2*pi*t / T)
        c_hat_i(t) = (1 + g(t)) * c_i + eps_i,   eps_i ~ N(0, sigma^2)

    Reading: "g of t equals A sine of two pi t over T"; "estimated c i
    equals one plus g of t, times c i, plus the shot noise".

    Consequence on the norm: ||psi_hat||^2 ~ (1 + g(t))^2 * ||psi||^2. A
    VALID state read at the drift peak shows (1 +/- A)^2 ~ 1 +/- 2A: with
    A = 0.08 the apparent norm sweeps [0.85, 1.17] - far outside the
    +/-0.05 validity band. A fixed threshold, however well calibrated at
    time t0, becomes systematically wrong elsewhere in the cycle.

    Physics
    -------
    This is THE central problem of embedded instrumentation: orbital
    thermal cycling (a satellite's day/night cycle), ageing acquisition
    chains, readout-laser power drift. The classic engineering answer is
    periodic recalibration; the ML answer is to learn the calibration map
    g(t) from data. Notebook 10 confronts the two (winner: the hybrid).

    Parameters
    ----------
    df              : dataset in the standard schema (row order defines the
                      acquisition time).
    n_shots         : measurement budget N (shot noise sigma = 1/(2 sqrt(N))).
    drift_amplitude : A, relative gain-drift amplitude (e.g. 0.08 = +/-8%).
                      Must satisfy |A| < 1.
    drift_period    : T, drift period in number of acquired states.
    seed            : RNG seed (shot noise) for reproducibility.

    Returns
    -------
    Copy of ``df`` with an ``acquisition_time`` column (t), drifted+noisy
    amplitudes, and ``norm_squared`` recomputed. Label untouched: validity
    is that of the UNDERLYING state, not of its drifted reading.
    """
    if not (abs(drift_amplitude) < 1.0):
        raise ValueError(
            f"drift_amplitude must satisfy |A| < 1, got: {drift_amplitude}"
        )
    if drift_period <= 0:
        raise ValueError(f"drift_period must be > 0, got: {drift_period}")

    sigma = sigma_from_shots(n_shots)
    rng = np.random.default_rng(seed)
    out = df.copy()

    amp_cols = [
        c
        for c in df.columns
        if c.startswith("c") and (c.endswith("_real") or c.endswith("_imag"))
    ]
    if not amp_cols:
        raise ValueError("No amplitude columns c{i}_real / c{i}_imag found.")

    n = len(df)
    t = np.arange(n, dtype=float)
    gain = 1.0 + drift_amplitude * np.sin(2.0 * np.pi * t / drift_period)

    noise = rng.normal(0.0, sigma, size=(n, len(amp_cols)))
    out[amp_cols] = df[amp_cols].to_numpy() * gain[:, np.newaxis] + noise
    out["acquisition_time"] = t

    if "norm_squared" in out.columns:
        out["norm_squared"] = norm_squared(extract_amplitudes(out))
    return out
