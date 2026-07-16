"""
Module: data_generation.py
Purpose: generate valid and invalid quantum states
Author: Mkl Zenin
Date: 2024-11-12

This module contains the functions to build quantum-state datasets with
several generation strategies.
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
    Generate VALID quantum states (||psi||^2 = 1 to machine precision).

    Strategies
    ----------
    - "random"    : c_i = x_i + i*y_i with x, y ~ N(0, 1), then division by
      ||psi||. Reading: "c i equals x i plus i times y i, the state is then
      divided by its norm". Key property: the isotropic Gaussian measure
      induces, after normalization, the UNIFORM measure on the complex unit
      sphere (Haar measure on pure states) - the most "neutral" possible
      sampling of state space.
    - "dirichlet" : probabilities p_i ~ Dirichlet(alpha, ..., alpha) then
      c_i = sqrt(p_i) * exp(i*phi_i), phi_i ~ U[0, 2*pi). Concentration
      control: alpha = 1 uniform on the simplex, alpha > 1 balanced states,
      alpha < 1 peaked states.
    - "basis"     : pure canonical-basis states |k> (trivial cases, useful
      to bound the features: H = 0, purity = 1).

    Parameters
    ----------
    n_samples : number of states (> 0).
    dim       : Hilbert-space dimension (> 0).
    strategy  : "random" | "dirichlet" | "basis".
    alpha     : Dirichlet concentration parameter (dirichlet strategy).
    seed      : RNG seed for reproducibility.

    Returns
    -------
    states : complex ndarray (n_samples, dim), each row of norm 1.
    """
    # === Parameter validation ===
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got: {n_samples}")

    if dim <= 0:
        raise ValueError(f"dim must be > 0, got: {dim}")

    if strategy not in ["random", "dirichlet", "basis"]:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choices: 'random', 'dirichlet', 'basis'"
        )

    # === Random generator initialization ===
    rng = np.random.default_rng(seed)  # Modern NumPy generator

    # === Generation according to the chosen strategy ===

    if strategy == "random":
        # Strategy 1: Gaussian generation + normalization

        # Independent real and imaginary parts
        # Standard normal distribution N(0, 1)
        real_parts = rng.normal(loc=0.0, scale=1.0, size=(n_samples, dim))
        imag_parts = rng.normal(loc=0.0, scale=1.0, size=(n_samples, dim))

        # Build the complex coefficients
        states = real_parts + 1j * imag_parts

        # Normalize each state
        # norms: shape (n_samples,) holding ||psi|| for each state
        norms = np.sqrt(np.sum(np.abs(states) ** 2, axis=1, keepdims=True))

        # Avoid division by zero (extremely rare case)
        norms = np.where(norms == 0, 1.0, norms)

        states = states / norms

    elif strategy == "dirichlet":
        # Strategy 2: Dirichlet distribution for the probabilities

        # Draw the probabilities from a Dirichlet
        # alpha_vec: concentration parameter vector
        alpha_vec = np.full(dim, alpha)

        # probabilities: shape (n_samples, dim)
        # Each row sums to 1.0
        probabilities = rng.dirichlet(alpha_vec, size=n_samples)

        # Uniform random phases in [0, 2*pi]
        phases = rng.uniform(0, 2 * np.pi, size=(n_samples, dim))

        # Build the complex coefficients
        # c_i = sqrt(p_i) * exp(i*phi_i) = sqrt(p_i) * (cos(phi_i) + i*sin(phi_i))
        amplitudes = np.sqrt(probabilities)
        states = amplitudes * np.exp(1j * phases)

        # Check (should already be normalized by construction)
        # but normalize anyway to guard against numerical error
        norms = np.sqrt(np.sum(np.abs(states) ** 2, axis=1, keepdims=True))
        states = states / norms

    elif strategy == "basis":
        # Strategy 3: pure canonical-basis states - vectorized (Q7).
        # One random basis index per state, then advanced indexing:
        # states[n, indices[n]] = 1 for every row n in a single operation.
        states = np.zeros((n_samples, dim), dtype=complex)
        basis_indices = rng.integers(0, dim, size=n_samples)
        states[np.arange(n_samples), basis_indices] = 1.0 + 0j

    # === Final check (optional, for debugging) ===
    # Uncomment to verify that all states are properly normalized
    # norms_check = np.sum(np.abs(states)**2, axis=1)
    # assert np.allclose(norms_check, 1.0), "Some states are not normalized!"

    return states


def verify_normalization(
    states: np.ndarray, tolerance: float = 1e-6
) -> Tuple[bool, np.ndarray]:
    """
    Check normalization: | ||psi||^2 - 1 | <= tolerance for each state.

    Reading: "the absolute value of the squared norm minus one is less
    than or equal to the tolerance".

    Tolerance semantics (audit fix Q6)
    ----------------------------------
    The criterion is STRICTLY absolute. The previous implementation used
    ``np.allclose(..., atol=tolerance)`` whose default ``rtol=1e-5``
    remained active and ADDED to atol (effective criterion:
    |x - 1| <= atol + rtol*|1|). For a validator, the threshold definition
    must be exact, with no hidden term.

    Parameters
    ----------
    states    : complex ndarray (n_samples, dim).
    tolerance : absolute bound on | ||psi||^2 - 1 | (default 1e-6, far
                above float64 rounding ~ 1e-15 for d <= 100).

    Returns
    -------
    (all_valid, norms_squared) : global boolean + individual squared norms.
    """
    norms_squared = np.sum(np.abs(states) ** 2, axis=1)
    all_valid = bool(np.all(np.abs(norms_squared - 1.0) <= tolerance))
    return all_valid, norms_squared


# === Utility functions ===


def get_strategy_info() -> dict:
    """
    Return a dictionary describing the available strategies.

    Returns
    -------
    info : dict
        {strategy_name: description} dictionary.
    """

    info = {
        "random": (
            "Gaussian generation + normalization. "
            "Explores quantum state space uniformly. "
            "Recommended for general use."
        ),
        "dirichlet": (
            "Dirichlet distribution for the probabilities. "
            "The alpha parameter controls the spread: "
            "alpha=1 (uniform), alpha>1 (balanced), alpha<1 (peaked). "
            "Useful to test different probability distributions."
        ),
        "basis": (
            "Pure canonical-basis states. "
            "Generates states of the form |0>, |1>, ..., |n-1>. "
            "Useful to include trivial cases in the dataset."
        ),
    }

    return info


def print_strategy_info():
    """
    Log the information about the generation strategies.
    """
    info = get_strategy_info()

    logger.info("=" * 70)
    logger.info("VALID-STATE GENERATION STRATEGIES")
    logger.info("=" * 70)

    for strategy, description in info.items():
        logger.info(f"\n {strategy.upper()}")
        logger.info(f"   {description}")

    logger.info("\n" + "=" * 70)


# === Usage example (when the script is run directly) ===

# ============================================================================
# INVALID (NON-NORMALIZED) STATE GENERATION
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
    Generate invalid (non-normalized) states with the chosen strategy.

    Class boundary (audit fix F2, 2026-07-07)
    -----------------------------------------
    Every returned state satisfies the GUARANTEE:
    | ||psi||^2 - 1 | >= norm_margin.

    Reading: "the absolute value of the squared norm minus one is greater
    than or equal to the margin".

    Before this fix, each strategy had its own notion of "close to 1"
    (scaling excluded k in [0.95, 1.05], noise/direct only guaranteed
    | ||psi||^2 - 1 | > 1e-4): the class boundary was not uniquely
    defined. The definition is now centralized here: valid states have
    ||psi||^2 = 1 (to machine precision), invalid states have ||psi||^2
    outside [1 - margin, 1 + margin]. The forbidden band embodies the
    physical ambiguity: a state at ||psi||^2 = 1.001 is indistinguishable
    from a rounding error.

    Parameters
    ----------
    n_samples   : number of states to generate (> 0).
    dim         : Hilbert-space dimension (> 0).
    strategy    : "scaling" | "noise" | "direct" | "extreme" | "mixed".
    scale_range : (k_min, k_max) for the scaling strategy.
    noise_level : noise standard deviation for the noise strategy.
    extreme_prob: fraction of extreme cases for the mixed strategy.
    norm_margin : half-width of the forbidden band around ||psi||^2 = 1
                  (0 < norm_margin < 1). States generated inside the band
                  are pushed outside.
    seed        : RNG seed for reproducibility.

    Returns
    -------
    states : complex ndarray (n_samples, dim), all outside the band.
    """

    # Validation
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got: {n_samples}")

    if dim <= 0:
        raise ValueError(f"dim must be > 0, got: {dim}")

    if not (0.0 < norm_margin < 1.0):
        raise ValueError(f"norm_margin must be in (0, 1), got: {norm_margin}")

    valid_strategies = ["scaling", "noise", "direct", "extreme", "mixed"]
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown strategy '{strategy}'. " f"Choices: {valid_strategies}"
        )

    rng = np.random.default_rng(seed)

    # === SCALING STRATEGY ===
    if strategy == "scaling":
        # Start from valid states
        states_valid = generate_valid_states(
            n_samples=n_samples,
            dim=dim,
            strategy="random",
            seed=int(
                rng.integers(0, int(1e9))
            ),  # Distinct random seed (cast to Python int)
        )

        # Draw scaling factors k
        # The interval [0.95, 1.05] is avoided to prevent ambiguity
        k_min, k_max = scale_range

        # Draw k uniformly in [k_min, k_max]
        factors = rng.uniform(k_min, k_max, size=n_samples)

        # Exclude the interval [0.95, 1.05]
        # If k lands inside, push it out
        mask_ambiguous = (factors >= 0.95) & (factors <= 1.05)
        n_ambiguous = mask_ambiguous.sum()

        if n_ambiguous > 0:
            # Replace ambiguous values with unambiguous ones
            # 50% below 0.95, 50% above 1.05
            new_factors = np.where(
                rng.random(n_ambiguous) < 0.5,
                rng.uniform(k_min, 0.95, size=n_ambiguous),
                rng.uniform(1.05, k_max, size=n_ambiguous),
            )
            factors[mask_ambiguous] = new_factors

        # Apply the scaling
        # Broadcasting: (n_samples,) x (n_samples, dim)
        states = states_valid * factors[:, np.newaxis]

    # === NOISE STRATEGY ===
    elif strategy == "noise":
        # Start from valid states
        states_valid = generate_valid_states(
            n_samples=n_samples,
            dim=dim,
            strategy="random",
            seed=int(rng.integers(0, int(1e9))),
        )

        # Draw complex noise
        noise_real = rng.normal(0, noise_level, size=(n_samples, dim))
        noise_imag = rng.normal(0, noise_level, size=(n_samples, dim))
        noise = noise_real + 1j * noise_imag

        # Add the noise (WITHOUT renormalizing!)
        states = states_valid + noise

    # === DIRECT STRATEGY ===
    elif strategy == "direct":
        # Generate directly without normalizing
        real_parts = rng.normal(0, 1, size=(n_samples, dim))
        imag_parts = rng.normal(0, 1, size=(n_samples, dim))
        states = real_parts + 1j * imag_parts

        # Apply a random scaling to spread ||psi||^2
        scale_factors = rng.uniform(0.1, 3.0, size=n_samples)
        states = states * scale_factors[:, np.newaxis]

    # === EXTREME STRATEGY ===
    elif strategy == "extreme":
        # Pathological cases: near-null, huge, unbalanced.
        # Exposed as a first-class strategy for the multiclass
        # invalidity-cause diagnosis (milestone 4).
        states = _generate_extreme_states(n_samples, dim, rng)

    # === MIXED STRATEGY ===
    elif strategy == "mixed":
        states = np.zeros((n_samples, dim), dtype=complex)

        # Sub-strategy split
        n_extreme = int(n_samples * extreme_prob)
        n_remaining = n_samples - n_extreme

        # Distribute the rest among scaling, noise, direct
        n_scaling = n_remaining // 3
        n_noise = n_remaining // 3
        n_direct = n_remaining - n_scaling - n_noise

        idx = 0

        # 1. Extreme cases
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

        # Shuffle randomly
        rng.shuffle(states)

    # === CLASS-BOUNDARY GUARANTEE (F2) ===
    # Push out of the forbidden band [1 - margin, 1 + margin] any state
    # whose squared norm landed inside (possible for noise/direct).
    # Vectorized: for each offending state pick a target
    # ||psi||^2_target = 1 +/- margin*u with u ~ U(1, 2) (random side),
    # then multiply the state by sqrt(target / ||psi||^2) - scaling a
    # vector by f multiplies its squared norm by f^2.
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
            # Near-null state
            states[i] = rng.normal(0, 0.01, dim) + 1j * rng.normal(0, 0.01, dim)

        elif case_type == "huge":
            # Very large state
            states[i] = rng.normal(10, 5, dim) + 1j * rng.normal(10, 5, dim)

        elif case_type == "unbalanced":
            # One huge component, the others small
            dominant_idx = rng.integers(0, dim)
            states[i] = rng.normal(0, 0.1, dim) + 1j * rng.normal(0, 0.1, dim)
            states[i, dominant_idx] = rng.uniform(50, 100) + 1j * rng.uniform(50, 100)

    return states


def get_invalid_strategy_info() -> dict:

    info = {
        "scaling": (
            "Multiplies valid states by a factor k != 1. "
            "Control: k in [k_min, k_max] avoiding [0.95, 1.05]. "
            "Produces: ||psi||^2 = k^2. "
            "Recommended for general use."
        ),
        "noise": (
            "Adds noise to valid states without renormalizing. "
            "The noise_level parameter controls the intensity. "
            "Produces 'almost valid' states (useful for robustness)."
        ),
        "direct": (
            "Generates coefficients directly without normalization. "
            "Produces a wide ||psi||^2 distribution. "
            "Good diversity."
        ),
        "mixed": (
            "Combines scaling, noise, direct + extreme cases. "
            "The extreme_prob parameter controls the outlier share. "
            "Produces a highly diverse dataset. "
            "Recommended for the final dataset."
        ),
    }
    return info


def print_invalid_strategy_info():

    info = get_invalid_strategy_info()

    logger.info("=" * 70)
    logger.info("INVALID-STATE GENERATION STRATEGIES")
    logger.info("=" * 70)

    for strategy, description in info.items():
        logger.info(f"\n {strategy.upper()}")
        logger.info(f"   {description}")

    logger.info("\n" + "=" * 70)


# ============================================================================
# FULL DATASET CREATION
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

    # Random generator initialization
    rng = np.random.default_rng(seed)

    # Seeds for the sub-generators (for reproducibility)
    # Integer bounds, result cast to a Python int
    seed_valid = int(rng.integers(0, 1_000_000_000)) if seed is not None else None
    seed_invalid = int(rng.integers(0, 1_000_000_000)) if seed is not None else None

    # === VALID STATE GENERATION ===
    logger.info(f"Generating {n_valid} valid states (strategy: {valid_strategy})...")

    valid_kwargs = valid_kwargs or {}
    states_valid = generate_valid_states(
        n_samples=n_valid,
        dim=dim,
        strategy=valid_strategy,
        seed=seed_valid,
        **valid_kwargs,
    )

    # === INVALID STATE GENERATION ===
    logger.info(
        f"Generating {n_invalid} invalid states (strategy: {invalid_strategy})..."
    )

    invalid_kwargs = invalid_kwargs or {}
    states_invalid = generate_invalid_states(
        n_samples=n_invalid,
        dim=dim,
        strategy=invalid_strategy,
        seed=seed_invalid,
        **invalid_kwargs,
    )

    # === DATAFRAME CONSTRUCTION ===
    logger.info("Building the DataFrame...")

    # Combine the two sets
    all_states = np.vstack([states_valid, states_invalid])
    n_total = n_valid + n_invalid

    # Build the labels
    labels = np.concatenate(
        [
            np.ones(n_valid, dtype=int),  # 1 for valid
            np.zeros(n_invalid, dtype=int),  # 0 for invalid
        ]
    )

    # Squared norms
    norms_squared = np.sum(np.abs(all_states) ** 2, axis=1)

    # Build the data dictionary
    data_dict = {}

    # state_id column
    data_dict["state_id"] = np.arange(n_total)

    # Ensure all_states is a complex ndarray for safe operations
    all_states = np.asarray(all_states, dtype=complex)

    # c{i}_real and c{i}_imag columns (numpy functions avoid typing issues)
    for i in range(dim):
        data_dict[f"c{i}_real"] = np.real(all_states[:, i])
        data_dict[f"c{i}_imag"] = np.imag(all_states[:, i])

    # norm_squared column
    data_dict["norm_squared"] = norms_squared

    # is_valid column (target)
    data_dict["is_valid"] = labels

    # Build the DataFrame
    df = pd.DataFrame(data_dict)

    # === SHUFFLE ===
    if shuffle:
        logger.info("Shuffling the dataset...")
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        # Reassign state_id after the shuffle
        df["state_id"] = np.arange(len(df))

    # === STATISTICS ===
    logger.info("\n" + "=" * 70)
    logger.info("DATASET CREATED")
    logger.info("=" * 70)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Dimension: {dim}")
    logger.info(f"Valid states: {n_valid} ({n_valid/n_total*100:.1f}%)")
    logger.info(f"Invalid states: {n_invalid} ({n_invalid/n_total*100:.1f}%)")
    logger.info(f"\n is_valid distribution:")
    logger.info(df["is_valid"].value_counts().sort_index())

    logger.info(f"\n norm_squared statistics:")
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
    Multiclass dataset: each invalid state carries the label of the CAUSE
    of its invalidity (milestone 4 - diagnosis, not just detection).

    Why this is a genuine learning problem
    --------------------------------------
    Unlike validity (a deterministic function of the norm), the cause is
    NOT computable from the amplitudes: a state of norm 1.3 may come from
    a scaling (k ~ 1.14), from additive noise or from direct generation.
    The cause distributions overlap - the classifier must exploit fine
    statistical signatures (component profile, residual correlations with
    a normalized state). This is the regime where ML earns its place
    against the threshold test.

    Engineering analogy: a satellite FDIR system (Fault Detection,
    Isolation and Recovery) does not stop at detecting that a sensor
    drifts - it must isolate the cause (calibration bias? abnormal noise?
    outright failure?) to pick the appropriate recovery.

    Parameters
    ----------
    n_valid     : number of valid states (cause "valid").
    n_per_cause : number of states PER invalid cause
                  ("scaling", "noise", "direct", "extreme").
    dim         : Hilbert-space dimension.
    norm_margin : F2 boundary guarantee (see generate_invalid_states).
    seed        : RNG seed for reproducibility.
    shuffle     : final DataFrame shuffle.

    Returns
    -------
    DataFrame in the standard schema + ``cause`` (str) and ``is_valid``
    columns.
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
        "Multiclass dataset created: %d states (%d valid, %d per invalid cause)",
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

    # Create the folder if needed
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Full path
    filepath = data_path / filename

    # Save
    df.to_csv(filepath, index=False)

    # File size
    file_size = filepath.stat().st_size / 1024  # in KB

    logger.info(f"\n Dataset saved:")
    logger.info(f"   Path: {filepath}")
    logger.info(f"   Size: {file_size:.2f} KB")
    logger.info(f"   Rows: {len(df)}")
    logger.info(f"   Columns: {len(df.columns)}")

    return filepath


def load_dataset(
    filename: str = "quantum_states_dataset.csv", data_dir: str = "data/processed"
) -> pd.DataFrame:

    filepath = Path(data_dir) / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"File {filepath} does not exist. "
            f"Use create_dataset() then save_dataset() first."
        )

    df = pd.read_csv(filepath)

    logger.info(f"Dataset loaded from {filepath}")
    logger.info(f"Shape: {df.shape}")

    return df


def get_dataset_info(df: pd.DataFrame) -> dict:

    # Infer the dimension
    # Columns: state_id, c0_real, c0_imag, ..., c{d-1}_real, c{d-1}_imag, norm_squared, is_valid
    # Number of c{i}_* columns = 2*dim
    n_c_columns = len([col for col in df.columns if col.startswith("c")])
    dim = n_c_columns // 2

    info = {
        "n_samples": len(df),
        "n_features": n_c_columns + 1,  # +1 for norm_squared
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
    logger.info("DATASET INFORMATION")
    logger.info("=" * 70)
    logger.info(f"\n Size:")
    logger.info(f"   Samples: {info['n_samples']}")
    logger.info(f"   Features: {info['n_features']}")
    logger.info(f"   Dimension: {info['dim']}")

    logger.info(f"\n Class distribution:")
    logger.info(
        f"   Valid (1): {info['n_valid']} ({info['n_valid']/info['n_samples']*100:.1f}%)"
    )
    logger.info(
        f"   Invalid (0): {info['n_invalid']} ({info['n_invalid']/info['n_samples']*100:.1f}%)"
    )
    logger.info(f"   Ratio: {info['balance_ratio']:.3f}")

    if abs(info["balance_ratio"] - 1.0) < 0.05:
        logger.info(f"    Well-balanced dataset")
    else:
        logger.info(f"     Unbalanced dataset (ideal: ratio ~ 1.0)")

    logger.info(f"\n norm_squared statistics:")
    for key, value in info["norm_stats"].items():
        logger.info(f"   {key:8s}: {value:.6f}")

    logger.info("=" * 70)


if __name__ == "__main__":
    # This block only runs with: python src/qsv/data_generation.py
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("Testing the data_generation module\n")

    # Show the available strategies
    print_strategy_info()

    # Test the 3 strategies
    logger.info("\n" + "=" * 70)
    logger.info("GENERATION TESTS")
    logger.info("=" * 70)

    dim = 3
    n_samples = 5

    for strategy in ["random", "dirichlet", "basis"]:
        logger.info(f"\n--- Strategy: {strategy} ---")

        states = generate_valid_states(
            n_samples=n_samples, dim=dim, strategy=strategy, seed=42
        )

        logger.info(f"Array shape: {states.shape}")
        logger.info(f"Data type: {states.dtype}")

        # Check
        all_valid, norms = verify_normalization(states)
        logger.info(f"All states normalized? {all_valid}")
        logger.info(f"Squared norms: {norms}")

        # Show the first 2 states
        logger.info(f"\nFirst 2 states:")
        for i in range(min(2, n_samples)):
            logger.info(f"  State {i}: {states[i]}")
            logger.info(f"    ||psi||^2 = {norms[i]:.10f}")
