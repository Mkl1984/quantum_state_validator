"""
Density-matrix validation - mixed states enter the validator.

A density matrix rho (d x d complex) describes a general quantum state
(pure or mixed). It is physically valid iff three conditions hold:

1. Hermiticity:      rho = rho^dagger
2. Unit trace:       Tr(rho) = 1        (probability conservation)
3. Positivity:       all eigenvalues >= 0  (no negative probabilities)

Purity Tr(rho^2) in [1/d, 1] separates pure states (=1) from mixed ones,
and the VON NEUMANN entropy S = -sum(lambda_i ln lambda_i) is now the real
thing (for state vectors it was identically zero - the naming lesson of
the README applies here in reverse).

The tomography fact that drives the noisy mode
----------------------------------------------
Linear-inversion tomography returns Hermitian, unit-trace estimates that
GENERICALLY have small negative eigenvalues - rejecting them naively
rejects essentially every real reconstruction. Under Hermitian
(GUE-type) noise of per-element scale sigma, the eigenvalues of a valid
rho are perturbed by up to ~2*sigma*sqrt(d) (semicircle edge), so the
noisy positivity check uses the threshold

    lambda_min >= -EIG_EDGE_FACTOR * sigma * sqrt(d)

and the trace check uses |Tr(rho) - 1| <= TRACE_FACTOR * sigma * sqrt(d)
(Var[Tr(H)] = d * sigma^2 for GUE noise). Both constants were calibrated
empirically (see notebook 14): with EIG_EDGE_FACTOR = 2.5, lambda_min of noisy VALID
states never crossed the threshold over 3600 trials (d in {2..16},
N in {100..10000}); TRACE_FACTOR = 3.0 is a 3-sigma bound by construction
(false-rejection ~0.3%, observed 0.7% at d=16 within sampling error).
"""

import logging
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from qsv.features import sigma_from_shots

logger = logging.getLogger(__name__)

__all__ = [
    "DensityValidationResult",
    "validate_density_matrix",
    "from_state_vector",
    "random_density_matrix",
    "make_invalid_density",
    "add_tomography_noise",
]

EXACT_TOLERANCE = 1e-6
EIG_EDGE_FACTOR = 2.5
TRACE_FACTOR = 3.0


@dataclass
class DensityValidationResult:
    mode: str
    valid: bool
    hermitian_ok: bool
    trace_ok: bool
    positive_ok: bool
    error_type: str
    trace: float
    min_eigenvalue: float
    hermiticity_deviation: float
    purity: float
    von_neumann_entropy: float
    explanation: str
    sigma: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _as_matrix(rho) -> np.ndarray:
    if hasattr(rho, "data") and not isinstance(rho, np.ndarray):
        rho = rho.data  # duck-typing: qiskit.quantum_info.DensityMatrix
    arr = np.asarray(rho, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"Expected a square (d, d) matrix, got shape {arr.shape}")
    return arr


def validate_density_matrix(
    rho, n_shots: Optional[int] = None, tolerance: float = EXACT_TOLERANCE
) -> DensityValidationResult:
    """
    Validate a density matrix (exact data or finite-shot tomography).

    Exact mode (n_shots=None): the three conditions are checked against
    ``tolerance`` (strictly absolute, as everywhere in qsv).
    Noisy mode: hermiticity stays a strict check (tomographic estimates are
    Hermitian by construction); trace and positivity thresholds scale with
    the noise, sigma = 1/(2*sqrt(N)) per element (see module docstring).
    """
    m = _as_matrix(rho)
    d = m.shape[0]

    herm_dev = float(np.max(np.abs(m - m.conj().T)))
    trace = complex(np.trace(m))
    # Eigenvalues on the Hermitian part (well-defined even if m is not Hermitian)
    eigvals = np.linalg.eigvalsh((m + m.conj().T) / 2)
    lam_min = float(eigvals.min())
    purity = float(np.real(np.trace(m @ m)))
    pos = np.clip(eigvals, 1e-15, None)
    entropy = float(-(pos * np.log(pos)).sum()) if lam_min > -1e-12 else float("nan")

    if n_shots is None:
        mode, sigma = "exact", None
        hermitian_ok = herm_dev <= tolerance
        trace_ok = abs(trace - 1.0) <= tolerance
        positive_ok = lam_min >= -tolerance
    else:
        if n_shots <= 0:
            raise ValueError(f"n_shots must be > 0 (got {n_shots})")
        mode = "noisy"
        sigma = float(sigma_from_shots(n_shots))
        scale = sigma * np.sqrt(d)
        hermitian_ok = herm_dev <= tolerance
        trace_ok = abs(trace - 1.0) <= TRACE_FACTOR * scale
        positive_ok = lam_min >= -EIG_EDGE_FACTOR * scale

    failed = [
        name
        for ok, name in [
            (hermitian_ok, "hermiticity"),
            (trace_ok, "trace"),
            (positive_ok, "positivity"),
        ]
        if not ok
    ]
    error_type = "ok" if not failed else "_and_".join(failed) + "_error"
    valid = not failed

    return DensityValidationResult(
        mode=mode,
        valid=valid,
        hermitian_ok=bool(hermitian_ok),
        trace_ok=bool(trace_ok),
        positive_ok=bool(positive_ok),
        error_type=error_type,
        trace=float(trace.real),
        min_eigenvalue=lam_min,
        hermiticity_deviation=herm_dev,
        purity=purity,
        von_neumann_entropy=entropy,
        explanation=(
            f"{mode} mode: |rho - rho^H|_max = {herm_dev:.2e}, "
            f"Tr = {trace.real:.6f}, lambda_min = {lam_min:.6f}, "
            f"purity = {purity:.4f}. "
            + (
                "All three conditions hold."
                if valid
                else f"Failed: {', '.join(failed)}."
            )
        ),
        sigma=sigma,
    )


def from_state_vector(real: Sequence[float], imag: Sequence[float]) -> np.ndarray:
    """rho = |psi><psi| - the pure-state bridge to the vector formalism."""
    psi = np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)
    if psi.ndim != 1 or psi.size == 0:
        raise ValueError("real/imag must be non-empty 1-D sequences")
    return np.outer(psi, psi.conj())


def random_density_matrix(
    dim: int, rank: Optional[int] = None, seed: Optional[int] = None
) -> np.ndarray:
    """
    Valid random density matrix via the Ginibre construction:
    A ~ complex Gaussian (d x r), rho = A A^dagger / Tr(A A^dagger).
    rank=1 gives a pure state, rank=d (default) a full-rank mixed state.
    """
    if dim <= 0:
        raise ValueError(f"dim must be > 0 (got {dim})")
    rank = dim if rank is None else rank
    if not (1 <= rank <= dim):
        raise ValueError(f"rank must be in [1, {dim}] (got {rank})")
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def make_invalid_density(
    dim: int, cause: str = "trace", magnitude: float = 0.2, seed: Optional[int] = None
) -> np.ndarray:
    """
    Invalid density matrices with a known cause:
    - "trace":        valid rho scaled by (1 + magnitude) - Tr != 1
    - "nonhermitian": valid rho + magnitude * (non-Hermitian perturbation)
    - "nonpositive":  unit-trace Hermitian matrix whose smallest eigenvalue
                      is GUARANTEED to equal exactly -magnitude (direct
                      spectral construction - the F2-margin lesson applied
                      to positivity: a violation without a guaranteed floor
                      is undetectable under noise)
    """
    rng = np.random.default_rng(seed)
    rho = random_density_matrix(dim, seed=int(rng.integers(1e9)))
    if cause == "trace":
        return (1.0 + magnitude) * rho
    if cause == "nonhermitian":
        g = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        antiherm = (g - g.conj().T) / 2
        # Remove the (purely imaginary) trace so ONLY hermiticity breaks -
        # otherwise the complex trace also fails the trace check and the
        # cause is no longer isolated.
        antiherm -= (np.trace(antiherm) / dim) * np.eye(dim)
        return rho + magnitude * antiherm
    if cause == "nonpositive":
        if dim < 2:
            raise ValueError("nonpositive requires dim >= 2")
        eigvals, eigvecs = np.linalg.eigh(rho)
        # Plant the violation, rescale the rest to keep Tr = 1 exactly
        rest = eigvals[1:]
        new_vals = np.concatenate([[-magnitude], rest * (1.0 + magnitude) / rest.sum()])
        return (eigvecs * new_vals) @ eigvecs.conj().T
    raise ValueError(f"Unknown cause '{cause}' (trace | nonhermitian | nonpositive)")


def add_tomography_noise(
    rho, n_shots: int = 1000, seed: Optional[int] = None
) -> np.ndarray:
    """
    Hermitian (GUE-type) additive noise of per-element scale
    sigma = 1/(2*sqrt(N)) - the density-matrix analogue of
    features.add_measurement_noise. The estimate stays Hermitian
    (as linear-inversion tomography does) but NOT necessarily positive:
    small negative eigenvalues are the expected, generic outcome.
    """
    m = _as_matrix(rho)
    d = m.shape[0]
    sigma = float(sigma_from_shots(n_shots))
    rng = np.random.default_rng(seed)
    diag = rng.normal(0.0, sigma, size=d)
    off = rng.normal(0.0, sigma / np.sqrt(2), size=(d, d)) + 1j * rng.normal(
        0.0, sigma / np.sqrt(2), size=(d, d)
    )
    h = np.diag(diag).astype(complex)
    iu = np.triu_indices(d, 1)
    h[iu] = off[iu]
    h[(iu[1], iu[0])] = off[iu].conj()
    return m + h
