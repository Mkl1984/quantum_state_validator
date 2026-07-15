"""
Pure decision logic - the library face of the Quantum State Validator.

These functions are the milestone-4 winning validators (notebooks 08-12),
usable in any Python project without the HTTP service:

    from qsv import validate_state, preparation_qa

    result = validate_state([1.0, 0, 0, 0], [0.0, 0, 0, 0])
    result = validate_state(real, imag, n_shots=500)   # noisy tomography data

The HTTP API (qsv.api) is a thin wrapper around this module: both usage
modes share exactly the same logic and can be combined freely.
"""

import logging
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

from qsv.features import sigma_from_shots

logger = logging.getLogger(__name__)

__all__ = ["ValidationResult", "PreparationResult", "validate_state", "preparation_qa"]

# Exact-arithmetic tolerance: way above float64 rounding, way below any
# physical margin (see verify_normalization in qsv.data_generation).
EXACT_TOLERANCE = 1e-6


@dataclass
class ValidationResult:
    mode: str
    valid: bool
    norm_squared: float
    statistic: float
    threshold: float
    explanation: str
    sigma: Optional[float] = None
    bias_correction: Optional[float] = None
    budget_sufficient: Optional[bool] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PreparationResult:
    prep_ok: bool
    error_type: str
    norm_squared: float
    fidelity: float
    gain_channel_ok: bool
    pointing_channel_ok: bool
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def _as_state(real: Sequence[float], imag: Sequence[float]) -> np.ndarray:
    real = np.asarray(real, dtype=float)
    imag = np.asarray(imag, dtype=float)
    if real.size == 0 or real.shape != imag.shape or real.ndim != 1:
        raise ValueError(
            f"real and imag must be non-empty 1-D sequences of equal length "
            f"(got shapes {real.shape} and {imag.shape})"
        )
    return real + 1j * imag


def validate_state(
    real: Sequence[float],
    imag: Sequence[float],
    n_shots: Optional[int] = None,
    margin: float = 0.05,
) -> ValidationResult:
    """
    Anonymous state validation: is ||psi||^2 = 1?

    Exact mode (n_shots=None): strict computation |norm^2 - 1| <= 1e-6 -
    validity on exact amplitudes is computed, not inferred (the notebook 07
    lesson). Noisy mode (n_shots given): bias-corrected threshold test from
    notebook 08, statistic |norm^2 - 1 - 2*d*sigma^2| against margin/2, with
    a budget-sufficiency flag backed by the notebook 12 sizing curve.
    """
    if not (0.0 < margin < 1.0):
        raise ValueError(f"margin must be in ]0, 1[ (got {margin})")
    state = _as_state(real, imag)
    d = state.size
    norm2 = float(np.sum(np.abs(state) ** 2))

    if n_shots is None:
        deviation = abs(norm2 - 1.0)
        return ValidationResult(
            mode="exact",
            valid=deviation <= EXACT_TOLERANCE,
            norm_squared=norm2,
            statistic=deviation,
            threshold=EXACT_TOLERANCE,
            explanation=(
                "Exact amplitudes: validity is computed, not inferred. "
                f"|norm^2 - 1| = {deviation:.3e} vs tolerance {EXACT_TOLERANCE:.0e}."
            ),
        )

    if n_shots <= 0:
        raise ValueError(f"n_shots must be > 0 (got {n_shots})")
    # float() cast: sigma_from_shots returns np.float64, and numpy scalars
    # (np.bool in particular) do not survive JSON serialisation downstream.
    sigma = float(sigma_from_shots(n_shots))
    bias = 2 * d * sigma**2
    statistic = abs(norm2 - 1.0 - bias)
    threshold = margin / 2
    # First-order std of the norm estimate for a state near the unit sphere.
    norm_noise_std = 2 * sigma
    budget_ok = bool(norm_noise_std <= threshold)

    return ValidationResult(
        mode="noisy",
        valid=bool(statistic <= threshold),
        norm_squared=norm2,
        statistic=statistic,
        threshold=threshold,
        sigma=sigma,
        bias_correction=bias,
        budget_sufficient=budget_ok,
        explanation=(
            f"Finite-shot tomography (N={n_shots}, sigma={sigma:.4f}). "
            f"Bias-corrected statistic |norm^2 - 1 - 2d*sigma^2| = {statistic:.4f} "
            f"vs threshold margin/2 = {threshold:.4f}."
            + (
                ""
                if budget_ok
                else (
                    " WARNING: norm-estimate noise (~2*sigma = "
                    f"{norm_noise_std:.4f}) exceeds the decision threshold - "
                    "the verdict is noise-dominated. Increase N "
                    "(see the sizing curve, notebook 12)."
                )
            )
        ),
    )


def preparation_qa(
    real: Sequence[float],
    imag: Sequence[float],
    target_real: Sequence[float],
    target_imag: Sequence[float],
    margin: float = 0.05,
    fidelity_threshold: float = 0.95,
) -> PreparationResult:
    """
    Known-target preparation QA: the two-channel monitor from notebook 11.

    Gain channel: |norm^2 - 1| vs margin (blind to pointing errors).
    Pointing channel: fidelity to the target (blind to gain errors).
    Both must pass; the failing channel names the error type.
    """
    if not (0.0 < margin < 1.0):
        raise ValueError(f"margin must be in ]0, 1[ (got {margin})")
    if not (0.0 < fidelity_threshold < 1.0):
        raise ValueError(
            f"fidelity_threshold must be in ]0, 1[ (got {fidelity_threshold})"
        )
    state = _as_state(real, imag)
    target = _as_state(target_real, target_imag)
    if state.size != target.size:
        raise ValueError(
            f"state and target dimensions differ ({state.size} vs {target.size})"
        )
    norm2 = float(np.sum(np.abs(state) ** 2))
    target_norm2 = float(np.sum(np.abs(target) ** 2))
    if abs(target_norm2 - 1.0) > 1e-6:
        raise ValueError(f"target must be normalised (got norm^2 = {target_norm2:.6f})")
    if norm2 < 1e-12:
        raise ValueError("state has near-zero norm")

    fidelity = float(np.abs(np.sum(np.conj(target) * state)) ** 2 / norm2)
    gain_ok = bool(abs(norm2 - 1.0) <= margin)
    pointing_ok = bool(fidelity >= fidelity_threshold)

    if gain_ok and pointing_ok:
        error_type = "ok"
    elif gain_ok:
        error_type = "pointing_error"
    elif pointing_ok:
        error_type = "gain_error"
    else:
        error_type = "gain_and_pointing_error"

    return PreparationResult(
        prep_ok=gain_ok and pointing_ok,
        error_type=error_type,
        norm_squared=norm2,
        fidelity=fidelity,
        gain_channel_ok=gain_ok,
        pointing_channel_ok=pointing_ok,
        explanation=(
            f"Two-channel monitor: gain |norm^2 - 1| = {abs(norm2 - 1.0):.4f} "
            f"(margin {margin}), pointing fidelity = {fidelity:.4f} "
            f"(threshold {fidelity_threshold}). Each channel is blind "
            "to the other error mechanism (notebook 11)."
        ),
    )
