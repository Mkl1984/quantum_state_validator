"""
Quantum State Validator - prediction API.

Design note
-----------
The API ships the validators that WON the milestone-4 benchmark
(notebooks 08-12), not a trained ML model:

- anonymous validation: bias-corrected two-sided threshold test on the
  estimated norm (notebook 08 - matches a 100-tree Random Forest);
- known-target preparation QA: the (norm, fidelity) statistic pair
  (notebook 11 - each statistic alone is blind to one error class).

A statistic is auditable, sizeable against a measurement budget
(notebook 12) and has no training-drift failure mode. Serving one is an
engineering decision, not a shortcut. Regimes where a learned component
would earn its place (calibration drift, notebook 10) are out of scope
for this first service and tracked in the ROADMAP.

Run locally:  uvicorn src.api:app --reload
Interactive docs are generated at /docs (OpenAPI).
"""

import logging
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import sigma_from_shots

logger = logging.getLogger(__name__)

__version__ = "0.5.0-dev"

# Exact-arithmetic tolerance: way above float64 rounding, way below any
# physical margin (see verify_normalization in src/data_generation.py).
EXACT_TOLERANCE = 1e-6

app = FastAPI(
    title="Quantum State Validator",
    version=__version__,
    description=(
        "Physics-based validation of quantum state vectors. "
        "Statistics and decision rules come from the executed notebooks 08-12."
    ),
)


class StateInput(BaseModel):
    real: List[float] = Field(..., description="Real parts of the amplitudes")
    imag: List[float] = Field(..., description="Imaginary parts of the amplitudes")
    n_shots: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Measurement budget N if the amplitudes come from finite-shot "
            "tomography (noise sigma = 1/(2*sqrt(N))). Omit for exact data."
        ),
    )
    margin: float = Field(
        0.05, gt=0, lt=1, description="Validity band half-width on |norm^2 - 1|"
    )


class PreparationInput(StateInput):
    target_real: List[float] = Field(..., description="Real parts of the target")
    target_imag: List[float] = Field(..., description="Imaginary parts of the target")
    fidelity_threshold: float = Field(
        0.95, gt=0, lt=1, description="Minimum fidelity to accept the pointing"
    )


def _to_state(real: List[float], imag: List[float]) -> np.ndarray:
    if not real or len(real) != len(imag):
        raise HTTPException(
            status_code=422,
            detail=f"real and imag must be non-empty and of equal length "
            f"(got {len(real)} and {len(imag)})",
        )
    return np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/validate")
def validate(payload: StateInput):
    """
    Anonymous state validation: is ||psi||^2 = 1?

    Exact mode (no n_shots): strict check |norm^2 - 1| <= 1e-6 - this is a
    computation, not an inference (the notebook 07 lesson).
    Noisy mode (n_shots given): bias-corrected threshold test from notebook
    08, statistic |norm^2 - 1 - 2*d*sigma^2| against margin/2, with a
    budget-sufficiency warning based on the notebook 12 sizing curve.
    """
    state = _to_state(payload.real, payload.imag)
    d = state.size
    norm2 = float(np.sum(np.abs(state) ** 2))

    if payload.n_shots is None:
        deviation = abs(norm2 - 1.0)
        valid = deviation <= EXACT_TOLERANCE
        return {
            "mode": "exact",
            "valid": valid,
            "norm_squared": norm2,
            "deviation": deviation,
            "tolerance": EXACT_TOLERANCE,
            "explanation": (
                "Exact amplitudes: validity is computed, not inferred. "
                f"|norm^2 - 1| = {deviation:.3e} vs tolerance {EXACT_TOLERANCE:.0e}."
            ),
        }

    # float() casts: sigma_from_shots returns np.float64, and numpy scalars
    # (np.bool in particular) are not JSON-serialisable by FastAPI.
    sigma = float(sigma_from_shots(payload.n_shots))
    bias = 2 * d * sigma**2
    statistic = abs(norm2 - 1.0 - bias)
    threshold = payload.margin / 2
    # First-order std of the norm estimate for a state near the unit sphere.
    norm_noise_std = 2 * sigma
    budget_ok = bool(norm_noise_std <= threshold)

    return {
        "mode": "noisy",
        "valid": bool(statistic <= threshold),
        "norm_squared": norm2,
        "statistic": statistic,
        "threshold": threshold,
        "sigma": sigma,
        "bias_correction": bias,
        "budget_sufficient": budget_ok,
        "explanation": (
            f"Finite-shot tomography (N={payload.n_shots}, sigma={sigma:.4f}). "
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
    }


@app.post("/preparation-qa")
def preparation_qa(payload: PreparationInput):
    """
    Known-target preparation QA: the two-channel monitor from notebook 11.

    Gain channel: |norm^2 - 1| vs margin (blind to pointing errors).
    Pointing channel: fidelity to the target (blind to gain errors).
    Both must pass; the failing channel names the error type.
    """
    state = _to_state(payload.real, payload.imag)
    target = _to_state(payload.target_real, payload.target_imag)
    if state.size != target.size:
        raise HTTPException(
            status_code=422,
            detail=f"state and target dimensions differ "
            f"({state.size} vs {target.size})",
        )

    norm2 = float(np.sum(np.abs(state) ** 2))
    target_norm2 = float(np.sum(np.abs(target) ** 2))
    if abs(target_norm2 - 1.0) > 1e-6:
        raise HTTPException(
            status_code=422,
            detail=f"target must be normalised (got norm^2 = {target_norm2:.6f})",
        )
    if norm2 < 1e-12:
        raise HTTPException(status_code=422, detail="state has near-zero norm")

    fidelity = float(np.abs(np.sum(np.conj(target) * state)) ** 2 / norm2)

    gain_ok = abs(norm2 - 1.0) <= payload.margin
    pointing_ok = fidelity >= payload.fidelity_threshold

    if gain_ok and pointing_ok:
        error_type = "ok"
    elif gain_ok:
        error_type = "pointing_error"
    elif pointing_ok:
        error_type = "gain_error"
    else:
        error_type = "gain_and_pointing_error"

    return {
        "prep_ok": bool(gain_ok and pointing_ok),
        "error_type": error_type,
        "norm_squared": norm2,
        "fidelity": fidelity,
        "gain_channel_ok": gain_ok,
        "pointing_channel_ok": pointing_ok,
        "explanation": (
            f"Two-channel monitor: gain |norm^2 - 1| = {abs(norm2 - 1.0):.4f} "
            f"(margin {payload.margin}), pointing fidelity = {fidelity:.4f} "
            f"(threshold {payload.fidelity_threshold}). Each channel is blind "
            "to the other error mechanism (notebook 11)."
        ),
    }
