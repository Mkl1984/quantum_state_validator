"""
Quantum State Validator - HTTP service.

Thin wrapper around qsv.validators: the library (import qsv) and this
service share exactly the same decision logic. See qsv/validators.py for
the science (milestone-4 winning validators, notebooks 08-12) and the
design rationale (no trained ML model is served).

Run locally:  uvicorn qsv.api:app --reload
Interactive docs are generated at /docs (OpenAPI).
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qsv import __version__
from qsv.validators import preparation_qa, validate_state

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quantum State Validator",
    version=__version__,
    description=(
        "Physics-based validation of quantum state vectors. "
        "Statistics and decision rules come from the executed notebooks 08-12. "
        "The same logic is importable as a library: from qsv import validate_state."
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


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/validate")
def validate_endpoint(payload: StateInput):
    """Anonymous state validation - see qsv.validators.validate_state."""
    try:
        result = validate_state(
            payload.real, payload.imag, n_shots=payload.n_shots, margin=payload.margin
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result.to_dict()


@app.post("/preparation-qa")
def preparation_qa_endpoint(payload: PreparationInput):
    """Known-target preparation QA - see qsv.validators.preparation_qa."""
    try:
        result = preparation_qa(
            payload.real,
            payload.imag,
            payload.target_real,
            payload.target_imag,
            margin=payload.margin,
            fidelity_threshold=payload.fidelity_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result.to_dict()
