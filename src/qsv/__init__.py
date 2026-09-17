"""
Quantum State Validator (qsv) - physics-based validation of quantum state
vectors, with an honest measurement-noise model behind every decision rule.

Two distinct, combinable usage modes:

1. Library (this package, no service required):

       from qsv import validate_state, preparation_qa
       result = validate_state(real, imag, n_shots=500)

2. HTTP service (thin wrapper around the same logic):

       uvicorn qsv.api:app        # then POST /validate, /preparation-qa

Scientific background: executed notebooks 08-12 in the repository.
"""

from qsv.density import DensityValidationResult, validate_density_matrix
from qsv.tomography import (
    CountingResult,
    NormEstimate,
    estimate_norm_squared,
    sample_counts,
    validate_counts,
)
from qsv.validators import (
    PreparationResult,
    ValidationResult,
    preparation_qa,
    validate_state,
)

__version__ = "0.5.0"

__all__ = [
    "validate_state",
    "preparation_qa",
    "validate_density_matrix",
    "validate_counts",
    "sample_counts",
    "estimate_norm_squared",
    "DensityValidationResult",
    "ValidationResult",
    "PreparationResult",
    "CountingResult",
    "NormEstimate",
    "__version__",
]
