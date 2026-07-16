"""
Framework adapters - one thin conversion layer, one decision core.

Architectural rule (docs/STRATEGY.md section 2): every integration converts
a foreign state representation into (real, imag) sequences and calls
qsv.validators. No adapter reimplements any decision logic.

The conversion is DUCK-TYPED, so qsv carries no dependency on any quantum
framework:

- NumPy arrays (complex or real), lists, tuples  -> used directly.
  Covers PennyLane (``qml.state()``) and Cirq (``final_state_vector``),
  which both return plain complex ndarrays.
- Objects exposing a ``data`` attribute holding the amplitudes -> unwrapped.
  Covers Qiskit's ``Statevector`` (``Statevector.data`` is a complex ndarray).

Examples
--------
>>> from qsv.adapters import validate, preparation
>>> validate(qiskit_statevector)                    # Qiskit
>>> validate(circuit_fn(params), n_shots=500)       # PennyLane / Cirq
>>> preparation(state, target, margin=0.05)         # any of the above
"""

from typing import Any, Optional, Tuple

import numpy as np

from qsv.validators import (
    PreparationResult,
    ValidationResult,
    preparation_qa,
    validate_state,
)

__all__ = ["to_amplitudes", "validate", "preparation"]


def to_amplitudes(state: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert any supported state representation to (real, imag) 1-D arrays.

    Accepts complex/real ndarrays, lists, tuples, and objects with a
    ``data`` attribute (e.g. qiskit.quantum_info.Statevector). Raises
    TypeError/ValueError on anything else.
    """
    if hasattr(state, "data") and not isinstance(state, np.ndarray):
        state = state.data
    arr = np.asarray(state)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(
            f"Expected a non-empty 1-D state vector, got shape {arr.shape}"
        )
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"State amplitudes must be numeric, got dtype {arr.dtype}")
    arr = arr.astype(complex)
    return arr.real.copy(), arr.imag.copy()


def validate(
    state: Any, n_shots: Optional[int] = None, margin: float = 0.05
) -> ValidationResult:
    """Validate any supported state representation (see qsv.validators)."""
    real, imag = to_amplitudes(state)
    return validate_state(real, imag, n_shots=n_shots, margin=margin)


def preparation(
    state: Any,
    target: Any,
    margin: float = 0.05,
    fidelity_threshold: float = 0.95,
) -> PreparationResult:
    """Known-target preparation QA for any supported representation."""
    s_re, s_im = to_amplitudes(state)
    t_re, t_im = to_amplitudes(target)
    return preparation_qa(
        s_re, s_im, t_re, t_im, margin=margin, fidelity_threshold=fidelity_threshold
    )
