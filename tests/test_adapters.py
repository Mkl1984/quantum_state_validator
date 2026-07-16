"""Tests for qsv/adapters.py - duck-typed framework conversion."""

import numpy as np
import pytest

from qsv.adapters import preparation, to_amplitudes, validate


class FakeStatevector:
    """Mimics qiskit.quantum_info.Statevector (exposes .data)."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=complex)


VALID = np.array([0.6, 0.8j, 0.0, 0.0])  # norm^2 = 1, complex


def test_to_amplitudes_ndarray_and_lists():
    re, im = to_amplitudes(VALID)
    np.testing.assert_allclose(re, [0.6, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(im, [0.0, 0.8, 0.0, 0.0])
    re2, im2 = to_amplitudes([1.0, 0.0])  # real list
    np.testing.assert_allclose(re2, [1.0, 0.0])
    np.testing.assert_allclose(im2, [0.0, 0.0])


def test_to_amplitudes_statevector_like():
    re, im = to_amplitudes(FakeStatevector(VALID))
    np.testing.assert_allclose(re, VALID.real)
    np.testing.assert_allclose(im, VALID.imag)


def test_to_amplitudes_rejects_bad_inputs():
    with pytest.raises(ValueError):
        to_amplitudes(np.zeros((2, 2)))  # not 1-D
    with pytest.raises(ValueError):
        to_amplitudes([])
    with pytest.raises(TypeError):
        to_amplitudes(np.array(["a", "b"]))


def test_validate_matches_core():
    r = validate(FakeStatevector(VALID))
    assert r.valid is True and r.mode == "exact"
    r = validate(1.2 * VALID, n_shots=500)
    assert r.mode == "noisy" and r.valid is False


def test_preparation_matches_core():
    r = preparation(1.3 * VALID, VALID)
    assert r.error_type == "gain_error"
    assert r.fidelity == pytest.approx(1.0)
