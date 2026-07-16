"""Tests for qsv/density.py - the three conditions and the noisy thresholds."""

import numpy as np
import pytest

from qsv.density import (
    add_tomography_noise,
    from_state_vector,
    make_invalid_density,
    random_density_matrix,
    validate_density_matrix,
)

DIM = 4


def test_ginibre_states_are_valid():
    for rank in (1, 2, DIM):
        rho = random_density_matrix(DIM, rank=rank, seed=3)
        r = validate_density_matrix(rho)
        assert r.valid and r.error_type == "ok"
        assert r.trace == pytest.approx(1.0)


def test_pure_state_bridge():
    """rho = |psi><psi| of a valid vector: purity 1, entropy 0, valid."""
    rho = from_state_vector([0.6, 0.8, 0.0, 0.0], [0.0] * 4)
    r = validate_density_matrix(rho)
    assert r.valid
    assert r.purity == pytest.approx(1.0)
    assert r.von_neumann_entropy == pytest.approx(0.0, abs=1e-9)


def test_rank1_ginibre_is_pure():
    r = validate_density_matrix(random_density_matrix(DIM, rank=1, seed=5))
    assert r.purity == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize(
    "cause,expected",
    [
        ("trace", "trace"),
        ("nonhermitian", "hermiticity"),
        ("nonpositive", "positivity"),
    ],
)
def test_invalid_causes_named(cause, expected):
    bad = make_invalid_density(DIM, cause=cause, magnitude=0.2, seed=7)
    r = validate_density_matrix(bad)
    assert not r.valid
    assert expected in r.error_type


def test_nonpositive_violation_is_guaranteed():
    """The F2-margin lesson applied to positivity: exact planted eigenvalue."""
    bad = make_invalid_density(DIM, cause="nonpositive", magnitude=0.2, seed=1)
    ev = np.linalg.eigvalsh(bad)
    assert ev.min() == pytest.approx(-0.2, abs=1e-9)
    assert np.trace(bad).real == pytest.approx(1.0, abs=1e-9)


def test_noisy_mode_accepts_valid_despite_negative_eigenvalues():
    """The tomography fact: small negative eigenvalues must NOT reject."""
    rho = random_density_matrix(DIM, rank=1, seed=11)  # pure: lam_min = 0
    noisy = add_tomography_noise(rho, n_shots=200, seed=2)
    ev = np.linalg.eigvalsh((noisy + noisy.conj().T) / 2)
    assert ev.min() < 0  # generic negative eigenvalue is present
    r = validate_density_matrix(noisy, n_shots=200)
    assert r.valid  # and the calibrated threshold tolerates it


def test_noisy_mode_catches_planted_violation():
    bad = make_invalid_density(DIM, cause="nonpositive", magnitude=0.2, seed=3)
    noisy = add_tomography_noise(bad, n_shots=1000, seed=4)
    r = validate_density_matrix(noisy, n_shots=1000)
    assert not r.valid and "positivity" in r.error_type


def test_exact_mode_is_strict():
    rho = random_density_matrix(DIM, seed=9) * 1.001  # tiny trace error
    assert not validate_density_matrix(rho).valid


def test_tomography_noise_stays_hermitian_and_reproducible():
    rho = random_density_matrix(DIM, seed=6)
    a = add_tomography_noise(rho, n_shots=500, seed=8)
    b = add_tomography_noise(rho, n_shots=500, seed=8)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a, a.conj().T, atol=1e-12)


def test_input_validation():
    with pytest.raises(ValueError):
        validate_density_matrix(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        random_density_matrix(0)
    with pytest.raises(ValueError):
        random_density_matrix(4, rank=5)
    with pytest.raises(ValueError):
        make_invalid_density(4, cause="ghost")
    with pytest.raises(ValueError):
        validate_density_matrix(random_density_matrix(2, seed=1), n_shots=0)


def test_ducktyped_density_object():
    class FakeDensityMatrix:
        def __init__(self, data):
            self.data = data

    rho = random_density_matrix(DIM, seed=13)
    assert validate_density_matrix(FakeDensityMatrix(rho)).valid
