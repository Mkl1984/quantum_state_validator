"""
Tests for qsv/tomography.py - the exact finite-shot counting model.

The two tests that matter most are
``test_estimator_is_unbiased`` (the property the Gaussian model does not have)
and ``test_conditional_frequencies_are_blind_to_the_norm`` (the reason a norm
check needs a calibrated exposure and not just outcome frequencies).
"""

import numpy as np
import pytest

from qsv.features import sigma_from_shots
from qsv.tomography import (
    conditional_probabilities,
    estimate_norm_squared,
    poisson_two_sided_p_value,
    sample_counts,
    validate_counts,
)

DIM = 4


def reference_state(dim=DIM, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_counts_are_non_negative_integers():
    psi = reference_state()
    counts = sample_counts(psi.real, psi.imag, n_shots=500, seed=1)
    assert counts.shape == (DIM,)
    assert np.issubdtype(counts.dtype, np.integer)
    assert (counts >= 0).all()


def test_sampling_is_reproducible():
    psi = reference_state()
    a = sample_counts(psi.real, psi.imag, n_shots=500, seed=4)
    b = sample_counts(psi.real, psi.imag, n_shots=500, seed=4)
    np.testing.assert_array_equal(a, b)


def test_sampling_validation():
    psi = reference_state()
    with pytest.raises(ValueError):
        sample_counts(psi.real, psi.imag, n_shots=0)
    with pytest.raises(ValueError):
        sample_counts([1.0, 0.0], [0.0])
    with pytest.raises(ValueError):
        sample_counts([], [])


# ---------------------------------------------------------------------------
# Estimator - unbiasedness and variance law
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_norm2", [0.90, 1.00, 1.10])
def test_estimator_is_unbiased(true_norm2):
    """
    E[||psi_hat||^2] = ||psi||^2, with NO 2*d*sigma^2 correction.

    The Gaussian amplitude-noise model carries a bias of 2*d*sigma^2 = d/(2N),
    which at d = 4, N = 1000 is 2e-3 - about 4 Monte-Carlo standard errors of
    this test. So this tolerance really does separate the two models.
    """
    psi = reference_state() * np.sqrt(true_norm2)
    n_shots, reps = 1000, 4000
    values = np.array(
        [
            estimate_norm_squared(
                sample_counts(psi.real, psi.imag, n_shots, seed=i), n_shots
            ).value
            for i in range(reps)
        ]
    )
    mc_error = values.std() / np.sqrt(reps)
    assert abs(values.mean() - true_norm2) < 4 * mc_error


@pytest.mark.parametrize("true_norm2", [0.90, 1.00, 1.10])
def test_estimator_variance_law(true_norm2):
    """sd(||psi_hat||^2) = sqrt(||psi||^2 / N)."""
    psi = reference_state() * np.sqrt(true_norm2)
    n_shots, reps = 1000, 4000
    values = np.array(
        [
            estimate_norm_squared(
                sample_counts(psi.real, psi.imag, n_shots, seed=i), n_shots
            ).value
            for i in range(reps)
        ]
    )
    assert values.std() == pytest.approx(np.sqrt(true_norm2 / n_shots), rel=0.06)


def test_spread_matches_the_gaussian_model_at_leading_order():
    """
    At ||psi||^2 = 1 both models give sd = 1/sqrt(N): the Gaussian
    simplification was right about the spread. Only the bias differs.
    """
    n_shots = 1000
    gaussian_sd = 2 * sigma_from_shots(n_shots)  # 2*sigma*||psi|| at ||psi|| = 1
    counting_sd = np.sqrt(1.0 / n_shots)
    assert counting_sd == pytest.approx(gaussian_sd, rel=1e-12)


def test_plug_in_variance_matches_the_estimate():
    psi = reference_state()
    n_shots = 2000
    est = estimate_norm_squared(
        sample_counts(psi.real, psi.imag, n_shots, seed=2), n_shots
    )
    assert est.variance == pytest.approx(est.total_counts / n_shots**2)
    assert est.std_error == pytest.approx(np.sqrt(est.variance))
    assert est.value == pytest.approx(est.total_counts / n_shots)


def test_estimator_validation():
    with pytest.raises(ValueError):
        estimate_norm_squared([1, 2, 3], n_shots=0)
    with pytest.raises(ValueError):
        estimate_norm_squared([], n_shots=10)
    with pytest.raises(ValueError):
        estimate_norm_squared([1, -2], n_shots=10)


# ---------------------------------------------------------------------------
# Non-identifiability - why a calibrated exposure is required
# ---------------------------------------------------------------------------


def test_conditional_frequencies_are_blind_to_the_norm():
    """
    Rescaling the state leaves the conditional outcome law unchanged, so no
    statistic built from frequencies alone can decide validity. The norm lives
    entirely in the total count.
    """
    psi = reference_state()
    exact = np.abs(psi) ** 2
    for scale in (0.85, 1.0, 1.20):
        counts = sample_counts(
            (psi * np.sqrt(scale)).real,
            (psi * np.sqrt(scale)).imag,
            n_shots=400_000,
            seed=11,
        )
        np.testing.assert_allclose(conditional_probabilities(counts), exact, atol=3e-3)


def test_conditional_probabilities_handle_zero_counts():
    """No click at all: fall back to uniform instead of dividing by zero."""
    out = conditional_probabilities(np.zeros(DIM, dtype=int))
    np.testing.assert_allclose(out, np.full(DIM, 1 / DIM))


# ---------------------------------------------------------------------------
# Exact Poisson p-value
# ---------------------------------------------------------------------------


def test_p_value_is_one_at_the_mean():
    assert poisson_two_sided_p_value(100, 100.0) == pytest.approx(1.0, abs=1e-9)


def test_p_value_decreases_with_deviation():
    p = [poisson_two_sided_p_value(k, 100.0) for k in (105, 120, 140, 160)]
    assert all(np.diff(p) < 0)
    assert p[-1] < 1e-6


def test_p_value_bounds_and_validation():
    for k in (0, 1, 50, 100, 300):
        assert 0.0 <= poisson_two_sided_p_value(k, 100.0) <= 1.0
    with pytest.raises(ValueError):
        poisson_two_sided_p_value(10, 0.0)
    with pytest.raises(ValueError):
        poisson_two_sided_p_value(-1, 10.0)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def test_validate_counts_accepts_a_clean_unit_norm():
    """Exact expected counts for a unit-norm state must be accepted."""
    psi = reference_state()
    n_shots = 10_000
    counts = np.round(n_shots * np.abs(psi) ** 2).astype(int)
    result = validate_counts(counts, n_shots)
    assert result.valid
    assert result.norm_squared == pytest.approx(1.0, abs=1e-3)
    assert result.budget_ok


def test_validate_counts_rejects_a_scaled_state():
    psi = reference_state() * np.sqrt(1.30)
    n_shots = 10_000
    counts = np.round(n_shots * np.abs(psi) ** 2).astype(int)
    result = validate_counts(counts, n_shots)
    assert not result.valid
    assert result.norm_squared == pytest.approx(1.30, abs=1e-3)
    assert result.p_value < 1e-6


def test_budget_flag_follows_the_sizing_rule():
    """budget_ok iff 1/sqrt(N) <= margin/2, i.e. N >= 4/margin^2 = 1600."""
    counts = np.array([400, 400, 400, 400])
    assert not validate_counts(counts, n_shots=1599).budget_ok
    assert validate_counts(counts, n_shots=1600).budget_ok
    assert "Budget warning" in validate_counts(counts, n_shots=100).explanation


def test_statistic_carries_no_bias_correction():
    """The statistic is |n2 - 1| exactly - no 2*d*sigma^2 term."""
    counts = np.array([300, 300, 300, 300])
    n_shots = 1000
    result = validate_counts(counts, n_shots)
    assert result.statistic == pytest.approx(abs(1200 / 1000 - 1.0))


def test_validate_counts_validation():
    counts = np.array([250, 250, 250, 250])
    with pytest.raises(ValueError):
        validate_counts(counts, n_shots=1000, margin=0.0)
    with pytest.raises(ValueError):
        validate_counts(counts, n_shots=1000, margin=1.0)
    with pytest.raises(ValueError):
        validate_counts(counts, n_shots=1000, alpha=0.0)


def test_result_is_serialisable():
    counts = np.array([500, 500, 500, 500])
    payload = validate_counts(counts, n_shots=2000).to_dict()
    assert isinstance(payload["valid"], bool)
    assert isinstance(payload["norm_squared"], float)
    assert isinstance(payload["total_counts"], int)
