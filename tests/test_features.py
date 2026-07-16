"""
Tests for qsv/features.py.

The most important test is ``test_invariant_features_are_scale_invariant``:
it mechanically verifies the property that prevents target leakage.
"""

import numpy as np
import pandas as pd
import pytest

from qsv.data_generation import create_dataset, generate_valid_states
from qsv.features import (
    add_measurement_noise,
    compute_features,
    extract_amplitudes,
    norm_squared,
    scale_invariant_features,
    scale_sensitive_features,
    sigma_from_shots,
)


@pytest.fixture()
def small_dataset() -> pd.DataFrame:
    return create_dataset(n_valid=200, n_invalid=200, dim=4, seed=42, shuffle=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_extract_amplitudes_roundtrip(small_dataset):
    states = extract_amplitudes(small_dataset)
    assert states.shape == (400, 4)
    assert states.dtype == complex
    # The dataset's norm_squared column must match the recomputation
    np.testing.assert_allclose(
        norm_squared(states), small_dataset["norm_squared"].to_numpy(), rtol=1e-10
    )


def test_extract_amplitudes_missing_columns():
    with pytest.raises(ValueError):
        extract_amplitudes(pd.DataFrame({"foo": [1.0]}))


# ---------------------------------------------------------------------------
# Scale invariance - THE anti-leakage test
# ---------------------------------------------------------------------------


def test_invariant_features_are_scale_invariant():
    """c -> k*c must change NO invariant feature."""
    states = generate_valid_states(100, dim=4, strategy="random", seed=7)
    for k in (0.1, 0.5, 2.0, 37.0):
        f_base = scale_invariant_features(states)
        f_scaled = scale_invariant_features(k * states)
        pd.testing.assert_frame_equal(f_base, f_scaled, atol=1e-9, rtol=1e-9)


def test_sensitive_features_do_leak_the_norm():
    """Counter-check: sensitive features MUST change under c -> k*c."""
    states = generate_valid_states(50, dim=4, strategy="random", seed=7)
    f_base = scale_sensitive_features(states)
    f_scaled = scale_sensitive_features(2.0 * states)
    # norm_squared exactly quadruples (k^2 = 4)
    np.testing.assert_allclose(
        f_scaled["norm_squared"], 4.0 * f_base["norm_squared"], rtol=1e-10
    )
    # purity_raw is multiplied by k^4 = 16
    np.testing.assert_allclose(
        f_scaled["purity_raw"], 16.0 * f_base["purity_raw"], rtol=1e-10
    )


def test_invariant_features_bounds():
    """H in [0, ln d], purity in [1/d, 1], participation in [1, d], max_prob in [1/d, 1]."""
    states = generate_valid_states(500, dim=4, strategy="dirichlet", seed=3)
    f = scale_invariant_features(states)
    d = 4
    assert (f["entropy_shannon"] >= -1e-9).all()
    assert (f["entropy_shannon"] <= np.log(d) + 1e-9).all()
    assert (f["purity_normalized"] >= 1 / d - 1e-9).all()
    assert (f["purity_normalized"] <= 1 + 1e-9).all()
    assert (f["participation_ratio"] >= 1 - 1e-9).all()
    assert (f["participation_ratio"] <= d + 1e-9).all()
    assert (f["max_prob"] >= 1 / d - 1e-9).all()
    assert (f["max_prob"] <= 1 + 1e-9).all()


def test_basis_states_extreme_values():
    """Pure basis state: H = 0, purity = 1, participation = 1, max_prob = 1."""
    states = generate_valid_states(10, dim=4, strategy="basis", seed=1)
    f = scale_invariant_features(states)
    np.testing.assert_allclose(f["entropy_shannon"], 0.0, atol=1e-8)
    np.testing.assert_allclose(f["purity_normalized"], 1.0, atol=1e-10)
    np.testing.assert_allclose(f["max_prob"], 1.0, atol=1e-10)


def test_near_zero_state_falls_back_to_uniform():
    """Near-null state: renormalization impossible -> uniform distribution."""
    states = np.full((1, 4), 1e-15 + 0j)
    f = scale_invariant_features(states)
    np.testing.assert_allclose(f["purity_normalized"].iloc[0], 0.25, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_features
# ---------------------------------------------------------------------------


def test_compute_features_kinds(small_dataset):
    inv = compute_features(small_dataset, kind="invariant")
    sen = compute_features(small_dataset, kind="sensitive")
    both = compute_features(small_dataset, kind="all")
    assert list(both.columns) == list(inv.columns) + list(sen.columns)
    assert "norm_deviation" not in inv.columns  # no leak on the invariant side
    with pytest.raises(ValueError):
        compute_features(small_dataset, kind="magic")


# ---------------------------------------------------------------------------
# Measurement noise
# ---------------------------------------------------------------------------


def test_sigma_from_shots_scaling():
    """sigma follows the 1/sqrt(N) law: quadrupling N halves sigma."""
    assert sigma_from_shots(400) == pytest.approx(sigma_from_shots(100) / 2)
    with pytest.raises(ValueError):
        sigma_from_shots(0)


def test_add_measurement_noise_properties(small_dataset):
    noisy = add_measurement_noise(small_dataset, n_shots=1000, seed=0)
    # The label is untouched
    pd.testing.assert_series_equal(noisy["is_valid"], small_dataset["is_valid"])
    # The amplitudes moved
    assert not np.allclose(
        noisy["c0_real"].to_numpy(), small_dataset["c0_real"].to_numpy()
    )
    # norm_squared recomputed on the noisy amplitudes
    np.testing.assert_allclose(
        noisy["norm_squared"].to_numpy(),
        norm_squared(extract_amplitudes(noisy)),
        rtol=1e-10,
    )
    # Seed reproducibility
    noisy2 = add_measurement_noise(small_dataset, n_shots=1000, seed=0)
    pd.testing.assert_frame_equal(noisy, noisy2)


def test_noise_shrinks_with_budget(small_dataset):
    """Large N -> ||psi_hat||^2 concentrated near ||psi||^2; small N -> spread."""
    base = small_dataset["norm_squared"].to_numpy()
    err = {}
    for n in (100, 10_000):
        noisy = add_measurement_noise(small_dataset, n_shots=n, seed=1)
        err[n] = np.abs(noisy["norm_squared"].to_numpy() - base).mean()
    assert err[10_000] < err[100] / 3  # ~x10 expected, x3 with wide margin


# ---------------------------------------------------------------------------
# Correlated noise (milestone 4)
# ---------------------------------------------------------------------------


def test_correlated_noise_structure(small_dataset):
    """Corr(eps_i, eps_j) ~ rho off-diagonal, Var(eps_i) ~ sigma^2 unchanged."""
    from qsv.features import add_correlated_noise

    rho, n_shots = 0.8, 50
    noisy = add_correlated_noise(small_dataset, n_shots=n_shots, rho=rho, seed=3)
    real_cols = [f"c{i}_real" for i in range(4)]
    eps = noisy[real_cols].to_numpy() - small_dataset[real_cols].to_numpy()
    corr = np.corrcoef(eps.T)
    off_diag = corr[np.triu_indices(4, 1)]
    assert np.abs(off_diag.mean() - rho) < 0.1
    sigma = sigma_from_shots(n_shots)
    assert np.abs(eps.std() - sigma) / sigma < 0.15


def test_correlated_noise_rho_zero_is_iid(small_dataset):
    """rho = 0: off-diagonal correlation ~ 0 (equivalent to iid)."""
    from qsv.features import add_correlated_noise

    noisy = add_correlated_noise(small_dataset, n_shots=100, rho=0.0, seed=3)
    real_cols = [f"c{i}_real" for i in range(4)]
    eps = noisy[real_cols].to_numpy() - small_dataset[real_cols].to_numpy()
    corr = np.corrcoef(eps.T)
    assert np.abs(corr[np.triu_indices(4, 1)]).max() < 0.15


def test_correlated_noise_validation(small_dataset):
    from qsv.features import add_correlated_noise

    with pytest.raises(ValueError):
        add_correlated_noise(small_dataset, rho=1.0)
    with pytest.raises(ValueError):
        add_correlated_noise(small_dataset, rho=-0.1)


# ---------------------------------------------------------------------------
# Calibration drift (milestone 4b)
# ---------------------------------------------------------------------------


def test_calibration_drift_structure(small_dataset):
    """The gain follows A*sin(2*pi*t/T): verifiable on the valid states."""
    from qsv.features import add_calibration_drift

    A, T = 0.1, 100.0
    drifted = add_calibration_drift(
        small_dataset, n_shots=100_000, drift_amplitude=A, drift_period=T, seed=0
    )
    assert "acquisition_time" in drifted.columns
    # Near-zero shot noise (huge N): ||psi_hat||^2 ~ (1+g(t))^2 * ||psi||^2.
    # Verified on VALID states only: for near-null extreme states
    # (||psi||^2 ~ 1e-4) the noise/norm ratio diverges by construction.
    t = drifted["acquisition_time"].to_numpy()
    expected_gain2 = (1 + A * np.sin(2 * np.pi * t / T)) ** 2
    mask = small_dataset["is_valid"].to_numpy() == 1
    ratio = drifted["norm_squared"].to_numpy()[mask] / (
        small_dataset["norm_squared"].to_numpy()[mask] * expected_gain2[mask]
    )
    np.testing.assert_allclose(ratio, 1.0, atol=5e-2)
    # Label untouched
    pd.testing.assert_series_equal(drifted["is_valid"], small_dataset["is_valid"])


def test_calibration_drift_validation(small_dataset):
    from qsv.features import add_calibration_drift

    with pytest.raises(ValueError):
        add_calibration_drift(small_dataset, drift_amplitude=1.5)
    with pytest.raises(ValueError):
        add_calibration_drift(small_dataset, drift_period=0)


def test_calibration_drift_reproducible(small_dataset):
    from qsv.features import add_calibration_drift

    a = add_calibration_drift(small_dataset, n_shots=500, seed=4)
    b = add_calibration_drift(small_dataset, n_shots=500, seed=4)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# Dimension-agnostic behaviour (milestone 4d)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 8, 16])
def test_invariant_features_bounds_any_dimension(dim):
    """The feature pipeline must not be hard-wired to d = 4."""
    states = generate_valid_states(300, dim=dim, strategy="random", seed=13)
    f = scale_invariant_features(states)
    assert (f["entropy_shannon"] >= -1e-9).all()
    assert (f["entropy_shannon"] <= np.log(dim) + 1e-9).all()
    assert (f["purity_normalized"] >= 1 / dim - 1e-9).all()
    assert (f["participation_ratio"] <= dim + 1e-9).all()


@pytest.mark.parametrize("dim", [2, 8])
def test_measurement_noise_any_dimension(dim):
    df = create_dataset(n_valid=100, n_invalid=100, dim=dim, seed=3)
    noisy = add_measurement_noise(df, n_shots=500, seed=1)
    assert noisy.shape == df.shape
    np.testing.assert_allclose(
        noisy["norm_squared"].to_numpy(),
        norm_squared(extract_amplitudes(noisy)),
        rtol=1e-10,
    )
