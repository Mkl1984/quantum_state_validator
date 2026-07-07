"""
Tests du module src/features.py.

Le test le plus important est ``test_invariant_features_are_scale_invariant`` :
il vérifie mécaniquement la propriété qui empêche le target leakage.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_generation import create_dataset, generate_valid_states
from src.features import (
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
    # La colonne norm_squared du dataset doit correspondre au recalcul
    np.testing.assert_allclose(
        norm_squared(states), small_dataset["norm_squared"].to_numpy(), rtol=1e-10
    )


def test_extract_amplitudes_missing_columns():
    with pytest.raises(ValueError):
        extract_amplitudes(pd.DataFrame({"foo": [1.0]}))


# ---------------------------------------------------------------------------
# Invariance d'échelle — LE test anti-leakage
# ---------------------------------------------------------------------------


def test_invariant_features_are_scale_invariant():
    """c → k·c ne doit changer AUCUNE feature invariante."""
    states = generate_valid_states(100, dim=4, strategy="random", seed=7)
    for k in (0.1, 0.5, 2.0, 37.0):
        f_base = scale_invariant_features(states)
        f_scaled = scale_invariant_features(k * states)
        pd.testing.assert_frame_equal(f_base, f_scaled, atol=1e-9, rtol=1e-9)


def test_sensitive_features_do_leak_the_norm():
    """Contre-épreuve : les features sensibles DOIVENT changer sous c → k·c."""
    states = generate_valid_states(50, dim=4, strategy="random", seed=7)
    f_base = scale_sensitive_features(states)
    f_scaled = scale_sensitive_features(2.0 * states)
    # norm_squared quadruple exactement (k² = 4)
    np.testing.assert_allclose(
        f_scaled["norm_squared"], 4.0 * f_base["norm_squared"], rtol=1e-10
    )
    # purity_raw est multipliée par k⁴ = 16
    np.testing.assert_allclose(
        f_scaled["purity_raw"], 16.0 * f_base["purity_raw"], rtol=1e-10
    )


def test_invariant_features_bounds():
    """H ∈ [0, ln d], pureté ∈ [1/d, 1], participation ∈ [1, d], max_prob ∈ [1/d, 1]."""
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
    """État de base pur : H = 0, pureté = 1, participation = 1, max_prob = 1."""
    states = generate_valid_states(10, dim=4, strategy="basis", seed=1)
    f = scale_invariant_features(states)
    np.testing.assert_allclose(f["entropy_shannon"], 0.0, atol=1e-8)
    np.testing.assert_allclose(f["purity_normalized"], 1.0, atol=1e-10)
    np.testing.assert_allclose(f["max_prob"], 1.0, atol=1e-10)


def test_near_zero_state_falls_back_to_uniform():
    """État quasi nul : renormalisation impossible → distribution uniforme."""
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
    assert "norm_deviation" not in inv.columns  # pas de fuite côté invariant
    with pytest.raises(ValueError):
        compute_features(small_dataset, kind="magic")


# ---------------------------------------------------------------------------
# Bruit de mesure
# ---------------------------------------------------------------------------


def test_sigma_from_shots_scaling():
    """σ suit la loi 1/√N : quadrupler N divise σ par 2."""
    assert sigma_from_shots(400) == pytest.approx(sigma_from_shots(100) / 2)
    with pytest.raises(ValueError):
        sigma_from_shots(0)


def test_add_measurement_noise_properties(small_dataset):
    noisy = add_measurement_noise(small_dataset, n_shots=1000, seed=0)
    # Le label n'est pas touché
    pd.testing.assert_series_equal(noisy["is_valid"], small_dataset["is_valid"])
    # Les amplitudes ont bougé
    assert not np.allclose(
        noisy["c0_real"].to_numpy(), small_dataset["c0_real"].to_numpy()
    )
    # norm_squared recalculée sur amplitudes bruitées
    np.testing.assert_allclose(
        noisy["norm_squared"].to_numpy(),
        norm_squared(extract_amplitudes(noisy)),
        rtol=1e-10,
    )
    # Reproductibilité par graine
    noisy2 = add_measurement_noise(small_dataset, n_shots=1000, seed=0)
    pd.testing.assert_frame_equal(noisy, noisy2)


def test_noise_shrinks_with_budget(small_dataset):
    """Grand N → ‖ψ̂‖² concentrée près de ‖ψ‖² ; petit N → dispersée."""
    base = small_dataset["norm_squared"].to_numpy()
    err = {}
    for n in (100, 10_000):
        noisy = add_measurement_noise(small_dataset, n_shots=n, seed=1)
        err[n] = np.abs(noisy["norm_squared"].to_numpy() - base).mean()
    assert err[10_000] < err[100] / 3  # ~×10 attendu, ×3 avec marge large


# ---------------------------------------------------------------------------
# Bruit corrélé (jalon 4)
# ---------------------------------------------------------------------------


def test_correlated_noise_structure(small_dataset):
    """Corr(εᵢ, εⱼ) ≈ ρ hors diagonale, Var(εᵢ) ≈ σ² inchangée."""
    from src.features import add_correlated_noise

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
    """ρ = 0 : corrélation hors diagonale ≈ 0 (équivalent i.i.d.)."""
    from src.features import add_correlated_noise

    noisy = add_correlated_noise(small_dataset, n_shots=100, rho=0.0, seed=3)
    real_cols = [f"c{i}_real" for i in range(4)]
    eps = noisy[real_cols].to_numpy() - small_dataset[real_cols].to_numpy()
    corr = np.corrcoef(eps.T)
    assert np.abs(corr[np.triu_indices(4, 1)]).max() < 0.15


def test_correlated_noise_validation(small_dataset):
    from src.features import add_correlated_noise

    with pytest.raises(ValueError):
        add_correlated_noise(small_dataset, rho=1.0)
    with pytest.raises(ValueError):
        add_correlated_noise(small_dataset, rho=-0.1)
