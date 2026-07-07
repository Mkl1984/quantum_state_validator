"""
Tests du module src/data_generation.py — dont la garantie F2 de frontière
de classe (audit 2026-07-07).
"""

import numpy as np
import pytest

from src.data_generation import (
    create_dataset,
    generate_invalid_states,
    generate_valid_states,
    verify_normalization,
)

DIM = 4


# ---------------------------------------------------------------------------
# États valides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["random", "dirichlet", "basis"])
def test_valid_states_are_normalized(strategy):
    states = generate_valid_states(500, dim=DIM, strategy=strategy, seed=42)
    all_valid, norms = verify_normalization(states)
    assert all_valid
    np.testing.assert_allclose(norms, 1.0, atol=1e-9)


def test_valid_states_reproducible():
    a = generate_valid_states(50, dim=DIM, strategy="random", seed=123)
    b = generate_valid_states(50, dim=DIM, strategy="random", seed=123)
    np.testing.assert_array_equal(a, b)


def test_valid_states_input_validation():
    with pytest.raises(ValueError):
        generate_valid_states(0, dim=DIM)
    with pytest.raises(ValueError):
        generate_valid_states(10, dim=-1)
    with pytest.raises(ValueError):
        generate_valid_states(10, dim=DIM, strategy="teleportation")


# ---------------------------------------------------------------------------
# États invalides — garantie F2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["scaling", "noise", "direct", "mixed"])
def test_invalid_states_respect_norm_margin(strategy):
    """GARANTIE F2 : aucun état invalide dans la bande [1 − marge, 1 + marge]."""
    margin = 0.05
    states = generate_invalid_states(
        2000, dim=DIM, strategy=strategy, norm_margin=margin, seed=7
    )
    norms_sq = np.sum(np.abs(states) ** 2, axis=1)
    assert (np.abs(norms_sq - 1.0) >= margin - 1e-12).all(), (
        f"{(np.abs(norms_sq - 1.0) < margin).sum()} état(s) dans la bande "
        f"interdite pour la stratégie '{strategy}'"
    )


def test_invalid_states_custom_margin():
    """La garantie doit suivre une marge non standard."""
    margin = 0.2
    states = generate_invalid_states(
        1000, dim=DIM, strategy="noise", noise_level=0.05, norm_margin=margin, seed=3
    )
    norms_sq = np.sum(np.abs(states) ** 2, axis=1)
    assert (np.abs(norms_sq - 1.0) >= margin - 1e-12).all()


def test_invalid_states_margin_validation():
    with pytest.raises(ValueError):
        generate_invalid_states(10, dim=DIM, norm_margin=0.0)
    with pytest.raises(ValueError):
        generate_invalid_states(10, dim=DIM, norm_margin=1.5)


# ---------------------------------------------------------------------------
# Dataset complet
# ---------------------------------------------------------------------------


def test_create_dataset_schema_and_balance():
    df = create_dataset(n_valid=300, n_invalid=200, dim=DIM, seed=11)
    assert len(df) == 500
    assert df["is_valid"].sum() == 300
    expected_cols = (
        ["state_id"]
        + [f"c{i}_{part}" for i in range(DIM) for part in ("real", "imag")]
        + ["norm_squared", "is_valid"]
    )
    assert sorted(df.columns) == sorted(expected_cols)


def test_create_dataset_class_separation():
    """Cohérence label ↔ norme : le cœur de la définition du problème."""
    df = create_dataset(n_valid=300, n_invalid=300, dim=DIM, seed=11)
    valid_norms = df.loc[df["is_valid"] == 1, "norm_squared"]
    invalid_norms = df.loc[df["is_valid"] == 0, "norm_squared"]
    np.testing.assert_allclose(valid_norms, 1.0, atol=1e-9)
    assert (np.abs(invalid_norms - 1.0) >= 0.05 - 1e-12).all()


def test_create_dataset_reproducible():
    a = create_dataset(n_valid=50, n_invalid=50, dim=DIM, seed=99)
    b = create_dataset(n_valid=50, n_invalid=50, dim=DIM, seed=99)
    assert a.equals(b)


# ---------------------------------------------------------------------------
# Dataset multiclasse (jalon 4)
# ---------------------------------------------------------------------------


def test_extreme_strategy_respects_margin():
    from src.data_generation import generate_invalid_states

    states = generate_invalid_states(500, dim=DIM, strategy="extreme", seed=2)
    norms_sq = np.sum(np.abs(states) ** 2, axis=1)
    assert (np.abs(norms_sq - 1.0) >= 0.05 - 1e-12).all()


def test_multiclass_dataset_schema_and_labels():
    from src.data_generation import create_multiclass_dataset

    df = create_multiclass_dataset(n_valid=200, n_per_cause=50, dim=DIM, seed=9)
    assert len(df) == 200 + 4 * 50
    counts = df["cause"].value_counts().to_dict()
    assert counts["valid"] == 200
    for cause in ("scaling", "noise", "direct", "extreme"):
        assert counts[cause] == 50
    # Cohérence cause ↔ is_valid ↔ norme
    assert ((df["cause"] == "valid") == (df["is_valid"] == 1)).all()
    valid_norms = df.loc[df.is_valid == 1, "norm_squared"]
    invalid_norms = df.loc[df.is_valid == 0, "norm_squared"]
    np.testing.assert_allclose(valid_norms, 1.0, atol=1e-9)
    assert (np.abs(invalid_norms - 1.0) >= 0.05 - 1e-12).all()


def test_multiclass_dataset_reproducible():
    from src.data_generation import create_multiclass_dataset

    a = create_multiclass_dataset(50, 20, dim=DIM, seed=7)
    b = create_multiclass_dataset(50, 20, dim=DIM, seed=7)
    assert a.equals(b)
