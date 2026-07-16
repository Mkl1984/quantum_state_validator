"""
Tests for qsv/data_generation.py - including the F2 class-boundary
guarantee (audit 2026-07-07).
"""

import numpy as np
import pytest

from qsv.data_generation import (
    create_dataset,
    generate_invalid_states,
    generate_valid_states,
    verify_normalization,
)

DIM = 4


# ---------------------------------------------------------------------------
# Valid states
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
# Invalid states - F2 guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["scaling", "noise", "direct", "mixed"])
def test_invalid_states_respect_norm_margin(strategy):
    """F2 GUARANTEE: no invalid state inside the band [1 - margin, 1 + margin]."""
    margin = 0.05
    states = generate_invalid_states(
        2000, dim=DIM, strategy=strategy, norm_margin=margin, seed=7
    )
    norms_sq = np.sum(np.abs(states) ** 2, axis=1)
    assert (np.abs(norms_sq - 1.0) >= margin - 1e-12).all(), (
        f"{(np.abs(norms_sq - 1.0) < margin).sum()} state(s) inside the "
        f"forbidden band for strategy '{strategy}'"
    )


def test_invalid_states_custom_margin():
    """The guarantee must follow a non-default margin."""
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
# Full dataset
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
    """Label <-> norm consistency: the core of the problem definition."""
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
# Multiclass dataset (milestone 4)
# ---------------------------------------------------------------------------


def test_extreme_strategy_respects_margin():
    from qsv.data_generation import generate_invalid_states

    states = generate_invalid_states(500, dim=DIM, strategy="extreme", seed=2)
    norms_sq = np.sum(np.abs(states) ** 2, axis=1)
    assert (np.abs(norms_sq - 1.0) >= 0.05 - 1e-12).all()


def test_multiclass_dataset_schema_and_labels():
    from qsv.data_generation import create_multiclass_dataset

    df = create_multiclass_dataset(n_valid=200, n_per_cause=50, dim=DIM, seed=9)
    assert len(df) == 200 + 4 * 50
    counts = df["cause"].value_counts().to_dict()
    assert counts["valid"] == 200
    for cause in ("scaling", "noise", "direct", "extreme"):
        assert counts[cause] == 50
    # cause <-> is_valid <-> norm consistency
    assert ((df["cause"] == "valid") == (df["is_valid"] == 1)).all()
    valid_norms = df.loc[df.is_valid == 1, "norm_squared"]
    invalid_norms = df.loc[df.is_valid == 0, "norm_squared"]
    np.testing.assert_allclose(valid_norms, 1.0, atol=1e-9)
    assert (np.abs(invalid_norms - 1.0) >= 0.05 - 1e-12).all()


def test_multiclass_dataset_reproducible():
    from qsv.data_generation import create_multiclass_dataset

    a = create_multiclass_dataset(50, 20, dim=DIM, seed=7)
    b = create_multiclass_dataset(50, 20, dim=DIM, seed=7)
    assert a.equals(b)


def test_verify_normalization_strict_tolerance():
    """Q6: strictly absolute criterion, no hidden rtol."""
    states = np.array([[1.0 + 0j, 0j, 0j, 0j]])
    ok, norms = verify_normalization(states, tolerance=1e-6)
    assert ok and np.isclose(norms[0], 1.0)
    # Offset of 2e-5: the old np.allclose (rtol 1e-5 + atol 1e-6 -> 1.1e-5
    # of effective slack) accepted beyond the advertised atol.
    # The new strict criterion at atol=1e-6 must REFUSE.
    bad = states * np.sqrt(1.0 + 2e-5)
    ok_bad, _ = verify_normalization(bad, tolerance=1e-6)
    assert not ok_bad
    # And accept when the advertised tolerance covers the offset.
    ok_wide, _ = verify_normalization(bad, tolerance=1e-4)
    assert ok_wide


def test_basis_strategy_vectorized_correctness():
    """Q7: every basis state has exactly one component at 1, the rest at 0."""
    states = generate_valid_states(200, dim=DIM, strategy="basis", seed=11)
    assert np.all(np.sum(np.abs(states) > 0, axis=1) == 1)
    assert np.allclose(np.max(np.abs(states), axis=1), 1.0)
