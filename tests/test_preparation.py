"""Tests for src/preparation.py (milestone 4c)."""

import numpy as np
import pytest

from qsv.features import extract_amplitudes
from qsv.preparation import (
    create_preparation_dataset,
    fidelity_to_target,
    make_targets,
)

DIM = 4


def test_targets_are_normalized():
    t = make_targets(6, DIM, seed=1)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-12)


def test_fidelity_bounds_and_scale_invariance():
    t = make_targets(3, DIM, seed=2)
    # fidelity of a target with itself is 1, and is scale-invariant
    np.testing.assert_allclose(fidelity_to_target(t, t), 1.0, atol=1e-12)
    np.testing.assert_allclose(fidelity_to_target(3.7 * t, t), 1.0, atol=1e-12)
    # orthogonal construction: fidelity between two different Haar states < 1
    f_cross = fidelity_to_target(t, np.roll(t, 1, axis=0))
    assert (f_cross < 1.0 - 1e-6).all()


def test_preparation_dataset_class_signatures():
    """Each error class must hide from exactly one statistic."""
    df, targets = create_preparation_dataset(400, dim=DIM, n_shots=100_000, seed=5)
    states = extract_amplitudes(df)
    fid = fidelity_to_target(states, targets[df["target_id"]])

    ok = df["error_type"] == "ok"
    rot = df["error_type"] == "rotated"
    sca = df["error_type"] == "scaled"

    # ok: high fidelity, norm ~ 1
    assert fid[ok].min() > 0.99
    np.testing.assert_allclose(df.loc[ok, "norm_squared"], 1.0, atol=0.02)
    # rotated: normalised (norm-blind) but fidelity inside the requested range
    np.testing.assert_allclose(df.loc[rot, "norm_squared"], 1.0, atol=0.02)
    assert fid[rot].max() < 0.95 and fid[rot].min() > 0.45
    # scaled: direction preserved (fidelity-blind) but norm outside the band
    assert fid[sca].min() > 0.99
    assert (np.abs(df.loc[sca, "norm_squared"] - 1.0) > 0.02).all()


def test_preparation_dataset_labels_and_reproducibility():
    df, _ = create_preparation_dataset(100, dim=DIM, seed=7)
    assert len(df) == 300
    assert ((df["error_type"] == "ok") == (df["prep_ok"] == 1)).all()
    df2, _ = create_preparation_dataset(100, dim=DIM, seed=7)
    assert df.equals(df2)


def test_preparation_dataset_validation():
    with pytest.raises(ValueError):
        create_preparation_dataset(0)
    with pytest.raises(ValueError):
        create_preparation_dataset(10, rotation_fidelity=(0.9, 0.5))
