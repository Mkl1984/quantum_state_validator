"""
Tests du module src/preprocessing.py — split stratifié et absence de
fuite train→val/test dans le scaling.
"""

import numpy as np
import pytest

from src.data_generation import create_dataset
from src.preprocessing import prepare_features_and_target, scale_features, split_data


@pytest.fixture()
def xy():
    df = create_dataset(n_valid=400, n_invalid=400, dim=4, seed=5)
    return prepare_features_and_target(df, include_norm_squared=False)


def test_prepare_features_excludes_leaky_column(xy):
    X, y = xy
    assert "norm_squared" not in X.columns
    assert "state_id" not in X.columns
    assert len(X) == len(y) == 800


def test_split_proportions_and_stratification(xy):
    X, y = xy
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(X, y, random_state=0)
    n = len(X)
    assert len(X_tr) == pytest.approx(0.6 * n, abs=2)
    assert len(X_val) == pytest.approx(0.2 * n, abs=2)
    assert len(X_te) == pytest.approx(0.2 * n, abs=2)
    # Stratification : proportion de valides conservée à ±2 % dans chaque split
    for split in (y_tr, y_val, y_te):
        assert split.mean() == pytest.approx(y.mean(), abs=0.02)
    # Pas de chevauchement d'indices entre les splits
    assert set(X_tr.index).isdisjoint(X_val.index)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert set(X_val.index).isdisjoint(X_te.index)


def test_scaler_fits_on_train_only(xy, tmp_path):
    """Le scaler doit être ajusté sur train UNIQUEMENT (pas de fuite)."""
    X, y = xy
    X_tr, X_val, X_te, *_ = split_data(X, y, random_state=0)
    X_tr_s, X_val_s, X_te_s, scaler = scale_features(
        X_tr,
        X_val,
        X_te,
        save_scaler=True,
        scaler_path=str(tmp_path / "scaler.joblib"),
    )
    # Train standardisé : moyenne ≈ 0, écart-type ≈ 1
    np.testing.assert_allclose(X_tr_s.mean(), 0.0, atol=1e-10)
    np.testing.assert_allclose(X_tr_s.std(ddof=0), 1.0, atol=1e-10)
    # Les paramètres du scaler proviennent du train, pas de val/test
    np.testing.assert_allclose(scaler.mean_, X_tr.mean().to_numpy(), rtol=1e-10)
    # Val/test transformés avec les stats du train → moyenne ≠ 0 en général
    assert (tmp_path / "scaler.joblib").exists()
    # Colonnes et index préservés
    assert list(X_val_s.columns) == list(X_val.columns)
    assert (X_val_s.index == X_val.index).all()
