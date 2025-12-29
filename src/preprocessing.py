"""
Module: preprocessing.py
Objectif: Préparation des données pour ML
Auteur: Mandem
Date: 2024-11-17
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import joblib
from pathlib import Path


def prepare_features_and_target(
    df: pd.DataFrame,
    target_col: str = "is_valid",
    exclude_cols: Optional[List[str]] = None,
    include_norm_squared: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare features X et target y.

    IMPORTANT:
    - Si include_norm_squared=False : vrai challenge
    - Si include_norm_squared=True : tâche facile
    """

    if exclude_cols is None:
        exclude_cols = ["state_id"]

    cols_to_exclude = set(exclude_cols + [target_col])

    if not include_norm_squared:
        cols_to_exclude.add("norm_squared")

    feature_cols = [col for col in df.columns if col not in cols_to_exclude]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split en train/val/test avec stratification.
    """

    stratify_y = y if stratify else None

    # Premier split : train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_y, random_state=random_state
    )

    # Ajustement du ratio pour val
    val_size_adjusted = val_size / (1 - test_size)

    # Second split : train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp if stratify else None,
        random_state=random_state,
    )

    print(f"Split effectué:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
    save_scaler: bool = True,
    scaler_path: str = "models/scaler.joblib",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalise les features (fit sur train, transform sur val/test).
    """

    scaler = StandardScaler()

    # FIT sur train uniquement
    scaler.fit(X_train)

    # TRANSFORM sur tous
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    print(f"Normalisation effectuée ({method})")
    print(
        f"  Train - μ: {X_train_scaled.mean().mean():.6f}, σ: {X_train_scaled.std().mean():.6f}"
    )

    if save_scaler:
        scaler_file = Path(scaler_path)
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_file)
        print(f"  Scaler sauvegardé: {scaler_file}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# Test si exécuté directement
if __name__ == "__main__":
    print("Test du module preprocessing.py")

    # Import du module de génération
    from data_generation import load_dataset

    # Charge le dataset
    df = load_dataset()

    # Test
    X, y = prepare_features_and_target(df, include_norm_squared=False)
    print(f"\nFeatures: {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    print("\n Module preprocessing fonctionnel!")
