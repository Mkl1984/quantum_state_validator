"""
Module: preprocessing.py
Objectif: Préparation des données pour ML
Auteur: Mkl Zenin
Date: 2024-11-17
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


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

    logger.info(f"Split effectué:")
    logger.info(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
    save_scaler: bool = True,
    scaler_path: str = "models/scaler.joblib",
):
    """
    Normalise les features : fit sur train UNIQUEMENT, transform sur val/test
    (aucune fuite d'information des ensembles d'évaluation vers le scaler).

    Paramètres
    ----------
    method : "standard" | "minmax" | "robust"
        - "standard" : z = (x − μ)/σ. Lecture : « x moins la moyenne, divisé
          par l'écart-type ». Sensible aux outliers (μ et σ le sont).
        - "minmax"   : z = (x − min)/(max − min), résultat dans [0, 1].
          Très sensible aux outliers (un seul extrême écrase tout le reste).
        - "robust"   : z = (x − médiane)/IQR (écart interquartile).
          Recommandé pour ce projet : les features à queues lourdes
          (purity_raw ~ 10⁸ sur les états extrêmes) détruisent μ et σ,
          mais laissent médiane et quartiles quasi intacts — c'est
          précisément la pathologie observée au notebook 08 (effondrement
          de la régression logistique).

    Rappel : le scaling est requis pour les modèles à géométrie métrique ou
    à gradient (régression logistique, SVM, k-NN, réseaux). Les arbres et
    forêts y sont insensibles (invariance par transformation monotone).

    Retourne
    --------
    (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"method '{method}' inconnue. Choix : {sorted(scalers)}")
    scaler = scalers[method]()

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

    logger.info(f"Normalisation effectuée ({method})")
    logger.info(
        f"  Train - μ: {X_train_scaled.mean().mean():.6f}, σ: {X_train_scaled.std().mean():.6f}"
    )

    if save_scaler:
        scaler_file = Path(scaler_path)
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_file)
        logger.info(f"  Scaler sauvegardé: {scaler_file}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# Test si exécuté directement
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Test du module preprocessing.py")

    # Import du module de génération
    from data_generation import load_dataset

    # Charge le dataset
    df = load_dataset()

    # Test
    X, y = prepare_features_and_target(df, include_norm_squared=False)
    logger.info(f"\nFeatures: {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    logger.info("\n Module preprocessing fonctionnel!")
