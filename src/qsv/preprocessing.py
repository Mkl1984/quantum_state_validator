"""
Module: preprocessing.py
Purpose: data preparation for ML
Author: Mkl Zenin
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
    Split features X and target y.

    IMPORTANT:
    - include_norm_squared=False: the honest setting (but see notebooks
      07-08 - on exact data even the raw coordinates determine the label)
    - include_norm_squared=True: the trivial task
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
    Train/val/test split with stratification.
    """

    stratify_y = y if stratify else None

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=stratify_y, random_state=random_state
    )

    # Adjust the ratio for the validation set
    val_size_adjusted = val_size / (1 - test_size)

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp if stratify else None,
        random_state=random_state,
    )

    logger.info(f"Split done:")
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
    Scale the features: fit on train ONLY, transform on val/test (no
    information leaks from the evaluation sets into the scaler).

    Parameters
    ----------
    method : "standard" | "minmax" | "robust"
        - "standard": z = (x - mu) / sigma. Reading: "x minus the mean,
          divided by the standard deviation". Outlier-sensitive (both mu
          and sigma are).
        - "minmax": z = (x - min) / (max - min), result in [0, 1]. Very
          outlier-sensitive (a single extreme crushes everything else).
        - "robust": z = (x - median) / IQR (interquartile range).
          Recommended for this project: the heavy-tailed features
          (purity_raw up to ~1e8 on extreme states) destroy mu and sigma
          but leave the median and quartiles nearly intact - precisely the
          pathology observed in notebook 08 (the logistic-regression
          collapse).

    Reminder: scaling is required for metric- or gradient-based models
    (logistic regression, SVM, k-NN, neural networks). Trees and forests
    are insensitive to it (invariance under monotone transformations).

    Returns
    -------
    (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Unknown method '{method}'. Choices: {sorted(scalers)}")
    scaler = scalers[method]()

    # FIT on train only
    scaler.fit(X_train)

    # TRANSFORM on all sets
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    logger.info(f"Scaling done ({method})")
    logger.info(
        f"  Train - mean: {X_train_scaled.mean().mean():.6f}, std: {X_train_scaled.std().mean():.6f}"
    )

    if save_scaler:
        scaler_file = Path(scaler_path)
        scaler_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_file)
        logger.info(f"  Scaler saved: {scaler_file}")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


# Self-test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Testing the preprocessing module")

    # Import the generation module
    from qsv.data_generation import load_dataset

    # Load the dataset
    df = load_dataset()

    # Test
    X, y = prepare_features_and_target(df, include_norm_squared=False)
    logger.info(f"\nFeatures: {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    logger.info("\n Preprocessing module functional!")
