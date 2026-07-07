"""
Type stubs for preprocessing module
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

class QuantumPreprocessor:
    """
    Preprocessor for quantum state data with feature engineering.
    """

    scaler: StandardScaler
    feature_names_: list[str]

    def __init__(self) -> None: ...
    def compute_entropy(self, probas: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute von Neumann entropy.

        Args:
            probas: Probability distributions (n_samples, n_dims)

        Returns:
            Entropy values (n_samples,)
        """
        ...

    def compute_purity(self, probas: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute quantum purity.

        Args:
            probas: Probability distributions (n_samples, n_dims)

        Returns:
            Purity values (n_samples,)
        """
        ...

    def compute_norm_deviation(
        self, probas: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute deviation from normalization.

        Args:
            probas: Probability distributions (n_samples, n_dims)

        Returns:
            Norm deviation values (n_samples,)
        """
        ...

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "QuantumPreprocessor":
        """
        Fit the preprocessor on training data.

        Args:
            X: Input features
            y: Target (optional, not used)

        Returns:
            self
        """
        ...

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        """
        Transform data with feature engineering and scaling.

        Args:
            X: Input features

        Returns:
            Transformed and scaled features
        """
        ...

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> NDArray[np.float64]:
        """
        Fit and transform in one step.

        Args:
            X: Input features
            y: Target (optional, not used)

        Returns:
            Transformed and scaled features
        """
        ...

def load_preprocessor(filepath: str) -> QuantumPreprocessor:
    """
    Load a saved preprocessor.

    Args:
        filepath: Path to saved .pkl file

    Returns:
        Loaded preprocessor
    """
    ...

def save_preprocessor(preprocessor: QuantumPreprocessor, filepath: str) -> None:
    """
    Save a preprocessor to disk.

    Args:
        preprocessor: Preprocessor instance
        filepath: Path to save .pkl file
    """
    ...
