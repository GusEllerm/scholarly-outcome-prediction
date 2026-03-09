"""Build feature matrix and target from normalized DataFrame.

Feature-building functions only:
- select feature columns and validate they exist
- extract the target column
- apply target transform if configured

They do NOT fill missing values, fit imputers, or encode categoricals.
All imputation and encoding happen in the sklearn preprocessing pipeline fit on the training split only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_metadata_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_name: str = "cited_by_count",
    target_transform: str | None = "log1p",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (metadata features only) and y (target) from normalized dataframe.

    - Only selects and validates columns; does not impute or encode.
    - Applies target_transform to y: "log1p" or None.
    - Rows with missing target are kept; caller may drop them before split.
    """
    feature_cols = numeric_features + categorical_features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in dataframe: {missing}")
    if target_name not in df.columns:
        raise ValueError(f"Target column not found: {target_name}")

    X = df[feature_cols].copy()
    y = df[target_name].copy()
    y = y.astype(float)
    if target_transform == "log1p":
        # Only transform non-NaN so we can detect missing target if needed
        y = y.apply(lambda v: np.log1p(v) if pd.notna(v) else np.nan)
    return X, y


def build_feature_matrix(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    target_name: str = "cited_by_count",
    target_transform: str | None = "log1p",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X and y from normalized dataframe (metadata features only).

    Alias for build_metadata_features for backward compatibility.
    Does not perform any imputation or encoding.
    """
    return build_metadata_features(
        df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_name=target_name,
        target_transform=target_transform,
    )


def get_feature_column_names(
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Return list of feature names in order: numeric first, then categorical."""
    return numeric_features + categorical_features


# --- Future modality extensions (not implemented yet) ---
# def build_text_features(df: pd.DataFrame, ...) -> pd.DataFrame: ...
# def build_hybrid_features(metadata_df: pd.DataFrame, text_df: pd.DataFrame, ...) -> pd.DataFrame: ...
