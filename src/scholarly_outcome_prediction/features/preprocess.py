"""Scikit-learn ColumnTransformer: numeric median impute, categorical constant impute + one-hot."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def _impute_categorical_object(X: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Replace missing values with __missing__ and ensure object dtype for OneHotEncoder."""
    if isinstance(X, pd.DataFrame):
        out = X.fillna("__missing__").astype(str).values
    else:
        out = np.asarray(X, dtype=object)
        out = np.where(pd.isna(out), "__missing__", out)
    return out.astype(object)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    All imputation and encoding happens here (fit on training data only).

    Numeric: median imputation. Categorical: constant imputation (__missing__) then one-hot.
    """
    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = Pipeline([
        ("impute", FunctionTransformer(_impute_categorical_object)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    ct = ColumnTransformer(
        [
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )
    return ct
