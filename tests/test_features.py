"""Tests for feature building output columns and preprocessor."""

import numpy as np
import pandas as pd

from scholarly_outcome_prediction.features import (
    build_feature_matrix,
    build_metadata_features,
    build_preprocessor,
    get_feature_column_names,
)


NUMERIC = ["publication_year", "referenced_works_count", "authors_count", "institutions_count"]
CATEGORICAL = ["type", "language", "venue_name", "primary_topic", "open_access_is_oa"]


def test_build_feature_matrix_output_columns(tiny_normalized_df: pd.DataFrame) -> None:
    X, y = build_feature_matrix(
        tiny_normalized_df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        target_name="cited_by_count",
        target_transform="log1p",
    )
    expected = NUMERIC + CATEGORICAL
    assert list(X.columns) == expected
    assert len(y) == len(X)
    assert y.name == "cited_by_count"
    np.testing.assert_allclose(y.iloc[0], np.log1p(5))
    np.testing.assert_allclose(y.iloc[1], np.log1p(20))


def test_build_feature_matrix_no_transform(tiny_normalized_df: pd.DataFrame) -> None:
    X, y = build_feature_matrix(
        tiny_normalized_df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        target_name="cited_by_count",
        target_transform=None,
    )
    assert list(y) == [5, 20, 12, 0, 7]


def test_get_feature_column_names() -> None:
    names = get_feature_column_names(NUMERIC, CATEGORICAL)
    assert names == NUMERIC + CATEGORICAL


def test_build_preprocessor_fit_transform(tiny_normalized_df: pd.DataFrame) -> None:
    X, _ = build_feature_matrix(
        tiny_normalized_df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        target_transform=None,
    )
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    Xt = preprocessor.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
    assert Xt.ndim == 2
    assert not np.any(np.isnan(Xt))


def test_build_metadata_features_does_not_impute() -> None:
    """Feature building must not fill missing values; preprocessing does that after split."""
    df = pd.DataFrame({
        "publication_year": [2018, 2019],
        "referenced_works_count": [10, np.nan],
        "authors_count": [2, 3],
        "institutions_count": [1, 2],
        "type": ["article", "article"],
        "language": ["en", None],
        "venue_name": ["V1", "V2"],
        "primary_topic": ["CS", "Bio"],
        "open_access_is_oa": [True, False],
        "cited_by_count": [5, 10],
    })
    X, y = build_metadata_features(
        df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        target_name="cited_by_count",
        target_transform=None,
    )
    assert np.isnan(X.loc[1, "referenced_works_count"])
    assert pd.isna(X.loc[1, "language"])
