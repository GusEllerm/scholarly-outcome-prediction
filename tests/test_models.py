"""Smoke tests for baseline and XGBoost training."""

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from scholarly_outcome_prediction.features import build_preprocessor
from scholarly_outcome_prediction.models import get_model_builder, BaselineRegressor


def _xgboost_available() -> bool:
    try:
        from xgboost import XGBRegressor  # noqa: F401

        return True
    except Exception:
        return False


NUMERIC = ["publication_year", "referenced_works_count", "authors_count", "institutions_count"]
CATEGORICAL = ["type", "language", "venue_name", "primary_topic", "open_access_is_oa"]


@pytest.fixture
def small_X_y() -> tuple[np.ndarray, np.ndarray]:
    """Small numeric-only matrix for quick model fit (no categorical for simplicity)."""
    np.random.seed(42)
    X = np.random.randn(20, 4).astype(np.float32)
    y = np.random.randn(20)
    return X, y


def test_baseline_regressor_fit_predict(small_X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_X_y
    model = BaselineRegressor()
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(y),)
    np.testing.assert_allclose(pred, np.mean(y))


@pytest.mark.skipif(
    not _xgboost_available(), reason="XGBoost not loadable (e.g. missing libomp on macOS)"
)
def test_xgboost_regressor_fit_predict(small_X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_X_y
    model = get_model_builder("xgboost")(params={"n_estimators": 5, "max_depth": 2})
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(y),)


def test_registry_baseline() -> None:
    builder = get_model_builder("baseline")
    model = builder(params={})
    assert isinstance(model, BaselineRegressor)


@pytest.mark.skipif(
    not _xgboost_available(), reason="XGBoost not loadable (e.g. missing libomp on macOS)"
)
def test_registry_xgboost() -> None:
    builder = get_model_builder("xgboost")
    model = builder(params={"n_estimators": 2})
    assert model is not None
    assert hasattr(model, "fit") and hasattr(model, "predict")


def test_registry_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        get_model_builder("unknown_model")


def test_registry_median_baseline(small_X_y: tuple[np.ndarray, np.ndarray]) -> None:
    from scholarly_outcome_prediction.models.median_baseline import MedianBaselineRegressor

    X, y = small_X_y
    model = get_model_builder("median_baseline")(params={})
    assert isinstance(model, MedianBaselineRegressor)
    model.fit(X, y)
    pred = model.predict(X)
    np.testing.assert_allclose(pred, np.median(y))


def test_registry_ridge(small_X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_X_y
    model = get_model_builder("ridge")(params={"alpha": 0.1})
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_year_conditioned_baseline_behavior() -> None:
    """Year-conditioned baseline: per-year median and fallback for unseen year."""
    from scholarly_outcome_prediction.models.year_conditioned_baseline import YearConditionedBaselineRegressor

    # X: first column = year (2018, 2018, 2019, 2019), then dummy cols
    X = np.array([[2018, 1, 0], [2018, 2, 0], [2019, 1, 0], [2019, 2, 0]], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0, 40.0])  # 2018 median=15, 2019 median=35
    model = YearConditionedBaselineRegressor(use_mean=False, year_column_index=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (4,)
    assert pred[0] == pred[1] == 15.0
    assert pred[2] == pred[3] == 35.0
    # Unseen year gets global median
    X_unseen = np.array([[2020, 1, 0], [2020, 2, 0]])
    pred_unseen = model.predict(X_unseen)
    global_median = 25.0  # median(10,20,30,40)
    assert pred_unseen[0] == pred_unseen[1] == global_median


def test_registry_year_conditioned(small_X_y: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = small_X_y
    # First column as "year" for consistency
    X[:, 0] = np.array([2018] * 10 + [2019] * 10)
    model = get_model_builder("year_conditioned")(params={"year_column_index": 0})
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape


def test_registry_hurdle(small_X_y: tuple[np.ndarray, np.ndarray]) -> None:
    """Hurdle baseline: zero vs nonzero then regress on positive."""
    y = np.array([0.0, 0.0, 1.0, 2.0, 3.0] * 4)
    X = np.random.randn(20, 4).astype(np.float32)
    model = get_model_builder("hurdle")(params={})
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert (pred >= 0).all()


def test_pipeline_with_preprocessor_and_baseline() -> None:
    """Full pipeline: preprocessor + baseline on 2 samples."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "publication_year": [2018, 2019],
            "referenced_works_count": [10, 20],
            "authors_count": [2, 3],
            "institutions_count": [1, 2],
            "type": ["article", "article"],
            "language": ["en", "en"],
            "venue_name": ["V1", "V2"],
            "primary_topic": ["CS", "Bio"],
            "open_access_is_oa": [True, False],
        }
    )
    y = np.array([1.0, 2.0])
    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    model = BaselineRegressor()
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(df, y)
    pred = pipe.predict(df)
    assert pred.shape == (2,)
    np.testing.assert_allclose(pred, np.mean(y))
