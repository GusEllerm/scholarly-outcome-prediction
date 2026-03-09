"""Minimal end-to-end smoke test with fixture data (no live API)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scholarly_outcome_prediction.features import build_feature_matrix, build_preprocessor
from scholarly_outcome_prediction.models import get_model_builder
from scholarly_outcome_prediction.evaluation import (
    compute_metrics,
    save_metrics,
    save_model_pipeline,
    load_model_pipeline,
)
from scholarly_outcome_prediction.utils.io import write_parquet, read_parquet
from scholarly_outcome_prediction.utils.seeds import set_global_seed
from sklearn.pipeline import Pipeline


NUMERIC = ["publication_year", "referenced_works_count", "authors_count", "institutions_count"]
CATEGORICAL = ["type", "language", "venue_name", "primary_topic", "open_access_is_oa"]


@pytest.fixture
def tiny_processed_path(tiny_normalized_df: pd.DataFrame) -> Path:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tiny.parquet"
        write_parquet(tiny_normalized_df, path)
        yield path


def test_smoke_train_evaluate_save_load(
    tiny_normalized_df: pd.DataFrame,
    tiny_processed_path: Path,
) -> None:
    """Train baseline on tiny data, evaluate, save/load pipeline, recompute metrics."""
    set_global_seed(42)
    df = read_parquet(tiny_processed_path)
    X, y = build_feature_matrix(
        df,
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        target_name="cited_by_count",
        target_transform="log1p",
    )
    full = pd.concat([X, y], axis=1)
    # Use 3 train / 2 test so R² is well-defined (sklearn warns for n_samples < 2)
    train_df = full.iloc[:3]
    test_df = full.iloc[3:]
    X_train = train_df[NUMERIC + CATEGORICAL]
    y_train = train_df["cited_by_count"]
    X_test = test_df[NUMERIC + CATEGORICAL]
    y_test = test_df["cited_by_count"].values

    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    model = get_model_builder("baseline")(params={})
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, metric_names=["rmse", "mae", "r2"])
    assert "rmse" in metrics and "mae" in metrics and "r2" in metrics

    with tempfile.TemporaryDirectory() as d:
        model_path = Path(d) / "model.joblib"
        save_model_pipeline(pipe, model_path)
        loaded = load_model_pipeline(model_path)
        pred2 = loaded.predict(X_test)
        assert pred2.shape == y_pred.shape

        save_metrics(metrics, Path(d) / "metrics.json")
        assert (Path(d) / "metrics.json").exists()


def test_smoke_time_split_train_evaluate(
    tiny_normalized_df: pd.DataFrame,
) -> None:
    """Smoke test: time-based split then train/evaluate (no live API)."""
    from scholarly_outcome_prediction.data import train_test_split_df

    set_global_seed(42)
    full = tiny_normalized_df.dropna(subset=["cited_by_count"])
    train_df, test_df = train_test_split_df(
        full,
        test_size=0.2,
        split_kind="time",
        time_column="publication_year",
    )
    assert len(train_df) >= 1 and len(test_df) >= 1
    assert test_df["publication_year"].min() >= train_df["publication_year"].max()

    X_train = train_df[NUMERIC + CATEGORICAL]
    y_train = train_df["cited_by_count"]
    X_test = test_df[NUMERIC + CATEGORICAL]
    y_test = test_df["cited_by_count"].values

    preprocessor = build_preprocessor(NUMERIC, CATEGORICAL)
    model = get_model_builder("baseline")(params={})
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, metric_names=["rmse", "mae", "r2"])
    assert "rmse" in metrics
