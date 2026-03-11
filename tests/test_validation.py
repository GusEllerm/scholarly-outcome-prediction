"""Tests for dataset validation (no live API)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scholarly_outcome_prediction.validation.dataset_validation import (
    validate_processed_dataset,
    validate_raw_records,
    run_validation_and_save,
)


@pytest.fixture
def healthy_df() -> pd.DataFrame:
    """Multi-year, venue populated, enough rows."""
    return pd.DataFrame({
        "publication_year": [2017] * 40 + [2018] * 35 + [2019] * 25,
        "type": ["article"] * 100,
        "cited_by_count": [10] * 100,
        "venue_name": ["Journal X"] * 90 + [None] * 10,
    })


@pytest.fixture
def single_year_df() -> pd.DataFrame:
    """All same year -> validation should fail for min_years_with_data >= 2."""
    return pd.DataFrame({
        "publication_year": [2019] * 150,
        "type": ["article"] * 150,
        "cited_by_count": [5] * 150,
        "venue_name": ["J"] * 150,
    })


@pytest.fixture
def high_venue_missing_df() -> pd.DataFrame:
    """Venue mostly missing -> fail when max_venue_missingness_pct < 96."""
    return pd.DataFrame({
        "publication_year": [2018] * 50 + [2019] * 50,
        "type": ["article"] * 100,
        "cited_by_count": [1] * 100,
        "venue_name": [None] * 96 + ["J"] * 4,
    })


def test_validate_healthy_passes(healthy_df: pd.DataFrame) -> None:
    """Healthy multi-year dataset with venue passes validation."""
    result = validate_processed_dataset(
        healthy_df,
        min_row_count=50,
        min_years_with_data=2,
        max_venue_missingness_pct=95.0,
    )
    assert result["passed"] is True
    assert result["row_count"] == 100
    assert result["publication_year"]["n_unique"] >= 2
    assert result["venue_name_non_null_pct"] > 80


def test_validate_single_year_fails(single_year_df: pd.DataFrame) -> None:
    """Single-year collapse is detected and validation fails."""
    result = validate_processed_dataset(
        single_year_df,
        min_row_count=50,
        min_years_with_data=2,
        max_venue_missingness_pct=95.0,
    )
    assert result["passed"] is False
    assert any("distinct value" in str(e) for e in result["errors"])


def test_validate_venue_missingness_fails(high_venue_missing_df: pd.DataFrame) -> None:
    """Excessive venue missingness fails validation."""
    result = validate_processed_dataset(
        high_venue_missing_df,
        min_row_count=50,
        min_years_with_data=2,
        max_venue_missingness_pct=95.0,
    )
    assert result["passed"] is False
    assert any("venue_name" in str(e) for e in result["errors"])


def test_validate_raw_records() -> None:
    """Raw validation returns type/year counts and venue-like rate."""
    records = [
        {"type": "article", "publication_year": 2018, "primary_location": {"source": {"display_name": "J1"}}},
        {"type": "article", "publication_year": 2019, "primary_location": {}},
    ]
    result = validate_raw_records(records)
    assert result["row_count"] == 2
    assert result["type_counts"].get("article") == 2
    assert result["venue_like_available_count"] == 1
    assert result["venue_like_available_pct"] == 50.0


def test_validate_representative_elite_corpus_fails() -> None:
    """Representative mode: corpus with very high median citations fails (elite-only check)."""
    # Median 600 > default threshold 500
    df = pd.DataFrame({
        "publication_year": [2018] * 50 + [2019] * 50,
        "type": ["article"] * 100,
        "cited_by_count": [600.0] * 100,
        "venue_name": ["J" + str(i) for i in range(50)] * 2,
    })
    result = validate_processed_dataset(
        df,
        min_row_count=50,
        min_years_with_data=2,
        max_venue_missingness_pct=95.0,
        dataset_mode="representative",
        max_median_citations_representative=500.0,
        min_distinct_venues_representative=10,
    )
    assert result["passed"] is False
    assert any("median" in str(e).lower() or "representative" in str(e).lower() for e in result["errors"])


def test_run_validation_and_save_run_scoped(healthy_df: pd.DataFrame) -> None:
    """run_validation_and_save with run_id writes run-scoped report with run_id."""
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        result, json_path, md_path = run_validation_and_save(
            raw_records=None,
            df=healthy_df,
            processed_path=Path("/fake/test_dataset.parquet"),
            out_dir=out_dir,
            run_id="2025-01-15T12:00:00Z",
            min_row_count=10,
            min_years_with_data=2,
            max_venue_missingness_pct=95.0,
        )
        assert result["passed"] is True
        assert json_path.exists()
        assert json_path.name == "test_dataset_dataset_validation.json"
        assert result.get("report_scope") == "run"
        assert result.get("run_id") == "2025-01-15T12:00:00Z"
        assert "processed" in result
        assert "run_id" in result


def test_run_validation_and_save_dataset_scoped(healthy_df: pd.DataFrame) -> None:
    """run_validation_and_save without run_id writes dataset-scoped report with report_id, no run_id."""
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        result, json_path, md_path = run_validation_and_save(
            raw_records=None,
            df=healthy_df,
            processed_path=Path("/fake/my_dataset.parquet"),
            out_dir=out_dir,
            run_id=None,
            min_row_count=10,
            min_years_with_data=2,
            max_venue_missingness_pct=95.0,
        )
        assert result["passed"] is True
        assert json_path.name == "my_dataset_dataset_validation.json"
        assert result.get("report_scope") == "dataset"
        assert "run_id" not in result
        assert result.get("report_id") == "my_dataset_dataset_validation"
        assert result.get("source_dataset_id") == "my_dataset"


def test_run_validation_and_save_includes_provenance_when_provided(healthy_df: pd.DataFrame) -> None:
    """run_validation_and_save records provenance (source_config_path, generation_params, work_id_fingerprint)."""
    df = healthy_df.copy()
    df["openalex_id"] = [f"W{i}" for i in range(len(df))]
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        result, _, _ = run_validation_and_save(
            raw_records=None,
            df=df,
            processed_path=Path("/fake/foo.parquet"),
            out_dir=out_dir,
            run_id=None,
            min_row_count=10,
            min_years_with_data=2,
            max_venue_missingness_pct=95.0,
            source_config_path="configs/data/representative.yaml",
            generation_params={"stratify_by_year": True, "use_random_sample": True},
        )
        assert "provenance" in result
        prov = result["provenance"]
        assert prov.get("source_config_path") == "configs/data/representative.yaml"
        assert prov.get("generation_params", {}).get("use_random_sample") is True
        assert "work_id_fingerprint" in prov
        assert prov["work_id_fingerprint"] is not None
        assert "selection_strategy_summary" in prov


def test_provenance_representative_vs_temporal_wording(healthy_df: pd.DataFrame) -> None:
    """Provenance selection_strategy_summary distinguishes representative vs temporal semantics."""
    df = healthy_df.copy()
    df["openalex_id"] = [f"W{i}" for i in range(len(df))]
    params = {"stratify_by_year": True, "use_random_sample": True}
    with tempfile.TemporaryDirectory() as d:
        out_dir = Path(d)
        # Representative: should say broad sample, representative-oriented
        result_rep, _, _ = run_validation_and_save(
            raw_records=None,
            df=df,
            processed_path=Path("/fake/rep.parquet"),
            out_dir=out_dir,
            dataset_mode="representative",
            generation_params=params,
        )
        summary_rep = result_rep["provenance"].get("selection_strategy_summary", "")
        assert "Representative dataset" in summary_rep or "representative" in summary_rep.lower()
        assert "broad" in summary_rep.lower() or "representative" in summary_rep.lower()
        # Temporal: should say time-ordered evaluation, not merely representative sample
        result_temp, _, _ = run_validation_and_save(
            raw_records=None,
            df=df,
            processed_path=Path("/fake/temp.parquet"),
            out_dir=out_dir,
            dataset_mode="temporal",
            generation_params=params,
        )
        summary_temp = result_temp["provenance"].get("selection_strategy_summary", "")
        assert "Temporal dataset" in summary_temp or "temporal" in summary_temp.lower()
        assert "time-ordered" in summary_temp
        assert "Do not interpret this dataset as a representative sample" in summary_temp


def test_validate_article_only_with_expected_work_types_no_warning() -> None:
    """When expected_work_types=['article'] and corpus is article-only, do not add 'single type only' warning."""
    df = pd.DataFrame({
        "publication_year": [2018] * 50 + [2019] * 50,
        "type": ["article"] * 100,
        "cited_by_count": [5.0] * 100,
        "venue_name": ["J"] * 100,
    })
    result = validate_processed_dataset(
        df,
        min_row_count=50,
        min_years_with_data=2,
        max_venue_missingness_pct=95.0,
        expected_work_types=["article"],
    )
    assert result["passed"] is True
    warnings_text = " ".join(result.get("warnings", []))
    assert "Single type only" not in warnings_text
    messages = result.get("messages", [])
    expected_msgs = [m for m in messages if m.get("severity") == "expected"]
    assert any("article" in m.get("text", "") and "work_types" in m.get("text", "") for m in expected_msgs)
