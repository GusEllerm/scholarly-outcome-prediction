"""Tests for train/test split (random and time-based)."""

import pandas as pd
import pytest

from scholarly_outcome_prediction.data.split import train_test_split_df


@pytest.fixture
def df_with_year() -> pd.DataFrame:
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "publication_year": [2018, 2018, 2019, 2019, 2019, 2020, 2020, 2020, 2020, 2020],
    })


def test_time_split_deterministic(df_with_year: pd.DataFrame) -> None:
    """Time split is deterministic: same data and test_size yield same train/test."""
    train1, test1 = train_test_split_df(
        df_with_year, test_size=0.2, split_kind="time", time_column="publication_year"
    )
    train2, test2 = train_test_split_df(
        df_with_year, test_size=0.2, split_kind="time", time_column="publication_year"
    )
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(test1, test2)


def test_time_split_last_n_as_test(df_with_year: pd.DataFrame) -> None:
    """Time split: last floor(n * test_size) rows are test; rest train."""
    train, test = train_test_split_df(
        df_with_year, test_size=0.2, split_kind="time", time_column="publication_year"
    )
    # 10 * 0.2 = 2 test rows
    assert len(test) == 2
    assert len(train) == 8
    assert test["publication_year"].min() >= train["publication_year"].max()


def test_time_split_requires_time_column() -> None:
    """split_kind=time without time_column raises."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="time_column"):
        train_test_split_df(df, split_kind="time", time_column=None)


def test_time_split_column_missing_raises() -> None:
    """time_column not in DataFrame raises."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="not in DataFrame"):
        train_test_split_df(df, split_kind="time", time_column="publication_year")


def test_time_split_all_null_raises() -> None:
    """time_column all null raises."""
    df = pd.DataFrame({"x": [1, 2], "publication_year": [None, None]})
    with pytest.raises(ValueError, match="all null"):
        train_test_split_df(df, split_kind="time", time_column="publication_year")


def test_random_split_unchanged() -> None:
    """Random split still works and uses random_state."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    train, test = train_test_split_df(df, test_size=0.4, random_state=42, split_kind="random")
    assert len(test) == 2
    assert len(train) == 3


def test_time_split_by_year_boundaries() -> None:
    """Time split with train_year_end and test_year_start uses explicit boundaries."""
    df = pd.DataFrame({
        "x": range(10),
        "publication_year": [2017, 2017, 2018, 2018, 2018, 2019, 2019, 2020, 2020, 2020],
    })
    train, test = train_test_split_df(
        df,
        split_kind="time",
        time_column="publication_year",
        train_year_end=2018,
        test_year_start=2019,
    )
    assert train["publication_year"].max() <= 2018
    assert test["publication_year"].min() >= 2019
    assert len(train) == 5 and len(test) == 5


def test_time_split_single_year_raises() -> None:
    """Time split fails when only one distinct year (no temporal variation)."""
    df = pd.DataFrame({"x": [1, 2, 3], "publication_year": [2019, 2019, 2019]})
    with pytest.raises(ValueError, match="only one distinct value"):
        train_test_split_df(df, split_kind="time", time_column="publication_year")
