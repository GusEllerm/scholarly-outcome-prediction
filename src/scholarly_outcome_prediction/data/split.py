"""Train/test split with configurable strategy (random or time-based)."""

from __future__ import annotations

import math

import pandas as pd
from sklearn.model_selection import train_test_split

from scholarly_outcome_prediction.utils.seeds import set_global_seed


def train_test_split_df(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_kind: str = "random",
    time_column: str | None = None,
    train_year_end: int | None = None,
    test_year_start: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train and test.

    - split_kind="random": shuffle and split (reproducible with random_state).
    - split_kind="time": by time_column. If train_year_end and test_year_start are set,
      train = rows with time_column <= train_year_end, test = rows with time_column >= test_year_start.
      Otherwise take last test_size fraction as test. Fails if no temporal variation.
    """
    set_global_seed(random_state)
    if split_kind == "time":
        if not time_column:
            raise ValueError("split_kind=time requires time_column (e.g. publication_year)")
        if time_column not in df.columns:
            raise ValueError(f"time_column '{time_column}' not in DataFrame columns: {list(df.columns)}")
        if df[time_column].isna().all():
            raise ValueError(f"time_column '{time_column}' is all null; cannot perform time-based split")
        out = df.dropna(subset=[time_column]).copy()
        if out[time_column].nunique() < 2:
            raise ValueError(
                f"time_column '{time_column}' has only one distinct value; need temporal variation for time split"
            )
        # Explicit year boundaries
        if train_year_end is not None and test_year_start is not None:
            train = out[out[time_column] <= train_year_end]
            test = out[out[time_column] >= test_year_start]
            if len(train) == 0:
                raise ValueError(f"No rows with {time_column} <= {train_year_end}; cannot form train set")
            if len(test) == 0:
                raise ValueError(f"No rows with {time_column} >= {test_year_start}; cannot form test set")
            return train, test
        # Fraction-based: last test_size fraction as test
        out = out.sort_values(time_column)
        n = len(out)
        n_test = max(1, math.floor(n * test_size))
        n_train = n - n_test
        train = out.iloc[:n_train]
        test = out.iloc[n_train:]
        return train, test
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test
