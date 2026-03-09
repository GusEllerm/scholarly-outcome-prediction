"""Data acquisition from OpenAlex."""

from scholarly_outcome_prediction.acquisition.fetch import fetch_and_save
from scholarly_outcome_prediction.acquisition.openalex_client import (
    fetch_works_page,
    fetch_works_sample,
)

__all__ = ["fetch_and_save", "fetch_works_page", "fetch_works_sample"]
