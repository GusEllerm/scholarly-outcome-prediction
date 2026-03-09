"""Shared fixtures: sample OpenAlex record, tiny DataFrame."""

import pytest
import pandas as pd


@pytest.fixture
def sample_openalex_work() -> dict:
    """One raw OpenAlex work for normalization tests."""
    return {
        "id": "https://openalex.org/W123",
        "display_name": "A test paper",
        "publication_year": 2018,
        "publication_date": "2018-06-15",
        "type": "article",
        "language": "en",
        "cited_by_count": 10,
        "referenced_works": ["https://openalex.org/W1", "https://openalex.org/W2"],
        "authorships": [
            {"author": {}, "institutions": [{"id": "I1"}, {"id": "I2"}]},
            {"author": {}, "institutions": []},
        ],
        "primary_location": {"display_name": "Nature", "venue": {"display_name": "Nature"}},
        "host_venue": None,
        "open_access": {"is_oa": True},
        "topics": [{"display_name": "Computer science"}, {"display_name": "Biology"}],
    }


@pytest.fixture
def normalized_columns() -> list[str]:
    """Expected columns after normalization (includes optional text fields)."""
    from scholarly_outcome_prediction.data.normalize import NORMALIZED_COLUMNS
    return list(NORMALIZED_COLUMNS)


@pytest.fixture
def tiny_normalized_df(normalized_columns: list[str]) -> pd.DataFrame:
    """Minimal normalized table for pipeline smoke tests (5 rows so test set has ≥2 for R²)."""
    return pd.DataFrame(
        [
            {
                "openalex_id": "W1",
                "title": "Paper A",
                "publication_year": 2018,
                "publication_date": "2018-01-01",
                "type": "article",
                "language": "en",
                "cited_by_count": 5,
                "referenced_works_count": 10,
                "authors_count": 2,
                "institutions_count": 1,
                "venue_name": "Venue X",
                "open_access_is_oa": True,
                "primary_topic": "CS",
            },
            {
                "openalex_id": "W2",
                "title": "Paper B",
                "publication_year": 2019,
                "publication_date": "2019-02-01",
                "type": "article",
                "language": "en",
                "cited_by_count": 20,
                "referenced_works_count": 15,
                "authors_count": 3,
                "institutions_count": 2,
                "venue_name": "Venue Y",
                "open_access_is_oa": False,
                "primary_topic": "Bio",
            },
            {
                "openalex_id": "W3",
                "title": "Paper C",
                "publication_year": 2018,
                "publication_date": "2018-06-01",
                "type": "article",
                "language": "en",
                "cited_by_count": 12,
                "referenced_works_count": 8,
                "authors_count": 1,
                "institutions_count": 0,
                "venue_name": "Venue X",
                "open_access_is_oa": True,
                "primary_topic": "CS",
            },
            {
                "openalex_id": "W4",
                "title": "Paper D",
                "publication_year": 2020,
                "publication_date": "2020-01-01",
                "type": "article",
                "language": "en",
                "cited_by_count": 0,
                "referenced_works_count": 20,
                "authors_count": 4,
                "institutions_count": 3,
                "venue_name": "Venue Z",
                "open_access_is_oa": False,
                "primary_topic": "Bio",
            },
            {
                "openalex_id": "W5",
                "title": "Paper E",
                "publication_year": 2019,
                "publication_date": "2019-09-01",
                "type": "article",
                "language": "en",
                "cited_by_count": 7,
                "referenced_works_count": 12,
                "authors_count": 2,
                "institutions_count": 1,
                "venue_name": "Venue Y",
                "open_access_is_oa": True,
                "primary_topic": "CS",
            },
        ],
        columns=normalized_columns,
    )
