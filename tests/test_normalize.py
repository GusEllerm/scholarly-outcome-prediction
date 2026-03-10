"""Tests for normalization of OpenAlex works."""

from scholarly_outcome_prediction.data import (
    normalize_work,
    normalize_works_to_dataframe,
    NORMALIZED_COLUMNS,
)


def test_normalize_one_work(sample_openalex_work: dict) -> None:
    out = normalize_work(sample_openalex_work)
    assert list(out.keys()) == NORMALIZED_COLUMNS
    assert out["openalex_id"] == "https://openalex.org/W123"
    assert out["title"] == "A test paper"
    assert out["publication_year"] == 2018
    assert out["cited_by_count"] == 10
    assert out["referenced_works_count"] == 2
    assert out["authors_count"] == 2
    assert out["institutions_count"] == 2
    assert out["venue_name"] == "Nature"
    assert out["open_access_is_oa"] is True
    assert out["primary_topic"] == "Computer science"


def test_normalize_works_to_dataframe(sample_openalex_work: dict) -> None:
    df = normalize_works_to_dataframe([sample_openalex_work])
    assert list(df.columns) == NORMALIZED_COLUMNS
    assert len(df) == 1
    assert df["cited_by_count"].iloc[0] == 10


def test_normalize_empty_list() -> None:
    df = normalize_works_to_dataframe([])
    assert list(df.columns) == NORMALIZED_COLUMNS
    assert len(df) == 0


def test_normalize_missing_fields() -> None:
    minimal = {"id": "https://openalex.org/W0"}
    out = normalize_work(minimal)
    assert out["openalex_id"] == "https://openalex.org/W0"
    assert out["title"] is None
    assert out["cited_by_count"] is None
    assert out["venue_name"] is None


def test_normalize_missing_topics() -> None:
    """Missing or empty topics -> primary_topic None."""
    out = normalize_work({"id": "https://openalex.org/W1", "topics": None})
    assert out["primary_topic"] is None
    out2 = normalize_work({"id": "https://openalex.org/W2", "topics": []})
    assert out2["primary_topic"] is None


def test_normalize_missing_primary_location() -> None:
    """No primary_location; venue from host_venue or None."""
    out = normalize_work({"id": "https://openalex.org/W1"})
    assert out["venue_name"] is None
    out2 = normalize_work({
        "id": "https://openalex.org/W2",
        "host_venue": {"display_name": "Journal X"},
    })
    assert out2["venue_name"] == "Journal X"


def test_normalize_empty_authorships() -> None:
    """Empty authorships -> authors_count 0, institutions_count 0."""
    out = normalize_work({"id": "https://openalex.org/W1", "authorships": []})
    assert out["authors_count"] == 0
    assert out["institutions_count"] == 0


def test_normalize_missing_open_access() -> None:
    """Missing open_access -> open_access_is_oa None."""
    out = normalize_work({"id": "https://openalex.org/W1"})
    assert out["open_access_is_oa"] is None


def test_normalize_institution_counting() -> None:
    """Institutions counted across authorships; missing institutions list -> 0 for that row."""
    out = normalize_work({
        "id": "https://openalex.org/W1",
        "authorships": [
            {"institutions": [{"id": "I1"}]},
            {"institutions": None},
            {},
        ],
    })
    assert out["authors_count"] == 3
    assert out["institutions_count"] == 1


def test_normalize_venue_from_primary_location() -> None:
    """Venue from primary_location.display_name or .venue.display_name."""
    out = normalize_work({
        "id": "https://openalex.org/W1",
        "primary_location": {"display_name": "Nature"},
    })
    assert out["venue_name"] == "Nature"
    out2 = normalize_work({
        "id": "https://openalex.org/W2",
        "primary_location": {"venue": {"display_name": "Science"}},
    })
    assert out2["venue_name"] == "Science"


def test_normalize_optional_text_fields_absent() -> None:
    """Optional text fields present in schema but None when not in source."""
    out = normalize_work({"id": "https://openalex.org/W1"})
    assert out["abstract_text"] is None
    assert out["fulltext_text"] is None
    assert out["has_abstract"] is None
    assert out["has_fulltext"] is None
    assert out["fulltext_origin"] is None


def test_normalize_counts_by_year_preserved() -> None:
    """counts_by_year from OpenAlex is normalized to counts_by_year_json (JSON string)."""
    import json
    raw = {
        "id": "https://openalex.org/W1",
        "publication_year": 2019,
        "counts_by_year": [
            {"year": 2019, "cited_by_count": 1},
            {"year": 2020, "cited_by_count": 3},
        ],
    }
    out = normalize_work(raw)
    assert out["counts_by_year_json"] is not None
    parsed = json.loads(out["counts_by_year_json"])
    assert parsed == [{"year": 2019, "cited_by_count": 1}, {"year": 2020, "cited_by_count": 3}]
    # Missing counts_by_year -> None
    out2 = normalize_work({"id": "https://openalex.org/W2"})
    assert out2["counts_by_year_json"] is None
