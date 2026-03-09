"""Tests for venue_name extraction (source.display_name, raw_source_name, locations)."""

from scholarly_outcome_prediction.data import normalize_work


def test_venue_from_primary_location_source_display_name() -> None:
    """Journal article: venue from primary_location.source.display_name."""
    out = normalize_work({
        "id": "https://openalex.org/W1",
        "primary_location": {
            "source": {"display_name": "Nature"},
        },
    })
    assert out["venue_name"] == "Nature"


def test_venue_from_primary_location_raw_source_name() -> None:
    """Book chapter / no source object: venue from primary_location.raw_source_name."""
    out = normalize_work({
        "id": "https://openalex.org/W2",
        "primary_location": {
            "raw_source_name": "Springer eBook",
        },
    })
    assert out["venue_name"] == "Springer eBook"


def test_venue_fallback_display_name_when_source_empty() -> None:
    """When source has no display_name, fall back to primary_location.display_name."""
    out = normalize_work({
        "id": "https://openalex.org/W3",
        "primary_location": {"display_name": "Some Journal"},
    })
    assert out["venue_name"] == "Some Journal"


def test_venue_from_locations_when_primary_missing() -> None:
    """No primary_location: use locations[0].source.display_name."""
    out = normalize_work({
        "id": "https://openalex.org/W4",
        "locations": [
            {"source": {"display_name": "Proceedings of ACM"}},
        ],
    })
    assert out["venue_name"] == "Proceedings of ACM"


def test_venue_from_host_venue_source() -> None:
    """host_venue.source.display_name when primary_location absent."""
    out = normalize_work({
        "id": "https://openalex.org/W5",
        "host_venue": {"source": {"display_name": "Conference X"}},
    })
    assert out["venue_name"] == "Conference X"


def test_venue_missing_all_null() -> None:
    """No location info -> venue_name None."""
    out = normalize_work({"id": "https://openalex.org/W0"})
    assert out["venue_name"] is None
