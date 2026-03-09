"""Tests for OpenAlex query construction (no live API)."""

from unittest.mock import patch, MagicMock

from scholarly_outcome_prediction.acquisition.openalex_client import (
    fetch_works_page,
    fetch_works_sample_stratified,
    _years_from_date_range,
    OPENALEX_WORKS_URL,
)


def test_fetch_works_page_params_no_type_no_sort() -> None:
    """Without work_types or sort, filter is date-only and no sort param."""
    resp = MagicMock()
    resp.json.return_value = {"results": [], "meta": {}}
    resp.raise_for_status = MagicMock()
    with patch("scholarly_outcome_prediction.acquisition.openalex_client.requests.get", return_value=resp) as m:
        fetch_works_page(
            from_date="2018-01-01",
            to_date="2018-12-31",
            per_page=25,
            cursor="*",
        )
        m.assert_called_once()
        call_kwargs = m.call_args[1]
        params = call_kwargs["params"]
        assert "from_publication_date:2018-01-01" in params["filter"]
        assert "to_publication_date:2018-12-31" in params["filter"]
        assert "type:" not in params["filter"]
        assert "sort" not in params


def test_fetch_works_page_params_with_work_types_and_sort() -> None:
    """With work_types and sort, filter includes type and sort is set."""
    resp = MagicMock()
    resp.json.return_value = {"results": [], "meta": {}}
    resp.raise_for_status = MagicMock()
    with patch("scholarly_outcome_prediction.acquisition.openalex_client.requests.get", return_value=resp) as m:
        fetch_works_page(
            from_date="2015-01-01",
            to_date="2020-12-31",
            per_page=200,
            cursor="*",
            work_types=["article"],
            sort="publication_date:asc",
        )
        m.assert_called_once()
        call_kwargs = m.call_args[1]
        params = call_kwargs["params"]
        assert "article" in params["filter"]
        assert params.get("sort") == "publication_date:asc"


def test_fetch_works_page_url() -> None:
    """Request goes to OpenAlex works endpoint."""
    resp = MagicMock()
    resp.json.return_value = {"results": [], "meta": {}}
    resp.raise_for_status = MagicMock()
    with patch("scholarly_outcome_prediction.acquisition.openalex_client.requests.get", return_value=resp) as m:
        fetch_works_page(from_date="2018-01-01", to_date="2018-12-31")
        m.assert_called_once()
        assert m.call_args[0][0] == OPENALEX_WORKS_URL


def test_years_from_date_range() -> None:
    """Year range parsing for stratified fetch."""
    assert _years_from_date_range("2015-01-01", "2020-12-31") == [2015, 2016, 2017, 2018, 2019, 2020]
    assert _years_from_date_range("2018-06-01", "2018-08-31") == [2018]


def test_fetch_works_random_sample_uses_sample_and_seed() -> None:
    """Representative within-year random sampling uses OpenAlex sample+seed."""
    from scholarly_outcome_prediction.acquisition.openalex_client import fetch_works_random_sample

    resp = MagicMock()
    resp.json.return_value = {"results": [{"id": f"https://openalex.org/W{i}"} for i in range(5)], "meta": {}}
    resp.raise_for_status = MagicMock()
    with patch("scholarly_outcome_prediction.acquisition.openalex_client.requests.get", return_value=resp) as m:
        fetch_works_random_sample(
            from_date="2018-01-01",
            to_date="2018-12-31",
            sample_size=10,
            seed=42,
            work_types=["article"],
        )
        m.assert_called_once()
        params = m.call_args[1]["params"]
        assert params.get("sample") == 10
        assert params.get("seed") == 42
        assert "article" in params["filter"]


def test_fetch_works_sample_stratified_does_not_use_sort() -> None:
    """Representative/temporal stratified fetch must not use sort (avoids oldest-first slice)."""
    resp = MagicMock()
    resp.json.return_value = {"results": [{"id": f"https://openalex.org/W{i}"} for i in range(5)], "meta": {}}
    resp.raise_for_status = MagicMock()
    with patch("scholarly_outcome_prediction.acquisition.openalex_client.requests.get", return_value=resp) as m:
        out = fetch_works_sample_stratified(
            from_date="2018-01-01",
            to_date="2018-12-31",
            sample_size=10,
            work_types=["article"],
        )
        # Should have called for year 2018 only; no sort param
        assert m.called
        for call in m.call_args_list:
            params = call[1]["params"]
            assert "sort" not in params or params.get("sort") is None
