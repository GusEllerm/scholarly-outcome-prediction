"""Pydantic schemas for normalized work records (optional; used for validation)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class NormalizedWork(BaseModel):
    """Flat record derived from OpenAlex work; all fields optional for robustness.

    Includes optional text-modality fields for future use (abstracts, full text).
    Training currently uses metadata only.
    """

    openalex_id: str | None = None
    title: str | None = None
    publication_year: int | None = None
    publication_date: str | None = None
    type: str | None = None
    language: str | None = None
    cited_by_count: int | None = None
    referenced_works_count: int | None = None
    authors_count: int | None = None
    institutions_count: int | None = None
    venue_name: str | None = None
    open_access_is_oa: bool | None = None
    primary_topic: str | None = None
    # Optional text-modality fields (not used in training yet)
    abstract_text: str | None = None
    fulltext_text: str | None = None
    has_abstract: bool | None = None
    has_fulltext: bool | None = None
    fulltext_origin: str | None = None

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_openalex_work(cls, raw: dict[str, Any]) -> NormalizedWork:
        """Build from raw OpenAlex work dict (see normalize.py for extraction logic)."""
        return cls(
            openalex_id=_safe_id(raw.get("id")),
            title=raw.get("display_name"),
            publication_year=_safe_int(raw.get("publication_year")),
            publication_date=_safe_str(raw.get("publication_date")),
            type=_safe_str(raw.get("type")),
            language=_safe_str(raw.get("language")),
            cited_by_count=_safe_int(raw.get("cited_by_count")),
            referenced_works_count=_count_list(raw.get("referenced_works")),
            authors_count=_count_authorships(raw.get("authorships")),
            institutions_count=_count_institutions(raw.get("authorships")),
            venue_name=_venue_name(raw),
            open_access_is_oa=_is_oa(raw.get("open_access")),
            primary_topic=_primary_topic(raw.get("topics")),
            abstract_text=_safe_str(_abstract(raw)),
            fulltext_text=None,  # Not in OpenAlex works API by default
            has_abstract=_has_abstract(raw),
            has_fulltext=None,
            fulltext_origin=None,
        )


def _safe_id(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _safe_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _count_list(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, list):
        return len(v)
    return None


def _count_authorships(authorships: Any) -> int | None:
    if authorships is None:
        return None
    if isinstance(authorships, list):
        return len(authorships)  # 0 for empty list
    return None


def _count_institutions(authorships: Any) -> int | None:
    if not isinstance(authorships, list):
        return None
    n = 0
    for a in authorships:
        inst = a.get("institutions") if isinstance(a, dict) else None
        if isinstance(inst, list):
            n += len(inst)
    return n


def _venue_name(raw: dict[str, Any]) -> str | None:
    """
    Extract venue/source name from OpenAlex work.
    Prefer primary_location.source.display_name (journal/serial name);
    fall back to raw_source_name, display_name, venue.display_name;
    then host_venue and locations[0].
    """
    # 1. primary_location.source.display_name (journal/serial)
    pl = raw.get("primary_location")
    if isinstance(pl, dict):
        src = pl.get("source")
        if isinstance(src, dict) and src.get("display_name"):
            out = _safe_str(src["display_name"])
            if out:
                return out
        if pl.get("raw_source_name"):
            out = _safe_str(pl["raw_source_name"])
            if out:
                return out
        if pl.get("display_name"):
            out = _safe_str(pl["display_name"])
            if out:
                return out
        venue = pl.get("venue")
        if isinstance(venue, dict) and venue.get("display_name"):
            out = _safe_str(venue["display_name"])
            if out:
                return out
    # 2. host_venue
    hv = raw.get("host_venue")
    if isinstance(hv, dict):
        if hv.get("display_name"):
            out = _safe_str(hv["display_name"])
            if out:
                return out
        src = hv.get("source")
        if isinstance(src, dict) and src.get("display_name"):
            out = _safe_str(src["display_name"])
            if out:
                return out
    # 3. locations[0].source.display_name
    locs = raw.get("locations")
    if isinstance(locs, list) and locs:
        first = locs[0]
        if isinstance(first, dict):
            src = first.get("source")
            if isinstance(src, dict) and src.get("display_name"):
                out = _safe_str(src["display_name"])
                if out:
                    return out
    return None


def _is_oa(open_access: Any) -> bool | None:
    if open_access is None:
        return None
    if isinstance(open_access, dict):
        return bool(open_access.get("is_oa"))
    return None


def _primary_topic(topics: Any) -> str | None:
    if not isinstance(topics, list) or not topics:
        return None
    first = topics[0]
    if isinstance(first, dict) and "display_name" in first:
        return _safe_str(first["display_name"])
    if isinstance(first, str):
        return first
    return None


def _abstract(raw: dict[str, Any]) -> Any:
    """Extract abstract if present (OpenAlex may provide abstract_inverted_index)."""
    # OpenAlex does not always expose raw abstract; placeholder for future sources
    return raw.get("abstract_inverted_index") or raw.get("abstract")


def _has_abstract(raw: dict[str, Any]) -> bool | None:
    a = _abstract(raw)
    if a is None:
        return None
    if isinstance(a, str):
        return bool(a.strip())
    if isinstance(a, dict):
        return bool(a)
    return None
