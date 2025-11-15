"""Data models for advanced search functionality.

This module provides data models for search filters, facets, and results
used throughout the advanced search system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class SearchFilter:
    """Represents a search filter criterion."""

    field: str
    operator: str  # 'eq', 'ne', 'in', 'not_in', 'contains', 'starts_with', 'ends_with', 'range'
    value: str | list[str] | tuple[Any, Any]
    negate: bool = False


@dataclass
class SearchFacet:
    """Represents a search facet with possible values."""

    name: str
    values: list[tuple[str, int]]  # (value, count) tuples
    facet_type: str = "terms"  # 'terms', 'range', 'date'


@dataclass
class SearchResult:
    """Enhanced search result with metadata."""

    content_id: str
    content_type: str
    title: str
    content: str
    score: float
    project: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    facets: dict[str, Any] = field(default_factory=dict)
