"""
Search filtering utilities for refining results.

This module provides filtering capabilities for search results based on:
- Source type (slack, github, google_docs)
- Channel/location
- Author/user
- Date ranges
- Metadata attributes
"""

import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SearchFilter:
    """
    Filter for refining search results.

    Attributes:
        source_types: List of allowed source types
        channels: List of allowed channels
        authors: List of allowed authors
        date_from: Start date for filtering
        date_to: End date for filtering
        metadata_filters: Additional metadata filters
    """

    def __init__(
        self,
        source_types: Optional[list[str]] = None,
        channels: Optional[list[str]] = None,
        authors: Optional[list[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        metadata_filters: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize search filter.

        Args:
            source_types: Filter by source types (e.g., ["slack", "github"])
            channels: Filter by channels (e.g., ["#engineering", "main"])
            authors: Filter by authors (e.g., ["alice@example.com"])
            date_from: Start date (inclusive)
            date_to: End date (inclusive)
            metadata_filters: Additional metadata key-value filters
        """
        self.source_types = source_types or []
        self.channels = channels or []
        self.authors = authors or []
        self.date_from = date_from
        self.date_to = date_to
        self.metadata_filters = metadata_filters or {}

        logger.debug(
            f"SearchFilter created: sources={source_types}, "
            f"channels={channels}, authors={authors}"
        )

    def apply(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Apply filters to search results.

        Args:
            results: List of search results

        Returns:
            list: Filtered results
        """
        filtered = results

        # Apply source type filter
        if self.source_types:
            filtered = [
                r for r in filtered
                if r.get("source") in self.source_types
            ]

        # Apply channel filter
        if self.channels:
            filtered = [
                r for r in filtered
                if r.get("channel") in self.channels
            ]

        # Apply author filter
        if self.authors:
            filtered = [
                r for r in filtered
                if r.get("author") in self.authors
            ]

        # Apply date range filter
        if self.date_from or self.date_to:
            filtered = self._filter_by_date(filtered)

        # Apply metadata filters
        if self.metadata_filters:
            filtered = self._filter_by_metadata(filtered)

        logger.info(f"Filtering: {len(results)} -> {len(filtered)} results")

        return filtered

    def _filter_by_date(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter results by date range.

        Args:
            results: Search results

        Returns:
            list: Results within date range
        """
        filtered = []

        for result in results:
            timestamp = result.get("timestamp")

            if not timestamp:
                # Skip if no timestamp
                continue

            # Convert to datetime if needed
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning(f"Invalid timestamp format: {timestamp}")
                    continue

            # Check date range
            if self.date_from and timestamp < self.date_from:
                continue

            if self.date_to and timestamp > self.date_to:
                continue

            filtered.append(result)

        return filtered

    def _filter_by_metadata(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Filter results by metadata attributes.

        Args:
            results: Search results

        Returns:
            list: Results matching metadata filters
        """
        filtered = []

        for result in results:
            metadata = result.get("message_metadata", {})

            # Check if all metadata filters match
            matches = True
            for key, value in self.metadata_filters.items():
                if metadata.get(key) != value:
                    matches = False
                    break

            if matches:
                filtered.append(result)

        return filtered

    def is_empty(self) -> bool:
        """
        Check if filter has any active conditions.

        Returns:
            bool: True if no filters are set
        """
        return (
            not self.source_types
            and not self.channels
            and not self.authors
            and self.date_from is None
            and self.date_to is None
            and not self.metadata_filters
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert filter to dictionary representation.

        Returns:
            dict: Filter configuration
        """
        return {
            "source_types": self.source_types,
            "channels": self.channels,
            "authors": self.authors,
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
            "metadata_filters": self.metadata_filters,
        }


def filter_results(
    results: list[dict[str, Any]],
    source_types: Optional[list[str]] = None,
    channels: Optional[list[str]] = None,
    authors: Optional[list[str]] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    """
    Convenience function to filter search results.

    Args:
        results: Search results to filter
        source_types: Filter by source types
        channels: Filter by channels
        authors: Filter by authors
        date_from: Start date
        date_to: End date

    Returns:
        list: Filtered results
    """
    search_filter = SearchFilter(
        source_types=source_types,
        channels=channels,
        authors=authors,
        date_from=date_from,
        date_to=date_to,
    )

    return search_filter.apply(results)


def apply_threshold(
    results: list[dict[str, Any]],
    threshold: float = 0.0,
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """
    Filter results by minimum score threshold.

    Args:
        results: Search results with scores
        threshold: Minimum score threshold (0.0-1.0)
        score_key: Key name for score field

    Returns:
        list: Results above threshold
    """
    filtered = [
        r for r in results
        if r.get(score_key, 0.0) >= threshold
    ]

    logger.debug(
        f"Threshold filter ({threshold}): {len(results)} -> {len(filtered)} results"
    )

    return filtered
