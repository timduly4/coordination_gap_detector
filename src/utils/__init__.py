"""
Utility functions for the coordination gap detector.
"""
from src.utils.text_processing import (
    chunk_text,
    clean_text,
    combine_text_fields,
    extract_keywords,
    normalize_whitespace,
    remove_mentions,
    remove_urls,
    truncate_text,
)
from src.utils.time_utils import (
    calculate_edit_freshness,
    calculate_recency_score,
    calculate_response_velocity,
    calculate_temporal_relevance,
    detect_activity_burst,
    time_window_overlap,
)

__all__ = [
    # Text processing
    "clean_text",
    "remove_urls",
    "remove_mentions",
    "normalize_whitespace",
    "chunk_text",
    "truncate_text",
    "extract_keywords",
    "combine_text_fields",
    # Time utilities
    "calculate_recency_score",
    "detect_activity_burst",
    "calculate_response_velocity",
    "calculate_temporal_relevance",
    "calculate_edit_freshness",
    "time_window_overlap",
]
