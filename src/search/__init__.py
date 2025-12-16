"""
Search module for multi-source information retrieval.

Provides:
- Keyword-based retrieval with BM25
- Hybrid search (semantic + keyword)
- Query parsing and understanding
- Cross-source search
- Search filters
"""

from src.search.filters import SearchFilter, filter_results, apply_threshold
from src.search.hybrid_search import HybridSearchFusion, deduplicate_results
from src.search.query_parser import QueryParser, parse_query
from src.search.retrieval import KeywordRetriever

__all__ = [
    "KeywordRetriever",
    "HybridSearchFusion",
    "deduplicate_results",
    "QueryParser",
    "parse_query",
    "SearchFilter",
    "filter_results",
    "apply_threshold",
]
