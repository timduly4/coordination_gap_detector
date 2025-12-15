"""
Search module for multi-source information retrieval.

Provides:
- Keyword-based retrieval with BM25
- Hybrid search (semantic + keyword)
- Query parsing and understanding
- Cross-source search
- Search filters
"""

from src.search.retrieval import KeywordRetriever

__all__ = ["KeywordRetriever"]
