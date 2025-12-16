"""
Query parsing and understanding for search intent detection.

This module analyzes search queries to:
- Detect query type (keyword, semantic, question)
- Extract query terms for keyword search
- Identify special operators or filters
- Recommend optimal search strategy
"""

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryParser:
    """
    Parse and analyze search queries to determine intent and optimal strategy.

    Attributes:
        default_strategy: Default search strategy if intent unclear
    """

    def __init__(self, default_strategy: str = "hybrid_rrf") -> None:
        """
        Initialize query parser.

        Args:
            default_strategy: Default strategy when intent is unclear
        """
        self.default_strategy = default_strategy
        logger.info(f"QueryParser initialized with default strategy: {default_strategy}")

    def parse(self, query: str) -> dict[str, Any]:
        """
        Parse a search query and extract intent.

        Args:
            query: Raw search query string

        Returns:
            dict: Parsed query information including:
                - original: Original query text
                - normalized: Normalized query text
                - terms: List of individual search terms
                - intent: Detected query intent
                - recommended_strategy: Recommended search strategy
                - filters: Extracted filters (if any)
        """
        if not query or not query.strip():
            return {
                "original": query,
                "normalized": "",
                "terms": [],
                "intent": "empty",
                "recommended_strategy": self.default_strategy,
                "filters": {},
            }

        normalized = self._normalize_query(query)
        terms = self._extract_terms(normalized)
        intent = self._detect_intent(query, terms)
        strategy = self._recommend_strategy(intent, terms)
        filters = self._extract_filters(query)

        parsed = {
            "original": query,
            "normalized": normalized,
            "terms": terms,
            "intent": intent,
            "recommended_strategy": strategy,
            "filters": filters,
        }

        logger.debug(f"Parsed query: {query} -> intent={intent}, strategy={strategy}")

        return parsed

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text.

        - Convert to lowercase
        - Remove extra whitespace
        - Keep special characters for now

        Args:
            query: Raw query string

        Returns:
            str: Normalized query
        """
        # Convert to lowercase
        normalized = query.lower()

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        return normalized

    def _extract_terms(self, normalized_query: str) -> list[str]:
        """
        Extract individual search terms from normalized query.

        Args:
            normalized_query: Normalized query string

        Returns:
            list: List of search terms
        """
        # Simple tokenization (split on whitespace)
        # In production, could use more sophisticated tokenization
        terms = normalized_query.split()

        # Remove very short terms (likely noise)
        terms = [t for t in terms if len(t) > 1]

        return terms

    def _detect_intent(self, query: str, terms: list[str]) -> str:
        """
        Detect the intent of the search query.

        Intent types:
        - question: User asking a question (who, what, when, where, why, how)
        - exact_phrase: Looking for exact phrase (quoted text)
        - keyword: Specific keyword lookup
        - semantic: Conceptual/semantic search
        - boolean: Boolean search (AND, OR, NOT operators)

        Args:
            query: Original query string
            terms: Extracted search terms

        Returns:
            str: Detected intent
        """
        query_lower = query.lower()

        # Check for question words
        question_words = ["who", "what", "when", "where", "why", "how", "which"]
        if any(query_lower.startswith(qw) for qw in question_words):
            return "question"

        # Check for quoted phrases (exact match intent)
        if '"' in query or "'" in query:
            return "exact_phrase"

        # Check for boolean operators
        boolean_ops = [" and ", " or ", " not ", "+", "-"]
        if any(op in query_lower for op in boolean_ops):
            return "boolean"

        # Check for single-word keyword search
        if len(terms) == 1:
            return "keyword"

        # Check for very short queries (1-2 words) - likely keyword
        if len(terms) <= 2:
            return "keyword"

        # Longer queries with multiple terms - likely semantic
        if len(terms) >= 3:
            return "semantic"

        # Default
        return "semantic"

    def _recommend_strategy(self, intent: str, terms: list[str]) -> str:
        """
        Recommend search strategy based on query intent.

        Strategy recommendations:
        - question/semantic -> hybrid_rrf or weighted (with higher semantic weight)
        - keyword -> bm25 or hybrid_weighted (with higher keyword weight)
        - exact_phrase -> bm25 (keyword search better for exact matches)
        - boolean -> bm25 (Elasticsearch handles boolean well)

        Args:
            intent: Detected query intent
            terms: Extracted search terms

        Returns:
            str: Recommended strategy name
        """
        if intent in ["question", "semantic"]:
            # Semantic search is better for questions and conceptual queries
            return "hybrid_rrf"

        elif intent == "keyword":
            if len(terms) == 1:
                # Single keyword - pure BM25 might be better
                return "bm25"
            else:
                # Multiple keywords - hybrid with keyword emphasis
                return "hybrid_weighted"

        elif intent in ["exact_phrase", "boolean"]:
            # Keyword search handles these better
            return "bm25"

        # Default to hybrid RRF
        return self.default_strategy

    def _extract_filters(self, query: str) -> dict[str, Any]:
        """
        Extract filters from query (e.g., source:slack, channel:#engineering).

        Supported filter syntax:
        - source:slack
        - channel:#engineering
        - author:alice@example.com
        - date:2024-12-01

        Args:
            query: Original query string

        Returns:
            dict: Extracted filters
        """
        filters = {}

        # Extract source filter
        source_match = re.search(r'source:(\w+)', query, re.IGNORECASE)
        if source_match:
            filters["source"] = source_match.group(1)

        # Extract channel filter
        channel_match = re.search(r'channel:(#?\w+)', query, re.IGNORECASE)
        if channel_match:
            filters["channel"] = channel_match.group(1)

        # Extract author filter
        author_match = re.search(r'author:([\w@.]+)', query, re.IGNORECASE)
        if author_match:
            filters["author"] = author_match.group(1)

        # Extract date filter
        date_match = re.search(r'date:(\d{4}-\d{2}-\d{2})', query, re.IGNORECASE)
        if date_match:
            filters["date"] = date_match.group(1)

        return filters

    def remove_filters_from_query(self, query: str) -> str:
        """
        Remove filter syntax from query to get clean search text.

        Args:
            query: Query with potential filters

        Returns:
            str: Query with filters removed
        """
        # Remove all filter patterns
        clean_query = re.sub(r'source:\w+', '', query, flags=re.IGNORECASE)
        clean_query = re.sub(r'channel:#?\w+', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'author:[\w@.]+', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'date:\d{4}-\d{2}-\d{2}', '', clean_query, flags=re.IGNORECASE)

        # Clean up extra whitespace
        clean_query = " ".join(clean_query.split())

        return clean_query.strip()


def parse_query(query: str, default_strategy: str = "hybrid_rrf") -> dict[str, Any]:
    """
    Convenience function to parse a query.

    Args:
        query: Search query string
        default_strategy: Default strategy if intent unclear

    Returns:
        dict: Parsed query information
    """
    parser = QueryParser(default_strategy=default_strategy)
    return parser.parse(query)
