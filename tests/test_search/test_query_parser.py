"""
Unit tests for query parsing and intent detection.
"""

import pytest

from src.search.query_parser import QueryParser, parse_query


class TestQueryParser:
    """Test suite for QueryParser class."""

    @pytest.fixture
    def parser(self):
        """Create a query parser instance."""
        return QueryParser()

    def test_initialization(self, parser):
        """Test parser initialization."""
        assert parser.default_strategy == "hybrid_rrf"

    def test_parse_basic_query(self, parser):
        """Test parsing a basic query."""
        result = parser.parse("OAuth implementation")

        assert result["original"] == "OAuth implementation"
        assert result["normalized"] == "oauth implementation"
        assert "oauth" in result["terms"]
        assert "implementation" in result["terms"]
        assert result["intent"] is not None
        assert result["recommended_strategy"] is not None

    def test_parse_empty_query(self, parser):
        """Test parsing an empty query."""
        result = parser.parse("")

        assert result["intent"] == "empty"
        assert result["terms"] == []

    def test_parse_whitespace_only(self, parser):
        """Test parsing whitespace-only query."""
        result = parser.parse("   ")

        assert result["intent"] == "empty"
        assert result["normalized"] == ""

    def test_detect_question_intent(self, parser):
        """Test detection of question queries."""
        questions = [
            "How do I implement OAuth?",
            "What is the best authentication method?",
            "When should we migrate?",
            "Where is the config file?",
            "Why did this fail?",
            "Which library should we use?",
        ]

        for question in questions:
            result = parser.parse(question)
            assert result["intent"] == "question", f"Failed for: {question}"

    def test_detect_exact_phrase_intent(self, parser):
        """Test detection of exact phrase queries."""
        result = parser.parse('"exact match phrase"')
        assert result["intent"] == "exact_phrase"

        result = parser.parse("'another exact phrase'")
        assert result["intent"] == "exact_phrase"

    def test_detect_boolean_intent(self, parser):
        """Test detection of boolean queries."""
        boolean_queries = [
            "OAuth AND authentication",
            "security OR privacy",
            "database NOT migration",
            "+required -excluded",
        ]

        for query in boolean_queries:
            result = parser.parse(query)
            assert result["intent"] == "boolean", f"Failed for: {query}"

    def test_detect_keyword_intent_single_word(self, parser):
        """Test detection of single keyword queries."""
        result = parser.parse("OAuth")
        assert result["intent"] == "keyword"
        assert len(result["terms"]) == 1

    def test_detect_keyword_intent_two_words(self, parser):
        """Test detection of two-word keyword queries."""
        result = parser.parse("OAuth implementation")
        assert result["intent"] == "keyword"

    def test_detect_semantic_intent(self, parser):
        """Test detection of semantic queries."""
        result = parser.parse("best practices for implementing secure authentication")
        assert result["intent"] == "semantic"
        assert len(result["terms"]) >= 3

    def test_recommend_strategy_question(self, parser):
        """Test strategy recommendation for questions."""
        result = parser.parse("How do I implement OAuth?")
        assert result["recommended_strategy"] == "hybrid_rrf"

    def test_recommend_strategy_semantic(self, parser):
        """Test strategy recommendation for semantic queries."""
        result = parser.parse("best practices for secure authentication flows")
        assert result["recommended_strategy"] == "hybrid_rrf"

    def test_recommend_strategy_single_keyword(self, parser):
        """Test strategy recommendation for single keyword."""
        result = parser.parse("OAuth")
        assert result["recommended_strategy"] == "bm25"

    def test_recommend_strategy_exact_phrase(self, parser):
        """Test strategy recommendation for exact phrases."""
        result = parser.parse('"exact match"')
        assert result["recommended_strategy"] == "bm25"

    def test_recommend_strategy_boolean(self, parser):
        """Test strategy recommendation for boolean queries."""
        result = parser.parse("OAuth AND authentication")
        assert result["recommended_strategy"] == "bm25"

    def test_extract_source_filter(self, parser):
        """Test extraction of source filter."""
        result = parser.parse("OAuth implementation source:slack")
        assert result["filters"].get("source") == "slack"

    def test_extract_channel_filter(self, parser):
        """Test extraction of channel filter."""
        result = parser.parse("bug fix channel:#engineering")
        assert result["filters"].get("channel") == "#engineering"

        result = parser.parse("channel:engineering")
        assert result["filters"].get("channel") == "engineering"

    def test_extract_author_filter(self, parser):
        """Test extraction of author filter."""
        result = parser.parse("OAuth author:alice@example.com")
        assert result["filters"].get("author") == "alice@example.com"

    def test_extract_date_filter(self, parser):
        """Test extraction of date filter."""
        result = parser.parse("migration date:2024-12-01")
        assert result["filters"].get("date") == "2024-12-01"

    def test_extract_multiple_filters(self, parser):
        """Test extraction of multiple filters."""
        result = parser.parse(
            "OAuth source:slack channel:#engineering author:alice@example.com"
        )

        assert result["filters"].get("source") == "slack"
        assert result["filters"].get("channel") == "#engineering"
        assert result["filters"].get("author") == "alice@example.com"

    def test_remove_filters_from_query(self, parser):
        """Test removing filters from query text."""
        query = "OAuth implementation source:slack channel:#engineering"
        clean_query = parser.remove_filters_from_query(query)

        assert clean_query == "OAuth implementation"
        assert "source:" not in clean_query
        assert "channel:" not in clean_query

    def test_remove_all_filter_types(self, parser):
        """Test removing all filter types from query."""
        query = "test source:slack channel:#eng author:alice@example.com date:2024-12-01"
        clean_query = parser.remove_filters_from_query(query)

        assert clean_query == "test"

    def test_normalize_query_lowercase(self, parser):
        """Test query normalization converts to lowercase."""
        result = parser.parse("OAuth Implementation")
        assert result["normalized"] == "oauth implementation"

    def test_normalize_query_whitespace(self, parser):
        """Test query normalization removes extra whitespace."""
        result = parser.parse("OAuth    implementation    guide")
        assert result["normalized"] == "oauth implementation guide"

    def test_extract_terms_filters_short(self, parser):
        """Test that very short terms are filtered out."""
        result = parser.parse("a OAuth implementation b")

        # 'a' and 'b' should be filtered out (single characters)
        assert "oauth" in result["terms"]
        assert "implementation" in result["terms"]
        assert "a" not in result["terms"]
        assert "b" not in result["terms"]

    def test_parse_query_with_special_characters(self, parser):
        """Test parsing queries with special characters."""
        result = parser.parse("OAuth-2.0 implementation!")

        # Should handle special characters gracefully
        assert result["normalized"] is not None
        assert len(result["terms"]) > 0

    def test_default_strategy_override(self):
        """Test overriding default strategy."""
        parser = QueryParser(default_strategy="semantic")
        assert parser.default_strategy == "semantic"


class TestParseQueryFunction:
    """Test suite for parse_query convenience function."""

    def test_parse_query_function(self):
        """Test the parse_query convenience function."""
        result = parse_query("OAuth implementation")

        assert "original" in result
        assert "normalized" in result
        assert "terms" in result
        assert "intent" in result
        assert "recommended_strategy" in result

    def test_parse_query_custom_strategy(self):
        """Test parse_query with custom default strategy."""
        result = parse_query("OAuth", default_strategy="semantic")

        # Parser should be initialized with custom strategy
        assert result["recommended_strategy"] is not None


class TestQueryIntentEdgeCases:
    """Test suite for edge cases in query intent detection."""

    @pytest.fixture
    def parser(self):
        return QueryParser()

    def test_question_word_mid_sentence(self, parser):
        """Test that question words in middle of sentence don't trigger question intent."""
        result = parser.parse("I wonder how this works")

        # Should not be detected as question since "how" is not at start
        assert result["intent"] != "question"

    def test_mixed_case_filters(self, parser):
        """Test that filters are case-insensitive."""
        result = parser.parse("OAuth SOURCE:slack CHANNEL:#engineering")

        assert result["filters"].get("source") == "slack"
        assert result["filters"].get("channel") == "#engineering"

    def test_query_with_numbers(self, parser):
        """Test parsing queries with numbers."""
        result = parser.parse("OAuth 2.0 implementation")

        assert "oauth" in result["terms"]
        assert "2.0" in result["terms"] or "2" in result["terms"]

    def test_very_long_query(self, parser):
        """Test parsing very long queries."""
        long_query = " ".join(["word"] * 100)
        result = parser.parse(long_query)

        assert result["intent"] == "semantic"
        assert len(result["terms"]) == 100

    def test_unicode_characters(self, parser):
        """Test handling of unicode characters."""
        result = parser.parse("OAuth implementation 日本語")

        # Should handle gracefully
        assert result["normalized"] is not None
        assert len(result["terms"]) > 0
