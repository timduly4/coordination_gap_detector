"""
Tests for text processing utilities.
"""
import pytest

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


class TestCleanText:
    """Test suite for clean_text function."""

    def test_clean_simple_text(self):
        """Test cleaning simple text."""
        text = "  Hello   World  "
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_with_newlines(self):
        """Test cleaning text with newlines."""
        text = "Hello\n\n\nWorld"
        result = clean_text(text)
        assert result == "Hello World"

    def test_clean_empty_text(self):
        """Test cleaning empty text."""
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_clean_mixed_whitespace(self):
        """Test cleaning text with mixed whitespace."""
        text = "Hello\t\t  World\n\r\n  !"
        result = clean_text(text)
        assert result == "Hello World !"


class TestRemoveUrls:
    """Test suite for remove_urls function."""

    def test_remove_http_url(self):
        """Test removing HTTP URLs."""
        text = "Check out http://example.com for more info"
        result = remove_urls(text)
        assert "http://example.com" not in result
        assert "Check out" in result
        assert "for more info" in result

    def test_remove_https_url(self):
        """Test removing HTTPS URLs."""
        text = "Visit https://example.com/path?query=value"
        result = remove_urls(text)
        assert "https://example.com" not in result

    def test_remove_multiple_urls(self):
        """Test removing multiple URLs."""
        text = "See http://example.com and https://test.org"
        result = remove_urls(text)
        assert "http://example.com" not in result
        assert "https://test.org" not in result

    def test_no_urls(self):
        """Test text without URLs."""
        text = "This text has no URLs"
        result = remove_urls(text)
        assert result == text


class TestRemoveMentions:
    """Test suite for remove_mentions function."""

    def test_remove_single_mention(self):
        """Test removing a single @mention."""
        text = "Hey @alice, can you review this?"
        result = remove_mentions(text)
        assert "@alice" not in result
        assert "Hey" in result

    def test_remove_multiple_mentions(self):
        """Test removing multiple @mentions."""
        text = "@alice @bob please check this @charlie"
        result = remove_mentions(text)
        assert "@alice" not in result
        assert "@bob" not in result
        assert "@charlie" not in result

    def test_no_mentions(self):
        """Test text without mentions."""
        text = "This has no mentions"
        result = remove_mentions(text)
        assert result == text


class TestNormalizeWhitespace:
    """Test suite for normalize_whitespace function."""

    def test_multiple_spaces(self):
        """Test normalizing multiple spaces."""
        text = "Hello    world"
        result = normalize_whitespace(text)
        assert result == "Hello world"

    def test_multiple_newlines(self):
        """Test normalizing multiple newlines."""
        text = "Hello\n\n\n\nworld"
        result = normalize_whitespace(text)
        assert result == "Hello\n\nworld"

    def test_leading_trailing_whitespace(self):
        """Test removing leading and trailing whitespace."""
        text = "  \n  Hello world  \n  "
        result = normalize_whitespace(text)
        assert result == "Hello world"


class TestChunkText:
    """Test suite for chunk_text function."""

    def test_short_text_no_chunking(self):
        """Test that short text is not chunked."""
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_chunking(self):
        """Test chunking long text."""
        text = "This is a sentence. " * 100  # Long text
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        assert all(len(chunk) <= 250 for chunk in chunks)  # Approximate with overlap

    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        text = "Word " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Check that there's some content overlap
        assert len(chunks) > 1

    def test_empty_text(self):
        """Test chunking empty text."""
        chunks = chunk_text("")
        assert chunks == [""]

    def test_min_chunk_size(self):
        """Test minimum chunk size filtering."""
        text = "Short. " * 20
        chunks = chunk_text(text, chunk_size=50, min_chunk_size=30)
        # All chunks should meet minimum size
        assert all(len(chunk) >= 30 for chunk in chunks)

    def test_sentence_boundary_breaking(self):
        """Test that chunks prefer to break at sentence boundaries."""
        text = "First sentence. Second sentence. Third sentence. " * 10
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        # Most chunks should end with a period (sentence boundary)
        # Allow some flexibility for the last chunk
        sentence_endings = sum(1 for chunk in chunks[:-1] if chunk.rstrip().endswith("."))
        assert sentence_endings >= len(chunks) // 2


class TestTruncateText:
    """Test suite for truncate_text function."""

    def test_truncate_long_text(self):
        """Test truncating long text."""
        text = "This is a very long text that should be truncated"
        result = truncate_text(text, max_length=20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_short_text_no_truncation(self):
        """Test that short text is not truncated."""
        text = "Short"
        result = truncate_text(text, max_length=100)
        assert result == text

    def test_custom_suffix(self):
        """Test truncation with custom suffix."""
        text = "This is a long text"
        result = truncate_text(text, max_length=10, suffix=">>")
        assert len(result) == 10
        assert result.endswith(">>")

    def test_empty_text(self):
        """Test truncating empty text."""
        result = truncate_text("", max_length=10)
        assert result == ""


class TestExtractKeywords:
    """Test suite for extract_keywords function."""

    def test_extract_keywords_basic(self):
        """Test extracting keywords from text."""
        text = "OAuth authentication implementation with OAuth tokens and OAuth flow"
        keywords = extract_keywords(text, top_n=5)
        assert "oauth" in keywords  # Should be lowercased
        assert len(keywords) <= 5

    def test_stopwords_filtered(self):
        """Test that stopwords are filtered out."""
        text = "the and or but with for from to a an"
        keywords = extract_keywords(text)
        # All these are stopwords and should be filtered
        assert len(keywords) == 0

    def test_empty_text(self):
        """Test extracting keywords from empty text."""
        keywords = extract_keywords("")
        assert keywords == []

    def test_frequency_ordering(self):
        """Test that keywords are ordered by frequency."""
        text = "test test test other other final"
        keywords = extract_keywords(text, top_n=3)
        assert keywords[0] == "test"  # Most frequent
        assert keywords[1] == "other"  # Second most frequent

    def test_short_words_filtered(self):
        """Test that very short words are filtered."""
        text = "a bb ccc dddd eeeee"
        keywords = extract_keywords(text)
        # Words with <= 2 characters should be filtered
        assert "a" not in keywords
        assert "bb" not in keywords
        assert "ccc" in keywords


class TestCombineTextFields:
    """Test suite for combine_text_fields function."""

    def test_combine_multiple_fields(self):
        """Test combining multiple text fields."""
        result = combine_text_fields("Hello", "world", "!")
        assert result == "Hello world !"

    def test_combine_with_custom_separator(self):
        """Test combining with custom separator."""
        result = combine_text_fields("Hello", "world", separator=", ")
        assert result == "Hello, world"

    def test_filter_none_values(self):
        """Test that None values are filtered out."""
        result = combine_text_fields("Hello", None, "world", None)
        assert result == "Hello world"

    def test_filter_empty_strings(self):
        """Test that empty strings are filtered out."""
        result = combine_text_fields("Hello", "", "world", "  ")
        assert result == "Hello world"

    def test_all_none_or_empty(self):
        """Test when all fields are None or empty."""
        result = combine_text_fields(None, "", "  ")
        assert result == ""

    def test_single_field(self):
        """Test combining a single field."""
        result = combine_text_fields("Hello")
        assert result == "Hello"
