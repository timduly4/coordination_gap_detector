"""
Text processing utilities for message analysis and chunking.

This module provides utilities for:
- Text cleaning and normalization
- Document chunking for long messages
- Token counting and management
"""
import logging
import re
from typing import List

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean and normalize text for embedding and analysis.

    Args:
        text: Raw text to clean

    Returns:
        Cleaned text

    Examples:
        >>> clean_text("  Hello\\n\\nWorld!  ")
        "Hello World!"
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def remove_urls(text: str) -> str:
    """
    Remove URLs from text.

    Args:
        text: Text potentially containing URLs

    Returns:
        Text with URLs removed

    Examples:
        >>> remove_urls("Check out https://example.com for more info")
        "Check out  for more info"
    """
    url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    return re.sub(url_pattern, "", text)


def remove_mentions(text: str) -> str:
    """
    Remove @mentions from text.

    Args:
        text: Text potentially containing @mentions

    Returns:
        Text with mentions removed

    Examples:
        >>> remove_mentions("Hey @alice, can you review this?")
        "Hey , can you review this?"
    """
    return re.sub(r"@\w+", "", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Text with potentially irregular whitespace

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with double newline
    text = re.sub(r"\n\n+", "\n\n", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    min_chunk_size: int = 100,
) -> List[str]:
    """
    Split text into overlapping chunks for embedding.

    This is useful for long documents that exceed the embedding model's
    context window or for better semantic granularity.

    Args:
        text: Text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum chunk size (chunks smaller than this are discarded)

    Returns:
        List of text chunks

    Examples:
        >>> text = "This is a long text. " * 50
        >>> chunks = chunk_text(text, chunk_size=100, overlap=20)
        >>> len(chunks) > 1
        True
    """
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Calculate end position
        end = start + chunk_size

        # If this is not the last chunk, try to break at a sentence or word boundary
        if end < len(text):
            # Look for sentence boundary (., !, ?) within the last 20% of the chunk
            search_start = end - int(chunk_size * 0.2)
            sentence_match = re.search(r"[.!?]\s", text[search_start:end])

            if sentence_match:
                end = search_start + sentence_match.end()
            else:
                # If no sentence boundary, look for word boundary
                space_match = re.search(r"\s", text[end - 50 : end])
                if space_match:
                    end = (end - 50) + space_match.end()

        # Extract chunk
        chunk = text[start:end].strip()

        # Only add chunk if it meets minimum size
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        # Move start position (with overlap)
        # Ensure we always make forward progress to prevent infinite loops
        next_start = end - overlap
        if next_start <= start:
            # If overlap is too large, just move forward by at least 1 character
            next_start = start + max(1, chunk_size // 2)

        start = next_start

        # Stop if we've reached or passed the end
        if start >= len(text):
            break

    return chunks


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncating

    Returns:
        Truncated text

    Examples:
        >>> truncate_text("Hello world", max_length=5)
        "He..."
    """
    if not text or len(text) <= max_length:
        return text

    return text[: max_length - len(suffix)] + suffix


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract simple keywords from text.

    This is a basic implementation that extracts the most common words
    after filtering out common stopwords.

    Args:
        text: Input text
        top_n: Number of top keywords to return

    Returns:
        List of keywords
    """
    if not text:
        return []

    # Simple stopwords list (English)
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "will",
        "with",
        "we",
        "you",
        "i",
        "me",
        "my",
        "this",
        "these",
        "those",
    }

    # Convert to lowercase and extract words
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter stopwords and short words
    words = [w for w in words if w not in stopwords and len(w) > 2]

    # Count word frequency
    word_counts: dict = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    # Return top N keywords
    return [word for word, count in sorted_words[:top_n]]


def combine_text_fields(*fields: str, separator: str = " ") -> str:
    """
    Combine multiple text fields into a single string.

    Args:
        *fields: Variable number of text fields
        separator: Separator to use between fields

    Returns:
        Combined text

    Examples:
        >>> combine_text_fields("Hello", "world", separator=" ")
        "Hello world"
        >>> combine_text_fields("Hello", None, "world")
        "Hello world"
    """
    # Filter out None and empty strings
    valid_fields = [f.strip() for f in fields if f and f.strip()]
    return separator.join(valid_fields)


# Entity extraction utilities


def normalize_username(username: str, domain: str = "company.com") -> str:
    """
    Normalize a username to email format.

    Args:
        username: Username (with or without @ prefix)
        domain: Email domain to append

    Returns:
        Normalized email address

    Examples:
        >>> normalize_username("@alice", "example.com")
        "alice@example.com"
        >>> normalize_username("alice@example.com", "example.com")
        "alice@example.com"
    """
    # Remove @ prefix if present
    username = username.lstrip("@")

    # If already an email, return as-is
    if "@" in username:
        return username.lower()

    # Add domain
    return f"{username}@{domain}".lower()


def normalize_team_name(team: str) -> str:
    """
    Normalize a team name to standard format.

    Args:
        team: Team name (with or without prefixes/suffixes)

    Returns:
        Normalized team name

    Examples:
        >>> normalize_team_name("@platform-team")
        "platform-team"
        >>> normalize_team_name("#platform")
        "platform-team"
    """
    # Remove @ and # prefixes
    team = team.lstrip("@#").lower()

    # Replace underscores with hyphens
    team = team.replace("_", "-")

    # Ensure -team suffix
    if not team.endswith("-team"):
        team = f"{team}-team"

    return team


def is_email(text: str) -> bool:
    """
    Check if text is a valid email address.

    Args:
        text: Text to check

    Returns:
        True if valid email format

    Examples:
        >>> is_email("alice@example.com")
        True
        >>> is_email("not an email")
        False
    """
    email_pattern = r"^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(email_pattern, text))


def extract_domain_from_email(email: str) -> str:
    """
    Extract domain from email address.

    Args:
        email: Email address

    Returns:
        Domain portion of email

    Examples:
        >>> extract_domain_from_email("alice@example.com")
        "example.com"
    """
    if "@" in email:
        return email.split("@")[1].lower()
    return ""


def is_mention(text: str) -> bool:
    """
    Check if text is a @mention.

    Args:
        text: Text to check

    Returns:
        True if text is a mention

    Examples:
        >>> is_mention("@alice")
        True
        >>> is_mention("alice")
        False
    """
    return bool(re.match(r"^@[a-zA-Z0-9_-]+$", text))


def is_channel(text: str) -> bool:
    """
    Check if text is a #channel.

    Args:
        text: Text to check

    Returns:
        True if text is a channel

    Examples:
        >>> is_channel("#engineering")
        True
        >>> is_channel("engineering")
        False
    """
    return bool(re.match(r"^#[a-zA-Z0-9_-]+$", text))
