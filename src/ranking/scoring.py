"""
BM25 scoring implementation for keyword-based search.

BM25 (Best Matching 25) is a probabilistic ranking function that
estimates the relevance of documents to a given search query.

Formula:
    score(D,Q) = Σ IDF(qi) · (f(qi,D) · (k1 + 1)) / (f(qi,D) + k1 · (1 - b + b · |D|/avgdl))

Where:
    - f(qi,D) = term frequency of qi in document D
    - |D| = document length (number of terms)
    - avgdl = average document length in the collection
    - k1 = term frequency saturation parameter (default: 1.5)
    - b = length normalization parameter (default: 0.75)
    - IDF(qi) = inverse document frequency of term qi
"""

import logging
import math
from collections import Counter
from typing import Any, Optional

from src.ranking.constants import BM25_B, BM25_K1

logger = logging.getLogger(__name__)


class BM25Scorer:
    """
    BM25 probabilistic ranking function for keyword search.

    Attributes:
        k1: Term frequency saturation parameter (default: 1.5)
        b: Length normalization parameter (default: 0.75)
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        """
        Initialize BM25 scorer with configurable parameters.

        Args:
            k1: Term frequency saturation (1.2-2.0, default: 1.5)
            b: Length normalization (0.0-1.0, default: 0.75)
        """
        self.k1 = k1
        self.b = b
        logger.info(f"BM25Scorer initialized with k1={k1}, b={b}")

    def calculate_idf(
        self,
        term: str,
        document_frequency: int,
        total_documents: int
    ) -> float:
        """
        Calculate Inverse Document Frequency (IDF) for a term.

        IDF measures how rare a term is across the document collection.
        Rare terms get higher weights.

        Formula:
            IDF(t) = ln((N - df + 0.5) / (df + 0.5))

        Where:
            - N = total number of documents
            - df = number of documents containing the term

        Args:
            term: The search term
            document_frequency: Number of documents containing this term
            total_documents: Total number of documents in collection

        Returns:
            float: IDF score (higher = rarer term)
        """
        # Avoid division by zero
        if total_documents == 0:
            return 0.0

        # Classic IDF formula with smoothing
        numerator = total_documents - document_frequency + 0.5
        denominator = document_frequency + 0.5

        idf = math.log(numerator / denominator)

        return max(0.0, idf)  # Ensure non-negative

    def calculate_term_frequency(
        self,
        term: str,
        document: str
    ) -> int:
        """
        Calculate term frequency in a document.

        Args:
            term: The search term
            document: Document text

        Returns:
            int: Number of times term appears in document
        """
        # Simple tokenization (split on whitespace and lowercase)
        # In production, use more sophisticated tokenization
        tokens = document.lower().split()
        term_lower = term.lower()

        return tokens.count(term_lower)

    def normalize_document_length(
        self,
        doc_length: int,
        avg_doc_length: float
    ) -> float:
        """
        Calculate document length normalization factor.

        Longer documents naturally contain more term occurrences.
        This normalization prevents bias toward long documents.

        Formula:
            norm = 1 - b + b · (|D| / avgdl)

        Args:
            doc_length: Length of current document
            avg_doc_length: Average document length in collection

        Returns:
            float: Normalization factor
        """
        if avg_doc_length == 0:
            return 1.0

        return 1 - self.b + self.b * (doc_length / avg_doc_length)

    def score(
        self,
        query_terms: list[str],
        document: str,
        document_length: int,
        avg_doc_length: float,
        term_idfs: dict[str, float]
    ) -> float:
        """
        Calculate BM25 score for a document given a query.

        Args:
            query_terms: List of terms in the search query
            document: Document text
            document_length: Number of terms in document
            avg_doc_length: Average document length in collection
            term_idfs: Pre-calculated IDF scores for query terms

        Returns:
            float: BM25 relevance score (higher = more relevant)
        """
        score = 0.0

        # Calculate length normalization factor once
        length_norm = self.normalize_document_length(
            document_length,
            avg_doc_length
        )

        for term in query_terms:
            # Get IDF for this term
            idf = term_idfs.get(term, 0.0)

            # Get term frequency
            tf = self.calculate_term_frequency(term, document)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm

            term_score = idf * (numerator / denominator)
            score += term_score

        return score

    def score_with_explanation(
        self,
        query_terms: list[str],
        document: str,
        document_length: int,
        avg_doc_length: float,
        term_idfs: dict[str, float]
    ) -> dict[str, Any]:
        """
        Calculate BM25 score with detailed explanation.

        Args:
            query_terms: List of terms in the search query
            document: Document text
            document_length: Number of terms in document
            avg_doc_length: Average document length in collection
            term_idfs: Pre-calculated IDF scores for query terms

        Returns:
            dict: Score and detailed breakdown by term
        """
        total_score = 0.0
        term_scores = []

        length_norm = self.normalize_document_length(
            document_length,
            avg_doc_length
        )

        for term in query_terms:
            idf = term_idfs.get(term, 0.0)
            tf = self.calculate_term_frequency(term, document)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm

            term_score = idf * (numerator / denominator)
            total_score += term_score

            term_scores.append({
                "term": term,
                "tf": tf,
                "idf": idf,
                "score": term_score
            })

        return {
            "total_score": total_score,
            "term_scores": term_scores,
            "document_length": document_length,
            "avg_doc_length": avg_doc_length,
            "length_normalization": length_norm,
            "parameters": {
                "k1": self.k1,
                "b": self.b
            }
        }

    def batch_score(
        self,
        query_terms: list[str],
        documents: list[dict[str, Any]],
        avg_doc_length: float,
        term_idfs: dict[str, float]
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Score multiple documents for a query.

        Args:
            query_terms: List of terms in the search query
            documents: List of document dictionaries with 'content' and 'length'
            avg_doc_length: Average document length in collection
            term_idfs: Pre-calculated IDF scores for query terms

        Returns:
            list: List of (document, score) tuples sorted by score descending
        """
        scored_docs = []

        for doc in documents:
            content = doc.get("content", "")
            doc_length = doc.get("length", len(content.split()))

            score = self.score(
                query_terms,
                content,
                doc_length,
                avg_doc_length,
                term_idfs
            )

            scored_docs.append((doc, score))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs


def calculate_collection_stats(documents: list[str]) -> dict[str, Any]:
    """
    Calculate collection-wide statistics needed for BM25.

    Args:
        documents: List of document texts

    Returns:
        dict: Statistics including avg_doc_length and term document frequencies
    """
    if not documents:
        return {
            "total_documents": 0,
            "avg_doc_length": 0.0,
            "term_document_frequencies": {}
        }

    # Calculate average document length
    total_length = 0
    term_doc_freq: dict[str, int] = {}

    for doc in documents:
        tokens = doc.lower().split()
        total_length += len(tokens)

        # Count unique terms per document
        unique_terms = set(tokens)
        for term in unique_terms:
            term_doc_freq[term] = term_doc_freq.get(term, 0) + 1

    avg_doc_length = total_length / len(documents) if documents else 0.0

    return {
        "total_documents": len(documents),
        "avg_doc_length": avg_doc_length,
        "term_document_frequencies": term_doc_freq
    }
