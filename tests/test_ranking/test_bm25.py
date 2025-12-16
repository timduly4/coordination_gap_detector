"""
Unit tests for BM25 scoring implementation.
"""

import math

import pytest

from src.ranking.scoring import BM25Scorer, calculate_collection_stats


class TestBM25Scorer:
    """Test suite for BM25Scorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a BM25 scorer with default parameters."""
        return BM25Scorer(k1=1.5, b=0.75)

    @pytest.fixture
    def custom_scorer(self):
        """Create a BM25 scorer with custom parameters."""
        return BM25Scorer(k1=2.0, b=0.5)

    @pytest.fixture
    def sample_documents(self):
        """Sample document collection for testing."""
        return [
            "OAuth implementation using JWT tokens for authentication",
            "OAuth is a common authentication protocol",
            "Database migration planning and execution steps",
            "API authentication with OAuth and security best practices",
        ]

    def test_initialization_default_params(self, scorer):
        """Test BM25Scorer initializes with default parameters."""
        assert scorer.k1 == 1.5
        assert scorer.b == 0.75

    def test_initialization_custom_params(self, custom_scorer):
        """Test BM25Scorer initializes with custom parameters."""
        assert custom_scorer.k1 == 2.0
        assert custom_scorer.b == 0.5

    def test_calculate_idf_basic(self, scorer):
        """Test IDF calculation for a term."""
        # Term appears in 1 out of 10 documents (rare)
        idf_rare = scorer.calculate_idf("rare", document_frequency=1, total_documents=10)

        # Term appears in 3 out of 10 documents (less common)
        idf_common = scorer.calculate_idf("common", document_frequency=3, total_documents=10)

        # Rare terms should have higher IDF
        assert idf_rare > idf_common
        assert idf_rare > 0
        assert idf_common > 0

    def test_calculate_idf_formula(self, scorer):
        """Test IDF calculation matches expected formula."""
        total_docs = 100
        doc_freq = 10

        idf = scorer.calculate_idf("term", document_frequency=doc_freq, total_documents=total_docs)

        # Expected: ln((N - df + 0.5) / (df + 0.5))
        expected = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        assert abs(idf - expected) < 0.001

    def test_calculate_idf_edge_cases(self, scorer):
        """Test IDF calculation edge cases."""
        # Term in every document (should be low/zero)
        idf_all = scorer.calculate_idf("common", document_frequency=10, total_documents=10)
        assert idf_all >= 0  # Should be non-negative due to max(0, ...)

        # Empty collection
        idf_empty = scorer.calculate_idf("term", document_frequency=0, total_documents=0)
        assert idf_empty == 0.0

    def test_calculate_term_frequency(self, scorer):
        """Test term frequency calculation."""
        document = "OAuth implementation using OAuth for authentication"

        tf_oauth = scorer.calculate_term_frequency("OAuth", document)
        tf_auth = scorer.calculate_term_frequency("authentication", document)
        tf_missing = scorer.calculate_term_frequency("database", document)

        assert tf_oauth == 2  # Appears twice
        assert tf_auth == 1  # Appears once
        assert tf_missing == 0  # Doesn't appear

    def test_calculate_term_frequency_case_insensitive(self, scorer):
        """Test that term frequency is case-insensitive."""
        document = "OAuth OAUTH oauth OaUtH"

        tf = scorer.calculate_term_frequency("oauth", document)

        assert tf == 4

    def test_normalize_document_length(self, scorer):
        """Test document length normalization."""
        avg_length = 100.0

        # Document same as average
        norm_avg = scorer.normalize_document_length(100, avg_length)
        assert abs(norm_avg - 1.0) < 0.001

        # Short document (should get boost)
        norm_short = scorer.normalize_document_length(50, avg_length)
        assert norm_short < 1.0

        # Long document (should get penalty)
        norm_long = scorer.normalize_document_length(200, avg_length)
        assert norm_long > 1.0

    def test_normalize_document_length_no_normalization(self):
        """Test length normalization with b=0 (no normalization)."""
        scorer_no_norm = BM25Scorer(k1=1.5, b=0.0)

        norm = scorer_no_norm.normalize_document_length(200, 100)

        # With b=0, normalization should always be 1.0
        assert abs(norm - 1.0) < 0.001

    def test_normalize_document_length_full_normalization(self):
        """Test length normalization with b=1.0 (full normalization)."""
        scorer_full_norm = BM25Scorer(k1=1.5, b=1.0)

        norm = scorer_full_norm.normalize_document_length(200, 100)

        # With b=1.0, norm = |D| / avgdl = 200 / 100 = 2.0
        assert abs(norm - 2.0) < 0.001

    def test_score_basic(self, scorer):
        """Test basic BM25 scoring."""
        query_terms = ["OAuth", "authentication"]
        document = "OAuth implementation for authentication services"
        doc_length = len(document.split())
        avg_doc_length = 8.0

        # Mock IDF values
        term_idfs = {
            "OAuth": 2.0,
            "authentication": 1.5
        }

        score = scorer.score(
            query_terms,
            document,
            doc_length,
            avg_doc_length,
            term_idfs
        )

        # Score should be positive for relevant document
        assert score > 0

    def test_score_no_matching_terms(self, scorer):
        """Test BM25 score when no query terms match."""
        query_terms = ["database", "migration"]
        document = "OAuth authentication implementation"
        doc_length = len(document.split())
        avg_doc_length = 8.0

        term_idfs = {"database": 2.0, "migration": 2.0}

        score = scorer.score(
            query_terms,
            document,
            doc_length,
            avg_doc_length,
            term_idfs
        )

        # Score should be zero for non-matching document
        assert score == 0.0

    def test_score_relevance_ordering(self, scorer):
        """Test that more relevant documents get higher scores."""
        query_terms = ["OAuth"]

        # Document 1: Multiple mentions of OAuth
        doc1 = "OAuth OAuth OAuth implementation"
        doc1_length = len(doc1.split())

        # Document 2: Single mention of OAuth
        doc2 = "OAuth implementation"
        doc2_length = len(doc2.split())

        avg_doc_length = 5.0
        term_idfs = {"OAuth": 2.0}

        score1 = scorer.score(query_terms, doc1, doc1_length, avg_doc_length, term_idfs)
        score2 = scorer.score(query_terms, doc2, doc2_length, avg_doc_length, term_idfs)

        # Document with more term occurrences should score higher
        assert score1 > score2

    def test_score_with_explanation(self, scorer):
        """Test BM25 scoring with detailed explanation."""
        query_terms = ["OAuth", "authentication"]
        document = "OAuth authentication implementation"
        doc_length = len(document.split())
        avg_doc_length = 5.0
        term_idfs = {"OAuth": 2.0, "authentication": 1.5}

        result = scorer.score_with_explanation(
            query_terms,
            document,
            doc_length,
            avg_doc_length,
            term_idfs
        )

        # Check structure
        assert "total_score" in result
        assert "term_scores" in result
        assert "parameters" in result
        assert "length_normalization" in result

        # Check parameters
        assert result["parameters"]["k1"] == 1.5
        assert result["parameters"]["b"] == 0.75

        # Check term scores
        assert len(result["term_scores"]) == 2

        # Each term should have score breakdown
        for term_score in result["term_scores"]:
            assert "term" in term_score
            assert "tf" in term_score
            assert "idf" in term_score
            assert "score" in term_score

    def test_score_explanation_term_details(self, scorer):
        """Test that score explanation includes correct term details."""
        query_terms = ["test"]
        document = "test test test"
        doc_length = 3
        avg_doc_length = 5.0
        term_idfs = {"test": 1.0}

        result = scorer.score_with_explanation(
            query_terms,
            document,
            doc_length,
            avg_doc_length,
            term_idfs
        )

        # Check term score details
        term_score = result["term_scores"][0]
        assert term_score["term"] == "test"
        assert term_score["tf"] == 3
        assert term_score["idf"] == 1.0
        assert term_score["score"] > 0

    def test_batch_score(self, scorer, sample_documents):
        """Test scoring multiple documents."""
        query_terms = ["OAuth", "authentication"]

        # Prepare documents with lengths
        documents = [
            {"content": doc, "length": len(doc.split())}
            for doc in sample_documents
        ]

        avg_doc_length = sum(d["length"] for d in documents) / len(documents)
        term_idfs = {"OAuth": 2.0, "authentication": 1.5}

        scored_docs = scorer.batch_score(
            query_terms,
            documents,
            avg_doc_length,
            term_idfs
        )

        # Should return same number of documents
        assert len(scored_docs) == len(documents)

        # Each result should be (doc, score) tuple
        for doc, score in scored_docs:
            assert "content" in doc
            assert isinstance(score, float)
            assert score >= 0

        # Results should be sorted by score descending
        scores = [score for _, score in scored_docs]
        assert scores == sorted(scores, reverse=True)

    def test_batch_score_relevance(self, scorer, sample_documents):
        """Test that batch scoring ranks relevant documents higher."""
        query_terms = ["OAuth", "authentication"]

        documents = [
            {"content": doc, "length": len(doc.split())}
            for doc in sample_documents
        ]

        avg_doc_length = sum(d["length"] for d in documents) / len(documents)
        term_idfs = {"OAuth": 2.0, "authentication": 1.5}

        scored_docs = scorer.batch_score(
            query_terms,
            documents,
            avg_doc_length,
            term_idfs
        )

        # First result should be most relevant (contains both terms)
        top_doc, top_score = scored_docs[0]
        assert "OAuth" in top_doc["content"]
        assert "authentication" in top_doc["content"] or "JWT" in top_doc["content"]
        assert top_score > 0


class TestCollectionStats:
    """Test suite for collection statistics calculation."""

    def test_calculate_collection_stats_basic(self):
        """Test basic collection stats calculation."""
        documents = [
            "OAuth implementation",
            "OAuth authentication",
            "database migration"
        ]

        stats = calculate_collection_stats(documents)

        assert stats["total_documents"] == 3
        assert stats["avg_doc_length"] > 0
        assert "term_document_frequencies" in stats

    def test_calculate_collection_stats_avg_length(self):
        """Test average document length calculation."""
        documents = [
            "short doc",  # 2 terms
            "longer document with more terms",  # 5 terms
            "medium length"  # 2 terms
        ]

        stats = calculate_collection_stats(documents)

        # Average should be (2 + 5 + 2) / 3 = 3.0
        assert abs(stats["avg_doc_length"] - 3.0) < 0.001

    def test_calculate_collection_stats_term_frequencies(self):
        """Test term document frequency calculation."""
        documents = [
            "OAuth implementation",
            "OAuth authentication",
            "database migration"
        ]

        stats = calculate_collection_stats(documents)
        term_freqs = stats["term_document_frequencies"]

        # "oauth" appears in 2 documents
        assert term_freqs.get("oauth", 0) == 2

        # "implementation" appears in 1 document
        assert term_freqs.get("implementation", 0) == 1

        # "database" appears in 1 document
        assert term_freqs.get("database", 0) == 1

    def test_calculate_collection_stats_empty(self):
        """Test collection stats with empty collection."""
        documents = []

        stats = calculate_collection_stats(documents)

        assert stats["total_documents"] == 0
        assert stats["avg_doc_length"] == 0.0
        assert stats["term_document_frequencies"] == {}

    def test_calculate_collection_stats_single_doc(self):
        """Test collection stats with single document."""
        documents = ["single document test"]

        stats = calculate_collection_stats(documents)

        assert stats["total_documents"] == 1
        assert stats["avg_doc_length"] == 3.0
        assert len(stats["term_document_frequencies"]) == 3


class TestBM25Parameters:
    """Test suite for BM25 parameter effects."""

    def test_k1_effect_on_saturation(self):
        """Test that k1 parameter affects term frequency saturation."""
        # Low k1 (more saturation)
        scorer_low_k1 = BM25Scorer(k1=0.5, b=0.75)

        # High k1 (less saturation)
        scorer_high_k1 = BM25Scorer(k1=3.0, b=0.75)

        # Document with high term frequency
        query_terms = ["test"]
        document = "test " * 10  # 10 occurrences
        doc_length = 10
        avg_doc_length = 5.0
        term_idfs = {"test": 1.0}

        score_low_k1 = scorer_low_k1.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )
        score_high_k1 = scorer_high_k1.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # Higher k1 should give more weight to term frequency
        assert score_high_k1 > score_low_k1

    def test_b_effect_on_length_normalization(self):
        """Test that b parameter affects length normalization."""
        # No length normalization
        scorer_no_norm = BM25Scorer(k1=1.5, b=0.0)

        # Full length normalization
        scorer_full_norm = BM25Scorer(k1=1.5, b=1.0)

        # Long document with single term match
        query_terms = ["test"]
        document = "test " + "filler " * 50  # 51 terms total
        doc_length = 51
        avg_doc_length = 10.0
        term_idfs = {"test": 1.0}

        score_no_norm = scorer_no_norm.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )
        score_full_norm = scorer_full_norm.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # With b=0, long documents aren't penalized
        # With b=1, long documents are penalized
        assert score_no_norm > score_full_norm
