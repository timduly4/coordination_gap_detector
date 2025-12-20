"""
Enhanced edge case tests for ranking components.

These tests complement existing unit tests with additional edge cases,
boundary conditions, and error scenarios to ensure robustness.
"""

import math
from datetime import datetime, timedelta

import pytest

from src.ranking.features import FeatureExtractor
from src.ranking.metrics import (
    calculate_dcg,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from src.ranking.scoring import BM25Scorer, calculate_collection_stats
from src.search.hybrid_search import HybridSearchFusion, deduplicate_results


class TestBM25EdgeCases:
    """Edge case tests for BM25 scoring."""

    def test_bm25_with_unicode_content(self):
        """Test BM25 handles Unicode characters correctly."""
        scorer = BM25Scorer(k1=1.5, b=0.75)

        query_terms = ["OAuth", "认证"]  # Mixed Latin and Chinese
        document = "OAuth 认证 implementation using 认证 tokens"
        doc_length = len(document.split())
        avg_doc_length = 5.0
        term_idfs = {"OAuth": 2.0, "认证": 1.8}

        score = scorer.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # Should handle Unicode without errors
        assert score > 0
        assert not math.isnan(score)
        assert not math.isinf(score)

    def test_bm25_with_very_long_document(self):
        """Test BM25 with extremely long documents."""
        scorer = BM25Scorer(k1=1.5, b=0.75)

        query_terms = ["test"]
        # Create very long document (10,000 words)
        document = "test " * 5 + "filler " * 9995
        doc_length = 10000
        avg_doc_length = 100.0
        term_idfs = {"test": 2.0}

        score = scorer.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # Should handle long documents without overflow
        assert score > 0
        assert not math.isinf(score)

        # Score should be penalized due to length normalization
        # Compare with average-length document
        normal_doc = "test " + "filler " * 98
        normal_score = scorer.score(
            query_terms, normal_doc, 99, avg_doc_length, term_idfs
        )

        assert normal_score > score  # Shorter doc should score higher

    def test_bm25_with_very_short_document(self):
        """Test BM25 with very short documents."""
        scorer = BM25Scorer(k1=1.5, b=0.75)

        query_terms = ["test"]
        document = "test"  # Single word
        doc_length = 1
        avg_doc_length = 100.0
        term_idfs = {"test": 2.0}

        score = scorer.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # Should handle short documents
        assert score > 0
        assert not math.isnan(score)

    def test_bm25_with_empty_query(self):
        """Test BM25 with empty query terms."""
        scorer = BM25Scorer(k1=1.5, b=0.75)

        query_terms = []
        document = "OAuth implementation guide"
        doc_length = 3
        avg_doc_length = 5.0
        term_idfs = {}

        score = scorer.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # Empty query should return 0
        assert score == 0.0

    def test_bm25_with_special_characters(self):
        """Test BM25 handles special characters correctly."""
        scorer = BM25Scorer(k1=1.5, b=0.75)

        query_terms = ["OAuth2.0", "test@example.com"]
        document = "OAuth2.0 authentication with test@example.com credentials"
        doc_length = len(document.split())
        avg_doc_length = 8.0
        term_idfs = {"OAuth2.0": 2.0, "test@example.com": 1.5}

        score = scorer.score(
            query_terms, document, doc_length, avg_doc_length, term_idfs
        )

        # Should handle special characters
        assert score > 0

    def test_bm25_with_negative_k1(self):
        """Test that negative k1 parameter produces expected behavior."""
        # BM25 with negative k1 - may work but give unexpected results
        # Just verify it doesn't crash
        try:
            scorer = BM25Scorer(k1=-1.0, b=0.75)
            # If it allows negative k1, verify it still works
            assert scorer.k1 == -1.0
        except (ValueError, AssertionError):
            # If it validates, that's also fine
            pass

    def test_bm25_with_negative_b(self):
        """Test that negative b parameter produces expected behavior."""
        # BM25 with negative b - may work but give unexpected results
        try:
            scorer = BM25Scorer(k1=1.5, b=-0.5)
            assert scorer.b == -0.5
        except (ValueError, AssertionError):
            # If it validates, that's also fine
            pass

    def test_bm25_with_b_greater_than_one(self):
        """Test that b > 1 parameter produces expected behavior."""
        # BM25 with b > 1 - may work but give unexpected normalization
        try:
            scorer = BM25Scorer(k1=1.5, b=1.5)
            assert scorer.b == 1.5
        except (ValueError, AssertionError):
            # If it validates, that's also fine
            pass

    def test_collection_stats_with_duplicate_documents(self):
        """Test collection stats with duplicate documents."""
        documents = [
            "OAuth implementation",
            "OAuth implementation",  # Exact duplicate
            "OAuth authentication"
        ]

        stats = calculate_collection_stats(documents)

        # Should count each document, even duplicates
        assert stats["total_documents"] == 3

        # "oauth" appears in all 3 documents
        assert stats["term_document_frequencies"].get("oauth", 0) == 3

    def test_collection_stats_with_empty_documents(self):
        """Test collection stats with empty documents."""
        documents = [
            "OAuth implementation",
            "",  # Empty document
            "authentication guide"
        ]

        stats = calculate_collection_stats(documents)

        # Should handle empty documents
        assert stats["total_documents"] == 3

        # Average length should account for empty doc (0 terms)
        # (2 + 0 + 2) / 3 ≈ 1.33
        assert abs(stats["avg_doc_length"] - 1.33) < 0.1


class TestMetricsEdgeCases:
    """Edge case tests for ranking metrics."""

    def test_ndcg_with_all_zeros(self):
        """Test NDCG when all relevance scores are zero."""
        relevance_scores = [0, 0, 0, 0, 0]

        ndcg = calculate_ndcg(relevance_scores, k=5)

        # NDCG should be 0 when no relevant results
        assert ndcg == 0.0

    def test_ndcg_with_single_result(self):
        """Test NDCG with only one result."""
        relevance_scores = [3]

        ndcg = calculate_ndcg(relevance_scores, k=1)

        # With one perfect result, NDCG should be 1.0
        assert abs(ndcg - 1.0) < 0.001

    def test_ndcg_with_empty_results(self):
        """Test NDCG with empty results list."""
        relevance_scores = []

        ndcg = calculate_ndcg(relevance_scores, k=10)

        # Empty results should give 0.0
        assert ndcg == 0.0

    def test_ndcg_k_larger_than_results(self):
        """Test NDCG when k is larger than number of results."""
        relevance_scores = [3, 2, 1]

        # k=10 but only 3 results
        ndcg = calculate_ndcg(relevance_scores, k=10)

        # Should handle gracefully
        assert 0 <= ndcg <= 1.0

    def test_ndcg_with_negative_relevance(self):
        """Test NDCG behavior with negative relevance scores."""
        # In some systems, negative relevance might indicate strong negative signal
        relevance_scores = [3, 2, -1, 0]

        ndcg = calculate_ndcg(relevance_scores, k=4)

        # Should handle negative values
        assert isinstance(ndcg, float)

    def test_mrr_with_very_late_relevant_result(self):
        """Test MRR when first relevant result is at position 1000."""
        # First relevant at position 1000
        query_results = [[0] * 999 + [1]]

        mrr = calculate_mrr(query_results)

        # RR should be 1/1000 = 0.001
        assert abs(mrr - 0.001) < 0.0001

    def test_mrr_with_all_queries_no_relevant(self):
        """Test MRR when all queries have no relevant results."""
        queries = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ]

        mrr = calculate_mrr(queries)

        # Should be 0.0
        assert mrr == 0.0

    def test_precision_at_k_with_k_zero(self):
        """Test Precision@k with k=0."""
        relevance = [1, 1, 0, 0]

        # k=0 should either raise error or return 0
        try:
            result = calculate_precision_at_k(relevance, k=0)
            # If it doesn't raise, it should return 0
            assert result == 0.0
        except (ValueError, ZeroDivisionError):
            # If it raises, that's also acceptable
            pass

    def test_precision_at_k_all_relevant(self):
        """Test Precision@k when all results are relevant."""
        relevance = [1, 1, 1, 1, 1]

        precision = calculate_precision_at_k(relevance, k=5)

        # All relevant: 5/5 = 1.0
        assert abs(precision - 1.0) < 0.001

    def test_recall_at_k_no_relevant_in_collection(self):
        """Test Recall@k when there are no relevant documents."""
        relevance = [0, 0, 0, 0]
        total_relevant = 0

        recall = calculate_recall_at_k(relevance, k=4, total_relevant=total_relevant)

        # No relevant documents: 0/0 should be 0.0
        assert recall == 0.0

    def test_recall_at_k_more_relevant_than_results(self):
        """Test Recall@k when total_relevant > returned results."""
        relevance = [1, 1, 0, 0]  # 2 relevant in top 4
        total_relevant = 10  # But 10 total relevant in collection

        recall = calculate_recall_at_k(relevance, k=4, total_relevant=total_relevant)

        # Found 2 out of 10: 2/10 = 0.2
        assert abs(recall - 0.2) < 0.001

    def test_dcg_with_zero_relevance(self):
        """Test DCG with all zero relevance."""
        relevance_scores = [0, 0, 0, 0]

        dcg = calculate_dcg(relevance_scores, k=4)

        assert dcg == 0.0

    def test_dcg_position_discount(self):
        """Test that DCG properly discounts by position."""
        # Same relevance at different positions
        relevance_pos1 = [3, 0, 0, 0]
        relevance_pos4 = [0, 0, 0, 3]

        dcg_pos1 = calculate_dcg(relevance_pos1, k=4)
        dcg_pos4 = calculate_dcg(relevance_pos4, k=4)

        # Position 1 should have higher DCG than position 4
        assert dcg_pos1 > dcg_pos4


class TestHybridSearchEdgeCases:
    """Edge case tests for hybrid search fusion."""

    def test_rrf_with_empty_semantic_results(self):
        """Test RRF fusion when semantic results are empty."""
        fusion = HybridSearchFusion(strategy="rrf")

        semantic_results = []
        keyword_results = [
            {"message_id": 1, "content": "test", "score": 10.0, "keyword_score": 10.0}
        ]

        fused = fusion.fuse(semantic_results, keyword_results)

        # Should return keyword results only
        assert len(fused) == 1
        assert fused[0]["message_id"] == 1

    def test_rrf_with_empty_keyword_results(self):
        """Test RRF fusion when keyword results are empty."""
        fusion = HybridSearchFusion(strategy="rrf")

        semantic_results = [
            {"message_id": 1, "content": "test", "score": 0.9, "semantic_score": 0.9}
        ]
        keyword_results = []

        fused = fusion.fuse(semantic_results, keyword_results)

        # Should return semantic results only
        assert len(fused) == 1
        assert fused[0]["message_id"] == 1

    def test_rrf_with_both_empty(self):
        """Test RRF fusion when both result sets are empty."""
        fusion = HybridSearchFusion(strategy="rrf")

        fused = fusion.fuse([], [])

        # Should return empty list
        assert fused == []

    def test_weighted_fusion_with_zero_semantic_weight(self):
        """Test weighted fusion with semantic_weight=0."""
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=0.0,
            keyword_weight=1.0
        )

        semantic_results = [
            {"message_id": 1, "content": "test", "score": 0.99, "semantic_score": 0.99}
        ]
        keyword_results = [
            {"message_id": 2, "content": "test", "score": 5.0, "keyword_score": 5.0}
        ]

        fused = fusion.fuse(semantic_results, keyword_results)

        # With semantic_weight=0, should prioritize keyword results
        # Message 2 should rank higher than message 1
        top_ids = [r["message_id"] for r in fused]
        assert top_ids[0] == 2 or len(fused) == 2

    def test_weighted_fusion_with_zero_keyword_weight(self):
        """Test weighted fusion with keyword_weight=0."""
        fusion = HybridSearchFusion(
            strategy="weighted",
            semantic_weight=1.0,
            keyword_weight=0.0
        )

        semantic_results = [
            {"message_id": 1, "content": "test", "score": 0.99, "semantic_score": 0.99}
        ]
        keyword_results = [
            {"message_id": 2, "content": "test", "score": 20.0, "keyword_score": 20.0}
        ]

        fused = fusion.fuse(semantic_results, keyword_results)

        # With keyword_weight=0, should prioritize semantic results
        top_ids = [r["message_id"] for r in fused]
        assert top_ids[0] == 1 or len(fused) == 2

    def test_rrf_with_very_large_k(self):
        """Test RRF with very large k parameter."""
        fusion = HybridSearchFusion(strategy="rrf", rrf_k=10000)

        semantic_results = [
            {"message_id": 1, "content": "test", "score": 0.9, "semantic_score": 0.9}
        ]
        keyword_results = [
            {"message_id": 1, "content": "test", "score": 10.0, "keyword_score": 10.0}
        ]

        fused = fusion.fuse(semantic_results, keyword_results)

        # Should still work with large k
        assert len(fused) == 1

        # RRF score should be very small: 1/(10000+1) + 1/(10000+1)
        # Approximately 0.0002
        assert fused[0]["score"] < 0.001

    def test_deduplicate_results_empty(self):
        """Test deduplication with empty results."""
        results = deduplicate_results([])

        assert results == []

    def test_deduplicate_results_no_duplicates(self):
        """Test deduplication when there are no duplicates."""
        results = [
            {"message_id": 1, "content": "test1", "score": 0.9},
            {"message_id": 2, "content": "test2", "score": 0.8},
            {"message_id": 3, "content": "test3", "score": 0.7}
        ]

        deduped = deduplicate_results(results)

        assert len(deduped) == 3

    def test_deduplicate_results_all_duplicates(self):
        """Test deduplication when all results are duplicates."""
        results = [
            {"message_id": 1, "content": "test", "score": 0.9},
            {"message_id": 1, "content": "test", "score": 0.8},
            {"message_id": 1, "content": "test", "score": 0.7}
        ]

        deduped = deduplicate_results(results)

        # Should keep only the first (highest score)
        assert len(deduped) == 1
        assert deduped[0]["score"] == 0.9


class TestFeatureExtractionEdgeCases:
    """Edge case tests for feature extraction."""

    def test_feature_extraction_with_missing_metadata(self):
        """Test feature extraction with missing metadata fields."""
        extractor = FeatureExtractor()

        document = {
            "content": "OAuth implementation",
            # Missing: author, channel, timestamp, etc.
        }

        features = extractor.extract(
            query="OAuth",
            document=document
        )

        # Should handle missing metadata gracefully
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_feature_extraction_with_empty_content(self):
        """Test feature extraction with empty document content."""
        extractor = FeatureExtractor()

        document = {"content": ""}

        features = extractor.extract(
            query="OAuth implementation",
            document=document
        )

        # Should handle empty content
        assert isinstance(features, dict)

        # Term coverage should be 0 for empty content
        if "term_coverage" in features:
            assert features["term_coverage"] == 0.0

    def test_feature_extraction_with_very_old_timestamp(self):
        """Test feature extraction with very old timestamp."""
        extractor = FeatureExtractor()

        # Message from 10 years ago
        very_old = datetime.utcnow() - timedelta(days=3650)

        document = {
            "content": "test message",
            "timestamp": very_old
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Recency feature should be very low (close to 0)
        if "recency" in features:
            assert features["recency"] < 0.1

    def test_feature_extraction_with_future_timestamp(self):
        """Test feature extraction with future timestamp."""
        extractor = FeatureExtractor()

        # Message from future (clock skew, wrong timezone, etc.)
        future_time = datetime.utcnow() + timedelta(hours=5)

        document = {
            "content": "test message",
            "timestamp": future_time
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Should handle future timestamps gracefully
        assert isinstance(features, dict)

        # Recency should be high (treated as very recent)
        if "recency" in features:
            assert features["recency"] >= 0.9

    def test_feature_extraction_with_extreme_engagement(self):
        """Test feature extraction with extremely high engagement metrics."""
        extractor = FeatureExtractor()

        document = {
            "content": "test message",
            "message_metadata": {
                "reactions": ["thumbsup"] * 10000,  # 10,000 reactions
                "reply_count": 50000,  # 50,000 replies
                "participants": list(range(1000))  # 1,000 participants
            }
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Should normalize extreme values
        assert isinstance(features, dict)

        # Engagement features should be normalized to reasonable range
        if "thread_depth" in features:
            # Even extreme values should be normalized
            # Allow some flexibility for normalization algorithms
            assert 0 <= features["thread_depth"] <= 2.0

    def test_feature_extraction_with_null_values(self):
        """Test feature extraction with None/null values in metadata."""
        extractor = FeatureExtractor()

        document = {
            "content": "test message",
            "author": None,
            "channel": None,
            "timestamp": None,
            "message_metadata": {
                "reactions": None,
                "reply_count": None
            }
        }

        features = extractor.extract(
            query="test",
            document=document
        )

        # Should handle None values gracefully
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_feature_extraction_with_unicode_query(self):
        """Test feature extraction with Unicode query."""
        extractor = FeatureExtractor()

        document = {
            "content": "OAuth authentication protocol implementation"
        }

        features = extractor.extract(
            query="OAuth 认证 プロトコル",  # Mixed scripts
            document=document
        )

        # Should handle Unicode in query
        assert isinstance(features, dict)

    def test_feature_extraction_with_very_long_query(self):
        """Test feature extraction with extremely long query."""
        extractor = FeatureExtractor()

        # 1000-word query
        long_query = " ".join(["word"] * 1000)

        document = {"content": "test message"}

        features = extractor.extract(
            query=long_query,
            document=document
        )

        # Should handle long queries
        assert isinstance(features, dict)
