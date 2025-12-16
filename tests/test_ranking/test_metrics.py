"""
Unit tests for ranking metrics.
"""

import math

import pytest

from src.ranking.metrics import (
    RankingMetrics,
    calculate_all_metrics,
    calculate_average_precision,
    calculate_dcg,
    calculate_f1_at_k,
    calculate_idcg,
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
    calculate_precision_at_k,
    calculate_recall_at_k,
)


class TestMRR:
    """Test suite for Mean Reciprocal Rank (MRR)."""

    def test_mrr_first_result_relevant(self):
        """Test MRR when first result is relevant."""
        queries = [[1, 0, 0, 0]]
        mrr = calculate_mrr(queries)
        assert abs(mrr - 1.0) < 0.001

    def test_mrr_second_result_relevant(self):
        """Test MRR when second result is relevant."""
        queries = [[0, 1, 0, 0]]
        mrr = calculate_mrr(queries)
        assert abs(mrr - 0.5) < 0.001

    def test_mrr_multiple_queries(self):
        """Test MRR with multiple queries."""
        queries = [
            [0, 1, 0, 0],  # First relevant at rank 2 -> RR = 0.5
            [1, 0, 0, 0],  # First relevant at rank 1 -> RR = 1.0
            [0, 0, 0, 0],  # No relevant -> RR = 0.0
        ]
        mrr = calculate_mrr(queries)
        expected = (0.5 + 1.0 + 0.0) / 3
        assert abs(mrr - expected) < 0.001

    def test_mrr_with_k_cutoff(self):
        """Test MRR with k cutoff."""
        queries = [
            [0, 0, 0, 1, 0],  # Relevant at rank 4
        ]
        # With k=3, relevant result is beyond cutoff
        mrr_k3 = calculate_mrr(queries, k=3)
        assert mrr_k3 == 0.0

        # With k=5, relevant result is included
        mrr_k5 = calculate_mrr(queries, k=5)
        assert abs(mrr_k5 - 0.25) < 0.001

    def test_mrr_empty_queries(self):
        """Test MRR with empty query list."""
        mrr = calculate_mrr([])
        assert mrr == 0.0

    def test_mrr_no_relevant_results(self):
        """Test MRR when no results are relevant."""
        queries = [[0, 0, 0, 0]]
        mrr = calculate_mrr(queries)
        assert mrr == 0.0

    def test_mrr_all_relevant(self):
        """Test MRR when all queries have first result relevant."""
        queries = [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
        mrr = calculate_mrr(queries)
        assert abs(mrr - 1.0) < 0.001


class TestDCG:
    """Test suite for Discounted Cumulative Gain (DCG)."""

    def test_dcg_basic(self):
        """Test basic DCG calculation."""
        relevance = [3, 2, 1, 0]
        dcg = calculate_dcg(relevance)

        # Manual calculation:
        # (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + (2^0-1)/log2(5)
        # = 7/1 + 3/1.585 + 1/2 + 0/2.322
        # = 7.0 + 1.893 + 0.5 + 0.0 = 9.393
        assert abs(dcg - 9.393) < 0.01

    def test_dcg_with_k(self):
        """Test DCG with k cutoff."""
        relevance = [3, 2, 1, 0, 0]
        dcg_k2 = calculate_dcg(relevance, k=2)

        # Only first 2 results
        # 7/1 + 3/1.585 = 8.893
        assert abs(dcg_k2 - 8.893) < 0.01

    def test_dcg_empty(self):
        """Test DCG with empty relevance list."""
        dcg = calculate_dcg([])
        assert dcg == 0.0

    def test_dcg_all_zeros(self):
        """Test DCG when all relevance is zero."""
        relevance = [0, 0, 0, 0]
        dcg = calculate_dcg(relevance)
        assert dcg == 0.0

    def test_dcg_perfect_ranking(self):
        """Test DCG with perfect ranking (descending relevance)."""
        relevance = [3, 2, 1, 0]
        dcg = calculate_dcg(relevance)
        # This is the best possible DCG for these scores
        assert dcg > 0


class TestIDCG:
    """Test suite for Ideal DCG (iDCG)."""

    def test_idcg_sorts_descending(self):
        """Test that iDCG sorts relevance scores descending."""
        relevance = [1, 3, 0, 2]
        idcg = calculate_idcg(relevance)

        # Should calculate DCG for [3, 2, 1, 0]
        expected_dcg = calculate_dcg([3, 2, 1, 0])
        assert abs(idcg - expected_dcg) < 0.001

    def test_idcg_with_k(self):
        """Test iDCG with k cutoff."""
        relevance = [1, 0, 3, 2, 0]
        idcg_k3 = calculate_idcg(relevance, k=3)

        # Should use top 3 of sorted [3, 2, 1, 0, 0]
        expected = calculate_dcg([3, 2, 1], k=3)
        assert abs(idcg_k3 - expected) < 0.001

    def test_idcg_empty(self):
        """Test iDCG with empty list."""
        idcg = calculate_idcg([])
        assert idcg == 0.0


class TestNDCG:
    """Test suite for Normalized DCG (NDCG)."""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking."""
        relevance = [3, 2, 1, 0]  # Already sorted descending
        ndcg = calculate_ndcg(relevance)
        assert abs(ndcg - 1.0) < 0.001

    def test_ndcg_worst_ranking(self):
        """Test NDCG with worst ranking."""
        relevance = [0, 1, 2, 3]  # Reversed order
        ndcg = calculate_ndcg(relevance)
        # Should be significantly less than 1.0
        assert ndcg < 0.7
        assert ndcg > 0.0

    def test_ndcg_with_k(self):
        """Test NDCG with k cutoff."""
        relevance = [2, 3, 1, 0]
        ndcg_k2 = calculate_ndcg(relevance, k=2)

        dcg = calculate_dcg([2, 3], k=2)
        idcg = calculate_idcg([2, 3, 1, 0], k=2)
        expected = dcg / idcg

        assert abs(ndcg_k2 - expected) < 0.001

    def test_ndcg_empty(self):
        """Test NDCG with empty list."""
        ndcg = calculate_ndcg([])
        assert ndcg == 0.0

    def test_ndcg_all_zeros(self):
        """Test NDCG when all relevance is zero."""
        relevance = [0, 0, 0]
        ndcg = calculate_ndcg(relevance)
        # iDCG will be 0, so NDCG returns 0.0
        assert ndcg == 0.0

    def test_ndcg_range(self):
        """Test that NDCG is always between 0 and 1."""
        test_cases = [
            [3, 2, 1, 0],
            [0, 1, 2, 3],
            [1, 1, 1, 1],
            [3, 0, 0, 0],
        ]

        for relevance in test_cases:
            ndcg = calculate_ndcg(relevance)
            assert 0.0 <= ndcg <= 1.0


class TestPrecisionAtK:
    """Test suite for Precision@k."""

    def test_precision_all_relevant(self):
        """Test precision when all results are relevant."""
        relevance = [1, 1, 1, 1]
        precision = calculate_precision_at_k(relevance, k=4)
        assert abs(precision - 1.0) < 0.001

    def test_precision_none_relevant(self):
        """Test precision when no results are relevant."""
        relevance = [0, 0, 0, 0]
        precision = calculate_precision_at_k(relevance, k=4)
        assert precision == 0.0

    def test_precision_half_relevant(self):
        """Test precision with 50% relevant results."""
        relevance = [1, 0, 1, 0]
        precision = calculate_precision_at_k(relevance, k=4)
        assert abs(precision - 0.5) < 0.001

    def test_precision_with_k_cutoff(self):
        """Test precision with k smaller than result list."""
        relevance = [1, 1, 0, 0, 0]
        precision_k2 = calculate_precision_at_k(relevance, k=2)
        assert abs(precision_k2 - 1.0) < 0.001  # Top 2 are both relevant

        precision_k3 = calculate_precision_at_k(relevance, k=3)
        assert abs(precision_k3 - 0.667) < 0.01  # 2 out of 3

    def test_precision_k_zero(self):
        """Test precision with k=0."""
        relevance = [1, 1, 1]
        precision = calculate_precision_at_k(relevance, k=0)
        assert precision == 0.0

    def test_precision_empty_list(self):
        """Test precision with empty relevance list."""
        precision = calculate_precision_at_k([], k=5)
        assert precision == 0.0


class TestRecallAtK:
    """Test suite for Recall@k."""

    def test_recall_all_found(self):
        """Test recall when all relevant items are found."""
        relevance = [1, 1, 1, 0]
        recall = calculate_recall_at_k(relevance, k=4, total_relevant=3)
        assert abs(recall - 1.0) < 0.001

    def test_recall_none_found(self):
        """Test recall when no relevant items are found."""
        relevance = [0, 0, 0, 0]
        recall = calculate_recall_at_k(relevance, k=4, total_relevant=3)
        assert recall == 0.0

    def test_recall_partial(self):
        """Test recall with partial relevant items found."""
        relevance = [1, 0, 1, 0, 0]
        # Found 2 out of 5 total relevant
        recall = calculate_recall_at_k(relevance, k=3, total_relevant=5)
        assert abs(recall - 0.4) < 0.001

    def test_recall_auto_total(self):
        """Test recall with auto-calculated total relevant."""
        relevance = [1, 0, 1, 0, 1]
        recall = calculate_recall_at_k(relevance, k=3)
        # Found 2 out of 3 total in list
        assert abs(recall - 0.667) < 0.01

    def test_recall_k_zero(self):
        """Test recall with k=0."""
        relevance = [1, 1, 1]
        recall = calculate_recall_at_k(relevance, k=0, total_relevant=3)
        assert recall == 0.0

    def test_recall_no_relevant_items(self):
        """Test recall when there are no relevant items."""
        relevance = [0, 0, 0]
        recall = calculate_recall_at_k(relevance, k=3, total_relevant=0)
        assert recall == 0.0


class TestF1AtK:
    """Test suite for F1@k."""

    def test_f1_perfect(self):
        """Test F1 with perfect precision and recall."""
        relevance = [1, 1, 1]
        f1 = calculate_f1_at_k(relevance, k=3, total_relevant=3)
        assert abs(f1 - 1.0) < 0.001

    def test_f1_zero(self):
        """Test F1 when both precision and recall are zero."""
        relevance = [0, 0, 0]
        f1 = calculate_f1_at_k(relevance, k=3, total_relevant=5)
        assert f1 == 0.0

    def test_f1_harmonic_mean(self):
        """Test F1 is harmonic mean of precision and recall."""
        relevance = [1, 0, 1, 0]
        k = 4
        total_relevant = 5

        precision = calculate_precision_at_k(relevance, k)
        recall = calculate_recall_at_k(relevance, k, total_relevant)
        expected_f1 = 2 * (precision * recall) / (precision + recall)

        f1 = calculate_f1_at_k(relevance, k, total_relevant)

        assert abs(f1 - expected_f1) < 0.001


class TestAveragePrecision:
    """Test suite for Average Precision (AP)."""

    def test_ap_perfect_ranking(self):
        """Test AP with perfect ranking."""
        relevance = [1, 1, 1, 0, 0]
        ap = calculate_average_precision(relevance)
        # P@1=1.0, P@2=1.0, P@3=1.0
        # AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert abs(ap - 1.0) < 0.001

    def test_ap_alternating(self):
        """Test AP with alternating relevant/irrelevant."""
        relevance = [1, 0, 1, 0, 1]
        ap = calculate_average_precision(relevance)
        # Relevant at positions 1, 3, 5
        # P@1 = 1/1 = 1.0
        # P@3 = 2/3 = 0.667
        # P@5 = 3/5 = 0.6
        # AP = (1.0 + 0.667 + 0.6) / 3 = 0.756
        assert abs(ap - 0.756) < 0.01

    def test_ap_no_relevant(self):
        """Test AP when no results are relevant."""
        relevance = [0, 0, 0, 0]
        ap = calculate_average_precision(relevance)
        assert ap == 0.0

    def test_ap_empty(self):
        """Test AP with empty list."""
        ap = calculate_average_precision([])
        assert ap == 0.0


class TestMAP:
    """Test suite for Mean Average Precision (MAP)."""

    def test_map_multiple_queries(self):
        """Test MAP with multiple queries."""
        queries = [
            [1, 1, 1],  # AP = 1.0
            [0, 1, 1],  # AP = (0.5 + 0.667) / 2 = 0.583
        ]
        map_score = calculate_map(queries)
        # MAP = (1.0 + 0.583) / 2 = 0.792
        assert abs(map_score - 0.792) < 0.01

    def test_map_empty(self):
        """Test MAP with empty query list."""
        map_score = calculate_map([])
        assert map_score == 0.0

    def test_map_single_query(self):
        """Test MAP with single query."""
        queries = [[1, 0, 1]]
        map_score = calculate_map(queries)
        ap = calculate_average_precision([1, 0, 1])
        assert abs(map_score - ap) < 0.001


class TestRankingMetrics:
    """Test suite for RankingMetrics container class."""

    def test_ranking_metrics_initialization(self):
        """Test RankingMetrics initialization."""
        metrics = RankingMetrics(
            mrr=0.75,
            ndcg=0.85,
            precision=0.8,
            recall=0.7
        )

        assert metrics.mrr == 0.75
        assert metrics.ndcg == 0.85
        assert metrics.precision == 0.8
        assert metrics.recall == 0.7

    def test_ranking_metrics_to_dict(self):
        """Test RankingMetrics to_dict conversion."""
        metrics = RankingMetrics(mrr=0.5, ndcg=0.6)
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["mrr"] == 0.5
        assert result["ndcg"] == 0.6
        assert "precision" in result

    def test_ranking_metrics_repr(self):
        """Test RankingMetrics string representation."""
        metrics = RankingMetrics(mrr=0.75, ndcg=0.85)
        repr_str = repr(metrics)

        assert "RankingMetrics" in repr_str
        assert "mrr=0.7500" in repr_str
        assert "ndcg=0.8500" in repr_str


class TestCalculateAllMetrics:
    """Test suite for calculate_all_metrics."""

    def test_calculate_all_metrics_complete(self):
        """Test calculating all metrics together."""
        relevance = [3, 2, 1, 0]
        k = 4
        total_relevant = 3

        metrics = calculate_all_metrics(
            relevance_scores=relevance,
            k=k,
            total_relevant=total_relevant
        )

        assert metrics.ndcg is not None
        assert metrics.dcg is not None
        assert metrics.precision is not None
        assert metrics.recall is not None
        assert metrics.f1 is not None
        assert metrics.ap is not None

    def test_calculate_all_metrics_with_mrr(self):
        """Test calculating all metrics including MRR."""
        relevance = [3, 2, 1, 0]
        queries_for_mrr = [[1, 0, 0], [0, 1, 0]]

        metrics = calculate_all_metrics(
            relevance_scores=relevance,
            k=3,
            queries_for_mrr=queries_for_mrr
        )

        assert metrics.mrr is not None
        assert 0.0 <= metrics.mrr <= 1.0

    def test_calculate_all_metrics_empty(self):
        """Test calculating metrics with empty relevance."""
        metrics = calculate_all_metrics(
            relevance_scores=[],
            k=5
        )

        # Most metrics should handle empty input gracefully
        assert isinstance(metrics, RankingMetrics)


class TestMetricsEdgeCases:
    """Test suite for edge cases across metrics."""

    def test_graded_relevance(self):
        """Test that metrics handle graded relevance (0-3) correctly."""
        relevance = [3, 2, 1, 0]

        # NDCG should work with graded relevance
        ndcg = calculate_ndcg(relevance)
        assert 0.0 <= ndcg <= 1.0

        # DCG should produce meaningful scores
        dcg = calculate_dcg(relevance)
        assert dcg > 0

    def test_binary_relevance(self):
        """Test that metrics handle binary relevance (0-1) correctly."""
        relevance = [1, 1, 0, 0]

        precision = calculate_precision_at_k(relevance, k=4)
        recall = calculate_recall_at_k(relevance, k=4, total_relevant=2)

        assert precision == 0.5
        assert recall == 1.0

    def test_large_k(self):
        """Test metrics with k larger than result set."""
        relevance = [1, 1, 0]

        precision = calculate_precision_at_k(relevance, k=10)
        # Should only consider available results
        assert precision > 0

    def test_single_result(self):
        """Test metrics with single result."""
        relevance = [1]

        precision = calculate_precision_at_k(relevance, k=1)
        assert precision == 1.0

        ndcg = calculate_ndcg(relevance, k=1)
        assert ndcg == 1.0
