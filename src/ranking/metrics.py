"""
Information retrieval metrics for evaluating search quality.

This module implements standard IR metrics:
- Mean Reciprocal Rank (MRR): Rank of first relevant result
- NDCG@k: Normalized Discounted Cumulative Gain with graded relevance
- DCG: Discounted Cumulative Gain
- Precision@k: Fraction of relevant items in top k
- Recall@k: Fraction of all relevant items found in top k
- F1@k: Harmonic mean of Precision and Recall

These metrics are used to evaluate and compare search ranking quality.
"""

import logging
import math
from typing import Any, List, Optional, Union

logger = logging.getLogger(__name__)


def calculate_mrr(
    queries: List[List[int]],
    k: Optional[int] = None
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) across multiple queries.

    MRR measures the average rank of the first relevant result.
    Formula: MRR = (1/|Q|) · Σ 1/rank_i

    Args:
        queries: List of relevance lists. Each inner list contains binary
                relevance (1=relevant, 0=not relevant) for each result position.
                Example: [[0, 1, 0], [1, 0, 0]] means first query has first
                relevant at rank 2, second query at rank 1.
        k: Optional cutoff - only consider first k results (default: all)

    Returns:
        float: MRR score in range [0, 1], where 1 is perfect (all first results relevant)

    Example:
        >>> queries = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]]
        >>> calculate_mrr(queries)
        0.5  # (0.5 + 1.0 + 0.0) / 3
    """
    if not queries:
        return 0.0

    reciprocal_ranks = []

    for query_results in queries:
        # Apply k cutoff if specified
        if k is not None:
            query_results = query_results[:k]

        # Find position of first relevant result (1-indexed)
        reciprocal_rank = 0.0
        for i, relevance in enumerate(query_results, start=1):
            if relevance > 0:
                reciprocal_rank = 1.0 / i
                break

        reciprocal_ranks.append(reciprocal_rank)

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    logger.debug(f"MRR calculated: {mrr:.4f} over {len(queries)} queries")

    return mrr


def calculate_dcg(
    relevance_scores: List[Union[int, float]],
    k: Optional[int] = None
) -> float:
    """
    Calculate Discounted Cumulative Gain (DCG).

    DCG measures ranking quality with graded relevance judgments.
    Higher relevance scores at higher ranks contribute more.

    Formula: DCG@k = Σ (2^rel_i - 1) / log2(i + 1)

    Args:
        relevance_scores: List of relevance scores (typically 0-3 scale)
                         in ranked order. Higher scores = more relevant.
        k: Optional cutoff - only consider first k results (default: all)

    Returns:
        float: DCG score (higher is better, no fixed upper bound)

    Example:
        >>> relevance_scores = [3, 2, 1, 0, 0]
        >>> calculate_dcg(relevance_scores, k=3)
        6.89  # (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
    """
    if not relevance_scores:
        return 0.0

    # Apply k cutoff if specified
    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        # DCG formula: (2^rel - 1) / log2(i + 1)
        # For i=1, log2(2)=1, for i=2, log2(3)≈1.585, etc.
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    logger.debug(f"DCG@{k if k else 'all'} calculated: {dcg:.4f}")

    return dcg


def calculate_idcg(
    relevance_scores: List[Union[int, float]],
    k: Optional[int] = None
) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain (iDCG).

    iDCG is the DCG of the ideal ranking (relevance scores sorted descending).
    Used to normalize DCG into NDCG.

    Args:
        relevance_scores: List of relevance scores for all relevant documents
        k: Optional cutoff - only consider first k positions

    Returns:
        float: iDCG score (maximum possible DCG for these relevance scores)
    """
    if not relevance_scores:
        return 0.0

    # Sort relevance scores in descending order for ideal ranking
    ideal_scores = sorted(relevance_scores, reverse=True)

    # Calculate DCG for ideal ranking
    idcg = calculate_dcg(ideal_scores, k=k)

    return idcg


def calculate_ndcg(
    relevance_scores: List[Union[int, float]],
    k: Optional[int] = None,
    ideal_relevance_scores: Optional[List[Union[int, float]]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG).

    NDCG normalizes DCG by dividing by the ideal DCG (iDCG).
    This produces a score in [0, 1] regardless of the number of relevant items.

    Formula: NDCG@k = DCG@k / iDCG@k

    Args:
        relevance_scores: List of relevance scores in ranked order
        k: Optional cutoff - only consider first k results
        ideal_relevance_scores: Optional list of all relevant items' scores
                               for iDCG calculation. If not provided, uses
                               the provided relevance_scores sorted descending.

    Returns:
        float: NDCG score in range [0, 1], where 1 is perfect ranking

    Example:
        >>> relevance_scores = [3, 2, 1, 0]  # Good ranking
        >>> calculate_ndcg(relevance_scores, k=4)
        1.0  # Perfect ranking

        >>> relevance_scores = [0, 1, 2, 3]  # Reversed ranking
        >>> calculate_ndcg(relevance_scores, k=4)
        0.49  # Poor ranking
    """
    if not relevance_scores:
        return 0.0

    # Calculate DCG for the actual ranking
    dcg = calculate_dcg(relevance_scores, k=k)

    # Calculate ideal DCG
    if ideal_relevance_scores is None:
        # Use the provided scores themselves to determine ideal ranking
        ideal_relevance_scores = relevance_scores

    idcg = calculate_idcg(ideal_relevance_scores, k=k)

    # Avoid division by zero
    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg

    logger.debug(
        f"NDCG@{k if k else 'all'} calculated: {ndcg:.4f} "
        f"(DCG={dcg:.4f}, iDCG={idcg:.4f})"
    )

    return ndcg


def calculate_precision_at_k(
    relevance_scores: List[int],
    k: int
) -> float:
    """
    Calculate Precision@k.

    Precision@k is the fraction of relevant items in the top k results.
    Formula: P@k = (relevant items in top k) / k

    Args:
        relevance_scores: Binary relevance (1=relevant, 0=not) in ranked order
        k: Number of top results to consider

    Returns:
        float: Precision@k in range [0, 1]

    Example:
        >>> relevance_scores = [1, 1, 0, 1, 0]
        >>> calculate_precision_at_k(relevance_scores, k=3)
        0.667  # 2 relevant out of 3
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    # Take top k results
    top_k = relevance_scores[:k]

    # Count relevant items (relevance > 0)
    relevant_count = sum(1 for rel in top_k if rel > 0)

    precision = relevant_count / k

    logger.debug(f"Precision@{k} calculated: {precision:.4f}")

    return precision


def calculate_recall_at_k(
    relevance_scores: List[int],
    k: int,
    total_relevant: Optional[int] = None
) -> float:
    """
    Calculate Recall@k.

    Recall@k is the fraction of all relevant items found in the top k results.
    Formula: R@k = (relevant items in top k) / (total relevant items)

    Args:
        relevance_scores: Binary relevance (1=relevant, 0=not) in ranked order
        k: Number of top results to consider
        total_relevant: Total number of relevant items (if known).
                       If None, uses count of relevant items in relevance_scores.

    Returns:
        float: Recall@k in range [0, 1]

    Example:
        >>> relevance_scores = [1, 1, 0, 1, 0]
        >>> calculate_recall_at_k(relevance_scores, k=3, total_relevant=3)
        0.667  # Found 2 out of 3 total relevant items
    """
    if k <= 0 or not relevance_scores:
        return 0.0

    # Take top k results
    top_k = relevance_scores[:k]

    # Count relevant items in top k
    relevant_in_k = sum(1 for rel in top_k if rel > 0)

    # Determine total relevant
    if total_relevant is None:
        total_relevant = sum(1 for rel in relevance_scores if rel > 0)

    # Avoid division by zero
    if total_relevant == 0:
        return 0.0

    recall = relevant_in_k / total_relevant

    logger.debug(
        f"Recall@{k} calculated: {recall:.4f} "
        f"({relevant_in_k}/{total_relevant})"
    )

    return recall


def calculate_f1_at_k(
    relevance_scores: List[int],
    k: int,
    total_relevant: Optional[int] = None
) -> float:
    """
    Calculate F1@k score (harmonic mean of Precision@k and Recall@k).

    Formula: F1@k = 2 · (P@k · R@k) / (P@k + R@k)

    Args:
        relevance_scores: Binary relevance in ranked order
        k: Number of top results to consider
        total_relevant: Total number of relevant items

    Returns:
        float: F1@k score in range [0, 1]

    Example:
        >>> relevance_scores = [1, 1, 0, 1, 0]
        >>> calculate_f1_at_k(relevance_scores, k=3, total_relevant=3)
        0.667
    """
    precision = calculate_precision_at_k(relevance_scores, k)
    recall = calculate_recall_at_k(relevance_scores, k, total_relevant)

    # Avoid division by zero
    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    logger.debug(
        f"F1@{k} calculated: {f1:.4f} "
        f"(P@{k}={precision:.4f}, R@{k}={recall:.4f})"
    )

    return f1


def calculate_average_precision(relevance_scores: List[int]) -> float:
    """
    Calculate Average Precision (AP) for a single query.

    AP is the average of precision values at each relevant result position.
    Formula: AP = (Σ P(k) · rel(k)) / (total relevant items)

    Args:
        relevance_scores: Binary relevance (1=relevant, 0=not) in ranked order

    Returns:
        float: Average Precision in range [0, 1]

    Example:
        >>> relevance_scores = [1, 0, 1, 0, 1]
        >>> calculate_average_precision(relevance_scores)
        0.778  # (1.0 + 0.667 + 0.6) / 3
    """
    if not relevance_scores:
        return 0.0

    total_relevant = sum(relevance_scores)

    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, rel in enumerate(relevance_scores, start=1):
        if rel > 0:
            relevant_count += 1
            # Precision at this position
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i

    ap = precision_sum / total_relevant

    logger.debug(f"Average Precision calculated: {ap:.4f}")

    return ap


def calculate_map(queries_relevance: List[List[int]]) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.

    MAP is the mean of Average Precision scores across all queries.
    Formula: MAP = (1/|Q|) · Σ AP(q)

    Args:
        queries_relevance: List of relevance lists, one per query

    Returns:
        float: MAP score in range [0, 1]

    Example:
        >>> queries = [[1, 0, 1], [1, 1, 0]]
        >>> calculate_map(queries)
        0.875  # Mean of AP scores
    """
    if not queries_relevance:
        return 0.0

    ap_scores = [calculate_average_precision(rel) for rel in queries_relevance]

    map_score = sum(ap_scores) / len(ap_scores)

    logger.debug(
        f"MAP calculated: {map_score:.4f} over {len(queries_relevance)} queries"
    )

    return map_score


class RankingMetrics:
    """
    Container for multiple ranking metrics calculated together.

    Attributes:
        mrr: Mean Reciprocal Rank
        ndcg: Normalized Discounted Cumulative Gain at k
        dcg: Discounted Cumulative Gain at k
        precision: Precision at k
        recall: Recall at k
        f1: F1 score at k
        ap: Average Precision
    """

    def __init__(
        self,
        mrr: Optional[float] = None,
        ndcg: Optional[float] = None,
        dcg: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None,
        f1: Optional[float] = None,
        ap: Optional[float] = None,
    ):
        self.mrr = mrr
        self.ndcg = ndcg
        self.dcg = dcg
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.ap = ap

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "dcg": self.dcg,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "ap": self.ap,
        }

    def __repr__(self) -> str:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if v is not None else f"{k}=None"
            for k, v in self.to_dict().items()
        )
        return f"RankingMetrics({metrics_str})"


def calculate_all_metrics(
    relevance_scores: List[Union[int, float]],
    k: int,
    total_relevant: Optional[int] = None,
    queries_for_mrr: Optional[List[List[int]]] = None
) -> RankingMetrics:
    """
    Calculate all ranking metrics for a result set.

    Args:
        relevance_scores: Relevance scores in ranked order
        k: Cutoff for @k metrics
        total_relevant: Total relevant items (for recall)
        queries_for_mrr: Multiple queries for MRR calculation

    Returns:
        RankingMetrics: Container with all calculated metrics
    """
    # Convert to binary for precision/recall if needed
    binary_relevance = [1 if r > 0 else 0 for r in relevance_scores]

    metrics = RankingMetrics(
        mrr=calculate_mrr(queries_for_mrr, k=k) if queries_for_mrr else None,
        ndcg=calculate_ndcg(relevance_scores, k=k),
        dcg=calculate_dcg(relevance_scores, k=k),
        precision=calculate_precision_at_k(binary_relevance, k=k),
        recall=calculate_recall_at_k(binary_relevance, k=k, total_relevant=total_relevant),
        f1=calculate_f1_at_k(binary_relevance, k=k, total_relevant=total_relevant),
        ap=calculate_average_precision(binary_relevance),
    )

    logger.info(f"All metrics calculated: {metrics}")

    return metrics
