"""
Evaluation framework for search ranking quality.

This module provides utilities for evaluating search systems using
relevance judgments and standard IR metrics.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.ranking.metrics import (
    RankingMetrics,
    calculate_all_metrics,
    calculate_map,
    calculate_mrr,
    calculate_ndcg,
)

logger = logging.getLogger(__name__)


@dataclass
class RelevanceJudgment:
    """
    Relevance judgment for a query-document pair.

    Attributes:
        query_id: Unique identifier for the query
        document_id: Unique identifier for the document
        relevance: Relevance score (typically 0-3 scale)
                  3 = highly relevant
                  2 = relevant
                  1 = partially relevant
                  0 = not relevant
        query_text: Optional query text
        document_content: Optional document content
        metadata: Additional metadata
    """

    query_id: str
    document_id: str
    relevance: int
    query_text: Optional[str] = None
    document_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate relevance score."""
        if not 0 <= self.relevance <= 3:
            raise ValueError(
                f"Relevance must be 0-3, got {self.relevance}"
            )


@dataclass
class QueryResults:
    """
    Search results for a single query with rankings.

    Attributes:
        query_id: Unique identifier for the query
        query_text: Query text
        results: List of (document_id, score) tuples in ranked order
        metadata: Additional metadata
    """

    query_id: str
    query_text: str
    results: List[tuple[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Evaluation results for a query or set of queries.

    Attributes:
        query_id: Query identifier (or "aggregate" for multiple queries)
        metrics: Calculated ranking metrics
        num_results: Number of results evaluated
        num_relevant: Number of relevant results
        details: Additional evaluation details
    """

    query_id: str
    metrics: RankingMetrics
    num_results: int
    num_relevant: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "query_id": self.query_id,
            "metrics": self.metrics.to_dict(),
            "num_results": self.num_results,
            "num_relevant": self.num_relevant,
            "details": self.details,
        }


class RankingEvaluator:
    """
    Evaluator for search ranking quality using relevance judgments.

    This class manages relevance judgments and evaluates search results
    using standard IR metrics.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.judgments: Dict[str, Dict[str, int]] = {}  # query_id -> {doc_id: relevance}
        self.queries: Dict[str, str] = {}  # query_id -> query_text
        logger.info("RankingEvaluator initialized")

    def add_judgment(
        self,
        query_id: str,
        document_id: str,
        relevance: int,
        query_text: Optional[str] = None
    ) -> None:
        """
        Add a single relevance judgment.

        Args:
            query_id: Query identifier
            document_id: Document identifier
            relevance: Relevance score (0-3)
            query_text: Optional query text
        """
        if query_id not in self.judgments:
            self.judgments[query_id] = {}

        self.judgments[query_id][document_id] = relevance

        if query_text:
            self.queries[query_id] = query_text

        logger.debug(
            f"Added judgment: query={query_id}, doc={document_id}, rel={relevance}"
        )

    def add_judgments(self, judgments: List[RelevanceJudgment]) -> None:
        """
        Add multiple relevance judgments.

        Args:
            judgments: List of RelevanceJudgment objects
        """
        for judgment in judgments:
            self.add_judgment(
                query_id=judgment.query_id,
                document_id=judgment.document_id,
                relevance=judgment.relevance,
                query_text=judgment.query_text,
            )

        logger.info(f"Added {len(judgments)} relevance judgments")

    def load_judgments_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Load relevance judgments from JSON file.

        Expected format:
        [
            {
                "query_id": "q1",
                "document_id": "doc1",
                "relevance": 3,
                "query_text": "optional query text"
            },
            ...
        ]

        Args:
            file_path: Path to JSON file with judgments
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Judgments file not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        judgments = [RelevanceJudgment(**item) for item in data]
        self.add_judgments(judgments)

        logger.info(f"Loaded {len(judgments)} judgments from {file_path}")

    def get_relevance_scores(
        self,
        query_id: str,
        document_ids: List[str]
    ) -> List[int]:
        """
        Get relevance scores for a list of documents.

        Args:
            query_id: Query identifier
            document_ids: List of document identifiers in ranked order

        Returns:
            List of relevance scores (0 if no judgment available)
        """
        if query_id not in self.judgments:
            logger.warning(f"No judgments found for query: {query_id}")
            return [0] * len(document_ids)

        query_judgments = self.judgments[query_id]

        relevance_scores = [
            query_judgments.get(doc_id, 0)
            for doc_id in document_ids
        ]

        return relevance_scores

    def evaluate_query(
        self,
        query_id: str,
        document_ids: List[str],
        k: int = 10
    ) -> EvaluationResult:
        """
        Evaluate ranking quality for a single query.

        Args:
            query_id: Query identifier
            document_ids: List of document IDs in ranked order
            k: Cutoff for @k metrics

        Returns:
            EvaluationResult with calculated metrics
        """
        # Get relevance scores
        relevance_scores = self.get_relevance_scores(query_id, document_ids)

        # Get total relevant documents for this query
        if query_id in self.judgments:
            total_relevant = sum(
                1 for rel in self.judgments[query_id].values() if rel > 0
            )
        else:
            total_relevant = 0

        # Calculate metrics
        metrics = calculate_all_metrics(
            relevance_scores=relevance_scores,
            k=k,
            total_relevant=total_relevant,
        )

        # Count relevant in results
        num_relevant = sum(1 for r in relevance_scores if r > 0)

        result = EvaluationResult(
            query_id=query_id,
            metrics=metrics,
            num_results=len(document_ids),
            num_relevant=num_relevant,
            details={
                "k": k,
                "total_relevant_available": total_relevant,
                "relevance_scores": relevance_scores[:k],
            }
        )

        logger.info(
            f"Evaluated query {query_id}: NDCG@{k}={metrics.ndcg:.4f}, "
            f"P@{k}={metrics.precision:.4f}"
        )

        return result

    def evaluate_queries(
        self,
        query_results: List[QueryResults],
        k: int = 10
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple queries.

        Args:
            query_results: List of QueryResults objects
            k: Cutoff for @k metrics

        Returns:
            Dictionary mapping query_id to EvaluationResult
        """
        results = {}

        for query_result in query_results:
            document_ids = [doc_id for doc_id, _ in query_result.results]

            evaluation = self.evaluate_query(
                query_id=query_result.query_id,
                document_ids=document_ids,
                k=k
            )

            results[query_result.query_id] = evaluation

        logger.info(f"Evaluated {len(results)} queries")

        return results

    def calculate_aggregate_metrics(
        self,
        evaluations: Dict[str, EvaluationResult]
    ) -> EvaluationResult:
        """
        Calculate aggregate metrics across multiple queries.

        Args:
            evaluations: Dictionary of per-query evaluation results

        Returns:
            EvaluationResult with aggregated metrics
        """
        if not evaluations:
            return EvaluationResult(
                query_id="aggregate",
                metrics=RankingMetrics(),
                num_results=0,
                num_relevant=0,
            )

        # Collect metrics from all queries
        mrr_scores = []
        ndcg_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ap_scores = []

        for eval_result in evaluations.values():
            m = eval_result.metrics

            if m.mrr is not None:
                mrr_scores.append(m.mrr)
            if m.ndcg is not None:
                ndcg_scores.append(m.ndcg)
            if m.precision is not None:
                precision_scores.append(m.precision)
            if m.recall is not None:
                recall_scores.append(m.recall)
            if m.f1 is not None:
                f1_scores.append(m.f1)
            if m.ap is not None:
                ap_scores.append(m.ap)

        # Calculate means
        aggregate_metrics = RankingMetrics(
            mrr=sum(mrr_scores) / len(mrr_scores) if mrr_scores else None,
            ndcg=sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else None,
            precision=sum(precision_scores) / len(precision_scores) if precision_scores else None,
            recall=sum(recall_scores) / len(recall_scores) if recall_scores else None,
            f1=sum(f1_scores) / len(f1_scores) if f1_scores else None,
            ap=sum(ap_scores) / len(ap_scores) if ap_scores else None,
        )

        total_results = sum(e.num_results for e in evaluations.values())
        total_relevant = sum(e.num_relevant for e in evaluations.values())

        aggregate = EvaluationResult(
            query_id="aggregate",
            metrics=aggregate_metrics,
            num_results=total_results,
            num_relevant=total_relevant,
            details={
                "num_queries": len(evaluations),
            }
        )

        ndcg_str = f"{aggregate_metrics.ndcg:.4f}" if aggregate_metrics.ndcg is not None else "N/A"
        mrr_str = f"{aggregate_metrics.mrr:.4f}" if aggregate_metrics.mrr is not None else "N/A"

        logger.info(
            f"Aggregate metrics: NDCG={ndcg_str}, MRR={mrr_str}"
        )

        return aggregate

    def get_query_count(self) -> int:
        """Get number of queries with judgments."""
        return len(self.judgments)

    def get_judgment_count(self) -> int:
        """Get total number of judgments."""
        return sum(len(docs) for docs in self.judgments.values())

    def has_judgments_for_query(self, query_id: str) -> bool:
        """Check if judgments exist for a query."""
        return query_id in self.judgments


def load_labeled_queries_from_file(
    file_path: Union[str, Path]
) -> tuple[List[RelevanceJudgment], RankingEvaluator]:
    """
    Convenience function to load judgments and create evaluator.

    Args:
        file_path: Path to judgments JSON file

    Returns:
        Tuple of (judgments list, evaluator with loaded judgments)
    """
    file_path = Path(file_path)

    with open(file_path, 'r') as f:
        data = json.load(f)

    judgments = [RelevanceJudgment(**item) for item in data]

    evaluator = RankingEvaluator()
    evaluator.add_judgments(judgments)

    logger.info(
        f"Loaded {len(judgments)} judgments for "
        f"{evaluator.get_query_count()} queries from {file_path}"
    )

    return judgments, evaluator
