"""
Ranking module for search quality and scoring.

Provides:
- BM25 scoring for keyword search
- Feature extraction for ranking
- Ranking metrics (MRR, NDCG, DCG, Precision@k, Recall@k)
- Evaluation framework with relevance judgments
"""

from src.ranking.evaluation import (
    EvaluationResult,
    QueryResults,
    RankingEvaluator,
    RelevanceJudgment,
    load_labeled_queries_from_file,
)
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
from src.ranking.scoring import BM25Scorer

__all__ = [
    # BM25 Scoring
    "BM25Scorer",
    # Metrics
    "RankingMetrics",
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_dcg",
    "calculate_idcg",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_f1_at_k",
    "calculate_average_precision",
    "calculate_map",
    "calculate_all_metrics",
    # Evaluation
    "RankingEvaluator",
    "RelevanceJudgment",
    "QueryResults",
    "EvaluationResult",
    "load_labeled_queries_from_file",
]
