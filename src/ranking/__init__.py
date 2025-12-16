"""
Ranking module for search quality and scoring.

Provides:
- BM25 scoring for keyword search
- Feature extraction for ranking
- Ranking metrics (MRR, NDCG, DCG)
- Evaluation framework
"""

from src.ranking.scoring import BM25Scorer

__all__ = ["BM25Scorer"]
