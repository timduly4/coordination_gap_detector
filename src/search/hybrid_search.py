"""
Hybrid search combining semantic similarity and keyword matching.

This module implements fusion strategies to combine results from:
- Semantic search (vector similarity via ChromaDB)
- Keyword search (BM25 via Elasticsearch)

Fusion strategies:
1. Reciprocal Rank Fusion (RRF) - Combines rankings from multiple sources
2. Weighted Score Fusion - Weighted combination of normalized scores
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# RRF constant (standard value from literature)
RRF_K = 60


class HybridSearchFusion:
    """
    Combines results from semantic and keyword search using various fusion strategies.

    Attributes:
        strategy: Fusion strategy to use ("rrf" or "weighted")
        semantic_weight: Weight for semantic scores (used in weighted fusion)
        keyword_weight: Weight for keyword scores (used in weighted fusion)
        rrf_k: Constant for RRF algorithm (default: 60)
    """

    def __init__(
        self,
        strategy: str = "rrf",
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rrf_k: int = RRF_K,
    ) -> None:
        """
        Initialize hybrid search fusion.

        Args:
            strategy: Fusion strategy ("rrf" or "weighted", default: "rrf")
            semantic_weight: Weight for semantic scores (default: 0.7)
            keyword_weight: Weight for keyword scores (default: 0.3)
            rrf_k: RRF constant parameter (default: 60)

        Raises:
            ValueError: If strategy is not supported or weights don't sum to 1.0
        """
        if strategy not in ["rrf", "weighted"]:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")

        if strategy == "weighted":
            total_weight = semantic_weight + keyword_weight
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(
                    f"Weights must sum to 1.0, got {total_weight} "
                    f"(semantic={semantic_weight}, keyword={keyword_weight})"
                )

        self.strategy = strategy
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k

        logger.info(
            f"HybridSearchFusion initialized: strategy={strategy}, "
            f"weights=(semantic={semantic_weight}, keyword={keyword_weight}), "
            f"rrf_k={rrf_k}"
        )

    def fuse(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Fuse results from semantic and keyword search.

        Args:
            semantic_results: Results from semantic search with scores
            keyword_results: Results from keyword search with scores

        Returns:
            list: Fused results sorted by combined score
        """
        if self.strategy == "rrf":
            return self._reciprocal_rank_fusion(semantic_results, keyword_results)
        elif self.strategy == "weighted":
            return self._weighted_score_fusion(semantic_results, keyword_results)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.

        RRF combines rankings from multiple sources using:
        score(d) = Σ 1/(k + rank_i(d))

        where:
        - k is a constant (typically 60)
        - rank_i(d) is the rank of document d in source i

        Benefits:
        - Doesn't require score normalization
        - Robust to different scoring scales
        - Emphasizes top-ranked results

        Args:
            semantic_results: Semantic search results (assumes sorted by score)
            keyword_results: Keyword search results (assumes sorted by score)

        Returns:
            list: Combined results sorted by RRF score
        """
        # Build ranking dictionaries (document_id -> rank)
        semantic_ranks = {}
        keyword_ranks = {}

        # Track all unique documents
        all_docs = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            doc_id = self._get_document_id(result)
            semantic_ranks[doc_id] = rank
            all_docs[doc_id] = result

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = self._get_document_id(result)
            keyword_ranks[doc_id] = rank

            # If not seen in semantic results, add it
            if doc_id not in all_docs:
                all_docs[doc_id] = result

        # Calculate RRF scores
        rrf_scores = []

        for doc_id, doc in all_docs.items():
            rrf_score = 0.0

            # Add contribution from semantic ranking
            if doc_id in semantic_ranks:
                rrf_score += 1.0 / (self.rrf_k + semantic_ranks[doc_id])

            # Add contribution from keyword ranking
            if doc_id in keyword_ranks:
                rrf_score += 1.0 / (self.rrf_k + keyword_ranks[doc_id])

            # Enrich document with ranking details
            enriched_doc = doc.copy()
            enriched_doc["score"] = rrf_score
            enriched_doc["ranking_details"] = {
                "semantic_rank": semantic_ranks.get(doc_id),
                "keyword_rank": keyword_ranks.get(doc_id),
                "semantic_score": doc.get("semantic_score"),
                "keyword_score": doc.get("keyword_score"),
                "fusion_method": "rrf",
                "rrf_k": self.rrf_k,
            }

            rrf_scores.append(enriched_doc)

        # Sort by RRF score descending
        rrf_scores.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"RRF fusion: {len(semantic_results)} semantic + "
            f"{len(keyword_results)} keyword -> {len(rrf_scores)} results"
        )

        return rrf_scores

    def _weighted_score_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Weighted score fusion strategy.

        Combines normalized scores using:
        score(d) = α·semantic_score(d) + β·keyword_score(d)

        where α + β = 1.0

        Args:
            semantic_results: Semantic search results with scores
            keyword_results: Keyword search results with scores

        Returns:
            list: Combined results sorted by weighted score
        """
        # Normalize scores to [0, 1] range
        semantic_normalized = self._normalize_scores(semantic_results, "semantic_score")
        keyword_normalized = self._normalize_scores(keyword_results, "keyword_score")

        # Build document lookup by ID
        semantic_by_id = {
            self._get_document_id(doc): doc
            for doc in semantic_normalized
        }
        keyword_by_id = {
            self._get_document_id(doc): doc
            for doc in keyword_normalized
        }

        # Get all unique document IDs
        all_doc_ids = set(semantic_by_id.keys()) | set(keyword_by_id.keys())

        # Calculate weighted scores
        weighted_results = []

        for doc_id in all_doc_ids:
            semantic_doc = semantic_by_id.get(doc_id)
            keyword_doc = keyword_by_id.get(doc_id)

            # Get normalized scores (0.0 if document not in that result set)
            semantic_score = semantic_doc.get("normalized_score", 0.0) if semantic_doc else 0.0
            keyword_score = keyword_doc.get("normalized_score", 0.0) if keyword_doc else 0.0

            # Calculate weighted combination
            weighted_score = (
                self.semantic_weight * semantic_score +
                self.keyword_weight * keyword_score
            )

            # Use the document from whichever source had it (prefer semantic)
            doc = semantic_doc.copy() if semantic_doc else keyword_doc.copy()

            # Update with combined score and details
            doc["score"] = weighted_score
            doc["ranking_details"] = {
                "semantic_score": semantic_doc.get("semantic_score") if semantic_doc else None,
                "keyword_score": keyword_doc.get("keyword_score") if keyword_doc else None,
                "normalized_semantic": semantic_score,
                "normalized_keyword": keyword_score,
                "fusion_method": "weighted",
                "semantic_weight": self.semantic_weight,
                "keyword_weight": self.keyword_weight,
            }

            weighted_results.append(doc)

        # Sort by weighted score descending
        weighted_results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            f"Weighted fusion: {len(semantic_results)} semantic + "
            f"{len(keyword_results)} keyword -> {len(weighted_results)} results"
        )

        return weighted_results

    def _normalize_scores(
        self,
        results: list[dict[str, Any]],
        score_key: str,
    ) -> list[dict[str, Any]]:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            results: List of results with scores
            score_key: Key to use for normalization

        Returns:
            list: Results with added 'normalized_score' field
        """
        if not results:
            return []

        # Extract scores
        scores = [r.get(score_key, 0.0) for r in results]

        min_score = min(scores)
        max_score = max(scores)

        # Handle case where all scores are the same
        if max_score - min_score < 1e-10:
            normalized = [{"normalized_score": 1.0, **r} for r in results]
            return normalized

        # Min-max normalization
        normalized = []
        for result, score in zip(results, scores):
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append({
                "normalized_score": norm_score,
                **result
            })

        return normalized

    def _get_document_id(self, doc: dict[str, Any]) -> str:
        """
        Extract unique document identifier.

        Uses message_id if available, otherwise external_id.

        Args:
            doc: Document dictionary

        Returns:
            str: Unique document identifier
        """
        if "message_id" in doc:
            return f"msg_{doc['message_id']}"
        elif "external_id" in doc:
            return doc["external_id"]
        else:
            # Fallback: use content hash
            content = doc.get("content", "")
            return f"hash_{hash(content)}"


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Remove duplicate documents from results.

    Deduplication is based on message_id or external_id.
    When duplicates are found, keeps the one with higher score.

    Args:
        results: List of search results

    Returns:
        list: Deduplicated results maintaining score order
    """
    seen_ids = {}
    deduplicated = []

    for result in results:
        # Get document ID
        doc_id = None
        if "message_id" in result:
            doc_id = f"msg_{result['message_id']}"
        elif "external_id" in result:
            doc_id = result["external_id"]

        if doc_id is None:
            # No ID available, keep it
            deduplicated.append(result)
            continue

        # Check if already seen
        if doc_id in seen_ids:
            # Keep the one with higher score
            existing_score = seen_ids[doc_id].get("score", 0.0)
            current_score = result.get("score", 0.0)

            if current_score > existing_score:
                # Replace with higher-scored version
                seen_ids[doc_id] = result
        else:
            seen_ids[doc_id] = result

    # Build deduplicated list preserving order
    deduplicated = list(seen_ids.values())

    # Re-sort by score to maintain ranking
    deduplicated.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)} results")

    return deduplicated
