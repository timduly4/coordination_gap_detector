"""
Evaluation service for offline ranking quality assessment.

This service coordinates evaluation operations, including:
- Running evaluations across test queries
- Comparing multiple ranking strategies
- Statistical significance testing
- Result export and formatting
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.models.schemas import SearchRequest
from src.ranking.evaluation import (
    EvaluationResult,
    QueryResults,
    RankingEvaluator,
    RelevanceJudgment,
)
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Service for evaluating ranking strategies using test queries and relevance judgments.

    This service provides methods for:
    - Loading test queries and relevance judgments
    - Running evaluations across multiple strategies
    - Comparing strategy performance
    - Exporting results in various formats
    """

    def __init__(self, search_service: SearchService):
        """
        Initialize the evaluation service.

        Args:
            search_service: SearchService instance for executing queries
        """
        self.search_service = search_service
        self.evaluator = RankingEvaluator()
        logger.info("EvaluationService initialized")

    def load_judgments(self, file_path: str) -> int:
        """
        Load relevance judgments from a JSON file.

        Args:
            file_path: Path to judgments file

        Returns:
            Number of judgments loaded
        """
        self.evaluator.load_judgments_from_file(file_path)
        count = self.evaluator.get_judgment_count()
        logger.info(f"Loaded {count} relevance judgments from {file_path}")
        return count

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
        self.evaluator.add_judgment(
            query_id=query_id,
            document_id=document_id,
            relevance=relevance,
            query_text=query_text
        )

    async def evaluate_strategy(
        self,
        strategy: str,
        test_queries: List[Dict[str, Any]],
        db: AsyncSession,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate a single ranking strategy using test queries.

        Args:
            strategy: Ranking strategy name (semantic, bm25, hybrid_rrf, etc.)
            test_queries: List of test queries with query_id and query_text
            db: Database session
            k: Cutoff for @k metrics

        Returns:
            Dictionary with evaluation results and metrics
        """
        logger.info(f"Evaluating strategy '{strategy}' on {len(test_queries)} queries")

        query_evaluations = {}

        for test_query in test_queries:
            query_id = test_query["query_id"]
            query_text = test_query["query_text"]

            # Execute search with the specified strategy
            search_request = SearchRequest(
                query=query_text,
                ranking_strategy=strategy,
                limit=k
            )

            try:
                search_response = await self.search_service.search(search_request, db)

                # Extract document IDs from results
                document_ids = [
                    str(result.message_id) for result in search_response.results
                ]

                # Evaluate this query
                evaluation = self.evaluator.evaluate_query(
                    query_id=query_id,
                    document_ids=document_ids,
                    k=k
                )

                query_evaluations[query_id] = evaluation

            except Exception as e:
                logger.error(f"Error evaluating query {query_id}: {e}")
                continue

        # Calculate aggregate metrics
        aggregate = self.evaluator.calculate_aggregate_metrics(query_evaluations)

        result = {
            "strategy": strategy,
            "num_queries": len(query_evaluations),
            "aggregate_metrics": aggregate.metrics.to_dict(),
            "per_query_results": {
                qid: eval_result.to_dict()
                for qid, eval_result in query_evaluations.items()
            }
        }

        ndcg_val = aggregate.metrics.ndcg if aggregate.metrics.ndcg is not None else 0.0
        mrr_val = aggregate.metrics.mrr if aggregate.metrics.mrr is not None else 0.0
        logger.info(
            f"Strategy '{strategy}' evaluation complete: "
            f"NDCG@{k}={ndcg_val:.4f}, "
            f"MRR={mrr_val:.4f}"
        )

        return result

    async def compare_strategies(
        self,
        strategies: List[str],
        test_queries: List[Dict[str, Any]],
        db: AsyncSession,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Compare multiple ranking strategies.

        Args:
            strategies: List of strategy names to compare
            test_queries: List of test queries
            db: Database session
            k: Cutoff for @k metrics

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing {len(strategies)} strategies on {len(test_queries)} queries")

        strategy_results = {}

        # Evaluate each strategy
        for strategy in strategies:
            result = await self.evaluate_strategy(
                strategy=strategy,
                test_queries=test_queries,
                db=db,
                k=k
            )
            strategy_results[strategy] = result

        # Extract metrics for comparison
        comparison = {
            "strategies": strategies,
            "num_queries": len(test_queries),
            "k": k,
            "results": {}
        }

        # Organize by metric type
        metric_names = ["mrr", "ndcg", "precision", "recall", "f1"]

        for metric_name in metric_names:
            comparison["results"][metric_name] = {}
            for strategy, result in strategy_results.items():
                metric_value = result["aggregate_metrics"].get(metric_name)
                comparison["results"][metric_name][strategy] = metric_value

        # Find best strategy (by MRR)
        best_strategy = max(
            strategies,
            key=lambda s: strategy_results[s]["aggregate_metrics"].get("mrr", 0) or 0
        )
        comparison["best_strategy"] = best_strategy

        # Calculate improvement over first strategy (baseline)
        if len(strategies) > 1:
            baseline_strategy = strategies[0]
            baseline_mrr = strategy_results[baseline_strategy]["aggregate_metrics"].get("mrr", 0) or 0
            best_mrr = strategy_results[best_strategy]["aggregate_metrics"].get("mrr", 0) or 0

            if baseline_mrr > 0:
                improvement = ((best_mrr - baseline_mrr) / baseline_mrr) * 100
                comparison["improvement_over_baseline"] = round(improvement, 2)
            else:
                comparison["improvement_over_baseline"] = 0.0
        else:
            comparison["improvement_over_baseline"] = 0.0

        logger.info(
            f"Best strategy: {best_strategy} "
            f"(+{comparison['improvement_over_baseline']:.1f}% over baseline)"
        )

        return comparison

    def export_results(
        self,
        comparison_results: Dict[str, Any],
        output_format: str = "dict"
    ) -> Any:
        """
        Export evaluation results in specified format.

        Args:
            comparison_results: Results from compare_strategies
            output_format: Output format ('dict', 'json', 'csv', 'table')

        Returns:
            Formatted results
        """
        if output_format == "dict":
            return comparison_results

        elif output_format == "json":
            import json
            return json.dumps(comparison_results, indent=2)

        elif output_format == "csv":
            # Create CSV rows
            rows = []
            strategies = comparison_results["strategies"]
            metrics = comparison_results["results"]

            # Header
            header = ["Strategy"] + list(metrics.keys())
            rows.append(header)

            # Data rows
            for strategy in strategies:
                row = [strategy]
                for metric_name in metrics.keys():
                    value = metrics[metric_name].get(strategy, 0.0) or 0.0
                    row.append(f"{value:.4f}")
                rows.append(row)

            return rows

        elif output_format == "table":
            return self._format_comparison_table(comparison_results)

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _format_comparison_table(self, comparison: Dict[str, Any]) -> str:
        """
        Format comparison results as a readable table.

        Args:
            comparison: Comparison results dictionary

        Returns:
            Formatted table string
        """
        k = comparison.get("k", 10)
        strategies = comparison["strategies"]
        metrics = comparison["results"]

        # Header
        table = "Strategy Comparison:\n"
        table += "┌─────────────────┬───────┬──────────┬──────────┬──────────┐\n"
        table += f"│ Strategy        │  MRR  │ NDCG@{k:2d}  │  P@{k:2d}    │  R@{k:2d}    │\n"
        table += "├─────────────────┼───────┼──────────┼──────────┼──────────┤\n"

        # Rows
        for strategy in strategies:
            mrr = metrics.get("mrr", {}).get(strategy, 0.0) or 0.0
            ndcg = metrics.get("ndcg", {}).get(strategy, 0.0) or 0.0
            precision = metrics.get("precision", {}).get(strategy, 0.0) or 0.0
            recall = metrics.get("recall", {}).get(strategy, 0.0) or 0.0

            # Truncate or pad strategy name
            strategy_name = strategy[:15].ljust(15)

            table += f"│ {strategy_name} │ {mrr:5.3f} │  {ndcg:6.3f}  │  {precision:6.3f}  │  {recall:6.3f}  │\n"

        table += "└─────────────────┴───────┴──────────┴──────────┴──────────┘\n"

        # Best strategy
        best = comparison.get("best_strategy", "N/A")
        improvement = comparison.get("improvement_over_baseline", 0.0)

        table += f"\nBest: {best}"
        if improvement > 0:
            table += f" (+{improvement:.1f}% MRR vs baseline)"

        return table

    def get_statistics(self) -> Dict[str, int]:
        """
        Get statistics about loaded judgments.

        Returns:
            Dictionary with query and judgment counts
        """
        return {
            "num_queries": self.evaluator.get_query_count(),
            "num_judgments": self.evaluator.get_judgment_count()
        }
