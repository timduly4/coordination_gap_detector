"""
Comprehensive tests for the evaluation framework.

Tests cover:
- Evaluation service functionality
- Strategy comparison
- Test query generation
- API endpoints
- End-to-end evaluation workflow
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import create_app
from src.models.schemas import SearchResponse, SearchResultItem
from src.ranking.evaluation import RankingEvaluator, RelevanceJudgment
from src.services.evaluation_service import EvaluationService
from src.services.search_service import SearchService


class TestEvaluationService:
    """Test suite for EvaluationService."""

    @pytest.fixture
    def mock_search_service(self):
        """Create a mock search service."""
        service = AsyncMock(spec=SearchService)
        return service

    @pytest.fixture
    def evaluation_service(self, mock_search_service):
        """Create evaluation service with mock search."""
        return EvaluationService(search_service=mock_search_service)

    @pytest.fixture
    def sample_test_queries(self):
        """Sample test queries."""
        return [
            {"query_id": "q1", "query_text": "OAuth implementation"},
            {"query_id": "q2", "query_text": "authentication flow"},
            {"query_id": "q3", "query_text": "database migration"}
        ]

    @pytest.fixture
    def sample_judgments(self):
        """Sample relevance judgments."""
        return [
            RelevanceJudgment("q1", "1", 3, "OAuth implementation"),
            RelevanceJudgment("q1", "2", 2, "OAuth implementation"),
            RelevanceJudgment("q1", "3", 1, "OAuth implementation"),
            RelevanceJudgment("q2", "1", 3, "authentication flow"),
            RelevanceJudgment("q2", "4", 2, "authentication flow"),
        ]

    def test_service_initialization(self, evaluation_service):
        """Test service initializes correctly."""
        assert evaluation_service.evaluator is not None
        assert isinstance(evaluation_service.evaluator, RankingEvaluator)

    def test_add_judgment(self, evaluation_service):
        """Test adding a single judgment."""
        evaluation_service.add_judgment(
            query_id="q1",
            document_id="doc1",
            relevance=3,
            query_text="test query"
        )

        stats = evaluation_service.get_statistics()
        assert stats["num_queries"] == 1
        assert stats["num_judgments"] == 1

    def test_get_statistics_empty(self, evaluation_service):
        """Test statistics with no judgments."""
        stats = evaluation_service.get_statistics()
        assert stats["num_queries"] == 0
        assert stats["num_judgments"] == 0

    def test_get_statistics_with_judgments(self, evaluation_service, sample_judgments):
        """Test statistics with loaded judgments."""
        evaluation_service.evaluator.add_judgments(sample_judgments)

        stats = evaluation_service.get_statistics()
        assert stats["num_queries"] == 2  # q1 and q2
        assert stats["num_judgments"] == 5

    @pytest.mark.asyncio
    async def test_evaluate_strategy_basic(
        self,
        evaluation_service,
        mock_search_service,
        sample_test_queries,
        sample_judgments
    ):
        """Test evaluating a single strategy."""
        # Setup mock search responses
        def create_mock_response(query_text: str) -> SearchResponse:
            """Create mock search response."""
            return SearchResponse(
                results=[
                    SearchResultItem(
                        content=f"Result for {query_text}",
                        source="slack",
                        channel="#test",
                        author="test@test.com",
                        timestamp=datetime(2024, 1, 1, 0, 0, 0),
                        score=0.9,
                        message_id=1,
                        external_id="msg_1"
                    )
                ],
                total=1,
                query=query_text,
                query_time_ms=10,
                threshold=0.0,
                ranking_strategy="semantic"
            )

        mock_search_service.search.return_value = create_mock_response("test")

        # Add judgments
        evaluation_service.evaluator.add_judgments(sample_judgments)

        # Mock database session
        mock_db = AsyncMock()

        # Evaluate strategy
        result = await evaluation_service.evaluate_strategy(
            strategy="semantic",
            test_queries=sample_test_queries,
            db=mock_db,
            k=10
        )

        # Verify results structure
        assert result["strategy"] == "semantic"
        assert result["num_queries"] == len(sample_test_queries)
        assert "aggregate_metrics" in result
        assert "per_query_results" in result

        # Verify metrics exist
        metrics = result["aggregate_metrics"]
        assert "mrr" in metrics
        assert "ndcg" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    @pytest.mark.asyncio
    async def test_compare_strategies(
        self,
        evaluation_service,
        mock_search_service,
        sample_test_queries,
        sample_judgments
    ):
        """Test comparing multiple strategies."""
        # Setup mock
        mock_response = SearchResponse(
            results=[],
            total=0,
            query="test",
            query_time_ms=10,
            threshold=0.0,
            ranking_strategy="semantic"
        )
        mock_search_service.search.return_value = mock_response

        # Add judgments
        evaluation_service.evaluator.add_judgments(sample_judgments)

        # Mock database
        mock_db = AsyncMock()

        # Compare strategies
        strategies = ["semantic", "bm25", "hybrid_rrf"]
        comparison = await evaluation_service.compare_strategies(
            strategies=strategies,
            test_queries=sample_test_queries[:1],  # Use fewer queries for speed
            db=mock_db,
            k=10
        )

        # Verify comparison structure
        assert comparison["strategies"] == strategies
        assert comparison["num_queries"] == 1
        assert comparison["k"] == 10
        assert "results" in comparison
        assert "best_strategy" in comparison
        assert "improvement_over_baseline" in comparison

        # Verify results for each metric
        assert "mrr" in comparison["results"]
        assert "ndcg" in comparison["results"]
        assert "precision" in comparison["results"]
        assert "recall" in comparison["results"]

        # Each strategy should have a score
        for metric_name, metric_values in comparison["results"].items():
            assert len(metric_values) == len(strategies)

    def test_export_results_dict(self, evaluation_service):
        """Test exporting results as dictionary."""
        comparison = {
            "strategies": ["semantic", "bm25"],
            "results": {
                "mrr": {"semantic": 0.75, "bm25": 0.65},
                "ndcg": {"semantic": 0.80, "bm25": 0.70}
            },
            "best_strategy": "semantic",
            "improvement_over_baseline": 15.4
        }

        result = evaluation_service.export_results(comparison, output_format="dict")
        assert result == comparison

    def test_export_results_json(self, evaluation_service):
        """Test exporting results as JSON."""
        comparison = {
            "strategies": ["semantic"],
            "results": {"mrr": {"semantic": 0.75}},
            "best_strategy": "semantic",
            "improvement_over_baseline": 0.0
        }

        result = evaluation_service.export_results(comparison, output_format="json")
        assert isinstance(result, str)

        # Parse JSON to verify it's valid
        parsed = json.loads(result)
        assert parsed["best_strategy"] == "semantic"

    def test_export_results_csv(self, evaluation_service):
        """Test exporting results as CSV."""
        comparison = {
            "strategies": ["semantic", "bm25"],
            "results": {
                "mrr": {"semantic": 0.75, "bm25": 0.65},
                "ndcg": {"semantic": 0.80, "bm25": 0.70}
            },
            "best_strategy": "semantic",
            "improvement_over_baseline": 15.4
        }

        result = evaluation_service.export_results(comparison, output_format="csv")
        assert isinstance(result, list)
        assert len(result) == 3  # Header + 2 strategies

        # Check header
        assert result[0][0] == "Strategy"

        # Check data rows
        assert result[1][0] == "semantic"
        assert result[2][0] == "bm25"

    def test_export_results_table(self, evaluation_service):
        """Test exporting results as table."""
        comparison = {
            "strategies": ["semantic", "bm25"],
            "k": 10,
            "results": {
                "mrr": {"semantic": 0.75, "bm25": 0.65},
                "ndcg": {"semantic": 0.80, "bm25": 0.70},
                "precision": {"semantic": 0.70, "bm25": 0.60},
                "recall": {"semantic": 0.65, "bm25": 0.55}
            },
            "best_strategy": "semantic",
            "improvement_over_baseline": 15.4
        }

        result = evaluation_service.export_results(comparison, output_format="table")
        assert isinstance(result, str)
        assert "semantic" in result
        assert "bm25" in result
        assert "Best: semantic" in result

    def test_export_results_invalid_format(self, evaluation_service):
        """Test exporting with invalid format raises error."""
        comparison = {"strategies": []}

        with pytest.raises(ValueError, match="Unsupported output format"):
            evaluation_service.export_results(comparison, output_format="invalid")


class TestEvaluationAPI:
    """Test suite for evaluation API endpoints."""

    @pytest.fixture
    def mock_service(self):
        """Create a mock evaluation service."""
        return MagicMock(spec=EvaluationService)

    @pytest.fixture
    def client(self, mock_service):
        """Create test client with overridden dependency."""
        from src.api.routes.evaluation import get_evaluation_service

        app = create_app()

        # Override the dependency
        app.dependency_overrides[get_evaluation_service] = lambda: mock_service

        client = TestClient(app)
        yield client

        # Clean up
        app.dependency_overrides.clear()

    def test_add_judgment_endpoint(self, client, mock_service):
        """Test adding a judgment via API."""
        response = client.post(
            "/api/v1/evaluation/judgments",
            json={
                "query_id": "q1",
                "document_id": "doc1",
                "relevance": 3,
                "query_text": "test query"
            }
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"

        # Verify service was called
        mock_service.add_judgment.assert_called_once_with(
            query_id="q1",
            document_id="doc1",
            relevance=3,
            query_text="test query"
        )

    def test_add_judgment_invalid_relevance(self, client, mock_service):
        """Test adding judgment with invalid relevance."""
        # Pydantic validation catches this before reaching the service
        response = client.post(
            "/api/v1/evaluation/judgments",
            json={
                "query_id": "q1",
                "document_id": "doc1",
                "relevance": 5,  # Invalid - will be caught by Pydantic validation
                "query_text": "test"
            }
        )

        # Pydantic validation returns 422 for invalid input
        assert response.status_code == 422

        # Service should not be called
        mock_service.add_judgment.assert_not_called()

    def test_get_statistics_endpoint(self, client, mock_service):
        """Test getting statistics via API."""
        mock_service.get_statistics.return_value = {
            "num_queries": 50,
            "num_judgments": 412
        }

        response = client.get("/api/v1/evaluation/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["num_queries"] == 50
        assert data["num_judgments"] == 412

        # Verify service was called
        mock_service.get_statistics.assert_called_once()


class TestIntegrationWorkflow:
    """Integration tests for complete evaluation workflow."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for test files."""
        return tmp_path

    def test_end_to_end_evaluation_workflow(self, temp_dir):
        """Test complete evaluation workflow from queries to results."""
        # 1. Create test queries file
        queries_file = temp_dir / "queries.jsonl"
        queries = [
            {"query_id": "q1", "query_text": "OAuth implementation"},
            {"query_id": "q2", "query_text": "authentication flow"}
        ]

        with open(queries_file, 'w') as f:
            for query in queries:
                f.write(json.dumps(query) + '\n')

        # 2. Create judgments file
        judgments_file = temp_dir / "judgments.json"
        judgments = [
            {
                "query_id": "q1",
                "document_id": "1",
                "relevance": 3,
                "query_text": "OAuth implementation"
            },
            {
                "query_id": "q1",
                "document_id": "2",
                "relevance": 2,
                "query_text": "OAuth implementation"
            }
        ]

        with open(judgments_file, 'w') as f:
            json.dump(judgments, f)

        # 3. Load and verify
        with open(queries_file, 'r') as f:
            loaded_queries = [json.loads(line) for line in f]

        assert len(loaded_queries) == 2
        assert loaded_queries[0]["query_id"] == "q1"

        with open(judgments_file, 'r') as f:
            loaded_judgments = json.load(f)

        assert len(loaded_judgments) == 2
        assert loaded_judgments[0]["relevance"] == 3

        # 4. Create evaluator and load judgments
        evaluator = RankingEvaluator()
        evaluator.load_judgments_from_file(judgments_file)

        stats = {
            "num_queries": evaluator.get_query_count(),
            "num_judgments": evaluator.get_judgment_count()
        }

        assert stats["num_queries"] == 1  # Only q1 has judgments
        assert stats["num_judgments"] == 2
