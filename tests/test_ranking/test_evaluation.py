"""
Unit tests for ranking evaluation framework.
"""

from pathlib import Path

import pytest

from src.ranking.evaluation import (
    EvaluationResult,
    QueryResults,
    RankingEvaluator,
    RelevanceJudgment,
    load_labeled_queries_from_file,
)
from src.ranking.metrics import RankingMetrics


class TestRelevanceJudgment:
    """Test suite for RelevanceJudgment dataclass."""

    def test_relevance_judgment_creation(self):
        """Test creating a relevance judgment."""
        judgment = RelevanceJudgment(
            query_id="q1",
            document_id="doc1",
            relevance=3,
            query_text="test query"
        )

        assert judgment.query_id == "q1"
        assert judgment.document_id == "doc1"
        assert judgment.relevance == 3
        assert judgment.query_text == "test query"

    def test_relevance_judgment_validation(self):
        """Test that relevance score is validated."""
        # Valid relevance scores
        for rel in [0, 1, 2, 3]:
            judgment = RelevanceJudgment("q1", "doc1", rel)
            assert judgment.relevance == rel

        # Invalid relevance scores
        with pytest.raises(ValueError):
            RelevanceJudgment("q1", "doc1", 4)

        with pytest.raises(ValueError):
            RelevanceJudgment("q1", "doc1", -1)

    def test_relevance_judgment_default_metadata(self):
        """Test default metadata is empty dict."""
        judgment = RelevanceJudgment("q1", "doc1", 2)
        assert judgment.metadata == {}


class TestQueryResults:
    """Test suite for QueryResults dataclass."""

    def test_query_results_creation(self):
        """Test creating query results."""
        results = QueryResults(
            query_id="q1",
            query_text="test query",
            results=[("doc1", 0.9), ("doc2", 0.8)]
        )

        assert results.query_id == "q1"
        assert results.query_text == "test query"
        assert len(results.results) == 2
        assert results.results[0] == ("doc1", 0.9)


class TestRankingEvaluator:
    """Test suite for RankingEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a fresh evaluator."""
        return RankingEvaluator()

    @pytest.fixture
    def sample_judgments(self):
        """Sample relevance judgments."""
        return [
            RelevanceJudgment("q1", "doc1", 3, "query 1"),
            RelevanceJudgment("q1", "doc2", 2, "query 1"),
            RelevanceJudgment("q1", "doc3", 1, "query 1"),
            RelevanceJudgment("q1", "doc4", 0, "query 1"),
            RelevanceJudgment("q2", "doc5", 3, "query 2"),
            RelevanceJudgment("q2", "doc6", 1, "query 2"),
        ]

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.get_query_count() == 0
        assert evaluator.get_judgment_count() == 0

    def test_add_single_judgment(self, evaluator):
        """Test adding a single judgment."""
        evaluator.add_judgment("q1", "doc1", 3, "test query")

        assert evaluator.get_query_count() == 1
        assert evaluator.get_judgment_count() == 1
        assert evaluator.has_judgments_for_query("q1")

    def test_add_multiple_judgments(self, evaluator, sample_judgments):
        """Test adding multiple judgments."""
        evaluator.add_judgments(sample_judgments)

        assert evaluator.get_query_count() == 2
        assert evaluator.get_judgment_count() == 6
        assert evaluator.has_judgments_for_query("q1")
        assert evaluator.has_judgments_for_query("q2")

    def test_get_relevance_scores(self, evaluator, sample_judgments):
        """Test getting relevance scores for documents."""
        evaluator.add_judgments(sample_judgments)

        scores = evaluator.get_relevance_scores("q1", ["doc1", "doc2", "doc3", "doc4"])

        assert scores == [3, 2, 1, 0]

    def test_get_relevance_scores_unknown_query(self, evaluator):
        """Test getting relevance scores for unknown query."""
        scores = evaluator.get_relevance_scores("unknown", ["doc1", "doc2"])

        # Should return zeros for unknown query
        assert scores == [0, 0]

    def test_get_relevance_scores_unknown_docs(self, evaluator, sample_judgments):
        """Test getting relevance scores for unknown documents."""
        evaluator.add_judgments(sample_judgments)

        scores = evaluator.get_relevance_scores("q1", ["doc1", "unknown1", "doc2", "unknown2"])

        # Unknown docs should get 0
        assert scores == [3, 0, 2, 0]

    def test_evaluate_query(self, evaluator, sample_judgments):
        """Test evaluating a single query."""
        evaluator.add_judgments(sample_judgments)

        # Perfect ranking
        result = evaluator.evaluate_query(
            query_id="q1",
            document_ids=["doc1", "doc2", "doc3", "doc4"],
            k=4
        )

        assert result.query_id == "q1"
        assert result.metrics.ndcg == 1.0  # Perfect ranking
        assert result.num_results == 4
        assert result.num_relevant == 3  # doc1, doc2, doc3 are relevant

    def test_evaluate_query_imperfect_ranking(self, evaluator, sample_judgments):
        """Test evaluating query with imperfect ranking."""
        evaluator.add_judgments(sample_judgments)

        # Reversed ranking
        result = evaluator.evaluate_query(
            query_id="q1",
            document_ids=["doc4", "doc3", "doc2", "doc1"],
            k=4
        )

        assert result.metrics.ndcg < 0.7  # Poor ranking
        assert result.metrics.ndcg > 0.0

    def test_evaluate_query_with_k_cutoff(self, evaluator, sample_judgments):
        """Test evaluating query with k cutoff."""
        evaluator.add_judgments(sample_judgments)

        result = evaluator.evaluate_query(
            query_id="q1",
            document_ids=["doc1", "doc2", "doc3", "doc4"],
            k=2
        )

        assert result.details["k"] == 2
        assert len(result.details["relevance_scores"]) == 2

    def test_evaluate_multiple_queries(self, evaluator, sample_judgments):
        """Test evaluating multiple queries."""
        evaluator.add_judgments(sample_judgments)

        query_results = [
            QueryResults("q1", "query 1", [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]),
            QueryResults("q2", "query 2", [("doc5", 0.95), ("doc6", 0.85)]),
        ]

        results = evaluator.evaluate_queries(query_results, k=3)

        assert len(results) == 2
        assert "q1" in results
        assert "q2" in results
        assert isinstance(results["q1"], EvaluationResult)

    def test_calculate_aggregate_metrics(self, evaluator, sample_judgments):
        """Test calculating aggregate metrics."""
        evaluator.add_judgments(sample_judgments)

        # Create some evaluations
        eval1 = evaluator.evaluate_query("q1", ["doc1", "doc2", "doc3"], k=3)
        eval2 = evaluator.evaluate_query("q2", ["doc5", "doc6"], k=2)

        evaluations = {"q1": eval1, "q2": eval2}

        aggregate = evaluator.calculate_aggregate_metrics(evaluations)

        assert aggregate.query_id == "aggregate"
        assert aggregate.metrics.ndcg is not None
        assert aggregate.details["num_queries"] == 2

    def test_calculate_aggregate_metrics_empty(self, evaluator):
        """Test aggregate metrics with empty evaluations."""
        aggregate = evaluator.calculate_aggregate_metrics({})

        assert aggregate.query_id == "aggregate"
        assert aggregate.num_results == 0
        assert aggregate.num_relevant == 0

    def test_load_judgments_from_file(self, evaluator, tmp_path):
        """Test loading judgments from JSON file."""
        import json

        # Create temporary JSON file
        judgments_data = [
            {
                "query_id": "q1",
                "document_id": "doc1",
                "relevance": 3,
                "query_text": "test query"
            },
            {
                "query_id": "q1",
                "document_id": "doc2",
                "relevance": 1
            }
        ]

        json_file = tmp_path / "test_judgments.json"
        with open(json_file, 'w') as f:
            json.dump(judgments_data, f)

        evaluator.load_judgments_from_file(json_file)

        assert evaluator.get_query_count() == 1
        assert evaluator.get_judgment_count() == 2

    def test_load_judgments_file_not_found(self, evaluator):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            evaluator.load_judgments_from_file("nonexistent.json")


class TestEvaluationResult:
    """Test suite for EvaluationResult dataclass."""

    def test_evaluation_result_creation(self):
        """Test creating an evaluation result."""
        metrics = RankingMetrics(mrr=0.75, ndcg=0.85)
        result = EvaluationResult(
            query_id="q1",
            metrics=metrics,
            num_results=10,
            num_relevant=5
        )

        assert result.query_id == "q1"
        assert result.metrics.mrr == 0.75
        assert result.num_results == 10
        assert result.num_relevant == 5

    def test_evaluation_result_to_dict(self):
        """Test converting evaluation result to dictionary."""
        metrics = RankingMetrics(mrr=0.75, ndcg=0.85)
        result = EvaluationResult("q1", metrics, 10, 5)

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["query_id"] == "q1"
        assert "metrics" in result_dict
        assert result_dict["num_results"] == 10


class TestLoadLabeledQueries:
    """Test suite for load_labeled_queries_from_file function."""

    def test_load_labeled_queries(self):
        """Test loading labeled queries from fixture file."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "labeled_queries.json"

        if not fixture_path.exists():
            pytest.skip("Labeled queries fixture not found")

        judgments, evaluator = load_labeled_queries_from_file(fixture_path)

        assert len(judgments) > 0
        assert evaluator.get_query_count() > 0
        assert evaluator.get_judgment_count() == len(judgments)

    def test_load_labeled_queries_file_structure(self):
        """Test that loaded queries have correct structure."""
        fixture_path = Path(__file__).parent.parent / "fixtures" / "labeled_queries.json"

        if not fixture_path.exists():
            pytest.skip("Labeled queries fixture not found")

        judgments, _ = load_labeled_queries_from_file(fixture_path)

        for judgment in judgments:
            assert isinstance(judgment, RelevanceJudgment)
            assert isinstance(judgment.query_id, str)
            assert isinstance(judgment.document_id, str)
            assert 0 <= judgment.relevance <= 3


class TestEndToEndEvaluation:
    """End-to-end integration tests for evaluation."""

    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create evaluator
        evaluator = RankingEvaluator()

        # Add judgments
        evaluator.add_judgment("q1", "doc1", 3, "OAuth guide")
        evaluator.add_judgment("q1", "doc2", 2, "OAuth guide")
        evaluator.add_judgment("q1", "doc3", 1, "OAuth guide")
        evaluator.add_judgment("q1", "doc4", 0, "OAuth guide")

        # Create search results (imperfect ranking)
        query_results = QueryResults(
            query_id="q1",
            query_text="OAuth guide",
            results=[("doc2", 0.9), ("doc1", 0.85), ("doc4", 0.7), ("doc3", 0.6)]
        )

        # Evaluate
        result = evaluator.evaluate_query(
            query_id=query_results.query_id,
            document_ids=[doc_id for doc_id, _ in query_results.results],
            k=4
        )

        # Verify results
        assert result.query_id == "q1"
        assert 0.0 < result.metrics.ndcg < 1.0  # Imperfect ranking
        assert result.metrics.precision is not None
        assert result.metrics.recall is not None
        assert result.num_results == 4

    def test_batch_evaluation(self):
        """Test evaluating multiple queries in batch."""
        evaluator = RankingEvaluator()

        # Add judgments for multiple queries
        for q_id in ["q1", "q2", "q3"]:
            for i in range(1, 4):
                doc_id = f"{q_id}_doc{i}"
                evaluator.add_judgment(q_id, doc_id, 3 - i + 1)

        # Create query results
        query_results = [
            QueryResults(f"q{i}", f"query {i}", [(f"q{i}_doc{j}", 1.0 - j*0.1) for j in range(1, 4)])
            for i in range(1, 4)
        ]

        # Evaluate all queries
        results = evaluator.evaluate_queries(query_results, k=3)

        assert len(results) == 3

        # Calculate aggregate
        aggregate = evaluator.calculate_aggregate_metrics(results)

        assert aggregate.query_id == "aggregate"
        assert aggregate.metrics.ndcg is not None
        assert aggregate.details["num_queries"] == 3
