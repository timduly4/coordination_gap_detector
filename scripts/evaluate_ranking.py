#!/usr/bin/env python3
"""
Offline evaluation script for ranking strategies.

This script evaluates and compares multiple ranking strategies using
test queries and relevance judgments.

Usage:
    # Evaluate all strategies
    python scripts/evaluate_ranking.py \
        --queries data/test_queries/queries.jsonl \
        --judgments data/test_queries/relevance_judgments.jsonl

    # Evaluate specific strategies
    python scripts/evaluate_ranking.py \
        --queries queries.jsonl \
        --judgments judgments.jsonl \
        --strategies semantic,bm25,hybrid_rrf

    # Export results to CSV
    python scripts/evaluate_ranking.py \
        --queries queries.jsonl \
        --judgments judgments.jsonl \
        --output results.csv \
        --format csv
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.db.postgres import async_engine, get_db
from src.db.vector_store import get_vector_store
from src.search.retrieval import KeywordRetriever
from src.services.evaluation_service import EvaluationService
from src.services.search_service import SearchService


def load_queries_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load test queries from JSONL file.

    Args:
        file_path: Path to queries file

    Returns:
        List of query dictionaries
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Queries file not found: {file_path}")

    queries = []
    with open(file_path, 'r') as f:
        for line in f:
            query = json.loads(line.strip())
            queries.append(query)

    print(f"✓ Loaded {len(queries)} test queries from {file_path}")
    return queries


def load_judgments_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load relevance judgments from JSON file.

    Args:
        file_path: Path to judgments file

    Returns:
        List of judgment dictionaries
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Judgments file not found: {file_path}")

    with open(file_path, 'r') as f:
        judgments = json.load(f)

    print(f"✓ Loaded {len(judgments)} relevance judgments from {file_path}")
    return judgments


def save_results_to_file(
    results: Any,
    output_path: Path,
    format: str = "json"
) -> None:
    """
    Save evaluation results to file.

    Args:
        results: Results to save
        output_path: Path to output file
        format: Output format (json, csv)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, 'w') as f:
            if isinstance(results, str):
                f.write(results)
            else:
                json.dump(results, f, indent=2)

    elif format == "csv":
        import csv
        with open(output_path, 'w', newline='') as f:
            if isinstance(results, list) and len(results) > 0:
                writer = csv.writer(f)
                writer.writerows(results)
            else:
                # Convert dict results to CSV
                writer = csv.writer(f)
                writer.writerow(["Strategy", "MRR", "NDCG@10", "P@10", "R@10"])
                # This is a simplified version
                for strategy, metrics in results.get("results", {}).get("mrr", {}).items():
                    row = [strategy]
                    for metric in ["mrr", "ndcg", "precision", "recall"]:
                        value = results["results"].get(metric, {}).get(strategy, 0.0)
                        row.append(f"{value:.4f}")
                    writer.writerow(row)

    print(f"✓ Saved results to {output_path}")


async def main(args):
    """Main script execution."""
    print("=" * 60)
    print("Ranking Strategy Evaluation")
    print("=" * 60)

    # Load queries
    queries_path = Path(args.queries)
    queries = load_queries_from_file(queries_path)

    # Load judgments
    judgments_path = Path(args.judgments)

    # Determine strategies to evaluate
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(',')]
    else:
        # Default strategies
        strategies = ["semantic", "bm25", "hybrid_rrf", "hybrid_weighted"]

    print(f"\nStrategies to evaluate: {', '.join(strategies)}")
    print(f"Number of test queries: {len(queries)}")
    print(f"K value: {args.k}")
    print()

    # Initialize services
    vector_store = get_vector_store()
    keyword_retriever = KeywordRetriever()
    search_service = SearchService(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever
    )
    evaluation_service = EvaluationService(search_service=search_service)

    # Load relevance judgments
    try:
        num_judgments = evaluation_service.load_judgments(str(judgments_path))
        print(f"✓ Loaded {num_judgments} relevance judgments")
    except FileNotFoundError:
        print(f"⚠ Warning: No judgments file found at {judgments_path}")
        print("  Continuing with evaluation (metrics may not be meaningful)")
    except Exception as e:
        print(f"⚠ Warning: Error loading judgments: {e}")
        print("  Continuing with evaluation")

    print()
    print("=" * 60)
    print("Running Evaluation...")
    print("=" * 60)
    print()

    # Get database session
    async with async_engine.begin() as conn:
        await conn.run_sync(lambda _: None)

    async for db in get_db():
        try:
            # Run comparison
            comparison = await evaluation_service.compare_strategies(
                strategies=strategies,
                test_queries=queries,
                db=db,
                k=args.k
            )

            # Display results
            print()
            print("=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print()

            # Format as table
            table = evaluation_service.export_results(comparison, output_format="table")
            print(table)

            # Export results if requested
            if args.output:
                output_path = Path(args.output)
                output_format = args.format

                results = evaluation_service.export_results(
                    comparison,
                    output_format=output_format
                )

                save_results_to_file(results, output_path, output_format)

            print()
            print("=" * 60)
            print("Evaluation Complete!")
            print("=" * 60)
            print()

            # Summary
            print("Summary:")
            print(f"  Best strategy: {comparison['best_strategy']}")
            print(f"  Improvement over baseline: +{comparison['improvement_over_baseline']:.1f}%")
            print(f"  Queries evaluated: {comparison['num_queries']}")
            print(f"  Strategies compared: {len(strategies)}")

            if args.output:
                print(f"  Results saved to: {args.output}")

        except Exception as e:
            print(f"\n❌ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        finally:
            await db.close()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate and compare ranking strategies"
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to test queries JSONL file"
    )
    parser.add_argument(
        "--judgments",
        type=str,
        required=True,
        help="Path to relevance judgments JSON file"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated list of strategies to evaluate (default: all)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Cutoff for @k metrics (default: 10)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (optional)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="json",
        choices=["json", "csv", "table"],
        help="Output format (default: json)"
    )

    args = parser.parse_args()

    asyncio.run(main(args))
