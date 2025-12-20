#!/bin/bash
set -e

echo "========================================="
echo "Milestone 2 Quick Start"
echo "========================================="
echo

# Check if services are running
echo "✓ Services Status:"
docker compose ps | grep -E "NAME|coordination-" | head -6
echo

# Test health
echo "✓ API Health:"
curl -s http://localhost:8000/health | jq .
echo

# Load mock data if needed
echo "📦 Loading mock data..."
if uv run python scripts/generate_mock_data.py --load 2>&1 | grep -q "error"; then
    echo "⚠️  Mock data may already be loaded or script needs attention"
else
    echo "✓ Mock data loaded"
fi
echo

# Test basic search
echo "🔍 Testing Basic Search (BM25):"
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth", "limit": 3, "ranking_strategy": "bm25"}' | jq '{total: .total, results: .results | length, strategy: .ranking_strategy}'
echo

# Test semantic search
echo "🔍 Testing Semantic Search:"
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth implementation", "limit": 3, "ranking_strategy": "semantic"}' | jq '{total: .total, results: .results | length, strategy: .ranking_strategy}'
echo

# Test hybrid search
echo "🔍 Testing Hybrid Search (RRF):"
curl -s -X POST http://localhost:8000/api/v1/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "OAuth security", "limit": 3, "ranking_strategy": "hybrid_rrf"}' | jq '{total: .total, results: .results | length, strategy: .ranking_strategy}'
echo

echo "========================================="
echo "✅ Milestone 2 is running!"
echo "========================================="
echo
echo "Next steps:"
echo "1. Open API docs: http://localhost:8000/docs"
echo "2. Read: EXPLORE_MILESTONE_2.md"
echo "3. Try different ranking strategies"
echo "4. Run tests: uv run pytest tests/test_ranking/ -v"
echo

