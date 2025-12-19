# Test Queries Directory

This directory contains test queries and relevance judgments for offline evaluation of ranking strategies.

## Files

### `queries.jsonl`
Test queries in JSONL format (one query per line). Each query should have:
- `query_id`: Unique identifier for the query
- `query_text`: The query string
- `category`: Query category (factual, technical, temporal, ambiguous, multi_source)
- Optional metadata

Example:
```json
{"query_id": "factual_1", "query_text": "OAuth implementation", "category": "factual"}
{"query_id": "technical_1", "query_text": "OAuth implementation code", "category": "technical"}
```

Generate queries with:
```bash
python scripts/generate_test_queries.py --output data/test_queries/queries.jsonl --count 50
```

### `relevance_judgments.json`
Relevance judgments in JSON format. Each judgment should have:
- `query_id`: Links to a query in queries.jsonl
- `document_id`: Message ID from database
- `relevance`: Relevance score (0-3 scale)
  - 3: Highly relevant (perfect match)
  - 2: Relevant (good match)
  - 1: Partially relevant (somewhat related)
  - 0: Not relevant
- `query_text`: Optional query text for reference

Example:
```json
[
  {
    "query_id": "factual_1",
    "document_id": "1",
    "relevance": 3,
    "query_text": "OAuth implementation"
  },
  {
    "query_id": "factual_1",
    "document_id": "2",
    "relevance": 2,
    "query_text": "OAuth implementation"
  }
]
```

See `relevance_judgments.example.json` for a complete example.

## Usage

### 1. Generate Test Queries
```bash
python scripts/generate_test_queries.py \
  --output data/test_queries/queries.jsonl \
  --count 50
```

### 2. Create Relevance Judgments
Manually review the generated queries and create relevance judgments for each query. For each query:
1. Run the query against your system
2. Review the top 10-20 results
3. Assign relevance scores (0-3) to each result
4. Save in `relevance_judgments.json`

### 3. Run Evaluation
```bash
python scripts/evaluate_ranking.py \
  --queries data/test_queries/queries.jsonl \
  --judgments data/test_queries/relevance_judgments.json \
  --strategies semantic,bm25,hybrid_rrf,hybrid_weighted
```

## Query Categories

### Factual
Direct fact-finding queries with clear intent.
- Examples: "OAuth implementation", "Who is working on authentication"

### Technical
Queries seeking technical details or code.
- Examples: "OAuth implementation code", "API rate limiting implementation"

### Temporal
Time-sensitive queries about recent events.
- Examples: "Recent discussions about authentication", "Latest decisions on database"

### Ambiguous
Queries with unclear intent that could match multiple topics.
- Examples: "Login issues", "Performance problems"

### Multi-Source
Queries that should match content across multiple sources.
- Examples: "Authentication mentioned in Slack and GitHub"

## Best Practices

1. **Diverse Queries**: Include queries from all categories
2. **Sufficient Judgments**: Aim for at least 50 queries with 10+ judgments each
3. **Graded Relevance**: Use the full 0-3 scale, not just binary (0/1)
4. **Consistency**: Use consistent criteria when judging relevance
5. **Inter-Rater Agreement**: Have multiple people judge the same queries to ensure consistency

## Metrics Interpretation

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant result. Good for "find one good result" tasks.
- **NDCG@k**: Normalized Discounted Cumulative Gain. Considers graded relevance and position. Good for overall ranking quality.
- **Precision@k**: Fraction of relevant items in top k. Good for understanding recall.
- **Recall@k**: Fraction of all relevant items found in top k. Good for understanding coverage.

## Troubleshooting

### No judgments for query
- Ensure query_id in judgments matches query_id in queries.jsonl
- Check JSON syntax in relevance_judgments.json

### Low metric scores
- Verify your ranking strategy is working correctly
- Check that document_ids in judgments match actual message IDs in database
- Ensure relevance scores are assigned correctly

### Evaluation fails
- Verify both queries.jsonl and relevance_judgments.json exist and are valid JSON
- Check database connection and that messages exist
- Review logs for specific error messages
